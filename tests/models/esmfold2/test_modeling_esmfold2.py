# Copyright 2026 Biohub and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing suite for the PyTorch ESMFold2 model.

ESMFold2 is an all-atom structure predictor: its forward takes ~18 structural
feature tensors (built from a sequence by ``prepare_protein_features``) rather
than the standard ``input_ids``/``attention_mask``, so it does not plug into
``ModelTesterMixin`` (the file is registered in
``utils/check_repo.py::TEST_FILES_WITH_NO_COMMON_TESTS``). Coverage here is the
config (round-trip / nesting), a CPU forward smoke test across attention
backends, weight save/load, and a slow real-weight integration test.
"""

import tempfile
import unittest

from transformers import ESMFold2Config, is_torch_available
from transformers.testing_utils import (
    TestCasePlus,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester


if is_torch_available():
    import torch

    from transformers import ESMFold2Model
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2SWA3DRoPEAttention

# TEMP: the public ``biohub/ESMFold2`` snapshot does not yet bundle the ESMC-6B
# backbone under ``esmc.*`` (it loads random → garbage outputs). Point the slow
# integration tests at the locally-bundled checkpoint for now. REVERT to
# "biohub/ESMFold2" once the backbone is bundled there.
_INTEGRATION_CKPT = "Rocketknight1/ESMFold2-merged-temp"


def get_tiny_config(**overrides) -> "ESMFold2Config":
    """A minimal but internally consistent ESMFold2 config for CPU testing.

    Constraints (see modeling): 3D RoPE needs ``3*n_spatial + n_uid <= head_dim//2``
    (head_dim = d_atom/n_heads = c_atom/atom_num_heads = 8 here), and
    ``inputs.d_inputs == 67 + d_token//2 == structure_head.diffusion_module.c_s_inputs``
    (the feature-concat width; 83 = 67 + 32//2).
    """
    kwargs = {
        "d_single": 32,
        "d_pair": 16,
        "num_loops": 1,
        "num_diffusion_samples": 1,
        "esmc_config": {"d_model": 32, "n_heads": 2, "n_layers": 1, "vocab_size": 64},
        "inputs": {
            "d_inputs": 83,
            "atom_encoder": {
                "d_atom": 16,
                "d_token": 32,
                "n_blocks": 1,
                "n_heads": 2,
                "swa_window_size": 8,
                "n_spatial_rope_pairs_per_axis": 1,
                "n_uid_rope_pairs": 1,
            },
        },
        "folding_trunk": {"n_layers": 1},
        "structure_head": {
            "diffusion_module": {
                "c_atom": 16,
                "c_token": 32,
                "c_z": 16,
                "c_s_inputs": 83,
                "atom_num_blocks": 1,
                "atom_num_heads": 2,
                "token_num_blocks": 1,
                "token_num_heads": 2,
            },
            "distogram_bins": 8,
        },
        "confidence_head": {
            "num_plddt_bins": 4,
            "num_pde_bins": 4,
            "num_pae_bins": 4,
            "distogram_bins": 8,
            "folding_trunk": {"n_layers": 1},
        },
        "parcae": {"coda_n_layers": 1},
        "lm_encoder": {"n_layers": 1},
    }
    kwargs.update(overrides)
    return ESMFold2Config(**kwargs)


class ESMFold2ConfigTester(ConfigTester):
    def create_and_test_config_from_and_save_pretrained_composite(self):
        # ESMFold2's sub-configs are internal architecture details (no model_type,
        # not in the auto mappings), so they can't be reloaded standalone from the
        # parent config dir — skip this check. (A no-op rather than SkipTest so the
        # remaining run_common_tests checks still run.)
        pass


@require_torch
class ESMFold2ConfigTest(unittest.TestCase):
    def setUp(self):
        # ESMFold2Config is composite (sub_configs) with no vocab/hidden_size.
        self.config_tester = ESMFold2ConfigTester(
            self, config_class=ESMFold2Config, has_text_modality=False, num_loops=5
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_type_validation(self):
        # Only the "release" variant is supported in this port.
        ESMFold2Config(type="release")
        with self.assertRaises(ValueError):
            ESMFold2Config(type="experimental")

    def test_nested_config_round_trip(self):
        config = ESMFold2Config(d_pair=72, inputs={"d_inputs": 99, "atom_encoder": {"d_atom": 64, "n_heads": 8}})
        with tempfile.TemporaryDirectory() as tmp:
            config.save_pretrained(tmp)
            reloaded = ESMFold2Config.from_pretrained(tmp)

        self.assertEqual(reloaded.to_dict(), config.to_dict())
        # Sub-configs round-trip as the right (PreTrainedConfig) types, not dicts.
        self.assertEqual(type(reloaded.inputs).__name__, "InputsEmbedderConfig")
        self.assertEqual(type(reloaded.inputs.atom_encoder).__name__, "AtomAttentionConfig")
        self.assertEqual(reloaded.inputs.d_inputs, 99)
        self.assertEqual(reloaded.inputs.atom_encoder.d_atom, 64)

    def test_attn_implementation_propagates_to_subconfigs(self):
        config = ESMFold2Config(attn_implementation="sdpa")
        self.assertEqual(config._attn_implementation, "sdpa")
        self.assertEqual(config.inputs._attn_implementation, "sdpa")


@require_torch
class ESMFold2ModelTest(unittest.TestCase):
    seq = "MKLVAAG"

    # These are pure-PyTorch correctness smoke tests, run on CPU for portability
    # (the diffusion sampler is tiny here); GPU is covered by the slow integration
    # test below.
    def _build(self, attn_implementation="sdpa"):
        torch.manual_seed(0)
        config = get_tiny_config(attn_implementation=attn_implementation)
        return ESMFold2Model(config).eval()

    def test_forward_runs_on_both_backends(self):
        # The ESMC backbone is a bundled (tiny, randomly-initialised) submodule, so this
        # exercises the full pure-PyTorch stack on CPU end-to-end: backbone + trunk +
        # diffusion + confidence head.
        for impl in ("sdpa", "eager"):
            with self.subTest(attn_implementation=impl):
                model = self._build(impl)
                self.assertIsInstance(model.esmc, torch.nn.Module)
                with torch.no_grad():
                    out = model.infer_protein(self.seq, num_loops=1, num_diffusion_samples=1, num_sampling_steps=2)
                coords = out["sample_atom_coords"]
                self.assertEqual(coords.shape[0], 1)  # num_diffusion_samples
                self.assertEqual(coords.shape[-1], 3)  # xyz
                self.assertTrue(torch.isfinite(coords).all())
                self.assertEqual(out["distogram_logits"].shape[-1], model.config.structure_head.distogram_bins)

    def test_attention_dispatch_attached(self):
        model = self._build("eager")
        swa_modules = [m for m in model.modules() if isinstance(m, ESMFold2SWA3DRoPEAttention)]
        # Both atom sites (inputs embedder + diffusion decoder) contribute SWA modules.
        self.assertGreaterEqual(len(swa_modules), 1)
        self.assertTrue(all(m.config is model.config for m in swa_modules))
        self.assertTrue(all(m.config._attn_implementation == "eager" for m in swa_modules))

    def test_save_load(self):
        # The forward is intentionally stochastic (parcae diffusion-loop scheduler),
        # so save/load fidelity is checked at the weight level, then the reloaded
        # model is run to confirm it is usable.
        model = self._build()
        state_before = model.state_dict()

        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            # The (tiny) ESMC backbone is bundled in the saved checkpoint and reloaded
            # like any other submodule — no separate backbone load.
            reloaded = ESMFold2Model.from_pretrained(tmp).eval()

        state_after = reloaded.state_dict()
        self.assertEqual(set(state_before), set(state_after))
        for key, tensor in state_before.items():
            torch.testing.assert_close(state_after[key], tensor, rtol=0, atol=0)

        with torch.no_grad():
            out = reloaded.infer_protein(self.seq, num_loops=1, num_diffusion_samples=1, num_sampling_steps=1)
        self.assertTrue(torch.isfinite(out["sample_atom_coords"]).all())


@require_torch
class ESMFold2IntegrationTest(TestCasePlus):
    @slow
    @require_torch_accelerator
    def test_inference_protein_folding(self):
        # bf16 is the intended inference regime; the ESMC backbone is bundled in the
        # checkpoint and loaded with the model.
        model = ESMFold2Model.from_pretrained(_INTEGRATION_CKPT, dtype=torch.bfloat16).to(torch_device).eval()

        # Ubiquitin (PDB 1UBQ), a textbook well-folding 76-residue domain. These
        # diffusion folders draw several samples and the best-ranked is the
        # prediction, so assert on the best of N.
        seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
        torch.manual_seed(0)
        with torch.no_grad():
            output = model.infer_protein(seq, num_diffusion_samples=8, num_sampling_steps=68)

        coords = output["sample_atom_coords"]
        self.assertEqual(coords.shape[-1], 3)
        self.assertTrue(torch.isfinite(coords).all())

        # pLDDT and pTM are on a 0-1 scale in this model; ESMFold2 folds ubiquitin
        # confidently (CPU-fp32 reference: best pLDDT ~0.80, best pTM ~0.74).
        plddt = output["plddt"].float()  # [num_samples, n_res]
        ptm = output["ptm"].float()  # [num_samples]
        best_plddt = plddt.mean(dim=1).max().item()
        best_ptm = ptm.max().item()
        self.assertGreater(best_plddt, 0.7)
        self.assertGreater(best_ptm, 0.6)

    @slow
    def test_inference_deterministic_cpu_fp32(self):
        model = ESMFold2Model.from_pretrained(_INTEGRATION_CKPT, dtype=torch.float32).eval()

        seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
        torch.manual_seed(0)
        with torch.no_grad():
            output = model.infer_protein(seq, num_loops=4, num_diffusion_samples=2, num_sampling_steps=32)

        expected_distogram = torch.tensor([6.5849, 7.9825, 9.6068, 9.6403, 16.5200, 18.9912, 19.9698, 23.0489])
        torch.testing.assert_close(
            output["distogram_logits"][0, 0, 1, :8].float(), expected_distogram, rtol=1e-3, atol=1e-3
        )
        self.assertAlmostEqual(output["ptm"].max().item(), 0.7427, delta=1e-2)

    @slow
    @require_torch_accelerator
    def test_inference_deterministic_bf16(self):
        prev = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
            torch.backends.cuda.matmul.allow_tf32,
        )
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False

            model = ESMFold2Model.from_pretrained(_INTEGRATION_CKPT, dtype=torch.bfloat16).to(torch_device).eval()
            seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
            torch.manual_seed(0)
            with torch.no_grad():
                output = model.infer_protein(seq, num_loops=4, num_diffusion_samples=2, num_sampling_steps=32)

            expected_distogram = torch.tensor([6.47, 7.72, 9.44, 9.31, 16.12, 18.62, 19.62, 22.62])
            torch.testing.assert_close(
                output["distogram_logits"][0, 0, 1, :8].float().cpu(), expected_distogram, rtol=0, atol=0.2
            )
            self.assertAlmostEqual(output["ptm"].max().item(), 0.742, delta=0.05)
        finally:
            torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])
            torch.backends.cudnn.deterministic = prev[2]
            torch.backends.cudnn.benchmark = prev[3]
            torch.backends.cuda.matmul.allow_tf32 = prev[4]
