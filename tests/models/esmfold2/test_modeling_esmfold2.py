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
feature tensors (built from a sequence by ``prepare_protein_features``) and
returns a plain ``dict`` rather than a ``ModelOutput``, so it does not plug into
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
    from transformers.models.esmfold2.modeling_esmfold2_common import SWA3DRoPEAttention


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
        "lm_d_model": 32,
        "lm_num_layers": 1,
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
        "folding_trunk": {"n_layers": 1, "n_heads": 2},
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
            "folding_trunk": {"n_layers": 1, "n_heads": 2},
        },
        "parcae": {"coda_n_layers": 1},
        "lm_encoder": {"n_layers": 1},
    }
    kwargs.update(overrides)
    return ESMFold2Config(**kwargs)


@require_torch
class ESMFold2ConfigTest(unittest.TestCase):
    def setUp(self):
        # ESMFold2Config is composite (sub_configs) with no vocab/hidden_size.
        self.config_tester = ConfigTester(self, config_class=ESMFold2Config, has_text_modality=False, num_loops=5)

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
        # No ESMC backbone is loaded -> LM conditioning is skipped (a valid path),
        # so this exercises the full pure-PyTorch structural stack on CPU.
        for impl in ("sdpa", "eager"):
            with self.subTest(attn_implementation=impl):
                model = self._build(impl)
                self.assertIsNone(model._esmc)
                with torch.no_grad():
                    out = model.infer_protein(self.seq, num_loops=1, num_diffusion_samples=1, num_sampling_steps=2)
                coords = out["sample_atom_coords"]
                self.assertEqual(coords.shape[0], 1)  # num_diffusion_samples
                self.assertEqual(coords.shape[-1], 3)  # xyz
                self.assertTrue(torch.isfinite(coords).all())
                self.assertEqual(out["distogram_logits"].shape[-1], model.config.structure_head.distogram_bins)

    def test_attention_dispatch_attached(self):
        model = self._build("eager")
        swa_modules = [m for m in model.modules() if isinstance(m, SWA3DRoPEAttention)]
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
            # load_esmc=False: skip auto-loading the (real, multi-GB) ESMC backbone;
            # the saved model has no backbone either, so both take the no-LM path.
            reloaded = ESMFold2Model.from_pretrained(tmp, load_esmc=False).eval()

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
        # from_pretrained auto-loads the ESMC backbone (load_esmc=True by default).
        model = ESMFold2Model.from_pretrained("biohub/ESMFold2").to(torch_device).eval()

        seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
        with torch.no_grad():
            output = model.infer_protein(seq, num_diffusion_samples=1)

        coords = output["sample_atom_coords"]
        plddt = output["plddt"]
        self.assertEqual(coords.shape[-1], 3)
        self.assertTrue(torch.isfinite(coords).all())
        # pLDDT is a 0-100 confidence; a real fold of this sequence should be confident.
        self.assertTrue((plddt >= 0).all() and (plddt <= 100).all())
        self.assertGreater(plddt.mean().item(), 50.0)
