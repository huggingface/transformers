# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from transformers import EsmFold2Config, is_torch_available
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

    from transformers import EsmFold2Model
    from transformers.models.esmfold2.modeling_esmfold2 import EsmFold2AtomInputs, EsmFold2SWA3DRoPEAttention

# TEMP: the public ``biohub/ESMFold2`` snapshot does not yet bundle the ESMC-6B
# backbone under ``esmc.*`` (it loads random → garbage outputs). Point the slow
# integration tests at the locally-bundled checkpoint for now. REVERT to
# "biohub/ESMFold2" once the backbone is bundled there.
_INTEGRATION_CKPT = "Rocketknight1/ESMFold2-merged-temp"


def get_tiny_config(**overrides) -> "EsmFold2Config":
    """A minimal but internally consistent ESMFold2 config for CPU testing.

    Constraints (see modeling): 3D RoPE needs ``3*n_spatial + n_uid <= head_dim//2``
    (head_dim = hidden_size / num_attention_heads of each atom sub-config — the inputs
    ``atom_encoder`` and ``structure_head.diffusion_module.atom_encoder`` — = 8 here). The
    inputs atom encoder's ``output_dim`` is derived as
    ``single_inputs_size - (2 * num_res_types + 1)`` (83 - 67 = 16 here), which also feeds
    the diffusion conditioning.
    """
    kwargs = {
        "hidden_size": 32,
        "pairwise_hidden_size": 16,
        "single_inputs_size": 83,
        "num_loops": 1,
        "num_diffusion_samples": 1,
        "esmc_config": {"d_model": 32, "n_heads": 2, "n_layers": 1, "vocab_size": 64},
        "folding_trunk_num_hidden_layers": 1,
        "sliding_window": 8,
        "parcae_num_coda_layers": 1,
        "atom_encoder": {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "n_spatial_rope_pairs_per_axis": 1,
            "n_uid_rope_pairs": 1,
        },
        "structure_head": {
            "distogram_bins": 8,
            "diffusion_module": {
                "token_hidden_size": 32,
                "token_num_blocks": 1,
                "token_num_heads": 2,
                "atom_encoder": {
                    "hidden_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                },
            },
        },
        "confidence_head": {
            "num_plddt_bins": 4,
            "num_pde_bins": 4,
            "num_pae_bins": 4,
            "distogram_bins": 8,
            "num_hidden_layers": 1,
        },
        "lm_encoder": {"num_hidden_layers": 1},
    }
    kwargs.update(overrides)
    return EsmFold2Config(**kwargs)


class EsmFold2ConfigTester(ConfigTester):
    @unittest.skip("ESMFold2 sub-configs are not standalone auto-registered configs")
    def create_and_test_config_from_and_save_pretrained_composite(self):
        pass


@require_torch
class EsmFold2ConfigTest(unittest.TestCase):
    def setUp(self):
        # EsmFold2Config is composite (sub_configs) with no vocab/hidden_size.
        self.config_tester = EsmFold2ConfigTester(
            self, config_class=EsmFold2Config, has_text_modality=False, num_loops=5
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_config_round_trip(self):
        config = EsmFold2Config(pairwise_hidden_size=72, single_inputs_size=99, atom_encoder={"hidden_size": 64})
        with tempfile.TemporaryDirectory() as tmp:
            config.save_pretrained(tmp)
            reloaded = EsmFold2Config.from_pretrained(tmp)

        self.assertEqual(reloaded.to_dict(), config.to_dict())
        self.assertEqual(reloaded.pairwise_hidden_size, 72)
        self.assertEqual(reloaded.single_inputs_size, 99)
        self.assertEqual(reloaded.atom_encoder.hidden_size, 64)
        # The bundled ESMC backbone round-trips as a PreTrainedConfig sub-config, not a dict.
        self.assertEqual(type(reloaded.esmc_config).__name__, "EsmcConfig")

    def test_attn_implementation_propagates_to_subconfigs(self):
        config = EsmFold2Config(attn_implementation="sdpa")
        self.assertEqual(config._attn_implementation, "sdpa")
        self.assertEqual(config.esmc_config._attn_implementation, "sdpa")


@require_torch
class EsmFold2ModelTest(unittest.TestCase):
    seq = "MKLVAAG"

    # These are pure-PyTorch correctness smoke tests, run on CPU for portability
    # (the diffusion sampler is tiny here); GPU is covered by the slow integration
    # test below.
    def _build(self, attn_implementation="sdpa"):
        torch.manual_seed(0)
        config = get_tiny_config(attn_implementation=attn_implementation)
        return EsmFold2Model(config).eval()

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
        swa_modules = [m for m in model.modules() if isinstance(m, EsmFold2SWA3DRoPEAttention)]
        # Both atom sites (inputs embedder + diffusion decoder) contribute SWA modules.
        self.assertGreaterEqual(len(swa_modules), 1)
        self.assertTrue(all(m.config is model.config for m in swa_modules))
        self.assertTrue(all(m.config._attn_implementation == "eager" for m in swa_modules))

    @staticmethod
    def _pad_features(features, num_tokens, num_atoms):
        """Right-pad a single-sequence feature dict out to ``(num_tokens, num_atoms)``.

        Every axis matching the source token count is padded to ``num_tokens`` and every axis
        matching the source atom count to ``num_atoms``; the zero fill also clears the two
        ``*_attention_mask`` entries, which is what marks the added positions as padding.
        """
        src_tokens = features["token_attention_mask"].shape[1]
        src_atoms = features["atom_attention_mask"].shape[1]
        padded = {}
        for key, value in features.items():
            target = list(value.shape)
            for dim in range(1, value.dim()):
                if value.shape[dim] == src_tokens:
                    target[dim] = num_tokens
                elif value.shape[dim] == src_atoms:
                    target[dim] = num_atoms
            spec = []
            for dim in reversed(range(value.dim())):
                spec.extend([0, target[dim] - value.shape[dim]])
            padded[key] = torch.nn.functional.pad(value, spec) if any(spec) else value
        return padded

    def test_swa_mask_excludes_padded_atoms(self):
        """The sliding-window atom mask must not connect valid atoms and padding, in either direction.

        `prepare_protein_features` right-pads the atom axis (a short sequence fills only part of its
        atom budget), so this path is live in every fold. Asserted on the mask directly rather than
        end-to-end: the extra padding a batch introduces lands beyond the +/-half_window reach of any
        valid atom, so an output comparison cannot see a padding-blind mask.
        """
        from transformers.models.esmfold2.protein_utils import prepare_protein_features

        features = prepare_protein_features(self.seq)
        valid = features["atom_attention_mask"][0].bool()
        self.assertLess(int(valid.sum()), valid.numel())  # there is genuinely padding to exclude

        model = self._build()
        _res, _profile, _deletion, ref_element_oh, ref_chars_oh, atom_to_token = model._prepare_features(
            res_type=features["res_type"],
            tok_mask=features["token_attention_mask"],
            msa=None,
            msa_attention_mask=None,
            deletion_mean=None,
            ref_element=features["ref_element"],
            ref_atom_name_chars=features["ref_atom_name_chars"],
            atom_attention_mask=features["atom_attention_mask"],
            atom_to_token=features["atom_to_token"],
        )
        # Pass the raw feature tensors through, exactly as ``forward`` does: the atom mask is boolean
        # (the sliding-window ``and`` mask composes it with ``&``) and ``ref_charge`` is an integer
        # promoted by the featurizer's concat.
        atom_inputs = EsmFold2AtomInputs(
            ref_pos=features["ref_pos"],
            ref_charge=features["ref_charge"],
            atom_attention_mask=features["atom_attention_mask"],
            ref_element=ref_element_oh,
            ref_atom_name_chars=ref_chars_oh,
            ref_space_uid=features["ref_space_uid"],
            atom_to_token=atom_to_token,
        )
        with torch.no_grad():
            encoder = model.inputs_atom_encoder
            _c_base, position_embeddings = encoder.embed_atoms(atom_inputs)
            mask = encoder.build_attention_mask(atom_inputs.atom_attention_mask, position_embeddings)

        per_head = mask[0, 0]
        self.assertFalse(bool(per_head[valid][:, ~valid].any()), "a valid atom may not attend to padding")
        self.assertFalse(bool(per_head[~valid][:, valid].any()), "padding may not attend to a valid atom")

    def test_padded_batch_matches_single_sequence(self):
        """A right-padded sequence folded in a batch must match folding it on its own.

        Covers batched folding and the token-axis padding it introduces (the trunk's pair mask),
        which nothing else exercises: `prepare_protein_features` featurizes one sequence at a time,
        so every other test and the folding-regression script run with a single, full-length batch.
        The atom-axis mask is covered by `test_swa_mask_excludes_padded_atoms` instead.
        """
        from unittest.mock import patch

        from transformers.models.esmfold2.protein_utils import prepare_protein_features

        long_features = prepare_protein_features("MKLVAAGKLQ")
        short_features = prepare_protein_features(self.seq)
        num_tokens = long_features["token_attention_mask"].shape[1]
        num_atoms = long_features["atom_attention_mask"].shape[1]
        short_length = short_features["token_attention_mask"].shape[1]
        self.assertLess(short_length, num_tokens)  # there is genuinely something to pad

        padded_short = self._pad_features(short_features, num_tokens, num_atoms)
        batch = {key: torch.cat([long_features[key], padded_short[key]], dim=0) for key in long_features}

        model = self._build()
        # The trunk is stochastic (random initial pair state + per-loop LM dropout), and a batch of 2
        # would not draw the same randomness as a batch of 1, so pin both to compare like with like.
        model.config.lm_encoder.lm_dropout = 0.0
        kwargs = {"num_loops": 1, "num_diffusion_samples": 1, "num_sampling_steps": 1}
        with (
            patch.object(EsmFold2Model, "_init_pair_state", lambda self, ref: torch.zeros_like(ref)),
            torch.no_grad(),
        ):
            batched = model.fold(**batch, **kwargs)
            alone = model.fold(**short_features, **kwargs)

        # Only the distogram is comparable: it is computed from the trunk before any diffusion
        # sampling, so it does not depend on the sampler's RNG (which the batch size perturbs).
        torch.testing.assert_close(
            batched.distogram_logits[1, :short_length, :short_length],
            alone.distogram_logits[0],
            rtol=1e-4,
            atol=1e-4,
        )
        self.assertTrue(torch.isfinite(batched.distogram_logits).all())
        self.assertTrue(torch.isfinite(batched.sample_atom_coords).all())

    def test_output_to_pdb(self):
        """The PDB writer must round-trip every predicted atom, tag chains, and rank samples.

        These are the three things the previous OpenFold-based writer could not do: it projected
        atoms onto a canonical 37-slot protein layout (silently dropping anything else), rendered
        every chain as chain A, and always emitted diffusion sample 0.
        """
        from transformers.models.esmfold2.protein_utils import (
            _encode_atom_name,
            output_to_pdb,
            prepare_protein_features,
        )

        model = self._build()
        features = prepare_protein_features("MKLVAAGCWQ")
        with torch.no_grad():
            output = model.fold(**features, num_loops=1, num_diffusion_samples=4, num_sampling_steps=1)

        def atom_lines(pdb):
            return [line for line in pdb.splitlines() if line.startswith("ATOM")]

        # Every valid atom is written, and the columnar record is the right width.
        pdb = output_to_pdb(output, features)
        num_valid_atoms = int(features["atom_attention_mask"].sum())
        self.assertEqual(len(atom_lines(pdb)), num_valid_atoms)
        self.assertTrue(all(len(line) == 80 for line in atom_lines(pdb)))
        self.assertTrue(pdb.endswith("END\n"))

        # A non-canonical atom name survives instead of being dropped.
        renamed = {key: value.clone() for key, value in features.items()}
        renamed["ref_atom_name_chars"][0, 4] = torch.tensor(_encode_atom_name("ZN"))
        names = [line[12:16].strip() for line in atom_lines(output_to_pdb(output, renamed))]
        self.assertIn("ZN", names)
        self.assertEqual(len(names), num_valid_atoms)

        # A second chain gets its own tag and its own TER record.
        multi_chain = {key: value.clone() for key, value in features.items()}
        multi_chain["asym_id"][0, 5:] = 1
        pdb = output_to_pdb(output, multi_chain)
        self.assertEqual(sorted({line[21] for line in atom_lines(pdb)}), ["A", "B"])
        self.assertEqual(sum(line.startswith("TER") for line in pdb.splitlines()), 2)

        # The rendered sample is the best-ranked one, not sample 0.
        best = int(output["ptm"].float().argmax())
        self.assertEqual(output_to_pdb(output, features), output_to_pdb(output, features, sample_idx=best))

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
            reloaded = EsmFold2Model.from_pretrained(tmp).eval()

        state_after = reloaded.state_dict()
        self.assertEqual(set(state_before), set(state_after))
        for key, tensor in state_before.items():
            torch.testing.assert_close(state_after[key], tensor, rtol=0, atol=0)

        with torch.no_grad():
            out = reloaded.infer_protein(self.seq, num_loops=1, num_diffusion_samples=1, num_sampling_steps=1)
        self.assertTrue(torch.isfinite(out["sample_atom_coords"]).all())


@require_torch
class EsmFold2IntegrationTest(TestCasePlus):
    @slow
    @require_torch_accelerator
    def test_inference_protein_folding(self):
        # bf16 is the intended inference regime; the ESMC backbone is bundled in the
        # checkpoint and loaded with the model.
        model = EsmFold2Model.from_pretrained(_INTEGRATION_CKPT, dtype=torch.bfloat16).to(torch_device).eval()

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
        model = EsmFold2Model.from_pretrained(_INTEGRATION_CKPT, dtype=torch.float32).eval()

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

            model = EsmFold2Model.from_pretrained(_INTEGRATION_CKPT, dtype=torch.bfloat16).to(torch_device).eval()
            seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
            torch.manual_seed(0)
            with torch.no_grad():
                output = model.infer_protein(seq, num_loops=4, num_diffusion_samples=2, num_sampling_steps=32)

            expected_distogram = torch.tensor([6.03, 7.38, 9.00, 8.94, 15.75, 18.25, 19.25, 22.25])
            torch.testing.assert_close(
                output["distogram_logits"][0, 0, 1, :8].float().cpu(), expected_distogram, rtol=0, atol=0.2
            )
            self.assertAlmostEqual(output["ptm"].max().item(), 0.743, delta=0.05)
        finally:
            torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])
            torch.backends.cudnn.deterministic = prev[2]
            torch.backends.cudnn.benchmark = prev[3]
            torch.backends.cuda.matmul.allow_tf32 = prev[4]
