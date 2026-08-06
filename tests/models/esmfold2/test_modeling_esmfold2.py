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

ESMFold2's forward takes ~18 structural feature tensors rather than the standard
``input_ids``/``attention_mask``, so it does not plug into ``ModelTesterMixin`` (the file is listed in
``utils/check_repo.py::TEST_FILES_WITH_NO_COMMON_TESTS``).
"""

import tempfile
import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError

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
    from transformers.models.esmfold2.modeling_esmfold2 import EsmFold2AtomAttention, EsmFold2AtomInputs

# TEMP: revert to "biohub/ESMFold2" once that snapshot bundles the ESMC-6B backbone under ``esmc.*``.
_INTEGRATION_CKPT = "Rocketknight1/ESMFold2-merged-temp"


def get_tiny_config(**overrides) -> "EsmFold2Config":
    """A minimal but internally consistent ESMFold2 config for CPU testing.

    The widths ``EsmFold2Config.validate_architecture`` pins are spelled out; see there for the relations.
    """
    kwargs = {
        "hidden_size": 32,
        "pairwise_hidden_size": 16,
        "single_inputs_size": 83,
        "pair_transition_intermediate_size": 64,
        "num_loops": 1,
        "esmc_config": {"hidden_size": 32, "num_attention_heads": 2, "num_hidden_layers": 1, "vocab_size": 64},
        "folding_trunk_num_hidden_layers": 1,
        "sliding_window": 8,
        "parcae_num_coda_layers": 1,
        "atom_encoder": {
            "hidden_size": 16,
            "intermediate_size": 32,
            "output_dim": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_spatial_rope_pairs_per_axis": 1,
            "num_uid_rope_pairs": 1,
        },
        "structure_head": {
            "num_distogram_bins": 8,
            "num_diffusion_samples": 1,
            "diffusion_module": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "pair_intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "atom_encoder": {
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "output_dim": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_spatial_rope_pairs_per_axis": 1,
                    "num_uid_rope_pairs": 1,
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
        config = EsmFold2Config(
            pairwise_hidden_size=72,
            single_inputs_size=99,
            atom_encoder={
                "hidden_size": 64,
                "output_dim": 32,
                "num_spatial_rope_pairs_per_axis": 1,
                "num_uid_rope_pairs": 4,
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            config.save_pretrained(tmp)
            reloaded = EsmFold2Config.from_pretrained(tmp)

        self.assertEqual(reloaded.to_dict(), config.to_dict())
        self.assertEqual(reloaded.pairwise_hidden_size, 72)
        self.assertEqual(reloaded.single_inputs_size, 99)
        self.assertEqual(reloaded.atom_encoder.hidden_size, 64)
        # The bundled ESMC backbone round-trips as a PreTrainedConfig sub-config, not a dict.
        self.assertEqual(type(reloaded.esmc_config).__name__, "EsmcConfig")

    def test_inconsistent_widths_are_rejected(self):
        # single inputs vs. the atom aggregation they contain
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "atom_encoder.output_dim"):
            EsmFold2Config(single_inputs_size=99)
        # the denoiser's atom stack vs. the token width it scatters into
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "output_dim"):
            EsmFold2Config(structure_head={"diffusion_module": {"atom_encoder": {"output_dim": 64}}})
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "frequency pairs"):
            EsmFold2Config(atom_encoder={"num_uid_rope_pairs": 64})

    def test_attn_implementation_propagates_to_subconfigs(self):
        config = EsmFold2Config(attn_implementation="sdpa")
        self.assertEqual(config._attn_implementation, "sdpa")
        self.assertEqual(config.esmc_config._attn_implementation, "sdpa")


@require_torch
class EsmFold2ModelTest(unittest.TestCase):
    seq = "MKLVAAG"

    # Run on CPU for portability; GPU is covered by the slow integration tests below.
    def _build(self, attn_implementation="sdpa"):
        torch.manual_seed(0)
        config = get_tiny_config(attn_implementation=attn_implementation)
        return EsmFold2Model(config).eval()

    def test_forward_runs_on_both_backends(self):
        # End-to-end: the bundled (tiny, random) ESMC backbone, trunk, diffusion and confidence head.
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
                self.assertEqual(out["distogram_logits"].shape[-1], model.config.structure_head.num_distogram_bins)

    def test_attention_dispatch_attached(self):
        model = self._build("eager")
        swa_modules = [m for m in model.modules() if isinstance(m, EsmFold2AtomAttention)]
        # Both atom sites (inputs embedder + diffusion decoder) contribute SWA modules.
        self.assertGreaterEqual(len(swa_modules), 1)
        self.assertTrue(all(m.config is model.config for m in swa_modules))
        self.assertTrue(all(m.config._attn_implementation == "eager" for m in swa_modules))

    @staticmethod
    def _pad_features(features, num_tokens, num_atoms):
        """Right-pad a single-sequence feature dict out to ``(num_tokens, num_atoms)``.

        The zero fill also clears the ``*_attention_mask`` entries, marking the added positions as padding.
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
        """No valid atom may attend to padding, and valid-to-valid attention is exactly the window.

        Padding is passed as the standard 2D ``attention_mask``, which masks *keys*, so a padded query
        row may still see valid keys. That is harmless and deliberate: a padded atom is never itself
        reachable as a key, and its row is dropped at the atom->token scatter. The invariant that
        matters is the key direction, asserted on the mask itself because a batch's extra padding
        lands beyond the window reach of any valid atom, so an output comparison could not see a
        padding-blind mask.
        """
        from transformers.models.esmfold2.protein_utils import prepare_protein_features

        features = prepare_protein_features(self.seq)
        valid = features["atom_attention_mask"][0].bool()
        self.assertLess(int(valid.sum()), valid.numel())  # there is genuinely padding to exclude

        model = self._build()
        _res, _profile, _deletion, ref_element_oh, ref_chars_oh, atom_to_token = model._prepare_features(
            res_type=features["res_type"],
            token_mask=features["token_attention_mask"],
            msa=None,
            msa_attention_mask=None,
            deletion_mean=None,
            ref_element=features["ref_element"],
            ref_atom_name_chars=features["ref_atom_name_chars"],
            atom_attention_mask=features["atom_attention_mask"],
            atom_to_token=features["atom_to_token"],
        )
        # Raw feature tensors, exactly as ``forward`` passes them: boolean mask, integer ``ref_charge``.
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
            atom_embeds, _position_embeddings = encoder.embed_atoms(atom_inputs)
            mask = encoder.build_attention_mask(atom_inputs.atom_attention_mask, atom_embeds)

        per_head = mask[0, 0]
        self.assertFalse(bool(per_head[valid][:, ~valid].any()), "a valid atom may not attend to padding")
        self.assertFalse(bool(per_head[:, ~valid].any()), "a padded atom may never be attended to as a key")

        # Valid-to-valid attention is exactly the symmetric window of radius ``sliding_window // 2``.
        radius = model.config.sliding_window // 2
        index = torch.arange(valid.shape[0])
        within_window = (index[:, None] - index[None, :]).abs() <= radius
        torch.testing.assert_close(per_head[valid][:, valid], within_window[valid][:, valid])

    def test_padded_batch_matches_single_sequence(self):
        """A right-padded sequence folded in a batch must match folding it on its own.

        Covers the token-axis padding batching introduces (the trunk's pair mask); the atom-axis mask
        is covered by `test_swa_mask_excludes_padded_atoms` instead.
        """
        from unittest.mock import patch

        batch, short_features = self._build_padded_batch()
        short_length = short_features["token_attention_mask"].shape[1]
        self.assertLess(short_length, batch["token_attention_mask"].shape[1])  # something to pad

        model = self._build()
        # The trunk is stochastic and batch size perturbs the draws, so pin both sources of randomness.
        model.config.lm_encoder.lm_dropout = 0.0
        kwargs = {"num_loops": 1, "num_diffusion_samples": 1, "num_sampling_steps": 1}
        with (
            patch.object(EsmFold2Model, "_init_pair_state", lambda self, ref: torch.zeros_like(ref)),
            torch.no_grad(),
        ):
            batched = model.fold(**batch, **kwargs)
            alone = model.fold(**short_features, **kwargs)

        # Only the distogram is comparable: it is read off the trunk, before the sampler's RNG.
        torch.testing.assert_close(
            batched.distogram_logits[1, :short_length, :short_length],
            alone.distogram_logits[0],
            rtol=1e-4,
            atol=1e-4,
        )
        self.assertTrue(torch.isfinite(batched.distogram_logits).all())
        self.assertTrue(torch.isfinite(batched.sample_atom_coords).all())

    def _build_padded_batch(self):
        """A batch of two right-padded sequences, plus the shorter one on its own."""
        from transformers.models.esmfold2.protein_utils import prepare_protein_features

        long_features = prepare_protein_features("MKLVAAGKLQ")
        short_features = prepare_protein_features(self.seq)
        num_tokens = long_features["token_attention_mask"].shape[1]
        num_atoms = long_features["atom_attention_mask"].shape[1]
        padded_short = self._pad_features(short_features, num_tokens, num_atoms)
        batch = {key: torch.cat([long_features[key], padded_short[key]], dim=0) for key in long_features}
        return batch, short_features

    def test_denoiser_conditioning_broadcasts_over_diffusion_samples(self):
        """The denoiser's two attention masks must not be materialised per diffusion sample at batch 1.

        They are the largest tensors held across the sampling loop (the per-block token biases are
        ~2.9 GB at length 1000 with eight samples if expanded), they are identical across samples, and
        they are only ever broadcast against — so at `batch_size == 1` their leading dim stays 1 however
        many samples are drawn. A batch of 2 cannot broadcast over the flattened sample batch, so there
        they must be expanded; both shapes are asserted to keep the two paths honest.
        """
        model = self._build()
        denoiser = model.structure_head
        batch, single = self._build_padded_batch()

        def conditioning_for(features, samples):
            feature_kwargs = {k: v for k, v in features.items() if k != "distogram_atom_idx"}
            with torch.no_grad():
                trunk = model(**feature_kwargs)
                return denoiser.prepare_conditioning(
                    atom_inputs=trunk.atom_inputs,
                    pair_trunk=trunk.pair_states,
                    relative_position_encoding=trunk.relative_position_encoding,
                    single_inputs=trunk.single_inputs,
                    token_attention_mask=features["token_attention_mask"],
                    num_diffusion_samples=samples,
                )

        for samples in (1, 4):
            with self.subTest(batch_size=1, num_diffusion_samples=samples):
                conditioning = conditioning_for(single, samples)
                self.assertEqual(conditioning.attention_mask.shape[0], 1)
                self.assertTrue(all(bias.shape[0] == 1 for bias in conditioning.token_attention_bias))
                # The per-sample tensors *are* expanded, which is what the masks broadcast against.
                self.assertEqual(conditioning.atom_embeds.shape[0], samples)
                self.assertEqual(conditioning.projected_single_inputs.shape[0], samples)

        with self.subTest(batch_size=2, num_diffusion_samples=3):
            conditioning = conditioning_for(batch, 3)
            self.assertEqual(conditioning.attention_mask.shape[0], 6)
            self.assertTrue(all(bias.shape[0] == 6 for bias in conditioning.token_attention_bias))

    def test_batched_fold_with_multiple_diffusion_samples(self):
        """Batch > 1 combined with several diffusion samples: the path where the masks are expanded.

        `test_padded_batch_matches_single_sequence` only draws one sample, so nothing else exercises
        the batch-and-samples combination end-to-end.
        """
        batch, _ = self._build_padded_batch()
        model = self._build()
        model.config.lm_encoder.lm_dropout = 0.0
        with torch.no_grad():
            output = model.fold(**batch, num_loops=1, num_diffusion_samples=3, num_sampling_steps=2)

        num_atoms = batch["atom_attention_mask"].shape[1]
        # Sampler output is flattened over (batch, samples).
        self.assertEqual(output["sample_atom_coords"].shape, (2 * 3, num_atoms, 3))
        self.assertTrue(torch.isfinite(output["sample_atom_coords"]).all())
        self.assertTrue(torch.isfinite(output["plddt"]).all())

    def test_output_to_pdb(self):
        """The PDB writer must round-trip every predicted atom, tag chains, and rank samples."""
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
        # The forward is intentionally stochastic, so fidelity is checked at the weight level.
        model = self._build()
        state_before = model.state_dict()

        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            # The ESMC backbone round-trips as a bundled submodule, with no separate load.
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
        # bf16 is the intended inference regime.
        model = EsmFold2Model.from_pretrained(_INTEGRATION_CKPT, dtype=torch.bfloat16).to(torch_device).eval()

        # Ubiquitin (PDB 1UBQ), a textbook well-folding 76-residue domain. The prediction is the
        # best-ranked of the drawn samples, so assert on the best of N.
        seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
        torch.manual_seed(0)
        with torch.no_grad():
            output = model.infer_protein(seq, num_diffusion_samples=8, num_sampling_steps=68)

        coords = output["sample_atom_coords"]
        self.assertEqual(coords.shape[-1], 3)
        self.assertTrue(torch.isfinite(coords).all())

        # 0-1 scale; the CPU-fp32 reference folds ubiquitin at best pLDDT ~0.80, best pTM ~0.74.
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

            expected_distogram = torch.tensor([6.22, 7.44, 9.19, 9.19, 16.00, 18.50, 19.50, 22.50])
            torch.testing.assert_close(
                output["distogram_logits"][0, 0, 1, :8].float().cpu(), expected_distogram, rtol=0, atol=0.2
            )
            self.assertAlmostEqual(output["ptm"].max().item(), 0.743, delta=0.05)
        finally:
            torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])
            torch.backends.cudnn.deterministic = prev[2]
            torch.backends.cudnn.benchmark = prev[3]
            torch.backends.cuda.matmul.allow_tf32 = prev[4]
