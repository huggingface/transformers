# Copyright 2026 the HuggingFace Team. All rights reserved.
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

import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device


if is_torch_available():
    import torch

    from transformers import (
        BlockRefinementSampler,
        BlockRefinementSamplerConfig,
        DiffusionGemmaGenerationConfig,
        DiffusionGemmaGenerationMixin,
        DiscreteDDIMSampler,
        DiscreteDDIMSamplerConfig,
        EntropyBoundSampler,
        EntropyBoundSamplerConfig,
        GenerationConfig,
        LinearTemperatureScheduleLogitsProcessor,
        StableAndConfidentStoppingCriteria,
    )


@require_torch
class DiffusionGemmaGenerationClassesTester(unittest.TestCase):
    def test_generation_config_interface(self):
        """
        Test to confirm that basic `GenerationConfig` are also accepted in `DiffusionGemmaGenerationConfig`.

        If this test pass, it implies that text diffusion has roughly the same interface as AR generation, since we
        can pass kwargs from `generate` to the generation config.
        """
        basic_parameterization = {
            "max_length": 128,
            "max_new_tokens": 64,
            "cache_implementation": "dynamic",
            "pad_token_id": 0,
            "eos_token_id": 1,
        }
        diffusion_generation_config = DiffusionGemmaGenerationConfig(**basic_parameterization)
        ar_generation_config = GenerationConfig(**basic_parameterization)

        for attr in basic_parameterization.keys():
            self.assertEqual(getattr(diffusion_generation_config, attr), getattr(ar_generation_config, attr))

    def test_bad_diffusion_generation_config_parameterization(self):
        """
        Test to ensure that we raise an error when users try to add AR-specific parameters to the
        `DiffusionGemmaGenerationConfig`.
        """
        # Some AR-specific parameters
        ar_parameters = {
            "do_sample": True,
            "num_beams": 4,
            "num_beam_groups": 1,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "encoder_no_repeat_ngram_size": 0,
            "length_penalty": 1.0,
            "early_stopping": False,
            "num_return_sequences": 1,
            "foo": "bar",  # random kwargs are also not accepted
        }
        # All these should raise an exception
        for ar_param, value in ar_parameters.items():
            with self.assertRaises(ValueError, msg=f"key={ar_param}"):
                DiffusionGemmaGenerationConfig(**{ar_param: value})

    def test_save_load_generation_config(self):
        """
        Tests that we can save and load a DiffusionGemmaGenerationConfig, including its inner config dataclasses
        (e.g. a sampler config)
        """
        original_config = DiffusionGemmaGenerationConfig(
            max_new_tokens=64,
            sampler_config=EntropyBoundSamplerConfig(entropy_bound=0.1),
            t_min=0.4,
            t_max=0.8,
            stability_threshold=1,
            confidence_threshold=0.005,
        )
        test_attrs = (
            "max_new_tokens",
            "sampler_config",
            "t_min",
            "t_max",
            "stability_threshold",
            "confidence_threshold",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            original_config.save_pretrained(tmp_dir)
            loaded_config = DiffusionGemmaGenerationConfig.from_pretrained(tmp_dir)
            for attr_name in test_attrs:
                original_attr = getattr(original_config, attr_name)
                loaded_attr = getattr(loaded_config, attr_name)
                self.assertEqual(original_attr, loaded_attr)  # same class, same contents

    def test_save_load_sampler_configs(self):
        """
        Tests that each sampler config round-trips through save/load of the generation config.
        """
        sampler_configs = (
            EntropyBoundSamplerConfig(entropy_bound=0.1),
            DiscreteDDIMSamplerConfig(),
            BlockRefinementSamplerConfig(threshold=0.9, editing_threshold=0.99),
        )
        for sampler_config in sampler_configs:
            original_config = DiffusionGemmaGenerationConfig(sampler_config=sampler_config)
            with tempfile.TemporaryDirectory() as tmp_dir:
                original_config.save_pretrained(tmp_dir)
                loaded_config = DiffusionGemmaGenerationConfig.from_pretrained(tmp_dir)
                self.assertEqual(loaded_config.sampler_config, sampler_config)

    def test_bad_sampler_config(self):
        """
        Tests that an unknown sampler config type is rejected at validation time.
        """
        with self.assertRaises(ValueError):
            DiffusionGemmaGenerationConfig(sampler_config=GenerationConfig())

    def test_eb_sampler_initialize_canvas(self):
        """
        Tests that `initialize_canvas` is working as expected for `EntropyBoundSampler`.
        Canvas ininitalization is random. Two samples are extremelly unlikely to be the same.
        """
        sampler = _get_eb_sampler()
        canvas_1 = sampler.initialize_canvas(batch_size=1, device=torch_device)
        canvas_2 = sampler.initialize_canvas(batch_size=1, device=torch_device)
        self.assertFalse((canvas_1 == canvas_2).all())

    def test_eb_sampler_accept_canvas(self):
        """
        Tests that `accept_canvas` is working as expected for `EntropyBoundSampler`.
        Please see comments in the test for expected logic and corner cases.
        """

        # very loose explanation: the `entropy-bound` (EB) variable controls how much entropy we're willing to accept
        sampler_low_eb = _get_eb_sampler(entropy_bound=1e-2)
        sampler_high_eb = _get_eb_sampler(entropy_bound=1e-1)

        current_canvas = sampler_high_eb.initialize_canvas(batch_size=1, device=torch_device)
        denoiser_canvas = sampler_high_eb.initialize_canvas(batch_size=1, device=torch_device)

        # create logits such that all positions have high entropy, except for a few select cases
        # NOTE: the first token above the threshold is accepted
        logits = torch.zeros((1, 256, 10000), device=torch_device)  # [bsz, canvas_len, vocab_size]
        logits[0, 0, 0] = 1.8e1  # token entropy at position 0 = 2.9e-3 -> accepted in both cases
        logits[0, 1, 1] = 1.45e1  # token entropy at position 1 = 7.8e-2 -> accepted in both cases
        logits[0, 2, 2] = 1.45e1  # token entropy at position 2 = 7.8e-2 -> accepted only in the high eb case

        # higher EB -> more accepted tokens
        accepted_high_eb = sampler_high_eb.accept_canvas(
            current_canvas=current_canvas, denoiser_canvas=denoiser_canvas, logits=logits, cur_step=None
        )
        accepted_low_eb = sampler_low_eb.accept_canvas(
            current_canvas=current_canvas, denoiser_canvas=denoiser_canvas, logits=logits, cur_step=None
        )
        num_accepted_high_eb = (accepted_high_eb == denoiser_canvas).sum().item()
        num_accepted_low_eb = (accepted_low_eb == denoiser_canvas).sum().item()
        self.assertTrue(num_accepted_high_eb == num_accepted_low_eb + 1)

    def test_eb_sampler_renoise_canvas(self):
        """
        Tests that `renoise_canvas` is working as expected for `EntropyBoundSampler`.
        All non-accepted tokens are renoised.
        """
        sampler = _get_eb_sampler(entropy_bound=1e-1)

        # NOTE: `renoise_canvas` is stateful: depends on the outcome of `accept_canvas`, as it renoises all
        # non-accepted tokens
        current_canvas = sampler.initialize_canvas(batch_size=1, device=torch_device)
        denoiser_canvas = sampler.initialize_canvas(batch_size=1, device=torch_device)

        logits = torch.zeros((1, 256, 10000), device=torch_device)  # [bsz, canvas_len, vocab_size]
        # corresponding token entropy = 0 -> these 9 tokens will definitely get accepted and, therefore, not renoised
        # (but the first token above the threshold is also accepted, so we'll have 9+1 accepted tokens)
        logits[0, :9, 0] = 1e6

        accepted_canvas = sampler.accept_canvas(
            current_canvas=current_canvas, denoiser_canvas=denoiser_canvas, logits=logits, cur_step=None
        )
        renoised_canvas = sampler.renoise_canvas(accepted_canvas=accepted_canvas, cur_step=None)
        num_not_renoised_canvas = (renoised_canvas == accepted_canvas).sum().item()
        self.assertGreaterEqual(num_not_renoised_canvas, 10)  # can be >10 if the same token is sampled in the same pos
        self.assertTrue((accepted_canvas[0, :9] == renoised_canvas[0, :9]).all())

    def test_ddim_sampler_accept_canvas(self):
        """
        Tests that `accept_canvas` is working as expected for `DiscreteDDIMSampler`.
        Please see comments in the test for expected logic and corner cases.
        """
        sampler = _get_ddim_sampler()
        current_canvas = sampler.initialize_canvas(batch_size=1, device=torch_device)
        denoiser_canvas = sampler.initialize_canvas(batch_size=1, device=torch_device)
        logits = torch.zeros((1, 256, 10000), device=torch_device)

        # the last step has alpha_s = 1: the posterior deterministically commits the predicted clean tokens
        accepted_canvas = sampler.accept_canvas(
            current_canvas=current_canvas, denoiser_canvas=denoiser_canvas, logits=logits, cur_step=1
        )
        self.assertTrue((accepted_canvas == denoiser_canvas).all())
        self.assertTrue(sampler.accepted_token_mask.all())

        # at an intermediate step, positions where the denoiser agrees with the current canvas carry almost all the
        # posterior mass on the clean route, so nearly the whole canvas is kept
        accepted_canvas = sampler.accept_canvas(
            current_canvas=current_canvas, denoiser_canvas=current_canvas, logits=logits, cur_step=24
        )
        num_kept = (accepted_canvas == current_canvas).sum().item()
        self.assertGreaterEqual(num_kept, 250)

    def test_ddim_sampler_renoise_canvas(self):
        """
        Tests that `renoise_canvas` is a no-op for `DiscreteDDIMSampler`: the noise route of `accept_canvas`
        already renoises.
        """
        sampler = _get_ddim_sampler()
        accepted_canvas = sampler.initialize_canvas(batch_size=1, device=torch_device)
        renoised_canvas = sampler.renoise_canvas(accepted_canvas=accepted_canvas, cur_step=24)
        self.assertTrue((renoised_canvas == accepted_canvas).all())

    def test_block_refinement_sampler_accept_canvas(self):
        """
        Tests that `accept_canvas` is working as expected for `BlockRefinementSampler`.
        Please see comments in the test for expected logic and corner cases.
        """
        # threshold = 1.0 disables threshold commits (confidence is strictly below 1), so only the quota applies
        sampler = _get_block_refinement_sampler(threshold=1.0)
        current_canvas = sampler.initialize_canvas(batch_size=1, device=torch_device)
        denoiser_canvas = sampler.initialize_canvas(batch_size=1, device=torch_device)
        logits = torch.zeros((1, 256, 10000), device=torch_device)

        # first step (`cur_step` counts down from `max_denoising_steps`): the quota is ceil(256 / 48) = 6 positions
        sampler.accept_canvas(
            current_canvas=current_canvas, denoiser_canvas=denoiser_canvas, logits=logits, cur_step=48
        )
        self.assertEqual(sampler.accepted_token_mask.sum().item(), 6)

        # second step: the cumulative quota grows to ceil(2 * 256 / 48) = 11, committed positions stay committed
        committed_before = sampler.accepted_token_mask.clone()
        sampler.accept_canvas(
            current_canvas=current_canvas, denoiser_canvas=denoiser_canvas, logits=logits, cur_step=47
        )
        self.assertEqual(sampler.accepted_token_mask.sum().item(), 11)
        self.assertTrue(sampler.accepted_token_mask[committed_before].all())

        # last step: the whole canvas is committed
        sampler.accept_canvas(
            current_canvas=current_canvas, denoiser_canvas=denoiser_canvas, logits=logits, cur_step=1
        )
        self.assertTrue(sampler.accepted_token_mask.all())

        # tokens above `threshold` are committed beyond the quota of 6
        sampler = _get_block_refinement_sampler(threshold=0.5)
        sampler.initialize_canvas(batch_size=1, device=torch_device)
        confident_logits = logits.clone()
        confident_logits[0, torch.arange(20), denoiser_canvas[0, :20]] = 1e6
        sampler.accept_canvas(
            current_canvas=current_canvas, denoiser_canvas=denoiser_canvas, logits=confident_logits, cur_step=48
        )
        self.assertEqual(sampler.accepted_token_mask.sum().item(), 20)

    def test_block_refinement_sampler_editing(self):
        """
        Tests that `editing_threshold` lets the sampler replace a committed token when the denoiser confidently
        disagrees with it.
        """
        sampler = _get_block_refinement_sampler(threshold=1.0, editing_threshold=0.5)
        current_canvas = sampler.initialize_canvas(batch_size=1, device=torch_device)
        sampler.accepted_token_mask[:] = True  # pretend the whole canvas is already committed

        denoiser_canvas = (current_canvas + 1) % 10000  # the denoiser disagrees everywhere...
        logits = torch.zeros((1, 256, 10000), device=torch_device)
        logits[0, 0, denoiser_canvas[0, 0]] = 1e6  # ...but is only confident at position 0

        accepted_canvas = sampler.accept_canvas(
            current_canvas=current_canvas, denoiser_canvas=denoiser_canvas, logits=logits, cur_step=24
        )
        self.assertEqual(accepted_canvas[0, 0], denoiser_canvas[0, 0])
        self.assertTrue((accepted_canvas[0, 1:] == current_canvas[0, 1:]).all())

    def test_block_refinement_sampler_renoise_canvas(self):
        """
        Tests that `renoise_canvas` is working as expected for `BlockRefinementSampler`.
        Committed tokens are kept and all other positions are renoised.
        """
        sampler = _get_block_refinement_sampler(threshold=1.0)
        accepted_canvas = sampler.initialize_canvas(batch_size=1, device=torch_device)
        sampler.accepted_token_mask[0, :6] = True

        renoised_canvas = sampler.renoise_canvas(accepted_canvas=accepted_canvas, cur_step=47)
        self.assertTrue((renoised_canvas[0, :6] == accepted_canvas[0, :6]).all())
        self.assertFalse((renoised_canvas[0, 6:] == accepted_canvas[0, 6:]).all())

    def test_linear_temperature_schedule(self):
        t_min = 0.4
        t_max = 0.8
        max_dns = 48
        logits_processor = LinearTemperatureScheduleLogitsProcessor(
            t_min=t_min, t_max=t_max, max_denoising_steps=max_dns
        )
        scores = torch.ones((1, 10), device=torch_device)

        # cur_step == max_denoising_steps -> applies maximum temperature
        modified_scores = logits_processor(input_ids=None, scores=scores, cur_step=max_dns)
        self.assertTrue((modified_scores == scores / t_max).all())

        # cur_step == max_denoising_steps/2 -> applies (t_max + t_min)/2
        modified_scores = logits_processor(input_ids=None, scores=scores, cur_step=max_dns / 2)
        self.assertTrue((modified_scores == scores / ((t_max + t_min) / 2)).all())

    def test_stable_and_confident_stopping_criteria_confidence(self):
        """
        Tests the behaviour of `confidence_threshold` in `StableAndConfidentStoppingCriteria`
        """
        stopping_criteria_strict = StableAndConfidentStoppingCriteria(stability_threshold=0, confidence_threshold=1e-2)
        # vocab size = 10000 -> max entropy = 9.21 -> a confidence threshold >9.21 will accept everything
        stopping_criteria_lax = StableAndConfidentStoppingCriteria(stability_threshold=0, confidence_threshold=9.20)
        stopping_criteria_too_lax = StableAndConfidentStoppingCriteria(
            stability_threshold=0, confidence_threshold=9.22
        )

        # this should NEVER trigger the stopping criteria, assuming the the theshold is < ln(1/vocab_size)
        logits_max_entropy = torch.zeros((1, 10, 10000), device=torch_device)
        self.assertFalse(stopping_criteria_strict(argmax_canvas=None, logits=logits_max_entropy).all())
        self.assertFalse(stopping_criteria_lax(argmax_canvas=None, logits=logits_max_entropy).all())
        # # sanity-check
        self.assertTrue(stopping_criteria_too_lax(argmax_canvas=None, logits=logits_max_entropy).all())

        # mean entropy = 7.8e-2 -> only the lax triggers
        logits_medium_entropy = torch.zeros((1, 10, 10000), device=torch_device)
        logits_medium_entropy[:, :, 0] = 1.45e1
        self.assertFalse(stopping_criteria_strict(argmax_canvas=None, logits=logits_medium_entropy).all())
        self.assertTrue(stopping_criteria_lax(argmax_canvas=None, logits=logits_medium_entropy).all())

        # mean entropy = 2.9e-3 -> both trigger
        logits_low_entropy = torch.zeros((1, 10, 10000), device=torch_device)
        logits_low_entropy[:, :, 0] = 1.8e1
        self.assertTrue(stopping_criteria_strict(argmax_canvas=None, logits=logits_low_entropy).all())
        self.assertTrue(stopping_criteria_lax(argmax_canvas=None, logits=logits_low_entropy).all())

    def test_stable_and_confident_stopping_criteria_stability(self):
        """
        Tests the behaviour of `stability_threshold` in `StableAndConfidentStoppingCriteria`
        """
        # vocab size = 10000 -> max entropy = 9.21 -> a confidence threshold >9.21 will accept everything
        stopping_criteria_1 = StableAndConfidentStoppingCriteria(stability_threshold=1, confidence_threshold=9.22)
        stopping_criteria_2 = StableAndConfidentStoppingCriteria(stability_threshold=2, confidence_threshold=9.22)

        logits = torch.zeros((1, 10, 10000), device=torch_device)  # mean entropy = 9.21
        argmax_canvas_1 = torch.randint(low=0, high=10000, size=(1, 10), device=torch_device)
        argmax_canvas_2 = torch.randint(low=0, high=10000, size=(1, 10), device=torch_device)

        # In both cases, they won't trigger after 1 canvas (needs to meet the stability criteria)
        self.assertFalse(stopping_criteria_1(argmax_canvas=argmax_canvas_1, logits=logits).all())
        self.assertFalse(stopping_criteria_2(argmax_canvas=argmax_canvas_1, logits=logits).all())

        # `stopping_criteria_1` will be happy after 2 steps with the same canvas
        self.assertTrue(stopping_criteria_1(argmax_canvas=argmax_canvas_1, logits=logits).all())
        self.assertFalse(stopping_criteria_2(argmax_canvas=argmax_canvas_1, logits=logits).all())

        # both will be happy after 3 steps with the same canvas
        self.assertTrue(stopping_criteria_1(argmax_canvas=argmax_canvas_1, logits=logits).all())
        self.assertTrue(stopping_criteria_2(argmax_canvas=argmax_canvas_1, logits=logits).all())

        # If we pass a different canvas, the stability criteria will be set to false
        self.assertFalse(stopping_criteria_1(argmax_canvas=argmax_canvas_2, logits=logits).all())
        self.assertFalse(stopping_criteria_2(argmax_canvas=argmax_canvas_2, logits=logits).all())

    def test_tokens_per_forward(self):
        """
        Tests that the tokens per forward implementation is working as expected, for bsz == 1
        """
        input_ids = torch.tensor([[5] * 100], dtype=torch.int)
        decoder_forward_passes = torch.tensor([10], dtype=torch.int)
        initial_input_ids_len = 0
        pad_token_id = 1

        tokens_per_forward = DiffusionGemmaGenerationMixin._compute_tokens_per_forward(
            input_ids, decoder_forward_passes, initial_input_ids_len, pad_token_id
        )
        self.assertEqual(tokens_per_forward[0], 100 / 10)

        initial_input_ids_len = 10
        tokens_per_forward = DiffusionGemmaGenerationMixin._compute_tokens_per_forward(
            input_ids, decoder_forward_passes, initial_input_ids_len, pad_token_id
        )
        self.assertEqual(tokens_per_forward[0], (100 - 10) / 10)

        input_ids[:, -30:] = pad_token_id
        tokens_per_forward = DiffusionGemmaGenerationMixin._compute_tokens_per_forward(
            input_ids, decoder_forward_passes, initial_input_ids_len, pad_token_id
        )
        self.assertEqual(tokens_per_forward[0], (100 - 10 - 30) / 10)

    def test_tokens_per_forward_batched(self):
        """
        Tests that the tokens per forward implementation is working as expected, for bsz > 1
        """
        input_ids = torch.tensor([[5] * 100] * 2, dtype=torch.int)
        decoder_forward_passes = torch.tensor([10, 7], dtype=torch.int)
        initial_input_ids_len = 0
        pad_token_id = 1

        tokens_per_forward = DiffusionGemmaGenerationMixin._compute_tokens_per_forward(
            input_ids, decoder_forward_passes, initial_input_ids_len, pad_token_id
        )
        torch.testing.assert_close(tokens_per_forward, torch.tensor([100 / 10, 100 / 7]))

        initial_input_ids_len = 10
        tokens_per_forward = DiffusionGemmaGenerationMixin._compute_tokens_per_forward(
            input_ids, decoder_forward_passes, initial_input_ids_len, pad_token_id
        )
        torch.testing.assert_close(tokens_per_forward, torch.tensor([(100 - 10) / 10, (100 - 10) / 7]))

        input_ids[0, -30:] = pad_token_id
        input_ids[1, -15:] = pad_token_id
        tokens_per_forward = DiffusionGemmaGenerationMixin._compute_tokens_per_forward(
            input_ids, decoder_forward_passes, initial_input_ids_len, pad_token_id
        )
        torch.testing.assert_close(tokens_per_forward, torch.tensor([(100 - 10 - 30) / 10, (100 - 10 - 15) / 7]))


def _get_eb_sampler(entropy_bound: float = 0.1) -> EntropyBoundSampler:
    """Returns a parameterized `EntropyBoundSampler`"""
    sampler_config = EntropyBoundSamplerConfig(entropy_bound=entropy_bound)
    sampler = EntropyBoundSampler(
        config=sampler_config,
        canvas_length=256,
        vocab_size=10000,
        max_denoising_steps=48,
    )
    return sampler


def _get_ddim_sampler() -> DiscreteDDIMSampler:
    """Returns a parameterized `DiscreteDDIMSampler`"""
    sampler = DiscreteDDIMSampler(
        config=DiscreteDDIMSamplerConfig(),
        canvas_length=256,
        vocab_size=10000,
        max_denoising_steps=48,
    )
    return sampler


def _get_block_refinement_sampler(
    threshold: float = 0.95, editing_threshold: float | None = None
) -> BlockRefinementSampler:
    """Returns a parameterized `BlockRefinementSampler`"""
    sampler_config = BlockRefinementSamplerConfig(threshold=threshold, editing_threshold=editing_threshold)
    sampler = BlockRefinementSampler(
        config=sampler_config,
        canvas_length=256,
        vocab_size=10000,
        max_denoising_steps=48,
    )
    return sampler
