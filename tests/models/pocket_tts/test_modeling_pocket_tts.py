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
"""Testing suite for the PyTorch Pocket TTS model."""

import json
import math
import os
import tempfile
import unittest
from unittest import mock

from huggingface_hub import download_bucket_files

from transformers import PocketTTSConfig, PocketTTSFlowHeadConfig
from transformers.testing_utils import is_torch_available, require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin, with_grad
from ...test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch
    from safetensors.torch import load_file
    from torch import nn

    from transformers import (
        AutoModelForTextToWaveform,
        AutoProcessor,
        DynamicCache,
        PocketTTSForConditionalGeneration,
        PocketTTSModel,
        Trainer,
        TrainingArguments,
    )
    from transformers.loss.loss_lsd import PocketTTSForConditionalGenerationLoss


@require_torch
class PocketTTSModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_prompt_frames=2,
        is_training=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        latent_dim=8,
        flow_hidden_size=16,
        flow_num_hidden_layers=2,
        flow_type="lsd",
    ):
        self.parent = parent
        self.model_class = PocketTTSForConditionalGeneration if is_torch_available() else None
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_prompt_frames = num_prompt_frames
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.latent_dim = latent_dim
        self.flow_hidden_size = flow_hidden_size
        self.flow_num_hidden_layers = flow_num_hidden_layers
        self.flow_type = flow_type
        self.pad_token_id = 0
        self.audio_prompt_token_id = vocab_size - 4
        self.audio_latent_token_id = vocab_size - 3
        self.audio_prompt_bos_token_id = vocab_size - 2
        self.audio_bos_token_id = vocab_size - 1
        # what precedes the voice placeholders, and the width of the voice latents
        self.voice_prefix_ids = [self.audio_prompt_bos_token_id]
        self.audio_prompt_dim = latent_dim

    def get_config(self):
        return PocketTTSConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            latent_dim=self.latent_dim,
            pad_token_id=self.pad_token_id,
            audio_prompt_token_id=self.audio_prompt_token_id,
            audio_latent_token_id=self.audio_latent_token_id,
            audio_prompt_bos_token_id=self.audio_prompt_bos_token_id,
            audio_bos_token_id=self.audio_bos_token_id,
            flow_head_config=PocketTTSFlowHeadConfig(
                hidden_size=self.flow_hidden_size,
                num_hidden_layers=self.flow_num_hidden_layers,
                latent_dim=self.latent_dim,
                condition_dim=self.hidden_size,
                flow_type=self.flow_type,
            ),
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        # text tokens (ids in [1, vocab_size - 5]) closed by the first audio position, as the processor lays them out
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 5) + 1
        input_ids[:, -1] = self.audio_bos_token_id
        attention_mask = torch.ones_like(input_ids)
        return config, input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}

    def _with_voice_placeholders(self, input_ids, attention_mask, num_frames):
        """Prefixes each row with the voice (BOS + `num_frames[i]` placeholders) and left-pads, as the processor does."""
        rows = [
            torch.cat(
                [
                    torch.tensor(self.voice_prefix_ids + [self.audio_prompt_token_id] * frames),
                    input_ids[i].cpu(),
                ]
            )
            for i, frames in enumerate(num_frames)
        ]
        length = max(len(row) for row in rows)
        padded = torch.full((len(rows), length), self.pad_token_id)
        mask = torch.zeros((len(rows), length), dtype=torch.long)
        for i, row in enumerate(rows):
            padded[i, length - len(row) :] = row
            mask[i, length - len(row) :] = 1
        return padded.to(torch_device), mask.to(torch_device)

    def prepare_config_and_inputs_with_voice(self):
        """Inputs with precomputed voice latents, one voice length per row, right-padded with their mask."""
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        num_frames = [self.num_prompt_frames, self.num_prompt_frames + 1, self.num_prompt_frames]
        input_ids, attention_mask = self._with_voice_placeholders(input_ids, attention_mask, num_frames)
        audio_prompt_latents = torch.randn(
            self.batch_size, max(num_frames), self.audio_prompt_dim, device=torch_device
        )
        audio_prompt_latents_mask = torch.zeros(
            self.batch_size, max(num_frames), dtype=torch.long, device=torch_device
        )
        for row, frames in enumerate(num_frames):
            audio_prompt_latents_mask[row, :frames] = 1
        return config, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_prompt_latents": audio_prompt_latents,
            "audio_prompt_latents_mask": audio_prompt_latents_mask,
        }

    def with_training_targets(self, inputs_dict, num_frames=(2, 3, 1)):
        """
        Appends each row's target latents: after the first audio position, one latent position per frame. Rows without a
        voice get one, so every parameter takes part in the loss. The values are fixed: two calls prepare the same inputs.
        """
        generator = torch.Generator().manual_seed(0)
        if "audio_prompt_latents" not in inputs_dict:
            num_voice_frames = [self.num_prompt_frames] * self.batch_size
            input_ids, attention_mask = self._with_voice_placeholders(
                inputs_dict["input_ids"], inputs_dict["attention_mask"], num_voice_frames
            )
            inputs_dict = {
                **inputs_dict,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio_prompt_latents": torch.randn(
                    self.batch_size, self.num_prompt_frames, self.latent_dim, generator=generator
                ).to(torch_device),
            }
        rows = []
        for row, frames in enumerate(num_frames):
            ids = inputs_dict["input_ids"][row][inputs_dict["attention_mask"][row].bool()].cpu()
            rows.append(torch.cat([ids, torch.full((frames,), self.audio_latent_token_id)]))
        length = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), length), self.pad_token_id)
        attention_mask = torch.zeros((len(rows), length), dtype=torch.long)
        for row, ids in enumerate(rows):
            input_ids[row, length - len(ids) :] = ids
            attention_mask[row, length - len(ids) :] = 1
        labels = torch.randn(len(rows), max(num_frames), self.latent_dim, generator=generator)
        labels_mask = torch.zeros(len(rows), max(num_frames), dtype=torch.long)
        for row, frames in enumerate(num_frames):
            labels_mask[row, :frames] = 1
        return {
            **inputs_dict,
            "input_ids": input_ids.to(torch_device),
            "attention_mask": attention_mask.to(torch_device),
            "labels": labels.to(torch_device),
            "labels_mask": labels_mask.to(torch_device),
        }

    def create_and_check_training(self, config, inputs_dict):
        inputs_dict = self.with_training_targets(inputs_dict)
        config.latent_stats_ema_steps = 2
        model = PocketTTSForConditionalGeneration(config).to(torch_device).train()

        # the latent statistics follow the targets for `latent_stats_ema_steps` training forwards, then stay
        mean = model.latent_mean.clone()
        model(**inputs_dict)
        self.parent.assertFalse(torch.equal(model.latent_mean, mean))
        model(**inputs_dict)
        frozen = model.latent_mean.clone()
        model(**inputs_dict)
        torch.testing.assert_close(model.latent_mean, frozen)
        self.parent.assertEqual(model.latent_stats_num_updates.item(), 2)
        model.eval()
        model(**inputs_dict)
        self.parent.assertEqual(model.latent_stats_num_updates.item(), 2)

        # one self-distillation draw per training forward, identical for identically seeded models (every rank)
        model = PocketTTSForConditionalGeneration(config).to(torch_device).train()
        other = PocketTTSForConditionalGeneration(config).to(torch_device).train()
        other.load_state_dict(model.state_dict())
        for _ in range(4):
            torch.manual_seed(0)
            loss = model(**inputs_dict).loss
            torch.manual_seed(0)
            torch.testing.assert_close(other(**inputs_dict).loss, loss)
        self.parent.assertTrue(torch.equal(model.distill_generator.get_state(), other.distill_generator.get_state()))
        # outside training (e.g. evaluation, under `no_grad`), nothing is drawn
        state = model.distill_generator.get_state()
        model.eval()
        with torch.no_grad():
            model(**inputs_dict)
        self.parent.assertTrue(torch.equal(model.distill_generator.get_state(), state))

        # every parameter but the generation-only ones gets a gradient, the loss weighting included
        model.train()
        model.zero_grad()
        model(**inputs_dict).loss.backward()
        self.parent.assertIsNotNone(model.loss_weighting.proj.bias.grad)
        self.parent.assertIsNotNone(model.eos_head.weight.grad)
        self.parent.assertIsNotNone(model.model.audio_prompt_proj.weight.grad)

        # a row without voice (classifier-free guidance dropout) trains like the others
        voiceless = dict(inputs_dict)
        input_ids, attention_mask = voiceless["input_ids"].clone(), voiceless["attention_mask"].clone()
        voice_positions = input_ids[0] == self.audio_prompt_token_id
        input_ids[0, voice_positions], attention_mask[0, voice_positions] = self.pad_token_id, 0
        voice_mask = voiceless["audio_prompt_latents_mask"].clone()
        voice_mask[0] = 0
        voiceless.update(input_ids=input_ids, attention_mask=attention_mask, audio_prompt_latents_mask=voice_mask)
        self.parent.assertTrue(torch.isfinite(model(**voiceless).loss))

        # `labels` build the teacher-forced latents: `audio_latents` alongside is an error
        with self.parent.assertRaises(ValueError):
            model(**inputs_dict, audio_latents=torch.zeros(1, 1, self.latent_dim, device=torch_device))

    def create_and_check_voice_cache(self, config):
        """A voice given as the cache of its prefix generates what the same voice given as latents does."""
        model = PocketTTSForConditionalGeneration(config).to(torch_device).eval()
        _, text_ids, text_mask = self.prepare_config_and_inputs()
        voice = torch.randn(1, self.num_prompt_frames, self.latent_dim, device=torch_device).expand(
            self.batch_size, -1, -1
        )
        num_frames = [self.num_prompt_frames] * self.batch_size
        input_ids, attention_mask = self._with_voice_placeholders(text_ids, text_mask, num_frames)
        generate_kwargs = {
            "eos_threshold": 1e9,
            "do_sample": False,
            "max_new_tokens": 3,
            "return_dict_in_generate": True,
        }
        with torch.no_grad():
            from_latents = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, audio_prompt_latents=voice, **generate_kwargs
            )
            # the voice prefix run once, then prompts of the text only
            prefix_length = 1 + self.num_prompt_frames
            voice_cache = model(
                input_ids=input_ids[:, :prefix_length], audio_prompt_latents=voice, use_cache=True
            ).past_key_values
            from_cache = model.generate(
                input_ids=input_ids[:, prefix_length:],
                attention_mask=attention_mask[:, prefix_length:],
                past_key_values=voice_cache,
                **generate_kwargs,
            )
        torch.testing.assert_close(from_cache.audio_latents, from_latents.audio_latents, atol=1e-5, rtol=1e-5)

    def create_and_check_voice_prompt(self, config, inputs_dict):
        model = self.model_class(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(**inputs_dict)
        sequence_length = inputs_dict["input_ids"].shape[1]
        self.parent.assertEqual(outputs.logits.shape, (self.batch_size, sequence_length, 1))
        self.parent.assertEqual(outputs.last_hidden_state.shape, (self.batch_size, sequence_length, self.hidden_size))
        # a placeholder count that does not match the voice prompt is an error
        mask = inputs_dict["audio_prompt_latents_mask"].clone()
        mask[0, 0] = 0
        with self.parent.assertRaises(ValueError):
            model(**{**inputs_dict, "audio_prompt_latents_mask": mask})

    def create_and_check_generate(self, config, inputs_dict):
        model = self.model_class(config).to(torch_device).eval()
        # the generation config sets the defaults of the Pocket TTS-specific parameters
        model.generation_config.min_frames_before_eos = 0
        frames_after_eos = torch.tensor([1, 3, 1], device=torch_device)
        with torch.no_grad():
            # every frame is end of speech: the tail length decides the number of frames
            outputs = model.generate(
                **inputs_dict,
                frames_after_eos=frames_after_eos,
                eos_threshold=-1e9,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        num_frames = outputs.audio_latents_mask.sum(-1)
        self.parent.assertListEqual(num_frames.tolist(), (1 + frames_after_eos).tolist())
        self.parent.assertEqual(outputs.audio_latents.shape[1], 4)
        self.parent.assertEqual(outputs.sequences.shape[1], inputs_dict["input_ids"].shape[1] + 4)
        # the latents past a row's frames are zeroed
        padding = outputs.audio_latents_mask == 0
        self.parent.assertTrue((outputs.audio_latents[padding] == 0).all())
        with torch.no_grad():
            # without the dict, one tensor of latents per row, cut to its frames
            latents = model.generate(
                **inputs_dict, frames_after_eos=frames_after_eos, eos_threshold=-1e9, do_sample=False
            )
        self.parent.assertIsInstance(latents, list)
        for row, row_latents in enumerate(latents):
            torch.testing.assert_close(row_latents, outputs.audio_latents[row, : num_frames[row]])
        with torch.no_grad():
            # end of speech is ignored on the first `min_frames_before_eos` frames
            delayed = model.generate(
                **inputs_dict,
                frames_after_eos=frames_after_eos,
                eos_threshold=-1e9,
                min_frames_before_eos=2,
                do_sample=False,
                return_dict_in_generate=True,
            )
        self.parent.assertListEqual(delayed.audio_latents_mask.sum(-1).tolist(), (3 + frames_after_eos).tolist())
        with torch.no_grad():
            # no end of speech: `max_new_tokens` frames, with and without a cache, same latents
            cached = model.generate(
                **inputs_dict, eos_threshold=1e9, do_sample=False, max_new_tokens=3, return_dict_in_generate=True
            )
            uncached = model.generate(
                **inputs_dict,
                eos_threshold=1e9,
                do_sample=False,
                max_new_tokens=3,
                use_cache=False,
                return_dict_in_generate=True,
            )
        self.parent.assertListEqual(cached.audio_latents_mask.sum(-1).tolist(), [3, 3, 3])
        torch.testing.assert_close(cached.audio_latents, uncached.audio_latents, atol=1e-4, rtol=1e-4)
        # without a length, the original's budget for the longest text: `ceil((num_text_tokens / 3 + 2) * frame_rate)`
        frame_rate, model.config.frame_rate = model.config.frame_rate, 0.5
        placeholder_ids = torch.tensor(
            [config.audio_prompt_bos_token_id, config.audio_prompt_token_id, config.audio_bos_token_id],
            device=torch_device,
        )
        is_text = inputs_dict["attention_mask"].bool() & ~torch.isin(inputs_dict["input_ids"], placeholder_ids)
        budget = math.ceil((is_text.sum(-1).max().item() / 3 + 2) * 0.5)
        with torch.no_grad():
            budgeted = model.generate(**inputs_dict, eos_threshold=1e9, do_sample=False, return_dict_in_generate=True)
        self.parent.assertListEqual(budgeted.audio_latents_mask.sum(-1).tolist(), [budget] * self.batch_size)
        model.config.frame_rate = frame_rate
        # explicit position ids (what the mask implies) change nothing
        position_ids = (inputs_dict["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            with_positions = model.generate(
                **inputs_dict,
                position_ids=position_ids,
                eos_threshold=1e9,
                do_sample=False,
                max_new_tokens=3,
                return_dict_in_generate=True,
            )
        torch.testing.assert_close(with_positions.audio_latents, cached.audio_latents, atol=1e-5, rtol=1e-5)
        with torch.no_grad():
            two_steps = model.generate(
                **inputs_dict,
                eos_threshold=1e9,
                do_sample=False,
                max_new_tokens=2,
                num_flow_steps=2,
                return_dict_in_generate=True,
            )
        self.parent.assertEqual(two_steps.audio_latents.shape, (self.batch_size, 2, self.latent_dim))
        with torch.no_grad():
            # with `num_return_sequences`, every returned sequence keeps its own row's voice and end-of-speech tail
            expanded = model.generate(
                **inputs_dict,
                frames_after_eos=frames_after_eos,
                eos_threshold=-1e9,
                do_sample=True,
                num_return_sequences=2,
                return_dict_in_generate=True,
                output_scores=True,
            )
        self.parent.assertListEqual(
            expanded.audio_latents_mask.sum(-1).tolist(), (1 + frames_after_eos).repeat_interleave(2).tolist()
        )
        # the first end-of-speech logit only depends on the prompt, so it is the unexpanded row's
        torch.testing.assert_close(expanded.scores[0], outputs.scores[0].repeat_interleave(2))


@require_torch
class PocketTTSModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (PocketTTSModel, PocketTTSForConditionalGeneration) if is_torch_available() else ()
    _is_composite = True
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = PocketTTSModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PocketTTSConfig, has_text_modality=True)
        self.skip_unsupported_generate()

    def skip_unsupported_generate(self):
        # Pocket TTS generates continuous latents, not tokens: the returned `sequences` are placeholders and the `scores`
        # are end-of-speech logits, so the common generation tests that read token outputs, vocabulary-sized scores or
        # the generation strategies built on them (beam search, assisted, contrastive, ...) do not apply. The supported
        # paths (greedy / sampled, with and without cache, custom flow steps) are covered by `test_generate`.
        skippable_tests = [
            "test_assisted",
            "test_beam",
            "test_sample_generate",
            "test_greedy_generate",
            "test_generate_continue_from_past_key_values",
            "test_generate_from_random_inputs_embeds",
            "test_generate_from_inputs_embeds",
            "test_generate_methods_with_logits_to_keep",
            "test_model_parallel_beam_search",
            "test_prompt_lookup_decoding",
            "test_contrastive_generate",
            "test_dola_decoding",
            "test_generate_with_quant_cache",
            "test_generate_continue_from_inputs_embeds",
            "test_generate_with_and_without_position_ids",  # reads token sequences; `test_generate` checks position_ids
            "test_generate_compilation_all_outputs",  # checks vocabulary-sized scores
        ]
        for test in skippable_tests:
            if self._testMethodName.startswith(test):
                self.skipTest(
                    reason="Pocket TTS generates continuous latents; token-based generation tests do not apply."
                )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_voice_prompt(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_with_voice()
        self.model_tester.create_and_check_voice_prompt(*config_and_inputs)

    def test_generate(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_with_voice()
        self.model_tester.create_and_check_generate(*config_and_inputs)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if return_labels and model_class is PocketTTSForConditionalGeneration:
            inputs_dict = self.model_tester.with_training_targets(inputs_dict)
        return inputs_dict

    def test_model_outputs_equivalence(self):
        # the training loss draws noise: reseed before every forward, so the tuple and dict calls draw alike (VITS)
        forward = PocketTTSForConditionalGeneration.forward

        def seeded_forward(model, *args, **kwargs):
            torch.manual_seed(0)
            return forward(model, *args, **kwargs)

        with mock.patch.object(PocketTTSForConditionalGeneration, "forward", seeded_forward):
            super().test_model_outputs_equivalence()

    def test_training_loss(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_with_voice()
        self.model_tester.create_and_check_training(*config_and_inputs)

    def test_loss_function(self):
        model = PocketTTSForConditionalGeneration(self.model_tester.get_config())
        self.assertIs(model.loss_function, PocketTTSForConditionalGenerationLoss)

    def test_training_gradient_accumulation(self):
        # the loss is a per-batch mean: with gradient accumulation, the Trainer divides it by the number of micro-batches
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_with_voice()
        inputs_dict = self.model_tester.with_training_targets(inputs_dict)
        config.lsd_distill_probability = 1.0
        config.latent_stats_ema_steps = 0
        model = PocketTTSForConditionalGeneration(config).to(torch_device).train()

        def gradients(step):
            model.zero_grad()
            torch.manual_seed(0)
            step()
            return torch.cat(
                [parameter.grad.flatten() for parameter in model.parameters() if parameter.grad is not None]
            )

        expected = gradients(lambda: model(**inputs_dict).loss.backward())
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(model, TrainingArguments(tmp_dir, gradient_accumulation_steps=4, report_to=[]))
            trainer.current_gradient_accumulation_steps = 4  # set by the training loop
            num_items_in_batch = trainer._get_num_items_in_batch([inputs_dict] * 4, torch_device)
            accumulated = gradients(lambda: trainer.training_step(model, dict(inputs_dict), num_items_in_batch))
        torch.testing.assert_close(accumulated, expected / 4)

    def test_flow_layer_norm_matches_torch(self):
        # `PocketTTSFlowLayerNorm` only rewrites `nn.LayerNorm`'s forward (torch's fused one has wrong gradients
        # through a JVP): same parameters and outputs, so `nn.LayerNorm` replaces it as is once torch is fixed
        from transformers.models.pocket_tts.modeling_pocket_tts import PocketTTSFlowLayerNorm

        for elementwise_affine in (True, False):
            norm = PocketTTSFlowLayerNorm(16, eps=1e-6, elementwise_affine=elementwise_affine)
            reference = nn.LayerNorm(16, eps=1e-6, elementwise_affine=elementwise_affine)
            if elementwise_affine:
                with torch.no_grad():
                    norm.weight.uniform_(0.5, 1.5)
                    norm.bias.uniform_(-0.5, 0.5)
            reference.load_state_dict(norm.state_dict())
            self.assertListEqual(list(norm.state_dict()), list(reference.state_dict()))
            hidden_states = torch.randn(3, 5, 16)
            torch.testing.assert_close(norm(hidden_states), reference(hidden_states))

    def test_training_drifting_unsupported(self):
        # the LSD loss does not apply to the drifting head (no time conditioning, the output is the latent itself)
        self.model_tester.flow_type = "drifting"
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_with_voice()
        model = PocketTTSForConditionalGeneration(config).to(torch_device).train()
        with self.assertRaises(ValueError):
            model(**self.model_tester.with_training_targets(inputs_dict))

    def test_voice_cache(self):
        config = self.model_tester.get_config()
        self.model_tester.create_and_check_voice_cache(config)

    def test_generate_drifting(self):
        self.model_tester.flow_type = "drifting"
        config_and_inputs = self.model_tester.prepare_config_and_inputs_with_voice()
        self.model_tester.create_and_check_generate(*config_and_inputs)


@require_torch
class PocketTTSIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    """
    reproducer (uploads the golden to
    ``hf://buckets/hf-internal-testing/pocket-tts-integration-tests/<case>/expected.safetensors``, holding the original's
    waveforms, per-frame latents and end-of-speech logits, voice latents or training loss and gradients, with the frame
    counts, token ids and texts in ``<case>/metadata.json``):
        https://gist.github.com/eustlb/cf6dd7b0159c162330d435550455d609
    """

    # Tolerances against the goldens from the bucket, measured on the CI A10G (2 runs, identical):
    RTOL = 0
    # teacher-forced steps, per-frame latents and end-of-speech logits: up to 2.2e-5 (drifting)
    STEP_ATOL = 5e-5
    # free-running waveforms, op-order noise amplified over the generated frames: up to 2.5e-4 on the A10G and 7.1e-4
    # on an A100 against the same goldens, so the margin covers a GPU or CI image change
    AUDIO_ATOL = 1e-3
    # free-running sampled latents (CPU), in the flow head's normalised space: up to 1.2e-4
    SAMPLED_ATOL = 5e-4
    # the audio tokenizer alone, decoded waveform and voice latents: up to 1.1e-5
    CODEC_ATOL = 5e-5
    # training loss (0.0) and gradients (up to 3.1e-7, on gradients up to 0.12)
    TRAINING_ATOL = 1e-6

    # --- single source of truth, mirrored from the reproducers' common.py ---
    CASES = [
        ("Hello world. I am Kyutai's Pocket TTS. I'm fast enough to run on small CPUs.", "alba"),
        ("Hello world", "jean"),
    ]
    SAMPLED_TEMPERATURE = 0.3
    SAMPLED_SEED = 0
    VOICE_CLONING_AUDIO = "en-Alice_woman.wav"
    VOICE_CLONING_BATCH_AUDIO = ["en-Alice_woman.wav", "en-Carter_man.wav"]

    checkpoint_name = "eustlb/pocket-tts"
    drifting_checkpoint_name = "eustlb/pocket-tts-english-drifting"
    bucket = "hf-internal-testing/pocket-tts-integration-tests"

    def setUp(self):
        super().setUp()
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_name)

    def _load_voices(self, model, names, checkpoint_name=None, inputs=None):
        """
        The checkpoint's shipped voices (the cache of their voice prefix), one per row, as one `DynamicCache`. Voices of
        different lengths are left-padded to the longest; `inputs["attention_mask"]` then gets their mask prepended.
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        states = [
            load_file(hf_hub_download(checkpoint_name or self.checkpoint_name, f"voices/{name}.safetensors"))
            for name in names
        ]
        lengths = [state["layers.0.keys"].shape[2] for state in states]
        length = max(lengths)
        layers = [
            tuple(
                torch.cat(
                    [
                        torch.nn.functional.pad(state[f"layers.{i}.{kind}"], (0, 0, length - state_length, 0))
                        for state, state_length in zip(states, lengths)
                    ]
                ).to(model.device)
                for kind in ("keys", "values")
            )
            for i in range(model.config.num_hidden_layers)
        ]
        if inputs is not None:
            voice_mask = torch.tensor([[0] * (length - n) + [1] * n for n in lengths])
            inputs["attention_mask"] = torch.cat(
                [voice_mask.to(inputs["attention_mask"]), inputs["attention_mask"]], -1
            )
        elif len(set(lengths)) > 1:
            raise ValueError("Voices of different lengths need `inputs`, to mask the padding of the shorter ones.")
        return DynamicCache(layers, config=model.config)

    def _load_expected(self, case):
        with tempfile.TemporaryDirectory() as tmp_dir:
            files = [
                (f"{case}/{name}", os.path.join(tmp_dir, name)) for name in ("expected.safetensors", "metadata.json")
            ]
            download_bucket_files(self.bucket, files=files)
            with open(files[1][1], encoding="utf-8") as f:
                return load_file(files[0][1]), json.load(f)

    def _load_voice_audio(self, filename):
        """The original's reading of a voice recording: float32, mono, resampled to 24 kHz with `resample_poly`."""
        import soundfile as sf
        from huggingface_hub import hf_hub_download
        from scipy.signal import resample_poly

        path = hf_hub_download("hf-internal-testing/dummy-audio-samples", filename, repo_type="dataset")
        audio, sampling_rate = sf.read(path, dtype="float32")
        if sampling_rate != 24000:
            audio = resample_poly(audio, 24000 // 8000, sampling_rate // 8000).astype("float32")
        return audio

    def _prepare(self, processor, texts, voices=None, audio=None):
        """
        One conversation per text, the voice given as recordings (in the conversations), as precomputed latents, or
        neither: the text only, the voice coming as a cache (`past_key_values`).
        """
        conversations = []
        for row, text in enumerate(texts):
            content = [{"type": "text", "text": text}]
            if audio is not None:
                content.insert(0, {"type": "audio", "audio": audio[row]})
            elif voices is not None:
                content.insert(0, {"type": "audio"})
            conversations.append([{"role": "user", "content": content}])
        return processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            processor_kwargs={"audio_prompt_latents": voices} if voices is not None else {},
        )

    def _generate(self, model, inputs, **kwargs):
        with torch.no_grad():
            return model.generate(
                **inputs.to(model.device), return_dict_in_generate=True, output_scores=True, **kwargs
            )

    def _teacher_forced(self, model, inputs, expected_latents, voice_cache=None):
        """
        Every generation step at once, fed the original's latents: the per-step latents and end-of-speech logits, free
        of the autoregressive amplification that makes long free-running generations drift apart by op-order noise.
        """
        num_frames = expected_latents.shape[0]
        inputs = inputs.to(model.device)
        # the first audio position (the BOS latent) closes the prompt; every later one holds the previous frame
        placeholders = torch.full((1, num_frames - 1), model.config.audio_latent_token_id, device=model.device)
        input_ids = torch.cat([inputs["input_ids"], placeholders], dim=-1)
        audio_latents = expected_latents[None, :-1].to(model.device)
        if voice_cache is not None:
            voice = {"past_key_values": voice_cache}
            attention_mask = torch.ones(1, voice_cache.get_seq_length() + input_ids.shape[1], device=model.device)
        else:
            voice = {key: inputs[key] for key in ("audio_prompt_latents", "audio_prompt_latents_mask")}
            attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, audio_latents=audio_latents, **voice)
            noise = torch.zeros(num_frames, model.config.latent_dim, device=model.device)
            latents = model.sample_latent(outputs.last_hidden_state[0, -num_frames:], noise, num_flow_steps=1)
        return latents.cpu(), outputs.logits[0, -num_frames:, 0].cpu()

    @slow
    def test_single(self):
        """
        CASES[0] with the shipped voice, temperature 0.

        reproducer: https://gist.github.com/eustlb/cf6dd7b0159c162330d435550455d609#file-reproducer-single-py
        """
        expected, metadata = self._load_expected("single")
        model = AutoModelForTextToWaveform.from_pretrained(self.checkpoint_name, device_map=torch_device)
        text, voice = self.CASES[0]
        # the shipped voice is the cache of its prefix: the prompt is the text only
        inputs = self._prepare(self.processor, [text])
        output = self._generate(model, inputs, past_key_values=self._load_voices(model, [voice]), do_sample=False)

        self.assertEqual(output.audio_latents_mask.sum().item(), metadata["sanity"]["num_frames"])
        latents, eos_logits = self._teacher_forced(
            model, inputs, expected["audio_latents"], voice_cache=self._load_voices(model, [voice])
        )
        torch.testing.assert_close(latents, expected["audio_latents"], rtol=self.RTOL, atol=self.STEP_ATOL)
        torch.testing.assert_close(eos_logits, expected["eos_logits"], rtol=self.RTOL, atol=self.STEP_ATOL)
        audio = self.processor.decode(output.audio_latents, audio_latents_mask=output.audio_latents_mask)
        torch.testing.assert_close(audio[0], expected["audio"], rtol=self.RTOL, atol=self.AUDIO_ATOL)

    @slow
    def test_batch(self):
        """
        Both CASES in one padded batch, each row against its own reference, temperature 0.

        reproducer: https://gist.github.com/eustlb/cf6dd7b0159c162330d435550455d609#file-reproducer-batch-py
        """
        expected, metadata = self._load_expected("batch")
        model = AutoModelForTextToWaveform.from_pretrained(self.checkpoint_name, device_map=torch_device)
        texts, voices = zip(*self.CASES)
        # one voice cache per row, the shorter voice left-padded, and the shorter text left-padded after the cached voice:
        # the positions follow the mask, which covers both
        inputs = self._prepare(self.processor, texts)
        voice_cache = self._load_voices(model, voices, inputs=inputs)
        output = self._generate(model, inputs, past_key_values=voice_cache, do_sample=False)

        self.assertListEqual(
            output.audio_latents_mask.sum(-1).tolist(), [row["sanity"]["num_frames"] for row in metadata["rows"]]
        )
        audio = self.processor.decode(output.audio_latents, audio_latents_mask=output.audio_latents_mask)
        torch.testing.assert_close(audio[0], expected["audio_0"], rtol=self.RTOL, atol=self.AUDIO_ATOL)
        torch.testing.assert_close(audio[1], expected["audio_1"], rtol=self.RTOL, atol=self.AUDIO_ATOL)

    @slow
    def test_sampled(self):
        """
        CASES[0] at temperature 0.3, the original's noise replaced by the draw transformers makes. The noise is `torch.randn` on CPU under `torch.manual_seed`, so the model runs on CPU here: a GPU generator draws a
        different sequence.
        """
        expected, metadata = self._load_expected("sampled")
        model = AutoModelForTextToWaveform.from_pretrained(self.checkpoint_name, device_map="cpu")
        text, voice = self.CASES[0]
        inputs = self._prepare(self.processor, [text])
        voice_cache = self._load_voices(model, [voice])
        torch.manual_seed(self.SAMPLED_SEED)
        output = self._generate(
            model, inputs, past_key_values=voice_cache, do_sample=True, temperature=self.SAMPLED_TEMPERATURE
        )

        self.assertEqual(output.audio_latents_mask.sum().item(), metadata["sanity"]["num_frames"])
        # compared in the flow head's normalised space (the original's)
        latents = (output.audio_latents[0] - model.latent_mean) / model.latent_std
        torch.testing.assert_close(latents.cpu(), expected["audio_latents"], rtol=self.RTOL, atol=self.SAMPLED_ATOL)

    @slow
    def test_drifting(self):
        """
        CASES[0] on english_drifting_26-09, temperature 0. The one-step drifting head amplifies op-order noise much more than LSD over a free-running generation (the
        latents part by 0.5 around frame 50 while the frame count stays the same), so the step-level check is teacher
        forced.

        reproducer: https://gist.github.com/eustlb/cf6dd7b0159c162330d435550455d609#file-reproducer-drifting-py
        """
        expected, metadata = self._load_expected("drifting")
        model = AutoModelForTextToWaveform.from_pretrained(self.drifting_checkpoint_name, device_map=torch_device)
        processor = AutoProcessor.from_pretrained(self.drifting_checkpoint_name)
        text, voice = self.CASES[0]
        inputs = self._prepare(processor, [text])
        voices = (model, [voice], self.drifting_checkpoint_name)
        output = self._generate(model, inputs, past_key_values=self._load_voices(*voices), do_sample=False)

        self.assertEqual(output.audio_latents_mask.sum().item(), metadata["sanity"]["num_frames"])
        latents, eos_logits = self._teacher_forced(
            model, inputs, expected["audio_latents"], voice_cache=self._load_voices(*voices)
        )
        torch.testing.assert_close(latents, expected["audio_latents"], rtol=self.RTOL, atol=self.STEP_ATOL)
        torch.testing.assert_close(eos_logits, expected["eos_logits"], rtol=self.RTOL, atol=self.STEP_ATOL)

    @slow
    def test_voice_cloning(self):
        """
        CASES[0] with the voice read from a recording, temperature 0. The original reads the 16 kHz file and resamples it to 24 kHz with `scipy.signal.resample_poly`, done the same
        way here before the processor.

        reproducer: https://gist.github.com/eustlb/cf6dd7b0159c162330d435550455d609#file-reproducer-voice-cloning-py
        """
        expected, metadata = self._load_expected("voice_cloning")
        model = AutoModelForTextToWaveform.from_pretrained(self.checkpoint_name, device_map=torch_device)
        audio = self._load_voice_audio(self.VOICE_CLONING_AUDIO)
        text, _ = self.CASES[0]
        inputs = self._prepare(self.processor, [text], audio=[audio])
        output = self._generate(model, inputs, do_sample=False)

        self.assertEqual(output.audio_latents_mask.sum().item(), metadata["sanity"]["num_frames"])
        # the feature extractor pads to the audio tokenizer's frames
        self.assertEqual(self.processor.feature_extractor.frame_size, self.processor.audio_tokenizer.config.frame_size)
        # voice latents (end-on-pause, the processor's audio tokenizer)
        torch.testing.assert_close(
            inputs["audio_prompt_latents"][0].cpu(),
            expected["audio_prompt_latents"],
            rtol=self.RTOL,
            atol=self.CODEC_ATOL,
        )
        latents, eos_logits = self._teacher_forced(model, inputs, expected["audio_latents"])
        torch.testing.assert_close(latents, expected["audio_latents"], rtol=self.RTOL, atol=self.STEP_ATOL)
        torch.testing.assert_close(eos_logits, expected["eos_logits"], rtol=self.RTOL, atol=self.STEP_ATOL)

    @slow
    def test_decode(self):
        """
        The original's latents of `test_single`, decoded with the 5 ms fade-in.

        reproducer: https://gist.github.com/eustlb/cf6dd7b0159c162330d435550455d609#file-reproducer-single-py
        """
        expected, _ = self._load_expected("single")
        model = AutoModelForTextToWaveform.from_pretrained(self.checkpoint_name, device_map=torch_device)
        # the original's latents are the flow head's (normalised); the audio tokenizer reads the raw ones
        latents = model.denormalize_latents(expected["audio_latents"].to(model.device))
        audio = self.processor.decode(latents)
        torch.testing.assert_close(audio, expected["audio"], rtol=self.RTOL, atol=self.CODEC_ATOL)

    @slow
    def test_voice_cloning_batch(self):
        """
        Two recordings of different lengths (9.3 s and 27.7 s) in one padded batch: each row's voice latents and first
        end-of-speech logit (which only depends on the prompt) match its own reference, and so does its number of frames.

        reproducer: https://gist.github.com/eustlb/cf6dd7b0159c162330d435550455d609#file-reproducer-voice-cloning-batch-py
        """
        expected, metadata = self._load_expected("voice_cloning_batch")
        model = AutoModelForTextToWaveform.from_pretrained(self.checkpoint_name, device_map=torch_device)
        audio = [self._load_voice_audio(filename) for filename in self.VOICE_CLONING_BATCH_AUDIO]
        text, _ = self.CASES[0]
        inputs = self._prepare(self.processor, [text, text], audio=audio)
        output = self._generate(model, inputs, do_sample=False)

        self.assertListEqual(
            output.audio_latents_mask.sum(-1).tolist(), [row["sanity"]["num_frames"] for row in metadata["rows"]]
        )
        voice_latents = inputs["audio_prompt_latents"].cpu()
        for row, row_metadata in enumerate(metadata["rows"]):
            num_frames = row_metadata["num_voice_frames"]
            # padding must not change a row's voice latents
            torch.testing.assert_close(
                voice_latents[row, :num_frames],
                expected[f"audio_prompt_latents_{row}"],
                rtol=self.RTOL,
                atol=self.CODEC_ATOL,
            )
            torch.testing.assert_close(
                output.scores[0][row].cpu(), expected[f"eos_logits_{row}"][0], rtol=self.RTOL, atol=self.STEP_ATOL
            )

    @slow
    @with_grad
    def test_training(self):
        """
        Kyutai's training code (`training/`): the LSD finetune loss on two LibriSpeech samples, voice prompt + target +
        transcript, self-distillation term at every step, no CFG dropout, frozen latent statistics. The original's inputs (its voice and target latents, and its tokenisation of the transcripts, checked) go through
        the processor's training conversations; the loss and the gradients of the heads, the input projections and a
        backbone layer match. CPU: the noise comes from the CPU generator, after the two draws the original spends on its
        CFG decisions.

        reproducer: https://gist.github.com/eustlb/cf6dd7b0159c162330d435550455d609#file-reproducer-training-py
        """
        expected, metadata = self._load_expected("training")
        model = AutoModelForTextToWaveform.from_pretrained(
            self.checkpoint_name, lsd_distill_probability=1.0, latent_stats_ema_steps=0, device_map="cpu"
        )
        model.train()
        conversations = [
            [
                {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": text}]},
                {"role": "assistant", "content": [{"type": "audio"}]},
            ]
            for text in metadata["texts"]
        ]
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            processor_kwargs={
                "audio_prompt_latents": [expected["voice_latents_0"], expected["voice_latents_1"]],
                "labels": [expected["target_latents_0"], expected["target_latents_1"]],
            },
        )
        placeholder_ids = {
            model.config.audio_prompt_token_id,
            model.config.audio_latent_token_id,
            model.config.audio_prompt_bos_token_id,
            model.config.audio_bos_token_id,
        }
        for row, token_ids in enumerate(metadata["token_ids"]):
            ids = inputs["input_ids"][row][inputs["attention_mask"][row].bool()].tolist()
            self.assertListEqual([i for i in ids if i not in placeholder_ids], token_ids)

        torch.manual_seed(metadata["seed"])
        torch.rand(len(conversations)), torch.rand(len(conversations))
        loss = model(**inputs).loss
        loss.backward()
        torch.testing.assert_close(loss.detach(), expected["loss"], rtol=self.RTOL, atol=self.TRAINING_ATOL)
        parameters = dict(model.named_parameters())
        for name in [key for key in expected if key.startswith("grad.")]:
            torch.testing.assert_close(
                parameters[name[len("grad.") :]].grad, expected[name], rtol=self.RTOL, atol=self.TRAINING_ATOL
            )
