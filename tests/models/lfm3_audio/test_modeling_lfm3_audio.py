# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch LFM3-Audio model."""

import copy
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pytest

from transformers import GenerationConfig, Lfm3AudioConfig, LogitsProcessor, LogitsProcessorList, is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_accelerate,
    require_torch,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_modeling_common import ModelTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        AutoConfig,
        AutoProcessor,
        GenerationConfig,
        Lfm3AudioForConditionalGeneration,
        Lfm3AudioModel,
    )
    from transformers.generation.streamers import BaseStreamer
    from transformers.models.lfm3_audio.modeling_lfm3_audio import Lfm3AudioCodebookNorm


AUDIO_TOKEN_ID, AUDIO_OUT_TOKEN_ID, AUDIO_START_ID, TEXT_END_ID, MIXED_START_ID = 250, 251, 252, 253, 254
NUM_CODEBOOKS, CODEBOOK_SIZE = 4, 7


def tiny_config(**kwargs):
    text_config = {
        "model_type": "lfm3",
        "vocab_size": 256,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "layer_types": ["conv", "conv", "full_attention", "sliding_attention"],
        "sliding_window": 8,
        "conv_L_cache": 3,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    audio_encoder_config = {
        "num_mel_bins": 16,
        "conv_channels": 8,
        "hidden_size": 16,
        "sliding_window": 4,
        "text_config": {
            "model_type": "lfm3",
            "vocab_size": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "layer_types": ["conv", "sliding_attention", "conv"],
            "sliding_window": 4,
            "conv_L_cache": 3,
            "rope_parameters": {"sliding_attention": {"rope_type": "default", "rope_theta": 1e6}},
            "use_sandwich_norm": False,
            "use_attention_gate": False,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": None,
        },
    }
    depth_config = {
        "model_type": "lfm3",
        "vocab_size": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": NUM_CODEBOOKS,
        "layer_types": ["full_attention"] * 2,
        "rope_parameters": {
            "full_attention": {"rope_type": "default", "rope_theta": 1e6, "partial_rotary_factor": 1.0}
        },
        "sliding_window": None,
        "use_sandwich_norm": False,
    }
    values = {
        "text_config": text_config,
        "audio_encoder_config": audio_encoder_config,
        "depth_config": depth_config,
        "num_codebooks": NUM_CODEBOOKS,
        "codebook_size": CODEBOOK_SIZE,
        "audio_eos_code": CODEBOOK_SIZE - 1,
        "audio_token_id": AUDIO_TOKEN_ID,
        "audio_out_token_id": AUDIO_OUT_TOKEN_ID,
        "audio_start_token_id": AUDIO_START_ID,
        "text_end_token_id": TEXT_END_ID,
        "mixed_start_token_id": MIXED_START_ID,
    }
    return Lfm3AudioConfig(**{**values, **kwargs})


def tiny_inputs(device):
    # two clips of 40 and 24 spectrogram frames -> 5 and 3 encoder tokens, plus two prompt speech frames
    input_features = torch.randn(2, 16, 40, device=device)
    feature_attention_mask = torch.ones(2, 40, dtype=torch.long, device=device)
    feature_attention_mask[1, 24:] = 0
    input_ids = torch.tensor(
        [[1, 10, 11] + [AUDIO_TOKEN_ID] * 5 + [12, 13] + [AUDIO_TOKEN_ID] * 3 + [AUDIO_OUT_TOKEN_ID] * 2 + [14]],
        device=device,
    )
    audio_out_codes = torch.randint(0, CODEBOOK_SIZE - 1, (2, NUM_CODEBOOKS), device=device)
    return {
        "input_ids": input_ids,
        "input_features": input_features,
        "feature_attention_mask": feature_attention_mask,
        "audio_out_codes": audio_out_codes,
    }


class Boost(LogitsProcessor):
    """Adds `boost` to the score of the token `choose(step, row)` names, so suppressed (`-inf`) tokens stay suppressed."""

    def __init__(self, choose, boost=1e4):
        self.choose, self.boost, self.step = choose, boost, 0

    def __call__(self, input_ids, scores):
        for row in range(scores.shape[0]):
            token = self.choose(self.step, row)
            if token is not None:
                scores[row, token] += self.boost
        self.step += 1
        return scores


def force_tokens(*tokens):
    """A `logits_processor` list that forces every row through `tokens` in order, then leaves the model alone."""
    return LogitsProcessorList([Boost(lambda step, row: tokens[step] if step < len(tokens) else None)])


def frames_of(code, num_rows=None):
    """A `sample_audio_frames` stand-in emitting frames filled with `code`."""
    return lambda hidden_states, *args, **kwargs: torch.full(
        (hidden_states.shape[0], NUM_CODEBOOKS), code, dtype=torch.long, device=hidden_states.device
    )


class Collect(BaseStreamer):
    def __init__(self):
        self.items, self.ended = [], False

    def put(self, value):
        self.items.append(value.clone())

    def end(self):
        self.ended = True


class Lfm3AudioBehaviorTestMixin:
    def test_generate_replays_terminal_audio_in_text_mode(self):
        model, config = self.model, self.model.config
        ids = torch.tensor([[1, config.audio_start_token_id]], device=torch_device)
        with patch.object(model, "sample_audio_frames", side_effect=frames_of(config.audio_eos_code)) as sampler:
            full = model.generate(ids, max_new_tokens=2, logits_processor=force_tokens(None, 10))
            prefix = model.generate(ids, max_new_tokens=1)
            sampler.reset_mock()
            # the terminal frame in the prompt resumes the text phase: no frame is sampled
            replay = model.generate(
                prefix.sequences,
                audio_out_codes=prefix.audio_codes[0].T,
                max_new_tokens=1,
                logits_processor=force_tokens(10),
            )
            torch.testing.assert_close(replay.sequences, full.sequences)
            sampler.assert_not_called()
            # different frame counts per row exercise the flattened prompt-code mapping
            batched_ids = torch.tensor(
                [
                    [1, config.audio_out_token_id, 11, config.audio_out_token_id],
                    [1, 10, 11, config.audio_out_token_id],
                ],
                device=torch_device,
            )
            eos_frame = torch.full((config.num_codebooks,), config.audio_eos_code, device=torch_device)
            normal = torch.ones_like(eos_frame)
            codes = torch.stack([normal, eos_frame, normal])
            batched = model.generate(
                batched_ids, audio_out_codes=codes, max_new_tokens=1, logits_processor=force_tokens(10)
            )
            self.assertEqual(batched.sequences[:, -1].tolist(), [10, config.audio_out_token_id])
            self.assertEqual(sampler.call_args[0][0].shape[0], 1)
            sampler.reset_mock()
            # an explicit mode still wins over the replayed terminal frame
            model.generate(
                prefix.sequences, audio_out_codes=prefix.audio_codes[0].T, generation_mode="audio", max_new_tokens=1
            )
            sampler.assert_called_once()

    def test_audio_loss_pairs_preceding_states_with_supervised_frames(self):
        model, config = self.model, self.model.config
        token = config.audio_out_token_id
        ids = torch.tensor([[token, 10, token, 11, token, 12], [1, token, 10, token, 11, token]], device=torch_device)
        hidden = torch.arange(12.0, device=torch_device).reshape(2, 6, 1).expand(2, 6, config.text_config.hidden_size)
        labels = torch.tensor(
            [
                [0] * config.num_codebooks,
                [1] * config.num_codebooks,
                [-100] * config.num_codebooks,
                [2] * config.num_codebooks,
                [3] * config.num_codebooks,
                [4] * config.num_codebooks,
            ],
            device=torch_device,
        )
        seen = []

        class Spy(torch.nn.Module):
            def forward(self, latents, codes, masked):
                seen.append((latents[:, 0].tolist(), codes.clone()))
                return hidden.new_zeros(codes.shape[0], config.num_codebooks, config.codebook_size)

        with patch.object(model.model, "depth_decoder", Spy()):
            model.audio_loss(ids, hidden, labels)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], [1.0, 6.0, 8.0, 10.0])
        torch.testing.assert_close(seen[0][1], labels[[1, 3, 4, 5]])

    @pytest.mark.torch_compile_test
    def test_text_only_forward_compiles_fullgraph(self):
        # the speech-placeholder check is data-dependent, so a compiled forward skips it
        ids = torch.tensor([[1, 10, 11, 12, 13, 14]], device=torch_device)
        torch._dynamo.reset()
        with torch.no_grad():
            expected = self.model(input_ids=ids).logits
            compiled = torch.compile(self.model, fullgraph=True)(input_ids=ids).logits
        torch._dynamo.reset()
        torch.testing.assert_close(compiled, expected, rtol=1e-4, atol=1e-4)

    @require_torch_accelerator
    def test_forwards_are_captured_by_a_cuda_graph(self):
        model, config = self.model, self.model.config
        ids = torch.tensor([[1, 10, 11, 12, 13, 14]], device=torch_device)
        speech_ids = torch.tensor(
            [[1, 10, config.audio_out_token_id, config.audio_out_token_id, 11]], device=torch_device
        )
        codes = torch.randint(0, config.codebook_size - 1, (2, config.num_codebooks), device=torch_device)
        with torch.no_grad():
            embeds = model.get_input_embeddings()(ids)
            speech_embeds = model.get_input_embeddings()(speech_ids)
        # the data-dependent placeholder checks skip under capture, each on its own
        cases = {
            "text ids": lambda: model(input_ids=ids, use_cache=False).logits,
            "text embeds": lambda: model(inputs_embeds=embeds, use_cache=False).logits,
            "speech frames": lambda: model(input_ids=speech_ids, audio_out_codes=codes, use_cache=False).logits,
            "codebook embeddings": lambda: model.model.get_audio_out_embeds(codes),
            "placeholder mask": lambda: model.model.get_placeholder_mask(
                speech_ids, speech_embeds, config.audio_out_token_id, 2
            ),
        }
        for name, run in cases.items():
            with self.subTest(name), torch.no_grad():
                expected = run()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run()
                graph.replay()
                torch.testing.assert_close(captured, expected, rtol=1e-5, atol=1e-5)
        # eagerly, a placeholder without its codes is still refused
        with self.assertRaisesRegex(ValueError, "audio_out_codes"):
            model(input_ids=speech_ids, use_cache=False)

    def _split_audio_loss(self, second_device):
        from accelerate.hooks import attach_execution_device_hook

        inputs = self.inputs
        labels = inputs["input_ids"].masked_fill(inputs["input_ids"] >= 248, -100)
        rand = torch.rand

        def same_draws(*size, device=None, **kwargs):
            # the frame masks are drawn on the hidden states' device: the same draws on every placement
            return rand(*size, generator=torch.Generator().manual_seed(0), **kwargs).to(device)

        def losses(model):
            with torch.no_grad(), patch("torch.rand", same_draws):
                output = model(**inputs, labels=labels, audio_labels=inputs["audio_out_codes"])
            return [value.to("cpu") for value in (output.loss, output.text_loss, output.audio_loss)]

        expected = losses(self.model)
        model = copy.deepcopy(self.model)
        language_model = model.model.language_model
        # the last decoder layer, the final norm and the depth decoder on a second device, as a balanced `device_map`
        # places them; the head stays with the embedding table it is tied to
        for module in (language_model.layers[-1], language_model.embedding_norm, model.model.depth_decoder):
            module.to(second_device)
            attach_execution_device_hook(module, second_device)
        attach_execution_device_hook(model.lm_head, torch_device)
        actual = losses(model)
        for value, reference in zip(actual, expected):
            self.assertTrue(torch.isfinite(value))
            torch.testing.assert_close(value, reference, rtol=1e-4, atol=1e-5)

    @require_torch_accelerator
    @require_accelerate
    def test_audio_loss_with_the_last_layers_on_another_device(self):
        self._split_audio_loss("cpu")

    @require_torch_multi_accelerator
    @require_accelerate
    def test_audio_loss_with_the_last_layers_on_a_second_accelerator(self):
        # two accelerators also put the 0-d losses and counts of `combined_loss` on different devices
        self._split_audio_loss(f"{torch.device(torch_device).type}:1")

    def test_live_transcription_refuses_suppressed_ids_outside_the_vocabulary(self):
        model, text_config = self.model, self.model.config.text_config
        prompt = torch.tensor([[1, 10, 11]], device=torch_device)
        features = torch.randn(1, 16, 96, device=torch_device)
        state = model.transcribe_live(features, input_ids=prompt, max_new_tokens=2).state
        encoder_cache, cache = state.encoder_cache, state.past_key_values
        lengths = (encoder_cache.num_tokens, cache.get_seq_length())
        with self.assertRaisesRegex(ValueError, "Suppressed token ids"):
            model.transcribe_live(features, state, suppress_token_ids=[text_config.vocab_size])
        eos_token_id = text_config.eos_token_id
        text_config.eos_token_id = text_config.vocab_size
        try:
            with self.assertRaisesRegex(ValueError, "Suppressed token ids"):
                model.transcribe_live(features, state, input_ids=prompt)
        finally:
            text_config.eos_token_id = eos_token_id
        # refused before the encoder advanced or a restart replaced the context
        self.assertIs(state.encoder_cache, encoder_cache)
        self.assertIs(state.past_key_values, cache)
        self.assertEqual((encoder_cache.num_tokens, cache.get_seq_length()), lengths)


@require_torch
class Lfm3AudioModelTest(Lfm3AudioBehaviorTestMixin, unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = tiny_config()
        self.model = Lfm3AudioForConditionalGeneration(self.config).to(torch_device).eval()
        self.inputs = tiny_inputs(torch_device)
        # the released checkpoints ask for the output object, which carries the speech frames
        self.model.generation_config.return_dict_in_generate = True

    def test_forward_merges_speech_features_at_the_placeholders(self):
        with torch.no_grad():
            outputs = self.model(**self.inputs)
        self.assertEqual(outputs.logits.shape, (1, self.inputs["input_ids"].shape[1], 256))
        self.assertEqual(outputs.audio_hidden_states.shape, (8, 32))
        # the audio placeholders no longer carry the placeholder embedding
        with torch.no_grad():
            text_only = self.model.model.get_input_embeddings()(self.inputs["input_ids"])
            merged = self.model.model(**self.inputs, output_hidden_states=True).hidden_states[0]
        placeholder = self.inputs["input_ids"][0] == AUDIO_TOKEN_ID
        self.assertFalse(torch.allclose(merged[0][placeholder], text_only[0][placeholder]))
        self.assertTrue(
            torch.equal(
                merged[0][~placeholder & (self.inputs["input_ids"][0] != AUDIO_OUT_TOKEN_ID)],
                text_only[0][~placeholder & (self.inputs["input_ids"][0] != AUDIO_OUT_TOKEN_ID)],
            )
        )

    def test_placeholder_count_mismatch_raises(self):
        inputs = dict(self.inputs)
        inputs["input_ids"] = inputs["input_ids"][:, :-4]
        with self.assertRaises(ValueError):
            self.model(**inputs)

    def test_encoder_window_bounds_the_attention(self):
        encoder = self.model.model.audio_encoder
        features, lengths = self.inputs["input_features"], self.inputs["feature_attention_mask"].sum(-1)
        window = encoder.lfm.config.sliding_window
        with torch.no_grad():
            windowed = encoder(features, lengths)[0].last_hidden_state
            encoder.lfm.config.sliding_window = 10_000
            unbounded = encoder(features, lengths)[0].last_hidden_state
            encoder.lfm.config.sliding_window = window
        # the first frames see the same past either way (up to kernel noise), the fifth frame loses the first one
        early = (windowed[:, :4] - unbounded[:, :4]).abs().max().item()
        late = (windowed[:, 4:] - unbounded[:, 4:]).abs().max().item()
        self.assertLess(early, 1e-6)
        self.assertGreater(late, 10 * max(early, 1e-9))

    def test_encoder_output_lengths(self):
        lengths = torch.tensor([1, 8, 9, 40])
        self.assertEqual(self.model.model.audio_encoder.downsample.output_lengths(lengths).tolist(), [1, 1, 2, 5])

    def test_depth_decoder_is_bidirectional_and_scores_masked_codebooks(self):
        depth = self.model.model.depth_decoder
        hidden = torch.randn(3, 32, device=torch_device)
        codes = torch.randint(0, CODEBOOK_SIZE - 1, (3, NUM_CODEBOOKS), device=torch_device)
        masked = torch.tensor([[True, False, False, False]] * 3, device=torch_device)
        with torch.no_grad():
            logits = depth(hidden, codes, masked)
            codes_changed = codes.clone()
            codes_changed[:, -1] = (codes_changed[:, -1] + 1) % (CODEBOOK_SIZE - 1)
            logits_changed = depth(hidden, codes_changed, masked)
        self.assertEqual(logits.shape, (3, NUM_CODEBOOKS, CODEBOOK_SIZE))
        # the first (masked) codebook attends to the last one
        self.assertFalse(torch.allclose(logits[:, 0], logits_changed[:, 0]))

    def test_codebook_norm_scales_in_fp32_and_rounds_once(self):
        # The heads were trained with fp32 normalize * weight and one cast; weight * bf16(x) is off by ulps.
        norm = Lfm3AudioCodebookNorm(tiny_config())
        torch.manual_seed(0)
        norm.weight.data.uniform_(0.5, 1.5)
        x = torch.randn(64, *norm.weight.shape).to(torch.bfloat16)
        norm = norm.to(torch.bfloat16)
        x32 = x.float()
        expected = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon) * norm.weight).to(
            torch.bfloat16
        )
        self.assertTrue(torch.equal(norm(x), expected))

    def test_depth_decoder_ties_its_logits_to_the_codebook_embeddings(self):
        depth = self.model.model.depth_decoder
        hidden = torch.randn(2, 32, device=torch_device)
        codes = torch.randint(0, CODEBOOK_SIZE - 1, (2, NUM_CODEBOOKS), device=torch_device)
        masked = torch.ones_like(codes, dtype=torch.bool)
        with torch.no_grad():
            logits = depth(hidden, codes, masked)
            normed = depth.codebook_norm(
                depth.trunk(depth.depth_linear(hidden).unflatten(-1, (NUM_CODEBOOKS, 16)) + depth.mask_embedding)
            )
            expected = torch.stack([normed[:, c] @ depth.codebook_embedding[c].T for c in range(NUM_CODEBOOKS)], dim=1)
        torch.testing.assert_close(logits, expected)
        offsets = torch.arange(NUM_CODEBOOKS, device=torch_device) * CODEBOOK_SIZE
        torch.testing.assert_close(depth.embed_codes(codes), depth.codebook_embedding.flatten(0, 1)[codes + offsets])

    def test_training_losses(self):
        labels = self.inputs["input_ids"].clone()
        labels[labels >= AUDIO_TOKEN_ID] = -100
        audio_labels = self.inputs["audio_out_codes"].clone()
        audio_labels[0, -1] = CODEBOOK_SIZE - 1
        outputs = self.model(**self.inputs, labels=labels, audio_labels=audio_labels)
        self.assertEqual(outputs.loss.shape, ())
        # the pooled mean of training: each modality's mean weighted by its number of supervised units
        text_count = (labels[:, 1:] != -100).sum()
        audio_count = (audio_labels != -100).all(-1).sum()
        expected = (text_count * outputs.text_loss + audio_count * outputs.audio_loss) / (
            text_count + audio_count + 1e-6
        )
        torch.testing.assert_close(outputs.loss, expected)
        outputs.loss.backward()
        self.assertIsNotNone(self.model.model.depth_decoder.mask_embedding.grad)
        self.assertIsNotNone(self.model.model.audio_encoder.downsample.conv_in.weight.grad)
        # ignored rows and a frame placeholder at the first position contribute nothing
        ignored = audio_labels.clone()
        ignored[1] = -100
        first = dict(self.inputs)
        first["input_ids"] = torch.cat(
            [torch.full((1, 1), AUDIO_OUT_TOKEN_ID, device=torch_device), first["input_ids"]], 1
        )
        first["audio_out_codes"] = torch.cat([audio_labels[:1], first["audio_out_codes"]])
        with torch.no_grad():
            self.assertEqual(self.model(**self.inputs, audio_labels=ignored).audio_loss.shape, ())
            self.assertEqual(
                self.model(**first, audio_labels=torch.cat([audio_labels[:1], audio_labels])).audio_loss.shape, ()
            )
            self.assertIsNone(self.model(**self.inputs).loss)
            # speech-only supervision: no text token to average over must not turn the pooled loss into NaN
            speech_only = self.model(**self.inputs, labels=torch.full_like(labels, -100), audio_labels=audio_labels)
            self.assertEqual(speech_only.text_loss.item(), 0.0)
            torch.testing.assert_close(speech_only.loss, speech_only.audio_loss)

    def test_audio_loss_masks_at_least_one_codebook_and_whole_eos_frames(self):
        depth, seen = self.model.model.depth_decoder, []

        class Spy(torch.nn.Module):
            def forward(self, hidden, codes, masked):
                seen.append(masked)
                return torch.zeros(codes.shape[0], NUM_CODEBOOKS, CODEBOOK_SIZE, device=codes.device)

        self.model.model.depth_decoder = Spy()
        audio_labels = torch.tensor([[1, 2, 3, 4], [CODEBOOK_SIZE - 1] * NUM_CODEBOOKS], device=torch_device)
        try:
            for _ in range(20):
                self.model.audio_loss(
                    self.inputs["input_ids"], torch.zeros(1, 16, 32, device=torch_device), audio_labels
                )
        finally:
            self.model.model.depth_decoder = depth
        self.assertTrue(all(bool(masked.any(-1).all()) for masked in seen))
        self.assertTrue(all(bool(masked[1].all()) for masked in seen))
        self.assertFalse(all(bool(masked[0].all()) for masked in seen))

    def test_generate_text_mode_stops_at_eos_or_budget(self):
        forwards = []
        language_model = self.model.model.language_model
        original_forward = language_model.forward
        language_model.forward = lambda *args, **kwargs: forwards.append(kwargs["inputs_embeds"].shape[1]) or (
            original_forward(*args, **kwargs)
        )
        outputs = self.model.generate(**self.inputs, max_new_tokens=5)
        self.assertEqual(outputs.sequences.shape[0], 1)
        self.assertLessEqual(outputs.sequences.shape[1] - self.inputs["input_ids"].shape[1], 5)
        self.assertEqual(outputs.audio_codes.shape[:2], (1, NUM_CODEBOOKS))
        # the prompt forward and one decode step per generated token but the last: nothing consumes its hidden state
        generated = outputs.sequences.shape[1] - self.inputs["input_ids"].shape[1]
        self.assertEqual(forwards, [self.inputs["input_ids"].shape[1]] + [1] * (generated - 1))

    def test_text_eos_is_returned_and_streamed_at_budget_boundary(self):
        eos = self.config.text_config.eos_token_id
        streamer = Collect()
        outputs = self.model.generate(
            **self.inputs, max_new_tokens=1, streamer=streamer, logits_processor=force_tokens(eos)
        )
        self.assertTrue(torch.equal(outputs.sequences[:, :-1], self.inputs["input_ids"]))
        self.assertEqual(outputs.sequences[0, -1].item(), eos)
        self.assertEqual(streamer.items[-1].item(), eos)
        self.assertTrue(streamer.ended)

    def test_generate_audio_mode_emits_frames(self):
        inputs = dict(self.inputs)
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], torch.tensor([[AUDIO_START_ID]], device=torch_device)], 1
        )
        torch.manual_seed(0)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=6,
            depth_decoder_temperature=1.0,
            depth_decoder_top_k=3,
            depth_decoder_repetition_penalty=1.3,
        )
        new_tokens = outputs.sequences[0, inputs["input_ids"].shape[1] :]
        self.assertEqual(int((new_tokens == AUDIO_OUT_TOKEN_ID).sum()), outputs.audio_codes.shape[-1])
        self.assertEqual(outputs.audio_lengths.item(), outputs.audio_codes.shape[-1])
        codes = outputs.audio_codes[0]
        terminal = (codes == CODEBOOK_SIZE - 1).any(0)
        # the frames before the end-of-audio frame hold codec codes; an end-of-audio frame is all end codes
        self.assertTrue(bool((codes[:, ~terminal] < CODEBOOK_SIZE - 1).all()))
        self.assertTrue(bool((codes[:, terminal] == CODEBOOK_SIZE - 1).all()))
        # a prompt ending on a speech frame continues the speech
        inputs["input_ids"] = self.inputs["input_ids"][:, :-1]
        with patch.object(self.model, "sample_audio_frames", side_effect=frames_of(0)):
            outputs = self.model.generate(**inputs, max_new_tokens=3)
        self.assertEqual(outputs.sequences[0, -3:].tolist(), [AUDIO_OUT_TOKEN_ID] * 3)

    def test_depth_decoder_generation_config_is_routed_and_persisted(self):
        inputs = dict(self.inputs)
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], torch.tensor([[AUDIO_START_ID]], device=torch_device)], 1
        )
        depth_config = self.model.depth_decoder.generation_config
        self.assertTrue(depth_config.do_sample)
        greedy = self.model.generate(**inputs, max_new_tokens=4, depth_decoder_do_sample=False)
        again = self.model.generate(**inputs, max_new_tokens=4, depth_decoder_do_sample=False)
        self.assertTrue(torch.equal(greedy.audio_codes, again.audio_codes))
        # `depth_decoder_*` kwargs update the depth decoder's own configuration, which the text settings never touch
        self.assertFalse(depth_config.do_sample)
        text_top_k = self.model.generation_config.top_k
        self.model.generate(
            **inputs, max_new_tokens=2, depth_decoder_do_sample=True, depth_decoder_top_k=3, do_sample=False
        )
        self.assertEqual((depth_config.do_sample, depth_config.top_k), (True, 3))
        self.assertEqual(self.model.generation_config.top_k, text_top_k)
        with tempfile.TemporaryDirectory() as tmp:
            self.model.save_pretrained(tmp)
            with open(os.path.join(tmp, "generation_config.json"), encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual((saved["depth_decoder_top_k"], saved["depth_decoder_repetition_window"]), (3, 64))
            reloaded = Lfm3AudioForConditionalGeneration.from_pretrained(tmp)
        self.assertEqual(reloaded.depth_decoder.generation_config.top_k, 3)
        self.assertFalse(hasattr(reloaded.generation_config, "depth_decoder_top_k"))

    def test_im_end_ends_an_interleaved_turn(self):
        inputs = dict(self.inputs)
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], torch.tensor([[self.config.mixed_start_token_id]], device=torch_device)], 1
        )
        eos = self.config.text_config.eos_token_id
        outputs = self.model.generate(**inputs, max_new_tokens=8, logits_processor=force_tokens(eos))
        self.assertEqual(outputs.sequences.shape[1], inputs["input_ids"].shape[1] + 1)
        self.assertEqual(outputs.sequences[0, -1].item(), eos)

    def test_text_end_outside_an_interleaved_block_ends_the_turn(self):
        text_end = self.config.text_end_token_id
        outputs = self.model.generate(**self.inputs, max_new_tokens=8, logits_processor=force_tokens(text_end))
        self.assertEqual(outputs.sequences.shape[1], self.inputs["input_ids"].shape[1] + 1)
        self.assertEqual(outputs.sequences[0, -1].item(), text_end)

    def test_explicit_generation_mode_primes_the_prompt_with_its_marker(self):
        config = self.config
        inputs = dict(self.inputs)
        prompt_length = inputs["input_ids"].shape[1]
        with patch.object(self.model, "sample_audio_frames", side_effect=frames_of(config.audio_eos_code)):
            outputs = self.model.generate(**inputs, generation_mode="audio", max_new_tokens=2)
            self.assertEqual(outputs.sequences[0, prompt_length].item(), config.audio_start_token_id)
            # the marker is part of the prompt: `max_new_tokens` slots follow it
            self.assertEqual(outputs.sequences.shape[1], prompt_length + 3)
            # a prompt already ending on the marker is left alone
            primed = dict(inputs)
            primed["input_ids"] = torch.cat(
                [inputs["input_ids"], torch.tensor([[config.audio_start_token_id]], device=torch_device)], 1
            )
            outputs = self.model.generate(**primed, generation_mode="audio", max_new_tokens=2)
            self.assertNotEqual(outputs.sequences[0, prompt_length + 1].item(), config.audio_start_token_id)
            # an explicit `max_length` bounds the prompt plus its marker
            outputs = self.model.generate(**inputs, generation_mode="audio", max_length=prompt_length + 3)
            self.assertEqual(outputs.sequences.shape[1], prompt_length + 3)
            # a primed row next to an unprimed one must be primed by the caller
            batch = {key: torch.cat([value, value]) for key, value in inputs.items()}
            mixed = dict(batch)
            mixed["input_ids"] = torch.cat(
                [batch["input_ids"], torch.tensor([[config.audio_start_token_id], [0]], device=torch_device)], 1
            )
            with self.assertRaises(ValueError):
                self.model.generate(**mixed, generation_mode="audio", max_new_tokens=2)
        with self.assertRaises(ValueError):
            self.model.generate(**inputs, generation_mode="speech", max_new_tokens=2)

    def test_explicit_text_mode_keeps_the_reply_in_text(self):
        config = self.config
        eos = config.text_config.eos_token_id
        prompt_length = self.inputs["input_ids"].shape[1]
        prefer_speech = lambda: LogitsProcessorList(  # noqa: E731
            [Boost(lambda step, row: config.mixed_start_token_id, 2e4), Boost(lambda step, row: eos, 1e4)]
        )
        inferred = self.model.generate(**self.inputs, max_new_tokens=1, logits_processor=prefer_speech())
        forced = self.model.generate(
            **self.inputs, generation_mode="text", max_new_tokens=8, logits_processor=prefer_speech()
        )
        self.assertEqual(inferred.sequences[0, prompt_length].item(), config.mixed_start_token_id)
        self.assertEqual(forced.sequences[0, prompt_length:].tolist(), [eos])

    def test_end_of_audio_frames_stay_in_the_output(self):
        inputs = dict(self.inputs)
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], torch.tensor([[AUDIO_START_ID]], device=torch_device)], 1
        )
        with patch.object(self.model, "sample_audio_frames", side_effect=frames_of(CODEBOOK_SIZE - 1)):
            outputs = self.model.generate(**inputs, max_new_tokens=2, logits_processor=force_tokens(None, 10))
        # the end-of-audio frame keeps its slot so the output can be fed back as a prompt; the turn goes on in text
        self.assertEqual(outputs.audio_codes.shape[-1], 1)
        self.assertEqual(outputs.sequences[0, inputs["input_ids"].shape[1] :].tolist(), [AUDIO_OUT_TOKEN_ID, 10])
        self.assertTrue(bool((outputs.audio_codes == CODEBOOK_SIZE - 1).all()))

    def test_sample_audio_frames_reveals_by_schedule(self):
        seen = []
        trunk = self.model.model.depth_decoder.trunk

        def spy(hidden):
            seen.append(hidden[0].clone())
            return trunk(hidden)

        self.model.model.depth_decoder.trunk = spy
        hidden = torch.randn(1, 32, device=torch_device)
        history = torch.randint(0, 5, (1, 3, NUM_CODEBOOKS), device=torch_device)
        self.model.depth_decoder.generation_config.update(repetition_penalty=1.5)
        torch.manual_seed(0)
        try:
            codes = self.model.sample_audio_frames(hidden, history)
        finally:
            self.model.depth_decoder.generation_config.update(repetition_penalty=1.0)
        self.assertEqual(len(seen), len(self.config.maskgit_schedule))
        self.assertEqual(codes.shape, (1, NUM_CODEBOOKS))
        # the first pass hides every codebook, later passes reveal the scheduled number each
        mask = self.model.model.depth_decoder.mask_embedding
        depth_in = self.model.model.depth_decoder.depth_linear(hidden[0]).unflatten(-1, (NUM_CODEBOOKS, 16))
        hidden_counts = [int(torch.isclose(x - depth_in, mask).all(-1).sum()) for x in seen]
        self.assertEqual(hidden_counts[0], NUM_CODEBOOKS)
        self.assertEqual(hidden_counts[1], NUM_CODEBOOKS - self.config.maskgit_schedule[0])

    def test_depth_repetition_penalty_reads_the_past_frames(self):
        hidden = torch.randn(1, 32, device=torch_device)
        depth = self.model.depth_decoder
        with patch("torch.randperm", side_effect=lambda n, **kw: torch.arange(n, **kw)):
            depth.generation_config.update(do_sample=False)
            try:
                plain = self.model.sample_audio_frames(hidden)
                history = plain[:, None].expand(-1, 3, -1)
                depth.generation_config.update(repetition_penalty=100.0)
                penalized = self.model.sample_audio_frames(hidden, history)
                # an empty history (rows padded with -1) applies no penalty
                padded = self.model.sample_audio_frames(hidden, torch.full_like(history, -1))
            finally:
                depth.generation_config.update(do_sample=True, repetition_penalty=1.0)
        self.assertFalse(torch.equal(plain, penalized))
        self.assertTrue(torch.equal(plain, padded))

    def test_streamer_receives_every_generated_slot(self):
        streamer = Collect()
        inputs = dict(self.inputs)
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], torch.tensor([[AUDIO_START_ID]], device=torch_device)], 1
        )
        torch.manual_seed(0)
        outputs = self.model.generate(**inputs, max_new_tokens=4, streamer=streamer)
        self.assertTrue(streamer.ended)
        generated = outputs.sequences[0, inputs["input_ids"].shape[1] :]
        # the mixin streams the prompt first, then one slot per step
        self.assertEqual(torch.cat(streamer.items[1:]).tolist(), generated.tolist())
        self.assertEqual(int((generated == AUDIO_OUT_TOKEN_ID).sum()), outputs.audio_codes.shape[-1])

    def test_generate_interleaved_mode_alternates_blocks(self):
        config = tiny_config(interleaved_n_text=2, interleaved_n_audio=3)
        torch.manual_seed(0)
        model = Lfm3AudioForConditionalGeneration(config).to(torch_device).eval()
        inputs = dict(self.inputs)
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], torch.tensor([[MIXED_START_ID]], device=torch_device)], 1
        )
        # text that never picks a marker or the end-of-sequence token; frames that never end the audio
        with patch.object(model, "sample_audio_frames", side_effect=frames_of(0)):
            outputs = model.generate(
                **inputs, max_new_tokens=10, logits_processor=force_tokens(*([10] * 10)), return_dict_in_generate=True
            )
        new_tokens = outputs.sequences[0, inputs["input_ids"].shape[1] :]
        is_frame = (new_tokens == AUDIO_OUT_TOKEN_ID).tolist()
        self.assertEqual(is_frame, [False, False, True, True, True, False, False, True, True, True])

    def test_text_end_in_eos_token_id_still_only_ends_the_text_of_an_interleaved_block(self):
        # text-only serving lists `<|text_end|>` as an end-of-sequence token; interleaved speech must keep reading
        # it as the end of a block's text, with the audio tail still to come
        config = tiny_config(interleaved_n_text=2, interleaved_n_audio=3)
        torch.manual_seed(0)
        model = Lfm3AudioForConditionalGeneration(config).to(torch_device).eval()
        inputs = dict(self.inputs)
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], torch.tensor([[MIXED_START_ID]], device=torch_device)], 1
        )
        eos_token_id = [config.text_config.eos_token_id, config.text_end_token_id]
        with patch.object(model, "sample_audio_frames", side_effect=frames_of(0)):
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                eos_token_id=eos_token_id,
                logits_processor=force_tokens(10, config.text_end_token_id),
                return_dict_in_generate=True,
            )
        new_tokens = outputs.sequences[0, inputs["input_ids"].shape[1] :]
        # `<|text_end|>` closes the text for the rest of the turn; the audio keeps going instead of the turn ending
        is_frame = (new_tokens == AUDIO_OUT_TOKEN_ID).tolist()
        self.assertEqual(is_frame, [False, False] + [True] * 8)
        self.assertEqual(new_tokens[1].item(), config.text_end_token_id)

    def test_mixed_start_opens_a_block_without_consuming_a_text_slot(self):
        config = tiny_config(interleaved_n_text=2, interleaved_n_audio=3)
        torch.manual_seed(0)
        model = Lfm3AudioForConditionalGeneration(config).to(torch_device).eval()
        ids = torch.tensor([[1, 10, 11]], device=torch_device)
        # the sampled marker opens the block; its two text slots follow, then three frames, then text again
        with patch.object(model, "sample_audio_frames", side_effect=frames_of(0)):
            outputs = model.generate(
                ids,
                max_new_tokens=7,
                return_dict_in_generate=True,
                logits_processor=force_tokens(MIXED_START_ID, *([10] * 6)),
            )
        new_tokens = outputs.sequences[0, 3:].tolist()
        self.assertEqual(
            new_tokens, [MIXED_START_ID, 10, 10, AUDIO_OUT_TOKEN_ID, AUDIO_OUT_TOKEN_ID, AUDIO_OUT_TOKEN_ID, 10]
        )

    def test_terminal_frame_leaves_the_interleaved_block(self):
        config = tiny_config(interleaved_n_text=2, interleaved_n_audio=3)
        torch.manual_seed(0)
        model = Lfm3AudioForConditionalGeneration(config).to(torch_device).eval()
        ids = torch.tensor([[1, 10, MIXED_START_ID]], device=torch_device)
        eos = config.text_config.eos_token_id
        # text_end closes the text, the first frame is terminal: the row is back in text and stops at its EOS
        with patch.object(model, "sample_audio_frames", side_effect=frames_of(config.audio_eos_code)):
            outputs = model.generate(
                ids,
                max_new_tokens=7,
                return_dict_in_generate=True,
                logits_processor=force_tokens(TEXT_END_ID, None, eos),
            )
        self.assertEqual(outputs.sequences[0, 3:].tolist(), [TEXT_END_ID, AUDIO_OUT_TOKEN_ID, eos])
        self.assertEqual(outputs.audio_lengths.tolist(), [1])

    def test_marker_priming_precedes_the_cache_and_the_streamer(self):
        ids = torch.tensor([[1, 10]], device=torch_device)
        streamer = Collect()
        with patch.object(self.model, "sample_audio_frames", side_effect=frames_of(0)):
            dynamic = self.model.generate(ids, generation_mode="audio", max_new_tokens=3, streamer=streamer)
            static = self.model.generate(ids, generation_mode="audio", max_new_tokens=3, cache_implementation="static")
        self.assertEqual(dynamic.sequences[0].tolist(), [1, 10, AUDIO_START_ID] + [AUDIO_OUT_TOKEN_ID] * 3)
        # the streamer sees the primed prompt, then one slot per step
        self.assertEqual(streamer.items[0].tolist(), [[1, 10, AUDIO_START_ID]])
        self.assertEqual(torch.cat(streamer.items[1:]).tolist(), [AUDIO_OUT_TOKEN_ID] * 3)
        torch.testing.assert_close(static.sequences, dynamic.sequences)

    def test_a_callers_stopping_criterion_is_honoured(self):
        from transformers.generation.stopping_criteria import EosTokenCriteria, StoppingCriteriaList

        ids = torch.tensor([[1, 4, 5]], device=torch_device)
        outputs = self.model.generate(
            ids,
            max_new_tokens=3,
            generation_mode="text",
            logits_processor=force_tokens(10, 11, 12),
            stopping_criteria=StoppingCriteriaList([EosTokenCriteria(10)]),
        )
        self.assertEqual(outputs.sequences[0, 3:].tolist(), [10])

    def test_cacheless_generation_serves_text_only(self):
        ids = torch.tensor([[1, 4, 5]], device=torch_device)
        cached = self.model.generate(ids, max_new_tokens=3, generation_mode="text")
        cacheless = self.model.generate(ids, max_new_tokens=3, generation_mode="text", use_cache=False)
        torch.testing.assert_close(cacheless.sequences, cached.sequences)
        with self.assertRaisesRegex(ValueError, "use_cache=True"):
            self.model.generate(ids, max_new_tokens=2, generation_mode="audio", use_cache=False)
        with self.assertRaisesRegex(ValueError, "prefill_chunk_size"):
            self.model.generate(ids, max_new_tokens=2, generation_mode="text", prefill_chunk_size=2)

    def test_depth_settings_on_a_passed_generation_config_are_routed(self):
        ids = torch.tensor([[1, 10, AUDIO_START_ID]], device=torch_device)
        depth = self.model.depth_decoder.generation_config
        self.model.generate(ids, generation_config=GenerationConfig(max_new_tokens=2, depth_decoder_top_k=5))
        self.assertEqual(depth.top_k, 5)
        # the call wins over the passed configuration
        self.model.generate(
            ids, generation_config=GenerationConfig(max_new_tokens=2, depth_decoder_top_k=5), depth_decoder_top_k=7
        )
        self.assertEqual(depth.top_k, 7)

    def test_saved_depth_settings_follow_resets_and_load_overrides(self):
        depth = self.model.depth_decoder.generation_config
        depth.update(top_k=3)
        with tempfile.TemporaryDirectory() as tmp:
            self.model.save_pretrained(tmp)
            depth.update(top_k=None)
            self.model.save_pretrained(tmp)
            with open(os.path.join(tmp, "generation_config.json"), encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertNotIn("depth_decoder_top_k", saved)
            self.assertNotIn("depth_decoder__from_model_config", saved)
            depth.update(top_k=3)
            self.model.save_pretrained(tmp)
            # a text-only generation config passed to `from_pretrained` keeps the checkpoint's depth settings
            reloaded = Lfm3AudioForConditionalGeneration.from_pretrained(
                tmp, generation_config=GenerationConfig(max_new_tokens=12)
            )
            self.assertEqual(reloaded.depth_decoder.generation_config.top_k, 3)
            self.assertEqual(reloaded.generation_config.max_new_tokens, 12)
            model, info = Lfm3AudioForConditionalGeneration.from_pretrained(tmp, output_loading_info=True)
            self.assertEqual({k: v for k, v in info.items() if v}, {})
            self.assertIsInstance(
                Lfm3AudioForConditionalGeneration.from_pretrained(tmp, output_loading_info=False),
                Lfm3AudioForConditionalGeneration,
            )

    def test_live_transcription_suppresses_the_markers_on_every_chunk_and_respects_a_zero_budget(self):
        prompt = torch.tensor([[1, 10, 11]], device=torch_device)
        features = torch.randn(1, 16, 96, device=torch_device)
        head, original = self.model.lm_head, self.model.lm_head.forward

        def boosted(hidden_states):  # the markers would win on every chunk
            logits = original(hidden_states)
            logits[..., [AUDIO_START_ID, MIXED_START_ID]] += 50.0
            return logits

        head.forward = boosted
        try:
            output = self.model.transcribe_live(features, input_ids=prompt, max_new_tokens=3)
            self.assertNotIn(AUDIO_START_ID, output.sequences[0].tolist())
            output = self.model.transcribe_live(features, output.state, is_final=True, max_new_tokens=3)
            self.assertNotIn(AUDIO_START_ID, output.sequences[0].tolist())
            self.assertNotIn(MIXED_START_ID, output.sequences[0].tolist())
        finally:
            head.forward = original
        # no text budget: the chunk only advances the context and keeps `<|text_end|>` pending
        output = self.model.transcribe_live(features, input_ids=prompt, max_new_tokens=0)
        self.assertEqual(output.sequences.shape[1], 0)
        self.assertFalse(output.closed)
        self.assertEqual(output.state.pending.shape[1], 1)
        self.assertEqual(output.state.past_key_values.get_seq_length(), 3 + 12)

    def test_decode_with_cache_matches_full_prefill(self):
        inputs = dict(self.inputs)
        with torch.no_grad():
            full = self.model(**inputs).logits[0, -1]
            cache_out = self.model.model(
                **{k: (v[:, :-1] if k == "input_ids" else v) for k, v in inputs.items()},
                use_cache=True,
            )
            step = self.model.model.language_model(
                inputs_embeds=self.model.get_input_embeddings()(inputs["input_ids"][:, -1:]),
                past_key_values=cache_out.past_key_values,
                use_cache=True,
            )
            cached = self.model.lm_head(step.last_hidden_state[0, -1])
        torch.testing.assert_close(cached, full, atol=1e-4, rtol=1e-4)

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.model.save_pretrained(tmp)
            reloaded, info = Lfm3AudioForConditionalGeneration.from_pretrained(tmp, output_loading_info=True)
            self.assertEqual({k: v for k, v in info.items() if v}, {})
            self.assertIsInstance(AutoConfig.from_pretrained(tmp), Lfm3AudioConfig)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value.cpu(), reloaded.state_dict()[key].cpu()), key)

    def test_batched_audio_features_match_unpadded_clips(self):
        features = torch.randn(2, 16, 57, device=torch_device)
        mask = torch.ones(2, 57, device=torch_device, dtype=torch.long)
        mask[0, 9:] = 0
        with torch.no_grad():
            expected = torch.cat(
                [
                    self.model.model.get_audio_features(features[:1, :, :9], mask[:1, :9]).pooler_output,
                    self.model.model.get_audio_features(features[1:], mask[1:]).pooler_output,
                ]
            )
            actual = self.model.model.get_audio_features(features, mask).pooler_output
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_batched_text_matches_individual_left_padded_prompts(self):
        ids = torch.tensor([[0, 0, 1, 10], [1, 11, 12, 13]], device=torch_device)
        mask = ids.ne(0).long()
        options = {"max_new_tokens": 5, "generation_mode": "text"}
        outputs = self.model.generate(ids, attention_mask=mask, **options)
        inferred = self.model.generate(ids, **options)
        torch.testing.assert_close(inferred.sequences, outputs.sequences)
        for row in range(2):
            prompt = ids[row][mask[row].bool()][None]
            single = self.model.generate(prompt, **options)
            generated = outputs.sequences[row, 4:]
            generated = generated[generated != self.config.text_config.pad_token_id]
            self.assertEqual(generated.tolist(), single.sequences[0, prompt.shape[1] :].tolist())
        self.assertEqual(outputs.audio_lengths.tolist(), [0, 0])

    def test_batch_has_independent_modes_eos_lengths_and_streamers(self):
        ids = torch.tensor([[1, 10], [1, AUDIO_START_ID], [1, MIXED_START_ID]], device=torch_device)
        streamer = Collect()
        # the first row ends immediately; the third row closes its text and continues speaking
        tokens = {(0, 0): 2, (0, 2): TEXT_END_ID, (1, 0): 12, (2, 0): 2}
        calls = []

        def frames(hidden_states, *args, **kwargs):
            calls.append(hidden_states.shape)
            codes = ids.new_zeros((hidden_states.shape[0], NUM_CODEBOOKS))
            if len(calls) == 2:
                codes[0] = CODEBOOK_SIZE - 1
            return codes

        processor = LogitsProcessorList([Boost(lambda step, row: tokens.get((step, row)))])
        with patch.object(self.model, "sample_audio_frames", side_effect=frames):
            outputs = self.model.generate(ids, max_new_tokens=4, streamer=streamer, logits_processor=processor)
        self.assertEqual(outputs.sequences[0].tolist(), [1, 10, 2, 0, 0, 0])
        self.assertEqual(outputs.audio_lengths.tolist(), [0, 2, 3])
        self.assertTrue(bool((outputs.audio_codes[0] == -1).all()))
        self.assertTrue(bool((outputs.audio_codes[1, :, 1] == CODEBOOK_SIZE - 1).all()))
        self.assertEqual(calls[1], torch.Size([2, 32]))
        self.assertTrue(streamer.ended)
        self.assertEqual(torch.stack(streamer.items[1:], 1).tolist(), outputs.sequences[:, 2:].tolist())

    def test_batched_maskgit_matches_single_rows_with_fixed_reveal_order(self):
        hidden = torch.randn(2, 32, device=torch_device)
        history = torch.randint(0, 5, (2, 3, NUM_CODEBOOKS), device=torch_device)
        history[1] = -1
        depth = self.model.depth_decoder
        depth.generation_config.update(do_sample=False, repetition_penalty=1.2)
        try:
            with torch.no_grad(), patch("torch.randperm", side_effect=lambda n, **kw: torch.arange(n, **kw)):
                batch = self.model.sample_audio_frames(hidden, history)
                singles = torch.cat(
                    [self.model.sample_audio_frames(hidden[row : row + 1], history[row : row + 1]) for row in range(2)]
                )
        finally:
            depth.generation_config.update(do_sample=True, repetition_penalty=1.0)
        torch.testing.assert_close(batch, singles)

    def test_streaming_encoder_matches_offline_and_bounds_state(self):
        encoder = self.model.model.audio_encoder
        for length in (1, 7, 8, 9, 15, 16, 57, 129):
            features = torch.randn(2, 16, length, device=torch_device)
            with torch.no_grad():
                expected = encoder(features, torch.tensor([length, length], device=torch_device))[0].last_hidden_state
                state, outputs = None, []
                for start in range(0, length, 5):
                    encoded, state = encoder.stream(features[..., start : start + 5], state)
                    outputs.append(encoded)
                    self.assertLessEqual(state.features.shape[-1], 15)
                encoded, state = encoder.stream(features[..., :0], state, is_final=True)
                outputs.append(encoded)
                actual = torch.cat(outputs, dim=1)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            self.assertIsNone(state.features)
            for layer in state.past_key_values.layers:
                if getattr(layer, "keys", None) is not None:
                    self.assertLessEqual(layer.keys.shape[-2], encoder.config.sliding_window)
            with self.assertRaises(ValueError):
                encoder.stream(features[..., :0], state)

    def test_cached_generation_matches_full_prompt_in_all_modes(self):
        prefix = torch.tensor([[0, 1, 4], [1, 5, 6]], device=torch_device)
        prefix_mask = prefix.ne(0).long()
        for last_token in (7, AUDIO_START_ID, MIXED_START_ID):
            with self.subTest(last_token=last_token), torch.no_grad():
                suffix = prefix.new_full((2, 2), last_token)
                ids = torch.cat((prefix, suffix), dim=-1)
                mask = ids.ne(0).long()
                torch.manual_seed(12)
                expected = self.model.generate(ids, attention_mask=mask, max_new_tokens=5)
                cache = self.model(
                    input_ids=prefix,
                    attention_mask=prefix_mask,
                    position_ids=prefix_mask.cumsum(-1).sub(1).clamp_min(0),
                    use_cache=True,
                ).past_key_values
                torch.manual_seed(12)
                actual = self.model.generate(suffix, past_key_values=cache, attention_mask=mask, max_new_tokens=5)
                torch.testing.assert_close(actual.sequences[:, 2:], expected.sequences[:, 5:])
                torch.testing.assert_close(actual.audio_codes, expected.audio_codes)
                torch.testing.assert_close(actual.audio_lengths, expected.audio_lengths)

    def test_projected_streaming_features_match_offline(self):
        features = torch.randn(1, 16, 57, device=torch_device)
        mask = torch.ones(1, 57, device=torch_device, dtype=torch.long)
        with torch.no_grad():
            expected = self.model.model.get_audio_features(features, mask).pooler_output
            first, state = self.model.model.get_audio_features_stream(features[..., :17])
            last, state = self.model.model.get_audio_features_stream(features[..., 17:], state, is_final=True)
        torch.testing.assert_close(torch.cat((first, last), dim=1)[0], expected, atol=2e-5, rtol=2e-5)

    def _live_blocks(self, chunks, prompt, **kwargs):
        state, blocks = None, []
        for index, chunk in enumerate(chunks):
            output = self.model.transcribe_live(
                chunk, state, input_ids=prompt if state is None else None, is_final=index == len(chunks) - 1, **kwargs
            )
            state = output.state
            blocks.append((output.num_audio_tokens, output.sequences[0].tolist(), output.closed))
        return blocks, state

    def test_live_transcription_matches_the_offline_layout(self):
        # <prompt> [A x 12] text<|text_end|> [A x 12] text<|text_end|> [A x r] text: every streamed token is the greedy
        # choice of an offline forward over the layout it produced, under the mid-stream suppression
        prompt = torch.tensor([[1, 10, 11]], device=torch_device)
        features = torch.randn(1, 16, 96 * 2 + 40, device=torch_device)
        chunks = [features[..., :96], features[..., 96:192], features[..., 192:]]
        suppress = [20, 21]
        suppressed = [AUDIO_START_ID, MIXED_START_ID, self.config.text_config.eos_token_id, *suppress]
        head, original = self.model.lm_head, self.model.lm_head.forward
        for boost in (0.0, 50.0):  # the boost makes `<|text_end|>` win, so every block closes at once

            def boosted(hidden_states):
                logits = original(hidden_states)
                logits[..., TEXT_END_ID] += boost
                return logits

            head.forward = boosted
            with self.subTest(boost=boost), torch.no_grad():
                blocks, state = self._live_blocks(
                    chunks, prompt, max_new_tokens=3, suppress_token_ids=suppress, stop_token_ids=[3]
                )
                self.assertTrue(state.finished)
                self.assertEqual([num_tokens for num_tokens, _, _ in blocks], [12, 12, 5])
                self.assertEqual([closed for _, _, closed in blocks], [boost > 0] * 3)
                ids, checks = prompt[0].tolist(), []
                for index, (num_tokens, text, closed) in enumerate(blocks):
                    final = index == len(blocks) - 1
                    ids += [AUDIO_TOKEN_ID] * num_tokens
                    for token in text:
                        checks.append((len(ids), token, final))
                        ids.append(token)
                    if not final:
                        # the model's own `<|text_end|>` is checked; a forced one only shapes the layout
                        if closed:
                            checks.append((len(ids), TEXT_END_ID, final))
                        ids.append(TEXT_END_ID)
                # the final block's last token stays unfed when the budget runs out: nothing follows it
                self.assertEqual(state.past_key_values.get_seq_length(), len(ids) - (0 if blocks[-1][2] else 1))
                logits = self.model(
                    input_ids=torch.tensor([ids], device=torch_device),
                    input_features=features,
                    feature_attention_mask=torch.ones(1, features.shape[-1], dtype=torch.long, device=torch_device),
                ).logits[0]
            head.forward = original
            self.assertGreater(len(checks), 1)
            for position, token, final in checks:
                row = logits[position - 1].clone()
                # the placeholders are never text, on the final chunk included
                row[[AUDIO_TOKEN_ID, AUDIO_OUT_TOKEN_ID]] = float("-inf")
                if not final:
                    row[suppressed] = float("-inf")
                self.assertEqual(int(row.argmax()), token)

    def test_live_transcription_restarts_contexts_and_validates(self):
        prompt = torch.tensor([[1, 10, 11]], device=torch_device)
        features = torch.randn(1, 16, 96, device=torch_device)
        with self.assertRaisesRegex(ValueError, "prompt"):
            self.model.transcribe_live(features)
        with self.assertRaisesRegex(ValueError, "one stream"):
            self.model.transcribe_live(features.expand(2, -1, -1), input_ids=prompt)
        output = self.model.transcribe_live(features, input_ids=prompt, max_new_tokens=2)
        self.assertEqual((output.num_audio_tokens, output.state.encoder_cache.num_tokens), (12, 12))
        # the cache holds the prompt, the audio and every text token but the ones kept pending for the next chunk
        pending = output.state.pending.shape[1]
        self.assertEqual(
            output.state.past_key_values.get_seq_length() + pending, 3 + 12 + output.sequences.shape[1] + 1
        )
        # a rejected restart leaves the previous context in place
        cache, length = output.state.past_key_values, output.state.past_key_values.get_seq_length()
        with self.assertRaisesRegex(ValueError, "one stream"):
            self.model.transcribe_live(features[:, :15], output.state, input_ids=prompt)
        with self.assertRaisesRegex(ValueError, "one prompt"):
            self.model.transcribe_live(features, output.state, input_ids=prompt.expand(2, -1))
        # the live audio arrives as features, so a prompt placeholder would never have any behind it
        for token_id in (AUDIO_TOKEN_ID, AUDIO_OUT_TOKEN_ID):
            placeholder_prompt = torch.tensor([[1, token_id, 11]], device=torch_device)
            with self.subTest(token_id=token_id):
                with self.assertRaisesRegex(ValueError, "speech placeholders"):
                    self.model.transcribe_live(features, input_ids=placeholder_prompt)
                with self.assertRaisesRegex(ValueError, "speech placeholders"):
                    self.model.transcribe_live(features, output.state, input_ids=placeholder_prompt)
        self.assertIs(output.state.past_key_values, cache)
        self.assertEqual(cache.get_seq_length(), length)
        self.assertEqual(output.state.encoder_cache.num_tokens, 12)
        # a new context restarts the language model from the prompt while the encoder keeps following the stream
        output = self.model.transcribe_live(features, output.state, input_ids=prompt, max_new_tokens=2)
        self.assertEqual(output.state.encoder_cache.num_tokens, 24)
        pending = output.state.pending.shape[1]
        self.assertEqual(
            output.state.past_key_values.get_seq_length() + pending, 3 + 12 + output.sequences.shape[1] + 1
        )
        # a chunk completing no token decodes nothing and keeps the closing `<|text_end|>` pending
        cached = output.state.past_key_values.get_seq_length()
        output = self.model.transcribe_live(features[..., :5], output.state)
        self.assertEqual((output.num_audio_tokens, output.sequences.shape[1]), (0, 0))
        self.assertIsNotNone(output.state.pending)
        self.assertEqual(output.state.past_key_values.get_seq_length(), cached)
        output = self.model.transcribe_live(features[..., :0], output.state, is_final=True, max_new_tokens=2)
        self.assertEqual(output.num_audio_tokens, 1)
        self.assertTrue(output.state.finished)
        with self.assertRaisesRegex(ValueError, "finalized"):
            self.model.transcribe_live(features, output.state)
        self.model.train()
        with self.assertRaisesRegex(ValueError, "eval mode"):
            self.model.transcribe_live(features, input_ids=prompt)

    def test_live_transcription_survives_empty_chunks(self):
        prompt = torch.tensor([[1, 10, 11]], device=torch_device)
        features = torch.randn(1, 16, 96, device=torch_device)
        output = self.model.transcribe_live(features, input_ids=prompt, max_new_tokens=2)
        # an empty chunk completes nothing; the empty final chunk still feeds the pending tokens and decodes
        cached, pending = output.state.past_key_values.get_seq_length(), output.state.pending.shape[1]
        output = self.model.transcribe_live(features[..., :0], output.state)
        self.assertEqual((output.num_audio_tokens, output.sequences.shape[1]), (0, 0))
        self.assertEqual(output.state.past_key_values.get_seq_length(), cached)
        output = self.model.transcribe_live(features[..., :0], output.state, is_final=True, max_new_tokens=2)
        self.assertEqual(output.num_audio_tokens, 0)
        fed = output.sequences.shape[1] - (0 if output.closed else 1)
        self.assertEqual(output.state.past_key_values.get_seq_length(), cached + pending + fed)
        self.assertTrue(output.state.finished)

    def test_live_transcription_list_eos_is_suppressed_then_stops(self):
        prompt = torch.tensor([[1, 10, 11]], device=torch_device)
        features = torch.randn(1, 16, 96, device=torch_device)
        self.model.config.text_config.eos_token_id = [2, 3]
        head, original = self.model.lm_head, self.model.lm_head.forward

        def boosted(hidden_states):  # token 3 always wins unless suppressed
            logits = original(hidden_states)
            logits[..., 3] += 50.0
            return logits

        head.forward = boosted
        try:
            output = self.model.transcribe_live(features, input_ids=prompt, max_new_tokens=3)
            self.assertEqual(output.sequences.shape[1], 3)
            self.assertNotIn(3, output.sequences[0].tolist())
            output = self.model.transcribe_live(features, output.state, is_final=True, max_new_tokens=3)
        finally:
            head.forward = original
        self.assertTrue(output.closed)
        self.assertEqual(output.sequences.shape[1], 0)

    def test_audio_models_are_not_speech_seq2seq(self):
        # the ASR pipeline would feed mel features as `input_ids`
        from transformers.models.auto.modeling_auto import MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES

        self.assertNotIn("lfm3_audio", MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES)
        self.assertNotIn("lfm3_omni", MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES)

    def test_audio_labels_shape_is_validated(self):
        inputs = dict(self.inputs)
        with self.assertRaisesRegex(ValueError, "audio_labels"):
            self.model(**inputs, audio_labels=inputs["audio_out_codes"][:1])

    def test_audio_label_codes_are_range_checked(self):
        # an unchecked code indexes the codebook embeddings: an IndexError on CPU, a device-side assert on CUDA
        valid = self.inputs["audio_out_codes"]
        for name, code in (
            ("codebook_size", CODEBOOK_SIZE),
            ("negative", -1),
            ("int64_min", torch.iinfo(torch.long).min),
            ("partly ignored", -100),
        ):
            labels = valid.clone()
            labels[1, 0] = code
            if name == "int64_min":
                labels[1] = code
            with self.subTest(name), self.assertRaisesRegex(ValueError, "audio_labels must be in"):
                self.model(**self.inputs, audio_labels=labels)
        labels = valid.clone()
        labels[1] = -100
        with torch.no_grad():
            self.assertTrue(torch.isfinite(self.model(**self.inputs, audio_labels=labels).audio_loss))

    def test_offline_speech_features_are_shape_checked(self):
        inputs = dict(self.inputs)
        with self.assertRaisesRegex(ValueError, "input_features of shape"):
            self.model(**{**inputs, "input_features": inputs["input_features"][:, :15]})
        with self.assertRaisesRegex(ValueError, "feature_attention_mask of shape"):
            self.model(**{**inputs, "feature_attention_mask": inputs["feature_attention_mask"][:, :30]})

    def test_generate_returns_plain_sequences_on_request(self):
        ids = torch.tensor([[1, 4, 5]], device=torch_device)
        with torch.no_grad():
            # the model's generation config asks for the output object (the released checkpoints do), the call wins
            rich = self.model.generate(ids, max_new_tokens=2, generation_mode="text")
            plain = self.model.generate(ids, max_new_tokens=2, generation_mode="text", return_dict_in_generate=False)
            from_config = self.model.generate(
                ids, generation_config=GenerationConfig(max_new_tokens=2), generation_mode="text"
            )
        self.assertIsInstance(plain, torch.Tensor)
        self.assertIs(type(from_config), type(rich))
        torch.testing.assert_close(plain, rich.sequences)
        # without it, `generate` returns the sequences like every other model
        self.model.generation_config.return_dict_in_generate = False
        with torch.no_grad():
            default = self.model.generate(ids, max_new_tokens=2, generation_mode="text")
        self.assertIsInstance(default, torch.Tensor)
        with self.assertRaisesRegex(ValueError, "greedy or sampling"):
            self.model.generate(ids, max_new_tokens=2, num_beams=2)

    def test_base_model_has_no_head(self):
        base = Lfm3AudioModel(self.config).to(torch_device).eval()
        with torch.no_grad():
            outputs = base(**self.inputs)
        self.assertEqual(outputs.last_hidden_state.shape, (1, self.inputs["input_ids"].shape[1], 32))

    def test_placeholders_without_their_features_are_rejected(self):
        inputs = dict(self.inputs)
        with self.assertRaisesRegex(ValueError, "audio_out_codes"):
            self.model(**{**inputs, "audio_out_codes": None})
        with self.assertRaisesRegex(ValueError, "input_features"):
            self.model(**{**inputs, "input_features": None, "feature_attention_mask": None})
        # the embedding form is checked the way `get_placeholder_mask` finds placeholders
        embeds = self.model.get_input_embeddings()(inputs["input_ids"])
        with self.assertRaisesRegex(ValueError, "input_features"):
            self.model(inputs_embeds=embeds)
        with self.assertRaisesRegex(ValueError, "input_features"):
            self.model(inputs_embeds=embeds, audio_out_codes=inputs["audio_out_codes"])
        with torch.no_grad():
            self.model(inputs_embeds=self.model.get_input_embeddings()(inputs["input_ids"][:, :3]))
        # an injected trailing placeholder must not switch `generate` into speech mode
        ids = torch.tensor([[1, 4, AUDIO_OUT_TOKEN_ID]], device=torch_device)
        with self.assertRaisesRegex(ValueError, "audio_out_codes"):
            self.model.generate(ids, max_new_tokens=2)
        with torch.no_grad():
            self.assertEqual(self.model(**inputs).logits.shape[1], inputs["input_ids"].shape[1])

    def test_placeholder_ids_are_never_sampled_as_text(self):
        prompt = torch.tensor([[1, 10, 11]], device=torch_device)
        features = torch.randn(1, 16, 96, device=torch_device)
        for boosted_id in (AUDIO_TOKEN_ID, AUDIO_OUT_TOKEN_ID):
            with self.subTest(boosted_id=boosted_id), torch.no_grad():
                boost = LogitsProcessorList([Boost(lambda step, row: boosted_id)])
                outputs = self.model.generate(
                    prompt, max_new_tokens=3, generation_mode="text", do_sample=True, logits_processor=boost
                )
                self.assertNotIn(boosted_id, outputs.sequences[0, 3:].tolist())
                self.assertEqual(outputs.audio_lengths.tolist(), [0])
                head, original = self.model.lm_head, self.model.lm_head.forward

                def boosted(hidden_states):
                    logits = original(hidden_states)
                    logits[..., boosted_id] += 50.0
                    return logits

                head.forward = boosted
                try:
                    output = self.model.transcribe_live(features, input_ids=prompt, max_new_tokens=3)
                    self.assertNotIn(boosted_id, output.sequences[0].tolist())
                    output = self.model.transcribe_live(features, output.state, is_final=True, max_new_tokens=3)
                    self.assertNotIn(boosted_id, output.sequences[0].tolist())
                finally:
                    head.forward = original

    def test_every_parameter_receives_a_gradient(self):
        model = Lfm3AudioForConditionalGeneration(self.config).to(torch_device).train()
        inputs = tiny_inputs(torch_device)
        labels = inputs["input_ids"].clone()
        outputs = model(**inputs, labels=labels, audio_labels=inputs["audio_out_codes"])
        outputs.loss.backward()
        missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
        self.assertEqual(missing, [])

    @require_torch_accelerator
    @require_accelerate
    def test_speech_paths_run_with_the_depth_decoder_on_another_device(self):
        from accelerate.hooks import attach_execution_device_hook

        def speech_paths(model, device):
            inputs = tiny_inputs(device)

            # text that never ends the turn, so the interleaved block reaches its speech frames
            with torch.no_grad():
                loss = model(**inputs, audio_labels=inputs["audio_out_codes"]).audio_loss
                self.assertTrue(torch.isfinite(loss))
                audio = model.generate(torch.tensor([[1, 10, AUDIO_START_ID]], device=device), max_new_tokens=3)
                self.assertEqual(audio.audio_codes.shape[:2], (1, NUM_CODEBOOKS))
                mixed = model.generate(
                    torch.tensor([[1, 10, MIXED_START_ID]], device=device),
                    max_new_tokens=8,
                    logits_processor=force_tokens(*([10] * 8)),
                )
                self.assertGreater(mixed.audio_lengths.item(), 0)
                live = model.transcribe_live(
                    torch.randn(1, 16, 40, device=device),
                    input_ids=torch.tensor([[1, 10, 11]], device=device),
                    is_final=True,
                    max_new_tokens=2,
                )
                self.assertEqual(live.sequences.ndim, 2)

        # the per-module execution-device hooks of a multi-device `device_map`, with the CPU as the second device
        model = self.model
        for name, device in (
            ("model.language_model", torch_device),
            ("lm_head", torch_device),
            ("model.depth_decoder", "cpu"),
            ("model.codebook_embedding", "cpu"),
            ("model.audio_encoder", "cpu"),
            ("model.audio_projector", "cpu"),
        ):
            module = model.get_submodule(name)
            module.to(device)
            attach_execution_device_hook(module, device)
        speech_paths(model, torch_device)
        # CPU offload keeps the depth decoder's weights off the accelerator until its forward runs
        with tempfile.TemporaryDirectory() as tmp:
            Lfm3AudioForConditionalGeneration(self.config).save_pretrained(tmp)
            offloaded = Lfm3AudioForConditionalGeneration.from_pretrained(
                tmp,
                device_map={
                    "model.language_model": torch_device,
                    "lm_head": torch_device,
                    "model.depth_decoder": "cpu",
                    "model.codebook_embedding": "cpu",
                    "model.audio_encoder": "cpu",
                    "model.audio_projector": "cpu",
                },
            ).eval()
            # an offloaded language model keeps its embedding table on `meta` outside the hooked forward
            lm_offloaded = Lfm3AudioForConditionalGeneration.from_pretrained(
                tmp,
                device_map={
                    "model.language_model": "cpu",
                    "lm_head": "cpu",
                    "model.depth_decoder": torch_device,
                    "model.codebook_embedding": torch_device,
                    "model.audio_encoder": torch_device,
                    "model.audio_projector": torch_device,
                },
            ).eval()
        speech_paths(offloaded, torch_device)
        speech_paths(lm_offloaded, torch_device)


class Lfm3AudioModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 2
        self.seq_length = 12
        self.is_training = True
        self.num_hidden_layers = 2
        self.hidden_size = 32
        self.num_attention_heads = 4
        self.vocab_size = 256

    def get_config(self):
        text_config = tiny_config().text_config.to_dict()
        text_config.update(num_hidden_layers=2, layer_types=["conv", "full_attention"])
        config = tiny_config()
        return Lfm3AudioConfig(**{**config.to_dict(), "text_config": text_config})

    def prepare_config_and_inputs_for_common(self):
        # text-only inputs: the common tests pass `input_ids` without the speech features, which the model rejects
        # for placeholders; the merged speech paths are covered by the behaviour tests
        input_ids = torch.randint(3, 200, (self.batch_size, self.seq_length), device=torch_device)
        input_ids[:, 0] = 1
        return self.get_config(), {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


@require_torch
class Lfm3AudioCommonTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """The common model and generation tests; the text/speech mode machine is covered by the behaviour tests above."""

    all_model_classes = (Lfm3AudioModel, Lfm3AudioForConditionalGeneration) if is_torch_available() else ()
    pipeline_model_mapping = {}
    _is_composite = True
    # the common inputs carry no speech; `test_every_parameter_receives_a_gradient` covers the speech parameters
    test_all_params_have_gradient = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_torch_exportable = False

    def setUp(self):
        self.model_tester = Lfm3AudioModelTester(self)

    def _get_conv_state_shape(self, batch_size: int, config):
        text_config = config.get_text_config()
        return (batch_size, text_config.hidden_size, text_config.conv_L_cache)

    def _audio_features_get_expected_num_attentions(self, model_tester=None):
        layer_types = self.model_tester.get_config().audio_encoder_config.text_config.layer_types
        return sum(layer_type != "conv" for layer_type in layer_types)

    def _audio_features_get_expected_num_hidden_states(self, model_tester=None):
        return self.model_tester.get_config().audio_encoder_config.text_config.num_hidden_layers + 1

    def _audio_features_prepare_config_and_inputs(self):
        config = self.model_tester.get_config()
        input_features = torch.randn(2, 16, 40, device=torch_device)
        feature_attention_mask = torch.ones(2, 40, dtype=torch.long, device=torch_device)
        feature_attention_mask[1, 24:] = 0
        return config, {"input_features": input_features, "feature_attention_mask": feature_attention_mask}

    @unittest.skip("LFM3-Audio generates with greedy search or sampling only")
    def test_generate_from_inputs_embeds_1_beam_search(self):
        pass

    @unittest.skip("the conv layers record no attention weights")
    def test_attention_outputs(self):
        pass

    @unittest.skip("the conv layers keep a conv state, not key/value pairs")
    def test_past_key_values_format(self):
        pass

    def test_reverse_loading_mapping(self):
        # the speech encoder and depth decoder renames live on the head's conversion mapping
        super().test_reverse_loading_mapping(skip_base_model=True)

    @unittest.skip("LFM3 keys `rope_parameters` by attention layer type, unsupported by this test")
    def test_model_rope_scaling_frequencies(self):
        pass

    @unittest.skip("LFM3 keys `rope_parameters` by attention layer type, unsupported by this test")
    def test_model_rope_scaling_from_config_0_linear(self):
        pass

    @unittest.skip("LFM3 keys `rope_parameters` by attention layer type, unsupported by this test")
    def test_model_rope_scaling_from_config_1_dynamic(self):
        pass

    @unittest.skip("LFM3 keys `rope_parameters` by attention layer type, unsupported by this test")
    def test_model_rope_scaling_from_config_2_yarn(self):
        pass

    @unittest.skip("LFM3-Audio generates with greedy search or sampling only")
    def test_beam_search_generate(self):
        pass

    @unittest.skip("LFM3-Audio generates with greedy search or sampling only")
    def test_beam_search_generate_dict_output(self):
        pass

    @unittest.skip("LFM3-Audio generates with greedy search or sampling only")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("LFM3-Audio generates with greedy search or sampling only")
    def test_beam_sample_generate(self):
        pass

    @unittest.skip("LFM3-Audio generates with greedy search or sampling only")
    def test_beam_sample_generate_dict_output(self):
        pass

    @unittest.skip("LFM3-Audio generates with greedy search or sampling only")
    def test_constrained_beam_search_generate(self):
        pass

    @unittest.skip("LFM3-Audio generates with greedy search or sampling only")
    def test_constrained_beam_search_generate_dict_output(self):
        pass

    @unittest.skip("LFM3-Audio generates with greedy search or sampling only")
    def test_group_beam_search_generate(self):
        pass

    @unittest.skip("LFM3-Audio generates with greedy search or sampling only")
    def test_group_beam_search_generate_dict_output(self):
        pass

    @unittest.skip("LFM3-Audio generates with greedy search or sampling only")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("LFM3-Audio generates with greedy search or sampling only")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("LFM3-Audio generates with greedy search or sampling only")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip("LFM3-Audio generates with greedy search or sampling only")
    def test_assisted_decoding_matches_greedy_search_0_random(self):
        pass

    @unittest.skip("LFM3-Audio generates with greedy search or sampling only")
    def test_assisted_decoding_matches_greedy_search_1_same(self):
        pass

    @unittest.skip("LFM3-Audio generates with greedy search or sampling only")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("LFM3-Audio generates with greedy search or sampling only")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @unittest.skip("LFM3-Audio generates with greedy search or sampling only")
    def test_dola_decoding_sample(self):
        pass

    @unittest.skip("LFM3-Audio generates with greedy search or sampling only")
    def test_assisted_decoding_in_different_gpu(self):
        pass

    @unittest.skip("LFM3-Audio generates with greedy search or sampling only")
    def test_assisted_decoding_model_in_gpu_assistant_in_cpu(self):
        pass


@slow
@require_torch_accelerator
class Lfm3AudioLiveTranscriptionIntegrationTest(unittest.TestCase):
    model_id = "LiquidAI/LFM3-Audio-2B-Preview"
    revision = "7cc8c1219c97ffa9ea2967917223c7efef75bb7a"
    # Chunk texts from the pinned LFM3-Audio-2B checkpoint, fed 100 ms of audio at a time.
    expected = [
        [
            "",
            "Mr. Quilter is the",
            " apostle of the",
            " middle classes, and",
            " we're glad to",
            " welcome his",
            " gospel.",
        ],
        ["", "Nor is Mr.", " Quilter's manner", " less interesting", " than his matter.", ""],
    ]

    def test_streamed_chunks_match_the_reference_implementation(self):
        from datasets import load_dataset

        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
        model = Lfm3AudioForConditionalGeneration.from_pretrained(
            self.model_id, revision=self.revision, dtype=torch.bfloat16, device_map=torch_device
        ).eval()
        tokenizer = processor.tokenizer
        chunk_frames = 8 * model.config.interleaved_n_audio
        prompt = tokenizer(
            "<|startoftext|><|im_start|>system\nPerform live ASR.<|im_end|>\n", add_special_tokens=False
        )
        prompt = prompt.input_ids + tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
        prompt = torch.tensor([prompt], device=model.device)
        suppress = tokenizer.convert_tokens_to_ids(["<|mixed_end|>", "<|endoftext|>"])
        stop = tokenizer.convert_tokens_to_ids(["<|endoftext|>", "<|im_start|>"])
        clips = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        for clip, expected in zip(clips, self.expected):
            waveform = torch.tensor(clip["audio"]["array"], dtype=torch.float32)
            self.assertEqual(clip["audio"]["sampling_rate"], 16000)
            state, feature_cache, texts = None, None, []
            mel = torch.empty((1, processor.feature_extractor.feature_size, 0), device=model.device)

            def transcribe(frames, is_final=False):
                nonlocal state
                output = model.transcribe_live(
                    frames,
                    state,
                    input_ids=prompt if state is None else None,
                    is_final=is_final,
                    max_new_tokens=96 if is_final else 48,
                    suppress_token_ids=suppress,
                    stop_token_ids=stop,
                )
                state = output.state
                texts.append(tokenizer.decode(output.sequences[0], skip_special_tokens=True))

            for start in range(0, waveform.numel(), 1600):
                features, feature_cache = processor.feature_extractor.stream(
                    waveform[start : start + 1600], feature_cache
                )
                mel = torch.cat((mel, features.input_features.to(model.device)), dim=-1)
                while mel.shape[-1] >= chunk_frames:
                    transcribe(mel[..., :chunk_frames])
                    mel = mel[..., chunk_frames:]
            features, feature_cache = processor.feature_extractor.stream([], feature_cache, is_final=True)
            mel = torch.cat((mel, features.input_features.to(model.device)), dim=-1)
            while mel.shape[-1] > chunk_frames:
                transcribe(mel[..., :chunk_frames])
                mel = mel[..., chunk_frames:]
            transcribe(mel, is_final=True)
            self.assertEqual(texts, expected)


@slow
@require_torch_accelerator
class Lfm3AudioIntegrationTest(unittest.TestCase):
    """Exact outputs of LiquidAI/LFM3-Audio-2B-Preview, recorded 2026-10-27 on an NVIDIA B300 (torch 2.12.0+cu130, transformers 5.21.0.dev0) with the refactored tree (lfm3/stack @ a6b781c242) by pr11-review-plan/goldens/golden_audio.py; OLD = the pre-refactor PR #11 code merged on upstream main (9a5bed21c9), compared in goldens/REPORT.md.
    The checkpoint is the released training export @ `revision` converted once to the canonical layout by
    pr11-review-plan/hub_rewrites/H4/convert_audio_export.py (config + weight names; the tensors are unchanged): point
    `model_id` at the converted directory, or at the Hub once the H4 upload lands (then pin that revision here).
    Audio input: clip 0 of hf-internal-testing/librispeech_asr_dummy (16 kHz), as the live-ASR test above.
    Greedy everywhere: `do_sample=False` for the text and `depth_decoder_do_sample=False` for the speech frames (the
    OLD implementation's `audio_temperature=0.0`), the depth repetition penalty 1.1 of the released generation config
    kept. Values equal to the OLD implementation bitwise: the bf16 text reply, the fp32 speech frames and the bf16
    live-transcription tokens. The bf16 speech frames are NOT pinned: they equal OLD's for the first five frames and
    diverge from the sixth on (the refactor keeps RoPE cos/sin in fp32, Arthur 288795; one argmax flip cascades
    through the MaskGIT passes and the frames after it), which is why the frame test runs in fp32."""

    model_id = "LiquidAI/LFM3-Audio-2B-Preview"
    revision = "0ae9d6093fc7435c764d16fbea0a7f73c10e1fd8"

    @classmethod
    def _clip(cls):
        from datasets import load_dataset

        clip = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")[0]
        assert clip["audio"]["sampling_rate"] == 16000
        return np.asarray(clip["audio"]["array"], dtype=np.float32)

    def _load(self, dtype):
        processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
        model = Lfm3AudioForConditionalGeneration.from_pretrained(
            self.model_id, revision=self.revision, dtype=dtype, device_map=torch_device
        ).eval()
        return processor, model

    def test_text_reply_to_audio_matches_the_recorded_ids(self):
        processor, model = self._load(torch.bfloat16)
        conversation = [
            {"role": "system", "content": "Perform ASR in English."},
            {"role": "user", "content": [{"type": "audio"}]},
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audio=[self._clip()], sampling_rate=16000, return_tensors="pt").to(torch_device)
        output = model.generate(
            **inputs, max_new_tokens=48, do_sample=False, generation_mode="text", return_dict_in_generate=True
        )
        generated = output.sequences[0, inputs["input_ids"].shape[1] :].tolist()
        self.assertEqual(generated, Expectations({(None, None): LFM3_AUDIO_GOLDEN_ASR_IDS}).get_expectation())
        self.assertEqual(
            processor.tokenizer.decode(generated, skip_special_tokens=True),
            "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.",
        )

    def test_speech_frames_match_the_recorded_codes(self):
        processor, model = self._load(torch.float32)
        conversation = [
            {"role": "system", "content": "Perform TTS in American English using a female voice."},
            {"role": "user", "content": "The quick brown fox jumps over the lazy dog."},
            {"role": "assistant", "content": "<|audio_start|>"},
        ]
        inputs = processor.apply_chat_template(
            conversation, continue_final_message=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)
        torch.manual_seed(0)  # the order the MaskGIT passes reveal the codebooks in
        output = model.generate(
            **inputs, max_new_tokens=24, do_sample=False, depth_decoder_do_sample=False, return_dict_in_generate=True
        )
        self.assertEqual(output.audio_lengths.tolist(), [24])
        self.assertEqual(
            output.sequences[0, inputs["input_ids"].shape[1] :].tolist(), [model.config.audio_out_token_id] * 24
        )
        expected = Expectations({(None, None): LFM3_AUDIO_GOLDEN_TTS_CODES}).get_expectation()
        self.assertEqual(output.audio_codes[0].tolist(), expected)

    def test_live_transcription_matches_the_recorded_ids(self):
        processor, model = self._load(torch.bfloat16)
        tokenizer = processor.tokenizer
        prompt = tokenizer(
            "<|startoftext|><|im_start|>system\nPerform live ASR.<|im_end|>\n", add_special_tokens=False
        )
        prompt = prompt.input_ids + tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
        prompt = torch.tensor([prompt], device=torch_device)
        suppress = tokenizer.convert_tokens_to_ids(["<|mixed_end|>", "<|endoftext|>"])
        stop = tokenizer.convert_tokens_to_ids(["<|endoftext|>", "<|im_start|>"])
        # the first 3 s of the clip: three chunks of 8 * interleaved_n_audio mel frames, then the final flush
        mel = processor.feature_extractor(self._clip()[:48000], sampling_rate=16000, return_tensors="pt")
        mel = mel.input_features.to(torch_device)
        chunk = 8 * model.config.interleaved_n_audio
        state, tokens = None, []
        for start in range(0, 3 * chunk, chunk):
            output = model.transcribe_live(
                mel[..., start : start + chunk],
                state,
                input_ids=prompt if state is None else None,
                max_new_tokens=48,
                suppress_token_ids=suppress,
                stop_token_ids=stop,
            )
            state, tokens = output.state, tokens + [output.sequences[0].tolist()]
        output = model.transcribe_live(
            mel[..., 3 * chunk :],
            state,
            is_final=True,
            max_new_tokens=96,
            suppress_token_ids=suppress,
            stop_token_ids=stop,
        )
        tokens.append(output.sequences[0].tolist())
        self.assertEqual(tokens, Expectations({(None, None): LFM3_AUDIO_GOLDEN_LIVE_TOKENS}).get_expectation())
        self.assertEqual(
            [tokenizer.decode(t, skip_special_tokens=True) for t in tokens],
            ["", "Mr. Quilter is the", " apostle of the", " middle class."],
        )


LFM3_AUDIO_GOLDEN_ASR_IDS = [
    22054,
    22,
    2767,
    7162,
    355,
    278,
    22539,
    298,
    302,
    278,
    6232,
    7549,
    20,
    309,
    522,
    437,
    26007,
    310,
    18902,
    709,
    37526,
    22,
    124900,
]
LFM3_AUDIO_GOLDEN_TTS_CODES = [
    [
        647,
        583,
        919,
        1391,
        1879,
        1426,
        1165,
        1212,
        488,
        853,
        1405,
        1460,
        68,
        688,
        1675,
        2003,
        1497,
        1414,
        1799,
        520,
        1759,
        1521,
        1181,
        1412,
    ],
    [
        603,
        204,
        1586,
        1128,
        152,
        1507,
        673,
        480,
        770,
        1227,
        1546,
        1323,
        539,
        568,
        785,
        515,
        554,
        880,
        544,
        1198,
        570,
        1530,
        1132,
        920,
    ],
    [
        1912,
        960,
        632,
        1565,
        1609,
        1594,
        1958,
        1865,
        432,
        1210,
        785,
        874,
        1432,
        690,
        1137,
        1185,
        1264,
        1491,
        1836,
        888,
        253,
        914,
        1536,
        1388,
    ],
    [
        689,
        688,
        1674,
        1655,
        575,
        21,
        1599,
        974,
        1100,
        759,
        410,
        577,
        574,
        510,
        350,
        237,
        766,
        1107,
        932,
        990,
        1558,
        122,
        1622,
        706,
    ],
    [
        1200,
        818,
        1074,
        1358,
        1271,
        1515,
        1438,
        1215,
        363,
        1377,
        743,
        1366,
        997,
        741,
        1511,
        1007,
        1089,
        961,
        188,
        587,
        1863,
        1383,
        1128,
        392,
    ],
    [
        399,
        398,
        883,
        1558,
        1838,
        1584,
        1823,
        1735,
        1886,
        834,
        1147,
        41,
        825,
        1870,
        988,
        543,
        1894,
        1673,
        1553,
        1499,
        1922,
        770,
        836,
        1545,
    ],
    [
        147,
        258,
        1826,
        1398,
        1832,
        1706,
        51,
        1881,
        1506,
        1735,
        1989,
        109,
        174,
        153,
        1825,
        65,
        1785,
        1737,
        426,
        498,
        1497,
        161,
        207,
        1858,
    ],
    [
        1622,
        1997,
        568,
        433,
        1276,
        507,
        101,
        950,
        633,
        1484,
        176,
        235,
        1318,
        993,
        1200,
        1491,
        1305,
        2011,
        1564,
        994,
        248,
        233,
        649,
        1553,
    ],
    [
        1080,
        1425,
        910,
        1468,
        1482,
        1934,
        736,
        1469,
        1185,
        685,
        125,
        916,
        1043,
        1691,
        1907,
        1862,
        854,
        593,
        1402,
        784,
        364,
        1515,
        849,
        1812,
    ],
    [
        1044,
        755,
        801,
        700,
        1180,
        1982,
        887,
        1084,
        1413,
        784,
        1372,
        392,
        1887,
        1530,
        1409,
        1483,
        1150,
        1271,
        687,
        1374,
        473,
        337,
        1879,
        911,
    ],
    [
        1460,
        1130,
        1926,
        1570,
        1350,
        503,
        1178,
        788,
        1630,
        1470,
        1436,
        1473,
        1466,
        1759,
        86,
        419,
        1574,
        1907,
        1401,
        622,
        1580,
        1402,
        957,
        1806,
    ],
    [
        1753,
        1769,
        641,
        497,
        1081,
        747,
        459,
        1417,
        1896,
        197,
        612,
        1227,
        1563,
        986,
        1678,
        1870,
        1786,
        1452,
        1508,
        1385,
        540,
        939,
        780,
        1556,
    ],
]
LFM3_AUDIO_GOLDEN_LIVE_TOKENS = [
    [],
    [22054, 22, 2767, 7162, 355, 278],
    [22539, 298, 302, 278],
    [125016, 6232, 1014, 22],
]
