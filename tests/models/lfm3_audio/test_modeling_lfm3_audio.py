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

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from transformers import (
    AutoProcessor,
    Lfm3AudioConfig,
    Lfm3AudioForConditionalGeneration,
    Lfm3AudioModel,
    LogitsProcessor,
    LogitsProcessorList,
    is_torch_available,
)
from transformers.generation.streamers import BaseStreamer
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...alm_tester import ALMModelTest, ALMModelTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin


if is_torch_available():
    import torch


AUDIO_TOKEN_ID, AUDIO_OUT_TOKEN_ID, AUDIO_START_ID, TEXT_END_ID, MIXED_START_ID = 250, 251, 252, 253, 254
NUM_CODEBOOKS, CODEBOOK_SIZE = 4, 7
# never samples the markers that would switch a row to speech
TEXT_ONLY = {"suppress_tokens": [AUDIO_START_ID, MIXED_START_ID]}


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


def frames_of(code):
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
            replay = model.generate(
                prefix.sequences,
                audio_out_codes=prefix.audio_codes[0].T,
                max_new_tokens=1,
                logits_processor=force_tokens(10),
            )
            torch.testing.assert_close(replay.sequences, full.sequences)
            sampler.assert_not_called()
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
        self.assertFalse(torch.allclose(logits[:, 0], logits_changed[:, 0]))

    def test_training_losses(self):
        labels = self.inputs["input_ids"].clone()
        labels[labels >= AUDIO_TOKEN_ID] = -100
        audio_labels = self.inputs["audio_out_codes"].clone()
        audio_labels[0, -1] = CODEBOOK_SIZE - 1
        outputs = self.model(**self.inputs, labels=labels, audio_labels=audio_labels)
        self.assertEqual(outputs.loss.shape, ())
        text_count = (labels[:, 1:] != -100).sum()
        audio_count = (audio_labels != -100).all(-1).sum()
        expected = (text_count * outputs.text_loss + audio_count * outputs.audio_loss) / (
            text_count + audio_count + 1e-6
        )
        torch.testing.assert_close(outputs.loss, expected)
        with torch.no_grad():
            speech_only = self.model(**self.inputs, labels=torch.full_like(labels, -100), audio_labels=audio_labels)
            self.assertEqual(speech_only.text_loss.item(), 0.0)
            torch.testing.assert_close(speech_only.loss, speech_only.audio_loss)

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
        self.assertFalse(depth_config.do_sample)
        text_top_k = self.model.generation_config.top_k
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=6,
            depth_decoder_do_sample=True,
            depth_decoder_top_k=3,
            depth_decoder_repetition_penalty=1.3,
            do_sample=False,
        )
        self.assertEqual((depth_config.do_sample, depth_config.top_k), (True, 3))
        self.assertEqual(self.model.generation_config.top_k, text_top_k)
        new_tokens = outputs.sequences[0, inputs["input_ids"].shape[1] :]
        self.assertEqual(int((new_tokens == AUDIO_OUT_TOKEN_ID).sum()), outputs.audio_codes.shape[-1])
        self.assertEqual(outputs.audio_lengths.item(), outputs.audio_codes.shape[-1])
        codes = outputs.audio_codes[0]
        terminal = (codes == CODEBOOK_SIZE - 1).any(0)
        self.assertTrue(bool((codes[:, ~terminal] < CODEBOOK_SIZE - 1).all()))
        self.assertTrue(bool((codes[:, terminal] == CODEBOOK_SIZE - 1).all()))
        with tempfile.TemporaryDirectory() as tmp:
            self.model.save_pretrained(tmp)
            with open(os.path.join(tmp, "generation_config.json"), encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual((saved["depth_decoder_top_k"], saved["depth_decoder_repetition_window"]), (3, 64))
            reloaded = Lfm3AudioForConditionalGeneration.from_pretrained(tmp)
        self.assertEqual(reloaded.depth_decoder.generation_config.top_k, 3)
        self.assertFalse(hasattr(reloaded.generation_config, "depth_decoder_top_k"))

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
        schedule = [1, 1, 2]  # the default passes for four codebooks
        self.assertEqual(len(seen), len(schedule))
        self.assertEqual(codes.shape, (1, NUM_CODEBOOKS))
        mask = self.model.model.depth_decoder.mask_embedding
        depth_in = self.model.model.depth_decoder.depth_linear(hidden[0]).unflatten(-1, (NUM_CODEBOOKS, 16))
        hidden_counts = [int(torch.isclose(x - depth_in, mask).all(-1).sum()) for x in seen]
        self.assertEqual(hidden_counts[0], NUM_CODEBOOKS)
        self.assertEqual(hidden_counts[1], NUM_CODEBOOKS - schedule[0])

    def test_generate_interleaved_mode_alternates_blocks(self):
        config = tiny_config()
        torch.manual_seed(0)
        model = Lfm3AudioForConditionalGeneration(config).to(torch_device).eval()
        model.generation_config.update(interleaved_n_text=2, interleaved_n_audio=3, allow_custom_entries=True)
        inputs = dict(self.inputs)
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], torch.tensor([[MIXED_START_ID]], device=torch_device)], 1
        )
        with patch.object(model, "sample_audio_frames", side_effect=frames_of(0)):
            outputs = model.generate(
                **inputs, max_new_tokens=10, logits_processor=force_tokens(*([10] * 10)), return_dict_in_generate=True
            )
        new_tokens = outputs.sequences[0, inputs["input_ids"].shape[1] :]
        is_frame = (new_tokens == AUDIO_OUT_TOKEN_ID).tolist()
        self.assertEqual(is_frame, [False, False, True, True, True, False, False, True, True, True])

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

    def test_batch_has_independent_modes_eos_lengths_and_streamers(self):
        ids = torch.tensor([[1, 10], [1, AUDIO_START_ID], [1, MIXED_START_ID]], device=torch_device)
        streamer, audio_streamer = Collect(), Collect()
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
            outputs = self.model.generate(
                ids, max_new_tokens=4, streamer=streamer, audio_streamer=audio_streamer, logits_processor=processor
            )
        self.assertEqual(outputs.sequences[0].tolist(), [1, 10, 2, 0, 0, 0])
        self.assertEqual(outputs.audio_lengths.tolist(), [0, 2, 3])
        self.assertTrue(bool((outputs.audio_codes[0] == -1).all()))
        self.assertTrue(bool((outputs.audio_codes[1, :, 1] == CODEBOOK_SIZE - 1).all()))
        self.assertEqual(calls[1], torch.Size([2, 32]))
        self.assertTrue(streamer.ended)
        self.assertEqual(torch.stack(streamer.items[1:], 1).tolist(), outputs.sequences[:, 2:].tolist())
        self.assertTrue(audio_streamer.ended)
        streamed = torch.stack(audio_streamer.items, -1)
        for row, length in enumerate(outputs.audio_lengths.tolist()):
            frames = streamed[row][:, streamed[row, 0] != -1]
            self.assertEqual(frames.tolist(), outputs.audio_codes[row, :, :length].tolist())

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
                    self.assertLessEqual(layer.keys.shape[-2], encoder.config.text_config.sliding_window)

    def test_cached_generation_matches_full_prompt_in_the_speech_modes(self):
        prefix = torch.tensor([[0, 1, 4], [1, 5, 6]], device=torch_device)
        prefix_mask = prefix.ne(0).long()
        for last_token in (AUDIO_START_ID, MIXED_START_ID):
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
        # every streamed token must be the greedy choice of an offline forward over the layout it produced
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
                # the placeholders would win every step, but are never sampled as text
                logits[..., [AUDIO_TOKEN_ID, AUDIO_OUT_TOKEN_ID]] += 50.0
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
                row[[AUDIO_TOKEN_ID, AUDIO_OUT_TOKEN_ID]] = float("-inf")
                if not final:
                    row[suppressed] = float("-inf")
                self.assertEqual(int(row.argmax()), token)

    def test_placeholder_ids_are_never_sampled_as_text(self):
        prompt = torch.tensor([[1, 10, 11]], device=torch_device)
        for boosted_id in (AUDIO_TOKEN_ID, AUDIO_OUT_TOKEN_ID):
            with self.subTest(boosted_id=boosted_id), torch.no_grad():
                boost = LogitsProcessorList([Boost(lambda step, row: boosted_id)])
                outputs = self.model.generate(
                    prompt, max_new_tokens=3, do_sample=True, logits_processor=boost, **TEXT_ONLY
                )
                self.assertNotIn(boosted_id, outputs.sequences[0, 3:].tolist())
                self.assertEqual(outputs.audio_lengths.tolist(), [0])


class Lfm3AudioModelTester(ALMModelTester):
    config_class = Lfm3AudioConfig
    base_model_class = Lfm3AudioModel
    conditional_generation_class = Lfm3AudioForConditionalGeneration
    audio_mask_key = "feature_attention_mask"

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("vocab_size", 256)
        kwargs.setdefault("head_dim", 8)
        kwargs.setdefault("num_key_value_heads", 1)
        kwargs.setdefault("conv_L_cache", 3)
        kwargs.setdefault("feat_seq_length", 40)
        kwargs.setdefault("num_mel_bins", 16)
        # a window past every prompt and every clip: the SDPA flash backend takes no explicit mask
        kwargs.setdefault("sliding_window", 4096)
        # two attention layers: the `num_hidden_layers` the common attention-output test expects
        kwargs.setdefault("text_layer_types", ["conv", "sliding_attention", "full_attention"])
        kwargs.setdefault("text_num_hidden_layers", 3)
        kwargs.setdefault("expected_num_hidden_layers", 4)
        kwargs.setdefault("audio_encoder_conv_channels", 8)
        kwargs.setdefault("audio_encoder_text_hidden_size", 16)
        kwargs.setdefault("audio_encoder_text_num_hidden_layers", 3)
        kwargs.setdefault("audio_encoder_text_layer_types", ["conv", "sliding_attention", "conv"])
        kwargs.setdefault("depth_hidden_size", 16)
        kwargs.setdefault("depth_max_position_embeddings", NUM_CODEBOOKS)
        kwargs.setdefault("depth_layer_types", ["full_attention"] * 2)
        kwargs.setdefault(
            "depth_rope_parameters",
            {"full_attention": {"rope_type": "default", "rope_theta": 1e6, "partial_rotary_factor": 1.0}},
        )
        kwargs.setdefault("num_codebooks", NUM_CODEBOOKS)
        kwargs.setdefault("codebook_size", CODEBOOK_SIZE)
        kwargs.setdefault("audio_eos_code", CODEBOOK_SIZE - 1)
        kwargs.setdefault("audio_token_id", AUDIO_TOKEN_ID)
        kwargs.setdefault("audio_out_token_id", AUDIO_OUT_TOKEN_ID)
        kwargs.setdefault("audio_start_token_id", AUDIO_START_ID)
        kwargs.setdefault("text_end_token_id", TEXT_END_ID)
        kwargs.setdefault("mixed_start_token_id", MIXED_START_ID)
        super().__init__(parent, **kwargs)

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {AUDIO_OUT_TOKEN_ID, AUDIO_START_ID, TEXT_END_ID, MIXED_START_ID}

    def get_audio_embeds_mask(self, audio_mask):
        lengths = -(audio_mask.sum(-1) // -8)
        return (torch.arange(int(lengths.max()), device=audio_mask.device)[None] < lengths[:, None]).long()

    def get_config(self):
        config = super().get_config()
        # the common audio-feature test reads the encoder width from `audio_config.hidden_size`
        config.audio_encoder_config.hidden_size = config.audio_encoder_config.text_config.hidden_size
        return config


@require_torch
class Lfm3AudioCommonTest(ALMModelTest, unittest.TestCase):
    """The common model and generation tests; the text/speech mode machine is covered by the behaviour tests above."""

    model_tester_class = Lfm3AudioModelTester
    test_resize_embeddings = False

    def setUp(self):
        super().setUp()
        # `generate` only runs greedy search or sampling, like Dia
        if any(mode in self._testMethodName for mode in ("beam", "assisted_decoding", "prompt_lookup")):
            self.skipTest("LFM3-Audio generates with greedy search or sampling only")

    def test_config(self):
        # `text_config` and `depth_config` are both LFM3 configs: `Lfm3Config.from_pretrained` reads back the first
        with patch.object(self.config_tester, "create_and_test_config_from_and_save_pretrained_composite"):
            self.config_tester.run_common_tests()

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if return_labels and model_class is Lfm3AudioForConditionalGeneration:
            # one end-of-audio frame per row reaches the speech embeddings and the depth decoder; every codebook of
            # such a frame is masked, so the loss draws no random mask and stays deterministic
            inputs_dict["input_ids"] = inputs_dict["input_ids"].clone()
            inputs_dict["input_ids"][:, -1] = AUDIO_OUT_TOKEN_ID
            frames = torch.full(
                (inputs_dict["input_ids"].shape[0], NUM_CODEBOOKS), CODEBOOK_SIZE - 1, device=torch_device
            )
            inputs_dict.update(audio_out_codes=frames, audio_labels=frames)
        return inputs_dict

    def _get_conv_state_shape(self, batch_size: int, config):
        text_config = config.get_text_config()
        return (batch_size, text_config.hidden_size, text_config.conv_L_cache)

    def _audio_features_get_expected_num_attentions(self, model_tester=None):
        return self.model_tester.audio_encoder_text_layer_types.count("sliding_attention")

    def _audio_features_get_expected_num_hidden_states(self, model_tester=None):
        return self.model_tester.audio_encoder_text_num_hidden_layers + 1

    def test_reverse_loading_mapping(self):
        # the speech encoder and depth decoder renames live on the head's conversion mapping
        super().test_reverse_loading_mapping(skip_base_model=True)


@slow
@require_torch_accelerator
class Lfm3AudioIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    """Greedy outputs of LFM3-Audio-2B on the first librispeech dummy clip."""

    model_id = "LiquidAI/LFM3-Audio-2B"

    @classmethod
    def _clip(cls):
        from datasets import load_dataset

        clip = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")[0]
        assert clip["audio"]["sampling_rate"] == 16000
        return np.asarray(clip["audio"]["array"], dtype=np.float32)

    def _load(self, dtype):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = Lfm3AudioForConditionalGeneration.from_pretrained(
            self.model_id, dtype=dtype, device_map=torch_device
        ).eval()
        return processor, model

    def test_text_reply_to_audio_matches_the_recorded_ids(self):
        processor, model = self._load(torch.bfloat16)
        conversation = [
            {"role": "system", "content": "Perform ASR in English."},
            {"role": "user", "content": [{"type": "audio"}]},
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audio=[self._clip()], sampling_rate=16000, return_tensors="pt").to(
            torch_device, torch.bfloat16
        )
        output = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
            suppress_tokens=[model.config.audio_start_token_id, model.config.mixed_start_token_id],
            return_dict_in_generate=True,
        )
        generated = output.sequences[0, inputs["input_ids"].shape[1] :].tolist()
        expected = Expectations(
            {
                (None, None): [
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
            }
        ).get_expectation()
        self.assertEqual(generated, expected)
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
        # fmt: off
        expected = Expectations(
            {
                (None, None): [
                    [647, 583, 919, 1391, 1879, 1426, 1165, 1212, 488, 853, 1405, 1460,
                     68, 688, 1675, 2003, 1497, 1414, 1799, 520, 1759, 1521, 1181, 1412],
                    [603, 204, 1586, 1128, 152, 1507, 673, 480, 770, 1227, 1546, 1323,
                     539, 568, 785, 515, 554, 880, 544, 1198, 570, 1530, 1132, 920],
                    [1912, 960, 632, 1565, 1609, 1594, 1958, 1865, 432, 1210, 785, 874,
                     1432, 690, 1137, 1185, 1264, 1491, 1836, 888, 253, 914, 1536, 1388],
                    [689, 688, 1674, 1655, 575, 21, 1599, 974, 1100, 759, 410, 577,
                     574, 510, 350, 237, 766, 1107, 932, 990, 1558, 122, 1622, 706],
                    [1200, 818, 1074, 1358, 1271, 1515, 1438, 1215, 363, 1377, 743, 1366,
                     997, 741, 1511, 1007, 1089, 961, 188, 587, 1863, 1383, 1128, 392],
                    [399, 398, 883, 1558, 1838, 1584, 1823, 1735, 1886, 834, 1147, 41,
                     825, 1870, 988, 543, 1894, 1673, 1553, 1499, 1922, 770, 836, 1545],
                    [147, 258, 1826, 1398, 1832, 1706, 51, 1881, 1506, 1735, 1989, 109,
                     174, 153, 1825, 65, 1785, 1737, 426, 498, 1497, 161, 207, 1858],
                    [1622, 1997, 568, 433, 1276, 507, 101, 950, 633, 1484, 176, 235,
                     1318, 993, 1200, 1491, 1305, 2011, 1564, 994, 248, 233, 649, 1553],
                    [1080, 1425, 910, 1468, 1482, 1934, 736, 1469, 1185, 685, 125, 916,
                     1043, 1691, 1907, 1862, 854, 593, 1402, 784, 364, 1515, 849, 1812],
                    [1044, 755, 801, 700, 1180, 1982, 887, 1084, 1413, 784, 1372, 392,
                     1887, 1530, 1409, 1483, 1150, 1271, 687, 1374, 473, 337, 1879, 911],
                    [1460, 1130, 1926, 1570, 1350, 503, 1178, 788, 1630, 1470, 1436, 1473,
                     1466, 1759, 86, 419, 1574, 1907, 1401, 622, 1580, 1402, 957, 1806],
                    [1753, 1769, 641, 497, 1081, 747, 459, 1417, 1896, 197, 612, 1227,
                     1563, 986, 1678, 1870, 1786, 1452, 1508, 1385, 540, 939, 780, 1556],
                ],
            }
        ).get_expectation()
        # fmt: on
        self.assertEqual(output.audio_codes[0].tolist(), expected)

    def test_streamed_chunks_match_the_reference_implementation(self):
        # chunk texts from the pinned checkpoint, fed 100 ms of audio at a time
        expected_texts = [
            [
                "",
                "Mr. Quilter is the",
                " apostle of the",
                " middle classes, and",
                " we're glad to",
                " welcome his",
                " gospel.",
            ],
            ["", "Nor is Mr.", " Quilter's manner", " less interesting", " than his matter,", ""],
        ]
        from datasets import load_dataset

        processor, model = self._load(torch.bfloat16)
        tokenizer = processor.tokenizer
        chunk_frames = 8 * model.generation_config.interleaved_n_audio
        prompt = tokenizer(
            "<|startoftext|><|im_start|>system\nPerform live ASR.<|im_end|>\n", add_special_tokens=False
        )
        prompt = prompt.input_ids + tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
        prompt = torch.tensor([prompt], device=model.device)
        suppress = tokenizer.convert_tokens_to_ids(["<|mixed_end|>", "<|endoftext|>"])
        stop = tokenizer.convert_tokens_to_ids(["<|endoftext|>", "<|im_start|>"])
        clips = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        for clip, expected in zip(clips, expected_texts):
            waveform = torch.tensor(clip["audio"]["array"], dtype=torch.float32)
            self.assertEqual(clip["audio"]["sampling_rate"], 16000)
            state, feature_cache, texts = None, None, []
            mel = torch.empty((1, processor.feature_extractor.feature_size, 0), device=model.device, dtype=model.dtype)

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
                mel = torch.cat((mel, features.input_features.to(model.device, model.dtype)), dim=-1)
                while mel.shape[-1] >= chunk_frames:
                    transcribe(mel[..., :chunk_frames])
                    mel = mel[..., chunk_frames:]
            features, feature_cache = processor.feature_extractor.stream([], feature_cache, is_final=True)
            mel = torch.cat((mel, features.input_features.to(model.device, model.dtype)), dim=-1)
            while mel.shape[-1] > chunk_frames:
                transcribe(mel[..., :chunk_frames])
                mel = mel[..., chunk_frames:]
            transcribe(mel, is_final=True)
            self.assertEqual(texts, expected)
