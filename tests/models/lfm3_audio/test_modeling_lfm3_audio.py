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

import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    Lfm3AudioConfig,
    Lfm3AudioForConditionalGeneration,
    Lfm3AudioModel,
    LogitsProcessor,
    LogitsProcessorList,
    is_torch_available,
)
from transformers.audio_utils import load_audio
from transformers.cache_utils import DynamicCache
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from ...alm_tester import ALMModelTest, ALMModelTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers.models.lfm3_audio.generation_lfm3_audio import (
        Lfm3AudioWindowedRepetitionPenaltyLogitsProcessor,
    )
    from transformers.models.lfm3_audio.modeling_lfm3_audio import Lfm3AudioEncoderCausalConvPaddingCache


class Boost(LogitsProcessor):
    """Adds `boost` to the score of token `choose(step, row)`; suppressed (`-inf`) tokens stay suppressed."""

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
    """Forces every row through `tokens` in order."""
    return LogitsProcessorList([Boost(lambda step, row: tokens[step] if step < len(tokens) else None)])


def only_code(config, code):
    """`generate` kwargs: the depth decoder can only sample `code`."""
    return {
        "depth_decoder_do_sample": True,
        "depth_decoder_suppress_tokens": [other for other in range(config.codebook_size) if other != code],
    }


class Lfm3AudioModelTester(ALMModelTester):
    config_class = Lfm3AudioConfig
    base_model_class = Lfm3AudioModel
    conditional_generation_class = Lfm3AudioForConditionalGeneration
    audio_mask_key = "input_features_mask"

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("feat_seq_length", 40)
        # 2 attention layers, as the common attention-output test expects
        kwargs.setdefault("text_layer_types", ["conv", "sliding_attention", "full_attention"])
        kwargs.setdefault("text_num_hidden_layers", 3)
        kwargs.setdefault("expected_num_hidden_layers", 4)
        kwargs.setdefault("subsampling_factor", 8)
        kwargs.setdefault("audio_num_hidden_layers", 3)
        kwargs.setdefault("audio_layer_types", ["conv", "sliding_attention", "conv"])
        # default range: encoder output ~1e-4, lost in float noise for the streaming / padding tests
        kwargs.setdefault("audio_initializer_range", 1.0)
        kwargs.setdefault("num_codebooks", 4)
        kwargs.setdefault("codebook_size", 7)
        kwargs.setdefault("audio_eos_code", 6)
        # inside the tester's vocabulary
        kwargs.setdefault("audio_out_token_id", 4)
        kwargs.setdefault("audio_start_token_id", 5)
        kwargs.setdefault("text_end_token_id", 6)
        kwargs.setdefault("mixed_start_token_id", 7)
        kwargs.setdefault("mixed_end_token_id", 8)

        super().__init__(parent, **kwargs)
        self.depth_decoder_backbone_hidden_size = self.hidden_size

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {
            self.audio_out_token_id,
            self.audio_start_token_id,
            self.text_end_token_id,
            self.mixed_start_token_id,
            self.mixed_end_token_id,
        }

    def create_audio_features(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return floats_tensor([batch_size, self.feat_seq_length, self.num_mel_bins])

    def get_audio_embeds_mask(self, audio_mask):
        return audio_mask[:, :: self.subsampling_factor]

    def prepare_config_and_inputs_for_speech(self):
        """One prompt: 2 clips (5 and 3 audio tokens), then 2 speech frames."""
        config = self.get_config()
        input_features_mask = torch.ones(2, self.feat_seq_length, dtype=torch.long, device=torch_device)
        input_features_mask[1, 24:] = 0

        input_ids = (
            [10, 11, 12]
            + [self.audio_token_id] * 5
            + [13, 14]
            + [self.audio_token_id] * 3
            + [self.audio_out_token_id] * 2
        )

        inputs_dict = {
            "input_ids": torch.tensor([input_ids + [15]], device=torch_device),
            "input_features": floats_tensor([2, self.feat_seq_length, self.num_mel_bins]),
            "input_features_mask": input_features_mask,
            "audio_input_ids": ids_tensor([2, self.num_codebooks], self.audio_eos_code),
        }

        return config, inputs_dict


@require_torch
class Lfm3AudioModelTest(ALMModelTest, unittest.TestCase):
    model_tester_class = Lfm3AudioModelTester

    def setUp(self):
        super().setUp()
        # greedy or sampling only, like Dia
        if any(mode in self._testMethodName for mode in ("beam", "assisted_decoding", "prompt_lookup")):
            self.skipTest("LFM3-Audio generates with greedy search or sampling only")

        if not getattr(self, self._testMethodName).__qualname__.startswith("Lfm3AudioModelTest."):
            # Common tests: random weights write placeholders / open speech (random frames) -> flaky comparisons.
            # Released checkpoints suppress them in `generation_config.json`; done here instead.
            tester = self.model_tester
            suppressed = [
                tester.audio_token_id,
                tester.audio_out_token_id,
                tester.audio_start_token_id,
                tester.mixed_start_token_id,
            ]

            prepare = Lfm3AudioForConditionalGeneration._prepare_generation_config

            def prepare_text_only(model, *args, **kwargs):
                generation_config, model_kwargs = prepare(model, *args, **kwargs)
                generation_config.suppress_tokens = [*(generation_config.suppress_tokens or []), *suppressed]
                return generation_config, model_kwargs

            text_only = patch.object(
                Lfm3AudioForConditionalGeneration, "_prepare_generation_config", prepare_text_only
            )
            text_only.start()
            self.addCleanup(text_only.stop)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if return_labels and model_class is Lfm3AudioForConditionalGeneration:
            # one end-of-audio frame per row: all codebooks masked, no random mask, deterministic loss
            inputs_dict["input_ids"] = inputs_dict["input_ids"].clone()
            inputs_dict["input_ids"][:, -1] = self.model_tester.audio_out_token_id
            frames = torch.full(
                (inputs_dict["input_ids"].shape[0], self.model_tester.num_codebooks),
                self.model_tester.audio_eos_code,
                device=torch_device,
            )
            inputs_dict.update(audio_input_ids=frames, audio_labels=frames)

            if inputs_dict.get("labels") is not None:
                inputs_dict["labels"] = inputs_dict["labels"].clone()
                inputs_dict["labels"][:, -1] = -100

        return inputs_dict

    def _get_conv_state_shape(self, batch_size: int, config):
        text_config = config.get_text_config()
        return (batch_size, text_config.hidden_size, text_config.conv_L_cache)

    def _audio_features_get_expected_num_attentions(self, model_tester=None):
        return self.model_tester.audio_layer_types.count("sliding_attention")

    def _audio_features_get_expected_num_hidden_states(self, model_tester=None):
        return self.model_tester.audio_num_hidden_layers + 1

    def test_reverse_loading_mapping(self):
        # released-checkpoint renames target the head's keys (`model.` prefix)
        super().test_reverse_loading_mapping(skip_base_model=True)

    def test_model_without_audio_encoder_or_depth_decoder(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_speech()
        model = Lfm3AudioForConditionalGeneration(config).to(torch_device).eval()

        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            asr, asr_info = Lfm3AudioForConditionalGeneration.from_pretrained(
                tmp, depth_decoder_config=None, output_loading_info=True
            )
            tts, tts_info = Lfm3AudioForConditionalGeneration.from_pretrained(
                tmp, audio_config=None, output_loading_info=True
            )
            _, base_info = Lfm3AudioModel.from_pretrained(tmp, output_loading_info=True)

        for info in (asr_info, tts_info, base_info):
            self.assertEqual((info["missing_keys"], info["unexpected_keys"]), (set(), set()))

        self.assertIsNone(asr.depth_decoder)
        self.assertIsNone(asr.model.embed_audio_tokens)
        self.assertIsNone(tts.model.audio_model)
        self.assertIsNone(tts.model.multi_modal_projector)

        asr = asr.to(torch_device)
        with torch.no_grad():
            text_only = {k: v for k, v in inputs_dict.items() if k != "audio_input_ids"}
            text_only["input_ids"] = text_only["input_ids"].masked_fill(
                text_only["input_ids"] == config.audio_out_token_id, 10
            )

            torch.testing.assert_close(asr(**text_only).audio_hidden_states, model(**text_only).audio_hidden_states)
            with self.assertRaisesRegex(ValueError, "depth_decoder_config=None"):
                asr(**inputs_dict)
            with self.assertRaisesRegex(ValueError, "depth_decoder_config=None"):
                asr.generate(torch.tensor([[10, config.audio_start_token_id]], device=torch_device))

        with self.assertRaisesRegex(ValueError, "audio_config=None"):
            tts.to(torch_device)(**inputs_dict)

    def test_audio_features_ignore_padding(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = Lfm3AudioModel(config).to(torch_device).eval()

        # zero biases would keep zero-padded frames at zero whatever the masking
        for name, parameter in model.audio_model.named_parameters():
            if name.endswith("bias"):
                torch.nn.init.normal_(parameter)

        features = floats_tensor([2, 57, config.audio_config.num_mel_bins])
        mask = torch.ones(2, 57, device=torch_device, dtype=torch.long)
        mask[0, 9:] = 0
        # the feature extractor pads with zeros
        features = features * mask[:, :, None]

        with torch.no_grad():
            expected = torch.cat(
                [
                    model.get_audio_features(features[:1, :9], mask[:1, :9]).pooler_output,
                    model.get_audio_features(features[1:], mask[1:]).pooler_output,
                ]
            )
            actual = model.get_audio_features(features, mask).pooler_output

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_streaming_encoder_matches_offline(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.audio_config.sliding_window = 4
        encoder = Lfm3AudioModel(config).to(torch_device).eval().audio_model
        chunk = 3 * encoder.config.subsampling_factor

        # whole chunks, then a tail of any length
        for length in (24, 25, 31, 48, 57, 72):
            features = floats_tensor([2, length, config.audio_config.num_mel_bins])

            with torch.no_grad():
                expected = encoder(features).last_hidden_state
                padding_cache, cache = (
                    Lfm3AudioEncoderCausalConvPaddingCache(),
                    DynamicCache(config=encoder.config),
                )
                actual = torch.cat(
                    [
                        encoder(
                            features[:, start : start + chunk],
                            padding_cache=padding_cache,
                            cache=cache,
                            use_cache=True,
                        ).last_hidden_state
                        for start in range(0, length, chunk)
                    ],
                    dim=1,
                )

            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            for layer in cache.layers:
                if getattr(layer, "keys", None) is not None:
                    self.assertLessEqual(layer.keys.shape[-2], encoder.config.sliding_window)

    def test_windowed_repetition_penalty(self):
        processor = Lfm3AudioWindowedRepetitionPenaltyLogitsProcessor(penalty=2.0, window=2)
        scores = torch.tensor([[1.0, -1.0, 4.0, 3.0]])
        # the oldest code falls out of the window; padding is ignored
        past = torch.tensor([[2, -1, 0, 1]])
        torch.testing.assert_close(processor(past, scores.clone()), torch.tensor([[0.5, -2.0, 4.0, 3.0]]))

    def test_generate_speech_with_depth_decoder_settings(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_speech()
        model = Lfm3AudioForConditionalGeneration(config).to(torch_device).eval()
        # as the released generation config: the placeholders are never written as text
        model.generation_config.suppress_tokens = [config.audio_token_id, config.audio_out_token_id]
        input_ids = torch.cat(
            [inputs_dict.pop("input_ids"), torch.tensor([[config.audio_start_token_id]], device=torch_device)], 1
        )

        # the `depth_decoder_*` settings reach the depth decoder: it can only sample the code they leave
        outputs = model.generate(
            input_ids, **inputs_dict, max_new_tokens=3, return_dict_in_generate=True, **only_code(config, 2)
        )
        self.assertEqual(outputs.audio_sequences.tolist(), [[[2] * 3] * config.num_codebooks])
        # per-call settings leave the model's generation config alone
        self.assertFalse(hasattr(model.generation_config, "depth_decoder_suppress_tokens"))

        outputs = model.generate(
            input_ids,
            **inputs_dict,
            max_new_tokens=6,
            depth_decoder_do_sample=True,
            depth_decoder_top_k=3,
            return_dict_in_generate=True,
        )

        # one `audio_out_token_id` per frame; a frame holding the end-of-audio code is a whole end-of-audio frame
        new_tokens = outputs.sequences[0, input_ids.shape[1] :]
        num_frames = int((outputs.audio_sequences[0, 0] != config.codebook_size).sum())
        self.assertEqual(int((new_tokens == config.audio_out_token_id).sum()), num_frames)
        codes = outputs.audio_sequences[0, :, :num_frames]
        terminal = (codes == config.audio_eos_code).any(0)
        self.assertTrue(bool((codes[:, ~terminal] < config.audio_eos_code).all()))
        self.assertTrue(bool((codes[:, terminal] == config.audio_eos_code).all()))
        self.assertEqual(
            outputs.audio_sequences_mask[0].tolist(), (~terminal).tolist() + [False] * (codes.shape[-1] - num_frames)
        )

        # each step feeds back only its new frame
        with self.assertRaisesRegex(ValueError, "use_cache=True"):
            model.generate(input_ids, **inputs_dict, max_new_tokens=1, use_cache=False)

    def test_generate_interleaved_mode_alternates_blocks(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_speech()
        config.interleaved_n_text, config.interleaved_n_audio = 2, 3
        model = Lfm3AudioForConditionalGeneration(config).to(torch_device).eval()
        input_ids = torch.cat(
            [inputs_dict.pop("input_ids"), torch.tensor([[config.mixed_start_token_id]], device=torch_device)], 1
        )

        outputs = model.generate(
            input_ids,
            **inputs_dict,
            max_new_tokens=10,
            logits_processor=force_tokens(*([10] * 10)),
            return_dict_in_generate=True,
            **only_code(config, 0),
        )

        is_audio_out_frame = (outputs.sequences[0, input_ids.shape[1] :] == config.audio_out_token_id).tolist()
        self.assertEqual(is_audio_out_frame, [False, False, True, True, True, False, False, True, True, True])

    def test_cached_generation_matches_full_prompt_in_the_speech_modes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.interleaved_n_text, config.interleaved_n_audio = 2, 2
        model = Lfm3AudioForConditionalGeneration(config).to(torch_device).eval()
        model.generation_config.return_dict_in_generate = True
        # as the released generation config: the placeholders are never written as text
        model.generation_config.suppress_tokens = [config.audio_token_id, config.audio_out_token_id]

        pad = config.pad_token_id
        # the first row is left-padded
        prefix = torch.tensor([[pad, 10, 11], [10, 12, 13]], device=torch_device)
        prefix_mask = prefix.ne(pad).long()

        for last_token in (config.audio_start_token_id, config.mixed_start_token_id):
            with self.subTest(last_token=last_token), torch.no_grad():
                suffix = prefix.new_full((2, 2), last_token)
                input_ids = torch.cat((prefix, suffix), dim=-1)
                mask = input_ids.ne(pad).long()

                torch.manual_seed(12)
                # the two text slots of a block are forced, so no row ends before its speech
                expected = model.generate(
                    input_ids, attention_mask=mask, max_new_tokens=5, logits_processor=force_tokens(10, 10)
                )

                cache = model(
                    input_ids=prefix,
                    attention_mask=prefix_mask,
                    position_ids=prefix_mask.cumsum(-1).sub(1).clamp_min(0),
                    use_cache=True,
                ).cache

                torch.manual_seed(12)
                actual = model.generate(
                    suffix,
                    cache=cache,
                    attention_mask=mask,
                    max_new_tokens=5,
                    logits_processor=force_tokens(10, 10),
                )

                torch.testing.assert_close(actual.sequences[:, 2:], expected.sequences[:, 5:])
                self.assertGreater(expected.audio_sequences.shape[-1], 0)
                torch.testing.assert_close(actual.audio_sequences, expected.audio_sequences)
                torch.testing.assert_close(actual.audio_sequences_mask, expected.audio_sequences_mask)


@require_torch
class Lfm3AudioIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    """
    Goldens: Hub remote code of LiquidAI/LFM3-Audio-2B @ 2d6c1f99. Token ids, speech codes, training losses.
    reproducers: https://gist.github.com/eustlb/1befc95bf557e88c808d61c9f7b199d8

    Speech codes: greedy (`depth_decoder_do_sample=False`) + seed. Sampled ones can't match: reference draws per
    codebook, ours per reveal pass.
    Speech and live transcription in float32, not the checkpoint's bfloat16: there a few conv outputs land 1 step
    apart, and one flipped code or token changes the rest.
    """

    # every generation ends before `max_new_tokens`, except row 2 of `test_asr_batch`
    checkpoint = "eustlb/LFM3-Audio-2B"
    revision = "refs/pr/1"
    samples = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/"

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parent.parent.parent / "fixtures/lfm3_audio/expected_results.json"
        with open(path, encoding="utf-8") as handle:
            cls.expected = json.load(handle)

    @slow
    def test_asr(self):
        """
        reproducer: https://gist.github.com/eustlb/1befc95bf557e88c808d61c9f7b199d8#file-reproducer-asr-py
        """
        model = AutoModelForMultimodalLM.from_pretrained(
            self.checkpoint, revision=self.revision, device_map=torch_device
        )
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision, device_map=torch_device)

        conversation = [
            {"role": "system", "content": "Perform ASR in English."},
            {"role": "user", "content": [{"type": "audio", "url": self.samples + "librispeech_mr_quilter.wav"}]},
        ]
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)

        outputs = model.generate(**inputs, max_new_tokens=128)
        generated_ids = outputs.sequences[0, inputs["input_ids"].shape[1] :]
        transcription = processor.decode(generated_ids, skip_special_tokens=True)

        self.assertEqual(generated_ids.tolist(), self.expected["asr"]["generated_ids"])
        self.assertEqual(transcription, self.expected["asr"]["text"])

    @slow
    def test_asr_batch(self):
        """
        reproducer: https://gist.github.com/eustlb/1befc95bf557e88c808d61c9f7b199d8#file-reproducer-asr-py
        """
        model = AutoModelForMultimodalLM.from_pretrained(
            self.checkpoint, revision=self.revision, device_map=torch_device
        )
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision, device_map=torch_device)

        conversations = [
            [
                {"role": "system", "content": "Perform ASR in English."},
                {"role": "user", "content": [{"type": "audio", "url": self.samples + "librispeech_mr_quilter.wav"}]},
            ],
            [
                {"role": "system", "content": "Perform ASR in English."},
                {"role": "user", "content": [{"type": "audio", "url": self.samples + "obama.mp3"}]},
            ],
        ]
        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)

        outputs = model.generate(**inputs, max_new_tokens=128)
        generated_ids = outputs.sequences[:, inputs["input_ids"].shape[1] :]
        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        self.assertEqual(generated_ids.tolist(), self.expected["asr_batch"]["generated_ids"])
        self.assertEqual(transcriptions, self.expected["asr_batch"]["text"])

    @slow
    def test_tts(self):
        """
        reproducer: https://gist.github.com/eustlb/1befc95bf557e88c808d61c9f7b199d8#file-reproducer-speech-py
        """
        model = AutoModelForMultimodalLM.from_pretrained(
            self.checkpoint, revision=self.revision, dtype=torch.float32, device_map=torch_device
        )
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision, device_map=torch_device)

        conversation = [
            {"role": "system", "content": "Perform TTS in American English using a female voice."},
            {"role": "user", "content": "The quick brown fox jumps over the lazy dog."},
        ]
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            generation_mode="audio",
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)

        torch.manual_seed(0)
        outputs = model.generate(**inputs, max_new_tokens=256, depth_decoder_do_sample=False)
        generated_ids = outputs.sequences[0, inputs["input_ids"].shape[1] :]
        audio = processor.decode(outputs.audio_sequences, outputs.audio_sequences_mask)

        expected_codes = torch.tensor(self.expected["tts"]["audio_codes"])
        num_frames = expected_codes.shape[-1]
        self.assertEqual(generated_ids.tolist(), self.expected["tts"]["generated_ids"])
        self.assertEqual(outputs.audio_sequences.shape[-1], num_frames)
        torch.testing.assert_close(outputs.audio_sequences[0].cpu(), expected_codes)

        # end-of-audio frame not decoded; 1920 samples per frame
        self.assertEqual(outputs.audio_sequences_mask[0].sum().item(), num_frames - 1)
        self.assertEqual([waveform.shape[-1] for waveform in audio], [(num_frames - 1) * 1920])

    @slow
    def test_tts_batch(self):
        """
        reproducer: https://gist.github.com/eustlb/1befc95bf557e88c808d61c9f7b199d8#file-reproducer-speech-py
        """
        model = AutoModelForMultimodalLM.from_pretrained(
            self.checkpoint, revision=self.revision, dtype=torch.float32, device_map=torch_device
        )
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision, device_map=torch_device)

        conversations = [
            [
                {"role": "system", "content": "Perform TTS in American English using a female voice."},
                {"role": "user", "content": "The quick brown fox jumps over the lazy dog."},
            ],
            [
                {"role": "system", "content": "Perform TTS in American English using a female voice."},
                {"role": "user", "content": "Hello there!"},
            ],
        ]
        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            generation_mode="audio",
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)

        torch.manual_seed(0)
        outputs = model.generate(**inputs, max_new_tokens=256, depth_decoder_do_sample=False)
        generated_ids = outputs.sequences[:, inputs["input_ids"].shape[1] :]
        audio = processor.decode(outputs.audio_sequences, outputs.audio_sequences_mask)

        self.assertEqual(generated_ids.tolist(), self.expected["tts_batch"]["generated_ids"])

        # rows share the RNG: goldens from a batched reference run
        expected_codes = [torch.tensor(codes) for codes in self.expected["tts_batch"]["audio_codes"]]
        num_frames = [codes.shape[-1] for codes in expected_codes]
        self.assertEqual(outputs.audio_sequences.shape[-1], max(num_frames))
        for row, codes in zip(outputs.audio_sequences.cpu(), expected_codes, strict=True):
            torch.testing.assert_close(row[:, : codes.shape[-1]], codes)

        # end-of-audio frame not decoded; 1920 samples per frame
        self.assertEqual(outputs.audio_sequences_mask.sum(-1).tolist(), [num - 1 for num in num_frames])
        self.assertEqual([waveform.shape[-1] for waveform in audio], [(num - 1) * 1920 for num in num_frames])

    @slow
    def test_speech_conversation(self):
        """
        reproducer: https://gist.github.com/eustlb/1befc95bf557e88c808d61c9f7b199d8#file-reproducer-speech-py
        """
        model = AutoModelForMultimodalLM.from_pretrained(
            self.checkpoint, revision=self.revision, dtype=torch.float32, device_map=torch_device
        )
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision, device_map=torch_device)

        conversation = [
            {"role": "system", "content": "Respond with interleaved text and audio."},
            {"role": "user", "content": [{"type": "audio", "url": self.samples + "librispeech_mr_quilter.wav"}]},
        ]
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            generation_mode="interleaved",
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)

        torch.manual_seed(0)
        outputs = model.generate(**inputs, max_new_tokens=1024, depth_decoder_do_sample=False)
        generated_ids = outputs.sequences[0, inputs["input_ids"].shape[1] :]
        text = processor.decode(generated_ids, skip_special_tokens=True)
        audio = processor.decode(outputs.audio_sequences, outputs.audio_sequences_mask)

        expected_codes = torch.tensor(self.expected["speech_conversation"]["audio_codes"])
        self.assertEqual(generated_ids.tolist(), self.expected["speech_conversation"]["generated_ids"])
        self.assertEqual(text, self.expected["speech_conversation"]["text"])
        self.assertEqual(outputs.audio_sequences.shape[-1], expected_codes.shape[-1])
        torch.testing.assert_close(outputs.audio_sequences[0].cpu(), expected_codes)

        # end-of-audio frames not decoded; 1920 samples per frame
        num_decoded_frames = (expected_codes[0] != model.config.audio_eos_code).sum().item()
        self.assertEqual(outputs.audio_sequences_mask[0].sum().item(), num_decoded_frames)
        self.assertEqual(sum(waveform.shape[-1] for waveform in audio), num_decoded_frames * 1920)

    @slow
    def test_speech_conversation_batch(self):
        """
        reproducer: https://gist.github.com/eustlb/1befc95bf557e88c808d61c9f7b199d8#file-reproducer-speech-py
        """
        model = AutoModelForMultimodalLM.from_pretrained(
            self.checkpoint, revision=self.revision, dtype=torch.float32, device_map=torch_device
        )
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision, device_map=torch_device)

        conversations = [
            [
                {"role": "system", "content": "Respond with interleaved text and audio."},
                {"role": "user", "content": [{"type": "audio", "url": self.samples + "librispeech_mr_quilter.wav"}]},
            ],
            [
                {"role": "system", "content": "Respond with interleaved text and audio."},
                {"role": "user", "content": [{"type": "audio", "url": self.samples + "obama.mp3"}]},
            ],
        ]
        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            generation_mode="interleaved",
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)

        torch.manual_seed(0)
        outputs = model.generate(**inputs, max_new_tokens=1024, depth_decoder_do_sample=False)
        generated_ids = outputs.sequences[:, inputs["input_ids"].shape[1] :]
        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        audio = processor.decode(outputs.audio_sequences, outputs.audio_sequences_mask)

        self.assertEqual(generated_ids.tolist(), self.expected["speech_conversation_batch"]["generated_ids"])
        self.assertEqual(texts, self.expected["speech_conversation_batch"]["text"])

        # rows share the RNG: goldens from a batched reference run
        expected_codes = [torch.tensor(codes) for codes in self.expected["speech_conversation_batch"]["audio_codes"]]
        self.assertEqual(outputs.audio_sequences.shape[-1], max(codes.shape[-1] for codes in expected_codes))
        for row, codes in zip(outputs.audio_sequences.cpu(), expected_codes, strict=True):
            torch.testing.assert_close(row[:, : codes.shape[-1]], codes)

        # end-of-audio frames not decoded; 1920 samples per frame
        num_decoded_frames = [(codes[0] != model.config.audio_eos_code).sum().item() for codes in expected_codes]
        self.assertEqual(outputs.audio_sequences_mask.sum(-1).tolist(), num_decoded_frames)
        self.assertEqual(sum(waveform.shape[-1] for waveform in audio), sum(num_decoded_frames) * 1920)

    @slow
    def test_batch_of_transcription_and_speech(self):
        """
        reproducer: https://gist.github.com/eustlb/1befc95bf557e88c808d61c9f7b199d8#file-reproducer-speech-py
        """
        model = AutoModelForMultimodalLM.from_pretrained(
            self.checkpoint, revision=self.revision, dtype=torch.float32, device_map=torch_device
        )
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision, device_map=torch_device)

        transcribe = [
            {"role": "system", "content": "Perform ASR in English."},
            {"role": "user", "content": [{"type": "audio", "url": self.samples + "librispeech_mr_quilter.wav"}]},
        ]
        speak = [
            {"role": "system", "content": "Perform TTS in American English using a female voice."},
            {"role": "user", "content": "Hello there!"},
        ]
        inputs = processor.apply_chat_template(
            [transcribe, speak],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)

        torch.manual_seed(0)
        outputs = model.generate(**inputs, max_new_tokens=128, depth_decoder_do_sample=False)
        generated_ids = outputs.sequences[:, inputs["input_ids"].shape[1] :]
        transcription = processor.decode(generated_ids[0], skip_special_tokens=True)
        audio = processor.decode(outputs.audio_sequences, outputs.audio_sequences_mask)

        self.assertEqual(generated_ids.tolist(), self.expected["batch"]["generated_ids"])
        self.assertEqual(transcription, self.expected["batch"]["text"][0])

        # only row 2 speaks
        expected_codes = torch.tensor(self.expected["batch"]["audio_codes"][1])
        num_frames = expected_codes.shape[-1]
        self.assertEqual(outputs.audio_sequences.shape[-1], num_frames)
        torch.testing.assert_close(outputs.audio_sequences[1].cpu(), expected_codes)
        self.assertEqual(outputs.audio_sequences_mask.sum(-1).tolist(), [0, num_frames - 1])
        self.assertEqual([waveform.shape[-1] for waveform in audio], [(num_frames - 1) * 1920])

    @slow
    def test_live_transcription(self):
        """
        reproducer: https://gist.github.com/eustlb/1befc95bf557e88c808d61c9f7b199d8#file-reproducer-live-transcription-py
        """
        model = AutoModelForMultimodalLM.from_pretrained(
            self.checkpoint, revision=self.revision, dtype=torch.float32, device_map=torch_device
        )
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision)

        audio = load_audio(self.samples + "dude_where_is_my_car.wav", sampling_rate=16000)
        window, hop = processor.num_samples_per_audio_chunk, processor.audio_chunk_hop_length

        def input_features_generator():
            starts = range(0, len(audio) - (window - hop), hop)
            for start in starts:
                yield processor(
                    audio=audio[start : start + window],
                    sampling_rate=16000,
                    is_last_audio_chunk=start == starts[-1],
                    return_tensors="pt",
                ).to(model.device, dtype=model.dtype)

        prompt = processor.apply_chat_template(
            [{"role": "system", "content": "Perform live ASR."}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        chunks = input_features_generator()
        caches = {}
        generated_ids = []
        while segment := list(itertools.islice(chunks, processor.max_audio_chunks_per_context)):
            outputs = model.generate(**prompt, input_features=iter(segment), return_dict_in_generate=True, **caches)
            caches = {"padding_cache": outputs.padding_cache, "encoder_cache": outputs.encoder_cache}
            generated_ids.append(outputs.sequences[0, prompt["input_ids"].shape[1] :])

        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        self.assertEqual(prompt["input_ids"][0].tolist(), self.expected["live_transcription"]["prompt_ids"])
        self.assertEqual(texts, self.expected["live_transcription"]["text"])

        # per chunk: audio tokens, text, `<|text_end|>`
        audio_token, text_end = model.config.audio_token_id, model.config.text_end_token_id
        expected = self.expected["live_transcription"]
        for context_ids, num_audio_tokens, chunks in zip(
            generated_ids, expected["num_audio_tokens"], expected["chunks"], strict=True
        ):
            expected_ids = []
            for num, chunk in zip(num_audio_tokens, chunks, strict=True):
                expected_ids += [audio_token] * num + chunk + [text_end]
            self.assertEqual(context_ids.tolist(), expected_ids)

    @slow
    def test_training(self):
        """
        reproducer: https://gist.github.com/eustlb/1befc95bf557e88c808d61c9f7b199d8#file-reproducer-training-py
        """
        model = AutoModelForMultimodalLM.from_pretrained(
            self.checkpoint, revision=self.revision, device_map=torch_device
        )
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision, device_map=torch_device)
        model.train()

        transcript = "Mister Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
        transcription = [
            {"role": "system", "content": "Perform ASR in English."},
            {"role": "user", "content": [{"type": "audio", "url": self.samples + "librispeech_mr_quilter.wav"}]},
            {"role": "assistant", "content": transcript},
        ]
        text_to_speech = [
            {"role": "system", "content": "Perform TTS in American English using a female voice."},
            {"role": "user", "content": transcript},
            {
                "role": "assistant",
                "content": [{"type": "audio_out", "url": self.samples + "librispeech_mr_quilter.wav"}],
            },
        ]
        inputs = processor.apply_chat_template(
            [transcription, text_to_speech],
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            return_tensors="pt",
            processor_kwargs={"output_labels": True},
        ).to(model.device, dtype=model.dtype)
        inputs["labels"][inputs.pop("assistant_masks") == 0] = -100

        # seeds the depth decoder's random codebook mask, as the reference
        torch.manual_seed(0)
        outputs = model(**inputs)

        expected = self.expected["training"]
        # 2x diff measured vs reference (A100, bfloat16): text 5.6e-4, speech 9.8e-3, total 7.4e-3
        torch.testing.assert_close(outputs.text_loss.item(), expected["text_loss"], rtol=0, atol=1.2e-3)
        torch.testing.assert_close(outputs.audio_loss.item(), expected["audio_loss"], rtol=0, atol=2e-2)
        torch.testing.assert_close(outputs.loss.item(), expected["loss"], rtol=0, atol=1.5e-2)
