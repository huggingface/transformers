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
"""Testing suite for the PyTorch Voxtral TTS model."""

import unittest

from transformers import (
    VoxtralTtsAcousticTransformerConfig,
    VoxtralTtsCodecConfig,
    VoxtralTtsConfig,
)
from transformers.testing_utils import (
    cleanup,
    is_torch_available,
    require_mistral_common,
    require_torch,
    slow,
    torch_device,
)

from ...model_tester_config import ModelTesterConfigMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        AutoProcessor,
        DynamicCache,
        VoxtralTtsCodecModel,
        VoxtralTtsForTextToSpeech,
        VoxtralTtsModel,
    )


@require_torch
class VoxtralTtsModelTester(ModelTesterConfigMixin):
    defaults = {}
    small_config_overrides = {
        "batch_size": 2,
        "seq_length": 9,
        "num_voice_tokens": 3,
        "is_training": False,
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "semantic_codebook_size": 30,
        "acoustic_codebook_size": 5,
        "num_acoustic_codebooks": 4,
        "audio_token_id": 24,
        "begin_audio_token_id": 25,
        "pad_token_id": 11,
    }

    def __init__(self, parent, **kwargs):
        self.parent = parent
        for key, value in self.get_small_config_defaults(**kwargs).items():
            setattr(self, key, value)

    def get_config(self):
        return VoxtralTtsConfig(
            text_config={
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "max_position_embeddings": 512,
                "pad_token_id": self.pad_token_id,
            },
            acoustic_transformer_config=VoxtralTtsAcousticTransformerConfig(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_hidden_layers=1,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                llm_hidden_size=self.hidden_size,
                semantic_codebook_size=self.semantic_codebook_size,
                acoustic_codebook_size=self.acoustic_codebook_size,
                num_acoustic_codebooks=self.num_acoustic_codebooks,
                num_flow_matching_steps=2,
            ),
            codec_config=VoxtralTtsCodecConfig(
                semantic_codebook_size=self.semantic_codebook_size,
                semantic_dim=8,
                acoustic_codebook_size=self.acoustic_codebook_size,
                acoustic_dim=self.num_acoustic_codebooks,
                hidden_size=16,
                intermediate_size=32,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                patch_size=4,
                patch_proj_kernel_size=3,
                sliding_window=4,
                decoder_num_layers=[1, 1],
                decoder_kernel_sizes=[3, 4],
                decoder_strides=[1, 2],
            ),
            audio_token_id=self.audio_token_id,
            begin_audio_token_id=self.begin_audio_token_id,
            audio_vocab_size=128,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        # text tokens, never the audio placeholder token
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 30) + 30
        input_ids[:, -1] = self.begin_audio_token_id
        attention_mask = torch.ones_like(input_ids)
        return config, input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}

    def prepare_config_and_inputs_with_voice(self):
        """Inputs laid out as the processor does: `[BOS] [BEGIN_AUDIO] [AUDIO] * n text [BEGIN_AUDIO]`."""
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        input_ids[:, 0] = 1
        input_ids[:, 1] = self.begin_audio_token_id
        input_ids[:, 2 : 2 + self.num_voice_tokens] = self.audio_token_id
        voice_embeddings = torch.randn(self.batch_size * self.num_voice_tokens, self.hidden_size, device=torch_device)
        return config, {"input_ids": input_ids, "attention_mask": attention_mask, "voice_embeddings": voice_embeddings}


@require_torch
class VoxtralTtsModelTest(ModelTesterMixin, unittest.TestCase):
    model_tester_class = VoxtralTtsModelTester
    all_model_classes = (VoxtralTtsModel, VoxtralTtsForTextToSpeech) if is_torch_available() else ()
    # `generate` is custom: there are no text logits to sample, every step produces an audio frame with the acoustic
    # transformer (see `test_generate*` below)
    all_generative_model_classes = ()
    _is_composite = True
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = VoxtralTtsModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VoxtralTtsConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def _random_frames(self, config, batch_size, num_frames):
        at = config.acoustic_transformer_config
        semantic = torch.randint(2, 2 + at.semantic_codebook_size, (batch_size, num_frames, 1))
        acoustic = torch.randint(2, 2 + at.acoustic_codebook_size, (batch_size, num_frames, at.num_acoustic_codebooks))
        return torch.cat([semantic, acoustic], dim=-1).to(torch_device)

    def test_voice_embeddings_replace_placeholders(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_with_voice()
        model = VoxtralTtsModel(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(**inputs)
            inputs_embeds = model.get_input_embeddings()(inputs["input_ids"])
            inputs_embeds[inputs["input_ids"] == config.audio_token_id] = inputs["voice_embeddings"]
            expected = model(inputs_embeds=inputs_embeds, attention_mask=inputs["attention_mask"])
        torch.testing.assert_close(outputs.last_hidden_state, expected.last_hidden_state)

        with self.assertRaisesRegex(ValueError, "voice embedding vectors"):
            model(input_ids=inputs["input_ids"], voice_embeddings=inputs["voice_embeddings"][:-1])

    def test_audio_codes_extend_prompt_attention_mask(self):
        # Appending audio frames to a prompt whose attention mask only covers the prompt (as returned by the
        # processor) must behave as if the mask covered the frames too.
        config, inputs = self.model_tester.prepare_config_and_inputs_with_voice()
        model = VoxtralTtsModel(config).to(torch_device).eval()
        frames = self._random_frames(config, self.model_tester.batch_size, 4)
        full_mask = torch.cat([inputs["attention_mask"], torch.ones_like(frames[..., 0])], dim=-1)
        with torch.no_grad():
            outputs = model(**inputs, audio_codes=frames)
            expected = model(**{**inputs, "attention_mask": full_mask}, audio_codes=frames)
            unmasked = model(
                input_ids=inputs["input_ids"], voice_embeddings=inputs["voice_embeddings"], audio_codes=frames
            )
        torch.testing.assert_close(outputs.last_hidden_state, expected.last_hidden_state)
        torch.testing.assert_close(outputs.last_hidden_state, unmasked.last_hidden_state)

    def test_incremental_decoding_matches_full_forward(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_with_voice()
        model = VoxtralTtsModel(config).to(torch_device).eval()
        frames = self._random_frames(config, self.model_tester.batch_size, 3)
        with torch.no_grad():
            full = model(**inputs, audio_codes=frames).last_hidden_state
            cache = DynamicCache(config=config.text_config)
            prompt = model(**inputs, cache=cache, use_cache=True).last_hidden_state
            steps = []
            attention_mask = inputs["attention_mask"]
            for i in range(frames.shape[1]):
                attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, :1])], dim=-1)
                position_ids = torch.full((frames.shape[0], 1), attention_mask.shape[1] - 1, device=torch_device)
                step = model(
                    audio_codes=frames[:, i : i + 1],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache=cache,
                    use_cache=True,
                )
                steps.append(step.last_hidden_state)
        incremental = torch.cat([prompt] + steps, dim=1)
        torch.testing.assert_close(incremental, full, rtol=1e-4, atol=1e-4)

    def test_left_padding(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_with_voice()
        model = VoxtralTtsModel(config).to(torch_device).eval()
        num_pad = 3
        pad = torch.full((1, num_pad), self.model_tester.pad_token_id, device=torch_device)
        padded_ids = torch.cat([pad, inputs["input_ids"][:1]], dim=-1)
        padded_mask = torch.cat([torch.zeros_like(pad), inputs["attention_mask"][:1]], dim=-1)
        voice = inputs["voice_embeddings"][: self.model_tester.num_voice_tokens]
        with torch.no_grad():
            expected = model(input_ids=inputs["input_ids"][:1], voice_embeddings=voice).last_hidden_state
            padded = model(input_ids=padded_ids, attention_mask=padded_mask, voice_embeddings=voice).last_hidden_state
        torch.testing.assert_close(padded[:, num_pad:], expected, rtol=1e-4, atol=1e-4)

    def test_generate(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_with_voice()
        model = VoxtralTtsForTextToSpeech(config).to(torch_device).eval()
        outputs = model.generate(**inputs, max_new_frames=5, generator=torch.Generator(torch_device).manual_seed(0))
        batch_size, num_codebooks = self.model_tester.batch_size, config.num_codebooks
        self.assertEqual(outputs.audio_codes.shape[0], batch_size)
        self.assertEqual(outputs.audio_codes.shape[2], num_codebooks)
        self.assertLessEqual(outputs.audio_codes.shape[1], 5)
        self.assertEqual(len(outputs.audio), batch_size)
        for audio, length in zip(outputs.audio, outputs.audio_lengths):
            self.assertEqual(audio.shape, (int(length) * config.codec_config.hop_length,))

        # Same seed, same audio
        again = model.generate(**inputs, max_new_frames=5, generator=torch.Generator(torch_device).manual_seed(0))
        torch.testing.assert_close(again.audio_codes, outputs.audio_codes)

    def test_generate_stops_on_end_audio(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_with_voice()
        model = VoxtralTtsForTextToSpeech(config).to(torch_device).eval()
        # All semantic logits equal: `[EMPTY_AUDIO]` is excluded, so the argmax is `[END_AUDIO]`
        model.acoustic_transformer.semantic_head.weight.data.zero_()
        outputs = model.generate(**inputs, max_new_frames=5)
        self.assertEqual(outputs.audio_codes.shape[1], 1)
        self.assertTrue((outputs.audio_codes[:, 0, 0] == 1).all())
        self.assertTrue((outputs.audio_lengths == 0).all())
        self.assertTrue(all(audio.numel() == 0 for audio in outputs.audio))

    def test_acoustic_transformer_frame_codes(self):
        config = self.model_tester.get_config()
        model = VoxtralTtsForTextToSpeech(config).to(torch_device).eval()
        at_config = config.acoustic_transformer_config
        hidden = torch.randn(3, self.model_tester.hidden_size, device=torch_device)
        noise = torch.randn(3, at_config.num_acoustic_codebooks, device=torch_device)
        frame = model.acoustic_transformer.generate_frame(hidden, noise=noise)
        self.assertEqual(frame.shape, (3, 1 + at_config.num_acoustic_codebooks))
        torch.testing.assert_close(frame, model.acoustic_transformer.generate_frame(hidden, noise=noise))
        # semantic codes are never `[EMPTY_AUDIO]`, acoustic codes are shifted FSQ levels
        self.assertTrue((frame[:, 0] >= 1).all() and (frame[:, 0] < 2 + at_config.semantic_codebook_size).all())
        acoustic = frame[frame[:, 0] != 1, 1:]
        self.assertTrue((acoustic >= 2).all() and (acoustic < 2 + at_config.acoustic_codebook_size).all())

    def test_codec_is_causal(self):
        config = self.model_tester.get_config()
        codec = VoxtralTtsCodecModel(config.codec_config).to(torch_device).eval()
        codec_config = config.codec_config
        codes = torch.cat(
            [
                torch.randint(0, codec_config.semantic_codebook_size, (2, 12, 1)),
                torch.randint(0, codec_config.acoustic_codebook_size, (2, 12, codec_config.acoustic_dim)),
            ],
            dim=-1,
        ).to(torch_device)
        with torch.no_grad():
            full = codec.decode(codes).audio_values
            prefix = codec.decode(codes[:, :7]).audio_values
        self.assertEqual(full.shape, (2, 12 * codec_config.hop_length))
        torch.testing.assert_close(full[:, : prefix.shape[1]], prefix, rtol=1e-4, atol=1e-4)


@slow
@require_torch
@require_mistral_common
class VoxtralTtsIntegrationTest(unittest.TestCase):
    # TODO: point to the Transformers-format files once they are on the Hub
    checkpoint = "mistralai/Voxtral-4B-TTS-2603"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_generate_casual_male(self):
        """
        Reproducer, values obtained on CPU (bf16): the original model (a transcription of the vllm-omni reference) and
        this port generate the same 23 frames for this prompt, which Whisper transcribes as "Paris is a beautiful city."
        """
        processor = AutoProcessor.from_pretrained(self.checkpoint)
        model = VoxtralTtsForTextToSpeech.from_pretrained(
            self.checkpoint, dtype=torch.bfloat16, device_map=torch_device
        )
        inputs = processor("Paris is a beautiful city!", voice="casual_male").to(torch_device)
        outputs = model.generate(**inputs, generator=torch.Generator(torch_device).manual_seed(0))

        # fmt: off
        expected_semantic_codes = [855, 7008, 7096, 6071, 1438, 855, 10, 10, 6226, 663, 3950, 2072, 8023, 8023, 8023, 10, 10, 10, 3684, 3684, 7973, 5246, 1]
        expected_first_frame = [855, 2, 15, 6, 6, 16, 10, 6, 6, 18, 5, 18, 5, 7, 20, 13, 11, 9, 6, 17, 10, 15, 7, 8, 14, 9, 9, 18, 21, 16, 20, 4, 6, 6, 17, 3, 4]
        # fmt: on
        self.assertEqual(outputs.audio_codes[0, :, 0].tolist(), expected_semantic_codes)
        self.assertEqual(outputs.audio_codes[0, 0].tolist(), expected_first_frame)
        self.assertEqual(outputs.audio[0].shape, (22 * model.config.codec_config.hop_length,))
