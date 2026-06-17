# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Tests for Qwen3TTSProcessor."""

import os
import tempfile
import unittest

import numpy as np

from transformers import Qwen2TokenizerFast, Qwen3TTSFeatureExtractor, Qwen3TTSProcessor, is_torch_available
from transformers.testing_utils import require_torch, slow
from transformers.utils import is_soundfile_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        Qwen3TTSTokenizerMultiCodebookConfig,
        Qwen3TTSTokenizerMultiCodebookModel,
    )


def _build_tiny_audio_tokenizer(num_quantizers=4):
    """Build a tiny Qwen3TTSTokenizerMultiCodebookModel for decode/save_audio tests."""
    encoder_config = {
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
        "num_filters": 8,
        "kernel_size": 7,
        "residual_kernel_size": 3,
        "last_kernel_size": 3,
        "num_residual_layers": 1,
        "upsampling_ratios": [8, 6],
        "codebook_size": 8,
        "codebook_dim": 4,
        "vector_quantization_hidden_dimension": 4,
        "num_quantizers": num_quantizers,
        "num_semantic_quantizers": 1,
        "upsample_groups": 8,
    }
    decoder_config = {
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "intermediate_size": 32,
        "num_quantizers": num_quantizers,
        "codebook_size": 8,
        "codebook_dim": 8,
        "latent_dim": 16,
        "decoder_dim": 32,
        "upsample_rates": [2, 2],
        "upsampling_ratios": [2, 2],
    }
    config = Qwen3TTSTokenizerMultiCodebookConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        encoder_valid_num_quantizers=num_quantizers,
    )
    return Qwen3TTSTokenizerMultiCodebookModel(config).eval()


@require_torch
class Qwen3TTSProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Qwen3TTSProcessor
    model_id = None

    @classmethod
    def _setup_tokenizer(cls):
        return Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2-0.5B")

    @classmethod
    def _setup_feature_extractor(cls):
        return Qwen3TTSFeatureExtractor()

    @unittest.skip(reason="Qwen3TTS is a TTS processor; audio chat template tests are not applicable")
    def test_apply_chat_template_audio_0(self):
        pass

    @unittest.skip(reason="Qwen3TTS is a TTS processor; audio chat template tests are not applicable")
    def test_apply_chat_template_audio_1(self):
        pass

    @unittest.skip(reason="Qwen3TTS is a TTS processor; audio chat template tests are not applicable")
    def test_apply_chat_template_audio_2(self):
        pass

    @unittest.skip(reason="Qwen3TTS is a TTS processor; audio chat template tests are not applicable")
    def test_apply_chat_template_audio_3(self):
        pass

    @unittest.skip(reason="Qwen3TTS chat template returns a list, not a plain string")
    def test_chat_template_jinja_kwargs(self):
        pass

    @unittest.skip(reason="decode/batch_decode are overridden to decode audio codes, not text tokens")
    def test_tokenizer_decode_defaults(self):
        pass

    @unittest.skip(reason="decode/batch_decode are overridden to decode audio codes, not text tokens")
    def test_apply_chat_template_assistant_mask(self):
        pass

    @unittest.skip(reason="Qwen3TTS apply_chat_template is custom and does not use chat_template serialization")
    def test_chat_template_save_loading(self):
        pass

    @unittest.skip(reason="Qwen3TTS __call__ accepts text/audio, not the generic multimodal argument set")
    def test_model_input_names(self):
        pass

    @unittest.skip(reason="Qwen3TTS is text/audio only")
    def test_image_processor_defaults(self):
        pass

    @unittest.skip(reason="Qwen3TTS is text/audio only")
    def test_video_processor_defaults(self):
        pass

    @unittest.skip(reason="Qwen3TTS is a text/audio processor")
    def test_processor_text_has_no_visual(self):
        pass

    @unittest.skip(reason="Qwen3TTS uses custom text/audio kwarg routing")
    def test_tokenizer_defaults_preserved_by_kwargs_audio(self):
        pass

    @unittest.skip(reason="Qwen3TTS uses custom text/audio kwarg routing")
    def test_kwargs_overrides_default_tokenizer_kwargs_audio(self):
        pass

    @unittest.skip(reason="Qwen3TTS uses custom text/audio kwarg routing")
    def test_unstructured_kwargs_audio(self):
        pass

    @unittest.skip(reason="Qwen3TTS uses custom text/audio kwarg routing")
    def test_doubly_passed_kwargs_audio(self):
        pass

    @unittest.skip(reason="Qwen3TTS uses custom text/audio kwarg routing")
    def test_structured_kwargs_audio_nested(self):
        pass

    @unittest.skip(reason="Qwen3TTS does not combine text with image inputs")
    def test_overlapping_text_image_kwargs_handling(self):
        pass

    @unittest.skip(reason="Qwen3TTS uses custom text/audio kwarg routing")
    def test_overlapping_text_audio_kwargs_handling(self):
        pass

    @unittest.skip(reason="Qwen3TTS has no multimodal token counting helper")
    def test_get_num_multimodal_tokens_matches_processor_call(self):
        pass

    def test_call_text(self):
        processor = self.get_processor()
        text = "Hello there."

        inputs = processor(text=text, return_tensors="pt")
        tokenizer_inputs = processor.tokenizer([text], padding=False, padding_side="left", return_tensors="pt")

        self.assertEqual(set(inputs.keys()), set(tokenizer_inputs.keys()))
        for key in tokenizer_inputs:
            torch.testing.assert_close(inputs[key], tokenizer_inputs[key])

    def test_call_audio(self):
        processor = self.get_processor()
        audio = np.zeros(2048, dtype=np.float32)

        inputs = processor(audio=audio, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt")
        feature_inputs = processor.feature_extractor(
            audio, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt"
        )

        self.assertEqual(set(inputs.keys()), {"input_features"})
        torch.testing.assert_close(inputs["input_features"], feature_inputs["input_features"])

    def test_call_text_and_audio(self):
        processor = self.get_processor()
        inputs = processor(
            text="Hello there.",
            audio=np.zeros(2048, dtype=np.float32),
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
        )

        self.assertIn("input_ids", inputs)
        self.assertIn("input_features", inputs)

    def test_call_requires_input(self):
        processor = self.get_processor()
        with self.assertRaises(ValueError):
            processor()

    def test_apply_chat_template_basic(self):
        processor = self.get_processor()
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": "Hello there."}]},
        ]
        inputs = processor.apply_chat_template(conversation)

        self.assertEqual(len(inputs["input_ids"]), 1)
        self.assertEqual(inputs["input_ids"][0].dim(), 2)
        self.assertEqual(inputs["languages"], ["auto"])
        self.assertEqual(inputs["speakers"], [None])
        self.assertNotIn("instruct_ids", inputs)

    def test_apply_chat_template_language_and_speaker(self):
        processor = self.get_processor()
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Welcome."}],
                "language": "English",
                "speaker": "Ryan",
            },
        ]
        inputs = processor.apply_chat_template(conversation)

        self.assertEqual(inputs["languages"], ["English"])
        self.assertEqual(inputs["speakers"], ["Ryan"])

    def test_apply_chat_template_instruct(self):
        processor = self.get_processor()
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "A calm female voice."}]},
            {"role": "user", "content": [{"type": "text", "text": "Good morning."}], "language": "English"},
        ]
        inputs = processor.apply_chat_template(conversation)

        self.assertIn("instruct_ids", inputs)
        self.assertEqual(len(inputs["instruct_ids"]), 1)
        self.assertEqual(inputs["languages"], ["English"])

    def test_apply_chat_template_batch(self):
        processor = self.get_processor()
        conversations = [
            [{"role": "user", "content": [{"type": "text", "text": "First sentence."}]}],
            [{"role": "user", "content": [{"type": "text", "text": "Second sentence."}]}],
        ]
        inputs = processor.apply_chat_template(conversations)

        self.assertEqual(len(inputs["input_ids"]), 2)
        self.assertEqual(inputs["languages"], ["auto", "auto"])

    def test_apply_chat_template_plain_string_content(self):
        processor = self.get_processor()
        conversation = [{"role": "user", "content": "Plain string content."}]
        inputs = processor.apply_chat_template(conversation)

        self.assertEqual(len(inputs["input_ids"]), 1)

    def test_apply_chat_template_requires_user_message(self):
        processor = self.get_processor()
        conversation = [{"role": "system", "content": [{"type": "text", "text": "Only an instruction."}]}]
        with self.assertRaises(ValueError):
            processor.apply_chat_template(conversation)

    def test_batch_decode_and_save_audio(self):
        if not is_soundfile_available():
            self.skipTest("soundfile is required to save audio")
        processor = self.get_processor()
        processor.audio_tokenizer = _build_tiny_audio_tokenizer(num_quantizers=4)

        codes = [torch.randint(0, 8, (6, 4)), torch.randint(0, 8, (5, 4))]
        audios = processor.batch_decode(codes)

        self.assertEqual(len(audios), 2)
        self.assertTrue(all(isinstance(a, torch.Tensor) for a in audios))

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = [os.path.join(tmpdir, "out_0.wav"), os.path.join(tmpdir, "out_1.wav")]
            processor.save_audio(audios, paths)
            self.assertTrue(all(os.path.isfile(p) for p in paths))

    def test_decode_single(self):
        processor = self.get_processor()
        processor.audio_tokenizer = _build_tiny_audio_tokenizer(num_quantizers=4)

        codes = torch.randint(0, 8, (6, 4))
        audio = processor.decode(codes)

        self.assertIsInstance(audio, torch.Tensor)

    @slow
    def test_can_load_processor_from_pretrained(self):
        processor = Qwen3TTSProcessor.from_pretrained("qwen3_tts_converted")
        self.assertIsNotNone(processor.tokenizer)
        self.assertIsNotNone(processor.feature_extractor)
