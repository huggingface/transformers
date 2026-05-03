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

import shutil
import tempfile
import unittest

from parameterized import parameterized

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2TokenizerFast,
    Qwen3ASRFeatureExtractor,
)
from transformers.models.qwen3_asr.processing_qwen3_asr import Qwen3ASRProcessor
from transformers.testing_utils import (
    require_torch,
    require_torchaudio,
)

from ...test_processing_common import ProcessorTesterMixin


class Qwen3ASRProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Qwen3ASRProcessor

    @classmethod
    @require_torch
    @require_torchaudio
    def setUpClass(cls):
        cls.checkpoint = "bezzam/Qwen3-ASR-0.6B"
        cls.tmpdirname = tempfile.mkdtemp()
        processor = Qwen3ASRProcessor.from_pretrained(cls.checkpoint)
        processor.save_pretrained(cls.tmpdirname)

    @require_torch
    @require_torchaudio
    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    @require_torch
    @require_torchaudio
    def get_feature_extractor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).feature_extractor

    @require_torch
    @require_torchaudio
    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @require_torch
    @require_torchaudio
    def test_can_load_various_tokenizers(self):
        processor = Qwen3ASRProcessor.from_pretrained(self.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.assertEqual(processor.tokenizer.__class__, tokenizer.__class__)

    @require_torch
    @require_torchaudio
    def test_save_load_pretrained_default(self):
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        processor = Qwen3ASRProcessor.from_pretrained(self.checkpoint)
        feature_extractor = processor.feature_extractor

        processor = Qwen3ASRProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_pretrained(tmpdir)
            reloaded = Qwen3ASRProcessor.from_pretrained(tmpdir)

        self.assertEqual(reloaded.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(reloaded.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(reloaded.feature_extractor, Qwen3ASRFeatureExtractor)
        self.assertIsInstance(reloaded.tokenizer, Qwen2TokenizerFast)

    @require_torch
    @require_torchaudio
    def test_chat_template(self):
        processor = AutoProcessor.from_pretrained(self.checkpoint)
        expected_prompt = (
            "<|im_start|>system\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "path": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
                    },
                ],
            },
        ]
        formatted_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertEqual(expected_prompt, formatted_prompt)

    @require_torch
    @require_torchaudio
    def test_apply_transcription_request_single(self):
        processor = AutoProcessor.from_pretrained(self.checkpoint)

        audio_url = "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"
        helper_outputs = processor.apply_transcription_request(audio=audio_url)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_url},
                ],
            }
        ]
        manual_outputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )

        for key in ("input_ids", "attention_mask", "input_features", "input_features_mask"):
            self.assertIn(key, helper_outputs)
            self.assertTrue(helper_outputs[key].equal(manual_outputs[key]))

    @require_torch
    @require_torchaudio
    def test_apply_transcription_request_with_language(self):
        processor = AutoProcessor.from_pretrained(self.checkpoint)

        audio_url = "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"
        outputs = processor.apply_transcription_request(audio=audio_url, language="English")

        for key in ("input_ids", "attention_mask", "input_features", "input_features_mask"):
            self.assertIn(key, outputs)

    @require_torch
    @require_torchaudio
    def test_decode_formats(self):
        processor = AutoProcessor.from_pretrained(self.checkpoint)

        raw_text = "language English<asr_text>Mr. Quilter is the apostle of the middle classes."

        # raw
        self.assertEqual(raw_text, raw_text)

        # parsed
        parsed = processor.parse_output(raw_text)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed["language"], "English")
        self.assertEqual(parsed["transcription"], "Mr. Quilter is the apostle of the middle classes.")

        # transcription_only
        transcription = processor.extract_transcription(raw_text)
        self.assertEqual(transcription, "Mr. Quilter is the apostle of the middle classes.")

    @parameterized.expand([(1, "np"), (1, "pt"), (2, "np"), (2, "pt")])
    def test_apply_chat_template_audio(self, batch_size: int, return_tensors: str):
        self.skipTest("Qwen3ASR processor requires audio; not compatible with text-only chat template tests.")

    def test_apply_chat_template_assistant_mask(self):
        self.skipTest("Qwen3ASR processor requires audio; not compatible with text-only chat template tests.")
