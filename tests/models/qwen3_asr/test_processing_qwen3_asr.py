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
from transformers.testing_utils import require_torch

from ...test_processing_common import ProcessorTesterMixin


class Qwen3ASRProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Qwen3ASRProcessor

    @classmethod
    @require_torch
    def setUpClass(cls):
        cls.checkpoint = "Qwen/Qwen3-ASR-0.6B-hf"
        cls.revision = "refs/pr/3"  # TODO: set to main after merge
        cls.tmpdirname = tempfile.mkdtemp()

        processor = Qwen3ASRProcessor.from_pretrained(cls.checkpoint, revision=cls.revision)
        processor.save_pretrained(cls.tmpdirname)

    @require_torch
    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    @require_torch
    def get_feature_extractor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).feature_extractor

    @require_torch
    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @require_torch
    def test_can_load_various_tokenizers(self):
        processor = Qwen3ASRProcessor.from_pretrained(self.checkpoint, revision=self.revision)
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, revision=self.revision)
        self.assertEqual(processor.tokenizer.__class__, tokenizer.__class__)

    @require_torch
    def test_save_load_pretrained_default(self):
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, revision=self.revision)
        processor = Qwen3ASRProcessor.from_pretrained(self.checkpoint, revision=self.revision)
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
    def test_chat_template(self):
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision)
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
                        "path": "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
                    },
                ],
            },
        ]
        formatted_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertEqual(expected_prompt, formatted_prompt)

    @require_torch
    def test_apply_transcription_request_with_language(self):
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision)

        audio_url = "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"
        outputs = processor.apply_transcription_request(audio=audio_url, language="English")

        for key in ("input_ids", "attention_mask", "input_features", "input_features_mask"):
            self.assertIn(key, outputs)

        # The language is forced by appending "language <NAME><asr_text>" after the generation prompt
        decoded = processor.tokenizer.decode(outputs["input_ids"][0])
        self.assertTrue(decoded.endswith("<|im_start|>assistant\nlanguage English<asr_text>"))

    @require_torch
    def test_apply_transcription_request_with_prompt(self):
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision)

        audio_url = "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"
        context = "Vocabulary: Quilter, apostle, gospel."
        outputs = processor.apply_transcription_request(audio=audio_url, prompt=context, language="English")

        decoded = processor.tokenizer.decode(outputs["input_ids"][0])
        # The context/hotwords prompt goes into the system turn
        self.assertIn(f"<|im_start|>system\n{context}<|im_end|>", decoded)
        self.assertTrue(decoded.endswith("<|im_start|>assistant\nlanguage English<asr_text>"))

    @require_torch
    def test_apply_transcription_request_mixed_batch(self):
        """Mixed batch: forced-language samples get the prefill, auto-detect samples a bare generation prompt."""
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision)

        audio_url = "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"
        outputs = processor.apply_transcription_request(audio=[audio_url, audio_url], language=[None, "zh"])

        decoded_auto = processor.tokenizer.decode(outputs["input_ids"][0], skip_special_tokens=False)
        decoded_forced = processor.tokenizer.decode(outputs["input_ids"][1])
        self.assertTrue(decoded_auto.replace("<|endoftext|>", "").endswith("<|im_start|>assistant\n"))
        self.assertTrue(decoded_forced.endswith("<|im_start|>assistant\nlanguage Chinese<asr_text>"))

    @require_torch
    def test_decode_formats(self):
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision)

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

    @require_torch
    def test_output_labels(self):
        import torch

        processor = self.get_processor()
        audio = self.prepare_audio_inputs(batch_size=1)[0]

        conversation = [
            [
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio": audio}],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "language English<asr_text>Hello world."}]},
            ],
        ]
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            processor_kwargs={"output_labels": True},
        )

        self.assertIn("labels", inputs)
        self.assertNotIn("mm_token_type_ids", inputs)
        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        self.assertEqual(labels.shape, input_ids.shape)

        # audio token positions (including audio bos/eos) are masked
        audio_positions = torch.isin(input_ids, torch.tensor(processor.audio_token_ids, dtype=input_ids.dtype))
        self.assertTrue(audio_positions.any())
        self.assertTrue((labels[audio_positions] == -100).all())

        # non-audio positions match input_ids
        kept_positions = ~audio_positions
        self.assertTrue(kept_positions.any())
        self.assertTrue((labels[kept_positions] == input_ids[kept_positions]).all())
