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

import shutil
import tempfile
import unittest

from parameterized import parameterized

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    GlmAsrProcessor,
    WhisperFeatureExtractor,
)
from transformers.testing_utils import require_librosa, require_torch

from ...test_processing_common import MODALITY_INPUT_DATA, ProcessorTesterMixin


class GlmAsrProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = GlmAsrProcessor
    # Tiny processor created with make_tiny_processor.py from "zai-org/GLM-ASR-Nano-2512"
    tiny_model_id = "hf-internal-testing/tiny-processor-glmasr"

    @classmethod
    @require_torch
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        processor = GlmAsrProcessor.from_pretrained(cls.tiny_model_id)
        processor.save_pretrained(cls.tmpdirname)

    @require_torch
    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    @require_torch
    def get_audio_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).audio_processor

    @require_torch
    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @require_torch
    def test_can_load_various_tokenizers(self):
        processor = GlmAsrProcessor.from_pretrained(self.tiny_model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.tiny_model_id)
        self.assertEqual(processor.tokenizer.__class__, tokenizer.__class__)

    @require_torch
    def test_save_load_pretrained_default(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tiny_model_id)
        processor = GlmAsrProcessor.from_pretrained(self.tiny_model_id)
        feature_extractor = processor.feature_extractor

        processor = GlmAsrProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_pretrained(tmpdir)
            reloaded = GlmAsrProcessor.from_pretrained(tmpdir)

        self.assertEqual(reloaded.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(reloaded.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(reloaded.feature_extractor, WhisperFeatureExtractor)

    # Overwrite to remove skip numpy inputs (still need to keep as many cases as parent)
    @require_librosa
    @parameterized.expand([(1, "np"), (1, "pt"), (2, "np"), (2, "pt")])
    def test_apply_chat_template_audio(self, batch_size: int, return_tensors: str):
        if return_tensors == "np":
            self.skipTest("GlmAsr only supports PyTorch tensors")
        self._test_apply_chat_template(
            "audio", batch_size, return_tensors, "audio_input_name", "feature_extractor", MODALITY_INPUT_DATA["audio"]
        )

    @require_torch
    def test_output_labels_with_audio(self):
        processor = self.get_processor()
        audio_token_id = processor.audio_token_id
        pad_token_id = processor.tokenizer.pad_token_id

        # Different text lengths so that padding is applied
        text = [
            f"{processor.audio_token} Transcribe the input speech.",
            f"{processor.audio_token} What can you hear in this audio clip?",
        ]
        audio = self.prepare_audio_inputs(batch_size=2)

        inputs = processor(text=text, audio=audio, output_labels=True)

        self.assertIn("labels", inputs)
        self.assertNotIn("mm_token_type_ids", inputs)
        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        self.assertEqual(labels.shape, input_ids.shape)

        # audio token positions are masked
        audio_positions = input_ids == audio_token_id
        self.assertTrue(audio_positions.any())
        self.assertTrue((labels[audio_positions] == -100).all())

        # padding positions are masked
        pad_positions = input_ids == pad_token_id
        self.assertTrue(pad_positions.any())
        self.assertTrue((labels[pad_positions] == -100).all())

        # all other positions match input_ids
        kept_positions = ~(audio_positions | pad_positions)
        self.assertTrue(kept_positions.any())
        self.assertTrue((labels[kept_positions] == input_ids[kept_positions]).all())

    @require_torch
    def test_output_labels_without_audio(self):
        processor = self.get_processor()
        pad_token_id = processor.tokenizer.pad_token_id

        # Different text lengths so that padding is applied
        text = ["Transcribe the input speech.", "Hello!"]
        inputs = processor(text=text, output_labels=True)

        self.assertIn("labels", inputs)
        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        self.assertEqual(labels.shape, input_ids.shape)

        # without audio, only padding positions are masked
        pad_positions = input_ids == pad_token_id
        self.assertTrue(pad_positions.any())
        self.assertTrue((labels[pad_positions] == -100).all())
        kept_positions = ~pad_positions
        self.assertTrue((labels[kept_positions] == input_ids[kept_positions]).all())
