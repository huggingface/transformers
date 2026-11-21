# Copyright 2023 The HuggingFace Team. All rights reserved.
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


import gc
import shutil
import tempfile
import unittest

from transformers import ClvpAudioProcessor, ClvpProcessor, ClvpTokenizer
from transformers.testing_utils import require_torch

from ...test_processing_common import floats_list


@require_torch
class ClvpProcessorTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint = "susnato/clvp_dev"
        self.tmpdirname = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmpdirname)
        gc.collect()

    # Copied from transformers.tests.models.whisper.test_processing_whisper.WhisperProcessorTest.get_tokenizer with Whisper->Clvp
    def get_tokenizer(self, **kwargs):
        return ClvpTokenizer.from_pretrained(self.checkpoint, **kwargs)

    # Copied from transformers.tests.models.whisper.test_processing_whisper.WhisperProcessorTest.get_audio_processor with Whisper->Clvp
    def get_audio_processor(self, **kwargs):
        return ClvpAudioProcessor.from_pretrained(self.checkpoint, **kwargs)

    # Copied from transformers.tests.models.whisper.test_processing_whisper.WhisperProcessorTest.test_save_load_pretrained_default with Whisper->Clvp
    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()

        processor = ClvpProcessor(tokenizer=tokenizer, audio_processor=audio_processor)

        processor.save_pretrained(self.tmpdirname)
        processor = ClvpProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, ClvpTokenizer)

        self.assertEqual(processor.audio_processor.to_json_string(), audio_processor.to_json_string())
        self.assertIsInstance(processor.audio_processor, ClvpAudioProcessor)

    # Copied from transformers.tests.models.whisper.test_processing_whisper.WhisperProcessorTest.test_audio_processor with Whisper->Clvp,processor(raw_speech->processor(raw_speech=raw_speech
    def test_audio_processor(self):
        audio_processor = self.get_audio_processor()
        tokenizer = self.get_tokenizer()

        processor = ClvpProcessor(tokenizer=tokenizer, audio_processor=audio_processor)

        raw_speech = floats_list((3, 1000))

        input_feat_extract = audio_processor(raw_speech, return_tensors="np")
        input_processor = processor(raw_speech=raw_speech, return_tensors="np")

        for key in input_feat_extract:
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    # Copied from transformers.tests.models.whisper.test_processing_whisper.WhisperProcessorTest.test_tokenizer with Whisper->Clvp
    def test_tokenizer(self):
        audio_processor = self.get_audio_processor()
        tokenizer = self.get_tokenizer()

        processor = ClvpProcessor(tokenizer=tokenizer, audio_processor=audio_processor)

        input_str = "This is a test string"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str)

        for key in encoded_tok:
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    # Copied from transformers.tests.models.whisper.test_processing_whisper.WhisperProcessorTest.test_tokenizer_decode with Whisper->Clvp
    def test_tokenizer_decode(self):
        audio_processor = self.get_audio_processor()
        tokenizer = self.get_tokenizer()

        processor = ClvpProcessor(tokenizer=tokenizer, audio_processor=audio_processor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_save_load_pretrained_additional_features(self):
        processor = ClvpProcessor(tokenizer=self.get_tokenizer(), audio_processor=self.get_audio_processor())
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(pad_token="(PAD)")
        audio_processor_add_kwargs = self.get_audio_processor(sampling_rate=16000)

        processor = ClvpProcessor.from_pretrained(
            self.tmpdirname,
            pad_token="(PAD)",
            sampling_rate=16000,
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, ClvpTokenizer)

        self.assertEqual(processor.audio_processor.to_json_string(), audio_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.audio_processor, ClvpAudioProcessor)

    def test_text_and_audio_attention_mask(self):
        # When both `text` and `audio` are passed, the CLVP model consumes the *text* attention mask.
        # Ensure the audio feature extractor's (much longer) attention mask does not override the text one
        # in the merged output. Regression test for the merged-output attention mask collision.
        audio_processor = self.get_audio_processor()
        tokenizer = self.get_tokenizer()
        processor = ClvpProcessor(tokenizer=tokenizer, audio_processor=audio_processor)

        raw_speech = floats_list((3, 1000))
        input_str = "This is a test string"

        inputs = processor(text=input_str, raw_speech=raw_speech, return_tensors="pt")

        self.assertIn("input_ids", inputs)
        self.assertIn("input_features", inputs)
        self.assertIn("attention_mask", inputs)
        # The attention mask must match the text `input_ids`, not the audio features.
        self.assertEqual(inputs["attention_mask"].shape, inputs["input_ids"].shape)

    def test_text_and_audio_flat_kwargs(self):
        # Flat (backward-compatible) kwargs must still be forwarded to the tokenizer when both `text` and
        # `audio` are passed. Regression test ensuring the audio-mask handling does not discard flat kwargs.
        audio_processor = self.get_audio_processor()
        tokenizer = self.get_tokenizer()
        processor = ClvpProcessor(tokenizer=tokenizer, audio_processor=audio_processor)

        raw_speech = floats_list((3, 1000))
        input_str = "This is a test string"

        inputs = processor(
            text=input_str,
            raw_speech=raw_speech,
            return_tensors="pt",
            padding="max_length",
            max_length=20,
        )

        self.assertEqual(inputs["input_ids"].shape[-1], 20)
        self.assertEqual(inputs["attention_mask"].shape[-1], 20)
