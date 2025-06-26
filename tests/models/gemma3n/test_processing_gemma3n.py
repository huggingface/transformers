# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import numpy as np
from parameterized import parameterized

from transformers import GemmaTokenizerFast, SiglipImageProcessorFast, is_speech_available
from transformers.testing_utils import require_sentencepiece, require_torch, require_torchaudio, require_vision

from .test_feature_extraction_gemma3n import floats_list


if is_speech_available():
    from transformers.models.gemma3n import Gemma3nAudioFeatureExtractor, Gemma3nProcessor


@require_torch
@require_torchaudio
@require_vision
@require_sentencepiece
class Gemma3nProcessorTest(unittest.TestCase):
    def setUp(self):
        # TODO: update to google?
        self.model_id = "Google/gemma-3n-E4B-it"
        self.tmpdirname = tempfile.mkdtemp(suffix="gemma3n")
        self.maxDiff = None

    def get_tokenizer(self, **kwargs):
        return GemmaTokenizerFast.from_pretrained(self.model_id, **kwargs)

    def get_feature_extractor(self, **kwargs):
        return Gemma3nAudioFeatureExtractor.from_pretrained(self.model_id, **kwargs)

    def get_image_processor(self, **kwargs):
        return SiglipImageProcessorFast.from_pretrained(self.model_id, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        # NOTE: feature_extractor and image_processor both use the same filename, preprocessor_config.json, when saved to
        # disk, but the files are overwritten by processor.save_pretrained(). This test does not attempt to address
        # this potential issue, and as such, does not guarantee content accuracy.

        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()
        image_processor = self.get_image_processor()

        processor = Gemma3nProcessor(
            tokenizer=tokenizer, feature_extractor=feature_extractor, image_processor=image_processor
        )

        processor.save_pretrained(self.tmpdirname)
        processor = Gemma3nProcessor.from_pretrained(self.tmpdirname)

        self.assertIsInstance(processor.tokenizer, GemmaTokenizerFast)
        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())

        self.assertIsInstance(processor.feature_extractor, Gemma3nAudioFeatureExtractor)
        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())

    def test_save_load_pretrained_additional_features(self):
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()
        image_processor = self.get_image_processor()

        processor = Gemma3nProcessor(
            tokenizer=tokenizer, feature_extractor=feature_extractor, image_processor=image_processor
        )
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS-BOS)", eos_token="(EOS-EOS)")
        feature_extractor_add_kwargs = self.get_feature_extractor(dither=5.0, padding_value=1.0)

        processor = Gemma3nProcessor.from_pretrained(
            self.tmpdirname, bos_token="(BOS-BOS)", eos_token="(EOS-EOS)", dither=5.0, padding_value=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, GemmaTokenizerFast)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, Gemma3nAudioFeatureExtractor)

    @parameterized.expand([256, 512, 768, 1024])
    def test_image_processor(self, image_size: int):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()
        image_processor = self.get_image_processor()
        processor = Gemma3nProcessor(
            tokenizer=tokenizer, feature_extractor=feature_extractor, image_processor=image_processor
        )

        raw_image = np.random.randint(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
        input_image_processor = image_processor(raw_image, return_tensors="pt")
        input_processor = processor(text="Describe:", images=raw_image, return_tensors="pt")

        for key in input_image_processor.keys():
            self.assertAlmostEqual(input_image_processor[key].sum(), input_processor[key].sum(), delta=1e-2)
            if "pixel_values" in key:
                # NOTE: all images should be re-scaled to 768x768
                self.assertEqual(input_image_processor[key].shape, (1, 3, 768, 768))
                self.assertEqual(input_processor[key].shape, (1, 3, 768, 768))

    def test_audio_feature_extractor(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()
        image_processor = self.get_image_processor()
        processor = Gemma3nProcessor(
            tokenizer=tokenizer, feature_extractor=feature_extractor, image_processor=image_processor
        )

        raw_speech = floats_list((3, 1000))
        input_feat_extract = feature_extractor(raw_speech, return_tensors="pt")
        input_processor = processor(text="Transcribe:", audio=raw_speech, return_tensors="pt")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()
        image_processor = self.get_image_processor()
        processor = Gemma3nProcessor(
            tokenizer=tokenizer, feature_extractor=feature_extractor, image_processor=image_processor
        )

        input_str = "This is a test string"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str)

        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key][0])

    def test_tokenizer_decode(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()
        image_processor = self.get_image_processor()
        processor = Gemma3nProcessor(
            tokenizer=tokenizer, feature_extractor=feature_extractor, image_processor=image_processor
        )

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()
        image_processor = self.get_image_processor()
        processor = Gemma3nProcessor(
            tokenizer=tokenizer, feature_extractor=feature_extractor, image_processor=image_processor
        )

        for key in feature_extractor.model_input_names:
            self.assertIn(
                key,
                processor.model_input_names,
            )

        for key in image_processor.model_input_names:
            self.assertIn(
                key,
                processor.model_input_names,
            )
