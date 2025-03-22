# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
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
import tempfile
import unittest

import numpy as np
import pytest

from transformers import (
    AutoProcessor,
    Qwen2_5OmniProcessor,
    Qwen2Tokenizer,
    WhisperFeatureExtractor,
)
from transformers.testing_utils import require_torch, require_torchaudio, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin, floats_list


if is_vision_available():
    from transformers import Qwen2VLImageProcessor


@require_vision
@require_torch
@require_torchaudio
class Qwen2_5OmniProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Qwen2_5OmniProcessor

    #  text + audio kwargs testing
    @require_torch
    def test_tokenizer_defaults_preserved_by_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")
        feature_extractor = self.get_component("feature_extractor")
        if hasattr(self, "get_tokenizer"):
            tokenizer = self.get_tokenizer(max_length=117, padding="max_length")
        elif hasattr(self, "get_component"):
            tokenizer = self.get_component("tokenizer", max_length=117, padding="max_length")
        else:
            self.assertTrue(False, "Processor doesn't have get_tokenizer or get_component defined")
        if not tokenizer.pad_token:
            tokenizer.pad_token = "[TEST_PAD]"
        if "omni_processor" not in self.processor_class.attributes:
            self.skipTest(f"omni_processor attribute not present in {self.processor_class}")
        omni_processor = self.get_component("omni_processor")
        processor = self.processor_class(
            tokenizer=tokenizer, feature_extractor=feature_extractor, omni_processor=omni_processor
        )
        self.skip_processor_without_typed_kwargs(processor)
        input_str = "lower newer"
        raw_speech = floats_list((3, 1000))
        inputs = processor(text=input_str, audio=raw_speech, return_tensors="pt")
        if "input_ids" in inputs:
            self.assertEqual(len(inputs["input_ids"][0]), 2)
        elif "labels" in inputs:
            self.assertEqual(len(inputs["labels"][0]), 2)

    @require_torch
    @require_vision
    def test_structured_kwargs_audio_nested(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")
        feature_extractor = self.get_component("feature_extractor")
        if hasattr(self, "get_tokenizer"):
            tokenizer = self.get_tokenizer()
        elif hasattr(self, "get_component"):
            tokenizer = self.get_component("tokenizer")
        if not tokenizer.pad_token:
            tokenizer.pad_token = "[TEST_PAD]"
        if "omni_processor" not in self.processor_class.attributes:
            self.skipTest(f"omni_processor attribute not present in {self.processor_class}")
        omni_processor = self.get_component("omni_processor")
        processor = self.processor_class(
            tokenizer=tokenizer, feature_extractor=feature_extractor, omni_processor=omni_processor
        )
        self.skip_processor_without_typed_kwargs(processor)

        input_str = ["lower newer"]
        raw_speech = floats_list((3, 1000))

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "audio_kwargs": {"max_length": 2},
        }

        inputs = processor(text=input_str, audio=raw_speech, **all_kwargs)
        if "input_ids" in inputs:
            self.assertEqual(len(inputs["input_ids"][0]), 2)
        elif "labels" in inputs:
            self.assertEqual(len(inputs["labels"][0]), 2)

    @require_torch
    def test_unstructured_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")
        feature_extractor = self.get_component("feature_extractor")
        if hasattr(self, "get_tokenizer"):
            tokenizer = self.get_tokenizer(max_length=117)
        elif hasattr(self, "get_component"):
            tokenizer = self.get_component("tokenizer", max_length=117)
        if not tokenizer.pad_token:
            tokenizer.pad_token = "[TEST_PAD]"
        if "omni_processor" not in self.processor_class.attributes:
            self.skipTest(f"omni_processor attribute not present in {self.processor_class}")
        omni_processor = self.get_component("omni_processor")
        processor = self.processor_class(
            tokenizer=tokenizer, feature_extractor=feature_extractor, omni_processor=omni_processor
        )
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        raw_speech = floats_list((3, 1000))
        inputs = processor(
            text=input_str,
            audio=raw_speech,
            return_tensors="pt",
            padding="max_length",
            max_length=76,
        )

        if "input_ids" in inputs:
            self.assertEqual(len(inputs["input_ids"][0]), 76)
        elif "labels" in inputs:
            self.assertEqual(len(inputs["labels"][0]), 76)

    @require_torch
    def test_doubly_passed_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")
        feature_extractor = self.get_component("feature_extractor")
        if hasattr(self, "get_tokenizer"):
            tokenizer = self.get_tokenizer()
        elif hasattr(self, "get_component"):
            tokenizer = self.get_component("tokenizer")
        if not tokenizer.pad_token:
            tokenizer.pad_token = "[TEST_PAD]"
        if "omni_processor" not in self.processor_class.attributes:
            self.skipTest(f"omni_processor attribute not present in {self.processor_class}")
        omni_processor = self.get_component("omni_processor")
        self.processor_class(tokenizer=tokenizer, feature_extractor=feature_extractor, omni_processor=omni_processor)

    @require_torch
    def test_kwargs_overrides_default_tokenizer_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")
        feature_extractor = self.get_component("feature_extractor")
        if hasattr(self, "get_tokenizer"):
            tokenizer = self.get_tokenizer(max_length=117)
        elif hasattr(self, "get_component"):
            tokenizer = self.get_component("tokenizer", max_length=117)
        if not tokenizer.pad_token:
            tokenizer.pad_token = "[TEST_PAD]"
        if "omni_processor" not in self.processor_class.attributes:
            self.skipTest(f"omni_processor attribute not present in {self.processor_class}")
        omni_processor = self.get_component("omni_processor")
        self.processor_class(tokenizer=tokenizer, feature_extractor=feature_extractor, omni_processor=omni_processor)

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_omni_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).omni_processor

    def get_feature_extractor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).feature_extractor

    def prepare_audio_inputs(self):
        """This function prepares a list of numpy audios."""
        audio_inputs = [np.random.rand(160000) * 2 - 1] * 3  # batch-size=3
        return audio_inputs

    def test_save_load_pretrained_default(self):
        omni_processor = self.get_omni_processor()
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = Qwen2_5OmniProcessor(
            omni_processor=omni_processor, feature_extractor=feature_extractor, tokenizer=tokenizer
        )
        processor.save_pretrained(self.tmpdirname)
        processor = Qwen2_5OmniProcessor.from_pretrained(self.tmpdirname, use_fast=False)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(processor.omni_processor.to_json_string(), omni_processor.to_json_string())
        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.tokenizer, Qwen2Tokenizer)
        self.assertIsInstance(processor.omni_processor, Qwen2VLImageProcessor)
        self.assertIsInstance(processor.feature_extractor, WhisperFeatureExtractor)

    def test_omni_processor(self):
        omni_processor = self.get_omni_processor()
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = Qwen2_5OmniProcessor(
            omni_processor=omni_processor, feature_extractor=feature_extractor, tokenizer=tokenizer
        )

        image_input = self.prepare_image_inputs()

        input_image_proc = omni_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, text="dummy", return_tensors="np")

        for key in input_image_proc.keys():
            self.assertAlmostEqual(input_image_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_processor(self):
        omni_processor = self.get_omni_processor()
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = Qwen2_5OmniProcessor(
            omni_processor=omni_processor, feature_extractor=feature_extractor, tokenizer=tokenizer
        )

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        audio_input = self.prepare_audio_inputs()
        inputs = processor(text=input_str, images=image_input, audios=audio_input)
        keys = list(inputs.keys())
        self.assertListEqual(
            keys,
            [
                "input_ids",
                "attention_mask",
                "pixel_values",
                "image_grid_thw",
                "feature_attention_mask",
                "input_features",
            ],
        )

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

        # test if it raises when no text is passed
        with pytest.raises(ValueError):
            processor(images=image_input)

    def test_model_input_names(self):
        omni_processor = self.get_omni_processor()
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = Qwen2_5OmniProcessor(
            omni_processor=omni_processor, feature_extractor=feature_extractor, tokenizer=tokenizer
        )

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        video_inputs = self.prepare_video_inputs()
        audio_input = self.prepare_audio_inputs()

        inputs = processor(text=input_str, images=image_input, videos=video_inputs, audios=audio_input)
        self.assertListEqual(sorted(inputs.keys()), sorted(processor.model_input_names))
