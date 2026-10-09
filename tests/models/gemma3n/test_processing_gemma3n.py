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

import unittest

import numpy as np

from transformers.models.gemma3n import Gemma3nProcessor
from transformers.testing_utils import (
    require_sentencepiece,
    require_torch,
    require_torchaudio,
    require_vision,
)

from ...test_processing_common import ProcessorTesterMixin
from .test_feature_extraction_gemma3n import floats_list


# TODO: omni-modal processor can't run tests from `ProcessorTesterMixin`
@require_torch
@require_torchaudio
@require_vision
@require_sentencepiece
class Gemma3nProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Gemma3nProcessor
    # Tiny processor created with make_tiny_processor.py from "hf-internal-testing/namespace-google-repo_name-gemma-3n-E4B-it"
    tiny_model_id = "hf-internal-testing/tiny-processor-gemma3n"

    def prepare_images_inputs(self, batch_size: int | None = None, nested: bool = False):
        return super().prepare_images_inputs(batch_size=batch_size, nested=True)

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.boi_token

    def test_get_num_multimodal_tokens_matches_processor_call(self):
        "Tests that the helper used internally in vLLM works correctly"

        processor = self.get_processor()
        image_sizes = [(100, 100), (300, 100), (500, 30), (213, 167)]
        # Overwritten because Gemma3n needs nested image inputs and its own token type ids
        images = [[np.random.randint(255, size=(h, w, 3), dtype=np.uint8)] for h, w in image_sizes]
        inputs = processor(
            text=[f"This is an image {processor.image_token}"] * len(images),
            images=images,
            padding=True,
            return_tensors="pt",
        )
        num_image_tokens_from_call = inputs.token_type_ids.eq(1).sum(-1).tolist()
        num_image_tokens_from_helper = processor._get_num_multimodal_tokens(image_sizes=image_sizes)
        self.assertListEqual(num_image_tokens_from_call, num_image_tokens_from_helper["num_image_tokens"])

        sampling_rate = processor.feature_extractor.sampling_rate
        audio_lengths = [sampling_rate, 4 * sampling_rate]
        inputs = processor(
            text=[f"This is an audio {processor.audio_token}"] * len(audio_lengths),
            audio=[np.zeros(length, dtype=np.float32) for length in audio_lengths],
            padding=True,
            return_tensors="pt",
        )
        num_audio_tokens_from_call = inputs.token_type_ids.eq(3).sum(-1).tolist()
        num_audio_tokens_from_helper = processor._get_num_multimodal_tokens(audio_lengths=audio_lengths)
        self.assertListEqual(num_audio_tokens_from_call, num_audio_tokens_from_helper["num_audio_tokens"])

    def test_audio_feature_extractor(self):
        processor = self.get_processor()
        feature_extractor = self.get_component("feature_extractor")

        raw_speech = floats_list((3, 1000))
        input_feat_extract = feature_extractor(raw_speech, return_tensors="pt")
        input_processor = processor(text="Transcribe:", audio=raw_speech, return_tensors="pt")

        for key in input_feat_extract:
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)
