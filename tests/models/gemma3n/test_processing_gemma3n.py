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

from transformers.models.gemma3n import Gemma3nProcessor
from transformers.testing_utils import require_sentencepiece, require_torch, require_torchaudio, require_vision

from ...test_processing_common import ProcessorTesterMixin
from .test_feature_extraction_gemma3n import floats_list


# TODO: omni-modal processor can't run tests from `ProcessorTesterMixin`
@require_torch
@require_torchaudio
@require_vision
@require_sentencepiece
class Gemma3nProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Gemma3nProcessor
    model_id = "hf-internal-testing/namespace-google-repo_name-gemma-3n-E4B-it"

    def prepare_image_inputs(self, batch_size: int | None = None, nested: bool = False):
        return super().prepare_image_inputs(batch_size=batch_size, nested=True)

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.boi_token

    def test_audio_feature_extractor(self):
        processor = self.get_processor()
        feature_extractor = self.get_component("feature_extractor")

        raw_speech = floats_list((3, 1000))
        input_feat_extract = feature_extractor(raw_speech, return_tensors="pt")
        input_processor = processor(text="Transcribe:", audio=raw_speech, return_tensors="pt")

        for key in input_feat_extract:
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)
