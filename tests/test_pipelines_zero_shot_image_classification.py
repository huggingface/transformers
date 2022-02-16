# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoFeatureExtractor,
    AutoModel,
    AutoTokenizer,
    is_vision_available,
)
from transformers.pipelines import ZeroShotImageClassificationPipeline, pipeline
from transformers.testing_utils import is_pipeline_test, require_tf, require_torch, require_vision

from .test_pipelines_common import ANY, PipelineTestCaseMeta


if is_vision_available():
    from PIL import Image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass


@require_vision
@require_torch
@is_pipeline_test
class ZeroShotImageClassificationPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING

    def get_test_pipeline(self, model, tokenizer, feature_extractor):
        if tokenizer is None:
            # Side effect of no Fast Tokenizer class for these model, so skipping
            # But the slow tokenizer test should still run as they're quite small
            self.skipTest("No tokenizer available")
            return
            # return None, None

        speech_recognizer = ZeroShotImageClassificationPipeline(
            model=model, tokenizer=tokenizer, feature_extractor=feature_extractor
        )

        # test with a raw waveform
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        image2 = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        return speech_recognizer, [image, image2]

    def run_pipeline_test(self, pipe, examples):
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        outputs = pipe(image, candidate_labels=["A", "B"])
        self.assertEqual(outputs, {"text": ANY(str)})

        # Batching
        outputs = pipe([image] * 3, batch_size=2, candidate_labels=["A", "B"])

    @require_tf
    def test_small_model_tf(self):
        self.skipTest("Not implemented in Tensorflow")

    @require_torch
    def test_small_model_pt(self):
        speech_recognizer = pipeline(
            task="zero-shot-image-classification",
            model="hf-internal-testing/tiny-random-clip",
        )
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        output = speech_recognizer(image, candidate_labels=["A", "B", "C"])
        self.assertEqual(output, {"text": "(Applaudissements)"})
