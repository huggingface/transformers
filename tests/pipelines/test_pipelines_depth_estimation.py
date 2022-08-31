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

import hashlib
import unittest

from datasets import load_dataset

from transformers import (
    MODEL_FOR_DEPTH_ESTIMATION_MAPPING,
    is_vision_available
)
from transformers.pipelines import DepthEstimationPipeline, depth_estimation, pipeline
from transformers.testing_utils import (
    is_pipeline_test,
    require_tf,
    require_timm,
    require_torch,
    require_vision,
)

from .test_pipelines_common import ANY, PipelineTestCaseMeta


if is_vision_available():
    from PIL import Image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass

def hashimage(image: Image) -> str:
    m = hashlib.md5(image.tobytes())
    return m.hexdigest()

@require_vision
@require_timm
@require_torch
@is_pipeline_test
class DepthEstimationPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_FOR_DEPTH_ESTIMATION_MAPPING

    def get_test_pipeline(self, model, tokenizer, feature_extractor):
        depth_estimator = DepthEstimationPipeline(model=model, feature_extractor=feature_extractor)
        return depth_estimator, [
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
        ]

    def run_pipeline_test(self, depth_estimator, examples):
        ...

    @require_tf
    @unittest.skip("Depth estimation is not implemented in TF")
    def test_small_model_tf(self):
        pass

    @require_torch
    def test_small_model_pt(self):
        model_id = "Intel/dpt-large"
        depth_estimator = pipeline("depth-estimation",model=model_id)
        outputs = depth_estimator("http://images.cocodataset.org/val2017/000000039769.jpg")
        outputs["depth"]=hashimage(outputs["depth"])
        self.assertEqual(outputs["depth"],"906b03064c8c68dae2b1f1f96c7c48f5")
