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

from transformers import MODEL_FOR_DEPTH_ESTIMATION_MAPPING, is_torch_available, is_vision_available
from transformers.pipelines import DepthEstimationPipeline, pipeline
from transformers.testing_utils import nested_simplify, require_tf, require_timm, require_torch, require_vision, slow

from .test_pipelines_common import ANY, PipelineTestCaseMeta


if is_torch_available():
    import torch

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
class DepthEstimationPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_FOR_DEPTH_ESTIMATION_MAPPING

    def get_test_pipeline(self, model, tokenizer, processor):
        depth_estimator = DepthEstimationPipeline(model=model, image_processor=processor)
        return depth_estimator, [
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
        ]

    def run_pipeline_test(self, depth_estimator, examples):
        outputs = depth_estimator("./tests/fixtures/tests_samples/COCO/000000039769.png")
        self.assertEqual({"predicted_depth": ANY(torch.Tensor), "depth": ANY(Image.Image)}, outputs)
        import datasets

        dataset = datasets.load_dataset("hf-internal-testing/fixtures_image_utils", "image", split="test")
        outputs = depth_estimator(
            [
                Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                # RGBA
                dataset[0]["file"],
                # LA
                dataset[1]["file"],
                # L
                dataset[2]["file"],
            ]
        )
        self.assertEqual(
            [
                {"predicted_depth": ANY(torch.Tensor), "depth": ANY(Image.Image)},
                {"predicted_depth": ANY(torch.Tensor), "depth": ANY(Image.Image)},
                {"predicted_depth": ANY(torch.Tensor), "depth": ANY(Image.Image)},
                {"predicted_depth": ANY(torch.Tensor), "depth": ANY(Image.Image)},
                {"predicted_depth": ANY(torch.Tensor), "depth": ANY(Image.Image)},
            ],
            outputs,
        )

    @require_tf
    @unittest.skip("Depth estimation is not implemented in TF")
    def test_small_model_tf(self):
        pass

    @slow
    @require_torch
    def test_large_model_pt(self):
        model_id = "Intel/dpt-large"
        depth_estimator = pipeline("depth-estimation", model=model_id)
        outputs = depth_estimator("http://images.cocodataset.org/val2017/000000039769.jpg")
        outputs["depth"] = hashimage(outputs["depth"])

        # This seems flaky.
        # self.assertEqual(outputs["depth"], "1a39394e282e9f3b0741a90b9f108977")
        self.assertEqual(nested_simplify(outputs["predicted_depth"].max().item()), 29.304)
        self.assertEqual(nested_simplify(outputs["predicted_depth"].min().item()), 2.662)

    @require_torch
    def test_small_model_pt(self):
        # This is highly irregular to have no small tests.
        self.skipTest("There is not hf-internal-testing tiny model for either GLPN nor DPT")
