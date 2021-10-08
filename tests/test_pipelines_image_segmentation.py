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

from transformers import (
    MODEL_FOR_IMAGE_SEGMENTATION_MAPPING,
    AutoFeatureExtractor,
    AutoModelForImageSegmentation,
    ImageSegmentationPipeline,
    is_vision_available,
    pipeline,
)
from transformers.testing_utils import (
    is_pipeline_test,
    nested_simplify,
    require_datasets,
    require_tf,
    require_timm,
    require_torch,
    require_vision,
    slow,
)

from .test_pipelines_common import ANY, PipelineTestCaseMeta


if is_vision_available():
    from PIL import Image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass


@require_vision
@require_timm
@require_torch
@is_pipeline_test
class ImageSegmentationPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_FOR_IMAGE_SEGMENTATION_MAPPING

    @require_datasets
    def run_pipeline_test(self, model, tokenizer, feature_extractor):
        image_segmenter = ImageSegmentationPipeline(model=model, feature_extractor=feature_extractor)
        outputs = image_segmenter("./tests/fixtures/tests_samples/COCO/000000039769.png", threshold=0.0)
        self.assertEqual(outputs, [{"score": ANY(float), "label": ANY(str), "mask": ANY(str)}] * 12)

        import datasets

        dataset = datasets.load_dataset("Narsil/image_dummy", "image", split="test")

        batch = [
            Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            # RGBA
            dataset[0]["file"],
            # LA
            dataset[1]["file"],
            # L
            dataset[2]["file"],
        ]
        outputs = image_segmenter(batch, threshold=0.0)

        self.assertEqual(len(batch), len(outputs))
        self.assertEqual(
            outputs,
            [
                [{"score": ANY(float), "label": ANY(str), "mask": ANY(str)}] * 12,
                [{"score": ANY(float), "label": ANY(str), "mask": ANY(str)}] * 12,
                [{"score": ANY(float), "label": ANY(str), "mask": ANY(str)}] * 12,
                [{"score": ANY(float), "label": ANY(str), "mask": ANY(str)}] * 12,
                [{"score": ANY(float), "label": ANY(str), "mask": ANY(str)}] * 12,
            ],
        )

    @require_tf
    @unittest.skip("Image segmentation not implemented in TF")
    def test_small_model_tf(self):
        pass

    @require_torch
    def test_small_model_pt(self):
        model_id = "mishig/tiny-detr-mobilenetsv3-panoptic"

        model = AutoModelForImageSegmentation.from_pretrained(model_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        image_segmenter = ImageSegmentationPipeline(model=model, feature_extractor=feature_extractor)

        outputs = image_segmenter("http://images.cocodataset.org/val2017/000000039769.jpg", threshold=0.0)
        for o in outputs:
            # shortening by hashing
            o["mask"] = hashlib.sha1(o["mask"].encode("UTF-8")).hexdigest()

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {
                    "score": 0.004,
                    "label": "LABEL_0",
                    "mask": "8423ef82b9a8e8790346bc452b557aa78ea997bc",
                },
                {
                    "score": 0.004,
                    "label": "LABEL_0",
                    "mask": "8423ef82b9a8e8790346bc452b557aa78ea997bc",
                },
            ],
        )

        outputs = image_segmenter(
            [
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                "http://images.cocodataset.org/val2017/000000039769.jpg",
            ],
            threshold=0.0,
        )
        for output in outputs:
            for o in output:
                o["mask"] = hashlib.sha1(o["mask"].encode("UTF-8")).hexdigest()

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [
                    {
                        "score": 0.004,
                        "label": "LABEL_0",
                        "mask": "8423ef82b9a8e8790346bc452b557aa78ea997bc",
                    },
                    {
                        "score": 0.004,
                        "label": "LABEL_0",
                        "mask": "8423ef82b9a8e8790346bc452b557aa78ea997bc",
                    },
                ],
                [
                    {
                        "score": 0.004,
                        "label": "LABEL_0",
                        "mask": "8423ef82b9a8e8790346bc452b557aa78ea997bc",
                    },
                    {
                        "score": 0.004,
                        "label": "LABEL_0",
                        "mask": "8423ef82b9a8e8790346bc452b557aa78ea997bc",
                    },
                ],
            ],
        )

    @require_torch
    @slow
    def test_integration_torch_image_segmentation(self):
        model_id = "facebook/detr-resnet-50-panoptic"

        image_segmenter = pipeline("image-segmentation", model=model_id)

        outputs = image_segmenter("http://images.cocodataset.org/val2017/000000039769.jpg")
        for o in outputs:
            o["mask"] = hashlib.sha1(o["mask"].encode("UTF-8")).hexdigest()

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9094, "label": "blanket", "mask": "f939d943609821ad27cdb92844f2754ad3735b52"},
                {"score": 0.9941, "label": "cat", "mask": "32913606de3958812ced0090df7b699abb6e2644"},
                {"score": 0.9987, "label": "remote", "mask": "f3988d35f3065f591fa6a0a9414614d98a9ca13e"},
                {"score": 0.9995, "label": "remote", "mask": "ff0d541ace4fe386fc14ced0c546490a8e7001d7"},
                {"score": 0.9722, "label": "couch", "mask": "543c3244b291c4aec134f1d8f92af553da795529"},
                {"score": 0.9994, "label": "cat", "mask": "891313e21290200e6169613e6a9cb7aff9e7b22f"},
            ],
        )

        outputs = image_segmenter(
            [
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                "http://images.cocodataset.org/val2017/000000039769.jpg",
            ],
            threshold=0.0,
        )
        for output in outputs:
            for o in output:
                o["mask"] = hashlib.sha1(o["mask"].encode("UTF-8")).hexdigest()

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [
                    {"score": 0.9094, "label": "blanket", "mask": "f939d943609821ad27cdb92844f2754ad3735b52"},
                    {"score": 0.9941, "label": "cat", "mask": "32913606de3958812ced0090df7b699abb6e2644"},
                    {"score": 0.9987, "label": "remote", "mask": "f3988d35f3065f591fa6a0a9414614d98a9ca13e"},
                    {"score": 0.9995, "label": "remote", "mask": "ff0d541ace4fe386fc14ced0c546490a8e7001d7"},
                    {"score": 0.9722, "label": "couch", "mask": "543c3244b291c4aec134f1d8f92af553da795529"},
                    {"score": 0.9994, "label": "cat", "mask": "891313e21290200e6169613e6a9cb7aff9e7b22f"},
                ],
                [
                    {"score": 0.9094, "label": "blanket", "mask": "f939d943609821ad27cdb92844f2754ad3735b52"},
                    {"score": 0.9941, "label": "cat", "mask": "32913606de3958812ced0090df7b699abb6e2644"},
                    {"score": 0.9987, "label": "remote", "mask": "f3988d35f3065f591fa6a0a9414614d98a9ca13e"},
                    {"score": 0.9995, "label": "remote", "mask": "ff0d541ace4fe386fc14ced0c546490a8e7001d7"},
                    {"score": 0.9722, "label": "couch", "mask": "543c3244b291c4aec134f1d8f92af553da795529"},
                    {"score": 0.9994, "label": "cat", "mask": "891313e21290200e6169613e6a9cb7aff9e7b22f"},
                ],
            ],
        )

    @require_torch
    @slow
    def test_threshold(self):
        threshold = 0.999
        model_id = "facebook/detr-resnet-50-panoptic"

        image_segmenter = pipeline("image-segmentation", model=model_id)

        outputs = image_segmenter("http://images.cocodataset.org/val2017/000000039769.jpg", threshold=threshold)

        for o in outputs:
            o["mask"] = hashlib.sha1(o["mask"].encode("UTF-8")).hexdigest()

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9995, "label": "remote", "mask": "ff0d541ace4fe386fc14ced0c546490a8e7001d7"},
                {"score": 0.9994, "label": "cat", "mask": "891313e21290200e6169613e6a9cb7aff9e7b22f"},
            ],
        )
