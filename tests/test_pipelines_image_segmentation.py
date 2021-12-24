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

    def get_test_pipeline(self, model, tokenizer, feature_extractor):
        image_segmenter = ImageSegmentationPipeline(model=model, feature_extractor=feature_extractor)
        return image_segmenter, [
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
        ]

    def run_pipeline_test(self, image_segmenter, examples):
        outputs = image_segmenter("./tests/fixtures/tests_samples/COCO/000000039769.png", threshold=0.0)
        self.assertEqual(outputs, [{"score": ANY(float), "label": ANY(str), "mask": ANY(str)}] * 12)

        import datasets

        dataset = datasets.load_dataset("hf-internal-testing/fixtures_image_utils", "image", split="test")

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
                    "mask": "4276f7db4ca2983b2666f7e0c102d8186aed20be",
                },
                {
                    "score": 0.004,
                    "label": "LABEL_0",
                    "mask": "4276f7db4ca2983b2666f7e0c102d8186aed20be",
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
                        "mask": "4276f7db4ca2983b2666f7e0c102d8186aed20be",
                    },
                    {
                        "score": 0.004,
                        "label": "LABEL_0",
                        "mask": "4276f7db4ca2983b2666f7e0c102d8186aed20be",
                    },
                ],
                [
                    {
                        "score": 0.004,
                        "label": "LABEL_0",
                        "mask": "4276f7db4ca2983b2666f7e0c102d8186aed20be",
                    },
                    {
                        "score": 0.004,
                        "label": "LABEL_0",
                        "mask": "4276f7db4ca2983b2666f7e0c102d8186aed20be",
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
                {"score": 0.9094, "label": "blanket", "mask": "36517c16f4356f7af4b298f4eae387f9fe37eaf8"},
                {"score": 0.9941, "label": "cat", "mask": "d63196cbe08c7655c158dbabbc5e6b413cbb3b2d"},
                {"score": 0.9987, "label": "remote", "mask": "4e190e0c3934ad852aaa51aa2c54e314b9a1152e"},
                {"score": 0.9995, "label": "remote", "mask": "39dc07a07238048a06b0c2474de01ba3c09cc44f"},
                {"score": 0.9722, "label": "couch", "mask": "df5815755b6bcf328f6b6811f8794cad26f79b35"},
                {"score": 0.9994, "label": "cat", "mask": "88b37bd2202c750cc9dd191518050a9b0ca5228c"},
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
                    {"score": 0.9094, "label": "blanket", "mask": "36517c16f4356f7af4b298f4eae387f9fe37eaf8"},
                    {"score": 0.9941, "label": "cat", "mask": "d63196cbe08c7655c158dbabbc5e6b413cbb3b2d"},
                    {"score": 0.9987, "label": "remote", "mask": "4e190e0c3934ad852aaa51aa2c54e314b9a1152e"},
                    {"score": 0.9995, "label": "remote", "mask": "39dc07a07238048a06b0c2474de01ba3c09cc44f"},
                    {"score": 0.9722, "label": "couch", "mask": "df5815755b6bcf328f6b6811f8794cad26f79b35"},
                    {"score": 0.9994, "label": "cat", "mask": "88b37bd2202c750cc9dd191518050a9b0ca5228c"},
                ],
                [
                    {"score": 0.9094, "label": "blanket", "mask": "36517c16f4356f7af4b298f4eae387f9fe37eaf8"},
                    {"score": 0.9941, "label": "cat", "mask": "d63196cbe08c7655c158dbabbc5e6b413cbb3b2d"},
                    {"score": 0.9987, "label": "remote", "mask": "4e190e0c3934ad852aaa51aa2c54e314b9a1152e"},
                    {"score": 0.9995, "label": "remote", "mask": "39dc07a07238048a06b0c2474de01ba3c09cc44f"},
                    {"score": 0.9722, "label": "couch", "mask": "df5815755b6bcf328f6b6811f8794cad26f79b35"},
                    {"score": 0.9994, "label": "cat", "mask": "88b37bd2202c750cc9dd191518050a9b0ca5228c"},
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
                {"score": 0.9995, "label": "remote", "mask": "39dc07a07238048a06b0c2474de01ba3c09cc44f"},
                {"score": 0.9994, "label": "cat", "mask": "88b37bd2202c750cc9dd191518050a9b0ca5228c"},
            ],
        )
