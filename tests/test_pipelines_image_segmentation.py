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
        def test_outputs(outputs):
            png_string, segments_info = outputs["png_string"], outputs["segments_info"]
            self.assertEqual(png_string, ANY(str))
            self.assertGreater(len(segments_info), 0)
            for segment_info in segments_info:
                self.assertEqual(
                    segment_info,
                    {
                        "id": ANY(int),
                        "score": ANY(float),
                        "label": ANY(str),
                    },
                )

        image_segmenter = ImageSegmentationPipeline(model=model, feature_extractor=feature_extractor)
        outputs = image_segmenter("./tests/fixtures/tests_samples/COCO/000000039769.png", threshold=0.0)
        test_outputs(outputs)

        subtasks = [None, "panoptic", "instance", "semantic"]
        for subtask in subtasks:
            outputs = image_segmenter(
                "./tests/fixtures/tests_samples/COCO/000000039769.png", threshold=0.0, subtask=subtask
            )
            test_outputs(outputs)

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
        batch_outputs = image_segmenter(batch, threshold=0.0)

        self.assertEqual(len(batch), len(batch_outputs))
        for outputs in batch_outputs:
            test_outputs(outputs)

    @require_tf
    @unittest.skip("Image segmentation not implemented in TF")
    def test_small_model_tf(self):
        pass

    @require_torch
    def test_small_model_pt(self):
        model_id = "mishig/tiny-detr-mobilenetsv3-panoptic"

        model = AutoModelForImageSegmentation.from_pretrained(model_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
        image_segmenter = ImageSegmentationPipeline(model=model, feature_extractor=feature_extractor)

        outputs = image_segmenter("http://images.cocodataset.org/val2017/000000039769.jpg", threshold=0.0)

        # shortening by hashing
        outputs["png_string"] = hashlib.sha1(outputs["png_string"].encode("UTF-8")).hexdigest()
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            {
                "png_string": "2a286c4660866fcaf4c9da6ba565ff658bd663ca",
                "segments_info": [
                    {"id": 0, "label": "LABEL_0", "score": 0.004},
                ],
            },
        )

        outputs = image_segmenter(
            [
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                "http://images.cocodataset.org/val2017/000000039769.jpg",
            ],
            threshold=0.0,
        )
        for o in outputs:
            o["png_string"] = hashlib.sha1(o["png_string"].encode("UTF-8")).hexdigest()

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {
                    "png_string": "2a286c4660866fcaf4c9da6ba565ff658bd663ca",
                    "segments_info": [
                        {"id": 0, "label": "LABEL_0", "score": 0.004},
                    ],
                },
                {
                    "png_string": "2a286c4660866fcaf4c9da6ba565ff658bd663ca",
                    "segments_info": [
                        {"id": 0, "label": "LABEL_0", "score": 0.004},
                    ],
                },
            ],
        )

    @require_torch
    @slow
    def test_integration_torch_image_segmentation(self):
        model_id = "facebook/detr-resnet-50-panoptic"

        image_segmenter = pipeline("image-segmentation", model=model_id)

        outputs = image_segmenter("http://images.cocodataset.org/val2017/000000039769.jpg")
        outputs["png_string"] = hashlib.sha1(outputs["png_string"].encode("UTF-8")).hexdigest()
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            {
                "segments_info": [
                    {"id": 0, "score": 0.9094, "label": "blanket"},
                    {"id": 1, "score": 0.9941, "label": "cat"},
                    {"id": 2, "score": 0.9987, "label": "remote"},
                    {"id": 3, "score": 0.9995, "label": "remote"},
                    {"id": 4, "score": 0.9722, "label": "couch"},
                    {"id": 5, "score": 0.9994, "label": "cat"},
                ],
                "png_string": "d0c3d58467818e568604e369c945b18ce05e28c0",
            },
        )

        outputs = image_segmenter(
            [
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                "http://images.cocodataset.org/val2017/000000039769.jpg",
            ],
            threshold=0.0,
        )
        for o in outputs:
            o["png_string"] = hashlib.sha1(o["png_string"].encode("UTF-8")).hexdigest()

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {
                    "segments_info": [
                        {"id": 0, "score": 0.9094, "label": "blanket"},
                        {"id": 1, "score": 0.9941, "label": "cat"},
                        {"id": 2, "score": 0.9987, "label": "remote"},
                        {"id": 3, "score": 0.9995, "label": "remote"},
                        {"id": 4, "score": 0.9722, "label": "couch"},
                        {"id": 5, "score": 0.9994, "label": "cat"},
                    ],
                    "png_string": "d0c3d58467818e568604e369c945b18ce05e28c0",
                },
                {
                    "segments_info": [
                        {"id": 0, "score": 0.9094, "label": "blanket"},
                        {"id": 1, "score": 0.9941, "label": "cat"},
                        {"id": 2, "score": 0.9987, "label": "remote"},
                        {"id": 3, "score": 0.9995, "label": "remote"},
                        {"id": 4, "score": 0.9722, "label": "couch"},
                        {"id": 5, "score": 0.9994, "label": "cat"},
                    ],
                    "png_string": "d0c3d58467818e568604e369c945b18ce05e28c0",
                },
            ],
        )

    @require_torch
    @slow
    def test_subtask(self):
        model_id = "facebook/detr-resnet-50-panoptic"

        image_segmenter = pipeline("image-segmentation", model=model_id)

        outputs = image_segmenter("http://images.cocodataset.org/val2017/000000039769.jpg", subtask="panoptic")
        outputs["png_string"] = hashlib.sha1(outputs["png_string"].encode("UTF-8")).hexdigest()
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            {
                "segments_info": [
                    {"id": 0, "score": 0.9094, "label": "blanket"},
                    {"id": 1, "score": 0.9941, "label": "cat"},
                    {"id": 2, "score": 0.9987, "label": "remote"},
                    {"id": 3, "score": 0.9995, "label": "remote"},
                    {"id": 4, "score": 0.9722, "label": "couch"},
                    {"id": 5, "score": 0.9994, "label": "cat"},
                ],
                "png_string": "d0c3d58467818e568604e369c945b18ce05e28c0",
            },
        )

        outputs = image_segmenter("http://images.cocodataset.org/val2017/000000039769.jpg", subtask="instance")
        outputs["png_string"] = hashlib.sha1(outputs["png_string"].encode("UTF-8")).hexdigest()
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            {
                "segments_info": [
                    {"id": 1, "score": 0.9094, "label": "blanket"},
                    {"id": 2, "score": 0.9941, "label": "cat"},
                    {"id": 3, "score": 0.9987, "label": "remote"},
                    {"id": 4, "score": 0.9995, "label": "remote"},
                    {"id": 5, "score": 0.9722, "label": "couch"},
                    {"id": 6, "score": 0.9994, "label": "cat"},
                ],
                "png_string": "14e626f2b18c2614b6aef6dbbf1ea2b8880a7bae",
            },
        )

    @require_torch
    @slow
    def test_threshold(self):
        threshold = 0.995
        model_id = "facebook/detr-resnet-50-panoptic"

        image_segmenter = pipeline("image-segmentation", model=model_id)

        outputs = image_segmenter("http://images.cocodataset.org/val2017/000000039769.jpg", threshold=threshold)
        outputs["png_string"] = hashlib.sha1(outputs["png_string"].encode("UTF-8")).hexdigest()
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            {
                "segments_info": [
                    {"id": 0, "score": 0.9987, "label": "remote"},
                    {"id": 1, "score": 0.9995, "label": "remote"},
                    {"id": 2, "score": 0.9994, "label": "cat"},
                ],
                "png_string": "6a7d047acc1346ce5c3e3fe9b929b871a5d0a247",
            },
        )
