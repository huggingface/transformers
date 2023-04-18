# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import Dict

import numpy as np
import requests
from datasets import load_dataset

from transformers import (
    MODEL_FOR_AUTOMATIC_MASK_GENERATION_MAPPING,
    AutoImageProcessor,
    AutomaticMaskGenerationPipeline,
    is_vision_available,
    pipeline,
)
from transformers.testing_utils import (
    is_pipeline_test,
    nested_simplify,
    require_tf,
    require_torch,
    require_vision,
    slow,
)


if is_vision_available():
    from PIL import Image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass


def hashimage(image: Image) -> str:
    m = hashlib.md5(image.tobytes())
    return m.hexdigest()[:10]


def mask_to_test_readable(mask: Image) -> Dict:
    npimg = np.array(mask)
    white_pixels = (npimg == 255).sum()
    shape = npimg.shape
    return {"hash": hashimage(mask), "white_pixels": white_pixels, "shape": shape}


def mask_to_test_readable_only_shape(mask: Image) -> Dict:
    npimg = np.array(mask)
    shape = npimg.shape
    return {"shape": shape}


@is_pipeline_test
@require_vision
@require_torch
class AutomaticMaskGenerationPipelineTests(unittest.TestCase):
    model_mapping = dict(
        (
            list(MODEL_FOR_AUTOMATIC_MASK_GENERATION_MAPPING.items())
            if MODEL_FOR_AUTOMATIC_MASK_GENERATION_MAPPING
            else []
        )
    )

    def get_test_pipeline(self, model, tokenizer, processor):
        image_segmenter = AutomaticMaskGenerationPipeline(model=model, image_processor=processor)
        return image_segmenter, [
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
        ]

    @require_tf
    @unittest.skip("Image segmentation not implemented in TF")
    def test_small_model_tf(self):
        pass

    @require_torch
    def test_small_model_pt(self):
        model_id = "younesb/sam-vit-h"

        model = SamModelForMaskGeneration.from_pretrained(model_id)
        image_processor = AutoImageProcessor.from_pretrained(model_id)
        image_segmenter = AutomaticMaskGenerationPipeline(
            model=model,
            image_processor=image_processor,
            subtask="panoptic",
            threshold=0.0,
            mask_threshold=0.0,
            overlap_mask_area_threshold=0.0,
        )

        outputs = image_segmenter(
            "http://images.cocodataset.org/val2017/000000039769.jpg",
        )

        # Shortening by hashing
        for o in outputs:
            o["mask"] = mask_to_test_readable(o["mask"])

        # This is extremely brittle, and those values are made specific for the CI.
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {
                    "score": 0.004,
                    "label": "LABEL_215",
                    "mask": {"hash": "a01498ca7c", "shape": (480, 640), "white_pixels": 307200},
                },
            ],
        )

        outputs = image_segmenter(
            [
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                "http://images.cocodataset.org/val2017/000000039769.jpg",
            ],
        )
        for output in outputs:
            for o in output:
                o["mask"] = mask_to_test_readable(o["mask"])

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [
                    {
                        "score": 0.004,
                        "label": "LABEL_215",
                        "mask": {"hash": "a01498ca7c", "shape": (480, 640), "white_pixels": 307200},
                    },
                ],
                [
                    {
                        "score": 0.004,
                        "label": "LABEL_215",
                        "mask": {"hash": "a01498ca7c", "shape": (480, 640), "white_pixels": 307200},
                    },
                ],
            ],
        )

        output = image_segmenter("http://images.cocodataset.org/val2017/000000039769.jpg", subtask="instance")
        for o in output:
            o["mask"] = mask_to_test_readable(o["mask"])
        self.assertEqual(
            nested_simplify(output, decimals=4),
            [
                {
                    "score": 0.004,
                    "label": "LABEL_215",
                    "mask": {"hash": "a01498ca7c", "shape": (480, 640), "white_pixels": 307200},
                },
            ],
        )

        # This must be surprising to the reader.
        # The `panoptic` returns only LABEL_215, and this returns 3 labels.
        #
        output = image_segmenter("http://images.cocodataset.org/val2017/000000039769.jpg", subtask="semantic")

        output_masks = [o["mask"] for o in output]

        # page links (to visualize)
        expected_masks = [
            "https://huggingface.co/datasets/hf-internal-testing/mask-for-image-segmentation-tests/blob/main/mask_0.png",
            "https://huggingface.co/datasets/hf-internal-testing/mask-for-image-segmentation-tests/blob/main/mask_1.png",
            "https://huggingface.co/datasets/hf-internal-testing/mask-for-image-segmentation-tests/blob/main/mask_2.png",
        ]
        # actual links to get files
        expected_masks = [x.replace("/blob/", "/resolve/") for x in expected_masks]
        expected_masks = [Image.open(requests.get(image, stream=True).raw) for image in expected_masks]

        # Convert masks to numpy array
        output_masks = [np.array(x) for x in output_masks]
        expected_masks = [np.array(x) for x in expected_masks]

        self.assertEqual(output_masks[0].shape, expected_masks[0].shape)
        self.assertEqual(output_masks[1].shape, expected_masks[1].shape)
        self.assertEqual(output_masks[2].shape, expected_masks[2].shape)

        # With un-trained tiny random models, the output `logits` tensor is very likely to contain many values
        # close to each other, which cause `argmax` to give quite different results when running the test on 2
        # environments. We use a lower threshold `0.9` here to avoid flakiness.
        self.assertGreaterEqual(np.mean(output_masks[0] == expected_masks[0]), 0.9)
        self.assertGreaterEqual(np.mean(output_masks[1] == expected_masks[1]), 0.9)
        self.assertGreaterEqual(np.mean(output_masks[2] == expected_masks[2]), 0.9)

        for o in output:
            o["mask"] = mask_to_test_readable_only_shape(o["mask"])
        self.maxDiff = None
        self.assertEqual(
            nested_simplify(output, decimals=4),
            [
                {
                    "label": "LABEL_88",
                    "mask": {"shape": (480, 640)},
                    "score": None,
                },
                {
                    "label": "LABEL_101",
                    "mask": {"shape": (480, 640)},
                    "score": None,
                },
                {
                    "label": "LABEL_215",
                    "mask": {"shape": (480, 640)},
                    "score": None,
                },
            ],
        )

    @require_torch
    @slow
    def test_integration_torch_image_segmentation(self):
        model_id = "facebook/detr-resnet-50-panoptic"
        image_segmenter = pipeline(
            "image-segmentation",
            model=model_id,
            threshold=0.0,
            overlap_mask_area_threshold=0.0,
        )

        outputs = image_segmenter(
            "http://images.cocodataset.org/val2017/000000039769.jpg",
        )

        # Shortening by hashing
        for o in outputs:
            o["mask"] = mask_to_test_readable(o["mask"])

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {
                    "score": 0.9094,
                    "label": "blanket",
                    "mask": {"hash": "dcff19a97a", "shape": (480, 640), "white_pixels": 16617},
                },
                {
                    "score": 0.9941,
                    "label": "cat",
                    "mask": {"hash": "9c0af87bd0", "shape": (480, 640), "white_pixels": 59185},
                },
                {
                    "score": 0.9987,
                    "label": "remote",
                    "mask": {"hash": "c7870600d6", "shape": (480, 640), "white_pixels": 4182},
                },
                {
                    "score": 0.9995,
                    "label": "remote",
                    "mask": {"hash": "ef899a25fd", "shape": (480, 640), "white_pixels": 2275},
                },
                {
                    "score": 0.9722,
                    "label": "couch",
                    "mask": {"hash": "37b8446ac5", "shape": (480, 640), "white_pixels": 172380},
                },
                {
                    "score": 0.9994,
                    "label": "cat",
                    "mask": {"hash": "6a09d3655e", "shape": (480, 640), "white_pixels": 52561},
                },
            ],
        )

        outputs = image_segmenter(
            [
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                "http://images.cocodataset.org/val2017/000000039769.jpg",
            ],
        )

        # Shortening by hashing
        for output in outputs:
            for o in output:
                o["mask"] = mask_to_test_readable(o["mask"])

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [
                    {
                        "score": 0.9094,
                        "label": "blanket",
                        "mask": {"hash": "dcff19a97a", "shape": (480, 640), "white_pixels": 16617},
                    },
                    {
                        "score": 0.9941,
                        "label": "cat",
                        "mask": {"hash": "9c0af87bd0", "shape": (480, 640), "white_pixels": 59185},
                    },
                    {
                        "score": 0.9987,
                        "label": "remote",
                        "mask": {"hash": "c7870600d6", "shape": (480, 640), "white_pixels": 4182},
                    },
                    {
                        "score": 0.9995,
                        "label": "remote",
                        "mask": {"hash": "ef899a25fd", "shape": (480, 640), "white_pixels": 2275},
                    },
                    {
                        "score": 0.9722,
                        "label": "couch",
                        "mask": {"hash": "37b8446ac5", "shape": (480, 640), "white_pixels": 172380},
                    },
                    {
                        "score": 0.9994,
                        "label": "cat",
                        "mask": {"hash": "6a09d3655e", "shape": (480, 640), "white_pixels": 52561},
                    },
                ],
                [
                    {
                        "score": 0.9094,
                        "label": "blanket",
                        "mask": {"hash": "dcff19a97a", "shape": (480, 640), "white_pixels": 16617},
                    },
                    {
                        "score": 0.9941,
                        "label": "cat",
                        "mask": {"hash": "9c0af87bd0", "shape": (480, 640), "white_pixels": 59185},
                    },
                    {
                        "score": 0.9987,
                        "label": "remote",
                        "mask": {"hash": "c7870600d6", "shape": (480, 640), "white_pixels": 4182},
                    },
                    {
                        "score": 0.9995,
                        "label": "remote",
                        "mask": {"hash": "ef899a25fd", "shape": (480, 640), "white_pixels": 2275},
                    },
                    {
                        "score": 0.9722,
                        "label": "couch",
                        "mask": {"hash": "37b8446ac5", "shape": (480, 640), "white_pixels": 172380},
                    },
                    {
                        "score": 0.9994,
                        "label": "cat",
                        "mask": {"hash": "6a09d3655e", "shape": (480, 640), "white_pixels": 52561},
                    },
                ],
            ],
        )

    @require_torch
    @slow
    def test_threshold(self):
        model_id = "facebook/detr-resnet-50-panoptic"
        image_segmenter = pipeline("image-segmentation", model=model_id)

        outputs = image_segmenter("http://images.cocodataset.org/val2017/000000039769.jpg", threshold=0.999)
        # Shortening by hashing
        for o in outputs:
            o["mask"] = mask_to_test_readable(o["mask"])

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {
                    "score": 0.9995,
                    "label": "remote",
                    "mask": {"hash": "d02404f578", "shape": (480, 640), "white_pixels": 2789},
                },
                {
                    "score": 0.9994,
                    "label": "cat",
                    "mask": {"hash": "eaa115b40c", "shape": (480, 640), "white_pixels": 304411},
                },
            ],
        )

        outputs = image_segmenter("http://images.cocodataset.org/val2017/000000039769.jpg", threshold=0.5)

        for o in outputs:
            o["mask"] = mask_to_test_readable(o["mask"])

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {
                    "score": 0.9941,
                    "label": "cat",
                    "mask": {"hash": "9c0af87bd0", "shape": (480, 640), "white_pixels": 59185},
                },
                {
                    "score": 0.9987,
                    "label": "remote",
                    "mask": {"hash": "c7870600d6", "shape": (480, 640), "white_pixels": 4182},
                },
                {
                    "score": 0.9995,
                    "label": "remote",
                    "mask": {"hash": "ef899a25fd", "shape": (480, 640), "white_pixels": 2275},
                },
                {
                    "score": 0.9722,
                    "label": "couch",
                    "mask": {"hash": "37b8446ac5", "shape": (480, 640), "white_pixels": 172380},
                },
                {
                    "score": 0.9994,
                    "label": "cat",
                    "mask": {"hash": "6a09d3655e", "shape": (480, 640), "white_pixels": 52561},
                },
            ],
        )

    @require_torch
    @slow
    def test_oneformer(self):
        image_segmenter = pipeline(model="shi-labs/oneformer_ade20k_swin_tiny")

        image = load_dataset("hf-internal-testing/fixtures_ade20k", split="test")
        file = image[0]["file"]
        outputs = image_segmenter(file, threshold=0.99)
        # Shortening by hashing
        for o in outputs:
            o["mask"] = mask_to_test_readable(o["mask"])

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {
                    "score": 0.9981,
                    "label": "grass",
                    "mask": {"hash": "3a92904d4c", "white_pixels": 118131, "shape": (512, 683)},
                },
                {
                    "score": 0.9992,
                    "label": "sky",
                    "mask": {"hash": "fa2300cc9a", "white_pixels": 231565, "shape": (512, 683)},
                },
            ],
        )

        # Different task
        outputs = image_segmenter(file, threshold=0.99, subtask="instance")
        # Shortening by hashing
        for o in outputs:
            o["mask"] = mask_to_test_readable(o["mask"])

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {
                    "score": 0.9991,
                    "label": "sky",
                    "mask": {"hash": "8b1ffad016", "white_pixels": 230566, "shape": (512, 683)},
                },
                {
                    "score": 0.9981,
                    "label": "grass",
                    "mask": {"hash": "9bbdf83d3d", "white_pixels": 119130, "shape": (512, 683)},
                },
            ],
        )

        # Different task
        outputs = image_segmenter(file, subtask="semantic")
        # Shortening by hashing
        for o in outputs:
            o["mask"] = mask_to_test_readable(o["mask"])

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {
                    "score": None,
                    "label": "wall",
                    "mask": {"hash": "897fb20b7f", "white_pixels": 14506, "shape": (512, 683)},
                },
                {
                    "score": None,
                    "label": "building",
                    "mask": {"hash": "f2a68c63e4", "white_pixels": 125019, "shape": (512, 683)},
                },
                {
                    "score": None,
                    "label": "sky",
                    "mask": {"hash": "e0ca3a548e", "white_pixels": 135330, "shape": (512, 683)},
                },
                {
                    "score": None,
                    "label": "tree",
                    "mask": {"hash": "7c9544bcac", "white_pixels": 16263, "shape": (512, 683)},
                },
                {
                    "score": None,
                    "label": "road, route",
                    "mask": {"hash": "2c7704e491", "white_pixels": 2143, "shape": (512, 683)},
                },
                {
                    "score": None,
                    "label": "grass",
                    "mask": {"hash": "bf6c2867e0", "white_pixels": 53040, "shape": (512, 683)},
                },
                {
                    "score": None,
                    "label": "plant",
                    "mask": {"hash": "93c4b7199e", "white_pixels": 3335, "shape": (512, 683)},
                },
                {
                    "score": None,
                    "label": "house",
                    "mask": {"hash": "93ec419ad5", "white_pixels": 60, "shape": (512, 683)},
                },
            ],
        )
