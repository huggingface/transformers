# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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
from datasets import load_dataset

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_feature_extraction_common import FeatureExtractionSavingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

    if is_vision_available():
        from transformers import OneFormerFeatureExtractor
        from transformers.models.oneformer.image_processing_oneformer import binary_mask_to_rle
        from transformers.models.oneformer.modeling_oneformer import OneFormerForUniversalSegmentationOutput

if is_vision_available():
    from PIL import Image


ADE20K_150_CATEGORIES = [
    {"color": [120, 120, 120], "id": 0, "isthing": 0, "name": "wall"},
    {"color": [180, 120, 120], "id": 1, "isthing": 0, "name": "building"},
    {"color": [6, 230, 230], "id": 2, "isthing": 0, "name": "sky"},
    {"color": [80, 50, 50], "id": 3, "isthing": 0, "name": "floor"},
    {"color": [4, 200, 3], "id": 4, "isthing": 0, "name": "tree"},
    {"color": [120, 120, 80], "id": 5, "isthing": 0, "name": "ceiling"},
    {"color": [140, 140, 140], "id": 6, "isthing": 0, "name": "road, route"},
    {"color": [204, 5, 255], "id": 7, "isthing": 1, "name": "bed"},
    {"color": [230, 230, 230], "id": 8, "isthing": 1, "name": "window "},
    {"color": [4, 250, 7], "id": 9, "isthing": 0, "name": "grass"},
    {"color": [224, 5, 255], "id": 10, "isthing": 1, "name": "cabinet"},
    {"color": [235, 255, 7], "id": 11, "isthing": 0, "name": "sidewalk, pavement"},
    {"color": [150, 5, 61], "id": 12, "isthing": 1, "name": "person"},
    {"color": [120, 120, 70], "id": 13, "isthing": 0, "name": "earth, ground"},
    {"color": [8, 255, 51], "id": 14, "isthing": 1, "name": "door"},
    {"color": [255, 6, 82], "id": 15, "isthing": 1, "name": "table"},
    {"color": [143, 255, 140], "id": 16, "isthing": 0, "name": "mountain, mount"},
    {"color": [204, 255, 4], "id": 17, "isthing": 0, "name": "plant"},
    {"color": [255, 51, 7], "id": 18, "isthing": 1, "name": "curtain"},
    {"color": [204, 70, 3], "id": 19, "isthing": 1, "name": "chair"},
    {"color": [0, 102, 200], "id": 20, "isthing": 1, "name": "car"},
    {"color": [61, 230, 250], "id": 21, "isthing": 0, "name": "water"},
    {"color": [255, 6, 51], "id": 22, "isthing": 1, "name": "painting, picture"},
    {"color": [11, 102, 255], "id": 23, "isthing": 1, "name": "sofa"},
    {"color": [255, 7, 71], "id": 24, "isthing": 1, "name": "shelf"},
    {"color": [255, 9, 224], "id": 25, "isthing": 0, "name": "house"},
    {"color": [9, 7, 230], "id": 26, "isthing": 0, "name": "sea"},
    {"color": [220, 220, 220], "id": 27, "isthing": 1, "name": "mirror"},
    {"color": [255, 9, 92], "id": 28, "isthing": 0, "name": "rug"},
    {"color": [112, 9, 255], "id": 29, "isthing": 0, "name": "field"},
    {"color": [8, 255, 214], "id": 30, "isthing": 1, "name": "armchair"},
    {"color": [7, 255, 224], "id": 31, "isthing": 1, "name": "seat"},
    {"color": [255, 184, 6], "id": 32, "isthing": 1, "name": "fence"},
    {"color": [10, 255, 71], "id": 33, "isthing": 1, "name": "desk"},
    {"color": [255, 41, 10], "id": 34, "isthing": 0, "name": "rock, stone"},
    {"color": [7, 255, 255], "id": 35, "isthing": 1, "name": "wardrobe, closet, press"},
    {"color": [224, 255, 8], "id": 36, "isthing": 1, "name": "lamp"},
    {"color": [102, 8, 255], "id": 37, "isthing": 1, "name": "tub"},
    {"color": [255, 61, 6], "id": 38, "isthing": 1, "name": "rail"},
    {"color": [255, 194, 7], "id": 39, "isthing": 1, "name": "cushion"},
    {"color": [255, 122, 8], "id": 40, "isthing": 0, "name": "base, pedestal, stand"},
    {"color": [0, 255, 20], "id": 41, "isthing": 1, "name": "box"},
    {"color": [255, 8, 41], "id": 42, "isthing": 1, "name": "column, pillar"},
    {"color": [255, 5, 153], "id": 43, "isthing": 1, "name": "signboard, sign"},
    {
        "color": [6, 51, 255],
        "id": 44,
        "isthing": 1,
        "name": "chest of drawers, chest, bureau, dresser",
    },
    {"color": [235, 12, 255], "id": 45, "isthing": 1, "name": "counter"},
    {"color": [160, 150, 20], "id": 46, "isthing": 0, "name": "sand"},
    {"color": [0, 163, 255], "id": 47, "isthing": 1, "name": "sink"},
    {"color": [140, 140, 140], "id": 48, "isthing": 0, "name": "skyscraper"},
    {"color": [250, 10, 15], "id": 49, "isthing": 1, "name": "fireplace"},
    {"color": [20, 255, 0], "id": 50, "isthing": 1, "name": "refrigerator, icebox"},
    {"color": [31, 255, 0], "id": 51, "isthing": 0, "name": "grandstand, covered stand"},
    {"color": [255, 31, 0], "id": 52, "isthing": 0, "name": "path"},
    {"color": [255, 224, 0], "id": 53, "isthing": 1, "name": "stairs"},
    {"color": [153, 255, 0], "id": 54, "isthing": 0, "name": "runway"},
    {"color": [0, 0, 255], "id": 55, "isthing": 1, "name": "case, display case, showcase, vitrine"},
    {
        "color": [255, 71, 0],
        "id": 56,
        "isthing": 1,
        "name": "pool table, billiard table, snooker table",
    },
    {"color": [0, 235, 255], "id": 57, "isthing": 1, "name": "pillow"},
    {"color": [0, 173, 255], "id": 58, "isthing": 1, "name": "screen door, screen"},
    {"color": [31, 0, 255], "id": 59, "isthing": 0, "name": "stairway, staircase"},
    {"color": [11, 200, 200], "id": 60, "isthing": 0, "name": "river"},
    {"color": [255, 82, 0], "id": 61, "isthing": 0, "name": "bridge, span"},
    {"color": [0, 255, 245], "id": 62, "isthing": 1, "name": "bookcase"},
    {"color": [0, 61, 255], "id": 63, "isthing": 0, "name": "blind, screen"},
    {"color": [0, 255, 112], "id": 64, "isthing": 1, "name": "coffee table"},
    {
        "color": [0, 255, 133],
        "id": 65,
        "isthing": 1,
        "name": "toilet, can, commode, crapper, pot, potty, stool, throne",
    },
    {"color": [255, 0, 0], "id": 66, "isthing": 1, "name": "flower"},
    {"color": [255, 163, 0], "id": 67, "isthing": 1, "name": "book"},
    {"color": [255, 102, 0], "id": 68, "isthing": 0, "name": "hill"},
    {"color": [194, 255, 0], "id": 69, "isthing": 1, "name": "bench"},
    {"color": [0, 143, 255], "id": 70, "isthing": 1, "name": "countertop"},
    {"color": [51, 255, 0], "id": 71, "isthing": 1, "name": "stove"},
    {"color": [0, 82, 255], "id": 72, "isthing": 1, "name": "palm, palm tree"},
    {"color": [0, 255, 41], "id": 73, "isthing": 1, "name": "kitchen island"},
    {"color": [0, 255, 173], "id": 74, "isthing": 1, "name": "computer"},
    {"color": [10, 0, 255], "id": 75, "isthing": 1, "name": "swivel chair"},
    {"color": [173, 255, 0], "id": 76, "isthing": 1, "name": "boat"},
    {"color": [0, 255, 153], "id": 77, "isthing": 0, "name": "bar"},
    {"color": [255, 92, 0], "id": 78, "isthing": 1, "name": "arcade machine"},
    {"color": [255, 0, 255], "id": 79, "isthing": 0, "name": "hovel, hut, hutch, shack, shanty"},
    {"color": [255, 0, 245], "id": 80, "isthing": 1, "name": "bus"},
    {"color": [255, 0, 102], "id": 81, "isthing": 1, "name": "towel"},
    {"color": [255, 173, 0], "id": 82, "isthing": 1, "name": "light"},
    {"color": [255, 0, 20], "id": 83, "isthing": 1, "name": "truck"},
    {"color": [255, 184, 184], "id": 84, "isthing": 0, "name": "tower"},
    {"color": [0, 31, 255], "id": 85, "isthing": 1, "name": "chandelier"},
    {"color": [0, 255, 61], "id": 86, "isthing": 1, "name": "awning, sunshade, sunblind"},
    {"color": [0, 71, 255], "id": 87, "isthing": 1, "name": "street lamp"},
    {"color": [255, 0, 204], "id": 88, "isthing": 1, "name": "booth"},
    {"color": [0, 255, 194], "id": 89, "isthing": 1, "name": "tv"},
    {"color": [0, 255, 82], "id": 90, "isthing": 1, "name": "plane"},
    {"color": [0, 10, 255], "id": 91, "isthing": 0, "name": "dirt track"},
    {"color": [0, 112, 255], "id": 92, "isthing": 1, "name": "clothes"},
    {"color": [51, 0, 255], "id": 93, "isthing": 1, "name": "pole"},
    {"color": [0, 194, 255], "id": 94, "isthing": 0, "name": "land, ground, soil"},
    {
        "color": [0, 122, 255],
        "id": 95,
        "isthing": 1,
        "name": "bannister, banister, balustrade, balusters, handrail",
    },
    {
        "color": [0, 255, 163],
        "id": 96,
        "isthing": 0,
        "name": "escalator, moving staircase, moving stairway",
    },
    {
        "color": [255, 153, 0],
        "id": 97,
        "isthing": 1,
        "name": "ottoman, pouf, pouffe, puff, hassock",
    },
    {"color": [0, 255, 10], "id": 98, "isthing": 1, "name": "bottle"},
    {"color": [255, 112, 0], "id": 99, "isthing": 0, "name": "buffet, counter, sideboard"},
    {
        "color": [143, 255, 0],
        "id": 100,
        "isthing": 0,
        "name": "poster, posting, placard, notice, bill, card",
    },
    {"color": [82, 0, 255], "id": 101, "isthing": 0, "name": "stage"},
    {"color": [163, 255, 0], "id": 102, "isthing": 1, "name": "van"},
    {"color": [255, 235, 0], "id": 103, "isthing": 1, "name": "ship"},
    {"color": [8, 184, 170], "id": 104, "isthing": 1, "name": "fountain"},
    {
        "color": [133, 0, 255],
        "id": 105,
        "isthing": 0,
        "name": "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
    },
    {"color": [0, 255, 92], "id": 106, "isthing": 0, "name": "canopy"},
    {
        "color": [184, 0, 255],
        "id": 107,
        "isthing": 1,
        "name": "washer, automatic washer, washing machine",
    },
    {"color": [255, 0, 31], "id": 108, "isthing": 1, "name": "plaything, toy"},
    {"color": [0, 184, 255], "id": 109, "isthing": 0, "name": "pool"},
    {"color": [0, 214, 255], "id": 110, "isthing": 1, "name": "stool"},
    {"color": [255, 0, 112], "id": 111, "isthing": 1, "name": "barrel, cask"},
    {"color": [92, 255, 0], "id": 112, "isthing": 1, "name": "basket, handbasket"},
    {"color": [0, 224, 255], "id": 113, "isthing": 0, "name": "falls"},
    {"color": [112, 224, 255], "id": 114, "isthing": 0, "name": "tent"},
    {"color": [70, 184, 160], "id": 115, "isthing": 1, "name": "bag"},
    {"color": [163, 0, 255], "id": 116, "isthing": 1, "name": "minibike, motorbike"},
    {"color": [153, 0, 255], "id": 117, "isthing": 0, "name": "cradle"},
    {"color": [71, 255, 0], "id": 118, "isthing": 1, "name": "oven"},
    {"color": [255, 0, 163], "id": 119, "isthing": 1, "name": "ball"},
    {"color": [255, 204, 0], "id": 120, "isthing": 1, "name": "food, solid food"},
    {"color": [255, 0, 143], "id": 121, "isthing": 1, "name": "step, stair"},
    {"color": [0, 255, 235], "id": 122, "isthing": 0, "name": "tank, storage tank"},
    {"color": [133, 255, 0], "id": 123, "isthing": 1, "name": "trade name"},
    {"color": [255, 0, 235], "id": 124, "isthing": 1, "name": "microwave"},
    {"color": [245, 0, 255], "id": 125, "isthing": 1, "name": "pot"},
    {"color": [255, 0, 122], "id": 126, "isthing": 1, "name": "animal"},
    {"color": [255, 245, 0], "id": 127, "isthing": 1, "name": "bicycle"},
    {"color": [10, 190, 212], "id": 128, "isthing": 0, "name": "lake"},
    {"color": [214, 255, 0], "id": 129, "isthing": 1, "name": "dishwasher"},
    {"color": [0, 204, 255], "id": 130, "isthing": 1, "name": "screen"},
    {"color": [20, 0, 255], "id": 131, "isthing": 0, "name": "blanket, cover"},
    {"color": [255, 255, 0], "id": 132, "isthing": 1, "name": "sculpture"},
    {"color": [0, 153, 255], "id": 133, "isthing": 1, "name": "hood, exhaust hood"},
    {"color": [0, 41, 255], "id": 134, "isthing": 1, "name": "sconce"},
    {"color": [0, 255, 204], "id": 135, "isthing": 1, "name": "vase"},
    {"color": [41, 0, 255], "id": 136, "isthing": 1, "name": "traffic light"},
    {"color": [41, 255, 0], "id": 137, "isthing": 1, "name": "tray"},
    {"color": [173, 0, 255], "id": 138, "isthing": 1, "name": "trash can"},
    {"color": [0, 245, 255], "id": 139, "isthing": 1, "name": "fan"},
    {"color": [71, 0, 255], "id": 140, "isthing": 0, "name": "pier"},
    {"color": [122, 0, 255], "id": 141, "isthing": 0, "name": "crt screen"},
    {"color": [0, 255, 184], "id": 142, "isthing": 1, "name": "plate"},
    {"color": [0, 92, 255], "id": 143, "isthing": 1, "name": "monitor"},
    {"color": [184, 255, 0], "id": 144, "isthing": 1, "name": "bulletin board"},
    {"color": [0, 133, 255], "id": 145, "isthing": 0, "name": "shower"},
    {"color": [255, 214, 0], "id": 146, "isthing": 1, "name": "radiator"},
    {"color": [25, 194, 194], "id": 147, "isthing": 1, "name": "glass, drinking glass"},
    {"color": [102, 255, 0], "id": 148, "isthing": 1, "name": "clock"},
    {"color": [92, 0, 255], "id": 149, "isthing": 1, "name": "flag"},
]


def prepare_metadata(class_info):
    metadata = {}
    class_names = []
    thing_ids = []
    for idx in range(len(class_info)):
        info = class_info[idx]
        id = info["id"]
        metadata[str(id)] = info["name"]
        class_names.append(info["name"])
        if info["isthing"]:
            thing_ids.append(id)
    metadata["thing_ids"] = thing_ids
    metadata["class_names"] = class_names
    return metadata


class OneFormerFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=30,
        max_resolution=400,
        size=None,
        do_resize=True,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        num_labels=10,
        reduce_labels=True,
        ignore_index=255,
        max_seq_length=77,
        task_seq_length=77,
        repo_path="shi-labs/oneformer_ade20k_swin_tiny",
        class_info=ADE20K_150_CATEGORIES,
        num_text=10,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = {"shortest_edge": 32, "longest_edge": 1333} if size is None else size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.size_divisor = 0
        self.max_seq_length = max_seq_length
        self.task_seq_length = task_seq_length
        self.class_info = class_info
        self.metadata = prepare_metadata(class_info)
        self.num_text = num_text
        self.repo_path = repo_path

        # for the post_process_functions
        self.batch_size = 2
        self.num_queries = 10
        self.num_classes = 10
        self.height = 3
        self.width = 4
        self.num_labels = num_labels
        self.reduce_labels = reduce_labels
        self.ignore_index = ignore_index

    def prepare_feat_extract_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "size_divisor": self.size_divisor,
            "num_labels": self.num_labels,
            "reduce_labels": self.reduce_labels,
            "ignore_index": self.ignore_index,
            "max_seq_length": self.max_seq_length,
            "task_seq_length": self.task_seq_length,
            "class_info": self.class_info,
            "metadata": self.metadata,
            "num_text": self.num_text,
            "repo_path": self.repo_path,
        }

    def get_expected_values(self, image_inputs, batched=False):
        """
        This function computes the expected height and width when providing images to OneFormerFeatureExtractor,
        assuming do_resize is set to True with a scalar size.
        """
        if not batched:
            image = image_inputs[0]
            if isinstance(image, Image.Image):
                w, h = image.size
            else:
                h, w = image.shape[1], image.shape[2]
            if w < h:
                expected_height = int(self.size["shortest_edge"] * h / w)
                expected_width = self.size["shortest_edge"]
            elif w > h:
                expected_height = self.size["shortest_edge"]
                expected_width = int(self.size["shortest_edge"] * w / h)
            else:
                expected_height = self.size["shortest_edge"]
                expected_width = self.size["shortest_edge"]

        else:
            expected_values = []
            for image in image_inputs:
                expected_height, expected_width, expected_sequence_length = self.get_expected_values([image])
                expected_values.append((expected_height, expected_width, expected_sequence_length))
            expected_height = max(expected_values, key=lambda item: item[0])[0]
            expected_width = max(expected_values, key=lambda item: item[1])[1]

        expected_sequence_length = self.max_seq_length

        return expected_height, expected_width, expected_sequence_length

    def get_fake_oneformer_outputs(self):
        return OneFormerForUniversalSegmentationOutput(
            # +1 for null class
            class_queries_logits=torch.randn((self.batch_size, self.num_queries, self.num_classes + 1)),
            masks_queries_logits=torch.randn((self.batch_size, self.num_queries, self.height, self.width)),
        )


@require_torch
@require_vision
class OneFormerFeatureExtractionTest(FeatureExtractionSavingTestMixin, unittest.TestCase):
    feature_extraction_class = OneFormerFeatureExtractor if (is_vision_available() and is_torch_available()) else None

    def setUp(self):
        self.feature_extract_tester = OneFormerFeatureExtractionTester(self)

    @property
    def feat_extract_dict(self):
        return self.feature_extract_tester.prepare_feat_extract_dict()

    def test_feat_extract_properties(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        self.assertTrue(hasattr(feature_extractor, "image_mean"))
        self.assertTrue(hasattr(feature_extractor, "image_std"))
        self.assertTrue(hasattr(feature_extractor, "do_normalize"))
        self.assertTrue(hasattr(feature_extractor, "do_resize"))
        self.assertTrue(hasattr(feature_extractor, "size"))
        self.assertTrue(hasattr(feature_extractor, "max_size"))
        self.assertTrue(hasattr(feature_extractor, "ignore_index"))
        self.assertTrue(hasattr(feature_extractor, "num_labels"))
        self.assertTrue(hasattr(feature_extractor, "max_seq_length"))
        self.assertTrue(hasattr(feature_extractor, "task_seq_length"))
        self.assertTrue(hasattr(feature_extractor, "class_info"))
        self.assertTrue(hasattr(feature_extractor, "num_text"))
        self.assertTrue(hasattr(feature_extractor, "repo_path"))

    def test_batch_feature(self):
        pass

    def test_call_pil(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random PIL images
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input
        encoded_images = feature_extractor(image_inputs[0], ["semantic"], return_tensors="pt").pixel_values

        expected_height, expected_width, expected_sequence_length = self.feature_extract_tester.get_expected_values(
            image_inputs
        )

        self.assertEqual(
            encoded_images.shape,
            (1, self.feature_extract_tester.num_channels, expected_height, expected_width),
        )

        tokenized_task_inputs = feature_extractor(image_inputs[0], ["semantic"], return_tensors="pt").task_inputs

        self.assertEqual(
            tokenized_task_inputs.shape,
            (1, expected_sequence_length),
        )

        # Test batched
        expected_height, expected_width, expected_sequence_length = self.feature_extract_tester.get_expected_values(
            image_inputs, batched=True
        )

        encoded_images = feature_extractor(
            image_inputs, ["semantic"] * len(image_inputs), return_tensors="pt"
        ).pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

        tokenized_task_inputs = feature_extractor(
            image_inputs, ["semantic"] * len(image_inputs), return_tensors="pt"
        ).task_inputs

        self.assertEqual(
            tokenized_task_inputs.shape,
            (self.feature_extract_tester.batch_size, expected_sequence_length),
        )

    def test_call_numpy(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random numpy tensors
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input
        encoded_images = feature_extractor(image_inputs[0], ["semantic"], return_tensors="pt").pixel_values

        expected_height, expected_width, expected_sequence_length = self.feature_extract_tester.get_expected_values(
            image_inputs
        )

        self.assertEqual(
            encoded_images.shape,
            (1, self.feature_extract_tester.num_channels, expected_height, expected_width),
        )

        tokenized_task_inputs = feature_extractor(image_inputs[0], ["semantic"], return_tensors="pt").task_inputs

        self.assertEqual(
            tokenized_task_inputs.shape,
            (1, expected_sequence_length),
        )

        # Test batched
        expected_height, expected_width, expected_sequence_length = self.feature_extract_tester.get_expected_values(
            image_inputs, batched=True
        )

        encoded_images = feature_extractor(
            image_inputs, ["semantic"] * len(image_inputs), return_tensors="pt"
        ).pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

        tokenized_task_inputs = feature_extractor(
            image_inputs, ["semantic"] * len(image_inputs), return_tensors="pt"
        ).task_inputs

        self.assertEqual(
            tokenized_task_inputs.shape,
            (self.feature_extract_tester.batch_size, expected_sequence_length),
        )

    def test_call_pytorch(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random PyTorch tensors
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test not batched input
        encoded_images = feature_extractor(image_inputs[0], ["semantic"], return_tensors="pt").pixel_values

        expected_height, expected_width, expected_sequence_length = self.feature_extract_tester.get_expected_values(
            image_inputs
        )

        self.assertEqual(
            encoded_images.shape,
            (1, self.feature_extract_tester.num_channels, expected_height, expected_width),
        )

        tokenized_task_inputs = feature_extractor(image_inputs[0], ["semantic"], return_tensors="pt").task_inputs

        self.assertEqual(
            tokenized_task_inputs.shape,
            (1, expected_sequence_length),
        )

        # Test batched
        expected_height, expected_width, expected_sequence_length = self.feature_extract_tester.get_expected_values(
            image_inputs, batched=True
        )

        encoded_images = feature_extractor(
            image_inputs, ["semantic"] * len(image_inputs), return_tensors="pt"
        ).pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

        tokenized_task_inputs = feature_extractor(
            image_inputs, ["semantic"] * len(image_inputs), return_tensors="pt"
        ).task_inputs

        self.assertEqual(
            tokenized_task_inputs.shape,
            (self.feature_extract_tester.batch_size, expected_sequence_length),
        )

    def test_equivalence_pad_and_create_pixel_mask(self):
        # Initialize feature_extractors
        feature_extractor_1 = self.feature_extraction_class(**self.feat_extract_dict)
        feature_extractor_2 = self.feature_extraction_class(
            do_resize=False,
            do_normalize=False,
            do_rescale=False,
            num_labels=self.feature_extract_tester.num_classes,
            max_seq_length=77,
            task_seq_length=77,
            class_info=ADE20K_150_CATEGORIES,
            num_text=self.feature_extract_tester.num_text,
            repo_path="shi-labs/oneformer_ade20k_swin_tiny",
        )
        # create random PyTorch tensors
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test whether the method "pad_and_return_pixel_mask" and calling the feature extractor return the same tensors
        encoded_images_with_method = feature_extractor_1.encode_inputs(
            image_inputs, ["semantic"] * len(image_inputs), return_tensors="pt"
        )
        encoded_images = feature_extractor_2(image_inputs, ["semantic"] * len(image_inputs), return_tensors="pt")

        self.assertTrue(
            torch.allclose(encoded_images_with_method["pixel_values"], encoded_images["pixel_values"], atol=1e-4)
        )
        self.assertTrue(
            torch.allclose(encoded_images_with_method["pixel_mask"], encoded_images["pixel_mask"], atol=1e-4)
        )

    def comm_get_feature_extractor_inputs(
        self, with_segmentation_maps=False, is_instance_map=False, segmentation_type="np"
    ):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # prepare image and target
        num_labels = self.feature_extract_tester.num_labels
        annotations = None
        instance_id_to_semantic_id = None
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False)
        if with_segmentation_maps:
            high = num_labels
            if is_instance_map:
                labels_expanded = list(range(num_labels)) * 2
                instance_id_to_semantic_id = {
                    instance_id: label_id for instance_id, label_id in enumerate(labels_expanded)
                }
            annotations = [
                np.random.randint(0, high * 2, (img.size[1], img.size[0])).astype(np.uint8) for img in image_inputs
            ]
            if segmentation_type == "pil":
                annotations = [Image.fromarray(annotation) for annotation in annotations]

        inputs = feature_extractor(
            image_inputs,
            ["semantic"] * len(image_inputs),
            annotations,
            return_tensors="pt",
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            pad_and_return_pixel_mask=True,
        )

        return inputs

    def test_init_without_params(self):
        pass

    def test_with_size_divisor(self):
        size_divisors = [8, 16, 32]
        weird_input_sizes = [(407, 802), (582, 1094)]
        for size_divisor in size_divisors:
            feat_extract_dict = {**self.feat_extract_dict, **{"size_divisor": size_divisor}}
            feature_extractor = self.feature_extraction_class(**feat_extract_dict)
            for weird_input_size in weird_input_sizes:
                inputs = feature_extractor([np.ones((3, *weird_input_size))], ["semantic"], return_tensors="pt")
                pixel_values = inputs["pixel_values"]
                # check if divisible
                self.assertTrue((pixel_values.shape[-1] % size_divisor) == 0)
                self.assertTrue((pixel_values.shape[-2] % size_divisor) == 0)

    def test_call_with_segmentation_maps(self):
        def common(is_instance_map=False, segmentation_type=None):
            inputs = self.comm_get_feature_extractor_inputs(
                with_segmentation_maps=True, is_instance_map=is_instance_map, segmentation_type=segmentation_type
            )

            mask_labels = inputs["mask_labels"]
            class_labels = inputs["class_labels"]
            pixel_values = inputs["pixel_values"]
            text_inputs = inputs["text_inputs"]

            # check the batch_size
            for mask_label, class_label, text_input in zip(mask_labels, class_labels, text_inputs):
                self.assertEqual(mask_label.shape[0], class_label.shape[0])
                # this ensure padding has happened
                self.assertEqual(mask_label.shape[1:], pixel_values.shape[2:])
                self.assertEqual(text_inputs.shape[1], self.feature_extract_tester.num_text)

        common()
        common(is_instance_map=True)
        common(is_instance_map=False, segmentation_type="pil")
        common(is_instance_map=True, segmentation_type="pil")

    def test_integration_semantic_segmentation(self):
        # load 2 images and corresponding panoptic annotations from the hub
        dataset = load_dataset("nielsr/ade20k-panoptic-demo")
        image1 = dataset["train"][0]["image"]
        image2 = dataset["train"][1]["image"]
        segments_info1 = dataset["train"][0]["segments_info"]
        segments_info2 = dataset["train"][1]["segments_info"]
        annotation1 = dataset["train"][0]["label"]
        annotation2 = dataset["train"][1]["label"]

        def rgb_to_id(color):
            if isinstance(color, np.ndarray) and len(color.shape) == 3:
                if color.dtype == np.uint8:
                    color = color.astype(np.int32)
                return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
            return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

        def create_panoptic_map(annotation, segments_info):
            annotation = np.array(annotation)
            # convert RGB to segment IDs per pixel
            # 0 is the "ignore" label, for which we don't need to make binary masks
            panoptic_map = rgb_to_id(annotation)

            # create mapping between segment IDs and semantic classes
            inst2class = {segment["id"]: segment["category_id"] for segment in segments_info}

            return panoptic_map, inst2class

        panoptic_map1, inst2class1 = create_panoptic_map(annotation1, segments_info1)
        panoptic_map2, inst2class2 = create_panoptic_map(annotation2, segments_info2)

        # create a feature extractor
        feature_extractor = OneFormerFeatureExtractor(
            reduce_labels=True,
            ignore_index=0,
            size=(512, 512),
            max_seq_length=77,
            task_seq_length=77,
            class_info=ADE20K_150_CATEGORIES,
            num_text=self.feature_extract_tester.num_text,
            repo_path="shi-labs/oneformer_ade20k_swin_tiny",
        )

        # prepare the images and annotations
        pixel_values_list = [np.moveaxis(np.array(image1), -1, 0), np.moveaxis(np.array(image2), -1, 0)]
        inputs = feature_extractor.encode_inputs(
            pixel_values_list,
            ["semantic", "semantic"],
            [panoptic_map1, panoptic_map2],
            instance_id_to_semantic_id=[inst2class1, inst2class2],
            return_tensors="pt",
        )

        # verify the pixel values, task inputs, text inputs and pixel mask
        self.assertEqual(inputs["pixel_values"].shape, (2, 3, 512, 711))
        self.assertEqual(inputs["pixel_mask"].shape, (2, 512, 711))
        self.assertEqual(inputs["task_inputs"].shape, (2, 77))
        self.assertEqual(inputs["text_inputs"].shape, (2, self.feature_extract_tester.num_text, 77))

        # verify the class labels
        self.assertEqual(len(inputs["class_labels"]), 2)
        # fmt: off
        expected_class_labels = torch.tensor([4, 17, 32, 42, 12, 3, 5, 0, 43, 96, 104, 31, 125, 138, 87, 149])  # noqa: E231
        # fmt: on
        self.assertTrue(torch.allclose(inputs["class_labels"][0], expected_class_labels))
        # fmt: off
        expected_class_labels = torch.tensor([19, 67, 82, 17, 12, 42, 3, 14, 5, 0, 115, 43, 8, 138, 125, 143])  # noqa: E231
        # fmt: on
        self.assertTrue(torch.allclose(inputs["class_labels"][1], expected_class_labels))

        # verify the task inputs
        self.assertEqual(len(inputs["task_inputs"]), 2)
        self.assertEqual(inputs["task_inputs"][0].sum().item(), 141082)
        self.assertEqual(inputs["task_inputs"][0].sum().item(), inputs["task_inputs"][1].sum().item())

        # verify the text inputs
        self.assertEqual(len(inputs["text_inputs"]), 2)
        self.assertEqual(inputs["text_inputs"][0].sum().item(), 1095752)
        self.assertEqual(inputs["text_inputs"][1].sum().item(), 1062468)

        # verify the mask labels
        self.assertEqual(len(inputs["mask_labels"]), 2)
        self.assertEqual(inputs["mask_labels"][0].shape, (16, 512, 711))
        self.assertEqual(inputs["mask_labels"][1].shape, (16, 512, 711))
        self.assertEqual(inputs["mask_labels"][0].sum().item(), 315193.0)
        self.assertEqual(inputs["mask_labels"][1].sum().item(), 350747.0)

    def test_integration_instance_segmentation(self):
        # load 2 images and corresponding panoptic annotations from the hub
        dataset = load_dataset("nielsr/ade20k-panoptic-demo")
        image1 = dataset["train"][0]["image"]
        image2 = dataset["train"][1]["image"]
        segments_info1 = dataset["train"][0]["segments_info"]
        segments_info2 = dataset["train"][1]["segments_info"]
        annotation1 = dataset["train"][0]["label"]
        annotation2 = dataset["train"][1]["label"]

        def rgb_to_id(color):
            if isinstance(color, np.ndarray) and len(color.shape) == 3:
                if color.dtype == np.uint8:
                    color = color.astype(np.int32)
                return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
            return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

        def create_panoptic_map(annotation, segments_info):
            annotation = np.array(annotation)
            # convert RGB to segment IDs per pixel
            # 0 is the "ignore" label, for which we don't need to make binary masks
            panoptic_map = rgb_to_id(annotation)

            # create mapping between segment IDs and semantic classes
            inst2class = {segment["id"]: segment["category_id"] for segment in segments_info}

            return panoptic_map, inst2class

        panoptic_map1, inst2class1 = create_panoptic_map(annotation1, segments_info1)
        panoptic_map2, inst2class2 = create_panoptic_map(annotation2, segments_info2)

        # create a feature extractor
        feature_extractor = OneFormerFeatureExtractor(
            reduce_labels=True,
            ignore_index=0,
            size=(512, 512),
            max_seq_length=77,
            task_seq_length=77,
            class_info=ADE20K_150_CATEGORIES,
            num_text=self.feature_extract_tester.num_text,
            repo_path="shi-labs/oneformer_ade20k_swin_tiny",
        )

        # prepare the images and annotations
        pixel_values_list = [np.moveaxis(np.array(image1), -1, 0), np.moveaxis(np.array(image2), -1, 0)]
        inputs = feature_extractor.encode_inputs(
            pixel_values_list,
            ["instance", "instance"],
            [panoptic_map1, panoptic_map2],
            instance_id_to_semantic_id=[inst2class1, inst2class2],
            return_tensors="pt",
        )

        # verify the pixel values, task inputs, text inputs and pixel mask
        self.assertEqual(inputs["pixel_values"].shape, (2, 3, 512, 711))
        self.assertEqual(inputs["pixel_mask"].shape, (2, 512, 711))
        self.assertEqual(inputs["task_inputs"].shape, (2, 77))
        self.assertEqual(inputs["text_inputs"].shape, (2, self.feature_extract_tester.num_text, 77))

        # verify the class labels
        self.assertEqual(len(inputs["class_labels"]), 2)
        # fmt: off
        expected_class_labels = torch.tensor([32, 42, 42, 42, 42, 42, 42, 42, 32, 12, 12, 12, 12, 12, 42, 42, 12, 12, 12, 42, 12, 12, 12, 12, 12, 12, 12, 12, 12, 42, 42, 42, 12, 42, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 43, 43, 43, 43, 104, 43, 31, 125, 31, 125, 138, 87, 125, 149, 138, 125, 87, 87])  # noqa: E231
        # fmt: on
        self.assertTrue(torch.allclose(inputs["class_labels"][0], expected_class_labels))
        # fmt: off
        expected_class_labels = torch.tensor([19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 67, 82, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 12, 12, 42, 12, 12, 12, 12, 14, 12, 12, 12, 12, 12, 12, 12, 12, 14, 12, 12, 115, 43, 43, 115, 43, 43, 43, 8, 8, 8, 138, 138, 125, 143])  # noqa: E231
        # fmt: on
        self.assertTrue(torch.allclose(inputs["class_labels"][1], expected_class_labels))

        # verify the task inputs
        self.assertEqual(len(inputs["task_inputs"]), 2)
        self.assertEqual(inputs["task_inputs"][0].sum().item(), 144985)
        self.assertEqual(inputs["task_inputs"][0].sum().item(), inputs["task_inputs"][1].sum().item())

        # verify the text inputs
        self.assertEqual(len(inputs["text_inputs"]), 2)
        self.assertEqual(inputs["text_inputs"][0].sum().item(), 1037040)
        self.assertEqual(inputs["text_inputs"][1].sum().item(), 1044078)

        # verify the mask labels
        self.assertEqual(len(inputs["mask_labels"]), 2)
        self.assertEqual(inputs["mask_labels"][0].shape, (73, 512, 711))
        self.assertEqual(inputs["mask_labels"][1].shape, (57, 512, 711))
        self.assertEqual(inputs["mask_labels"][0].sum().item(), 35040.0)
        self.assertEqual(inputs["mask_labels"][1].sum().item(), 98228.0)

    def test_integration_panoptic_segmentation(self):
        # load 2 images and corresponding panoptic annotations from the hub
        dataset = load_dataset("nielsr/ade20k-panoptic-demo")
        image1 = dataset["train"][0]["image"]
        image2 = dataset["train"][1]["image"]
        segments_info1 = dataset["train"][0]["segments_info"]
        segments_info2 = dataset["train"][1]["segments_info"]
        annotation1 = dataset["train"][0]["label"]
        annotation2 = dataset["train"][1]["label"]

        def rgb_to_id(color):
            if isinstance(color, np.ndarray) and len(color.shape) == 3:
                if color.dtype == np.uint8:
                    color = color.astype(np.int32)
                return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
            return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

        def create_panoptic_map(annotation, segments_info):
            annotation = np.array(annotation)
            # convert RGB to segment IDs per pixel
            # 0 is the "ignore" label, for which we don't need to make binary masks
            panoptic_map = rgb_to_id(annotation)

            # create mapping between segment IDs and semantic classes
            inst2class = {segment["id"]: segment["category_id"] for segment in segments_info}

            return panoptic_map, inst2class

        panoptic_map1, inst2class1 = create_panoptic_map(annotation1, segments_info1)
        panoptic_map2, inst2class2 = create_panoptic_map(annotation2, segments_info2)

        # create a feature extractor
        feature_extractor = OneFormerFeatureExtractor(
            reduce_labels=True,
            ignore_index=0,
            size=(512, 512),
            max_seq_length=77,
            task_seq_length=77,
            class_info=ADE20K_150_CATEGORIES,
            num_text=self.feature_extract_tester.num_text,
            repo_path="shi-labs/oneformer_ade20k_swin_tiny",
        )

        # prepare the images and annotations
        pixel_values_list = [np.moveaxis(np.array(image1), -1, 0), np.moveaxis(np.array(image2), -1, 0)]
        inputs = feature_extractor.encode_inputs(
            pixel_values_list,
            ["panoptic", "panoptic"],
            [panoptic_map1, panoptic_map2],
            instance_id_to_semantic_id=[inst2class1, inst2class2],
            return_tensors="pt",
        )

        # verify the pixel values, task inputs, text inputs and pixel mask
        self.assertEqual(inputs["pixel_values"].shape, (2, 3, 512, 711))
        self.assertEqual(inputs["pixel_mask"].shape, (2, 512, 711))
        self.assertEqual(inputs["task_inputs"].shape, (2, 77))
        self.assertEqual(inputs["text_inputs"].shape, (2, self.feature_extract_tester.num_text, 77))

        # verify the class labels
        self.assertEqual(len(inputs["class_labels"]), 2)
        # fmt: off
        expected_class_labels = torch.tensor([4, 17, 32, 42, 42, 42, 42, 42, 42, 42, 32, 12, 12, 12, 12, 12, 42, 42, 12, 12, 12, 42, 12, 12, 12, 12, 12, 3, 12, 12, 12, 12, 42, 42, 42, 12, 42, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 5, 12, 12, 12, 12, 12, 12, 12, 0, 43, 43, 43, 96, 43, 104, 43, 31, 125, 31, 125, 138, 87, 125, 149, 138, 125, 87, 87])  # noqa: E231
        # fmt: on
        self.assertTrue(torch.allclose(inputs["class_labels"][0], expected_class_labels))
        # fmt: off
        expected_class_labels = torch.tensor([19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 67, 82, 19, 19, 17, 19, 19, 19, 19, 19, 19, 19, 19, 19, 12, 12, 42, 12, 12, 12, 12, 3, 14, 12, 12, 12, 12, 12, 12, 12, 12, 14, 5, 12, 12, 0, 115, 43, 43, 115, 43, 43, 43, 8, 8, 8, 138, 138, 125, 143])  # noqa: E231
        # fmt: on
        self.assertTrue(torch.allclose(inputs["class_labels"][1], expected_class_labels))

        # verify the task inputs
        self.assertEqual(len(inputs["task_inputs"]), 2)
        self.assertEqual(inputs["task_inputs"][0].sum().item(), 136240)
        self.assertEqual(inputs["task_inputs"][0].sum().item(), inputs["task_inputs"][1].sum().item())

        # verify the text inputs
        self.assertEqual(len(inputs["text_inputs"]), 2)
        self.assertEqual(inputs["text_inputs"][0].sum().item(), 1048653)
        self.assertEqual(inputs["text_inputs"][1].sum().item(), 1067160)

        # verify the mask labels
        self.assertEqual(len(inputs["mask_labels"]), 2)
        self.assertEqual(inputs["mask_labels"][0].shape, (79, 512, 711))
        self.assertEqual(inputs["mask_labels"][1].shape, (61, 512, 711))
        self.assertEqual(inputs["mask_labels"][0].sum().item(), 315193.0)
        self.assertEqual(inputs["mask_labels"][1].sum().item(), 350747.0)

    def test_binary_mask_to_rle(self):
        fake_binary_mask = np.zeros((20, 50))
        fake_binary_mask[0, 20:] = 1
        fake_binary_mask[1, :15] = 1
        fake_binary_mask[5, :10] = 1

        rle = binary_mask_to_rle(fake_binary_mask)
        self.assertEqual(len(rle), 4)
        self.assertEqual(rle[0], 21)
        self.assertEqual(rle[1], 45)

    def test_post_process_sem_seg_output(self):
        fature_extractor = self.feature_extraction_class(
            num_labels=self.feature_extract_tester.num_classes,
            max_seq_length=77,
            task_seq_length=77,
            class_info=ADE20K_150_CATEGORIES,
            num_text=self.feature_extract_tester.num_text,
            repo_path="shi-labs/oneformer_ade20k_swin_tiny",
        )
        outputs = self.feature_extract_tester.get_fake_oneformer_outputs()
        segmentation = fature_extractor.post_process_sem_seg_output(outputs)

        self.assertEqual(
            segmentation.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_classes,
                self.feature_extract_tester.height,
                self.feature_extract_tester.width,
            ),
        )

        target_size = (1, 4)
        segmentation = fature_extractor.post_process_sem_seg_output(outputs, target_size=target_size)

        self.assertEqual(
            segmentation.shape,
            (self.feature_extract_tester.batch_size, self.feature_extract_tester.num_classes, *target_size),
        )

    def test_post_process_semantic_segmentation(self):
        fature_extractor = self.feature_extraction_class(
            num_labels=self.feature_extract_tester.num_classes,
            max_seq_length=77,
            task_seq_length=77,
            class_info=ADE20K_150_CATEGORIES,
            num_text=self.feature_extract_tester.num_text,
            repo_path="shi-labs/oneformer_ade20k_swin_tiny",
        )
        outputs = self.feature_extract_tester.get_fake_oneformer_outputs()

        segmentation = fature_extractor.post_process_semantic_segmentation(outputs)

        self.assertEqual(len(segmentation), self.feature_extract_tester.batch_size)
        self.assertEqual(
            segmentation[0].shape,
            (
                self.feature_extract_tester.height,
                self.feature_extract_tester.width,
            ),
        )

        target_sizes = [(1, 4) for i in range(self.feature_extract_tester.batch_size)]
        segmentation = fature_extractor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)

        self.assertEqual(segmentation[0].shape, target_sizes[0])

    def test_post_process_instance_segmentation(self):
        feature_extractor = self.feature_extraction_class(
            num_labels=self.feature_extract_tester.num_classes,
            max_seq_length=77,
            task_seq_length=77,
            class_info=ADE20K_150_CATEGORIES,
            num_text=self.feature_extract_tester.num_text,
            repo_path="shi-labs/oneformer_ade20k_swin_tiny",
        )
        outputs = self.feature_extract_tester.get_fake_oneformer_outputs()
        segmentation = feature_extractor.post_process_instance_segmentation(outputs, threshold=0)

        self.assertTrue(len(segmentation) == self.feature_extract_tester.batch_size)
        for el in segmentation:
            self.assertTrue("segmentation" in el)
            self.assertTrue("segments_info" in el)
            self.assertEqual(type(el["segments_info"]), list)
            self.assertEqual(
                el["segmentation"].shape, (self.feature_extract_tester.height, self.feature_extract_tester.width)
            )

    def test_post_process_panoptic_segmentation(self):
        feature_extractor = self.feature_extraction_class(
            num_labels=self.feature_extract_tester.num_classes,
            max_seq_length=77,
            task_seq_length=77,
            class_info=ADE20K_150_CATEGORIES,
            num_text=self.feature_extract_tester.num_text,
            repo_path="shi-labs/oneformer_ade20k_swin_tiny",
        )
        outputs = self.feature_extract_tester.get_fake_oneformer_outputs()
        segmentation = feature_extractor.post_process_panoptic_segmentation(outputs, threshold=0)

        self.assertTrue(len(segmentation) == self.feature_extract_tester.batch_size)
        for el in segmentation:
            self.assertTrue("segmentation" in el)
            self.assertTrue("segments_info" in el)
            self.assertEqual(type(el["segments_info"]), list)
            self.assertEqual(
                el["segmentation"].shape, (self.feature_extract_tester.height, self.feature_extract_tester.width)
            )
