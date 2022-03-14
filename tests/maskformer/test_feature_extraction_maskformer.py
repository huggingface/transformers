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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ..test_feature_extraction_common import FeatureExtractionSavingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

    if is_vision_available():
        from transformers import MaskFormerFeatureExtractor
        from transformers.models.maskformer.modeling_maskformer import MaskFormerForInstanceSegmentationOutput

if is_vision_available():
    from PIL import Image


class MaskFormerFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=32,
        max_size=1333,  # by setting max_size > max_resolution we're effectively not testing this :p
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        num_labels=10,
        reduce_labels=True,
        ignore_index=255,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.max_size = max_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.size_divisibility = 0
        # for the post_process_functions
        self.batch_size = 2
        self.num_queries = 3
        self.num_classes = 2
        self.height = 3
        self.width = 4
        self.num_labels = num_labels
        self.reduce_labels = reduce_labels
        self.ignore_index = ignore_index

    def prepare_feat_extract_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "max_size": self.max_size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "size_divisibility": self.size_divisibility,
            "num_labels": self.num_labels,
            "reduce_labels": self.reduce_labels,
            "ignore_index": self.ignore_index,
        }

    def get_expected_values(self, image_inputs, batched=False):
        """
        This function computes the expected height and width when providing images to MaskFormerFeatureExtractor,
        assuming do_resize is set to True with a scalar size.
        """
        if not batched:
            image = image_inputs[0]
            if isinstance(image, Image.Image):
                w, h = image.size
            else:
                h, w = image.shape[1], image.shape[2]
            if w < h:
                expected_height = int(self.size * h / w)
                expected_width = self.size
            elif w > h:
                expected_height = self.size
                expected_width = int(self.size * w / h)
            else:
                expected_height = self.size
                expected_width = self.size

        else:
            expected_values = []
            for image in image_inputs:
                expected_height, expected_width = self.get_expected_values([image])
                expected_values.append((expected_height, expected_width))
            expected_height = max(expected_values, key=lambda item: item[0])[0]
            expected_width = max(expected_values, key=lambda item: item[1])[1]

        return expected_height, expected_width

    def get_fake_maskformer_outputs(self):
        return MaskFormerForInstanceSegmentationOutput(
            # +1 for null class
            class_queries_logits=torch.randn((self.batch_size, self.num_queries, self.num_classes + 1)),
            masks_queries_logits=torch.randn((self.batch_size, self.num_queries, self.height, self.width)),
        )


@require_torch
@require_vision
class MaskFormerFeatureExtractionTest(FeatureExtractionSavingTestMixin, unittest.TestCase):

    feature_extraction_class = MaskFormerFeatureExtractor if (is_vision_available() and is_torch_available()) else None

    def setUp(self):
        self.feature_extract_tester = MaskFormerFeatureExtractionTester(self)

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
        encoded_images = feature_extractor(image_inputs[0], return_tensors="pt").pixel_values

        expected_height, expected_width = self.feature_extract_tester.get_expected_values(image_inputs)

        self.assertEqual(
            encoded_images.shape,
            (1, self.feature_extract_tester.num_channels, expected_height, expected_width),
        )

        # Test batched
        expected_height, expected_width = self.feature_extract_tester.get_expected_values(image_inputs, batched=True)

        encoded_images = feature_extractor(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

    def test_call_numpy(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random numpy tensors
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input
        encoded_images = feature_extractor(image_inputs[0], return_tensors="pt").pixel_values

        expected_height, expected_width = self.feature_extract_tester.get_expected_values(image_inputs)

        self.assertEqual(
            encoded_images.shape,
            (1, self.feature_extract_tester.num_channels, expected_height, expected_width),
        )

        # Test batched
        encoded_images = feature_extractor(image_inputs, return_tensors="pt").pixel_values

        expected_height, expected_width = self.feature_extract_tester.get_expected_values(image_inputs, batched=True)

        self.assertEqual(
            encoded_images.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

    def test_call_pytorch(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random PyTorch tensors
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test not batched input
        encoded_images = feature_extractor(image_inputs[0], return_tensors="pt").pixel_values

        expected_height, expected_width = self.feature_extract_tester.get_expected_values(image_inputs)

        self.assertEqual(
            encoded_images.shape,
            (1, self.feature_extract_tester.num_channels, expected_height, expected_width),
        )

        # Test batched
        encoded_images = feature_extractor(image_inputs, return_tensors="pt").pixel_values

        expected_height, expected_width = self.feature_extract_tester.get_expected_values(image_inputs, batched=True)

        self.assertEqual(
            encoded_images.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

    def test_equivalence_pad_and_create_pixel_mask(self):
        # Initialize feature_extractors
        feature_extractor_1 = self.feature_extraction_class(**self.feat_extract_dict)
        feature_extractor_2 = self.feature_extraction_class(
            do_resize=False, do_normalize=False, num_labels=self.feature_extract_tester.num_classes
        )
        # create random PyTorch tensors
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test whether the method "pad_and_return_pixel_mask" and calling the feature extractor return the same tensors
        encoded_images_with_method = feature_extractor_1.encode_inputs(image_inputs, return_tensors="pt")
        encoded_images = feature_extractor_2(image_inputs, return_tensors="pt")

        self.assertTrue(
            torch.allclose(encoded_images_with_method["pixel_values"], encoded_images["pixel_values"], atol=1e-4)
        )
        self.assertTrue(
            torch.allclose(encoded_images_with_method["pixel_mask"], encoded_images["pixel_mask"], atol=1e-4)
        )

    def comm_get_feature_extractor_inputs(self, with_segmentation_maps=False, segmentation_type="np"):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # prepare image and target
        batch_size = self.feature_extract_tester.batch_size
        annotations = None

        if with_segmentation_maps:
            annotations = [
                np.random.randint(0, self.feature_extract_tester.num_labels, (384, 384)).astype(np.uint8)
                for _ in range(batch_size)
            ]
            if segmentation_type == "pil":
                annotations = [Image.fromarray(annotation) for annotation in annotations]

        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False)

        inputs = feature_extractor(image_inputs, annotations, return_tensors="pt", pad_and_return_pixel_mask=True)

        return inputs

    def test_init_without_params(self):
        pass

    def test_with_size_divisibility(self):
        size_divisibilities = [8, 16, 32]
        weird_input_sizes = [(407, 802), (582, 1094)]
        for size_divisibility in size_divisibilities:
            feat_extract_dict = {**self.feat_extract_dict, **{"size_divisibility": size_divisibility}}
            feature_extractor = self.feature_extraction_class(**feat_extract_dict)
            for weird_input_size in weird_input_sizes:
                inputs = feature_extractor([np.ones((3, *weird_input_size))], return_tensors="pt")
                pixel_values = inputs["pixel_values"]
                # check if divisible
                self.assertTrue((pixel_values.shape[-1] % size_divisibility) == 0)
                self.assertTrue((pixel_values.shape[-2] % size_divisibility) == 0)

    def test_call_with_numpy_segmentation_maps(self):
        batch_size = self.feature_extract_tester.batch_size

        inputs = self.comm_get_feature_extractor_inputs(with_segmentation_maps=True)

        # check the batch_size
        for el in inputs.values():
            self.assertEqual(el.shape[0], batch_size)

        pixel_values = inputs["pixel_values"]
        mask_labels = inputs["mask_labels"]
        class_labels = inputs["class_labels"]

        self.assertEqual(pixel_values.shape[-2], mask_labels.shape[-2])
        self.assertEqual(pixel_values.shape[-1], mask_labels.shape[-1])
        self.assertEqual(mask_labels.shape[1], class_labels.shape[1])
        self.assertEqual(mask_labels.shape[1], self.feature_extract_tester.num_labels)

    def test_call_with_pil_segmentation_maps(self):
        batch_size = self.feature_extract_tester.batch_size

        inputs = self.comm_get_feature_extractor_inputs(with_segmentation_maps=True, segmentation_type="pil")

        # check the batch_size
        for el in inputs.values():
            self.assertEqual(el.shape[0], batch_size)

        pixel_values = inputs["pixel_values"]
        mask_labels = inputs["mask_labels"]
        class_labels = inputs["class_labels"]

        self.assertEqual(pixel_values.shape[-2], mask_labels.shape[-2])
        self.assertEqual(pixel_values.shape[-1], mask_labels.shape[-1])
        self.assertEqual(mask_labels.shape[1], class_labels.shape[1])
        self.assertEqual(mask_labels.shape[1], self.feature_extract_tester.num_labels)

    def test_post_process_segmentation(self):
        fature_extractor = self.feature_extraction_class(num_labels=self.feature_extract_tester.num_classes)
        outputs = self.feature_extract_tester.get_fake_maskformer_outputs()
        segmentation = fature_extractor.post_process_segmentation(outputs)

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
        segmentation = fature_extractor.post_process_segmentation(outputs, target_size=target_size)

        self.assertEqual(
            segmentation.shape,
            (self.feature_extract_tester.batch_size, self.feature_extract_tester.num_classes, *target_size),
        )

    def test_post_process_semantic_segmentation(self):
        fature_extractor = self.feature_extraction_class(num_labels=self.feature_extract_tester.num_classes)
        outputs = self.feature_extract_tester.get_fake_maskformer_outputs()

        segmentation = fature_extractor.post_process_semantic_segmentation(outputs)

        self.assertEqual(
            segmentation.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.height,
                self.feature_extract_tester.width,
            ),
        )

        target_size = (1, 4)

        segmentation = fature_extractor.post_process_semantic_segmentation(outputs, target_size=target_size)

        self.assertEqual(segmentation.shape, (self.feature_extract_tester.batch_size, *target_size))

    def test_post_process_panoptic_segmentation(self):
        fature_extractor = self.feature_extraction_class(num_labels=self.feature_extract_tester.num_classes)
        outputs = self.feature_extract_tester.get_fake_maskformer_outputs()
        segmentation = fature_extractor.post_process_panoptic_segmentation(outputs, object_mask_threshold=0)

        self.assertTrue(len(segmentation) == self.feature_extract_tester.batch_size)
        for el in segmentation:
            self.assertTrue("segmentation" in el)
            self.assertTrue("segments" in el)
            self.assertEqual(type(el["segments"]), list)
            self.assertEqual(
                el["segmentation"].shape, (self.feature_extract_tester.height, self.feature_extract_tester.width)
            )
