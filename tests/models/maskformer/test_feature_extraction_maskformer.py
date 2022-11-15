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

from huggingface_hub import hf_hub_download
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_feature_extraction_common import FeatureExtractionSavingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

    if is_vision_available():
        from transformers import MaskFormerFeatureExtractor
        from transformers.models.maskformer.feature_extraction_maskformer import binary_mask_to_rle
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

    def comm_get_feature_extractor_inputs(
        self, with_segmentation_maps=False, is_instance_map=False, segmentation_type="np"
    ):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # prepare image and target
        batch_size = self.feature_extract_tester.batch_size
        num_labels = self.feature_extract_tester.num_labels
        annotations = None
        instance_id_to_semantic_id = None
        if with_segmentation_maps:
            high = num_labels
            if is_instance_map:
                high * 2
                labels_expanded = list(range(num_labels)) * 2
                instance_id_to_semantic_id = {
                    instance_id: label_id for instance_id, label_id in enumerate(labels_expanded)
                }
            annotations = [np.random.randint(0, high, (384, 384)).astype(np.uint8) for _ in range(batch_size)]
            if segmentation_type == "pil":
                annotations = [Image.fromarray(annotation) for annotation in annotations]

        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False)
        inputs = feature_extractor(
            image_inputs,
            annotations,
            return_tensors="pt",
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            pad_and_return_pixel_mask=True,
        )

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

    def test_call_with_segmentation_maps(self):
        def common(is_instance_map=False, segmentation_type=None):
            inputs = self.comm_get_feature_extractor_inputs(
                with_segmentation_maps=True, is_instance_map=is_instance_map, segmentation_type=segmentation_type
            )

            mask_labels = inputs["mask_labels"]
            class_labels = inputs["class_labels"]
            pixel_values = inputs["pixel_values"]

            # check the batch_size
            for mask_label, class_label in zip(mask_labels, class_labels):
                self.assertEqual(mask_label.shape[0], class_label.shape[0])
                # this ensure padding has happened
                self.assertEqual(mask_label.shape[1:], pixel_values.shape[2:])

        common()
        common(is_instance_map=True)
        common(is_instance_map=False, segmentation_type="pil")
        common(is_instance_map=True, segmentation_type="pil")

    def test_integration_instance_segmentation(self):
        # load 2 images and corresponding annotations from the hub
        repo_id = "nielsr/image-segmentation-toy-data"
        image1 = Image.open(
            hf_hub_download(repo_id=repo_id, filename="instance_segmentation_image_1.png", repo_type="dataset")
        )
        image2 = Image.open(
            hf_hub_download(repo_id=repo_id, filename="instance_segmentation_image_2.png", repo_type="dataset")
        )
        annotation1 = Image.open(
            hf_hub_download(repo_id=repo_id, filename="instance_segmentation_annotation_1.png", repo_type="dataset")
        )
        annotation2 = Image.open(
            hf_hub_download(repo_id=repo_id, filename="instance_segmentation_annotation_2.png", repo_type="dataset")
        )

        # get instance segmentations and instance-to-segmentation mappings
        def get_instance_segmentation_and_mapping(annotation):
            instance_seg = np.array(annotation)[:, :, 1]
            class_id_map = np.array(annotation)[:, :, 0]
            class_labels = np.unique(class_id_map)

            # create mapping between instance IDs and semantic category IDs
            inst2class = {}
            for label in class_labels:
                instance_ids = np.unique(instance_seg[class_id_map == label])
                inst2class.update({i: label for i in instance_ids})

            return instance_seg, inst2class

        instance_seg1, inst2class1 = get_instance_segmentation_and_mapping(annotation1)
        instance_seg2, inst2class2 = get_instance_segmentation_and_mapping(annotation2)

        # create a feature extractor
        feature_extractor = MaskFormerFeatureExtractor(reduce_labels=True, ignore_index=255, size=(512, 512))

        # prepare the images and annotations
        inputs = feature_extractor(
            [image1, image2],
            [instance_seg1, instance_seg2],
            instance_id_to_semantic_id=[inst2class1, inst2class2],
            return_tensors="pt",
        )

        # verify the pixel values and pixel mask
        self.assertEqual(inputs["pixel_values"].shape, (2, 3, 512, 512))
        self.assertEqual(inputs["pixel_mask"].shape, (2, 512, 512))

        # verify the class labels
        self.assertEqual(len(inputs["class_labels"]), 2)
        self.assertTrue(torch.allclose(inputs["class_labels"][0], torch.tensor([30, 55])))
        self.assertTrue(torch.allclose(inputs["class_labels"][1], torch.tensor([4, 4, 23, 55])))

        # verify the mask labels
        self.assertEqual(len(inputs["mask_labels"]), 2)
        self.assertEqual(inputs["mask_labels"][0].shape, (2, 512, 512))
        self.assertEqual(inputs["mask_labels"][1].shape, (4, 512, 512))
        self.assertEquals(inputs["mask_labels"][0].sum().item(), 41527.0)
        self.assertEquals(inputs["mask_labels"][1].sum().item(), 26259.0)

    def test_integration_semantic_segmentation(self):
        # load 2 images and corresponding semantic annotations from the hub
        repo_id = "nielsr/image-segmentation-toy-data"
        image1 = Image.open(
            hf_hub_download(repo_id=repo_id, filename="semantic_segmentation_image_1.png", repo_type="dataset")
        )
        image2 = Image.open(
            hf_hub_download(repo_id=repo_id, filename="semantic_segmentation_image_2.png", repo_type="dataset")
        )
        annotation1 = Image.open(
            hf_hub_download(repo_id=repo_id, filename="semantic_segmentation_annotation_1.png", repo_type="dataset")
        )
        annotation2 = Image.open(
            hf_hub_download(repo_id=repo_id, filename="semantic_segmentation_annotation_2.png", repo_type="dataset")
        )

        # create a feature extractor
        feature_extractor = MaskFormerFeatureExtractor(reduce_labels=True, ignore_index=255, size=(512, 512))

        # prepare the images and annotations
        inputs = feature_extractor(
            [image1, image2],
            [annotation1, annotation2],
            return_tensors="pt",
        )

        # verify the pixel values and pixel mask
        self.assertEqual(inputs["pixel_values"].shape, (2, 3, 512, 512))
        self.assertEqual(inputs["pixel_mask"].shape, (2, 512, 512))

        # verify the class labels
        self.assertEqual(len(inputs["class_labels"]), 2)
        self.assertTrue(torch.allclose(inputs["class_labels"][0], torch.tensor([2, 4, 60])))
        self.assertTrue(torch.allclose(inputs["class_labels"][1], torch.tensor([0, 3, 7, 8, 15, 28, 30, 143])))

        # verify the mask labels
        self.assertEqual(len(inputs["mask_labels"]), 2)
        self.assertEqual(inputs["mask_labels"][0].shape, (3, 512, 512))
        self.assertEqual(inputs["mask_labels"][1].shape, (8, 512, 512))
        self.assertEquals(inputs["mask_labels"][0].sum().item(), 170200.0)
        self.assertEquals(inputs["mask_labels"][1].sum().item(), 257036.0)

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
        feature_extractor = MaskFormerFeatureExtractor(ignore_index=0, do_resize=False)

        # prepare the images and annotations
        pixel_values_list = [np.moveaxis(np.array(image1), -1, 0), np.moveaxis(np.array(image2), -1, 0)]
        inputs = feature_extractor.encode_inputs(
            pixel_values_list,
            [panoptic_map1, panoptic_map2],
            instance_id_to_semantic_id=[inst2class1, inst2class2],
            return_tensors="pt",
        )

        # verify the pixel values and pixel mask
        self.assertEqual(inputs["pixel_values"].shape, (2, 3, 512, 711))
        self.assertEqual(inputs["pixel_mask"].shape, (2, 512, 711))

        # verify the class labels
        self.assertEqual(len(inputs["class_labels"]), 2)
        # fmt: off
        expected_class_labels = torch.tensor([4, 17, 32, 42, 42, 42, 42, 42, 42, 42, 32, 12, 12, 12, 12, 12, 42, 42, 12, 12, 12, 42, 12, 12, 12, 12, 12, 3, 12, 12, 12, 12, 42, 42, 42, 12, 42, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 5, 12, 12, 12, 12, 12, 12, 12, 0, 43, 43, 43, 96, 43, 104, 43, 31, 125, 31, 125, 138, 87, 125, 149, 138, 125, 87, 87])  # noqa: E231
        # fmt: on
        self.assertTrue(torch.allclose(inputs["class_labels"][0], torch.tensor(expected_class_labels)))
        # fmt: off
        expected_class_labels = torch.tensor([19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 67, 82, 19, 19, 17, 19, 19, 19, 19, 19, 19, 19, 19, 19, 12, 12, 42, 12, 12, 12, 12, 3, 14, 12, 12, 12, 12, 12, 12, 12, 12, 14, 5, 12, 12, 0, 115, 43, 43, 115, 43, 43, 43, 8, 8, 8, 138, 138, 125, 143])  # noqa: E231
        # fmt: on
        self.assertTrue(torch.allclose(inputs["class_labels"][1], expected_class_labels))

        # verify the mask labels
        self.assertEqual(len(inputs["mask_labels"]), 2)
        self.assertEqual(inputs["mask_labels"][0].shape, (79, 512, 711))
        self.assertEqual(inputs["mask_labels"][1].shape, (61, 512, 711))
        self.assertEquals(inputs["mask_labels"][0].sum().item(), 315193.0)
        self.assertEquals(inputs["mask_labels"][1].sum().item(), 350747.0)

    def test_binary_mask_to_rle(self):
        fake_binary_mask = np.zeros((20, 50))
        fake_binary_mask[0, 20:] = 1
        fake_binary_mask[1, :15] = 1
        fake_binary_mask[5, :10] = 1

        rle = binary_mask_to_rle(fake_binary_mask)
        self.assertEqual(len(rle), 4)
        self.assertEqual(rle[0], 21)
        self.assertEqual(rle[1], 45)

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

    def test_post_process_panoptic_segmentation(self):
        feature_extractor = self.feature_extraction_class(num_labels=self.feature_extract_tester.num_classes)
        outputs = self.feature_extract_tester.get_fake_maskformer_outputs()
        segmentation = feature_extractor.post_process_panoptic_segmentation(outputs, threshold=0)

        self.assertTrue(len(segmentation) == self.feature_extract_tester.batch_size)
        for el in segmentation:
            self.assertTrue("segmentation" in el)
            self.assertTrue("segments_info" in el)
            self.assertEqual(type(el["segments_info"]), list)
            self.assertEqual(
                el["segmentation"].shape, (self.feature_extract_tester.height, self.feature_extract_tester.width)
            )
