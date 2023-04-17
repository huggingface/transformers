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

from ...test_image_processing_common import ImageProcessingSavingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

    if is_vision_available():
        from transformers import VideoMask2FormerImageProcessor
        from transformers.models.mask2former.image_processing_video_mask2former import binary_mask_to_rle
        from transformers.models.mask2former.modeling_video_mask2former import VideoMask2FormerForVideoSegmentationOutput

if is_vision_available():
    from PIL import Image


class VideoMask2FormerImageProcessingTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=2,
        num_frames=2,
        num_channels=3,
        min_resolution=30,
        max_resolution=400,
        size=None,
        do_resize=True,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        num_labels=10,
        do_reduce_labels=True,
        ignore_index=255,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = {"shortest_edge": 32, "longest_edge": 1333} if size is None else size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.size_divisor = 0
        # for the post_process_functions
        self.batch_dim = 1
        self.num_queries = 10
        self.num_classes = 2
        self.height = 3
        self.width = 4
        self.num_labels = num_labels
        self.do_reduce_labels = do_reduce_labels
        self.ignore_index = ignore_index

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "size_divisor": self.size_divisor,
            "num_labels": self.num_labels,
            "do_reduce_labels": self.do_reduce_labels,
            "ignore_index": self.ignore_index,
        }

    def get_expected_values(self, image_inputs, batched=False):
        """
        This function computes the expected height and width when providing images to VideoMask2FormerImageProcessor,
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
                expected_height, expected_width = self.get_expected_values([image])
                expected_values.append((expected_height, expected_width))
            expected_height = max(expected_values, key=lambda item: item[0])[0]
            expected_width = max(expected_values, key=lambda item: item[1])[1]

        return expected_height, expected_width

    def get_fake_video_mask2former_outputs(self):
        return VideoMask2FormerForVideoSegmentationOutput(
            # +1 for null class
            class_queries_logits=torch.randn((self.batch_dim, self.num_queries, self.num_classes + 1)),
            masks_queries_logits=torch.randn((self.num_queries, self.num_frames, self.height, self.width)),
        )


@require_torch
@require_vision
class VideoMask2FormerImageProcessingTest(ImageProcessingSavingTestMixin, unittest.TestCase):
    image_processing_class = VideoMask2FormerImageProcessor if (is_vision_available() and is_torch_available()) else None

    def setUp(self):
        self.image_processor_tester = VideoMask2FormerImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))
        self.assertTrue(hasattr(image_processing, "max_size"))
        self.assertTrue(hasattr(image_processing, "ignore_index"))
        self.assertTrue(hasattr(image_processing, "num_labels"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"shortest_edge": 32, "longest_edge": 1333})
        self.assertEqual(image_processor.size_divisor, 0)

        image_processor = self.image_processing_class.from_dict(
            self.image_processor_dict, size=42, max_size=84, size_divisibility=8
        )
        self.assertEqual(image_processor.size, {"shortest_edge": 42, "longest_edge": 84})
        self.assertEqual(image_processor.size_divisor, 8)

    def test_batch_feature(self):
        pass

    def test_call_pil(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PIL images
        image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values

        expected_height, expected_width = self.image_processor_tester.get_expected_values(image_inputs)

        self.assertEqual(
            encoded_images.shape,
            (1, self.image_processor_tester.num_channels, expected_height, expected_width),
        )

        # Test batched
        expected_height, expected_width = self.image_processor_tester.get_expected_values(image_inputs, batched=True)

        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

    def test_call_numpy(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random numpy tensors
        image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False, numpify=True)
        
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values

        expected_height, expected_width = self.image_processor_tester.get_expected_values(image_inputs)

        self.assertEqual(
            encoded_images.shape,
            (1, self.image_processor_tester.num_channels, expected_height, expected_width),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values

        expected_height, expected_width = self.image_processor_tester.get_expected_values(image_inputs, batched=True)

        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

    def test_call_pytorch(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PyTorch tensors
        image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values

        expected_height, expected_width = self.image_processor_tester.get_expected_values(image_inputs)

        self.assertEqual(
            encoded_images.shape,
            (1, self.image_processor_tester.num_channels, expected_height, expected_width),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values

        expected_height, expected_width = self.image_processor_tester.get_expected_values(image_inputs, batched=True)

        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

    def test_equivalence_pad_and_create_pixel_mask(self):
        # Initialize image_processings
        image_processing_1 = self.image_processing_class(**self.image_processor_dict)
        image_processing_2 = self.image_processing_class(
            do_resize=False, do_normalize=False, do_rescale=False, num_labels=self.image_processor_tester.num_classes
        )
        # create random PyTorch tensors
        image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test whether the method "pad_and_return_pixel_mask" and calling the image processor return the same tensors
        encoded_images_with_method = image_processing_1.encode_inputs(image_inputs, return_tensors="pt")
        encoded_images = image_processing_2(image_inputs, return_tensors="pt")

        self.assertTrue(
            torch.allclose(encoded_images_with_method["pixel_values"], encoded_images["pixel_values"], atol=1e-4)
        )
        self.assertTrue(
            torch.allclose(encoded_images_with_method["pixel_mask"], encoded_images["pixel_mask"], atol=1e-4)
        )

    def comm_get_image_processing_inputs(
        self, with_segmentation_maps=False, is_instance_map=False, segmentation_type="np"
    ):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # prepare image and target
        num_labels = self.image_processor_tester.num_labels
        annotations = None
        instance_id_to_semantic_id = None
        image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False)
        if with_segmentation_maps:
            high = num_labels
            if is_instance_map:
                labels_expanded = list(range(num_labels)) * 2
                instance_id_to_semantic_id = dict(enumerate(labels_expanded))
            annotations = [
                np.random.randint(0, high * 2, (img.size[1], img.size[0])).astype(np.uint8) for img in image_inputs
            ]
            if segmentation_type == "pil":
                annotations = [Image.fromarray(annotation) for annotation in annotations]
        
        inputs = image_processing(
            image_inputs,
            annotations,
            size=(480,640),
            is_train=True,
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
            image_processor_dict = {**self.image_processor_dict, **{"size_divisor": size_divisor}}
            image_processing = self.image_processing_class(**image_processor_dict)
            for weird_input_size in weird_input_sizes:
                inputs = image_processing([np.ones((3, *weird_input_size))], return_tensors="pt")
                pixel_values = inputs["pixel_values"]
                # check if divisible
                self.assertTrue((pixel_values.shape[-1] % size_divisor) == 0)
                self.assertTrue((pixel_values.shape[-2] % size_divisor) == 0)

    def test_call_with_segmentation_maps(self):
        def common(is_instance_map=False, segmentation_type=None):
            inputs = self.comm_get_image_processing_inputs(
                with_segmentation_maps=True, is_instance_map=is_instance_map, segmentation_type=segmentation_type
            )

            mask_labels = inputs["mask_labels"]
            class_labels = inputs["class_labels"]
            pixel_values = inputs["pixel_values"]
            
            # check that mask_label is of shape (`num_labels, num_frames, height, width`)
            self.assertEqual(mask_labels[0].shape, (class_labels[0].shape[0], 2, 480, 640))

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

        # create a image processor
        image_processing = VideoMask2FormerImageProcessor(reduce_labels=True, ignore_index=255, size=(512, 512))

        # prepare the images and annotations
        inputs = image_processing(
            [image1, image2],
            [instance_seg1, instance_seg2],
            is_train=True,
            instance_id_to_semantic_id=[inst2class1, inst2class2],
            return_tensors="pt",
        )

        # verify the pixel values and pixel mask
        self.assertEqual(inputs["pixel_values"].shape, (2, 3, 512, 512))
        self.assertEqual(inputs["pixel_mask"].shape, (2, 512, 512))

        # verify the class labels
        self.assertEqual(len(inputs["class_labels"][0]), 6)
        self.assertTrue(torch.allclose(inputs["class_labels"][0], torch.tensor([30, 55,  4,  4, 23, 55])))
        
        # verify the mask labels
        self.assertEqual(len(inputs["mask_labels"][0]), 6)
        self.assertEqual(inputs["mask_labels"][0].shape, (6, 2, 512, 512))
        self.assertEquals(inputs["mask_labels"][0].sum().item(), 67786.0)


    def test_binary_mask_to_rle(self):
        fake_binary_mask = np.zeros((20, 50))
        fake_binary_mask[0, 20:] = 1
        fake_binary_mask[1, :15] = 1
        fake_binary_mask[5, :10] = 1

        rle = binary_mask_to_rle(fake_binary_mask)
        self.assertEqual(len(rle), 4)
        self.assertEqual(rle[0], 21)
        self.assertEqual(rle[1], 45)

    def test_post_process_instance_segmentation(self):
        image_processor = self.image_processing_class(num_labels=self.image_processor_tester.num_classes)
        outputs = self.image_processor_tester.get_fake_video_mask2former_outputs()
        
        segmentation = image_processor.post_process_instance_segmentation(outputs, threshold=0)

        self.assertTrue(len(segmentation) == self.image_processor_tester.batch_dim)

        for el in segmentation:
            self.assertTrue("segmentation" in el)
            self.assertTrue("segments_info" in el)
            self.assertEqual(type(el["segments_info"]), list)
            self.assertEqual(el["segmentation"].shape, (480, 640))

        segmentation = image_processor.post_process_instance_segmentation(
            outputs, threshold=0, return_binary_maps=True
        )

        self.assertTrue(len(segmentation) == self.image_processor_tester.batch_dim)
        for el in segmentation:
            self.assertTrue("segmentation" in el)
            self.assertTrue("segments_info" in el)
            self.assertEqual(type(el["segments_info"]), list)
            self.assertEqual(len(el["segmentation"].shape), 3)
            self.assertEqual(el["segmentation"].shape[1:], (480, 640))