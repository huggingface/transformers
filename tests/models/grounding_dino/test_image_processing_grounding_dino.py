# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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


import json
import pathlib
import unittest

from transformers.testing_utils import require_torch, require_vision, slow
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import AnnotationFormatTestMixin, ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

    from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoObjectDetectionOutput

if is_vision_available():
    from PIL import Image

    from transformers import GroundingDinoImageProcessor


class GroundingDinoImageProcessingTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_rescale=True,
        rescale_factor=1 / 255,
        do_pad=True,
    ):
        # by setting size["longest_edge"] > max_resolution we're effectively not testing this :p
        size = size if size is not None else {"shortest_edge": 18, "longest_edge": 1333}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_pad = do_pad
        self.num_queries = 5
        self.embed_dim = 5

    # Copied from tests.models.deformable_detr.test_image_processing_deformable_detr.DeformableDetrImageProcessingTester.prepare_image_processor_dict with DeformableDetr->GroundingDino
    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_pad": self.do_pad,
        }

    # Copied from tests.models.deformable_detr.test_image_processing_deformable_detr.DeformableDetrImageProcessingTester.get_expected_values with DeformableDetr->GroundingDino
    def get_expected_values(self, image_inputs, batched=False):
        """
        This function computes the expected height and width when providing images to GroundingDinoImageProcessor,
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

    # Copied from tests.models.deformable_detr.test_image_processing_deformable_detr.DeformableDetrImageProcessingTester.expected_output_image_shape with DeformableDetr->GroundingDino
    def expected_output_image_shape(self, images):
        height, width = self.get_expected_values(images, batched=True)
        return self.num_channels, height, width

    def get_fake_grounding_dino_output(self):
        torch.manual_seed(42)
        return GroundingDinoObjectDetectionOutput(
            pred_boxes=torch.rand(self.batch_size, self.num_queries, 4),
            logits=torch.rand(self.batch_size, self.num_queries, self.embed_dim),
        )

    # Copied from tests.models.deformable_detr.test_image_processing_deformable_detr.DeformableDetrImageProcessingTester.prepare_image_inputs with DeformableDetr->GroundingDino
    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


@require_torch
@require_vision
class GroundingDinoImageProcessingTest(AnnotationFormatTestMixin, ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = GroundingDinoImageProcessor if is_vision_available() else None

    def setUp(self):
        self.image_processor_tester = GroundingDinoImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    # Copied from tests.models.deformable_detr.test_image_processing_deformable_detr.DeformableDetrImageProcessingTest.test_image_processor_properties with DeformableDetr->GroundingDino
    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "do_rescale"))
        self.assertTrue(hasattr(image_processing, "do_pad"))
        self.assertTrue(hasattr(image_processing, "size"))

    # Copied from tests.models.deformable_detr.test_image_processing_deformable_detr.DeformableDetrImageProcessingTest.test_image_processor_from_dict_with_kwargs with DeformableDetr->GroundingDino
    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"shortest_edge": 18, "longest_edge": 1333})
        self.assertEqual(image_processor.do_pad, True)

        image_processor = self.image_processing_class.from_dict(
            self.image_processor_dict, size=42, max_size=84, pad_and_return_pixel_mask=False
        )
        self.assertEqual(image_processor.size, {"shortest_edge": 42, "longest_edge": 84})
        self.assertEqual(image_processor.do_pad, False)

    def test_post_process_object_detection(self):
        image_processor = self.image_processing_class(**self.image_processor_dict)
        outputs = self.image_processor_tester.get_fake_grounding_dino_output()
        results = image_processor.post_process_object_detection(outputs, threshold=0.0)

        self.assertEqual(len(results), self.image_processor_tester.batch_size)
        self.assertEqual(list(results[0].keys()), ["scores", "labels", "boxes"])
        self.assertEqual(results[0]["boxes"].shape, (self.image_processor_tester.num_queries, 4))
        self.assertEqual(results[0]["scores"].shape, (self.image_processor_tester.num_queries,))

        expected_scores = torch.tensor([0.7050, 0.7222, 0.7222, 0.6829, 0.7220])
        self.assertTrue(torch.allclose(results[0]["scores"], expected_scores, atol=1e-4))

        expected_box_slice = torch.tensor([0.6908, 0.4354, 1.0737, 1.3947])
        self.assertTrue(torch.allclose(results[0]["boxes"][0], expected_box_slice, atol=1e-4))

    @slow
    # Copied from tests.models.deformable_detr.test_image_processing_deformable_detr.DeformableDetrImageProcessingTest.test_call_pytorch_with_coco_detection_annotations with DeformableDetr->GroundingDino
    def test_call_pytorch_with_coco_detection_annotations(self):
        # prepare image and target
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        with open("./tests/fixtures/tests_samples/COCO/coco_annotations.txt", "r") as f:
            target = json.loads(f.read())

        target = {"image_id": 39769, "annotations": target}

        # encode them
        image_processing = GroundingDinoImageProcessor()
        encoding = image_processing(images=image, annotations=target, return_tensors="pt")

        # verify pixel values
        expected_shape = torch.Size([1, 3, 800, 1066])
        self.assertEqual(encoding["pixel_values"].shape, expected_shape)

        expected_slice = torch.tensor([0.2796, 0.3138, 0.3481])
        self.assertTrue(torch.allclose(encoding["pixel_values"][0, 0, 0, :3], expected_slice, atol=1e-4))

        # verify area
        expected_area = torch.tensor([5887.9600, 11250.2061, 489353.8438, 837122.7500, 147967.5156, 165732.3438])
        self.assertTrue(torch.allclose(encoding["labels"][0]["area"], expected_area))
        # verify boxes
        expected_boxes_shape = torch.Size([6, 4])
        self.assertEqual(encoding["labels"][0]["boxes"].shape, expected_boxes_shape)
        expected_boxes_slice = torch.tensor([0.5503, 0.2765, 0.0604, 0.2215])
        self.assertTrue(torch.allclose(encoding["labels"][0]["boxes"][0], expected_boxes_slice, atol=1e-3))
        # verify image_id
        expected_image_id = torch.tensor([39769])
        self.assertTrue(torch.allclose(encoding["labels"][0]["image_id"], expected_image_id))
        # verify is_crowd
        expected_is_crowd = torch.tensor([0, 0, 0, 0, 0, 0])
        self.assertTrue(torch.allclose(encoding["labels"][0]["iscrowd"], expected_is_crowd))
        # verify class_labels
        expected_class_labels = torch.tensor([75, 75, 63, 65, 17, 17])
        self.assertTrue(torch.allclose(encoding["labels"][0]["class_labels"], expected_class_labels))
        # verify orig_size
        expected_orig_size = torch.tensor([480, 640])
        self.assertTrue(torch.allclose(encoding["labels"][0]["orig_size"], expected_orig_size))
        # verify size
        expected_size = torch.tensor([800, 1066])
        self.assertTrue(torch.allclose(encoding["labels"][0]["size"], expected_size))

    @slow
    def test_call_pytorch_with_coco_detection_annotations_batched(self):
        # prepare image and target
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        with open("./tests/fixtures/tests_samples/COCO/coco_annotations.txt", "r") as f:
            target = json.loads(f.read())

        target = {"image_id": 39769, "annotations": target}

        # Creating a batch of images and targets
        image_batch = [image, image]
        target_batch = [target, target]

        # encode them
        image_processing = GroundingDinoImageProcessor()
        encoding = image_processing(images=image_batch, annotations=target_batch, return_tensors="pt")

        # verify pixel values
        expected_shape = torch.Size([2, 3, 800, 1066])
        self.assertEqual(encoding["pixel_values"].shape, expected_shape)

        expected_slice = torch.tensor([0.2796, 0.3138, 0.3481])
        self.assertTrue(torch.allclose(encoding["pixel_values"][0, 0, 0, :3], expected_slice, atol=1e-4))
        self.assertTrue(torch.allclose(encoding["pixel_values"][1, 0, 0, :3], expected_slice, atol=1e-4))

        # verify area
        expected_area = torch.tensor([5887.9600, 11250.2061, 489353.8438, 837122.7500, 147967.5156, 165732.3438])
        self.assertTrue(torch.allclose(encoding["labels"][0]["area"], expected_area))
        self.assertTrue(torch.allclose(encoding["labels"][1]["area"], expected_area))
        # verify boxes
        expected_boxes_shape = torch.Size([6, 4])
        self.assertEqual(encoding["labels"][0]["boxes"].shape, expected_boxes_shape)
        self.assertEqual(encoding["labels"][1]["boxes"].shape, expected_boxes_shape)
        expected_boxes_slice = torch.tensor([0.5503, 0.2765, 0.0604, 0.2215])
        self.assertTrue(torch.allclose(encoding["labels"][0]["boxes"][0], expected_boxes_slice, atol=1e-3))
        self.assertTrue(torch.allclose(encoding["labels"][1]["boxes"][0], expected_boxes_slice, atol=1e-3))
        # verify image_id
        expected_image_id = torch.tensor([39769])
        self.assertTrue(torch.allclose(encoding["labels"][0]["image_id"], expected_image_id))
        self.assertTrue(torch.allclose(encoding["labels"][1]["image_id"], expected_image_id))
        # verify is_crowd
        expected_is_crowd = torch.tensor([0, 0, 0, 0, 0, 0])
        self.assertTrue(torch.allclose(encoding["labels"][0]["iscrowd"], expected_is_crowd))
        self.assertTrue(torch.allclose(encoding["labels"][1]["iscrowd"], expected_is_crowd))
        # verify class_labels
        expected_class_labels = torch.tensor([75, 75, 63, 65, 17, 17])
        self.assertTrue(torch.allclose(encoding["labels"][0]["class_labels"], expected_class_labels))
        self.assertTrue(torch.allclose(encoding["labels"][1]["class_labels"], expected_class_labels))
        # verify orig_size
        expected_orig_size = torch.tensor([480, 640])
        self.assertTrue(torch.allclose(encoding["labels"][0]["orig_size"], expected_orig_size))
        self.assertTrue(torch.allclose(encoding["labels"][1]["orig_size"], expected_orig_size))
        # verify size
        expected_size = torch.tensor([800, 1066])
        self.assertTrue(torch.allclose(encoding["labels"][0]["size"], expected_size))
        self.assertTrue(torch.allclose(encoding["labels"][1]["size"], expected_size))

    @slow
    # Copied from tests.models.deformable_detr.test_image_processing_deformable_detr.DeformableDetrImageProcessingTest.test_call_pytorch_with_coco_panoptic_annotations with DeformableDetr->GroundingDino
    def test_call_pytorch_with_coco_panoptic_annotations(self):
        # prepare image, target and masks_path
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        with open("./tests/fixtures/tests_samples/COCO/coco_panoptic_annotations.txt", "r") as f:
            target = json.loads(f.read())

        target = {"file_name": "000000039769.png", "image_id": 39769, "segments_info": target}

        masks_path = pathlib.Path("./tests/fixtures/tests_samples/COCO/coco_panoptic")

        # encode them
        image_processing = GroundingDinoImageProcessor(format="coco_panoptic")
        encoding = image_processing(images=image, annotations=target, masks_path=masks_path, return_tensors="pt")

        # verify pixel values
        expected_shape = torch.Size([1, 3, 800, 1066])
        self.assertEqual(encoding["pixel_values"].shape, expected_shape)

        expected_slice = torch.tensor([0.2796, 0.3138, 0.3481])
        self.assertTrue(torch.allclose(encoding["pixel_values"][0, 0, 0, :3], expected_slice, atol=1e-4))

        # verify area
        expected_area = torch.tensor([147979.6875, 165527.0469, 484638.5938, 11292.9375, 5879.6562, 7634.1147])
        self.assertTrue(torch.allclose(encoding["labels"][0]["area"], expected_area))
        # verify boxes
        expected_boxes_shape = torch.Size([6, 4])
        self.assertEqual(encoding["labels"][0]["boxes"].shape, expected_boxes_shape)
        expected_boxes_slice = torch.tensor([0.2625, 0.5437, 0.4688, 0.8625])
        self.assertTrue(torch.allclose(encoding["labels"][0]["boxes"][0], expected_boxes_slice, atol=1e-3))
        # verify image_id
        expected_image_id = torch.tensor([39769])
        self.assertTrue(torch.allclose(encoding["labels"][0]["image_id"], expected_image_id))
        # verify is_crowd
        expected_is_crowd = torch.tensor([0, 0, 0, 0, 0, 0])
        self.assertTrue(torch.allclose(encoding["labels"][0]["iscrowd"], expected_is_crowd))
        # verify class_labels
        expected_class_labels = torch.tensor([17, 17, 63, 75, 75, 93])
        self.assertTrue(torch.allclose(encoding["labels"][0]["class_labels"], expected_class_labels))
        # verify masks
        expected_masks_sum = 822873
        self.assertEqual(encoding["labels"][0]["masks"].sum().item(), expected_masks_sum)
        # verify orig_size
        expected_orig_size = torch.tensor([480, 640])
        self.assertTrue(torch.allclose(encoding["labels"][0]["orig_size"], expected_orig_size))
        # verify size
        expected_size = torch.tensor([800, 1066])
        self.assertTrue(torch.allclose(encoding["labels"][0]["size"], expected_size))

    @slow
    def test_call_pytorch_with_coco_panoptic_annotations_batched(self):
        # prepare image, target and masks_path
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        with open("./tests/fixtures/tests_samples/COCO/coco_panoptic_annotations.txt", "r") as f:
            target = json.loads(f.read())

        target = {"file_name": "000000039769.png", "image_id": 39769, "segments_info": target}

        masks_path = pathlib.Path("./tests/fixtures/tests_samples/COCO/coco_panoptic")

        image_batch = [image, image]
        target_batch = [target, target]

        # encode them
        image_processing = GroundingDinoImageProcessor(format="coco_panoptic")
        encoding = image_processing(
            images=image_batch, annotations=target_batch, masks_path=masks_path, return_tensors="pt"
        )

        # verify pixel values
        expected_shape = torch.Size([2, 3, 800, 1066])
        self.assertEqual(encoding["pixel_values"].shape, expected_shape)

        expected_slice = torch.tensor([0.2796, 0.3138, 0.3481])
        self.assertTrue(torch.allclose(encoding["pixel_values"][0, 0, 0, :3], expected_slice, atol=1e-4))
        self.assertTrue(torch.allclose(encoding["pixel_values"][1, 0, 0, :3], expected_slice, atol=1e-4))

        # verify area
        expected_area = torch.tensor([147979.6875, 165527.0469, 484638.5938, 11292.9375, 5879.6562, 7634.1147])
        self.assertTrue(torch.allclose(encoding["labels"][0]["area"], expected_area))
        self.assertTrue(torch.allclose(encoding["labels"][1]["area"], expected_area))
        # verify boxes
        expected_boxes_shape = torch.Size([6, 4])
        self.assertEqual(encoding["labels"][0]["boxes"].shape, expected_boxes_shape)
        self.assertEqual(encoding["labels"][1]["boxes"].shape, expected_boxes_shape)
        expected_boxes_slice = torch.tensor([0.2625, 0.5437, 0.4688, 0.8625])
        self.assertTrue(torch.allclose(encoding["labels"][0]["boxes"][0], expected_boxes_slice, atol=1e-3))
        self.assertTrue(torch.allclose(encoding["labels"][1]["boxes"][0], expected_boxes_slice, atol=1e-3))
        # verify image_id
        expected_image_id = torch.tensor([39769])
        self.assertTrue(torch.allclose(encoding["labels"][0]["image_id"], expected_image_id))
        self.assertTrue(torch.allclose(encoding["labels"][1]["image_id"], expected_image_id))
        # verify is_crowd
        expected_is_crowd = torch.tensor([0, 0, 0, 0, 0, 0])
        self.assertTrue(torch.allclose(encoding["labels"][0]["iscrowd"], expected_is_crowd))
        self.assertTrue(torch.allclose(encoding["labels"][1]["iscrowd"], expected_is_crowd))
        # verify class_labels
        expected_class_labels = torch.tensor([17, 17, 63, 75, 75, 93])
        self.assertTrue(torch.allclose(encoding["labels"][0]["class_labels"], expected_class_labels))
        self.assertTrue(torch.allclose(encoding["labels"][1]["class_labels"], expected_class_labels))
        # verify masks
        expected_masks_sum = 822873
        self.assertEqual(encoding["labels"][0]["masks"].sum().item(), expected_masks_sum)
        self.assertEqual(encoding["labels"][1]["masks"].sum().item(), expected_masks_sum)
        # verify orig_size
        expected_orig_size = torch.tensor([480, 640])
        self.assertTrue(torch.allclose(encoding["labels"][0]["orig_size"], expected_orig_size))
        self.assertTrue(torch.allclose(encoding["labels"][1]["orig_size"], expected_orig_size))
        # verify size
        expected_size = torch.tensor([800, 1066])
        self.assertTrue(torch.allclose(encoding["labels"][0]["size"], expected_size))
        self.assertTrue(torch.allclose(encoding["labels"][1]["size"], expected_size))
