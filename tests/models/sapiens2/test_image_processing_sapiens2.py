# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

    from transformers import Sapiens2ImageProcessor
    from transformers.modeling_outputs import SemanticSegmenterOutput
    from transformers.models.sapiens2.modeling_sapiens2 import (
        Sapiens2ImageMattingOutput,
        Sapiens2NormalEstimatorOutput,
        Sapiens2PointmapEstimatorOutput,
    )


class Sapiens2ImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        do_reduce_labels=False,
        num_labels=5,
    ):
        super().__init__()
        size = size if size is not None else {"height": 20, "width": 18}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_reduce_labels = do_reduce_labels
        self.num_labels = num_labels

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_reduce_labels": self.do_reduce_labels,
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["height"], self.size["width"]

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

    def prepare_post_process_semantic_segmentation_inputs(self):
        inputs = {
            "outputs": SemanticSegmenterOutput(
                logits=torch.randn(self.batch_size, self.num_labels, self.size["height"], self.size["width"])
            )
        }
        expected_shape = {
            "num_labels": self.num_labels,
            "height": self.size["height"],
            "width": self.size["width"],
        }
        return inputs, expected_shape


@require_torch
@require_vision
class Sapiens2ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = Sapiens2ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_reduce_labels"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"height": 20, "width": 18})
            self.assertEqual(image_processor.do_reduce_labels, False)

            image_processor = image_processing_class.from_dict(
                self.image_processor_dict, size={"height": 42, "width": 42}, do_reduce_labels=True
            )
            self.assertEqual(image_processor.size, {"height": 42, "width": 42})
            self.assertEqual(image_processor.do_reduce_labels, True)

    def test_call_segmentation_maps(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
            maps = [torch.zeros(image.shape[-2:]).long() for image in image_inputs]

            # Single image + map
            encoding = image_processing(image_inputs[0], maps[0], return_tensors="pt")
            self.assertEqual(
                encoding["pixel_values"].shape,
                (
                    1,
                    self.image_processor_tester.num_channels,
                    self.image_processor_tester.size["height"],
                    self.image_processor_tester.size["width"],
                ),
            )
            self.assertEqual(
                encoding["labels"].shape,
                (1, self.image_processor_tester.size["height"], self.image_processor_tester.size["width"]),
            )
            self.assertEqual(encoding["labels"].dtype, torch.long)
            self.assertTrue(encoding["labels"].min().item() >= 0)
            self.assertTrue(encoding["labels"].max().item() <= 255)

            # Batched images + maps
            encoding = image_processing(image_inputs, maps, return_tensors="pt")
            self.assertEqual(
                encoding["pixel_values"].shape,
                (
                    self.image_processor_tester.batch_size,
                    self.image_processor_tester.num_channels,
                    self.image_processor_tester.size["height"],
                    self.image_processor_tester.size["width"],
                ),
            )
            self.assertEqual(
                encoding["labels"].shape,
                (
                    self.image_processor_tester.batch_size,
                    self.image_processor_tester.size["height"],
                    self.image_processor_tester.size["width"],
                ),
            )
            self.assertEqual(encoding["labels"].dtype, torch.long)
            self.assertTrue(encoding["labels"].min().item() >= 0)
            self.assertTrue(encoding["labels"].max().item() <= 255)

    def test_post_process_normal_estimation(self):
        image_processor = Sapiens2ImageProcessor()
        batch_size = 2
        num_labels = 3
        height = width = 16
        outputs = Sapiens2NormalEstimatorOutput(normals=torch.randn(batch_size, num_labels, height, width))

        # without target_sizes: spatial dims match normals, values are L2-normalized
        result = image_processor.post_process_normal_estimation(outputs)
        self.assertEqual(len(result), batch_size)
        self.assertEqual(result[0]["normals"].shape, torch.Size([num_labels, height, width]))
        norms = result[0]["normals"].norm(p=2, dim=0)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4)

        # with target_sizes: output is resized before normalization
        target_sizes = [(height * 2, width * 2)] * batch_size
        result = image_processor.post_process_normal_estimation(outputs, target_sizes=target_sizes)
        self.assertEqual(len(result), batch_size)
        self.assertEqual(result[0]["normals"].shape, torch.Size([num_labels, height * 2, width * 2]))

        # mismatched batch size raises ValueError
        with self.assertRaises(ValueError):
            image_processor.post_process_normal_estimation(outputs, target_sizes=[(100, 100)])

    def test_post_process_pointmap_estimation(self):
        image_processor = Sapiens2ImageProcessor()
        batch_size = 2
        num_labels = 3
        height = width = 16
        outputs = Sapiens2PointmapEstimatorOutput(pointmaps=torch.randn(batch_size, num_labels, height, width))

        # without target_sizes: spatial dims match pointmap
        result = image_processor.post_process_pointmap_estimation(outputs)
        self.assertEqual(len(result), batch_size)
        self.assertEqual(result[0]["pointmap"].shape, torch.Size([num_labels, height, width]))

        # with target_sizes: output is resized to requested size
        target_sizes = [(height * 2, width * 2)] * batch_size
        result = image_processor.post_process_pointmap_estimation(outputs, target_sizes=target_sizes)
        self.assertEqual(len(result), batch_size)
        self.assertEqual(result[0]["pointmap"].shape, torch.Size([num_labels, height * 2, width * 2]))

        # with scales: scale division is applied
        scale = torch.tensor([[2.0], [0.5]])
        outputs_with_scale = Sapiens2PointmapEstimatorOutput(
            pointmaps=torch.ones(batch_size, num_labels, height, width), scales=scale
        )
        result = image_processor.post_process_pointmap_estimation(outputs_with_scale)
        torch.testing.assert_close(result[0]["pointmap"], torch.full((num_labels, height, width), 0.5))
        torch.testing.assert_close(result[1]["pointmap"], torch.full((num_labels, height, width), 2.0))

        # mismatched batch size raises ValueError
        with self.assertRaises(ValueError):
            image_processor.post_process_pointmap_estimation(outputs, target_sizes=[(100, 100)])

    def test_post_process_image_matting(self):
        image_processor = Sapiens2ImageProcessor()
        batch_size = 2
        height = width = 16
        outputs = Sapiens2ImageMattingOutput(
            foregrounds=torch.rand(batch_size, 3, height, width),
            alphas=torch.rand(batch_size, 1, height, width),
        )

        # without target_sizes: spatial dims unchanged
        result = image_processor.post_process_image_matting(outputs)
        self.assertEqual(len(result), batch_size)
        self.assertEqual(result[0]["foreground"].shape, torch.Size([3, height, width]))
        self.assertEqual(result[0]["alpha"].shape, torch.Size([1, height, width]))
        # values stay in [0, 1]
        self.assertGreaterEqual(result[0]["alpha"].min().item(), 0.0)
        self.assertLessEqual(result[0]["alpha"].max().item(), 1.0)

        # with target_sizes: output is resized
        target_sizes = [(height * 2, width * 2)] * batch_size
        result = image_processor.post_process_image_matting(outputs, target_sizes=target_sizes)
        self.assertEqual(result[0]["foreground"].shape, torch.Size([3, height * 2, width * 2]))

        # mismatched batch size raises ValueError
        with self.assertRaises(ValueError):
            image_processor.post_process_image_matting(outputs, target_sizes=[(100, 100)])
