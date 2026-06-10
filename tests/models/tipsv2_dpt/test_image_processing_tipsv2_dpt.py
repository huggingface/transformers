# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

    from transformers import Tipsv2DptImageProcessor
    from transformers.models.tipsv2_dpt.modeling_tipsv2_dpt import Tipsv2DptNormalEstimatorOutput
    from transformers.modeling_outputs import DepthEstimatorOutput, SemanticSegmenterOutput


class Tipsv2DptImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=False,
        do_convert_rgb=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size if size is not None else {"height": 18, "width": 18}
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "do_convert_rgb": self.do_convert_rgb,
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


@require_torch
@require_vision
class Tipsv2DptImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = Tipsv2DptImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "do_rescale"))
            self.assertTrue(hasattr(image_processing, "rescale_factor"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "do_convert_rgb"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"height": 18, "width": 18})
            self.assertFalse(image_processor.do_normalize)

            image_processor = image_processing_class.from_dict(self.image_processor_dict, size=42)
            self.assertEqual(image_processor.size, {"height": 42, "width": 42})

    def test_post_process_depth_estimation(self):
        image_processor = Tipsv2DptImageProcessor()

        batch_size = 2
        height = width = 16
        outputs = DepthEstimatorOutput(predicted_depth=torch.randn(batch_size, height, width))

        # without target_sizes: spatial dims match predicted_depth
        result = image_processor.post_process_depth_estimation(outputs)
        self.assertEqual(len(result), batch_size)
        self.assertEqual(result[0]["predicted_depth"].shape, torch.Size([height, width]))

        # with target_sizes: output is resized to requested size
        target_sizes = [(height * 2, width * 2)] * batch_size
        result = image_processor.post_process_depth_estimation(outputs, target_sizes=target_sizes)
        self.assertEqual(len(result), batch_size)
        self.assertEqual(result[0]["predicted_depth"].shape, torch.Size([height * 2, width * 2]))

        # mismatched batch size raises ValueError
        with self.assertRaises(ValueError):
            image_processor.post_process_depth_estimation(outputs, target_sizes=[(100, 100)])

    def test_post_process_normal_estimation(self):
        image_processor = Tipsv2DptImageProcessor()

        batch_size = 2
        height = width = 16
        outputs = Tipsv2DptNormalEstimatorOutput(normals=torch.randn(batch_size, 3, height, width))

        # without target_sizes: spatial dims match normals, values are L2-normalized
        result = image_processor.post_process_normal_estimation(outputs)
        self.assertEqual(len(result), batch_size)
        self.assertEqual(result[0]["normals"].shape, torch.Size([3, height, width]))
        norms = result[0]["normals"].norm(p=2, dim=0)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4)

        # with target_sizes: output is resized and re-normalized
        target_sizes = [(height * 2, width * 2)] * batch_size
        result = image_processor.post_process_normal_estimation(outputs, target_sizes=target_sizes)
        self.assertEqual(len(result), batch_size)
        self.assertEqual(result[0]["normals"].shape, torch.Size([3, height * 2, width * 2]))
        norms = result[0]["normals"].norm(p=2, dim=0)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4)

        # mismatched batch size raises ValueError
        with self.assertRaises(ValueError):
            image_processor.post_process_normal_estimation(outputs, target_sizes=[(100, 100)])

    def test_post_process_semantic_segmentation(self):
        image_processor = Tipsv2DptImageProcessor()

        batch_size = 2
        num_labels = 3
        height = width = 16
        outputs = SemanticSegmenterOutput(logits=torch.randn(batch_size, num_labels, height, width))

        # without target_sizes: argmax at decoder resolution
        segmentation = image_processor.post_process_semantic_segmentation(outputs)
        self.assertEqual(len(segmentation), batch_size)
        self.assertEqual(segmentation[0].shape, torch.Size([height, width]))

        # with target_sizes: logits resized before argmax
        target_sizes = [(height * 2, width * 2)] * batch_size
        segmentation = image_processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        self.assertEqual(len(segmentation), batch_size)
        self.assertEqual(segmentation[0].shape, torch.Size([height * 2, width * 2]))

        # mismatched batch size raises ValueError
        with self.assertRaises(ValueError):
            image_processor.post_process_semantic_segmentation(outputs, target_sizes=[(100, 100)])
