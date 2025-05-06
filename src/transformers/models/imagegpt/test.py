# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.

import unittest
import numpy as np
import torch

from transformers import ImageGPTImageProcessor, ImageGPTImageProcessorFast
from transformers.testing_utils import require_torch, require_vision, slow
from transformers.image_processing_utils import BatchFeature
from transformers.test_image_processing_common import ImageProcessorTesterMixin


@require_torch
@require_vision
class ImageGPTImageProcessingTest(ImageProcessorTesterMixin, unittest.TestCase):
    image_processor_class = ImageGPTImageProcessor
    image_processor_fast_class = ImageGPTImageProcessorFast
    test_resize = True
    test_normalize = True
    test_color_quant = True

    def get_image_processor_tester(self, **kwargs):
        from transformers import ImageGPTImageProcessor

        clusters = np.random.randint(0, 255, (512, 3), dtype=np.uint8)
        return ImageGPTImageProcessor(
            clusters=clusters,
            do_resize=True,
            size={"height": 32, "width": 32},
            do_normalize=True,
            do_color_quantize=True,
            **kwargs,
        )

    def prepare_inputs(self):
        images = [np.random.randint(0, 256, (3, 64, 64), dtype=np.uint8) for _ in range(2)]
        return {"images": images}

    def test_fast_and_slow_equal(self):
        inputs = self.prepare_inputs()
        slow_processor = self.get_image_processor_tester()
        fast_processor = ImageGPTImageProcessorFast(clusters=slow_processor.clusters)

        slow_output = slow_processor(**inputs, return_tensors="pt")
        fast_output = fast_processor.preprocess(torch.tensor(inputs["images"], dtype=torch.float32), return_tensors="pt")

        self.assertIn("input_ids", fast_output)
        self.assertEqual(slow_output["input_ids"].shape, fast_output["input_ids"].shape)
