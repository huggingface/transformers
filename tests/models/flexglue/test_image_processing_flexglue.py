# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from tests.models.superglue.test_image_processing_superglue import (
    SuperGlueImageProcessingTest,
    SuperGlueImageProcessingTester,
)
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

    from transformers.models.flexglue.modeling_flexglue import FlexGlueKeypointMatchingOutput


class FlexGlueImageProcessingTester(SuperGlueImageProcessingTester):
    """Tester for FlexGlueImageProcessor"""

    def prepare_image_processor_dict(self):
        return {"size": self.size, "do_normalize": True}

    def prepare_keypoint_matching_output(self, pixel_values):
        output = super().prepare_keypoint_matching_output(pixel_values)
        return FlexGlueKeypointMatchingOutput(
            mask=output.mask,
            keypoints=output.keypoints,
            matches=output.matches,
            matching_scores=output.matching_scores,
        )


@require_torch
@require_vision
class FlexGlueImageProcessingTest(SuperGlueImageProcessingTest, unittest.TestCase):
    image_processor_tester_class = FlexGlueImageProcessingTester

    def test_image_processing(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "do_rescale"))
            self.assertTrue(hasattr(image_processing, "rescale_factor"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertFalse(hasattr(image_processing, "do_grayscale"))

    @unittest.skip(reason="FlexGlue takes RGB images, it does not convert them to grayscale")
    def test_input_image_properly_converted_to_grayscale(self):
        pass

    def test_input_image_is_normalized(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            image = torch.full((3, 64, 64), 255, dtype=torch.uint8)
            pixel_values = image_processor([image, image], return_tensors="pt")["pixel_values"]
            expected = (1 - torch.tensor(image_processor.image_mean)) / torch.tensor(image_processor.image_std)
            torch.testing.assert_close(pixel_values[0, 0, :, 0, 0], expected, atol=1e-4, rtol=1e-4)
