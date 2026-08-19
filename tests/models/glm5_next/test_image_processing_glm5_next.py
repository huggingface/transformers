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

import hashlib
import tempfile
import unittest

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available


if is_torch_available():
    import torch

if is_torchvision_available():
    from transformers import AutoImageProcessor, Glm5NextImageProcessor, Glm5NextImageProcessorPil
    from transformers.models.glm5_next.image_processing_glm5_next import smart_resize


@require_torch
@require_vision
class Glm5NextImageProcessingTest(unittest.TestCase):
    def get_processor_kwargs(self, **overrides):
        kwargs = {
            "patch_size": 2,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "patch_expand_factor": 2,
            "min_image_tokens": 1,
            "max_image_tokens": 64,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
        }
        return kwargs | overrides

    def test_smart_resize_reference_cases(self):
        cases = [
            ((2, 5, 9, 2, 8, 8, 32, 2048), (8, 16)),
            ((2, 5, 9, 2, 8, 8, 512, 2048), (16, 24)),
            ((2, 37, 53, 2, 8, 8, 32, 256), (8, 16)),
        ]
        for args, expected in cases:
            self.assertEqual(smart_resize(*args), expected)

    def test_bit_exact_reference_outputs(self):
        cases = [
            ("pad", 1, 64, 5, 9, [[1, 4, 8]], "26b79d03c22c0fe2ff0f4bb1ac88057dce6bf2ca70f194183685d131b0f5533a"),
            ("pad", 1, 8, 37, 53, [[1, 4, 8]], "52242a3eda2d9eec3c279369a9756d000066770dbdfdc53db6b4d6f64c6afda8"),
            ("resize", 1, 64, 5, 9, [[1, 4, 8]], "85df36be8e690726baf39dd61183eafc14d707da469d9f6760270b7590f21cbe"),
        ]
        for mode, min_tokens, max_tokens, height, width, grid, expected_hash in cases:
            processor = Glm5NextImageProcessor(
                **self.get_processor_kwargs(
                    resize_mode=mode,
                    min_image_tokens=min_tokens,
                    max_image_tokens=max_tokens,
                )
            )
            image = torch.arange(3 * height * width, dtype=torch.uint8).reshape(3, height, width)
            output = processor(image, return_tensors="pt")
            digest = hashlib.sha256(output.pixel_values.contiguous().numpy().tobytes()).hexdigest()
            self.assertEqual(output.image_grid_thw.tolist(), grid)
            self.assertEqual(digest, expected_hash)

    def test_invalid_resize_mode(self):
        processor = Glm5NextImageProcessor(resize_mode="invalid")
        with self.assertRaisesRegex(ValueError, "resize_mode"):
            processor(torch.zeros(3, 5, 9, dtype=torch.uint8))

    def test_torchvision_and_pil_padding_are_bit_exact(self):
        kwargs = self.get_processor_kwargs(do_rescale=False, do_normalize=False)
        image = torch.arange(3 * 5 * 9, dtype=torch.uint8).reshape(3, 5, 9)
        torchvision_output = Glm5NextImageProcessor(**kwargs)(image, return_tensors="pt")
        pil_output = Glm5NextImageProcessorPil(**kwargs)(image, return_tensors="pt")
        self.assertTrue(torch.equal(torchvision_output.pixel_values, pil_output.pixel_values))
        self.assertTrue(torch.equal(torchvision_output.image_grid_thw, pil_output.image_grid_thw))

    def test_per_call_resize_overrides(self):
        processor = Glm5NextImageProcessor(**self.get_processor_kwargs())
        image = torch.arange(3 * 37 * 53, dtype=torch.uint8).reshape(3, 37, 53)
        overrides = {"max_image_tokens": 8, "patch_expand_factor": 1, "resize_mode": "resize"}
        actual = processor(image, return_tensors="pt", **overrides)
        expected = Glm5NextImageProcessor(**self.get_processor_kwargs(**overrides))(image, return_tensors="pt")
        self.assertTrue(torch.equal(actual.pixel_values, expected.pixel_values))
        self.assertTrue(torch.equal(actual.image_grid_thw, expected.image_grid_thw))

    def test_auto_image_processor_round_trip(self):
        processor = Glm5NextImageProcessor(**self.get_processor_kwargs())
        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_pretrained(tmpdir)
            loaded = AutoImageProcessor.from_pretrained(tmpdir)
        self.assertIsInstance(loaded, Glm5NextImageProcessor)
        self.assertEqual(loaded.to_dict(), processor.to_dict())

    def test_new_processor_config_bit_exact(self):
        processor = Glm5NextImageProcessor(
            patch_size=14,
            temporal_patch_size=2,
            merge_size=2,
            patch_expand_factor=1,
            min_image_tokens=16,
            max_image_tokens=8000,
            resize_mode="pad",
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
        )
        image = torch.arange(3 * 37 * 53, dtype=torch.uint8).reshape(3, 37, 53)
        output = processor(image, return_tensors="pt")
        digest = hashlib.sha256(output.pixel_values.contiguous().numpy().tobytes()).hexdigest()
        self.assertEqual(output.image_grid_thw.tolist(), [[1, 8, 10]])
        self.assertEqual(digest, "b1c71008b109e4d6c753f70f95301ade14774a9909ff20f05ddece9ab188fccd")

    def test_get_number_of_image_patches(self):
        processor = Glm5NextImageProcessor(**self.get_processor_kwargs())
        output = processor(torch.zeros(3, 5, 9, dtype=torch.uint8), return_tensors="pt")
        self.assertEqual(processor.get_number_of_image_patches(5, 9), output.image_grid_thw.prod().item())


if __name__ == "__main__":
    unittest.main()
