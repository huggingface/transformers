import unittest

import torch

from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import UnlimitedOcrProcessor


@require_vision
class UnlimitedOcrProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = UnlimitedOcrProcessor
    # TODO: Change before merge
    model_id = "guarin/Unlimited-OCR"

    # Defaults from mixin are too small as a single image expands to 273 image tokens
    # for this checkpoint (size=1024)
    image_text_kwargs_max_length = 320
    image_text_kwargs_override_max_length = 310
    image_unstructured_max_length = 300

    def test_replace_image_tokens(self):
        processor = self.get_processor()

        images = torch.randint(0, 256, (1, 3, 200, 300), dtype=torch.uint8)
        prompt = "<image>document parsing."

        inputs = processor(images=images, text=prompt, return_tensors="pt")
        num_image_tokens = (inputs["input_ids"] == processor.image_token_id).sum().item()

        # image resized to 1024, followed by patch size 16 and 4x downsampling = 16 x 16 patches
        # 273 = 16 rows * (16 cols + 1 newline) + 1 view separator
        self.assertEqual(num_image_tokens, 273)
        self.assertNotIn("pixel_values_local", inputs)

    def test_replace_image_tokens_with_local(self):
        processor = self.get_processor()

        images = torch.randint(0, 256, (1, 3, 500, 700), dtype=torch.uint8)
        prompt = "<image>document parsing."

        inputs = processor(images=images, text=prompt, return_tensors="pt")
        num_image_tokens = (inputs["input_ids"] == processor.image_token_id).sum().item()

        # global is same as in test above
        # 500 x 700 image is split into 3x4 tiles
        # local tiles are 640, followed by patch size 16 and 4x downsample = 10 x 10 patches
        # 1503 = 273 global + (3 rows * 10) * (4 cols * 10 + 1)) local
        self.assertEqual(num_image_tokens, 1503)
        self.assertIn("pixel_values_local", inputs)

    def test_replace_image_tokens_no_crop(self):
        processor = self.get_processor()

        images = torch.randint(0, 256, (1, 3, 500, 700), dtype=torch.uint8)
        prompt = "<image>document parsing."

        inputs = processor(images=images, text=prompt, return_tensors="pt", crop_to_patches=False)
        num_image_tokens = (inputs["input_ids"] == processor.image_token_id).sum().item()

        # same as in test_replace_image_tokens
        self.assertEqual(num_image_tokens, 273)
        self.assertNotIn("pixel_values_local", inputs)
