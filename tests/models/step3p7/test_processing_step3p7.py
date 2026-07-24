# Copyright 2026 The StepFun and HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the Step3p7 processor."""

import unittest

from transformers import Step3p7Processor
from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch


@require_vision
@require_torch
@require_torchvision
class Step3p7ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Step3p7Processor
    # No `model_id`: the real checkpoint uses vendor `custom_code` (its own preprocessor_config.json
    # format), so `Step3p7Processor.from_pretrained(model_id)` can't parse it. Build every component
    # locally instead — only the tokenizer (which has all the special tokens this processor relies on:
    # <im_start>, <im_patch>, ..., <patch_newline>) is pulled from the real repo.

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        return tokenizer_class.from_pretrained("stepfun-ai/Step-3.7-Flash")

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        # `patch_size` close to `size` keeps the generic mixin tests' default (400x400-after-squaring)
        # test image down to a handful of placeholder tokens, well under every `*_max_length` used below.
        return image_processor_class(
            do_resize=True,
            size={"height": 64, "width": 64},
            patch_size=200,
            num_image_features=1,
            num_patch_features=1,
            do_rescale=True,
            rescale_factor=1 / 255,
            do_normalize=True,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
            do_convert_rgb=True,
        )

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token

    @unittest.skip(
        "The real stepfun-ai/Step-3.7-Flash chat template uses a `fromjson` Jinja filter for tool-call "
        "arguments, which `chat_template_utils.py`'s renderer doesn't register (only `tojson` is). Pre-existing "
        "gap unrelated to Step3p7-specific code."
    )
    def test_apply_chat_template_tool_calls_no_content(self):
        pass

    def test_unused_input_names_excludes_patch_newline_masks(self):
        processor = self.get_processor()
        self.assertIn("patch_newline_masks", processor.unused_input_names)

    def test_replace_image_token_expands_patches_and_global_view(self):
        processor = self.get_processor()
        # 250x200 (H x W): long_side=250 > image_size=64, ratio 1.25 <= 4 -> window_size = patch_size
        # (200). Snapped crop is 200x400 -> 1x2 = 2 patches (one newline row) plus the global view.
        image = torch.randint(0, 256, (3, 250, 200), dtype=torch.uint8)

        inputs = processor(text=self.image_token, images=[image], return_tensors="pt")
        decoded = processor.tokenizer.decode(inputs["input_ids"][0])

        self.assertIn("<im_start>", decoded)
        self.assertIn("<im_end>", decoded)
        self.assertIn("<patch_start>", decoded)
        self.assertIn("<patch_end>", decoded)
        self.assertIn("<patch_newline>", decoded)
        # patch placeholder block(s) must come before the global-view placeholder block
        self.assertLess(decoded.index("<patch_start>"), decoded.index("<im_start>"))

    def test_replace_image_token_no_patches_for_small_image(self):
        processor = self.get_processor()
        # A small square image (48x48) fits the global view; no local patches, no patch tokens.
        image = torch.randint(0, 256, (3, 48, 48), dtype=torch.uint8)

        inputs = processor(text=self.image_token, images=[image], return_tensors="pt")
        decoded = processor.tokenizer.decode(inputs["input_ids"][0])

        self.assertNotIn("<patch_start>", decoded)
        self.assertNotIn("<patch_end>", decoded)
        self.assertIn("<im_start>", decoded)
        self.assertIn("<im_end>", decoded)
