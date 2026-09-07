# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Testing suite for the NemotronH_Omni processor.

The published checkpoint has no tiny counterpart on the Hub, so the tokenizer is built locally
from a word-level vocabulary carrying the placeholder tokens the processor expands.
"""

import unittest

from transformers.testing_utils import require_tokenizers, require_torch, require_vision
from transformers.utils import is_vision_available


if is_vision_available():
    from PIL import Image

    from transformers import (
        NemotronH_Omni_Reasoning_V3ImageProcessor,
        NemotronH_Omni_Reasoning_V3Processor,
    )


SPECIAL_TOKENS = ["<unk>", "<image>", "<video>", "<so_embedding>", "<so_start>", "<so_end>", "<img>", "</img>"]


@require_torch
@require_vision
@require_tokenizers
class NemotronH_Omni_Reasoning_V3ProcessorTest(unittest.TestCase):
    patch_size = 16

    def _tokenizer(self):
        from tokenizers import Tokenizer, models, pre_tokenizers

        from transformers import PreTrainedTokenizerFast

        vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS + ["describe", "this", "picture"])}
        backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
        tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS[1:]})
        return tokenizer

    def _processor(self):
        image_processor = NemotronH_Omni_Reasoning_V3ImageProcessor(
            norm_mean=[0.5, 0.5, 0.5],
            norm_std=[0.5, 0.5, 0.5],
            patch_size=self.patch_size,
            downsample_ratio=0.5,
            min_num_patches=4,
            max_num_patches=16,
            max_model_len=1024,
        )
        return NemotronH_Omni_Reasoning_V3Processor(image_processor=image_processor, tokenizer=self._tokenizer())

    def test_placeholder_token_ids_resolve(self):
        processor = self._processor()
        for token, token_id in (
            (processor.image_token, processor.image_token_id),
            (processor.video_token, processor.video_token_id),
            (processor.audio_token, processor.audio_token_id),
        ):
            self.assertEqual(processor.tokenizer.convert_tokens_to_ids(token), token_id)
            self.assertNotEqual(token_id, processor.tokenizer.unk_token_id)

    def test_text_only(self):
        processor = self._processor()
        out = processor(text="describe this picture")
        self.assertIn("input_ids", out)
        self.assertNotIn("pixel_values", out)

    def test_image_token_is_expanded(self):
        processor = self._processor()
        image = Image.new("RGB", (128, 96))
        out = processor(text="<image> describe this picture", images=image, return_tensors="pt")

        self.assertIn("pixel_values", out)
        input_ids = out["input_ids"][0]
        num_placeholders = int((input_ids == processor.image_token_id).sum())
        # a single `<image>` expands to one placeholder per post-pixel-shuffle token
        self.assertEqual(num_placeholders, int(out["num_tokens"][0]))
        self.assertGreater(num_placeholders, 1)

    def test_model_input_names_are_exposed(self):
        processor = self._processor()
        self.assertIn("input_ids", processor.model_input_names)
