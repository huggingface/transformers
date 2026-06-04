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

from transformers import is_tokenizers_available, is_torch_available
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.testing_utils import require_tokenizers, require_torch


if is_torch_available():
    import torch

if is_tokenizers_available():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Sequence, Split, WhitespaceSplit

    from transformers import PreTrainedTokenizerFast, Zaya1VLProcessor


class DummyZaya1VLImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "image_grid_thw"]
    merge_size = 2

    def __call__(self, images=None, **kwargs):
        return BatchFeature(
            {
                "pixel_values": torch.zeros(4, 3),
                "image_grid_thw": torch.tensor([[1, 4, 4]]),
            }
        )

    def get_number_of_image_patches(self, height, width, images_kwargs):
        return height * width


def get_tokenizer():
    tokenizer = Tokenizer(WordLevel({"<pad>": 0, "<unk>": 1, "<image>": 2, "hello": 3}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Sequence([Split("<image>", behavior="isolated"), WhitespaceSplit()])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>")
    tokenizer.image_token = "<image>"
    tokenizer.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    return tokenizer


@require_torch
@require_tokenizers
class Zaya1VLProcessorTest(unittest.TestCase):
    def get_processor(self):
        return Zaya1VLProcessor(DummyZaya1VLImageProcessor(), get_tokenizer())

    def test_image_token_expansion_without_default_token_type_ids(self):
        inputs = self.get_processor()(text="<image> hello", images=[object()], return_tensors="pt")

        self.assertEqual(inputs.input_ids.tolist(), [[2, 2, 2, 2, 3]])
        self.assertEqual(inputs.pixel_values.shape, (4, 3))
        self.assertNotIn("mm_token_type_ids", inputs)

    def test_return_mm_token_type_ids(self):
        inputs = self.get_processor()(
            text="<image> hello",
            images=[object()],
            return_mm_token_type_ids=True,
            return_tensors="pt",
        )

        self.assertEqual(inputs.mm_token_type_ids.tolist(), [[1, 1, 1, 1, 0]])

    def test_get_num_multimodal_tokens(self):
        output = self.get_processor()._get_num_multimodal_tokens(image_sizes=[(4, 4), (8, 4)])

        self.assertEqual(output["num_image_patches"], [16, 32])
        self.assertEqual(output["num_image_tokens"], [4, 8])

    def test_model_input_names_excludes_default_mm_token_type_ids(self):
        self.assertEqual(
            self.get_processor().model_input_names,
            ["pixel_values", "image_grid_thw", "input_ids", "attention_mask"],
        )


if __name__ == "__main__":
    unittest.main()
