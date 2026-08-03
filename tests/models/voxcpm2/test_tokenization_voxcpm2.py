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

from transformers.models.llama.tokenization_llama import LlamaTokenizer
from transformers.models.voxcpm2.tokenization_voxcpm2 import VoxCPM2Tokenizer
from transformers.testing_utils import require_torch


def get_tiny_voxcpm2_tokenizer() -> VoxCPM2Tokenizer:
    vocab = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "<pad>": 3,
        "▁": 4,
        "你": 5,
        "好": 6,
        "你好": 7,
        "▁你好": 8,
        "A": 9,
        "B": 10,
        "AB": 11,
        "▁AB": 12,
    }
    merges = [("你", "好"), ("▁", "你好"), ("A", "B"), ("▁", "AB")]
    return VoxCPM2Tokenizer(vocab=vocab, merges=merges, pad_token="<pad>")


def test_chinese_characters_are_split_before_bpe_merges():
    tokenizer = get_tiny_voxcpm2_tokenizer()
    base_tokenizer = LlamaTokenizer(
        vocab=tokenizer._vocab,
        merges=tokenizer._merges,
        pad_token="<pad>",
        add_prefix_space=True,
    )

    assert base_tokenizer("你好", add_special_tokens=False).input_ids == [8]
    assert tokenizer("你好", add_special_tokens=False).input_ids == [4, 5, 6]
    assert tokenizer.tokenize("你好") == ["▁", "你", "好"]
    assert (
        tokenizer("AB", add_special_tokens=False).input_ids == base_tokenizer("AB", add_special_tokens=False).input_ids
    )


@require_torch
def test_chinese_splitting_preserves_batch_padding_and_tensor_conversion():
    tokenizer = get_tiny_voxcpm2_tokenizer()
    expected_input_ids = [[4, 5, 6], [3, 3, 12]]
    expected_attention_mask = [[1, 1, 1], [0, 0, 1]]

    list_batch = tokenizer(["你好", "AB"], add_special_tokens=False, padding=True)
    assert list_batch.input_ids == expected_input_ids
    assert list_batch.attention_mask == expected_attention_mask

    for tensor_type in ("np", "pt"):
        tensor_batch = tokenizer(
            ["你好", "AB"],
            add_special_tokens=False,
            padding=True,
            return_tensors=tensor_type,
        )
        assert tensor_batch.input_ids.tolist() == expected_input_ids
        assert tensor_batch.attention_mask.tolist() == expected_attention_mask
