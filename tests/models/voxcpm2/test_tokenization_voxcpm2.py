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

from transformers.models.voxcpm2.tokenization_voxcpm2 import VoxCPM2Tokenizer


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
