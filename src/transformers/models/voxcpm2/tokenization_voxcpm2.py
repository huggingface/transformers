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

from tokenizers import Regex, pre_tokenizers

from ...tokenization_utils_base import _get_prepend_scheme
from ..llama.tokenization_llama import LlamaTokenizer


class VoxCPM2Tokenizer(LlamaTokenizer):
    """Llama tokenizer that keeps Han characters as individual tokens."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        metaspace = pre_tokenizers.Metaspace(
            replacement="▁",
            prepend_scheme=_get_prepend_scheme(self.add_prefix_space, self),
            split=False,
        )
        han_characters = pre_tokenizers.Split(Regex(r"\p{Han}"), behavior="isolated")
        self.backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([metaspace, han_characters])


__all__ = ["VoxCPM2Tokenizer"]
