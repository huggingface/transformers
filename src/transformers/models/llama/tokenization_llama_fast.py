# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils.versions import require_version


require_version("tokenizers>=0.13.3")


class LlamaTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.

    This uses notably ByteFallback and no normalization.

    ```
    from transformers import LlamaTokenizerFast

    tokenizer = LlaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    tokenizer.encode("Hello this is a test")
    >>> [1, 15043, 445, 338, 263, 1243]
    ```
    """

    def __init__(
        self,
        *args,
        clean_up_tokenization_spaces=False,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        **kwargs,
    ):
        super().__init__(
            *args,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )
