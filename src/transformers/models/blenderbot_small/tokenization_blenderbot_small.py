# coding=utf-8
# Copyright 2021, The Facebook, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Fast tokenization class for BlenderbotSmall."""

from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_config_file": "tokenizer_config.json",
}


class BlenderbotSmallTokenizer(TokenizersBackend):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        bos_token="__start__",
        eos_token="__end__",
        unk_token="__unk__",
        pad_token="__null__",
        add_prefix_space=False,
        vocab=None,
        merges=None,
        trim_offsets=True,
        **kwargs,
    ):
        self._tokenizer = Tokenizer(
            BPE(
                {token: idx for idx, (token, _score) in enumerate(vocab)} if isinstance(vocab, list) else vocab,
                merges if merges is not None else generate_merges(filtered_vocab),
                continuing_subword_prefix="",
                end_of_word_suffix="",
            )
        )
        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)
        self._tokenizer.decoder = decoders.ByteLevel()
        self._tokenizer.post_processor = processors.ByteLevel(trim_offsets=trim_offsets)

        super().__init__(
            tokenizer_object=self._tokenizer,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets,
            **kwargs,
        )


__all__ = ["BlenderbotSmallTokenizer"]
