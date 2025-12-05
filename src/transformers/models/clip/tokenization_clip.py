# coding=utf-8
# Copyright 2021 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for CLIP."""

from typing import Optional

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}


class CLIPTokenizer(TokenizersBackend):
    """
    Construct a CLIP tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|startoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
        vocab (`dict`, *optional*):
            Vocabulary dict to use for the tokenizer.
        merges (`list`, *optional*):
            Merges list to use for the BPE tokenizer.
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None

    def __init__(
        self,
        unk_token: str = "<|endoftext|>",
        bos_token: str = "<|startoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|endoftext|>",
        vocab: Optional[dict] = None,
        merges: Optional[list] = None,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        **kwargs,
    ):
        self.vocab_file = vocab_file
        self.merges_file = merges_file

        if vocab is not None:
            _vocab = {token: idx for idx, (token, _score) in enumerate(vocab)} if isinstance(vocab, list) else vocab
        else:
            _vocab = {
                str(bos_token): 0,
                str(eos_token): 1,
                str(pad_token): 2,
            }

        if merges is not None:
            _merges = [tuple(merge) if isinstance(merge, list) else merge for merge in merges]
        else:
            _merges = []

        self._tokenizer = Tokenizer(
            BPE(
                vocab=_vocab,
                merges=_merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="</w>",
                fuse_unk=False,
                unk_token=str(unk_token),
            )
        )

        self._tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFC(), normalizers.Replace(Regex(r"\s+"), " "), normalizers.Lowercase()]
        )

        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    Regex(
                        r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""
                    ),
                    behavior="removed",
                    invert=True,
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False),
            ]
        )

        self._tokenizer.decoder = decoders.ByteLevel()

        bos_token_id = _vocab.get(str(bos_token), 0)
        eos_token_id = _vocab.get(str(eos_token), 1)

        self._tokenizer.post_processor = processors.RobertaProcessing(
            sep=(str(eos_token), eos_token_id),
            cls=(str(bos_token), bos_token_id),
            add_prefix_space=False,
            trim_offsets=False,
        )

        tokenizer_object = self._tokenizer

        super().__init__(
            tokenizer_object=tokenizer_object,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

        if hasattr(self, "_post_init"):
            self._post_init()

    def _post_init(self):
        super()._post_init()
        self._wrap_decode_method_backend_tokenizer()

    # Very ugly hack to enable padding to have a correct decoding see https://github.com/huggingface/tokenizers/issues/872
    def _wrap_decode_method_backend_tokenizer(self):
        orig_decode_method = self.backend_tokenizer.decode

        ## define this as a local variable to avoid circular reference
        ## See: https://github.com/huggingface/transformers/issues/30930
        end_of_word_suffix = self.backend_tokenizer.model.end_of_word_suffix

        def new_decode_method(*args, **kwargs):
            text = orig_decode_method(*args, **kwargs)
            text = text.replace(end_of_word_suffix, " ").strip()
            return text

        self.backend_tokenizer.decode = new_decode_method


__all__ = ["CLIPTokenizer"]
