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
import os
from shutil import copyfile
from typing import Optional

from tokenizers import processors
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE, Unigram

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging, requires_backends
from ...create_fast_tokenizer import _get_prepend_scheme, generate_merges


logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model", "tokenizer_file": "tokenizer.json"}

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# fmt: off
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""
# fmt: on


class LlamaTokenizer(PreTrainedTokenizerFast):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.

    This uses notably ByteFallback and no normalization.

    ```python
    >>> from transformers import LlamaTokenizer

    >>> tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    >>> tokenizer.encode("Hello this is a test")
    [1, 15043, 445, 338, 263, 1243]
    ```

    If you want to change the `bos_token` or the `eos_token`, make sure to specify them when initializing the model, or
    call `tokenizer.update_post_processor()` to make sure that the post-processing is correctly done (otherwise the
    values of the first token and final token of an encoded sequence will not be correct). For more details, checkout
    [post-processors] (https://huggingface.co/docs/tokenizers/api/post-processors) documentation.


    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .model extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        tokenizer_file (`str`, *optional*):
            [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add an `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of sequences.
        use_default_system_prompt (`bool`, *optional*, defaults to `False`):
            Whether or not the default system prompt for Llama should be used
        legacy (`bool`, *optional*):
            Whether or not the `legacy` behavior of the tokenizer should be used. Legacy is before the merge of #24622
            and #25224 which includes fixes to properly handle tokens that appear after special tokens.
            Make sure to also set `from_slow` to `True`.
            A simple example:

            - `legacy=True`:
            ```python
            >>> from transformers import LlamaTokenizer

            >>> tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b", legacy=True, from_scratch=True)
            >>> tokenizer.encode("Hello <s>.") # 869 is '▁.'
            [1, 15043, 29871, 1, 869]
            ```
            - `legacy=False`:
            ```python
            >>> from transformers import LlamaTokenizer

            >>> tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b", legacy=False, from_scratch=True)
            >>> tokenizer.encode("Hello <s>.")  # 29889 is '.'
            [1, 15043, 29871, 1, 29889]
            ```
            Checkout the [pull request](https://github.com/huggingface/transformers/pull/24565) for more details.
        add_prefix_space (`bool`, *optional*):
            Whether or not the tokenizer should automatically add a prefix space
        from_scratch (`bool`, *optional*, defaults to `False`):
            Whether to create an empty trainable tokenizer from scratch. When `True`, creates a minimal tokenizer
            with only basic special tokens that can be trained on new data.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = None  # No slow tokenizer class needed
    padding_side = "left"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        clean_up_tokenization_spaces=False,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        add_bos_token=True,
        add_eos_token=False,
        use_default_system_prompt=False,
        legacy=False,
        add_prefix_space=None,
        vocab=None,
        merges=None,
        **kwargs,
    ):
        self.legacy = legacy
        
        # Set add_prefix_space attribute for use in override methods
        self.add_prefix_space = add_prefix_space if add_prefix_space is not None else True

        self._vocab = vocab if vocab is not None else self._vocab()
        self._merges = merges if merges is not None else generate_merges(self._vocab)

        # Prepare base-class construction helpers
        metaspace_override = None
        tokenizer_backend_config = None
        if tokenizer_file is None:
            tokenizer_backend_config = {
                "type": "spm",
                "handle_byte_fallback": True,
                "legacy": legacy,
                "add_prefix_space": add_prefix_space if add_prefix_space is not None else True,
                "vocab": self._vocab,
                "normalizer": self._normalizer,
                "pre_tokenizer": self._pre_tokenizer,
                "decoder": self._decoder,
                "tokenizer": self._tokenizer,
            }

        # Initialize the base class which will build the backend tokenizer
        super().__init__(
            tokenizer_file=tokenizer_file,
            tokenizer_backend_config=tokenizer_backend_config,
            metaspace_override=metaspace_override,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            use_default_system_prompt=use_default_system_prompt,
            add_prefix_space=add_prefix_space,
            legacy=legacy,
            **kwargs,
        )

        # TODO: how to do this cleanly? Need to trigger re-adding special tokens after setting the normalizer in Tokenizers
        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="first", split=False)
        self._tokenizer.normalizer = None #normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(pattern=" ", content="▁")])
        self.add_tokens([AddedToken(token, special=True) for token in self.all_special_tokens])

        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token
        self.update_post_processor()
        
        self.use_default_system_prompt = use_default_system_prompt
        self.vocab_file = vocab_file
        

    def _tokenizer(self):
        """Tokenizer configuration for this tokenizer."""
        return Tokenizer(BPE(vocab=self._vocab, merges=self._merges, fuse_unk=True, byte_fallback=True, dropout=None))

    def _vocab(self):
        """Vocabulary handling for this tokenizer."""
        vocab = {
            "<unk>": 0,
            "<s>": 1,
            "</s>": 2,
        }
        return vocab

    def _decoder(self, replacement, add_prefix_space):
        """Decoder configuration for this tokenizer."""
        sequence = [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
        ]
        if add_prefix_space:
            sequence += [decoders.Strip(content=" ", left=1)]
        return decoders.Sequence(sequence)

    def _normalizer(self):
        """Normalizer configuration for this tokenizer."""
        if self.legacy:
            sequence = []
            if self.add_prefix_space:
                sequence += [normalizers.Prepend(prepend="▁")]
            sequence += [normalizers.Replace(pattern=" ", content="▁")]
            return normalizers.Sequence(sequence)
        return None

    def _pre_tokenizer(self, replacement, add_prefix_space):
        """Pre-tokenizer configuration for this tokenizer."""
        if not self.legacy:
            prepend_scheme = _get_prepend_scheme(add_prefix_space, self)
            return pre_tokenizers.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme, split=False)
        return None


__all__ = ["LlamaTokenizer"]
