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
"""
Tokenization classes for fast tokenizers (provided by HuggingFace's tokenizers library). For slow (python) tokenizers
see tokenization_utils.py
"""

import copy
import json
import os
from collections import defaultdict
from collections.abc import Iterable
from shutil import copyfile
from typing import Any

import tokenizers.pre_tokenizers as pre_tokenizers_fast
from huggingface_hub import is_offline_mode
from tokenizers import AddedToken, processors
from tokenizers import Encoding as EncodingFast
from tokenizers import Tokenizer as TokenizerFast
from tokenizers.decoders import Decoder as DecoderFast
from tokenizers.models import BPE, Unigram
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer

from transformers.utils.hub import cached_file

from .integrations.ggml import convert_gguf_tokenizer
from .modeling_gguf_pytorch_utils import load_gguf_checkpoint
from .tokenization_utils_base import (
    INIT_TOKENIZER_DOCSTRING,
    BatchEncoding,
    PreTokenizedInput,
    PreTrainedTokenizerBase,
    TextInput,
    TruncationStrategy,
    generate_merges,
)
from .utils import PaddingStrategy, add_end_docstrings, logging


logger = logging.get_logger(__name__)

# Fast tokenizers (provided by HuggingFace tokenizer's library) can be saved in a single file
TOKENIZER_FILE = "tokenizer.json"
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
TIKTOKEN_VOCAB_FILE = "tokenizer.model"

# Slow tokenizers have an additional added tokens files
ADDED_TOKENS_FILE = "added_tokens.json"

INIT_TOKENIZER_DOCSTRING += """
        tokenizer_object ([`tokenizers.Tokenizer`]):
            A [`tokenizers.Tokenizer`] object from 🤗 tokenizers to instantiate from. See [Using tokenizers from 🤗
            tokenizers](../fast_tokenizers) for more information.
        tokenizer_file ([`str`]):
            A path to a local JSON file representing a previously serialized [`tokenizers.Tokenizer`] object from 🤗
            tokenizers.
"""

MODEL_TO_TRAINER_MAPPING = {
    "BPE": BpeTrainer,
    "Unigram": UnigramTrainer,
    "WordLevel": WordLevelTrainer,
    "WordPiece": WordPieceTrainer,
}

VOCAB_FILES_NAMES = {"tokenizer_file": TOKENIZER_FILE, "vocab_file": TIKTOKEN_VOCAB_FILE}


@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class TokenizersBackend(PreTrainedTokenizerBase):
    """
    Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

    Inherits from [`~tokenization_utils_base.PreTrainedTokenizerBase`].

    Handles all the shared methods for tokenization and special tokens, as well as methods for
    downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.

    This class also contains the added tokens in a unified way on top of all tokenizers so we don't have to handle the
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model = None
    _tokenizer = None

    @classmethod
    def convert_to_native_format(cls, trust_remote_code=False, **kwargs):
        """s
        Build a `tokenizers.Tokenizer` backend from the available serialization files (tokenizer.json, sentencepiece
        models, tekken.json, vocab/merges).
        """
        # Preserve kwargs for possible downstream use
        local_kwargs = dict(kwargs)
        fast_tokenizer_file = local_kwargs.pop("tokenizer_file", None)

        if (
            fast_tokenizer_file is not None
            and os.path.isfile(fast_tokenizer_file)
            and (cls is TokenizersBackend or "__init__" not in cls.__dict__ or trust_remote_code)
        ):
            local_kwargs["tokenizer_object"] = TokenizerFast.from_file(fast_tokenizer_file)
            return local_kwargs
        elif fast_tokenizer_file is not None and os.path.isfile(fast_tokenizer_file):
            # we extract vocab / merges from the tokenizer file to pass them to __init__
            processor = TokenizerFast.from_file(fast_tokenizer_file).post_processor
            with open(fast_tokenizer_file, encoding="utf-8") as tokenizer_handle:
                tokenizer_json = json.load(tokenizer_handle)
            vocab = tokenizer_json.get("model", {}).get("vocab", None)
            if cls.model is None:
                if isinstance(vocab, list):
                    vocab = list(map(tuple, vocab))  # TODO just for now
            elif cls.model.__name__ == "Unigram":
                if vocab and isinstance(vocab[0], (list, tuple)):
                    vocab = [tuple(item) for item in vocab]
            elif cls.model.__name__ == "WordLevel":
                vocab = {token: i for i, token in enumerate(vocab)}
            elif cls.model.__name__ == "BPE" or cls.model.__name__ == "WordPiece":
                if isinstance(vocab, list):
                    vocab = {token[0] if isinstance(token, list) else token: i for i, token in enumerate(vocab)}
            local_kwargs["vocab"] = vocab

            model_type = getattr(cls, "model", None)
            if "merges" in tokenizer_json.get("model", {}) and (model_type and model_type.__name__ == "BPE"):
                merges = tokenizer_json["model"]["merges"]
                merges = [tuple(merge.split(" ")) if isinstance(merge, str) else tuple(merge) for merge in merges]
                local_kwargs["merges"] = merges

            if processor is not None:
                local_kwargs["post_processor"] = processor
            return local_kwargs

        vocab_file = local_kwargs.get("vocab_file")
        merges_file = local_kwargs.get("merges_file")
        vocab = local_kwargs.get("vocab")
        merges = local_kwargs.get("merges")

        # Tekken converter (Mistral)
        if isinstance(vocab_file, str) and vocab_file.endswith("tekken.json") and os.path.isfile(vocab_file):
            from .convert_slow_tokenizer import MistralConverter

            local_kwargs["vocab"], local_kwargs["merges"] = MistralConverter(
                vocab_file=vocab_file
            ).extract_vocab_merges_from_model(vocab_file)
            return local_kwargs

        # SentencePiece model (with TikToken fallback)
        if isinstance(vocab_file, str) and os.path.isfile(vocab_file) and vocab_file.endswith(".model"):
            try:
                from .convert_slow_tokenizer import SentencePieceExtractor

                local_kwargs = SentencePieceExtractor(vocab_file).extract(cls.model, **local_kwargs)
                try:
                    from .convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS

                    converter_class = SLOW_TO_FAST_CONVERTERS.get(cls.__name__)
                    if converter_class is not None and hasattr(converter_class, "convert_from_spm"):
                        local_kwargs = converter_class.convert_from_spm(**local_kwargs)
                except Exception as e:
                    logger.warning(
                        f"Could not reorder vocab using converter for {cls.__name__} due to {e}. Falling back to raw SentencePiece extraction."
                    )
                # what used to be in `convert_slow`
                if hasattr(cls, "convert_from_spm_model"):
                    local_kwargs = cls.convert_from_spm_model(**local_kwargs)
            except Exception as e:  # TODO only catch deserialization error here!
                logger.warning(
                    f"Could not extract SentencePiece model from {vocab_file} using sentencepiece library due to {e}. "
                    "Falling back to TikToken extractor."
                )
                from .convert_slow_tokenizer import TikTokenConverter

                local_kwargs["vocab"], local_kwargs["merges"] = TikTokenConverter(
                    vocab_file=vocab_file, extra_special_tokens=local_kwargs.get("extra_special_tokens")
                ).extract_vocab_merges_from_model(vocab_file)

            return local_kwargs

        # Fallback to standard vocab/merges files if they existed!
        if vocab is None and isinstance(vocab_file, str) and os.path.isfile(vocab_file):
            local_kwargs["vocab"] = vocab_file
            vocab = local_kwargs["vocab"]
        if merges is None and isinstance(merges_file, str) and os.path.isfile(merges_file):
            local_kwargs["merges"] = merges_file
            merges = local_kwargs["merges"]

        # Generate merges automatically when not provided for BPE tokenizers
        if merges is None and cls.model is not None and cls.model.__name__ == "BPE" and isinstance(vocab, dict):
            # Gather special tokens from kwargs to skip in merge generation
            def _iter_special_tokens(values: Iterable[Any]) -> list[str]:
                collected: list[str] = []
                for val in values:
                    if val is None:
                        continue
                    if isinstance(val, (list, tuple)):
                        collected.extend(_iter_special_tokens(val))
                    else:
                        collected.append(str(val))
                return collected

            special_tokens_keys = [
                "pad_token",
                "unk_token",
                "bos_token",
                "eos_token",
                "sep_token",
                "cls_token",
                "mask_token",
                "additional_special_tokens",
                "extra_special_tokens",
            ]
            skip_tokens: set[str] = set()
            for key in special_tokens_keys:
                if key in local_kwargs:
                    skip_tokens.update(_iter_special_tokens([local_kwargs[key]]))

            merges = generate_merges(vocab, skip_tokens=skip_tokens)
            local_kwargs["merges"] = merges
        return local_kwargs

    def __init__(self, *args, **kwargs):
        tokenizer_object = kwargs.pop("tokenizer_object", None)
        gguf_file = kwargs.pop("gguf_file", None)
        fast_tokenizer_file = kwargs.pop("tokenizer_file", None)
        # Note: added_tokens_decoder is NOT popped - it's passed to super().__init__() for processing
        added_tokens_decoder = kwargs.get("added_tokens_decoder", {})
        # Store add_prefix_space before super().__init__() to ensure it's not overridden
        add_prefix_space = kwargs.get("add_prefix_space", False)
        vocab_file = kwargs.get("vocab_file")

        vocab = kwargs.get("vocab")
        merges = kwargs.get("merges")

        fast_tokenizer = None
        if tokenizer_object is not None:
            fast_tokenizer = copy.deepcopy(tokenizer_object)
        elif fast_tokenizer_file is not None and os.path.isfile(fast_tokenizer_file):
            # We have a serialization from tokenizers which let us directly build the backend
            fast_tokenizer = TokenizerFast.from_file(fast_tokenizer_file)
        elif gguf_file is not None:
            # We need to convert a slow tokenizer to build the backend
            gguf_path = cached_file(kwargs.get("name_or_path", ""), gguf_file, **kwargs)
            gguf_param = load_gguf_checkpoint(gguf_path)
            architecture = gguf_param["config"]["model_type"]
            tokenizer_dict = gguf_param["tokenizer"]
            tokenizer_config = gguf_param["tokenizer_config"]
            fast_tokenizer, additional_kwargs = convert_gguf_tokenizer(architecture, tokenizer_dict)
            kwargs.update(tokenizer_config)
            if len(additional_kwargs) > 0:
                kwargs.update(additional_kwargs)
        elif self._tokenizer is None and vocab is not None:
            # Build from vocab/merges extracted by convert_to_native_format
            if merges is not None:
                vocab_dict = vocab if isinstance(vocab, dict) else {w: i for i, (w, _) in enumerate(vocab)}
                fast_tokenizer = TokenizerFast(BPE(vocab=vocab_dict, merges=merges, fuse_unk=True, dropout=None))
            elif isinstance(vocab, dict):
                fast_tokenizer = TokenizerFast(BPE(vocab=vocab, merges=[], fuse_unk=True, dropout=None))
            elif isinstance(vocab, list) and vocab and isinstance(vocab[0], (tuple, list)):
                fast_tokenizer = TokenizerFast(Unigram(vocab=vocab, unk_id=kwargs.get("unk_id", 0)))
        elif self._tokenizer is None:
            raise ValueError(
                "Couldn't instantiate the backend tokenizer from one of: \n"
                "(1) a `tokenizers` library serialization file, \n"
                "(2) a slow tokenizer instance to convert or \n"
                "(3) an equivalent slow tokenizer class to instantiate and convert. \n"
                "You need to have sentencepiece or tiktoken installed to convert a slow tokenizer to a fast one."
            )
        # Only set defaults when creating TokenizersBackend from scratch
        if fast_tokenizer_file is None and tokenizer_object is None and self._tokenizer is None:
            kwargs.setdefault("bos_token", "<s>")
            kwargs.setdefault("eos_token", "</s>")

        if fast_tokenizer is not None:
            self._tokenizer = fast_tokenizer

        if self._tokenizer is None:
            raise ValueError("The backend tokenizer is not correctly initialized.")

        _truncation = self._tokenizer.truncation

        if _truncation is not None:
            self._tokenizer.enable_truncation(**_truncation)
            kwargs.setdefault("max_length", _truncation["max_length"])
            kwargs.setdefault("truncation_side", _truncation["direction"])
            kwargs.setdefault("stride", _truncation["stride"])
            kwargs.setdefault("truncation_strategy", _truncation["strategy"])
        else:
            self._tokenizer.no_truncation()

        _padding = self._tokenizer.padding
        if _padding is not None:
            self._tokenizer.enable_padding(**_padding)
            kwargs.setdefault("pad_token", _padding["pad_token"])
            kwargs.setdefault("pad_token_type_id", _padding["pad_type_id"])
            kwargs.setdefault("padding_side", _padding["direction"])
            kwargs.setdefault("max_length", _padding["length"])
            kwargs.setdefault("pad_to_multiple_of", _padding["pad_to_multiple_of"])

        # Set backend to "tokenizers" if not already set
        if "backend" not in kwargs:
            kwargs["backend"] = "tokenizers"

        explicit_bos_eos_in_kwargs = "add_bos_token" in kwargs or "add_eos_token" in kwargs
        self._add_bos_token = kwargs.get("add_bos_token", False)
        self._add_eos_token = kwargs.get("add_eos_token", False)
        if post_processor := kwargs.pop("post_processor", None):  # most reliable way to get the post-processor
            self._tokenizer.post_processor = post_processor
        self._should_update_post_processor = explicit_bos_eos_in_kwargs or self._tokenizer.post_processor is None
        # We call this after having initialized the backend tokenizer because we update it.
        super().__init__(**kwargs)

        if vocab_file is not None:
            self.vocab_file = vocab_file
        # Ensure add_prefix_space is set correctly after parent init
        self.add_prefix_space = add_prefix_space
        self._tokenizer.encode_special_tokens = self.split_special_tokens

        added_tokens_decoder_hash = {hash(repr(token)) for token in self.added_tokens_decoder}
        tokens_to_add = [
            token
            for index, token in sorted(added_tokens_decoder.items(), key=lambda x: x[0])
            if hash(repr(token)) not in added_tokens_decoder_hash
        ]
        encoder = list(self.added_tokens_encoder.keys()) + [str(token) for token in tokens_to_add]
        # if some of the special tokens are not already in the tokenizer, add them
        # V5: Check both named special tokens and extra special tokens
        # Iterate over _special_tokens_map to preserve AddedToken properties (lstrip, rstrip, etc.)
        for special_token_value in self._special_tokens_map.values():
            if special_token_value is None:
                continue
            if str(special_token_value) not in encoder and special_token_value not in tokens_to_add:
                tokens_to_add.append(special_token_value)

        # Also check extra special tokens
        for token in self._extra_special_tokens:
            if str(token) not in encoder and token not in tokens_to_add:
                tokens_to_add.append(token)

        if len(tokens_to_add) > 0:
            tokens = []
            all_named_tokens = [str(t) for t in self._special_tokens_map.values() if t]
            for token in tokens_to_add:
                if isinstance(token, str):
                    # Convert string to AddedToken, assuming it's special
                    token = AddedToken(token, special=True)
                elif isinstance(token, AddedToken):
                    # Ensure the special flag is set correctly for special tokens
                    if not token.special and str(token) in all_named_tokens:
                        token.special = True
                tokens.append(token)
            if tokens:
                # These tokens are from the special tokens map
                self.add_tokens(tokens)

        try:
            vocab_size = self._tokenizer.get_vocab_size()
        except NotImplementedError:
            vocab_size = 0

        # Optionally patches mistral tokenizers with wrong regex
        if vocab_size > 100000 and getattr(self._tokenizer, "pre_tokenizer", None) is not None:
            kwargs.pop("tokenizer", None)
            self._tokenizer = self._patch_mistral_regex(
                self._tokenizer,
                self.init_kwargs.get("name_or_path", None),
                init_kwargs=self.init_kwargs,
                fix_mistral_regex=kwargs.get("fix_mistral_regex"),
                **kwargs,
            )

        self._should_update_post_processor = (
            self._should_update_post_processor or self._tokenizer.post_processor is None
        )
        if self._should_update_post_processor:
            self.update_post_processor()

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def can_save_slow_tokenizer(self) -> bool:
        """
        `bool`: Whether or not the slow tokenizer can be saved. For a sentencepiece based slow tokenizer, this
        can only be `True` if the original `"sentencepiece.model"` was not deleted.
        """
        if "vocab_file" in self.vocab_files_names and self.vocab_files_names["vocab_file"].endswith(".model"):
            if hasattr(self, "vocab_file") and self.vocab_file:
                # If the vocab file is a sentencepiece model, we can save it
                return os.path.isfile(self.vocab_file)
            return False
        else:
            return True

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

    def update_post_processor(self):
        """
        Updates the underlying post processor with the current `bos_token` and `eos_token`.
        """
        bos = self.bos_token
        bos_token_id = self.bos_token_id
        if bos is None and self.add_bos_token:
            self.add_bos_token = False

        eos = self.eos_token
        eos_token_id = self.eos_token_id
        if eos is None and self.add_eos_token:
            self.add_eos_token = False

        single = f"{(bos + ':0 ') if self.add_bos_token else ''}$A:0{(' ' + eos + ':0') if self.add_eos_token else ''}"
        pair = f"{single}{(' ' + bos + ':1') if self.add_bos_token else ''} $B:1{(' ' + eos + ':1') if self.add_eos_token else ''}"

        special_tokens = []
        if self.add_bos_token:
            special_tokens.append((bos, bos_token_id))
        if self.add_eos_token:
            special_tokens.append((eos, eos_token_id))
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=single, pair=pair, special_tokens=special_tokens
        )

    @property
    def add_eos_token(self):
        return getattr(self, "_add_eos_token", False)

    @property
    def add_bos_token(self):
        return getattr(self, "_add_bos_token", False)

    @add_eos_token.setter
    def add_eos_token(self, value):
        object.__setattr__(self, "_add_eos_token", value)
        self.update_post_processor()

    @add_bos_token.setter
    def add_bos_token(self, value):
        object.__setattr__(self, "_add_bos_token", value)
        self.update_post_processor()

    def _post_init(self):
        """
        Post-initialization hook that runs after the tokenizer is fully set up.
        This is called by from_pretrained() after loading the tokenizer, which allows
        us to add any special tokens that may have been passed as AddedToken objects.

        Child classes should call super()._post_init() if they override this method.
        """
        tokens_to_add = []
        # V5: Check named special tokens
        for token_value in self._special_tokens_map.values():
            if token_value is None:
                continue
            if isinstance(token_value, AddedToken):
                tokens_to_add.append(token_value)
            elif isinstance(token_value, str):
                tokens_to_add.append(AddedToken(token_value, special=True, normalized=False))

        # V5: Check extra special tokens
        for token in self._extra_special_tokens:
            if isinstance(token, AddedToken):
                tokens_to_add.append(token)
            elif isinstance(token, str):
                tokens_to_add.append(AddedToken(token, special=True, normalized=False))

        if tokens_to_add:
            # Ensure special tokens are added as such to the backend
            self.add_tokens(tokens_to_add, special_tokens=True)

        if getattr(self, "_should_update_post_processor", True) or self._tokenizer.post_processor is None:
            self.update_post_processor()

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    def get_vocab(self) -> dict[str, int]:
        return self._tokenizer.get_vocab(with_added_tokens=True)

    @property
    def vocab(self) -> dict[str, int]:
        return self.get_vocab()

    @property
    def added_tokens_encoder(self) -> dict[str, int]:
        """
        Returns the sorted mapping from string to index. The added tokens encoder is cached for performance
        optimisation in `self._added_tokens_encoder` for the slow tokenizers.
        """
        return {k.content: v for v, k in sorted(self.added_tokens_decoder.items(), key=lambda item: item[0])}

    @property
    def added_tokens_decoder(self) -> dict[int, AddedToken]:
        """
        Returns the added tokens in the vocabulary as a dictionary of index to AddedToken.

        Returns:
            `dict[str, int]`: The added tokens.
        """
        return self._tokenizer.get_added_tokens_decoder()

    # BC v5: expose ``_added_tokens_encoder`` / ``_added_tokens_decoder`` attrs for custom tokenizers that expect
    # them from slow tokenizers. Only supports read, not write (won't sync to Rust backend, use add_tokens() instead
    _added_tokens_encoder = added_tokens_encoder
    _added_tokens_decoder = added_tokens_decoder

    def get_added_vocab(self) -> dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index.

        Returns:
            `dict[str, int]`: The added tokens.
        """
        return {k.content: v for v, k in sorted(self.added_tokens_decoder.items(), key=lambda item: item[0])}

    def __bool__(self) -> bool:
        """
        Returns True, to avoid expensive `assert tokenizer` gotchas.
        """
        return True

    def __len__(self) -> int:
        """
        Size of the full vocabulary with the added tokens.
        """
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def backend_tokenizer(self) -> TokenizerFast:
        """
        `tokenizers.implementations.BaseTokenizer`: The Rust tokenizer used as a backend.
        """
        return self._tokenizer

    @property
    def decoder(self) -> DecoderFast:
        """
        `tokenizers.decoders.Decoder`: The Rust decoder for this tokenizer.
        """
        return self._tokenizer.decoder

    def _convert_encoding(
        self,
        encoding: EncodingFast,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> tuple[dict[str, Any], list[EncodingFast]]:
        """
        Convert the encoding representation (from low-level HuggingFace tokenizer output) to a python Dict and a list
        of encodings, take care of building a batch from overflowing tokens.

        Overflowing tokens are converted to additional examples (like batches) so the output values of the dict are
        lists (overflows) of lists (tokens).

        Output shape: (overflows, sequence length)
        """
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        if return_overflowing_tokens and encoding.overflowing is not None:
            encodings = [encoding] + encoding.overflowing
        else:
            encodings = [encoding]

        encoding_dict = defaultdict(list)
        for e in encodings:
            encoding_dict["input_ids"].append(e.ids)

            if return_token_type_ids:
                encoding_dict["token_type_ids"].append(e.type_ids)
            if return_attention_mask:
                encoding_dict["attention_mask"].append(e.attention_mask)
            if return_special_tokens_mask:
                encoding_dict["special_tokens_mask"].append(e.special_tokens_mask)
            if return_offsets_mapping:
                encoding_dict["offset_mapping"].append(e.offsets)
            if return_length:
                encoding_dict["length"].append(len(e.ids))

        return encoding_dict, encodings

    def _convert_token_to_id_with_added_voc(self, token: str) -> int:
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index

    def _convert_id_to_token(self, index: int) -> str | None:
        return self._tokenizer.id_to_token(int(index))

    def _add_tokens(self, new_tokens: list[str | AddedToken], special_tokens=False) -> int:
        if special_tokens:
            return self._tokenizer.add_special_tokens(new_tokens)

        return self._tokenizer.add_tokens(new_tokens)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        <Tip>

        This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
        this inside your training loop.

        </Tip>

        Args:
            pair (`bool`, *optional*, defaults to `False`):
                Whether the number of added tokens should be computed in the case of a sequence pair or a single
                sequence.

        Returns:
            `int`: Number of special tokens added to sequences.
        """
        return self._tokenizer.num_special_tokens_to_add(pair)

    def convert_ids_to_tokens(self, ids: int | list[int], skip_special_tokens: bool = False) -> str | list[str]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `list[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `list[str]`: The decoded token(s).
        """
        if isinstance(ids, int):
            return self._tokenizer.id_to_token(ids)
        tokens = []
        # self.all_special_ids is an @property which may be slow, so only compute it once before the loop
        ids_to_skip = set(self.all_special_ids) if skip_special_tokens else set()
        for index in ids:
            index = int(index)
            if index in ids_to_skip:
                continue
            tokens.append(self._tokenizer.id_to_token(index))
        return tokens

    def tokenize(self, text: str, pair: str | None = None, add_special_tokens: bool = False, **kwargs) -> list[str]:
        return self._encode_plus(text=text, text_pair=pair, add_special_tokens=add_special_tokens, **kwargs).tokens()

    def set_truncation_and_padding(
        self,
        padding_strategy: PaddingStrategy,
        truncation_strategy: TruncationStrategy,
        max_length: int,
        stride: int,
        pad_to_multiple_of: int | None,
        padding_side: str | None,
    ):
        """
        Define the truncation and the padding strategies for fast tokenizers (provided by HuggingFace tokenizers
        library) and restore the tokenizer settings afterwards.

        The provided tokenizer has no padding / truncation strategy before the managed section. If your tokenizer set a
        padding / truncation strategy before, then it will be reset to no padding / truncation when exiting the managed
        section.

        Args:
            padding_strategy ([`~utils.PaddingStrategy`]):
                The kind of padding that will be applied to the input
            truncation_strategy ([`~tokenization_utils_base.TruncationStrategy`]):
                The kind of truncation that will be applied to the input
            max_length (`int`):
                The maximum size of a sequence.
            stride (`int`):
                The stride to use when handling overflow.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
            padding_side (`str`, *optional*):
                The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                Default value is picked from the class attribute of the same name.
        """
        _truncation = self._tokenizer.truncation
        _padding = self._tokenizer.padding
        # Set truncation and padding on the backend tokenizer
        if truncation_strategy == TruncationStrategy.DO_NOT_TRUNCATE:
            if _truncation is not None:
                self._tokenizer.no_truncation()
        else:
            target = {
                "max_length": max_length,
                "stride": stride,
                "strategy": truncation_strategy.value,
                "direction": self.truncation_side,
            }

            # _truncation might contain more keys that the target `transformers`
            # supports. Use only the target keys to trigger `enable_truncation`.
            # This should enable this code to works on various `tokenizers`
            # targets.
            if _truncation is None:
                current = None
            else:
                current = {k: _truncation.get(k, None) for k in target}

            if current != target:
                self._tokenizer.enable_truncation(**target)

        if padding_strategy == PaddingStrategy.DO_NOT_PAD:
            if _padding is not None:
                self._tokenizer.no_padding()
        else:
            length = max_length if padding_strategy == PaddingStrategy.MAX_LENGTH else None
            target = {
                "length": length,
                "direction": padding_side if padding_side is not None else self.padding_side,
                "pad_id": self.pad_token_id,
                "pad_token": self.pad_token,
                "pad_type_id": self.pad_token_type_id,
                "pad_to_multiple_of": pad_to_multiple_of,
            }
            if _padding != target:
                self._tokenizer.enable_padding(**target)

    def _encode_plus(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        text_pair: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: int | None = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: int | None = None,
        padding_side: str | None = None,
        return_tensors: bool | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        split_special_tokens: bool | None = None,
        **kwargs,
    ) -> BatchEncoding:
        # Input validation (from _call_one)
        def _is_valid_text_input(t):
            if isinstance(t, str):
                return True
            elif isinstance(t, (list, tuple)):
                if len(t) == 0:
                    return True
                elif isinstance(t[0], str):
                    return True
                elif isinstance(t[0], (list, tuple)):
                    if len(t[0]) == 0 or isinstance(t[0][0], str):
                        return True
                    elif isinstance(t[0][0], (list, tuple)):
                        return len(t[0][0]) == 0 or isinstance(t[0][0][0], str)
                    else:
                        return False
                else:
                    return False
            else:
                return False

        if not _is_valid_text_input(text):
            raise ValueError(
                "text input must be of type `str` (single example), `list[str]` (batch or single pretokenized example) "
                "or `list[list[str]]` (batch of pretokenized examples) or `list[tuple[list[str], list[str]]]` (batch of pretokenized sequence pairs)."
            )

        if text_pair is not None and not _is_valid_text_input(text_pair):
            raise ValueError(
                "text input must be of type `str` (single example), `list[str]` (batch or single pretokenized example) "
                "or `list[list[str]]` (batch of pretokenized examples) or `list[tuple[list[str], list[str]]]` (batch of pretokenized sequence pairs)."
            )

        # Batch detection (from _call_one)
        if is_split_into_words:
            is_batched = isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))
        else:
            is_batched = isinstance(text, (list, tuple))

        if is_batched:
            # Batch validation
            if isinstance(text_pair, str):
                raise TypeError(
                    "when tokenizing batches of text, `text_pair` must be a list or tuple with the same length as"
                    " `text`."
                )
            if text_pair is not None and len(text) != len(text_pair):
                raise ValueError(
                    f"batch length of `text`: {len(text)} does not match batch length of `text_pair`:"
                    f" {len(text_pair)}."
                )
            batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
        else:
            # Single input - convert to batch format
            batch_text_or_text_pairs = [(text, text_pair)] if text_pair else [text]

        # Set tokenizer configuration (from _batch_encode_plus)
        if not isinstance(batch_text_or_text_pairs, (tuple, list)):
            raise TypeError(
                f"batch_text_or_text_pairs has to be a list or a tuple (got {type(batch_text_or_text_pairs)})"
            )

        self.set_truncation_and_padding(
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
        )

        # Use self.split_special_tokens as default if not explicitly provided
        if split_special_tokens is None:
            split_special_tokens = self.split_special_tokens

        if self._tokenizer.encode_special_tokens != split_special_tokens:
            self._tokenizer.encode_special_tokens = split_special_tokens

        # Direct rust backend call
        encodings = self._tokenizer.encode_batch(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            is_pretokenized=is_split_into_words,
        )

        # Convert encodings to BatchEncoding format
        tokens_and_encodings = [
            self._convert_encoding(
                encoding=encoding,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
            )
            for encoding in encodings
        ]

        # Convert the output to have dict[list] from list[dict]
        sanitized_tokens = {}
        for key in tokens_and_encodings[0][0]:
            stack = [e for item, _ in tokens_and_encodings for e in item[key]]
            sanitized_tokens[key] = stack
        sanitized_encodings = [e for _, item in tokens_and_encodings for e in item]

        # If returning overflowing tokens, we need to return a mapping
        if return_overflowing_tokens:
            overflow_to_sample_mapping = []
            for i, (toks, _) in enumerate(tokens_and_encodings):
                overflow_to_sample_mapping += [i] * len(toks["input_ids"])
            sanitized_tokens["overflow_to_sample_mapping"] = overflow_to_sample_mapping

        for input_ids in sanitized_tokens["input_ids"]:
            self._eventual_warn_about_too_long_sequence(input_ids, max_length, verbose)

        batched_output = BatchEncoding(sanitized_tokens, sanitized_encodings, tensor_type=return_tensors)

        # If single input, remove the batch dimension (unless returning overflowing tokens)
        if not is_batched and return_tensors is None and not return_overflowing_tokens:
            batched_output = BatchEncoding(
                {
                    key: (value[0] if len(value) > 0 and isinstance(value[0], list) else value)
                    for key, value in batched_output.items()
                },
                batched_output.encodings,
            )

        return batched_output

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return (
            self.backend_tokenizer.decoder.decode(tokens)
            if self.backend_tokenizer.decoder is not None
            else " ".join(tokens)
        )

    def _decode(
        self,
        token_ids: int | list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs,
    ) -> str:
        # Removed: use_source_tokenizer parameter (unused)
        kwargs.pop("use_source_tokenizer", None)  # Pop if present to avoid errors

        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if isinstance(token_ids, dict):
            token_ids = token_ids["input_ids"]
        text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            text = self.clean_up_tokenization(text)

        return text

    def _save_pretrained(
        self,
        save_directory: str | os.PathLike,
        file_names: tuple[str, ...],
        legacy_format: bool | None = None,
        filename_prefix: str | None = None,
    ) -> tuple[str, ...]:
        save_directory = str(save_directory)

        tokenizer_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_FILE
        )
        self.backend_tokenizer.save(tokenizer_file)
        file_names = file_names + (tokenizer_file,)

        return file_names

    def train_new_from_iterator(
        self,
        text_iterator,
        vocab_size,
        length=None,
        new_special_tokens=None,
        special_tokens_map=None,
        **kwargs,
    ):
        """
        Trains a tokenizer on a new corpus with the same defaults (in terms of special tokens or tokenization pipeline)
        as the current one.

        Args:
            text_iterator (generator of `list[str]`):
                The training corpus. Should be a generator of batches of texts, for instance a list of lists of texts
                if you have everything in memory.
            vocab_size (`int`):
                The size of the vocabulary you want for your tokenizer.
            length (`int`, *optional*):
                The total number of sequences in the iterator. This is used to provide meaningful progress tracking
            new_special_tokens (list of `str` or `AddedToken`, *optional*):
                A list of new special tokens to add to the tokenizer you are training.
            special_tokens_map (`dict[str, str]`, *optional*):
                If you want to rename some of the special tokens this tokenizer uses, pass along a mapping old special
                token name to new special token name in this argument.
            kwargs (`dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the trainer from the 🤗 Tokenizers library.

        Returns:
            [`PreTrainedTokenizerFast`]: A new tokenizer of the same type as the original one, trained on
            `text_iterator`.

        """
        tokenizer_json = json.loads(self._tokenizer.to_str())
        # Remove added tokens for now (uses IDs of tokens)
        added_tokens = tokenizer_json.pop("added_tokens")
        # Remove post processor for now (uses IDs of tokens)
        post_processor = tokenizer_json.pop("post_processor")

        unk_token = None
        # Remove vocab
        if tokenizer_json["model"]["type"] == "BPE":
            tokenizer_json["model"]["vocab"] = {}
            tokenizer_json["model"]["merges"] = []
        elif tokenizer_json["model"]["type"] == "Unigram":
            if tokenizer_json["model"]["unk_id"] is not None:
                unk_id = tokenizer_json["model"]["unk_id"]
                unk_token = tokenizer_json["model"]["vocab"][unk_id][0]
                if special_tokens_map is not None and unk_token in special_tokens_map:
                    unk_token = special_tokens_map[unk_token]
                tokenizer_json["model"]["unk_id"] = 0
                tokenizer_json["model"]["vocab"] = [[unk_token, 0.0]]
        elif tokenizer_json["model"]["type"] in ["WordLevel", "WordPiece"]:
            tokenizer_json["model"]["vocab"] = {}
        else:
            raise ValueError(
                f"This method does not support this type of tokenizer (found {tokenizer_json['model']['type']}) "
                "only BPE, Unigram, WordLevel and WordPiece."
            )

        if (
            special_tokens_map is not None
            and "unk_token" in tokenizer_json["model"]
            and tokenizer_json["model"]["unk_token"] in special_tokens_map
        ):
            tokenizer_json["model"]["unk_token"] = special_tokens_map[tokenizer_json["model"]["unk_token"]]

        tokenizer = TokenizerFast.from_str(json.dumps(tokenizer_json))

        # Get the special tokens from the current tokenizer if none are specified.
        special_tokens = []
        for added_token in added_tokens:
            special = added_token.pop("special", None)
            _ = added_token.pop("id", None)
            if tokenizer_json["model"]["type"] != "Unigram" and not special:
                continue
            if special_tokens_map is not None and added_token["content"] in special_tokens_map:
                added_token["content"] = special_tokens_map[added_token["content"]]
            special_tokens.append(AddedToken(**added_token))

        if new_special_tokens is not None:
            special_tokens.extend(new_special_tokens)

        # Trainer needs to know the end of word / continuing subword thingies in BPE
        if (
            tokenizer_json["model"]["type"] == "BPE"
            and "continuing_subword_prefix" not in kwargs
            and tokenizer_json["model"]["continuing_subword_prefix"] is not None
        ):
            kwargs["continuing_subword_prefix"] = tokenizer_json["model"]["continuing_subword_prefix"]
        if (
            tokenizer_json["model"]["type"] == "BPE"
            and "end_of_word_suffix" not in kwargs
            and tokenizer_json["model"]["end_of_word_suffix"] is not None
        ):
            kwargs["end_of_word_suffix"] = tokenizer_json["model"]["end_of_word_suffix"]
        if tokenizer_json["model"]["type"] == "Unigram" and unk_token is not None:
            kwargs["unk_token"] = unk_token
        if tokenizer_json["pre_tokenizer"] is not None:
            if (
                tokenizer_json["pre_tokenizer"]["type"] == "ByteLevel"
                or tokenizer_json["pre_tokenizer"]["type"] == "Sequence"
                and "pretokenizers" in tokenizer_json["pre_tokenizer"]
                and any(
                    pretokenizer["type"] == "ByteLevel"
                    for pretokenizer in tokenizer_json["pre_tokenizer"]["pretokenizers"]
                )
            ):
                kwargs["initial_alphabet"] = pre_tokenizers_fast.ByteLevel.alphabet()

        trainer_class = MODEL_TO_TRAINER_MAPPING[tokenizer_json["model"]["type"]]
        trainer = trainer_class(vocab_size=vocab_size, special_tokens=special_tokens, **kwargs)
        tokenizer.train_from_iterator(text_iterator, length=length, trainer=trainer)

        if post_processor is not None:
            trained_tokenizer_json = json.loads(tokenizer.to_str())
            # Almost done, we just have to adjust the token IDs in the post processor
            if "special_tokens" in post_processor:
                for key in post_processor["special_tokens"]:
                    tokens = post_processor["special_tokens"][key]["tokens"]
                    if special_tokens_map is not None:
                        tokens = [special_tokens_map.get(token, token) for token in tokens]
                    post_processor["special_tokens"][key]["tokens"] = tokens
                    for token in tokens:
                        token_id = tokenizer.token_to_id(token)
                        if token_id is None:
                            raise ValueError(
                                "Attempted to set a token in the post processor that does not exist in the mapping"
                            )

                    post_processor["special_tokens"][key]["ids"] = [tokenizer.token_to_id(token) for token in tokens]

            for special_token in ["cls", "sep"]:
                if special_token in post_processor:
                    token, _ = post_processor[special_token]
                    if special_tokens_map is not None and token in special_tokens_map:
                        token = special_tokens_map[token]
                    token_id = tokenizer.token_to_id(token)
                    if token_id is None:
                        raise ValueError(
                            "Attempted to set a token in the post processor that does not exist in the mapping"
                        )
                    post_processor[special_token] = [token, token_id]

            trained_tokenizer_json["post_processor"] = post_processor
            tokenizer = TokenizerFast.from_str(json.dumps(trained_tokenizer_json))

        kwargs = self.init_kwargs.copy()
        # V5: Map pad/cls/mask token at the Transformers level (named tokens only)
        for token in PreTrainedTokenizerBase.SPECIAL_TOKENS_ATTRIBUTES:
            if getattr(self, token) is not None:
                special_token = getattr(self, token)
                if special_tokens_map is not None and special_token in special_tokens_map:
                    special_token = special_tokens_map[special_token]

                special_token_full = self._special_tokens_map.get(token, None)
                if isinstance(special_token_full, AddedToken):
                    # Create an added token with the same parameters except the content
                    kwargs[token] = AddedToken(
                        special_token,
                        single_word=special_token_full.single_word,
                        lstrip=special_token_full.lstrip,
                        rstrip=special_token_full.rstrip,
                        normalized=special_token_full.normalized,
                        special=True,
                    )
                else:
                    kwargs[token] = special_token

        # V5: Handle extra special tokens
        extra_special_tokens = self.extra_special_tokens.copy() if self.extra_special_tokens else []
        if new_special_tokens is not None:
            extra_special_tokens.extend(new_special_tokens)
        if len(extra_special_tokens) > 0:
            kwargs["extra_special_tokens"] = extra_special_tokens

        # Always try to pass tokenizer_object in kwargs first (standard TokenizersBackend usage)
        # If the class creates its own tokenizer and passes it explicitly to super().__init__(),
        # this will cause a TypeError, which we catch and handle by removing tokenizer_object
        # from kwargs and setting _tokenizer directly after initialization.
        kwargs["tokenizer_object"] = tokenizer
        try:
            return self.__class__(**kwargs)
        except TypeError as e:
            # Check if the error is due to multiple values for tokenizer_object
            if "multiple values for keyword argument 'tokenizer_object'" in str(e):
                # Class creates its own tokenizer and passes it explicitly (like LayoutLMv3Tokenizer)
                # Remove tokenizer_object from kwargs and set _tokenizer directly
                kwargs.pop("tokenizer_object", None)
                new_tokenizer = self.__class__(**kwargs)
                new_tokenizer._tokenizer = tokenizer
                return new_tokenizer
            else:
                # Some other TypeError, re-raise it
                raise

    @classmethod
    def _patch_mistral_regex(
        cls,
        tokenizer,
        pretrained_model_name_or_path,
        token=None,
        cache_dir=None,
        local_files_only=False,
        _commit_hash=None,
        is_local=False,
        init_kwargs=None,
        fix_mistral_regex=None,
        **kwargs,
    ):
        """
        Patches mistral related tokenizers with incorrect regex if detected
            1) Local file with an associated config saved next to it
                >> Model type one of the mistral models (on older versions)
            2) Remote models on the hub from official mistral models
                >> Tags including `base_model:.*mistralai`
        """
        import re

        from huggingface_hub import model_info
        from packaging import version

        from transformers.utils.hub import cached_file

        def is_base_mistral(model_id: str) -> bool:
            model = model_info(model_id)
            if model.tags is not None:
                if re.search("base_model:.*mistralai", "".join(model.tags)):
                    return True
            return False

        if is_offline_mode():
            is_local = True

        if pretrained_model_name_or_path is not None and (
            is_local or (not is_local and is_base_mistral(pretrained_model_name_or_path))
        ):
            _config_file = cached_file(
                pretrained_model_name_or_path,
                "config.json",
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
                _commit_hash=_commit_hash,
            )

            # Detected using a (local) mistral tokenizer
            mistral_config_detected = False
            if _config_file is not None:
                with open(_config_file, encoding="utf-8") as f:
                    _config = json.load(f)
                transformers_version = _config.get("transformers_version")
                transformers_model_type = _config.get("model_type")

                # Detect if we can skip the mistral fix by
                #   a) having a non-mistral tokenizer
                #   b) fixed version of transformers
                if transformers_version and version.parse(transformers_version) <= version.parse("4.57.2"):
                    if (
                        is_local
                        and transformers_model_type is not None
                        and transformers_model_type
                        not in [
                            "mistral",
                            "mistral3",
                            "voxtral",
                            "ministral",
                            "pixtral",
                        ]
                    ):
                        return tokenizer
                elif transformers_version and version.parse(transformers_version) > version.parse("4.57.3"):
                    return tokenizer

                mistral_config_detected = True

            if mistral_config_detected or (not is_local and is_base_mistral(pretrained_model_name_or_path)):
                # Expose the `fix_mistral_regex` flag on the tokenizer when provided, even if no correction is applied.
                if init_kwargs and "fix_mistral_regex" in init_kwargs:
                    setattr(tokenizer, "fix_mistral_regex", init_kwargs["fix_mistral_regex"])

                # only warn if its not explicitly passed
                if fix_mistral_regex is None and not getattr(tokenizer, "fix_mistral_regex", False):
                    setattr(tokenizer, "fix_mistral_regex", False)
                    logger.warning(
                        f"The tokenizer you are loading from '{pretrained_model_name_or_path}'"
                        f" with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e."
                        " This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue."
                    )
                elif fix_mistral_regex is True or getattr(tokenizer, "fix_mistral_regex", False):
                    setattr(tokenizer, "fix_mistral_regex", True)
                    import tokenizers

                    split_pretokenizer = tokenizers.pre_tokenizers.Split(
                        pattern=tokenizers.Regex(
                            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
                        ),
                        behavior="isolated",
                    )
                    current_pretokenizer = tokenizer.backend_tokenizer.pre_tokenizer
                    # Check if it's already a Sequence
                    if isinstance(current_pretokenizer, tokenizers.pre_tokenizers.Sequence):
                        # Replace the first element (the Split pattern)
                        tokenizer.backend_tokenizer.pre_tokenizer[0] = split_pretokenizer
                    else:
                        # Replace Metaspace with ByteLevel when adding Split, as Metaspace(split=False) doesn't
                        # work correctly with the Split pre-tokenizer and causes spaces to be lost during encoding
                        if isinstance(current_pretokenizer, tokenizers.pre_tokenizers.Metaspace):
                            current_pretokenizer = tokenizers.pre_tokenizers.ByteLevel(
                                add_prefix_space=False, use_regex=False
                            )

                        # Not a Sequence, so create one with Split + current pretokenizer
                        tokenizer.backend_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
                            [
                                split_pretokenizer,
                                current_pretokenizer,
                            ]
                        )

        return tokenizer


# Backward-compatible alias: allow referring to TokenizersBackend as PreTrainedTokenizerFast
PreTrainedTokenizerFast = TokenizersBackend
