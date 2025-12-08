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
SentencePiece-based tokenization class for loading from sentencepiece.model files.
"""

import os
from shutil import copyfile
from typing import Optional, Union


try:
    import sentencepiece as spm
except ImportError:
    spm = None

from .convert_slow_tokenizer import import_protobuf
from .tokenization_python import PreTrainedTokenizer
from .tokenization_utils_base import (
    INIT_TOKENIZER_DOCSTRING,
    AddedToken,
    generate_merges,
)
from .utils import add_end_docstrings, logging, requires_backends


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

SPIECE_UNDERLINE = "▁"


@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class SentencePieceBackend(PreTrainedTokenizer):
    """
    Base class for SentencePiece-based tokenizers that load from sentencepiece.model files.

    Inherits from [`~tokenization_utils.PreTrainedTokenizer`].

    Handle all the shared methods for tokenization and special tokens as well as methods downloading/caching/loading
    pretrained tokenizers as well as adding tokens to the vocabulary.

    This class also contain the added tokens in a unified way on top of all tokenizers so we don't have to handle the
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).
    """

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(self, **kwargs):
        # Ensure optional dependency is available before loading
        requires_backends(self, "sentencepiece")

        # Extract sentencepiece-specific parameters
        self.vocab_file = kwargs.get("vocab_file")
        self.legacy = kwargs.get("legacy", True)
        self.sp_model_kwargs = kwargs.pop("sp_model_kwargs", {})

        # Set backend to "sentencepiece" if not already set
        if "backend" not in kwargs:
            kwargs["backend"] = "sentencepiece"

        # Load the SentencePiece model before calling parent __init__
        # This is needed because parent __init__ may call methods that depend on sp_model
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        tokenizer.Load(self.vocab_file)

        if not self.legacy:
            model_pb2 = import_protobuf()
            proto = model_pb2.ModelProto.FromString(tokenizer.serialized_model_proto())
            if proto.normalizer_spec.add_dummy_prefix:
                proto.normalizer_spec.add_dummy_prefix = False
                tokenizer.LoadFromSerializedProto(proto.SerializeToString())

        self.sp_model = tokenizer

        # Initialize total_vocab_size before parent __init__ (which may call _add_tokens -> len(self))
        self.total_vocab_size = self.sp_model.get_piece_size()

        # Add sp_model_kwargs back to kwargs so it gets stored in init_kwargs
        kwargs["sp_model_kwargs"] = self.sp_model_kwargs

        # Call parent class __init__ (PreTrainedTokenizer)
        # This handles tokens_trie, _added_tokens_decoder, _added_tokens_encoder,
        # token_type_ids_pattern, special_tokens_pattern, and adds special tokens
        super().__init__(**kwargs)
        self._update_trie()

    @property
    def vocab_size(self) -> int:
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _add_tokens(self, new_tokens: Union[list[str], list[AddedToken]], special_tokens: bool = False) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary. Special tokens are sometimes already in the
        vocab which is why they have to be handled specifically.

        Args:
            new_tokens (`list[str]`or `list[tokenizers.AddedToken]`):
                Token(s) to add in vocabulary. A token is counted as added if it's not already in the vocabulary
                (tested by checking if the tokenizer assign the index of the `unk_token` to them). If a token is part
                of the vocabulary then we simply mark this token as an `AddedToken` which allows to control the
                stripping and normalization of this token. This is NOT possible in `tokenizers`.
            special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the tokens should be added as special tokens.

        Returns:
            `int`: The number of tokens actually added to the vocabulary.

        Examples:

        ```python
        # Let's see how to increase the vocabulary of Bert model and tokenizer
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        model = BertModel.from_pretrained("google-bert/bert-base-uncased")

        num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
        print("We have added", num_added_toks, "tokens")
        # Note: resize_token_embeddings expects to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))
        ```"""
        if not new_tokens:
            return 0

        next_index = len(self)  # total size (base + added)
        num_added = 0
        for token in new_tokens:
            if not isinstance(token, (str, AddedToken)):
                raise TypeError(f"Token {token} is not a string but a {type(token)}.")
            if str(token) == "":
                continue
            if isinstance(token, str):
                if token in self._added_tokens_encoder:
                    continue
                is_special = token in self.all_special_tokens or special_tokens
                token = AddedToken(token, rstrip=False, lstrip=False, normalized=not is_special, special=is_special)
            elif special_tokens:
                # doing token.special=True changes the normalization! will fix in rust
                # this is important and the only reason why the AddedTokens in each class are normalized by default
                token.__setstate__({"special": True, "normalized": token.normalized})

            if token in self._added_tokens_decoder.values():
                continue
            if not token.special and token.normalized and getattr(self, "do_lower_case", False):
                token.content = token.content.lower()

            # Check if token already exists in the SentencePiece base vocab
            tok_id = self.sp_model.piece_to_id(token.content)
            in_base_vocab = (
                tok_id < self.sp_model.get_piece_size() and self.sp_model.IdToPiece(tok_id) == token.content
            )

            if in_base_vocab:
                token_index = tok_id
            else:
                token_index = next_index
                next_index += 1
                num_added += 1

            if token.special and str(token) not in self.all_special_tokens:
                self._extra_special_tokens.append(token)
            # the setter automatically updates the reverse map
            self._added_tokens_decoder[token_index] = token
            self._added_tokens_encoder[token.content] = token_index
            if self.verbose:
                logger.info(f"Adding {token} to the vocabulary")

        self._update_trie()
        self._update_total_vocab_size()
        return num_added

    def _update_trie(self, unique_no_split_tokens: Optional[list[str]] = None):
        # Add all added tokens
        for token in self._added_tokens_decoder.values():
            if token.content not in self.tokens_trie._tokens:
                self.tokens_trie.add(token.content)
        # Also add all special tokens (even if they're in base vocab) so they get split during tokenization
        for token in self.all_special_tokens:
            if token not in self.tokens_trie._tokens:
                self.tokens_trie.add(token)
        # Add any additional no-split tokens
        for token in unique_no_split_tokens or []:
            if token not in self.tokens_trie._tokens:
                self.tokens_trie.add(token)

    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE. For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give
        `['H', 'e', 'y']` instead of `['▁He', 'y']`. Thus we always encode `f"{unk_token}text"` and strip the
        `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """
        if self.legacy or not text.startswith((SPIECE_UNDERLINE, " ")):
            return self.sp_model.encode(text, out_type=str)

        # 1. Encode string + prefix ex: "<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. Remove self.unk_token from ['<','unk','>', '▁Hey']
        unk_token_length = len(self.sp_model.encode(str(self.unk_token)))
        return tokens[unk_token_length:] if len(tokens) >= unk_token_length else tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) to an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Converts a sequence of tokens (string) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        """
        Save the sentencepiece vocabulary (copy original file) to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def _decode(
        self,
        token_ids: Union[int, list[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        spaces_between_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        """
        Decode token ids to string.

        Uses the generic decode path from PreTrainedTokenizer which works for all vocabularies,
        including custom vocabularies that override _convert_id_to_token.
        """
        # Use parent class's generic decode method - it's simpler and works for all cases
        return super()._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )


class SentencePieceExtractor:
    """
    Extractor implementation for SentencePiece trained models. https://github.com/google/sentencepiece
    """

    def __init__(self, model: str):
        requires_backends(self, "sentencepiece")
        from sentencepiece import SentencePieceProcessor

        self.sp = SentencePieceProcessor()
        self.sp.Load(model)

    def extract(self, vocab_scores=None) -> tuple[dict[str, int], list[tuple[str, float]], list[tuple]]:
        """
        By default will return vocab and merges with respect to their order, by sending `vocab_scores` we're going to
        order the merges with respect to the piece scores instead.
        """
        sp = self.sp
        vocab_ids = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())}

        vocab_scores_dict = {sp.id_to_piece(i): sp.get_score(i) for i in range(sp.GetPieceSize())}

        merges = generate_merges(vocab_ids, vocab_scores_dict)

        vocab_scores_list = [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.GetPieceSize())]

        return vocab_ids, vocab_scores_list, merges
