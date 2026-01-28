# Copyright 2024 The HuggingFace Team. All rights reserved.
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

"""Tokenization class for RishAI."""

import json
from pathlib import Path

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import add_end_docstrings, logging


logger = logging.get_logger(__name__)


@add_end_docstrings
class RishAITokenizer(PreTrainedTokenizerBase):
    """
    Construct a RishAI tokenizer. Based on byte-level Byte-Pair-Encoding.

    This tokenizer inherits from [`PreTrainedTokenizerBase`] which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*):
            The token used for padding, for example when batching sequences of different lengths.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
        split_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not the special tokens should be split during the encoding.
    """

    vocab_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
    }
    pretrained_vocab_files_map = {
        "vocab_file": {},
        "merges_file": {},
    }
    max_model_input_sizes = {"default": 4096}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token=None,
        clean_up_tokenization_spaces=False,
        split_special_tokens=False,
        **kwargs,
    ):
        # Set default special tokens if not provided
        if pad_token is None:
            pad_token = "<|endoftext|>"

        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            split_special_tokens=split_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self.merges_file = merges_file

        # Initialize vocabulary
        self._vocab = {}
        self._merges = []
        self._bpe_ranks = {}

        if vocab_file is not None and merges_file is not None:
            self._load_vocab_and_merges(vocab_file, merges_file)

    def _load_vocab_and_merges(self, vocab_file, merges_file):
        """Load vocabulary and merges from files."""
        # Load vocabulary
        self._vocab = json.loads(Path(vocab_file).read_text(encoding="utf-8"))

        # Load merges
        self._merges = Path(merges_file).read_text(encoding="utf-8").split("\n")
        self._merges = [merge for merge in self._merges if merge.strip()]

        # Build BPE ranks
        self._bpe_ranks = {merge: i for i, merge in enumerate(self._merges)}

    @property
    def vocab_size(self) -> int:
        """Returns vocab size."""
        return len(self._vocab)

    def get_vocab(self) -> dict[str, int]:
        """Returns vocab as a dict."""
        return dict(self._vocab)

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        """Tokenize a string."""
        # Simple whitespace tokenization for now
        # In a real implementation, this would use BPE
        return text.split()

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) to an id using the vocab."""
        return self._vocab.get(token, self._vocab.get(self.unk_token, 0))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) to a token (str) using the vocab."""
        for token, idx in self._vocab.items():
            if idx == index:
                return token
        return self.unk_token

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Converts a sequence of tokens (string) in a single string."""
        # Simple detokenization - join with spaces
        return " ".join(tokens)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str, str]:
        """Save the vocabulary and merges files to a directory."""
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your tokenizer does not have the necessary information to save the vocabulary. "
                "Please use a tokenizer that has been trained with the correct parameters."
            )

        vocab_file = (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        merges_file = (filename_prefix + "-" if filename_prefix else "") + "merges.txt"

        vocab_file_path = f"{save_directory}/{vocab_file}"
        merges_file_path = f"{save_directory}/{merges_file}"

        with open(vocab_file_path, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)

        Path(merges_file_path).write_text("\n".join(self._merges), encoding="utf-8")

        return vocab_file_path, merges_file_path

    @property
    def can_save_slow_tokenizer(self) -> bool:
        """Check if the tokenizer can be saved."""
        return self._vocab is not None and self._merges is not None
