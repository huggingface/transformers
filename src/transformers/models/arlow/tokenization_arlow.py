import json
import os
from typing import Dict, List, Optional, Tuple

import regex as re

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}


class ArlowTokenizer(PreTrainedTokenizer):
    """
    ArlowTokenizer is a custom BPE tokenizer used for the ArlowGPT model.

    This is the slow (Python-based) implementation that provides subword tokenization,
    padding, truncation, and special token handling.

    Example:
        >>> from transformers.models.arlow.tokenization_arlow import ArlowTokenizer
        >>> tokenizer = ArlowTokenizer.from_pretrained("path/to/tokenizer")
        >>> tokens = tokenizer("Hello, world!")
        >>> print(tokens)

    Attributes:
        vocab_file (str): Path to the vocabulary file.
        merges_file (str): Path to the merges file.
        bos_token (`str`, *optional*, defaults to `"<|startoftext|>"`): <fill_docstring>
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`): <fill_docstring>
        unk_token (`str`, *optional*, defaults to `"<|unk|>"`): <fill_docstring>
        pad_token (`str`, *optional*, defaults to `"<|pad|>"`): <fill_docstring>
        mask_token (`str`, *optional*, defaults to `"<|mask|>"`): <fill_docstring>
        additional_special_tokens (`Optional`, *optional*): <fill_docstring>
    """

    vocab_files_names = VOCAB_FILES_NAMES
    fast_tokenizer_class = "ArlowTokenizerFast"  # (Not used during conversion)
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        merges_file: str,
        bos_token: str = "<|startoftext|>",
        eos_token: str = "<|endoftext|>",
        unk_token: str = "<|unk|>",
        pad_token: str = "<|pad|>",
        mask_token: str = "<|mask|>",
        additional_special_tokens: Optional[List[str]] = None,
        **kwargs,
    ):
        # Save file paths so they can be accessed during conversion.
        self.vocab_file = vocab_file
        self.merges_file = merges_file

        # (Optional) You could convert tokens via AddedToken here as in Qwen, if desired.
        default_special_tokens = ["<|im_start|>", "<|im_end|>"]
        if additional_special_tokens is not None and isinstance(additional_special_tokens, list):
            default_special_tokens.extend([t for t in additional_special_tokens if t not in default_special_tokens])

        # Load vocabulary.
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}

        # Load merges and create BPE ranks.
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")
            merges = [m for m in merges if m and not m.startswith("#")]
            self.bpe_ranks = {tuple(merge.split()): i for i, merge in enumerate(merges)}

        # Cache for BPE.
        self.cache = {}

        # Compile regex pattern.
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            additional_special_tokens=default_special_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.encoder)

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]

        word = list(token)
        pairs = get_pairs(word)
        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break

                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        subwords = " ".join(word)
        self.cache[token] = subwords
        return subwords

    def _tokenize(self, text: str) -> List[str]:
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            # Simulate byte-level encoding.
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace")
        return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, _ in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != 0:
                    writer.write("\n")
                writer.write(" ".join(bpe_tokens))
                index += 1

        return vocab_file, merge_file


def get_pairs(word: List[str]) -> set:
    """
    Return set of symbol pairs in a word.
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


# Helper dictionaries for byte encoding/decoding.
_base_byte_encoder = {i: chr(i) for i in range(256)}
_base_byte_decoder = {v: k for k, v in _base_byte_encoder.items()}

ArlowTokenizer.byte_encoder = _base_byte_encoder
ArlowTokenizer.byte_decoder = _base_byte_decoder

__all__ = ["ArlowTokenizer"]
