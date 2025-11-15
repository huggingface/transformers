# src/transformers/models/evo2/tokenization_evo2.py

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        # You can fill these in once you upload a checkpoint
        # "arcinstitute/evo2-1b-8k": "https://huggingface.co/arcinstitute/evo2-1b-8k/resolve/main/vocab.json",
    }
}

PRETRAINED_INIT_CONFIGURATION = {
    # "arcinstitute/evo2-1b-8k": {},
}


class Evo2Tokenizer(PreTrainedTokenizer):
    """
    Hugging Face wrapper around the Evo2 CharLevelTokenizer.

    - Encoding:
        text.encode("utf-8") -> list of uint8 bytes in [0, 255]
    - Token IDs:
        those bytes directly used as IDs (0..255).
        `vocab_size` can be larger (e.g. 512), but extra IDs are unused.
    - Decoding:
        clamp each id with `clamp(n) = max(32, min(n, vocab_size))`
        then `chr(clamp(n))` and join.

    We implement vocab as stringified integers: "0" -> 0, "1" -> 1, etc.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        vocab_size: int = 512,
        # Match original CharLevelTokenizer semantics:
        # eod_id = eos_id = 0, pad_id = 1
        eos_token: str = "0",
        pad_token: str = "1",
        unk_token: str = "0",  # there is no real "unknown" in char-level; anything maps to a byte
        bos_token: Optional[str] = None,
        **kwargs,
    ):
        self._vocab_size = vocab_size

        if vocab_file is None:
            # Default vocab: token "0" -> id 0, "1" -> id 1, ..., up to vocab_size-1
            self.vocab: Dict[str, int] = {str(i): i for i in range(vocab_size)}
        else:
            with open(vocab_file, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
            # Ensure ids are ints
            self.vocab = {str(k): int(v) for k, v in self.vocab.items()}

        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        # Call parent ctor (this also sets pad/eos/bos/unk attributes)
        super().__init__(
            eos_token=eos_token,
            pad_token=pad_token,
            bos_token=bos_token,  # None by default; CharLevelTokenizer has no BOS
            unk_token=unk_token,
            **kwargs,
        )

        # Cache some commonly used ids
        self._eos_id = int(eos_token) if bos_token is None else self.vocab[eos_token]
        self._pad_id = int(pad_token)
        self._unk_id = int(unk_token)

    # ---- Char-level core logic ---------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    def clamp(self, n: int) -> int:
        # Same as in CharLevelTokenizer: max(32, min(n, vocab_size))
        return max(32, min(n, self._vocab_size))

    # HF will call this to get string "tokens" before converting to ids
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        # CharLevelTokenizer.tokenize:
        #   list(np.frombuffer(text.encode('utf-8'), dtype=np.uint8))
        # We can replicate with Python directly:
        byte_ids = list(text.encode("utf-8"))  # each in [0, 255]
        # Represent each id as a string token "id"
        return [str(b) for b in byte_ids]

    def _convert_token_to_id(self, token: str) -> int:
        # Tokens we produce are numeric strings "0", "1", ...
        try:
            idx = int(token)
        except ValueError:
            # Shouldn't really happen with our _tokenize, but just in case
            return self._unk_id
        # CharLevelTokenizer allows any 0..255; we don't clamp on encode.
        # (clamp is only used on decode)
        if 0 <= idx < self._vocab_size:
            return idx
        # If out-of-range, fall back to unk
        return self._unk_id

    def _convert_id_to_token(self, index: int) -> str:
        # Represent ids as numeric strings consistently
        if 0 <= index < self._vocab_size:
            return str(index)
        return str(self._unk_id)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        # CharLevelTokenizer.detokenize:
        #   "".join(chr(clamp(token)) for token in token_ids)
        chars: List[str] = []
        for tok in tokens:
            try:
                idx = int(tok)
            except ValueError:
                idx = self._unk_id
            c = chr(self.clamp(idx))
            chars.append(c)
        return "".join(chars)

    # ---- Special tokens / sequence helpers ---------------------------------

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        """
        CharLevelTokenizer does *not* add BOS/EOS automatically, so we just
        return the sequence as-is.

        We also do not support sentence pairs.
        """
        if token_ids_1 is not None:
            raise ValueError("Evo2Tokenizer (CharLevel) does not support sentence pairs.")

        return token_ids_0

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Mark eos/eod (id 0) and pad (id 1) as special, everything else as 0.
        """
        if token_ids_1 is not None:
            raise ValueError("Evo2Tokenizer (CharLevel) does not support sentence pairs.")

        if already_has_special_tokens:
            # Just mark known special IDs
            return [
                1 if t in {self._eos_id, self._pad_id} else 0
                for t in token_ids_0
            ]

        # We don't auto-add any extra tokens, so same as above
        return [
            1 if t in {self._eos_id, self._pad_id} else 0
            for t in token_ids_0
        ]

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        """
        No token type IDs; everything is 0, like most decoder-only models.
        """
        if token_ids_1 is not None:
            raise ValueError("Evo2Tokenizer (CharLevel) does not support sentence pairs.")

        return [0] * len(token_ids_0)

    # ---- Saving / loading vocab --------------------------------------------

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        vocab_file = (
            (filename_prefix + "-" if filename_prefix else "")
            + VOCAB_FILES_NAMES["vocab_file"]
        )
        vocab_path = os.path.join(save_directory, vocab_file)

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        return (vocab_path,)
