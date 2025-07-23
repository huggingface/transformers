import base64
import logging
import os
import unicodedata
from typing import Collection, Optional, Union

import tiktoken

from transformers import AddedToken, PreTrainedTokenizer


logger = logging.getLogger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "hy.tiktoken"}

PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
# PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
ENDOFTEXT = "<|endoftext|>"
STARTOFTEXT = "<|startoftext|>"
BOSTOKEN = "<|bos|>"
EOSTOKEN = "<|eos|>"
PADTOKEN = "<|pad|>"

# as the default behavior is changed to allow special tokens in
# regular texts, the surface forms of special tokens need to be
# as different as possible to minimize the impact
EXTRAS = tuple((f"<|extra_{i}|>" for i in range(205)))
# changed to use actual index to avoid misconfiguration with vocabulary expansion


SPECIAL_START_ID = 127957


def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> dict[bytes, int]:
    # with open(tiktoken_bpe_file, "rb") as f:
    #     contents = f.read()
    dic = {}
    rank = 0
    for line in open(tiktoken_bpe_file, "rb"):
        if line:
            token, _ = line.split()
            if base64.b64decode(token) in dic:
                continue
            dic[base64.b64decode(token)] = int(rank)
            rank += 1
    global SPECIAL_START_ID
    SPECIAL_START_ID = rank
    return dic


# NOTE: Please use the code line to check `SPECIAL_START_ID` right, this will affect the SPECIAL_START_ID
# print(SPECIAL_START_ID)

SPECIAL_TOKENS = tuple(
    enumerate(
        (
            (
                ENDOFTEXT,
                STARTOFTEXT,
                BOSTOKEN,
                EOSTOKEN,
                PADTOKEN,
            )
            + EXTRAS
        ),
        start=SPECIAL_START_ID,
    )
)
# NOTE: Unused Token ID starts from 127962
SPECIAL_TOKENS_SET = {t for _, t in SPECIAL_TOKENS}


class HYTokenizer(PreTrainedTokenizer):
    """hunyuan tokenizer."""

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        errors="replace",
        extra_vocab_file=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # how to handle errors in decoding UTF-8 byte sequences
        # use ignore if you are in streaming inference
        self.errors = errors

        self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)  # type: Dict[bytes, int]
        self.special_tokens = {token: index for index, token in SPECIAL_TOKENS}

        # try load extra vocab from file
        if extra_vocab_file is not None:
            used_ids = set(self.mergeable_ranks.values()) | set(self.special_tokens.values())
            extra_mergeable_ranks = _load_tiktoken_bpe(extra_vocab_file)
            for token, index in extra_mergeable_ranks.items():
                if token in self.mergeable_ranks:
                    logger.info(f"extra token {token} exists, skipping")
                    continue
                if index in used_ids:
                    logger.info(f"the index {index} for extra token {token} exists, skipping")
                    continue
                self.mergeable_ranks[token] = index
            # the index may be sparse after this, but don't worry tiktoken.Encoding will handle this

        enc = tiktoken.Encoding(
            "HunYuan",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        assert len(self.mergeable_ranks) + len(self.special_tokens) == enc.n_vocab, (
            f"{len(self.mergeable_ranks)} + {len(self.special_tokens)} != {enc.n_vocab} in encoding"
        )

        self.decoder = {v: k for k, v in self.mergeable_ranks.items()}  # type: dict[int, bytes|str]
        self.decoder.update({v: k for k, v in self.special_tokens.items()})

        self.tokenizer = enc  # type: tiktoken.Encoding

        self.eod_id = self.tokenizer.eot_token
        self.bod_id = self.special_tokens[STARTOFTEXT]
        self.bos_id = self.special_tokens[BOSTOKEN]
        self.eos_id = self.special_tokens[EOSTOKEN]
        self.pad_id = self.special_tokens[PADTOKEN]

    def __getstate__(self):
        # for pickle lovers
        state = self.__dict__.copy()
        del state["tokenizer"]
        return state

    def __setstate__(self, state):
        # tokenizer is not python native; don't pass it; rebuild it
        self.__dict__.update(state)
        enc = tiktoken.Encoding(
            "HunYuan",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.tokenizer = enc

    def __len__(self) -> int:
        return self.tokenizer.n_vocab

    def get_vocab(self) -> dict[bytes, int]:
        return self.mergeable_ranks

    def convert_tokens_to_ids(self, tokens: Union[bytes, str, list[Union[bytes, str]]]) -> list[int]:
        ids = []
        if isinstance(tokens, (str, bytes)):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.mergeable_ranks.get(tokens)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.mergeable_ranks.get(token))
        return ids

    def _add_tokens(
        self,
        new_tokens: Union[list[str], list[AddedToken]],
        special_tokens: bool = False,
    ) -> int:
        if not special_tokens and new_tokens:
            raise ValueError("Adding regular tokens is not supported")
        for token in new_tokens:
            surface_form = token.content if isinstance(token, AddedToken) else token
            if surface_form not in SPECIAL_TOKENS_SET:
                raise ValueError("Adding unknown special tokens is not supported")
        return 0

    def save_vocabulary(self, save_directory: str, **kwargs) -> tuple[str]:
        """
        Save only the vocabulary of the tokenizer (vocabulary).
        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        file_path = os.path.join(save_directory, "hunyuan.tiktoken")
        with open(file_path, "w", encoding="utf8") as w:
            for k, v in self.mergeable_ranks.items():
                line = base64.b64encode(k).decode("utf8") + " " + str(v) + "\n"
                w.write(line)
        return (file_path,)

    def tokenize(
        self,
        text: str,
        allowed_special: Union[set, str] = "all",
        disallowed_special: Union[Collection, str] = (),
        **kwargs,
    ) -> list[Union[bytes, str]]:
        """
        Converts a string in a sequence of tokens.
        Args:
            text (`str`):
                The sequence to be encoded.
            allowed_special (`Literal["all"]` or `set`):
                The surface forms of the tokens to be encoded as special tokens in regular texts.
                Default to "all".
            disallowed_special (`Literal["all"]` or `Collection`):
                The surface forms of the tokens that should not be in regular texts and trigger errors.
                Default to an empty tuple.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific encode method.
        Returns:
            `List[bytes|str]`: The list of tokens.
        """
        tokens = []
        text = unicodedata.normalize("NFC", text)

        # this implementation takes a detour: text -> token id -> token surface forms
        for t in self.tokenizer.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special):
            tokens.append(self.decoder[t])
        return tokens

    def convert_tokens_to_string(self, tokens: list[Union[bytes, str]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors=self.errors)
                    temp = b""
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type types or str")
        if temp:
            text += temp.decode("utf-8", errors=self.errors)
        return text

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    def _convert_id_to_token(self, index: int) -> Union[bytes, str]:
        """Converts an id to a token, special tokens included"""
        if index in self.decoder:
            return self.decoder[index]
        raise ValueError("unknown ids")

    def _convert_token_to_id(self, token: Union[bytes, str]) -> int:
        """Converts a token to an id using the vocab, special tokens included"""
        if token in self.special_tokens:
            return self.special_tokens[token]
        if token in self.mergeable_ranks:
            return self.mergeable_ranks[token]
        raise ValueError("unknown token")

    def _tokenize(self, text: str, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).
        Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def _decode(
        self,
        token_ids: Union[int, list[int]],
        skip_special_tokens: bool = False,
        errors: Optional[str] = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            token_ids = [i for i in token_ids if i < self.eod_id]
        return self.tokenizer.decode(token_ids, errors=errors or self.errors)


__all__ = ["HYTokenizer"]
