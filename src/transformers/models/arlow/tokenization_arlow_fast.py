from typing import Optional, Tuple

from transformers.models.arlow.tokenization_arlow import ArlowTokenizer
from transformers.tokenization_utils import AddedToken
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",  # For a full fast tokenizer JSON.
}


class ArlowTokenizerFast(PreTrainedTokenizerFast):
    """
    ArlowTokenizerFast is a custom fast tokenizer for the ArlowGPT model.
    It is backed by Hugging Face’s fast tokenizer library and supports subword tokenization,
    padding, truncation, and special token handling.

    Example:
        >>> from transformers.models.arlow.tokenization_arlow_fast import ArlowTokenizerFast
        >>> tokenizer = ArlowTokenizerFast.from_pretrained("path/to/tokenizer")
        >>> tokens = tokenizer("Hello, world!")
        >>> print(tokens)
    """

    vocab_files_names = VOCAB_FILES_NAMES
    # Set slow_tokenizer_class to the actual slow class.
    slow_tokenizer_class = ArlowTokenizer
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        unk_token: str = "<|unk|>",
        bos_token: Optional[str] = "<|startoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|pad|>",
        mask_token: str = "<|mask|>",
        additional_special_tokens: Optional[list] = None,
        **kwargs,
    ):
        # Similar to Qwen2TokenizerFast, convert tokens to AddedToken with proper flags.
        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(unk_token, str)
            else unk_token
        )
        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(pad_token, str)
            else pad_token
        )
        mask_token = (
            AddedToken(mask_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )

        # Call the parent class initializer.
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)


__all__ = ["ArlowTokenizerFast"]
