import os
from shutil import copyfile
from typing import List, Optional, Tuple

from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast, logger
from transformers.models.UDOP import UdopTokenizer
from transformers.models.UDOP.tokenization_udop import convert_slow_udoptokenizer


VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}


class UdopTokenizerFast(PreTrainedTokenizerFast):

    slow_tokenizer_class = UdopTokenizer
    prefix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file,
        tokenizer_file=None,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        loc_extra_ids=201,
        other_extra_ids=200,
        additional_special_tokens=None,
        **kwargs
    ):
        # Add extra_ids to the special token list
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = ["<extra_id_{}>".format(i) for i in range(extra_ids)]
            additional_special_tokens.extend(["<extra_l_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["</extra_l_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["<extra_t_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["</extra_t_id_{}>".format(i) for i in range(extra_ids)])

        if loc_extra_ids > 0 and not "<loc_0>" in additional_special_tokens:
            additional_special_tokens.extend(["<loc_{}>".format(i) for i in range(loc_extra_ids)])

        if other_extra_ids > 0 and not "<other_0>" in additional_special_tokens:
            additional_special_tokens.extend(["<other_{}>".format(i) for i in range(other_extra_ids)])

        slow_tokenizer = self.slow_tokenizer_class(
            vocab_file,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            loc_extra_ids=loc_extra_ids,
            other_extra_ids=other_extra_ids,
            **kwargs,
        )
        fast_tokenizer = convert_slow_udoptokenizer(slow_tokenizer)
        self._tokenizer = fast_tokenizer

        PreTrainedTokenizerBase.__init__(
            self,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self._loc_extra_ids = loc_extra_ids
        self._other_extra_ids = other_extra_ids

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
            logger.info(f"Copy vocab file to {out_vocab_file}")

        return (out_vocab_file,)
