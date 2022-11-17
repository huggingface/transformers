import os
from shutil import copyfile
from typing import List, Optional, Tuple, Dict, Any

import sentencepiece as spm

from ... import GptSw3Tokenizer
from ...tokenization_utils import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging


logger = logging.get_logger(__name__)
# TODO: rename spm.model -> spiece.model?
VOCAB_FILES_NAMES = {"vocab_file": "spm.model"}

# TODO: rename spm.model -> spiece.model?
# TODO: This does not seem to be used at all though except for some unit testing? The file is found with the above:
#  VOCAB_FILES_NAMES, which is sufficient to find the tokenizer since there is one per model size
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "AI-Sweden/gpt-sw3-126m": "https://huggingface.co/AI-Sweden/gpt-sw3-126m/resolve/main/spm.model",
        "AI-Sweden/gpt-sw3-350m": "https://huggingface.co/AI-Sweden/gpt-sw3-350m/resolve/main/spm.model",
        "AI-Sweden/gpt-sw3-1.6b": "https://huggingface.co/AI-Sweden/gpt-sw3-1.6b/resolve/main/spm.model",
        "AI-Sweden/gpt-sw3-6.7b": "https://huggingface.co/AI-Sweden/gpt-sw3-6.7b/resolve/main/spm.model",
        "AI-Sweden/gpt-sw3-20b": "https://huggingface.co/AI-Sweden/gpt-sw3-20b/resolve/main/spm.model",
    }
}

# TODO: This does not seem to be used except for prompting a warning when tokenizing sequences longer than these
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "AI-Sweden/gpt-sw3-126m": 2048,
    "AI-Sweden/gpt-sw3-350m": 2048,
    "AI-Sweden/gpt-sw3-1.6b": 2048,
    "AI-Sweden/gpt-sw3-6.7b": 2048,
    "AI-Sweden/gpt-sw3-20b": 2048,
}


class GptSw3TokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = GptSw3Tokenizer

    def __init__(
            self,
            vocab_file,
            do_lower_case=False,
            remove_space=False,
            keep_accents=False,
            unk_token="<|endoftext|>",
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            sp_model_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> None:
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        super().__init__(
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

        # print("MODEL NAME OR PATH")
        # print(self.name_or_path)
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)


    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)
