import os
import unicodedata
import re
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import sentencepiece as spm
import numpy as np

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging, to_py_obj, is_torch_tensor, TensorType

logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

# This does not seem to be used at all except for some unit testing? The file is found with the above:
#  VOCAB_FILES_NAMES, which is sufficient to find the tokenizer since there is one per model size
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "AI-Sweden/gpt-sw3-126m": "https://huggingface.co/AI-Sweden/gpt-sw3-126m/resolve/main/spiece.model",
        "AI-Sweden/gpt-sw3-350m": "https://huggingface.co/AI-Sweden/gpt-sw3-350m/resolve/main/spiece.model",
        "AI-Sweden/gpt-sw3-1.6b": "https://huggingface.co/AI-Sweden/gpt-sw3-1.6b/resolve/main/spiece.model",
        "AI-Sweden/gpt-sw3-6.7b": "https://huggingface.co/AI-Sweden/gpt-sw3-6.7b/resolve/main/spiece.model",
        "AI-Sweden/gpt-sw3-20b": "https://huggingface.co/AI-Sweden/gpt-sw3-20b/resolve/main/spiece.model",
    }
}

# This does not seem to be used except for prompting a warning when tokenizing sequences longer than these
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "AI-Sweden/gpt-sw3-126m": 2048,
    "AI-Sweden/gpt-sw3-350m": 2048,
    "AI-Sweden/gpt-sw3-1.6b": 2048,
    "AI-Sweden/gpt-sw3-6.7b": 2048,
    "AI-Sweden/gpt-sw3-20b": 2048,
}


class GptSw3Tokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

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

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        self.whitespaces = {
            " ",
            " ",
            " ",
            " ",
            " ",
            "　",
            " ",
            " ",
            " ",
            " ",
            "￼",
            "",
        }

        # Control chars except newlines and tabs, soft-hyphens, and non-breaking space, zero-width space
        additional_chars_to_remove = [160, 173, 8203]
        self.non_printing_characters_re = re.compile(
            f"[{''.join(map(chr, list(range(0, 9)) + list(range(11, 32)) + list(range(127, 160)) + additional_chars_to_remove))}]"
        )

    @property
    def vocab_size(self) -> int:
        return len(self.sp_model)

    def preprocess_text(self, inputs):
        # Remove non-printing characters
        outputs = self.non_printing_characters_re.sub("", inputs)

        # Normalize whitespaces
        outputs = "".join(
            [char if char not in self.whitespaces else " " for char in outputs]
        )

        # NFC Unicode normalization
        outputs = unicodedata.normalize("NFC", outputs)
        return outputs

    def _tokenize(self, text, **kwargs):
        """
            Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
            vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

            Do NOT take care of added tokens.
        """

        text = self.preprocess_text(text)
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    # This method is not abstract, but is required to function correctly
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.sp_model.decode(tokens)

    # This method is not abstract, but overriding this ensures that e.g. bytes gets represented as intended
    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        spaces_between_special_tokens: bool = True,
        **kwargs
    ) -> str:
        return self.sp_model.decode(token_ids)

    def get_vocab(self) -> Dict[str, int]:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

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

    def _to_py_obj_light(self, obj):
        # Like to_py_obj but removes checks for UserDict, list of tuples, TF, and Jax
        # This gives ~2x speedup compared to supporting everything, and ~0.5x speedup compared to only supporting list
        if isinstance(obj, (np.ndarray, np.number)):  # tolist also works on 0d np arrays
            return obj.tolist()
        elif is_torch_tensor(obj):
            return obj.detach().cpu().tolist()
        else:
            return obj

    # Overriding decode instead of _decode yields significant speedup but removes some functionality
    def decode_fast(self, token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor"]) -> str:
        # token_ids = to_py_obj(token_ids)
        token_ids = self._to_py_obj_light(token_ids)

        return self.sp_model.decode(token_ids)

    # Fastest but only works for python ints or lists
    def decode_faster(self, token_ids: Union[int, List[int]]) -> str:
        # token_ids = to_py_obj(token_ids)

        return self.sp_model.decode(token_ids)

    def encode_fast(self, text: str, return_tensors: Optional[Union[str, TensorType]] = None) -> List[int]:
        text = self.preprocess_text(text)
        token_ids = self.sp_model.encode(text)
        if return_tensors == "pt":
            token_ids = torch.tensor(token_ids)
        return token_ids

    # Almost 10x speedup to override this for batch encode, 2x speedup for single-text encode
    def call_fast(self, text: Union[str, List[str]], return_tensors: Optional[Union[str, TensorType]] = None
                  ) -> Dict[str, Union[Any]]:

        if isinstance(text, str):
            text = self.preprocess_text(text)
            token_ids = self.sp_model.encode(text)
            attention_mask = [1 for _ in token_ids]
        else:
            text = [self.preprocess_text(t) for t in text]
            token_ids = self.sp_model.encode(text)
            attention_mask = [[1] * len(ids_list) for ids_list in token_ids]

        if return_tensors == "pt":
            token_ids = torch.tensor(token_ids)
            attention_mask = torch.tensor(attention_mask)

        out_dict = {"input_ids": token_ids, "attention_mask": attention_mask}
        return out_dict
