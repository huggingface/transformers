# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for Whisper."""
import json
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import regex as re

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
from .english_normalizer import EnglishTextNormalizer


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "tokenizer_file": "tokenizer.json",
    "merges_file": "merges.txt",
    "normalizer_file": "normalizer.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openai/whisper-base": "https://huggingface.co/openai/whisper-base/resolve/main/vocab.json",
    },
    "merges_file": {"openai/whisper-base": "https://huggingface.co/openai/whisper-base/resolve/main/merges_file.txt"},
    "normalizer_file": {
        "openai/whisper-base": "https://huggingface.co/openai/whisper-base/resolve/main/normalizer.json"
    },
}

MAX_MODEL_INPUT_SIZES = {
    "openai/whisper-base": 448,
}


# Copied from transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


logger = logging.get_logger(__name__)


# Copied from transformers.models.gpt2.tokenization_gpt2.get_pairs
def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}

TASK_IDS = ["translate", "transcribe"]


class WhisperTokenizer(PreTrainedTokenizer):
    """
    Construct a Whisper tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.

     Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        normalizer_file (`str`, *optional*, defaults to `None`):
            Path to the normalizer_file file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|startoftranscript|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word.
        language (`str`, *optional*):
            The language of the transcription text. The corresponding language id token is appended to the start of the
            sequence for multilingual speech recognition and speech translation tasks, e.g. for Spanish the token
            `"<|es|>"` is appended to the start of sequence. This should be used for multilingual fine-tuning only.
        task (`str`, *optional*):
            Task identifier to append at the start of sequence (if any). This should be used for mulitlingual
            fine-tuning, with `"transcribe"` for speech recognition and `"translate"` for speech translation.
        predict_timestamps (`bool`, *optional*, defaults to `False`):
            Whether to omit the `<|notimestamps|>` token at the start of the sequence.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = MAX_MODEL_INPUT_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        normalizer_file=None,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|startoftranscript|>",
        eos_token="<|endoftext|>",
        pad_token=None,
        add_prefix_space=False,
        language=None,
        task=None,
        predict_timestamps=False,
        **kwargs,
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.add_prefix_space = add_prefix_space

        if normalizer_file is not None:
            with open(normalizer_file, encoding="utf-8") as vocab_handle:
                self.english_spelling_normalizer = json.load(vocab_handle)
        else:
            self.english_spelling_normalizer = None

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.language = language
        self.task = task
        self.predict_timestamps = predict_timestamps

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.bpe with GPT2 -> Whisper
    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
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
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def set_prefix_tokens(self, language: str = None, task: str = None, predict_timestamps: bool = None):
        """
        Override the prefix tokens appended to the start of the label sequence. This method can be used standalone to
        update the prefix tokens as required when fine-tuning. Example:

        ```python
        >>> # instantiate the tokenizer and set the prefix token to Spanish
        >>> tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="spanish")
        >>> # now switch the prefix token from Spanish to French
        >>> tokenizer.set_prefix_tokens(language="french")
        ```

        Args:
            language (`str`, *optional*, defaults to `None`):
                The language of the transcription text.
            task (`str`, *optional*, defaults to `None`):
                Task identifier to append at the start of sequence (if any).
            predict_timestamps (`bool`, *optional*, defaults to `None`):
                Whether to omit the `<|notimestamps|>` token at the start of the sequence.
        """
        self.language = language if language is not None else self.language
        self.task = task if task is not None else self.task
        self.predict_timestamps = predict_timestamps if predict_timestamps is not None else self.predict_timestamps

    @property
    def prefix_tokens(self) -> List[int]:
        all_special_ids = self.all_special_ids
        bos_token_id = all_special_ids[-106]
        translate_token_id = all_special_ids[-6]
        transcribe_token_id = all_special_ids[-5]
        notimestamps_token_id = all_special_ids[-1]
        langs = tuple(LANGUAGES.keys())

        if self.language is not None:
            self.language = self.language.lower()
            if self.language in TO_LANGUAGE_CODE:
                language_id = TO_LANGUAGE_CODE[self.language]
            elif self.language in TO_LANGUAGE_CODE.values():
                language_id = self.language
            else:
                is_language_code = len(self.language) == 2
                raise ValueError(
                    f"Unsupported language: {self.language}. Language should be one of:"
                    f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                )

        if self.task is not None:
            if self.task not in TASK_IDS:
                raise ValueError(f"Unsupported task: {self.task}. Task should be in: {TASK_IDS}")

        bos_sequence = [bos_token_id]
        if self.language is not None:
            bos_sequence.append(bos_token_id + 1 + langs.index(language_id))
        if self.task is not None:
            bos_sequence.append(transcribe_token_id if self.task == "transcribe" else translate_token_id)
        if not self.predict_timestamps:
            bos_sequence.append(notimestamps_token_id)
        return bos_sequence

    # Copied from transformers.models.speech_to_text.tokenization_speech_to_text.Speech2TextTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + [self.eos_token_id]

    # Copied from transformers.models.speech_to_text.tokenization_speech_to_text.Speech2TextTokenizer.get_special_tokens_mask
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1]
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._tokenize with GPT2 -> Whisper
    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._convert_token_to_id with GPT2 -> Whisper
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """
        Converts an index (integer) in a token (str) using the vocab. Whisper's base tokenizer always decodes OOV
        tokens as "", thus we do not use the `unk_token` here.
        """
        return self.decoder.get(index, "")

    def _normalize(self, text):
        """
        Normalize a given string using the `EnglishTextNormalizer` class, which preforms commons transformation on
        english text.
        """
        normalizer = EnglishTextNormalizer(self.english_spelling_normalizer)
        return normalizer(text)

    def _decode_with_timestamps(self, token_ids, time_precision=0.02) -> str:
        """
        Timestamp tokens are above the special tokens' id range and are ignored by `decode()`. This method decodes
        given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        timestamp_begin = self.all_special_ids[-1] + 1
        outputs = [[]]
        for token in token_ids:
            if token >= timestamp_begin:
                timestamp = f"<|{(token - timestamp_begin) * time_precision:.2f}|>"
                outputs.append(timestamp)
                outputs.append([])
            else:
                outputs[-1].append(token)
        outputs = [s if isinstance(s, str) else self.decode(s) for s in outputs]
        return "".join(outputs)

    def _compute_offsets(self, token_ids, time_precision=0.02):
        """
        Compute offsets for a given tokenized input

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            time_precision (`float`, `optional`, defaults to 0.02):
                The time ratio to convert from token to time.
        """
        offsets = []
        token_ids = np.array(token_ids)
        if token_ids.shape[0] > 1 and len(token_ids.shape) > 1:
            raise ValueError("Can only process a single input at a time")
        timestamp_begin = self.all_special_ids[-1] + 1
        timestamp_tokens = token_ids >= timestamp_begin

        consecutive = np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0] + 1
        if consecutive.shape[0] == 0 and timestamp_tokens.sum() <= 1:
            # either there are no timestamps or there are no consecutive ones
            return []
        elif np.where(timestamp_tokens)[0][-1] + 1 not in consecutive:
            # we add the final timestamp if it is not already in the list
            consecutive = np.append(consecutive, np.where(timestamp_tokens)[0][-1] + 1)

        last_slice = np.where(timestamp_tokens)[0][0]
        for current_slice in consecutive:
            sliced_tokens = token_ids[last_slice:current_slice]
            if len(sliced_tokens) > 1:
                start_timestamp_position = sliced_tokens[0].item() - timestamp_begin
                end_timestamp_position = sliced_tokens[-1].item() - timestamp_begin
                offsets.append(
                    {
                        "text": self._decode(sliced_tokens),
                        "timestamp": (
                            start_timestamp_position * time_precision,
                            end_timestamp_position * time_precision,
                        ),
                    }
                )
            last_slice = current_slice

        return offsets

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        output_offsets: bool = False,
        time_precision=0.02,
        decode_with_timestamps: bool = False,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `True`):
                Whether or not to clean up the tokenization spaces.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.
            output_offsets (`bool`, *optional*, defaults to `False`):
                Whether or not to output the offsets of the tokens. This should only be set if the model predicted
                timestamps.
            decode_with_timestamps (`bool`, *optional*, defaults to `False`):
                WHether or not to decode with timestamps included in the raw text.
        Returns:
            `str`: The decoded sentence.
        """
        text = super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        if decode_with_timestamps:
            text = self._decode_with_timestamps(token_ids, time_precision=time_precision)
        # retrieve offsets
        if output_offsets:
            offsets = None
            offsets = self._compute_offsets(token_ids, time_precision=time_precision)
            return {"text": text, "offsets": offsets}
        return text

    def _decode(
        self, token_ids: Union[int, List[int]], skip_special_tokens: bool = False, normalize: bool = False, **kwargs
    ) -> str:
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        text = "".join(sub_texts)

        if normalize:
            clean_text = self._normalize(text)
            return clean_text
        else:
            return text

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.convert_tokens_to_string with GPT2 -> Whisper
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )
        normalizer_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["normalizer_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        if self.english_spelling_normalizer is not None:
            with open(normalizer_file, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(self.english_spelling_normalizer, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                )

        return vocab_file, merge_file, normalizer_file

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.prepare_for_tokenization with GPT2 -> Whisper
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._build_conversation_input_ids with GPT2 -> Whisper
    def _build_conversation_input_ids(self, conversation) -> List[int]:
        input_ids = []
        for is_user, text in conversation.iter_texts():
            input_ids.extend(self.encode(text, add_special_tokens=False) + [self.eos_token_id])
        if len(input_ids) > self.model_max_length:
            input_ids = input_ids[-self.model_max_length :]
        return input_ids

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        self.set_prefix_tokens(task=task, language=language, predict_timestamps=not no_timestamps)
        # prefix tokens are of the form: <|startoftranscript|> <|lang_id|> <|task|> <|notimestamps|>
        # we don't want to force the bos token at position 1, as this is the starting token
        # when we generate, so we slice the prefix tokens to: <|lang_id|> <|task|> <|notimestamps|>
        # to get the forced tokens
        forced_tokens = self.prefix_tokens[1:]
        forced_decoder_ids = [(rank + 1, token) for rank, token in enumerate(forced_tokens)]
        return forced_decoder_ids

    def _decode_asr(self, model_outputs, *, return_timestamps, return_language, time_precision):
        return _decode_asr(
            self,
            model_outputs,
            return_timestamps=return_timestamps,
            return_language=return_language,
            time_precision=time_precision,
        )


def _decode_asr(tokenizer, model_outputs, *, return_timestamps, return_language, time_precision):
    """
    Internal method meant to only be used by asr pipeline. Handles all the little quirks specific to whisper to handle
    the various options not allowed in other seq2seq models
    """

    # =========== Overview ============
    # - iterate over all outputs
    # - all tokens within output
    # - Each token can be
    #   - language token
    #   - special token
    #   - timestamp token
    #   - text token
    # - We accumulate the text tokens.
    # - We split on end timestamps
    # - Lots of complexity comes from stride and timestamps

    last_language = None

    def new_chunk():
        return {"language": last_language, "timestamp": [None, None], "text": ""}

    # Welcome to the state machine !
    chunks = []
    chunk = new_chunk()
    time_offset = 0.0
    timestamp_begin = tokenizer.convert_tokens_to_ids("<|notimestamps|>") + 1
    previous_tokens = []
    skip = False
    right_stride_start = None

    all_special_ids = set(tokenizer.all_special_ids)
    # - iterate over all outputs
    for chunk_id, output in enumerate(model_outputs):
        # We can drop everything to Python list, it's going to make
        # our lives easier
        token_ids = output["tokens"][0].tolist()

        # Those keep track of timestamps within strides
        # Which need to be skipped and resolve all tokens in a single
        # chunk.
        last_timestamp = None
        first_timestamp = timestamp_begin

        if "stride" in output:
            chunk_len, stride_left, stride_right = output["stride"]
            # Offset the timings to account for the other `model_outputs`.
            time_offset -= stride_left
            right_stride_start = chunk_len - stride_right

            # Keeping track of timestamps within strides
            # We're going to NOT split on those, and delay until we're
            # out of BOTH stride. Otherwise lots of issues occur and
            # corner cases
            if stride_left:
                first_timestamp = stride_left / time_precision + timestamp_begin
            if stride_right:
                for token in reversed(token_ids):
                    if token >= timestamp_begin:
                        # There can be several token in the right stride
                        # But the last one is ALWAYS going to be skipped
                        if (
                            last_timestamp is not None
                            and (token - timestamp_begin) * time_precision < right_stride_start
                        ):
                            break
                        last_timestamp = token

        current_tokens = []

        # - all tokens within output
        for i, token in enumerate(token_ids):
            # 4 possible states for each token
            # - 1/ Language code
            # - 2/ all other special tokens (which we ignore)
            # - 3/ Timestamp
            # - 4/ Regular text
            if token in all_special_ids:
                # Either language code or other
                text = tokenizer.decode([token])
                # Removing outer shell <|XX|>
                text = text[2:-2]
                language = LANGUAGES.get(text, None)
                if language is not None:
                    # 1/ Indeed some language
                    # TODO Handle when language is different from the previous
                    # one, and we cannot use timestamped tokens to create chunks
                    if last_language and language != last_language and not return_timestamps:
                        previous_tokens.append(current_tokens)
                        resolved_tokens = _find_longest_common_sequence(previous_tokens)
                        resolved_text = tokenizer.decode(resolved_tokens)
                        chunk["text"] = resolved_text
                        chunks.append(chunk)

                        # Flush all our temporary context
                        previous_tokens = []
                        current_tokens = []
                        chunk = new_chunk()
                    chunk["language"] = language
                    last_language = language
                else:
                    # 2/ This is a regular special token, ignoring it
                    pass
            elif token >= timestamp_begin:
                # 3/ Timestamp token
                time = (token - timestamp_begin) * time_precision + time_offset
                time = round(time, 2)
                if last_timestamp and token >= last_timestamp:
                    # Whisper outputted a timestamp token, but it falls within
                    # our stride, so we're going to skip it for the time being
                    # and resolve this later
                    # Skip is necessary because timestamp tokens always come
                    # by pair, so we need to skip the next one too (which would mark the start of another chunk).
                    skip = True
                elif skip or (previous_tokens and token < first_timestamp):
                    skip = False
                elif chunk["timestamp"][0] is None:
                    chunk["timestamp"][0] = time
                else:
                    # This is the end of the timestamp chunk
                    if time == chunk["timestamp"][0]:
                        # This is a bug in timestamp token output
                        # where we're taking the duplicate token
                        # as a stop where it should be a start.
                        # This is an issue in the underlying model output
                        # Let's just skip it so it becomes de-factor
                        # a start agin
                        pass
                    else:
                        chunk["timestamp"][1] = time
                        # Handling merges.
                        previous_tokens.append(current_tokens)
                        resolved_tokens = _find_longest_common_sequence(previous_tokens)
                        resolved_text = tokenizer.decode(resolved_tokens)
                        chunk["text"] = resolved_text
                        chunks.append(chunk)

                        # Flush all our temporary context
                        previous_tokens = []
                        current_tokens = []
                        chunk = new_chunk()
            else:
                # 4/ Regular token
                # We just append to the list of all tokens so we can handle
                # merges later and decode into text.
                current_tokens.append(token)

        if "stride" in output:
            time_offset += chunk_len - stride_right

        # Leftover tokens
        if current_tokens:
            previous_tokens.append(current_tokens)
        elif not (any(p for p in previous_tokens)):
            # print("Flushing previous tokens (END)")
            chunk = new_chunk()
            previous_tokens = []
            current_tokens = []

    if previous_tokens:
        if return_timestamps:
            # Last token should always be timestamps, so there shouldn't be
            # leftover
            raise ValueError(
                "There was an error while processing timestamps, we haven't found a timestamp as last token. Was"
                " WhisperTimeStampLogitsProcessor used?"
            )
        # Happens when we don't use timestamps
        resolved_tokens = _find_longest_common_sequence(previous_tokens)
        # print("Flushing previous tokens (FINAL)")
        resolved_text = tokenizer.decode(resolved_tokens)
        chunk["text"] = resolved_text
        chunks.append(chunk)

    # Preparing and cleaning up the pipeline output
    full_text = "".join(chunk["text"] for chunk in chunks)
    if return_timestamps or return_language:
        for chunk in chunks:
            if not return_timestamps:
                chunk.pop("timestamp")
            else:
                chunk["timestamp"] = tuple(chunk["timestamp"])
            if not return_language:
                chunk.pop("language")
        optional = {"chunks": chunks}
    else:
        optional = {}
    return full_text, optional


def _find_longest_common_sequence(sequences):
    # It would be much harder to do O(n) because of fault tolerance.
    # We actually have a really good property which is that the total sequence
    # MUST be those subsequences in order.
    left_sequence = sequences[0]
    left_length = len(left_sequence)
    total_sequence = []
    for right_sequence in sequences[1:]:
        # index = 0
        max_ = 0.0
        max_indices = (left_length, left_length, 0, 0)
        # Here we're sliding matches
        # [a, b, c, d]
        #          [c, d, f]
        # =        [c] == [d]
        #
        # [a, b, c, d]
        #       [c, d, f]
        # =     [c, d] == [c, d]
        #
        #
        # [a, b, c, d]
        #    [c, d, f]
        #
        # =  [b, c, d] == [c, d, f]
        #
        # [a, b, c, d]
        # [c, d, f]
        #
        # [a, b, c] == [c, d, f]
        #
        # [a, b, c, d]
        # [d, f]
        #
        # [a, b] == [d, f]
        #
        # [a, b, c, d]
        # [f]
        #
        # [a] == [f]
        right_length = len(right_sequence)
        for i in range(1, left_length + right_length):
            # epsilon to favor long perfect matches
            eps = i / 10000.0

            # Slightly convoluted because we don't want out of bound indices
            # This will be necessary for a small conflict resolution optimization
            # later
            left_start = max(0, left_length - i)
            left_stop = min(left_length, left_length + right_length - i)
            left = np.array(left_sequence[left_start:left_stop])

            right_start = max(0, i - left_length)
            right_stop = min(right_length, i)
            right = np.array(right_sequence[right_start:right_stop])

            # We can only match subsequences of the same size.
            if len(left) != len(right):
                raise RuntimeError(
                    "There is a bug within whisper `decode_asr` function, please report it. Dropping to prevent bad inference."
                )

            matches = np.sum(left == right)
            matching = matches / i + eps
            if matches > 1 and matching > max_:
                max_ = matching
                max_indices = (left_start, left_stop, right_start, right_stop)

        (left_start, left_stop, right_start, right_stop) = max_indices

        # This is a small conflict optimization since those sequences overlap
        # in audio.
        # We're going to give more confidence to the left sequence
        # for the left of the overlap,
        # and to the right of the sequence, for the right of the overlap
        left_mid = (left_stop + left_start) // 2
        right_mid = (right_stop + right_start) // 2
        total_sequence.extend(left_sequence[:left_mid])
        left_sequence = right_sequence[right_mid:]
        left_length = len(left_sequence)

    total_sequence.extend(left_sequence)

    return total_sequence
