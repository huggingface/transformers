# coding=utf-8
# Copyright 2020 Google and The HuggingFace Inc. team.
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
"""Tokenization class for model PEGASUS."""


# ಕೋಡಿಂಗ್=utf-8
# ಕೃತಿಸ್ವಾಮ್ಯ 2020 ಗೂಗಲ್ ಮತ್ತು ಹಗಿಂಗ್ ಫೇಸ್ ಇಂಕ್ ತಂಡ.
#
# ಈ ಫೈಲನ್ನು ಅನುಮೋದಿತ ಪರವಾನಗಿಯ ಅಡಿಯಲ್ಲಿ ಮಾತ್ರ ಬಳಸಬಹುದು
# ಪರವಾನಗಿಯ ಪ್ರತಿಯನ್ನು ನೀವು ಈ ಲಿಂಕ್‌ನಲ್ಲಿ ಪಡೆಯಬಹುದು:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ತಿಳುವಳಿಕೆ ಅಥವಾ ಬರವಣಿಗೆಯಲ್ಲಿ ಒಪ್ಪಿಗೆಯಾದ ಹೊರತು ಈ ಸಾಫ್ಟ್‌ವೇರ್
# ಪರವಾನಗಿಯ ಅಡಿಯಲ್ಲಿ ವಿತರಿಸಲ್ಪಡುತ್ತದೆ "ಯಥಾಸ್ಥಿತಿ" ಆಧಾರದಲ್ಲಿ,
# ಯಾವುದೇ ಖಾತರಿ ಅಥವಾ ಷರತ್ತುಗಳಿಲ್ಲದೆ. ಅನುಮತಿಗಳು ಮತ್ತು
# ಪರವಾನಗಿಯ ನಿಯಮಗಳನ್ನು ನಿಯಂತ್ರಿಸುವ ಕಾನೂನುಗಳ ಪ್ರಕಾರ ನೋಡಿ.
"""ಪೆಗಾಸಸ್ ಮಾದರಿಗೆ ಟೋಕನೈಜೇಶನ್ ವರ್ಗ."""



import os
from shutil import copyfile
from typing import List, Optional, Tuple

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging


if is_sentencepiece_available():
    from .tokenization_pegasus import PegasusTokenizer
else:
    PegasusTokenizer = None


logger = logging.get_logger(__name__)


SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}


class PegasusTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" PEGASUS tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        mask_token (`str`, *optional*, defaults to `"<mask_2>"`):
            The token used for masking single token values. This is the token used when training this model with masked
            language modeling (MLM). This is the token that the PEGASUS encoder will try to predict during pretraining.
            It corresponds to *[MASK2]* in [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive
            Summarization](https://arxiv.org/pdf/1912.08777.pdf).
        mask_token_sent (`str`, *optional*, defaults to `"<mask_1>"`):
            The token used for masking whole target sentences. This is the token used when training this model with gap
            sentences generation (GSG). This is the sentence that the PEGASUS decoder will try to predict during
            pretraining. It corresponds to *[MASK1]* in [PEGASUS: Pre-training with Extracted Gap-sentences for
            Abstractive Summarization](https://arxiv.org/pdf/1912.08777.pdf).
        additional_special_tokens (`List[str]`, *optional*):
            Additional special tokens used by the tokenizer. If no additional_special_tokens are provided <mask_2> and
            <unk_2, ..., unk_102> are used as additional special tokens corresponding to the [original PEGASUS
            tokenizer](https://github.com/google-research/pegasus/blob/939830367bcf411193d2b5eca2f2f90f3f9260ca/pegasus/ops/pretrain_parsing_ops.cc#L66)
            that uses the tokens 2 - 104 only for pretraining
    """


    r"""
    "ಫಾಸ್ಟ್" ಪೆಗಾಸಸ್ ಟೋಕನೈಜರ್ ನಿರ್ಮಿಸು (ಹಗಿಂಗ್ ಫೇಸ್ನ ಟೋಕನೈಜರ್ಸ್ ಗ್ರಂಥಾಲಯದ ಬೆಂಬಲದೊಂದಿಗೆ). ಆಧಾರಿತವಾಗಿದೆ
    [ಯುನಿಗ್ರಾಮ್](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

    ಈ ಟೋಕನೈಜರ್ ಈ ಮೇಲ್ವಿಚಾರಕ ವರ್ಗದಿಂದ ಪಾರಂಪರಿಕ ವಿಧಾನಗಳನ್ನು ಹೊಂದಿದೆ [`PreTrainedTokenizerFast`]. ಬಳಕೆದಾರರು ಈ
    ಮುಖ್ಯ ವಿಧಾನಗಳಿಗಾಗಿ ಈ ಮೇಲ್ವಿಚಾರಕ ವರ್ಗವನ್ನು ಉಲ್ಲೇಖಿಸಬಹುದು.

    ಹೊಂದಿಸಿದ ವಿಧಾನಗಳು:
        vocab_file (`str`):
            [ಸೆಂಟೆನ್ಸ್‌ಪೀಸ್](https://github.com/google/sentencepiece) ಫೈಲ್ (ಸಾಮಾನ್ಯವಾಗಿ *.spm* ವಿಸ್ತರಣೆಯನ್ನು ಹೊಂದಿದೆ) ಅದು
            ಟೋಕನೈಜರ್ ಉದ್ಭವಗೊಳ್ಳಲು ಅಗತ್ಯವಿರುವ ಶಬ್ದಸಂಗ್ರಹವನ್ನು ಒಳಗೊಂಡಿದೆ.
        pad_token (`str`, *ಐಚ್ಛಿಕ*, ಡೀಫಾಲ್ಟ್ ಗೆ `"<pad>"`):
            ಉದಾಹರಣೆಗೆ ವಿವಿಧ ಉದ್ದಗಳ ಸರಣಿಗಳನ್ನು ಬ್ಯಾಚ್ ಮಾಡುವಾಗ ಬಳಸುವ ಪ್ಯಾಡ್ ಟೋಕನ್.
        eos_token (`str`, *ಐಚ್ಛಿಕ*, ಡೀಫಾಲ್ಟ್ ಗೆ `"</s>"`):
            ಸರಣಿಯ ಕೊನೆಯ ಟೋಕನ್.

            <ಸಲಹೆ>

            ವಿಶೇಷ ಟೋಕನ್ಗಳನ್ನು ಬಳಸುವಾಗ ಈ ಟೋಕನ್ ಸರಣಿಯ ಕೊನೆಗೆ ಬಳಸುವ ಟೋಕನ್ ಅಲ್ಲ.
            ಬಳಸುವ ಟೋಕನ್ `sep_token`.

            </ಸಲಹೆ>

        unk_token (`str`, *ಐಚ್ಛಿಕ*, ಡೀಫಾಲ್ಟ್ ಗೆ `"<unk>"`):
            ಅಜ್ಞಾತ ಟೋಕನ್. ಶಬ್ದಸಂಗ್ರಹದಲ್ಲಿಲ್ಲದ ಟೋಕನ್ನು ID ಗೆ ಪರಿವರ್ತಿಸಲಾಗದು ಮತ್ತು ಇದನ್ನು ಈ ಟೋಕನ್ ಎಂದು ಹೊಂದಿಸಲಾಗುತ್ತದೆ.
        mask_token (`str`, *ಐಚ್ಛಿಕ*, ಡೀಫಾಲ್ಟ್ ಗೆ `"<mask_2>"`):
            ಮಾಸ್ಕ್ ಮಾಡಲು ಬಳಸುವ ಟೋಕನ್. ಇದು ಮಾಸ್ಕ್ಡ್ ಭಾಷಾ ಮಾದರಿಕರಣ (MLM) ಜೊತೆಗೆ ಈ ಮಾದರಿಯನ್ನು ತರಬೇತಿ ಮಾಡುವಾಗ ಬಳಸಲಾಗುತ್ತದೆ.
            ಇದು [ಪೆಗಾಸಸ್: ಸಾರಾಂಶಕ್ಕಾಗಿ ತೆಗೆದುಕೊಂಡ ಗ್ಯಾಪ್-ವಾಕ್ಯಗಳೊಂದಿಗೆ ಪೂರ್ವ-ತರಬೇತಿ](https://arxiv.org/pdf/1912.08777.pdf)ನಲ್ಲಿ *[MASK2]*ಗೆ ಹೊಂದಿಸಲಾಗಿದೆ.
        mask_token_sent (`str`, *ಐಚ್ಛಿಕ*, ಡೀಫಾಲ್ಟ್ ಗೆ `"<mask_1>"`):
            ಗುರಿ ವಾಕ್ಯಗಳನ್ನು ಮಾಸ್ಕ್ ಮಾಡಲು ಬಳಸುವ ಟೋಕನ್. ಇದು ಗ್ಯಾಪ್ ವಾಕ್ಯಗಳ ಉತ್ಪಾದನೆ (GSG) ಜೊತೆಗೆ ಈ ಮಾದರಿಯನ್ನು ತರಬೇತಿ ಮಾಡುವಾಗ ಬಳಸಲಾಗುತ್ತದೆ.
            ಇದು [ಪೆಗಾಸಸ್: ಸಾರಾಂಶಕ್ಕಾಗಿ ತೆಗೆದುಕೊಂಡ ಗ್ಯಾಪ್-ವಾಕ್ಯಗಳೊಂದಿಗೆ ಪೂರ್ವ-ತರಬೇತಿ](https://arxiv.org/pdf/1912.08777.pdf)ನಲ್ಲಿ *[MASK1]*ಗೆ ಹೊಂದಿಸಲಾಗಿದೆ.
        additional_special_tokens (`List[str]`, *ಐಚ್ಛಿಕ*):
            ಟೋಕನೈಜರ್ ಬಳಸುವ ಹೆಚ್ಚುವರಿ ವಿಶೇಷ ಟೋಕನ್ಗಳು. ಹೆಚ್ಚುವರಿ ವಿಶೇಷ ಟೋಕನ್ಗಳನ್ನು ಒದಗಿಸದಿದ್ದರೆ, <mask_2> ಮತ್ತು <unk_2, ..., unk_102> ಗಳನ್ನು ಬಳಸಲಾಗುತ್ತದೆ.
            ಪ್ರಿಟ್ರೇನಿಂಗ್ಗಾಗಿ ಟೋಕನ್ಗಳು 2 - 104 ಅನ್ನು ಮಾತ್ರ ಬಳಸುವ [ಮೂಲ ಪೆಗಾಸಸ್ ಟೋಕನೈಜರ್](https://github.com/google-research/pegasus/blob/939830367bcf411193d2b5eca2f2f90f3f9260ca/pegasus/ops/pretrain_parsing_ops.cc#L66) ಗೆ ಹೊಂದಿಸಲಾಗಿದೆ.
   """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = PegasusTokenizer
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        mask_token="<mask_2>",
        mask_token_sent="<mask_1>",
        additional_special_tokens=None,
        offset=103,  # entries 2 - 104 are only used for pretraining  
                     # ಪ್ರಿಟ್ರೇನಿಂಗ್ಗಾಗಿ ಮಾತ್ರ 2 - 104 ಎಂಟ್ರಿಗಳನ್ನು ಬಳಸಲಾಗುತ್ತದೆ
        **kwargs,
    ):
        self.offset = offset

        if additional_special_tokens is not None:
            if not isinstance(additional_special_tokens, list):
                raise TypeError(
                    f"additional_special_tokens should be of type {type(list)}, but is"
                    f" {type(additional_special_tokens)}"
                )

            additional_special_tokens_extended = (
                ([mask_token_sent] + additional_special_tokens)
                if mask_token_sent not in additional_special_tokens and mask_token_sent is not None
                else additional_special_tokens
            )
            # fill additional tokens with ..., <unk_token_102> in case not all additional tokens are already taken

            # ಎಲ್ಲಾ ಹೆಚ್ಚುವರಿ ಟೋಕನ್ಗಳನ್ನು ತುಂಬಿಸಲು..., <unk_token_102> ಬಳಸಲಾಗುತ್ತದೆ
            additional_special_tokens_extended += [
                f"<unk_{i}>" for i in range(len(additional_special_tokens_extended), self.offset - 1)
            ]

            if len(set(additional_special_tokens_extended)) != len(additional_special_tokens_extended):
                raise ValueError(
                    "Please make sure that the provided additional_special_tokens do not contain an incorrectly"
                    f" shifted list of <unk_x> tokens. Found {additional_special_tokens_extended}."
                )
            additional_special_tokens = additional_special_tokens_extended
        else:
            additional_special_tokens = [mask_token_sent] if mask_token_sent is not None else []
            additional_special_tokens += [f"<unk_{i}>" for i in range(2, self.offset)]

        # pegasus was design to support changing the index of the first tokens. If one of the padding/eos/unk/mask token
        # is different from default, we must rebuild the vocab

        # ಪೆಗಾಸಸ್ ಮೊದಲ ಟೋಕನ್ಗಳ ಸೂಚ್ಯಂಕವನ್ನು ಬದಲಾವಣೆ ಮಾಡಲು ವಿನ್ಯಾಸವಾಗಿದೆ. ಪ್ಯಾಡ್/ಇಓಎಸ್/ಅಂಕ್/ಮಾಸ್ಕ್ ಟೋಕನ್
        # ಡೀಫಾಲ್ಟ್ ಗಿಂತ ಬೇರೆಯಾಗಿದ್ದರೆ, ನಾವು ಶಬ್ದಸಂಗ್ರಹವನ್ನು ಮರುನಿರ್ಮಾಣ ಮಾಡಬೇಕು
        from_slow = kwargs.pop("from_slow", None)
        from_slow = from_slow or str(pad_token) != "<pad>" or str(eos_token) != "</s>" or str(unk_token) != "<unk>"

        kwargs.pop("added_tokens_decoder", {})

        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            pad_token=pad_token,
            eos_token=eos_token,
            unk_token=unk_token,
            mask_token=mask_token,
            mask_token_sent=mask_token_sent,
            offset=offset,
            additional_special_tokens=additional_special_tokens,
            from_slow=from_slow,
            **kwargs,
        )
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def _special_token_mask(self, seq):
        all_special_ids = set(self.all_special_ids)  # call it once instead of inside list comp  
                                                     # ಒಮ್ಮೆ ಕರೆಯಲಾಗುವುದು ಬದಲು ಲಿಸ್ಟ್ ಕಂಪ್ರಹೆನ್ಶನ್ನಲ್ಲಿ
        all_special_ids.remove(self.unk_token_id)  # <unk> is only sometimes special  
                                                   # ಒಮ್ಮೆ ಕರೆಯಲಾಗುವುದು ಬದಲು ಲಿಸ್ಟ್ ಕಂಪ್ರಹೆನ್ಶನ್ನಲ್ಲಿ

        if all_special_ids != set(range(len(self.additional_special_tokens) + 3)):
            raise ValueError(
                "There should be 3 special tokens: mask_token, pad_token, and eos_token +"
                f" {len(self.additional_special_tokens)} additional_special_tokens, but got {all_special_ids}"
            )

        return [1 if x in all_special_ids else 0 for x in seq]

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """Get list where entries are [1] if a token is [eos] or [pad] else 0."""
        if already_has_special_tokens:
            return self._special_token_mask(token_ids_0)
        elif token_ids_1 is None:
            return self._special_token_mask(token_ids_0) + [1]
        else:
            return self._special_token_mask(token_ids_0 + token_ids_1) + [1]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """
        Build model inputs from a sequence by adding eos to the end. no bos token is added to the front.

        - single sequence: `X </s>`
        - pair of sequences: `A B </s>` (not intended use)

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """



        """
        ಒಂದು ಸರಣಿಯಿಂದ ಮಾದರಿ ನಿರ್ಮಾಣಗಳನ್ನು ಕೊನೆಯ ಟೋಕನ್ ಸೇರಿಸುವ ಮೂಲಕ ಕಟ್ಟುವುದು. ಮುಂದಿನ ಬಾಸ್ ಟೋಕನ್ ಸೇರ್ಪಡೆಯಾಗಿಲ್ಲ.

        - ಏಕ ಸರಣಿ: `X </s>`
        - ಜೋಡಿ ಸರಣಿಗಳು: `A B </s>` (ಉದ್ದೇಶಿತ ಬಳಕೆಯಲ್ಲಿಲ್ಲ)

        ಹೊಂದಿಸಿದ ವಿಧಾನಗಳು:
            token_ids_0 (`List[int]`):
                ವಿಶೇಷ ಟೋಕನ್ಗಳನ್ನು ಸೇರಿಸಲಾಗುವ IDಗಳ ಪಟ್ಟಿ
            token_ids_1 (`List[int]`, *ಐಚ್ಛಿಕ*):
                ಜೋಡಿ ಸರಣಿಗಳಿಗಾಗಿ ಎರಡನೇ ಐಚ್ಛಿಕ IDಗಳ ಪಟ್ಟಿ.

        ಮರುಪಡೆಯುವಿಕೆಗಳು:
            `List[int]`: ಸೂಕ್ತ ವಿಶೇಷ ಟೋಕನ್ಗಳೊಂದಿಗೆ [input IDs](../glossary#input-ids) ಪಟ್ಟಿ.
        """

        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        # ಜೋಡಿ ಸರಣಿಗಳನ್ನು ಸಂಸ್ಕರಿಸುವುದನ್ನು ನಾವು ನಿರೀಕ್ಷಿಸುತ್ತಿಲ್ಲ, ಆದರೆ API ಸಮರ್ಥನೆಗಾಗಿ ಜೋಡಿ ತರ್ಕವನ್ನು ಬಿಟ್ಟಿದ್ದೇವೆ
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

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

        return (out_vocab_file,)









