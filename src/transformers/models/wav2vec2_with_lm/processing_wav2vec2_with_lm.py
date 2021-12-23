# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""
Speech processor class for Wav2Vec2
"""
import os
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import Pool
from typing import TYPE_CHECKING, Iterable, List, Optional, Union

import numpy as np

from ...feature_extraction_utils import FeatureExtractionMixin
from ...file_utils import ModelOutput, requires_backends
from ...tokenization_utils import PreTrainedTokenizer
from ..wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
from ..wav2vec2.tokenization_wav2vec2 import Wav2Vec2CTCTokenizer


if TYPE_CHECKING:
    from pyctcdecode import BeamSearchDecoderCTC


@dataclass
class Wav2Vec2DecoderWithLMOutput(ModelOutput):
    """
    Output type of [`Wav2Vec2DecoderWithLM`], with transcription.

    Args:
        text (list of `str`):
            Decoded logits in text from. Usually the speech transcription.
    """

    text: Union[List[str], str]


class Wav2Vec2ProcessorWithLM:
    r"""
    Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor, a Wav2Vec2 CTC tokenizer and a decoder
    with language model support into a single processor for language model boosted speech recognition decoding.

    Args:
        feature_extractor ([`Wav2Vec2FeatureExtractor`]):
            An instance of [`Wav2Vec2FeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`Wav2Vec2CTCTokenizer`]):
            An instance of [`Wav2Vec2CTCTokenizer`]. The tokenizer is a required input.
        decoder (`pyctcdecode.BeamSearchDecoderCTC`):
            An instance of [`pyctcdecode.BeamSearchDecoderCTC`]. The decoder is a required input.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractionMixin,
        tokenizer: PreTrainedTokenizer,
        decoder: "BeamSearchDecoderCTC",
    ):
        from pyctcdecode import BeamSearchDecoderCTC

        if not isinstance(feature_extractor, Wav2Vec2FeatureExtractor):
            raise ValueError(
                f"`feature_extractor` has to be of type {Wav2Vec2FeatureExtractor.__class__}, but is {type(feature_extractor)}"
            )
        if not isinstance(tokenizer, Wav2Vec2CTCTokenizer):
            # TODO(PVP) - this can be relaxed in the future to allow other kinds of tokenizers
            raise ValueError(
                f"`tokenizer` has to be of type {Wav2Vec2CTCTokenizer.__class__}, but is {type(tokenizer)}"
            )
        if not isinstance(decoder, BeamSearchDecoderCTC):
            raise ValueError(f"`decoder` has to be of type {BeamSearchDecoderCTC.__class__}, but is {type(decoder)}")

        # make sure that decoder's alphabet and tokenizer's vocab match in content
        missing_decoder_tokens = self.get_missing_alphabet_tokens(decoder, tokenizer)
        if len(missing_decoder_tokens) > 0:
            raise ValueError(
                f"The tokens {missing_decoder_tokens} are defined in the tokenizer's "
                "vocabulary, but not in the decoder's alphabet. "
                f"Make sure to include {missing_decoder_tokens} in the decoder's alphabet."
            )

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.decoder = decoder
        self.current_processor = self.feature_extractor

    def save_pretrained(self, save_directory):
        """
        Save the Wav2Vec2 feature_extractor, a tokenizer object and a pyctcdecode decoder to the directory
        `save_directory`, so that they can be re-loaded using the
        [`~Wav2Vec2ProcessorWithLM.from_pretrained`] class method.

        <Tip>

        This class method is simply calling
        [`~feature_extraction_utils.FeatureExtractionMixin.save_pretrained,`]
        [`~tokenization_utils_base.PreTrainedTokenizer.save_pretrained`] and pyctcdecode's
        [`pyctcdecode.BeamSearchDecoderCTC.save_to_dir`].

        Please refer to the docstrings of the methods above for more information.

        </Tip>

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
        """
        self.feature_extractor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        self.decoder.save_to_dir(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate a [`Wav2Vec2ProcessorWithLM`] from a pretrained Wav2Vec2 processor.

        <Tip>

        This class method is simply calling Wav2Vec2FeatureExtractor's
        [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`],
        Wav2Vec2CTCTokenizer's [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`],
        and [`pyctcdecode.BeamSearchDecoderCTC.load_from_hf_hub`].

        Please refer to the docstrings of the methods above for more information.

        </Tip>

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~SequenceFeatureExtractor.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - a path or url to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            **kwargs
                Additional keyword arguments passed along to both [`SequenceFeatureExtractor`] and
                [`PreTrainedTokenizer`]
        """
        requires_backends(cls, "pyctcdecode")
        from pyctcdecode import BeamSearchDecoderCTC

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        if os.path.isdir(pretrained_model_name_or_path):
            decoder = BeamSearchDecoderCTC.load_from_dir(pretrained_model_name_or_path)
        else:
            # BeamSearchDecoderCTC has no auto class
            kwargs.pop("_from_auto", None)

            decoder = BeamSearchDecoderCTC.load_from_hf_hub(pretrained_model_name_or_path, **kwargs)

        # set language model attributes
        for attribute in ["alpha", "beta", "unk_score_offset", "score_boundary"]:
            value = kwargs.pop(attribute, None)

            if value is not None:
                cls._set_language_model_attribute(decoder, attribute, value)

        # make sure that decoder's alphabet and tokenizer's vocab match in content
        missing_decoder_tokens = cls.get_missing_alphabet_tokens(decoder, tokenizer)
        if len(missing_decoder_tokens) > 0:
            raise ValueError(
                f"The tokens {missing_decoder_tokens} are defined in the tokenizer's "
                "vocabulary, but not in the decoder's alphabet. "
                f"Make sure to include {missing_decoder_tokens} in the decoder's alphabet."
            )

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer, decoder=decoder)

    @staticmethod
    def _set_language_model_attribute(decoder: "BeamSearchDecoderCTC", attribute: str, value: float):
        setattr(decoder.model_container[decoder._model_key], attribute, value)

    @property
    def language_model(self):
        return self.decoder.model_container[self.decoder._model_key]

    @staticmethod
    def get_missing_alphabet_tokens(decoder, tokenizer):
        from pyctcdecode.alphabet import BLANK_TOKEN_PTN, UNK_TOKEN, UNK_TOKEN_PTN

        # we need to make sure that all of the tokenizer's except the special tokens
        # are present in the decoder's alphabet. Retrieve missing alphabet token
        # from decoder
        tokenizer_vocab_list = list(tokenizer.get_vocab().keys())

        # replace special tokens
        for i, token in enumerate(tokenizer_vocab_list):
            if BLANK_TOKEN_PTN.match(token):
                tokenizer_vocab_list[i] = ""
            if token == tokenizer.word_delimiter_token:
                tokenizer_vocab_list[i] = " "
            if UNK_TOKEN_PTN.match(token):
                tokenizer_vocab_list[i] = UNK_TOKEN

        # are any of the extra tokens no special tokenizer tokens?
        missing_tokens = set(tokenizer_vocab_list) - set(decoder._alphabet.labels)

        return missing_tokens

    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor's
        [`~Wav2Vec2FeatureExtractor.__call__`] and returns its output. If used in the context
        [`~Wav2Vec2ProcessorWithLM.as_target_processor`] this method forwards all its arguments to
        Wav2Vec2CTCTokenizer's [`~Wav2Vec2CTCTokenizer.__call__`]. Please refer to the docstring of
        the above two methods for more information.
        """
        return self.current_processor(*args, **kwargs)

    def pad(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor's
        [`~Wav2Vec2FeatureExtractor.pad`] and returns its output. If used in the context
        [`~Wav2Vec2ProcessorWithLM.as_target_processor`] this method forwards all its arguments to
        Wav2Vec2CTCTokenizer's [`~Wav2Vec2CTCTokenizer.pad`]. Please refer to the docstring of the
        above two methods for more information.
        """
        return self.current_processor.pad(*args, **kwargs)

    def batch_decode(
        self,
        logits: np.ndarray,
        num_processes: Optional[int] = None,
        beam_width: Optional[int] = None,
        beam_prune_logp: Optional[float] = None,
        token_min_logp: Optional[float] = None,
        hotwords: Optional[Iterable[str]] = None,
        hotword_weight: Optional[float] = None,
    ):
        """
        Batch decode output logits to audio transcription with language model support.

        <Tip>

        This function makes use of Python's multiprocessing.

        </Tip>

        Args:
            logits (`np.ndarray`):
                The logits output vector of the model representing the log probabilities for each token.
            num_processes (`int`, *optional*):
                Number of processes on which the function should be parallelized over. Defaults to the number of
                available CPUs.
            beam_width (`int`, *optional*):
                Maximum number of beams at each step in decoding. Defaults to pyctcdecode's DEFAULT_BEAM_WIDTH.
            beam_prune_logp (`int`, *optional*):
                Beams that are much worse than best beam will be pruned Defaults to pyctcdecode's DEFAULT_PRUNE_LOGP.
            token_min_logp (`int`, *optional*):
                Tokens below this logp are skipped unless they are argmax of frame Defaults to pyctcdecode's
                DEFAULT_MIN_TOKEN_LOGP.
            hotwords (`List[str]`, *optional*):
                List of words with extra importance, can be OOV for LM
            hotword_weight (`int`, *optional*):
                Weight factor for hotword importance Defaults to pyctcdecode's DEFAULT_HOTWORD_WEIGHT.

        Returns:
            [`~models.wav2vec2.Wav2Vec2DecoderWithLMOutput`] or `tuple`.

        """
        from pyctcdecode.constants import (
            DEFAULT_BEAM_WIDTH,
            DEFAULT_HOTWORD_WEIGHT,
            DEFAULT_MIN_TOKEN_LOGP,
            DEFAULT_PRUNE_LOGP,
        )

        # set defaults
        beam_width = beam_width if beam_width is not None else DEFAULT_BEAM_WIDTH
        beam_prune_logp = beam_prune_logp if beam_prune_logp is not None else DEFAULT_PRUNE_LOGP
        token_min_logp = token_min_logp if token_min_logp is not None else DEFAULT_MIN_TOKEN_LOGP
        hotword_weight = hotword_weight if hotword_weight is not None else DEFAULT_HOTWORD_WEIGHT

        # create multiprocessing pool and list numpy arrays
        logits_list = [array for array in logits]
        pool = Pool(num_processes)

        # pyctcdecode
        decoded_beams = self.decoder.decode_beams_batch(
            pool,
            logits_list=logits_list,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
            hotwords=hotwords,
            hotword_weight=hotword_weight,
        )

        # extract text
        batch_texts = [d[0][0] for d in decoded_beams]

        # more output features will be added in the future
        return Wav2Vec2DecoderWithLMOutput(text=batch_texts)

    def decode(
        self,
        logits: np.ndarray,
        beam_width: Optional[int] = None,
        beam_prune_logp: Optional[float] = None,
        token_min_logp: Optional[float] = None,
        hotwords: Optional[Iterable[str]] = None,
        hotword_weight: Optional[float] = None,
    ):
        """
        Decode output logits to audio transcription with language model support.

        Args:
            logits (`np.ndarray`):
                The logits output vector of the model representing the log probabilities for each token.
            beam_width (`int`, *optional*):
                Maximum number of beams at each step in decoding. Defaults to pyctcdecode's DEFAULT_BEAM_WIDTH.
            beam_prune_logp (`int`, *optional*):
                A threshold to prune beams with log-probs less than best_beam_logp + beam_prune_logp. The value should
                be <= 0. Defaults to pyctcdecode's DEFAULT_PRUNE_LOGP.
            token_min_logp (`int`, *optional*):
                Tokens with log-probs below token_min_logp are skipped unless they are have the maximum log-prob for an
                utterance. Defaults to pyctcdecode's DEFAULT_MIN_TOKEN_LOGP.
            hotwords (`List[str]`, *optional*):
                List of words with extra importance which can be missing from the LM's vocabulary, e.g. ["huggingface"]
            hotword_weight (`int`, *optional*):
                Weight multiplier that boosts hotword scores. Defaults to pyctcdecode's DEFAULT_HOTWORD_WEIGHT.

        Returns:
            [`~models.wav2vec2.Wav2Vec2DecoderWithLMOutput`] or `tuple`.

        """
        from pyctcdecode.constants import (
            DEFAULT_BEAM_WIDTH,
            DEFAULT_HOTWORD_WEIGHT,
            DEFAULT_MIN_TOKEN_LOGP,
            DEFAULT_PRUNE_LOGP,
        )

        # set defaults
        beam_width = beam_width if beam_width is not None else DEFAULT_BEAM_WIDTH
        beam_prune_logp = beam_prune_logp if beam_prune_logp is not None else DEFAULT_PRUNE_LOGP
        token_min_logp = token_min_logp if token_min_logp is not None else DEFAULT_MIN_TOKEN_LOGP
        hotword_weight = hotword_weight if hotword_weight is not None else DEFAULT_HOTWORD_WEIGHT

        # pyctcdecode
        decoded_beams = self.decoder.decode_beams(
            logits,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
            hotwords=hotwords,
            hotword_weight=hotword_weight,
        )

        # more output features will be added in the future
        return Wav2Vec2DecoderWithLMOutput(text=decoded_beams[0][0])

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning
        Wav2Vec2.
        """
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
