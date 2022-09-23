"""
Vision-Text processor class for TrOCR
"""
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import get_context
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Union

import numpy as np

from ...processing_utils import ProcessorMixin
from ...utils import ModelOutput, requires_backends


if TYPE_CHECKING:
    from pyctcdecode import BeamSearchDecoderCTC
    from ...feature_extraction_utils import FeatureExtractionMixin
    from ...tokenization_utils import PreTrainedTokenizerBase


ListOfDict = List[Dict[str, Union[int, str]]]


@dataclass
class TrOCRDecoderWithLMOutput(ModelOutput):
    """
    Output type of [`TrOCRDecoderWithLMOutput`], with transcription.
    Args:
        text (list of `str` or `str`):
            Decoded logits in text from. Usually the speech transcription.
        logit_score (list of `float` or `float`):
            Total logit score of the beam associated with produced text.
        lm_score (list of `float`):
            Fused lm_score of the beam associated with produced text.
        word_offsets (list of `List[Dict[str, Union[int, str]]]` or `List[Dict[str, Union[int, str]]]`):
            Offsets of the decoded words. In combination with sampling rate and model downsampling rate word offsets
            can be used to compute time stamps for each word.
    """

    text: Union[List[str], str]
    logit_score: Union[List[float], float] = None
    lm_score: Union[List[float], float] = None
    word_offsets: Union[List[ListOfDict], ListOfDict] = None


class TrOCRProcessorWithLM(ProcessorMixin):
    r"""
    Constructs a TrOCR processor which wraps a TrOCR feature extractor, a TrOCR tokenizer and a decoder
    with language model support into a single processor for language model boosted speech recognition decoding.
    Args:
        feature_extractor ([`AutoFeatureExtractor`]):
            An instance of [`AutoFeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`AutoTokenizer`]):
            An instance of [`AutoTokenizer`]. The tokenizer is a required input.
        decoder (`pyctcdecode.BeamSearchDecoderCTC`):
            An instance of [`pyctcdecode.BeamSearchDecoderCTC`]. The decoder is a required input.
    """
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor: "FeatureExtractionMixin",
        tokenizer: "PreTrainedTokenizerBase",
        decoder: "BeamSearchDecoderCTC",
    ):
        from pyctcdecode import BeamSearchDecoderCTC

        super().__init__(feature_extractor, tokenizer)
        if not isinstance(decoder, BeamSearchDecoderCTC):
            raise ValueError(f"`decoder` has to be of type {BeamSearchDecoderCTC.__class__}, but is {type(decoder)}")

        # make sure that decoder's alphabet and tokenizer's vocab match in content
        missing_decoder_tokens = self.get_missing_alphabet_tokens(decoder, tokenizer)
        if len(missing_decoder_tokens) > 0:
            raise ValueError(
                f"The tokens {(missing_decoder_tokens)} are defined in the tokenizer's "
                "vocabulary, but not in the decoder's alphabet. "
                f"Make sure to include {(missing_decoder_tokens)} in the decoder's alphabet."
            )

        self.decoder = decoder
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    def save_pretrained(self, save_directory):
        super().save_pretrained(save_directory)
        self.decoder.save_to_dir(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate a [`TrOCRProcessorWithLM`] from a pretrained TrOCR processor.
        <Tip>
        This class method is simply calling TrOCRFeatureExtractor's
        [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`], TrOCRTokenizer's
        [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`], and
        [`pyctcdecode.BeamSearchDecoderCTC.load_from_hf_hub`].
        Please refer to the docstrings of the methods above for more information.
        </Tip>
        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:
                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~SequenceFeatureExtractor.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            **kwargs
                Additional keyword arguments passed along to both [`SequenceFeatureExtractor`] and
                [`PreTrainedTokenizer`]
        """
        requires_backends(cls, "pyctcdecode")
        from pyctcdecode import BeamSearchDecoderCTC

        feature_extractor, tokenizer = super()._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)

        if os.path.isdir(pretrained_model_name_or_path) or os.path.isfile(pretrained_model_name_or_path):
            decoder = BeamSearchDecoderCTC.load_from_dir(pretrained_model_name_or_path)
        else:
            # BeamSearchDecoderCTC has no auto class
            kwargs.pop("_from_auto", None)
            # snapshot_download has no `trust_remote_code` flag
            kwargs.pop("trust_remote_code", None)

            # make sure that only relevant filenames are downloaded
            language_model_filenames = os.path.join(BeamSearchDecoderCTC._LANGUAGE_MODEL_SERIALIZED_DIRECTORY, "*")
            alphabet_filename = BeamSearchDecoderCTC._ALPHABET_SERIALIZED_FILENAME
            allow_regex = [language_model_filenames, alphabet_filename]

            decoder = BeamSearchDecoderCTC.load_from_hf_hub(
                pretrained_model_name_or_path, allow_regex=allow_regex, **kwargs
            )

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
        from pyctcdecode.alphabet import (
            BLANK_TOKEN_PTN,
            UNK_TOKEN,
            UNK_TOKEN_PTN,
            _normalize_bpe_alphabet,
            SPECIAL_TOKEN_PTN,
        )

        # we need to make sure that all of the tokenizer's except the special tokens
        # are present in the decoder's alphabet. Retrieve missing alphabet token
        # from decoder
        tokenizer_vocab_list = list(tokenizer.get_vocab().keys())

        tokenizer_vocab_list = _normalize_bpe_alphabet(tokenizer_vocab_list)

        # replace special tokens
        for i, token in enumerate(tokenizer_vocab_list):
            if BLANK_TOKEN_PTN.match(token):
                tokenizer_vocab_list[i] = ""
            if SPECIAL_TOKEN_PTN.match(token):
                tokenizer_vocab_list[i] = ""
            if UNK_TOKEN_PTN.match(token):
                tokenizer_vocab_list[i] = UNK_TOKEN

        # are any of the extra tokens no special tokenizer tokens?
        # data = str({"labels":tokenizer_vocab_list,"is_bpe":"True"})
        # data = data.replace("\'", "\"")
        # decoder._alphabet.loads(data)

        updated_missing_tokens = set(tokenizer_vocab_list) - set(decoder._alphabet.labels)

        return updated_missing_tokens

    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to TrOCRFeatureExtractor's
        [`~TrOCRFeatureExtractor.__call__`] and returns its output. If used in the context
        [`~TrOCRProcessorWithLM.as_target_processor`] this method forwards all its arguments to
        TrOCRCTCTokenizer's [`~TrOCRCTCTokenizer.__call__`]. Please refer to the docstring of the above two
        methods for more information.
        """
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        images = kwargs.pop("images", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            images = args[0]
            args = args[1:]

        if images is None and text is None:
            raise ValueError("You need to specify either an `images` or `text` input to process.")

        if images is not None:
            inputs = self.feature_extractor(images, *args, **kwargs)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs
        elif images is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(
        self,
        logits: np.ndarray,
        num_processes: Optional[int] = None,
        beam_width: Optional[int] = None,
        beam_prune_logp: Optional[float] = None,
        token_min_logp: Optional[float] = None,
        hotwords: Optional[Iterable[str]] = None,
        hotword_weight: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        unk_score_offset: Optional[float] = None,
        lm_score_boundary: Optional[bool] = None,
        output_word_offsets: bool = False,
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
            alpha (`float`, *optional*):
                Weight for language model during shallow fusion
            beta (`float`, *optional*):
                Weight for length score adjustment of during scoring
            unk_score_offset (`float`, *optional*):
                Amount of log score offset for unknown tokens
            lm_score_boundary (`bool`, *optional*):
                Whether to have kenlm respect boundaries when scoring
            output_word_offsets (`bool`, *optional*, defaults to `False`):
                Whether or not to output word offsets. Word offsets can be used in combination with the sampling rate
                and model downsampling rate to compute the time-stamps of transcribed words.
                <Tip>
                Please take a look at the Example of [`~model.TrOCR_with_lm.processing_TrOCR_with_lm.decode`] to
                better understand how to make use of `output_word_offsets`.
                [`~model.TrOCR_with_lm.processing_TrOCR_with_lm.batch_decode`] works the same way with batched
                output.
                </Tip>
        Returns:
            [`~models.TrOCR.TrOCRDecoderWithLMOutput`] or `tuple`.
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

        # reset params at every forward call. It's just a `set` method in pyctcdecode
        self.decoder.reset_params(
            alpha=alpha, beta=beta, unk_score_offset=unk_score_offset, lm_score_boundary=lm_score_boundary
        )

        # create multiprocessing pool and list numpy arrays
        # filter out logits padding
        logits_list = [array[(array != -100.0).all(axis=-1)] for array in logits]
        pool = get_context("fork").Pool(num_processes)

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

        # clone multi-processing pool
        pool.close()

        # extract text and scores
        batch_texts, logit_scores, lm_scores, word_offsets = [], [], [], []
        for d in decoded_beams:
            batch_texts.append(d[0][0])
            logit_scores.append(d[0][-2])
            lm_scores.append(d[0][-1])
            word_offsets.append([{"word": t[0], "start_offset": t[1][0], "end_offset": t[1][1]} for t in d[0][1]])

        word_offsets = word_offsets if output_word_offsets else None

        return TrOCRDecoderWithLMOutput(
            text=batch_texts, logit_score=logit_scores, lm_score=lm_scores, word_offsets=word_offsets
        )

    def decode(
        self,
        logits: np.ndarray,
        beam_width: Optional[int] = None,
        beam_prune_logp: Optional[float] = None,
        token_min_logp: Optional[float] = None,
        hotwords: Optional[Iterable[str]] = None,
        hotword_weight: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        unk_score_offset: Optional[float] = None,
        lm_score_boundary: Optional[bool] = None,
        output_word_offsets: bool = False,
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
            alpha (`float`, *optional*):
                Weight for language model during shallow fusion
            beta (`float`, *optional*):
                Weight for length score adjustment of during scoring
            unk_score_offset (`float`, *optional*):
                Amount of log score offset for unknown tokens
            lm_score_boundary (`bool`, *optional*):
                Whether to have kenlm respect boundaries when scoring
            output_word_offsets (`bool`, *optional*, defaults to `False`):
                Whether or not to output word offsets. Word offsets can be used in combination with the sampling rate
                and model downsampling rate to compute the time-stamps of transcribed words.
                <Tip>
                Please take a look at the example of [`~models.TrOCR_with_lm.processing_TrOCR_with_lm.decode`] to
                better understand how to make use of `output_word_offsets`.
                </Tip>
        Returns:
            [`~models.TrOCR.TrOCRDecoderWithLMOutput`] or `tuple`.
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

        # reset params at every forward call. It's just a `set` method in pyctcdecode
        self.decoder.reset_params(
            alpha=alpha, beta=beta, unk_score_offset=unk_score_offset, lm_score_boundary=lm_score_boundary
        )

        # pyctcdecode
        decoded_beams = self.decoder.decode_beams(
            logits,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
            hotwords=hotwords,
            hotword_weight=hotword_weight,
        )

        word_offsets = None
        if output_word_offsets:
            word_offsets = [
                {"word": word, "start_offset": start_offset, "end_offset": end_offset}
                for word, (start_offset, end_offset) in decoded_beams[0][2]
            ]

        # more output features will be added in the future
        return TrOCRDecoderWithLMOutput(
            text=decoded_beams[0][0],
            logit_score=decoded_beams[0][-2],
            lm_score=decoded_beams[0][-1],
            word_offsets=word_offsets,
        )

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the processor for processing the target. Useful for encoding the labels when fine-tuning
        TrOCR.
        """
        warnings.warn(
            "`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your "
            "labels by using the argument `text` of the regular `__call__` method (either in the same call as "
            "your audio inputs, or in a separate call."
        )
        self._in_target_context_manager = True
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False
