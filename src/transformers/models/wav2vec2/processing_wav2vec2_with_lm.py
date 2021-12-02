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

from .feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
from .tokenization_wav2vec2 import Wav2Vec2CTCTokenizer
from ...file_utils import is_pyctcdecode_available, requires_backends
from ...feature_extraction_utils import FeatureExtractionMixin
from ...tokenization_utils import PreTrainedTokenizer


if is_pyctcdecode_available():
    from pyctcdecode import BeamSearchDecoderCTC


class Wav2Vec2ProcessorWithLM:
    r"""
    Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor, a Wav2Vec2 CTC tokenizer and a language model into a single
    processor for language model boosted speech recognition decoding.

    :class:`~transformers.Wav2Vec2Processor` offers all the functionalities of
    :class:`~transformers.Wav2Vec2FeatureExtractor` and :class:`~transformers.Wav2Vec2CTCTokenizer`. See the docstring
    of :meth:`~transformers.Wav2Vec2Processor.__call__` and :meth:`~transformers.Wav2Vec2Processor.decode` for more
    information.

    Args:
        feature_extractor (:obj:`Wav2Vec2FeatureExtractor`):
            An instance of :class:`~transformers.Wav2Vec2FeatureExtractor`. The feature extractor is a required input.
        tokenizer (:obj:`Wav2Vec2CTCTokenizer`):
            An instance of :class:`~transformers.Wav2Vec2CTCTokenizer`. The tokenizer is a required input.
    """

    def __init__(self, feature_extractor: FeatureExtractionMixin, tokenizer: PreTrainedTokenizer, decoder: BeamSearchDecoderCTC):
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
            raise ValueError(
                f"`decoder` has to be of type {BeamSearchDecoderCTC.__class__}, but is {type(decoder)}"
            )

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.decoder = decoder
        self.current_processor = self.feature_extractor

    def save_pretrained(self, save_directory):
        """
        Save a Wav2Vec2 feature_extractor object and Wav2Vec2 tokenizer object to the directory ``save_directory``, so
        that it can be re-loaded using the :func:`~transformers.Wav2Vec2Processor.from_pretrained` class method.

        .. note::

            This class method is simply calling
            :meth:`~transformers.feature_extraction_utils.FeatureExtractionMixin.save_pretrained,`
            :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizer.save_pretrained` and pyctcdecode's :meth:`BeamSearchDecoderCTC.save_to_dir`. Please refer to the docstrings of the methods above for more information.

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
        """
        self.feature_extractor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        self.decoder.save_to_dir(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate a :class:`~transformers.Wav2Vec2Processor` from a pretrained Wav2Vec2 processor.

        .. note::

            This class method is simply calling Wav2Vec2FeatureExtractor's
            :meth:`~transformers.feature_extraction_utils.FeatureExtractionMixin.from_pretrained` and
            Wav2Vec2CTCTokenizer's :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizer.from_pretrained`.
            Please refer to the docstrings of the methods above for more information.

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                This can be either:

                - a string, the `model id` of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                  namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing a feature extractor file saved using the
                  :meth:`~transformers.SequenceFeatureExtractor.save_pretrained` method, e.g.,
                  ``./my_model_directory/``.
                - a path or url to a saved feature extractor JSON `file`, e.g.,
                  ``./my_model_directory/preprocessor_config.json``.
            **kwargs
                Additional keyword arguments passed along to both :class:`~transformers.SequenceFeatureExtractor` and
                :class:`~transformers.PreTrainedTokenizer`
        """
        requires_backends(cls, "pyctcdecode")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        if os.path.isdir(pretrained_model_name_or_path):
            decoder = BeamSearchDecoderCTC.load_from_dir(pretrained_model_name_or_path)
        else:
            decoder = BeamSearchDecoderCTC.load_from_hf_hub(pretrained_model_name_or_path, **kwargs)

        # make sure that decoder's alphabet and tokenizer's vocab match
        if not cls.decoder_matches_vocab(decoder, tokenizer):
            raise ValueError("...")

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer, decoder=decoder)

    @staticmethod
    def decoder_matches_vocab(decoder, tokenizer):
        # we need to make sure that all of the tokenizer's except the special tokens
        # are present in the decoder's alphabet
        tokenizer_vocab = set([t.lower() for t in tokenizer.get_vocab().keys()])

        # special tokens consist of special_tokens_map & word delimiter token
        tokenizer_special_tokens = set(list(tokenizer.special_tokens_map.values()))
        if hasattr(tokenizer, "word_delimiter_token"):
            tokenizer_special_tokens.add(tokenizer.word_delimiter_token)

        # get tokens that are present in tokenizer, but not in decoder
        tokenizer_extra_tokens = tokenizer_vocab - set(decoder._alphabet.labels)

        import ipdb; ipdb.set_trace()
        # are any of the extra tokens no special tokenizer tokens?
        if len(tokenizer_extra_tokens - tokenizer_special_tokens) > 0:
            return False

        return True

    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor's
        :meth:`~transformers.Wav2Vec2FeatureExtractor.__call__` and returns its output. If used in the context
        :meth:`~transformers.Wav2Vec2Processor.as_target_processor` this method forwards all its arguments to
        Wav2Vec2CTCTokenizer's :meth:`~transformers.Wav2Vec2CTCTokenizer.__call__`. Please refer to the docstring of
        the above two methods for more information.
        """
        return self.current_processor(*args, **kwargs)

    def pad(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor's
        :meth:`~transformers.Wav2Vec2FeatureExtractor.pad` and returns its output. If used in the context
        :meth:`~transformers.Wav2Vec2Processor.as_target_processor` this method forwards all its arguments to
        Wav2Vec2CTCTokenizer's :meth:`~transformers.Wav2Vec2CTCTokenizer.pad`. Please refer to the docstring of the
        above two methods for more information.
        """
        return self.current_processor.pad(*args, **kwargs)

    def batch_decode(self, logits, **kwargs):
        logits_list = [array for array in logits.numpy()]
        return self.decoder.decode_batch(logits_list=logits_list, **kwargs)

    def decode(self, logits, **kwargs):
        return self.decoder.decode(logits.numpy(), **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning
        Wav2Vec2.
        """
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
