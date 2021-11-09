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
from contextlib import contextmanager

from .feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
from .tokenization_wav2vec2 import Wav2Vec2CTCTokenizer


class Wav2Vec2Processor:
    r"""
    Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor and a Wav2Vec2 CTC tokenizer into a single
    processor.

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

    def __init__(self, feature_extractor, tokenizer):
        if not isinstance(feature_extractor, Wav2Vec2FeatureExtractor):
            raise ValueError(
                f"`feature_extractor` has to be of type {Wav2Vec2FeatureExtractor.__class__}, but is {type(feature_extractor)}"
            )
        if not isinstance(tokenizer, Wav2Vec2CTCTokenizer):
            raise ValueError(
                f"`tokenizer` has to be of type {Wav2Vec2CTCTokenizer.__class__}, but is {type(tokenizer)}"
            )

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.current_processor = self.feature_extractor

    def save_pretrained(self, save_directory):
        """
        Save a Wav2Vec2 feature_extractor object and Wav2Vec2 tokenizer object to the directory ``save_directory``, so
        that it can be re-loaded using the :func:`~transformers.Wav2Vec2Processor.from_pretrained` class method.

        .. note::

            This class method is simply calling
            :meth:`~transformers.feature_extraction_utils.FeatureExtractionMixin.save_pretrained` and
            :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizer.save_pretrained`. Please refer to the
            docstrings of the methods above for more information.

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
        """

        self.feature_extractor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

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
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

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

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Wav2Vec2CTCTokenizer's
        :meth:`~transformers.PreTrainedTokenizer.batch_decode`. Please refer to the docstring of this method for more
        information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Wav2Vec2CTCTokenizer's
        :meth:`~transformers.PreTrainedTokenizer.decode`. Please refer to the docstring of this method for more
        information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning
        Wav2Vec2.
        """
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor


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

    def __init__(self, feature_extractor, tokenizer, ctc_decoder):
        if not isinstance(feature_extractor, Wav2Vec2FeatureExtractor):
            raise ValueError(
                f"`feature_extractor` has to be of type {Wav2Vec2FeatureExtractor.__class__}, but is {type(feature_extractor)}"
            )
        if not isinstance(tokenizer, Wav2Vec2CTCTokenizer):
            raise ValueError(
                f"`tokenizer` has to be of type {Wav2Vec2CTCTokenizer.__class__}, but is {type(tokenizer)}"
            )

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.ctc_decoder = ctc_decoder
        self.current_processor = self.feature_extractor

    def save_pretrained(self, save_directory):
        """
        Save a Wav2Vec2 feature_extractor object and Wav2Vec2 tokenizer object to the directory ``save_directory``, so
        that it can be re-loaded using the :func:`~transformers.Wav2Vec2Processor.from_pretrained` class method.

        .. note::

            This class method is simply calling
            :meth:`~transformers.feature_extraction_utils.FeatureExtractionMixin.save_pretrained` and
            :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizer.save_pretrained`. Please refer to the
            docstrings of the methods above for more information.

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
        """

        self.feature_extractor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @staticmethod
    def _load_ctc_decoder(pretrained_model_name_or_path, vocab_dict, **kwargs):
        from pyctcdecode import Alphabet, BeamSearchDecoderCTC

        # i.) build alphabet
        # check https://github.com/kensho-technologies/pyctcdecode/blob/94dfdae1d18ad95e799286173826aec2dec9a6b2/pyctcdecode/alphabet.py#L122
        sorted_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
        vocab_labels = list(sorted_dict.keys())
        alphabet = Alphabet.build_alphabet(vocab_labels)

        # ii.) build languag model
        # different design options

        # 1) either:
        # ---------------------
        from pyctcdecode import AutoLanguageModel
        language_model = AutoLanguageModel.from_pretrained(...)
        # (this requires the following:
        # a. add `AutoLanguageModel` class in https://github.com/kensho-technologies/pyctcdecode/blob/main/pyctcdecode/language_model.py
        # b. add `.from_pretrained(...)` to `AutoLanguageModel` in kensho-technologies/pyctcdecode
        # => requires some work, but should be easy (need to discuss with pyctcdecode)

        # 2) or:
        # ---------------------
        from pyctcdecode import LanguageModel
        if self._is_ken_lm_model(pretrained_model_name_or_path):
            language_model = LanguageModel.load_from_hf_hub("...")
        elif self._is_hf_lm_model(pretrained_model_name_or_path):
            language_model = HfLanguageModel.load_from_hf_hub("...")
        # (this requires the followirg:
        # a. add `.from_pretrained(...)` class in kensho-technologies/pyctcdecode
        # => requires very little work and should be pretty easy (need to discuss with pyctcdecode)
        # b. (Future Work): add `HfLanguageModel` or `AutoLanguageModel`

        # 3) or:
        # ---------------------
        # do the whole model loading ourselves and create a `AutoLanguageModel` class in `transformers`
        # => requires fair amount of work but no need to discuss with pyctcdecode
        language_model = AutoLanguageModel.load_from_hf_hub("...")

        # iii.) Build ctc decoder
        # see: https://github.com/kensho-technologies/pyctcdecode/blob/94dfdae1d18ad95e799286173826aec2dec9a6b2/pyctcdecode/decoder.py#L181
        ctc_decoder = BeamSearchDecoderCTC(alphabet, language_model)

        return ctc_decoder

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
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        ctc_decoder = cls._load_ctc_decoder(pretrained_model_name_or_path, vocab_dict=tokenizer.get_vocab(), **kwargs)

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer, ctc_decoder=ctc_decoder)

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

    def batch_decode(self, *args, **kwargs):
        """
        # TODO (PVP): build switch so that both tokenizer and lm model can be used for decoding
        """
        return self._batch_lm_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        # TODO (PVP): build switch so that both tokenizer and lm model can be used for decoding
        """
        return self._lm_decode(*args, **kwargs)

    def _batch_lm_decode(self, logits: Union[torch.FloatTensor, tf.Tensor, jnp.ndarray]):
        array_list = [array for array in logits.numpy()]
        """
            logits are outputs of Wav2Vec2-like model
            **kwargs will be all arguments of https://github.com/kensho-technologies/pyctcdecode/blob/94dfdae1d18ad95e799286173826aec2dec9a6b2/pyctcdecode/decoder.py#L633
        """

        return self.ctc_decoder.decode_batch(array_list)

    def _lm_decode(self, logits: Union[torch.FloatTensor, tf.Tensor, jnp.ndarray], **kwargs):
        """
            logits are outputs of Wav2Vec2-like model
            **kwargs will be all arguments of https://github.com/kensho-technologies/pyctcdecode/blob/94dfdae1d18ad95e799286173826aec2dec9a6b2/pyctcdecode/decoder.py#L600
        """
        return self.ctc_decoder.decode(logits.numpy())

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning
        Wav2Vec2.
        """
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
