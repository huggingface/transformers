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

import warnings
from contextlib import contextmanager
from typing import Optional, Union

from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput
from .feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
from .tokenization_wav2vec2 import Wav2Vec2CTCTokenizer


class Wav2Vec2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


class Wav2Vec2Processor(ProcessorMixin):
    r"""
    Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor and a Wav2Vec2 CTC tokenizer into a single
    processor.

    [`Wav2Vec2Processor`] offers all the functionalities of [`Wav2Vec2FeatureExtractor`] and [`PreTrainedTokenizer`].
    See the docstring of [`~Wav2Vec2Processor.__call__`] and [`~Wav2Vec2Processor.decode`] for more information.

    Args:
        feature_extractor (`Wav2Vec2FeatureExtractor`):
            An instance of [`Wav2Vec2FeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`PreTrainedTokenizer`]):
            An instance of [`PreTrainedTokenizer`]. The tokenizer is a required input.
    """

    feature_extractor_class = "Wav2Vec2FeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except (OSError, ValueError):
            warnings.warn(
                f"Loading a tokenizer inside {cls.__name__} from a config that does not"
                " include a `tokenizer_class` attribute is deprecated and will be "
                "removed in v5. Please add `'tokenizer_class': 'Wav2Vec2CTCTokenizer'`"
                " attribute to either your `config.json` or `tokenizer_config.json` "
                "file to suppress this warning: ",
                FutureWarning,
            )

            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

            return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __call__(
        self,
        audio: Optional[AudioInput] = None,
        text: Optional[Union[str, list[str], TextInput, PreTokenizedInput]] = None,
        images=None,
        videos=None,
        **kwargs: Unpack[Wav2Vec2ProcessorKwargs],
    ):
        """
        This method forwards all arguments to [`Wav2Vec2FeatureExtractor.__call__`] and/or
        [`PreTrainedTokenizer.__call__`] depending on the input modality and returns their outputs. If both modalities are passed, [`Wav2Vec2FeatureExtractor.__call__`] and [`PreTrainedTokenizer.__call__`] are called.

        Args:
            audio (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`, *optional*):
                An audio input is passed to [`Wav2Vec2FeatureExtractor.__call__`].
            text (`str`, `List[str]`, *optional*):
                A text input is passed to [`PreTrainedTokenizer.__call__`].


        Returns:
            This method returns the results of each `call` method. If both are used, the output is a dictionary containing the results of both.
        """
        if "raw_speech" in kwargs:
            warnings.warn("Using `raw_speech` as a keyword argument is deprecated. Use `audio` instead.")
            audio = kwargs.pop("raw_speech")

        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        output_kwargs = self._merge_kwargs(
            Wav2Vec2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(
                audio,
                **output_kwargs["audio_kwargs"],
                **output_kwargs["text_kwargs"],
                **output_kwargs["common_kwargs"],
            )

        if audio is not None:
            inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])

        if text is None:
            return inputs
        elif audio is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def pad(self, *args, **kwargs):
        """
        This method operates on batches of extracted features and/or tokenized text. It forwards all arguments to
        [`Wav2Vec2FeatureExtractor.pad`] and/or [`PreTrainedTokenizer.pad`] depending on the input modality and returns their outputs. If both modalities are passed, [`Wav2Vec2FeatureExtractor.pad`] and [`PreTrainedTokenizer.pad`] are called.

        Args:
            input_features:
                When the first argument is a dictionary containing a batch of tensors, or the `input_features` argument is present, it is passed to [`Wav2Vec2FeatureExtractor.pad`].
            labels:
                When the `label` argument is present, it is passed to [`PreTrainedTokenizer.pad`].

        Returns:
            This method returns the results of each `pad` method. If both are used, the output is a dictionary containing the results of both.
        """
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor.pad(*args, **kwargs)

        input_features = kwargs.pop("input_features", None)
        labels = kwargs.pop("labels", None)
        if len(args) > 0:
            input_features = args[0]
            args = args[1:]

        if input_features is not None:
            input_features = self.feature_extractor.pad(input_features, *args, **kwargs)
        if labels is not None:
            labels = self.tokenizer.pad(labels, **kwargs)

        if labels is None:
            return input_features
        elif input_features is None:
            return labels
        else:
            input_features["labels"] = labels["input_ids"]
            return input_features

    @property
    def model_input_names(self):
        # The processor doesn't return text ids and the model seems to not need them
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return feature_extractor_input_names + ["labels"]

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning
        Wav2Vec2.
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


__all__ = ["Wav2Vec2Processor"]
