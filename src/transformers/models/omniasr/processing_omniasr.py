# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Processor class for OmniASR."""

from typing import Optional, Union

import torch

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging


logger = logging.get_logger(__name__)


class OmniASRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": False,
            "return_attention_mask": True,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class OmniASRProcessor(ProcessorMixin):
    r"""
    Constructs an OmniASR processor which wraps an OmniASR feature extractor and a tokenizer into a single processor.

    [`OmniASRProcessor`] offers all the functionalities of [`OmniASRFeatureExtractor`] and [`PreTrainedTokenizer`].
    See the docstring of [`~OmniASRProcessor.__call__`] and [`~OmniASRProcessor.decode`] for more information.

    Args:
        feature_extractor (`OmniASRFeatureExtractor`):
            An instance of [`OmniASRFeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`PreTrainedTokenizer`]):
            An instance of [`PreTrainedTokenizer`]. The tokenizer is a required input.
        language_mapping (`dict`, *optional*):
            A dictionary mapping language codes (e.g., `"eng_latn"`) to integer language IDs used by the
            LLM variant of OmniASR. This is stored in the model config as `language_mapping`.
    """

    feature_extractor_class = "OmniASRFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor, tokenizer, language_mapping=None):
        super().__init__(feature_extractor, tokenizer)
        self.language_mapping = language_mapping

    def __call__(
        self,
        audio=None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput], None] = None,
        language: Optional[Union[str, list[str]]] = None,
        sampling_rate: Optional[int] = None,
        **kwargs: Unpack[OmniASRProcessorKwargs],
    ):
        """
        Processes audio input and optionally text/language for OmniASR models.

        For the CTC variant, pass `audio` (and optionally `text` for training labels).
        For the LLM variant, pass `audio` and `language` (e.g., `["eng_Latn"]`).

        Args:
            audio (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`, *optional*):
                The audio input, passed to the feature extractor.
            text (`str`, `list[str]`, *optional*):
                Text input, passed to the tokenizer (used for training labels).
            language (`str` or `list[str]`, *optional*):
                Language code(s) for the LLM variant (e.g., `"eng_Latn"` or `["eng_Latn", "fra_Latn"]`).
                Converted to integer `language_ids` using `language_mapping`. When not provided,
                the model defaults to a language-agnostic mode (language ID 0). Providing language
                codes is recommended for better transcription quality.
            sampling_rate (`int`, *optional*):
                The sampling rate of the audio input. Will warn if not provided.

        Returns:
            [`BatchFeature`]: A dictionary-like object with `input_values` and optionally
            `attention_mask`, `language_ids`, and `labels`.
        """
        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        output_kwargs = self._merge_kwargs(
            OmniASRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if sampling_rate is not None and sampling_rate != self.feature_extractor.sampling_rate:
            raise ValueError(
                f"The sampling rate of the audio ({sampling_rate}) does not match the expected sampling rate "
                f"({self.feature_extractor.sampling_rate}). Please resample the audio."
            )

        inputs = BatchFeature()
        if audio is not None:
            inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])

        if language is not None:
            if self.language_mapping is None:
                raise ValueError(
                    "Cannot convert language codes to IDs: `language_mapping` was not provided to the processor. "
                    "This is needed for the LLM variant of OmniASR."
                )
            if isinstance(language, str):
                language = [language]
            language_ids = []
            for lang in language:
                lang_lower = lang.lower()
                if lang_lower in self.language_mapping:
                    language_ids.append(self.language_mapping[lang_lower])
                else:
                    raise ValueError(
                        f"Language '{lang}' not found in language_mapping. "
                        f"Available languages: {list(self.language_mapping.keys())[:10]}... "
                        f"({len(self.language_mapping)} total)"
                    )
            inputs["language_ids"] = torch.tensor(language_ids, dtype=torch.long)

        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])
            inputs["labels"] = encodings["input_ids"]

        return inputs

    def batch_decode(self, *args, **kwargs):
        if self.language_mapping is not None:
            # for LLM variant, ensure that `group_tokens=False` is set to preserve repeated tokens
            kwargs.setdefault("group_tokens", False)
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if self.language_mapping is not None:
            # for LLM variant, ensure that `group_tokens=False` is set to preserve repeated tokens
            kwargs.setdefault("group_tokens", False)
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return feature_extractor_input_names + ["language_ids", "labels"]


__all__ = ["OmniASRProcessor"]
