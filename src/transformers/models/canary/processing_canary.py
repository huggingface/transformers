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

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class CanaryProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": "longest",
            "return_attention_mask": True,
        },
        "text_kwargs": {
            "add_special_tokens": False,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


@auto_docstring
class CanaryProcessor(ProcessorMixin):
    r"""
    Constructs a Canary processor which wraps a [`ParakeetFeatureExtractor`] and a [`TokenizersBackend`] tokenizer and
    builds the multitask decoder prompt (the `canary2` format).
    """

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    def _build_prompt_tokens(self, source_lang: str, target_lang: str, pnc: bool, timestamps: bool) -> list[str]:
        # canary2 prompt: emotion/itn/diarize slots are fixed to the trained defaults; only language/pnc/timestamps are exposed.
        return [
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            f"<|{source_lang}|>",
            f"<|{target_lang}|>",
            "<|pnc|>" if pnc else "<|nopnc|>",
            "<|noitn|>",
            "<|timestamp|>" if timestamps else "<|notimestamp|>",
            "<|nodiarize|>",
        ]

    @auto_docstring
    def __call__(
        self,
        audio: AudioInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        source_lang: str = "en",
        target_lang: str | None = None,
        pnc: bool = True,
        timestamps: bool = False,
        sampling_rate: int | None = None,
        **kwargs: Unpack[CanaryProcessorKwargs],
    ) -> BatchFeature:
        r"""
        source_lang (`str`, *optional*, defaults to `"en"`):
            The ISO language code of the input speech (e.g. `"en"`, `"de"`, `"fr"`).
        target_lang (`str`, *optional*):
            The ISO language code of the output text. Defaults to `source_lang` (transcription); set it to a different
            language for speech-to-text translation.
        pnc (`bool`, *optional*, defaults to `True`):
            Whether to request punctuation and capitalization in the output.
        timestamps (`bool`, *optional*, defaults to `False`):
            Whether to request segment/word-level timestamp tokens in the output.
        sampling_rate (`int`, *optional*):
            The sampling rate of the input `audio` in Hz. If provided, it is validated against the sampling rate
            expected by the feature extractor (16000 Hz).
        """
        if audio is None:
            raise ValueError("`audio` is required for the Canary processor.")
        if target_lang is None:
            target_lang = source_lang

        output_kwargs = self._merge_kwargs(
            CanaryProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if sampling_rate is not None and sampling_rate != output_kwargs["audio_kwargs"]["sampling_rate"]:
            raise ValueError(
                f"The sampling rate of the audio ({sampling_rate}) does not match the sampling rate expected by the "
                f"processor ({output_kwargs['audio_kwargs']['sampling_rate']}). Please resample the audio."
            )

        audio = make_list_of_audio(audio)
        inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])

        prompt_tokens = self._build_prompt_tokens(source_lang, target_lang, pnc, timestamps)
        prompt_ids = self.tokenizer.convert_tokens_to_ids(prompt_tokens)
        batch_size = len(inputs["input_features"])
        return_tensors = output_kwargs["audio_kwargs"].get("return_tensors")
        inputs["decoder_input_ids"] = [list(prompt_ids) for _ in range(batch_size)]

        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])
            inputs["labels"] = encodings["input_ids"]

        return BatchFeature(data=dict(inputs), tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return feature_extractor_input_names + ["decoder_input_ids", "labels"]


__all__ = ["CanaryProcessor"]
