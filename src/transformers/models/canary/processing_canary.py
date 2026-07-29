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

from ...audio_utils import (
    AudioInput,
    make_audio_chat_template_content,
    make_list_of_audio_chat_template,
    prepare_language_inputs,
)
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


# fmt: off
# Languages supported by Canary. See https://huggingface.co/nvidia/canary-1b-v2 for details.
LANGUAGE_CODE_TO_NAME = {
    "bg": "Bulgarian",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hu": "Hungarian",
    "it": "Italian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mt": "Maltese",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sv": "Swedish",
    "ru": "Russian",
    "uk": "Ukrainian",
}
# fmt: on


class CanaryProcessorKwargs(ProcessingKwargs, total=False):  # trf-ignore: TRF019
    _defaults = {
        "audio_kwargs": {
            "sampling_rate": 16000,
        },
    }


@auto_docstring
class CanaryProcessor(ProcessorMixin):
    r"""
    Constructs a Canary processor which wraps a [`ParakeetFeatureExtractor`] and a [`TokenizersBackend`] tokenizer.

    The multitask decoder prompt (the `canary2` format) is produced by a chat template through
    [`~CanaryProcessor.apply_transcription_request`]; [`~CanaryProcessor.__call__`] only runs the feature extractor on
    the audio and tokenizes the resulting prompt into `decoder_input_ids`.
    """

    valid_processor_kwargs = CanaryProcessorKwargs

    def __init__(self, feature_extractor=None, tokenizer=None, chat_template=None):
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    @auto_docstring
    def __call__(
        self,
        audio: AudioInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        output_labels: bool = False,
        **kwargs: Unpack[CanaryProcessorKwargs],
    ) -> BatchFeature:
        r"""
        text (`str`, `list[str]`, *optional*):
            The decoder prompt(s) produced by the chat template. It is tokenized into `decoder_input_ids`.
        output_labels (`bool`, *optional*, defaults to `False`):
            Whether to also return the tokenized `text` as `labels` for training.
        """
        if audio is None:
            raise ValueError("You need to specify an `audio` input to process.")

        # Check only if passed explicitly as another value since by default we'll use `pt`
        if "return_tensors" in kwargs and kwargs["return_tensors"] != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        model_inputs = super().__call__(audio=audio, text=text, **kwargs)
        if text is not None:
            model_inputs["decoder_input_ids"] = model_inputs.pop("input_ids")
            if output_labels:
                model_inputs["labels"] = model_inputs["decoder_input_ids"]
        return BatchFeature(data=model_inputs, tensor_type="pt")

    def apply_transcription_request(
        self,
        audio: AudioInput | list[AudioInput],
        source_language: str | list[str] = "en",
        target_language: str | list[str] | None = None,
        punctuation: bool = True,
        **kwargs: Unpack[CanaryProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Prepare inputs for transcription or translation without manually writing the chat template.

        Args:
            audio (`AudioInput` or `list[AudioInput]`):
                Audio to transcribe or translate. Can be a URL string, local path, numpy array, or a list of these.
            source_language (`str` or `list[str]`, *optional*, defaults to `"en"`):
                The language of the input speech. Accepts ISO codes (e.g. `"en"`, `"de"`, `"fr"`) or full names
                (e.g. `"English"`, `"German"`, `"French"`).
            target_language (`str` or `list[str]`, *optional*):
                The language of the output text. Accepts ISO codes or full names. Defaults to `source_language`
                (transcription); set it to a different language for speech-to-text translation.
            punctuation (`bool`, *optional*, defaults to `True`):
                Whether to request punctuation and capitalization in the output.
            **kwargs:
                Additional keyword arguments forwarded to [`~CanaryProcessor.apply_chat_template`].

        Returns:
            [`BatchFeature`]: Processor outputs ready to be passed to
            [`CanaryForConditionalGeneration.generate`].
        """
        audio_items = make_list_of_audio_chat_template(audio)
        batch_size = len(audio_items)
        if batch_size == 0:
            raise ValueError("`audio` must contain at least one sample.")

        source_languages = prepare_language_inputs(source_language, batch_size, LANGUAGE_CODE_TO_NAME)
        if target_language is None:
            target_languages = list(source_languages)
        else:
            target_languages = prepare_language_inputs(target_language, batch_size, LANGUAGE_CODE_TO_NAME)

        conversations = []
        for source, target, audio_item in zip(source_languages, target_languages, audio_items):
            content = [
                make_audio_chat_template_content(audio_item),
                {
                    "type": "text",
                    "source_language": source,
                    "target_language": target,
                    "punctuation": punctuation,
                },
            ]
            conversations.append([{"role": "user", "content": content}])

        return self.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            **kwargs,
        )

    @property
    def model_input_names(self):
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return feature_extractor_input_names + ["decoder_input_ids", "labels"]


__all__ = ["CanaryProcessor"]
