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

from ...audio_utils import AudioInput, make_list_of_audio, make_list_of_audio_chat_template
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


def _audio_content_item(audio_item) -> dict:
    """Build a chat-template content dict for a single audio item."""
    if isinstance(audio_item, str):
        return {"type": "audio", "path": audio_item}
    return {"type": "audio", "audio": audio_item}


def _prepare_language_inputs(language: str | list[str] | None, batch_size: int) -> list[str | None]:
    """Broadcast / validate a language argument to match batch_size."""
    if language is None:
        return [None] * batch_size
    if isinstance(language, str):
        return [language] * batch_size
    if isinstance(language, (list, tuple)):
        if len(language) != batch_size:
            raise ValueError(f"Got {len(language)} language(s) for {batch_size} sample(s); counts must match.")
        return list(language)
    raise TypeError("`language` must be a string, a list of strings, or `None`.")


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
    Constructs a Canary processor which wraps a [`ParakeetFeatureExtractor`] and a [`TokenizersBackend`] tokenizer.

    The multitask decoder prompt (the `canary2` format) is produced by a chat template through
    [`~CanaryProcessor.apply_transcription_request`]; [`~CanaryProcessor.__call__`] only runs the feature extractor on
    the audio and tokenizes the resulting prompt into `decoder_input_ids`.
    """

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

        output_kwargs = self._merge_kwargs(
            CanaryProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        return_tensors = output_kwargs["audio_kwargs"].get("return_tensors")

        audio = make_list_of_audio(audio)
        data = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])

        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])
            data["decoder_input_ids"] = encodings["input_ids"]
            if output_labels:
                data["labels"] = encodings["input_ids"]

        return BatchFeature(data=data, tensor_type=return_tensors)

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
                The ISO language code of the input speech (e.g. `"en"`, `"de"`, `"fr"`).
            target_language (`str` or `list[str]`, *optional*):
                The ISO language code of the output text. Defaults to `source_language` (transcription); set it to a
                different language for speech-to-text translation.
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

        source_languages = _prepare_language_inputs(source_language, batch_size)
        if target_language is None:
            target_languages = list(source_languages)
        else:
            target_languages = _prepare_language_inputs(target_language, batch_size)

        conversations = []
        for source, target, audio_item in zip(source_languages, target_languages, audio_items):
            content = [
                _audio_content_item(audio_item),
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

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return feature_extractor_input_names + ["decoder_input_ids", "labels"]


__all__ = ["CanaryProcessor"]
