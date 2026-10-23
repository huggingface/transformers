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
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, prepare_prompt_input
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_torch_available, logging
from ...utils.import_utils import requires


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


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


class CanaryProcessorKwargs(ProcessingKwargs, total=False):  # trf-ignore: TRF019
    _defaults = {
        "text_kwargs": {
            "padding": True,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
        },
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


@requires(backends=("torch",))
@auto_docstring
class CanaryProcessor(ProcessorMixin):
    valid_processor_kwargs = CanaryProcessorKwargs
    # The decoder prompt (task tokens) ends with `<|nodiarize|>`, and the transcript that follows with `<|endoftext|>`
    assistant_header = "<|nodiarize|>"
    assistant_end = "<|endoftext|>"

    def __init__(self, feature_extractor=None, tokenizer=None, chat_template=None):
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    @auto_docstring
    def __call__(
        self,
        audio: AudioInput,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        output_labels: bool = False,
        **kwargs: Unpack[CanaryProcessorKwargs],
    ) -> BatchFeature:
        r"""
        text (`str`, `list[str]`, *optional*):
            The decoder prompt(s) produced by the chat template. It is tokenized into `decoder_input_ids`.
        output_labels (`bool`, *optional*, defaults to `False`):
            Whether to return labels for training. Only the transcript (up to and including its end-of-text token) is
            kept, so that the decoder prompt and the padding are masked out of the loss. Text without the decoder
            prompt only has its padding masked.
        """

        if "return_tensors" in kwargs and kwargs["return_tensors"] != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        model_inputs = super().__call__(audio=audio, text=text, **kwargs)
        if text is not None:
            input_ids = model_inputs.pop("input_ids")
            if output_labels:
                # the decoder inputs are already right-shifted with respect to `labels`
                model_inputs["decoder_input_ids"] = input_ids[..., :-1]
                labels = self._mask_non_assistant_tokens(input_ids, input_ids.clone())[..., 1:]
                labels[labels == self.tokenizer.pad_token_id] = -100
                model_inputs["labels"] = labels
            else:
                model_inputs["decoder_input_ids"] = input_ids
        return BatchFeature(data=model_inputs, tensor_type="pt")

    def _mask_non_assistant_tokens(self, input_ids: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
        """
        Mask everything outside the assistant turns, e.g. the decoder prompt, out of the loss. Each assistant turn is
        kept from the token after `assistant_header` up to and including `assistant_end` (or the end of the
        sequence). Sequences without an assistant turn are left unchanged.
        """
        header_ids = self.tokenizer.encode(self.assistant_header, add_special_tokens=False)
        end_id = self.tokenizer.convert_tokens_to_ids(self.assistant_end)
        header_len = len(header_ids)
        for row, ids in enumerate(input_ids.tolist()):
            starts = [i + header_len for i in range(len(ids)) if ids[i : i + header_len] == header_ids]
            if not starts:
                continue
            keep = torch.zeros(len(ids), dtype=torch.bool)
            for start in starts:
                end = ids.index(end_id, start) + 1 if end_id in ids[start:] else len(ids)
                keep[start:end] = True
            labels[row, ~keep] = -100
        return labels

    def apply_transcription_request(
        self,
        audio: AudioInput | list[AudioInput],
        source_language: str | list[str] = "en",
        target_language: str | list[str] | None = None,
        punctuation: bool = True,
        transcription: str | list[str] | None = None,
        **kwargs: Unpack[CanaryProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Prepare inputs for transcription or translation without manually writing the chat template, either for
        inference or, when `transcription` is given, for training.

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
            transcription (`str` or `list[str]`, *optional*):
                Target transcript(s) (or translations) for training, one per audio. When given, each transcript is
                written as the assistant turn and `labels` are returned (see `output_labels` in
                [`~CanaryProcessor.__call__`]).
            **kwargs:
                Additional keyword arguments forwarded to [`~CanaryProcessor.apply_chat_template`].
                Processor kwargs (for example `text_kwargs`, `audio_kwargs`, ...) should be passed as
                `processor_kwargs={...}`.

        Returns:
            [`BatchFeature`]: Processor outputs ready to be passed to
            [`CanaryForConditionalGeneration.generate`], or to [`CanaryForConditionalGeneration.forward`] to compute
            the loss when `transcription` is given.
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

        is_training = transcription is not None
        if isinstance(transcription, str):
            transcription = [transcription]
        transcriptions = prepare_prompt_input(transcription, batch_size, input_name="transcription")
        if is_training:
            kwargs["processor_kwargs"] = {"output_labels": True, **kwargs.get("processor_kwargs", {})}

        conversations = []
        for source, target, audio_item, transcript in zip(
            source_languages, target_languages, audio_items, transcriptions
        ):
            content = [
                make_audio_chat_template_content(audio_item),
                {
                    "type": "text",
                    "source_language": source,
                    "target_language": target,
                    "punctuation": punctuation,
                },
            ]
            conversation = [{"role": "user", "content": content}]
            if transcript is not None:
                # the chat template expects the assistant content as a plain string
                conversation.append({"role": "assistant", "content": transcript})
            conversations.append(conversation)

        return self.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=not is_training,
            return_dict=True,
            **kwargs,
        )

    @property
    def model_input_names(self):
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return feature_extractor_input_names + ["decoder_input_ids", "labels"]


__all__ = ["CanaryProcessor"]
