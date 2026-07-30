# Copyright 2026 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

import json
import re

import numpy as np

from ...audio_utils import AudioInput, make_list_of_audio, make_list_of_audio_chat_template
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, prepare_prompt_input
from ...tokenization_utils_base import TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class VibeVoiceAsrProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
            "add_special_tokens": False,
            "return_tensors": "pt",
        },
        "audio_kwargs": {
            "sampling_rate": 24000,
            "pad_to_multiple_of": 3200,  # tokenizer hop length
        },
        "common_kwargs": {
            "return_attention_mask": True,
        },
    }


class VibeVoiceAsrProcessor(ProcessorMixin):
    r"""
    Constructs a VibeVoice ASR processor which wraps [`VibeVoiceAcousticTokenizerFeatureExtractor`] and
    [`Qwen2TokenizerFast`] into a single processor that inherits both the audio feature extraction and
    tokenizer functionalities.

    See the [`~VibeVoiceAsrProcessor.__call__`] for more information.

    Args:
        feature_extractor (`VibeVoiceAcousticTokenizerFeatureExtractor`):
            The feature extractor for audio processing.
        tokenizer (`Qwen2TokenizerFast`):
            The tokenizer for text processing.
        chat_template (`str`, *optional*):
            A Jinja template which will be used to convert lists of messages in a chat into a tokenizable string.
        audio_token (`str`, *optional*, defaults to `"<|box_start|>"`):
            The audio token placeholder to use in the chat template.
        audio_bos_token (`str`, *optional*, defaults to `"<|object_ref_start|>"`):
            The audio begin-of-sequence token placeholder to use in the chat template.
        audio_eos_token (`str`, *optional*, defaults to `"<|object_ref_end|>"`):
            The audio end-of-sequence token placeholder to use in the chat template.
        audio_duration_token (`str`, *optional*, defaults to `"<|AUDIO_DURATION|>"`):
            The audio duration token placeholder to use in the chat template.
    """

    valid_processor_kwargs = VibeVoiceAsrProcessorKwargs
    feature_extractor_class = "VibeVoiceAcousticTokenizerFeatureExtractor"
    tokenizer_class = "Qwen2TokenizerFast"

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
        audio_token="<|box_start|>",
        audio_bos_token="<|object_ref_start|>",
        audio_eos_token="<|object_ref_end|>",
        audio_duration_token="<|AUDIO_DURATION|>",
    ):
        self.audio_token = audio_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
        self.audio_bos_token = audio_bos_token
        self.audio_bos_token_id = tokenizer.convert_tokens_to_ids(audio_bos_token)
        self.audio_eos_token = audio_eos_token
        self.audio_eos_token_id = tokenizer.convert_tokens_to_ids(audio_eos_token)
        self.audio_duration_token = audio_duration_token
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    @auto_docstring
    def __call__(
        self,
        text: TextInput | list[TextInput],
        audio: AudioInput | None = None,
        output_labels: bool | None = False,
        **kwargs: Unpack[VibeVoiceAsrProcessorKwargs],
    ) -> BatchFeature:
        r"""
        output_labels (bool, *optional*, default=False):
            Whether to return labels for training.

        Returns:
            [`BatchFeature`]: A dictionary with tokenized text (`input_ids`, `attention_mask`) and
            audio features (`input_values`, `padding_mask`).
        """
        # Check only if passed explicitly as another value since by default we'll use `pt`
        for group in (kwargs, kwargs.get("text_kwargs", {}), kwargs.get("common_kwargs", {})):
            if group.get("return_tensors", "pt") != "pt":
                raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, (list, tuple)):
            raise ValueError("text input must be a string or list of strings")

        if audio is not None:
            audio = self.feature_extractor.fetch_audio(audio, sampling_rate=self.feature_extractor.sampling_rate)
            audio = make_list_of_audio(audio)
            for example in audio:
                if example.ndim != 1:
                    raise ValueError(f"Audio should be mono, got shape: {example.shape}")

            # Replace audio duration placeholders in text
            audio_durations = iter([len(el) / self.feature_extractor.sampling_rate for el in audio])
            audio_duration_pattern = re.compile(re.escape(self.audio_duration_token))
            for i in range(len(text)):
                text[i] = audio_duration_pattern.sub(lambda _: f"{next(audio_durations):.2f}", text[i])

        model_inputs = super().__call__(text=text, audio=audio, **kwargs)

        if output_labels:
            labels = model_inputs["input_ids"].clone()
            labels[labels == self.audio_token_id] = -100
            labels[labels == self.audio_bos_token_id] = -100
            labels[labels == self.audio_eos_token_id] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            model_inputs["labels"] = labels

        return BatchFeature(data=model_inputs, tensor_type="pt", skip_tensor_conversion=self.skip_tensor_conversion)

    def _process_audio(self, audio: AudioInput, **kwargs):
        audio_inputs = self.feature_extractor(audio, **kwargs)
        audio_lengths = audio_inputs["padding_mask"].sum(dim=-1).cpu().numpy()
        audio_inputs["num_audio_tokens"] = np.ceil(audio_lengths / kwargs["pad_to_multiple_of"]).astype(int).tolist()

        audio_replacements = [self.replace_audio_token(audio_inputs, audio_idx=idx) for idx in range(len(audio))]
        return audio_inputs, audio_replacements

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        num_audio_tokens = audio_inputs["num_audio_tokens"][audio_idx]
        return self.audio_token * num_audio_tokens

    @property
    def unused_input_names(self) -> list[str]:
        "Input names returned always by subprocessors but not used in model's `forward`"
        return ["num_audio_tokens"]

    def apply_transcription_request(
        self,
        audio: str | list[str] | AudioInput,
        prompt: str | list[str] | None = None,
        **kwargs: Unpack[VibeVoiceAsrProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare inputs for automatic speech recognition without manually writing the chat template.

        Args:
            audio (`str`, `list[str]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                Audio to transcribe. Strings are interpreted as local paths or URLs and will be loaded automatically by
                the chat template loader; NumPy arrays and PyTorch tensors are forwarded directly.
            prompt (`str` or `list[str]`, *optional*):
                Custom prompt(s) to include in the user turn as extra context. A list must be the same length as the
                batch. When `None`, no additional context is provided.
            **kwargs:
                Additional keyword arguments forwarded to [`~VibeVoiceAsrProcessor.apply_chat_template`] (for example
                `text_kwargs`, `audio_kwargs`, ...).

        Returns:
            [`BatchFeature`]: Processor outputs ready to be passed to [`VibeVoiceAsrForConditionalGeneration.generate`].
        """

        audio_items: list[str | np.ndarray] = list(make_list_of_audio_chat_template(audio))

        batch_size = len(audio_items)
        if batch_size == 0:
            raise ValueError("`audio` must contain at least one sample.")

        prompts = prepare_prompt_input(prompt, batch_size, input_name="prompt")

        conversations = []
        for prompt_text, audio_item in zip(prompts, audio_items):
            content = []
            if isinstance(audio_item, str):
                content.append({"type": "audio", "path": audio_item})
            else:
                content.append({"type": "audio", "audio": audio_item})

            if prompt_text is not None:
                content.append({"type": "text", "text": prompt_text})

            conversations.append([{"role": "user", "content": content}])

        return self.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            **kwargs,
        )

    def decode(self, *args, return_format="raw", **kwargs):
        """
        Forward arguments to [`~PreTrainedTokenizer.decode`] and optionally parse the dict-like output.

        VibeVoice ASR outputs transcriptions in a dictionary-like format, e.g.:
        ```
        [
            'assistant\n[{"Start":0.0,"End":7.56,"Speaker":0,"Content":"text"}]\n',
            'assistant\n[{"Start":0,"End":5.20,"Speaker":0,"Content":"text"}]\n'
        ]
        ```

        Args:
            return_format (`str`, *optional*, defaults to `"raw"`):
                Options are:
                - `"raw"`: Return a list of raw decoded strings from the tokenizer, without any parsing.
                - `"parsed"`: Return a list of list of parsed dictionary objects for each speaker utterance with timestamps.
                - `"transcription_only"`: Return a list of extracted transcription strings.

                `skip_special_tokens` is automatically enforced (hard-set) to `True` for `"parsed"` and `"transcription_only"`.
        """
        return_types = ["raw", "parsed", "transcription_only"]
        if return_format not in return_types:
            raise ValueError(f"return_format must be one of {return_types}.")
        if return_format != "raw":
            kwargs["skip_special_tokens"] = True  # for other formats this does not make sense, we can silently ignore

        decoded = self.tokenizer.decode(*args, **kwargs)

        if return_format == "parsed":
            decoded = self.extract_speaker_dict(decoded)
        elif return_format == "transcription_only":
            decoded = self.extract_transcription(decoded)
        return decoded

    def extract_speaker_dict(self, text: str | list[str]) -> list[dict] | str | list[list[dict] | str]:
        """
        Extract speaker dictionary from raw output, returning original text on failure.

        Args:
            text (`str` or `list[str]`):
                Single text or batch of texts to parse from the output of `decode` with `return_format="raw"`.

        Returns:
            Parsed output(s). For single input, returns `list[dict]` or `str`.
            For batch input, returns `list[list[dict] | str]`.
        """
        is_single = isinstance(text, str)
        if is_single:
            text = [text]

        speaker_dict = []
        for t in text:
            t = t.strip()
            if t.startswith("assistant"):
                t = t[len("assistant") :].strip()

            if not t.startswith("["):
                logger.warning("Output doesn't start with '[', likely not JSON array.")
                speaker_dict.append(t)
                continue

            segments = json.loads(t)
            if not isinstance(segments, list):
                logger.warning(f"Expected list, got {type(segments).__name__}.")
                speaker_dict.append(t)
                continue

            if segments and not all(isinstance(seg, dict) and "Content" in seg for seg in segments):
                logger.warning("Not all segments have expected structure.")
                speaker_dict.append(t)
                continue

            time_stamps_valid = True
            for seg in segments:
                if isinstance(seg, dict):
                    for key in ("Start", "End"):
                        val = seg.get(key, None)
                        if val is not None and not isinstance(val, float):
                            if isinstance(val, (int, float)):
                                seg[key] = float(val)
                            else:
                                logger.warning(f"Expected '{key}' to be a number, got {type(val).__name__}.")
                                time_stamps_valid = False
                                break
                else:
                    logger.warning(f"Expected segment to be dict, got {type(seg).__name__}.")
                    time_stamps_valid = False
                    break

            if not time_stamps_valid:
                speaker_dict.append(t)
            else:
                speaker_dict.append(segments)

        return speaker_dict[0] if is_single else speaker_dict

    def extract_transcription(self, text: str | list[str]) -> str | list[str]:
        """
        Extract and concatenate 'Content' fields from the raw output, returning original text on failure.

        Args:
            text (`str` or `list[str]`):
                Single text or batch of texts to parse from the output of `decode` with `return_format="raw"`.

        Returns:
            Extracted transcription(s). For single input, returns `str`.
            For batch input, returns `list[str]`.
        """
        is_single = isinstance(text, str)
        if is_single:
            text = [text]

        transcriptions = []
        for t in text:
            dict_output = self.extract_speaker_dict(t)

            # If parsing failed, dict_output is the original string
            if isinstance(dict_output, str):
                transcriptions.append(dict_output)
            else:
                contents = [seg.get("Content", "") for seg in dict_output if isinstance(seg, dict)]
                transcriptions.append(" ".join(contents).strip())

        return transcriptions[0] if is_single else transcriptions


__all__ = ["VibeVoiceAsrProcessor"]
