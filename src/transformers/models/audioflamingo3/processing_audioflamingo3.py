# coding=utf-8
# Copyright 2025 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
# reserved.
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

import re
from typing import Optional, Union

import numpy as np

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import is_torch_available, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)

MAX_AUDIO_LEN = 10 * 60  # 10 minutes
DEFAULT_TRANSCRIPTION_PROMPT = "Transcribe the input speech."


class AudioFlamingo3ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "chunk_length": 30.0,
            "return_attention_mask": True,
            "padding": "max_length",
        },
        "common_kwargs": {
            "return_tensors": "pt",
            "padding_side": "left",
        },
    }


class AudioFlamingo3Processor(ProcessorMixin):
    r"""
    Constructs an AudioFlamingo3 processor which wraps an AudioFlamingo3 feature extractor and an AudioFlamingo3
    tokenizer into a single processor.

    [`AudioFlamingo3Processor`] offers all the functionalities of [`WhisperFeatureExtractor`] and
    [`Qwen2TokenizerFast`]. See the [`~AudioFlamingo3Processor.__call__`] for more information.

    Args:
        feature_extractor ([`WhisperFeatureExtractor`]):
            The feature extractor is a required input.
        tokenizer ([`Qwen2TokenizerFast`]):
            The tokenizer is a required input.
        chat_template (`Optional[str]`, *optional*):
            The Jinja template to use for formatting the conversation. If not provided, the tokenizer's default chat
            template will be used.
        audio_token (`Optional[str]`, *optional*, defaults to `"<sound>"`):
            Special token used to represent audio inputs in the chat template.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "Qwen2TokenizerFast"

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
        audio_token="<sound>",
    ):
        self.audio_token = audio_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[TextInput, list[TextInput]],
        audio: Optional[AudioInput] = None,
        output_labels: Optional[bool] = False,
        **kwargs: Unpack[AudioFlamingo3ProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Main method to prepare one or several text sequence(s) and audio waveform(s) for the model. This
        method expands `<sound>` placeholders in the text based on the post-pool frame counts of the
        audio windows, then tokenizes the provided strings as-is, and extracts log-mel features
        with [`WhisperFeatureExtractor`]. If `audio` is `None`, no audio processing is performed and
        the text is tokenized as-is (LM-only behavior).

        Args:
            text (`str` or `list[str]`):
                Input sequence or batch of sequences.
            audio (`np.ndarray` or `list[np.ndarray]`):
                Input audio or batch of audios as NumPy arrays. If provided, there must be as many `text` inputs as
                `audio` inputs.
            output_labels (bool, *optional*, default=False):
                Whether to return labels for training.

        Returns:
            [`BatchFeature`]: A dictionary with tokenized text (`input_ids`, `attention_mask`) and
            audio features (`input_features`, `input_features_mask`).
        """

        # Merge defaults with user kwargs
        call_kwargs = self._merge_kwargs(
            AudioFlamingo3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = call_kwargs["text_kwargs"]
        audio_kwargs = call_kwargs["audio_kwargs"]
        return_tensors = text_kwargs.get("return_tensors")
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if isinstance(text, str):
            text = [text]
        elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        audio_inputs = {}
        if audio is not None:
            audio = make_list_of_audio(audio)
            if len(text) != len(audio):
                raise ValueError(f"Got {len(text)} text but {len(audio)} audios; they must match 1:1.")

            # Determine number of chunks per sample, and flatten
            window_size = int(audio_kwargs["sampling_rate"] * audio_kwargs["chunk_length"])
            max_windows = int(MAX_AUDIO_LEN // audio_kwargs["chunk_length"])

            per_sample_windows: list[int] = []
            flat_chunks: list[np.ndarray] = []

            for audio_el in audio:
                n_samples = int(audio_el.shape[0])
                n_win = max(1, (n_samples + window_size - 1) // window_size)
                if n_win > max_windows:
                    logger.warning(
                        f"Audio duration ({n_samples / audio_kwargs['sampling_rate']:.1f}s) exceeds {MAX_AUDIO_LEN}s; truncating to first {MAX_AUDIO_LEN}s."
                    )
                    n_win = max_windows
                per_sample_windows.append(n_win)

                time_cap = min(n_samples, n_win * window_size)
                for i in range(n_win):
                    start = i * window_size
                    end = min((i + 1) * window_size, time_cap)
                    flat_chunks.append(audio_el[start:end])

            # Feature extraction
            audio_inputs = self.feature_extractor(flat_chunks, **audio_kwargs)
            padding_mask = audio_inputs.pop("attention_mask")
            audio_inputs["input_features_mask"] = padding_mask

            # Compute sequence lengths token counting
            audio_lengths = torch.stack([s.sum() for s in torch.split(padding_mask.sum(-1), per_sample_windows)])
            conv_output_lengths = (audio_lengths - 1) // 2 + 1  # After conv2 downsampling
            audio_tokens_lengths = (conv_output_lengths - 2) // 2 + 1  # After avg pooling

            # expand audio tokens in text
            for i, audio_length in enumerate(audio_tokens_lengths):
                expanded = re.sub(re.escape(self.audio_token), self.audio_token * audio_length, text[i])
                text[i] = expanded

        # Tokenize
        text_inputs = self.tokenizer(text, **text_kwargs)

        data = {**text_inputs, **audio_inputs}
        if output_labels:
            labels = data["input_ids"].clone()
            labels[labels == self.audio_token_id] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            data["labels"] = labels

        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def model_input_names(self) -> list[str]:
        tok_names = self.tokenizer.model_input_names
        fea_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tok_names + fea_names + ["input_features_mask"]))

    def apply_transcription_request(
        self,
        audio: Union[str, list[str], AudioInput],
        prompt: Optional[Union[str, list[str]]] = None,
        **kwargs: Unpack[AudioFlamingo3ProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare inputs for automatic speech recognition without manually writing the default transcription prompt.

        Args:
            audio (`str`, `list[str]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                Audio to transcribe. Strings are interpreted as local paths or URLs and will be loaded automatically by
                the chat template loader; NumPy arrays and PyTorch tensors are forwarded directly.
            prompt (`str` or `list[str]`, *optional*):
                Custom prompt(s) to include in the user turn. A list must be the same length as the batch. When `None`,
                each sample uses `"Transcribe the input speech."`.
            **kwargs:
                Additional keyword arguments forwarded to [`~AudioFlamingo3Processor.apply_chat_template`] (for example
                `text_kwargs`, `audio_kwargs`, ...).

        Returns:
            [`BatchFeature`]: Processor outputs ready to be passed to [`AudioFlamingo3ForConditionalGeneration.generate`].

        """

        if isinstance(audio, str):
            audio_items: list[Union[str, np.ndarray]] = [audio]
        elif isinstance(audio, (list, tuple)) and audio and all(isinstance(el, str) for el in audio):
            audio_items = list(audio)
        else:
            audio_items = list(make_list_of_audio(audio))
            if is_torch_available():
                audio_items = [el.detach().cpu().numpy() if isinstance(el, torch.Tensor) else el for el in audio_items]

        batch_size = len(audio_items)
        if batch_size == 0:
            raise ValueError("`audio` must contain at least one sample.")

        if prompt is None:
            prompts = [DEFAULT_TRANSCRIPTION_PROMPT] * batch_size
        elif isinstance(prompt, str):
            prompts = [prompt] * batch_size
        elif isinstance(prompt, (list, tuple)):
            if len(prompt) != batch_size:
                raise ValueError(
                    f"Received {len(prompt)} prompt(s) for {batch_size} audio sample(s); counts must match."
                )
            prompts = []
            for item in prompt:
                if item is None:
                    prompts.append(DEFAULT_TRANSCRIPTION_PROMPT)
                elif isinstance(item, str):
                    prompts.append(item)
                else:
                    raise TypeError("Each prompt must be a string or `None`.")
        else:
            raise TypeError("`prompt` must be a string, a sequence of strings, or `None`.")

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "audio", "path": audio_item}
                        if isinstance(audio_item, str)
                        else {"type": "audio", "audio": audio_item},
                    ],
                }
            ]
            for prompt_text, audio_item in zip(prompts, audio_items)
        ]

        return self.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            **kwargs,
        )

    def batch_decode(self, *args, strip_prefix=False, **kwargs):
        """
        Forward arguments to [`~PreTrainedTokenizer.batch_decode`] and optionally remove the assistant framing the model
        was trained to produce.

        AF3 transcription requests respond with sentences such as `"The spoken content of the audio is \"...\"."`.
        Setting `strip_prefix=True` trims the fixed prefix for just the transcription text.
        """
        decoded = self.tokenizer.batch_decode(*args, **kwargs)
        if strip_prefix:
            decoded = [self._strip_assistant_prefix_and_quotes(text) for text in decoded]
        return decoded

    def _strip_assistant_prefix_and_quotes(self, text: str) -> str:
        """
        Remove the assistant prefix and surrounding quotes from a decoded transcription string.
        """

        stripped = text.strip()

        for prefix in (
            "The spoken content of the audio is",
            "The transcription of the audio is",
        ):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :].strip()
                break

        if stripped.endswith("."):
            stripped = stripped[:-1].strip()

        if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
            stripped = stripped[1:-1].strip()

        return stripped


__all__ = ["AudioFlamingo3Processor"]
