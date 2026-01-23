# Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
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

MAX_AUDIO_LEN = 20 * 60  # 20 minutes


class MusicFlamingoProcessorKwargs(ProcessingKwargs, total=False):
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


class MusicFlamingoProcessor(ProcessorMixin):
    r"""
    Constructs an MusicFlamingo processor which wraps an MusicFlamingo feature extractor and an MusicFlamingo
    tokenizer into a single processor.

    [`MusicFlamingoProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`] and
    [`Qwen2TokenizerFast`]. See the [`~MusicFlamingoProcessor.__call__`] for more information.

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
        audio_bos_token (`Optional[str]`, *optional*, defaults to `"<|sound_bos|>"`):
            Special token used to represent the beginning of an audio sequence.
        audio_eos_token (`Optional[str]`, *optional*, defaults to `"<|sound_eos|>"`):
            Special token used to represent the end of an audio sequence.
    """

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
        audio_token="<sound>",
        audio_bos_token="<|sound_bos|>",
        audio_eos_token="<|sound_eos|>",
    ):
        self.audio_token = audio_token
        self.audio_bos_token = audio_bos_token
        self.audio_eos_token = audio_eos_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
        self.audio_bos_token_id = tokenizer.convert_tokens_to_ids(audio_bos_token)
        self.audio_eos_token_id = tokenizer.convert_tokens_to_ids(audio_eos_token)
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[TextInput, list[TextInput]],
        audio: Optional[AudioInput] = None,
        output_labels: Optional[bool] = False,
        **kwargs: Unpack[MusicFlamingoProcessorKwargs],
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
            MusicFlamingoProcessorKwargs,
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
            audio_times_list: list[torch.Tensor] = []

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
                    # Calculate the start time of this audio chunk in seconds
                    start_sec = start / audio_kwargs["sampling_rate"]

                    # Generate 750 timestamps at 40ms intervals (30s / 750 = 0.04s)
                    chunk_times = torch.arange(750).float() * 0.04 + start_sec
                    audio_times_list.append(chunk_times)

            # Feature extraction
            audio_inputs = self.feature_extractor(flat_chunks, **audio_kwargs)
            padding_mask = audio_inputs.pop("attention_mask")
            audio_inputs["input_features_mask"] = padding_mask

            # Add audio times as tensor
            if return_tensors == "pt":
                audio_inputs["audio_times"] = torch.stack(audio_times_list).to(dtype=torch.float32)

            # Compute sequence lengths token counting
            audio_lengths = torch.stack([s.sum() for s in torch.split(padding_mask.sum(-1), per_sample_windows)])
            conv_output_lengths = (audio_lengths - 1) // 2 + 1  # After conv2 downsampling
            audio_tokens_lengths = (conv_output_lengths - 2) // 2 + 1  # After avg pooling

            # expand audio tokens in text
            for i, audio_length in enumerate(audio_tokens_lengths):
                expanded = re.sub(
                    re.escape(self.audio_token),
                    self.audio_bos_token + self.audio_token * audio_length + self.audio_eos_token,
                    text[i],
                )
                text[i] = expanded

        # Tokenize
        text_inputs = self.tokenizer(text, **text_kwargs)

        data = {**text_inputs, **audio_inputs}
        if output_labels:
            labels = data["input_ids"].clone()
            labels[labels == self.audio_token_id] = -100
            labels[labels == self.audio_bos_token_id] = -100
            labels[labels == self.audio_eos_token_id] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            data["labels"] = labels

        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def model_input_names(self) -> list[str]:
        tok_names = self.tokenizer.model_input_names
        fea_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tok_names + fea_names + ["input_features_mask", "audio_times"]))

    def batch_decode(self, *args, **kwargs):
        """
        Forward arguments to [`~PreTrainedTokenizer.batch_decode`].
        """
        return self.tokenizer.batch_decode(*args, **kwargs)


__all__ = ["MusicFlamingoProcessor"]
