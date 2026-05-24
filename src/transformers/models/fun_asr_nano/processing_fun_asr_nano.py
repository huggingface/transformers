# Copyright 2025 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.
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
"""Processor for Fun-ASR-Nano."""

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging


logger = logging.get_logger(__name__)


class FunAsrNanoProcessor(ProcessorMixin):
    r"""
    Constructs a Fun-ASR-Nano processor which wraps a feature extractor and a tokenizer.

    [`FunAsrNanoProcessor`] offers all the functionalities of [`FunAsrNanoFeatureExtractor`] and
    [`Qwen2Tokenizer`] (or compatible tokenizer). See the [`~FunAsrNanoProcessor.__call__`] and
    [`~FunAsrNanoProcessor.decode`] for more information.

    Args:
        feature_extractor (`FunAsrNanoFeatureExtractor`):
            The feature extractor for audio preprocessing.
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer for text encoding/decoding (typically Qwen3 tokenizer).
        chat_template (`str`, *optional*):
            Jinja template for formatting chat messages.
        audio_token (`str`, *optional*, defaults to `"<|startofspeech|>"`):
            Token used to mark audio input positions.

    Example:

    ```python
    >>> from transformers import FunAsrNanoProcessor, FunAsrNanoFeatureExtractor, AutoTokenizer

    >>> feature_extractor = FunAsrNanoFeatureExtractor()
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    >>> processor = FunAsrNanoProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    ```
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "FunAsrNanoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        audio_token="<|startofspeech|>",
        audio_end_token="<|endofspeech|>",
        **kwargs,
    ):
        self.audio_token = audio_token
        self.audio_end_token = audio_end_token
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template, **kwargs)

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        audio=None,
        sampling_rate: int | None = None,
        return_tensors: str | None = None,
        **kwargs,
    ):
        """
        Process text and audio inputs for the Fun-ASR-Nano model.

        Args:
            text: Text input(s). Should contain audio placeholder tokens if audio is provided.
            audio: Raw audio waveform(s) as numpy arrays or lists of floats.
            sampling_rate: Sampling rate of the audio (must be 16000).
            return_tensors: Type of tensors to return ("pt", "np", "tf").

        Returns:
            BatchFeature with input_ids, attention_mask, input_features, and feature_lengths.
        """
        if audio is not None and text is not None:
            # Process audio
            audio_features = self.feature_extractor(
                audio,
                sampling_rate=sampling_rate or self.feature_extractor.sampling_rate,
                return_tensors=return_tensors,
                **kwargs,
            )

            # Process text
            text_inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=True,
                **kwargs,
            )

            # Combine
            return {**text_inputs, **audio_features}

        elif text is not None:
            return self.tokenizer(text, return_tensors=return_tensors, **kwargs)

        elif audio is not None:
            return self.feature_extractor(
                audio,
                sampling_rate=sampling_rate or self.feature_extractor.sampling_rate,
                return_tensors=return_tensors,
                **kwargs,
            )

        else:
            raise ValueError("You must provide either `text` or `audio` input.")

    def batch_decode(self, *args, **kwargs):
        """Forward to tokenizer's batch_decode."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Forward to tokenizer's decode."""
        return self.tokenizer.decode(*args, **kwargs)

    def build_chat_input(
        self,
        audio_path_or_array,
        prompt: str = "语音转写：",
        language: str | None = None,
        hotwords: list[str] | None = None,
        system_prompt: str = "You are a helpful assistant.",
        sampling_rate: int | None = None,
        return_tensors: str = "pt",
    ):
        """
        Build complete model input from audio and optional prompt parameters.

        This is a convenience method that constructs the ChatML-formatted input
        expected by Fun-ASR-Nano, including audio placeholder tokens.

        Args:
            audio_path_or_array: Audio file path or numpy array.
            prompt: Task prompt (default: "语音转写：").
            language: Language hint (e.g., "中文", "English", "日文").
            hotwords: List of hotwords for domain adaptation.
            system_prompt: System prompt for the chat template.
            sampling_rate: Audio sampling rate.
            return_tensors: Type of tensors to return.

        Returns:
            Dict with input_ids, attention_mask, input_features, feature_lengths.

        Example:

        ```python
        >>> processor = FunAsrNanoProcessor.from_pretrained("FunAudioLLM/Fun-ASR-Nano-2512-hf")
        >>> inputs = processor.build_chat_input("audio.wav", language="中文", hotwords=["开放时间"])
        ```
        """
        import numpy as np

        # Build prompt
        if hotwords and len(hotwords) > 0:
            hotwords_str = ", ".join(hotwords)
            task_prompt = f"请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n热词列表：[{hotwords_str}]\n"
        else:
            task_prompt = ""

        if language is None:
            task_prompt += "语音转写"
        else:
            task_prompt += f"语音转写成{language}"
        task_prompt += "："

        # Build ChatML text with audio placeholder
        audio_placeholder = f"{self.audio_token}!audio{self.audio_end_token}"
        chat_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{task_prompt}{audio_placeholder}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # Load audio if path
        if isinstance(audio_path_or_array, str):
            import soundfile as sf

            audio_array, sr = sf.read(audio_path_or_array, dtype="float32")
            if sr != (sampling_rate or self.feature_extractor.sampling_rate):
                import torch
                import torchaudio

                audio_tensor = torch.from_numpy(audio_array).float()
                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                audio_tensor = torchaudio.functional.resample(audio_tensor, sr, self.feature_extractor.sampling_rate)
                audio_array = audio_tensor.squeeze(0).numpy()
        else:
            audio_array = np.asarray(audio_path_or_array, dtype=np.float32)

        # Extract audio features
        audio_features = self.feature_extractor(
            audio_array,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors=return_tensors,
        )

        # Determine number of audio tokens needed
        num_audio_frames = audio_features["feature_lengths"][0]
        if hasattr(self.feature_extractor, "lfr_n"):
            # After adaptor with low_frame_rate: conv stride reduces further
            # Original formula from FunASR: olens = 1 + (T - 3 + 2) // 2 twice, then // 2 + 1
            T = int(num_audio_frames)
            olens = 1 + (T - 3 + 2) // 2
            olens = 1 + (olens - 3 + 2) // 2
            num_audio_tokens = (olens - 1) // 2 + 1
        else:
            num_audio_tokens = int(num_audio_frames)

        # Build input_ids with audio placeholder tokens expanded
        # First tokenize text parts around the audio placeholder
        parts = chat_text.split(audio_placeholder)
        prefix_ids = self.tokenizer.encode(parts[0], add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(parts[1], add_special_tokens=False)

        # Audio token index (from config, default 151646)
        audio_token_id = 151646  # Qwen3 unused token as placeholder
        audio_token_ids = [audio_token_id] * num_audio_tokens

        input_ids = prefix_ids + audio_token_ids + suffix_ids

        import torch

        result = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long),
            "input_features": audio_features["input_features"],
            "feature_lengths": audio_features["feature_lengths"],
        }

        return result

    @property
    def model_input_names(self):
        feature_extractor_input_names = self.feature_extractor.model_input_names
        tokenizer_input_names = self.tokenizer.model_input_names
        return list(dict.fromkeys(feature_extractor_input_names + tokenizer_input_names))


__all__ = ["FunAsrNanoProcessor"]
