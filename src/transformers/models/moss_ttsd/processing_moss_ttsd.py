# coding=utf-8
# Copyright 2025 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
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
Processor class for MOSS-TTSD.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional, Union

import numpy as np

from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import BatchEncoding
from ...utils import is_torch_available, is_torchaudio_available
from .. import AutoFeatureExtractor, AutoTokenizer


if is_torch_available():
    import torch

    from ..xy_tokenizer.modeling_xy_tokenizer import XYTokenizer

if is_torchaudio_available():
    import torchaudio


class MossTTSDProcessorKwargs(ProcessingKwargs, total=False):
    """
    Arguments for configuring MOSS-TTSD processing operations.

    Inherits from ProcessingKwargs and provides structured configuration for text and audio processing.
    """

    _defaults = {
        "text_kwargs": {
            "pad_token_id": 0,  # Fallback pad token ID, actual value comes from tokenizer.pad_token_id
        },
        "audio_kwargs": {
            "max_channels": 8,  # Maximum number of quantization channels
            "audio_pad_token_id": 1024,  # Padding token ID for non-text channels
            "silence_duration": 0.0,  # Duration of silence to append for encoder segmentation
            "input_sample_rate": 16000,  # Input audio sampling rate (fallback, inferred from audio_tokenizer.config)
            "encoder_downsample_rate": 320,  # Encoder downsampling rate (fallback, inferred from audio_tokenizer.config)
            "speech_token_range": [151665, 152689],  # Token range for speech tokens (first codebook offset mapping)
            "audio_bos_token": "<|begin_of_speech|>",
            "audio_eos_token": "<|end_of_speech|>",
        },
        "common_kwargs": {
            "return_tensors": "pt",
            "padding": True,
            "use_normalize": False,
        },
    }


@dataclass
class MossTTSDChatSample:
    """
    Intermediate representation of a single sample with T×C grid layout and metadata.

    Args:
        input_ids_2d (`torch.LongTensor`):
            Shape (T, C) tensor where column 0 contains text tokens and columns 1..C-1 contain
            quantized audio codebooks (or padding token 1024 for empty slots).
        label_ids_2d (`torch.LongTensor`, *optional*):
            Optional label tensor for training, same shape as input_ids_2d.
        meta (`dict`):
            Dictionary containing metadata for debugging and tracking purposes.
    """

    input_ids_2d: "torch.LongTensor"
    label_ids_2d: Optional["torch.LongTensor"]
    meta: dict


@dataclass
class MossTTSDBatchInput:
    """
    Batched input tensors for MOSS-TTSD model.

    Args:
        input_ids (`torch.LongTensor`):
            Shape (B, T, C) tensor containing batched input token IDs.
        attention_mask (`torch.LongTensor`):
            Shape (B, T) tensor containing attention mask for valid tokens.
        labels (`torch.LongTensor`, *optional*):
            Optional shape (B, T, C) tensor containing label token IDs for training.
    """

    input_ids: "torch.LongTensor"
    attention_mask: "torch.LongTensor"
    labels: Optional["torch.LongTensor"]


@dataclass
class MossTTSDResponse:
    """
    Unified response container for MOSS-TTSD inference outputs.

    Args:
        audio (`np.ndarray`, *optional*):
            Optional numpy array containing generated audio waveform.
        generated_text (`str`, *optional*, defaults to `""`):
            String containing generated text output.
        sampling_rate (`int`, *optional*):
            Optional integer specifying the sampling rate of the generated audio.
    """

    audio: Optional[np.ndarray] = None
    generated_text: str = ""
    sampling_rate: Optional[int] = None


class MossTTSDSampleProcessor:
    """
    Sample-level processor for MOSS-TTSD that handles individual sample processing without batch padding.

    This class handles per-sample processing logic:
    - Parses JSONL items (text/prompt_text/prompt_audio)
    - Optional text normalization
    - Audio loading/resampling/merging, feature extraction and encoding
    - Generates T×C grid and performs multi-channel shifting

    Args:
        tokenizer (`AutoTokenizer`):
            The text tokenizer for encoding text tokens.
        feature_extractor (`AutoFeatureExtractor`, *optional*):
            Optional feature extractor for audio preprocessing.
        audio_tokenizer (`AutoModel`, *optional*):
            Optional audio tokenizer for audio encoding/decoding.
        chat_template (`str`, *optional*):
            Optional chat template string for conversation formatting.
        speech_token_range (`List[int]`):
            List of [start, end] token IDs for speech token mapping.
        audio_bos_token (`str`):
            Beginning of speech token string.
        audio_eos_token (`str`):
            End of speech token string.
        audio_pad_token_id (`int`):
            Padding token ID for audio channels.
        max_channels (`int`):
            Maximum number of quantization channels.
        input_sample_rate (`int`):
            Target sample rate for input audio.
        encoder_downsample_rate (`int`):
            Downsampling rate of the audio encoder.
    """

    def __init__(
        self,
        tokenizer,
        feature_extractor: Optional = None,
        audio_tokenizer: Optional = None,
        *,
        chat_template: Optional[str],
        speech_token_range: list[int],
        audio_bos_token: str,
        audio_eos_token: str,
        audio_pad_token_id: int,
        max_channels: int,
        input_sample_rate: int,
        encoder_downsample_rate: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.audio_tokenizer = audio_tokenizer
        self.chat_template = chat_template
        self.speech_token_range = speech_token_range
        self.audio_bos_token = audio_bos_token
        self.audio_eos_token = audio_eos_token
        self.audio_pad_token_id = audio_pad_token_id
        self.max_channels = max_channels
        self.input_sample_rate = input_sample_rate
        self.encoder_downsample_rate = encoder_downsample_rate

    def prepare_sample(
        self,
        item: dict[str, Any],
        *,
        apply_chat_template: Callable[[str, dict], str],
        use_normalize: bool = False,
        silence_duration: float = 0.0,
        **kwargs,
    ) -> MossTTSDChatSample:
        """
        Prepare a single sample from JSONL item into MossTTSDChatSample format.

        Args:
            item (`dict`):
                Dictionary containing the input data (text, prompt_audio, etc.).
            apply_chat_template (`callable`):
                Function to apply chat template formatting.
            use_normalize (`bool`, *optional*, defaults to `False`):
                Whether to apply text normalization.
            silence_duration (`float`, *optional*, defaults to `0.0`):
                Duration of silence to append to audio for encoder segmentation.
            **kwargs:
                Additional keyword arguments passed to chat template.

        Returns:
            `MossTTSDChatSample`: Processed sample with 2D input tensor and metadata.
        """
        processed = self._process_jsonl_item(item)
        system_prompt = item.get("system_prompt")
        if isinstance(system_prompt, str):
            kwargs["system_prompt"] = system_prompt

        full_text = (processed["prompt_text"] or "") + processed["text"]
        original_full_text = full_text
        if use_normalize:
            full_text = self._normalize_text(full_text)
        final_text = full_text.replace("[S1]", "<speaker1>").replace("[S2]", "<speaker2>")

        # Load and resample audio (may be None)
        wav = self._process_audio_data(processed["prompt_audio"], target_sample_rate=self.input_sample_rate)

        # Assemble into grid (T, C)
        inputs_2d = self._build_inputs(
            text=final_text,
            audio_data=wav,
            apply_chat_template=apply_chat_template,
            silence_duration=silence_duration,
            **kwargs,
        )
        inputs_2d = self._shift_inputs(
            inputs_2d, pad_token_id=self.tokenizer.pad_token_id, max_channels=self.max_channels
        )

        meta = {
            "original_text": original_full_text,
            "normalized_text": self._normalize_text(original_full_text) if use_normalize else None,
            "final_text": final_text,
            "use_normalize": use_normalize,
        }
        ids_t = torch.tensor(inputs_2d, dtype=torch.long)
        return MossTTSDChatSample(input_ids_2d=ids_t, label_ids_2d=None, meta=meta)

    def collate(
        self,
        samples: list[MossTTSDChatSample],
        *,
        pad_token_id: int,
        audio_pad_token_id: int,
    ) -> MossTTSDBatchInput:
        """
        Collate multiple samples into a batch with proper padding.

        Args:
            samples (`List[MossTTSDChatSample]`):
                List of MossTTSDChatSample objects to collate.
            pad_token_id (`int`):
                Padding token ID for text tokens.
            audio_pad_token_id (`int`):
                Padding token ID for audio tokens.

        Returns:
            `MossTTSDBatchInput`: Batched input with padded tensors.
        """
        assert is_torch_available(), "PyTorch is required for collation."
        ids_list = [s.input_ids_2d for s in samples]
        labels_list = [s.label_ids_2d for s in samples]

        C = ids_list[0].shape[1]
        max_len = max(x.shape[0] for x in ids_list)
        padded_ids, padded_labels, padded_attn = [], [], []

        for ids, labels in zip(ids_list, labels_list):
            pad_len = max_len - ids.shape[0]
            pad_grid = torch.full((pad_len, C), audio_pad_token_id, dtype=torch.long)
            pad_grid[:, 0] = pad_token_id  # Text column uses tokenizer pad
            ids_padded = torch.cat([pad_grid, ids], dim=0)
            padded_ids.append(ids_padded)

            attn = torch.ones(ids.shape[0], dtype=torch.long)
            a_pad = torch.zeros(pad_len, dtype=torch.long)
            padded_attn.append(torch.cat([a_pad, attn], dim=0))

            if labels is None:
                padded_labels.append(None)
            else:
                lab_pad = torch.full((pad_len, C), audio_pad_token_id, dtype=torch.long)
                lab_pad[:, 0] = -100  # Text labels are ignored by default
                padded_labels.append(torch.cat([lab_pad, labels], dim=0))

        input_ids = torch.stack(padded_ids)  # (B, T, C)
        attention_mask = torch.stack(padded_attn)  # (B, T)
        labels = (
            torch.stack([l if l is not None else torch.full_like(input_ids[0], -100) for l in padded_labels])
            if any(l is not None for l in padded_labels)
            else None
        )

        return MossTTSDBatchInput(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    @staticmethod
    def _process_jsonl_item(item: dict[str, Any]) -> dict[str, Any]:
        """
        Process a JSONL item to extract text and audio data.

        Supports both single-speaker and multi-speaker formats:
        - Single: {"prompt_audio": path, "prompt_text": text}
        - Multi: {"prompt_audio_speaker1": path1, "prompt_text_speaker1": text1, ...}

        Args:
            item: Dictionary containing the JSONL item data.

        Returns:
            Dictionary with extracted "text", "prompt_text", and "prompt_audio" fields.
        """
        base_path = item.get("base_path", "")
        text = item.get("text", "")

        prompt_audio = None
        prompt_text = ""

        if "prompt_audio" in item and "prompt_text" in item:
            pa = item["prompt_audio"]
            if pa:
                prompt_audio = os.path.join(base_path, pa) if isinstance(pa, str) and base_path else pa
                prompt_text = item.get("prompt_text", "")
        else:
            pa1, pt1 = item.get("prompt_audio_speaker1", ""), item.get("prompt_text_speaker1", "")
            pa2, pt2 = item.get("prompt_audio_speaker2", ""), item.get("prompt_text_speaker2", "")
            has1 = (isinstance(pa1, str) and pa1) or isinstance(pa1, tuple)
            has2 = (isinstance(pa2, str) and pa2) or isinstance(pa2, tuple)
            if has1 or has2:
                spk1 = os.path.join(base_path, pa1) if isinstance(pa1, str) and base_path and pa1 else pa1
                spk2 = os.path.join(base_path, pa2) if isinstance(pa2, str) and base_path and pa2 else pa2
                prompt_audio = {"speaker1": spk1, "speaker2": spk2}
            tmp = ""
            if pt1:
                tmp += f"[S1]{pt1}"
            if pt2:
                tmp += f"[S2]{pt2}"
            prompt_text = tmp.strip()

        return {"text": text, "prompt_text": prompt_text, "prompt_audio": prompt_audio}

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text by applying various transformations for TTS processing.

        Performs speaker tag conversion, punctuation normalization, laughter conversion,
        and other text cleaning operations suitable for speech synthesis.

        Args:
            text: Input text string to normalize.

        Returns:
            Normalized text string.
        """
        text = re.sub(r"\[(\d+)\]", r"[S\1]", text)
        remove_chars = '【】《》（）『』「」"-""～~'
        text = re.sub(r"\[(?!S\d+\])([^\]]*)\]", r"\1", text)
        segments = re.split(r"(?=\[S\d+\])", text.replace("\n", " "))
        out = []
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            m = re.match(r"^(\[S\d+\])\s*(.*)", seg)
            tag, content = m.groups() if m else ("", seg)
            content = re.sub(f"[{re.escape(remove_chars)}]", "", content)
            content = re.sub(r"哈{2,}", "(笑)", content)
            content = re.sub(r"\b(ha(\s*ha)+)\b", "(laughs)", content, flags=re.IGNORECASE)
            content = content.replace("——", "，").replace("……", "，")
            trans = str.maketrans(
                {"！": "，", "!": ",", "；": "，", ";": ",", "：": "，", ":": ",", "、": "，", "？": "，", "?": ","}
            )
            content = content.translate(trans).strip()
            if len(content) > 1:
                last = "。" if content[-1] == "，" else ("." if content[-1] == "," else content[-1])
                body = content[:-1].replace("。", "，")
                content = body + last
            out.append(f"{tag}{content}".strip())
        return "".join(out)

    @staticmethod
    def _load_single_audio(audio_input: Union[str, tuple["torch.Tensor", int]]):
        """
        Load audio from file path or tensor tuple.

        Args:
            audio_input: Either a file path string or a tuple of (tensor, sample_rate).

        Returns:
            Tuple of (audio_tensor, sample_rate).

        Raises:
            ValueError: If audio input format is unsupported.
        """
        if isinstance(audio_input, tuple) and len(audio_input) == 2:
            return audio_input
        if isinstance(audio_input, str):
            try:
                return torchaudio.load(audio_input)
            except Exception:
                import soundfile as sf  # type: ignore

                data, sr = sf.read(audio_input, always_2d=True)
                data_t = torch.from_numpy(np.transpose(data))  # (C, T)
                return data_t, int(sr)
        raise ValueError(f"Unsupported audio input format: {type(audio_input)}")

    @staticmethod
    def _resample(audio: "torch.Tensor", sr: int, target_sr: int) -> tuple["torch.Tensor", int]:
        """
        Resample audio to target sample rate and convert to mono if needed.

        Args:
            audio: Input audio tensor with shape (channels, time).
            sr: Current sample rate.
            target_sr: Target sample rate.

        Returns:
            Tuple of (resampled_audio, target_sr) where audio is mono with shape (1, time).
        """
        if sr != target_sr:
            audio = torchaudio.functional.resample(audio, sr, target_sr)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        return audio, target_sr

    @classmethod
    def _load_audio_data(
        cls, audio_input: Union[str, tuple["torch.Tensor", int]], target_sample_rate: int
    ) -> tuple["torch.Tensor", int]:
        """
        Load and resample audio data to target sample rate.

        Args:
            audio_input: Audio file path or tensor tuple.
            target_sample_rate: Target sample rate for resampling.

        Returns:
            Tuple of (audio_tensor, target_sample_rate).
        """
        audio, sr = cls._load_single_audio(audio_input)
        return cls._resample(audio, sr, target_sample_rate)

    @classmethod
    def _merge_speaker_audios(
        cls,
        wav1: Union[str, tuple["torch.Tensor", int]],
        wav2: Union[str, tuple["torch.Tensor", int]],
        target_sample_rate: int,
    ) -> "torch.Tensor":
        """
        Merge two speaker audio inputs by concatenation.

        Args:
            wav1: Audio input for speaker 1.
            wav2: Audio input for speaker 2.
            target_sample_rate: Target sample rate for both audio inputs.

        Returns:
            Concatenated audio tensor.
        """
        a1, _ = cls._load_audio_data(wav1, target_sample_rate)
        a2, _ = cls._load_audio_data(wav2, target_sample_rate)
        return torch.cat([a1, a2], dim=1)

    @classmethod
    def _process_audio_data(
        cls, prompt_audio: Optional[Union[str, dict[str, Any], tuple["torch.Tensor", int]]], target_sample_rate: int
    ) -> Optional["torch.Tensor"]:
        """
        Process audio data from various input formats.

        Handles single audio files, multi-speaker audio dictionaries, or None input.

        Args:
            prompt_audio: Audio input in various formats (path, dict, tensor tuple, or None).
            target_sample_rate: Target sample rate for processing.

        Returns:
            Processed audio tensor or None if no audio provided.
        """
        if prompt_audio is None:
            return None
        if isinstance(prompt_audio, dict) and "speaker1" in prompt_audio and "speaker2" in prompt_audio:
            return cls._merge_speaker_audios(prompt_audio["speaker1"], prompt_audio["speaker2"], target_sample_rate)
        wav, _ = cls._load_audio_data(prompt_audio, target_sample_rate)
        return wav

    def _build_inputs(
        self,
        text: str,
        audio_data: Optional["torch.Tensor"],
        apply_chat_template: Callable[[str, dict], str],
        silence_duration: float,
        **kwargs,
    ) -> np.ndarray:
        """
        Build input grid from text and optional audio data.

        Creates a TxC grid where column 0 contains text tokens and columns 1..C-1 contain
        quantized audio codebook tokens. Audio tokens are mapped to speech token range.

        Args:
            text: Input text string to process.
            audio_data: Optional audio tensor with shape (channels, time).
            apply_chat_template: Function to apply chat template formatting.
            silence_duration: Duration of silence to append for encoder segmentation.
            **kwargs: Additional arguments for chat template.

        Returns:
            NumPy array with shape (T, max_channels) containing the input grid.
        """
        assert isinstance(text, str), "text must be a string"
        prompt = apply_chat_template(text, kwargs)

        text_ids = np.array(self.tokenizer.encode(prompt, add_special_tokens=False))
        grid = np.full((text_ids.shape[0], self.max_channels), self.audio_pad_token_id, dtype=np.int64)
        grid[:, 0] = text_ids

        if audio_data is not None:
            silence_samples = int(max(0.0, silence_duration) * self.input_sample_rate)
            silence = torch.zeros(audio_data.shape[0], silence_samples, device=audio_data.device)
            wav = torch.cat([audio_data, silence], dim=1)

            feat = self.feature_extractor(
                wav, sampling_rate=self.input_sample_rate, return_attention_mask=True, return_tensors="pt"
            )
            with torch.no_grad():
                enc = self.audio_tokenizer.encode(feat)
                # (time, codebooks)
                audio_codes = enc["audio_codes"][:, 0].permute(1, 0).cpu().numpy()
            # Map first codebook to speech token range
            audio_codes[:, 0] = audio_codes[:, 0] + self.speech_token_range[0]
            grid = np.concatenate([grid, audio_codes], axis=0)

            # Trim silence tokens at the end based on encoder downsampling
            silence_tokens = silence_duration * self.input_sample_rate / self.encoder_downsample_rate
            cut = math.floor(silence_tokens / 10) * 10
            if cut > 0:
                grid = grid[:-cut]

        return grid

    @staticmethod
    def _shift_inputs(input_ids: np.ndarray, pad_token_id: int, max_channels: int) -> np.ndarray:
        """
        Convert (T, C) grid to time-shifted multi-channel layout (preserving original implementation logic).

        Creates a shifted layout where new_len = T + C - 1, with column j shifted backwards by j positions.
        This enables the model to process multiple codebook channels with temporal alignment.

        Args:
            input_ids: Input grid with shape (T, C).
            pad_token_id: Padding token ID for text tokens.
            max_channels: Maximum number of channels.

        Returns:
            Shifted array with shape (T + max_channels - 1, max_channels).
        """
        T, _ = input_ids.shape
        new_len = T + max_channels - 1
        shifted = np.full((new_len, max_channels), fill_value=1024, dtype=np.int64)
        shifted[:, 0] = np.full(new_len, pad_token_id, dtype=np.int64)
        for j in range(max_channels):
            shifted[j : (T + j), j] = input_ids[:, j]
        return shifted


class MossTTSDProcessor(ProcessorMixin):
    r"""
    Constructs a MOSS-TTSD processor which wraps a tokenizer, feature extractor, and audio tokenizer into a single
    processor. It provides unified text-speech processing capabilities while maintaining backward compatibility with
    previous API versions.

    [`MossTTSDProcessor`] offers all the functionalities of [`AutoTokenizer`], [`AutoFeatureExtractor`] and
    [`XYTokenizer`]. See the [`~MossTTSDProcessor.__call__`] and [`~MossTTSDProcessor.decode`] for more information.

    Args:
            tokenizer ([`AutoTokenizer`]):
                An instance of [`AutoTokenizer`]. The tokenizer is a required input.
            feature_extractor ([`AutoFeatureExtractor`]):
                An instance of [`AutoFeatureExtractor`]. The feature extractor is a required input.
            audio_tokenizer ([`XYTokenizer`]):
                An instance of [`XYTokenizer`]. The audio tokenizer is a required input.
            chat_template (`str`, *optional*):
                A template string for chat formatting when combining text and audio interactions.
            speech_token_range (`List[int]`, *optional*, defaults to `[151665, 152689]`):
                Token range [start, end] for mapping speech tokens.
            audio_bos_token (`str`, *optional*, defaults to `"<|begin_of_speech|>"`):
                Beginning of speech token string.
            audio_eos_token (`str`, *optional*, defaults to `"<|end_of_speech|>"`):
                End of speech token string.
            audio_pad_token_id (`int`, *optional*, defaults to 1024):
                Padding token ID for audio channels.
    """

    attributes = ["tokenizer", "feature_extractor", "audio_tokenizer"]
    feature_extractor_class = "XYTokenizerFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    audio_tokenizer_class = "XYTokenizer"

    def __init__(
        self,
        tokenizer,
        feature_extractor,
        audio_tokenizer,
        chat_template: Optional[str] = None,
        speech_token_range: Optional[list[int]] = None,
        audio_bos_token: str = "<|begin_of_speech|>",
        audio_eos_token: str = "<|end_of_speech|>",
        audio_pad_token_id: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer, feature_extractor, audio_tokenizer, **kwargs)

        self.max_channels = (audio_tokenizer.quantizer.num_quantizers if audio_tokenizer else None) or 8
        cfg = getattr(audio_tokenizer, "config", None)
        # Prefer new names with fallbacks to deprecated ones
        self.input_sample_rate = (
            (getattr(cfg, "input_sampling_rate", None) if cfg else None)
            or (getattr(cfg, "input_sample_rate", None) if cfg else None)
            or 16000
        )
        self.output_sample_rate = (
            (getattr(cfg, "sampling_rate", None) if cfg else None)
            or (getattr(cfg, "output_sample_rate", None) if cfg else None)
            or 16000
        )
        self.encoder_downsample_rate = (
            getattr(audio_tokenizer, "config", None).encoder_downsample_rate if audio_tokenizer else None
        ) or 320

        # Use tokenizer's built-in chat template as primary
        self.chat_template = getattr(tokenizer, "chat_template", None) or chat_template

        # Read speech token range from tokenizer with fallback
        self.speech_token_range = (
            getattr(tokenizer, "speech_token_range", None) or speech_token_range or [151665, 152689]
        )
        self.audio_bos_token = getattr(tokenizer, "audio_bos_token", None) or audio_bos_token
        self.audio_eos_token = getattr(tokenizer, "audio_eos_token", None) or audio_eos_token
        self.audio_pad_token_id = getattr(tokenizer, "audio_pad_token_id", None) or audio_pad_token_id

        # Sample-level processor
        self.sample_processor = MossTTSDSampleProcessor(
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            audio_tokenizer=self.audio_tokenizer,
            chat_template=self.chat_template,
            speech_token_range=self.speech_token_range,
            audio_bos_token=self.audio_bos_token,
            audio_eos_token=self.audio_eos_token,
            audio_pad_token_id=self.audio_pad_token_id,
            max_channels=self.max_channels,
            input_sample_rate=self.input_sample_rate,
            encoder_downsample_rate=self.encoder_downsample_rate,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        """
        Instantiate a processor from a pretrained model.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The name of or path to the pretrained model.
            **kwargs:
                Additional keyword arguments passed to the respective component loaders.

        Returns:
            [`MossTTSDProcessor`]: A new instance of the processor.
        """
        audio_tokenizer_path = kwargs.pop(
            "audio_tokenizer_path", os.path.join(pretrained_model_name_or_path, "XY_Tokenizer")
        )
        if not isinstance(audio_tokenizer_path, str):
            raise ValueError(f"Unsupported audio_tokenizer_path input format: {type(audio_tokenizer_path)}")

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            audio_tokenizer_path, trust_remote_code=True, **kwargs
        )
        if not is_torch_available():
            raise ImportError(
                "XYTokenizer requires PyTorch to be installed. Please install it with `pip install torch`."
            )
        audio_tokenizer = XYTokenizer.from_pretrained(audio_tokenizer_path, trust_remote_code=True, **kwargs)

        return cls(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            audio_tokenizer=audio_tokenizer,
            **kwargs,
        )

    @classmethod
    def get_processor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        proc_dict, rest = super().get_processor_dict(pretrained_model_name_or_path, **kwargs)
        if "audio_tokenizer" in rest:
            proc_dict["audio_tokenizer"] = rest.pop("audio_tokenizer")
        for key in ("speech_token_range", "audio_bos_token", "audio_eos_token", "audio_pad_token_id"):
            if key in rest:
                proc_dict[key] = rest.pop(key)
        return proc_dict, rest

    def __call__(
        self,
        data: Union[dict[str, Any], list[dict[str, Any]]],
        **kwargs: Unpack[MossTTSDProcessorKwargs],
    ) -> BatchEncoding:
        """
        Main method to prepare inputs for the model from structured data.

        This method forwards the `data` and `kwargs` arguments to prepare inputs for MOSS-TTSD model. Please refer to the
        docstring of the respective methods for more information.

        Args:
            data (`dict` or `list[dict]`):
                Single dictionary or list of dictionaries containing input data. Expected keys include 'text',
                'prompt_text', 'prompt_audio', etc.
            **kwargs (`MossTTSDProcessorKwargs`):
                Additional processing arguments.

        Returns:
            [`BatchEncoding`]: Processed inputs ready for model consumption.
        """
        if isinstance(data, dict):
            data = [data]

        out_kwargs = self._merge_kwargs(MossTTSDProcessorKwargs, **kwargs)
        text_kwargs = out_kwargs["text_kwargs"]
        audio_kwargs = out_kwargs["audio_kwargs"]
        common_kwargs = out_kwargs["common_kwargs"]

        return_tensors = common_kwargs.get("return_tensors", "pt")
        padding = common_kwargs.get("padding", True)
        use_normalize = common_kwargs.get("use_normalize", False)

        pad_token_id = int(text_kwargs.get("pad_token_id", self.tokenizer.pad_token_id or 0))
        max_channels = int(audio_kwargs.get("max_channels", self.max_channels))
        audio_pad_token_id = int(audio_kwargs.get("audio_pad_token_id", self.audio_pad_token_id))
        silence_duration = float(audio_kwargs.get("silence_duration", 0.0))

        def _apply_chat_template(text: str, extra: dict) -> str:
            return self.apply_chat_template(conversation=None, text=text, **extra)

        samples: list[MossTTSDChatSample] = []
        for item in data:
            sample = self.sample_processor.prepare_sample(
                item,
                apply_chat_template=_apply_chat_template,
                use_normalize=use_normalize,
                silence_duration=silence_duration,
            )
            # Override with call-time max_channels (may differ from component initialization)
            if sample.input_ids_2d.shape[1] != max_channels:
                # Simplified: for clipping/extending channels, only pad/clip on the right side
                T, C = sample.input_ids_2d.shape
                if C > max_channels:
                    sample.input_ids_2d = sample.input_ids_2d[:, :max_channels]
                else:
                    pad = torch.full((T, max_channels - C), audio_pad_token_id, dtype=torch.long)
                    sample.input_ids_2d = torch.cat([sample.input_ids_2d, pad], dim=1)
            samples.append(sample)

        if not padding:
            raise NotImplementedError("Unpadded batches are not supported yet.")

        batch = self.sample_processor.collate(
            samples,
            pad_token_id=pad_token_id,
            audio_pad_token_id=audio_pad_token_id,
        )
        # Align with HiggsAudioProcessor: explicit dict -> BatchEncoding/Feature
        inputs = asdict(batch)
        inputs = {k: v for k, v in inputs.items() if v is not None}
        return BatchEncoding(inputs, tensor_type=return_tensors)

    def shifting_outputs(
        self,
        output_ids: "torch.Tensor",
        speech_token_range: list[int],
        max_channels: int = 8,
    ) -> "torch.Tensor":
        """
        Restore time-shifted layout to per-timestep C-channel arrangement and reverse-offset first codebook.

        Converts the time-shifted multi-channel output back to standard (batch, time, channels) format
        and maps the first codebook tokens back to their original space by subtracting the speech token offset.

        Args:
            output_ids: Time-shifted output tensor.
            speech_token_range: Speech token range for reverse mapping.
            max_channels: Number of codebook channels.

        Returns:
            Restored tensor with shape (batch, seq_len, max_channels).
        """
        seq_len = output_ids.shape[1] - max_channels + 1
        speech_ids = torch.full(
            (output_ids.shape[0], seq_len, max_channels), 0, dtype=output_ids.dtype, device=output_ids.device
        )
        for j in range(max_channels):
            speech_ids[..., j] = output_ids[:, j : seq_len + j, j]
            if j == 0:
                speech_ids[..., j] = speech_ids[..., j] - speech_token_range[0]
        return speech_ids

    def _find_max_valid_positions(self, data: "torch.Tensor", invalid_value: int = 1024):
        """
        Locate continuous valid audio segment intervals in each sequence (all non-text channels valid simultaneously).

        Identifies contiguous spans where all audio channels (columns 1+) contain valid tokens
        (not the invalid_value padding token).

        Args:
            data: Input tensor with shape (batch, time, channels).
            invalid_value: Token ID considered as invalid/padding.

        Returns:
            List of lists containing valid audio segments for each sequence in the batch.
        """
        mask = torch.all(data[:, :, 1:] != invalid_value, dim=2)
        valid_indices = torch.where(mask)
        result = [[] for _ in range(len(data))]
        if valid_indices[0].numel() == 0:
            return result
        grouped = []
        group_ids = []
        for i, seq_no in enumerate(valid_indices[0]):
            pos = valid_indices[1][i]
            if not group_ids or seq_no > group_ids[-1]:
                group_ids.append(seq_no)
                grouped.append([[pos, pos + 1]])
            elif pos == grouped[-1][-1][-1]:
                grouped[-1][-1][-1] += 1
            else:
                grouped[-1].append([pos, pos + 1])
        for gid, spans in zip(group_ids, grouped):
            for s, e in spans:
                result[gid].append(data[gid, s:e, :])
        return result

    def batch_decode(self, token_ids: "torch.Tensor", *args, **kwargs):
        """
        Decode a batch of token sequences into text and audio outputs.

        This method forwards the `token_ids` and `kwargs` arguments to decode text and audio outputs from the model.
        Please refer to the docstring of the respective methods for more information.

        Args:
            token_ids (`torch.Tensor`):
                Token tensor with shape (batch, time, channels).
            *args:
                Additional arguments passed to tokenizer.batch_decode.
            **kwargs:
                Additional keyword arguments passed to tokenizer.batch_decode.

        Returns:
            `tuple`: Tuple of (text_list, audio_list) where text_list contains decoded text strings and audio_list
                contains decoded audio arrays for each sequence.
        """
        assert token_ids.ndim == 3 and token_ids.shape[2] == self.max_channels
        text = self.tokenizer.batch_decode(token_ids[:, :, 0], *args, **kwargs)
        normal = self.shifting_outputs(token_ids, self.speech_token_range, self.max_channels)
        audio_frags = self._find_max_valid_positions(normal, self.audio_pad_token_id)
        decode_audio = []
        for seq_frags in audio_frags:
            if len(seq_frags):
                frag = torch.cat([f.permute(1, 0).unsqueeze(1) for f in seq_frags], dim=1)
                decode_audio.append(self.audio_tokenizer.decode(frag, overlap_seconds=10)["audio_values"])
            else:
                decode_audio.append([])
        return text, decode_audio

    def decode(self, token_ids: "torch.Tensor", *args, **kwargs) -> MossTTSDResponse:
        """
        Decode a single sequence of token IDs into text and audio.

        This method forwards the `token_ids` and `kwargs` arguments to decode a single sequence. Please refer to the
        docstring of the respective methods for more information.

        Args:
            token_ids (`torch.Tensor`):
                Token tensor with shape (time, channels).
            *args:
                Additional arguments passed to tokenizer.decode.
            **kwargs:
                Additional keyword arguments passed to tokenizer.decode.

        Returns:
            [`MossTTSDResponse`]: Response object containing generated text, audio, and sampling rate.
        """
        assert token_ids.ndim == 2 and token_ids.shape[1] == self.max_channels
        text = self.tokenizer.decode(token_ids[:, 0].squeeze(-1), *args, **kwargs)
        normal = self.shifting_outputs(token_ids.unsqueeze(0), self.speech_token_range, self.max_channels)
        audio_frags = self._find_max_valid_positions(normal, self.audio_pad_token_id)[0]
        if len(audio_frags):
            frag = torch.cat([f.permute(1, 0).unsqueeze(1) for f in audio_frags], dim=1)
            audio = self.audio_tokenizer.decode(frag, overlap_seconds=10)["audio_values"]
        else:
            audio = None
        return MossTTSDResponse(
            audio=None if audio is None else audio.detach().cpu().numpy(),
            generated_text=text,
            sampling_rate=self.output_sample_rate,
        )

    def save_audio(self, audios, output_dir="output", prefix="audio"):
        """
        Save multiple audio fragments to files.

        Args:
            audios: List of audio data fragments from batch_decode
            output_dir (str): Directory to save audio files
            prefix (str): Prefix for audio filenames
        """
        if not is_torchaudio_available():
            raise ImportError("Please install `torchaudio` to save audio files.")

        os.makedirs(output_dir, exist_ok=True)

        for i, data in enumerate(audios):
            for j, fragment in enumerate(data):
                filename = f"{output_dir}/{prefix}_{i}_{j}.wav"
                torchaudio.save(filename, fragment.cpu(), self.output_sample_rate)


__all__ = ["MossTTSDProcessor"]
