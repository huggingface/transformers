# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import math
import os
import re
from typing import Any, Optional, Union

import numpy as np

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin, ProcessingKwargs, Unpack
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType, logging, is_soundfile_available, is_torch_available
from .feature_extraction_vibevoice import VibeVoiceFeatureExtractor
from .tokenization_vibevoice_fast import VibeVoiceTextTokenizerFast


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

if is_soundfile_available():
    import soundfile as sf


class VibeVoiceProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "right",
            "add_special_tokens": False,
        },
        "audio_kwargs": {
            "sampling_rate": 24000,
            "padding": True,
            "return_attention_mask": True,
            # "eos_token_id": 1024,
            # "pad_token_id": 1025,
            # "bos_token_id": 1026,
            # "delay_pattern": [0, 8, 9, 10, 11, 12, 13, 14, 15],
            # "generation": True,
            
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class VibeVoiceProcessor(ProcessorMixin):
    r"""
    Constructs a VibeVoice processor which wraps a VibeVoice tokenizer and audio processor into a single processor.

    [`VibeVoiceProcessor`] offers all the functionalities of [`VibeVoiceTokenizer`] and [`VibeVoiceTokenizerProcessor`]. 
    See the [`~VibeVoiceProcessor.__call__`] and [`~VibeVoiceProcessor.decode`] for more information.

    Args:
        tokenizer (`VibeVoiceTextTokenizer` or `VibeVoiceTextTokenizerFast`):
            The tokenizer for text processing.
        audio_processor (`VibeVoiceFeatureExtractor`):
            The audio processor for speech processing.
        speech_tok_compress_ratio (`int`, *optional*, defaults to 3200):
            The compression ratio for speech tokenization.
        db_normalize (`bool`, *optional*, defaults to True):
            Whether to apply decibel normalization to audio inputs.
    """
    # TODO `audio_processor` or `feature_extractor`
    # TODO: add audio tokenizer?
    attributes = ["audio_processor", "tokenizer"]
    audio_processor_class = "VibeVoiceFeatureExtractor"
    tokenizer_class = ("VibeVoiceTextTokenizer", "VibeVoiceTextTokenizerFast")

    def __init__(self, audio_processor, tokenizer, speech_tok_compress_ratio=3200, db_normalize=True, **kwargs):
        super().__init__(audio_processor, tokenizer)
        self.speech_tok_compress_ratio = speech_tok_compress_ratio
        self.db_normalize = db_normalize
        self.system_prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Instantiate a VibeVoiceProcessor from a pretrained VibeVoice processor.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:
                - a string, the *model id* of a pretrained model
                - a path to a *directory* containing processor config

        Returns:
            [`VibeVoiceProcessor`]: The processor object instantiated from pretrained model.
        """

        # Load processor configuration
        config_path = os.path.join(pretrained_model_name_or_path, "preprocessor_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            logger.warning(f"No preprocessor_config.json found at {pretrained_model_name_or_path}, using defaults")
            config = {
                "speech_tok_compress_ratio": 3200,
                "db_normalize": True,
            }

        # Extract main processor parameters
        speech_tok_compress_ratio = config.get("speech_tok_compress_ratio", 3200)
        db_normalize = config.get("db_normalize", True)

        # Load tokenizer - try from model path first, then fallback to Qwen
        language_model_pretrained_name = config.get("language_model_pretrained_name") or kwargs.pop("language_model_pretrained_name", "Qwen/Qwen2.5-1.5B")
        logger.info(f"Loading tokenizer from {language_model_pretrained_name}")
        if 'qwen' in language_model_pretrained_name.lower():
            tokenizer = VibeVoiceTextTokenizerFast.from_pretrained(
                language_model_pretrained_name,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported tokenizer type for {language_model_pretrained_name}. Supported types: Qwen, Llama, Gemma.")

        # Load audio processor
        if "audio_processor" in config:
            # Create audio processor from config
            audio_config = config["audio_processor"]
            audio_processor = VibeVoiceFeatureExtractor(
                sampling_rate=audio_config.get("sampling_rate", 24000),
                normalize_audio=audio_config.get("normalize_audio", True),
                target_dB_FS=audio_config.get("target_dB_FS", -25),
                eps=audio_config.get("eps", 1e-6),
            )
        else:
            # Create default audio processor
            audio_processor = VibeVoiceFeatureExtractor()

        # Create and return the processor
        return cls(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
            speech_tok_compress_ratio=speech_tok_compress_ratio,
            db_normalize=db_normalize,
        )

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """
        Save a processor to a directory, so that it can be re-loaded using the
        [`~VibeVoiceProcessor.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the processor will be saved.
        """

        os.makedirs(save_directory, exist_ok=True)

        # Save processor configuration
        processor_config = {
            "processor_class": "VibeVoiceProcessor",
            "speech_tok_compress_ratio": self.speech_tok_compress_ratio,
            "db_normalize": self.db_normalize,
            "audio_processor": {
                "feature_extractor_type": "VibeVoiceTokenizerProcessor",
                "sampling_rate": getattr(self.audio_processor, 'sampling_rate', 24000),
                "normalize_audio": getattr(self.audio_processor, 'normalize_audio', True),
                "target_dB_FS": getattr(self.audio_processor, 'target_dB_FS', -25),
                "eps": getattr(self.audio_processor, 'eps', 1e-6),
            }
        }

        config_path = os.path.join(save_directory, "preprocessor_config.json")
        with open(config_path, 'w') as f:
            json.dump(processor_config, f, indent=2)

        logger.info(f"Processor configuration saved in {config_path}")

    def __call__(
        self,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]],
        voice_samples: Optional[AudioInput] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_attention_mask: bool = True,
        **kwargs: Unpack[VibeVoiceProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to process one or more podcast scripts with optional voice samples.

        Args:
            text (`str`, `List[str]`):
                The input text(s) to process. Can be:
                - A single script string
                - A list of script strings for batch processing
            voice_samples (`List[Union[str, np.ndarray]]`, `List[List[Union[str, np.ndarray]]]`, *optional*):
                Voice samples for each script. Can be:
                - A list of samples for a single script
                - A list of lists for batch processing
            padding (`bool`, `str` or `PaddingStrategy`, defaults to `True`):
                Whether to pad sequences to the same length
            truncation (`bool`, `str` or `TruncationStrategy`, defaults to `False`):
                Whether to truncate sequences
            max_length (`int`, *optional*):
                Maximum length of the returned sequences
            return_attention_mask (`bool`, defaults to `True`):
                Whether to return the attention mask

        Returns:
            `BatchFeature`: A BatchFeature with the following fields:
                - **input_ids** -- List of token id sequences or tensor
                - **attention_mask** -- List of attention masks or tensor
                - **speech_tensors** -- Padded speech inputs (if voice_samples provided)
                - **speech_masks** -- Speech masks (if voice_samples provided)
                - **speech_input_mask** -- Boolean masks indicating speech token positions
        """
        # Merge defaults with user kwargs
        call_kwargs = self._merge_kwargs(
            VibeVoiceProcessorKwargs,
            **kwargs,
        )

        text_kwargs = call_kwargs["text_kwargs"]
        audio_kwargs = call_kwargs["audio_kwargs"]
        common_kwargs = call_kwargs["common_kwargs"]
        return_tensors = common_kwargs.pop("return_tensors", None)
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        # Handle text
        if isinstance(text, str) or (isinstance(text, list) and len(text) > 0 and not isinstance(text[0], str)):
            texts = [text]
        else:
            texts = text

        # Handle voice samples (list of lists for each script)
        processed_audio = None
        if voice_samples is not None:
            voice_samples_list = [make_list_of_audio(_voices) for _voices in voice_samples]
            if len(texts) != len(voice_samples_list):
                raise ValueError(f"Got {len(texts)} texts but {len(voice_samples)} audio lists; they must match 1:1.")
        
            # flatten list of audio for audio processing
            voices = [voice for _voices in voice_samples_list for voice in _voices]

            # TODO remove duplicate voices based on speaker ID (need to parse script first then)
            processed_audio = self.audio_processor(voices, **audio_kwargs)
            padding_masks = processed_audio["padding_mask"]

            # Create speech masks for audio tokenizer based on its compression ratio
            if isinstance(padding_masks, torch.Tensor):
                vae_tok_seqlens = torch.ceil(padding_masks.sum(dim=-1) / self.speech_tok_compress_ratio).int().tolist()
                speech_masks = torch.zeros((len(padding_masks), max(vae_tok_seqlens)), dtype=torch.bool)
            else:
                vae_tok_seqlens = np.ceil(np.sum(padding_masks, axis=-1) / self.speech_tok_compress_ratio).astype(int).tolist()
                speech_masks = np.zeros((len(padding_masks), max(vae_tok_seqlens)), dtype=np.bool_)
            for i, seq_len in enumerate(vae_tok_seqlens):
                speech_masks[i, :seq_len] = True
        
        else:
            voice_samples_list = [None] * len(texts)

        # Process each input
        # TODO prepare text here instead of in process_single,
        # TODO extract speaker ID for each script to prepare the text
        all_encodings = []
        for text_input, voice_input in zip(texts, voice_samples_list):
            encoding = self._process_single(text_input, voice_input, audio_kwargs)
            all_encodings.append(encoding)

        # Combine batch
        batch_encoding = self._batch_encode(
            all_encodings,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
        )
        if processed_audio is not None:
            batch_encoding["speech_tensors"] = processed_audio["audio"]
            batch_encoding["speech_masks"] = speech_masks
        else:
            batch_encoding["speech_tensors"] = None
            batch_encoding["speech_masks"] = None

        return batch_encoding

    def _process_single(
        self,
        text: Union[str, TextInput],
        voice_samples: Optional[list[Union[str, np.ndarray]]] = None,
        audio_kwargs: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Process a single podcast script."""
        script = text

        # Parse the script
        parsed_lines = self._parse_script(script)
        all_speakers = list(set(speaker_id for speaker_id, _ in parsed_lines))

        # Process voice samples if provided
        if voice_samples:

            # pad and normalize audio
            wav = self.audio_processor(
                voice_samples[:len(all_speakers)], 
                **audio_kwargs,
            )
            voice_speech_padding = wav["padding_mask"]

            # prepare speech tokens
            vae_token_id = self.tokenizer.speech_diffusion_id
            voice_tokens = self.tokenizer.encode(' Voice input:\n', add_special_tokens=False)
            voice_speech_masks = [False] * len(voice_tokens)

            for speaker_id, padding_mask in enumerate(voice_speech_padding):
                prefix_tokens = self.tokenizer.encode(f" Speaker {speaker_id}:", add_special_tokens=False)

                # Calculate token length based on compression ratio
                vae_tok_len = math.ceil(padding_mask.sum() / self.speech_tok_compress_ratio)

                # Build tokens and masks
                speaker_tokens = (prefix_tokens +
                                [self.tokenizer.speech_start_id] +
                                [vae_token_id] * vae_tok_len +
                                [self.tokenizer.speech_end_id] +
                                self.tokenizer.encode('\n', add_special_tokens=False))

                vae_input_mask = ([False] * len(prefix_tokens) +
                                [False] +
                                [True] * vae_tok_len +
                                [False] +
                                [False])

                voice_tokens.extend(speaker_tokens)
                voice_speech_masks.extend(vae_input_mask)

        else:
            voice_tokens, voice_speech_masks = [], []

        # Build full token sequence
        # -- Start with system prompt
        # system_tokens = self.tokenizer.encode(self.system_prompt, add_special_tokens=False)
        system_tokens = self.tokenizer.encode(self.system_prompt)
        full_tokens = system_tokens + voice_tokens
        speech_input_mask = [False] * len(system_tokens) + voice_speech_masks

        # -- Add text input section
        full_tokens += self.tokenizer.encode(' Text input:\n', add_special_tokens=False)
        speech_input_mask += [False] * len(self.tokenizer.encode(' Text input:\n', add_special_tokens=False))

        for speaker_id, speaker_text in parsed_lines:
            speaker_text_tokens = self.tokenizer.encode(f" Speaker {speaker_id}:{speaker_text}\n", add_special_tokens=False)
            full_tokens += speaker_text_tokens
            speech_input_mask += [False] * len(speaker_text_tokens)

        # -- Add speech output section
        full_tokens += self.tokenizer.encode(' Speech output:\n', add_special_tokens=False) + [self.tokenizer.speech_start_id]
        speech_input_mask += [False] * (len(self.tokenizer.encode(' Speech output:\n', add_special_tokens=False)) + 1)

        return {
            "input_ids": full_tokens,
            "speech_input_mask": speech_input_mask,
            "parsed_script": parsed_lines,
            "all_speakers": all_speakers,
        }

    def _batch_encode(
        self,
        encodings: list[dict[str, Any]],
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: bool = True,
    ) -> BatchFeature:
        """Combine multiple encodings into a batch with padding."""
        # Extract input_ids and create attention_mask
        input_ids_list = [enc["input_ids"] for enc in encodings]
        speech_input_masks_list = [enc["speech_input_mask"] for enc in encodings]

        # Determine padding strategy
        if isinstance(padding, bool):
            padding_strategy = PaddingStrategy.LONGEST if padding else PaddingStrategy.DO_NOT_PAD
        elif isinstance(padding, str):
            padding_strategy = PaddingStrategy(padding)
        else:
            padding_strategy = padding

        # Apply padding to input_ids
        if padding_strategy != PaddingStrategy.DO_NOT_PAD:
            if padding_strategy == PaddingStrategy.LONGEST:
                max_len = max(len(ids) for ids in input_ids_list)
            elif padding_strategy == PaddingStrategy.MAX_LENGTH and max_length is not None:
                max_len = max_length
            else:
                max_len = max(len(ids) for ids in input_ids_list)

            # Pad sequences
            padded_input_ids = []
            attention_masks = []
            padded_speech_input_masks = []

            for input_ids, speech_mask in zip(input_ids_list, speech_input_masks_list):
                # Truncate if needed
                if truncation and len(input_ids) > max_len:
                    input_ids = input_ids[:max_len]
                    speech_mask = speech_mask[:max_len]

                # Pad
                padding_length = max_len - len(input_ids)
                # padded_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
                padded_ids = [self.tokenizer.pad_id] * padding_length + input_ids
                attention_mask = [0] * padding_length + [1] * len(input_ids)
                padded_speech_mask = [False] * padding_length + speech_mask

                padded_input_ids.append(padded_ids)
                attention_masks.append(attention_mask)
                padded_speech_input_masks.append(padded_speech_mask)

            input_ids_list = padded_input_ids
            speech_input_masks_list = padded_speech_input_masks
        else:
            # No padding, just create attention masks
            attention_masks = [[1] * len(ids) for ids in input_ids_list] if return_attention_mask else None

        # Prepare batch encoding
        batch_encoding = BatchFeature()

        # Handle tensor conversion
        if return_tensors is not None:
            batch_encoding["input_ids"] = torch.tensor(input_ids_list, dtype=torch.long)
            if return_attention_mask and attention_masks is not None:
                batch_encoding["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long)
            batch_encoding["speech_input_mask"] = torch.tensor(speech_input_masks_list, dtype=torch.bool)
        else:
            batch_encoding["input_ids"] = input_ids_list
            if return_attention_mask and attention_masks is not None:
                batch_encoding["attention_mask"] = attention_masks
            batch_encoding["speech_input_mask"] = speech_input_masks_list

        # Add metadata
        batch_encoding["parsed_scripts"] = [enc["parsed_script"] for enc in encodings]
        batch_encoding["all_speakers_list"] = [enc["all_speakers"] for enc in encodings]

        return batch_encoding

    def prepare_speech_inputs(
        self,
        speech_inputs: list[np.ndarray],
        return_tensors: Optional[Union[str, TensorType]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> dict[str, Any]:
        """
        Prepare speech inputs for model consumption.
        
        Args:
            speech_inputs: List of speech arrays
            return_tensors: Output tensor type
            device: Device to place tensors on
            dtype: Data type for tensors
            
        Returns:
            Dictionary with padded_speeches and speech_masks
        """
        if not speech_inputs:
            return {"padded_speeches": None, "speech_masks": None}

        # Calculate sequence lengths
        vae_tok_seqlens = [math.ceil(s.shape[1] / self.speech_tok_compress_ratio) for s in speech_inputs]
        # vae_tok_seqlens = [math.ceil(s.shape[0] / self.speech_tok_compress_ratio) if s.ndim == 1 else s.shape[0] for s in speech_inputs]
        max_speech_length = max(s.shape[1] for s in speech_inputs)

        # Pad speeches, TODO remove as feature extractor should handle this
        if speech_inputs[0].shape[0] == 1:
            padded_speeches = np.full((len(speech_inputs), max_speech_length), fill_value=0, dtype=np.float32)
        else:
            padded_speeches = np.full((len(speech_inputs), max_speech_length, speech_inputs[0].shape[0]), fill_value=0, dtype=np.float32)
        speech_masks = np.zeros((len(speech_inputs), max(vae_tok_seqlens)), dtype=np.bool_)

        for i, (speech, vae_tok_length) in enumerate(zip(speech_inputs, vae_tok_seqlens)):
            padded_speeches[i, :speech.shape[1]] = speech
            speech_masks[i, :vae_tok_length] = True

        result = {
            "padded_speeches": padded_speeches,
            "speech_masks": speech_masks,
        }

        # Convert to tensors if requested
        if return_tensors == "pt":
            result["padded_speeches"] = torch.tensor(padded_speeches, device=device, dtype=dtype or torch.float32)
            result["speech_masks"] = torch.tensor(speech_masks, device=device, dtype=torch.bool)

        return result

    def _parse_script(self, script: str) -> list[tuple[int, str]]:
        """Parse script into list of (speaker_id, text) tuples."""
        lines = script.strip().split("\n")
        parsed_lines = []
        speaker_ids = []

        # First pass: parse all lines and collect speaker IDs
        for line in lines:
            if not line.strip():
                continue

            # Use regex to handle edge cases like multiple colons
            match = re.match(r'^Speaker\s+(\d+)\s*:\s*(.*)$', line.strip(), re.IGNORECASE)

            if match:
                speaker_id = int(match.group(1))
                text = ' ' + match.group(2).strip()
                parsed_lines.append((speaker_id, text))
                speaker_ids.append(speaker_id)
            else:
                logger.warning(f"Could not parse line: '{line}'")

        if not parsed_lines:
            raise ValueError("No valid speaker lines found in script")

        # Check if we need to normalize speaker IDs (only if all are > 0)
        min_speaker_id = min(speaker_ids)
        if min_speaker_id > 0:
            # Normalize to start from 0
            normalized_lines = []
            for speaker_id, text in parsed_lines:
                normalized_lines.append((speaker_id - 1, text))
            return normalized_lines
        else:
            # Keep original IDs
            return parsed_lines

    def _merge_inputs(self, text_inputs: BatchEncoding, audio_inputs: dict) -> BatchEncoding:
        """Merge text and audio inputs into a single BatchEncoding."""
        # Start with text inputs
        merged = BatchEncoding(text_inputs)

        # Add audio-specific fields
        if "audio" in audio_inputs:
            merged["speech_inputs"] = audio_inputs["audio"]
        if "streaming" in audio_inputs:
            merged["streaming"] = audio_inputs["streaming"]

        return merged

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to VibeVoiceTextTokenizer's [`~PreTrainedTokenizer.batch_decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to VibeVoiceTextTokenizer's [`~PreTrainedTokenizer.decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """
        Return the list of inputs accepted by the model.
        """
        tokenizer_input_names = self.tokenizer.model_input_names
        audio_processor_input_names = self.audio_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + audio_processor_input_names + ["speech_inputs", "speech_input_mask"]))

    def save_audio(
        self,
        audio: Union[torch.Tensor, np.ndarray, list[Union[torch.Tensor, np.ndarray]]],
        output_path: str = "output.wav",
        sampling_rate: Optional[int] = None,
    ) -> list[str]:
        """
        Save audio data to WAV file(s).
        
        Args:
            audio: Audio data to save (tensor, array, or list of them)
            output_path: Output file path or directory for multiple files
            sampling_rate: Sampling rate for the saved audio
                
        Returns:
            List[str]: Paths to the saved audio files.
        """
        if sampling_rate is None:
            sampling_rate = self.audio_processor.sampling_rate

        if not is_soundfile_available():
            raise ImportError("Please install `soundfile` to save audio files.")

        # Convert various audio formats to list of 1D numpy arrays
        if isinstance(audio, list):
            # List of tensors/arrays
            audio_arrays = []
            for item in audio:
                if isinstance(item, torch.Tensor):
                    audio_arrays.append(item.detach().cpu().numpy().squeeze())
                else:
                    audio_arrays.append(np.array(item).squeeze())
        else:
            # Single tensor or array
            if isinstance(audio, torch.Tensor):
                audio_np = audio.detach().cpu().numpy()
            else:
                audio_np = np.array(audio)
            
            # Handle different shapes
            if audio_np.ndim == 1:
                audio_arrays = [audio_np]
            elif audio_np.ndim == 2:
                # Could be (batch, time) or (channels, time)
                if audio_np.shape[0] <= 2:  # Assume channels if <= 2
                    audio_arrays = [audio_np.mean(axis=0)]  # Convert to mono
                else:  # Assume batch dimension
                    audio_arrays = [audio_np[i] for i in range(audio_np.shape[0])]
            elif audio_np.ndim == 3:
                # (batch, channels, time) - extract each item and convert to mono
                audio_arrays = [audio_np[i].mean(axis=0) for i in range(audio_np.shape[0])]
            else:
                raise ValueError(f"Unsupported audio shape: {audio_np.shape}")
        
        # Save audio(s)
        if len(audio_arrays) == 1:
            # Single audio file
            sf.write(output_path, audio_arrays[0], sampling_rate)
        else:
            # Multiple audio files - save to directory
            os.makedirs(output_path, exist_ok=True)
            saved_paths = []
            for i, audio_array in enumerate(audio_arrays):
                file_path = os.path.join(output_path, f"audio_{i}.wav")
                sf.write(file_path, audio_array, sampling_rate)
                saved_paths.append(file_path)


__all__ = ["VibeVoiceProcessor"]
