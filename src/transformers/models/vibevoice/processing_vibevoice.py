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
import torch

from ...audio_utils import make_list_of_audio
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType, logging
from .feature_extraction_vibevoice import VibeVoiceFeatureExtractor, normalize_audio
from .tokenization_vibevoice_fast import VibeVoiceTextTokenizerFast


logger = logging.get_logger(__name__)


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
        text: Optional[Union[str, list[str], TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
        voice_samples: Optional[Union[list[Union[str, np.ndarray]], list[list[Union[str, np.ndarray]]]]] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Main method to process one or more podcast scripts with optional voice samples.

        Args:
            text (`str`, `List[str]`):
                The input text(s) to process. Can be:
                - A single script string
                - A list of script strings for batch processing
                - A path to a .json or .txt file
                - A list of paths
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
            return_tensors (`str` or `TensorType`, *optional*):
                If set, will return tensors of a particular framework
            return_attention_mask (`bool`, defaults to `True`):
                Whether to return the attention mask

        Returns:
            `BatchEncoding`: A BatchEncoding with the following fields:
                - **input_ids** -- List of token id sequences or tensor
                - **attention_mask** -- List of attention masks or tensor
                - **speech_tensors** -- Padded speech inputs (if voice_samples provided)
                - **speech_masks** -- Speech masks (if voice_samples provided)
                - **speech_input_mask** -- Boolean masks indicating speech token positions
        """
        # Handle single vs batch input
        if isinstance(text, str) or (isinstance(text, list) and len(text) > 0 and not isinstance(text[0], str)):
            texts = [text]
        else:
            texts = text

        # Handle voice samples (list of lists for each script)
        if voice_samples is not None:
            voice_samples_list = [make_list_of_audio(_voices) for _voices in voice_samples]
            if len(texts) != len(voice_samples_list):
                raise ValueError(f"Got {len(texts)} texts but {len(voice_samples)} audios; they must match 1:1.")
        else:
            voice_samples_list = [None] * len(texts)

        # Process each input
        all_encodings = []
        for text_input, voice_input in zip(texts, voice_samples_list):
            encoding = self._process_single(text_input, voice_input)
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

        return batch_encoding

    def _process_single(
        self,
        text: Union[str, TextInput],
        voice_samples: Optional[list[Union[str, np.ndarray]]] = None,
    ) -> dict[str, Any]:
        """Process a single podcast script."""
        # Determine if text is a file path or direct script
        script = None
        if isinstance(text, str):
            # Check if it's a file path
            if text.endswith('.json') and os.path.exists(text):
                script = self._convert_json_to_script(text)
            elif text.endswith('.txt') and os.path.exists(text):
                script = self._convert_text_to_script(text)
            else:
                # Assume it's the script content directly
                script = text

        if script is None:
            raise ValueError(f"Could not process input text: {text}")

        # Parse the script
        parsed_lines = self._parse_script(script)
        all_speakers = list(set(speaker_id for speaker_id, _ in parsed_lines))

        # Create system prompt
        # system_tokens = self.tokenizer.encode(self.system_prompt, add_special_tokens=False)
        system_tokens = self.tokenizer.encode(self.system_prompt)

        # Process voice samples if provided
        if voice_samples:
            voice_tokens, voice_speech_inputs, voice_speech_masks = self._create_voice_prompt(voice_samples[:len(all_speakers)])
        else:
            voice_tokens, voice_speech_inputs, voice_speech_masks = [], [], []

        # Build full token sequence
        full_tokens = system_tokens + voice_tokens
        speech_input_mask = [False] * len(system_tokens) + voice_speech_masks

        # Add text input section
        full_tokens += self.tokenizer.encode(' Text input:\n', add_special_tokens=False)
        speech_input_mask += [False] * len(self.tokenizer.encode(' Text input:\n', add_special_tokens=False))

        for speaker_id, speaker_text in parsed_lines:
            speaker_text_tokens = self.tokenizer.encode(f" Speaker {speaker_id}:{speaker_text}\n", add_special_tokens=False)
            full_tokens += speaker_text_tokens
            speech_input_mask += [False] * len(speaker_text_tokens)

        # Add speech output section
        full_tokens += self.tokenizer.encode(' Speech output:\n', add_special_tokens=False) + [self.tokenizer.speech_start_id]
        speech_input_mask += [False] * (len(self.tokenizer.encode(' Speech output:\n', add_special_tokens=False)) + 1)

        return {
            "input_ids": full_tokens,
            "speech_inputs": voice_speech_inputs if voice_speech_inputs else None,
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
    ) -> BatchEncoding:
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

        # Process speech inputs
        all_speech_inputs = []
        has_speech = False
        for enc in encodings:
            if enc["speech_inputs"] is not None:
                all_speech_inputs.extend(enc["speech_inputs"])
                has_speech = True

        # Prepare batch encoding
        batch_encoding = BatchEncoding()

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

        # Process speech tensors if present
        if has_speech:
            speech_dict = self.prepare_speech_inputs(
                all_speech_inputs,
                return_tensors=return_tensors,
            )
            batch_encoding["speech_tensors"] = speech_dict["padded_speeches"]
            batch_encoding["speech_masks"] = speech_dict["speech_masks"]
        else:
            batch_encoding["speech_tensors"] = None
            batch_encoding["speech_masks"] = None

        # Add metadata
        batch_encoding["parsed_scripts"] = [enc["parsed_script"] for enc in encodings]
        batch_encoding["all_speakers_list"] = [enc["all_speakers"] for enc in encodings]

        return batch_encoding

    def _create_voice_prompt(
        self,
        speaker_samples: list[Union[str, np.ndarray]]
    ) -> tuple[list[int], list[np.ndarray], list[bool]]:
        """
        Create voice prompt tokens and process audio samples.
        
        Returns:
            tuple: (voice_tokens, voice_speech_inputs, voice_speech_masks)
        """
        vae_token_id = self.tokenizer.speech_diffusion_id

        voice_full_tokens = self.tokenizer.encode(' Voice input:\n', add_special_tokens=False)
        voice_speech_inputs = []
        voice_speech_masks = [False] * len(voice_full_tokens)

        for speaker_id, speaker_audio in enumerate(speaker_samples):
            prefix_tokens = self.tokenizer.encode(f" Speaker {speaker_id}:", add_special_tokens=False)

            # Process audio
            wav = np.array(speaker_audio, dtype=np.float32)

            # Apply normalization if needed (TODO: use feature extractor)
            if self.db_normalize:
                wav = normalize_audio(wav)

            import pudb; pudb.set_trace()  # DEBUGGING

            # Calculate token length based on compression ratio
            # if speaker_audio.endswith('.pt') or speaker_audio.endswith('.npy'):
            #     vae_tok_len = wav.shape[0]
            # else:
            vae_tok_len = math.ceil(wav.shape[0] / self.speech_tok_compress_ratio)

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

            voice_full_tokens.extend(speaker_tokens)
            voice_speech_masks.extend(vae_input_mask)
            voice_speech_inputs.append(wav)

        return voice_full_tokens, voice_speech_inputs, voice_speech_masks

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
        vae_tok_seqlens = [math.ceil(s.shape[0] / self.speech_tok_compress_ratio) for s in speech_inputs]
        # vae_tok_seqlens = [math.ceil(s.shape[0] / self.speech_tok_compress_ratio) if s.ndim == 1 else s.shape[0] for s in speech_inputs]
        max_speech_length = max(s.shape[0] for s in speech_inputs)

        # Pad speeches
        if speech_inputs[0].ndim == 1:
            padded_speeches = np.full((len(speech_inputs), max_speech_length), fill_value=0, dtype=np.float32)
        else:
            padded_speeches = np.full((len(speech_inputs), max_speech_length, speech_inputs[0].shape[-1]), fill_value=0, dtype=np.float32)
        speech_masks = np.zeros((len(speech_inputs), max(vae_tok_seqlens)), dtype=np.bool_)

        for i, (speech, vae_tok_length) in enumerate(zip(speech_inputs, vae_tok_seqlens)):
            padded_speeches[i, :len(speech)] = speech
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

    def _convert_json_to_script(self, json_file: str) -> str:
        """
        Convert JSON format to script format.
        Expected JSON format:
        [
            {"speaker": "1", "text": "Hello everyone..."},
            {"speaker": "2", "text": "Great to be here..."}
        ]
        """

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of speaker entries")

        script_lines = []
        for item in data:
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dict entry: {item}")
                continue

            speaker = item.get('speaker')
            text = item.get('text')

            if speaker is None or text is None:
                logger.warning(f"Skipping entry missing speaker or text: {item}")
                continue

            # Ensure speaker ID is valid
            try:
                speaker_id = int(speaker)
            except (ValueError, TypeError):
                logger.warning(f"Invalid speaker ID: {speaker}, skipping entry")
                continue

            # Clean up text
            text = text.strip()
            if text:
                script_lines.append(f"Speaker {speaker_id}: {text}")

        if not script_lines:
            raise ValueError("No valid entries found in JSON file")

        return "\n".join(script_lines)

    def _convert_text_to_script(self, text_file: str) -> str:
        """
        Convert text file to script format.
        Handles multiple formats:
        1. Already formatted as "Speaker X: text"
        2. Plain text (assigns to Speaker 1)
        
        Handles edge cases like multiple colons in a line.
        """
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        script_lines = []
        current_speaker = 1

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to parse as "Speaker X: text" format
            # Use regex to be more robust
            speaker_match = re.match(r'^Speaker\s+(\d+)\s*:\s*(.*)$', line, re.IGNORECASE)

            if speaker_match:
                speaker_id = int(speaker_match.group(1))
                text = speaker_match.group(2).strip()
                if text:
                    script_lines.append(f"Speaker {speaker_id}: {text}")
            else:
                # Treat as plain text - assign to current speaker
                script_lines.append(f"Speaker {current_speaker}: {line}")

        if not script_lines:
            raise ValueError("No valid content found in text file")

        return "\n".join(script_lines)

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

    def save_audio(self,
        audio: Union[torch.Tensor, np.ndarray, list[Union[torch.Tensor, np.ndarray]]],
        output_path: str = "output.wav",
        sampling_rate: Optional[int] = None,
        normalize: bool = False,
        batch_prefix: str = "audio_",
    ) -> str:
        """
        Save audio data to a file.
        Args:
            audio (Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]]):
                The audio data to save. Can be a single tensor/array or a list of them.
            output_path (str, optional): Path to save the audio file. Defaults to "output.wav".
            sampling_rate (int, optional): Sampling rate for the audio. If None, uses the processor's default.
            normalize (bool, optional): Whether to normalize the audio before saving. Defaults to False.
            batch_prefix (str, optional): Prefix for batch audio files. Defaults to "audio_".
        Returns:
            str: The path to the saved audio file.
        """
        return self.audio_processor.save_audio(audio, output_path=output_path, sampling_rate=sampling_rate, normalize=normalize, batch_prefix=batch_prefix)

__all__ = ["VibeVoiceProcessor"]
