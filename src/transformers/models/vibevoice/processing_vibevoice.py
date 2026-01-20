# Copyright 2025 Microsoft and The HuggingFace Team. All rights reserved.
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
Processor class for VibeVoice.
"""

import json
import os

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TruncationStrategy
from ...utils import TensorType, auto_docstring, logging
from ...utils.hub import cached_file
from ..auto import AutoTokenizer


logger = logging.get_logger(__name__)


@auto_docstring
class VibeVoiceProcessor(ProcessorMixin):
    r"""
    Constructs a VibeVoice processor which wraps a Qwen2 tokenizer and handles speaker embeddings.

    [`VibeVoiceProcessor`] offers all the functionalities of [`Qwen2Tokenizer`]. See the
    [`~VibeVoiceProcessor.__call__`] for more information.

    Args:
        tokenizer ([`Qwen2Tokenizer`], [`Qwen2TokenizerFast`]):
            An instance of [`Qwen2Tokenizer`] or [`Qwen2TokenizerFast`]. The tokenizer is a required input.
        speaker_embeddings (`dict[str, dict[str, str]]`, *optional*):
            Optional nested speaker embeddings dictionary. The first level contains speaker names.
            The second level contains paths to embedding arrays for each speaker.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate of audio in Hz.
    """

    attributes = ["tokenizer"]
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self,
        tokenizer=None,
        speaker_embeddings: dict | None = None,
        sampling_rate: int = 24000,
        **kwargs,
    ):
        """
        Initialize the VibeVoice processor.

        Args:
            tokenizer ([`Qwen2Tokenizer`], [`Qwen2TokenizerFast`]):
                An instance of [`Qwen2Tokenizer`] or [`Qwen2TokenizerFast`]. The tokenizer is a required input.
            speaker_embeddings (`dict[str, dict[str, str]]`, *optional*):
                Optional nested speaker embeddings dictionary. The first level contains speaker names.
                The second level contains paths to embedding arrays for each speaker.
            sampling_rate (`int`, *optional*, defaults to 24000):
                The sampling rate of audio in Hz.
        """
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        self.speaker_embeddings = speaker_embeddings
        self.sampling_rate = sampling_rate

        super().__init__(tokenizer)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_processor_name_or_path: str | os.PathLike,
        speaker_embeddings_dict_path: str = "speaker_embeddings.json",
        **kwargs,
    ):
        r"""
        Instantiate a VibeVoice processor from a pretrained model.

        Args:
            pretrained_processor_name_or_path (`str` or `os.PathLike`):
                This can be either:
                - a string, the *model id* of a pretrained processor hosted on huggingface.co
                - a path to a *directory* containing processor files
            speaker_embeddings_dict_path (`str`, *optional*, defaults to `"speaker_embeddings.json"`):
                The name of the `.json` file containing speaker embeddings paths.
            **kwargs:
                Additional keyword arguments passed to the tokenizer.
        """
        token = kwargs.get("token")
        speaker_embeddings = None
        subfolder = kwargs.pop("subfolder", None)
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        if speaker_embeddings_dict_path is not None:
            speaker_embeddings_path = cached_file(
                pretrained_processor_name_or_path,
                speaker_embeddings_dict_path,
                subfolder=subfolder,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )

            if speaker_embeddings_path is not None:
                with open(speaker_embeddings_path) as f:
                    speaker_embeddings = json.load(f)
                if "repo_or_path" in speaker_embeddings:
                    speaker_embeddings["repo_or_path"] = pretrained_processor_name_or_path

        # Try to load tokenizer from the pretrained path, if not available, fallback to
        # language_model_pretrained_name from preprocessor_config.json
        tokenizer_path = pretrained_processor_name_or_path
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **kwargs)
        except (OSError, ValueError):
            # Tokenizer not found in the pretrained path, try to get from preprocessor config
            preprocessor_config_path = cached_file(
                pretrained_processor_name_or_path,
                "preprocessor_config.json",
                subfolder=subfolder,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
            if preprocessor_config_path is not None:
                with open(preprocessor_config_path) as f:
                    preprocessor_config = json.load(f)
                tokenizer_path = preprocessor_config.get("language_model_pretrained_name")
                if tokenizer_path is not None:
                    logger.info(f"Loading tokenizer from language model: {tokenizer_path}")
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **kwargs)
                else:
                    raise ValueError(
                        f"Could not find tokenizer in '{pretrained_processor_name_or_path}' and "
                        "no 'language_model_pretrained_name' specified in preprocessor_config.json"
                    )
            else:
                raise ValueError(
                    f"Could not find tokenizer in '{pretrained_processor_name_or_path}' and "
                    "no preprocessor_config.json found"
                )

        return cls(tokenizer=tokenizer, speaker_embeddings=speaker_embeddings)

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        speaker_embeddings_dict_path: str = "speaker_embeddings.json",
        speaker_embeddings_directory: str = "speaker_embeddings",
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save the processor attributes to the specified directory.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the processor files will be saved.
            speaker_embeddings_dict_path (`str`, *optional*, defaults to `"speaker_embeddings.json"`):
                Name of the speaker embeddings dictionary file.
            speaker_embeddings_directory (`str`, *optional*, defaults to `"speaker_embeddings"`):
                Name of the folder for speaker embedding arrays.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether to push to the Hugging Face Hub after saving.
            **kwargs:
                Additional arguments passed to the parent class.
        """
        if self.speaker_embeddings is not None:
            os.makedirs(os.path.join(save_directory, speaker_embeddings_directory), exist_ok=True)

            embeddings_dict = {"repo_or_path": save_directory}

            for speaker_name in self.available_speakers:
                speaker_data = self._load_speaker_embedding(speaker_name)

                speaker_dict = {}
                for key, value in speaker_data.items():
                    save_path = os.path.join(speaker_embeddings_directory, f"{speaker_name}_{key}.npy")
                    np.save(os.path.join(save_directory, save_path), value, allow_pickle=False)
                    speaker_dict[key] = save_path

                embeddings_dict[speaker_name] = speaker_dict

            with open(os.path.join(save_directory, speaker_embeddings_dict_path), "w") as f:
                json.dump(embeddings_dict, f)

        super().save_pretrained(save_directory, push_to_hub=push_to_hub, **kwargs)

    def _load_speaker_embedding(self, speaker_name: str, **kwargs) -> dict:
        """Load speaker embedding arrays from disk."""
        if self.speaker_embeddings is None or speaker_name not in self.speaker_embeddings:
            raise ValueError(f"Speaker '{speaker_name}' not found in available speakers.")

        speaker_paths = self.speaker_embeddings[speaker_name]
        speaker_data = {}
        token = kwargs.get("token")

        for key, rel_path in speaker_paths.items():
            path = cached_file(
                self.speaker_embeddings.get("repo_or_path", "/"),
                rel_path,
                subfolder=kwargs.pop("subfolder", None),
                cache_dir=kwargs.pop("cache_dir", None),
                force_download=kwargs.pop("force_download", False),
                proxies=kwargs.pop("proxies", None),
                local_files_only=kwargs.pop("local_files_only", False),
                token=token,
                revision=kwargs.pop("revision", None),
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )

            if path is None:
                raise ValueError(f"Could not load speaker embedding: {rel_path}")

            speaker_data[key] = np.load(path)

        return speaker_data

    @property
    def available_speakers(self) -> list[str]:
        """Returns a list of available speaker names."""
        if self.speaker_embeddings is None:
            return []

        speakers = list(self.speaker_embeddings.keys())
        if "repo_or_path" in speakers:
            speakers.remove("repo_or_path")
        return speakers

    @auto_docstring
    def __call__(
        self,
        text: str | list[str] | None = None,
        speaker: str | dict | np.ndarray | None = None,
        audio: np.ndarray | None = None,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy | None = None,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = TensorType.PYTORCH,
        return_attention_mask: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        r"""
        Main method to prepare inputs for the VibeVoice model.

        Args:
            text (`str` or `List[str]`, *optional*):
                The text or batch of texts to be encoded.
            speaker (`str`, `dict`, or `np.ndarray`, *optional*):
                Speaker information. Can be:
                - A string: name of a preloaded speaker from `available_speakers`
                - A dict: dictionary containing speaker embedding arrays
                - An np.ndarray: direct speaker embedding
            audio (`np.ndarray`, *optional*):
                Reference audio waveform for voice cloning. Shape should be (num_samples,) or
                (num_channels, num_samples) at the processor's sampling rate.
            padding (`bool`, `str`, or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Padding strategy for the tokenizer.
            truncation (`bool`, `str`, or [`~utils.TruncationStrategy`], *optional*):
                Truncation strategy for the tokenizer.
            max_length (`int`, *optional*):
                Maximum length for padding/truncation.
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `"pt"`):
                Type of tensors to return.
            return_attention_mask (`bool`, *optional*, defaults to `True`):
                Whether to return attention masks.

        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] containing:
                - `input_ids`: Token IDs for the text
                - `attention_mask`: Attention mask (if requested)
                - `speaker_embedding`: Speaker embedding (if provided)
                - `audio_values`: Processed audio (if provided)
        """
        if text is None and audio is None:
            raise ValueError("You must provide at least `text` or `audio`.")

        # Process text
        encoded_inputs = {}
        if text is not None:
            text_encoding = self.tokenizer(
                text,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                **kwargs,
            )
            encoded_inputs.update(text_encoding)

        # Process speaker embedding
        if speaker is not None:
            if isinstance(speaker, str):
                # Load from preloaded speakers
                speaker_data = self._load_speaker_embedding(speaker)
                speaker_embedding = BatchFeature(data=speaker_data, tensor_type=return_tensors)
            elif isinstance(speaker, dict):
                # Direct dictionary of arrays
                speaker_embedding = BatchFeature(data=speaker, tensor_type=return_tensors)
            elif isinstance(speaker, np.ndarray):
                # Direct array
                speaker_embedding = BatchFeature(data={"embedding": speaker}, tensor_type=return_tensors)
            else:
                raise ValueError(f"speaker must be a string, dict, or np.ndarray, got {type(speaker)}")
            encoded_inputs["speaker_embedding"] = speaker_embedding

        # Process audio
        if audio is not None:
            if isinstance(audio, np.ndarray):
                # Ensure audio is float32 and normalized
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)

                # Normalize if needed (assuming 16-bit int range)
                if audio.max() > 1.0 or audio.min() < -1.0:
                    audio = audio / 32768.0

                # Ensure proper shape (batch, channels, time) or (channels, time)
                if audio.ndim == 1:
                    audio = audio[np.newaxis, :]  # Add channel dimension

                encoded_inputs["audio_values"] = BatchFeature(data={"audio": audio}, tensor_type=return_tensors)
            else:
                raise ValueError(f"audio must be a np.ndarray, got {type(audio)}")

        return BatchEncoding(encoded_inputs)

    def batch_decode(self, *args, **kwargs):
        """Forward to tokenizer's batch_decode method."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Forward to tokenizer's decode method."""
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self) -> list[str]:
        """Returns the list of model input names."""
        tokenizer_input_names = self.tokenizer.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + ["speaker_embedding", "audio_values"]))


__all__ = ["VibeVoiceProcessor"]
