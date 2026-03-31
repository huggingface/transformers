# Copyright 2025 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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

from pathlib import Path

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...utils import auto_docstring, is_soundfile_available, is_torch_available, logging
from ...utils.hub import cached_file
from ..auto import AutoTokenizer


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

if is_soundfile_available():
    import soundfile as sf


KNOWN_VOICE_PRESETS = [
    "ar_male",
    "casual_female",
    "casual_male",
    "cheerful_female",
    "de_female",
    "de_male",
    "es_female",
    "es_male",
    "fr_female",
    "fr_male",
    "hi_female",
    "hi_male",
    "it_female",
    "it_male",
    "neutral_female",
    "neutral_male",
    "nl_female",
    "nl_male",
    "pt_female",
    "pt_male",
]


@auto_docstring
class VoxtralTtsProcessor(ProcessorMixin):
    r"""
    Constructs a Voxtral TTS processor which wraps a tokenizer and provides voice preset loading
    for the Voxtral TTS text-to-speech model.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            An instance of a tokenizer (e.g., `MistralCommonBackend` / Tekken).
        voice_presets (`dict[str, str]`, *optional*):
            Dictionary mapping voice preset names to their file paths relative to the checkpoint directory,
            plus a `"repo_or_path"` key pointing to the model repository or local directory.
    """

    VOICE_PRESET_DIR = "voice_embedding"

    def __init__(self, tokenizer, voice_presets=None):
        r"""
        voice_presets (`dict[str, str]`, *optional*):
            Dictionary mapping voice preset names to their file paths relative to the checkpoint directory,
            plus a `"repo_or_path"` key pointing to the model repository or local directory. Typically populated
            automatically by [`~VoxtralTtsProcessor.from_pretrained`].
        """
        super().__init__(tokenizer)
        self.voice_presets = voice_presets or {}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate a Voxtral TTS processor from a pretrained model checkpoint.

        Loads the tokenizer and discovers available voice presets from the `voice_embedding/` directory.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Either a model id on huggingface.co or a local directory path.
            **kwargs:
                Additional keyword arguments passed to the tokenizer's `from_pretrained` method.
        """
        token = kwargs.get("token")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        voice_presets = {"repo_or_path": pretrained_model_name_or_path}
        for name in KNOWN_VOICE_PRESETS:
            rel_path = f"{cls.VOICE_PRESET_DIR}/{name}.pt"
            resolved = cached_file(
                pretrained_model_name_or_path,
                rel_path,
                token=token,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
            if resolved is not None:
                voice_presets[name] = rel_path

        if len(voice_presets) == 1:
            logger.warning(
                f"No voice presets found in `{pretrained_model_name_or_path}/{cls.VOICE_PRESET_DIR}/`. "
                "Voice preset features will not be available."
            )

        return cls(tokenizer=tokenizer, voice_presets=voice_presets)

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        """
        Saves the processor's tokenizer to the specified directory.

        Voice presets are not re-saved since they are loaded directly from the original checkpoint.
        """
        super().save_pretrained(save_directory, push_to_hub=push_to_hub, **kwargs)

    def _load_voice_preset(self, voice_preset, **kwargs):
        """
        Load a voice preset tensor from a preset name, file path, or Hub-relative path.

        Args:
            voice_preset (`str`):
                Either a known voice preset name (e.g. `"neutral_female"`), a path to a local `.pt` file,
                or a Hub-relative path.

        Returns:
            `torch.Tensor`: Audio codes tensor of shape `(num_frames, num_codebooks)`.
        """
        token = kwargs.get("token")

        if voice_preset in self.voice_presets:
            rel_path = self.voice_presets[voice_preset]
            resolved = cached_file(
                self.voice_presets.get("repo_or_path", "."),
                rel_path,
                token=token,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
            if resolved is None:
                raise ValueError(
                    f"Voice preset '{voice_preset}' was registered but could not be loaded from "
                    f"`{self.voice_presets.get('repo_or_path')}/{rel_path}`."
                )
        elif Path(voice_preset).exists():
            resolved = voice_preset
        else:
            rel_path = f"{self.VOICE_PRESET_DIR}/{voice_preset}.pt"
            resolved = cached_file(
                self.voice_presets.get("repo_or_path", "."),
                rel_path,
                token=token,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
            if resolved is None:
                raise ValueError(
                    f"Voice preset '{voice_preset}' not found. Available presets: {self.available_voice_presets}. "
                    "You can also pass a path to a local `.pt` file."
                )

        return torch.load(resolved, map_location="cpu", weights_only=True)

    @property
    def available_voice_presets(self) -> list[str]:
        """Returns a list of available voice preset names."""
        return [k for k in self.voice_presets if k != "repo_or_path"]

    @auto_docstring
    def __call__(
        self,
        text=None,
        voice_preset=None,
        return_tensors="pt",
        add_special_tokens=True,
        **kwargs,
    ) -> BatchFeature:
        r"""
        text (`str` or `list[str]`):
            The text to synthesize. Will be tokenized using the underlying tokenizer.
        voice_preset (`str`, `torch.Tensor`, *optional*):
            Voice preset to use for generation. Can be:
            - A preset name (e.g. `"neutral_female"`)
            - A path to a local `.pt` file containing audio codes
            - A `torch.Tensor` of pre-computed audio codes
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of tensors to return.
        add_special_tokens (`bool`, *optional*, defaults to `True`):
            Whether to add special tokens when tokenizing the text.
        **kwargs:
            Additional keyword arguments passed to the tokenizer.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:
            - **input_ids** -- Token IDs for the input text.
            - **attention_mask** -- Attention mask for the input text.
            - **audio_codes** -- *(optional)* Voice reference audio codes of shape `(batch_size, num_frames,
              num_codebooks)`. Only present when `voice_preset` is provided.
        """
        if text is None:
            raise ValueError("You must provide `text` to the VoxtralTtsProcessor.")

        encoded = self.tokenizer(
            text,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )
        data = dict(encoded)

        if voice_preset is not None:
            if isinstance(voice_preset, str):
                audio_codes = self._load_voice_preset(voice_preset, **kwargs)
            elif is_torch_available() and isinstance(voice_preset, torch.Tensor):
                audio_codes = voice_preset
            else:
                raise ValueError(
                    f"Invalid voice_preset type: {type(voice_preset)}. "
                    "Expected a string (preset name or path) or a torch.Tensor."
                )

            if audio_codes.dim() == 2:
                audio_codes = audio_codes.unsqueeze(0)

            batch_size = data["input_ids"].shape[0]
            if audio_codes.shape[0] == 1 and batch_size > 1:
                audio_codes = audio_codes.expand(batch_size, -1, -1)

            data["audio_codes"] = audio_codes

        return BatchFeature(data=data, tensor_type=return_tensors)

    def save_audio(self, audio, saving_path, sampling_rate=24000):
        """
        Save an audio waveform tensor to a file.

        Args:
            audio (`torch.Tensor` or `np.ndarray`):
                Audio waveform to save. If a tensor, it will be converted to numpy.
            saving_path (`str` or `Path`):
                Output file path (e.g. `"output.wav"`).
            sampling_rate (`int`, *optional*, defaults to 24000):
                Sample rate of the audio in Hz.
        """
        if not is_soundfile_available():
            raise ImportError("Please install `soundfile` to save audio files: `pip install soundfile`.")

        if is_torch_available() and isinstance(audio, torch.Tensor):
            audio = audio.cpu().float().numpy()

        sf.write(str(saving_path), audio, sampling_rate)

    @property
    def model_input_names(self):
        return self.tokenizer.model_input_names


__all__ = ["VoxtralTtsProcessor"]
