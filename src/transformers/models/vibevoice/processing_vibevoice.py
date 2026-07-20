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

import os

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_soundfile_available, is_torch_available, logging
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

if is_soundfile_available():
    import soundfile as sf


class VibeVoiceProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


@requires(backends=("torch",))
class VibeVoiceProcessor(ProcessorMixin):
    valid_processor_kwargs = VibeVoiceProcessorKwargs

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
        audio_bos_token="<|vision_start|>",
        audio_eos_token="<|vision_end|>",
        audio_token="<|vision_pad|>",
    ):
        r"""
        audio_bos_token (`str`, *optional*, defaults to `"<|vision_start|>"`):
            The token used to indicate the beginning of audio generation.
        audio_eos_token (`str`, *optional*, defaults to `"<|vision_end|>"`):
            The token used to indicate the end of audio generation.
        audio_token (`str`, *optional*, defaults to `"<|vision_pad|>"`):
            The token used to indicate to continue generating audio.
        """
        self.audio_bos_token = audio_bos_token
        self.audio_bos_token_id = (
            tokenizer.audio_bos_token_id
            if getattr(tokenizer, "audio_bos_token_id", None)
            else tokenizer.convert_tokens_to_ids(audio_bos_token)
        )
        self.audio_eos_token = audio_eos_token
        self.audio_eos_token_id = (
            tokenizer.audio_eos_token_id
            if getattr(tokenizer, "audio_eos_token_id", None)
            else tokenizer.convert_tokens_to_ids(audio_eos_token)
        )
        self.audio_token = audio_token
        self.audio_token_id = (
            tokenizer.audio_token_id
            if getattr(tokenizer, "audio_token_id", None)
            else tokenizer.convert_tokens_to_ids(audio_token)
        )
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    def _process_audio(self, audio: AudioInput, **kwargs):
        processed_audio = self.feature_extractor(audio, **kwargs)
        pad_to_multiple_of = kwargs.get("pad_to_multiple_of") or self.feature_extractor.pad_to_multiple_of
        self._num_audio_tokens = (
            torch.ceil(processed_audio["padding_mask"].sum(dim=-1) / pad_to_multiple_of).int().tolist()
        )
        audio_replacements = [self.replace_audio_token(processed_audio, idx) for idx in range(len(audio))]
        return processed_audio, audio_replacements

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int) -> str:
        return self.audio_token * self._num_audio_tokens[audio_idx]

    @auto_docstring
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        audio: AudioInput | None = None,
        output_labels: bool | None = False,
        **kwargs: Unpack[VibeVoiceProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to process text inputs with optional voice samples.

        This method processes text inputs (typically prepared by apply_chat_template) and optional voice samples for
        voice cloning. It expands audio diffusion tokens based on the actual audio length.

        Args:
            text (`str`, `List[str]`):
                The input text(s) to process, typically prepared by apply_chat_template with audio token placeholders.
            audio (`List[Union[str, np.ndarray]]`, *optional*):
                Audio samples for speaker voice cloning. Should match the number of audio token placeholders in text.
            output_labels (bool, *optional*, default=False):
                Whether to return labels for training.
            **kwargs:
                Additional keyword arguments passed to the tokenizer and feature extractor.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:
            - **input_ids** -- List of token ids to be fed to the model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True`).
            - **input_values** -- List of audio values to be fed to the model. Returned when `audio` is not `None`.
            - **padding_mask** -- List of indices specifying which audio frames should be attended to by the model.
              Returned when `audio` is not `None`.
            - **labels** -- Labels for language model training. Only padding tokens are masked with -100. Audio
              tokens (bos, diffusion, eos) are kept as targets so the LM learns when and how much audio to generate.
              Returned when `output_labels=True`.
            - **acoustic_loss_mask** -- Boolean mask for positions where diffusion loss is computed. True at audio
              diffusion token positions. Returned when `output_labels=True`.
        """
        if "return_tensors" in kwargs and kwargs["return_tensors"] != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        data = super().__call__(text=text, audio=audio, **kwargs)
        if output_labels:
            labels = data["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            data["labels"] = labels
            # For diffusion loss
            acoustic_loss_mask = torch.zeros_like(data["input_ids"], dtype=torch.bool)
            if audio is not None:
                acoustic_loss_mask[data["input_ids"] == self.audio_token_id] = True
            data["acoustic_loss_mask"] = acoustic_loss_mask

        return data

    def save_audio(
        self,
        audio: AudioInput,
        output_path: str | None = None,
    ) -> list[str]:
        """
        Save audio data to WAV file(s).
        TODO eventually move to AudioProcessor base class.

        Args:
            audio: Audio output from the model to be saved
            output_path: Output file path or directory for multiple files

        Returns:
            List[str]: Paths to the saved audio files.
        """

        if not is_soundfile_available():
            raise ImportError("Please install `soundfile` to save audio files.")

        audio = make_list_of_audio(audio)
        for idx, item in enumerate(audio):
            audio[idx] = item.detach().cpu().float().numpy().squeeze()

        if len(audio) == 1:
            if output_path is None:
                output_path = "vibevoice_output.wav"
            sf.write(output_path, audio[0], self.feature_extractor.sampling_rate)
            return [output_path]
        else:
            if output_path is None:
                output_path = "vibevoice_outputs"
            os.makedirs(output_path, exist_ok=True)
            saved_paths = []
            for i, audio_array in enumerate(audio):
                file_path = os.path.join(output_path, f"audio_{i}.wav")
                sf.write(file_path, audio_array, self.feature_extractor.sampling_rate)
                saved_paths.append(file_path)
        return saved_paths


__all__ = ["VibeVoiceProcessor"]
