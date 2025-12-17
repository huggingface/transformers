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

import os
from typing import Optional, Union

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import is_soundfile_available, is_torch_available, logging


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

if is_soundfile_available():
    import soundfile as sf


class VibeVoiceRealTimeProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
            "add_special_tokens": False,
            "return_attention_mask": True,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class VibeVoiceRealTimeProcessor(ProcessorMixin):
    r"""
    Constructs a VibeVoice processor which wraps [`VibeVoiceFeatureExtractor`] and
    [`Qwen2TokenizerFast`] into a single processor that inherits both the audio feature extraction and
    tokenizer functionalities.

    See the [`~VibeVoiceRealTimeProcessor.__call__`] for more information.

    Args:
        tokenizer (`Qwen2TokenizerFast`):
            The tokenizer for text processing.
    """

    tokenizer_class = "Qwen2TokenizerFast"

    def __init__(self, tokenizer):
        super().__init__(tokenizer)

        if not hasattr(tokenizer, "pad_id"):
            # NOTE original used <image_pad>: https://github.com/microsoft/VibeVoice/blob/d295d1e1d0fff1ad42bc0450d5b593f8e59356b9/vibevoice/modular/modular_vibevoice_text_tokenizer.py#L181
            self.pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        else:
            self.pad_id = tokenizer.pad_id

    def _validate_voice_preset_dict(self, voice_preset: Optional[dict] = None):
        for key in ["lm", "tts_lm", "neg_tts_lm"]:
            if key not in voice_preset:
                raise ValueError(f"Voice preset unrecognized, missing {key} as a key.")
            for sub_key in ["last_hidden_state", "past_key_values"]:
                if sub_key not in voice_preset[key]:
                    raise ValueError(f"Voice preset unrecognized, missing {sub_key} in {key}.")
                
            if not isinstance(voice_preset[key]["last_hidden_state"], torch.Tensor):
                raise TypeError(f"voice_preset[{key}][{sub_key}] must be of type torch.Tensor.")
            
            if not isinstance(voice_preset[key]["past_key_values"], dict):
                raise TypeError(f"voice_preset[{key}][{sub_key}] must be of type dict.")
            
            for cache_key in ["key_cache", "value_cache"]:
                if cache_key not in voice_preset[key]["past_key_values"]:
                    raise ValueError(f"Voice preset unrecognized, missing {cache_key} in past_key_values of {key}.")
                if not isinstance(voice_preset[key]["past_key_values"][cache_key], list):
                    raise TypeError(f"voice_preset[{key}]['past_key_values'][{cache_key}] must be of type list.")
                for tensor in voice_preset[key]["past_key_values"][cache_key]:
                    if not isinstance(tensor, torch.Tensor):
                        raise TypeError(f"Each item in voice_preset[{key}]['past_key_values'][{cache_key}] must be of type torch.Tensor.")

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]],
        voice_preset: Union[str, dict["torch.Tensor"]] = None,
        **kwargs: Unpack[VibeVoiceRealTimeProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to process text inputs with optional voice preset argument.

        Args:
            text (`str`, `List[str]`):
                The input text(s) to tokenizer.
            voice_preset (`str`, `dict`):
                Preset(s) to set the voice for the generated audio.  It can either be a valid voice_preset name, e.g
                `"de-Spk0_man"`, or it can be a valid file name of a local `.pt` single voice preset containing the
                keys "lm", "tts_lm", and "neg_tts_lm". With each key contains the keys "last_hidden_state"
                and "past_key_values".
            **kwargs:
                Additional keyword arguments passed to the tokenizer and feature extractor.

        Returns:
            `BatchFeature`: A BatchFeature with the following fields:
                - **input_ids** -- Token ID sequences ready for the model
                - **attention_mask** -- Attention masks for the sequences
        """
        output_kwargs = self._merge_kwargs(
            VibeVoiceRealTimeProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = output_kwargs["text_kwargs"]
        return_tensors = text_kwargs.get("return_tensors", None)
        return_attention_mask = text_kwargs.get("return_attention_mask", True)
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")
        
        if voice_preset is None:
            # TODO (ebezzam) best way to default?
            logger.warning("Defaulting to `en-Frank_man` voice preset.")
            from huggingface_hub import hf_hub_download
            default_preset = "voice_presets/en-Frank_man_converted.pt"
            voice_preset = hf_hub_download(repo_id="bezzam/VibeVoice-0.5B", filename=default_preset)

        # make batch
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, (list, tuple)):
            raise ValueError("text input must be a string or list of strings")
        
        # TODO mimic their preprocessing but maybe not necessary?
        # https://github.com/microsoft/VibeVoice/blob/d295d1e1d0fff1ad42bc0450d5b593f8e59356b9/vibevoice/processor/vibevoice_streaming_processor.py#L219
        text_preprocessed = [t.strip() + "\n" for t in text]
        # NOTE (ebezzam) this will create "inputs_id" and "attention_mask" according to Transformers convention
        encoded_text = self.tokenizer(text_preprocessed, **text_kwargs)
        # TODO (ebezzam) hacks to imitate original, `tts_attention_mask` of below doesn't seem to be used by original?
        del encoded_text["attention_mask"]      # original doesn't use this, but manually creates it, and manual has different shape
        encoded_text["tts_input_ids"] = encoded_text.pop("input_ids")

        lm_input_ids = None    # input_ids in original
        tts_lm_input_ids = None # tts_lm_input_ids in original
        if voice_preset is not None:
            lm_input_ids = []
            tts_lm_input_ids = []

            # TODO how to handle batching? Bark doesn't
            # make batch
            if isinstance(voice_preset, (str, dict)):
                voice_preset = [voice_preset]
            elif not isinstance(voice_preset, (list, tuple)):
                raise ValueError("voice_preset input must be a string, dict, or list of strings/dicts")

            # load and validate each voice preset
            for i, _preset in enumerate(voice_preset):
                if isinstance(_preset, str) and _preset.endswith(".pt"):
                    voice_preset[i] = torch.load(_preset, weights_only=False)
                elif not isinstance(_preset, dict):
                    raise ValueError(f"voice_preset must be a dict containing the voice preset tensors if not a .pt file. Got {_preset}")
                self._validate_voice_preset_dict(voice_preset[i])

                lm_input_ids.append([self.pad_id] * voice_preset[i]['lm']['last_hidden_state'].size(1))
                tts_lm_input_ids.append([self.pad_id] * voice_preset[i]['tts_lm']['last_hidden_state'].size(1))

            lm_attention_masks = [[1] * len(ids) for ids in lm_input_ids] if return_attention_mask else None
            tts_lm_attention_masks = [[1] * len(ids) for ids in tts_lm_input_ids] if return_attention_mask else None

            # TODO (ebezzam) proper batching
            encoded_text.update({
                "input_ids": torch.tensor(lm_input_ids, dtype=torch.long),
                # NOTE (ebezzam) original seems to use this as the attention mask and NOT from tokenizer...
                "attention_mask": torch.tensor(lm_attention_masks, dtype=torch.long) if lm_attention_masks is not None else None,
                "tts_lm_input_ids": torch.tensor(tts_lm_input_ids, dtype=torch.long),
                "tts_lm_attention_mask": torch.tensor(tts_lm_attention_masks, dtype=torch.long) if tts_lm_attention_masks is not None else None,
            })

            # NOTE (ebezzam) like in Bark: https://github.com/huggingface/transformers/blob/66623a1fd62d54159ad757b68c0aed8dc229d917/src/transformers/models/bark/processing_bark.py#L330
            # TODO should we batch? not done in Bark
            encoded_text["history_prompt"] = voice_preset

        return encoded_text

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names))

    def save_audio(
        self,
        audio: AudioInput,
        output_path: Optional[str] = None,
        sampling_rate: Optional[int] = 24000,
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
            sf.write(output_path, audio[0], sampling_rate)
            return [output_path]
        else:
            if output_path is None:
                output_path = "vibevoice_outputs"
            os.makedirs(output_path, exist_ok=True)
            saved_paths = []
            for i, audio_array in enumerate(audio):
                file_path = os.path.join(output_path, f"audio_{i}.wav")
                sf.write(file_path, audio_array, sampling_rate)
                saved_paths.append(file_path)
        return saved_paths


__all__ = ["VibeVoiceRealTimeProcessor"]
