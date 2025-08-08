# Copyright 2023 The HuggingFace Team. All rights reserved.
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
# limitations under the License.from typing import List, Union
import inspect
from typing import Any, Optional, Union, overload

import numpy as np

from ..generation import GenerationConfig
from ..utils import is_soundfile_available, is_torch_available, logging
from .base import Pipeline


if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING
    from ..models.speecht5.modeling_speecht5 import SpeechT5HifiGan

if is_soundfile_available():
    import soundfile as sf


DEFAULT_VOCODER_ID = "microsoft/speecht5_hifigan"

logger = logging.get_logger(__name__)


class TextToAudioPipeline(Pipeline):
    """
    Text-to-audio generation pipeline using any `AutoModelForTextToWaveform` or `AutoModelForTextToSpectrogram`. This
    pipeline generates an audio waveform from an input text and optionally other conditional inputs.

    <Tip>

    Different audio generation models may support different conditionining inputs, like `voice` in text-to-speech
    models. Be sure to check the model card for model-specific information, including custom conditioning inputs,
    and the documentation in [`~TextToAudioPipeline.__call__`] for common audio conditioning inputs.

    </Tip>

    Unless the model you're using explicitly sets these generation parameters in its configuration files
    (`generation_config.json`), the following default values will be used:
    - max_new_tokens: 256

    Examples:

    ```python
    >>> from transformers import pipeline

    >>> pipe = pipeline(model="suno/bark-small")
    >>> output = pipe("Hey it's HuggingFace on the phone!")

    >>> audio = output["audio"]
    >>> sampling_rate = output["sampling_rate"]

    >>> pipe.save_audio(output, "audio.wav")
    ```

    ```python
    >>> from transformers import pipeline

    >>> music_generator = pipeline(task="text-to-audio", model="facebook/musicgen-small", framework="pt")

    >>> # diversify the music generation by adding randomness with a high temperature and set a maximum music length
    >>> generate_kwargs = {
    ...     "do_sample": True,
    ...     "temperature": 0.7,
    ...     "max_new_tokens": 35,
    ... }

    >>> outputs = music_generator("Techno music with high melodic riffs", **generate_kwargs)
    >>> music_generator.save_audio(outputs, "audio.wav")
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This pipeline can currently be loaded from [`pipeline`] using the following task identifiers: `"text-to-speech"` or
    `"text-to-audio"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=text-to-speech).
    """

    _pipeline_calls_generate = True
    _load_processor = None  # optional
    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = True

    # Make sure the docstring is updated when the default generation config is changed
    _default_generation_config = GenerationConfig(
        max_new_tokens=256,
    )

    def __init__(self, *args, vocoder=None, sampling_rate=None, **kwargs):
        super().__init__(*args, **kwargs)

        if self.framework == "tf":
            raise ValueError("The TextToAudioPipeline is only available in PyTorch.")

        self.vocoder = None
        if self.model.__class__ in MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING.values():
            self.vocoder = (
                SpeechT5HifiGan.from_pretrained(DEFAULT_VOCODER_ID).to(self.model.device)
                if vocoder is None
                else vocoder
            )

        self.audio_channels = self._get_audio_channels()
        self.sampling_rate = self._get_sampling_rate(sampling_rate)

        # Text to audio models are a diverse bunch: this section contains a best-effort to attempt to identify types
        # of models, to control and standardize the behavior of the pipeline (set `generate` flags, preprocess
        # differently, etc.)
        generate_parameters = inspect.signature(self.model.generate).parameters if self.model.can_generate() else []
        processor_parameters = inspect.signature(self.processor).parameters if self.processor is not None else []
        # 1 - audio output: some models can output audio directly from `generate`, but need a flag to do so. E.g. CSM
        output_audio_flags_list = [arg for arg in ["output_audio", "return_audio"] if arg in generate_parameters]
        self._output_audio_flag = output_audio_flags_list[0] if len(output_audio_flags_list) == 1 else None
        # 2 - voice selection: some TTS models use the `role` field of the chat template to select the voice, as a
        # digit. E.g. CSM
        chat_template = getattr(self.tokenizer, "chat_template", None)
        self._has_voice_digit_in_chat_role = "isdigit()" in chat_template if chat_template is not None else False
        # 3 - voice selection: some TTS models use speaker embeddings and have a `voice_preset` parameter in the
        # processor. E.g. Bark
        self._has_voice_preset_in_processor = "voice_preset" in processor_parameters
        self._accepts_voice_in_processor = self._has_voice_digit_in_chat_role or self._has_voice_preset_in_processor
        # 4 - voice selection: some TTS models accept `voice` or `speaker` in their `generate` kwargs
        voice_kwargs_in_model_list = [arg for arg in ["voice", "speaker"] if arg in generate_parameters]
        self._voice_kwarg_in_model = voice_kwargs_in_model_list[0] if len(voice_kwargs_in_model_list) == 1 else None

    def _get_sampling_rate(self, sampling_rate: Optional[int] = None) -> Optional[int]:
        """
        Get the sampling rate from the model config, generation config, processor, or vocoder. Can be overridden by
        `sampling_rate` in `__init__`.
        """
        if sampling_rate is not None:
            return sampling_rate

        if self.vocoder is not None:
            sampling_rate = self.vocoder.config.sampling_rate

        if sampling_rate is None:
            # get sampling_rate from config and generation config

            config = self.model.config
            gen_config = self.model.__dict__.get("generation_config", None)
            if gen_config is not None:
                config.update(gen_config.to_dict())

            for sampling_rate_name in ["sample_rate", "sampling_rate"]:
                sampling_rate = getattr(config, sampling_rate_name, None)
                if sampling_rate is not None:
                    break

        # last fallback to get the sampling rate based on processor
        if sampling_rate is None and self.processor is not None and hasattr(self.processor, "feature_extractor"):
            sampling_rate = self.processor.feature_extractor.sampling_rate
        return sampling_rate

    def _get_audio_channels(self) -> int:
        """
        Get the number of audio channels from the model config. If the attribute is not found, defaults to 1.
        """
        # the number of audio channels is stored in different places in the config, depending on the model
        for nested_name in [None, "audio_config", "codec_config"]:
            if nested_name is None:
                obj = self.model.config
            else:
                obj = getattr(self.model.config, nested_name, None)
            if obj is None:
                continue
            # searches among potential names for the number of audio channels
            for attr in ["audio_channels", "num_audio_channels"]:
                audio_channels = getattr(obj, attr, None)
                if audio_channels is not None:
                    break
        # WARNING: this default may cause issues in the future. We may need a more precise way of detecting the number
        # of channels.
        audio_channels = audio_channels or 1
        return audio_channels

    def preprocess(self, text, **kwargs):
        if isinstance(text, str):
            text = [text]

        if self.model.config.model_type == "bark":
            # bark Tokenizer is called with BarkProcessor which uses those kwargs
            new_kwargs = {
                "max_length": self.generation_config.semantic_config.get("max_input_semantic_length", 256),
                "add_special_tokens": False,
                "return_attention_mask": True,
                "return_token_type_ids": False,
            }
            # priority is given to kwargs
            new_kwargs.update(kwargs)
            kwargs = new_kwargs

        # Different ways to preprocess the voice (TTS models)
        if self._has_voice_digit_in_chat_role:
            voice = str(kwargs.pop("voice", "0"))
            if not voice.isdigit():
                logger.warning(
                    f"With {self.model.name_or_path}, the voice pipeline argument must be a digit. Got voice={voice}, "
                    "using voice=0 instead."
                )
                voice = "0"
            conversation = [{"role": voice, "content": [{"type": "text", "text": text[0]}]}]
            output = self.processor.apply_chat_template(
                conversation, tokenize=True, return_dict=True, return_tensors="pt", **kwargs
            )
        elif self._has_voice_preset_in_processor:
            voice = kwargs.pop("voice", None)
            output = self.processor(text, voice_preset=voice, **kwargs)
        # Default: no voice preprocessing
        else:
            processor_object = self.processor if self.processor is not None else self.tokenizer
            output = processor_object(text=text, **kwargs, return_tensors="pt")

        return output

    def _forward(self, model_inputs, **kwargs):
        if self.model.can_generate():
            output = self.model.generate(**model_inputs, **kwargs)
        else:
            output = self.model(**model_inputs, **kwargs)[0]

        if self.vocoder is not None:
            # in that case, the output is a spectrogram that needs to be converted into a waveform
            output = self.vocoder(output)

        return output

    @overload
    def __call__(self, text_inputs: str, **kwargs: Any) -> dict[str, Any]: ...

    @overload
    def __call__(self, text_inputs: list[str], **kwargs: Any) -> list[dict[str, Any]]: ...

    def __call__(self, text_inputs: Union[str, list[str]], **kwargs) -> Union[dict[str, Any], list[dict[str, Any]]]:
        """
        Generates speech/audio from the inputs. See the [`TextToAudioPipeline`] documentation for more information.

        Args:
            text_inputs (`str` or `list[str]`):
                The text(s) to generate audio from.
            voice (`str`, *optional*):
                The voice to use for the generation, if the model is a text-to-speech model that supports multiple
                voices. Please refer to the model docs in transformers for model-specific examples.
            kwargs (`dict`, *optional*):
                Parameters passed to the model generation (if the model supports generation) or forward method
                (if not). `kwargs` are always passed to the underlying model. For a complete overview of
                generate, check the [following guide](https://huggingface.co/docs/transformers/en/main_classes/text_generation).

        Return:
            A `dict` or a list of `dict`: The dictionaries have two keys:

            - **audio** (`np.ndarray` of shape `(nb_channels, audio_length)`) -- The generated audio waveform.
            - **sampling_rate** (`int`) -- The sampling rate of the generated audio waveform.
        """
        return super().__call__(text_inputs, **kwargs)

    def _sanitize_parameters(
        self,
        voice: Optional[str] = None,
        **kwargs,
    ):
        if voice is not None and not (self._voice_kwarg_in_model is not None or self._accepts_voice_in_processor):
            raise ValueError(
                f"The {self.model.name_or_path} model does not support voice selection through the `voice` "
                "parameter. Please remove the `voice` parameter."
            )

        # BC: we accepted `generate_kwargs` as a parameter. This was a more complex way of directly passing generation
        # parameters, `pipe(..., generate_kwargs={"max_new_tokens"=100})` vs `pipe(..., max_new_tokens=100)`. If the
        # model can't generate, these kwargs are forwarded to the model's `forward` method.
        # We also accepted `forward_params` and `preprocess_params`.
        # From now on, we should try to be **explicit** about the parameters the pipeline accepts, and document the
        # accepted parameters in `__call__` (the exception being `generate` kwargs).

        preprocess_params = {}
        if "preprocess_params" in kwargs:  # BC
            preprocess_params.update(kwargs.pop("preprocess_params"))

        if voice is not None and self._accepts_voice_in_processor:
            preprocess_params["voice"] = voice

        forward_params = {}
        if "forward_params" in kwargs:  # BC
            forward_params.update(kwargs.pop("forward_params"))
        if "generate_kwargs" in kwargs:  # BC, `generate_kwargs` take precedence over `forward_params`
            forward_params.update(kwargs.pop("generate_kwargs"))
        forward_params.update(kwargs)

        # generate-specific input preparation
        if self.model.can_generate():
            if getattr(self, "assistant_model", None) is not None:
                forward_params["assistant_model"] = self.assistant_model
            if getattr(self, "assistant_tokenizer", None) is not None:
                forward_params["tokenizer"] = self.tokenizer
                forward_params["assistant_tokenizer"] = self.assistant_tokenizer
            if getattr(self, "_output_audio_flag", None) is not None:
                forward_params[self._output_audio_flag] = True
                # Without `return_dict_in_generate`, these models have a non-standard output format. By returning a
                # dictionary, we can easily look for the right keys in `postprocess`
                forward_params["return_dict_in_generate"] = True
            if "generation_config" not in forward_params:
                forward_params["generation_config"] = self.generation_config

        if voice is not None and self._voice_kwarg_in_model is not None:
            forward_params[self._voice_kwarg_in_model] = voice

        forward_params = self._ensure_tensor_on_device(forward_params, device=self.device)

        postprocess_params = {}
        return preprocess_params, forward_params, postprocess_params

    def postprocess(self, audio):
        output_dict = {}

        # Extract the waveform(s) from the possible formats. This waveform may need further processing.
        if isinstance(audio, dict):
            if "waveform" in audio:  # e.g. SpeechT5
                waveform = audio["waveform"]
            elif "audio" in audio:  # e.g. CSM (may need stacking if in List[torch.FloatTensor] format)
                waveform = torch.stack(audio["audio"]) if isinstance(audio["audio"], list) else audio["audio"]
            elif "sequences" in audio:  # E.g Dia (these models will need the processor to postprocess)
                waveform = audio["sequences"]
            else:
                raise ValueError(
                    f"Unexpected keys in the audio output format: {audio.keys()}. Expected one of "
                    "`waveform` or `audio`"
                )
        elif isinstance(audio, (tuple, list)):
            waveform = audio[0]
        else:
            waveform = audio

        # If the data is a LongTensor, then it is a codebook that needs to be decoded. If the model is not doing the
        # decoding itself, then it's because the decoding happens in the processor.
        if isinstance(waveform, torch.LongTensor):
            waveform = self.processor.decode(waveform)

        # If we know there is only one audio channel, we can infer missing dimensions
        if self.audio_channels == 1:
            if len(waveform.shape) == 1:  # (audio_length) -> (bsz=1, audio_length)
                waveform = waveform.unsqueeze(0)
            if len(waveform.shape) == 2:  # (bsz, audio_length) -> (bsz, audio_channels=1, audio_length)
                waveform = waveform.unsqueeze(1)

        # The waveform MUST have shape (batch_size, audio_channels, audio_length) at this point
        if len(waveform.shape) != 3:
            raise ValueError(
                f"Unexpected waveform shape: {waveform.shape}. Expected (batch_size, audio_channels, audio_length)"
            )

        # bsz == 1 -> output a single dict
        if waveform.shape[0] == 1:
            output_dict["audio"] = waveform[0].to(device="cpu", dtype=torch.float).numpy()
            output_dict["sampling_rate"] = self.sampling_rate
            return output_dict
        # bsz > 1 -> output a list of dicts
        else:
            output_list = []
            for batch_idx in range(waveform.shape[0]):
                output_dict = {}
                output_dict["audio"] = waveform[batch_idx].to(device="cpu", dtype=torch.float).numpy()
                output_dict["sampling_rate"] = self.sampling_rate
                output_list.append(output_dict)
            return output_list

    def save_audio(self, audio: Union[dict[str, Any], list[dict[str, Any]]], path: Union[str, list[str]]):
        """
        Saves the audio returned by the pipeline to files.

        Args:
            audio (`dict[str, Any]` or `list[dict[str, Any]]`):
                The audio returned by the pipeline. The dictionary (or each dictionary, if it is a list) should
                contain two keys:
                - `"audio"`: The audio waveform.
                - `"sampling_rate"`: The sampling rate of the audio waveform.
            path (`str` or `list[str]`):
                The path(s) to save the audio to. If multiple audio files are provided, but only one path is provided,
                the audio files will be saved in the same directory, with the same name but with a number suffix (e.g.
                `"audio_0.wav", "audio_1.wav", ...`).
        """
        if isinstance(audio, dict):
            audio = [audio]
        if isinstance(path, str):
            path = [path]

        # 1 path for multiple audio files -> add a number suffix to the path
        if len(path) == 1 and len(audio) > 1:
            base_path, extension = path[0].rsplit(".", 1)
            path = [f"{base_path}_{i}.{extension}" for i in range(len(audio))]

        # by this point, each audio file should have a path
        if len(audio) != len(path):
            raise ValueError(
                f"The number of audio files ({len(audio)}) does not match the number of paths ({len(path)})."
            )

        # save each audio file
        for audio_dict, audio_file_path in zip(audio, path):
            if self.processor is not None and hasattr(self.processor, "save_audio"):
                self.processor.save_audio(audio_dict["audio"], audio_file_path)
            else:
                # If the processor does not have a save_audio method, does a best effort to save the audio. This may
                # fail if the audio data returned by the model is not in the format expected by soundfile's defaults.
                # Add more complexity if there is demand for it.
                # (nb_channels, audio_length) -> (audio_length, nb_channels), as expected by `soundfile`
                sf_audio = np.transpose(audio_dict["audio"])
                sf.write(audio_file_path, sf_audio, audio_dict["sampling_rate"])
