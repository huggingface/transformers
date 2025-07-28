# coding=utf-8
# Copyright 2025 Sesame and The HuggingFace Inc. team. All rights reserved.
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

import io
import warnings
from typing import Optional, Union

from ...utils import is_mistral_common_available, is_soundfile_available, is_torch_available, logging


if is_torch_available():
    import torch

if is_soundfile_available():
    import soundfile as sf

if is_mistral_common_available():
    from mistral_common.protocol.transcription.request import TranscriptionRequest

from ...audio_utils import AudioInput, load_audio_as, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import AllKwargsForChatTemplate, AudioKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput


logger = logging.get_logger(__name__)


class VoxtralAudioKwargs(AudioKwargs, total=False):
    max_source_positions: Optional[int]


class VoxtralProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": True,
            "truncation": False,
            "pad_to_multiple_of": 480000,
            "max_source_positions": 3000,
        },
        "common_kwargs": {
            "return_tensors": "pt",
            "return_dict": True,
            "tokenize": True,
        },
    }


class VoxtralProcessor(ProcessorMixin):
    r"""
    Constructs a Voxtral processor which wraps [`WhisperFeatureExtractor`] and
    [`MistralCommonTokenizer`] into a single processor that inherits both the audio feature extraction and
    tokenizer functionalities.

    Args:
        feature_extractor ([`WhisperFeatureExtractor`]):
            The feature extractor is a required input.
        tokenizer ([`MistralCommonTokenizer`]):
            The tokenizer is a required input.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "MistralCommonTokenizer"

    def __init__(
        self,
        feature_extractor,
        tokenizer,
    ):
        self.audio_token_id = 24
        self.audio_token = tokenizer.convert_ids_to_tokens(self.audio_token_id)

        super().__init__(feature_extractor, tokenizer)

    def _retreive_input_features(self, audio, max_source_positions, **kwargs):
        """
        Handles specific logic of Voxtral expected input features: audio arrays should be padded to next multiple of 480000 (duration is a multiple of 30s), see VoxtralProcessorKwargs' default audio_kwargs.
        Then mel input features are extracted and stacked along batch dimension, splitting into chunks of max_source_positions.
        """
        input_features_list = []
        for audio_array in audio:
            audio_inputs = self.feature_extractor(audio_array, **kwargs)

            # let's split into chunks of max_source_positions, and then stack them along batch dimension
            input_features = audio_inputs["input_features"].reshape(
                self.feature_extractor.feature_size, -1, max_source_positions
            )
            input_features_list.append(input_features.transpose(0, 1))

        return torch.cat(input_features_list)

    def apply_chat_template(
        self,
        conversation: Union[list[dict[str, str]], list[list[dict[str, str]]]],
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ) -> str:
        """
        This method applies the model's chat completion template given a conversation. It relies on MistralCommonTokenizer's
        [`~MistralCommonTokenizer.apply_chat_template`] to prepare input ids to the model and on WhisperFeatureExtractor's
        [`~WhisperFeatureExtractor.__call__`] to prepare input features to the model.

        Note that audio is padded to the nearest 30-second multiple prior to mel feature extraction.

        A `conversation` is a list of messages, where each message is a dictionary with a `role` and a `content` field.
        For Voxtral, `role` can be `"user"` or `"assistant"`.
        The `content` field can be a string or a list of dictionaries with a `type` field. See example below.

        ```python
        from huggingface_hub import hf_hub_download
        from transformers.audio_utils import load_audio_as

        audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3"
        audio_path = hf_hub_download(repo_id="hf-internal-testing/dummy-audio-samples", filename="bcn_weather.mp3", repo_type="dataset")
        audio_base64 = load_audio_as(audio_path, return_format="base64", force_mono=True)

        # audio + text
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "url": audio_url},
                    {"type": "audio", "path": audio_path},
                    {"type": "audio", "base64": audio_base64},
                    {"type": "text", "text": "How many audio do you hear?"},
                ],
            },
        ]

        processor = VoxtralProcessor.from_pretrained("mistralai/Voxtral-Mini-3B-2507")
        inputs = processor.apply_chat_template(conversation)
        ```

        Args:
            conversation (`Union[list[Dict, [str, str]], list[list[dict[str, str]]]]`):
                The conversation to format.
        """
        if kwargs.get("continue_final_message", False):
            if kwargs.get("add_generation_prompt", False):
                raise ValueError(
                    "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."
                )
            if kwargs.get("return_assistant_tokens_mask", False):
                raise ValueError("continue_final_message is not compatible with return_assistant_tokens_mask.")

        # Fill sets of kwargs that should be used by different parts of template
        processed_kwargs = {
            "mm_load_kwargs": {},
            "template_kwargs": {},
        }

        for kwarg_type in processed_kwargs:
            for key in AllKwargsForChatTemplate.__annotations__[kwarg_type].__annotations__.keys():
                kwarg_type_defaults = AllKwargsForChatTemplate.__annotations__[kwarg_type]
                default_value = getattr(kwarg_type_defaults, key, None)
                value = kwargs.pop(key, default_value)
                if value is not None and not isinstance(value, dict):
                    processed_kwargs[kwarg_type][key] = value

        # Pass unprocessed custom kwargs
        processed_kwargs["template_kwargs"].update(kwargs)

        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
        ):
            is_batched = True
            conversations = conversation
        else:
            is_batched = False
            conversations = [conversation]

        # Check for any overlapping keys between mm_load_kwargs and kwargs
        mm_load_kwargs = processed_kwargs["mm_load_kwargs"]
        if any(key in kwargs for key in mm_load_kwargs):
            overlapping_keys = [key for key in mm_load_kwargs if key in kwargs]
            logger.warning(
                f"{overlapping_keys[0] if len(overlapping_keys) == 1 else ', '.join(overlapping_keys)} load multimodal data kwarg{'s' if len(overlapping_keys) > 1 else ''} {'have' if len(overlapping_keys) > 1 else 'has'} been passed to the processor, but {'they are' if len(overlapping_keys) > 1 else 'it is'} not supported for VoxtralProcessor since it relies on mistral_common directly. {'They' if len(overlapping_keys) > 1 else 'It'} will be ignored."
            )

        output_kwargs = self._merge_kwargs(
            VoxtralProcessorKwargs,
            **kwargs,
        )
        text_kwargs = output_kwargs["text_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]
        common_kwargs = output_kwargs["common_kwargs"]

        return_tensors = common_kwargs.pop("return_tensors", None)
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        tokenizer_kwargs = {**processed_kwargs["template_kwargs"], **text_kwargs}
        tokenizer_kwargs["return_tensors"] = None  # let's not return tensors here
        tokenize = tokenizer_kwargs.pop("tokenize", False)
        return_dict = tokenizer_kwargs.pop("return_dict", False)

        encoded_instruct_inputs = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=tokenize,
            return_dict=return_dict,
            **tokenizer_kwargs,
        )

        if tokenize:
            if return_dict:
                audio = encoded_instruct_inputs.pop("audio", None)
                data = dict(encoded_instruct_inputs)
                if audio is not None:
                    max_source_positions = audio_kwargs.pop("max_source_positions")
                    data["input_features"] = self._retreive_input_features(audio, max_source_positions, **audio_kwargs)

                return BatchFeature(data=data, tensor_type=return_tensors)

        if not is_batched:
            return encoded_instruct_inputs[0]

        return encoded_instruct_inputs

    def __call__(
        self,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]],
        **kwargs: Unpack[VoxtralProcessorKwargs],
    ):
        r"""
        Method to prepare text to be fed as input to the model. This method forwards the `text`
        arguments to MistralCommonTokenizer's [`~MistralCommonTokenizer.__call__`] to encode
        the text. Please refer to the docstring of the above methods for more information.
        This methods does not support audio. To prepare the audio, please use:
        1. `apply_chat_template` [`~VoxtralProcessor.apply_chat_template`] method.
        2. `apply_transcription_request` [`~VoxtralProcessor.apply_transcription_request`] method.

        Args:
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                    - `'tf'`: Return TensorFlow `tf.constant` objects.
                    - `'pt'`: Return PyTorch `torch.Tensor` objects.
                    - `'np'`: Return NumPy `np.ndarray` objects.
                    - `'jax'`: Return JAX `jnp.ndarray` objects.
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **input_features** -- List of audio values to be fed to a model. Returned when `audio` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
        """
        if isinstance(text, str):
            text = [text]

        if any(self.audio_token in t for t in text):
            raise ValueError(
                f"{self.audio_token} is present in the provided text which is not supported by VoxtralProcessor. Please use the `apply_chat_template` method instead."
            )

        output_kwargs = self._merge_kwargs(
            VoxtralProcessorKwargs,
            **kwargs,
        )
        text_kwargs = output_kwargs["text_kwargs"]
        common_kwargs = output_kwargs["common_kwargs"]

        out = self.tokenizer(text, **text_kwargs)

        return BatchFeature(data=out, tensor_type=common_kwargs.pop("return_tensors", None))

    # TODO: @eustlb, this should be moved to mistral_common + testing
    def apply_transcription_request(
        self,
        language: Union[str, list[str]],
        audio: Union[str, list[str], AudioInput],
        model_id: str,
        sampling_rate: Optional[int] = None,
        format: Optional[Union[str, list[str]]] = None,
        **kwargs: Unpack[VoxtralProcessorKwargs],
    ):
        """
        This method applies the model's transcription request template given a language and audio.
        It relies on MistralCommonTokenizer and WhisperFeatureExtractor to prepare input ids and input features to the model.

        ```python
        from transformers import VoxtralProcessor

        model_id = "mistralai/Voxtral-Mini-3B-2507"
        processor = VoxtralProcessor.from_pretrained(model_id)

        language = "en"
        audio = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"

        inputs = processor.apply_transcription_request(language=language, audio=audio, model_id=model_id)
        ```

        Args:
            language (`str`, `list[str]`):
                The language or languages of the audio. If provided as a string, will be applied uniformly to all audio.
                If provided as a list, will be applied to each audio individually with a one-to-one mapping.
            audio (`str`, `list[str]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The audio or batch of audio to be prepared. If provided as a string, it should correspond to the path or url of the audio file.
            model_id (`str`:
                The hub model id of the model to use for transcription.
            sampling_rate (`int`, *optional*):
                The sampling rate of the audio. Necessary if it is provided as `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`.
                Used to avoid silent errors when passing audio that is not in the expected sampling rate.
            format (`str`, `list[str]`, *optional*):
                The format of the audio, necessary if is provided as `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`.
        """
        output_kwargs = self._merge_kwargs(
            VoxtralProcessorKwargs,
            **kwargs,
        )
        text_kwargs = output_kwargs["text_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]
        common_kwargs = output_kwargs["common_kwargs"]

        is_str = isinstance(audio, str)
        is_list_of_str = all(isinstance(el, str) for el in audio)
        is_list_of_audio = not (is_str or is_list_of_str)

        if is_list_of_audio:
            if sampling_rate is None:
                logger.warning_once(
                    f"You've provided audio without specifying the sampling rate. It will be assumed to be {audio_kwargs['sampling_rate']}, which can result in silent errors."
                )
            elif sampling_rate != audio_kwargs["sampling_rate"]:
                raise ValueError(
                    f"The sampling rate of the audio ({sampling_rate}) does not match the sampling rate of the processor ({audio_kwargs['sampling_rate']}). Please provide resampled the audio to the expected sampling rate."
                )

        sampling_rate = audio_kwargs["sampling_rate"]
        return_dict = common_kwargs.pop("return_dict", False)
        tokenize = common_kwargs.pop("tokenize", False)

        # make sure to remove from text_kwargs and audio_kwargs
        for k in ("return_dict", "tokenize"):
            text_kwargs.pop(k, None)
            audio_kwargs.pop(k, None)

        return_tensors = common_kwargs.pop("return_tensors", None)
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        # validate audio input
        if is_str:
            audio = [load_audio_as(audio, return_format="buffer", force_mono=True, sampling_rate=sampling_rate)]
        elif is_list_of_str:
            audio = [
                load_audio_as(el, return_format="buffer", force_mono=True, sampling_rate=sampling_rate) for el in audio
            ]
        else:
            audio = make_list_of_audio(audio)
            if len(audio) != len(format):
                raise ValueError(
                    f"When passed as a list of audio, the length ({len(audio)}) must match the number of format ({len(format)})"
                )
            audio_buffers = []
            for array, f in zip(audio, format):
                # Create new BytesIO object and write audio data to it
                buffer = io.BytesIO()
                # Convert to mono if needed
                if array.ndim == 2:
                    array = array.mean(axis=1)
                # Write to buffer with default format and sampling rate
                sf.write(buffer, array, samplerate=audio_kwargs["sampling_rate"], format=f)
                buffer.seek(0)
                audio_buffers.append(buffer)
            audio = audio_buffers

        # validate language input
        n_audio = len(audio)
        if isinstance(language, str):
            language = [language] * n_audio

        if len(language) != n_audio:
            raise ValueError(
                f"When passed as a list of languages, the length ({len(language)}) must match the number of audio ({n_audio})"
            )

        input_ids = []
        texts = []
        audio_arrays = []
        for audio_el, language_el in zip(audio, language):
            openai_transcription_request = {
                "model": model_id,
                "file": audio_el,
                "language": language_el,
            }

            transcription_request = TranscriptionRequest.from_openai(openai_transcription_request)
            tokenized_transcription_request = self.tokenizer.tokenizer.encode_transcription(transcription_request)

            input_ids.append(tokenized_transcription_request.tokens)
            texts.append(tokenized_transcription_request.text)
            audio_arrays.extend([el.audio_array for el in tokenized_transcription_request.audios])

        if tokenize:
            if return_dict:
                # text are already tokenized but we need to pad etc
                encoding = self.tokenizer(
                    input_ids,
                    add_special_tokens=False,
                    **text_kwargs,
                )
                data = dict(encoding)

                # extract the input features
                max_source_positions = audio_kwargs.pop("max_source_positions")
                data["input_features"] = self._retreive_input_features(
                    audio_arrays, max_source_positions, **audio_kwargs
                )

                return BatchFeature(data=data, tensor_type=return_tensors)

        return texts

    # Deprecated typo'd method for backward compatibility
    def apply_transcrition_request(self, *args, **kwargs):
        """
        Deprecated typo'd method. Use `apply_transcription_request` instead.
        """
        warnings.warn(
            "`apply_transcrition_request` is deprecated due to a typo and will be removed in a future release. Please use `apply_transcription_request` instead.",
            FutureWarning,
        )
        return self.apply_transcription_request(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to MistralCommonTokenizer's [`~MistralCommonTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to MistralCommonTokenizer's [`~MistralCommonTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)


__all__ = ["VoxtralProcessor"]
