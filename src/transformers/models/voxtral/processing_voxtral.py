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

import io
import warnings

import numpy as np

from ...utils import (
    auto_docstring,
    is_mistral_common_available,
    is_soundfile_available,
    is_torch_available,
    logging,
    requires_backends,
)
from ...utils.import_utils import requires


if is_torch_available():
    import torch

if is_soundfile_available():
    import soundfile as sf

if is_mistral_common_available():
    from mistral_common.protocol.transcription.request import TranscriptionRequest

from ...audio_utils import AudioInput, load_audio_as, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import AudioKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_mistral_common import MistralCommonBackend
from ...tokenization_utils_base import EncodedInput, PreTokenizedInput, TextInput


logger = logging.get_logger(__name__)

# Fallbacks used when the tokenizer does not expose a `mistral-common` audio encoder to derive them from.
DEFAULT_AUDIO_TOKEN_ID = 24
DEFAULT_RAW_AUDIO_LENGTH_PER_TOK = 1280


class VoxtralAudioKwargs(AudioKwargs, total=False):
    """
    max_source_positions (`int`, *optional*, defaults to `3000`):
        Maximum number of positions per chunk when splitting mel spectrogram features along the time dimension.
    """

    max_source_positions: int | None


class VoxtralProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: VoxtralAudioKwargs
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "add_special_tokens": False,
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
        },
    }


@requires(backends=("torch", "mistral-common"))
@auto_docstring
class VoxtralProcessor(ProcessorMixin):
    valid_processor_kwargs = VoxtralProcessorKwargs

    def __init__(
        self,
        feature_extractor,
        tokenizer,
    ):
        if not isinstance(tokenizer, MistralCommonBackend):
            raise ValueError("`tokenizer` must be a `MistralCommonBackend` tokenizer.")

        audio_encoder = self._get_audio_encoder(tokenizer)
        special_ids = getattr(audio_encoder, "special_ids", None)
        audio_token_id = getattr(special_ids, "audio", None)
        self.audio_token_id = DEFAULT_AUDIO_TOKEN_ID if audio_token_id is None else audio_token_id
        self.audio_token = tokenizer.convert_ids_to_tokens(self.audio_token_id)

        super().__init__(feature_extractor, tokenizer)

    @staticmethod
    def _get_audio_encoder(tokenizer):
        """`mistral-common`'s audio encoder, when the tokenizer exposes one."""
        mistral_tokenizer = getattr(tokenizer, "tokenizer", None)
        instruct_tokenizer = getattr(mistral_tokenizer, "instruct_tokenizer", None)
        return getattr(instruct_tokenizer, "audio_encoder", None)

    @staticmethod
    def _resolve_tokenize_and_return_dict(tokenize, return_dict):
        """Voxtral has always behaved as if both were `True`, unlike `ProcessorMixin` which defaults to `False`."""
        if tokenize is None or return_dict is None:
            warnings.warn(
                "`VoxtralProcessor` currently defaults to `tokenize=True, return_dict=True`, which differs from the "
                "`ProcessorMixin` defaults. In a future version these defaults will change to `tokenize=False, "
                "return_dict=False`. Pass `tokenize=True, return_dict=True` explicitly to keep the current behavior "
                "and silence this warning.",
                FutureWarning,
                stacklevel=3,
            )
        return True if tokenize is None else tokenize, True if return_dict is None else return_dict

    @property
    def mistral_common_audio_config(self):
        """`mistral-common`'s audio config, when the tokenizer exposes one."""
        return getattr(self._get_audio_encoder(self.tokenizer), "audio_config", None)

    @property
    def raw_audio_length_per_tok(self) -> int:
        """Number of raw audio samples represented by a single `audio_token`."""
        length_per_tok = getattr(self.mistral_common_audio_config, "raw_audio_length_per_tok", None)
        return DEFAULT_RAW_AUDIO_LENGTH_PER_TOK if length_per_tok is None else length_per_tok

    @property
    def unused_input_names(self) -> list[str]:
        "Input names returned always by subprocessors but not used in model's `forward`"
        return ["num_audio_tokens"]

    def _retrieve_input_features(self, audio, max_source_positions, **kwargs):
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

    def _get_audio_token_length(self, audio_lengths: "torch.Tensor", pad_to_multiple_of: int) -> "torch.Tensor":
        """
        Number of `audio_token` placeholders for each audio, once padded to a whole number of 30s chunks.
        Both quantities are derived from `mistral-common` so they cannot drift from the tokenizer.
        """
        num_chunks = (audio_lengths - 1) // pad_to_multiple_of + 1
        return num_chunks * (pad_to_multiple_of // self.raw_audio_length_per_tok)

    def _process_audio(self, audio: AudioInput, **kwargs):
        max_source_positions = kwargs.pop("max_source_positions")
        pad_to_multiple_of = kwargs["pad_to_multiple_of"]

        audio_inputs = {
            "input_features": self._retrieve_input_features(audio, max_source_positions, **kwargs),
            "num_audio_tokens": self._get_audio_token_length(
                torch.tensor([np.asarray(audio_array).shape[-1] for audio_array in audio]), pad_to_multiple_of
            ),
        }
        audio_replacements = [self.replace_audio_token(audio_inputs, audio_idx=idx) for idx in range(len(audio))]

        return audio_inputs, audio_replacements

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int) -> str:
        return self.audio_token * int(audio_inputs["num_audio_tokens"][audio_idx])

    def get_text_with_replacements(
        self,
        text,
        images_replacements: list[str] = [],
        videos_replacements: list[str] = [],
        audio_replacements: list[str] = [],
    ):
        """
        Same as [`ProcessorMixin.get_text_with_replacements`], but returns **token ids** instead of strings.

        `MistralCommonBackend` never encodes special tokens from a string: `encode("[AUDIO]")` returns the
        tokenization of the literal characters, not `audio_token_id`. The placeholder-expanded text therefore
        cannot be handed to the tokenizer as text. Since `MistralCommonBackend.__call__` accepts `EncodedInput`,
        we encode here and let [`ProcessorMixin.__call__`] tokenize the ids, which still gives us padding,
        `attention_mask` and `return_tensors` for free.

        Note that the returned replacement offsets refer to the expanded *text*, not to the returned ids.
        """

        def _is_encoded_input(text):
            """Whether `text` is already token ids, e.g. rendered by `apply_chat_template` through `mistral-common`."""
            return not isinstance(text[0], str)

        if _is_encoded_input(text):
            return text, []

        text, replacement_offsets = super().get_text_with_replacements(
            text, images_replacements, videos_replacements, audio_replacements
        )
        return [self._encode_with_audio_tokens(sample) for sample in text], replacement_offsets

    def _encode_with_audio_tokens(self, text: str) -> list[int]:
        """Encode `text`, mapping every `audio_token` occurrence to `audio_token_id`."""
        segments = text.split(self.audio_token)
        input_ids = self.tokenizer.encode(segments[0])
        for segment in segments[1:]:
            input_ids.append(self.audio_token_id)
            input_ids += self.tokenizer.encode(segment, add_special_tokens=False)

        return input_ids

    def _check_special_mm_tokens(self, text, text_inputs: BatchFeature, modalities: list[str]):
        """`text` holds token ids here (see `get_text_with_replacements`), so count ids on both sides."""
        expected = [list(ids).count(self.audio_token_id) for ids in text]
        got = [list(ids).count(self.audio_token_id) for ids in text_inputs["input_ids"]]
        if expected != got:
            raise ValueError(
                f"Mismatch in `audio` token count between text and `input_ids`. Got ids={got} and text={expected}. "
                "Likely due to `truncation='max_length'`. Please disable truncation or increase `max_length`."
            )

    def validate_inputs(self, images=None, text=None, videos=None, audio=None, **kwargs):
        super().validate_inputs(images=images, text=text, videos=videos, audio=audio, **kwargs)

        if text is None:
            raise ValueError(f"You need to specify `text` input to {self.__class__.__name__}.")

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        chat_template: str | None = None,
        tools: list[dict] | None = None,
        documents: list[dict[str, str]] | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        return_assistant_tokens_mask: bool = False,
        tokenize: bool | None = None,
        return_tensors: str | None = None,
        return_dict: bool | None = None,
        load_audio_from_video: bool = False,
        processor_kwargs: dict | None = None,
        **kwargs,
    ) -> str:
        """
        This method applies the model's chat completion template given a conversation. It relies on MistralCommonBackend's
        [`~MistralCommonBackend.apply_chat_template`] to prepare input ids to the model and on WhisperFeatureExtractor's
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
        inputs = processor.apply_chat_template(conversation, tokenize=True, return_dict=True)
        ```

        Args:
            conversation (`Union[list[Dict, [str, str]], list[list[dict[str, str]]]]`):
                The conversation to format.
        """

        tokenize, return_dict = self._resolve_tokenize_and_return_dict(tokenize, return_dict)

        if chat_template is not None:
            raise ValueError(
                f"{self.__class__.__name__} renders conversations with `mistral-common`, not with a Jinja template, "
                "so `chat_template` is not supported."
            )
        if documents is not None:
            raise ValueError(f"`documents` is not supported by {self.__class__.__name__}.")
        if return_assistant_tokens_mask:
            raise ValueError(
                "`return_assistant_tokens_mask` is not supported by `MistralCommonBackend`, which cannot return "
                "the offset mapping needed to infer token boundaries."
            )

        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
        ):
            is_batched = True
            conversations = conversation
        else:
            is_batched = False
            conversations = [conversation]

        # Users might still be passing processing kwargs in `**kwargs`. There is no Jinja template to introspect
        # here, so anything left in `**kwargs` is meant for `__call__`.
        processor_kwargs = processor_kwargs or {}
        if kwargs:
            logger.warning(
                "Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`"
            )
            processor_kwargs = {**processor_kwargs, **kwargs}

        if return_tensors is not None:
            processor_kwargs["return_tensors"] = return_tensors

        # `mistral-common` cannot batch audio into tensors, so always render without them and let `__call__` do it.
        rendered = self.tokenizer.apply_chat_template(
            conversations,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=tokenize,
            return_dict=True,
            return_tensors=None,
        )

        if tokenize:
            audio = rendered.pop("audio", None)
            if return_dict:
                return self(text=rendered["input_ids"], audio=audio, **processor_kwargs)
            else:
                return rendered["input_ids"]

        return rendered if is_batched else rendered[0]

    @auto_docstring(
        custom_intro=r"""
    Method to prepare text and audio to be fed as input to the model.

    `text` is either a string containing one `audio_token` per audio (which is expanded to the right number of
    placeholders here), or the token ids rendered by
    [`apply_chat_template`] [`~VoxtralProcessor.apply_chat_template`].
    """
    )
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | EncodedInput | None = None,
        audio: AudioInput | None = None,
        **kwargs: Unpack[VoxtralProcessorKwargs],
    ) -> BatchFeature:
        # Check only if passed explicitly as another value since by default we'll use `pt`
        if "return_tensors" in kwargs and kwargs["return_tensors"] != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        return super().__call__(text=text, audio=audio, **kwargs)

    # TODO: @eustlb, this should be moved to mistral_common + testing
    @requires(backends=("mistral-common",))
    def apply_transcription_request(
        self,
        audio: str | list[str] | AudioInput,
        model_id: str,
        language: str | list[str | None] | None = None,
        sampling_rate: int | None = None,
        format: str | list[str] | None = None,
        tokenize: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Unpack[VoxtralProcessorKwargs],
    ):
        """
        This method applies the model's transcription request template given a language and audio.
        It relies on MistralCommonBackend and WhisperFeatureExtractor to prepare input ids and input features to the model.

        ```python
        from transformers import VoxtralProcessor

        model_id = "mistralai/Voxtral-Mini-3B-2507"
        processor = VoxtralProcessor.from_pretrained(model_id)

        language = "en"
        audio = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"

        # set the language is already know for better accuracy
        inputs = processor.apply_transcription_request(
            language=language, audio=audio, model_id=model_id, tokenize=True, return_dict=True
        )

        # but you can also let the model detect the language automatically
        inputs = processor.apply_transcription_request(audio=audio, model_id=model_id, tokenize=True, return_dict=True)
        ```

        Args:
            audio (`str`, `list[str]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The audio or batch of audio to be prepared. If provided as a string, it should correspond to the path or url of the audio file.
            model_id (`str`:
                The hub model id of the model to use for transcription.
            language (`str`, `list[Union[str, None]]`, *optional*):
                The language or languages of the audio.
                If not provided or None, automatic language detection will be used for all audio.
                If provided as a string (a language code in the [ISO 639-1 alpha-2 format](https://en.wikipedia.org/wiki/ISO_639-1) e.g. `"en"`), it will be applied uniformly to all audio.
                If provided as a list of strings/ None values, e.g. `["en", None, "fr"]`, will be applied to each audio individually with a one-to-one mapping,
                with a None value indicating automatic language detection for that audio.
            sampling_rate (`int`, *optional*):
                The sampling rate of the audio. Necessary if it is provided as `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`.
                Used to avoid silent errors when passing audio that is not in the expected sampling rate.
            format (`str`, `list[str]`, *optional*):
                The format of the audio, necessary if is provided as `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`.
        """

        tokenize, return_dict = self._resolve_tokenize_and_return_dict(tokenize, return_dict)

        output_kwargs = self._merge_kwargs(
            VoxtralProcessorKwargs,
            **kwargs,
        )
        audio_kwargs = output_kwargs["audio_kwargs"]

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

        # validate audio input
        if is_str:
            audio = [load_audio_as(audio, return_format="buffer", force_mono=True, sampling_rate=sampling_rate)]
        elif is_list_of_str:
            audio = [
                load_audio_as(el, return_format="buffer", force_mono=True, sampling_rate=sampling_rate) for el in audio
            ]
        else:
            requires_backends(self, ["soundfile"])
            audio = make_list_of_audio(audio)
            if format is None:
                raise ValueError("`format` must be provided when passing audio arrays to VoxtralProcessor.")

            if isinstance(format, str):
                format = [format] * len(audio)

            if len(audio) != len(format):
                raise ValueError(
                    f"When passed as a list of audio, the length ({len(audio)}) must match the number of format ({len(format)})"
                )

            if not is_soundfile_available():
                raise ImportError("Please install `soundfile` to encode audio arrays with VoxtralProcessor.")

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
        elif language is None:
            language = [None] * n_audio
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
                # `text` is already tokenized, `__call__` takes care of padding and feature extraction
                return self(text=input_ids, audio=audio_arrays, **kwargs)
            else:
                return input_ids

        return texts


__all__ = ["VoxtralProcessor"]
