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

from dataclasses import dataclass

from ...audio_utils import AudioInput, make_list_of_audio
from ...cache_utils import Cache
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torch_available, logging
from ...utils.import_utils import requires
from ..nemotron_asr_streaming.modeling_nemotron_asr_streaming import (
    NemotronAsrStreamingForRNNT,
    NemotronAsrStreamingPreTrainedModel,
)
from ..nemotron_asr_streaming.processing_nemotron_asr_streaming import (
    NemotronAsrStreamingProcessor,
    NemotronAsrStreamingProcessorKwargs,
)
from .configuration_nemotron3_5_asr import Nemotron3_5AsrConfig
from .generation_nemotron3_5_asr import Nemotron3_5AsrGenerationMixin, Nemotron3_5AsrRNNTDecoderCache


if is_torch_available():
    import torch
    from torch import nn


logger = logging.get_logger(__name__)


DEFAULT_PROMPT_DICTIONARY = {
    "en-US": 0,
    "en": 0,
    "en-GB": 1,
    "enGB": 1,
    "es-ES": 2,
    "esES": 2,
    "es-US": 3,
    "es": 3,
    "zh-CN": 4,
    "zh-ZH": 4,
    "zh-TW": 5,
    "hi-IN": 6,
    "hi": 6,
    "hi-HI": 6,
    "ar-AR": 7,
    "ar": 7,
    "fr-FR": 8,
    "fr": 8,
    "de-DE": 9,
    "de": 9,
    "ja-JP": 10,
    "ja-JA": 10,
    "ru-RU": 11,
    "ru": 11,
    "pt-BR": 12,
    "pt-PT": 13,
    "pt": 13,
    "ko-KR": 14,
    "ko": 14,
    "ko-KO": 14,
    "it-IT": 15,
    "it": 15,
    "nl-NL": 16,
    "nl": 16,
    "pl-PL": 17,
    "pl": 17,
    "tr-TR": 18,
    "tr": 18,
    "uk-UA": 19,
    "uk": 19,
    "ro-RO": 20,
    "ro": 20,
    "el-GR": 21,
    "el": 21,
    "cs-CZ": 22,
    "cs": 22,
    "hu-HU": 23,
    "hu": 23,
    "sv-SE": 24,
    "sv": 24,
    "da-DK": 25,
    "da": 25,
    "fi-FI": 26,
    "fi": 26,
    "no-NO": 27,
    "no": 27,
    "nb-NO": 103,
    "nb": 103,
    "nn-NO": 104,
    "nn": 104,
    "sk-SK": 28,
    "sk": 28,
    "hr-HR": 29,
    "hr": 29,
    "bg-BG": 30,
    "bg": 30,
    "lt-LT": 31,
    "lt": 31,
    "et-EE": 60,
    "et": 60,
    "lv-LV": 61,
    "lv": 61,
    "sl-SI": 62,
    "sl": 62,
    "th-TH": 32,
    "vi-VN": 33,
    "id-ID": 34,
    "ms-MY": 35,
    "bn-IN": 36,
    "ur-PK": 37,
    "fa-IR": 38,
    "ta-IN": 39,
    "te-IN": 40,
    "mr-IN": 41,
    "gu-IN": 42,
    "kn-IN": 43,
    "ml-IN": 44,
    "si-LK": 45,
    "ne-NP": 46,
    "km-KH": 47,
    "sw-KE": 48,
    "am-ET": 49,
    "ha-NG": 50,
    "zu-ZA": 51,
    "yo-NG": 52,
    "ig-NG": 53,
    "af-ZA": 54,
    "rw-RW": 55,
    "so-SO": 56,
    "ny-MW": 57,
    "ln-CD": 58,
    "or-KE": 59,
    "he-IL": 64,
    "ku-TR": 65,
    "az-AZ": 66,
    "ka-GE": 67,
    "hy-AM": 68,
    "uz-UZ": 69,
    "tg-TJ": 70,
    "ky-KG": 71,
    "qu-PE": 80,
    "ay-BO": 81,
    "gn-PY": 82,
    "nah-MX": 83,
    "mi-NZ": 96,
    "haw-US": 97,
    "sm-WS": 98,
    "to-TO": 99,
    "fr-CA": 100,
    "mt-MT": 102,
    "auto": 101,
}


class Nemotron3_5AsrProcessorKwargs(NemotronAsrStreamingProcessorKwargs, total=False):
    pass


@requires(backends=("torch",))
@auto_docstring
class Nemotron3_5AsrProcessor(NemotronAsrStreamingProcessor):
    def __init__(
        self,
        feature_extractor,
        tokenizer,
        blank_token="<blank>",
        decoder_type=None,
        supported_num_lookahead_tokens=None,
        default_num_lookahead_tokens=None,
        prompt_dictionary=None,
        num_prompts=128,
    ):
        r"""
        blank_token (`str`, *optional*, defaults to `"<blank>"`):
            Blank token for RNN-T decoding.
        decoder_type (`str`, *optional*):
            Decoding/timestamp emission mode (e.g. `"ctc"`, `"rnnt"`, `"tdt"`). If `None` the decoder type
            is inferred automatically for backward compatibility.
        supported_num_lookahead_tokens (`list[int]`, *optional*):
            Supported right attention contexts (lookaheads, in subsampled encoder frames), mirroring
            `NemotronAsrStreamingEncoderConfig.supported_num_lookahead_tokens`. Used to validate
            `streaming_latency_ms` and to derive the returned `num_lookahead_tokens`.
        default_num_lookahead_tokens (`int`, *optional*):
            The right context used when `streaming_latency_ms` is not provided. Defaults to the first
            entry of `supported_num_lookahead_tokens`.
        prompt_dictionary (`dict[str, int]`, *optional*):
            Mapping from a target-language string (e.g. `"en-US"`, `"de-DE"`, the bare code `"de"`, or
            `"auto"`) to its prompt index. Defaults to the NeMo checkpoint's prompt dictionary.
        num_prompts (`int`, *optional*, defaults to 128):
            Number of language-prompt slots (size of the one-hot prompt vector), mirroring
            `Nemotron3_5AsrConfig.num_prompts`.
        """
        self.prompt_dictionary = prompt_dictionary if prompt_dictionary is not None else DEFAULT_PROMPT_DICTIONARY
        self.num_prompts = num_prompts
        super().__init__(
            feature_extractor,
            tokenizer,
            blank_token=blank_token,
            decoder_type=decoder_type,
            supported_num_lookahead_tokens=supported_num_lookahead_tokens,
            default_num_lookahead_tokens=default_num_lookahead_tokens,
        )

    def _resolve_prompt_ids(self, language: "str | list[str]", batch_size: int) -> "torch.LongTensor":
        if isinstance(language, str):
            language = [language] * batch_size
        if len(language) != batch_size:
            raise ValueError(f"Received {len(language)} `language` entries for {batch_size} audio input(s).")
        prompt_ids = []
        for lang in language:
            if lang not in self.prompt_dictionary:
                raise ValueError(f"Unknown `language={lang!r}`. Supported values: {sorted(self.prompt_dictionary)}.")
            prompt_ids.append(self.prompt_dictionary[lang])
        return torch.tensor(prompt_ids, dtype=torch.long)

    @auto_docstring
    def __call__(
        self,
        audio: AudioInput,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        sampling_rate: int | None = None,
        is_streaming: bool = False,
        is_first_audio_chunk: bool | None = True,
        language: str | list[str] = "auto",
        **kwargs: Unpack[Nemotron3_5AsrProcessorKwargs],
    ):
        r"""
        sampling_rate (`int`, *optional*):
            The sampling rate of the input audio in Hz. Validated against the feature extractor's
            expected sampling rate (defaults to 16000 Hz) when provided.
        is_streaming (`bool`, *optional*, defaults to `False`):
            Whether to process audio in streaming mode (chunked), using `is_first_audio_chunk` to
            distinguish the first chunk from subsequent ones.
        is_first_audio_chunk (`bool`, *optional*, defaults to `True`):
            Whether the current audio is the first chunk of a streaming session. Controls `center` in the
            feature extractor so per-chunk STFT reproduces a single full-utterance pass. Must be `True`
            when `is_streaming=False`.
        language (`str` or `list[str]`, *optional*, defaults to `"auto"`):
            Target language(s) for prompt conditioning. Either a
            single language string applied to the whole batch, or one string per audio. Accepts locales
            (`"en-US"`, `"de-DE"`, ...), bare codes (`"de"`), or `"auto"` for automatic language
            detection. Resolved via `prompt_dictionary` into the `prompt_ids` model input.

        Returns:
            [`BatchFeature`]: the [`NemotronAsrStreamingProcessor`] outputs, augmented with:

            - **prompt_ids** -- A `(batch_size,)` `torch.LongTensor` of language-prompt indices. Pass it
              to the model/`generate`; the model turns it into the broadcast one-hot used by
              `prompt_projector`.
        """
        if not is_streaming and not is_first_audio_chunk:
            raise ValueError("In non-streaming mode (`is_streaming=False`), `is_first_audio_chunk` must be `True`.")

        audio = make_list_of_audio(audio)

        output_kwargs = self._merge_kwargs(
            Nemotron3_5AsrProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if sampling_rate is None:
            logger.warning_once(
                f"You've provided audio without specifying the sampling rate. It will be assumed to be "
                f"{output_kwargs['audio_kwargs']['sampling_rate']}, which can result in silent errors."
            )
        elif sampling_rate != output_kwargs["audio_kwargs"]["sampling_rate"]:
            raise ValueError(
                f"The sampling rate of the audio ({sampling_rate}) does not match the sampling rate of the "
                f"processor ({output_kwargs['audio_kwargs']['sampling_rate']}). Please resample the audio to "
                f"the expected sampling rate."
            )

        if audio is not None:
            # `center=True` for the first/offline chunk, `center=False` for subsequent streaming chunks.
            inputs = self.feature_extractor(audio, center=bool(is_first_audio_chunk), **output_kwargs["audio_kwargs"])
        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])

        inputs["num_lookahead_tokens"] = self.default_num_lookahead_tokens
        inputs["prompt_ids"] = self._resolve_prompt_ids(language, len(audio))

        if text is None:
            return inputs

        inputs["labels"] = encodings["input_ids"]
        # Prepend the blank token to labels to form decoder_input_ids: the RNN-T decoder expects
        # [blank, label_0, ..., label_{U-1}] as input.
        if isinstance(text, str):
            text = [text]
        decoder_text = [self.blank_token + t for t in text]
        decoder_encodings = self.tokenizer(decoder_text, **output_kwargs["text_kwargs"])
        inputs["decoder_input_ids"] = decoder_encodings["input_ids"]
        return inputs

    @property
    def model_input_names(self):
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return feature_extractor_input_names + ["labels", "decoder_input_ids", "prompt_ids"]


@dataclass
class Nemotron3_5AsrRNNTOutput(BaseModelOutputWithPooling):
    """
    encoder_past_key_values (`Cache`, *optional*):
        Updated encoder attention K/V sliding-window cache, returned when encoding audio with `use_cache=True`
        (cache-aware streaming). Pass it to the next chunk's forward.
    padding_cache (`NemotronAsrStreamingEncoderCausalConvPaddingCache`, *optional*):
        Updated unified streaming conv cache (subsampling Conv2d + conformer depthwise Conv1d), returned when
        encoding audio with `use_cache=True`. Pass it to the next chunk's forward.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    decoder_cache: Nemotron3_5AsrRNNTDecoderCache | None = None

    encoder_past_key_values: Cache | None = None
    padding_cache: "NemotronAsrStreamingEncoderCausalConvPaddingCache | None" = None  # noqa: F821


class Nemotron3_5AsrPreTrainedModel(NemotronAsrStreamingPreTrainedModel):
    _no_split_modules = None
    _can_record_outputs = {}

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)


class Nemotron3_5AsrPromptProjector(nn.Module):
    def __init__(self, config: Nemotron3_5AsrConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.encoder_config.hidden_size + config.num_prompts, config.prompt_intermediate_size
        )
        self.act = nn.ReLU()
        self.linear_2 = nn.Linear(config.prompt_intermediate_size, config.encoder_config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@auto_docstring(
    custom_intro="""
    Nemotron3_5Asr Encoder with an RNN-T (Recurrent Neural Network Transducer) head and language-ID
    prompt conditioning.
    """
)
class Nemotron3_5AsrForRNNT(NemotronAsrStreamingForRNNT, Nemotron3_5AsrGenerationMixin):
    def __init__(self, config: Nemotron3_5AsrConfig):
        super().__init__(config)
        self.prompt_projector = Nemotron3_5AsrPromptProjector(config)
        self.post_init()

    @can_return_tuple
    def get_audio_features(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        prompt_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        encoder_outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = encoder_outputs.last_hidden_state

        if prompt_ids is None:
            logger.warning_once(
                "`prompt_ids` not provided; defaulting to "
                f"`config.default_prompt_id={self.config.default_prompt_id}` (auto language detection). "
                "Pass `language` to the processor or `prompt_ids` to condition on a specific language."
            )
            prompt_ids = torch.full(
                (hidden_states.shape[0],),
                self.config.default_prompt_id,
                dtype=torch.long,
                device=hidden_states.device,
            )
        prompt_ids = prompt_ids.to(hidden_states.device)
        one_hot = nn.functional.one_hot(prompt_ids, num_classes=self.config.num_prompts).to(hidden_states.dtype)
        one_hot = one_hot[:, None, :].expand(-1, hidden_states.shape[1], -1)
        fused = self.prompt_projector(torch.cat([hidden_states, one_hot], dim=-1))

        encoder_outputs.pooler_output = self.encoder_projector(fused)
        return encoder_outputs

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_cache: Nemotron3_5AsrRNNTDecoderCache | None = None,
        use_decoder_cache: bool | None = None,
        encoder_outputs: BaseModelOutputWithPooling | None = None,
        labels: torch.Tensor | None = None,
        num_lookahead_tokens: int | None = None,
        prompt_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Nemotron3_5AsrRNNTOutput:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*):
            Decoder input token ids for single-step inference.
        decoder_cache (`Nemotron3_5AsrRNNTDecoderCache`, *optional*):
            Decoder LSTM cache. Reused on blank predictions to skip the LSTM step.
        use_decoder_cache (`bool`, *optional*):
            Whether to allocate and use a decoder cache when none is provided.
        encoder_outputs (`tuple(torch.FloatTensor)`, *optional*):
            Pre-computed encoder outputs (last_hidden_state, pooler_output, ...).
        num_lookahead_tokens (`int`, *optional*):
            Right attention context (lookahead, in subsampled encoder frames) forwarded to the encoder.
            Defaults to `config.encoder_config.default_num_lookahead_tokens`.
        prompt_ids (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Language-prompt indices for language-ID conditioning. Produced by the processor from
            `language`. Turned into the broadcast one-hot consumed by `prompt_projector`.

        Example:

        ```python
        >>> from transformers import AutoProcessor, Nemotron3_5AsrForRNNT
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/nemotron-3.5-asr-streaming-0.6b"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = Nemotron3_5AsrForRNNT.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"], language="en-US")
        >>> outputs = model(**inputs)
        ```
        """
        if encoder_outputs is None:
            encoder_outputs = self.get_audio_features(
                input_features=input_features,
                attention_mask=attention_mask,
                num_lookahead_tokens=num_lookahead_tokens,
                prompt_ids=prompt_ids,
                **kwargs,
            )

        if use_decoder_cache and decoder_cache is None:
            decoder_cache = Nemotron3_5AsrRNNTDecoderCache()

        decoder_hidden_states = self.decoder(decoder_input_ids, cache=decoder_cache)
        logits = self.joint(
            encoder_hidden_states=encoder_outputs.pooler_output[:, :, None, :],
            decoder_hidden_states=decoder_hidden_states[:, None, :, :],
        ).squeeze(2)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, encoder_outputs=encoder_outputs)

        return Nemotron3_5AsrRNNTOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=encoder_outputs.last_hidden_state,
            pooler_output=encoder_outputs.pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            decoder_cache=decoder_cache,
            encoder_past_key_values=encoder_outputs.past_key_values,
            padding_cache=encoder_outputs.padding_cache,
        )


__all__ = [
    "Nemotron3_5AsrProcessor",
    "Nemotron3_5AsrRNNTOutput",
    "Nemotron3_5AsrForRNNT",
    "Nemotron3_5AsrPreTrainedModel",
]
