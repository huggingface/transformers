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

import re

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...audio_utils import (
    AudioInput,
    make_audio_chat_template_content,
    make_list_of_audio_chat_template,
)
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...modeling_outputs import BaseModelOutputWithPooling
from ...processing_utils import Unpack, prepare_keyword_inputs, prepare_prompt_input
from ...tokenization_utils_base import TextInput
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.import_utils import requires
from ..audioflamingo3.configuration_audioflamingo3 import AudioFlamingo3Config
from ..audioflamingo3.modeling_audioflamingo3 import (
    AudioFlamingo3CausalLMOutputWithPast,
    AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3Model,
    AudioFlamingo3ModelOutputWithPast,
    AudioFlamingo3MultiModalProjector,
    AudioFlamingo3PreTrainedModel,
)
from ..auto import CONFIG_MAPPING
from ..vibevoice_asr.processing_vibevoice_asr import VibeVoiceAsrProcessor, VibeVoiceAsrProcessorKwargs
from ..whisper.modeling_whisper import WhisperEncoder


logger = logging.get_logger(__name__)

# Diarized segments look like `[start][S01]text[end]`, e.g. `[0.00][S01]Hello there.[7.56]`.
_SEGMENT_PATTERN = re.compile(
    r"\[(?P<start>[^\[\]]+)\]\[S(?P<speaker>\d+)\](?P<content>.*?)\[(?P<end>[^\[\]]+)\]", re.DOTALL
)


@auto_docstring(checkpoint="itazap/MOSS-Transcribe-Diarize-HF")
@strict
class MossTranscribeDiarizeConfig(AudioFlamingo3Config):
    r"""
    audio_merge_size (`int`, *optional*, defaults to 4):
        Number of consecutive Whisper encoder frames concatenated before the multi-modal projector.
    audio_chunk_size (`int`, *optional*, defaults to 480000):
        Whisper encoder window size in raw audio samples, used with `padding_mask` to recover `audio_chunk_mapping`.

    Example:

    ```python
    >>> from transformers import MossTranscribeDiarizeForConditionalGeneration, MossTranscribeDiarizeConfig

    >>> # Initializing a MossTranscribeDiarize configuration
    >>> configuration = MossTranscribeDiarizeConfig()

    >>> # Initializing a model from the configuration
    >>> model = MossTranscribeDiarizeForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "moss_transcribe_diarize"
    keys_to_ignore_at_inference = ["past_key_values"]

    _default_text_config_kwargs = {
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "max_position_embeddings": 131_072,
        "rope_theta": 1_000_000.0,
    }

    _default_audio_config_kwargs = {
        "num_mel_bins": 80,
        "d_model": 1024,
        "encoder_layers": 24,
        "encoder_attention_heads": 16,
        "encoder_ffn_dim": 4096,
        "max_source_positions": 1500,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "activation_dropout": 0.0,
        "activation_function": "gelu",
        "encoder_layerdrop": 0.0,
        "scale_embedding": False,
    }

    audio_token_id: int = 151671
    audio_merge_size: int = 4
    projector_hidden_act: str = "silu"
    audio_chunk_size: int = 480_000
    # Not declared on `AudioFlamingo3Config`; needed so `PreTrainedModel.get_expanded_tied_weights_keys` actually
    # ties `lm_head.weight` per `MossTranscribeDiarizeForConditionalGeneration._tied_weights_keys`.
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config = CONFIG_MAPPING["whisper"](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["whisper"](**self._default_audio_config_kwargs)

        if isinstance(self.text_config, dict):
            self.text_config = CONFIG_MAPPING["qwen3"](**{**self._default_text_config_kwargs, **self.text_config})
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3"](**self._default_text_config_kwargs)

        PreTrainedConfig.__post_init__(self, **kwargs)

    @property
    def adaptor_input_dim(self) -> int:
        return self.audio_config.d_model * self.audio_merge_size


class MossTranscribeDiarizeProcessorKwargs(VibeVoiceAsrProcessorKwargs):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "add_special_tokens": False,
        },
        "common_kwargs": {
            "return_tensors": "pt",
        },
        "audio_kwargs": {
            "padding": "max_length",
            "return_attention_mask": True,
        },
    }


@requires(backends=("torch",))
@auto_docstring
class MossTranscribeDiarizeProcessor(VibeVoiceAsrProcessor):
    r"""
    Constructs a MOSS-Transcribe-Diarize processor which wraps [`WhisperFeatureExtractor`] and
    [`Qwen2TokenizerFast`] into a single processor that inherits both the audio feature extraction and
    tokenizer functionalities.

    See the [`~MossTranscribeDiarizeProcessor.__call__`] for more information.

    Args:
        feature_extractor (`WhisperFeatureExtractor`):
            The feature extractor for audio processing.
        tokenizer (`Qwen2TokenizerFast`):
            The tokenizer for text processing.
        chat_template (`str`, *optional*):
            A Jinja template which will be used to convert lists of messages in a chat into a tokenizable string.
    """

    valid_processor_kwargs = MossTranscribeDiarizeProcessorKwargs
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "Qwen2TokenizerFast"

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        chat_template: str | None = None,
        audio_token: str = "<|audio_pad|>",
        audio_bos_token: str = "<|audio_start|>",
        audio_eos_token: str = "<|audio_end|>",
        audio_tokens_per_second: float = 12.5,
        audio_merge_size: int = 4,
        time_marker_every_seconds: int = 2,
    ):
        r"""
        audio_token (`str`, *optional*, defaults to `"<|audio_pad|>"`):
            Special token used to represent audio inputs in the chat template.
        audio_bos_token (`str`, *optional*, defaults to `"<|audio_start|>"`):
            Special token marking the start of the audio span in the chat template.
        audio_eos_token (`str`, *optional*, defaults to `"<|audio_end|>"`):
            Special token marking the end of the audio span in the chat template.
        audio_tokens_per_second (`float`, *optional*, defaults to 12.5):
            Expected number of audio placeholder tokens per second of input audio.
        audio_merge_size (`int`, *optional*, defaults to 4):
            Whisper frame merge factor used when counting audio tokens.
        time_marker_every_seconds (`int`, *optional*, defaults to 2):
            Insert numeric time-marker tokens into the audio span every N seconds. Set to `0` to disable
            time markers and emit a plain audio placeholder span instead.
        """
        super().__init__(
            feature_extractor,
            tokenizer,
            chat_template=chat_template,
            audio_token=audio_token,
            audio_bos_token=audio_bos_token,
            audio_eos_token=audio_eos_token,
        )
        del self.audio_duration_token
        self.audio_tokens_per_second = audio_tokens_per_second
        self.audio_merge_size = int(audio_merge_size)
        self.time_marker_every_seconds = time_marker_every_seconds

    @auto_docstring
    def __call__(
        self,
        text: TextInput | list[TextInput],
        audio: AudioInput | None = None,
        output_labels: bool | None = False,
        **kwargs: Unpack[MossTranscribeDiarizeProcessorKwargs],
    ) -> BatchFeature:
        r"""
        output_labels (bool, *optional*, default=False):
            Whether to return labels for training.

        Returns:
            [`BatchFeature`]: A dictionary with tokenized text (`input_ids`, `attention_mask`) and
            audio features (`input_values`, `padding_mask`).
        """
        output_kwargs = self._merge_kwargs(self.valid_processor_kwargs, **kwargs)
        return_tensors = output_kwargs["text_kwargs"].get("return_tensors", None)

        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if audio is not None:
            _, text, _, audio = self.prepare_inputs_layout(text=text, audio=audio, **kwargs)

        model_inputs = super().__call__(text=text, audio=audio, **kwargs)

        if output_labels:
            labels = model_inputs["input_ids"].clone()
            labels[labels == self.audio_token_id] = -100
            labels[labels == self.audio_bos_token_id] = -100
            labels[labels == self.audio_eos_token_id] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            model_inputs["labels"] = labels

        return BatchFeature(data=model_inputs, tensor_type="pt", skip_tensor_conversion=self.skip_tensor_conversion)

    def _get_audio_token_length(self, audio_lengths, audio_chunk_mapping):
        num_samples = int(audio_chunk_mapping[-1]) + 1
        conv_lengths = (audio_lengths - 1) // 2 + 1
        # Each chunk rounds up to a whole number of merge groups independently (into its own
        # zero-padded tail) before summing. If done after concatenating chunks, a chunk's
        # trailing framees would be lost (up to `self.audio_merge_size - 1`) to floor-division.
        chunk_tokens = (conv_lengths + self.audio_merge_size - 1) // self.audio_merge_size
        per_sample_tokens = torch.zeros(num_samples, dtype=torch.long)
        per_sample_tokens.scatter_add_(0, audio_chunk_mapping, chunk_tokens)
        return per_sample_tokens

    def _process_audio(self, audio: AudioInput, **kwargs) -> tuple[dict[str, torch.Tensor], list[str]]:
        # Determine number of Whisper-window chunks per sample, and flatten
        window_size = int(self.feature_extractor.n_samples)

        per_sample_lengths: list[int] = []
        per_sample_windows: list[int] = []
        flat_chunks: list[np.ndarray] = []
        for audio_el in audio:
            waveform = np.asarray(audio_el, dtype=np.float32).squeeze()
            n_samples = int(waveform.shape[0])
            n_win = max(1, (n_samples + window_size - 1) // window_size)
            per_sample_lengths.append(n_samples)
            per_sample_windows.append(n_win)

            time_cap = min(n_samples, n_win * window_size)
            for i in range(n_win):
                start = i * window_size
                end = min((i + 1) * window_size, time_cap)
                flat_chunks.append(waveform[start:end])

        audio_inputs = self.feature_extractor(flat_chunks, **kwargs)
        audio_inputs["input_features_mask"] = audio_inputs.pop("attention_mask")

        # `input_features_mask` alone can't tell chunks apart at a window boundary, so `padding_mask` records
        # each sample's raw length instead; the model recovers `audio_chunk_mapping` from it via `audio_chunk_size`.
        padding_mask = torch.zeros(len(audio), max(per_sample_lengths), dtype=torch.long)
        for idx, length in enumerate(per_sample_lengths):
            padding_mask[idx, :length] = 1
        audio_inputs["padding_mask"] = padding_mask

        audio_chunk_mapping = torch.repeat_interleave(
            torch.arange(len(audio), dtype=torch.long), torch.tensor(per_sample_windows, dtype=torch.long)
        )

        # Based on `WhisperEncoder._get_feat_extract_output_lengths` (conv stride 2 only), so the placeholder
        # token count matches `get_audio_features` from the same mask.
        audio_lengths = audio_inputs["input_features_mask"].sum(-1)
        audio_inputs["num_audio_tokens"] = self._get_audio_token_length(audio_lengths, audio_chunk_mapping)
        audio_replacements = [self.replace_audio_token(audio_inputs, audio_idx=idx) for idx in range(len(audio))]
        return audio_inputs, audio_replacements

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        num_tokens = int(audio_inputs["num_audio_tokens"][audio_idx])

        tokens_per_marker = int(self.audio_tokens_per_second * self.time_marker_every_seconds)
        if tokens_per_marker <= 0:
            return self.audio_token * num_tokens

        # This model is trained with literal second-count tokens (ex 2, 4, ...) interleaved
        # into the audio span every `time_marker_every_seconds`, so the model can ground text to timestamps.
        duration = num_tokens / float(self.audio_tokens_per_second)
        parts, consumed = [], 0
        for sec in range(self.time_marker_every_seconds, int(duration) + 1, self.time_marker_every_seconds):
            pos = (sec // self.time_marker_every_seconds) * tokens_per_marker
            segment_len = pos - consumed
            if segment_len > 0:
                parts.append(self.audio_token * segment_len)
                consumed += segment_len
            parts.append(str(sec))

        # Since `duration` rarely lands on an exact multiple of `time_marker_every_seconds`, the loop above
        # stops at the last full interval and any audio tokens beyond that point get no marker.
        remainder = num_tokens - consumed
        if remainder > 0:
            parts.append(self.audio_token * remainder)
        return "".join(parts)

    @property
    def model_input_names(self) -> list[str]:
        return super().model_input_names + ["input_features_mask", "padding_mask"]

    def apply_transcription_request(
        self,
        audio: str | list[str] | AudioInput,
        prompt: str | list[str] | None = None,
        keywords: str | list[str] | list[list[str]] | None = None,
        **kwargs: Unpack[MossTranscribeDiarizeProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare inputs for transcription / diarization without manually writing the chat template.

        Args:
            audio (`str`, `list[str]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                Audio to transcribe. Strings are interpreted as local paths or URLs and will be loaded automatically by
                the chat template loader; NumPy arrays and PyTorch tensors are forwarded directly.
            prompt (`str` or `list[str]`, *optional*):
                Custom prompt(s) to include in the user turn. A list must be the same length as the batch. When
                `None`, the chat template supplies the default timestamped diarization prompt.
            keywords (`str`, `list[str]`, or `list[list[str]]`, *optional*):
                Hotwords/domain terms (e.g. proper nouns) to bias transcription toward the correct spelling. A
                string or flat list of strings is shared across the batch; a nested list supplies separate
                keywords per audio sample.
            **kwargs:
                Additional keyword arguments forwarded to [`~MossTranscribeDiarizeProcessor.apply_chat_template`].
        """
        audio_items: list[str | np.ndarray] = list(make_list_of_audio_chat_template(audio))
        audio_items = [el.detach().cpu().numpy() if isinstance(el, torch.Tensor) else el for el in audio_items]

        batch_size = len(audio_items)
        if batch_size == 0:
            raise ValueError("`audio` must contain at least one sample.")

        prompts = prepare_prompt_input(prompt, batch_size, input_name="prompt")
        keyword_batches = prepare_keyword_inputs(keywords, batch_size)

        conversations = []
        for audio_item, prompt_text, keyword_list in zip(audio_items, prompts, keyword_batches):
            content = make_audio_chat_template_content(audio_item, prompt_text)
            if keyword_list:
                content.append({"type": "keywords", "keywords": keyword_list})
            conversations.append([{"role": "user", "content": content}])

        return self.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            **kwargs,
        )

    @staticmethod
    def parse_timestamp(value: str) -> float:
        """
        Parse a timestamp string into seconds. Accepts a plain float (`"7.56"`) or a colon-separated
        `MM:SS` / `HH:MM:SS` string (`"1:23"`, `"01:02:03"`), for MOSS-Transcribe-Diarize's diarization output.
        """
        return sum(float(part) * 60**i for i, part in enumerate(reversed(value.strip().split(":"))))

    def extract_speaker_dict(self, text: str | list[str]) -> list[dict] | str | list[list[dict] | str]:
        """
        Extract speaker/timestamp segments from raw output, returning original text on failure.

        Args:
            text (`str` or `list[str]`):
                Single text or batch of texts to parse from the output of `decode` with `return_format="raw"`.

        Returns:
            Parsed output(s). For single input, returns `list[dict]` or `str`.
            For batch input, returns `list[list[dict] | str]`.
        """
        is_single = isinstance(text, str)
        if is_single:
            text = [text]

        speaker_dict = []
        for t in text:
            t = t.strip()
            if t.startswith("assistant"):
                t = t[len("assistant") :].strip()

            segments = [
                {
                    "Start": self.parse_timestamp(match["start"]),
                    "End": self.parse_timestamp(match["end"]),
                    "Speaker": int(match["speaker"]),
                    "Content": match["content"].strip(),
                }
                for match in _SEGMENT_PATTERN.finditer(t)
            ]

            if not segments:
                logger.warning("Output doesn't contain any `[start][S0N]text[end]` segments.")
                speaker_dict.append(t)
            else:
                speaker_dict.append(segments)

        return speaker_dict[0] if is_single else speaker_dict


class MossTranscribeDiarizeMultiModalProjector(AudioFlamingo3MultiModalProjector):
    """VQ-style adaptor: MLP over merge-packed Whisper frames, then LayerNorm into LM space."""

    def __init__(self, config: MossTranscribeDiarizeConfig):
        super().__init__(config)
        self.linear_1 = nn.Linear(config.adaptor_input_dim, config.text_config.hidden_size, bias=config.projector_bias)
        self.norm = nn.LayerNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps, bias=True)

    def forward(self, audio_features):
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return self.norm(hidden_states)


class MossTranscribeDiarizePreTrainedModel(AudioFlamingo3PreTrainedModel):
    config_class = MossTranscribeDiarizeConfig
    _no_split_modules = ["MossTranscribeDiarizeEncoderLayer"]


class MossTranscribeDiarizeEncoder(WhisperEncoder):
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor) -> torch.LongTensor:
        """Computes the output length of the convolutional layers."""
        return (input_lengths - 1) // 2 + 1


class MossTranscribeDiarizeModelOutputWithPast(AudioFlamingo3ModelOutputWithPast):
    pass


class MossTranscribeDiarizeCausalLMOutputWithPast(AudioFlamingo3CausalLMOutputWithPast):
    pass


@auto_docstring(
    custom_intro="""
    The MOSS-Transcribe-Diarize model: Whisper encoder, 4x time merge, multi-modal projector, and Qwen3 language model.
    """
)
class MossTranscribeDiarizeModel(AudioFlamingo3Model):
    def __init__(self, config: MossTranscribeDiarizeConfig):
        super().__init__(config)
        # `AutoModel.from_config(config.audio_config)` would resolve a plain `WhisperConfig` to the full
        # `WhisperModel` (encoder + decoder); this model only ever needs the encoder, so build it directly.
        self.audio_tower = MossTranscribeDiarizeEncoder(config.audio_config)
        self.post_init()

    @can_return_tuple
    @auto_docstring(
        custom_intro="Extract MOSS audio embeddings from log-mel features, reassembling multi-chunk audio per sample."
    )
    def get_audio_features(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_features_mask (`torch.Tensor` of shape `(num_chunks, feature_sequence_length)`):
            Mask marking valid (non-padded) feature indices, one row per chunked log-mel feature row in
            `input_features`. Used to compute each chunk's valid encoder-output length.
        padding_mask (`torch.Tensor` of shape `(batch_size, max_audio_length)`):
            Mask marking each audio sample's valid raw-audio length, one row per sample. Used with
            `config.audio_chunk_size` to recover `audio_chunk_mapping`.
        """
        device = input_features.device

        # `WhisperEncoder` does not support masking `input_features` (silence in the padded log-mel region is
        # ignored by convention), so only the post-hoc lengths are needed to trim the encoder's output below.
        # It also doesn't cast `input_features` to its own dtype/device internally (unlike `Qwen2AudioEncoder`),
        # so that has to happen here.
        merge_size = self.config.audio_merge_size
        conv_lengths = self.audio_tower._get_feat_extract_output_lengths(input_features_mask.sum(-1).to(device=device))
        input_features = input_features.to(
            device=self.audio_tower.conv1.weight.device, dtype=self.audio_tower.conv1.weight.dtype
        )

        audio_outputs = self.audio_tower(input_features, return_dict=True, **kwargs)
        audio_embeds = audio_outputs.last_hidden_state

        # Round each chunk's valid length up to a whole number of merge groups independently (zero-padded),
        # matching `_get_audio_token_length`. This happens before concatenation, so a >30s audio's
        # trailing Whisper window doesn't lose frames or leak them into a neighboring chunk's merge group.
        padded_lengths = ((conv_lengths + merge_size - 1) // merge_size) * merge_size
        keep_mask = torch.arange(audio_embeds.shape[1], device=device)[None, :] < padded_lengths[:, None]

        valid_frames = audio_embeds[keep_mask.to(audio_embeds.device)]

        hidden_size = valid_frames.shape[-1]
        merged_features = valid_frames.reshape(-1, merge_size * hidden_size)
        audio_outputs.pooler_output = self.multi_modal_projector(merged_features)
        return audio_outputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MossTranscribeDiarizeModelOutputWithPast:
        r"""
        input_features_mask (`torch.Tensor` of shape `(num_chunks, feature_sequence_length)`, *optional*):
            Mask marking valid (non-padded) feature indices, one row per chunked log-mel feature row in
            `input_features`. Used to compute each chunk's valid encoder-output length.
        padding_mask (`torch.Tensor` of shape `(batch_size, max_audio_length)`, *optional*):
            Mask marking each audio sample's valid raw-audio length. Used with `config.audio_chunk_size` to
            recover `audio_chunk_mapping`.
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        audio_embeds = None
        if input_features is not None and input_ids is not None:
            audio_embeds = self.get_audio_features(
                input_features=input_features,
                input_features_mask=input_features_mask,
                padding_mask=padding_mask,
            ).pooler_output
            special_audio_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, audio_features=audio_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_audio_mask, audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            )

        outputs = self.language_model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return MossTranscribeDiarizeModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=audio_embeds,
        )


@auto_docstring(
    custom_intro="""
    The MOSS-Transcribe-Diarize model for conditional generation with transcription and speaker diarization.
    """
)
class MossTranscribeDiarizeForConditionalGeneration(AudioFlamingo3ForConditionalGeneration):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: MossTranscribeDiarizeConfig):
        super().__init__(config)
        self.model = MossTranscribeDiarizeModel(config)
        self.post_init()

    @auto_docstring
    def get_audio_features(self, *args, **kwargs):
        return self.model.get_audio_features(*args, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MossTranscribeDiarizeCausalLMOutputWithPast:
        r"""
        input_features_mask (`torch.Tensor` of shape `(num_chunks, feature_sequence_length)`, *optional*):
            Mask marking valid (non-padded) feature indices, one row per chunked log-mel feature row in
            `input_features`. Used to compute each chunk's valid encoder-output length.
        padding_mask (`torch.Tensor` of shape `(batch_size, max_audio_length)`, *optional*):
            Mask marking each audio sample's valid raw-audio length. Used with `config.audio_chunk_size` to
            recover `audio_chunk_mapping`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss.

        Example:

        ```python
        >>> from transformers import MossTranscribeDiarizeForConditionalGeneration, AutoProcessor

        >>> model_id = "itazap/MOSS-Transcribe-Diarize-HF"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(model_id, device_map="auto")
        >>> inputs = processor.apply_transcription_request("https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3")
        >>> inputs = inputs.to(model.device, dtype=model.dtype)
        >>> outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)
        >>> processor.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            padding_mask=padding_mask,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return MossTranscribeDiarizeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=outputs.audio_hidden_states,
        )


__all__ = [
    "MossTranscribeDiarizeConfig",
    "MossTranscribeDiarizeProcessor",
    "MossTranscribeDiarizePreTrainedModel",
    "MossTranscribeDiarizeEncoder",
    "MossTranscribeDiarizeModel",
    "MossTranscribeDiarizeForConditionalGeneration",
]
