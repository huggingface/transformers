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

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...audio_utils import AudioInput, make_list_of_audio_chat_template
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...modeling_outputs import BaseModelOutputWithPooling
from ...processing_utils import ProcessorMixin, Unpack, prepare_prompt_input
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.import_utils import requires
from ..audioflamingo3.modeling_audioflamingo3 import (
    AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3MultiModalProjector,
)
from ..auto import CONFIG_MAPPING
from ..glmasr.configuration_glmasr import GlmAsrConfig
from ..glmasr.modeling_glmasr import (
    GlmAsrModel,
    GlmAsrModelOutputWithPast,
    GlmAsrPreTrainedModel,
)
from ..glmasr.processing_glmasr import GlmAsrProcessor, GlmAsrProcessorKwargs


@auto_docstring
@strict
class MossTranscribeDiarizeConfig(GlmAsrConfig):
    r"""
    audio_merge_size (`int`, *optional*, defaults to 4):
        Number of consecutive Whisper encoder frames concatenated before the multi-modal projector.
    audio_encoder_stride (`int`, *optional*, defaults to 2):
        Temporal downsampling factor from the Whisper encoder convolutions used when counting audio tokens.
    adaptor_input_dim (`int`, *optional*):
        Input dimension of the multi-modal projector. Defaults to `audio_config.d_model * audio_merge_size`.
    projector_bias (`bool`, *optional*, defaults to `True`):
        Whether to use bias in the multi-modal projector linear layers.
    """

    model_type = "moss_transcribe_diarize"
    keys_to_ignore_at_inference = ["past_key_values"]

    _default_text_config_kwargs = {
        "vocab_size": 151936,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 40960,
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
    audio_encoder_stride: int = 2
    adaptor_input_dim: int | None = None
    projector_hidden_act: str = "silu"
    projector_bias: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            audio_model_type = self.audio_config.get("model_type", "qwen2_audio_encoder")
            # Original checkpoints use WhisperConfig (`model_type="whisper"`).
            if audio_model_type in (None, "whisper"):
                audio_model_type = "qwen2_audio_encoder"
            self.audio_config["model_type"] = audio_model_type
            self.audio_config = CONFIG_MAPPING[audio_model_type](**self.audio_config)
        elif getattr(self.audio_config, "model_type", None) == "whisper":
            audio_config_dict = self.audio_config.to_dict()
            audio_config_dict["model_type"] = "qwen2_audio_encoder"
            self.audio_config = CONFIG_MAPPING["qwen2_audio_encoder"](**audio_config_dict)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["qwen2_audio_encoder"](**self._default_audio_config_kwargs)

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen3")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](
                **{**self._default_text_config_kwargs, **self.text_config}
            )
        elif self.text_config is None:
            num_layers = self._default_text_config_kwargs["num_hidden_layers"]
            self.text_config = CONFIG_MAPPING["qwen3"](
                **self._default_text_config_kwargs,
                tie_word_embeddings=self.tie_word_embeddings,
                layer_types=["full_attention"] * num_layers,
            )

        self.text_config.tie_word_embeddings = self.tie_word_embeddings
        if not getattr(self.text_config, "layer_types", None):
            self.text_config.layer_types = ["full_attention"] * self.text_config.num_hidden_layers

        if self.adaptor_input_dim is None:
            self.adaptor_input_dim = self.audio_config.d_model * self.audio_merge_size

        self.hidden_size = self.text_config.hidden_size
        # Skip GlmAsrConfig.__post_init__ (llama / glmasr_encoder defaults).
        PreTrainedConfig.__post_init__(self, **kwargs)


class MossTranscribeDiarizeProcessorKwargs(GlmAsrProcessorKwargs):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "add_special_tokens": False,
        },
        "common_kwargs": {
            "return_tensors": "pt",
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": "max_length",
            "return_tensors": "pt",
        },
    }


@requires(backends=("torch",))
@auto_docstring
class MossTranscribeDiarizeProcessor(GlmAsrProcessor):
    valid_processor_kwargs = MossTranscribeDiarizeProcessorKwargs

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        chat_template: str | None = None,
        audio_token: str = "<|audio_pad|>",
        audio_tokens_per_second: float = 12.5,
        audio_merge_size: int = 4,
        audio_encoder_stride: int = 2,
        time_marker_every_seconds: int = 2,
        enable_time_marker: bool = True,
    ):
        r"""
        audio_token (`str`, *optional*, defaults to `"<|audio_pad|>"`):
            Special token used to represent audio inputs in the chat template.
        audio_tokens_per_second (`float`, *optional*, defaults to 12.5):
            Expected number of audio placeholder tokens per second of input audio.
        audio_merge_size (`int`, *optional*, defaults to 4):
            Whisper frame merge factor used when counting audio tokens.
        audio_encoder_stride (`int`, *optional*, defaults to 2):
            Temporal downsampling factor from the Whisper encoder convolutions used when counting audio tokens.
        time_marker_every_seconds (`int`, *optional*, defaults to 2):
            Insert numeric time-marker tokens into the audio span every N seconds.
        enable_time_marker (`bool`, *optional*, defaults to `True`):
            Whether to inject time-marker tokens into the audio placeholder span.
        """
        self.audio_token = audio_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
        ProcessorMixin.__init__(self, feature_extractor, tokenizer, chat_template=chat_template)
        self.audio_tokens_per_second = audio_tokens_per_second
        self.audio_merge_size = int(audio_merge_size)
        self.audio_encoder_stride = int(audio_encoder_stride)
        self.time_marker_every_seconds = time_marker_every_seconds
        self.enable_time_marker = enable_time_marker
        self.audio_start_token = getattr(tokenizer, "audio_start_token", "<|audio_start|>")
        self.audio_end_token = getattr(tokenizer, "audio_end_token", "<|audio_end|>")

    def _process_audio(self, audio: AudioInput, **kwargs) -> tuple[dict[str, torch.Tensor], list[str]]:
        # Determine number of Whisper-window chunks per sample, and flatten
        window_size = int(self.feature_extractor.n_samples)
        token_stride = int(self.feature_extractor.hop_length) * self.audio_encoder_stride * self.audio_merge_size

        per_sample_windows: list[int] = []
        flat_chunks: list[np.ndarray] = []
        feature_lengths: list[int] = []
        for audio_el in audio:
            waveform = np.asarray(audio_el, dtype=np.float32).squeeze()
            n_samples = int(waveform.shape[0])
            n_win = max(1, (n_samples + window_size - 1) // window_size)
            per_sample_windows.append(n_win)

            time_cap = min(n_samples, n_win * window_size)
            for i in range(n_win):
                start = i * window_size
                end = min((i + 1) * window_size, time_cap)
                chunk = waveform[start:end]
                flat_chunks.append(chunk)
                feature_lengths.append((chunk.shape[0] - 1) // token_stride + 1)

        if flat_chunks:
            input_features = self.feature_extractor(flat_chunks, **kwargs)["input_features"]
        else:
            input_features = torch.empty(
                (0, int(self.feature_extractor.feature_size), int(self.feature_extractor.nb_max_frames)),
            )

        # MOSS chunks per-sample audio into Whisper windows upstream (unlike AF3-style token counting), so
        # the model needs `audio_chunk_mapping` to reassemble chunks per sample before merge + projection.
        audio_feature_lengths = torch.tensor(feature_lengths, dtype=torch.long)
        audio_chunk_mapping = torch.repeat_interleave(
            torch.arange(len(audio), dtype=torch.long), torch.tensor(per_sample_windows, dtype=torch.long)
        )
        num_audio_tokens = torch.zeros(len(audio), dtype=torch.long)
        num_audio_tokens.scatter_add_(0, audio_chunk_mapping, audio_feature_lengths)
        audio_inputs = {
            "input_features": input_features,
            "audio_feature_lengths": audio_feature_lengths,
            "audio_chunk_mapping": audio_chunk_mapping,
            "num_audio_tokens": num_audio_tokens,
        }

        audio_replacements = [self.replace_audio_token(audio_inputs, audio_idx=idx) for idx in range(len(audio))]
        return audio_inputs, audio_replacements

    def _get_audio_token_length(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("MOSS counts audio tokens per Whisper window in `_process_audio`.")

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        num_tokens = int(audio_inputs["num_audio_tokens"][audio_idx])
        if self.enable_time_marker and self.time_marker_every_seconds > 0:
            return self._build_time_marker_span(num_tokens)
        return self.audio_token * num_tokens
        
    def _build_time_marker_span(self, num_tokens: int) -> str:
        num_tokens = int(num_tokens)
        if num_tokens <= 0:
            return ""

        tokens_per_marker = int(self.audio_tokens_per_second * self.time_marker_every_seconds)
        if tokens_per_marker <= 0:
            return self.audio_token * num_tokens

        duration = num_tokens / float(self.audio_tokens_per_second)
        parts, consumed = [], 0
        for sec in range(self.time_marker_every_seconds, int(duration) + 1, self.time_marker_every_seconds):
            pos = (sec // self.time_marker_every_seconds) * tokens_per_marker
            segment_len = pos - consumed
            if segment_len > 0:
                parts.append(self.audio_token * segment_len)
                consumed += segment_len
            parts.append(str(sec))

        remainder = num_tokens - consumed
        if remainder > 0:
            parts.append(self.audio_token * remainder)
        return "".join(parts)



    @property
    def model_input_names(self) -> list[str]:
        names = [name for name in super().model_input_names if name != "input_features_mask"]
        return names + ["audio_feature_lengths", "audio_chunk_mapping"]

    def apply_transcription_request(
        self,
        audio: str | list[str] | AudioInput,
        prompt: str | list[str] | None = None,
        **kwargs: Unpack[MossTranscribeDiarizeProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare inputs for transcription / diarization without manually writing the chat template.

        Args:
            audio (`str`, `list[str]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                Audio to transcribe. Strings are interpreted as local paths or URLs and will be loaded automatically by
                the chat template loader; NumPy arrays and PyTorch tensors are forwarded directly.
            prompt (`str` or `list[str]`, *optional*):
                Custom prompt(s) to include in the user turn. A list must be the same length as the batch. When `None`,
                only the audio is included in the user turn.
            **kwargs:
                Additional keyword arguments forwarded to [`~MossTranscribeDiarizeProcessor.apply_chat_template`].
        """
        audio_items: list[str | np.ndarray] = list(make_list_of_audio_chat_template(audio))
        audio_items = [el.detach().cpu().numpy() if isinstance(el, torch.Tensor) else el for el in audio_items]

        batch_size = len(audio_items)
        if batch_size == 0:
            raise ValueError("`audio` must contain at least one sample.")

        prompts = prepare_prompt_input(prompt, batch_size, input_name="prompt")

        conversations = []
        for prompt_text, audio_item in zip(prompts, audio_items):
            content = []
            if isinstance(audio_item, str):
                content.append({"type": "audio", "path": audio_item})
            else:
                content.append({"type": "audio", "audio": audio_item})

            if prompt_text is not None:
                content.append({"type": "text", "text": prompt_text})

            conversations.append([{"role": "user", "content": content}])

        return self.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            **kwargs,
        )


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


class MossTranscribeDiarizePreTrainedModel(GlmAsrPreTrainedModel):
    config_class = MossTranscribeDiarizeConfig
    base_model_prefix = "model"
    input_modalities = ("audio", "text")
    _no_split_modules = []
    _skip_keys_device_placement = ["past_key_values"]


@auto_docstring
@dataclass
class MossTranscribeDiarizeModelOutputWithPast(GlmAsrModelOutputWithPast):
    pass


@auto_docstring(
    custom_intro="""
    The MOSS-Transcribe-Diarize model: Whisper encoder, 4x time merge, multi-modal projector, and Qwen3 language model.
    """
)
class MossTranscribeDiarizeModel(GlmAsrModel):
    @can_return_tuple
    @auto_docstring(
        custom_intro="Extract MOSS audio embeddings from log-mel features, reassembling multi-chunk audio per sample."
    )
    def get_audio_features(
        self,
        input_features: torch.Tensor,
        audio_feature_lengths: torch.LongTensor,
        audio_chunk_mapping: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        audio_feature_lengths (`torch.LongTensor` of shape `(num_chunks,)`):
            Number of output tokens per chunked log-mel feature row in `input_features`.
        audio_chunk_mapping (`torch.LongTensor` of shape `(num_chunks,)`):
            Index of the source audio sample for each row in `input_features`.
        """
        num_chunks = input_features.shape[0]
        if audio_feature_lengths.numel() != num_chunks:
            raise ValueError(
                "`audio_feature_lengths` must contain one length per `input_features` chunk: "
                f"got {audio_feature_lengths.numel()} lengths for {num_chunks} chunks."
            )
        if audio_chunk_mapping.numel() != num_chunks:
            raise ValueError(
                "`audio_chunk_mapping` must contain one sample index per `input_features` chunk: "
                f"got {audio_chunk_mapping.numel()} indices for {num_chunks} chunks."
            )

        audio_outputs = self.audio_tower(input_features, return_dict=True, **kwargs)
        whisper_features = audio_outputs.last_hidden_state
        device = whisper_features.device
        audio_feature_lengths = audio_feature_lengths.to(device=device)
        audio_chunk_mapping = audio_chunk_mapping.to(device=device)

        # Lengths are post-merge token counts; recover Whisper frames before merge + projection.
        merge_size = self.config.audio_merge_size
        encoder_lengths = audio_feature_lengths * merge_size
        valid_mask = torch.arange(whisper_features.shape[1], device=device)[None, :] < encoder_lengths[:, None]

        if num_chunks == 0:
            audio_outputs.pooler_output = whisper_features.new_zeros(0, self.config.text_config.hidden_size)
            return audio_outputs

        num_samples = int(audio_chunk_mapping.max().item()) + 1
        per_sample_chunks: list[list[torch.Tensor]] = [[] for _ in range(num_samples)]
        for chunk_idx in range(num_chunks):
            sample_idx = int(audio_chunk_mapping[chunk_idx].item())
            per_sample_chunks[sample_idx].append(whisper_features[chunk_idx, valid_mask[chunk_idx]])

        projected = []
        for sample_chunks in per_sample_chunks:
            if not sample_chunks:
                continue
            sample_features = torch.cat(sample_chunks, dim=0).unsqueeze(0).to(self.dtype)
            batch_size, seq_len, hidden_size = sample_features.shape
            trimmed_seq_len = (seq_len // merge_size) * merge_size
            if trimmed_seq_len == 0:
                continue
            sample_features = sample_features[:, :trimmed_seq_len].reshape(
                batch_size, trimmed_seq_len // merge_size, hidden_size * merge_size
            )
            projected.append(self.multi_modal_projector(sample_features).squeeze(0))

        audio_outputs.pooler_output = (
            torch.cat(projected, dim=0)
            if projected
            else whisper_features.new_zeros(0, self.config.text_config.hidden_size)
        )
        return audio_outputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        audio_feature_lengths: torch.LongTensor | None = None,
        audio_chunk_mapping: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MossTranscribeDiarizeModelOutputWithPast:
        r"""
        audio_feature_lengths (`torch.LongTensor` of shape `(num_chunks,)`, *optional*):
            Number of output tokens per chunked log-mel feature row in `input_features`.
        audio_chunk_mapping (`torch.LongTensor` of shape `(num_chunks,)`, *optional*):
            Index of the source audio sample for each row in `input_features`.
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        audio_embeds = None
        if input_features is not None and input_ids is not None:
            if audio_feature_lengths is None or audio_chunk_mapping is None:
                raise ValueError(
                    "`audio_feature_lengths` and `audio_chunk_mapping` must be provided with `input_features`."
                )
            audio_embeds = self.get_audio_features(
                input_features=input_features,
                audio_feature_lengths=audio_feature_lengths,
                audio_chunk_mapping=audio_chunk_mapping,
            ).pooler_output
            special_audio_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, audio_features=audio_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_audio_mask, audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
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
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        audio_feature_lengths: torch.LongTensor | None = None,
        audio_chunk_mapping: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        audio_feature_lengths (`torch.LongTensor` of shape `(num_chunks,)`, *optional*):
            Number of output tokens per chunked log-mel feature row in `input_features`.
        audio_chunk_mapping (`torch.LongTensor` of shape `(num_chunks,)`, *optional*):
            Index of the source audio sample for each row in `input_features`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss.

        Example:

        ```python
        >>> from transformers import MossTranscribeDiarizeForConditionalGeneration, AutoProcessor

        >>> model_id = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
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
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            audio_feature_lengths=audio_feature_lengths,
            audio_chunk_mapping=audio_chunk_mapping,
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
    "MossTranscribeDiarizeProcessorKwargs",
    "MossTranscribeDiarizePreTrainedModel",
    "MossTranscribeDiarizeModel",
    "MossTranscribeDiarizeForConditionalGeneration",
    "MossTranscribeDiarizeMultiModalProjector",
]
