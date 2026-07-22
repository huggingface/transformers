# Copyright 2026 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Fun-ASR-Nano model."""

from dataclasses import dataclass

import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...activations import ACT2FN
from ...audio_utils import AudioInput
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...tokenization_utils_base import TextInput
from ...utils import auto_docstring, can_return_tuple, is_torch_available, logging
from ..audioflamingo3.modeling_audioflamingo3 import (
    AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3Model,
    AudioFlamingo3ModelOutputWithPast,
    AudioFlamingo3PreTrainedModel,
)
from ..audioflamingo3.processing_audioflamingo3 import AudioFlamingo3Processor, AudioFlamingo3ProcessorKwargs
from ..auto import CONFIG_MAPPING, AutoConfig
from ..qwen3_asr.modeling_qwen3_asr import Qwen3ASRAudioAttention
from ..whisper.modeling_whisper import WhisperEncoder, WhisperEncoderLayer, eager_attention_forward


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


def _prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask[:, 0]
    return (1.0 - mask[:, None, None, :].to(dtype=dtype)) * torch.finfo(dtype).min


@auto_docstring(checkpoint="FunAudioLLM/Fun-ASR-Nano-2512-hf")
@strict
class FunAsrNanoEncoderConfig(PreTrainedConfig):
    r"""
    num_stacked_frames (`int`, *optional*, defaults to 7):
        Number of consecutive mel frames stacked by low-frame-rate feature extraction.
    num_timestamp_prediction_blocks (`int`, *optional*, defaults to 20):
        Number of timestamp prediction encoder blocks.
    kernel_size (`int`, *optional*, defaults to 11):
        Kernel size for the FSMN convolution.
    """

    model_type = "fun_asr_nano_encoder"
    attribute_map = {
        "output_size": "d_model",
        "output_dim": "d_model",
        "attention_heads": "encoder_attention_heads",
        "num_attention_heads": "encoder_attention_heads",
        "linear_units": "encoder_ffn_dim",
        "intermediate_size": "encoder_ffn_dim",
        "num_blocks": "encoder_layers",
        "tp_blocks": "num_timestamp_prediction_blocks",
        "dropout_rate": "dropout",
        "attention_dropout_rate": "attention_dropout",
    }

    num_mel_bins: int = 80
    num_stacked_frames: int = 7
    d_model: int = 512
    encoder_attention_heads: int = 4
    encoder_ffn_dim: int = 2048
    encoder_layers: int = 50
    num_timestamp_prediction_blocks: int = 20
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    activation_function: str = "relu"
    kernel_size: int = 11

    @property
    def input_size(self) -> int:
        return self.num_mel_bins * self.num_stacked_frames

    def __post_init__(self, **kwargs):
        legacy_input_size = kwargs.pop("input_size", None)
        if legacy_input_size is not None and legacy_input_size != self.input_size:
            raise ValueError(
                f"`input_size={legacy_input_size}` does not match `num_mel_bins * num_stacked_frames={self.input_size}`."
            )
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="FunAudioLLM/Fun-ASR-Nano-2512-hf")
@strict
class FunAsrNanoConfig(PreTrainedConfig):
    r"""
    encoder_config (`dict` or `PreTrainedConfig`, *optional*):
        Configuration for the audio encoder.
    adaptor_intermediate_size (`int`, *optional*, defaults to 2048):
        Hidden size of the adaptor feed-forward projection.
    adaptor_num_hidden_layers (`int`, *optional*, defaults to 2):
        Number of adaptor transformer layers.
    adaptor_num_attention_heads (`int`, *optional*, defaults to 8):
        Number of attention heads in the adaptor transformer layers.
    """

    model_type = "fun_asr_nano"
    attribute_map = {
        "audio_config": "encoder_config",
        "audio_encoder_config": "encoder_config",
        "audio_token_index": "audio_token_id",
        "adaptor_ffn_dim": "adaptor_intermediate_size",
        "adaptor_num_layers": "adaptor_num_hidden_layers",
        "adaptor_attention_heads": "adaptor_num_attention_heads",
    }
    sub_configs = {
        "encoder_config": AutoConfig,
        "text_config": AutoConfig,
    }

    encoder_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 151646
    adaptor_intermediate_size: int = 2048
    adaptor_num_hidden_layers: int = 2
    adaptor_num_attention_heads: int = 8
    activation_function: str = "relu"
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        audio_config = kwargs.pop("audio_config", None)
        audio_encoder_config = kwargs.pop("audio_encoder_config", None)
        if self.encoder_config is None:
            self.encoder_config = audio_encoder_config if audio_encoder_config is not None else audio_config

        if isinstance(self.encoder_config, dict):
            self.encoder_config["model_type"] = self.encoder_config.get("model_type", "fun_asr_nano_encoder")
            self.encoder_config = CONFIG_MAPPING[self.encoder_config["model_type"]](**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = CONFIG_MAPPING["fun_asr_nano_encoder"]()

        if isinstance(self.text_config, dict):
            text_config_model_type = self.text_config.get("model_type", "qwen3")
            self.text_config = CONFIG_MAPPING[text_config_model_type](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3"]()

        super().__post_init__(**kwargs)


class FunAsrNanoProcessorKwargs(AudioFlamingo3ProcessorKwargs):
    _defaults = {
        "audio_kwargs": {
            "sampling_rate": 16000,
            "return_attention_mask": True,
            "padding": True,
        },
        "text_kwargs": {
            "padding": True,
        },
        "common_kwargs": {
            "return_tensors": "pt",
            "padding_side": "left",
        },
    }


@auto_docstring
class FunAsrNanoProcessor(AudioFlamingo3Processor):
    valid_processor_kwargs = FunAsrNanoProcessorKwargs

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
        audio_token="<|object_ref_start|>",
        default_transcription_prompt="Transcribe the audio:",
    ):
        r"""
        audio_token (`str`, *optional*, defaults to `"<|object_ref_start|>"`):
            The token used as a placeholder for audio in the text.
        default_transcription_prompt (`str`, *optional*, defaults to `"Transcribe the audio:"`):
            Default prompt to use for transcription tasks when applying transcription requests.
        """
        super().__init__(
            feature_extractor,
            tokenizer,
            chat_template=chat_template,
            audio_token=audio_token,
            default_transcription_prompt=default_transcription_prompt,
            max_audio_len=None,
        )
        del self.max_audio_len

    def __call__(
        self,
        text: TextInput | list[TextInput],
        audio: AudioInput | None = None,
        output_labels: bool | None = False,
        **kwargs: Unpack[FunAsrNanoProcessorKwargs],
    ) -> BatchFeature:
        r"""
        output_labels (`bool`, *optional*, defaults to `False`):
            Whether to create causal-language-model labels from text token IDs. Padding and audio placeholder positions
            are masked with `-100`.
        """
        if "return_tensors" in kwargs and kwargs["return_tensors"] != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if output_labels:
            kwargs["return_mm_token_type_ids"] = True
        model_inputs = super().__call__(text=text, audio=audio, **kwargs)

        if output_labels:
            mm_token_type_ids = model_inputs.pop("mm_token_type_ids")
            labels = model_inputs["input_ids"].clone()
            labels.masked_fill_(mm_token_type_ids != 0, -100)
            if "attention_mask" in model_inputs:
                labels.masked_fill_(model_inputs["attention_mask"] == 0, -100)
            model_inputs["labels"] = labels
        return model_inputs

    def _get_audio_token_length(self, audio_lengths):
        return audio_lengths

    def _process_audio(self, audio, **kwargs):
        audio_inputs = self.feature_extractor(audio, **kwargs)
        if "attention_mask" not in audio_inputs:
            raise ValueError("FunAsrNanoProcessor requires an attention mask; set `return_attention_mask=True`.")
        audio_inputs["input_features_mask"] = audio_inputs.pop("attention_mask")
        audio_inputs["num_audio_tokens"] = self._get_audio_token_length(audio_inputs["feature_lengths"])
        audio_replacements = [self.replace_audio_token(audio_inputs, audio_idx=idx) for idx in range(len(audio))]
        return audio_inputs, audio_replacements

    def decode(self, *args, strip_prefix=False, **kwargs):
        """Decode token IDs and optionally remove common assistant framing from each transcription."""
        decoded = self.tokenizer.decode(*args, **kwargs)
        if not strip_prefix:
            return decoded
        if isinstance(decoded, str):
            return self._strip_assistant_prefix_and_quotes(decoded)
        return [self._strip_assistant_prefix_and_quotes(text) for text in decoded]

    @property
    def unused_input_names(self):
        return ["num_audio_tokens", "feature_lengths"]


@auto_docstring(
    custom_intro="""
    Base class for Fun-ASR-Nano outputs, with hidden states and attentions.
    """
)
@dataclass
class FunAsrNanoModelOutputWithPast(AudioFlamingo3ModelOutputWithPast):
    pass


@auto_docstring
class FunAsrNanoPreTrainedModel(AudioFlamingo3PreTrainedModel):
    config: FunAsrNanoConfig
    base_model_prefix = "model"
    input_modalities = ("audio", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["FunAsrNanoEncoderStem", "FunAsrNanoEncoderLayer", "FunAsrNanoAdaptorLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        from ..qwen3_omni_moe.modeling_qwen3_omni_moe import SinusoidsPositionEmbedding

        if isinstance(module, SinusoidsPositionEmbedding):
            position_embeddings = module.compute_default_singular_positional_embedding()
            init.copy_(module.positional_embedding, position_embeddings)


class FunAsrNanoAttention(Qwen3ASRAudioAttention):
    """Qwen3-ASR attention adapted for padded batch masks and checkpoint-compatible input projections."""

    def __init__(self, config: FunAsrNanoEncoderConfig, input_dim: int | None = None):
        super().__init__(config)
        input_dim = input_dim if input_dim is not None else config.d_model
        if input_dim != config.d_model:
            self.q_proj = nn.Linear(input_dim, config.d_model, bias=True)
            self.k_proj = nn.Linear(input_dim, config.d_model, bias=True)
            self.v_proj = nn.Linear(input_dim, config.d_model, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length, _ = hidden_states.shape
        target_shape = (batch_size, sequence_length, self.num_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(target_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(target_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(target_shape).transpose(1, 2)
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=self.dropout if self.training else 0.0,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, sequence_length, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights if output_attentions else None


class FunAsrNanoFSMN(nn.Module):
    """Depthwise feedforward sequential memory network used alongside self-attention."""

    def __init__(self, config: FunAsrNanoEncoderConfig):
        super().__init__()
        self.conv = nn.Conv1d(
            config.d_model,
            config.d_model,
            config.kernel_size,
            stride=1,
            padding=0,
            groups=config.d_model,
            bias=False,
        )
        left_padding = (config.kernel_size - 1) // 2
        right_padding = config.kernel_size - 1 - left_padding
        self.pad = nn.ConstantPad1d((left_padding, right_padding), 0.0)
        self.dropout = config.attention_dropout

    def forward(self, value_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is not None:
            if attention_mask.ndim == 3:
                attention_mask = attention_mask[:, 0]
            expanded_mask = attention_mask.unsqueeze(-1).to(dtype=value_states.dtype)
            value_states = value_states * expanded_mask
        else:
            expanded_mask = None

        hidden_states = self.conv(self.pad(value_states.transpose(1, 2))).transpose(1, 2)
        hidden_states = hidden_states + value_states
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if expanded_mask is not None:
            hidden_states = hidden_states * expanded_mask
        return hidden_states


class FunAsrNanoEncoderLayer(WhisperEncoderLayer):
    """SAN-M encoder layer combining standard self-attention with a separate FSMN branch."""

    def __init__(self, config: FunAsrNanoEncoderConfig):
        super().__init__(config)
        self.self_attn = FunAsrNanoAttention(config)
        self.fsmn = FunAsrNanoFSMN(config)

    def _forward_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.self_attn_layer_norm(hidden_states)
        value_states = self.self_attn.v_proj(hidden_states)
        additive_attention_mask = (
            _prepare_4d_attention_mask(attention_mask, hidden_states.dtype) if attention_mask is not None else None
        )
        attention_output, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=additive_attention_mask,
            **kwargs,
        )
        fsmn_output = self.fsmn(value_states, attention_mask)
        hidden_states = nn.functional.dropout(attention_output + fsmn_output, p=self.dropout, training=self.training)
        return hidden_states if residual is None else residual + hidden_states

    def _forward_feed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self._forward_attention(
            hidden_states,
            attention_mask,
            residual=hidden_states,
            **kwargs,
        )
        return self._forward_feed_forward(hidden_states)


class FunAsrNanoEncoderStem(FunAsrNanoEncoderLayer):
    """Position encoding and the first heterogeneous SAN-M layer."""

    @staticmethod
    def _create_position_embeddings(length: int, channels: int, max_timescale: int = 10000):
        from ..qwen3_omni_moe.modeling_qwen3_omni_moe import SinusoidsPositionEmbedding

        return SinusoidsPositionEmbedding(length, channels, max_timescale)

    def __init__(self, config: FunAsrNanoEncoderConfig):
        super().__init__(config)
        input_size = config.num_mel_bins * config.num_stacked_frames
        self.position_embeddings = self._create_position_embeddings(2049, input_size)
        self.self_attn_layer_norm = nn.LayerNorm(input_size)
        self.self_attn = FunAsrNanoAttention(config, input_dim=input_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states * (self.self_attn.embed_dim**0.5)
        sequence_length = hidden_states.shape[1]
        if sequence_length + 1 > self.position_embeddings.length:
            self.position_embeddings = self._create_position_embeddings(
                sequence_length + 1,
                self.position_embeddings.channels,
                self.position_embeddings.max_timescale,
            ).to(hidden_states.device)
        positions = self.position_embeddings(sequence_length + 1)[1:].to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        hidden_states = hidden_states + positions.unsqueeze(0)
        hidden_states = self._forward_attention(
            hidden_states,
            attention_mask,
            residual=None,
            **kwargs,
        )
        return self._forward_feed_forward(hidden_states)


@auto_docstring(
    custom_intro="""
    The Fun-ASR-Nano audio encoder (SenseVoice SAN-M architecture), without any head on top.
    """
)
class FunAsrNanoEncoder(WhisperEncoder):
    config_class = FunAsrNanoEncoderConfig
    main_input_name = "input_features"

    def __init__(self, config: FunAsrNanoEncoderConfig):
        PreTrainedModel.__init__(self, config)
        self.stem = FunAsrNanoEncoderStem(config)
        self.layers = nn.ModuleList([FunAsrNanoEncoderLayer(config) for _ in range(config.encoder_layers - 1)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.timestamp_prediction_layers = nn.ModuleList(
            [FunAsrNanoEncoderLayer(config) for _ in range(config.num_timestamp_prediction_blocks)]
        )
        self.timestamp_prediction_layer_norm = nn.LayerNorm(config.d_model)

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.stem

    def set_input_embeddings(self, value: nn.Module):
        self.stem = value

    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> BaseModelOutput:
        hidden_states = input_features.to(dtype=self.layer_norm.weight.dtype)

        all_hidden_states = () if output_hidden_states else None
        hidden_states = self.stem(hidden_states, input_features_mask, **kwargs)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        for layer in self.layers:
            hidden_states = layer(hidden_states, input_features_mask, **kwargs)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        hidden_states = self.layer_norm(hidden_states)

        for layer in self.timestamp_prediction_layers:
            hidden_states = layer(hidden_states, input_features_mask, **kwargs)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        hidden_states = self.timestamp_prediction_layer_norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)


class FunAsrNanoAdaptorAttention(FunAsrNanoAttention):
    pass


class FunAsrNanoAdaptorLayer(WhisperEncoderLayer):
    """Bidirectional self-attention adaptor layer."""

    def __init__(self, config: FunAsrNanoEncoderConfig):
        super().__init__(config)
        self.self_attn = FunAsrNanoAdaptorAttention(config)


class FunAsrNanoMultiModalProjector(nn.Module):
    """Audio adaptor projecting encoder output to the language-model dimension."""

    def __init__(self, config: FunAsrNanoConfig):
        nn.Module.__init__(self)
        self.config = config

        encoder_dim = config.encoder_config.d_model
        llm_dim = config.text_config.hidden_size

        self.linear_1 = nn.Linear(encoder_dim, config.adaptor_intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.linear_2 = nn.Linear(config.adaptor_intermediate_size, llm_dim)

        adaptor_config = FunAsrNanoEncoderConfig(
            num_mel_bins=1,
            num_stacked_frames=1,
            d_model=llm_dim,
            encoder_attention_heads=config.adaptor_num_attention_heads,
            encoder_ffn_dim=llm_dim // 4,
            encoder_layers=config.adaptor_num_hidden_layers,
            num_timestamp_prediction_blocks=0,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            activation_function=config.activation_function,
        )
        adaptor_config._attn_implementation = config.encoder_config._attn_implementation
        self.blocks = nn.ModuleList(
            [FunAsrNanoAdaptorLayer(adaptor_config) for _ in range(config.adaptor_num_hidden_layers)]
        )

    def forward(self, encoder_out: torch.Tensor, input_features_mask: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(encoder_out)
        x = self.act(x)
        x = self.linear_2(x)

        attention_mask = _prepare_4d_attention_mask(input_features_mask, x.dtype)
        for block in self.blocks:
            x = block(x, attention_mask)
        return x


@auto_docstring(
    custom_intro="""
    The Fun-ASR-Nano model (SenseVoice SAN-M audio encoder, a Transformer adaptor and a Qwen3 language model),
    without a language modeling head.
    """
)
class FunAsrNanoModel(AudioFlamingo3Model):
    @can_return_tuple
    @auto_docstring(
        custom_intro="This method is used to get the audio embeddings from input features, meaning inferring the audio encoder and the adaptor."
    )
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        r"""
        input_features (`torch.FloatTensor`):
            Audio features `(batch, time, feature_dim)` produced by the feature extractor (after LFR stacking).
        input_features_mask (`torch.Tensor`, *optional*):
            Padding mask for the audio feature sequence. When not provided, every sequence is assumed to be full length.

        Returns:
            [`~modeling_outputs.BaseModelOutputWithPooling`]: `last_hidden_state` holds the audio encoder output,
            `pooler_output` holds the projected audio embeddings (flattened over valid positions), and
            `hidden_states` holds the per-layer encoder states.
        """
        batch_size, max_len, _ = input_features.shape
        if input_features_mask is None:
            input_features_mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=input_features.device)

        encoder_outputs = self.audio_tower(
            input_features=input_features,
            input_features_mask=input_features_mask,
            return_dict=True,
            **kwargs,
        )
        encoder_out = encoder_outputs.last_hidden_state

        audio_embeds = self.multi_modal_projector(encoder_out, input_features_mask)
        pooler_output = audio_embeds[input_features_mask.to(device=audio_embeds.device, dtype=torch.bool)]

        return BaseModelOutputWithPooling(
            last_hidden_state=encoder_out,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@auto_docstring(
    custom_intro="""
    The Fun-ASR-Nano model for speech recognition: a SenseVoice SAN-M audio encoder, a Transformer adaptor and a
    Qwen3 language model with a language modeling head.
    """
)
class FunAsrNanoForConditionalGeneration(AudioFlamingo3ForConditionalGeneration):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}


__all__ = [
    "FunAsrNanoConfig",
    "FunAsrNanoEncoderConfig",
    "FunAsrNanoProcessor",
    "FunAsrNanoPreTrainedModel",
    "FunAsrNanoEncoder",
    "FunAsrNanoModel",
    "FunAsrNanoForConditionalGeneration",
]
