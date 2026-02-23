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
import math
from dataclasses import dataclass
from functools import cached_property
from types import GeneratorType

import torch
import torch.nn as nn

from ... import initialization as init
from ...activations import ACT2FN
from ...audio_utils import mel_filter_bank
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...models.lasr.feature_extraction_lasr import LasrFeatureExtractor
from ...models.llama.modeling_llama import LlamaRotaryEmbedding
from ...models.mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralForCausalLM,
    MistralMLP,
    MistralModel,
    MistralRMSNorm,
)
from ...models.voxtral.modeling_voxtral import (
    VoxtralForConditionalGeneration,
    VoxtralMultiModalProjector,
    VoxtralPreTrainedModel,
)
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
from ...utils.output_capturing import capture_outputs
from .configuration_voxtral_realtime import VoxtralRealtimeEncoderConfig


logger = logging.get_logger(__name__)


class VoxtralRealtimeFeatureExtractor(LasrFeatureExtractor):
    def __init__(
        self,
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        n_fft=400,
        win_length=400,
        padding_value=0.0,
        global_log_mel_max=1.5,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=win_length,
            padding_value=padding_value,
            **kwargs,
        )
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.global_log_mel_max = global_log_mel_max

    def _torch_extract_fbank_features(self, waveform, device: str = "cpu", center: bool = True):
        window = torch.hann_window(self.n_fft, device=device)
        stft = torch.stft(waveform, self.n_fft, self.hop_length, window=window, return_complex=True, center=center)
        magnitudes = stft[..., :-1].abs() ** 2

        mel_filters = torch.from_numpy(self.mel_filters).to(device, torch.float32)
        mel_spec = mel_filters.T @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if self.global_log_mel_max is not None:
            log_spec_max = torch.tensor(
                self.global_log_mel_max,
                device=log_spec.device,
                dtype=log_spec.dtype,
            )
        else:
            log_spec_max = log_spec.max()

        log_spec = torch.maximum(log_spec, log_spec_max - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        if device != "cpu":
            log_spec = log_spec.detach().cpu()
        return log_spec


class VoxtralRealtimeConv1dCacheLayer:
    def __init__(self):
        self.cache: torch.Tensor | None = None
        self.is_initialized: bool = False

    def lazy_initialization(self, hidden_states, conv_module):
        self.left_pad = conv_module.left_pad
        self.in_channels = conv_module.in_channels
        self.cache = torch.zeros(
            hidden_states.shape[0],
            self.in_channels,
            self.left_pad,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.cache)

        self.is_initialized = True

    def update(self, hidden_states, conv_module=None):
        if not self.is_initialized and conv_module is not None:
            self.lazy_initialization(hidden_states, conv_module)
        elif not self.is_initialized:
            raise ValueError(
                "VoxtralRealtimeConv1dCacheLayer is not initialized. Make sure to provide conv_module to the update method."
            )

        # get the padding states
        if self.left_pad > 0:
            shortfall = max(0, self.left_pad - hidden_states.shape[-1])
            if shortfall > 0:
                padding_states = torch.cat([self.cache[:, :, -shortfall:], hidden_states], dim=-1)
            else:
                padding_states = hidden_states[:, :, -self.left_pad :]
        else:
            padding_states = torch.empty(
                hidden_states.shape[0], self.in_channels, 0, dtype=hidden_states.dtype, device=hidden_states.device
            )

        current_cache = self.cache.clone()
        self.cache.copy_(padding_states)

        return current_cache


class VoxtralRealtimeConv1dPaddingCache:
    def __init__(self):
        self.layers = {}

    def update(self, hidden_states, cache_key, conv_module):
        if cache_key not in self.layers:
            self.layers[cache_key] = VoxtralRealtimeConv1dCacheLayer()

        padding_states = self.layers[cache_key].update(hidden_states, conv_module)
        padded_hidden_states = torch.cat([padding_states, hidden_states], dim=-1)
        return padded_hidden_states


@dataclass
class VoxtralRealtimeEncoderOutput(BaseModelOutputWithPast):
    padding_cache: VoxtralRealtimeConv1dPaddingCache | None = None


@dataclass
class VoxtralRealtimeCausalLMOutputWithPast(CausalLMOutputWithPast):
    r"""
    Args:
        encoder_past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and value in the self-attention blocks) for the audio encoder
            that can be used to speed up sequential decoding.
        padding_cache (`VoxtralRealtimeConv1dPaddingCache`, *optional*):
            Cache for padding in convolutional layers to maintain state across streaming chunks.
    """

    encoder_past_key_values: Cache | None = None
    padding_cache: VoxtralRealtimeConv1dPaddingCache | None = None


class VoxtralRealtimeRotaryEmbedding(LlamaRotaryEmbedding): ...


class VoxtralRealtimeCausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        cache_key: str,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.cache_key = cache_key

    @cached_property
    def left_pad(self):
        effective_kernel_size = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        return effective_kernel_size - self.stride[0]

    def forward(
        self,
        x: torch.Tensor,
        padding_cache: VoxtralRealtimeConv1dPaddingCache | None = None,
    ) -> torch.Tensor:
        if padding_cache is not None:
            x = padding_cache.update(x, self.cache_key, self)
        else:
            x = nn.functional.pad(x, (self.left_pad, 0))

        return super().forward(x)


class VoxtralRealtimeRMSNorm(MistralRMSNorm): ...


class VoxtralRealtimeAttention(MistralAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        # similar to Whisper's original implementation the k projection does **not** have a bias
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)


class VoxtralRealtimeMLP(MistralMLP):
    def __init__(self, config):
        super().__init__(config)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)


class VoxtralRealtimeEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv1 = VoxtralRealtimeCausalConv1d(
            config.num_mel_bins, config.hidden_size, kernel_size=3, cache_key="conv1"
        )
        self.conv2 = VoxtralRealtimeCausalConv1d(
            config.hidden_size, config.hidden_size, kernel_size=3, stride=2, cache_key="conv2"
        )

    def forward(self, input_features, padding_cache=None):
        inputs_embeds = nn.functional.gelu(self.conv1(input_features, padding_cache=padding_cache))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds, padding_cache=padding_cache))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        return inputs_embeds


class VoxtralRealtimeEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = VoxtralRealtimeAttention(config, layer_idx)
        self.self_attn_layer_norm = VoxtralRealtimeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.activation_fn = ACT2FN[config.activation_function]
        self.final_layer_norm = VoxtralRealtimeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = VoxtralRealtimeMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class VoxtralRealtimePreTrainedModel(VoxtralPreTrainedModel, PreTrainedModel):
    # TODO: @eustlb, this should be enabled soon
    _can_compile_fullgraph = False

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(module)
        if isinstance(module, VoxtralRealtimeTimeEmbedding):
            inv_freq = torch.exp(-math.log(module.theta) * torch.arange(module.dim // 2).float() / (module.dim // 2))
            init.copy_(module.inv_freq, inv_freq)


@auto_docstring(
    custom_intro="""
    The VoxtralRealtime encoder, which is a Whisper encoder.
    """
)
class VoxtralRealtimeEncoder(VoxtralRealtimePreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`VoxtralRealtimeEncoderLayer`].

    Args:
        config: VoxtralRealtimeEncoderConfig
    """

    config: VoxtralRealtimeEncoderConfig
    main_input_name = "input_features"
    input_modalities = "audio"
    _no_split_modules = ["VoxtralRealtimeEncoderLayer"]
    _can_record_outputs = {
        "attentions": VoxtralRealtimeAttention,
        "hidden_states": VoxtralRealtimeEncoderLayer,
    }

    def __init__(self, config):
        super().__init__(config)
        self.embedder = VoxtralRealtimeEmbedder(config)
        self.layers = nn.ModuleList(
            [VoxtralRealtimeEncoderLayer(config, layer_idx) for layer_idx in range(config.encoder_layers)]
        )
        self.norm = VoxtralRealtimeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = VoxtralRealtimeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_features: torch.FloatTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        padding_cache: VoxtralRealtimeConv1dPaddingCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        use_padding_cache: bool | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        padding_cache (`VoxtralRealtimeConv1dPaddingCache`, *optional*):
            Cache for padding in convolutional layers to maintain state across streaming chunks.
        use_padding_cache (`bool`, *optional*):
            Whether to use the padding cache.
        """
        if (input_features is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_features or inputs_embeds")

        if use_padding_cache and padding_cache is None:
            padding_cache = VoxtralRealtimeConv1dPaddingCache()

        if inputs_embeds is None:
            inputs_embeds = self.embedder(input_features, padding_cache)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return VoxtralRealtimeEncoderOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            padding_cache=padding_cache,
        )


class VoxtralRealtimeTextAdaRmsNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, 32, bias=False)
        self.linear2 = nn.Linear(32, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.linear1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class VoxtralRealtimeTextAttention(MistralAttention): ...


class VoxtralRealtimeTextMLP(MistralMLP): ...


class VoxtralRealtimeTextDecoderLayer(MistralDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.input_layernorm = VoxtralRealtimeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = VoxtralRealtimeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ada_rms_norm = VoxtralRealtimeTextAdaRmsNorm(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        t_cond: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states * (1 + self.ada_rms_norm(t_cond))

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class VoxtralRealtimeTextModel(MistralModel):
    def __init__(self, config):
        super().__init__(config)
        self.norm = VoxtralRealtimeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = VoxtralRealtimeRotaryEmbedding(config=config)


class VoxtralRealtimeTextForCausalLM(MistralForCausalLM):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, VoxtralRealtimeTextForCausalLM

        >>> model = VoxtralRealtimeTextForCausalLM.from_pretrained("mistralai/Voxtral-Mini-4B-Realtime-2602")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Voxtral-Mini-4B-Realtime-2602")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class VoxtralRealtimeTimeEmbedding(nn.Module):
    """Sinusoidal Embedding for encoding time"""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = torch.exp(-math.log(self.theta) * torch.arange(self.dim // 2).float() / (self.dim // 2))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, time_tensor: torch.Tensor) -> torch.Tensor:
        inv_freq = self.inv_freq.to(device=time_tensor.device, dtype=time_tensor.dtype)
        emb = time_tensor * inv_freq
        return torch.cat((emb.cos(), emb.sin()))


class VoxtralRealtimeMultiModalProjector(VoxtralMultiModalProjector):
    def __init__(self, config):
        super().__init__(config)
        self.linear_1 = nn.Linear(
            config.audio_config.hidden_size * config.downsample_factor, config.text_config.hidden_size, bias=False
        )


class VoxtralRealtimeForConditionalGeneration(VoxtralForConditionalGeneration, GenerationMixin):
    _keep_in_fp32_modules_strict = None

    def __init__(self, config):
        super().__init__(config)
        self.language_model = VoxtralRealtimeTextForCausalLM(config.text_config)
        self.time_embedding = VoxtralRealtimeTimeEmbedding(config.text_config.hidden_size)

    @can_return_tuple
    @auto_docstring(
        custom_intro="This method is used to get the audio embeddings from input features (a log mel spectrogram), meaning inferring the audio encoder and the multi-modal projector."
    )
    def get_audio_features(
        self,
        input_features: torch.FloatTensor = None,
        padding_cache: VoxtralRealtimeConv1dPaddingCache | None = None,
        encoder_inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_features (`torch.FloatTensor`):
            Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
            obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]` or a
            `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
            `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
            and conversion into a tensor of type `torch.FloatTensor`. See [`~VoxtralRealtimeFeatureExtractor.__call__`]

        padding_cache (`VoxtralRealtimeConv1dPaddingCache`, *optional*):
            Cache for padding in convolutional layers to maintain state across streaming chunks.

        encoder_inputs_embeds (`torch.FloatTensor`, *optional*):
            Optionally, instead of passing `input_features` you can choose to directly pass an embedded representation for the encoder.
        """
        audio_outputs = self.audio_tower(
            input_features=input_features,
            inputs_embeds=encoder_inputs_embeds,
            past_key_values=past_key_values,
            padding_cache=padding_cache,
            return_dict=True,
            use_cache=use_cache,
            use_padding_cache=use_cache,
            **kwargs,
        )
        audio_hidden_states = audio_outputs.last_hidden_state
        audio_hidden_states = audio_hidden_states.reshape(
            audio_hidden_states.shape[0], -1, self.config.audio_config.hidden_size * self.config.downsample_factor
        )
        audio_embeds = self.multi_modal_projector(audio_hidden_states)
        audio_outputs.pooler_output = audio_embeds

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
        encoder_past_key_values: Cache | None = None,
        padding_cache: VoxtralRealtimeConv1dPaddingCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        num_delay_tokens: int | torch.Tensor = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> VoxtralRealtimeCausalLMOutputWithPast:
        r"""
        encoder_past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and value in the self-attention blocks) for the encoder that can be used to speed up sequential decoding.
        padding_cache (`VoxtralRealtimeConv1dPaddingCache`, *optional*):
            Cache for padding in convolutional layers to maintain state across streaming chunks.
        encoder_inputs_embeds (`torch.FloatTensor`, *optional*):
            Optionally, instead of passing `input_features` you can choose to directly pass an embedded representation for the encoder.
        num_delay_tokens (`int` or `torch.Tensor`, *optional*):
            Number of delay tokens used when preparing inputs, see [`~VoxtralRealtimeProcessor`] for more details.

        Example:

        ```python
        >>> import torch
        >>> from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
        >>> from datasets import load_dataset

        >>> repo_id = "mistralai/Voxtral-Mini-4B-Realtime-2602"

        >>> processor = AutoProcessor.from_pretrained(repo_id)
        >>> model = VoxtralRealtimeForConditionalGeneration.from_pretrained(repo_id, dtype=torch.bfloat16, device_map="auto")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> audio = ds[0]["audio"]["array"]

        >>> inputs = processor(audio, return_tensors="pt")
        >>> inputs = inputs.to(model.device, dtype=model.dtype)

        >>> outputs = model.generate(**inputs)
        >>> processor.batch_decode(outputs, skip_special_tokens=True)
        ```"""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if (input_features is None) ^ (encoder_inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_features or encoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_features is not None or encoder_inputs_embeds is not None:
            audio_outputs = self.get_audio_features(
                input_features=input_features,
                encoder_inputs_embeds=encoder_inputs_embeds,
                past_key_values=encoder_past_key_values,
                padding_cache=padding_cache,
                use_cache=use_cache,
                return_dict=True,
            )
            inputs_embeds += audio_outputs.pooler_output.to(inputs_embeds.device)

        if num_delay_tokens is None:
            num_delay_tokens = self.config.default_num_delay_tokens
            logger.warning_once(
                f"`num_delay_tokens` was not provided. "
                f"Falling back to `config.default_num_delay_tokens={num_delay_tokens}`. "
                f"Consider preparing inputs with [`~VoxtralRealtimeProcessor.__call__`] which automatically sets this parameter."
            )

        time_tensor = torch.full(
            (1,),
            num_delay_tokens,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        t_cond = self.time_embedding(time_tensor)
        t_cond = t_cond[None, ...]  # broadcastable to batch size

        outputs: CausalLMOutputWithPast = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            t_cond=t_cond,
            **kwargs,
        )
        return VoxtralRealtimeCausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            encoder_past_key_values=audio_outputs.past_key_values if use_cache else None,
            padding_cache=audio_outputs.padding_cache if use_cache else None,
        )

    def prepare_inputs_for_generation(
        self,
        *args,
        encoder_inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        model_inputs = GenerationMixin.prepare_inputs_for_generation(*args, **kwargs)

        if encoder_inputs_embeds is not None:
            start_idx = model_inputs["cache_position"][0] * self.config.downsample_factor
            end_idx = (model_inputs["cache_position"][-1] + 1) * self.config.downsample_factor
            model_inputs["encoder_inputs_embeds"] = encoder_inputs_embeds[:, start_idx:end_idx, :]

        return model_inputs

    def _prepare_model_inputs(
        self,
        inputs: torch.Tensor | None = None,
        bos_token_id: torch.Tensor | None = None,
        model_kwargs: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, str | None, dict[str, torch.Tensor]]:
        inputs, input_name, model_kwargs = GenerationMixin._prepare_model_inputs(inputs, bos_token_id, model_kwargs)

        input_features = model_kwargs.get("input_features")
        if input_features is not None and not isinstance(input_features, GeneratorType):
            model_kwargs["encoder_inputs_embeds"] = self.audio_tower.embedder(model_kwargs.pop("input_features"))

        elif isinstance(input_features, GeneratorType):
            input_features_generator = model_kwargs.pop("input_features")
            model_kwargs["input_features_generator"] = input_features_generator
            try:
                model_kwargs["input_features"] = next(input_features_generator)
            except StopIteration:
                self._stream_exhausted = True

        return inputs, input_name, model_kwargs

    def _has_unfinished_sequences(self, this_peer_finished: bool, synced_gpus: bool, device: torch.device) -> bool:
        if getattr(self, "_stream_exhausted", False):
            self._stream_exhausted = False
            return False
        return GenerationMixin._has_unfinished_sequences(this_peer_finished, synced_gpus, device)

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ):
        model_kwargs = GenerationMixin._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )

        if hasattr(outputs, "encoder_past_key_values"):
            model_kwargs["encoder_past_key_values"] = outputs.encoder_past_key_values

        if hasattr(outputs, "padding_cache"):
            model_kwargs["padding_cache"] = outputs.padding_cache

        input_features_generator = model_kwargs.get("input_features_generator")
        if input_features_generator is not None:
            try:
                model_kwargs["input_features"] = next(input_features_generator)
            except StopIteration:
                self._stream_exhausted = True

        return model_kwargs

    def _prepare_cache_for_generation(
        self,
        generation_config,
        model_kwargs: dict,
        generation_mode,
        batch_size: int,
        max_cache_length: int,
    ):
        GenerationMixin._prepare_cache_for_generation(
            generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
        )

        # NOTE: we use the encoder prefix here this is not a classical encoder-decoder model - no cross-attention
        # the model is better seen as a VLM/ AudioLM, so with an encoder that can take psat_key_values for it's forward pass
        if generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in ("static", "offloaded_static"):
                model_kwargs["encoder_past_key_values"] = self._get_encoder_cache(
                    cache_implementation=generation_config.cache_implementation,
                    batch_size=batch_size,
                    max_cache_len=self.config.audio_config.sliding_window,
                )
            else:
                raise ValueError(f"{generation_config.cache_implementation} is not supported for VoxtralRealtime")

    def _get_encoder_cache(self, cache_implementation: str, batch_size: int, max_cache_len: int) -> Cache:
        offload_cache = "offloaded" in cache_implementation

        if hasattr(self, "_encoder_cache"):
            cache_to_check = self._encoder_cache

        need_new_cache = (
            not hasattr(self, "_encoder_cache")
            or cache_to_check.offloading != offload_cache
            or cache_to_check.max_batch_size != batch_size
            or cache_to_check.max_cache_len < max_cache_len
        )

        if need_new_cache:
            self_attention_cache_kwargs = {
                "config": self.config.audio_config,
                "max_cache_len": max_cache_len,
                "offloading": offload_cache,
            }
            self._encoder_cache = StaticCache(**self_attention_cache_kwargs)
        else:
            self._encoder_cache.reset()
        return self._encoder_cache

    def _prepare_generation_config(
        self,
        generation_config,
        **kwargs,
    ):
        # Check if user explicitly provided max_length or max_new_tokens BEFORE
        # the base class applies defaults
        user_set_max_length = kwargs.get("max_length") is not None or (
            generation_config is not None and generation_config.max_length is not None
        )
        user_set_max_new_tokens = kwargs.get("max_new_tokens") is not None or (
            generation_config is not None and generation_config.max_new_tokens is not None
        )

        generation_config, model_kwargs = GenerationMixin._prepare_generation_config(generation_config, **kwargs)

        input_features = model_kwargs.get("input_features")
        if input_features is not None and not isinstance(input_features, GeneratorType):
            audio_length = input_features.shape[-1]
            num_audio_tokens = math.ceil(audio_length / self.config.audio_length_per_tok)
            # Stash for use in _prepare_generated_length
            generation_config._num_audio_tokens = num_audio_tokens

            if not user_set_max_length and not user_set_max_new_tokens:
                # Default: generate exactly num_audio_tokens
                generation_config.max_length = num_audio_tokens
                generation_config.max_new_tokens = None
                generation_config._voxtral_set_max_length = True
            else:
                generation_config._voxtral_set_max_length = False

        elif isinstance(input_features, GeneratorType):
            # In streaming mode, generation length is controlled by stream exhaustion only
            generation_config.max_new_tokens = None
            generation_config.max_length = int(1e9)
            generation_config._voxtral_set_max_length = True

        return generation_config, model_kwargs

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        has_default_min_length,
        model_input_name,
        input_ids_length,
        inputs_tensor,
    ):
        # If we set max_length ourselves (user didn't provide any length param),
        # prevent the base class from overriding it
        if getattr(generation_config, "_voxtral_set_max_length", False):
            has_default_max_length = False

        generation_config = GenerationMixin._prepare_generated_length(
            generation_config,
            has_default_max_length,
            has_default_min_length,
            model_input_name,
            input_ids_length,
            inputs_tensor,
        )

        # num_audio_tokens is a hard upper bound: we can never generate more
        # tokens than there are in the audio. Clamp after the base class has
        # resolved max_new_tokens into max_length.
        num_audio_tokens = getattr(generation_config, "_num_audio_tokens", None)
        if num_audio_tokens is not None:
            generation_config.max_length = min(generation_config.max_length, num_audio_tokens)

        return generation_config


__all__ = [
    "VoxtralRealtimeForConditionalGeneration",
    "VoxtralRealtimeEncoder",
    "VoxtralRealtimeFeatureExtractor",
    "VoxtralRealtimePreTrainedModel",
]
