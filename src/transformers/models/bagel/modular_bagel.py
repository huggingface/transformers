# coding=utf-8
# Copyright 2025 ByteDance and The HuggingFace Team. All rights reserved.
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
from typing import Callable, Optional, Union

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PretrainedConfig
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, logging
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple, is_torch_available, torch_int
from ..auto import CONFIG_MAPPING, AutoConfig
from ..janus.modeling_janus import (
    JanusVQVAEConvDownsample,
    JanusVQVAEConvUpsample,
    JanusVQVAEEncoder,
    JanusVQVAEMidBlock,
    JanusVQVAEResnetBlock,
)
from ..qwen2.modeling_qwen2 import (
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)
from ..siglip.modeling_siglip import (
    SiglipAttention,
    SiglipEncoder,
    SiglipEncoderLayer,
    SiglipVisionEmbeddings,
)


if is_torch_available():
    import torch
    import torch.nn.functional as F
    from torch import nn

logger = logging.get_logger(__name__)


llm_config = {
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "hidden_act": "silu",
    "hidden_size": 3584,
    "initializer_range": 0.02,
    "intermediate_size": 18944,
    "max_position_embeddings": 32768,
    "max_window_layers": 28,
    "model_type": "qwen2",
    "num_attention_heads": 28,
    "num_hidden_layers": 28,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "sliding_window": 131072,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.43.1",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 152064,
}

vit_config = {
    "hidden_size": 1152,
    "image_size": 980,
    "intermediate_size": 4304,
    "model_type": "siglip_vision_model",
    "num_attention_heads": 16,
    "num_hidden_layers": 26,
    "patch_size": 14,
    "vision_use_head": False,
}


class BagelVQVAEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BambaModel`]. It is used to instantiate a
    BambaModel model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with defaults taken from [ibm-fms/Bamba-9.8b-2.2T-hf](https://huggingface.co/ibm-fms/Bamba-9.8b-2.2T-hf).

    The BambaModel is a hybrid [mamba2](https://github.com/state-spaces/mamba) architecture with SwiGLU.
    The checkpoints are  jointly trained by IBM, Princeton, and UIUC.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    model_type = "bagel_vqvae"
    base_config_key = "vq_config"

    def __init__(
        self,
        double_latent: bool = False,
        latent_channels: int = 16,
        num_patches: int = 32,
        latent_patch_size=2,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_multiplier: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        initializer_range=0.02,
        scale_factor=0.2611,
        shift_factor=0.1159,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.double_latent = double_latent
        self.latent_channels = latent_channels
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_multiplier = channel_multiplier
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.num_patches = num_patches
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor
        self.latent_patch_size = latent_patch_size


class BagelConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BambaModel`]. It is used to instantiate a
    BambaModel model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with defaults taken from [ibm-fms/Bamba-9.8b-2.2T-hf](https://huggingface.co/ibm-fms/Bamba-9.8b-2.2T-hf).

    The BambaModel is a hybrid [mamba2](https://github.com/state-spaces/mamba) architecture with SwiGLU.
    The checkpoints are  jointly trained by IBM, Princeton, and UIUC.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    model_type = "bagel"
    sub_configs = {
        "text_config": AutoConfig,
        "vision_config": AutoConfig,
        "vq_config": BagelVQVAEConfig,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        vq_config=None,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "qwen2")
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            logger.info("`text_config` is None. Initializing with default values")
            self.text_config = CONFIG_MAPPING["qwen2"](**llm_config)
        elif isinstance(text_config, PretrainedConfig):
            self.text_config = text_config
        else:
            raise ValueError(
                f"Invalid type for `text_config`. Must be either `dict` or `Qwen2Config`."
                f" Type found: {type(text_config)}"
            )

        if isinstance(text_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "siglip_vision_model")
            self.vision_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            logger.info("`vision_config` is None. Initializing with default values")
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"](**vit_config)
        elif isinstance(vision_config, PretrainedConfig):
            self.vision_config = vision_config
        else:
            raise ValueError(
                f"Invalid type for `vision_config`. Must be either `dict` or `SiglipVisionConfig`."
                f" Type found: {type(vision_config)}"
            )

        if vq_config is None:
            logger.info("`vq_config` is None. Initializing with default JanusVQVAEConfig values")
            self.vq_config = BagelVQVAEConfig()
        elif isinstance(vq_config, dict):
            self.vq_config = BagelVQVAEConfig(**vq_config)
        elif isinstance(vq_config, BagelVQVAEConfig):
            self.vq_config = vq_config
        else:
            raise ValueError(
                f"Invalid type for `vq_config`. Must be either `dict` or `JanusVQVAEConfig`."
                f" Type found: {type(vq_config)}"
            )
        self.vit_max_num_patch_per_side = 70
        self.max_latent_size = 64
        self.timestep_shift = 1.0
        super().__init__(**kwargs)


def patchify(image, patch_size):
    p = patch_size
    b, c, h, w = image.shape
    assert h % p == 0 and w % p == 0
    image = image.reshape(b, c, h // p, p, w // p, p)
    image = torch.einsum("bchpwq->bhwpqc", image)
    image = image.reshape(b, -1, p**2 * c)
    return image


def get_flattened_position_ids_extrapolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids


class BagelVisionEmbeddings(SiglipVisionEmbeddings):
    def __init__(self, config):
        super().__init__()

    def interpolate_pos_encoding(self):
        raise AttributeError("Not required")

    def convert_conv2d_to_linear(self, config, meta=False):
        W = self.patch_embedding.weight.permute(0, 2, 3, 1).reshape(
            self.embed_dim, config.num_channels * self.patch_size**2
        )
        linear_patch_embedding = nn.Linear(
            config.num_channels * self.patch_size**2, self.embed_dim, bias=True, device=W.device
        )
        linear_patch_embedding.weight.data = W
        linear_patch_embedding.bias.data = self.patch_embedding.bias.data
        self.patch_embedding = linear_patch_embedding

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)
        return patch_embeds


class BagelVisionAttention(SiglipAttention):
    pass


class BagelVisionEncoderLayer(SiglipEncoderLayer):
    def __init__(self, config):
        super().__init__()
        self.self_attn = BagelVisionAttention(config)


class BagelVisionEncoder(SiglipEncoder):
    def __init__(self, config: BagelConfig):
        super().__init__()
        self.layers = nn.ModuleList([BagelVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])


class BagelVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = BagelVisionEmbeddings(config)
        # Better way to do this
        self.embeddings.convert_conv2d_to_linear(config)
        self.encoder = BagelVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@auto_docstring
class BagelPreTrainedModel(PreTrainedModel):
    config_class = BagelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BagelTextDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, BagelTextRMSNorm):
            module.weight.data.fill_(1.0)


class BagelTextRMSNorm(Qwen2RMSNorm):
    pass


class BagelTextMLP(Qwen2MLP):
    pass


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def text_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class BagelTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_norm = BagelTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = BagelTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_norm_generation = BagelTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_generation = BagelTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_proj_generation = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj_generation = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj_generation = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj_generation = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        mode: Optional[str] = "und",
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = text_eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class BagelTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = BagelTextAttention(config=config, layer_idx=layer_idx)
        self.attention_type = config.layer_types[layer_idx]

        self.mlp = BagelTextMLP(config)
        self.mlp_generation = BagelTextMLP(config)
        self.input_layernorm = BagelTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_generation = BagelTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BagelTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_generation = BagelTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mode: Optional[str] = "und",
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class BagelTextRotaryEmbedding(Qwen2RotaryEmbedding):
    def __init__(self, config, device=None):
        super().__init__()


@auto_docstring
class BagelTextModel(BagelPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [BagelTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = BagelTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_generation = BagelTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = BagelTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        mode: Optional[str] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


###########
## VQVAE ##
###########


class BagelVQVAEResnetBlock(JanusVQVAEResnetBlock):
    pass


class BagelVQVAEMidBlock(JanusVQVAEMidBlock):
    pass


class BagelVQVAEConvDownsample(JanusVQVAEConvDownsample):
    pass


class BagelVQVAEConvUpsample(JanusVQVAEConvUpsample):
    pass


class BagelVQVAEEncoder(JanusVQVAEEncoder, nn.Module):
    def __init__(self, config: BagelVQVAEConfig):
        nn.Module().__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks
        base_channels = config.base_channels
        in_channels = config.in_channels
        double_latent = config.double_latent
        latent_channels = config.latent_channels
        channel_multiplier = config.channel_multiplier

        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        in_channel_multiplier = (1,) + tuple(channel_multiplier)
        self.in_channel_multiplier = in_channel_multiplier
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = base_channels * in_channel_multiplier[i_level]
            block_out = base_channels * channel_multiplier[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    BagelVQVAEResnetBlock(
                        config=config,
                        in_channels=block_in,
                        out_channels=block_out,
                    )
                )
                block_in = block_out

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = BagelVQVAEConvDownsample(block_in)
            self.down.append(down)

        self.mid = BagelVQVAEMidBlock(config, block_in)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(
            block_in,
            2 * latent_channels if double_latent else latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )


class BagelVQVAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks
        base_channels = config.base_channels
        latent_channels = config.latent_channels
        out_channels = config.out_channels
        block_in = base_channels * config.channel_multiplier[self.num_resolutions - 1]

        self.conv_in = nn.Conv2d(latent_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = BagelVQVAEMidBlock(config, block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = base_channels * config.channel_multiplier[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    BagelVQVAEResnetBlock(
                        config=config,
                        in_channels=block_in,
                        out_channels=block_out,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = BagelVQVAEConvUpsample(block_in)
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_state: torch.FloatTensor) -> torch.FloatTensor:
        hidden_state = self.conv_in(hidden_state)

        hidden_state = self.mid(hidden_state)

        # upsampling
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks + 1):
                hidden_state = self.up[i_level].block[i_block](hidden_state)
                if len(self.up[i_level].attn) > 0:
                    hidden_state = self.up[i_level].attn[i_block](hidden_state)
            if i_level != self.num_resolutions - 1:
                hidden_state = self.up[i_level].upsample(hidden_state)

        hidden_state = self.norm_out(hidden_state)
        hidden_state *= torch.sigmoid(hidden_state)
        hidden_state = self.conv_out(hidden_state)
        return hidden_state


class BagelVQVAEDiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean


@auto_docstring(
    custom_intro="""
    The VQ-VAE model used in Bagel for encoding/decoding images into discrete tokens.
    This model follows the "Make-a-scene: Scene-based text-to-image generation with human priors" paper from
    [ Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman](https://arxiv.org/abs/2203.13131).
    """
)
class BagelVQVAE(BagelPreTrainedModel):
    config_class = BagelVQVAEConfig
    _no_split_modules = [
        "BagelVQVAEAttnBlock",
        "BagelVQVAEResnetBlock",
    ]
    main_input_name = "pixel_values"

    def __init__(self, config: BagelVQVAEConfig):
        super().__init__(config)

        self.scale_factor = config.scale_factor
        self.shift_factor = config.shift_factor

        self.encoder = BagelVQVAEEncoder(config)
        self.decoder = BagelVQVAEDecoder(config)
        self.eval()  # Bagel's VQ model is frozen
        self.gradient_checkpointing = False

        # Initialize the VQVAE model.
        self.post_init()

    def encode(self, pixel_values: torch.LongTensor):
        hidden_states = self.encoder(pixel_values)
        hidden_states = self.scale_factor * (hidden_states - self.shift_factor)
        return hidden_states

    def decode(self, image_tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Decodes quantized token IDs into pixel values.
        Args:
            image_tokens (torch.LongTensor): Batch of token IDs.
        Returns:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                Pixel values decoded from the token IDs.
        """
        pixel_values = image_tokens / self.scale_factor + self.shift_factor
        pixel_values = self.decoder(pixel_values)
        return pixel_values

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
    ) -> torch.FloatTensor:
        hidden_states = self.encode(pixel_values)
        hidden_states = self.decode(hidden_states)
        return hidden_states


class BagelVisionConnector(nn.Module):
    def __init__(self, config: BagelConfig):
        super().__init__()
        in_dim = config.vision_config.hidden_size
        out_dim = config.text_config.hidden_size

        self.activation_fn = ACT2FN[config.vision_config.hidden_act]
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class BagelTimestepMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class BagelTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = BagelTimestepMLP(frequency_embedding_size, hidden_size)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.

        Args:
            t (torch.Tensor): A 1D tensor of shape (batch_size,) with scalar timesteps.
            dim (int): Output dimension of the embedding.
            max_period (int): Maximum period used for frequency calculation.

        Returns:
            torch.Tensor: Sinusoidal embedding of shape (batch_size, dim)
        """
        half = dim // 2
        device = t.device
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=device) / half)
        args = t[:, None].float() * freqs[None]  # shape: (B, half)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))  # pad to full dim if odd
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class BagelModel(BagelPreTrainedModel):
    def __init__(self, config: BagelConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.text_config.hidden_size
        self.latent_channel = config.vq_config.latent_channels
        self.latent_patch_size = config.vq_config.latent_patch_size
        self.patch_latent_dim = self.latent_patch_size**2 * self.latent_channel

        self.vision_tower = BagelVisionTransformer(config.vision_config)
        # self.language_model = BagelTextModel(config.text_config)
        # self.vq_model = BagelVQVAE(config.vq_config)

        self.vision_connecter = BagelVisionConnector(config)
        self.timestep_embedder = BagelTimestepEmbedder(config.text_config.hidden_size)
        self.vae2llm_connector = nn.Linear(self.patch_latent_dim, self.hidden_size)
        self.llm2vae_connector = nn.Linear(self.hidden_size, self.patch_latent_dim)

        vit_pos_embed = self.build_2d_sincos_position_embedding(
            size=config.vit_max_num_patch_per_side, embed_dim=config.text_config.hidden_size
        )
        latent_pos_embed = self.build_2d_sincos_position_embedding(
            size=config.max_latent_size,
            embed_dim=config.text_config.hidden_size,
        )

        self.vit_pos_embed = nn.Parameter(vit_pos_embed, requires_grad=False)
        self.latent_pos_embed = nn.Parameter(latent_pos_embed, requires_grad=False)

    @staticmethod
    def build_2d_sincos_position_embedding(
        size, embed_dim=256, temperature=10000.0, device="cpu", dtype=torch.float32
    ):
        grid_w = torch.arange(torch_int(size), device=device).to(dtype)
        grid_h = torch.arange(torch_int(size), device=device).to(dtype)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        if embed_dim % 4 != 0:
            raise ValueError("Embed dimension must be divisible by 4 for 2D sin-cos position embedding")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, device=device).to(dtype) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos()], dim=1)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs,
    ):
        position_ids = get_flattened_position_ids_extrapolate(
            pixel_values.size(-2),
            pixel_values.size(-1),
            self.config.vision_config.patch_size,
            max_num_patches_per_side=self.config.vit_max_num_patch_per_side,
        )
        pixel_values = patchify(pixel_values, self.config.vision_config.patch_size)

        image_embeddings = self.vision_tower(pixel_values).last_hidden_state
        image_embeddings = self.vision_connecter(image_embeddings)
        # rn pso_embed is of sahpe [num_pathces,x], make it compatible with batch
        pos_embed = self.vit_pos_embed[position_ids]
        image_embeddings = image_embeddings + pos_embed
        return image_embeddings

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_sizes: torch.Tensor = None,
        **kwargs,
    ):
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
            )

        # outputs = self.language_model(
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     use_cache=use_cache,
        #     cache_position=cache_position,
        #     **kwargs,
        # )
        return image_features


@auto_docstring
class BagelForConditionalGeneration(BagelPreTrainedModel):
    def __init__(self, config: BagelConfig):
        super().__init__(config)
        self.model = BagelModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # Initialize the model.
        # self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):
        return self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            **flash_attn_kwargs,
        )


__all__ = [
    "BagelVQVAEConfig",
    "BagelConfig",
    "BagelModel",
    "BagelTextModel",
    "BagelVQVAE",
    "BagelForConditionalGeneration",
]
