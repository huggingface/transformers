# Copyright 2025 Deepseek AI and The HuggingFace Team. All rights reserved.
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

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...generation import ClassifierFreeGuidanceLogitsProcessor, GenerationMixin, GenerationMode, LogitsProcessorList
from ...generation.utils import GenerateDecoderOnlyOutput
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_vision_available,
    logging,
    torch_compilable_check,
)
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..blip_2.modeling_blip_2 import Blip2VisionModel
from ..chameleon.configuration_chameleon import ChameleonVQVAEConfig
from ..chameleon.modeling_chameleon import (
    ChameleonVQVAE,
    ChameleonVQVAEEncoderAttnBlock,
    ChameleonVQVAEEncoderConvDownsample,
    ChameleonVQVAEEncoderResnetBlock,
    ChameleonVQVAEVectorQuantizer,
)
from ..idefics.modeling_idefics import IdeficsBaseModelOutputWithPast, IdeficsCausalLMOutputWithPast
from ..llama.modeling_llama import eager_attention_forward
from ..siglip.configuration_siglip import SiglipVisionConfig
from ..siglip.modeling_siglip import SiglipEncoder, SiglipEncoderLayer, SiglipVisionEmbeddings


if is_vision_available():
    pass

logger = logging.get_logger(__name__)

# General docstring


@auto_docstring(checkpoint="deepseek-community/Janus-Pro-1B")
@strict
class JanusVisionConfig(SiglipVisionConfig):
    r"""
    projection_dropout (`float`, *optional*, defaults to 0.0):
        Dropout probability for the projection layer.
    num_image_tokens (`int`, *optional*, defaults to 576):
        Number of image tokens.
    """

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    image_size: int | list[int] | tuple[int, int] = 384
    hidden_act: str = "gelu"
    mlp_ratio: float | int = 4.0
    attention_bias: bool = True
    hidden_dropout_rate: float | int = 0.0
    projection_dim: int = 2048
    projection_dropout: float | int = 0.0
    use_qk_norm: bool = False
    initializer_range: float = 0.02
    depth: int = 2
    num_image_tokens: int = 576
    intermediate_size = AttributeError()


@auto_docstring(checkpoint="deepseek-community/Janus-Pro-1B")
@strict
class JanusVQVAEConfig(ChameleonVQVAEConfig):
    r"""
    base_channels (`int`, *optional*, defaults to 128):
        Base channel count.
    channel_multiplier (`list[int]`, *optional*, defaults to `[1, 1, 2, 2, 4]`):
        Channel multipliers for each resolution.
    num_res_blocks (`int`, *optional*, defaults to 2):
        Number of residual blocks.
    num_patches (`int`, *optional*, defaults to 32):
        Num of patches the input images can be divided into.
    out_channels (`int`, *optional*, defaults to 3):
        Number of out channels.
    image_token_embed_dim (`int`, *optional*, defaults to 2048):
        Dimension of image embeddings. It should be same as the dimensionality of text embeddings.
    """

    embed_dim: int = 8
    num_embeddings: int = 16384
    double_latent: bool = False
    latent_channels: int = 256
    num_patches: int = 32
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 128
    channel_multiplier: list[int] | tuple[int, ...] = (1, 1, 2, 2, 4)
    num_res_blocks: int = 2
    dropout: float | int = 0.0
    initializer_range: float = 0.02
    projection_dim: int = 2048
    num_hidden_layers: int = 2
    hidden_act: str = "gelu"
    image_token_embed_dim: int = 2048

    resolution = AttributeError()
    attn_resolutions = AttributeError()
    attn_type = AttributeError()


@auto_docstring(checkpoint="deepseek-community/Janus-Pro-1B")
@strict
class JanusConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import JanusForConditionalGeneration, JanusConfig, JanusVisionConfig, JanusVQVAEConfig, LlamaConfig

    >>> # Initializing a Janus vision config
    >>> vision_config = JanusVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a VQ config
    >>> vq_config = JanusVQVAEConfig()

    >>> # Initializing a Janus Pro 1B style configuration
    >>> configuration = JanusConfig(vision_config=vision_config, text_config=text_config, vq_config=vq_config)

    >>> # Initializing a model from the Janus Pro 1B style configuration
    >>> model = JanusForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "janus"
    sub_configs = {
        "text_config": AutoConfig,
        "vision_config": JanusVisionConfig,
        "vq_config": JanusVQVAEConfig,
    }

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    vq_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 100581
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "llama")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            logger.info("`text_config` is None. Initializing with default values")
            self.text_config = CONFIG_MAPPING["llama"]()

        if self.vision_config is None:
            logger.info("`vision_config` is None. Initializing with default JanusVisionConfig values")
            self.vision_config = JanusVisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = JanusVisionConfig(**self.vision_config)

        if self.vq_config is None:
            logger.info("`vq_config` is None. Initializing with default JanusVQVAEConfig values")
            self.vq_config = JanusVQVAEConfig()
        elif isinstance(self.vq_config, dict):
            self.vq_config = JanusVQVAEConfig(**self.vq_config)

        # This dimension is required when decoding discrete image tokens to continuous input.
        self.vq_config.num_patches = self.vision_config.image_size // self.vision_config.patch_size
        super().__post_init__(**kwargs)


@auto_docstring
class JanusPreTrainedModel(PreTrainedModel):
    config: JanusConfig
    base_model_prefix = "model"
    input_modalities = ("image", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer", "JanusVisionEncoderLayer"]
    _skip_keys_device_placement = ["past_key_values", "causal_mask"]
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, JanusVisionEmbeddings):
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Janus VQ-VAE mode model outputs.
    """
)
class JanusVQVAEOutput(ModelOutput):
    r"""
    decoded_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
        Reconstructed pixel values after encoding and decoding the input.
    embedding_loss (`torch.FloatTensor`):
        Embedding loss.
    """

    decoded_pixel_values: torch.FloatTensor | None = None
    embedding_loss: torch.FloatTensor | None = None


class JanusBaseModelOutputWithPast(IdeficsBaseModelOutputWithPast):
    pass


class JanusCausalLMOutputWithPast(IdeficsCausalLMOutputWithPast):
    pass


class JanusVisionEmbeddings(SiglipVisionEmbeddings):
    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        if interpolate_pos_encoding:
            pos_embeds = self.interpolate_pos_encoding(embeddings, height, width)
        else:
            pos_embeds = self.position_embedding(self.position_ids)

        embeddings = embeddings + pos_embeds

        return embeddings


class JanusVisionAttention(nn.Module):
    """Attention Class for Janus Vision Encoder"""

    def __init__(self, config: JanusVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        proj_dropout = config.projection_dropout
        qk_norm = config.use_qk_norm
        self.is_causal = False

        # Janus has no MHA, hence for `eager_attention_forward` call setting `num_key_value_groups` to 1.
        self.num_key_value_groups = 1

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.projection_layer = nn.Linear(self.embed_dim, self.embed_dim)
        self.projection_dropout = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

        self.q_norm = nn.LayerNorm(self.embed_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.embed_dim) if qk_norm else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(-1, self.num_heads, self.head_dim)
        query_states = self.q_norm(query_states)

        key_states = key_states.reshape(-1, self.num_heads, self.head_dim)
        key_states = self.k_norm(key_states)

        query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scale,
            is_causal=self.is_causal,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        output = self.projection_layer(attn_output)
        output = self.projection_dropout(output)
        return output, attn_weights


class JanusVisionMLP(nn.Module):
    def __init__(self, config: JanusVisionConfig):
        super().__init__()
        self.config = config
        self.intermediate_size = int(config.hidden_size * config.mlp_ratio)
        self.activation_fn = ACT2FN[config.hidden_act]  # Gelu act
        self.fc1 = nn.Linear(config.hidden_size, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_rate)
        self.dropout2 = nn.Dropout(config.hidden_dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.dropout1(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        return hidden_states


class JanusVisionEncoderLayer(SiglipEncoderLayer):
    def __init__(self, config: JanusVisionConfig):
        super().__init__(config)
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = JanusVisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = JanusVisionMLP(config)


class JanusVisionEncoder(SiglipEncoder):
    def __init__(self, config: JanusVisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([JanusVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])


class JanusVisionModel(Blip2VisionModel):
    _can_record_outputs = {
        "hidden_states": JanusVisionEncoderLayer,
        "attentions": JanusVisionAttention,
    }

    def __init__(self, config: JanusVisionConfig):
        super().__init__(config)
        self.encoder = JanusVisionEncoder(config)

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )


class JanusVisionAlignerMLP(nn.Module):
    def __init__(self, config: JanusVisionConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.hidden_size, config.projection_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(config.projection_dim, config.projection_dim) for _ in range(1, config.depth)]
        )
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        for layer in self.hidden_layers:
            hidden_states = self.activation_fn(hidden_states)
            hidden_states = layer(hidden_states)
        return hidden_states


class JanusVQVAEVectorQuantizer(ChameleonVQVAEVectorQuantizer):
    def __init__(self, config: JanusVQVAEConfig):
        super().__init__(config)
        self.quant_state_dims = [config.num_patches] * 2

    def get_codebook_entry(self, image_tokens: torch.LongTensor) -> torch.FloatTensor:
        batch_size = image_tokens.shape[0]
        emb_dim: int = self.embedding.weight.shape[-1]

        # get quantized latent vectors
        hidden_state_quant = self.embedding(image_tokens)
        # l2 normalization on the last dimension
        hidden_state_quant = F.normalize(hidden_state_quant, p=2, dim=-1)

        # reshape back to match original input shape
        hidden_state_quant = hidden_state_quant.view((batch_size, *self.quant_state_dims, emb_dim))
        hidden_state_quant = hidden_state_quant.permute(0, 3, 1, 2).contiguous()

        return hidden_state_quant


class JanusVQVAEResnetBlock(ChameleonVQVAEEncoderResnetBlock):
    pass


class JanusVQVAEAttnBlock(ChameleonVQVAEEncoderAttnBlock):
    pass


class JanusVQVAEConvDownsample(ChameleonVQVAEEncoderConvDownsample):
    pass


class JanusVQVAEConvUpsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states):
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.conv(hidden_states)
        return hidden_states


class JanusVQVAEMidBlock(nn.Module):
    def __init__(self, config: JanusVQVAEConfig, channels: int):
        super().__init__()
        self.block_1 = JanusVQVAEResnetBlock(
            config=config,
            in_channels=channels,
            out_channels=channels,
        )
        self.attn_1 = JanusVQVAEAttnBlock(channels)
        self.block_2 = JanusVQVAEResnetBlock(
            config=config,
            in_channels=channels,
            out_channels=channels,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.block_1(hidden_states)
        hidden_states = self.attn_1(hidden_states)
        hidden_states = self.block_2(hidden_states)
        return hidden_states


class JanusVQVAEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks
        base_channels = config.base_channels
        in_channels = config.in_channels
        double_latent = config.double_latent
        latent_channels = config.latent_channels
        channel_multiplier = config.channel_multiplier

        self.conv_in = torch.nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

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
                    JanusVQVAEResnetBlock(
                        config=config,
                        in_channels=block_in,
                        out_channels=block_out,
                    )
                )
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn.append(JanusVQVAEAttnBlock(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = JanusVQVAEConvDownsample(block_in)
            self.down.append(down)

        self.mid = JanusVQVAEMidBlock(config, block_in)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * latent_channels if double_latent else latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, pixel_values: torch.LongTensor):
        # downsampling
        hidden_states = [self.conv_in(pixel_values)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                hidden_state = self.down[i_level].block[i_block](
                    hidden_states[-1],
                )
                if len(self.down[i_level].attn) > 0:
                    hidden_state = self.down[i_level].attn[i_block](hidden_state)
                hidden_states.append(hidden_state)
            if i_level != self.num_resolutions - 1:
                hidden_states.append(self.down[i_level].downsample(hidden_states[-1]))

        # middle
        last_hidden_state = hidden_states[-1]
        last_hidden_state = self.mid(last_hidden_state)

        # end
        last_hidden_state = self.norm_out(last_hidden_state)
        last_hidden_state *= torch.sigmoid(last_hidden_state)
        last_hidden_state = self.conv_out(last_hidden_state)
        return last_hidden_state


class JanusVQVAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks
        base_channels = config.base_channels
        latent_channels = config.latent_channels
        out_channels = config.out_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = base_channels * config.channel_multiplier[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = torch.nn.Conv2d(latent_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = JanusVQVAEMidBlock(config, block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = base_channels * config.channel_multiplier[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    JanusVQVAEResnetBlock(
                        config=config,
                        in_channels=block_in,
                        out_channels=block_out,
                    )
                )
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn.append(JanusVQVAEAttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = JanusVQVAEConvUpsample(block_in)
            self.up.append(up)

        # end
        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = torch.nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_state: torch.FloatTensor) -> torch.FloatTensor:
        hidden_state = self.conv_in(hidden_state)

        # middle
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


class JanusVQVAE(ChameleonVQVAE):
    _no_split_modules = [
        "JanusVQVAEAttnBlock",
        "JanusVQVAEResnetBlock",
        "JanusVQVAEVectorQuantizer",
    ]
    _can_record_outputs = {
        "hidden_states": JanusVQVAEResnetBlock,
        "attentions": JanusVQVAEAttnBlock,
    }
    main_input_name = "pixel_values"

    def __init__(self, config: JanusVQVAEConfig):
        super().__init__(config)
        self.decoder = JanusVQVAEDecoder(config)
        self.gradient_checkpointing = False

        # Initialize the VQVAE model.
        self.post_init()

    def decode(self, image_tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Decodes quantized token IDs into pixel values.
        Args:
            image_tokens (torch.LongTensor): Batch of token IDs.
        Returns:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                Pixel values decoded from the token IDs.
        """
        if image_tokens.shape[1] != self.quantize.quant_state_dims[0] * self.quantize.quant_state_dims[1]:
            raise ValueError(
                f"Expected `image_tokens` to have shape `(batch_size, {self.quantize.quant_state_dims[0] * self.quantize.quant_state_dims[1]})`, "
                f"but got shape `{image_tokens.shape}`."
            )
        codebook_entry = self.quantize.get_codebook_entry(image_tokens)
        hidden_states = self.post_quant_conv(codebook_entry)
        pixel_values = self.decoder(hidden_states)
        return pixel_values

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        batch_size = pixel_values.shape[0]
        encode_outputs = self.encode(pixel_values, return_dict=True, **kwargs)
        decoded_pixel_values = self.decode(encode_outputs.image_tokens.view(batch_size, -1))

        return JanusVQVAEOutput(decoded_pixel_values, encode_outputs.embedding_loss)


class JanusVQVAEAlignerMLP(nn.Module):
    def __init__(self, config: JanusVQVAEConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.embed_dim, config.projection_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(config.projection_dim, config.projection_dim) for _ in range(1, config.num_hidden_layers)]
        )
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        for layer in self.hidden_layers:
            hidden_states = self.activation_fn(hidden_states)
            hidden_states = layer(hidden_states)
        return hidden_states


class JanusVQVAEHead(nn.Module):
    """Head used for sampling tokens in image generation, replacing the usual lm head."""

    def __init__(self, config: JanusVQVAEConfig):
        super().__init__()
        self.proj_out = nn.Linear(config.image_token_embed_dim, config.projection_dim)
        self.activation_fn = ACT2FN[config.hidden_act]
        self.vision_head = nn.Linear(config.projection_dim, config.num_embeddings)

    def forward(self, hidden_states: torch.Tensor) -> torch.tensor:
        hidden_states = self.proj_out(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.vision_head(hidden_states)
        return hidden_states


@auto_docstring(
    custom_intro="""
    The Janus model which consists of a siglip vision backbone, a Llama language model and a VQ model.
    """
)
class JanusModel(JanusPreTrainedModel):
    def __init__(self, config: JanusConfig):
        super().__init__(config)
        self.config = config
        # This is necessary for backward compatibility, see SiglipModel initialization
        self.vision_model = JanusVisionModel._from_config(config.vision_config)
        self.aligner = JanusVisionAlignerMLP(self.vision_model.config)

        self.vqmodel = JanusVQVAE._from_config(config.vq_config)

        # Below generation_* modules are used for Image generation.
        # Embeddings used for image generation, instead of Janus vision embeddings.
        self.generation_embeddings = nn.Embedding(self.vqmodel.config.num_embeddings, self.vqmodel.config.embed_dim)
        self.generation_aligner = JanusVQVAEAlignerMLP(self.vqmodel.config)
        self.generation_head = JanusVQVAEHead(self.vqmodel.config)

        self.language_model = AutoModel.from_config(config=config.text_config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing.
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        vision_outputs = self.vision_model(pixel_values, return_dict=True, **kwargs)
        vision_outputs.pooler_output = self.aligner(vision_outputs.last_hidden_state)

        return vision_outputs

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        n_image_features = image_features.shape[0] * image_features.shape[1]
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        torch_compilable_check(
            inputs_embeds[special_image_mask].numel() == image_features.numel(),
            f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {n_image_features}",
        )
        return special_image_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> JanusBaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, return_dict=True).pooler_output
            image_features = image_embeds.reshape(-1, inputs_embeds.shape[-1])
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_attention_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_attention_mask, image_features)

        lm_output = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        return JanusBaseModelOutputWithPast(
            last_hidden_state=lm_output.last_hidden_state,
            past_key_values=lm_output.past_key_values,
            hidden_states=lm_output.hidden_states,
            attentions=lm_output.attentions,
            image_hidden_states=image_embeds if pixel_values is not None else None,
        )


class JanusForConditionalGeneration(JanusPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    output_modalities = ("image", "text")
    _can_compile_fullgraph = True

    def __init__(self, config: JanusConfig):
        super().__init__(config)
        self.config = config
        self.model = JanusModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # Initialize weights and apply final processing.
        self.post_init()

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    def prepare_embeddings_for_image_generation(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_state = self.model.generation_embeddings(inputs)
        hidden_state = self.model.generation_aligner(hidden_state)
        return hidden_state

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> JanusCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return JanusCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        pixel_values=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        logits_to_keep=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- extra custom processing

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # Pixel values are used only in the first iteration if available
        # In subsequent iterations, they are already merged with text and cached
        # NOTE: first iteration doesn't have to be prefill, it can be the first
        # iteration with a question and cached system prompt (continue generate from cache)
        if is_first_iteration or not kwargs.get("use_cache", True):
            model_inputs["pixel_values"] = pixel_values

        return model_inputs

    def decode_image_tokens(self, image_tokens: torch.Tensor):
        """
        Decodes generated image tokens from language model to continuous pixel values
        with VQGAN module via upsampling.
        Args:
            image_tokens (`torch.LongTensor` of shape `(batch_size, num_of_tokens)`):
                The tensors corresponding to the input images.
        """
        decoded_image = self.model.vqmodel.decode(image_tokens)
        decoded_image = decoded_image.permute(0, 2, 3, 1)
        return decoded_image

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        logits_processor: LogitsProcessorList | None = None,
        **kwargs,
    ):
        # 1. Handle generation config and model kwargs
        # Pop generation_mode first since it's specific to Janus
        generation_mode = kwargs.pop("generation_mode", "text")
        generation_config, model_kwargs = self._prepare_generation_config(
            kwargs.pop("generation_config", None), **kwargs
        )

        # Default to "text" generation if mode isn't provided
        if generation_mode == "text":
            # Set guidance_scale=None to prevent running UnbatchedCFG processor.
            return super().generate(
                inputs=inputs,
                attention_mask=attention_mask,
                generation_config=generation_config,
                guidance_scale=None,
                **model_kwargs,
            )

        # Validate generation mode
        if generation_config.get_generation_mode() not in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            raise ValueError(
                "Got incompatible mode for Image Generation, should be one of greedy or sampling. "
                "Ensure that beam search is de-activated by setting `num_beams=1`."
            )

        # Validate the configuration and model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Initialize logit processors
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()

        # Set `use_cache=True` as we will be using input embeds for generation.
        model_kwargs["use_cache"] = True

        if generation_config.guidance_scale is None:
            logger.warning("`guidance_scale` is required for CFG but not provided. Setting to default value of 5.")
            generation_config.guidance_scale = 5
        model_kwargs["guidance_scale"] = generation_config.guidance_scale

        # 3. Prepare model inputs
        input_ids, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        dtype, device = input_ids.dtype, input_ids.device

        if len(input_ids.shape) != 2:
            raise ValueError(
                f"Expected input ids of shape (batch_size, seq_len), but got {input_ids.shape}"
                "Passing `inputs embeds` is not supported currently."
            )

        # Prepare special tokens which will be used generate internally.
        kwargs_has_attention_mask = attention_mask is not None
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=input_ids.device)

        # 4. Add CFG processor along with user passed logit processor.
        if generation_config.guidance_scale and generation_config.guidance_scale > 1:
            logits_processor.append(ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale))
            generation_config.guidance_scale = None  # Reset to prevent processor duplication.

        # 5. Prepare logits processor
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids.shape[1],
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
            device=device,
        )

        # 6. Expand inputs for multiple image generations per prompt.
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            expand_size=generation_config.num_return_sequences,
            **model_kwargs,
        )

        # 7. Prepare input and model caches
        num_image_tokens = self.model.vision_model.config.num_image_tokens
        batch_size, seq_len = input_ids.shape

        input_tokens = input_ids.repeat(2, 1)  # Double batch size for conditional/unconditional logits
        attention_mask = model_kwargs.pop("attention_mask", None)
        attention_mask = attention_mask.repeat(2, 1)
        model_kwargs["attention_mask"] = attention_mask

        # Mask all the tokens that are neither BOS nor BOI with pad token in the unconditional logits.
        mask = (input_tokens[batch_size:, :] != generation_config.bos_token_id) & (
            input_tokens[batch_size:, :] != generation_config.generation_kwargs["boi_token_id"]
        )
        input_tokens[batch_size:, :].masked_fill_(mask, generation_config.pad_token_id)

        inputs_embeds = self.get_input_embeddings()(input_tokens)

        if model_kwargs.get("past_key_values", None) is None:
            # Prepare cache if not provided.
            model_kwargs["past_key_values"] = self._prepare_static_cache(
                cache_implementation=generation_config.cache_implementation or "static",
                # batch_size should account for both conditional/unconditional input; hence multiplied by 2.
                batch_size=batch_size * 2,
                # we should have at least a cache len of seq_len + num_image_tokens.
                max_cache_len=max(generation_config.max_length, num_image_tokens + seq_len),
                model_kwargs=model_kwargs,
            )

        # Placeholder for generated tokens.
        generated_tokens = torch.zeros((batch_size, num_image_tokens), dtype=dtype, device=device)

        # 8. init attention / hidden states / scores tuples
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate

        raw_scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None

        for i in range(num_image_tokens):
            # Set `is_first_iteration=True` to force using `inputs_embeds` instead of `input_ids`.
            # Without this, `prepare_inputs_for_generation` would use `input_ids` (the full prompt)
            # instead of our prepared `inputs_embeds` (1 new token).
            # This causes CUDA error: device-side assert triggered, seen around the call to ` self.self_attn`.
            # Set this to `True` is also necessary to match the expected output, see the more detailed comment
            # https://github.com/huggingface/transformers/pull/45044#discussion_r3020805374.
            model_inputs = self.prepare_inputs_for_generation(
                inputs_embeds=inputs_embeds, input_ids=input_tokens, is_first_iteration=True, **model_kwargs
            )
            if "attention_mask" in model_inputs:
                model_inputs["attention_mask"] = model_inputs["attention_mask"].to(inputs_embeds.device)

            outputs = self.model.language_model(
                **model_inputs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            # Update model_kwargs like attention_mask for next generation.
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
            hidden_state = outputs.last_hidden_state[:, -1, :].clone()

            # Generate scores using the generation head (Not using above defined LM Head)
            scores = self.model.generation_head(hidden_state)
            next_token_scores = logits_processor(input_ids, scores)

            # Sample next token.
            if generation_config.do_sample:
                probs = torch.softmax(next_token_scores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = torch.argmax(next_token_scores, dim=-1)

            generated_tokens[:, i] = next_token

            # Prepare embeddings for the next step.
            next_token = torch.cat([next_token, next_token])
            next_token = next_token.unsqueeze(-1)

            inputs_embeds = self.prepare_embeddings_for_image_generation(next_token)

        if return_dict_in_generate:
            if output_scores:
                raw_scores += (scores,)
            if output_logits:
                raw_logits += (hidden_state.float(),)
            if output_attentions:
                decoder_attentions += outputs.attentions
            if output_hidden_states:
                decoder_hidden_states += outputs.hidden_states

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=generated_tokens,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=outputs.past_key_values,
            )
        else:
            return generated_tokens


__all__ = [
    "JanusPreTrainedModel",
    "JanusForConditionalGeneration",
    "JanusModel",
    "JanusVQVAE",
    "JanusVisionModel",
    "JanusVQVAEConfig",
    "JanusVisionConfig",
    "JanusConfig",
]
