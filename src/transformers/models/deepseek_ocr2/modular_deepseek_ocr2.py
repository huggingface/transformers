# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.output_capturing import capture_outputs
from ..deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from ..deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2Model,
)
from ..llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
from ..llava_next.modeling_llava_next import (
    LlavaNextCausalLMOutputWithPast,
    LlavaNextForConditionalGeneration,
    LlavaNextModel,
    LlavaNextModelOutputWithPast,
    LlavaNextPreTrainedModel,
)
from ..qwen2.configuration_qwen2 import Qwen2Config
from ..qwen2.modeling_qwen2 import Qwen2Attention, Qwen2DecoderLayer, Qwen2Model
from ..sam.configuration_sam import SamVisionConfig
from ..sam.modeling_sam import (
    SamPatchEmbeddings,
    SamVisionAttention,
    SamVisionEncoder,
    SamVisionLayer,
    SamVisionNeck,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="thisisiron/DeepSeek-OCR-2-hf")
@strict
class DeepseekOcr2SamVisionConfig(SamVisionConfig):
    r"""
    output_channels (`int`, *optional*, defaults to 256):
        The number of output channels in the SAM neck.
    window_size (`int`, *optional*, defaults to 14):
        Window size for windowed attention layers.
    global_attn_indexes (`list[int]`, *optional*, defaults to `[2, 5, 8, 11]`):
        Indices of encoder layers that use global (non-windowed) attention.
    mlp_dim (`int`, *optional*):
        Dimensionality of the MLP layer in each vision encoder block. Defaults to `hidden_size * mlp_ratio`.
    downsample_channels (`list[int]`, *optional*):
        The channel dimensions for the multi-scale downsampling neck layers.
    """

    base_config_key = "sam_config"

    # Remove unused attribute inherited from SamVisionConfig
    num_pos_feats = AttributeError()

    downsample_channels: list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.downsample_channels is None:
            self.downsample_channels = [512, 896]
        super().__post_init__(**kwargs)


@auto_docstring
@strict
class DeepseekOcr2VisionConfig(Qwen2Config):
    r"""
    sam_config (`dict` or `PreTrainedConfig`, *optional*):
        Configuration for the SAM ViT-B vision encoder. Defaults to `DeepseekOcr2SamVisionConfig()`.
    """

    base_config_key = "vision_config"
    base_model_tp_plan = {}
    base_model_pp_plan = {}
    sub_configs = {
        "sam_config": DeepseekOcr2SamVisionConfig,
    }

    sam_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.sam_config is None:
            self.sam_config = DeepseekOcr2SamVisionConfig()
        elif isinstance(self.sam_config, dict):
            self.sam_config = DeepseekOcr2SamVisionConfig(**self.sam_config)

        super().__post_init__(**kwargs)


@auto_docstring
@strict
class DeepseekOcr2TextConfig(DeepseekV2Config):
    r"""
    first_k_dense_replace (`int`, *optional*, defaults to 0):
        The number of initial decoder layers that use dense MLP instead of MoE.
    n_group (`int`, *optional*):
        Number of groups for grouped top-k expert routing.
    topk_method (`str`, *optional*, defaults to `"greedy"`):
        Method for selecting top-k experts in MoE layers.
    """

    base_config_key = "text_config"

    # Override DeepseekV2's MLA TP plan with standard MHA projections
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    # Remove unused MLA attributes inherited from DeepseekV2Config
    kv_lora_rank = AttributeError()
    norm_topk_prob = AttributeError()
    q_lora_rank = AttributeError()
    qk_nope_head_dim = AttributeError()
    qk_rope_head_dim = AttributeError()
    v_head_dim = AttributeError()

    def __post_init__(self, **kwargs):
        self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        PreTrainedConfig.__post_init__(self, **kwargs)


@auto_docstring(checkpoint="thisisiron/DeepSeek-OCR-2-hf")
@strict
class DeepseekOcr2Config(PreTrainedConfig):
    r"""
    vision_config (`dict` or `DeepseekOcr2VisionConfig`, *optional*):
        Configuration for the vision encoders (SAM + hybrid encoder). Defaults to `DeepseekOcr2VisionConfig()`.
    projector_input_dim (`int`, *optional*, defaults to 896):
        Input dimensionality of the visual projector.
    projector_n_embed (`int`, *optional*, defaults to 1280):
        Output dimensionality of the visual projector (language model embedding size).
    projector_type (`str`, *optional*, defaults to `"linear"`):
        Type of projector to use. Can be `"linear"` for a single linear layer or `"mlp"` for a two-layer MLP
        with GELU activation.
    """

    model_type = "deepseek_ocr2"
    sub_configs = {
        "vision_config": DeepseekOcr2VisionConfig,
        "text_config": DeepseekOcr2TextConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 128815
    projector_input_dim: int = 896
    projector_n_embed: int = 1280
    projector_type: str = "linear"

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = DeepseekOcr2VisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = DeepseekOcr2VisionConfig(**self.vision_config)

        if self.text_config is None:
            self.text_config = DeepseekOcr2TextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = DeepseekOcr2TextConfig(**self.text_config)

        super().__post_init__(**kwargs)


@dataclass
class DeepseekOcr2ModelOutputWithPooling(BaseModelOutputWithPooling):
    local_last_hidden_state: torch.FloatTensor | None = None
    local_hidden_states: torch.FloatTensor | None = None
    local_attentions: torch.FloatTensor | None = None


class DeepseekOcr2ModelOutputWithPast(LlavaNextModelOutputWithPast):
    pass


class DeepseekOcr2CausalLMOutputWithPast(LlavaNextCausalLMOutputWithPast):
    pass


class DeepseekOcr2PreTrainedModel(LlavaNextPreTrainedModel):
    _no_split_modules = [
        "DeepseekOcr2SamVisionLayer",
        "DeepseekOcr2SamVisionEncoder",
        "DeepseekOcr2VisionModel",
        "DeepseekOcr2TextDecoderLayer",
    ]
    _can_compile_fullgraph = False
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = True

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, DeepseekOcr2SamVisionAttention):
            if module.use_rel_pos:
                init.zeros_(module.rel_pos_h)
                init.zeros_(module.rel_pos_w)
        elif isinstance(module, DeepseekOcr2SamVisionEncoder):
            if module.pos_embed is not None:
                init.zeros_(module.pos_embed)
        elif isinstance(module, DeepseekOcr2Model):
            embed_std = 1 / math.sqrt(self.config.projector_n_embed)
            init.normal_(module.view_separator, mean=0.0, std=embed_std)


class DeepseekOcr2SamVisionAttention(SamVisionAttention):
    pass


class DeepseekOcr2SamVisionLayer(SamVisionLayer):
    pass


class DeepseekOcr2SamVisionNeck(SamVisionNeck):
    pass


class DeepseekOcr2SamPatchEmbeddings(SamPatchEmbeddings):
    def forward(self, pixel_values):
        embeddings = self.projection(pixel_values).permute(0, 2, 3, 1)
        return embeddings


class DeepseekOcr2SamVisionProj(nn.Module):
    """Neck and multi-scale downsampling for SAM ViT-B output."""

    def __init__(self, config: DeepseekOcr2SamVisionConfig):
        super().__init__()
        self.conv1 = nn.Conv2d(
            config.output_channels,
            config.downsample_channels[0],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            config.downsample_channels[0],
            config.downsample_channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        return hidden_states


class DeepseekOcr2SamVisionEncoder(SamVisionEncoder, DeepseekOcr2PreTrainedModel):
    def __init__(self, config: DeepseekOcr2SamVisionConfig):
        super().__init__(config)
        self.proj = DeepseekOcr2SamVisionProj(config)

    def _interpolate_pos_encoding(self, pos_embed: torch.Tensor, target_size: int) -> torch.Tensor:
        src_size = pos_embed.shape[1]
        if src_size == target_size:
            return pos_embed
        pos_embed = pos_embed.permute(0, 3, 1, 2).float()
        pos_embed = torch.nn.functional.interpolate(
            pos_embed,
            size=(target_size, target_size),
            mode="bicubic",
            align_corners=False,
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    @capture_outputs(tie_last_hidden_states=False)
    def forward(self, pixel_values: torch.FloatTensor, **kwargs) -> BaseModelOutput:
        hidden_states = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            hidden_states = hidden_states + self._interpolate_pos_encoding(self.pos_embed, hidden_states.shape[1]).to(
                hidden_states.dtype
            )

        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)

        hidden_states = self.neck(hidden_states)
        hidden_states = self.proj(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class DeepseekOcr2VisionAttention(Qwen2Attention):
    pass


class DeepseekOcr2VisionDecoderLayer(Qwen2DecoderLayer):
    pass


@auto_docstring(custom_intro="Qwen2 backbone used as vision encoder inside DeepEncoderV2.")
class DeepseekOcr2VisionEncoder(Qwen2Model):
    def __init__(self, config):
        super().__init__(config)
        del self.embed_tokens

    @capture_outputs
    @auto_docstring
    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class DeepseekOcr2Projector(nn.Module):
    def __init__(self, config: DeepseekOcr2Config):
        super().__init__()
        if config.projector_type == "linear":
            self.proj = nn.Linear(config.projector_input_dim, config.projector_n_embed)
        else:
            self.proj = nn.Sequential(
                nn.Linear(config.projector_input_dim, config.projector_n_embed),
                nn.GELU(),
                nn.Linear(config.projector_n_embed, config.projector_n_embed),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def token_type_ids_mask_function(token_type_ids: torch.Tensor):
    """
    Creates an or_mask_function for `create_causal_mask` that allows
    bidirectional attention between image tokens (type_id=0).

    Args:
        token_type_ids: `(batch_size, seq_len)` tensor where 0=image, 1=query.

    Returns:
        A mask function compatible with `create_causal_mask(or_mask_function=...)`.
    """
    is_image = token_type_ids == 0

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return is_image[batch_idx, q_idx] & is_image[batch_idx, kv_idx]

    return inner_mask


class DeepseekOcr2VisionModel(DeepseekOcr2PreTrainedModel):
    """Vision pipeline: SAM ViT-B (with neck) then DeepEncoder V2."""

    def __init__(self, config: DeepseekOcr2VisionConfig):
        super().__init__(config)
        self.sam_encoder = DeepseekOcr2SamVisionEncoder(config.sam_config)
        self.vision_encoder = DeepseekOcr2VisionEncoder(config)

        # Resolution-specific learnable queries
        self.query_768 = nn.Embedding(144, config.hidden_size)  # 12x12 for 768px
        self.query_1024 = nn.Embedding(256, config.hidden_size)  # 16x16 for 1024px
        self.post_init()

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> BaseModelOutput:
        """
        Args:
            pixel_values: [B, 3, H, W] image tensor
        Returns:
            BaseModelOutput with query features as last_hidden_state
        """
        sam_out = self.sam_encoder(pixel_values, return_dict=True).last_hidden_state
        x = sam_out.flatten(2).transpose(1, 2)
        bsz, n_patches, _ = x.shape

        queries = self.query_768.weight if n_patches <= 144 else self.query_1024.weight
        n_queries = queries.shape[0]

        queries = queries.unsqueeze(0).expand(bsz, -1, -1)
        combined = torch.cat([x, queries], dim=1)

        token_type_ids = torch.cat(
            [
                torch.zeros(bsz, n_patches, dtype=torch.long, device=x.device),
                torch.ones(bsz, n_queries, dtype=torch.long, device=x.device),
            ],
            dim=1,
        )

        hybrid_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=combined,
            attention_mask=None,
            past_key_values=None,
            or_mask_function=token_type_ids_mask_function(token_type_ids),
        )

        encoder_outputs = self.vision_encoder(
            inputs_embeds=combined,
            attention_mask=hybrid_mask,
            **kwargs,
        )

        query_features = encoder_outputs.last_hidden_state[:, n_patches:, :]

        return BaseModelOutput(
            last_hidden_state=query_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class DeepseekOcr2TextRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class DeepseekOcr2TextAttention(LlamaAttention):
    pass


class DeepseekOcr2TextDecoderLayer(DeepseekV2DecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = DeepseekOcr2TextAttention(config=config, layer_idx=layer_idx)


class DeepseekOcr2TextModel(DeepseekV2Model):
    def __init__(self, config: DeepseekOcr2TextConfig):
        super().__init__(config)
        # Use (cos/sin) RoPE instead of complex RoPE to match LlamaAttention (MHA)
        self.rotary_emb = DeepseekOcr2TextRotaryEmbedding(config=config)


class DeepseekOcr2Model(LlavaNextModel):
    def __init__(self, config: DeepseekOcr2Config):
        super().__init__(config)
        del self.image_newline

        self.vision_tower = DeepseekOcr2VisionModel(config.vision_config)
        self.multi_modal_projector = DeepseekOcr2Projector(config)

        # Learnable separator between local and global views
        embed_std = 1.0 / math.sqrt(config.projector_n_embed)
        self.view_separator = nn.Parameter(torch.randn(config.projector_n_embed) * embed_std)

        self.language_model = DeepseekOcr2TextModel(config.text_config)

    def pack_image_features(self, *args, **kwargs):
        raise NotImplementedError("DeepseekOcr2 does not use pack_image_features")

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_values_local: torch.FloatTensor | None = None,
        num_local_patches: list[int] | torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        pixel_values_local (`torch.FloatTensor` of shape `(total_patches, 3, height, width)`, *optional*):
            All local patches flattened across the batch, or `None` if no local views.
        num_local_patches (`list[int]` or `torch.Tensor`, *optional*):
            Number of local patches per image, e.g. `[6, 0, 4]`.
        """
        batch_size = pixel_values.shape[0]

        global_vision_outputs = self.vision_tower(pixel_values, **kwargs)
        global_features = self.multi_modal_projector(global_vision_outputs.last_hidden_state)

        if pixel_values_local is not None:
            local_vision_outputs = self.vision_tower(pixel_values_local, **kwargs)
            all_local_features = self.multi_modal_projector(local_vision_outputs.last_hidden_state)
            per_image_local = torch.split(all_local_features, num_local_patches, dim=0)
        else:
            local_vision_outputs = None
            per_image_local = [None] * batch_size

        all_features = []
        for idx in range(batch_size):
            global_flat = global_features[idx].reshape(-1, global_features.shape[-1])

            if per_image_local[idx] is not None:
                local_flat = per_image_local[idx].reshape(-1, per_image_local[idx].shape[-1])
                all_features.append(torch.cat([local_flat, global_flat, self.view_separator.unsqueeze(0)], dim=0))
            else:
                all_features.append(torch.cat([global_flat, self.view_separator.unsqueeze(0)], dim=0))

        image_features = torch.cat(all_features, dim=0)
        return DeepseekOcr2ModelOutputWithPooling(
            last_hidden_state=global_vision_outputs.last_hidden_state,
            pooler_output=image_features,
            hidden_states=global_vision_outputs.hidden_states,
            attentions=global_vision_outputs.attentions,
            local_last_hidden_state=local_vision_outputs.last_hidden_state
            if local_vision_outputs is not None
            else None,
            local_hidden_states=local_vision_outputs.hidden_states if local_vision_outputs is not None else None,
            local_attentions=local_vision_outputs.attentions if local_vision_outputs is not None else None,
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_local: torch.FloatTensor | None = None,
        num_local_patches: list[int] | torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | DeepseekOcr2ModelOutputWithPast:
        r"""
        pixel_values_local (`torch.FloatTensor`, *optional*):
            Local patch pixel values of shape `(total_patches, 3, H, W)`.
        num_local_patches (`list[int]` or `torch.Tensor`, *optional*):
            Number of local patches per image in the batch.
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None:
            if isinstance(num_local_patches, torch.Tensor):
                num_local_patches = num_local_patches.tolist()
            image_features = self.get_image_features(
                pixel_values, pixel_values_local, num_local_patches, return_dict=True
            ).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

            special_image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        return DeepseekOcr2ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )


@auto_docstring
class DeepseekOcr2ForConditionalGeneration(LlavaNextForConditionalGeneration, GenerationMixin):
    def pack_image_features(self, *args, **kwargs):
        raise NotImplementedError("DeepseekOcr2 does not use pack_image_features")

    @can_return_tuple
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_values_local: torch.FloatTensor | None = None,
        num_local_patches: list[int] | torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, 3, height, width)`):
            The tensors corresponding to the global view input images.
        pixel_values_local (`torch.FloatTensor` of shape `(total_patches, 3, height, width)`, *optional*):
            All local patches flattened across the batch, or `None` if no local views.
        num_local_patches (`list[int]` or `torch.Tensor`, *optional*):
            Number of local patches per image, e.g. `[6, 0, 4]`.
        """
        return self.model.get_image_features(
            pixel_values=pixel_values,
            pixel_values_local=pixel_values_local,
            num_local_patches=num_local_patches,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_local=None,
        num_local_patches=None,
        attention_mask=None,
        logits_to_keep=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if is_first_iteration or not kwargs.get("use_cache", True):
            model_inputs["pixel_values"] = pixel_values
            model_inputs["pixel_values_local"] = pixel_values_local
            model_inputs["num_local_patches"] = num_local_patches

        return model_inputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_local: torch.FloatTensor | None = None,
        num_local_patches: list[int] | torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | DeepseekOcr2CausalLMOutputWithPast:
        r"""
        pixel_values_local (`torch.FloatTensor`, *optional*):
            Local patch pixel values of shape `(total_patches, 3, H, W)`.
        num_local_patches (`list[int]` or `torch.Tensor`, *optional*):
            Number of local patches per image in the batch.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_local=pixel_values_local,
            num_local_patches=num_local_patches,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

        return DeepseekOcr2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = [
    "DeepseekOcr2Config",
    "DeepseekOcr2PreTrainedModel",
    "DeepseekOcr2Model",
    "DeepseekOcr2ForConditionalGeneration",
]
