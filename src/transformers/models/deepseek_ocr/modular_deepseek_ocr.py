# Copyright 2025 Deepseek-AI and the HuggingFace Inc. team. All rights reserved.
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
from typing import Optional, Union

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPooling, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import check_model_inputs
from ..auto import CONFIG_MAPPING, AutoConfig
from ..clip.modeling_clip import (
    CLIPEncoder,
    CLIPEncoderLayer,
    CLIPVisionEmbeddings,
    CLIPVisionModel,
    CLIPVisionTransformer,
)
from ..deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from ..deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2Model,
    DeepseekV2PreTrainedModel,
    DeepseekV2RMSNorm,
)
from ..llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
from ..llava_next.modeling_llava_next import LlavaNextForConditionalGeneration, LlavaNextModel
from ..sam.modeling_sam import SamVisionEncoder, SamVisionNeck


logger = logging.get_logger(__name__)


class DeepseekOcrSamConfig(PreTrainedConfig):
    model_type = "deepseek_ocr_sam_vision"
    base_config_key = "sam_config"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=1024,
        patch_size=16,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        initializer_range=1e-10,
        qkv_bias=True,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        global_attn_indexes=None,
        mlp_ratio=4.0,
        output_channels=256,
        downsample_channels=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.qkv_bias = qkv_bias
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes if global_attn_indexes is not None else [2, 5, 8, 11]
        self.mlp_ratio = mlp_ratio
        self.output_channels = output_channels
        self.downsample_channels = downsample_channels if downsample_channels is not None else [512, 1024]
        self.mlp_dim = int(hidden_size * mlp_ratio)
        self.out_channels = output_channels


class DeepseekOcrCLIPVisionConfig(PreTrainedConfig):
    model_type = "deepseek_ocr_clip_vision"
    base_config_key = "clip_vision_config"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4096,
        projection_dim=768,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=224,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor


class DeepseekOcrProjectorConfig(PreTrainedConfig):
    model_type = "deepseek_ocr_projector"
    base_config_key = "projector_config"

    def __init__(
        self,
        input_dim=2048,
        n_embed=1280,
        projector_type="linear",
        depth=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.n_embed = n_embed
        self.projector_type = projector_type
        self.depth = depth


class DeepseekOcrVisionConfig(PreTrainedConfig):
    model_type = "deepseek_ocr_vision"
    base_config_key = "vision_config"
    sub_configs = {
        "sam_config": DeepseekOcrSamConfig,
        "clip_config": DeepseekOcrCLIPVisionConfig,
    }

    def __init__(self, sam_config=None, clip_config=None, **kwargs):
        super().__init__(**kwargs)

        if sam_config is None:
            self.sam_config = DeepseekOcrSamConfig()
        elif isinstance(sam_config, dict):
            self.sam_config = DeepseekOcrSamConfig(**sam_config)
        else:
            self.sam_config = sam_config

        if clip_config is None:
            self.clip_config = DeepseekOcrCLIPVisionConfig()
        elif isinstance(clip_config, dict):
            self.clip_config = DeepseekOcrCLIPVisionConfig(**clip_config)
        else:
            self.clip_config = clip_config

        # Aggregate commonly accessed vision attributes.
        self.image_size = self.sam_config.image_size
        self.patch_size = self.sam_config.patch_size


class DeepseekOcrTextConfig(DeepseekV2Config):
    pass


class DeepseekOcrConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekOcrForConditionalGeneration`]. It is used to instantiate a
    DeepseekOCR model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `DeepseekV2Config`):
            The config object or dictionary of the text backbone (DeepSeek-V2).
        vision_config (`DeepseekOcrVisionConfig` or `dict`, *optional*):
            The config object or dictionary of the vision encoders (SAM and CLIP).
        projector_config (`DeepseekOcrProjectorConfig` or `dict`, *optional*):
            The config object or dictionary of the projector that maps vision features to text embedding space.
        candidate_resolutions (`list`, *optional*, defaults to `[[1024, 1024]]`):
            List of candidate image resolutions for adaptive image processing.
        global_view_pos (`str`, *optional*, defaults to `"head"`):
            Position of the global view in the image sequence.
        tile_tag (`str`, *optional*, defaults to `"2D"`):
            Tag format for image tiles.
        image_token_index (`int`, *optional*, defaults to 100015):
            The index representing image tokens in the model's token vocabulary.

    Example:

    ```python
    >>> from transformers import DeepseekOcrConfig, DeepseekOcrForConditionalGeneration

    >>> # Initializing a DeepseekOCR configuration
    >>> configuration = DeepseekOcrConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = DeepseekOcrForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_ocr"
    sub_configs = {
        "text_config": AutoConfig,
        "vision_config": DeepseekOcrVisionConfig,
        "projector_config": DeepseekOcrProjectorConfig,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projector_config=None,
        candidate_resolutions=None,
        global_view_pos="head",
        tile_tag="2D",
        image_token_index=100015,
        image_grid_pinpoints=None,
        vision_feature_layer=None,
        vision_feature_select_strategy="default",
        **kwargs,
    ):
        if candidate_resolutions is None:
            candidate_resolutions = [[1024, 1024]]

        self.candidate_resolutions = candidate_resolutions
        self.global_view_pos = global_view_pos
        self.tile_tag = tile_tag
        self.image_token_index = image_token_index
        self.image_token_id = image_token_index
        self.image_grid_pinpoints = image_grid_pinpoints if image_grid_pinpoints is not None else [[1024, 1024]]
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy

        if text_config is None:
            text_config = CONFIG_MAPPING["deepseek_v2"](
                hidden_size=1280,
                intermediate_size=6848,
                num_hidden_layers=12,
                num_attention_heads=10,
                num_key_value_heads=10,
                moe_intermediate_size=896,
                n_routed_experts=64,
                n_shared_experts=2,
                num_experts_per_tok=6,
                first_k_dense_replace=1,
                vocab_size=129280,
                max_position_embeddings=8192,
                use_mla=False,
            )
        elif isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "deepseek_v2")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)

        self.text_config = text_config

        if vision_config is None:
            self.vision_config = DeepseekOcrVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = DeepseekOcrVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        if projector_config is None:
            self.projector_config = DeepseekOcrProjectorConfig()
        elif isinstance(projector_config, dict):
            self.projector_config = DeepseekOcrProjectorConfig(**projector_config)
        else:
            self.projector_config = projector_config

        self.hidden_size = self.text_config.hidden_size
        self.vocab_size = self.text_config.vocab_size

        super().__init__(**kwargs)


class DeepseekOcrPreTrainedModel(PreTrainedModel):
    config_class = DeepseekOcrConfig
    base_model_prefix = "model"


class DeepseekOcrProjector(PreTrainedModel):
    """
    Projector that maps concatenated SAM + CLIP features to language model space.
    """

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.Linear(config.input_dim, config.n_embed)

    def forward(self, x):
        return self.layers(x)


class DeepseekOcrSamVisionNeck(SamVisionNeck):
    def __init__(self, config):
        super().__init__(config)


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Deepseek OCR model outputs with optional image hidden states.
    """
)
class DeepseekOcrModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    image_hidden_states (`torch.FloatTensor`, *optional*):
        Hidden states extracted from the visual encoder and projected into the language embedding space.
    """

    image_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Deepseek OCR causal language model outputs with image hidden states.
    """
)
class DeepseekOcrCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modelling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modelling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        Hidden states produced by the visual encoder after multimodal projection.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


class DeepseekOcrSamVisionEncoder(SamVisionEncoder):
    """
    SAM ViT-B vision encoder with additional neck layers for Deepseek OCR.
    Wraps the SAM vision encoder and adds downsampling convolutions.
    """

    def __init__(self, config):
        super().__init__(config)
        out_channels = config.out_channels
        downsample_channels = config.downsample_channels

        # TODO move hardcoded values to config
        self.net_2 = nn.Conv2d(out_channels, downsample_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(
            downsample_channels[0], downsample_channels[1], kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, pixel_values):
        hidden_states = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        hidden_states = self.neck(hidden_states)
        hidden_states = self.net_2(hidden_states)
        hidden_states = self.net_3(hidden_states)

        return hidden_states


class DeepseekOcrVisionEmbeddings(CLIPVisionEmbeddings):
    def forward(self, pixel_values, patch_embeds=None, interpolate_pos_encoding=False) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape

        if patch_embeds is None:
            patch_embeds = self.patch_embedding(pixel_values)
        if patch_embeds.dim() == 4:
            patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        else:
            patch_embeds = patch_embeds
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embeddings = self.position_embedding(self.position_ids)
        if position_embeddings.shape[1] != embeddings.shape[1]:
            class_pos_embed = position_embeddings[:, :1]
            patch_pos_embed = position_embeddings[:, 1:]
            src_size = int(math.sqrt(patch_pos_embed.shape[1]))
            patch_pos_embed = patch_pos_embed.reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2)
            patch_pos_embed = patch_pos_embed.to(torch.float32)
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, height * width, -1)
            position_embeddings = torch.cat([class_pos_embed, patch_pos_embed.to(position_embeddings.dtype)], dim=1)
        embeddings = embeddings + position_embeddings
        return embeddings


class DeepseekOcrEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            **kwargs,
        )


class DeepseekOcrCLIPEncoder(CLIPEncoder):
    def __init__(self, config: DeepseekOcrCLIPVisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([DeepseekOcrEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,  # TODO get rid of this when we're done with the fwd pass
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        hidden_states = inputs_embeds

        all_hidden_states = [] if output_hidden_states else None

        for layer_module in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                **kwargs,
            )

        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            all_hidden_states = tuple(all_hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class DeepseekOcrCLIPVisionTransformer(CLIPVisionTransformer):
    def __init__(self, config):
        super().__init__(config)
        embed_dim = config.hidden_size
        self.embeddings = DeepseekOcrVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = DeepseekOcrCLIPEncoder(config)
        del self.post_layernorm

    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        patch_embeds = kwargs.pop("patch_embeds", None)
        hidden_states = self.embeddings(
            pixel_values,
            patch_embeds,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


class DeepseekOcrCLIPVisionModel(CLIPVisionModel):
    config_class = DeepseekOcrCLIPVisionConfig

    def __init__(self, config):
        super().__init__(config)
        self.vision_model = DeepseekOcrCLIPVisionTransformer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @check_model_inputs(tie_last_hidden_states=False)
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        Args:
            patch_embeds (`torch.FloatTensor`, *optional*):
                Precomputed patch embeddings derived from the SAM vision encoder. When provided, the transformer will
                reuse them instead of recomputing embeddings from `pixel_values`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, DeepseekOcrCLIPVisionModel

        >>> model = DeepseekOcrCLIPVisionModel.from_pretrained("openai/deepseek_ocr_c_l_i_p-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/deepseek_ocr_c_l_i_p-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""

        patch_embeds = kwargs.pop("patch_embeds", None)
        return self.vision_model(
            pixel_values=pixel_values,
            patch_embeds=patch_embeds,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )


class DeepseekOcrTextMLP(nn.Module):
    def __init__(self, config: DeepseekOcrTextConfig, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class DeepseekOcrTextExperts(nn.ModuleList):
    """
    ModuleList of experts.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        for _ in range(config.n_routed_experts):
            self.append(DeepseekOcrTextMLP(config, intermediate_size=config.moe_intermediate_size))

    def forward(self, hidden_states: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        tokens_per_expert = torch.bincount(topk_idx.view(-1), minlength=self.num_experts)

        flat_indices = topk_idx.view(-1)
        sorted_positions = flat_indices.argsort()
        original_token_indices = sorted_positions // self.top_k

        sorted_tokens = hidden_states[original_token_indices]
        combined_results = torch.empty_like(sorted_tokens)

        boundaries = torch.cumsum(tokens_per_expert, dim=0)
        start_indices = torch.cat((torch.tensor([0], device=boundaries.device), boundaries[:-1]))

        for i in range(self.num_experts):
            count = tokens_per_expert[i].item()
            if count == 0:
                continue

            start = start_indices[i].item()
            end = boundaries[i].item()

            combined_results[start:end] = self[i](sorted_tokens[start:end])

        dispatch_buffer = torch.empty_like(combined_results)
        dispatch_buffer.scatter_(0, sorted_positions.unsqueeze(-1).expand_as(combined_results), combined_results)

        dispatch_buffer = dispatch_buffer.view(topk_idx.shape[0], self.top_k, -1)
        weighted = dispatch_buffer.to(topk_weight.dtype) * topk_weight.unsqueeze(-1)

        return weighted.sum(dim=1).to(hidden_states.dtype)


class DeepseekOcrTextMoe(nn.Module):
    def __init__(self, config: DeepseekOcrTextConfig):
        super().__init__()
        self.config = config
        self.experts = DeepseekOcrTextExperts(config)
        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekOcrTextMLP(config=config, intermediate_size=intermediate_size)
        self.routed_scaling_factor = config.routed_scaling_factor
        self.topk_method = config.topk_method
        self.num_group = config.n_group
        self.top_k = config.num_experts_per_tok
        self.topk_group = config.topk_group
        self.norm_topk_prob = getattr(config, "norm_topk_prob", False)

    def route_tokens_to_experts(self, scores):
        if self.top_k is None or self.top_k <= 0:
            raise ValueError("`num_experts_per_tok` must be a positive integer for MoE routing.")

        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        elif self.topk_method == "group_limited_greedy":
            if self.num_group is None or self.topk_group is None:
                raise ValueError("`n_group` and `topk_group` must be provided for group_limited_greedy routing.")
            group_scores = scores.view(scores.shape[0], self.num_group, -1).max(dim=-1).values
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(scores.shape[0], self.num_group, scores.shape[-1] // self.num_group)
                .reshape(scores.shape[0], -1)
            )
            masked_scores = scores.masked_fill(~score_mask.bool(), 0.0)
            topk_weight, topk_idx = torch.topk(masked_scores, k=self.top_k, dim=-1, sorted=False)
        else:
            raise ValueError(f"Unsupported topk routing method: {self.topk_method}")

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True).clamp_min(1e-20)
            topk_weight = topk_weight / denominator

        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = nn.functional.linear(hidden_states.type(torch.float32), self.gate.weight.type(torch.float32))
        router_scores = router_logits.softmax(dim=-1, dtype=torch.float32)
        router_scores_flat = router_scores.view(-1, router_scores.shape[-1])
        topk_indices, topk_weights = self.route_tokens_to_experts(router_scores_flat)
        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])
        expert_output = self.experts(hidden_states_flat, topk_indices, topk_weights)
        hidden_states = expert_output.view(*orig_shape)

        if hasattr(self, "shared_experts"):
            hidden_states = hidden_states + self.shared_experts(residuals)

        return hidden_states


class DeepseekOcrTextAttention(LlamaAttention):
    pass


class DeepseekOcrTextDecoderLayer(DeepseekV2DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = DeepseekOcrTextAttention(config, layer_idx)
        self.mlp = (
            DeepseekOcrTextMoe(config) if layer_idx >= config.first_k_dense_replace else DeepseekOcrTextMLP(config)
        )


class DeepseekOcrTextRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class DeepseekOcrTextRMSNorm(DeepseekV2RMSNorm):
    pass


class DeepseekOcrTextPreTrainedModel(DeepseekV2PreTrainedModel):
    pass


class DeepseekOcrTextModel(DeepseekV2Model):
    config: DeepseekOcrTextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekOcrTextDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": DeepseekOcrTextDecoderLayer,
        "attentions": DeepseekOcrTextAttention,
    }

    def __init__(self, config):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [DeepseekOcrTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekOcrTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekOcrTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        for module in self.layers:
            if isinstance(module.mlp, DeepseekOcrTextMoe):
                module.mlp.gate.weight.data.normal_(mean=0.0, std=config.initializer_range)


class DeepseekOcrModel(LlavaNextModel):
    """
    Deepseek OCR model with dual vision encoders (SAM + CLIP) and a projector.
    """

    def __init__(self, config: DeepseekOcrConfig):
        super().__init__(config)
        del self.vision_tower
        del self.multi_modal_projector

        self.sam_model = DeepseekOcrSamVisionEncoder._from_config(config.vision_config.sam_config)
        self.clip_model = DeepseekOcrCLIPVisionModel._from_config(config.vision_config.clip_config)

        self.multi_modal_projector = DeepseekOcrProjector._from_config(config.projector_config)

        self.vocab_size = config.text_config.vocab_size
        self.language_model = DeepseekOcrTextModel._from_config(config.text_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        embed_std = 1 / math.sqrt(config.hidden_size)
        self.image_newline = nn.Parameter(torch.randn(config.hidden_size) * embed_std)
        self.view_seperator = nn.Parameter(
            torch.randn(config.hidden_size) * embed_std
        )  # TODO the typo is in the checkpoint

        self.post_init()

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_token_id):
        if input_ids is None:
            tok_embed = self.get_input_embeddings()(torch.tensor(image_token_id, device=inputs_embeds.device))
            mask = (inputs_embeds == tok_embed).all(dim=-1)
        else:
            mask = input_ids == self.config.image_token_id
        return mask.unsqueeze(-1).expand_as(inputs_embeds)

    def pack_image_features(
        self,
        image_features,
        image_sizes,
        vision_feature_select_strategy,
        image_newline=None,
        image_spatial_crops=None,
    ):
        new_image_features = []
        feature_lens = []

        for image_idx, features in enumerate(image_features):
            crop_shape = None
            if image_spatial_crops is not None:
                crop_shape = image_spatial_crops[image_idx]
                if isinstance(crop_shape, torch.Tensor):
                    crop_shape = crop_shape.tolist()
            width_crop_num = int(crop_shape[0]) if crop_shape is not None else 1
            height_crop_num = int(crop_shape[1]) if crop_shape is not None else 1
            has_local_crops = width_crop_num > 1 or height_crop_num > 1

            if has_local_crops and features.shape[0] >= width_crop_num * height_crop_num + 1:
                valid_patch_count = width_crop_num * height_crop_num + 1
            else:
                valid_patch_count = 1 if features.shape[0] > 0 else 0
                has_local_crops = False

            features = features[:valid_patch_count]
            if features.shape[0] == 0:
                new_image_features.append(features)
                feature_lens.append(0)
                continue

            global_feature = features[-1]
            local_features = features[:-1] if has_local_crops else features[:0]

            processed_parts = []

            if local_features.numel() > 0:
                local_tokens = local_features.shape[1]
                local_grid = int(math.isqrt(local_tokens))

                if local_grid * local_grid == local_tokens:
                    local_features = local_features.view(
                        height_crop_num,
                        width_crop_num,
                        local_grid,
                        local_grid,
                        -1,
                    )
                    local_features = local_features.permute(0, 2, 1, 3, 4).contiguous()
                    local_features = local_features.view(
                        height_crop_num * local_grid,
                        width_crop_num * local_grid,
                        -1,
                    )
                    if image_newline is not None:
                        newline = (
                            image_newline.unsqueeze(0)
                            .unsqueeze(0)
                            .to(local_features.device, dtype=local_features.dtype)
                            .expand(local_features.shape[0], 1, -1)
                        )
                        local_features = torch.cat((local_features, newline), dim=1)
                    local_features = local_features.view(-1, local_features.shape[-1])
                else:
                    local_features = local_features.view(-1, local_features.shape[-1])
                    if image_newline is not None:
                        newline = image_newline.unsqueeze(0).to(local_features.device, dtype=local_features.dtype)
                        local_features = torch.cat((local_features, newline), dim=0)

                processed_parts.append(local_features)

            global_tokens = global_feature.shape[0]
            global_grid = int(math.isqrt(global_tokens))

            if global_grid * global_grid == global_tokens:
                global_features = global_feature.view(global_grid, global_grid, -1)
                if image_newline is not None:
                    newline = (
                        image_newline.unsqueeze(0)
                        .unsqueeze(0)
                        .to(global_features.device, dtype=global_features.dtype)
                        .expand(global_grid, 1, -1)
                    )
                    global_features = torch.cat((global_features, newline), dim=1)
                global_features = global_features.view(-1, global_features.shape[-1])
            else:
                global_features = global_feature
                if image_newline is not None:
                    global_features = torch.cat(
                        (
                            global_features,
                            image_newline.unsqueeze(0).to(global_features.device, dtype=global_features.dtype),
                        ),
                        dim=0,
                    )

            processed_parts.append(global_features)

            combined = torch.cat(processed_parts, dim=0)
            new_image_features.append(combined)
            feature_lens.append(combined.size(0))

        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features[0].device)
        return new_image_features, feature_lens

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,  # (B, num_patches, 3, H, W) or (sum_patches, 3, H, W)
        image_sizes: torch.Tensor,  # (num_images, 2) actual (H, W)
        image_spatial_crops: Optional[torch.Tensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
    ):
        if pixel_values.dim() == 5:
            image_num_patches = [pv.shape[0] for pv in pixel_values]
            pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
        elif pixel_values.dim() == 4:
            image_num_patches = [pixel_values.shape[0]]
        else:
            raise ValueError(f"pixel_values has shape {pixel_values.shape}, expected 4D or 5D")

        sam_features = self.sam_model(pixel_values)
        sam_seq = sam_features.flatten(2).permute(0, 2, 1)

        clip_out = self.clip_model(
            pixel_values=pixel_values,
            patch_embeds=sam_features,
            output_hidden_states=True,
            return_dict=True,
            interpolate_pos_encoding=True,
        )

        clip_seq = clip_out.last_hidden_state
        vision_feature_layer_index = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )

        if vision_feature_layer_index is not None:
            if isinstance(vision_feature_layer_index, int):
                clip_seq = clip_out.hidden_states[vision_feature_layer_index]
            else:
                pool = [clip_out.hidden_states[i] for i in vision_feature_layer_index]
                clip_seq = torch.cat(pool, dim=-1)

        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        if vision_feature_select_strategy == "default":
            clip_seq = clip_seq[:, 1:]
        elif vision_feature_select_strategy != "full":
            raise ValueError(f"Unexpected vision_feature_select_strategy={vision_feature_select_strategy}")

        fused = torch.cat([clip_seq, sam_seq], dim=-1)
        proj = self.multi_modal_projector(fused)

        proj_list = torch.split(proj, image_num_patches, dim=0)

        new_image_features, feature_lens = self.pack_image_features(
            image_features=proj_list,
            image_sizes=image_sizes,
            vision_feature_select_strategy=vision_feature_select_strategy,
            image_newline=self.image_newline,
            image_spatial_crops=image_spatial_crops,
        )

        new_image_features = [
            torch.cat([pf, self.view_seperator.unsqueeze(0).to(pf.dtype)], dim=0) for pf in new_image_features
        ]
        feature_lens = feature_lens + 1  # account for view separator
        concatenated_features = torch.cat(new_image_features, dim=0)
        return concatenated_features, feature_lens

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_spatial_crop = kwargs.pop("image_spatial_crop", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_attention_mask = kwargs.pop("image_attention_mask", None)
        # num_img_tokens = kwargs.pop("num_img_tokens", None)
        if image_sizes is None and image_spatial_crop is not None:
            image_sizes = image_spatial_crop

        image_hidden_states = None
        if pixel_values is not None and pixel_values.abs().sum().item() != 0:
            if image_sizes is None:
                raise ValueError("image_sizes must be provided when pixel_values are passed to the model.")
            image_hidden_states, feature_lens = self.get_image_features(
                pixel_values,
                image_sizes,
                image_spatial_crops=image_spatial_crop,
            )

            if image_attention_mask is not None:
                token_mask = image_attention_mask.to(inputs_embeds.device)
            else:
                token_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds, self.config.image_token_index
                ).squeeze(-1)

            batch_size = token_mask.shape[0]
            start_idx = 0
            for batch_idx in range(batch_size):
                valid_len = feature_lens[batch_idx].item()
                if valid_len == 0:
                    continue
                mask_positions = token_mask[batch_idx].nonzero(as_tuple=True)[0]
                if mask_positions.numel() == 0:
                    continue
                if mask_positions.numel() > valid_len:
                    # deactivate surplus placeholders so they won't interfere with autoregressive decoding
                    extra_positions = mask_positions[valid_len:]
                    token_mask[batch_idx, extra_positions] = False
                    mask_positions = mask_positions[:valid_len]
                scatter_mask = torch.zeros_like(token_mask[batch_idx], dtype=torch.bool)
                scatter_mask[mask_positions] = True
                scatter_mask_expanded = scatter_mask.unsqueeze(-1).expand(-1, inputs_embeds.shape[-1])
                slice_features = image_hidden_states[start_idx : start_idx + valid_len].to(inputs_embeds.dtype)
                inputs_embeds[batch_idx] = inputs_embeds[batch_idx].masked_scatter(
                    scatter_mask_expanded, slice_features
                )
                start_idx += valid_len

        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        if not isinstance(outputs, BaseModelOutputWithPast):
            last_hidden_state = outputs[0]
            past = outputs[1] if len(outputs) > 1 else None
            hidden = outputs[2] if len(outputs) > 2 else None
            attn = outputs[3] if len(outputs) > 3 else None
        else:
            last_hidden_state = outputs.last_hidden_state
            past = outputs.past_key_values
            hidden = outputs.hidden_states
            attn = outputs.attentions

        return DeepseekOcrModelOutputWithPast(
            last_hidden_state=last_hidden_state,
            past_key_values=past,
            hidden_states=hidden,
            attentions=attn,
            image_hidden_states=image_hidden_states,
        )


@auto_docstring(
    custom_intro="""
    The Deepseek-OCR model which consists of two vision backbones and a deepseek language model.
    """
)
class DeepseekOcrForConditionalGeneration(LlavaNextForConditionalGeneration):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekOcrModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, DeepseekOcrCausalLMOutputWithPast]:
        image_spatial_crop = kwargs.pop("image_spatial_crop", None)
        image_sizes = kwargs.pop("image_sizes", None)
        if image_sizes is None and image_spatial_crop is not None:
            image_sizes = image_spatial_crop
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            image_spatial_crop=image_spatial_crop,
            image_sizes=image_sizes,
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

        return DeepseekOcrCausalLMOutputWithPast(
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
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        image_attention_mask=None,
        image_spatial_crop=None,
        num_img_tokens=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_sizes"] = image_sizes
            if image_attention_mask is not None:
                model_inputs["image_attention_mask"] = image_attention_mask
            if image_spatial_crop is not None:
                model_inputs["image_spatial_crop"] = image_spatial_crop
            if num_img_tokens is not None:
                model_inputs["num_img_tokens"] = num_img_tokens

        return model_inputs


__all__ = [
    "DeepseekOcrConfig",
    "DeepseekOcrVisionConfig",
    "DeepseekOcrSamConfig",
    "DeepseekOcrCLIPVisionConfig",
    "DeepseekOcrProjectorConfig",
    "DeepseekOcrModelOutputWithPast",
    "DeepseekOcrCausalLMOutputWithPast",
    "DeepseekOcrTextModel",
    "DeepseekOcrTextPreTrainedModel",
    "DeepseekOcrModel",
    "DeepseekOcrForConditionalGeneration",
    "DeepseekOcrPreTrainedModel",
    "DeepseekOcrProjector",
    "DeepseekOcrSamVisionEncoder",
    "DeepseekOcrCLIPVisionModel",
]
