# coding=utf-8
# Copyright 2024 University of Sydney and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch VitPose backbone model.

This code is the same as the original Vision Transformer (ViT) with 2 modifications:
- use of padding=2 in the patch embedding layer
- addition of a mixture-of-experts MLP layer
"""

import collections.abc
from collections.abc import Callable
from typing import Optional, Union

import torch
from torch import nn

from ...activations import ACT2FN
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.backbone_utils import BackboneMixin
from ...utils.generic import check_model_inputs
from .configuration_vitpose_backbone import VitPoseBackboneConfig


logger = logging.get_logger(__name__)


class VitPoseBackbonePatchEmbeddings(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, config):
        super().__init__()

        image_size = config.image_size
        patch_size = config.patch_size
        num_channels = config.num_channels
        embed_dim = config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=2)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        height, width = pixel_values.shape[-2:]
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        embeddings = self.projection(pixel_values)

        embeddings = embeddings.flatten(2).transpose(1, 2)
        return embeddings


class VitPoseBackboneEmbeddings(nn.Module):
    """
    Construct the position and patch embeddings.
    """

    def __init__(self, config: VitPoseBackboneConfig):
        super().__init__()

        self.patch_embeddings = VitPoseBackbonePatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.patch_embeddings(pixel_values)

        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings[:, 1:] + self.position_embeddings[:, :1]

        embeddings = self.dropout(embeddings)

        return embeddings


# Copied from transformers.models.bert.modeling_bert.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->VitPoseBackbone
class VitPoseBackboneSelfAttention(nn.Module):
    def __init__(self, config: VitPoseBackboneConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout_prob = config.attention_probs_dropout_prob
        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        new_shape = batch_size, -1, self.num_attention_heads, self.attention_head_size

        key_layer = self.key(hidden_states).view(*new_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*new_shape).transpose(1, 2)
        query_layer = self.query(hidden_states).view(*new_shape).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            None,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        return context_layer, attention_probs


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->VitPoseBackbone
class VitPoseBackboneSelfOutput(nn.Module):
    """
    The residual connection is defined in VitPoseBackboneLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: VitPoseBackboneConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->VitPoseBackbone
class VitPoseBackboneAttention(nn.Module):
    def __init__(self, config: VitPoseBackboneConfig):
        super().__init__()
        self.attention = VitPoseBackboneSelfAttention(config)
        self.output = VitPoseBackboneSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self_attn_output, _ = self.attention(hidden_states)
        output = self.output(self_attn_output, hidden_states)
        return output


class VitPoseNaiveMoe(nn.ModuleList):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.part_features = config.part_features

        hidden_features = int(config.hidden_size * config.mlp_ratio)
        part_features = config.part_features

        for _ in range(self.num_experts):
            self.append(nn.Linear(hidden_features, part_features))

    def forward(self, hidden_state, indices):
        expert_hidden_state = torch.zeros_like(hidden_state[:, :, -self.part_features :])
        for i in range(self.num_experts):
            selected_index = indices == i
            current_hidden_state = self[i](hidden_state) * selected_index
            expert_hidden_state = expert_hidden_state + current_hidden_state

        return expert_hidden_state


class VitPoseBackboneMoeMLP(nn.Module):
    def __init__(self, config: VitPoseBackboneConfig):
        super().__init__()

        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)

        num_experts = config.num_experts
        part_features = config.part_features

        self.part_features = part_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(hidden_features, out_features - part_features)

        self.num_experts = num_experts
        self.experts = VitPoseNaiveMoe(config)

    def forward(self, hidden_state: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.act(hidden_state)
        shared_hidden_state = self.fc2(hidden_state)
        indices = indices.view(-1, 1, 1)

        expert_hidden_state = self.experts(hidden_state, indices)
        hidden_state = torch.cat([shared_hidden_state, expert_hidden_state], dim=-1)

        return hidden_state


class VitPoseBackboneMLP(nn.Module):
    def __init__(self, config: VitPoseBackboneConfig):
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.activation = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class VitPoseBackboneLayer(GradientCheckpointingLayer):
    def __init__(self, config: VitPoseBackboneConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.attention = VitPoseBackboneAttention(config)
        self.mlp = VitPoseBackboneMLP(config) if self.num_experts == 1 else VitPoseBackboneMoeMLP(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        dataset_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Validate dataset_index when using multiple experts
        if self.num_experts > 1 and dataset_index is None:
            raise ValueError(
                "dataset_index must be provided when using multiple experts "
                f"(num_experts={self.num_experts}). Please provide dataset_index "
                "to the forward pass."
            )

        hidden_states_norm = self.layernorm_before(hidden_states)
        attention_output = self.attention(hidden_states_norm)

        # first residual connection
        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        if self.num_experts == 1:
            layer_output = self.mlp(layer_output)
        else:
            layer_output = self.mlp(layer_output, indices=dataset_index)

        # second residual connection
        layer_output = layer_output + hidden_states

        return layer_output


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->VitPoseBackbone
class VitPoseBackboneEncoder(nn.Module):
    def __init__(self, config: VitPoseBackboneConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([VitPoseBackboneLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        dataset_index: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutput:
        all_hidden_states = [hidden_states] if output_hidden_states else None
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, dataset_index)
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if all_hidden_states else None,
        )


@auto_docstring
class VitPoseBackbonePreTrainedModel(PreTrainedModel):
    config: VitPoseBackboneConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    input_modalities = "image"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VitPoseBackboneEmbeddings", "VitPoseBackboneLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _can_record_outputs = {
        "attentions": VitPoseBackboneSelfAttention,
    }

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm, VitPoseBackboneEmbeddings]):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, VitPoseBackboneEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)


@auto_docstring(
    custom_intro="""
    The VitPose backbone useful for downstream tasks.
    """
)
class VitPoseBackbone(VitPoseBackbonePreTrainedModel, BackboneMixin):
    def __init__(self, config: VitPoseBackboneConfig):
        super().__init__(config)
        super()._init_backbone(config)

        self.num_features = [config.hidden_size for _ in range(config.num_hidden_layers + 1)]
        self.embeddings = VitPoseBackboneEmbeddings(config)
        self.encoder = VitPoseBackboneEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        dataset_index: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ):
        r"""
        dataset_index (`torch.Tensor` of shape `(batch_size,)`):
            Index to use in the Mixture-of-Experts (MoE) blocks of the backbone.

            This corresponds to the dataset index used during training, e.g. index 0 refers to COCO.

        Examples:

        ```python
        >>> from transformers import VitPoseBackboneConfig, VitPoseBackbone
        >>> import torch

        >>> config = VitPoseBackboneConfig(out_indices=[-1])
        >>> model = VitPoseBackbone(config)

        >>> pixel_values = torch.randn(1, 3, 256, 192)
        >>> dataset_index = torch.tensor([1])
        >>> outputs = model(pixel_values, dataset_index)
        ```"""

        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        embedding_output = self.embeddings(pixel_values)
        outputs: BaseModelOutput = self.encoder(
            embedding_output, dataset_index=dataset_index, output_hidden_states=True
        )
        hidden_states = outputs.hidden_states

        feature_maps = []
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                hidden_state = self.layernorm(hidden_state)
                feature_maps.append(hidden_state)

        return BackboneOutput(
            feature_maps=tuple(feature_maps),
            hidden_states=outputs.hidden_states if output_hidden_states else None,
        )


__all__ = ["VitPoseBackbonePreTrainedModel", "VitPoseBackbone"]
