# coding=utf-8
# Copyright 2025 Mobile Perception Systems Lab at TU/e and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch EoMT model."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    auto_docstring,
    can_return_tuple,
    logging,
)
from ..dinov2.modeling_dinov2 import (
    Dinov2Embeddings,
    Dinov2Layer,
    Dinov2LayerScale,
    Dinov2PatchEmbeddings,
)
from ..mask2former.modeling_mask2former import Mask2FormerForUniversalSegmentation, Mask2FormerLoss
from ..siglip.modeling_siglip import SiglipAttention
from .configuration_eomt import EoMTConfig


logger = logging.get_logger(__name__)


@dataclass
class EoMTForUniversalSegmentationOutput(ModelOutput):
    """
    Class for outputs of [`EoMTForUniversalSegmentationOutput`].

    This output can be directly passed to [`~EoMTFormerImageProcessor.post_process_semantic_segmentation`] or
    [`~EoMTFormerImageProcessor.post_process_instance_segmentation`] or
    [`~EoMTFormerImageProcessor.post_process_panoptic_segmentation`] to compute final segmentation maps. Please, see
    [`~EoMTFormerImageProcessor] for details regarding usage.

    Args:
        loss (`torch.Tensor`, *optional*):
            The computed loss, returned when labels are present.
        class_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, num_labels + 1)` representing the proposed classes for each
            query. Note the `+ 1` is needed because we incorporate the null class.
        masks_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
            query.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last layer.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states all layers of the model.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tuple(torch.FloatTensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Self and Cross Attentions weights from transformer decoder.
    """

    loss: Optional[torch.FloatTensor] = None
    class_queries_logits: Optional[torch.FloatTensor] = None
    masks_queries_logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class EoMTLoss(Mask2FormerLoss):
    pass


class EoMTPatchEmbeddings(Dinov2PatchEmbeddings):
    pass


class EoMTEmbeddings(Dinov2Embeddings, nn.Module):
    def __init__(self, config: EoMTConfig) -> None:
        nn.Module().__init__()

        self.config = config
        self.patch_size = config.patch_size

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.register_tokens = nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size))

        self.patch_embeddings = EoMTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_prefix_tokens = 1 + config.num_register_tokens  # 1 for [CLS]
        self.position_embeddings = nn.Embedding(num_patches, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(num_patches).expand((1, -1)), persistent=False)

    def interpolate_pos_encoding(self):
        raise AttributeError("Not needed for EoMT")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, _ = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)

        embeddings = embeddings + self.position_embeddings(self.position_ids)
        embeddings = torch.cat([cls_tokens, register_tokens, embeddings], dim=1)

        embeddings = self.dropout(embeddings)

        return embeddings


class EoMTAttention(SiglipAttention):
    pass


class EoMTLayerScale(Dinov2LayerScale):
    pass


class EoMTLayer(Dinov2Layer):
    pass


class EoMTLayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = hidden_state.permute(0, 2, 3, 1)
        hidden_state = F.layer_norm(hidden_state, self.normalized_shape, self.weight, self.bias, self.eps)
        hidden_state = hidden_state.permute(0, 3, 1, 2)
        return hidden_state


class EoMTScaleLayer(nn.Module):
    def __init__(self, config: EoMTConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.conv1 = nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=2, stride=2)
        self.activation = ACT2FN[config.hidden_act]
        self.conv2 = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=3,
            padding=1,
            groups=hidden_size,
            bias=False,
        )

        self.layernorm2d = EoMTLayerNorm2d(hidden_size)

    def forward(self, hidden_states: torch.tensor) -> torch.Tensor:
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layernorm2d(hidden_states)
        return hidden_states


class EoMTScaleBlock(nn.Module):
    def __init__(self, config: EoMTConfig):
        super().__init__()
        self.num_blocks = config.num_upscale_blocks
        self.block = nn.ModuleList([EoMTScaleLayer(config) for _ in range(self.num_blocks)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.block:
            hidden_states = block(hidden_states)
        return hidden_states


class EoMTMaskHead(nn.Module):
    def __init__(self, config: EoMTConfig):
        super().__init__()

        hidden_size = config.hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation(self.fc1(hidden_states))
        hidden_states = self.activation(self.fc2(hidden_states))
        hidden_states = self.fc3(hidden_states)
        return hidden_states


@auto_docstring
class EoMTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EoMTConfig
    base_model_prefix = "eomt"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False
    _no_split_modules = ["EoMTMLP"]
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, EoMTLayerScale):
            if hasattr(module, "lambda1"):
                module.lambda1.data.fill_(self.config.layerscale_value)
        elif isinstance(module, EoMTEmbeddings):
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32), mean=0.0, std=std
            ).to(module.cls_token.dtype)
            module.register_tokens.data.zero_()


@auto_docstring(
    custom_intro="""
    The EoMT Model with heads on top for instance/semantic/panoptic segmentation.
    """
)
class EoMTForUniversalSegmentation(Mask2FormerForUniversalSegmentation, nn.Module):
    def __init__(self, config: EoMTConfig) -> None:
        nn.Module().__init__(config)
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.embeddings = EoMTEmbeddings(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.query = nn.Embedding(config.num_queries, config.hidden_size)
        self.layers = nn.ModuleList([EoMTLayer(config) for _ in range(config.num_hidden_layers)])

        self.upscale_block = EoMTScaleBlock(config)
        self.mask_head = EoMTMaskHead(config)

        self.class_predictor = nn.Linear(config.hidden_size, config.num_labels + 1)

        self.grid_size = (config.image_size // config.patch_size, config.image_size // config.patch_size)
        self.weight_dict: Dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        self.criterion = EoMTLoss(config=config, weight_dict=self.weight_dict)

        self.register_buffer("attn_mask_probs", torch.ones(config.num_blocks))

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def get_auxiliary_logits(self):
        raise AttributeError("Note needed for EoMT")

    def predict(self, logits: torch.Tensor):
        query_tokens = logits[:, : self.config.num_queries, :]
        class_logits = self.class_predictor(query_tokens)

        prefix_tokens = logits[:, self.config.num_queries + self.embeddings.num_prefix_tokens :, :]
        prefix_tokens = prefix_tokens.transpose(1, 2)

        prefix_tokens = prefix_tokens.reshape(prefix_tokens.shape[0], -1, *self.grid_size)

        query_tokens = self.mask_head(query_tokens)
        prefix_tokens = self.upscale_block(prefix_tokens)

        mask_logits = torch.einsum("bqc, bchw -> bqhw", query_tokens, prefix_tokens)

        return mask_logits, class_logits

    @staticmethod
    def _disable_attention_mask(attn_mask, prob, num_query_tokens, encoder_start_tokens, device):
        if prob < 1:
            # Generate random queries to disable based on the probs
            random_queries = torch.rand(attn_mask.shape[0], num_query_tokens, device=device) > prob

            # Disable attention to the query tokens, considering the prefix tokens
            attn_mask[:, :num_query_tokens, encoder_start_tokens:][random_queries] = 1

        return attn_mask

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        pixel_values: Tensor,
        mask_labels: Optional[List[Tensor]] = None,
        class_labels: Optional[List[Tensor]] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ):
        r"""
        mask_labels (`List[torch.Tensor]`, *optional*):
            List of mask labels of shape `(num_labels, height, width)` to be fed to a model
        class_labels (`List[torch.LongTensor]`, *optional*):
            list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the
            labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        masks_queries_logits_per_layer, class_queries_logits_per_layer = (), ()
        attention_mask = None

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)

        for idx, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if idx == self.num_hidden_layers - self.config.num_blocks:
                query = self.query.weight[None, :, :].expand(hidden_states.shape[0], -1, -1)
                hidden_states = torch.cat((query, hidden_states), dim=1)

            if self.training and idx >= self.num_hidden_layers - self.config.num_blocks:
                norm_hidden_states = self.layernorm(hidden_states)
                masks_queries_logits, class_queries_logits = self.predict(norm_hidden_states)

                masks_queries_logits_per_layer += (masks_queries_logits,)
                class_queries_logits_per_layer += (class_queries_logits,)

                attention_mask = torch.ones(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    hidden_states.shape[1],
                    device=hidden_states.device,
                    dtype=torch.bool,
                )

                interpolated_logits = F.interpolate(masks_queries_logits, size=self.grid_size, mode="bilinear")
                interpolated_logits = interpolated_logits.view(
                    interpolated_logits.size(0), interpolated_logits.size(1), -1
                )

                num_query_tokens = self.config.num_queries
                encoder_start_tokens = num_query_tokens + self.embeddings.num_prefix_tokens

                # Set attention mask for queries to focus on encoder tokens based on interpolated logits
                attention_mask[:, :num_query_tokens, encoder_start_tokens:] = interpolated_logits > 0

                # Disable attention mask for random query tokens.
                attention_mask = self._disable_attention_mask(
                    attention_mask,
                    prob=self.attn_mask_probs[idx - self.num_hidden_layers + self.config.num_blocks],
                    num_query_tokens=num_query_tokens,
                    encoder_start_tokens=encoder_start_tokens,
                    device=attention_mask.device,
                )

                # Expand attention mask to 4d mask.
                attention_mask = attention_mask[:, None, ...].expand(-1, self.config.num_attention_heads, -1, -1)
                attention_mask = attention_mask.float().masked_fill(~attention_mask, -1e9)

            layer_outputs = layer_module(hidden_states, attention_mask, output_attentions)
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        sequence_output = self.layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (sequence_output,)

        masks_queries_logits, class_queries_logits = self.predict(sequence_output)
        masks_queries_logits_per_layer += (masks_queries_logits,)
        class_queries_logits_per_layer += (class_queries_logits,)

        loss = None
        if mask_labels is not None and class_labels is not None:
            loss = 0.0
            for masks_queries_logits, class_queries_logits in zip(
                masks_queries_logits_per_layer, class_queries_logits_per_layer
            ):
                loss_dict = self.get_loss_dict(
                    masks_queries_logits=masks_queries_logits,
                    class_queries_logits=class_queries_logits,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                    auxiliary_predictions=None,
                )
                loss += self.get_loss(loss_dict)

        return EoMTForUniversalSegmentationOutput(
            loss=loss,
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            last_hidden_state=sequence_output,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


__all__ = ["EoMTPreTrainedModel", "EoMTForUniversalSegmentation"]
