# coding=utf-8
# Copyright 2022 School of EIC, Huazhong University of Science & Technology and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch YOLOS model."""

import collections.abc
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
from torch import nn

from ...activations import ACT2FN
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple, check_model_inputs
from .configuration_yolos import YolosConfig


logger = logging.get_logger(__name__)


@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`YolosForObjectDetection`].
    """
)
class YolosObjectDetectionOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
        Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
        bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
        scale-invariant IoU loss.
    loss_dict (`Dict`, *optional*):
        A dictionary containing the individual losses. Useful for logging.
    logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
        Classification logits (including no-object) for all queries.
    pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
        Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
        values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
        possible padding). You can use [`~YolosImageProcessor.post_process`] to retrieve the unnormalized bounding
        boxes.
    auxiliary_outputs (`list[Dict]`, *optional*):
        Optional, only returned when auxiliary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
        and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
        `pred_boxes`) for each decoder layer.
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
        Sequence of hidden-states at the output of the last layer of the decoder of the model.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[dict] = None
    logits: Optional[torch.FloatTensor] = None
    pred_boxes: Optional[torch.FloatTensor] = None
    auxiliary_outputs: Optional[list[dict]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


class YolosEmbeddings(nn.Module):
    """
    Construct the CLS token, detection tokens, position and patch embeddings.

    """

    def __init__(self, config: YolosConfig) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.detection_tokens = nn.Parameter(torch.zeros(1, config.num_detection_tokens, config.hidden_size))
        self.patch_embeddings = YolosPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + config.num_detection_tokens + 1, config.hidden_size)
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.interpolation = InterpolateInitialPositionEmbeddings(config)
        self.config = config

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        batch_size, seq_len, _ = embeddings.size()

        # add the [CLS] and detection tokens to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        detection_tokens = self.detection_tokens.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings, detection_tokens), dim=1)

        # add positional encoding to each token
        # this might require interpolation of the existing position embeddings
        position_embeddings = self.interpolation(self.position_embeddings, (height, width))

        embeddings = embeddings + position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class InterpolateInitialPositionEmbeddings(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def forward(self, pos_embed, img_size=(800, 1344)) -> torch.Tensor:
        cls_pos_embed = pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = pos_embed[:, -self.config.num_detection_tokens :, :]
        patch_pos_embed = pos_embed[:, 1 : -self.config.num_detection_tokens, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        batch_size, hidden_size, seq_len = patch_pos_embed.shape

        patch_height, patch_width = (
            self.config.image_size[0] // self.config.patch_size,
            self.config.image_size[1] // self.config.patch_size,
        )
        patch_pos_embed = patch_pos_embed.view(batch_size, hidden_size, patch_height, patch_width)

        height, width = img_size
        new_patch_height, new_patch_width = height // self.config.patch_size, width // self.config.patch_size
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed, size=(new_patch_height, new_patch_width), mode="bicubic", align_corners=False
        )
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1)
        return scale_pos_embed


class InterpolateMidPositionEmbeddings(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def forward(self, pos_embed, img_size=(800, 1344)) -> torch.Tensor:
        cls_pos_embed = pos_embed[:, :, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = pos_embed[:, :, -self.config.num_detection_tokens :, :]
        patch_pos_embed = pos_embed[:, :, 1 : -self.config.num_detection_tokens, :]
        patch_pos_embed = patch_pos_embed.transpose(2, 3)
        depth, batch_size, hidden_size, seq_len = patch_pos_embed.shape

        patch_height, patch_width = (
            self.config.image_size[0] // self.config.patch_size,
            self.config.image_size[1] // self.config.patch_size,
        )
        patch_pos_embed = patch_pos_embed.view(depth * batch_size, hidden_size, patch_height, patch_width)
        height, width = img_size
        new_patch_height, new_patch_width = height // self.config.patch_size, width // self.config.patch_size
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed, size=(new_patch_height, new_patch_width), mode="bicubic", align_corners=False
        )
        patch_pos_embed = (
            patch_pos_embed.flatten(2)
            .transpose(1, 2)
            .contiguous()
            .view(depth, batch_size, new_patch_height * new_patch_width, hidden_size)
        )
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=2)
        return scale_pos_embed


class YolosPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


# Copied from transformers.models.vit.modeling_vit.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    # Normalize the attention scores to probabilities.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Mask heads if we want to
    if attention_mask is not None:
        attn_weights = attn_weights * attention_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->Yolos
class YolosSelfAttention(nn.Module):
    def __init__(self, config: YolosConfig):
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

    def forward(
        self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            head_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        return context_layer, attention_probs


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->Yolos
class YolosSelfOutput(nn.Module):
    """
    The residual connection is defined in YolosLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: YolosConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->Yolos
class YolosAttention(nn.Module):
    def __init__(self, config: YolosConfig):
        super().__init__()
        self.attention = YolosSelfAttention(config)
        self.output = YolosSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: set[int]):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self_attn_output, _ = self.attention(hidden_states, head_mask)
        output = self.output(self_attn_output, hidden_states)
        return output


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate with ViT->Yolos
class YolosIntermediate(nn.Module):
    def __init__(self, config: YolosConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTOutput with ViT->Yolos
class YolosOutput(nn.Module):
    def __init__(self, config: YolosConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTLayer with ViT->Yolos,VIT->YOLOS
class YolosLayer(GradientCheckpointingLayer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: YolosConfig):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = YolosAttention(config)
        self.intermediate = YolosIntermediate(config)
        self.output = YolosOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states_norm = self.layernorm_before(hidden_states)
        attention_output = self.attention(hidden_states_norm, head_mask)

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in Yolos, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        return layer_output


class YolosEncoder(nn.Module):
    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([YolosLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        seq_length = (
            1 + (config.image_size[0] * config.image_size[1] // config.patch_size**2) + config.num_detection_tokens
        )
        self.mid_position_embeddings = (
            nn.Parameter(
                torch.zeros(
                    config.num_hidden_layers - 1,
                    1,
                    seq_length,
                    config.hidden_size,
                )
            )
            if config.use_mid_position_embeddings
            else None
        )

        self.interpolation = InterpolateMidPositionEmbeddings(config) if config.use_mid_position_embeddings else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        height: int,
        width: int,
        head_mask: Optional[torch.Tensor] = None,
    ) -> BaseModelOutput:
        if self.config.use_mid_position_embeddings:
            interpolated_mid_position_embeddings = self.interpolation(self.mid_position_embeddings, (height, width))

        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            hidden_states = layer_module(hidden_states, layer_head_mask)

            if self.config.use_mid_position_embeddings:
                if i < (self.config.num_hidden_layers - 1):
                    hidden_states = hidden_states + interpolated_mid_position_embeddings[i]

        return BaseModelOutput(last_hidden_state=hidden_states)


@auto_docstring
class YolosPreTrainedModel(PreTrainedModel):
    config: YolosConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": YolosLayer,
        "attentions": YolosSelfAttention,
    }

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@auto_docstring
class YolosModel(YolosPreTrainedModel):
    def __init__(self, config: YolosConfig, add_pooling_layer: bool = True):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        """
        super().__init__(config)
        self.config = config

        self.embeddings = YolosEmbeddings(config)
        self.encoder = YolosEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = YolosPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> YolosPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: dict[int, list[int]]) -> None:
        """
        Prunes heads of the model.

        Args:
            heads_to_prune (`dict`):
                See base class `PreTrainedModel`. The input dictionary must have the following format: {layer_num:
                list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values)

        height, width = pixel_values.shape[-2:]
        encoder_outputs: BaseModelOutput = self.encoder(
            embedding_output, height=height, width=width, head_mask=head_mask
        )
        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output)


class YolosPooler(nn.Module):
    def __init__(self, config: YolosConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.detr.modeling_detr.DetrMLPPredictionHead with Detr->Yolos
class YolosMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@auto_docstring(
    custom_intro="""
    YOLOS Model (consisting of a ViT encoder) with object detection heads on top, for tasks such as COCO detection.
    """
)
class YolosForObjectDetection(YolosPreTrainedModel):
    def __init__(self, config: YolosConfig):
        super().__init__(config)

        # YOLOS (ViT) encoder model
        self.vit = YolosModel(config, add_pooling_layer=False)

        # Object detection heads
        # We add one for the "no object" class
        self.class_labels_classifier = YolosMLPPredictionHead(
            input_dim=config.hidden_size, hidden_dim=config.hidden_size, output_dim=config.num_labels + 1, num_layers=3
        )
        self.bbox_predictor = YolosMLPPredictionHead(
            input_dim=config.hidden_size, hidden_dim=config.hidden_size, output_dim=4, num_layers=3
        )

        # Initialize weights and apply final processing
        self.post_init()

    # taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[list[dict]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> YolosObjectDetectionOutput:
        r"""
        labels (`list[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: `'class_labels'` and `'boxes'` (the class labels and bounding boxes of an image in the
            batch respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding
            boxes in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image,
            4)`.

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModelForObjectDetection
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
        >>> model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
        ...     0
        ... ]

        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected remote with confidence 0.991 at location [46.48, 72.78, 178.98, 119.3]
        Detected remote with confidence 0.908 at location [336.48, 79.27, 368.23, 192.36]
        Detected cat with confidence 0.934 at location [337.18, 18.06, 638.14, 373.09]
        Detected cat with confidence 0.979 at location [10.93, 53.74, 313.41, 470.67]
        Detected remote with confidence 0.974 at location [41.63, 72.23, 178.09, 119.99]
        ```"""

        # First, sent images through YOLOS base model to obtain hidden states
        outputs: BaseModelOutputWithPooling = self.vit(pixel_values, **kwargs)
        sequence_output = outputs.last_hidden_state

        # Take the final hidden states of the detection tokens
        sequence_output = sequence_output[:, -self.config.num_detection_tokens :, :]

        # Class logits + predicted bounding boxes
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            outputs_class, outputs_coord = None, None
            if self.config.auxiliary_loss:
                intermediate = outputs.hidden_states
                outputs_class = self.class_labels_classifier(intermediate)
                outputs_coord = self.bbox_predictor(intermediate).sigmoid()
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits, labels, self.device, pred_boxes, self.config, outputs_class, outputs_coord
            )

        return YolosObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["YolosForObjectDetection", "YolosModel", "YolosPreTrainedModel"]
