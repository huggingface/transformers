# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""PyTorch SuperGlue model."""

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn

from transformers import PreTrainedModel
from transformers.models.superglue.configuration_superglue import SuperGlueConfig

from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, auto_docstring, logging
from ..auto import AutoModelForKeypointDetection


logger = logging.get_logger(__name__)


def concat_pairs(tensor_tuple0: tuple[torch.Tensor], tensor_tuple1: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
    """
    Concatenate two tuples of tensors pairwise

    Args:
        tensor_tuple0 (`tuple[torch.Tensor]`):
            Tuple of tensors.
        tensor_tuple1 (`tuple[torch.Tensor]`):
            Tuple of tensors.

    Returns:
        (`tuple[torch.Tensor]`): Tuple of concatenated tensors.
    """
    return tuple([torch.cat([tensor0, tensor1]) for tensor0, tensor1 in zip(tensor_tuple0, tensor_tuple1)])


def normalize_keypoints(keypoints: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Normalize keypoints locations based on image image_shape

    Args:
        keypoints (`torch.Tensor` of shape `(batch_size, num_keypoints, 2)`):
            Keypoints locations in (x, y) format.
        height (`int`):
            Image height.
        width (`int`):
            Image width.

    Returns:
        Normalized keypoints locations of shape (`torch.Tensor` of shape `(batch_size, num_keypoints, 2)`).
    """
    size = torch.tensor([width, height], device=keypoints.device, dtype=keypoints.dtype)[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (keypoints - center[:, None, :]) / scaling[:, None, :]


def log_sinkhorn_iterations(
    log_cost_matrix: torch.Tensor,
    log_source_distribution: torch.Tensor,
    log_target_distribution: torch.Tensor,
    num_iterations: int,
) -> torch.Tensor:
    """
    Perform Sinkhorn Normalization in Log-space for stability

    Args:
        log_cost_matrix (`torch.Tensor` of shape `(batch_size, num_rows, num_columns)`):
            Logarithm of the cost matrix.
        log_source_distribution (`torch.Tensor` of shape `(batch_size, num_rows)`):
            Logarithm of the source distribution.
        log_target_distribution (`torch.Tensor` of shape `(batch_size, num_columns)`):
            Logarithm of the target distribution.

    Returns:
        log_cost_matrix (`torch.Tensor` of shape `(batch_size, num_rows, num_columns)`): Logarithm of the optimal
        transport matrix.
    """
    log_u_scaling = torch.zeros_like(log_source_distribution)
    log_v_scaling = torch.zeros_like(log_target_distribution)
    for _ in range(num_iterations):
        log_u_scaling = log_source_distribution - torch.logsumexp(log_cost_matrix + log_v_scaling.unsqueeze(1), dim=2)
        log_v_scaling = log_target_distribution - torch.logsumexp(log_cost_matrix + log_u_scaling.unsqueeze(2), dim=1)
    return log_cost_matrix + log_u_scaling.unsqueeze(2) + log_v_scaling.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, reg_param: torch.Tensor, iterations: int) -> torch.Tensor:
    """
    Perform Differentiable Optimal Transport in Log-space for stability

    Args:
        scores: (`torch.Tensor` of shape `(batch_size, num_rows, num_columns)`):
            Cost matrix.
        reg_param: (`torch.Tensor` of shape `(batch_size, 1, 1)`):
            Regularization parameter.
        iterations: (`int`):
            Number of Sinkhorn iterations.

    Returns:
        log_optimal_transport_matrix: (`torch.Tensor` of shape `(batch_size, num_rows, num_columns)`): Logarithm of the
        optimal transport matrix.
    """
    batch_size, num_rows, num_columns = scores.shape
    one_tensor = scores.new_tensor(1)
    num_rows_tensor, num_columns_tensor = (num_rows * one_tensor).to(scores), (num_columns * one_tensor).to(scores)

    source_reg_param = reg_param.expand(batch_size, num_rows, 1)
    target_reg_param = reg_param.expand(batch_size, 1, num_columns)
    reg_param = reg_param.expand(batch_size, 1, 1)

    couplings = torch.cat([torch.cat([scores, source_reg_param], -1), torch.cat([target_reg_param, reg_param], -1)], 1)

    log_normalization = -(num_rows_tensor + num_columns_tensor).log()
    log_source_distribution = torch.cat(
        [log_normalization.expand(num_rows), num_columns_tensor.log()[None] + log_normalization]
    )
    log_target_distribution = torch.cat(
        [log_normalization.expand(num_columns), num_rows_tensor.log()[None] + log_normalization]
    )
    log_source_distribution, log_target_distribution = (
        log_source_distribution[None].expand(batch_size, -1),
        log_target_distribution[None].expand(batch_size, -1),
    )

    log_optimal_transport_matrix = log_sinkhorn_iterations(
        couplings, log_source_distribution, log_target_distribution, num_iterations=iterations
    )
    log_optimal_transport_matrix = log_optimal_transport_matrix - log_normalization  # multiply probabilities by M+N
    return log_optimal_transport_matrix


def arange_like(x, dim: int) -> torch.Tensor:
    return x.new_ones(x.shape[dim]).cumsum(0) - 1


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for outputs of keypoint matching models. Due to the nature of keypoint detection and matching, the number
    of keypoints is not fixed and can vary from image to image, which makes batching non-trivial. In the batch of
    images, the maximum number of matches is set as the dimension of the matches and matching scores. The mask tensor is
    used to indicate which values in the keypoints, matches and matching_scores tensors are keypoint matching
    information.
    """
)
class KeypointMatchingOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
        Loss computed during training.
    matches (`torch.FloatTensor` of shape `(batch_size, 2, num_matches)`):
        Index of keypoint matched in the other image.
    matching_scores (`torch.FloatTensor` of shape `(batch_size, 2, num_matches)`):
        Scores of predicted matches.
    keypoints (`torch.FloatTensor` of shape `(batch_size, num_keypoints, 2)`):
        Absolute (x, y) coordinates of predicted keypoints in a given image.
    mask (`torch.IntTensor` of shape `(batch_size, num_keypoints)`):
        Mask indicating which values in matches and matching_scores are keypoint matching information.
    hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*):
        Tuple of `torch.FloatTensor` (one for the output of each stage) of shape `(batch_size, 2, num_channels,
        num_keypoints)`, returned when `output_hidden_states=True` is passed or when
        `config.output_hidden_states=True`)
    attentions (`tuple[torch.FloatTensor, ...]`, *optional*):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, 2, num_heads, num_keypoints,
        num_keypoints)`, returned when `output_attentions=True` is passed or when `config.output_attentions=True`)
    """

    loss: Optional[torch.FloatTensor] = None
    matches: Optional[torch.FloatTensor] = None
    matching_scores: Optional[torch.FloatTensor] = None
    keypoints: Optional[torch.FloatTensor] = None
    mask: Optional[torch.IntTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


class SuperGlueMultiLayerPerceptron(nn.Module):
    def __init__(self, config: SuperGlueConfig, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.linear(hidden_state)
        hidden_state = hidden_state.transpose(-1, -2)
        hidden_state = self.batch_norm(hidden_state)
        hidden_state = hidden_state.transpose(-1, -2)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class SuperGlueKeypointEncoder(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        layer_sizes = config.keypoint_encoder_sizes
        hidden_size = config.hidden_size
        # 3 here consists of 2 for the (x, y) coordinates and 1 for the score of the keypoint
        encoder_channels = [3] + layer_sizes + [hidden_size]

        layers = [
            SuperGlueMultiLayerPerceptron(config, encoder_channels[i - 1], encoder_channels[i])
            for i in range(1, len(encoder_channels) - 1)
        ]
        layers.append(nn.Linear(encoder_channels[-2], encoder_channels[-1]))
        self.encoder = nn.ModuleList(layers)

    def forward(
        self,
        keypoints: torch.Tensor,
        scores: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor]]]:
        scores = scores.unsqueeze(2)
        hidden_state = torch.cat([keypoints, scores], dim=2)
        all_hidden_states = () if output_hidden_states else None
        for layer in self.encoder:
            hidden_state = layer(hidden_state)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
        return hidden_state, all_hidden_states


class SuperGlueSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.Tensor]:
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None
        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        attention_mask = encoder_attention_mask if is_cross_attention else attention_mask

        batch_size = hidden_states.shape[0]
        key_layer = (
            self.key(current_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        value_layer = (
            self.value(current_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        query_layer = (
            self.query(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in SuperGlueModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (None,)
        return outputs


class SuperGlueSelfOutput(nn.Module):
    def __init__(self, config: SuperGlueConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, *args) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return hidden_states


SUPERGLUE_SELF_ATTENTION_CLASSES = {
    "eager": SuperGlueSelfAttention,
}


class SuperGlueAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = SUPERGLUE_SELF_ATTENTION_CLASSES[config._attn_implementation](
            config,
            position_embedding_type=position_embedding_type,
        )
        self.output = SuperGlueSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class SuperGlueAttentionalPropagation(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.attention = SuperGlueAttention(config)
        mlp_channels = [hidden_size * 2, hidden_size * 2, hidden_size]
        layers = [
            SuperGlueMultiLayerPerceptron(config, mlp_channels[i - 1], mlp_channels[i])
            for i in range(1, len(mlp_channels) - 1)
        ]
        layers.append(nn.Linear(mlp_channels[-2], mlp_channels[-1]))
        self.mlp = nn.ModuleList(layers)

    def forward(
        self,
        descriptors: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor]], Optional[tuple[torch.Tensor]]]:
        attention_outputs = self.attention(
            descriptors,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        output = attention_outputs[0]
        attention = attention_outputs[1:]

        hidden_state = torch.cat([descriptors, output], dim=2)

        all_hidden_states = () if output_hidden_states else None
        for layer in self.mlp:
            hidden_state = layer(hidden_state)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

        return hidden_state, all_hidden_states, attention


class SuperGlueAttentionalGNN(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layers_types = config.gnn_layers_types
        self.layers = nn.ModuleList([SuperGlueAttentionalPropagation(config) for _ in range(len(self.layers_types))])

    def forward(
        self,
        descriptors: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: Optional[bool] = False,
    ) -> tuple[torch.Tensor, Optional[tuple], Optional[tuple]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        batch_size, num_keypoints, _ = descriptors.shape
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (descriptors,)

        for gnn_layer, layer_type in zip(self.layers, self.layers_types):
            encoder_hidden_states = None
            encoder_attention_mask = None
            if layer_type == "cross":
                encoder_hidden_states = (
                    descriptors.reshape(-1, 2, num_keypoints, self.hidden_size)
                    .flip(1)
                    .reshape(batch_size, num_keypoints, self.hidden_size)
                )
                encoder_attention_mask = (
                    mask.reshape(-1, 2, 1, 1, num_keypoints).flip(1).reshape(batch_size, 1, 1, num_keypoints)
                    if mask is not None
                    else None
                )

            gnn_outputs = gnn_layer(
                descriptors,
                attention_mask=mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            delta = gnn_outputs[0]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + gnn_outputs[1]
            if output_attentions:
                all_attentions = all_attentions + gnn_outputs[2]

            descriptors = descriptors + delta
        return descriptors, all_hidden_states, all_attentions


class SuperGlueFinalProjection(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.final_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        return self.final_proj(descriptors)


@auto_docstring
class SuperGluePreTrainedModel(PreTrainedModel):
    config: SuperGlueConfig
    base_model_prefix = "superglue"
    main_input_name = "pixel_values"

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm1d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if hasattr(module, "bin_score"):
            module.bin_score.data.fill_(1.0)


@auto_docstring(
    custom_intro="""
    SuperGlue model taking images as inputs and outputting the matching of them.
    """
)
class SuperGlueForKeypointMatching(SuperGluePreTrainedModel):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://huggingface.co/papers/1911.11763
    """

    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__(config)

        self.keypoint_detector = AutoModelForKeypointDetection.from_config(config.keypoint_detector_config)

        self.keypoint_encoder = SuperGlueKeypointEncoder(config)
        self.gnn = SuperGlueAttentionalGNN(config)
        self.final_projection = SuperGlueFinalProjection(config)

        bin_score = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter("bin_score", bin_score)

        self.post_init()

    def _match_image_pair(
        self,
        keypoints: torch.Tensor,
        descriptors: torch.Tensor,
        scores: torch.Tensor,
        height: int,
        width: int,
        mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple, tuple]:
        """
        Perform keypoint matching between two images.

        Args:
            keypoints (`torch.Tensor` of shape `(batch_size, 2, num_keypoints, 2)`):
                Keypoints detected in the pair of image.
            descriptors (`torch.Tensor` of shape `(batch_size, 2, descriptor_dim, num_keypoints)`):
                Descriptors of the keypoints detected in the image pair.
            scores (`torch.Tensor` of shape `(batch_size, 2, num_keypoints)`):
                Confidence scores of the keypoints detected in the image pair.
            height (`int`): Image height.
            width (`int`): Image width.
            mask (`torch.Tensor` of shape `(batch_size, 2, num_keypoints)`, *optional*):
                Mask indicating which values in the keypoints, matches and matching_scores tensors are keypoint matching
                information.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors. Default to `config.output_attentions`.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. Default to `config.output_hidden_states`.

        Returns:
            matches (`torch.Tensor` of shape `(batch_size, 2, num_keypoints)`):
                For each image pair, for each keypoint in image0, the index of the keypoint in image1 that was matched
                with. And for each keypoint in image1, the index of the keypoint in image0 that was matched with.
            matching_scores (`torch.Tensor` of shape `(batch_size, 2, num_keypoints)`):
                Scores of predicted matches for each image pair
            all_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
                Tuple of `torch.FloatTensor` (one for the output of each stage) of shape `(1, 2, num_keypoints,
                num_channels)`.
            all_attentions (`tuple(torch.FloatTensor)`, *optional*):
                Tuple of `torch.FloatTensor` (one for each layer) of shape `(1, 2, num_heads, num_keypoints,
                num_keypoints)`.
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if keypoints.shape[2] == 0:  # no keypoints
            shape = keypoints.shape[:-1]
            return (
                keypoints.new_full(shape, -1, dtype=torch.int),
                keypoints.new_zeros(shape),
                all_hidden_states,
                all_attentions,
            )

        batch_size, _, num_keypoints, _ = keypoints.shape
        # (batch_size, 2, num_keypoints, 2) -> (batch_size * 2, num_keypoints, 2)
        keypoints = keypoints.reshape(batch_size * 2, num_keypoints, 2)
        descriptors = descriptors.reshape(batch_size * 2, num_keypoints, self.config.hidden_size)
        scores = scores.reshape(batch_size * 2, num_keypoints)
        mask = mask.reshape(batch_size * 2, num_keypoints) if mask is not None else None

        # Keypoint normalization
        keypoints = normalize_keypoints(keypoints, height, width)

        encoded_keypoints = self.keypoint_encoder(keypoints, scores, output_hidden_states=output_hidden_states)

        last_hidden_state = encoded_keypoints[0]

        # Keypoint MLP encoder.
        descriptors = descriptors + last_hidden_state

        if mask is not None:
            input_shape = descriptors.size()
            extended_attention_mask = self.get_extended_attention_mask(mask, input_shape)
        else:
            extended_attention_mask = torch.ones((batch_size, num_keypoints), device=keypoints.device)

        # Multi-layer Transformer network.
        gnn_outputs = self.gnn(
            descriptors,
            mask=extended_attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        descriptors = gnn_outputs[0]

        # Final MLP projection.
        projected_descriptors = self.final_projection(descriptors)

        # (batch_size * 2, num_keypoints, descriptor_dim) -> (batch_size, 2, num_keypoints, descriptor_dim)
        final_descriptors = projected_descriptors.reshape(batch_size, 2, num_keypoints, self.config.hidden_size)
        final_descriptors0 = final_descriptors[:, 0]
        final_descriptors1 = final_descriptors[:, 1]

        # Compute matching descriptor distance.
        scores = final_descriptors0 @ final_descriptors1.transpose(1, 2)
        scores = scores / self.config.hidden_size**0.5

        if mask is not None:
            mask = mask.reshape(batch_size, 2, num_keypoints)
            mask0 = mask[:, 0].unsqueeze(-1).expand(-1, -1, num_keypoints)
            scores = scores.masked_fill(mask0 == 0, -1e9)

        # Run the optimal transport.
        scores = log_optimal_transport(scores, self.bin_score, iterations=self.config.sinkhorn_iterations)

        # Get the matches with score above "match_threshold".
        max0 = scores[:, :-1, :-1].max(2)
        max1 = scores[:, :-1, :-1].max(1)
        indices0 = max0.indices
        indices1 = max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        matching_scores0 = torch.where(mutual0, max0.values.exp(), zero)
        matching_scores0 = torch.where(matching_scores0 > self.config.matching_threshold, matching_scores0, zero)
        matching_scores1 = torch.where(mutual1, matching_scores0.gather(1, indices1), zero)
        valid0 = mutual0 & (matching_scores0 > zero)
        valid1 = mutual1 & valid0.gather(1, indices1)
        matches0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        matches1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        matches = torch.cat([matches0, matches1], dim=1).reshape(batch_size, 2, -1)
        matching_scores = torch.cat([matching_scores0, matching_scores1], dim=1).reshape(batch_size, 2, -1)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + encoded_keypoints[1]
            all_hidden_states = all_hidden_states + gnn_outputs[1]
            all_hidden_states = all_hidden_states + (projected_descriptors,)
            all_hidden_states = tuple(
                x.reshape(batch_size, 2, num_keypoints, -1).transpose(-1, -2) for x in all_hidden_states
            )
        if output_attentions:
            all_attentions = all_attentions + gnn_outputs[2]
            all_attentions = tuple(x.reshape(batch_size, 2, -1, num_keypoints, num_keypoints) for x in all_attentions)

        return (
            matches,
            matching_scores,
            all_hidden_states,
            all_attentions,
        )

    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, KeypointMatchingOutput]:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModel
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/assets/phototourism_sample_images/london_bridge_78916675_4568141288.jpg?raw=true"
        >>> image1 = Image.open(requests.get(url, stream=True).raw)
        >>> url = "https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/assets/phototourism_sample_images/london_bridge_19481797_2295892421.jpg?raw=true"
        >>> image2 = Image.open(requests.get(url, stream=True).raw)
        >>> images = [image1, image2]

        >>> processor = AutoImageProcessor.from_pretrained("magic-leap-community/superglue_outdoor")
        >>> model = AutoModel.from_pretrained("magic-leap-community/superglue_outdoor")

        >>> with torch.no_grad():
        >>>     inputs = processor(images, return_tensors="pt")
        >>>     outputs = model(**inputs)
        ```"""
        loss = None
        if labels is not None:
            raise ValueError("SuperGlue is not trainable, no labels should be provided.")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values.ndim != 5 or pixel_values.size(1) != 2:
            raise ValueError("Input must be a 5D tensor of shape (batch_size, 2, num_channels, height, width)")

        batch_size, _, channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(batch_size * 2, channels, height, width)
        keypoint_detections = self.keypoint_detector(pixel_values)

        keypoints, scores, descriptors, mask = keypoint_detections[:4]
        keypoints = keypoints.reshape(batch_size, 2, -1, 2).to(pixel_values)
        scores = scores.reshape(batch_size, 2, -1).to(pixel_values)
        descriptors = descriptors.reshape(batch_size, 2, -1, self.config.hidden_size).to(pixel_values)
        mask = mask.reshape(batch_size, 2, -1)

        absolute_keypoints = keypoints.clone()
        absolute_keypoints[:, :, :, 0] = absolute_keypoints[:, :, :, 0] * width
        absolute_keypoints[:, :, :, 1] = absolute_keypoints[:, :, :, 1] * height

        matches, matching_scores, hidden_states, attentions = self._match_image_pair(
            absolute_keypoints,
            descriptors,
            scores,
            height,
            width,
            mask=mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if not return_dict:
            return tuple(
                v
                for v in [loss, matches, matching_scores, keypoints, mask, hidden_states, attentions]
                if v is not None
            )

        return KeypointMatchingOutput(
            loss=loss,
            matches=matches,
            matching_scores=matching_scores,
            keypoints=keypoints,
            mask=mask,
            hidden_states=hidden_states,
            attentions=attentions,
        )


__all__ = ["SuperGluePreTrainedModel", "SuperGlueForKeypointMatching"]
