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

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn

from transformers import PreTrainedModel, add_start_docstrings
from transformers.models.superglue.configuration_superglue import SuperGlueConfig

from ...utils import ModelOutput, add_start_docstrings_to_model_forward, logging
from ..auto import AutoModelForKeypointDetection


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC_ = "SuperGlueConfig"

_CHECKPOINT_FOR_DOC_ = "stevenbucaille/superglue_indoor"


def concat_pairs(tensor_tuple0: Tuple[torch.Tensor], tensor_tuple1: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
    """
    Concatenate two tuples of tensors pairwise

    Args:
        tensor_tuple0 (`Tuple[torch.Tensor]`): Tuple of tensors.
        tensor_tuple1 (`Tuple[torch.Tensor]`): Tuple of tensors.

    Returns:
        (`Tuple[torch.Tensor]`): Tuple of concatenated tensors.
    """
    return tuple([torch.cat([tensor0, tensor1]) for tensor0, tensor1 in zip(tensor_tuple0, tensor_tuple1)])


def normalize_keypoints(keypoints: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Normalize keypoints locations based on image image_shape

    Args:
        keypoints (`torch.Tensor` of shape `(batch_size, num_keypoints, 2)`): Keypoints locations in (x, y) format.
        height (`int`): Image height.
        width (`int`): Image width.
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
        log_cost_matrix (`torch.Tensor` of shape `(batch_size, num_rows, num_columns)`): Logarithm of the cost matrix.
        log_source_distribution (`torch.Tensor` of shape `(batch_size, num_rows)`): Logarithm of the source
        distribution.
        log_target_distribution (`torch.Tensor` of shape `(batch_size, num_columns)`): Logarithm of the target
        distribution.

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
        scores: (`torch.Tensor` of shape `(batch_size, num_rows, num_columns)`): Cost matrix.
        reg_param: (`torch.Tensor` of shape `(batch_size, 1, 1)`): Regularization parameter.
        iterations: (`int`): Number of Sinkhorn iterations.

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
class KeypointMatchingOutput(ModelOutput):
    """
    Base class for outputs of keypoint matching models. Due to the nature of keypoint detection and matching, the number
    of keypoints is not fixed and can vary from image to image, which makes batching non-trivial. In the batch of
    images, the maximum number of matches is set as the dimension of the matches and matching scores. The mask tensor is
    used to indicate which values in the keypoints, matches and matching_scores tensors are keypoint matching
    information.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Loss computed during training.
        mask (`torch.BoolTensor` of shape `(batch_size, num_keypoints)`):
            Mask indicating which values in matches and matching_scores are keypoint matching information.
        matches (`torch.FloatTensor` of shape `(batch_size, 2, num_matches)`):
            Index of keypoint matched in the other image.
        matching_scores (`torch.FloatTensor` of shape `(batch_size, 2, num_matches)`):
            Scores of predicted matches.
        keypoints (`torch.FloatTensor` of shape `(batch_size, num_keypoints, 2)`):
            Absolute (x, y) coordinates of predicted keypoints in a given image.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*:
            Tuple of `torch.FloatTensor` (one for the output of each stage) of shape `(batch_size, 2, num_channels,
            num_keypoints)`, returned when `output_hidden_states=True` is passed or when
            `config.output_hidden_states=True`)
        attentions (`tuple(torch.FloatTensor)`, *optional*:
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, 2, num_heads, num_keypoints,
            num_keypoints)`, returned when `output_attentions=True` is passed or when `config.output_attentions=True`)
    """

    loss: Optional[torch.FloatTensor] = None
    mask: torch.FloatTensor = None
    matches: torch.FloatTensor = None
    matching_scores: torch.FloatTensor = None
    keypoints: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SuperGlueMultiLayerPerceptron(nn.Module):
    def __init__(self, config: SuperGlueConfig, in_channels: int, out_channels: int, activate: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.activate = activate
        if self.activate:
            self.batch_norm = nn.BatchNorm1d(out_channels)
            self.activation = nn.ReLU()

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.linear(hidden_state)
        if self.activate:
            hidden_state = hidden_state.transpose(-1, -2)
            hidden_state = self.batch_norm(hidden_state)
            hidden_state = hidden_state.transpose(-1, -2)
            hidden_state = self.activation(hidden_state)
        return hidden_state


# class SuperGlueMultiLayerPerceptron(nn.Module):
#     def __init__(self, config: SuperGlueConfig, channels: List[int]) -> None:
#         super().__init__()
#         num_layers = len(channels)
#         layers = []
#         for i in range(1, num_layers):
#             layers.append(nn.Linear(channels[i - 1], channels[i]))
#             if i < (num_layers - 1):
#                 layers.append(nn.BatchNorm1d(channels[i]))
#                 layers.append(nn.ReLU())
#         self.layers = nn.Sequential(*layers)
#
#     def forward(
#         self, hidden_state: torch.Tensor, output_hidden_states: Optional[bool] = False
#     ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
#         all_hidden_states = () if output_hidden_states else None
#         for layer in self.layers:
#             if isinstance(layer, nn.BatchNorm1d):
#                 hidden_state = hidden_state.transpose(-1, -2)
#             hidden_state = layer(hidden_state)
#             if isinstance(layer, nn.BatchNorm1d):
#                 hidden_state = hidden_state.transpose(-1, -2)
#             if output_hidden_states and isinstance(layer, nn.Linear):
#                 all_hidden_states = all_hidden_states + (hidden_state,)
#         return hidden_state, all_hidden_states


class SuperGlueKeypointEncoder(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        layer_sizes = config.keypoint_encoder_sizes
        feature_dim = config.descriptor_dim
        # 3 here consists of 2 for the (x, y) coordinates and 1 for the score of the keypoint
        encoder_channels = [3] + layer_sizes + [feature_dim]
        layers = []
        for i in range(1, len(encoder_channels)):
            layers.append(
                SuperGlueMultiLayerPerceptron(
                    config, encoder_channels[i - 1], encoder_channels[i], i < len(encoder_channels) - 1
                )
            )
        self.encoder = nn.ModuleList(layers)

    def forward(
        self,
        keypoints: torch.Tensor,
        scores: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        scores = scores.unsqueeze(2)
        hidden_state = torch.cat([keypoints, scores], dim=2)
        all_hidden_states = () if output_hidden_states else None
        for layer in self.encoder:
            hidden_state = layer(hidden_state)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
        return hidden_state, all_hidden_states


class SuperGlueMultiHeadAttention(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        feature_dim = config.descriptor_dim
        self.num_heads = config.num_heads
        self.head_dim = feature_dim // self.num_heads

        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        batch_dim, seq_len, _ = query.size()

        query = self.q_proj(query).view(batch_dim, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(key).view(batch_dim, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(value).view(batch_dim, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
        output = torch.matmul(attention_probs, value)
        output = output.transpose(1, 2).contiguous().view(batch_dim, seq_len, -1)
        output = self.out_proj(output)

        output = (output, attention_probs) if output_attentions else (output,)

        return output


class SuperGlueAttentionalPropagation(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        descriptor_dim = config.descriptor_dim
        self.attention = SuperGlueMultiHeadAttention(config)
        mlp_channels = [descriptor_dim * 2, descriptor_dim * 2, descriptor_dim]
        layers = []
        for i in range(1, len(mlp_channels)):
            layers.append(
                SuperGlueMultiLayerPerceptron(config, mlp_channels[i - 1], mlp_channels[i], i < len(mlp_channels) - 1)
            )
        self.mlp = nn.ModuleList(layers)

    def forward(
        self,
        descriptors: torch.Tensor,
        source: torch.Tensor,
        mask: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[Tuple[torch.Tensor]]]:
        attention_outputs = self.attention(descriptors, source, source, mask, output_attentions=output_attentions)
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
        self.descriptor_dim = config.descriptor_dim
        self.layers_types = config.gnn_layers_types
        self.layers = nn.ModuleList([SuperGlueAttentionalPropagation(config) for _ in range(len(self.layers_types))])

    def forward(
        self,
        descriptors: torch.Tensor,
        mask: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple], Optional[Tuple]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if output_hidden_states:
            batch_size, _, num_keypoints, _ = descriptors.shape
            all_hidden_states = all_hidden_states + (
                descriptors.reshape(batch_size * 2, num_keypoints, self.descriptor_dim),
            )

        for gnn_layer, layer_type in zip(self.layers, self.layers_types):
            if layer_type == "cross":
                source0, source1 = descriptors[:, 1], descriptors[:, 0]
                mask0, mask1 = mask[:, 1], mask[:, 0]
            elif layer_type == "self":
                source0, source1 = descriptors[:, 0], descriptors[:, 1]
                mask0, mask1 = mask[:, 0], mask[:, 1]
            else:
                raise ValueError(f"Unrecognized layer type {layer_type}")

            gnn_outputs0 = gnn_layer(
                descriptors[:, 0],
                source0,
                mask0,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            gnn_outputs1 = gnn_layer(
                descriptors[:, 1],
                source1,
                mask1,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )

            delta0 = gnn_outputs0[0]
            delta1 = gnn_outputs1[0]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + concat_pairs(gnn_outputs0[1], gnn_outputs1[1])
            if output_attentions:
                all_attentions = all_attentions + concat_pairs(gnn_outputs0[2], gnn_outputs1[2])

            descriptors[:, 0] = descriptors[:, 0] + delta0
            descriptors[:, 1] = descriptors[:, 1] + delta1
        return descriptors, all_hidden_states, all_attentions


class SuperGlueFinalProjection(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        descriptor_dim = config.descriptor_dim
        self.final_proj = nn.Linear(descriptor_dim, descriptor_dim, bias=True)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        return self.final_proj(descriptors)


class SuperGluePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SuperGlueConfig
    base_model_prefix = "superglue"
    main_input_name = "pixel_values"

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # elif isinstance(module, SuperGlueMultiLayerPerceptron):
        #     nn.init.constant_(module.layers[-1].bias, 0.0)


SUPERGLUE_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SuperGlueConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

SUPERGLUE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`SuperGlueImageProcessor`]. See
            [`SuperGlueImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors. See `attentions` under returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "SuperGlue model taking images as inputs and outputting the matching of them.",
    SUPERGLUE_START_DOCSTRING,
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
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763
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
        mask: torch.Tensor = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple, Tuple]:
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
        scores = scores.reshape(batch_size * 2, num_keypoints)

        # Keypoint normalization
        keypoints = normalize_keypoints(keypoints, height, width)

        encoded_keypoints = self.keypoint_encoder(keypoints, scores, output_hidden_states=output_hidden_states)

        last_hidden_state = encoded_keypoints[0]

        # (batch_size * 2, num_keypoints, descriptor_dim) -> (batch_size, 2, num_keypoints, descriptor_dim)
        last_hidden_state = last_hidden_state.reshape(batch_size, 2, num_keypoints, self.config.descriptor_dim)

        # Keypoint MLP encoder.
        descriptors = descriptors + last_hidden_state

        # Multi-layer Transformer network.
        gnn_outputs = self.gnn(
            descriptors,
            mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        descriptors = gnn_outputs[0]

        # (batch_size, 2, num_keypoints, descriptor_dim) -> (batch_size * 2, num_keypoints, descriptor_dim)
        descriptors = descriptors.reshape(batch_size * 2, num_keypoints, self.config.descriptor_dim)

        # Final MLP projection.
        projected_descriptors = self.final_projection(descriptors)

        # (batch_size * 2, num_keypoints, descriptor_dim) -> (batch_size, 2, num_keypoints, descriptor_dim)
        final_descriptors = projected_descriptors.reshape(batch_size, 2, num_keypoints, self.config.descriptor_dim)
        final_descriptors0 = final_descriptors[:, 0]
        final_descriptors1 = final_descriptors[:, 1]

        # Compute matching descriptor distance.
        scores = torch.einsum("bnd,bmd->bnm", final_descriptors0, final_descriptors1)
        scores = scores / self.config.descriptor_dim**0.5

        if mask is not None:
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

        matches = torch.cat([matches0, matches1]).reshape(batch_size, 2, -1)
        matching_scores = torch.cat([matching_scores0, matching_scores1]).reshape(batch_size, 2, -1)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (encoded_keypoints[1])
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

    @add_start_docstrings_to_model_forward(SUPERGLUE_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, KeypointMatchingOutput]:
        """
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

        >>> processor = AutoImageProcessor.from_pretrained("stevenbucaille/superglue_outdoor")
        >>> model = AutoModel.from_pretrained("stevenbucaille/superglue_outdoor")

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

        if len(pixel_values.size()) != 5 or pixel_values.size(1) != 2:
            raise ValueError("Input must be a 5D tensor of shape (batch_size, 2, num_channels, height, width)")

        batch_size, _, channels, height, width = pixel_values.shape

        list_keypoint_detection = [self.keypoint_detector(image_pair) for image_pair in pixel_values]

        max_keypoints = max([keypoint_detection[0].shape[1] for keypoint_detection in list_keypoint_detection])
        keypoints = torch.zeros((batch_size, 2, max_keypoints, 2), device=pixel_values.device)
        scores = torch.zeros((batch_size, 2, max_keypoints), device=pixel_values.device)
        descriptors = torch.zeros(
            (batch_size, 2, max_keypoints, self.config.descriptor_dim), device=pixel_values.device
        )
        mask = torch.zeros((batch_size, 2, max_keypoints), device=pixel_values.device, dtype=torch.int)

        for i, keypoint_detection_output in enumerate(list_keypoint_detection):
            _keypoints, _scores, _descriptors, _mask = keypoint_detection_output[:4]
            keypoints[i, :, : _keypoints.shape[1], :] = _keypoints
            scores[i, :, : _scores.shape[1]] = _scores
            descriptors[i, :, : _descriptors.shape[1], :] = _descriptors
            mask[i, :, : _mask.shape[1]] = _mask

        matches, matching_scores, hidden_states, attentions = self._match_image_pair(
            keypoints,
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
                for v in [loss, mask, matches, matching_scores, keypoints, hidden_states, attentions]
                if v is not None
            )

        return KeypointMatchingOutput(
            loss=loss,
            mask=mask,
            matches=matches,
            matching_scores=matching_scores,
            keypoints=keypoints,
            hidden_states=hidden_states,
            attentions=attentions,
        )
