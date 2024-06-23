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
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from transformers import PreTrainedModel, add_start_docstrings
from transformers.models.superglue.configuration_superglue import SuperGlueConfig

from ...utils import ModelOutput, add_start_docstrings_to_model_forward, logging
from ..auto import AutoModelForKeypointDetection


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC_ = "SuperGlueConfig"

_CHECKPOINT_FOR_DOC_ = "stevenbucaille/superglue_indoor"


def concat_attentions_tuples_pair(
    attention_probs_0: Tuple[torch.Tensor], attention_probs_1: Tuple[torch.Tensor]
) -> Tuple[torch.Tensor]:
    """
    Concatenate two tuple of attention probabilities into one.
    We assume that the attention probabilities are of shape (batch_size, num_heads, num_keypoints_a, num_keypoints_b).
    The tuples are assumed to have the same length and of the following form:
    attention_probs_0 = (attention_probs_0_0, attention_probs_0_1, ...)
    attention_probs_1 = (attention_probs_1_0, attention_probs_1_1, ...)
    The output will then be of the form:
    new_attention_probs = (torch.cat(attention_probs_0_0, attention_probs_1_0), torch.cat(attention_probs_0_1,
    attention_probs_1_1), ...)
    If the attention probabilities have different shapes, the smaller one will be padded with zeros :
    (batch_size * 2, num_heads, max(num_keypoints_a_0, num_keypoints_a_1), max(num_keypoints_b_0, num_keypoints_b_1))
    """
    new_attention_probs = ()
    for attention_prob_0, attention_prob_1 in zip(attention_probs_0, attention_probs_1):
        if attention_prob_0.size() != attention_prob_1.size():
            max_dim2 = max(attention_prob_0.shape[2], attention_prob_1.shape[2])
            max_dim3 = max(attention_prob_0.shape[3], attention_prob_1.shape[3])
            new_attention_prob = torch.zeros(
                2, attention_prob_0.shape[1], max_dim2, max_dim3, device=attention_prob_0.device
            )
            new_attention_prob[0, :, : attention_prob_0.shape[2], : attention_prob_0.shape[3]] = attention_prob_0
            new_attention_prob[1, :, : attention_prob_1.shape[2], : attention_prob_1.shape[3]] = attention_prob_1
            new_attention_probs = new_attention_probs + (new_attention_prob,)
        else:
            new_attention_probs = new_attention_probs + (torch.cat([attention_prob_0, attention_prob_1]),)
    return new_attention_probs


def stack_attention_probs_list(attention_probs: List[torch.Tensor]) -> torch.Tensor:
    current_shape = attention_probs[0].shape
    all_same_shape = all(attention_prob.shape == current_shape for attention_prob in attention_probs)
    if all_same_shape:
        return torch.stack(attention_probs, dim=0)

    max_dim2 = max(attention_prob.shape[2] for attention_prob in attention_probs)
    max_dim3 = max(attention_prob.shape[3] for attention_prob in attention_probs)
    stacked_attention_probs = torch.zeros(
        len(attention_probs), 2, attention_probs[0].shape[1], max_dim2, max_dim3, device=attention_probs[0].device
    )
    for i, attention_prob in enumerate(attention_probs):
        stacked_attention_probs[i, :, :, : attention_prob.shape[2], : attention_prob.shape[3]] = attention_prob
    return stacked_attention_probs


def batch_attention_probs_list(
    attention_probs: Union[List[torch.Tensor], List[Tuple[torch.Tensor]]],
) -> Union[List[torch.Tensor], List[Tuple[torch.Tensor]]]:
    """
    Given a list of attention probabilities, batch them together.
    We assume that the attention probabilities are of shape (batch_size, num_heads, num_keypoints_a, num_keypoints_b).
    The list must be in the following form :
    - List of attention probabilities: [attention_probs_0, attention_probs_1, ...]
    We stack the attention probabilities along the batch dimension for each tuple:
    -> [torch.stack([attention_probs_0_0, attention_probs_1_0, ...], dim=0), torch.stack([attention_probs_0_1,
    attention_probs_1_1, ...], dim=0), ...]
    """

    list_of_tuples = list(zip(*map(list, attention_probs)))
    return [stack_attention_probs_list(element) for element in list_of_tuples]


def concat_hidden_states_tuples_pair(
    hidden_states_0: Tuple[torch.Tensor], hidden_states_1: Tuple[torch.Tensor]
) -> Tuple[torch.Tensor]:
    """
    Concatenate two tuple of hidden states into one.
    We assume that the hidden states are of shape (batch_size, hidden_state_size, num_keypoints).
    The tuples are assumed to have the same length and of the following form:
    hidden_states_0 = (hidden_state_0_0, hidden_state_0_1, ...)
    hidden_states_1 = (hidden_state_1_0, hidden_state_1_1, ...)
    The output will then be of the form:
    new_hidden_states = (torch.cat(hidden_state_0_0, hidden_state_1_0), torch.cat(hidden_state_0_1, hidden_state_1_1),
    ...)
    If the number of keypoints are different among hidden_states, the smaller one will be padded with zeros :
    (batch_size * 2, hidden_state_size, max(num_keypoints_0, num_keypoints_1))
    """
    hidden_states = ()
    for hidden_state_0, hidden_state_1 in zip(hidden_states_0, hidden_states_1):
        if hidden_state_0.shape != hidden_state_1.shape:
            max_num_keypoints = max(hidden_state_0.shape[2], hidden_state_1.shape[2])
            new_hidden_state = torch.zeros(2, hidden_state_0.shape[1], max_num_keypoints, device=hidden_state_0.device)
            new_hidden_state[0, :, : hidden_state_0.shape[2]] = hidden_state_0
            new_hidden_state[1, :, : hidden_state_1.shape[2]] = hidden_state_1
            hidden_states = hidden_states + (new_hidden_state,)
        else:
            hidden_states = hidden_states + (torch.cat([hidden_state_0, hidden_state_1]),)
    return hidden_states


def stack_hidden_states_list(hidden_states: List[torch.Tensor]) -> torch.Tensor:
    """
    Given a list of hidden states tensors, stack them together using torch.stack.
    We assume that the hidden states are of shape (batch_size, hidden_state_size, num_keypoints).
    If all hidden states have the same shape, we stack them along the batch dimension:
    [hidden_state_0, hidden_state_1, ...] -> torch.stack([hidden_state_0, hidden_state_1, ...], dim=0)
    If the hidden states have different shapes, the smaller ones will be padded with zeros:
    (batch_size * 2, hidden_state_size, max(num_keypoints_0, num_keypoints_1))
    """
    current_shape = hidden_states[0].shape
    all_same_shape = all(hidden_state.shape == current_shape for hidden_state in hidden_states)
    if all_same_shape:
        return torch.stack(hidden_states, dim=0)

    max_num_keypoints = max(hidden_state.shape[1] for hidden_state in hidden_states)
    stacked_hidden_state = torch.zeros(
        len(hidden_states),
        2,
        max_num_keypoints,
        hidden_states[0].shape[2],
        device=hidden_states[0].device,
    )
    for i, hidden_state in enumerate(hidden_states):
        stacked_hidden_state[i, :, : hidden_state.shape[1], :] = hidden_state
    return stacked_hidden_state


def batch_hidden_states(
    hidden_states: Union[List[torch.Tensor], List[Tuple[torch.Tensor]]],
) -> Union[List[torch.Tensor], List[Tuple[torch.Tensor]]]:
    """
    Given a list of hidden states, batch them together using torch.stack.
    We assume that the hidden states are of shape (batch_size, hidden_state_size, num_keypoints).
    The list must be in the following form :
    [(hidden_state_0_0, hidden_state_1_0, ...), (hidden_state_0_1, hidden_state_1_1, ...), ...]
    We stack the hidden states along the batch dimension for each tuple:
    -> [torch.stack([hidden_state_0_0, hidden_state_1_0, ...], dim=0), torch.stack([hidden_state_0_1, hidden_state_1_1,
    ...], dim=0), ...]
    """
    list_of_tuples = list(zip(*map(list, hidden_states)))
    return [stack_hidden_states_list(element) for element in list_of_tuples]


def normalize_keypoints(keypoints: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Normalize keypoints locations based on image image_shape

    Args:
        keypoints (`torch.Tensor` of shape `(batch_size, num_keypoints, 2)`): Keypoints locations.
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
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
        when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of each stage) of shape `(batch_size, 2, num_channels,
            num_keypoints)`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, 2, num_heads, num_keypoints,
            num_keypoints)`.
    """

    loss: Optional[torch.FloatTensor] = None
    mask: torch.FloatTensor = None
    matches: torch.FloatTensor = None
    matching_scores: torch.FloatTensor = None
    keypoints: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SuperGlueMultiLayerPerceptron(nn.Module):
    def __init__(self, config: SuperGlueConfig, channels: List[int]) -> None:
        super().__init__()
        num_layers = len(channels)
        layers = []
        for i in range(1, num_layers):
            layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
            if i < (num_layers - 1):
                layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(
        self, input: torch.Tensor, output_hidden_states: Optional[bool] = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            input = layer(input)
            if output_hidden_states and isinstance(layer, nn.Conv1d):
                all_hidden_states = all_hidden_states + (input,)
        output = input
        return output, all_hidden_states


class SuperGlueKeypointEncoder(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        layer_sizes = config.keypoint_encoder_sizes
        feature_dim = config.descriptor_dim
        self.encoder = SuperGlueMultiLayerPerceptron(config, [3] + layer_sizes + [feature_dim])

    def forward(
        self,
        keypoints: torch.Tensor,
        scores: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        keypoints = keypoints.transpose(1, 2)
        scores = scores.unsqueeze(1)
        inputs = torch.cat([keypoints, scores], dim=1)
        return self.encoder(inputs, output_hidden_states=output_hidden_states)


class SuperGlueMultiHeadAttention(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        feature_dim = config.descriptor_dim
        self.num_heads = config.num_heads
        self.head_dim = feature_dim // self.num_heads
        self.merge = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        self.proj = nn.ModuleList([nn.Conv1d(feature_dim, feature_dim, kernel_size=1) for _ in range(3)])

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        batch_dim = query.size(0)
        query, key, value = [
            layer(x).view(batch_dim, self.head_dim, self.num_heads, -1)
            for layer, x in zip(self.proj, (query, key, value))
        ]
        query_dim = query.shape[1]
        scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / query_dim**0.5
        attention_probs = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.einsum("bhnm,bdhm->bdhn", attention_probs, value)
        output = self.merge(output.contiguous().view(batch_dim, self.head_dim * self.num_heads, -1))

        output = (output, attention_probs) if output_attentions else (output,)

        return output


class SuperGlueAttentionalPropagation(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        descriptor_dim = config.descriptor_dim
        self.attention = SuperGlueMultiHeadAttention(config)
        self.mlp = SuperGlueMultiLayerPerceptron(
            config,
            [descriptor_dim * 2, descriptor_dim * 2, descriptor_dim],
        )

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[Tuple[torch.Tensor]]]:
        attention_outputs = self.attention(x, source, source, output_attentions=output_attentions)
        output = attention_outputs[0]
        attention = attention_outputs[1:]

        output = torch.cat([x, output], dim=1)
        layer_outputs = self.mlp(output, output_hidden_states=output_hidden_states)

        last_hidden_state = layer_outputs[0]
        hidden_states = layer_outputs[1]

        return last_hidden_state, hidden_states, attention


class SuperGlueAttentionalGNN(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        self.descriptor_dim = config.descriptor_dim
        self.layers_types = config.gnn_layers_types
        self.layers = nn.ModuleList([SuperGlueAttentionalPropagation(config) for _ in range(len(self.layers_types))])

    def forward(
        self,
        descriptors_0: torch.Tensor,
        descriptors_1: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple], Optional[Tuple]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if output_hidden_states:
            same_num_keypoints = descriptors_0.shape[-1] == descriptors_1.shape[-1]
            if not same_num_keypoints:
                max_num_keypoints = max(descriptors_0.shape[-1], descriptors_1.shape[-1])
                new_hidden_state = torch.zeros((2, self.descriptor_dim, max_num_keypoints)).to(descriptors_0.device)
                new_hidden_state[:, :, : descriptors_0.shape[-1]] = descriptors_0
                new_hidden_state[:, :, : descriptors_1.shape[-1]] = descriptors_1
            else:
                new_hidden_state = torch.cat([descriptors_0, descriptors_1])
            all_hidden_states = all_hidden_states + (new_hidden_state,)

        for gnn_layer, layer_type in zip(self.layers, self.layers_types):
            if layer_type == "cross":
                source_0, source_1 = descriptors_1, descriptors_0
            else:  # if type == 'self':
                source_0, source_1 = descriptors_0, descriptors_1

            gnn_outputs0 = gnn_layer(
                descriptors_0, source_0, output_hidden_states=output_hidden_states, output_attentions=output_attentions
            )
            gnn_outputs1 = gnn_layer(
                descriptors_1, source_1, output_hidden_states=output_hidden_states, output_attentions=output_attentions
            )

            delta0 = gnn_outputs0[0]
            delta1 = gnn_outputs1[0]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + concat_hidden_states_tuples_pair(
                    gnn_outputs0[1], gnn_outputs1[1]
                )
            if output_attentions:
                all_attentions = all_attentions + concat_attentions_tuples_pair(gnn_outputs0[2], gnn_outputs1[2])

            descriptors_0 = descriptors_0 + delta0
            descriptors_1 = descriptors_1 + delta1
        return descriptors_0, descriptors_1, all_hidden_states, all_attentions


class SuperGlueFinalProjection(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        descriptor_dim = config.descriptor_dim
        self.final_proj = nn.Conv1d(descriptor_dim, descriptor_dim, kernel_size=1, bias=True)

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
    supports_gradient_checkpointing = False

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm, nn.Conv1d]) -> None:
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
        elif isinstance(module, SuperGlueMultiLayerPerceptron):
            nn.init.constant_(module.layers[-1].bias, 0.0)


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

    def match_image_pair(
        self,
        image0_keypoints: torch.Tensor,
        image0_descriptors: torch.Tensor,
        image0_scores: torch.Tensor,
        image1_keypoints: torch.Tensor,
        image1_descriptors: torch.Tensor,
        image1_scores: torch.Tensor,
        height: int,
        width: int,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple, Tuple]:
        """
        Perform keypoint matching between two images.
        Args:
            image0_keypoints (`torch.Tensor` of shape `(1, num_keypoints, 2)`):
                Keypoints detected in the first image.
            image0_descriptors (`torch.Tensor` of shape `(1, descriptor_dim, num_keypoints)`):
                Descriptors of the keypoints detected in the first image.
            image0_scores (`torch.Tensor` of shape `(1, num_keypoints)`):
                Confidence scores of the keypoints detected in the first image.
            image1_keypoints (`torch.Tensor` of shape `(1, num_keypoints, 2)`):
                Keypoints detected in the second image.
            image1_descriptors (`torch.Tensor` of shape `(1, descriptor_dim, num_keypoints)`):
                Descriptors of the keypoints detected in the second image.
            image1_scores (`torch.Tensor` of shape `(1, num_keypoints)`):
                Confidence scores of the keypoints detected in the second image.
            height (`int`): Image height.
            width (`int`): Image width.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors. Default to `config.output_attentions`.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. Default to `config.output_hidden_states`.
        Returns:
            matches_0 (`torch.Tensor` of shape `(1, num_keypoints)`):
                Index of keypoint matched in the second image.
            matches_1 (`torch.Tensor` of shape `(1, num_keypoints)`):
                Index of keypoint matched in the first image.
            matching_scores_0 (`torch.Tensor` of shape `(1, num_keypoints)`):
                Scores of predicted matches from the first image.
            matching_scores_1 (`torch.Tensor` of shape `(1, num_keypoints)`):
                Scores of predicted matches from the second image.
            all_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
                Tuple of `torch.FloatTensor` (one for the output of each stage) of shape `(1, 2, num_keypoints,
                num_channels)`.
            all_attentions (`tuple(torch.FloatTensor)`, *optional*):
                Tuple of `torch.FloatTensor` (one for each layer) of shape `(1, 2, num_heads, num_keypoints,
                num_keypoints)`.
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if image0_keypoints.shape[1] == 0 or image1_keypoints.shape[1] == 0:  # no keypoints
            shape0, shape1 = image0_keypoints.shape[:-1], image1_keypoints.shape[:-1]
            return (
                image0_keypoints.new_full(shape0, -1, dtype=torch.int),
                image1_keypoints.new_full(shape1, -1, dtype=torch.int),
                image0_keypoints.new_zeros(shape0),
                image1_keypoints.new_zeros(shape1),
                all_hidden_states,
                all_attentions,
            )

        descriptors_0 = torch.transpose(image0_descriptors, -1, -2)
        descriptors_1 = torch.transpose(image1_descriptors, -1, -2)

        # Keypoint normalization
        keypoints0 = normalize_keypoints(image0_keypoints, height, width)
        keypoints1 = normalize_keypoints(image1_keypoints, height, width)

        encoded_keypoints0 = self.keypoint_encoder(
            keypoints0, image0_scores, output_hidden_states=output_hidden_states
        )
        encoded_keypoints1 = self.keypoint_encoder(
            keypoints1, image1_scores, output_hidden_states=output_hidden_states
        )

        last_hidden_state_0 = encoded_keypoints0[0]
        last_hidden_state_1 = encoded_keypoints1[0]

        # Keypoint MLP encoder.
        descriptors_0 = descriptors_0 + last_hidden_state_0
        descriptors_1 = descriptors_1 + last_hidden_state_1

        # Multi-layer Transformer network.
        gnn_outputs = self.gnn(
            descriptors_0,
            descriptors_1,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        descriptors_0, descriptors_1 = gnn_outputs[:2]

        # Final MLP projection.
        projected_descriptors_0 = self.final_projection(descriptors_0)
        projected_descriptors_1 = self.final_projection(descriptors_1)

        # Compute matching descriptor distance.
        scores = torch.einsum("bdn,bdm->bnm", projected_descriptors_0, projected_descriptors_1)
        scores = scores / self.config.descriptor_dim**0.5

        # Run the optimal transport.
        scores = log_optimal_transport(scores, self.bin_score, iterations=self.config.sinkhorn_iterations)

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        matching_scores_0 = torch.where(mutual0, max0.values.exp(), zero)
        matching_scores_1 = torch.where(mutual1, matching_scores_0.gather(1, indices1), zero)
        valid0 = mutual0 & (matching_scores_0 > self.config.matching_threshold)
        valid1 = mutual1 & valid0.gather(1, indices1)
        matches_0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        matches_1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        if output_hidden_states:
            all_hidden_states = all_hidden_states + concat_hidden_states_tuples_pair(
                encoded_keypoints0[1], encoded_keypoints1[1]
            )
            all_hidden_states = all_hidden_states + gnn_outputs[2]
            all_hidden_states = all_hidden_states + concat_hidden_states_tuples_pair(
                (projected_descriptors_0,), (projected_descriptors_1,)
            )
            transposed_all_hidden_states = ()
            for i in range(len(all_hidden_states)):
                # From (batch_size, descriptor_dim, num_keypoints) to (batch_size, num_keypoints, descriptor_dim)
                transposed_all_hidden_states = transposed_all_hidden_states + (all_hidden_states[i].transpose(-1, -2),)
            all_hidden_states = transposed_all_hidden_states
        if output_attentions:
            all_attentions = all_attentions + gnn_outputs[3]

        return (
            matches_0,
            matches_1,
            matching_scores_0,
            matching_scores_1,
            all_hidden_states,
            all_attentions,
        )

    def batch_output(
        self,
        batch_size,
        list_attentions,
        list_hidden_states,
        list_keypoints_0,
        list_keypoints_1,
        list_matches_0,
        list_matches_1,
        list_matching_scores_0,
        list_matching_scores_1,
        pixel_values,
        output_attentions,
        output_hidden_states,
    ) -> Tuple[Optional[Tuple], Optional[Tuple], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        maximum_matches = max(
            max([matches_0.shape[1] for matches_0 in list_matches_0]),
            max([matches_1.shape[1] for matches_1 in list_matches_1]),
        )
        matches = torch.full((batch_size, 2, maximum_matches), -1, device=pixel_values.device, dtype=torch.int)
        matching_scores = torch.zeros((batch_size, 2, maximum_matches), device=pixel_values.device)
        matches_mask = torch.zeros((batch_size, 2, maximum_matches), device=pixel_values.device, dtype=torch.int)
        keypoints = torch.zeros((batch_size, 2, maximum_matches, 2), device=pixel_values.device)
        for i, (
            _matches_0,
            _matches_1,
            _matching_scores_0,
            _matching_scores_1,
            _keypoints_0,
            _keypoints_1,
        ) in enumerate(
            zip(
                list_matches_0,
                list_matches_1,
                list_matching_scores_0,
                list_matching_scores_1,
                list_keypoints_0,
                list_keypoints_1,
            )
        ):
            matches[i, 0, : _matches_0.shape[1]] = _matches_0
            matches[i, 1, : _matches_1.shape[1]] = _matches_1
            matching_scores[i, 0, : _matching_scores_0.shape[1]] = _matching_scores_0
            matching_scores[i, 1, : _matching_scores_1.shape[1]] = _matching_scores_1
            matches_mask[i, 0, : _matches_0.shape[1]] = 1
            matches_mask[i, 1, : _matches_1.shape[1]] = 1
            keypoints[i, 0, : _keypoints_0.shape[1], :] = _keypoints_0
            keypoints[i, 1, : _keypoints_1.shape[1], :] = _keypoints_1

        if output_hidden_states:
            hidden_states = batch_hidden_states(list_hidden_states)
        else:
            hidden_states = None
        if output_attentions:
            attentions = batch_attention_probs_list(list_attentions)
        else:
            attentions = None
        return attentions, hidden_states, keypoints, matches, matches_mask, matching_scores

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
        >>> im1 = Image.open(requests.get(url, stream=True).raw)
        >>> url = "https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/assets/phototourism_sample_images/london_bridge_19481797_2295892421.jpg?raw=true"
        >>> im2 = Image.open(requests.get(url, stream=True).raw)
        >>> images = [im1, im2]

        >>> processor = AutoImageProcessor.from_pretrained("stevenbucaille/superglue_outdoor")
        >>> model = AutoModel.from_pretrained("stevenbucaille/superglue_outdoor")

        >>> inputs = processor(images, return_tensors="pt")
        >>> outputs = model(**inputs)
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

        list_matches_0 = []
        list_matches_1 = []
        list_matching_scores_0 = []
        list_matching_scores_1 = []
        list_keypoints_0 = []
        list_keypoints_1 = []
        list_hidden_states = []
        list_attentions = []

        for image_pair in pixel_values:
            keypoint_detection_output = self.keypoint_detector(
                image_pair,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            keypoints, scores, descriptors, mask = keypoint_detection_output[:4]

            image0_indices = torch.nonzero(mask[0]).squeeze()
            image0_keypoints = torch.unsqueeze(keypoints[0][image0_indices], dim=0)
            image0_descriptors = torch.unsqueeze(descriptors[0][image0_indices], dim=0)
            image0_scores = torch.unsqueeze(scores[0][image0_indices], dim=0)
            image1_indices = torch.nonzero(mask[1]).squeeze()
            image1_keypoints = torch.unsqueeze(keypoints[1][image1_indices], dim=0)
            image1_descriptors = torch.unsqueeze(descriptors[1][image1_indices], dim=0)
            image1_scores = torch.unsqueeze(scores[1][image1_indices], dim=0)

            match_image_output = self.match_image_pair(
                image0_keypoints,
                image0_descriptors,
                image0_scores,
                image1_keypoints,
                image1_descriptors,
                image1_scores,
                height,
                width,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            matches_0, matches_1, matching_scores_0, matching_scores_1, hidden_states, attentions = match_image_output
            list_matches_0.append(matches_0)
            list_matches_1.append(matches_1)
            list_matching_scores_0.append(matching_scores_0)
            list_matching_scores_1.append(matching_scores_1)
            list_keypoints_0.append(image0_keypoints)
            list_keypoints_1.append(image1_keypoints)
            list_hidden_states.append(hidden_states)
            list_attentions.append(attentions)

        attentions, hidden_states, keypoints, matches, matches_mask, matching_scores = self.batch_output(
            batch_size,
            list_attentions,
            list_hidden_states,
            list_keypoints_0,
            list_keypoints_1,
            list_matches_0,
            list_matches_1,
            list_matching_scores_0,
            list_matching_scores_1,
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if not return_dict:
            return tuple(
                v
                for v in [loss, matches_mask, matches, matching_scores, keypoints, hidden_states, attentions]
                if v is not None
            )

        return KeypointMatchingOutput(
            loss=loss,
            mask=matches_mask,
            matches=matches,
            matching_scores=matching_scores,
            keypoints=keypoints,
            hidden_states=hidden_states,
            attentions=attentions,
        )
