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


def concat_inconsistent_pairs(
    inconsistent_tensor0: Tuple[torch.Tensor], inconsistent_tensor1: Tuple[torch.Tensor]
) -> Tuple[torch.Tensor]:
    """
    Concatenate two tuples of tensors with inconsistent dimensions pairwise. We assume that each tuple has the same
    number of tensors and and the tensors are at least 3D. These tensors can be anything but this function is mainly
    used to concatenate hidden states and attention probabilities tensors of a keypoint matching model. The
    inconsistency comes from the dimension containing the number of keypoints.

    If the pair of tensors are hidden states, they are assumed to be of shape `(batch_size, num_keypoints0,
    num_channels)` and `(batch_size, num_keypoints1, num_channels)` and the output will then be of the form:
    `(2 * batch_size, max(num_keypoints0, num_keypoints1), num_channels)`
    If the pair of tensors are attention probabilities, they are assumed to be of shape `(batch_size, num_heads,
    num_keypoints0, num_keypoints0)` and `(batch_size, num_heads, num_keypoints1, num_keypoints1)` and the output will
    then be of the form:
    `(2 * batch_size, num_heads, max(num_keypoints0, num_keypoints1), max(num_keypoints0, num_keypoints1))`

    For `inconsistent_tensors0 = (tensor0_1, tensor0_1, ..., tensor0_N)` and `inconsistent_tensors1 = (tensor1_0,
    tensor1_1, ..., tensor1_N)`, the output will be `(concat(tensor0_0, tensor1_0), concat(tensor0_1, tensor1_1), ...,
    concat(tensor0_N, tensor1_N))`. The concatenation is done by padding the tensors from the tuple with the lower
    number of keypoints with zeros.

    Args:
        inconsistent_tensor0 (`Tuple[torch.Tensor]` of shape `(batch_size, num_keypoints0, num_channels)` or
        `torch.Tensor` of shape `(batch_size, num_heads, num_keypoints0, num_keypoints0)`): Tuple of tensors with
        inconsistent dimensions.
        inconsistent_tensor1 (`Tuple[torch.Tensor]` of shape `(batch_size, num_keypoints1, num_channels)` or
        `torch.Tensor` of shape `(batch_size, num_heads, num_keypoints1, num_keypoints1)`): Tuple of tensors with
        inconsistent dimensions.

    Returns:
        consistent_tensors (`Tuple[torch.Tensor]` of shape `(2 * batch_size, max(num_keypoints0, num_keypoints1),
        num_channels)` or `Tuple[torch.Tensor]` of shape `(2 * batch_size, num_heads,
        max(num_keypoints0, num_keypoints1) ,max(num_keypoints0, num_keypoints1))`):
        Tuple of zero padded tensors with consistent dimensions.
    """
    if len(inconsistent_tensor0) != len(inconsistent_tensor1):
        raise ValueError("The two tuples must contain the same number of tensors.")
    if len(inconsistent_tensor0[0].shape) < 3:
        raise ValueError("The tensors must be at least 3D.")
    consistent_tensors = ()
    for tensor0, tensor1 in zip(inconsistent_tensor0, inconsistent_tensor1):
        if tensor0.shape != tensor1.shape:
            squeeze_tensors = len(tensor0.shape) == 3 and len(tensor1.shape) == 3
            if squeeze_tensors:
                tensor0 = tensor0[..., None]
                tensor1 = tensor1[..., None]
            # max_dim1 = num_heads if tensor is attention probabilities else num_keypoints
            max_dim1 = max(tensor0.shape[1], tensor1.shape[1])
            # max_dim2 = max_dim3 = num_keypoints if tensor is attention probabilities else num_channels
            max_dim2 = max(tensor0.shape[2], tensor1.shape[2])
            # max_dim3 = num_keypoints if tensor is attention probabilities else 1
            max_dim3 = max(tensor0.shape[3], tensor1.shape[3])
            consistent_tensor = torch.zeros(2, max_dim1, max_dim2, max_dim3, device=tensor0.device)
            consistent_tensor[0, : tensor0.shape[1], : tensor0.shape[2], : tensor0.shape[3]] = tensor0
            consistent_tensor[1, : tensor1.shape[1], : tensor1.shape[2], : tensor1.shape[3]] = tensor1
            if squeeze_tensors:
                consistent_tensor = consistent_tensor.squeeze(-1)
            consistent_tensors = consistent_tensors + (consistent_tensor,)
        else:
            consistent_tensors = consistent_tensors + (torch.cat([tensor0, tensor1]),)
    return consistent_tensors


def stack_inconsistent_tensor_list(tensor_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Stack a list of tensors with inconsistent dimensions. We assume that each tensor is at least 3D. The tensors can be
    anything but this function is mainly used to stack hidden states tensors of a keypoint matching model. The
    inconsistency comes from the dimension containing the number of keypoints.

    If the tensors are hidden states, they are assumed to be of shape `(batch_size, num_keypoints0, num_channels),
    (batch_size, num_keypoints1, num_channels), ..., (batch_size, num_keypointsN, num_channels)` and the output will
    then be of the form: `(N, batch_size, max(num_keypoints0, num_keypoints1, ..., num_keypointsN), num_channels)`.
    If the tensors are attention probabilities, they are assumed to be of shape `(batch_size, num_heads, num_keypoints0,
    num_keypoints0), (batch_size, num_heads, num_keypoints1, num_keypoints1), ..., (batch_size, num_heads,
    num_keypointsN, num_keypointsN)` and the output will then be of the form: `(N, batch_size, num_heads,
    max(num_keypoints0, num_keypoints1, ..., num_keypointsN), max(num_keypoints0, num_keypoints1, ..., num_keypointsN))`.

    For `tensor_list = [tensor0, tensor1, ..., tensorN]` and `max_number_of_keypoints = max(num_keypoints0,
    num_keypoints1, ..., num_keypointsN)`, the output will be a tensor of shape `(N, batch_size,
    max_number_of_keypoints, num_channels)`  or `(N, batch_size, num_heads,max_number_of_keypoints,
    max_number_of_keypoints)`. The stacking is done by padding the tensors with the lower number of keypoints with
    zeros.


    Args:
        tensor_list (`List[torch.Tensor]` of shape `(batch_size, num_keypoints, num_channels)` or `List[torch.Tensor]`
        of shape `(batch_size, num_heads, num_keypoints, num_keypoints)`): List of tensors with inconsistent dimensions.

    Returns:
        (`torch.Tensor` of shape `(N, batch_size, max(num_keypoints0, num_keypoints1, ..., num_keypointsN),
        num_channels)` or `torch.Tensor` of shape `(N, batch_size, num_heads, max(num_keypoints0, num_keypoints1, ...,
        num_keypointsN), max(num_keypoints0, num_keypoints1, ..., num_keypointsN))`): Stacked tensors with consistent
        dimensions.
    """
    current_shape = tensor_list[0].shape
    all_same_shape = all(tensor.shape == current_shape for tensor in tensor_list)
    if all_same_shape:
        return torch.stack(tensor_list, dim=0)

    squeeze_tensors = len(tensor_list[0].shape) == 3
    if squeeze_tensors:
        tensor_list = [tensor[..., None] for tensor in tensor_list]

    max_dim1 = max(tensor.shape[1] for tensor in tensor_list)
    max_dim2 = max(tensor.shape[2] for tensor in tensor_list)
    max_dim3 = max(tensor.shape[3] for tensor in tensor_list)
    stacked_tensors = torch.zeros(len(tensor_list), 2, max_dim1, max_dim2, max_dim3, device=tensor_list[0].device)
    for i, tensor in enumerate(tensor_list):
        stacked_tensors[i, :, : tensor.shape[1], : tensor.shape[2], : tensor.shape[3]] = tensor
    if squeeze_tensors:
        stacked_tensors = stacked_tensors.squeeze(-1)
    return stacked_tensors


def batch_inconsistent_tensor_list(tensor_list: List[Tuple[torch.Tensor]]) -> List[torch.Tensor]:
    """
    Batch a list of tuples of tensors with inconsistent dimensions.
    For a list of N tuples of T tensors : `[(tensor0_0, tensor0_1, ..., tensor0_T), (tensor1_0, tensor1_1, ...,
    tensor1_T), ..., (tensorN_0, tensorN_1, ..., tensorN_T)]`, the output will be a list of T stacked tensors:
    `[stack(tensor0_0, tensor1_0, ..., tensorN_0), stack(tensor0_1, tensor1_1, ..., tensorN_1), ..., stack(tensor0_T,
    tensor1_T, ..., tensorN_T)]`.

    Args:
        tensor_list (`List[Tuple[torch.Tensor]]`): List of tuples of tensors with inconsistent dimensions.

    Returns: (`List[torch.Tensor]`): List of tensors with consistent dimensions.
    """
    list_of_tuples = list(zip(*map(list, tensor_list)))
    return [stack_inconsistent_tensor_list(element) for element in list_of_tuples]


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
            layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1))
            if i < (num_layers - 1):
                layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(
        self, hidden_state: torch.Tensor, output_hidden_states: Optional[bool] = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            hidden_state = layer(hidden_state)
            if output_hidden_states and isinstance(layer, nn.Conv1d):
                all_hidden_states = all_hidden_states + (hidden_state,)
        return hidden_state, all_hidden_states


class SuperGlueKeypointEncoder(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        layer_sizes = config.keypoint_encoder_sizes
        feature_dim = config.descriptor_dim
        # 3 here consists of 2 for the (x, y) coordinates and 1 for the score of the keypoint
        encoder_channels = [3] + layer_sizes + [feature_dim]
        self.encoder = SuperGlueMultiLayerPerceptron(config, channels=encoder_channels)

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

        self.query = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        self.key = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        self.value = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        self.merge = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        batch_dim = query.size(0)

        query = self.query(query).view(batch_dim, self.head_dim, self.num_heads, -1)
        key = self.key(key).view(batch_dim, self.head_dim, self.num_heads, -1)
        value = self.value(value).view(batch_dim, self.head_dim, self.num_heads, -1)

        query_dim = query.shape[1]
        scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / query_dim**0.5
        attention_probs = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.einsum("bhnm,bdhm->bdhn", attention_probs, value)
        output = output.contiguous().view(batch_dim, self.head_dim * self.num_heads, -1)
        output = self.merge(output)

        output = (output, attention_probs) if output_attentions else (output,)

        return output


class SuperGlueAttentionalPropagation(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        descriptor_dim = config.descriptor_dim
        self.attention = SuperGlueMultiHeadAttention(config)
        mlp_channels = [descriptor_dim * 2, descriptor_dim * 2, descriptor_dim]
        self.mlp = SuperGlueMultiLayerPerceptron(config, channels=mlp_channels)

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
        descriptors0: torch.Tensor,
        descriptors1: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple], Optional[Tuple]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if output_hidden_states:
            new_hidden_state = concat_inconsistent_pairs((descriptors0,), (descriptors1,))
            all_hidden_states = all_hidden_states + new_hidden_state

        for gnn_layer, layer_type in zip(self.layers, self.layers_types):
            if layer_type == "cross":
                source0, source1 = descriptors1, descriptors0
            elif layer_type == "self":
                source0, source1 = descriptors0, descriptors1
            else:
                raise ValueError(f"Unrecognized layer type {layer_type}")

            gnn_outputs0 = gnn_layer(
                descriptors0, source0, output_hidden_states=output_hidden_states, output_attentions=output_attentions
            )
            gnn_outputs1 = gnn_layer(
                descriptors1, source1, output_hidden_states=output_hidden_states, output_attentions=output_attentions
            )

            delta0 = gnn_outputs0[0]
            delta1 = gnn_outputs1[0]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + concat_inconsistent_pairs(gnn_outputs0[1], gnn_outputs1[1])
            if output_attentions:
                all_attentions = all_attentions + concat_inconsistent_pairs(gnn_outputs0[2], gnn_outputs1[2])

            descriptors0 = descriptors0 + delta0
            descriptors1 = descriptors1 + delta1
        return descriptors0, descriptors1, all_hidden_states, all_attentions


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
            matches0 (`torch.Tensor` of shape `(1, num_keypoints)`):
                For each keypoint in image0, the index of the keypoint in image1 that was matched with.
            matches1 (`torch.Tensor` of shape `(1, num_keypoints)`):
                For each keypoint in image1, the index of the keypoint in image0 that was matched with.
            matching_scores0 (`torch.Tensor` of shape `(1, num_keypoints)`):
                Scores of predicted matches from the first image to the second image
            matching_scores1 (`torch.Tensor` of shape `(1, num_keypoints)`):
                Scores of predicted matches from the second image to the first image
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

        # Keypoint normalization
        keypoints0 = normalize_keypoints(image0_keypoints, height, width)
        keypoints1 = normalize_keypoints(image1_keypoints, height, width)

        encoded_keypoints0 = self.keypoint_encoder(
            keypoints0, image0_scores, output_hidden_states=output_hidden_states
        )
        encoded_keypoints1 = self.keypoint_encoder(
            keypoints1, image1_scores, output_hidden_states=output_hidden_states
        )

        last_hidden_state0 = encoded_keypoints0[0]
        last_hidden_state1 = encoded_keypoints1[0]

        image0_descriptors = torch.transpose(image0_descriptors, -1, -2)
        image1_descriptors = torch.transpose(image1_descriptors, -1, -2)

        # Keypoint MLP encoder.
        descriptors0 = image0_descriptors + last_hidden_state0
        descriptors1 = image1_descriptors + last_hidden_state1

        # Multi-layer Transformer network.
        gnn_outputs = self.gnn(
            descriptors0,
            descriptors1,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        descriptors0, descriptors1 = gnn_outputs[:2]

        # Final MLP projection.
        projected_descriptors0 = self.final_projection(descriptors0)
        projected_descriptors1 = self.final_projection(descriptors1)

        # Compute matching descriptor distance.
        scores = torch.einsum("bdn,bdm->bnm", projected_descriptors0, projected_descriptors1)
        scores = scores / self.config.descriptor_dim**0.5

        # Run the optimal transport.
        scores = log_optimal_transport(scores, self.bin_score, iterations=self.config.sinkhorn_iterations)

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        matching_scores0 = torch.where(mutual0, max0.values.exp(), zero)
        matching_scores1 = torch.where(mutual1, matching_scores0.gather(1, indices1), zero)
        valid0 = mutual0 & (matching_scores0 > self.config.matching_threshold)
        valid1 = mutual1 & valid0.gather(1, indices1)
        matches0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        matches1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        if output_hidden_states:
            all_hidden_states = all_hidden_states + concat_inconsistent_pairs(
                encoded_keypoints0[1], encoded_keypoints1[1]
            )
            all_hidden_states = all_hidden_states + gnn_outputs[2]
            all_hidden_states = all_hidden_states + concat_inconsistent_pairs(
                (projected_descriptors0,), (projected_descriptors1,)
            )
            all_hidden_states = tuple(x.transpose(-1, -2) for x in all_hidden_states)
        if output_attentions:
            all_attentions = all_attentions + gnn_outputs[3]

        return (
            matches0,
            matches1,
            matching_scores0,
            matching_scores1,
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

        list_hidden_states = []
        list_attentions = []

        list_keypoint_detection = [self.keypoint_detector(image_pair) for image_pair in pixel_values]

        max_keypoints = max([keypoint_detection[0].shape[1] for keypoint_detection in list_keypoint_detection])

        matches = torch.full((batch_size, 2, max_keypoints), -1, device=pixel_values.device, dtype=torch.int)
        matching_scores = torch.zeros((batch_size, 2, max_keypoints), device=pixel_values.device)
        matches_mask = torch.zeros((batch_size, 2, max_keypoints), device=pixel_values.device, dtype=torch.int)
        keypoints = torch.zeros((batch_size, 2, max_keypoints, 2), device=pixel_values.device)

        for i, keypoint_detection_output in enumerate(list_keypoint_detection):
            _keypoints, scores, descriptors, mask = keypoint_detection_output[:4]
            image0_indices, image1_indices = mask != 0
            image0_keypoints = _keypoints[0][image0_indices][None]
            image0_descriptors = descriptors[0][image0_indices][None]
            image0_scores = scores[0][image0_indices][None]
            image1_keypoints = _keypoints[1][image1_indices][None]
            image1_descriptors = descriptors[1][image1_indices][None]
            image1_scores = scores[1][image1_indices][None]

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

            _matches0, _matches1, _matching_scores0, _matching_scores1, _hidden_states, _attentions = (
                match_image_output
            )
            num_matches0, num_matches1 = _matches0.shape[1], _matches1.shape[1]
            matches[i, 0, :num_matches0] = _matches0
            matches[i, 1, :num_matches1] = _matches1
            matching_scores[i, 0, :num_matches0] = _matching_scores0
            matching_scores[i, 1, :num_matches1] = _matching_scores1
            matches_mask[i, 0, :num_matches0] = 1
            matches_mask[i, 1, :num_matches1] = 1
            keypoints[i, :, : _keypoints.shape[1], :] = _keypoints
            list_hidden_states.append(_hidden_states)
            list_attentions.append(_attentions)

        hidden_states = attentions = None
        if output_hidden_states:
            hidden_states = batch_inconsistent_tensor_list(list_hidden_states)
        if output_attentions:
            attentions = batch_inconsistent_tensor_list(list_attentions)

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
