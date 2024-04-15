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
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from transformers import PreTrainedModel, add_start_docstrings
from transformers.models.superglue.configuration_superglue import SuperGlueConfig

from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...utils import (
    ModelOutput,
    add_start_docstrings_to_model_forward,
    logging,
)
from ..auto import AutoModelForKeypointDetection


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC_ = "SuperGlueConfig"

_CHECKPOINT_FOR_DOC_ = "stevenbucaille/superglue_indoor"

SUPERGLUE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "stevenbucaille/superglue_indoor",
    "stevenbucaille/superglue_outdoor",
]


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim**0.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    output = torch.einsum("bhnm,bdhm->bdhn", prob, value)
    return output, prob


def normalize_keypoints(keypoints: torch.Tensor, height: int, width: int):
    """Normalize keypoints locations based on image image_shape"""
    one = keypoints.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (keypoints - center[:, None, :]) / scaling[:, None, :]


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 1)

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1


@dataclass
class ImageMatchingOutput(ModelOutput):
    """
    Base class for outputs of image matching models. Due to the nature of keypoint detection and matching, the number of
    keypoints is not fixed and can vary from image to image, which makes batching non-trivial. In the batch of images,
    the maximum number of matches is set as the dimension of the matches and matching scores. The mask
    tensor is used to indicate which values in the matches and matching_scores, scores and descriptors tensors are
    keypoint matching information and which are padding.

    Args:
        mask (`torch.BoolTensor` of shape `(batch_size, num_keypoints)`):
            Mask indicating which values in matches and matching_scores are keypoint matching information.
        matches (`torch.FloatTensor` of shape `(batch_size, 2, num_matches)`):
            Index of keypoint matched in the other image.
        matching_scores (`torch.FloatTensor` of shape `(batch_size, 2, num_matches)`):
            Scores of predicted matches.
    """

    last_hidden_state: torch.FloatTensor = None
    mask: torch.FloatTensor = None
    matches: torch.FloatTensor = None
    matching_scores: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class SuperGlueMultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        config: SuperGlueConfig,
        channels: List[int],
        do_batch_norm: bool = True,
    ):
        super().__init__()
        num_layers = len(channels)
        layers = []
        for i in range(1, num_layers):
            layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
            if i < (num_layers - 1):
                if do_batch_norm:
                    layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        nn.init.constant_(self.layers[-1].bias, 0.0)

    def forward(
        self,
        input: Tensor,
        output_hidden_states: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            input = layer(input)
            if output_hidden_states and isinstance(layer, nn.Conv1d):
                all_hidden_states = all_hidden_states + (input,)
        output = input
        return output, all_hidden_states


class SuperGlueKeypointEncoder(nn.Module):
    def __init__(
        self,
        config: SuperGlueConfig,
    ):
        super().__init__()
        self.layer_sizes = config.keypoint_encoder_sizes
        self.feature_dim = config.descriptor_dim
        self.encoder = SuperGlueMultiLayerPerceptron(config, [3] + self.layer_sizes + [self.feature_dim])

    def forward(
        self,
        keypoints: Tensor,
        scores: Tensor,
        output_hidden_states: Optional[bool] = False,
    ) -> Tensor:
        keypoints = keypoints.transpose(1, 2)
        scores = scores.unsqueeze(1)
        inputs = torch.cat([keypoints, scores], dim=1)
        return self.encoder(inputs, output_hidden_states=output_hidden_states)


class SuperGlueMultiHeadAttention(nn.Module):
    def __init__(self, config: SuperGlueConfig):
        super().__init__()
        self.feature_dim = config.descriptor_dim
        self.num_heads = config.num_heads
        self.dim = self.feature_dim // self.num_heads
        self.merge = nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [
            layer(x).view(batch_dim, self.dim, self.num_heads, -1) for layer, x in zip(self.proj, (query, key, value))
        ]
        x, _ = attention(query, key, value)
        output = self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))
        return output


class SuperGlueAttentionalPropagation(nn.Module):
    def __init__(
        self,
        config: SuperGlueConfig,
    ):
        super().__init__()
        self.descriptor_dim = config.descriptor_dim
        self.num_heads = config.num_heads
        self.attention = SuperGlueMultiHeadAttention(config)
        self.mlp = SuperGlueMultiLayerPerceptron(
            config,
            [self.descriptor_dim * 2, self.descriptor_dim * 2, self.descriptor_dim],
        )
        nn.init.constant_(self.mlp.layers[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attention(x, source, source)
        message = torch.cat([x, message], dim=1)
        message = self.mlp(message)
        return message


class SuperGlueAttentionalGNN(nn.Module):
    def __init__(
        self,
        config: SuperGlueConfig,
    ):
        super().__init__()
        self.descriptor_dim = config.descriptor_dim
        self.num_heads = config.num_heads
        self.layers_types = config.gnn_layers_types
        self.num_layers = len(self.layers_types)
        self.layers = nn.ModuleList(
            [
                SuperGlueAttentionalPropagation(
                    config,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(
        self,
        descriptors_0: torch.Tensor,
        descriptors_1: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for gnn_layer, type in zip(self.layers, self.layers_types):
            if type == "cross":
                source_0, source_1 = descriptors_1, descriptors_0
            else:  # if type == 'self':
                source_0, source_1 = descriptors_0, descriptors_1

            delta0, hidden_states_0 = gnn_layer(descriptors_0, source_0)
            delta1, hidden_states_0 = gnn_layer(descriptors_1, source_1)
            descriptors_0 = descriptors_0 + delta0
            descriptors_1 = descriptors_1 + delta1
        return descriptors_0, descriptors_1


class SuperGlueFinalProjection(nn.Module):
    def __init__(
        self,
        config: SuperGlueConfig,
    ):
        super().__init__()
        self.descriptor_dim = config.descriptor_dim
        self.final_proj = nn.Conv1d(self.descriptor_dim, self.descriptor_dim, kernel_size=1, bias=True)

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
            Pixel values. Pixel values can be obtained using [`SuperPointImageProcessor`]. See
            [`SuperPointImageProcessor.__call__`] for details.

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
class SuperGlueForImageMatching(SuperGluePreTrainedModel):
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

    def __init__(self, config: SuperGlueConfig):
        super().__init__(config)

        self.keypoint_detector = AutoModelForKeypointDetection.from_config(config.keypoint_detector_config)

        self.keypoint_encoder = SuperGlueKeypointEncoder(config)
        self.gnn = SuperGlueAttentionalGNN(config)
        self.final_projection = SuperGlueFinalProjection(config)

        bin_score = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter("bin_score", bin_score)

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
        output_hidden_states: Optional[bool] = None,
    ):
        if image0_keypoints.shape[1] == 0 or image1_keypoints.shape[1] == 0:  # no keypoints
            shape0, shape1 = image0_keypoints.shape[:-1], image1_keypoints.shape[:-1]
            return (
                image0_keypoints.new_full(shape0, -1, dtype=torch.int),
                image1_keypoints.new_full(shape1, -1, dtype=torch.int),
                image0_keypoints.new_zeros(shape0),
                image1_keypoints.new_zeros(shape1),
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

        if output_hidden_states:
            hidden_states_0, hidden_states_1 = (
                encoded_keypoints0[1],
                encoded_keypoints1[1],
            )

        else:
            hidden_states_0 = None
            hidden_states_1 = None

        # Keypoint MLP encoder.
        descriptors_0 = descriptors_0 + last_hidden_state_0
        descriptors_1 = descriptors_1 + last_hidden_state_1

        # Multi-layer Transformer network.
        descriptors_0, descriptors_1 = self.gnn(descriptors_0, descriptors_1)

        # Final MLP projection.
        projected_descriptors_0 = self.final_projection(descriptors_0)
        projected_descriptors_1 = self.final_projection(descriptors_1)

        # Compute matching descriptor distance.
        scores = torch.einsum("bdn,bdm->bnm", projected_descriptors_0, projected_descriptors_1)
        scores = scores / self.config.descriptor_dim**0.5

        # Run the optimal transport.
        scores = log_optimal_transport(scores, self.bin_score, iters=self.config.sinkhorn_iterations)

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
        return (
            matches_0,
            matches_1,
            matching_scores_0,
            matching_scores_1,
            last_hidden_state_0,
            last_hidden_state_1,
            hidden_states_0,
            hidden_states_1,
        )

    @add_start_docstrings_to_model_forward(SUPERGLUE_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageMatchingOutput]:
        if labels is not None:
            raise ValueError(
                f"SuperGlue is not trainable, no labels should be provided. Therefore, labels should be None but were {type(labels)}"
            )

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
        list_last_hidden_state_0 = []
        list_last_hidden_state_1 = []
        if output_hidden_states:
            list_hidden_states_0 = []
            list_hidden_states_1 = []

        for i in range(pixel_values.size(0)):
            image_pair = pixel_values[i]
            keypoint_detection_output = self.keypoint_detector(
                image_pair,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            if return_dict:
                mask = keypoint_detection_output.mask
                keypoints = keypoint_detection_output.keypoints
                descriptors = keypoint_detection_output.descriptors
                scores = keypoint_detection_output.scores
                hidden_states = keypoint_detection_output.hidden_states
            else:
                if output_hidden_states:
                    (
                        keypoints,
                        scores,
                        descriptors,
                        mask,
                        hidden_states
                    ) = keypoint_detection_output
                else:
                    (
                        keypoints,
                        scores,
                        descriptors,
                        mask,
                    ) = keypoint_detection_output

            image0_indices = torch.nonzero(mask[0]).squeeze()
            image0_keypoints = torch.unsqueeze(keypoints[0][image0_indices], dim=0)
            image0_descriptors = torch.unsqueeze(descriptors[0][image0_indices], dim=0)
            image0_scores = torch.unsqueeze(scores[0][image0_indices], dim=0)
            image1_indices = torch.nonzero(mask[1]).squeeze()
            image1_keypoints = torch.unsqueeze(keypoints[1][image1_indices], dim=0)
            image1_descriptors = torch.unsqueeze(descriptors[1][image1_indices], dim=0)
            image1_scores = torch.unsqueeze(scores[1][image1_indices], dim=0)

            (
                matches_0,
                matches_1,
                matching_scores_0,
                matching_scores_1,
                last_hidden_state_0,
                last_hidden_state_1,
                hidden_states_0,
                hidden_states_1,
            ) = self.match_image_pair(
                image0_keypoints,
                image0_descriptors,
                image0_scores,
                image1_keypoints,
                image1_descriptors,
                image1_scores,
                height,
                width,
                output_hidden_states=output_hidden_states,
            )
            list_matches_0.append(matches_0)
            list_matches_1.append(matches_1)
            list_matching_scores_0.append(matching_scores_0)
            list_matching_scores_1.append(matching_scores_1)
            list_last_hidden_state_0.append(last_hidden_state_0)
            list_last_hidden_state_1.append(last_hidden_state_1)
            if output_hidden_states:
                list_hidden_states_0.append(hidden_states_0)
                list_hidden_states_1.append(hidden_states_1)

        maximum_matches = max(
            [
                max([matches_0.size(1) for matches_0 in list_matches_0]),
                max([matches_1.size(1) for matches_1 in list_matches_1]),
            ]
        )
        matches = torch.full(
            (batch_size, 2, maximum_matches),
            -1,
            device=pixel_values.device,
            dtype=torch.int,
        )
        matching_scores = torch.zeros((batch_size, 2, maximum_matches), device=pixel_values.device)
        matches_mask = torch.zeros(
            (batch_size, 2, maximum_matches),
            device=pixel_values.device,
            dtype=torch.int,
        )
        last_hidden_state = torch.zeros(
            (batch_size, 2, self.config.descriptor_dim, maximum_matches),
            device=pixel_values.device,
        )
        if output_hidden_states:
            hidden_states = [
                torch.zeros(
                    (
                        batch_size,
                        2,
                        v.shape[1],
                        maximum_matches,
                    ),
                    device=pixel_values.device,
                )
                for v in list_hidden_states_0[0]
            ]
        else:
            hidden_states = None

        for i, (
            _matches_0,
            _matches_1,
            _matching_scores_0,
            _matching_scores_1,
            _last_hidden_state_0,
            _last_hidden_state_1,
        ) in enumerate(
            zip(
                list_matches_0,
                list_matches_1,
                list_matching_scores_0,
                list_matching_scores_1,
                list_last_hidden_state_0,
                list_last_hidden_state_1,
            )
        ):
            matches[i, 0, : _matches_0.shape[1]] = _matches_0
            matches[i, 1, : _matches_1.shape[1]] = _matches_1
            matching_scores[i, 0, : _matching_scores_0.shape[1]] = _matching_scores_0
            matching_scores[i, 1, : _matching_scores_1.shape[1]] = _matching_scores_1
            matches_mask[i, 0, : _matches_0.shape[1]] = 1
            matches_mask[i, 1, : _matches_1.shape[1]] = 1
            last_hidden_state[i, 0, :, : _last_hidden_state_0.shape[2]] = _last_hidden_state_0
            last_hidden_state[i, 1, :, : _last_hidden_state_1.shape[2]] = _last_hidden_state_1

            if output_hidden_states:
                for j in range(len(list_hidden_states_0[i])):
                    _hidden_states_0 = list_hidden_states_0[i][j]
                    _hidden_states_1 = list_hidden_states_1[i][j]
                    hidden_states[j][i, 0, :, : _hidden_states_0.shape[2]] = _hidden_states_0
                    hidden_states[j][i, 1, :, : _hidden_states_1.shape[2]] = _hidden_states_1

        if not return_dict:
            return tuple(
                v
                for v in [
                    last_hidden_state,
                    matches_mask,
                    matches,
                    matching_scores,
                    hidden_states,
                ]
                if v is not None
            )

        return ImageMatchingOutput(
            last_hidden_state=last_hidden_state,
            mask=matches_mask,
            matches=matches,
            matching_scores=matching_scores,
            hidden_states=hidden_states,
        )
