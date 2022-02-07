# coding=utf-8
# Copyright 2022 Facebook AI Research and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch MaskFormer model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from numbers import Number
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np
import torch
import torchvision
from torch import Tensor, nn
from torch.nn import functional as F

from einops import rearrange
from einops.einops import repeat

from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    requires_backends,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ..detr import DetrConfig
from ..detr.modeling_detr import DetrDecoder, DetrDecoderOutput
from ..swin import SwinConfig, SwinModel
from .configuration_maskformer import ClassSpec, MaskFormerConfig


logger = logging.get_logger(__name__)
import torch.distributed as dist


_CONFIG_FOR_DOC = "MaskFormerConfig"
_CHECKPOINT_FOR_DOC = "facebook/maskformer-swin-base-ade-640"

MASKFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/maskformer-swin-base-ade-640",
    # See all MaskFormer models at https://huggingface.co/models?filter=maskformer
]

PREDICTIONS_MASKS_KEY = "masks_queries_logits"
PREDICTIONS_LOGITS_KEY = "class_queries_logits"
TARGETS_MASKS_KEY = "pixel"
TARGETS_LABELS_KEY = "classes"

# TODO this has to go away!
from detectron2.utils.comm import get_world_size
from scipy.optimize import linear_sum_assignment


@dataclass
class MaskFormerOutput(ModelOutput):
    class_queries_logits: torch.FloatTensor
    masks_queries_logits: torch.FloatTensor = None


@dataclass
class MaskFormerForSemanticSegmentationOutput(ModelOutput):
    segmentation: torch.FloatTensor = None
    class_queries_logits: torch.FloatTensor = None
    masks_queries_logits: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None


@dataclass
class MaskFormerForPanopticSegmentationOutput(ModelOutput):
    segmentation: torch.FloatTensor = None
    class_queries_logits: torch.FloatTensor = None
    masks_queries_logits: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    segments: List[List[PanopticSegmentationSegment]] = None


# copied from original implementation
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def upsample_like(x: Tensor, like: Tensor, mode: str = "bilinear") -> Tensor:
    """An utility function that upsamples `x` to match the dimension of `like`

    Args:
        x (Tensor): The tensor we wish to upsample
        like (Tensor): The tensor we wish to use as size target
        mode (str, optional): The interpolation mode. Defaults to "bilinear".

    Returns:
        Tensor: The upsampled tensor
    """
    _, _, height, width = like.shape
    upsampled: Tensor = F.interpolate(
        x,
        size=(height, width),
        mode=mode,
        align_corners=False,
    )
    return upsampled


# refactored from original implementation
def dice_loss(inputs: Tensor, labels: Tensor, num_masks: float) -> Tensor:
    r"""
    Compute the DICE loss, similar to generalized IOU for masks as follow

    $$
        \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x \cap y }{x \cup y + 1}}

    $$ In practice, since `labels` is a binary mask, (only 0s and 1s), dice can be computed as follow

    $$
        \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x * y }{x + y + 1}}
    $$

    Args:
        inputs (Tensor): A tensor representing a mask
        labels (Tensor):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).


    Returns:
        Tensor: The computed loss
    """
    probs: Tensor = inputs.sigmoid().flatten(1)
    numerator: Tensor = 2 * (probs * labels).sum(-1)
    denominator: Tensor = probs.sum(-1) + labels.sum(-1)
    loss: Tensor = 1 - (numerator + 1) / (denominator + 1)
    loss = loss.sum() / num_masks
    return loss


# copied from original implementation
def sigmoid_focal_loss(
    inputs: Tensor, labels: Tensor, num_masks: int, alpha: float = 0.25, gamma: float = 2
) -> Tensor:
    r"""
       Focal loss proposed in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) originally used
       in RetinaNet. The loss is computed as follows

    $$
         \mathcal{L}_{\text{focal loss} = -(1 - p_t)^{\gamma}\log{(p_t)}
    $$

    where $CE(p_t) = -\log{(p_t)}}$, CE is the standard Cross Entropy Loss

    Please refer to equation (1,2,3) of the paper for a better understanding.


    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        labels (Tensor,):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).
        alpha (float, optional): Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma (float, optional): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
    Returns:
        Tensor: The computed loss
    """
    probs: Tensor = inputs.sigmoid()
    cross_entropy_loss: Tensor = F.binary_cross_entropy_with_logits(inputs, labels, reduction="none")
    p_t: Tensor = probs * labels + (1 - probs) * (1 - labels)
    loss: Tensor = cross_entropy_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t: Tensor = alpha * labels + (1 - alpha) * (1 - labels)
        loss = alpha_t * loss

    loss = loss.mean(1).sum() / num_masks
    return loss


def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    """
    A pair wise version of the dice loss, see `dice_loss` for usage

    Args:
        inputs (Tensor): A tensor representing a mask
        labels (Tensor):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).


    Returns:
        Tensor: The computed loss between each pairs
    """
    inputs: Tensor = inputs.sigmoid()
    # TODO this .flatten seems to be unecessary because the shape is 2d
    inputs: Tensor = inputs.flatten(1)
    # TODO why 1 is not added to the number to avoid numerator = 0 in edge cases?
    numerator: Tensor = 2 * torch.einsum("nc,mc->nm", inputs, labels)
    # using broadcasting to get a [NUM_QUERIES, NUM_CLASSES] matrix
    denominator: Tensor = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    loss: Tensor = 1 - (numerator + 1) / (denominator + 1)
    return loss


def pair_wise_sigmoid_focal_loss(inputs: Tensor, labels: Tensor, alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    """
    A pair wise version of the focal loss, see `sigmoid_focal_loss` for usage

    Args:
        inputs (Tensor): A tensor representing a mask
        labels (Tensor):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).


    Returns:
        Tensor: The computed loss between each pairs
    """
    if alpha < 0:
        raise ValueError(f"alpha must be positive")

    hw: int = inputs.shape[1]

    prob: Tensor = inputs.sigmoid()
    cross_entropy_loss_pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    focal_pos: Tensor = ((1 - prob) ** gamma) * cross_entropy_loss_pos
    focal_pos *= alpha

    cross_entropy_loss_neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")

    focal_neg: Tensor = (prob ** gamma) * cross_entropy_loss_neg
    focal_neg *= 1 - alpha

    loss: Tensor = torch.einsum("nc,mc->nm", focal_pos, labels) + torch.einsum("nc,mc->nm", focal_neg, (1 - labels))

    return loss / hw


# refactored from original implementation
class MaskFormerHungarianMatcher(nn.Module):
    """This class computes an assignment between the labels and the predictions of the network

    For efficiency reasons, the labels don't include the no_object. Because of this, in general, there are more
    predictions than labels. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All costs cant be 0")
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(self, outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> List[Tuple[Tensor]]:
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            labels: This is a list of labels (len(labels) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected labels (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        indices: List[Tuple[np.array]] = []

        preds_masks: Tensor = outputs[PREDICTIONS_MASKS_KEY]
        labels_masks: Tensor = labels[TARGETS_MASKS_KEY]
        preds_probs: Tensor = outputs[PREDICTIONS_LOGITS_KEY].softmax(dim=-1)
        # downsample all masks in one go -> save memory
        labels_masks: Tensor = F.interpolate(labels_masks, size=preds_masks.shape[-2:], mode="nearest")
        # iterate through batch size
        for pred_probs, pred_mask, target_mask, labels in zip(
            preds_probs, preds_masks, labels_masks, labels[TARGETS_LABELS_KEY]
        ):
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class: Tensor = -pred_probs[:, labels]
            # flatten spatial dimension
            pred_mask_flat: Tensor = rearrange(pred_mask, "q h w -> q (h w)")  # [num_queries, H*W]
            target_mask_flat: Tensor = rearrange(target_mask, "c h w -> c (h w)")  # [num_total_labels, H*W]
            # compute the focal loss between each mask pairs -> shape [NUM_QUERIES, CLASSES]
            cost_mask: Tensor = pair_wise_sigmoid_focal_loss(pred_mask_flat, target_mask_flat)
            # Compute the dice loss betwen each mask pairs -> shape [NUM_QUERIES, CLASSES]
            cost_dice: Tensor = pair_wise_dice_loss(pred_mask_flat, target_mask_flat)
            # final cost matrix
            cost_matrix: Tensor = (
                self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            )
            # do the assigmented using the hungarian algorithm in scipy
            assigned_indices: Tuple[np.array] = linear_sum_assignment(cost_matrix.cpu())
            indices.append(assigned_indices)

        # TODO this is a little weird, they can be stacked in one tensor
        matched_indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices
        ]
        return matched_indices

    def __repr__(self):
        head = "Matcher " + self.__class__.__name__
        body = [
            f"cost_class: {self.cost_class}",
            f"cost_mask: {self.cost_mask}",
            f"cost_dice: {self.cost_dice}",
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


# copied from original implementation
class MaskFormerLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: MaskFormerHungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float,
        losses: List[str],
    ):
        """The MaskFormer Loss. The loss is computed very similar to DETR. The process happens in two steps:
        1) we compute hungarian assignment between ground truth masks and the outputs of the model 2) we supervise each
        pair of matched ground-truth / prediction (supervise class and mask)

        Args:
            num_classes (int): The number of classes
            matcher (MaskFormerHungarianMatcher):
                A torch module that computes the assigments between the predictions and labels
            weight_dict (Dict[str, float]): A dictionary of weights to be applied to the different losses
            eos_coef (float): TODO no idea
            losses (List[str]): A list of losses to be used TODO probably remove it
        """

        super().__init__()
        requires_backends(self, ["scipy"])
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight: Tensor = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(
        self, outputs: Dict[str, Tensor], labels: Dict[str, Tensor], indices: Tuple[np.array], num_masks: float
    ) -> Dict[str, Tensor]:
        """Classification loss (NLL)
        # TODO this doc was copied by the authors labels dicts must contain the key "labels" containing a tensor of dim
        [nb_target_masks]
        """

        pred_logits: Tensor = outputs[PREDICTIONS_LOGITS_KEY]
        b, q, _ = pred_logits.shape

        idx = self._get_src_permutation_idx(indices)
        # shape = [BATCH, N_QUERIES]
        target_classes_o: Tensor = torch.cat(
            [target[j] for target, (_, j) in zip(labels[TARGETS_LABELS_KEY], indices)]
        )
        # shape = [BATCH, N_QUERIES]
        target_classes: Tensor = torch.full(
            (b, q), fill_value=self.num_classes, dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o
        loss_ce: Tensor = F.cross_entropy(rearrange(pred_logits, "b q c -> b c q"), target_classes, self.empty_weight)
        losses: Tensor = {"loss_cross_entropy": loss_ce}
        return losses

    def loss_masks(
        self, outputs: Dict[str, Tensor], labels: Dict[str, Tensor], indices: Tuple[np.array], num_masks: int
    ) -> Dict[str, Tensor]:
        """Compute the losses related to the masks: the focal loss and the dice loss.
        labels dicts must contain the key "masks" containing a tensor of dim [nb_target_masks, h, w]
        """
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        pred_masks = outputs[PREDICTIONS_MASKS_KEY]  # shape [BATCH, NUM_QUERIES, H, W]
        pred_masks = pred_masks[src_idx]  # shape [BATCH * NUM_QUERIES, H, W]
        target_masks = labels[TARGETS_MASKS_KEY]  # shape [BATCH, NUM_QUERIES, H, W]
        target_masks = target_masks[tgt_idx]  # shape [BATCH * NUM_QUERIES, H, W]
        # upsample predictions to the target size, we have to add one dim to use interpolate
        pred_masks = F.interpolate(
            pred_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        pred_masks = pred_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(pred_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(pred_masks, target_masks, num_masks),
            "loss_dice": dice_loss(pred_masks, target_masks, num_masks),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute labels following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, labels, indices, num_masks):
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        if loss not in loss_map:
            raise KeyError(f"{loss} not in loss_map")
        return loss_map[loss](outputs, labels, indices, num_masks)

    def forward(self, outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             labels: list of dicts, such that len(labels) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # TODO in theory here we can just take the `pred_masks` key
        outputs_without_aux = {
            PREDICTIONS_MASKS_KEY: outputs[PREDICTIONS_MASKS_KEY],
            PREDICTIONS_LOGITS_KEY: outputs[PREDICTIONS_LOGITS_KEY],
        }

        # Retrieve the matching between the outputs of the last layer and the labels
        indices = self.matcher(outputs_without_aux, labels)

        # Compute the average number of target masks accross all nodes, for normalization purposes
        num_masks: Number = self.get_num_masks(labels, device=next(iter(outputs.values())).device)

        # Compute all the requested losses
        losses: Dict[str, Tensor] = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, labels, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxilary_predictions" in outputs:
            for i, aux_outputs in enumerate(outputs["auxilary_predictions"]):
                indices = self.matcher(aux_outputs, labels)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, labels, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def get_num_masks(self, labels: Dict[str, Tensor], device: torch.device) -> Number:
        # Compute the average number of target masks accross all nodes, for normalization purposes
        num_masks: int = labels[TARGETS_LABELS_KEY].shape[0]
        num_masks_pt: Tensor = torch.as_tensor([num_masks], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks_pt)
        num_masks_clamped: Number = torch.clamp(num_masks_pt / get_world_size(), min=1).item()
        return num_masks_clamped


class SwinTransformerBackbone(nn.Module):
    def __init__(self, config: SwinConfig):
        super().__init__()
        # TODO should add a pretrain
        self.model = SwinModel(config)
        self.hidden_states_norms = nn.ModuleList([nn.LayerNorm(out_shape) for out_shape in self.outputs_shapes])
        # little hack, our swin transformer has already the last norm, so let's switch the refence of the last item
        self.hidden_states_norms[-1] = self.model.layernorm

    def forward(self, *args, **kwargs) -> List[Tensor]:
        output = self.model(*args, **kwargs, output_hidden_states=True)
        hidden_states_permuted: List[Tensor] = []
        # we need to reshape the hidden state to their original spatial dimensions
        # skipping the embeddings
        hidden_states: Tuple[Tuple[Tensor]] = output.hidden_states[1:]
        # spatial dimensions contains all the heights and widths of each stage, including after the embeddings
        spatial_dimensions: Tuple[Tuple[int, int]] = output.hidden_states_spatial_dimensions
        for i, (hidden_state, (h, w)) in enumerate(zip(hidden_states, spatial_dimensions)):
            norm = self.hidden_states_norms[i]
            # the last element corespond to the layer's last block output but before patch merging
            hidden_state_unpolled: Tensor = hidden_state[-1]
            hidden_state_norm = norm(hidden_state_unpolled)
            # our pixel decoder (FPN) expect 3D tensors (features)
            hidden_state_permuted = rearrange(hidden_state_norm, "b (h w) d -> b d h w", h=h, w=w).contiguous()
            hidden_states_permuted.append(hidden_state_permuted)
        return hidden_states_permuted

    @property
    def input_resolutions(self) -> List[int]:
        return [layer.input_resolution for layer in self.model.encoder.layers]

    @property
    def outputs_shapes(self) -> List[int]:
        return [layer.dim for layer in self.model.encoder.layers]


class ConvLayer(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, kernel_size: int = 3, padding: int = 1):
        """A basic module that executs conv - norm - in sequence used in MaskFormer.

        Args:
            in_features (int): The number of input features (channels)
            out_features (int): The number of outputs features (channels)
        """
        super().__init__(
            nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(32, out_features),
            nn.ReLU(inplace=True),
        )


class FPNLayer(nn.Module):
    def __init__(self, in_features: int, lateral_features: int):
        """A Feature Pyramid Network Layer. It creates a feature map by aggregating features from the previous and backbone layer.
        Due to the spartial mismatch, the tensor coming from the previous layer is upsample.

        Args:
            in_features (int): The number of input features (channels)
            lateral_features (int): The number of lateral features (channels)
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(lateral_features, in_features, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(32, in_features),
        )

        self.block = ConvLayer(in_features, in_features)

    def forward(self, down: Tensor, left: Tensor) -> Tensor:
        left = self.proj(left)
        down = F.interpolate(down, size=left.shape[-2:], mode="nearest")
        down += left
        down = self.block(down)
        return down


class FPNModel(nn.Module):
    def __init__(self, in_features: int, lateral_widths: List[int], feature_size: int = 256):
        """Feature Pyramid Network, given an input tensor and a set of features map of different feature/spatial size, it creates
        a list of features map with different the same feature size.

                Args:
                    in_features (int): The number of input features (channels)
                    lateral_widths (List[int]): A list with the features (channels) size of each lateral connection
                    feature_size (int, optional):
                        The features (channels) of the resulting feature maps. Defaults to 256.
        """
        super().__init__()
        self.stem = ConvLayer(in_features, feature_size)
        self.layers = nn.Sequential(*[FPNLayer(feature_size, lateral_width) for lateral_width in lateral_widths[::-1]])

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        fpn_features: List[Tensor] = []
        last_feature: Tensor = features.pop()
        x: Tensor = self.stem(last_feature)
        for layer, left in zip(self.layers, features[::-1]):
            x = layer(x, left)
            fpn_features.append(x)
        return fpn_features


class MaskFormerPixelDecoder(nn.Module):
    def __init__(self, *args, feature_size: int = 256, mask_feature_size: int = 256, **kwargs):
        """Pixel Decoder Module proposed in [Per-Pixel Classification is Not All You Need for Semantic
        Segmentation](https://arxiv.org/abs/2107.06278). It first run the backbone's feature into a Feature Pyramid
        Network creating a list of features map. Then, it projects the last one to the correct `mask_size`

                Args:
                    feature_size (int, optional): The features (channels) of FPN feature maps. Defaults to 256.
                    mask_feature_size (int, optional):
                        The features (channels) of the target masks size $C_{\epsilon}$ in the paper. Defaults to 256.
        """
        super().__init__()
        self.fpn = FPNModel(*args, feature_size=feature_size, **kwargs)
        self.mask_projection = nn.Conv2d(feature_size, mask_feature_size, kernel_size=3, padding=1)

    def forward(self, features: List[Tensor]) -> Tensor:
        fpn_features: List[Tensor] = self.fpn(features)
        # we use the last feature map
        x = self.mask_projection(fpn_features[-1])
        return x


# copied from original implementation, also practically equal to DetrSinePositionEmbedding
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None
    ):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * torch.pi if scale is None else scale

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class MaskformerMLPPredictionHead(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        """A classic Multi Layer Perceptron (MLP)

        Args:
            input_dim (int): [description]
            hidden_dim (int): [description]
            output_dim (int): [description]
            num_layers (int, optional): [description]. Defaults to 3.
        """
        in_dims: List[int] = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_dims: List[int] = [hidden_dim] * (num_layers - 1) + [output_dim]

        layers: List[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            # TODO should name them, e.g. fc, act ...
            layer: nn.Module = nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True) if i < num_layers - 1 else nn.Identity()
            )
            layers.append(layer)

        super().__init__(*layers)


class MaskFormerPixelLevelModule(nn.Module):
    def __init__(self, config: MaskFormerConfig):
        """Pixel Level Module proposed in [Per-Pixel Classification is Not All You Need for Semantic
        Segmentation](https://arxiv.org/abs/2107.06278). It runs the input image trough a backbone and a pixel decoder,
        generating a image features and pixel embeddings."""
        super().__init__()
        self.encoder = SwinTransformerBackbone(config.backbone)
        self.decoder = MaskFormerPixelDecoder(
            in_features=self.encoder.outputs_shapes[-1],
            feature_size=config.fpn_feature_size,
            mask_feature_size=config.mask_feature_size,
            lateral_widths=self.encoder.outputs_shapes[:-1],
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        features: List[Tensor] = self.encoder(x)
        # the last feature is actually the output from the last layer
        image_features: Tensor = features[-1]
        pixel_embeddings: Tensor = self.decoder(features)
        return image_features, pixel_embeddings


class MaskFormerTransformerModule(nn.Module):
    def __init__(self, in_features: int, config: MaskFormerConfig):
        super().__init__()
        hidden_size: int = config.transformer_decoder.hidden_size
        self.position_embedder = PositionEmbeddingSine(num_pos_feats=hidden_size // 2, normalize=True)
        self.queries_embedder = nn.Embedding(config.transformer_decoder.num_queries, hidden_size)
        should_project = in_features != hidden_size
        self.input_projection = nn.Conv2d(in_features, hidden_size, kernel_size=1) if should_project else nn.Identity()
        self.transformer_decoder = DetrDecoder(config=config.transformer_decoder)

    def forward(self, image_features: Tensor) -> Tuple[Tensor]:
        image_features = self.input_projection(image_features)
        position_embeddings: Tensor = self.position_embedder(image_features)
        queries_embeddings: Tensor = repeat(self.queries_embedder.weight, "q c -> b q c", b=image_features.shape[0])
        inputs_embeds = torch.zeros_like(queries_embeddings)
        image_features = rearrange(image_features, "b c h w -> (h w) b c")
        position_embeddings = rearrange(position_embeddings, "b c h w -> (h w) b c")

        transformer_decoder_output: ModelOutput = self.transformer_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            encoder_hidden_states=image_features,
            encoder_attention_mask=None,
            position_embeddings=position_embeddings,
            query_position_embeddings=queries_embeddings,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=None,
        )
        return transformer_decoder_output.hidden_states


class MaskFormerSegmentationModule(nn.Module):
    def __init__(self, config: MaskFormerConfig):
        super().__init__()
        hidden_size: int = config.transformer_decoder.hidden_size
        # + 1 because we add the "null" class
        self.mask_classification = config.mask_classification
        self.class_predictor = nn.Linear(hidden_size, config.num_labels + 1)
        self.mask_embedder = MaskformerMLPPredictionHead(hidden_size, hidden_size, config.mask_feature_size)

    def forward(
        self, decoder_outputs: Tuple[Tensor], pixel_embeddings: Tensor, use_auxilary_loss: bool = True
    ) -> Dict[str, Tensor]:

        out: Dict[str, Tensor] = {}

        # NOTE this code is a little bit cumbersome, an easy fix is to always return a list of predictions, if we have auxilary loss then we are going to return more than one element in the list
        if use_auxilary_loss:
            stacked_decoder_outputs: Tensor = torch.stack(decoder_outputs)
            classes: Tensor = self.class_predictor(stacked_decoder_outputs)
            out.update({PREDICTIONS_LOGITS_KEY: classes[-1]})
            # get the masks
            mask_embeddings: Tensor = self.mask_embedder(stacked_decoder_outputs)
            # sum up over the channels for each embedding
            binaries_masks: Tensor = torch.einsum("lbqc,   bchw -> lbqhw", mask_embeddings, pixel_embeddings)
            binary_masks: Tensor = binaries_masks[-1]
            # get the auxilary predictions (one for each decoder's layer)
            auxilary_predictions: List[str, Tensor] = []
            # go til [:-1] because the last one is always used
            for aux_binary_masks, aux_classes in zip(binaries_masks[:-1], classes[:-1]):
                auxilary_predictions.append(
                    {PREDICTIONS_MASKS_KEY: aux_binary_masks, PREDICTIONS_LOGITS_KEY: aux_classes}
                )
            out.update({"auxilary_predictions": auxilary_predictions})

        else:
            last_decoder_output: Tensor = decoder_outputs[-1]
            classes: Tensor = self.class_predictor(last_decoder_output)
            out.update({PREDICTIONS_LOGITS_KEY: classes})
            # get the masks
            mask_embeddings: Tensor = self.mask_embedder(last_decoder_output)
            # sum up over the channels
            binary_masks: Tensor = torch.einsum("bqc,   bchw -> bqhw", mask_embeddings, pixel_embeddings)
        out.update({PREDICTIONS_MASKS_KEY: binary_masks})
        return out


MASKFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MaskFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MASKFORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoFeatureExtractor`]. See
            [`AutoFeatureExtractor.__call__`] for details.
"""
# TODO add the others but first we need to decide on their names/format


class MaskFormerPretrainedModel(PreTrainedModel):
    config_class = MaskFormerConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        if isinstance(module, SwinModel):
            # okay but how do I call the Swin init weights on this module???
            module._init_weights()
        elif isinstance(module, DetrDecoder):
            # same here
            pass
        # TODO code the rest
        # - ffpn
        # - probably a couple of mapping


class MaskFormerForPretraining(PreTrainedModel):
    def __init__(self, config: MaskFormerConfig):
        super().__init__(config)
        losses = ["labels", "masks"]

        self.matcher = MaskFormerHungarianMatcher(
            cost_class=1.0, cost_dice=config.dice_weight, cost_mask=config.mask_weight
        )

        self.weight_dict: Dict[str, float] = {
            "loss_cross_entropy": config.cross_entropy_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        self.criterion = MaskFormerLoss(
            config.num_labels,
            matcher=self.matcher,
            weight_dict=self.weight_dict,
            eos_coef=config.no_object_weight,
            losses=losses,
        )

    def get_loss_dict(self, outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> Dict[str, Tensor]:
        loss_dict: Dict[str, Tensor] = self.criterion(outputs, labels)
        # weight each loss by `self.weight_dict[<LOSS_NAME>]`
        weighted_loss_dict: Dict[str, Tensor] = {
            k: v * self.weight_dict[k] for k, v in loss_dict.items() if k in self.weight_dict
        }
        return weighted_loss_dict

    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        # probably an awkward way to reduce it
        return torch.tensor(list(loss_dict.values()), dtype=torch.float).sum()


@add_start_docstrings(
    "The bare MaskFormer Model outputting raw hidden-states without any specific head on top.",
    MASKFORMER_START_DOCSTRING,
)
class MaskFormerModel(MaskFormerPretrainedModel):
    def __init__(self, config: MaskFormerConfig):
        super().__init__(config)
        self.pixel_level_module = MaskFormerPixelLevelModule(config)
        self.transformer_module = MaskFormerTransformerModule(
            in_features=self.pixel_level_module.encoder.outputs_shapes[-1], config=config
        )
        self.segmentation_module = MaskFormerSegmentationModule(config)

    @add_start_docstrings_to_model_forward(MASKFORMER_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=MaskFormerOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Tensor,
        pixel_mask: Optional[Tensor] = None,
    ) -> MaskFormerOutput:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        image_features, pixel_embeddings = self.pixel_level_module(pixel_values)
        queries = self.transformer_module(image_features)
        outputs: Dict[str, Tensor] = self.segmentation_module(queries, pixel_embeddings, self.config.use_auxilary_loss)

        return MaskFormerOutput(**outputs)


class PanopticSegmentationSegment(TypedDict):
    id: int
    category_id: int
    is_thing: bool
    label: str


# NOTE this to be moved inside the feature extractor
@add_start_docstrings(
    """
    MaskFormer for panoptic segmentation
    """,
    MASKFORMER_START_DOCSTRING,
)
class MaskFormerForPanopticSegmentation(MaskFormerForSemanticSegmentation):
    def __init__(
        self,
        config: MaskFormerConfig,
    ):
        super().__init__(config)

    def remove_low_and_no_objects(
        self,
        masks: Tensor,
        scores: Tensor,
        labels: Tensor,
        object_mask_threshold: float,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if not (masks.shape[0] == scores.shape[0] == labels.shape[0]):
            raise ValueError("mask, scores and labels must have the same shape!")

        to_keep: Tensor = labels.ne(self.model.config.num_labels) & (scores > object_mask_threshold)

        return masks[to_keep], scores[to_keep], labels[to_keep]

    @add_start_docstrings_to_model_forward(MASKFORMER_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=MaskFormerForPanopticSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        *args,
        object_mask_threshold: Optional[float] = 0.8,
        overlap_mask_area_threshold: Optional[float] = 0.8,
        **kwargs
    ):
        outputs: MaskFormerOutput = self.model(*args, **kwargs)
        preds_logits: Tensor = outputs.preds_logits
        preds_masks: Tensor = outputs.preds_masks
        # since all images are padded, they all have the same spatial dimensions
        _, _, height, width = preds_masks.shape
        # for each query, the best scores and their indeces
        pred_scores, pred_labels = F.softmax(preds_logits, dim=-1).max(-1)
        # pred_scores and pred_labels shape = [BATH,NUM_QUERIES]
        mask_probs = preds_masks.sigmoid()
        # mask probs has shape [BATCH, QUERIES, HEIGHT, WIDTH]
        # now, we need to iterate over the batch size to correctly process the segmentation we got from the queries using our thresholds. Even if the original predicted masks have the same shape across the batch, they won't after thresholding so batch-wise operations are impossible
        # TODO we need to store the segmentations and the segments in a list
        for (mask_probs, pred_scores, pred_labels) in zip(mask_probs, pred_scores, pred_labels):

            mask_probs, pred_scores, pred_labels = self.remove_low_and_no_objects(
                mask_probs, pred_scores, pred_labels, object_mask_threshold
            )
            we_detect_something: bool = mask_probs.shape[0] > 0

            segmentation: Tensor = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
            segments: List[PanopticSegmentationSegment] = []

            if we_detect_something:
                current_segment_id: int = 0
                # weight each mask by its score
                mask_probs *= pred_scores.view(-1, 1, 1)
                # find out for each pixel what is the most likely class to be there
                mask_labels: Tensor = mask_probs.argmax(0)
                # mask_labels shape = [H,W] where each pixel has a class label
                stuff_memory_list: Dict[str, int] = {}
                # this is a map between stuff and segments id, the used it to keep track of the instances of one class
                for k in range(pred_labels.shape[0]):
                    pred_class: int = pred_labels[k].item()
                    # check if pred_class is not a "thing", so it can be merged with other instance. For example, class "sky" cannot have more then one instance
                    class_spec: ClassSpec = self.model.config.dataset_metadata["classes"][pred_class]
                    is_stuff = not class_spec["is_thing"]
                    # get the mask associated with the k class
                    mask_k: Tensor = mask_labels == k
                    # create the area, since bool we just need to sum :)
                    mask_k_area: Tensor = mask_k.sum()
                    # this is the area of all the stuff in query k
                    # TODO not 100%, why are the taking the k query here????
                    original_area: Tensor = (mask_probs[k] >= 0.5).sum()

                    mask_does_exist: bool = mask_k_area > 0 and original_area > 0

                    if mask_does_exist:
                        # find out how much of the all area mask_k is using
                        area_ratio: float = mask_k_area / original_area
                        mask_k_is_overlapping_enough: bool = area_ratio.item() > overlap_mask_area_threshold

                        if mask_k_is_overlapping_enough:
                            # merge stuff regions
                            if pred_class in stuff_memory_list:
                                current_segment_id = stuff_memory_list[pred_class]
                            else:
                                current_segment_id += 1
                            # then we update out mask with the current segment
                            segmentation[mask_k] = current_segment_id
                            segments.append(
                                {
                                    "id": current_segment_id,
                                    "category_id": pred_class,
                                    "is_thing": not is_stuff,
                                    "label": class_spec["label"],
                                }
                            )
                            if is_stuff:
                                stuff_memory_list[pred_class] = current_segment_id

            return MaskFormerForPanopticSegmentationOutput(segmentation=segmentation, segments=segments, **outputs)
