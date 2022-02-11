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
from optparse import Option
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.functional import interpolate, binary_cross_entropy_with_logits, cross_entropy

from transformers.modeling_outputs import BaseModelOutput

from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    requires_backends,
    add_code_sample_docstrings,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ..detr.modeling_detr import DetrDecoder, DetrDecoderOutput, DetrPreTrainedModel
from ..swin import SwinConfig, SwinModel
from .configuration_maskformer import MaskFormerConfig


logger = logging.get_logger(__name__)
import torch.distributed as dist


_CONFIG_FOR_DOC = "MaskFormerConfig"
_CHECKPOINT_FOR_DOC = "facebook/maskformer-swin-base-ade-640"
_FEAT_EXTRACTOR_FOR_DOC = "MaskFormerFeatureExtractor"

MASKFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/maskformer-swin-base-ade-640",
    # See all MaskFormer models at https://huggingface.co/models?filter=maskformer
]


from scipy.optimize import linear_sum_assignment

# TODO ask what I should do with dist code
def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


@dataclass
class MaskFormerPixelLevelModuleOutput(ModelOutput):
    """MaskFormer's pixel level module output. It returns both the last and (optionally) the hidden states from the `encoder` and `decoder`. By default, the `encoder` is a Swin Transformer and the `decoder` is a Feature Pyramid Network (FPN).

    The `encoder_last_hidden_state` are referred on the paper as **images features**, while `decoder_last_hidden_state` as **pixel embeddings**
     Args:
        encoder_last_hidden_state (`torch.FloatTensor` of shape`(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the encoder.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at
            the output of each stage.
        decoder_last_hidden_state (`torch.FloatTensor` of shape`(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the decoder.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at
            the output of each stage.
    """

    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class MaskFormerPixelDecoderOutput(BaseModelOutput):
    """
    MaskFormer's pixel decoder module output, practically a Feature Pyramid Network. It returns the last hidden state and (optionally) the hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_channels, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    """


@dataclass
class MaskFormerOutput(ModelOutput):
    """Base class for outputs of MaskFormer model. This class returns all the needed hidden states to compute the logits.

    Args:
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the encoder model (backbone).
        pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the pixel decoder model (FPN).
        transformer_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Last hidden states (final feature map) of the last stage of the transformer decoder model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder model at
            the output of each stage.
        pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel decoder model at
            the output of each stage.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the transformer decoder at
            the output of each stage.
    """

    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    pixel_decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    transformer_decoder_last_hidden_state: Optional[torch.FloatTensor] = None

    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MaskFormerForInstanceSegmentationOutput(ModelOutput):
    """
    Output type of [`MaskFormerForInstanceSegmentation`].

    This output can be directly passed to [`~MaskFormerFeatureExtractor.post_process_segmentation`] or  [`~MaskFormerFeatureExtractor.post_process_panoptic_segmentation`] depending on the task. Please, see [`~MaskFormerFeatureExtractor] for a detail usage.

    Args:
        class_queries_logits (torch.FloatTensor): A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each query.
        masks_queries_logits (torch.FloatTensor):  A tensor of shape `(batch_size, num_queries, num_classes + 1)` representing the proposed classes for each query. Note the `+ 1` is needed because we incorporate the null class.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the encoder model (backbone).
        pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the pixel decoder model (FPN).
        transformer_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Last hidden states (final feature map) of the last stage of the transformer decoder model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder model at
            the output of each stage.
        pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel decoder model at
            the output of each stage.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the transformer decoder at
            the output of each stage.
    """

    class_queries_logits: torch.FloatTensor = None
    masks_queries_logits: torch.FloatTensor = None
    auxilary_logits: torch.FloatTensor = None

    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    pixel_decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    transformer_decoder_last_hidden_state: Optional[torch.FloatTensor] = None

    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict[str, torch.FloatTensor]] = None


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
    upsampled: Tensor = interpolate(
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


# refactored from original implementation
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
    cross_entropy_loss: Tensor = binary_cross_entropy_with_logits(inputs, labels, reduction="none")
    p_t: Tensor = probs * labels + (1 - probs) * (1 - labels)
    loss: Tensor = cross_entropy_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t: Tensor = alpha * labels + (1 - alpha) * (1 - labels)
        loss = alpha_t * loss

    loss = loss.mean(1).sum() / num_masks
    return loss


# refactored from original implementation
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
    inputs: Tensor = inputs.sigmoid().flatten(1)
    numerator: Tensor = 2 * torch.einsum("nc,mc->nm", inputs, labels)
    # using broadcasting to get a [NUM_QUERIES, NUM_CLASSES] matrix
    denominator: Tensor = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    loss: Tensor = 1 - (numerator + 1) / (denominator + 1)
    return loss


# refactored from original implementation
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

    height_and_width: int = inputs.shape[1]

    prob: Tensor = inputs.sigmoid()
    cross_entropy_loss_pos = binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    focal_pos: Tensor = ((1 - prob) ** gamma) * cross_entropy_loss_pos
    focal_pos *= alpha

    cross_entropy_loss_neg = binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")

    focal_neg: Tensor = (prob ** gamma) * cross_entropy_loss_neg
    focal_neg *= 1 - alpha

    loss: Tensor = torch.einsum("nc,mc->nm", focal_pos, labels) + torch.einsum("nc,mc->nm", focal_neg, (1 - labels))

    return loss / height_and_width


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
                 "masks_queries_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "class_queries_logits": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            labels: This is a list of labels (len(labels) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "mask_labels": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected labels (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        indices: List[Tuple[np.array]] = []

        preds_masks: Tensor = outputs["masks_queries_logits"]
        labels_masks: Tensor = labels["mask_labels"]
        preds_probs: Tensor = outputs["class_queries_logits"].softmax(dim=-1)
        # downsample all masks in one go -> save memory
        labels_masks: Tensor = interpolate(labels_masks, size=preds_masks.shape[-2:], mode="nearest")
        # iterate through batch size
        for pred_probs, pred_mask, target_mask, labels in zip(
            preds_probs, preds_masks, labels_masks, labels["class_labels"]
        ):
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class: Tensor = -pred_probs[:, labels]
            # flatten spatial dimension "q h w -> q (h w)"
            num_queries, height, width = pred_mask.shape
            pred_mask_flat: Tensor = pred_mask.view(num_queries, height * width)  # [num_queries, H*W]
            # same for target_mask "c h w -> c (h w)"
            num_channels, height, width = target_mask.shape
            target_mask_flat: Tensor = target_mask.view(num_channels, height * width)  # [num_total_labels, H*W]
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


# copied and adapted from original implementation
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
            eos_coef (float): Weight to apply to the null class
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

        pred_logits: Tensor = outputs["class_queries_logits"]
        b, q, _ = pred_logits.shape

        idx = self._get_src_permutation_idx(indices)
        # shape = [BATCH, N_QUERIES]
        target_classes_o: Tensor = torch.cat([target[j] for target, (_, j) in zip(labels["class_labels"], indices)])
        # shape = [BATCH, N_QUERIES]
        target_classes: Tensor = torch.full(
            (b, q), fill_value=self.num_classes, dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o
        # target_classes is a [BATCH, CLASSES, N_QUERIES], we need to permute pred_logits "b q c -> b c q"
        pred_logits_permuted: Tensor = pred_logits.permute(0, 2, 1)
        loss_ce: Tensor = cross_entropy(pred_logits_permuted, target_classes, self.empty_weight)
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
        pred_masks = outputs["masks_queries_logits"]  # shape [BATCH, NUM_QUERIES, H, W]
        pred_masks = pred_masks[src_idx]  # shape [BATCH * NUM_QUERIES, H, W]
        target_masks = labels["mask_labels"]  # shape [BATCH, NUM_QUERIES, H, W]
        target_masks = target_masks[tgt_idx]  # shape [BATCH * NUM_QUERIES, H, W]
        # upsample predictions to the target size, we have to add one dim to use interpolate
        pred_masks = interpolate(
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
        outputs_without_aux = {
            "masks_queries_logits": outputs["masks_queries_logits"],
            "class_queries_logits": outputs["class_queries_logits"],
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
        num_masks: int = labels["class_labels"].shape[0]
        num_masks_pt: Tensor = torch.as_tensor([num_masks], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks_pt)
        num_masks_clamped: Number = torch.clamp(num_masks_pt / get_world_size(), min=1).item()
        return num_masks_clamped


class SwinTransformerBackbone(nn.Module):
    """This class uses [`SwinModel`] to reshape it's `hidden_states` from (`batch_size, sequence_length, hidden_size)` to (`batch_size, num_channels, height, width)`).

    Args:
        config (SwinConfig): The configuration used by [`SwinModel`]
    """

    def __init__(self, config: SwinConfig):
        super().__init__()
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
        for i, (hidden_state, (height, width)) in enumerate(zip(hidden_states, spatial_dimensions)):
            norm = self.hidden_states_norms[i]
            # the last element corespond to the layer's last block output but before patch merging
            hidden_state_unpolled: Tensor = hidden_state[-1]
            hidden_state_norm = norm(hidden_state_unpolled)
            # our pixel decoder (FPN) expect 3D tensors (features)
            batch_size, sequence_length, hidden_size = hidden_state_norm.shape
            # reshape our tensor "b (h w) d -> b d h w"
            hidden_state_permuted = (
                hidden_state_norm.permute(0, 2, 1).view((batch_size, hidden_size, height, width)).contiguous()
            )
            hidden_states_permuted.append(hidden_state_permuted)
        return hidden_states_permuted

    @property
    def input_resolutions(self) -> List[int]:
        return [layer.input_resolution for layer in self.model.encoder.layers]

    @property
    def outputs_shapes(self) -> List[int]:
        return [layer.dim for layer in self.model.encoder.layers]


class FPNConvLayer(nn.Sequential):
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

        self.block = FPNConvLayer(in_features, in_features)

    def forward(self, down: Tensor, left: Tensor) -> Tensor:
        left = self.proj(left)
        down = interpolate(down, size=left.shape[-2:], mode="nearest")
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
        self.stem = FPNConvLayer(in_features, feature_size)
        self.layers = nn.Sequential(*[FPNLayer(feature_size, lateral_width) for lateral_width in lateral_widths[::-1]])

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        fpn_features: List[Tensor] = []
        last_feature: Tensor = features[-1]
        other_features: List[Tensor] = features[:-1]
        output: Tensor = self.stem(last_feature)
        for layer, left in zip(self.layers, other_features[::-1]):
            x = layer(output, left)
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

    def forward(
        self, features: List[Tensor], output_hidden_states: Optional[bool] = False
    ) -> MaskFormerPixelDecoderOutput:
        fpn_features: List[Tensor] = self.fpn(features)
        # we use the last feature map
        last_feature_projected = self.mask_projection(fpn_features[-1])
        return MaskFormerPixelDecoderOutput(
            last_hidden_state=last_feature_projected, hidden_states=tuple(fpn_features) if output_hidden_states else ()
        )


# copied and adapted from original implementation, also practically equal to DetrSinePositionEmbedding
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
            input_dim (int): The input dimensions
            hidden_dim (int): The hidden dimensions
            output_dim (int): The output dimensions
            num_layers (int, optional): The number of layers. Defaults to 3.
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

    def forward(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = False
    ) -> MaskFormerPixelLevelModuleOutput:
        features: List[Tensor] = self.encoder(pixel_values)
        decoder_output: MaskFormerPixelDecoderOutput = self.decoder(features, output_hidden_states)
        return MaskFormerPixelLevelModuleOutput(
            # the last feature is actually the output from the last layer
            encoder_last_hidden_state=features[-1],
            decoder_last_hidden_state=decoder_output.last_hidden_state,
            encoder_hidden_states=tuple(features) if output_hidden_states else (),
            decoder_hidden_states=decoder_output.hidden_states if output_hidden_states else (),
        )


class MaskFormerTransformerModule(nn.Module):
    """The MaskFormer's transformer module."""

    def __init__(self, in_features: int, config: MaskFormerConfig):
        super().__init__()
        hidden_size: int = config.transformer_decoder.hidden_size
        self.position_embedder = PositionEmbeddingSine(num_pos_feats=hidden_size // 2, normalize=True)
        self.queries_embedder = nn.Embedding(config.transformer_decoder.num_queries, hidden_size)
        should_project = in_features != hidden_size
        self.input_projection = nn.Conv2d(in_features, hidden_size, kernel_size=1) if should_project else None
        self.transformer_decoder = DetrDecoder(config=config.transformer_decoder)

    def forward(self, image_features: Tensor, output_hidden_states: Optional[bool] = False) -> DetrDecoderOutput:
        if self.input_projection is not None:
            image_features = self.input_projection(image_features)
        position_embeddings: Tensor = self.position_embedder(image_features)
        # repeat the queries "q c -> b q c"
        batch_size: int = image_features.shape[0]
        queries_embeddings: Tensor = self.queries_embedder.weight.repeat(1, batch_size, 1)
        inputs_embeds = torch.zeros_like(queries_embeddings)

        batch_size, num_channels, height, width = image_features.shape
        # rearrange both iamge_features and position_embeddings "b c h w -> (h w) b c"
        image_features = image_features.view(batch_size, num_channels, height * width).permute(0, 2, 1)
        position_embeddings = position_embeddings.view(batch_size, num_channels, height * width).permute(0, 2, 1)

        transformer_decoder_output: DetrDecoderOutput = self.transformer_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            encoder_hidden_states=image_features,
            encoder_attention_mask=None,
            position_embeddings=position_embeddings,
            query_position_embeddings=queries_embeddings,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=None,
        )
        return transformer_decoder_output


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

        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).

            [What are attention masks?](../glossary#attention-mask)

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
"""


class MaskFormerPretrainedModel(PreTrainedModel):
    config_class = MaskFormerConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"

    def _init_weights(self, module: nn.Module):
        std = self.config.init_std
        xavier_std = self.config.init_xavier_std
        if isinstance(module, MaskFormerTransformerModule):
            if module.input_projection is not None:
                nn.init.xavier_uniform_(module.input_projection.weight, gain=xavier_std)
                nn.init.constant_(module.input_projection.bias, 0)
        # FPN
        elif isinstance(module, FPNModel):
            nn.init.xavier_uniform_(module.stem[0].weight, gain=xavier_std)

        elif isinstance(module, FPNLayer):
            nn.init.xavier_uniform_(module.proj[0].weight, gain=xavier_std)

        elif isinstance(module, FPNConvLayer):
            nn.init.xavier_uniform_(module[0].weight, gain=xavier_std)
        # The MLP head
        elif isinstance(module, MaskformerMLPPredictionHead):
            # I was not able to find the correct initializer in the original implementation
            # we'll use xavier
            for layer in module:
                nn.init.xavier_uniform_(layer[0].weight, gain=xavier_std)
                nn.init.constant_(layer[0].bias, 0)


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

        self.post_init()

    @add_start_docstrings_to_model_forward(MASKFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskFormerOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
    )
    def forward(
        self,
        pixel_values: Tensor,
        pixel_mask: Optional[Tensor] = None,
        output_hidden_states: Option[bool] = False,
    ) -> MaskFormerOutput:

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        batch_size, _, height, width = pixel_values.shape
        # TODO I am not passing pixel_mask, I need to pass it and use it!!!
        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=pixel_values.device)
        pixel_level_module_output: MaskFormerPixelLevelModuleOutput = self.pixel_level_module(
            pixel_values, output_hidden_states
        )
        image_features: Tensor = pixel_level_module_output.encoder_last_hidden_state
        pixel_embeddings: Tensor = pixel_level_module_output.decoder_last_hidden_state

        transformer_module_output: DetrDecoderOutput = self.transformer_module(image_features, output_hidden_states)
        queries: Tensor = transformer_module_output.last_hidden_state

        return MaskFormerOutput(
            encoder_last_hidden_state=image_features,
            pixel_decoder_last_hidden_state=pixel_embeddings,
            transformer_decoder_last_hidden_state=queries,
            encoder_hidden_states=transformer_module_output.hidden_states if output_hidden_states else (),
            pixel_decoder_hidden_states=pixel_level_module_output.decoder_hidden_states
            if output_hidden_states
            else (),
            transformer_decoder_hidden_states=transformer_module_output.hidden_states if output_hidden_states else (),
        )


class MaskFormerForInstanceSegmentation(MaskFormerPretrainedModel):
    def __init__(self, config: MaskFormerConfig):
        super().__init__(config)
        self.model = MaskFormerModel(config)
        hidden_size: int = config.transformer_decoder.hidden_size
        # + 1 because we add the "null" class
        self.class_predictor = nn.Linear(hidden_size, config.num_labels + 1)
        self.mask_embedder = MaskformerMLPPredictionHead(hidden_size, hidden_size, config.mask_feature_size)

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

        self.post_init()

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

    def get_logits(
        self,
        outputs: MaskFormerOutput,
    ) -> Tuple[Tensor, Tensor, List[str, Tensor]]:
        pixel_embeddings: Tensor = outputs.pixel_decoder_last_hidden_state
        # get the auxilary predictions (one for each decoder's layer)
        auxilary_logits: List[str, Tensor] = []
        # This code is a little bit cumbersome, an improvement can be to return a list of predictions. If we have auxilary loss then we are going to return more than one element in the list
        if self.config.use_auxilary_loss:
            stacked_decoder_outputs: Tensor = torch.stack(outputs.transformer_decoder_hidden_states)
            classes: Tensor = self.class_predictor(stacked_decoder_outputs)
            class_queries_logits: Tensor = classes[-1]
            # get the masks
            mask_embeddings: Tensor = self.mask_embedder(stacked_decoder_outputs)
            # sum up over the channels for each embedding
            binaries_masks: Tensor = torch.einsum("lbqc,   bchw -> lbqhw", mask_embeddings, pixel_embeddings)
            masks_queries_logits: Tensor = binaries_masks[-1]
            # go til [:-1] because the last one is always used
            for aux_binary_masks, aux_classes in zip(binaries_masks[:-1], classes[:-1]):
                auxilary_logits.append({"masks_queries_logits": aux_binary_masks, "class_queries_logits": aux_classes})

        else:
            transformer_decoder_hidden_states: Tensor = outputs.transformer_decoder_last_hidden_state
            classes: Tensor = self.class_predictor(transformer_decoder_hidden_states)
            class_queries_logits: Tensor = classes
            # get the masks
            mask_embeddings: Tensor = self.mask_embedder(transformer_decoder_hidden_states)
            # sum up over the channels
            masks_queries_logits: Tensor = torch.einsum("bqc,   bchw -> bqhw", mask_embeddings, pixel_embeddings)

        return class_queries_logits, masks_queries_logits, auxilary_logits

    def forward(
        self,
        pixel_values: Tensor,
        mask_labels: Optional[Tensor] = None,
        class_labels: Optional[Tensor] = None,
        pixel_mask: Optional[Tensor] = None,
        output_hidden_states: Option[bool] = False,
    ) -> MaskFormerForInstanceSegmentationOutput:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the classification and binary mask loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'mask_labels' (the class labels and masks labels of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of shape (`num_classes) and the mask_labels a `torch.FloatTensor` of shape `(num_classes, height, width)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import MaskFormerFeatureExtractor, MaskFormerForObjectDetection
        >>> from PIL import Image
        >>> import requests
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-ade-640")
        >>> model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade-640")
        >>> inputs = feature_extractor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits
        >>> # you can pass them to feature_extractor for postprocessing
        >>> segmentation = feature_extractor.post_process_segmentation(outputs)
        >>> segmentation = feature_extractor.post_process_panoptic_segmentation(outputs)

        """

        outputs: MaskFormerOutput = self.model(pixel_values, pixel_mask, output_hidden_states)

        class_queries_logits, masks_queries_logits, auxilary_logits = self.get_logits(outputs)

        we_have_labels: bool = mask_labels is not None and class_labels is not None

        if we_have_labels:
            logits: Dict[str, Tensor] = {
                "masks_queries_logits": masks_queries_logits,
                "class_queries_logits": class_queries_logits,
                "auxilary_logits": auxilary_logits,
            }
            labels: Dict[str, Tensor] = {"mask_labels": mask_labels, "class_labels": class_labels}
            loss_dict: Dict[str, Tensor] = self.get_loss_dict(logits, labels)
            loss: Tensor = self.get_loss(loss_dict)

        return MaskFormerForInstanceSegmentationOutput(
            **outputs,
            class_queries_logits=class_queries_logits,
            masks_queries_logits=masks_queries_logits,
            auxilary_logits=auxilary_logits,
            loss_dict=loss_dict if we_have_labels else None,
            loss=loss if we_have_labels else None,
        )
