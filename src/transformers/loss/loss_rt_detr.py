# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import is_accelerate_available, is_scipy_available, is_vision_available, requires_backends
from .loss_for_object_detection import (
    _set_aux_loss,
    generalized_box_iou,
    linear_sum_assignment,
)
from .loss_rt_detr import DeformableDetrImageLoss


if is_accelerate_available():
    pass

if is_scipy_available():
    from scipy.optimize import linear_sum_assignment


if is_vision_available():
    from transformers.image_transforms import center_to_corners_format


class RTDetrHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Args:
        config: RTDetrConfig
    """

    def __init__(self, config):
        super().__init__()
        requires_backends(self, ["scipy"])

        self.class_cost = config.matcher_class_cost
        self.bbox_cost = config.matcher_bbox_cost
        self.giou_cost = config.matcher_giou_cost

        self.use_focal_loss = config.use_focal_loss
        self.alpha = config.matcher_alpha
        self.gamma = config.matcher_gamma

        if self.class_cost == self.bbox_cost == self.giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        # Also concat the target labels and boxes
        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_bbox = torch.cat([v["boxes"] for v in targets])
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["logits"].flatten(0, 1))
            out_prob = out_prob[:, target_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            class_cost = pos_cost_class - neg_cost_class
        else:
            out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            class_cost = -out_prob[:, target_ids]

        # Compute the L1 cost between boxes
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)
        # Compute the giou cost betwen boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))
        # Compute the final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def RTDetrForObjectDetectionLoss(
    logits, labels, device, pred_boxes, config, outputs_class=None, outputs_coord=None, **kwargs
):
    # First: create the matcher
    matcher = RTDetrHungarianMatcher(
        class_cost=config.class_cost, bbox_cost=config.bbox_cost, giou_cost=config.giou_cost
    )
    # Second: create the criterion
    losses = ["labels", "boxes", "cardinality"]
    criterion = DeformableDetrImageLoss(
        matcher=matcher,
        num_classes=config.num_labels,
        focal_alpha=config.focal_alpha,
        losses=losses,
    )
    criterion.to(device)
    # Third: compute the losses, based on outputs and labels
    outputs_loss = {}
    auxiliary_outputs = None
    outputs_loss["logits"] = logits
    outputs_loss["pred_boxes"] = pred_boxes
    if config.auxiliary_loss:
        auxiliary_outputs = _set_aux_loss(outputs_class, outputs_coord)
        outputs_loss["auxiliary_outputs"] = auxiliary_outputs

    loss_dict = criterion(outputs_loss, labels)
    # Fourth: compute total loss, as a weighted sum of the various losses
    weight_dict = {"loss_ce": 1, "loss_bbox": config.bbox_loss_coefficient}
    weight_dict["loss_giou"] = config.giou_loss_coefficient
    if config.auxiliary_loss:
        aux_weight_dict = {}
        for i in range(config.decoder_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    return loss, loss_dict, auxiliary_outputs
