# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from ..image_transforms import corners_to_center_format
from ..utils import is_scipy_available, is_vision_available, requires_backends


if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

if is_vision_available():
    from ..image_transforms import center_to_corners_format

from .loss_for_object_detection import (
    box_iou,
    dice_loss,
    generalized_box_iou,
    sigmoid_focal_loss,
)


# taken from https://github.com/facebookresearch/sam3/blob/main/sam3/train/matcher.py
class Sam3HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions
    of the network

    For efficiency reasons, the targets don't include the no_object. Because of
    this, in general, there are more predictions than targets. In this case, we
    do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    This is a more efficient implementation of BinaryHungarianMatcher.

    Args:
        class_cost:
            The relative weight of the classification error in the matching cost.
        bbox_cost:
            The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost:
            The relative weight of the giou loss of the bounding box in the matching cost.
    """

    def __init__(self, cost_class: float = 2.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        requires_backends(self, ["scipy"])

        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "pred_logits": Tensor of dim [batch_size, num_queries, 1] with the
                  binary classification logits
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the
                  predicted box coordinates in cxcywh format.
            targets (`dict`):
                A dictionary containing:
                * "boxes_padded": Tensor of dim [batch_size, max_gt_boxes, 4] with
                  ground-truth boxes in cxcywh format, zero-padded for images with
                  fewer than max_gt_boxes targets.
                * "num_boxes": Tensor of dim [batch_size] with the number of
                  ground-truth boxes per image.

        Returns:
            `tuple[Tensor]`: A tuple of three tensors `(batch_idx, src_idx, tgt_idx)` where:
            - `batch_idx` is the batch index for each matched pair
            - `src_idx` is the index of the selected prediction query
            - `tgt_idx` is the index into the packed target tensor `targets["boxes"]`
            For each batch element, len(matched) = min(num_queries, num_target_boxes).
        """
        out_score = outputs["pred_logits"].squeeze(-1)
        out_bbox = outputs["pred_boxes"]
        tgt_bbox = targets["boxes_padded"]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        B, Q = out_bbox.shape[:2]
        max_gt = tgt_bbox.shape[1]
        sizes = targets["num_boxes"].tolist()
        cost_giou = torch.zeros(B, Q, max_gt, device=out_bbox.device)
        for b in range(B):
            n = sizes[b]
            if n > 0:
                cost_giou[b, :, :n] = -generalized_box_iou(
                    center_to_corners_format(out_bbox[b]),
                    center_to_corners_format(tgt_bbox[b, :n]),
                )

        # Focal binary class cost
        alpha = 0.25
        gamma = 2.0
        out_prob = out_score.sigmoid()

        log_out_prob = F.logsigmoid(out_score)
        log_one_minus_out_prob = F.logsigmoid(-out_score)
        cost_class = (
            -alpha * (1 - out_prob) ** gamma * log_out_prob + (1 - alpha) * out_prob**gamma * log_one_minus_out_prob
        )
        cost_class = cost_class.unsqueeze(-1).expand_as(cost_bbox)

        # Final cost matrix
        cost_matrix = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        cost_matrix = cost_matrix.cpu()

        num_boxes = targets["num_boxes"]
        sizes = num_boxes.tolist()
        src_indices = []
        for i, c in enumerate(cost_matrix):
            n_targets = sizes[i]
            if n_targets == 0:
                src_indices.append((torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64)))
            else:
                assignment = linear_sum_assignment(c[:, :n_targets].numpy())
                src_indices.append(
                    (
                        torch.as_tensor(assignment[0], dtype=torch.int64),
                        torch.as_tensor(assignment[1], dtype=torch.int64),
                    )
                )

        tgt_parts, offset = [], 0
        for i, (_, tgt) in enumerate(src_indices):
            # indices could be an empty list (since we remove samples w/ 0 GT boxes)
            if len(tgt) > 0:
                tgt_parts.append(tgt + offset)
            offset += sizes[i]
        tgt_idx = torch.cat(tgt_parts) if tgt_parts else torch.zeros(0, dtype=torch.int64)

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(src_indices)])
        src_idx = torch.cat([src for (src, _) in src_indices])

        return batch_idx, src_idx, tgt_idx


class Sam3Loss(nn.Module):
    """
    This class computes the losses for Sam3Model. The process happens in two steps:
        1) we compute Hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class, boxes, and masks).
    """

    def __init__(self, config):
        super().__init__()
        self.matcher = Sam3HungarianMatcher()
        self.auxiliary_loss = config.auxiliary_loss

        self.weight_dict = {
            "loss_ce": config.ce_loss_coefficient,
            "loss_bbox": config.bbox_loss_coefficient,
            "loss_giou": config.giou_loss_coefficient,
            "loss_mask": config.mask_loss_coefficient,
            "loss_dice": config.dice_loss_coefficient,
        }

        self.ce_pos_weight = config.ce_pos_weight
        self.ce_alpha = config.ce_alpha
        self.ce_gamma = config.ce_gamma
        self.mask_focal_alpha = config.mask_focal_alpha
        self.mask_focal_gamma = config.mask_focal_gamma

    def _prepare_targets(self, labels):
        """
        Convert list[dict] labels to packed target dict.
        Args:
            labels: list of dicts, each with:
                "boxes": (N, 4) xyxy normalized in [0, 1]
                "masks": (N, H, W) binary masks, or None
                "is_valid_mask": (N,) bool, or None

        Returns:
             dict: packed targets with keys "boxes_padded", "num_boxes", "boxes",
                      "boxes_xyxy", "masks", "is_valid_mask"
        """
        batch_size = len(labels)
        boxes_xyxy_list = []
        masks_list = []
        valid_list = []
        num_boxes = []
        has_masks = False

        for label in labels:
            boxes = label["boxes"]  # xyxy
            n = len(boxes)
            num_boxes.append(n)
            if n == 0:
                continue
            boxes_xyxy_list.append(boxes)
            masks = label.get("masks")
            if masks is not None:
                has_masks = True
                masks_list.append(masks)
                valid = label.get("is_valid_mask", torch.ones(n, dtype=torch.bool, device=boxes.device))
                valid_list.append(valid)

        if len(boxes_xyxy_list) == 0:
            device = labels[0]["boxes"].device if len(labels) > 0 else torch.device("cpu")
            return {
                "boxes_padded": torch.zeros(batch_size, 0, 4, device=device),
                "num_boxes": torch.tensor(num_boxes, dtype=torch.int64, device=device),
                "boxes": torch.zeros(0, 4, device=device),
                "boxes_xyxy": torch.zeros(0, 4, device=device),
                "masks": None,
                "is_valid_mask": None,
            }

        boxes_xyxy = torch.cat(boxes_xyxy_list)
        boxes_cxcywh = corners_to_center_format(boxes_xyxy)
        num_boxes_tensor = torch.tensor(num_boxes, dtype=torch.int64)

        max_gt = max(num_boxes)
        device = boxes_xyxy.device
        boxes_padded = torch.zeros(batch_size, max_gt, 4, device=device)
        offset = 0
        for i, n in enumerate(num_boxes):
            if n > 0:
                boxes_padded[i, :n] = boxes_cxcywh[offset : offset + n]
                offset += n

        masks_packed = None
        is_valid_packed = None
        if has_masks:
            masks_packed = torch.cat(masks_list)
            is_valid_packed = torch.cat(valid_list)

        return {
            "boxes_padded": boxes_padded,
            "num_boxes": num_boxes_tensor,
            "boxes": boxes_cxcywh,
            "boxes_xyxy": boxes_xyxy,
            "masks": masks_packed,
            "is_valid_mask": is_valid_packed,
        }

    def loss_ce(self, outputs, targets, indices, num_boxes):
        """IABCEMdetr — IoU-Adaptive Binary Cross Entropy."""
        if "pred_logits" not in outputs:
            raise KeyError("No pred_logits found in outputs")
        if "pred_boxes" not in outputs:
            raise KeyError("No pred_boxes found in outputs")
        src_logits = outputs["pred_logits"].squeeze(-1)
        prob = src_logits.sigmoid()

        with torch.no_grad():
            target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.float, device=src_logits.device)
            target_classes[(indices[0], indices[1])] = 1

            src_boxes_xyxy = center_to_corners_format(outputs["pred_boxes"][(indices[0], indices[1])])
            target_boxes_xyxy = targets["boxes_xyxy"][indices[2]]
            iou_mat, _ = box_iou(src_boxes_xyxy, target_boxes_xyxy)
            iou = torch.diag(iou_mat)
            t = prob[(indices[0], indices[1])] ** self.ce_alpha * iou ** (1 - self.ce_alpha)
            t = torch.clamp(t, min=0.01).detach()

            positive_target = target_classes.clone()
            positive_target[(indices[0], indices[1])] = t

        # Positive loss: BCE with IoU-adaptive target, weighted by pos_weight
        loss_bce = F.binary_cross_entropy_with_logits(src_logits, positive_target, reduction="none")
        loss_bce = loss_bce * target_classes * self.ce_pos_weight

        # Negative loss: BCE with focal weight prob^gamma
        loss_bce = loss_bce + F.binary_cross_entropy_with_logits(src_logits, target_classes, reduction="none") * (
            1 - target_classes
        ) * (prob**self.ce_gamma)

        loss_bce = loss_bce.mean()
        return {"loss_ce": loss_bce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute L1 regression loss and GIoU loss."""
        if "pred_boxes" not in outputs:
            raise KeyError("No pred_boxes found in outputs")

        src_boxes = outputs["pred_boxes"][(indices[0], indices[1])]
        target_boxes = targets["boxes"][indices[2]]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        src_boxes_xyxy = center_to_corners_format(src_boxes)
        target_boxes_xyxy = targets["boxes_xyxy"][indices[2]]

        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy))

        return {
            "loss_bbox": loss_bbox.sum() / num_boxes,
            "loss_giou": loss_giou.sum() / num_boxes,
        }

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute focal loss and dice loss for masks."""
        if "pred_masks" not in outputs:
            raise KeyError("No pred_masks found in outputs")

        src_masks = outputs["pred_masks"]

        if targets["masks"] is None or len(indices[0]) == 0:
            return {
                "loss_mask": torch.tensor(0.0, device=src_masks.device),
                "loss_dice": torch.tensor(0.0, device=src_masks.device),
            }

        target_masks = targets["masks"][indices[2]].to(src_masks)
        keep = targets["is_valid_mask"]
        if keep is not None and indices[2] is not None:
            keep = keep[indices[2]]

        src_masks = src_masks[(indices[0], indices[1])]

        if keep is not None:
            src_masks = src_masks[keep]
            target_masks = target_masks[keep]

        if src_masks.shape[0] == 0:
            return {
                "loss_mask": torch.tensor(0.0, device=src_masks.device),
                "loss_dice": torch.tensor(0.0, device=src_masks.device),
            }

        # Upsample predictions to target size
        if src_masks.shape[-2:] != target_masks.shape[-2:]:
            if len(src_masks.shape) == 3:
                src_masks = src_masks[:, None]
            src_masks = F.interpolate(
                src_masks,
                size=target_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            src_masks = src_masks[:, 0].flatten(1)
        else:
            src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)

        return {
            "loss_mask": sigmoid_focal_loss(
                src_masks,
                target_masks,
                num_boxes,
                alpha=self.mask_focal_alpha,
                gamma=self.mask_focal_gamma,
            ),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }

    def forward(self, outputs, targets):
        """
        Args:
             outputs:
                    "pred_logits": (B, Q, 1) binary classification logits
                    "pred_boxes": (B, Q, 4) cxcywh normalized
                    "pred_masks": (B, Q, H, W) mask logits
                    "auxiliary_outputs": list of dict, each with "pred_logits" and "pred_boxes"
            targets: list[dict] with "boxes" (xyxy), "masks", "is_valid_mask"

        Returns:
            tuple: (total_loss, loss_dict)
        """
        target_dict = self._prepare_targets(targets)

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        indices = self.matcher(outputs_without_aux, target_dict)

        num_boxes = target_dict["num_boxes"].sum().float()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        loss_dict = {}
        loss_dict.update(self.loss_ce(outputs_without_aux, target_dict, indices, num_boxes))
        loss_dict.update(self.loss_boxes(outputs_without_aux, target_dict, indices, num_boxes))
        loss_dict.update(self.loss_masks(outputs_without_aux, target_dict, indices, num_boxes))

        for key, weight in self.weight_dict.items():
            if key in loss_dict:
                loss_dict[key] = loss_dict[key] * weight

        if self.auxiliary_loss and "auxiliary_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["auxiliary_outputs"]):
                aux_indices = self.matcher(aux_outputs, target_dict)
                aux_dict = {}
                aux_dict.update(self.loss_ce(aux_outputs, target_dict, aux_indices, num_boxes))
                aux_dict.update(self.loss_boxes(aux_outputs, target_dict, aux_indices, num_boxes))
                for key, weight in self.weight_dict.items():
                    if key in aux_dict:
                        aux_dict[key] = aux_dict[key] * weight
                for key, value in aux_dict.items():
                    loss_dict[f"{key}_aux_{i}"] = value

        total_loss = sum(loss_dict.values())
        return total_loss, loss_dict
