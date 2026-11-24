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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

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


def _set_aux_loss(outputs_class, outputs_coord, outputs_presence=None):
    if outputs_presence is not None:
        return [
            {"logits": a, "pred_boxes": b, "presence_logits": c}
            for a, b, c in zip(outputs_class, outputs_coord, outputs_presence)
        ]
    return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]


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
        config:
            Sam3Config or Sam3LiteTextConfig containing matcher hyperparameters.
    """

    def __init__(self, config):
        super().__init__()
        requires_backends(self, ["scipy", "vision"])

        self.class_cost = config.matcher_class_cost
        self.bbox_cost = config.matcher_bbox_cost
        self.giou_cost = config.matcher_giou_cost
        self.alpha = config.matcher_alpha
        self.gamma = config.matcher_gamma
        if self.class_cost == 0 and self.bbox_cost == 0 and self.giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries] or [batch_size, num_queries, 1] with the
                  binary classification logits
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the
                  predicted box coordinates in cxcywh format.
            targets (`list[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "boxes": Tensor of dim [num_target_boxes, 4] with ground-truth boxes in cxcywh format.

        Returns:
            `list[tuple[Tensor, Tensor]]`: A list of size batch_size, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, len(index_i) = len(index_j) = min(num_queries, num_target_boxes).
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        out_logits = outputs["logits"].reshape(batch_size * num_queries).float()
        out_bbox = outputs["pred_boxes"].flatten(0, 1).float()  # [batch_size * num_queries, 4]
        target_bbox = torch.cat([v["boxes"] for v in targets]).float()

        # Focal binary class cost
        out_prob = out_logits.sigmoid()
        log_out_prob = F.logsigmoid(out_logits)
        log_one_minus_out_prob = F.logsigmoid(-out_logits)
        cost_class = (
            -self.alpha * (1 - out_prob) ** self.gamma * log_out_prob
            + (1 - self.alpha) * out_prob**self.gamma * log_one_minus_out_prob
        )
        cost_class = cost_class.unsqueeze(-1).expand(-1, target_bbox.shape[0])

        cost_bbox = torch.cdist(out_bbox, target_bbox, p=1)
        cost_giou = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        cost_matrix = self.bbox_cost * cost_bbox + self.class_cost * cost_class + self.giou_cost * cost_giou
        max_value = torch.finfo(cost_matrix.dtype).max
        cost_matrix = torch.nan_to_num(cost_matrix, nan=max_value, posinf=max_value, neginf=max_value)
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu().float()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class Sam3Loss(nn.Module):
    """
    This class computes the losses for Sam3Model. The process happens in two steps:
        1) we compute Hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class, presence, boxes, and masks).
    """

    def __init__(self, config):
        super().__init__()
        requires_backends(self, ["scipy", "vision"])
        self.matcher = Sam3HungarianMatcher(config)

        self.weight_dict = {
            "loss_ce": config.ce_loss_coefficient,
            "loss_bbox": config.bbox_loss_coefficient,
            "loss_giou": config.giou_loss_coefficient,
            "loss_mask": config.mask_loss_coefficient,
            "loss_dice": config.dice_loss_coefficient,
            "loss_presence": config.presence_loss_coefficient,
        }
        self.losses = ["ce", "boxes", "masks", "presence"]

        self.ce_pos_weight = config.ce_pos_weight
        self.ce_alpha = config.ce_alpha
        self.ce_gamma = config.ce_gamma
        self.mask_focal_alpha = config.mask_focal_alpha
        self.mask_focal_gamma = config.mask_focal_gamma
        self.presence_alpha = config.presence_alpha
        self.presence_gamma = config.presence_gamma

    def loss_ce(self, outputs, targets, indices, num_boxes):
        """IABCEMdetr — IoU-Adaptive Binary Cross Entropy."""
        if "logits" not in outputs:
            raise KeyError("No logits found in outputs")
        if "pred_boxes" not in outputs:
            raise KeyError("No pred_boxes found in outputs")
        src_logits = outputs["logits"]
        if src_logits.dim() == 3:
            src_logits = src_logits.squeeze(-1)
        prob = src_logits.sigmoid()

        idx = self._get_source_permutation_idx(indices)
        with torch.no_grad():
            target_classes = torch.zeros(src_logits.shape[:2], dtype=src_logits.dtype, device=src_logits.device)
            target_classes[idx] = 1.0
            positive_target = target_classes.clone()

            if len(idx[0]) > 0:
                src_boxes_xyxy = center_to_corners_format(outputs["pred_boxes"][idx])
                target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
                target_boxes_xyxy = center_to_corners_format(target_boxes)
                iou_mat, _ = box_iou(src_boxes_xyxy, target_boxes_xyxy)
                iou = torch.nan_to_num(torch.diag(iou_mat), nan=0.0)
                t = prob[idx] ** self.ce_alpha * iou.to(prob.dtype) ** (1 - self.ce_alpha)
                t = torch.clamp(t, min=0.01).detach()
                positive_target[idx] = t.to(positive_target.dtype)

        # Positive loss: BCE with IoU-adaptive target, weighted by pos_weight
        loss_bce = F.binary_cross_entropy_with_logits(src_logits, positive_target, reduction="none")
        loss_bce = loss_bce * target_classes * self.ce_pos_weight

        # Negative loss: BCE with focal weight prob^gamma
        loss_bce = loss_bce + F.binary_cross_entropy_with_logits(src_logits, target_classes, reduction="none") * (
            1 - target_classes
        ) * (prob**self.ce_gamma)

        if outputs.get("presence_logits") is not None and self.weight_dict["loss_presence"] > 0:
            keep_loss = torch.tensor(
                [len(t["boxes"]) > 0 for t in targets], dtype=loss_bce.dtype, device=loss_bce.device
            ).unsqueeze(-1)
            loss_bce = loss_bce * keep_loss

        loss_bce = loss_bce.mean()
        return {"loss_ce": loss_bce}

    def loss_presence(self, outputs, targets, indices, num_boxes):
        """Compute focal presence loss indicating whether any target exists in each image."""
        if outputs.get("presence_logits") is None:
            return {"loss_presence": torch.tensor(0.0, device=outputs["logits"].device)}

        presence_logits = outputs["presence_logits"].reshape(-1, 1)
        presence_targets = torch.tensor(
            [len(t["boxes"]) > 0 for t in targets],
            dtype=presence_logits.dtype,
            device=presence_logits.device,
        ).unsqueeze(-1)

        batch_size = presence_logits.shape[0]
        return {
            "loss_presence": sigmoid_focal_loss(
                presence_logits,
                presence_targets,
                num_boxes=batch_size,
                alpha=self.presence_alpha,
                gamma=self.presence_gamma,
            )
        }

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute L1 regression loss and GIoU loss."""
        if "pred_boxes" not in outputs:
            raise KeyError("No pred_boxes found in outputs")

        idx = self._get_source_permutation_idx(indices)
        if len(idx[0]) == 0:
            zero = outputs["pred_boxes"].sum(dtype=torch.float32) * 0.0
            return {"loss_bbox": zero, "loss_giou": zero}

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_boxes.dtype)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(src_boxes), center_to_corners_format(target_boxes))
        )

        return {
            "loss_bbox": loss_bbox.sum() / num_boxes,
            "loss_giou": loss_giou.sum() / num_boxes,
        }

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute focal loss and dice loss for masks."""
        src_masks = outputs.get("pred_masks")
        if src_masks is None:
            zero = torch.tensor(0.0, device=outputs["logits"].device, dtype=outputs["logits"].dtype)
            return {"loss_mask": zero, "loss_dice": zero}

        zero = src_masks.sum(dtype=torch.float32) * 0.0
        loss_mask = zero
        loss_dice = zero
        for batch_idx, (target, (src_idx, tgt_idx)) in enumerate(zip(targets, indices)):
            if len(tgt_idx) == 0:
                continue
            masks = target.get("masks")
            if masks is None or masks.shape[0] == 0:
                continue
            valid = target.get("is_valid_mask")
            if valid is not None:
                keep = valid[tgt_idx].bool().cpu()
                src_idx = src_idx[keep]
                tgt_idx = tgt_idx[keep]
            if len(tgt_idx) == 0:
                continue

            cur_src = src_masks[batch_idx, src_idx].float()
            cur_tgt = masks[tgt_idx]
            if cur_src.shape[-2:] != cur_tgt.shape[-2:]:
                cur_src = F.interpolate(
                    cur_src[:, None],
                    size=cur_tgt.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]

            cur_src = cur_src.flatten(1)
            cur_tgt = cur_tgt.to(device=cur_src.device, dtype=torch.float32).flatten(1)

            loss_mask = loss_mask + sigmoid_focal_loss(
                cur_src,
                cur_tgt,
                num_boxes,
                alpha=self.mask_focal_alpha,
                gamma=self.mask_focal_gamma,
            )
            loss_dice = loss_dice + dice_loss(cur_src, cur_tgt, num_boxes)

        return {"loss_mask": loss_mask, "loss_dice": loss_dice}

    def _get_source_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "ce": self.loss_ce,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "presence": self.loss_presence,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """
        Args:
            outputs:
                "logits": (B, Q) or (B, Q, 1) binary classification logits
                "pred_boxes": (B, Q, 4) cxcywh normalized
                "pred_masks": (B, Q, H, W) mask logits
                "presence_logits": (B, 1) optional presence logits
                "auxiliary_outputs": list of dict, each with "logits", "pred_boxes", and optional "presence_logits"
            targets: list[dict] with "boxes" (cxcywh), and optional "masks", "is_valid_mask"

        Returns:
            dict: weighted dictionary of individual loss terms
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs["logits"].device)
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_boxes, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def Sam3ForObjectDetectionLoss(
    logits: torch.Tensor,
    labels: list[dict[str, torch.Tensor]],
    device: torch.device | str | None = None,
    pred_boxes: torch.Tensor | None = None,
    pred_masks: torch.Tensor | None = None,
    presence_logits: torch.Tensor | None = None,
    config=None,
    outputs_class: torch.Tensor | None = None,
    outputs_coord: torch.Tensor | None = None,
    outputs_presence: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[str, torch.Tensor]] | None]:
    if device is None and logits is not None:
        device = logits.device

    criterion = Sam3Loss(config)
    if device is not None:
        criterion.to(device)

    outputs_loss = {
        "logits": logits,
        "pred_boxes": pred_boxes,
        "pred_masks": pred_masks,
        "presence_logits": presence_logits,
    }
    auxiliary_outputs = None
    if config.auxiliary_loss and outputs_class is not None and outputs_coord is not None:
        aux_presence = outputs_presence[:-1] if outputs_presence is not None else None
        auxiliary_outputs = _set_aux_loss(outputs_class[:-1], outputs_coord[:-1], aux_presence)
        outputs_loss["auxiliary_outputs"] = auxiliary_outputs

    loss_dict = criterion(outputs_loss, labels)
    loss = sum(loss_dict.values())
    return loss, loss_dict, auxiliary_outputs
