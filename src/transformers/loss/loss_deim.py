# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""
DEIM (DETR with Improved Matching) Loss Implementation
Modified from D-FINE loss for improved convergence with Matching-Aware Loss (MAL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import is_vision_available
from .loss_for_object_detection import box_iou
from .loss_rt_detr import RTDetrHungarianMatcher, RTDetrLoss

if is_vision_available():
    from transformers.image_transforms import center_to_corners_format


def weighting_function(max_num_bins: int, up: torch.Tensor, reg_scale: int) -> torch.Tensor:
    """
    Generates the non-uniform Weighting Function W(n) for bounding box regression.
    
    Args:
        max_num_bins (int): Max number of the discrete bins.
        up (Tensor): Controls upper bounds of the sequence.
        reg_scale (float): Controls the curvature of the Weighting Function.
    
    Returns:
        Tensor: Sequence of Weighting Function.
    """
    upper_bound1 = abs(up[0]) * abs(reg_scale)
    upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
    step = (upper_bound1 + 1) ** (2 / (max_num_bins - 2))
    left_values = [-((step) ** i) + 1 for i in range(max_num_bins // 2 - 1, 0, -1)]
    right_values = [(step) ** i - 1 for i in range(1, max_num_bins // 2)]
    values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
    values = [v if v.dim() > 0 else v.unsqueeze(0) for v in values]
    values = torch.cat(values, 0)
    return values


def translate_gt(gt: torch.Tensor, max_num_bins: int, reg_scale: int, up: torch.Tensor):
    """
    Decodes bounding box ground truth values into distribution-based GT representations.
    
    Args:
        gt (Tensor): Ground truth bounding box values.
        max_num_bins (int): Maximum number of discrete bins.
        reg_scale (float): Controls the curvature of the Weighting Function.
        up (Tensor): Controls the upper bounds of the Weighting Function.
    
    Returns:
        tuple[Tensor, Tensor, Tensor]: indices, weight_right, weight_left
    """
    gt = gt.reshape(-1)
    function_values = weighting_function(max_num_bins, up, reg_scale)
    
    # Find the closest left-side indices for each value
    diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)
    mask = diffs <= 0
    closest_left_indices = torch.sum(mask, dim=1) - 1
    
    # Calculate the weights for interpolation
    indices = closest_left_indices.float()
    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)
    
    valid_idx_mask = (indices >= 0) & (indices < max_num_bins)
    valid_indices = indices[valid_idx_mask].long()
    
    # Obtain distances
    left_values = function_values[valid_indices]
    right_values = function_values[valid_indices + 1]
    
    left_diffs = torch.abs(gt[valid_idx_mask] - left_values)
    right_diffs = torch.abs(right_values - gt[valid_idx_mask])
    
    # Valid weights
    weight_right[valid_idx_mask] = left_diffs / (left_diffs + right_diffs)
    weight_left[valid_idx_mask] = 1.0 - weight_right[valid_idx_mask]
    
    # Invalid weights (out of range)
    invalid_idx_mask_neg = indices < 0
    weight_right[invalid_idx_mask_neg] = 0.0
    weight_left[invalid_idx_mask_neg] = 1.0
    indices[invalid_idx_mask_neg] = 0.0
    
    invalid_idx_mask_pos = indices >= max_num_bins
    weight_right[invalid_idx_mask_pos] = 1.0
    weight_left[invalid_idx_mask_pos] = 0.0
    indices[invalid_idx_mask_pos] = max_num_bins - 0.1
    
    return indices, weight_right, weight_left


def bbox2distance(points, bbox, max_num_bins, reg_scale, up, eps=0.1):
    """
    Converts bounding box coordinates to distances from a reference point.
    
    Args:
        points (Tensor): (n, 4) [x, y, w, h], where (x, y) is the center.
        bbox (Tensor): (n, 4) bounding boxes in "xyxy" format.
        max_num_bins (float): Maximum bin value.
        reg_scale (float): Controlling curvature of W(n).
        up (Tensor): Controlling upper bounds of W(n).
        eps (float): Small value to ensure target < max_num_bins.
    
    Returns:
        Tensor: Decoded distances with interpolation weights.
    """
    reg_scale = abs(reg_scale)
    left = (points[:, 0] - bbox[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    top = (points[:, 1] - bbox[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    right = (bbox[:, 2] - points[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    bottom = (bbox[:, 3] - points[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    four_lens = torch.stack([left, top, right, bottom], -1)
    four_lens, weight_right, weight_left = translate_gt(four_lens, max_num_bins, reg_scale, up)
    if max_num_bins is not None:
        four_lens = four_lens.clamp(min=0, max=max_num_bins - eps)
    return four_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()


@torch.jit.unused
def _set_aux_loss(outputs_class, outputs_coord):
    """Helper function for auxiliary outputs without corners."""
    return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]


@torch.jit.unused
def _set_aux_loss_with_corners(
    outputs_class, outputs_coord, outputs_corners, outputs_ref, 
    teacher_corners=None, teacher_logits=None
):
    """Helper function for auxiliary outputs with corners and distillation."""
    return [
        {
            "logits": a,
            "pred_boxes": b,
            "pred_corners": c,
            "ref_points": d,
            "teacher_corners": teacher_corners,
            "teacher_logits": teacher_logits,
        }
        for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)
    ]


class DEIMLoss(RTDetrLoss):
    """
    This class computes the losses for DEIM (DETR with Improved Matching).
    The process happens in two steps:
    1) Compute hungarian assignment between ground truth boxes and model outputs
    2) Supervise each pair of matched ground-truth/prediction (class and box)
    
    DEIM introduces the Matching-Aware Loss (MAL) for improved convergence.
    
    Args:
        config: Configuration object containing loss parameters
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.matcher = RTDetrHungarianMatcher(config)
        self.max_num_bins = config.max_num_bins
        self.reg_scale = config.reg_scale
        self.up = nn.Parameter(torch.tensor([config.up]), requires_grad=False)
        
        # DEIM specific parameters
        self.mal_alpha = getattr(config, 'mal_alpha', None)
        self.use_uni_set = getattr(config, 'use_uni_set', True)
        
        # Weight dictionary for DEIM losses
        self.weight_dict = {
            "loss_vfl": getattr(config, 'weight_loss_vfl', 1.0),
            "loss_mal": getattr(config, 'weight_loss_mal', 1.0),
            "loss_bbox": config.weight_loss_bbox,
            "loss_giou": config.weight_loss_giou,
            "loss_fgl": getattr(config, 'weight_loss_fgl', 1.0),
            "loss_ddf": getattr(config, 'weight_loss_ddf', 1.0),
        }
        
        # Select which losses to compute
        self.losses = ["vfl", "mal", "boxes", "local"] if self.mal_alpha is not None else ["vfl", "boxes", "local"]
        
        # Cache for targets
        self.fgl_targets = None
        self.fgl_targets_dn = None
        self.num_pos = None
        self.num_neg = None
    
    def loss_labels_mal(self, outputs, targets, indices, num_boxes):
        """
        Compute Matching-Aware Loss (MAL) for DEIM.
        This loss improves upon VFL by applying gamma power to target scores.
        
        Args:
            outputs: Model outputs containing logits and boxes
            targets: Ground truth targets
            indices: Matching indices from Hungarian matcher
            num_boxes: Number of boxes for normalization
        
        Returns:
            Dictionary with MAL loss
        """
        idx = self._get_source_permutation_idx(indices)
        
        # Compute IoU between matched predictions and targets
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious = torch.diag(
            box_iou(center_to_corners_format(src_boxes), center_to_corners_format(target_boxes))[0]
        ).detach()
        
        # Prepare classification targets
        src_logits = outputs["logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        
        # One-hot encoding
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        
        # Create IoU-aware target scores
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target
        
        # Apply gamma power to target score (key difference from VFL)
        target_score = target_score.pow(self.gamma)
        
        # Compute prediction scores for weighting
        pred_score = F.sigmoid(src_logits).detach()
        
        # Compute weight with optional mal_alpha
        if self.mal_alpha is not None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target
        
        # Binary cross entropy loss
        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        
        return {"loss_mal": loss}
    
    def unimodal_distribution_focal_loss(
        self, pred, label, weight_right, weight_left, weight=None, reduction="sum", avg_factor=None
    ):
        """
        Compute unimodal distribution focal loss for fine-grained localization.
        
        Args:
            pred: Predicted corner distributions
            label: Ground truth corner indices
            weight_right: Interpolation weight for right bin
            weight_left: Interpolation weight for left bin
            weight: Additional weighting factor (e.g., IoU)
            reduction: Reduction method
            avg_factor: Normalization factor
        
        Returns:
            Computed loss value
        """
        dis_left = label.long()
        dis_right = dis_left + 1
        
        loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(-1) + \
               F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(-1)
        
        if weight is not None:
            weight = weight.float()
            loss = loss * weight
        
        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        
        return loss
    
    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """
        Compute Fine-Grained Localization (FGL) and Decoupled Distillation Focal (DDF) losses.
        
        Args:
            outputs: Model outputs with corner predictions
            targets: Ground truth targets
            indices: Matching indices
            num_boxes: Number of boxes for normalization
            T: Temperature for distillation
        
        Returns:
            Dictionary with FGL and DDF losses
        """
        losses = {}
        
        if "pred_corners" in outputs:
            idx = self._get_source_permutation_idx(indices)
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            
            pred_corners = outputs["pred_corners"][idx].reshape(-1, (self.max_num_bins + 1))
            ref_points = outputs["ref_points"][idx].detach()
            
            # Cache target corners computation
            with torch.no_grad():
                is_dn = "is_dn" in outputs
                if is_dn and self.fgl_targets_dn is None:
                    self.fgl_targets_dn = bbox2distance(
                        ref_points,
                        center_to_corners_format(target_boxes),
                        self.max_num_bins,
                        self.reg_scale,
                        self.up,
                    )
                elif not is_dn and self.fgl_targets is None:
                    self.fgl_targets = bbox2distance(
                        ref_points,
                        center_to_corners_format(target_boxes),
                        self.max_num_bins,
                        self.reg_scale,
                        self.up,
                    )
            
            target_corners, weight_right, weight_left = (
                self.fgl_targets_dn if "is_dn" in outputs else self.fgl_targets
            )
            
            # Compute IoU weights
            ious = torch.diag(
                box_iou(
                    center_to_corners_format(outputs["pred_boxes"][idx]),
                    center_to_corners_format(target_boxes)
                )[0]
            )
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
            
            # FGL loss
            losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                weight_targets,
                avg_factor=num_boxes,
            )
            
            # DDF loss (distillation)
            if "teacher_corners" in outputs:
                pred_corners = outputs["pred_corners"].reshape(-1, (self.max_num_bins + 1))
                target_corners = outputs["teacher_corners"].reshape(-1, (self.max_num_bins + 1))
                
                if not torch.equal(pred_corners, target_corners):
                    weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]
                    
                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)
                    
                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(
                        weight_targets_local.dtype
                    )
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
                    
                    # KL divergence loss
                    loss_match_local = (
                        weight_targets_local
                        * (T**2)
                        * (
                            nn.KLDivLoss(reduction="none")(
                                F.log_softmax(pred_corners / T, dim=1),
                                F.softmax(target_corners.detach() / T, dim=1),
                            )
                        ).sum(-1)
                    )
                    
                    # Compute positive/negative balance
                    if "is_dn" not in outputs:
                        batch_scale = 8 / outputs["pred_boxes"].shape[0]
                        self.num_pos = (mask.sum() * batch_scale) ** 0.5
                        self.num_neg = ((~mask).sum() * batch_scale) ** 0.5
                    
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    
                    losses["loss_ddf"] = (
                        loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg
                    ) / (self.num_pos + self.num_neg)
                else:
                    losses["loss_ddf"] = pred_corners.sum() * 0
        
        return losses
    
    def _get_unified_indices(self, indices, indices_aux_list):
        """
        Get a matching union set across all decoder layers.
        This is used when use_uni_set is True for consistent matching.
        
        Args:
            indices: Base matching indices
            indices_aux_list: List of auxiliary matching indices
        
        Returns:
            Unified matching indices
        """
        results = []
        
        # Combine all indices
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]
        
        # Get unique matches per batch
        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        
        return results
    
    def _clear_cache(self):
        """Clear cached targets between forward passes."""
        self.fgl_targets = None
        self.fgl_targets_dn = None
        self.num_pos = None
        self.num_neg = None
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        """
        Route loss computation to appropriate loss function.
        
        Args:
            loss: Loss type string
            outputs: Model outputs
            targets: Ground truth targets
            indices: Matching indices
            num_boxes: Number of boxes for normalization
        
        Returns:
            Computed loss dictionary
        """
        loss_map = {
            "cardinality": self.loss_cardinality,
            "local": self.loss_local,
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            "mal": self.loss_labels_mal,
        }
        
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        
        return loss_map[loss](outputs, targets, indices, num_boxes)


def DEIMForObjectDetectionLoss(
    logits,
    labels,
    device,
    pred_boxes,
    config,
    outputs_class=None,
    outputs_coord=None,
    enc_topk_logits=None,
    enc_topk_bboxes=None,
    denoising_meta_values=None,
    predicted_corners=None,
    initial_reference_points=None,
    **kwargs,
):
    """
    Compute DEIM losses for object detection.
    
    This function handles the complete loss computation including:
    - Main detection losses (MAL/VFL, boxes, local)
    - Auxiliary losses from intermediate layers
    - Denoising losses if applicable
    - Encoder losses if applicable
    
    Args:
        logits: Final layer classification logits
        labels: Ground truth labels
        device: Device for computation
        pred_boxes: Final layer box predictions
        config: Model configuration
        outputs_class: Classification outputs from all decoder layers
        outputs_coord: Box coordinate outputs from all decoder layers
        enc_topk_logits: Encoder top-k logits
        enc_topk_bboxes: Encoder top-k boxes
        denoising_meta_values: Denoising metadata
        predicted_corners: Corner predictions for FGL
        initial_reference_points: Reference points for corner predictions
        **kwargs: Additional arguments
    
    Returns:
        tuple: (total_loss, loss_dict, auxiliary_outputs)
    """
    criterion = DEIMLoss(config)
    criterion.to(device)
    criterion._clear_cache()
    
    # Prepare main outputs
    outputs_loss = {}
    outputs_loss["logits"] = logits
    outputs_loss["pred_boxes"] = pred_boxes.clamp(min=0, max=1)
    
    # Add corner predictions if available
    if predicted_corners is not None:
        outputs_loss["pred_corners"] = predicted_corners[:, :, -1]
        outputs_loss["ref_points"] = initial_reference_points[:, :, -1]
    
    auxiliary_outputs = None
    
    # Handle auxiliary outputs for intermediate layers
    if config.auxiliary_loss:
        if denoising_meta_values is not None:
            # Split denoising and regular outputs
            dn_num_split = denoising_meta_values["dn_num_split"]
            
            dn_out_coord, outputs_coord = torch.split(
                outputs_coord.clamp(min=0, max=1), dn_num_split, dim=2
            )
            dn_out_class, outputs_class = torch.split(outputs_class, dn_num_split, dim=2)
            
            if predicted_corners is not None:
                dn_out_corners, out_corners = torch.split(predicted_corners, dn_num_split, dim=2)
                dn_out_refs, out_refs = torch.split(initial_reference_points, dn_num_split, dim=2)
                
                # Create auxiliary outputs with corners
                auxiliary_outputs = _set_aux_loss_with_corners(
                    outputs_class[:, :-1].transpose(0, 1),
                    outputs_coord[:, :-1].transpose(0, 1),
                    out_corners[:, :-1].transpose(0, 1),
                    out_refs[:, :-1].transpose(0, 1),
                    out_corners[:, -1],
                    outputs_class[:, -1],
                )
                
                # Add denoising auxiliary outputs
                dn_auxiliary_outputs = _set_aux_loss_with_corners(
                    dn_out_class.transpose(0, 1),
                    dn_out_coord.transpose(0, 1),
                    dn_out_corners.transpose(0, 1),
                    dn_out_refs.transpose(0, 1),
                    dn_out_corners[:, -1],
                    dn_out_class[:, -1],
                )
                
                # Mark denoising outputs
                for dn_aux in dn_auxiliary_outputs:
                    dn_aux["is_dn"] = True
                
                outputs_loss["dn_auxiliary_outputs"] = dn_auxiliary_outputs
            else:
                # Create auxiliary outputs without corners
                auxiliary_outputs = _set_aux_loss(
                    outputs_class[:, :-1].transpose(0, 1),
                    outputs_coord[:, :-1].transpose(0, 1),
                )
                
                dn_auxiliary_outputs = _set_aux_loss(
                    dn_out_class.transpose(0, 1),
                    dn_out_coord.transpose(0, 1),
                )
                outputs_loss["dn_auxiliary_outputs"] = dn_auxiliary_outputs
            
            outputs_loss["auxiliary_outputs"] = auxiliary_outputs
            outputs_loss["denoising_meta_values"] = denoising_meta_values
            
            # Add encoder outputs if available
            if enc_topk_logits is not None and enc_topk_bboxes is not None:
                outputs_loss["auxiliary_outputs"].extend(
                    _set_aux_loss([enc_topk_logits], [enc_topk_bboxes.clamp(min=0, max=1)])
                )
        else:
            # No denoising, just regular auxiliary outputs
            if predicted_corners is not None:
                auxiliary_outputs = _set_aux_loss_with_corners(
                    outputs_class[:, :-1].transpose(0, 1),
                    outputs_coord[:, :-1].transpose(0, 1),
                    predicted_corners[:, :-1].transpose(0, 1),
                    initial_reference_points[:, :-1].transpose(0, 1),
                    predicted_corners[:, -1],
                    outputs_class[:, -1],
                )
            else:
                auxiliary_outputs = _set_aux_loss(
                    outputs_class[:, :-1].transpose(0, 1),
                    outputs_coord[:, :-1].transpose(0, 1),
                )
            
            outputs_loss["auxiliary_outputs"] = auxiliary_outputs
            
            # Add encoder outputs
            if enc_topk_logits is not None and enc_topk_bboxes is not None:
                outputs_loss["auxiliary_outputs"].extend(
                    _set_aux_loss([enc_topk_logits], [enc_topk_bboxes.clamp(min=0, max=1)])
                )
    
    # Compute all losses
    loss_dict = criterion(outputs_loss, labels)
    
    # Sum all weighted losses
    loss = sum(loss_dict.values())
    
    return loss, loss_dict, auxiliary_outputs