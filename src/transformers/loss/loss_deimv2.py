# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import torch.nn.functional as F

from ..utils import is_vision_available
from .loss_d_fine import DFineLoss, _set_aux_loss, _set_aux_loss2
from .loss_for_object_detection import box_iou


if is_vision_available():
    from transformers.image_transforms import center_to_corners_format


class Deimv2Loss(DFineLoss):
    def __init__(self, config):
        super().__init__(config)
        self.weight_dict = {
            "loss_mal": config.weight_loss_mal,
            "loss_bbox": config.weight_loss_bbox,
            "loss_giou": config.weight_loss_giou,
            "loss_fgl": config.weight_loss_fgl,
            "loss_ddf": config.weight_loss_ddf,
        }
        self.losses = ["mal", "boxes", "local"]
        self.mal_alpha = config.mal_alpha
        self.use_dense_o2o = config.use_dense_o2o

    def loss_labels_mal(self, outputs, targets, indices, num_boxes):
        idx = self._get_source_permutation_idx(indices)

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(center_to_corners_format(src_boxes), center_to_corners_format(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs["logits"]
        target_classes_o = torch.cat([t["class_labels"][i] for t, (_, i) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha is not None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_mal": loss}

    def _get_dense_o2o_indices(self, indices, indices_aux_list):
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx_pair in unique_sorted:
                row_idx, col_idx = idx_pair[0].item(), idx_pair[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
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


def Deimv2ForObjectDetectionLoss(
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
    criterion = Deimv2Loss(config)
    criterion.to(device)
    outputs_loss = {}
    outputs_loss["logits"] = logits
    outputs_loss["pred_boxes"] = pred_boxes.clamp(min=0, max=1)
    auxiliary_outputs = None
    if config.auxiliary_loss:
        if denoising_meta_values is not None:
            dn_out_coord, outputs_coord = torch.split(
                outputs_coord.clamp(min=0, max=1), denoising_meta_values["dn_num_split"], dim=2
            )
            dn_out_class, outputs_class = torch.split(outputs_class, denoising_meta_values["dn_num_split"], dim=2)
            dn_out_corners, out_corners = torch.split(predicted_corners, denoising_meta_values["dn_num_split"], dim=2)
            dn_out_refs, out_refs = torch.split(initial_reference_points, denoising_meta_values["dn_num_split"], dim=2)

            auxiliary_outputs = _set_aux_loss2(
                outputs_class[:, :-1].transpose(0, 1),
                outputs_coord[:, :-1].transpose(0, 1),
                out_corners[:, :-1].transpose(0, 1),
                out_refs[:, :-1].transpose(0, 1),
                out_corners[:, -1],
                outputs_class[:, -1],
            )

            outputs_loss["auxiliary_outputs"] = auxiliary_outputs
            outputs_loss["auxiliary_outputs"].extend(
                _set_aux_loss([enc_topk_logits], [enc_topk_bboxes.clamp(min=0, max=1)])
            )

            dn_auxiliary_outputs = _set_aux_loss2(
                dn_out_class.transpose(0, 1),
                dn_out_coord.transpose(0, 1),
                dn_out_corners.transpose(0, 1),
                dn_out_refs.transpose(0, 1),
                dn_out_corners[:, -1],
                dn_out_class[:, -1],
            )
            outputs_loss["dn_auxiliary_outputs"] = dn_auxiliary_outputs
            outputs_loss["denoising_meta_values"] = denoising_meta_values

    loss_dict = criterion(outputs_loss, labels)

    loss = sum(loss_dict.values())
    return loss, loss_dict, auxiliary_outputs
