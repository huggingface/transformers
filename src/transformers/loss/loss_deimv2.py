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
        self.use_dense_one_to_one = config.use_dense_one_to_one

    def loss_labels_mal(self, outputs, targets, indices, num_boxes):
        """Compute the Matching Aware Loss (MAL), which uses IoU-weighted soft labels
        instead of hard one-hot targets, with focal-style weighting controlled by `mal_alpha`.
        """
        idx = self._get_source_permutation_idx(indices)

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(center_to_corners_format(src_boxes), center_to_corners_format(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs["logits"]
        target_classes_original = torch.cat([t["class_labels"][i] for t, (_, i) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_original
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_original = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_original[idx] = ious.to(target_score_original.dtype)
        target_score = target_score_original.unsqueeze(-1) * target

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

        for index in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(index, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx_pair in unique_sorted:
                row_idx, col_idx = idx_pair[0].item(), idx_pair[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=index.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=index.device)
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

    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`list[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        if not self.use_dense_one_to_one:
            return super().forward(outputs, targets)

        # Retrieve the matching between the outputs of the last layer and the targets
        outputs_without_aux = {k: v for k, v in outputs.items() if "auxiliary_outputs" not in k}
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Handle auxiliary outputs matching
        cached_indices = []
        indices_aux_list = []
        if "auxiliary_outputs" in outputs:
            for auxiliary_outputs in outputs["auxiliary_outputs"]:
                aux_indices = self.matcher(auxiliary_outputs, targets)
                cached_indices.append(aux_indices)
                indices_aux_list.append(aux_indices)

        # Dense one-to-one matching
        indices_go = self._get_dense_o2o_indices(indices, indices_aux_list)
        num_boxes_go = sum(len(x[0]) for x in indices_go)
        num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes_go = torch.clamp(num_boxes_go, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            use_union = loss in ("boxes", "local")
            indices_in = indices_go if use_union else indices
            num_boxes_in = num_boxes_go if use_union else num_boxes
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                for loss in self.losses:
                    use_union = loss in ("boxes", "local")
                    indices_in = indices_go if use_union else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_union else num_boxes
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices_in, num_boxes_in)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For deimv2
        if "dn_auxiliary_outputs" in outputs:
            if "denoising_meta_values" not in outputs:
                raise ValueError(
                    "The output must have the 'denoising_meta_values` key. "
                    "Please, ensure that 'outputs' includes a 'denoising_meta_values' entry."
                )
            dn_indices = self.get_cdn_matched_indices(outputs["denoising_meta_values"], targets)
            dn_num_boxes = num_boxes * outputs["denoising_meta_values"]["dn_num_group"]
            for i, auxiliary_outputs in enumerate(outputs["dn_auxiliary_outputs"]):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, dn_indices, dn_num_boxes)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


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

    outputs_loss = {"logits": logits, "pred_boxes": pred_boxes.clamp(min=0, max=1)}
    auxiliary_outputs = None

    if config.auxiliary_loss:
        if denoising_meta_values is not None:
            dn_out_coord, normal_out_coord = torch.split(
                outputs_coord.clamp(min=0, max=1), denoising_meta_values["dn_num_split"], dim=2
            )
            dn_out_class, normal_out_class = torch.split(outputs_class, denoising_meta_values["dn_num_split"], dim=2)
            # https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/deim/deim_decoder.py#L562-L571
            # The original splits denoising queries in the decoder; here it happens in the loss since the decoder returns unsplit tensors.
            _, normal_logits = torch.split(logits, denoising_meta_values["dn_num_split"], dim=1)
            _, normal_pred_boxes = torch.split(pred_boxes, denoising_meta_values["dn_num_split"], dim=1)
            dn_out_corners, out_corners = torch.split(predicted_corners, denoising_meta_values["dn_num_split"], dim=2)
            dn_out_refs, out_refs = torch.split(initial_reference_points, denoising_meta_values["dn_num_split"], dim=2)

            outputs_loss["logits"] = normal_logits
            outputs_loss["pred_boxes"] = normal_pred_boxes.clamp(min=0, max=1)
        else:
            normal_out_coord = outputs_coord.clamp(min=0, max=1)
            normal_out_class = outputs_class
            out_corners = predicted_corners
            out_refs = initial_reference_points

        auxiliary_outputs = _set_aux_loss2(
            normal_out_class[:, :-1].transpose(0, 1),
            normal_out_coord[:, :-1].transpose(0, 1),
            out_corners[:, :-1].transpose(0, 1),
            out_refs[:, :-1].transpose(0, 1),
            out_corners[:, -1],
            normal_out_class[:, -1],
        )

        outputs_loss["auxiliary_outputs"] = auxiliary_outputs
        outputs_loss["auxiliary_outputs"].extend(
            _set_aux_loss([enc_topk_logits], [enc_topk_bboxes.clamp(min=0, max=1)])
        )

        if denoising_meta_values is not None:
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
