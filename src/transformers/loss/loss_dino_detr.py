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
import copy

import torch
import torch.nn as nn

from ..utils import (
    is_accelerate_available,
    is_scipy_available,
    is_vision_available,
)


if is_accelerate_available():
    from accelerate import PartialState
    from accelerate.utils import reduce

if is_scipy_available():
    pass


if is_vision_available():
    from transformers.image_transforms import center_to_corners_format

from .loss_for_object_detection import (
    HungarianMatcher,
    _set_aux_loss,
    dice_loss,
    generalized_box_iou,
    nested_tensor_from_tensor_list,
    sigmoid_focal_loss,
)


class DinoDetrImageLoss(nn.Module):
    """
    This class computes the losses for DetrForObjectDetection/DetrForSegmentation. The process happens in two steps: 1)
    we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair
    of matched ground-truth / prediction (supervise class and box).

    A note on the `num_classes` argument (copied from original repo in detr.py): "the naming of the `num_classes`
    parameter of the criterion is somewhat misleading. It indeed corresponds to `max_obj_id` + 1, where `max_obj_id` is
    the maximum id for a class in your dataset. For example, COCO has a `max_obj_id` of 90, so we pass `num_classes` to
    be 91. As another example, for a dataset that has a single class with `id` 1, you should pass `num_classes` to be 2
    (`max_obj_id` + 1). For more details on this, check the following discussion
    https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223"


    Args:
        matcher (`DetrHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        eos_coef (`float`):
            Relative classification weight applied to the no-object category.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """

    def __init__(self, num_classes, matcher, focal_alpha, losses):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.focal_alpha = focal_alpha
        self.losses = losses

    # removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim
        [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]

        idx = self._get_source_permutation_idx(indices)
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            source_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=source_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [
                source_logits.shape[0],
                source_logits.shape[1],
                source_logits.shape[2] + 1,
            ],
            dtype=source_logits.dtype,
            layout=source_logits.layout,
            device=source_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            sigmoid_focal_loss(
                source_logits,
                target_classes_onehot,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=2,
            )
            * source_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        return losses

    @torch.no_grad()
    def cardinality_error(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]
        device = logits.device
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                center_to_corners_format(source_boxes),
                center_to_corners_format(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        if "pred_masks" not in outputs:
            raise KeyError("No predicted masks found in outputs")

        source_idx = self._get_source_permutation_idx(indices)
        target_idx = self._get_target_permutation_idx(indices)
        source_masks = outputs["pred_masks"]
        source_masks = source_masks[source_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(source_masks)
        target_masks = target_masks[target_idx]

        # upsample predictions to the target size
        source_masks = nn.functional.interpolate(
            source_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        source_masks = source_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(source_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(source_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(source_masks, target_masks, num_boxes),
        }
        return losses

    def _get_source_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    def _get_target_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "class_labels": self.loss_labels,
            "cardinality": self.cardinality_error,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets, return_indices=False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        device = next(iter(outputs.values())).device
        indices = self.matcher(outputs_without_aux, targets)

        indices0_copy = indices
        indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        world_size = 1
        if is_accelerate_available():
            if PartialState._shared_state != {}:
                num_boxes = reduce(num_boxes)
                world_size = PartialState().num_processes
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for denoising loss
        denoising_meta = outputs["denoising_meta"]

        if self.training and denoising_meta and "output_known_lbs_bboxes" in denoising_meta:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_denoising(denoising_meta)

            denoising_pos_idx = []
            denoising_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]["class_labels"]) > 0:
                    t = torch.range(0, len(targets[i]["class_labels"]) - 1).long().to(device)
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().to(device).unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().to(device)

                denoising_pos_idx.append((output_idx, tgt_idx))
                denoising_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes = denoising_meta["output_known_lbs_bboxes"]
            l_dict = {}
            for loss in self.losses:
                l_dict.update(
                    self.get_loss(
                        loss,
                        output_known_lbs_bboxes,
                        targets,
                        denoising_pos_idx,
                        num_boxes * scalar,
                    )
                )

            l_dict = {k + "_denoising": v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            l_dict = {
                "loss_bbox_denoising": torch.as_tensor(0.0).to(device),
                "loss_giou_denoising": torch.as_tensor(0.0).to(device),
                "loss_ce_denoising": torch.as_tensor(0.0).to(device),
                "loss_xy_denoising": torch.as_tensor(0.0).to(device),
                "loss_hw_denoising": torch.as_tensor(0.0).to(device),
                "cardinality_error_denoising": torch.as_tensor(0.0).to(device),
            }
            losses.update(l_dict)

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for idx, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{idx}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and denoising_meta and "output_known_lbs_bboxes" in denoising_meta:
                    aux_outputs_known = output_known_lbs_bboxes["aux_outputs"][idx]
                    l_dict = {}
                    for loss in self.losses:
                        l_dict.update(
                            self.get_loss(
                                loss,
                                aux_outputs_known,
                                targets,
                                denoising_pos_idx,
                                num_boxes * scalar,
                            )
                        )

                    l_dict = {k + f"_denoising_{idx}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    l_dict = {
                        "loss_bbox_denoising": torch.as_tensor(0.0).to(device),
                        "loss_giou_denoising": torch.as_tensor(0.0).to(device),
                        "loss_ce_denoising": torch.as_tensor(0.0).to(device),
                        "loss_xy_denoising": torch.as_tensor(0.0).to(device),
                        "loss_hw_denoising": torch.as_tensor(0.0).to(device),
                        "cardinality_error_denoising": torch.as_tensor(0.0).to(device),
                    }
                    l_dict = {k + f"_{idx}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if "interm_outputs" in outputs:
            interm_outputs = outputs["interm_outputs"]
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes)
                l_dict = {k + "_interm": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if "enc_outputs" in outputs:
            for i, enc_outputs in enumerate(outputs["enc_outputs"]):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_denoising(self, denoising_meta):
        output_known_lbs_bboxes = denoising_meta["output_known_lbs_bboxes"]
        denoising_num_groups, num_denoising_queries = (
            denoising_meta["dn_num_group"],
            denoising_meta["dn_num_split"][0],
        )
        single_query = num_denoising_queries // denoising_num_groups

        return output_known_lbs_bboxes, single_query, denoising_num_groups


def DinoDetrForObjectDetectionLoss(
    logits,
    labels,
    device,
    pred_boxes,
    denoising_meta,
    class_cost,
    bbox_cost,
    giou_cost,
    num_labels,
    focal_alpha,
    auxiliary_loss,
    cls_loss_coefficient,
    bbox_loss_coefficient,
    giou_loss_coefficient,
    mask_loss_coefficient,
    use_denoising,
    use_masks,
    dice_loss_coefficient,
    num_decoder_layers,
    outputs_class=None,
    outputs_coord=None,
):
    # First: create the matcher
    matcher = HungarianMatcher(
        class_cost=class_cost,
        bbox_cost=bbox_cost,
        giou_cost=giou_cost,
    )
    # Second: create the criterion
    losses = ["class_labels", "boxes", "cardinality"]
    criterion = DinoDetrImageLoss(
        num_classes=num_labels,
        matcher=matcher,
        focal_alpha=focal_alpha,
        losses=losses,
    )
    criterion.to(device)
    # Third: compute the losses, based on outputs and labels
    outputs_loss = {}
    auxiliary_outputs = None
    outputs_loss["logits"] = logits
    outputs_loss["pred_boxes"] = pred_boxes
    outputs_loss["denoising_meta"] = denoising_meta
    if auxiliary_loss:
        auxiliary_outputs = _set_aux_loss(outputs_class, outputs_coord)
        outputs_loss["auxiliary_outputs"] = auxiliary_outputs

    loss_dict = criterion(outputs_loss, labels)
    # Fourth: compute total loss, as a weighted sum of the various losses
    weight_dict = compute_weight_dict(
        cls_loss_coefficient,
        bbox_loss_coefficient,
        giou_loss_coefficient,
        mask_loss_coefficient,
        use_denoising,
        use_masks,
        dice_loss_coefficient,
        auxiliary_loss,
        num_decoder_layers,
    )
    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    return loss, loss_dict, auxiliary_outputs


def compute_weight_dict(
    cls_loss_coefficient,
    bbox_loss_coefficient,
    giou_loss_coefficient,
    mask_loss_coefficient,
    use_denoising,
    use_masks,
    dice_loss_coefficient,
    auxiliary_loss,
    num_decoder_layers,
):
    # prepare weight dict
    weight_dict = {
        "loss_ce": cls_loss_coefficient,
        "loss_bbox": bbox_loss_coefficient,
    }
    weight_dict["loss_giou"] = giou_loss_coefficient
    clean_weight_dict_wo_denoising = copy.deepcopy(weight_dict)

    # for denoising training
    if use_denoising:
        weight_dict["loss_ce_denoising"] = cls_loss_coefficient
        weight_dict["loss_bbox_denoising"] = bbox_loss_coefficient
        weight_dict["loss_giou_denoising"] = giou_loss_coefficient

    if use_masks:
        weight_dict["loss_mask"] = mask_loss_coefficient
        weight_dict["loss_dice"] = dice_loss_coefficient
    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if auxiliary_loss:
        aux_weight_dict = {}
        for i in range(num_decoder_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    interm_weight_dict = {}
    _coeff_weight_dict = {
        "loss_ce": 1.0,
        "loss_bbox": 1.0,
        "loss_giou": 1.0,
    }
    interm_weight_dict.update(
        {k + "_interm": v * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_denoising.items()}
    )
    weight_dict.update(interm_weight_dict)
    return weight_dict
