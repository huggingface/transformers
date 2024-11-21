import torch
import torch.nn as nn
from torch.nn import functional as F

from ..image_transforms import center_to_corners_format
from ..utils import is_scipy_available, requires_backends
from .loss_for_object_detection import (
    _set_aux_loss,
    box_iou,
    generalized_box_iou,
)


if is_scipy_available():
    from scipy.optimize import linear_sum_assignment


def reduce_loss(loss, reduction: str = "sum"):
    if reduction == "none":
        return loss

    if reduction == "sum":
        return loss.sum()

    if reduction == "mean":
        num_elements = max(loss.numel(), 1)
        return loss.sum() / num_elements

    raise ValueError("Only sum, mean and none are valid reduction")


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    target_score = targets.to(inputs.dtype)
    weight = (1 - alpha) * prob**gamma * (1 - targets) + targets * alpha * (1 - prob) ** gamma
    # according to original implementation, sigmoid_focal_loss keep gradient on weight
    loss = F.binary_cross_entropy_with_logits(inputs, target_score, reduction="none")
    loss = loss * weight
    # we use sum/num to replace mean to avoid NaN
    return (loss.sum(1) / max(loss.shape[1], 1)).sum() / num_boxes


def vari_sigmoid_focal_loss(
    inputs, targets, gt_score, num_boxes, alpha: float = 0.25, gamma: float = 2
):
    prob = (
        inputs.sigmoid().detach()
    )
    target_score = targets * gt_score.unsqueeze(-1)
    weight = (1 - alpha) * prob.pow(gamma) * (1 - targets) + target_score
    loss = F.binary_cross_entropy_with_logits(inputs, target_score, weight=weight, reduction="none")
    # we use sum/num to replace mean to avoid NaN
    return (loss.sum(1) / max(loss.shape[1], 1)).sum() / num_boxes


def score_aware_vari_sigmoid_focal_loss(
    inputs, targets, gt_score, alpha: float = 0.25, gamma: float = 2, reduction: str = "sum"
):
    prob = inputs.sigmoid().detach()
    target_score = targets * gt_score
    # NOTE: The only difference from sigmoid_focal_loss
    pos_weight = target_score  # pos_weight = gt_score.unsqueeze(-1)
    neg_weight = (1 - alpha) * prob.pow(gamma)
    weight = neg_weight * (1 - targets) + targets * pos_weight
    loss = F.binary_cross_entropy_with_logits(
        inputs, target_score, weight=weight, reduction="none"
    )
    return reduce_loss(loss, reduction)


class RelationDetrHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Args:
        class_cost:
            The relative weight of the classification error in the matching cost.
        bbox_cost:
            The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost:
            The relative weight of the giou loss of the bounding box in the matching cost.
    """

    def __init__(
        self,
        class_cost: float = 1,
        bbox_cost: float = 1,
        giou_cost: float = 1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Differences:
        - out_prob = outputs["logits"].flatten(0, 1).sigmoid() instead of softmax
        - class_cost uses alpha and gamma
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = (
            outputs["logits"].flatten(0, 1).sigmoid()
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        class_cost = pos_cost_class[:, target_ids] - neg_cost_class[:, target_ids]

        # Compute the L1 cost between boxes
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(
            center_to_corners_format(out_bbox), center_to_corners_format(target_bbox)
        )

        # Final cost matrix
        cost_matrix = (
            self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        )
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


class RelationDetrLoss(nn.Module):
    """
    This class computes the losses for RelationDetr. The process happens in two steps: 1) we compute hungarian assignment
    between ground truth boxes and the outputs of the model 2) we supervise each pair of matched ground-truth /
    prediction (supervise class and box).

    Args:
        matcher (`DetrHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        weight_dict (`Dict`):
            Dictionary relating each loss with its weights. These losses are configured in RelationDetrConf as
            `weight_loss_vfl`, `weight_loss_bbox`, `weight_loss_giou`
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
        alpha (`float`):
            Parameter alpha used to compute the focal loss.
        gamma (`float`):
            Parameter gamma used to compute the focal loss.
        eos_coef (`float`):
            Relative classification weight applied to the no-object category.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
    """

    def __init__(self, config):
        super().__init__()

        self.matcher = RelationDetrHungarianMatcher(config)
        self.num_classes = config.num_labels
        self.weight_dict = {
            "loss_vfl": config.weight_loss_vfl,
            "loss_bbox": config.weight_loss_bbox,
            "loss_giou": config.weight_loss_giou,
        }
        self.losses = ["vfl", "boxes"]
        self.eos_coef = config.eos_coefficient
        empty_weight = torch.ones(config.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.alpha = config.focal_loss_alpha
        self.gamma = config.focal_loss_gamma

    # removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (Binary focal loss) targets dicts must contain the key "class_labels" containing a tensor
        of dim [nb_target_boxes]
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        iou_score = torch.diag(
            box_iou(center_to_corners_format(src_boxes), center_to_corners_format(target_boxes))
        ).detach()

        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        # construct onehot targets, shape: (batch_size, num_queries, num_classes)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target_classes_onehot = F.one_hot(target_classes, self.num_classes + 1)[..., :-1]

        # construct iou_score, shape: (batch_size, num_queries)
        target_score = torch.zeros_like(target_classes, dtype=iou_score.dtype)
        target_score[idx] = iou_score

        loss_class = (
            score_aware_vari_sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                target_score.unsqueeze(-1),
                alpha=self.alpha,
                gamma=self.gamma,
            )
            / num_boxes
        )
        losses = {"loss_class": loss_class}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss. Targets dicts must
        contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes are expected in
        format (center_x, center_y, w, h), normalized by the image size.
        """
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        idx = self._get_source_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                center_to_corners_format(src_boxes), center_to_corners_format(target_boxes)
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
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
        loss_map = {"labels": self.loss_labels, "boxes": self.loss_boxes}
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t["class_labels"]) for t in targets]
        device = targets[0]["class_labels"].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=device),
                        torch.zeros(0, dtype=torch.int64, device=device),
                    )
                )

        return dn_match_indices

    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`List[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if "auxiliary_outputs" not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if "dn_auxiliary_outputs" in outputs:
            if "denoising_meta_values" not in outputs:
                raise ValueError(
                    "The output must have the 'denoising_meta_values` key. Please, ensure that 'outputs' includes a 'denoising_meta_values' entry."
                )
            indices = self.get_cdn_matched_indices(outputs["denoising_meta_values"], targets)
            num_boxes = num_boxes * outputs["denoising_meta_values"]["dn_num_group"]

            for i, auxiliary_outputs in enumerate(outputs["dn_auxiliary_outputs"]):
                # indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(
                        loss, auxiliary_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def RelationDetrForObjectDetectionLoss(
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
    **kwargs,
):
    criterion = RelationDetrLoss(config)
    criterion.to(device)
    # Second: compute the losses, based on outputs and labels
    outputs_loss = {}
    outputs_loss["logits"] = logits
    outputs_loss["pred_boxes"] = pred_boxes
    if config.auxiliary_loss:
        if denoising_meta_values is not None:
            dn_out_coord, outputs_coord = torch.split(
                outputs_coord, denoising_meta_values["dn_num_split"], dim=2
            )
            dn_out_class, outputs_class = torch.split(
                outputs_class, denoising_meta_values["dn_num_split"], dim=2
            )

        auxiliary_outputs = _set_aux_loss(
            outputs_class[:, :-1].transpose(0, 1), outputs_coord[:, :-1].transpose(0, 1)
        )
        outputs_loss["auxiliary_outputs"] = auxiliary_outputs
        outputs_loss["auxiliary_outputs"].extend(
            _set_aux_loss([enc_topk_logits], [enc_topk_bboxes])
        )
        if denoising_meta_values is not None:
            outputs_loss["dn_auxiliary_outputs"] = _set_aux_loss(
                dn_out_class.transpose(0, 1), dn_out_coord.transpose(0, 1)
            )
            outputs_loss["denoising_meta_values"] = denoising_meta_values

    loss_dict = criterion(outputs_loss, labels)

    loss = sum(loss_dict.values())
    return loss, loss_dict, auxiliary_outputs
