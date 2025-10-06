import torch
import torch.nn as nn

from ..image_transforms import center_to_corners_format
from ..utils import is_scipy_available
from .loss_for_object_detection import (
    HungarianMatcher,
    ImageLoss,
    _set_aux_loss,
    generalized_box_iou,
    sigmoid_focal_loss,
)


if is_scipy_available():
    from scipy.optimize import linear_sum_assignment


class DeformableDetrHungarianMatcher(HungarianMatcher):
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Differences:
        - out_prob = outputs["logits"].flatten(0, 1).sigmoid() instead of softmax
        - class_cost uses alpha and gamma
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
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
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class DeformableDetrImageLoss(ImageLoss):
    def __init__(self, matcher, num_classes, focal_alpha, losses):
        nn.Module.__init__(self)
        self.matcher = matcher
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.losses = losses

    # removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (Binary focal loss) targets dicts must contain the key "class_labels" containing a tensor
        of dim [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]

        idx = self._get_source_permutation_idx(indices)
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [source_logits.shape[0], source_logits.shape[1], source_logits.shape[2] + 1],
            dtype=source_logits.dtype,
            layout=source_logits.layout,
            device=source_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            sigmoid_focal_loss(source_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
            * source_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        return losses


def DeformableDetrForSegmentationLoss(
    logits, labels, device, pred_boxes, pred_masks, config, outputs_class=None, outputs_coord=None, **kwargs
):
    # First: create the matcher
    matcher = HungarianMatcher(class_cost=config.class_cost, bbox_cost=config.bbox_cost, giou_cost=config.giou_cost)
    # Second: create the criterion
    losses = ["labels", "boxes", "cardinality", "masks"]
    criterion = DeformableDetrImageLoss(
        matcher=matcher,
        num_classes=config.num_labels,
        focal_alpha=config.focal_alpha,
        losses=losses,
    )
    criterion.to(device)
    # Third: compute the losses, based on outputs and labels
    outputs_loss = {}
    outputs_loss["logits"] = logits
    outputs_loss["pred_boxes"] = pred_boxes
    outputs_loss["pred_masks"] = pred_masks

    auxiliary_outputs = None
    if config.auxiliary_loss:
        auxiliary_outputs = _set_aux_loss(outputs_class, outputs_coord)
        outputs_loss["auxiliary_outputs"] = auxiliary_outputs

    loss_dict = criterion(outputs_loss, labels)
    # Fourth: compute total loss, as a weighted sum of the various losses
    weight_dict = {"loss_ce": 1, "loss_bbox": config.bbox_loss_coefficient}
    weight_dict["loss_giou"] = config.giou_loss_coefficient
    weight_dict["loss_mask"] = config.mask_loss_coefficient
    weight_dict["loss_dice"] = config.dice_loss_coefficient
    if config.auxiliary_loss:
        aux_weight_dict = {}
        for i in range(config.decoder_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
    return loss, loss_dict, auxiliary_outputs


def DeformableDetrForObjectDetectionLoss(
    logits, labels, device, pred_boxes, config, outputs_class=None, outputs_coord=None, **kwargs
):
    # First: create the matcher
    matcher = DeformableDetrHungarianMatcher(
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
    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
    return loss, loss_dict, auxiliary_outputs
