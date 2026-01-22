import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import requires_backends
from ..image_transforms import center_to_corners_format
from ..utils import is_accelerate_available, is_scipy_available, is_vision_available
from .loss_for_object_detection import (
    dice_loss,
    generalized_box_iou,
)
from .loss_lw_detr import LwDetrImageLoss


if is_vision_available():
    pass

if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

if is_accelerate_available():
    from accelerate import PartialState
    from accelerate.utils import reduce


def sigmoid_cross_entropy_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.mean(1).sum() / num_masks
    return loss


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio=3, importance_sample_ratio=0.75
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def batch_sigmoid_cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))

    return loss / hw


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


class RfDetrHungarianMatcher(nn.Module):
    def __init__(
        self,
        class_cost: float = 1,
        bbox_cost: float = 1,
        giou_cost: float = 1,
        mask_point_sample_ratio: int = 16,
        cost_mask_class_cost: float = 1,
        cost_mask_dice_cost: float = 1,
    ):
        super().__init__()
        requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

        self.mask_point_sample_ratio = mask_point_sample_ratio
        self.cost_mask_class = cost_mask_class_cost
        self.cost_mask_dice = cost_mask_dice_cost

    @torch.no_grad()
    def forward(self, outputs, targets, group_detr):
        """
        Differences:
        - out_prob = outputs["logits"].flatten(0, 1).sigmoid() instead of softmax
        - class_cost uses alpha and gamma
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        out_masks = outputs["pred_masks"].flatten(0, 1)  # [batch_size * num_queries, H, W]

        # Also concat the target labels and boxes
        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_bbox = torch.cat([v["boxes"] for v in targets])
        target_masks = torch.cat([v["masks"] for v in targets])

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        class_cost = pos_cost_class[:, target_ids] - neg_cost_class[:, target_ids]

        # Compute the L1 cost between boxes, cdist only supports float32
        dtype = out_bbox.dtype
        out_bbox = out_bbox.to(torch.float32)
        target_bbox = target_bbox.to(torch.float32)
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)
        bbox_cost = bbox_cost.to(dtype)

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # Compute mask cost
        num_points = out_masks.shape[-2] * out_masks.shape[-1] // self.mask_point_sample_ratio
        tgt_masks = target_masks.to(out_masks.dtype)
        torch.manual_seed(0)  # TODO REMOVE
        point_coords = torch.rand(1, num_points, 2, device=out_masks.device)
        pred_masks_logits = point_sample(
            out_masks.unsqueeze(1), point_coords.repeat(out_masks.shape[0], 1, 1), align_corners=False
        ).squeeze(1)
        tgt_masks_flat = point_sample(
            tgt_masks.unsqueeze(1), point_coords.repeat(tgt_masks.shape[0], 1, 1), align_corners=False, mode="nearest"
        ).squeeze(1)

        cost_mask_class = batch_sigmoid_cross_entropy_loss(pred_masks_logits, tgt_masks_flat)
        cost_mask_dice = batch_dice_loss(pred_masks_logits, tgt_masks_flat)

        # Final cost matrix
        cost_matrix = (
            self.bbox_cost * bbox_cost
            + self.class_cost * class_cost
            + self.giou_cost * giou_cost
            + self.cost_mask_class * cost_mask_class
            + self.cost_mask_dice * cost_mask_dice
        )
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        # we assume any good match will not cause NaN or Inf, so we replace them with a large value
        max_cost = cost_matrix.max() if cost_matrix.numel() > 0 else 0
        cost_matrix[cost_matrix.isinf() | cost_matrix.isnan()] = max_cost * 2

        # Hungarian matching
        sizes = [len(v["masks"]) for v in targets]
        indices = []
        group_num_queries = num_queries // group_detr
        cost_matrix_list = cost_matrix.split(group_num_queries, dim=1)
        for group_id in range(group_detr):
            group_cost_matrix = cost_matrix_list[group_id]
            group_indices = [linear_sum_assignment(c[i]) for i, c in enumerate(group_cost_matrix.split(sizes, -1))]
            if group_id == 0:
                indices = group_indices
            else:
                indices = [
                    (
                        np.concatenate([indice1[0], indice2[0] + group_num_queries * group_id]),
                        np.concatenate([indice1[1], indice2[1]]),
                    )
                    for indice1, indice2 in zip(indices, group_indices)
                ]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class RfDetrImageLoss(LwDetrImageLoss):
    def __init__(self, matcher, num_classes, focal_alpha, losses, group_detr, mask_point_sample_ratio):
        nn.Module.__init__(self)
        self.matcher = matcher
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.losses = losses
        self.group_detr = group_detr
        self.mask_point_sample_ratio = mask_point_sample_ratio

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        if "pred_masks" not in outputs:
            raise KeyError("No predicted masks found in outputs")

        source_idx = self._get_source_permutation_idx(indices)
        source_masks = outputs["pred_masks"]
        source_masks = source_masks[source_idx]
        if source_masks.numel() == 0:
            return {
                "loss_mask_ce": source_masks.sum(),
                "loss_mask_dice": source_masks.sum(),
            }

        # gather matched target masks
        target_masks = torch.cat([t["masks"][j] for t, (_, j) in zip(targets, indices)], dim=0)

        source_masks = source_masks.unsqueeze(1)
        target_masks = target_masks.unsqueeze(1).float()

        num_points = max(
            source_masks.shape[-2], source_masks.shape[-2] * source_masks.shape[-1] // self.mask_point_sample_ratio
        )

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                source_masks,
                lambda logits: calculate_uncertainty(logits),
                num_points,
                3,
                0.75,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
                mode="nearest",
            ).squeeze(1)

        point_logits = point_sample(
            source_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask_ce": sigmoid_cross_entropy_loss(point_logits, point_labels, num_boxes),
            "loss_mask_dice": dice_loss(point_logits, point_labels, num_boxes),
        }
        return losses

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
        group_detr = self.group_detr if self.training else 1
        outputs_without_aux_and_enc = {
            k: v for k, v in outputs.items() if k != "enc_outputs" and k != "auxiliary_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux_and_enc, targets, group_detr)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = num_boxes * group_detr
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        world_size = 1
        if is_accelerate_available():
            if PartialState._shared_state != {}:
                num_boxes = reduce(num_boxes)
                world_size = PartialState().num_processes
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets, group_detr)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            indices = self.matcher(enc_outputs, targets, group_detr=group_detr)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


def _set_aux_loss(outputs_class, outputs_coord, outputs_masks):
    return [
        {"logits": a, "pred_boxes": b, "pred_masks": c}
        for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_masks[:-1])
    ]


def RfDetrForSegmentationLoss(
    logits,
    labels,
    device,
    pred_boxes,
    pred_masks,
    config,
    outputs_class=None,
    outputs_coord=None,
    outputs_masks=None,
    enc_outputs_class=None,
    enc_outputs_coord=None,
    enc_outputs_masks=None,
    **kwargs,
):
    # First: create the matcher
    matcher = RfDetrHungarianMatcher(
        class_cost=config.class_cost,
        bbox_cost=config.bbox_cost,
        giou_cost=config.giou_cost,
        mask_point_sample_ratio=config.mask_point_sample_ratio,
        cost_mask_class_cost=config.mask_class_loss_coefficient,
        cost_mask_dice_cost=config.mask_dice_loss_coefficient,
    )
    # Second: create the criterion
    losses = ["labels", "boxes", "cardinality", "masks"]
    criterion = RfDetrImageLoss(
        matcher=matcher,
        num_classes=config.num_labels,
        focal_alpha=config.focal_alpha,
        losses=losses,
        group_detr=config.group_detr,
        mask_point_sample_ratio=config.mask_point_sample_ratio,
    )
    criterion.to(device)
    # Third: compute the losses, based on outputs and labels
    outputs_loss = {}
    auxiliary_outputs = None
    outputs_loss["logits"] = logits
    outputs_loss["pred_boxes"] = pred_boxes
    outputs_loss["pred_masks"] = pred_masks
    outputs_loss["enc_outputs"] = {
        "logits": enc_outputs_class,
        "pred_boxes": enc_outputs_coord,
        "pred_masks": enc_outputs_masks,
    }
    if config.auxiliary_loss:
        auxiliary_outputs = _set_aux_loss(outputs_class, outputs_coord, outputs_masks)
        outputs_loss["auxiliary_outputs"] = auxiliary_outputs
    loss_dict = criterion(outputs_loss, labels)
    # Fourth: compute total loss, as a weighted sum of the various losses
    weight_dict = {"loss_ce": config.class_loss_coefficient, "loss_bbox": config.bbox_loss_coefficient}
    weight_dict["loss_giou"] = config.giou_loss_coefficient
    weight_dict["loss_mask_ce"] = config.mask_class_loss_coefficient
    weight_dict["loss_mask_dice"] = config.mask_dice_loss_coefficient
    if config.auxiliary_loss:
        aux_weight_dict = {}
        for i in range(config.decoder_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
    return loss, loss_dict, auxiliary_outputs
