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
from .loss_deformable_detr import DeformableDetrHungarianMatcher, DeformableDetrImageLoss
from .loss_for_object_detection import _set_aux_loss


def OmDetTurboForObjectDetectionLoss(
    logits,
    labels,
    device,
    pred_boxes,
    config,
    outputs_class=None,
    outputs_coord=None,
    **kwargs,
):
    """
    Compute the open-vocabulary object detection loss for OmDet-Turbo.

    OmDet-Turbo is a DETR-style (RT-DETR-like) detector that predicts, for every decoder query, a box and one
    classification logit per candidate class. We therefore reuse the DeformableDETR-style bipartite matcher and image
    loss (sigmoid focal classification + L1 + generalized IoU). As the model is open-vocabulary, the number of classes
    is not fixed by the config but given at runtime by the last dimension of `logits`. When `config.auxiliary_loss` is
    enabled, a loss is also computed at every intermediate decoder layer (`outputs_class`/`outputs_coord` hold the
    per-layer predictions).
    """
    # First: create the matcher
    matcher = DeformableDetrHungarianMatcher(
        class_cost=config.class_cost, bbox_cost=config.bbox_cost, giou_cost=config.giou_cost
    )
    # Second: create the criterion. The number of classes equals the number of candidate classes (last logits dim).
    num_classes = logits.shape[-1]
    losses = ["labels", "boxes", "cardinality"]
    criterion = DeformableDetrImageLoss(
        matcher=matcher,
        num_classes=num_classes,
        focal_alpha=config.focal_alpha,
        losses=losses,
    )
    criterion.to(device)
    # Third: compute the losses, based on outputs and labels
    outputs_loss = {"logits": logits, "pred_boxes": pred_boxes}
    auxiliary_outputs = None
    if config.auxiliary_loss and outputs_class is not None and outputs_coord is not None:
        # `outputs_class`/`outputs_coord` are stacked over decoder layers; `_set_aux_loss` keeps all but the last one.
        auxiliary_outputs = _set_aux_loss(outputs_class, outputs_coord)
        outputs_loss["auxiliary_outputs"] = auxiliary_outputs

    loss_dict = criterion(outputs_loss, labels)
    # Fourth: compute total loss, as a weighted sum of the various losses
    weight_dict = {
        "loss_ce": config.class_loss_coefficient,
        "loss_bbox": config.bbox_loss_coefficient,
        "loss_giou": config.giou_loss_coefficient,
    }
    if auxiliary_outputs is not None:
        aux_weight_dict = {}
        for i in range(len(auxiliary_outputs)):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
    return loss, loss_dict, auxiliary_outputs
