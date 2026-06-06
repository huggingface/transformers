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


def OwlViTForObjectDetectionLoss(logits, labels, device, pred_boxes, config, **kwargs):
    """
    Compute the open-vocabulary object detection loss for OWL-ViT / OWLv2.

    Both models predict, for every image patch, one classification logit per text query (i.e. per candidate class),
    together with a box. This is structurally identical to a DeformableDETR-style detector with sigmoid classification,
    so we reuse its bipartite matcher and image loss (sigmoid focal classification + L1 + generalized IoU). Unlike the
    closed-vocabulary detectors, the number of classes is not fixed by the config but given at runtime by the number of
    text queries, so it is read from the last dimension of `logits`.
    """
    # First: create the matcher
    matcher = DeformableDetrHungarianMatcher(
        class_cost=config.class_cost, bbox_cost=config.bbox_cost, giou_cost=config.giou_cost
    )
    # Second: create the criterion. The number of classes equals the number of text queries (last logits dimension).
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
    loss_dict = criterion(outputs_loss, labels)
    # Fourth: compute total loss, as a weighted sum of the various losses
    weight_dict = {
        "loss_ce": config.class_loss_coefficient,
        "loss_bbox": config.bbox_loss_coefficient,
        "loss_giou": config.giou_loss_coefficient,
    }
    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
    # OWL-ViT has no decoder stack, hence no auxiliary outputs.
    return loss, loss_dict, None
