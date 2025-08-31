from typing import TYPE_CHECKING, Optional, Union

from transformers.models.detr.image_processing_detr_fast import DetrImageProcessorFast

from ...image_transforms import center_to_corners_format
from ...utils import (
    TensorType,
    is_torch_available,
    logging,
)


if TYPE_CHECKING:
    from .modeling_grounding_dino import GroundingDinoObjectDetectionOutput

if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


def _scale_boxes(boxes, target_sizes):
    """
    Scale batch of bounding boxes to the target sizes.

    Args:
        boxes (`torch.Tensor` of shape `(batch_size, num_boxes, 4)`):
            Bounding boxes to scale. Each box is expected to be in (x1, y1, x2, y2) format.
        target_sizes (`list[tuple[int, int]]` or `torch.Tensor` of shape `(batch_size, 2)`):
            Target sizes to scale the boxes to. Each target size is expected to be in (height, width) format.

    Returns:
        `torch.Tensor` of shape `(batch_size, num_boxes, 4)`: Scaled bounding boxes.
    """

    if isinstance(target_sizes, (list, tuple)):
        image_height = torch.tensor([i[0] for i in target_sizes])
        image_width = torch.tensor([i[1] for i in target_sizes])
    elif isinstance(target_sizes, torch.Tensor):
        image_height, image_width = target_sizes.unbind(1)
    else:
        raise TypeError("`target_sizes` must be a list, tuple or torch.Tensor")

    scale_factor = torch.stack([image_width, image_height, image_width, image_height], dim=1)
    scale_factor = scale_factor.unsqueeze(1).to(boxes.device)
    boxes = boxes * scale_factor
    return boxes


class GroundingDinoImageProcessorFast(DetrImageProcessorFast):
    def post_process_object_detection(
        self,
        outputs: "GroundingDinoObjectDetectionOutput",
        threshold: float = 0.1,
        target_sizes: Optional[Union[TensorType, list[tuple]]] = None,
    ):
        """
        Converts the raw output of [`GroundingDinoForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format.

        Args:
            outputs ([`GroundingDinoObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*, defaults to 0.1):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `list[tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.

        Returns:
            `list[Dict]`: A list of dictionaries, each dictionary containing the following keys:
            - "scores": The confidence scores for each predicted box on the image.
            - "labels": Indexes of the classes predicted by the model on the image.
            - "boxes": Image bounding boxes in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format.
        """
        batch_logits, batch_boxes = outputs.logits, outputs.pred_boxes
        batch_size = len(batch_logits)

        if target_sizes is not None and len(target_sizes) != batch_size:
            raise ValueError("Make sure that you pass in as many target sizes as images")

        # batch_logits of shape (batch_size, num_queries, num_classes)
        batch_class_logits = torch.max(batch_logits, dim=-1)
        batch_scores = torch.sigmoid(batch_class_logits.values)
        batch_labels = batch_class_logits.indices

        # Convert to [x0, y0, x1, y1] format
        batch_boxes = center_to_corners_format(batch_boxes)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            batch_boxes = _scale_boxes(batch_boxes, target_sizes)

        results = []
        for scores, labels, boxes in zip(batch_scores, batch_labels, batch_boxes):
            keep = scores > threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]
            results.append({"scores": scores, "labels": labels, "boxes": boxes})

        return results

    def post_process():
        raise NotImplementedError("Post-processing is not implemented for Grounding-Dino yet.")

    def post_process_segmentation():
        raise NotImplementedError("Segmentation post-processing is not implemented for Grounding-Dino yet.")

    def post_process_instance():
        raise NotImplementedError("Instance post-processing is not implemented for Grounding-Dino yet.")

    def post_process_panoptic():
        raise NotImplementedError("Panoptic post-processing is not implemented for Grounding-Dino yet.")

    def post_process_instance_segmentation():
        raise NotImplementedError("Segmentation post-processing is not implemented for Grounding-Dino yet.")

    def post_process_semantic_segmentation():
        raise NotImplementedError("Semantic segmentation post-processing is not implemented for Grounding-Dino yet.")

    def post_process_panoptic_segmentation():
        raise NotImplementedError("Panoptic segmentation post-processing is not implemented for Grounding-Dino yet.")


__all__ = ["GroundingDinoImageProcessorFast"]
