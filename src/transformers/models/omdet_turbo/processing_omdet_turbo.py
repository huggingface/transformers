# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
Processor class for OmDet-Turbo.
"""

from typing import List, Tuple, Union


try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

from ...image_transforms import center_to_corners_format
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput
from ...utils import (
    TensorType,
    is_torch_available,
    is_torchvision_available,
)


if is_torch_available():
    import torch

if is_torchvision_available():
    from torchvision.ops.boxes import batched_nms


def clip_boxes(box: torch.Tensor, box_size: Tuple[int, int]) -> torch.Tensor:
    """
    Clip the boxes by limiting x coordinates to the range [0, width]
    and y coordinates to the range [0, height].

    Args:
        box (Tensor): The box to be clipped.
        box_size (height, width): The clipping box's size.
    """
    assert torch.isfinite(box).all(), "Box tensor contains infinite or NaN!"
    h, w = box_size
    x1 = box[:, 0].clamp(min=0, max=h)
    y1 = box[:, 1].clamp(min=0, max=w)
    x2 = box[:, 2].clamp(min=0, max=h)
    y2 = box[:, 3].clamp(min=0, max=w)
    box = torch.stack((x1, y1, x2, y2), dim=-1)

    return box


def handle_text(
    text: Union[str, List[str], List[List[str]], TextInput, PreTokenizedInput],
) -> Tuple[List[str], List[List[str]]]:
    if isinstance(text, str):
        # Text needs to be in this format: "Detect cat, dog, bird"
        tasks = [text]
        labels = "".join(text.split(" ")[1:])
        if len(labels) == 0:
            labels = [[text]]
        else:
            labels = labels.split(",")
    elif isinstance(text, list):
        if isinstance(text[0], str) and len(text) == 2 and isinstance(text[1], list):
            tasks = [text[0]]
            labels = [text[1]]
        elif isinstance(text[0], str) and len(text) > 2:
            tasks = ["Detect {}.".format(",".join(text))]
            labels = text
        elif isinstance(text[0], list) and len(text) == 2 and isinstance(text[1][0], list):
            tasks = text[0]
            labels = text[1]
        else:
            raise ValueError("Invalid input format for `text`.")
    else:
        raise ValueError("Invalid input format for `text`.")

    return tasks, labels


class OmDetTurboProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": "max_length",
            "truncation": True,
            "max_length": 77,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_token_type_ids": False,
            "return_length": False,
            "verbose": True,
        },
        "images_kwargs": {},
    }


class OmDetTurboProcessor(ProcessorMixin):
    r"""
    Constructs a OmDet-Turbo processor which wraps a Deformable DETR image processor and an Auto tokenizer into a
    single processor.

    [`OmDetTurboProcessor`] offers all the functionalities of [`DetrImageProcessor`] and
    [`AutoTokenizer`]. See the docstring of [`~OmDetTurboProcessor.__call__`] and [`~OmDetTurboProcessor.decode`]
    for more information.

    Args:
        image_processor (`DetrImageProcessor`):
            An instance of [`DetrImageProcessor`]. The image processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "DetrImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[str, List[str], List[List[str]], TextInput, PreTokenizedInput] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[OmDetTurboProcessorKwargs],
    ) -> BatchEncoding:
        """
        This method uses [`DetrImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`CLIPTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """
        if images is None or text is None:
            raise ValueError("You have to specify `images` and `text`.")

        tasks, labels = handle_text(text)

        output_kwargs = self._merge_kwargs(
            OmDetTurboProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        encoding_image_processor = self.image_processor(images, **output_kwargs["images_kwargs"])
        tasks_encoding = self.tokenizer(text=tasks, **output_kwargs["text_kwargs"])

        labels_encoding = []
        for label in labels:
            label_encoding = self.tokenizer(text=label, **output_kwargs["text_kwargs"])
            labels_encoding.append(label_encoding)
        # workaround to group the labels encoding by task in a BatchEncoding
        labels_encoding = BatchEncoding({str(i): label_encoding for i, label_encoding in enumerate(labels_encoding)})

        encoding = BatchEncoding({"tasks": tasks_encoding, "labels": labels_encoding})
        encoding.update(encoding_image_processor)

        return encoding

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.batch_decode with BertTokenizerFast->PreTrainedTokenizer
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.decode with BertTokenizerFast->PreTrainedTokenizer
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def process_single_image(
        self,
        boxes,
        scores,
        labels,
        label_names,
        image_size: Tuple[int, int],
        num_classes: int,
        score_thresh: float,
        nms_thresh: float,
        max_num_det: int = None,
    ):
        """
        Filter predicted results using given thresholds and NMS.
        Args:
            boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
                boxes for each image. Element i has shape (Ri, K * 4) if doing
                class-specific regression, or (Ri, 4) if doing class-agnostic
                regression, where Ri is the number of predicted objects for image i.
                This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
            scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
            labels (list[Tensor]): A list of Tensors of predicted class labels for each image.
                Element i has shape (Ri,), where Ri is the number of predicted objects for image i.
            labels_name (list[str]): A list of class labels for each class.
            image_size (list[tuple]): A list of (width, height) tuples for each image in the batch.
            num_classes (int): The number of classes given for this image.
            score_thresh (float): Only return detections with a confidence score exceeding this
                threshold.
            nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
            max_num_det (int, optional): The maximum number of detections to return. Default is None.
        Returns:
            list[dict]: A list of dictionaries, each containing the following keys:
                "pred_boxes" (Tensor): A tensor of shape (N, 4), containing the predicted boxes in (x1, y1, x2, y2) format.
                "scores" (Tensor): A tensor of shape (N,), containing the predicted confidence scores for each detection.
                "pred_classes" (list[str]): A list of strings, where each string is the predicted label for the
                    corresponding detection
        """
        proposal_num = len(boxes) if max_num_det is None else max_num_det
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(proposal_num, sorted=False)
        labels_per_image = labels[topk_indices]
        box_pred_per_image = boxes.view(-1, 1, 4).repeat(1, num_classes, 1).view(-1, 4)
        box_pred_per_image = box_pred_per_image[topk_indices]

        # Score filtering
        box_pred_per_image = center_to_corners_format(box_pred_per_image)
        box_pred_per_image = box_pred_per_image * torch.tensor(image_size).repeat(2).to(box_pred_per_image.device)
        filter_mask = scores_per_image > score_thresh  # R x K
        score_keep = filter_mask.nonzero(as_tuple=False).view(-1)
        box_pred_per_image = box_pred_per_image[score_keep]
        scores_per_image = scores_per_image[score_keep]
        labels_per_image = labels_per_image[score_keep]

        filter_labels_mask = labels_per_image < len(label_names)
        labels_keep = filter_labels_mask.nonzero(as_tuple=False).view(-1)
        box_pred_per_image = box_pred_per_image[labels_keep]
        scores_per_image = scores_per_image[labels_keep]
        labels_per_image = labels_per_image[labels_keep]

        # NMS
        keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, nms_thresh)
        box_pred_per_image = box_pred_per_image[keep]
        scores_per_image = scores_per_image[keep]
        labels_per_image = labels_per_image[keep]
        labels_per_image = [label_names[i] for i in labels_per_image]

        # create an instance
        result = {}
        result["pred_boxes"] = clip_boxes(box_pred_per_image, image_size)
        result["scores"] = scores_per_image
        result["pred_classes"] = labels_per_image

        return result

    def compute_score(self, boxes):
        # TODO modify for training
        num_classes = boxes.shape[2]
        proposal_num = boxes.shape[1]
        scores = torch.sigmoid(boxes)
        labels = torch.arange(num_classes, device=boxes.device).unsqueeze(0).repeat(proposal_num, 1).flatten(0, 1)
        return scores, labels

    def post_process_grounded_object_detection(
        self,
        outputs,
        labels_names: List[str],
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        target_sizes: Union[TensorType, List[Tuple]] = None,
        max_num_det=None,
    ):
        """
        Converts the raw output of [`OmDetTurboForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format and get the associated text label.

        Args:
            outputs ([`OmDetTurboObjectDetectionOutput`]):
                Raw outputs of the model.
            labels_name (list[str]): A list of class labels for each class.
            score_thresh (float): Only return detections with a confidence score exceeding this
                threshold.
            nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        boxes = outputs.decoder_bboxes
        logits = outputs.decoder_cls
        scores, labels = self.compute_score(logits)
        num_classes = logits.shape[2]
        results = []
        print(labels_names)
        for scores_img, box_per_img, image_size, label_names in zip(scores, boxes, target_sizes, labels_names):
            results.append(
                self.process_single_image(
                    box_per_img,
                    scores_img,
                    labels,
                    label_names,
                    image_size,
                    num_classes,
                    score_thresh=score_threshold,
                    nms_thresh=nms_threshold,
                    max_num_det=max_num_det,
                )
            )

        return results
