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

from typing import List, Optional, Tuple, Union

from torchvision.ops.boxes import batched_nms

from ...image_transforms import center_to_corners_format
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType, is_torch_available


if is_torch_available():
    import torch


class OmDetTurboProcessor(ProcessorMixin):
    r"""
    Constructs a OmDet-Turbo processor which wraps a Deformable DETR image processor and a BERT tokenizer into a
    single processor.

    [`OmDetTurboProcessor`] offers all the functionalities of [`OmDetTurboImageProcessor`] and
    [`AutoTokenizer`]. See the docstring of [`~OmDetTurboProcessor.__call__`] and [`~OmDetTurboProcessor.decode`]
    for more information.

    Args:
        image_processor (`OmDetTurboImageProcessor`):
            An instance of [`OmDetTurboImageProcessor`]. The image processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "OmDetTurboImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: ImageInput = None,
        tasks: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        labels: Union[
            List[TextInput], List[PreTokenizedInput], List[List[TextInput]], List[List[PreTokenizedInput]]
        ] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = "max_length",
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = 77,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = True,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:
        """
        This method uses [`OmDetTurboImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`OmDetTurboTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """
        encoding_image_processor = self.image_processor(images, return_tensors=return_tensors)

        tasks_encoding = self.tokenizer(
            text=tasks,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_token_type_ids=return_token_type_ids,
            return_length=return_length,
            verbose=verbose,
            return_tensors=return_tensors,
            **kwargs,
        )

        if type(labels[0]) not in [list, tuple]:
            labels = [labels]
        labels_encoding = []
        for label in labels:
            label_encoding = self.tokenizer(
                text=label,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
            labels_encoding.append(label_encoding)
        # workaround to group the labels encoding by task in a BatchEncoding
        labels_encoding = BatchEncoding({str(i): labels_encoding[i] for i in range(len(labels_encoding))})

        encoding = BatchEncoding({"tasks": tasks_encoding, "labels": labels_encoding})
        encoding.update(encoding_image_processor)

        return encoding

    def clip(self, box, box_size: Tuple[int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
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

    def inference_single_image(
        self,
        boxes,
        scores,
        labels,
        labels_name,
        image_size: Tuple[int, int],
        num_classes: int,
        score_thresh: float,
        nms_thresh: float,
        max_num_det: int = None,
    ):
        """
        Call `fast_rcnn_inference_single_image` for all images.
        Args:
            boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
                boxes for each image. Element i has shape (Ri, K * 4) if doing
                class-specific regression, or (Ri, 4) if doing class-agnostic
                regression, where Ri is the number of predicted objects for image i.
                This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
            scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
            image_size (list[tuple]): A list of (width, height) tuples for each image in the batch.
            score_thresh (float): Only return detections with a confidence score exceeding this
                threshold.
            nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        Returns:
            instances: (list[Instances]): A list of N instances, one for each image in the batch,
                that stores the topk most confidence detections.
            kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
                the corresponding boxes/scores index in [0, Ri) from the input, for image i.
        """
        # scores_per_image: num_proposal
        # labels_per_image: num_proposal
        # box_per_images: num_proposal x 4'
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

        # NMS
        keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, nms_thresh)
        box_pred_per_image = box_pred_per_image[keep]
        scores_per_image = scores_per_image[keep]
        labels_per_image = labels_per_image[keep]
        labels_per_image = [labels_name[i] for i in labels_per_image]

        # create an instance
        result = {}
        result["pred_boxes"] = self.clip(box_pred_per_image, image_size)
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
        labels_name: List[str],
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
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The token ids of the input text.
            box_threshold (`float`, *optional*, defaults to 0.25):
                Score threshold to keep object detection predictions.
            text_threshold (`float`, *optional*, defaults to 0.25):
                Score threshold to keep text detection predictions.
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
        for i, (scores_img, box_per_img, image_size) in enumerate(zip(scores, boxes, target_sizes)):
            results.append(
                self.inference_single_image(
                    box_per_img,
                    scores_img,
                    labels,
                    labels_name,
                    image_size,
                    num_classes,
                    score_thresh=score_threshold,
                    nms_thresh=nms_threshold,
                    max_num_det=max_num_det,
                )
            )

        return results
