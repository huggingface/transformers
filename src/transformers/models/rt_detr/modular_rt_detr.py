# Copyright 2024 Baidu Inc and The HuggingFace Inc. team.
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
import math
import pathlib
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tvF
from torch import nn

from ... import initialization as init
from ...activations import ACT2CLS, ACT2FN
from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast, SizeDict, get_max_height_width
from ...image_transforms import center_to_corners_format, corners_to_center_format
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    AnnotationFormat,
    AnnotationType,
    ChannelDimension,
    PILImageResampling,
    get_image_size,
    validate_annotations,
)
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import compile_compatible_method_lru_cache
from ...utils import (
    ModelOutput,
    TensorType,
    TransformersKwargs,
    auto_docstring,
    logging,
    requires_backends,
    torch_int,
)
from ...utils.backbone_utils import load_backbone
from ...utils.generic import can_return_tuple, check_model_inputs
from ..conditional_detr.modeling_conditional_detr import inverse_sigmoid
from ..deformable_detr.modeling_deformable_detr import DeformableDetrMultiscaleDeformableAttention
from ..detr.image_processing_detr_fast import DetrImageProcessorFast
from ..detr.modeling_detr import DetrFrozenBatchNorm2d, DetrMLPPredictionHead, DetrSelfAttention, replace_batch_norm
from .configuration_rt_detr import RTDetrConfig
from .image_processing_rt_detr import RTDetrImageProcessorKwargs


logger = logging.get_logger(__name__)

SUPPORTED_ANNOTATION_FORMATS = (AnnotationFormat.COCO_DETECTION,)


def prepare_coco_detection_annotation(
    image,
    target,
    return_segmentation_masks: bool = False,
    input_data_format: ChannelDimension | str | None = None,
):
    """
    Convert the target in COCO format into the format expected by RT-DETR.
    """
    image_height, image_width = image.size()[-2:]

    image_id = target["image_id"]
    image_id = torch.as_tensor([image_id], dtype=torch.int64, device=image.device)

    # Get all COCO annotations for the given image.
    annotations = target["annotations"]
    classes = []
    area = []
    boxes = []
    keypoints = []
    for obj in annotations:
        if "iscrowd" not in obj or obj["iscrowd"] == 0:
            classes.append(obj["category_id"])
            area.append(obj["area"])
            boxes.append(obj["bbox"])
            if "keypoints" in obj:
                keypoints.append(obj["keypoints"])

    classes = torch.as_tensor(classes, dtype=torch.int64, device=image.device)
    area = torch.as_tensor(area, dtype=torch.float32, device=image.device)
    iscrowd = torch.zeros_like(classes, dtype=torch.int64, device=image.device)
    # guard against no boxes via resizing
    boxes = torch.as_tensor(boxes, dtype=torch.float32, device=image.device).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=image_width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=image_height)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

    new_target = {
        "image_id": image_id,
        "class_labels": classes[keep],
        "boxes": boxes[keep],
        "area": area[keep],
        "iscrowd": iscrowd[keep],
        "orig_size": torch.as_tensor([int(image_height), int(image_width)], dtype=torch.int64, device=image.device),
    }

    if keypoints:
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=image.device)
        # Apply the keep mask here to filter the relevant annotations
        keypoints = keypoints[keep]
        num_keypoints = keypoints.shape[0]
        keypoints = keypoints.reshape((-1, 3)) if num_keypoints else keypoints
        new_target["keypoints"] = keypoints

    return new_target


class RTDetrImageProcessorFast(DetrImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    format = AnnotationFormat.COCO_DETECTION
    do_convert_annotations = True
    do_resize = True
    do_rescale = True
    do_normalize = False
    do_pad = False
    size = {"height": 640, "width": 640}
    default_to_square = False
    model_input_names = ["pixel_values", "pixel_mask"]
    valid_kwargs = RTDetrImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[RTDetrImageProcessorKwargs]) -> None:
        # Backwards compatibility
        do_convert_annotations = kwargs.get("do_convert_annotations")
        do_normalize = kwargs.get("do_normalize")
        if do_convert_annotations is None and getattr(self, "do_convert_annotations", None) is None:
            self.do_convert_annotations = do_normalize if do_normalize is not None else self.do_normalize

        BaseImageProcessorFast.__init__(self, **kwargs)

    def prepare_annotation(
        self,
        image: torch.Tensor,
        target: dict,
        format: AnnotationFormat | None = None,
        return_segmentation_masks: bool | None = None,
        masks_path: str | pathlib.Path | None = None,
        input_data_format: str | ChannelDimension | None = None,
    ) -> dict:
        format = format if format is not None else self.format

        if format == AnnotationFormat.COCO_DETECTION:
            return_segmentation_masks = False if return_segmentation_masks is None else return_segmentation_masks
            target = prepare_coco_detection_annotation(
                image, target, return_segmentation_masks, input_data_format=input_data_format
            )
        else:
            raise ValueError(f"Format {format} is not supported.")
        return target

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        annotations: AnnotationType | list[AnnotationType] | None,
        masks_path: str | pathlib.Path | None,
        return_segmentation_masks: bool,
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["tvF.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        do_convert_annotations: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool,
        pad_size: SizeDict | None,
        format: str | AnnotationFormat | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or a batch of images so that it can be used by the model.
        """

        if annotations is not None and isinstance(annotations, dict):
            annotations = [annotations]

        if annotations is not None and len(images) != len(annotations):
            raise ValueError(
                f"The number of images ({len(images)}) and annotations ({len(annotations)}) do not match."
            )

        format = AnnotationFormat(format)
        if annotations is not None:
            validate_annotations(format, SUPPORTED_ANNOTATION_FORMATS, annotations)

        data = {}
        processed_images = []
        processed_annotations = []
        pixel_masks = []  # Initialize pixel_masks here
        for image, annotation in zip(images, annotations if annotations is not None else [None] * len(images)):
            # prepare (COCO annotations as a list of Dict -> DETR target as a single Dict per image)
            if annotations is not None:
                annotation = self.prepare_annotation(
                    image,
                    annotation,
                    format,
                    return_segmentation_masks=return_segmentation_masks,
                    masks_path=masks_path,
                    input_data_format=ChannelDimension.FIRST,
                )

            if do_resize:
                resized_image = self.resize(image, size=size, interpolation=interpolation)
                if annotations is not None:
                    annotation = self.resize_annotation(
                        annotation,
                        orig_size=image.size()[-2:],
                        target_size=resized_image.size()[-2:],
                    )
                image = resized_image
            # Fused rescale and normalize
            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)
            if do_convert_annotations and annotations is not None:
                annotation = self.normalize_annotation(annotation, get_image_size(image, ChannelDimension.FIRST))

            processed_images.append(image)
            processed_annotations.append(annotation)
        images = processed_images
        annotations = processed_annotations if annotations is not None else None

        if do_pad:
            # depends on all resized image shapes so we need another loop
            if pad_size is not None:
                padded_size = (pad_size.height, pad_size.width)
            else:
                padded_size = get_max_height_width(images)

            padded_images = []
            padded_annotations = []
            for image, annotation in zip(images, annotations if annotations is not None else [None] * len(images)):
                # Pads images and returns their mask: {'pixel_values': ..., 'pixel_mask': ...}
                if padded_size == image.size()[-2:]:
                    padded_images.append(image)
                    pixel_masks.append(torch.ones(padded_size, dtype=torch.int64, device=image.device))
                    padded_annotations.append(annotation)
                    continue
                image, pixel_mask, annotation = self.pad(
                    image, padded_size, annotation=annotation, update_bboxes=do_convert_annotations
                )
                padded_images.append(image)
                padded_annotations.append(annotation)
                pixel_masks.append(pixel_mask)
            images = padded_images
            annotations = padded_annotations if annotations is not None else None
            data.update({"pixel_mask": torch.stack(pixel_masks, dim=0)})

        data.update({"pixel_values": torch.stack(images, dim=0)})
        encoded_inputs = BatchFeature(data, tensor_type=return_tensors)
        if annotations is not None:
            encoded_inputs["labels"] = [
                BatchFeature(annotation, tensor_type=return_tensors) for annotation in annotations
            ]
        return encoded_inputs

    def post_process_object_detection(
        self,
        outputs,
        threshold: float = 0.5,
        target_sizes: TensorType | list[tuple] = None,
        use_focal_loss: bool = True,
    ):
        """
        Converts the raw output of [`DetrForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`DetrObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*, defaults to 0.5):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `list[tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
            use_focal_loss (`bool` defaults to `True`):
                Variable informing if the focal loss was used to predict the outputs. If `True`, a sigmoid is applied
                to compute the scores of each detection, otherwise, a softmax function is used.

        Returns:
            `list[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        requires_backends(self, ["torch"])
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes
        # convert from relative cxcywh to absolute xyxy
        boxes = center_to_corners_format(out_bbox)
        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )
            if isinstance(target_sizes, list):
                img_h, img_w = torch.as_tensor(target_sizes).unbind(1)
            else:
                img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        num_top_queries = out_logits.shape[1]
        num_classes = out_logits.shape[2]

        if use_focal_loss:
            scores = torch.nn.functional.sigmoid(out_logits)
            scores, index = torch.topk(scores.flatten(1), num_top_queries, axis=-1)
            labels = index % num_classes
            index = index // num_classes
            boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
        else:
            scores = torch.nn.functional.softmax(out_logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > num_top_queries:
                scores, index = torch.topk(scores, num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        results = []
        for score, label, box in zip(scores, labels, boxes):
            results.append(
                {
                    "scores": score[score > threshold],
                    "labels": label[score > threshold],
                    "boxes": box[score > threshold],
                }
            )

        return results

    def post_process_instance_segmentation(self):
        raise NotImplementedError("Segmentation post-processing is not implemented for RT-DETR yet.")

    def post_process_semantic_segmentation(self):
        raise NotImplementedError("Semantic segmentation post-processing is not implemented for RT-DETR yet.")

    def post_process_panoptic_segmentation(self):
        raise NotImplementedError("Panoptic segmentation post-processing is not implemented for RT-DETR yet.")


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for outputs of the RTDetrDecoder. This class adds two attributes to
    BaseModelOutputWithCrossAttentions, namely:
    - a stacked tensor of intermediate decoder hidden states (i.e. the output of each decoder layer)
    - a stacked tensor of intermediate reference points.
    """
)
class RTDetrDecoderOutput(ModelOutput):
    r"""
    intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
        Stacked intermediate hidden states (output of each layer of the decoder).
    intermediate_logits (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, config.num_labels)`):
        Stacked intermediate logits (logits of each layer of the decoder).
    intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, hidden_size)`):
        Stacked intermediate reference points (reference points of each layer of the decoder).
    intermediate_predicted_corners (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked intermediate predicted corners (predicted corners of each layer of the decoder).
    initial_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked initial reference points (initial reference points of each layer of the decoder).
    cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
        sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
        used to compute the weighted average in the cross-attention heads.
    """

    last_hidden_state: torch.FloatTensor | None = None
    intermediate_hidden_states: torch.FloatTensor | None = None
    intermediate_logits: torch.FloatTensor | None = None
    intermediate_reference_points: torch.FloatTensor | None = None
    intermediate_predicted_corners: torch.FloatTensor | None = None
    initial_reference_points: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    cross_attentions: tuple[torch.FloatTensor] | None = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for outputs of the RT-DETR encoder-decoder model.
    """
)
class RTDetrModelOutput(ModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
        Sequence of hidden-states at the output of the last layer of the decoder of the model.
    intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
        Stacked intermediate hidden states (output of each layer of the decoder).
    intermediate_logits (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, config.num_labels)`):
        Stacked intermediate logits (logits of each layer of the decoder).
    intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked intermediate reference points (reference points of each layer of the decoder).
    intermediate_predicted_corners (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked intermediate predicted corners (predicted corners of each layer of the decoder).
    initial_reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
        Initial reference points used for the first decoder layer.
    init_reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
        Initial reference points sent through the Transformer decoder.
    enc_topk_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
        Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
        picked as region proposals in the encoder stage. Output of bounding box binary classification (i.e.
        foreground and background).
    enc_topk_bboxes (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`):
        Logits of predicted bounding boxes coordinates in the encoder stage.
    enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
        picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
        foreground and background).
    enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Logits of predicted bounding boxes coordinates in the first stage.
    denoising_meta_values (`dict`):
        Extra dictionary for the denoising related values.
    """

    last_hidden_state: torch.FloatTensor | None = None
    intermediate_hidden_states: torch.FloatTensor | None = None
    intermediate_logits: torch.FloatTensor | None = None
    intermediate_reference_points: torch.FloatTensor | None = None
    intermediate_predicted_corners: torch.FloatTensor | None = None
    initial_reference_points: torch.FloatTensor | None = None
    decoder_hidden_states: tuple[torch.FloatTensor] | None = None
    decoder_attentions: tuple[torch.FloatTensor] | None = None
    cross_attentions: tuple[torch.FloatTensor] | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    encoder_hidden_states: tuple[torch.FloatTensor] | None = None
    encoder_attentions: tuple[torch.FloatTensor] | None = None
    init_reference_points: torch.FloatTensor | None = None
    enc_topk_logits: torch.FloatTensor | None = None
    enc_topk_bboxes: torch.FloatTensor | None = None
    enc_outputs_class: torch.FloatTensor | None = None
    enc_outputs_coord_logits: torch.FloatTensor | None = None
    denoising_meta_values: dict | None = None


@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`RTDetrForObjectDetection`].
    """
)
class RTDetrObjectDetectionOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
        Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
        bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
        scale-invariant IoU loss.
    loss_dict (`Dict`, *optional*):
        A dictionary containing the individual losses. Useful for logging.
    logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
        Classification logits (including no-object) for all queries.
    pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
        Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
        values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
        possible padding). You can use [`~RTDetrImageProcessor.post_process_object_detection`] to retrieve the
        unnormalized (absolute) bounding boxes.
    auxiliary_outputs (`list[Dict]`, *optional*):
        Optional, only returned when auxiliary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
        and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
        `pred_boxes`) for each decoder layer.
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
        Sequence of hidden-states at the output of the last layer of the decoder of the model.
    intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
        Stacked intermediate hidden states (output of each layer of the decoder).
    intermediate_logits (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, config.num_labels)`):
        Stacked intermediate logits (logits of each layer of the decoder).
    intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked intermediate reference points (reference points of each layer of the decoder).
    intermediate_predicted_corners (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked intermediate predicted corners (predicted corners of each layer of the decoder).
    initial_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked initial reference points (initial reference points of each layer of the decoder).
    init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
        Initial reference points sent through the Transformer decoder.
    enc_topk_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Logits of predicted bounding boxes coordinates in the encoder.
    enc_topk_bboxes (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Logits of predicted bounding boxes coordinates in the encoder.
    enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
        picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
        foreground and background).
    enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Logits of predicted bounding boxes coordinates in the first stage.
    denoising_meta_values (`dict`):
        Extra dictionary for the denoising related values
    """

    loss: torch.FloatTensor | None = None
    loss_dict: dict | None = None
    logits: torch.FloatTensor | None = None
    pred_boxes: torch.FloatTensor | None = None
    auxiliary_outputs: list[dict] | None = None
    last_hidden_state: torch.FloatTensor | None = None
    intermediate_hidden_states: torch.FloatTensor | None = None
    intermediate_logits: torch.FloatTensor | None = None
    intermediate_reference_points: torch.FloatTensor | None = None
    intermediate_predicted_corners: torch.FloatTensor | None = None
    initial_reference_points: torch.FloatTensor | None = None
    decoder_hidden_states: tuple[torch.FloatTensor] | None = None
    decoder_attentions: tuple[torch.FloatTensor] | None = None
    cross_attentions: tuple[torch.FloatTensor] | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    encoder_hidden_states: tuple[torch.FloatTensor] | None = None
    encoder_attentions: tuple[torch.FloatTensor] | None = None
    init_reference_points: tuple[torch.FloatTensor] | None = None
    enc_topk_logits: torch.FloatTensor | None = None
    enc_topk_bboxes: torch.FloatTensor | None = None
    enc_outputs_class: torch.FloatTensor | None = None
    enc_outputs_coord_logits: torch.FloatTensor | None = None
    denoising_meta_values: dict | None = None


class RTDetrMLP(nn.Module):
    def __init__(self, config: RTDetrConfig, hidden_size: int, intermediate_size: int, activation_function: str):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation_fn = ACT2FN[activation_function]
        self.activation_dropout = config.activation_dropout
        self.dropout = config.dropout

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


class RTDetrFrozenBatchNorm2d(DetrFrozenBatchNorm2d):
    pass


class RTDetrSelfAttention(DetrSelfAttention):
    pass


def get_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_denoising_queries=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
):
    """
    Creates a contrastive denoising training group using ground-truth samples. It adds noise to labels and boxes.

    Args:
        targets (`list[dict]`):
            The target objects, each containing 'class_labels' and 'boxes' for objects in an image.
        num_classes (`int`):
            Total number of classes in the dataset.
        num_queries (`int`):
            Number of query slots in the transformer.
        class_embed (`callable`):
            A function or a model layer to embed class labels.
        num_denoising_queries (`int`, *optional*, defaults to 100):
            Number of denoising queries.
        label_noise_ratio (`float`, *optional*, defaults to 0.5):
            Ratio of noise applied to labels.
        box_noise_scale (`float`, *optional*, defaults to 1.0):
            Scale of noise applied to bounding boxes.
    Returns:
        `tuple` comprising various elements:
        - **input_query_class** (`torch.FloatTensor`) --
          Class queries with applied label noise.
        - **input_query_bbox** (`torch.FloatTensor`) --
          Bounding box queries with applied box noise.
        - **attn_mask** (`torch.FloatTensor`) --
           Attention mask for separating denoising and reconstruction queries.
        - **denoising_meta_values** (`dict`) --
          Metadata including denoising positive indices, number of groups, and split sizes.
    """

    if num_denoising_queries <= 0:
        return None, None, None, None

    num_ground_truths = [len(t["class_labels"]) for t in targets]
    device = targets[0]["class_labels"].device

    max_gt_num = max(num_ground_truths)
    if max_gt_num == 0:
        return None, None, None, None

    num_groups_denoising_queries = num_denoising_queries // max_gt_num
    num_groups_denoising_queries = 1 if num_groups_denoising_queries == 0 else num_groups_denoising_queries
    # pad gt to max_num of a batch
    batch_size = len(num_ground_truths)

    input_query_class = torch.full([batch_size, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([batch_size, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([batch_size, max_gt_num], dtype=torch.bool, device=device)

    for i in range(batch_size):
        num_gt = num_ground_truths[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]["class_labels"]
            input_query_bbox[i, :num_gt] = targets[i]["boxes"]
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_groups_denoising_queries])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_groups_denoising_queries, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_groups_denoising_queries])
    # positive and negative mask
    negative_gt_mask = torch.zeros([batch_size, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_groups_denoising_queries, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    denoise_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    denoise_positive_idx = torch.split(
        denoise_positive_idx, [n * num_groups_denoising_queries for n in num_ground_truths]
    )
    # total denoising queries
    num_denoising_queries = torch_int(max_gt_num * 2 * num_groups_denoising_queries)

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    if box_noise_scale > 0:
        known_bbox = center_to_corners_format(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        input_query_bbox = corners_to_center_format(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    input_query_class = class_embed(input_query_class)

    target_size = num_denoising_queries + num_queries
    attn_mask = torch.full([target_size, target_size], 0, dtype=torch.float, device=device)
    # match query cannot see the reconstruction
    attn_mask[num_denoising_queries:, :num_denoising_queries] = -torch.inf

    # reconstructions cannot see each other
    for i in range(num_groups_denoising_queries):
        idx_block_start = max_gt_num * 2 * i
        idx_block_end = max_gt_num * 2 * (i + 1)
        attn_mask[idx_block_start:idx_block_end, :idx_block_start] = -torch.inf
        attn_mask[idx_block_start:idx_block_end, idx_block_end:num_denoising_queries] = -torch.inf

    denoising_meta_values = {
        "dn_positive_idx": denoise_positive_idx,
        "dn_num_group": num_groups_denoising_queries,
        "dn_num_split": [num_denoising_queries, num_queries],
    }

    return input_query_class, input_query_bbox, attn_mask, denoising_meta_values


class RTDetrConvEncoder(nn.Module):
    """
    Convolutional backbone using the modeling_rt_detr_resnet.py.

    nn.BatchNorm2d layers are replaced by RTDetrFrozenBatchNorm2d as defined above.
    https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/src/nn/backbone/presnet.py#L142
    """

    def __init__(self, config):
        super().__init__()

        backbone = load_backbone(config)

        if config.freeze_backbone_batch_norms:
            # replace batch norm by frozen batch norm
            with torch.no_grad():
                replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = self.model.channels

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # send pixel_values through the model to get list of feature maps
        features = self.model(pixel_values).feature_maps

        out = []
        for feature_map in features:
            # downsample pixel_mask to match shape of corresponding feature_map
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out


class RTDetrConvNormLayer(nn.Module):
    def __init__(self, config, in_channels, out_channels, kernel_size, stride, padding=None, activation=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels, config.batch_norm_eps)
        self.activation = nn.Identity() if activation is None else ACT2CLS[activation]()

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        hidden_state = self.norm(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class RTDetrEncoderLayer(nn.Module):
    def __init__(self, config: RTDetrConfig):
        super().__init__()
        self.normalize_before = config.normalize_before
        self.hidden_size = config.encoder_hidden_dim

        # self-attention
        self.self_attn = RTDetrSelfAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout=config.dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.dropout = config.dropout
        self.mlp = RTDetrMLP(config, self.hidden_size, config.encoder_ffn_dim, config.encoder_activation_function)
        self.final_layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        spatial_position_embeddings: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, hidden_size)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            spatial_position_embeddings (`torch.FloatTensor`, *optional*):
                Spatial position embeddings (2D positional encodings of image locations), to be added to both
                the queries and keys in self-attention (but not to values).
        """
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=spatial_position_embeddings,
            **kwargs,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
        residual = hidden_states

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states


class RTDetrRepVggBlock(nn.Module):
    """
    RepVGG architecture block introduced by the work "RepVGG: Making VGG-style ConvNets Great Again".
    """

    def __init__(self, config: RTDetrConfig):
        super().__init__()

        activation = config.activation_function
        hidden_channels = int(config.encoder_hidden_dim * config.hidden_expansion)
        self.conv1 = RTDetrConvNormLayer(config, hidden_channels, hidden_channels, 3, 1, padding=1)
        self.conv2 = RTDetrConvNormLayer(config, hidden_channels, hidden_channels, 1, 1, padding=0)
        self.activation = nn.Identity() if activation is None else ACT2CLS[activation]()

    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        return self.activation(y)


class RTDetrCSPRepLayer(nn.Module):
    """
    Cross Stage Partial (CSP) network layer with RepVGG blocks.
    """

    def __init__(self, config: RTDetrConfig):
        super().__init__()

        in_channels = config.encoder_hidden_dim * 2
        out_channels = config.encoder_hidden_dim
        num_blocks = 3
        activation = config.activation_function

        hidden_channels = int(out_channels * config.hidden_expansion)
        self.conv1 = RTDetrConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.conv2 = RTDetrConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.bottlenecks = nn.Sequential(*[RTDetrRepVggBlock(config) for _ in range(num_blocks)])
        if hidden_channels != out_channels:
            self.conv3 = RTDetrConvNormLayer(config, hidden_channels, out_channels, 1, 1, activation=activation)
        else:
            self.conv3 = nn.Identity()

    def forward(self, hidden_state):
        hidden_state_1 = self.conv1(hidden_state)
        hidden_state_1 = self.bottlenecks(hidden_state_1)
        hidden_state_2 = self.conv2(hidden_state)
        return self.conv3(hidden_state_1 + hidden_state_2)


class RTDetrMultiscaleDeformableAttention(DeformableDetrMultiscaleDeformableAttention):
    pass


class RTDetrDecoderLayer(nn.Module):
    def __init__(self, config: RTDetrConfig):
        super().__init__()
        self.hidden_size = config.d_model

        # self-attention
        self.self_attn = RTDetrSelfAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_attention_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        # cross-attention
        self.encoder_attn = RTDetrMultiscaleDeformableAttention(
            config,
            num_heads=config.decoder_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        # feedforward neural networks
        self.mlp = RTDetrMLP(config, self.hidden_size, config.decoder_ffn_dim, config.decoder_activation_function)
        self.final_layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        object_queries_position_embeddings: torch.Tensor | None = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, hidden_size)`.
            object_queries_position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings for the object query slots. These are added to both queries and keys
                in the self-attention layer (not values).
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, hidden_size)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
        """
        residual = hidden_states

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            position_embeddings=object_queries_position_embeddings,
            **kwargs,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states

        # Cross-Attention
        hidden_states, _ = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=object_queries_position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            **kwargs,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class RTDetrSinePositionEmbedding(nn.Module):
    """
    2D sinusoidal position embedding used in RT-DETR hybrid encoder.
    """

    def __init__(self, embed_dim: int = 256, temperature: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

    @compile_compatible_method_lru_cache(maxsize=32)
    def forward(
        self,
        width: int,
        height: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Generate 2D sinusoidal position embeddings.

        Returns:
            Position embeddings of shape (1, height*width, embed_dim)
        """
        grid_w = torch.arange(torch_int(width), device=device).to(dtype)
        grid_h = torch.arange(torch_int(height), device=device).to(dtype)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="xy")
        if self.embed_dim % 4 != 0:
            raise ValueError("Embed dimension must be divisible by 4 for 2D sin-cos position embedding")
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, device=device).to(dtype) / pos_dim
        omega = 1.0 / (self.temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos()], dim=1)[None, :, :]


class RTDetrAIFILayer(nn.Module):
    """
    AIFI (Attention-based Intra-scale Feature Interaction) layer used in RT-DETR hybrid encoder.
    """

    def __init__(self, config: RTDetrConfig):
        super().__init__()
        self.config = config
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.eval_size = config.eval_size

        self.position_embedding = RTDetrSinePositionEmbedding(
            embed_dim=self.encoder_hidden_dim,
            temperature=config.positional_encoding_temperature,
        )
        self.layers = nn.ModuleList([RTDetrEncoderLayer(config) for _ in range(config.encoder_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, channels, height, width)`):
                Feature map to process.
        """
        batch_size = hidden_states.shape[0]
        height, width = hidden_states.shape[2:]

        hidden_states = hidden_states.flatten(2).permute(0, 2, 1)

        if self.training or self.eval_size is None:
            pos_embed = self.position_embedding(
                width=width,
                height=height,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        else:
            pos_embed = None

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                spatial_position_embeddings=pos_embed,
                **kwargs,
            )

        hidden_states = (
            hidden_states.permute(0, 2, 1).reshape(batch_size, self.encoder_hidden_dim, height, width).contiguous()
        )

        return hidden_states


class RTDetrMLPPredictionHead(DetrMLPPredictionHead):
    pass


@auto_docstring
class RTDetrPreTrainedModel(PreTrainedModel):
    config: RTDetrConfig
    base_model_prefix = "rt_detr"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _no_split_modules = [r"RTDetrHybridEncoder", r"RTDetrDecoderLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True
    _supports_flex_attn = True

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, RTDetrForObjectDetection):
            if module.model.decoder.class_embed is not None:
                for layer in module.model.decoder.class_embed:
                    prior_prob = self.config.initializer_bias_prior_prob or 1 / (self.config.num_labels + 1)
                    bias = float(-math.log((1 - prior_prob) / prior_prob))
                    init.xavier_uniform_(layer.weight)
                    init.constant_(layer.bias, bias)

            if module.model.decoder.bbox_embed is not None:
                for layer in module.model.decoder.bbox_embed:
                    init.constant_(layer.layers[-1].weight, 0)
                    init.constant_(layer.layers[-1].bias, 0)

        elif isinstance(module, RTDetrMultiscaleDeformableAttention):
            init.constant_(module.sampling_offsets.weight, 0.0)
            default_dtype = torch.get_default_dtype()
            thetas = torch.arange(module.n_heads, dtype=torch.int64).to(default_dtype) * (
                2.0 * math.pi / module.n_heads
            )
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.n_heads, 1, 1, 2)
                .repeat(1, module.n_levels, module.n_points, 1)
            )
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1

            init.copy_(module.sampling_offsets.bias, grid_init.view(-1))
            init.constant_(module.attention_weights.weight, 0.0)
            init.constant_(module.attention_weights.bias, 0.0)
            init.xavier_uniform_(module.value_proj.weight)
            init.constant_(module.value_proj.bias, 0.0)
            init.xavier_uniform_(module.output_proj.weight)
            init.constant_(module.output_proj.bias, 0.0)

        elif isinstance(module, RTDetrModel):
            prior_prob = self.config.initializer_bias_prior_prob or 1 / (self.config.num_labels + 1)
            bias = float(-math.log((1 - prior_prob) / prior_prob))
            init.xavier_uniform_(module.enc_score_head.weight)
            init.constant_(module.enc_score_head.bias, bias)

        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
            if getattr(module, "running_mean", None) is not None:
                init.zeros_(module.running_mean)
                init.ones_(module.running_var)
                init.zeros_(module.num_batches_tracked)

        elif isinstance(module, nn.LayerNorm):
            init.ones_(module.weight)
            init.zeros_(module.bias)

        if hasattr(module, "weight_embedding") and self.config.learn_initial_query:
            init.xavier_uniform_(module.weight_embedding.weight)
        if hasattr(module, "denoising_class_embed") and self.config.num_denoising > 0:
            init.xavier_uniform_(module.denoising_class_embed.weight)


class RTDetrHybridEncoder(RTDetrPreTrainedModel):
    """
    Hybrid encoder consisting of AIFI (Attention-based Intra-scale Feature Interaction) layers,
    a top-down Feature Pyramid Network (FPN) and a bottom-up Path Aggregation Network (PAN).
    More details on the paper: https://huggingface.co/papers/2304.08069

    Args:
        config: RTDetrConfig
    """

    _can_record_outputs = {
        "hidden_states": RTDetrAIFILayer,
        "attentions": RTDetrSelfAttention,
    }

    def __init__(self, config: RTDetrConfig):
        super().__init__(config)
        self.config = config
        self.in_channels = config.encoder_in_channels
        self.feat_strides = config.feat_strides
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.encode_proj_layers = config.encode_proj_layers
        self.positional_encoding_temperature = config.positional_encoding_temperature
        self.eval_size = config.eval_size
        self.out_channels = [self.encoder_hidden_dim for _ in self.in_channels]
        self.out_strides = self.feat_strides
        self.num_fpn_stages = len(self.in_channels) - 1
        self.num_pan_stages = len(self.in_channels) - 1

        # AIFI (Attention-based Intra-scale Feature Interaction) layers
        self.aifi = nn.ModuleList([RTDetrAIFILayer(config) for _ in range(len(self.encode_proj_layers))])

        # top-down FPN
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(self.num_fpn_stages):
            lateral_conv = RTDetrConvNormLayer(
                config,
                in_channels=self.encoder_hidden_dim,
                out_channels=self.encoder_hidden_dim,
                kernel_size=1,
                stride=1,
                activation=config.activation_function,
            )
            fpn_block = RTDetrCSPRepLayer(config)
            self.lateral_convs.append(lateral_conv)
            self.fpn_blocks.append(fpn_block)

        # bottom-up PAN
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(self.num_pan_stages):
            downsample_conv = RTDetrConvNormLayer(
                config,
                in_channels=self.encoder_hidden_dim,
                out_channels=self.encoder_hidden_dim,
                kernel_size=3,
                stride=2,
                activation=config.activation_function,
            )
            pan_block = RTDetrCSPRepLayer(config)
            self.downsample_convs.append(downsample_conv)
            self.pan_blocks.append(pan_block)

        self.post_init()

    @check_model_inputs(tie_last_hidden_states=False)
    def forward(
        self,
        inputs_embeds=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
        """
        feature_maps = inputs_embeds

        # AIFI: Apply transformer encoder to specified feature levels
        if self.config.encoder_layers > 0:
            for i, enc_ind in enumerate(self.encode_proj_layers):
                feature_maps[enc_ind] = self.aifi[i](feature_maps[enc_ind], **kwargs)

        # top-down FPN
        fpn_feature_maps = [feature_maps[-1]]
        for idx, (lateral_conv, fpn_block) in enumerate(zip(self.lateral_convs, self.fpn_blocks)):
            backbone_feature_map = feature_maps[self.num_fpn_stages - idx - 1]
            top_fpn_feature_map = fpn_feature_maps[-1]
            # apply lateral block
            top_fpn_feature_map = lateral_conv(top_fpn_feature_map)
            fpn_feature_maps[-1] = top_fpn_feature_map
            # apply fpn block
            top_fpn_feature_map = F.interpolate(top_fpn_feature_map, scale_factor=2.0, mode="nearest")
            fused_feature_map = torch.concat([top_fpn_feature_map, backbone_feature_map], dim=1)
            new_fpn_feature_map = fpn_block(fused_feature_map)
            fpn_feature_maps.append(new_fpn_feature_map)

        fpn_feature_maps.reverse()

        # bottom-up PAN
        pan_feature_maps = [fpn_feature_maps[0]]
        for idx, (downsample_conv, pan_block) in enumerate(zip(self.downsample_convs, self.pan_blocks)):
            top_pan_feature_map = pan_feature_maps[-1]
            fpn_feature_map = fpn_feature_maps[idx + 1]
            downsampled_feature_map = downsample_conv(top_pan_feature_map)
            fused_feature_map = torch.concat([downsampled_feature_map, fpn_feature_map], dim=1)
            new_pan_feature_map = pan_block(fused_feature_map)
            pan_feature_maps.append(new_pan_feature_map)

        return BaseModelOutput(last_hidden_state=pan_feature_maps)


class RTDetrDecoder(RTDetrPreTrainedModel):
    _can_record_outputs = {
        "hidden_states": RTDetrDecoderLayer,
        "attentions": RTDetrSelfAttention,
        "cross_attentions": RTDetrMultiscaleDeformableAttention,
    }

    def __init__(self, config: RTDetrConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layers = nn.ModuleList([RTDetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.query_pos_head = RTDetrMLPPredictionHead(4, 2 * config.d_model, config.d_model, num_layers=2)

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs()
    def forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:
                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).
            reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)` is `as_two_stage` else `(batch_size, num_queries, 2)` or , *optional*):
                Reference point in range `[0, 1]`, top-left (0,0), bottom-right (1, 1), including padding area.
            spatial_shapes (`torch.FloatTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of the feature maps.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`, *optional*):
                Indexes for the start of each feature level. In range `[0, sequence_length]`.
        """
        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # decoder layers
        intermediate = ()
        intermediate_reference_points = ()
        intermediate_logits = ()

        reference_points = F.sigmoid(reference_points)

        # https://github.com/lyuwenyu/RT-DETR/blob/94f5e16708329d2f2716426868ec89aa774af016/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py#L252
        for idx, decoder_layer in enumerate(self.layers):
            reference_points_input = reference_points.unsqueeze(2)
            object_queries_position_embeddings = self.query_pos_head(reference_points)

            hidden_states = decoder_layer(
                hidden_states,
                object_queries_position_embeddings=object_queries_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                encoder_attention_mask=encoder_attention_mask,
                **kwargs,
            )

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                predicted_corners = self.bbox_embed[idx](hidden_states)
                new_reference_points = F.sigmoid(predicted_corners + inverse_sigmoid(reference_points))
                reference_points = new_reference_points.detach()

            intermediate += (hidden_states,)
            intermediate_reference_points += (
                (new_reference_points,) if self.bbox_embed is not None else (reference_points,)
            )

            if self.class_embed is not None:
                logits = self.class_embed[idx](hidden_states)
                intermediate_logits += (logits,)

        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)
        if self.class_embed is not None:
            intermediate_logits = torch.stack(intermediate_logits, dim=1)

        return RTDetrDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_logits=intermediate_logits,
            intermediate_reference_points=intermediate_reference_points,
        )


@auto_docstring(
    custom_intro="""
    RT-DETR Model (consisting of a backbone and encoder-decoder) outputting raw hidden states without any head on top.
    """
)
class RTDetrModel(RTDetrPreTrainedModel):
    def __init__(self, config: RTDetrConfig):
        super().__init__(config)

        # Create backbone
        self.backbone = RTDetrConvEncoder(config)
        intermediate_channel_sizes = self.backbone.intermediate_channel_sizes

        # Create encoder input projection layers
        # https://github.com/lyuwenyu/RT-DETR/blob/94f5e16708329d2f2716426868ec89aa774af016/rtdetr_pytorch/src/zoo/rtdetr/hybrid_encoder.py#L212
        num_backbone_outs = len(intermediate_channel_sizes)
        encoder_input_proj_list = []
        for i in range(num_backbone_outs):
            in_channels = intermediate_channel_sizes[i]
            encoder_input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, config.encoder_hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(config.encoder_hidden_dim),
                )
            )
        self.encoder_input_proj = nn.ModuleList(encoder_input_proj_list)

        # Create encoder
        self.encoder = RTDetrHybridEncoder(config)

        # denoising part
        if config.num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                config.num_labels + 1, config.d_model, padding_idx=config.num_labels
            )

        # decoder embedding
        if config.learn_initial_query:
            self.weight_embedding = nn.Embedding(config.num_queries, config.d_model)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
        )
        self.enc_score_head = nn.Linear(config.d_model, config.num_labels)
        self.enc_bbox_head = RTDetrMLPPredictionHead(config.d_model, config.d_model, 4, num_layers=3)

        # init encoder output anchors and valid_mask
        if config.anchor_image_size:
            self.anchors, self.valid_mask = self.generate_anchors(dtype=self.dtype)

        # Create decoder input projection layers
        # https://github.com/lyuwenyu/RT-DETR/blob/94f5e16708329d2f2716426868ec89aa774af016/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py#L412
        num_backbone_outs = len(config.decoder_in_channels)
        decoder_input_proj_list = []
        for i in range(num_backbone_outs):
            in_channels = config.decoder_in_channels[i]
            decoder_input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, config.d_model, kernel_size=1, bias=False),
                    nn.BatchNorm2d(config.d_model, config.batch_norm_eps),
                )
            )
        for _ in range(config.num_feature_levels - num_backbone_outs):
            decoder_input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(config.d_model, config.batch_norm_eps),
                )
            )
            in_channels = config.d_model
        self.decoder_input_proj = nn.ModuleList(decoder_input_proj_list)

        # decoder
        self.decoder = RTDetrDecoder(config)

        self.post_init()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(True)

    @compile_compatible_method_lru_cache(maxsize=32)
    def generate_anchors(self, spatial_shapes=None, grid_size=0.05, device="cpu", dtype=torch.float32):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.config.anchor_image_size[0] / s), int(self.config.anchor_image_size[1] / s)]
                for s in self.config.feat_strides
            ]
        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=height, device=device).to(dtype),
                torch.arange(end=width, device=device).to(dtype),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)
            grid_xy = grid_xy.unsqueeze(0) + 0.5
            grid_xy[..., 0] /= width
            grid_xy[..., 1] /= height
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**level)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, height * width, 4))
        # define the valid range for anchor coordinates
        eps = 1e-2
        anchors = torch.concat(anchors, 1)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.tensor(torch.finfo(dtype).max, dtype=dtype, device=device))

        return anchors, valid_mask

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = None,
        encoder_outputs: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: list[dict] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | RTDetrModelOutput:
        r"""
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
            can choose to directly pass a flattened representation of an image.
        labels (`list[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, RTDetrModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
        >>> model = RTDetrModel.from_pretrained("PekingU/rtdetr_r50vd")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 300, 256]
        ```"""
        if pixel_values is None and inputs_embeds is None:
            raise ValueError("You have to specify either pixel_values or inputs_embeds")

        if inputs_embeds is None:
            batch_size, num_channels, height, width = pixel_values.shape
            device = pixel_values.device
            if pixel_mask is None:
                pixel_mask = torch.ones(((batch_size, height, width)), device=device)
            features = self.backbone(pixel_values, pixel_mask)
            proj_feats = [self.encoder_input_proj[level](source) for level, (source, mask) in enumerate(features)]
        else:
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device
            proj_feats = inputs_embeds

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                proj_feats,
                **kwargs,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput
        elif not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # Equivalent to def _get_encoder_input
        # https://github.com/lyuwenyu/RT-DETR/blob/94f5e16708329d2f2716426868ec89aa774af016/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py#L412
        sources = []
        for level, source in enumerate(encoder_outputs.last_hidden_state):
            sources.append(self.decoder_input_proj[level](source))

        # Lowest resolution feature maps are obtained via 3x3 stride 2 convolutions on the final stage
        if self.config.num_feature_levels > len(sources):
            _len_sources = len(sources)
            sources.append(self.decoder_input_proj[_len_sources](encoder_outputs.last_hidden_state)[-1])
            for i in range(_len_sources + 1, self.config.num_feature_levels):
                sources.append(self.decoder_input_proj[i](encoder_outputs.last_hidden_state[-1]))

        # Prepare encoder inputs (by flattening)
        source_flatten = []
        spatial_shapes_list = []
        spatial_shapes = torch.empty((len(sources), 2), device=device, dtype=torch.long)
        for level, source in enumerate(sources):
            height, width = source.shape[-2:]
            spatial_shapes[level, 0] = height
            spatial_shapes[level, 1] = width
            spatial_shapes_list.append((height, width))
            source = source.flatten(2).transpose(1, 2)
            source_flatten.append(source)
        source_flatten = torch.cat(source_flatten, 1)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # prepare denoising training
        if self.training and self.config.num_denoising > 0 and labels is not None:
            (
                denoising_class,
                denoising_bbox_unact,
                attention_mask,
                denoising_meta_values,
            ) = get_contrastive_denoising_training_group(
                targets=labels,
                num_classes=self.config.num_labels,
                num_queries=self.config.num_queries,
                class_embed=self.denoising_class_embed,
                num_denoising_queries=self.config.num_denoising,
                label_noise_ratio=self.config.label_noise_ratio,
                box_noise_scale=self.config.box_noise_scale,
            )
        else:
            denoising_class, denoising_bbox_unact, attention_mask, denoising_meta_values = None, None, None, None

        batch_size = len(source_flatten)
        device = source_flatten.device
        dtype = source_flatten.dtype

        # prepare input for decoder
        if self.training or self.config.anchor_image_size is None:
            # Pass spatial_shapes as tuple to make it hashable and make sure
            # lru_cache is working for generate_anchors()
            spatial_shapes_tuple = tuple(spatial_shapes_list)
            anchors, valid_mask = self.generate_anchors(spatial_shapes_tuple, device=device, dtype=dtype)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
            anchors, valid_mask = anchors.to(device, dtype), valid_mask.to(device, dtype)

        # use the valid_mask to selectively retain values in the feature map where the mask is `True`
        memory = valid_mask.to(source_flatten.dtype) * source_flatten

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_logits = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.config.num_queries, dim=1)

        reference_points_unact = enc_outputs_coord_logits.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_logits.shape[-1])
        )

        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)

        enc_topk_logits = enc_outputs_class.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1])
        )

        # extract region features
        if self.config.learn_initial_query:
            target = self.weight_embedding.tile([batch_size, 1, 1])
        else:
            target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        init_reference_points = reference_points_unact.detach()

        # decoder
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=attention_mask,
            reference_points=init_reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            **kwargs,
        )

        return RTDetrModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_logits=decoder_outputs.intermediate_logits,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            intermediate_predicted_corners=decoder_outputs.intermediate_predicted_corners,
            initial_reference_points=decoder_outputs.initial_reference_points,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            init_reference_points=init_reference_points,
            enc_topk_logits=enc_topk_logits,
            enc_topk_bboxes=enc_topk_bboxes,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
            denoising_meta_values=denoising_meta_values,
        )


@auto_docstring(
    custom_intro="""
    RT-DETR Model (consisting of a backbone and encoder-decoder) outputting bounding boxes and logits to be further
    decoded into scores and classes.
    """
)
class RTDetrForObjectDetection(RTDetrPreTrainedModel):
    # When using clones, all layers > 0 will be clones, but layer 0 *is* required
    # We can't initialize the model on meta device as some weights are modified during the initialization
    _no_split_modules = None

    def __init__(self, config: RTDetrConfig):
        super().__init__(config)
        self.model = RTDetrModel(config)
        num_pred = config.decoder_layers
        self.model.decoder.class_embed = nn.ModuleList(
            [torch.nn.Linear(config.d_model, config.num_labels) for _ in range(num_pred)]
        )
        self.model.decoder.bbox_embed = nn.ModuleList(
            [RTDetrMLPPredictionHead(config.d_model, config.d_model, 4, num_layers=3) for _ in range(num_pred)]
        )
        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        self.post_init()

    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = None,
        encoder_outputs: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: list[dict] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | RTDetrObjectDetectionOutput:
        r"""
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
            can choose to directly pass a flattened representation of an image.
        labels (`list[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Examples:

        ```python
        >>> from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
        >>> from PIL import Image
        >>> import requests
        >>> import torch

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
        >>> model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> list(logits.shape)
        [1, 300, 80]

        >>> boxes = outputs.pred_boxes
        >>> list(boxes.shape)
        [1, 300, 4]

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
        ...     0
        ... ]

        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected sofa with confidence 0.97 at location [0.14, 0.38, 640.13, 476.21]
        Detected cat with confidence 0.96 at location [343.38, 24.28, 640.14, 371.5]
        Detected cat with confidence 0.958 at location [13.23, 54.18, 318.98, 472.22]
        Detected remote with confidence 0.951 at location [40.11, 73.44, 175.96, 118.48]
        Detected remote with confidence 0.924 at location [333.73, 76.58, 369.97, 186.99]
        ```"""
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs,
        )

        denoising_meta_values = outputs.denoising_meta_values if self.training else None

        outputs_class = outputs.intermediate_logits
        outputs_coord = outputs.intermediate_reference_points
        predicted_corners = outputs.intermediate_predicted_corners
        initial_reference_points = outputs.initial_reference_points

        logits = outputs_class[:, -1]
        pred_boxes = outputs_coord[:, -1]

        loss, loss_dict, auxiliary_outputs, enc_topk_logits, enc_topk_bboxes = None, None, None, None, None
        if labels is not None:
            enc_topk_logits = outputs.enc_topk_logits
            enc_topk_bboxes = outputs.enc_topk_bboxes
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits,
                labels,
                self.device,
                pred_boxes,
                self.config,
                outputs_class,
                outputs_coord,
                enc_topk_logits=enc_topk_logits,
                enc_topk_bboxes=enc_topk_bboxes,
                denoising_meta_values=denoising_meta_values,
                predicted_corners=predicted_corners,
                initial_reference_points=initial_reference_points,
                **kwargs,
            )

        return RTDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_logits=outputs.intermediate_logits,
            intermediate_reference_points=outputs.intermediate_reference_points,
            intermediate_predicted_corners=outputs.intermediate_predicted_corners,
            initial_reference_points=outputs.initial_reference_points,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            init_reference_points=outputs.init_reference_points,
            enc_topk_logits=outputs.enc_topk_logits,
            enc_topk_bboxes=outputs.enc_topk_bboxes,
            enc_outputs_class=outputs.enc_outputs_class,
            enc_outputs_coord_logits=outputs.enc_outputs_coord_logits,
            denoising_meta_values=outputs.denoising_meta_values,
        )


__all__ = [
    "RTDetrImageProcessorFast",
    "RTDetrForObjectDetection",
    "RTDetrModel",
    "RTDetrPreTrainedModel",
]
