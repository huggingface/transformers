#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file was automatically generated from src/transformers/models/rt_detr/modular_rt_detr.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. If any change should be done, please apply the change to the
#                          modular_rt_detr.py file directly. One of our CI enforces this.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    SizeDict,
    get_image_size_for_max_height_width,
    get_max_height_width,
    safe_squeeze,
)
from ...image_transforms import center_to_corners_format, corners_to_center_format
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    AnnotationFormat,
    AnnotationType,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    validate_annotations,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    requires_backends,
)
from .image_processing_rt_detr import get_size_with_aspect_ratio


if is_torch_available():
    import torch


if is_torchvision_v2_available():
    from torchvision.transforms.v2 import functional as F
elif is_torchvision_available():
    from torchvision.transforms import functional as F


class RTDetrFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    r"""
    format (`str`, *optional*, defaults to `AnnotationFormat.COCO_DETECTION`):
        Data format of the annotations. One of "coco_detection" or "coco_panoptic".
    do_convert_annotations (`bool`, *optional*, defaults to `True`):
        Controls whether to convert the annotations to the format expected by the RT_DETR model. Converts the
        bounding boxes to the format `(center_x, center_y, width, height)` and in the range `[0, 1]`.
        Can be overridden by the `do_convert_annotations` parameter in the `preprocess` method.
    do_pad (`bool`, *optional*, defaults to `True`):
        Controls whether to pad the image. Can be overridden by the `do_pad` parameter in the `preprocess`
        method. If `True`, padding will be applied to the bottom and right of the image with zeros.
        If `pad_size` is provided, the image will be padded to the specified dimensions.
        Otherwise, the image will be padded to the maximum height and width of the batch.
    pad_size (`Dict[str, int]`, *optional*):
        The size `{"height": int, "width" int}` to pad the images to. Must be larger than any image size
        provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest
        height and width in the batch.
    return_segmentation_masks (`bool`, *optional*, defaults to `False`):
        Whether to return segmentation masks.
    """

    format: Optional[Union[str, AnnotationFormat]]
    do_convert_annotations: Optional[bool]
    do_pad: Optional[bool]
    pad_size: Optional[Dict[str, int]]
    return_segmentation_masks: Optional[bool]


SUPPORTED_ANNOTATION_FORMATS = (AnnotationFormat.COCO_DETECTION, AnnotationFormat.COCO_PANOPTIC)


def prepare_coco_detection_annotation(
    image,
    target,
    return_segmentation_masks: bool = False,
    input_data_format: Optional[Union[ChannelDimension, str]] = None,
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


@auto_docstring
class RTDetrImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    format = AnnotationFormat.COCO_DETECTION
    do_resize = True
    do_rescale = True
    do_normalize = False
    do_pad = False
    size = {"height": 640, "width": 640}
    default_to_square = False
    model_input_names = ["pixel_values", "pixel_mask"]
    valid_kwargs = RTDetrFastImageProcessorKwargs
    do_convert_annotations = True

    def __init__(self, **kwargs: Unpack[RTDetrFastImageProcessorKwargs]) -> None:
        # Backwards compatibility
        do_convert_annotations = kwargs.get("do_convert_annotations", None)
        do_normalize = kwargs.get("do_normalize", None)
        if do_convert_annotations is None and getattr(self, "do_convert_annotations", None) is None:
            self.do_convert_annotations = do_normalize if do_normalize is not None else self.do_normalize

        super().__init__(**kwargs)

    def prepare_annotation(
        self,
        image: torch.Tensor,
        target: Dict,
        format: Optional[AnnotationFormat] = None,
        return_segmentation_masks: Optional[bool] = None,
        masks_path: Optional[Union[str, pathlib.Path]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Dict:
        """
        Prepare an annotation for feeding into RT_DETR model.
        """
        format = format if format is not None else self.format

        if format == AnnotationFormat.COCO_DETECTION:
            return_segmentation_masks = False if return_segmentation_masks is None else return_segmentation_masks
            target = prepare_coco_detection_annotation(
                image, target, return_segmentation_masks, input_data_format=input_data_format
            )
        else:
            raise ValueError(f"Format {format} is not supported.")
        return target

    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Resize the image to the given size. Size can be `min_size` (scalar) or `(height, width)` tuple. If size is an
        int, smaller edge of the image will be matched to this number.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Size of the image's `(height, width)` dimensions after resizing. Available options are:
                    - `{"height": int, "width": int}`: The image will be resized to the exact size `(height, width)`.
                        Do NOT keep the aspect ratio.
                    - `{"shortest_edge": int, "longest_edge": int}`: The image will be resized to a maximum size respecting
                        the aspect ratio and keeping the shortest edge less or equal to `shortest_edge` and the longest edge
                        less or equal to `longest_edge`.
                    - `{"max_height": int, "max_width": int}`: The image will be resized to the maximum size respecting the
                        aspect ratio and keeping the height less or equal to `max_height` and the width less or equal to
                        `max_width`.
            interpolation (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                Resampling filter to use if resizing the image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR
        if size.shortest_edge and size.longest_edge:
            # Resize the image so that the shortest edge or the longest edge is of the given size
            # while maintaining the aspect ratio of the original image.
            new_size = get_size_with_aspect_ratio(
                image.size()[-2:],
                size["shortest_edge"],
                size["longest_edge"],
            )
        elif size.max_height and size.max_width:
            new_size = get_image_size_for_max_height_width(image.size()[-2:], size["max_height"], size["max_width"])
        elif size.height and size.width:
            new_size = (size["height"], size["width"])
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )

        image = F.resize(
            image,
            size=new_size,
            interpolation=interpolation,
            **kwargs,
        )
        return image

    def resize_annotation(
        self,
        annotation: Dict[str, Any],
        orig_size: Tuple[int, int],
        target_size: Tuple[int, int],
        threshold: float = 0.5,
        interpolation: "F.InterpolationMode" = None,
    ):
        """
        Resizes an annotation to a target size.

        Args:
            annotation (`Dict[str, Any]`):
                The annotation dictionary.
            orig_size (`Tuple[int, int]`):
                The original size of the input image.
            target_size (`Tuple[int, int]`):
                The target size of the image, as returned by the preprocessing `resize` step.
            threshold (`float`, *optional*, defaults to 0.5):
                The threshold used to binarize the segmentation masks.
            resample (`InterpolationMode`, defaults to `InterpolationMode.NEAREST`):
                The resampling filter to use when resizing the masks.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.NEAREST
        ratio_height, ratio_width = [target / orig for target, orig in zip(target_size, orig_size)]

        new_annotation = {}
        new_annotation["size"] = target_size

        for key, value in annotation.items():
            if key == "boxes":
                boxes = value
                scaled_boxes = boxes * torch.as_tensor(
                    [ratio_width, ratio_height, ratio_width, ratio_height], dtype=torch.float32, device=boxes.device
                )
                new_annotation["boxes"] = scaled_boxes
            elif key == "area":
                area = value
                scaled_area = area * (ratio_width * ratio_height)
                new_annotation["area"] = scaled_area
            elif key == "masks":
                masks = value[:, None]
                masks = [F.resize(mask, target_size, interpolation=interpolation) for mask in masks]
                masks = torch.stack(masks).to(torch.float32)
                masks = masks[:, 0] > threshold
                new_annotation["masks"] = masks
            elif key == "size":
                new_annotation["size"] = target_size
            else:
                new_annotation[key] = value

        return new_annotation

    def normalize_annotation(self, annotation: Dict, image_size: Tuple[int, int]) -> Dict:
        image_height, image_width = image_size
        norm_annotation = {}
        for key, value in annotation.items():
            if key == "boxes":
                boxes = value
                boxes = corners_to_center_format(boxes)
                boxes /= torch.as_tensor(
                    [image_width, image_height, image_width, image_height], dtype=torch.float32, device=boxes.device
                )
                norm_annotation[key] = boxes
            else:
                norm_annotation[key] = value
        return norm_annotation

    def _update_annotation_for_padded_image(
        self,
        annotation: Dict,
        input_image_size: Tuple[int, int],
        output_image_size: Tuple[int, int],
        padding,
        update_bboxes,
    ) -> Dict:
        """
        Update the annotation for a padded image.
        """
        new_annotation = {}
        new_annotation["size"] = output_image_size
        ratio_height, ratio_width = (input / output for output, input in zip(output_image_size, input_image_size))

        for key, value in annotation.items():
            if key == "masks":
                masks = value
                masks = F.pad(
                    masks,
                    padding,
                    fill=0,
                )
                masks = safe_squeeze(masks, 1)
                new_annotation["masks"] = masks
            elif key == "boxes" and update_bboxes:
                boxes = value
                boxes *= torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height], device=boxes.device)
                new_annotation["boxes"] = boxes
            elif key == "size":
                new_annotation["size"] = output_image_size
            else:
                new_annotation[key] = value
        return new_annotation

    def pad(
        self,
        image: torch.Tensor,
        padded_size: Tuple[int, int],
        annotation: Optional[Dict[str, Any]] = None,
        update_bboxes: bool = True,
        fill: int = 0,
    ):
        original_size = image.size()[-2:]
        padding_bottom = padded_size[0] - original_size[0]
        padding_right = padded_size[1] - original_size[1]
        if padding_bottom < 0 or padding_right < 0:
            raise ValueError(
                f"Padding dimensions are negative. Please make sure that the padded size is larger than the "
                f"original size. Got padded size: {padded_size}, original size: {original_size}."
            )
        if original_size != padded_size:
            padding = [0, 0, padding_right, padding_bottom]
            image = F.pad(image, padding, fill=fill)
            if annotation is not None:
                annotation = self._update_annotation_for_padded_image(
                    annotation, original_size, padded_size, padding, update_bboxes
                )

        # Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.
        pixel_mask = torch.zeros(padded_size, dtype=torch.int64, device=image.device)
        pixel_mask[: original_size[0], : original_size[1]] = 1

        return image, pixel_mask, annotation

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = None,
        masks_path: Optional[Union[str, pathlib.Path]] = None,
        **kwargs: Unpack[RTDetrFastImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        annotations (`AnnotationType` or `List[AnnotationType]`, *optional*):
            List of annotations associated with the image or batch of images. If annotation is for object
            detection, the annotations should be a dictionary with the following keys:
            - "image_id" (`int`): The image id.
            - "annotations" (`List[Dict]`): List of annotations for an image. Each annotation should be a
                dictionary. An image can have no annotations, in which case the list should be empty.
            If annotation is for segmentation, the annotations should be a dictionary with the following keys:
            - "image_id" (`int`): The image id.
            - "segments_info" (`List[Dict]`): List of segments for an image. Each segment should be a dictionary.
                An image can have no segments, in which case the list should be empty.
            - "file_name" (`str`): The file name of the image.
        format (`str`, *optional*, defaults to `AnnotationFormat.COCO_DETECTION`):
            Data format of the annotations. One of "coco_detection" or "coco_panoptic".
        masks_path (`str` or `pathlib.Path`, *optional*):
            Path to the directory containing the segmentation masks.

            annotations (`AnnotationType` or `List[AnnotationType]`, *optional*):
                List of annotations associated with the image or batch of images. If annotation is for object
                detection, the annotations should be a dictionary with the following keys:
                - "image_id" (`int`): The image id.
                - "annotations" (`List[Dict]`): List of annotations for an image. Each annotation should be a
                    dictionary. An image can have no annotations, in which case the list should be empty.
                If annotation is for segmentation, the annotations should be a dictionary with the following keys:
                - "image_id" (`int`): The image id.
                - "segments_info" (`List[Dict]`): List of segments for an image. Each segment should be a dictionary.
                    An image can have no segments, in which case the list should be empty.
                - "file_name" (`str`): The file name of the image.
            masks_path (`str` or `pathlib.Path`, *optional*):
                Path to the directory containing the segmentation masks.
        """
        return super().preprocess(images, annotations=annotations, masks_path=masks_path, **kwargs)

    def _preprocess(
        self,
        images: List["torch.Tensor"],
        annotations: Optional[Union[AnnotationType, List[AnnotationType]]],
        return_segmentation_masks: bool,
        masks_path: Optional[Union[str, pathlib.Path]],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        do_convert_annotations: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        do_pad: bool,
        pad_size: Optional[Dict[str, int]],
        format: Optional[Union[str, AnnotationFormat]],
        return_tensors: Optional[Union[str, TensorType]],
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
                padded_size = (pad_size["height"], pad_size["width"])
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
        target_sizes: Union[TensorType, List[Tuple]] = None,
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
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
            use_focal_loss (`bool` defaults to `True`):
                Variable informing if the focal loss was used to predict the outputs. If `True`, a sigmoid is applied
                to compute the scores of each detection, otherwise, a softmax function is used.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
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
            if isinstance(target_sizes, List):
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


__all__ = ["RTDetrImageProcessorFast"]
