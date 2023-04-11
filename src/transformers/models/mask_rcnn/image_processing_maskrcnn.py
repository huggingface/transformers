# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for Mask-RCNN."""

import pathlib
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import normalize, rescale, resize, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...nms import multiclass_nms
from ...utils import ExplicitEnum, TensorType, is_torch_available, logging
from .modeling_maskrcnn import MaskRCNNDeltaXYWHBBoxCoder


if is_torch_available():
    import torch
    from torch import nn

logger = logging.get_logger(__name__)


# TODO move this to general utils?
AnnotationType = Dict[str, Union[int, str, List[Dict]]]


class AnnotionFormat(ExplicitEnum):
    COCO_DETECTION = "coco_detection"
    COCO_PANOPTIC = "coco_panoptic"


BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


# Copied from transformers.models.detr.image_processing_detr.get_size_with_aspect_ratio
def get_size_with_aspect_ratio(image_size, size, max_size=None) -> Tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size.

    Args:
        image_size (`Tuple[int, int]`):
            The input image size.
        size (`int`):
            The desired output size.
        max_size (`int`, *optional*):
            The maximum allowed output size.
    """
    height, width = image_size
    if max_size is not None:
        min_original_size = float(min((height, width)))
        max_original_size = float(max((height, width)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (height <= width and height == size) or (width <= height and width == size):
        return height, width

    if width < height:
        ow = size
        oh = int(size * height / width)
    else:
        oh = size
        ow = int(size * width / height)
    return (oh, ow)


# Copied from transformers.models.detr.image_processing_detr.get_resize_output_image_size
def get_resize_output_image_size(
    input_image: np.ndarray, size: Union[int, Tuple[int, int], List[int]], max_size: Optional[int] = None
) -> Tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size. If the desired output size
    is a tuple or list, the output image size is returned as is. If the desired output size is an integer, the output
    image size is computed by keeping the aspect ratio of the input image size.

    Args:
        image_size (`Tuple[int, int]`):
            The input image size.
        size (`int`):
            The desired output size.
        max_size (`int`, *optional*):
            The maximum allowed output size.
    """
    image_size = get_image_size(input_image)
    if isinstance(size, (list, tuple)):
        return size

    return get_size_with_aspect_ratio(image_size, size, max_size)


def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks according to boxes.

    This implementation is modified from https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor):
            N, 1, H, W
        boxes (Tensor):
            N, 4
        img_h (int):
            Height of the image to be pasted.
        img_w (int):
            Width of the image to be pasted.
        skip_empty (bool):
            Only paste masks within the region that tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device).to(torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device).to(torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    # IsInf op is not supported with ONNX<=1.7.0
    if not torch.onnx.is_in_onnx_export():
        if torch.isinf(img_x).any():
            inds = torch.where(torch.isinf(img_x))
            img_x[inds] = 0
        if torch.isinf(img_y).any():
            inds = torch.where(torch.isinf(img_y))
            img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = nn.functional.grid_sample(masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()


class MaskRCNNImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Mask R-CNN image processor.

    Args:
        TODO
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = {"shortest_edge": 800, "longest_edge": 1333},
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Union[float, List[float]] = None,
        image_std: Union[float, List[float]] = None,
        test_cfg={
            "score_thr": 0.05,
            "nms": {"type": "nms", "iou_threshold": 0.5},
            "max_per_img": 100,
            "mask_thr_binary": 0.5,
        },
        num_classes=80,
        bbox_head_bbox_coder_target_means=[0.0, 0.0, 0.0, 0.0],
        bbox_head_bbox_coder_target_stds=[0.1, 0.1, 0.2, 0.2],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.test_cfg = test_cfg
        self.num_classes = num_classes
        self.bbox_coder = MaskRCNNDeltaXYWHBBoxCoder(
            target_means=bbox_head_bbox_coder_target_means, target_stds=bbox_head_bbox_coder_target_stds
        )
        # TODO remove this attribute
        self.class_agnostic = False

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[ChannelDimension] = None,
    ) -> np.ndarray:
        """
        Resize the image to the given size. Size can be `min_size` (scalar) or `(height, width)` tuple. If size is an
        int, smaller edge of the image will be matched to this number.
        """
        if "shortest_edge" in size and "longest_edge" in size:
            size = get_resize_output_image_size(image, size["shortest_edge"], size["longest_edge"])
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        image = resize(image, size=size, resample=resample, data_format=data_format)
        return image

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.rescale
    def rescale(
        self, image: np.ndarray, rescale_factor: Union[float, int], data_format: Optional[ChannelDimension] = None
    ) -> np.ndarray:
        """
        Rescale the image by the given factor.
        """
        return rescale(image, rescale_factor, data_format=data_format)

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.normalize
    def normalize(
        self,
        image: np.ndarray,
        mean: Union[float, Iterable[float]],
        std: Union[float, Iterable[float]],
        data_format: Optional[ChannelDimension] = None,
    ) -> np.ndarray:
        """
        Normalize the image with the given mean and standard deviation.
        """
        return normalize(image, mean=mean, std=std, data_format=data_format)

    def preprocess(
        self,
        images: ImageInput,
        annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = None,
        return_segmentation_masks: bool = None,
        masks_path: Optional[Union[str, pathlib.Path]] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample=None,  # PILImageResampling
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[Union[int, float]] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        format: Optional[Union[str, AnnotionFormat]] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or a batch of images so that it can be used by the model.

        Args:
            ...

        """

        do_resize = self.do_resize if do_resize is None else do_resize
        size = self.size if size is None else size
        resample = self.resample if resample is None else resample
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std

        if do_resize is not None and size is None:
            raise ValueError("Size and max_size must be specified if do_resize is True.")

        if do_rescale is not None and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize is not None and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if annotations is not None:
            raise NotImplementedError("To do")

        # All transformations expect numpy arrays
        images = [to_numpy_array(image) for image in images]
        image_shapes = [image.shape for image in images]

        # transformations
        if do_resize:
            if annotations is not None:
                raise NotImplementedError("To do")
            else:
                images = [self.resize(image, size=size, resample=resample) for image in images]
                image_shapes = [image.shape for image in images]

        if do_rescale:
            images = [self.rescale(image, rescale_factor) for image in images]

        if do_normalize:
            images = [self.normalize(image, image_mean, image_std) for image in images]
            if annotations is not None:
                raise NotImplementedError("To do")

        images = [to_channel_dimension_format(image, data_format) for image in images]
        data = {"pixel_values": images}

        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        # add metadata
        # TODO perhaps remove?
        img_metas = [{"img_shape": image_shape, "pad_shape": image_shape} for image_shape in image_shapes]
        encoded_inputs["img_metas"] = img_metas

        if annotations is not None:
            raise NotImplementedError("To do")

        return encoded_inputs

    def get_bboxes(self, rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False, cfg=None):
        # # some loss (Seesaw loss..) may have custom activation
        # if self.custom_cls_channels:
        #     scores = self.loss_cls.get_activation(cls_score)
        # else:
        scores = nn.functional.softmax(cls_score, dim=-1) if cls_score is not None else None
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                img_shape = img_shape[-2:]
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            # it's here that we create a different amount of objects per image in a batch
            det_bboxes, det_labels = multiclass_nms(bboxes, scores, cfg["score_thr"], cfg["nms"], cfg["max_per_img"])

            return det_bboxes, det_labels

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of model, whose type is Tensor, while for
                multi-scale testing, it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape (Tuple): original image height and width, shape (2,)
            scale_factor(ndarray | Tensor): If `rescale is True`, box
                coordinates are divided by this scale factor to fit `ori_shape`.
            rescale (bool): If True, the resulting masks will be rescaled to
                `ori_shape`.

        Returns:
            list[list]: encoded masks. The c-th item in the outer list
                corresponds to the c-th class. Given the c-th outer list, the i-th item in that inner list is the mask
                for the i-th box with class label c.
        Example:
            >>> import mmcv >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import * # NOQA >>> N = 7 # N =
            number of extracted ROIs >>> C, H, W = 11, 32, 32 >>> # Create example instance of FCN Mask Head. >>> self
            = FCNMaskHead(num_classes=C, num_convs=0) >>> inputs = torch.rand(N, self.in_channels, H, W) >>> mask_pred
            = self.forward(inputs) >>> # Each input is associated with some bounding box >>> det_bboxes =
            torch.Tensor([[1, 1, 42, 42 ]] * N) >>> det_labels = torch.randint(0, C, size=(N,)) >>> rcnn_test_cfg =
            mmcv.Config({'mask_thr_binary': 0, }) >>> ori_shape = (H * 4, W * 4) >>> scale_factor =
            torch.FloatTensor((1, 1)) >>> rescale = False >>> # Encoded masks are a list for each category. >>>
            encoded_masks = self.get_seg_masks( >>> mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, >>>
            scale_factor, rescale >>> ) >>> assert len(encoded_masks) == C >>> assert sum(list(map(len,
            encoded_masks))) == N
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            # In AugTest, has been activated before
            mask_pred = det_bboxes.new_tensor(mask_pred)

        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)]  # BG is not included in num_classes
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        if rescale:
            img_h, img_w = ori_shape[-2:]
            bboxes = bboxes / scale_factor.to(bboxes)
        else:
            ori_shape = ori_shape[-2:]
            w_scale, h_scale = scale_factor[0], scale_factor[1]
            img_h = np.round(ori_shape[0] * h_scale.item()).astype(np.int32)
            img_w = np.round(ori_shape[1] * w_scale.item()).astype(np.int32)

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == "cpu":
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            # the types of img_w and img_h are np.int32,
            # when the image resolution is large,
            # the calculation of num_chunks will overflow.
            # so we need to change the types of img_w and img_h to int.
            # See https://github.com/open-mmlab/mmdetection/pull/5191
            num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            if num_chunks > N:
                raise ValueError("Default GPU_MEM_LIMIT is too small; try increasing it")
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg["mask_thr_binary"]
        im_mask = torch.zeros(N, img_h, img_w, device=device, dtype=torch.bool if threshold >= 0 else torch.uint8)

        if not self.class_agnostic:
            mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds], bboxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
            )

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds,) + spatial_inds] = masks_chunk

        for i in range(N):
            cls_segms[labels[i]].append(im_mask[i].detach().cpu().numpy())
        return cls_segms

    def post_process_object_detection(
        self,
        outputs,
        threshold: float = 0.5,
        target_sizes=None,
        scale_factors=None,
    ):
        """
        Converts the output of [`MaskRCNNForObjectDetection`] into the format expected by the COCO api. Only supports
        PyTorch.

        Args:
            outputs ([`MaskRCNNForObjectDetection`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*, defaults to `None`):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                (height, width) of each image in the batch. If left to None, predictions will not be resized.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        rois, proposals, cls_score, bbox_pred = outputs.rois, outputs.proposals, outputs.logits, outputs.pred_boxes

        # reshape back to (batch_size*num_proposals_per_image, ...)
        cls_score = cls_score.reshape(-1, cls_score.shape[-1])
        bbox_pred = bbox_pred.reshape(-1, bbox_pred.shape[-1])

        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None,) * len(proposals)

        # calculate img shapes based on target sizes (ori shapes) + scale factors
        img_shapes = []
        for target_size, scale_factor in zip(target_sizes, scale_factors):
            height, width = target_size[-2:]
            img_shape = (3, height * scale_factor[1], width * scale_factor[0])
            img_shapes.append(img_shape)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0,), dtype=torch.long)
                if self.test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros((0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=True,  # we rescale by default
                    cfg=self.test_cfg,
                )
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        # turn into COCO API
        results = []
        for example_idx in range(len(det_bboxes)):
            scores = det_bboxes[example_idx][:, -1]
            boxes = det_bboxes[example_idx][:, :-1]
            labels = det_labels[example_idx]
            results.append(
                {
                    "scores": scores[scores > threshold],
                    "labels": labels[scores > threshold],
                    "boxes": boxes[scores > threshold],
                }
            )

        return results

    def post_process_instance_segmentation(
        self,
        object_detection_results,
        mask_pred,
        target_sizes=None,
        scale_factors=None,
    ):
        det_bboxes = [result["boxes"] for result in object_detection_results]
        det_labels = [result["labels"] for result in object_detection_results]

        # split batch mask prediction back to each image
        num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
        mask_preds = mask_pred.split(num_mask_roi_per_img, 0)

        num_imgs = len(object_detection_results)

        rescale = True  # we rescale by default
        scale_factors = [torch.from_numpy(scale_factor).to(det_bboxes[0].device) for scale_factor in scale_factors]
        _bboxes = [
            # det_bboxes[i][:, :4] * scale_factors[i] if rescale else det_bboxes[i][:, :4]
            det_bboxes[i] * scale_factors[i] if rescale else det_bboxes[i]
            for i in range(num_imgs)
        ]

        # apply mask post-processing to each image individually
        segm_results = []
        for i in range(num_imgs):
            if det_bboxes[i].shape[0] == 0:
                segm_results.append([[] for _ in range(self.num_classes)])
            else:
                segm_result = self.get_seg_masks(
                    mask_preds[i],
                    _bboxes[i],
                    det_labels[i],
                    self.test_cfg,
                    target_sizes[i],
                    scale_factors[i],
                    rescale=rescale,
                )
                segm_results.append(segm_result)

        return segm_results
