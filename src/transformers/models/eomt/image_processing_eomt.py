# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file ehidden_statescept in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either ehidden_statespress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for EoMT."""

import math
from typing import Dict, List, Optional, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    PaddingMode,
    pad,
    resize,
)
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    TensorType,
    filter_out_non_signature_kwargs,
    is_torch_available,
    logging,
)


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch
    import torch.nn.functional as F


class EoMTImageProcessor(BaseImageProcessor):
    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = 640,
        size_divisor: int = 32,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Union[float, List[float]] = None,
        image_std: Union[float, List[float]] = None,
        ignore_index: Optional[int] = None,
        do_reduce_labels: bool = False,
        num_labels: Optional[int] = None,
        do_pad=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        size = get_size_dict(size, default_to_square=True)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.ignore_index = ignore_index
        self.do_reduce_labels = do_reduce_labels
        self.num_labels = num_labels
        self.do_pad = do_pad

    def scale_image_size(self, image_size, segmentation_type="semantic"):
        target_h, target_w = self.size["height"], self.size["width"]
        orig_h, orig_w = image_size

        # For semantic segmentation: scale up so that both sides are ≥ target size
        if segmentation_type == "semantic":
            scale_factor = max(target_h / orig_h, target_w / orig_w)
        else:  # instance/panoptic: scale so that both sides are ≤ target
            scale_factor = min(target_h / orig_h, target_w / orig_w)

        output_h = round(orig_h * scale_factor)
        output_w = round(orig_w * scale_factor)

        return (output_h, output_w)

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format=None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> torch.tensor:
        image_size = get_image_size(
            image,
        )

        # How to pass panoptic value.
        output_size = self.scale_image_size(image_size, "semantic")

        image = resize(
            image=image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            return_numpy=True,
            **kwargs,
        )

        return image

    def _preprocessing_semantic_segmentation(self, image):
        crops, origins = [], []

        image_size = get_image_size(image=image)  # (H, W)
        crop_size = self.size["height"]  # or 'width', both are equal

        long_side = max(image_size)

        num_crops = math.ceil(long_side / crop_size)
        overlap = num_crops * crop_size - long_side
        overlap_per_crop = (overlap / (num_crops - 1)) if num_crops > 1 else 0

        for i in range(num_crops):
            start_idx = int(i * (crop_size - overlap_per_crop))
            end_idx = start_idx + crop_size

            if image_size[0] > image_size[1]:  # taller image
                crop = image[:, start_idx:end_idx, :]
            else:  # wider image
                crop = image[:, :, start_idx:end_idx]

            crops.append(crop)
            origins.append([0, start_idx, end_idx])

        return crops, origins

    def _preprocessing_instance_panoptic_segmentation(self, image):
        h, w = get_image_size(image)
        pad_h = max(0, self.size[0] - h)
        pad_w = max(0, self.size[1] - w)

        padding = ((0, pad_h), (0, pad_w))

        # channel axis is last, so no need to override data_format
        padded_image = pad(image=image, padding=padding, mode=PaddingMode.CONSTANT, constant_values=0.0)

        return padded_image

    filter_out_non_signature_kwargs()

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if do_resize:
            images = [
                self.resize(
                    image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format
                )
                for image in images
            ]

        crops_list, origins_list = [], []

        for image in images:
            crops, origins = self._preprocessing_semantic_segmentation(image)
            crops_list.append(crops)
            origins_list.append(origins)

        crops_list = np.stack(crops_list).squeeze(0)

        if do_rescale:
            images = [
                self.rescale(image, scale=rescale_factor, input_data_format=input_data_format) for image in crops_list
            ]

        if do_normalize:
            image_mean = np.array(image_mean).reshape(1, -1, 1, 1)
            image_std = np.array(image_std).reshape(1, -1, 1, 1)
            images = (images - image_mean) / image_std

        # # Normalize not working properly fix later
        # if do_normalize:
        #     images = [self.normalize(image, mean=image_mean, std=image_std, input_data_format=ChannelDimension.FIRST) for image in crops_list]

        origins_list = np.array(origins_list).squeeze(0)
        return images, origins_list

    def _revert_preprocessing_semantic(self, segmentation_logits, origins, original_image_sizes):
        logit_sums, logit_counts = [], []

        for image_size in original_image_sizes:
            height, width = self.scale_image_size(image_size)
            logit_sums.append(torch.zeros((segmentation_logits.shape[1], height, width)))
            logit_counts.append(torch.zeros((segmentation_logits.shape[1], height, width)))

        for crop_idx, (image_idx, start, end) in enumerate(origins):
            if original_image_sizes[image_idx][0] > original_image_sizes[image_idx][1]:  # Tall image
                logit_sums[image_idx][:, start:end, :] += segmentation_logits[crop_idx]
                logit_counts[image_idx][:, start:end, :] += 1
            else:  # Wide image
                logit_sums[image_idx][:, :, start:end] += segmentation_logits[crop_idx]
                logit_counts[image_idx][:, :, start:end] += 1

        output_logits = []

        for i, (sums, counts) in enumerate(zip(logit_sums, logit_counts)):
            combined = sums / counts.clamp(min=1)  # avoid division by zero
            combined = F.interpolate(combined[None, ...], size=original_image_sizes[i], mode="bilinear")[0]
            output_logits.append(combined)

        return output_logits

    def postprocess_semnatic_segmentation(self, outputs, origins, original_image_sizes):
        class_queries_logits = torch.tensor(outputs[1])  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = torch.tensor(outputs[0])  # [batch_size, num_queries, height, width]

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits,
            size=(640, 640),
            mode="bilinear",
        )

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation_logits = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

        output_logits = self._revert_preprocessing_semantic(segmentation_logits, origins, original_image_sizes)

        preds = output_logits[0].argmax(0).cpu().numpy()

        return preds


__all__ = ["EoMTImageProcessor"]
