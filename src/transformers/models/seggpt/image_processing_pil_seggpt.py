# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for SegGPT."""

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring, requires_backends
from ...utils.import_utils import requires
from .image_processing_seggpt import SegGptImageProcessorKwargs, build_palette


@auto_docstring
@requires(backends=("vision", "torch", "torchvision"))
class SegGptImageProcessorPil(PilBackend):
    valid_kwargs = SegGptImageProcessorKwargs

    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 448, "width": 448}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    num_labels = None

    def __init__(self, **kwargs: Unpack[SegGptImageProcessorKwargs]):
        super().__init__(**kwargs)

    def get_palette(self, num_labels: int) -> list[tuple[int, int, int]]:
        """Build a palette to map the prompt mask from a single channel to a 3-channel RGB.

        Args:
            num_labels (`int`):
                Number of classes in the segmentation task (excluding the background).

        Returns:
            `list[tuple[int, int, int]]`: Palette to map the prompt mask to RGB.
        """
        return build_palette(num_labels)

    def mask_to_rgb(self, mask: np.ndarray, palette: list[tuple[int, int, int]] | None = None) -> np.ndarray:
        """Converts a segmentation map to RGB format.

        Args:
            mask (`np.ndarray`):
                Segmentation map with shape `(height, width)` or `(1, height, width)` where pixel values
                represent the class index.
            palette (`list[tuple[int, int, int]]`, *optional*):
                Palette to use to convert the mask to RGB format. If unset, the mask is duplicated across
                the channel dimension.

        Returns:
            `np.ndarray`: The mask in RGB format with shape `(3, height, width)`.
        """
        if mask.ndim == 3:
            mask = mask.squeeze(0)

        height, width = mask.shape

        if palette is not None:
            rgb_mask = np.zeros((3, height, width), dtype=np.uint8)
            classes_in_mask = np.unique(mask)
            for class_idx in classes_in_mask:
                rgb_value = palette[class_idx]
                class_mask = (mask == class_idx).astype(np.uint8)
                class_rgb = np.array(rgb_value, dtype=np.uint8).reshape(3, 1, 1)
                rgb_mask += (np.expand_dims(class_mask, 0) * class_rgb).astype(np.uint8)
            rgb_mask = np.clip(rgb_mask, 0, 255).astype(np.uint8)
        else:
            rgb_mask = np.repeat(mask[np.newaxis, ...], 3, axis=0)

        return rgb_mask

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput | None = None,
        prompt_images: ImageInput | None = None,
        prompt_masks: ImageInput | None = None,
        **kwargs: Unpack[SegGptImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        prompt_images (`ImageInput`, *optional*):
            Prompt images to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255.
        prompt_masks (`ImageInput`, *optional*):
            Prompt masks to preprocess. Can be in the format of segmentation maps (no channels) or RGB images.
            If in the format of RGB images, `do_convert_rgb` should be set to `False`. If in the format of
            segmentation maps, specifying `num_labels` is recommended to build a palette to map the prompt mask
            from a single channel to a 3-channel RGB. If `num_labels` is not specified, the prompt mask will be
            duplicated across the channel dimension.
        """
        if all(v is None for v in [images, prompt_images, prompt_masks]):
            raise ValueError("At least one of images, prompt_images, prompt_masks must be specified.")

        _images_input = images if images is not None else []
        return super().preprocess(_images_input, prompt_images, prompt_masks, **kwargs)

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        prompt_images: ImageInput | None,
        prompt_masks: ImageInput | None,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        return_tensors: str | TensorType | None,
        num_labels: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        data = {}

        # Process regular images (do_convert_rgb=False: assume RGB, no mask conversion)
        # Check for the empty-list sentinel passed when images=None
        _images_provided = not (isinstance(images, list) and len(images) == 0)
        if _images_provided:
            prepared_images = self._prepare_image_like_inputs(
                images=images, do_convert_rgb=False, input_data_format=input_data_format
            )
            data["pixel_values"] = self._preprocess(prepared_images, **kwargs)

        # Process prompt images (same as regular images)
        if prompt_images is not None:
            prepared_prompt_images = self._prepare_image_like_inputs(
                images=prompt_images, do_convert_rgb=False, input_data_format=input_data_format
            )
            data["prompt_pixel_values"] = self._preprocess(prepared_prompt_images, **kwargs)

        # Process prompt masks with special handling
        if prompt_masks is not None:
            if do_convert_rgb:
                # 2D segmentation maps → convert to 3-channel RGB via palette
                prepared_masks = self._prepare_image_like_inputs(
                    images=prompt_masks,
                    expected_ndims=2,
                    do_convert_rgb=False,
                    input_data_format=ChannelDimension.FIRST,
                )
                palette = self.get_palette(num_labels) if num_labels is not None else None
                prepared_masks = [self.mask_to_rgb(mask, palette=palette) for mask in prepared_masks]
            else:
                # Already 3-channel RGB masks
                prepared_masks = self._prepare_image_like_inputs(
                    images=prompt_masks, expected_ndims=3, do_convert_rgb=False, input_data_format=input_data_format
                )

            masks_kwargs = dict(kwargs)
            masks_kwargs["resample"] = PILImageResampling.NEAREST
            data["prompt_masks"] = self._preprocess(prepared_masks, **masks_kwargs)

        return BatchFeature(data=data, tensor_type=return_tensors)

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        **kwargs,
    ) -> list[np.ndarray]:
        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image, size, resample)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        return processed_images

    def post_process_semantic_segmentation(
        self, outputs, target_sizes: list[tuple[int, int]] | None = None, num_labels: int | None = None
    ):
        """
        Converts the output of [`SegGptImageSegmentationOutput`] into segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`SegGptImageSegmentationOutput`]):
                Raw outputs of the model.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                List of length `batch_size`, where each item corresponds to the requested final size `(height, width)`
                of each prediction. If left to `None`, predictions will not be resized.
            num_labels (`int`, *optional*):
                Number of classes in the segmentation task (excluding the background). If specified, a palette will be
                built to map prediction masks from RGB values back to class indices. Should match the value used during
                preprocessing.

        Returns:
            `list[torch.Tensor]` of length `batch_size`, where each item is a semantic segmentation map of shape
            `(height, width)`. Each entry corresponds to a semantic class id.
        """

        requires_backends(self, ["torch"])

        masks = outputs.pred_masks
        masks = masks[:, :, masks.shape[2] // 2 :, :]

        std = torch.tensor(self.image_std).to(masks.device)
        mean = torch.tensor(self.image_mean).to(masks.device)
        masks = masks.permute(0, 2, 3, 1) * std + mean
        masks = masks.permute(0, 3, 1, 2)

        masks = torch.clip(masks * 255, 0, 255)

        semantic_segmentation = []
        palette_tensor = None
        palette = self.get_palette(num_labels) if num_labels is not None else None
        if palette is not None:
            palette_tensor = torch.tensor(palette).to(device=masks.device, dtype=torch.float)
            _, num_channels, _, _ = masks.shape
            palette_tensor = palette_tensor.view(1, 1, num_labels + 1, num_channels)

        for idx, mask in enumerate(masks):
            if target_sizes is not None:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=target_sizes[idx], mode="nearest")[0]

            if num_labels is not None:
                channels, height, width = mask.shape
                dist = mask.permute(1, 2, 0).view(height, width, 1, channels)
                dist = dist - palette_tensor
                dist = torch.pow(dist, 2)
                dist = torch.sum(dist, dim=-1)
                pred = dist.argmin(dim=-1)
            else:
                pred = mask.mean(dim=0).int()

            semantic_segmentation.append(pred)

        return semantic_segmentation


__all__ = ["SegGptImageProcessorPil"]
