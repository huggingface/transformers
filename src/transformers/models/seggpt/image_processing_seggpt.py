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

from typing import Union

import numpy as np

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, is_torch_available, is_torchvision_available, requires_backends


if is_torch_available():
    import torch

if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF


# See https://huggingface.co/papers/2212.02499 at 3.1 Redefining Output Spaces as "Images" - Semantic Segmentation
# Taken from https://github.com/Abdullah-Meda/Painter/blob/main/Painter/data/coco_semseg/gen_color_coco_panoptic_segm.py#L31
def build_palette(num_labels: int) -> list[tuple[int, int, int]]:
    base = int(num_labels ** (1 / 3)) + 1
    margin = 256 // base

    # class_idx 0 is the background which is mapped to black
    color_list = [(0, 0, 0)]
    for location in range(num_labels):
        num_seq_r = location // base**2
        num_seq_g = (location % base**2) // base
        num_seq_b = location % base

        R = 255 - num_seq_r * margin
        G = 255 - num_seq_g * margin
        B = 255 - num_seq_b * margin

        color_list.append((R, G, B))

    return color_list


class SegGptImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    num_labels (`int`, *optional*):
        Number of classes in the segmentation task (excluding the background). If specified, a palette will be
        built, assuming that class_idx 0 is the background, to map the prompt mask from a plain segmentation map
        to a 3-channel RGB image. Not specifying this will result in the prompt mask being duplicated across the
        channel dimension when `do_convert_rgb` is `True`.
    """

    num_labels: int


@auto_docstring
class SegGptImageProcessor(TorchvisionBackend):
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

        # Pass an empty list as sentinel when images is None; _preprocess_image_like_inputs handles it
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
        device: Union[str, "torch.device"] | None = None,
        num_labels: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        data = {}

        # Process regular images (do_convert_rgb=False: assume RGB, no mask conversion)
        # Check for the empty-list sentinel passed when images=None
        _images_provided = not (isinstance(images, list) and len(images) == 0)
        if _images_provided:
            prepared_images = self._prepare_image_like_inputs(
                images=images, do_convert_rgb=False, input_data_format=input_data_format, device=device
            )
            data["pixel_values"] = self._preprocess(prepared_images, **kwargs)

        # Process prompt images (same as regular images)
        if prompt_images is not None:
            prepared_prompt_images = self._prepare_image_like_inputs(
                images=prompt_images, do_convert_rgb=False, input_data_format=input_data_format, device=device
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
                    device=device,
                )
                palette = self.get_palette(num_labels) if num_labels is not None else None
                converted = []
                for mask_tensor in prepared_masks:
                    mask_np = mask_tensor.squeeze(0).numpy()
                    rgb_np = self.mask_to_rgb(mask_np, palette=palette)
                    converted.append(torch.from_numpy(rgb_np.astype(np.float32)))
                prepared_masks = converted
            else:
                # Already 3-channel RGB masks
                prepared_masks = self._prepare_image_like_inputs(
                    images=prompt_masks,
                    expected_ndims=3,
                    do_convert_rgb=False,
                    input_data_format=input_data_format,
                    device=device,
                )

            masks_kwargs = dict(kwargs)
            masks_kwargs["resample"] = PILImageResampling.NEAREST
            data["prompt_masks"] = self._preprocess(prepared_masks, **masks_kwargs)

        return BatchFeature(data=data, tensor_type=return_tensors)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        disable_grouping: bool | None,
        **kwargs,
    ) -> list["torch.Tensor"]:
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(stacked_images, size, resample)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        return reorder_images(processed_images_grouped, grouped_images_index)

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

        # batch_size x num_channels x 2*height x width
        masks = outputs.pred_masks

        # Predicted mask and prompt are concatenated in the height dimension
        # batch_size x num_channels x height x width
        masks = masks[:, :, masks.shape[2] // 2 :, :]

        # Unnormalize: permute to channel-last, apply std/mean, permute back
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
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0),
                    size=target_sizes[idx],
                    mode="nearest",
                )[0]

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


__all__ = ["SegGptImageProcessor"]
