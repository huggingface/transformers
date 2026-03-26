# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for Fuyu."""

import math

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_backends import PilBackend
from ...image_processing_utils import get_size_dict
from ...image_utils import ImageInput, PILImageResampling, SizeDict, get_image_size
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring, requires_backends
from ...utils.import_utils import requires
from .image_processing_fuyu import FuyuBatchFeature, FuyuImagesKwargs, make_list_of_list_of_images


@auto_docstring
@requires(backends=("vision", "torch", "torchvision"))
class FuyuImageProcessorPil(PilBackend):
    do_resize = True
    size = {"height": 1080, "width": 1920}
    patch_size = {"height": 30, "width": 30}
    resample = PILImageResampling.BILINEAR
    do_pad = True
    padding_value = 1.0
    padding_mode = "constant"
    do_normalize = True
    image_mean = 0.5
    image_std = 0.5
    do_rescale = True
    rescale_factor = 1 / 255
    model_input_names = [
        "images",
        "image_input_ids",
        "image_patches",
        "image_patch_indices_per_batch",
        "image_patch_indices_per_subsequence",
    ]
    valid_kwargs = FuyuImagesKwargs

    def __init__(self, **kwargs: Unpack[FuyuImagesKwargs]):
        super().__init__(**kwargs)

    def _prepare_images_structure(self, images: ImageInput, expected_ndims: int = 3) -> ImageInput:
        images = self.fetch_images(images)
        return make_list_of_list_of_images(images)

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to fit within `(size.height, size.width)` while maintaining aspect ratio.
        Only resizes if the image is larger than the target size.
        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the max size of the output image.
            resample (`PILImageResampling | tvF.InterpolationMode | int`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resizing the image.
        """

        input_height, input_width = image.shape[-2:]
        target_height, target_width = size.height, size.width
        # Only resize if image is larger than target
        if input_width <= target_width and input_height <= target_height:
            return image
        # Calculate optimal scale factor to fit within target size
        height_scale_factor = target_height / input_height
        width_scale_factor = target_width / input_width
        optimal_scale_factor = min(height_scale_factor, width_scale_factor)

        new_height = int(input_height * optimal_scale_factor)
        new_width = int(input_width * optimal_scale_factor)

        return super().resize(image, SizeDict(height=new_height, width=new_width), resample=resample)

    def _preprocess(
        self,
        images: list[list[np.ndarray]],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        padding_value: float | None,
        padding_mode: str | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> FuyuBatchFeature:
        # Process nested images one by one
        original_image_sizes = []
        processed_images = []
        for batch_images in images:
            if batch_images:
                original_image_sizes.append(batch_images[0].shape[-2:])
                processed_batch = []
                for image in batch_images:
                    if do_resize:
                        image = self.resize(image=image, size=size, resample=resample)
                    processed_batch.append(image)
                processed_images.append(processed_batch)
            else:
                processed_images.append([])

        image_sizes = [batch_image[0].shape[-2:] for batch_image in processed_images if batch_image]
        image_unpadded_heights = [[image_size[0]] for image_size in image_sizes]
        image_unpadded_widths = [[image_size[1]] for image_size in image_sizes]
        image_scale_factors = [
            [resized_size[0] / original_size[0]]
            for original_size, resized_size in zip(original_image_sizes, image_sizes)
        ]

        if do_pad:
            # Handle nested padding manually since PIL backend doesn't support is_nested
            target_height, target_width = size.height, size.width
            for batch_idx, batch_images in enumerate(processed_images):
                for img_idx, image in enumerate(batch_images):
                    from ...image_utils import ChannelDimension

                    height, width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
                    padding_height = target_height - height
                    padding_width = target_width - width
                    if padding_height > 0 or padding_width > 0:
                        pad_width = ((0, 0), (0, padding_height), (0, padding_width))
                        if padding_mode == "constant":
                            image = np.pad(image, pad_width, mode="constant", constant_values=padding_value)
                        else:
                            image = np.pad(image, pad_width, mode=padding_mode)
                        processed_images[batch_idx][img_idx] = image

        # Process rescale and normalize one by one
        for batch_idx, batch_images in enumerate(processed_images):
            for img_idx, image in enumerate(batch_images):
                if do_rescale:
                    image = self.rescale(image, rescale_factor)
                if do_normalize:
                    image = self.normalize(image, image_mean, image_std)
                processed_images[batch_idx][img_idx] = image

        return FuyuBatchFeature(
            data={
                "images": processed_images,
                "image_unpadded_heights": image_unpadded_heights,
                "image_unpadded_widths": image_unpadded_widths,
                "image_scale_factors": image_scale_factors,
            },
            tensor_type=return_tensors,
        )

    def get_num_patches(self, image_height: int, image_width: int, patch_size: SizeDict | None = None) -> int:
        """
        Calculate number of patches required to encode an image.
        Args:
            image_height (`int`):
                Height of the image.
            image_width (`int`):
                Width of the image.
            patch_size (`SizeDict`, *optional*):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the patches.
        """
        if patch_size is None:
            if isinstance(self.patch_size, SizeDict):
                patch_size = self.patch_size
            else:
                patch_size = SizeDict(**self.patch_size)
        patch_height, patch_width = patch_size.height, patch_size.width
        if image_height % patch_height != 0:
            raise ValueError(f"{image_height=} must be divisible by {patch_height}")
        if image_width % patch_width != 0:
            raise ValueError(f"{image_width=} must be divisible by {patch_width}")
        num_patches_per_dim_h = image_height // patch_height
        num_patches_per_dim_w = image_width // patch_width
        num_patches = num_patches_per_dim_h * num_patches_per_dim_w
        return num_patches

    def patchify_image(
        self, image: "np.ndarray | torch.Tensor", patch_size: SizeDict | None = None
    ) -> "np.ndarray | torch.Tensor":
        """
        Convert an image into a tensor of patches using numpy operations.
        Args:
            image (`np.ndarray` or `torch.Tensor`):
                Image to convert. Shape: [batch, channels, height, width] or [channels, height, width]
            patch_size (`SizeDict`, *optional*):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the patches.
        """
        requires_backends(self, ["torch"])
        import torch

        if patch_size is None:
            if isinstance(self.patch_size, SizeDict):
                patch_size = self.patch_size
            else:
                patch_size = SizeDict(**self.patch_size)
        patch_height, patch_width = patch_size.height, patch_size.width

        # Handle torch tensors by converting to numpy
        is_torch = isinstance(image, torch.Tensor)
        if is_torch:
            image_np = image.cpu().numpy()
            device = image.device
        else:
            image_np = image
            device = None

        # Handle batch dimension
        if len(image_np.shape) == 4:
            batch_size, channels, height, width = image_np.shape
        elif len(image_np.shape) == 3:
            batch_size = 1
            channels, height, width = image_np.shape
            image_np = image_np[np.newaxis, ...]
        else:
            raise ValueError(
                f"Expected image shape [batch, channels, height, width] or [channels, height, width], got {image_np.shape}"
            )

        # Extract patches using numpy operations to match torch unfold behavior exactly
        # Torch: unfold(2) -> unfold(3) -> view(b, c, -1, h, w) -> permute(0,2,3,4,1) -> reshape(b, -1, c*h*w)
        num_patches_h = height // patch_height
        num_patches_w = width // patch_width
        num_patches = num_patches_h * num_patches_w

        patches_list = []
        for b in range(batch_size):
            # Simulate torch unfold: extract patches along height, then width
            # After unfold(2) and unfold(3), shape is (channels, num_patches_h, patch_height, num_patches_w, patch_width)
            # After view: (channels, num_patches, patch_height, patch_width) where num_patches = num_patches_h * num_patches_w
            # After permute(0,2,3,4,1): (num_patches, patch_height, patch_width, channels)
            # After reshape: (num_patches, channels * patch_height * patch_width)

            # Reshape to extract patches: (channels, num_patches_h, patch_height, num_patches_w, patch_width)
            img_reshaped = image_np[b].reshape(channels, num_patches_h, patch_height, num_patches_w, patch_width)
            # Transpose to (channels, num_patches, patch_height, patch_width) where num_patches = num_patches_h * num_patches_w
            img_reshaped = img_reshaped.transpose(0, 1, 3, 2, 4).reshape(
                channels, num_patches, patch_height, patch_width
            )
            # Permute to (num_patches, patch_height, patch_width, channels) - matching torch permute(0,2,3,4,1)
            img_permuted = img_reshaped.transpose(1, 2, 3, 0)
            # Flatten to (num_patches, channels * patch_height * patch_width)
            patches = img_permuted.reshape(num_patches, channels * patch_height * patch_width)
            patches_list.append(patches)

        patches_array = np.stack(patches_list, axis=0) if batch_size > 1 else patches_list[0]

        # Convert back to torch if input was torch
        if is_torch:
            patches_array = torch.from_numpy(patches_array).to(device)

        return patches_array

    def preprocess_with_tokenizer_info(
        self,
        image_input: "torch.Tensor",
        image_present: "torch.Tensor",
        image_unpadded_h: "torch.Tensor",
        image_unpadded_w: "torch.Tensor",
        image_placeholder_id: int,
        image_newline_id: int,
        variable_sized: bool,
        patch_size: dict[str, int] | None = None,
    ) -> FuyuBatchFeature:
        """
        Process images for model input. In particular, variable-sized images are handled here.
        This method uses PyTorch operations as it operates on model inputs which are tensors.

        Args:
            image_input (`torch.Tensor` of shape [batch_size, subsequence_size, num_channels, height, width]):
                Tensor of images padded to model input size.
            image_present (`torch.Tensor` of shape [batch_size, subsequence_size, num_images]):
                Tensor of 1s and 0s indicating whether an image is present.
            image_unpadded_h (`torch.Tensor` of shape [batch_size, subsequence_size]):
                Tensor of unpadded image heights.
            image_unpadded_w (`torch.Tensor` of shape [batch_size, subsequence_size]):
                Tensor of unpadded image widths.
            image_placeholder_id (int):
                The id of the image placeholder token. Comes from an associated tokenizer.
            image_newline_id (int):
                The id of the image newline token. Comes from an associated tokenizer.
            variable_sized (bool):
                Whether to process images as variable-sized.
            patch_size (`dict[str, int]`, *optional*):
                Size of the patches.
        """
        requires_backends(self, ["torch"])
        import torch

        if patch_size is None:
            if isinstance(self.patch_size, SizeDict):
                patch_size = self.patch_size
            else:
                patch_size = SizeDict(**self.patch_size)
        elif not isinstance(patch_size, SizeDict):
            patch_size = SizeDict(**patch_size)
        patch_height, patch_width = patch_size.height, patch_size.width
        # Only images that are present
        images: list[list[torch.Tensor]] = []
        batch_image_patches: list[list[torch.Tensor]] = []
        # Image input ids for every subsequence, including ones with no image present
        batch_image_input_ids: list[list[torch.Tensor]] = []
        for batch_index in range(image_input.shape[0]):
            image_input_ids = []
            image_patches = []
            for subseq_index in range(image_input.shape[1]):
                if image_present[batch_index, subseq_index]:
                    image = image_input[batch_index, subseq_index]
                    image_height, image_width = image.shape[1], image.shape[2]
                    if variable_sized:
                        # Calculate new dimensions based on unpadded size
                        # The min() is required here due to floating point issues
                        new_h = min(
                            image_height,
                            math.ceil(image_unpadded_h[batch_index, subseq_index] / patch_height) * patch_height,
                        )
                        new_w = min(
                            image_width,
                            math.ceil(image_unpadded_w[batch_index, subseq_index] / patch_width) * patch_width,
                        )
                        image = image[:, :new_h, :new_w]
                        image_height, image_width = new_h, new_w
                    num_patches = self.get_num_patches(
                        image_height=image_height, image_width=image_width, patch_size=patch_size
                    )
                    # Create tensor of placeholder IDs
                    tensor_of_image_ids = torch.full(
                        [num_patches], image_placeholder_id, dtype=torch.int32, device=image_input.device
                    )
                    # Patchify the image - convert to numpy, patchify, convert back
                    image_np = image.cpu().numpy()
                    patches_np = self.patchify_image(image_np, patch_size=patch_size)
                    patches = torch.from_numpy(patches_np).to(image_input.device)
                    assert num_patches == patches.shape[0]
                    if variable_sized:
                        # Terminate each line with newline ID
                        tensor_of_image_ids = tensor_of_image_ids.reshape(-1, image_width // patch_width)
                        newline_ids = torch.full(
                            [tensor_of_image_ids.shape[0], 1],
                            image_newline_id,
                            dtype=torch.int32,
                            device=image_input.device,
                        )
                        tensor_of_image_ids = torch.cat([tensor_of_image_ids, newline_ids], dim=1)
                        tensor_of_image_ids = tensor_of_image_ids.reshape(-1)
                    images.append([image])
                    image_input_ids.append(tensor_of_image_ids)
                    image_patches.append(patches)
                else:
                    image_input_ids.append(torch.tensor([], dtype=torch.int32, device=image_input.device))
            batch_image_input_ids.append(image_input_ids)
            batch_image_patches.append(image_patches)
        # Create image patch indices
        image_patch_indices_per_batch: list[list[torch.Tensor]] = []
        image_patch_indices_per_subsequence: list[list[torch.Tensor]] = []

        for sample_image_input_ids in batch_image_input_ids:
            index_offset = 0
            per_batch_indices = []
            per_subsequence_indices = []
            for subseq_image_input_ids in sample_image_input_ids:
                # Indices of image patches
                patches_mask = subseq_image_input_ids == image_placeholder_id
                num_patches = torch.count_nonzero(patches_mask)
                indices = torch.arange(num_patches, dtype=torch.int64, device=subseq_image_input_ids.device).type_as(
                    subseq_image_input_ids
                )
                # Place those indices in the image input ids token stream, with -1 representing non-index tokens
                indices_in_stream_per_batch = torch.full_like(subseq_image_input_ids, -1)
                indices_in_stream_per_subsequence = torch.full_like(subseq_image_input_ids, -1)
                patches_inds = torch.nonzero(patches_mask, as_tuple=True)[0]

                indices_in_stream_per_batch[patches_inds] = indices + index_offset
                indices_in_stream_per_subsequence[patches_inds] = indices

                per_batch_indices.append(indices_in_stream_per_batch)
                per_subsequence_indices.append(indices_in_stream_per_subsequence)
                index_offset += num_patches

            image_patch_indices_per_batch.append(per_batch_indices)
            image_patch_indices_per_subsequence.append(per_subsequence_indices)
        return FuyuBatchFeature(
            data={
                "images": images,
                "image_input_ids": batch_image_input_ids,
                "image_patches": batch_image_patches,
                "image_patch_indices_per_batch": image_patch_indices_per_batch,
                "image_patch_indices_per_subsequence": image_patch_indices_per_subsequence,
            }
        )

    def _standardize_kwargs(self, patch_size: dict[str, int] | SizeDict | None = None, **kwargs) -> dict:
        """
        Process Fuyu-specific kwargs before validation.
        """
        kwargs = super()._standardize_kwargs(**kwargs)
        if patch_size is not None and not isinstance(patch_size, SizeDict):
            patch_size = SizeDict(**get_size_dict(patch_size, param_name="patch_size"))
        kwargs["patch_size"] = patch_size
        return kwargs


__all__ = ["FuyuImageProcessorPil"]
