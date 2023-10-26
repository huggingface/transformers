# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
from typing import Dict, List, Optional, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
    pad,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    ChannelDimension,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, is_torch_available, is_vision_available, logging, requires_backends


if is_vision_available():
    pass

if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class FuyuImageProcessor(BaseImageProcessor):
    """
    This class should handle the image processing part before the main FuyuForCausalLM. In particular, it should
    handle:

    - Processing Images:
        Taking a batch of images as input. If the images are variable-sized, it resizes them based on the desired patch
        dimensions. The image output is always
        img_h ........................................... 1080
        img_w ........................................... 1920

        Then, it patches up these images using the patchify_image function.

    - Creating Image Input IDs:
        For each patch, a placeholder ID is given to identify where these patches belong in a token sequence. For
        variable-sized images, each line of patches is terminated with a newline ID.

    - Image Patch Indices:
        For each image patch, the code maintains an index where these patches should be inserted in a token stream.

    """

    model_input_names = [
        "images",
        "image_input_ids",
        "image_patches",
        "image_patch_indices_per_batch",
        "image_patch_indices_per_subsequence",
    ]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # FIXME check default value
        do_pad: bool = True,
        padding_value: float = 1.0,
        padding_mode: str = "constant",
        do_normalize: bool = True,
        image_mean: Union[float, List[float]] = 0.5,
        image_std: Union[float, List[float]] = 0.5,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        patch_size: Dict[str, int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size if size is not None else {"height": 1080, "width": 1920}
        self.resample = resample
        self.do_pad = do_pad
        self.padding_value = padding_value
        self.padding_mode = padding_mode
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.patch_size = patch_size if patch_size is not None else {"height": 30, "width": 30}

    def resize(
        self,
        image: np.ndarray,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        image_height, image_width = get_image_size(image)

        target_width = self.size["width"]
        target_height = self.size["height"]

        if image_width <= target_width and image_height <= target_height:
            return image

        height_scale_factor = target_height / image_height
        width_scale_factor = target_width / image_width
        optimal_scale_factor = min(height_scale_factor, width_scale_factor)

        new_height = int(image_height * optimal_scale_factor)
        new_width = int(image_width * optimal_scale_factor)

        scaled_image = resize(
            image=image, size=(new_height, new_width), data_format=data_format, input_data_format=input_data_format
        )
        return scaled_image

    def pad_image(
        self,
        image: np.ndarray,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        image_height, image_width, _ = image.shape

        target_width = self.size["width"]
        target_height = self.size["height"]

        padding_top = 0
        padding_left = 0
        padding_bottom = target_height - image_height
        padding_right = target_width - image_width
        padded_image = pad(
            image,
            padding=((padding_top, padding_bottom), (padding_left, padding_right)),
            mode=self.padding_mode,
            constant_values=self.padding_value,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        return padded_image

    def preprocess(
        self,
        images,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: Optional[PILImageResampling] = None,
        do_pad: Optional[bool] = None,
        padding_value: Optional[float] = None,
        padding_mode: Optional[str] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[float] = None,
        image_std: Optional[float] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        patch_size: Optional[Dict[str, int]] = None,
        data_format: Optional[Union[str, ChannelDimension]] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        return_tensors: Optional[TensorType] = None,
    ):
        """Utility function to preprocess the images and extract necessary information about original formats."""

        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        padding_value = padding_value if padding_value is not None else self.padding_value
        padding_mode = padding_mode if padding_mode is not None else self.padding_mode
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        patch_size = patch_size if patch_size is not None else self.patch_size

        images = make_list_of_images(images)

        batch_images = []
        image_unpadded_heights = []
        image_unpadded_widths = []
        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_resize and size is None:
            raise ValueError("Size must be specified if do_resize is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize and image_mean is None or image_std is None:
            raise ValueError("image_mean and image_std must be specified if do_normalize is True.")

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        image_sizes = [get_image_size(image, channel_dim=input_data_format) for image in images]
        image_unpadded_heights = [[image_size[0]] for image_size in image_sizes]
        image_unpadded_widths = [[image_size[1]] for image_size in image_sizes]

        if do_resize:
            images = [self.resize(image) for image in images]

        if do_pad:
            images = [self.pad_image(image) for image in images]

        if do_rescale:
            images = [self.rescale(image, rescale_factor) for image in images]

        if do_normalize:
            images = [self.normalize(image, image_mean, image_std) for image in images]

        if data_format is not None:
            images = [to_channel_dimension_format(image, data_format, input_data_format) for image in images]

        batch_images = [[torch.Tensor(image)] for image in images]

        return batch_images, torch.Tensor(image_unpadded_heights), torch.Tensor(image_unpadded_widths)

    def get_num_patches(self, image_height: int, image_width: int) -> int:
        """Calculate number of patches required to encode an image."""
        patch_height, patch_width = self.patch_size["height"], self.patch_size["width"]

        if image_height % patch_height != 0:
            raise ValueError(f"{image_height=} must be divisible by {patch_height}")
        if image_width % patch_width != 0:
            raise ValueError(f"{image_width=} must be divisible by {patch_width}")

        num_patches_per_dim_h = image_height // patch_height
        num_patches_per_dim_w = image_width // patch_width
        num_patches = num_patches_per_dim_h * num_patches_per_dim_w
        return num_patches

    def patchify_image(self, image: "torch.Tensor") -> "torch.Tensor":
        """
        Convert an image into a tensor of patches.

        Args:
            image: Image to convert. Shape: [batch, channels, height, width]
        """
        requires_backends(self, ["torch"])

        # TODO refer to https://github.com/ArthurZucker/transformers/blob/0f0a3fe5ca5697ee58faeb5b53f049af720b5e98/src/transformers/models/vit_mae/modeling_vit_mae.py#L871
        # torch implementation is faster but does not handle non-squares

        batch_size, channels, _, _ = image.shape
        unfolded_along_height = image.unfold(2, self.patch_size["height"], self.patch_size["height"])
        patches = unfolded_along_height.unfold(3, self.patch_size["width"], self.patch_size["width"])

        patches_reshaped = patches.contiguous().view(
            batch_size, channels, -1, self.patch_size["height"], self.patch_size["width"]
        )

        patches_final = patches_reshaped.permute(0, 2, 3, 4, 1).reshape(
            batch_size, -1, channels * self.patch_size["height"] * self.patch_size["width"]
        )

        patches_reshaped = patches.contiguous().view(batch_size, channels, -1, self.patch_size["height"], self.patch_size["width"])
        patches_final = patches_reshaped.permute(0, 2, 3, 4, 1)
        patches_final = patches_final.reshape(batch_size, -1, channels * self.patch_size["height"] * self.patch_size["width"])
        return patches_final

    def postprocess_with_tokenizer_info(
        self,
        image_input: "torch.Tensor",
        image_present: "torch.Tensor",
        image_unpadded_h: "torch.Tensor",
        image_unpadded_w: "torch.Tensor",
        image_placeholder_id: int,
        image_newline_id: int,
        variable_sized: bool,
        patch_size: Dict[str, int] = None,
    ) -> dict:
        """Process images for model input. In particular, variable-sized images are handled here.

        Args:
            image_input: [batch_size, subsequence_size, num_channels, height, width] tensor of images padded to model input size.
            image_present: [batch_size, subsequence_size] tensor of 1s and 0s indicating whether an image is present.
            image_unpadded_h: [batch_size, subsequence_size] tensor of unpadded image heights.
            image_unpadded_w: [batch_size, subsequence_size] tensor of unpadded image widths.
            image_placeholder_id: The id of the image placeholder token. Comes from an associated tokenizer.
            image_newline_id: The id of the image newline token. Comes from an associated tokenizer.
            variable_sized: Whether to process images as variable-sized.
        """
        requires_backends(self, ["torch"])

        patch_size = patch_size if patch_size is not None else self.patch_size

        # Only images that are present.
        images: List[List[torch.Tensor]] = []
        image_patches: List[List[torch.Tensor]] = []
        # Image input ids for every subsequence, including ones with no image present.
        image_input_ids: List[List[torch.Tensor]] = []
        for batch_index in range(image_input.shape[0]):
            images.append([])
            image_input_ids.append([])
            image_patches.append([])
            for subseq_index in range(image_input.shape[1]):
                if image_present[batch_index, subseq_index]:
                    image = image_input[batch_index, subseq_index]
                    if variable_sized:
                        # The min() is required here due to floating point issues:
                        # math.ceil(torch.tensor(300).cuda() / 30) == 11
                        new_h = min(
                            image.shape[1],
                            math.ceil(image_unpadded_h[batch_index, subseq_index] / self.patch_size["height"])
                            * self.patch_size["height"],
                        )
                        new_w = min(
                            image.shape[2],
                            math.ceil(image_unpadded_w[batch_index, subseq_index] / self.patch_size["width"])
                            * self.patch_size["width"],
                        )
                        image = image[:, :new_h, :new_w]

                    images[batch_index].append(image)
                    num_patches = self.get_num_patches(image_height=image.shape[1], image_width=image.shape[2])
                    tensor_of_image_ids = torch.full(
                        [num_patches], image_placeholder_id, dtype=torch.int32, device=image_input.device
                    )
                    tensor_of_image_ids = torch.full(
                        [num_patches], image_placeholder_id, dtype=torch.int32, device=image_input.device
                    )
                    patches = self.patchify_image(image=image.unsqueeze(0)).squeeze(0)
                    if variable_sized:
                        # Now terminate each line with |NEWLINE|.
                        tensor_of_image_ids = tensor_of_image_ids.reshape(-1, new_w // patch_size["width"])
                        tensor_of_image_ids = torch.cat(
                            [
                                tensor_of_image_ids,
                                torch.full(
                                    [tensor_of_image_ids.shape[0], 1],
                                    image_newline_id,
                                    dtype=torch.int32,
                                    device=image_input.device,
                                ),
                            ],
                            dim=1,
                        )
                        tensor_of_image_ids = tensor_of_image_ids.reshape(-1)
                    image_input_ids[batch_index].append(tensor_of_image_ids)
                    image_patches[batch_index].append(patches)
                else:
                    image_input_ids[batch_index].append(torch.tensor([], dtype=torch.int32, device=image_input.device))

        # Create image_patch_input_indices, where non-negative values correspond to image patches to be inserted in
        # the stream.
        image_patch_indices_per_batch: List[List[torch.Tensor]] = []
        image_patch_indices_per_subsequence: List[List[torch.Tensor]] = []

        for batch_index in range(len(image_input_ids)):
            image_patch_indices_per_batch.append([])
            image_patch_indices_per_subsequence.append([])
            index_offset = 0
            for subseq_index in range(len(image_input_ids[batch_index])):
                # Indices of image patches.
                num_patches = torch.count_nonzero(image_input_ids[batch_index][subseq_index] == image_placeholder_id)
                indices = torch.arange(
                    num_patches,
                    dtype=image_input_ids[batch_index][subseq_index].dtype,
                    device=image_input_ids[batch_index][subseq_index].device,
                )

                # Place those indices in the image input ids token stream, with -1 representing non-index tokens.
                indices_in_stream_per_batch = torch.full_like(image_input_ids[batch_index][subseq_index], -1)
                indices_in_stream_per_subsequence = torch.full_like(image_input_ids[batch_index][subseq_index], -1)
                indices_in_stream_per_batch[
                    torch.nonzero(image_input_ids[batch_index][subseq_index] == image_placeholder_id, as_tuple=True)[0]
                ] = (indices + index_offset)
                indices_in_stream_per_subsequence[
                    torch.nonzero(image_input_ids[batch_index][subseq_index] == image_placeholder_id, as_tuple=True)[0]
                ] = indices

                image_patch_indices_per_batch[batch_index].append(indices_in_stream_per_batch)
                image_patch_indices_per_subsequence[batch_index].append(indices_in_stream_per_subsequence)
                index_offset += num_patches

        return {
            "images": images,
            "image_input_ids": image_input_ids,
            "image_patches": image_patches,
            "image_patch_indices_per_batch": image_patch_indices_per_batch,
            "image_patch_indices_per_subsequence": image_patch_indices_per_subsequence,
        }
