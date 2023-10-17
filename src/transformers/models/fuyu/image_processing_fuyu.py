import math
import warnings
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...image_transforms import (
    PaddingMode,
    get_resize_output_image_size,
    pad,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_batched,
    is_scaled_image,
    to_numpy_array,
    valid_images,
)
from ...utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    TensorType,
    is_torch_available,
    is_torch_tensor,
    logging,
)

import numpy as np


import torch
import einops

from typing import List
import math

from torchvision.transforms import ConvertImageDtype, Normalize, Compose
import PIL.Image


class PaddedSampleTransform:
    def __init__(
        self,
        *,
        params_dtype: torch.dtype,
        image_size: Tuple[int, int],
        normalize_img_mean_std: Optional[List[float]] = None,
    ):
        height, width = image_size
        super().__init__(image_height=height, image_width=width, image_dtype=params_dtype)
        preprocess_transformations = [ConvertImageDtype(torch.float)]

        evaluation_transformations = []

        common_transformations = [
            AspectRatioPreservingScalingWithPad(width=width, height=height, fill=1.0, padding_mode="constant"),
        ]
        if normalize_img_mean_std:
            # split it into two lists of 3 floats each.
            assert len(normalize_img_mean_std) == 6, "Couldn't parse normalize_img_mean_std argument"
            img_mean, img_std = normalize_img_mean_std[:3], normalize_img_mean_std[3:]
            common_transformations.append(Normalize(img_mean, img_std))

        common_transformations.append(ConvertImageDtype(self.image_dtype))
        self.evaluation_transformations = (
            preprocess_transformations + evaluation_transformations + common_transformations
        )
        return Compose(evaluation_transformations)


class AspectRatioPreservingScalingWithPad:
    """
    A transformer to scale images while preserving their aspect ratio and then padding them 
    to a desired target size. Meant to closely emulate the image augmentation method used 
    in the Fuyu-8b model by Adept during inference.
    """

    def __init__(self, target_width: int, target_height: int, padding_value: float, padding_mode: str = "constant"):
        self.target_width = target_width
        self.target_height = target_height
        self.padding_value = padding_value
        self.padding_mode = padding_mode

    def _scale_to_target_aspect_ratio(self, image: np.ndarray) -> np.ndarray:
        image_height, image_width, _ = image.shape
        if image_width <= self.target_width and image_height <= self.target_height:
            return image

        height_scale_factor = self.target_height / image_height
        width_scale_factor = self.target_width / image_width
        optimal_scale_factor = min(height_scale_factor, width_scale_factor)

        new_height = int(image_height * optimal_scale_factor)
        new_width = int(image_width * optimal_scale_factor)

        scaled_image = resize(image=image, size=(new_width, new_height))
        return np.array(scaled_image)

    def _pad_to_target_size(self, image: np.ndarray) -> np.ndarray:
        image_height, image_width, _ = image.shape

        padding_top = 0
        padding_left = 0
        padding_bottom = self.target_height - image_height
        padding_right = self.target_width - image_width

        padded_image = pad(image, ((padding_top, padding_bottom),
                           (padding_left, padding_right)), mode=self.padding_mode, constant_values=self.padding_value)
        return padded_image

    def apply_transformation(self, image: Union[np.ndarray, PIL.Image.Image]) -> np.ndarray:
        if isinstance(image, PIL.Image.Image):
            image = to_numpy_array(image)
        scaled_image = self._scale_to_target_aspect_ratio(image)
        padded_image = self._pad_to_target_size(scaled_image)
        normalized_padded_image = normalize(padded_image)
        return normalized_padded_image


class FuyuImageProcessor(BaseImageProcessor):
    """
    This class should handle the image processing part before the main FuyuModel.
    In particular, it should handle:

    - Processing Images:
        Taking a batch of images as input. 
        If the images are variable-sized, it resizes them based on the desired patch dimensions. The image output is 
        always
        img_h ........................................... 1080                                                                                                                                       
        img_w ........................................... 1920  
        Then, it patches up these images using the patchify_image function.

    - Creating Image Input IDs:
        For each patch, a placeholder ID is given to identify where these patches belong in a token sequence.
        For variable-sized images, each line of patches is terminated with a newline ID.

    - Image Patch Indices:
        For each image patch, the code maintains an index where these patches should be inserted in a token stream.

    """
    # TODO make patchify logic consistent with FuyuViTModel

    def __init__(self, aspectratio_preserving_padding=AspectRatioPreservingScalingWithPad, **kwargs):
        super().__init__(**kwargs)
        self.aspectratio_preserving_padding = aspectratio_preserving_padding(
            target_height=1080, target_width=1920, padding_value=1.0)

    def get_num_patches(self, img_h: int, img_w: int, patch_dim_h: int, patch_dim_w: int) -> int:
        """Calculate number of patches required to encode an image."""
        assert img_h % patch_dim_h == 0, f"{img_h=} must be divisible by {patch_dim_h=}"
        assert img_w % patch_dim_w == 0, f"{img_w=} must be divisible by {patch_dim_w=}"

        num_patches_per_dim_h = img_h // patch_dim_h
        num_patches_per_dim_w = img_w // patch_dim_w
        num_patches = num_patches_per_dim_h * num_patches_per_dim_w

        return num_patches

    def patchify_image(self, image: torch.Tensor, patch_dim_h: int, patch_dim_w: int) -> torch.Tensor:
        """Convert an image into a tensor of patches.

        Args:
            image: Image to convert. Shape: [batch, channels, height, width]
            patch_dim_h: Height of each patch.
            patch_dim_w: Width of each patch.
        """
        rearranged_input = einops.rearrange(
            image,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=patch_dim_h,
            p2=patch_dim_w,
        )
        assert rearranged_input.shape[1] == self.get_num_patches(
            img_h=image.shape[2], img_w=image.shape[3], patch_dim_h=patch_dim_h, patch_dim_w=patch_dim_w
        )
        return rearranged_input

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.pad

    def process_images_for_model_input(
        self,
        image_input: torch.Tensor,
        image_present: torch.Tensor,
        image_unpadded_h: torch.Tensor,
        image_unpadded_w: torch.Tensor,
        image_patch_dim_h: int,
        image_patch_dim_w: int,
        image_placeholder_id: int,
        image_newline_id: int,
        variable_sized: bool,
    ) -> dict:
        """Process images for model input. In particular, variable-sized images are handled here.

        Args:
            image_input: [batch_size, num_sub_sequences, c, h, w] tensor of images padded to model input size.
            image_present: [batch_size, num_sub_sequences] tensor of 1s and 0s indicating whether an image is present.
            image_unpadded_h: [batch_size, num_sub_sequences] tensor of unpadded image heights.
            image_unpadded_w: [batch_size, num_sub_sequences] tensor of unpadded image widths.
            image_patch_dim_h: The height of the image patches.
            image_patch_dim_w: The width of the image patches.
            image_placeholder_id: The id of the image placeholder token.
            image_newline_id: The id of the image newline token.
            variable_sized: Whether to process images as variable-sized.
        """
        # Only images that are present.
        images: List[List[torch.Tensor]] = []
        image_patches: List[List[torch.Tensor]] = []
        # Image input ids for every subsequence, including ones with no image present.
        image_input_ids: List[List[torch.Tensor]] = []
        for bi in range(image_input.shape[0]):
            images.append([])
            image_input_ids.append([])
            image_patches.append([])
            for si in range(image_input.shape[1]):
                if image_present[bi, si]:
                    image = image_input[bi, si]
                    if variable_sized:
                        # The min() is required here due to floating point issues:
                        # math.ceil(torch.tensor(300).cuda() / 30) == 11
                        new_h = min(
                            image.shape[1], math.ceil(image_unpadded_h[bi, si] / image_patch_dim_h) * image_patch_dim_h
                        )
                        new_w = min(
                            image.shape[2], math.ceil(image_unpadded_w[bi, si] / image_patch_dim_w) * image_patch_dim_w
                        )
                        image = image[:, :new_h, :new_w]
                    images[bi].append(image)
                    num_patches = self.get_num_patches(
                        img_h=image.shape[1],
                        img_w=image.shape[2],
                        patch_dim_h=image_patch_dim_h,
                        patch_dim_w=image_patch_dim_w,
                    )
                    print(num_patches)
                    ids = torch.full([num_patches], image_placeholder_id, dtype=torch.int32, device=image_input.device)
                    patches = self.patchify_image(
                        image=image.unsqueeze(0), patch_dim_h=image_patch_dim_h, patch_dim_w=image_patch_dim_w
                    ).squeeze(0)
                    if variable_sized:
                        # Now terminate each line with |NEWLINE|.
                        ids = ids.reshape(-1, new_w // image_patch_dim_w)
                        ids = torch.cat(
                            [
                                ids,
                                torch.full(
                                    [ids.shape[0], 1], image_newline_id, dtype=torch.int32, device=image_input.device
                                ),
                            ],
                            dim=1,
                        )
                        ids = ids.reshape(-1)
                    image_input_ids[bi].append(ids)
                    image_patches[bi].append(patches)
                else:
                    image_input_ids[bi].append(torch.tensor([], dtype=torch.int32, device=image_input.device))

        # Create image_patch_input_indices, where non-negative values correspond to image patches to be inserted in
        # the stream.
        image_patch_indices_per_batch: List[List[torch.Tensor]] = []
        image_patch_indices_per_subsequence: List[List[torch.Tensor]] = []
        for bi in range(len(image_input_ids)):
            image_patch_indices_per_batch.append([])
            image_patch_indices_per_subsequence.append([])
            index_offset = 0
            for si in range(len(image_input_ids[bi])):
                # Indices of image patches.
                num_patches = torch.count_nonzero(image_input_ids[bi][si] == image_placeholder_id)
                indices = torch.arange(
                    num_patches,
                    dtype=image_input_ids[bi][si].dtype,
                    device=image_input_ids[bi][si].device,
                )

                # Place those indices in the image input ids token stream, with -1 representing non-index tokens.
                indices_in_stream_per_batch = torch.full_like(image_input_ids[bi][si], -1)
                indices_in_stream_per_subsequence = torch.full_like(image_input_ids[bi][si], -1)
                indices_in_stream_per_batch[
                    torch.nonzero(image_input_ids[bi][si] == image_placeholder_id, as_tuple=True)[0]
                ] = (indices + index_offset)
                indices_in_stream_per_subsequence[
                    torch.nonzero(image_input_ids[bi][si] == image_placeholder_id, as_tuple=True)[0]
                ] = indices

                image_patch_indices_per_batch[bi].append(indices_in_stream_per_batch)
                image_patch_indices_per_subsequence[bi].append(indices_in_stream_per_subsequence)
                index_offset += num_patches

        return {'images': images,
                'image_input_ids': image_input_ids,
                'image_patches': image_patches,
                'image_patch_indices_per_batch': image_patch_indices_per_batch,
                'image_patch_indices_per_subsequence': image_patch_indices_per_subsequence}
