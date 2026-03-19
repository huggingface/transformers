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

import torch

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    ImageInput,
    PILImageResampling,
    SizeDict,
    is_valid_image,
    make_list_of_images,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    is_torchvision_available,
    logging,
    requires_backends,
)


if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF


logger = logging.get_logger(__name__)


def make_list_of_list_of_images(
    images: list[list[ImageInput]] | list[ImageInput] | ImageInput,
) -> list[list[ImageInput]]:
    if is_valid_image(images):
        return [[images]]

    if isinstance(images, list) and all(isinstance(image, list) for image in images):
        return images

    if isinstance(images, list):
        return [make_list_of_images(image) for image in images]

    raise ValueError("images must be a list of list of images or a list of images or an image.")


class FuyuImagesKwargs(ImagesKwargs, total=False):
    r"""
    patch_size (`dict[str, int]`, *optional*, defaults to `{"height": 30, "width": 30}`):
        Dictionary in the format `{"height": int, "width": int}` specifying the size of the patches.
    padding_value (`float`, *optional*, defaults to 1.0):
        The value to pad the image with.
    padding_mode (`str`, *optional*, defaults to "constant"):
        The padding mode to use when padding the image.
    """

    patch_size: SizeDict | None
    padding_value: float
    padding_mode: str


class FuyuBatchFeature(BatchFeature):
    """
    BatchFeature class for Fuyu image processor and processor.

    The outputs dictionary from the processors contains a mix of tensors and lists of tensors.
    """

    def convert_to_tensors(self, tensor_type: str | TensorType | None = None, **kwargs):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                `None`, no modification is done.
        """
        if tensor_type is None:
            return self

        is_tensor, as_tensor = self._get_is_as_tensor_fns(tensor_type=tensor_type)

        def _convert_tensor(elem):
            if is_tensor(elem):
                return elem
            return as_tensor(elem)

        def _safe_convert_tensor(elem):
            try:
                return _convert_tensor(elem)
            except:  # noqa E722
                if key == "overflowing_values":
                    raise ValueError("Unable to create tensor returning overflowing values of different lengths. ")
                raise ValueError(
                    "Unable to create tensor, you should probably activate padding "
                    "with 'padding=True' to have batched tensors with the same length."
                )

        # Do the tensor conversion in batch
        for key, value in self.items():
            if isinstance(value, list) and isinstance(value[0], list):
                # list[list[Any]] -> list[list[Tensor]]
                self[key] = [[_safe_convert_tensor(elem) for elem in elems] for elems in value]
            elif isinstance(value, list):
                # list[Any] -> list[Tensor]
                self[key] = [_safe_convert_tensor(elem) for elem in value]
            else:
                # Any -> Tensor
                self[key] = _safe_convert_tensor(value)
        return self

    def to(self, *args, **kwargs) -> "BatchFeature":
        """
        Send all values to device by calling `v.to(*args, **kwargs)` (PyTorch only). This should support casting in
        different `dtypes` and sending the `BatchFeature` to a different `device`.

        Args:
            args (`Tuple`):
                Will be passed to the `to(...)` function of the tensors.
            kwargs (`Dict`, *optional*):
                Will be passed to the `to(...)` function of the tensors.

        Returns:
            [`BatchFeature`]: The same instance after modification.
        """
        requires_backends(self, ["torch"])
        import torch

        from ...utils import is_torch_device, is_torch_dtype

        new_data = {}
        device = kwargs.get("device")
        # Check if the args are a device or a dtype
        if device is None and len(args) > 0:
            # device should be always the first argument
            arg = args[0]
            if is_torch_dtype(arg):
                # The first argument is a dtype
                pass
            elif isinstance(arg, str) or is_torch_device(arg) or isinstance(arg, int):
                device = arg
            else:
                # it's something else
                raise ValueError(f"Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.")

        def _to(elem):
            # check if v is a floating point
            if torch.is_floating_point(elem):
                # cast and send to device
                return elem.to(*args, **kwargs)
            if device is not None:
                return elem.to(device=device)

            return elem

        # We cast only floating point tensors to avoid issues with tokenizers casting `LongTensor` to `FloatTensor`
        for k, v in self.items():
            if isinstance(v, list) and isinstance(v[0], list):
                # Data structure is a list of lists
                new_v = []
                for elems in v:
                    new_v.append([_to(elem) for elem in elems])
                new_data[k] = new_v
            elif isinstance(v, list):
                # Data structure is a list
                new_data[k] = [_to(elem) for elem in v]
            else:
                new_data[k] = _to(v)
        self.data = new_data
        return self


@auto_docstring
class FuyuImageProcessor(TorchvisionBackend):
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

    def _prepare_images_structure(
        self,
        images: ImageInput,
        expected_ndims: int = 3,
    ) -> ImageInput:
        images = self.fetch_images(images)
        return make_list_of_list_of_images(images)

    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = None,
        antialias: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Resize an image to fit within `(size.height, size.width)` while maintaining aspect ratio.
        Only resizes if the image is larger than the target size.
        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the max size of the output image.
            resample (`PILImageResampling | tvF.InterpolationMode | int`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resizing the image.
            antialias (`bool`, *optional*, defaults to `True`):
                Whether to apply antialiasing when resizing.
        """
        if resample is None:
            resample = PILImageResampling.BILINEAR
        image_height, image_width = image.shape[-2:]
        target_height, target_width = size.height, size.width
        # Only resize if image is larger than target
        if image_width <= target_width and image_height <= target_height:
            return image
        # Calculate optimal scale factor to fit within target size
        height_scale_factor = target_height / image_height
        width_scale_factor = target_width / image_width
        optimal_scale_factor = min(height_scale_factor, width_scale_factor)

        new_height = int(image_height * optimal_scale_factor)
        new_width = int(image_width * optimal_scale_factor)

        return super().resize(
            image, SizeDict(height=new_height, width=new_width), resample=resample, antialias=antialias
        )

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
        do_pad: bool | None,
        padding_value: float | None,
        padding_mode: str | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> FuyuBatchFeature:
        # Group images by size for batched resizing
        original_image_sizes = [batch_image[0].shape[-2:] for batch_image in images if batch_image]
        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping, is_nested=True
        )
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, resample=resample)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index, is_nested=True)

        image_sizes = [batch_image[0].shape[-2:] for batch_image in resized_images if batch_image]
        image_unpadded_heights = [[image_size[0]] for image_size in image_sizes]
        image_unpadded_widths = [[image_size[1]] for image_size in image_sizes]
        image_scale_factors = [
            [resized_size[0] / original_size[0]]
            for original_size, resized_size in zip(original_image_sizes, image_sizes)
        ]
        if do_pad:
            resized_images = self.pad(
                resized_images,
                pad_size=size,
                fill_value=padding_value,
                padding_mode=padding_mode,
                disable_grouping=disable_grouping,
                is_nested=True,
            )
        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(
            resized_images, disable_grouping=disable_grouping, is_nested=True
        )
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index, is_nested=True)

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

    def patchify_image(self, image: torch.Tensor, patch_size: SizeDict | None = None) -> torch.Tensor:
        """
        Convert an image into a tensor of patches using PyTorch's unfold operation.
        Args:
            image (`torch.Tensor`):
                Image to convert. Shape: [batch, channels, height, width]
            patch_size (`SizeDict`, *optional*):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the patches.
        """
        requires_backends(self, ["torch"])
        if patch_size is None:
            if isinstance(self.patch_size, SizeDict):
                patch_size = self.patch_size
            else:
                patch_size = SizeDict(**self.patch_size)
        patch_height, patch_width = patch_size.height, patch_size.width
        batch_size, channels, _, _ = image.shape
        # Use unfold to extract patches
        unfolded_along_height = image.unfold(2, patch_height, patch_height)
        patches = unfolded_along_height.unfold(3, patch_width, patch_width)
        patches = patches.contiguous()
        # Reshape to [batch, num_patches, channels * patch_h * patch_w]
        patches = patches.view(batch_size, channels, -1, patch_height, patch_width)
        patches = patches.permute(0, 2, 3, 4, 1)
        patches = patches.reshape(batch_size, -1, channels * patch_height * patch_width)
        return patches

    def preprocess_with_tokenizer_info(
        self,
        image_input: torch.Tensor,
        image_present: torch.Tensor,
        image_unpadded_h: torch.Tensor,
        image_unpadded_w: torch.Tensor,
        image_placeholder_id: int,
        image_newline_id: int,
        variable_sized: bool,
        patch_size: dict[str, int] | None = None,
    ) -> FuyuBatchFeature:
        """
        Process images for model input. In particular, variable-sized images are handled here.

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
                    # Patchify the image
                    patches = self.patchify_image(image=image.unsqueeze(0), patch_size=patch_size).squeeze(0)
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

    def _standardize_kwargs(
        self,
        patch_size: dict[str, int] | SizeDict | None = None,
        **kwargs,
    ) -> dict:
        """
        Process Fuyu-specific kwargs before validation.
        """
        kwargs = super()._standardize_kwargs(**kwargs)
        if patch_size is not None and not isinstance(patch_size, SizeDict):
            patch_size = SizeDict(**get_size_dict(patch_size, param_name="patch_size"))
        kwargs["patch_size"] = patch_size
        return kwargs


__all__ = ["FuyuImageProcessor"]
