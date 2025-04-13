# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fast Image processor class for Fuyu."""

from typing import Dict, List, Optional, Tuple, Union

# Correct imports for base classes and specific utils
from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BaseImageProcessorFast,
    group_images_by_shape,
)
from ...image_utils import (
    ImageInput,
    get_image_size,
    infer_channel_dimension_format,
    to_numpy_array,
)
from ...utils import (
    add_start_docstrings,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    logging,
    requires_backends,  # Added for patchify_image check
)


# Conditional imports for torch and torchvision
if is_torch_available():
    import torch

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import InterpolationMode
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as F

logger = logging.get_logger(__name__)


# Helper function - Correct
def make_list_of_list_of_images_fast(images: ImageInput) -> List[List[ImageInput]]:
    """Turn images into list of list of images if needed."""
    if isinstance(images, list) and len(images) > 0 and isinstance(images[0], list):
        return images
    elif isinstance(images, list):
        return [[img] for img in images]
    else:
        return [[images]]


# Class Definition - Correct
@add_start_docstrings(
    "Constructs a fast Fuyu image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class FuyuImageProcessorFast(BaseImageProcessorFast):
    do_resize: bool = True
    size: Dict[str, int] = {"height": 1080, "width": 1920}
    resample: InterpolationMode = InterpolationMode.BILINEAR
    do_pad: bool = True
    padding_value: float = 1.0
    padding_mode: str = "constant"
    do_normalize: bool = True
    image_mean: Union[float, List[float]] = 0.5
    image_std: Union[float, List[float]] = 0.5
    do_rescale: bool = True
    patch_size: Dict[str, int] = {"height": 30, "width": 30}
    do_convert_rgb: bool = False  # Explicitly False - Correct

    # model_input_names - OK for documentation
    model_input_names = [
        "images",
        "image_input_ids",
        "image_patches",
        "image_patch_indices_per_batch",
        "image_patch_indices_per_subsequence",
    ]

    # __init__ - Correct (simple super call)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # _get_tensor_image_size - Correct
    def _get_tensor_image_size(self, image: "torch.Tensor") -> Tuple[int, int]:
        """Helper to get (height, width) from a (C, H, W) or (B, C, H, W) tensor."""
        # ... (implementation as provided previously) ...
        if image.ndim == 3:  # C, H, W
            return image.shape[1], image.shape[2]
        elif image.ndim == 4:  # B, C, H, W
            return image.shape[2], image.shape[3]
        else:
            raise ValueError(f"Unsupported tensor dimension: {image.ndim}. Expected 3 or 4.")

    # resize - Correct implementation of Fuyu's specific logic
    def resize(
        self,
        image: "torch.Tensor",
        size: Dict[str, int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> "torch.Tensor":
        """Resize an image tensor (C, H, W) or (B, C, H, W) preserving aspect ratio, only scaling down."""
        # ... (implementation as provided previously) ...
        image_height, image_width = self._get_tensor_image_size(image)
        target_height, target_width = size["height"], size["width"]

        if image_width <= target_width and image_height <= target_height:
            return image

        height_scale_factor = target_height / image_height
        width_scale_factor = target_width / image_width
        optimal_scale_factor = min(height_scale_factor, width_scale_factor)

        new_height = int(image_height * optimal_scale_factor)
        new_width = int(image_width * optimal_scale_factor)

        scaled_image = F.resize(image, [new_height, new_width], interpolation=interpolation, antialias=True)
        return scaled_image

    # pad_image - Correct implementation of Fuyu's specific logic
    def pad_image(
        self,
        image: "torch.Tensor",
        size: Dict[str, int],
        mode: str = "constant",
        value: float = 1.0,  # This is the config value (e.g., 1.0)
    ) -> "torch.Tensor":
        image_height, image_width = self._get_tensor_image_size(image)
        target_height, target_width = size["height"], size["width"]

        padding_bottom = target_height - image_height
        padding_right = target_width - image_width

        if padding_bottom < 0 or padding_right < 0:
            raise ValueError("Image dimensions after resize are larger than target padding size.")

        padding_ltrb = [0, 0, padding_right, padding_bottom]

        if mode != "constant":
            logger.warning_once(f"Padding mode '{mode}' not directly supported, using 'constant'.")
            mode = "constant"

        # --- ADJUST PADDING VALUE ---
        # If the input tensor `image` is expected to be in [0, 1] range (due to F.to_tensor)
        # and the original config `value` was meant for [0, 255] range,
        # adjust the fill value used by F.pad.
        # We assume do_rescale=True is the standard path leading here.
        # This aligns the effective padding value *before* normalization.
        fill_value = value / 255.0 if self.do_rescale else value
        # --- END ADJUSTMENT ---

        padded_image = F.pad(image, padding=padding_ltrb, fill=fill_value, padding_mode=mode)  # Use fill_value
        return padded_image

    # _convert_and_prepare_images - Correct override needed for Fuyu
    def _convert_and_prepare_images(
        self,
        images: List[List[ImageInput]],
        do_convert_rgb: Optional[bool] = None,  # Fuyu ignores this (do_convert_rgb=False)
    ) -> Tuple[List[List["torch.Tensor"]], List[Tuple[int, int]]]:
        """
        Converts images to standardized 3-channel float tensors (3, H, W),
        stores original sizes. Expects List[List[ImageInput]].
        """
        if not isinstance(images, list) or not all(isinstance(i, list) for i in images):
            raise TypeError("Input must be a list of lists of images.")

        prepared_image_lists = []
        original_sizes = []

        for image_list in images:
            if len(image_list) != 1:
                raise ValueError("Multiple images per sample not yet supported.")
            if not image_list:
                raise ValueError("Received empty list of images for a sample.")

            img_input = image_list[0]
            is_tensor_input = isinstance(img_input, torch.Tensor)

            # --- Get Original Size ---
            try:
                if is_tensor_input:
                    np_image = img_input.cpu().numpy()
                else:
                    np_image = to_numpy_array(img_input)

                # Infer format only if ndim > 2
                input_df = None
                if np_image.ndim > 2:
                    try:
                        input_df = infer_channel_dimension_format(np_image)
                    except ValueError:
                        input_df = None
                        logger.warning_once(
                            f"Could not infer channel dimension format for input shape {np_image.shape}. Assuming channels first."
                        )

                orig_h, orig_w = get_image_size(np_image, channel_dim=input_df)
                original_sizes.append((orig_h, orig_w))
            except Exception as e:
                raise ValueError(f"Could not get original size of image. Error: {e}")

            # --- Convert to Tensor ---
            if not is_tensor_input:
                # Convert PIL/Numpy to tensor using F.to_tensor
                # F.to_tensor outputs CHW float [0,1] for uint8 HWC/PIL
                # Handles grayscale PIL/Numpy (H,W) -> (1,H,W) tensor
                tensor = F.to_tensor(img_input)
            else:
                tensor = img_input  # Already a tensor

            # --- Ensure 3 Channels (C, H, W) ---
            if tensor.ndim == 2:  # H, W -> 1, H, W -> 3, H, W
                tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
            elif tensor.ndim == 3:
                # Check if H, W, C
                if tensor.shape[2] in [1, 3] and tensor.shape[0] not in [1, 3]:
                    tensor = tensor.permute(2, 0, 1)  # H, W, C -> C, H, W
                # Now tensor is C, H, W. Check if C=1.
                if tensor.shape[0] == 1:
                    tensor = tensor.repeat(3, 1, 1)  # 1, H, W -> 3, H, W
                elif tensor.shape[0] != 3:
                    raise ValueError(f"Input tensor has unexpected channel dimension: {tensor.shape[0]}")
            elif tensor.ndim == 4:
                # Assume B, C, H, W or B, H, W, C
                if tensor.shape[3] in [1, 3] and tensor.shape[1] not in [1, 3]:
                    tensor = tensor.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W
                # Check if C=1
                if tensor.shape[1] == 1:
                    tensor = tensor.repeat(1, 3, 1, 1)  # B, 1, H, W -> B, 3, H, W
                elif tensor.shape[1] != 3:
                    raise ValueError(f"Input tensor has unexpected channel dimension: {tensor.shape[1]}")
                # Squeeze batch dim if B=1 and input was originally 4D tensor
                if is_tensor_input and img_input.ndim == 4 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)  # -> 3, H, W
            else:
                raise ValueError(f"Unsupported tensor dimension: {tensor.ndim}")

            # --- Ensure Float32 ---
            if tensor.dtype != torch.float32:
                tensor = tensor.to(torch.float32)

            # --- Final Check ---
            if tensor.ndim != 3 or tensor.shape[0] != 3:
                raise ValueError(f"Final tensor format is incorrect: {tensor.shape}")

            prepared_image_lists.append([tensor])  # Store 3, H, W tensor

        return prepared_image_lists, original_sizes

    # preprocess - Correct override needed to bridge custom _convert and _preprocess
    def preprocess(self, images: ImageInput, **kwargs) -> BatchFeature:
        """
        Preprocess an image or batch of images. Handles List[List] structure,
        custom Fuyu logic, and manual argument resolution due to override.

        Args:
            images (`ImageInput`):
                Image to preprocess. Accepts single images, lists, or lists of lists.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return.
            **kwargs:
                Additional keyword arguments passed to the processing steps, overriding
                class defaults (e.g., `do_resize`, `size`, `do_pad`, etc.).

        Returns:
            [`BatchFeature`]: A BatchFeature containing the processed images (`images`) and metadata
            (`image_unpadded_heights`, `image_unpadded_widths`, `image_scale_factors`).
        """
        # 1. Handle input structure
        images_list_list = make_list_of_list_of_images_fast(images)

        # 2. Manually resolve arguments by merging kwargs with class defaults
        # Get defaults from self attributes
        do_resize = kwargs.pop("do_resize", self.do_resize)
        size = kwargs.pop("size", self.size)
        resample = kwargs.pop("resample", self.resample)  # Uses InterpolationMode if available
        do_pad = kwargs.pop("do_pad", self.do_pad)
        padding_value = kwargs.pop("padding_value", self.padding_value)
        padding_mode = kwargs.pop("padding_mode", self.padding_mode)
        do_normalize = kwargs.pop("do_normalize", self.do_normalize)
        image_mean = kwargs.pop("image_mean", self.image_mean)
        image_std = kwargs.pop("image_std", self.image_std)
        do_rescale = kwargs.pop("do_rescale", self.do_rescale)
        # rescale_factor usually derived from base, but we need a default if not overridden
        rescale_factor = kwargs.pop("rescale_factor", getattr(self, "rescale_factor", 1.0 / 255.0))
        do_convert_rgb = kwargs.pop("do_convert_rgb", self.do_convert_rgb)
        return_tensors = kwargs.pop("return_tensors", None)

        # Warn about unused kwargs (optional, good practice)
        if kwargs:
            logger.warning(f"Unused kwargs: {list(kwargs.keys())}")

        # 3. Prepare images (validate, convert to tensor, handle RGB)
        # Manually call our overridden _convert_and_prepare_images
        # Pass resolved do_convert_rgb
        prepared_image_lists, original_sizes = self._convert_and_prepare_images(
            images=images_list_list,
            do_convert_rgb=do_convert_rgb,
            # do_rescale=False is hardcoded inside _convert_...
        )

        # 4. Store original sizes temporarily for _preprocess
        self._fuyu_preprocess_original_sizes = original_sizes

        # 5. Call the custom _preprocess logic, passing resolved arguments
        # Create dict of resolved args needed by _preprocess
        _preprocess_kwargs = {
            "do_resize": do_resize,
            "size": size,
            "interpolation": resample,  # Pass the resolved resample value
            "do_pad": do_pad,
            "padding_value": padding_value,
            "padding_mode": padding_mode,
            "do_normalize": do_normalize,
            "image_mean": image_mean,
            "image_std": image_std,
            "do_rescale": do_rescale,
            "rescale_factor": rescale_factor,
        }
        output = self._preprocess(prepared_image_lists, **_preprocess_kwargs)

        # 6. Clean up temporary attribute
        delattr(self, "_fuyu_preprocess_original_sizes")

        # 7. Format output (handle return_tensors)
        if return_tensors is not None:
            output = output.convert_to_tensors(return_tensors)

        return output

    # _preprocess - Correct scope and implementation for image transforms + metadata
    def _preprocess(
        self,
        images: List[List["torch.Tensor"]],  # Input List[List[Tensor]] from custom _convert
        do_resize: bool,
        size: Dict[str, int],
        interpolation: InterpolationMode,
        do_pad: bool,
        padding_value: float,
        padding_mode: str,
        do_normalize: bool,
        image_mean: Union[float, List[float]],
        image_std: Union[float, List[float]],
        do_rescale: bool,  # Note: This should be False when calling rescale_and_normalize later
        rescale_factor: float,
        **kwargs,
    ) -> BatchFeature:
        """
        Applies Fuyu-specific resize, pad, rescale, normalize.
        Returns BatchFeature with images and metadata.
        (Assumes rescale_and_normalize will be called with do_rescale=False)
        """
        processed_images_list = []
        image_unpadded_heights_list = []
        image_unpadded_widths_list = []
        image_scale_factors_list = []

        # Retrieve original sizes stored by the overridden preprocess method
        original_sizes = getattr(self, "_fuyu_preprocess_original_sizes", None)
        if original_sizes is None or len(original_sizes) != len(images):
            logger.warning(
                "Original sizes not found or mismatched, calculating scale factor based on input tensor size."
            )
            original_sizes = [self._get_tensor_image_size(img_list[0]) for img_list in images if img_list]

        target_height = size["height"]
        target_width = size["width"]

        # --- Step 1 & 2: Resize, Pad, and Collect Metadata (Loop) ---
        for idx, image_list in enumerate(images):
            # ... (resize, pad, metadata collection logic - unchanged) ...
            if len(image_list) != 1:
                raise ValueError("Multiple images per sample not yet supported.")
            if not image_list:
                logger.warning(f"Skipping empty image list at index {idx}")
                continue
            image = image_list[0]
            original_height, original_width = (
                original_sizes[idx] if idx < len(original_sizes) else self._get_tensor_image_size(image)
            )
            if do_resize:
                resized_image = self.resize(image, size=size, interpolation=interpolation)
            else:
                resized_image = image
            resized_height, resized_width = self._get_tensor_image_size(resized_image)
            scale_factor = (resized_height / original_height) if original_height > 0 else 1.0
            image_unpadded_heights_list.append([resized_height])
            image_unpadded_widths_list.append([resized_width])
            image_scale_factors_list.append([scale_factor])
            if do_pad:
                final_image = self.pad_image(resized_image, size=size, mode=padding_mode, value=padding_value)
            else:
                final_image = resized_image
            # Optional shape check after padding
            final_shape = final_image.shape
            expected_num_dims = 3  # Expect C, H, W
            if len(final_shape) != expected_num_dims:
                raise ValueError(f"Unexpected final_image dimension {len(final_shape)}")
            if do_pad and (final_shape[-2] != target_height or final_shape[-1] != target_width):
                raise RuntimeError(
                    f"Padding failed for image index {idx}. Shape was {final_shape}, expected H={target_height}, W={target_width}"
                )
            processed_images_list.append(final_image)

        # --- Step 3: Batch Rescale and Normalize ---
        if not processed_images_list:
            final_processed_images_list = []
        elif len(processed_images_list) == 1:
            # Simplified path for single image
            single_image = processed_images_list[0]
            if single_image.ndim != 3:
                raise ValueError(
                    f"Expected 3D tensor for single image, got {single_image.ndim}D shape {single_image.shape}"
                )
            image_batch = single_image.unsqueeze(0)
            processed_batch = self.rescale_and_normalize(
                image_batch,
                do_rescale=False,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
            )
            final_processed_images_list = [processed_batch.squeeze(0)]
        else:
            # Batch processing using grouping/reordering for B > 1
            input_shapes = [img.shape for img in processed_images_list]
            if not all(len(s) == 3 and s[0] == 3 for s in input_shapes):
                raise ValueError(
                    f"Incorrect shapes entering group_images_by_shape: {input_shapes}. Expected (3, H, W)."
                )

            # Assume group_images_by_shape keys on H, W and returns Dict[Shape, BatchTensor], Dict[int, Tuple[Shape, int]]
            grouped_images, grouped_images_index_dict = group_images_by_shape(processed_images_list)

            # Log structure if needed
            # logger.debug(f"grouped_images keys: {list(grouped_images.keys())}")
            # logger.debug(f"grouped_images_index_dict: {grouped_images_index_dict}")

            # Verify grouped_images keys are 2D shapes (H, W)
            if not all(isinstance(key, (torch.Size, tuple)) and len(key) == 2 for key in grouped_images.keys()):
                logger.warning(
                    f"Unexpected keys from group_images_by_shape: {list(grouped_images.keys())}. Expected (H, W)."
                )

            final_processed_batches = {}
            for (
                shape_key_hw,
                image_batch,
            ) in grouped_images.items():  # shape_key_hw is (H, W)
                processed_batch = self.rescale_and_normalize(
                    image_batch,
                    do_rescale=False,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                )
                final_processed_batches[shape_key_hw] = processed_batch  # Store with (H, W) key

            # --- Reorder manually using the dictionary index ---
            if grouped_images_index_dict:
                if not isinstance(grouped_images_index_dict, dict):
                    # If it's not a dict, the assumption based on the error is wrong.
                    raise TypeError(
                        f"grouped_images_index was expected to be a dict based on error, but got {type(grouped_images_index_dict)}"
                    )

                num_images = len(processed_images_list)  # Original number of images
                final_processed_images_list = [None] * num_images
                processed_count = 0

                # Iterate through the dictionary: original_idx -> (shape_hw, batch_idx)
                for original_idx, idx_info in grouped_images_index_dict.items():
                    if not isinstance(idx_info, (list, tuple)) or len(idx_info) != 2:
                        raise ValueError(
                            f"grouped_images_index_dict item {original_idx} has unexpected format: {idx_info}"
                        )

                    shape_key_hw, batch_idx = idx_info

                    # Check if shape key is valid and exists
                    if not isinstance(shape_key_hw, (torch.Size, tuple)) or len(shape_key_hw) != 2:
                        raise ValueError(
                            f"grouped_images_index_dict item {original_idx} has invalid shape key: {shape_key_hw}"
                        )
                    if shape_key_hw not in final_processed_batches:
                        raise KeyError(
                            f"Shape key {shape_key_hw} from index dict not found in processed batches. Available keys: {list(final_processed_batches.keys())}"
                        )

                    # Retrieve the correct processed tensor
                    try:
                        image_tensor = final_processed_batches[shape_key_hw][batch_idx]
                        final_processed_images_list[original_idx] = image_tensor
                        processed_count += 1
                    except IndexError:
                        raise IndexError(
                            f"Batch index {batch_idx} out of range for shape {shape_key_hw} in final_processed_batches."
                        )
                    except Exception as e:
                        raise RuntimeError(f"Error retrieving processed image for original index {original_idx}: {e}")

                # Final check after manual reordering
                if processed_count != num_images or None in final_processed_images_list:
                    raise RuntimeError(
                        f"Manual reordering failed. Processed {processed_count}/{num_images}. Result list: {final_processed_images_list}"
                    )

            else:
                logger.warning("grouped_images_index_dict was empty unexpectedly for batch size > 1.")
                final_processed_images_list = []

        # --- Step 4: Stack Final Images ---
        if len(final_processed_images_list) > 0:
            try:
                final_images_tensor = torch.stack(final_processed_images_list, dim=0)
            except RuntimeError as e:
                logger.error(f"Stacking failed. Images likely have different sizes. Error: {e}")
                shapes_in_list = [img.shape for img in final_processed_images_list]
                logger.error(f"Shapes attempted to stack: {shapes_in_list}")
                raise RuntimeError(
                    "Cannot stack images of different sizes. Enable padding (`do_pad=True`) "
                    "or ensure all inputs resize to the same dimensions."
                ) from e
        else:
            # Handle empty input case
            target_c = 3
            final_target_h = size.get("height", 0)
            final_target_w = size.get("width", 0)
            if final_target_h == 0 or final_target_w == 0:
                logger.warning("Target size not available for empty tensor creation, using 0x0.")
            final_images_tensor = torch.empty((0, target_c, final_target_h, final_target_w), dtype=torch.float32)

        # --- Step 5: Construct Output BatchFeature ---
        tensor_kwargs_int = {"dtype": torch.int64}
        tensor_kwargs_float = {"dtype": torch.float32}
        heights_tensor = (
            torch.tensor(image_unpadded_heights_list, **tensor_kwargs_int)
            if image_unpadded_heights_list
            else torch.empty((0, 1), **tensor_kwargs_int)
        )
        widths_tensor = (
            torch.tensor(image_unpadded_widths_list, **tensor_kwargs_int)
            if image_unpadded_widths_list
            else torch.empty((0, 1), **tensor_kwargs_int)
        )
        scale_factors_tensor = (
            torch.tensor(image_scale_factors_list, **tensor_kwargs_float)
            if image_scale_factors_list
            else torch.empty((0, 1), **tensor_kwargs_float)
        )

        data = {
            "images": final_images_tensor,
            "image_unpadded_heights": heights_tensor,
            "image_unpadded_widths": widths_tensor,
            "image_scale_factors": scale_factors_tensor,
        }
        return BatchFeature(data=data)

    # patchify_image - Correct logic, requires ValueError fix
    def patchify_image(self, image: "torch.Tensor", patch_size: Optional[Dict[str, int]] = None) -> "torch.Tensor":
        """Convert an image tensor into a tensor of patches using torch.unfold.
        Matches the slow processor's implementation exactly."""
        requires_backends(self, ["torch"])
        patch_size = patch_size if patch_size is not None else self.patch_size
        patch_height, patch_width = patch_size["height"], patch_size["width"]

        if image.ndim == 3:
            image = image.unsqueeze(0)
        elif image.ndim != 4:
            raise ValueError(f"Input image must have 3 or 4 dimensions, got {image.ndim}")

        batch_size, channels, height, width = image.shape

        if height % patch_height != 0 or width % patch_width != 0:
            raise ValueError(
                f"Image size ({height}, {width}) must be divisible by patch size ({patch_height}, {patch_width})."
            )

        # Match slow processor's unfolding and permutation exactly
        unfolded_along_height = image.unfold(2, patch_height, patch_height)
        patches = unfolded_along_height.unfold(3, patch_width, patch_width)
        patches = patches.contiguous()
        patches = patches.view(batch_size, channels, -1, patch_height, patch_width)
        patches = patches.permute(0, 2, 3, 4, 1)
        patches = patches.reshape(batch_size, -1, channels * patch_height * patch_width)

        if image.ndim == 3:
            patches = patches.squeeze(0)

        return patches

    # Removed __call__ override as preprocess now handles list conversion


__all__ = ["FuyuImageProcessorFast"]
