# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""PyTorch Molmo2 model."""

import math
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
    make_nested_list_of_images,
)
from ...masking_utils import create_causal_mask, create_masks_for_generate
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
    torch_compilable_check,
)
from ...utils.output_capturing import capture_outputs
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import VideoMetadata
from ..gemma3.modeling_gemma3 import get_block_sequence_ids_for_mask
from ..llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaModelOutputWithPast,
)
from ..olmo2.modeling_olmo2 import Olmo2Attention
from ..phi3.modeling_phi3 import (
    Phi3DecoderLayer,
    Phi3MLP,
)
from ..siglip2.modeling_siglip2 import (
    Siglip2Encoder,
    Siglip2EncoderLayer,
    Siglip2MLP,
)
from .configuration_molmo2 import Molmo2AdapterConfig, Molmo2Config, Molmo2TextConfig, Molmo2VitConfig


logger = logging.get_logger(__name__)


def select_tiling(height: int, width: int, patch_size: int, max_num_crops: int) -> tuple[int, int]:
    """Select the image tiling in the same height/width order as the original Molmo2 processor."""
    tilings = []
    for tile_height in range(1, max_num_crops + 1):
        for tile_width in range(1, max_num_crops + 1):
            if tile_height * tile_width <= max_num_crops:
                tilings.append((tile_height, tile_width))
    tilings.sort(key=lambda x: (x[0] * x[1], x[0]))

    candidate_resolutions = np.array(tilings, dtype=np.int32) * patch_size
    original_size = np.stack([height, width], dtype=np.float32)

    with np.errstate(divide="ignore"):
        required_scales = candidate_resolutions.astype(np.float32) / original_size
    required_scale = np.min(required_scales, axis=-1, keepdims=True)

    if np.all(required_scale < 1):
        return tilings[np.argmax(required_scale)]

    required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
    return tilings[np.argmin(required_scale)]


def batch_pixels_to_patches(array: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Reshape images of [n_images, h, w, 3] -> [n_images, n_patches, pixels_per_patch]"""
    if len(array.shape) == 3:
        n_crops, height, width = array.shape
        h_patches = height // patch_size
        w_patches = width // patch_size
        array = array.reshape(n_crops, h_patches, patch_size, w_patches, patch_size)
        array = array.permute(0, 1, 3, 2, 4)
        array = array.reshape(n_crops, h_patches * w_patches, patch_size * patch_size)
        return array
    else:
        n_crops, height, width, channels = array.shape
        h_patches = height // patch_size
        w_patches = width // patch_size
        array = array.reshape(n_crops, h_patches, patch_size, w_patches, patch_size, channels)
        array = array.permute(0, 1, 3, 2, 4, 5)
        array = array.reshape(n_crops, h_patches * w_patches, patch_size * patch_size * channels)
        return array


def arange_for_pooling(
    idx_arr: torch.Tensor,
    pool_h: int,
    pool_w: int,
) -> torch.Tensor:
    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    idx_arr = F.pad(
        idx_arr,
        (w_pad // 2, (w_pad + 1) // 2, h_pad // 2, (h_pad + 1) // 2),
        mode="constant",
        value=-1,
    )
    num_rows, num_cols = idx_arr.shape[0] // pool_h, idx_arr.shape[1] // pool_w
    return (
        idx_arr.reshape(num_rows, pool_h, num_cols, pool_w)
        .permute(0, 2, 1, 3)
        .reshape(num_rows, num_cols, pool_h * pool_w)
    )


def resize_and_normalize_image(
    backend: "TorchvisionBackend",
    image_chw: torch.Tensor,
    output_size: list[int],
    resample: PILImageResampling,
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: list[float],
    image_std: list[float],
) -> torch.Tensor:
    input_dtype = image_chw.dtype
    resized = backend.resize(
        image_chw,
        size=SizeDict(height=output_size[0], width=output_size[1]),
        resample=resample,
        antialias=False,
    )
    if torch.is_floating_point(image_chw):
        resized = torch.clip(resized, 0.0, 1.0).to(input_dtype)
    else:
        if image_chw.dtype != torch.uint8:
            raise TypeError(f"Molmo2 expects float images or uint8 images, but got {image_chw.dtype}")
        resized = torch.clip(resized, 0, 255).to(input_dtype)

    resized = resized.to(torch.float32)
    if do_rescale and input_dtype == torch.uint8:
        resized = resized / (1.0 / rescale_factor)
    if do_normalize:
        mean = resized.new_tensor(image_mean)[:, None, None]
        std = resized.new_tensor(image_std)[:, None, None]
        resized = (resized - mean) / std
    return resized


def build_resized_image(
    backend: "TorchvisionBackend",
    image_chw: torch.Tensor,
    base_image_input_size: list[int],
    resample: PILImageResampling,
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: list[float],
    image_std: list[float],
    image_patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    chw_resized = resize_and_normalize_image(
        backend,
        image_chw,
        base_image_input_size,
        resample,
        do_rescale=do_rescale,
        rescale_factor=rescale_factor,
        do_normalize=do_normalize,
        image_mean=image_mean,
        image_std=image_std,
    )
    resized = chw_resized.permute(1, 2, 0).unsqueeze(0)
    crop_patch_w = base_image_input_size[1] // image_patch_size
    crop_patch_h = base_image_input_size[0] // image_patch_size
    resize_idx = torch.arange(crop_patch_w * crop_patch_h, dtype=torch.int32).reshape(crop_patch_h, crop_patch_w)
    return resized, resize_idx


class Molmo2ImagesKwargs(ImagesKwargs, total=False):
    """
    max_crops (`int`, *optional*, defaults to 8):
        Maximum number of crops to use per image.
    overlap_margins (`list[int]`, *optional*, defaults to `[4, 4]`):
        Overlap margins (in patches) for overlapping crop extraction.
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    pooling_size (`list[int]`, *optional*, defaults to `[2, 2]`):
        The pooling size of the vision adapter.
    """

    max_crops: int | None
    overlap_margins: list[int] | None
    patch_size: int | None
    pooling_size: list[int] | None


@auto_docstring
class Molmo2ImageProcessor(TorchvisionBackend):
    valid_kwargs = Molmo2ImagesKwargs
    model_input_names = ["pixel_values", "image_token_pooling", "image_grids", "image_num_crops"]
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 378, "width": 378}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    max_crops = 8
    overlap_margins = [4, 4]
    patch_size = 14
    pooling_size = [2, 2]

    def __init__(self, **kwargs: Unpack[Molmo2ImagesKwargs]):
        super().__init__(**kwargs)

    def _build_overlapping_crops(
        self,
        image_chw: torch.Tensor,
        max_crops: int,
        overlap_margins: list[int],
        base_image_input_size: list[int],
        resample: PILImageResampling,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: list[float],
        image_std: list[float],
        image_patch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tile `image_chw` into overlapping square crops and return per-patch global indices with overlap patches collapsed to a single owner."""
        if base_image_input_size[0] != base_image_input_size[1]:
            raise ValueError(f"Expected square base_image_input_size, got {base_image_input_size}")
        crop_size = base_image_input_size[0]
        left_margin, right_margin = overlap_margins
        crop_patches = crop_size // image_patch_size
        window_patches = crop_patches - (left_margin + right_margin)
        window_size = window_patches * image_patch_size
        margin_size = (left_margin + right_margin) * image_patch_size

        _, original_h, original_w = image_chw.shape
        tiling_h, tiling_w = select_tiling(original_h - margin_size, original_w - margin_size, window_size, max_crops)
        src_h = tiling_h * window_size + margin_size
        src_w = tiling_w * window_size + margin_size

        src = resize_and_normalize_image(
            self,
            image_chw,
            [src_h, src_w],
            resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
        )

        crops = src.unfold(1, crop_size, window_size).unfold(2, crop_size, window_size)
        crops = crops.permute(1, 2, 3, 4, 0).reshape(tiling_h * tiling_w, crop_size, crop_size, 3).contiguous()

        patch_idx = torch.arange(tiling_h * tiling_w * crop_patches * crop_patches, dtype=torch.int32).reshape(
            tiling_h, tiling_w, crop_patches, crop_patches
        )
        if left_margin:
            patch_idx[1:, :, :left_margin, :] = -1
            patch_idx[:, 1:, :, :left_margin] = -1
        if right_margin:
            patch_idx[:-1, :, -right_margin:, :] = -1
            patch_idx[:, :-1, :, -right_margin:] = -1

        patch_idx = patch_idx.permute(0, 2, 1, 3).reshape(-1)
        patch_idx = patch_idx[patch_idx >= 0].reshape(src_h // image_patch_size, src_w // image_patch_size)

        return crops, patch_idx

    def _image_to_patches_and_grids(
        self,
        image_chw: torch.Tensor,
        max_crops: int,
        overlap_margins: list[int],
        base_image_input_size: list[int],
        resample: PILImageResampling,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: list[float],
        image_std: list[float],
        image_patch_size: int,
        image_pooling_w: int,
        image_pooling_h: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(base_image_input_size, int):
            base_image_input_size = (base_image_input_size, base_image_input_size)

        base_image_input_d = image_patch_size
        pooling_w = image_pooling_w
        pooling_h = image_pooling_h
        crop_patch_w = base_image_input_size[1] // base_image_input_d
        crop_patch_h = base_image_input_size[0] // base_image_input_d

        crop_arr, patch_idx_arr = self._build_overlapping_crops(
            image_chw,
            max_crops,
            overlap_margins,
            base_image_input_size,
            resample,
            do_rescale,
            rescale_factor,
            do_normalize,
            image_mean,
            image_std,
            image_patch_size,
        )
        pooling_idx = arange_for_pooling(patch_idx_arr, pooling_h, pooling_w)
        num_patch_rows, num_patch_cols = pooling_idx.shape[:2]
        pooling_idx = pooling_idx.reshape([-1, pooling_h * pooling_w])

        resized, resize_idx = build_resized_image(
            self,
            image_chw,
            base_image_input_size,
            resample,
            do_rescale,
            rescale_factor,
            do_normalize,
            image_mean,
            image_std,
            image_patch_size,
        )
        crop_arr = torch.cat([resized, crop_arr], 0)

        resize_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
        resized_h, resized_w = resize_idx.shape[:2]
        resize_idx = resize_idx.reshape([-1, pooling_h * pooling_w])

        pooling_idx = torch.where(pooling_idx >= 0, pooling_idx + crop_patch_h * crop_patch_w, -1)
        pooling_idx = torch.cat([resize_idx, pooling_idx])
        image_grid = torch.tensor([[resized_h, resized_w, num_patch_rows, num_patch_cols]], dtype=torch.int64)

        return image_grid, batch_pixels_to_patches(crop_arr, image_patch_size), pooling_idx

    def _prepare_images_structure(
        self,
        images: ImageInput,
        expected_ndims: int = 3,
    ) -> ImageInput:
        images = self.fetch_images(images)
        return make_nested_list_of_images(images, expected_ndims=expected_ndims)

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[Molmo2ImagesKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list[list["torch.Tensor"]],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: list[float],
        image_std: list[float],
        do_convert_rgb: bool,
        max_crops: int,
        overlap_margins: list[int],
        patch_size: int,
        pooling_size: list[int],
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        base_image_input_size = [size.height, size.width]
        image_pooling_h, image_pooling_w = pooling_size

        all_grids: list[torch.Tensor] = []
        all_crops: list[torch.Tensor] = []
        all_pooled: list[torch.Tensor] = []
        all_num_crops: list[int] = []
        patch_offset = 0

        for sample_images in images:
            for image in sample_images:
                image_grid, crops, pooled_idx = self._image_to_patches_and_grids(
                    image,
                    max_crops,
                    overlap_margins,
                    base_image_input_size,
                    resample,
                    do_rescale,
                    rescale_factor,
                    do_normalize,
                    image_mean,
                    image_std,
                    patch_size,
                    image_pooling_w,
                    image_pooling_h,
                )
                pooled_idx = torch.where(pooled_idx >= 0, pooled_idx + patch_offset, pooled_idx)
                patch_offset += crops.shape[0] * crops.shape[1]

                all_grids.append(image_grid)
                all_crops.append(crops)
                all_pooled.append(pooled_idx)
                all_num_crops.append(crops.shape[0])

        data = {
            "pixel_values": torch.cat(all_crops, dim=0),
            "image_token_pooling": torch.cat(all_pooled, dim=0),
            "image_grids": torch.cat(all_grids, dim=0),
            "image_num_crops": torch.tensor(all_num_crops, dtype=torch.int64),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None) -> int:
        if images_kwargs is None:
            images_kwargs = {}
        max_crops = images_kwargs.get("max_crops", self.max_crops)
        overlap_margins = images_kwargs.get("overlap_margins", self.overlap_margins)
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        pooling_size = images_kwargs.get("pooling_size", self.pooling_size)
        size = images_kwargs.get("size", self.size)

        base_h = size["height"] if isinstance(size, dict) else size.height
        base_w = size["width"] if isinstance(size, dict) else size.width
        left_margin, right_margin = overlap_margins
        pooling_h, pooling_w = pooling_size

        total_margin_pixels = patch_size * (left_margin + right_margin)
        crop_patches = base_h // patch_size
        crop_window_patches = crop_patches - (left_margin + right_margin)
        crop_window_size = crop_window_patches * patch_size

        effective_h = height - total_margin_pixels
        effective_w = width - total_margin_pixels
        tiling_h, tiling_w = select_tiling(effective_h, effective_w, crop_window_size, max_crops)

        high_res_h = tiling_h * crop_window_patches + left_margin + right_margin
        high_res_w = tiling_w * crop_window_patches + left_margin + right_margin
        num_patch_rows_high = math.ceil(high_res_h / pooling_h)
        num_patch_cols_high = math.ceil(high_res_w / pooling_w)

        crop_patch_h = base_h // patch_size
        crop_patch_w = base_w // patch_size
        resized_h = math.ceil(crop_patch_h / pooling_h)
        resized_w = math.ceil(crop_patch_w / pooling_w)

        return resized_h * resized_w + num_patch_rows_high * num_patch_cols_high


class Molmo2VideosKwargs(VideosKwargs, total=False):
    """
    patch_size (`int`, *optional*):
        Side length in pixels of each ViT patch for video frames.
    pooling_size (`list[int]`, *optional*):
        `[pool_h, pool_w]` pooling window applied to video patch features.
    max_fps (`int`, *optional*):
        Maximum sampling rate in frames per second for short videos.
    """

    patch_size: int | None
    pooling_size: list[int] | None
    max_fps: int | None


@auto_docstring
class Molmo2VideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR
    size = {"height": 378, "width": 378}
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    patch_size = 14
    pooling_size = [3, 3]
    num_frames = 64
    do_sample_frames = True
    max_fps = 2
    valid_kwargs = Molmo2VideosKwargs
    model_input_names = ["pixel_values_videos", "video_token_pooling", "video_grids"]

    def __init__(self, **kwargs: Unpack[Molmo2VideosKwargs]):
        super().__init__(**kwargs)
        if self.size is not None and (self.size.get("height", None) is None or self.size.get("width", None) is None):
            raise ValueError("size must contain 'height' and 'width' keys.")

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: int | None = None,
        fps: int | float | None = None,
        max_fps: int | float | None = None,
        **kwargs,
    ):
        if fps is not None and num_frames is not None:
            raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")

        if max_fps is None:
            max_fps = self.max_fps

        if metadata.fps is None:
            metadata.fps = fps if fps is not None else max_fps
            logger.warning_once(
                "Molmo2 inserts frame timestamps into video prompts, but the input video's `fps` was not provided "
                f"or could not be inferred. Defaulting to `fps={metadata.fps}`. Please provide `video_metadata` "
                "for more accurate timestamps."
            )
        if metadata.duration is None:
            metadata.duration = metadata.total_num_frames / metadata.fps

        if fps is not None:
            target_num_frames = int(metadata.duration * fps)
        else:
            target_num_frames = num_frames if num_frames is not None else self.num_frames
            if max_fps is not None and metadata.fps > max_fps:
                target_num_frames = min(target_num_frames, int(metadata.duration * max_fps))

        target_num_frames = max(min(target_num_frames, metadata.total_num_frames), 1)
        total = metadata.total_num_frames
        return torch.arange(0, total, total / target_num_frames).int()

    def _build_frame_patches(
        self,
        frame_chw: torch.Tensor,
        base_image_input_size: list[int],
        resample: PILImageResampling,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: list[float],
        image_std: list[float],
        image_patch_size: int,
        image_pooling_h: int,
        image_pooling_w: int,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        hwc, resize_idx = build_resized_image(
            self,
            frame_chw,
            base_image_input_size,
            resample,
            do_rescale,
            rescale_factor,
            do_normalize,
            image_mean,
            image_std,
            image_patch_size,
        )
        pooling_idx = arange_for_pooling(resize_idx, image_pooling_h, image_pooling_w)
        num_patch_rows, num_patch_cols = pooling_idx.shape[:2]
        pooling_idx = pooling_idx.reshape([-1, image_pooling_h * image_pooling_w])
        return [num_patch_rows, num_patch_cols], batch_pixels_to_patches(hwc, image_patch_size), pooling_idx

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        size: SizeDict | None = None,
        resample: PILImageResampling | None = None,
        do_rescale: bool = True,
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        patch_size: int | None = None,
        pooling_size: list[int] | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        base_image_input_size = [size.height, size.width]
        image_pooling_h, image_pooling_w = pooling_size

        all_crops: list[torch.Tensor] = []
        all_pooled: list[torch.Tensor] = []
        all_grids: list[torch.Tensor] = []
        patch_offset = 0

        for video in videos:
            for frame_chw in video:
                image_grid, crops, pooled_idx = self._build_frame_patches(
                    frame_chw,
                    base_image_input_size,
                    resample,
                    do_rescale,
                    rescale_factor,
                    do_normalize,
                    image_mean,
                    image_std,
                    patch_size,
                    image_pooling_h,
                    image_pooling_w,
                )
                all_pooled.append(torch.where(pooled_idx >= 0, pooled_idx + patch_offset, pooled_idx))
                all_crops.append(crops)
                patch_offset += crops.shape[0] * crops.shape[1]

            all_grids.append(torch.tensor([len(video), image_grid[0], image_grid[1]], dtype=torch.int64))

        data = {
            "pixel_values_videos": torch.cat(all_crops, dim=0),
            "video_token_pooling": torch.cat(all_pooled, dim=0),
            "video_grids": torch.stack(all_grids, dim=0),
        }
        return BatchFeature(data, tensor_type=return_tensors)


class Molmo2ProcessorImagesKwargs(Molmo2ImagesKwargs, total=False):
    """
    max_crops (`int`, *optional*):
        Maximum number of image crops produced by the image processor.
    overlap_margins (`list[int]`, *optional*):
        Pixel margins `[left_right, top_bottom]` to overlap between neighboring crops.
    patch_size (`int`, *optional*):
        Side length in pixels of each ViT patch.
    pooling_size (`list[int]`, *optional*):
        `[pool_h, pool_w]` pooling window applied to patch features in the vision adapter.
    """


class Molmo2ProcessorKwargs(ProcessingKwargs, total=False):
    """Molmo2 processor kwargs"""

    images_kwargs: Molmo2ProcessorImagesKwargs
    videos_kwargs: Molmo2VideosKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": True,
        },
        "videos_kwargs": {"return_metadata": True},
    }


@auto_docstring
class Molmo2Processor(ProcessorMixin):
    valid_processor_kwargs = Molmo2ProcessorKwargs
    image_token = "<|image|>"
    video_token = "<|video|>"
    image_patch_tokens = (
        "<im_patch>",
        "<im_col>",
        "<im_start>",
        "<low_res_im_start>",
        "<frame_start>",
        "<im_end>",
        "<frame_end>",
        "<im_low>",
    )

    @property
    def model_input_names(self):
        return super().model_input_names + ["mm_token_type_ids"]

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.image_token)

    @property
    def video_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.video_token)

    @property
    def image_token_ids(self) -> list[int]:
        return [self.tokenizer.convert_tokens_to_ids(token) for token in self.image_patch_tokens]

    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        tokenizer=None,
        chat_template: str | None = None,
        image_use_col_tokens: bool | None = True,
        use_single_crop_col_tokens: bool | None = None,
        use_single_crop_start_token: bool | None = True,
        video_use_col_tokens: bool | None = False,
        use_frame_special_tokens: bool | None = True,
        **kwargs,
    ) -> None:
        r"""
        image_use_col_tokens (`bool`, *optional*, defaults to `True`):
            Whether to append column-separator tokens (`<im_col>`) after each patch row of the high-resolution image
            view.
        use_single_crop_col_tokens (`bool`, *optional*):
            Whether to append column-separator tokens after each patch row of the low-resolution (single-crop) image
            view. If `None`, falls back to `image_use_col_tokens`.
        use_single_crop_start_token (`bool`, *optional*, defaults to `True`):
            Whether to start the low-resolution image view with `<low_res_im_start>` instead of the regular
            `<im_start>`.
        video_use_col_tokens (`bool`, *optional*, defaults to `False`):
            Whether to append column-separator tokens after each patch row of video frames.
        use_frame_special_tokens (`bool`, *optional*, defaults to `True`):
            Whether to wrap each video frame with `<frame_start>` / `<frame_end>` tokens. If `False`, falls back to
            `<im_start>` / `<im_end>`.
        """
        self.image_use_col_tokens = image_use_col_tokens
        self.use_single_crop_col_tokens = use_single_crop_col_tokens
        self.use_single_crop_start_token = use_single_crop_start_token
        self.video_use_col_tokens = video_use_col_tokens
        self.use_frame_special_tokens = use_frame_special_tokens
        super().__init__(image_processor, video_processor, tokenizer, chat_template=chat_template)

    def get_video_string(self, video_grid, timestamps) -> str:
        if hasattr(video_grid, "tolist"):
            video_grid = video_grid.tolist()
        start_token = "<frame_start>" if self.use_frame_special_tokens else "<im_start>"
        end_token = "<frame_end>" if self.use_frame_special_tokens else "<im_end>"

        num_frames, num_patch_rows, num_patch_cols = video_grid
        video_string = ""
        for frame_idx, frame_time in enumerate(timestamps):
            prev_space = " " if frame_idx > 0 else ""
            video_string += prev_space + f"{frame_time:.1f} "
            per_row = ["<im_patch>"] * num_patch_cols
            if self.video_use_col_tokens:
                per_row = per_row + ["<im_col>"]
            video_string += "".join([start_token] + per_row * num_patch_rows + [end_token])

        return video_string

    def validate_inputs(
        self,
        images=None,
        text=None,
        videos=None,
        audio=None,
        **kwargs,
    ):
        super().validate_inputs(images=images, text=text, videos=videos, audio=audio, **kwargs)
        if videos is not None and text is not None:
            for sample in text:
                if sample.count(self.video_token) > 1:
                    raise ValueError("At most one video is supported per sample.")

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        image_grid = image_inputs["image_grids"][image_idx]
        if hasattr(image_grid, "tolist"):
            image_grid = image_grid.tolist()
        resized_h, resized_w, height, width = image_grid

        per_row = ["<im_patch>"] * width
        if self.image_use_col_tokens:
            per_row = per_row + ["<im_col>"]
        high_res_tokens = ["<im_start>"] + per_row * height + ["<im_end>"]

        per_row = ["<im_patch>"] * resized_w
        use_single_crop_col_tokens = (
            self.image_use_col_tokens if self.use_single_crop_col_tokens is None else self.use_single_crop_col_tokens
        )
        image_start_token = "<low_res_im_start>" if self.use_single_crop_start_token else "<im_start>"
        if use_single_crop_col_tokens:
            per_row = per_row + ["<im_col>"]
        low_res_tokens = [image_start_token] + per_row * resized_h + ["<im_end>"]

        return "".join(low_res_tokens + high_res_tokens)

    def _process_videos(self, videos, **kwargs):
        video_inputs = self.video_processor(videos=videos, **kwargs)
        video_grids = video_inputs["video_grids"]
        video_metadata = video_inputs.get("video_metadata", [])
        video_replacements = []
        for video_idx, video_grid in enumerate(video_grids):
            metadata = video_metadata[video_idx] if video_idx < len(video_metadata) else None
            if metadata is not None:
                if metadata.frames_indices is None:
                    metadata.frames_indices = list(range(int(video_grid[0].item())))
                if metadata.fps is None:
                    metadata.fps = self.video_processor.max_fps or 2
                    logger.warning_once(
                        "Molmo2 inserts frame timestamps into video prompts, but the input video's `fps` was not "
                        f"provided or could not be inferred. Defaulting to `fps={metadata.fps}`. Please provide "
                        "`video_metadata` for more accurate timestamps."
                    )
                timestamps = metadata.timestamps
            else:
                fps = self.video_processor.max_fps or 2
                num_frames = int(video_grid[0].item())
                timestamps = [i / fps for i in range(num_frames)]
            video_replacements.append(self.get_video_string(video_grid, timestamps))
        return video_inputs, video_replacements

    def apply_chat_template(
        self,
        conversation,
        chat_template: str | None = None,
        **kwargs,
    ):
        uses_default_template = chat_template is None
        if chat_template is None:
            if isinstance(self.chat_template, dict):
                chat_template = self.chat_template.get("default")
            else:
                chat_template = self.chat_template
        elif isinstance(self.chat_template, dict) and chat_template in self.chat_template:
            uses_default_template = True
            chat_template = self.chat_template[chat_template]

        if (
            uses_default_template
            and isinstance(chat_template, str)
            and self.tokenizer.bos_token is not None
            and "{{ bos_token" not in chat_template
            and not chat_template.lstrip().startswith(self.tokenizer.bos_token)
        ):
            chat_template = "{{ bos_token }}" + chat_template

        return super().apply_chat_template(conversation, chat_template=chat_template, **kwargs)


# Output dataclasses - same structure as LLaVA
class Molmo2CausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    pass


class Molmo2ModelOutputWithPast(LlavaModelOutputWithPast):
    pass


class Molmo2VisionMLP(Siglip2MLP):
    pass


class Molmo2GQAAttention(nn.Module):
    def __init__(
        self,
        config: Molmo2VitConfig | Molmo2AdapterConfig,
        input_dim: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        input_dim = config.hidden_size if input_dim is None else input_dim
        self.q_proj = nn.Linear(input_dim, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(input_dim, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(input_dim, self.num_key_value_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        key_value_states = hidden_states if key_value_states is None else key_value_states

        batch_size = hidden_states.shape[0]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(key_value_states)
        value_states = self.v_proj(key_value_states)

        query_states = query_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        target_dtype = query_states.dtype
        if getattr(self.config, "float32_attention", False):
            query_states = query_states.float()
            key_states = key_states.float()
            if self.config._attn_implementation == "sdpa" and not torch.is_autocast_enabled():
                value_states = value_states.float()

        dropout = 0.0 if not self.training else self.attention_dropout
        if self.config._attn_implementation == "eager":
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            attn_weights = torch.matmul(query_states / math.sqrt(query_states.size(-1)), key_states.transpose(2, 3))
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=self.training)
            attn_output = torch.matmul(attn_weights.to(value_states.dtype), value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                is_causal=self.is_causal,
                scaling=self.scaling,
                dropout=dropout,
            )

        attn_output = attn_output.to(target_dtype)
        attn_output = attn_output.reshape(batch_size, -1, self.num_heads * self.head_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class Molmo2VisionEncoderLayer(Siglip2EncoderLayer):
    def __init__(self, config: Molmo2VitConfig):
        super().__init__(config)
        self.self_attn = Molmo2GQAAttention(config)
        self.mlp = Molmo2VisionMLP(config)


class Molmo2VisionEncoder(Siglip2Encoder):
    def __init__(self, config: Molmo2VitConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([Molmo2VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])


class Molmo2VisionModel(PreTrainedModel):
    config_class = Molmo2VitConfig
    _no_split_modules = ["Molmo2VisionEncoderLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _can_record_outputs = {
        "hidden_states": Molmo2VisionEncoderLayer,
        "attentions": Molmo2GQAAttention,
    }

    def _init_weights(self, module):
        if isinstance(module, Molmo2VisionModel):
            init.normal_(module.positional_embedding, mean=0.0, std=self.config.initializer_range)
        else:
            super()._init_weights(module)

    def __init__(self, config: Molmo2VitConfig):
        super().__init__(config)
        self.config = config
        self.image_default_input_size = config.image_default_input_size

        # positional embeddings
        self.scale = config.hidden_size**-0.5
        self.num_prefix_tokens: int = 0  # no class embeddings
        self.positional_embedding = nn.Parameter(
            torch.zeros(config.image_num_pos, config.hidden_size),
        )

        image_patch_size = config.image_patch_size
        self.patch_embedding = nn.Linear(
            image_patch_size * image_patch_size * 3,
            config.hidden_size,
            bias=True,
        )

        self.encoder = Molmo2VisionEncoder(config)

        self.post_init()

    @capture_outputs(tie_last_hidden_states=False)
    def forward(self, pixel_values: torch.Tensor, **kwargs) -> BaseModelOutputWithPooling:
        target_dtype = self.patch_embedding.weight.dtype
        hidden_states = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        # patch count == image_num_pos, locked by config; only retraining with a different grid breaks this.
        hidden_states = hidden_states + self.positional_embedding[None, :, :].to(hidden_states.dtype)

        encoder_outputs = self.encoder(hidden_states, **kwargs)
        last_hidden_state = encoder_outputs.last_hidden_state
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=last_hidden_state.mean(dim=1),
        )


class Molmo2ImageProjectorMLP(nn.Module):
    def __init__(self, config: Molmo2AdapterConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.text_hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Molmo2VisionBackbone(PreTrainedModel):
    config_class = Molmo2AdapterConfig

    def __init__(self, vit_config: Molmo2VitConfig, adapter_config: Molmo2AdapterConfig):
        super().__init__(adapter_config)
        self.pooling_attention_mask = adapter_config.pooling_attention_mask
        # `vit_config.num_hidden_layers` and `adapter_config.vit_layers` are normalized in `Molmo2Config.__post_init__`.
        self.vit_layers = list(adapter_config.vit_layers)
        self.image_vit = Molmo2VisionModel(vit_config)

        pool_dim = vit_config.hidden_size * len(adapter_config.vit_layers)
        self.image_pooling_2d = Molmo2GQAAttention(adapter_config, input_dim=pool_dim)
        self.image_projector = Molmo2ImageProjectorMLP(adapter_config)
        self.image_feature_dropout = nn.Dropout(adapter_config.image_feature_dropout)
        self.post_init()

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        image_shape = None
        if images.dim() == 4:
            image_shape = images.shape[:3]
            images = images.reshape(-1, images.shape[-2], images.shape[-1])

        image_outputs = self.image_vit(images, output_hidden_states=True)
        image_features = image_outputs.hidden_states
        features = [image_features[layer + 1] for layer in self.vit_layers]
        image_features = torch.cat(features, dim=-1)

        if image_shape is not None:
            image_features = image_features.reshape(*image_shape, -1)
        return image_features

    def forward(
        self,
        images: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        image_features = self.encode_image(images)
        image_features = self.image_feature_dropout(image_features)
        dim = image_features.shape[-1]
        flat_features = image_features.reshape(-1, dim)

        valid = pooled_patches_idx >= 0
        valid_token = torch.any(valid, -1)

        to_pool = flat_features[torch.clip(pooled_patches_idx, 0)]
        to_pool = to_pool * valid.to(to_pool.dtype)[..., None]

        if self.pooling_attention_mask:
            attn_mask = valid.reshape(-1, 1, 1, valid.shape[-1])
            denom = valid.float().sum(-1)
            denom = torch.where(denom == 0, 1, denom)
            query = to_pool.sum(-2, keepdim=True) / denom[:, None, None].to(to_pool.dtype)
        else:
            attn_mask = None
            query = to_pool.mean(-2, keepdim=True)

        pooled_features, _ = self.image_pooling_2d(query, to_pool, attention_mask=attn_mask)
        pooled_features = pooled_features.squeeze(1)
        pooled_features = self.image_projector(pooled_features)
        return pooled_features[valid_token]


class Molmo2RotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, config: Molmo2TextConfig, rope_type: str | None = None):
        nn.Module.__init__(self)
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_type = rope_type or config.rope_parameters["rope_type"]
        rope_init_fn = (
            self.compute_default_rope_parameters
            if self.rope_type == "default"
            else ROPE_INIT_FUNCTIONS[self.rope_type]
        )
        inv_freq, self.attention_scaling = rope_init_fn(config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)


class Molmo2RMSNorm(LlamaRMSNorm):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__(hidden_size, eps=eps)
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=hidden_states.device.type):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Molmo2Attention(Olmo2Attention):
    """Molmo2 attention: Olmo2-style q/k RMSNorm with a fused QKV projection and renamed output projection."""

    def __init__(self, config: Molmo2TextConfig, layer_idx: int) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.fused_dims = (
            config.num_attention_heads * config.head_dim,
            config.head_dim * config.num_key_value_heads,
            config.head_dim * config.num_key_value_heads,
        )
        self.att_proj = nn.Linear(config.hidden_size, sum(self.fused_dims), bias=config.qkv_bias)
        self.attn_out = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)

        if not config.use_qk_norm:
            raise ValueError(
                "Molmo2 requires `use_qk_norm=True`; all shipped checkpoints apply Q/K norm and "
                "no q_norm/k_norm-less path is provided."
            )
        self.qk_norm_type = config.qk_norm_type
        if self.qk_norm_type == "qwen3":
            q_norm_dim = config.head_dim
            k_norm_dim = config.head_dim
        elif self.qk_norm_type == "olmo":
            q_norm_dim = config.num_attention_heads * config.head_dim
            k_norm_dim = config.num_key_value_heads * config.head_dim
        else:
            raise ValueError(f"Unsupported `qk_norm_type`: {self.qk_norm_type}")
        self.q_norm = Molmo2RMSNorm(q_norm_dim, eps=config.layer_norm_eps)
        self.k_norm = Molmo2RMSNorm(k_norm_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        q_shape = (*input_shape, self.num_heads, self.head_dim)
        kv_shape = (*input_shape, self.num_key_value_heads, self.head_dim)

        qkv = self.att_proj(hidden_states)
        query_states, key_states, value_states = torch.split(qkv, self.fused_dims, dim=-1)

        value_states = value_states.view(kv_shape)

        if self.qk_norm_type == "olmo":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.view(q_shape)
        key_states = key_states.view(kv_shape)

        if self.qk_norm_type == "qwen3":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.attn_out(attn_output)
        return attn_output, attn_weights


class Molmo2MLP(Phi3MLP):
    def __init__(self, input_dim: int, intermediate_size: int, hidden_act: str):
        nn.Module.__init__(self)
        self.ff_proj = nn.Linear(input_dim, intermediate_size * 2, bias=False)
        self.ff_out = nn.Linear(intermediate_size, input_dim, bias=False)
        self.act = ACT2FN[hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ff_proj(hidden_states)
        up_states, gate = hidden_states.chunk(2, dim=-1)
        hidden_states = self.act(gate) * up_states
        return self.ff_out(hidden_states)


class Molmo2DecoderLayer(Phi3DecoderLayer):
    def __init__(self, config: Molmo2TextConfig, layer_idx: int | None = None):
        GradientCheckpointingLayer.__init__(self)
        self.config = config

        self.self_attn = Molmo2Attention(config, layer_idx)
        self.attn_norm = Molmo2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.residual_dropout)
        self.mlp = Molmo2MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.ff_norm = Molmo2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = residual + self.dropout(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ff_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + self.dropout(hidden_states)
        return hidden_states


class Molmo2PostNormDecoderLayer(Molmo2DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = self.attn_norm(hidden_states)

        hidden_states = residual + self.dropout(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.ff_norm(hidden_states)

        hidden_states = residual + self.dropout(hidden_states)
        return hidden_states


class Molmo2Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        num_new_embeddings: int,
        features: int,
    ):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(num_embeddings, features))
        self.new_embedding = nn.Parameter(torch.zeros(num_new_embeddings, features))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, torch.cat([self.embedding, self.new_embedding], dim=0))


class Molmo2PreTrainedModel(LlamaPreTrainedModel):
    config: Molmo2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "Molmo2DecoderLayer",
        "Molmo2PostNormDecoderLayer",
        "Molmo2VisionEncoderLayer",
        "Molmo2GQAAttention",
    ]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Molmo2DecoderLayer,
        "attentions": Molmo2Attention,
    }

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, Molmo2Embedding):
            init.normal_(module.embedding, mean=0.0, std=std)
            init.normal_(module.new_embedding, mean=0.0, std=std)
        elif isinstance(module, Molmo2VisionModel):
            init.normal_(module.positional_embedding, mean=0.0, std=std)
        else:
            super()._init_weights(module)


class Molmo2TextModel(Molmo2PreTrainedModel):
    config: Molmo2TextConfig
    _input_embed_layer = "wte"

    def __init__(self, config: Molmo2TextConfig):
        super().__init__(config)
        if config.additional_vocab_size is not None:
            self.wte = Molmo2Embedding(
                config.vocab_size,
                config.additional_vocab_size,
                config.hidden_size,
            )
        else:
            self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_drop = nn.Dropout(config.embedding_dropout)
        decoder_layer = Molmo2PostNormDecoderLayer if config.norm_after else Molmo2DecoderLayer
        self.blocks = nn.ModuleList(
            [decoder_layer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Molmo2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_embs = nn.ModuleDict({"default": Molmo2RotaryEmbedding(config, rope_type="default")})
        if config.rope_scaling_layers:
            self.rotary_embs["scaling"] = Molmo2RotaryEmbedding(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            ).unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            causal_mask_mapping = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds

        position_embeddings = {key: self.rotary_embs[key](hidden_states, position_ids) for key in self.rotary_embs}

        for layer_idx, decoder_block in enumerate(self.blocks[: self.config.num_hidden_layers]):
            rope_key = "scaling" if layer_idx in self.config.rope_scaling_layers else "default"
            hidden_states = decoder_block(
                hidden_states,
                attention_mask=causal_mask_mapping,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings[rope_key],
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


def create_causal_mask_mapping(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None,
    mm_token_type_ids: torch.Tensor | None = None,
    has_multimodal_inputs: bool = False,
    is_training: bool = False,
    is_first_iteration: bool | None = None,
    **kwargs,
) -> dict:
    """
    Create the causal mask mapping for Molmo2 forward passes. Multimodal spans use bidirectional attention within
    each contiguous image/video group.
    """
    if is_training and mm_token_type_ids is None:
        raise ValueError("`mm_token_type_ids` is required as a model input when training")

    mask_kwargs = {
        "config": config.get_text_config(),
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }

    is_first_iteration = (
        is_first_iteration
        if is_first_iteration is not None
        else (past_key_values is None or not past_key_values.is_initialized or has_multimodal_inputs)
    )
    if mm_token_type_ids is not None and is_first_iteration:
        mask_kwargs["block_sequence_ids"] = get_block_sequence_ids_for_mask(
            mm_token_type_ids, device=inputs_embeds.device
        )

    return create_causal_mask(**mask_kwargs)


class Molmo2Model(Molmo2PreTrainedModel):
    config: Molmo2Config

    def __init__(self, config: Molmo2Config):
        super().__init__(config)
        self.language_model: Molmo2TextModel = Molmo2TextModel(config.text_config)
        self.image_col_id = config.image_col_id
        self.image_low_res_id = config.image_low_res_id
        self.vision_backbone: Molmo2VisionBackbone | None = None
        if config.vit_config is not None and config.adapter_config is not None:
            self.vision_backbone = Molmo2VisionBackbone(config.vit_config, config.adapter_config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.language_model.wte

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.language_model.wte = value

    def merge_visual_inputs(
        self,
        pixel_values: torch.Tensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if pixel_values is not None and pixel_values_videos is not None:
            raise ValueError("pixel_values and pixel_values_videos are provided at the same time")
        if pixel_values is not None:
            if image_token_pooling is None:
                raise ValueError("`image_token_pooling` must be provided when `pixel_values` is passed.")
            return pixel_values, image_token_pooling
        if pixel_values_videos is not None:
            return pixel_values_videos, video_token_pooling
        return None, None

    @can_return_tuple
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_token_pooling: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        return self.vision_backbone.image_vit(pixel_values, **kwargs)

    # Copied from transformers.models.llava.modeling_llava.LlavaModel.get_placeholder_mask
    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        n_image_features = image_features.shape[0] * image_features.shape[1]
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        torch_compilable_check(
            inputs_embeds[special_image_mask].numel() == image_features.numel(),
            f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {n_image_features}",
        )
        return special_image_mask

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
        video_grids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Molmo2ModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.text_config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        images, token_pooling = self.merge_visual_inputs(
            pixel_values=pixel_values,
            image_token_pooling=image_token_pooling,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
        )

        if images is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both images and inputs_embeds at the same time.")

        if inputs_embeds is None:
            inputs_embeds = self.language_model.wte(input_ids)

        image_features: torch.FloatTensor | None = None
        if images is not None:
            image_features = self.vision_backbone(images, token_pooling)
            image_token_counts = (input_ids == self.config.image_token_id).sum(dim=-1)
            total_image_tokens = image_token_counts.sum()
            if total_image_tokens != image_features.shape[0] and total_image_tokens % image_features.shape[0] == 0:
                num_beams = total_image_tokens // image_features.shape[0]
                original_image_token_counts = image_token_counts[::num_beams]
                if original_image_token_counts.sum() == image_features.shape[0]:
                    image_features = torch.cat(
                        [
                            image_features_slice
                            for image_features_slice in image_features.split(
                                [int(count) for count in original_image_token_counts.tolist()]
                            )
                            for _ in range(num_beams)
                        ],
                        dim=0,
                    )
            special_image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features)
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask,
                inputs_embeds[special_image_mask] + image_features.reshape(-1),
            )

        inputs_embeds = self.language_model.emb_drop(inputs_embeds)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            causal_mask_mapping = create_causal_mask_mapping(
                self.config,
                inputs_embeds,
                attention_mask,
                past_key_values,
                position_ids,
                mm_token_type_ids,
                has_multimodal_inputs=images is not None,
                is_training=self.training,
            )

        outputs = self.language_model(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return Molmo2ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if images is not None else None,
        )


class Molmo2ForConditionalGeneration(Molmo2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = []  # Weights are not tied
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Molmo2Config

    def __init__(self, config: Molmo2Config):
        super().__init__(config)

        self.model = Molmo2Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.vocab_size = config.text_config.vocab_size

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.Tensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
        video_grids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Molmo2CausalLMOutputWithPast:
        r"""
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from ... import AutoProcessor, Molmo2ForConditionalGeneration

        >>> model = Molmo2ForConditionalGeneration.from_pretrained("...")
        >>> processor = AutoProcessor.from_pretrained("...")

        >>> prompt = "What's the content of the image?"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}]}]

        >>> inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=15)
        >>> generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
        >>> processor.post_process_image_text_to_text(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a bustling street scene in what appears to be a Chinatown area. There's ..."
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_pooling=image_token_pooling,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size)

        return Molmo2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ):
        visual_keys = (
            "pixel_values",
            "image_token_pooling",
            "image_grids",
            "image_num_crops",
            "pixel_values_videos",
            "video_token_pooling",
            "video_grids",
        )
        visual = {k: model_kwargs.pop(k) for k in visual_keys if k in model_kwargs}
        input_ids, model_kwargs = super()._expand_inputs_for_generation(
            expand_size=expand_size,
            is_encoder_decoder=is_encoder_decoder,
            input_ids=input_ids,
            **model_kwargs,
        )
        model_kwargs.update(visual)
        return input_ids, model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
        video_grids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor | None = None,
        is_first_iteration: bool = False,
        use_cache: bool = True,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            mm_token_type_ids=mm_token_type_ids,
            is_first_iteration=is_first_iteration,
            use_cache=use_cache,
            **kwargs,
        )

        if is_first_iteration or not use_cache:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_token_pooling"] = image_token_pooling
            model_inputs["image_grids"] = image_grids
            model_inputs["image_num_crops"] = image_num_crops
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["video_token_pooling"] = video_token_pooling
            model_inputs["video_grids"] = video_grids

        return model_inputs

    @staticmethod
    def create_masks_for_generate(
        config: PreTrainedConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        mask_kwargs = {
            "config": config.get_text_config(),
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        if mm_token_type_ids is not None and inputs_embeds.shape[1] != 1:
            mask_kwargs["block_sequence_ids"] = get_block_sequence_ids_for_mask(
                mm_token_type_ids, device=inputs_embeds.device
            )

        return create_masks_for_generate(**mask_kwargs)


__all__ = [
    "Molmo2ForConditionalGeneration",
    "Molmo2ImageProcessor",
    "Molmo2Model",
    "Molmo2PreTrainedModel",
    "Molmo2Processor",
    "Molmo2TextModel",
    "Molmo2VideoProcessor",
    "Molmo2VisionBackbone",
    "Molmo2VisionModel",
]
