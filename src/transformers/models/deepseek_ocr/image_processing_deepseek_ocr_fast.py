import math
from typing import Optional, Sequence, Union

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    BatchFeature,
    Unpack,
)
from ...image_utils import ImageInput, SizeDict
from ...processing_utils import ImagesKwargs
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class DeepseekOcrImageProcessorKwargs(ImagesKwargs, total=False):
    size: dict
    do_resize: bool
    do_rescale: bool
    rescale_factor: float
    do_normalize: bool
    image_mean: Union[float, Sequence[float]]
    image_std: Union[float, Sequence[float]]
    return_tensors: Union[str, TensorType]
    disable_grouping: bool
    do_pad: bool
    base_size: int
    patch_size_side: int
    patch_size: int
    downsample_ratio: int
    max_tiles: int


class DeepseekOcrImageProcessorFast(BaseImageProcessorFast):
    """
    Torch-based image processor used for DeepSeek OCR.

    It prepares one global view per image and a variable number of local crops.
    Each view is resized to the backbone ``base_size`` so the vision tower can
    be executed in batches on GPU.
    """

    resample = InterpolationMode.BICUBIC
    do_resize = False
    do_rescale = True
    do_normalize = True
    do_pad = True
    size = {"height": 1024, "width": 1024}
    image_mean = (0.5, 0.5, 0.5)
    image_std = (0.5, 0.5, 0.5)
    base_size = 1024
    patch_size_side = 640
    patch_size = 16
    downsample_ratio = 4
    max_tiles = 9
    valid_kwargs = DeepseekOcrImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[DeepseekOcrImageProcessorKwargs]):
        super().__init__(**kwargs)

        self.base_size = self._resolve_scalar(self.base_size)
        self.patch_size_side = self._resolve_scalar(self.patch_size_side)
        self.patch_size = int(self.patch_size)
        self.downsample_ratio = int(self.downsample_ratio)
        self.max_tiles = int(self.max_tiles)
        self.do_pad = getattr(self, "do_pad", True)

        self._vision_grid = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

    @staticmethod
    def _resolve_scalar(value: Union[int, dict, SizeDict]) -> int:
        if isinstance(value, SizeDict):
            return int(value.height if value.height is not None else value.width)
        if isinstance(value, dict):
            height = value.get("height")
            if height is not None:
                return int(height)
            width = value.get("width")
            if width is not None:
                return int(width)
        return int(value)

    @staticmethod
    def _square_pad(image: torch.Tensor, fill: torch.Tensor) -> torch.Tensor:
        c, h, w = image.shape
        if h == w:
            return image
        size = max(h, w)
        padded = torch.broadcast_to(fill, (c, size, size)).clone()
        top = (size - h) // 2
        left = (size - w) // 2
        padded[:, top : top + h, left : left + w] = image
        return padded

    @staticmethod
    def _find_best_grid(width: int, height: int, max_tiles: int) -> tuple[int, int]:
        aspect_ratio = width / height
        candidates = [
            (i, j)
            for tiles in range(2, max_tiles + 1)
            for i in range(1, tiles + 1)
            for j in range(1, tiles + 1)
            if 1 < i * j <= max_tiles
        ]
        best_ratio = (1, 1)
        best_diff = float("inf")
        for cols, rows in candidates:
            ratio = cols / rows
            diff = abs(aspect_ratio - ratio)
            if diff < best_diff or (diff == best_diff and cols * rows > best_ratio[0] * best_ratio[1]):
                best_diff = diff
                best_ratio = (cols, rows)
        return best_ratio

    def _dynamic_preprocess(self, image: torch.Tensor, patch_side: int) -> tuple[list[torch.Tensor], tuple[int, int]]:
        c, h, w = image.shape
        if max(h, w) <= patch_side:
            return [], (1, 1)

        cols, rows = self._find_best_grid(w, h, self.max_tiles)
        if cols == 1 and rows == 1:
            return [], (1, 1)

        target_w = cols * patch_side
        target_h = rows * patch_side
        resized = F.resize(
            image,
            size=[target_h, target_w],
            interpolation=self.resample,
            antialias=True,
        )

        crops: list[torch.Tensor] = []
        for row in range(rows):
            for col in range(cols):
                top = row * patch_side
                left = col * patch_side
                crops.append(resized[:, top : top + patch_side, left : left + patch_side])
        return crops, (cols, rows)

    def _prepare_views(
        self,
        image: torch.Tensor,
        interpolation: Optional[InterpolationMode],
    ) -> tuple[list[torch.Tensor], tuple[int, int], tuple[int, int]]:
        image = image.to(dtype=torch.float32)
        c, h, w = image.shape
        original_size = (h, w)

        if self.do_pad:
            background = torch.tensor(self.image_mean, dtype=torch.float32, device=image.device).view(-1, 1, 1)
            padded = self._square_pad(image, background.to(dtype=image.dtype))
        else:
            padded = image
        global_view = F.resize(
            padded,
            size=[self.base_size, self.base_size],
            interpolation=interpolation or self.resample,
            antialias=True,
        )

        local_crops, (cols, rows) = self._dynamic_preprocess(image, self.patch_size_side)
        resized_locals: list[torch.Tensor] = []
        if cols * rows > 1 and local_crops:
            for patch in local_crops:
                resized = F.resize(
                    patch,
                    size=[self.base_size, self.base_size],
                    interpolation=interpolation or self.resample,
                    antialias=True,
                )
                resized_locals.append(resized)

        views = [global_view] + resized_locals
        spatial_crop = (cols, rows) if resized_locals else (1, 1)
        return views, spatial_crop, original_size

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional[InterpolationMode],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, Sequence[float]]],
        image_std: Optional[Union[float, Sequence[float]]],
        do_pad: Optional[bool],
        pad_size: Optional[SizeDict],
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        if not images:
            raise ValueError("No images were provided to the processor.")

        flat_images: list[torch.Tensor] = []
        for img in images:
            if isinstance(img, (list, tuple)):
                flat_images.extend(img)
            else:
                flat_images.append(img)

        if not flat_images:
            raise ValueError("No valid images were provided to the processor.")

        device = flat_images[0].device
        views_per_image: list[list[torch.Tensor]] = []
        spatial_crops: list[tuple[int, int]] = []
        original_sizes: list[tuple[int, int]] = []

        interp = interpolation or self.resample
        for image in flat_images:
            views, crop_shape, orig_size = self._prepare_views(image, interp)
            views_per_image.append(views)
            spatial_crops.append(crop_shape)
            original_sizes.append(orig_size)

        max_views = max(len(sample_views) for sample_views in views_per_image)
        channel = views_per_image[0][0].shape[0]
        pixel_values = torch.zeros(
            (len(images), max_views, channel, self.base_size, self.base_size),
            dtype=torch.float32,
            device=device,
        )
        valid_counts = torch.zeros(len(images), dtype=torch.long, device=device)

        for idx, sample_views in enumerate(views_per_image):
            count = len(sample_views)
            valid_counts[idx] = count
            stacked = torch.stack(sample_views, dim=0)
            pixel_values[idx, :count] = stacked

        if do_rescale:
            pixel_values = pixel_values * rescale_factor

        if do_normalize:
            mean = torch.as_tensor(image_mean if image_mean is not None else self.image_mean, dtype=pixel_values.dtype, device=pixel_values.device)
            std = torch.as_tensor(image_std if image_std is not None else self.image_std, dtype=pixel_values.dtype, device=pixel_values.device)
            pixel_values = (pixel_values - mean.view(1, 1, -1, 1, 1)) / std.view(1, 1, -1, 1, 1)

        # zero-out padded views so they are ignored downstream
        if max_views > 1:
            view_ids = torch.arange(max_views, device=device)
            mask = view_ids.unsqueeze(0) >= valid_counts.unsqueeze(1)
            pixel_values = pixel_values.masked_fill(mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 0)

        if pixel_values.dtype != torch.bfloat16:
            pixel_values = pixel_values.to(dtype=torch.bfloat16)

        data = {
            "pixel_values": pixel_values,
            "image_spatial_crop": torch.tensor(spatial_crops, dtype=torch.long, device=device),
            "image_sizes": torch.tensor(original_sizes, dtype=torch.long, device=device),
        }

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["DeepseekOcrImageProcessorFast"]
