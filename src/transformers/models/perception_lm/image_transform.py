# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
from functools import reduce
from logging import getLogger
from typing import Any, Callable, Tuple, Union, Sequence

import numpy as np
import torch
import torchvision.transforms as tv
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import ToPILImage, PILToTensor


logger = getLogger()


MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


"""
Resize the image to the given size. Supports both PIL images and torch.Tensor.
If the image is a tensor, it's supposed to be a batch of images with shape (B, C, H, W) and dtype uint8.
If use_pil_resize is True, the images will be resized using PIL implementation of interpolation.
"""


def _resize(
    image: Union[Image.Image, torch.Tensor],
    size: Sequence[int],
    use_pil_resize: bool = True,
) -> Union[Image.Image, torch.Tensor]:
    if isinstance(image, torch.Tensor) and use_pil_resize:
        ims = []
        for im in image:
            im = ToPILImage()(im)
            im = F.resize(im, size, interpolation=InterpolationMode.BICUBIC)
            ims.append(PILToTensor()(im))
        return torch.stack(ims, dim=0)
    else:
        return F.resize(image, size, interpolation=InterpolationMode.BICUBIC)


def get_image_transform(
    vision_input_type: str = "vanilla",
    image_res: int = 336,
    max_num_tiles: int = 1,
    normalize_img: bool = True,
) -> Tuple[Callable, int]:

    if vision_input_type == "thumb+tile":
        transforms = VariableSizeImageTransform(
            size=image_res,
            max_num_tiles=max_num_tiles,
            normalize_img=normalize_img,
            use_thumbnail="before",
        )
    else:
        transforms = ImageTransform(
            size=image_res,
            normalize_img=normalize_img,
        )

    logger.info(
        f"Initalized transforms with: vision_input_type: '{vision_input_type}' and max_num_tiles: {max_num_tiles}."
    )

    return transforms


class ImageTransform(object):
    """
    Image transform will resize the longer edge to a given size and pad the shorter edge with mean pixel value of the image.
    """

    def __init__(
        self,
        size: int = 336,
        normalize_img: bool = True,
    ) -> None:
        self.size = size
        self._mean = MEAN
        self._std = STD
        self.normalize_img = normalize_img

        logger.info(f"ImageTransform size: {self.size}")

        self.to_tensor = tv.ToTensor()
        self.normalize = (
            tv.Normalize(
                mean=self._mean,
                std=self._std,
                inplace=True,
            )
            if normalize_img
            else lambda x: x
        )

    def to_dict(self):
        return {
            "size": self.size,
            "normalize_img": self.normalize_img,
        }

    def __call__(self, image: Union[Image.Image, torch.Tensor]):
        if isinstance(image, Image.Image):
            w, h = image.size
        else:
            h, w = image.shape[-2:]

        image = _resize(
            image,
            (self.size, self.size),
            use_pil_resize=False,
        )
        if isinstance(image, Image.Image):
            image = self.to_tensor(image)
        else:
            image = F.convert_image_dtype(image, torch.float32)
        image = self.normalize(image)

        # Add chunk dim to make it compatible with existing dataloaders
        image = image.view(1, -1, 3, self.size, self.size)
        return image, (w, h)


class VariableSizeImageTransform(object):
    """
    The variable size image transform will resize the image dynamically
    based on the image aspect ratio and the number of image chunks we allow.

    The algorithm will not upsample low-res images to fit a certain aspect
    ratio, because that leads to a significant degradation in image quality.

    For example, if an input image is of size 300x800, and we want to allow
    a maximum of 16 image chunks, it will find the closest aspect ratio that
    is allowed within 16 image chunks, i.e., 2:5 = 2 horizontal patches and
    5 vertical patches, giving a total of 10 chunks.

    The image will then be resized to products of the base size (default is
    224px because MetaCLIP takes that), so in this case it will  be resized to
    2*224:5*224 = 448:1120, where we maintain the original aspect ratio and
    pad with the mean value for the rest. This approach minimizes the amount
    of padding required for any arbitrary resolution.

    The final output will therefore be of shape (11, 3, 224, 224), where 10
    patches are coming from the resizing and chunking, and the first patch
    is a downsampled version of the image that preserves aspect ratios.
    """

    def __init__(
        self,
        size: int = 336,
        normalize_img: bool = True,
        max_num_tiles: int = 1,
        use_thumbnail: str = "no",
        area_limit: bool = False,
    ) -> None:
        self.size = size
        self._mean = MEAN
        self._std = STD

        logger.info(f"VariableSizeImageTransform size: {self.size}")

        self.to_tensor = tv.ToTensor()
        self.normalize = (
            tv.Normalize(
                mean=self._mean,
                std=self._std,
                inplace=True,
            )
            if normalize_img
            else lambda x: x
        )
        self.area_limit = area_limit
        self.max_num_tiles = max_num_tiles
        self.use_thumbnail = use_thumbnail
        self.normalize_img = normalize_img
        if self.use_thumbnail != "no":
            self.thumbnail_transform = ImageTransform(
                size=self.size,
                normalize_img=normalize_img,
            )

    def to_dict(self):
        return {
            "size": self.size,
            "normalize_img": self.normalize_img,
            "max_num_tiles": self.max_num_tiles,
            "use_thumbnail": self.use_thumbnail,
        }

    @staticmethod
    def _factors(n: int):
        """Return all factors of a number."""
        return set(
            reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
            )
        )

    def _find_supported_aspect_ratios(self):
        """
        This function computes all the allowed aspect ratios for a fixed
        number of input chunks.

        For example, with `num_tiles=5`, it will return:
        {
            0.2: [(1, 5)],
            5.0: [(5, 1)],
            0.25: [(1, 4)],
            1.0: [(2, 2), (1, 1)],
            4.0: [(4, 1)],
            0.3333333333333333: [(1, 3)],
            3.0: [(3, 1)],
            0.5: [(1, 2)],
            2.0: [(2, 1)]
        }
        """
        asp_dict = {}
        for chunk_size in range(self.max_num_tiles, 0, -1):
            _factors = sorted(VariableSizeImageTransform._factors(chunk_size))
            _asp_ratios = [(x, chunk_size // x) for x in _factors]
            for ratio in _asp_ratios:
                k = ratio[0] / ratio[1]
                if k not in asp_dict:
                    asp_dict[k] = [ratio]
                else:
                    asp_dict[k].append(ratio)
        return asp_dict

    def _find_closest_aspect_ratio(self, img_width: int, img_height: int) -> Tuple:
        """
        Given an image width, height and target number of chunks
        this function will find the closest supported aspect ratio.
        """
        tgt_ar = img_width / img_height
        asp_dict = self._find_supported_aspect_ratios()
        cl_d, cl_p = 1e23, None
        if tgt_ar >= 1:
            cl_p = min(
                [k for k in asp_dict.keys() if k <= tgt_ar],
                key=lambda x: abs(x - tgt_ar),
            )
            v = asp_dict[cl_p]
            # select width
            widths = [(idx, self.size * vv[0]) for idx, vv in enumerate(v)]
            tgt_idx = max(widths, key=lambda x: x[1])[0]
        else:
            cl_p = min(
                [k for k in asp_dict.keys() if k > tgt_ar],
                key=lambda x: abs(1 / x - 1 / tgt_ar),
            )
            v = asp_dict[cl_p]
            # select height
            heights = [(idx, self.size * vv[1]) for idx, vv in enumerate(v)]
            tgt_idx = max(heights, key=lambda x: x[1])[0]
        out = v[tgt_idx]
        return out

    def _pad(
        self, image: Union[Image.Image, torch.Tensor], new_width: int, new_height: int
    ) -> Union[Image.Image, torch.Tensor]:
        if isinstance(image, Image.Image):
            new_im = Image.new(mode="RGB", size=(new_width, new_height), color=(0, 0, 0))  # type: ignore
            new_im.paste(image)
            return new_im
        else:
            return F.pad(
                image, (0, 0, new_width - image.shape[-1], new_height - image.shape[-2])
            )

    def _split(self, image: torch.Tensor, ncw: int, nch: int) -> torch.Tensor:
        # Split image into number of required tiles (width x height)
        batch_size, num_channels, height, width = image.size()
        image = image.view(
            batch_size, num_channels, nch, height // nch, ncw, width // ncw
        )
        # Permute dimensions to reorder the axes
        image = image.permute(0, 2, 4, 1, 3, 5).contiguous()
        # Reshape into the desired output shape (batch_size * 4, num_channels, width/2, height/2)
        image = image.view(
            batch_size, ncw * nch, num_channels, height // nch, width // ncw
        )
        return image

    def _get_image_height_width(
        self, image_width: int, image_height: int, target_width: int, target_height: int
    ) -> Tuple[int, int]:
        """
        Given image width, height and target width, height for the canvas, return the dimensions of how the image would be resized
        with aspect ratio preservation.
        """
        scale = image_width / image_height

        if scale > 1.0:
            # Width is larger than height

            # Rescaling factor is the minimum of the two scaling factors. Else one side would be outside of the canvas.
            rescaling_factor = min(
                target_width / image_width, target_height / image_height
            )

            # Set new width to target width and height to the rescaled height.
            new_w = rescaling_factor * image_width
            new_h = math.floor(new_w / scale)

        else:
            # Height is larger than width

            # Rescaling factor is the minimum of the two scaling factors. Else one side would be outside of the canvas.
            rescaling_factor = min(
                target_width / image_width, target_height / image_height
            )

            # Set new height to target height and width to the rescaled width.
            new_h = rescaling_factor * image_height
            new_w = math.floor(new_h * scale)

        return new_w, new_h

    def _fit_image_to_canvas(
        self, img_width: int, img_height: int, area_limit: bool
    ) -> Any:
        """
        Given an image width, height and target number of chunks this function will see if the image
        can be fit into any of the canvases that can be build from arranging the tiles in a grid.
        If the image can be fit onto several canvases, it will return the canvas where the shorter edge
        of the image will be largest.

        If area_limit is set to True, the tie-breaking prefers the canvas where area is less than 2x the original area.
        """
        # Initialize the optimal canvas to None. If no canvas is found where image fits, function returns None.
        optimal_canvas = None
        optimal_image_width_height = None

        scale = img_width / img_height

        # Gather all potential supported image resolutions and iterate through them to find best match
        potential_arrangements = [
            item
            for sublist in self._find_supported_aspect_ratios().values()
            for item in sublist
        ]
        for n_w, n_h in potential_arrangements:
            # Compute the canvas size
            canvas_width, canvas_height = n_w * self.size, n_h * self.size

            # Check if image can fit into the canvas without downsampling
            if canvas_width >= img_width and canvas_height >= img_height:
                # If we did not find a good canvas yet, we will use the current one
                if optimal_canvas is None:
                    # Set optimal canvas and determine the actual image height and width in the canvas with aspect ratio preserving resampling
                    optimal_canvas = (n_w, n_h)
                    optimal_image_width_height = self._get_image_height_width(
                        image_width=img_width,
                        image_height=img_height,
                        target_width=n_w * self.size,
                        target_height=n_h * self.size,
                    )
                else:
                    # If we already found an optimal canvas before, we will check if the shorter edge of the image will be larger than the current optimal canvas.
                    # This means we can potentially upsample the image resolution which is beneficial to performance.
                    image_width_height = self._get_image_height_width(
                        image_width=img_width,
                        image_height=img_height,
                        target_width=n_w * self.size,
                        target_height=n_h * self.size,
                    )
                    if area_limit:
                        # Prioritize aspect ratio, and choose best within area limit when tied.
                        curr_scale = image_width_height[0] / image_width_height[1]
                        optim_scale = (
                            optimal_image_width_height[0]
                            / optimal_image_width_height[1]
                        )
                        if abs(scale - curr_scale) < abs(scale - optim_scale):
                            # 1. optimize aspect ratio
                            optimal_canvas = (n_w, n_h)
                            optimal_image_width_height = image_width_height
                        elif abs(scale - curr_scale) == abs(scale - optim_scale):
                            # 2. optimize area
                            if (
                                image_width_height[0] * image_width_height[1]
                                < 2 * img_width * img_height
                            ):
                                # 2.1 area is less than 2x the original area
                                optimal_canvas = (n_w, n_h)
                                optimal_image_width_height = image_width_height
                    else:
                        # NOTE: L3V dynamid tiling. Priortize biggest canvas.
                        if (
                            scale < 1.0
                            and (image_width_height[0] >= optimal_image_width_height[0])
                        ) or (
                            scale >= 1.0
                            and (image_width_height[1] >= optimal_image_width_height[1])
                        ):
                            optimal_canvas = (n_w, n_h)
                            optimal_image_width_height = image_width_height
        return optimal_canvas

    def __call__(
        self, image: Union[Image.Image, torch.Tensor] = None
    ) -> Tuple[Any, Any]:
        if self.use_thumbnail != "no":
            thumbnail = self.thumbnail_transform(image)[0]

        if isinstance(image, Image.Image):
            w, h = image.size
        else:
            h, w = image.shape[-2:]

        # Check if the image can be fit to the canvas without downsampling
        ar = self._fit_image_to_canvas(
            img_width=w, img_height=h, area_limit=self.area_limit
        )
        if ar is None:
            # If we did not find a canvas, we have to find the closest aspect ratio and downsample the image
            ar = self._find_closest_aspect_ratio(img_width=w, img_height=h)

        image = _resize(
            image,
            (ar[1] * self.size, ar[0] * self.size),  # (h, w)
        )
        image = self._pad(image, ar[0] * self.size, ar[1] * self.size)

        if isinstance(image, Image.Image):
            image = self.to_tensor(image)
        else:
            image = F.convert_image_dtype(image, torch.float32)

        image = self.normalize(image)
        image = self._split(image, ar[0], ar[1])  # type: ignore
        if self.use_thumbnail == "before":
            image = torch.cat((thumbnail, image), dim=1)
        elif self.use_thumbnail == "after":
            image = torch.cat((image, thumbnail), dim=1)
        elif self.use_thumbnail == "both":
            image = torch.cat((thumbnail, image, thumbnail), dim=1)
        return image, ar
