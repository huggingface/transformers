import copy
import math

import numpy as np
import torch
from PIL import Image

from transformers.image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    get_size_dict,
)
from transformers.image_transforms import (
    convert_to_rgb,
    get_resize_output_image_size,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.processing_utils import ImagesKwargs
from transformers.utils import TensorType, logging


logger = logging.get_logger(__name__)


class HCXVisionImageProcessorKwargs(ImagesKwargs, total=False):
    """
    Additional keyword arguments for [`HyperClovaXVisionImageProcessor`].

    Args:
        anyres (`bool`):
            Whether to enable any-resolution image processing, which divides images into variable-size
            grids based on their aspect ratio.
        unpad (`bool`):
            When `anyres=True`, whether to remove padding visual tokens (tokens corresponding to
            purely padded regions) from the LLM input.
        num_queries_vis_abstractor (`int`):
            Number of visual query tokens per grid when using a resampler/abstractor projector.
        possible_resolutions (`List[List[int]]`):
            Pre-computed list of `[height, width]` resolution pairs for anyres processing,
            e.g. `[[336, 336], [336, 672], [672, 336]]`.
        patch_size (`int`):
            The patch size of the vision encoder (ViT).
        pad_to_square (`bool`):
            Whether to pad images to square before processing. If `False`, images retain their
            original aspect ratio and are passed through center crop before the ViT.
    """

    anyres: bool
    unpad: bool
    num_queries_vis_abstractor: int
    possible_resolutions: list[list[int]]
    patch_size: int
    pad_to_square: bool


def determine_possible_resolutions(anyres: bool, max_num_grids: int, grid_size: int, use_1x1_grid: bool = False):
    """
    Computes all valid `[height, width]` resolution combinations up to `max_num_grids` total grid cells.

    For example, with `max_num_grids=4` and `grid_size=336`, the valid grid combinations are
    [1×2, 1×3, 1×4, 2×1, 2×2, 3×1, 4×1] (1×1 excluded unless `use_1x1_grid=True`), yielding:

    >>> possible_resolutions = determine_possible_resolutions(anyres=True, max_num_grids=4, grid_size=336)
    >>> print(possible_resolutions)
    [[336, 672], [336, 1008], [336, 1344], [672, 336], [672, 672], [1008, 336], [1344, 336]]

    Args:
        anyres (`bool`): Whether any-resolution mode is enabled. Returns an empty list if `False`.
        max_num_grids (`int`): Maximum total number of grid cells (height_grids × width_grids ≤ max_num_grids).
        grid_size (`int`): The pixel size of each grid cell, typically equal to the vision encoder input size.
        use_1x1_grid (`bool`, *optional*, defaults to `False`):
            If `True`, includes the 1×1 grid (single patch) as a valid resolution.

    Returns:
        `list[list[int]]`: A list of `[height, width]` resolution pairs in pixels.
    """
    possible_resolutions = []
    if anyres:
        assert max_num_grids > 0
        for i in range(1, max_num_grids + 1):
            for j in range(1, max_num_grids + 1):
                if i == 1 and j == 1 and not use_1x1_grid:
                    continue
                if i * j <= max_num_grids:
                    possible_resolutions.append([i, j])

        possible_resolutions = [[ys * grid_size, xs * grid_size] for ys, xs in possible_resolutions]

    return possible_resolutions


def divide_to_grids(image: np.array, grid_size: int, input_data_format=None) -> list[np.array]:
    """Divides an image into a list of square patches, each of size `grid_size × grid_size` pixels."""
    grids = []
    height, width = get_image_size(image, channel_dim=input_data_format)
    for i in range(0, height, grid_size):
        for j in range(0, width, grid_size):
            if input_data_format == ChannelDimension.LAST:
                grid = image[i : i + grid_size, j : j + grid_size]
            else:
                grid = image[:, i : i + grid_size, j : j + grid_size]
            grids.append(grid)

    return grids


def pad(image: np.array, target_size: tuple, background_color=(127, 127, 127), input_data_format=None) -> np.array:
    """Pads `image` symmetrically (centered) to reach `target_size` (height, width) pixels,
    filling the surrounding area with `background_color`."""
    target_height, target_width = target_size
    height, width = get_image_size(image, channel_dim=input_data_format)

    # result = np.ones((target_height, target_width, image.shape[2]), dtype=image.dtype) * background_color
    result = np.empty((target_height, target_width, image.shape[2]), dtype=image.dtype)
    for i in range(image.shape[2]):
        result[..., i].fill(background_color[i])

    paste_x = (target_width - width) // 2
    paste_y = (target_height - height) // 2

    result[paste_y : paste_y + height, paste_x : paste_x + width, :] = image

    return result


def expand2square(
    image: np.array, bboxes_dict=None, background_color=(127, 127, 127), input_data_format=None
) -> np.array:
    """
    Pads `image` to a square canvas by centering it and filling the surrounding area with `background_color`.
    The image is placed at the center, with padding added symmetrically on either the sides (wide image)
    or top/bottom (tall image).

    Args:
        image (`np.ndarray`): Input image as a numpy array.
        bboxes_dict (`dict`, *optional*):
            Dictionary mapping category names to bounding box arrays of shape `(N, 4, 2)`, where
            each box is in `[[xtl, ytl], [xtr, ytr], [xbr, ybr], [xbl, ybl]]` format. Coordinates
            are adjusted to account for the padding offset.
        background_color (`tuple`): RGB fill color for the padding area.
        input_data_format (`ChannelDimension`, *optional*): Channel dimension format of the input image.
    """
    height, width = get_image_size(image, channel_dim=input_data_format)
    if width == height:
        return image, bboxes_dict
    elif width > height:
        # result = np.ones((width, width, image.shape[2]), dtype=image.dtype) * background_color
        result = np.empty((width, width, image.shape[2]), dtype=image.dtype)
        for i in range(image.shape[2]):
            result[..., i].fill(background_color[i])

        result[(width - height) // 2 : (width - height) // 2 + height, :] = image
        if bboxes_dict is not None:
            for key in bboxes_dict:
                bboxes_dict[key][:, :, 1] += (width - height) // 2
        return result, bboxes_dict
    else:
        # result = np.ones((height, height, image.shape[2]), dtype=image.dtype) * background_color
        result = np.empty((height, height, image.shape[2]), dtype=image.dtype)
        for i in range(image.shape[2]):
            result[..., i].fill(background_color[i])

        result[:, (height - width) // 2 : (height - width) // 2 + width] = image
        if bboxes_dict is not None:
            for key in bboxes_dict:
                bboxes_dict[key][:, :, 0] += (height - width) // 2
        return result, bboxes_dict


def resize_longside(
    image: np.array,
    size: int,
    resample: PILImageResampling = PILImageResampling.BICUBIC,
    data_format: str | ChannelDimension | None = None,
    input_data_format: str | ChannelDimension | None = None,
):
    """Resizes `image` so that its longest side equals `size` pixels, preserving the aspect ratio."""
    height, width = get_image_size(image, channel_dim=input_data_format)

    if width == height:
        target_height, target_width = size, size
    elif width > height:
        target_width = size
        target_height = math.ceil(height / width * size)
    else:
        target_width = math.ceil(width / height * size)
        target_height = size

    return resize(
        image,
        size=(target_height, target_width),
        resample=resample,
        data_format=data_format,
        input_data_format=input_data_format,
    )


def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    """From LLaVA-Next (https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/llava_next/image_processing_llava_next.py)
    Selects the best resolution from a list of possible resolutions based on the original size.
    This is done by calculating the effective and wasted resolution for each possible resolution.
    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).
        possible_resolutions (list):
            A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

    Returns:
        tuple: The best fit resolution in the format (height, width).
    """
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


def _get_local_grids_output_size(image: np.array, target_resolution: tuple, input_data_format=None):
    original_height, original_width = get_image_size(image, channel_dim=input_data_format)
    target_height, target_width = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    return new_height, new_width


def determine_anyres_num_vision_patches(
    num_grids,
    image_size,
    grid_size,
    patch_size,
    possible_resolutions,
    anyres=False,
    unpad=True,
    num_queries_vis_abstractor=0,
    num_queries_vis_abstractor_slow=0,
    video=False,
    first_last_frames_slow=False,
    is_first_or_last_frames=False,
):
    """Computes the number of visual tokens that an image or video frame will produce after preprocessing."""
    if not anyres:
        return num_queries_vis_abstractor if num_queries_vis_abstractor > 0 else (grid_size // patch_size) ** 2

    if num_queries_vis_abstractor > 0:
        num_patch_per_grid = int(num_queries_vis_abstractor**0.5)
    else:
        num_patch_per_grid = grid_size // patch_size

    num_global_per_grid = num_patch_per_grid

    # anyres는 global image가 있어서 2개 이상이지만, video에는 global image가 없어서, 1개가 들어올 수 있어서 주석 처리
    # assert num_grids > 1

    # patch 수 계산
    height, width = select_best_resolution(image_size, possible_resolutions)

    num_patch_height = (height // grid_size) * num_patch_per_grid
    num_patch_width = (width // grid_size) * num_patch_per_grid

    # local images
    if unpad:
        original_height, original_width = image_size

        original_aspect_ratio = original_width / original_height
        current_aspect_ratio = num_patch_width / num_patch_height

        if original_aspect_ratio > current_aspect_ratio:
            scale_factor = num_patch_width / original_width
            new_height = int(original_height * scale_factor)
            padding = (num_patch_height - new_height) // 2
            num_patch_height = num_patch_height - padding * 2
        else:
            scale_factor = num_patch_height / original_height
            new_width = int(original_width * scale_factor)
            padding = (num_patch_width - new_width) // 2
            num_patch_width = num_patch_width - padding * 2

        num_patches = num_patch_width * num_patch_height + num_patch_height
    else:
        num_patches = num_patch_width * num_patch_height

    # slow는 첫프레임 마지막 프레임 적용 전략일때는 첫프레임과 마지막 프레임만 적용
    if num_queries_vis_abstractor_slow > 0:
        if first_last_frames_slow:
            if is_first_or_last_frames:
                num_patches += num_queries_vis_abstractor_slow - num_queries_vis_abstractor
        else:
            num_patches += num_queries_vis_abstractor_slow - num_queries_vis_abstractor
        # slowfast 기능은 unpad False 에만 적용
        assert unpad is False

    # video 에는 global image 가 포함되지 않음
    if not video:
        num_patches += num_global_per_grid**2

    return num_patches


class HyperClovaXVisionImageProcessor(BaseImageProcessor):
    r"""
    Constructs a HyperClovaX Vision image processor. Based on [`CLIPImageProcessor`] with additional
    support for any-resolution (anyres) image processing inspired by LLaVA-Next, where images are
    divided into optimal grid configurations to preserve spatial information across different aspect ratios.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`.
        size (`dict`, *optional*, defaults to `{"shortest_edge": 336}`):
            Controls the size of the output image after resizing. Can be a dict with `"shortest_edge"` or
            `"height"` and `"width"` keys.
        anyres (`bool`, *optional*, defaults to `False`):
            If `True`, enables any-resolution image processing. The image is divided into variable-size
            grids based on its original aspect ratio to minimize information loss from resizing.
        unpad (`bool`, *optional*, defaults to `False`):
            If `True` and `anyres=True`, removes visual tokens corresponding to purely padded regions
            before passing features to the language model, reducing unnecessary computation.
        num_queries_vis_abstractor (`int`, *optional*, defaults to 0):
            Number of visual query tokens per grid when using a resampler/CAbstractor projector.
            Set to 0 when not using a resampler.
        possible_resolutions (`list`, *optional*, defaults to `[]`):
            Pre-computed list of `[height, width]` resolution pairs for anyres grid selection,
            e.g. `[[336, 336], [336, 672], [672, 336]]`.
        patch_size (`int`, *optional*, defaults to 14):
            The patch size of the vision encoder (ViT). Used to calculate the number of visual tokens.
        pad_to_square (`bool`, *optional*, defaults to `True`):
            If `True`, pads images to square before processing by centering them on a square canvas.
            If `False`, images retain their original aspect ratio and pass through center crop.
        resample (`PILImageResampling`, *optional*, defaults to `BICUBIC`):
            Resampling filter to use when resizing the image.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center-crop the image to the specified `crop_size`.
        crop_size (`dict`, *optional*, defaults to `{"height": 336, "width": 336}`):
            Size of the center crop. Defaults to 336×336.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified `rescale_factor`.
        rescale_factor (`float`, *optional*, defaults to `1/255`):
            Scale factor to use when rescaling the image pixel values.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image using the given `image_mean` and `image_std`.
        image_mean (`list[float]`, *optional*, defaults to CLIP mean `[0.48145466, 0.4578275, 0.40821073]`):
            Mean values for each channel used in normalization.
        image_std (`list[float]`, *optional*, defaults to CLIP std `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation values for each channel used in normalization.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB format before processing.
    """

    model_input_names = ["pixel_values"]
    valid_kwargs = HCXVisionImageProcessorKwargs

    def __init__(
        self,
        do_resize: bool = True,
        size: dict[str, int] | None = None,
        anyres: bool = False,
        unpad: bool = False,
        num_queries_vis_abstractor: int = 0,
        possible_resolutions: list = [],
        patch_size: int = 14,
        pad_to_square: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = True,
        crop_size: dict[str, int] | None = None,
        do_rescale: bool = True,
        rescale_factor: int | float = 1 / 255,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"shortest_edge": 336}
        size = get_size_dict(size, default_to_square=False)
        crop_size = crop_size if crop_size is not None else {"height": 336, "width": 336}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.do_resize = do_resize
        self.size = size
        self.anyres = anyres
        self.unpad = unpad
        self.num_queries_vis_abstractor = num_queries_vis_abstractor
        self.possible_resolutions = list(possible_resolutions)
        self.patch_size = patch_size
        self.pad_to_square = pad_to_square
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb

    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: str | ChannelDimension | None = None,
        input_data_format: str | ChannelDimension | None = None,
        **kwargs,
    ) -> np.ndarray:
        default_to_square = True
        if "shortest_edge" in size:
            size = size["shortest_edge"]
            default_to_square = False
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")

        output_size = get_resize_output_image_size(
            image,
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )

        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def _resize_for_local_grids(
        self, image: np.array, target_resolution: tuple, resample, input_data_format: ChannelDimension
    ) -> np.array:
        new_height, new_width = _get_local_grids_output_size(image, target_resolution, input_data_format)

        # Resize the image
        resized_image = resize(image, (new_height, new_width), resample=resample, input_data_format=input_data_format)

        return resized_image

    def _pad_for_patching(
        self, image: np.array, target_resolution: tuple, input_data_format: ChannelDimension
    ) -> np.array:
        """
        Pad an image to a target resolution while maintaining aspect ratio.
        """
        target_height, target_width = target_resolution

        background_color = tuple(int(x * 255) for x in self.image_mean)
        padded_image = pad(
            image,
            target_size=(target_height, target_width),
            background_color=background_color,
            input_data_format=input_data_format,
        )

        return padded_image

    def get_image_grids(
        self,
        image: np.array,
        possible_resolutions,
        grid_size: int,
        resample: PILImageResampling,
        data_format: ChannelDimension,
        input_data_format: ChannelDimension,
    ) -> list[np.array]:
        if not isinstance(possible_resolutions, list):
            raise ValueError("possible_resolutions must be a list of possible resolutions.")

        image_size = get_image_size(image, channel_dim=input_data_format)
        best_resolution = select_best_resolution(image_size, possible_resolutions)
        resized_image = self._resize_for_local_grids(
            image, best_resolution, resample=resample, input_data_format=input_data_format
        )
        padded_image = self._pad_for_patching(resized_image, best_resolution, input_data_format=input_data_format)
        local_grids = divide_to_grids(padded_image, grid_size=grid_size, input_data_format=input_data_format)

        # make sure that all patches are in the input data format
        local_grids = [
            to_channel_dimension_format(grid, channel_dim=data_format, input_channel_dim=input_data_format)
            for grid in local_grids
        ]

        return local_grids

    def _preprocess(
        self,
        images: ImageInput,
        do_resize: bool = False,
        size: dict[str, int] | None = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = False,
        crop_size: int | None = None,
        do_rescale: bool = False,
        rescale_factor: float | None = None,
        do_normalize: bool = False,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        data_format: ChannelDimension | None = ChannelDimension.FIRST,
        input_data_format: str | ChannelDimension | None = None,
    ) -> Image.Image:
        images = make_list_of_images(images)

        if do_resize:
            images = [
                self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
                for image in images
            ]

        if do_center_crop:
            images = [
                self.center_crop(image=image, size=crop_size, input_data_format=input_data_format) for image in images
            ]

        if do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                for image in images
            ]

        if do_normalize:
            images = [
                self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                for image in images
            ]

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        return images

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = False,
        size: dict[str, int] | None = None,
        anyres: bool = False,
        unpad: bool = False,
        video: bool = False,
        num_queries_vis_abstractor: int | None = None,
        possible_resolutions: list = [],
        patch_size: int | None = None,
        pad_to_square: bool = False,
        resample: PILImageResampling = None,
        do_center_crop: bool = False,
        crop_size: int | None = None,
        do_rescale: bool = False,
        rescale_factor: float | None = None,
        do_normalize: bool = False,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool = False,
        return_tensors: str | TensorType | None = None,
        data_format: ChannelDimension | None = ChannelDimension.FIRST,
        input_data_format: str | ChannelDimension | None = None,
        return_dummy_image: bool = False,
        num_queries_vis_abstractor_slow: int = 0,
        first_last_frames_slow: bool = False,
        is_first_or_last_frames: bool = False,
    ):
        """
        Preprocesses a batch of images into pixel value tensors, original image sizes, and visual token counts.

        Returns:
            [`BatchFeature`] containing:
            - **pixel_values** -- List of tensors with shape `(num_grids, channels, height, width)`.
              `num_grids` is 1 for standard processing, or `1 + num_local_grids` for anyres images
              (first entry is the global image thumbnail).
            - **image_sizes** -- List of `{"width": W, "height": H}` dicts with the original image
              dimensions before any preprocessing.
            - **vision_query_lengths** -- List of `int` indicating the number of visual tokens each
              image produces when passed to the language model.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, param_name="size", default_to_square=False)
        anyres = anyres if anyres is not None else self.anyres
        unpad = unpad if unpad is not None else self.unpad
        if video:
            unpad = False
        num_queries_vis_abstractor = (
            num_queries_vis_abstractor if num_queries_vis_abstractor is not None else self.num_queries_vis_abstractor
        )
        possible_resolutions = possible_resolutions if possible_resolutions is not None else self.possible_resolutions
        patch_size = patch_size if patch_size is not None else self.patch_size
        pad_to_square = pad_to_square if pad_to_square is not None else self.pad_to_square
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size, param_name="crop_size", default_to_square=True)
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        if return_dummy_image:
            images = Image.new("RGB", (224, 224), (0, 0, 0))

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

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

        new_images = []
        image_sizes = [get_image_size(image, channel_dim=input_data_format) for image in images]
        vision_query_lengths = []

        assert crop_size["height"] == crop_size["width"]

        # global image 의 padding 연산은, image original width, height 가 클 때 bottleneck 이 될 수 있음
        # 장축의 길이를 size["shortest_edge"] 로 resize 를 먼저 한 뒤에, padding
        if anyres:
            anyres_global_images = copy.deepcopy(images)
            if pad_to_square:
                background_color = tuple(int(x * 255) for x in self.image_mean)
                anyres_global_images = [
                    resize_longside(copy.deepcopy(image), size["shortest_edge"], resample, input_data_format)
                    for image in anyres_global_images
                ]
                anyres_global_images = [
                    expand2square(image, background_color=background_color, input_data_format=input_data_format)[0]
                    for image in anyres_global_images
                ]
            else:
                anyres_global_images = [
                    self.resize(
                        image=image,
                        size={"height": size["shortest_edge"], "width": size["shortest_edge"]},
                        resample=resample,
                        input_data_format=input_data_format,
                    )
                    for image in anyres_global_images
                ]
        else:
            anyres_global_images = [None for _ in range(len(images))]
            if pad_to_square:
                background_color = tuple(int(x * 255) for x in self.image_mean)
                images = [
                    resize_longside(image, size["shortest_edge"], resample, input_data_format) for image in images
                ]
                images = [
                    expand2square(image, background_color=background_color, input_data_format=input_data_format)[0]
                    for image in images
                ]

        for image, anyres_global_image, image_size in zip(images, anyres_global_images, image_sizes):
            if anyres:
                # convert image into a list of grids
                # we intentially use the same data format as the input data format
                image_grids = self.get_image_grids(
                    image,
                    possible_resolutions,
                    grid_size=crop_size["height"],
                    resample=resample,
                    data_format=input_data_format,
                    input_data_format=input_data_format,
                )
                # video 에 대해서는 global image (thumbnail) 를 사용하지 않음
                if not video:
                    image_grids = [anyres_global_image] + image_grids
            else:
                image_grids = [image]

            pixel_values = self._preprocess(
                image_grids,
                do_resize=do_resize,
                size=size,
                resample=resample,
                do_center_crop=do_center_crop,
                crop_size=crop_size,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
                input_data_format=input_data_format,
            )

            pixel_values = np.array(pixel_values)
            new_images.append(pixel_values)

            num_grids = pixel_values.shape[0]

            vision_query_length = determine_anyres_num_vision_patches(
                num_grids=num_grids,
                image_size=image_size,
                grid_size=crop_size["height"],
                patch_size=patch_size,
                possible_resolutions=possible_resolutions,
                anyres=anyres,
                unpad=unpad,
                num_queries_vis_abstractor=num_queries_vis_abstractor,
                num_queries_vis_abstractor_slow=num_queries_vis_abstractor_slow,
                video=video,
                first_last_frames_slow=first_last_frames_slow,
                is_first_or_last_frames=is_first_or_last_frames,
            )

            vision_query_lengths.append(vision_query_length)

        if return_dummy_image:
            vision_query_lengths = []

        data = {
            "pixel_values": [torch.tensor(new_image) for new_image in new_images],
            "image_sizes": [{"width": image_size[1], "height": image_size[0]} for image_size in image_sizes],
            "vision_query_lengths": vision_query_lengths,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["HyperClovaXVisionImageProcessor"]
