from typing import Dict, List, Optional, Union

from PIL import Image

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


class InternVL2_5ImageProcessor(BaseImageProcessor):
    """
    Image processor for InternVL models. Handles dynamic tiling and preprocessing of images.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified size.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 448, "width": 448}`):
            Size of output image after resizing.
        dynamic_size (`Dict`, *optional*):
            Size of output image after dynamic tiling.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale factor.
        rescale_factor (`float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_MEAN`):
            Mean to use if normalizing the image.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STD`):
            Standard deviation to use if normalizing the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        min_tiles (`int`, *optional*, defaults to 1):
            Minimum number of tiles to split image into.
        max_tiles (`int`, *optional*, defaults to 12):
            Maximum number of tiles to split image into.
        use_thumbnail (`bool`, *optional*, defaults to `True`):
            Whether to include a thumbnail of the full image in addition to tiles.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        downsample_ratio (`float`, *optional*, defaults to 0.5):
            The downsample ratio for dynamic tiling.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        dynamic_size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_tiles: int = 1,
        max_tiles: int = 12,
        use_thumbnail: bool = True,
        patch_size: int = 14,
        downsample_ratio: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        size = size if size is not None else {"height": 448, "width": 448}
        size = get_size_dict(size, default_to_square=False)
        self.size = size
        dynamic_size = dynamic_size if dynamic_size is not None else {"height": 448, "width": 448}
        dynamic_size = get_size_dict(dynamic_size, default_to_square=False)
        self.dynamic_size = dynamic_size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_convert_rgb = do_convert_rgb

        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.num_image_token = int((dynamic_size // self.patch_size) ** 2 * (self.downsample_ratio**2))
        self.min_tiles = min_tiles
        self.max_tiles = max_tiles
        self.use_thumbnail = use_thumbnail

    def preprocess(
        self,
        images: Union[ImageInput, List[ImageInput]],
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        dynamic_size: Optional[Dict[str, int]] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        min_tiles: Optional[int] = None,
        max_tiles: Optional[int] = None,
        use_thumbnail: Optional[bool] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)
        dynamic_size = dynamic_size if dynamic_size is not None else self.dynamic_size
        dynamic_size = get_size_dict(dynamic_size, default_to_square=False)
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        min_tiles = min_tiles if min_tiles is not None else self.min_tiles
        max_tiles = max_tiles if max_tiles is not None else self.max_tiles
        use_thumbnail = use_thumbnail if use_thumbnail is not None else self.use_thumbnail

        images = make_flat_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # Convert to PIL Image for tiling
        images = [
            Image.fromarray(to_channel_dimension_format(to_numpy_array(image), ChannelDimension.LAST))
            if not isinstance(image, Image.Image)
            else image
            for image in images
        ]

        # Dynamic tiling
        tiled_images = []
        num_tiles = []
        for image in images:
            tiles = dynamic_preprocess(
                image,
                min_num=self.min_tiles,
                max_num=self.max_tiles,
                use_thumbnail=self.use_thumbnail,
                image_size=dynamic_size["height"],  # Assuming square tiles
            )
            tiled_images.extend(tiles)
            num_tiles.append(len(tiles))

        # Convert back to numpy for standard processing
        tiled_images = [to_numpy_array(image) for image in tiled_images]

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(tiled_images[0])

        # Convert images to numpy arrays with correct channel format
        tiled_images = [to_channel_dimension_format(image, ChannelDimension.FIRST) for image in tiled_images]

        if do_resize:
            tiled_images = [
                resize(
                    image=image,
                    size=(size["height"], size["width"]),
                    resample=resample.value if hasattr(resample, "value") else resample,
                    data_format=ChannelDimension.FIRST,
                    input_data_format=ChannelDimension.FIRST,
                )
                for image in tiled_images
            ]

        if do_rescale:
            tiled_images = [
                self.rescale(
                    image=image,
                    scale=rescale_factor,
                    data_format=ChannelDimension.FIRST,
                    input_data_format=ChannelDimension.FIRST,
                )
                for image in tiled_images
            ]

        if do_normalize:
            tiled_images = [
                self.normalize(
                    image=image,
                    mean=image_mean,
                    std=image_std,
                    data_format=ChannelDimension.FIRST,
                    input_data_format=ChannelDimension.FIRST,
                )
                for image in tiled_images
            ]

        tiled_images = [
            to_channel_dimension_format(image, data_format, ChannelDimension.FIRST)
            if data_format != ChannelDimension.FIRST
            else image
            for image in tiled_images
        ]

        data = {"pixel_values": tiled_images, "num_patches": num_tiles}
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["InternVL2_5ImageProcessor"]
