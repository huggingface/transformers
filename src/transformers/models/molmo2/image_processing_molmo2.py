"""Image processor class for Molmo2"""
from typing import Optional, Union
import numpy as np
import einops
import torch
import torchvision.transforms

from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    make_flat_list_of_images,
    valid_images,
    to_numpy_array,
)
from ...image_transforms import convert_to_rgb
from ...processing_utils import ImagesKwargs
from ...image_processing_utils import BaseImageProcessor, get_size_dict
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


def normalize_image(
    image: np.ndarray,
    image_mean: list[float],
    image_std: list[float],
) -> np.ndarray:
    image -= np.array(image_mean, dtype=np.float32)[None, None, :]
    image /= np.array(image_std, dtype=np.float32)[None, None, :]
    return image


def resize_image(
    image: np.ndarray,
    desired_output_size: list[int],
    resample: PILImageResampling,
) -> np.ndarray:
    image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    dtype = image.dtype
    if torch.is_floating_point(image):
        in_min = 0.0
        in_max = 1.0
        resized = torchvision.transforms.Resize(
            desired_output_size,
            resample,
            antialias=False,
        )(image)
        resized = torch.clip(resized, 0.0, 1.0).to(dtype)
    else:
        assert image.dtype == torch.uint8, "SigLIP expects float images or uint8 images, but got {}".format(image.dtype)
        in_min = 0.0
        in_max = 255.0
        resized = torchvision.transforms.Resize(
            desired_output_size,
            resample,
            antialias=False,
        )(image)
        resized = torch.clip(resized, 0, 255).to(dtype)

    resized = resized.to(torch.float32)
    resized = (resized - in_min) / (in_max - in_min)

    resized = torch.permute(resized, [1, 2, 0]).numpy()

    return resized


def select_tiling(h, w, patch_size, max_num_crops):
    """Divide in image of size [w, h] in up to max_num_patches of size patch_size"""
    original_size = np.stack([h, w])  # [1, 2]
    original_res = h * w
    tilings = []
    for i in range(1, max_num_crops + 1):
        for j in range(1, max_num_crops + 1):
            if i*j <= max_num_crops:
                tilings.append((i, j))
    # sort so argmin and argmax favour smaller tilings in the event of a tie
    tilings.sort(key=lambda x: (x[0]*x[1], x[0]))
    candidate_tilings = np.array(tilings, dtype=np.int32)  # [n_resolutions, 2]
    candidate_resolutions = candidate_tilings * patch_size  # [n_resolutions, 2]

    # How much we would need to scale the image to fit exactly in each tiling
    original_size = np.stack([h, w], dtype=np.float32)  # [1, 2]

    # The original size can be zero in rare cases if the image is smaller than the margin
    # In those cases letting the scale become infinite means the tiling is based on the
    # other side, or falls back to the smallest tiling
    with np.errstate(divide='ignore'):
        required_scale_d = candidate_resolutions.astype(np.float32) / original_size,
    required_scale = np.min(required_scale_d, axis=-1, keepdims=True)  # [n_resolutions, 1]
    if np.all(required_scale < 1):
        # We are forced to downscale, so try to minimize the amount of downscaling
        ix = np.argmax(required_scale)
    else:
        # Pick the resolution that required the least upscaling so that it most closely fits the image
        required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
        ix = np.argmin(required_scale)
    return candidate_tilings[ix]


def build_resized_image(
    image: np.ndarray,
    base_image_input_size: list[int],
    resample: PILImageResampling,
    image_mean: list[float],
    image_std: list[float],
    image_patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    resized = resize_image(
        image, base_image_input_size, resample,
    )
    resized = normalize_image(resized, image_mean, image_std)
    if len(resized.shape) == 3:
        resized = np.expand_dims(resized, 0)
    crop_patch_w = base_image_input_size[1] // image_patch_size
    crop_patch_h = base_image_input_size[0] // image_patch_size
    resize_idx = np.arange(crop_patch_w*crop_patch_h).reshape([crop_patch_h, crop_patch_w])
    return resized, resize_idx


def build_overlapping_crops(
    image: np.ndarray,
    max_crops: int,
    overlap_margins: list[int],
    base_image_input_size: list[int],
    resample: PILImageResampling,
    image_mean: list[float],
    image_std: list[float],
    image_patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose an image into a set of overlapping crops

    :return crop_arr: [n_crops, h, w, 3] The crops
    :return patch_idx: [overlap_patch_h, overlap_patch_w] For each patch in the resized image
                        the crops were extracted from, what patch in `crop_arr` it corresponds to
    """
    original_image_h, original_image_w = image.shape[:2]
    crop_size = base_image_input_size[0]
    assert base_image_input_size[0] == base_image_input_size[1]

    left_margin, right_margin = overlap_margins
    total_margin_pixels = image_patch_size * (right_margin + left_margin)  # pixels removed per dim
    crop_patches = base_image_input_size[0] // image_patch_size  # patches per crop dim
    crop_window_patches = crop_patches - (right_margin + left_margin)  # usable patches
    crop_window_size = crop_window_patches * image_patch_size
    crop_patch_w = base_image_input_size[1] // image_patch_size
    crop_patch_h = base_image_input_size[0] // image_patch_size
    original_image_h, original_image_w = image.shape[:2]
    crop_size = base_image_input_size[0]

    # Decide how to tile the image, to account for the overlap margins we compute the tiling
    # as if we had an image without the margins and were using a crop size without the margins
    tiling = select_tiling(
        original_image_h - total_margin_pixels,
        original_image_w - total_margin_pixels,
        crop_window_size,
        max_crops,
    )

    src = resize_image(
        image,
        [tiling[0]*crop_window_size+total_margin_pixels, tiling[1]*crop_window_size+total_margin_pixels],
        resample,
    )
    src = normalize_image(src, image_mean, image_std)

    # Now we have to split the image into crops, and track what patches came from
    # where in `patch_idx_arr`
    n_crops = tiling[0] * tiling[1]
    crop_arr = np.zeros([n_crops, crop_size, crop_size, 3], dtype=src.dtype)
    patch_idx_arr = np.zeros([n_crops, crop_patch_h, crop_patch_w], dtype=np.int32)
    on_crop = 0
    for i in range(tiling[0]):
        # Slide over `src` by `crop_window_size` steps, but extract crops of size `crops_size`
        # which results in overlapping crop windows
        y0 = i*crop_window_size
        for j in range(tiling[1]):
            x0 = j*crop_window_size
            crop_arr[on_crop] = src[y0:y0+crop_size, x0:x0+crop_size]
            patch_idx = np.arange(crop_patch_w*crop_patch_h).reshape(crop_patch_h, crop_patch_w)
            patch_idx += on_crop * crop_patch_h * crop_patch_w

            # Mask out idx that are in the overlap region
            if i != 0:
                patch_idx[:left_margin, :] = -1
            if j != 0:
                patch_idx[:, :left_margin] = -1
            if i != tiling[0]-1:
                patch_idx[-right_margin:, :] = -1
            if j != tiling[1]-1:
                patch_idx[:, -right_margin:] = -1
            patch_idx_arr[on_crop] = patch_idx
            on_crop += 1

    # `patch_idx_arr` is ordered crop-by-crop, here we transpose `patch_idx_arr`
    # so it is ordered left-to-right order
    patch_idx_arr = np.reshape(
        patch_idx_arr,
        [tiling[0], tiling[1], crop_patch_h, crop_patch_w]
    )
    patch_idx_arr = np.transpose(patch_idx_arr, [0, 2, 1, 3])
    patch_idx_arr = np.reshape(patch_idx_arr, [-1])

    # Now get the parts not in the overlap region, so it should map each patch in `src`
    # to the correct patch it should come from in `crop_arr`
    patch_idx_arr = patch_idx_arr[patch_idx_arr >= 0].reshape(
        src.shape[0]//image_patch_size,
        src.shape[1]//image_patch_size,
    )
    return crop_arr, patch_idx_arr


def batch_pixels_to_patches(array: np.ndarray, patch_size: int) -> np.ndarray:
    """Reshape images of [n_images, h, w, 3] -> [n_images, n_patches, pixels_per_patch]"""
    if len(array.shape) == 3:
        n_crops, h, w = array.shape
        h_patches = h//patch_size
        w_patches = w//patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size])
        array = np.transpose(array, [0, 1, 3, 2, 4])
        array = np.reshape(array, [n_crops, h_patches*w_patches, patch_size*patch_size])
        return array
    else:
        n_crops, h, w, c = array.shape
        h_patches = h//patch_size
        w_patches = w//patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size, c])
        array = np.transpose(array, [0, 1, 3, 2, 4, 5])
        array = np.reshape(array, [n_crops, h_patches*w_patches, patch_size*patch_size*c])
        return array


def arange_for_pooling(
    idx_arr: np.ndarray,
    pool_h: int,
    pool_w: int,
) -> np.ndarray:
    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    idx_arr = np.pad(idx_arr, [[h_pad//2, (h_pad+1)//2], [w_pad//2, (w_pad+1)//2]],
                     mode='constant',constant_values=-1)
    return einops.rearrange(
        idx_arr, "(h dh) (w dw) -> h w (dh dw)", dh=pool_h, dw=pool_w)


def image_to_patches_and_grids(
    image: np.ndarray,
    max_crops: int,
    overlap_margins: list[int],
    base_image_input_size: list[int],
    resample: PILImageResampling,
    image_mean: list[float],
    image_std: list[float],
    image_patch_size: int,
    image_pooling_w: int,
    image_pooling_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :return image_grids, the shape of each (low-res, high-res) image after pooling
    :return crops, the image crops to processes with the ViT
    :return pooled_patch_idx, for each patch_id tokens in `image_tokens`, the indices of the
                                patches in `crops` to pool for that token, masked with -1
    """
    if isinstance(base_image_input_size, int):
        base_image_input_size = (base_image_input_size, base_image_input_size)
    
    base_image_input_d = image_patch_size
    pooling_w = image_pooling_w
    pooling_h = image_pooling_h
    crop_patch_w = base_image_input_size[1] // base_image_input_d
    crop_patch_h = base_image_input_size[0] // base_image_input_d

    crop_arr, patch_idx_arr = build_overlapping_crops(
        image,
        max_crops,
        overlap_margins,
        base_image_input_size,
        resample,
        image_mean,
        image_std,
        image_patch_size,
    )
    pooling_idx = arange_for_pooling(patch_idx_arr, pooling_h, pooling_w)
    h, w = pooling_idx.shape[:2]
    pooling_idx = pooling_idx.reshape([-1, pooling_h*pooling_w])
    
    # Finally do the same for the global image
    resized, resize_idx = build_resized_image(
        image,
        base_image_input_size,
        resample,
        image_mean,
        image_std,
        image_patch_size,
    )
    crop_arr = np.concatenate([resized, crop_arr], 0)

    resize_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
    resized_h, resized_w = resize_idx.shape[:2]
    resize_idx = resize_idx.reshape([-1, pooling_h*pooling_w])

    # Global image goes first, so the order of patches in previous crops gets increased
    pooling_idx = np.where(
        pooling_idx >= 0,
        pooling_idx + crop_patch_h*crop_patch_w,
        -1
    )
    pooling_idx = np.concatenate([resize_idx, pooling_idx])
    image_grid = [np.array([resized_h, resized_w, h, w])]

    return (
        np.stack(image_grid, 0),
        batch_pixels_to_patches(crop_arr, image_patch_size),
        pooling_idx
    )


class Molmo2ImagesKwargs(ImagesKwargs, total=False):
    max_crops: Optional[int]
    overlap_margins: Optional[list[int]]
    patch_size: Optional[int]
    pooling_size: Optional[list[int]]


class Molmo2ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Molmo2 image processor that preprocesses images for the model.

    Args:
        size (`dict[str, int]` *optional*, defaults to `{"height": 378, "width": 378}`):
            Size of the image after resizing.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use when resizing the image.
        image_mean (`float` or `list[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
        image_std (`float` or `list[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats for each channel in the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        max_crops (`int`, *optional*, defaults to `8`):
            Maximum number of crops to use per image.
        overlap_margins (`list[int]`, *optional*, defaults to `[4, 4]`):
            Overlap margins to use.
        patch_size (`int`, *optional*, defaults to 14):
            The spatial patch size of the vision encoder.
        pooling_size (`list[int]`, *optional*, defaults to `[2, 2]`):
            The pooling size of the vision adapter.
    """

    model_input_names = ["pixel_values", "image_token_pooling", "image_grids", "image_num_crops"]

    def __init__(
        self,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: bool = True,
        max_crops: int = 8,
        overlap_margins: list[int] = [4, 4],
        patch_size: int = 14,
        pooling_size: list[int] = [2, 2],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 378, "width": 378}
        size = get_size_dict(size, default_to_square=True)
        self.size = size

        self.resample = resample
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_convert_rgb = do_convert_rgb

        self.max_crops = max_crops
        self.overlap_margins = overlap_margins
        self.patch_size = patch_size
        self.pooling_size = pooling_size
    
    def preprocess(
        self,
        images: ImageInput,
        size: Optional[dict[str, int]] = None,
        resample: Optional[PILImageResampling] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        max_crops: Optional[int] = None,
        overlap_margins: Optional[list[int]] = None,
        patch_size: Optional[int] = None,
        pooling_size: Optional[list[int]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
            images (`ImageInput`):
                Image to preprocess.
            size (`dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use when resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            max_crops (`int`, *optional*, defaults to `self.max_crops`):
                Maximum number of crops to use per image.
            overlap_margins (`list[int]`, *optional*, defaults to `self.overlap_margins`):
                Overlap margins to use.
            patch_size (`int`, *optional*, defaults to `self.patch_size`):
                The spatial patch size of the vision encoder.
            pooling_size (`list[int]`, *optional*, defaults to `self.pooling_size`):
                The pooling size of the vision adapter.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.

        Returns:
            A `BatchFeature` containing the following keys:
                - `pixel_values`: The preprocessed images.
                - `image_token_pooling`: The indices of the patches in `crops` to pool for each token in `image_tokens`.
                - `image_grids`: The image grids.
                - `image_num_crops`: The number of crops for each image.
        """
        if size is not None:
            if "height" not in size or "width" not in size:
                raise ValueError("size must contain 'height' and 'width' keys.")
        else:
            size = {**self.size}
        
        base_image_input_size = [size["height"], size["width"]]
        
        resample = resample or self.resample
        image_mean = image_mean or self.image_mean
        image_std = image_std or self.image_std
        do_convert_rgb = do_convert_rgb or self.do_convert_rgb

        max_crops = max_crops or self.max_crops
        overlap_margins = overlap_margins or self.overlap_margins
        patch_size = patch_size or self.patch_size
        pooling_size = pooling_size or self.pooling_size

        image_pooling_h, image_pooling_w = pooling_size

        if images is not None:
            images = self.fetch_images(images)
            images = make_flat_list_of_images(images)
        
        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        data = {}
        if images is not None:
            batch_grids = []
            batch_crops = []
            batch_pooled_patches_idx = []
            batch_num_crops = []

            for image in images:
                image_grid, crops, pooled_idx = image_to_patches_and_grids(
                    image,
                    max_crops,
                    overlap_margins,
                    base_image_input_size,
                    resample,
                    image_mean,
                    image_std,
                    patch_size,
                    image_pooling_w,
                    image_pooling_h,
                )
                batch_grids.append(image_grid)
                batch_crops.append(crops)
                batch_pooled_patches_idx.append(pooled_idx)
                batch_num_crops.append(crops.shape[0])
            
            pixel_values = np.concatenate(batch_crops, 0)
            image_token_pooling = np.concatenate(batch_pooled_patches_idx, 0)
            image_grids = np.concatenate(batch_grids, 0)
            image_num_crops = np.array(batch_num_crops)

            data.update(
                pixel_values=pixel_values,
                image_token_pooling=image_token_pooling,
                image_grids=image_grids,
                image_num_crops=image_num_crops,
            )

        return BatchFeature(data, tensor_type=return_tensors)


Molmo2ImageProcessor.register_for_auto_class()
