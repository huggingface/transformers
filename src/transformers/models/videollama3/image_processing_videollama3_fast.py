"""Fast Image processor class for VideoLLaMA3."""

from typing import Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
    validate_fast_preprocess_arguments,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    logging,
)
from ...video_utils import VideoInput, make_batched_videos
from .image_processing_videollama3 import smart_resize


if is_torch_available():
    import torch


if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F

logger = logging.get_logger(__name__)


class Videollama3FastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    min_tokens (`int`, *optional*, defaults to `16`):
        The min tokens of the image to resize the image.
    max_tokens (`int`, *optional*, defaults to `16384`):
        The max tokens of the image to resize the image.
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    image_merge_size (`int`, *optional*, defaults to 1):
        The spatial downsampling ratio of each image feature.
    video_merge_size (`int`, *optional*, defaults to 2):
        The spatial downsampling ratio of each video feature.
    """

    min_tokens: Optional[int]
    max_tokens: Optional[int]
    patch_size: Optional[int]
    image_merge_size: Optional[int]
    video_merge_size: Optional[int]


@auto_docstring
class Videollama3ImageProcessorFast(BaseImageProcessorFast):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    do_rescale = True
    do_normalize = True
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_convert_rgb = True
    patch_size = 14
    image_merge_size = 1
    video_merge_size = 2
    min_tokens = 16
    max_tokens = 16384
    valid_kwargs = Videollama3FastImageProcessorKwargs
    model_input_names = [
        "pixel_values",
        "grid_sizes",
        "merge_sizes",
        "pixel_values_videos",
        "grid_sizes_videos",
        "merge_sizes_videos",
    ]

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        videos: Optional[VideoInput] = None,
        **kwargs: Unpack[Videollama3FastImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, videos, **kwargs)

    def _validate_preprocess_kwargs(
        self,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, tuple[float]]] = None,
        image_std: Optional[Union[float, tuple[float]]] = None,
        do_center_crop: Optional[bool] = None,
        crop_size: Optional[SizeDict] = None,
        resample: Optional[Union["PILImageResampling", "F.InterpolationMode"]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = None,
        **kwargs,
    ):
        """
        validate the kwargs for the preprocess method.
        """
        validate_fast_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            resample=resample,
            return_tensors=return_tensors,
            data_format=data_format,
        )

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        videos: VideoInput,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: Optional[Union[str, "torch.device"]] = None,
        **kwargs: Unpack[DefaultFastImageProcessorKwargs],
    ) -> BatchFeature:
        """
        Preprocess image-like inputs.
        To be overridden by subclasses when image-like inputs other than images should be processed.
        It can be used for segmentation maps, depth maps, etc.
        """
        # Prepare input images
        batch_feature = BatchFeature()
        if images is not None:
            images = self._prepare_image_like_inputs(
                images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
            )
            merge_size = kwargs.get("image_merge_size", self.image_merge_size)
            batch_feature = self._preprocess(images, merge_size=merge_size, **kwargs)
        if videos is not None:
            logger.warning(
                "`Videollama3ImageProcessorFast` works only with image inputs and doesn't process videos anymore. "
                "This is a deprecated behavior and will be removed in v5.0. "
                "Your videos should be forwarded to `Qwen2VLVideoProcessor`. "
            )
            # Can't change _prepare_images_structure to work with videos because it also needs to work with images.
            videos = make_batched_videos(videos)
            videos = [
                torch.stack(self._prepare_image_like_inputs(video, do_convert_rgb, input_data_format, device))
                for video in videos
            ]
            merge_size = kwargs.get("video_merge_size", self.video_merge_size)
            video_outputs = self._preprocess(videos, merge_size=merge_size, **kwargs)
            batch_feature.update(
                {
                    "pixel_values_videos": video_outputs.pixel_values,
                    "grid_sizes_videos": video_outputs.grid_sizes,
                    "merge_sizes_videos": video_outputs.merge_sizes,
                }
            )
        return batch_feature

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        min_tokens: int,
        max_tokens: int,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        patch_size: int,
        merge_size: int,
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ):
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=min_tokens * (patch_size * merge_size) ** 2,
                    max_pixels=max_tokens * (patch_size * merge_size) ** 2,
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]
            # Fused rescale and normalize
            patches = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            if patches.ndim == 4:
                # add a temporal dimension if we have images
                patches = patches.unsqueeze(1)
            batch_size, t, channel = patches.shape[:3]
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                t,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            # Reorder dimensions to group grid and patch information for subsequent flattening.
            # (batch, t, grid_h, grid_w, merge_h, merge_w, channel, patch_h, patch_w)
            patches = patches.permute(0, 1, 3, 6, 4, 7, 2, 5, 8)
            flatten_patches = patches.reshape(
                batch_size,
                t * grid_h * grid_w,
                channel * patch_size * patch_size,
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids = reorder_images(processed_grids, grouped_images_index)
        pixel_values = torch.cat(processed_images, dim=0)
        grid_sizes = torch.tensor(processed_grids)
        merge_sizes = torch.tensor(
            [merge_size] * grid_sizes.size(0),
            dtype=grid_sizes.dtype,
            device=grid_sizes.device,
        )

        return BatchFeature(
            data={"pixel_values": pixel_values, "grid_sizes": grid_sizes, "merge_sizes": merge_sizes},
            tensor_type=return_tensors,
        )


__all__ = ["Videollama3ImageProcessorFast"]
