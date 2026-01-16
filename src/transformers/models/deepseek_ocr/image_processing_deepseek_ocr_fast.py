# Copyright 2026 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
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

import math

import torch
from torchvision.transforms import InterpolationMode

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    BatchFeature,
    Unpack,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...processing_utils import ImagesKwargs
from ...utils import TensorType, auto_docstring, logging


logger = logging.get_logger(__name__)


class DeepseekOcrImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    patch_size (`int`, *optional*):
        The size of the patch.
    patch_size_side (`int`, *optional*):
        The resolution of each high-resolution crop.
    base_size (`int`, *optional*):
        The base size for the global image view.
    max_crops (`int`, *optional*):
        The maximum number of crops per image.
    """

    patch_size: int
    patch_size_side: int
    base_size: int
    max_crops: int


@auto_docstring(
    custom_intro="""
    This image processor produces two tensors:
      * `pixel_values`: the global 1024x1024 image view that feeds both the SAM and CLIP encoder branches.
      * `pixel_values_local`: a batch of up to `max_crops` high-resolution 640x640 crops per image (padded with zeros).
    Metadata tensors (`num_local_crops`, `image_spatial_crop`, `num_img_tokens`) describe how many local crops are valid
    and how they should be spatially reassembled before projection.
    """
)
class DeepseekOcrImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    size = {"height": 1024, "width": 1024}
    base_size = {"height": 1024, "width": 1024}
    patch_size = 16
    patch_size_side = 640
    max_crops = 9
    downsample_ratio = 4
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    valid_kwargs = DeepseekOcrImageProcessorKwargs
    model_input_names = [
        "pixel_values",
        "pixel_values_local",
        "num_local_crops",
        "image_attention_mask",
        "image_spatial_crop",
        "num_img_tokens",
    ]

    def __init__(self, **kwargs: Unpack[DeepseekOcrImageProcessorKwargs]):
        super().__init__(**kwargs)

    def _further_process_kwargs(self, base_size=None, **kwargs):
        kwargs = super()._further_process_kwargs(**kwargs)
        if base_size is not None:
            kwargs["base_size"] = SizeDict(**base_size) if isinstance(base_size, dict) else base_size
        return kwargs

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
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
        return best_ratio

    def _resize_to_patches_grid(
        self,
        image: "torch.Tensor",
        patch_image_size: int,
        max_num: int = 36,
        min_num: int = 2,
        interpolation_mode: InterpolationMode = InterpolationMode.BICUBIC,
    ):
        height, width = image.shape[-2:]
        aspect_ratio = width / height

        target_ratios = {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        }
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, width, height, patch_image_size
        )

        target_width = patch_image_size * target_aspect_ratio[0]
        target_height = patch_image_size * target_aspect_ratio[1]

        resized_img = (
            super()
            .resize(
                image.unsqueeze(0),
                size=SizeDict(height=target_height, width=target_width),
                interpolation=interpolation_mode,
            )
            .squeeze(0)
        )

        return resized_img, target_aspect_ratio

    def _split_to_patches(
        self,
        image: "torch.Tensor",
        patch_image_size: int,
        target_aspect_ratio: tuple,
    ):
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.view(
            image.shape[0],
            target_aspect_ratio[1],
            patch_image_size,
            target_aspect_ratio[0],
            patch_image_size,
        )
        resized_img = (
            resized_img.permute(1, 3, 0, 2, 4)
            .contiguous()
            .view(blocks, image.shape[0], patch_image_size, patch_image_size)
        )

        processed_images = [resized_img[idx] for idx in range(blocks)]

        return processed_images

    def dynamic_preprocess(
        self,
        image: "torch.Tensor",
        patch_image_size: int,
        max_num: int = 36,
        min_num: int = 2,
        interpolation_mode: InterpolationMode = InterpolationMode.BICUBIC,
    ):
        resized_img, target_aspect_ratio = self._resize_to_patches_grid(
            image, patch_image_size, max_num, min_num, interpolation_mode
        )
        processed_images = self._split_to_patches(resized_img, patch_image_size, target_aspect_ratio)

        return processed_images, target_aspect_ratio

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[DeepseekOcrImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        size: SizeDict,
        base_size: SizeDict,
        interpolation: InterpolationMode | None,
        patch_size: int,
        patch_size_side: int,
        max_crops: int,
        do_rescale: bool,
        rescale_factor: float | None,
        do_normalize: bool,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        disable_grouping: bool | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ):
        patch_image_size = patch_size_side
        base_image_size = base_size.height
        downsample_ratio = self.downsample_ratio

        grouped_originals, grouped_orig_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}

        for shape, stacked_images in grouped_originals.items():
            height, width = shape
            scale = base_image_size / max(height, width)
            new_height = max(int(round(height * scale)), 1)
            new_width = max(int(round(width * scale)), 1)

            resized = super().resize(
                stacked_images,
                size=SizeDict(height=new_height, width=new_width),
                interpolation=interpolation,
            )
            resized_images_grouped[shape] = resized

        resized_images = reorder_images(resized_images_grouped, grouped_orig_index)

        grouped_resized, grouped_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)

        padded_grouped = {}
        mean_fill_values = [int(x * 255) for x in image_mean]
        for shape, stacked_images in grouped_resized.items():
            batch_size = stacked_images.shape[0]
            height, width = shape

            device = stacked_images.device
            dtype = stacked_images.dtype

            mean_fill = torch.tensor(mean_fill_values, dtype=dtype, device=device).view(1, 3, 1, 1)
            canvas = mean_fill.repeat(batch_size, 1, base_image_size, base_image_size).clone()

            y_offset = int(round((base_image_size - height) * 0.5))
            x_offset = int(round((base_image_size - width) * 0.5))
            canvas[:, :, y_offset : y_offset + height, x_offset : x_offset + width] = stacked_images

            canvas = canvas.to(torch.float32)
            canvas = self.rescale_and_normalize(
                canvas,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
            )
            padded_grouped[shape] = canvas

        global_views = reorder_images(padded_grouped, grouped_index)

        all_crops = []
        crop_to_image_idx = []
        crop_infos = []

        for img_idx, image in enumerate(images):
            height, width = image.shape[-2:]

            if width <= patch_image_size and height <= patch_image_size:
                crop_ratio = (1, 1)
                num_local = 0
            else:
                images_crop_raw, crop_ratio = self.dynamic_preprocess(
                    image,
                    patch_image_size,
                    max_num=max_crops,
                    interpolation_mode=interpolation,
                )
                num_local = len(images_crop_raw)

                for crop in images_crop_raw:
                    all_crops.append(crop)
                    crop_to_image_idx.append(img_idx)

            width_crop_num, height_crop_num = crop_ratio
            num_queries_base = math.ceil((base_image_size // 16) / downsample_ratio)
            num_queries = math.ceil((patch_image_size // 16) / downsample_ratio)

            tokenized_image_len = (num_queries_base + 1) * num_queries_base + 1
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image_len += (num_queries * width_crop_num + 1) * (num_queries * height_crop_num)

            crop_infos.append(
                {
                    "crop_ratio": (width_crop_num, height_crop_num),
                    "num_local": num_local,
                    "num_tokens": tokenized_image_len,
                }
            )

        if all_crops:
            grouped_crops, grouped_crop_idx, crops_index = group_images_by_shape(
                all_crops, crop_to_image_idx, disable_grouping=disable_grouping
            )

            processed_crops_grouped = {}
            for shape, stacked_crops in grouped_crops.items():
                stacked_crops = stacked_crops.to(torch.float32)
                processed = self.rescale_and_normalize(
                    stacked_crops,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                )
                processed_crops_grouped[shape] = processed

            processed_crops_list = reorder_images(processed_crops_grouped, crops_index)
        else:
            processed_crops_list = []

        local_views = []
        crop_idx = 0
        for img_idx in range(len(images)):
            num_crops = crop_infos[img_idx]["num_local"]

            if num_crops > 0:
                img_crops = processed_crops_list[crop_idx : crop_idx + num_crops]
                local_tensor = torch.stack(img_crops, dim=0)
                crop_idx += num_crops
            else:
                local_tensor = torch.empty(
                    0,
                    3,
                    patch_image_size,
                    patch_image_size,
                    dtype=global_views[0].dtype,
                    device=global_views[0].device,
                )

            local_views.append(local_tensor)

        global_views_unsqueezed = [view.unsqueeze(0) for view in global_views]
        pixel_values_global = torch.stack(global_views_unsqueezed, dim=0)

        local_counts = [info["num_local"] for info in crop_infos]
        max_local = max(local_counts) if local_counts else 0

        padded_locals = []
        for local_tensor in local_views:
            if max_local == 0:
                padded_locals.append(local_tensor.unsqueeze(0))
            elif local_tensor.shape[0] < max_local:
                pad_size = max_local - local_tensor.shape[0]
                local_tensor = torch.nn.functional.pad(local_tensor, (0, 0, 0, 0, 0, 0, 0, pad_size))
                padded_locals.append(local_tensor.unsqueeze(0))
            else:
                padded_locals.append(local_tensor.unsqueeze(0))

        if max_local > 0:
            pixel_values_local = torch.cat(padded_locals, dim=0)
        else:
            batch_size = len(local_views)
            pixel_values_local = torch.zeros(
                batch_size,
                0,
                3,
                patch_image_size,
                patch_image_size,
                dtype=pixel_values_global.dtype,
                device=pixel_values_global.device,
            )

        images_spatial_crop = torch.tensor([info["crop_ratio"] for info in crop_infos], dtype=torch.long)
        num_local_crops = torch.tensor(local_counts, dtype=torch.long)
        images_tokens = [info["num_tokens"] for info in crop_infos]

        data = {
            "pixel_values": pixel_values_global,
            "pixel_values_local": pixel_values_local,
            "num_local_crops": num_local_crops,
            "image_spatial_crop": images_spatial_crop,
            "num_img_tokens": images_tokens,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["DeepseekOcrImageProcessorFast"]
