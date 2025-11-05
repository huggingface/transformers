# Copyright 2025 Deepseek-AI and the HuggingFace Inc. team. All rights reserved.
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
from typing import Optional, Union

import numpy as np
import torch
from torchvision.transforms import InterpolationMode

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    BatchFeature,
    Unpack,
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


@auto_docstring
class DeepseekOcrImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    size = {"height": 1024, "width": 1024}
    base_size = {"height": 1024, "width": 1024}
    patch_size = 16
    patch_size_side = 1024
    max_crops = 9
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    valid_kwargs = DeepseekOcrImageProcessorKwargs
    model_input_names = [
        "pixel_values",
        "pixel_values_global",
        "pixel_values_local",
        "num_local_crops",
        "image_attention_mask",
        "image_spatial_crop",
        "num_img_tokens",
    ]

    def __init__(self, **kwargs: Unpack[DeepseekOcrImageProcessorKwargs]):
        super().__init__(**kwargs)
        # original implementation capped the number of local crops to 9 tiles.
        if self.max_crops is None or self.max_crops > 9:
            self.max_crops = 9

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

    def pad_to_max_num_crops(self, images, max_crops=5):
        """Pad images tensor to max_crops."""
        B, _, H, W = images.shape
        if B < max_crops:
            pad_size = max_crops - B
            images = torch.nn.functional.pad(images, (0, 0, 0, 0, 0, 0, 0, pad_size))
        return images

    def pad_image(
        self,
        image: "torch.Tensor",
        target_size: int,
        interpolation_mode: InterpolationMode,
        mean_fill: "torch.Tensor",
    ) -> "torch.Tensor":
        """Resize with preserved aspect ratio and pad to a square of target_size."""
        height, width = image.shape[-2:]

        if height == target_size and width == target_size:
            return image

        scale = target_size / max(height, width)
        new_height = max(int(round(height * scale)), 1)
        new_width = max(int(round(width * scale)), 1)

        resized = (
            super()
            .resize(
                image.unsqueeze(0),
                size=SizeDict(height=new_height, width=new_width),
                interpolation=interpolation_mode,
            )
            .squeeze(0)
        )

        canvas = mean_fill.repeat(1, target_size, target_size).clone()
        y_offset = int(round((target_size - new_height) * 0.5))
        x_offset = int(round((target_size - new_width) * 0.5))
        canvas[:, y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized

        return canvas

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
        interpolation: Optional[InterpolationMode],
        patch_size: int,
        patch_size_side: int,
        max_crops: int,
        do_rescale: bool,
        rescale_factor: Optional[float],
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        patch_image_size = patch_size_side
        base_image_size = base_size.height
        downsample_ratio = 4

        global_views = []
        local_views = []
        images_spatial_crop = []
        images_tokens = []
        local_counts = []

        for image in images:
            height, width = image.shape[-2:]

            if width <= patch_image_size and height <= patch_image_size:
                crop_ratio = (1, 1)
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = self.dynamic_preprocess(
                    image,
                    patch_image_size,
                    max_num=max_crops,
                    interpolation_mode=interpolation,
                )

            mean_fill_values = [int(x * 255) for x in image_mean]
            mean_fill = torch.tensor(mean_fill_values, dtype=image.dtype, device=image.device).view(3, 1, 1)
            global_view = self.pad_image(image, base_image_size, interpolation, mean_fill)
            global_view = global_view.to(torch.float32)
            global_view = self.rescale_and_normalize(
                global_view,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
            )

            width_crop_num, height_crop_num = crop_ratio

            processed_crops = []
            if width_crop_num > 1 or height_crop_num > 1:
                crops_batch = torch.stack(images_crop_raw, dim=0).to(torch.float32)
                processed_batch = self.rescale_and_normalize(
                    crops_batch,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                )
                processed_crops = list(processed_batch)

            num_queries_base = math.ceil((base_image_size // 16) / downsample_ratio)
            num_queries = math.ceil((patch_image_size // 16) / downsample_ratio)

            tokenized_image_len = (num_queries_base + 1) * num_queries_base + 1
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image_len += (num_queries * width_crop_num + 1) * (num_queries * height_crop_num)

            local_tensor = (
                torch.stack(processed_crops, dim=0)
                if processed_crops
                else torch.empty(
                    0, 3, patch_image_size, patch_image_size, dtype=global_view.dtype, device=global_view.device
                )
            )

            global_views.append(global_view.unsqueeze(0))
            local_views.append(local_tensor)
            local_counts.append(local_tensor.shape[0])
            images_spatial_crop.append([width_crop_num, height_crop_num])
            images_tokens.append(tokenized_image_len)

        pixel_values_global = torch.stack(global_views, dim=0)

        max_local = max(local_counts) if local_counts else 0
        padded_locals = []
        for local_tensor in local_views:
            if max_local == 0:
                padded_locals.append(local_tensor.unsqueeze(0))
                continue
            if local_tensor.shape[0] < max_local:
                pad_size = max_local - local_tensor.shape[0]
                local_tensor = torch.nn.functional.pad(local_tensor, (0, 0, 0, 0, 0, 0, 0, pad_size))
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

        images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
        num_local_crops = torch.tensor(local_counts, dtype=torch.long)

        data = {
            "pixel_values": pixel_values_global,
            "pixel_values_global": pixel_values_global,
            "pixel_values_local": pixel_values_local,
            "num_local_crops": num_local_crops,
            "image_spatial_crop": images_spatial_crop,
            "num_img_tokens": images_tokens,
        }

        batch = BatchFeature(data=data, tensor_type=return_tensors)
        batch["num_img_tokens"] = images_tokens
        return batch

    def extract_coordinates_and_label(self, ref_text, image_width, image_height):
        """Extract bounding box coordinates and label from model output."""
        try:
            label_type = ref_text[1]
            cor_list = eval(ref_text[2])
        except Exception as e:
            logger.warning(f"Failed to extract coordinates: {e}")
            return None

        return (label_type, cor_list)

    def visualize_results(self, image, ref_texts, output_path):
        """
        Visualize results by drawing bounding boxes on the image.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            logger.error("PIL is required for visualization. Install it with `pip install pillow`")
            return None

        if isinstance(image, torch.Tensor):
            image = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

        image_width, image_height = image.size

        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)

        overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
        draw2 = ImageDraw.Draw(overlay)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        img_idx = 0

        for i, ref in enumerate(ref_texts):
            try:
                result = self.extract_coordinates_and_label(ref, image_width, image_height)
                if result:
                    label_type, points_list = result

                    color = (
                        np.random.randint(0, 200),
                        np.random.randint(0, 200),
                        np.random.randint(0, 255),
                    )
                    color_a = color + (20,)

                    for points in points_list:
                        x1, y1, x2, y2 = points

                        x1 = int(x1 / 999 * image_width)
                        y1 = int(y1 / 999 * image_height)
                        x2 = int(x2 / 999 * image_width)
                        y2 = int(y2 / 999 * image_height)

                        if label_type == "image":
                            try:
                                cropped = image.crop((x1, y1, x2, y2))
                                cropped.save(f"{output_path}/images/{img_idx}.jpg")
                            except Exception as e:
                                logger.warning(f"Failed to save cropped image: {e}")
                            img_idx += 1

                        try:
                            if label_type == "title":
                                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                                draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                            else:
                                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                                draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                            text_x = x1
                            text_y = max(0, y1 - 15)

                            if font:
                                text_bbox = draw.textbbox((0, 0), label_type, font=font)
                                text_width = text_bbox[2] - text_bbox[0]
                                text_height = text_bbox[3] - text_bbox[1]
                                draw.rectangle(
                                    [text_x, text_y, text_x + text_width, text_y + text_height],
                                    fill=(255, 255, 255, 30),
                                )
                                draw.text((text_x, text_y), label_type, font=font, fill=color)
                        except Exception as e:
                            logger.warning(f"Failed to draw bounding box: {e}")
            except Exception as e:
                logger.warning(f"Failed to process reference: {e}")
                continue

        img_draw.paste(overlay, (0, 0), overlay)
        return img_draw


__all__ = ["DeepseekOcrImageProcessorFast"]
