# Copyright 2025 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
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

# TODO protect this import
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.functional import to_pil_image

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
    base_size (`int`, *optional*):
        The base size for the global image view.
    dynamic_hd (`int`, *optional*):
        The maximum number of crops per image.
    """

    patch_size: int
    base_size: int
    dynamic_hd: int


@auto_docstring
class DeepseekOcrImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    size = {"height": 1024, "width": 1024}
    base_size = {"height": 1024, "width": 1024}
    patch_size = 16
    dynamic_hd = 36
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    valid_kwargs = DeepseekOcrImageProcessorKwargs
    model_input_names = ["pixel_values", "image_attention_mask", "image_spatial_crop"]

    def __init__(self, **kwargs: Unpack[DeepseekOcrImageProcessorKwargs]):
        super().__init__(**kwargs)

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

    def dynamic_preprocess(self, image, base_size, patch_size, max_num=36, min_num=1):
        """
        Dynamically preprocess images with aspect ratio handling.

        Returns:
            processed_images: list of preprocessed image tensors
            target_aspect_ratio: tuple (width_crops, height_crops)
        """
        if not isinstance(image, Image.Image):
            image = to_pil_image(image)

        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        }
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, patch_size
        )

        target_width = patch_size * target_aspect_ratio[0]
        target_height = patch_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))

        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // patch_size)) * patch_size,
                (i // (target_width // patch_size)) * patch_size,
                ((i % (target_width // patch_size)) + 1) * patch_size,
                ((i // (target_width // patch_size)) + 1) * patch_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        return processed_images, target_aspect_ratio

    def pad_to_max_num_crops(self, images, max_crops=5):
        """Pad images tensor to max_crops."""
        B, _, H, W = images.shape
        if B < max_crops:
            pad = torch.zeros(max_crops - B, 3, H, W, dtype=images.dtype, device=images.device)
            images = torch.cat([images, pad], dim=0)
        return images

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
        interpolation: Optional["F.InterpolationMode"],
        patch_size: int,
        dynamic_hd: int,
        do_rescale: bool,
        rescale_factor: Optional[float],
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        if not isinstance(size, SizeDict):
            size = SizeDict(**size)
        if not isinstance(base_size, SizeDict):
            base_size = SizeDict(**base_size)
        patch_image_size = size.height
        base_image_size = base_size.height
        downsample_ratio = 4

        images_transformed = []
        images_spatial_crop = []
        images_tokens = []

        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=self.image_mean, std=self.image_std)
        mean_fill = tuple(int(x * 255) for x in self.image_mean)

        for image in images:
            if not isinstance(image, Image.Image):
                image = to_pil_image(image)

            orig_width, orig_height = image.size

            if orig_width <= patch_image_size and orig_height <= patch_image_size:
                crop_ratio = [1, 1]
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = self.dynamic_preprocess(
                    image, base_image_size, patch_image_size, max_num=dynamic_hd
                )

            global_view = ImageOps.pad(image, (base_image_size, base_image_size), color=mean_fill)
            if base_image_size != patch_image_size:
                global_view = global_view.resize((patch_image_size, patch_image_size), interpolation)

            global_view = normalize(to_tensor(global_view)).to(torch.bfloat16)

            ratio = 1 - ((max(orig_width, orig_height) - min(orig_width, orig_height)) / max(orig_width, orig_height))

            if base_image_size == 1024:
                valid_img_tokens = int(256 * ratio)
            elif base_image_size == 640:
                valid_img_tokens = int(100 * ratio)
            else:
                num_queries_base = math.ceil((base_image_size // 16) / downsample_ratio)
                valid_img_tokens = int(num_queries_base * num_queries_base * ratio)

            width_crop_num, height_crop_num = crop_ratio

            if width_crop_num > 1 or height_crop_num > 1:
                processed_crops = []
                for crop in images_crop_raw:
                    crop_tensor = normalize(to_tensor(crop)).to(torch.bfloat16)
                    processed_crops.append(crop_tensor)

                if patch_image_size == 640:
                    valid_img_tokens += len(processed_crops) * 100

                crops_tensor = torch.stack(processed_crops, dim=0)
            else:
                processed_crops = []
                crops_tensor = torch.empty(0, dtype=torch.bfloat16)

            num_queries_base = math.ceil((base_image_size // 16) / downsample_ratio)
            num_queries = math.ceil((patch_image_size // 16) / downsample_ratio)

            tokenized_image_len = (num_queries_base + 1) * num_queries_base + 1
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image_len += (num_queries * width_crop_num + 1) * (num_queries * height_crop_num) + 1

            if crops_tensor.numel() > 0:
                hd_images = torch.cat([crops_tensor, global_view.unsqueeze(0)], dim=0)
            else:
                hd_images = global_view.unsqueeze(0)

            max_crops = hd_images.size(0)
            hd_images = self.pad_to_max_num_crops(hd_images, max_crops).to(torch.bfloat16)

            images_transformed.append(hd_images)
            images_spatial_crop.append([width_crop_num, height_crop_num])
            images_tokens.append(tokenized_image_len)

        max_crops = max(img.size(0) for img in images_transformed)
        images_transformed = [self.pad_to_max_num_crops(im, max_crops) for im in images_transformed]
        images_transformed = torch.stack(images_transformed, dim=0)
        images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)

        data = {
            "pixel_values": images_transformed,
            "image_spatial_crop": images_spatial_crop,
            "num_img_tokens": images_tokens,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)

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

        Args:
            image: PIL Image
            ref_texts: list of reference texts from model output
            output_path: path to save the visualization

        Returns:
            PIL Image with bounding boxes drawn
        """
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
