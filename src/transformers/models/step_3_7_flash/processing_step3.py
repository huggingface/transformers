from itertools import product
from math import ceil
from typing import Literal, TypedDict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from transformers import BaseImageProcessor
from transformers.feature_extraction_utils import BatchFeature, TensorType
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_tokenizers import TokenizersBackend


MAX_IMAGE_SIZE: int = 3024


class Step3VLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    patch_pixel_values: torch.Tensor | None
    num_patches: list[int]


class Step3VLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: torch.Tensor


ImageWithPatches = tuple[Image.Image, list[Image.Image], list[int] | None]


class GPUToTensor(torch.nn.Module):
    def forward(self, raw_image: np.ndarray | Image.Image) -> torch.Tensor:
        if isinstance(raw_image, Image.Image):
            return transforms.ToTensor()(raw_image)
        if raw_image.ndim == 2:
            raw_image = raw_image[:, :, None].repeat(3, -1)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        image_tensor = torch.from_numpy(raw_image).to(device)
        image_tensor = torch.permute(image_tensor, (2, 0, 1)).contiguous()
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.to(torch.float32).div(255)
        return image_tensor


class Step3VisionProcessor(BaseImageProcessor):
    def __init__(self, size, interpolation_mode="bicubic", patch_size=None):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        patch_size = patch_size if patch_size is not None else size

        self.transform = transforms.Compose(
            [
                GPUToTensor(),
                transforms.Normalize(mean, std),
                transforms.Resize(
                    (size, size),
                    interpolation=InterpolationMode.BICUBIC
                    if interpolation_mode == "bicubic"
                    else InterpolationMode.BILINEAR,
                    antialias=True,
                ),
            ]
        )

        self.patch_transform = (
            transforms.Compose(
                [
                    GPUToTensor(),
                    transforms.Normalize(mean, std),
                    transforms.Resize(
                        (patch_size, patch_size),
                        interpolation=InterpolationMode.BICUBIC
                        if interpolation_mode == "bicubic"
                        else InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ]
            )
            if patch_size is not None
            else None
        )

    def __call__(self, image, is_patch=False):
        if is_patch:
            return {"pixel_values": self.patch_transform(image).unsqueeze(0)}
        else:
            return {"pixel_values": self.transform(image).unsqueeze(0)}


class ImagePatcher:
    def determine_window_size(self, long: int, short: int) -> int:
        if long <= 728:
            return short if long / short > 1.5 else 0
        return min(short, 504) if long / short > 4 else 504

    def slide_window(
        self,
        width: int,
        height: int,
        sizes: list[tuple[int, int]],
        steps: list[tuple[int, int]],
        img_rate_thr: float = 0.6,
    ) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
        assert 1 >= img_rate_thr >= 0, "The `in_rate_thr` should lie in 0~1"
        windows = []
        # Sliding windows.
        for size, step in zip(sizes, steps):
            size_w, size_h = size
            step_w, step_h = step

            x_num = 1 if width <= size_w else ceil((width - size_w) / step_w + 1)
            x_start = [step_w * i for i in range(x_num)]
            if len(x_start) > 1 and x_start[-1] + size_w > width:
                x_start[-1] = width - size_w

            y_num = 1 if height <= size_h else ceil((height - size_h) / step_h + 1)
            y_start = [step_h * i for i in range(y_num)]
            if len(y_start) > 1 and y_start[-1] + size_h > height:
                y_start[-1] = height - size_h

            start = np.array(list(product(y_start, x_start)), dtype=int)
            start[:, [0, 1]] = start[:, [1, 0]]
            windows.append(np.concatenate([start, start + size], axis=1))
        windows = np.concatenate(windows, axis=0)

        return [(int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])) for box in windows], (
            x_num,
            y_num,
        )

    def square_pad(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        size = max(w, h)
        padded = Image.new(img.mode, (size, size), 0)
        padded.paste(img, (0, 0))
        return padded

    def get_image_size_for_padding(self, img_width: int, img_height: int) -> tuple[int, int]:
        ratio = img_width / img_height
        if min(img_height, img_width) < 32 and (ratio > 4 or ratio < 1 / 4):
            new_size = max(img_height, img_width)
            return new_size, new_size
        return img_width, img_height

    def get_image_size_for_preprocess(self, img_width: int, img_height: int) -> tuple[int, int]:
        if max(img_height, img_width) > MAX_IMAGE_SIZE:
            scale_factor = MAX_IMAGE_SIZE / max(img_height, img_width)
            img_width = int(img_width * scale_factor)
            img_height = int(img_height * scale_factor)
        return img_width, img_height

    def get_image_size_for_crop(self, img_width: int, img_height: int, window_size: int):
        w_ratio = img_width / window_size
        h_ratio = img_height / window_size

        if w_ratio < 1:
            width_new = img_width
        else:
            decimal_w = w_ratio - img_width // window_size
            w_ratio = int(w_ratio) + 1 if decimal_w > 0.2 else int(w_ratio)
            width_new = window_size * w_ratio
        if h_ratio < 1:
            height_new = img_height
        else:
            decimal_h = h_ratio - img_height // window_size
            h_ratio = int(h_ratio) + 1 if decimal_h > 0.2 else int(h_ratio)
            height_new = window_size * h_ratio
        return int(width_new), int(height_new)

    def patch_crop(self, img: Image.Image, i: int, j: int, th: int, tw: int):
        target = img.crop((j, i, j + tw, i + th))
        return target

    def get_num_patches(self, img_width: int, img_height: int) -> tuple[int, int]:
        img_width, img_height = self.get_image_size_for_padding(img_width, img_height)
        img_width, img_height = self.get_image_size_for_preprocess(img_width, img_height)
        window_size = self.determine_window_size(max(img_height, img_width), min(img_height, img_width))
        if window_size == 0:
            return 0, 0
        else:
            img_width, img_height = self.get_image_size_for_crop(img_width, img_height, window_size)
            center_list, (x_num, y_num) = self.slide_window(
                img_width, img_height, [(window_size, window_size)], [(window_size, window_size)]
            )
            full_rows = (len(center_list) - 1) // x_num + 1
            if len(center_list) > 0 and len(center_list) % x_num == 0:
                full_rows -= 1
            return len(center_list), full_rows

    def __call__(self, img: Image.Image) -> tuple[Image.Image, list[Image.Image], list[bool] | None]:
        img_width, img_height = img.size
        new_img_width, new_img_height = self.get_image_size_for_padding(img_width, img_height)
        if new_img_width != img_width or new_img_height != img_height:
            img = self.square_pad(img)
            img_width, img_height = img.size

        new_img_width, new_img_height = self.get_image_size_for_preprocess(img_width, img_height)
        img = img.resize((new_img_width, new_img_height), Image.Resampling.BILINEAR)
        window_size = self.determine_window_size(
            max(new_img_height, new_img_width), min(new_img_height, new_img_width)
        )
        # return img, [], None
        if window_size == 0:
            return img, [], None
        else:
            new_img_width, new_img_height = self.get_image_size_for_crop(new_img_width, new_img_height, window_size)
            if (new_img_width, new_img_height) != (img_width, img_height):
                img_for_crop = img.resize((new_img_width, new_img_height), Image.Resampling.BILINEAR)
            else:
                img_for_crop = img

            patches = []
            newlines = []
            center_list, (x_num, y_num) = self.slide_window(
                new_img_width, new_img_height, [(window_size, window_size)], [(window_size, window_size)]
            )
            for patch_id, center_lf_point in enumerate(center_list):
                x, y, patch_w, patch_h = center_lf_point
                big_patch = self.patch_crop(img_for_crop, y, x, patch_h, patch_w)
                patches.append(big_patch)
                if (patch_id + 1) % x_num == 0:
                    newlines.append(patch_id)

            if newlines and newlines[-1] == len(patches) - 1:
                newlines.pop()

            return img, patches, [i in newlines for i in range(len(patches))] if len(patches) > 0 else None


class Step3VLProcessor(ProcessorMixin):
    # Align ProcessorMixin with our custom components.
    # We only have an image processor (not a feature extractor) plus a tokenizer.
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    @classmethod
    def _load_tokenizer_from_pretrained(
        cls, sub_processor_type, pretrained_model_name_or_path, subfolder="", **kwargs
    ):
        return TokenizersBackend.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            **kwargs,
        )

    def __init__(self, tokenizer=None, chat_template=None, **kwargs) -> None:
        self.image_size = 728
        self.patch_size = 504

        self.image_preprocessor = Step3VisionProcessor(self.image_size, "bilinear", self.patch_size)

        self.num_image_feature_size = 169
        self.num_patch_feature_size = 81
        self.image_token = "<im_patch>"
        self.image_feature_placeholder = self.image_token * self.num_image_feature_size
        self.patch_feature_placeholder = self.image_token * self.num_patch_feature_size
        super().__init__(tokenizer=tokenizer, chat_template=chat_template, **kwargs)
        self.patcher = ImagePatcher()

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[self.image_token]

    def get_num_image_tokens(self, img_width: int, img_height: int) -> int:
        num_patches, num_newlines = self.patcher.get_num_patches(img_width, img_height)

        return num_patches * (self.num_patch_feature_size + 2) + self.num_image_feature_size + 2 + num_newlines

    def _split_images(self, images: list[Image.Image]) -> list[ImageWithPatches]:
        result = []
        for img in images:
            result.append(self.patcher(img))
        return result

    def _convert_images_to_pixel_values(
        self,
        images: list[Image.Image],
        is_patch: bool = False,
    ) -> list[torch.Tensor]:
        return [self.image_preprocessor(img, is_patch=is_patch)["pixel_values"] for img in images]

    def _get_patch_repl(
        self,
        num_patches: int,
        patch_newline_mask: list[bool] | None,
    ) -> tuple[str, list[int]]:
        text = ""
        token_ids = []
        for i in range(num_patches):
            assert len(patch_newline_mask) == num_patches
            text += f"<patch_start>{self.patch_feature_placeholder}<patch_end>"
            token_ids.extend(
                [self.tokenizer.convert_tokens_to_ids("<patch_start>")]
                + [self.image_token_id] * self.num_patch_feature_size
                + [self.tokenizer.convert_tokens_to_ids("<patch_end>")]
            )
            if patch_newline_mask and patch_newline_mask[i]:
                text += "<patch_newline>"
                token_ids.append(self.tokenizer.convert_tokens_to_ids("<patch_newline>"))
        return text, token_ids

    def _get_image_repl(
        self,
        num_images: int,
    ) -> tuple[str, list[int]]:
        text = f"<im_start>{self.image_feature_placeholder}<im_end>"
        token_ids = (
            [self.tokenizer.convert_tokens_to_ids("<im_start>")]
            + [self.image_token_id] * self.num_image_feature_size
            + [self.tokenizer.convert_tokens_to_ids("<im_end>")]
        )
        return text * num_images, token_ids * num_images

    def _get_image_repl_features(
        self,
        num_images: int,
        num_patches: int,
        patch_new_line_idx: list[bool] | None,
    ) -> tuple[str, list[int]]:
        if num_patches > 0:
            patch_repl, patch_repl_ids = self._get_patch_repl(num_patches, patch_new_line_idx)
        else:
            patch_repl = ""
            patch_repl_ids = []
        image_repl, image_repl_ids = self._get_image_repl(num_images)
        return patch_repl + image_repl, patch_repl_ids + image_repl_ids

    def replace_placeholder(self, text: str, placeholder: str, repls: list[str]) -> str:
        parts = text.split(placeholder)

        if len(parts) - 1 != len(repls):
            raise ValueError(
                "The number of placeholders does not match the number of replacements."  # noqa: E501
            )

        result = [parts[0]]
        for i, repl in enumerate(repls):
            result.append(repl)
            result.append(parts[i + 1])

        return "".join(result)

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: ImageInput | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        if images is not None:
            images = self.image_preprocessor.fetch_images(images)
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if images is None:
            images = []
        elif not isinstance(images, list):
            images = [images]
        elif isinstance(images[0], list):
            images = images[0]

        if len(images) == 0:
            image_inputs = {}
            text_inputs = self.tokenizer(text)
        else:
            splitted_images_data = self._split_images(images)
            pixel_values_lst = []
            patch_pixel_values_lst = []
            patch_newline_mask_lst = []
            image_repl_str_lst = []
            image_repl_ids_lst = []
            num_patches = []
            for raw_img, img_patches, patch_newline_mask in splitted_images_data:  # noqa: E501
                pixel_values_lst.extend(self._convert_images_to_pixel_values([raw_img]))

                if len(img_patches) > 0:
                    patch_pixel_values_lst.extend(self._convert_images_to_pixel_values(img_patches, is_patch=True))
                num_patches.append(len(img_patches))

                image_repl_str, image_repl_ids = self._get_image_repl_features(1, len(img_patches), patch_newline_mask)
                image_repl_str_lst.append(image_repl_str)
                image_repl_ids_lst.extend(image_repl_ids)

                if patch_newline_mask is not None:
                    patch_newline_mask_lst.extend(patch_newline_mask)

            image_inputs = {
                "pixel_values": torch.cat(pixel_values_lst),
                "num_patches": num_patches,
            }
            if patch_pixel_values_lst:
                image_inputs["patch_pixel_values"] = torch.cat(patch_pixel_values_lst)
            if patch_newline_mask_lst:
                image_inputs["patch_newline_mask"] = torch.tensor(patch_newline_mask_lst, dtype=torch.bool)

            text = [self.replace_placeholder(t, self.image_token, image_repl_str_lst) for t in text]
            text_inputs = self.tokenizer(text)

        return BatchFeature(
            {
                **text_inputs,
                **image_inputs,
            },
            tensor_type=return_tensors,
        )

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Gemma
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Gemma
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)


__all__ = ["Step3VLProcessor"]
