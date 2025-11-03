import math
from typing import Optional, Union

import torch

from ...image_processing_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging


logger = logging.get_logger(__name__)


class DeepseekOcrProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False, "return_tensors": "pt"},
        "common_kwargs": {"return_tensors": "pt"},
        "images_kwargs": {"return_tensors": "pt"},
    }


class DeepseekOcrProcessor(ProcessorMixin):
    """
    Processor that wraps the DeepSeek OCR image processor and tokenizer.

    It expands ``<image>`` placeholders into the exact number of visual tokens required by
    the model based on the dynamic cropping strategy.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer, chat_template: Optional[str] = None):
        self.image_token = getattr(tokenizer, "image_token", "<image>")
        self.patch_size = getattr(image_processor, "patch_size", 16)
        self.downsample_ratio = getattr(image_processor, "downsample_ratio", 4)
        self.base_size = getattr(image_processor, "base_size", 1024)
        self.patch_size_side = getattr(image_processor, "patch_size_side", 640)
        self.global_query_count = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
        self.local_query_count = math.ceil((self.patch_size_side // self.patch_size) / self.downsample_ratio)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def _image_placeholder_block(self, width_crop_num: int, height_crop_num: int) -> tuple[str, int]:
        global_row = self.image_token * self.global_query_count + self.image_token
        global_block = global_row * self.global_query_count + self.image_token
        token_count = self.global_query_count * (self.global_query_count + 1) + 1

        local_block = ""
        if width_crop_num > 1 or height_crop_num > 1:
            local_row = self.image_token * (self.local_query_count * width_crop_num) + self.image_token
            local_block = local_row * (self.local_query_count * height_crop_num)
            token_count += (self.local_query_count * width_crop_num + 1) * (
                self.local_query_count * height_crop_num
            )

        return global_block + local_block, token_count

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput], None] = None,
        images=None,
        **kwargs: Unpack[DeepseekOcrProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            DeepseekOcrProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )

        defaults = DeepseekOcrProcessorKwargs._defaults

        text_kwargs = dict(defaults["text_kwargs"])
        text_kwargs.update(output_kwargs.get("text_kwargs", {}))

        images_kwargs = dict(defaults["images_kwargs"])
        images_kwargs.update(output_kwargs.get("images_kwargs", {}))

        common_kwargs = dict(defaults["common_kwargs"])
        common_kwargs.update(output_kwargs.get("common_kwargs", {}))

        texts: Optional[list[str]] = None
        if text is not None:
            if isinstance(text, str):
                texts = [text]
            elif isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text):
                texts = list(text)
            else:
                raise ValueError("`text` must be a string or a list of strings.")

        image_outputs = {}
        placeholder_counts = [prompt.count(self.image_token) for prompt in texts] if texts is not None else []

        if images is not None:
            image_outputs = self.image_processor(images, **images_kwargs)
            pixel_values = image_outputs["pixel_values"]
            image_spatial_crop = image_outputs["image_spatial_crop"]
            image_sizes = image_outputs.get("image_sizes")

            total_images = pixel_values.shape[0]

            num_img_tokens: list[int] = []
            if texts is not None:
                total_placeholders = sum(placeholder_counts)
                if total_placeholders != total_images:
                    raise ValueError(
                        f"Found {total_placeholders} image placeholder(s) in the text but {total_images} image crop(s) were provided."
                    )

                crop_list = image_spatial_crop.tolist()
                crop_index = 0
                expanded_prompts: list[str] = []

                for prompt, num_placeholders in zip(texts, placeholder_counts):
                    if num_placeholders == 0:
                        expanded_prompts.append(prompt)
                        continue

                    segments = prompt.split(self.image_token)
                    if len(segments) != num_placeholders + 1:
                        raise ValueError(
                            "Mismatch between image placeholders and detected crops while expanding the prompt."
                        )

                    parts = [segments[0]]
                    for _ in range(num_placeholders):
                        crop = crop_list[crop_index]
                        crop_index += 1
                        block, count = self._image_placeholder_block(int(crop[0]), int(crop[1]))
                        num_img_tokens.append(count)
                        parts.append(block)
                        parts.append(segments[_ + 1])
                    expanded_prompts.append("".join(parts))

                if crop_index != total_images:
                    raise ValueError("Not all image crops were consumed while building the prompts.")

                texts = expanded_prompts
            else:
                texts = [self.image_token] * total_images

            image_outputs = {
                "pixel_values": pixel_values,
                "image_spatial_crop": image_spatial_crop,
                "image_sizes": image_sizes,
            }

            if num_img_tokens:
                image_outputs["num_img_tokens"] = torch.tensor(
                    num_img_tokens, dtype=torch.long, device=pixel_values.device
                )

        data = {}
        if texts is not None:
            tokenized = self.tokenizer(texts, **text_kwargs)
            data.update(tokenized)
            image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
            if image_token_id is None:
                raise ValueError(f"The tokenizer does not know the special image token `{self.image_token}`.")
            images_seq_mask = (tokenized["input_ids"] == image_token_id).to(torch.bool)
            data["image_attention_mask"] = images_seq_mask

        if image_outputs:
            data.update(
                {
                    key: value
                    for key, value in image_outputs.items()
                    if value is not None
                }
            )

        return BatchFeature(data=data, tensor_type=common_kwargs.get("return_tensors", "pt"))

    @property
    def model_input_names(self):
        tokenizer_inputs = getattr(self.tokenizer, "model_input_names", [])
        vision_inputs = ["pixel_values", "image_spatial_crop", "image_sizes", "image_attention_mask", "num_img_tokens"]
        return list(dict.fromkeys(tokenizer_inputs + vision_inputs))


__all__ = ["DeepseekOcrProcessor"]
