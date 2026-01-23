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

import json
import math
import re
from textwrap import dedent

import numpy as np

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import AddedToken, PreTrainedTokenizerBase, TextInput
from ...utils import is_torch_available, is_vision_available
from ..auto.tokenization_auto import AutoTokenizer
from .image_processing_deepseek_ocr_fast import DeepseekOcrImageProcessorFast


if is_torch_available():
    import torch

if is_vision_available():
    # used only within the post-processing methods
    from PIL import Image, ImageDraw, ImageFont


DEEPSEEK_OCR_DEFAULT_CHAT_TEMPLATE = dedent(
    """
    {%- for message in messages %}
        {%- if message['content'] is string %}
{{ message['content'].rstrip() }}
        {%- else %}
            {%- set ns = namespace(previous_was_image=False) %}
            {%- for content in message['content'] %}
                {%- if content['type'] == 'image' %}
<image>
                    {%- set ns.previous_was_image = True %}
                {%- elif content['type'] == 'text' %}
{{- ('\n' if ns.previous_was_image else '') + content['text'].rstrip() }}
                    {%- set ns.previous_was_image = False %}
                {%- endif %}
            {%- endfor %}
        {%- endif %}
        {%- if not loop.last %}

        {%- endif %}
    {%- endfor %}
    """
).strip()


# Ensure DeepSeek OCR uses a default chat template even when loading a bare tokenizer (e.g., via AutoProcessor fallback).
if not getattr(PreTrainedTokenizerBase, "_deepseek_ocr_chat_patch", False):
    _orig_apply_chat_template = PreTrainedTokenizerBase.apply_chat_template

    def _deepseek_ocr_apply_chat_template(self, conversation, chat_template=None, **kwargs):
        template = chat_template
        if template is None and getattr(self, "chat_template", None) is None:
            model_type = getattr(getattr(self, "config", None), "model_type", None)
            name_or_path = (getattr(self, "name_or_path", "") or "").lower()
            if model_type == "deepseek_ocr" or "deepseek-ocr" in name_or_path:
                template = DEEPSEEK_OCR_DEFAULT_CHAT_TEMPLATE
                self.chat_template = template
        return _orig_apply_chat_template(self, conversation, chat_template=template, **kwargs)

    PreTrainedTokenizerBase.apply_chat_template = _deepseek_ocr_apply_chat_template
    PreTrainedTokenizerBase._deepseek_ocr_chat_patch = True


class DeepseekOcrProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"text_kwargs": {"return_mm_token_type_ids": False}}


class DeepseekOcrProcessor(ProcessorMixin):
    r"""
    Constructs a DeepSeek OCR processor which wraps an image processor and a tokenizer into a single processor.

    [`DeepseekOcrProcessor`] offers all the functionalities of [`DeepseekOcrImageProcessorFast`] and tokenizer.
    See the [`~DeepseekOcrProcessor.__call__`] and [`~DeepseekOcrProcessor.decode`] for more information.

    Args:
        image_processor (`DeepseekOcrImageProcessorFast`):
            The image processor to use for images.
        tokenizer (PreTrainedTokenizer):
            The tokenizer to use for text.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            The image token to use.
    """

    attributes = ["image_processor", "tokenizer"]
    tokenizer_class = "AutoTokenizer"
    image_processor_class = "DeepseekOcrImageProcessorFast"

    def __init__(
        self,
        image_processor,
        tokenizer,
        image_token="<image>",
        **kwargs,
    ):
        if hasattr(tokenizer, "image_token"):
            self.image_token = tokenizer.image_token
        else:
            self.image_token = image_token
            if tokenizer.convert_tokens_to_ids(self.image_token) is None:
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": [AddedToken(self.image_token, normalized=False, special=True)]}
                )

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        if self.image_token_id is None:
            raise ValueError(
                f"The tokenizer does not contain the special image token `{self.image_token}`. "
                "Please make sure it is added to the vocabulary before instantiating the processor."
            )
        self.image_token_ids = [self.image_token_id]
        tokenizer_chat_template = getattr(tokenizer, "chat_template", None)
        if kwargs.get("chat_template") is None:
            if tokenizer_chat_template is not None:
                kwargs["chat_template"] = tokenizer_chat_template
            else:
                kwargs["chat_template"] = DEEPSEEK_OCR_DEFAULT_CHAT_TEMPLATE
                tokenizer.chat_template = kwargs["chat_template"]

        super().__init__(image_processor, tokenizer, **kwargs)

    # @molbap hacky hacky, so that from_pretrained does not yell because there is no processor config in the hub repo
    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, processor_dict=None, **kwargs):
        try:
            return super()._get_arguments_from_pretrained(pretrained_model_name_or_path, processor_dict, **kwargs)
        except OSError as error:
            if "preprocessor_config" not in str(error):
                raise
            image_processor = DeepseekOcrImageProcessorFast()
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
            if getattr(tokenizer, "chat_template", None) is None:
                tokenizer.chat_template = DEEPSEEK_OCR_DEFAULT_CHAT_TEMPLATE
            return [image_processor, tokenizer]

    def __call__(
        self,
        text: TextInput | list[TextInput],
        images: ImageInput | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s).

        Args:
            text (`str`, `list[str]`):
                The sequence or batch of sequences to be encoded.
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, etc.):
                The image or batch of images to be prepared.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
            - **pixel_values** -- Pixel values to be fed to a model.
            - **image_attention_mask** -- Mask for image tokens in the input sequence.
            - **image_spatial_crop** -- Spatial crop information for images.
        """

        output_kwargs = self._merge_kwargs(DeepseekOcrProcessorKwargs, self.tokenizer.init_kwargs, **kwargs)
        image_kwargs = output_kwargs["images_kwargs"]

        image_inputs = self.image_processor(images, **image_kwargs) if images is not None else {}

        num_img_tokens = image_inputs.pop("num_img_tokens", [])

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        concatenated_prompt = "".join(text)
        if concatenated_prompt.count(self.image_token) != len(num_img_tokens):
            raise ValueError(
                f"Number of image tokens ({concatenated_prompt.count(self.image_token)}) in text "
                f"does not match number of images ({len(num_img_tokens)}). "
                f"Please add {self.image_token} token for each image."
            )

        image_count_iter = iter(num_img_tokens)
        processed_text = [
            re.sub(re.escape(self.image_token), lambda _: self.image_token * next(image_count_iter), t) for t in text
        ]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(
            processed_text,
            **output_kwargs["text_kwargs"],
            return_tensors=return_tensors,
        )
        self._check_special_mm_tokens(processed_text, text_inputs, modalities=["image"])

        input_ids = text_inputs["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            image_attention_mask = (input_ids == self.image_token_id).to(dtype=torch.bool)
        elif isinstance(input_ids, list):
            image_attention_mask: list[list[bool]] = []
            for ids in input_ids:
                mask = [token == self.image_token_id for token in ids]
                image_attention_mask.append(mask)
        else:
            raise TypeError("Unsupported type for input_ids returned by the tokenizer.")

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        data = {
            **text_inputs,
            **image_inputs,
            "image_attention_mask": image_attention_mask,
            "num_img_tokens": num_img_tokens,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, messages, **kwargs):
        """
        Ensure a chat template is available on the tokenizer before delegating.
        """
        chat_template = kwargs.pop("chat_template", None)
        if chat_template is None:
            if getattr(self, "chat_template", None) is None:
                self.chat_template = DEEPSEEK_OCR_DEFAULT_CHAT_TEMPLATE
            chat_template = self.chat_template
        if getattr(self.tokenizer, "chat_template", None) is None:
            self.tokenizer.chat_template = chat_template
        return super().apply_chat_template(messages, chat_template=chat_template, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names + image_processor_input_names + ["image_attention_mask", "num_img_tokens"]
            )
        )

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """Compute placeholder counts needed per image for vLLM multimodal scheduling."""

        vision_data = {}
        if image_sizes is None:
            return MultiModalData(**vision_data)

        base_size = getattr(self.image_processor, "base_size", {"height": 0})
        base_image_size = self._extract_dimension(base_size)
        patch_image_size = getattr(self.image_processor, "patch_size_side", None)
        if patch_image_size is None:
            patch_image_size = self._extract_dimension(getattr(self.image_processor, "size", {"height": 0}))
        max_crops = getattr(self.image_processor, "max_crops", 1)
        ratio_candidates = self._build_ratio_candidates(max_crops)
        downsample_ratio = getattr(self.image_processor, "downsample_ratio", 4)

        num_image_tokens = []
        num_image_patches = []
        for image_size in image_sizes:
            height, width = self._coerce_hw_tuple(image_size)
            if width <= patch_image_size and height <= patch_image_size:
                crop_ratio = (1, 1)
            elif ratio_candidates:
                aspect_ratio = width / height
                crop_ratio = self.image_processor.find_closest_aspect_ratio(
                    aspect_ratio, ratio_candidates, width, height, patch_image_size
                )
            else:
                crop_ratio = (1, 1)

            width_crop_num, height_crop_num = crop_ratio

            num_queries_base = math.ceil((base_image_size // 16) / downsample_ratio)
            num_queries = math.ceil((patch_image_size // 16) / downsample_ratio)
            tokenized_image_len = (num_queries_base + 1) * num_queries_base + 1
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image_len += (num_queries * width_crop_num + 1) * (num_queries * height_crop_num)

            num_image_tokens.append(tokenized_image_len)
            num_local_patches = max(width_crop_num * height_crop_num, 1)
            num_image_patches.append(num_local_patches)

        vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})
        return MultiModalData(**vision_data)

    def _build_ratio_candidates(self, max_crops):
        if max_crops < 2:
            return []
        target_ratios = {
            (i, j)
            for n in range(2, max_crops + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if 2 <= i * j <= max_crops
        }
        return sorted(target_ratios, key=lambda x: x[0] * x[1])

    @staticmethod
    def _extract_dimension(size_dict):
        if isinstance(size_dict, dict):
            return size_dict.get("height") or size_dict.get("shortest_edge") or size_dict.get("width", 0)
        height = getattr(size_dict, "height", None)
        if height is not None:
            return height
        shortest = getattr(size_dict, "shortest_edge", None)
        if shortest is not None:
            return shortest
        width = getattr(size_dict, "width", None)
        return width or 0

    @staticmethod
    def _coerce_hw_tuple(image_size):
        if isinstance(image_size, dict):
            height = image_size.get("height")
            width = image_size.get("width")
        elif hasattr(image_size, "tolist"):
            height, width = image_size.tolist()
        else:
            height, width = image_size
        return int(height), int(width)

    # All that follows is the original post-processing a bit tweaked that allows to have a nice OCR output on images.
    # unsure to keep there, why not, still belongs to processing utils.

    def _find_refs_with_spans(self, text: str):
        """
        Returns list of dicts:
        {"label": str, "coords_literal": str, "triplet": ["<ref>", label, coords_literal],
        "span": (start, end), "is_image": bool}
        """
        refs = []
        pat = re.compile(
            r"([A-Za-z_]\w*)\s*\[\[\s*"
            r"(\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+"
            r"(?:\s*\]\s*,\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+)*"
            r")\s*\]\]",
            re.MULTILINE,
        )
        for m in pat.finditer(text):
            label = m.group(1)
            coords_literal = "[[" + m.group(2) + "]]"  # JSON-compatible
            refs.append(
                {
                    "label": label,
                    "coords_literal": coords_literal,
                    "triplet": ["<ref>", label, coords_literal],
                    "span": (m.start(), m.end()),
                    "is_image": (label == "image"),
                }
            )
        return refs

    def extract_coordinates_and_label(self, ref_text):
        label_type = ref_text[1]
        cor_list = json.loads(ref_text[2])
        cor_list = [[int(a), int(b), int(c), int(d)] for a, b, c, d in cor_list]
        return (label_type, cor_list)

    def build_ref_texts_from_output(self, text: str):
        return [r["triplet"] for r in self._find_refs_with_spans(text)]

    def visualize_results(self, image, ref_texts):
        if isinstance(image, torch.Tensor):
            image = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        width, height = image.size
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
        draw2 = ImageDraw.Draw(overlay)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for ref in ref_texts:
            label_type, boxes = self.extract_coordinates_and_label(ref)
            color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
            color_a = color + (20,)
            for x1, y1, x2, y2 in boxes:
                x1 = int(x1 / 999 * width)
                y1 = int(y1 / 999 * height)
                x2 = int(x2 / 999 * width)
                y2 = int(y2 / 999 * height)
                if label_type == "title":
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                    draw2.rectangle([x1, y1, x2, y2], fill=color_a)
                else:
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    draw2.rectangle([x1, y1, x2, y2], fill=color_a)
                if font:
                    text_x, text_y = x1, max(0, y1 - 15)
                    textbox_width, textbox_height = draw.textbbox((0, 0), label_type, font=font)[2:]
                    draw.rectangle(
                        [text_x, text_y, text_x + textbox_width, text_y + textbox_height], fill=(255, 255, 255, 30)
                    )
                    draw.text((text_x, text_y), label_type, font=font, fill=color)
        img_draw.paste(overlay, (0, 0), overlay)
        return img_draw

    def return_image_crops_from_output(self, text: str, image):
        if isinstance(image, torch.Tensor):
            image = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        W, H = image.size
        crops = []
        for triplet in self.build_ref_texts_from_output(text):
            label_type, boxes = self.extract_coordinates_and_label(triplet)
            if label_type != "image":
                continue
            for x1, y1, x2, y2 in boxes:
                X1 = int(x1 / 999 * W)
                Y1 = int(y1 / 999 * H)
                X2 = int(x2 / 999 * W)
                Y2 = int(y2 / 999 * H)
                if X2 > X1 and Y2 > Y1:
                    crops.append(image.crop((X1, Y1, X2, Y2)))
        return crops

    def extract_markdown_and_crops(self, text: str, image, image_placeholder_fmt="![](images/{i}.png)", cleanup=True):
        """
        Returns (markdown_text, crops). Replaces image refs with placeholders and emoves non-image refs.
        """
        refs = self._find_refs_with_spans(text)
        i = 0
        pieces = []
        last = 0
        crops = []
        if isinstance(image, torch.Tensor):
            image = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        W, H = image.size

        for r in refs:
            start, end = r["span"]
            pieces.append(text[last:start])
            if r["is_image"]:
                _, boxes = self.extract_coordinates_and_label(r["triplet"])
                for x1, y1, x2, y2 in boxes:
                    X1 = int(x1 / 999 * W)
                    Y1 = int(y1 / 999 * H)
                    X2 = int(x2 / 999 * W)
                    Y2 = int(y2 / 999 * H)
                    if X2 > X1 and Y2 > Y1:
                        crops.append(image.crop((X1, Y1, X2, Y2)))
                        pieces.append(image_placeholder_fmt.format(i=i))
                        i += 1
            # non-image refs are dropped from text
            last = end
        pieces.append(text[last:])

        md = "".join(pieces)
        if cleanup:
            md = md.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
        return md, crops


__all__ = ["DeepseekOcrProcessor"]
