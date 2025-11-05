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

import json
import re
from typing import Optional, Union

import numpy as np

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import AddedToken, TextInput
from ...utils import TensorType, is_torch_available, is_vision_available, logging


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image, ImageDraw, ImageFont


logger = logging.get_logger(__name__)


class DeepseekOcrProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


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
        if "chat_template" not in kwargs and getattr(tokenizer, "chat_template", None) is not None:
            kwargs["chat_template"] = tokenizer.chat_template

        super().__init__(image_processor, tokenizer, **kwargs)

    def __call__(
        self,
        text: Union[TextInput, list[TextInput]],
        images: Optional[ImageInput] = None,
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

        text_kwargs = output_kwargs["text_kwargs"]
        return_tensors = text_kwargs.pop("return_tensors", None)
        tokenizer_kwargs = text_kwargs.copy()
        if return_tensors is not None:
            tokenizer_kwargs["return_tensors"] = return_tensors
        if (
            tokenizer_kwargs.get("padding") == "max_length"
            and "max_length" in tokenizer_kwargs
            and "truncation" not in tokenizer_kwargs
        ):
            tokenizer_kwargs["truncation"] = True
        text_inputs = self.tokenizer(processed_text, **tokenizer_kwargs)

        input_ids = text_inputs["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
                text_inputs["input_ids"] = input_ids
                for key in ("attention_mask", "position_ids", "token_type_ids"):
                    value = text_inputs.get(key)
                    if isinstance(value, torch.Tensor) and value.ndim == 1:
                        text_inputs[key] = value.unsqueeze(0)
            image_attention_mask = (input_ids == self.image_token_id).to(dtype=torch.bool)
        elif isinstance(input_ids, list):
            image_attention_mask: list[list[bool]] = []
            for ids in input_ids:
                mask = [token == self.image_token_id for token in ids]
                image_attention_mask.append(mask)
        else:
            raise TypeError("Unsupported type for input_ids returned by the tokenizer.")

        data = {
            **text_inputs,
            **image_inputs,
            "image_attention_mask": image_attention_mask,
            "num_img_tokens": num_img_tokens,
        }

        batch = BatchFeature(data=data, tensor_type=return_tensors)
        batch["num_img_tokens"] = num_img_tokens
        return batch

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

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names + image_processor_input_names + ["image_attention_mask", "num_img_tokens"]
            )
        )

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        outputs = super().apply_chat_template(conversations, chat_template=chat_template, **kwargs)
        tensor_type = kwargs.get("return_tensors")
        if hasattr(tensor_type, "value"):
            tensor_type = tensor_type.value
        if tensor_type == TensorType.PYTORCH:
            tensor_type = "pt"
        if isinstance(outputs, BatchFeature) and tensor_type == "pt" and "num_img_tokens" in outputs:
            num_img_tokens = outputs["num_img_tokens"]
            if not isinstance(num_img_tokens, torch.Tensor):
                outputs["num_img_tokens"] = torch.tensor(num_img_tokens)
        return outputs

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
