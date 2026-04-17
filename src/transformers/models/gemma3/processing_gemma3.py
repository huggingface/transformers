# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image, make_nested_list_of_images, valid_images
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring


class Gemma3ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": True,
        },
        "images_kwargs": {
            "do_convert_rgb": True,
            "do_pan_and_scan": False,
            "pan_and_scan_min_crop_size": 256,
            "pan_and_scan_max_num_crops": 4,
            "pan_and_scan_min_ratio_to_activate": 1.2,
        },
    }


@auto_docstring
class Gemma3Processor(ProcessorMixin):
    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template=None,
        image_seq_length: int = 256,
        **kwargs,
    ):
        self.image_seq_length = image_seq_length
        self.image_token_id = tokenizer.image_token_id
        self.boi_token = tokenizer.boi_token
        self.image_token = tokenizer.boi_token
        image_tokens_expanded = "".join([tokenizer.image_token] * image_seq_length)
        self.full_image_sequence = f"\n\n{tokenizer.boi_token}{image_tokens_expanded}{tokenizer.eoi_token}\n\n"

        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[Gemma3ProcessorKwargs],
    ) -> BatchFeature:
        self.validate_inputs(images=images, text=text, **kwargs)
        images, text = self.prepare_inputs_layout(images=images, text=text)

        output_kwargs = self._merge_kwargs(
            Gemma3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = {}
        images_replacements = []
        if images is not None:
            image_inputs, images_replacements = self._process_images(images, **output_kwargs["images_kwargs"])
            image_inputs.pop("num_crops", None)  # unused by model

        # Replace image tokens by the full expanded sequence
        text, text_replacement_offsets = self.get_text_replacement(text, images_replacements=images_replacements)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])
        # self._check_special_mm_tokens(text, text_inputs, modalities=["image"]) # BOI token in gemma, FIXME

        if return_mm_token_type_ids:
            text_inputs["token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def prepare_inputs_layout(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
    ):
        if text is not None and isinstance(text, str):
            text = [text]

        if images is not None:
            images = make_nested_list_of_images(images)

        # Create empty text to be replaced with placeholders
        if images and not text:
            text = [" ".join([self.boi_token] * len(image_list)) for image_list in images]

        return images, text

    def validate_inputs(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        super().validate_inputs(images, text, **kwargs)

        if text is None and images is None:
            raise ValueError("You must provide either `text` or `images`.")

        if text is not None:
            n_images_in_text = [sample.count(self.boi_token) for sample in text]
            if images is not None and isinstance(images, (list, tuple)) and is_valid_image(images[0]):
                n_images_in_text = [sample.count(self.boi_token) for sample in text]
                if sum(n_images_in_text) != len(images):
                    raise ValueError(
                        f"The total number of {self.boi_token} tokens in the prompts should be the same as the number of images passed."
                        f" Found {sum(n_images_in_text)} {self.boi_token} tokens and {len(images)} images."
                    )
            elif images is None and any(n_images_in_text):
                raise ValueError(
                    f"Found {sum(n_images_in_text)} {self.boi_token} tokens in the text but no images were passed."
                )

        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid input images. Please provide a single image or a list of images or a list of list of images."
            )

    def replace_image_token(self, processed_images: dict, image_idx: int) -> str:
        num_crops = processed_images["num_crops"][image_idx]
        if num_crops > 0:
            formatted_image_text = (
                f"Here is the original image {self.full_image_sequence} and here are some crops to help you see better "
                + " ".join([self.full_image_sequence] * num_crops)
            )
            return formatted_image_text
        else:
            return self.full_image_sequence

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.

        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.

        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """

        vision_data = {}
        if image_sizes is not None:
            # NOTE: no image cropping supported yet
            num_image_tokens = [self.image_seq_length] * len(image_sizes)
            num_image_patches = [1] * len(image_sizes)

            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names + ["token_type_ids"]
        image_processor_input_names = self.image_processor.model_input_names

        image_processor_input_names = [name for name in image_processor_input_names if name != "num_crops"]
        return list(tokenizer_input_names + image_processor_input_names)


__all__ = ["Gemma3Processor"]
