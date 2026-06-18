# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
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

from ...image_utils import ImageInput, make_flat_list_of_images, make_nested_list_of_images
from ...processing_utils import (
    BatchFeature,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class Lfm2VlTextKwargs(TextKwargs, total=False):
    """
    use_image_special_tokens (`bool`, *optional*, defaults to `True`):
        Whether to use special image tokens (`<|image_start|>` and `<|image_end|>`) to delimit image sequences
        in the text. When enabled, images are wrapped with these tokens to clearly mark image boundaries.
        When disabled, only the image token itself is used without delimiters.
    """

    use_image_special_tokens: bool | None


class Lfm2VlProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: Lfm2VlTextKwargs
    _defaults = {
        "images_kwargs": {
            "return_row_col_info": True,
        },
        "text_kwargs": {
            "use_image_special_tokens": True,
            "add_special_tokens": False,
            "padding": False,
            "is_split_into_words": False,
        },
    }


@auto_docstring
class Lfm2VlProcessor(ProcessorMixin):
    valid_processor_kwargs = Lfm2VlProcessorKwargs
    unused_input_names = ["image_rows", "image_cols", "image_sizes", "token_type_ids"]

    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template: str | None = None,
        **kwargs,
    ):
        self.image_token = getattr(tokenizer, "image_token", "<image>")
        self.image_token_id = (
            tokenizer.image_token_id
            if hasattr(tokenizer, "image_token_id")
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.image_start_token = getattr(tokenizer, "image_start_token", "<|image_start|>")
        self.image_end_token = getattr(tokenizer, "image_end_token", "<|image_end|>")
        self.image_thumbnail_token = getattr(tokenizer, "image_thumbnail_token", "<|img_thumbnail|>")
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        images, text, *_ = self.prepare_inputs_layout(images=images, text=text, **kwargs)
        self.validate_inputs(images=images, text=text, **kwargs)

        merged_kwargs = self._merge_kwargs(
            self.valid_processor_kwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs if hasattr(self, "tokenizer") else {},
            **kwargs,
        )
        merged_kwargs["images_kwargs"]["use_image_special_tokens"] = merged_kwargs["text_kwargs"].pop(
            "use_image_special_tokens"
        )

        processed_images = {}
        images_replacements = []
        if images is not None:
            processed_images, images_replacements = self._process_images(images, **merged_kwargs["images_kwargs"])

        text_inputs = {}
        return_tensors = merged_kwargs["text_kwargs"].get("return_tensors", None)
        if text is not None:
            return_mm_token_type_ids = merged_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
            return_text_replacement_offsets = merged_kwargs["text_kwargs"].pop(
                "return_text_replacement_offsets", False
            )

            text, text_replacement_offsets = self.get_text_with_replacements(text, images_replacements)
            text_inputs = self.tokenizer(text, **merged_kwargs["text_kwargs"])
            self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

            if return_text_replacement_offsets:
                text_inputs["text_replacement_offsets"] = text_replacement_offsets

            if return_mm_token_type_ids:
                text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])

        # Pop unused keys from the inputs, e.g. inputs used only to compute number of image tokens
        data = {**text_inputs, **processed_images}
        data = {k: v for k, v in data.items() if k not in self.unused_input_names}
        return BatchFeature(data, tensor_type=return_tensors, skip_tensor_conversion=self.skip_tensor_conversion)

    def prepare_inputs_layout(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos=None,
        audio=None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        images, text, *_ = super().prepare_inputs_layout(
            images=images, text=text, videos=videos, audio=audio, **kwargs
        )
        if images is not None:
            images = self.image_processor.fetch_images(images)
            images = make_nested_list_of_images(images)
        return images, text, videos, audio

    def validate_inputs(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        super().validate_inputs(images=images, text=text, **kwargs)
        if text is None and images is None:
            raise ValueError("You must provide one of `text` or `images`.")

        if images is not None and text is None:
            raise ValueError(
                "You must provide `text` when `images` is provided. Minimal text consists of a single image token."
            )

        if text is not None:
            n_images_in_text = [sample.count(self.image_token) for sample in text]
            if sum(n_images_in_text) > 0 and images is None:
                raise ValueError(f"We detected {sum(n_images_in_text)} tokens in the text but no images were passed")

            if images is not None:
                n_images_in_images = [len(sublist) for sublist in images]
                if n_images_in_images != n_images_in_text:
                    raise ValueError(
                        f"The number of images in the text {n_images_in_text} and images {n_images_in_images} should be the same."
                    )

    def _process_images(self, images: ImageInput, **kwargs):
        use_image_special_tokens = kwargs.pop("use_image_special_tokens")
        processed_images = self.image_processor(images, **kwargs)

        image_replacements = []
        images = make_flat_list_of_images(images)
        for idx in range(len(images)):
            replacement_text = self.replace_image_token(
                processed_images, image_idx=idx, use_image_special_tokens=use_image_special_tokens, **kwargs
            )
            image_replacements.append(replacement_text)
        return processed_images, image_replacements

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        rows = image_inputs["image_rows"][image_idx]
        cols = image_inputs["image_cols"][image_idx]
        image_size = image_inputs["image_sizes"][image_idx]

        use_thumbnail = kwargs.get("use_thumbnail", self.image_processor.use_thumbnail)
        tokens_per_tile, tokens_for_image = self._get_image_num_tokens(image_size, **kwargs)
        placeholder_tokens = self._build_image_tokens(
            rows, cols, tokens_per_tile, tokens_for_image, use_thumbnail, kwargs.get("use_image_special_tokens")
        )
        return placeholder_tokens

    def _build_image_tokens(
        self,
        rows: int,
        cols: int,
        tokens_per_tile: int,
        tokens_for_image: int,
        use_thumbnail: bool,
        use_image_special_tokens: bool,
    ) -> str:
        """Build the expanded token string for a single image."""
        parts = []

        if use_image_special_tokens:
            parts.append(self.image_start_token)

        is_multi_tile = rows > 1 or cols > 1
        if is_multi_tile:
            for row in range(rows):
                for col in range(cols):
                    if use_image_special_tokens:
                        parts.append(f"<|img_row_{row + 1}_col_{col + 1}|>")
                    parts.append(self.image_token * tokens_per_tile)

            if use_thumbnail:
                if use_image_special_tokens:
                    parts.append(self.image_thumbnail_token)
                parts.append(self.image_token * tokens_for_image)
        else:
            parts.append(self.image_token * tokens_for_image)

        if use_image_special_tokens:
            parts.append(self.image_end_token)

        return "".join(parts)

    def _compute_tokens_per_tile(self, tile_size: int, encoder_patch_size: int, downsample_factor: int) -> int:
        """Compute the number of tokens for a single tile."""
        num_patches = tile_size // encoder_patch_size
        downsampled_patches = math.ceil(num_patches / downsample_factor)
        return downsampled_patches * downsampled_patches

    def _compute_tokens_for_image(self, image_size: list[int], encoder_patch_size: int, downsample_factor: int) -> int:
        """Compute the number of tokens for a resized image (used for single-tile or thumbnail)."""
        image_height, image_width = image_size
        patches_h = math.ceil((image_height // encoder_patch_size) / downsample_factor)
        patches_w = math.ceil((image_width // encoder_patch_size) / downsample_factor)
        return patches_h * patches_w

    def _get_image_num_tokens(self, image_size: list[int], **images_kwargs) -> tuple[int, int]:
        """
        Compute token counts for image processing.

        Returns:
            tuple[int, int]: (tokens_per_tile, tokens_for_image)
                - tokens_per_tile: tokens for each tile in multi-tile mode
                - tokens_for_image: tokens for the resized image (single-tile) or thumbnail (multi-tile)
        """
        tile_size = images_kwargs.get("tile_size", self.image_processor.tile_size)
        downsample_factor = images_kwargs.get("downsample_factor", self.image_processor.downsample_factor)
        encoder_patch_size = images_kwargs.get("encoder_patch_size", self.image_processor.encoder_patch_size)

        tokens_per_tile = self._compute_tokens_per_tile(tile_size, encoder_patch_size, downsample_factor)
        tokens_for_image = self._compute_tokens_for_image(image_size, encoder_patch_size, downsample_factor)

        return tokens_per_tile, tokens_for_image


__all__ = ["Lfm2VlProcessor"]
