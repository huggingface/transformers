# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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


import numpy as np

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, get_image_size, is_valid_image
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class GotOcr2TextKwargs(TextKwargs, total=False):
    """
    format (`bool`, *optional*, defaults to `False`):
        Whether to request formatted output from the OCR model. When enabled, the model is instructed to return
        structured and formatted text output rather than raw OCR results.
    """

    format: bool | None


# Kept here for BC. Identical to GotOcr2ImageProcessorKwargs once deprecated arguments are removed.
class GotOcr2ImagesKwargs(ImagesKwargs, total=False):
    """
    crop_to_patches (`bool`, *optional*, defaults to `False`):
        Whether to crop images into patches before processing. When enabled, large images are divided into
        smaller patches for more efficient OCR processing.
    min_patches (`int`, *optional*, defaults to `1`):
        Minimum number of patches to generate when cropping images. This ensures that even small images are
        processed with at least this many patches.
    max_patches (`int`, *optional*, defaults to `12`):
        Maximum number of patches to generate when cropping images. Large images will be divided into at most
        this many patches to control computational complexity.
    box (`list`, `tuple[float, float]`, or `tuple[float, float, float, float]`, *optional*):
        Deprecated, pass directly as `box` to [`GotOcr2Processor.__call__`] instead.
        Bounding box coordinates for OCR region of interest. Can be specified as a single box `[x1, y1, x2, y2]`
        or a list of boxes. Coordinates are normalized to the range [0, 1000] based on the image dimensions.
        If not provided, OCR is performed on the entire image.
    color (`str`, *optional*):
        Deprecated, use [`GotOcr2ProcessorKwargs.color`] instead.
        Color filter specification for OCR. When provided, the OCR query is prefixed with the color information
        to focus on text of a specific color (e.g., "red", "blue").
    num_image_tokens (`int`, *optional*, defaults to `256`):
        Deprecated, use [`GotOcr2ProcessorKwargs.num_image_tokens`] instead.
        Number of image tokens (patches) to use per image. This controls the resolution of the image representation
        passed to the model. Higher values provide more detail but increase computational cost.
    multi_page (`bool`, *optional*, defaults to `False`):
        Deprecated, use [`GotOcr2ProcessorKwargs.multi_page`] instead.
        Whether the input consists of multi-page documents. When enabled, images can be provided as nested lists
        where each inner list represents a page, and OCR is performed across all pages with appropriate handling
        of page boundaries.
    """

    crop_to_patches: bool
    min_patches: int
    max_patches: int
    box: list | tuple[float, float] | tuple[float, float, float, float] | None
    color: str | None
    num_image_tokens: int
    multi_page: bool


class GotOcr2ProcessorKwargs(ProcessingKwargs, total=False):
    """
    color (`str`, *optional*):
        Color filter specification for OCR. When provided, the OCR query is prefixed with the color information
        to focus on text of a specific color (e.g., "red", "blue").
    num_image_tokens (`int`, *optional*, defaults to `256`):
        Number of image tokens (patches) to use per image. This controls the resolution of the image representation
        passed to the model. Higher values provide more detail but increase computational cost.
    multi_page (`bool`, *optional*, defaults to `False`):
        Whether the input consists of multi-page documents. When enabled, images can be provided as nested lists
        where each inner list represents a page, and OCR is performed across all pages with appropriate handling
        of page boundaries.
    """

    text_kwargs: GotOcr2TextKwargs
    images_kwargs: GotOcr2ImagesKwargs

    color: str | None
    num_image_tokens: int
    multi_page: bool


def preprocess_box_annotation(box: list | tuple, image_size: tuple[int, int]) -> list:
    """
    Convert box annotation to the format [x1, y1, x2, y2] in the range [0, 1000].
    """
    width, height = image_size
    if len(box) == 4:
        box[0] = int(box[0] / width * 1000)
        box[1] = int(box[1] / height * 1000)
        box[2] = int(box[2] / width * 1000)
        box[3] = int(box[3] / height * 1000)
    else:
        raise ValueError("Box must be a list or tuple of lists in the form [x1, y1, x2, y2].")

    return list(box)


@auto_docstring
class GotOcr2Processor(ProcessorMixin):
    valid_processor_kwargs = GotOcr2ProcessorKwargs

    color: str | None = None
    num_image_tokens: int = 256
    multi_page: bool = False

    text_kwargs = {
        "format": False,
    }
    images_kwargs = {
        "crop_to_patches": False,
        "min_patches": 1,
        "max_patches": 12,
    }

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.message_start_token = "<|im_start|>"
        self.message_end_token = "<|im_end|>"
        self.img_start_token = "<img>"
        self.img_end_token = "</img>"
        self.img_pad_token = "<imgpad>"
        self.image_token = "<imgpad>"  # keep the above for BC, but we need to call it `image_token`
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.system_query = "system\nYou should follow the instructions carefully and explain your answers in detail."

    def _make_list_of_inputs(self, images, text, box, color, multi_page):
        if not isinstance(images, (list, tuple)):
            images = [images]
            if multi_page:
                logger.warning("Multi-page inference is enabled but only one image is passed.")
                images = [images]
        elif isinstance(images[0], (list, tuple)) and not multi_page:
            raise ValueError("Nested images are only supported with `multi_page` set to `True`.")
        elif not isinstance(images[0], (list, tuple)) and multi_page:
            images = [images]

        if isinstance(text, str):
            text = [text]

        if not isinstance(box[0], (list, tuple)):
            # Use the same box for all images
            box = [box for _ in range(len(images))]
        if not isinstance(color, (list, tuple)):
            color = [color for _ in range(len(images))]

        return images, text, box, color

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        box: list | tuple[float, float] | tuple[float, float, float, float] | None = None,
        **kwargs: Unpack[GotOcr2ProcessorKwargs],
    ) -> BatchFeature:
        r"""
        box (`list`, `tuple[float, float]`, or `tuple[float, float, float, float]`, *optional*):
            Bounding box coordinates for OCR region of interest. Can be specified as a single box `[x1, y1, x2, y2]`
            or a list of boxes. Coordinates are normalized to the range [0, 1000] based on the image dimensions.
            If not provided, OCR is performed on the entire image.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """

        merged_kwargs = self._merge_kwargs(
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        for key in ("box", "color", "num_image_tokens", "multi_page"):
            if key in merged_kwargs["images_kwargs"]:
                logger.warning_once(
                    f"Passing `{key}` in `images_kwargs` is deprecated "
                    "and will be removed in a future version. "
                    "Pass it directly to the processor instead."
                )

        # "images_kwargs" has priority for backwards compatibility
        box = merged_kwargs["images_kwargs"].pop("box", box)
        for key in ("color", "num_image_tokens", "multi_page"):
            merged_kwargs[key] = merged_kwargs["images_kwargs"].pop(key, merged_kwargs[key])

        images, text, *_ = self.prepare_inputs_layout(images=images, text=text, box=box, **merged_kwargs)
        self.validate_inputs(images=images, text=text, **merged_kwargs)

        processed_images = {}
        images_replacements = []
        if images is not None and hasattr(self, "image_processor"):
            processed_images, images_replacements = self._process_images(
                images,
                multi_page=merged_kwargs["multi_page"],
                num_image_tokens=merged_kwargs["num_image_tokens"],
                **merged_kwargs["images_kwargs"],
            )

        text_inputs = {}
        return_tensors = merged_kwargs["text_kwargs"].get("return_tensors", None)
        if getattr(self, "tokenizer", None) is not None and text is not None:
            for key in ("return_mm_token_type_ids", "return_text_replacement_offsets"):
                if key in merged_kwargs["text_kwargs"]:
                    logger.warning_once(
                        f"Passing `{key}` in `text_kwargs` is deprecated "
                        "and will be removed in a future version. "
                        "Pass it directly to the processor instead."
                    )
            # return_mm_token_type_ids in text_kwargs has priority for backwards compatibility
            return_mm_token_type_ids = merged_kwargs["text_kwargs"].pop(
                "return_mm_token_type_ids", merged_kwargs["return_mm_token_type_ids"]
            )
            # return_text_replacement_offsets in text_kwargs has priority for backwards compatibility
            return_text_replacement_offsets = merged_kwargs["text_kwargs"].pop(
                "return_text_replacement_offsets", merged_kwargs["return_text_replacement_offsets"]
            )

            text, text_replacement_offsets = self.get_text_with_replacements(
                text,
                images_replacements,
            )
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

    def prepare_inputs_layout(self, images=None, text=None, box=None, color=None, multi_page=False, **kwargs):
        if images is None:
            raise ValueError(f"You must provide valid `images` for {self.__class__.__name__}, found `None`")

        images, text, *_ = super().prepare_inputs_layout(images=images, text=text, **kwargs)

        box = [None] if box is None else box
        format_output = kwargs["text_kwargs"].pop("format")
        crop_to_patches = kwargs["images_kwargs"].get("crop_to_patches")
        images, text, box, color = self._make_list_of_inputs(images, text, box, color, multi_page)

        if text is None:
            text = []
            if multi_page:
                image_sizes = [get_image_size(image) for image_group in images for image in image_group]
            else:
                image_sizes = [get_image_size(image) for image in images]

            for box_single, color_single, size in zip(box, color, image_sizes):
                if box_single[0] is not None:
                    box_single = preprocess_box_annotation(box_single, size)
                query = (
                    f"{f'[{color_single}] ' if color_single is not None else ''}"
                    f"{str(box_single) if box_single[0] is not None else ''} "
                    "OCR"
                    f"{' with format' if format_output else ''}"
                    f"{' across multi pages' if multi_page else ''}"
                    f"{' upon the patch reference' if crop_to_patches else ''}"
                    ": "
                )
                # build a conversation manually, ckpt has not chat template :(
                prompt = (
                    self.message_start_token
                    + self.system_query
                    + self.message_end_token
                    + self.message_start_token
                    + "user\n"
                    + self.img_start_token
                    + self.image_token  # <- the token to be expanded later
                    + self.img_end_token
                    + "\n"
                    + query
                    + self.message_end_token
                    + self.message_start_token
                    + "assistant\n"
                )
                text.append(prompt)

        return images, text, None, None

    def _process_images(self, images: ImageInput, multi_page: bool = False, num_image_tokens: int = 256, **kwargs):
        processed_images = self.image_processor(images, **kwargs)

        if multi_page:
            num_pages_per_batch = [len(image_group) for image_group in images]
        else:
            num_pages_per_batch = [1 for _ in range(len(images))]
        patch_indices = np.cumsum(num_pages_per_batch)

        image_replacements = []
        # Important to not flatten the images here!
        for idx in range(len(images)):
            if is_valid_image(images[idx]) or (isinstance(images, (list, tuple)) and len(images[idx]) > 0):
                replacement_text = self.replace_image_token(
                    processed_images,
                    image_idx=idx,
                    num_pages_per_batch=num_pages_per_batch,
                    patch_indices=patch_indices,
                    num_image_tokens=num_image_tokens,
                )
                image_replacements.append(replacement_text)
        return processed_images, image_replacements

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        num_pages = kwargs["num_pages_per_batch"][image_idx]
        num_image_tokens = kwargs["num_image_tokens"]
        current_patch_index = kwargs["patch_indices"][image_idx - 1] if image_idx > 0 else 0
        num_patches = sum(image_inputs["num_patches"][current_patch_index : current_patch_index + num_pages])
        return self.image_token * num_image_tokens * num_patches

    @property
    def unused_input_names(self) -> list[str]:
        "Input names returned always by subprocessors but not used in model's `forward`"
        return ["num_patches"]


__all__ = ["GotOcr2Processor"]
