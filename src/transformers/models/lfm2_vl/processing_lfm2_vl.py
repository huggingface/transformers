import math
from typing import Union

import torch
from PIL import Image

from ...feature_extraction_utils import BatchFeature
from ...image_transforms import to_pil_image
from ...image_utils import ImageInput, make_nested_list_of_images
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from ...tokenization_utils_base import BatchEncoding, TextInput
from ...utils import logging


logger = logging.get_logger(__name__)


# resize adapted from qwen2.5
# https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
def round_by_factor(number: float, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    """Find the closest aspect ratio from target_ratios to match the input aspect ratio.

    Args:
        aspect_ratio: The aspect ratio to match (width/height).
        target_ratios: List of possible aspect ratios as tuples of (width, height) integers.
        width: Original image width in pixels.
        height: Original image height in pixels.
        image_size: Base size for calculating target area.

    Returns:
        tuple[int, int]: The best matching ratio as (width, height) integers.
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)

        # update best ratio if we found a closer match
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        # if equally close, prefer the ratio that better matches the original image area
        elif ratio_diff == best_ratio_diff:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best_ratio = ratio

    return best_ratio


class Lfm2VlImagesKwargs(ImagesKwargs, total=False):
    return_row_col_info: bool | None
    max_image_size: dict[str, int] | None


class Lfm2VlProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Lfm2VlImagesKwargs

    _defaults = {
        "text_kwargs": {
            "add_special_tokens": False,
            "padding": False,
            "is_split_into_words": False,
        },
        "images_kwargs": {
            "do_resize": False,
        },
    }


class Lfm2VlProcessor(ProcessorMixin):
    r"""
    Constructs a Lfm2Vl processor which wraps a Lfm2Tokenizer tokenizer and Lfm2Vl image processor into a single processor.

    [`Lfm2VlProcessor`] offers all the functionalities of [`Siglip2ImageProcessor`] and [`Lfm2Tokenizer`].

    Args:
        image_processor (`Siglip2ImageProcessor`):
            An instance of [`Siglip2ImageProcessor`]. The image processor is a required input.
        tokenizer (`PreTrainedTokenizerBase`):
            An instance of [`PreTrainedTokenizerBase`]. This should correspond with the model's text model. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Siglip2ImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template: str,
        use_image_special_tokens: bool,
        downsample_factor: int,
        do_image_splitting: bool,
        min_tiles: int,
        max_tiles: int,
        use_thumbnail: bool,
        min_image_tokens: int,
        max_image_tokens: int,
        encoder_patch_size: int,
        tile_size: int,
        max_pixels_tolerance: float,
        auto_map: dict[str, str] | None = None,
        **kwargs,
    ):
        self.image_token = getattr(tokenizer, "image_token", "<image>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.use_image_special_tokens = use_image_special_tokens
        self.image_start_token = getattr(tokenizer, "image_start_token", "<|image_start|>")
        self.image_end_token = getattr(tokenizer, "image_end_token", "<|image_end|>")
        self.image_thumbnail_token = getattr(tokenizer, "image_thumbnail", "<|img_thumbnail|>")
        self.downsample_factor = downsample_factor
        self.do_image_splitting = do_image_splitting
        self.min_tiles = min_tiles
        self.max_tiles = max_tiles
        self.use_thumbnail = use_thumbnail
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        self.encoder_patch_size = encoder_patch_size
        self.tile_size = tile_size
        self.max_pixels_tolerance = max_pixels_tolerance
        self.chat_template = chat_template
        self.auto_map = auto_map
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

        max_thumbnail_image_patches = max_image_tokens * downsample_factor**2
        tile_size_patches = (tile_size // encoder_patch_size) ** 2 if self.do_image_splitting else 0
        self.max_num_patches = max(
            max_thumbnail_image_patches,
            tile_size_patches,
        )

        self.image_processor.max_num_patches = self.max_num_patches

    def _high_res_preprocessor(
        self,
        image: Image.Image,
        min_tiles,
        max_tiles,
        tile_size,
    ) -> tuple[list[Image.Image], int, int, int]:
        """Process a high resolution image into patches.
        This method splits a high resolution image into a grid of smaller patches while trying to maintain
        the original aspect ratio. It finds the optimal grid configuration within the specified tile constraints.
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # generate valid patch grid configurations (width, height)
        target_ratios = [
            (w, h)
            for n in range(min_tiles, max_tiles + 1)
            for w in range(1, n + 1)
            for h in range(1, n + 1)
            if min_tiles <= w * h <= max_tiles
        ]
        target_ratios = sorted(set(target_ratios), key=lambda x: x[0] * x[1])

        # default to 1x1 if no valid configurations found
        if not target_ratios:
            return [], 0, 0

        # find best matching grid configuration
        grid_width, grid_height = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, tile_size
        )

        target_width = tile_size * grid_width
        target_height = tile_size * grid_height
        total_patches = grid_width * grid_height

        # resize and split image into patches
        resized_img = image.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)
        patches = []

        for i in range(total_patches):
            # calculate patch coordinates
            col = i % grid_width
            row = i // grid_width
            box = (
                col * tile_size,
                row * tile_size,
                (col + 1) * tile_size,
                (row + 1) * tile_size,
            )
            patch = resized_img.crop(box)
            patches.append(patch)

        num_rows = grid_height
        num_columns = grid_width

        return patches, num_rows, num_columns

    def _smart_resize(
        self,
        image: Image.Image,
        downsample_factor: int,
        min_image_tokens: int,
        max_image_tokens: int,
        encoder_patch_size: int,
    ) -> Image.Image:
        """
        Rescales the image so that the following conditions are met:
        1. Both dimensions (height and width) are divisible by 'encoder_patch_size' * 'downsample_factor'.
           This ensures no padding is needed in the downsampling step.
        2. The total number of pixels is within the range ['smart_resize_min_pixels', 'smart_resize_max_pixels'].
        3. The aspect ratio of the image is maintained as closely as possible.
        """
        width, height = image.size

        total_factor = encoder_patch_size * downsample_factor
        smart_resize_min_pixels = min_image_tokens * encoder_patch_size**2 * downsample_factor**2
        smart_resize_max_pixels = max_image_tokens * encoder_patch_size**2 * downsample_factor**2

        h_bar = max(total_factor, round_by_factor(height, total_factor))
        w_bar = max(total_factor, round_by_factor(width, total_factor))

        if h_bar * w_bar > smart_resize_max_pixels:
            beta = math.sqrt((height * width) / smart_resize_max_pixels)
            h_bar = max(total_factor, floor_by_factor(height / beta, total_factor))
            w_bar = max(total_factor, floor_by_factor(width / beta, total_factor))
        elif h_bar * w_bar < smart_resize_min_pixels:
            beta = math.sqrt(smart_resize_min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, total_factor)
            w_bar = ceil_by_factor(width * beta, total_factor)

        resized_img = image.resize((w_bar, h_bar), resample=Image.Resampling.BILINEAR)
        return resized_img

    def _get_tokens_num(self, image_height: int, image_width: int) -> int:
        num_patches_height = image_height // self.encoder_patch_size
        num_patches_width = image_width // self.encoder_patch_size

        dwn_num_patches_height = math.ceil(num_patches_height / self.downsample_factor)
        dwn_num_patches_width = math.ceil(num_patches_width / self.downsample_factor)

        return dwn_num_patches_height * dwn_num_patches_width

    def _is_img_too_large(
        self,
        image: Image.Image,
        max_image_tokens: int,
        encoder_patch_size: int,
        max_pixels_tolerance: float,
    ) -> bool:
        """Check if the image is too large to be processed as one tile."""
        width, height = image.size

        h_bar = max(encoder_patch_size, round_by_factor(height, encoder_patch_size))
        w_bar = max(encoder_patch_size, round_by_factor(width, encoder_patch_size))
        return (
            h_bar * w_bar > max_image_tokens * encoder_patch_size**2 * self.downsample_factor**2 * max_pixels_tolerance
        )

    def _resize_and_maybe_split(
        self,
        image: ImageInput,
        downsample_factor: int,
        min_tiles: int,
        max_tiles: int,
        use_thumbnail: bool,
        min_image_tokens: int,
        max_image_tokens: int,
        encoder_patch_size: int,
        tile_size: int,
        max_pixels_tolerance: float,
    ) -> tuple[list[Image.Image], int, int, int, int]:
        """Apply smart resize and maybe split the image into tiles if image too large.
        Return:
            image_tiles: ImageInput
            num_tokens_per_tile: int
            num_rows: int
            num_cols: int
            num_thumbnail_tokens: int
        """
        image = to_pil_image(image)
        do_image_splitting = not min_tiles == max_tiles == 1
        if (
            self._is_img_too_large(
                image,
                max_image_tokens,
                encoder_patch_size,
                max_pixels_tolerance,
            )
            and do_image_splitting
        ):
            image_tiles, num_rows, num_cols = self._high_res_preprocessor(image, min_tiles, max_tiles, tile_size)
            if len(image_tiles) > 1:
                num_thumbnail_tokens = 0
                if use_thumbnail:
                    thumbnail_image = self._smart_resize(
                        image,
                        downsample_factor,
                        min_image_tokens,
                        max_image_tokens,
                        encoder_patch_size,
                    )
                    num_thumbnail_tokens = self._get_tokens_num(thumbnail_image.height, thumbnail_image.width)
                    image_tiles.append(thumbnail_image)

                return (
                    image_tiles,
                    self._get_tokens_num(tile_size, tile_size),
                    num_rows,
                    num_cols,
                    num_thumbnail_tokens,
                )
        else:
            image = self._smart_resize(
                image,
                downsample_factor,
                min_image_tokens,
                max_image_tokens,
                encoder_patch_size,
            )
            return [image], self._get_tokens_num(image.height, image.width), 1, 1, 0

    def process_vision(
        self,
        text: list[str],
        images: list[list[ImageInput]],
        use_image_special_tokens: bool,
        downsample_factor: int,
        min_tiles: int,
        max_tiles: int,
        use_thumbnail: bool,
        min_image_tokens: int,
        max_image_tokens: int,
        encoder_patch_size: int,
        tile_size: int,
        max_pixels_tolerance: float,
        output_kwargs: dict,
    ):
        if text is not None:
            n_images_in_text = [sample.count(self.image_token) for sample in text]

        n_images_in_images = [len(sublist) for sublist in images]

        if n_images_in_images != n_images_in_text:
            raise ValueError(
                f"The number of images in the text {n_images_in_text} and images {n_images_in_images} should be the same."
            )

        prompt_strings = []
        image_inputs = []

        for sample_text, sample_images in zip(text, images, strict=False):
            split_sample = sample_text.split(self.image_token)
            sample_tiles = []
            sample_text_with_image_tokens = ""
            for i, image in enumerate(sample_images):
                sample_text_with_image_tokens += split_sample[i]
                if use_image_special_tokens:
                    sample_text_with_image_tokens += self.image_start_token
                (
                    image_tiles,
                    num_tokens_per_tile,
                    num_rows,
                    num_cols,
                    num_thumbnail_tokens,
                ) = self._resize_and_maybe_split(
                    image,
                    downsample_factor,
                    min_tiles,
                    max_tiles,
                    use_thumbnail,
                    min_image_tokens,
                    max_image_tokens,
                    encoder_patch_size,
                    tile_size,
                    max_pixels_tolerance,
                )

                if len(image_tiles) > 1:
                    for row in range(num_rows):
                        for col in range(num_cols):
                            if use_image_special_tokens:
                                sample_text_with_image_tokens += f"<|img_row_{row + 1}_col_{col + 1}|>"
                            sample_text_with_image_tokens += self.image_token * num_tokens_per_tile

                    if num_thumbnail_tokens > 0:
                        if use_image_special_tokens:
                            sample_text_with_image_tokens += self.image_thumbnail_token
                        sample_text_with_image_tokens += self.image_token * num_thumbnail_tokens
                else:
                    sample_text_with_image_tokens += self.image_token * num_tokens_per_tile

                if use_image_special_tokens:
                    sample_text_with_image_tokens += self.image_end_token

                sample_text_with_image_tokens += split_sample[i + 1]
                sample_tiles.extend(image_tiles)

            prompt_strings.append(sample_text_with_image_tokens)
            image_inputs.append(sample_tiles)

        torch.save(image_inputs, "raw_image_inputs_hf.pt")

        image_inputs = self.image_processor(image_inputs, **output_kwargs["images_kwargs"])

        if text is None:
            return None, image_inputs

        return prompt_strings, image_inputs

    def __call__(
        self,
        images: ImageInput | list[ImageInput] | list[list[ImageInput]] = None,
        text: Union[TextInput, "PreTokenizedInput", list[TextInput], list["PreTokenizedInput"]] = None,
        use_image_special_tokens: bool | None = None,
        downsample_factor: int | None = None,
        min_image_tokens: int | None = None,
        max_image_tokens: int | None = None,
        do_image_splitting: bool | None = None,
        min_tiles: int | None = None,
        max_tiles: int | None = None,
        use_thumbnail: bool | None = None,
        encoder_patch_size: int | None = None,
        tile_size: int | None = None,
        max_pixels_tolerance: float | None = None,
        **kwargs: Unpack[Lfm2VlProcessorKwargs],
    ) -> BatchEncoding:
        """
        Processes the input prompts and returns a BatchFeature.

        Example:

        ```python
        >>> import requests
        >>> from transformers import AutoProcessor
        >>> from transformers.image_utils import load_image
        >>> processor = AutoProcessor.from_pretrained("", trust_remote_code=True)

        >>> url1 = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        >>> url2 = "https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg"

        >>> image1, image2 = load_image(url1), load_image(url2)
        >>> images = [image1, image2]

        >>> conversation = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "url": image1},
        ...             {"type": "image", "url": image2},
        ...             {"type": "text", "text": "Compare the two images."},
        ...         ],
        ...     },
        ... ]
        >>> chat_inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        >>> outputs = processor(images=images, text=chat_inputs, return_tensors="pt")
        >>> input_ids = outputs.input_ids
        >>> input_tokens = processor.tokenizer.batch_decode(input_ids)
        >>> print(input_tokens)
        '['user\nCompare the two images.\nassistant\n']'
        ```

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`, *optional*):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. If is of type `list[ImageInput]`, it's assumed that this is for a single prompt i.e. of batch size 1.
            text (`TextInput`, *optional*):
                The sequence or batch of sequences to be encoded.
                Wherever an image token, `<image>` is encountered it is expanded to a proper sequence of image tokens.
            return_tensors (`str | TensorType`, *optional*):
                If set, will return tensors of a particular framework. See [`PreTrainedTokenizerFast.__call__`] for more
                information.
        """
        use_image_special_tokens = (
            use_image_special_tokens if use_image_special_tokens is not None else self.use_image_special_tokens
        )
        downsample_factor = downsample_factor if downsample_factor is not None else self.downsample_factor
        do_image_splitting = do_image_splitting if do_image_splitting is not None else self.do_image_splitting

        min_tiles = min_tiles if min_tiles is not None else self.min_tiles
        max_tiles = max_tiles if max_tiles is not None else self.max_tiles

        if not do_image_splitting:
            min_tiles = 1
            max_tiles = 1
            logger.debug(
                "Image splitting is disabled, setting min_tiles and max_tiles to 1. Set do_image_splitting=True to enable splitting."
            )

        if do_image_splitting and min_tiles > max_tiles:
            raise ValueError("min_tiles must be less than or equal to max_tiles")

        use_thumbnail = use_thumbnail if use_thumbnail is not None else self.use_thumbnail
        min_image_tokens = min_image_tokens if min_image_tokens is not None else self.min_image_tokens
        max_image_tokens = max_image_tokens if max_image_tokens is not None else self.max_image_tokens
        encoder_patch_size = encoder_patch_size if encoder_patch_size is not None else self.encoder_patch_size
        tile_size = tile_size if tile_size is not None else self.tile_size
        max_pixels_tolerance = max_pixels_tolerance if max_pixels_tolerance is not None else self.max_pixels_tolerance

        max_thumbnail_image_patches = max_image_tokens * downsample_factor**2
        tile_size_patches = (tile_size // encoder_patch_size) ** 2 if do_image_splitting else 0
        max_num_patches = max(
            max_thumbnail_image_patches,
            tile_size_patches,
        )

        self.image_processor.max_num_patches = max_num_patches

        if text is None and images is None:
            raise ValueError("You must provide one of `text` or `images`.")

        output_kwargs = self._merge_kwargs(
            Lfm2VlProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")
            n_images_in_text = sum([sample.count(self.image_token) for sample in text])
            if n_images_in_text > 0 and (images is None):
                raise ValueError(f"We detected {n_images_in_text} tokens in the text but no images were passed")

        inputs = {}

        if images is not None:
            images = make_nested_list_of_images(images)
            text, vision_inputs = self.process_vision(
                text,
                images,
                use_image_special_tokens,
                downsample_factor,
                min_tiles,
                max_tiles,
                use_thumbnail,
                min_image_tokens,
                max_image_tokens,
                encoder_patch_size,
                tile_size,
                max_pixels_tolerance,
                output_kwargs,
            )
            inputs.update(vision_inputs)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)

        if text is not None:
            text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
            self._check_special_mm_tokens(text, text_inputs, modalities=["image"])
            inputs.update(text_inputs)

        return BatchFeature(inputs, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LFM2Tokeniser's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        batched_decode_output = self.tokenizer.batch_decode(*args, **kwargs)
        return batched_decode_output

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LFM2Tokeniser's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        decode_output = self.tokenizer.decode(*args, **kwargs)
        return decode_output

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(image_processor_input_names + tokenizer_input_names))


__all__ = ["Lfm2VlProcessor"]
