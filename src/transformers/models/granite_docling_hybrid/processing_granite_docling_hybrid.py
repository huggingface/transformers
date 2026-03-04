# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""
Processor class for GraniteDoclingHybrid.
"""

from typing import TYPE_CHECKING, Optional, Union

from ...image_utils import ImageInput
from ...processing_utils import Unpack
from ...tokenization_utils_base import BatchEncoding, TextInput
from ..got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas
from ..idefics3.processing_idefics3 import (
    Idefics3Processor,
    Idefics3ProcessorKwargs,
    get_image_prompt_string,
    is_url,
    load_image,
)


if TYPE_CHECKING:
    from ...tokenization_utils_base import PreTokenizedInput


class GraniteDoclingHybridProcessor(Idefics3Processor):
    r"""
    Constructs a GraniteDoclingHybrid processor which wraps a tokenizer and GotOcr2 image processor into a single processor.

    [`GraniteDoclingHybridProcessor`] offers all the functionalities of [`GotOcr2ImageProcessor`]. See
    the docstring of [`~GraniteDoclingHybridProcessor.__call__`] and [`~GraniteDoclingHybridProcessor.decode`] for more information.

    Args:
        image_processor (`GotOcr2ImageProcessor`):
            An instance of [`GotOcr2ImageProcessor`]. The image processor is a required input.
        tokenizer (`PreTrainedTokenizerBase`, *optional*):
            An instance of [`PreTrainedTokenizerBase`]. This should correspond with the model's text model. The tokenizer is a required input.
        image_seq_len (`int`, *optional*, defaults to 169):
            The length of the image sequence i.e. the number of <image> tokens per image in the input.
            This parameter is used to build the string from the input prompt and image tokens and should match the
            value the model used. It is computed as: image_seq_len = int(((image_size // patch_size) ** 2) / (scale_factor**2))
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        fake_image_token (`str` or `AddedToken`, *optional*, defaults to "<fake_token_around_image>"):
            Token used to wrap expanded image sequences. Override to use a custom token.
        image_token (`str` or `AddedToken`, *optional*, defaults to "<image>"):
            Token in the text prompt indicating where image patches should be inserted.
        end_of_utterance_token (`str` or `AddedToken`, *optional*, defaults to "<end_of_utterance>"):
            Token inserted between user and assistant turns. Override to match a custom tokenizer vocabulary.
        global_image_tag (`str` or `AddedToken`, *optional*, defaults to "<global-img>"):
            Token corresponding to the global image representation.
        extra_special_tokens_pattern (`str`, *optional*):
            Regular expression pattern used to strip redundant special tokens when preparing chat prompts. By default
            this pattern is derived from the configured `global_image_tag`.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "GotOcr2ImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __call__(
        self,
        images: Union[ImageInput, list[ImageInput], list[list[ImageInput]]] = None,
        text: Union[TextInput, "PreTokenizedInput", list[TextInput], list["PreTokenizedInput"]] = None,
        audio=None,
        videos=None,
        image_seq_len: Optional[int] = None,
        **kwargs: Unpack[Idefics3ProcessorKwargs],
    ) -> BatchEncoding:
        """
        Processes the input prompts and returns a BatchEncoding.

        This method extends the Idefics3Processor to handle GotOcr2ImageProcessor specifics.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`, *optional*):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. If is of type `list[ImageInput]`, it's assumed that this is for a single prompt i.e. of batch size 1.
            text (`Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
                Wherever an image token, `<image>` is encountered it is expanded to
                `<fake_token_around_image>` + `<row_x_col_y>` + `<image>` * `image_seq_len` * <fake_token_around_image>`.
            image_seq_len (`int`, *optional*):
                The length of the image sequence. If not provided, the default value of self.image_seq_len is used.
                image_seq_len should be equal to int(((image_size // patch_size) ** 2) / (scale_factor**2))
            return_tensors (`Union[str, TensorType]`, *optional*):
                If set, will return tensors of a particular framework. See [`PreTrainedTokenizerFast.__call__`] for more
                information.
        """
        if text is None and images is None:
            raise ValueError("You must provide either `text` or `images`.")

        output_kwargs = self._merge_kwargs(
            Idefics3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # GotOcr2ImageProcessor doesn't use return_row_col_info
        output_kwargs["images_kwargs"].pop("return_row_col_info", None)

        image_seq_len = image_seq_len if image_seq_len is not None else self.image_seq_len
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)

        n_images_in_text = []
        n_images_in_images = []
        inputs = {}

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")
            n_images_in_text = [sample.count(self.image_token) for sample in text]

        if images is not None:
            from itertools import accumulate

            from ..idefics3.processing_idefics3 import is_image_or_image_url

            if is_image_or_image_url(images):
                images = [[images]]
            elif isinstance(images, (list, tuple)) and is_image_or_image_url(images[0]):
                if text is not None:
                    if sum(n_images_in_text) != len(images):
                        raise ValueError(
                            f"The total number of {self.image_token} tokens in the prompts should be the same as the number of images passed."
                            f" Found {sum(n_images_in_text)} {self.image_token} tokens and {len(images)} images."
                        )
                    # Reorganize the images to match the prompts
                    cumsum_images_in_text = [0] + list(accumulate(n_images_in_text))
                    images = [
                        images[cumsum_images_in_text[i] : cumsum_images_in_text[i + 1]]
                        for i in range(len(n_images_in_text))
                    ]
                else:
                    images = [images]
            elif (
                not isinstance(images, (list, tuple))
                and not isinstance(images[0], (list, tuple))
                and not is_image_or_image_url(images[0][0])
            ):
                raise ValueError(
                    "Invalid input images. Please provide a single image or a list of images or a list of list of images."
                )
            n_images_in_images = [len(sample) for sample in images]

            # Load images if they are URLs
            images = [[load_image(im) if is_url(im) else im for im in sample] for sample in images]

            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            inputs.update(image_inputs)

            if text is not None:
                if n_images_in_images != n_images_in_text:
                    raise ValueError(
                        f"The number of images in the text {n_images_in_text} and images {n_images_in_images} should be the same."
                    )

                # GotOcr2ImageProcessor doesn't return rows/cols, compute them
                image_rows = []
                image_cols = []
                for sample_images in images:
                    sample_image_rows = []
                    sample_image_cols = []
                    for img in sample_images:
                        width, height = img.size
                        n_cols, n_rows = get_optimal_tiled_canvas(
                            (height, width),
                            (
                                self.image_processor.size["height"],
                                self.image_processor.size["width"],
                            ),
                            self.image_processor.min_patches,
                            self.image_processor.max_patches,
                        )
                        sample_image_rows.append(n_rows)
                        sample_image_cols.append(n_cols)
                    image_rows.append(sample_image_rows)
                    image_cols.append(sample_image_cols)

                # Post-process inputs for GotOcr2ImageProcessor
                inputs.pop("num_patches", None)  # Not needed downstream
                pixel_values = inputs.get("pixel_values")
                if pixel_values is not None and len(pixel_values.shape) == 4:
                    # Make 5D to match Idefics3 expected format: (batch, num_images, num_channels, height, width)
                    inputs["pixel_values"] = pixel_values.unsqueeze(0)

                fake_image_token = self.fake_image_token
                image_token = self.image_token
                global_img_token = self.global_image_tag

                prompt_strings = []
                batch_image_seq_lengths = []
                for sample, sample_rows, sample_cols in zip(text, image_rows, image_cols):
                    # Replace the image token with fake tokens around the expanded image token sequence of length `image_seq_len`
                    image_prompt_strings = []
                    image_seq_lengths = []
                    for n_rows, n_cols in zip(sample_rows, sample_cols):
                        image_prompt_string = get_image_prompt_string(
                            n_rows,
                            n_cols,
                            image_seq_len,
                            image_token=image_token,
                            fake_token_around_image=fake_image_token,
                            global_img_token=global_img_token,
                        )
                        # Add +2 and +3 for special BOI/EOI/fake_image_wrapper tokens
                        row_length = (self.image_seq_len + 2) * n_cols + 1
                        image_seq_lengths.append((self.image_seq_len + 3) + row_length * n_rows)
                        image_prompt_strings.append(image_prompt_string)

                    batch_image_seq_lengths.append(image_seq_lengths)
                    split_sample = sample.split(image_token)
                    if len(split_sample) == 0:
                        raise ValueError("The image token should be present in the text.")

                    # Place in the image prompt strings where the image tokens are
                    sample = split_sample[0]
                    for i, image_prompt_string in enumerate(image_prompt_strings):
                        sample += image_prompt_string + split_sample[i + 1]
                    prompt_strings.append(sample)

                text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])
                self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])
                inputs.update(text_inputs)

        elif text is not None:
            if any(n_images_in_text):
                raise ValueError(
                    f"Found {sum(n_images_in_text)} {self.image_token} tokens in the text but no images were passed."
                )
            text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])
            inputs.update(text_inputs)

        if return_mm_token_type_ids:
            import numpy as np

            array_ids = np.array(inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(array_ids)
            for i, seq_lengths in enumerate(batch_image_seq_lengths):
                image_start_positions = np.where(array_ids[i] == self.fake_image_token_id)[0]
                j = 0
                for seq_len in seq_lengths:
                    if j >= len(image_start_positions):
                        break
                    start = image_start_positions[j]
                    end = start + seq_len
                    mm_token_type_ids[i, start:end] = 1
                    j = np.searchsorted(image_start_positions, end)

            inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        from ...feature_extraction_utils import BatchFeature

        return BatchFeature(data=inputs, tensor_type=return_tensors)


__all__ = ["GraniteDoclingHybridProcessor"]
