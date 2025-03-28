import math
import os
from typing import Iterable, List, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import select_best_resolution
from ...image_utils import ImageInput, VideoInput, get_image_size, to_numpy_array
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging
from ..auto import AutoImageProcessor


logger = logging.get_logger(__name__)


class Ovis2ProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "image_kwargs": {},
    }


class Ovis2Processor(ProcessorMixin):
    r"""
    Constructs a LLaVa-Onevision processor which wraps LLaVa-NeXT image processor and a LLaMa tokenizer into a single processor.

    [`LlavaNextProcessor`] offers all the functionalities of [`Ovis2VideoProcessor`], [`Ovis2ImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~Ovis2VideoProcessor.__call__`], [`~LlavaNextProcessor.__call__`] and [`~LlavaNextProcessor.decode`] for more information.

    Args:
        image_processor ([`Ovis2ImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        image_seq_length (`int`, *optional*, defaults to `256`):
            The maximum sequence length for image tokens.
            Shoudl be same as in model's config
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "image_token",
        "image_seq_length",
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_token="<image>",
        image_seq_length=256,
        **kwargs,
    ):
        self.image_seq_length = image_seq_length
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        **kwargs: Unpack[Ovis2ProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        LlavaNextImageProcessor's [`~LlavaNextImageProcessor.__call__`] if `images` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **image_sizes** -- Size of each image that will be used to unpad an image. Returned when `images` is not `None`.
        """

        output_kwargs = self._merge_kwargs(
            Ovis2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        image_inputs = {}

        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            grids = iter(image_inputs["grids"])
            text = self._expand_image_tokens(text, grids, self.image_token)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        return BatchFeature(data={**text_inputs, **image_inputs})

    def _expand_image_tokens(
        self,
        text: List[TextInput],
        grids: Iterable[Union[List[int], int]],
        special_token: str,
    ):
        processed_text = []
        for sample in text:
            while "<image>" in sample:
                grid = next(grids)
                row, col = grid[0], grid[1]
                placeholder = f"<IMG_START>{'<IMG_ATOM>' * self.image_seq_length}<IMG_GRID>"
                if row * col > 1:
                    for r in range(row):
                        for c in range(col):
                            placeholder += f"{'<IMG_ATOM>' * self.image_seq_length}"
                            if c < col - 1:
                                placeholder += f"<IMG_COL>"
                        if r < row - 1:
                            placeholder += f"<IMG_ROW>" 
                placeholder += f"<IMG_END>"

                sample = sample.replace("<image>", placeholder, 1)
            processed_text.append(sample)
        return processed_text

    # Adapted from transformers.models.llava_next.processing_llava_next.LlavaNextProcessor._get_unpadded_features
    def _get_unpadded_features(self, height, width, patches_height, patches_width, scale_height, scale_width):
        """
        Get number of features for a given image with height/width. LLaVA-NeXT is different from LLaVA
        because it divided each image into patches depending on its resolution. Therefore we need to calculate how many
        patches an image is divided into and get the number of features from that.
        """
        current_height = patches_height * scale_height
        current_width = patches_width * scale_width

        original_aspect_ratio = width / height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            new_height = int(round(height * (current_width / width), 7))
            padding = (current_height - new_height) // 2
            current_height -= padding * 2
        else:
            new_width = int(round(width * (current_height / height), 7))
            padding = (current_width - new_width) // 2
            current_width -= padding * 2

        unpadded_features = current_height * current_width
        newline_features = current_height

        ratio = math.sqrt(current_height * current_width / (9 * patches_height**2))
        if ratio > 1.1:
            unpadded_features = int(current_height // ratio) * int(current_width // ratio)
            newline_features = int(current_height // ratio)

        return (unpadded_features, newline_features)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))



__all__ = ["Ovis2Processor"]