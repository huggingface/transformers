from typing import List, Optional, Union

from transformers.utils import logging

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PreTokenizedInput, TextInput


logger = logging.get_logger(__name__)


class InternVL2_5Processor(ProcessorMixin):
    r"""
    Constructs a InternVL processor which wraps a InternVL image processor and a tokenizer into a single processor.

    Args:
        image_processor (`InternVLImageProcessor`):
            The image processor is a required input.
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): 
            A Jinja template which will be used to format chat conversations.
        image_token (`str`, *optional*, defaults to "<image>"):
            Special token used to denote image placeholders in text.
    """
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "InternVL2_5ImageProcessor"
    tokenizer_class = ("AutoTokenizer")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.image_placeholder = "<image>"
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else "<IMG_CONTEXT>"
        self.num_image_token = image_processor.num_image_token

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        padding: bool = False,
        **kwargs
    ) -> BatchFeature:

        if text is None:
            raise ValueError("Input text cannot be None")

        if not isinstance(text, list):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        total_placeholders = sum(text_item.count(self.image_placeholder) for text_item in text)

        encoding = {}
        if images is not None:
            image_inputs = self.image_processor(images, **kwargs)
            encoding.update(image_inputs)

            # Check if number of images matches placeholders
            if len(image_inputs['num_patches']) != total_placeholders:
                raise ValueError(
                    f"Number of image placeholders ({total_placeholders}) does not match "
                    f"number of processed images ({len(image_inputs['num_patches'])})"
                )

            index = 0
            for i in range(len(text)):
                while self.image_placeholder in text[i]:
                    text[i] = text[i].replace(
                        self.image_placeholder,
                        f"<img>{('<|place_holder|>' * image_inputs['num_patches'][index] * self.num_image_token)}</img>",
                        1
                    )
                    index += 1
                text[i] = text[i].replace("<|place_holder|>", self.image_token)

        text_features = self.tokenizer(
            text,
            add_special_tokens=True,
            padding=padding,
            **kwargs
        )
        encoding.update(text_features)

        return BatchFeature(data=encoding)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = getattr(self.tokenizer, "model_input_names", [])
        image_processor_input_names = getattr(self.image_processor, "model_input_names", [])
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["InternVL2_5Processor"]
