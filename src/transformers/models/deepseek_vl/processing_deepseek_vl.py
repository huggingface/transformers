import warnings
import torch
from typing import List, Optional, Union
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin, ProcessingKwargs, Unpack
from ...tokenization_utils_base import BatchEncoding, AddedToken
from .image_processing_deepseek_vl import DeekseekImageProcessor
from ..llama.tokenization_llama_fast import LlamaTokenizerFast
from ..llama.tokenization_llama import LlamaTokenizer
from ...image_utils import ImageInput, is_valid_image, make_flat_list_of_images
from ...tokenization_utils_base import (
    AddedToken,
    PreTokenizedInput,
    TextInput,
)
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
    _validate_images_text_input_order,
)

IMAGE_TOKEN = "<image_placeholder>"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language.\n\n"
)

def add_image_tokens_to_input_ids(
    input_ids: torch.LongTensor,
    image_indices: List[int],
    image_token_id: int,
    num_image_tokens: int,
    add_special_tokens: bool = False,
    ):
    """
    Args:
        image_indices (List[int]): [index_0, index_1, ..., index_j]
        input_ids (torch.LongTensor): [N]

    Returns:
        input_ids (torch.LongTensor): [N + image tokens]
        num_image_tokens (torch.IntTensor): [n_images]
    """
    input_slices = []
    start = 0
    for index in image_indices:
        if add_special_tokens:
            end =  index + 1
        else:
            end = index
        
        input_slices.append(input_ids[start:end])

        input_slices.append(
            image_token_id * torch.ones((num_image_tokens,), dtype=torch.long)
        )
        start = index + 1
    input_slices.append(input_ids[start:])

    input_ids = torch.cat(input_slices, dim=0)
    num_image_tokens = torch.IntTensor([num_image_tokens] * len(image_indices))

    return input_ids, num_image_tokens


class DeepseekVLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": False,
            "padding": False,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_token_type_ids": False,
            "return_length": False,
            "verbose": True,
        },
        "images_kwargs": {},
    }


class DeepseekVLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

    def __init__(
            self, 
            image_processor=None, 
            tokenizer=None, 
            chat_template=None,
            use_deafult_system_prompt=True, 
            **kwargs,
        ):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        
        self.num_image_tokens = 576
        self.use_default_system_prompt = use_deafult_system_prompt
        

        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )

    def __call__(
            self, 
            images: ImageInput = None,
            text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
            **kwargs: Unpack[DeepseekVLProcessorKwargs]
        ) -> BatchFeature:
        """
        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
            text (`str`, `List[str]`, `List[List[str]]`):
        """
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
            DeepseekVLProcessorKwargs,
            **kwargs,
        )
        if text is None and images is None:
            raise ValueError("You must provide either text or images.")
        
        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")
        
        # add and replace image tokens
        prompt_strings = []
        single_image_tokes = IMAGE_TOKEN * self.num_image_tokens
        for prompt in text:
            prompt = prompt.replace(IMAGE_TOKEN, single_image_tokes)
            if self.use_default_system_prompt:
                prompt = DEFAULT_SYSTEM_PROMPT + prompt
            prompt_strings.append(prompt)

        data = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])

        # process images if pixel_values are provided
        if images is not None:
            data["pixel_values"] = self.image_processor(images, **output_kwargs["images_kwargs"])["pixel_values"]
        
        return BatchFeature(data=data)


    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
    


__all__ = ["DeepseekVLProcessor"]

