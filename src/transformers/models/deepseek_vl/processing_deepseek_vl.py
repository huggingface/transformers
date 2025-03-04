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
            **kwargs,
        ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.num_image_tokens = 576

        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        
        image_id = self.tokenizer.vocab.get(IMAGE_TOKEN)
        if image_id is None:
            image_token = AddedToken(IMAGE_TOKEN, normalized=False, special=True)
            tokens_to_add = {"additional_special_tokens": [image_token]}
            tokenizer.add_special_tokens(tokens_to_add)
            self.image_token_id = self.tokenizer.vocab.get(IMAGE_TOKEN)
        else:  
            self.image_token_id = image_id


        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
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
        add_special_tokens = output_kwargs["text_kwargs"]["add_special_tokens"]
        # tokenize text
        if text is not None:
            input_ids = self.tokenizer.encode(text)
            input_ids = torch.LongTensor(input_ids)
        
        # add image tokens to the input_ids
        image_token_mask: torch.BoolTensor = input_ids == self.image_token_id
        image_indices = image_token_mask.nonzero()
        input_ids, num_image_tokens = add_image_tokens_to_input_ids(
            input_ids = input_ids, 
            image_indices = image_indices,
            image_token_id = self.image_token_id,
            num_image_tokens = self.num_image_tokens,
            add_special_tokens = add_special_tokens
        )
        
        pixel_values = self.image_processor(images, **output_kwargs["images_kwargs"])["pixel_values"]
        return_data = {
            input_ids: input_ids,
            pixel_values: pixel_values,
            num_image_tokens: num_image_tokens,
        }
        return BatchFeature(data=return_data)



    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names with CLIP->PaliGemma
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["DeepseekVLProcessor"]

