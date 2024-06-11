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
Processor class for MPLUGDocOwl.
"""


from typing import List, Optional, Union, Tuple
#FIXME change the import from transformers to import from ...
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType
from .constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
#FIXME need to add image processing class name
#from transformers.models.mplugdocowl.image_processing_mplugdocowl import MPLUGDocOwlImageProcessor
import numpy as np
import torch

'''
def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    #breakpoint()
    prompt_chunks = [tokenizer(chunk).input_ids if len(chunk) > 0 else [] for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]
    print(prompt_chunks)
    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    #breakpoint()
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])
    #breakpoint()
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids
'''
class MPLUGDocOwlProcessor(ProcessorMixin):
    r"""
    Constructs a MPLUGDocOwl processor which wraps a MPLUGDocOwl image processor and a MPLUGDocOwl tokenizer into a single processor.

    [`MPLUGDocOwlProcessor`] offers all the functionalities of [`MPLUGDocOwlImageProcessor`] and [`MPLUGDocOwlTokenizerFast`]. See the
    [`~MPLUGDocOwlProcessor.__call__`] and [`~MPLUGDocOwlProcessor.decode`] for more information.

    Args:
        image_processor ([`MPLUGDocOwlImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`MPLUGDocOwlTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "MPLUGDocOwlImageProcessor"
    tokenizer_class = ("AutoTokenizer")#, "AutoTokenizerFast")
    
    def __init__(self, image_processor=None, tokenizer=None):
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        add_textual_crop_indicator: bool = True,
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        do_rescale: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to MPLUGDocOwlTokenizerFast's [`~MPLUGDocOwlTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        MPLUGDocOwlImageProcessor's [`~MPLUGDocOwlImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        #FIXME need to add image processing class name properly
        
        if images is not None:
            pixel_values = self.image_processor(images, do_rescale=do_rescale, do_convert_rgb=True, do_shape_adaptive_cropping=True, do_resize=True, do_normalize=True, return_tensors=return_tensors,image_mean=(0.48145466, 0.4578275, 0.40821073), image_std=(0.26862954, 0.26130258, 0.27577711),size={'width':448, 'height':448}, do_anchor_resize=True)
        else:
            pixel_values = None
        #text prpeocessing
        media_token = '<|image|>'
        assert media_token in text
        patch_positions = pixel_values['patch_positions']
        num_patches = pixel_values['num_patches']
        anchor_max = pixel_values['anchor_max']
        text_list = text.split(media_token)
        text = text_list[0]
        image_token_ptr = 0
        for next_text in text_list[1:]:
            if add_textual_crop_indicator:
                # generate image placeholders with interleaved texutual crop indicator
                # e.g. <global_img><|image|><crop_img_row0_col0><|image|><crop_img_row0_col1><|image|>...
                for patch_pos in patch_positions.tolist():
                    # global non-crop image
                    if patch_pos[0] == anchor_max and patch_pos[1] == anchor_max:
                        text += '<global_img><|image|>'
                    else:
                        row_col = 'row'+str(patch_pos[0])+'_col'+str(patch_pos[1])
                        text += '<crop_img_'+row_col+'><|image|>'
            else: 
                # generate successive image placeholders for a image, 1 crop img == 1 <|image|>
                text += '<|image|>'*num_patches
            text += next_text
            image_token_ptr += 1
        print(text)
        #breakpoint()
        #input_ids = tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors=return_tensors).unsqueeze(0)
        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )
        print(text_inputs)

        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values['pixel_values'], "patch_positions": patch_positions})
        #return BatchFeature(data={"input_ids": input_ids, "attention_mask": text_inputs.attention_mask, "pixel_values": pixel_values['pixel_values'], "patch_positions": patch_positions})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to MPLUGDocOwlTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to MPLUGDocOwlTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

#test the code
