# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
Image/Text processor class for OneFormer
"""

from typing import List, Optional
import torch

from ...processing_utils import ProcessorMixin


def pad_tokens_to_max_len(tokens, max_len=77):
    if isinstance(tokens, list):
        tmp_tokens = []
        for token in tokens:
            tmp_tokens.append(token["input_ids"])
        tokens = tmp_tokens
        del tmp_tokens
    else:
        tokens = tokens["input_ids"]

    padded_tokens = torch.zeros(len(tokens), max_len, dtype=torch.long)
    for i in range(len(tokens)):
        token = tokens[i]
        padded_tokens[i][: len(token)] = torch.tensor(token).long()
    return padded_tokens


class OneFormerProcessor(ProcessorMixin):
    r"""
    Constructs an OneFormer processor which wraps [`OneFormerImageProcessor`] and [`CLIPTokenizer`]/[`CLIPTokenizerFast`]
    into a single processor that interits both the image processor and tokenizer functionalities. See the
    [`~OneFormerProcessor.__call__`] and [`~OneFormerProcessor.decode`] for more information.

    Args:
        image_processor ([`OneFormerImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`CLIPTokenizer`, `CLIPTokenizerFast`]):
            The tokenizer is a required input.
        max_seq_len (`int`, *optional*, defaults to 77)):
            Sequence length for input text list.
        task_seq_len (`int`, *optional*, defaults to 77):
            Sequence length for input task token.
    """
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "OneFormerImageProcessor"
    tokenizer_class = ("CLIPTokenizer")

    def __init__(
        self, 
        image_processor=None, 
        tokenizer=None, 
        max_seq_length: Optional[int] = 77, 
        task_seq_length: int = 77, 
        **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        
        self.max_seq_length = max_seq_length
        self.task_seq_length = task_seq_length

        super().__init__(image_processor, tokenizer)

    def __call__(self, images=None, task_inputs=None, segmentation_maps=None, **kwargs):
        """
        Main method to prepare for the model one or several task input(s) and image(s). This method forwards the `task_inputs` and
        `kwargs` arguments to CLIPTokenizer's [`~CLIPTokenizer.__call__`] if `task_inputs` is not `None` to encode. 
        To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        OneFormerImageProcessor's [`~OneFormerImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            task_inputs (`str`, `List[str]`):
                The sequence or batch of task_inputs sequences to be encoded. Each sequence can be a string or a list of strings
                of the template "the task is {task}".
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`,
            `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            segmentation_maps (`ImageInput`, *optional*):
                The corresponding semantic segmentation maps with the pixel-wise annotations.

             (`bool`, *optional*, defaults to `True`):
                Whether or not to pad images up to the largest image in a batch and create a pixel mask.

                If left to the default, will return a pixel mask that is:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:
            - **task_inputs** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """

        if task_inputs is None:
            raise ValueError(
                "You have to specify the task_input. Found None."
            )
        elif images is None:
            raise ValueError(
                "You have to specify the image. Found None."
            )

        encoded_inputs = self.image_processor(images, task_inputs, segmentation_maps, **kwargs)

        if isinstance(task_inputs, str):
            task_inputs = [task_inputs]

        if isinstance(task_inputs, List) and isinstance(task_inputs[0], str):
            task_token_inputs = []
            for task in task_inputs:
                task_input = [f"the task is {task}"]
                task_token = self.tokenizer(task_input)
                task_token = pad_tokens_to_max_len(task_token, max_len=self.task_seq_length)
                task_token_inputs.append(task_token)

            encoded_inputs["task_inputs"] = torch.cat(task_token_inputs, dim=0)
        else:
            raise TypeError("Task Inputs should be a string or a list of strings.")
        
        if hasattr(encoded_inputs, "texts_list"):
            texts_list = encoded_inputs.texts_list
        
            text_inputs = []
            for texts in texts_list:
                text_input_list = [self.tokenizer(texts[i]) for i in range(len(texts))]
                text_input_list = pad_tokens_to_max_len(text_input_list, max_len=self.max_seq_length)
                text_inputs.append(text_input_list.unsqueeze(0))

            encoded_inputs["text_inputs"] = torch.cat(text_inputs, dim=0)

        return encoded_inputs
    
    def encode_inputs(self, images=None, task_inputs=None, segmentation_maps=None, **kwargs):
        """
        This method forwards all its arguments to [`OneFormerImageProcessor.encode_inputs`] and then tokenizes the task_inputs.
        Please refer to the docstring of this method for more information.
        """
        encoded_inputs = self.image_processor.encode_inputs(images, task_inputs, segmentation_maps, **kwargs)

        if isinstance(task_inputs, str):
            task_inputs = [task_inputs]

        if isinstance(task_inputs, List) and isinstance(task_inputs[0], str):
            task_token_inputs = []
            for task in task_inputs:
                task_input = [f"the task is {task}"]
                task_token = self.tokenizer(task_input)
                task_token = pad_tokens_to_max_len(task_token, max_len=self.task_seq_length)
                task_token_inputs.append(task_token)

            encoded_inputs["task_inputs"] = torch.cat(task_token_inputs, dim=0)
        else:
            raise TypeError("Task Inputs should be a string or a list of strings.")
        
        if hasattr(encoded_inputs, "texts_list"):
            texts_list = encoded_inputs.texts_list
        
            text_inputs = []
            for texts in texts_list:
                text_input_list = [self.tokenizer(texts[i]) for i in range(len(texts))]
                text_input_list = pad_tokens_to_max_len(text_input_list, max_len=self.max_seq_length)
                text_inputs.append(text_input_list.unsqueeze(0))

            encoded_inputs["text_inputs"] = torch.cat(text_inputs, dim=0)

        return encoded_inputs

    def post_process_semantic_segmentation(self, *args, **kwargs):
        """
        This method forwards all its arguments to [`OneFormerImageProcessor.post_process_semantic_segmentation`].
        Please refer to the docstring of this method for more information.
        """
        return self.image_processor.post_process_semantic_segmentation(*args, **kwargs)
    
    def post_process_instance_segmentation(self, *args, **kwargs):
        """
        This method forwards all its arguments to [`OneFormerImageProcessor.post_process_instance_segmentation`].
        Please refer to the docstring of this method for more information.
        """
        return self.image_processor.post_process_instance_segmentation(*args, **kwargs)
    
    def post_process_panoptic_segmentation(self, *args, **kwargs):
        """
        This method forwards all its arguments to [`OneFormerImageProcessor.post_process_panoptic_segmentation`].
        Please refer to the docstring of this method for more information.
        """
        return self.image_processor.post_process_panoptic_segmentation(*args, **kwargs)