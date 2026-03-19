# Copyright 2022 SHI Labs and The HuggingFace Inc. team.
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

from ...processing_utils import ProcessorMixin
from ...utils import auto_docstring, is_torch_available


if is_torch_available():
    import torch


@auto_docstring
class OneFormerProcessor(ProcessorMixin):
    def __init__(
        self, image_processor=None, tokenizer=None, max_seq_length: int = 77, task_seq_length: int = 77, **kwargs
    ):
        r"""
        max_seq_length (`int`, *optional*, defaults to `77`):
            Maximum sequence length for encoding class names and text inputs. This parameter controls the
            maximum number of tokens used when tokenizing class names and other text inputs for the model.
        task_seq_length (`int`, *optional*, defaults to `77`):
            Maximum sequence length specifically for encoding task descriptions. Task descriptions (e.g.,
            "the task is semantic") are tokenized with this length limit, which may differ from the general
            text sequence length.
        """
        self.max_seq_length = max_seq_length
        self.task_seq_length = task_seq_length

        super().__init__(image_processor, tokenizer)

    def _preprocess_text(self, text_list=None, max_length=77):
        if text_list is None:
            raise ValueError("tokens cannot be None.")

        tokens = self.tokenizer(text_list, padding="max_length", max_length=max_length, truncation=True)

        attention_masks, input_ids = tokens["attention_mask"], tokens["input_ids"]

        token_inputs = []
        for attn_mask, input_id in zip(attention_masks, input_ids):
            token = torch.tensor(attn_mask) * torch.tensor(input_id)
            token_inputs.append(token.unsqueeze(0))

        token_inputs = torch.cat(token_inputs, dim=0)
        return token_inputs

    @auto_docstring
    def __call__(self, images=None, task_inputs=None, segmentation_maps=None, **kwargs):
        r"""
        task_inputs (`str` or `list[str]`, *required*):
            The task type(s) for segmentation. Must be one or more of `"semantic"`, `"instance"`, or `"panoptic"`.
            Can be a single string for a single image, or a list of strings (one per image) for batch processing.
            The task type determines which type of segmentation the model will perform on the input images.
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
            raise ValueError("You have to specify the task_input. Found None.")
        elif images is None:
            raise ValueError("You have to specify the image. Found None.")

        if not all(task in ["semantic", "instance", "panoptic"] for task in task_inputs):
            raise ValueError("task_inputs must be semantic, instance, or panoptic.")

        encoded_inputs = self.image_processor(images, task_inputs, segmentation_maps, **kwargs)

        if isinstance(task_inputs, str):
            task_inputs = [task_inputs]

        if isinstance(task_inputs, list) and all(isinstance(task_input, str) for task_input in task_inputs):
            task_token_inputs = []
            for task in task_inputs:
                task_input = f"the task is {task}"
                task_token_inputs.append(task_input)
            encoded_inputs["task_inputs"] = self._preprocess_text(task_token_inputs, max_length=self.task_seq_length)
        else:
            raise TypeError("Task Inputs should be a string or a list of strings.")

        if hasattr(encoded_inputs, "text_inputs"):
            texts_list = encoded_inputs.text_inputs

            text_inputs = []
            for texts in texts_list:
                text_input_list = self._preprocess_text(texts, max_length=self.max_seq_length)
                text_inputs.append(text_input_list.unsqueeze(0))

            encoded_inputs["text_inputs"] = torch.cat(text_inputs, dim=0)

        return encoded_inputs

    def encode_inputs(self, images=None, task_inputs=None, segmentation_maps=None, **kwargs):
        """
        This method forwards all its arguments to [`OneFormerImageProcessor.encode_inputs`] and then tokenizes the
        task_inputs. Please refer to the docstring of this method for more information.
        """

        if task_inputs is None:
            raise ValueError("You have to specify the task_input. Found None.")
        elif images is None:
            raise ValueError("You have to specify the image. Found None.")

        if not all(task in ["semantic", "instance", "panoptic"] for task in task_inputs):
            raise ValueError("task_inputs must be semantic, instance, or panoptic.")

        encoded_inputs = self.image_processor.encode_inputs(images, task_inputs, segmentation_maps, **kwargs)

        if isinstance(task_inputs, str):
            task_inputs = [task_inputs]

        if isinstance(task_inputs, list) and all(isinstance(task_input, str) for task_input in task_inputs):
            task_token_inputs = []
            for task in task_inputs:
                task_input = f"the task is {task}"
                task_token_inputs.append(task_input)
            encoded_inputs["task_inputs"] = self._preprocess_text(task_token_inputs, max_length=self.task_seq_length)
        else:
            raise TypeError("Task Inputs should be a string or a list of strings.")

        if hasattr(encoded_inputs, "text_inputs"):
            texts_list = encoded_inputs.text_inputs

            text_inputs = []
            for texts in texts_list:
                text_input_list = self._preprocess_text(texts, max_length=self.max_seq_length)
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


__all__ = ["OneFormerProcessor"]
