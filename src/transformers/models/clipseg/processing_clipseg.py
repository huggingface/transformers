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
Image/Text processor class for CLIPSeg
"""

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils import auto_docstring


@auto_docstring
class CLIPSegProcessor(ProcessorMixin):
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer)

    @auto_docstring
    def __call__(self, text=None, images=None, visual_prompt=None, return_tensors=None, **kwargs):
        r"""
        visual_prompt (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
            The visual prompt image or batch of images to be prepared. Each visual prompt image can be a PIL image,
            NumPy array or PyTorch tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape
            (C, H, W), where C is a number of channels, H and W are image height and width.

        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if text is None and visual_prompt is None and images is None:
            raise ValueError("You have to specify either text, visual prompt or images.")

        if text is not None and visual_prompt is not None:
            raise ValueError("You have to specify exactly one type of prompt. Either text or visual prompt.")

        output_kwargs = self._merge_kwargs(
            self.valid_processor_kwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )

        if text is not None:
            encoding = self.tokenizer(text, return_tensors=return_tensors, **output_kwargs["text_kwargs"])

        if visual_prompt is not None:
            prompt_features = self.image_processor(
                visual_prompt, return_tensors=return_tensors, **output_kwargs["images_kwargs"]
            )

        if images is not None:
            image_features = self.image_processor(
                images, return_tensors=return_tensors, **output_kwargs["images_kwargs"]
            )

        if visual_prompt is not None and images is not None:
            encoding = {
                "pixel_values": image_features.pixel_values,
                "conditional_pixel_values": prompt_features.pixel_values,
            }
            return encoding
        elif text is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values
            return encoding
        elif text is not None:
            return encoding
        elif visual_prompt is not None:
            encoding = {
                "conditional_pixel_values": prompt_features.pixel_values,
            }
            return encoding
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)


__all__ = ["CLIPSegProcessor"]
