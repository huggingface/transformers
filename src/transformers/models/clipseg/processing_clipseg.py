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
Image/Text processor class for CLIPSeg
"""

import sys
import warnings
from typing import List, Optional, Union

from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput


if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack


class CLIPSegProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


class CLIPSegProcessor(ProcessorMixin):
    r"""
    Constructs a CLIPSeg processor which wraps a CLIPSeg image processor and a CLIP tokenizer into a single processor.

    [`CLIPSegProcessor`] offers all the functionalities of [`ViTImageProcessor`] and [`CLIPTokenizerFast`]. See the
    [`~CLIPSegProcessor.__call__`] and [`~CLIPSegProcessor.decode`] for more information.

    Args:
        image_processor ([`ViTImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`CLIPTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "ViTImageProcessor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")
    # For backward compatibility. See transformers.processing_utils.ProcessorMixin.prepare_and_validate_optional_call_args for more details.
    optional_call_args = ["visual_prompt"]

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        images: Optional[ImageInput] = None,
        # The following is to capture `visual_prompt` argument that may be passed as a positional argument.
        # See transformers.processing_utils.ProcessorMixin.prepare_and_validate_optional_call_args for more details.
        # This behavior is only needed for backward compatibility and will be removed in future versions.
        *args,
        audio=None,
        videos=None,
        **kwargs: Unpack[CLIPSegProcessorKwargs],
    ):
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to CLIPTokenizerFast's [`~CLIPTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        ViTImageProcessor's [`~ViTImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring of
        the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            visual_prompt (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
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

        output_kwargs = self._merge_kwargs(
            CLIPSegProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
            **self.prepare_and_validate_optional_call_args(*args),
        )

        visual_prompt = output_kwargs["images_kwargs"].pop("visual_prompt", None)

        if text is None and visual_prompt is None and images is None:
            raise ValueError("You have to specify either text, visual prompt or images.")

        if text is not None and visual_prompt is not None:
            raise ValueError("You have to specify exactly one type of prompt. Either text or visual prompt.")

        if text is not None:
            encoding = self.tokenizer(text, **output_kwargs["text_kwargs"])

        if visual_prompt is not None:
            prompt_features = self.image_processor(visual_prompt, **output_kwargs["images_kwargs"])

        if images is not None:
            image_features = self.image_processor(images, **output_kwargs["images_kwargs"])

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
            return BatchEncoding(
                data=dict(**image_features), tensor_type=output_kwargs["common_kwargs"].get("return_tensors")
            )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor
