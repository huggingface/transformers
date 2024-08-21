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
Processor class for LLaVa-NeXT.
"""

from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import select_best_resolution
from ...image_utils import ImageInput, get_image_size, to_numpy_array
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class LlavaNextProcessor(ProcessorMixin):
    r"""
    Constructs a LLaVa-NeXT processor which wraps a LLaVa-NeXT image processor and a LLaMa tokenizer into a single processor.

    [`LlavaNextProcessor`] offers all the functionalities of [`LlavaNextImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~LlavaNextProcessor.__call__`] and [`~LlavaNextProcessor.decode`] for more information.

    Args:
        image_processor ([`LlavaNextImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        patch_size (`int`, *optional*):
            Patch size from the vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Shoudl be same as in model's config
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "patch_size", "vision_feature_select_strategy", "image_token"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<image>",  # set the default and let users change if they have peculiar special tokens in rare cases
        **kwargs,
    ):
        self.patch_size = patch_size
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = image_token
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        do_pad: Optional[bool] = True,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        LlavaNextImageProcessor's [`~LlavaNextImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
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
            do_pad (`bool`, *optional*, defaults to self.do_pad):
                Whether to pad the image. If `True` will pad the images in the batch to the largest image in the batch
                and create a pixel mask. Padding will be applied to the bottom and right of the image with zeros.
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
        if images is not None:
            image_inputs = self.image_processor(images, do_pad=do_pad, return_tensors=return_tensors)
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        if self.patch_size is None or self.vision_feature_select_strategy is None:
            prompt_strings = text
            logger.warning_once(
                "Expanding inputs for image tokens in LLaVa-NeXT should be done in processing. "
                "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
            )
        # cannot infer image expansion length if no images are found
        elif not image_inputs:
            prompt_strings = text
        else:
            image_sizes = image_inputs["image_sizes"]
            height, width = get_image_size(to_numpy_array(image_inputs["pixel_values"][0][0]))
            prompt_strings = []
            for image_size, sample in zip(image_sizes, text):
                # Replace the image token with the expanded image token sequence
                orig_height, orig_width = image_size
                num_image_tokens = self._get_number_of_features(orig_height, orig_width, height, width)
                if self.vision_feature_select_strategy == "default":
                    num_image_tokens -= 1

                sample = sample.replace(self.image_token, self.image_token * num_image_tokens)
                prompt_strings.append(sample)

        text_inputs = self.tokenizer(
            prompt_strings,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )

        return BatchFeature(data={**text_inputs, **image_inputs})

    def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints

        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints
        )
        scale_height, scale_width = height_best_resolution // height, width_best_resolution // width

        patches_height = height // self.patch_size
        patches_width = width // self.patch_size
        unpadded_features, newline_features = self._get_unpadded_features(
            orig_height, orig_width, patches_height, patches_width, scale_height, scale_width
        )
        # The base patch covers the entire image (+1 for the CLS)
        base_features = patches_height * patches_width + 1
        num_image_tokens = unpadded_features + newline_features + base_features
        return num_image_tokens

    def _get_unpadded_features(self, height, width, patches_height, patches_width, scale_height, scale_width):
        """
        Get number of features for a given image with height/width. LLaVA-NeXT is different from LLaVA
        because it divided each image into patches depending on its resolution. Therefore we need to calculate how many
        patches an image is divided into and get the number of features from that.
        """
        current_width = patches_height * scale_height
        current_height = patches_width * scale_width

        original_aspect_ratio = width / height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            new_height = (height * current_width) // width
            padding = (current_height - new_height) // 2
            current_height -= padding * 2
        else:
            new_width = (width * current_height) // height
            padding = (current_width - new_width) // 2
            current_width -= padding * 2

        unpadded_features = current_height * current_width
        newline_features = current_height
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
