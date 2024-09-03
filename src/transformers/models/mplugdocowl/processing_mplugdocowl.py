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

from typing import Dict, List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType


class MPLUGDocOwlProcessor(ProcessorMixin):
    r"""
    Constructs a MPLUGDocOwl processor which wraps a MPLUGDocOwl image processor and a MPLUGDocOwl tokenizer into a single processor.

    [`MPLUGDocOwlProcessor`] offers all the functionalities of [`MPLUGDocOwlImageProcessor`] and [`AutoTokenizer`]. See the
    [`~MPLUGDocOwlProcessor.__call__`] and [`~MPLUGDocOwlProcessor.decode`] for more information.

    Args:
        image_processor ([`MPLUGDocOwlImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`AutoTokenizer`], *optional*):
            The tokenizer is a required input.
        num_image_tokens (`int`, *optional*, defaults to 257):
            The sequence length of image embeddings after the HReducer module.
        image_token (`str`, *optional*, defaults to "<image>"):
            The string form of the token corresponding to the special `image` token used as a placeholder.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "num_image_tokens", "image_token"]
    image_processor_class = "MPLUGDocOwlImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        num_image_tokens=257,
        image_token="<image>",
        **kwargs,
    ):
        self.num_image_tokens = num_image_tokens
        self.image_token = image_token
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def generate_text_with_placeholders(
        self, text, patch_positions, anchor_max, num_patches, add_textual_crop_indicator
    ):
        """
        Generates a text string with placeholders for images and optional textual crop indicators.

        Parameters:
            - text (str): The input text containing <image> tokens where image placeholders should be inserted.
            - patch_positions (numpy.ndarray): Array of patch positions indicating the location of cropped images.
            - anchor_max (int): The maximum anchor value used to identify global images.
            - num_patches (int): The number of patches (or cropped images) to be represented in the text.
            - add_textual_crop_indicator (bool): Flag indicating whether to add textual crop indicators in the output.

        Returns:
            - str: The generated text with appropriate image placeholders and optional crop indicators.
        """
        media_token = "<image>"
        if media_token not in text:
            raise ValueError("The prompt must contain the media token '<image>'")
        text_list = text.split(media_token)
        text = "USER: "
        image_token_count = 0

        for next_text in text_list[1:]:
            if add_textual_crop_indicator:
                # Generate image placeholders with interleaved textual crop indicator
                for patch_pos in patch_positions.tolist():
                    if patch_pos[0] == anchor_max and patch_pos[1] == anchor_max:
                        text += "<global_img><image>"
                    else:
                        row_col = f"row{patch_pos[0]}_col{patch_pos[1]}"
                        text += f"<crop_img_{row_col}><image>"
            else:
                # Generate successive image placeholders for an image, 1 crop img == 1
                text += "<image>" * num_patches

            text += next_text
            image_token_count += 1

        text += " ASSISTANT:"
        return text

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        add_textual_crop_indicator: bool = True,
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        do_rescale: bool = True,
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = (0.48145466, 0.4578275, 0.40821073),
        image_std: Optional[Union[float, List[float]]] = (0.26862954, 0.26130258, 0.27577711),
        size: Dict[str, int] = {"width": 448, "height": 448},
        do_anchor_resize: bool = True,
        do_shape_adaptive_cropping: bool = True,
        do_add_global_image: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to AutoTokenizer's [`~AutoTokenizer.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        MPLUGDocOwlImageProcessor's [`~MPLUGDocOwlImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            images (ImageInput, optional):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array, or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]], optional):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string).
            add_textual_crop_indicator (bool, optional):
                Whether to add a textual crop indicator to the images. Defaults to True.
            padding (Union[bool, str, PaddingStrategy], optional):
                Select a strategy to pad the returned sequences. Defaults to True.
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence is provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                lengths).
            truncation (Union[bool, str, TruncationStrategy], optional):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            max_length (int, optional):
                Maximum length of the returned list and optionally padding length.
            do_rescale (bool, optional):
                Whether to rescale the image. Defaults to True.
            do_convert_rgb (bool, optional):
                Whether to convert the image to RGB. Defaults to True.
            do_resize (bool, optional):
                Whether to resize the image. Defaults to True.
            do_normalize (bool, optional):
                Whether to normalize the image. Defaults to None.
            image_mean (Optional[Union[float, List[float]]], optional):
                The mean values for image normalization. Defaults to (0.48145466, 0.4578275, 0.40821073).
            image_std (Optional[Union[float, List[float]]], optional):
                The standard deviation values for image normalization. Defaults to (0.26862954, 0.26130258, 0.27577711).
            size (Dict[str, int], optional):
                A dictionary specifying the desired width and height for resizing. Defaults to {"width": 448, "height": 448}.
            do_anchor_resize (bool, optional):
                Whether to resize the image based on the specified anchor. Defaults to True.
            do_shape_adaptive_cropping (bool, optional):
                Whether to do a shape adaptive cropping of the input image. Should be only called if the `do_anchor_resize` is True. Defaults to True.
            do_add_global_image (bool, optional):
                Whether to add the global image to the image input. Defaults to True.
            return_tensors (Optional[Union[str, TensorType]], optional):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects. Defaults to TensorType.PYTORCH.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """

        if images is not None:
            pixel_values = self.image_processor(
                images,
                do_rescale=do_rescale,
                do_convert_rgb=do_convert_rgb,
                do_shape_adaptive_cropping=do_shape_adaptive_cropping,
                do_resize=do_resize,
                do_normalize=do_normalize,
                return_tensors=return_tensors,
                image_mean=image_mean,
                image_std=image_std,
                size=size,
                do_anchor_resize=do_anchor_resize,
                do_add_global_image=do_add_global_image,
            )
        else:
            pixel_values = None
        # text preprocessing
        patch_positions = pixel_values["patch_positions"]
        num_patches = pixel_values["num_patches"]
        anchor_max = pixel_values["anchor_max"]

        if not isinstance(text, list):
            text = [text]

        texts = [
            self.generate_text_with_placeholders(txt, patch_pos, anch_max, n_patches, add_textual_crop_indicator)
            for txt, patch_pos, anch_max, n_patches in zip(text, patch_positions, anchor_max, num_patches)
        ]

        prompt_strings = []
        for sample in texts:
            sample = sample.replace(self.image_token, self.image_token * self.num_image_tokens)
            prompt_strings.append(sample)

        text_inputs = self.tokenizer(
            prompt_strings,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )

        return BatchFeature(
            data={**text_inputs, "pixel_values": pixel_values["pixel_values"], "patch_positions": patch_positions}
        )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to AutoTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to AutoTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
