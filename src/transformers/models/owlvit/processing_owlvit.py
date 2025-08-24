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
Image/Text processor class for OWL-ViT
"""

import warnings
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
    _validate_images_text_input_order,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TensorType, is_flax_available, is_tf_available, is_torch_available


if TYPE_CHECKING:
    from .modeling_owlvit import OwlViTImageGuidedObjectDetectionOutput, OwlViTObjectDetectionOutput


class OwlViTImagesKwargs(ImagesKwargs, total=False):
    query_images: Optional[ImageInput]


class OwlViTProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: OwlViTImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": "max_length",
        },
        "images_kwargs": {},
        "common_kwargs": {
            "return_tensors": "np",
        },
    }


class OwlViTProcessor(ProcessorMixin):
    r"""
    Constructs an OWL-ViT processor which wraps [`OwlViTImageProcessor`] and [`CLIPTokenizer`]/[`CLIPTokenizerFast`]
    into a single processor that interits both the image processor and tokenizer functionalities. See the
    [`~OwlViTProcessor.__call__`] and [`~OwlViTProcessor.decode`] for more information.

    Args:
        image_processor ([`OwlViTImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`CLIPTokenizer`, `CLIPTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "OwlViTImageProcessor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")
    # For backward compatibility. See transformers.processing_utils.ProcessorMixin.prepare_and_validate_optional_call_args for more details.
    optional_call_args = ["query_images"]

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
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        # The following is to capture `query_images` argument that may be passed as a positional argument.
        # See transformers.processing_utils.ProcessorMixin.prepare_and_validate_optional_call_args for more details,
        # or this conversation for more context: https://github.com/huggingface/transformers/pull/32544#discussion_r1720208116
        # This behavior is only needed for backward compatibility and will be removed in future versions.
        #
        *args,
        audio=None,
        videos=None,
        **kwargs: Unpack[OwlViTProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several text(s) and image(s). This method forwards the `text` and
        `kwargs` arguments to CLIPTokenizerFast's [`~CLIPTokenizerFast.__call__`] if `text` is not `None` to encode:
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`,
            `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            query_images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The query image to be prepared, one query image is expected per target image to be queried. Each image
                can be a PIL image, NumPy array or PyTorch tensor. In case of a NumPy array/PyTorch tensor, each image
                should be of shape (C, H, W), where C is a number of channels, H and W are image height and width.
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
            - **query_pixel_values** -- Pixel values of the query images to be fed to a model. Returned when `query_images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            OwlViTProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
            **self.prepare_and_validate_optional_call_args(*args),
        )
        query_images = output_kwargs["images_kwargs"].pop("query_images", None)
        return_tensors = output_kwargs["common_kwargs"]["return_tensors"]

        if text is None and query_images is None and images is None:
            raise ValueError(
                "You have to specify at least one text or query image or image. All three cannot be none."
            )
        # check if images and text inputs are reversed for BC
        images, text = _validate_images_text_input_order(images, text)

        data = {}
        if text is not None:
            if isinstance(text, str) or (isinstance(text, List) and not isinstance(text[0], List)):
                encodings = [self.tokenizer(text, **output_kwargs["text_kwargs"])]

            elif isinstance(text, List) and isinstance(text[0], List):
                encodings = []

                # Maximum number of queries across batch
                max_num_queries = max([len(text_single) for text_single in text])

                # Pad all batch samples to max number of text queries
                for text_single in text:
                    if len(text_single) != max_num_queries:
                        text_single = text_single + [" "] * (max_num_queries - len(text_single))

                    encoding = self.tokenizer(text_single, **output_kwargs["text_kwargs"])
                    encodings.append(encoding)
            else:
                raise TypeError("Input text should be a string, a list of strings or a nested list of strings")

            if return_tensors == "np":
                input_ids = np.concatenate([encoding["input_ids"] for encoding in encodings], axis=0)
                attention_mask = np.concatenate([encoding["attention_mask"] for encoding in encodings], axis=0)

            elif return_tensors == "jax" and is_flax_available():
                import jax.numpy as jnp

                input_ids = jnp.concatenate([encoding["input_ids"] for encoding in encodings], axis=0)
                attention_mask = jnp.concatenate([encoding["attention_mask"] for encoding in encodings], axis=0)

            elif return_tensors == "pt" and is_torch_available():
                import torch

                input_ids = torch.cat([encoding["input_ids"] for encoding in encodings], dim=0)
                attention_mask = torch.cat([encoding["attention_mask"] for encoding in encodings], dim=0)

            elif return_tensors == "tf" and is_tf_available():
                import tensorflow as tf

                input_ids = tf.stack([encoding["input_ids"] for encoding in encodings], axis=0)
                attention_mask = tf.stack([encoding["attention_mask"] for encoding in encodings], axis=0)

            else:
                raise ValueError("Target return tensor type could not be returned")

            data["input_ids"] = input_ids
            data["attention_mask"] = attention_mask

        if query_images is not None:
            query_pixel_values = self.image_processor(query_images, **output_kwargs["images_kwargs"]).pixel_values
            # Query images always override the text prompt
            data = {"query_pixel_values": query_pixel_values}

        if images is not None:
            image_features = self.image_processor(images, **output_kwargs["images_kwargs"])
            data["pixel_values"] = image_features.pixel_values

        return BatchFeature(data=data, tensor_type=return_tensors)

    def post_process(self, *args, **kwargs):
        """
        This method forwards all its arguments to [`OwlViTImageProcessor.post_process`]. Please refer to the docstring
        of this method for more information.
        """
        return self.image_processor.post_process(*args, **kwargs)

    def post_process_object_detection(self, *args, **kwargs):
        """
        This method forwards all its arguments to [`OwlViTImageProcessor.post_process_object_detection`]. Please refer
        to the docstring of this method for more information.
        """
        warnings.warn(
            "`post_process_object_detection` method is deprecated for OwlVitProcessor and will be removed in v5. "
            "Use `post_process_grounded_object_detection` instead.",
            FutureWarning,
        )
        return self.image_processor.post_process_object_detection(*args, **kwargs)

    def post_process_grounded_object_detection(
        self,
        outputs: "OwlViTObjectDetectionOutput",
        threshold: float = 0.1,
        target_sizes: Optional[Union[TensorType, List[Tuple]]] = None,
        text_labels: Optional[List[List[str]]] = None,
    ):
        """
        Converts the raw output of [`OwlViTForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format.

        Args:
            outputs ([`OwlViTObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*, defaults to 0.1):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
            text_labels (`List[List[str]]`, *optional*):
                List of lists of text labels for each image in the batch. If unset, "text_labels" in output will be
                set to `None`.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the following keys:
            - "scores": The confidence scores for each predicted box on the image.
            - "labels": Indexes of the classes predicted by the model on the image.
            - "boxes": Image bounding boxes in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format.
            - "text_labels": The text labels for each predicted bounding box on the image.
        """
        output = self.image_processor.post_process_object_detection(
            outputs=outputs, threshold=threshold, target_sizes=target_sizes
        )

        if text_labels is not None and len(text_labels) != len(output):
            raise ValueError("Make sure that you pass in as many lists of text labels as images")

        # adding text labels to the output
        if text_labels is not None:
            for image_output, image_text_labels in zip(output, text_labels):
                object_text_labels = [image_text_labels[i] for i in image_output["labels"]]
                image_output["text_labels"] = object_text_labels
        else:
            for image_output in output:
                image_output["text_labels"] = None

        return output

    def post_process_image_guided_detection(
        self,
        outputs: "OwlViTImageGuidedObjectDetectionOutput",
        threshold: float = 0.0,
        nms_threshold: float = 0.3,
        target_sizes: Optional[Union[TensorType, List[Tuple]]] = None,
    ):
        """
        Converts the output of [`OwlViTForObjectDetection.image_guided_detection`] into the format expected by the COCO
        api.

        Args:
            outputs ([`OwlViTImageGuidedObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*, defaults to 0.0):
                Minimum confidence threshold to use to filter out predicted boxes.
            nms_threshold (`float`, *optional*, defaults to 0.3):
                IoU threshold for non-maximum suppression of overlapping boxes.
            target_sizes (`torch.Tensor`, *optional*):
                Tensor of shape (batch_size, 2) where each entry is the (height, width) of the corresponding image in
                the batch. If set, predicted normalized bounding boxes are rescaled to the target sizes. If left to
                None, predictions will not be unnormalized.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the following keys:
            - "scores": The confidence scores for each predicted box on the image.
            - "boxes": Image bounding boxes in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format.
            - "labels": Set to `None`.
        """
        return self.image_processor.post_process_image_guided_detection(
            outputs=outputs, threshold=threshold, nms_threshold=nms_threshold, target_sizes=target_sizes
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


__all__ = ["OwlViTProcessor"]
