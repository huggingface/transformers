# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
Image/Text processor class for OwlViT
"""
from typing import List

import numpy as np
import jax.numpy as jnp

from ...utils import is_torch_available
from ...utils.generic import _is_torch
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding


def is_torch_tensor(obj):
    return _is_torch(obj)

class OwlViTProcessor(ProcessorMixin):
    r"""
    Constructs a OwlViT processor which wraps a OwlViT feature extractor and a CLIP tokenizer into a single processor.
    [`OwlViTProcessor`] offers all the functionalities of [`OwlViTFeatureExtractor`] and [`CLIPTokenizerFast`]. See the
    [`~OwlViTProcessor.__call__`] and [`~OwlViTProcessor.decode`] for more information.
    Args:
        feature_extractor ([`OwlViTFeatureExtractor`]):
            The feature extractor is a required input.
        tokenizer ([`CLIPTokenizerFast`]):
            The tokenizer is a required input.
    """
    feature_extractor_class = "OwlViTFeatureExtractor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to CLIPTokenizerFast's [`~CLIPTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPFeatureExtractor's [`~CLIPFeatureExtractor.__call__`] if `images` is not `None`. Please refer to the
        doctsring of the above two methods for more information.
        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:
            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """

        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            if isinstance(text, str):
                encodings = [self.tokenizer(text, return_tensors=return_tensors, **kwargs)]

            if isinstance(text, List) and not isinstance(text[0], List):
                encodings = [self.tokenizer(text, return_tensors=return_tensors, **kwargs)]

            if isinstance(text, List) and isinstance(text[0], List):
                encodings = []

                # Maximum number of queries across batch
                max_num_queries = max([len(t) for t in text])

                # Pad all batch samples to max number of text queries
                for t in text:
                    if len(t) != max_num_queries:
                        t.extend([""]*(max_num_queries - len(t)))
                        encoding = self.tokenizer(t, return_tensors=return_tensors, **kwargs)
                        encodings.append(encoding)
                    else:
                        encoding = self.tokenizer(t, return_tensors=return_tensors, **kwargs)
                        encodings.append(encoding)        

            output = encodings[0]

            if return_tensors == "np":
                input_ids = [np.expand_dims(encoding["input_ids"], axis=0) for encoding in encodings]
                input_ids = np.concatenate(input_ids)

                attention_mask = [np.expand_dims(encoding["attention_mask"], axis=0) for encoding in encodings]
                attention_mask = np.concatenate(attention_mask)

            elif return_tensors == "jax":
                input_ids = [jnp.expand_dims(encoding["input_ids"], axis=0) for encoding in encodings]
                input_ids = jnp.concatenate(input_ids)

                attention_mask = [jnp.expand_dims(encoding["attention_mask"], axis=0) for encoding in encodings]
                attention_mask = jnp.concatenate(attention_mask)

            elif return_tensors == "pt":
                import torch
                input_ids= [encoding["input_ids"].unsqueeze(0) for encoding in encodings]
                input_ids = torch.cat(input_ids)

                attention_mask= [encoding["attention_mask"].unsqueeze(0) for encoding in encodings]
                attention_mask = torch.cat(attention_mask)
            else:
                import tensorflow as tf
                input_ids = [tf.expand_dims(encoding["input_ids"], axis=0) for encoding in encodings]
                input_ids = tf.concat(input_ids, axis=0)

                attention_mask = [tf.expand_dims(encoding["attention_mask"], axis=0) for encoding in encodings]
                attention_mask = tf.concat(attention_mask, axis=0)

            output["input_ids"] = input_ids
            output["attention_mask"] = attention_mask

        if images is not None:
            image_features = self.feature_extractor(images, return_tensors=return_tensors, **kwargs)

        if text is not None and images is not None:
            output["pixel_values"] = image_features.pixel_values
            return output
        elif text is not None:
            return output
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)

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