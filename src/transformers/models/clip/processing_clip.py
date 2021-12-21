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
Image/Text processor class for CLIP
"""
from ...tokenization_utils_base import BatchEncoding
from .feature_extraction_clip import CLIPFeatureExtractor
from .tokenization_clip import CLIPTokenizer


class CLIPProcessor:
    r"""
    Constructs a CLIP processor which wraps a CLIP feature extractor and a CLIP tokenizer into a single processor.

    [`CLIPProcessor`] offers all the functionalities of [`CLIPFeatureExtractor`]
    and [`CLIPTokenizer`]. See the [`~CLIPProcessor.__call__`] and
    [`~CLIPProcessor.decode`] for more information.

    Args:
        feature_extractor ([`CLIPFeatureExtractor`]):
            The feature extractor is a required input.
        tokenizer ([`CLIPTokenizer`]):
            The tokenizer is a required input.
    """

    def __init__(self, feature_extractor, tokenizer):
        if not isinstance(feature_extractor, CLIPFeatureExtractor):
            raise ValueError(
                f"`feature_extractor` has to be of type CLIPFeatureExtractor, but is {type(feature_extractor)}"
            )
        if not isinstance(tokenizer, CLIPTokenizer):
            raise ValueError(f"`tokenizer` has to be of type CLIPTokenizer, but is {type(tokenizer)}")

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.current_processor = self.feature_extractor

    def save_pretrained(self, save_directory):
        """
        Save a CLIP feature extractor object and CLIP tokenizer object to the directory `save_directory`, so that it
        can be re-loaded using the [`~CLIPProcessor.from_pretrained`] class method.

        <Tip>

        This class method is simply calling [`~PreTrainedFeatureExtractor.save_pretrained`] and
        [`~tokenization_utils_base.PreTrainedTokenizer.save_pretrained`]. Please refer to the
        docstrings of the methods above for more information.

        </Tip>

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
        """

        self.feature_extractor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate a [`CLIPProcessor`] from a pretrained CLIP processor.

        <Tip>

        This class method is simply calling CLIPFeatureExtractor's
        [`~PreTrainedFeatureExtractor.from_pretrained`] and CLIPTokenizer's
        [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`]. Please refer to the
        docstrings of the methods above for more information.

        </Tip>

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like `clip-vit-base-patch32`, or
                  namespaced under a user or organization name, like `openai/clip-vit-base-patch32`.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~PreTrainedFeatureExtractor.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - a path or url to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.

            **kwargs
                Additional keyword arguments passed along to both [`PreTrainedFeatureExtractor`] and
                [`PreTrainedTokenizer`]
        """
        feature_extractor = CLIPFeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the
        `text` and `kwargs` arguments to CLIPTokenizer's [`~CLIPTokenizer.__call__`] if
        `text` is not `None` to encode the text. To prepare the image(s), this method forwards the
        `images` and `kwrags` arguments to CLIPFeatureExtractor's
        [`~CLIPFeatureExtractor.__call__`] if `images` is not `None`. Please refer to the
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

            return_tensors (`str` or [`~file_utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if
              `text` is not `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """

        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, return_tensors=return_tensors, **kwargs)

        if images is not None:
            image_features = self.feature_extractor(images, return_tensors=return_tensors, **kwargs)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values
            return encoding
        elif text is not None:
            return encoding
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizer's
        [`~PreTrainedTokenizer.batch_decode`]. Please refer to the docstring of this method for more
        information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizer's [`~PreTrainedTokenizer.decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)
