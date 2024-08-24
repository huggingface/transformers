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
Image/Text processor class for ImageBind
"""

from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import BatchEncoding


class ImageBindProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    _defaults = {
        "text_kwargs": {
            "padding": "max_length",
            "max_length": 64,
        },
    }


class ImageBindProcessor(ProcessorMixin):
    r"""
    Constructs a ImageBind processor which wraps a ImageBind image processor and feature extracotr and a CLIP tokenizer into a single processor.

    [`ImageBindProcessor`] offers all the functionalities of [`ImageBindImageProcessor`], [`ImageBindFeatureExtractor`] and [`CLIPTokenizerFast`].
    See the [`~ImageBindProcessor.__call__`] and [`~ImageBindProcessor.decode`] for more information.

    Args:
        image_processor ([`ImageBindImageProcessor`]):
            An instance of [`ImageBindImageProcessor`] to process the images. This is a required input.
        tokenizer ([`CLIPTokenizer`, `CLIPTokenizerFast`]):
            An instance of ['PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]. The tokenizer is a required input.
        feature_extractor ([`ImageBindFeatureExtractor`]):
            An instance of [`ImageBindFeatureExtractor`] to extract features from the audio. This is a required input.
    """

    attributes = ["image_processor", "tokenizer", "feature_extractor"]
    image_processor_class = "ImageBindImageProcessor"
    feature_extractor_class = "ImageBindFeatureExtractor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    def __init__(self, image_processor, tokenizer, feature_extractor):
        super().__init__(image_processor, tokenizer, feature_extractor)

    def __call__(self, images=None, text=None, audio=None, return_tensors=None, **kwargs):
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to ImageBindTokenizerFast's [`~ImageBindTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        ImageBindImageProcessor's [`~ImageBindImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.
        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            audio (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`, `List[List[List[float]]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of numpy
                arrays or a (possibly nested) list of float values. The supported input types are as follows:

                - unbatched: `List[float]`, `np.ndarray` (`ndim=1`)
                - batched: `List[List[float]]`, `List[np.ndarray]` (`ndim=1`), `np.ndarray` (`ndim=2`)
                - batched with clips: `List[List[List[float]]]`, `List[List[np.ndarray]]` (`ndim=1`), `List[np.ndarray]` (`ndim=2`), np.ndarray (`ndim=3`)

                The input will always be interpreted as mono channel audio, not stereo, i.e. a single float per timestep.
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
            - **input_features** -- List of input features to be fed to a model. Returned when `audio` is not `None`.
        """

        if text is None and images is None and audio is None:
            raise ValueError("You have to specify either text, images or audio. Both cannot be none.")

        data = {}

        if text is not None:
            encoding = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
            data.update(encoding)

        if images is not None:
            image_features = self.image_processor(images, return_tensors=return_tensors)
            data.update(image_features)

        if audio is not None:
            audio_features = self.feature_extractor(audio, return_tensors=return_tensors)
            data.update(audio_features)

        return BatchEncoding(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to ImageBindTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to ImageBindTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names + feature_extractor_input_names))
