# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
Processor class for Cosmos.
"""

from typing import List, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, VideoInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging


logger = logging.get_logger(__name__)


class CosmosProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": "max_length",
            "max_length": 512,
        },
        "videos_kwargs": {},
    }


class CosmosProcessor(ProcessorMixin):
    r"""
    Constructs a Cosmos processor which wraps a LLaVa image processor and a LLaMa tokenizer into a single processor.

    [`CosmosProcessor`] offers all the functionalities of [`CosmosVideoProcessor`] and [`T5TokenizerFast`]. See the
    [`~CosmosProcessor.__call__`] and [`~CosmosProcessor.decode`] for more information.

    Args:
        videos_processor ([`CosmosVideoProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`T5TokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["video_processor", "tokenizer"]
    video_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        video_processor=None,
        tokenizer=None,
        **kwargs,
    ):
        super().__init__(video_processor, tokenizer)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos: VideoInput = None,
        **kwargs: Unpack[CosmosProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to T5TokenizerFast's [`~T5TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CosmosVideoProcessor's [`~CosmosVideoProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            videos (`np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a list of PIL frames, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
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
        output_kwargs = self._merge_kwargs(
            CosmosProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        data = self.video_processor(videos, **output_kwargs["videos_kwargs"])

        if text is not None:
            text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
            text_inputs = {f"encoder_{k}": v for k, v in text_inputs.items()}

            data.update(text_inputs)
        return BatchFeature(data=data)

    def postprocess(self, videos, **kwargs):
        return self.video_processor.postprocess(videos)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["CosmosProcessor"]
