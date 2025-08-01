# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from typing import Union, Unpack

from ...audio_utils import AudioInput
from ...processing_utils import ProcessorMixin, ProcessingKwargs
from ...feature_extraction_utils import BatchFeature
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging


logger = logging.get_logger(__name__)


class ParakeetProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {"sampling_rate": 16000},
        "common_kwargs": {"return_tensors": "pt"},
    }

class ParakeetProcessor(ProcessorMixin):
    r"""
    Constructs a Parakeet processor which wraps a Parakeet feature extractor and a Parakeet tokenizer into a single processor.

    [`ParakeetProcessor`] offers all the functionalities of [`ParakeetFeatureExtractor`] and [`ParakeetCTCTokenizer`]. See
    the [`~ParakeetProcessor.__call__`] and [`~ParakeetProcessor.decode`] for more information.

    Args:
        feature_extractor ([`ParakeetFeatureExtractor`], *optional*):
            An instance of [`ParakeetFeatureExtractor`].
        tokenizer ([`ParakeetCTCTokenizer`], *optional*):
            An instance of [`ParakeetCTCTokenizer`].
    """

    attributes = ["feature_extractor", "tokenizer"]
    tokenizer_class = "ParakeetCTCTokenizer"
    feature_extractor_class = "ParakeetFeatureExtractor"

    def __init__(self, feature_extractor=None, tokenizer=None):
        super().__init__(feature_extractor, tokenizer)

    def __call__(
        self,
        audio: AudioInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        **kwargs: Unpack[ParakeetProcessorKwargs],
    ):
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to ParakeetCTCTokenizer's [`~ParakeetCTCTokenizer.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audio` and `kwargs` arguments to
        ParakeetFeatureExtractor's [`~ParakeetFeatureExtractor.__call__`] if `audio` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:
            audio (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The audio or batch of audios to be prepared. Each audio can be a NumPy array or PyTorch
                tensor.
            text (`str`, `list[str]`, `list[list[str]]`):
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
            - **input_features** -- Audio features to be fed to a model. Returned when `audio` is not `None`.
            - **attention_mask** -- List of indices specifying which input features should be attended to by the model.
        """
        if audio is None and text is None:
            raise ValueError("You have to specify at least one of `audio` or `text`.")
        
        output_kwargs = self._merge_kwargs(
            ParakeetProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = output_kwargs["text_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]
        common_kwargs = output_kwargs["common_kwargs"]

        data = {}
        if audio is not None:
            inputs = self.feature_extractor(audio, **audio_kwargs)
            data.update(inputs)

        if text is not None:
            encodings = self.tokenizer(text, **text_kwargs)   
            data.update(encodings)

        return BatchFeature(data=data, tensor_type=common_kwargs["return_tensors"])
    
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->ParakeetCTC
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to ParakeetCTCTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->ParakeetCTC
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to ParakeetCTCTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)


__all__ = ["ParakeetProcessor"]
