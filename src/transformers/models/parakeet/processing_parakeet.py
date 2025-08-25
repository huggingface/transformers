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
""" Parakeet processor class. """

from ...audio_utils import AudioInput
from ...processing_utils import ProcessorMixin, ProcessingKwargs, Unpack
from ...feature_extraction_utils import BatchFeature
from ...utils import logging, is_torch_available


if is_torch_available():
    import torch


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
        self.sampling_rate = self.feature_extractor.sampling_rate

    def __call__(
        self,
        audio: AudioInput = None,
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

        output_kwargs = self._merge_kwargs(
            ParakeetProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        audio_kwargs = output_kwargs["audio_kwargs"]
        common_kwargs = output_kwargs["common_kwargs"]
        data = self.feature_extractor(audio, **audio_kwargs)
        return BatchFeature(data=data, tensor_type=common_kwargs["return_tensors"])
    
    def batch_decode(
        self,
        predicted_ids: "torch.Tensor",
        **kwargs: Unpack[ParakeetProcessorKwargs],
    ) -> list["torch.Tensor"]:
        """
        Decodes a batch of ParakeetForCTC model outputs into text.

        Args:
            predicted_ids (`torch.Tensor`):
                The predicted token ids from the model output. This should be a tensor of shape `(batch_size, sequence_length)`.
        """
        return self.tokenizer.batch_decode(predicted_ids, **kwargs)

    def decode(
        self, 
        predicted_ids: "torch.Tensor",
        **kwargs
    ) -> "torch.Tensor":
        """
        Decodes a single output from the ParakeetForCTC model into text.

        Args:
            predicted_ids (`torch.Tensor`):
                The predicted token ids from the model output. This should be a tensor of shape `(1, sequence_length)`.
        """
        if predicted_ids.shape[0] != 1:
            raise ValueError(
                f"Expecting a single output to be decoded but received {predicted_ids.shape[0]} samples instead."
            )
        return self.tokenizer.decode(predicted_ids[0], **kwargs)


__all__ = ["ParakeetProcessor"]
