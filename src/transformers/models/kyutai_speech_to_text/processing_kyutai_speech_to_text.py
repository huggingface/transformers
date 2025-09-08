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

from typing import Optional

from ...audio_utils import AudioInput, make_list_of_audio
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack


class KyutaiSpeechToTextProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "sampling_rate": 24000,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class KyutaiSpeechToTextProcessor(ProcessorMixin):
    r"""
    Constructs a Moshi ASR processor which wraps [`EncodecFeatureExtractor`] and
    [`PreTrainedTokenizerFast`] into a single processor that inherits both the audio feature extraction and
    tokenizer functionalities. See the [`~KyutaiSpeechToTextProcessor.__call__`] for more
    information.
    """

    feature_extractor_class = "KyutaiSpeechToTextFeatureExtractor"
    tokenizer_class = "PreTrainedTokenizerFast"

    def __call__(
        self,
        audio: Optional[AudioInput] = None,
        **kwargs: Unpack[KyutaiSpeechToTextProcessorKwargs],
    ):
        r"""
        Main method to prepare audio to be fed as input to the model. This method forwards the `audio`
        arguments to KyutaiSpeechToTextFeatureExtractor's [`~KyutaiSpeechToTextFeatureExtractor.__call__`]. Please refer
        to the docstring of the above method for more information.

        Args:
            audio (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The audio or batch of audio to be prepared. Each audio can be a NumPy array or PyTorch
                tensor.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                    - `'tf'`: Return TensorFlow `tf.constant` objects.
                    - `'pt'`: Return PyTorch `torch.Tensor` objects.
                    - `'np'`: Return NumPy `np.ndarray` objects.
                    - `'jax'`: Return JAX `jnp.ndarray` objects.
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_values** -- List of audio values to be fed to a model. Returned when `audio` is not `None`.
            - **padding_mask** -- List of indices specifying which input values should be ignored by the model.
        """

        if audio is None:
            raise ValueError("`audio` is required.")

        output_kwargs = self._merge_kwargs(
            KyutaiSpeechToTextProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        audio_kwargs = output_kwargs["audio_kwargs"]

        # ensure audio in correct format
        audio = make_list_of_audio(audio)

        inputs = self.feature_extractor(
            audio,
            **audio_kwargs,
        )

        return inputs


__all__ = ["KyutaiSpeechToTextProcessor"]
