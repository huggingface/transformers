# Copyright 2026 The HuggingFace Inc. team.
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


import torch
from torch.nn.functional import layer_norm

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging
from ...utils.import_utils import is_torch_available, requires


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


@requires(backends=("torch",))
class OmniASRFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a OmniASR feature extractor.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding values.
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether or not [`~OmniASRFeatureExtractor.__call__`] should return `attention_mask`. OmniASR needs the
            mask to know which frames are padding, so batched inputs are only encoded correctly when it is returned.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize each audio sample to zero mean and unit variance. This is applied per sample,
            before padding, to match the [original implementation](https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/datasets/utils/audio.py#L23).
    """

    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        return_attention_mask=True,
        do_normalize=True,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize

    def __call__(
        self,
        audio: AudioInput,
        sampling_rate: int | None = None,
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        truncation: bool = False,
        return_attention_mask: bool | None = None,
        return_tensors: str | TensorType | None = "pt",
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
            audio (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`:
                The sequence or batch of sequences to be processed. Each sequence can be a numpy array, a torch tensor,
                a list of numpy arrays or a list of torch tensors.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `audio` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a
                  single sequence is provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'`: No padding. Since `return_tensors="pt"` is required, this only works for a
                  single sequence or for sequences that already share the same length.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, the value of
                `self.return_attention_mask` is used.
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `"pt"`):
                Only `"pt"` is supported, i.e. returning PyTorch `torch.Tensor` objects.
        """

        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided `raw_speech` input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        # Ensure batch of mono audio
        audio = make_list_of_audio(audio)
        for idx, example in enumerate(audio):
            example = torch.tensor(example, dtype=torch.float32)
            if example.ndim != 1:
                raise ValueError(
                    f"Only mono-channel audio is supported for input to {self}, got shape: {example.shape}"
                )
            if self.do_normalize:
                # Zero mean and unit variance per example, before padding, as in the original implementation:
                # https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/datasets/utils/audio.py#L162
                # https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/datasets/utils/audio.py#L23
                # Normalizing the padded batch instead would make each example's statistics depend on the other
                # examples in the batch and on the amount of padding.
                with torch.no_grad():
                    example = layer_norm(example, example.shape)
            audio[idx] = example

        encoded_inputs = BatchFeature({"input_values": audio})
        return self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
        )


__all__ = ["OmniASRFeatureExtractor"]
