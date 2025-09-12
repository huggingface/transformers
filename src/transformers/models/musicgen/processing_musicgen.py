# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
Text/audio processor class for MusicGen
"""

from typing import Any

import numpy as np

from ...processing_utils import ProcessorMixin
from ...utils import to_numpy


class MusicgenProcessor(ProcessorMixin):
    r"""
    Constructs a MusicGen processor which wraps an EnCodec feature extractor and a T5 tokenizer into a single processor
    class.

    [`MusicgenProcessor`] offers all the functionalities of [`EncodecFeatureExtractor`] and [`TTokenizer`]. See
    [`~MusicgenProcessor.__call__`] and [`~MusicgenProcessor.decode`] for more information.

    Args:
        feature_extractor (`EncodecFeatureExtractor`):
            An instance of [`EncodecFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`T5Tokenizer`):
            An instance of [`T5Tokenizer`]. The tokenizer is a required input.
    """

    feature_extractor_class = "EncodecFeatureExtractor"
    tokenizer_class = ("T5Tokenizer", "T5TokenizerFast")

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        return self.tokenizer.get_decoder_prompt_ids(task=task, language=language, no_timestamps=no_timestamps)

    def __call__(self, *args, **kwargs):
        """
        Forwards the `audio` argument to EncodecFeatureExtractor's [`~EncodecFeatureExtractor.__call__`] and the `text`
        argument to [`~T5Tokenizer.__call__`]. Please refer to the docstring of the above two methods for more
        information.
        """
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        if len(args) > 0:
            kwargs["audio"] = args[0]
        return super().__call__(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        This method is used to decode either batches of audio outputs from the MusicGen model, or batches of token ids
        from the tokenizer. In the case of decoding token ids, this method forwards all its arguments to T5Tokenizer's
        [`~PreTrainedTokenizer.batch_decode`]. Please refer to the docstring of this method for more information.
        """
        audio_values = kwargs.pop("audio", None)
        padding_mask = kwargs.pop("padding_mask", None)

        if len(args) > 0:
            audio_values = args[0]
            args = args[1:]

        if audio_values is not None:
            return self._decode_audio(audio_values, padding_mask=padding_mask)
        else:
            return self.tokenizer.batch_decode(*args, **kwargs)

    def _decode_audio(self, audio_values, padding_mask: Any = None) -> list[np.ndarray]:
        """
        This method strips any padding from the audio values to return a list of numpy audio arrays.
        """
        audio_values = to_numpy(audio_values)
        bsz, channels, seq_len = audio_values.shape

        if padding_mask is None:
            return list(audio_values)

        padding_mask = to_numpy(padding_mask)

        # match the sequence length of the padding mask to the generated audio arrays by padding with the **non-padding**
        # token (so that the generated audio values are **not** treated as padded tokens)
        difference = seq_len - padding_mask.shape[-1]
        padding_value = 1 - self.feature_extractor.padding_value
        padding_mask = np.pad(padding_mask, ((0, 0), (0, difference)), "constant", constant_values=padding_value)

        audio_values = audio_values.tolist()
        for i in range(bsz):
            sliced_audio = np.asarray(audio_values[i])[
                padding_mask[i][None, :] != self.feature_extractor.padding_value
            ]
            audio_values[i] = sliced_audio.reshape(channels, -1)

        return audio_values


__all__ = ["MusicgenProcessor"]
