# coding=utf-8
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
Text/audio processor class for MusicGen Melody
"""
from typing import List, Optional

import numpy as np

from ...processing_utils import ProcessorMixin
from ...utils import to_numpy

class MusicgenMelodyProcessor(ProcessorMixin):
    r"""
    Constructs a MusicGen Melody processor which wraps a Wav2Vec2 feature extractor - for raw audio waveform processing - and a T5 tokenizer into a single processor
    class.

    [`MusicgenProcessor`] offers all the functionalities of [`Wav2Vec2FeatureExtractor`] and [`T5Tokenizer`]. See
    [`~MusicgenProcessor.__call__`] and [`~MusicgenProcessor.decode`] for more information.

    Args:
        feature_extractor (`Wav2Vec2FeatureExtractor`):
            An instance of [`Wav2Vec2FeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`T5Tokenizer`):
            An instance of [`T5Tokenizer`]. The tokenizer is a required input.
    """

    feature_extractor_class = "Wav2Vec2FeatureExtractor"
    tokenizer_class = ("T5Tokenizer", "T5TokenizerFast")

    # Copied from transformers.models.musicgen.processing_musicgen.MusicgenProcessor.__init__
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    # Copied from transformers.models.musicgen.processing_musicgen.MusicgenProcessor.get_decoder_prompt_ids
    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        return self.tokenizer.get_decoder_prompt_ids(task=task, language=language, no_timestamps=no_timestamps)

    # Copied from transformers.models.musicgen.processing_musicgen.MusicgenProcessor.__call__ with Encodec -> Wav2Vec2
    def __call__(self, *args, **kwargs):
        """
        Forwards the `audio` argument to Wav2Vec2FeatureExtractor's [`~Wav2Vec2FeatureExtractor.__call__`] and the `text`
        argument to [`~T5Tokenizer.__call__`]. Please refer to the doctsring of the above two methods for more
        information.
        """
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        if text is not None:
            inputs = self.tokenizer(text, **kwargs)

        if audio is not None:
            audio_inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)

        if audio is None:
            return inputs

        elif text is None:
            return audio_inputs

        else:
            inputs["input_values"] = audio_inputs["input_values"]
            if "padding_mask" in audio_inputs:
                inputs["padding_mask"] = audio_inputs["padding_mask"]
            return inputs

    # Copied from transformers.models.musicgen.processing_musicgen.MusicgenProcessor.batch_decode
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

    # Copied from transformers.models.musicgen.processing_musicgen.MusicgenProcessor.decode
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to T5Tokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the
        docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    # Copied from transformers.models.musicgen.processing_musicgen.MusicgenProcessor._decode_audio
    def _decode_audio(self, audio_values, padding_mask: Optional = None) -> List[np.ndarray]:
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
