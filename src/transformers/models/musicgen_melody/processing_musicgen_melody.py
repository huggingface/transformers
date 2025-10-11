# coding=utf-8
# Copyright 2024 Meta AI and The HuggingFace Inc. team. All rights reserved.
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

from typing import Any

import numpy as np

from ...processing_utils import ProcessorMixin
from ...utils import to_numpy
from ...utils.import_utils import requires


@requires(backends=("torchaudio",))
class MusicgenMelodyProcessor(ProcessorMixin):
    r"""
    Constructs a MusicGen Melody processor which wraps a Wav2Vec2 feature extractor - for raw audio waveform processing - and a T5 tokenizer into a single processor
    class.

    [`MusicgenProcessor`] offers all the functionalities of [`MusicgenMelodyFeatureExtractor`] and [`T5Tokenizer`]. See
    [`~MusicgenProcessor.__call__`] and [`~MusicgenProcessor.decode`] for more information.

    Args:
        feature_extractor (`MusicgenMelodyFeatureExtractor`):
            An instance of [`MusicgenMelodyFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`T5Tokenizer`):
            An instance of [`T5Tokenizer`]. The tokenizer is a required input.
    """

    feature_extractor_class = "MusicgenMelodyFeatureExtractor"
    tokenizer_class = ("T5Tokenizer", "T5TokenizerFast")

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    # Copied from transformers.models.musicgen.processing_musicgen.MusicgenProcessor.get_decoder_prompt_ids
    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        return self.tokenizer.get_decoder_prompt_ids(task=task, language=language, no_timestamps=no_timestamps)

    def __call__(self, *args, **kwargs):
        """
        Forwards the `audio` argument to EncodecFeatureExtractor's [`~EncodecFeatureExtractor.__call__`] and the `text`
        argument to [`~T5Tokenizer.__call__`]. Please refer to the docstring of the above two methods for more
        information.
        """

        if len(args) > 0:
            kwargs["audio"] = args[0]
        return super().__call__(*args, **kwargs)

    # Copied from transformers.models.musicgen.processing_musicgen.MusicgenProcessor.batch_decode with padding_mask->attention_mask
    def batch_decode(self, *args, **kwargs):
        """
        This method is used to decode either batches of audio outputs from the MusicGen model, or batches of token ids
        from the tokenizer. In the case of decoding token ids, this method forwards all its arguments to T5Tokenizer's
        [`~PreTrainedTokenizer.batch_decode`]. Please refer to the docstring of this method for more information.
        """
        audio_values = kwargs.pop("audio", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if len(args) > 0:
            audio_values = args[0]
            args = args[1:]

        if audio_values is not None:
            return self._decode_audio(audio_values, attention_mask=attention_mask)
        else:
            return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.musicgen.processing_musicgen.MusicgenProcessor._decode_audio with padding_mask->attention_mask
    def _decode_audio(self, audio_values, attention_mask: Any = None) -> list[np.ndarray]:
        """
        This method strips any padding from the audio values to return a list of numpy audio arrays.
        """
        audio_values = to_numpy(audio_values)
        bsz, channels, seq_len = audio_values.shape

        if attention_mask is None:
            return list(audio_values)

        attention_mask = to_numpy(attention_mask)

        # match the sequence length of the padding mask to the generated audio arrays by padding with the **non-padding**
        # token (so that the generated audio values are **not** treated as padded tokens)
        difference = seq_len - attention_mask.shape[-1]
        padding_value = 1 - self.feature_extractor.padding_value
        attention_mask = np.pad(attention_mask, ((0, 0), (0, difference)), "constant", constant_values=padding_value)

        audio_values = audio_values.tolist()
        for i in range(bsz):
            sliced_audio = np.asarray(audio_values[i])[
                attention_mask[i][None, :] != self.feature_extractor.padding_value
            ]
            audio_values[i] = sliced_audio.reshape(channels, -1)

        return audio_values

    def get_unconditional_inputs(self, num_samples=1, return_tensors="pt"):
        """
        Helper function to get null inputs for unconditional generation, enabling the model to be used without the
        feature extractor or tokenizer.

        Args:
            num_samples (int, *optional*):
                Number of audio samples to unconditionally generate.

        Example:
        ```python
        >>> from transformers import MusicgenMelodyForConditionalGeneration, MusicgenMelodyProcessor

        >>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")

        >>> # get the unconditional (or 'null') inputs for the model
        >>> processor = MusicgenMelodyProcessor.from_pretrained("facebook/musicgen-melody")
        >>> unconditional_inputs = processor.get_unconditional_inputs(num_samples=1)

        >>> audio_samples = model.generate(**unconditional_inputs, max_new_tokens=256)
        ```"""
        inputs = self.tokenizer([""] * num_samples, return_tensors=return_tensors, return_attention_mask=True)
        inputs["attention_mask"][:] = 0

        return inputs


__all__ = ["MusicgenMelodyProcessor"]
