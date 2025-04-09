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

from typing import List, Optional

import numpy as np

from ...processing_utils import ProcessorMixin
from ...utils import to_numpy


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

    def __call__(self, audio=None, text=None, **kwargs):
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `audio`
        and `kwargs` arguments to MusicgenMelodyFeatureExtractor's [`~MusicgenMelodyFeatureExtractor.__call__`] if `audio` is not
        `None` to pre-process the audio. It also forwards the `text` and `kwargs` arguments to
        PreTrainedTokenizer's [`~PreTrainedTokenizer.__call__`] if `text` is not `None`. Please refer to the docstring of the above two methods for more information.

        Args:
            audio (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The audio or batch of audios to be prepared. Each audio can be NumPy array or PyTorch tensor. In case
                of a NumPy array/PyTorch tensor, each audio should be a mono-stereo signal of shape (T), where T is the sample length of the audio.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to the feature extractor and/or the
                tokenizer.
        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:
            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **input_features** -- Audio input features to be fed to a model. Returned when `audio` is not `None`.
            - **attention_mask** -- List of token indices specifying which tokens should be attended to by the model when `text` is not `None`.
            When only `audio` is specified, returns the timestamps attention mask.
        """

        sampling_rate = kwargs.pop("sampling_rate", None)

        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        if text is not None:
            inputs = self.tokenizer(text, **kwargs)
        if audio is not None:
            audio_inputs = self.feature_extractor(audio, sampling_rate=sampling_rate, **kwargs)

        if text is None:
            return audio_inputs
        elif audio is None:
            return inputs
        else:
            inputs["input_features"] = audio_inputs["input_features"]
            return inputs

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

    # Copied from transformers.models.musicgen.processing_musicgen.MusicgenProcessor.decode
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to T5Tokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the
        docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    # Copied from transformers.models.musicgen.processing_musicgen.MusicgenProcessor._decode_audio with padding_mask->attention_mask
    def _decode_audio(self, audio_values, attention_mask: Optional = None) -> List[np.ndarray]:
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
