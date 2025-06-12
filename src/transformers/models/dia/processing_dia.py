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
"""Processor class for Dia"""

import math

import torch

from ...processing_utils import ProcessorMixin
from ..auto import AutoModel


class DiaProcessor(ProcessorMixin):
    r"""
    Constructs a Dia processor which wraps a [`DiaFeatureExtractor`], [`DiaTokenizer`], and a [`DacModel`] into
    a single processor. It inherits, the audio feature extraction, tokenizer, and audio encode/decode functio-
    nalities. See [`~DiaProcessor.__call__`], [`~DiaProcessor.encode`], and [`~DiaProcessor.decode`] for more
    information.

    Args:
        feature_extractor (`DiaFeatureExtractor`):
            An instance of [`DiaFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`DiaTokenizer`):
            An instance of [`DiaTokenizer`]. The tokenizer is a required input.
        audio_model (`str`, *optional*, defaults to `"descript/dac_44khz"`):
            The model to use for audio encoding and decoding.
    """

    feature_extractor_class = "DiaFeatureExtractor"
    tokenizer_class = "DiaTokenizer"

    def __init__(self, feature_extractor, tokenizer, audio_model="descript/dac_44khz"):
        super().__init__(feature_extractor, tokenizer)
        self.audio_tokenizer = AutoModel.from_pretrained(audio_model)

    def __call__(self, *args, **kwargs):
        """
        The `audio` argument is forwarded to the DiaFeatureExtractor's [`~DiaFeatureExtractor.__call__`] and
        subsequently to the DacModel's [`~DacModel.encode`]. The `text` argument to [`~DiaTokenizer.__call__`].
        Please refer to the docstring of the above two methods for more information.
        """
        audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)

        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        if audio is not None:
            input_audios = self.feature_extractor(
                audio, *args, sampling_rate=sampling_rate, return_tensors="pt", **kwargs
            )

            compression_rate = math.prod(self.audio_tokenizer.config.downsampling_ratios)
            max_encoded_sequence_len = input_audios["padding_mask"][0].shape[-1] // compression_rate

            decoder_input_ids = []
            decoder_attention_mask = []
            # TODO: dac with batching is currently broken, but non-batch is working
            # refer to https://gist.github.com/vasqu/643a45b680cf39fd7467271ee2eb6f80 for validation script
            for padding_mask, audio in zip(input_audios["padding_mask"], input_audios["input_values"]):
                # get current length with hop length in mind (as if it were sampled as a single audio)
                base_pad_len = self.feature_extractor.hop_length
                current_audio_len = math.ceil(padding_mask.sum(dim=-1) / base_pad_len) * base_pad_len

                encoded_sequence_len = current_audio_len // compression_rate
                padding_len = max_encoded_sequence_len - encoded_sequence_len

                # compute non-padded forward pass
                input_ids = self.audio_tokenizer.encode(audio[None, ..., :current_audio_len], **kwargs).audio_codes.transpose(1, 2)
                attention_mask = torch.tensor([1] * encoded_sequence_len + [0] * padding_len, device=input_ids.device, dtype=torch.long)[None, :]

                # pad so we get it as a batch
                if padding_len > 0:
                    # TODO left pad with BOS token? above too then
                    input_ids = torch.nn.functional.pad(input_ids, pad=(0, 0, 0, padding_len, 0, 0), mode="constant", value=0)

                decoder_input_ids.append(input_ids)
                decoder_attention_mask.append(attention_mask)

            decoder_input_ids = torch.cat(decoder_input_ids, dim=0)
            decoder_attention_mask = torch.cat(decoder_attention_mask, dim=0)
            inputs = {"decoder_input_ids": decoder_input_ids, "decoder_attention_mask": decoder_attention_mask}

        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs
        elif audio is None:
            return encodings
        else:
            # TODO: labels based on decoder input ids -- add BOS before --> defaults?
            inputs["input_ids"] = encodings["input_ids"]
            inputs["attention_mask"] = encodings["attention_mask"]
            return inputs

    # TODO
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to DiaTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.audio_tokenizer.decode(*args, **kwargs).audio_values

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to DiaTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.audio_tokenizer.decode(*args, **kwargs).audio_values


__all__ = ["DiaProcessor"]
