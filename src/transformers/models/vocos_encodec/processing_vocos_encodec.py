# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Processor class for VocosEncodec"""

from collections.abc import Sequence

from ...audio_utils import AudioInput, make_list_of_audio
from ...processing_utils import AudioKwargs, BatchFeature, ProcessingKwargs, ProcessorMixin, Unpack
from ...utils import is_torch_available


if is_torch_available():
    import torch
    import torch.nn.functional as F


class VocosAudioKwargs(AudioKwargs, total=False):
    bandwidths: Sequence[float]


class VocosProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: VocosAudioKwargs
    _defaults = {
        "audio_kwargs": {
            "bandwidths": [1.5, 3.0, 6.0, 12.0],
            "sampling_rate": 24000,
            "padding": True,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class VocosEncodecProcessor(ProcessorMixin):
    r"""
    Constructs a VocosEncodec processor which  prepares inputs for [`VocosEncodecModel`] using the audio tokenizer [`EncodecModel`]. It supports two workflows:
        - Audio reconstruction from audio: The processor runs the wrapped [`EncodecModel`] to obtain
        quantized codes at the requested `bandwidth`, then converts them into embeddings (`input_features`) expected by
        [`VocosEncodecModel`]. For this provide `audio` and `bandwidth.

        - Audio reconstruction from precomputed codes: The processor converts `codes` into embeddings
        (`input_features`) expected by [`VocosEncodecModel`]. For this provide `codes` and `bandwidth`.

    Args:
        feature_extractor (`VocosFeatureExtractor`):
            Used only for padding `attention_mask` when processing audio batches. It does not compute
            mel-spectrogram features in this processor.
        audio_tokenizer (`EncodecModel`):
            Audio tokenizer used to encode raw audio into discrete codebooks via the EnCodec model, then
            turns those codes into token embeddings.
    """

    feature_extractor_class = "VocosFeatureExtractor"
    audio_tokenizer_class = "EncodecModel"
    attributes = ["feature_extractor", "audio_tokenizer"]

    def __init__(self, feature_extractor, audio_tokenizer):
        super().__init__(feature_extractor=feature_extractor, audio_tokenizer=audio_tokenizer)

    def __call__(
        self,
        audio: AudioInput | None = None,
        codes=None,
        bandwidth: float | None = None,
        return_tensors: str = "pt",
        device: str | None = None,
        **kwargs: Unpack[VocosProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare inputs for [`VocosEncodecModel`].


        - EnCodec variant (`codes` or `bandwidth` provided): if `audio` provided and `codes` not provided, embeddings
        are computed with the neural audio codec EnCodec [`EncodecModel`] with the given `bandwidth`; if `codes` are
        provided, the corresponding embeddings are computed for the target bandwidth.

        Args:
            audio (`np.ndarray`, `torch.Tensor`, *optional*):
                Audio input to be processed of shape `(sequence_length,)` or `(batch_size, sequence_length)`.
            codes (`torch.LongTensor`, *optional*):
                Pre-computed EnCodec quantized codes of shape `(num_codebooks, sequence_length)` or `(num_codebooks, batch_size, sequence_length)`
            bandwidth (`float`):
                Target EnCodec bandwidth in kbps, it must be one of the configured `bandwidths` [1.5, 3.0, 6.0, 12.0].
            return_tensors (`str`, defaults to `"pt"`):
                Only `"pt"` (PyTorch tensors) is supported.
            device (`str`, *optional*):
                Device on which EnCodec encoding and embedding are computed.

        Returns:
            [`BatchFeature`]: Contains:
            - `input_features` (`torch.FloatTensor`): EnCodec-embedded features of shape `(batch_size, feature_dim, time_dim)`.
            - `bandwidth` (`float`): The bandwidth used to obtain the codes / embeddings.
            - `attention_mask` (`torch.Tensor`, *optional*): Present when `audio` was provided and padding was applied.
              The mask corresponds to the padded input waveform samples and is passed through by the model (not used).
        """

        output_kwargs = self._merge_kwargs(VocosProcessorKwargs, **kwargs)
        audio_kwargs = output_kwargs["audio_kwargs"]
        return_tensors = audio_kwargs.get("return_tensors", None)
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if audio is None and codes is None:
            raise ValueError("Either `audio` or `codes` must be provided.")

        if audio is not None and codes is not None:
            raise ValueError("Both `audio` and `codes` were set, make sure you only set one.")

        if bandwidth is None:
            raise ValueError("`bandwidth` must be provided for EnCodec processing.")

        self.sampling_rate = audio_kwargs["sampling_rate"]
        self.bandwidths = audio_kwargs["bandwidths"]

        if bandwidth not in self.bandwidths:
            raise ValueError(f"bandwidth {bandwidth} is not supported, supported bandwidths are {self.bandwidths}")

        if audio is not None:
            if bandwidth is not None:
                audio_list = make_list_of_audio(audio)
                pad_to_multiple_of = None if len(audio_list) == 1 else self.audio_tokenizer.config.hop_length
                audio_list = [torch.as_tensor(_audio).view(-1, 1) for _audio in audio_list]
                batch = BatchFeature({"input_features": audio_list})
                padded_audio = self.feature_extractor.pad(
                    batch, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=True, return_tensors="pt"
                )

                attention_mask = padded_audio["attention_mask"]
                padded_audio = padded_audio["input_features"].transpose(1, 2)

                with torch.no_grad():
                    # encode audio as in original:
                    # https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/feature_extractors.py#L79
                    self.audio_tokenizer.to(device)
                    embeddings = self.audio_tokenizer.encoder(padded_audio.to(device))
                    codes = self.audio_tokenizer.quantizer.encode(embeddings, bandwidth=bandwidth)

        if codes is not None:
            if codes.ndim not in (2, 3):
                raise ValueError(
                    f"`codes` must have shape (num_codebooks, sequence_length) or (num_codebooks, batch_size, sequence_length), but got {codes.shape}."
                )
            if codes.dim() == 2:
                codes = codes.unsqueeze(1)

            # Extract codebook weights: https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/feature_extractors.py#L71
            num_quantizers = self.audio_tokenizer.quantizer.get_num_quantizers_for_bandwidth(max(self.bandwidths))
            codebook_weights = torch.cat(
                [layer.codebook.embed for layer in self.audio_tokenizer.quantizer.layers[:num_quantizers]], dim=0
            ).to(codes.device)

            num_bins = self.audio_tokenizer.quantizer.codebook_size

            # Embed with position https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/pretrained.py#L117
            offsets = torch.arange(0, num_bins * len(codes), num_bins, device=codes.device).reshape(-1, 1, 1)
            embeddings_idxs = codes + offsets
            input_features = F.embedding(embeddings_idxs, codebook_weights).sum(dim=0).transpose(1, 2)

        data = {"input_features": input_features, "bandwidth": float(bandwidth)}
        if audio is not None:
            data["attention_mask"] = attention_mask

        return BatchFeature(data, tensor_type=return_tensors)


__all__ = ["VocosEncodecProcessor"]
