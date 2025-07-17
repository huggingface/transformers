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
from pathlib import Path
from typing import Optional, Union

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import AudioKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...utils import is_soundfile_available, is_torch_available


if is_torch_available():
    import torch

if is_soundfile_available():
    import soundfile as sf


class DiaAudioKwargs(AudioKwargs, total=False):
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    delay_pattern: list[int]
    generation: bool


class DiaProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: DiaAudioKwargs
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "right",
            "add_special_tokens": False,
        },
        "audio_kwargs": {
            "eos_token_id": 1024,
            "pad_token_id": 1025,
            "bos_token_id": 1026,
            "delay_pattern": [0, 8, 9, 10, 11, 12, 13, 14, 15],
            "generation": True,
            "sampling_rate": 44100,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


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
        audio_tokenizer (`DacModel`):
            An instance of [`DacModel`] used to encode/decode audio into/from codebooks. It is is a required input.
    """

    feature_extractor_class = "DiaFeatureExtractor"
    tokenizer_class = "DiaTokenizer"
    audio_tokenizer_class = "DacModel"

    def __init__(self, feature_extractor, tokenizer, audio_tokenizer):
        super().__init__(feature_extractor, tokenizer, audio_tokenizer=audio_tokenizer)

    @property
    def model_input_names(self):
        """
        We no longer pass the raw audio values but the codebooks encoded by the `audio_tokenizer`.
        Conventions may differ between audio models due to architectural choices.
        """
        tokenizer_input_names = self.tokenizer.model_input_names
        audio_tokenizer_input_names = ["decoder_input_ids", "decoder_attention_mask"]
        return list(dict.fromkeys(tokenizer_input_names + audio_tokenizer_input_names))

    def __call__(
        self,
        text: Union[str, list[str]],
        audio: Optional[AudioInput] = None,
        output_labels: Optional[bool] = False,
        **kwargs: Unpack[DiaProcessorKwargs],
    ):
        """
        Main method to prepare text(s) and audio to be fed as input to the model. The `audio` argument is
        forwarded to the DiaFeatureExtractor's [`~DiaFeatureExtractor.__call__`] and subsequently to the
        DacModel's [`~DacModel.encode`]. The `text` argument to [`~DiaTokenizer.__call__`]. Please refer
        to the docstring of the above methods for more information.
        """
        if not is_torch_available():
            raise ValueError(
                "The `DiaProcessor` relies on the `audio_tokenizer` which requires `torch` but we couldn't "
                "find it in your environment. You can install torch via `pip install torch`."
            )

        if text is None:
            raise ValueError("You need to specify the `text` input to process.")

        output_kwargs = self._merge_kwargs(
            DiaProcessorKwargs,
            **kwargs,
        )

        text_kwargs = output_kwargs["text_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]
        common_kwargs = output_kwargs["common_kwargs"]

        return_tensors = common_kwargs.pop("return_tensors", None)
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        data = {}

        # Text
        if isinstance(text, str):
            text = [text]
        elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        encodings = self.tokenizer(text, **text_kwargs)
        data.update(encodings)

        # Audio
        delay_pattern = audio_kwargs.pop("delay_pattern", None)
        audio_bos_token_id = audio_kwargs.pop("bos_token_id", None)
        audio_eos_token_id = audio_kwargs.pop("eos_token_id", None)
        audio_pad_token_id = audio_kwargs.pop("pad_token_id", None)
        generation = audio_kwargs.pop("generation", True)
        if (
            audio_bos_token_id is None
            or audio_eos_token_id is None
            or audio_pad_token_id is None
            or delay_pattern is None
        ):
            raise ValueError(
                "To enable processing for Dia, we need the `bos_token_id`, `eos_token_id`, "
                "`pad_token_id`, and `delay_pattern`. You may have accidentally overwritten one of those."
            )

        if generation and output_labels:
            raise ValueError(
                f"Labels with `generation` is incompatible, got generation={generation}, output_labels={output_labels}."
            )

        batch_size = data["input_ids"].shape[0]
        num_channels = len(delay_pattern)
        max_delay = max(delay_pattern)

        # Voice cloning generation / general training
        if audio is not None:
            audio = make_list_of_audio(audio)
            input_audios = self.feature_extractor(audio, **audio_kwargs)

            compression_rate = math.prod(self.audio_tokenizer.config.downsampling_ratios)
            max_encoded_sequence_len = input_audios["padding_mask"][0].shape[-1] // compression_rate

            decoder_input_ids = []
            decoder_attention_mask = []
            # TODO: dac with batching is currently broken, but non-batch is working
            # refer to https://gist.github.com/vasqu/643a45b680cf39fd7467271ee2eb6f80 for a validation script
            for padding_mask, audio in zip(input_audios["padding_mask"], input_audios["input_values"]):
                # get current length with hop length in mind (as if it were sampled as a single audio)
                base_pad_len = self.feature_extractor.hop_length
                current_audio_len = math.ceil(padding_mask.sum(dim=-1) / base_pad_len) * base_pad_len

                encoded_sequence_len = current_audio_len // compression_rate
                padding_len = max_encoded_sequence_len - encoded_sequence_len

                # compute non-padded forward pass; one extra bos (and eos if training) is added
                with torch.no_grad():
                    audio = audio[None, ..., :current_audio_len].to(self.audio_tokenizer.device)
                    input_ids = self.audio_tokenizer.encode(audio).audio_codes.transpose(1, 2)

                if not generation:
                    input_ids = torch.nn.functional.pad(
                        input_ids, pad=(0, 0, 0, 1, 0, 0), mode="constant", value=audio_eos_token_id
                    )

                # apply padding
                # +1 for the bos within the real sequence
                input_ids = torch.nn.functional.pad(
                    input_ids, pad=(0, 0, padding_len + 1, 0, 0, 0), mode="constant", value=audio_bos_token_id
                )
                num_valid_inputs = encoded_sequence_len + 1 + max_delay  # sequence + bos + delay
                num_valid_inputs += 0 if generation else 1  # eos if training
                attention_mask = torch.tensor([0] * padding_len + [1] * num_valid_inputs, dtype=torch.long)[None, :]

                decoder_input_ids.append(input_ids)
                decoder_attention_mask.append(attention_mask)

            decoder_input_ids = torch.cat(decoder_input_ids, dim=0)
            decoder_attention_mask = torch.cat(decoder_attention_mask, dim=0)
        # TTS generation
        elif generation:
            # all bos to start with TTS
            decoder_input_ids = torch.full((batch_size, 1, num_channels), audio_bos_token_id, dtype=torch.long)

            # we preemptively add the delay
            decoder_attention_mask = torch.ones(size=(batch_size, 1 + max_delay), dtype=torch.long)
        else:
            raise ValueError("If you try to train, you should provide audio data as well.")

        if batch_size != decoder_input_ids.shape[0]:
            raise ValueError(
                f"Need the same amount of samples for both text and audio, but got text samples={batch_size} and "
                f"audio samples = {decoder_input_ids.shape[0]} instead."
            )

        # prepare shift indices per delay
        max_seq_len = decoder_attention_mask.shape[-1]
        max_audio_len = max_seq_len - max_delay
        precomputed_idx = self.build_indices(
            bsz=batch_size,
            seq_len=max_seq_len,
            num_channels=num_channels,
            delay_pattern=delay_pattern,
            revert=False,
        )

        # create delay pattern input
        # the pad token will be used for masking which input is valid for prediction during generation
        prefill = torch.full(
            (batch_size, max_seq_len, num_channels),
            fill_value=audio_pad_token_id,
            dtype=torch.int,
        )
        prefill[:, :max_audio_len] = decoder_input_ids

        delayed_decoder_input_ids = self.apply_audio_delay(
            audio=prefill,
            pad_token_id=audio_pad_token_id,
            bos_token_id=audio_bos_token_id,
            precomputed_idx=precomputed_idx,
        )

        data.update({"decoder_input_ids": delayed_decoder_input_ids, "decoder_attention_mask": decoder_attention_mask})

        if output_labels:
            # Base idea is to shift on the sequence dim
            labels = data["decoder_input_ids"].clone()[:, 1:]
            labels[labels == audio_pad_token_id] = -100
            labels[labels == audio_bos_token_id] = -100

            data["labels"] = labels.transpose(1, 2).reshape(batch_size * num_channels, -1).contiguous().long()
            data["decoder_input_ids"] = data["decoder_input_ids"][:, :-1]
            data["decoder_attention_mask"] = data["decoder_attention_mask"][:, :-1]

        return BatchFeature(data=data, tensor_type=return_tensors)

    def batch_decode(
        self,
        decoder_input_ids: "torch.Tensor",
        audio_prompt_len: Optional[int] = None,
        **kwargs: Unpack[DiaProcessorKwargs],
    ) -> list["torch.Tensor"]:
        """
        Decodes a batch of audio codebook sequences into their respective audio waveforms via the
        `audio_tokenizer`. See [`~DacModel.decode`] for more information.

        Args:
            decoder_input_ids (`torch.Tensor`): The complete output sequence of the decoder.
            audio_prompt_len (`int`): The audio prefix length (e.g. when using voice cloning).
        """
        output_kwargs = self._merge_kwargs(
            DiaProcessorKwargs,
            **kwargs,
        )
        audio_kwargs = output_kwargs["audio_kwargs"]

        delay_pattern = audio_kwargs.pop("delay_pattern", None)
        audio_bos_token_id = audio_kwargs.pop("bos_token_id", None)
        audio_pad_token_id = audio_kwargs.pop("pad_token_id", None)
        if audio_bos_token_id is None or audio_pad_token_id is None or delay_pattern is None:
            raise ValueError(
                "To enable decoding for Dia, we need the `bos_token_id`, `pad_token_id`, "
                "and `delay_pattern`. You may have accidentally overwritten one of those."
            )

        # either decode the whole audio sequence or only the generated parts
        if audio_prompt_len is not None:
            audio_prompt_len = torch.tensor(audio_prompt_len, device=decoder_input_ids.device, dtype=torch.long)
            start_of_generation_idx = audio_prompt_len[None].expand(decoder_input_ids.shape[0])
        else:
            start_of_generation_idx = (decoder_input_ids[:, :, 0] == audio_bos_token_id).sum(dim=-1)
        # -1 for the eos token
        end_of_generation_idx = (
            decoder_input_ids.shape[1] - (decoder_input_ids[:, :, 0] == audio_pad_token_id).sum(dim=-1) - 1
        )

        # revert delay
        bsz, seq_len, num_channels = decoder_input_ids.shape
        precomputed_idx = self.build_indices(
            bsz=bsz,
            seq_len=seq_len,
            num_channels=num_channels,
            delay_pattern=delay_pattern,
            revert=True,
        )

        output_sequences = self.apply_audio_delay(
            audio=decoder_input_ids,
            # We do not care about these values as we cut them out
            # with `start_of_generation_idx` and `end_of_generation_idx`
            pad_token_id=-1,
            bos_token_id=-1,
            precomputed_idx=precomputed_idx,
        ).transpose(1, 2)

        # retrieve the correct sequences each
        audios = []
        # TODO: see above, dac doesn't work in batches yet
        with torch.no_grad():
            for i in range(start_of_generation_idx.shape[0]):
                output_i = output_sequences[i, :, start_of_generation_idx[i] : end_of_generation_idx[i]][None, ...]
                output_i = output_i.to(self.audio_tokenizer.device)
                audio_i = self.audio_tokenizer.decode(audio_codes=output_i).audio_values.cpu().squeeze()
                audios.append(audio_i)

        return audios

    def decode(
        self,
        decoder_input_ids: "torch.Tensor",
        audio_prompt_len: Optional[int] = None,
        **kwargs: Unpack[DiaProcessorKwargs],
    ) -> "torch.Tensor":
        """
        Decodes a single sequence of audio codebooks into the respective audio waveform via the
        `audio_tokenizer`. See [`~DacModel.decode`] and [`~DiaProcessor.batch_decode`] for more information.
        """
        if decoder_input_ids.shape[0] != 1:
            raise ValueError(
                f"Expecting a single output to be decoded but received {decoder_input_ids.shape[0]} samples instead."
            )

        return self.batch_decode(decoder_input_ids, audio_prompt_len, **kwargs)[0]

    def get_audio_prompt_len(
        self,
        decoder_attention_mask: "torch.Tensor",
        **kwargs: Unpack[DiaProcessorKwargs],
    ) -> int:
        """Utility function to get the audio prompt length."""
        output_kwargs = self._merge_kwargs(
            DiaProcessorKwargs,
            **kwargs,
        )
        audio_kwargs = output_kwargs["audio_kwargs"]

        delay_pattern = audio_kwargs.pop("delay_pattern", None)
        if delay_pattern is None:
            raise ValueError(
                "To enable the utility of retrieving the prompt length for Dia, we need the "
                "`delay_pattern`. You may have accidentally overwritten this."
            )
        return decoder_attention_mask.shape[1] - max(delay_pattern)

    # Copied from transformers.models.csm.processing_csm.CsmProcessor.save_audio with Csm->Dia
    def save_audio(
        self,
        audio: AudioInput,
        saving_path: Union[str, Path, list[Union[str, Path]]],
        **kwargs: Unpack[DiaProcessorKwargs],
    ):
        # TODO: @eustlb, this should be in AudioProcessor
        if not is_soundfile_available():
            raise ImportError("Please install `soundfile` to save audio files.")

        # ensure correct audio input
        audio = make_list_of_audio(audio)

        # ensure correct saving path
        if isinstance(saving_path, (str, Path)):
            saving_path = [saving_path]
        elif not (isinstance(saving_path, (list, tuple)) and all(isinstance(p, (str, Path)) for p in saving_path)):
            raise ValueError("Invalid input path. Please provide a string, or a list of strings")

        if len(audio) != len(saving_path):
            raise ValueError("The number of audio and saving paths must be the same")

        output_kwargs = self._merge_kwargs(
            DiaProcessorKwargs,
            **kwargs,
        )
        audio_kwargs = output_kwargs["audio_kwargs"]
        sampling_rate = audio_kwargs["sampling_rate"]

        for audio_value, p in zip(audio, saving_path):
            if isinstance(audio_value, torch.Tensor):
                audio_value = audio_value.cpu().float().numpy()
            sf.write(p, audio_value, sampling_rate)

    @staticmethod
    def build_indices(
        bsz: int,
        seq_len: int,
        num_channels: int,
        delay_pattern: list[int],
        revert: bool = False,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        Precompute (sequence_idx, all_idx) so that out[seq, channel] = in[seq - delay[channel], channel]
        or in[seq, channel] = out[seq + delay[channel], channel] if `revert`.
        Negative sequence_idx => BOS; sequence_idx >= seq_len => PAD.
        """
        delay_array = torch.tensor(delay_pattern, dtype=torch.int32)

        # (0..seq_len-1)
        sequence_idx = torch.arange(seq_len, dtype=torch.int32)[None, :].expand(bsz, seq_len)[..., None]
        # + or - delay depending if we delay or revert the delay
        if not revert:
            sequence_idx = sequence_idx - delay_array[None, None, :]
        else:
            sequence_idx = sequence_idx + delay_array[None, None, :]
        # if delay goes over the range we clamp back to valid values
        valid_sequence_idx = torch.clamp(sequence_idx, 0, seq_len - 1)

        batch_idx = torch.arange(bsz, dtype=torch.int32)[:, None, None].expand(bsz, seq_len, num_channels)
        channel_idx = torch.arange(num_channels, dtype=torch.int32)[None, None, :].expand(bsz, seq_len, num_channels)

        all_idx = torch.stack(
            [batch_idx.reshape(-1), valid_sequence_idx.reshape(-1), channel_idx.reshape(-1)],
            dim=1,
        ).long()

        return sequence_idx, all_idx

    @staticmethod
    def apply_audio_delay(
        audio: "torch.Tensor",
        pad_token_id: int,
        bos_token_id: int,
        precomputed_idx: tuple["torch.Tensor", "torch.Tensor"],
    ) -> "torch.Tensor":
        """
        Applies or reverts the delay pattern to batched audio tokens using precomputed indices,
        inserting BOS where sequence_idx < 0 and PAD where sequence_idx >= seq_len.

        Args:
            audio: audio tokens of shape [bsz, seq_len, num_channels]
            pad_token_id: the PAD token
            bos_token_id: the BOS token
            precomputed_idx: from `build_indices`

        Returns:
            final_audio: delayed or reverted audio tokens of shape [bsz, seq_len, num_channels]
        """
        # Move everything to the same device
        device = audio.device
        sequence_idx, all_idx = precomputed_idx
        sequence_idx = sequence_idx.to(device)
        all_idx = all_idx.to(device)

        # Gather per precomputed indices
        batch_idx, valid_sequence_idx, channel_idx = torch.unbind(all_idx, dim=-1)
        gathered_audio = audio[batch_idx, valid_sequence_idx, channel_idx].view(audio.size())

        # Mask according to negative sequence_idx => BOS; sequence_idx >= seq_len => PAD
        mask_bos = sequence_idx < 0
        mask_pad = sequence_idx >= audio.shape[1]
        final_audio = torch.where(mask_bos, bos_token_id, torch.where(mask_pad, pad_token_id, gathered_audio))

        return final_audio


__all__ = ["DiaProcessor"]
