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
from typing import List, Optional, Tuple, Union

import torch

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import AudioKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ..auto import AutoModel


class DiaAudioKwargs(AudioKwargs, total=False):
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    delay_pattern: List[int]
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


# TODO: check for correctness
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

    def __call__(
        self,
        text: Union[str, List[str]],
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

        # TODO: check for correctness
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
                "To enable processing for Dia, we need the `bos_token_id`, `eos_token_id`, `pad_token_id`, and `delay_pattern`. You may have accidentally overwritten one of those."
            )

        if generation and output_labels:
            raise ValueError(
                f"Labels with `generation` is incompatible, got generation={generation}, output_labels={output_labels}"
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
                input_ids = self.audio_tokenizer.encode(audio[None, ..., :current_audio_len]).audio_codes.transpose(
                    1, 2
                )
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

        # prepare shift indices per delay
        max_seq_len = decoder_attention_mask.shape[-1]
        max_audio_len = max_seq_len - max_delay
        delay_precomp = self.build_delay_indices(
            B=batch_size,
            T=max_seq_len,
            C=num_channels,
            delay_pattern=delay_pattern,
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
            audio_BxTxC=prefill,
            pad_value=audio_pad_token_id,
            bos_value=audio_bos_token_id,
            precomp=delay_precomp,
        )

        data.update({"decoder_input_ids": delayed_decoder_input_ids, "decoder_attention_mask": decoder_attention_mask})

        if output_labels:
            # Base idea is to shift on the sequence dim
            labels = data["decoder_input_ids"].clone()[:, 1:]
            labels[labels == audio_pad_token_id] = -100
            # TODO: is this correct? this is based on that the delay doesn't need to predict
            labels[labels == audio_bos_token_id] = -100

            data["labels"] = labels
            data["decoder_input_ids"] = data["decoder_input_ids"][:, :-1]
            data["decoder_attention_mask"] = data["decoder_attention_mask"][:, :-1]

        return BatchFeature(data=data, tensor_type=return_tensors)

    def decode(
        self,
        input_ids,
        attention_mask,
        pad_token_id,
        delay_pattern,
    ):
        # +1 for the bos token
        start_of_generation_idx = (~attention_mask.bool()).sum(dim=-1) + 1
        end_of_generation_idx = input_ids.shape[1] - (input_ids[:, :, 0] == pad_token_id).sum(dim=-1)

        # revert delay
        bsz, seq_len, num_channels = input_ids.shape
        revert_precomp = self.build_revert_indices(
            B=bsz,
            T=seq_len,
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        output_sequences = self.revert_audio_delay(
            audio_BxTxC=input_ids,
            pad_value=-1,
            precomp=revert_precomp,
            T=seq_len,
        ).transpose(1, 2)

        # retrieve the correct sequences each
        audios = []
        for i in range(start_of_generation_idx.shape[0]):
            output_i = output_sequences[i, :, start_of_generation_idx[i] : end_of_generation_idx[i]]
            # TODO: see above, dac doesn't work in batches yet
            audio_i = self.audio_tokenizer.decode(audio_codes=output_i).audio_values
            audios.append(audio_i)

        # TODO: numpify, save, etc.

        return audios

    # TODO: rewrite with better namings
    @staticmethod
    def build_delay_indices(B: int, T: int, C: int, delay_pattern: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute (t_idx_BxTxC, indices_BTCx3) so that out[t, c] = in[t - delay[c], c].
        Negative t_idx => BOS; t_idx >= T => PAD.
        """
        delay_arr = torch.tensor(delay_pattern, dtype=torch.int32)

        t_idx_BxT = torch.broadcast_to(
            torch.arange(T, dtype=torch.int32)[None, :],
            [B, T],
        )
        t_idx_BxTx1 = t_idx_BxT[..., None]
        t_idx_BxTxC = t_idx_BxTx1 - delay_arr.view(1, 1, C)

        b_idx_BxTxC = torch.broadcast_to(
            torch.arange(B, dtype=torch.int32).view(B, 1, 1),
            [B, T, C],
        )
        c_idx_BxTxC = torch.broadcast_to(
            torch.arange(C, dtype=torch.int32).view(1, 1, C),
            [B, T, C],
        )

        # We must clamp time indices to [0..T-1] so gather_nd equivalent won't fail
        t_clamped_BxTxC = torch.clamp(t_idx_BxTxC, 0, T - 1)

        indices_BTCx3 = torch.stack(
            [
                b_idx_BxTxC.reshape(-1),
                t_clamped_BxTxC.reshape(-1),
                c_idx_BxTxC.reshape(-1),
            ],
            dim=1,
        ).long()  # Ensure indices are long type for indexing

        return t_idx_BxTxC, indices_BTCx3

    @staticmethod
    def apply_audio_delay(
        audio_BxTxC: torch.Tensor,
        pad_value: int,
        bos_value: int,
        precomp: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Applies the delay pattern to batched audio tokens using precomputed indices,
        inserting BOS where t_idx < 0 and PAD where t_idx >= T.

        Args:
            audio_BxTxC: [B, T, C] int16 audio tokens (or int32/float)
            pad_value: the padding token
            bos_value: the BOS token
            precomp:  (t_idx_BxTxC, indices_BTCx3) from build_delay_indices

        Returns:
            result_BxTxC: [B, T, C] delayed audio tokens
        """
        device = audio_BxTxC.device  # Get device from input tensor
        t_idx_BxTxC, indices_BTCx3 = precomp
        t_idx_BxTxC = t_idx_BxTxC.to(device)  # Move precomputed indices to device
        indices_BTCx3 = indices_BTCx3.to(device)

        # Equivalent of tf.gather_nd using advanced indexing
        # Ensure indices are long type if not already (build_delay_indices should handle this)
        gathered_flat = audio_BxTxC[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
        gathered_BxTxC = gathered_flat.view(audio_BxTxC.shape)

        # Create masks on the correct device
        mask_bos = t_idx_BxTxC < 0  # => place bos_value
        mask_pad = t_idx_BxTxC >= audio_BxTxC.shape[1]  # => place pad_value

        # Create scalar tensors on the correct device
        bos_tensor = torch.tensor(bos_value, dtype=audio_BxTxC.dtype, device=device)
        pad_tensor = torch.tensor(pad_value, dtype=audio_BxTxC.dtype, device=device)

        # If mask_bos, BOS; else if mask_pad, PAD; else original gather
        # All tensors should now be on the same device
        result_BxTxC = torch.where(mask_bos, bos_tensor, torch.where(mask_pad, pad_tensor, gathered_BxTxC))

        return result_BxTxC

    @staticmethod
    def build_revert_indices(B: int, T: int, C: int, delay_pattern: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute indices for the revert operation using PyTorch.

        Returns:
            A tuple (t_idx_BxTxC, indices_BTCx3) where:
                - t_idx_BxTxC is a tensor of shape [B, T, C] computed as time indices plus the delay.
                - indices_BTCx3 is a tensor of shape [B*T*C, 3] used for gathering, computed from:
                    batch indices, clamped time indices, and channel indices.
        """
        # Use default device unless specified otherwise; assumes inputs might define device later
        device = None  # Or determine dynamically if needed, e.g., from a model parameter

        delay_arr = torch.tensor(delay_pattern, dtype=torch.int32, device=device)

        t_idx_BT1 = torch.broadcast_to(torch.arange(T, device=device).unsqueeze(0), [B, T])
        t_idx_BT1 = t_idx_BT1.unsqueeze(-1)

        t_idx_BxTxC = torch.minimum(
            t_idx_BT1 + delay_arr.view(1, 1, C),
            torch.tensor(T - 1, device=device),
        )
        b_idx_BxTxC = torch.broadcast_to(torch.arange(B, device=device).view(B, 1, 1), [B, T, C])
        c_idx_BxTxC = torch.broadcast_to(torch.arange(C, device=device).view(1, 1, C), [B, T, C])

        indices_BTCx3 = torch.stack(
            [
                b_idx_BxTxC.reshape(-1),
                t_idx_BxTxC.reshape(-1),
                c_idx_BxTxC.reshape(-1),
            ],
            axis=1,
        ).long()  # Ensure indices are long type

        return t_idx_BxTxC, indices_BTCx3

    @staticmethod
    def revert_audio_delay(
        audio_BxTxC: torch.Tensor,
        pad_value: int,
        precomp: Tuple[torch.Tensor, torch.Tensor],
        T: int,
    ) -> torch.Tensor:
        """
        Reverts a delay pattern from batched audio tokens using precomputed indices (PyTorch version).

        Args:
            audio_BxTxC: Input delayed audio tensor
            pad_value: Padding value for out-of-bounds indices
            precomp: Precomputed revert indices tuple containing:
                - t_idx_BxTxC: Time offset indices tensor
                - indices_BTCx3: Gather indices tensor for original audio
            T: Original sequence length before padding

        Returns:
            Reverted audio tensor with same shape as input
        """
        t_idx_BxTxC, indices_BTCx3 = precomp
        device = audio_BxTxC.device  # Get device from input tensor

        # Move precomputed indices to the same device as audio_BxTxC if they aren't already
        t_idx_BxTxC = t_idx_BxTxC.to(device)
        indices_BTCx3 = indices_BTCx3.to(device)

        # Using PyTorch advanced indexing (equivalent to tf.gather_nd or np equivalent)
        gathered_flat = audio_BxTxC[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
        gathered_BxTxC = gathered_flat.view(audio_BxTxC.size())  # Use .size() for robust reshaping

        # Create pad_tensor on the correct device
        pad_tensor = torch.tensor(pad_value, dtype=audio_BxTxC.dtype, device=device)
        # Create T tensor on the correct device for comparison
        T_tensor = torch.tensor(T, device=device)

        result_BxTxC = torch.where(
            t_idx_BxTxC >= T_tensor, pad_tensor, gathered_BxTxC
        )  # Changed np.where to torch.where

        return result_BxTxC


__all__ = ["DiaProcessor"]
