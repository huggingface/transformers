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
"""
Speech processor class for Dia
"""

import typing

import torch

from ..auto import AutoModel
from ...processing_utils import ProcessorMixin


def build_delay_indices(
    B: int, T: int, C: int, delay_pattern: typing.List[int]
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
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


def apply_audio_delay(
    audio_BxTxC: torch.Tensor,
    pad_value: int,
    bos_value: int,
    precomp: typing.Tuple[torch.Tensor, torch.Tensor],
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


def build_revert_indices(
    B: int, T: int, C: int, delay_pattern: typing.List[int]
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
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


def revert_audio_delay(
    audio_BxTxC: torch.Tensor,
    pad_value: int,
    precomp: typing.Tuple[torch.Tensor, torch.Tensor],
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

    result_BxTxC = torch.where(t_idx_BxTxC >= T_tensor, pad_tensor, gathered_BxTxC)  # Changed np.where to torch.where

    return result_BxTxC


class DiaProcessor(ProcessorMixin):
    r"""
    Constructs a Dia processor which wraps a Dia feature extractor and a Dia tokenizer into a single
    processor.

    [`DiaProcessor`] offers all the functionalities of [`DiaAudioProcessor`] and [`DiaTokenizer`]. See
    the [`~DiaProcessor.__call__`] and [`~DiaProcessor.decode`] for more information.

    Args:
        feature_extractor (`DiaAudioProcessor`):
            An instance of [`DiaAudioProcessor`]. The feature extractor is a required input.
        tokenizer (`DiaTokenizer`):
            An instance of [`DiaTokenizer`]. The tokenizer is a required input.
    """

    feature_extractor_class = "DacFeatureExtractor"
    tokenizer_class = "DiaTokenizer"

    def __init__(self, feature_extractor, tokenizer, audio_model=None):
        super().__init__(feature_extractor, tokenizer)
        self.audio_tokenizer = AutoModel.from_pretrained(audio_model)

    def _prepare_audio_prompt(self, audio_prompt: torch.Tensor | None) -> tuple[torch.Tensor, int]:
        num_channels = self.channels
        audio_bos_value = self.audio_bos_value
        audio_pad_value = self.audio_pad_value
        delay_pattern = self.delay_pattern
        max_delay_pattern = max(delay_pattern)

        prefill = torch.full(
            (1, num_channels),
            fill_value=audio_bos_value,
            dtype=torch.int,
            device=self.device,
        )

        prefill_step = 1

        if audio_prompt is not None:
            prefill_step += audio_prompt.shape[0]
            prefill = torch.cat([prefill, audio_prompt], dim=0)

        delay_pad_tensor = torch.full(
            (max_delay_pattern, num_channels), fill_value=-1, dtype=torch.int, device=self.device
        )
        prefill = torch.cat([prefill, delay_pad_tensor], dim=0)

        delay_precomp = build_delay_indices(
            B=1,
            T=prefill.shape[0],
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        prefill = apply_audio_delay(
            audio_BxTxC=prefill.unsqueeze(0),
            pad_value=audio_pad_value,
            bos_value=audio_bos_value,
            precomp=delay_precomp,
        ).squeeze(0)

        return prefill, prefill_step

    def __call__(self, *args, **kwargs):
        """
        Forwards the `audio` argument to DiaAudioProcessor's [`~DiaAudioProcessor.__call__`] and the `text`
        argument to [`~DiaTokenizer.__call__`]. Please refer to the docstring of the above two methods for more
        information.
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
            input_audio = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
            audio_tokens = self.audio_tokenizer.encode(input_audio["input_values"], **kwargs)
            inputs = {
                "input_values": input_audio["input_values"],
                "attention_mask": input_audio["attention_mask"],
                "audio_tokens": audio_tokens["input_ids"],
            }

        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs

        elif audio is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to DiaTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.audio_tokenizer.decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to DiaTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.audio_tokenizer.decode(*args, **kwargs)


__all__ = ["DiaProcessor"]
