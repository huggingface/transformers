from typing import List, Optional, Tuple

import torch


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

    result_BxTxC = torch.where(t_idx_BxTxC >= T_tensor, pad_tensor, gathered_BxTxC)  # Changed np.where to torch.where

    return result_BxTxC


def prepare_audio(audios: List[Optional[torch.Tensor]], delay_pattern: List[int], batch_size: Optional[int] = None):
    if audios is None and batch_size is None:
        raise ValueError(
            "To process audio in Dia, we need either a batch of processed audios or an associated batch size based on the text input."
        )

    audios = [None] * batch_size if audios is None else audios

    # TODO: audios might be the batched output of dac --> overwrite padding?
    audio_lens = [a.shape[0] if a is not None else 0 for a in audios]
    max_audio_len = max(audio_lens)
    audio_padding_sizes = [max_audio_len - audio_len for audio_len in audio_lens]

    max_delay = max(delay_pattern)
    batch_size = len(audios)
    # +1 for bos
    max_len = max_audio_len + max_delay + 1
    num_channels = len(delay_pattern)

    return audios, audio_padding_sizes, (batch_size, max_len, num_channels)


def prefill_audios(
    audios: List[Optional[torch.Tensor]], bos_id: int, delay_pattern: List[int], batch_size: Optional[int] = None
):
    audios, audio_padding_sizes, (bsz, seq_len, channels) = prepare_audio(audios, delay_pattern, batch_size)

    # dummy values
    prefill = torch.full(
        (bsz, seq_len, channels),
        fill_value=-1,
        dtype=torch.int,
    )

    # write values with appropriate padding
    for i in range(bsz):
        padding_size = audio_padding_sizes[i]
        prefill[i, : padding_size + 1, :] = bos_id

        prompt = audios[i]
        if prompt is not None:
            prompt = prompt.to(dtype=torch.int)
            prefill[i, padding_size + 1 : padding_size + 1 + prompt.shape[0], :] = prompt

    delay_precomp = build_delay_indices(
        B=bsz,
        T=seq_len,
        C=channels,
        delay_pattern=delay_pattern,
    )

    delayed_batch = apply_audio_delay(
        audio_BxTxC=prefill,
        pad_value=-1,
        bos_value=bos_id,
        precomp=delay_precomp,
    )

    delayed_input = delayed_batch[:, : max(audio_padding_sizes) + 1, :]
    forced_output = delayed_batch[:, max(audio_padding_sizes) + 1 :, :]

    return delayed_batch, delayed_input, forced_output, audio_padding_sizes, (bsz, seq_len, channels)


class HangoverLogitsProcessor:
    def __init__(
        self, delay_pattern: List[int], forced_output: torch.Tensor, batch_size: int, channels: int, vocab_size: int
    ):
        delay_tensor = torch.tensor(delay_pattern, dtype=torch.int32)[None, :].expand(batch_size, -1)
        self.cannot_predict_counter = delay_tensor.clone()

        self.batch_size = batch_size
        self.channels = channels
        self.vocab_size = vocab_size

        self.batch_idx = torch.arange(self.batch_size)[:, None].expand(-1, self.channels)
        self.seq_idx = torch.arange(self.channels)[None, :].expand(self.batch_size, -1)

        self.force_counter = 0
        # TODO: remove clamp on upper bound - just for my dummy thing
        self.forced_output = torch.clamp(forced_output, 0, 4)

    def forward(self, scores: torch.Tensor):
        # base condition that generation got out of the forced input
        if self.force_counter >= self.forced_output.shape[1]:
            return scores

        # force logits to be -inf except for the predetermined chosen vocab
        all_inf_tensor = (
            (torch.ones(self.vocab_size) * torch.finfo(scores.dtype).min)[None, None, :]
            .expand(self.batch_size, self.channels, -1)
            .clone()
        )
        current_forced_output = self.forced_output[:, self.force_counter, :]
        all_inf_tensor[self.batch_idx, self.seq_idx, current_forced_output] = 0.0

        # overwrite where we determine it otherwise we allow the base scores
        modifying_scores_idx = torch.where(self.cannot_predict_counter > 0, 1, 0).bool()
        scores = torch.where(modifying_scores_idx[..., None], all_inf_tensor, scores)

        # internal counters to keep track of what we force
        self.cannot_predict_counter -= 1
        self.force_counter += 1

        return scores


def revert_delay(
    delayed_batch: torch.Tensor,
    shape: Tuple[int, int, int],
    delay_pattern: List[int],
    pad_id: int,
):
    bsz, seq_len, channels = shape

    revert_precomp = build_revert_indices(
        B=bsz,
        T=seq_len,
        C=channels,
        delay_pattern=delay_pattern,
    )
    codebook = revert_audio_delay(
        audio_BxTxC=delayed_batch,
        pad_value=pad_id,
        precomp=revert_precomp,
        T=seq_len,
    )[:, : -max(delay_pattern), :]

    min_valid_index = 0
    max_valid_index = 1023
    invalid_mask = (codebook < min_valid_index) | (codebook > max_valid_index)
    codebook[invalid_mask] = 0

    return codebook


audios = [torch.arange(2 * 9).view(2, 9), torch.arange(4 * 9).view(4, 9)]
delay_pattern = [0, 8, 9, 10, 11, 12, 13, 14, 15]
audio_bos_id = 1026
audio_pad_id = 1025

delayed_batch, delayed_input, forced_output, audio_padding_sizes, (bsz, seq_len, channels) = prefill_audios(
    audios, audio_bos_id, delay_pattern
)


bos = 2
vocab = 5
channels = 9

processor = HangoverLogitsProcessor(delay_pattern, forced_output, bsz, channels, vocab)
predictions = torch.arange(channels * bsz * vocab).view(bsz, channels, vocab).half()
prediction_1 = processor.forward(predictions)
prediction_2 = processor.forward(predictions)


# revert
codebooks = revert_delay(delayed_batch, (bsz, seq_len, channels), delay_pattern, pad_id=audio_pad_id)

print(codebooks)
