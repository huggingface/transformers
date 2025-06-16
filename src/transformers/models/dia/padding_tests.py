# TODO: to be removed after everything


from typing import List, Optional, Tuple

import torch


def build_indices(
    bsz: int,
    seq_len: int,
    num_channels: int,
    delay_pattern: List[int],
    revert: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
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

def apply_audio_delay(
    audio: torch.Tensor,
    pad_token_id: int,
    bos_token_id: int,
    precomputed_idx: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
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

    delay_precomp = build_indices(
        bsz=bsz,
        seq_len=seq_len,
        num_channels=channels,
        delay_pattern=delay_pattern,
        revert=False,
    )

    delayed_batch = apply_audio_delay(
        audio=prefill,
        pad_token_id=-1,
        bos_token_id=bos_id,
        precomputed_idx=delay_precomp,
    )

    delayed_input = delayed_batch[:, : max(audio_padding_sizes) + 1, :]
    forced_output = delayed_batch[:, max(audio_padding_sizes) + 1 :, :]

    return delayed_batch, delayed_input, forced_output, audio_padding_sizes, (bsz, seq_len, channels)


def revert_delay(
    delayed_batch: torch.Tensor,
    shape: Tuple[int, int, int],
    delay_pattern: List[int],
    pad_id: int,
):
    bsz, seq_len, channels = shape

    revert_precomp = build_indices(
        bsz=bsz,
        seq_len=seq_len,
        num_channels=channels,
        delay_pattern=delay_pattern,
        revert=True,
    )
    codebook = apply_audio_delay(
        audio=delayed_batch,
        pad_token_id=pad_id,
        bos_token_id=-1,
        precomputed_idx=revert_precomp,
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

# revert
codebooks = revert_delay(delayed_batch, (bsz, seq_len, channels), delay_pattern, pad_id=audio_pad_id)

print(codebooks)
