# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from .utils.import_utils import is_torchdynamo_compiling


@dataclass
class AttentionMaskConverter:
    """
    A utility attention mask class that allows one to:
      - Create a causal 4d mask
      - Create a causal 4d mask with sliding window
      - Convert a 2d attention mask (batch_size, seq_len) to a 4d attention mask (batch_size, 1, tgt_seq_len, src_seq_len)
        that can be multiplied with attention scores.
    
    Example:
    ```python
    >>> import torch
    >>> from transformers.modeling_attn_mask_utils import AttentionMaskConverter
    >>> converter = AttentionMaskConverter(True)
    >>> converter.to_4d(torch.tensor([[0, 0, 0, 1, 1]]), 5, key_value_length=5, dtype=torch.float32)
    tensor([[[[-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
              [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
              [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
              [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00, -3.4028e+38],
              [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00,  0.0000e+00]]]])
    ```
    
    Parameters:
        is_causal (`bool`): Whether the attention mask should be causal (uni-directional).
        sliding_window (`int`, optional): If provided, creates a sliding window mask.
    """
    is_causal: bool
    sliding_window: int

    def __init__(self, is_causal: bool, sliding_window: Optional[int] = None):
        self.is_causal = is_causal
        self.sliding_window = sliding_window
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`"
            )

    def to_causal_4d(
        self,
        batch_size: int,
        query_length: int,
        key_value_length: int,
        dtype: torch.dtype,
        device: Union[torch.device, str] = "cpu",
    ) -> Optional[torch.Tensor]:
        if not self.is_causal:
            raise ValueError(f"Please use `to_causal_4d` only if {self.__class__} has `is_causal` set to True.")
        input_shape = (batch_size, query_length)
        past_key_values_length = key_value_length - query_length
        causal_4d_mask = None
        if input_shape[-1] > 1 or self.sliding_window is not None:
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )
        return causal_4d_mask

    def to_4d(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Converts a 2D attention mask to a 4D attention mask.
        If the provided 2D mask is "packed" (its width is a multiple of the individual sequence length), it
        will be restructured into a block-diagonal mask.
        If `is_causal` is True, a causal mask is applied block-wise.
        """
        batch_size = attention_mask_2d.size(0)
        total_length = attention_mask_2d.size(1)

        # --- Begin Packed Mask Handling ---
        if total_length != query_length and total_length % query_length == 0:
            num_blocks = total_length // query_length
            # Reshape the 2D mask: (batch_size, total_length) -> (batch_size, num_blocks, query_length)
            reshaped_mask = attention_mask_2d.view(batch_size, num_blocks, query_length)
            # Build a block-diagonal mask: each block corresponds to one sample's sequence.
            block_diag_mask = torch.zeros((batch_size, total_length), dtype=reshaped_mask.dtype, device=reshaped_mask.device)
            for b in range(batch_size):
                for i in range(num_blocks):
                    start = i * query_length
                    end = start + query_length
                    block_diag_mask[b, start:end] = reshaped_mask[b, i].clone()
            expanded_attn_mask = self._expand_mask(block_diag_mask, dtype, tgt_len=total_length).to(attention_mask_2d.device)
        else:
            expanded_attn_mask = self._expand_mask(attention_mask_2d, dtype, tgt_len=query_length).to(attention_mask_2d.device)
        # --- End Packed Mask Handling ---

        # --- Apply Causal Mask if Needed ---
        causal_4d_mask = None
        if (attention_mask_2d.size(1) > 1 or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                )
            past_key_values_length = key_value_length - query_length
            causal_4d_mask = self._make_causal_mask(
                (batch_size, query_length),
                dtype,
                device=attention_mask_2d.device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )
            if total_length != query_length:
                # For a packed mask, apply the causal mask to each block individually.
                fixed_mask = torch.full(
                    (batch_size, 1, total_length, total_length),
                    torch.finfo(dtype).min,
                    device=attention_mask_2d.device,
                    dtype=dtype,
                )
                num_blocks = total_length // query_length
                for b in range(batch_size):
                    for i in range(num_blocks):
                        start = i * query_length
                        end = start + query_length
                        # Extract only the last `query_length` columns from the causal mask,
                        # which gives a (query_length x query_length) mask for the block.
                        block_causal = causal_4d_mask[b:b+1, :, :, -query_length:]
                        fixed_mask[b, :, start:end, start:end] = block_causal
                expanded_attn_mask = fixed_mask.masked_fill(expanded_attn_mask.bool(), torch.finfo(dtype).min)
            else:
                expanded_attn_mask = causal_4d_mask.masked_fill(expanded_attn_mask.bool(), torch.finfo(dtype).min)
        # --- End Causal Mask Application ---

        return expanded_attn_mask

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        sliding_window: Optional[int] = None,
    ):
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)
        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window - 1
            context_mask = torch.tril(torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal)
            if is_torchdynamo_compiling():
                mask = mask.clone()
            mask.masked_fill_(context_mask, torch.finfo(dtype).min)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        inverted_mask = 1.0 - expanded_mask
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    @staticmethod
    def _unmask_unattended(
        expanded_mask: torch.FloatTensor,
        min_dtype: float,
    ):
        if expanded_mask.dtype == torch.bool:
            raise ValueError(
                "AttentionMaskConverter._unmask_unattended expects a float `expanded_mask`, got a BoolTensor."
            )
        return expanded_mask.mul(~torch.all(expanded_mask == min_dtype, dim=-1, keepdim=True))

    @staticmethod
    def _ignore_causal_mask_sdpa(
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
        sliding_window: Optional[int] = None,
        is_training: bool = False,
    ) -> bool:
        _, query_length = inputs_embeds.shape[0], inputs_embeds.shape[1]
        key_value_length = query_length + past_key_values_length
        is_tracing = torch.jit.is_tracing() or isinstance(inputs_embeds, torch.fx.Proxy) or is_torchdynamo_compiling()
        ignore_causal_mask = False
        if attention_mask is None:
            if (
                (is_training or not is_tracing)
                and (query_length == 1 or key_value_length == query_length)
                and (sliding_window is None or key_value_length < sliding_window)
            ):
                ignore_causal_mask = True
        elif sliding_window is None or key_value_length < sliding_window:
            if len(attention_mask.shape) == 4:
                return False
            elif not is_tracing and torch.all(attention_mask == 1):
                if query_length == 1 or key_value_length == query_length:
                    ignore_causal_mask = True
        return ignore_causal_mask


def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)
    key_value_length = input_shape[-1] + past_key_values_length
    if attention_mask is not None and len(attention_mask.shape) == 2:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype
        )
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        else:
            inverted_mask = 1.0 - attention_mask
            attention_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
            )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
    return attention_mask


def _prepare_4d_causal_attention_mask_for_sdpa(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)
    key_value_length = input_shape[-1] + past_key_values_length
    is_tracing = torch.jit.is_tracing() or isinstance(inputs_embeds, torch.fx.Proxy) or is_torchdynamo_compiling()
    ignore_causal_mask = AttentionMaskConverter._ignore_causal_mask_sdpa(
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        past_key_values_length=past_key_values_length,
        sliding_window=sliding_window,
    )
    if ignore_causal_mask:
        expanded_4d_mask = None
    elif attention_mask is None:
        expanded_4d_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
    else:
        if attention_mask.dim() == 4:
            expanded_4d_mask = attention_mask
        else:
            expanded_4d_mask = attn_mask_converter.to_4d(
                attention_mask,
                input_shape[-1],
                dtype=inputs_embeds.dtype,
                key_value_length=key_value_length,
            )
        if not is_tracing and expanded_4d_mask.device.type == "cuda":
            expanded_4d_mask = AttentionMaskConverter._unmask_unattended(
                expanded_4d_mask, min_dtype=torch.finfo(inputs_embeds.dtype).min
            )
    return expanded_4d_mask


def _prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _prepare_4d_attention_mask_for_sdpa(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    _, key_value_length = mask.shape
    tgt_len = tgt_len if tgt_len is not None else key_value_length
    is_tracing = torch.jit.is_tracing() or isinstance(mask, torch.fx.Proxy) or is_torchdynamo_compiling()
    if not is_tracing and torch.all(mask == 1):
        return None
    else:
        return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _create_4d_causal_attention_mask(
    input_shape: Union[torch.Size, Tuple, List],
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: Optional[int] = None,
) -> Optional[torch.Tensor]:
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)
    key_value_length = past_key_values_length + input_shape[-1]
    attention_mask = attn_mask_converter.to_causal_4d(
        input_shape[0], input_shape[-1], key_value_length, dtype=dtype, device=device
    )
    return attention_mask
