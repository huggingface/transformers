import copy
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch

from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_6

from .configuration_utils import PretrainedConfig
from .utils import (
    is_hqq_available,
    is_quanto_greater,
    is_torch_greater_or_equal,
    is_torchdynamo_compiling,
    logging,
)


if _is_quanto_greater_than_0_2_5 := is_quanto_greater("0.2.5", accept_dev=True):
    from optimum.quanto import MaxOptimizer, qint2, qint4, quantize_weight

if is_hqq_available():
    from hqq.core.quantize import Quantizer as HQQQuantizer

_is_torch_greater_or_equal_than_2_7 = is_torch_greater_or_equal("2.7", accept_dev=True)


logger = logging.get_logger(__name__)


class CacheLayerMixin(ABC):
    """Base, abstract class for a single layer's cache."""

    is_compileable = False

    def __init__(self):
        self.keys, self.values = None, None

    @abstractmethod
    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: Optional[dict[str, Any]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def lazy_initialization(self, key_states: torch.Tensor): ...

    @abstractmethod
    def get_seq_length(self, cache_position=None) -> int: ...

    @abstractmethod
    def get_max_cache_shape(self) -> int: ...

    @abstractmethod
    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]: ...

    def offload(self):
        """Offload this layer's data to CPU device."""
        if self.keys is not None:
            self.keys = self.keys.to("cpu", non_blocking=True)
            self.values = self.values.to("cpu", non_blocking=True)

    def prefetch(self):
        """In case of layer offloading, this allows to move the data back to the layer's device ahead of time."""
        if self.keys is not None and self.keys.device != self.device:
            self.keys = self.keys.to(self.device, non_blocking=True)
            self.values = self.values.to(self.device, non_blocking=True)

    def reset(self) -> None:
        """Resets the cache values while preserving the objects"""
        if self.keys is not None:
            self.keys.zero_()
            self.values.zero_()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reorders this layer's cache for beam search."""
        if self.keys.numel():
            device = self.keys.device
            self.keys = self.keys.index_select(0, beam_idx.to(device))
        if self.values.numel():
            device = self.values.device
            self.values = self.values.index_select(0, beam_idx.to(device))


class DynamicLayer(CacheLayerMixin):
    """
    A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the Key and Value states as tensors with shape `[batch_size, num_heads, seq_len, head_dim]`.

    See `CacheLayerMixin` for details on common methods that are implemented by all cache layers.
    """

    is_sliding = False

    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicLayer`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Lazy initialization
        if self.keys is None:
            self.lazy_initialization(key_states)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values

    def get_seq_length(self, cache_position=None) -> int:
        """Returns the sequence length of the cached states."""
        if self.keys is None or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length of the cache object. DynamicLayer does not have a maximum length."""
        return -1

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorders the cache for beam search, given the selected beam indices."""
        if self.keys is not None and self.keys.numel():
            self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
            self.values = self.values.index_select(0, beam_idx.to(self.values.device))

    def crop(self, max_length: int) -> None:
        """
        Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens.
        """
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        if self.keys is not None and self.keys.numel():
            self.keys = self.keys[..., :max_length, :]
            self.values = self.values[..., :max_length, :]

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat the cache `repeats` times in the batch dimension."""
        if self.keys is not None and self.keys.numel():
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Only keep the `indices` in the batch dimension of the cache."""
        if self.keys is not None and self.keys.numel():
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the mask"""
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    @classmethod
    def from_tensors(cls, keys: torch.Tensor, values: torch.Tensor) -> "DynamicLayer":
        """
        Build a `DynamicLayer` instance from pre-existing key/value tensors.

        Args:
            keys (`torch.Tensor`):
                Key cache tensor of shape ``[batch_size, num_heads, seq_len, head_dim]``.
            values (`torch.Tensor`):
                Value cache tensor of shape ``[batch_size, num_heads, seq_len, head_dim]``.

        Returns:
            `DynamicLayer`: The newly constructed layer whose internal cache directly references
            the supplied tensors.
        """
        layer = cls()
        layer.dtype, layer.device = keys.dtype, keys.device
        layer.keys = keys
        layer.values = values
        return layer


class StaticLayer(CacheLayerMixin):
    """
    A static cache layer that stores the Key and Value states as static tensors with shape `[batch_size, num_heads, seq_len, head_dim]`.
    It allocates its full backing tensors up-front and mutates them in-place. Built for `torch.compile` support.

    See `CacheLayerMixin` for details on common methods that are implemented by all cache layers.
    """

    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int):
        """
        Args:
            max_cache_len (`int`):
                Maximum number of tokens that can be stored, used for tensor preallocation.
        """
        super().__init__()
        self.max_cache_len = max_cache_len

    def lazy_initialization(self, key_states: torch.Tensor):
        """
        Lazy initialization of the keys and values tensors. This allows to get all properties (dtype, device,
        num_heads in case of TP etc...) at runtime directly, which is extremely practical as it avoids moving
        devices, dtypes etc later on for each `update` (which could break the static dynamo addresses as well).

        If this is unwanted, one can call `early_initialization(...)` on the Cache directly, which will call this
        function ahead-of-time (this is required for `torch.export` for example). Note that for `compile`, as we
        internally don't compile the prefill, this is guaranteed to have been called already when compiling.
        If compiling the prefill as well, e.g. calling `model.compile(...)` before `generate` with a static cache,
        it is still supported in general, but without guarantees depending on the compilation options (e.g. cuda graphs,
        i.e. `mode="reduce-overhead"` is known to fail). But it will in general work correctly, and prefill should
        not be compiled anyway for performances!
        """
        self.max_batch_size, self.num_heads, _, self.head_dim = key_states.shape
        self.dtype, self.device = key_states.dtype, key_states.device

        self.keys = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        # Note: `mark_static_address` is used to tag the cache as a fixed data pointer, preventing compiled graph
        # breaks when updating the cache. However, it is not supported when tracing the graph, so we skip it in this case.
        # As prefill should never be compiled, this is not an issue and it will still be run (except when users compile
        # prefill explicitly, but this should be avoided!)
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the static cache tensors in place.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The updated key and value states.
        """
        # Lazy initialization
        if self.keys is None:
            self.lazy_initialization(key_states)

        # Some old models give None for `cache_position` or even omit passing `cache_kwargs` when used as cross-attention,
        # in which case we should copy the whole Layer (key_states.shape[-2] == self.max_cache_len)
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        cache_position = (
            cache_position if cache_position is not None else torch.arange(key_states.shape[-2], device=self.device)
        )

        # Update the cache
        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            # Fallback for devices like MPS where index_copy_ might not be supported.
            self.keys[:, :, cache_position] = key_states
            self.values[:, :, cache_position] = value_states
        return self.keys, self.values

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.max_cache_len

    def get_seq_length(self, cache_position=None) -> int:
        """Returns the sequence length of the cached states."""
        if cache_position is not None:
            return int(cache_position[-1] + 1)
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        seq_length = (self.keys[0, 0].any(dim=-1)).sum() if self.keys is not None else 0
        return seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorders the cache for beam search, given the selected beam indices."""
        dev = self.keys.device
        beam_idx_dev = beam_idx.to(dev)
        self.keys = self.keys.index_select(0, beam_idx_dev)
        self.values = self.values.index_select(0, beam_idx_dev)

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        kv_offset = 0
        kv_length = self.max_cache_len
        return kv_length, kv_offset


class SlidingWindowLayer(StaticLayer):
    """
    A static cache layer that implements sliding window attention caching.

    See `CacheLayerMixin` for details on common methods that are implemented by all cache layers.
    """

    is_sliding = True

    def __init__(self, max_cache_len: int, sliding_window: int):
        """
        Args:
            max_cache_len (`int`):
                Maximum number of tokens that can be stored, used for tensor preallocation.
            sliding_window (`int`):
                The size of the sliding window.
        """
        effective_max_cache_len = min(sliding_window, max_cache_len)
        super().__init__(max_cache_len=effective_max_cache_len)
        self.cumulative_length = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the sliding window cache tensors in place.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The updated key and value states.
        """
        # Lazy initialization
        if self.keys is None:
            self.lazy_initialization(key_states)

        cache_position = cache_kwargs.get("cache_position")

        is_full = self.cumulative_length >= self.max_cache_len
        # Update it now that we saved the value above
        self.cumulative_length += key_states.shape[-2]

        # Handle prefill phase when prompt length > sliding_window_size.
        # Note that we store cropped key/value states in the cache but return the full key/value states.
        if cache_position.shape[0] > self.max_cache_len:
            self.keys.copy_(key_states[:, :, -self.max_cache_len :, :])
            self.values.copy_(value_states[:, :, -self.max_cache_len :, :])
            # Return the full states here
            return key_states, value_states

        # Here we only assume decoding stage, i.e. 1 token at a time
        if is_full:
            # Roll all values to the left by 1 position
            new_keys = self.keys.roll(-1, dims=-2)
            new_values = self.values.roll(-1, dims=-2)
            # Overwrite the last position with new states
            # (note: very important to use a tensor to index here, see https://github.com/pytorch/pytorch/issues/159855)
            index = torch.tensor([-1], dtype=int, device=self.device)
            new_keys[:, :, index] = key_states
            new_values[:, :, index] = value_states

            # Copy back into `self` (do not just assign again) in order to keep the static dynamo address
            self.keys.copy_(new_keys)
            self.values.copy_(new_values)
        else:
            try:
                self.keys.index_copy_(2, cache_position, key_states)
                self.values.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                self.keys[:, :, cache_position] = key_states
                self.values[:, :, cache_position] = value_states

        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        query_length = cache_position.shape[0]
        first_cache_position = cache_position[0]

        kv_offset = torch.clamp(first_cache_position - self.max_cache_len + 1, min=0)
        # This is not general (see HybridChunkedCache for the whole general case), but it's what the cache returns
        kv_length = max(query_length, self.max_cache_len)
        return kv_length, kv_offset

    def reset(self) -> None:
        super().reset()
        self.cumulative_length = 0

    def get_seq_length(self, cache_position=None) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length


class ChunkedSlidingLayer(SlidingWindowLayer):
    """
    An extended SlidingWindowLayer that supports prefill chunking, originally implemented for Llama 4.

    See `SlidingWindowLayer` for details on common methods that are implemented by all cache layers.
    """

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Lazy initialization
        if self.keys is None:
            self.lazy_initialization(key_states)

        cache_position = cache_kwargs.get("cache_position")

        cumulative_length = self.cumulative_length
        is_full = cumulative_length >= self.max_cache_len
        # Update it now that we saved the value above
        self.cumulative_length += key_states.shape[-2]

        if is_full:
            full_key_states = torch.cat((self.keys[:, :, 1:, :], key_states), dim=-2)
            full_value_states = torch.cat((self.values[:, :, 1:, :], value_states), dim=-2)
            # Fast decoding path -> here as the effective size is still sliding window, it is extremely important
            # to return `self.key_cache[layer_idx]` and `self.value_cache[layer_idx]`, as they have the fixed address
            # in memory (the values are the same as the full states, but not the address!!)
            if key_states.shape[-2] == 1:
                self.keys.copy_(full_key_states)
                self.values.copy_(full_value_states)
                return self.keys, self.values
        elif not is_full and cumulative_length + key_states.shape[2] > self.max_cache_len:
            # Fast prefill path, no need to cat() in this case, as the cache is currently empty
            if cumulative_length == 0:
                full_key_states = key_states
                full_value_states = value_states
            else:
                full_key_states = torch.cat((self.keys[:, :, :cumulative_length, :], key_states), dim=-2)
                full_value_states = torch.cat((self.values[:, :, :cumulative_length, :], value_states), dim=-2)
        else:
            try:
                self.keys.index_copy_(2, cache_position, key_states)
                self.values.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                self.keys[:, :, cache_position] = key_states
                self.values[:, :, cache_position] = value_states
            return self.keys, self.values

        self.keys.copy_(full_key_states[:, :, -self.max_cache_len :, :])
        self.values.copy_(full_value_states[:, :, -self.max_cache_len :, :])
        # we should return the whole states instead of `self.keys/values` here, as otherwise we lose some context
        # which is outside the window
        return full_key_states, full_value_states

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        query_length = cache_position.shape[0]
        first_cache_position = cache_position[0]
        sliding_window = self.max_cache_len

        kv_offset = torch.clamp(first_cache_position - sliding_window + 1, min=0)
        # This is the true general case for any Cache using local attention (sliding or chunked)
        if first_cache_position >= sliding_window:
            # Here the Cache is already full
            kv_length = sliding_window + query_length - 1
        elif first_cache_position < sliding_window and first_cache_position + query_length > sliding_window:
            # Here the Cache becomes full with the new input
            kv_length = first_cache_position + query_length
        else:
            # Here the Cache is still smaller than the local size, but we return the local size as it's static
            kv_length = sliding_window
        return kv_length, kv_offset


class QuantizedLayer(DynamicLayer):
    """
    A quantized layer similar to what is described in the [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://huggingface.co/papers/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for Key and Value cache by
    applying quantization.

    The cache has two types of storage, one for original precision and one for the quantized cache. A `residual length`
    is set as a maximum capacity for the original precision cache. When the length goes beyond maximum capacity, the original
    precision cache is discarded and moved into the quantized cache. The quantization is done per-channel with a set `q_group_size`
    for both Keys and Values, in contrast to what was described in the paper.
    """

    def __init__(
        self,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        super().__init__(self)
        self.nbits = nbits
        self.axis_key = axis_key
        self.axis_value = axis_value
        self.q_group_size = q_group_size
        self.residual_length = residual_length
        self.cumulative_length = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicLayer`.

        Return:
            A tuple containing the updated key and value states.
        """
        self.cumulative_length += key_states.shape[-2]

        # Lazy initialization
        if self.keys is None:
            self.lazy_initialization(key_states)
            self._quantized_keys = self._quantize(key_states.contiguous(), axis=self.axis_key)
            self._quantized_values = self._quantize(value_states.contiguous(), axis=self.axis_value)
            return key_states, value_states

        dequant_keys = self._dequantize(self._quantized_keys)
        dequant_values = self._dequantize(self._quantized_values)
        keys_to_return = torch.cat([dequant_keys, self.keys, key_states], dim=-2)
        values_to_return = torch.cat([dequant_values, self.values, value_states], dim=-2)
        if self.keys.dim() == 4 and self.keys.shape[-2] + 1 >= self.residual_length:
            self._quantized_keys = self._quantize(keys_to_return.contiguous(), axis=self.axis_key)
            self._quantized_values = self._quantize(values_to_return.contiguous(), axis=self.axis_value)
            self.keys = torch.tensor([], dtype=key_states.dtype, device=key_states.device)
            self.values = torch.tensor([], dtype=key_states.dtype, device=key_states.device)
        else:
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)

        return keys_to_return, values_to_return

    def get_seq_length(self, cache_position=None) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length

    @abstractmethod
    def _quantize(self, tensor, axis): ...

    @abstractmethod
    def _dequantize(self, q_tensor): ...


class QuantoQuantizedLayer(QuantizedLayer):
    def __init__(
        self,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        super().__init__(
            nbits=nbits,
            axis_key=axis_key,
            axis_value=axis_value,
            q_group_size=q_group_size,
            residual_length=residual_length,
        )

        if not _is_quanto_greater_than_0_2_5:
            raise ImportError(
                "You need optimum-quanto package version to be greater or equal than 0.2.5 to use `QuantoQuantizedCache`. "
                "Detected version {optimum_quanto_version}."
            )

        if self.nbits not in [2, 4]:
            raise ValueError(f"`nbits` for `quanto` backend has to be one of [`2`, `4`] but got {self.nbits}")

        if self.axis_key not in [0, -1]:
            raise ValueError(f"`axis_key` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_key}")

        if self.axis_value not in [0, -1]:
            raise ValueError(
                f"`axis_value` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_value}"
            )

        self.qtype = qint4 if self.nbits == 4 else qint2
        self.optimizer = MaxOptimizer()  # hardcode as it's the only one for per-channel quantization

    def _quantize(self, tensor, axis):
        scale, zeropoint = self.optimizer(tensor, self.qtype, axis, self.q_group_size)
        qtensor = quantize_weight(tensor, self.qtype, axis, scale, zeropoint, self.q_group_size)
        return qtensor

    def _dequantize(self, qtensor):
        return qtensor.dequantize()


class HQQQuantizedLayer(QuantizedLayer):
    def __init__(
        self,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        super().__init__(
            nbits=nbits,
            axis_key=axis_key,
            axis_value=axis_value,
            q_group_size=q_group_size,
            residual_length=residual_length,
        )

        if not is_hqq_available():
            raise ImportError("You need to install `hqq` to use `HQQQuantizedLayer`")

        if self.nbits not in [1, 2, 3, 4, 8]:
            raise ValueError(
                f"`nbits` for `HQQ` backend has to be one of [`1`, `2`, `3`, `4`, `8`] but got {self.nbits}"
            )

        if self.axis_key not in [0, 1]:
            raise ValueError(f"`axis_key` for `HQQ` backend has to be one of [`0`, `1`] but got {self.axis_key}")

        if self.axis_value not in [0, 1]:
            raise ValueError(f"`axis_value` for `HQQ` backend has to be one of [`0`, `1`] but got {self.axis_value}")

        self.quantizer = HQQQuantizer

    def _quantize(self, tensor, axis):
        qtensor, meta = self.quantizer.quantize(
            tensor,
            axis=axis,
            device=self.keys.device,
            compute_dtype=self.keys.dtype,
            nbits=self.nbits,
            group_size=self.q_group_size,
        )
        meta["compute_dtype"] = self.keys.dtype
        self.quantizer.cuda(qtensor, meta=meta, device=self.keys.device)  # Move to device and cast to dtype
        meta["scale"] = meta["scale"].to(qtensor.device)
        meta["zero"] = meta["zero"].to(qtensor.device)
        return qtensor, meta

    def _dequantize(self, qtensor):
        quant_tensor, meta = qtensor
        tensor = self.quantizer.dequantize(quant_tensor, meta)
        return tensor


LAYER_CLASS_MAP: dict[str, type[CacheLayerMixin]] = {
    "full_attention": StaticLayer,
    "sliding_attention": SlidingWindowLayer,
    "chunked_attention": ChunkedSlidingLayer,
}


class KeyValuesWrapper:
    """Helper class for Cache that simulates layer-indexed key/value lists from a layered cache.
    This allows for BC access and writing, e.g., cache.key_cache[idx] = ...
    Deprecated in favor of Cache.layers[idx].keys/values. TODO: remove in v4.56.0"""

    def __init__(self, layers, cache_type="keys"):
        self.layers = layers
        self.cache_type = cache_type

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [getattr(layer, self.cache_type) for layer in self.layers[idx]]
        return getattr(self.layers[idx], self.cache_type)

    def __setitem__(self, idx, value):
        if isinstance(idx, slice):
            for layer, val in zip(self.layers[idx], value):
                setattr(layer, self.cache_type, val)
        else:
            setattr(self.layers[idx], self.cache_type, value)

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        for layer in self.layers:
            yield getattr(layer, self.cache_type)

    def __bool__(self):
        return bool(self.layers)


class Cache:
    """
    A `Cache` is mostly a list of `CacheLayerMixin` objects, one per model layer. It serves as a container for
    the Cache of each layer.

    Parameters:
        layers (`Optional`, *optional*):
            A list of pre-created `CacheLayerMixin`. If omitted (`None`), then `layer_class_to_replicate` will
            be used.
        layer_class_to_replicate (`type[CacheLayerMixin]`, *optional*):
            Only used if `layers` is omitted (`None`), in which case it will be used as the base class for each layer,
            and the layers will be added lazily as soon as `update` is called with a `layer_idx` greater than the current
            list of layers.
        offloading (`bool`, *optional*, defaults to `False`):
            Whether to perform offloading of the layers to `cpu`, to save GPU memory.
        offload_only_non_sliding (`bool`, *optional*, defaults to `True`):
            If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
            usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).
    """

    def __init__(
        self,
        layers: Optional[list[CacheLayerMixin]] = None,
        layer_class_to_replicate: Optional[type[CacheLayerMixin]] = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
    ):
        if layers is not None and layer_class_to_replicate is not None:
            raise ValueError(
                "You can construct a Cache either from a list `layers` of all the predefined `CacheLayer`, or from a "
                "`layer_class_to_replicate`, in which case the Cache will append a new layer corresponding to "
                "`layer_class_to_replicate` for each new call to `update` with an idx not already in the Cache."
            )
        if layers is None and layer_class_to_replicate is None:
            raise ValueError(
                "You should provide exactly one of `layers` or `layer_class_to_replicate` to initialize a Cache."
            )
        self.layers = layers if layers is not None else []
        self.layer_class_to_replicate = layer_class_to_replicate
        self.offloading = offloading
        if self.offloading:
            self.only_non_sliding = offload_only_non_sliding
            self.prefetch_stream = torch.Stream() if _is_torch_greater_or_equal_than_2_7 else torch.cuda.Stream()

    def __repr__(self):
        return f"{self.__class__.__name__}(layers={self.layers})"

    def prefetch(self, layer_idx: int, only_non_sliding: bool = True):
        """
        Prefetch a given layer on its device. If `only_non_sliding` is True, it will try to prefetch only the layers
        which are non-sliding. If the `layer_idx` is outside the range, this will circle back to the first layers.
        Note that we use a non-default stream for this, to avoid blocking.
        """
        if only_non_sliding:
            # Try to find next non-sliding, starting at `layer_idx`
            try:
                layer_idx = layer_idx + self.is_sliding[layer_idx:].index(False)
            # In this case, we need to circle back to the begining
            except ValueError:
                layer_idx = self.is_sliding.index(False)
        else:
            layer_idx = layer_idx if layer_idx < len(self.layers) else 0

        # Prefetch
        with self.prefetch_stream if _is_torch_greater_or_equal_than_2_7 else torch.cuda.stream(self.prefetch_stream):
            self.layers[layer_idx].prefetch()

    def offload(self, layer_idx: int, only_non_sliding: bool = True):
        """
        Offload a given `layer_idx`. If `only_non_sliding` is True, it will offload `layer_idx` only if it is a
        non-sliding layer. Note that we do it on the default stream, so that we ensure all earlier
        computation in the layer's `update` methods are finished.
        """
        if not (only_non_sliding and self.is_sliding[layer_idx]):
            self.layers[layer_idx].offload()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        # In this case, the `layers` were not provided, and we must append as much as `layer_idx`
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        if self.offloading:
            # Wait for the stream to finish if needed, and start prefetching the next layer
            torch.cuda.default_stream(key_states.device).wait_stream(self.prefetch_stream)
            self.prefetch(layer_idx + 1, self.only_non_sliding)

        keys, values = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

        if self.offloading:
            self.offload(layer_idx, self.only_non_sliding)

        return keys, values

    def early_initialization(
        self, batch_size: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device
    ):
        """
        Initialize all the layers in advance (it's otherwise lazily initialized on the first `update` call).
        This is useful for our `export` recipes, as `export` needs everything in advance.
        """
        # Note that the initialization needs all dimensions (except -2), as well as device and dtype, so we use
        # this fake tensor approach. It has size 0 on the -2 dimension, so it does not allocate any data (it only
        # creates an empty tensor with correct shape, dtype and device), which is very efficient and practical
        fake_keys_tensor = torch.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype, device=device)
        # Init all layers
        for layer in self.layers:
            layer.lazy_initialization(fake_keys_tensor)

    def get_seq_length(self, layer_idx: int = 0, cache_position=None) -> int:
        """Returns the sequence length of the cache for the given layer."""
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length(cache_position)

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
        """
        # For DynamicCache, where the layers are created at runtime -> if it was not yet created, the size is
        # simply the shape of `cache_position`
        if layer_idx >= len(self.layers):
            return cache_position.shape[0], 0
        return self.layers[layer_idx].get_mask_sizes(cache_position)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Returns maximum sequence length of the cache object. Dynamic caches do not have a maximum length."""
        # For DynamicCache, where the layers are created at runtime -> if it was not yet created, return -1
        # as DynamicLayer does
        if layer_idx >= len(self.layers):
            return -1
        return self.layers[layer_idx].get_max_cache_shape()

    def reset(self):
        """Recursively reset all layers tensors"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].reset()

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder the cache for beam search"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].reorder_cache(beam_idx)

    def crop(self, max_length: int):
        """Crop the cache to the given length"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].crop(max_length)

    def batch_repeat_interleave(self, repeats: int):
        """Repeat and interleave the cache"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        """Select indices from the cache"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].batch_select_indices(indices)

    @property
    def max_batch_size(self) -> int:
        """Return the maximum batch size of the cache"""
        values = [layer.max_batch_size for layer in self.layers]
        if len(set(values)) > 1:
            raise ValueError(f"Max batch size is not consistent across layers: {values}")
        return values[0]

    @property
    def max_cache_len(self) -> int:
        """Return the maximum cache length of the cache"""
        values = [layer.max_cache_len for layer in self.layers]
        return max(values)

    @property
    def is_compileable(self) -> bool:
        """Return whether the cache is compileable"""
        # For DynamicCache dispatching the layers lazily (otherwise, all([]) is True)
        if len(self.layers) == 0:
            return False
        return all(layer.is_compileable for layer in self.layers)

    @property
    def is_sliding(self) -> list[bool]:
        """Return whether the layers of the cache are sliding window"""
        return [getattr(layer, "is_sliding", False) for layer in self.layers]

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Support for backwards-compatible `past_key_values` indexing, e.g. `past_key_values[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self.layers):
            return self.layers[layer_idx].keys, self.layers[layer_idx].values
        else:
            raise KeyError(
                f"Cache only has {len(self.layers)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_values` iteration, e.g. `for x in past_key_values:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.layers[layer_idx].keys, self.layers[layer_idx].values)

    def __len__(self):
        """
        This value corresponds to the number of layers in the model.
        """
        # Note: for DynamicCache, layers are initialized lazily, so this will not be accurate before the first
        # forward through all the layers
        return len(self.layers)

    @property
    def key_cache(self) -> KeyValuesWrapper:
        """List-like object of key cache tensors indexed by layer. Deprecated in favor of `cache.layers[idx].keys`"""
        logger.warning_once(
            "`cache.key_cache[idx]` is deprecated and will be removed in v4.56.0. Use `cache.layers[idx].keys` instead."
        )
        return KeyValuesWrapper(self.layers, "keys")

    @property
    def value_cache(self) -> KeyValuesWrapper:
        """List-like object of value cache tensors indexed by layer. Deprecated in favor of `cache.layers[idx].values`"""
        logger.warning_once(
            "`cache.value_cache[idx]` is deprecated and will be removed in v4.56.0. Use `cache.layers[idx].values` instead."
        )
        return KeyValuesWrapper(self.layers, "values")


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = DynamicCache()
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        DynamicCache()
        ```
    """

    # Specialized constructor for DDP cache data, needed for BC
    def __init__(self, ddp_cache_data: Optional[Iterable[tuple[torch.Tensor, torch.Tensor]]] = None):
        # `ddp_cache_data` was originally added for compatibility with `torch.distributed` (DDP). See #36212
        # and #36373 for more information. In a nutshell, it is `map(gather_map, zip(*caches))`, i.e. each item in the
        # iterable contains the key and value states for a layer gathered across replicas by torch.distributed
        # (shape=[global batch size, num_heads, seq_len, head_dim]).
        if ddp_cache_data is not None:
            layers = []
            for key_states, value_states in ddp_cache_data:
                layers.append(DynamicLayer.from_tensors(key_states, value_states))
            super().__init__(layers=layers)
        else:
            super().__init__(layer_class_to_replicate=DynamicLayer)

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """
        Converts the `Cache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility.
        """
        legacy_cache = ()
        for layer in self.layers:
            legacy_cache += ((layer.keys, layer.values),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: tuple[tuple[torch.FloatTensor, torch.FloatTensor], ...]) -> "Cache":
        """
        Converts a cache in the legacy cache format into an equivalent `Cache`. Used for
        backward compatibility.
        """
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache


# Utilities for `DynamicCache` <> torch.export support

if is_torch_greater_or_equal("2.3"):

    def _get_cache_dict(cache: DynamicCache):
        if any(not isinstance(layer, DynamicLayer) for layer in cache.layers):
            raise RuntimeError("This pytree flattening function should only be applied to DynamicCache")

        if not is_torch_greater_or_equal_than_2_6:
            logger.warning_once(
                "DynamicCache + torch.export is tested on torch 2.6.0+ and may not work on earlier versions."
            )

        return {
            "key_cache": [layer.keys for layer in cache.layers if layer.keys is not None],
            "value_cache": [layer.values for layer in cache.layers if layer.values is not None],
        }

    def _unflatten_dynamic_cache(
        values,
        context: torch.utils._pytree.Context,
    ):
        dictionary = torch.utils._pytree._dict_unflatten(values, context)
        cache = DynamicCache()
        # Reconstruct layers from keys and values lists
        key_list = dictionary.get("key_cache", [])
        value_list = dictionary.get("value_cache", [])
        for idx in range(max(len(key_list), len(value_list))):
            key = key_list[idx] if idx < len(key_list) else None
            value = value_list[idx] if idx < len(value_list) else None
            cache.update(key, value, idx)
        return cache

    torch.utils._pytree.register_pytree_node(
        DynamicCache,
        lambda dynamic_cache: torch.utils._pytree._dict_flatten(_get_cache_dict(dynamic_cache)),
        _unflatten_dynamic_cache,
        serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
        flatten_with_keys_fn=lambda dynamic_cache: torch.utils._pytree._dict_flatten_with_keys(
            _get_cache_dict(dynamic_cache)
        ),
    )
    # TODO (tmanlaibaatar) This won't be needed in torch 2.7.
    torch.fx._pytree.register_pytree_flatten_spec(
        DynamicCache, lambda cache, spec: torch.fx._pytree._dict_flatten_spec(_get_cache_dict(cache), spec)
    )


class OffloadedCache(Cache):
    """
    A drop-in replacement for DynamicCache that conserves accelerator (GPU, XPU) memory at the expense of more CPU memory.
    Useful for generating from models with very long context.

    See `Cache` for details on common methods that are implemented by all cache classes.
    """

    def __init__(self) -> None:
        super().__init__(layer_class_to_replicate=DynamicLayer, offloading=True)


class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)` and `torch.export()`.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

        >>> inputs = tokenizer(text="My name is Llama", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = StaticCache(max_cache_len=max_generated_length, config=model.config)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        StaticCache()
        ```
    """

    # Pass-in kwargs as well to avoid crashing for BC (it used more arguments before)
    def __init__(self, max_cache_len: int, config: PretrainedConfig, **kwargs):
        layers = [StaticLayer(max_cache_len) for _ in range(config.num_hidden_layers)]
        super().__init__(layers=layers)


class OffloadedStaticCache(Cache):
    """
    A drop-in replacement for StaticCache that conserves accelerator memory by offloading
    cache tensors to CPU when not actively being used.

    This cache maintains the compilation-friendly properties of StaticCache while enabling
    much longer sequences by offloading inactive layers to CPU memory.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:
        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, OffloadedStaticCache

        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

        >>> inputs = tokenizer(text="My name is GPT2", return_tensors="pt")

        >>> # Prepare a cache class with offloading
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = OffloadedStaticCache(max_cache_len=max_generated_length, config=model.config)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache with offloaded layers
        OffloadedStaticCache()
        ```
    """

    # Pass-in kwargs as well to avoid crashing for BC (it used more arguments before)
    def __init__(self, max_cache_len: int, config: PretrainedConfig, **kwargs):
        layers = [StaticLayer(max_cache_len) for _ in range(config.num_hidden_layers)]
        super().__init__(layers=layers, offloading=True)


class SlidingWindowCache(Cache):
    """
    Sliding Window Cache class to be used with `torch.compile` for models like Mistral that support sliding window attention.
    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, SlidingWindowCache

        >>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

        >>> inputs = tokenizer(text="My name is Mistral", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = SlidingWindowCache(max_cache_len=max_generated_length, config=model.config)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        SlidingWindowCache()
        ```
    """

    # Pass-in kwargs as well to avoid crashing for BC (it used more arguments before)
    def __init__(self, max_cache_len: int, config: PretrainedConfig, **kwargs):
        layers = [SlidingWindowLayer(max_cache_len, config.sliding_window) for _ in range(config.num_hidden_layers)]
        super().__init__(layers=layers)


class HybridCache(Cache):
    """
    Hybrid Cache class to be used with `torch.compile` for models that alternate between a local sliding window
    attention and global attention in every other layer (originally implemented for Gemma2).
    Under the hood, Hybrid Cache leverages ["SlidingWindowLayer"] for sliding window attention and ["StaticLayer"]
    for global attention. For more information, see the documentation of those layer types.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, HybridCache

        >>> model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

        >>> inputs = tokenizer(text="My name is Gemma", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = HybridCache(max_cache_len=max_generated_length, config=model.config)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        HybridCache()
        ```
    """

    # Pass-in kwargs as well to avoid crashing for BC (it used more arguments before)
    def __init__(self, max_cache_len: int, config: PretrainedConfig, **kwargs):
        if hasattr(config, "layer_types"):
            layers = []
            for layer_type in config.layer_types:
                init_kwargs = {"max_cache_len": max_cache_len}
                if layer_type == "sliding_attention":
                    init_kwargs["sliding_window"] = config.sliding_window
                elif layer_type == "chunked_attention":
                    init_kwargs["sliding_window"] = config.attention_chunk_size
                layers.append(LAYER_CLASS_MAP[layer_type](**init_kwargs))
        else:
            # In this case, fall back to StaticCache
            layers = [StaticLayer(max_cache_len) for _ in range(config.num_hidden_layers)]
        super().__init__(layers=layers)


# The mapping already handles dispatching the correct layers in Hybrid, this is only used for BC
class HybridChunkedCache(HybridCache): ...


class OffloadedHybridCache(Cache):
    """
    A drop-in replacement for HybridChunkedCache that conserves accelerator memory by offloading
    cache tensors to CPU when not actively being used.

    This cache maintains the compilation-friendly properties of HybridChunkedCache while enabling
    much longer sequences by offloading inactive layers to CPU memory.

    See `Cache` for details on common methods that are implemented by all cache classes.
    """

    # Pass-in kwargs as well to avoid crashing for BC (it used more arguments before)
    def __init__(self, max_cache_len: int, config: PretrainedConfig, **kwargs):
        if hasattr(config, "layer_types"):
            layers = []
            for layer_type in config.layer_types:
                init_kwargs = {"max_cache_len": max_cache_len}
                if layer_type == "sliding_attention":
                    init_kwargs["sliding_window"] = config.sliding_window
                elif layer_type == "chunked_attention":
                    init_kwargs["sliding_window"] = config.attention_chunk_size
                layers.append(LAYER_CLASS_MAP[layer_type](**init_kwargs))
        else:
            # In this case, fall back to StaticCache
            layers = [StaticLayer(max_cache_len) for _ in range(config.num_hidden_layers)]
        super().__init__(layers=layers, offloading=True)


class QuantizedCache(Cache):
    """
    A quantizer cache similar to what is described in the
    [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://arxiv.org/abs/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for keys and values
    by applying quantization.
    The cache has two types of storage, one for original precision and one for the
    quantized cache. A `residual length` is set as a maximum capacity for the original precision cache. When the
    length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache.
    The quantization is done per-channel with a set `q_group_size` for both keys and values, in contrast to what was
    described in the paper.

    See `Cache` for details on common methods that are implemented by all cache classes.
    """

    def __init__(
        self,
        backend: str,
        config: PretrainedConfig,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        if backend == "quanto":
            layer_class = QuantoQuantizedLayer
        elif backend == "hqq":
            layer_class = HQQQuantizedLayer
        else:
            raise ValueError(f"Unknown quantization backend `{backend}`")

        layers = [
            layer_class(nbits, axis_key, axis_value, q_group_size, residual_length)
            for _ in range(config.num_hidden_layers)
        ]
        super().__init__(layers=layers)


class QuantoQuantizedCache(QuantizedCache):
    """
    A quantizer cache similar to what is described in the
    [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://arxiv.org/abs/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for keys and values
    by applying quantization.
    The cache has two types of storage, one for original precision and one for the
    quantized cache. A `residual length` is set as a maximum capacity for the original precision cache. When the
    length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache.
    The quantization is done per-channel with a set `q_group_size` for both keys and values, in contrast to what was
    described in the paper.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

        ```python
        >>> # Run pip install quanto first if you don't have it yet
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoQuantizedCache

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = QuantoQuantizedCache(config=model.config, nbits=4)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        QuantoQuantizedCache()
        ```
    """

    def __init__(
        self,
        config: PretrainedConfig,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        super().__init__("quanto", config, nbits, axis_key, axis_value, q_group_size, residual_length)


class HQQQuantizedCache(QuantizedCache):
    """
    A quantizer cache similar to what is described in the
    [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://arxiv.org/abs/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for keys and values
    by applying quantization.
    The cache has two types of storage, one for original precision and one for the
    quantized cache. A `residual length` is set as a maximum capacity for the original precision cache. When the
    length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache.
    The quantization is done per-channel with a set `q_group_size` for both keys and values, in contrast to what was
    described in the paper.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

        ```python
        >>> # Run pip install hqq first if you don't have it yet
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, HQQQuantizedCache

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = HQQQuantizedCache(config=model.config, nbits=4, axis_key=1, axis_value=1)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        HQQQuantizedCache()
        ```
    """

    def __init__(
        self,
        config: PretrainedConfig,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        super().__init__("hqq", config, nbits, axis_key, axis_value, q_group_size, residual_length)


class EncoderDecoderCache(Cache):
    """
    Base, abstract class for all encoder-decoder caches. Can be used to hold combinations of self-attention and
    cross-attention caches.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForCausalLM, DynamicCache, EncoderDecoderCache

        >>> model = AutoModelForCausalLM.from_pretrained("openai/whisper-small")
        >>> processor = AutoProcessor.from_pretrained("openai/whisper-small")

        >>> inputs = processor(audio=YOUR-AUDIO, return_tensors="pt")

        >>> # Prepare cache classes for encoder and decoder and pass it to model's forward
        >>> self_attention_cache = DynamicCache()
        >>> cross_attention_cache = DynamicCache()
        >>> past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        EncoderDecoderCache()
        ```
    """

    # Override @property from Cache -> this will be set in __init__ on the instances
    is_compileable = False

    def __init__(self, self_attention_cache: Cache, cross_attention_cache: Cache):
        self.self_attention_cache = self_attention_cache
        self.cross_attention_cache = cross_attention_cache
        # Override @property from Cache
        self.is_compileable = getattr(self.self_attention_cache, "is_compileable", False)

        self.is_updated = {}
        for layer_idx in range(len(cross_attention_cache)):
            self.is_updated[layer_idx] = bool(cross_attention_cache.get_seq_length(layer_idx) > 0)

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_values` iteration, e.g. `for x in past_key_values:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (
                self.self_attention_cache.layers[layer_idx].keys,
                self.self_attention_cache.layers[layer_idx].values,
                self.cross_attention_cache.layers[layer_idx].keys,
                self.cross_attention_cache.layers[layer_idx].values,
            )

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Support for backwards-compatible `past_key_values` indexing, e.g. `past_key_values[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (
                self.self_attention_cache.layers[layer_idx].keys,
                self.self_attention_cache.layers[layer_idx].values,
                self.cross_attention_cache.layers[layer_idx].keys,
                self.cross_attention_cache.layers[layer_idx].values,
            )
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __len__(self):
        """
        Support for backwards-compatible `past_key_values` length, e.g. `len(past_key_values)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.self_attention_cache)

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor]]:
        """Converts the `EncoderDecoderCache` instance into its equivalent in the legacy cache format."""
        legacy_cache = ()
        if len(self.cross_attention_cache) > 0:
            for self_attn, cross_attn in zip(
                self.self_attention_cache.to_legacy_cache(), self.cross_attention_cache.to_legacy_cache()
            ):
                legacy_cache += (self_attn + cross_attn,)
        else:
            legacy_cache = self.self_attention_cache.to_legacy_cache()
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: tuple[tuple[torch.FloatTensor, torch.FloatTensor], ...]
    ) -> "EncoderDecoderCache":
        """Converts a cache in the legacy cache format into an equivalent `EncoderDecoderCache`."""
        cache = cls(
            self_attention_cache=DynamicCache(),
            cross_attention_cache=DynamicCache(),
        )
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx][:2]
                cache.self_attention_cache.update(key_states, value_states, layer_idx)
                if len(past_key_values[layer_idx]) > 2:
                    key_states, value_states = past_key_values[layer_idx][2:]
                    cache.cross_attention_cache.update(key_states, value_states, layer_idx)
                    cache.is_updated[layer_idx] = True
        return cache

    def get_seq_length(self, layer_idx: Optional[int] = 0, cache_position=None) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # check if empty list because in case of static cache it will be a tensors and we can't check `if not torch.Tensor`
        return self.self_attention_cache.get_seq_length(layer_idx, cache_position)

    def reset(self):
        if hasattr(self.self_attention_cache, "reset"):
            self.self_attention_cache.reset()
        if hasattr(self.cross_attention_cache, "reset"):
            self.cross_attention_cache.reset()
        elif not hasattr(self.self_attention_cache, "reset") and not hasattr(self.cross_attention_cache, "reset"):
            raise ValueError(
                "Neither self nor cross-attention cache have valid `.reset()` methods. `.reset()` should "
                "only be called on compatible cache classes, such as `StaticCache` or `SlidingWindowCache`. "
                f"Got {self.self_attention_cache.__str__()} for the self attention cache and "
                f"{self.cross_attention_cache.__str__()} for the cross attention cache."
            )
        for layer_idx in self.is_updated:
            self.is_updated[layer_idx] = False

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        self.self_attention_cache.reorder_cache(beam_idx)
        self.cross_attention_cache.reorder_cache(beam_idx)

    def check_dynamic_cache(self, method: str):
        if not (
            isinstance(self.self_attention_cache, DynamicCache)
            and isinstance(self.cross_attention_cache, DynamicCache)
        ):
            raise ValueError(
                f"`{method}` is only defined for dynamic cache, got {self.self_attention_cache.__str__()} for the self "
                f"attention cache and {self.cross_attention_cache.__str__()} for the cross attention cache."
            )

    # TODO(gante, sanchit-gandhi): move following functionality into `.generate`
    def crop(self, maximum_length: int):
        """
        Crop the past key values up to a new `maximum_length` in terms of tokens. `maximum_length` can also be
        negative to remove `maximum_length` tokens. This is used in assisted decoding and contrastive search.
        """
        self.check_dynamic_cache(self.crop.__name__)
        self.self_attention_cache.crop(maximum_length)

    def batch_split(self, full_batch_size: int, split_size: int) -> "list[EncoderDecoderCache]":
        """
        Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`
        """
        self.check_dynamic_cache(self.batch_split.__name__)
        self_attention_cache = self.self_attention_cache.batch_split(full_batch_size, split_size)
        cross_attention_cache = self.cross_attention_cache.batch_split(full_batch_size, split_size)

        out = []
        for self_attn, cross_attn in zip(self_attention_cache, cross_attention_cache):
            out.append(EncoderDecoderCache(self_attn, cross_attn))
        return out

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        self.check_dynamic_cache(self.batch_repeat_interleave.__name__)
        self.self_attention_cache.batch_repeat_interleave(repeats)
        self.cross_attention_cache.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        self.check_dynamic_cache(self.batch_select_indices.__name__)
        self.self_attention_cache.batch_select_indices(indices)
        self.cross_attention_cache.batch_select_indices(indices)

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object"""
        return self.self_attention_cache.get_max_cache_shape()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        return self.self_attention_cache.get_mask_sizes(cache_position, layer_idx)


### Deprecated classes


class SinkCache(Cache):
    """
    Is its now a `custom_generate` repository on the Hub: https://huggingface.co/transformers-community/sink_cache.
    See [these docs](https://huggingface.co/docs/transformers/generation_strategies#custom-decoding-methods) for
    general `custom_generate`usage.
    """

    # TODO (joao, manuel): Remove this class in v4.59.0
    def __init__(self, **kwargs) -> None:
        raise NotImplementedError(
            "`SinkCache` has been moved as a `custom_generate` repository on the Hub: "
            "https://huggingface.co/transformers-community/sink_cache. See the repository for usage examples."
        )


@dataclass
class CacheConfig:
    """
    Base class for cache configs. Deprecated in favor of a simpler dictionary.
    """

    cache_implementation: None

    def __post_init__(self):
        logger.warning_once(
            "CacheConfig is deprecated and will be removed in v4.55.0 in favor of a simpler dictionary."
        )

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Constructs a CacheConfig instance from a dictionary of parameters.
        Args:
            config_dict (dict[str, Any]): Dictionary containing configuration parameters.
            **kwargs: Additional keyword arguments to override dictionary values.

        Returns:
            CacheConfig: Instance of CacheConfig constructed from the dictionary.
        """
        logger.warning_once(
            "CacheConfig is deprecated and will be removed in v4.55.0 in favor of a simpler dictionary."
        )
        config = cls(**config_dict)
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        return config

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.to_json_file
    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default
                `QuantizationConfig()` is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            writer.write(json_string)

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.to_dict
    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return copy.deepcopy(self.__dict__)

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.__iter__
    def __iter__(self):
        """allows `dict(obj)` for situations where obj may be a dict or QuantizationConfigMixin"""
        for attr, value in copy.deepcopy(self.__dict__).items():
            yield attr, value

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.__repr__
    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self):
        """
        Serializes this instance to a JSON formatted string.
        Returns:
            str: JSON formatted string representing the configuration instance.
        """
        return json.dumps(self.__dict__, indent=2) + "\n"

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.update
    def update(self, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
        returning all the unused kwargs.

        Args:
            kwargs (`dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # Remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs


@dataclass
class QuantizedCacheConfig(CacheConfig):
    """
    Configuration class for quantized cache settings. Deprecated in favor of a simpler dictionary.

    Attributes:
        backend (`str`, *optional*, defaults to `"quanto"`):
            Backend to use when performing quantization, Can be one of [`quanto`, `HQQ`]
        nbits (`Optional[int]`, *optional*, defaults to 4):
            Number of bits, can be 2 or 4 for the `quanto` backend and one of [1, 2, 3, 4, 8] for the `HQQ` backend. Defaults to 2.
        axis_key (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the key tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        axis_value (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the value tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        q_group_size (`Optional[int]`, *optional*, defaults to 64):
            Size of the quantization group, should be a divisor of the model's hidden dimension.
            Defaults to 64.
        residual_length (`Optional[int]`, *optional*, defaults to 128):
            Length of the residual cache which will always be stored in original precision.
            Defaults to 128.
        compute_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            The default dtype used for computations in the model. Keys and Values will be cast to this dtype after dequantization.
        device (`str`, *optional*, defaults to `"cpu"`):
            Device on which to perform computations, should be same as the model's device.
    """

    def __init__(
        self,
        backend: str = "quanto",
        nbits: Optional[int] = 4,
        axis_key: Optional[int] = 0,
        axis_value: Optional[int] = 0,
        q_group_size: Optional[int] = 64,
        residual_length: Optional[int] = 128,
        compute_dtype: Optional[torch.dtype] = torch.float16,
        device: Optional[str] = "cpu",
    ):
        logger.warning_once(
            "CacheConfig is deprecated and will be removed in v4.55.0 in favor of a simpler dictionary."
        )
        self.backend = backend
        self.nbits = nbits
        self.axis_key = axis_key
        self.axis_value = axis_value
        self.q_group_size = q_group_size
        self.residual_length = residual_length
        self.compute_dtype = compute_dtype
        self.device = device

    def validate(self):
        """Validates if the arguments passed are correct"""

        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        # Check that the values are reasonable in general (nbits, axis)
        # Later in QuantizedCache init we check if they are supported for that particular backend
        if self.nbits not in [1, 2, 3, 4, 8]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="nbits",
                    correct_value="2 or 4 or 8",
                    found_value=self.nbits,
                ),
            )
        if self.q_group_size <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="q_group_size",
                    correct_value="a positive integer",
                    found_value=self.q_group_size,
                ),
            )
        if self.residual_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="residual_length",
                    correct_value="a positive integer",
                    found_value=self.residual_length,
                ),
            )

        if self.axis_key not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_key",
                    correct_value="`1` or `0`, `-1`",
                    found_value=self.axis_key,
                ),
            )

        if self.axis_value not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_value",
                    correct_value="`1` or `0` or `-1`",
                    found_value=self.axis_value,
                ),
            )


@dataclass
class StaticCacheConfig(CacheConfig):
    """
    Configuration class for static cache settings.
    """

    cache_implementation = "static"

    def __init__(self, batch_size: int, max_cache_len: int, device="cpu"):
        logger.warning_once(
            "CacheConfig is deprecated and will be removed in v4.55.0 in favor of a simpler dictionary."
        )
        self.batch_size = batch_size
        self.max_cache_len = max_cache_len
        self.device = device


# TODO (manuel, joao): remove this class, it is here only for backwards compatibility
# PEP 562: Lazy loading for deprecated location of MambaCache
def __getattr__(name: str) -> Any:
    if name == "MambaCache":
        logger.warning_once(
            "Importing `MambaCache` from `transformers.cache_utils` is deprecated and will be removed "
            "in a future version. Please import it from `transformers` or `transformers.models.mamba.cache_mamba` instead."
        )

        class MambaCache:
            """
            Importing `MambaCache` from `transformers.cache_utils` is deprecated and will be removed
            in a future version. Please import it from `transformers` or `transformers.models.mamba.cache_mamba` instead.

            Cache for mamba model which does not have attention mechanism and key value states.

            Arguments:
                config (`PretrainedConfig):
                    The configuration file defining the shape-related attributes required to initialize the static cache.
                max_batch_size (`int`):
                    The maximum batch size with which the model will be used. Note that a new instance must be instantiated if a smaller batch size is used.
                dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
                    The default `dtype` to use when initializing the layer.
                device (`torch.device` or `str`, *optional*):
                    The device on which the cache should be initialized. Should be the same as the layer.

            Example:

                ```python
                >>> from transformers import AutoTokenizer, MambaForCausalLM, MambaCache

                >>> model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
                >>> tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")

                >>> inputs = tokenizer(text="My name is Mamba", return_tensors="pt")

                >>> # Prepare a cache class and pass it to model's forward
                >>> past_key_values = MambaCache(config=model.config, max_batch_size=1, device=model.device, dtype=model.dtype)
                >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
                >>> outputs.past_key_values
                MambaCache()
                ```
            """

            is_compileable = True

            # TODO (joao): add layer_device_map arg and update code in `generate` accordingly
            def __init__(
                self,
                config,
                max_batch_size: int,
                dtype: torch.dtype = torch.float16,
                device: Union[torch.device, str, None] = None,
            ):
                self.max_batch_size = max_batch_size
                self._dtype = dtype
                self.intermediate_size = config.intermediate_size
                self.ssm_state_size = config.state_size
                self.conv_kernel_size = config.conv_kernel

                self.conv_states: list[torch.Tensor] = []
                self.ssm_states: list[torch.Tensor] = []
                device = torch.device(device) if device is not None else None
                for _ in range(config.num_hidden_layers):
                    conv_state: torch.Tensor = torch.zeros(
                        self.max_batch_size,
                        self.intermediate_size,
                        self.conv_kernel_size,
                        device=device,
                        dtype=self._dtype,
                    )
                    ssm_state: torch.Tensor = torch.zeros(
                        self.max_batch_size,
                        self.intermediate_size,
                        self.ssm_state_size,
                        device=device,
                        dtype=self._dtype,
                    )

                    torch._dynamo.mark_static_address(conv_state)
                    torch._dynamo.mark_static_address(ssm_state)
                    self.conv_states.append(conv_state)
                    self.ssm_states.append(ssm_state)

            def update_conv_state(
                self, layer_idx: int, new_conv_state: torch.Tensor, cache_position: torch.LongTensor
            ) -> torch.Tensor:
                # This `if` blocks is only reached in multigpu and if `layer_device_map` is not passed. It is used
                # when the cache is initialized in the forward pass (e.g. Mamba)
                if self.conv_states[layer_idx].device != new_conv_state.device:
                    self.conv_states[layer_idx] = self.conv_states[layer_idx].to(new_conv_state.device)

                conv_state = self.conv_states[layer_idx]
                cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)

                conv_state = conv_state.roll(shifts=-1, dims=-1)
                conv_state[:, :, cache_position] = new_conv_state.to(device=conv_state.device, dtype=conv_state.dtype)
                self.conv_states[layer_idx].zero_()
                self.conv_states[layer_idx] += conv_state
                return self.conv_states[layer_idx]

            def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
                self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states[layer_idx].device)
                return self.ssm_states[layer_idx]

            def reset(self):
                for layer_idx in range(len(self.conv_states)):
                    # In-place ops prevent breaking the static address
                    self.conv_states[layer_idx].zero_()
                    self.ssm_states[layer_idx].zero_()

        return MambaCache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
