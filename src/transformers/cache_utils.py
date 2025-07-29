import copy
import functools
import importlib.metadata
import inspect
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
from packaging import version

from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_6

from .configuration_utils import PretrainedConfig
from .utils import is_hqq_available, is_optimum_quanto_available, is_torch_greater_or_equal, logging


if is_hqq_available():
    from hqq.core.quantize import Quantizer as HQQQuantizer


logger = logging.get_logger(__name__)


class CacheLayerMixin(ABC):
    """Base, abstract class for a single layer's cache."""

    is_compileable = False

    def __init__(self):
        self.keys, self.values = None, None

    @abstractmethod
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def get_seq_length(self, cache_position=None) -> int: ...

    @abstractmethod
    def get_max_cache_shape(self) -> int: ...

    @abstractmethod
    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]: ...

    def reset(self) -> None:
        """Resets the cache values while preserving the objects"""
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
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
        else:
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

    def __init__(
        self,
        max_cache_len: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        sliding_window: Optional[int] = None,
    ):
        """
        Args:
            max_cache_len (`int`):
                Maximum number of tokens that can be stored, used for tensor preallocation.
            batch_size (`int`):
                Maximum batch size the cache is pre-allocated for.
            num_heads (`int`):
                Number of attention heads.
            head_dim (`int`):
                Per-head hidden dimension.
            dtype (`torch.dtype`, defaults to `torch.float32`):
                Data type of the cache tensors.
            device (`str` or `torch.device`, defaults to `"cpu"`):
                Device on which the cache tensors will be materialised.

        Notes:
            Static layers allocate their full backing tensors up-front and mutate them
            in-place. See the documentation of `Cache` for shared helper methods that
            operate uniformly across all layer types.
        """
        self.max_cache_len = max_cache_len
        self.max_batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self.keys = torch.zeros(
            (batch_size, num_heads, self.max_cache_len, head_dim),
            dtype=dtype,
            device=device,
        )
        self.values = torch.zeros(
            (batch_size, num_heads, self.max_cache_len, head_dim),
            dtype=dtype,
            device=device,
        )
        # Note: `mark_static_address` is used to tag the cache as a fixed data pointer,
        # preventing compiled graph breaks when updating the cache.
        torch._dynamo.mark_static_address(self.keys)
        torch._dynamo.mark_static_address(self.values)

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.max_cache_len

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
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs else None
        key_states = key_states.to(self.keys.dtype)
        value_states = value_states.to(self.values.dtype)

        # This may be needed if the Layer was not created with the right device in the beginning, i.e. if it did not respect
        # the device_map. However, even if it is the case, this will only run once, because then the new states received
        # will always have the same device
        if self.device != key_states.device:
            self.device = key_states.device
            self.keys = self.keys.to(self.device)
            self.values = self.values.to(self.device)

        if cache_position is None:
            # Prefill phase where seq_len potentially equals max_cache_len. Directly copy.
            self.keys.copy_(key_states)
            self.values.copy_(value_states)
        else:
            # Generation phase. Update specific positions.
            # Use index_copy_ for in-place update (compile-friendly).
            try:
                self.keys.index_copy_(2, cache_position, key_states)
                self.values.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                # Fallback for devices like MPS where index_copy_ might not be supported.
                self.keys[:, :, cache_position] = key_states
                self.values[:, :, cache_position] = value_states
        return self.keys, self.values

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

    def __init__(self, sliding_window, *args, **kwargs):
        """
        Args:
            sliding_window (`int`):
                Effective window size: number of tokens that are kept on each update call.
        """
        max_cache_len = kwargs.pop("max_cache_len", None)
        max_cache_len = min(sliding_window, max_cache_len) if max_cache_len is not None else sliding_window
        super().__init__(*args, max_cache_len=max_cache_len, *args, **kwargs)

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
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs else None
        if cache_position is None:
            raise ValueError("`cache_position` must be provided for SlidingWindowLayer.")

        # This may be needed if the Layer was not created with the right device in the beginning, i.e. if it did not respect
        # the device_map. However, even if it is the case, this will only run once, because then the new states received
        # will always have the same device
        if self.device != key_states.device:
            self.device = key_states.device
            self.keys = self.keys.to(self.device)
            self.values = self.values.to(self.device)

        key_states = key_states.to(self.keys.dtype)
        value_states = value_states.to(self.values.dtype)

        # Handle prefill phase when prompt length > sliding_window_size.
        # Note that we store cropped key/value states in the cache but return the full key/value states.
        if cache_position.shape[0] > self.max_cache_len:
            new_k = key_states[:, :, -self.max_cache_len :, :]
            new_v = value_states[:, :, -self.max_cache_len :, :]
            self.keys.copy_(new_k)
            self.values.copy_(new_v)
            return key_states, value_states

        # Sliding window logic for generation phase or prefill < window
        slicing = torch.arange(self.max_cache_len, device=self.device)
        current_seq_len = cache_position[-1] + 1  # Use last position to determine current length
        to_shift = current_seq_len > self.max_cache_len
        indices = (slicing + to_shift.sum()) % self.max_cache_len

        k_out_shifted = self.keys[:, :, indices]
        v_out_shifted = self.values[:, :, indices]

        # Clamp cache_position to determine the *target index* within the shifted cache view
        update_position = cache_position.clamp(min=0, max=self.max_cache_len - 1)

        try:
            k_out_updated = k_out_shifted.index_copy(2, update_position, key_states)
            v_out_updated = v_out_shifted.index_copy(2, update_position, value_states)
        except NotImplementedError:
            # Fallback for MPS: clone and modify the clone
            k_out_updated = k_out_shifted.clone()
            v_out_updated = v_out_shifted.clone()
            k_out_updated[:, :, update_position] = key_states
            v_out_updated[:, :, update_position] = value_states

        self.keys.copy_(k_out_updated)
        self.values.copy_(v_out_updated)
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        query_length = cache_position.shape[0]
        first_cache_position = cache_position[0]

        kv_offset = torch.clamp(first_cache_position - self.max_cache_len + 1, min=0)
        # This is not general (see HybridChunkedCache for the whole general case), but it's what the cache returns
        kv_length = max(query_length, self.max_cache_len)
        return kv_length, kv_offset


class ChunkedSlidingLayer(SlidingWindowLayer):
    """
    An extended SlidingWindowLayer that supports prefill chunking, originally implemented for Llama 4.

    See `SlidingWindowLayer` for details on common methods that are implemented by all cache layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cumulative_length = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs else None
        if cache_position is None:
            raise ValueError("`cache_position` must be provided for ChunkedSlidingLayer.")

        # This may be needed if the Layer was not created with the right device in the beginning, i.e. if it did not respect
        # the device_map. However, even if it is the case, this will only run once, because then the new states received
        # will always have the same device
        if self.device != key_states.device:
            self.device = key_states.device
            self.keys = self.keys.to(self.device)
            self.values = self.values.to(self.device)

        cumulative_length = self.cumulative_length
        self.cumulative_length += key_states.shape[-2]
        is_full = cumulative_length >= self.max_cache_len

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
        return full_key_states, full_value_states

    def reset(self) -> None:
        super().reset()
        self.cumulative_length = 0

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


class CacheProcessor:
    """
    Base class for cache processors. It defines a pre-update and post-update methods that are called before and after the cache update.
    This class should be subclassed.
    """

    def __init__(self, cache: "Cache", **kwargs) -> None:
        """
        Initialize the processor and perform compatibility checks with the cache.

        Args:
            cache (`Cache`): The cache instance this processor will be applied to.
            **kwargs: Additional arguments that may be needed for initialization.
        """
        raise NotImplementedError(f"Make sure to implement `init` in {self.__class__.__name__}.")

    def pre_update(
        self,
        cache: "Cache",
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Function called before the cache update. Can modify the key/value states.

        Args:
            cache (`Cache`): The cache instance.
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            layer_idx (`int`): The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            The modified key and value states.
        """
        return key_states, value_states

    def post_update(
        self,
        cache: "Cache",
        key_tensors: torch.Tensor,
        value_tensors: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Function called after the cache update. Can process the cached data.

        Args:
            cache (`Cache`): The cache instance.
            key_states (`torch.Tensor`): The key states that were cached.
            value_states (`torch.Tensor`): The value states that were cached.
            layer_idx (`int`): The index of the layer that was updated.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            The final key and value states to return to the model.
        """
        return key_tensors, value_tensors


class OffloadedCacheProcessor(CacheProcessor):
    """
    A cache processor that offloads cache tensors to conserve accelerator memory.

    This processor manages moving cache tensors between accelerator and CPU memory,
    using asynchronous prefetching to minimize performance impact. Works with both
    dynamic and static layers.
    """

    def __init__(self, cache: "Cache", offload_device: Union[str, torch.device] = "cpu", **kwargs):
        """Initialize the offload processor and check device compatibility."""
        self.offload_device = torch.device(offload_device)
        self.original_device = []
        self.prefetch_stream = None
        self.beam_idx = None

        if not (
            torch.cuda.is_available()
            or (is_torch_greater_or_equal("2.7", accept_dev=True) and torch.xpu.is_available())
        ):
            raise RuntimeError(
                "OffloadedCacheProcessor can only be used with a GPU"
                + (" or XPU" if is_torch_greater_or_equal("2.7", accept_dev=True) else "")
            )

        self.is_static = any(isinstance(layer, StaticLayer) for layer in cache.layers)
        if self.is_static:
            for i, layer in enumerate(cache.layers):
                device = cache.layer_init_kwargs["device"] if i == 0 else self.offload_device
                layer.keys = layer.keys.to(device)
                layer.values = layer.values.to(device)
                self.original_device.append(cache.layer_init_kwargs["device"])
            if len(cache) != cache.num_hidden_layers:
                raise ValueError("If static layers are used, all cache layers must be initialized")

        self.prefetch_stream = (
            torch.Stream() if is_torch_greater_or_equal("2.7", accept_dev=True) else torch.cuda.Stream()
        )

    def pre_update(
        self,
        cache: "Cache",
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Handles prefetching and eviction before cache update."""
        # Update the cache
        if len(cache) < layer_idx:
            raise ValueError("OffloadedCache does not support model usage where layers are skipped. Use DynamicCache.")
        elif len(cache) == layer_idx:
            self.original_device.append(key_states.device)
            self._evict_previous_layer(cache, layer_idx)
        else:
            # Wait for the previous layer to be evicted (on default stream)
            if is_torch_greater_or_equal("2.7", accept_dev=True):
                torch.accelerator.current_stream().synchronize()
            else:
                torch.cuda.current_stream().synchronize()
            self._evict_previous_layer(cache, layer_idx)
            self._ensure_layer_on_device(cache, layer_idx)

            # Prefetch the next layer
            self._prefetch_layer(cache, (layer_idx + 1) % len(cache))
        return key_states, value_states

    def _prefetch_layer(self, cache: "Cache", layer_idx: int):
        """Starts prefetching the next layer cache."""
        if layer_idx < len(cache):
            with (
                self.prefetch_stream
                if is_torch_greater_or_equal("2.7", accept_dev=True)
                else torch.cuda.stream(self.prefetch_stream)
            ):
                # Prefetch next layer tensors to GPU
                device = self.original_device[layer_idx]
                cache.layers[layer_idx].keys = cache.layers[layer_idx].keys.to(device, non_blocking=True)
                cache.layers[layer_idx].values = cache.layers[layer_idx].values.to(device, non_blocking=True)

    def _evict_previous_layer(self, cache: "Cache", layer_idx: int):
        """Moves the previous layer cache to the CPU."""
        if len(cache) >= 2:  # Layer 0 stays on device to be on-device after all layers are created
            # We do it on the default stream so it occurs after all earlier computations on these tensors are done
            prev_layer_idx = (layer_idx - 1) % len(cache)
            cache.layers[prev_layer_idx].keys = cache.layers[prev_layer_idx].keys.to(
                self.offload_device, non_blocking=True
            )
            cache.layers[prev_layer_idx].values = cache.layers[prev_layer_idx].values.to(
                self.offload_device, non_blocking=True
            )

    def _ensure_layer_on_device(self, cache: "Cache", layer_idx: int):
        """Ensures the current layer is on the original device."""
        if layer_idx < len(cache):
            # Wait for the previous prefetch to be done
            self.prefetch_stream.synchronize()

            # Handle delayed beam search operations
            if self.beam_idx is not None:
                self.beam_idx = self.beam_idx.to(self.original_device[layer_idx])
                cache.layers[layer_idx].keys = cache.layers[layer_idx].keys.index_select(0, self.beam_idx)
                cache.layers[layer_idx].values = cache.layers[layer_idx].values.index_select(0, self.beam_idx)


class QuantizedCacheProcessor(CacheProcessor):
    """
    A cache processor that applies quantization to cache tensors to reduce memory usage.

    This processor quantizes cache tensors after they are stored, maintaining a residual
    length in original precision and quantizing older tokens.
    """

    def __init__(
        self,
        cache: "Cache",
        backend: str = "quanto",
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ):
        """
        Parameters:
            backend (`str`, defaults to `"quanto"`):
                Backend to use when performing quantization, Can be one of [`quanto`, `HQQ`]
            nbits (`int`, defaults to 4):
                Number of bits, can be 2 or 4 for the `quanto` backend and one of [1, 2, 3, 4, 8] for the `HQQ` backend. Defaults to 2.
            axis_key (`int`, defaults to 0):
                Axis over which to perform grouping for the key tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
            axis_value (`int`, defaults to 0):
                Axis over which to perform grouping for the value tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
            q_group_size (`int`, defaults to 64):
                Size of the quantization group, should be a divisor of the model's hidden dimension.
                Defaults to 64.
            residual_length (`int`, defaults to 128):
                Length of the residual cache which will always be stored in original precision.
                Defaults to 128.
            compute_dtype (`torch.dtype`, defaults to `torch.float16`):
                The default dtype used for computations in the model. Keys and Values will be cast to this dtype after dequantization.
            device (`str`, defaults to `"cpu"`):
                Device on which to perform computations, should be same as the model's device.
        """
        self.backend = backend
        self.nbits = nbits
        self.axis_key = axis_key
        self.axis_value = axis_value
        self.q_group_size = q_group_size
        self.residual_length = residual_length
        self.compute_dtype = compute_dtype
        self.device = device
        self._quantized_keys: list[torch.Tensor] = []
        self._quantized_values: list[torch.Tensor] = []

        self.validate()
        self.erased_length = 0

        # Only compatible with DynamicCache
        if not isinstance(cache.layers[0], DynamicLayer):
            raise ValueError("QuantizedCacheProcessor is only compatible with DynamicCache")

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

    def post_update(
        self,
        cache: "Cache",
        key_tensors: torch.Tensor,
        value_tensors: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply quantization after cache update."""

        if len(cache) < layer_idx:
            raise ValueError("QuantizedCache does not support model usage where layers are skipped. Use DynamicCache.")

        # `key_tensors` is the content of the residual cache, after having been updated by DynamicLayer
        # On the first forward pass, we quantize the whole prompt (prefill, quantize_length=0)
        # On subsequent passes, we accumulate the tokens in the residual cache and quantize when it is full.
        if self._is_quantized_length_zero(layer_idx):
            self._quantized_keys.append(self._quantize(key_tensors.contiguous(), axis=self.axis_key))
            self._quantized_values.append(self._quantize(value_tensors.contiguous(), axis=self.axis_value))

            # Clear the residual cache
            self.erased_length = key_tensors.shape[-2]
            cache.layers[layer_idx].keys = torch.zeros(
                0,
                dtype=key_tensors.dtype,
                device=key_tensors.device,
            )
            cache.layers[layer_idx].values = torch.zeros(
                0,
                dtype=value_tensors.dtype,
                device=value_tensors.device,
            )
            # On prefill, we return the original prompt
            keys_to_return, values_to_return = key_tensors, value_tensors

        else:
            # Prepend the previously quantized cache
            dequant_key = self._dequantize(self._quantized_keys[layer_idx])
            dequant_value = self._dequantize(self._quantized_values[layer_idx])
            keys_to_return = torch.cat([dequant_key, key_tensors], dim=-2)
            values_to_return = torch.cat([dequant_value, value_tensors], dim=-2)
            if key_tensors.shape[-2] >= self.residual_length:
                # Quantize and store
                self._quantized_keys[layer_idx] = self._quantize(keys_to_return.contiguous(), axis=self.axis_key)
                self._quantized_values[layer_idx] = self._quantize(values_to_return.contiguous(), axis=self.axis_value)

                # Clear the residual cache
                self.erased_length += key_tensors.shape[-2]
                cache.layers[layer_idx].keys = torch.zeros(
                    0,
                    dtype=key_tensors.dtype,
                    device=key_tensors.device,
                )
                cache.layers[layer_idx].values = torch.zeros(
                    0,
                    dtype=value_tensors.dtype,
                    device=value_tensors.device,
                )

        return keys_to_return, values_to_return

    def _quantize(self, tensor: torch.Tensor, axis: int) -> torch.Tensor:
        """Quantize a tensor - to be implemented by specific quantization backends."""
        raise NotImplementedError("Quantization backend must implement _quantize method")

    def _dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Dequantize a tensor - to be implemented by specific quantization backends."""
        raise NotImplementedError("Quantization backend must implement _dequantize method")

    def _is_quantized_length_zero(self, layer_idx: int) -> bool:
        """Check if quantized cache is empty for layer. Note: shape[-2] is unreliable since quantized tensors are bit-packed and flattened."""
        return layer_idx >= len(self._quantized_keys)


class QuantoQuantizedCacheProcessor(QuantizedCacheProcessor):
    """
    Quantized cache processor that uses `quanto` as a backend to perform quantization.
    Current implementation supports `int2` and `int4` dtypes only.
    """

    def __init__(
        self,
        cache: "Cache",
        backend: str = "quanto",
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ) -> None:
        """Initialize the quanto quantization processor."""
        super().__init__(
            cache, backend, nbits, axis_key, axis_value, q_group_size, residual_length, compute_dtype, device
        )

        if backend != "quanto":
            raise ValueError(f"QuantoQuantizedCacheProcessor only supports `quanto` backend, but got {backend}")

        if is_optimum_quanto_available():
            optimum_quanto_version = version.parse(importlib.metadata.version("optimum-quanto"))
            if optimum_quanto_version <= version.parse("0.2.5"):
                raise ImportError(
                    f"You need optimum-quanto package version to be greater or equal than 0.2.5 to use `QuantoQuantizedCacheProcessor`. Detected version {optimum_quanto_version}."
                )
            from optimum.quanto import MaxOptimizer, qint2, qint4

        if self.nbits not in [2, 4]:
            raise ValueError(f"`nbits` for `quanto` backend has to be one of [`2`, `4`] but got {self.nbits}")

        if self.axis_key not in [0, -1]:
            raise ValueError(f"`axis_key` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_key}")

        if self.axis_value not in [0, -1]:
            raise ValueError(
                f"`axis_value` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_value}"
            )

        self.qtype = qint4 if self.nbits == 4 else qint2
        self.optimizer = MaxOptimizer()

    def _quantize(self, tensor: torch.Tensor, axis: int) -> torch.Tensor:
        """Quantize tensor using quanto backend."""
        if is_optimum_quanto_available():
            from optimum.quanto import quantize_weight

            scale, zeropoint = self.optimizer(tensor, self.qtype, axis, self.q_group_size)
            qtensor = quantize_weight(tensor, self.qtype, axis, scale, zeropoint, self.q_group_size)
            return qtensor

    def _dequantize(self, qtensor: torch.Tensor) -> torch.Tensor:
        """Dequantize tensor using quanto backend."""
        return qtensor.dequantize()


class HQQQuantizedCacheProcessor(QuantizedCacheProcessor):
    """
    Quantized cache processor that uses `HQQ` as a backend to perform quantization.
    Current implementation supports `int2`, `int4`, `int8` dtypes.
    """

    def __init__(
        self,
        cache: "Cache",
        backend: str = "quanto",
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ) -> None:
        """Initialize the HQQ quantization processor."""
        super().__init__(
            cache, backend, nbits, axis_key, axis_value, q_group_size, residual_length, compute_dtype, device
        )

        if backend != "quanto":
            raise ValueError(f"HQQQuantizedCacheProcessor only supports `quanto` backend, but got {backend}")

        if self.nbits not in [1, 2, 3, 4, 8]:
            raise ValueError(
                f"`nbits` for `HQQ` backend has to be one of [`1`, `2`, `3`, `4`, `8`] but got {self.nbits}"
            )

        if self.axis_key not in [0, 1]:
            raise ValueError(f"`axis_key` for `HQQ` backend has to be one of [`0`, `1`] but got {self.axis_key}")

        if self.axis_value not in [0, 1]:
            raise ValueError(f"`axis_value` for `HQQ` backend has to be one of [`0`, `1`] but got {self.axis_value}")

        self.quantizer = HQQQuantizer

    def _quantize(self, tensor: torch.Tensor, axis: int) -> tuple[torch.Tensor, dict]:
        """Quantize tensor using HQQ backend."""
        qtensor, meta = self.quantizer.quantize(
            tensor,
            axis=axis,
            device=self.device,
            compute_dtype=self.compute_dtype,
            nbits=self.nbits,
            group_size=self.q_group_size,
        )
        meta["compute_dtype"] = self.compute_dtype
        self.quantizer.cuda(qtensor, meta=meta, device=self.device)  # Move to device and cast to dtype
        meta["scale"] = meta["scale"].to(qtensor.device)
        meta["zero"] = meta["zero"].to(qtensor.device)
        return qtensor, meta

    def _dequantize(self, qtensor_and_meta: tuple[torch.Tensor, dict]) -> torch.Tensor:
        """Dequantize tensor using HQQ backend."""
        quant_tensor, meta = qtensor_and_meta
        tensor = self.quantizer.dequantize(quant_tensor, meta)
        return tensor


def apply_processors(
    fn: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    @functools.wraps(fn)
    def _wrapped_update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Wrapper around the update method to apply cache processors.
        """
        if self.cache_processor is not None:
            key_states, value_states = self.cache_processor.pre_update(
                self, key_states, value_states, layer_idx, cache_kwargs
            )

        key_tensors, value_tensors = fn(self, key_states, value_states, layer_idx, cache_kwargs)

        if self.cache_processor is not None:
            key_tensors, value_tensors = self.cache_processor.post_update(
                self, key_tensors, value_tensors, layer_idx, cache_kwargs
            )

        return key_tensors, value_tensors

    return _wrapped_update


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
    Base container for per-layer key/value caches.

    A `Cache` behaves like a list of `CacheLayerMixin` objects, one per model layer.
    Sub-classes such as `DynamicCache`, `StaticCache`, or `SlidingWindowCache`
    simply pre-select which `CacheLayerMixin` class to use and may attach a
    `CacheProcessor` (off-loading, quantization).

    Example
    -------
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    tok   = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    inputs = tok("Hello", return_tensors="pt")

    cache = DynamicCache()
    outputs = model(**inputs, past_key_values=cache, use_cache=True)
    ```

    Parameters:
        layer_classes (`type[CacheLayerMixin]` or `list[type[CacheLayerMixin]]`):
            A list of `CacheLayerMixin` classes to instantiate for the cache. If only a `CacheLayerMixin` class is
            provided, then it is used for all layers.
        config (`PretrainedConfig`, *optional*):
            Model configuration used to infer number of layers, head sizes, default
            device/dtype, etc.
        cache_processor (`CacheProcessor` or `str`, *optional*):
            Cache processor to apply (e.g., "offloaded", "quanto_quantized", "hqq_quantized")
            or a CacheProcessor class.
        max_batch_size (`int`, *optional*): Maximum batch size for static caches.
        max_cache_len (`int`, *optional*): Maximum sequence length. For hybrid caches, SlidingWindowLayers are
            clamped to `min(sliding_window, max_cache_len)`, StaticLayers use full `max_cache_len`.
        device (`torch.device`, *optional*): Device for cache tensors.
        dtype (`torch.dtype`, *optional*): Data type for cache tensors.
        layer_device_map (`dict[int, Union[str, torch.device]]`, *optional*): Per-layer device mapping.
        tp_size (`int`, *optional*): Tensor parallel size to adjust the number of key/value heads.

    Additional keyword arguments are forwarded to the chosen layers constructor(s) and CacheProcessors. See the
    documentation of the relevant `CacheLayerMixin` class and `CacheProcessor` class for more details.
    """

    def __init__(
        self,
        layer_classes: Union[list[type[CacheLayerMixin]], type[CacheLayerMixin]],
        config: Optional[PretrainedConfig] = None,
        cache_processor: Optional[Union[str, type[CacheProcessor]]] = None,
        max_batch_size: Optional[int] = None,
        max_cache_len: Optional[int] = None,
        device: Union[torch.device, str, None] = None,
        dtype: Optional[torch.dtype] = None,
        layer_device_map: Optional[dict[int, torch.device]] = None,
        tp_size: Optional[int] = None,
        **kwargs,
    ):
        self.layers: list[CacheLayerMixin] = []
        self.layer_classes = layer_classes

        processor_class = PROCESSOR_CLASS_MAP[cache_processor] if isinstance(cache_processor, str) else cache_processor
        kwargs.update(
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
            layer_device_map=layer_device_map,
            tp_size=tp_size,
        )
        processor_kwargs, kwargs = parse_processor_args(processor_class, kwargs)

        self.layer_init_kwargs = parse_layer_args_from_model_config(config, **kwargs)
        self.num_hidden_layers = getattr(config, "num_hidden_layers", 1)

        self.append_new_layers(self.num_hidden_layers - 1)
        self.cache_processor = processor_class(self, **processor_kwargs) if processor_class is not None else None

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
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
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.layers[layer_idx].keys, self.layers[layer_idx].values)

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        # Best effort BC support for old-style caches like Mambas, Falcon, HybridChunked that rely on __len__
        if getattr(self, "layers", None) is None:
            if getattr(self, "key_cache", None) is not None:
                return len(self.key_cache)
            return 0
        # Empty dynamic caches initialize an empty layer to be ready for first update
        dynamic_empty = (
            getattr(self, "layers", None) is not None
            and len(self.layers) == 1
            and isinstance(self.layers[0], DynamicLayer)
            and self.layers[0].keys is None
        )
        return len(self.layers) if not dynamic_empty else 0

    def __repr__(self):
        return f"{self.__class__.__name__}(layers={self.layers})"

    def append_new_layers(self, layer_idx: int) -> None:
        """
        Appends layers to the cache until the layer `layer_idx` is reached.
        Used for preallocation in static caches and on the fly in dynamic caches.

        Args:
            layer_idx (`int`):
                The index of the layer to append.
        """
        while len(self.layers) <= layer_idx:
            kwargs = self.layer_init_kwargs.copy()
            if self.layer_init_kwargs.get("layer_device_map", None) is not None:
                kwargs["device"] = kwargs.pop("layer_device_map")[layer_idx]

            new_layer_class = (
                self.layer_classes[len(self.layers)] if isinstance(self.layer_classes, list) else self.layer_classes
            )
            new_layer = new_layer_class(**kwargs)
            self.layers.append(new_layer)

    @apply_processors
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
        self.append_new_layers(layer_idx)
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0, cache_position=None) -> int:
        """Returns the sequence length of the cache for the given layer. TODO: deprecate in favor of cache_position"""
        if layer_idx >= len(self.layers):
            return 0
        # Hack since QuantizedCache messes with keys shape as it becomes the residual cache
        if self.cache_processor is not None and isinstance(self.cache_processor, QuantizedCacheProcessor):
            return self.cache_processor.erased_length + self.layers[layer_idx].get_seq_length(cache_position)
        return self.layers[layer_idx].get_seq_length(cache_position)

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns (i.e. sliding_window, chunk_size),
        for each layer.
        """
        kv_length, kv_offset = self.layers[layer_idx].get_mask_sizes(cache_position)
        return kv_length, kv_offset

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

    ### Wrappers for layer operations and properties ###

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Returns maximum sequence length of the cache object. Dynamic caches do not have a maximum length."""
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
        return all(layer.is_compileable for layer in self.layers)

    @property
    def is_sliding(self) -> list[bool]:
        """Return whether the layers of the cache are sliding window"""
        return [getattr(layer, "is_sliding", False) for layer in self.layers]


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
    def __init__(self, ddp_cache_data: Optional[Iterable[tuple[torch.Tensor, torch.Tensor]]] = None, *args, **kwargs):
        super().__init__(layer_classes=DynamicLayer, *args, **kwargs)
        # `ddp_cache_data` was originally added for compatibility with `torch.distributed` (DDP). See #36212
        # and #36373 for more information. In a nutshell, it is `map(gather_map, zip(*caches))`, i.e. each item in the
        # iterable contains the key and value states for a layer gathered across replicas by torch.distributed
        # (shape=[global batch size, num_heads, seq_len, head_dim]).
        # WARNING: `ddp_cache_data` must be the first argument in `__init__`, otherwise we'll break
        # compatibility. The name of the argument doesn't matter.
        if ddp_cache_data is not None:
            for key_states, value_states in ddp_cache_data:
                self.layers.append(DynamicLayer.from_tensors(key_states, value_states))

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


class OffloadedCache(DynamicCache):
    """
    A drop-in replacement for DynamicCache that conserves accelerator(GPU, XPU) memory at the expense of more CPU memory.
    Useful for generating from models with very long context.

    In addition to the default accelerator stream, where all forward() computations happen,
    this class uses another stream, the prefetch stream, which it creates itself.
    Since scheduling of operations on separate streams happens independently, this class uses
    the prefetch stream to asynchronously prefetch the KV cache of layer k+1 when layer k is executing.
    The movement of the layer k-1 cache to the CPU is handled by the default stream as a simple way to
    ensure the eviction is scheduled after all computations on that cache are finished.
    """

    def __init__(self) -> None:
        # Create the underlying cache with offload processor
        super().__init__(cache_processor=OffloadedCacheProcessor)


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
        >>> past_key_values = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        StaticCache()
        ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(layer_classes=StaticLayer, *args, **kwargs)


class OffloadedStaticCache(StaticCache):
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
        >>> past_key_values = OffloadedStaticCache(
        ...     config=model.config,
        ...     max_batch_size=1,
        ...     max_cache_len=max_generated_length,
        ...     device=model.device,
        ...     dtype=model.dtype
        ... )
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache with offloaded layers
        OffloadedStaticCache()
        ```
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, cache_processor=OffloadedCacheProcessor, **kwargs)


class SlidingWindowCache(Cache):
    """
    Sliding Window Cache class to be used with `torch.compile` for models like Mistral that support sliding window attention.
    Every time when we try to update the cache, we compute the `indices` based on `cache_position >= self.sliding_window - 1`,
    if true(which means the cache can not hold all the old key value states and new states together because of the sliding window constraint),
    we need to do a cycle shift based on `indices` to replace the oldest states by the new key value states passed in.

    The `to_shift` is only true once we are above sliding_window. Thus with `sliding_window==64`:

    indices = (slicing + to_shift[-1].sum()-1) % self.sliding_window
    tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63,  0])

    We overwrite the cache using these, then we always write at cache_position (clamped to `sliding_window`)

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
        >>> past_key_values = SlidingWindowCache(config=model.config, max_batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        SlidingWindowCache()
        ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(layer_classes=SlidingWindowLayer, *args, **kwargs)


class HybridCache(Cache):
    """
    Hybrid Cache class to be used with `torch.compile` for models that alternate between a local sliding window
    attention and global attention in every other layer (originally implemented for Gemma2).
    Under the hood, Hybrid Cache leverages ["SlidingWindowCache"] for sliding window attention and ["StaticCache"]
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
        >>> past_key_values = HybridCache(config=model.config, max_batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        HybridCache()
        ```
    """

    def __init__(self, config: PretrainedConfig, *args, **kwargs):
        if hasattr(config, "layer_types"):
            layer_classes = [LAYER_CLASS_MAP[layer_type] for layer_type in config.layer_types]
        else:
            # In this case, fall back to StaticCache
            layer_classes = [StaticLayer] * config.num_hidden_layers
        super().__init__(config=config, layer_classes=layer_classes, *args, **kwargs)


# The mapping already handles dispatching the correct layers in Hybrid, this is only used for BC
class HybridChunkedCache(HybridCache): ...


class OffloadedHybridCache(HybridChunkedCache):
    """
    A drop-in replacement for HybridChunkedCache that conserves accelerator memory by offloading
    cache tensors to CPU when not actively being used.

    This cache maintains the compilation-friendly properties of HybridChunkedCache while enabling
    much longer sequences by offloading inactive layers to CPU memory.

    See `Cache` for details on common methods that are implemented by all cache classes.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, cache_processor=OffloadedCacheProcessor, **kwargs)


class QuantizedCache(DynamicCache):
    """
    A quantizer cache similar to what is described in the [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://arxiv.org/abs/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for Key and Value cache by applying quantization.

    The cache has two types of storage, one for original precision and one for the quantized cache. A `residual length` is set as a maximum capacity for the
    original precision cache. When the length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache. The
    quantization is done per-channel with a set `q_group_size` for both Keys and Values, in contrast to what was described in the paper.

    It stores Keys and Values a list of quantized tensors (tuples in case we need to store metadata), one for each layer. Additionally, it stores the Key and
    Value in original precision states as a list of tensors, one for each layer. The size of each tensor
    is `[batch_size, num_heads, seq_len - residual_length, head_dim]`.

    See `Cache` for details on common methods that are implemented by all cache classes.
    """

    def __init__(self, backend, **kwargs) -> None:
        if backend == "quanto":
            processor = QuantoQuantizedCacheProcessor
        elif backend == "hqq":
            processor = HQQQuantizedCacheProcessor
        else:
            raise ValueError(f"Unknown quantization backend `{backend}`")

        super().__init__(cache_processor=processor, **kwargs)


class QuantoQuantizedCache(QuantizedCache):
    """
    A quantizer cache similar to what is described in the [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://huggingface.co/papers/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for Key and Value cache by applying quantization.

    The cache has two types of storage, one for original precision and one for the quantized cache. A `residual length` is set as a maximum capacity for the
    original precision cache. When the length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache. The
    quantization is done per-channel with a set `q_group_size` for both Keys and Values, in contrast to what was described in the paper.

    It stores Keys and Values a list of quantized tensors (tuples in case we need to store metadata), one for each layer. Additionally, it stores the Key and
    Value in original precision states as a list of tensors, one for each layer. The size of each tensor
    is `[batch_size, num_heads, seq_len - residual_length, head_dim]`

    Uses `quanto` as a backend to perform quantization. Current implementation supports `int2` and `int4` dtypes only.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

        ```python
        >>> # Run pip install quanto first if you don't have it yet
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoQuantizedCache, QuantizedCacheConfig

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> cache_config = QuantizedCacheConfig(nbits=4)
        >>> past_key_values = QuantoQuantizedCache(cache_config=cache_config)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        QuantoQuantizedCache()
        ```
    """

    def __init__(self, **kwargs) -> None:
        DynamicCache.__init__(self, cache_processor=QuantoQuantizedCacheProcessor, **kwargs)


class HQQQuantizedCache(QuantizedCache):
    """
    A quantizer cache similar to what is described in the [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://arxiv.org/abs/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for Key and Value cache by applying quantization.

    The cache has two types of storage, one for original precision and one for the quantized cache. A `residual length` is set as a maximum capacity for the
    original precision cache. When the length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache. The
    quantization is done per-channel with a set `q_group_size` for both Keys and Values, in contrast to what was described in the paper.

    It stores Keys and Values a list of quantized tensors (tuples in case we need to store metadata), one for each layer. Additionally, it stores the Key and
    Value in original precision states as a list of tensors, one for each layer. The size of each tensor
    is `[batch_size, num_heads, seq_len - residual_length, head_dim]`

    Uses `HQQ` as a backend to perform quantization. Current implementation supports `int2`, `int4`, `int8` dtypes.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

        ```python
        >>> # Run pip install hqq first if you don't have it yet
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, HQQQuantizedCache, QuantizedCacheConfig

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> cache_config = QuantizedCacheConfig(nbits=4, axis_key=1, axis_value=1)
        >>> past_key_values = HQQQuantizedCache(cache_config=cache_config)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        HQQQuantizedCache()
        ```
    """

    def __init__(self, backend="HQQ", **kwargs) -> None:
        assert backend == "HQQ"
        DynamicCache.__init__(self, cache_processor=HQQQuantizedCacheProcessor, **kwargs)


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

    # Override @property from Cache
    is_compileable = None

    def __init__(self, self_attention_cache: Cache, cross_attention_cache: Cache):
        super().__init__(layer_classes=DynamicLayer)
        self.self_attention_cache = self_attention_cache
        self.cross_attention_cache = cross_attention_cache
        self.is_compileable = getattr(self.self_attention_cache, "is_compileable", False)

        self.is_updated = {}
        for layer_idx in range(len(cross_attention_cache)):
            self.is_updated[layer_idx] = bool(cross_attention_cache.get_seq_length(layer_idx) > 0)

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
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
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
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
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
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


def parse_processor_args(processor_class: Optional[type["CacheProcessor"]], kwargs: dict) -> tuple[dict, dict]:
    """
    Parse processor arguments from kwargs based on the processor class init signature.

    Args:
        processor_class: The processor class to inspect, or None
        kwargs: Dictionary of keyword arguments

    Returns:
        tuple: (processor_kwargs, remaining_kwargs)
    """
    try:
        params = list(inspect.signature(processor_class.__init__).parameters)[2:]
    except Exception:
        return {}, kwargs

    processor_kwargs = {k: kwargs[k] for k in params if k in kwargs}
    remaining_kwargs = {k: v for k, v in kwargs.items() if k not in processor_kwargs}
    return processor_kwargs, remaining_kwargs


def parse_layer_args_from_model_config(
    config: Optional[PretrainedConfig],
    batch_size: Optional[int] = None,
    max_cache_len: Optional[int] = None,
    device: Union[torch.device, str, None] = None,
    dtype: Optional[torch.dtype] = None,
    layer_device_map: Optional[dict[int, torch.device]] = None,
    tp_size: Optional[int] = None,
    max_batch_size: Optional[int] = None,
) -> dict:
    """
    Parse layer arguments from model configuration for cache initialization.

    Args:
        config (`Optional[PretrainedConfig]`): Model configuration containing shape/device info.
        batch_size (`Optional[int]`): Batch size for cache initialization.
        max_cache_len (`Optional[int]`): Maximum sequence length for cache.
        device (`Union[torch.device, str, None]`): Device for cache tensors.
        dtype (`Optional[torch.dtype]`): Data type for cache tensors.
        layer_device_map: Per-layer device mapping.
        tp_size (`Optional[int]`): Tensor parallel size to adjust number of key/value heads.
        max_batch_size (`Optional[int]`): Maximum batch size for cache initialization.

    Returns:
        `dict`: Dictionary containing parsed layer arguments for cache initialization.
    """
    # No model config -> must be a dynamic cache, return bare dict
    if config is None:
        return {}
    # Build the args dict for hybrid, sliding or static
    else:
        # Hybrid/Sliding caches require a config that supports sliding_window (max_cache_len already used)
        if (
            getattr(config, "layer_types", None) is not None
            and "sliding_attention" in config.layer_types
            and "full_attention" in config.layer_types
        ):
            if getattr(config, "sliding_window", None) is None:
                raise ValueError(
                    "Setting up a hybrid or sliding window KVCache requires the model config supporting "
                    "sliding window attention, please check if there is a `sliding_window` field in the model "
                    "config and it's not set to None."
                )
        # Adjust max_cache_len for sliding window layers (they can't be larger than sliding window)
        max_cache_len = max_cache_len or config.max_position_embeddings
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads:
        head_dim = (
            config.head_dim
            if getattr(config, "head_dim", None) is not None
            else config.hidden_size // config.num_attention_heads
        )
        num_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )
        if tp_size is not None and tp_size > 1:
            if num_heads % tp_size != 0:
                raise ValueError(
                    f"Number of key value heads {num_heads} must be divisible by tensor parallel size {tp_size}."
                )
            # If the model is using tensor parallelism, we need to adjust the number of heads accordingly.
            num_heads //= tp_size
        layer_args = {
            "batch_size": max_batch_size if max_batch_size is not None else batch_size,
            "max_cache_len": max_cache_len,
            "device": torch.device(device) if device is not None else None,
            "dtype": dtype,
            "layer_device_map": layer_device_map,
            "head_dim": head_dim,
            "num_heads": num_heads,
            "sliding_window": getattr(config, "sliding_window", None),
        }
        return {k: v for k, v in layer_args.items() if v is not None}


LAYER_CLASS_MAP: dict[str, type["CacheLayerMixin"]] = {
    "full_attention": StaticLayer,
    "sliding_attention": SlidingWindowLayer,
    "chunked_attention": ChunkedSlidingLayer,
}
PROCESSOR_CLASS_MAP: dict[str, type["CacheProcessor"]] = {
    "offloaded": OffloadedCacheProcessor,
    "quanto_quantized": QuantizedCacheProcessor,
    "hqq_quantized": HQQQuantizedCacheProcessor,
}


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

    def initialise_cache_layer(self, layer_idx, key_states):
        """Overridden to use the correct device if offloaded layer (and pin memory)."""
        if len(self.key_cache) > layer_idx:
            return

        num_key_value_heads = key_states.shape[1]
        device = key_states.device if self.is_sliding[layer_idx] else self.offload_device
        pin_memory = not self.is_sliding[layer_idx]
        global_cache_shape = (self.max_batch_size, num_key_value_heads, self.max_cache_len, self.head_dim)
        sliding_cache_shape = (self.max_batch_size, num_key_value_heads, self.sliding_window, self.head_dim)
        # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
        # breaks when updating the cache.
        cache_shape = sliding_cache_shape if self.is_sliding[layer_idx] else global_cache_shape
        new_layer_key_cache = torch.zeros(cache_shape, dtype=self._dtype, device=device, pin_memory=pin_memory)
        new_layer_value_cache = torch.zeros(cache_shape, dtype=self._dtype, device=device, pin_memory=pin_memory)
        torch._dynamo.mark_static_address(new_layer_key_cache)
        torch._dynamo.mark_static_address(new_layer_value_cache)
        self.key_cache.append(new_layer_key_cache)
        self.value_cache.append(new_layer_value_cache)

        # Make sure to initialize the on-device layer if it does not already exist
        if self.device_key_cache is None and not self.is_sliding[layer_idx]:
            self.device_key_cache = []
            self.device_value_cache = []
            # We need 2 layers to avoid race conditions when prefetching the next one
            for _ in range(2):
                device_layer_key_cache = torch.zeros(cache_shape, dtype=self._dtype, device=key_states.device)
                device_layer_value_cache = torch.zeros(cache_shape, dtype=self._dtype, device=key_states.device)
                torch._dynamo.mark_static_address(new_layer_key_cache)
                torch._dynamo.mark_static_address(new_layer_value_cache)
                self.device_key_cache.append(device_layer_key_cache)
                self.device_value_cache.append(device_layer_value_cache)

    def _static_update(self, cache_position, layer_idx, key_states, value_states, k_out, v_out, max_cache_len):
        # Wait for prefetch stream if needed
        if self._prefetch_stream is not None:
            torch.cuda.default_stream(key_states.device).wait_stream(self._prefetch_stream)

        # Get correct on-device layer
        k_out = self.device_key_cache[self.active_device_layer]
        v_out = self.device_value_cache[self.active_device_layer]

        # Let's prefetch the next layer as soon as possible
        self._prefetch_next_layer(layer_idx)

        # Copy to on-device layer
        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states

        # Copy to offloaded device
        self.key_cache[layer_idx][:, :, cache_position] = key_states.to(self.offload_device)
        self.value_cache[layer_idx][:, :, cache_position] = value_states.to(self.offload_device)

        return k_out, v_out

    def _prefetch_next_layer(self, layer_idx: int) -> None:
        """Based on current layer_idx, prefetch next full layer to the device."""

        # Switch the active layer
        self.active_device_layer = 0 if self.active_device_layer == 1 else 1

        # Find the next non-sliding layer
        try:
            next_layer = layer_idx + 1 + self.is_sliding[layer_idx + 1 :].index(False)
        # In this case, we are at the last layer, and we go back to prefect the first one
        except ValueError:
            next_layer = self.is_sliding.index(False)

        # Alternate between two on-device caches.
        if self._prefetch_stream is not None:
            with torch.cuda.stream(self._prefetch_stream):
                self._prefetch_layer_in_context(next_layer)
        else:
            self._prefetch_layer_in_context(next_layer)

    def _prefetch_layer_in_context(self, layer_idx: int) -> None:
        """Performs the actual copy of the layer to device cache."""
        if len(self.key_cache) > layer_idx:
            self.device_key_cache[self.active_device_layer].copy_(self.key_cache[layer_idx], non_blocking=True)
            self.device_value_cache[self.active_device_layer].copy_(self.value_cache[layer_idx], non_blocking=True)
        # The layer was not yet initialized
        else:
            self.device_key_cache[self.active_device_layer].fill_(0.0)
            self.device_value_cache[self.active_device_layer].fill_(0.0)


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
