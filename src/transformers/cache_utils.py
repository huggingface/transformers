from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch

from .configuration_utils import PreTrainedConfig
from .utils import (
    is_hqq_available,
    is_optimum_quanto_available,
    is_quanto_greater,
    is_torch_greater_or_equal,
    is_torchdynamo_compiling,
    logging,
)


if is_hqq_available():
    from hqq.core.quantize import Quantizer as HQQQuantizer

_is_torch_greater_or_equal_than_2_7 = is_torch_greater_or_equal("2.7", accept_dev=True)


logger = logging.get_logger(__name__)


class CacheLayerMixin(ABC):
    """Base, abstract class for a single layer's cache."""

    is_compileable = False
    supports_early_init = True
    # Subclasses can set `_layer_type` to auto-register themselves in the mappings, if the class definition lives in a modeling
    # file instead of this file. This allows to update the mapping only when the modeling file is imported, which simplifies imports
    _layer_type: str | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._layer_type is not None:
            if issubclass(cls, StaticLayer):
                STATIC_LAYER_TYPE_MAPPING[cls._layer_type] = cls
            else:
                DYNAMIC_LAYER_TYPE_MAPPING[cls._layer_type] = cls

    def __init__(self, **kwargs):
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.is_initialized = False

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None: ...

    @abstractmethod
    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def get_mask_sizes(self, query_length: int) -> tuple[int, int]: ...

    @abstractmethod
    def get_seq_length(self) -> int: ...

    @abstractmethod
    def get_max_length(self) -> int:
        """
        Returns the maximum sequence length the layer can hold. A value of `-1` means no maximum, or an undefined
        maximum, for example a dynamic attention layer that grows indefinitely or a linear attention layer that has no
        sequence length dimension.
        """
        ...

    def offload(self):
        """Offload this layer's data to CPU device."""
        if self.is_initialized:
            self.keys = self.keys.to("cpu", non_blocking=True)
            self.values = self.values.to("cpu", non_blocking=True)

    def prefetch(self):
        """In case of layer offloading, this allows to move the data back to the layer's device ahead of time."""
        if self.is_initialized and self.keys.device != self.device:
            self.keys = self.keys.to(self.device, non_blocking=True)
            self.values = self.values.to(self.device, non_blocking=True)

    def reset(self) -> None:
        """Resets the cache values while preserving the objects"""
        if self.is_initialized:
            self.keys.zero_()
            self.values.zero_()
        # This attribute is set on several Layers
        if hasattr(self, "cumulative_length"):
            # It can either be an int for dynamic layers, or a tensor for static layers
            if isinstance(self.cumulative_length, int):
                self.cumulative_length = 0
            else:
                self.cumulative_length.zero_()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorders this layer's cache for beam search."""
        if self.get_seq_length() > 0:
            self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
            self.values = self.values.index_select(0, beam_idx.to(self.values.device))

    def get_max_cache_shape(self) -> int:
        logger.warning(
            "`get_max_cache_shape` is deprecated, and will be removed in version 5.16. Please use `get_max_length` instead"
        )
        return self.get_max_length()


class DynamicLayer(CacheLayerMixin):
    """
    A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as tensors of shape `[batch_size, num_heads, seq_len, head_dim]`.
    """

    is_sliding = False

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the mask"""
        kv_offset = 0
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        if not self.is_initialized or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]

    def get_max_length(self) -> int:
        """Returns the maximum sequence length of the cache object. DynamicLayer does not have a maximum length."""
        return -1

    def crop(self, max_length: int) -> None:
        """
        Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be negative
        to remove `max_length` tokens.
        """
        if max_length <= 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self.keys = self.keys[..., :max_length, :]
        self.values = self.values[..., :max_length, :]

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat the cache `repeats` times in the batch dimension."""
        if self.get_seq_length() > 0:
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Only keep the `indices` in the batch dimension of the cache."""
        if self.get_seq_length() > 0:
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]


class DynamicSlidingWindowLayer(DynamicLayer):
    """
    A cache layer that grows dynamically as more tokens are generated, up until the sliding window size.
    It stores the key and value states as tensors of shape `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.
    """

    is_sliding = True

    def __init__(self, sliding_window: int, **kwargs):
        super().__init__()
        self.sliding_window = sliding_window
        self.cumulative_length = 0
        self._sliding_window_tensor = torch.tensor(self.sliding_window, dtype=torch.long)

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        super().lazy_initialization(key_states, value_states)
        self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.cumulative_length += key_states.shape[-2]

        # Compute the full states
        full_key_states = torch.cat([self.keys, key_states], dim=-2)
        full_value_states = torch.cat([self.values, value_states], dim=-2)
        # Only cache the last `self.sliding_window - 1` tokens (or all of them if lower than that)
        self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

        # Return the full states
        return full_key_states, full_value_states

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        is_full = self.cumulative_length >= self.sliding_window

        kv_offset = max(self.cumulative_length - self.sliding_window + 1, 0)
        if is_full:
            kv_length = self.sliding_window - 1 + query_length
        else:
            kv_length = self.cumulative_length + query_length

        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length

    def get_max_length(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.sliding_window

    def crop(self, max_length: int) -> None:
        """
        Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens.
        """
        if self.get_seq_length() >= self.sliding_window:
            raise ValueError(
                "Cannot `crop` a `DynamicSlidingWindowLayer` after it has seen more tokens than its"
                "sliding window (otherwise some states are lost)"
            )
        super().crop(max_length)
        self.cumulative_length = self.keys.shape[-2]


class DynamicIndexedLayer(DynamicLayer):
    """
    A cache layer that extends `DynamicLayer` with an extra indexer key cache for Dynamic Sparse Attention (DSA)
    models (e.g. GLM MoE DSA, DeepSeek V32).

    The main K/V cache stores tensors of shape `[batch_size, num_heads, seq_len, head_dim]` (inherited).
    The indexer key cache stores a tensor of shape `[batch_size, seq_len, index_head_dim]` (3D, single-head).
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.indexer_keys: torch.Tensor | None = None
        self.is_indexer_initialized: bool = False

    def lazy_initialization_indexer(self, indexer_key_states: torch.Tensor) -> None:
        self.indexer_dtype, self.indexer_device = indexer_key_states.dtype, indexer_key_states.device
        self.indexer_keys = torch.tensor([], dtype=self.indexer_dtype, device=self.indexer_device)
        self.is_indexer_initialized = True

    def update_indexer(self, indexer_key_states: torch.Tensor) -> torch.Tensor:
        """
        Update the indexer key cache by concatenation, and return the full indexer keys.

        Args:
            indexer_key_states (`torch.Tensor`): New indexer keys, shape `[batch_size, seq_len, index_head_dim]`.

        Returns:
            `torch.Tensor`: The full cached indexer keys, shape `[batch_size, total_len, index_head_dim]`.
        """
        if not self.is_indexer_initialized:
            self.lazy_initialization_indexer(indexer_key_states)
        self.indexer_keys = torch.cat([self.indexer_keys, indexer_key_states], dim=1)
        return self.indexer_keys

    def offload(self):
        super().offload()
        if self.is_indexer_initialized:
            self.indexer_keys = self.indexer_keys.to("cpu", non_blocking=True)

    def prefetch(self):
        super().prefetch()
        if self.is_indexer_initialized and self.indexer_keys.device != self.device:
            self.indexer_keys = self.indexer_keys.to(self.device, non_blocking=True)

    def reset(self) -> None:
        super().reset()
        if self.is_indexer_initialized:
            self.indexer_keys.zero_()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        if self.is_indexer_initialized and self.indexer_keys.numel() > 0:
            self.indexer_keys = self.indexer_keys.index_select(0, beam_idx.to(self.indexer_keys.device))

    def crop(self, max_length: int) -> None:
        super().crop(max_length)
        if not self.is_indexer_initialized or self.indexer_keys.numel() == 0:
            return
        effective = max_length if max_length >= 0 else self.indexer_keys.shape[1] - abs(max_length)
        if self.indexer_keys.shape[1] > effective:
            self.indexer_keys = self.indexer_keys[:, :effective, :]

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        if self.is_indexer_initialized and self.indexer_keys.numel() > 0:
            self.indexer_keys = self.indexer_keys.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        super().batch_select_indices(indices)
        if self.is_indexer_initialized and self.indexer_keys.numel() > 0:
            self.indexer_keys = self.indexer_keys[indices, ...]


class StaticLayer(CacheLayerMixin):
    """
    A static cache layer that stores the key and value states as static tensors of shape `[batch_size, num_heads, max_cache_len), head_dim]`.
    It lazily allocates its full backing tensors, and then mutates them in-place. Built for `torch.compile` support.

    Args:
        max_cache_len (`int`):
            Maximum number of tokens that can be stored, used for tensor preallocation.
    """

    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int, **kwargs):
        super().__init__()
        self.max_cache_len = max_cache_len
        # Very important that it's a tensor here, to avoid recompiling when we update it and use it to create positions
        self.cumulative_length = torch.tensor(0, dtype=int)

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """
        Lazy initialization of the keys and values tensors. This allows to get all properties (dtype, device,
        num_heads in case of TP etc...) at runtime directly, which is extremely practical as it avoids moving
        devices, dtypes etc later on for each `update` (which could break the static dynamo addresses as well).

        If this is unwanted, one can call `early_initialization(...)` on the Cache directly, which will call this
        function ahead-of-time (this is required for `torch.export` for example). It is also required whenever the
        prefill itself ends up in a compiled region (with chunked prefill for instance).
        """
        self.dtype, self.device = key_states.dtype, key_states.device
        self.batch_size, self.num_heads = key_states.shape[:2]
        self.v_head_dim = value_states.shape[-1]
        self.k_head_dim = key_states.shape[-1]

        self.keys = torch.zeros(
            (self.batch_size, self.num_heads, self.max_cache_len, self.k_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.batch_size, self.num_heads, self.max_cache_len, self.v_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.cumulative_length = self.cumulative_length.to(self.device)
        # Note: `mark_static_address` is used to tag the tensors as a fixed data pointer, preventing compiled graph
        # breaks or cudagraph skips due to inplace mutations when updating the cache. However, it is not supported when
        # tracing the graph, so we skip it in this case. As prefill should never be compiled, this is not an issue and it
        # will still be run (except when users compile prefill explicitly, but this should be avoided!)
        # Without this, we cannot use cudagraphs!!
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)
            torch._dynamo.mark_static_address(self.cumulative_length)

        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        # Create a tensor to slice the static kv at the correct indices
        kv_length = key_states.shape[-2]
        cache_position = torch.arange(kv_length, device=self.device) + self.cumulative_length
        # Note that has to be performed in-place, as we have a static address that we need to keep
        self.cumulative_length.add_(kv_length)

        # Update the cache
        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            # Fallback for devices like MPS where index_copy_ might not be supported.
            self.keys[:, :, cache_position] = key_states
            self.values[:, :, cache_position] = value_states

        return self.keys, self.values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        kv_offset = 0
        kv_length = self.max_cache_len
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length if self.is_initialized else 0

    def get_max_length(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.max_cache_len


class StaticSlidingWindowLayer(StaticLayer):
    """
    A static cache layer that stores the key and value states as static tensors of shape
    `[batch_size, num_heads, min(max_cache_len, sliding_window), head_dim]`. It lazily allocates its full backing
    tensors, and then mutates them in-place. Built for `torch.compile` support.

    Args:
        max_cache_len (`int`):
            Maximum number of tokens that can be stored, used for tensor preallocation.
        sliding_window (`int`):
            The size of the sliding window.
    """

    is_sliding = True

    def __init__(self, max_cache_len: int, sliding_window: int, **kwargs):
        effective_max_cache_len = min(sliding_window, max_cache_len)
        super().__init__(max_cache_len=effective_max_cache_len)
        # Here, to avoid data-dependent control flows, we also need to use a python int to keep track of the cumulative length
        self.cumulative_length_int = 0

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        kv_length = key_states.shape[-2]
        current_length = self.cumulative_length_int
        is_full = current_length >= self.max_cache_len
        # Update it now that we saved the value above
        self.cumulative_length_int += kv_length

        if is_full:
            # In general, we should use a much simpler `cat` here as well, independently of the states size. However,
            # dynamo is currently bugged when doing it - see https://github.com/pytorch/pytorch/issues/159855 for more details
            if key_states.shape[-2] == 1:
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

                # Very important to return the `self` tensors here, as they have the static dynamo address
                return self.keys, self.values
            # Already full but using more than 1 new token (e.g. prefill caching, chat continuation, etc...)
            else:
                full_key_states = torch.cat((self.keys[:, :, 1:, :], key_states), dim=-2)
                full_value_states = torch.cat((self.values[:, :, 1:, :], value_states), dim=-2)
        # Not yet full, but becoming full on this update
        elif current_length + kv_length > self.max_cache_len:
            # Fast prefill path, no need to cat() in this case, as the cache is currently empty
            if current_length == 0:
                full_key_states = key_states
                full_value_states = value_states
            else:
                full_key_states = torch.cat((self.keys[:, :, :current_length, :], key_states), dim=-2)
                full_value_states = torch.cat((self.values[:, :, :current_length, :], value_states), dim=-2)
        else:
            # Note: very important to use the tensor version of the cumulative length here, as otherwise cudagraphs
            # (triggered by mode="reduced_overhead") will lead to random crashes, as the int would be overwritten
            cache_position = torch.arange(kv_length, device=self.device) + self.cumulative_length
            try:
                self.keys.index_copy_(2, cache_position, key_states)
                self.values.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                self.keys[:, :, cache_position] = key_states
                self.values[:, :, cache_position] = value_states

            # Update the tensor version of the length in-place (we don't need to update it if we are already outside
            # of this branch, as we don't need the tensor anymore)
            self.cumulative_length.add_(kv_length)

            # Very important to return the `self` tensors here, as they have the static dynamo address
            return self.keys, self.values

        # We only cache the last `sliding_window` tokens
        self.keys.copy_(full_key_states[:, :, -self.max_cache_len :, :])
        self.values.copy_(full_value_states[:, :, -self.max_cache_len :, :])
        # we should return the whole states instead of `self.keys/values` here, as otherwise we lose some context
        return full_key_states, full_value_states

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        sliding_window = self.max_cache_len
        is_full = self.cumulative_length_int >= self.max_cache_len

        kv_offset = max(self.cumulative_length_int - sliding_window + 1, 0)
        # The cache is already full
        if is_full:
            kv_length = sliding_window + query_length - 1
        # Not yet full, but becoming full on this update
        elif self.cumulative_length_int + query_length > sliding_window:
            kv_length = self.cumulative_length_int + query_length
        # Here the Cache is still smaller than the local size, but we return the local size as it's static
        else:
            kv_length = sliding_window

        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length_int

    def reset(self):
        super().reset()
        self.cumulative_length_int = 0


class StaticIndexedLayer(StaticLayer):
    """
    A `StaticLayer` with an additional statically-allocated indexer key cache for Dynamic Sparse
    Attention (DSA) models (e.g. GLM MoE DSA, DeepSeek V32). This is the static, `torch.compile`-friendly
    counterpart of `DynamicIndexedLayer`: the indexer key buffer is preallocated once and mutated in-place.

    The main K/V cache is inherited from `StaticLayer` (`[batch_size, num_heads, max_cache_len, head_dim]`).
    The indexer key cache stores a tensor of shape `[batch_size, max_cache_len, index_head_dim]` (3D, single-head).
    """

    def __init__(self, max_cache_len: int, **kwargs):
        super().__init__(max_cache_len=max_cache_len)
        self.indexer_keys: torch.Tensor | None = None
        self.is_indexer_initialized: bool = False
        # The indexer update runs independently of (and after) the main K/V `update` in the attention
        # forward, so it tracks its own cumulative length rather than reusing `self.cumulative_length`.
        self.indexer_cumulative_length = torch.tensor(0, dtype=int)

    def lazy_initialization_indexer(self, indexer_key_states: torch.Tensor) -> None:
        self.indexer_dtype, self.indexer_device = indexer_key_states.dtype, indexer_key_states.device
        batch_size, _, index_head_dim = indexer_key_states.shape
        self.indexer_keys = torch.zeros(
            (batch_size, self.max_cache_len, index_head_dim),
            dtype=self.indexer_dtype,
            device=self.indexer_device,
        )
        self.indexer_cumulative_length = self.indexer_cumulative_length.to(self.indexer_device)
        # Tag as static addresses for cudagraphs / compile, mirroring the main K/V buffers.
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.indexer_keys)
            torch._dynamo.mark_static_address(self.indexer_cumulative_length)
        self.is_indexer_initialized = True

    def update_indexer(self, indexer_key_states: torch.Tensor) -> torch.Tensor:
        """
        Update the indexer key cache in-place at the current positions, and return the full static buffer.

        Args:
            indexer_key_states (`torch.Tensor`): New indexer keys, shape `[batch_size, seq_len, index_head_dim]`.

        Returns:
            `torch.Tensor`: The full static indexer key cache, shape `[batch_size, max_cache_len, index_head_dim]`.
                Unfilled positions are masked out downstream by the indexer's attention mask, exactly as the
                main `StaticLayer` returns its full preallocated K/V.
        """
        if not self.is_indexer_initialized:
            self.lazy_initialization_indexer(indexer_key_states)

        seq_len = indexer_key_states.shape[1]
        cache_position = torch.arange(seq_len, device=self.indexer_device) + self.indexer_cumulative_length
        # In-place to preserve the static data pointer (required for cudagraphs).
        self.indexer_cumulative_length.add_(seq_len)
        try:
            self.indexer_keys.index_copy_(1, cache_position, indexer_key_states)
        except NotImplementedError:
            # Fallback for devices like MPS where index_copy_ might not be supported.
            self.indexer_keys[:, cache_position] = indexer_key_states

        return self.indexer_keys

    def reset(self) -> None:
        super().reset()
        if self.is_indexer_initialized:
            self.indexer_keys.zero_()
            self.indexer_cumulative_length.zero_()


class QuantizedLayer(DynamicLayer):
    """
    A quantized layer similar to what is described in the [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://huggingface.co/papers/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for the key and value caches by
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
        super().__init__()
        self.nbits = nbits
        self.axis_key = axis_key
        self.axis_value = axis_value
        self.q_group_size = q_group_size
        self.residual_length = residual_length
        self.cumulative_length = 0

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        self.cumulative_length += key_states.shape[-2]

        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
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

    @abstractmethod
    def _quantize(self, tensor, axis): ...

    @abstractmethod
    def _dequantize(self, q_tensor): ...

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length


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

        # We need to import quanto here to avoid circular imports due to optimum/quanto/models/transformers_models.py
        if not is_optimum_quanto_available():
            raise ImportError(
                "You need to install optimum-quanto in order to use KV cache quantization with optimum-quanto "
                "backend. Please install it via  with `pip install optimum-quanto`"
            )
        elif is_quanto_greater("0.2.5", accept_dev=True):
            from optimum.quanto import MaxOptimizer, qint2, qint4
        else:
            raise ImportError(
                "You need optimum-quanto package version to be greater or equal than 0.2.5 to use `QuantoQuantizedLayer`. "
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
        from optimum.quanto import quantize_weight

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
            raise ImportError(
                "You need to install `HQQ` in order to use KV cache quantization with HQQ backend. "
                "Please install it via  with `pip install hqq`"
            )

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


class LinearAttentionCacheLayerMixin(ABC):
    """Base, abstract class for a linear attention single layer's cache."""

    # All shapes are static by essence in a LinearAttention layer, so it is compileable
    is_compileable = True
    # Linear attention layers track their own conv/recurrent states; they don't use the key/value early-init path.
    supports_early_init = False

    def __init__(self, number_of_states: int = 1, **kwargs):
        self.number_of_states = number_of_states
        # We allow to have an arbitrary number of cached states inside a single layer
        self.conv_states: dict[int, torch.Tensor | None] = dict.fromkeys(range(number_of_states))
        self.recurrent_states: dict[int, torch.Tensor | None] = dict.fromkeys(range(number_of_states))
        self.is_conv_states_initialized = dict.fromkeys(range(number_of_states), False)
        self.is_recurrent_states_initialized = dict.fromkeys(range(number_of_states), False)
        self.has_previous_state = dict.fromkeys(range(number_of_states), False)
        self.conv_kernel_size = dict.fromkeys(range(number_of_states))
        self.device = None
        self.dtype = None
        self.record_past = False

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def lazy_initialization(
        self,
        conv_states: torch.Tensor | None = None,
        recurrent_states: torch.Tensor | None = None,
        state_idx: int = 0,
    ) -> None: ...

    @abstractmethod
    def update_conv_state(self, conv_states: torch.Tensor, state_idx: int = 0) -> torch.Tensor: ...

    @abstractmethod
    def update_recurrent_state(self, recurrent_states: torch.Tensor, state_idx: int = 0) -> torch.Tensor: ...

    def offload(self):
        """Offload this layer's data to CPU device."""
        for i in range(self.number_of_states):
            if self.is_conv_states_initialized[i]:
                self.conv_states[i] = self.conv_states[i].to("cpu", non_blocking=True)
            if self.is_recurrent_states_initialized[i]:
                self.recurrent_states[i] = self.recurrent_states[i].to("cpu", non_blocking=True)

    def prefetch(self):
        """In case of layer offloading, this allows to move the data back to the layer's device ahead of time."""
        for i in range(self.number_of_states):
            if self.is_conv_states_initialized[i] and self.conv_states[i].device != self.device:
                self.conv_states[i] = self.conv_states[i].to(self.device, non_blocking=True)
            if self.is_recurrent_states_initialized[i] and self.recurrent_states[i].device != self.device:
                self.recurrent_states[i] = self.recurrent_states[i].to(self.device, non_blocking=True)

    def reset(self) -> None:
        """Resets the cache values while preserving the objects"""
        for i in range(self.number_of_states):
            if self.is_conv_states_initialized[i]:
                self.conv_states[i].zero_()
            if self.is_recurrent_states_initialized[i]:
                self.recurrent_states[i].zero_()
            self.has_previous_state[i] = False

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for i in range(self.number_of_states):
            if self.is_conv_states_initialized[i]:
                self.conv_states[i] = self.conv_states[i].index_select(0, beam_idx.to(self.device))
            # recurrent_states can stay empty sometimes, see e.g. lfm2 which only uses the conv_states
            if self.is_recurrent_states_initialized[i]:
                self.recurrent_states[i] = self.recurrent_states[i].index_select(0, beam_idx.to(self.device))

    def activate_past_recording(self):
        """
        Calling this function will activate past state recording, meaning that a call to `update_conv_states` will
        wait for a call to `crop` before restricting the size of the `conv_states` to `conv_kernel_size`, to be able
        to retrieve previous full states.
        """
        self.record_past = True

    def crop(self, tokens_to_remove: int):
        if not self.record_past:
            raise RuntimeError(
                "`crop` was called, but the current layer does not track past states. Call `activate_past_recording` before "
                "`crop` to be able to rollback the cache."
            )
        if tokens_to_remove > 0:
            raise RuntimeError(
                "Linear attention layers can only be cropped by passing a negative int, to specify how many tokens to remove"
            )
        for i in range(self.number_of_states):
            tokens_to_remove = abs(tokens_to_remove)
            # This both crop the last `tokens_to_remove`, as well as resize the conv states to `conv_kernel_size` as we never
            # need more for the next forward
            if tokens_to_remove == 0:
                self.conv_states[i] = self.conv_states[i][..., -self.conv_kernel_size[i] :]
            else:
                self.conv_states[i] = self.conv_states[i][
                    ..., -tokens_to_remove - self.conv_kernel_size[i] : -tokens_to_remove
                ]

    def get_max_length(self) -> int:
        # LinearAttention layer have no sequence length dimension, so simply return -1 here
        return -1


class LinearAttentionLayer(LinearAttentionCacheLayerMixin):
    def lazy_initialization(
        self,
        conv_states: torch.Tensor | None = None,
        recurrent_states: torch.Tensor | None = None,
        state_idx: int = 0,
        conv_kernel_size: int | None = None,
    ) -> None:
        if conv_states is not None:
            if self.device is None:
                self.dtype, self.device = conv_states.dtype, conv_states.device
            # Even if prefill is larger/shorter than the conv_size, the tensor is usually either padded or truncated, except if
            # self.record_past is true and conv_kernel_size is provided explicitly
            conv_kernel_size = conv_states.shape[-1] if conv_kernel_size is None else conv_kernel_size
            self.conv_kernel_size[state_idx] = conv_kernel_size
            # The shape is always static, so we init as such
            self.conv_states[state_idx] = torch.zeros(
                (*conv_states.shape[:-1], conv_kernel_size),
                dtype=conv_states.dtype,
                device=conv_states.device,
            )
            # Mark as static address to be able to use cudagraphs
            if not is_torchdynamo_compiling() and not self.record_past:
                torch._dynamo.mark_static_address(self.conv_states[state_idx])
            self.is_conv_states_initialized[state_idx] = True

        if recurrent_states is not None:
            # The shape is always static, so we init as such
            self.recurrent_states[state_idx] = torch.zeros_like(recurrent_states)
            # Mark as static address to be able to use cudagraphs
            if not is_torchdynamo_compiling():
                torch._dynamo.mark_static_address(self.recurrent_states[state_idx])
            self.is_recurrent_states_initialized[state_idx] = True

    def update_conv_state(
        self, conv_states: torch.Tensor, state_idx: int = 0, conv_kernel_size: int | None = None, **kwargs
    ) -> torch.Tensor:
        """
        Update the linear attention cache in-place, and return the necessary conv states.

        Args:
            conv_states (`torch.Tensor`): The new conv states to cache.

        Returns:
            `torch.Tensor`: The updated conv states.
        """
        # Lazy initialization
        if not self.is_conv_states_initialized[state_idx]:
            self.lazy_initialization(conv_states=conv_states, state_idx=state_idx, conv_kernel_size=conv_kernel_size)

        # This is prefill, simply pad the conv_states if necessary
        if not self.has_previous_state[state_idx]:
            full_conv_states = conv_states
            self.has_previous_state[state_idx] = True
            # In this case, need to pad it to fit the conv_kernel_size
            if not self.record_past and full_conv_states.shape[-1] < self.conv_kernel_size[state_idx]:
                padding_length = self.conv_kernel_size[state_idx] - full_conv_states.shape[-1]
                full_conv_states = torch.nn.functional.pad(full_conv_states, (padding_length, 0), value=0)
        # We need to return the concatenation of the current state and the full new one so that the causal conv can see the
        # correct left context - however we usually cache only the last part
        else:
            full_conv_states = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)

        # Usually, keep only the last `conv_kernel_size` tokens
        if not self.record_past:
            # Copy instead of assigning to keep the static address
            self.conv_states[state_idx].copy_(full_conv_states[..., -self.conv_kernel_size[state_idx] :])
        # If we need to record the past, keep the full states for now to be able to rollback later
        else:
            self.conv_states[state_idx] = full_conv_states

        # Return full states no matter what
        return full_conv_states

    def update_recurrent_state(self, recurrent_states: torch.Tensor, state_idx: int = 0, **kwargs) -> torch.Tensor:
        """
        Update the linear attention cache in-place, and return the necessary ssm states.

        Args:
            smm_states (`torch.Tensor`): The new ssm states to cache.

        Returns:
            `torch.Tensor`: The updated ssm states.
        """
        if not self.is_recurrent_states_initialized[state_idx]:
            self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
        # Note that we copy instead of assigning, to preserve the static address for cudagraphs
        self.recurrent_states[state_idx].copy_(recurrent_states)
        return self.recurrent_states[state_idx]


class LinearAttentionAndFullAttentionLayer(LinearAttentionLayer, DynamicLayer):
    # The dynamic Attention part makes it non-compileable
    is_compileable = False

    def __init__(self, number_of_states: int = 1, **kwargs):
        DynamicLayer.__init__(self)
        LinearAttentionLayer.__init__(self, number_of_states=number_of_states)

    def lazy_initialization(self, *args, **kwargs) -> None:
        # When the Attention cache is used with `update`, `lazy_initialization` is called with 2 positional args
        if len(args) == 2 and len(kwargs) == 0:
            DynamicLayer.lazy_initialization(self, *args)
        # Otherwise, for the LinearAttention cache, when it's called in `update_conv_state` or `update_recurrent_state`, it's
        # always called with 1, 2 or 3 kwarg(s) (cause it needs to know if it's for the conv or ssm states)
        if len(args) == 0 and len(kwargs) in (1, 2, 3):
            LinearAttentionLayer.lazy_initialization(self, **kwargs)

    def offload(self):
        DynamicLayer.offload(self)
        LinearAttentionLayer.offload(self)

    def prefetch(self):
        DynamicLayer.prefetch(self)
        LinearAttentionLayer.prefetch(self)

    def reset(self) -> None:
        LinearAttentionLayer.reset(self)
        DynamicLayer.reset(self)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        LinearAttentionLayer.reorder_cache(self, beam_idx)
        DynamicLayer.reorder_cache(self, beam_idx)

    def crop(self, max_length: int) -> None:
        LinearAttentionLayer.crop(self, max_length)
        DynamicLayer.crop(self, max_length)


class LinearAttentionAndSlidingWindowAttentionLayer(LinearAttentionLayer, DynamicSlidingWindowLayer):
    # The dynamic sliding attention part makes it non-compileable
    is_compileable = False

    def __init__(self, sliding_window: int, number_of_states: int = 1, **kwargs):
        DynamicSlidingWindowLayer.__init__(self, sliding_window=sliding_window)
        LinearAttentionLayer.__init__(self, number_of_states=number_of_states)

    def lazy_initialization(self, *args, **kwargs) -> None:
        # When the Attention cache is used with `update`, `lazy_initialization` is called with 2 positional args
        if len(args) == 2 and len(kwargs) == 0:
            DynamicSlidingWindowLayer.lazy_initialization(self, *args)
        # Otherwise, for the LinearAttention cache, when it's called in `update_conv_state` or `update_recurrent_state`, it's
        # always called with 1, 2 or 3 kwarg(s) (cause it needs to know if it's for the conv or ssm states)
        if len(args) == 0 and len(kwargs) in (1, 2, 3):
            LinearAttentionLayer.lazy_initialization(self, **kwargs)

    def reset(self) -> None:
        LinearAttentionLayer.reset(self)
        DynamicSlidingWindowLayer.reset(self)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        LinearAttentionLayer.reorder_cache(self, beam_idx)
        DynamicSlidingWindowLayer.reorder_cache(self, beam_idx)

    def crop(self, max_length: int) -> None:
        LinearAttentionLayer.crop(self, max_length)
        DynamicSlidingWindowLayer.crop(self, max_length)


class LinearAttentionAndStaticFullAttentionLayer(LinearAttentionLayer, StaticLayer):
    def __init__(self, max_cache_len: int, number_of_states: int = 1, **kwargs):
        StaticLayer.__init__(self, max_cache_len)
        LinearAttentionLayer.__init__(self, number_of_states=number_of_states)

    def lazy_initialization(self, *args, **kwargs) -> None:
        # When the Attention cache is used with `update`, `lazy_initialization` is called with 2 positional args
        if len(args) == 2 and len(kwargs) == 0:
            StaticLayer.lazy_initialization(self, *args)
        # Otherwise, for the LinearAttention cache, when it's called in `update_conv_state` or `update_recurrent_state`, it's
        # always called with 1, 2 or 3 kwarg(s) (cause it needs to know if it's for the conv or ssm states)
        if len(args) == 0 and len(kwargs) in (1, 2, 3):
            LinearAttentionLayer.lazy_initialization(self, **kwargs)

    def offload(self):
        StaticLayer.offload(self)
        LinearAttentionLayer.offload(self)

    def prefetch(self):
        StaticLayer.prefetch(self)
        LinearAttentionLayer.prefetch(self)

    def reset(self) -> None:
        LinearAttentionLayer.reset(self)
        StaticLayer.reset(self)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        LinearAttentionLayer.reorder_cache(self, beam_idx)
        StaticLayer.reorder_cache(self, beam_idx)


class LinearAttentionAndStaticSlidingWindowAttentionLayer(LinearAttentionLayer, StaticSlidingWindowLayer):
    def __init__(self, max_cache_len: int, sliding_window: int, number_of_states: int = 1, **kwargs):
        StaticSlidingWindowLayer.__init__(self, max_cache_len=max_cache_len, sliding_window=sliding_window)
        LinearAttentionLayer.__init__(self, number_of_states=number_of_states)

    def lazy_initialization(self, *args, **kwargs) -> None:
        # When the Attention cache is used with `update`, `lazy_initialization` is called with 2 positional args
        if len(args) == 2 and len(kwargs) == 0:
            StaticSlidingWindowLayer.lazy_initialization(self, *args)
        # Otherwise, for the LinearAttention cache, when it's called in `update_conv_state` or `update_recurrent_state`, it's
        # always called with 1, 2 or 3 kwarg(s) (cause it needs to know if it's for the conv or ssm states)
        if len(args) == 0 and len(kwargs) in (1, 2, 3):
            LinearAttentionLayer.lazy_initialization(self, **kwargs)

    def reset(self) -> None:
        LinearAttentionLayer.reset(self)
        StaticSlidingWindowLayer.reset(self)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        LinearAttentionLayer.reorder_cache(self, beam_idx)
        StaticSlidingWindowLayer.reorder_cache(self, beam_idx)


# Mappings from layer_type to layer cache class
DYNAMIC_LAYER_TYPE_MAPPING = {
    "full_attention": DynamicLayer,
    # From a cache point of view, sliding and chunked are the same in how they should behave, only the mask differs
    "sliding_attention": DynamicSlidingWindowLayer,
    "chunked_attention": DynamicSlidingWindowLayer,
    # Linear-attention-shaped placeholders (no per-token KV; recurrent state only).
    # "conv" reuses the same cache shape as linear attention but stores a conv state buffer rather than recurrent SSM state
    "conv": LinearAttentionLayer,
    "moe": LinearAttentionLayer,
    "linear_attention": LinearAttentionLayer,
    # Hybrid layers carry both a linear-attention state and a dynamic-attention state.
    "hybrid": LinearAttentionAndFullAttentionLayer,
    "hybrid_sliding": LinearAttentionAndSlidingWindowAttentionLayer,
    # More exotic implementations
    "deepseek_sparse_attention": DynamicIndexedLayer,
}
# Same but for StaticCache
STATIC_LAYER_TYPE_MAPPING = {
    "full_attention": StaticLayer,
    # From a cache point of view, sliding and chunked are the same in how they should behave, only the mask differs
    "sliding_attention": StaticSlidingWindowLayer,
    "chunked_attention": StaticSlidingWindowLayer,
    # LinearAttention layers are considered both static and dynamic (they are static, but are used as-is for any cache type)
    "conv": LinearAttentionLayer,
    "moe": LinearAttentionLayer,
    "linear_attention": LinearAttentionLayer,
    # Hybrid layers carry both a linear-attention state and a dynamic-attention state.
    "hybrid": LinearAttentionAndStaticFullAttentionLayer,
    "hybrid_sliding": LinearAttentionAndStaticSlidingWindowAttentionLayer,
    # More exotic implementations
    "deepseek_sparse_attention": StaticIndexedLayer,
}


class Cache:
    """
    A `Cache` is mostly a list of `CacheLayerMixin` objects, one per model layer. It serves as a container for
    the Cache of each layer.

    Args:
        layers (`Optional`, *optional*):
            A list of pre-created `CacheLayerMixin` or `LinearAttentionCacheLayerMixin`. If omitted (`None`), then `layer_class_to_replicate`
            will be used.
        layer_class_to_replicate (`type[CacheLayerMixin | LinearAttentionCacheLayerMixin]`, *optional*):
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
        layers: list[CacheLayerMixin | LinearAttentionCacheLayerMixin] | None = None,
        layer_class_to_replicate: type[CacheLayerMixin | LinearAttentionCacheLayerMixin] | None = None,
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

    def __len__(self):
        """
        This value corresponds to the number of layers in the model.
        """
        # Note: for DynamicCache, layers are initialized lazily, so this will not be accurate before the first
        # forward through all the layers
        return len(self.layers)

    def prefetch(self, layer_idx: int, only_non_sliding: bool = True):
        """
        Prefetch the next offloaded layer on its device, starting at `layer_idx` and circling back to the beginning
        if needed. Linear-attention layers are never offloaded and are skipped, as are sliding layers when
        `only_non_sliding`. Note that we use a non-default stream for this, to avoid blocking.
        """
        # Whether each layer is offloaded, hence worth prefetching: linear-attention layers never go through the
        # offloading `update` path, and sliding layers are skipped when `only_non_sliding` (kept resident).
        is_offloaded = [
            not is_linear and not (only_non_sliding and is_sliding)
            for is_linear, is_sliding in zip(self.is_linear, self.is_sliding)
        ]
        try:
            # Try to find the next offloaded layer, starting at `layer_idx`
            layer_idx = layer_idx + is_offloaded[layer_idx:].index(True)
        # In this case, we need to circle back to the beginning
        except ValueError:
            layer_idx = is_offloaded.index(True)

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
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args, **kwargs
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

        keys, values = self.layers[layer_idx].update(key_states, value_states, *args, **kwargs)

        if self.offloading:
            self.offload(layer_idx, self.only_non_sliding)

        return keys, values

    def update_conv_state(
        self, conv_states: torch.Tensor, layer_idx: int, state_idx: int = 0, **kwargs
    ) -> torch.Tensor:
        """
        Updates the cache with the new `conv_states` for the layer `layer_idx`.

        Parameters:
            conv_states (`torch.Tensor`):
                The new conv states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.

        Return:
            `torch.Tensor`: The updated conv states.
        """
        # NOTE: if we slightly break `update` arg order, we could combine this with it, and allow offloading support
        # out of the box
        if not isinstance(self.layers[layer_idx], LinearAttentionCacheLayerMixin):
            raise ValueError("Cannot call `update_conv_state` on a non-LinearAttention layer!")
        conv_states = self.layers[layer_idx].update_conv_state(conv_states, state_idx, **kwargs)
        return conv_states

    def update_recurrent_state(
        self, recurrent_states: torch.Tensor, layer_idx: int, state_idx: int = 0, **kwargs
    ) -> torch.Tensor:
        """
        Updates the cache with the new `recurrent_states` for the layer `layer_idx`.

        Parameters:
            smm_states (`torch.Tensor`):
                The new ssm states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.

        Return:
            `torch.Tensor`: The updated ssm states.
        """
        # NOTE: if we slightly break `update` arg order, we could combine this with it, and allow offloading support
        # out of the box
        if not isinstance(self.layers[layer_idx], LinearAttentionCacheLayerMixin):
            raise ValueError("Cannot call `update_conv_state` on a non-LinearAttention layer!")
        recurrent_states = self.layers[layer_idx].update_recurrent_state(recurrent_states, state_idx, **kwargs)
        return recurrent_states

    def update_indexer(self, indexer_key_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Updates the indexer key cache for layer `layer_idx`.

        Parameters:
            indexer_key_states (`torch.Tensor`):
                The new indexer key states to cache, shape `[batch_size, seq_len, index_head_dim]`.
            layer_idx (`int`):
                The index of the layer to cache the states for.

        Return:
            `torch.Tensor`: The updated indexer key states (full cache).
        """
        if not hasattr(self.layers[layer_idx], "update_indexer"):
            raise ValueError(
                f"Cannot call `update_indexer` on layer {layer_idx} which is a "
                f"{type(self.layers[layer_idx]).__name__}; it has no indexer key cache "
                f"(expected a `DynamicIndexedLayer` or `StaticIndexedLayer`)."
            )
        return self.layers[layer_idx].update_indexer(indexer_key_states)

    def early_initialization(
        self,
        batch_size: int,
        num_heads: int | list[int],
        head_dim: int | list[int],
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        Initialize all the layers in advance (it's otherwise lazily initialized on the first `update` call).
        This is useful for our `export` recipes, as `export` needs everything in advance.
        """
        # To allow different num_heads and head_dim depending on layers, we accept lists
        if isinstance(num_heads, int):
            num_heads = [num_heads] * len(self)
        if isinstance(head_dim, int):
            head_dim = [head_dim] * len(self)

        if len(num_heads) != len(self.layers):
            raise ValueError(
                f"`num_head` was provided as a list of length {len(num_heads)}, but the Cache currently has {len(self.layers)} layers"
            )
        if len(head_dim) != len(self.layers):
            raise ValueError(
                f"`head_dim` was provided as a list of length {len(num_heads)}, but the Cache currently has {len(self.layers)} layers"
            )

        for layer, layer_num_heads, layer_head_dim in zip(self.layers, num_heads, head_dim):
            if not layer.supports_early_init or layer.is_initialized:
                continue
            # Note that the initialization needs all dimensions (except -2), as well as device and dtype, so we use
            # this fake tensor approach. It has size 0 on the -2 dimension, so it does not allocate any data (it only
            # creates an empty tensor with correct shape, dtype and device), which is very efficient and practical
            fake_kv_tensor = torch.zeros((batch_size, layer_num_heads, 0, layer_head_dim), dtype=dtype, device=device)
            # Init the layer
            layer.lazy_initialization(fake_kv_tensor, fake_kv_tensor)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cache for the given layer."""
        if layer_idx >= len(self.layers):
            return 0

        # For alternating attention/linear attention  caches, `get_seq_length` needs to use attention layer idx when called with default layer_idx
        if not isinstance(self.layers[layer_idx], CacheLayerMixin):
            # If this is called with non-default arg, raise
            if layer_idx != 0:
                raise ValueError(
                    f"You called `get_seq_length` on layer index {layer_idx}, but this layer is a LinearAttention layer, which "
                    "does not track sequence length."
                )
            try:
                # Use the first attention layer
                layer_idx = next(idx for idx in range(len(self)) if isinstance(self.layers[idx], CacheLayerMixin))
            except StopIteration:
                raise ValueError(
                    "`get_seq_length` can only be called on Attention layers, and the current Cache seem to only contain "
                    "LinearAttention layers."
                )

        return self.layers[layer_idx].get_seq_length()

    def get_max_length(self, layer_idx: int | None = None) -> int:
        """
        Returns the maximum length of the cache. If `layer_idx` is not provided (default), this returns the maximum
        accross all layers. Otherwise, return the maximum supported value for the given layer.
        A value of `-1` means no maximum, or undefined maximum, e.g. for dynamic attention layers that can grow indefinitely,
        or linear attention layer that do not have a sequence length dimension.
        """
        # For DynamicCache, where the layers are created at runtime
        if layer_idx is not None and layer_idx >= len(self.layers):
            return -1

        if layer_idx is None:
            return max(layer.get_max_length() for layer in self.layers)
        else:
            return self.layers[layer_idx].get_max_length()

    def has_previous_state(self, layer_idx: int | None = None, state_idx: int | None = None) -> bool:
        """Returns whether the LinearAttention layer at index `layer_idx` has previous state or not."""
        if layer_idx is not None and layer_idx >= len(self.layers):
            return False

        # In this case, use last LinearAttention layer
        if layer_idx is None:
            try:
                layer_idx = next(
                    idx
                    for idx in range(len(self) - 1, -1, -1)
                    if isinstance(self.layers[idx], LinearAttentionCacheLayerMixin)
                )
            except StopIteration:
                raise ValueError(
                    "`has_previous_state` can only be called on LinearAttention layers, and the current Cache seem to "
                    "only contain Attention layers."
                )
        elif not isinstance(self.layers[layer_idx], LinearAttentionCacheLayerMixin):
            raise ValueError(
                f"You called `has_previous_state` on layer index {layer_idx}, but this layer is an Attention layer, which "
                "does not support calling it."
            )

        # We may have several conv/recurrent states in the same layers. In this case, if `state_idx` is not provided, check if all
        # of them have previous state
        if state_idx is None:
            return all(self.layers[layer_idx].has_previous_state.values())
        return self.layers[layer_idx].has_previous_state[state_idx]

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
        """
        # For DynamicCache, where the layers are created at runtime -> if it was not yet created, the size is
        # simply the query_length
        if layer_idx >= len(self.layers):
            return query_length, 0

        # For alternating attention/linear attention caches, `get_mask_sizes` needs to use attention layer idx when called with default layer_idx
        if not isinstance(self.layers[layer_idx], CacheLayerMixin):
            # If this is called with non-default arg, raise
            if layer_idx != 0:
                raise ValueError(
                    f"You called `get_mask_sizes` on layer index {layer_idx}, but this layer is a LinearAttention layer, which "
                    "does not track sequence length."
                )
            try:
                # Use the first attention layer
                layer_idx = next(idx for idx in range(len(self)) if isinstance(self.layers[idx], CacheLayerMixin))
            except StopIteration:
                raise ValueError(
                    "`get_mask_sizes` can only be called on Attention layers, and the current Cache seem to only contain "
                    "LinearAttention layers."
                )

        return self.layers[layer_idx].get_mask_sizes(query_length)

    def get_query_offset(self, layer_idx: int) -> int:
        """Returns the current offset of the query for the given `layer_idx`. It's always equal to the cache length, i.e.
        `get_seq_length(layer_idx)`, except for MTP layers.
        """
        # It's simply equal to the length of the past states, except in very specific cases, see `MtpCache`
        return self.get_seq_length(layer_idx=layer_idx)

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

    def activate_past_recording(self):
        """
        Calling this function will activate past state recording, meaning that cache with fixed size such as a linear cache will
        wait for a call to `crop` before restricting the size of its cached states, in order to be able to retrieve previous full states.
        """
        for layer_idx in range(len(self.layers)):
            if hasattr(self.layers[layer_idx], "activate_past_recording"):
                self.layers[layer_idx].activate_past_recording()

    @property
    def batch_size(self) -> int:
        """Return the batch size of the cache, or ``-1`` if no layer has been initialized yet
        (e.g. an all-linear-attention cache queried before the first forward)."""
        # ``LinearAttentionLayer`` sets ``batch_size`` lazily — skip layers that haven't been
        # initialized yet (``generate`` queries this on a fresh cache during cache-reuse checks).
        values = [layer.batch_size for layer in self.layers if hasattr(layer, "batch_size")]
        if not values:
            return -1
        if len(set(values)) > 1:
            raise ValueError(f"The batch size is not consistent across layers: {values}")
        return values[0]

    @property
    def is_compileable(self) -> bool:
        """Return whether the cache is compilable"""
        # For DynamicCache dispatching the layers lazily (otherwise, all([]) is True)
        if len(self.layers) == 0:
            return False
        return all(layer.is_compileable for layer in self.layers)

    @property
    def is_initialized(self) -> bool:
        """Return whether the cache data is initialized"""
        layers = [layer for layer in self.layers if layer.supports_early_init]
        return len(layers) > 0 and all(layer.is_initialized for layer in layers)

    @property
    def is_sliding(self) -> list[bool]:
        """Return whether the layers of the cache are sliding window"""
        return [getattr(layer, "is_sliding", False) for layer in self.layers]

    @property
    def is_linear(self) -> list[bool]:
        """Return whether the layers of the cache are linear attention (Mamba/SSM) layers. Note that layers containing
        both linear and full attention states will return False by this function"""
        return [
            isinstance(layer, LinearAttentionCacheLayerMixin)
            and not isinstance(layer, LinearAttentionAndFullAttentionLayer)
            for layer in self.layers
        ]

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        logger.warning_once(
            "`get_max_cache_shape` is deprecated, and will be removed in version 5.16. Please use `get_max_length` instead"
        )
        return self.get_max_length(layer_idx)

    @property
    def max_cache_len(self) -> int:
        logger.warning_once(
            "`max_cache_len` is deprecated, and will be removed in version 5.16. Please use `get_max_length()` instead"
        )
        return self.get_max_length()

    @property
    def max_batch_size(self) -> int:
        logger.warning_once(
            "`max_batch_size` is deprecated, and will be removed in version 5.16. Please use the simpler `batch_size` instead"
        )
        return self.batch_size


def get_layer_types_and_kwargs(config: PreTrainedConfig) -> tuple[list[str], dict]:
    """
    From a `config`, extract the layer types if not present already, as well as the kwargs needed to initialize
    the corresponding layer caches.
    """
    layer_types = getattr(config, "layer_types", None)
    # If `layer_types` is not explicitly provided, infer it from config fields
    if layer_types is None:
        if getattr(config, "sliding_window", None) is not None:
            layer_types = ["sliding_attention" for _ in range(config.num_hidden_layers)]
        elif getattr(config, "attention_chunk_size", None) is not None:
            layer_types = ["chunked_attention" for _ in range(config.num_hidden_layers)]
        else:
            layer_types = ["full_attention" for _ in range(config.num_hidden_layers)]

    # Some models have shared layers thus no cache is needed for them (e.g. Gemma3n)
    num_kv_shared_layers = getattr(config, "num_kv_shared_layers", None)
    if num_kv_shared_layers is not None and num_kv_shared_layers > 0:
        layer_types = layer_types[: -config.num_kv_shared_layers]

    # Prepare additional kwargs that may be needed to __init__ the cache layers
    layer_kwargs = {}
    if "sliding_attention" in layer_types or "hybrid_sliding" in layer_types:
        layer_kwargs["sliding_window"] = config.sliding_window
    if "chunked_attention" in layer_types:
        layer_kwargs["sliding_window"] = config.attention_chunk_size
    # In this case, we need to pass the config as well to properly __init__ the layer classes
    if "heavily_compressed_attention" in layer_types or "compressed_sparse_attention" in layer_types:
        layer_kwargs["config"] = config
    # We may need more than 1 conv/recurrent state
    if any(layer_type in ("conv", "linear_attention", "hybrid", "hybrid_sliding") for layer_type in layer_types):
        layer_kwargs["number_of_states"] = getattr(config, "number_of_conv_states", 1)

    return layer_types, layer_kwargs


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as a list of `CacheLayer`, one for each layer. The expected shape for each tensor
    in the `CacheLayer`s is `[batch_size, num_heads, seq_len, head_dim]`.
    If a config is passed, it will additionally check for sliding or hybrid cache structure, greatly reducing the
    memory requirement of the cached tensors to `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        ddp_cache_data (`Iterable[tuple[torch.Tensor, torch.Tensor]]`, *optional*):
            It was originally added for compatibility with `torch.distributed` (DDP). In a nutshell, it is
            `map(gather_map, zip(*caches))`, i.e. each item in the iterable contains the key and value states
            for a layer gathered across replicas by torch.distributed (shape=[global batch size, num_heads, seq_len, head_dim]).
            Note: it needs to be the 1st arg as well to work correctly
        config (`PreTrainedConfig`, *optional*):
            The config of the model for which this Cache will be used. If passed, it will be used to check for sliding
            or hybrid layer structure, greatly reducing the memory requirement of the cached tensors to
            `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.
        offloading (`bool`, *optional*, defaults to `False`):
            Whether to perform offloading of the layers to `cpu`, to save GPU memory.
        offload_only_non_sliding (`bool`, *optional*, defaults to `False`):
            If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
            usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).

    Example:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

    >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

    >>> # Prepare a cache class and pass it to model's forward
    >>> past_key_values = DynamicCache(config=model.config)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    ```
    """

    def __init__(
        self,
        ddp_cache_data: Iterable[tuple[torch.Tensor | None, ...]] | None = None,
        config: PreTrainedConfig | None = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        layers = []
        # If a config is passed, use it to infer the layer types and initialize accordingly
        if config is not None:
            decoder_config = config.get_text_config(decoder=True)
            layer_types, layer_kwargs = get_layer_types_and_kwargs(decoder_config)
            # Dispatch the layer types
            layers = [DYNAMIC_LAYER_TYPE_MAPPING[layer_type](**layer_kwargs) for layer_type in layer_types]

        # In this case, use the passed data to already fill in the Cache
        if ddp_cache_data is not None:
            # Init all the layers with the data
            for layer_idx, kv_and_optional_sliding in enumerate(ddp_cache_data):
                # If the config was not passed above, initialize a new cache layer for each entry of the ddp_data
                if config is None:
                    # kv_and_optional_sliding contains at least two elements: the key and value states. It can also
                    # contain a third element, which is an optional sliding window tensor.
                    sliding_window_tensor = kv_and_optional_sliding[2] if len(kv_and_optional_sliding) == 3 else None
                    # If there is a sliding window tensor, use it to initialize the layer
                    if sliding_window_tensor is not None:
                        # Since the same layer is dispatched across replicas, sliding_window is the same for all
                        sliding_window = sliding_window_tensor[0].item()
                        layers.append(DynamicSlidingWindowLayer(sliding_window=sliding_window))
                    else:
                        layers.append(DynamicLayer())
                # Update the layer with the data
                _, _ = layers[layer_idx].update(kv_and_optional_sliding[0], kv_and_optional_sliding[1])

        # If neither of config nor ddp_data was passed, then simply lazy init a full cache of DynamicLayer
        if len(layers) == 0:
            super().__init__(
                layer_class_to_replicate=DynamicLayer,
                offloading=offloading,
                offload_only_non_sliding=offload_only_non_sliding,
            )
        else:
            super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)

    def __iter__(self):
        for layer in self.layers:
            yield layer.keys, layer.values, getattr(layer, "_sliding_window_tensor", None)


class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)` and `torch.export()`. It will check the `config`
    for potential hybrid cache structure, and initialize each layer accordingly.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        config (`PreTrainedConfig`):
            The config of the model for which this Cache will be used. It will be used to check for sliding
            or hybrid layer structure, and initialize each layer accordingly.
        max_cache_len (`int`):
            The maximum number of tokens that this Cache should hold.
        offloading (`bool`, *optional*, defaults to `False`):
            Whether to perform offloading of the layers to `cpu`, to save GPU memory.
        offload_only_non_sliding (`bool`, *optional*, defaults to `True`):
            If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
            usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).

    Example:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

    >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    >>> inputs = tokenizer(text="My name is Llama", return_tensors="pt")

    >>> # Prepare a cache class and pass it to model's forward
    >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
    >>> max_generated_length = inputs.input_ids.shape[1] + 10
    >>> past_key_values = StaticCache(config=model.config, max_cache_len=max_generated_length)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    StaticCache()
    ```
    """

    # Pass-in kwargs as well to avoid crashing for BC (it used more arguments before)
    def __init__(
        self,
        config: PreTrainedConfig,
        max_cache_len: int,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
        **kwargs,
    ):
        layer_types, layer_kwargs = get_layer_types_and_kwargs(config.get_text_config(decoder=True))
        layer_kwargs["max_cache_len"] = max_cache_len
        # Dispatch the layer types
        layers = [STATIC_LAYER_TYPE_MAPPING[layer_type](**layer_kwargs) for layer_type in layer_types]
        super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)


class QuantizedCache(Cache):
    """
    A quantizer cache similar to what is described in the
    [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://huggingface.co/papers/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for keys and values
    by applying quantization.
    The cache has two types of storage, one for original precision and one for the
    quantized cache. A `residual length` is set as a maximum capacity for the original precision cache. When the
    length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache.
    The quantization is done per-channel with a set `q_group_size` for both keys and values, in contrast to what was
    described in the paper.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        backend (`str`):
            The quantization backend to use. One of `("quanto", "hqq").
        config (`PreTrainedConfig`):
            The config of the model for which this Cache will be used.
        nbits (`int`, *optional*, defaults to 4):
            The number of bits for quantization.
        axis_key (`int`, *optional*, defaults to 0):
            The axis on which to quantize the keys.
        axis_value (`int`, *optional*, defaults to 0):
            The axis on which to quantize the values.
        q_group_size (`int`, *optional*, defaults to 64):
            Quantization is done per-channel according to a set `q_group_size` for both keys and values.
        residual_length (`int`, *optional*, defaults to 128):
            Maximum capacity for the original precision cache
    """

    def __init__(
        self,
        backend: str,
        config: PreTrainedConfig,
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

        config = config.get_text_config(decoder=True)
        layer_types, _ = get_layer_types_and_kwargs(config)
        invalid_layer_types = set(layer_types) - {"full_attention"}
        if len(invalid_layer_types) > 0:
            raise ValueError(
                "`QuantizedCache` is only supported for models with only full attention layers. We found the following invalid layer "
                f"types: {invalid_layer_types}"
            )
        layers = [
            layer_class(nbits, axis_key, axis_value, q_group_size, residual_length)
            for _ in range(config.num_hidden_layers)
        ]
        super().__init__(layers=layers)


class EncoderDecoderCache(Cache):
    """
    Base, abstract class for all encoder-decoder caches. Can be used to hold combinations of self-attention and
    cross-attention caches.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        caches (`Iterable`):
            Usually an iterable of length 2, containing 2 `Cache` objects, the first one for self-attention, the
            second one for cross-attention. Can optionally also be an iterable of length 1, containing a
            `tuple[tuple[torch.Tensor]]` (usually used for compatibility with torch dp and ddp).

    Example:

    ```python
    >>> from transformers import AutoProcessor, AutoModelForCausalLM, DynamicCache, EncoderDecoderCache

    >>> model = AutoModelForCausalLM.from_pretrained("openai/whisper-small")
    >>> processor = AutoProcessor.from_pretrained("openai/whisper-small")

    >>> inputs = processor(audio=YOUR-AUDIO, return_tensors="pt")

    >>> # Prepare cache classes for encoder and decoder and pass it to model's forward
    >>> self_attention_cache = DynamicCache(config=self.config)
    >>> cross_attention_cache = DynamicCache(config=self.config)
    >>> past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    EncoderDecoderCache()
    ```
    """

    def __init__(self, *caches) -> None:
        # For dp and ddp support, if only one argument is passed, it should be an iterable of DynamicCache ddp data
        if len(caches) == 1:
            self_attention_cache_data, cross_attention_cache_data = [], []
            for combined_cache_data in caches[0]:
                if len(combined_cache_data) == 6:  # two tuple of style (self_attn_k, self_attn_v, self_attn_sliding)
                    self_attention_cache_data.append(combined_cache_data[:3])
                    cross_attention_cache_data.append(combined_cache_data[3:])
                # To support old DDP-style init, we handle the case where the tuple has no sliding window tensor
                elif len(combined_cache_data) == 4:  # two tuple of style (self_attn_k, self_attn_v)
                    self_attention_cache_data.append(combined_cache_data[:2])
                    cross_attention_cache_data.append(combined_cache_data[2:])
                else:
                    raise ValueError(f"Expected {len(combined_cache_data) = } to be 4 or 6.\n{combined_cache_data = }")
            self.self_attention_cache = DynamicCache(self_attention_cache_data)
            self.cross_attention_cache = DynamicCache(cross_attention_cache_data)
        # Otherwise, we should get two arguments, a self-attention cache and a cross-attention cache
        elif len(caches) == 2:
            if not isinstance(caches[0], Cache) or not isinstance(caches[1], Cache):
                raise TypeError(f"One of the two arguments is not a Cache: {type(caches[0]) = }, {type(caches[1]) = }")
            self.self_attention_cache = caches[0]
            self.cross_attention_cache = caches[1]
        # Error case
        else:
            raise ValueError(f"Expected 1 or 2 arguments, got {len(caches)}")

        self.is_updated = {}
        for layer_idx in range(len(self.cross_attention_cache)):
            self.is_updated[layer_idx] = bool(self.cross_attention_cache.get_seq_length(layer_idx) > 0)

    def __iter__(self):
        """Returns tuples of style (self_attn_k, self_attn_v, self_attn_sliding, cross_attn_k, cross_attn_v, cross_attn_sliding)"""
        for self_attention_layer, cross_attention_layer in zip(self.self_attention_cache, self.cross_attention_cache):
            yield self_attention_layer + cross_attention_layer

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(self_attention_cache={self.self_attention_cache}, cross_attention_cache="
            f"{self.cross_attention_cache})"
        )

    def __len__(self):
        """
        Support for backwards-compatible `past_key_values` length, e.g. `len(past_key_values)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.self_attention_cache)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return self.self_attention_cache.get_seq_length(layer_idx)

    def get_max_length(self, layer_idx: int | None = None) -> int:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object"""
        return self.self_attention_cache.get_max_length(layer_idx)

    def reset(self):
        self.self_attention_cache.reset()
        self.cross_attention_cache.reset()
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
            raise TypeError(
                f"`{method}` is only defined for dynamic cache, got {self.self_attention_cache.__str__()} for the self "
                f"attention cache and {self.cross_attention_cache.__str__()} for the cross attention cache."
            )

    # TODO(gante, sanchit-gandhi): move following functionality into `.generate`
    def crop(self, maximum_length: int):
        """
        Crop the past key values up to a new `maximum_length` in terms of tokens. `maximum_length` can also be
        negative to remove `maximum_length` tokens. This is used in assisted decoding and contrastive search (on the Hub).
        """
        self.check_dynamic_cache(self.crop.__name__)
        self.self_attention_cache.crop(maximum_length)

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search (on the Hub)."""
        self.check_dynamic_cache(self.batch_repeat_interleave.__name__)
        self.self_attention_cache.batch_repeat_interleave(repeats)
        self.cross_attention_cache.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search (on the Hub)."""
        self.check_dynamic_cache(self.batch_select_indices.__name__)
        self.self_attention_cache.batch_select_indices(indices)
        self.cross_attention_cache.batch_select_indices(indices)

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        return self.self_attention_cache.get_mask_sizes(query_length, layer_idx)

    @property
    def is_sliding(self):
        return self.self_attention_cache.is_sliding

    @property
    def is_compileable(self) -> bool:
        return self.self_attention_cache.is_compileable

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        logger.warning_once(
            "`get_max_cache_shape` is deprecated, and will be removed in version 5.16. Please use `get_max_length` instead"
        )
        return self.get_max_length(layer_idx)


# Deprecated alias: SlidingWindowCache was removed in transformers v5. StaticCache is the replacement.
SlidingWindowCache = StaticCache


class MtpCache(DynamicCache):
    def get_query_offset(self, layer_idx=0):
        # Queries of MTP depth k run k+1 tokens ahead of the main_model, i.e. they have an offset of k+1
        mtp_offset = layer_idx + 1
        return super().get_query_offset(layer_idx) + mtp_offset

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        mtp_offset = layer_idx + 1
        kv_length, kv_offset = super().get_mask_sizes(query_length, layer_idx)
        return kv_length, kv_offset + mtp_offset
