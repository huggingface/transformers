import copy
import importlib.metadata
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version

from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_6

from .configuration_utils import PretrainedConfig
from .utils import is_hqq_available, is_optimum_quanto_available, is_torch_greater_or_equal, logging


if is_hqq_available():
    from hqq.core.quantize import Quantizer as HQQQuantizer

logger = logging.get_logger(__name__)


# Utility functions for static/sliding cache update logic
def _static_cache_update(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_position: Optional[torch.LongTensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Updates the static cache tensors in place.

    Args:
        k_cache (`torch.Tensor`): The key cache tensor to update.
        v_cache (`torch.Tensor`): The value cache tensor to update.
        key_states (`torch.Tensor`): The new key states to add.
        value_states (`torch.Tensor`): The new value states to add.
        cache_position (`Optional[torch.LongTensor]`): The position indices where the new states should be inserted.
                                                       If None, the entire cache is overwritten (prefill).

    Returns:
        Tuple[`torch.Tensor`, `torch.Tensor`]: The updated key and value cache tensors (modified in-place).
    """
    if cache_position is None:
        # Prefill phase where seq_len potentially equals max_cache_len. Directly copy.
        k_cache.copy_(key_states)
        v_cache.copy_(value_states)
    else:
        # Generation phase. Update specific positions.
        # Use index_copy_ for in-place update (compile-friendly).
        try:
            k_cache.index_copy_(2, cache_position, key_states)
            v_cache.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            # Fallback for devices like MPS where index_copy_ might not be supported.
            k_cache[:, :, cache_position] = key_states
            v_cache[:, :, cache_position] = value_states
    return k_cache, v_cache


def _sliding_cache_update(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_position: torch.LongTensor,
    max_cache_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Updates the sliding window cache tensors, returning the potentially modified tensors.

    Args:
        k_cache (`torch.Tensor`): The key cache tensor to update.
        v_cache (`torch.Tensor`): The value cache tensor to update.
        key_states (`torch.Tensor`): The new key states to add.
        value_states (`torch.Tensor`): The new value states to add.
        cache_position (`torch.LongTensor`): The position indices where the new states should be inserted.
        max_cache_len (`int`): The maximum length of the sliding window cache.

    Returns:
        Tuple[`torch.Tensor`, `torch.Tensor`]: The key and value tensors representing the cache state after the update.
                                               For prefill > window, these are the full input states.
                                               Otherwise, they are the updated cache tensors.
    """
    # Handle prefill phase when prompt length > sliding_window_size
    if cache_position.shape[0] > max_cache_len:
        new_k = key_states[:, :, -max_cache_len:, :]
        new_v = value_states[:, :, -max_cache_len:, :]
        k_cache.copy_(new_k)
        v_cache.copy_(new_v)
        return key_states, value_states

    # Sliding window logic for generation phase or prefill < window
    slicing = torch.arange(max_cache_len, device=value_states.device)
    current_seq_len = cache_position[-1] + 1  # Use last position to determine current length
    to_shift = current_seq_len > max_cache_len
    indices = (slicing + to_shift.sum()) % max_cache_len

    k_out_shifted = k_cache[:, :, indices]
    v_out_shifted = v_cache[:, :, indices]

    # Clamp cache_position to determine the *target index* within the shifted cache view
    update_position = cache_position.clamp(min=0, max=max_cache_len - 1)

    try:
        k_out_updated = k_out_shifted.index_copy(2, update_position, key_states)
        v_out_updated = v_out_shifted.index_copy(2, update_position, value_states)
    except NotImplementedError:
        # Fallback for MPS: clone and modify the clone
        k_out_updated = k_out_shifted.clone()
        v_out_updated = v_out_shifted.clone()
        k_out_updated[:, :, update_position] = key_states
        v_out_updated[:, :, update_position] = value_states

    k_cache.copy_(k_out_updated)
    v_cache.copy_(v_out_updated)
    return k_out_updated, v_out_updated


class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    is_compileable = False

    def __init__(self):
        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object"""
        raise NotImplementedError("Make sure to implement `get_max_cache_shape` in a subclass.")

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_cache_shape()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel():
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            if self.value_cache[layer_idx].numel():
                device = self.value_cache[layer_idx].device
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    @property
    def seen_tokens(self):
        logger.warning_once(
            "The `seen_tokens` attribute is deprecated and will be removed in v4.41. Use the `cache_position` "
            "model input instead."
        )
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns (i.e. sliding_window, chunk_size),
        for each layer.
        """
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, 0


@dataclass
class CacheConfig:
    """
    Base class for cache configs
    """

    cache_implementation: None

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Constructs a CacheConfig instance from a dictionary of parameters.
        Args:
            config_dict (Dict[str, Any]): Dictionary containing configuration parameters.
            **kwargs: Additional keyword arguments to override dictionary values.

        Returns:
            CacheConfig: Instance of CacheConfig constructed from the dictionary.
        """
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
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
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
            kwargs (`Dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `Dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
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
    Configuration class for quantized cache settings.

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
        self.batch_size = batch_size
        self.max_cache_len = max_cache_len
        self.device = device

    def validate(self):
        """Validates if the arguments passed are correct"""

        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )

        if self.batch_size <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="batch_size",
                    correct_value="> 0",
                    found_value=self.batch_size,
                ),
            )

        if self.max_cache_len <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="max_cache_len",
                    correct_value="> 0",
                    found_value=self.max_cache_len,
                ),
            )


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

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

    def __init__(self, _distributed_cache_data: Optional[Iterable] = None) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        # `_distributed_cache_data` was originally added for compatibility with `torch.distributed` (DDP). See #36121
        # and #36373 for more information. In a nutshell, it is `map(gather_map, zip(*caches))`, i.e. each item in the
        # iterable contains the key and value states for a layer gathered across replicas by torch.distributed
        # (shape=[global batch size, num_heads, seq_len, head_dim]).
        # WARNING: `_distributed_cache_data` must be the first argument in `__init__`, otherwise we'll break
        # compatibility. The name of the argument doesn't matter.
        if _distributed_cache_data is not None:
            for key_states, value_states in _distributed_cache_data:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif (
                not self.key_cache[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or not self.key_cache[layer_idx].numel()  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor, torch.FloatTensor]]] = None
    ) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
        backward compatibility."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            if self.key_cache[idx].numel():
                self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    def batch_split(self, full_batch_size: int, split_size: int) -> List["DynamicCache"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = DynamicCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(cls, splits: List["DynamicCache"]) -> "DynamicCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls()
        for idx in range(len(splits[0])):
            key_cache = [current.key_cache[idx] for current in splits if current.key_cache[idx].numel()]
            value_cache = [current.value_cache[idx] for current in splits if current.value_cache[idx].numel()]
            if key_cache != []:
                layer_keys = torch.cat(key_cache, dim=0)
                layer_values = torch.cat(value_cache, dim=0)
                cache.update(layer_keys, layer_values, idx)
        return cache

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]


# Utilities for `DynamicCache` <> torch.export support
def _flatten_dynamic_cache(
    dynamic_cache: DynamicCache,
):
    """Flattens DynamicCache into flat list of tensors for `torch.export.export` to consume"""
    if not isinstance(dynamic_cache, DynamicCache):
        raise RuntimeError("This pytree flattening function should only be applied to DynamicCache")

    if not is_torch_greater_or_equal_than_2_6:
        logger.warning_once(
            "DynamicCache + torch.export is tested on torch 2.6.0+ and may not work on earlier versions."
        )

    # NOTE it seems _seen_tokens is deprecated, so probably doesn't need tracking
    dictionary = {
        "key_cache": getattr(dynamic_cache, "key_cache"),
        "value_cache": getattr(dynamic_cache, "value_cache"),
    }
    return torch.utils._pytree._dict_flatten(dictionary)


def _flatten_with_keys_dynamic_cache(dynamic_cache: DynamicCache):
    dictionary = {
        "key_cache": getattr(dynamic_cache, "key_cache"),
        "value_cache": getattr(dynamic_cache, "value_cache"),
    }
    return torch.utils._pytree._dict_flatten_with_keys(dictionary)


def _unflatten_dynamic_cache(
    values,
    context: torch.utils._pytree.Context,
):
    dictionary = torch.utils._pytree._dict_unflatten(values, context)
    cache = DynamicCache()
    for k, v in dictionary.items():
        setattr(cache, k, v)
    return cache


def _flatten_dynamic_cache_for_fx(cache, spec):
    dictionary = {
        "key_cache": getattr(cache, "key_cache"),
        "value_cache": getattr(cache, "value_cache"),
    }
    return torch.utils._pytree.tree_flatten(dictionary)[0]


if is_torch_greater_or_equal("2.3"):
    torch.utils._pytree.register_pytree_node(
        DynamicCache,
        _flatten_dynamic_cache,
        _unflatten_dynamic_cache,
        serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
        flatten_with_keys_fn=_flatten_with_keys_dynamic_cache,
    )
    # TODO (tmanlaibaatar) This won't be needed in torch 2.7.
    torch.fx._pytree.register_pytree_flatten_spec(DynamicCache, _flatten_dynamic_cache_for_fx)


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
        if not (
            torch.cuda.is_available()
            or (is_torch_greater_or_equal("2.7", accept_dev=True) and torch.xpu.is_available())
        ):
            raise RuntimeError(
                "OffloadedCache can only be used with a GPU"
                + (" or XPU" if is_torch_greater_or_equal("2.7", accept_dev=True) else "")
            )

        super().__init__()
        self.original_device = []
        self.prefetch_stream = None
        self.prefetch_stream = (
            torch.Stream() if is_torch_greater_or_equal("2.7", accept_dev=True) else torch.cuda.Stream()
        )
        self.beam_idx = None  # used to delay beam search operations

    def prefetch_layer(self, layer_idx: int):
        "Starts prefetching the next layer cache"
        if layer_idx < len(self):
            with (
                self.prefetch_stream
                if is_torch_greater_or_equal("2.7", accept_dev=True)
                else torch.cuda.stream(self.prefetch_stream)
            ):
                # Prefetch next layer tensors to GPU
                device = self.original_device[layer_idx]
                self.key_cache[layer_idx] = self.key_cache[layer_idx].to(device, non_blocking=True)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].to(device, non_blocking=True)

    def evict_previous_layer(self, layer_idx: int):
        "Moves the previous layer cache to the CPU"
        if len(self) > 2:
            # We do it on the default stream so it occurs after all earlier computations on these tensors are done
            prev_layer_idx = (layer_idx - 1) % len(self)
            self.key_cache[prev_layer_idx] = self.key_cache[prev_layer_idx].to("cpu", non_blocking=True)
            self.value_cache[prev_layer_idx] = self.value_cache[prev_layer_idx].to("cpu", non_blocking=True)

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        "Gets the cache for this layer to the device. Prefetches the next and evicts the previous layer."
        if layer_idx < len(self):
            # Evict the previous layer if necessary
            if is_torch_greater_or_equal("2.7", accept_dev=True):
                torch.accelerator.current_stream().synchronize()
            else:
                torch.cuda.current_stream().synchronize()
            self.evict_previous_layer(layer_idx)
            # Load current layer cache to its original device if not already there
            original_device = self.original_device[layer_idx]
            self.prefetch_stream.synchronize()
            key_tensor = self.key_cache[layer_idx]
            value_tensor = self.value_cache[layer_idx]
            # Now deal with beam search ops which were delayed
            if self.beam_idx is not None:
                self.beam_idx = self.beam_idx.to(original_device)
                key_tensor = key_tensor.index_select(0, self.beam_idx)
                value_tensor = value_tensor.index_select(0, self.beam_idx)
            # Prefetch the next layer
            self.prefetch_layer((layer_idx + 1) % len(self))
            return (key_tensor, value_tensor)
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Saves the beam indices and reorders the cache when the tensor is back to its device."""
        # We delay this operation until the tensors are back to their original
        # device because performing torch.index_select on the CPU is very slow
        del self.beam_idx
        self.beam_idx = beam_idx.clone()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `OffloadedCache`.
        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) < layer_idx:
            raise ValueError("OffloadedCache does not support model usage where layers are skipped. Use DynamicCache.")
        elif len(self.key_cache) == layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.original_device.append(key_states.device)
            self.evict_previous_layer(layer_idx)
        else:
            key_tensor, value_tensor = self[layer_idx]
            self.key_cache[layer_idx] = torch.cat([key_tensor, key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([value_tensor, value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    # According to https://docs.python.org/3/library/exceptions.html#NotImplementedError
    # if a method is not supposed to be supported in a subclass we should set it to None
    from_legacy_cache = None

    to_legacy_cache = None


class QuantizedCache(DynamicCache):
    """
    A quantizer cache similar to what is described in the [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://huggingface.co/papers/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for Key and Value cache by applying quantization.

    The cache has two types of storage, one for original precision and one for the quantized cache. A `residual length` is set as a maximum capacity for the
    original precision cache. When the length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache. The
    quantization is done per-channel with a set `q_group_size` for both Keys and Values, in contrast to what was described in the paper.

    It stores Keys and Values a list of quantized tensors (tuples in case we need to store metadata), one for each layer. Additionally, it stores the Key and
    Value in original precision states as a list of tensors, one for each layer. The size of each tensor
    is `[batch_size, num_heads, seq_len - residual_length, head_dim]`
    """

    def __init__(self, cache_config: QuantizedCacheConfig) -> None:
        super().__init__()
        self._quantized_key_cache: List[torch.Tensor] = []
        self._quantized_value_cache: List[torch.Tensor] = []

        self.nbits = cache_config.nbits
        self.residual_length = cache_config.residual_length
        self.q_group_size = cache_config.q_group_size
        self.axis_key = cache_config.axis_key
        self.axis_value = cache_config.axis_value
        self.compute_dtype = cache_config.compute_dtype
        self.device = cache_config.device

        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) < layer_idx:
            raise ValueError("QuantizedCache does not support model usage where layers are skipped. Use DynamicCache.")
        elif len(self.key_cache) == layer_idx:
            self._quantized_key_cache.append(self._quantize(key_states.contiguous(), axis=self.axis_key))
            self._quantized_value_cache.append(self._quantize(value_states.contiguous(), axis=self.axis_value))
            self.key_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
            self.value_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
            keys_to_return, values_to_return = key_states, value_states
        else:
            dequant_key = self._dequantize(self._quantized_key_cache[layer_idx])
            dequant_value = self._dequantize(self._quantized_value_cache[layer_idx])
            keys_to_return = [dequant_key, self.key_cache[layer_idx], key_states]
            values_to_return = [dequant_value, self.value_cache[layer_idx], value_states]

            keys_to_return = torch.cat(keys_to_return, dim=-2)
            values_to_return = torch.cat(values_to_return, dim=-2)
            if (
                self.key_cache[layer_idx].dim() == 4
                and self.key_cache[layer_idx].shape[-2] + 1 >= self.residual_length
            ):
                self._quantized_key_cache[layer_idx] = self._quantize(keys_to_return.contiguous(), axis=self.axis_key)
                self._quantized_value_cache[layer_idx] = self._quantize(
                    values_to_return.contiguous(), axis=self.axis_value
                )
                self.key_cache[layer_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
                self.value_cache[layer_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        # since we cannot get the seq_length of each layer directly and rely on `_seen_tokens` which is
        # updated every "layer_idx" == 0, this is a hack to get the actual seq_length for the given layer_idx
        # this part of code otherwise fails when used to verify attn_weight shape in some models
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

    def _quantize(self, tensor, axis):
        """Quantizes a key/value using a defined quantization method."""
        raise NotImplementedError("Make sure to implement `_quantize` in a subclass.")

    def _dequantize(self, q_tensor):
        """Dequantizes back the tensor that was quantized by `self._quantize()`"""
        raise NotImplementedError("Make sure to implement `_dequantize` in a subclass.")


class QuantoQuantizedCache(QuantizedCache):
    """
    Quantized Cache class that uses `quanto` as a backend to perform quantization. Current implementation supports `int2` and `int4` dtypes only.

    Parameters:
        cache_config (`QuantizedCacheConfig`):
            A configuration containing all the arguments to be used by the quantizer, including axis, qtype and group size.

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

    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)

        if is_optimum_quanto_available():
            optimum_quanto_version = version.parse(importlib.metadata.version("optimum-quanto"))
            if optimum_quanto_version <= version.parse("0.2.5"):
                raise ImportError(
                    f"You need optimum-quanto package version to be greater or equal than 0.2.5 to use `QuantoQuantizedCache`. Detected version {optimum_quanto_version}."
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
        self.optimizer = MaxOptimizer()  # hardcode as it's the only one for per-channel quantization

    def _quantize(self, tensor, axis):
        # We have two different API since in optimum-quanto, we don't use AffineQuantizer anymore
        if is_optimum_quanto_available():
            from optimum.quanto import quantize_weight

            scale, zeropoint = self.optimizer(tensor, self.qtype, axis, self.q_group_size)
            qtensor = quantize_weight(tensor, self.qtype, axis, scale, zeropoint, self.q_group_size)
            return qtensor

    def _dequantize(self, qtensor):
        return qtensor.dequantize()


class HQQQuantizedCache(QuantizedCache):
    """
    Quantized Cache class that uses `HQQ` as a backend to perform quantization. Current implementation supports `int2`, `int4`, `int8` dtypes.

    Parameters:
        cache_config (`QuantizedCacheConfig`):
            A configuration containing all the arguments to be used by the quantizer, including axis, qtype and group size.

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

    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)
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

    def _dequantize(self, qtensor):
        quant_tensor, meta = qtensor
        tensor = self.quantizer.dequantize(quant_tensor, meta)
        return tensor


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




class SepCache(Cache):
    """
    A cache as described in the [SepLLM paper - ICML 2025](https://arxiv.org/abs/2412.12094). In the training phase, 
    SepLLM condenses the segment information into the KV of the separator that divides the segment. In the inference phase, the 
    corresponding SepCache only needs to store the KVs of initial tokens, separator tokens, and recent tokens for generation.

    It stores the Key and Value states as lists of tensors, two lists for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Frequently-Used Parameters:

        `init_cache_size: Union[int, List]`:
            The maximum number of KVs to be stored for initial tokens.                
                
        `sep_cache_size: Union[int, List]`:
            The maximum number of KVs to be stored for separator tokens.

        `local_size: Union[int, List]`: 
            The maximum number of KVs to be stored for local tokens (i.e., sliding window).

        `cache_size: Union[int, List]`:    
            The maximum number of KVs to be stored for all the tokens, i.e., the size for the whole KV cache.  

        Concerning these four parameters above:
            When a list is passed (its length must be `layer_num`), it represents different values for each layer. 
            When an integer is passed, it means the setting is the same for all layers.
        
        
        `separator_token_ids: List[int]`:
            The token ids of the separator tokens for the current model's tokenizer.            
            We have some examples, such as the Llama-3 series models, where setting `model_type='llama'` allows you 
                to skip setting `separator_token_ids` and `PADDING_ID` (SepCache will auto-fill them).

        `PADDING_ID: int`:
            The token id of the padding token. You can just set `PADDING_ID` to the id of "<|endoftext|>" token of the tokenizer for the pretrained model.

    Example:

        ```python        
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, SepCache
        >>> import torch
        >>> from huggingface_hub import login
        >>> login("hf_xxxXXXxxx")


        >>> def to_cuda(a_dict: dict) -> dict:
        >>>    new_dict = {}    
        >>>    for k,v in a_dict.items():
        >>>        if isinstance(v, torch.Tensor):
        >>>            new_dict[k] = v.cuda()
        >>>        else:
        >>>            new_dict[k] = v
        >>>    return new_dict

        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", attn_implementation="flash_attention_2", device_map="cuda:0")
        >>> model.bfloat16().cuda()
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        >>> inputs = tokenizer(text="My name is Llama 3", return_tensors="pt")
        >>> inputs = to_cuda(inputs)
        >>> # Prepare a cache and pass it to model's forward; `layer_num` is the number of layers for the pretrained model.
        >>> past_key_values = SepCache(init_cache_size=4, sep_cache_size=64, local_size=256, cache_size=512, layer_num=32, USE_MAX_SEP_CACHE=True, model_type='llama')
        >>> # `separator_token_ids` and `PADDING_ID` must also be provided if you are not using `model_type='llama'` like this demo.
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access SepCache filled with keys/values
        SepCache()
        ```

        ```python
        >>> ## When using the `update` function of SepCache to update the keys/values and the past token ids (necessary in SepCache), the current `input_ids` must also be provided.        
        >>> key_states, value_states = past_key_values.update(                
                    key_states = key_states,
                    value_states = value_states,    
                    input_ids = input_ids,
                    layer_idx = layer_idx,     
                    PREFILLING_FLAG = q_len > 1, ## `q_len` is the sequence length of the current `query_states`
                    )

        ```
        For detailed usage instructions, please refer to https://github.com/HKUDS/SepLLM
    """
    # is_sliding = True
    
    @staticmethod
    def slice_on_1d(x, start, end):
        return x[:, start:end, ...]
    @staticmethod
    def slice_on_2d(x, start, end):
        return x[:, :, start:end, ...]
    @staticmethod
    def slice_on_3d(x, start, end):
        return x[:, :, :, start:end, ...]


    @staticmethod
    def sep_1bat_select_on_1d(x, Bid, sep_index, min_sep_num=None, max_sep_num=None, SEP_PADDING_IN_BATCH=True):    
        """
        For the record with index `Bid` in a batch, extract the K/V states corresponding to the separators on dimension 1. 
           If `SEP_PADDING_IN_BATCH=True`, pad to the longest length (i.e. `max_sep_num`); 
           otherwise, truncate to the shortest length (i.e. `min_sep_num`). 
        """
        sep_index = sep_index.to(x.device)

        if SEP_PADDING_IN_BATCH: ## Need padding
            assert max_sep_num is not None, f"if `SEP_PADDING_IN_BATCH=True`, `max_sep_num` should not be None"
            new_x_sep =  x[Bid, sep_index, ...]   # # batch x seqlen x head x dim  -->  sep_num x head x dim  
            padding_num = max_sep_num -  new_x_sep.shape[0]
            if padding_num > 0 :
                assert padding_num <= x.shape[1], f"`padding_num` should be <= `x.shape[1]`, i.e.  x's seqlen"
                new_x_pad = x[Bid, -padding_num: , ...]    #  padding_num x head x dim     
                return torch.cat([new_x_sep, new_x_pad ] , dim=0) # max_sep_num x head x dim 
            else:
                return new_x_sep #  max_sep_num x head x dim 

        if min_sep_num is None:
            return x[Bid, sep_index, ...]  # # batch x seqlen x head x dim -->  sep_num x head x dim    
        else: ## `min_sep_num` is provided. Need truncation
            new_x =  x[Bid, sep_index, ...]   # # batch x seqlen x head x dim -->  sep_num x head x dim               
            return new_x[ :min_sep_num, ...] # #  min_sep_num x head x dim      


    @staticmethod
    def sep_1bat_select_on_2d(x, Bid, sep_index, min_sep_num=None, max_sep_num=None, SEP_PADDING_IN_BATCH=True):    
        """
        For the record with index `Bid` in a batch, extract the K/V states corresponding to the separators on dimension 2. 
           If `SEP_PADDING_IN_BATCH=True`, pad to the longest length (i.e. `max_sep_num`); 
           otherwise, truncate to the shortest length (i.e. `min_sep_num`). 
        """
        sep_index = sep_index.to(x.device)

        if SEP_PADDING_IN_BATCH: ## Need padding
            assert max_sep_num is not None, f"if `SEP_PADDING_IN_BATCH=True`, `max_sep_num` should not be None"
            new_x_sep =  x[Bid, :, sep_index, ...]   # # batch x head x seqlen x dim -->  head x sep_num x dim  
            padding_num = max_sep_num -  new_x_sep.shape[-2]
            if padding_num > 0 :
                assert padding_num<= x.shape[-2], f"`padding_num` should be <= `x.shape[-2]`, i.e.  x's seqlen"
                new_x_pad = x[Bid, :, -padding_num: , ...]    # head x padding_num x dim     
                return torch.cat([new_x_sep, new_x_pad ] , dim=-2) # head x max_sep_num x dim 
            else:
                return new_x_sep # head x max_sep_num x dim 

        if min_sep_num is None:
            return x[Bid, :, sep_index, ...]  # # batch x head x seqlen x dim -->  head x sep_num x dim    
        else: ## `min_sep_num` is provided. Need truncation
            new_x =  x[Bid, :, sep_index, ...]   # # batch x head x seqlen x dim -->  head x sep_num x dim            
            return new_x[:, :min_sep_num, ...] # #  head x min_sep_num x dim      


    @staticmethod
    def sep_1bat_select_on_3d(x, Bid, sep_index, min_sep_num=None, max_sep_num=None, SEP_PADDING_IN_BATCH=True):    
        """
        For the record with index `Bid` in a batch, extract the K/V states corresponding to the separators on dimension 3. 
           If `SEP_PADDING_IN_BATCH=True`, pad to the longest length (i.e. `max_sep_num`); 
           otherwise, truncate to the shortest length (i.e. `min_sep_num`). 
        """        
        sep_index = sep_index.to(x.device)

        if SEP_PADDING_IN_BATCH: ## Need padding
            assert max_sep_num is not None, f"if `SEP_PADDING_IN_BATCH=True`, `max_sep_num` should not be None"
            new_x_sep =  x[Bid, :, :, sep_index, ...]   # # batch x head x dim x seqlen  -->  head x dim x sep_num 
            padding_num = max_sep_num -  new_x_sep.shape[-1]
            if padding_num > 0 :
                assert padding_num <= x.shape[-1], f"`padding_num` should be <= `x.shape[-1]`, i.e.  x's seqlen"
                new_x_pad = x[Bid, :, :, -padding_num:, ...]    # head x dim x padding_num     
                return torch.cat([new_x_sep, new_x_pad] , dim=-1) # head x dim x max_sep_num 
            else:
                return new_x_sep # head x dim x max_sep_num 

        if min_sep_num is None:
            return x[Bid, :, :, sep_index, ...]  # # batch x head x dim x seqlen -->  head x dim x sep_num    
        else: ## `min_sep_num` is provided. Need truncation
            new_x =  x[Bid, :, :, sep_index, ...]   # # batch x head x dim x seqlen -->  head x dim x sep_num          
            return new_x[:, :, :min_sep_num, ...] # #  head x dim x min_sep_num       

    DIM_TO_SLICE = {
        1: slice_on_1d,
        2: slice_on_2d,
        3: slice_on_3d,
    }
    
    BAT_DIM_TO_SELECT = {
        1: sep_1bat_select_on_1d,
        2: sep_1bat_select_on_2d,
        3: sep_1bat_select_on_3d,
    }

    def __init__(self,                                                
                ## For SepLLM                                
                init_cache_size: Union[int, List] = 4,        
                sep_cache_size: Union[int, List] = 64,
                local_size: Union[int, List]=256, 
                cache_size: Union[int, List]=512,    
                SEP_ACCUMULATION: bool = True,
                USE_MAX_SEP_CACHE: bool = False,
                SEP_PADDING_IN_BATCH: bool = False,
                separator_token_ids: List[int] = None, ## required for initialization if `model_type` is not provided.
                PADDING_ID: int = None, ## required for initialization if `model_type` is not provided.

                ## For inheritance & initialization states
                past_tok_ids: List[torch.Tensor] = None,  ## It saves all the token ids corresponding to the saved KVs for all layers in SepCache.                
                key_cache: List[torch.Tensor] = None,          
                value_cache: List[torch.Tensor] = None,

                ## For debugging
                PRINT_KV_RATIO_INSIDE: bool = False,
                print_KV_inside_per_steps: int = 1000,   
                _seen_tokens: int = 0, 
                _kept_kv_ratio: List[Tuple[int]] = None,
                
                ### For positional encoding shifting
                APPLY_PE_SHIFT: bool = False,
                APPLY_PES_INSIDE: bool = True,
                _shifted_position_ids:  List[torch.Tensor] = None,
                _rope_unsqueeze_dim: int = 1, ## The unsqueeze_dim when applying RoPE.
                _rope_seq_dim: int=1, ## The seq_len dimension for the `cos` or `sin` tensors.
                pe_scaling_factor:float = 1.0,
                pe_dim:int=128, ## The number of dims for positional encoding. Typically, just set the `head_dim` to this.
                max_position_embeddings: int = 8192, 
                base: int=10000,  ## The base for RoPE.               
                
                ## For basic transformer architecture
                k_seq_dim: int=2, ## The dimension for seq_len in key tensors
                v_seq_dim: int=2, ## The dimension for seq_len in value tensors
                layer_num: int = None, ## required for initialization

                model_type: str = None,  ## The model type for running the example. choose from ['llama', 'pythia','falcon'].
                device = None          
                 ) -> None:
        """        
        `SEP_ACCUMULATION`: If True, it means we will try to accumulate all the KVs for seperators. If False, only the `new_sep_kv` compressed from the `past_win_kv` will be kept (see function `compress_kv_cache_and_tokids_layer_wise`).
                                                             
        `USE_MAX_SEP_CACHE`: If True, it means we only keep at most `self.sep_cache_size` seperators' KVs.  If the number exceeds this limit, older separator's KVs will be discarded, keeping only the most recent `self.sep_cache_size` KVs. In the paper, the hyperparameter `s` is an abbreviated alias for `self.sep_cache_size`.

        `SEP_PADDING_IN_BATCH`: If True, it means that SepCache will pad separator tokens in other records to be aligned with the record with the most separators in a batch. If False, it means that SepCache will truncate older separator tokens in other records to be aligned with the record with the fewest separators in a batch.
        
        Note: If `SEP_ACCUMULATION=True` and `USE_MAX_SEP_CACHE=False`, as the number of input tokens increases, the number of separators in the KV cache will also accumulate endlessly 
              and `self.cache_size` will also be infinitely expanded (no longer fixed).

              When `SEP_PADDING_IN_BATCH=True` is used in combination with `USE_MAX_SEP_CACHE=False` and `SEP_ACCUMULATION=True`, the KV cache will accumulate indefinitely, 
              and since `SEP_PADDING_IN_BATCH=True`, the KVs of all separators will be retained (rather than being truncated).


        For detailed usage instructions, please refer to https://github.com/HKUDS/SepLLM
        """    

        super().__init__()               
        if (key_cache is not None) or (value_cache is not None) or (past_tok_ids is not None):
            assert isinstance(key_cache, list)
            assert isinstance(value_cache, list)
            assert isinstance(past_tok_ids, list), f"For SepCache, if `key_cache` and `value_cache` are given (e.g., provided from legacy `past_key_values`), `past_tok_ids` corresponding to `key_cache` and `value_cache` must also be provided to initialize SepCache."

            assert len(key_cache) == len(past_tok_ids), f"The length of `key_cache` ({len(key_cache)}) should be equal to that of `past_tok_ids` ({len(past_tok_ids)})."
            assert len(value_cache) == len(past_tok_ids), f"The length of `value_cache` ({len(value_cache)}) should be equal to that of `past_tok_ids` ({len(past_tok_ids)})."
        assert layer_num is not None, f"`layer_num` must be provided according to the pretrained model."

        ## For basic parameters & states    
        self.key_cache: List[torch.Tensor] = key_cache if key_cache is not None else []
        self.value_cache: List[torch.Tensor] = value_cache if value_cache is not None else []    

        self.k_seq_dim = k_seq_dim ## The dimension for the seq_len in key states. Typically, 2.
        self.v_seq_dim = v_seq_dim ## The dimension for the seq_len in value states. Typically, 2.

        self.k_slice = self.DIM_TO_SLICE[k_seq_dim]
        self.v_slice = self.DIM_TO_SLICE[v_seq_dim]
        
        self.k_bat_dim_select = self.BAT_DIM_TO_SELECT[k_seq_dim]
        self.v_bat_dim_select = self.BAT_DIM_TO_SELECT[v_seq_dim]
        self._seen_tokens: int = _seen_tokens  # Used in `generate` to keep tally of how many tokens the cache has seen as well as performing statistics.
        self.layer_num =  layer_num
        self.device = device # Deprecated


        ## For debugging
        self.PRINT_KV_RATIO_INSIDE = PRINT_KV_RATIO_INSIDE
        self.print_KV_inside_per_steps = print_KV_inside_per_steps
        self._print_kv_ratio_count = 0
        self._kept_kv_ratio: List[Tuple[int]] = _kept_kv_ratio if _kept_kv_ratio is not None else []   

        ## For Streaming SepLLM
        self.past_tok_ids: List[torch.Tensor] = past_tok_ids if past_tok_ids is not None else []  ## It saves all the token ids corresponding to the saved KVs for all layers in SepCache      
        self.init_offset = None
        self._set_layer_wise_attribute("init_cache_size", init_cache_size, layer_num)
        self._set_layer_wise_attribute("local_size", local_size, layer_num)
        self._set_layer_wise_attribute("cache_size", cache_size, layer_num)
        self._set_layer_wise_attribute("sep_cache_size", sep_cache_size, layer_num)
        self._set_layer_wise_attribute("sep_exrange", 0, layer_num) # runtime right boundary for separators, excluded
        self._set_layer_wise_attribute("max_sep_exidx", self._list_element_add(self.sep_cache_size, self.init_cache_size), layer_num) # max right boundary for separators, excluded
        self.SEP_ACCUMULATION = SEP_ACCUMULATION
        self.USE_MAX_SEP_CACHE = USE_MAX_SEP_CACHE
        self.SEP_PADDING_IN_BATCH = SEP_PADDING_IN_BATCH
        

        ### For positional encoding shifting
        self.APPLY_PE_SHIFT = APPLY_PE_SHIFT
        self.APPLY_PES_INSIDE = APPLY_PES_INSIDE

        self.cos_sin_rerotation_cache = {}
        self._cos_cache = None
        self._sin_cache = None        
        self._shifted_position_ids: List[torch.Tensor] = _shifted_position_ids if _shifted_position_ids is not None else []        
        self._rope_unsqueeze_dim = _rope_unsqueeze_dim
        self._rope_seq_dim = _rope_seq_dim        

        self.pe_dim = pe_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.pe_dim, 2, dtype=torch.int64).float().to(device) / self.pe_dim))
        self.inv_freq = inv_freq
        self.pe_scaling_factor = pe_scaling_factor
        self._sin_cached = None
        self._cos_cached = None

        if model_type is None:
            assert isinstance(separator_token_ids, list), f"`separator_token_ids` must be provided for initialization unless `model_type` is given, which will auto-fiil `separator_token_ids`."
            assert len(separator_token_ids) > 0, f"`separator_token_ids` should NOT be empty."
            assert isinstance(PADDING_ID, int), f"`PADDING_ID` must be provided for initialization unless `model_type` is given, which will auto-fiil `PADDING_ID`."
            self.separator_token_ids = separator_token_ids
            self.PADDING_ID = PADDING_ID                               
        else:
            if 'llama' in  model_type.lower():
                # print("Debug: For Llama's default separators")
                self.separator_token_ids = [128000, 13, 11, 30, 0, 26, 25, 198, 220, 662, 1174, 949, 758, 2652, 551, 720, 256,262] # llama3 8b
                self.PADDING_ID = 128009
            elif ( 'pythia' in model_type.lower() ) or ( 'gpt_neox' in model_type.lower() ):
                # print("Debug: For GPTNeox's default separators")
                self.separator_token_ids = [15, 13, 32, 2, 28, 27, 209, 186, 187,    964, 1157, 3736, 2195, 3706, 1163, 2490,  50276,    586, 4928, 50275 ]       # pythia 14b
                self.PADDING_ID = 0
            elif 'falcon' in model_type.lower():
                # print(f"Debug: For Falcon's default separators")
                self.separator_token_ids = [25, 23,  42, 12, 38, 37, 193,  4610,  204, 258, 1212, 23787, 466 ]       # falcon-40b
                self.PADDING_ID = 11
            else:
                raise NotImplementedError(f"NOT implemented for the tokenizer of the backbone model type: `{model_type}`. You must provide `separator_token_ids` and `PADDING_ID` for initialization in this case! ")
        
        if APPLY_PE_SHIFT:
            print(">>>>>>>>---------#####################################################################################-----------<<<<<<<<")
            print(">>>>>>>>---------                                                                                     -----------<<<<<<<<")
            print(">>>>>>>>---------  Warning: When `APPLY_PE_SHIFT=True`, SepCache must store the key/value states       ----------<<<<<<<<")
            print(">>>>>>>>---------              before applying positional encoding (specifically RoPE)                -----------<<<<<<<<")
            print(">>>>>>>>---------#####################################################################################-----------<<<<<<<<")
                
        if APPLY_PES_INSIDE:
            print(">>>>>>>>---------#####################################################################################-----------<<<<<<<<")
            print(">>>>>>>>---------                                                                                     -----------<<<<<<<<")
            print(">>>>>>>>---------  Warning: When `APPLY_PES_INSIDE=True`, there is no need to apply rotary positional embedding--<<<<<<<<")
            print(">>>>>>>>---------  within the self_attention function, as this operation will be handled inside the `update`  ---<<<<<<<<")
            print(">>>>>>>>---------  function of SepCache. Note that `APPLY_PES_INSIDE=True` is typically used together with     ---<<<<<<<<")
            print(">>>>>>>>---------  `APPLY_PE_SHIFT=True`.                                                                     ---<<<<<<<<")
            print(">>>>>>>>---------#####################################################################################-----------<<<<<<<<")                            
            

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        input_ids: torch.Tensor,
        layer_idx: int,        
        PREFILLING_FLAG: bool = True,
        query_states: Optional[torch.Tensor] = None,        
        position_ids: Optional[torch.Tensor]=None,                
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor],Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:        
            `key_states` (`torch.Tensor`):
                The new key states to cache.
            `value_states` (`torch.Tensor`):
                The new value states to cache.
            `input_ids` (`torch.Tensor`)
                The ids of the input tokens (context tokens or autoregressive tokens)                
            `layer_idx` (`int`):
                The index of the layer to cache the states for.
            `PREFILLING_FLAG` (`bool`)
                It should be `True` at pre-filling phase and `False` when decoding

            `query_states` (`Optional[torch.Tensor]`)
                The query states that need positional encoding shifting. Only useful when `self.APPLY_PE_SHIFT=True`
            `position_ids` (`Optional[torch.Tensor]`)
                The original positional ids of the tokens in the input sequence (i.e., indices of positions of each input sequence tokens in the position embeddings)
                Only useful when `self.APPLY_PE_SHIFT=True`, i.e., SepCache will utilize `position_ids` to calculate positional shifting.
            `cache_kwargs` (`Dict[str, Any]`, optional):
                Additional arguments for the cache update. The following arguments can be used in `SepCache`: `sin`,
                `cos`, `sin_q`, `cos_q`, `shifted_pos_ids` and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the
                rotation as the tokens are shifted. (These are only useful when `self.APPLY_PE_SHIFT=True`)

                Only useful when `self.APPLY_PE_SHIFT=True` and `self.APPLY_PES_INSIDE=False`:
                    `cos` and `sin` are the shifted rotation matrices for key states
                    `cos_q` and `sin_q` are the shifted rotation matrices for query states
                    `shifted_pos_ids` is the shifted positional ids for key states
                    
                When `self.APPLY_PE_SHIFT=True` and `self.APPLY_PES_INSIDE=True`:
                    SepCache will utilize `position_ids` to calculate positional shifting.
                
                `partial_rotation_size` means that `partial_rotation_size` slices along certain dimension need to be shifted (i.e., [0, 1, ..., `partial_rotation_size-1`] slices along certain dimension)

        Return:
            A tuple containing the updated key, value, and query states (query states are optional: only applicable when `self.APPLY_PE_SHIFT=True`).

        For detailed usage instructions, please refer to https://github.com/HKUDS/SepLLM
        """

        APPLY_PE_SHIFT = self.APPLY_PE_SHIFT
        APPLY_PES_INSIDE = self.APPLY_PES_INSIDE
        SEP_ACCUMULATION = self.SEP_ACCUMULATION
        USE_MAX_SEP_CACHE = self.USE_MAX_SEP_CACHE
        SEP_PADDING_IN_BATCH = self.SEP_PADDING_IN_BATCH

        assert (self.APPLY_PE_SHIFT and (query_states is not None)) or not APPLY_PE_SHIFT, f"If `APPLY_PE_SHIFT=True`, `query_states` should be provided and it will be updated and returned"
                
        # Update the number of seen tokens
        if layer_idx == 0:
            assert key_states.shape[-2] == input_ids.shape[-1], f"`key_states.shape[-2]` ({key_states.shape[-2]}) should be equal to `input_ids.shape[-1]` ({input_ids.shape[-1]})."
            self._seen_tokens += input_ids.shape[-1]

        # [bsz, num_heads, seq_len, head_dim]
        new_kv_pair = (key_states, value_states)
                
        if (key_states.shape[self.k_seq_dim] + self.get_usable_length(layer_idx) < self.cache_size[layer_idx]) or PREFILLING_FLAG:  ## For prefilling
            assert  (PREFILLING_FLAG and key_states.shape[self.k_seq_dim] >= 1)  or (not PREFILLING_FLAG and key_states.shape[self.k_seq_dim] == 1)

            # Update cache and past token ids                
            self.update_kv_cache_and_past_tok_ids(new_kv_pair, input_ids, layer_idx, COMPRESS_KV=False, SEP_ACCUMULATION=SEP_ACCUMULATION, USE_MAX_SEP_CACHE=USE_MAX_SEP_CACHE, SEP_PADDING_IN_BATCH=SEP_PADDING_IN_BATCH)
            
            if APPLY_PE_SHIFT:                     
                shifted_keys, shifted_queries = self.apply_shifted_pos_emb(layer_idx, APPLY_PES_INSIDE, PREFILLING_FLAG, key_states, query_states, position_ids, cache_kwargs ) 
                query_states  = shifted_queries
                self.set_kv_cache( (shifted_keys, self.value_cache[layer_idx]), layer_idx)
            
            if PREFILLING_FLAG and layer_idx == 0:
                self.init_offset = self.get_initial_pos_offset(layer_idx)

            ## Count KV usage
            kv_len_ori = self.get_seq_length(layer_idx)
            kv_len_cmp = self.get_usable_length(layer_idx)
            self._update_kv_ratio(kv_len_cmp=kv_len_cmp, kv_len_ori=kv_len_ori, layer_idx=layer_idx)

        else:
            ## Update the KV cache, count KV usage, and compress the KV cache if necessary                        
            kv_len_ori = self.get_seq_length(layer_idx)
            offset_init_size_layer = self.update_kv_cache_and_past_tok_ids(new_kv_pair, input_ids, layer_idx, COMPRESS_KV=True, SEP_ACCUMULATION=SEP_ACCUMULATION, USE_MAX_SEP_CACHE=USE_MAX_SEP_CACHE, SEP_PADDING_IN_BATCH=SEP_PADDING_IN_BATCH)
            kv_len_cmp = self.get_usable_length(layer_idx)
            self._update_kv_ratio(kv_len_cmp=kv_len_cmp, kv_len_ori=kv_len_ori, layer_idx=layer_idx)
                        
            if APPLY_PE_SHIFT:                
                shifted_keys, shifted_queries = self.apply_shifted_pos_emb(layer_idx, APPLY_PES_INSIDE, PREFILLING_FLAG, key_states, query_states, position_ids, cache_kwargs )                 
                query_states  = shifted_queries
                self.set_kv_cache( (shifted_keys, self.value_cache[layer_idx]), layer_idx)
            
        if self.PRINT_KV_RATIO_INSIDE:    
            self._print_kv_ratio(layer_idx)

        if query_states is not None:
            return self.key_cache[layer_idx], self.value_cache[layer_idx], query_states
        else:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
            
    
    def update_kv_cache_and_past_tok_ids(self, new_kv_pair: Tuple[torch.Tensor], input_ids: torch.Tensor, layer_idx: int, COMPRESS_KV=False, SEP_ACCUMULATION:bool=True, USE_MAX_SEP_CACHE:bool=False, SEP_PADDING_IN_BATCH:bool=True) -> None:
        """Update the KV cache and past token ids; compress the KV cache if necessary."""
        assert layer_idx is not None, f"`layer_idx` must be given"
        assert len(new_kv_pair) == 2, f"The length of `new_kv_pair` must be 2."
        assert len(self.key_cache) == len(self.value_cache), f"The layer numbers of stored `self.key_cache` and `self.value_cache` must be the same."

        self.append_past_tok_ids(input_ids, layer_idx)

        key, value = new_kv_pair
                
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key)                        
            self.value_cache.append(value)
            assert len(self.key_cache) - 1  == layer_idx, f"The key_cache should be updated sequentially according to the layer numbering."              
            assert len(self.value_cache) - 1  == layer_idx, f"The value_cache should be updated sequentially according to the layer numbering."      
        else:            
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx] , key], dim=self.k_seq_dim)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx] , value], dim=self.v_seq_dim)

        assert len(self.key_cache) == len(self.value_cache), f"The layer numbers of stored key_cache and value_cache must be the same."
        assert self.key_cache[layer_idx].shape[self.k_seq_dim] == self.value_cache[layer_idx].shape[self.v_seq_dim], "The seq length for key_cache and value_cache must be the same."

        if COMPRESS_KV:
            cmp_past_kv_pairs, cmp_past_tok_ids, offset_init_size_layer = self.compress_kv_cache_and_tokids_layer_wise((self.key_cache[layer_idx], self.value_cache[layer_idx]), layer_idx ,SEP_ACCUMULATION=SEP_ACCUMULATION, USE_MAX_SEP_CACHE=USE_MAX_SEP_CACHE, SEP_PADDING_IN_BATCH=SEP_PADDING_IN_BATCH )
            self.set_kv_cache(cmp_past_kv_pairs, layer_idx)
            self.set_past_tok_ids(cmp_past_tok_ids, layer_idx)            
            return offset_init_size_layer
        

    def append_past_tok_ids(self, input_ids: torch.Tensor, layer_idx: int) -> None:
        """Naively append the new `input_ids` to `self.past_tok_ids[layer_idx]`"""    
        assert layer_idx is not None, f"`layer_idx` must be given"
        
        if len(self.past_tok_ids) <= layer_idx:                        
            self.past_tok_ids.append(input_ids)
            assert len(self.past_tok_ids) - 1  == layer_idx, f"The past_tok_ids should be updated sequentially according to the layer numbering."                        
        else:             
            self.past_tok_ids[layer_idx] = torch.cat([self.past_tok_ids[layer_idx] , input_ids], dim=-1)


    def compress_kv_cache_and_tokids_layer_wise(self, past_kv_pairs, layer_idx:int ,SEP_ACCUMULATION=False, USE_MAX_SEP_CACHE=False, SEP_PADDING_IN_BATCH=True ):
        """        
        `SEP_ACCUMULATION`: If True, it means we will try to accumulate all the KVs for seperators. If False, only the `new_sep_kv` compressed from the `past_win_kv` will be kept (see function `compress_kv_cache_and_tokids_layer_wise`).
                                                             
        `USE_MAX_SEP_CACHE`: If True, it means we only keep at most `self.sep_cache_size` seperators' KVs.  If the number exceeds this limit, older separator's KVs will be discarded, keeping only the most recent `self.sep_cache_size` KVs. In the paper, the hyperparameter `s` is an abbreviated alias for `self.sep_cache_size`.

        `SEP_PADDING_IN_BATCH`: If True, it means that SepCache will pad separator tokens in other records to be aligned with the record with the most separators in a batch. If False, it means that SepCache will truncate older separator tokens in other records to be aligned with the record with the fewest separators in a batch.
        
        Note: If `SEP_ACCUMULATION=True` and `USE_MAX_SEP_CACHE=False`, as the number of input tokens increases, the number of separators in the KV cache will also accumulate endlessly 
              and `self.cache_size` will also be infinitely expanded (no longer fixed).

              When `SEP_PADDING_IN_BATCH=True` is used in combination with `USE_MAX_SEP_CACHE=False` and `SEP_ACCUMULATION=True`, the KV cache will accumulate indefinitely, 
              and since `SEP_PADDING_IN_BATCH=True`, the KVs of all separators will be retained (rather than being truncated).


        For detailed usage instructions, please refer to https://github.com/HKUDS/SepLLM
        """    

        key, value = past_kv_pairs
        seq_len = key.size(self.k_seq_dim)
        assert seq_len == self.get_usable_length(layer_idx), f"The seq_len of cached past key and value states should be the same as the return of `get_usable_length()`, which is {self.get_usable_length(layer_idx)}"

        
        init_offset =  self.init_offset        
        assert init_offset is not None
        offset_init_size_layer = self.init_cache_size[layer_idx] + init_offset
        self._set_layer_wise_attribute("max_sep_exidx", self._list_element_add(self.sep_cache_size, self.init_cache_size, bias=init_offset), self.layer_num)
        self._CHECK_PARAMS_VALIDITY(layer_idx, init_offset)

        if self.sep_exrange[layer_idx] <=0:            
            self.sep_exrange[layer_idx] = offset_init_size_layer

        assert seq_len - self.local_size[layer_idx] > self.sep_exrange[layer_idx]
        
        if offset_init_size_layer > 0:                                                       
            initial_kv, initial_tokids =  self.slice_kv_cache_and_tokids( past_kv_pairs, self.past_tok_ids[layer_idx], 0, offset_init_size_layer, seq_len=seq_len, _CHECK_IDX=True )        

        Before_First_Time_Compress_Flag = (self.sep_exrange[layer_idx] == offset_init_size_layer)  ## If true, it means the present timestamp is before t1: the 1st time to compress the past window, in which only seperators' kv are kept.
        
        if SEP_ACCUMULATION and not Before_First_Time_Compress_Flag: ## To get the old sep kv and sep token ids.           
            past_sep_kv, past_sep_tokids =  self.slice_kv_cache_and_tokids( past_kv_pairs, self.past_tok_ids[layer_idx], offset_init_size_layer, self.sep_exrange[layer_idx], seq_len=seq_len, _CHECK_IDX=True )            
        
        past_win_kv, past_win_tokids =  self.slice_kv_cache_and_tokids( past_kv_pairs, self.past_tok_ids[layer_idx], self.sep_exrange[layer_idx], seq_len - self.local_size[layer_idx], seq_len=seq_len, _CHECK_IDX=True )        
        
        
        local_kv, local_tokids  =  self.slice_kv_cache_and_tokids( past_kv_pairs, self.past_tok_ids[layer_idx], seq_len - self.local_size[layer_idx], seq_len, seq_len=seq_len, _CHECK_IDX=True )
        
        new_sep_kv, new_sep_tokids, min_sep_num, max_sep_num = self.compress_past_win_2_seps( past_win_kv, past_win_tokids, SEP_PADDING_IN_BATCH = SEP_PADDING_IN_BATCH ) ## To get the new sep kv and sep token ids that were just compressed from the past window
        
        if SEP_ACCUMULATION and not Before_First_Time_Compress_Flag:  ## Try to accumulate all the seen seps           
            sep_kv, sep_tokids  = self.cat_kv_cache_and_tokids( [ past_sep_kv, new_sep_kv ] ,  [past_sep_tokids, new_sep_tokids ] )                
            new_sep_len = new_sep_tokids.shape[-1]
            sep_len = sep_tokids.shape[-1]  
        else: ## Only keep the newly obtained kv (those just compressed from the past window)
            sep_kv, sep_tokids = new_sep_kv, new_sep_tokids
            # new_sep_len = new_sep_tokids.shape[-1]
            sep_len = sep_tokids.shape[-1]            
            assert (SEP_PADDING_IN_BATCH and max_sep_num==sep_len) or ( (not SEP_PADDING_IN_BATCH) and min_sep_num==sep_len)


        if USE_MAX_SEP_CACHE: ## Fixed sep cache size, i.e., only keep max_sep_len seps' kv in the cache. 
            if offset_init_size_layer + sep_len > self.max_sep_exidx[layer_idx]:
                max_sep_len = self.max_sep_exidx[layer_idx] - offset_init_size_layer
                assert sep_kv[0].shape[-2] == sep_tokids.shape[-1], f"The seq_len for seps' KVs and tok_ids should be the same."

                sep_kv, sep_tokids =  self.slice_kv_cache_and_tokids( sep_kv, sep_tokids, sep_len-max_sep_len, sep_len, seq_len = sep_tokids.shape[-1] ,_CHECK_IDX=True )
                self.sep_exrange[layer_idx] =  self.max_sep_exidx[layer_idx]  
            else:
                self.sep_exrange[layer_idx] =  offset_init_size_layer + sep_len             

        else:    ## Extend the sep cache and the whole cache if USE_MAX_SEP_CACHE is not set                           
            self.sep_exrange[layer_idx] =  offset_init_size_layer + sep_len
            if self.sep_exrange[layer_idx] > self.max_sep_exidx[layer_idx]:                    
                cache_incremental_gap = self.sep_exrange[layer_idx] - self.max_sep_exidx[layer_idx]
                self.max_sep_exidx[layer_idx] = self.sep_exrange[layer_idx] 
                self.sep_cache_size[layer_idx] = self.sep_cache_size[layer_idx] + cache_incremental_gap
                self.cache_size[layer_idx] = self.cache_size[layer_idx] + cache_incremental_gap

        if offset_init_size_layer > 0:                                
            cmp_past_kv_pairs, cmp_past_tok_ids  = self.cat_kv_cache_and_tokids( [initial_kv, sep_kv, local_kv ] ,  [initial_tokids, sep_tokids, local_tokids  ] )
        else:
            cmp_past_kv_pairs, cmp_past_tok_ids  = self.cat_kv_cache_and_tokids( [sep_kv, local_kv ] ,  [sep_tokids, local_tokids  ] )
                
        return cmp_past_kv_pairs, cmp_past_tok_ids, offset_init_size_layer
            

    def compress_past_win_2_seps(self, past_win_kv: Tuple[torch.Tensor], past_win_tokids: torch.Tensor, MIN_SEP_ALERT: bool=False, SEP_PADDING_IN_BATCH: bool=True ) -> Tuple[Union[Tuple[torch.Tensor], torch.Tensor, int ]]:
        """Compress the KVs in the past window into the sep cache where only separators' KVs are kept. Padding or Truncating if necessary."""
        sep_index_tensor = torch.zeros_like(past_win_tokids).bool()  # batch x seq_len

        for sp_id in self.separator_token_ids:            
            sep_index_tensor = sep_index_tensor | ( past_win_tokids == sp_id ) # batch x seq_len

        sep_cnt = sep_index_tensor.int().sum(-1)
        min_sep_num = sep_cnt.min()  # the min sep number for the seqs in a batch
        max_sep_num = sep_cnt.max()  # the max sep number for the seqs in a batch

        
        if MIN_SEP_ALERT and not SEP_PADDING_IN_BATCH:
            assert min_sep_num>0, f"The min sep number for each compressing time in a batch should be at least one if `MIN_SEP_ALERT=True` and `SEP_PADDING_IN_BATCH=False`"
                
        batch1_sep_ids_list = []
        batch_size = past_win_tokids.shape[0]
        for b_id in range(batch_size):            
            batch1_sep_ids = past_win_tokids[b_id, sep_index_tensor[b_id]] # #  sep_num
            if SEP_PADDING_IN_BATCH: ## padding
                sep_num = batch1_sep_ids.shape[-1]
                padding_num =  max_sep_num - sep_num                       
                if padding_num > 0:
                    assert padding_num <= past_win_tokids.shape[-1], f"padding_num: {padding_num} should be <= past_win_tokids.shape[-1]:{past_win_tokids.shape[-1]}"
                    batch1_sep_ids = batch1_sep_ids  # #  sep_num
                    batch1_pad_ids = past_win_tokids[b_id, -padding_num:]  # #  padding_num
                    batch1_sep_ids =  torch.cat([batch1_sep_ids, batch1_pad_ids], dim =-1)   ##  max_sep_num                
            else: ## truncating
                batch1_sep_ids = batch1_sep_ids[..., :min_sep_num ]  # #  min_sep_num
            batch1_sep_ids_list.append(batch1_sep_ids)                                                           
            
        new_sep_tokids = torch.stack(batch1_sep_ids_list, dim=0) # #  B x min_sep_num
        key_cache, value_cache = past_win_kv

        assert batch_size==key_cache.shape[0]
        batch1_sep_k_list = []
        batch1_sep_v_list = []
        batch1_sep_ids_list = []
        for b_id in range(batch_size):
            batch1_sep_k = self.k_bat_dim_select(key_cache, b_id, sep_index_tensor[b_id], min_sep_num, max_sep_num, SEP_PADDING_IN_BATCH)
            batch1_sep_k_list.append(batch1_sep_k)

            batch1_sep_v = self.v_bat_dim_select(value_cache, b_id, sep_index_tensor[b_id], min_sep_num, max_sep_num, SEP_PADDING_IN_BATCH)
            batch1_sep_v_list.append( batch1_sep_v )   
        
        sep_k = torch.stack(batch1_sep_k_list, dim=0)  ## batch x head x min_sep_num x dim
        sep_v = torch.stack(batch1_sep_v_list, dim=0)  ## batch x head x min_sep_num x dim                   
        new_sep_kv = (sep_k, sep_v)

        return new_sep_kv, new_sep_tokids, min_sep_num, max_sep_num      


    def apply_shifted_pos_emb(self, layer_idx: int, APPLY_PES_INSIDE: bool, PREFILLING_FLAG: bool, key_states: torch.Tensor, query_states: torch.Tensor, position_ids: torch.Tensor, cache_kwargs: Optional[Dict[str, Any]] = None ) -> torch.Tensor:        
        """Perform positional encoding shifting if required"""
        seq_len = self.get_usable_length(layer_idx)
        keys_to_shift = self.key_cache[layer_idx]
        queries_to_shift = query_states
        assert keys_to_shift.shape[self.k_seq_dim] == seq_len
        
        if cache_kwargs is None:
            cache_kwargs = {}

        if APPLY_PES_INSIDE:           
            if len(self._shifted_position_ids) <= layer_idx:
                self._shifted_position_ids.append(None)

            if PREFILLING_FLAG: ## for prefilling
                assert position_ids.shape[-1] >= seq_len, f"The length of position_ids should be >= the usable length of kv cache when prefilling."                
                self._shifted_position_ids[layer_idx] = position_ids[:, :seq_len].detach()
                shifted_pos_ids = self._shifted_position_ids[layer_idx]

            elif self._shifted_position_ids[layer_idx].shape[-1] >= seq_len:  ## for generation
                assert position_ids.shape[-1] == 1, f"The length of query and position_ids should be 1 during generation."
                shifted_pos_ids = self._shifted_position_ids[layer_idx][:, :seq_len].detach()

            elif self._shifted_position_ids[layer_idx].shape[-1] < seq_len:   ## for generation
                assert position_ids.shape[-1] == 1, f"The length of query and position_ids should be 1 during generation."
                increased_gap = seq_len - self._shifted_position_ids[layer_idx].shape[-1]
                assert increased_gap < self._shifted_position_ids[layer_idx].shape[-1], f"Normally, for auto-regressive model, the input length for each step should be 1 during generation."

                new_position_ids = self._shifted_position_ids[layer_idx][:, -increased_gap: ] + increased_gap
                self._shifted_position_ids[layer_idx] = torch.cat([self._shifted_position_ids[layer_idx], new_position_ids.detach()], dim=-1)
                shifted_pos_ids = self._shifted_position_ids[layer_idx]
            else:
                raise RuntimeError

            cos, sin = self._get_naive_shifted_cos_sin(
                key_states, shifted_pos_ids, seq_len
            )

            q_rope_idx = torch.arange( seq_len - query_states.shape[self.k_seq_dim],  seq_len).to(cos.device)
            cos_q, sin_q = cos.index_select(self._rope_seq_dim, q_rope_idx), sin.index_select(self._rope_seq_dim, q_rope_idx)

        else:
            sin = cache_kwargs.get("sin")
            cos = cache_kwargs.get("cos")                         
            sin_q = cache_kwargs.get("sin_q")
            cos_q = cache_kwargs.get("cos_q")    
            shifted_pos_ids = cache_kwargs.get("shifted_pos_ids") 
            assert (sin is not None) and (cos is not None), f"sin and cos matrices should be be provided"
            if sin_q is None:
                q_rope_idx = torch.arange( seq_len - query_states.shape[self.k_seq_dim],  seq_len).to(sin.device)
                sin_q = sin.index_select(self._rope_seq_dim, q_rope_idx)
            if cos_q is None:
                q_rope_idx = torch.arange( seq_len - query_states.shape[self.k_seq_dim],  seq_len).to(cos.device)
                cos_q = cos.index_select(self._rope_seq_dim, q_rope_idx)
            
        partial_rotation_size = cache_kwargs.get("partial_rotation_size")
        
        # On RoPE models, we need to recompute the Key rotation as the tokens are shifted
        if partial_rotation_size is not None:
            keys_to_shift, keys_pass = (
                keys_to_shift[..., :partial_rotation_size],
                keys_to_shift[..., partial_rotation_size:]
            )
            queries_to_shift, queries_pass = (
                queries_to_shift[..., :partial_rotation_size],
                queries_to_shift[..., partial_rotation_size:]
            )
                                    
        shifted_keys = self._apply_rotary_pos_emb_single(keys_to_shift, cos, sin, shifted_pos_ids, unsqueeze_dim=self._rope_unsqueeze_dim)
        shifted_queries = self._apply_rotary_pos_emb_single(queries_to_shift, cos_q, sin_q, shifted_pos_ids[:,  -queries_to_shift.shape[self.k_seq_dim] : ], unsqueeze_dim=self._rope_unsqueeze_dim)

        if partial_rotation_size is not None:
            shifted_keys = torch.cat( [shifted_keys, keys_pass], dim=-1)
            shifted_queries = torch.cat( [shifted_queries, queries_pass], dim=-1)


        return shifted_keys, shifted_queries


    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the seen tokens. A layer index can be optionally passed."""                
        return self._seen_tokens


    def get_usable_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the actual cached states. A layer index must be passed."""         
        if len(self.key_cache) <= layer_idx :
            return 0
        assert self.key_cache[layer_idx].shape[self.k_seq_dim] == self.value_cache[layer_idx].shape[self.v_seq_dim], f"`self.key_cache` and `self.value_cache` should have the same length."        
        return self.key_cache[layer_idx].shape[self.k_seq_dim]

    def get_initial_pos_offset(self, layer_idx:int = 0) -> int:      
        """Return the number of padding tokens in the record with the most left padding tokens in a batch."""
        assert isinstance(self.PADDING_ID, int), f"`self.PADDING_ID` should be correctly set."
        assert len(self.past_tok_ids) > layer_idx, f"`self.past_tok_ids` for layer {layer_idx} must have been properly set."
                
        past_tok_ids = self.past_tok_ids[layer_idx]
        assert past_tok_ids is not None, f"`past_tok_ids` for layer {layer_idx} should not be None"

        pad_index_tensor = (past_tok_ids == self.PADDING_ID)  ## batch x seq_len
        pad_toks_cnt = pad_index_tensor.int().sum(-1)  ## [batch]
        offset = pad_toks_cnt.max().item()

        return offset

                             
    def get_batch_size(self) -> int:
        """Return the batch size."""
        assert self.key_cache is not None, f"`self.key_cache` should not be None."
        assert self.value_cache is not None, f"`self.value_cache` should not be None."
        assert len(self.key_cache) > 0, f"`self.key_cache` is empty. No batch size is available."
        assert len(self.value_cache) > 0, f"self.value_cache is empty. No batch size is available."

        assert len(self.value_cache) == len(self.key_cache), f"self.value_cache and self.key_cache should be at the same length."
        assert self.value_cache[0].shape[0] == self.key_cache[0].shape[0], f"self.value_cache and self.key_cache should have the same batch size."

        return self.value_cache[0].shape[0]

    def get_kv_pair(self, layer_idx: int = None) -> Tuple[torch.Tensor]:
        assert layer_idx is not None, f"`layer_idx` must be given."

        if (len(self.key_cache) <= layer_idx) and (len(self.value_cache) <= layer_idx ):
            key = self.key_cache[layer_idx]
            value = self.value_cache[layer_idx]
        else:
            raise RuntimeError(f"The KV for layer:{layer_idx} have not been set.")
        return (key, value)


    def set_kv_cache(self, kv_pair: Tuple , layer_idx: int ) -> None:
        assert len(kv_pair) == 2, f"The length of `kv_pair` must be 2."
        self.key_cache[layer_idx] = kv_pair[0]
        self.value_cache[layer_idx] = kv_pair[1]
    
    def set_past_tok_ids(self, tok_ids: torch.Tensor, layer_idx:int) -> None:
        self.past_tok_ids[layer_idx] = tok_ids


    def cat_kv_cache_and_tokids(self, kv_pairs_list: List[Tuple[torch.Tensor]] , tok_ids_list:List[torch.Tensor]) -> Tuple[Union[Tuple[torch.Tensor],torch.Tensor]]:
        
        return self.cat_kv_cache(kv_pairs_list), self.cat_token_ids(tok_ids_list)


    def slice_kv_cache_and_tokids(self, kv_pair:Tuple[torch.Tensor], tok_ids_list:torch.Tensor, start:int, end:int, seq_len:int=None, _CHECK_IDX:bool=True, ) -> Tuple[Union[Tuple[torch.Tensor], torch.Tensor]]:
                             
        sliced_kv = self._slice_kv(start, end,  kv_pair=kv_pair, seq_len=seq_len, _CHECK_IDX=_CHECK_IDX,)                                    
        sliced_tids = self._slice_tok_ids(start, end, tok_ids_list = tok_ids_list, seq_len=seq_len, _CHECK_IDX=_CHECK_IDX)
        
        return sliced_kv , sliced_tids


    def cat_kv_cache(self, kv_pairs_list: List[Tuple[torch.Tensor]] ) -> Tuple[torch.Tensor]:               
        assert len(kv_pairs_list) >= 1 
        
        if len(kv_pairs_list) == 1 :
            return kv_pairs_list[0]
        else:
            ret = None 
            for i, kv_pair in enumerate(kv_pairs_list): # enumerate all the KVs needed to be cat
                if i == 0:
                    ret = kv_pair
                else:
                    ret = self._cat_kv(ret, kv_pair)
            return ret


    def cat_token_ids(self, tok_ids_list:List[torch.Tensor]  ) -> torch.Tensor :
        assert len(tok_ids_list) >= 1 
        
        return torch.cat(tok_ids_list, dim=-1)     


    def _cat_kv(self, kv_pair_a:Tuple[torch.Tensor],  kv_pair_b:Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:            
        k_a, v_a = kv_pair_a
        k_b, v_b = kv_pair_b
        
        cat_k = torch.cat([k_a, k_b], dim=self.k_seq_dim)
        cat_v = torch.cat([v_a, v_b], dim=self.v_seq_dim)
        return (cat_k, cat_v)


    def _slice_kv(self, start:int, end:int, kv_pair: Tuple[torch.Tensor],   seq_len:int=None, _CHECK_IDX:bool=True)  -> Tuple[torch.Tensor] :
        assert kv_pair is not None, f"kv_pair must NOT be None when slicing it."
        key_cache = kv_pair[0]
        value_cache = kv_pair[1]

        if _CHECK_IDX:                                 
            assert seq_len is not None, f"seq_len must be given for checking the index for slicing"
            start, end = self._CHECK_IDX(start, end, seq_len)   
            
        sliced_key_cache = self.k_slice(key_cache, start, end) 
        sliced_value_cache = self.v_slice(value_cache, start, end)

        return ( sliced_key_cache, sliced_value_cache)


    def _slice_tok_ids(self, start:int, end:int, tok_ids_list:torch.Tensor , seq_len:int=None, _CHECK_IDX:bool=False) -> torch.Tensor:
        assert tok_ids_list is not None, f"tok_ids_list must NOT be None when slicing it."
        
        if _CHECK_IDX:
            assert seq_len is not None, f"seq_len must be given for checking the index for slicing"
            start, end = self._CHECK_IDX(start, end, seq_len)        
          
        sliced_tok_ids = tok_ids_list[:, start:end]
        return sliced_tok_ids

    def _set_layer_wise_attribute(self, name: str, value: Any, layer_num:int ):
        """Set layer-wise attributes"""
        if isinstance(value, int):        
            setattr(self, name, [value] * layer_num)
        elif isinstance(value, (list, tuple)):
            assert len(value) == layer_num, f"The length of {name}: {len(value)} must be equal to `layer_num`: {layer_num}"
            setattr(self, name, list(value))
        else:
            raise TypeError(f"{name} must be of the type `int` or `list` but got `{type(value)}`")

    def _list_element_add(self, list_a: List, list_b: List, bias: int=0, dtype = int, device = 'cpu') -> List:  
        """Element-wise addition between two lists."""      
        assert len(list_a) == len(list_b), f"The length of `list_a` ({len(list_a)}) must be equal to that of `list_b` ({len(list_b)})."
        tensor_c = torch.tensor(list_a, dtype=dtype, device=device) + torch.tensor(list_b, dtype=dtype, device=device) + torch.tensor([bias], dtype=dtype, device=device)
        return tensor_c.int().tolist()
        
    def _CHECK_IDX(self, start: int = 0, end: int = 100, seq_len: int = 1000):
        assert isinstance(start, int) and isinstance(end, int) and isinstance(seq_len, int), f"`start`, `end`, `seq_len` must be `int`."
        assert seq_len>0 , f"`seq_len` must > 0"
        
        if start <0 :
            start = start % seq_len
        if end < 0 :
            end = end % seq_len
        assert (start >=0) and (start < seq_len) , f"start:{start}, end:{end}, seq_len:{seq_len}"
        assert (end >= 0) and (end <= seq_len) , f"start:{start}, end:{end}, seq_len:{seq_len}"
        assert  start < end, f"start:{start}, end:{end}, seq_len:{seq_len}"

        return start,end

    def _CHECK_PARAMS_VALIDITY(self, layer_idx:int, init_offset:int):
        assert len(self.cache_size) > layer_idx
        assert len(self.init_cache_size) > layer_idx
        assert len(self.sep_cache_size) > layer_idx
        assert len(self.max_sep_exidx) > layer_idx
        assert len(self.local_size) > layer_idx

        assert self.cache_size[layer_idx] > 0 , f"`self.cache_size` for layer:{layer_idx} must be greater than 0"
        assert self.init_cache_size[layer_idx] >= 0 , f"`self.init_cache_size` for layer:{layer_idx} must be greater than (equal to) 0"
        assert self.local_size[layer_idx] > 0 , f"`self.local_size` for layer:{layer_idx} must be greater than 0"
                    
        assert self.sep_cache_size[layer_idx] > 0 , f"`self.sep_cache_size` for layer:{layer_idx} must be greater than 0"
        assert self.max_sep_exidx[layer_idx] > 0 , f"`self.max_sep_exidx` for layer:{layer_idx} must be greater than 0"
        assert self.init_cache_size[layer_idx] + self.sep_cache_size[layer_idx] + self.local_size[layer_idx] + init_offset < self.cache_size[layer_idx], f"`init_cache_size` ({self.init_cache_size[layer_idx]}) + `sep_cache_size` ({self.sep_cache_size[layer_idx]}) + `local_size` ({self.local_size[layer_idx]}) + `init_offset` ({init_offset}) for layer {layer_idx} should be less than `cache_size`:({self.cache_size[layer_idx]}) for layer {layer_idx}, i.e., a + s + w + (init_offset) < c. Please increase `cache_size` if applicable."
        


    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb_single(self, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """        
        cos = cos.unsqueeze(unsqueeze_dim)   # batch x seq_len x dim  --> batch x 1 x seq_len x dim
        sin = sin.unsqueeze(unsqueeze_dim)        
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return  k_embed


    def _get_naive_shifted_cos_sin(self, x: torch.Tensor, position_ids: torch.Tensor=None, seq_len=None):
        # x: [batch, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        # backwards compatibility
        self._cos_cached = cos
        self._sin_cached = sin
        return cos, sin
    

    def _get_scaled_shifted_cos_sin(self, x, position_ids, seq_len=None):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = self._get_naive_shifted_cos_sin(x, position_ids, seq_len)
        return cos, sin


    def _get_dynamicNTK_scaling_shifted_cos_sin(self, x, position_ids, seq_len=None):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO: this may break with compilation

        cos, sin = self._get_naive_shifted_cos_sin(x, position_ids, seq_len)
        return cos, sin


    def _update_kv_ratio(self, kv_len_cmp:int, kv_len_ori:int, layer_idx: int=0) -> None:
        """Update the KV ratios which are for statistics and debugging."""
        if len(self._kept_kv_ratio) <= layer_idx:
            self._kept_kv_ratio.append( (kv_len_cmp,  kv_len_ori ) )    
        else:
            old_kv_len_cmp = self._kept_kv_ratio[layer_idx][0]
            old_kv_len_ori = self._kept_kv_ratio[layer_idx][1]
            self._kept_kv_ratio[layer_idx] = (old_kv_len_cmp + kv_len_cmp,  old_kv_len_ori + kv_len_ori )
            
    def _print_kv_ratio(self, layer_idx : int, LAYER_WISE: bool = False):
        """Print the KV ratios."""
        self._print_kv_ratio_count += 1 
        if LAYER_WISE:
            if self._print_kv_ratio_count % self.print_KV_inside_per_steps == 0:      
                print(f"######################## [Kept Tokens, Seen Tokens] : {self._kept_kv_ratio[layer_idx]}, Ratio: { (self._kept_kv_ratio[layer_idx][0]+1e-6) / (self._kept_kv_ratio[layer_idx][1]+1e-6) :.4f} ########################")    

        elif self._print_kv_ratio_count % (self.print_KV_inside_per_steps * self.layer_num) == 0:                
            print(f"######################## [Kept Tokens, Seen Tokens] : {self._kept_kv_ratio[layer_idx]}, Ratio: { (self._kept_kv_ratio[layer_idx][0]+1e-6) / (self._kept_kv_ratio[layer_idx][1]+1e-6) :.4f} ########################")    


    @classmethod ## Deprecated
    def from_legacy_cache(cls, 
                past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,

                ## For SepLLM                                
                init_cache_size: Union[int, List] = 4,        
                sep_cache_size: Union[int, List] = 64,
                local_size: Union[int, List]=256, 
                cache_size: Union[int, List]=512,    
                SEP_ACCUMULATION: bool = True,
                USE_MAX_SEP_CACHE: bool = False,
                SEP_PADDING_IN_BATCH: bool = False,
                separator_token_ids: List[int] = None, ## required for initialization if `model_type` is not provided. set it to `[-1]` to degrade SepCache to StreamingLLM's SinkCache
                PADDING_ID: int = None, ## required for initialization if `model_type` is not provided.

                ## For inheritance & initialization states
                past_tok_ids: List[torch.Tensor] = None,  ## It saves all the token ids corresponding to the saved KVs for all layers in SepCache.                
                key_cache: List[torch.Tensor] = None,          
                value_cache: List[torch.Tensor] = None,

                ## For debugging
                PRINT_KV_RATIO_INSIDE: bool = False,
                print_KV_inside_per_steps: int = 1000,   
                _seen_tokens: int = 0, 
                _kept_kv_ratio: List[Tuple[int]] = None,
                
                ### For positional encoding shifting
                APPLY_PE_SHIFT: bool = False,
                APPLY_PES_INSIDE: bool = True,
                _shifted_position_ids:  List[torch.Tensor] = None,
                _rope_unsqueeze_dim: int = 1, ## The unsqueeze_dim when applying RoPE.
                _rope_seq_dim: int=1, ## The seq_len dimension for the `cos` or `sin` tensors.
                pe_scaling_factor:float = 1.0,
                pe_dim:int=128, ## The number of dims for positional encoding. Typically, just set the `head_dim` to this.
                max_position_embeddings: int = 8192, 
                base: int=10000,  ## The base for RoPE.               
                
                ## For basic transformer architecture
                k_seq_dim: int=2, ## The dimension for seq_len in key tensors
                v_seq_dim: int=2, ## The dimension for seq_len in value tensors
                layer_num: int = None, ## required for initialization

                model_type: str = None,  ## The model type for running the example. choose from ['llama', 'pythia','falcon'].
                device = None    
    ) -> "SepCache":
        """Deprecated: Converts a cache in the legacy cache format into `SepCache`."""                     
        if past_key_values is not None:
            key_cache = []
            value_cache = []               
            
            for i, kv in enumerate(past_key_values):
                if i == 0:
                    past_tok_ids = [] if len(kv) == 4  else past_tok_ids       

                if len(kv) == 4:
                    k, v, p_tok_ids, _seen_tokens  = kv
                    key_cache.append(k)
                    value_cache.append(v)
                    past_tok_ids.append(p_tok_ids)
                    _seen_tokens = _seen_tokens
                elif len(kv) == 2:
                    k, v = kv
                    key_cache.append(k)
                    value_cache.append(v)
                    
        cache = cls(
                ## For SepLLM                
                init_cache_size = init_cache_size,        
                sep_cache_size = sep_cache_size,
                local_size = local_size, 
                cache_size = cache_size,                    
                SEP_ACCUMULATION = SEP_ACCUMULATION,
                USE_MAX_SEP_CACHE = USE_MAX_SEP_CACHE,
                SEP_PADDING_IN_BATCH = SEP_PADDING_IN_BATCH,
                separator_token_ids = separator_token_ids,
                PADDING_ID = PADDING_ID,

                ## For inheritance & initialization states
                past_tok_ids = past_tok_ids,  ## It saves all the token ids corresponding to the saved KVs for all layers in SepCache        
                key_cache = key_cache,          
                value_cache = value_cache,

                ## For debugging
                PRINT_KV_RATIO_INSIDE = PRINT_KV_RATIO_INSIDE,
                print_KV_inside_per_steps = print_KV_inside_per_steps,   
                _seen_tokens = _seen_tokens, 
                _kept_kv_ratio = _kept_kv_ratio,
                
                ### For positional encoding shifting
                APPLY_PE_SHIFT = APPLY_PE_SHIFT,
                APPLY_PES_INSIDE = APPLY_PES_INSIDE,
                _shifted_position_ids = _shifted_position_ids,
                _rope_unsqueeze_dim = _rope_unsqueeze_dim,
                _rope_seq_dim = _rope_seq_dim, 
                pe_scaling_factor = pe_scaling_factor,
                pe_dim = pe_dim,
                max_position_embeddings = max_position_embeddings, 
                base = base,                 
                
                ## For basic transformer architecture
                k_seq_dim = k_seq_dim,
                v_seq_dim = v_seq_dim,
                layer_num = layer_num,
                
                model_type = model_type,  
                device = device,   
        )

        return cache

    
    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]: ## Deprecated
        """Deprecated: Converts the `SepCache` instance into the legacy cache format, i.e., tuple."""
        print(">>>>>>>>>>>Warnings: Please try to avoid using this deprecated `to_legacy_cache` function since it will drop many useful parameters or states in SepCache.<<<<<<<<<<<")
        legacy_cache = ()
        for layer_idx in range(len(self.key_cache)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx], self.past_tok_ids[layer_idx], self._seen_tokens), )
        return legacy_cache


    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        if self.key_cache is not None:
            return len(self.key_cache)
        else:
            return 0

    @property
    def seen_tokens(self):
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None



class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)` and `torch.export()`.

    Parameters:
        config (`PretrainedConfig`):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used. If you are manually setting the batch size, make sure to take into account the
            number of beams if you are running beam search
        max_cache_len (`int`, *optional*):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`, *optional*):
            The device on which the cache should be initialized. If you're using more than 1 computation device, you
            should pass the `layer_device_map` argument instead.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
        layer_device_map (`Optional[Dict[int, Union[str, torch.device, int]]]]`, *optional*):
            Mapping between the layers and its device. This is required when you are manually initializing the cache
            and the model is split between different gpus. You can know which layers mapped to which device by
            checking the associated device_map: `model.hf_device_map`.


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

    is_compileable = True

    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_cache_len: Optional[int] = None,
        device: Union[torch.device, str, None] = None,
        dtype: torch.dtype = torch.float32,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len

        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        self._dtype = dtype
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        # Note: There will be significant perf decrease if switching to use 5D tensors instead.
        cache_shape = (self.max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        device = torch.device(device) if device is not None else None
        for idx in range(config.num_hidden_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[idx]
            else:
                layer_device = device
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self._dtype, device=layer_device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self._dtype, device=layer_device)
            # Note: `mark_static_address` is used to tag the cache as a fixed data pointer,
            # preventing compiled graph breaks when updating the cache.
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        if cache_kwargs is None:
            cache_kwargs = {}

        key_states = key_states.to(self.key_cache[layer_idx].dtype)
        value_states = value_states.to(self.value_cache[layer_idx].dtype)
        return _static_cache_update(
            self.key_cache[layer_idx],
            self.value_cache[layer_idx],
            key_states,
            value_states,
            cache_kwargs.get("cache_position"),
        )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns (i.e. sliding_window, chunk_size),
        for each layer.
        """
        kv_length = self.get_max_cache_shape()
        return kv_length, 0


class SlidingWindowCache(StaticCache):
    """
    Sliding Window Cache class to be used with `torch.compile` for models like Mistral that support sliding window attention.
    Every time when we try to update the cache, we compute the `indices` based on `cache_position >= self.config.sliding_window - 1`,
    if true(which means the cache can not hold all the old key value states and new states together because of the sliding window constraint),
    we need to do a cycle shift based on `indices` to replace the oldest states by the new key value states passed in.

    The `to_shift` is only true once we are above sliding_window. Thus with `sliding_window==64`:

    indices = (slicing + to_shift[-1].sum()-1) % self.config.sliding_window
    tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63,  0])

    We overwrite the cache using these, then we always write at cache_position (clamped to `sliding_window`)

    Parameters:
        config (`PretrainedConfig`):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used.
        max_cache_len (`int`, *optional*):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`, *optional*):
            The device on which the cache should be initialized. If you're using more than 1 computation device, you
            should pass the `layer_device_map` argument instead.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
        layer_device_map (`Optional[Dict[int, Union[str, torch.device, int]]]]`, *optional*):
            Mapping between the layers and its device. This is required when you are manually initializing the cache
            and the model is split between different gpus. You can know which layers mapped to which device by
            checking the associated device_map: `model.hf_device_map`.

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

    is_compileable = True

    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_cache_len: Optional[int] = None,
        device: Union[torch.device, str, None] = None,
        dtype: torch.dtype = torch.float32,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        if not hasattr(config, "sliding_window") or config.sliding_window is None:
            raise ValueError(
                "Setting `cache_implementation` to 'sliding_window' requires the model config supporting "
                "sliding window attention, please check if there is a `sliding_window` field in the model "
                "config and it's not set to None."
            )
        max_cache_len = min(config.sliding_window, max_cache_len)
        self.sliding_window = config.sliding_window
        super().__init__(
            config=config,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
            layer_device_map=layer_device_map,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cache_kwargs is None:
            cache_kwargs = {}
        cache_position = cache_kwargs.get("cache_position")

        if cache_position is None:
            raise ValueError("`cache_position` must be provided for SlidingWindowCache.")

        key_states = key_states.to(self.key_cache[layer_idx].dtype)
        value_states = value_states.to(self.value_cache[layer_idx].dtype)

        return _sliding_cache_update(
            self.key_cache[layer_idx],
            self.value_cache[layer_idx],
            key_states,
            value_states,
            cache_position,
            self.max_cache_len,
        )

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len

    def reset(self):
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns (i.e. sliding_window, chunk_size),
        for each layer.
        """
        query_length = cache_position.shape[0]
        first_cache_position = cache_position[0]
        # torch.clamp() is equivalent to max() but should be compile-friendly/exportable as first_cache_position is a Tensor
        kv_offset = torch.clamp(first_cache_position - self.sliding_window + 1, min=0)
        # This is not general (see HybridChunkedCache for the whole general case), but it's what the cache returns
        kv_length = max(query_length, self.get_max_cache_shape())
        return kv_length, kv_offset


class EncoderDecoderCache(Cache):
    """
    Base, abstract class for all encoder-decoder caches. Can be used to hold combinations of self-attention and
    cross-attention caches.

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

    def __init__(self, self_attention_cache: Cache, cross_attention_cache: Cache):
        super().__init__()
        self.self_attention_cache = self_attention_cache
        self.cross_attention_cache = cross_attention_cache
        self.is_compileable = getattr(self.self_attention_cache, "is_compileable", False)

        self.is_updated = {}
        for layer_idx in range(len(cross_attention_cache.key_cache)):
            self.is_updated[layer_idx] = bool(cross_attention_cache.get_seq_length(layer_idx) > 0)

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (
                self.self_attention_cache.key_cache[layer_idx],
                self.self_attention_cache.value_cache[layer_idx],
                self.cross_attention_cache.key_cache[layer_idx],
                self.cross_attention_cache.value_cache[layer_idx],
            )
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.self_attention_cache)

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor]]:
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
        cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
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

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # check if empty list because in case of static cache it will be a tensors and we can't check `if not torch.Tensor`
        return self.self_attention_cache.get_seq_length(layer_idx)

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
        """Crop the past key values up to a new `maximum_length` in terms of tokens. `maximum_length` can also be
        negative to remove `maximum_length` tokens. This is used in assisted decoding and contrastive search."""
        self.check_dynamic_cache(self.crop.__name__)
        self.self_attention_cache.crop(maximum_length)

    def batch_split(self, full_batch_size: int, split_size: int) -> "List[EncoderDecoderCache]":
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        self.check_dynamic_cache(self.batch_split.__name__)
        self_attention_cache = self.self_attention_cache.batch_split(full_batch_size, split_size)
        cross_attention_cache = self.cross_attention_cache.batch_split(full_batch_size, split_size)

        out = []
        for self_attn, cross_attn in zip(self_attention_cache, cross_attention_cache):
            out.append(EncoderDecoderCache(self_attn, cross_attn))
        return out

    @classmethod
    def from_batch_splits(cls, splits: List["EncoderDecoderCache"]) -> "EncoderDecoderCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        self_attention_cache = DynamicCache()
        cross_attention_cache = DynamicCache()
        for idx in range(len(splits[0])):
            layer_keys = torch.cat([current.self_attention_cache.key_cache[idx] for current in splits], dim=0)
            layer_values = torch.cat([current.self_attention_cache.value_cache[idx] for current in splits], dim=0)
            self_attention_cache.update(layer_keys, layer_values, idx)

            layer_keys = torch.cat([current.cross_attention_cache.key_cache[idx] for current in splits], dim=0)
            layer_values = torch.cat([current.cross_attention_cache.value_cache[idx] for current in splits], dim=0)
            cross_attention_cache.update(layer_keys, layer_values, idx)
        return cls(self_attention_cache, cross_attention_cache)

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

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object"""
        return self.self_attention_cache.get_max_cache_shape()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns (i.e. sliding_window, chunk_size),
        for each layer.
        """
        return self.self_attention_cache.get_mask_sizes(cache_position, layer_idx)


class HybridCache(Cache):
    """
    Hybrid Cache class to be used with `torch.compile` for models that alternate between a local sliding window
    attention and global attention in every other layer (originally implemented for Gemma2).
    Under the hood, Hybrid Cache leverages ["SlidingWindowCache"] for sliding window attention and ["StaticCache"]
    for global attention.For more information, see the documentation of each subcomponent cache class.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used.
        max_cache_len (`int`, *optional*):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`, *optional*):
            The device on which the cache should be initialized. If you're using more than 1 computation device, you
            should pass the `layer_device_map` argument instead.
        dtype (torch.dtype, *optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
        layer_device_map (`Optional[Dict[int, Union[str, torch.device, int]]]]`, *optional*):
            Mapping between the layers and its device. This is required when you are manually initializing the cache
            and the model is split between different gpus. You can know which layers mapped to which device by
            checking the associated device_map: `model.hf_device_map`.

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

    is_compileable = True

    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_cache_len: Optional[int] = None,
        device: Union[torch.device, str, None] = None,
        dtype: torch.dtype = torch.float32,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        super().__init__()
        if not hasattr(config, "sliding_window") or config.sliding_window is None:
            raise ValueError(
                "Setting `cache_implementation` to 'hybrid' requires the model config supporting "
                "sliding window attention, please check if there is a `sliding_window` field in the model "
                "config and it's not set to None."
            )
        self.max_cache_len = max_cache_len if max_cache_len is not None else config.max_position_embeddings
        # Sliding layers can't be larger than the overall max cache len
        self.sliding_window_len = min(config.sliding_window, self.max_cache_len)
        self.max_batch_size = max_batch_size
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self._dtype = dtype
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )

        # If the attribute does not exist in the config, fallback to a simple StaticCache
        if hasattr(config, "layer_types"):
            self.is_sliding = [layer_type != "full_attention" for layer_type in config.layer_types]
        else:
            self.is_sliding = [False] * config.num_hidden_layers

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        global_cache_shape = (self.max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        sliding_cache_shape = (self.max_batch_size, self.num_key_value_heads, self.sliding_window_len, self.head_dim)
        self.sliding_window = min(config.sliding_window, max_cache_len)
        device = torch.device(device) if device is not None else None
        for i in range(config.num_hidden_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[i]
            else:
                layer_device = device
            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            cache_shape = sliding_cache_shape if self.is_sliding[i] else global_cache_shape
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self._dtype, device=layer_device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self._dtype, device=layer_device)
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cache_kwargs is None:
            cache_kwargs = {}
        cache_position = cache_kwargs.get("cache_position")
        if cache_position is None:
            raise ValueError("`cache_position` must be provided for HybridCache.")

        is_sliding_layer = self.is_sliding[layer_idx]

        # These two `if` blocks are only reached in multigpu and if `layer_device_map` is not passed. They are used
        # when the cache is initialized in the forward pass (e.g. Gemma2)
        if self.key_cache[layer_idx].device != key_states.device:
            self.key_cache[layer_idx] = self.key_cache[layer_idx].to(key_states.device)
        if self.value_cache[layer_idx].device != value_states.device:
            self.value_cache[layer_idx] = self.value_cache[layer_idx].to(value_states.device)

        k_cache = self.key_cache[layer_idx]
        v_cache = self.value_cache[layer_idx]
        key_states = key_states.to(k_cache.dtype)
        value_states = value_states.to(v_cache.dtype)

        if is_sliding_layer:
            return _sliding_cache_update(
                k_cache,
                v_cache,
                key_states,
                value_states,
                cache_position,
                k_cache.shape[2],  # Use actual cache dim as max cache len
            )
        else:
            return _static_cache_update(k_cache, v_cache, key_states, value_states, cache_position)

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len

    def get_seq_length(self, layer_idx: Optional[int] = 0):
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        if layer_idx != 0:
            raise ValueError(
                "`get_seq_length` on `HybridCache` may get inconsistent results depending on the layer index. "
                "Using the `layer_idx` argument is not supported."
            )
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns (i.e. sliding_window, chunk_size),
        for each layer.
        """
        if self.is_sliding[layer_idx]:
            query_length = cache_position.shape[0]
            first_cache_position = cache_position[0]

            local_mask_kv_offset = torch.clamp(first_cache_position - self.sliding_window + 1, min=0)
            # This is not general (see HybridChunkedCache for the whole general case), but it's what the cache returns
            local_mask_kv_length = max(query_length, self.sliding_window)
            return local_mask_kv_length, local_mask_kv_offset

        full_mask_kv_offset = 0
        full_mask_kv_length = self.get_max_cache_shape()
        return full_mask_kv_length, full_mask_kv_offset


class HybridChunkedCache(Cache):
    """
    Hybrid Cache class to be used with `torch.compile` for models that alternate between a local sliding window
    attention and global attention in every other layer, with support for chunked attention (originally implemented
    for Llama4).
    Under the hood, Hybrid Cache leverages ["SlidingWindowCache"] for sliding window attention and ["StaticCache"]
    for global attention. For more information, see the documentation of each subcomponent cache class.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used.
        max_cache_len (`int`, *optional*):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`, *optional*):
            The device on which the cache should be initialized. If you're using more than 1 computation device, you
            should pass the `layer_device_map` argument instead.
        dtype (torch.dtype, *optional*, defaults to `torch.bfloat16`):
            The default `dtype` to use when initializing the layer.
        layer_device_map (`Optional[Dict[int, Union[str, torch.device, int]]]]`, *optional*):
            Mapping between the layers and its device. This is required when you are manually initializing the cache
            and the model is split between different gpus. You can know which layers mapped to which device by
            checking the associated device_map: `model.hf_device_map`.

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

    is_compileable = True

    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_cache_len: Optional[int] = None,
        device: Union[torch.device, str, None] = None,
        dtype: torch.dtype = torch.bfloat16,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        super().__init__()
        if not hasattr(config, "sliding_window") or config.sliding_window is None:
            self.sliding_window = getattr(config.get_text_config(), "attention_chunk_size", 8192)
        else:
            self.sliding_window = config.sliding_window
        self.max_cache_len = max_cache_len
        # Sliding layers can't be larger than the overall max cache len
        self.sliding_window = min(self.sliding_window, self.max_cache_len)
        self.max_batch_size = max_batch_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self._dtype = dtype

        # If the attribute does not exist in the config, fallback to a simple StaticCache
        if hasattr(config, "layer_types"):
            self.is_sliding = [layer_type != "full_attention" for layer_type in config.layer_types]
        else:
            self.is_sliding = [False] * config.num_hidden_layers

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.cumulative_length = [0 for _ in range(config.num_hidden_layers)]

    def initialise_cache_layer(self, layer_idx, key_states):
        if len(self.key_cache) > layer_idx:
            return

        num_key_value_heads = key_states.shape[1]
        device = key_states.device
        global_cache_shape = (self.max_batch_size, num_key_value_heads, self.max_cache_len, self.head_dim)
        sliding_cache_shape = (self.max_batch_size, num_key_value_heads, self.sliding_window, self.head_dim)
        # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
        # breaks when updating the cache.
        cache_shape = sliding_cache_shape if self.is_sliding[layer_idx] else global_cache_shape
        new_layer_key_cache = torch.zeros(cache_shape, dtype=self._dtype, device=device)
        new_layer_value_cache = torch.zeros(cache_shape, dtype=self._dtype, device=device)
        torch._dynamo.mark_static_address(new_layer_key_cache)
        torch._dynamo.mark_static_address(new_layer_value_cache)
        self.key_cache.append(new_layer_key_cache)
        self.value_cache.append(new_layer_value_cache)

    def _sliding_update(self, cache_position, layer_idx, key_states, value_states, k_out, v_out, max_cache_len):
        cumulative_length = self.cumulative_length[layer_idx]
        # Update it now that we saved the value above
        self.cumulative_length[layer_idx] += key_states.shape[-2]
        is_full = cumulative_length >= max_cache_len
        if is_full:
            full_key_states = torch.cat((k_out[:, :, 1:, :], key_states), dim=-2)
            full_value_states = torch.cat((v_out[:, :, 1:, :], value_states), dim=-2)
            # Fast decoding path -> here as the effective size is still sliding window, it is extremely important
            # to return `self.key_cache[layer_idx]` and `self.value_cache[layer_idx]`, as they have the fixed address
            # in memory (the values are the same as the full states, but not the address!!)
            if key_states.shape[-2] == 1:
                self.key_cache[layer_idx].copy_(full_key_states)
                self.value_cache[layer_idx].copy_(full_value_states)
                return self.key_cache[layer_idx], self.value_cache[layer_idx]
        elif not is_full and cumulative_length + key_states.shape[2] > max_cache_len:
            # Fast prefill path, no need to cat() in this case (which creates a copy even if cating from 0 dim)
            if cumulative_length == 0:
                full_key_states = key_states
                full_value_states = value_states
            else:
                full_key_states = torch.cat((k_out[:, :, :cumulative_length, :], key_states), dim=-2)
                full_value_states = torch.cat((v_out[:, :, :cumulative_length, :], value_states), dim=-2)
        else:
            self.key_cache[layer_idx].index_copy_(2, cache_position, key_states)
            self.value_cache[layer_idx].index_copy_(2, cache_position, value_states)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        self.key_cache[layer_idx].copy_(full_key_states[:, :, -max_cache_len:, :])
        self.value_cache[layer_idx].copy_(full_value_states[:, :, -max_cache_len:, :])
        # we should return the whole states instead of k_out, v_out to take the whole prompt
        # into consideration when building kv cache instead of just throwing away tokens outside of the window
        return full_key_states, full_value_states

    def _static_update(self, cache_position, layer_idx, key_states, value_states, k_out, v_out, max_cache_len):
        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states

        self.key_cache[layer_idx] = k_out
        self.value_cache[layer_idx] = v_out
        return k_out, v_out

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cache_kwargs is None:
            cache_kwargs = {}
        cache_position = cache_kwargs.get("cache_position")
        self.initialise_cache_layer(layer_idx, key_states)

        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        key_states = key_states.to(k_out.dtype)
        value_states = value_states.to(v_out.dtype)

        update_fn = self._sliding_update if self.is_sliding[layer_idx] else self._static_update
        return update_fn(
            cache_position,
            layer_idx,
            key_states,
            value_states,
            k_out,
            v_out,
            k_out.shape[2],
        )

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len

    def get_seq_length(self, layer_idx: Optional[int] = 0):
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        if layer_idx != 0:
            raise ValueError(
                "`get_seq_length` on `HybridCache` may get inconsistent results depending on the layer index. "
                "Using the `layer_idx` argument is not supported."
            )
        if len(self.key_cache) == 0:
            return 0
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()
        self.cumulative_length = [0 for _ in range(len(self.cumulative_length))]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns (i.e. sliding_window, chunk_size),
        for each layer.
        """
        if self.is_sliding[layer_idx]:
            query_length = cache_position.shape[0]
            first_cache_position = cache_position[0]

            local_mask_kv_offset = torch.clamp(first_cache_position - self.sliding_window + 1, min=0)
            # This is the true general case for any Cache using local attention (sliding or chunked)
            if first_cache_position >= self.sliding_window:
                # Here the Cache is already full
                local_mask_kv_length = self.sliding_window + query_length - 1
            elif (
                first_cache_position < self.sliding_window
                and first_cache_position + query_length > self.sliding_window
            ):
                # Here the Cache becomes full with the new input
                local_mask_kv_length = first_cache_position + query_length
            else:
                # Here the Cache is still smaller than the local size, but we return the local size as it's static
                local_mask_kv_length = self.sliding_window
            return local_mask_kv_length, local_mask_kv_offset

        full_mask_kv_offset = 0
        full_mask_kv_length = self.get_max_cache_shape()
        return full_mask_kv_length, full_mask_kv_offset


class OffloadedHybridCache(HybridChunkedCache):
    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_cache_len: Optional[int] = None,
        device: Union[torch.device, str, None] = None,
        dtype: torch.dtype = torch.bfloat16,
        offload_device: Union[str, torch.device] = torch.device("cpu"),
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ):
        super().__init__(config, max_batch_size, max_cache_len, device, dtype, layer_device_map)

        # TODO (joao): to enable this cache on multiple devicesuse the pattern from `OffloadedCache`, which keeps
        # track of the original device of each layer
        unique_devices = set(layer_device_map.values()) if layer_device_map else set()
        if len(unique_devices) > 1:
            raise ValueError(f"OffloadedHybridCache does not support multiple devices. Got devices: {unique_devices}")

        self.offload_device = torch.device(offload_device)
        # Create new CUDA stream for parallel prefetching.
        self._prefetch_stream = torch.cuda.Stream() if torch._C._get_accelerator().type == "cuda" else None
        # Those will be dynamically created as the other layers (for TP)
        self.device_key_cache = None
        self.device_value_cache = None
        # This gives the index of which on-device full layer to use (we need 2 to avoid race conditions when prefetching)
        self.active_device_layer = 0

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


class MambaCache:
    """
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
        config: PretrainedConfig,
        max_batch_size: int,
        dtype: torch.dtype = torch.float16,
        device: Union[torch.device, str, None] = None,
    ):
        self.max_batch_size = max_batch_size
        self._dtype = dtype
        self.intermediate_size = config.intermediate_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel

        self.conv_states: List[torch.Tensor] = []
        self.ssm_states: List[torch.Tensor] = []
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


class OffloadedStaticCache(StaticCache):
    """
    Static cache class to be used with `torch.compile(model)` that offloads to the CPU or
    another device.

    Args:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize
            the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`Union[str, torch.device]`):
            The device on which the cache should be initialized. If you're using more than 1 computation device, you
            should pass the `layer_device_map` argument instead.
        dtype (`torch.dtype`, *optional*):
            The default `dtype` to use when initializing the cache.
        offload_device (`Union[str, torch.device]`, *optional*, defaults to `cpu`):
            The device to offload to. Defaults to CPU.
        layer_device_map (`Dict[int, Union[str, torch.device, int]]`, *optional*):
            Mapping between the layers and its device. This is required when you are manually initializing the cache
            and the model is split between different gpus. You can know which layers mapped to which device by
            checking the associated device_map: `model.hf_device_map`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, OffloadedStaticCache

        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

        >>> inputs = tokenizer(text="My name is GPT2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = OffloadedStaticCache(config=model.config, max_batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> past_kv_length = outputs.past_key_values # access cache filled with key/values from generation
        ```
    """

    is_compileable = True

    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_cache_len: Optional[int],
        device: Union[str, torch.device],
        dtype: Optional[torch.dtype] = None,
        offload_device: Union[str, torch.device] = torch.device("cpu"),
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        super(Cache, self).__init__()

        # TODO (joao): to enable this cache on multiple devicesuse the pattern from `OffloadedCache`, which keeps
        # track of the original device of each layer
        unique_devices = set(layer_device_map.values()) if layer_device_map else set()
        if len(unique_devices) > 1:
            raise ValueError(f"OffloadedStaticCache does not support multiple devices. Got devices: {unique_devices}")

        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        self.device = torch.device(device) if layer_device_map is None else torch.device(layer_device_map[0])
        self.offload_device = torch.device(offload_device)
        self._dtype = dtype if dtype is not None else torch.float32

        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads

        num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )

        cache_shape = (max_batch_size, num_key_value_heads, self.max_cache_len, head_dim)

        # Create offloaded CPU tensors.
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        for i in range(config.num_hidden_layers):
            # First layer is always on-device.
            device = self.device if i == 0 else self.offload_device

            key_cache, value_cache = self._create_key_value_cache_tensors(cache_shape, device)

            self.key_cache.append(key_cache)
            self.value_cache.append(value_cache)

        # Create device tensors.
        self._device_key_cache: List[torch.Tensor] = []
        self._device_value_cache: List[torch.Tensor] = []

        for i in range(2):
            key_cache, value_cache = self._create_key_value_cache_tensors(cache_shape, self.device)

            self._device_key_cache.append(key_cache)
            self._device_value_cache.append(value_cache)

        # For backwards compatibility.
        # TODO(gante): Remove this.
        self._seen_tokens = 0

        # Create new CUDA stream for parallel prefetching.
        self._prefetch_stream = torch.cuda.Stream() if self.device.type == "cuda" else None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. The `OffloadedStaticCache` needs the
                `cache_position` input to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """

        key_states = key_states.to(self.key_cache[layer_idx].dtype)
        value_states = value_states.to(self.value_cache[layer_idx].dtype)

        if layer_idx == 0:
            # Update seen tokens.
            # TODO(gante): Remove this.
            self._seen_tokens += key_states.shape[-2]

            # Always there.
            k_out = self.key_cache[0]
            v_out = self.value_cache[0]
        else:
            # Wait for prefetch stream.
            if self._prefetch_stream is not None:
                torch.cuda.default_stream(self.device).wait_stream(self._prefetch_stream)

            k_out = self._device_key_cache[layer_idx & 1]
            v_out = self._device_value_cache[layer_idx & 1]

        self._prefetch_layer(layer_idx + 1)

        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)

            # Copy the values to the offloaded device as well.
            if layer_idx == 0:
                self.key_cache[layer_idx].copy_(key_states.to(self.offload_device))
                self.value_cache[layer_idx].copy_(value_states.to(self.offload_device))
        else:
            # Note: here we use `tensor.index_copy_(dim, index, tensor)` that is equivalent to
            # `tensor[:, :, index] = tensor`, but the first one is compile-friendly and it does
            # explicitly an in-place operation, that avoids copies and uses less memory.
            try:
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                # The operator 'aten::index_copy.out' is not currently implemented for the MPS
                # device.
                k_out[:, :, cache_position] = key_states
                v_out[:, :, cache_position] = value_states

            # Copy the values to the offloaded device as well.
            if layer_idx != 0:
                cache_position = cache_position.to(self.offload_device)
                key_states = key_states.to(self.offload_device)
                value_states = value_states.to(self.offload_device)

                try:
                    self.key_cache[layer_idx].index_copy_(2, cache_position, key_states)
                    self.value_cache[layer_idx].index_copy_(2, cache_position, value_states)
                except NotImplementedError:
                    # The operator 'aten::index_copy.out' is not currently implemented for the MPS
                    # device.
                    self.key_cache[layer_idx][:, :, cache_position] = key_states
                    self.value_cache[layer_idx][:, :, cache_position] = value_states

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""

        # TODO(gante): Remove this.
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""

        return self.max_cache_len

    def reset(self) -> None:
        """Resets the cache values while preserving the objects."""

        # For backwards compatibility.
        # TODO(gante): Remove this.
        self._seen_tokens = 0

        # Zero out cache.
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address.
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()

    @property
    def seen_tokens(self) -> int:
        # For backwards compatibility.
        # TODO(gante): Remove this.
        return self._seen_tokens

    def _create_key_value_cache_tensors(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates K/V cache tensors on a device. Pins memory for CPU tensors. Marks them as static
        addresses for non-CPU tensors.

        Args:
            shape (`Tuple[int, ...]`): Shape.
            device (`torch.device`): Device.

        Returns:
            Key and value cache tensors as a tuple.
        """

        is_cpu_device = device == torch.device("cpu")

        key_cache = torch.zeros(shape, dtype=self._dtype, device=device, pin_memory=is_cpu_device)
        value_cache = torch.zeros(shape, dtype=self._dtype, device=device, pin_memory=is_cpu_device)

        # Note: `mark_static_address` is used to tag the cache as a fixed data pointer,
        # preventing compiled graph breaks when updating the cache.
        torch._dynamo.mark_static_address(key_cache)
        torch._dynamo.mark_static_address(value_cache)

        return key_cache, value_cache

    def _prefetch_layer(self, layer_idx: int) -> None:
        """Prefetch a layer to the device. Needs to be called in order of layer indices."""

        # Don't fetch layers that do not exist.
        if layer_idx >= len(self.key_cache):
            return

        # Alternate between two on-device caches.
        if self._prefetch_stream is not None:
            with torch.cuda.stream(self._prefetch_stream):
                self._prefetch_layer_in_context(layer_idx)
        else:
            self._prefetch_layer_in_context(layer_idx)

    def _prefetch_layer_in_context(self, layer_idx: int) -> None:
        """Performs the actual copy of the layer to device cache."""

        self._device_key_cache[layer_idx & 1].copy_(self.key_cache[layer_idx], non_blocking=True)
        self._device_value_cache[layer_idx & 1].copy_(self.value_cache[layer_idx], non_blocking=True)
