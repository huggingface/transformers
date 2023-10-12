from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, TypeVar

import torch


T = TypeVar("T")


class Cache(ABC):
    def __init__(self) -> None:
        self.key_cache: Dict[int, Tuple[torch.Tensor]] = {}
        self.value_cache: Dict[int, Tuple[torch.Tensor]] = {}

    @abstractmethod
    def update(self, key_states, value_states, layer_idx: int) -> None:
        pass

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx not in self.key_cache:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        return (
            tuple(self.key_cache[layer_idx] for layer_idx in range(len(self.key_cache))),
            tuple(self.value_cache[layer_idx] for layer_idx in range(len(self.value_cache))),
        )

    @classmethod
    def from_past_key_values(cls, past_key_values: Optional[List[torch.FloatTensor]]) -> "DynamicCache":
        if past_key_values is None:
            return cls()
        cache = cls()
        for layer_idx, (key_states, value_states) in enumerate(zip(*past_key_values)):
            cache.update(key_states, value_states, layer_idx)
        return cache

    @classmethod
    def from_past_key_value(cls, past_key_value: Optional[torch.FloatTensor]) -> "DynamicCache":
        if past_key_value is None:
            return cls()
        cache = cls()
        cache.update(past_key_value[0], past_key_value[1], 0)
        return cache


class DynamicCache(Cache):
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> None:
        if layer_idx not in self.key_cache:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class SinkCache(Cache):
    def __init__(self, window_length: int, num_sink_tokens: int) -> None:
        super().__init__()
        self.is_prefill = False
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        self.index = torch.arange(num_sink_tokens, window_length)

    def update_pre_rotation(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> None:
        # idx is either 0 for key, 1 for values
        if layer_idx not in self.key_cache:
            # first in
            sink_keys = key_states[: self.num_sink_tokens]
            sink_values = value_states[: self.num_sink_tokens]

            cached_keys = torch.cat([sink_keys, key_states[:, -self.window_length :]], dim=-1)
            cached_values = torch.cat([sink_values, value_states[:, -self.window_length :]], dim=-1)

            self.key_cache[layer_idx] = torch.cat([cached_keys[None, :], cached_values[None, :]], dim=0)
        elif key_states.shape[1] < self.index.shape[-1] + self.num_sink_tokens:
            # auto-regressive
            key_len = key_states.shape[1]

            # roll cache to the left
            self.key_cache[layer_idx]._index_copy(
                0, self.index[:key_len], self.key_cache[layer_idx][0][self.num_sink_tokens + key_len :]
            )
            self.key_cache[layer_idx]._index_copy(
                1, self.index[:key_len], self.key_cache[layer_idx][1][self.num_sink_tokens + key_len :]
            )

            # add new tokens
            self.key_cache[layer_idx]._index_copy(0, self.index[-key_len:], key_states)
            self.key_cache[layer_idx]._index_copy(1, self.index[-key_len:], value_states)
        else:
            self.key_cache[layer_idx]._index_copy(
                0, self.index, key_states[:, : self.window_length - self.num_sink_tokens]
            )
            self.key_cache[layer_idx]._index_copy(
                1, self.index, value_states[:, : self.window_length - self.num_sink_tokens]
            )

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        pass
