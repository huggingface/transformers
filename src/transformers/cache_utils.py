from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, TypeVar
import torch

T = TypeVar("T")


class Cache(ABC):
    def __init__(self) -> None:
        self.cache: Dict[int, Tuple[torch.Tensor]] = {}
        self.layer_idx = 0

    @abstractmethod
    def update_pre_rotation(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        pass

    @abstractmethod
    def update(self, key_states, value_states) -> None:
        pass

    def __getitem__(self, index: int):
        return self.cache[self.layer_idx][index]

    def set_layer_idx(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx

    def __bool__(self) -> bool:
        return bool(self.cache) and self.layer_idx in self.cache

class DynamicCache(Cache):
    def update_pre_rotation(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        pass

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        kv_states = torch.cat([key_states[None, :], value_states[None, :]], dim=0)
        if self.layer_idx not in self.cache:
            self.cache[self.layer_idx] = kv_states
        else:
            self.cache[self.layer_idx] = torch.cat([self.cache[self.layer_idx], kv_states], dim=-2)

    @classmethod
    def from_past_key_values(cls, past_key_values: List[torch.FloatTensor]) -> "DynamicCache":
        raise NotImplementedError()


class SinkCache(Cache):
    def __init__(self, window_length: int, num_sink_tokens: int) -> None:
        super().__init__()
        self.is_prefill = False
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        self.index = torch.arange(num_sink_tokens, window_length)

    def update_pre_rotation(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        # idx is either 0 for key, 1 for values
        if self.layer_idx not in self.cache:
            # first in
            sink_keys = key_states[: self.num_sink_tokens]
            sink_values = value_states[: self.num_sink_tokens]

            cached_keys = torch.cat([sink_keys, key_states[:, -self.window_length :]], dim=-1)
            cached_values = torch.cat([sink_values, value_states[:, -self.window_length :]], dim=-1)

            self.cache[self.layer_idx] = torch.cat([cached_keys[None, :], cached_values[None, :]], dim=0)
        elif key_states.shape[1] < self.index.shape[-1] + self.num_sink_tokens:
            # auto-regressive
            key_len = key_states.shape[1]

            # roll cache to the left
            self.cache[self.layer_idx]._index_copy(
                0, self.index[:key_len], self.cache[self.layer_idx][0][self.num_sink_tokens + key_len :]
            )
            self.cache[self.layer_idx]._index_copy(
                1, self.index[:key_len], self.cache[self.layer_idx][1][self.num_sink_tokens + key_len :]
            )

            # add new tokens
            self.cache[self.layer_idx]._index_copy(0, self.index[-key_len:], key_states)
            self.cache[self.layer_idx]._index_copy(1, self.index[-key_len:], value_states)
        else:
            self.cache[self.layer_idx]._index_copy(
                0, self.index, key_states[:, : self.window_length - self.num_sink_tokens]
            )
            self.cache[self.layer_idx]._index_copy(
                1, self.index, value_states[:, : self.window_length - self.num_sink_tokens]
            )

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        pass