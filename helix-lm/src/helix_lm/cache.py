# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Decode-time state for HELIX.

This is the architecture's memory claim made concrete. Every layer holds a fixed-size recurrent state and
two fixed-size convolution states. Layers running strands L + R only cap their key/value history at the
local window, so their cache is O(1) in the context length. Only the layers whose index can actually reach
into the distant past keep the whole history -- and even there, a decode step reads just
``index_topk * block_size`` tokens of it plus ``O(log N)`` landmark probes.
"""

from __future__ import annotations

import torch

from .config import HelixConfig


class HelixLayerCache:
    """State for one layer. ``sliding_window`` is ``None`` on layers that keep the full history."""

    def __init__(self, sliding_window: int | None, number_of_states: int = 2) -> None:
        self.sliding_window = sliding_window
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        # Absolute token count, which stays right even after a windowed cache has dropped older tokens.
        self.cumulative_length = 0
        self.conv_states: dict[int, torch.Tensor | None] = dict.fromkeys(range(number_of_states))
        self.recurrent_states: dict[int, torch.Tensor | None] = {0: None}
        # Episodic memory for strand I.
        self.leaf_landmarks: torch.Tensor | None = None
        self.landmark_levels: list[torch.Tensor] = []
        self.route_hidden: torch.Tensor | None = None

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append and return the full visible history, truncating the stored copy on windowed layers."""
        if self.keys is None:
            keys, values = key_states, value_states
        else:
            keys = torch.cat([self.keys, key_states], dim=-2)
            values = torch.cat([self.values, value_states], dim=-2)
        self.cumulative_length += key_states.shape[-2]
        if self.sliding_window is None:
            self.keys, self.values = keys, values
        else:
            # Keep `sliding_window - 1`: the token being decoded next makes up the window's last slot.
            self.keys = keys[..., -self.sliding_window + 1 :, :]
            self.values = values[..., -self.sliding_window + 1 :, :]
        return keys, values

    def update_conv_state(self, conv_states: torch.Tensor, state_idx: int, conv_kernel_size: int) -> torch.Tensor:
        """Return the new states with the cached left context prepended, and keep the tail for next time."""
        cached = self.conv_states.get(state_idx)
        if cached is None:
            full = conv_states
            if full.shape[-1] < conv_kernel_size:
                full = torch.nn.functional.pad(full, (conv_kernel_size - full.shape[-1], 0))
        else:
            full = torch.cat([cached, conv_states], dim=-1)
        self.conv_states[state_idx] = full[..., -conv_kernel_size:].contiguous()
        return full

    def update_recurrent_state(self, recurrent_states: torch.Tensor, state_idx: int = 0) -> torch.Tensor:
        self.recurrent_states[state_idx] = recurrent_states
        return recurrent_states

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def reorder(self, beam_idx: torch.Tensor) -> None:
        """Reorder the batch dimension, for beam search."""
        for name in ("keys", "values", "leaf_landmarks", "route_hidden"):
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.index_select(0, beam_idx.to(tensor.device)))
        self.landmark_levels = [level.index_select(0, beam_idx.to(level.device)) for level in self.landmark_levels]
        for store in (self.conv_states, self.recurrent_states):
            for key, tensor in store.items():
                if tensor is not None:
                    store[key] = tensor.index_select(0, beam_idx.to(tensor.device))


class HelixCache:
    """One :class:`HelixLayerCache` per layer, sized from the config's per-layer schedule."""

    def __init__(self, config: HelixConfig) -> None:
        self.layers = [
            HelixLayerCache(
                sliding_window=None if layer_type == "helix" else config.sliding_window,
                number_of_states=config.number_of_conv_states,
            )
            for layer_type in config.layer_types
        ]

    def __len__(self) -> int:
        return len(self.layers)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.layers[layer_idx].get_seq_length()

    def update(self, key_states, value_states, layer_idx: int):
        return self.layers[layer_idx].update(key_states, value_states)

    def update_conv_state(self, conv_states, layer_idx: int, state_idx: int = 0, conv_kernel_size: int | None = None):
        return self.layers[layer_idx].update_conv_state(conv_states, state_idx, conv_kernel_size)

    def update_recurrent_state(self, recurrent_states, layer_idx: int, state_idx: int = 0):
        return self.layers[layer_idx].update_recurrent_state(recurrent_states, state_idx)

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        for layer in self.layers:
            layer.reorder(beam_idx)

    def byte_size(self) -> int:
        """Total bytes held. What long-context serving actually costs."""
        total = 0
        for layer in self.layers:
            tensors = [layer.keys, layer.values, layer.route_hidden, *layer.landmark_levels]
            tensors += list(layer.conv_states.values()) + list(layer.recurrent_states.values())
            total += sum(t.numel() * t.element_size() for t in tensors if t is not None)
        return total
