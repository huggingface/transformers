# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Hold the activations saved for the gradient-checkpoint recompute in host memory.

Gradient checkpointing keeps one activation per checkpointed layer on the accelerator -- `layers x sequence x
hidden` bytes -- which is the dominant term at long sequence lengths. This module moves those to pinned host
memory during the forward and brings them back during the backward, on dedicated copy streams, so the
transfers overlap compute instead of serializing against it.

Three things produce the overlap:

- The copies run on their own streams, ordered against the compute stream by events. The forward does not wait
  for a device-to-host copy, and the backward does not wait for the host-to-device copy of a tensor it is not
  using yet.
- The `keep_last_n` most recently saved activations stay resident. Backward consumes saved tensors in reverse,
  so those are the first ones it asks for, and copying them out and straight back is pure latency.
- Pulling one saved tensor starts the reload of the one saved before it, which is the next one backward asks
  for.

Requires non-reentrant checkpointing (`use_reentrant=False`, the default). Under reentrant checkpointing the
autograd graph retains each layer's outputs, so moving the inputs off the device frees nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch

from .utils import logging


logger = logging.get_logger(__name__)

# Below this, a copy costs more than the memory it reclaims: the transfer is latency-bound and the tensor is
# not what makes the activation footprint. Checkpoint boundaries at any real sequence length are far above it.
DEFAULT_MIN_OFFLOAD_BYTES = 1 << 20


@dataclass
class ActivationOffloadStats:
    """Counters for what the manager moved, for logging and for tests that assert the path was taken."""

    offloaded_tensors: int = 0
    offloaded_bytes: int = 0
    restored_tensors: int = 0
    restored_bytes: int = 0
    kept_small: int = 0
    kept_non_contiguous: int = 0
    dropped_unused: int = 0

    def reset(self) -> None:
        for field in fields(self):
            setattr(self, field.name, 0)


class _Slot:
    """One saved tensor: on the device, in flight, or in host memory, with the events that order the copies."""

    __slots__ = ("device", "dtype", "host", "device_tensor", "nbytes", "offloaded", "shape", "to_host", "to_device")

    def __init__(self, tensor: torch.Tensor):
        self.device = tensor.device
        self.dtype = tensor.dtype
        self.shape = tensor.shape
        self.nbytes = tensor.numel() * tensor.element_size()
        self.device_tensor: Optional[torch.Tensor] = tensor
        self.host: Optional[torch.Tensor] = None
        self.offloaded = False
        self.to_host = None
        self.to_device = None


class ActivationOffloadManager:
    """Moves the tensors autograd saves for backward between the accelerator and pinned host memory.

    One instance per model. Slots are packed in forward order and unpacked in reverse during backward, and the
    bookkeeping is global to the instance rather than per checkpointed layer, which is what lets a pull
    prefetch the tensor the next pull will need. Because an unpack hook travels with the tensor it was
    registered for, installing `hooks_ctx()` around each checkpointed layer still yields one order across the
    whole model.
    """

    def __init__(
        self,
        device_type: str,
        keep_last_n: int = 1,
        min_offload_bytes: int = DEFAULT_MIN_OFFLOAD_BYTES,
        use_streams: bool = True,
    ) -> None:
        self.device_type = device_type
        self.device_module = torch.get_device_module(device_type)
        self.keep_last_n = keep_last_n
        self.min_offload_bytes = min_offload_bytes
        # A backend without streams still offloads; its copies simply run where the compute runs, which is the
        # behavior `torch.autograd.graph.save_on_cpu` gives on every backend.
        self.use_streams = use_streams and hasattr(self.device_module, "Stream")
        self.stats = ActivationOffloadStats()
        self._offload_stream = None
        self._reload_stream = None
        self._slots: dict[int, _Slot] = {}
        self._order: list[int] = []
        self._order_index: dict[int, int] = {}
        self._next_id = 0

    def hooks_ctx(self):
        return torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)

    def reset_pending(self) -> None:
        """Drop slots the previous backward never asked for, so this forward starts a fresh order.

        A tensor can be packed and then pruned from the graph -- a layer whose output does not reach the loss,
        or a forward run without a backward. Its slot is never unpacked, and leaving it behind would put the
        reverse order out of step with what backward pulls.
        """
        for slot in self._slots.values():
            self.stats.dropped_unused += 1
            slot.device_tensor = None
            slot.host = None
        self._slots.clear()
        self._order.clear()
        self._order_index.clear()

    def _eligible(self, tensor: torch.Tensor) -> bool:
        if tensor.device.type != self.device_type or isinstance(tensor, torch.nn.Parameter):
            return False
        if tensor.numel() * tensor.element_size() < self.min_offload_bytes:
            self.stats.kept_small += 1
            return False
        # A non-contiguous save is a view, and its elements may alias each other; copying it into host memory
        # and back would have to reproduce the layout to stay a view of the same storage.
        if not tensor.is_contiguous():
            self.stats.kept_non_contiguous += 1
            return False
        return True

    def _pack(self, tensor: torch.Tensor):
        if not self._eligible(tensor):
            return (False, tensor)
        self._ensure_streams(tensor.device)
        slot_id = self._next_id
        self._next_id += 1
        self._slots[slot_id] = _Slot(tensor)
        self._order_index[slot_id] = len(self._order)
        self._order.append(slot_id)
        # Offload the tensor `keep_last_n` behind this one: it is now far enough from the head of the reverse
        # order that its copy has the rest of the forward to complete in.
        offload_index = len(self._order) - 1 - self.keep_last_n
        if offload_index >= 0:
            self._start_offload(self._slots[self._order[offload_index]])
        return (True, slot_id)

    def _unpack(self, payload):
        offloaded, value = payload
        return self._pull(value) if offloaded else value

    def _ensure_streams(self, device: torch.device) -> None:
        if not self.use_streams or self._offload_stream is not None:
            return
        self._offload_stream = self.device_module.Stream(device=device)
        self._reload_stream = self.device_module.Stream(device=device)

    def _pinned_like(self, slot: _Slot) -> torch.Tensor:
        return torch.empty(slot.shape, dtype=slot.dtype, pin_memory=self.device_module.is_available())

    def _start_offload(self, slot: _Slot) -> None:
        if slot.offloaded or slot.device_tensor is None:
            return
        device_tensor = slot.device_tensor
        host = self._pinned_like(slot)
        if self.use_streams:
            self._offload_stream.wait_stream(self.device_module.current_stream(device_tensor.device))
            with self.device_module.stream(self._offload_stream):
                host.copy_(device_tensor, non_blocking=True)
                slot.to_host = self.device_module.Event()
                slot.to_host.record(self._offload_stream)
            # The compute stream is free to drop its reference the moment the forward moves on; this keeps the
            # allocator from handing that block to another tensor while the copy is still reading it.
            device_tensor.record_stream(self._offload_stream)
        else:
            host.copy_(device_tensor)
        slot.host = host
        slot.device_tensor = None
        slot.offloaded = True
        self.stats.offloaded_tensors += 1
        self.stats.offloaded_bytes += slot.nbytes

    def _start_reload(self, slot: _Slot) -> None:
        if not slot.offloaded or slot.device_tensor is not None:
            return
        if not self.use_streams:
            slot.device_tensor = slot.host.to(slot.device)
            return
        device_tensor = torch.empty(slot.shape, dtype=slot.dtype, device=slot.device)
        self._reload_stream.wait_stream(self.device_module.current_stream(slot.device))
        if slot.to_host is not None:
            # The same host buffer is the destination of the outbound copy; reading it before that copy has
            # landed would return whatever it held at the time.
            self._reload_stream.wait_event(slot.to_host)
        with self.device_module.stream(self._reload_stream):
            device_tensor.record_stream(self._reload_stream)
            device_tensor.copy_(slot.host, non_blocking=True)
            slot.to_device = self.device_module.Event()
            slot.to_device.record(self._reload_stream)
        slot.device_tensor = device_tensor

    def _prefetch_previous(self, slot_id: int) -> None:
        index = self._order_index.get(slot_id)
        if index is None or index == 0:
            return
        self._start_reload(self._slots[self._order[index - 1]])

    def _pull(self, slot_id: int) -> torch.Tensor:
        slot = self._slots[slot_id]
        # Already in flight if the previous pull prefetched it; otherwise this starts the copy and waits.
        self._start_reload(slot)
        self._prefetch_previous(slot_id)
        device_tensor = slot.device_tensor
        if slot.offloaded:
            if self.use_streams and slot.to_device is not None:
                self.device_module.current_stream(slot.device).wait_event(slot.to_device)
                device_tensor.record_stream(self.device_module.current_stream(slot.device))
            slot.host = None
            self.stats.restored_tensors += 1
            self.stats.restored_bytes += slot.nbytes
        slot.device_tensor = None
        self._slots.pop(slot_id, None)
        self._drop_from_order(slot_id)
        return device_tensor

    def _drop_from_order(self, slot_id: int) -> None:
        index = self._order_index.pop(slot_id, None)
        if index is None:
            return
        last_id = self._order[-1]
        self._order[index] = last_id
        self._order_index[last_id] = index
        self._order.pop()


__all__ = ["ActivationOffloadManager", "ActivationOffloadStats", "DEFAULT_MIN_OFFLOAD_BYTES"]
