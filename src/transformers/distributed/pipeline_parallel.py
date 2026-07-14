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
from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

from ..utils import is_torch_available

if TYPE_CHECKING:
    import torch.nn as nn

if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.nn as nn


class PipelineIdentityLayer(nn.Identity):
    """A placeholder layer for missing layers in a pipeline parallel model."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        """Return the first arg from args or the first value from kwargs."""
        return args[0] if args else next(iter(kwargs.values()))


class PipelineStage:
    """Pipeline-parallel stage metadata derived from a 1-D PP device mesh."""

    def __init__(
        self,
        pp_rank: int,
        pp_size: int,
        pp_group: dist.ProcessGroup,
        pp_is_first_stage: bool,
        pp_is_last_stage: bool,
        pp_prev_rank: int | None,
        pp_next_rank: int | None,
    ):
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.pp_group = pp_group
        self.pp_is_first_stage = pp_is_first_stage
        self.pp_is_last_stage = pp_is_last_stage
        self.pp_prev_rank = pp_prev_rank
        self.pp_next_rank = pp_next_rank
        self.comm_on_cpu = dist.get_backend(pp_group) == "gloo"

    @classmethod
    def from_device_mesh(cls, pp_mesh) -> PipelineStage | None:
        if pp_mesh is None:
            return None
        mesh_dim_names = getattr(pp_mesh, "mesh_dim_names", None)
        if mesh_dim_names is None or "pp" not in mesh_dim_names:
            return None
        if pp_mesh.ndim != 1 or pp_mesh.size() <= 1:
            return None

        pp_rank = pp_mesh.get_local_rank()
        pp_size = pp_mesh.size()
        pp_group = pp_mesh.get_group()
        pp_is_first_stage = pp_rank == 0
        pp_is_last_stage = pp_rank == pp_size - 1
        pp_prev_rank = pp_rank - 1 if pp_rank > 0 else None
        pp_next_rank = pp_rank + 1 if pp_rank < pp_size - 1 else None

        return cls(pp_rank, pp_size, pp_group, pp_is_first_stage, pp_is_last_stage, pp_prev_rank, pp_next_rank)

    def communicate(
        self,
        operation: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        tensor: torch.Tensor | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor | None:
        """Point-to-point pipeline communication between adjacent stages."""
        comm_device = torch.device("cpu") if self.comm_on_cpu else device
        src = dest = None

        if operation == "recv_forward":
            if self.pp_is_first_stage:
                return None
            # Receive hidden states from the previous stage.
            src = self.pp_prev_rank
            # Shape is provided by the caller (derived from input_ids / inputs_embeds).
            tensor = torch.empty(shape, dtype=dtype, device=comm_device)

        elif operation == "send_forward":
            if self.pp_is_last_stage:
                return None
            # Send hidden states to the next stage.
            dest = self.pp_next_rank
            tensor = tensor.to(device=comm_device, dtype=dtype).contiguous()

        else:
            raise ValueError(f"Unsupported pipeline communication operation: {operation}")

        # Shared P2P: one isend/irecv with the adjacent rank.
        is_send = operation.startswith("send")
        peer_rank = dest if is_send else src
        op = dist.P2POp(dist.isend if is_send else dist.irecv, tensor, peer_rank, group=self.pp_group)
        # Wait for the communication to complete.
        for req in dist.batch_isend_irecv([op]):
            req.wait()
        if comm_device.type == "cuda":
            torch.cuda.synchronize()

        return None if is_send else tensor.to(device=device, dtype=dtype)

    def layer_range_for_rank(self, rank: int, num_layers: int) -> tuple[int, int]:
        """Return [start, end) owned by rank."""
        layers_per_rank = num_layers // self.pp_size
        start_layer = rank * layers_per_rank
        end_layer = num_layers if rank == self.pp_size - 1 else start_layer + layers_per_rank
        return start_layer, end_layer

    def find_rank_for_key(self, key: str, num_layers: int, base_model_prefix: str) -> int | None:
        """Return the PP rank that owns a checkpoint parameter key, or ``None`` if unknown."""
        base_prefix = f"{base_model_prefix}."

        if key.startswith(f"{base_prefix}embed_tokens."):
            return 0

        if key.startswith(f"{base_prefix}norm.") or key.startswith("lm_head."):
            return self.pp_size - 1

        layers_prefix = f"{base_prefix}layers."
        if not key.startswith(layers_prefix):
            return None

        layer_idx = int(key.split(".")[2])
        for rank in range(self.pp_size):
            start_layer, end_layer = self.layer_range_for_rank(rank, num_layers)
            if start_layer <= layer_idx < end_layer:
                return rank

        return None

    def broadcast_from_last(
        self,
        tensor: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Broadcast a tensor from the last PP stage to every PP rank."""
        if self.pp_size <= 1:
            return tensor

        comm_device = torch.device("cpu") if self.comm_on_cpu else device
        src = self.pp_size - 1

        if self.pp_is_last_stage:
            tensor = tensor.detach().to(device=comm_device, dtype=dtype).contiguous()
            meta = torch.tensor(list(tensor.shape), dtype=torch.long, device=comm_device)
        else:
            meta = torch.empty(3, dtype=torch.long, device=comm_device)

        dist.broadcast(meta, src=src, group=self.pp_group)

        if not self.pp_is_last_stage:
            tensor = torch.empty(tuple(meta.tolist()), dtype=dtype, device=comm_device)
        dist.broadcast(tensor, src=src, group=self.pp_group)
        return tensor.to(device=device, dtype=dtype)


def apply_pipeline_parallelism(model: nn.Module, pp_mesh: torch.distributed.device_mesh.DeviceMesh) -> nn.Module:
    """Naive even split of ``base_model.layers`` across PP ranks."""
    stage = PipelineStage.from_device_mesh(pp_mesh)
    if stage is None:
        return model

    base_model = getattr(model, model.base_model_prefix)
    layers = base_model.layers
    num_layers = len(layers)

    start_layer, end_layer = stage.layer_range_for_rank(stage.pp_rank, num_layers)

    if not stage.pp_is_first_stage:
        base_model.embed_tokens = PipelineIdentityLayer()

    for layer_idx in range(num_layers):
        if layer_idx < start_layer or layer_idx >= end_layer:
            layers[layer_idx] = PipelineIdentityLayer()

    if not stage.pp_is_last_stage:
        base_model.norm = PipelineIdentityLayer()
        model.lm_head = PipelineIdentityLayer()

    _wrap_forward_for_pipeline_parallel(model)
    return model


def _hidden_states_shape(model: nn.Module, args: tuple, kwargs: dict) -> tuple[int, ...]:
    inputs_embeds = kwargs.get("inputs_embeds")
    if inputs_embeds is not None:
        return tuple(inputs_embeds.shape)

    input_ids = kwargs.get("input_ids")
    if input_ids is None and args:
        input_ids = args[0]
    if input_ids is not None:
        batch_size, sequence_length = input_ids.shape[:2]
        return (batch_size, sequence_length, model.config.hidden_size)

    raise ValueError("Cannot determine hidden states shape for pipeline recv_forward")


def _wrap_forward_for_pipeline_parallel(model: nn.Module) -> None:
    # Only wrap the forward method once.
    if getattr(model, "_pp_forward_wrapped", False):
        return

    original_forward = model.forward

    # TODO(3outeille): add different pipeline parallelism schedules.
    @wraps(original_forward)
    def pp_forward(*args, **kwargs):
        stage = PipelineStage.from_device_mesh(model._device_mesh)
        if stage is None:
            return original_forward(*args, **kwargs)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Non-first ranks: replace input_ids with activations from prev stage
        if not stage.pp_is_first_stage:
            shape = _hidden_states_shape(model, args, kwargs)
            hidden_states = stage.communicate("recv_forward", device=device, dtype=dtype, shape=shape)
            kwargs.pop("input_ids", None)
            kwargs["inputs_embeds"] = hidden_states
            if args:
                args = (None, *args[1:])

        # Run this rank's part of the model
        if stage.pp_is_last_stage:
            outputs = original_forward(*args, **kwargs)
            logits = outputs.logits
            past_key_values = outputs.past_key_values
        else:
            base_model = getattr(model, model.base_model_prefix)
            model_kwargs = {k: v for k, v in kwargs.items() if k not in {"labels", "logits_to_keep"}}
            base_outputs = base_model(*args, **model_kwargs)
            past_key_values = base_outputs.past_key_values
            logits = None
            # Non-last ranks: send activations to next stage
            stage.communicate(
                "send_forward",
                device=device,
                dtype=dtype,
                tensor=base_outputs.last_hidden_state,
            )
        # 4. Every rank gets logits from the last stage
        logits = stage.broadcast_from_last(logits, dtype=dtype, device=device)
        from ..modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    model.forward = pp_forward
    model._pp_forward_wrapped = True
