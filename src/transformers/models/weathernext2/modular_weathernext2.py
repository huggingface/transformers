# Copyright 2026 Google DeepMind and The HuggingFace Inc. team. All rights reserved.
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

"""PyTorch WeatherNext 2 model.

An encode-process-decode graph network with local mesh attention. A global noise vector conditions
the normalization layers to produce ensemble members.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_bidirectional_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, ModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..bert.modeling_bert import eager_attention_forward
from ..clip.modeling_clip import CLIPMLP
from .configuration_weathernext2 import WeatherNext2Config
from .generation_weathernext2 import WeatherNext2GenerationConfig, WeatherNext2GenerationMixin


class WeatherNext2MLP(CLIPMLP):
    pass


class WeatherNext2NoiseEncoder(nn.Linear):
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return super().forward(noise.to(dtype=self.weight.dtype))


class WeatherNext2ConditionedNorm(nn.Module):
    """LayerNorm with scale and offset derived from the global noise conditioning."""

    def __init__(self, config: WeatherNext2Config, num_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_features, eps=config.layer_norm_eps, elementwise_affine=False)
        self.linear = nn.Linear(config.noise_channels, 2 * num_features)

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        scale, offset = self.linear(conditioning).chunk(2, dim=-1)
        # Broadcast conditioning over node and optional block axes.
        broadcast_shape = (scale.shape[0], *([1] * (hidden_states.ndim - 2)), scale.shape[-1])
        return hidden_states * (1.0 + scale.view(broadcast_shape)) + offset.view(broadcast_shape)


@use_kernel_forward_from_hub("WeatherNext2ConditionedMlp")
class WeatherNext2ConditionedMlp(nn.Module):
    """Two linear projections followed by a globally conditioned LayerNorm."""

    def __init__(
        self,
        config: WeatherNext2Config,
        in_features: int,
        hidden_features: int,
        out_features: int,
        chunk_size: int | None,
    ):
        super().__init__()
        self.activation_fn = ACT2FN[config.graph_config.hidden_act]
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.norm = WeatherNext2ConditionedNorm(config, out_features)
        self.chunk_size = chunk_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        spatial_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(dtype=self.fc1.weight.dtype)
        if spatial_features is not None:
            if hidden_states.ndim == 2:
                hidden_states = hidden_states.unsqueeze(1).expand(-1, spatial_features.shape[0], -1)
            spatial_features = spatial_features.to(hidden_states).unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
            hidden_states = torch.cat([spatial_features, hidden_states], dim=-1)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return self.norm(hidden_states, conditioning)


class WeatherNext2GridToMeshEdgeUpdate(nn.Module):
    """Updates edge features using the sending grid nodes."""

    def __init__(self, config: WeatherNext2Config):
        super().__init__()
        # Apply the shared bias once, through the edge projection.
        self.edge_proj = nn.Linear(config.graph_config.edge_hidden_size, config.hidden_size)
        self.sender_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = ACT2FN[config.graph_config.hidden_act]
        self.norm = WeatherNext2ConditionedNorm(config, config.hidden_size)

    def forward(
        self,
        edge_states: torch.Tensor,
        sender_states: torch.Tensor,
        senders: torch.Tensor,
        conditioning: torch.Tensor,
        projected_senders: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if projected_senders is None:
            projected_senders = self.sender_proj(sender_states)
        messages = self.edge_proj(edge_states) + projected_senders[:, senders]
        return self.norm(self.out_proj(self.act_fn(messages)), conditioning), projected_senders


class WeatherNext2MeshToGridEdgeUpdate(nn.Module):
    """Updates edge features using sending mesh nodes and receiving grid nodes."""

    def __init__(self, config: WeatherNext2Config):
        super().__init__()
        self.edge_proj = nn.Linear(config.graph_config.edge_hidden_size, config.hidden_size)
        self.sender_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = ACT2FN[config.graph_config.hidden_act]
        self.norm = WeatherNext2ConditionedNorm(config, config.hidden_size)
        self.receiver_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        edge_states: torch.Tensor,
        sender_states: torch.Tensor,
        receiver_states: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        conditioning: torch.Tensor,
        projected_senders: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if projected_senders is None:
            projected_senders = self.sender_proj(sender_states)
        messages = self.edge_proj(edge_states) + projected_senders[:, senders]
        messages = messages + self.receiver_proj(receiver_states)[:, receivers]
        return self.norm(self.out_proj(self.act_fn(messages)), conditioning), projected_senders


@use_kernel_forward_from_hub("WeatherNext2GridToMesh")
class WeatherNext2GridToMesh(nn.Module):
    """Updates mesh nodes from receiver-sorted grid messages and applies a residual grid update."""

    def __init__(self, config: WeatherNext2Config):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        graph_config = config.graph_config
        self.edge_encoder = WeatherNext2ConditionedMlp(
            config,
            graph_config.num_edge_spatial_features,
            graph_config.edge_hidden_size,
            graph_config.edge_hidden_size,
            None,
        )
        self.aggregate_normalization = graph_config.aggregate_normalization
        self.edge_update = WeatherNext2GridToMeshEdgeUpdate(config)
        self.mesh_node_update = WeatherNext2ConditionedMlp(config, 2 * hidden_size, hidden_size, hidden_size, None)
        self.grid_node_update = WeatherNext2ConditionedMlp(
            config, hidden_size, hidden_size, hidden_size, config.chunk_size
        )

    def forward(
        self,
        grid_states: torch.Tensor,
        mesh_states: torch.Tensor,
        edge_features: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_features = edge_features.unsqueeze(0).expand(grid_states.shape[0], -1, -1)
        edge_features = edge_features.to(self.edge_encoder.fc1.weight.dtype)
        messages, _ = self.edge_update(
            self.edge_encoder(edge_features, conditioning), grid_states, senders, conditioning
        )
        offsets = torch.searchsorted(receivers, torch.arange(mesh_states.shape[1] + 1, device=receivers.device))
        indices = torch.arange(receivers.shape[0], device=receivers.device)
        aggregated = torch.stack(
            [
                nn.functional.embedding_bag(
                    indices, batch_messages.float(), offsets, mode="sum", include_last_offset=True
                )
                for batch_messages in messages
            ]
        )
        if self.aggregate_normalization is not None:
            aggregated = aggregated / self.aggregate_normalization
        mesh_states = mesh_states + self.mesh_node_update(
            torch.cat([mesh_states, aggregated.to(mesh_states.dtype)], dim=-1), conditioning
        )
        grid_states = grid_states + self.grid_node_update(grid_states, conditioning)
        return grid_states, mesh_states


@use_kernel_forward_from_hub("WeatherNext2MeshToGrid")
class WeatherNext2MeshToGrid(nn.Module):
    """Updates grid nodes from three receiver-sorted mesh messages and applies a residual mesh update."""

    def __init__(self, config: WeatherNext2Config):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        graph_config = config.graph_config
        self.edge_encoder = WeatherNext2ConditionedMlp(
            config,
            graph_config.num_edge_spatial_features,
            graph_config.edge_hidden_size,
            graph_config.edge_hidden_size,
            None,
        )
        self.edge_update = WeatherNext2MeshToGridEdgeUpdate(config)
        self.mesh_node_update = WeatherNext2ConditionedMlp(config, hidden_size, hidden_size, hidden_size, None)
        self.grid_node_update = WeatherNext2ConditionedMlp(config, 2 * hidden_size, hidden_size, hidden_size, None)

    def forward(
        self,
        grid_states: torch.Tensor,
        mesh_states: torch.Tensor,
        edge_features: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_features = edge_features.unsqueeze(0).expand(grid_states.shape[0], -1, -1)
        edge_features = edge_features.to(self.edge_encoder.fc1.weight.dtype)
        messages, _ = self.edge_update(
            self.edge_encoder(edge_features, conditioning),
            mesh_states,
            grid_states,
            senders,
            receivers,
            conditioning,
        )
        aggregated = messages.reshape(*grid_states.shape[:2], 3, -1).sum(dim=2, dtype=torch.float32)
        grid_states = grid_states + self.grid_node_update(
            torch.cat([grid_states, aggregated.to(grid_states.dtype)], dim=-1), conditioning
        )
        mesh_states = mesh_states + self.mesh_node_update(mesh_states, conditioning)
        return grid_states, mesh_states


def banded_mask_function(attention_mask: torch.Tensor) -> Callable:
    """Reads the geometry's banded mask as an index function, the form `masking_utils` expects.

    The block axis is folded into the batch axis before attention, so entry `b` of the batch
    corresponds to mesh-node block `b % num_blocks`.
    """
    num_blocks = torch.scalar_tensor(attention_mask.shape[0], dtype=torch.int64, device=attention_mask.device)
    mask = attention_mask[:, 0]

    def inner(batch_idx, head_idx, q_idx, kv_idx):
        # Select independent axes to avoid full-size integer index tensors.
        selected = mask.index_select(0, batch_idx.reshape(-1) % num_blocks)
        selected = selected.index_select(1, q_idx.reshape(-1))
        selected = selected.index_select(2, kv_idx.reshape(-1))
        return selected.unsqueeze(1) if batch_idx.ndim == 4 else selected.reshape(())

    return inner


@use_kernel_forward_from_hub("WeatherNext2Attention")
class WeatherNext2Attention(nn.Module):
    """Local mesh attention over each block and its two neighbours, restricted by the geometry mask."""

    def __init__(self, config: WeatherNext2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    @staticmethod
    def _gather_neighbouring_blocks(states: torch.Tensor) -> torch.Tensor:
        """Concatenates each block with its two neighbours, zero-padded at the ends."""
        padding = torch.zeros_like(states[:, :1])
        padded = torch.cat([padding, states, padding], dim=1)
        return torch.cat([padded[:, :-2], padded[:, 1:-1], padded[:, 2:]], dim=3)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Attend over neighbouring blocks of mesh nodes."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # [batch, blocks, block, hidden] -> [batch, blocks, heads, block, head_dim]
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(2, 3)
        key_states = self._gather_neighbouring_blocks(self.k_proj(hidden_states).view(hidden_shape).transpose(2, 3))
        value_states = self._gather_neighbouring_blocks(self.v_proj(hidden_states).view(hidden_shape).transpose(2, 3))

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        # Fold the block axis into the batch axis and upcast, as in the original implementation.
        query_states = query_states.reshape(-1, *query_states.shape[-3:]).float()
        key_states = key_states.reshape(-1, *key_states.shape[-3:]).float()
        value_states = value_states.reshape(-1, *value_states.shape[-3:]).float()

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # Attention runs in fp32; restore the model dtype before the output projection.
        attn_output = attn_output.to(hidden_states.dtype).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class WeatherNext2Layer(GradientCheckpointingLayer):
    def __init__(self, config: WeatherNext2Config, layer_idx: int):
        super().__init__()
        self.self_attn = WeatherNext2Attention(config, layer_idx)
        self.mlp = WeatherNext2MLP(config)
        self.input_layernorm = WeatherNext2ConditionedNorm(config, config.hidden_size)
        self.post_attention_layernorm = WeatherNext2ConditionedNorm(config, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        conditioning: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states, conditioning)
        hidden_states, _ = self.self_attn(hidden_states, attention_mask, **kwargs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states, conditioning)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class WeatherNext2PreTrainedModel(PreTrainedModel):
    config: WeatherNext2Config
    base_model_prefix = "model"
    main_input_name = "grid_features"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "WeatherNext2ConditionedNorm",
        "WeatherNext2ConditionedMlp",
        "WeatherNext2GridToMesh",
        "WeatherNext2MeshToGrid",
        "WeatherNext2Layer",
        "WeatherNext2ForecastHead",
    ]
    _supports_sdpa = True
    _supports_flex_attn = True
    # Flash attention cannot take an arbitrary mask, and mesh adjacency is one.
    _supports_flash_attn = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "attentions": WeatherNext2Attention,
        "hidden_states": WeatherNext2Layer,
    }

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, WeatherNext2Geometry):
            for name, value in module.get_initial_geometry().items():
                init.copy_(getattr(module, name), value)
        elif isinstance(module, WeatherNext2ForecastHead):
            gate, shifts = module.get_output_activation_buffers()
            init.copy_(module.sigmoid_gate, gate)
            init.copy_(module.sigmoid_shift, shifts)


@use_kernel_forward_from_hub("WeatherNext2MeshEncoder")
class WeatherNext2MeshEncoder(WeatherNext2PreTrainedModel):
    """The processor: a stack of pre-norm blocks over the mesh nodes."""

    main_input_name = "hidden_states"
    _tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.fc1": "colwise",
        "layers.*.mlp.fc2": "rowwise",
    }

    def __init__(self, config: WeatherNext2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [WeatherNext2Layer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = WeatherNext2ConditionedNorm(config, config.hidden_size)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | dict,
        conditioning: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        hidden_states (`torch.FloatTensor` of shape `(batch_size, num_mesh_nodes, hidden_size)`):
            Encoded mesh-node features.
        attention_mask (`torch.BoolTensor` of shape `(num_blocks, 1, block_size, 3 * block_size)` or `dict`):
            Banded mesh adjacency shared by all ensemble members, or a precomputed `"full_attention"` mask.
        conditioning (`torch.FloatTensor` of shape `(batch_size, noise_channels)`):
            Encoded global noise for each ensemble member.
        """
        batch_size, num_nodes, hidden_size = hidden_states.shape
        if isinstance(attention_mask, dict):
            block_size = min(self.config.attention_bandwidth, num_nodes)
            num_blocks = (num_nodes + block_size - 1) // block_size
        else:
            num_blocks, _, block_size, _ = attention_mask.shape

        hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, num_blocks * block_size - num_nodes))
        hidden_states = hidden_states.view(batch_size, num_blocks, block_size, hidden_size)

        if not isinstance(attention_mask, dict):
            attention_mask = {
                "full_attention": create_bidirectional_mask(
                    self.config,
                    hidden_states.flatten(0, 1),
                    None,
                    encoder_hidden_states=hidden_states.new_empty((batch_size * num_blocks, 3 * block_size, 0)),
                    and_mask_function=banded_mask_function(attention_mask),
                )
            }

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask["full_attention"], conditioning, **kwargs)

        hidden_states = hidden_states.reshape(batch_size, num_blocks * block_size, hidden_size)
        return BaseModelOutput(last_hidden_state=self.norm(hidden_states[:, :num_nodes], conditioning))


@auto_docstring(custom_intro="Latent representation of the atmosphere on the lat/lon grid.")
@dataclass
class WeatherNext2ModelOutput(ModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_grid_points, hidden_size)`):
        Grid-point features after the mesh-to-grid graph network.
    mesh_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_mesh_nodes, hidden_size)`):
        Mesh-node features after the transformer.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*):
        Mesh-node features entering the transformer and leaving each of its layers, so `num_hidden_layers + 1`
        entries. Attention runs over blocks of neighbouring mesh nodes, so these are shaped
        `(batch_size, num_blocks, block_size, hidden_size)` and the tail of the last block is padding.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
        Attention weights for each mesh-transformer layer, shaped
        `(batch_size * num_blocks, num_attention_heads, block_size, 3 * block_size)`.
        Each query block attends to its preceding, current, and following blocks.
    """

    last_hidden_state: torch.FloatTensor = None
    mesh_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring(custom_intro="A single forecast step.")
@dataclass
class WeatherNext2ForecastOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `()`, *optional*, returned when `labels` is provided):
        Latitude-, pressure- and variable-weighted unbiased ensemble CRPS, averaged over initial conditions.
    prediction (`torch.FloatTensor` of shape `(batch_size, num_output_channels, num_latitudes, num_longitudes)`):
        Predicted state for the next time step, in the model's normalized space, with channels ordered as in
        [`WeatherNext2Config.target_channel_layout`]. For variables that are also inputs this is a normalized
        *residual* on the last input frame; for the others it is the normalized value itself. Use
        [`WeatherNext2FeatureExtractor.postprocess`] to get physical units.
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_grid_points, hidden_size)`):
        Grid-point features the prediction was decoded from.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*):
        Mesh-node features after each transformer layer, shaped as in [`WeatherNext2ModelOutput`].
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
        Mesh-transformer attention weights, shaped as in [`WeatherNext2ModelOutput`].
    """

    loss: torch.FloatTensor | None = None
    prediction: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class WeatherNext2Geometry(nn.Module):
    """Exact checkpoint geometry, stored as persistent buffers."""

    def __init__(self, config: WeatherNext2Config):
        super().__init__()
        self.config = config
        for name, value in self.get_initial_geometry().items():
            self.register_buffer(name, value)

    def get_initial_geometry(self) -> dict[str, torch.Tensor]:
        """Builds placeholder geometry for a randomly initialized model, without reconstructing the mesh."""
        config = self.config
        edges = config.num_grid_to_mesh_edges
        mesh_to_grid_edges = 3 * config.num_grid_points
        shapes = {
            "grid_spatial_features": ((config.num_grid_points, config.num_node_spatial_features), torch.float32),
            "mesh_spatial_features": ((config.num_mesh_nodes, config.num_node_spatial_features), torch.float32),
            "grid_to_mesh_senders": ((edges,), torch.int64),
            "grid_to_mesh_receivers": ((edges,), torch.int64),
            "grid_to_mesh_edge_features": ((edges, config.graph_config.num_edge_spatial_features), torch.float32),
            "mesh_to_grid_senders": ((mesh_to_grid_edges,), torch.int64),
            "mesh_to_grid_receivers": ((mesh_to_grid_edges,), torch.int64),
            "mesh_to_grid_edge_features": (
                (mesh_to_grid_edges, config.graph_config.num_edge_spatial_features),
                torch.float32,
            ),
        }
        values = {name: torch.zeros(shape, dtype=dtype) for name, (shape, dtype) in shapes.items()}
        values["mesh_to_grid_receivers"] = torch.arange(config.num_grid_points).repeat_interleave(3)
        return values

    def forward(self) -> dict[str, torch.Tensor]:
        return dict(self.named_buffers(recurse=False))


@auto_docstring
class WeatherNext2Model(WeatherNext2PreTrainedModel):
    def __init__(self, config: WeatherNext2Config):
        super().__init__(config)
        hidden_size = config.hidden_size

        self.noise_encoder = WeatherNext2NoiseEncoder(config.noise_channels, config.noise_channels, bias=False)
        self.grid_encoder = WeatherNext2ConditionedMlp(
            config, config.num_grid_input_channels, hidden_size, hidden_size, config.chunk_size
        )
        self.mesh_encoder = WeatherNext2ConditionedMlp(
            config, config.num_mesh_input_channels, hidden_size, hidden_size, None
        )
        self.grid_to_mesh = WeatherNext2GridToMesh(config)
        self.mesh_transformer = WeatherNext2MeshEncoder(config)
        self.mesh_to_grid = WeatherNext2MeshToGrid(config)
        self.geometry = WeatherNext2Geometry(config)
        self.post_init()

    @merge_with_config_defaults
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        grid_features: torch.Tensor,
        global_features: torch.Tensor,
        noise: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> WeatherNext2ModelOutput:
        r"""
        grid_features (`torch.FloatTensor` of shape `(batch_size, num_channels, num_latitudes, num_longitudes)`):
            Normalized input fields, with channels ordered as in [`WeatherNext2Config.input_channel_layout`].
            Variables with no spatial extent must already be broadcast over the grid.
        global_features (`torch.FloatTensor` of shape `(batch_size, num_global_channels)`):
            Normalized values of the variables with no spatial extent, ordered as in
            [`WeatherNext2Config.mesh_channel_layout`]. These reach the mesh encoder directly, which is how the
            mesh learns anything about the state of the calendar.
        noise (`torch.FloatTensor` of shape `(batch_size, noise_channels)`):
            One standard normal draw per ensemble member.
        attention_mask (`torch.BoolTensor` of shape `(num_blocks, 1, block_size, 3 * block_size)`):
            Exact banded mesh geometry returned by [`WeatherNext2FeatureExtractor`], shared by all ensemble members.
        """
        conditioning = self.noise_encoder(noise)
        geometry = self.geometry()

        grid_states = self.grid_encoder(
            grid_features.flatten(2).transpose(1, 2),
            conditioning,
            spatial_features=geometry["grid_spatial_features"],
        )
        mesh_states = self.mesh_encoder(
            global_features, conditioning, spatial_features=geometry["mesh_spatial_features"]
        )

        grid_states, mesh_states = self.grid_to_mesh(
            grid_states,
            mesh_states,
            geometry["grid_to_mesh_edge_features"],
            geometry["grid_to_mesh_senders"],
            geometry["grid_to_mesh_receivers"],
            conditioning,
        )
        mesh_outputs = self.mesh_transformer(mesh_states, attention_mask, conditioning, return_dict=True, **kwargs)
        mesh_states = mesh_outputs.last_hidden_state
        grid_states, mesh_states = self.mesh_to_grid(
            grid_states,
            mesh_states,
            geometry["mesh_to_grid_edge_features"],
            geometry["mesh_to_grid_senders"],
            geometry["mesh_to_grid_receivers"],
            conditioning,
        )

        return WeatherNext2ModelOutput(
            last_hidden_state=grid_states,
            mesh_hidden_state=mesh_states,
            hidden_states=mesh_outputs.hidden_states,
            attentions=mesh_outputs.attentions,
        )


@use_kernel_forward_from_hub("WeatherNext2ForecastHead")
class WeatherNext2ForecastHead(nn.Module):
    """Decodes grid-point features into the predicted state, as `[batch, channels, lat, lon]`."""

    def __init__(self, config: WeatherNext2Config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.graph_config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.num_output_channels)
        gate, shifts = self.get_output_activation_buffers()
        self.sigmoid_gate = nn.Buffer(gate, persistent=False)
        self.sigmoid_shift = nn.Buffer(shifts, persistent=False)

    def get_output_activation_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Builds the output-channel sigmoid gates and shifts from the config."""
        shifts = torch.zeros(self.config.num_output_channels)
        gate = torch.zeros(self.config.num_output_channels, dtype=torch.bool)
        offset = 0
        for variable, _, levels in self.config.target_channel_layout:
            if variable in (self.config.sigmoid_shifted_outputs or {}):
                gate[offset : offset + levels] = True
                shifts[offset : offset + levels] = self.config.sigmoid_shifted_outputs[variable]
            offset += levels
        return gate, shifts

    def forward(self, grid_states: torch.Tensor) -> torch.Tensor:
        grid_states = self.fc1(grid_states)
        grid_states = self.activation_fn(grid_states)
        prediction = self.fc2(grid_states)
        prediction = torch.where(self.sigmoid_gate, torch.sigmoid(prediction - self.sigmoid_shift), prediction)
        return prediction.transpose(1, 2).reshape(
            prediction.shape[0], -1, self.config.grid_latitudes, self.config.grid_longitudes
        )


@auto_docstring(
    custom_intro="WeatherNext 2 with its forecasting head: advances the global atmospheric state by one time step."
)
class WeatherNext2ForWeatherForecasting(WeatherNext2PreTrainedModel, WeatherNext2GenerationMixin):
    generation_config_class = WeatherNext2GenerationConfig

    def __init__(self, config: WeatherNext2Config):
        super().__init__(config)
        self.model = WeatherNext2Model(config)
        self.head = WeatherNext2ForecastHead(config)
        self.post_init()

    def _crps_loss(self, prediction: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """The original FGN CRPS on normalized targets, with the sample axis unflattened."""
        if self.config.loss_weights is None:
            raise ValueError("Set `config.loss_weights` from the original checkpoint before computing the loss.")
        if labels.ndim != 5 or prediction.shape != (labels.shape[0] * labels.shape[1], *labels.shape[2:]):
            raise ValueError(
                "`labels` must have shape (batch_size, num_members, channels, latitudes, longitudes), "
                "matching the flattened prediction batch."
            )
        num_members = labels.shape[1]
        if num_members < 2:
            raise ValueError("Unbiased ensemble CRPS requires at least two members per initial condition.")
        prediction = prediction.float().reshape(labels.shape)
        labels = labels.to(device=prediction.device, dtype=torch.float32)
        # Fix the target across members, including residuals from different initial states.
        shift = labels - labels.nanmean(dim=1, keepdim=True)
        targets = labels - shift
        prediction = (prediction - shift).masked_fill(targets.isnan(), float("nan"))

        num_latitudes = prediction.shape[-2]
        if num_latitudes < 2:
            raise ValueError("Latitude weighting requires at least two latitudes, including both poles.")
        latitudes = torch.linspace(-90, 90, num_latitudes, device=prediction.device)
        half_spacing = math.pi / (2 * (num_latitudes - 1))
        latitude_weights = torch.cos(torch.deg2rad(latitudes)) * math.sin(half_spacing)
        latitude_weights[0] = latitude_weights[-1] = math.sin(half_spacing / 2) ** 2
        latitude_weights = (latitude_weights / latitude_weights.mean()).view(1, 1, 1, -1, 1)

        level_weights, variable_weights, mask_nan = [], [], []
        for variable, _, levels in self.config.target_channel_layout:
            pressure = self.config.pressure_levels if variable in self.config.atmospheric_variables else [1]
            mean_pressure = sum(pressure) / len(pressure)
            level_weights.extend(level / mean_pressure for level in pressure)
            variable_weights.extend([self.config.loss_weights.get(variable, 1.0) / levels] * levels)
            mask_nan.extend([variable in self.config.vars_with_nan_targets] * levels)
        level_weights = prediction.new_tensor(level_weights).view(1, 1, -1, 1, 1)
        variable_weights = prediction.new_tensor(variable_weights).view(1, 1, -1, 1, 1)
        mask_nan = torch.tensor(mask_nan, device=prediction.device, dtype=torch.bool).view(1, 1, -1, 1, 1)

        def weighted_mae(predicted, target):
            valid = ~(mask_nan & target.isnan())
            error = (predicted.masked_fill(~valid, 0) - target.masked_fill(~valid, 0)).abs()
            weighted = error * latitude_weights * level_weights
            weighted = torch.where(valid, weighted, self.config.loss_for_nan_targets)
            return (weighted * variable_weights).sum(dim=2).mean(dim=(1, 2, 3))

        absolute_error, absolute_difference = 0.0, 0.0
        for i in range(num_members):
            absolute_error = absolute_error + weighted_mae(prediction[:, i : i + 1], targets)
            for j in range(i):
                absolute_difference = absolute_difference + weighted_mae(
                    prediction[:, j : j + 1], prediction[:, i : i + 1]
                )
        return (absolute_error / num_members - absolute_difference / (num_members * (num_members - 1))).mean()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        grid_features: torch.Tensor,
        global_features: torch.Tensor,
        attention_mask: torch.Tensor,
        noise: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> WeatherNext2ForecastOutput:
        r"""
        grid_features (`torch.FloatTensor` of shape `(batch_size, num_channels, num_latitudes, num_longitudes)`):
            Normalized input fields, with channels ordered as in [`WeatherNext2Config.input_channel_layout`].
        global_features (`torch.FloatTensor` of shape `(batch_size, num_global_channels)`):
            Normalized values of the variables with no spatial extent.
        attention_mask (`torch.BoolTensor` of shape `(num_blocks, 1, block_size, 3 * block_size)`):
            Exact banded mesh geometry returned by [`WeatherNext2FeatureExtractor`], shared by all ensemble members.
        noise (`torch.FloatTensor` of shape `(batch_size, noise_channels)`):
            Standard normal draw defining the ensemble member, returned by [`WeatherNext2FeatureExtractor`].
        labels (`torch.FloatTensor` of shape `(batch_size, num_members, num_output_channels, num_latitudes, num_longitudes)`, *optional*):
            Targets in the same normalized space as `prediction`. Inputs flatten the first two axes in
            initial-condition-major order, as returned by the feature extractor with `num_members > 1`.
            At least two independently sampled members are required. The loss uses the checkpoint's
            `loss_weights`, `vars_with_nan_targets` and `loss_for_nan_targets`.

        Example:

        ```python
        >>> import torch
        >>> from transformers import WeatherNext2Config, WeatherNext2ForWeatherForecasting

        >>> config = WeatherNext2Config(
        ...     mesh_splits=2, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        ...     grid_latitudes=19, grid_longitudes=36,
        ...     num_grid_to_mesh_edges=2048, attention_bandwidth=64,
        ... )
        >>> model = WeatherNext2ForWeatherForecasting(config)

        >>> grid = torch.randn(2, config.num_grid_input_channels - 3, 19, 36, dtype=model.dtype)
        >>> global_features = torch.randn(2, config.num_mesh_input_channels - 3, dtype=model.dtype)
        >>> blocks = -(-config.num_mesh_nodes // config.attention_bandwidth)
        >>> mask = torch.ones(blocks, 1, config.attention_bandwidth, 3 * config.attention_bandwidth, dtype=torch.bool)
        >>> noise = torch.randn(2, config.noise_channels, dtype=model.dtype)
        >>> outputs = model(grid_features=grid, global_features=global_features, attention_mask=mask, noise=noise)
        >>> list(outputs.prediction.shape)
        [2, 101, 19, 36]
        ```
        """
        outputs: WeatherNext2ModelOutput = self.model(
            grid_features=grid_features,
            global_features=global_features,
            noise=noise,
            attention_mask=attention_mask,
            **kwargs,
        )

        prediction = self.head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = self._crps_loss(prediction, labels)

        return WeatherNext2ForecastOutput(
            loss=loss,
            prediction=prediction,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "WeatherNext2ForWeatherForecasting",
    "WeatherNext2Model",
    "WeatherNext2PreTrainedModel",
]
