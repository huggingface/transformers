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

# Reviewers: @Rocketknight1
"""PyTorch WeatherNext 2 model.

The model is an encode-process-decode graph network:

    grid encoder ─┐
                  ├─► grid-to-mesh GNN ─► mesh transformer ─► mesh-to-grid GNN ─► decoder
    mesh encoder ─┘

made probabilistic by a single global noise vector, which is projected once and then modulates the
scale and offset of *every* normalization layer in the network (`WeatherNext2ConditionedNorm`). Two ensemble
members therefore differ only in that 32-dimensional vector.

All tensors below are batch-first and node-major within the batch: grid points and mesh nodes are a
flat sequence axis, `[batch, num_nodes, channels]`.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import ModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..bert.modeling_bert import eager_attention_forward
from ..clip.modeling_clip import CLIPMLP
from .configuration_weathernext2 import WeatherNext2Config
from .generation_weathernext2 import WeatherNext2GenerationMixin


class WeatherNext2MLP(CLIPMLP):
    pass


class WeatherNext2ConditionedNorm(nn.Module):
    """LayerNorm whose scale and offset are FiLM-derived from the global conditioning vector.

    The normalization itself has no affine parameters of its own: this is the only place the noise
    vector defining an ensemble member enters the network.
    """

    def __init__(self, config: WeatherNext2Config, num_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_features, eps=config.layer_norm_eps, elementwise_affine=False)
        self.linear = nn.Linear(config.noise_channels, 2 * num_features)

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        scale, offset = self.linear(conditioning).chunk(2, dim=-1)
        # `conditioning` is [batch, noise_channels]; `hidden_states` is [batch, nodes, hidden] outside
        # the mesh transformer and [batch, blocks, block, hidden] inside it, so the axes in between
        # are filled with singletons.
        broadcast_shape = (scale.shape[0], *([1] * (hidden_states.ndim - 2)), scale.shape[-1])
        return hidden_states * (1.0 + scale.view(broadcast_shape)) + offset.view(broadcast_shape)


class WeatherNext2ConditionedMlp(CLIPMLP):
    """The model's universal building block: [`WeatherNext2MLP`] followed by a conditioned norm.

    Used for the grid, mesh and edge encoders and for both node updates in each graph network. Only
    the widths vary, which is why they are arguments rather than read from the config.

    `chunk_size` has no default and every call site states it: the MLPs over the full grid are
    chunked along the points, while the ones over the mesh, or already inside a chunked loop, are
    not. Chunking is exact up to the reassociation of the matrix multiplications.
    """

    def __init__(
        self,
        config: WeatherNext2Config,
        in_features: int,
        hidden_features: int,
        out_features: int,
        chunk_size: int | None,
    ):
        nn.Module.__init__(self)
        self.config = config
        self.activation_fn = ACT2FN[config.mlp_act]
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.norm = WeatherNext2ConditionedNorm(config, out_features)
        # A falsy chunk size means one chunk spanning the whole axis, i.e. unchunked.
        self.chunk_size = chunk_size

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        num_rows = hidden_states.shape[1]
        chunk_size = self.chunk_size or num_rows
        chunks: list[torch.Tensor] = []
        for start in range(0, num_rows, chunk_size):
            chunk = self.fc2(self.activation_fn(self.fc1(hidden_states[:, start : start + chunk_size])))
            chunks.append(self.norm(chunk, conditioning))
        return torch.cat(chunks, dim=1)


class WeatherNext2EdgeUpdate(nn.Module):
    """Computes a message on every grid<->mesh edge.

    The first projection of the edge MLP is split across the edge features, the sender node and
    (in the mesh-to-grid direction) the receiver node. Each part is applied to the *nodes* and only
    then gathered onto the edges, which is much cheaper than gathering first: summing the parts is
    exactly the concatenated first matmul. The senders arrive already projected, because the graph
    network projects them once and then visits the edges in chunks.
    """

    def __init__(self, config: WeatherNext2Config, use_receiver_proj: bool):
        super().__init__()
        hidden_size = config.hidden_size
        # The bias of the concatenated first matmul rides on `edge_proj`, the one part that is
        # always present.
        self.edge_proj = nn.Linear(config.edge_hidden_size, hidden_size)
        self.sender_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.receiver_proj = nn.Linear(hidden_size, hidden_size, bias=False) if use_receiver_proj else None
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.act_fn = ACT2FN[config.mlp_act]
        self.norm = WeatherNext2ConditionedNorm(config, hidden_size)

    def forward(
        self,
        edge_states: torch.Tensor,
        projected_senders: torch.Tensor,
        receiver_states: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        messages = self.edge_proj(edge_states) + projected_senders[:, senders]
        if self.receiver_proj is not None:
            messages = messages + self.receiver_proj(receiver_states)[:, receivers]
        return self.norm(self.out_proj(self.act_fn(messages)), conditioning)


class WeatherNext2BipartiteGraphNetwork(nn.Module):
    """One round of message passing between the lat/lon grid and the icosahedral mesh.

    Edges run in a single direction. The node set on the receiving end is updated from its own
    features concatenated with the summed incoming messages; the sending node set is updated from
    its own features alone. Both updates are residual, so the two directions share this class and
    differ only in which node set receives messages and whether the receiver contributes to them.

    The edges are visited `config.chunk_size` at a time, in the way that suits each direction. A mesh
    node receives dozens of edges from the grid, so the grid-to-mesh direction chunks the edges and
    accumulates into the small mesh. A grid point receives exactly three edges from the mesh, so the
    mesh-to-grid direction chunks the grid points and finishes each block as it goes.
    """

    def __init__(self, config: WeatherNext2Config, grid_to_mesh: bool):
        super().__init__()
        self.grid_to_mesh = grid_to_mesh
        self.chunk_size = config.chunk_size
        # Only the grid-to-mesh direction rescales its aggregate, and only for the checkpoints that
        # set it: the number of grid points per mesh node varies with the grid resolution.
        self.aggregate_normalization = config.aggregate_normalization if grid_to_mesh else None
        hidden_size = config.hidden_size

        # The edge encoder runs inside the edge loop, and the receiving side's update inside the
        # block loop, so neither chunks again; only the grid-wide update on the sending side does.
        self.edge_encoder = WeatherNext2ConditionedMlp(
            config, config.num_edge_spatial_features, config.edge_hidden_size, config.edge_hidden_size, None
        )
        self.edge_update = WeatherNext2EdgeUpdate(config, use_receiver_proj=not grid_to_mesh)
        self.mesh_node_update = WeatherNext2ConditionedMlp(
            config, 2 * hidden_size if grid_to_mesh else hidden_size, hidden_size, hidden_size, None
        )
        self.grid_node_update = WeatherNext2ConditionedMlp(
            config,
            hidden_size if grid_to_mesh else 2 * hidden_size,
            hidden_size,
            hidden_size,
            config.chunk_size if grid_to_mesh else None,
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
        if self.grid_to_mesh:
            projected_senders = self.edge_update.sender_proj(grid_states)
            aggregated = torch.zeros(mesh_states.shape, dtype=torch.float32, device=mesh_states.device)
            num_edges = senders.numel()
            chunk_size = self.chunk_size or num_edges
            for start in range(0, num_edges, chunk_size):
                edges = slice(start, start + chunk_size)
                messages = self.edge_update(
                    self.edge_encoder(edge_features[:, edges], conditioning),
                    projected_senders,
                    mesh_states,
                    senders[edges],
                    receivers[edges],
                    conditioning,
                )
                aggregated.index_add_(1, receivers[edges], messages.float())
            mesh_states = mesh_states + self.mesh_node_update(
                self.with_aggregate(mesh_states, aggregated), conditioning
            )
            grid_states = grid_states + self.grid_node_update(grid_states, conditioning)
            return grid_states, mesh_states

        num_points = grid_states.shape[1]
        if senders.numel() != 3 * num_points:
            raise ValueError(
                f"The mesh-to-grid graph has {senders.numel()} edges, but every one of the {num_points} grid points "
                "receives exactly three, one from each vertex of its mesh triangle."
            )
        projected_senders = self.edge_update.sender_proj(mesh_states)
        chunk_size = self.chunk_size or num_points
        # Each finished block is written into one output, so the grid is never held twice.
        updated = torch.empty_like(grid_states)
        for start in range(0, num_points, chunk_size):
            block = grid_states[:, start : start + chunk_size]
            # The edges are sorted by receiver, three per grid point, so a block of points owns one
            # contiguous run of edges.
            edges = slice(3 * start, 3 * (start + block.shape[1]))
            local_receivers = receivers[edges] - start
            messages = self.edge_update(
                self.edge_encoder(edge_features[:, edges], conditioning),
                projected_senders,
                block,
                senders[edges],
                local_receivers,
                conditioning,
            )
            aggregated = torch.zeros(block.shape, dtype=torch.float32, device=block.device)
            aggregated.index_add_(1, local_receivers, messages.float())
            updated[:, start : start + block.shape[1]] = block + self.grid_node_update(
                self.with_aggregate(block, aggregated), conditioning
            )
        mesh_states = mesh_states + self.mesh_node_update(mesh_states, conditioning)
        return updated, mesh_states

    def with_aggregate(self, receiver_states: torch.Tensor, aggregated: torch.Tensor) -> torch.Tensor:
        """The receiving node's input: its own features next to its summed messages.

        The messages are summed in float32 and only then cast back: a mesh node can receive hundreds
        of them, and bf16 accumulation loses meaningful precision over that many terms.
        """
        if self.aggregate_normalization is not None:
            aggregated = aggregated / self.aggregate_normalization
        return torch.cat([receiver_states, aggregated.to(receiver_states.dtype)], dim=-1)


def gather_neighbouring_blocks(states: torch.Tensor) -> torch.Tensor:
    """Concatenates each block of nodes with the block before and after it, zero-padded at the ends.

    `states` is `[batch, num_blocks, heads, block_size, head_dim]`; the result has `3 * block_size`
    keys per query block.
    """
    padding = torch.zeros_like(states[:, :1])
    padded = torch.cat([padding, states, padding], dim=1)
    return torch.cat([padded[:, :-2], padded[:, 1:-1], padded[:, 2:]], dim=3)


def banded_mask_function(attention_mask: torch.Tensor) -> Callable:
    """Reads the geometry's banded mask as an index function, the form `masking_utils` expects.

    The block axis is folded into the batch axis before attention, so entry `b` of the batch
    corresponds to mesh-node block `b % num_blocks`.
    """
    num_blocks = attention_mask.shape[0]
    mask = attention_mask[:, 0]

    def inner(batch_idx, head_idx, q_idx, kv_idx):
        if batch_idx.ndim == 4:
            # Broadcast advanced indexing can materialize three full-size integer tensors.
            # Select each independent axis instead; scalar callbacks (flex/vmap) keep the path below.
            selected = mask.index_select(0, batch_idx.flatten() % num_blocks)
            selected = selected.index_select(1, q_idx.flatten())
            return selected.index_select(2, kv_idx.flatten()).unsqueeze(1)
        return mask[batch_idx % num_blocks, q_idx, kv_idx]

    return inner


@use_kernel_forward_from_hub("WeatherNext2Attention")
class WeatherNext2Attention(nn.Module):
    """Local self-attention over mesh nodes.

    There are no positional encodings: position is carried entirely by the attention mask, which
    connects each mesh node to every node within `config.attention_k_hop` mesh edges. Because the
    mesh nodes are ordered by a reverse Cuthill-McKee permutation that mask is banded, so a node can
    only ever attend inside its own block of `bandwidth` nodes and the two adjacent blocks.
    Attention is computed over exactly those three blocks, which keeps a mask that would be 1.7 GB
    dense at 0.25 degrees down to a few hundred MB and avoids materializing the full score matrix.
    """

    def __init__(self, config: WeatherNext2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        hidden_states (`torch.FloatTensor` of shape `(batch_size, num_blocks, block_size, hidden_size)`)
        attention_mask (`torch.Tensor` or `BlockMask` of shape `(batch_size * num_blocks, 1, block_size, 3 * block_size)`):
            Which of the three candidate blocks each query may attend to, node by node, as prepared by
            [`WeatherNext2MeshTransformer.forward`].
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # [batch, blocks, block, hidden] -> [batch, blocks, heads, block, head_dim]
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(2, 3)
        key_states = gather_neighbouring_blocks(self.k_proj(hidden_states).view(hidden_shape).transpose(2, 3))
        value_states = gather_neighbouring_blocks(self.v_proj(hidden_states).view(hidden_shape).transpose(2, 3))

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


@use_kernel_forward_from_hub("WeatherNext2AttentionMask")
class WeatherNext2AttentionMask(nn.Module):
    """Prepares the shared mesh mask for the selected attention backend, once per forward.

    It calls the mask interface directly rather than `create_bidirectional_mask`, which switches to vmap
    whenever a custom mask function is passed; the banded one is index-based, and at 0.25 degrees vmap
    would build index tensors over all 1.3 billion entries of the mask.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, attention_mask, batch_size, dtype):
        num_blocks, _, block_size, kv_length = attention_mask.shape
        mask_interface = ALL_MASK_ATTENTION_FUNCTIONS.get(self.config._attn_implementation)
        return (
            mask_interface(
                batch_size=batch_size * num_blocks,
                q_length=block_size,
                kv_length=kv_length,
                mask_function=banded_mask_function(attention_mask),
                allow_is_causal_skip=False,
                allow_is_bidirectional_skip=False,
                dtype=dtype,
                device=attention_mask.device,
                use_vmap=False,
            )
            if mask_interface is not None
            else None
        )


class WeatherNext2MeshTransformer(nn.Module):
    """The processor: a stack of pre-norm blocks over the mesh nodes."""

    def __init__(self, config: WeatherNext2Config):
        super().__init__()
        self.config = config
        self.mask_preparer = WeatherNext2AttentionMask(config)
        self.layers = nn.ModuleList(
            [WeatherNext2Layer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = WeatherNext2ConditionedNorm(config, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        conditioning: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        batch_size, num_nodes, hidden_size = hidden_states.shape
        num_blocks, _, block_size, kv_length = attention_mask.shape

        hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, num_blocks * block_size - num_nodes))
        hidden_states = hidden_states.view(batch_size, num_blocks, block_size, hidden_size)

        # Build the backend's geometry mask once and share it across all layers.
        attention_mask = self.mask_preparer(attention_mask, batch_size, hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, conditioning, **kwargs)

        hidden_states = hidden_states.reshape(batch_size, num_blocks * block_size, hidden_size)
        return self.norm(hidden_states[:, :num_nodes], conditioning)


@auto_docstring
class WeatherNext2PreTrainedModel(PreTrainedModel):
    config: WeatherNext2Config
    base_model_prefix = "model"
    main_input_name = "grid_features"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WeatherNext2Layer"]
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
        if isinstance(module, WeatherNext2Model):
            module.init_geometry_buffers()
        elif isinstance(module, WeatherNext2ForecastHead):
            module.init_output_activation_buffers()


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
    """

    last_hidden_state: torch.FloatTensor = None
    mesh_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring(custom_intro="A single forecast step.")
@dataclass
class WeatherNext2ForecastOutput(ModelOutput):
    r"""
    prediction (`torch.FloatTensor` of shape `(batch_size, num_output_channels, num_latitudes, num_longitudes)`):
        Predicted state for the next time step, in the model's normalized space, with channels ordered as in
        [`WeatherNext2Config.target_channel_layout`]. For variables that are also inputs this is a normalized
        *residual* on the last input frame; for the others it is the normalized value itself. Use
        [`WeatherNext2FeatureExtractor.postprocess`] to get physical units.
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_grid_points, hidden_size)`):
        Grid-point features the prediction was decoded from.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*):
        Mesh-node features after each transformer layer, shaped as in [`WeatherNext2ModelOutput`].
    """

    prediction: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


# The buffers the geometry lives in, in the order they are registered.
GEOMETRY_BUFFERS = (
    "grid_spatial_features",
    "mesh_spatial_features",
    "grid_to_mesh_senders",
    "grid_to_mesh_receivers",
    "grid_to_mesh_edge_features",
    "mesh_to_grid_senders",
    "mesh_to_grid_receivers",
    "mesh_to_grid_edge_features",
    "attention_mask",
)


@auto_docstring
class WeatherNext2Model(WeatherNext2PreTrainedModel):
    def __init__(self, config: WeatherNext2Config):
        super().__init__(config)
        hidden_size = config.hidden_size

        self.noise_encoder = nn.Linear(config.noise_channels, config.noise_channels, bias=False)
        self.grid_encoder = WeatherNext2ConditionedMlp(
            config, config.num_grid_input_channels, hidden_size, hidden_size, config.chunk_size
        )
        self.mesh_encoder = WeatherNext2ConditionedMlp(
            config, config.num_mesh_input_channels, hidden_size, hidden_size, None
        )
        self.grid_to_mesh = WeatherNext2BipartiteGraphNetwork(config, grid_to_mesh=True)
        self.mesh_transformer = WeatherNext2MeshTransformer(config)
        self.mesh_to_grid = WeatherNext2BipartiteGraphNetwork(config, grid_to_mesh=False)

        self.allocate_geometry_buffers()
        self.post_init()

    def allocate_geometry_buffers(self):
        """Registers the geometry buffers, empty, at the shapes the checkpoint stores them in.

        The mesh, the two bipartite graphs and the banded attention mask are not learned: they follow
        from the mesh refinement level and the grid, and deriving them is slow and pulls in libraries
        the forward pass has no other use for. A few grid points lie exactly on a shared triangle
        edge, so reconstructing the graph can also choose a different adjacent face as numerical
        dependencies change. Every checkpoint therefore carries the exact conversion-time geometry.
        Allocating it here needs only the two sizes the config records, which are the ones that cannot
        be derived in closed form.
        """
        config = self.config
        edges = config.num_grid_to_mesh_edges
        block_size = min(config.attention_bandwidth, config.num_mesh_nodes)
        num_blocks = -(-config.num_mesh_nodes // block_size)
        mesh_to_grid_edges = 3 * config.num_grid_points
        shapes = {
            "grid_spatial_features": ((config.num_grid_points, config.num_node_spatial_features), torch.float32),
            "mesh_spatial_features": ((config.num_mesh_nodes, config.num_node_spatial_features), torch.float32),
            "grid_to_mesh_senders": ((edges,), torch.int64),
            "grid_to_mesh_receivers": ((edges,), torch.int64),
            "grid_to_mesh_edge_features": ((edges, config.num_edge_spatial_features), torch.float32),
            "mesh_to_grid_senders": ((mesh_to_grid_edges,), torch.int64),
            "mesh_to_grid_receivers": ((mesh_to_grid_edges,), torch.int64),
            "mesh_to_grid_edge_features": ((mesh_to_grid_edges, config.num_edge_spatial_features), torch.float32),
            "attention_mask": ((num_blocks, 1, block_size, 3 * block_size), torch.bool),
        }
        for name in GEOMETRY_BUFFERS:
            shape, dtype = shapes[name]
            self.register_buffer(name, torch.zeros(shape, dtype=dtype), persistent=True)
        self.init_geometry_buffers()

    def init_geometry_buffers(self):
        """Fills the geometry buffers of a model that has none, so that it is at least runnable.

        A model built from a config rather than loaded has no mesh, so the node coordinates and the
        edges are zero and every mesh node attends to its own block and nothing else. That is a
        placeholder rather than a geometry: it exists so a randomly initialized model can run a
        forward pass, and so a model moved off the meta device has no uninitialized tensor left. A
        real forecast needs the geometry that came with the weights.
        """
        # The initialization helpers leave checkpoint geometry unchanged.
        for name in GEOMETRY_BUFFERS[:-1]:
            init.zeros_(getattr(self, name))
        # Every grid point receives exactly three edges from the mesh, which the mesh-to-grid network relies
        # on, so the placeholder graph keeps that layout too.
        receivers = self.mesh_to_grid_receivers
        init.copy_(receivers, torch.arange(receivers.numel() // 3, device=receivers.device).repeat_interleave(3))
        if not getattr(self.attention_mask, "_is_hf_initialized", False):
            init.zeros_(self.attention_mask)
            block_size = self.attention_mask.shape[2]
            init.ones_(self.attention_mask[:, :, :, block_size : 2 * block_size])

    @merge_with_config_defaults
    # `last_hidden_state` is the grid representation, not the last mesh-transformer layer, so it must
    # not be tied over the last recorded hidden state.
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        grid_features: torch.Tensor,
        global_features: torch.Tensor,
        noise: torch.Tensor,
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
        """
        batch_size = grid_features.shape[0]
        dtype = self.grid_encoder.fc1.weight.dtype
        conditioning = self.noise_encoder(noise.to(dtype=dtype))

        num_mesh_nodes = self.mesh_spatial_features.shape[0]
        mesh_inputs = torch.cat(
            [
                self.mesh_spatial_features.unsqueeze(0).expand(batch_size, -1, -1).to(dtype),
                global_features.unsqueeze(1).expand(-1, num_mesh_nodes, -1).to(dtype),
            ],
            dim=-1,
        )

        # The grid input is built inside the call, so it is freed as soon as the encoder returns rather
        # than living through the whole forward: at 0.25 degrees that is a full-grid tensor.
        grid_states = self.grid_encoder(
            torch.cat(
                [
                    self.grid_spatial_features.unsqueeze(0).expand(batch_size, -1, -1).to(dtype),
                    # [batch, channels, lat, lon] -> [batch, num_grid_points, channels]
                    grid_features.flatten(2).transpose(1, 2).to(dtype),
                ],
                dim=-1,
            ),
            conditioning,
        )
        mesh_states = self.mesh_encoder(mesh_inputs, conditioning)

        grid_states, mesh_states = self.grid_to_mesh(
            grid_states,
            mesh_states,
            self.grid_to_mesh_edge_features.unsqueeze(0).expand(batch_size, -1, -1).to(dtype),
            self.grid_to_mesh_senders,
            self.grid_to_mesh_receivers,
            conditioning,
        )
        mesh_states = self.mesh_transformer(mesh_states, self.attention_mask, conditioning, **kwargs)
        grid_states, mesh_states = self.mesh_to_grid(
            grid_states,
            mesh_states,
            self.mesh_to_grid_edge_features.unsqueeze(0).expand(batch_size, -1, -1).to(dtype),
            self.mesh_to_grid_senders,
            self.mesh_to_grid_receivers,
            conditioning,
        )

        return WeatherNext2ModelOutput(last_hidden_state=grid_states, mesh_hidden_state=mesh_states)


class WeatherNext2ForecastHead(nn.Module):
    """Decodes grid-point features into the predicted state, as `[batch, channels, lat, lon]`."""

    def __init__(self, config: WeatherNext2Config):
        super().__init__()
        self.config = config
        self.decoder_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.num_output_channels)
        self.act_fn = ACT2FN[config.mlp_act]
        gate, shifts = self.get_output_activation_buffers()
        self.sigmoid_gate = nn.Buffer(gate, persistent=False)
        self.sigmoid_shift = nn.Buffer(shifts, persistent=False)

    def get_output_activation_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns which output channels are squashed, and by how much.

        A few targets are probabilities rather than physical quantities. The negative shift keeps
        their prior mass near zero, since cyclones are rare.
        """
        config = self.config
        shifts = torch.zeros(config.num_output_channels)
        gate = torch.zeros(config.num_output_channels, dtype=torch.bool)
        offset = 0
        for variable, _, levels in config.target_channel_layout:
            # CODEPATH: only checkpoints predicting `cyclone_exists_gaussian_unit_mode` shift an
            # output; for every other target the dict is empty and no channel is gated.
            if variable in (config.sigmoid_shifted_outputs or {}):
                gate[offset : offset + levels] = True
                shifts[offset : offset + levels] = config.sigmoid_shifted_outputs[variable]
            offset += levels
        return gate, shifts

    def init_output_activation_buffers(self):
        gate, shifts = self.get_output_activation_buffers()
        init.copy_(self.sigmoid_gate, gate)
        init.copy_(self.sigmoid_shift, shifts)

    def forward(self, grid_states: torch.Tensor) -> torch.Tensor:
        num_points = grid_states.shape[1]
        chunk_size = self.config.chunk_size or num_points
        chunks: list[torch.Tensor] = []
        for start in range(0, num_points, chunk_size):
            chunk = grid_states[:, start : start + chunk_size]
            chunks.append(self.output_proj(self.act_fn(self.decoder_proj(chunk))))
        prediction = torch.cat(chunks, dim=1)
        prediction = torch.where(self.sigmoid_gate, torch.sigmoid(prediction - self.sigmoid_shift), prediction)
        return prediction.transpose(1, 2).reshape(
            prediction.shape[0], -1, self.config.grid_latitudes, self.config.grid_longitudes
        )


@auto_docstring(
    custom_intro="WeatherNext 2 with its forecasting head: advances the global atmospheric state by one time step."
)
class WeatherNext2ForWeatherForecasting(WeatherNext2PreTrainedModel, WeatherNext2GenerationMixin):
    def __init__(self, config: WeatherNext2Config):
        super().__init__(config)
        self.model = WeatherNext2Model(config)
        self.head = WeatherNext2ForecastHead(config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        grid_features: torch.Tensor,
        global_features: torch.Tensor,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> WeatherNext2ForecastOutput:
        r"""
        grid_features (`torch.FloatTensor` of shape `(batch_size, num_channels, num_latitudes, num_longitudes)`):
            Normalized input fields, with channels ordered as in [`WeatherNext2Config.input_channel_layout`].
        global_features (`torch.FloatTensor` of shape `(batch_size, num_global_channels)`):
            Normalized values of the variables with no spatial extent.
        noise (`torch.FloatTensor` of shape `(batch_size, noise_channels)`, *optional*):
            Standard normal draw defining the ensemble member. Sampled if not given. To produce an `n`-member
            ensemble, repeat the inputs `n` times along the batch axis and pass `n` different draws.
        generator (`torch.Generator`, *optional*):
            Generator used to sample `noise`, for reproducible ensembles.

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

        >>> grid = torch.randn(2, config.num_grid_input_channels - 3, 19, 36)
        >>> global_features = torch.randn(2, config.num_mesh_input_channels - 3)
        >>> outputs = model(grid_features=grid, global_features=global_features)  # a 2-member ensemble
        >>> list(outputs.prediction.shape)
        [2, 101, 19, 36]
        ```
        """
        if noise is None:
            # Drawn on the generator's own device - a CPU generator cannot seed a CUDA draw - so that
            # `torch.Generator().manual_seed(...)` gives the same ensemble whatever the model runs on.
            device = generator.device if generator is not None else grid_features.device
            noise = torch.randn(
                grid_features.shape[0],
                self.config.noise_channels,
                generator=generator,
                device=device,
                dtype=grid_features.dtype,
            ).to(grid_features.device)

        outputs: WeatherNext2ModelOutput = self.model(
            grid_features=grid_features, global_features=global_features, noise=noise, **kwargs
        )

        return WeatherNext2ForecastOutput(
            prediction=self.head(outputs.last_hidden_state),
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "WeatherNext2ForWeatherForecasting",
    "WeatherNext2Model",
    "WeatherNext2PreTrainedModel",
]
