# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The model.

An iHELIX block braids the same three strands as `helix-lm`, with "position in a sequence" replaced by
"position on a manifold" in the two spatial ones:

* **local**  -- exact attention over metric-space neighbours, per-head physical radii;
* **index**  -- content-addressed retrieval of distant regions through a hierarchy;
* **time**   -- a gated delta rule along whichever axis is sequential, carrying a fixed-size state per
  sample. For a forecast, a video or a fluid rollout this is the axis that has an order; space does not,
  and the model never pretends otherwise.

Above the block sits an encode-process-decode shell. Data arrives on whatever grid it lives on, is read
onto a fixed internal mesh, processed there, and written back out to whatever grid is asked for. Since
every read and write is geometric, resolution is a property of the *data*, not of the weights: train on
one-degree fields, run on quarter-degree, evaluate at scattered station coordinates, all with the same
parameters.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import nn

from .attention import CrossAttention, GeodesicAttention, GridLink, IndexAttention
from .geometry import Geometry
from .grid import FieldGrid
from .kernels import (
    FeedForward,
    IHelixLandmarkPooler,
    IHelixRMSNorm,
    IHelixRMSNormGated,
    apply_mask_to_padding_states,
    causal_conv1d_fn,
    torch_chunk_gated_delta_rule,
)


@dataclass
class IHelixConfig:
    """
    Configuration. Nothing here mentions a resolution, a grid shape, or a number of dimensions.

    Args:
        in_channels: channels per sample in the input field.
        out_channels: channels per sample to predict. Defaults to ``in_channels``.
        hidden_size: width of the residual stream.
        intermediate_size: width of the pointwise feed-forward. Defaults to ``4 * hidden_size``.
        num_layers: processor blocks.
        num_heads / num_kv_heads / head_dim: attention shape, shared by every strand.
        min_radius / max_radius: the physical receptive radii of the narrowest and widest heads, in the
            distance units the geometry was built with. These are lengths, so they keep their meaning
            when the sampling density changes -- which is the whole point.
        landmark_dim / index_branching / index_beam_width / index_topk: the long-range hierarchy.
        index_layer_stride: run the index strand on every n-th block; the others are local + time only.
        num_distance_buckets: logarithmic region-distance buckets biasing retrieval.
        use_temporal: enable the delta-rule strand along the sequential axis. Leave off for a single
            static field (one image, one snapshot).
        num_recurrent_heads / recurrent_head_dim / recurrent_value_head_dim: the temporal state, which is
            a matrix per head per sample and does not grow with how many frames have been seen.
        recurrent_chunk_size / conv_kernel_size: the chunkwise-parallel delta rule and its short causal
            convolution.
        latent_points: samples on the internal mesh when using the encode-process-decode model.
        encode_neighbours / decode_neighbours: how many source samples each read draws from.
        encode_radius: physical radius of the encode and decode reads.
    """

    in_channels: int = 8
    out_channels: int | None = None
    hidden_size: int = 256
    intermediate_size: int | None = None
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 4
    head_dim: int = 32

    min_radius: float = 200.0
    max_radius: float = 2000.0

    landmark_dim: int = 64
    index_branching: int = 4
    index_beam_width: int = 4
    index_topk: int = 4
    index_layer_stride: int = 2
    num_distance_buckets: int = 24

    use_temporal: bool = True
    num_recurrent_heads: int = 4
    recurrent_head_dim: int = 32
    recurrent_value_head_dim: int = 32
    recurrent_chunk_size: int = 16
    conv_kernel_size: int = 4

    latent_points: int = 2048
    encode_neighbours: int = 16
    decode_neighbours: int = 16
    encode_radius: float = 400.0

    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    relative_hidden: int = 64

    def __post_init__(self) -> None:
        if self.out_channels is None:
            self.out_channels = self.in_channels
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
        if self.num_heads % self.num_kv_heads:
            raise ValueError(f"num_heads ({self.num_heads}) must divide by num_kv_heads ({self.num_kv_heads}).")
        if not 0 < self.min_radius <= self.max_radius:
            raise ValueError(f"Need 0 < min_radius <= max_radius, got {self.min_radius}, {self.max_radius}.")
        for name in ("index_branching", "index_beam_width", "index_topk", "index_layer_stride"):
            if getattr(self, name) < 1:
                raise ValueError(f"`{name}` must be >= 1, got {getattr(self, name)}.")
        if self.index_branching < 2:
            raise ValueError(f"`index_branching` must be >= 2, got {self.index_branching}.")

    @property
    def layer_types(self) -> list[str]:
        """Which strands each block runs."""
        return ["local+index" if (i + 1) % self.index_layer_stride == 0 else "local" for i in range(self.num_layers)]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> IHelixConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})


class TemporalStrand(nn.Module):
    """
    The sequential strand: a gated delta rule run independently at every sample.

    Space has no order, so nothing recurrent runs across it. Time does, and this is where it runs. The
    state is a matrix per head per sample whose size does not depend on how many frames have gone by, so
    a rollout costs the same at frame 1000 as at frame 1 -- which is the property that makes long
    autoregressive integrations affordable.
    """

    def __init__(self, config: IHelixConfig) -> None:
        super().__init__()
        self.num_heads = config.num_recurrent_heads
        self.head_k_dim = config.recurrent_head_dim
        self.head_v_dim = config.recurrent_value_head_dim
        self.chunk_size = config.recurrent_chunk_size
        self.conv_kernel_size = config.conv_kernel_size
        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim

        self.in_proj_qkvz = nn.Linear(config.hidden_size, self.conv_dim + self.value_dim, bias=False)
        self.in_proj_ba = nn.Linear(config.hidden_size, 2 * self.num_heads, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        self.A_log = nn.Parameter(torch.empty(self.num_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.num_heads))
        self.norm = IHelixRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None, return_state: bool = False):
        """``x`` is ``(B, T, N, C)``; the sample axis is folded into the batch and each evolves alone."""
        batch, steps, num_points, _ = x.shape
        folded = x.permute(0, 2, 1, 3).reshape(batch * num_points, steps, -1)
        folded = apply_mask_to_padding_states(folded, None)

        mixed, z = torch.split(self.in_proj_qkvz(folded), [self.conv_dim, self.value_dim], dim=-1)
        mixed = causal_conv1d_fn(mixed.transpose(1, 2), self.conv1d.weight.squeeze(1), None, activation="silu")
        mixed = mixed[:, :, -steps:].transpose(1, 2)
        query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        beta_logits, a = torch.split(self.in_proj_ba(folded), [self.num_heads, self.num_heads], dim=-1)
        g = -self.A_log.float().exp() * nn.functional.softplus(a.float() + self.dt_bias)

        shape = (batch * num_points, steps, self.num_heads, -1)
        core, new_state = torch_chunk_gated_delta_rule(
            query.view(shape),
            key.view(shape),
            value.view(shape),
            g=g,
            beta=beta_logits.sigmoid(),
            chunk_size=self.chunk_size,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        core = self.norm(core, z.view(batch * num_points, steps, self.num_heads, self.head_v_dim))
        out = self.out_proj(core.reshape(batch * num_points, steps, self.value_dim))
        out = out.view(batch, num_points, steps, -1).permute(0, 2, 1, 3)
        return (out, new_state) if return_state else out


class IHelixBlock(nn.Module):
    """One braided block: local attention, optional long-range index, optional time, then feed-forward."""

    def __init__(self, config: IHelixConfig, geometry: Geometry, layer_idx: int) -> None:
        super().__init__()
        self.has_index = config.layer_types[layer_idx] == "local+index"
        self.has_temporal = config.use_temporal
        self.norm_mix = IHelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.local = GeodesicAttention(
            config.hidden_size,
            config.num_heads,
            config.num_kv_heads,
            config.head_dim,
            geometry.embed_dim,
            config.min_radius,
            config.max_radius,
            config.relative_hidden,
        )
        if self.has_index:
            self.index = IndexAttention(
                config.hidden_size,
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                config.landmark_dim,
                config.index_branching,
                config.index_beam_width,
                config.index_topk,
                config.num_distance_buckets,
                config.max_radius,
            )
            # Per-sample, per-head choice between what is nearby and what was retrieved.
            self.strand_gate = nn.Linear(config.hidden_size, config.num_heads * 2, bias=False)
            self.num_heads, self.head_dim = config.num_heads, config.head_dim
        if self.has_temporal:
            self.temporal = TemporalStrand(config)
        self.norm_ffn = IHelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = FeedForward(config.hidden_size, config.intermediate_size)

    def forward(self, x: torch.Tensor, grid: FieldGrid) -> torch.Tensor:
        batch, steps, num_points, channels = x.shape
        normed = self.norm_mix(x)
        flat = normed.reshape(batch * steps, num_points, channels)

        mixed = self.local(flat, grid)
        if self.has_index:
            retrieved = self.index(flat, grid)
            gate = self.strand_gate(flat).view(batch * steps, num_points, self.num_heads, 2).softmax(-1)
            shape = (batch * steps, num_points, self.num_heads, -1)
            mixed = (gate[..., :1] * mixed.view(shape) + gate[..., 1:] * retrieved.view(shape)).reshape(
                batch * steps, num_points, channels
            )
        mixed = mixed.view(batch, steps, num_points, channels)
        if self.has_temporal and steps > 1:
            mixed = mixed + self.temporal(normed)
        x = x + mixed
        return x + self.ffn(self.norm_ffn(x))


def init_weights(module: nn.Module, config: IHelixConfig) -> None:
    """Initialize one module; applied recursively at construction."""
    with torch.no_grad():
        if isinstance(module, nn.Linear):
            module.weight.normal_(mean=0.0, std=config.initializer_range)
            if module.bias is not None:
                module.bias.zero_()
        elif isinstance(module, (IHelixRMSNorm, IHelixRMSNormGated)):
            module.weight.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            module.weight.normal_(mean=0.0, std=config.initializer_range)
        elif isinstance(module, TemporalStrand):
            module.A_log.copy_(torch.empty(module.num_heads).uniform_(0.01, 16).log_())
            module.dt_bias.fill_(1.0)
        elif isinstance(module, IHelixLandmarkPooler):
            module.query.normal_(mean=0.0, std=config.initializer_range)
        elif isinstance(module, IndexAttention):
            module.distance_bias.zero_()
            module.level_bias.zero_()


class GradientCheckpointing:
    """
    Mixin adding ``gradient_checkpointing_enable()`` to a stack of blocks.

    Checkpointing trades compute for memory: a block's activations are thrown away on the forward pass
    and recomputed during the backward one, costing roughly a third more time and saving most of the
    activation memory. On a large card that is not a consolation prize -- it is how you convert spare
    VRAM into a bigger batch or a longer history, which is usually worth far more than the time it costs.
    """

    gradient_checkpointing: bool = False

    def gradient_checkpointing_enable(self, enable: bool = True) -> None:
        """Turn activation recomputation on or off for every block."""
        self.gradient_checkpointing = enable

    def _run_blocks(self, x: torch.Tensor, grid: FieldGrid) -> torch.Tensor:
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(block, x, grid, use_reentrant=False)
            else:
                x = block(x, grid)
        return x


class IHelixField(GradientCheckpointing, nn.Module):
    """
    The processor: blocks operating directly on the grid the data lives on.

    Use this when input and output share a grid -- denoising an image, stepping a simulation on its own
    mesh, filling in a field. For resolution independence across *different* grids, use
    :class:`IHelixFieldModel`, which puts this between a geometric encoder and decoder.
    """

    def __init__(self, config: IHelixConfig, geometry: Geometry) -> None:
        super().__init__()
        self.config, self.geometry = config, geometry
        self.embed = nn.Linear(config.in_channels, config.hidden_size)
        self.blocks = nn.ModuleList(IHelixBlock(config, geometry, i) for i in range(config.num_layers))
        self.norm = IHelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.out_channels)
        self.apply(lambda m: init_weights(m, config))

    def forward(self, values: torch.Tensor, grid: FieldGrid) -> torch.Tensor:
        """
        Args:
            values: ``(B, N, C)`` a single field, or ``(B, T, N, C)`` a sequence of them.
            grid: the :class:`FieldGrid` those samples live on.

        Returns:
            The same shape, with ``out_channels`` channels.
        """
        squeezed = values.ndim == 3
        if squeezed:
            values = values.unsqueeze(1)
        if values.shape[-2] != grid.num_points:
            raise ValueError(f"Field has {values.shape[-2]} samples but the grid has {grid.num_points}.")
        x = self._run_blocks(self.embed(values), grid)
        out = self.head(self.norm(x))
        return out.squeeze(1) if squeezed else out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class IHelixFieldModel(GradientCheckpointing, nn.Module):
    """
    Encode onto a fixed internal mesh, process there, decode back out.

    This is what makes resolution a property of the data. The processor always runs on the same mesh, so
    the weights never see how finely the input was sampled; the encoder reads whatever arrived and the
    decoder writes wherever asked. Train on a coarse grid and run on a fine one, or read a regular grid
    and write to scattered observation sites, without retraining or interpolating by hand.

    Args:
        config: model configuration.
        latent_grid: the internal mesh. On a sphere use :func:`~ihelix.grid.fibonacci_sphere`, whose cells
            are near-equal-area, so there is no pole to distort and no seam to cross.
    """

    def __init__(self, config: IHelixConfig, latent_grid: FieldGrid) -> None:
        super().__init__()
        self.config = config
        self.latent_grid = latent_grid
        geometry = latent_grid.geometry
        self.input_embed = nn.Linear(config.in_channels, config.hidden_size)
        self.latent_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.encoder = CrossAttention(
            config.hidden_size,
            config.num_heads,
            config.num_kv_heads,
            config.head_dim,
            geometry.embed_dim,
            config.encode_radius,
            config.relative_hidden,
        )
        self.encode_norm = IHelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.blocks = nn.ModuleList(IHelixBlock(config, geometry, i) for i in range(config.num_layers))
        self.norm = IHelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.decoder = CrossAttention(
            config.hidden_size,
            config.num_heads,
            config.num_kv_heads,
            config.head_dim,
            geometry.embed_dim,
            config.encode_radius,
            config.relative_hidden,
        )
        self.decode_norm = IHelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.out_channels)
        self.apply(lambda m: init_weights(m, config))
        self._links: dict[tuple[int, int], GridLink] = {}

    def link(self, target: FieldGrid, source: FieldGrid, num_neighbours: int) -> GridLink:
        """Cached correspondence between two grids, built on first use."""
        key = (id(target), id(source))
        if key not in self._links:
            self._links[key] = GridLink(target, source, num_neighbours)
        return self._links[key]

    def forward(
        self,
        values: torch.Tensor,
        input_grid: FieldGrid,
        output_grid: FieldGrid | None = None,
    ) -> torch.Tensor:
        """
        Args:
            values: ``(B, N, C)`` or ``(B, T, N, C)`` samples on ``input_grid``.
            input_grid: where the data was sampled.
            output_grid: where to write the prediction. Defaults to ``input_grid``; pass a different one
                to change resolution, change mesh, or evaluate at scattered points.
        """
        squeezed = values.ndim == 3
        if squeezed:
            values = values.unsqueeze(1)
        output_grid = output_grid or input_grid
        batch, steps = values.shape[0], values.shape[1]
        latent = self.latent_grid

        source = self.input_embed(values).reshape(batch * steps, input_grid.num_points, -1)
        seed = self.latent_token.expand(batch * steps, latent.num_points, -1)
        x = self.encode_norm(
            seed + self.encoder(seed, source, self.link(latent, input_grid, self.config.encode_neighbours))
        )

        x = self._run_blocks(x.view(batch, steps, latent.num_points, -1), latent)
        x = self.norm(x).reshape(batch * steps, latent.num_points, -1)

        target_seed = self.latent_token.expand(batch * steps, output_grid.num_points, -1)
        out = self.decode_norm(
            self.decoder(target_seed, x, self.link(output_grid, latent, self.config.decode_neighbours))
        )
        out = self.head(out).view(batch, steps, output_grid.num_points, -1)
        return out.squeeze(1) if squeezed else out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2) + "\n")
        torch.save(self.state_dict(), directory / "model.pt")
        return directory
