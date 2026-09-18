# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Configuration for HELIX."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class HelixConfig:
    """
    Configuration for a HELIX model.

    Every block braids three sequence mixers over the same residual stream:

    * **strand L** — exact softmax attention over a short, multi-scale sliding window, run on a block
      staircase so compute and activation memory are ``O(N * local_span)``;
    * **strand R** — a gated delta-rule recurrence with a matrix-valued state, trained with the
      chunkwise-parallel (UT transform) form and decoded from a fixed-size state;
    * **strand I** — a hierarchical landmark index over the whole past. Each query *block* descends an
      ``index_branching``-ary tree of pooled summaries with beam width ``index_beam_width``, selects
      ``index_topk`` leaf blocks, and attends exactly over them.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Width of the residual stream.
        intermediate_size: Width of the SwiGLU MLP.
        num_hidden_layers: Number of HELIX blocks.
        num_attention_heads: Query heads shared by strands L and I.
        num_key_value_heads: Key/value (and index-routing) head groups.
        head_dim: Per-head dimension of strands L and I. Defaults to ``hidden_size // num_attention_heads``.
        block_size: Memory block granularity — the unit of landmark pooling, index selection, the local
            staircase, and index routing.
        local_blocks: Number of *previous* memory blocks visible to strand L. The widest per-head window is
            ``local_blocks * block_size`` tokens.
        num_window_scales: Query heads split into this many contiguous groups with geometrically increasing
            windows. This is the multi-scale locality prior; 1 gives every head the same window.
        index_layer_stride: Strand I runs on every ``index_layer_stride``-th layer. The others run L + R and
            keep a window-capped key/value cache instead of a full one.
        landmark_dim: Dimension of the landmark/route space.
        index_branching: Fan-out of the landmark tree.
        index_beam_width: Nodes kept at each level of the beam descent.
        index_topk: Leaf memory blocks each query block attends to.
        index_max_levels: Number of distinct learned per-level biases. The tree itself always grows to
            whatever depth the memory needs.
        index_num_distance_buckets: Logarithmic relative *block* distance buckets biasing the descent.
        attention_tile_blocks: Query blocks attended at once. Caps peak activation memory independently of
            context length; 0 processes the whole sequence in one tile. No effect on the result.
        num_recurrent_heads: Heads of the delta-rule recurrence.
        recurrent_head_dim: Key dimension of the recurrent matrix state.
        recurrent_value_head_dim: Value dimension of the recurrent matrix state.
        recurrent_chunk_size: Chunk length of the chunkwise-parallel delta rule.
        conv_kernel_size: Short depthwise causal convolution on the recurrent q/k/v stream.
        use_surprise_gating: Modulate delta-rule write strength by how novel the current key is against a
            short causal pool of the keys before it.
        surprise_kernel_size: Width of the causal pool driving the surprise signal.
        hidden_act: MLP activation (``silu`` or ``gelu``).
        mlp_bias: Whether the MLP projections carry a bias.
        max_position_embeddings: Longest sequence the model is expected to handle.
        rope_theta: RoPE base.
        initializer_range: Standard deviation of the normal initializer.
        rms_norm_eps: Epsilon of the RMS norms.
        tie_word_embeddings: Whether to tie input and output embeddings.
        pad_token_id / bos_token_id / eos_token_id: Special token ids.
        layer_types: Per-layer schedule of ``"helix"`` (L+R+I) or ``"helix_local"`` (L+R). Derived from
            ``index_layer_stride`` when omitted.
    """

    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int | None = None

    block_size: int = 64
    local_blocks: int = 4
    num_window_scales: int = 4

    index_layer_stride: int = 3
    landmark_dim: int = 64
    index_branching: int = 8
    index_beam_width: int = 4
    index_topk: int = 8
    index_max_levels: int = 8
    index_num_distance_buckets: int = 32
    attention_tile_blocks: int = 64

    num_recurrent_heads: int = 8
    recurrent_head_dim: int = 128
    recurrent_value_head_dim: int = 128
    recurrent_chunk_size: int = 64
    conv_kernel_size: int = 4
    use_surprise_gating: bool = True
    surprise_kernel_size: int = 8

    hidden_act: str = "silu"
    mlp_bias: bool = False
    max_position_embeddings: int = 1048576
    rope_theta: float = 500000.0
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False

    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | None = 2

    layer_types: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})."
            )
        if not 1 <= self.num_window_scales <= self.num_attention_heads:
            raise ValueError(
                f"num_window_scales must be in [1, num_attention_heads]; got {self.num_window_scales} "
                f"for {self.num_attention_heads} heads."
            )
        if self.index_branching < 2:
            raise ValueError(f"index_branching must be >= 2, got {self.index_branching}.")
        if self.index_layer_stride < 1:
            raise ValueError(f"index_layer_stride must be >= 1, got {self.index_layer_stride}.")

        if not self.layer_types:
            self.layer_types = [
                "helix" if (i + 1) % self.index_layer_stride == 0 else "helix_local"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but num_hidden_layers is {self.num_hidden_layers}."
            )
        unknown = set(self.layer_types) - {"helix", "helix_local"}
        if unknown:
            raise ValueError(f"layer_types entries must be 'helix' or 'helix_local'; got {sorted(unknown)}.")

        for name, minimum in (
            ("block_size", 1),
            ("local_blocks", 0),
            ("index_topk", 1),
            ("index_beam_width", 1),
            ("landmark_dim", 1),
            ("index_num_distance_buckets", 1),
            ("num_recurrent_heads", 1),
            ("recurrent_head_dim", 1),
            ("recurrent_value_head_dim", 1),
            ("recurrent_chunk_size", 1),
            ("conv_kernel_size", 1),
            # The surprise pool averages the `surprise_kernel_size - 1` positions before the current one,
            # so a width of 1 would leave it with nothing to average.
            ("surprise_kernel_size", 2),
        ):
            if getattr(self, name) < minimum:
                raise ValueError(f"`{name}` must be >= {minimum}, got {getattr(self, name)}.")
        self.sliding_window = (self.local_blocks + 1) * self.block_size
        self.number_of_conv_states = 2 if self.use_surprise_gating else 1

    @property
    def local_span(self) -> int:
        """Widest per-head local window, in tokens."""
        return self.local_blocks * self.block_size

    @property
    def local_window_sizes(self) -> list[int]:
        """Per-head window length, one entry per query head, narrowest group first."""
        # Floored at `block_size`: strand I only reads blocks ending strictly before the query's own
        # block, so the narrowest window must still span a full block or the two would leave a gap.
        heads_per_scale = self.num_attention_heads / self.num_window_scales
        return [
            max(self.block_size, self.local_span >> (self.num_window_scales - 1 - int(head // heads_per_scale)))
            for head in range(self.num_attention_heads)
        ]

    def index_num_levels(self, num_blocks: int) -> int:
        """Landmark-tree levels built *above* the leaves for a memory of ``num_blocks`` blocks."""
        if num_blocks < self.index_branching:
            return 0
        return int(math.log(num_blocks, self.index_branching))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> HelixConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})

    def save_pretrained(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "config.json"
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> HelixConfig:
        return cls.from_dict(json.loads((Path(directory) / "config.json").read_text()))
