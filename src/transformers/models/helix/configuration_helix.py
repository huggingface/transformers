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
"""HELIX (Hierarchical Episodic Linear IndeX) model configuration."""

import math

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring
@strict
class HelixConfig(PreTrainedConfig):
    r"""
    HELIX braids three sequence mixers inside every block, all of which are linear or near-linear in the
    sequence length and all of which carry a bounded amount of *hot* state at decode time:

    * **Strand L (local)** — exact softmax attention over a short, multi-scale sliding window. Runs on a
      block staircase (`local_blocks` memory blocks of `block_size` tokens plus the query's own block) so
      both compute and activation memory are `O(N * local_span)`.
    * **Strand R (recurrent)** — a gated delta-rule linear recurrence with a matrix-valued state, trained
      with the chunkwise-parallel (UT transform) form and decoded with a fixed-size state.
    * **Strand I (index)** — a hierarchical landmark index over the whole past. Each query *block* descends
      a `index_branching`-ary tree of pooled summaries with beam width `index_beam_width`, selects
      `index_topk` leaf blocks, and attends exactly over them.

    The three strand outputs are mixed by a per-token, per-head softmax gate.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the HELIX model.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 5632):
            Dimension of the SwiGLU MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of HELIX blocks.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of query heads shared by strand L and strand I.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            Number of key/value (and index-routing) head groups. Strand I routes one block selection per
            key/value group.
        head_dim (`int`, *optional*):
            Per-head dimension of strands L and I. Defaults to `hidden_size // num_attention_heads`.
        block_size (`int`, *optional*, defaults to 64):
            Memory block granularity. Blocks are the unit of landmark pooling, of index selection, and of
            the local staircase. Also the query-block granularity for index routing.
        local_blocks (`int`, *optional*, defaults to 4):
            Number of *previous* memory blocks visible to strand L (its own block is always visible). The
            widest per-head window is `local_blocks * block_size` tokens.
        num_window_scales (`int`, *optional*, defaults to 4):
            Query heads are split into this many contiguous groups with geometrically increasing windows
            (`local_span >> (num_window_scales - 1 - group)`, floored at `block_size`). This is the
            multi-scale locality prior; set to 1 to give every head the same window.
        index_layer_stride (`int`, *optional*, defaults to 3):
            Strand I is instantiated on every `index_layer_stride`-th layer (the others run L + R only,
            and keep a sliding-window KV cache instead of a full one). Set to 1 to index in every layer.
        landmark_dim (`int`, *optional*, defaults to 64):
            Dimension of the landmark/route space used by the index tree.
        index_branching (`int`, *optional*, defaults to 8):
            Fan-out of the landmark tree. Level `l + 1` summarizes `index_branching` level-`l` nodes.
        index_beam_width (`int`, *optional*, defaults to 4):
            Number of nodes kept at each level of the beam descent.
        index_topk (`int`, *optional*, defaults to 8):
            Number of leaf memory blocks each query block attends to.
        index_max_levels (`int`, *optional*, defaults to 8):
            Number of distinct learned per-level biases in the landmark tree. The tree itself always
            grows to whatever depth the memory needs; levels beyond this share the last bias.
        index_num_distance_buckets (`int`, *optional*, defaults to 32):
            Number of logarithmic relative *block* distance buckets used to bias the landmark descent.
            The bucket table saturates, so coarse distance keeps meaning something at ranges where a
            rotary phase has long since wrapped past anything seen in training.
        attention_tile_blocks (`int`, *optional*, defaults to 64):
            Number of query blocks whose attention is computed at once. The gathered keys, values and
            masks are the largest transient tensors in a block, so this caps peak activation memory
            independently of the context length. `0` processes the whole sequence in one tile. It has no
            effect on the result.
        num_recurrent_heads (`int`, *optional*, defaults to 8):
            Number of heads of the delta-rule recurrence.
        recurrent_head_dim (`int`, *optional*, defaults to 128):
            Key dimension of the recurrent matrix state (state is `recurrent_head_dim x recurrent_value_head_dim`).
        recurrent_value_head_dim (`int`, *optional*, defaults to 128):
            Value dimension of the recurrent matrix state.
        recurrent_chunk_size (`int`, *optional*, defaults to 64):
            Chunk length of the chunkwise-parallel delta rule used during training/prefill.
        conv_kernel_size (`int`, *optional*, defaults to 4):
            Kernel size of the short depthwise causal convolution applied to the recurrent q/k/v stream.
        use_surprise_gating (`bool`, *optional*, defaults to `True`):
            Modulate the delta-rule write strength by how novel the current key is relative to a short
            causal pool of the preceding keys, so that state capacity is spent on unpredictable content.
        surprise_kernel_size (`int`, *optional*, defaults to 8):
            Width of the causal pool used by the surprise signal.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation of the MLP.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether the MLP projections carry a bias.
        max_position_embeddings (`int`, *optional*, defaults to 1048576):
            Maximum sequence length the model is expected to handle.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            Epsilon of the RMS norms.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return the braided cache.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the input and output embeddings.
        rope_parameters (`RopeParameters`, *optional*):
            RoPE parameters. Strands L and I share one set of rotated queries and keys, so a retrieved
            block keeps its internal token order; strand I adds `index_num_distance_buckets` on top to
            carry coarse block distance.
        layer_types (`list[str]`, *optional*):
            Per-layer strand schedule, one of `"helix"` (L + R + I) or `"helix_local"` (L + R). Derived
            from `index_layer_stride` when not given.

    ```python
    >>> from transformers import HelixModel, HelixConfig

    >>> configuration = HelixConfig()
    >>> model = HelixModel(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "helix"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 500000.0

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_tp_plan = {
        "layers.*.mixer.q_proj": "colwise",
        "layers.*.mixer.k_proj": "colwise",
        "layers.*.mixer.v_proj": "colwise",
        "layers.*.mixer.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int | None = None

    # strand L / shared block geometry
    block_size: int = 64
    local_blocks: int = 4
    num_window_scales: int = 4

    # strand I
    index_layer_stride: int = 3
    landmark_dim: int = 64
    index_branching: int = 8
    index_beam_width: int = 4
    index_topk: int = 8
    index_max_levels: int = 8
    index_num_distance_buckets: int = 32
    attention_tile_blocks: int = 64

    # strand R
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
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    layer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"`num_attention_heads` ({self.num_attention_heads}) must be divisible by `num_key_value_heads` "
                f"({self.num_key_value_heads})."
            )
        if self.num_window_scales < 1 or self.num_window_scales > self.num_attention_heads:
            raise ValueError(
                f"`num_window_scales` must be in [1, num_attention_heads]; got {self.num_window_scales} for "
                f"{self.num_attention_heads} heads."
            )
        if self.index_branching < 2:
            raise ValueError(f"`index_branching` must be >= 2, got {self.index_branching}.")
        if self.index_layer_stride < 1:
            raise ValueError(f"`index_layer_stride` must be >= 1, got {self.index_layer_stride}.")

        if self.layer_types is None:
            self.layer_types = [
                "helix" if (i + 1) % self.index_layer_stride == 0 else "helix_local"
                for i in range(self.num_hidden_layers)
            ]

        # Strand L must cover everything strand I cannot see: strand I only reads memory blocks that end
        # strictly before the query's own block, so the window has to span at least one full block.
        if self.local_window_sizes[0] < self.block_size:
            raise ValueError(
                f"The narrowest per-head window ({self.local_window_sizes[0]}) is smaller than `block_size` "
                f"({self.block_size}), which would leave a hole in the receptive field. Reduce "
                f"`num_window_scales` or raise `local_blocks`."
            )
        # The recurrent state is what carries information between chunks, so a chunk must not exceed a block.
        self.sliding_window = (self.local_blocks + 1) * self.block_size
        # Strand R keeps one convolution state for its q/k/v stream, plus a second one for the causal
        # key pool that drives surprise gating.
        self.number_of_conv_states = 2 if self.use_surprise_gating else 1
        super().__post_init__(**kwargs)

    @property
    def local_span(self) -> int:
        """Widest per-head local window, in tokens."""
        return self.local_blocks * self.block_size

    @property
    def local_window_sizes(self) -> list[int]:
        """Per-head window length, one entry per query head, narrowest group first."""
        heads_per_scale = self.num_attention_heads / self.num_window_scales
        windows = []
        for head in range(self.num_attention_heads):
            scale = int(head // heads_per_scale)
            windows.append(max(self.block_size, self.local_span >> (self.num_window_scales - 1 - scale)))
        return windows

    def index_num_levels(self, num_blocks: int) -> int:
        """Number of landmark-tree levels built *above* the leaves for a memory of `num_blocks` blocks."""
        if num_blocks < self.index_branching:
            return 0
        return int(math.log(num_blocks, self.index_branching))


__all__ = ["HelixConfig"]
