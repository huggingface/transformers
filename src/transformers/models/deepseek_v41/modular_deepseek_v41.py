# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import math
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache, DynamicSlidingWindowLayer, KVSharingDict
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import ChannelDimension, ImageInput, ImageType, PILImageResampling, SizeDict, get_image_type
from ...integrations import use_experts_implementation
from ...masking_utils import create_recurrent_attention_mask, create_sliding_window_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MoeModelOutputWithPast,
    MultimodalMoeBaseModelOutputWithPast,
    MultimodalMoeCausalLMOutputWithPast,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ImagesKwargs, ProcessorMixin, Unpack
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
    torch_compilable_check,
)
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from ..deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Experts,
    DeepseekV4GroupedLinear,
    DeepseekV4MLP,
    DeepseekV4RMSNorm,
    DeepseekV4RotaryEmbedding,
    DeepseekV4TopKRouter,
    apply_rotary_pos_emb,
)
from ..gpt_oss.modeling_gpt_oss import eager_attention_forward


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V4.1-Flash")
@strict
class DeepseekV41VisionConfig(PreTrainedConfig):
    r"""
    rope_theta (`float`, *optional*, defaults to 10000.0):
        Base of the 2D rotary position embedding.
    downsample_ratio (`int`, *optional*, defaults to 3):
        Side of the square of patches that the projector merges into one language model token.
    max_image_tokens (`int`, *optional*, defaults to 1024):
        The most language model tokens an image may take, delimiters included. Only used for preprocessing.
    min_pixels (`int`, *optional*, defaults to 295936):
        Images with fewer pixels are scaled up to this area. Only used for preprocessing.
    max_wh_ratio (`float`, *optional*):
        Images wider than this width-to-height ratio are squashed to it. Only used for preprocessing.
    """

    model_type = "deepseek_v41_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1024
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    intermediate_size: int = 2816
    patch_size: int = 14
    num_channels: int = 3
    rope_theta: float | int = 10000.0
    downsample_ratio: int = 3
    max_image_tokens: int = 1024
    min_pixels: int = 295936
    max_wh_ratio: float | int | None = None
    rms_norm_eps: float = 1.0e-6
    initializer_range: float = 0.02


@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V4.1-Flash")
@strict
class DeepseekV41TextConfig(DeepseekV4Config):
    r"""
    scoring_func (`str`):
        Router activation — `sqrtsoftplus`, `softmax`, or `sigmoid`.
    rope_theta (`float`):
        RoPE base for the sliding-window-only layers.
    layer_types (`list[str]`):
        Per-layer cache type. Every DeepSeek-V4.1 layer uses `"shared_compressed_attention"`, so this is filled in
        automatically.
    compress_ratios (`list[int]`, *optional*):
        Per-layer KV compression ratio. 0 is sliding-window attention only, and `r > 0` adds attention over
        KV that are pooled `r`-to-1. The checkpoint's config lists the DSpark layers after the main ones, and
        those extra entries are ignored here. Defaults to sliding-window attention everywhere.
    kv_source_layer_ids (`list[int]`, *optional*):
        The layers that compress their own KV, and run an indexer to pick which compressed entries each query
        attends to. Every later layer with the same ratio reads that KV, up to the next source.
    index_source_layer_ids (`list[int]`, *optional*):
        The layers that run an indexer. A layer that is not in `kv_source_layer_ids` runs it over the compressed
        KV that it reads, and the layers in between reuse the selection of the last indexer.
    candidate_source_layer_id (`int`, *optional*):
        The indexer of this layer also keeps the `candidate_topk_blocks` best blocks of compressed positions per
        query, and the indexers after it only choose inside those blocks.
    candidate_topk_blocks (`int`, *optional*, defaults to 2048):
        Number of blocks the candidate source keeps per query.
    candidate_block_size (`int`, *optional*, defaults to 8):
        Number of compressed positions in a candidate block.
    compress_rope_theta (`float`):
        RoPE base for the layers that attend to compressed KV (paired with `rope_scaling` for YaRN).
    hc_mult (`int`):
        Number of parallel streams the hyper-connections carry the residual in.
    hc_sinkhorn_iters (`int`):
        Sinkhorn-Knopp iterations that make the hyper-connection mixing matrix doubly stochastic.
    hc_eps (`float`):
        Numerical floor for the Sinkhorn-Knopp normalization.
    swiglu_limit (`float`):
        Clip the experts' gate/up pre-activations.
    sliding_window (`int`):
        Size of the sliding window that every layer attends over.
    o_groups (`int`):
        Number of head groups in the grouped output projection.
    o_lora_rank (`int`):
        Per-group intermediate dim of the grouped output projection.
    index_n_heads (`int`):
        Number of indexer query heads.
    index_head_dim (`int`):
        Indexer head dim.
    index_topk (`int`):
        Number of compressed entries the indexer keeps per query.
    ple_layer_ids (`list[int]`, *optional*):
        The layers that add an Engram n-gram lookup to the residual stream before their attention. These are
        zero-based layer indices, as in the checkpoint's `engram_layer_ids`.
    ngram_num_embeddings (`list[int]`, *optional*):
        Rows in the hash table of each Engram layer. This is the sum of the layer's bucket sizes, so it is only
        checked against them.
    ngram_size (`int`, *optional*, defaults to 4):
        Longest n-gram that is hashed. Every size from 2 up to this one gets its own lookups.
    ngram_vocab_size_base (`int`, *optional*, defaults to 16000000):
        Each (n-gram size, head) pair gets a bucket range whose size is the next unused prime above this.
    heads_per_ngram (`int`, *optional*, defaults to 8):
        Number of hash heads per n-gram size.
    ngram_head_dim (`int`, *optional*, defaults to 256):
        Dimension of the rows of the Engram hash tables.
    ngram_pad_token_id (`int`, *optional*, defaults to 2):
        Token that fills the slots of an n-gram that reach back before the start of the sequence.
    ngram_compressed_vocab_size (`int`, *optional*, defaults to 99092):
        Size of the vocabulary once tokens that normalize alike are merged. The hash multipliers are derived
        from it.
    num_nextn_predict_layers (`int`):
        Number of DSpark draft layers in the checkpoint (not instantiated here).
    dspark_block_size (`int`, *optional*, defaults to 5):
        Number of tokens a DSpark draft predicts (not used yet).
    dspark_noise_token_id (`int`, *optional*, defaults to 128799):
        Token that fills the positions a DSpark draft has yet to predict (not used yet).
    dspark_target_layer_ids (`list[int]`, *optional*):
        The layers whose inputs the DSpark draft layers read (not used yet).
    dspark_markov_rank (`int`, *optional*, defaults to 256):
        Rank of the DSpark Markov head (not used yet).
    dspark_n_routed_experts (`int`, *optional*, defaults to 128):
        Number of routed experts in the DSpark draft layers (not used yet).
    dspark_num_experts_per_tok (`int`, *optional*, defaults to 3):
        Number of experts each token is routed to in the DSpark draft layers (not used yet).
    partial_rotary_factor (`float`, *optional*):
        Fraction of head_dim that gets RoPE.
        Defaults to `qk_rope_head_dim / head_dim` so cos/sin sizes to `qk_rope_head_dim`.
    """

    model_type = "deepseek_v41_text"
    base_config_key = "text_config"

    hidden_size: int = 5120
    moe_intermediate_size: int = 2304
    num_hidden_layers: int = 40
    q_lora_rank: int = 1280
    n_routed_experts: int = 384
    index_n_heads: int = 32
    num_nextn_predict_layers: int = 3
    rms_norm_eps: float = 1.0e-20
    pad_token_id: int | None = 2

    # The checkpoint's config calls the n-gram fields `engram_*`. We name them like Qwen4-Exp names its own n-gram
    # tables (`ple_*` and `ngram_*`), and this map reads the checkpoint's names into them.
    # TODO: that is the naming the review of Hy4-flash (#50903) asked for, but the PR itself still says
    # `engram_*`. We're watching which one it merges with, to follow it.
    attribute_map = {
        "num_local_experts": "n_routed_experts",
        "intermediate_size": "moe_intermediate_size",
        "engram_layer_ids": "ple_layer_ids",
        "engram_num_embeddings": "ngram_num_embeddings",
        "engram_max_ngram_size": "ngram_size",
        "engram_vocab_size": "ngram_vocab_size_base",
        "engram_n_heads": "heads_per_ngram",
        "engram_head_dim": "ngram_head_dim",
        "engram_pad_token_id": "ngram_pad_token_id",
        "engram_compressed_vocab_size": "ngram_compressed_vocab_size",
    }

    compress_ratios: list[int] | None = None
    kv_source_layer_ids: list[int] | None = None
    index_source_layer_ids: list[int] | None = None
    candidate_source_layer_id: int | None = None
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8

    ple_layer_ids: list[int] | None = None
    ngram_num_embeddings: list[int] | None = None
    ngram_size: int = 4
    ngram_vocab_size_base: int = 16000000
    heads_per_ngram: int = 8
    ngram_head_dim: int = 256
    ngram_pad_token_id: int = 2
    ngram_compressed_vocab_size: int = 99092

    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: list[int] | None = None
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 128
    dspark_num_experts_per_tok: int = 3

    compress_rates = AttributeError()
    default_compress_rates = AttributeError()
    mlp_layer_types = AttributeError()
    default_num_hash_layers = AttributeError()

    def validate_layer_type(self):
        """Checks that the attention schedule is one the model can run: every layer that attends to compressed
        KV needs a source for that KV and for its selection somewhere before it."""
        if self.num_hidden_layers is None or self.compress_ratios is None:
            return
        kv_source_layer_ids = self.kv_source_layer_ids or []
        index_source_layer_ids = self.index_source_layer_ids or []

        source_ratio = None
        for layer_idx in range(self.num_hidden_layers):
            ratio = self.compress_ratios[layer_idx]
            if ratio == 0:
                # A sliding window and nothing else, so there is no compressed KV to find a source for
                continue
            if layer_idx in kv_source_layer_ids:
                # Compresses its own KV, and the index keys are made from them
                if layer_idx not in index_source_layer_ids:
                    raise ValueError(
                        f"Layer {layer_idx} is in `kv_source_layer_ids`, so it must be in `index_source_layer_ids` "
                        f"too: the index keys are made from the KV it compresses. Got {index_source_layer_ids}."
                    )
                source_ratio = ratio
            elif ratio != source_ratio:
                # Reads the compressed KV of the last source, which must be pooled with its own ratio
                raise ValueError(
                    f"Layer {layer_idx} has a compression ratio of {ratio}, but the last layer in "
                    f"`kv_source_layer_ids` before it compresses with a ratio of {source_ratio}. Add {layer_idx} "
                    "to `kv_source_layer_ids` or change its entry in `compress_ratios`."
                )
        if self.candidate_source_layer_id is not None and self.candidate_source_layer_id not in index_source_layer_ids:
            raise ValueError(
                f"`candidate_source_layer_id` ({self.candidate_source_layer_id}) must be one of "
                f"`index_source_layer_ids` ({index_source_layer_ids})."
            )

    def validate_fake_quantization(self):
        """Checks that the activations which are rounded to FP8 or FP4 split into whole blocks of 32 values."""
        widths = {
            "head_dim": self.head_dim,
            "num_attention_heads * head_dim // o_groups": self.num_attention_heads * self.head_dim // self.o_groups,
        }
        if self.compress_ratios is not None and any(self.compress_ratios):
            # Only the layers that attend to compressed KV have index keys and queries
            widths["index_head_dim"] = self.index_head_dim
        bad = {name: width for name, width in widths.items() if width % 32 != 0}
        if bad:
            raise ValueError(
                "The KV, the input of the grouped output projection and the index keys and queries are rounded to "
                f"FP8 or FP4 in blocks of 32 values, so their widths must be multiples of 32. Got {bad}."
            )

    def __post_init__(self, **kwargs):
        legacy_qk_rope_head_dim = kwargs.pop("qk_rope_head_dim", None)
        PreTrainedConfig.__post_init__(self, **kwargs)
        n = self.num_hidden_layers

        # Without a schedule, every layer is sliding-window attention only
        if self.compress_ratios is None:
            self.compress_ratios = [0] * n
        if len(self.compress_ratios) < n:
            raise ValueError(
                f"`compress_ratios` needs an entry per layer, but has {len(self.compress_ratios)} entries for "
                f"{n} layers."
            )
        self.kv_source_layer_ids = list(self.kv_source_layer_ids or [])
        self.index_source_layer_ids = list(self.index_source_layer_ids or [])
        self.ple_layer_ids = list(self.ple_layer_ids or [])
        self.dspark_target_layer_ids = list(self.dspark_target_layer_ids or [])

        # `layer_types` picks the cache class, and all layers keep their state in the same one
        self.layer_types = ["shared_compressed_attention"] * n

        # `partial_rotary_factor` = legacy `qk_rope_head_dim / head_dim` if given, else default.
        # `qk_rope_head_dim` is a runtime-only attr (never a dataclass field).
        if self.partial_rotary_factor is None:
            self.partial_rotary_factor = (
                legacy_qk_rope_head_dim / self.head_dim
                if legacy_qk_rope_head_dim is not None
                else self.default_partial_rotary_factor
            )
        self.qk_rope_head_dim = int(self.head_dim * self.partial_rotary_factor)

        # As in DeepSeek-V4, YaRN only applies to the layers that attend to compressed KV, which rotate with
        # `compress_rope_theta`. The sliding-window-only layers use plain RoPE with `rope_theta`. The reference
        # does not multiply cos/sin by YaRN's mscale, so we force `attention_factor=1.0` for YaRN.
        rp = self.rope_parameters or {}
        if isinstance(rp.get("main"), dict) and isinstance(rp.get("compress"), dict):
            # Already nested - drop any leftover top-level keys
            self.rope_parameters = {"main": rp["main"], "compress": rp["compress"]}
        else:
            yarn = {k: v for k, v in rp.items() if k not in ("main", "compress")}
            main = {
                "rope_type": "default",
                "rope_theta": self.rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            }
            compress = {
                **yarn,
                "rope_theta": self.compress_rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            }
            compress.setdefault("rope_type", "default")
            if compress["rope_type"] == "yarn":
                compress.setdefault("attention_factor", 1.0)
            self.rope_parameters = {"main": main, "compress": compress}

    @property
    def kv_source_layers(self) -> list[int | None]:
        """
        For each layer, the layer whose compressed KV it attends to. That is the layer itself if it compresses its
        own, and `None` if it attends over its sliding window only.
        """
        kv_source_layers, source = [], None
        for layer_idx in range(self.num_hidden_layers):
            if self.compress_ratios[layer_idx] == 0:
                # A sliding window and nothing else
                kv_source_layers.append(None)
            elif layer_idx in self.kv_source_layer_ids:
                # Compresses its own KV, and stays the source until the next layer that does
                source = layer_idx
                kv_source_layers.append(layer_idx)
            else:
                # Reads the compressed KV of the last source
                kv_source_layers.append(source)
        return kv_source_layers

    @property
    def kv_sharing_modes(self) -> list[str | None]:
        """
        How each layer takes part in the sharing of compressed KV:

        - `"store"` compresses KV of its own, and the layers after it read them.
        - `"reuse_compressed"` reads the compressed KV of the last layer that made some, and attends to the entries
          that the last indexer before it selected.
        - `"reuse_compressed_but_indexer"` reads those compressed KV too, but selects from them with an indexer of
          its own.
        - `None` shares nothing: the layer attends over its sliding window only, or no layer reads the KV that it
          compresses.

        Whatever its mode, every layer computes and caches the KV of its own sliding window.
        """
        # These are not the plain `"reuse"` of the models whose layers share all of their KV. The cache hands a
        # `"reuse"` layer a `DummyLayer` that stores nothing, and ours still have a sliding window to store.
        kv_source_layers = self.kv_source_layers
        kv_sharing_modes = []
        for layer_idx, source in enumerate(kv_source_layers):
            if source is None:
                # A sliding window and nothing else
                kv_sharing_modes.append(None)
            elif source != layer_idx:
                # Reads the compressed KV of an earlier layer, with or without an indexer of its own
                has_indexer = layer_idx in self.index_source_layer_ids
                kv_sharing_modes.append("reuse_compressed_but_indexer" if has_indexer else "reuse_compressed")
            else:
                # Compresses its own KV, which are only shared if a later layer reads them
                kv_sharing_modes.append("store" if kv_source_layers.count(layer_idx) > 1 else None)
        return kv_sharing_modes


@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V4.1-Flash")
@strict
class DeepseekV41Config(PreTrainedConfig):
    model_type = "deepseek_v41"
    sub_configs_defaults = {
        "vision_config": DeepseekV41VisionConfig,
        "text_config": DeepseekV41TextConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 129264
    pad_token_id: int | None = 2
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

        # The quantization config of the checkpoint only describes its FP8 linears and its FP4 experts, so we fill
        # in the rest here, like DeepSeek-V4 does for `expert_dtype`. Anything the config already sets is left alone.
        # TODO: revisit after #50245. With a quantization plan, the tables can get an entry of their own
        quantization_config = getattr(self, "quantization_config", None)
        if isinstance(quantization_config, dict) and quantization_config.get("quant_method") == "fp8":
            block_size = quantization_config.get("weight_block_size") or (128, 128)
            defaults = {
                # The Engram tables are stored in FP8 too, and they are far too big to dequantize. A block of a
                # table is as wide as a block of a linear, but it never spans rows
                "modules_to_convert": ["ple.ngram_embedding"],
                "embedding_block_size": block_size[1],
                # These linears are stored in BF16. They have no scales, so the quantizer has to skip them
                "modules_to_not_convert": [
                    "lm_head",
                    r".*vision_tower",
                    r".*multi_modal_projector",
                    "self_attn.compressor.kv_proj",
                    "self_attn.compressor.gate_proj",
                    "self_attn.indexer.k_proj",
                    "self_attn.indexer.weights_proj",
                ],
            }
            given = {key: value for key, value in quantization_config.items() if value is not None}
            self.quantization_config = {**defaults, **given}


class DeepseekV41CacheLayer(DynamicSlidingWindowLayer):
    r"""
    Cache layer for DeepSeek-V4.1. Every layer keeps its sliding window of KV here. On top of that, a few layers
    keep state that other layers read:

      * `compressed_kv` / `index_keys` - the compressed KV entries that a layer which compresses its own KV has
        emitted so far, and the index keys made from them. The KV-sharing layers after it attend to these too.
        Entry `g` of a row is its `g`-th group of real tokens, so the rows of a padded batch do not have the
        same number of entries. A row with fewer of them than the others has zeros after its last one.
      * `pooling_buffer` - the KV and the gates of the last `compress_ratio - 1` real tokens of each row. The
        last few of them are the tokens of a pooling group that is still filling up. The group only becomes a
        compressed entry once its last token arrives.
      * `token_history` - the last few token ids, which the Engram n-gram hashes look back at. Only the first
        layer keeps it.
    """

    # TODO: #51202 adds cache layers for compressed states to `cache_utils` (`DynamicSlidingWindowCompressedLayer`,
    # with the pooling groups, the compressed entries and the selection behind one API for every indexer model).
    # We're watching for it to merge: the compressed states then move onto that class, and this one and its
    # `"shared_compressed_attention"` layer type go away.
    _layer_type = "shared_compressed_attention"
    # The compressed entries, the pooling buffer and the token history only ever move forward, so this cache
    # cannot go back to an earlier length. That rules out assisted generation and the other modes that crop it.
    is_croppable = False

    def __init__(self, config: "DeepseekV41TextConfig", **kwargs):
        super().__init__(sliding_window=config.sliding_window)
        self.compressed_kv: torch.Tensor | None = None
        self.index_keys: torch.Tensor | None = None
        self.pooling_buffer: torch.Tensor | None = None
        self.token_history: torch.Tensor | None = None

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        """
        Sliding-window update for shared KV: the keys and the values are the same tensor, so we only store it
        once.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self.values = self.keys
        self.cumulative_length += key_states.shape[-2]
        full = torch.cat([self.keys, key_states], dim=-2)
        self.keys = full[:, :, -self.sliding_window + 1 :, :]
        self.values = self.keys
        return full, full

    def update_compressed_states(
        self,
        compressed_kv: torch.Tensor,
        index_keys: torch.Tensor,
        num_entries: torch.Tensor,
        num_new_entries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Write the new compressed KV entries and their index keys after the ones that each row already has, and
        return all of them so far. Row `i` has `num_entries[i]` entries, and takes the first
        `num_new_entries[i]` of the new ones.
        """
        if self.compressed_kv is None:
            self.compressed_kv, self.index_keys = compressed_kv[:, :0], index_keys[:, :0]
        num_entries_after = num_entries + num_new_entries
        width = max(self.compressed_kv.shape[1], int(num_entries_after.max()))
        # Each row writes at its own offset, as the rows of a padded batch do not complete their groups in the
        # same calls. The new entries that a row does not take all land in one extra slot at the end, which we
        # drop again.
        slots = num_entries[:, None] + torch.arange(compressed_kv.shape[1], device=compressed_kv.device)
        slots = torch.where(slots < num_entries_after[:, None], slots, width)
        for name, new_states in (("compressed_kv", compressed_kv), ("index_keys", index_keys)):
            states = getattr(self, name)
            states = F.pad(states, (0, 0, 0, width + 1 - states.shape[1]))
            states = states.scatter(1, slots[..., None].expand_as(new_states), new_states)
            setattr(self, name, states[:, :width])
        return self.compressed_kv, self.index_keys

    def update_token_history(self, token_ids: torch.Tensor, history_length: int) -> torch.Tensor:
        """
        Return the new `token_ids` with the last `history_length` tokens before them, then remember the last
        `history_length` of those for the next call. Before the start of the sequence, the history is `-1`.
        """
        if self.token_history is None:
            self.token_history = token_ids.new_full((token_ids.shape[0], history_length), -1)
        token_ids = torch.cat([self.token_history, token_ids], dim=1)
        self.token_history = token_ids[:, -history_length:]
        return token_ids

    def _extra_state_names(self) -> tuple[str, ...]:
        return ("compressed_kv", "index_keys", "pooling_buffer", "token_history")

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        for name in self._extra_state_names():
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.index_select(0, beam_idx.to(tensor.device)))

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        for name in self._extra_state_names():
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.repeat_interleave(repeats, dim=0))

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        super().batch_select_indices(indices)
        for name in self._extra_state_names():
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor[indices, ...])

    def reset(self) -> None:
        super().reset()
        # Dropped rather than zeroed, as they grow by concatenation, like the main states
        for name in self._extra_state_names():
            setattr(self, name, None)


class DeepseekV41RMSNorm(DeepseekV4RMSNorm):
    pass


class DeepseekV41RotaryEmbedding(DeepseekV4RotaryEmbedding):
    pass


class DeepseekV41GroupedLinear(DeepseekV4GroupedLinear):
    pass


class DeepseekV41Float32Linear(nn.Linear):
    """
    A linear layer that computes in float32, whatever dtype its weight is stored in. The reference promotes the
    pooling projections of the compressor like this.

    The promotion happens in here rather than in the compressor, because offloaded weights (`device_map="auto"`
    on a machine the model does not fit on) are only loaded while the module that owns them runs.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states.float(), self.weight.float())


# The low-precision formats that activations are rounded to, as (largest value, mantissa bits, lowest normal
# exponent). FP8 is E4M3 and FP4 is E2M1.
MINIFLOAT_FORMATS = {"fp8": (448.0, 3, -6), "fp4": (6.0, 1, 0)}


def _round_to_minifloat(values: torch.Tensor, value_format: str) -> torch.Tensor:
    max_value, mantissa_bits, min_exponent = MINIFLOAT_FORMATS[value_format]
    # Casting to these formats saturates, so we clamp before we round
    values = values.clamp(-max_value, max_value)
    # Between two powers of two, the values of the format are evenly spaced, so we round to a multiple of the
    # spacing. `frexp` gives `values = mantissa * 2**exponent` with `0.5 <= |mantissa| < 1`, so the power of two
    # below a value is `2**(exponent - 1)`. Under the lowest normal exponent the spacing stops shrinking, which
    # is where the subnormals are.
    _, exponent = torch.frexp(values)
    spacing = torch.ldexp(torch.ones_like(values), (exponent - 1).clamp(min=min_exponent) - mantissa_bits)
    # `torch.round` sends ties to the even multiple, like the cast does
    return torch.round(values / spacing) * spacing


def fake_quantize(
    hidden_states: torch.Tensor, value_format: str, block_size: int, scale_format: str = "ue8m0"
) -> torch.Tensor:
    """
    Rounds activations to the values that a low-precision float format can hold, and returns them in their own
    dtype. The reference implementation does this to the KV, the attention output, the index keys and queries
    and the compressed KV, mirroring the quantized formats that they are kept in during training. Nothing gets
    smaller: the point is that the model sees the same rounded values as it was trained with.

    The last dim is cut into blocks of `block_size` values, and each block gets a scale of its own. The block
    is divided by its scale, which brings its largest value to the top of the format's range, then rounded to
    the format, then multiplied by the scale again.

    Args:
        hidden_states (`torch.Tensor` of shape `(..., hidden_size)`):
            The activations to round. `hidden_size` must be a multiple of `block_size`.
        value_format (`str`):
            The format to round to: `"fp8"` (E4M3, 3 mantissa bits, up to 448) or `"fp4"` (E2M1, 1 mantissa bit,
            up to 6).
        block_size (`int`):
            Number of consecutive values that share a scale.
        scale_format (`str`, *optional*, defaults to `"ue8m0"`):
            The format of the scales: `"ue8m0"` for a power of two, or `"fp8"` for an FP8 number.

    Returns:
        `torch.Tensor`: The rounded activations, with the shape and dtype of `hidden_states`. The gradient skips
        the rounding and goes straight through to `hidden_states`.

    Example:

    ```python
    >>> import torch
    >>> hidden_states = torch.tensor([[0.3, -1.0, 2.6, 5.5]])
    >>> fake_quantize(hidden_states, value_format="fp4", block_size=4)
    tensor([[ 0.5000, -1.0000,  3.0000,  6.0000]])

    ```
    """
    max_value = MINIFLOAT_FORMATS[value_format][0]
    # The reference computes the scales and the rounding in float32, whatever the dtype of the activations
    blocks = hidden_states.detach().float().unflatten(-1, (-1, block_size))
    largest = blocks.abs().amax(dim=-1, keepdim=True)
    if scale_format == "ue8m0":
        # A power of two: the smallest one that still brings the largest value of the block into range. The
        # floors keep the scale of an all-zero block away from zero.
        largest = largest.clamp(min=1e-4 if value_format == "fp8" else max_value * 2.0**-126)
        # We want `2**ceil(log2(x))` to the bit, which `torch.log2` cannot promise around powers of two.
        # `frexp` can: `x = mantissa * 2**exponent` with `0.5 <= mantissa < 1`, and only a power of two has a
        # mantissa of exactly 0.5.
        mantissa, exponent = torch.frexp(largest * (1.0 / max_value))
        scale = torch.ldexp(torch.ones_like(largest), exponent - (mantissa == 0.5).int())
    elif scale_format == "fp8":
        # An FP8 number, which follows the largest value more closely than a power of two does. The floor is
        # the smallest FP8 number, so again the scale of an all-zero block is not zero.
        scale = _round_to_minifloat(largest.clamp(min=max_value * 2.0**-9) / max_value, "fp8")
    else:
        raise ValueError(f'`scale_format` must be "ue8m0" or "fp8", but got "{scale_format}".')

    quantized = _round_to_minifloat(blocks / scale, value_format) * scale
    quantized = quantized.flatten(-2).to(hidden_states.dtype)
    if hidden_states.requires_grad:
        # Straight-through estimator: rounding has a gradient of zero almost everywhere, so the backward pass
        # treats it as the identity. The term that we add is exactly zero, so the values stay the rounded ones.
        quantized = quantized + (hidden_states - hidden_states.detach())
    return quantized


class DeepseekV41Compressor(nn.Module):
    """
    Pools every `compress_ratio` consecutive tokens into one KV latent with a learned softmax gate. With a
    ratio of 1 there is nothing to pool, so the latent is a plain projection of each token.

    The groups are counted from the first real token of each row, and padding takes no part in them. So a
    left-padded row gets the same latents as it would without the padding, but the rows of a batch do not
    complete their groups in the same calls.

    Returns the latents of the groups that this call completes, *before* RoPE: the indexer makes its keys
    from the unrotated latents, so the attention layer rotates them afterwards. The RoPE of each latent and
    the number of groups that each row completes come back with them, and the latents after that number are
    not real. A group whose last token has not arrived yet waits in the cache, or is dropped when there is no
    cache.
    """

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.compress_ratio = config.compress_ratios[layer_idx]
        if self.compress_ratio > 1:
            # Several tokens per group, which are pooled in float32
            self.kv_proj = DeepseekV41Float32Linear(config.hidden_size, config.head_dim, bias=False)
            self.gate_proj = DeepseekV41Float32Linear(config.hidden_size, config.head_dim, bias=False)
        else:
            # One token per group, so no gate and no pooling
            self.kv_proj = nn.Linear(config.hidden_size, config.head_dim, bias=False)
            self.gate_proj = None
        self.kv_norm = DeepseekV41RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        is_real_token: torch.Tensor,
        num_seen_tokens: torch.Tensor,
        cache_layer: DeepseekV41CacheLayer | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        batch_size = hidden_states.shape[0]
        ratio = self.compress_ratio
        # First, what the groups are made of. The KV and the gate of a token stay together until the pooling,
        # so we keep them in one tensor. The same goes for the cos and sin of the RoPE: for each token, the
        # model has computed the RoPE of the entry that the token completes, if it is the last one of a group.
        if self.gate_proj is None:
            # One token per group, so no gate and no pooling
            states = self.kv_proj(hidden_states)
        else:
            # The reference promotes the pooling weights to float32, and pools in float32 too
            states = torch.cat([self.kv_proj(hidden_states), self.gate_proj(hidden_states)], dim=-1)
        rope = torch.cat(position_embeddings, dim=-1).expand(batch_size, -1, -1)

        if cache_layer is not None and ratio > 1:
            # The tokens that are still waiting for their group to fill up go first. The cache keeps the last
            # `ratio - 1` real tokens of each row, and the ones that are waiting are the last of those.
            if cache_layer.pooling_buffer is None:
                cache_layer.pooling_buffer = states.new_zeros((batch_size, ratio - 1, states.shape[-1]))
            num_waiting = num_seen_tokens % ratio
            is_waiting = torch.arange(ratio - 1, device=states.device) >= ratio - 1 - num_waiting[:, None]
            states = torch.cat([cache_layer.pooling_buffer, states], dim=1)
            is_real_token = torch.cat([is_waiting, is_real_token], dim=1)
            rope = F.pad(rope, (0, 0, ratio - 1, 0))

        # Next, we bring the real tokens of each row to the front, in the order that they came in. Group `g` of
        # a row is then its real tokens `g * ratio` to `(g + 1) * ratio - 1`, whatever padding the row has,
        # and all that is not real ends up after the row's last whole group.
        order = (~is_real_token).long().argsort(dim=1, stable=True)
        states = states.gather(1, order[..., None].expand_as(states))
        rope = rope.gather(1, order[..., None].expand_as(rope))
        num_real_tokens = is_real_token.sum(dim=1)

        if cache_layer is not None and ratio > 1:
            # The last `ratio - 1` real tokens of each row go back to the cache. A row with fewer of them reads
            # its first token several times, but those copies come before the tokens that are waiting.
            last_tokens = num_real_tokens[:, None] - (ratio - 1) + torch.arange(ratio - 1, device=states.device)
            last_tokens = last_tokens.clamp(min=0)[..., None].expand(-1, -1, states.shape[-1])
            cache_layer.pooling_buffer = states.gather(1, last_tokens)

        # Finally, we pool the whole groups. The others are dropped here, and a row that has fewer real groups
        # than the longest row gets latents that are made of padding, which the caller must not use.
        usable = (states.shape[1] // ratio) * ratio
        if self.gate_proj is None:
            latents = self.kv_norm(states)
        else:
            kv, gate = states[:, :usable].unflatten(1, (-1, ratio)).chunk(2, dim=-1)
            pooled = (kv * gate.softmax(dim=2)).sum(dim=2)
            latents = self.kv_norm(pooled.to(hidden_states.dtype))
        cos, sin = rope[:, ratio - 1 : usable : ratio].chunk(2, dim=-1)
        return latents, (cos, sin), num_real_tokens // ratio


def select_candidate_blocks(
    scores: torch.Tensor, num_visible: torch.Tensor, topk_blocks: int, block_size: int
) -> torch.Tensor:
    """
    First level of the two-level top-k: keep the `topk_blocks` best blocks of compressed positions per query.

    Args:
        scores (`torch.Tensor` of shape `(batch_size, seq_len, num_entries)`):
            Index scores, with the entries a query cannot see yet already at `-inf`.
        num_visible (`torch.Tensor` of shape `(batch_size, seq_len)`):
            How many compressed entries each query can see.
        topk_blocks (`int`):
            Number of blocks to keep per query.
        block_size (`int`):
            Number of compressed positions per block.

    Returns:
        `torch.Tensor`: A boolean mask shaped like `scores`, `True` for the entries inside a kept block.

    Example:

    ```python
    >>> import torch
    >>> scores = torch.tensor([[[0.1, 0.9, 0.3, 0.2, 0.8, float("-inf")]]])
    >>> select_candidate_blocks(scores, num_visible=torch.tensor([[5]]), topk_blocks=2, block_size=2)
    tensor([[[ True,  True, False, False,  True,  True]]])

    ```
    """
    num_entries = scores.shape[-1]
    # Score each block by its best position, with `-inf` padding the last block out to `block_size`
    block_scores = F.pad(scores, (0, -num_entries % block_size), value=float("-inf"))
    block_scores = block_scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = block_scores.shape[-1]

    # The block with the query's newest position is only partly filled, so we always keep it: it holds the
    # most recent tokens, but an older full block could outscore it
    last_block = (num_visible.unsqueeze(-1) - 1) // block_size
    is_last_block = torch.arange(num_blocks, device=scores.device) == last_block
    block_scores = block_scores.masked_fill(is_last_block, float("inf"))

    top = block_scores.topk(min(topk_blocks, num_blocks), dim=-1)
    # With fewer visible blocks than `topk_blocks`, the leftover picks come back as `-inf`, so we drop them
    keep = torch.zeros_like(block_scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :num_entries]


class DeepseekV41Indexer(nn.Module):
    """
    Picks the `index_topk` compressed entries that each query attends to.

    This is a small side attention: several query heads are scored against one shared key per compressed
    entry, the scores are rectified, and then combined across heads with per-token weights. The keys are made
    from the compressor's latents, so only the indexer of a layer that compresses its own KV has the modules to
    make them (`k_proj` and `k_norm`). The other indexers read those same keys.

    With a candidate source this is the second of two levels, see `select_candidate_blocks` for the first.
    """

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.scaling = self.head_dim**-0.5 * self.num_heads**-0.5
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        if layer_idx in config.kv_source_layer_ids:
            self.k_proj = nn.Linear(config.head_dim, self.head_dim, bias=False)
            self.k_norm = DeepseekV41RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        candidate_source = config.candidate_source_layer_id
        self.is_candidate_source = candidate_source is not None and layer_idx == candidate_source
        self.uses_candidates = candidate_source is not None and layer_idx > candidate_source
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        index_keys: torch.Tensor,
        num_real_tokens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        candidates: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = hidden_states.shape
        num_entries = index_keys.shape[1]
        if num_entries == 0:
            # No group is complete yet, so there is nothing to select
            empty = torch.zeros((batch_size, seq_len, 0), dtype=torch.bool, device=hidden_states.device)
            return empty, empty if self.is_candidate_source else candidates

        cos, sin = position_embeddings
        queries = self.q_b_proj(q_residual).view(batch_size, seq_len, self.num_heads, self.head_dim)
        queries = apply_rotary_pos_emb(queries, cos, sin, unsqueeze_dim=2)
        # The index keys are FP4 as well, see the attention layer that makes them
        queries = fake_quantize(queries, "fp4", block_size=32)
        scores = torch.einsum("bshd,btd->bsht", queries, index_keys.to(queries.dtype))
        weights = self.weights_proj(hidden_states) * self.scaling
        scores = (F.relu(scores) * weights.unsqueeze(-1)).sum(dim=2)

        # A query sees a compressed entry once it has passed the last token of the entry's group. The groups are
        # made of real tokens only, so we count those, up to and including the query. This is also what hides
        # the slots after the last entry of a row that has fewer entries than the others.
        num_visible = num_real_tokens // self.compress_ratio
        visible = torch.arange(num_entries, device=scores.device) < num_visible.unsqueeze(-1)
        scores = scores.masked_fill(~visible, float("-inf"))

        if self.is_candidate_source:
            # First level: this layer chooses the blocks that the indexers after it may select from
            candidates = select_candidate_blocks(
                scores, num_visible, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates:
            # Second level: we score with our own weights, but only inside the source's candidate blocks
            scores = scores.masked_fill(~candidates, float("-inf"))

        top_scores, top_indices = scores.topk(min(self.index_topk, num_entries), dim=-1)
        # Queries with fewer reachable entries than `index_topk` get `-inf` picks back, which we drop
        selected = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top_indices, top_scores > float("-inf"))
        return selected, candidates


class DeepseekV41Attention(nn.Module):
    r"""
    Attention over a sliding window, which is all that the layers with a compression ratio of 0 do. The other
    layers also attend to compressed KV entries, which go after the window on the KV axis. What they do for them
    depends on their entry in `config.kv_sharing_modes`:

    * A layer that compresses KV of its own also makes the index keys for them and runs an indexer over those
      keys. With `"store"`, the KV-sharing layers after it read all three: the compressed KV, the index keys and
      the selection.
    * `"reuse_compressed_but_indexer"` reads the compressed KV of the last layer that made some, and runs its own
      indexer to select which of them each query attends to.
    * `"reuse_compressed"` reads those compressed KV too, and reuses the selection of the last layer that ran an
      indexer.

    Shared with DeepSeek-V4:
    * Shared-KV multi-query attention: there is a single KV head, and the same tensor is read as both key and
      value.
    * Partial RoPE on the last `qk_rope_head_dim` dims of each head. Since the values carry RoPE too, the
      output is rotated back at the query's position before the output projection.
    * A learnable attention sink per head, and a grouped low-rank output projection.
    """

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.is_sliding = self.compress_ratio == 0
        # TODO: one class that branches on the sharing mode is how the models on `main` do it. #50189 would
        # replace the modes with an `attention_layer_types` field and one attention class per entry. We're
        # watching for it to merge, to split this class the same way.
        kv_sharing_mode = config.kv_sharing_modes[layer_idx]
        self.kv_source_layer = config.kv_source_layers[layer_idx]
        self.is_kv_shared_layer = kv_sharing_mode in ("reuse_compressed", "reuse_compressed_but_indexer")
        self.store_compressed_kv = kv_sharing_mode == "store"
        # A layer that reads compressed KV uses their RoPE base for everything, its window included
        self.rope_layer_type = "main" if self.is_sliding else "compress"
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads  # single KV head, broadcast to all
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = DeepseekV41RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV41RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_a_proj = DeepseekV41GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups, config.o_groups * config.o_lora_rank, config.o_groups
        )
        self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        # Only the layers that compress their own KV have a compressor. They index them too, and so do the
        # KV-sharing layers that the config lists as index sources.
        has_compressor = not self.is_sliding and not self.is_kv_shared_layer
        has_indexer = has_compressor or kv_sharing_mode == "reuse_compressed_but_indexer"
        self.indexer = DeepseekV41Indexer(config, layer_idx) if has_indexer else None
        self.compressor = DeepseekV41Compressor(config, layer_idx) if has_compressor else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
        attention_mask: torch.Tensor,
        is_real_token: torch.Tensor,
        num_seen_tokens: torch.Tensor,
        cache: Cache | None = None,
        shared_kv_states: KVSharingDict | None = None,
        selected_entries: torch.Tensor | None = None,
        candidate_entries: torch.Tensor | None = None,
        compressed_position_embeddings: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # The model computes RoPE for both bases, and each layer picks the one that matches its attention type
        cos, sin = position_embeddings[self.rope_layer_type]

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = apply_rotary_pos_emb(q, cos, sin)

        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)
        # The KV are FP8 from here on, in the cache too. They are rounded after RoPE, so on the rotated dims
        # it is the rotated values that land on the FP8 grid.
        kv = fake_quantize(kv, "fp8", block_size=32)
        if cache is not None:
            kv = cache.update(kv, kv, self.layer_idx)[0]

        if self.is_kv_shared_layer:
            # This layer reads the compressed KV of an earlier layer, and the index keys made from them
            compressed_kv, index_keys = shared_kv_states[str(self.kv_source_layer)]
        elif self.compressor is not None:
            # This layer compresses the groups that this call completes. A row may complete none of them, e.g.
            # when decoding a token that is not the last of its group.
            cache_layer: DeepseekV41CacheLayer | None = cache.layers[self.layer_idx] if cache is not None else None
            latents, (entry_cos, entry_sin), num_new_entries = self.compressor(
                hidden_states,
                compressed_position_embeddings[self.compress_ratio],
                is_real_token,
                num_seen_tokens,
                cache_layer,
            )
            # The index keys come from the latents before RoPE, so we make them before we rotate the latents
            index_keys = self.indexer.k_norm(self.indexer.k_proj(latents))
            index_keys = apply_rotary_pos_emb(index_keys.unsqueeze(1), entry_cos, entry_sin).squeeze(1)
            compressed_kv = apply_rotary_pos_emb(latents.unsqueeze(1), entry_cos, entry_sin).squeeze(1)
            # Both are FP4 from here on. The compressed KV get a finer scale than the index keys: one per 16
            # values instead of 32, and an FP8 number instead of a power of two.
            index_keys = fake_quantize(index_keys, "fp4", block_size=32)
            compressed_kv = fake_quantize(compressed_kv, "fp4", block_size=16, scale_format="fp8")
            if cache_layer is not None:
                # Each row has one entry per whole group of the tokens that it had seen before this call
                compressed_kv, index_keys = cache_layer.update_compressed_states(
                    compressed_kv, index_keys, num_seen_tokens // self.compress_ratio, num_new_entries
                )
            if self.store_compressed_kv:
                # The KV-sharing layers after us read both under our layer index. No other layer writes there,
                # so they still find them when gradient checkpointing runs them again in the backward pass.
                shared_kv_states[str(self.layer_idx)] = compressed_kv, index_keys

        if self.indexer is not None:
            # This layer selects which compressed entries each query attends to, and the `"reuse_compressed"`
            # layers after it reuse the selection
            selected_entries, candidate_entries = self.indexer(
                hidden_states,
                q_residual,
                index_keys,
                num_seen_tokens[:, None] + is_real_token.cumsum(dim=-1),
                (cos, sin),
                candidate_entries,
            )

        if not self.is_sliding:
            # The compressed entries go after the window on the KV axis. The model built the mask for the
            # window only, so we extend it with the entries that each query selected.
            kv = torch.cat([kv, compressed_kv.unsqueeze(1).to(kv.dtype)], dim=2)
            compressed_mask = torch.zeros_like(selected_entries, dtype=attention_mask.dtype)
            compressed_mask = compressed_mask.masked_fill(~selected_entries, torch.finfo(attention_mask.dtype).min)
            attention_mask = torch.cat([attention_mask, compressed_mask.unsqueeze(1)], dim=-1)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            q,
            kv,
            kv,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,
            **kwargs,
        )

        # The values are the keys, so they picked up RoPE too. We undo it on the output with the conjugate
        # rotation at the query's position, before the grouped output projection mixes the heads.
        attn_output = apply_rotary_pos_emb(attn_output, cos, -sin, unsqueeze_dim=2)

        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        # The grouped projection was trained as an FP8 matmul, which rounds its input to FP8
        grouped = fake_quantize(grouped, "fp8", block_size=32)
        grouped = self.o_a_proj(grouped).flatten(2)
        output = self.o_b_proj(grouped)
        return output, attn_weights, selected_entries, candidate_entries


class DeepseekV41HyperConnection(nn.Module):
    r"""
    Computes the hyper-connection weights of one sublayer (attention or MLP) from the `hc_mult` residual
    streams. As in DeepSeek-V4, there are three of them:

    - pre: collapses the `hc_mult` streams into the single input of a sublayer
    - post: expands the output of the sublayer back into `hc_mult` streams
    - comb: mixes the `hc_mult` input streams into the output streams, and is made doubly stochastic with a few
      iterations of the Sinkhorn-Knopp algorithm

    Unlike DeepSeek-V4, "pre" is not used by the sublayer that these weights belong to. It collapses the
    streams for the *next* sublayer, so we return it instead of applying it here. See
    `DeepseekV41DecoderLayer` for how the weights are passed along.
    """

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        # The three weights come out of a single projection: pre and post have `hc_mult` entries each, and comb
        # is a `hc_mult x hc_mult` matrix
        concatenated_weights_size = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(concatenated_weights_size, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(concatenated_weights_size))
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hc = self.hc_mult
        # The streams are normalized together, with one statistic per token
        flattened = hidden_streams.flatten(2).float()
        rsqrt = torch.rsqrt(flattened.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flattened, self.fn.float()) * rsqrt

        pre_w, post_w, comb_w = mixes.split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.float().split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.float().unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb = comb_w.unflatten(-1, (hc, hc)) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb, dim=-1) + self.hc_eps

        # Sinkhorn-Knopp: alternately normalize the columns and the rows, so that both end up summing to 1
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        return pre, post, comb


class DeepseekV41MLP(DeepseekV4MLP):
    pass


@use_experts_implementation
class DeepseekV41Experts(DeepseekV4Experts):
    pass


class DeepseekV41TopKRouter(DeepseekV4TopKRouter):
    """
    Top-k router. The correction bias only steers which experts are selected, and the routing weights come
    from the unbiased scores. The tokens of an image use a separate bias.
    """

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__(config)
        self.e_score_correction_bias_vl = nn.Buffer(torch.zeros(self.num_experts, dtype=torch.float32))

    def forward(
        self, hidden_states: torch.Tensor, image_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        # The reference routes in float32, and the selection is sensitive enough that bfloat16 changes it
        logits = F.linear(flat.float(), self.weight.float())
        scores = self.score_fn(logits)
        bias = self.e_score_correction_bias
        if image_mask is not None:
            bias = torch.where(image_mask.reshape(-1, 1), self.e_score_correction_bias_vl, bias)
        indices = torch.topk(scores + bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV41SparseMoeBlock(nn.Module):
    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__()
        self.gate = DeepseekV41TopKRouter(config)
        self.experts = DeepseekV41Experts(config)
        self.shared_experts = DeepseekV41MLP(config)

    def forward(self, hidden_states: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        _, weights, indices = self.gate(hidden_states, image_mask)
        routed = self.experts(hidden_states.view(-1, hidden_dim), indices, weights)
        return routed.view(batch_size, seq_len, hidden_dim) + self.shared_experts(hidden_states)


def _is_prime(number: int) -> bool:
    if number < 2:
        return False
    if number % 2 == 0:
        return number == 2
    divisor = 3
    while divisor * divisor <= number:
        if number % divisor == 0:
            return False
        divisor += 2
    return True


def _ngram_bucket_sizes(config: DeepseekV41TextConfig) -> list[list[int]]:
    # Every (n-gram size, head) pair of every Engram layer hashes into its own range of buckets. The size of
    # each range is a prime, and no prime is used twice across the whole model. For each n-gram size the
    # search restarts just below `ngram_vocab_size_base`, so we end up handing out consecutive primes.
    bucket_sizes, seen = [], set()
    for _ in config.ple_layer_ids:
        layer_sizes = []
        for _ in range(config.ngram_size - 1):
            candidate = config.ngram_vocab_size_base - 1
            for _ in range(config.heads_per_ngram):
                candidate += 1
                while candidate in seen or not _is_prime(candidate):
                    candidate += 1
                seen.add(candidate)
                layer_sizes.append(candidate)
        bucket_sizes.append(layer_sizes)
    return bucket_sizes


def _ngram_hash_multipliers(config: DeepseekV41TextConfig) -> np.ndarray:
    # One multiplier per (layer, lookback), from a per-layer RNG so that the layers hash differently. They
    # are kept odd, and bounded so that `token_id * multiplier` cannot overflow int64.
    bound = max(1, (np.iinfo(np.int64).max // config.ngram_compressed_vocab_size) // 2)
    rows = []
    for layer_idx in config.ple_layer_ids:
        generator = np.random.default_rng(10007 * layer_idx)
        values = generator.integers(low=0, high=bound, size=(config.ngram_size,), dtype=np.int64)
        rows.append(values * 2 + 1)
    return np.stack(rows)


def _default_ngram_token_map(config: DeepseekV41TextConfig) -> torch.Tensor:
    if config.ngram_compressed_vocab_size == config.vocab_size:
        # A compressed vocab as large as the vocab itself merges no tokens, so the map is the identity
        return torch.arange(config.vocab_size)
    # Otherwise the map comes from the tokenizer (see `build_ngram_token_map`), and -1 marks it as missing
    return torch.full((config.vocab_size,), -1, dtype=torch.long)


def build_ngram_token_map(tokenizer) -> torch.Tensor:
    """
    Builds the map from token ids to the smaller id space that the Engram n-grams are hashed over. Tokens that
    normalize alike are merged there, so that " The", "the" and "THE" all hash the same way.

    The DeepSeek-V4.1 checkpoints do not store this map. The reference implementation builds it from the
    tokenizer when it creates the model, and this function does the same.

    Args:
        tokenizer ([`PreTrainedTokenizerFast`]):
            The tokenizer of the checkpoint.

    Returns:
        `torch.Tensor` of shape `(len(tokenizer),)`: The compressed id of each token.

    Example:

    ```python
    >>> from transformers import AutoTokenizer
    >>> from transformers.models.deepseek_v41.modeling_deepseek_v41 import build_ngram_token_map

    >>> tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V4.1-Flash")
    >>> token_map = build_ngram_token_map(tokenizer)
    >>> token_map[tokenizer.convert_tokens_to_ids("The")] == token_map[tokenizer.convert_tokens_to_ids("the")]
    tensor(True)

    ```
    """
    from tokenizers import Regex, normalizers

    # A private-use char, so that a token that is exactly one space survives `Strip()` instead of collapsing
    # to the empty string and merging with unrelated tokens
    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # We decode with the raw Rust tokenizer, which is what training did (no `clean_up_tokenization_spaces`)
    backend = tokenizer.backend_tokenizer
    key_to_compressed_id: dict[str, int] = {}
    token_map = []
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            # A partial UTF-8 byte token has nothing to normalize, so we key it by its raw form
            key = backend.id_to_token(token_id)
        else:
            key = normalizer.normalize_str(text) or text
        token_map.append(key_to_compressed_id.setdefault(key, len(key_to_compressed_id)))
    return torch.tensor(token_map, dtype=torch.long)


class DeepseekV41NGramHasher(nn.Module):
    """
    Maps each position to the hash ids of the n-grams that end there, for all Engram layers at once.

    Token ids first go through `token_map`, then each position is hashed together with the
    `ngram_size - 1` tokens before it. The look-back stops at the start of the sequence and at any
    masked token (padding or an image), so an n-gram never spans one.
    """

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__()
        self.config = config
        self.ngram_size = config.ngram_size
        self.num_heads = config.heads_per_ngram
        self.pad_token_id = config.ngram_pad_token_id
        bucket_sizes = torch.tensor(_ngram_bucket_sizes(config))
        self.bucket_sizes = nn.Buffer(bucket_sizes, persistent=False)
        self.offsets = nn.Buffer(bucket_sizes.cumsum(-1) - bucket_sizes, persistent=False)
        self.multipliers = nn.Buffer(torch.from_numpy(_ngram_hash_multipliers(config)), persistent=False)
        # Loaded with the weights if the checkpoint has it, and set with `set_ngram_token_map` otherwise. An
        # entry of -1 is one that neither has filled in yet. A buffer that is `None` would say "not set" more
        # plainly, but it is not part of the state dict, so a checkpoint could never load into it.
        self.token_map = nn.Buffer(_default_ngram_token_map(config))

    def forward(
        self,
        input_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        cache_layer: DeepseekV41CacheLayer | None = None,
    ) -> torch.Tensor:
        compressed_ids = self.token_map[input_ids]
        if (compressed_ids < 0).any():
            raise ValueError(
                "The Engram token map has no entry for some of the input ids. The DeepSeek-V4.1 checkpoints do "
                "not store this map, so it has to be built from the tokenizer once after loading the model:\n"
                "`model.set_ngram_token_map(tokenizer)`"
            )
        # A masked token takes no part in any n-gram. We mark it with -1, like the start of the sequence.
        if token_mask is not None:
            compressed_ids = torch.where(token_mask, compressed_ids, -1)
        history_length = self.ngram_size - 1
        if cache_layer is not None:
            compressed_ids = cache_layer.update_token_history(compressed_ids, history_length)
        else:
            compressed_ids = F.pad(compressed_ids, (history_length, 0), value=-1)

        # One window per position: the token itself comes first, then the tokens before it, nearest first.
        # Once the look-back hits a -1, that slot and all the ones behind it read as padding.
        windows = compressed_ids.unfold(1, self.ngram_size, 1).flip(-1)
        blocked = (windows == -1).cummax(dim=-1).values
        windows = torch.where(blocked, self.token_map[self.pad_token_id], windows)

        # We XOR the multiplied ids together one look-back at a time, so the running value after step `i` is
        # the hash of the (i+1)-gram. Each one lands in its own prime-sized range of buckets.
        products = windows.unsqueeze(2) * self.multipliers
        rolling, hash_ids = products[..., 0], []
        for ngram_index in range(history_length):
            rolling = torch.bitwise_xor(rolling, products[..., ngram_index + 1])
            sizes = self.bucket_sizes[:, ngram_index * self.num_heads : (ngram_index + 1) * self.num_heads]
            hash_ids.append(rolling.unsqueeze(-1) % sizes)
        return torch.cat(hash_ids, dim=-1) + self.offsets


class DeepseekV41PLELayer(nn.Module):
    """
    DeepSeek's Engram, which is a per-layer embedding (PLE): it writes an n-gram lookup into the residual
    streams, gated by how well it matches each stream.

    The hash ids fetch one row of the table per (n-gram size, head) pair. `kv_proj` turns the concatenated
    rows into one key per stream plus a value that the streams share, and the gate of a stream is the
    normalized dot product of the stream with its key.
    """

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        ple_index = config.ple_layer_ids.index(layer_idx)
        num_embeddings = sum(_ngram_bucket_sizes(config)[ple_index])
        if config.ngram_num_embeddings is not None and config.ngram_num_embeddings[ple_index] != num_embeddings:
            raise ValueError(
                f"The Engram table of layer {layer_idx} needs {num_embeddings} rows, which is the sum of its bucket "
                f"sizes, but `ngram_num_embeddings` says {config.ngram_num_embeddings[ple_index]}. The bucket "
                "sizes come from `ngram_vocab_size_base`, `heads_per_ngram`, `ngram_size` and "
                "`ple_layer_ids`, so one of those does not match the checkpoint."
            )
        num_lookups = (config.ngram_size - 1) * config.heads_per_ngram
        self.ngram_embedding = nn.Embedding(num_embeddings, config.ngram_head_dim)
        self.kv_proj = nn.Linear(
            num_lookups * config.ngram_head_dim, config.hidden_size * (config.hc_mult + 1), bias=False
        )
        self.q_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))
        self.k_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))

    def forward(
        self, hidden_streams: torch.Tensor, hash_ids: torch.Tensor, token_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        key_value = self.kv_proj(self.ngram_embedding(hash_ids).flatten(-2).to(hidden_streams.dtype))
        key, value = key_value.split([self.hc_mult * self.hidden_size, self.hidden_size], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.hidden_size))
        streams = hidden_streams.float()
        # The two weights are only ever used as a product
        weight = self.q_weight.float() * self.k_weight.float()
        # Normalized per (token, stream) over the hidden dim, NOT jointly over the streams
        rsqrt = torch.rsqrt(streams.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (streams * weight * key).sum(-1) * rsqrt * self.hidden_size**-0.5
        # Signed square root before the sigmoid, as in training
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (streams + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden_streams.dtype)


class DeepseekV41DecoderLayer(GradientCheckpointingLayer):
    r"""
    DeepSeek-V4.1 decoder block. As in DeepSeek-V4, the residual is a stack of `hc_mult` parallel streams of
    shape `[B, S, hc_mult, D]`, which hyper-connections collapse into the input of each sublayer and expand
    back out afterwards.

    What changes is where the collapsing weights come from. Each sublayer's hyper-connection computes "pre",
    "post" and "comb" from the streams that enter the sublayer, but only "post" and "comb" are used there. Its
    "pre" collapses the streams for the *next* sublayer: the attention uses the "pre" of the previous layer's
    MLP, and the MLP uses the "pre" of this layer's attention. So the layer takes one "pre" in and returns one.
    """

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV41Attention(config, layer_idx)
        self.mlp = DeepseekV41SparseMoeBlock(config)
        self.input_layernorm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = DeepseekV41HyperConnection(config).to(dtype=torch.float32)
        self.ffn_hc = DeepseekV41HyperConnection(config).to(dtype=torch.float32)
        self.ple = DeepseekV41PLELayer(config, layer_idx) if layer_idx in config.ple_layer_ids else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        pre: torch.Tensor,
        ngram_hash_ids: torch.Tensor | None = None,
        ngram_mask: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        selected_entries: torch.Tensor | None = None,
        candidate_entries: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # hidden_states throughout: [B, S, hc_mult, hidden]. The hyper-connection weights are float32, and
        # comb is used transposed: sum_j comb[j, k] * residual[j], which is comb.T @ residual.
        dtype = hidden_states.dtype
        if self.ple is not None:
            hidden_states = self.ple(hidden_states, ngram_hash_ids, ngram_mask)

        attn_pre, post, comb = self.attn_hc(hidden_states)
        collapsed = (pre.unsqueeze(-1) * hidden_states.float()).sum(dim=2).to(dtype)
        attn_output, _, selected_entries, candidate_entries = self.self_attn(
            self.input_layernorm(collapsed),
            selected_entries=selected_entries,
            candidate_entries=candidate_entries,
            **kwargs,
        )
        hidden_states = post.unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.transpose(-1, -2), hidden_states.float()
        )
        hidden_states = hidden_states.to(dtype)

        mlp_pre, post, comb = self.ffn_hc(hidden_states)
        collapsed = (attn_pre.unsqueeze(-1) * hidden_states.float()).sum(dim=2).to(dtype)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), image_mask=image_mask)
        hidden_states = post.unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
            comb.transpose(-1, -2), hidden_states.float()
        )
        return hidden_states.to(dtype), mlp_pre, selected_entries, candidate_entries


def vision_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float | int = 0.0,
    **kwargs,
):
    # Plain attention. The eager function of the language model adds an attention sink, which the vision encoder
    # does not have.
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


class DeepseekV41VisionAttention(nn.Module):
    """Bidirectional attention over the patches of one image, with 2D RoPE."""

    def __init__(self, config: DeepseekV41VisionConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_patches = hidden_states.shape[0]
        query, key, value = self.qkv_proj(hidden_states).view(num_patches, 3, self.num_heads, self.head_dim).unbind(1)
        query, key = _apply_vision_rotary(query, cos, sin), _apply_vision_rotary(key, cos, sin)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, vision_eager_attention_forward
        )
        # The attention functions take a batch, which here is the one image
        attn_output, attn_weights = attention_interface(
            self,
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            attention_mask=None,
            scaling=self.scaling,
            is_causal=False,
        )
        # Only eager attention returns the weights
        attn_weights = attn_weights.squeeze(0) if attn_weights is not None else None
        return self.o_proj(attn_output.reshape(num_patches, -1)), attn_weights


def _apply_vision_rotary(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # The first half of each head is rotated against the second half, in float32
    first, second = hidden_states.float().chunk(2, dim=-1)
    rotated = torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)
    return rotated.to(hidden_states.dtype)


class DeepseekV41VisionMLP(nn.Module):
    def __init__(self, config: DeepseekV41VisionConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class DeepseekV41VisionLayer(GradientCheckpointingLayer):
    def __init__(self, config: DeepseekV41VisionConfig):
        super().__init__()
        self.layer_norm1 = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DeepseekV41VisionAttention(config)
        self.layer_norm2 = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = DeepseekV41VisionMLP(config)

    def forward(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.self_attn(self.layer_norm1(hidden_states), cos, sin)
        hidden_states = hidden_states + attn_output
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class DeepseekV41MultiModalProjector(nn.Module):
    """Merges each `downsample_ratio x downsample_ratio` square of patches into one language model token."""

    def __init__(self, config: "DeepseekV41Config"):
        super().__init__()
        self.downsample_ratio = config.vision_config.downsample_ratio
        input_size = config.vision_config.hidden_size * self.downsample_ratio**2
        self.linear_1 = nn.Linear(input_size, config.text_config.hidden_size)
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)

    def forward(self, image_features: torch.Tensor, height: int, width: int) -> torch.Tensor:
        ratio = self.downsample_ratio
        image_features = image_features.view(height, width, -1).permute(2, 0, 1)
        # A grid that is not a multiple of the ratio is zero-padded at the bottom and on the right
        image_features = F.pad(image_features, (0, -width % ratio, 0, -height % ratio))
        image_features = F.unfold(image_features.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1)
        return self.linear_2(F.gelu(self.linear_1(image_features)))


@auto_docstring
class DeepseekV41PreTrainedModel(PreTrainedModel):
    config: DeepseekV41Config
    base_model_prefix = "model"
    input_modalities = ("image", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV41DecoderLayer", "DeepseekV41VisionLayer"]
    # Eager only, for the same reasons as DeepSeek-V4: the head dim of 512 is above what FlashAttention
    # supports, SDPA has no attention sink, and the compressed entries are added to the KV axis inside the
    # attention, after the mask was built.
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    # The compressed KV, the pooling buffers and the Engram token history live in `DeepseekV41CacheLayer`,
    # which has no static version
    _can_compile_fullgraph = False
    # Everything listed here is also created in float32 in its `__init__`. `from_pretrained` enforces this
    # list, but a model built straight from a config would otherwise get the default dtype
    _keep_in_fp32_modules_strict = [
        "attn_hc",
        "ffn_hc",
        "sinks",
        "e_score_correction_bias",
        "e_score_correction_bias_vl",
    ]
    # The DSpark draft layers are not supported yet
    _keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]
    # The original checkpoints do not store the Engram token map, which is not a weight that went missing: the
    # map stays unset, and the model asks for `set_ngram_token_map` when it is called. A checkpoint that does
    # store the map loads it like any other buffer.
    _keys_to_ignore_on_load_missing = [r"ngram_hasher\.token_map"]
    _can_record_outputs = {
        "router_logits": OutputRecorder(DeepseekV41TopKRouter, index=0),
        "hidden_states": OutputRecorder(DeepseekV41DecoderLayer, index=0),
        "attentions": OutputRecorder(DeepseekV41Attention, index=1),
    }

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.get_text_config().initializer_range
        if isinstance(module, DeepseekV41TopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.e_score_correction_bias)
            init.zeros_(module.e_score_correction_bias_vl)
        elif isinstance(module, DeepseekV41Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, DeepseekV41Attention):
            init.zeros_(module.sinks)
        elif isinstance(module, DeepseekV41HyperConnection):
            init.normal_(module.fn, mean=0.0, std=std)
            init.zeros_(module.base)
            init.ones_(module.scale)
        elif isinstance(module, DeepseekV41PLELayer):
            init.ones_(module.q_weight)
            init.ones_(module.k_weight)
        elif isinstance(module, DeepseekV41NGramHasher):
            bucket_sizes = torch.tensor(_ngram_bucket_sizes(module.config))
            init.copy_(module.bucket_sizes, bucket_sizes)
            init.copy_(module.offsets, bucket_sizes.cumsum(-1) - bucket_sizes)
            init.copy_(module.multipliers, torch.from_numpy(_ngram_hash_multipliers(module.config)))
            init.copy_(module.token_map, _default_ngram_token_map(module.config))
        elif isinstance(module, DeepseekV41Model):
            init.normal_(module.image_start, mean=0.0, std=std)
            init.normal_(module.image_end, mean=0.0, std=std)
            init.normal_(module.image_newline, mean=0.0, std=std)
        elif isinstance(module, DeepseekV41RotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)

    def activation_checkpointing_enable(
        self, activation_checkpointing_kwargs=None, every_n_layers: int = 1, offload: bool = False
    ):
        """
        Same as [`PreTrainedModel.activation_checkpointing_enable`], except that `use_reentrant=True` is refused.

        Reentrant checkpointing only tracks the gradient of the tensors that a layer gets as positional arguments.
        The compressed KV reaches the KV-sharing layers in a dict, so the layer that compressed it would silently
        lose the gradient that comes back through them.
        """
        if activation_checkpointing_kwargs is not None and activation_checkpointing_kwargs.get("use_reentrant"):
            raise ValueError(
                "DeepSeek-V4.1 does not support activation checkpointing with `use_reentrant=True`, because the "
                "gradient of the compressed KV that its layers share would be lost. Use `use_reentrant=False`, "
                "which is the default.\n"
                f"activation_checkpointing_kwargs: {activation_checkpointing_kwargs}"
            )
        super().activation_checkpointing_enable(
            activation_checkpointing_kwargs, every_n_layers=every_n_layers, offload=offload
        )

    # The structure linter reads any `set_*` method as a hyperparameter that should live on the config. This one
    # fills a buffer that is saved and loaded with the weights, and it only exists because the original
    # checkpoints left that buffer out.
    def set_ngram_token_map(self, tokenizer_or_token_map):  # trf-ignore: TRF033
        """
        Sets the map from token ids to compressed ids that the Engram n-grams are hashed over. The checkpoints do
        not store it, so it has to be set once after loading the model.

        Args:
            tokenizer_or_token_map ([`PreTrainedTokenizerFast`] or `torch.Tensor`):
                The tokenizer of the checkpoint, or a map that `build_ngram_token_map` made from it.
        """
        hasher = next((module for module in self.modules() if isinstance(module, DeepseekV41NGramHasher)), None)
        if hasher is None:
            raise ValueError("This model has no Engram layers, so there is no token map to set.")
        if isinstance(tokenizer_or_token_map, torch.Tensor):
            # An existing map, e.g. one that was built earlier and saved
            token_map = tokenizer_or_token_map
        else:
            # A tokenizer, which we build the map from in the same way as the reference implementation
            token_map = build_ngram_token_map(tokenizer_or_token_map)
        num_compressed_ids = int(token_map.max()) + 1
        if num_compressed_ids != hasher.config.ngram_compressed_vocab_size:
            raise ValueError(
                f"The token map has {num_compressed_ids} compressed ids, but the model was trained with "
                f"{hasher.config.ngram_compressed_vocab_size} (`config.ngram_compressed_vocab_size`). The hash "
                "multipliers are derived from that number, so this is not the tokenizer the checkpoint was "
                "trained with."
            )
        if token_map.shape[0] > hasher.token_map.shape[0]:
            raise ValueError(
                f"The token map has {token_map.shape[0]} entries, but the model only has "
                f"{hasher.token_map.shape[0]} tokens (`config.vocab_size`)."
            )
        hasher.token_map[: token_map.shape[0]] = token_map.to(hasher.token_map.device)

    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ):
        """
        Not supported: the Engram layers look token ids up in a token map and hash them into n-gram tables, and
        both are fixed to the vocabulary that the checkpoint was trained with.
        """
        # A new token would need an entry in the token map, and a compressed id that the hash multipliers and
        # the n-gram tables were never trained with. We'd rather fail here than hand back a model whose new
        # tokens index past the end of the map.
        raise NotImplementedError(
            "DeepSeek-V4.1 does not support resizing the token embeddings, because its Engram layers hash "
            "n-grams of token ids through a token map that is fixed to the vocabulary of the checkpoint. "
            "If you need a new token, reuse one of the placeholder tokens that the tokenizer already has.\n"
            f"new_num_tokens: {new_num_tokens}"
        )


@auto_docstring(
    custom_intro="""
    The DeepSeek-V4.1 vision encoder: a ViT with 2D RoPE that runs full attention over the patches of one image.
    """
)
class DeepseekV41VisionModel(DeepseekV41PreTrainedModel):
    config: DeepseekV41VisionConfig
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    # Unlike the language model, the vision encoder has plain attention with no sink
    _supports_sdpa = True
    _can_record_outputs = {
        "hidden_states": DeepseekV41VisionLayer,
        "attentions": DeepseekV41VisionAttention,
    }

    def __init__(self, config: DeepseekV41VisionConfig):
        super().__init__(config)
        self.rope_dim = config.hidden_size // config.num_attention_heads // 2
        self.patch_embed = nn.Linear(config.num_channels * config.patch_size**2, config.hidden_size)
        self.layers = nn.ModuleList([DeepseekV41VisionLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def get_position_embeddings(self, height: int, width: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """2D RoPE: the first half of the frequencies encodes the row of a patch, and the second half its column."""
        exponents = torch.arange(0, self.rope_dim, 2, dtype=torch.float32, device=device) / self.rope_dim
        inv_freq = 1.0 / (self.config.rope_theta**exponents)
        rows = torch.arange(height, device=device).unsqueeze(1).expand(height, width)
        columns = torch.arange(width, device=device).unsqueeze(0).expand(height, width)
        freqs = torch.stack([rows, columns], dim=-1).reshape(-1, 2, 1).float() * inv_freq
        freqs = freqs.flatten(1)
        return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self, pixel_values: torch.Tensor, height: int, width: int, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutput:
        r"""
        pixel_values (`torch.Tensor` of shape `(height * width, num_channels * patch_size * patch_size)`):
            The flattened patches of one image, in row-major order.
        height (`int`):
            Number of rows in the image's patch grid.
        width (`int`):
            Number of columns in the image's patch grid.
        """
        hidden_states = self.patch_embed(pixel_values.flatten(1).to(self.patch_embed.weight.dtype))
        cos, sin = self.get_position_embeddings(height, width, hidden_states.device)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)
        return BaseModelOutput(last_hidden_state=self.norm(hidden_states))


@auto_docstring(
    custom_intro="""
    The DeepSeek-V4.1 language model, without the vision encoder and the language modeling head.
    """
)
class DeepseekV41TextModel(DeepseekV41PreTrainedModel):
    config: DeepseekV41TextConfig
    input_modalities = ("text",)

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepseekV41DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV41RotaryEmbedding(config)
        self.ngram_hasher = DeepseekV41NGramHasher(config) if config.ple_layer_ids else None
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        image_mask: torch.BoolTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        r"""
        image_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            `True` for the tokens of an image. They are routed with a separate bias and take no part in the
            Engram n-grams.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify at least one of input_ids or inputs_embeds")
        if self.ngram_hasher is not None and input_ids is None:
            raise ValueError(
                "DeepSeek-V4.1 hashes n-grams of token ids for its Engram layers, so it needs `input_ids` even "
                "when `inputs_embeds` is passed."
            )
        if use_cache and cache is None:
            cache = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_len, _ = inputs_embeds.shape
        past_seen = cache.get_seq_length() if cache is not None else 0
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)

        # First, the padding. Attention only needs to mask it, but the n-gram hashes and the pooling groups are
        # made of consecutive real tokens, so they need to know which tokens of this call are real, and how
        # many real tokens each row had seen before it. As everywhere else, the mask of a call that continues
        # from a cache covers the cached tokens too.
        is_real_token = torch.ones((batch_size, seq_len), dtype=torch.bool, device=inputs_embeds.device)
        num_seen_tokens = torch.full((batch_size,), past_seen, device=inputs_embeds.device)
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
            padding_mask = create_recurrent_attention_mask(self.config, inputs_embeds, attention_mask, cache)
            if padding_mask is not None:
                is_real_token = padding_mask.bool()
            if attention_mask.shape[1] == past_seen + seq_len:
                num_seen_tokens = attention_mask[:, :past_seen].bool().sum(dim=-1)

        # Next, the n-gram hashes of all the Engram layers. A padded token must not end up in the n-grams of
        # the tokens after it, and neither must an image token.
        ngram_hash_ids, ngram_mask = None, None
        if self.ngram_hasher is not None:
            hash_mask = is_real_token
            if image_mask is not None:
                ngram_mask = ~image_mask
                hash_mask = hash_mask & ngram_mask
            ngram_hash_ids = self.ngram_hasher(input_ids, hash_mask, cache.layers[0] if cache is not None else None)

        # `generate()` may pass a per-layer-type mask dict already built by `create_masks_for_generate`. All
        # layers use the same sliding-window mask, so we use the prebuilt one directly.
        if isinstance(attention_mask, dict):
            causal_mask = next(iter(attention_mask.values()))
        else:
            causal_mask = create_sliding_window_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache=cache,
                position_ids=position_ids,
                allow_is_causal_skip=False,
            )

        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        # The first attention reads the first stream as it is
        pre = torch.zeros((batch_size, seq_len, self.config.hc_mult), device=hidden_states.device)
        pre[..., 0] = 1.0
        position_embeddings = {
            "main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
        }
        # The compressed entries get RoPE as well. An entry stands for the first token of its group, so it
        # rotates at the position of that token. It only comes to exist with the last token of the group
        # though, and by then the first one may be a call behind us. So we go back from the last token: for
        # each token, this is the RoPE of the entry that it completes if it is the last one of a group of
        # `ratio`, and the compressor keeps the ones of the tokens that are.
        compressed_position_embeddings = {}
        for layer_idx, source in enumerate(self.config.kv_source_layers):
            ratio = self.config.compress_ratios[layer_idx]
            if source == layer_idx and ratio not in compressed_position_embeddings:
                compressed_position_embeddings[ratio] = self.rotary_emb(
                    inputs_embeds, position_ids=position_ids - (ratio - 1), layer_type="compress"
                )
        # What the attention layers share instead of recomputing, for this forward only. A layer that compresses
        # its own KV leaves them in this dict with their index keys, for the KV-sharing layers after it.
        shared_kv_states = KVSharingDict()
        # The selection of the last indexer and the candidate blocks go from layer to layer instead.
        # TODO: #51202 has the cache layer hand the selection out as top-k indices (`select_indices`), where we
        # pass boolean masks. We're watching for it to merge, to pass the selection the same way.
        selected_entries, candidate_entries = None, None

        for layer_idx, layer in enumerate(self.layers):
            hidden_states, pre, selected_entries, candidate_entries = layer(
                hidden_states,
                pre,
                ngram_hash_ids=(
                    ngram_hash_ids[:, :, self.config.ple_layer_ids.index(layer_idx)]
                    if layer_idx in self.config.ple_layer_ids
                    else None
                ),
                ngram_mask=ngram_mask,
                image_mask=image_mask,
                position_embeddings=position_embeddings,
                compressed_position_embeddings=compressed_position_embeddings,
                is_real_token=is_real_token,
                num_seen_tokens=num_seen_tokens,
                attention_mask=causal_mask,
                cache=cache,
                shared_kv_states=shared_kv_states,
                selected_entries=selected_entries,
                candidate_entries=candidate_entries,
                **kwargs,
            )

        # There is no hyper-connection head: the "pre" of the last MLP collapses the streams for the final norm
        hidden_states = (pre.unsqueeze(-1) * hidden_states.float()).sum(dim=2).to(hidden_states.dtype)
        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, cache=cache)


@auto_docstring(
    custom_intro="""
    The DeepSeek-V4.1 model, which consists of a vision encoder and a language model, without a language modeling
    head.
    """
)
class DeepseekV41Model(DeepseekV41PreTrainedModel):
    def __init__(self, config: DeepseekV41Config):
        super().__init__(config)
        self.vision_tower = DeepseekV41VisionModel(config.vision_config)
        self.multi_modal_projector = DeepseekV41MultiModalProjector(config)
        self.language_model = DeepseekV41TextModel(config.text_config)
        # Learned embeddings for the delimiters of an image: its start, its end, and the end of each row
        self.image_start = nn.Parameter(torch.empty(config.text_config.hidden_size))
        self.image_end = nn.Parameter(torch.empty(config.text_config.hidden_size))
        self.image_newline = nn.Parameter(torch.empty(config.text_config.hidden_size))
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
        Encodes the images and lays each one out as the language model reads it: a start embedding, then the rows
        of image tokens with a newline embedding after each row, then an end embedding. The `pooler_output` has
        one entry per image, of shape `(num_image_positions, hidden_size)`.
        """
    )
    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.LongTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.Tensor` of shape `(total_patches, num_channels * patch_size * patch_size)`):
            The flattened patches of all the images, one image after the other.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal size, height and width of each image's patch grid. The temporal size is always 1.
        """
        grids = image_grid_thw.tolist()
        # An image is a single frame. The temporal size is only there so that the input has the same layout as
        # the `image_grid_thw` of the models that take videos as well.
        if any(num_frames != 1 for num_frames, _, _ in grids):
            raise ValueError(
                "DeepSeek-V4.1 only takes images, so the temporal size of every grid must be 1.\n"
                f"`image_grid_thw`: {grids}"
            )
        patches_per_image = pixel_values.split([height * width for _, height, width in grids])
        vision_outputs, image_features = [], []
        for patches, (_, height, width) in zip(patches_per_image, grids):
            outputs = self.vision_tower(patches, height, width, return_dict=True, **kwargs)
            tokens = self.multi_modal_projector(outputs.last_hidden_state, height, width)
            # The projector rounds a partial square up, so the token grid is the patch grid divided by the ratio
            num_rows = -(-height // self.multi_modal_projector.downsample_ratio)
            rows = tokens.view(num_rows, -1, tokens.shape[-1])
            newlines = self.image_newline.to(tokens.dtype).expand(num_rows, 1, -1)
            rows = torch.cat([rows, newlines], dim=1).flatten(0, 1)
            image_start, image_end = self.image_start.to(tokens.dtype)[None], self.image_end.to(tokens.dtype)[None]
            image_features.append(torch.cat([image_start, rows, image_end]))
            vision_outputs.append(outputs)

        # The vision encoder ran one image at a time, so we put its outputs back in the layout of `pixel_values`:
        # the patches of all the images, one image after the other
        hidden_states, attentions = None, None
        if vision_outputs[0].hidden_states is not None:
            # `output_hidden_states` can select layers, which leaves the other ones as None
            per_layer = zip(*(outputs.hidden_states for outputs in vision_outputs))
            hidden_states = tuple(torch.cat(layer) if layer[0] is not None else None for layer in per_layer)
        if vision_outputs[0].attentions is not None:
            # A patch only attends to its own image, so the weights of a layer are one block per image
            per_layer = zip(*(outputs.attentions for outputs in vision_outputs))
            attentions = tuple(torch.stack([torch.block_diag(*head) for head in zip(*layer)]) for layer in per_layer)
        return BaseModelOutputWithPooling(
            last_hidden_state=torch.cat([outputs.last_hidden_state for outputs in vision_outputs]),
            pooler_output=tuple(image_features),
            hidden_states=hidden_states,
            attentions=attentions,
        )

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MultimodalMoeBaseModelOutputWithPast:
        r"""
        pixel_values (`torch.Tensor` of shape `(total_patches, num_channels * patch_size * patch_size)`, *optional*):
            The flattened patches of all the images in the batch, one image after the other.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal size, height and width of each image's patch grid. The temporal size is always 1.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify at least one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # The images come in either as `pixel_values`, which we encode here, or already encoded. `generate` does
        # the latter: it encodes them once before the first step, so that it can repeat whole images for beam
        # search. Repeating the rows of `pixel_values` would repeat patches instead.
        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if pixel_values is not None:
            if mm_encoder_outputs.get("image") is not None:
                raise ValueError(
                    "Got both `pixel_values` and images that are already encoded in `mm_encoder_outputs`. "
                    "Pass only one of the two."
                )
            mm_encoder_outputs["image"] = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)

        # Every position of an image carries `image_token_id`, the delimiters included
        image_mask = input_ids == self.config.image_token_id if input_ids is not None else None
        if mm_encoder_outputs.get("image") is not None:
            if image_mask is None:
                raise ValueError("`input_ids` are needed to find where the images go in the prompt.")
            features_per_image = mm_encoder_outputs["image"].pooler_output
            image_features = torch.cat(features_per_image).to(inputs_embeds.device, inputs_embeds.dtype)
            num_image_tokens = image_mask.sum()
            torch_compilable_check(
                num_image_tokens == image_features.shape[0],
                lambda: (
                    f"The prompt has {int(num_image_tokens)} image tokens, but the images take up "
                    f"{image_features.shape[0]} tokens (per image: {[len(features) for features in features_per_image]})."
                ),
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask.unsqueeze(-1), image_features)

        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            image_mask=image_mask,
            **kwargs,
        )
        return MultimodalMoeBaseModelOutputWithPast(**outputs, image_outputs=mm_encoder_outputs.get("image"))


@auto_docstring(
    custom_intro="""
    The DeepSeek-V4.1 model with a language modeling head on top. It takes text, optionally interleaved with
    images.
    """
)
class DeepseekV41ForCausalLM(DeepseekV41PreTrainedModel, GenerationMixin):
    def __init__(self, config: DeepseekV41Config):
        super().__init__(config)
        self.model = DeepseekV41Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MultimodalMoeCausalLMOutputWithPast:
        r"""
        pixel_values (`torch.Tensor` of shape `(total_patches, num_channels * patch_size * patch_size)`, *optional*):
            The flattened patches of all the images in the batch, one image after the other.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal size, height and width of each image's patch grid. The temporal size is always 1.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        outputs: MultimodalMoeBaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        # Only compute the necessary logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.text_config.vocab_size, **kwargs)

        return MultimodalMoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            image_outputs=outputs.image_outputs,
        )


def get_resized_image_size(
    height: int,
    width: int,
    patch_size: int,
    downsample_ratio: int,
    max_image_tokens: int,
    min_pixels: int,
    max_wh_ratio: float | None = None,
) -> tuple[int, int]:
    """
    Works out the size that an image is resized to before it is cut into patches. Both sides come out as a
    multiple of `patch_size`, small images are scaled up to about `min_pixels`, and large ones are scaled down
    until they fit in `max_image_tokens`.

    The model merges every `downsample_ratio x downsample_ratio` square of patches into one token. An image
    with `rows x columns` of those takes `rows * (columns + 1) + 2` positions in the prompt: a newline after
    each row, plus a start and an end delimiter.

    Args:
        height (`int`):
            Height of the image.
        width (`int`):
            Width of the image.
        patch_size (`int`):
            Side of the square patches that the vision encoder embeds.
        downsample_ratio (`int`):
            Side of the square of patches that the model merges into one token.
        max_image_tokens (`int`):
            The most positions the image may take in the prompt, delimiters included.
        min_pixels (`int`):
            Images with fewer pixels are scaled up to this area.
        max_wh_ratio (`float`, *optional*):
            Images wider than this width-to-height ratio are planned as if they had exactly this ratio.

    Returns:
        `tuple[int, int]`: The height and width to resize the image to.

    Example:

    ```python
    >>> from transformers.models.deepseek_v41.image_processing_deepseek_v41 import get_resized_image_size

    >>> limits = {"patch_size": 14, "downsample_ratio": 3, "max_image_tokens": 1024, "min_pixels": 295936}
    >>> get_resized_image_size(701, 1024, **limits)
    (714, 1036)
    >>> # A small image is scaled up
    >>> get_resized_image_size(200, 320, **limits)
    (434, 700)
    >>> # ... and one that would take more than 1024 positions is scaled down
    >>> get_resized_image_size(2000, 3000, **limits)
    (1050, 1568)

    ```
    """
    # First, an image that is too wide is planned at the widest ratio allowed. It is squashed to that ratio
    # afterwards, not cropped.
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    # Next, a small image is scaled up to `min_pixels`. The sides are truncated, so it can end up just below.
    if 0 < width * height < min_pixels:
        scale = (min_pixels / (width * height)) ** 0.5
        width, height = int(width * scale), int(height * scale)
    # Both sides are rounded up to whole patches, and a partial square of patches still makes a token
    resized_height = math.ceil(height / patch_size) * patch_size
    resized_width = math.ceil(width / patch_size) * patch_size
    rows = math.ceil(resized_height // patch_size / downsample_ratio)
    columns = math.ceil(resized_width // patch_size / downsample_ratio)
    if rows * (columns + 1) + 2 <= max_image_tokens:
        return resized_height, resized_width

    # Finally, an image that takes too many positions is scaled down. With `rows = aspect * columns`, the
    # budget `rows * (columns + 1) + 2 <= max_image_tokens` is a quadratic in `columns`, and this is its
    # positive root.
    aspect = height / width
    max_columns = math.sqrt((max_image_tokens - 2) / aspect + 0.25) - 0.5
    max_rows = max_columns * aspect
    token_size = patch_size * downsample_ratio
    if max_columns < 1.0:
        # Too tall for one full column at its own aspect ratio, so it becomes a single column of tokens
        return (max_image_tokens - 2) // 2 * token_size, token_size
    if max_rows < 1.0:
        # Too wide for one full row, so it becomes a single row of tokens
        return token_size, (max_image_tokens - 3) * token_size
    # Otherwise it is scaled to fit the largest whole grid of tokens, and keeps the whole patches that it covers
    scale = min(math.floor(max_columns) * token_size / width, math.floor(max_rows) * token_size / height)
    return math.floor(height * scale / patch_size) * patch_size, math.floor(width * scale / patch_size) * patch_size


class DeepseekV41ImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    patch_size (`int`, *optional*, defaults to 14):
        Side of the square patches that the vision encoder embeds.
    downsample_ratio (`int`, *optional*, defaults to 3):
        Side of the square of patches that the model merges into one token.
    max_image_tokens (`int`, *optional*, defaults to 1024):
        The most positions an image may take in the prompt, delimiters included. Larger images are scaled down.
    min_pixels (`int`, *optional*, defaults to 295936):
        Images with fewer pixels are scaled up to this area.
    max_wh_ratio (`float`, *optional*):
        Images wider than this width-to-height ratio are squashed to it. The released checkpoints set no limit.
    """

    patch_size: int
    downsample_ratio: int
    max_image_tokens: int
    min_pixels: int
    max_wh_ratio: float | None


@auto_docstring(
    custom_intro="""
    Constructs a DeepSeek-V4.1 image processor. Each image keeps its own resolution and aspect ratio: it is
    resized to a whole number of patches, padded with gray where needed, and cut into flattened patches.
    """
)
class DeepseekV41ImageProcessor(TorchvisionBackend):
    do_convert_rgb = True
    do_resize = True
    resample = PILImageResampling.BICUBIC
    do_rescale = True
    do_normalize = True
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    patch_size = 14
    downsample_ratio = 3
    max_image_tokens = 1024
    min_pixels = 295936
    max_wh_ratio = None
    valid_kwargs = DeepseekV41ImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[DeepseekV41ImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[DeepseekV41ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _validate_preprocess_kwargs(self, **kwargs):
        # The output size comes from `min_pixels` and `max_image_tokens`, so there is no `size` for the base
        # class to check
        kwargs.pop("do_resize", None)
        super()._validate_preprocess_kwargs(**kwargs)

    def process_image(
        self,
        image: ImageInput,
        do_convert_rgb: bool | None = None,
        input_data_format: str | ChannelDimension | None = None,
        device: "torch.device | None" = None,
        **kwargs: Unpack[ImagesKwargs],
    ) -> "torch.Tensor":
        # The original preprocessing converts with a plain `convert("RGB")`, which drops an alpha channel. The
        # base class would composite it onto a white background instead.
        if do_convert_rgb and get_image_type(image) == ImageType.PIL:
            image = tvF.pil_to_tensor(image.convert("RGB"))
            input_data_format = ChannelDimension.FIRST
        return super().process_image(
            image, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device, **kwargs
        )

    def resize_and_pad(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        fill_value: int = 127,
    ) -> "torch.Tensor":
        """
        Resizes an image to fit inside `size` without changing its aspect ratio, and centers it on a canvas of
        that size. This is what `PIL.ImageOps.pad` does, which the original preprocessing uses with a gray
        canvas.

        Args:
            image (`torch.Tensor` of shape `(num_channels, height, width)`):
                The image to resize.
            size (`SizeDict`):
                The height and width of the canvas.
            resample (`PILImageResampling` or `InterpolationMode`):
                The filter to resize with.
            fill_value (`int`, *optional*, defaults to 127):
                The value of the canvas around the image, on the 0-255 scale.

        Returns:
            `torch.Tensor` of shape `(num_channels, size.height, size.width)`: The image on its canvas.
        """
        height, width = image.shape[-2:]
        content_height, content_width = size.height, size.width
        if width / height > size.width / size.height:
            # Relatively wider than the canvas: it fills the width, with a bar above and below
            content_height = round(height / width * size.width)
        elif width / height < size.width / size.height:
            # Relatively taller than the canvas: it fills the height, with a bar on the left and right
            content_width = round(width / height * size.height)
        image = self.resize(image, SizeDict(height=content_height, width=content_width), resample)
        if (content_height, content_width) == (size.height, size.width):
            return image

        canvas = image.new_full((*image.shape[:-2], size.height, size.width), fill_value)
        top = round((size.height - content_height) * 0.5)
        left = round((size.width - content_width) * 0.5)
        canvas[..., top : top + content_height, left : left + content_width] = image
        return canvas

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        downsample_ratio: int,
        max_image_tokens: int,
        min_pixels: int,
        max_wh_ratio: float | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        patches, image_grid_thw = [], []
        for image in images:
            channels, height, width = image.shape
            if do_resize:
                resized_height, resized_width = get_resized_image_size(
                    height, width, patch_size, downsample_ratio, max_image_tokens, min_pixels, max_wh_ratio
                )
                size = SizeDict(height=resized_height, width=resized_width)
                if max_wh_ratio is not None and width >= max_wh_ratio * height:
                    # Too wide: squashed to the size that was planned at the widest ratio allowed
                    image = self.resize(image, size, resample)
                else:
                    # Everything else keeps its aspect ratio, with gray bars to fill up the last patches
                    image = self.resize_and_pad(image, size, resample)
                height, width = resized_height, resized_width
            if height % patch_size != 0 or width % patch_size != 0:
                raise ValueError(
                    "With `do_resize=False`, the height and width of an image must be multiples of the patch "
                    f"size.\nImage size: {(height, width)}\nPatch size: {patch_size}"
                )
            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)
            # One row per patch in reading order, each flattened as (channels, patch height, patch width)
            rows, columns = height // patch_size, width // patch_size
            image = image.reshape(channels, rows, patch_size, columns, patch_size).permute(1, 3, 0, 2, 4)
            patches.append(image.reshape(rows * columns, -1))
            # The grid has a temporal size as well, like in the models that take videos. An image is one frame.
            image_grid_thw.append([1, rows, columns])
        return BatchFeature(
            data={"pixel_values": torch.cat(patches), "image_grid_thw": torch.tensor(image_grid_thw)},
            tensor_type=return_tensors,
        )


@auto_docstring
class DeepseekV41Processor(ProcessorMixin):
    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, image_token="<｜deepseek_image｜>"):
        r"""
        image_token (`str`, *optional*, defaults to `"<｜deepseek_image｜>"`):
            The placeholder for an image in the prompt. The processor repeats it once for every position that
            the image takes up.
        """
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        # The model merges every `downsample_ratio x downsample_ratio` square of patches into a token, and a
        # partial square at the edge still makes one
        downsample_ratio = kwargs.get("downsample_ratio") or self.image_processor.downsample_ratio
        _, height, width = (int(size) for size in image_inputs["image_grid_thw"][image_idx])
        rows, columns = math.ceil(height / downsample_ratio), math.ceil(width / downsample_ratio)
        # Every position of the image carries the image token: its start, each row with a newline after it,
        # and its end. The model tells them apart by where they are in that layout.
        return self.image_token * (rows * (columns + 1) + 2)


__all__ = [
    "DeepseekV41Config",
    "DeepseekV41TextConfig",
    "DeepseekV41VisionConfig",
    "DeepseekV41PreTrainedModel",
    "DeepseekV41VisionModel",
    "DeepseekV41TextModel",
    "DeepseekV41Model",
    "DeepseekV41ForCausalLM",
    "DeepseekV41ImageProcessor",
    "DeepseekV41Processor",
]
