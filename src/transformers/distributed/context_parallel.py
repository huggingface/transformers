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
"""Context parallelism: each rank of the `cp` mesh holds a contiguous slice of every sequence.

A context parallel plan pairs two kinds of styles:

- `"cp_split_sequence"` on the decoder layers, which run on this rank's slice. The model built the attention mask and
  the position embeddings for the whole sequence before the first layer.
- An attention style, `"cp_ulysses"`, on their attention modules, whose attention function then runs over the whole
  sequence. [`UlyssesAttention`] all-to-alls the query, key and value, so that every rank holds the whole sequence for
  `num_heads / cp_size` heads, runs the model's own attention function with the model's own mask, and all-to-alls the
  output back. Another algorithm is a [`ContextParallelAttention`] subclass registered in `ALL_PARALLEL_STYLES`.

`apply_context_parallelism` installs such a plan on a model, and [`ModelContextParallel`] on the model itself: the
model checks, a forward that refuses a cache, and a `loss_function` that reports the loss of the whole sequences on
every rank. The styles share [`ContextParallelLayer`], which has the `install_forward` contract of the tensor and
expert parallel styles.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial, wraps
from typing import TYPE_CHECKING

from ..loss.loss_utils import LOSS_MAPPING, ForCausalLMLoss
from ..utils.import_utils import is_torch_available, is_torch_distributed_available
from .parallel_utils import ALL_PARALLEL_STYLES


if TYPE_CHECKING:
    from torch import nn
    from torch.distributed.device_mesh import DeviceMesh

if is_torch_available():
    import torch

if is_torch_distributed_available():
    import torch.distributed as dist
    from torch.distributed._functional_collectives import all_to_all_single_autograd


def split_sequence(tensor: torch.Tensor, dim: int, cp_mesh: DeviceMesh) -> torch.Tensor:
    """This rank's contiguous slice of `tensor` along its sequence dimension `dim`."""
    cp_size = cp_mesh.size()
    if tensor.shape[dim] % cp_size:
        raise ValueError(
            f"Context parallelism splits every sequence across the {cp_size} ranks of the `cp` mesh, so the sequence "
            f"length must be divisible by `cp_size`, got {tensor.shape[dim]}. Pad the sequences to a multiple of "
            "`cp_size`."
        )
    return tensor.chunk(cp_size, dim=dim)[cp_mesh.get_local_rank()].contiguous()


def wrap_attention_function(attn_implementation: str) -> None:
    """
    Wrap the `ALL_ATTENTION_FUNCTIONS` entry of `attn_implementation`, so that the modules an attention style gave a
    `cp_attention` run it through it. A no-op if the entry is wrapped already: call it again after registering another
    function under that key, as loading a hub kernel does.
    """
    from ..modeling_utils import ALL_ATTENTION_FUNCTIONS

    attention_function = ALL_ATTENTION_FUNCTIONS[attn_implementation]
    if getattr(attention_function, "is_context_parallel", False):
        return

    def context_parallel_attention(module, *args, **kwargs):
        if hasattr(module, "cp_attention"):
            return module.cp_attention(attention_function, module, *args, **kwargs)
        return attention_function(module, *args, **kwargs)

    context_parallel_attention.is_context_parallel = True
    ALL_ATTENTION_FUNCTIONS.register(attn_implementation, context_parallel_attention)


class ContextParallelLayer:
    """
    Base class of the context parallel styles, with the `install_forward` contract of `TensorParallelLayer`: replace
    `module.forward` by one that runs `transform_inputs_pre_forward` first. The styles shard no parameter, so the
    parameter hooks of the other parallelisms have no counterpart here.
    """

    def transform_inputs_pre_forward(self, module: nn.Module, args: tuple, kwargs: dict, cp_mesh: DeviceMesh):
        return args, kwargs

    def install_forward(self, module: nn.Module, cp_mesh: DeviceMesh, **options):
        """Replace `module.forward`; the keyword `options` go to `transform_inputs_pre_forward`."""
        original_forward = module.forward

        @wraps(original_forward)
        def cp_forward(*args, **kwargs):
            args, kwargs = self.transform_inputs_pre_forward(module, args, kwargs, cp_mesh, **options)
            return original_forward(*args, **kwargs)

        module.forward = cp_forward
        return module


class SplitSequenceParallel(ContextParallelLayer):
    """
    Decoder layer: run on this rank's slice of the sequence. The first layer of the plan cuts `hidden_states`, and every
    layer cuts the `position_embeddings` the model computed for the whole sequence. The attention mask and
    `position_ids` stay those of the whole sequence, which the attention style gives the attention function.
    """

    def transform_inputs_pre_forward(
        self, module: nn.Module, args: tuple, kwargs: dict, cp_mesh: DeviceMesh, *, split_hidden_states: bool = False
    ):
        if split_hidden_states:
            if args:
                args = (split_sequence(args[0], 1, cp_mesh), *args[1:])
            else:
                kwargs["hidden_states"] = split_sequence(kwargs["hidden_states"], 1, cp_mesh)
        position_embeddings = kwargs.get("position_embeddings")
        if isinstance(position_embeddings, tuple):
            kwargs["position_embeddings"] = tuple(split_sequence(x, -2, cp_mesh) for x in position_embeddings)
        elif position_embeddings is not None:
            kwargs["position_embeddings"] = split_sequence(position_embeddings, -2, cp_mesh)
        return args, kwargs


class ContextParallelAttention(ContextParallelLayer):
    """
    Attention module: run its attention function over the whole sequence, when each rank of `cp_mesh` holds a slice
    of it. `install_forward` checks the module, gives it `attention` as its `cp_attention`, and wraps the model's entry
    in `ALL_ATTENTION_FUNCTIONS` so that the attention call of that module runs through it; the forward stays as is.
    A subclass is an algorithm, registered in `ALL_PARALLEL_STYLES` and named in the plan like any style.
    """

    def check(self, module: nn.Module, cp_mesh: DeviceMesh) -> None:
        """Raise a `ValueError` if this attention cannot run on `module` (or on a whole model)."""

    def attention(self, cp_mesh: DeviceMesh, attention_function, module, query, key, value, attention_mask, **kwargs):
        raise NotImplementedError

    def install_forward(self, module: nn.Module, cp_mesh: DeviceMesh):
        self.check(module, cp_mesh)
        module.cp_attention = partial(self.attention, cp_mesh)
        wrap_attention_function(module.config._attn_implementation)
        return module


class UlyssesAttention(ContextParallelAttention):
    """
    All-to-all to the whole sequence with `1 / cp_size` of the heads this rank holds, run the attention, all-to-all
    back. Under tensor parallelism this rank holds `num_heads / tp_size` heads.
    """

    def check(self, module: nn.Module, cp_mesh: DeviceMesh) -> None:
        from ..modeling_utils import ALL_ATTENTION_FUNCTIONS

        attn_implementation = module.config._attn_implementation
        if attn_implementation not in ALL_ATTENTION_FUNCTIONS or attn_implementation == "flex_attention":
            raise ValueError(
                f"Ulysses context parallelism routes the attention calls of {module.__class__.__name__} through "
                f"`ALL_ATTENTION_FUNCTIONS`, but it uses `attn_implementation={attn_implementation}`, which takes no "
                "attention mask of the whole sequence. Load the model with `sdpa`, `flash_attention_2`, "
                "`flash_attention_3` or a hub flash kernel, or disable context parallelism."
            )

        config = module.config.get_text_config()
        num_kv_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
        cp_size = cp_mesh.size()
        distributed_config = getattr(module.config, "distributed_config", None)
        tp_size = 1 if distributed_config is None else distributed_config.tp_size
        if config.num_attention_heads % (cp_size * tp_size) or num_kv_heads % (cp_size * tp_size):
            raise ValueError(
                f"Ulysses context parallelism splits the attention heads across the {cp_size} ranks of the `cp` mesh "
                f"and the {tp_size} ranks of the `tp` mesh, so `cp_size * tp_size` must divide both the number of "
                f"attention heads ({config.num_attention_heads}) and the number of key/value heads ({num_kv_heads}) "
                f"of {module.__class__.__name__}. Use smaller `cp_size` or `tp_size`, or disable context parallelism."
            )

    def attention(self, cp_mesh: DeviceMesh, attention_function, module, query, key, value, attention_mask, **kwargs):
        cp_size, group = cp_mesh.size(), cp_mesh.get_group()

        def _all_to_all(tensor):
            """Split dim 1 across the ranks of the `cp` mesh, concatenate what they send back along dim 2."""
            chunks = torch.stack(tensor.chunk(cp_size, dim=1))
            return torch.cat(all_to_all_single_autograd(chunks, None, None, group).unbind(), dim=2)

        # [batch, heads, seq / cp_size, head_dim] -> [batch, heads / cp_size, seq, head_dim]
        query, key, value = (_all_to_all(x) for x in (query, key, value))
        if kwargs.get("s_aux") is not None:  # attention sinks: one logit per head
            kwargs["s_aux"] = kwargs["s_aux"].chunk(cp_size)[cp_mesh.get_local_rank()]
        output, attn_weights = attention_function(module, query, key, value, attention_mask, **kwargs)
        # [batch, seq, heads / cp_size, head_dim] -> [batch, seq / cp_size, heads, head_dim]
        return _all_to_all(output), attn_weights


if is_torch_distributed_available():
    ALL_PARALLEL_STYLES.register("cp_split_sequence", SplitSequenceParallel())
    ALL_PARALLEL_STYLES.register("cp_ulysses", UlyssesAttention())


def context_parallel_styles() -> list[str]:
    """The names of the context parallel styles in `ALL_PARALLEL_STYLES`, the registered attention algorithms included."""
    return [name for name in ALL_PARALLEL_STYLES if isinstance(ALL_PARALLEL_STYLES[name], ContextParallelLayer)]


# Layer types that only mix tokens through softmax attention over the whole sequence. Every other one (recurrent,
# convolutional, sparse-indexer, ...) mixes tokens outside attention and would silently be cut at the slice boundaries.
SOFTMAX_ATTENTION_LAYER_TYPES = ("full_attention", "sliding_attention", "chunked_attention")


class ModelContextParallel(ContextParallelLayer):
    """
    The model itself, outside the plan since no plan key names the root module: `check` refuses what context
    parallelism would silently get wrong, the forward refuses a cache, which context parallelism does not split, and
    `loss_function` reports the loss of the whole sequences on every rank.
    """

    def check(self, model: nn.Module) -> None:
        """Raise a `ValueError` if context parallelism would silently change the logits or the loss of `model`."""
        if not model._supports_attention_backend:
            raise ValueError(
                f"Context parallelism routes attention through `ALL_ATTENTION_FUNCTIONS`, but {model.__class__.__name__} "
                "does not declare that all its attention layers go through it with the model's kwargs "
                "(`_supports_attention_backend`), so they could run on this rank's slice only. Use another model, or "
                "disable context parallelism."
            )

        layer_types = set(getattr(model.config.get_text_config(), "layer_types", None) or ())
        if unsupported := layer_types - set(SOFTMAX_ATTENTION_LAYER_TYPES):
            raise ValueError(
                f"Context parallelism does not support the {sorted(unsupported)} layers of {model.__class__.__name__}: "
                "they mix tokens outside softmax attention, so they would run on this rank's slice only and be silently "
                "cut at the slice boundaries. Use another model, or disable context parallelism."
            )

        # `loss_type=None` (base models, and heads whose name maps to no loss) falls back to the causal LM loss
        if model.loss_type is not None and LOSS_MAPPING.get(model.loss_type) is not ForCausalLMLoss:
            raise ValueError(
                f"Context parallelism only supports the causal language modeling loss, but {model.__class__.__name__} "
                f"uses `loss_type={model.loss_type}`. Its loss would be computed from this rank's slice of the shifted "
                "labels while the logits stay sharded. Use a `*ForCausalLM` model, or disable context parallelism."
            )

    def transform_inputs_pre_forward(self, module: nn.Module, args: tuple, kwargs: dict, cp_mesh: DeviceMesh):
        if kwargs.get("past_key_values") is not None:
            raise ValueError(
                "Context parallelism only supports training: `past_key_values` are not allowed. Run the model on full "
                "sequences without a cache, or disable context parallelism."
            )
        kwargs["use_cache"] = False
        # a hub kernel loaded since may have registered another function under the attention key
        wrap_attention_function(module.config._attn_implementation)
        return args, kwargs

    def loss(
        self,
        loss_function: Callable,
        cp_mesh: DeviceMesh,
        logits: torch.Tensor,
        labels: torch.Tensor,
        vocab_size: int,
        num_items_in_batch: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        The loss of the whole sequences, which every rank reports, from this rank's slice of the logits and its slice of
        the shifted labels. Its gradient stays this rank's, multiplied by `cp_size`, so that FSDP2's average over the cp
        ranks gives the gradient of the full-sequence loss.
        """
        shift_labels = kwargs.pop("shift_labels", None)
        if shift_labels is None:
            shift_labels = torch.nn.functional.pad(labels[..., 1:], (0, 1), value=-100)
        if num_items_in_batch is None:
            num_items_in_batch = (shift_labels != -100).sum()
        kwargs["shift_labels"] = split_sequence(shift_labels, 1, cp_mesh)
        loss = loss_function(logits, labels, vocab_size, num_items_in_batch=num_items_in_batch, **kwargs)

        total = loss.detach().clone()
        dist.all_reduce(total, group=cp_mesh.get_group())
        return total + (loss - loss.detach()) * cp_mesh.size()

    def install_forward(self, model: nn.Module, cp_mesh: DeviceMesh):
        self.check(model)
        model.loss_function = partial(self.loss, model.loss_function, cp_mesh)
        return super().install_forward(model, cp_mesh)


def apply_context_parallelism(model: nn.Module, cp_mesh: DeviceMesh, cp_plan: dict[str, str]) -> nn.Module:
    """
    Let the ranks of `cp_mesh` split every sequence: `ModelContextParallel` on the model, then the plan, whose
    `"cp_split_sequence"` decoder layers run on this rank's slice and whose attention modules run their attention over
    the whole sequence.
    """
    ModelContextParallel().install_forward(model, cp_mesh)
    modules = dict(model.named_modules())
    # the first decoder layer still gets the whole sequence: it is the one that cuts it
    first_layer = next(name for name, style in cp_plan.items() if style == "cp_split_sequence")
    for name, style in cp_plan.items():
        options = {"split_hidden_states": name == first_layer} if style == "cp_split_sequence" else {}
        ALL_PARALLEL_STYLES[style].install_forward(modules[name], cp_mesh, **options)
    return model
