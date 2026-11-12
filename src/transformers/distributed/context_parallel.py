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
- An attention style, `"cp_ulysses"`, on their attention modules, whose attention function runs over the whole
  sequence while the module runs. [`UlyssesAttention`] all-to-alls the query, key and value, so that every rank holds the whole sequence for
  `num_heads / cp_size` heads, runs the model's own attention function with the model's own mask, and all-to-alls the
  output back. Another algorithm is a [`ContextParallelAttention`] subclass added to `ALL_CP_STYLES`.

`apply_context_parallelism` checks the model, then installs such a plan on it, [`ModelContextParallel`] on the model
itself (a forward that refuses a cache and pads the sequences to a multiple of `cp_size`), and `context_parallel_loss`
as its `loss_function`, which reports the loss of the whole sequences on every rank. The styles share [`ContextParallelLayer`], which has the `install_forward` contract of the tensor and
expert parallel styles.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from functools import partial, wraps
from typing import TYPE_CHECKING

from ..loss.loss_utils import LOSS_MAPPING, ForCausalLMLoss
from ..utils.import_utils import is_torch_available, is_torch_distributed_available


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


class ContextParallelLayer:
    """
    Base class of the context parallel styles, with the hooks of `TensorParallelLayer`: `install_forward` replaces
    `module.forward` by one that runs `transform_inputs_pre_forward` first, then the original forward inside
    `context_around_forward`. The styles shard no parameter, so the parameter hooks of the other parallelisms have no
    counterpart here.
    """

    def transform_inputs_pre_forward(self, module: nn.Module, args: tuple, kwargs: dict, cp_mesh: DeviceMesh):
        return args, kwargs

    def context_around_forward(self, module: nn.Module, cp_mesh: DeviceMesh):
        return contextlib.nullcontext()

    def install_forward(self, module: nn.Module, cp_mesh: DeviceMesh, **options):
        """Replace `module.forward`; the keyword `options` go to `transform_inputs_pre_forward`."""
        original_forward = module.forward

        @wraps(original_forward)
        def cp_forward(*args, **kwargs):
            args, kwargs = self.transform_inputs_pre_forward(module, args, kwargs, cp_mesh, **options)
            with self.context_around_forward(module, cp_mesh):
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
    of it. While the module runs, its entry in `ALL_ATTENTION_FUNCTIONS` is `attention` around the registered
    function, and it is restored after; the module's forward itself stays as is. A subclass is an algorithm, added
    to `ALL_CP_STYLES` and named in the plan like any style.
    """

    def check(self, module: nn.Module, cp_mesh: DeviceMesh) -> None:
        """Raise a `ValueError` if this attention cannot run on `module` (or on a whole model)."""

    def attention(self, cp_mesh: DeviceMesh, attention_function, module, query, key, value, attention_mask, **kwargs):
        raise NotImplementedError

    @contextlib.contextmanager
    def context_around_forward(self, module: nn.Module, cp_mesh: DeviceMesh):
        from ..modeling_utils import ALL_ATTENTION_FUNCTIONS

        attn_implementation = module.config._attn_implementation
        # the function registered now, so that a hub kernel loaded since the model was built is the one run
        attention_function = ALL_ATTENTION_FUNCTIONS[attn_implementation]
        ALL_ATTENTION_FUNCTIONS.register(attn_implementation, partial(self.attention, cp_mesh, attention_function))
        try:
            yield
        finally:
            ALL_ATTENTION_FUNCTIONS.register(attn_implementation, attention_function)

    def install_forward(self, module: nn.Module, cp_mesh: DeviceMesh):
        self.check(module, cp_mesh)
        return super().install_forward(module, cp_mesh)


class UlyssesAttention(ContextParallelAttention):
    """All-to-all to the whole sequence with `num_heads / cp_size` heads, run the attention, all-to-all back."""

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
        if config.num_attention_heads % cp_size or num_kv_heads % cp_size:
            raise ValueError(
                f"Ulysses context parallelism splits the attention heads across the {cp_size} ranks of the `cp` mesh, "
                f"so `cp_size` must divide both the number of attention heads ({config.num_attention_heads}) and the "
                f"number of key/value heads ({num_kv_heads}) of {module.__class__.__name__}. Use another `cp_size`, or "
                "disable context parallelism."
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


ALL_CP_STYLES: dict[str, ContextParallelLayer] = {
    "cp_split_sequence": SplitSequenceParallel(),
    "cp_ulysses": UlyssesAttention(),
}


def pad_sequence(tensor: torch.Tensor, pad_len: int, value) -> torch.Tensor:
    """`tensor` with `pad_len` copies of `value` appended along its sequence dimension 1."""
    padding = tensor.new_full((tensor.shape[0], pad_len, *tensor.shape[2:]), value)
    return torch.cat([tensor, padding], dim=1)


class ModelContextParallel(ContextParallelLayer):
    """
    The model itself, outside the plan since no plan key names the root module: its forward refuses a cache, which
    context parallelism does not split, and pads every sequence at the end to a multiple of `cp_size`, before the
    model builds its masks from them.
    """

    def transform_inputs_pre_forward(self, module: nn.Module, args: tuple, kwargs: dict, cp_mesh: DeviceMesh):
        if kwargs.get("past_key_values") is not None:
            raise ValueError(
                "Context parallelism only supports training: `past_key_values` are not allowed. Run the model on full "
                "sequences without a cache, or disable context parallelism."
            )
        kwargs["use_cache"] = False

        tokens = args[0] if args else kwargs.get("input_ids")
        if tokens is None:
            tokens = kwargs["inputs_embeds"]
        pad_len = -tokens.shape[1] % cp_mesh.size()
        if pad_len == 0:
            return args, kwargs

        pad_token_id = getattr(module.config, "pad_token_id", None) or 0
        if args:
            args = (pad_sequence(args[0], pad_len, pad_token_id),) + args[1:]
        padding_values = {
            "input_ids": pad_token_id,
            "inputs_embeds": 0,
            "attention_mask": 0,
            "labels": -100,
            "shift_labels": -100,
        }
        for name, value in padding_values.items():
            if kwargs.get(name) is not None:
                kwargs[name] = pad_sequence(kwargs[name], pad_len, value)
        if kwargs.get("position_ids") is not None:
            # the padding continues the positions of the last document of every row
            position_ids = kwargs["position_ids"]
            steps = torch.arange(1, pad_len + 1, device=position_ids.device)
            kwargs["position_ids"] = torch.cat([position_ids, position_ids[..., -1:] + steps], dim=-1)
        return args, kwargs


def context_parallel_loss(
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


def apply_context_parallelism(model: nn.Module, cp_mesh: DeviceMesh, cp_plan: dict[str, str]) -> nn.Module:
    """
    Let the ranks of `cp_mesh` split every sequence: `ModelContextParallel` on the model and `context_parallel_loss`
    as its loss, then the plan, whose `"cp_split_sequence"` decoder layers run on this rank's slice and whose attention
    modules run their attention over the whole sequence. The model's own forward refuses a cache and pads the sequences
    to a multiple of `cp_size`. Raises a `ValueError` if context parallelism would silently change the logits or the
    loss of `model`.
    """
    if not model._supports_attention_backend:
        raise ValueError(
            f"Context parallelism routes attention through `ALL_ATTENTION_FUNCTIONS`, but {model.__class__.__name__} "
            "does not declare that all its attention layers go through it with the model's kwargs "
            "(`_supports_attention_backend`), so they could run on this rank's slice only. Use another model, or "
            "disable context parallelism."
        )

    # every other layer type (recurrent, convolutional, sparse-indexer, ...) mixes tokens outside attention
    layer_types = set(getattr(model.config.get_text_config(), "layer_types", None) or ())
    if unsupported := layer_types - {"full_attention", "sliding_attention", "chunked_attention"}:
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

    ModelContextParallel().install_forward(model, cp_mesh)
    model.loss_function = partial(context_parallel_loss, model.loss_function, cp_mesh)
    modules = dict(model.named_modules())
    # the first decoder layer still gets the whole sequence: it is the one that cuts it
    first_layer = next(name for name, style in cp_plan.items() if style == "cp_split_sequence")
    for name, style in cp_plan.items():
        options = {"split_hidden_states": name == first_layer} if style == "cp_split_sequence" else {}
        ALL_CP_STYLES[style].install_forward(modules[name], cp_mesh, **options)
    return model
