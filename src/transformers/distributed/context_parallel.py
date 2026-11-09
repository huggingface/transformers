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
"""Context parallelism: each rank of a `cp` device mesh holds a contiguous slice of the sequence.

Attention still has to see the whole sequence. [`UlyssesCP`] gets there with two all-to-all exchanges around every
attention call: the first gathers the sequence and scatters the heads, so the model's own attention function runs on
the full sequence with the heads this rank holds divided by `cp_size`, and the second reverses it. Under tensor
parallelism this rank holds `num_heads / tp_size` of them.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

from ..loss.loss_utils import LOSS_MAPPING, ForCausalLMLoss
from ..masking_utils import find_packed_sequence_indices
from ..utils.import_utils import is_torch_available, is_torch_distributed_available


if TYPE_CHECKING:
    from torch import nn
    from torch.distributed.device_mesh import DeviceMesh

if is_torch_available():
    import torch

if is_torch_distributed_available():
    import torch.distributed as dist


class CPImplementation:
    """How attention sees the whole sequence when each rank of the `cp` mesh holds a contiguous slice of it."""

    def __init__(self, cp_mesh: DeviceMesh):
        self.cp_mesh = cp_mesh
        # positions of the full sequence for the forward being run, set by the model's forward pre-hook
        self.position_ids: torch.Tensor | None = None

    def check(self, model: nn.Module) -> None:
        """Raise a `ValueError` if this implementation cannot compute the attention of `model`."""

    def attention(
        self,
        attention_function: Callable,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run one attention call of `module` over the whole sequence, and return the output and attention weights."""
        raise NotImplementedError


def _all_to_all(tensor: torch.Tensor, scatter_dim: int, gather_dim: int, group) -> torch.Tensor:
    """Split `tensor` along `scatter_dim` across the ranks of `group`, concatenate what comes back along `gather_dim`."""
    group_size = dist.get_world_size(group)
    if tensor.shape[scatter_dim] % group_size != 0:
        raise ValueError(
            f"Cannot split a tensor of shape {tuple(tensor.shape)} along dim {scatter_dim} across {group_size} "
            "context parallel ranks: the size of that dim must be divisible by `cp_size`."
        )
    # Bring the chunk index in front: `all_to_all_single` sends chunk `i` of dim 0 to rank `i` and receives into slot
    # `j` of dim 0 what rank `j` sent, so afterwards dim 0 indexes the pieces of the gathered dim, in rank order.
    shape = list(tensor.shape)
    shape[scatter_dim : scatter_dim + 1] = [group_size, shape[scatter_dim] // group_size]
    tensor = tensor.reshape(shape).movedim(scatter_dim, 0).contiguous()
    output = torch.empty_like(tensor)
    dist.all_to_all_single(output, tensor, group=group)
    output = output.movedim(0, gather_dim)
    shape = list(output.shape)
    shape[gather_dim : gather_dim + 2] = [shape[gather_dim] * shape[gather_dim + 1]]
    return output.reshape(shape)


class _AllToAll(torch.autograd.Function):
    """Differentiable all-to-all: the backward is the same collective with the scatter and gather dims swapped."""

    @staticmethod
    def forward(ctx, group, tensor, scatter_dim, gather_dim):
        ctx.group, ctx.scatter_dim, ctx.gather_dim = group, scatter_dim, gather_dim
        return _all_to_all(tensor, scatter_dim, gather_dim, group)

    @staticmethod
    def backward(ctx, grad_output):
        return None, _all_to_all(grad_output, ctx.gather_dim, ctx.scatter_dim, ctx.group), None, None


def _gather_sequence_scatter_heads(tensor: torch.Tensor, group) -> torch.Tensor:
    """`[batch, heads, local_seq, head_dim]` -> `[batch, heads / cp_size, seq, head_dim]`."""
    return _AllToAll.apply(group, tensor, 1, 2)


def _gather_heads_scatter_sequence(tensor: torch.Tensor, group) -> torch.Tensor:
    """`[batch, seq, heads / cp_size, head_dim]` -> `[batch, local_seq, heads, head_dim]`."""
    return _AllToAll.apply(group, tensor, 1, 2)


def _packed_causal_mask(position_ids: torch.Tensor) -> torch.Tensor | None:
    """
    Boolean `[batch, 1, seq, seq]` mask that is causal within each packed document and blocks attention across
    documents. Returns `None` when nothing is packed, so the caller can rely on `is_causal` instead.
    """
    sequence_ids = find_packed_sequence_indices(position_ids)
    if sequence_ids is None:
        return None
    seq_len = position_ids.shape[1]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=position_ids.device).tril()
    return causal & (sequence_ids[:, None, :, None] == sequence_ids[:, None, None, :])


class UlyssesCP(CPImplementation):
    """
    Attention over the whole sequence with `num_heads / (cp_size * tp_size)` heads per rank: two all-to-all exchanges
    around the model's own attention function.
    """

    def check(self, model: nn.Module) -> None:
        config = model.config
        attn_implementation = config._attn_implementation
        if (
            attn_implementation in (None, "eager")
            or "flex_attention" in attn_implementation
            or attn_implementation.startswith("paged|")
        ):
            raise ValueError(
                "Ulysses context parallelism needs an attention implementation that takes the `position_ids` of the "
                "full sequence instead of a materialized attention mask, but "
                f"{model.__class__.__name__} uses `{attn_implementation}`. Load the model with `sdpa`, "
                "`flash_attention_2`, `flash_attention_3` or a hub flash kernel, or disable context parallelism."
            )

        text_config = config.get_text_config()
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is not None and "chunked_attention" in layer_types:
            raise ValueError(
                f"Ulysses context parallelism does not support `chunked_attention` layers of {model.__class__.__name__}: "
                "their mask cannot be rebuilt for the full sequence. Use another model, or disable context "
                "parallelism."
            )
        # Models that predate `layer_types` (Mistral, for one) apply a sliding window to every layer whenever
        # `sliding_window` is set.
        has_sliding_window = "sliding_attention" in (layer_types or []) or (
            layer_types is None and getattr(text_config, "sliding_window", None) is not None
        )
        if has_sliding_window and attn_implementation == "sdpa":
            raise ValueError(
                f"Ulysses context parallelism with `sdpa` does not support the sliding-window attention of "
                f"{model.__class__.__name__}: the local mask is dropped, so those layers would silently attend over "
                "the whole window. Load the model with `flash_attention_2`, `flash_attention_3` or a hub flash "
                "kernel, which apply `sliding_window` themselves, or disable context parallelism."
            )

        num_heads = text_config.num_attention_heads
        num_kv_heads = getattr(text_config, "num_key_value_heads", None) or num_heads
        cp_size = self.cp_mesh.size()
        # the tp plan already keeps `num_heads / tp_size` heads on this rank, and those are what cp splits
        distributed_config = getattr(config, "distributed_config", None)
        tp_size = 1 if distributed_config is None else distributed_config.tp_size
        splits = cp_size * tp_size
        if num_heads % splits != 0 or num_kv_heads % splits != 0:
            raise ValueError(
                f"Ulysses context parallelism splits the attention heads across the {cp_size} ranks of the `cp` mesh "
                f"and the {tp_size} ranks of the `tp` mesh, so `cp_size * tp_size` must divide both the number of "
                f"attention heads ({num_heads}) and the number of key/value heads ({num_kv_heads}) of "
                f"{model.__class__.__name__}. Use smaller `cp_size` or `tp_size`, or disable context parallelism."
            )

    def attention(
        self,
        attention_function: Callable,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        position_ids = self.position_ids
        if position_ids is None:
            raise ValueError(
                "Ulysses context parallelism needs the `position_ids` of the whole sequence: after gathering the "
                "sequence, they tell the attention kernel where each token sits and where packed documents start. "
                "None were seen for this forward. Pass `position_ids` to the model, or disable context parallelism."
            )
        # Multimodal rotary positions are `[3, batch, seq]`, of which the text axis is what matters here
        if position_ids.ndim == 3:
            position_ids = position_ids[0]
        group = self.cp_mesh.get_group()

        query = _gather_sequence_scatter_heads(query, group)
        key = _gather_sequence_scatter_heads(key, group)
        value = _gather_sequence_scatter_heads(value, group)
        kwargs["position_ids"] = position_ids
        # The kernel takes precomputed sequence boundaries over `position_ids`; those of the local slice describe
        # documents that are cut at the shard boundaries, so drop them and let the kernel rebuild them.
        for name in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
            kwargs.pop(name, None)
        # Attention sinks are one learnable logit per head: keep the ones of the heads this rank now holds.
        s_aux = kwargs.get("s_aux")
        if isinstance(s_aux, torch.Tensor) and s_aux.ndim == 1:
            kwargs["s_aux"] = s_aux.chunk(self.cp_mesh.size())[self.cp_mesh.get_local_rank()]

        # `attention_mask` was built for the local slice. Flash kernels rebuild the document boundaries from
        # `position_ids`; sdpa gets a mask over the full sequence, or `is_causal` when nothing is packed.
        if module.config._attn_implementation == "sdpa":
            attention_mask = _packed_causal_mask(position_ids)
        else:
            attention_mask = None

        attn_output, attn_weights = attention_function(module, query, key, value, attention_mask, **kwargs)
        return _gather_heads_scatter_sequence(attn_output, group), attn_weights


# Layer types that only mix tokens through softmax attention over the full sequence. Every other one (recurrent,
# convolutional, sparse-indexer, ...) mixes tokens outside attention and would silently be cut at the slice boundaries.
SOFTMAX_ATTENTION_LAYER_TYPES = ("full_attention", "sliding_attention", "chunked_attention")


def _check_context_parallelism(model: nn.Module) -> None:
    """Raise a `ValueError` if context parallelism would silently change the logits or the loss of `model`."""
    if not model._supports_attention_backend:
        raise ValueError(
            f"Context parallelism routes attention through `ALL_ATTENTION_FUNCTIONS`, but {model.__class__.__name__} "
            "does not declare that all its attention layers go through it with the model's kwargs "
            "(`_supports_attention_backend`), so they could run on this rank's slice only. Use another model, or "
            "disable context parallelism."
        )

    if model._is_stateful:
        raise ValueError(
            f"Context parallelism does not support {model.__class__.__name__}: its recurrent layers carry a state "
            "along the sequence, so they would run on this rank's slice only and be silently cut at the slice "
            "boundaries. Use another model, or disable context parallelism."
        )

    text_config = model.config.get_text_config()
    layer_types = set(getattr(text_config, "layer_types", None) or ())
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


cp_implementations: dict[str, type[CPImplementation]] = {"ulysses": UlyssesCP}


class _ContextParallelAttention:
    """Attention function of `ALL_ATTENTION_FUNCTIONS` routing the registered modules through a `CPImplementation`."""

    def __init__(self, attention_function: Callable):
        self.attention_function = attention_function
        self.implementations = weakref.WeakKeyDictionary()

    def register_module(self, module: nn.Module, implementation: CPImplementation) -> None:
        self.implementations[module] = implementation

    def __call__(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        implementation = self.implementations.get(module)
        if implementation is None:
            return self.attention_function(module, query, key, value, attention_mask, **kwargs)
        return implementation.attention(self.attention_function, module, query, key, value, attention_mask, **kwargs)


_context_parallel_attention_functions: dict[str, _ContextParallelAttention] = {}


def _install_context_parallel_attention(attn_implementation: str) -> _ContextParallelAttention:
    """
    Return the wrapper of `attn_implementation` in `ALL_ATTENTION_FUNCTIONS`. Installs it the first time, and again
    around the new function if another one was registered under that key since (loading a hub kernel does that).
    """
    from ..modeling_utils import ALL_ATTENTION_FUNCTIONS

    if attn_implementation not in ALL_ATTENTION_FUNCTIONS:
        raise ValueError(
            f"Attention implementation `{attn_implementation}` is not registered in `ALL_ATTENTION_FUNCTIONS`, so "
            f"context parallelism cannot route attention through it. Valid implementations are "
            f"{ALL_ATTENTION_FUNCTIONS.valid_keys()}."
        )
    current = ALL_ATTENTION_FUNCTIONS[attn_implementation]
    attention = _context_parallel_attention_functions.get(attn_implementation)
    if attention is None:
        attention = _ContextParallelAttention(current)
        _context_parallel_attention_functions[attn_implementation] = attention
    elif current is not attention:
        attention.attention_function = current
    if current is not attention:
        ALL_ATTENTION_FUNCTIONS.register(attn_implementation, attention)
    return attention


def register_context_parallel_attention(model: nn.Module, implementation: CPImplementation) -> None:
    """Route the attention calls of the modules of `model` through `implementation`."""
    attention = _install_context_parallel_attention(model.config._attn_implementation)
    for module in model.modules():
        attention.register_module(module, implementation)


def _slice_sequence(tensor: torch.Tensor, cp_rank: int, cp_size: int, dim: int = 1) -> torch.Tensor:
    """This rank's contiguous slice of `tensor` along its sequence dimension `dim`."""
    return tensor.chunk(cp_size, dim=dim)[cp_rank].contiguous()


def _shard_inputs(
    module: nn.Module, args: tuple, kwargs: dict, *, implementation: CPImplementation, attn_implementation: str
):
    """Replace the full-sequence inputs of `module` by this rank's contiguous slice of every sequence."""
    if args:
        kwargs.setdefault("input_ids", args[0])
        args = args[1:]

    if kwargs.get("past_key_values") is not None:
        raise ValueError(
            "Context parallelism only supports training: `past_key_values` are not allowed. Run the model on full "
            "sequences without a cache, or disable context parallelism."
        )
    kwargs["use_cache"] = False

    if module.config._attn_implementation != attn_implementation:
        raise ValueError(
            f"Context parallelism was set up for `attn_implementation='{attn_implementation}'`, but "
            f"{module.__class__.__name__} now uses `{module.config._attn_implementation}`, so its attention no longer "
            "sees the whole sequence. Reload the model with the attention implementation you want, or disable context "
            "parallelism."
        )
    # a kernel loaded after the model may have registered another function under the key
    _install_context_parallel_attention(attn_implementation)

    tokens_name = "input_ids" if kwargs.get("input_ids") is not None else "inputs_embeds"
    tokens = kwargs.get(tokens_name)
    if tokens is None:
        raise ValueError(
            f"Context parallelism splits the `input_ids` or `inputs_embeds` of {module.__class__.__name__} across the "
            "ranks of the `cp` mesh, and neither was given to `forward`."
        )

    cp_mesh = implementation.cp_mesh
    cp_size, cp_rank = cp_mesh.size(), cp_mesh.get_local_rank()
    seq_len = tokens.shape[1]
    if seq_len % cp_size:
        raise ValueError(
            f"Context parallelism splits every sequence across the {cp_size} ranks of the `cp` mesh, so the sequence "
            f"length must be divisible by `cp_size`, got {seq_len}. Pad the sequences to a multiple of `cp_size`."
        )

    position_ids = kwargs.get("position_ids")
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
    implementation.position_ids = position_ids

    attention_mask = kwargs.get("attention_mask")
    if attention_mask is not None:
        if not torch.all(attention_mask == 1):
            raise ValueError(
                "Context parallelism does not support attention masks yet: the ranks of a `cp` mesh already hold "
                "different slices of the same sequences. Pack the sequences together and pass `position_ids` that "
                "restart at each document boundary."
            )
        kwargs["attention_mask"] = None

    labels = kwargs.get("labels")
    if labels is not None:
        shift_labels = kwargs.get("shift_labels")
        if shift_labels is None:
            shift_labels = torch.nn.functional.pad(labels[..., 1:], (0, 1), value=-100)
        if kwargs.get("num_items_in_batch") is None:
            kwargs["num_items_in_batch"] = (shift_labels != -100).sum()
        kwargs["shift_labels"] = _slice_sequence(shift_labels, cp_rank, cp_size)
        kwargs["labels"] = _slice_sequence(labels, cp_rank, cp_size)

    kwargs[tokens_name] = _slice_sequence(tokens, cp_rank, cp_size)
    # rotary positions can be `[3, batch, seq]`, so their sequence axis is the last one
    kwargs["position_ids"] = _slice_sequence(position_ids, cp_rank, cp_size, dim=-1)
    return args, kwargs


def _sum_loss_over_cp_group(module: nn.Module, args: tuple, output, *, cp_mesh: DeviceMesh):
    """
    Replace the loss by the loss of the full sequences, which every rank reports. Its gradient stays this rank's,
    multiplied by `cp_size`, so that FSDP2's average over the cp ranks gives the gradient of the full-sequence loss.
    """
    loss = getattr(output, "loss", None)
    if loss is None:
        return output
    total = loss.detach().clone()
    dist.all_reduce(total, group=cp_mesh.get_group())
    output.loss = total + (loss - loss.detach()) * cp_mesh.size()
    return output


def apply_context_parallelism(model: nn.Module, cp_mesh: DeviceMesh, cp_implementation: str) -> nn.Module:
    """
    Shard the inputs of `model` across the ranks of `cp_mesh` and let its attention see the whole sequence.

    `model` still takes the full sequences, as in an unsharded run: a forward pre-hook splits them, and a forward hook
    sums the loss back over the `cp` group.
    """
    if cp_implementation not in cp_implementations:
        raise ValueError(
            f"Unknown `cp_implementation={cp_implementation}`: registered implementations are "
            f"{list(cp_implementations)}."
        )
    implementation = cp_implementations[cp_implementation](cp_mesh)
    _check_context_parallelism(model)
    implementation.check(model)
    register_context_parallel_attention(model, implementation)

    model.register_forward_pre_hook(
        partial(_shard_inputs, implementation=implementation, attn_implementation=model.config._attn_implementation),
        with_kwargs=True,
    )
    model.register_forward_hook(partial(_sum_loss_over_cp_group, cp_mesh=cp_mesh))
    return model
