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

import contextlib
import inspect
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


def split_sequence(tensor: torch.Tensor, dim: int, sp_mesh: DeviceMesh) -> torch.Tensor:
    """This rank's contiguous slice of `tensor` along its sequence dimension `dim`."""
    sp_size = sp_mesh.size()
    if tensor.shape[dim] % sp_size:
        raise ValueError(
            f"Sequence parallelism splits every sequence across the {sp_size} ranks of the `sp` mesh, so the sequence "
            f"length must be divisible by `sp_size`, got {tensor.shape[dim]}. Pad the sequences to a multiple of "
            "`sp_size`."
        )
    return tensor.chunk(sp_size, dim=dim)[sp_mesh.get_local_rank()].contiguous()


class SequenceParallelLayer:
    """
    Base class of the sequence parallel styles, with the hooks of `TensorParallelLayer`: `install_forward` replaces
    `module.forward` by one that runs `transform_inputs_pre_forward` first, then the original forward inside
    `context_around_forward`, then `transform_output_post_forward`. The styles shard no parameter, so the parameter
    hooks of the other parallelisms have no counterpart here. Not to be confused with the `"sequence_parallel"` style
    of `tensor_parallel.py`, which shards activations between tensor parallel layers.
    """

    def transform_inputs_pre_forward(self, module: nn.Module, args: tuple, kwargs: dict, sp_mesh: DeviceMesh):
        return args, kwargs

    def transform_output_post_forward(self, module: nn.Module, output, sp_mesh: DeviceMesh):
        return output

    def context_around_forward(self, module: nn.Module, sp_mesh: DeviceMesh):
        return contextlib.nullcontext()

    def install_forward(self, module: nn.Module, sp_mesh: DeviceMesh, **options):
        """Replace `module.forward`; the keyword `options` go to `transform_inputs_pre_forward`."""
        original_forward = module.forward

        @wraps(original_forward)
        def sp_forward(*args, **kwargs):
            args, kwargs = self.transform_inputs_pre_forward(module, args, kwargs, sp_mesh, **options)
            with self.context_around_forward(module, sp_mesh):
                output = original_forward(*args, **kwargs)
            return self.transform_output_post_forward(module, output, sp_mesh)

        module.forward = sp_forward
        return module


class SplitSequenceParallel(SequenceParallelLayer):
    """
    Give each decoder layer only the tokens assigned to this rank:

        Before the first layer:           [0 1 2 3 | 4 5 6 7]  (full sequence)
        Rank 0, first layer:              [0 1 2 3]            (split hidden states and positions)
        Rank 1, first layer:              [4 5 6 7]
        Rank 0/1, later layers:           already-local hidden states; split only positions

    The model computes `position_embeddings` for all tokens before the layers run, and passes them to every layer.
    Each layer therefore takes the positions matching its local hidden states.
    """

    def transform_inputs_pre_forward(
        self, module: nn.Module, args: tuple, kwargs: dict, sp_mesh: DeviceMesh, *, split_hidden_states: bool = False
    ):
        if split_hidden_states:
            if args:
                args = (split_sequence(args[0], 1, sp_mesh), *args[1:])
            else:
                kwargs["hidden_states"] = split_sequence(kwargs["hidden_states"], 1, sp_mesh)
        position_embeddings = kwargs.get("position_embeddings")
        if isinstance(position_embeddings, tuple):
            kwargs["position_embeddings"] = tuple(split_sequence(x, -2, sp_mesh) for x in position_embeddings)
        elif position_embeddings is not None:
            kwargs["position_embeddings"] = split_sequence(position_embeddings, -2, sp_mesh)
        return args, kwargs


class SequenceParallelAttention(SequenceParallelLayer):
    """
    Keep the attention module unchanged, but temporarily wrap its attention function (e.g. SDPA):

    1) On entry, save the selected function in `ALL_ATTENTION_FUNCTIONS` and replace it with
       `partial(self.attention, sp_mesh, original_attention)`.
    2) At `yield`, the module's forward pass runs. Its attention lookup now returns the wrapper, which can
       exchange data across ranks before and after calling the original attention function.
    3) On exit, restore the original entry in `finally`, even if the forward pass raises an error.
    """

    def attention(self, sp_mesh: DeviceMesh, attention_function, module, query, key, value, attention_mask, **kwargs):
        raise NotImplementedError

    @contextlib.contextmanager
    def context_around_forward(self, module: nn.Module, sp_mesh: DeviceMesh):
        from ..modeling_utils import ALL_ATTENTION_FUNCTIONS

        attn_implementation = module.config._attn_implementation
        original_attention = ALL_ATTENTION_FUNCTIONS[attn_implementation]
        sp_attention = partial(self.attention, sp_mesh, original_attention)
        ALL_ATTENTION_FUNCTIONS.register(attn_implementation, sp_attention)
        try:
            yield
        finally:
            ALL_ATTENTION_FUNCTIONS.register(attn_implementation, original_attention)


class UlyssesAttention(SequenceParallelAttention):
    """
    Trade sequence slices for head slices, so attention on each rank can see every token.
    Example: eight tokens (0-7), four heads (h0-h3), two ranks:

                                  Rank 0                           Rank 1
        Local Q/K/V          tokens [0 1 2 3], heads [h0 h1 h2 h3]    tokens [4 5 6 7], heads [h0 h1 h2 h3]
                                             |                                    |
                                             +-------- first all-to-all --------+
                                             |                                    |
        Attention Q/K/V      tokens [0 1 2 3 4 5 6 7], heads [h0 h1]   tokens [0 1 2 3 4 5 6 7], heads [h2 h3]
                             attend with full mask                    attend with full mask
                                             |                                    |
                                             +-------- second all-to-all -------+
                                             |                                    |
        Local output        tokens [0 1 2 3], heads [h0 h1 h2 h3]    tokens [4 5 6 7], heads [h0 h1 h2 h3]

    The first exchange gathers the whole sequence for each rank's assigned heads. The original attention function
    runs on those heads; the second exchange returns each token's output with all heads to its original rank.
    """

    def install_forward(self, module: nn.Module, sp_mesh: DeviceMesh):
        self._validate_attention(module, sp_mesh)
        return super().install_forward(module, sp_mesh)

    def _validate_attention(self, module: nn.Module, sp_mesh: DeviceMesh) -> None:
        from ..modeling_utils import ALL_ATTENTION_FUNCTIONS

        attn_implementation = module.config._attn_implementation
        if attn_implementation not in ALL_ATTENTION_FUNCTIONS or attn_implementation == "flex_attention":
            raise ValueError(
                f"Ulysses sequence parallelism routes the attention calls of {module.__class__.__name__} through "
                f"`ALL_ATTENTION_FUNCTIONS`, but it uses `attn_implementation={attn_implementation}`, which takes no "
                "attention mask of the whole sequence. Load the model with `sdpa`, `flash_attention_2`, "
                "`flash_attention_3` or a hub flash kernel, or disable sequence parallelism."
            )

        config = module.config.get_text_config()
        num_kv_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
        sp_size = sp_mesh.size()
        if config.num_attention_heads % sp_size or num_kv_heads % sp_size:
            raise ValueError(
                f"Ulysses sequence parallelism splits the attention heads across the {sp_size} ranks of the `sp` "
                f"mesh, so `sp_size` must divide both the number of attention heads "
                f"({config.num_attention_heads}) and the number of key/value heads ({num_kv_heads}) of "
                f"{module.__class__.__name__}. Use another `sp_size`, or disable sequence parallelism."
            )

    def attention(self, sp_mesh: DeviceMesh, attention_function, module, query, key, value, attention_mask, **kwargs):
        sp_size, group = sp_mesh.size(), sp_mesh.get_group()

        def _all_to_all(tensor):
            """Split dim 1 across the ranks of the `sp` mesh, concatenate what they send back along dim 2."""
            chunks = torch.stack(tensor.chunk(sp_size, dim=1))
            return torch.cat(all_to_all_single_autograd(chunks, None, None, group).unbind(), dim=2)

        # [batch, heads, seq / sp_size, head_dim] -> [batch, heads / sp_size, seq, head_dim]
        query, key, value = (_all_to_all(x) for x in (query, key, value))
        if kwargs.get("s_aux") is not None:  # attention sinks: one logit per head
            kwargs["s_aux"] = kwargs["s_aux"].chunk(sp_size)[sp_mesh.get_local_rank()]
        output, attn_weights = attention_function(module, query, key, value, attention_mask, **kwargs)
        # [batch, seq, heads / sp_size, head_dim] -> [batch, seq / sp_size, heads, head_dim]
        return _all_to_all(output), attn_weights


class ModelSequenceParallel(SequenceParallelLayer):
    """
    Handle the model-level work for sequence parallelism. In an `sp_plan`, the empty path `""` selects the model itself
    and applies this style under the name `"sp_model"`.

    Pre-forward (`transform_inputs_pre_forward`):
      1) Reject a past-key-value cache and disable caching.
      2) Pad the inputs so the sequence divides evenly across SP ranks.
      3) Give each rank its slice of shifted labels, but count valid labels across the full sequence so all partial
         losses use the same denominator.

    Post-forward (`transform_output_post_forward`):
      1) Sum the partial losses so every rank reports the full-sequence loss.
      2) Scale each rank's local loss gradient to account for gradient averaging during distributed training.
    """

    def transform_inputs_pre_forward(self, module: nn.Module, args: tuple, kwargs: dict, sp_mesh: DeviceMesh):
        # 1) Reject a cache (past tokens are not sharded) and bind positional inputs by name.
        if kwargs.get("past_key_values") is not None:
            raise ValueError(
                "Sequence parallelism only supports training: `past_key_values` are not allowed. Run the model on "
                "full sequences without a cache, or disable sequence parallelism."
            )
        kwargs["use_cache"] = False
        if args:
            kwargs, args = {**inspect.signature(module.forward).bind_partial(*args).arguments, **kwargs}, ()

        # 2) Pad the full sequence before the model builds masks and splits it across ranks.
        tokens = kwargs["inputs_embeds"] if kwargs.get("input_ids") is None else kwargs["input_ids"]
        if pad_len := -tokens.shape[1] % sp_mesh.size():
            padding_values = {
                "input_ids": getattr(module.config, "pad_token_id", None) or 0,
                "inputs_embeds": 0,
                "attention_mask": 0,
                "labels": -100,
                "shift_labels": -100,
            }
            for name, value in padding_values.items():
                if (tensor := kwargs.get(name)) is not None:
                    padding = tensor.new_full((tensor.shape[0], pad_len, *tensor.shape[2:]), value)
                    kwargs[name] = torch.cat([tensor, padding], dim=1)
            if (position_ids := kwargs.get("position_ids")) is not None:
                # the padding continues the positions of the last document of every row
                steps = torch.arange(1, pad_len + 1, device=position_ids.device)
                kwargs["position_ids"] = torch.cat([position_ids, position_ids[..., -1:] + steps], dim=-1)

        # 3) Give each rank its own shifted labels, but normalize by the full sequence's valid-token count.
        if (labels := kwargs.get("labels")) is not None:
            shift_labels = kwargs.get("shift_labels")
            if shift_labels is None:
                shift_labels = torch.nn.functional.pad(labels[..., 1:], (0, 1), value=-100)
            if kwargs.get("num_items_in_batch") is None:
                kwargs["num_items_in_batch"] = (shift_labels != -100).sum()
            kwargs["shift_labels"] = split_sequence(shift_labels, 1, sp_mesh)
        return args, kwargs

    def transform_output_post_forward(self, module: nn.Module, output, sp_mesh: DeviceMesh):
        loss = getattr(output, "loss", None)
        if loss is None:
            return output
        total = loss.detach().clone()
        # 1) Sum the partial losses so every rank reports the full-sequence loss.
        dist.all_reduce(total, group=sp_mesh.get_group())
        # 2) Scale each rank's local loss gradient to account for gradient averaging during distributed training.
        output.loss = total + (loss - loss.detach()) * sp_mesh.size()
        return output


ALL_SP_STYLES: dict[str, SequenceParallelLayer] = {
    "sp_model": ModelSequenceParallel(),
    "sp_split_sequence": SplitSequenceParallel(),
    "sp_ulysses": UlyssesAttention(),
}


def apply_sequence_parallelism(model: nn.Module, sp_mesh: DeviceMesh, sp_plan: dict[str, str]) -> nn.Module:
    if not model._supports_attention_backend:
        raise ValueError(
            f"Sequence parallelism routes attention through `ALL_ATTENTION_FUNCTIONS`, but {model.__class__.__name__} "
            "does not declare that all its attention layers go through it with the model's kwargs "
            "(`_supports_attention_backend`), so they could run on this rank's slice only. Use another model, or "
            "disable sequence parallelism."
        )

    layer_types = set(getattr(model.config.get_text_config(), "layer_types", None) or ())
    if unsupported := layer_types - {"full_attention", "sliding_attention", "chunked_attention"}:
        raise ValueError(
            f"Sequence parallelism does not support layer types {sorted(unsupported)} "
            f"in {model.__class__.__name__}. Use another model or disable sequence parallelism."
        )

    # `loss_type=None` (base models, and heads whose name maps to no loss) falls back to the causal LM loss
    if model.loss_type is not None and LOSS_MAPPING.get(model.loss_type) is not ForCausalLMLoss:
        raise ValueError(
            f"Sequence parallelism only supports the causal language modeling loss, but {model.__class__.__name__} "
            f"uses `loss_type={model.loss_type}`. Its loss would be computed from this rank's slice of the shifted "
            "labels while the logits stay sharded. Use a `*ForCausalLM` model, or disable sequence parallelism."
        )

    modules = dict(model.named_modules())
    # the first decoder layer still gets the whole sequence: it is the one that cuts it
    first_layer = next(name for name, style in sp_plan.items() if style == "sp_split_sequence")
    for name, style in sp_plan.items():
        options = {"split_hidden_states": name == first_layer} if style == "sp_split_sequence" else {}
        ALL_SP_STYLES[style].install_forward(modules[name], sp_mesh, **options)
    return model
