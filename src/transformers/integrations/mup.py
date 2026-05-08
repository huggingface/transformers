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
"""
Maximal Update Parametrization (μP).

Implementation based on https://github.com/microsoft/mup
Paper: https://huggingface.co/papers/2203.03466

The Adam recipe (Table 8 of the paper). With base width ``d0`` and current width ``d``,
``m = d / d0``:

* input weights, biases, LayerNorm: forward x1, init std unchanged, lr unchanged
* hidden weights (both fan-in and fan-out scale with width): init std divided by sqrt(m), lr unchanged
* output (readout) weights: forward divided by m, init std unchanged, lr divided by m

Attention logits use ``1/d_head`` instead of ``1/sqrt(d_head)``.

The architectural changes (readout swap, attention scale) and the hidden-weight init
rescale are applied by [`PreTrainedModel`] when ``config.mup`` is set; this module
exposes the public layer ([`MuReadout`]) and the optimizer / diagnostic helpers
([`build_mup_param_groups`], [`coord_check`]).
"""

import torch
from torch import nn


class MuReadout(nn.Linear):
    """
    Drop-in replacement for an output projection ([`torch.nn.Linear`]) under μP. The input is divided by
    `width_mult` before the linear, so that the matmul contribution to the logits is bounded as width grows; the
    bias (if any) is unaffected. Initialization, weight tying, and state-dict layout are identical to a regular
    [`torch.nn.Linear`] of the same shape.

    Args:
        in_features (`int`):
            Size of each input sample.
        out_features (`int`):
            Size of each output sample (typically the vocabulary size).
        bias (`bool`, *optional*, defaults to `False`):
            Whether to add a learnable bias to the output.
        width_mult (`float`, *optional*, defaults to `1.0`):
            Ratio of the current hidden size to the base hidden size (`config.hidden_size / config.mup_base_width`).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, width_mult: float = 1.0):
        super().__init__(in_features, out_features, bias=bias)
        self.width_mult = width_mult

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x / self.width_mult)


def _matrix_like_param_ids(model: nn.Module) -> set:
    """
    Return the ids of every parameter that should be treated as "matrix-like" under μP, i.e. weight tensors of
    [`torch.nn.Linear`] / [`~pytorch_utils.Conv1D`] modules whose fan-in and fan-out both scale with width.

    The output projection ([`MuReadout`]) is excluded because its output dimension is finite (vocabulary size) and
    therefore the readout weight has only one "infinite" dimension. Embeddings, biases, and LayerNorm parameters
    are also vector-like and excluded. Robust to weight tying.
    """
    from ..pytorch_utils import Conv1D

    ids = set()
    for module in model.modules():
        if isinstance(module, MuReadout):
            continue
        if isinstance(module, (nn.Linear, Conv1D)):
            ids.add(id(module.weight))
    return ids


def build_mup_param_groups(model: nn.Module, lr: float, weight_decay: float = 0.0) -> list:
    """
    Build [`torch.optim`] parameter groups implementing the μP learning-rate rule for Adam-style optimizers.

    Matrix-like parameters (hidden weight tensors with both fan-in and fan-out scaling with width) get `lr /
    width_mult`. Vector-like parameters (the readout weight, embeddings, biases, LayerNorm) keep `lr`. Each
    parameter is emitted in exactly one group, even when tied. This mirrors the canonical
    [`mup.MuAdam`](https://github.com/microsoft/mup/blob/main/mup/optim.py) grouping.

    Args:
        model (`torch.nn.Module`):
            The model whose parameters should be grouped.
        lr (`float`):
            The base learning rate.
        weight_decay (`float`, *optional*, defaults to `0.0`):
            Weight decay applied to every parameter group.

    Returns:
        `list[dict]`: A list of two parameter groups (matrix-like + vector-like) suitable for passing to a
        [`torch.optim`] optimizer such as [`torch.optim.AdamW`].
    """
    width_mult = 1.0
    for module in model.modules():
        if isinstance(module, MuReadout):
            width_mult = module.width_mult
            break

    matrix_ids = _matrix_like_param_ids(model)
    seen = set()
    matrix_like = []
    vector_like = []
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if id(p) in matrix_ids:
            matrix_like.append(p)
        else:
            vector_like.append(p)

    return [
        {"params": matrix_like, "lr": lr / width_mult, "weight_decay": weight_decay * width_mult},
        {"params": vector_like, "lr": lr, "weight_decay": weight_decay},
    ]


@torch.no_grad()
def coord_check(model_factory, widths, batch, n_steps: int = 4, lr: float = 1e-3) -> dict:
    """
    Run a μP coordinate check across widths.

    For each width, this function freshly instantiates a model, registers a forward hook on every linear/embedding
    submodule, runs `n_steps` optimizer steps on `batch`, and records the mean absolute value of the module
    outputs. Under correct μP the per-module curves are approximately width-invariant; under standard
    parametrization (SP, the default training regime that μP replaces) they fan out with width.

    Args:
        model_factory (`Callable[[int], torch.nn.Module]`):
            Callable returning a freshly-initialized model for the requested width.
        widths (`list[int]`):
            Widths to evaluate.
        batch (`dict[str, torch.Tensor]`):
            Inputs passed as `**batch` to the model. The model is expected to return an object with a `loss`
            attribute (e.g. an [`~modeling_outputs.CausalLMOutput`] when `labels` are provided).
        n_steps (`int`, *optional*, defaults to `4`):
            Number of optimizer steps to run before recording.
        lr (`float`, *optional*, defaults to `1e-3`):
            Base learning rate passed to [`build_mup_param_groups`].

    Returns:
        `dict[int, dict[str, list[float]]]`: Mapping from width to `{module_name: [mean_abs_output_per_step]}`.
    """
    out = {}
    for w in widths:
        model = model_factory(w)
        model.train()
        records = {}
        hooks = []

        def make_hook(name):
            def hook(_mod, _inp, output):
                t = output[0] if isinstance(output, tuple) else output
                if torch.is_tensor(t):
                    records.setdefault(name, []).append(t.detach().float().abs().mean().item())

            return hook

        from ..pytorch_utils import Conv1D

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, Conv1D, nn.Embedding, MuReadout)):
                hooks.append(module.register_forward_hook(make_hook(name)))

        groups = build_mup_param_groups(model, lr=lr)
        opt = torch.optim.AdamW(groups)
        for _ in range(n_steps):
            opt.zero_grad()
            with torch.enable_grad():
                output = model(**batch)
                loss = output.loss
            loss.backward()
            opt.step()

        for h in hooks:
            h.remove()
        out[w] = records
    return out
