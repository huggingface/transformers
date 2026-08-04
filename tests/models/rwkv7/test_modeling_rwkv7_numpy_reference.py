# Copyright 2026 The RWKV team and The HuggingFace Inc. team. All rights reserved.
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
"""An independent RWKV-7 forward pass, in numpy, to check the torch one against.

Every other test in this directory compares one part of the torch implementation
against another part of it -- chunked against sequential, prefill against
incremental decode, padded against unpadded. Those are strong tests of internal
consistency and they are blind by construction to anything both sides get wrong
together, which is the failure mode that actually happened here: the `ln_x`
group-norm epsilon was scaled by the head *count* instead of the head *width*, and
since the two coincide at hidden_size 4096, every measurement taken on the 7.2B
checkpoint agreed with every other one.

So this file re-derives the forward pass from the RWKV-7 reference definition, in
numpy, sharing no code with the model. It is written per token over a batch of one,
because the reference is a recurrence and the clearest statement of a recurrence is
a loop; the point is to be obviously right rather than fast.

The configuration is deliberately non-square -- more heads than head_dim -- so that
a quantity indexed by the wrong one of those does not silently agree.
"""

import unittest

import numpy as np

from transformers import Rwkv7Config, is_torch_available
from transformers.testing_utils import require_torch, torch_device


if is_torch_available():
    import torch

    from transformers import Rwkv7ForCausalLM


_INV_SQRT_E = 0.6065306597126334


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _layer_norm(x, weight, bias, eps):
    return (x - x.mean()) / np.sqrt(x.var() + eps) * weight + bias


def _group_norm(x, weight, bias, num_heads, head_dim, eps):
    """Normalise within each head, then scale and shift over the flat channels.

    `torch.nn.GroupNorm(num_heads, C, eps=...)` puts eps inside the square root
    alongside the biased variance, and this must match that exactly: the epsilon is
    the whole reason this file exists.
    """
    heads = x.reshape(num_heads, head_dim)
    normed = (heads - heads.mean(axis=1, keepdims=True)) / np.sqrt(heads.var(axis=1, keepdims=True) + eps)
    return normed.reshape(-1) * weight + bias


class NumpyRwkv7:
    """The reference forward, reading weights straight off a `Rwkv7ForCausalLM`."""

    def __init__(self, model):
        self.config = model.config
        self.num_heads = model.config.num_heads
        self.head_dim = model.config.head_dim
        self.norm_eps = model.config.norm_eps
        take = lambda t: t.detach().cpu().double().numpy()  # noqa: E731

        self.emb = take(model.rwkv7.emb.weight)
        self.head = take(model.head.weight)
        self.ln_out = (take(model.rwkv7.ln_out.weight), take(model.rwkv7.ln_out.bias))
        block0 = model.rwkv7.blocks[0]
        self.ln0 = (take(block0.ln0.weight), take(block0.ln0.bias))

        self.layers = []
        for block in model.rwkv7.blocks:
            att, ffn = block.att, block.ffn
            self.layers.append(
                {
                    "ln1": (take(block.ln1.weight), take(block.ln1.bias)),
                    "ln2": (take(block.ln2.weight), take(block.ln2.bias)),
                    "x_": {name: take(getattr(att, f"x_{name}")).reshape(-1) for name in "rwkvag"},
                    "w1": take(att.w1),
                    "w2": take(att.w2),
                    "w0": take(att.w0).reshape(-1),
                    "a1": take(att.a1),
                    "a2": take(att.a2),
                    "a0": take(att.a0).reshape(-1),
                    "g1": take(att.g1),
                    "g2": take(att.g2),
                    "v1": take(att.v1),
                    "v2": take(att.v2),
                    "v0": take(att.v0).reshape(-1),
                    "k_k": take(att.k_k).reshape(-1),
                    "k_a": take(att.k_a).reshape(-1),
                    "r_k": take(att.r_k),
                    "receptance": take(att.receptance.weight),
                    "key": take(att.key.weight),
                    "value": take(att.value.weight),
                    "output": take(att.output.weight),
                    "ln_x": (take(att.ln_x.weight), take(att.ln_x.bias)),
                    "ffn_x_k": take(ffn.x_k).reshape(-1),
                    "ffn_key": take(ffn.key.weight),
                    "ffn_value": take(ffn.value.weight),
                }
            )

    def _time_mix(self, x, shift, state, v_first, layer, layer_id):
        heads, width = self.num_heads, self.head_dim
        mixed = {name: x + layer["x_"][name] * (shift - x) for name in "rwkvag"}

        r = layer["receptance"] @ mixed["r"]
        k = layer["key"] @ mixed["k"]
        v = layer["value"] @ mixed["v"]

        w_log = -_INV_SQRT_E * _sigmoid(np.tanh(mixed["w"] @ layer["w1"]) @ layer["w2"] + layer["w0"])
        a = _sigmoid(mixed["a"] @ layer["a1"] @ layer["a2"] + layer["a0"])
        g = _sigmoid(mixed["g"] @ layer["g1"]) @ layer["g2"]

        if layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * _sigmoid(mixed["v"] @ layer["v1"] @ layer["v2"] + layer["v0"])

        kk = (k * layer["k_k"]).reshape(heads, width)
        kk = kk / np.linalg.norm(kk, axis=-1, keepdims=True)
        kk = kk.reshape(-1)
        k = k + k * (a - 1.0) * layer["k_a"]

        decay = np.exp(w_log).reshape(heads, width)
        r_h, k_h, v_h = r.reshape(heads, width), k.reshape(heads, width), v.reshape(heads, width)
        kk_h, b_h = kk.reshape(heads, width), (kk * a).reshape(heads, width)

        out = np.empty((heads, width), dtype=np.float64)
        for h in range(heads):
            # `sa` reads the state before this token updates it; `out` reads it after.
            sa = (-kk_h[h]) @ state[h]
            state[h] = decay[h][:, None] * state[h] + b_h[h][:, None] * sa[None, :] + k_h[h][:, None] * v_h[h][None, :]
            out[h] = r_h[h] @ state[h]

        y = _group_norm(out.reshape(-1), *layer["ln_x"], heads, width, self.norm_eps * width)
        bonus = ((r_h * k_h * layer["r_k"]).sum(axis=-1, keepdims=True) * v_h).reshape(-1)
        return layer["output"] @ ((y + bonus) * g), v_first

    def _channel_mix(self, x, shift, layer):
        xk = x + layer["ffn_x_k"] * (shift - x)
        return layer["ffn_value"] @ np.maximum(layer["ffn_key"] @ xk, 0.0) ** 2

    def __call__(self, token_ids):
        """Greedy forward over one sequence; returns logits at every position."""
        eps = self.norm_eps
        att_shift = [np.zeros(self.config.hidden_size) for _ in self.layers]
        ffn_shift = [np.zeros(self.config.hidden_size) for _ in self.layers]
        states = [np.zeros((self.num_heads, self.head_dim, self.head_dim)) for _ in self.layers]

        logits = []
        for token in token_ids:
            x = _layer_norm(self.emb[token], *self.ln0, eps)
            v_first = None
            for layer_id, layer in enumerate(self.layers):
                normed = _layer_norm(x, *layer["ln1"], eps)
                delta, v_first = self._time_mix(
                    normed, att_shift[layer_id], states[layer_id], v_first, layer, layer_id
                )
                att_shift[layer_id] = normed
                x = x + delta

                normed = _layer_norm(x, *layer["ln2"], eps)
                x = x + self._channel_mix(normed, ffn_shift[layer_id], layer)
                ffn_shift[layer_id] = normed

            logits.append(self.head @ _layer_norm(x, *self.ln_out, eps))
        return np.stack(logits)


def _non_square_config():
    """More heads than head_dim, so head count and head width cannot be confused.

    `norm_eps` is far above any realistic value, and that is the point. The bug this
    file exists to catch is `ln_x`'s epsilon being scaled by the head count instead
    of the head width; at `norm_eps=1e-5` those are 6e-5 and 8e-5 against a variance
    of order one, which moves the logits by less than fp32 noise. Written that way
    the test passed with the bug reinstated -- it was pinning the formula in name
    only. Raising the epsilon makes the axis it is indexed by visible, and the axis
    is what is under test; the value is not.
    """
    return Rwkv7Config(
        vocab_size=61,
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=3,
        num_heads=6,
        head_dim=8,
        norm_eps=0.05,
        decay_low_rank_dim=8,
        a_low_rank_dim=8,
        gate_low_rank_dim=8,
        v_low_rank_dim=8,
    )


@require_torch
class Rwkv7NumpyReferenceTest(unittest.TestCase):
    def _model(self, config):
        torch.manual_seed(0)
        model = Rwkv7ForCausalLM(config)
        # The initialiser leaves several tensors at zero, which would let a whole
        # branch be deleted without moving a logit. Give everything a scale.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.5)
        return model.to(torch_device).eval().to(torch.float64)

    def test_chunked_prefill_matches_a_numpy_forward_written_from_the_reference(self):
        config = _non_square_config()
        model = self._model(config)
        token_ids = [3, 17, 42, 8, 55, 1, 29, 33, 12, 60]

        with torch.no_grad():
            torch_logits = model(input_ids=torch.tensor([token_ids], device=torch_device)).logits[0]

        numpy_logits = NumpyRwkv7(model)(token_ids)
        # The WKV accumulates in fp32 whatever the activation dtype, by design, so
        # this is the width the comparison is really at -- not float64's.
        np.testing.assert_allclose(torch_logits.detach().cpu().numpy(), numpy_logits, rtol=2e-5, atol=2e-5)

    def test_sequential_decode_matches_the_same_numpy_forward(self):
        """The other WKV path, against the same external witness.

        `rwkv7_chunked` and `rwkv7_recurrent` already check each other; what that
        cannot tell you is which of them is right.
        """
        config = _non_square_config()
        model = self._model(config)
        token_ids = [7, 19, 2, 44, 31]

        with torch.no_grad():
            state, steps = None, []
            for token in token_ids:
                out = model(input_ids=torch.tensor([[token]], device=torch_device), state=state, use_cache=True)
                state = out.state
                steps.append(out.logits[0, 0])
            torch_logits = torch.stack(steps)

        numpy_logits = NumpyRwkv7(model)(token_ids)
        # The WKV accumulates in fp32 whatever the activation dtype, by design, so
        # this is the width the comparison is really at -- not float64's.
        np.testing.assert_allclose(torch_logits.detach().cpu().numpy(), numpy_logits, rtol=2e-5, atol=2e-5)
