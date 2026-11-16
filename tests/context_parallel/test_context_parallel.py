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
The context parallel styles, each on its own: what `cp_ulysses` gives an attention function on 2 ranks, its routing
and its refusals, `cp_split_sequence` on a decoder layer, and `cp_model` on the inputs and the loss of a model. What
the styles add up to, the logits, loss and gradients of real models, is tested by `ContextParallelTesterMixin`.
"""

import types
import unittest
from unittest import mock

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM, MistralConfig
from transformers.distributed.context_parallel import (
    ALL_CP_STYLES,
    ContextParallelAttention,
    SplitSequenceParallel,
    UlyssesAttention,
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.testing_utils import is_fsdp_test

from ..distributed_testing_utils import run_distributed


def _mesh(size=2, rank=0):
    """A `cp` mesh that only has to report its size and the caller's rank."""
    return types.SimpleNamespace(size=lambda: size, get_local_rank=lambda: rank)


def _config(**kwargs):
    """A `LlamaConfig` small enough for CPU tests, with `sdpa` attention."""
    defaults = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 99,
        "attn_implementation": "sdpa",
    }
    return LlamaConfig(**{**defaults, **kwargs})


class _Attention(torch.nn.Module):
    """An attention module whose forward runs the function `ALL_ATTENTION_FUNCTIONS` holds for it."""

    def __init__(self):
        super().__init__()
        self.config = _config()

    def forward(self, query, key, value, attention_mask=None, **kwargs):
        attention_function = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        return attention_function(self, query, key, value, attention_mask, **kwargs)


class _Layer(torch.nn.Module):
    """A decoder layer stand-in that records what its forward receives."""

    def __init__(self):
        super().__init__()
        self.seen = {}

    def forward(self, hidden_states, **kwargs):
        self.seen = {"hidden_states": hidden_states, **kwargs}
        return hidden_states


class _SpyAttention(ContextParallelAttention):
    """An attention style that records the modules it runs, then runs them as they are."""

    def __init__(self):
        self.seen = []

    def attention(self, cp_mesh, attention_function, module, *args, **kwargs):
        self.seen.append(module)
        return attention_function(module, *args, **kwargs)


# =============================================================================
# On 2 ranks
# =============================================================================


def _test_ulysses_packed_sliding(rank):
    """This rank's slice, and its q/k/v gradients, match the attention on the whole rows."""
    cp_size = dist.get_world_size()
    mesh = init_device_mesh("cpu", (cp_size,))
    batch, heads, kv_heads, seq, head_dim = 2, 4, 2, 8, 4
    config = MistralConfig(
        hidden_size=heads * head_dim,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        sliding_window=4,
        attn_implementation="sdpa",
    )
    module = MistralAttention(config, layer_idx=0)
    # two documents per row, 5 then 3 tokens, so that the window of position 4 reaches past the boundary at 4 and
    # stops before position 0: that window is what this test is about
    position_ids = torch.cat([torch.arange(5), torch.arange(3)]).expand(batch, -1)
    mask = create_sliding_window_causal_mask(
        config, torch.zeros(batch, seq, config.hidden_size), None, None, position_ids=position_ids
    )
    assert not mask[0, 0, 4, 0] and mask[0, 0, 4, 1:4].all()
    generator = torch.Generator().manual_seed(0)
    query, key, value = (
        torch.randn(batch, n, seq, head_dim, generator=generator, requires_grad=True)
        for n in (heads, kv_heads, kv_heads)
    )

    # this rank holds a slice of the sequence with every head, and the mask of the whole rows
    local = [t.detach().chunk(cp_size, dim=2)[rank].clone().requires_grad_() for t in (query, key, value)]
    output, _ = UlyssesAttention().attention(mesh, sdpa_attention_forward, module, *local, mask)
    reference, _ = sdpa_attention_forward(module, query, key, value, mask)
    (output.sum() + reference.sum()).backward()

    torch.testing.assert_close(output, reference.chunk(cp_size, dim=1)[rank])
    for local_tensor, tensor in zip(local, (query, key, value)):
        torch.testing.assert_close(local_tensor.grad, tensor.grad.chunk(cp_size, dim=2)[rank])


def _test_ulysses_kwargs(rank):
    """The attention function gets the model's kwargs unchanged, and its share of the attention sinks."""
    cp_size = dist.get_world_size()
    mesh = init_device_mesh("cpu", (cp_size,))
    batch, heads, seq, head_dim = 1, 4, 8, 2
    seen = {}

    def recording_attention(module, query, key, value, attention_mask, **kwargs):
        seen.update(kwargs, attention_mask=attention_mask)
        return torch.zeros(batch, seq, heads // cp_size, head_dim), None

    local = [torch.randn(batch, heads, seq // cp_size, head_dim) for _ in range(3)]
    mask = torch.ones(batch, 1, seq, seq).tril()
    position_ids, s_aux = torch.arange(seq).unsqueeze(0), torch.randn(heads)
    UlyssesAttention().attention(
        mesh, recording_attention, _Attention(), *local, mask, position_ids=position_ids, s_aux=s_aux, dropout=0.0
    )

    assert seen["attention_mask"] is mask and seen["position_ids"] is position_ids and seen["dropout"] == 0.0
    # attention sinks: one logit per head, so this rank's share of the heads
    torch.testing.assert_close(seen["s_aux"], s_aux.chunk(cp_size)[rank])


class TestContextParallelAttention(unittest.TestCase):
    """The attention style swaps the module's entry of `ALL_ATTENTION_FUNCTIONS` while the module runs."""

    def setUp(self):
        self.sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
        self.addCleanup(ALL_ATTENTION_FUNCTIONS.register, "sdpa", self.sdpa)
        self.spy, self.routed, self.plain = _SpyAttention(), _Attention(), _Attention()
        self.spy.install_forward(self.routed, _mesh())
        self.qkv = [torch.randn(2, 4, 8, 4) for _ in range(3)]

    def test_routes_the_installed_module_only(self):
        output, _ = self.routed(*self.qkv)
        torch.testing.assert_close(output, self.sdpa(self.routed, *self.qkv, None)[0])
        self.plain(*self.qkv)
        self.assertEqual(self.spy.seen, [self.routed])
        self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], self.sdpa)

    def test_runs_the_function_registered_at_call_time(self):
        # a hub kernel loaded after the model was built registers another function under the key
        kernel = mock.Mock(wraps=self.sdpa)
        ALL_ATTENTION_FUNCTIONS.register("sdpa", kernel)
        self.routed(*self.qkv)
        self.assertEqual(kernel.call_count, 1)
        self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], kernel)

    def test_restores_the_entry_when_the_attention_raises(self):
        failing = mock.Mock(side_effect=RuntimeError("kernel"))
        ALL_ATTENTION_FUNCTIONS.register("sdpa", failing)
        with self.assertRaisesRegex(RuntimeError, "kernel"):
            self.routed(*self.qkv)
        self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], failing)


@is_fsdp_test
class TestUlyssesAttention(unittest.TestCase):
    def test_matches_the_attention_on_the_whole_packed_sliding_rows(self):
        run_distributed(world_size=2, backend="gloo")(_test_ulysses_packed_sliding)()

    def test_passes_the_models_kwargs_and_its_share_of_the_attention_sinks(self):
        run_distributed(world_size=2, backend="gloo")(_test_ulysses_kwargs)()

    def _install(self, config, cp_size=2):
        UlyssesAttention().install_forward(MistralAttention(config, layer_idx=0), _mesh(cp_size))

    def test_refuses_attention_that_takes_no_mask(self):
        for attn_implementation in ("eager", "flex_attention"):
            with self.assertRaisesRegex(ValueError, "takes no attention mask"):
                self._install(_config(attn_implementation=attn_implementation))

    def test_refuses_heads_the_ranks_cannot_split(self):
        # 6 heads, then 8 heads with 2 key/value heads, over 4 ranks
        for heads, kv_heads in ((6, 6), (8, 2)):
            with self.assertRaisesRegex(ValueError, "must divide both the number of attention heads"):
                self._install(
                    _config(hidden_size=8 * heads, num_attention_heads=heads, num_key_value_heads=kv_heads), 4
                )


class TestSplitSequenceParallel(unittest.TestCase):
    def test_cuts_the_hidden_states_once_and_the_rotary_embeddings_everywhere(self):
        first, second = _Layer(), _Layer()
        SplitSequenceParallel().install_forward(first, _mesh(rank=1), split_hidden_states=True)
        SplitSequenceParallel().install_forward(second, _mesh(rank=1))
        hidden_states, cos, sin = (torch.randn(2, 8, 4) for _ in range(3))
        mask, position_ids = torch.ones(2, 1, 8, 8, dtype=torch.bool).tril(), torch.arange(8).expand(2, -1)

        local = first(hidden_states, attention_mask=mask, position_ids=position_ids, position_embeddings=(cos, sin))
        torch.testing.assert_close(local, hidden_states.chunk(2, dim=1)[1])
        torch.testing.assert_close(first.seen["position_embeddings"], (cos.chunk(2, dim=1)[1], sin.chunk(2, dim=1)[1]))
        # the mask and the positions stay those of the whole sequence, which the attention style needs
        self.assertIs(first.seen["attention_mask"], mask)
        self.assertIs(first.seen["position_ids"], position_ids)
        # some models pass the hidden states by keyword
        torch.testing.assert_close(first(hidden_states=hidden_states, position_embeddings=cos), local)
        # a later layer gets this rank's slice already, and cuts the embeddings the model computed for all of it
        second(local, position_embeddings=cos)
        torch.testing.assert_close(second.seen["hidden_states"], local)
        torch.testing.assert_close(second.seen["position_embeddings"], cos.chunk(2, dim=1)[1])

    def test_refuses_a_sequence_the_ranks_cannot_split(self):
        layer = _Layer()
        SplitSequenceParallel().install_forward(layer, _mesh(), split_hidden_states=True)
        with self.assertRaisesRegex(ValueError, "must be divisible by `cp_size`"):
            layer(torch.randn(1, 7, 4))


class TestModelContextParallel(unittest.TestCase):
    """`cp_model` on a model's inputs and loss: no cache, padding to a multiple of `cp_size`, a slice of the loss."""

    def _inputs(self, rank=0, *args, **kwargs):
        model = LlamaForCausalLM(_config(pad_token_id=7))
        args, kwargs = ALL_CP_STYLES["cp_model"].transform_inputs_pre_forward(model, args, kwargs, _mesh(rank=rank))
        self.assertEqual(args, ())  # a positional input is passed on by name
        return kwargs

    def test_refuses_a_cache_and_turns_it_off(self):
        input_ids = torch.tensor([[1, 2, 3, 4]])
        with self.assertRaisesRegex(ValueError, "`past_key_values` are not allowed"):
            self._inputs(0, input_ids=input_ids, past_key_values=DynamicCache())
        kwargs = self._inputs(0, input_ids, use_cache=True)
        self.assertIs(kwargs["input_ids"], input_ids)
        self.assertIs(kwargs["use_cache"], False)

    def test_pads_position_ids_after_a_packed_sequence(self):
        kwargs = self._inputs(
            0,
            input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            position_ids=torch.tensor([[0, 1, 2, 0, 1]]),
        )
        torch.testing.assert_close(kwargs["position_ids"], torch.tensor([[0, 1, 2, 0, 1, 2]]))

    def test_gives_every_rank_its_share_of_the_loss(self):
        """The ranks add up to the loss of the whole sequences, so FSDP2's average gives the full-sequence gradient."""
        vocab_size, seq = 11, 8
        model, style = LlamaForCausalLM(_config(vocab_size=vocab_size)), ALL_CP_STYLES["cp_model"]
        generator = torch.Generator().manual_seed(0)
        logits = torch.randn(2, seq, vocab_size, generator=generator)
        labels = torch.randint(0, vocab_size, (2, seq), generator=generator)

        def rank_loss(rank, **kwargs):
            """This rank's part: the model's loss on its slice of the logits, over the token count of the whole rows."""
            inputs = {"input_ids": labels, "labels": labels, **kwargs}
            _, kwargs = style.transform_inputs_pre_forward(model, (), inputs, _mesh(rank=rank))
            return model.loss_function(
                logits.chunk(2, dim=1)[rank].contiguous(),
                labels,
                vocab_size,
                kwargs["num_items_in_batch"],
                shift_labels=kwargs["shift_labels"],
            )

        # the whole rows' token count divides the slices' sum, also when the caller passes its own count instead
        for num_items_in_batch in (None, 7):
            full = model.loss_function(logits, labels, vocab_size, num_items_in_batch=num_items_in_batch)
            shards = sum(rank_loss(rank, num_items_in_batch=num_items_in_batch) for rank in (0, 1))
            torch.testing.assert_close(shards, full)

        # a rank whose slice holds no label contributes nothing, and reports no NaN
        labels[:, :5] = -100
        torch.testing.assert_close(rank_loss(0), torch.zeros(()))
        torch.testing.assert_close(rank_loss(1), model.loss_function(logits, labels, vocab_size))
