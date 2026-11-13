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
import copy
import inspect
import types
import unittest
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from parameterized import parameterized
from torch.distributed.device_mesh import init_device_mesh

from transformers import (
    DynamicCache,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaModel,
    MistralConfig,
)
from transformers.distributed.context_parallel import (
    ALL_CP_STYLES,
    ContextParallelAttention,
    ModelContextParallel,
    SplitSequenceParallel,
    UlyssesAttention,
    apply_context_parallelism,
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.testing_utils import is_fsdp_test

from ..distributed_testing_utils import run_distributed


class _MeshStub:
    """A `cp` mesh that only has to report its size and the caller's rank."""

    def __init__(self, size, rank=0):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank


class _StandInAttention(torch.nn.Module):
    """The attributes the attention functions read from a model's attention module, and the call a model's makes."""

    def __init__(self, attn_implementation="sdpa", num_key_value_groups=1):
        super().__init__()
        self.config = LlamaConfig(hidden_size=16, num_attention_heads=4, attn_implementation=attn_implementation)
        self.is_causal = True
        self.num_key_value_groups = num_key_value_groups
        self.layer_idx = 0

    def forward(self, query, key, value, attention_mask=None, **kwargs):
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, None)
        return attention_interface(self, query, key, value, attention_mask, **kwargs)


class _StandInLayer(torch.nn.Module):
    """A decoder layer stand-in that records the inputs its forward receives."""

    def __init__(self):
        super().__init__()
        self.seen = {}

    def forward(self, hidden_states, attention_mask=None, position_ids=None, position_embeddings=None):
        self.seen = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
        }
        return hidden_states


class _SpyAttention(ContextParallelAttention):
    """An attention style that records the modules it is asked to run, then runs them normally."""

    def __init__(self):
        self.modules_seen = []

    def attention(self, cp_mesh, attention_function, module, query, key, value, attention_mask, **kwargs):
        self.modules_seen.append(module)
        return attention_function(module, query, key, value, attention_mask, **kwargs)


def _same_on_every_rank(*shape, seed=0):
    """A tensor every rank agrees on, from which each rank takes its own slice."""
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _init_cp_mesh(device_type):
    return init_device_mesh(device_type, (dist.get_world_size(),))


def _tiny_config(**kwargs):
    """A `LlamaConfig` small enough for CPU tests, in float32 with `sdpa` attention."""
    return LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=99,
        attn_implementation="sdpa",
        **kwargs,
    )


def _full_sequence_mask(config, batch, seq, hidden_size, mask_name):
    """The attention mask a model builds over the whole sequence."""
    inputs = torch.zeros(batch, seq, hidden_size)
    if mask_name == "causal":
        causal = torch.ones(seq, seq, dtype=torch.bool).tril()
        return causal.view(1, 1, seq, seq).expand(batch, 1, seq, seq)

    # two packed documents
    position_ids = torch.cat([torch.arange(seq // 2), torch.arange(seq // 2)]).expand(batch, -1)
    mask_function = create_sliding_window_causal_mask if mask_name == "sliding" else create_causal_mask
    return mask_function(config, inputs, None, None, position_ids=position_ids)


def _test_ulysses_attention(rank, heads, kv_heads, mask_name, cp_size):
    mesh = _init_cp_mesh("cpu")
    batch, seq, head_dim, hidden_size = 2, 8, 4, 16
    config = MistralConfig(
        hidden_size=hidden_size,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        sliding_window=4,
        attn_implementation="sdpa",
    )
    mask = _full_sequence_mask(config, batch, seq, hidden_size, mask_name)

    query = _same_on_every_rank(batch, heads, seq, head_dim, seed=0).requires_grad_(True)
    key = _same_on_every_rank(batch, kv_heads, seq, head_dim, seed=1).requires_grad_(True)
    value = _same_on_every_rank(batch, kv_heads, seq, head_dim, seed=2).requires_grad_(True)
    upstream = _same_on_every_rank(batch, seq, heads, head_dim, seed=3)
    module = _StandInAttention(num_key_value_groups=heads // kv_heads)

    # this rank holds a contiguous slice of the sequence with every head, and the mask of the whole sequence
    local = [t.detach().chunk(cp_size, dim=2)[rank].clone().requires_grad_(True) for t in (query, key, value)]
    output, attn_weights = UlyssesAttention().attention(mesh, sdpa_attention_forward, module, *local, mask)
    assert attn_weights is None
    (output * upstream.chunk(cp_size, dim=1)[rank]).sum().backward()

    reference, _ = sdpa_attention_forward(module, query, key, value, mask)
    (reference * upstream).sum().backward()

    torch.testing.assert_close(output, reference.chunk(cp_size, dim=1)[rank])
    for local_tensor, full_tensor in zip(local, (query, key, value)):
        torch.testing.assert_close(local_tensor.grad, full_tensor.grad.chunk(cp_size, dim=2)[rank])


def _test_attention_kwargs(rank, cp_size):
    """The mask, the positions and the other kwargs the model built for the whole sequence are passed unchanged."""
    mesh = _init_cp_mesh("cpu")
    batch, heads, seq, head_dim = 1, 4, 8, 2
    local = [torch.randn(batch, heads, seq // cp_size, head_dim) for _ in range(3)]
    mask = torch.ones(batch, 1, seq, seq, dtype=torch.bool).tril()
    # attention sinks of the full head list, and boundaries a collator computed over the whole sequence
    s_aux = torch.randn(heads)
    position_ids = torch.cat([torch.arange(seq // 2), torch.arange(seq // 2)]).unsqueeze(0)
    cu_seq_lens = torch.tensor([0, seq // 2, seq])
    seen = {}

    def recording_attention(module, query, key, value, attention_mask, **kwargs):
        seen.update(kwargs, attention_mask=attention_mask, qkv_shape=tuple(query.shape))
        return torch.zeros(batch, seq, heads // cp_size, head_dim), None

    output, _ = UlyssesAttention().attention(
        mesh,
        recording_attention,
        _StandInAttention(),
        *local,
        mask,
        s_aux=s_aux,
        position_ids=position_ids,
        cu_seq_lens_q=cu_seq_lens,
        max_length_q=seq // 2,
        dropout=0.0,
    )

    assert seen["qkv_shape"] == (batch, heads // cp_size, seq, head_dim)
    assert seen["attention_mask"] is mask
    assert seen["dropout"] == 0.0
    torch.testing.assert_close(seen["position_ids"], position_ids)
    torch.testing.assert_close(seen["cu_seq_lens_q"], cu_seq_lens)
    assert seen["max_length_q"] == seq // 2
    torch.testing.assert_close(seen["s_aux"], s_aux.chunk(cp_size)[rank])
    assert tuple(output.shape) == (batch, seq // cp_size, heads, head_dim)


def _test_context_parallel_loss(rank, cp_size):
    """Every rank reports the loss of the whole sequences, with the gradient of its own slice times `cp_size`."""
    mesh = _init_cp_mesh("cpu")
    style, module = ALL_CP_STYLES["cp_model"], types.SimpleNamespace(config=types.SimpleNamespace(pad_token_id=0))
    batch, seq, vocab_size, num_items_in_batch = 2, 8, 11, 7
    full_logits = _same_on_every_rank(batch, seq, vocab_size)
    labels = torch.randint(0, vocab_size, (batch, seq), generator=torch.Generator().manual_seed(1))
    labels[:, -2:] = -100

    def loss_function(local_logits, labels, **kwargs):
        """The model between the two transforms: its own loss, on its slice of the logits."""
        _, kwargs = style.transform_inputs_pre_forward(
            module, (), {"input_ids": labels, "labels": labels, **kwargs}, mesh
        )
        loss = ForCausalLMLoss(
            local_logits, labels, vocab_size, kwargs["num_items_in_batch"], shift_labels=kwargs["shift_labels"]
        )
        return style.transform_output_post_forward(module, types.SimpleNamespace(loss=loss), mesh).loss

    # this rank holds its slice of the logits, and the labels of the whole sequence
    local_logits = full_logits.chunk(cp_size, dim=1)[rank].detach().clone().requires_grad_(True)
    loss = loss_function(local_logits, labels)
    loss.backward()
    torch.testing.assert_close(loss, ForCausalLMLoss(full_logits, labels, vocab_size))

    full_logits.requires_grad_(True)
    ForCausalLMLoss(full_logits, labels, vocab_size).backward()
    torch.testing.assert_close(local_logits.grad, full_logits.grad.chunk(cp_size, dim=1)[rank] * cp_size)

    # a caller's `num_items_in_batch` scales the summed loss, and is not replaced by the token count
    local_logits = full_logits.chunk(cp_size, dim=1)[rank].detach().clone().requires_grad_(True)
    loss = loss_function(local_logits, labels, num_items_in_batch=num_items_in_batch)
    expected = ForCausalLMLoss(full_logits, labels, vocab_size, num_items_in_batch=num_items_in_batch)
    torch.testing.assert_close(loss, expected)

    # the first rank's slice of the shifted labels is all `-100`: it contributes nothing, and reports no NaN
    no_targets = labels.clone()
    no_targets[:, : seq // cp_size + 1] = -100
    loss = loss_function(local_logits, no_targets)
    torch.testing.assert_close(loss, ForCausalLMLoss(full_logits, no_targets, vocab_size))


def _install_default_plan(model, cp_mesh):
    """`apply_context_parallelism` with the plan a `base_model_cp_plan` of a decoder-only model would resolve to."""
    plan = {}
    for layer in range(model.config.num_hidden_layers):
        plan[f"model.layers.{layer}"] = "cp_split_sequence"
        plan[f"model.layers.{layer}.self_attn"] = "cp_ulysses"
    return apply_context_parallelism(model, cp_mesh, plan)


def _test_model_matches_unsharded(rank, cp_size, seq_len=8):
    """A model with the plan installed, against the same weights on the full sequences."""
    mesh = _init_cp_mesh("cpu")
    torch.manual_seed(0)
    reference = LlamaForCausalLM(_tiny_config())
    model = _install_default_plan(copy.deepcopy(reference), mesh)

    input_ids = torch.randint(0, model.config.vocab_size, (2, seq_len), generator=torch.Generator().manual_seed(1))
    labels = input_ids.clone()
    reference_output = reference(input_ids=input_ids, labels=labels, use_cache=False)
    output = model(input_ids=input_ids, labels=labels)
    output.loss.backward()
    reference_output.loss.backward()

    # this rank computed the loss of the whole sequences, and its own slice of the logits of the padded sequence: the
    # real positions of the slice match the unsharded run, the padding at its end has no counterpart there
    local_seq = output.logits.shape[1]
    start = rank * local_seq
    real = max(0, min(local_seq, seq_len - start))
    torch.testing.assert_close(
        output.logits[:, :real], reference_output.logits[:, start : start + real], rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(output.loss, reference_output.loss, rtol=1e-5, atol=1e-5)
    assert output.past_key_values is None

    # FSDP2 averages the gradients over the cp ranks
    for (name, parameter), (reference_name, reference_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == reference_name
        gradient = parameter.grad.clone()
        dist.all_reduce(gradient, group=mesh.get_group())
        torch.testing.assert_close(gradient / cp_size, reference_parameter.grad, rtol=1e-5, atol=1e-5)


def _test_forward_refusals(rank, cp_size):
    """A cache raises instead of being silently recomputed wrong."""
    mesh = _init_cp_mesh("cpu")
    model = _install_default_plan(LlamaForCausalLM(_tiny_config()), mesh)

    input_ids = torch.randint(0, model.config.vocab_size, (2, 8), generator=torch.Generator().manual_seed(1))
    with pytest.raises(ValueError, match="`past_key_values` are not allowed"):
        model(input_ids=input_ids, past_key_values=DynamicCache())


@is_fsdp_test
class TestUlyssesDistributed(unittest.TestCase):
    """Two CPU ranks, gloo: the attention against plain attention on the full sequence."""

    @parameterized.expand(
        [
            (4, 4, "causal"),
            (4, 4, "packed"),
            (4, 4, "sliding"),
            (4, 2, "causal"),
            (4, 2, "packed"),
            (4, 2, "sliding"),
        ]
    )
    def test_attention_matches_full_attention(self, heads, kv_heads, mask_name):
        run_distributed(world_size=2, backend="gloo")(_test_ulysses_attention)(heads, kv_heads, mask_name, 2)

    def test_attention_kwargs_are_passed_through(self):
        run_distributed(world_size=2, backend="gloo")(_test_attention_kwargs)(2)


@is_fsdp_test
class TestApplyContextParallelism(unittest.TestCase):
    """Two CPU ranks, gloo: a model with the plan installed by hand, against the same weights on the full sequences."""

    def test_loss_matches_full_sequence_loss(self):
        run_distributed(world_size=2, backend="gloo")(_test_context_parallel_loss)(2)

    def test_model_matches_unsharded(self):
        run_distributed(world_size=2, backend="gloo")(_test_model_matches_unsharded)(2)

    def test_model_matches_unsharded_with_padding(self):
        # 7 tokens over 2 ranks: the model pads to 8, and the last rank holds 3 real positions
        run_distributed(world_size=2, backend="gloo")(_test_model_matches_unsharded)(2, 7)

    def test_forward_refusals(self):
        run_distributed(world_size=2, backend="gloo")(_test_forward_refusals)(2)


class TestContextParallelAttentionRouting(unittest.TestCase):
    def setUp(self):
        self.original = ALL_ATTENTION_FUNCTIONS["sdpa"]
        self.addCleanup(ALL_ATTENTION_FUNCTIONS.register, "sdpa", self.original)

    def test_installed_modules_are_routed(self):
        spy, routed, plain = _SpyAttention(), _StandInAttention(), _StandInAttention()
        spy.install_forward(routed, _MeshStub(2))
        query, key, value = (torch.randn(2, 4, 8, 4) for _ in range(3))

        output, _ = routed(query, key, value)
        self.assertEqual(spy.modules_seen, [routed])
        torch.testing.assert_close(output, self.original(routed, query, key, value, None)[0])
        # the entry is only replaced while the module runs: a module the style was not installed on keeps plain attention
        self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], self.original)
        torch.testing.assert_close(plain(query, key, value)[0], self.original(plain, query, key, value, None)[0])
        self.assertEqual(spy.modules_seen, [routed])

    def test_runs_the_function_registered_at_call_time(self):
        # a hub kernel loaded after the model was built registers another function under the key
        spy, routed = _SpyAttention(), _StandInAttention()
        spy.install_forward(routed, _MeshStub(2))
        replacement = mock.Mock(wraps=self.original)
        ALL_ATTENTION_FUNCTIONS.register("sdpa", replacement)
        query, key, value = (torch.randn(2, 4, 8, 4) for _ in range(3))

        routed(query, key, value)
        self.assertEqual(replacement.call_count, 1)
        self.assertEqual(spy.modules_seen, [routed])
        self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], replacement)

    def test_restores_the_entry_when_the_attention_raises(self):
        spy, routed = _SpyAttention(), _StandInAttention()
        spy.install_forward(routed, _MeshStub(2))
        failing = mock.Mock(side_effect=RuntimeError("kernel"))
        ALL_ATTENTION_FUNCTIONS.register("sdpa", failing)
        with self.assertRaisesRegex(RuntimeError, "kernel"):
            routed(torch.randn(2, 4, 8, 4), torch.randn(2, 4, 8, 4), torch.randn(2, 4, 8, 4))
        self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], failing)


class TestSplitSequenceParallel(unittest.TestCase):
    def test_first_layer_cuts_hidden_states(self):
        mesh = _MeshStub(size=2, rank=1)
        first, second = _StandInLayer(), _StandInLayer()
        SplitSequenceParallel().install_forward(first, mesh, split_hidden_states=True)
        SplitSequenceParallel().install_forward(second, mesh)

        hidden_states = torch.randn(2, 8, 4, generator=torch.Generator().manual_seed(0))
        attention_mask = torch.ones(2, 1, 8, 8, dtype=torch.bool).tril()
        position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)
        cos, sin = (torch.randn(2, 8, 2, generator=torch.Generator().manual_seed(seed)) for seed in (1, 2))

        local = hidden_states.chunk(2, dim=1)[1]
        output = first(
            hidden_states, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=(cos, sin)
        )
        torch.testing.assert_close(first.seen["hidden_states"], local)
        torch.testing.assert_close(first.seen["position_embeddings"][0], cos.chunk(2, dim=-2)[1])
        torch.testing.assert_close(first.seen["position_embeddings"][1], sin.chunk(2, dim=-2)[1])
        torch.testing.assert_close(first.seen["attention_mask"], attention_mask)
        torch.testing.assert_close(first.seen["position_ids"], position_ids)
        # some models pass the hidden states by keyword
        first(hidden_states=hidden_states, position_embeddings=(cos, sin))
        torch.testing.assert_close(first.seen["hidden_states"], local)

        # a later layer gets this rank's slice already, and cuts the rotary embeddings the model computed for all of it
        second(output, position_embeddings=cos)
        torch.testing.assert_close(second.seen["hidden_states"], local)
        torch.testing.assert_close(second.seen["position_embeddings"], cos.chunk(2, dim=-2)[1])

    def test_refuses_a_sequence_that_is_not_divisible(self):
        mesh = _MeshStub(size=2)
        layer = _StandInLayer()
        SplitSequenceParallel().install_forward(layer, mesh, split_hidden_states=True)
        with self.assertRaisesRegex(ValueError, "must be divisible by `cp_size`"):
            layer(torch.randn(1, 7, 4), position_embeddings=torch.randn(1, 7, 2))

        # the same check on the rotary embeddings of a later layer
        layer = _StandInLayer()
        SplitSequenceParallel().install_forward(layer, mesh)
        with self.assertRaisesRegex(ValueError, "must be divisible by `cp_size`"):
            layer(torch.randn(1, 4, 4), position_embeddings=torch.randn(1, 7, 2))


class TestUlyssesCheck(unittest.TestCase):
    def _check(self, config, cp_size=2):
        model = torch.nn.Module()
        model.config = config
        UlyssesAttention().check(model, _MeshStub(cp_size))

    def test_refuses_attention_it_cannot_route(self):
        for attn_implementation in ("eager", "flex_attention"):
            with self.subTest(attn_implementation), self.assertRaisesRegex(ValueError, "takes no attention mask"):
                self._check(LlamaConfig(attn_implementation=attn_implementation))

    def test_refuses_indivisible_heads(self):
        # 6 attention heads over 4 ranks
        config = LlamaConfig(hidden_size=48, num_attention_heads=6, num_key_value_heads=6, attn_implementation="sdpa")
        with self.assertRaisesRegex(ValueError, "must divide both the number of attention heads"):
            self._check(config, cp_size=4)
        # 8 attention heads but only 2 key/value heads over 4 ranks
        config = LlamaConfig(hidden_size=32, num_attention_heads=8, num_key_value_heads=2, attn_implementation="sdpa")
        with self.assertRaisesRegex(ValueError, "must divide both the number of attention heads"):
            self._check(config, cp_size=4)

    def test_accepts_supported_config(self):
        self._check(LlamaConfig(attn_implementation="sdpa", num_attention_heads=8, num_key_value_heads=4))
        # the mask the model builds over the whole sequence already has the sliding window in it
        self._check(MistralConfig(attn_implementation="sdpa", sliding_window=5))
        # flash kernels build the sliding window themselves
        self._check(MistralConfig(attn_implementation="flash_attention_2", sliding_window=5))


class TestModelContextParallel(unittest.TestCase):
    def test_refuses_a_cache_and_turns_it_off(self):
        model = LlamaForCausalLM(_tiny_config())
        style, input_ids = ALL_CP_STYLES["cp_model"], torch.tensor([[1, 2, 3, 4]])
        with self.assertRaisesRegex(ValueError, "`past_key_values` are not allowed"):
            style.transform_inputs_pre_forward(
                model, (), {"input_ids": input_ids, "past_key_values": DynamicCache()}, _MeshStub(2)
            )
        args, kwargs = style.transform_inputs_pre_forward(model, (input_ids,), {"use_cache": True}, _MeshStub(2))
        self.assertEqual(args, ())
        self.assertIs(kwargs["input_ids"], input_ids)
        self.assertIs(kwargs["use_cache"], False)

    def test_gives_the_loss_this_ranks_shifted_labels(self):
        model, labels = LlamaForCausalLM(_tiny_config()), torch.tensor([[1, 2, 3, 4]])
        _, kwargs = ALL_CP_STYLES["cp_model"].transform_inputs_pre_forward(
            model, (), {"input_ids": labels, "labels": labels}, _MeshStub(2, rank=1)
        )
        # the labels stay whole, the loss gets the second half of their shift and the count of the whole sequence
        torch.testing.assert_close(kwargs["labels"], labels)
        torch.testing.assert_close(kwargs["shift_labels"], torch.tensor([[4, -100]]))
        self.assertEqual(kwargs["num_items_in_batch"], 3)

        # a caller's shifted labels and token count are kept: the shift is only sliced
        _, kwargs = ALL_CP_STYLES["cp_model"].transform_inputs_pre_forward(
            model,
            (),
            {
                "input_ids": labels,
                "labels": labels,
                "shift_labels": torch.tensor([[2, 3, -100, -100]]),
                "num_items_in_batch": 9,
            },
            _MeshStub(2, rank=0),
        )
        torch.testing.assert_close(kwargs["shift_labels"], torch.tensor([[2, 3]]))
        self.assertEqual(kwargs["num_items_in_batch"], 9)

    def test_keeps_the_forward_signature(self):
        # `Trainer` reads it to drop the dataset columns the model does not take
        model = _install_default_plan(LlamaForCausalLM(_tiny_config()), _MeshStub(2))
        parameters = inspect.signature(model.forward).parameters
        self.assertIn("input_ids", parameters)
        self.assertIn("labels", parameters)
        self.assertNotIn("args", parameters)


class TestContextParallelPadding(unittest.TestCase):
    """`ModelContextParallel` pads every sequence-shaped input to a multiple of `cp_size`."""

    def _prepare(self, module, args=(), **kwargs):
        return ALL_CP_STYLES["cp_model"].transform_inputs_pre_forward(module, args, kwargs, _MeshStub(2))

    def _module(self, pad_token_id=7):
        """A model stand-in: the config the padding reads, and the signature the positional inputs bind to."""

        def forward(input_ids=None, attention_mask=None, labels=None, **kwargs):
            pass

        return types.SimpleNamespace(config=types.SimpleNamespace(pad_token_id=pad_token_id), forward=forward)

    def test_pads_input_ids_labels_and_the_mask(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        kwargs = {
            "input_ids": input_ids,
            "position_ids": torch.tensor([[0, 1, 2, 0, 1]]),
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
        }
        _, padded = self._prepare(self._module(), **kwargs)

        # the id of the padding is the pad token, and the positions continue those of the last document
        torch.testing.assert_close(padded["input_ids"], torch.tensor([[1, 2, 3, 4, 5, 7]]))
        torch.testing.assert_close(padded["position_ids"], torch.tensor([[0, 1, 2, 0, 1, 2]]))
        torch.testing.assert_close(padded["attention_mask"], torch.tensor([[1, 1, 1, 1, 1, 0]]))
        torch.testing.assert_close(padded["labels"], torch.tensor([[1, 2, 3, 4, 5, -100]]))
        assert padded["use_cache"] is False

    def test_pads_positionally_and_without_a_pad_token(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        args, padded = self._prepare(self._module(pad_token_id=None), (input_ids,), labels=input_ids.clone())

        # no `pad_token_id` on the config: the padding token is 0, and a positional input is passed on by name
        self.assertEqual(args, ())
        torch.testing.assert_close(padded["input_ids"], torch.tensor([[1, 2, 3, 4, 5, 0]]))
        torch.testing.assert_close(padded["labels"], torch.tensor([[1, 2, 3, 4, 5, -100]]))

    def test_leaves_a_divisible_sequence_alone(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        _, padded = self._prepare(self._module(), input_ids=input_ids, labels=input_ids.clone())
        torch.testing.assert_close(padded["input_ids"], input_ids)
        assert "position_ids" not in padded


class TestContextParallelChecks(unittest.TestCase):
    def test_accepts_causal_lms(self):
        config = _tiny_config()
        config.layer_types = ["full_attention", "sliding_attention"]
        _install_default_plan(LlamaForCausalLM(config), _MeshStub(2))
        # a base model computes no loss of its own, so the caller's loss is theirs to shard
        model, plan = LlamaModel(_tiny_config()), {}
        for layer in range(model.config.num_hidden_layers):
            plan[f"layers.{layer}"], plan[f"layers.{layer}.self_attn"] = "cp_split_sequence", "cp_ulysses"
        apply_context_parallelism(model, _MeshStub(2), plan)

    def test_refusals(self):
        class LlamaWithOwnAttention(LlamaForCausalLM):
            _supports_attention_backend = False

        mixed_config = _tiny_config()
        mixed_config.layer_types = ["full_attention", "linear_attention"]
        for model, message in (
            (LlamaForSequenceClassification(_tiny_config()), "only supports the causal language modeling loss"),
            (LlamaWithOwnAttention(_tiny_config()), "`_supports_attention_backend`"),
            (LlamaForCausalLM(mixed_config), r"does not support the \['linear_attention'\] layers"),
        ):
            # refused before anything is installed: the plan is not even read
            with self.subTest(type(model).__name__), self.assertRaisesRegex(ValueError, message):
                apply_context_parallelism(model, _MeshStub(2), {})


class TestContextParallelStyles(unittest.TestCase):
    def test_styles_are_registered(self):
        self.assertIsInstance(ALL_CP_STYLES["cp_model"], ModelContextParallel)
        self.assertIsInstance(ALL_CP_STYLES["cp_split_sequence"], SplitSequenceParallel)
        self.assertIsInstance(ALL_CP_STYLES["cp_ulysses"], UlyssesAttention)
