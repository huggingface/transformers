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
import functools
import inspect
import unittest
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from parameterized import parameterized
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from transformers import (
    AutoModelForCausalLM,
    DynamicCache,
    Gemma3TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaModel,
    MistralConfig,
    Qwen3MoeConfig,
)
from transformers.distributed import DistributedConfig
from transformers.distributed.context_parallel import (
    ContextParallelAttention,
    ModelContextParallel,
    SplitSequenceParallel,
    UlyssesAttention,
    apply_context_parallelism,
    context_parallel_styles,
)
from transformers.distributed.parallel_utils import ALL_PARALLEL_STYLES
from transformers.distributed.utils import gather_full_state_dict
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.testing_utils import is_fsdp_test
from transformers.utils import is_torch_greater_or_equal

from ..distributed_testing_utils import run_distributed
from ..test_fsdp_mixin import _deterministic_init_model_dir, _distributed_tmpdir


class _MeshStub:
    """A `cp` mesh that only has to report its size and the caller's rank."""

    def __init__(self, size, rank=0):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank


class _StandInAttention(torch.nn.Module):
    """The attributes the attention functions read from a model's attention module."""

    def __init__(self, attn_implementation="sdpa", num_key_value_groups=1):
        super().__init__()
        self.config = LlamaConfig(hidden_size=16, num_attention_heads=4, attn_implementation=attn_implementation)
        self.is_causal = True
        self.num_key_value_groups = num_key_value_groups
        self.layer_idx = 0


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
        **{
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 99,
            "attn_implementation": "sdpa",
            **kwargs,
        }
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
    batch, seq, vocab_size, num_items_in_batch = 2, 8, 11, 7
    full_logits = _same_on_every_rank(batch, seq, vocab_size)
    labels = torch.randint(0, vocab_size, (batch, seq), generator=torch.Generator().manual_seed(1))
    labels[:, -2:] = -100
    loss_function = functools.partial(ModelContextParallel().loss, ForCausalLMLoss, mesh)

    # this rank holds its slice of the logits, and the labels of the whole sequence
    local_logits = full_logits.chunk(cp_size, dim=1)[rank].detach().clone().requires_grad_(True)
    loss = loss_function(local_logits, labels, vocab_size)
    loss.backward()
    torch.testing.assert_close(loss, ForCausalLMLoss(full_logits, labels, vocab_size))

    full_logits.requires_grad_(True)
    ForCausalLMLoss(full_logits, labels, vocab_size).backward()
    torch.testing.assert_close(local_logits.grad, full_logits.grad.chunk(cp_size, dim=1)[rank] * cp_size)

    # a caller's `num_items_in_batch` scales the summed loss, and is not replaced by the token count
    local_logits = full_logits.chunk(cp_size, dim=1)[rank].detach().clone().requires_grad_(True)
    loss = loss_function(local_logits, labels, vocab_size, num_items_in_batch=num_items_in_batch)
    expected = ForCausalLMLoss(full_logits, labels, vocab_size, num_items_in_batch=num_items_in_batch)
    torch.testing.assert_close(loss, expected)

    # the first rank's slice of the shifted labels is all `-100`: it contributes nothing, and reports no NaN
    no_targets = labels.clone()
    no_targets[:, : seq // cp_size + 1] = -100
    loss = loss_function(local_logits, no_targets, vocab_size)
    torch.testing.assert_close(loss, ForCausalLMLoss(full_logits, no_targets, vocab_size))


def _install_default_plan(model, cp_mesh):
    """`apply_context_parallelism` with the plan a `base_model_cp_plan` of a decoder-only model would resolve to."""
    plan = {}
    for layer in range(model.config.num_hidden_layers):
        plan[f"model.layers.{layer}"] = "cp_split_sequence"
        plan[f"model.layers.{layer}.self_attn"] = "cp_ulysses"
    return apply_context_parallelism(model, cp_mesh, plan)


def _test_model_matches_unsharded(rank, cp_size):
    """A model with the plan installed, against the same weights on the full sequences."""
    mesh = _init_cp_mesh("cpu")
    torch.manual_seed(0)
    reference = LlamaForCausalLM(_tiny_config())
    model = _install_default_plan(copy.deepcopy(reference), mesh)

    input_ids = torch.randint(0, model.config.vocab_size, (2, 8), generator=torch.Generator().manual_seed(1))
    labels = input_ids.clone()
    reference_output = reference(input_ids=input_ids, labels=labels, use_cache=False)
    output = model(input_ids=input_ids, labels=labels)
    output.loss.backward()
    reference_output.loss.backward()

    # this rank computed the loss of the whole sequences, and its own slice of the logits
    torch.testing.assert_close(
        output.logits, reference_output.logits.chunk(cp_size, dim=1)[rank], rtol=1e-5, atol=1e-5
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
    """A cache and a sequence the ranks cannot split raise instead of being silently recomputed wrong."""
    mesh = _init_cp_mesh("cpu")
    model = _install_default_plan(LlamaForCausalLM(_tiny_config()), mesh)

    input_ids = torch.randint(0, model.config.vocab_size, (2, 8), generator=torch.Generator().manual_seed(1))
    with pytest.raises(ValueError, match="`past_key_values` are not allowed"):
        model(input_ids=input_ids, past_key_values=DynamicCache())
    with pytest.raises(ValueError, match="must be divisible by `cp_size`"):
        model(input_ids=input_ids[:, :-1])


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

    def test_forward_refusals(self):
        run_distributed(world_size=2, backend="gloo")(_test_forward_refusals)(2)


class TestContextParallelAttentionRouting(unittest.TestCase):
    def setUp(self):
        self.original = ALL_ATTENTION_FUNCTIONS["sdpa"]
        self.addCleanup(ALL_ATTENTION_FUNCTIONS.register, "sdpa", self.original)

    def test_installed_modules_are_routed(self):
        spy, routed, plain = _SpyAttention(), _StandInAttention(), _StandInAttention()
        spy.install_forward(routed, _MeshStub(2))
        attention = ALL_ATTENTION_FUNCTIONS["sdpa"]
        query, key, value = (torch.randn(2, 4, 8, 4) for _ in range(3))

        output, _ = attention(routed, query, key, value, None)
        self.assertEqual(spy.modules_seen, [routed])
        torch.testing.assert_close(output, self.original(routed, query, key, value, None)[0])
        # a module the style was not installed on keeps plain attention
        torch.testing.assert_close(
            attention(plain, query, key, value, None)[0], self.original(plain, query, key, value, None)[0]
        )

        # installing again keeps the wrapper, and a function registered since under the key (a hub kernel, for one) is
        # wrapped in turn
        spy.install_forward(_StandInAttention(), _MeshStub(2))
        self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], attention)
        replacement = mock.Mock(wraps=self.original)
        ALL_ATTENTION_FUNCTIONS.register("sdpa", replacement)
        spy.install_forward(_StandInAttention(), _MeshStub(2))
        ALL_ATTENTION_FUNCTIONS["sdpa"](routed, query, key, value, None)
        self.assertEqual(replacement.call_count, 1)
        self.assertEqual(spy.modules_seen, [routed, routed])


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

    def test_head_divisibility_covers_tensor_parallelism(self):
        # tensor parallelism keeps `num_heads / tp_size` heads on this rank, and those are what cp splits
        def config_with_heads(num_heads, num_kv_heads):
            config = LlamaConfig(
                hidden_size=32,
                num_attention_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                attn_implementation="sdpa",
            )
            config.distributed_config = DistributedConfig(tp_size=2, cp_size=2)
            return config

        self._check(config_with_heads(8, 4), cp_size=2)
        # 1 key/value head left on each rank cannot be split across 2 cp ranks
        with self.assertRaisesRegex(ValueError, r"`cp_size \* tp_size` must divide"):
            self._check(config_with_heads(8, 2), cp_size=2)
        # so few attention heads left on each rank that 2 cp ranks cannot split them either
        with self.assertRaisesRegex(ValueError, r"`cp_size \* tp_size` must divide"):
            self._check(config_with_heads(8, 4), cp_size=4)

    def test_accepts_supported_config(self):
        self._check(LlamaConfig(attn_implementation="sdpa", num_attention_heads=8, num_key_value_heads=4))
        # the mask the model builds over the whole sequence already has the sliding window in it
        self._check(MistralConfig(attn_implementation="sdpa", sliding_window=5))
        # flash kernels build the sliding window themselves
        self._check(MistralConfig(attn_implementation="flash_attention_2", sliding_window=5))


class TestModelContextParallel(unittest.TestCase):
    def setUp(self):
        self.addCleanup(ALL_ATTENTION_FUNCTIONS.register, "sdpa", ALL_ATTENTION_FUNCTIONS["sdpa"])

    def test_refuses_a_cache_and_turns_it_off(self):
        model = LlamaForCausalLM(_tiny_config())
        style, input_ids = ModelContextParallel(), torch.tensor([[1, 2, 3, 4]])
        with self.assertRaisesRegex(ValueError, "`past_key_values` are not allowed"):
            style.transform_inputs_pre_forward(
                model, (), {"input_ids": input_ids, "past_key_values": DynamicCache()}, _MeshStub(2)
            )
        args, kwargs = style.transform_inputs_pre_forward(model, (input_ids,), {"use_cache": True}, _MeshStub(2))
        self.assertEqual(args, (input_ids,))
        self.assertIs(kwargs["use_cache"], False)

    def test_keeps_the_forward_signature(self):
        # `Trainer` reads it to drop the dataset columns the model does not take
        model = _install_default_plan(LlamaForCausalLM(_tiny_config()), _MeshStub(2))
        parameters = inspect.signature(model.forward).parameters
        self.assertIn("input_ids", parameters)
        self.assertIn("labels", parameters)
        self.assertNotIn("args", parameters)


class TestContextParallelChecks(unittest.TestCase):
    def test_accepts_causal_lms(self):
        config = _tiny_config()
        config.layer_types = ["full_attention", "sliding_attention"]
        ModelContextParallel().check(LlamaForCausalLM(config))
        # a base model computes no loss of its own, so the caller's loss is theirs to shard
        ModelContextParallel().check(LlamaModel(_tiny_config()))

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
            with self.subTest(type(model).__name__), self.assertRaisesRegex(ValueError, message):
                ModelContextParallel().check(model)


class TestContextParallelStyles(unittest.TestCase):
    def test_styles_are_registered(self):
        self.assertIsInstance(ALL_PARALLEL_STYLES["cp_split_sequence"], SplitSequenceParallel)
        self.assertIsInstance(ALL_PARALLEL_STYLES["cp_ulysses"], UlyssesAttention)
        self.assertEqual(context_parallel_styles(), ["cp_split_sequence", "cp_ulysses"])

    def test_a_registered_attention_style_is_a_context_parallel_style(self):
        with mock.patch.dict(ALL_PARALLEL_STYLES._global_mapping, {"cp_spy": _SpyAttention()}):
            self.assertIn("cp_spy", context_parallel_styles())


# =============================================================================
# A model sharded with context parallelism, against the same weights on full sequences
# =============================================================================

CP_RTOL = 1e-5
CP_ATOL = 1e-5
# The all-reduce plan of `Qwen3Moe`, which overrides the token-dispatch rules of its own `ep_plan`: the router masks the
# experts of the other ranks and the partial expert outputs are all-reduced.
ALL_REDUCE_EP_PLAN = {"model.layers.*.mlp.gate": "ep_router", "model.layers.*.mlp.experts": "moe_tp_experts"}


def _tiny_batch(config, batch_size, seq_len, packed=False, seed=0):
    """A batch every rank builds alike, with `batch_size` rows for each data parallel rank."""
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), generator=generator)
    if not packed:
        return input_ids, None
    # two documents back to back, which `position_ids` has to describe for the whole sequence
    position_ids = torch.cat([torch.arange(seq_len // 2), torch.arange(seq_len // 2)]).expand(batch_size, -1)
    return input_ids, position_ids


def _local(tensor):
    """The full value of a tensor FSDP2 holds as a `DTensor`, or `tensor` itself."""
    return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor


def _data_parallel_rank_and_group(model, distributed_config):
    """
    Which rows of the batch this rank holds, and the group whose FSDP2 gradient contributions add up to the batch.

    The cp ranks of a cp group hold the same rows: they are part of the averaging group but not of the row split.
    """
    mesh_manager = model._mesh_manager
    fsdp_size, dp_replicate_size = distributed_config.fsdp_size, distributed_config.dp_replicate_size
    cp_size = distributed_config.cp_size
    dp_rank = 0
    if dp_replicate_size > 1:
        dp_rank = mesh_manager.get_mesh("dp_replicate").get_local_rank() * fsdp_size
    if fsdp_size > 1:
        dp_rank += mesh_manager.get_mesh("fsdp").get_local_rank()

    group_size = dp_replicate_size * fsdp_size
    if distributed_config.cp_size > 1:
        dims = ("dp_replicate", "fsdp_cp") if dp_replicate_size > 1 else "fsdp_cp"
        mesh = mesh_manager.get_mesh(dims)
        return dp_rank, [mesh.get_group(dim) for dim in mesh.mesh_dim_names], group_size * cp_size

    if group_size == 1:
        return dp_rank, [], 1
    mesh = mesh_manager.get_mesh("dp_replicate" if dp_replicate_size > 1 else "fsdp")
    return dp_rank, [mesh.get_group(dim) for dim in mesh.mesh_dim_names], group_size


def _average_over_group(tensor, groups, group_size):
    """Undo FSDP2's gradient averaging so the value covers the whole batch."""
    for group in groups:
        dist.all_reduce(tensor, group=group)
    return tensor / group_size


def _compare_with_unsharded(init_dir, batch, distributed_config, keep=None):
    """
    Run the full `batch` through an unsharded model and through the same weights sharded with context parallelism,
    then compare the loss, the logits and every gradient.

    `keep` masks the logits of the positions the comparison is about, e.g. the ones padding does not cover.
    """
    tokens = batch["input_ids"] if batch.get("input_ids") is not None else batch["inputs_embeds"]

    reference = AutoModelForCausalLM.from_pretrained(init_dir)
    reference_output = reference(**batch)
    reference_output.loss.backward()
    reference_grads = {
        name: param.grad.clone() for name, param in reference.named_parameters() if param.grad is not None
    }
    reference_loss, reference_logits = reference_output.loss.detach(), reference_output.logits.detach()
    del reference

    model = AutoModelForCausalLM.from_pretrained(init_dir, distributed_config=distributed_config)
    rows_per_rank = tokens.shape[0] // (distributed_config.fsdp_size * distributed_config.dp_replicate_size)
    dp_rank, groups, group_size = _data_parallel_rank_and_group(model, distributed_config)
    rows = slice(dp_rank * rows_per_rank, (dp_rank + 1) * rows_per_rank)
    # every entry is a full-sequence tensor on its first, batch dimension, apart from the flags
    local_batch = {name: value[rows] if isinstance(value, torch.Tensor) else value for name, value in batch.items()}

    output = model(**local_batch)
    output.loss.backward()

    # every rank computed the loss of the full sequences, and its own slice of the logits
    cp_rank = model._mesh_manager.get_mesh("cp").get_local_rank()
    expected_logits = reference_logits[rows].chunk(distributed_config.cp_size, dim=1)[cp_rank]
    logits = _local(output.logits)
    if keep is not None:
        expected_logits, logits = expected_logits * keep, logits * keep
    torch.testing.assert_close(logits, expected_logits, rtol=CP_RTOL, atol=CP_ATOL, msg="Logits mismatch")

    loss = _average_over_group(output.loss.detach(), groups, group_size)
    torch.testing.assert_close(loss, reference_loss, rtol=CP_RTOL, atol=CP_ATOL, msg="Loss mismatch")

    for name, param in model.named_parameters():
        grad = None if param.grad is None else _local(param.grad).float()
        expected = reference_grads.get(name)
        if grad is None or expected is None:
            assert grad is None and expected is None, f"Only one of the two runs has a gradient for {name}"
            continue
        grad = _average_over_group(grad, groups, group_size)
        torch.testing.assert_close(grad, expected, rtol=CP_RTOL, atol=CP_ATOL, msg=f"Gradient mismatch for {name}")
    return model


def _test_matches_unsharded_with_tp(rank, tp_size, cp_size, fsdp_size, packed):
    """The tp ranks hold the same batch and the same cp slice, and split the heads before cp splits them again."""
    # 4 attention heads and 4 key/value heads, both divisible by `tp_size * cp_size` = 4
    config = _tiny_config(num_key_value_heads=4)
    batch_size, seq_len = 2, 8
    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        input_ids, position_ids = _tiny_batch(config, batch_size * fsdp_size, seq_len, packed)
        distributed_config = DistributedConfig(tp_size=tp_size, cp_size=cp_size, fsdp_size=fsdp_size)
        batch = {"input_ids": input_ids, "labels": input_ids.clone(), "use_cache": False}
        if position_ids is not None:
            batch["position_ids"] = position_ids
        _compare_with_unsharded(init_dir, batch, distributed_config)
        dist.barrier()


def _test_tp_and_cp_compose(rank, tp_size, cp_size):
    """The tp plan shards the projections of an attention layer, the cp plan registers its attention call."""
    config = _tiny_config(num_key_value_heads=4)
    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        distributed_config = DistributedConfig(tp_size=tp_size, cp_size=cp_size)
        model = AutoModelForCausalLM.from_pretrained(init_dir, distributed_config=distributed_config)
        attention = model.model.layers[0].self_attn
        # tensor parallelism sharded the projections of the very module that context parallelism registered
        assert isinstance(attention.q_proj.weight, DTensor), type(attention.q_proj.weight)
        assert attention.cp_attention.func == ALL_PARALLEL_STYLES["cp_ulysses"].attention

        # and one forward goes through both: the logits of this rank's slice match the unsharded run
        input_ids, _ = _tiny_batch(config, 2, 8)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
        reference = AutoModelForCausalLM.from_pretrained(init_dir)
        with torch.no_grad():
            expected = reference(input_ids=input_ids).logits
        cp_rank = model._mesh_manager.get_mesh("cp").get_local_rank()
        torch.testing.assert_close(logits, expected.chunk(cp_size, dim=1)[cp_rank], rtol=CP_RTOL, atol=CP_ATOL)
        dist.barrier()


def _test_matches_unsharded(rank, cp_size, fsdp_size, dp_replicate_size, packed):
    config = _tiny_config()
    batch_size, seq_len = 2, 8
    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        input_ids, position_ids = _tiny_batch(config, batch_size * fsdp_size * dp_replicate_size, seq_len, packed)
        distributed_config = DistributedConfig(
            cp_size=cp_size, fsdp_size=fsdp_size, dp_replicate_size=dp_replicate_size
        )
        batch = {"input_ids": input_ids, "labels": input_ids.clone(), "use_cache": False}
        if position_ids is not None:
            batch["position_ids"] = position_ids
        _compare_with_unsharded(init_dir, batch, distributed_config)
        # `init_dir` is rank 0's temporary directory: wait for the other ranks before it is removed
        dist.barrier()


def _test_padded(rank, cp_size, config=None, packed=False):
    """Left and right padding through a 2D `attention_mask`; the padded positions carry no label."""
    config = config or _tiny_config()
    batch_size, seq_len = 2, 8
    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        input_ids, position_ids = _tiny_batch(config, batch_size, seq_len, packed)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[0, -3:] = 0  # right padding
        attention_mask[1, :2] = 0  # left padding
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.masked_fill(attention_mask == 0, -100),
            "use_cache": False,
        }
        if position_ids is not None:
            batch["position_ids"] = position_ids
        # the logits of the padded positions are not part of what the model is trained on
        keep = (attention_mask.chunk(cp_size, dim=1)[rank] == 1)[..., None]
        _compare_with_unsharded(init_dir, batch, DistributedConfig(cp_size=cp_size), keep=keep)
        dist.barrier()


def _tiny_moe_config(**kwargs):
    """A `Qwen3MoeConfig` small enough for CPU tests, whose default `ep_plan` dispatches the tokens to the experts."""
    return Qwen3MoeConfig(
        **{
            "hidden_size": 32,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": 99,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "attn_implementation": "sdpa",
            **kwargs,
        }
    )


def _test_matches_unsharded_with_ep(rank, cp_size, fsdp_size, ep_size, tp_size=1, dp_replicate_size=1):
    """
    Token dispatch with context parallelism: every rank sends the tokens of its own slice to the owners of the experts
    they selected, and the experts' gradients reduce over the ranks that hold a copy of them.
    """
    config = _tiny_moe_config()
    batch_size, seq_len = 2, 8
    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        rows = batch_size * fsdp_size * dp_replicate_size
        input_ids, _ = _tiny_batch(config, rows, seq_len)
        distributed_config = DistributedConfig(
            cp_size=cp_size,
            fsdp_size=fsdp_size,
            ep_size=ep_size,
            tp_size=tp_size,
            dp_replicate_size=dp_replicate_size,
        )
        batch = {"input_ids": input_ids, "labels": input_ids.clone(), "use_cache": False}
        model = _compare_with_unsharded(init_dir, batch, distributed_config)
        assert model.model.layers[0].mlp.experts.num_experts == config.num_experts // ep_size
        dist.barrier()


def _test_matches_unsharded_with_all_reduce_ep(rank, cp_size, ep_size, tp_size):
    """
    All-reduce experts need every rank of an ep group to see the same tokens: with `ep_size == tp_size` that group is
    the tp group, whose ranks hold the same cp slice.
    """
    config = _tiny_moe_config()
    batch_size, seq_len = 2, 8
    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        input_ids, _ = _tiny_batch(config, batch_size, seq_len)
        distributed_config = DistributedConfig(
            cp_size=cp_size, ep_size=ep_size, tp_size=tp_size, ep_plan=ALL_REDUCE_EP_PLAN
        )
        batch = {"input_ids": input_ids, "labels": input_ids.clone(), "use_cache": False}
        _compare_with_unsharded(init_dir, batch, distributed_config)
        dist.barrier()


def _test_inputs_embeds(rank, cp_size):
    """`inputs_embeds` is sharded along its sequence dim, not along its last, hidden one."""
    config = _tiny_config()
    batch_size, seq_len = 2, 8
    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        input_ids, _ = _tiny_batch(config, batch_size, seq_len)
        reference = AutoModelForCausalLM.from_pretrained(init_dir)
        inputs_embeds = reference.get_input_embeddings()(input_ids).detach()
        del reference

        batch = {"inputs_embeds": inputs_embeds, "labels": input_ids.clone(), "use_cache": False}
        _compare_with_unsharded(init_dir, batch, DistributedConfig(cp_size=cp_size))
        dist.barrier()


class _CountingUlysses(UlyssesAttention):
    """An Ulysses style that counts the attention calls routed through it."""

    def __init__(self):
        self.attention_calls = 0

    def attention(self, *args, **kwargs):
        self.attention_calls += 1
        return super().attention(*args, **kwargs)


def _test_custom_attention_style(rank, cp_size):
    """A registered attention style runs the attention when the plan names it; an unknown style is refused."""
    config = _tiny_config()
    counting = _CountingUlysses()
    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        with mock.patch.dict(ALL_PARALLEL_STYLES._global_mapping, {"cp_counting": counting}):
            distributed_config = DistributedConfig(
                cp_size=cp_size, cp_plan={"model.layers.*.self_attn": "cp_counting"}
            )
            model = AutoModelForCausalLM.from_pretrained(init_dir, distributed_config=distributed_config)

        input_ids, _ = _tiny_batch(config, 2, 8)
        model(input_ids=input_ids, labels=input_ids, use_cache=False)
        # every attention module runs through the style, once per forward
        assert all(layer.self_attn.cp_attention.func == counting.attention for layer in model.model.layers)
        assert counting.attention_calls == len(model.model.layers), counting.attention_calls

        unknown = DistributedConfig(cp_size=cp_size, cp_plan={"model.layers.*.self_attn": "cp_missing"})
        with pytest.raises(ValueError, match="Unsupported `cp_plan` styles"):
            AutoModelForCausalLM.from_pretrained(init_dir, distributed_config=unknown)
        dist.barrier()


def _test_save_pretrained(rank, cp_size, tp_size=1):
    """Weights saved from a context parallel model reload without a distributed config, unchanged."""
    config = _tiny_config(num_key_value_heads=4 if tp_size > 1 else 2)
    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        reference = AutoModelForCausalLM.from_pretrained(init_dir)
        expected = {name: value.clone() for name, value in reference.state_dict().items()}
        del reference

        distributed_config = DistributedConfig(cp_size=cp_size, tp_size=tp_size)
        model = AutoModelForCausalLM.from_pretrained(init_dir, distributed_config=distributed_config)
        with _distributed_tmpdir(rank) as save_dir:
            model.save_pretrained(save_dir, is_main_process=(rank == 0))
            dist.barrier()
            reloaded = AutoModelForCausalLM.from_pretrained(save_dir)
            # rank 0 removes `save_dir` when it leaves this block
            dist.barrier()

        for name, value in reloaded.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0, msg=f"Weight mismatch for {name}")
        dist.barrier()


def _test_save_load_distributed_checkpoint(rank, cp_size):
    """A rank-local checkpoint written by a context parallel model loads back into the same sharding."""
    config = _tiny_config()
    distributed_config = DistributedConfig(cp_size=cp_size)
    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        model = AutoModelForCausalLM.from_pretrained(init_dir, distributed_config=distributed_config)
        before = gather_full_state_dict(model)
        with _distributed_tmpdir(rank) as save_dir:
            model.save_pretrained(save_dir, is_main_process=(rank == 0), distributed_checkpoint=True)
            dist.barrier()
            reloaded = AutoModelForCausalLM.from_pretrained(save_dir, distributed_config=distributed_config)
            dist.barrier()
        after = gather_full_state_dict(reloaded)

        for name, value in before.items():
            torch.testing.assert_close(value, after[name], rtol=0, atol=0, msg=f"Weight mismatch for {name}")
        dist.barrier()


@unittest.skipUnless(is_torch_greater_or_equal("2.10"), "Context parallelism requires torch>=2.10")
class TestContextParallelConfig(unittest.TestCase):
    def test_refuses_pipeline_parallelism(self):
        with self.assertRaisesRegex(ValueError, "not supported yet with pipeline parallelism"):
            DistributedConfig(cp_size=2, pp_size=2)

    def test_refuses_a_model_without_a_cp_plan(self):
        config = _tiny_config()
        config.base_model_cp_plan = None
        model = LlamaForCausalLM(config)
        with self.assertRaisesRegex(ValueError, r"`cp_size=2` was requested, but `LlamaForCausalLM` has no `cp_plan`"):
            model._resolve_parallel_plans(DistributedConfig(cp_size=2))


@unittest.skipUnless(is_torch_greater_or_equal("2.10"), "Context parallelism requires torch>=2.10")
class TestContextParallelPlan(unittest.TestCase):
    def test_resolves_the_default_plan(self):
        model = LlamaForCausalLM(_tiny_config())
        _, _, _, cp_plan = model._resolve_parallel_plans(DistributedConfig(cp_size=2))
        expected = {}
        for layer in range(model.config.num_hidden_layers):
            expected[f"model.layers.{layer}"] = "cp_split_sequence"
            expected[f"model.layers.{layer}.self_attn"] = "cp_ulysses"
        self.assertEqual(cp_plan, expected)

    def test_refuses_a_plan_missing_a_style(self):
        for base_model_cp_plan, missing in (
            ({"layers.*": "cp_split_sequence"}, ["cp_ulysses"]),
            ({"layers.*.self_attn": "cp_ulysses"}, ["cp_split_sequence"]),
        ):
            config = _tiny_config()
            config.base_model_cp_plan = base_model_cp_plan
            model = LlamaForCausalLM(config)
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, rf"has no `{missing[0]}` rule"):
                model._resolve_parallel_plans(DistributedConfig(cp_size=2))


@is_fsdp_test
class TestContextParallelModel(unittest.TestCase):
    """A model sharded with context parallelism reproduces the same run on the full sequences."""

    def setUp(self):
        if torch._C._get_accelerator().type == "mps":
            self.skipTest("Distributed meshes are built on the default accelerator, which MPS cannot back")
        if not is_torch_greater_or_equal("2.10"):
            self.skipTest("Context parallelism requires torch>=2.10")

    def test_matches_unsharded(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                run_distributed(world_size=2, backend="gloo")(_test_matches_unsharded)(2, 1, 1, packed)

    def test_matches_unsharded_with_fsdp(self):
        run_distributed(world_size=4, backend="gloo")(_test_matches_unsharded)(2, 2, 1, False)

    def test_matches_unsharded_with_hsdp(self):
        run_distributed(world_size=4, backend="gloo")(_test_matches_unsharded)(2, 1, 2, False)

    def test_padded(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                run_distributed(world_size=2, backend="gloo")(_test_padded)(2, None, packed)

    def test_sliding_window(self):
        config = MistralConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=99,
            sliding_window=4,
            attn_implementation="sdpa",
        )
        for packed in (False, True):
            with self.subTest(packed=packed):
                run_distributed(world_size=2, backend="gloo")(_test_padded)(2, config, packed)

    def test_sliding_window_and_full_layers(self):
        config = Gemma3TextConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=99,
            sliding_window=4,
            sliding_window_pattern=2,
            attn_implementation="sdpa",
        )
        run_distributed(world_size=2, backend="gloo")(_test_padded)(2, config, False)

    def test_matches_unsharded_with_tensor_parallelism(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                run_distributed(world_size=4, backend="gloo")(_test_matches_unsharded_with_tp)(2, 2, 1, packed)

    def test_tensor_parallelism_and_context_parallelism_compose(self):
        run_distributed(world_size=4, backend="gloo")(_test_tp_and_cp_compose)(2, 2)

    def test_matches_unsharded_with_expert_parallelism(self):
        run_distributed(world_size=2, backend="gloo")(_test_matches_unsharded_with_ep)(2, 1, 2)

    def test_matches_unsharded_with_fsdp_and_expert_parallelism(self):
        # ep_size = fsdp_size * cp_size, so efsdp_size is 1 and every expert lives whole on a single rank
        run_distributed(world_size=4, backend="gloo")(_test_matches_unsharded_with_ep)(2, 2, 4)

    def test_matches_unsharded_with_expert_fsdp(self):
        # efsdp_size = fsdp_size * cp_size * tp_size / ep_size = 2: pairs of ranks hold the same experts, sharded
        # between them
        run_distributed(world_size=4, backend="gloo")(_test_matches_unsharded_with_ep)(2, 2, 2)

    def test_matches_unsharded_with_tensor_parallelism_and_expert_parallelism(self):
        # ep_size = 2 * tp_size puts both cp slices of the batch, two ranks each, in the same ep group
        run_distributed(world_size=4, backend="gloo")(_test_matches_unsharded_with_ep)(2, 1, 4, 2)

    def test_matches_unsharded_with_hsdp_and_expert_parallelism(self):
        # the expert gradient reduces over efsdp and then over the two replicas: dp_size = 2 * 1 * 2 ranks
        run_distributed(world_size=4, backend="gloo")(_test_matches_unsharded_with_ep)(2, 1, 2, 1, 2)

    def test_matches_unsharded_with_all_reduce_expert_parallelism(self):
        run_distributed(world_size=4, backend="gloo")(_test_matches_unsharded_with_all_reduce_ep)(2, 2, 2)

    def test_inputs_embeds(self):
        run_distributed(world_size=2, backend="gloo")(_test_inputs_embeds)(2)

    def test_custom_attention_style(self):
        run_distributed(world_size=2, backend="gloo")(_test_custom_attention_style)(2)

    def test_save_pretrained(self):
        run_distributed(world_size=2, backend="gloo")(_test_save_pretrained)(2)

    def test_save_pretrained_with_tensor_parallelism(self):
        run_distributed(world_size=4, backend="gloo")(_test_save_pretrained)(2, 2)

    def test_save_load_distributed_checkpoint(self):
        run_distributed(world_size=2, backend="gloo")(_test_save_load_distributed_checkpoint)(2)
