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
import types
import unittest
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from transformers import (
    AutoModelForCausalLM,
    DynamicCache,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaModel,
    MistralConfig,
)
from transformers.distributed import DistributedConfig, context_parallel
from transformers.distributed.context_parallel import (
    CPImplementation,
    UlyssesCP,
    _gather_heads_scatter_sequence,
    _gather_sequence_scatter_heads,
    cp_implementations,
    register_context_parallel_attention,
)
from transformers.distributed.utils import gather_full_state_dict
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import create_causal_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.testing_utils import require_flash_attn, require_torch_multi_accelerator

from ..distributed_testing_utils import run_distributed
from ..test_fsdp_mixin import _deterministic_init_model_dir, _distributed_tmpdir


class _MeshStub:
    """A `cp` mesh that only has to report its size."""

    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class _StandInAttention(torch.nn.Module):
    """The attributes the attention functions read from a model's attention module."""

    def __init__(self, attn_implementation="sdpa", num_key_value_groups=1):
        super().__init__()
        self.config = types.SimpleNamespace(_attn_implementation=attn_implementation)
        self.is_causal = True
        self.num_key_value_groups = num_key_value_groups
        self.layer_idx = 0


class _StandInModel(torch.nn.Module):
    def __init__(self, attn_implementation="sdpa", num_layers=2):
        super().__init__()
        self.config = types.SimpleNamespace(_attn_implementation=attn_implementation)
        self.layers = torch.nn.ModuleList([_StandInAttention(attn_implementation) for _ in range(num_layers)])


class _SpyImplementation(CPImplementation):
    """Records the modules it is asked to run, then runs them normally."""

    def __init__(self, cp_mesh):
        super().__init__(cp_mesh)
        self.modules_seen = []

    def attention(self, attention_function, module, query, key, value, attention_mask, **kwargs):
        self.modules_seen.append(module)
        return attention_function(module, query, key, value, attention_mask, **kwargs)


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


def _same_on_every_rank(*shape, seed=0):
    """A tensor every rank agrees on, from which each rank takes its own slice."""
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _init_cp_mesh(device_type):
    return init_device_mesh(device_type, (dist.get_world_size(),))


def _test_all_to_all(rank, cp_size):
    mesh = _init_cp_mesh("cpu")
    group = mesh.get_group()
    batch, heads, seq, head_dim = 2, 4, 8, 3
    full = _same_on_every_rank(batch, heads, seq, head_dim)

    local = full.chunk(cp_size, dim=2)[rank].contiguous()
    gathered = _gather_sequence_scatter_heads(local, group)
    assert tuple(gathered.shape) == (batch, heads // cp_size, seq, head_dim)
    torch.testing.assert_close(gathered, full.reshape(batch, cp_size, heads // cp_size, seq, head_dim)[:, rank])

    # around the attention call the output is in [batch, seq, heads, head_dim] layout, with the heads split by rank
    full_output = full.transpose(1, 2)
    heads_chunked = full_output.reshape(batch, seq, cp_size, heads // cp_size, head_dim)[:, :, rank].contiguous()
    gathered_output = _gather_heads_scatter_sequence(heads_chunked, group)
    assert tuple(gathered_output.shape) == (batch, seq // cp_size, heads, head_dim)
    torch.testing.assert_close(gathered_output, full_output.chunk(cp_size, dim=1)[rank])

    back = _gather_heads_scatter_sequence(gathered.transpose(1, 2).contiguous(), group)
    torch.testing.assert_close(back, local.transpose(1, 2))

    with pytest.raises(ValueError, match="divisible by `cp_size`"):
        _gather_sequence_scatter_heads(torch.randn(batch, 3, seq, head_dim), group)


def _test_all_to_all_gradients(rank, cp_size):
    mesh = _init_cp_mesh("cpu")
    group = mesh.get_group()
    batch, heads, seq, head_dim = 2, 4, 8, 3
    full = _same_on_every_rank(batch, heads, seq, head_dim)
    # each rank backpropagates its own upstream gradient through its own piece of the gathered sequence
    upstreams = [
        _same_on_every_rank(batch, heads // cp_size, seq, head_dim, seed=1 + other) for other in range(cp_size)
    ]

    local = full.chunk(cp_size, dim=2)[rank].clone().requires_grad_(True)
    _gather_sequence_scatter_heads(local, group).backward(upstreams[rank])

    reference = full.clone().requires_grad_(True)
    head_chunks = reference.reshape(batch, cp_size, heads // cp_size, seq, head_dim)
    (head_chunks * torch.stack(upstreams).permute(1, 0, 2, 3, 4)).sum().backward()
    torch.testing.assert_close(local.grad, reference.grad.chunk(cp_size, dim=2)[rank])


def _test_ulysses_attention(rank, heads, kv_heads, packed, cp_size):
    mesh = _init_cp_mesh("cpu")
    batch, seq, head_dim = 2, 8, 4
    positions = torch.arange(seq)
    if packed:
        positions = torch.cat([torch.arange(seq // 2), torch.arange(seq // 2)])
    position_ids = positions.unsqueeze(0)

    query = _same_on_every_rank(batch, heads, seq, head_dim, seed=0).requires_grad_(True)
    key = _same_on_every_rank(batch, kv_heads, seq, head_dim, seed=1).requires_grad_(True)
    value = _same_on_every_rank(batch, kv_heads, seq, head_dim, seed=2).requires_grad_(True)
    upstream = _same_on_every_rank(batch, seq, heads, head_dim, seed=3)
    module = _StandInAttention(num_key_value_groups=heads // kv_heads)

    # this rank's slice, and the causal mask the model built for it
    local = [
        tensor.detach().chunk(cp_size, dim=2)[rank].clone().requires_grad_(True) for tensor in (query, key, value)
    ]
    local_mask = torch.ones(seq // cp_size, seq // cp_size, dtype=torch.bool).tril()[None, None]
    implementation = UlyssesCP(mesh)
    implementation.position_ids = position_ids
    output, attn_weights = implementation.attention(sdpa_attention_forward, module, *local, local_mask)
    assert attn_weights is None
    (output * upstream.chunk(cp_size, dim=1)[rank]).sum().backward()

    # the same attention over the whole sequence, with the mask the model would build there
    config = LlamaConfig(
        hidden_size=16, num_attention_heads=heads, num_key_value_heads=kv_heads, attn_implementation="sdpa"
    )
    mask = create_causal_mask(config, torch.zeros(batch, seq, 16), None, None, position_ids=position_ids)
    reference, _ = sdpa_attention_forward(module, query, key, value, mask)
    (reference * upstream).sum().backward()

    torch.testing.assert_close(output, reference.chunk(cp_size, dim=1)[rank])
    for local_tensor, full_tensor in zip(local, (query, key, value)):
        torch.testing.assert_close(local_tensor.grad, full_tensor.grad.chunk(cp_size, dim=2)[rank])


def _test_flash_attention(rank, cp_size):
    from transformers.integrations.flash_attention import flash_attention_forward
    from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids

    mesh = _init_cp_mesh("cuda")
    batch, heads, kv_heads, seq, head_dim = 1, 4, 2, 16, 32
    device, dtype = torch.device("cuda", rank), torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(0)
    query, key, value = (
        torch.randn(shape, generator=generator, device=device, dtype=dtype)
        for shape in (
            (batch, heads, seq, head_dim),
            (batch, kv_heads, seq, head_dim),
            (batch, kv_heads, seq, head_dim),
        )
    )
    # two packed documents, so the kernel has to rebuild the sequence boundaries of the gathered sequence
    position_ids = torch.cat([torch.arange(seq // 2), torch.arange(seq // 2)]).unsqueeze(0).to(device)
    s_aux = torch.randn(heads, generator=generator, device=device, dtype=torch.float32)
    module = _StandInAttention("flash_attention_2", num_key_value_groups=heads // kv_heads)

    # a collator computed those for the local slice, which describes documents cut at the shard boundary
    stale = prepare_fa_kwargs_from_position_ids(position_ids.chunk(cp_size, dim=1)[rank].clone())
    (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = stale

    seen = {}

    def recording_flash_attention_forward(module, query, key, value, attention_mask, **kwargs):
        seen.update(kwargs)
        return flash_attention_forward(module, query, key, value, attention_mask, **kwargs)

    implementation = UlyssesCP(mesh)
    implementation.position_ids = position_ids
    local = [tensor.chunk(cp_size, dim=2)[rank].contiguous() for tensor in (query, key, value)]
    output, _ = implementation.attention(
        recording_flash_attention_forward,
        module,
        *local,
        None,
        cu_seq_lens_q=cu_seq_lens_q,
        cu_seq_lens_k=cu_seq_lens_k,
        max_length_q=max_length_q,
        max_length_k=max_length_k,
        s_aux=s_aux,
        dropout=0.0,
    )

    assert tuple(seen["position_ids"].shape) == (1, seq) and torch.equal(seen["position_ids"], position_ids)
    assert not {"cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"} & set(seen)
    torch.testing.assert_close(seen["s_aux"], s_aux.chunk(cp_size)[rank])

    reference, _ = flash_attention_forward(module, query, key, value, None, position_ids=position_ids)
    torch.testing.assert_close(output.float(), reference.chunk(cp_size, dim=1)[rank].float(), atol=1e-2, rtol=1e-2)


class TestUlyssesDistributed(unittest.TestCase):
    def test_all_to_all(self):
        run_distributed(world_size=2, backend="gloo")(_test_all_to_all)(2)

    def test_all_to_all_gradients(self):
        run_distributed(world_size=2, backend="gloo")(_test_all_to_all_gradients)(2)

    def test_attention_matches_full_attention(self):
        cases = ((4, 4, False), (4, 4, True), (4, 2, False))
        for heads, kv_heads, packed in cases:
            with self.subTest(heads=heads, kv_heads=kv_heads, packed=packed):
                run_distributed(world_size=2, backend="gloo")(_test_ulysses_attention)(heads, kv_heads, packed, 2)


class TestContextParallelRouting(unittest.TestCase):
    def test_registered_modules_are_routed(self):
        original = ALL_ATTENTION_FUNCTIONS["sdpa"]
        try:
            model, other_model = _StandInModel(), _StandInModel()
            unregistered = _StandInAttention()
            implementation = _SpyImplementation(_MeshStub(2))
            register_context_parallel_attention(model, implementation)
            attention = ALL_ATTENTION_FUNCTIONS["sdpa"]
            self.assertIsNot(attention, original)

            query, key, value = (torch.randn(2, 4, 8, 4) for _ in range(3))
            output, _ = attention(model.layers[0], query, key, value, None)
            self.assertEqual(implementation.modules_seen, [model.layers[0]])
            torch.testing.assert_close(output, original(model.layers[0], query, key, value, None)[0])

            # a module that is not part of a registered model keeps plain attention
            attention(unregistered, query, key, value, None)
            self.assertEqual(len(implementation.modules_seen), 1)
            torch.testing.assert_close(
                attention(unregistered, query, key, value, None)[0], original(unregistered, query, key, value, None)[0]
            )

            # a second model reuses the wrapper already registered for that attention implementation
            register_context_parallel_attention(other_model, implementation)
            self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], attention)
            attention(other_model.layers[1], query, key, value, None)
            self.assertEqual(implementation.modules_seen[-1], other_model.layers[1])

            # another function registered under the key afterwards (a hub kernel, for one) is wrapped in turn
            replacement_calls = []

            def replacement(module, *args, **kwargs):
                replacement_calls.append(module)
                return original(module, *args, **kwargs)

            ALL_ATTENTION_FUNCTIONS.register("sdpa", replacement)
            self.assertIs(context_parallel._install_context_parallel_attention("sdpa"), attention)
            self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], attention)
            ALL_ATTENTION_FUNCTIONS["sdpa"](model.layers[1], query, key, value, None)
            self.assertEqual(implementation.modules_seen[-1], model.layers[1])
            self.assertEqual(replacement_calls, [model.layers[1]])
        finally:
            ALL_ATTENTION_FUNCTIONS.register("sdpa", original)
            context_parallel._context_parallel_attention_functions.clear()

    def test_unregistered_attention_implementation(self):
        model = _StandInModel(attn_implementation="not_a_kernel")
        with self.assertRaisesRegex(ValueError, "is not registered in `ALL_ATTENTION_FUNCTIONS`"):
            register_context_parallel_attention(model, _SpyImplementation(_MeshStub(2)))


class TestUlyssesCheck(unittest.TestCase):
    def _check(self, config, cp_size=2):
        model = torch.nn.Module()
        model.config = config
        UlyssesCP(_MeshStub(cp_size)).check(model)

    def test_refuses_eager_attention(self):
        with self.assertRaisesRegex(ValueError, "takes the `position_ids` of the full sequence"):
            self._check(LlamaConfig(attn_implementation="eager"))

    def test_refuses_flex_attention(self):
        with self.assertRaisesRegex(ValueError, "takes the `position_ids` of the full sequence"):
            self._check(LlamaConfig(attn_implementation="flex_attention"))

    def test_refuses_paged_attention(self):
        with self.assertRaisesRegex(ValueError, "takes the `position_ids` of the full sequence"):
            self._check(LlamaConfig(attn_implementation="paged|eager"))

    def test_refuses_chunked_attention(self):
        config = LlamaConfig(attn_implementation="sdpa")
        config.layer_types = ["full_attention", "chunked_attention"]
        with self.assertRaisesRegex(ValueError, "does not support `chunked_attention` layers"):
            self._check(config)

    def test_refuses_sliding_window_with_sdpa(self):
        with self.assertRaisesRegex(ValueError, "does not support the sliding-window attention"):
            self._check(MistralConfig(attn_implementation="sdpa"))
        config = LlamaConfig(attn_implementation="sdpa", sliding_window=4)
        config.layer_types = ["sliding_attention", "full_attention"]
        with self.assertRaisesRegex(ValueError, "does not support the sliding-window attention"):
            self._check(config)

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
        # flash kernels apply the sliding window themselves
        self._check(MistralConfig(attn_implementation="flash_attention_2"))
        config = LlamaConfig(attn_implementation="sdpa")
        config.layer_types = ["full_attention", "full_attention"]
        self._check(config)


class TestContextParallelChecks(unittest.TestCase):
    def _check(self, model):
        context_parallel._check_context_parallelism(model)

    def test_accepts_causal_lm_and_base_model(self):
        self._check(LlamaForCausalLM(_tiny_config()))
        # A base model computes no loss of its own, so the caller's loss is theirs to shard.
        self._check(LlamaModel(_tiny_config()))

        # a head whose name maps to no loss uses the causal LM loss
        class LlamaLMHeadModel(LlamaForCausalLM):
            pass

        self._check(LlamaLMHeadModel(_tiny_config()))

    def test_accepts_softmax_attention_layer_types(self):
        config = _tiny_config()
        config.layer_types = ["full_attention", "sliding_attention", "chunked_attention"]
        self._check(LlamaForCausalLM(config))

    def test_refuses_other_heads(self):
        with self.assertRaisesRegex(ValueError, "only supports the causal language modeling loss"):
            self._check(LlamaForSequenceClassification(_tiny_config()))

    def test_refuses_attention_outside_the_interface(self):
        class LlamaWithOwnAttention(LlamaForCausalLM):
            _supports_attention_backend = False

        with self.assertRaisesRegex(ValueError, "`_supports_attention_backend`"):
            self._check(LlamaWithOwnAttention(_tiny_config()))

    def test_refuses_stateful_models(self):
        class StatefulLlama(LlamaForCausalLM):
            _is_stateful = True

        with self.assertRaisesRegex(ValueError, "carry a state along the sequence"):
            self._check(StatefulLlama(_tiny_config()))

    def test_refuses_layers_that_mix_tokens_outside_attention(self):
        config = _tiny_config()
        config.layer_types = ["full_attention", "linear_attention"]
        with self.assertRaisesRegex(ValueError, r"does not support the \['linear_attention'\] layers"):
            self._check(LlamaForCausalLM(config))


@require_flash_attn
@require_torch_multi_accelerator
class TestUlyssesFlashAttention(unittest.TestCase):
    def test_packed_flash_attention(self):
        run_distributed(world_size=2, backend="nccl")(_test_flash_attention)(2)


# =============================================================================
# A model sharded with context parallelism, against the same weights on full sequences
# =============================================================================

CP_RTOL = 1e-5
CP_ATOL = 1e-5


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


def _data_parallel_rank_and_group(model, fsdp_size, dp_replicate_size):
    """
    Which rows of the batch this rank holds, and the group FSDP2 averages its gradients over.

    The tests use one data parallel axis at a time, so that axis is the group.
    """
    if fsdp_size == 1 and dp_replicate_size == 1:
        return 0, None
    mesh_manager = model._mesh_manager
    dp_dim = "dp_replicate" if dp_replicate_size > 1 else "fsdp"
    dp_rank = mesh_manager.get_mesh("dp_replicate").get_local_rank() * fsdp_size
    dp_rank += mesh_manager.get_mesh("fsdp").get_local_rank()
    return dp_rank, mesh_manager.get_mesh(dp_dim).get_group()


def _average_over_data_parallel(tensor, dp_group, dp_size):
    """Undo FSDP2's gradient averaging so the value covers the whole batch."""
    if dp_group is None:
        return tensor
    dist.all_reduce(tensor, group=dp_group)
    return tensor / dp_size


def _compare_with_unsharded(init_dir, batch, distributed_config):
    """
    Run the full `batch` through an unsharded model and through the same weights sharded with context parallelism,
    then compare the loss, the logits and every gradient.
    """
    fsdp_size, dp_replicate_size = distributed_config.fsdp_size, distributed_config.dp_replicate_size
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
    batch_size = tokens.shape[0] // (fsdp_size * dp_replicate_size)
    dp_rank, dp_group = _data_parallel_rank_and_group(model, fsdp_size, dp_replicate_size)
    rows = slice(dp_rank * batch_size, (dp_rank + 1) * batch_size)
    # every entry is a full-sequence tensor on its first, batch dimension, apart from the flags
    local_batch = {name: value[rows] if isinstance(value, torch.Tensor) else value for name, value in batch.items()}

    output = model(**local_batch)
    output.loss.backward()

    # every rank computed the loss of the full sequences, and its own slice of the logits
    cp_rank = model._mesh_manager.get_mesh("cp").get_local_rank()
    expected_logits = reference_logits[rows].chunk(distributed_config.cp_size, dim=1)[cp_rank]
    torch.testing.assert_close(
        _local(output.logits), expected_logits, rtol=CP_RTOL, atol=CP_ATOL, msg="Logits mismatch"
    )
    loss = _average_over_data_parallel(output.loss.detach(), dp_group, fsdp_size * dp_replicate_size)
    torch.testing.assert_close(loss, reference_loss, rtol=CP_RTOL, atol=CP_ATOL, msg="Loss mismatch")

    for name, param in model.named_parameters():
        grad = None if param.grad is None else _local(param.grad).float()
        expected = reference_grads.get(name)
        if grad is None or expected is None:
            assert grad is None and expected is None, f"Only one of the two runs has a gradient for {name}"
            continue
        grad = _average_over_data_parallel(grad, dp_group, fsdp_size * dp_replicate_size)
        torch.testing.assert_close(grad, expected, rtol=CP_RTOL, atol=CP_ATOL, msg=f"Gradient mismatch for {name}")


def _test_matches_unsharded(rank, cp_size, fsdp_size, dp_replicate_size, packed):
    config = _tiny_config()
    batch_size, seq_len = 2, 8
    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        rows = batch_size * fsdp_size * dp_replicate_size
        input_ids, position_ids = _tiny_batch(config, rows, seq_len, packed)
        distributed_config = DistributedConfig(
            cp_size=cp_size, fsdp_size=fsdp_size, dp_replicate_size=dp_replicate_size
        )
        batch = {"input_ids": input_ids, "labels": input_ids.clone(), "use_cache": False}
        if position_ids is not None:
            batch["position_ids"] = position_ids
        _compare_with_unsharded(init_dir, batch, distributed_config)
        # `init_dir` is rank 0's temporary directory: wait for the other ranks before it is removed
        dist.barrier()


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


def _test_ignored_labels(rank, cp_size):
    """`-100` on every shifted label of the first rank's slice: it contributes no loss, the rest is unchanged."""
    config = _tiny_config()
    batch_size, seq_len = 2, 8
    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        input_ids, _ = _tiny_batch(config, batch_size, seq_len)
        labels = input_ids.clone()
        # one past the slice: the shift pulls in the next token, which would be this rank's only valid target
        labels[:, : seq_len // cp_size + 1] = -100
        _compare_with_unsharded(
            init_dir,
            {"input_ids": input_ids, "labels": labels, "use_cache": False},
            DistributedConfig(cp_size=cp_size),
        )
        dist.barrier()


def _test_keeps_num_items_in_batch(rank, cp_size):
    """A caller's `num_items_in_batch` scales the summed loss, and is not replaced by the token count."""
    config = _tiny_config()
    batch_size, seq_len, num_items_in_batch = 2, 8, 7
    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        input_ids, _ = _tiny_batch(config, batch_size, seq_len)
        labels = input_ids.clone()

        reference = AutoModelForCausalLM.from_pretrained(init_dir)
        token_sum = reference(input_ids=input_ids, labels=labels, num_items_in_batch=1, use_cache=False).loss
        del reference

        model = AutoModelForCausalLM.from_pretrained(init_dir, distributed_config=DistributedConfig(cp_size=cp_size))
        batch = {"input_ids": input_ids, "labels": labels, "num_items_in_batch": num_items_in_batch}
        loss = model(**batch, use_cache=False).loss
        expected = token_sum.detach() / num_items_in_batch
        torch.testing.assert_close(loss.detach(), expected, rtol=CP_RTOL, atol=CP_ATOL, msg="Loss mismatch")
        dist.barrier()


class _CountingUlyssesCP(UlyssesCP):
    """A `UlyssesCP` that counts the attention calls routed through it."""

    def __init__(self, cp_mesh):
        super().__init__(cp_mesh)
        self.attention_calls = 0

    def attention(self, *args, **kwargs):
        self.attention_calls += 1
        return super().attention(*args, **kwargs)


def _test_custom_implementation(rank, cp_size):
    """A registered implementation runs the attention, and an unknown name is refused before anything is sharded."""
    config = _tiny_config()
    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        with mock.patch.dict(cp_implementations, {"counting": _CountingUlyssesCP}):
            distributed_config = DistributedConfig(cp_size=cp_size, cp_implementation="counting")
            model = AutoModelForCausalLM.from_pretrained(init_dir, distributed_config=distributed_config)

        input_ids, _ = _tiny_batch(config, 2, 8)
        model(input_ids=input_ids, labels=input_ids, use_cache=False)
        implementation = ALL_ATTENTION_FUNCTIONS["sdpa"].implementations[model.model.layers[0].self_attn]
        assert isinstance(implementation, _CountingUlyssesCP), type(implementation)
        assert implementation.attention_calls == config.num_hidden_layers, implementation.attention_calls

        # another function registered under the attention key since is wrapped again by the next forward
        wrapper = ALL_ATTENTION_FUNCTIONS["sdpa"]
        try:
            ALL_ATTENTION_FUNCTIONS.register("sdpa", sdpa_attention_forward)
            model(input_ids=input_ids, labels=input_ids, use_cache=False)
        finally:
            ALL_ATTENTION_FUNCTIONS.register("sdpa", wrapper)
        assert implementation.attention_calls == 2 * config.num_hidden_layers, implementation.attention_calls

        unknown = DistributedConfig(cp_size=cp_size, cp_implementation="missing")
        with pytest.raises(ValueError, match="Unknown `cp_implementation=missing`"):
            AutoModelForCausalLM.from_pretrained(init_dir, distributed_config=unknown)
        dist.barrier()


def _test_forward_refusals(rank, cp_size):
    """Inputs that context parallelism cannot shard correctly raise instead of being silently recomputed wrong."""
    config = _tiny_config()
    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        model = AutoModelForCausalLM.from_pretrained(init_dir, distributed_config=DistributedConfig(cp_size=cp_size))
        input_ids, _ = _tiny_batch(config, 2, 8)

        with pytest.raises(ValueError, match="does not support attention masks"):
            model(input_ids=input_ids, attention_mask=torch.tensor([[1] * 8, [1] * 7 + [0]]))

        with pytest.raises(ValueError, match="must be divisible by `cp_size`"):
            model(input_ids=input_ids[:, :-1])

        with pytest.raises(ValueError, match="`past_key_values` are not allowed"):
            model(input_ids=input_ids, past_key_values=DynamicCache())

        model.config._attn_implementation = "eager"
        with pytest.raises(ValueError, match="now uses `eager`"):
            model(input_ids=input_ids)
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


class TestContextParallelConfig(unittest.TestCase):
    def test_refuses_other_parallelisms(self):
        for other in ({"pp_size": 2}, {"ep_size": 2}):
            with self.subTest(**other), self.assertRaisesRegex(ValueError, "not supported yet with"):
                DistributedConfig(cp_size=2, **other)

    def test_accepts_tensor_parallelism(self):
        distributed_config = DistributedConfig(tp_size=2, cp_size=2)
        self.assertEqual((distributed_config.tp_size, distributed_config.cp_size), (2, 2))
        # the tp ranks hold the same batch: they are not data parallel ranks
        self.assertEqual(distributed_config.dp_size, 2)

    def test_refuses_size_below_one(self):
        with self.assertRaisesRegex(ValueError, "must be >= 1"):
            DistributedConfig(cp_size=0)


class TestContextParallelModel(unittest.TestCase):
    """A model sharded with context parallelism reproduces the same run on the full sequences."""

    def setUp(self):
        if torch._C._get_accelerator().type == "mps":
            self.skipTest("Distributed meshes are built on the default accelerator, which MPS cannot back")

    def test_matches_unsharded(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                run_distributed(world_size=2, backend="gloo")(_test_matches_unsharded)(2, 1, 1, packed)

    def test_matches_unsharded_with_fsdp(self):
        run_distributed(world_size=4, backend="gloo")(_test_matches_unsharded)(2, 2, 1, False)

    def test_matches_unsharded_with_hsdp(self):
        run_distributed(world_size=4, backend="gloo")(_test_matches_unsharded)(2, 1, 2, False)

    def test_matches_unsharded_with_tensor_parallelism(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                run_distributed(world_size=4, backend="gloo")(_test_matches_unsharded_with_tp)(2, 2, 1, packed)

    def test_matches_unsharded_with_tensor_parallelism_and_fsdp(self):
        run_distributed(world_size=8, backend="gloo")(_test_matches_unsharded_with_tp)(2, 2, 2, False)

    def test_inputs_embeds(self):
        run_distributed(world_size=2, backend="gloo")(_test_inputs_embeds)(2)

    def test_ignored_labels(self):
        run_distributed(world_size=2, backend="gloo")(_test_ignored_labels)(2)

    def test_keeps_num_items_in_batch(self):
        run_distributed(world_size=2, backend="gloo")(_test_keeps_num_items_in_batch)(2)

    def test_custom_implementation(self):
        run_distributed(world_size=2, backend="gloo")(_test_custom_implementation)(2)

    def test_forward_refusals(self):
        run_distributed(world_size=2, backend="gloo")(_test_forward_refusals)(2)

    def test_save_pretrained(self):
        run_distributed(world_size=2, backend="gloo")(_test_save_pretrained)(2)

    def test_save_pretrained_with_tensor_parallelism(self):
        run_distributed(world_size=4, backend="gloo")(_test_save_pretrained)(2, 2)

    def test_save_load_distributed_checkpoint(self):
        run_distributed(world_size=2, backend="gloo")(_test_save_load_distributed_checkpoint)(2)
