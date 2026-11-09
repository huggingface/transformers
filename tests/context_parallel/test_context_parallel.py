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

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from transformers import LlamaConfig, MistralConfig
from transformers.distributed import context_parallel
from transformers.distributed.context_parallel import (
    CPImplementation,
    UlyssesCP,
    _gather_heads_scatter_sequence,
    _gather_sequence_scatter_heads,
    register_context_parallel_attention,
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import create_causal_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.testing_utils import require_flash_attn, require_torch_multi_accelerator

from ..distributed_testing_utils import run_distributed


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

    def test_accepts_supported_config(self):
        self._check(LlamaConfig(attn_implementation="sdpa", num_attention_heads=8, num_key_value_heads=4))
        # flash kernels apply the sliding window themselves
        self._check(MistralConfig(attn_implementation="flash_attention_2"))
        config = LlamaConfig(attn_implementation="sdpa")
        config.layer_types = ["full_attention", "full_attention"]
        self._check(config)


@require_flash_attn
@require_torch_multi_accelerator
class TestUlyssesFlashAttention(unittest.TestCase):
    def test_packed_flash_attention(self):
        run_distributed(world_size=2, backend="nccl")(_test_flash_attention)(2)
