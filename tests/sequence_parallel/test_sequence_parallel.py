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
    Gemma3TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
)
from transformers.distributed import DistributedConfig
from transformers.distributed.sequence_parallel import ALL_SP_STYLES, SplitSequenceParallel, UlyssesAttention
from transformers.distributed.utils import gather_full_state_dict
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.mistral.modeling_mistral import MistralAttention, MistralRotaryEmbedding
from transformers.testing_utils import is_sequence_parallel_test
from transformers.utils import is_torch_greater_or_equal

from ..distributed_testing_utils import _deterministic_init_model_dir, _distributed_tmpdir, run_distributed


def _mesh(size=2, rank=0):
    """An `sp` mesh that only has to report its size and the caller's rank."""
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


class _Layer(torch.nn.Module):
    """A decoder layer stand-in that records what its forward receives."""

    def forward(self, hidden_states, **kwargs):
        self.seen = {"hidden_states": hidden_states, **kwargs}
        return hidden_states


def _test_ulysses_matches_full_attention(rank):
    """An attention layer on this rank's tokens matches the same layer on the full sequence."""
    sp_size = dist.get_world_size()
    # grouped-query attention (4 query heads, 2 key/value heads) with a sliding window of 4 tokens
    config = MistralConfig(
        hidden_size=16, num_attention_heads=4, num_key_value_heads=2, sliding_window=4, attn_implementation="sdpa"
    )
    torch.manual_seed(0)
    attention = MistralAttention(config, layer_idx=0)
    # two packed documents of 5 and 3 tokens: the window of token 4 reaches tokens 1-3, held by the other rank
    position_ids = torch.cat([torch.arange(5), torch.arange(3)]).expand(2, -1)
    hidden_states = torch.randn(2, 8, config.hidden_size, requires_grad=True)
    position_embeddings = MistralRotaryEmbedding(config)(hidden_states, position_ids)
    mask = create_sliding_window_causal_mask(config, hidden_states, None, None, position_ids=position_ids)

    expected, _ = attention(hidden_states, position_embeddings, mask)
    expected.sum().backward()

    sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
    UlyssesAttention().install_forward(attention, init_device_mesh("cpu", (sp_size,)))
    local_hidden_states = hidden_states.detach().chunk(sp_size, dim=1)[rank].clone().requires_grad_()
    local_position_embeddings = tuple(x.chunk(sp_size, dim=1)[rank] for x in position_embeddings)
    output, _ = attention(local_hidden_states, local_position_embeddings, mask)
    output.sum().backward()

    torch.testing.assert_close(output, expected.chunk(sp_size, dim=1)[rank])
    torch.testing.assert_close(local_hidden_states.grad, hidden_states.grad.chunk(sp_size, dim=1)[rank])
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is sdpa


@is_sequence_parallel_test
class TestUlyssesAttention(unittest.TestCase):
    def test_matches_full_attention(self):
        run_distributed(world_size=2, backend="gloo")(_test_ulysses_matches_full_attention)()

    def test_refusals(self):
        for config, sp_size, message in (
            (_config(attn_implementation="eager"), 2, "takes no attention mask"),
            (_config(attn_implementation="flex_attention"), 2, "takes no attention mask"),
            (_config(hidden_size=48, num_attention_heads=6, num_key_value_heads=6), 4, "must divide"),
            (_config(num_attention_heads=8, num_key_value_heads=2), 4, "must divide"),
        ):
            with self.subTest(message, sp_size=sp_size), self.assertRaisesRegex(ValueError, message):
                UlyssesAttention().install_forward(MistralAttention(config, layer_idx=0), _mesh(sp_size))


class TestSplitSequenceParallel(unittest.TestCase):
    def test_splits_the_sequence(self):
        first, later = _Layer(), _Layer()
        SplitSequenceParallel().install_forward(first, _mesh(rank=1), split_hidden_states=True)
        SplitSequenceParallel().install_forward(later, _mesh(rank=1))
        hidden_states, cos, sin = (torch.randn(2, 8, 4) for _ in range(3))
        mask = torch.ones(2, 1, 8, 8)

        # rank 1 gets tokens 4-7 and their position embeddings; the mask stays the full sequence's
        local = first(hidden_states, attention_mask=mask, position_embeddings=(cos, sin))
        torch.testing.assert_close(local, hidden_states[:, 4:])
        torch.testing.assert_close(first.seen["position_embeddings"], (cos[:, 4:], sin[:, 4:]))
        self.assertIs(first.seen["attention_mask"], mask)
        # some models pass the hidden states by keyword
        torch.testing.assert_close(first(hidden_states=hidden_states, position_embeddings=(cos, sin)), local)

        # a later layer already gets tokens 4-7: it only slices the position embeddings
        later(local, position_embeddings=(cos, sin))
        torch.testing.assert_close(later.seen["hidden_states"], local)
        torch.testing.assert_close(later.seen["position_embeddings"], (cos[:, 4:], sin[:, 4:]))

    def test_refuses_a_sequence_the_ranks_cannot_split(self):
        layer = _Layer()
        SplitSequenceParallel().install_forward(layer, _mesh(), split_hidden_states=True)
        with self.assertRaisesRegex(ValueError, "must be divisible by `sp_size`"):
            layer(torch.randn(1, 7, 4))


class TestModelSequenceParallel(unittest.TestCase):
    def test_pads_and_splits_the_loss(self):
        model, style = LlamaForCausalLM(_config(pad_token_id=0)), ALL_SP_STYLES["sp_model"]
        input_ids = torch.randint(1, 99, (2, 7), generator=torch.Generator().manual_seed(0))
        inputs = {
            "input_ids": input_ids,
            "labels": input_ids,
            "position_ids": torch.tensor([[0, 1, 2, 3, 0, 1, 2]] * 2),
        }

        # 7 tokens over 2 ranks: one padding token, with no label, continuing the positions of the last document
        _, padded = style.transform_inputs_pre_forward(model, (), dict(inputs), _mesh(rank=0))
        torch.testing.assert_close(padded["input_ids"][:, -1], torch.tensor([0, 0]))
        torch.testing.assert_close(padded["labels"][:, -1], torch.tensor([-100, -100]))
        torch.testing.assert_close(padded["position_ids"][0], torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))

        # each rank computes the loss of its 4 tokens; the two partial losses add up to the full-sequence loss
        logits = torch.randn(2, 8, 99, generator=torch.Generator().manual_seed(1))
        partial_losses = []
        for rank in (0, 1):
            _, kwargs = style.transform_inputs_pre_forward(model, (), dict(inputs), _mesh(rank=rank))
            local_logits = logits[:, 4 * rank : 4 * (rank + 1)].contiguous()
            partial_losses.append(
                model.loss_function(
                    local_logits, None, 99, kwargs["num_items_in_batch"], shift_labels=kwargs["shift_labels"]
                )
            )
        torch.testing.assert_close(sum(partial_losses), model.loss_function(logits[:, :7].contiguous(), input_ids, 99))

    def test_refuses_a_cache(self):
        with self.assertRaisesRegex(ValueError, "`past_key_values` are not allowed"):
            ALL_SP_STYLES["sp_model"].transform_inputs_pre_forward(
                LlamaForCausalLM(_config()),
                (),
                {"input_ids": torch.ones(1, 4), "past_key_values": DynamicCache()},
                _mesh(),
            )


# =============================================================================
# A model sharded with sequence parallelism, against the same weights on full sequences
# =============================================================================

SP_RTOL = 1e-5
SP_ATOL = 1e-5


def _tiny_batch(config, batch_size, seq_len, packed=False, seed=0):
    """A batch every rank builds alike, with `batch_size` rows for each data parallel rank."""
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), generator=generator)
    if not packed:
        return input_ids, None
    # two documents back to back, which `position_ids` has to describe for the whole sequence
    first = seq_len // 2
    position_ids = torch.cat([torch.arange(first), torch.arange(seq_len - first)]).expand(batch_size, -1)
    return input_ids, position_ids


def _local(tensor):
    """The full value of a tensor FSDP2 holds as a `DTensor`, or `tensor` itself."""
    return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor


def _data_parallel_rank_and_group(model, distributed_config):
    """
    Which rows of the batch this rank holds, and the group whose FSDP2 gradient contributions add up to the batch.

    The sp ranks of an sp group hold the same rows: they are part of the averaging group but not of the row split.
    """
    mesh_manager = model._mesh_manager
    fsdp_size, dp_replicate_size = distributed_config.fsdp_size, distributed_config.dp_replicate_size
    sp_size = distributed_config.sp_size
    dp_rank = 0
    if dp_replicate_size > 1:
        dp_rank = mesh_manager.get_mesh("dp_replicate").get_local_rank() * fsdp_size
    if fsdp_size > 1:
        dp_rank += mesh_manager.get_mesh("fsdp").get_local_rank()

    group_size = dp_replicate_size * fsdp_size
    if distributed_config.sp_size > 1:
        dims = ("dp_replicate", "sp") if dp_replicate_size > 1 else "sp"
        mesh = mesh_manager.get_mesh(dims)
        return dp_rank, [mesh.get_group(dim) for dim in mesh.mesh_dim_names], group_size * sp_size

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
    Run the full `batch` through an unsharded model and through the same weights sharded with sequence parallelism,
    then compare the loss, the logits and every gradient.

    `keep` is the 2D `attention_mask` of the unsharded batch: the logits of its padded positions are not part of what
    the model is trained on.
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

    # every rank computed the loss of the full sequences, and its own slice of the logits of the padded sequence: the
    # padding at the end of the last rank's slice has no counterpart in the unsharded run
    sp_rank = model._mesh_manager.get_mesh("sp").get_local_rank()
    local_seq = output.logits.shape[1]
    start = sp_rank * local_seq
    # the real positions of this rank's slice, which can be none of them when the padding fills it
    real = max(0, min(local_seq, reference_logits.shape[1] - start))
    expected_logits = reference_logits[rows][:, start : start + real]
    logits = _local(output.logits)[:, :real]
    if keep is not None:
        keep = (keep[:, start : start + real] == 1)[..., None]
        expected_logits, logits = expected_logits * keep, logits * keep
    torch.testing.assert_close(logits, expected_logits, rtol=SP_RTOL, atol=SP_ATOL, msg="Logits mismatch")

    loss = _average_over_group(output.loss.detach(), groups, group_size)
    torch.testing.assert_close(loss, reference_loss, rtol=SP_RTOL, atol=SP_ATOL, msg="Loss mismatch")

    for name, param in model.named_parameters():
        grad = None if param.grad is None else _local(param.grad).float()
        expected = reference_grads.get(name)
        if grad is None or expected is None:
            assert grad is None and expected is None, f"Only one of the two runs has a gradient for {name}"
            continue
        grad = _average_over_group(grad, groups, group_size)
        torch.testing.assert_close(grad, expected, rtol=SP_RTOL, atol=SP_ATOL, msg=f"Gradient mismatch for {name}")


def _test_matches_unsharded(rank, sp_size, fsdp_size, dp_replicate_size, packed, seq_len=8):
    config = _config()
    if sp_size > 2:  # Ulysses splits the heads over the ranks, so four ranks need four key/value heads
        config.num_key_value_heads = 4
    batch_size = 2
    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        input_ids, position_ids = _tiny_batch(config, batch_size * fsdp_size * dp_replicate_size, seq_len, packed)
        distributed_config = DistributedConfig(
            sp_size=sp_size, fsdp_size=fsdp_size, dp_replicate_size=dp_replicate_size
        )
        batch = {"input_ids": input_ids, "labels": input_ids.clone(), "use_cache": False}
        if position_ids is not None:
            batch["position_ids"] = position_ids
        _compare_with_unsharded(init_dir, batch, distributed_config)
        # `init_dir` is rank 0's temporary directory: wait for the other ranks before it is removed
        dist.barrier()


def _test_padded(rank, sp_size, config=None, packed=False, seq_len=8):
    """Left and right padding through a 2D `attention_mask`; the padded positions carry no label."""
    config = config or _config()
    batch_size = 2
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
        _compare_with_unsharded(init_dir, batch, DistributedConfig(sp_size=sp_size), keep=attention_mask)
        dist.barrier()


def _test_inputs_embeds(rank, sp_size, seq_len=8):
    """`inputs_embeds` is sharded along its sequence dim, not along its last, hidden one."""
    config = _config()
    if sp_size > 2:
        config.num_key_value_heads = 4
    batch_size = 2
    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        input_ids, _ = _tiny_batch(config, batch_size, seq_len)
        reference = AutoModelForCausalLM.from_pretrained(init_dir)
        inputs_embeds = reference.get_input_embeddings()(input_ids).detach()
        del reference

        batch = {"inputs_embeds": inputs_embeds, "labels": input_ids.clone(), "use_cache": False}
        _compare_with_unsharded(init_dir, batch, DistributedConfig(sp_size=sp_size))
        dist.barrier()


class _CountingUlysses(UlyssesAttention):
    """An Ulysses style that counts the attention calls routed through it."""

    def __init__(self):
        self.attention_calls = 0

    def attention(self, *args, **kwargs):
        self.attention_calls += 1
        return super().attention(*args, **kwargs)


def _test_custom_attention_style(rank, sp_size):
    """A registered attention style runs the attention when the plan names it; an unknown style is refused."""
    config = _config()
    counting = _CountingUlysses()
    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        with mock.patch.dict(ALL_SP_STYLES, {"sp_counting": counting}):
            distributed_config = DistributedConfig(
                sp_size=sp_size, sp_plan={"model.layers.*.self_attn": "sp_counting"}
            )
            model = AutoModelForCausalLM.from_pretrained(init_dir, distributed_config=distributed_config)

        input_ids, _ = _tiny_batch(config, 2, 8)
        model(input_ids=input_ids, labels=input_ids, use_cache=False)
        # every attention module runs through the style, once per forward
        assert counting.attention_calls == len(model.model.layers), counting.attention_calls

        unknown = DistributedConfig(sp_size=sp_size, sp_plan={"model.layers.*.self_attn": "sp_missing"})
        with pytest.raises(ValueError, match="Unsupported `sp_plan` styles"):
            AutoModelForCausalLM.from_pretrained(init_dir, distributed_config=unknown)
        dist.barrier()


def _test_save_pretrained(rank, sp_size):
    """Weights saved from a sequence parallel model reload without a distributed config, unchanged."""
    config = _config()
    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        reference = AutoModelForCausalLM.from_pretrained(init_dir)
        expected = {name: value.clone() for name, value in reference.state_dict().items()}
        del reference

        model = AutoModelForCausalLM.from_pretrained(init_dir, distributed_config=DistributedConfig(sp_size=sp_size))
        with _distributed_tmpdir(rank) as save_dir:
            model.save_pretrained(save_dir, is_main_process=(rank == 0))
            dist.barrier()
            reloaded = AutoModelForCausalLM.from_pretrained(save_dir)
            # rank 0 removes `save_dir` when it leaves this block
            dist.barrier()

        for name, value in reloaded.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0, msg=f"Weight mismatch for {name}")
        dist.barrier()


def _test_save_load_distributed_checkpoint(rank, sp_size):
    """A rank-local checkpoint written by a sequence parallel model loads back into the same sharding."""
    config = _config()
    distributed_config = DistributedConfig(sp_size=sp_size)
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


@unittest.skipUnless(is_torch_greater_or_equal("2.9"), "Sequence parallelism requires torch>=2.9")
class TestSequenceParallelConfig(unittest.TestCase):
    def test_refuses_other_parallelisms(self):
        for other in ({"pp_size": 2}, {"fsdp_size": 2}, {"tp_size": 2}, {"ep_size": 2}):
            with self.subTest(**other), self.assertRaisesRegex(ValueError, "not supported yet with"):
                DistributedConfig(sp_size=2, **other)

    def test_refuses_a_model_without_a_sp_plan(self):
        config = _config()
        config.base_model_sp_plan = None
        model = LlamaForCausalLM(config)
        with self.assertRaisesRegex(ValueError, r"`sp_plan` has no `sp_split_sequence` rule matching a module"):
            model._resolve_parallel_plans(DistributedConfig(sp_size=2))


@unittest.skipUnless(is_torch_greater_or_equal("2.9"), "Sequence parallelism requires torch>=2.9")
class TestSequenceParallelPlan(unittest.TestCase):
    def test_resolves_the_default_plan(self):
        model = LlamaForCausalLM(_config())
        _, _, _, sp_plan = model._resolve_parallel_plans(DistributedConfig(sp_size=2))
        expected = {"": "sp_model"}
        for layer in range(model.config.num_hidden_layers):
            expected[f"model.layers.{layer}"] = "sp_split_sequence"
            expected[f"model.layers.{layer}.self_attn"] = "sp_ulysses"
        self.assertEqual(sp_plan, expected)

    def test_every_model_is_the_root_of_its_plan(self):
        config = _config()
        config.base_model_sp_plan = {"layers.*": "sp_split_sequence", "layers.*.self_attn": "sp_ulysses"}
        model = LlamaForCausalLM(config)

        self.assertEqual(model.model.sp_plan, {"": "sp_model", **config.base_model_sp_plan})
        # the head, not its base model, is the `""` of the head's plan
        self.assertEqual(
            model.sp_plan,
            {"": "sp_model", "model.layers.*": "sp_split_sequence", "model.layers.*.self_attn": "sp_ulysses"},
        )
        self.assertNotIn("base_model_sp_plan", config.to_dict())

    def test_refuses_a_plan_missing_a_style(self):
        for base_model_sp_plan, missing in (
            ({"layers.*": "sp_split_sequence"}, ["sp_ulysses"]),
            ({"layers.*.self_attn": "sp_ulysses"}, ["sp_split_sequence"]),
        ):
            config = _config()
            config.base_model_sp_plan = base_model_sp_plan
            model = LlamaForCausalLM(config)
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, rf"has no `{missing[0]}` rule"):
                model._resolve_parallel_plans(DistributedConfig(sp_size=2))


@is_sequence_parallel_test
class TestSequenceParallelModel(unittest.TestCase):
    """A model sharded with sequence parallelism reproduces the same run on the full sequences."""

    def setUp(self):
        if torch._C._get_accelerator().type == "mps":
            self.skipTest("Distributed meshes are built on the default accelerator, which MPS cannot back")
        if not is_torch_greater_or_equal("2.9"):
            self.skipTest("Sequence parallelism requires torch>=2.9")

    def test_matches_unsharded(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                run_distributed(world_size=2, backend="gloo")(_test_matches_unsharded)(2, 1, 1, packed)

    def test_matches_unsharded_with_hsdp(self):
        run_distributed(world_size=4, backend="gloo")(_test_matches_unsharded)(2, 1, 2, False)

    def test_padded(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                run_distributed(world_size=2, backend="gloo")(_test_padded)(2, None, packed)

    def test_padded_sequences(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                run_distributed(world_size=2, backend="gloo")(_test_matches_unsharded)(2, 1, 1, packed, 7)

    def test_padded_sequences_with_four_ranks(self):
        run_distributed(world_size=4, backend="gloo")(_test_matches_unsharded)(4, 1, 1, False, 5)

    def test_padded_sequences_with_a_padding_mask(self):
        run_distributed(world_size=2, backend="gloo")(_test_padded)(2, None, False, 7)

    def test_padded_inputs_embeds(self):
        run_distributed(world_size=2, backend="gloo")(_test_inputs_embeds)(2, 7)

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

    def test_inputs_embeds(self):
        run_distributed(world_size=2, backend="gloo")(_test_inputs_embeds)(2)

    def test_custom_attention_style(self):
        run_distributed(world_size=2, backend="gloo")(_test_custom_attention_style)(2)

    def test_save_pretrained(self):
        run_distributed(world_size=2, backend="gloo")(_test_save_pretrained)(2)

    def test_save_load_distributed_checkpoint(self):
        run_distributed(world_size=2, backend="gloo")(_test_save_load_distributed_checkpoint)(2)
