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

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM, MistralConfig
from transformers.distributed.sequence_parallel import ALL_SP_STYLES, SplitSequenceParallel, UlyssesAttention
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.mistral.modeling_mistral import MistralAttention, MistralRotaryEmbedding
from transformers.testing_utils import is_fsdp_test

from ..distributed_testing_utils import run_distributed


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


@is_fsdp_test
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
