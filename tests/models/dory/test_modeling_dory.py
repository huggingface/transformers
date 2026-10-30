# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recurrence/cache contracts on tiny models; real-checkpoint parity is tested separately."""

import unittest

import pytest
import torch
from huggingface_hub.errors import StrictDataclassClassValidationError
from parameterized import parameterized

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DoryConfig,
    DoryForCausalLM,
    DoryModel,
)
from transformers.models.dory.modeling_dory import DoryCache, apply_rotary_pos_emb
from transformers.testing_utils import require_deepspeed, require_torch_accelerator, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import _deepspeed_zero3


def make_model(cache_mode="per_loop", loops=3, attn_implementation="eager", **overrides):
    torch.manual_seed(42)
    config = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "num_hidden_layers": 8,
        "n_input_layers": 2,
        "n_recurrent_layers": 4,
        "n_output_layers": 2,
        "n_recurrent_loops": loops,
        "recurrent_kv_cache_mode": cache_mode,
        "sliding_window": 3,
        "max_position_embeddings": 32,
    }
    return AutoModelForCausalLM.from_config(
        DoryConfig(**(config | overrides)), attn_implementation=attn_implementation
    ).eval()


@pytest.mark.parametrize("loops", [1, 3])
@torch.no_grad()
def test_per_loop_chunked_prefill_and_decode_match_full_forward(loops):
    """Separate loop histories must reproduce a full causal forward across a window boundary."""
    model = make_model(loops=loops)
    tokens = torch.tensor([[1, 3, 5, 7, 9, 11, 13]])
    expected = model(tokens, use_cache=False).logits
    cache = None
    actual = []
    for start, end in [(0, 3), (3, 6), (6, 7)]:
        output = model(tokens[:, start:end], use_cache=True, past_key_values=cache)
        cache = output.past_key_values
        actual.append(output.logits)
        assert cache.get_seq_length() == end
    torch.testing.assert_close(torch.cat(actual, dim=1), expected, atol=1e-6, rtol=1e-5)
    assert sum(layer.is_initialized for layer in cache.layers) == 2 + 2 * loops


def test_last_loop_cache_replaces_current_positions_and_keeps_history():
    """Later loops overwrite the current chunk without duplicating tokens or touching the prefix."""
    cache = DoryCache()
    first = torch.ones(1, 1, 3, 2)
    cache.update(first, first * 2, 2, start_position=0)
    cache.update(first * 3, first * 4, 2, start_position=0)
    next_token = torch.full((1, 1, 1, 2), 5.0)
    cache.update(next_token, next_token * 2, 2, start_position=3)
    keys, values = cache.update(next_token * 3, next_token * 4, 2, start_position=3)
    torch.testing.assert_close(keys, torch.cat((first * 3, next_token * 3), dim=-2))
    torch.testing.assert_close(values, torch.cat((first * 4, next_token * 4), dim=-2))
    assert cache.get_seq_length() == 4


@torch.no_grad()
def test_last_loop_equals_per_loop_at_one_loop():
    tokens = torch.tensor([[1, 3, 5, 7]])
    outputs = []
    for mode in ("per_loop", "last_loop"):
        model = make_model(mode, loops=1)
        prefill = model(tokens[:, :3], use_cache=True)
        outputs.append(model(tokens[:, 3:], past_key_values=prefill.past_key_values, use_cache=True).logits)
    torch.testing.assert_close(*outputs, atol=0, rtol=0)


@torch.no_grad()
def test_learned_vocabulary_scale_is_applied_after_projection():
    model = make_model()
    hidden = torch.randn(2, model.config.hidden_size)
    model.lm_head.logit_scale.copy_(torch.arange(1, 33) * model.config.initializer_range)
    expected = torch.nn.functional.linear(hidden, model.lm_head.weight).float() * torch.arange(1, 33)
    torch.testing.assert_close(model.lm_head(hidden), expected)


def test_invalid_loop_count_override_is_rejected_on_load(tmp_path):
    make_model().save_pretrained(tmp_path)
    with pytest.raises(StrictDataclassClassValidationError, match="n_recurrent_loops"):
        AutoModelForCausalLM.from_pretrained(tmp_path, n_recurrent_loops=0)


@pytest.mark.parametrize("implementation,boolean", [("eager", False), ("sdpa", False), ("sdpa", True)])
@torch.no_grad()
def test_custom_causal_masks_preserve_causality(implementation, boolean):
    model = make_model(attn_implementation=implementation, layer_types=["full_attention", "mlp"] * 4)
    tokens = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]])
    allowed = torch.ones(4, 4, dtype=torch.bool).tril()[None, None]
    mask = allowed if boolean else torch.zeros(1, 1, 4, 4).masked_fill(~allowed, float("-inf"))
    expected = model(tokens, use_cache=False).logits
    actual = model(tokens, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(actual, expected)
    tokens[:, -1] = 11
    changed = model(tokens, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(actual[:, :-1], changed[:, :-1], atol=0, rtol=0)


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@torch.no_grad()
def test_prepared_mask_mapping_and_left_padding(implementation):
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    model = make_model(attn_implementation=implementation)
    tokens = torch.tensor([[0, 0, 1, 3, 5], [2, 4, 6, 8, 10]])
    padding = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    positions = padding.cumsum(-1) - 1
    positions.masked_fill_(padding == 0, 0)
    kwargs = {
        "config": model.config,
        "inputs_embeds": model.get_input_embeddings()(tokens),
        "attention_mask": padding,
        "past_key_values": None,
        "position_ids": positions,
    }
    masks = {
        "full_attention": create_causal_mask(**kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**kwargs),
    }
    expected = model(tokens, attention_mask=padding, position_ids=positions, use_cache=False).logits
    actual = model(tokens, attention_mask=masks, position_ids=positions, use_cache=False).logits
    torch.testing.assert_close(actual, expected)
    unpadded = model(tokens[:1, 2:], use_cache=False).logits
    torch.testing.assert_close(actual[:1, 2:], unpadded)


@pytest.mark.parametrize("cache_mode", ["per_loop", "last_loop"])
@pytest.mark.parametrize("window", [1, 3])
@torch.no_grad()
def test_bounded_cache_matches_full_history_across_recurrent_chunks(cache_mode, window):
    model = make_model(cache_mode, sliding_window=window, attn_implementation="sdpa")
    bounded, reference = DoryCache(model.config), DoryCache()
    tokens = torch.arange(1, 17).unsqueeze(0)
    for start, end in [(0, 3), (3, 10), (10, 11), (11, 16)]:
        actual = model(tokens[:, start:end], past_key_values=bounded, use_cache=True).logits
        expected = model(tokens[:, start:end], past_key_values=reference, use_cache=True).logits
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        assert bounded.get_seq_length() == end
        for layer in bounded.layers:
            if layer.is_initialized and layer.is_sliding:
                assert layer.keys.shape[-2] <= window - 1
                assert layer.keys.untyped_storage().nbytes() == layer.keys.numel() * layer.keys.element_size()
    bounded.reset()
    assert bounded.get_seq_length() == 0
    torch.testing.assert_close(
        model(tokens[:, :3], past_key_values=bounded).logits,
        model(tokens[:, :3]).logits,
    )


@torch.no_grad()
def test_sliding_cache_rollback_with_recorded_past():
    model = make_model()
    cache = DoryCache(model.config)
    tokens = torch.arange(1, 11).unsqueeze(0)
    model(tokens[:, :5], past_key_values=cache)
    cache.activate_past_recording()
    model(tokens[:, 5:9], past_key_values=cache)
    cache.crop(-2)
    assert cache.get_seq_length() == 7
    actual = model(tokens[:, 7:], past_key_values=cache).logits
    expected = model(tokens, use_cache=False).logits[:, 7:]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("mean_resizing", [False, True])
@torch.no_grad()
def test_resize_vocabulary_preserves_projection_and_learned_scale(mean_resizing, tmp_path):
    model = make_model()
    model.lm_head.logit_scale.copy_(torch.linspace(0.01, 0.05, 32))
    tokens = torch.tensor([[1, 3, 5]])
    original = model(tokens, use_cache=False).logits
    scales = model.lm_head.logit_scale.clone()
    assert model.resize_token_embeddings() is model.get_input_embeddings()
    for size, multiple, expected_size in [(37, 8, 40), (20, None, 20)]:
        model.resize_token_embeddings(size, pad_to_multiple_of=multiple, mean_resizing=mean_resizing)
        assert model.config.vocab_size == expected_size
        assert model.lm_head.weight.shape == (expected_size, 16)
        assert model.lm_head.logit_scale.shape == (expected_size,)
        before_post_init = {name: value.clone() for name, value in model.lm_head.state_dict().items()}
        model.post_init()
        torch.testing.assert_close(model.lm_head.state_dict(), before_post_init, atol=0, rtol=0)
        count = min(32, expected_size)
        torch.testing.assert_close(model.lm_head.logit_scale[:count], scales[:count], atol=0, rtol=0)
        torch.testing.assert_close(model(tokens, use_cache=False).logits[..., :count], original[..., :count])
        if expected_size > 32:
            torch.testing.assert_close(model.lm_head.logit_scale[32:], scales.mean().expand(8))
            assert model(torch.tensor([[36]]), use_cache=False).logits.shape[-1] == expected_size
    model.save_pretrained(tmp_path)
    restored = AutoModelForCausalLM.from_pretrained(tmp_path)
    torch.testing.assert_close(restored(tokens, use_cache=False).logits, model(tokens, use_cache=False).logits)


@pytest.mark.parametrize("reentrant", [False, True])
def test_checkpointing_preserves_recurrent_gradients(reentrant):
    tokens = torch.tensor([[1, 3, 5, 7]])
    model = make_model().train()
    model(tokens, labels=tokens, use_cache=False).loss.backward()
    expected = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
    model(tokens, labels=tokens, use_cache=False).loss.backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, expected[name])


@pytest.mark.parametrize("cache_mode", ["per_loop", "last_loop"])
@torch.no_grad()
def test_beam_search_and_left_padded_batch(cache_mode):
    model = make_model(cache_mode)
    tokens = torch.tensor([[0, 0, 1, 3], [2, 4, 6, 8]])
    mask = tokens.ne(0)
    actual = model.generate(tokens, attention_mask=mask, num_beams=3, max_new_tokens=3, do_sample=False)
    for row, offset in [(0, 2), (1, 0)]:
        expected = model.generate(tokens[row : row + 1, offset:], num_beams=3, max_new_tokens=3, do_sample=False)
        torch.testing.assert_close(actual[row, -3:], expected[0, -3:])


@pytest.mark.parametrize("loops", [1, 3])
@torch.no_grad()
def test_last_loop_matches_explicit_final_loop_history(loops):
    shared, reference = make_model("last_loop", loops), make_model("per_loop", loops)
    shared_cache, reference_cache = DoryCache(), DoryCache()
    tokens = torch.arange(1, 12).unsqueeze(0)
    for start, end in [(0, 3), (3, 4), (4, 8), (8, 11)]:
        actual = shared(tokens[:, start:end], past_key_values=shared_cache).logits
        expected = reference(tokens[:, start:end], past_key_values=reference_cache).logits
        torch.testing.assert_close(actual, expected)
        for index in range(
            reference.config.n_input_layers, reference.config.n_input_layers + reference.config.n_recurrent_layers
        ):
            if reference.config.layer_types[index] == "mlp":
                continue
            final = reference_cache.layers[(loops - 1) * reference.config.num_hidden_layers + index]
            for loop in range(loops):
                layer = reference_cache.layers[loop * reference.config.num_hidden_layers + index]
                layer.keys, layer.values = final.keys.clone(), final.values.clone()


@pytest.mark.parametrize("legacy_full_rotary", [False, True])
@torch.no_grad()
def test_rotary_distinct_attention_frequencies(legacy_full_rotary):
    rope = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000},
        "full_attention": {"rope_type": "default", "rope_theta": 10000000},
    }
    if legacy_full_rotary:
        for parameters in rope.values():
            parameters["partial_rotary_factor"] = 1.0
    model = make_model(rope_parameters=rope)
    tokens = torch.arange(1, 8).unsqueeze(0)
    expected = model(tokens, use_cache=False).logits
    first = model(tokens[:, :4])
    actual = model(tokens[:, 4:], past_key_values=first.past_key_values).logits
    torch.testing.assert_close(actual, expected[:, 4:])
    rotary = model.backbone.rotary_emb
    values = torch.randn(1, 7, 2, 8)
    positions = torch.arange(7).unsqueeze(0) + 4096
    rotations = []
    for kind, parameters in rope.items():
        inverse_frequencies = 1.0 / parameters["rope_theta"] ** (torch.arange(0, 8, 2) / 8)
        frequencies = positions.float().unsqueeze(-1) * inverse_frequencies
        cosine, sine = frequencies.cos().unsqueeze(2), frequencies.sin().unsqueeze(2)
        first_half, second_half = values.chunk(2, dim=-1)
        expected = torch.cat(
            (first_half * cosine - second_half * sine, second_half * cosine + first_half * sine), dim=-1
        )
        rotated, _ = apply_rotary_pos_emb(values, values, *rotary(values, positions, kind), unsqueeze_dim=2)
        torch.testing.assert_close(rotated, expected)
        rotations.append(rotated)
    assert not torch.allclose(*rotations)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cache_mode", ["per_loop", "last_loop"])
@torch.no_grad()
def test_rotary_buffers_survive_save_load(tmp_path, dtype, cache_mode):
    make_model(cache_mode).save_pretrained(tmp_path / "original")
    # Load in the target dtype so non-persistent RoPE buffers retain FP32 precision.
    model = DoryForCausalLM.from_pretrained(tmp_path / "original", dtype=dtype, attn_implementation="eager").eval()
    tokens = torch.tensor([[1, 3, 5, 7]])
    positions = torch.arange(4).unsqueeze(0) + 4096
    expected = model(tokens, position_ids=positions, use_cache=False).logits
    model.save_pretrained(tmp_path)
    loaded = DoryForCausalLM.from_pretrained(tmp_path, dtype=dtype, attn_implementation="eager").eval()
    actual = loaded(tokens, position_ids=positions, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_recurrent_loops": 0},
        {"recurrent_kv_cache_mode": "shared"},
        {"n_input_layers": 3, "n_recurrent_layers": 4, "n_output_layers": 2},
        {"layer_types": ["full_attention"]},
        {"num_key_value_heads": 3},
        {"sliding_window": 0},
        {
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.5},
                "full_attention": {"rope_type": "default", "rope_theta": 10000000},
            }
        },
    ],
)
def test_invalid_architecture_is_rejected(overrides):
    with pytest.raises(StrictDataclassClassValidationError):
        make_model(**overrides)


@require_torch_accelerator
@torch.inference_mode()
def test_flex_attention_prefill_and_decode():
    model = make_model(loops=2, hidden_size=64, head_dim=32, intermediate_size=96, sliding_window=128)
    model.to(torch_device, dtype=torch.bfloat16)
    tokens = torch.arange(137, device=torch_device).unsqueeze(0) % 32
    expected = model(tokens, use_cache=False).logits
    model.set_attn_implementation("flex_attention")
    full = model(tokens, use_cache=False).logits
    prefill = model(tokens[:, :129])
    decoded = model(tokens[:, 129:], past_key_values=prefill.past_key_values).logits
    torch.testing.assert_close(full, expected, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(decoded, expected[:, 129:], atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("loops", [1, 4])
@pytest.mark.parametrize("cache_mode", ["per_loop", "last_loop"])
@torch.no_grad()
def test_changing_loop_count_between_sequences(cache_mode, loops):
    model = make_model(cache_mode, loops=2)
    model.config.n_recurrent_loops = loops
    reference = make_model(cache_mode, loops=loops)
    reference.load_state_dict(model.state_dict())
    tokens = torch.arange(1, 17).unsqueeze(0)
    cache, reference_cache = None, None
    for start, end in [(0, 8), (8, 9), (9, 16)]:
        actual = model(tokens[:, start:end], past_key_values=cache)
        expected = reference(tokens[:, start:end], past_key_values=reference_cache)
        cache, reference_cache = actual.past_key_values, expected.past_key_values
        torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
        for layer in cache.layers:
            if layer.is_initialized and layer.is_sliding:
                assert layer.keys.shape[-2] == model.config.sliding_window - 1


def test_mlp_scales_initialize_to_one():
    model = make_model()
    for layer in model.backbone.layers:
        if layer.layer_type == "mlp":
            torch.testing.assert_close(layer.mixer.suv_gate, torch.ones_like(layer.mixer.suv_gate))
            torch.testing.assert_close(layer.mixer.suv_up, torch.ones_like(layer.mixer.suv_up))


@pytest.mark.parametrize("mode", ["prompt_lookup", "assistant"])
@torch.no_grad()
def test_assisted_generation_creates_recurrent_cache(mode):
    model = make_model(loops=3)
    # Repeated prompt tokens exercise prompt lookup; a shallower Dory exercises draft rollback.
    tokens = torch.tensor([[1, 3, 1, 3, 1, 3]])
    if mode == "prompt_lookup":
        options = {"prompt_lookup_num_tokens": 3}
    else:
        assistant = make_model(loops=1)
        assistant.load_state_dict(model.state_dict())
        assistant.generation_config.num_assistant_tokens = 3
        assistant.generation_config.num_assistant_tokens_schedule = "constant"
        assistant.generation_config.assistant_confidence_threshold = 0.0
        options = {"assistant_model": assistant}
    expected = model.generate(tokens, max_new_tokens=8, do_sample=False)
    actual = model.generate(tokens, max_new_tokens=8, do_sample=False, **options)
    torch.testing.assert_close(actual, expected)


class DoryModelTester(CausalLMModelTester):
    base_model_class = DoryModel

    def __init__(self, parent, **kwargs):
        super().__init__(parent, hidden_act="silu", n_recurrent_loops=2, sliding_window=16, **kwargs)
        self.expected_num_hidden_layers = self.num_hidden_layers * self.n_recurrent_loops + 1


class DoryModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = DoryModelTester
    model_split_percents = [0.5, 0.8, 0.9]

    @parameterized.expand([("cpu",), ("none",)])
    @require_deepspeed
    @require_torch_accelerator
    def test_resize_vocabulary_scales_with_deepspeed(self, offload_device):
        import deepspeed

        ds_config = {
            "train_batch_size": 2,
            "zero_optimization": {"stage": 3, "offload_param": {"device": offload_device}},
        }
        with _deepspeed_zero3(ds_config):
            config = self.model_tester.get_config()
            vocab_size = config.vocab_size
            model = AutoModelForCausalLM.from_config(config)
            old_scale = model.lm_head.logit_scale
            with deepspeed.zero.GatheredParameters(old_scale, modifier_rank=0):
                scales = torch.linspace(1.0, 3.0, vocab_size, device=old_scale.device, dtype=old_scale.dtype)
                old_scale.data.copy_(scales)
            self.assertEqual(old_scale.numel(), 0)
            for size in (vocab_size + 8, vocab_size - 8):
                model.resize_token_embeddings(size)
                actual = model.lm_head.logit_scale
                self.assertTrue(actual.isfinite().all())
                count = min(vocab_size, size)
                torch.testing.assert_close(actual[:count], scales[:count])
                if size > vocab_size:
                    torch.testing.assert_close(actual[vocab_size:], scales.mean().expand(8))

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        self.assertIsInstance(past_key_values, DoryCache)
        loops = config.n_recurrent_loops if config.recurrent_kv_cache_mode == "per_loop" else 1
        self.assertEqual(len(past_key_values.layers), loops * config.num_hidden_layers)
        self.assertEqual(past_key_values.get_seq_length(), seq_length)
        for index, layer in enumerate(past_key_values.layers):
            loop, physical = divmod(index, config.num_hidden_layers)
            recurrent = config.n_input_layers <= physical < config.n_input_layers + config.n_recurrent_layers
            active = config.layer_types[physical] != "mlp" and (loop == 0 or recurrent)
            self.assertEqual(layer.is_initialized, active)
            if active:
                length = min(seq_length, config.sliding_window - 1) if layer.is_sliding else seq_length
                shape = (batch_size, config.num_key_value_heads, length, config.head_dim)
                self.assertEqual(layer.keys.shape, shape)
                self.assertEqual(layer.values.shape, shape)

    @unittest.skip("Dory requires untied input embeddings and a scaled output projection")
    def test_tied_weights_keys(self):
        pass

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    @unittest.skip("Dory supports default RoPE only")
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass

    @unittest.skip("Dory supports default RoPE only; per-type frequencies are tested separately")
    def test_model_rope_scaling_frequencies(self):
        pass


@slow
@require_torch_accelerator
class DoryIntegrationTest(unittest.TestCase):
    checkpoint = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"

    @torch.inference_mode()
    def test_bf16_generation(self):
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        model = DoryForCausalLM.from_pretrained(self.checkpoint, dtype=torch.bfloat16, device_map=torch_device)
        inputs = tokenizer("The capital of France is", return_tensors="pt").to(torch_device)
        output = model.generate(**inputs, max_new_tokens=24, do_sample=False)
        self.assertEqual(
            tokenizer.decode(output[0], skip_special_tokens=True),
            "The capital of France is Paris. The capital of Germany is Berlin. "
            "The capital of Italy is Rome. The capital of Spain is Madrid. The",
        )
