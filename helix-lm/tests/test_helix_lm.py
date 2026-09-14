# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Correctness tests for HELIX.

These check the properties the architecture claims, on small random-weight models -- things you can verify
without training anything: that it is causal, that its fixed-size decode state reproduces a full forward,
that padding behaves, and that its cost per token does not grow with the context.
"""

import pytest
import torch

from helix_lm import HelixCache, HelixConfig, HelixForCausalLM


def build(dtype=torch.float64, **overrides):
    config = HelixConfig(
        **{
            "vocab_size": 96,
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "block_size": 4,
            "local_blocks": 2,
            "num_window_scales": 2,
            "index_layer_stride": 2,
            "landmark_dim": 16,
            "index_branching": 2,
            "index_beam_width": 2,
            "index_topk": 2,
            "num_recurrent_heads": 2,
            "recurrent_head_dim": 16,
            "recurrent_value_head_dim": 16,
            "recurrent_chunk_size": 4,
            "conv_kernel_size": 4,
            "surprise_kernel_size": 4,
            **overrides,
        }
    )
    torch.manual_seed(0)
    return HelixForCausalLM(config).eval().to(dtype)


@pytest.fixture(scope="module")
def model():
    return build()


@pytest.fixture(scope="module")
def tokens(model):
    torch.manual_seed(1)
    return torch.randint(0, model.config.vocab_size, (1, 30))


def test_forward_is_finite(model, tokens):
    logits = model(tokens).logits
    assert logits.shape == (1, tokens.shape[1], model.config.vocab_size)
    assert torch.isfinite(logits).all()


def test_strictly_causal(model, tokens):
    """Editing token t must leave every logit before t bitwise unchanged."""
    reference = model(tokens).logits
    for position in (5, 12, 23):
        edited = tokens.clone()
        edited[0, position] = (edited[0, position] + 7) % model.config.vocab_size
        torch.testing.assert_close(model(edited).logits[:, :position], reference[:, :position], rtol=0, atol=0)


def test_incremental_decoding_matches_prefill(model, tokens):
    """A fixed-size decode state must reproduce the one-shot forward, token for token."""
    reference = model(tokens).logits
    cache = HelixCache(model.config)
    steps = [model(tokens[:, i : i + 1], past_key_values=cache, use_cache=True).logits for i in range(tokens.shape[1])]
    torch.testing.assert_close(torch.cat(steps, dim=1), reference, rtol=1e-6, atol=1e-6)


def test_chunked_prefill_matches_prefill(model, tokens):
    """Prefilling in two chunks that do not land on a memory-block boundary must be equivalent."""
    reference = model(tokens).logits
    cache = HelixCache(model.config)
    first = model(tokens[:, :13], past_key_values=cache, use_cache=True).logits
    second = model(tokens[:, 13:], past_key_values=cache, use_cache=True).logits
    torch.testing.assert_close(torch.cat([first, second], dim=1), reference, rtol=1e-6, atol=1e-6)


def test_right_padding_is_exactly_invariant(model, tokens):
    """Trailing padding cannot reach earlier positions, so it must not perturb them at all."""
    length, padding = tokens.shape[1], 7
    padded = torch.cat([tokens, torch.zeros(1, padding, dtype=torch.long)], dim=1)
    mask = torch.cat([torch.ones(1, length, dtype=torch.long), torch.zeros(1, padding, dtype=torch.long)], dim=1)
    torch.testing.assert_close(
        model(padded, attention_mask=mask).logits[:, :length], model(tokens).logits, rtol=0, atol=0
    )


def test_tiling_does_not_change_the_result(model, tokens):
    """Query-block tiling is a memory knob, not a modelling one: it must be bitwise inert."""
    outputs = {}
    for tile in (0, 1, 3, 64):
        for layer in model.model.layers:
            layer.mixer.tile_blocks = tile
        outputs[tile] = model(tokens).logits
    for layer in model.model.layers:
        layer.mixer.tile_blocks = model.config.attention_tile_blocks
    for tile, logits in outputs.items():
        torch.testing.assert_close(logits, outputs[0], rtol=0, atol=0, msg=f"tile={tile}")


def test_attention_width_is_independent_of_context_length(model):
    """The direct evidence for O(N) compute: keys read per query do not grow with the context."""
    seen = {}
    for layer in model.model.layers:
        braid = layer.mixer
        original = braid._blocked_attention

        def counted(query_blocks, keys, *args, _original=original, **kwargs):
            observed.append(keys.shape[3])
            return _original(query_blocks, keys, *args, **kwargs)

        braid._blocked_attention = counted

    span = model.config.local_span
    for length in (8 * span, 32 * span):
        observed = []
        model(torch.randint(0, model.config.vocab_size, (1, length)))
        seen[length] = sorted(set(observed))

    assert seen[8 * span] == seen[32 * span]
    budget = (model.config.local_blocks + 1 + model.config.index_topk) * model.config.block_size
    assert all(width <= budget for width in seen[8 * span])


def test_index_reaches_beyond_the_local_window(model):
    """Every eligible query block selects something, and the index can see past the local window."""
    braid = next(layer.mixer for layer in model.model.layers if layer.mixer.has_index)
    original, captured = braid._select_memory_blocks, []

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    braid._select_memory_blocks = record
    model(torch.randint(0, model.config.vocab_size, (1, 8 * model.config.local_span)))
    braid._select_memory_blocks = original

    blocks, scores = captured[0]
    live = scores > -1e29
    assert bool(live[..., 1:, :].any(-1).all()), "an eligible query block selected nothing"
    query_block = torch.arange(blocks.shape[2]).view(1, 1, -1, 1)
    assert bool((live & (blocks < query_block - model.config.local_blocks)).any()), "index never left the window"


def test_cache_holds_what_the_architecture_claims(model, tokens):
    """Fixed-size states everywhere; the full history only where the index can reach it."""
    config = model.config
    cache = model(tokens, use_cache=True).past_key_values
    key_dim = config.num_recurrent_heads * config.recurrent_head_dim
    conv_dim = 2 * key_dim + config.num_recurrent_heads * config.recurrent_value_head_dim

    for layer_idx, (layer, layer_type) in enumerate(zip(cache.layers, config.layer_types, strict=True)):
        assert layer.conv_states[0].shape == (1, conv_dim, config.conv_kernel_size), layer_idx
        assert layer.conv_states[1].shape == (1, key_dim, config.surprise_kernel_size), layer_idx
        assert layer.recurrent_states[0].shape == (
            1,
            config.num_recurrent_heads,
            config.recurrent_head_dim,
            config.recurrent_value_head_dim,
        ), layer_idx
        # A windowed layer retains sliding_window - 1 tokens; the token being decoded fills the last slot.
        expected = tokens.shape[1] if layer_type == "helix" else min(tokens.shape[1], config.sliding_window - 1)
        assert layer.keys.shape == (1, config.num_key_value_heads, expected, config.head_dim), layer_idx


def test_windowed_layers_keep_a_bounded_cache():
    """Doubling the context must not grow the cache of a layer that cannot see the distant past."""
    model = build(dtype=torch.float32)
    sizes = []
    for length in (200, 400):
        cache = model(torch.randint(0, model.config.vocab_size, (1, length)), use_cache=True).past_key_values
        local = [
            layer.keys.shape[2]
            for layer, kind in zip(cache.layers, model.config.layer_types, strict=True)
            if kind == "helix_local"
        ]
        sizes.append(set(local))
    assert sizes[0] == sizes[1] == {model.config.sliding_window - 1}


def test_gradients_reach_every_parameter():
    """Including the landmark poolers, which are only trained through the routing bias."""
    model = build(dtype=torch.float32).train()
    tokens = torch.randint(0, 96, (2, 40))
    model(tokens, labels=tokens).loss.backward()
    starved = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not starved, starved


def test_generate_runs_and_is_deterministic_when_greedy(model):
    prompt = torch.randint(0, model.config.vocab_size, (2, 11))
    first = model.generate(prompt, max_new_tokens=9, temperature=0)
    second = model.generate(prompt, max_new_tokens=9, temperature=0)
    assert first.shape == (2, 20)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_save_and_load_round_trips(tmp_path, model, tokens):
    model.save_pretrained(tmp_path)
    reloaded = HelixForCausalLM.from_pretrained(tmp_path).eval().to(torch.float64)
    torch.testing.assert_close(reloaded(tokens).logits, model(tokens).logits, rtol=0, atol=0)


def test_landmark_tree_depth_follows_the_memory(model):
    """The tree must always grow until its top level fits the descent's seed beam."""
    config = model.config
    braid = next(layer.mixer for layer in model.model.layers if layer.mixer.has_index)
    for num_blocks in (1, config.index_branching - 1, config.index_branching, 5 * config.index_branching**2):
        leaves = torch.zeros(1, config.num_key_value_heads, num_blocks, config.landmark_dim, dtype=torch.float64)
        levels = braid._build_landmark_tree(leaves)
        assert len(levels) - 1 == config.index_num_levels(num_blocks), num_blocks
        assert levels[-1].shape[2] < config.index_branching, num_blocks


@pytest.mark.parametrize(
    "bad",
    [
        {"index_branching": 1},
        {"num_window_scales": 99},
        {"num_key_value_heads": 3},
        {"index_topk": 0},
        {"surprise_kernel_size": 1},
        {"layer_types": ["helix"]},
        {"layer_types": ["nope"] * 2, "num_hidden_layers": 2},
    ],
)
def test_invalid_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        HelixConfig(**{"num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 2, **bad})


def test_four_dimensional_masks_are_rejected(model, tokens):
    with pytest.raises(ValueError, match="2D padding mask"):
        model(tokens, attention_mask=torch.ones(1, 1, 30, 30))
