import copy
import os
from collections import Counter

import pytest
import torch

from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM, T5Config, T5ForConditionalGeneration
from transformers.generation import utils as generation_utils
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList


def device():
    return "cuda:0" if os.getenv("HF_TEST_DEVICE") == "cuda" and torch.cuda.is_available() else "cpu"


def configure(model, **overrides):
    cfg = copy.deepcopy(model.generation_config)
    settings = {
        "return_dict_in_generate": True,
        "output_assistant_metrics": False,
        "max_new_tokens": 24,
        "eos_token_id": [],
        "pad_token_id": 0,
        "do_sample": False,
        "use_cache": True,
    }
    settings.update(overrides)
    for key, value in settings.items():
        setattr(cfg, key, value)
    return cfg


def configure_assistant(model):
    model.generation_config.num_assistant_tokens = 4
    model.generation_config.num_assistant_tokens_schedule = "constant"
    model.generation_config.assistant_confidence_threshold = 0.0
    return model


def check(m):
    assert m is not None
    assert 0 <= m["accepted_draft_tokens"] <= m["proposed_draft_tokens"]
    assert m["verification_rounds"] == sum(m["accepted_length_histogram"].values())
    assert m["accepted_draft_tokens"] == sum(k * v for k, v in m["accepted_length_histogram"].items())
    assert m["draft_acceptance_rate"] == pytest.approx(
        m["accepted_draft_tokens"] / m["proposed_draft_tokens"] if m["proposed_draft_tokens"] else 0
    )
    assert m["mean_emitted_tokens_per_round"] == pytest.approx(
        m["emitted_tokens"] / m["verification_rounds"] if m["verification_rounds"] else 0
    )


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(1337)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
    )
    target = LlamaForCausalLM(cfg).to(device()).eval()
    identical = configure_assistant(copy.deepcopy(target).eval())
    torch.manual_seed(2026)
    different = configure_assistant(LlamaForCausalLM(cfg).to(device()).eval())
    ids = torch.tensor([[1, 11, 12, 13, 14, 15, 16, 17]], device=device())
    return target, identical, different, ids


def generate(models, assistant=None, enabled=False, prompt=None, **options):
    target, _, _, ids = models
    ids = ids if prompt is None else prompt
    cfg = configure(target, output_assistant_metrics=enabled, **options)
    with torch.inference_mode():
        return target.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            generation_config=cfg,
            assistant_model=assistant,
        )


def test_opt_in_only(models):
    twin = models[1]
    assert "assistant_metrics" not in generate(models, twin)
    assert "assistant_metrics" not in generate(models, enabled=True)
    check(generate(models, twin, enabled=True).assistant_metrics)


def test_greedy_parity_and_acceptance(models):
    _, twin, different, ids = models
    baseline = generate(models).sequences
    good, bad = (generate(models, assistant, enabled=True) for assistant in (twin, different))
    assert torch.equal(baseline, good.sequences) and torch.equal(baseline, bad.sequences)
    assert good.assistant_metrics["proposed_draft_tokens"] > 0
    assert good.assistant_metrics["accepted_draft_tokens"] == good.assistant_metrics["proposed_draft_tokens"]
    assert bad.assistant_metrics["accepted_draft_tokens"] < good.assistant_metrics["accepted_draft_tokens"]
    for out in (good, bad):
        check(out.assistant_metrics)
        assert out.assistant_metrics["emitted_tokens"] == out.sequences.shape[1] - ids.shape[1]


def test_accepted_eos_is_counted(models):
    _, twin, _, ids = models
    first = int(generate(models).sequences[0, ids.shape[1]])
    out = generate(models, twin, enabled=True, eos_token_id=first, max_new_tokens=8)
    check(out.assistant_metrics)
    assert out.sequences.shape[1] == ids.shape[1] + 1
    assert out.assistant_metrics["accepted_draft_tokens"] == 1


def test_max_length_boundary(models):
    out = generate(models, models[1], enabled=True, max_new_tokens=5)
    check(out.assistant_metrics)
    assert out.assistant_metrics["emitted_tokens"] == 5
    assert out.sequences.shape[1] == models[3].shape[1] + 5


def test_sampling_same_seed_with_metrics_off_and_on(models):
    torch.manual_seed(101)
    off = generate(models, models[1], do_sample=True, max_new_tokens=12)
    torch.manual_seed(101)
    on = generate(models, models[1], enabled=True, do_sample=True, max_new_tokens=12)
    assert torch.equal(off.sequences, on.sequences)
    check(on.assistant_metrics)


def test_prompt_lookup(models):
    repeated = torch.tensor([[1, 5, 6, 7, 5, 6, 7, 5, 6]], device=device())
    assisted = generate(
        models,
        enabled=True,
        prompt=repeated,
        max_new_tokens=12,
        prompt_lookup_num_tokens=3,
        max_matching_ngram_size=2,
    )
    assert torch.equal(assisted.sequences, generate(models, prompt=repeated, max_new_tokens=12).sequences)
    check(assisted.assistant_metrics)


def test_configuration_serialization():
    cfg = GenerationConfig(return_dict_in_generate=True, output_assistant_metrics=True)
    assert GenerationConfig.from_dict(cfg.to_dict()).output_assistant_metrics is True


def test_plain_tensor_output(models):
    target, twin, _, ids = models
    cfg = configure(target, return_dict_in_generate=False, max_new_tokens=10)
    with torch.inference_mode():
        standard = target.generate(input_ids=ids, generation_config=cfg)
        assisted = target.generate(input_ids=ids, generation_config=cfg, assistant_model=twin)
    assert isinstance(assisted, torch.Tensor) and torch.equal(standard, assisted)


def test_sampling_counters_match_observed_verification(models, monkeypatch):
    target, _, assistant, ids = models
    proposals, matches, sampling_calls = [], [], []
    original_factory = target._get_candidate_generator
    original_sampling = generation_utils._speculative_sampling

    def factory(*args, **kwargs):
        generator = original_factory(*args, **kwargs)
        original_get = generator.get_candidates

        def get_candidates(input_ids, **options):
            candidate_ids, logits = original_get(input_ids, **options)
            proposals.append(candidate_ids.shape[-1] - input_ids.shape[-1])
            matches.append(0)
            return candidate_ids, logits

        generator.get_candidates = get_candidates
        return generator

    def sampling(*args, **kwargs):
        selected, count = original_sampling(*args, **kwargs)
        sampling_calls.append(1)
        matches[-1] = int(count)
        return selected, count

    class Observe(StoppingCriteria):
        def __init__(self):
            self.lengths = []

        def __call__(self, input_ids, scores, **kwargs):
            self.lengths.append(input_ids.shape[-1])
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    observer = Observe()
    monkeypatch.setattr(target, "_get_candidate_generator", factory)
    monkeypatch.setattr(generation_utils, "_speculative_sampling", sampling)

    torch.manual_seed(101)
    with torch.inference_mode():
        output = target.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            generation_config=configure(target, output_assistant_metrics=True, do_sample=True),
            assistant_model=assistant,
            stopping_criteria=StoppingCriteriaList([observer]),
        )

    lengths = [ids.shape[-1], *observer.lengths]
    emitted = [end - start for start, end in zip(lengths, lengths[1:])]
    committed = [min(m, n) for m, n in zip(matches, emitted)]
    stats = output.assistant_metrics

    check(stats)
    assert sampling_calls
    assert len(proposals) == len(matches) == len(emitted)
    assert stats["proposed_draft_tokens"] == sum(proposals)
    assert stats["accepted_draft_tokens"] == sum(committed)
    assert stats["verification_rounds"] == len(emitted)
    assert stats["emitted_tokens"] == sum(emitted) == output.sequences.shape[-1] - ids.shape[-1]
    assert stats["accepted_length_histogram"] == dict(Counter(committed))


def test_encoder_decoder_assistant_metrics():
    torch.manual_seed(2307)
    config = T5Config(
        vocab_size=128,
        d_model=64,
        d_ff=128,
        d_kv=16,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        dropout_rate=0,
        use_cache=True,
    )
    target = T5ForConditionalGeneration(config).to(device()).eval()
    assistant = configure_assistant(copy.deepcopy(target).eval())
    inputs = torch.tensor([[5, 8, 11, 14, 17, 20]], device=device())

    def run(assistant_model=None, enabled=False):
        cfg = configure(
            target,
            output_assistant_metrics=enabled,
            max_new_tokens=12,
            forced_eos_token_id=None,
            decoder_start_token_id=0,
        )
        with torch.inference_mode():
            return target.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                generation_config=cfg,
                assistant_model=assistant_model,
            )

    baseline, off, on = run(), run(assistant), run(assistant, True)
    assert on.__class__.__name__ == "GenerateEncoderDecoderOutput"
    assert torch.equal(baseline.sequences, off.sequences)
    assert torch.equal(baseline.sequences, on.sequences)
    assert "assistant_metrics" not in baseline and "assistant_metrics" not in off
    check(on.assistant_metrics)
    assert on.assistant_metrics["proposed_draft_tokens"] > 0
    assert on.assistant_metrics["emitted_tokens"] == on.sequences.shape[-1] - 1 == 12
