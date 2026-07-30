"""Validate text-only classes loaded from the composite checkpoint.

Covers config extraction, pruned-head loading, the padded-logits contract, generation, and the bare
text backbone.
"""

import warnings

import torch
from _common import bootstrap, finish, run_case, setup_failure


def setup():
    """SETUP

    Load the text config, tokenizer, and causal LM.
    """
    required = (
        "Apertus1p5TextConfig",
        "Apertus1p5TextForCausalLM",
        "Apertus1p5TextModel",
        "AutoTokenizer",
    )
    transformers, checkpoint = bootstrap(required)
    config = transformers.Apertus1p5TextConfig.from_pretrained(checkpoint)
    print("SETUP: loading text causal LM (bf16, CPU) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, info = transformers.Apertus1p5TextForCausalLM.from_pretrained(
            checkpoint, dtype=torch.bfloat16, output_loading_info=True
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    return transformers, checkpoint, config, model.eval(), tokenizer, info


def case_1_config(config):
    """CASE 1: CONFIG

    Extract the nested text configuration.
    """
    assert type(config).__name__ == "Apertus1p5TextConfig", f"unexpected config type {type(config).__name__}"
    assert config.output_vocab_size, "missing pruned output_vocab_size"
    return f"vocab {config.vocab_size}; output vocab {config.output_vocab_size}; hidden size {config.hidden_size}"


def case_2_clean_load(model, info):
    """CASE 2: CLEAN LOAD

    Validate remapped weights and the pruned head.
    """
    assert not info["missing_keys"], f"missing keys: {info['missing_keys']}"
    assert not info["mismatched_keys"], f"mismatched keys: {info['mismatched_keys']}"
    assert all(
        key.startswith(("model.vision_tokenizer.", "model.audio_tokenizer.")) for key in info["unexpected_keys"]
    ), f"unexpected non-tokenizer keys: {info['unexpected_keys']}"
    config = model.config
    head_rows = model.lm_head.out_features
    assert head_rows == (config.output_vocab_size or config.vocab_size), "lm_head has the wrong width"
    return f"0 missing; {len(info['unexpected_keys'])} expected extras; head width {head_rows}"


def case_3_padded_logits(model, tokenizer):
    """CASE 3: PADDED LOGITS

    Keep logits finite with a zero-probability padded tail.
    """
    config = model.config
    inputs = tokenizer("The capital of Switzerland is", return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    assert logits.shape[-1] == config.vocab_size, f"expected logits padded to {config.vocab_size}, got {logits.shape}"
    assert bool(torch.isfinite(logits).all()), "padded logits contain non-finite values"
    tail = logits[..., config.output_vocab_size :]
    assert (tail == torch.finfo(logits.dtype).min).all(), "padded tail must be finfo.min everywhere"

    probabilities = logits.float().softmax(-1)
    assert (probabilities[..., config.output_vocab_size :] == 0).all(), "padded tail leaks probability mass"
    head_mass = float(probabilities[..., : config.output_vocab_size].sum(-1).min())
    assert abs(head_mass - 1.0) < 1e-4, f"head probabilities do not sum to 1: {head_mass}"
    return f"logits {logits.shape[-1]} wide, finite; tail softmaxes to exactly 0; head mass {head_mass:.6f}"


def case_4_raw_generation(model, tokenizer):
    """CASE 4: RAW GENERATION

    Generate a raw-text continuation.
    """
    inputs = tokenizer("The capital of Switzerland is", return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    completion = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    assert "Bern" in completion, f"expected 'Bern', got {completion!r}"
    return f"completion {completion!r}"


def case_5_chat_generation(model, tokenizer):
    """CASE 5: CHAT GENERATION

    Keep greedy and beam outputs below the pruned cutoff.
    """
    messages = [{"role": "user", "content": "Name three Swiss cities, comma separated."}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    completions = {}
    head_rows = model.lm_head.out_features
    for label, kwargs in (
        ("greedy", {"do_sample": False}),
        ("beam-2", {"num_beams": 2, "do_sample": False}),
    ):
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=16, **kwargs)
        new_ids = output[0, inputs["input_ids"].shape[1] :]
        assert new_ids.numel(), f"{label} generation produced no new tokens"
        assert int(new_ids.max()) < head_rows, f"{label} generated an id above the pruned cutoff"
        completions[label] = tokenizer.decode(new_ids, skip_special_tokens=True)
    return f"greedy {completions['greedy']!r}; beam-2 {completions['beam-2']!r}"


def case_6_bare_backbone(transformers, checkpoint, tokenizer, config):
    """CASE 6: BARE BACKBONE

    Load the bare model and return finite hidden states.
    """
    print("CASE 6: loading bare text model (bf16, CPU) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backbone = transformers.Apertus1p5TextModel.from_pretrained(checkpoint, dtype=torch.bfloat16).eval()
    token_ids = tokenizer("The Alps are", return_tensors="pt")
    with torch.no_grad():
        hidden_states = backbone(**token_ids).last_hidden_state
    expected = (1, token_ids["input_ids"].shape[1], config.hidden_size)
    assert hidden_states.shape == expected, f"expected hidden-state shape {expected}, got {hidden_states.shape}"
    assert bool(torch.isfinite(hidden_states.float()).all()), "hidden states contain non-finite values"
    return f"last_hidden_state {tuple(hidden_states.shape)}; finite"


def main():
    try:
        transformers, checkpoint, config, model, tokenizer, info = setup()
    except Exception as error:
        results = [setup_failure(error)]
    else:
        results = [
            run_case(case_1_config, config),
            run_case(case_2_clean_load, model, info),
            run_case(case_3_padded_logits, model, tokenizer),
            run_case(case_4_raw_generation, model, tokenizer),
            run_case(case_5_chat_generation, model, tokenizer),
        ]
        del model
        results.append(run_case(case_6_bare_backbone, transformers, checkpoint, tokenizer, config))
    return finish(results)


if __name__ == "__main__":
    raise SystemExit(main())
