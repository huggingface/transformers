"""Validate generation options and token scoring with a pruned output head.

Load the text causal LM from the composite checkpoint once on CPU (bf16). Run five generation
cases by default, or all 13 with --include-optional. Synthetic prompts include explicit input-only
IDs to exercise vocabulary boundaries, not language quality or media preprocessing. Assertions
check generated IDs and score correctness without requiring padded logits.
"""

import argparse
from functools import partial

import torch
from _common import bootstrap, finish, run_case, setup_failure, skipped_result


def setup():
    transformers, checkpoint = bootstrap(("Apertus1p5TextForCausalLM", "WatermarkingConfig"))
    print("SETUP: loading text causal LM (bf16, CPU) ...")
    model = transformers.Apertus1p5TextForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16).eval()
    head_size = model.lm_head.out_features
    assert 16 < head_size < model.get_input_embeddings().num_embeddings, (
        "needs a pruned vocabulary with text IDs 11-15"
    )
    return transformers, model


def generation_cases(transformers, input_only_id):
    """Explicit prompts make the input-only repetition and bigram conditions deterministic."""
    required = [
        ("GREEDY", [[11, 12, 13]], {}, None),
        ("BEAM SEARCH", [[11, 12, 13]], {"num_beams": 2}, None),
        ("BEAM SAMPLING", [[11, 12, 13]], {"num_beams": 2, "do_sample": True}, None),
        ("REPETITION PENALTY", [[11, input_only_id, 12]], {"repetition_penalty": 1.1}, None),
        ("NO-REPEAT BIGRAM", [[11, 12, input_only_id, 12]], {"no_repeat_ngram_size": 2}, None),
    ]
    optional = [
        ("ENCODER REPETITION", [[11, input_only_id, 12]], {"encoder_repetition_penalty": 1.1}, None),
        ("ENCODER BIGRAM", [[11, 12, input_only_id, 12]], {"encoder_no_repeat_ngram_size": 2}, None),
        ("WATERMARK", [[11, 12, 13]], {"watermarking_config": transformers.WatermarkingConfig()}, None),
        ("NORMALIZED SCORES", [[11, 12, 13]], {}, True),
        ("BATCHED SCORES", [[11, 12, 13], [11, 14, 15]], {}, False),
        ("BAD WORDS", [[11, 12, 13]], {"bad_words_ids": [[input_only_id]]}, None),
        ("SEQUENCE BIAS", [[11, 12, 13]], {"sequence_bias": {(input_only_id,): -2.0}}, None),
        ("SUPPRESSION", [[11, 12, 13]], {"suppress_tokens": [input_only_id]}, None),
    ]
    return [(name, ids, options, normalize, False) for name, ids, options, normalize in required] + [
        (name, ids, options, normalize, True) for name, ids, options, normalize in optional
    ]


def check_generation(model, ids, options, normalize):
    torch.manual_seed(0)
    input_ids = torch.tensor(ids, device=model.device)
    generation_kwargs = {"do_sample": False, **options}
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=2,
            eos_token_id=None,
            pad_token_id=0,
            return_dict_in_generate=True,
            output_scores=True,
            **generation_kwargs,
        )
    generated = output.sequences[:, input_ids.shape[1] :]
    assert tuple(generated.shape) == (len(ids), 2), f"unexpected generated shape: {tuple(generated.shape)}"
    assert bool(((generated >= 0) & (generated < model.lm_head.out_features)).all()), "generated an input-only ID"
    if normalize is not None:
        actual = model.compute_transition_scores(output.sequences, output.scores, normalize_logits=normalize)
        scores = torch.stack(output.scores, dim=1)
        if normalize:
            scores = scores.double().log_softmax(dim=-1)
        expected = scores.gather(-1, generated.unsqueeze(-1)).squeeze(-1)
        assert bool(torch.isfinite(actual).all()), "non-finite transition scores"
        if normalize:
            # Float32 softmax reductions over the 266k vocabulary accumulate rounding error.
            # Compare against a float64 reference with relative tolerance for that reduction.
            torch.testing.assert_close(actual.double(), expected, rtol=1e-4, atol=1e-5)
        else:
            torch.testing.assert_close(actual, expected)
        return f"{len(ids)} prompt(s); token scores match independent reference"
    return f"{len(ids)} prompt(s); 2 generated tokens each, all below {model.lm_head.out_features}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-optional", action="store_true", help="Run optional generation and scoring cases.")
    args = parser.parse_args()
    try:
        transformers, model = setup()
    except Exception as error:
        return finish([setup_failure(error)])

    results = []
    for index, (name, ids, options, normalize, optional) in enumerate(
        generation_cases(transformers, model.lm_head.out_features), 1
    ):
        case = partial(check_generation, model, ids, options, normalize)
        case.__doc__ = f"CASE {index}: {name}" + (" (OPTIONAL)" if optional else "")
        if optional and not args.include_optional:
            results.append(skipped_result(case, "use --include-optional to run"))
        else:
            results.append(run_case(case))
    return finish(results)


if __name__ == "__main__":
    raise SystemExit(main())
