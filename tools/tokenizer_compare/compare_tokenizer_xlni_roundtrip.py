#!/usr/bin/env python
import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter

from datasets import load_dataset
from tqdm import tqdm

import transformers
from transformers import AutoTokenizer
#from transformers.tokenization_utils_tokenizers import TokenizersBackend

MODEL_CHECKPOINTS = [
    "almanach/camembert-base",
    "microsoft/mpnet-base",
    "google/rembert",
    "facebook/xglm-564M",
    "xlnet/xlnet-base-cased",
]


def _strip_separators_and_punct(text: str) -> str:
    # Keep only "letter-like" characters (drop whitespace, punctuation, control chars, etc.).
    return "".join(ch for ch in text if ch and unicodedata.category(ch)[0] not in {"P", "Z", "C"})


def _classify_roundtrip_diff(prev_decoded: str, curr_decoded: str) -> str:
    """
    Heuristic classification for roundtrip differences.

    Useful to quickly spot systemic issues like character-splitting with extra spaces.
    """
    if not prev_decoded and not curr_decoded:
        return "both-empty"

    prev_core = _strip_separators_and_punct(prev_decoded)
    curr_core = _strip_separators_and_punct(curr_decoded)

    if prev_core and prev_core == curr_core:
        # Same underlying letters, but something changed in whitespace / punctuation.
        prev_spaces = prev_decoded.count(" ")
        curr_spaces = curr_decoded.count(" ")
        if curr_spaces > prev_spaces * 2 and curr_spaces >= len(curr_decoded) // 4:
            return "character-splitting/extra-spaces"
        return "normalization-only"

    return "content-changed"


def _run_roundtrip_eval(tok, num_samples: int):
    results = []

    xnli_langs = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
    print("XNLI languages:", " ".join(f"[{lang}]" for lang in xnli_langs))

    # Pre-load datasets and compute total number of XNLI examples for a single global progress bar.
    xnli_datasets = {}
    xnli_total = 0
    for lang in xnli_langs:
        ds = load_dataset("facebook/xnli", lang, split="validation")
        limit = min(num_samples, len(ds))
        xnli_datasets[lang] = (ds, limit)
        xnli_total += limit

    xnli_pbar = tqdm(
        total=xnli_total,
        desc="XNLI",
        disable=not sys.stderr.isatty(),
    )

    for lang in xnli_langs:
        ds, limit = xnli_datasets[lang]
        for i, ex in enumerate(ds):
            if i >= limit:
                break
            for field in ("premise", "hypothesis"):
                text = ex.get(field, "")
                if not text:
                    continue
                ids = tok.encode(text, add_special_tokens=False)
                decoded = tok.decode(ids, skip_special_tokens=True)
                results.append(
                    {
                        "language": lang,
                        "index": i,
                        "field": field,
                        "text": text,
                        "ids": ids,
                        "decoded": decoded,
                        "roundtrip_ok": decoded == text,
                    }
                )
            xnli_pbar.update(1)

    xnli_pbar.close()

    try:
        code_ds = load_dataset("code_search_net", "python", split="train")
        limit = min(num_samples, len(code_ds))
        for i, ex in enumerate(
            tqdm(
                code_ds,
                total=limit,
                desc="Code [code_search_net/python]",
                disable=not sys.stderr.isatty(),
            )
        ):
            if i >= limit:
                break
            for field in ("func_code_string", "func_documentation_string", "code", "docstring"):
                text = ex.get(field)
                if not text:
                    continue
                ids = tok.encode(text, add_special_tokens=False)
                decoded = tok.decode(ids, skip_special_tokens=True)
                results.append(
                    {
                        "language": "code",
                        "source": "code_search_net/python",
                        "index": i,
                        "field": field,
                        "text": text,
                        "ids": ids,
                        "decoded": decoded,
                        "roundtrip_ok": decoded == text,
                    }
                )
    except Exception as e:
        print(f"Could not load code_search_net/python: {e}")

    return results


def _compare_backends(a_results, b_results, a_label: str, b_label: str):
    """
    Compare two backends sample-by-sample. A disagreement is any sample where
    the backends produce different token IDs or different decoded output.
    Roundtrip failures that are identical across both backends are NOT disagreements.
    Returns list of disagreement dicts.
    """
    n = min(len(a_results), len(b_results))
    if not n:
        print("No overlapping samples to compare.")
        return []

    print(f"\n--- Backend agreement: {a_label} vs {b_label} ---")
    print(f"Samples compared: {n}")

    disagreements = []
    ids_diff_langs = Counter()
    decoded_diff_langs = Counter()

    for i in range(n):
        a = a_results[i]
        b = b_results[i]
        if a.get("index") != b.get("index") or a.get("field") != b.get("field"):
            break

        ids_differ = a.get("ids") != b.get("ids")
        decoded_differs = a.get("decoded") != b.get("decoded")

        if not ids_differ and not decoded_differs:
            continue

        lang = a.get("language") or b.get("language") or "unknown"
        if ids_differ:
            ids_diff_langs[lang] += 1
        if decoded_differs:
            decoded_diff_langs[lang] += 1

        a_dec = a.get("decoded") or ""
        b_dec = b.get("decoded") or ""
        disagreements.append(
            {
                "language": lang,
                "source": a.get("source") or b.get("source"),
                "index": a.get("index"),
                "field": a.get("field"),
                "text": a.get("text", ""),
                "ids_differ": ids_differ,
                "decoded_differs": decoded_differs,
                f"{a_label}_ids": a.get("ids"),
                f"{b_label}_ids": b.get("ids"),
                f"{a_label}_decoded": a_dec,
                f"{b_label}_decoded": b_dec,
                "decoded_diff_pattern": _classify_roundtrip_diff(a_dec, b_dec) if decoded_differs else None,
            }
        )

    total_disagree = len(disagreements)
    print(f"Disagreements:    {total_disagree}/{n} ({100 * total_disagree / n:.1f}%)")
    print(f"  IDs differ:     {sum(d['ids_differ'] for d in disagreements)}")
    print(f"  Decoded differ: {sum(d['decoded_differs'] for d in disagreements)}")

    if ids_diff_langs:
        print("ID disagreements by language:")
        for lang, count in sorted(ids_diff_langs.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {lang}: {count}")

    if disagreements:
        print("\nFirst few disagreements:")
        for d in disagreements[:10]:
            text_preview = d["text"][:60] + ("..." if len(d["text"]) > 60 else "")
            print(f"  #{d['index']} [{d['language']}] field={d['field']}")
            print(f"    text         : {text_preview!r}")
            if d["ids_differ"]:
                a_ids = d[f"{a_label}_ids"] or []
                b_ids = d[f"{b_label}_ids"] or []
                print(f"    {a_label} ids: {a_ids[:20]}{'...' if len(a_ids) > 20 else ''} (len={len(a_ids)})")
                print(f"    {b_label} ids: {b_ids[:20]}{'...' if len(b_ids) > 20 else ''} (len={len(b_ids)})")
            if d["decoded_differs"]:
                a_dec = d[f"{a_label}_decoded"]
                b_dec = d[f"{b_label}_decoded"]
                max_len = max(len(a_dec), len(b_dec))
                pos = 0
                while pos < max_len and pos < len(a_dec) and pos < len(b_dec) and a_dec[pos] == b_dec[pos]:
                    pos += 1
                radius = 40
                start = max(0, pos - radius)
                end = min(max_len, pos + radius)
                marker = " " * max(0, pos - start) + "^"
                print(f"    {a_label} decoded: {a_dec[start:end]!r}")
                print(f"    {b_label} decoded: {b_dec[start:end]!r}")
                print(f"    diff_window     : {marker}")
                print(f"    pattern         : {d['decoded_diff_pattern']}")

    return disagreements


def _load_tokenizer(model_name: str, backend: str):
    if backend == "auto":
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    return TokenizersBackend.from_pretrained(model_name)


def _run_for_model(model_name: str, version_tag: str, args) -> dict:
    """Run roundtrip eval (and optional backend comparison) for one model. Returns an agreement entry."""
    safe_model = re.sub(r"[^0-9A-Za-z_.\-]+", "_", model_name.replace("/", "-").replace(" ", "_"))
    out_path = f"xlni_{safe_model}_{version_tag}"

    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}  [backend={args.backend}]")
    tok = _load_tokenizer(model_name, args.backend)
    print(f"  Loaded {type(tok).__name__}")

    if hasattr(tok, "_tokenizer") and tok._tokenizer is not None:
        with open(f"{out_path}_tokenizer.txt", "w", encoding="utf-8") as f:
            f.write(str(tok._tokenizer))

    results = _run_roundtrip_eval(tok, args.num_samples)

    disagreements = []
    if args.compare_backend:
        print(f"\nLoading comparison tokenizer: {model_name} [backend={args.compare_backend}]")
        tok_compare = _load_tokenizer(model_name, args.compare_backend)
        print(f"  Loaded {type(tok_compare).__name__}")

        compare_results = _run_roundtrip_eval(tok_compare, args.num_samples)
        disagreements = _compare_backends(
            results,
            compare_results,
            a_label=args.backend,
            b_label=args.compare_backend,
        )
    elif args.compare_to:
        with open(args.compare_to, "r", encoding="utf-8") as f:
            prev_payload = json.load(f)
        prev_results = prev_payload.get("results", [])
        _compare_backends(prev_results, results, a_label="saved", b_label=args.backend)

    payload = {
        "model_name": model_name,
        "model_name_sanitized": safe_model,
        "transformers_version": transformers.__version__,
        "num_samples": args.num_samples,
        "num_results": len(results),
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} roundtrip entries to '{out_path}'.")

    return {
        "model_name": model_name,
        "total_samples": len(results),
        "num_disagreements": len(disagreements),
        "disagreements": disagreements,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        type=str,
        nargs="?",
        default=None,
        help="Model checkpoint to evaluate. Omit to use --all-checkpoints.",
    )
    parser.add_argument("version_tag", type=str, nargs="?", default="v1")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--compare-to", type=str, default=None)
    parser.add_argument("--backend", type=str, choices=["auto", "tokenizers"], default="auto")
    parser.add_argument(
        "--compare-backend",
        type=str,
        choices=["auto", "tokenizers"],
        default=None,
        help="Compare roundtrip stats between two backends in a single run.",
    )
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Run eval for all MODEL_CHECKPOINTS instead of a single model.",
    )
    parser.add_argument(
        "--agreement-out",
        type=str,
        default="tools/tokenizer_compare/output/tokenizers_agreement.json",
        help="Output file for backend disagreements when --compare-backend is used.",
    )
    args = parser.parse_args()

    if args.compare_backend and args.compare_to:
        parser.error("--compare-backend and --compare-to cannot be used together.")

    if not args.all_checkpoints and not args.model_name:
        parser.error("Provide model_name or pass --all-checkpoints.")

    print(f"Transformers: {transformers.__version__}")

    checkpoints = MODEL_CHECKPOINTS if args.all_checkpoints else [args.model_name]
    version_tag = args.version_tag or "v1"

    all_agreement = []
    for model_name in checkpoints:
        entry = _run_for_model(model_name, version_tag, args)
        all_agreement.append(entry)

    if args.compare_backend:
        agreement_payload = {
            "transformers_version": transformers.__version__,
            "backend": args.backend,
            "compare_backend": args.compare_backend,
            "num_samples": args.num_samples,
            "models": all_agreement,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.agreement_out)), exist_ok=True)
        with open(args.agreement_out, "w", encoding="utf-8") as f:
            json.dump(agreement_payload, f, ensure_ascii=False, indent=2)
        total_disagreements = sum(e["num_disagreements"] for e in all_agreement)
        print(f"\nSaved {total_disagreements} disagreements across {len(checkpoints)} model(s) to '{args.agreement_out}'.")


if __name__ == "__main__":
    raise SystemExit(main())
