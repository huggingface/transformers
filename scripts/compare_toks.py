#compare toks

#compare toks

#!/usr/bin/env python
import argparse
import json
import re
import sys
import unicodedata
from collections import Counter

from datasets import load_dataset
from tqdm import tqdm

import transformers
from transformers import AutoTokenizer
from transformers.tokenization_utils_tokenizers import TokenizersBackend


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


def _compare_results(prev_results, curr_results, prev_label: str = "Prev", curr_label: str = "Curr"):
    n = min(len(prev_results), len(curr_results))
    if not n:
        print("No overlapping samples to compare.")
        return

    prev_ok = sum(1 for r in prev_results[:n] if r.get("roundtrip_ok"))
    curr_ok = sum(1 for r in curr_results[:n] if r.get("roundtrip_ok"))

    print("\n--- Roundtrip comparison ---")
    if prev_label != "Prev" or curr_label != "Curr":
        print(f"Prev label: {prev_label}")
        print(f"Curr label: {curr_label}")

    print(f"Samples compared: {n}")
    print(f"Prev roundtrip OK: {prev_ok}/{n} ({100 * prev_ok / n:.1f}%)")
    print(f"Curr roundtrip OK: {curr_ok}/{n} ({100 * curr_ok / n:.1f}%)")

    diffs = []
    diff_langs = Counter()
    diff_patterns = Counter()
    ids_diffs = []
    ids_diff_langs = Counter()
    for i in range(n):
        a = prev_results[i]
        b = curr_results[i]
        if a.get("index") != b.get("index") or a.get("field") != b.get("field"):
            break
        if a.get("roundtrip_ok") != b.get("roundtrip_ok") or a.get("decoded") != b.get("decoded"):
            prev_dec = a.get("decoded") or ""
            curr_dec = b.get("decoded") or ""
            pattern = _classify_roundtrip_diff(prev_dec, curr_dec)
            diffs.append((a, b, pattern))
            lang = a.get("language") or b.get("language") or "unknown"
            diff_langs[lang] += 1
            diff_patterns[pattern] += 1
        if a.get("ids") != b.get("ids"):
            lang = a.get("language") or b.get("language") or "unknown"
            ids_diffs.append((a, b))
            ids_diff_langs[lang] += 1

    print(f"Changed samples:   {len(diffs)}")
    if diff_langs:
        print("Changed by language:")
        for lang, count in sorted(diff_langs.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {lang}: {count}")
    if diff_patterns:
        print("Changed by pattern:")
        for label, count in sorted(diff_patterns.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {label}: {count}")
    if diffs:
        print("\nFirst few changed samples:")
        for a, b, pattern in diffs[:10]:
            print(f"  #{a['index']} prev_ok={a['roundtrip_ok']} curr_ok={b['roundtrip_ok']}")
            prev_dec = a.get("decoded") or ""
            curr_dec = b.get("decoded") or ""
            max_len = max(len(prev_dec), len(curr_dec))
            pos = 0
            while pos < max_len and pos < len(prev_dec) and pos < len(curr_dec) and prev_dec[pos] == curr_dec[pos]:
                pos += 1
            radius = 40
            start = max(0, pos - radius)
            end = min(max_len, pos + radius)
            prev_window = prev_dec[start:end]
            curr_window = curr_dec[start:end]
            marker = " " * max(0, pos - start) + "^"
            print(f"    prev_decoded: {prev_window!r}")
            print(f"    curr_decoded: {curr_window!r}")
            print(f"    diff_window : {marker}")
            print(f"    pattern     : {pattern}")

    print(f"\n--- Token ID comparison ---")
    print(f"Samples with different IDs: {len(ids_diffs)}/{n} ({100 * len(ids_diffs) / n:.1f}%)")
    if ids_diff_langs:
        print("ID diffs by language:")
        for lang, count in sorted(ids_diff_langs.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {lang}: {count}")
    if ids_diffs:
        print("\nFirst few ID-differing samples:")
        for a, b in ids_diffs[:10]:
            prev_ids = a.get("ids") or []
            curr_ids = b.get("ids") or []
            text = a.get("text", "")
            text_preview = text[:60] + ("..." if len(text) > 60 else "")
            print(f"  #{a['index']} [{a.get('language', '?')}] field={a.get('field', '?')}")
            print(f"    text    : {text_preview!r}")
            print(f"    prev ids: {prev_ids[:20]}{'...' if len(prev_ids) > 20 else ''} (len={len(prev_ids)})")
            print(f"    curr ids: {curr_ids[:20]}{'...' if len(curr_ids) > 20 else ''} (len={len(curr_ids)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("version_tag", type=str)
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
    args = parser.parse_args()

    if args.compare_backend and args.compare_to:
        parser.error("--compare-backend and --compare-to cannot be used together.")

    safe_model = re.sub(r"[^0-9A-Za-z_.\-]+", "_", args.model_name.replace("/", "-").replace(" ", "_"))
    out_path = f"xlni_{safe_model}_{args.version_tag}"

    print(f"Transformers: {transformers.__version__}")
    print(f"Loading tokenizer: {args.model_name} [backend={args.backend}]")
    if args.backend == "auto":
        tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
    else:
        tok = TokenizersBackend.from_pretrained(args.model_name)
    print(f"  Loaded {type(tok).__name__}")

    if hasattr(tok, "_tokenizer") and tok._tokenizer is not None:
        with open(f"{out_path}_tokenizer.txt", "w", encoding="utf-8") as f:
            f.write(str(tok._tokenizer))

    results = _run_roundtrip_eval(tok, args.num_samples)

    # Print roundtrip statistics
    total = len(results)
    successful = sum(1 for r in results if r.get("roundtrip_ok"))
    failed = total - successful

    print("\n--- Roundtrip Results ---")
    print(f"Total entries:      {total}")
    print(f"Successful:         {successful} ({100 * successful / total:.1f}%)")
    print(f"Failed:             {failed} ({100 * failed / total:.1f}%)")

    if failed > 0:
        print("\nFirst few failed roundtrips:")
        count = 0
        for r in results:
            if not r.get("roundtrip_ok") and count < 5:
                lang = r.get("language", "unknown")
                text = r.get("text", "")
                decoded = r.get("decoded", "")
                text_preview = text[:60] + ("..." if len(text) > 60 else "")
                decoded_preview = decoded[:60] + ("..." if len(decoded) > 60 else "")
                print(f"  [{lang}]")
                print(f"    Original: {text_preview!r}")
                print(f"    Decoded:  {decoded_preview!r}")
                count += 1

    if args.compare_backend:
        print(f"\nLoading comparison tokenizer: {args.model_name} [backend={args.compare_backend}]")
        if args.compare_backend == "auto":
            tok_compare = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
        else:
            tok_compare = TokenizersBackend.from_pretrained(args.model_name)
        print(f"  Loaded {type(tok_compare).__name__}")

        compare_results = _run_roundtrip_eval(tok_compare, args.num_samples)
        _compare_results(
            results,
            compare_results,
            prev_label=f"{args.backend} backend",
            curr_label=f"{args.compare_backend} backend",
        )
    elif args.compare_to:
        with open(args.compare_to, "r", encoding="utf-8") as f:
            prev_payload = json.load(f)
        prev_results = prev_payload.get("results", [])
        _compare_results(prev_results, results)

    payload = {
        "model_name": args.model_name,
        "model_name_sanitized": safe_model,
        "transformers_version": transformers.__version__,
        "num_samples": args.num_samples,
        "num_results": len(results),
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} roundtrip entries to '{out_path}'.")


if __name__ == "__main__":
    raise SystemExit(main())



