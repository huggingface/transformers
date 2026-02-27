#!/usr/bin/env python
import argparse
import json
import re

from datasets import load_dataset
from tqdm import tqdm

import transformers
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("version_tag", type=str)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--compare-to", type=str, default=None)
    args = parser.parse_args()

    safe_model = re.sub(r"[^0-9A-Za-z_.\-]+", "_", args.model_name.replace("/", "-").replace(" ", "_"))
    out_path = f"xlni_{safe_model}_{args.version_tag}"

    print(f"Transformers: {transformers.__version__}")
    print(f"Loading tokenizer: {args.model_name} (trust_remote_code=True, use_fast=True)")
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
    print(f"  Loaded {type(tok).__name__}")

    if hasattr(tok, "_tokenizer") and tok._tokenizer is not None:
        with open(f"{out_path}_tokenizer.txt", "w", encoding="utf-8") as f:
            f.write(str(tok._tokenizer))

    results = []

    for lang in ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]:
        ds = load_dataset("facebook/xnli", lang, split="validation")
        limit = min(args.num_samples, len(ds))
        for i, ex in enumerate(tqdm(ds, total=limit, desc=f"XNLI [{lang}]")):
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

    try:
        code_ds = load_dataset("code_search_net", "python", split="train")
        limit = min(args.num_samples, len(code_ds))
        for i, ex in enumerate(tqdm(code_ds, total=limit, desc="Code [code_search_net/python]")):
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

    if args.compare_to:
        with open(args.compare_to, "r", encoding="utf-8") as f:
            prev_payload = json.load(f)
        prev_results = prev_payload.get("results", [])
        n = min(len(prev_results), len(results))
        if n:
            prev_ok = sum(1 for r in prev_results[:n] if r.get("roundtrip_ok"))
            curr_ok = sum(1 for r in results[:n] if r.get("roundtrip_ok"))
            print("\n--- Roundtrip comparison ---")
            print(f"Samples compared: {n}")
            print(f"Prev roundtrip OK: {prev_ok}/{n} ({100 * prev_ok / n:.1f}%)")
            print(f"Curr roundtrip OK: {curr_ok}/{n} ({100 * curr_ok / n:.1f}%)")
            diffs = []
            for i in range(n):
                a = prev_results[i]
                b = results[i]
                if a.get("index") != b.get("index") or a.get("field") != b.get("field"):
                    break
                if a.get("roundtrip_ok") != b.get("roundtrip_ok") or a.get("decoded") != b.get("decoded"):
                    diffs.append((a, b))
            print(f"Changed samples:   {len(diffs)}")
            if diffs:
                print("\nFirst few changed samples:")
                for a, b in diffs[:10]:
                    print(f"  #{a['index']} prev_ok={a['roundtrip_ok']} curr_ok={b['roundtrip_ok']}")
                    prev_dec = a.get("decoded") or ""
                    curr_dec = b.get("decoded") or ""
                    max_len = max(len(prev_dec), len(curr_dec))
                    pos = 0
                    while (
                        pos < max_len
                        and pos < len(prev_dec)
                        and pos < len(curr_dec)
                        and prev_dec[pos] == curr_dec[pos]
                    ):
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

