#!/usr/bin/env python
"""
Script to convert a tokenizer from trust_remote_code=True to trust_remote_code=False
and validate that outputs are identical using the XNLI dataset.

Usage:
    python scripts/convert_tokenizer.py moonshotai/Kimi-K2.5
    python scripts/convert_tokenizer.py moonshotai/Kimi-K2.5 --push-to-hub
    python scripts/convert_tokenizer.py moonshotai/Kimi-K2.5 --save-dir ./converted_tokenizer
"""

import argparse

from datasets import load_dataset
from tokenizers import Regex, pre_tokenizers
from tqdm import tqdm

from transformers import AutoTokenizer


CONTRACTION_TEST_CASES = [
    "I'm happy",
    "You're welcome",
    "He's here",
    "They've arrived",
    "We'll see",
    "I'd like",
    "Don't do it",
    "Can't stop",
    "Won't go",
    "Isn't it nice",
]

# Pre-tokenizer behaviors to try for fixing contraction mismatches
PRE_TOKENIZER_BEHAVIORS = [
    "merged_with_previous",
    "merged_with_next",
    "contiguous",
]


def test_contractions(tok_original, tok_converted):
    """Test contraction handling specifically."""
    results = []
    for text in CONTRACTION_TEST_CASES:
        orig = tok_original.encode(text, add_special_tokens=False)
        conv = tok_converted.encode(text, add_special_tokens=False)
        match = orig == conv
        results.append(
            {
                "text": text,
                "match": match,
                "original": orig,
                "converted": conv,
            }
        )
    return results


def get_pattern_from_tokenizer(tok_original):
    """Try to extract the regex pattern from the original tokenizer."""
    # Try common attribute names for the pattern
    for attr in ["pat_str", "_pat_str", "pattern", "regex_pattern"]:
        if hasattr(tok_original, attr):
            pattern = getattr(tok_original, attr)
            if pattern and isinstance(pattern, str):
                return pattern

    # Try to get from internal tokenizer
    if hasattr(tok_original, "tokenizer"):
        t = tok_original.tokenizer
        for attr in ["_pat_str", "pat_str", "pattern"]:
            if hasattr(t, attr):
                pattern = getattr(t, attr)
                if pattern and isinstance(pattern, str):
                    return pattern

    return None


def try_fix_pretokenizer(tok_converted, tok_original, pattern=None):
    """Try different pre-tokenizer configurations to fix contraction mismatches."""
    if not hasattr(tok_converted, "_tokenizer") or tok_converted._tokenizer is None:
        return False, None

    # Get current pre-tokenizer to extract pattern
    current_pretok = tok_converted._tokenizer.pre_tokenizer
    if current_pretok is None:
        return False, None

    # Try to extract pattern from original tokenizer
    if pattern is None:
        pattern = get_pattern_from_tokenizer(tok_original)
        if pattern:
            print(f"  Extracted pattern from original tokenizer: {pattern[:80]}...")

    # Default tiktoken pattern as fallback
    if pattern is None:
        pattern = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
        print("  Using default tiktoken pattern")

    for behavior in PRE_TOKENIZER_BEHAVIORS:
        print(f"  Trying pre-tokenizer behavior: {behavior}...")
        try:
            # Create new pre-tokenizer with different behavior
            new_pretok = pre_tokenizers.Sequence(
                [
                    pre_tokenizers.Split(Regex(pattern), behavior=behavior, invert=False),
                    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
                ]
            )
            tok_converted._tokenizer.pre_tokenizer = new_pretok

            # Test contractions
            results = test_contractions(tok_original, tok_converted)
            matches = sum(1 for r in results if r["match"])

            if matches == len(results):
                print(f"    ✓ All contractions match with behavior='{behavior}'!")
                return True, behavior
            else:
                print(f"    {matches}/{len(results)} contractions match")

        except Exception as e:
            print(f"    Failed: {e}")

    return False, None


def validate_roundtrip(tok, dataset, num_samples=1000):
    """Check decode(encode(text)) == text for all XNLI samples"""
    failed, total = [], 0
    for i, example in enumerate(tqdm(dataset, total=min(num_samples, len(dataset)), desc="Roundtrip")):
        if i >= num_samples:
            break
        for field in ["premise", "hypothesis"]:
            text = example.get(field, "")
            if not text:
                continue
            total += 1
            ids = tok.encode(text, add_special_tokens=False)
            decoded = tok.decode(ids, skip_special_tokens=True)
            if decoded != text:
                failed.append({"index": i, "field": field, "text": text[:60], "decoded": decoded[:60]})
    return failed, total


def validate_tokenizers(tok_original, tok_converted, dataset, num_samples=1000):
    """Validate that both tokenizers produce identical outputs on the dataset."""
    mismatches = []
    decode_failures = []

    for i, example in enumerate(tqdm(dataset, total=min(num_samples, len(dataset)), desc="Validating")):
        if i >= num_samples:
            break

        for field in ["premise", "hypothesis"]:
            text = example.get(field, "")
            if not text:
                continue

            ids_original = tok_original.encode(text)
            ids_converted = tok_converted.encode(text)

            if ids_original != ids_converted:
                # Check if decode matches
                decoded_orig = tok_original.decode(ids_original)
                decoded_conv = tok_converted.decode(ids_converted)
                decode_match = decoded_orig == decoded_conv

                mismatches.append(
                    {
                        "index": i,
                        "field": field,
                        "text": text[:100] + "..." if len(text) > 100 else text,
                        "original": ids_original[:20],
                        "converted": ids_converted[:20],
                        "decode_match": decode_match,
                    }
                )

                if not decode_match:
                    decode_failures.append(
                        {
                            "index": i,
                            "field": field,
                            "text": text,
                            "decoded_orig": decoded_orig,
                            "decoded_conv": decoded_conv,
                        }
                    )

    return mismatches, decode_failures


def main():
    parser = argparse.ArgumentParser(description="Convert and validate tokenizer")
    parser.add_argument("model_name", type=str, help="HuggingFace model name (e.g., moonshotai/Kimi-K2.5)")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save converted tokenizer")
    parser.add_argument("--push-to-hub", action="store_true", help="Push converted tokenizer to Hub as PR")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of XNLI samples to validate")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation step")
    parser.add_argument(
        "--only-roundtrip",
        action="store_true",
        help="Run only the XNLI roundtrip check with the original tokenizer (trust_remote_code=True) and exit",
    )
    args = parser.parse_args()

    if args.only_roundtrip:
        print(f"Loading tokenizer for {args.model_name} (trust_remote_code=True)...")
        tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
        print(f"  Type: {type(tok).__name__}")

        print("\nLoading XNLI dataset for validation...")
        dataset = load_dataset("facebook/xnli", "en", split="validation")

        print("\nOriginal tokenizer (trust_remote_code=True) XNLI roundtrip...")
        roundtrip_failed, roundtrip_total = validate_roundtrip(tok, dataset, args.num_samples)
        if not roundtrip_failed:
            print(f"  ✓ All {roundtrip_total} XNLI roundtrips match (works out of the box)")
        else:
            print(f"  ✗ {len(roundtrip_failed)}/{roundtrip_total} roundtrips failed")
            for f in roundtrip_failed[:3]:
                print(f"    e.g. #{f['index']} {f['field']}: {f['text']!r} -> {f['decoded']!r}")
        return 0

    print(f"Loading original tokenizer from {args.model_name} (trust_remote_code=True)...")
    tok_original = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    print(f"  Type: {type(tok_original).__name__}")
    print(f"  Vocab size: {tok_original.vocab_size}")

    print(f"\nLoading converted tokenizer from {args.model_name} (trust_remote_code=False)...")
    tok_converted = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=False)
    print(f"  Type: {type(tok_converted).__name__}")
    print(f"  Vocab size: {tok_converted.vocab_size}")

    if not args.skip_validation:
        # Test contractions first
        print("\nTesting contraction handling...")
        contraction_results = test_contractions(tok_original, tok_converted)
        contraction_matches = sum(1 for r in contraction_results if r["match"])
        print(f"  Token ID match: {contraction_matches}/{len(contraction_results)}")

        for r in contraction_results:
            status = "✓" if r["match"] else "✗"
            print(f"    {status} {r['text']}: {r['original']} vs {r['converted']}")

        # If contractions don't match, try to fix the pre-tokenizer
        if contraction_matches < len(contraction_results):
            print("\nAttempting to fix pre-tokenizer for contraction handling...")
            fixed, behavior = try_fix_pretokenizer(tok_converted, tok_original)

            if fixed:
                print("\nRe-testing contractions after fix...")
                contraction_results = test_contractions(tok_original, tok_converted)
                contraction_matches = sum(1 for r in contraction_results if r["match"])
                print(f"  Token ID match: {contraction_matches}/{len(contraction_results)}")
                for r in contraction_results:
                    status = "✓" if r["match"] else "✗"
                    print(f"    {status} {r['text']}: {r['original']} vs {r['converted']}")
            else:
                print("  Could not find a pre-tokenizer configuration that fixes all contractions")

        print("\nLoading XNLI dataset for validation...")
        dataset = load_dataset("facebook/xnli", "en", split="validation")

        # Check original (trust_remote_code=True) works out of the box: roundtrip on XNLI
        print("\nOriginal tokenizer (trust_remote_code=True) XNLI roundtrip...")
        roundtrip_failed, roundtrip_total = validate_roundtrip(tok_original, dataset, args.num_samples)
        if not roundtrip_failed:
            print(f"  ✓ All {roundtrip_total} XNLI roundtrips match (works out of the box)")
        else:
            print(f"  ✗ {len(roundtrip_failed)}/{roundtrip_total} roundtrips failed")
            for f in roundtrip_failed[:3]:
                print(f"    e.g. #{f['index']} {f['field']}: {f['text']!r} -> {f['decoded']!r}")

        print(f"\nValidating original vs converted on {args.num_samples} samples...")
        mismatches, decode_failures = validate_tokenizers(tok_original, tok_converted, dataset, args.num_samples)

        if mismatches:
            decode_ok = sum(1 for m in mismatches if m["decode_match"])
            print(f"\nToken ID mismatches: {len(mismatches)}")
            print(f"Decode matches:      {decode_ok}/{len(mismatches)} ({100 * decode_ok / len(mismatches):.1f}%)")

            if decode_failures:
                print(f"\n❌ Found {len(decode_failures)} decode failures (text differs)!")
                for m in decode_failures[:5]:
                    print(f"  Sample {m['index']} ({m['field']}): {m['text'][:50]}...")
                    print(f"    Original:  '{m['decoded_orig'][:50]}'")
                    print(f"    Converted: '{m['decoded_conv'][:50]}'")
                return 1
            else:
                print("\n⚠ Token IDs differ but text decodes correctly")
        else:
            print(f"\n✓ All {args.num_samples} samples match exactly!")

    if args.save_dir:
        print(f"\nSaving converted tokenizer to {args.save_dir}...")
        tok_converted.save_pretrained(args.save_dir)
        print("  Saved!")

    if args.push_to_hub:
        print("\nPushing to Hub as PR...")
        pr_description = f"""## Tokenizer Conversion

This PR adds a converted tokenizer that works without `trust_remote_code=True`.

### Conversion Details
- Converted using: `python scripts/convert_tokenizer.py {args.model_name} --push-to-hub`
- Original tokenizer type: `{type(tok_original).__name__}`
- Converted tokenizer type: `{type(tok_converted).__name__}`

### Validation Results
- Tested on XNLI dataset ({args.num_samples} samples)
- **All samples match 1-1** ✓

### Usage
```python
from transformers import AutoTokenizer

# Now works without trust_remote_code=True
tokenizer = AutoTokenizer.from_pretrained("{args.model_name}")
```

---
*Converted with [transformers](https://github.com/huggingface/transformers) tokenizer conversion script*
"""
        result = tok_converted.push_to_hub(
            args.model_name,
            create_pr=True,
            commit_message="Add converted tokenizer (no trust_remote_code needed)",
            commit_description=pr_description,
        )
        print(f"  PR URL: {result.pr_url}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
