"""
Utility script that verifies `TokenizersBackend.from_pretrained` produces an equivalent
`_tokenizer` object as `AutoTokenizer.from_pretrained` for hub models of every model type
in `TOKENIZER_MAPPING_NAMES`.

For models that lack a `tokenizer.json` on the hub, the script downloads the sentencepiece
model, converts it to a `tokenizer.json`, and saves both under a local output directory.
After all models are processed, unified diffs between distinct converted tokenizer.json
files are printed.

Usage:
    python utils/check_tokenizers_backend_equivalence.py
    python utils/check_tokenizers_backend_equivalence.py --model-type bert --limit 5
    python utils/check_tokenizers_backend_equivalence.py --model-type albert --limit 3 --verbose
    python utils/check_tokenizers_backend_equivalence.py --model-type glm --verbose --output-dir ./my_outputs
"""

import argparse
import difflib
import hashlib
import json
import logging
import os

from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from huggingface_hub.utils import disable_progress_bars

from transformers import AutoTokenizer
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, convert_slow_tokenizer
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
from transformers.tokenization_utils_tokenizers import TokenizersBackend


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

disable_progress_bars()

api = HfApi()

SPM_FILENAMES = ("sentencepiece.model", "spiece.model", "tokenizer.model")


def _strip_vocab(d):
    """Recursively remove 'vocab' and 'merges' keys (large, not structurally relevant)."""
    if isinstance(d, dict):
        return {k: _strip_vocab(v) for k, v in d.items() if k not in ("vocab", "merges", "precompiled_charsmap")}
    if isinstance(d, list):
        return [_strip_vocab(i) for i in d]
    return d


def equal_tokenizers(t1, t2):
    """Compare two tokenizer objects by their JSON representation, ignoring vocab/merges."""
    j1 = _strip_vocab(json.loads(t1._tokenizer.to_str()))
    j2 = _strip_vocab(json.loads(t2._tokenizer.to_str()))
    return j1 == j2


def diff_tokenizers(t1, t2, label_a="auto", label_b="backend") -> list:
    """Return unified diff lines between two tokenizer objects (vocab/merges stripped)."""
    j1 = json.dumps(_strip_vocab(json.loads(t1._tokenizer.to_str())), indent=2).splitlines()
    j2 = json.dumps(_strip_vocab(json.loads(t2._tokenizer.to_str())), indent=2).splitlines()
    return list(difflib.unified_diff(j1, j2, fromfile=label_a, tofile=label_b, lineterm=""))


def get_converter_name(tok) -> str:
    """
    Return the converter class name that would be used to convert this tokenizer,
    or a descriptive path label (Mistral/TikToken/SPM-extract) if no explicit converter exists.
    """
    cls_name = type(tok).__name__
    converter = SLOW_TO_FAST_CONVERTERS.get(cls_name)
    if converter is not None:
        return converter.__name__
    vocab_file = getattr(tok, "vocab_file", "") or ""
    if vocab_file.endswith("tekken.json"):
        return "MistralConverter"
    if vocab_file.endswith(".model"):
        return "SentencePieceExtractor"
    return "TikTokenConverter"


def has_tokenizer_json(model_id: str) -> bool:
    """Check whether the hub repo contains a tokenizer.json file."""
    try:
        files = list(list_repo_files(model_id))
        return "tokenizer.json" in files
    except Exception:
        return True  # assume exists on error; don't try to download


def find_spm_filename(model_id: str):
    """Return the sentencepiece filename in the repo, or None if not found."""
    try:
        files = set(list_repo_files(model_id))
    except Exception:
        return None
    for name in SPM_FILENAMES:
        if name in files:
            return name
    return None


def file_sha256(path: str) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def download_and_convert(model_id: str, spm_filename: str, output_dir: str, verbose: bool):
    """
    Downloads sentencepiece model, loads slow tokenizer, converts to tokenizers.Tokenizer,
    saves tokenizer.json under output_dir/<model_slug>/. Returns (spm_path, json_path) or
    (None, None) on failure.
    """
    slug = model_id.replace("/", "--")
    folder = os.path.join(output_dir, slug)
    os.makedirs(folder, exist_ok=True)

    # Download sentencepiece model
    try:
        spm_path = hf_hub_download(model_id, spm_filename, local_dir=folder)
    except Exception as e:
        if verbose:
            print(f"  [ERROR] {model_id}: hf_hub_download({spm_filename}) failed: {e}")
        return None, None

    # Load slow tokenizer and convert
    try:
        slow_tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        fast = convert_slow_tokenizer(slow_tok)
        out_path = os.path.join(folder, "tokenizer.json")
        fast.save(out_path)
        return spm_path, out_path
    except Exception as e:
        if verbose:
            print(f"  [ERROR] {model_id}: conversion failed: {e}")
        return spm_path, None


def print_diffs(spm_registry: dict):
    """Print unified diffs between every unique pair of converted tokenizer.json files."""
    entries = list(spm_registry.values())  # list of (model_id, json_path)
    if len(entries) < 2:
        return
    print("\n=== Diffs between distinct converted tokenizer.json files ===")
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            id_a, path_a = entries[i]
            id_b, path_b = entries[j]
            try:
                with open(path_a) as f:
                    text_a = json.dumps(_strip_vocab(json.load(f)), indent=2).splitlines()
                with open(path_b) as f:
                    text_b = json.dumps(_strip_vocab(json.load(f)), indent=2).splitlines()
            except Exception as e:
                print(f"  [ERROR] Could not read files for diff ({id_a} vs {id_b}): {e}")
                continue
            diff = list(difflib.unified_diff(text_a, text_b, fromfile=id_a, tofile=id_b, lineterm=""))
            if diff:
                print(f"\n--- {id_a}  vs  {id_b} ---")
                print("\n".join(diff))
            else:
                print(f"\n[IDENTICAL (post-strip)] {id_a}  vs  {id_b}")


def check_model_type(model_type, limit, verbose, spm_registry, output_dir):
    """
    Check all hub models for a given model_type.
    Returns list of model IDs where AutoTokenizer and TokenizersBackend diverge.
    Downloads and converts sentencepiece models for repos without tokenizer.json.
    """
    mismatches = []

    try:
        models = list(api.list_models(filter=model_type, sort="downloads", direction=-1))
    except Exception as e:
        print(f"  [ERROR] Failed to list models for model_type={model_type!r}: {e}")
        return mismatches

    models = models[:limit]

    if not models:
        if verbose:
            print(f"  [INFO] No models found for model_type={model_type!r}")
        return mismatches

    for model_info in models:
        model_id = model_info.id

        # Handle models without tokenizer.json: download SPM and convert
        if not has_tokenizer_json(model_id):
            spm_name = find_spm_filename(model_id)
            if spm_name:
                spm_path, json_path = download_and_convert(model_id, spm_name, output_dir, verbose)
                if spm_path and json_path:
                    spm_hash = file_sha256(spm_path)
                    if spm_hash not in spm_registry:
                        spm_registry[spm_hash] = (model_id, json_path)
                        print(f"  [CONVERTED] {model_id} → {json_path}")
                    else:
                        print(f"  [DUPLICATE SPM] {model_id} matches {spm_registry[spm_hash][0]}")
            elif verbose:
                print(f"  [SKIP] {model_id}: no tokenizer.json and no SPM file found")

        try:
            auto_tok = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            if verbose:
                print(f"  [SKIP] {model_id}: AutoTokenizer.from_pretrained failed: {e}")
            continue

        # Only compare if the auto tokenizer has a fast backend
        if not hasattr(auto_tok, "_tokenizer") or auto_tok._tokenizer is None:
            if verbose:
                print(f"  [SKIP] {model_id}: no fast tokenizer (_tokenizer is None)")
            continue

        try:
            backend_tok = TokenizersBackend.from_pretrained(model_id)
        except Exception as e:
            if verbose:
                print(f"  [SKIP] {model_id}: TokenizersBackend.from_pretrained failed: {e}")
            continue

        if backend_tok._tokenizer is None:
            if verbose:
                print(f"  [SKIP] {model_id}: TokenizersBackend._tokenizer is None")
            continue

        try:
            diff_lines = diff_tokenizers(
                auto_tok,
                backend_tok,
                label_a=f"{model_id} (AutoTokenizer)",
                label_b=f"{model_id} (TokenizersBackend)",
            )
            if diff_lines:
                mismatches.append(model_id)
                converter = get_converter_name(auto_tok)
                print(f"MISMATCH: {model_id}  [model_type={model_type}]  [converter={converter}]")
                print("\n".join(diff_lines))
            elif verbose:
                print(f"  [OK] {model_id}")
        except Exception as e:
            if verbose:
                print(f"  [ERROR] {model_id}: comparison failed: {e}")

    return mismatches


def main():
    parser = argparse.ArgumentParser(
        description="Check TokenizersBackend equivalence across hub models."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Restrict to a single model type (e.g. bert, gpt2). Defaults to all types.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max number of models to check per model type (default: 100).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print load errors and skipped models.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tokenizer_outputs",
        help="Local directory to write downloaded/converted files (default: ./tokenizer_outputs).",
    )
    args = parser.parse_args()

    # Collect model types to iterate
    if args.model_type:
        model_types = [args.model_type]
    else:
        seen = set()
        model_types = []
        for model_type in TOKENIZER_MAPPING_NAMES:
            if model_type is None:
                continue
            if model_type not in seen:
                seen.add(model_type)
                model_types.append(model_type)

    all_mismatches = []
    # Maps spm_hash -> (model_id, tokenizer_json_path) for unique SPM files
    spm_registry = {}

    for model_type in model_types:
        print(f"\n=== Checking model_type: {model_type} ===")
        mismatches = check_model_type(model_type, args.limit, args.verbose, spm_registry, args.output_dir)
        all_mismatches.extend(mismatches)

    # Print cross-model diffs for converted tokenizer.json files
    print_diffs(spm_registry)

    print("\n" + "=" * 60)
    print("--- Models to add to ignore list ---")
    if all_mismatches:
        for m in all_mismatches:
            print(f'    "{m}",')
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
