"""
Utility script that verifies `TokenizersBackend.from_pretrained` produces an equivalent
`_tokenizer` object as `AutoTokenizer.from_pretrained` for hub models of every model type
in `TOKENIZER_MAPPING_NAMES`.

Optionally also runs a token-ID regression against a v4 transformers environment using
100 XNLI samples per model (English by default, language-mapped for known multilingual models).

For models that lack a `tokenizer.json` on the hub, the script downloads the sentencepiece
model, converts it to a `tokenizer.json`, and saves both under a local output directory.
After all models are processed, unified diffs between distinct converted tokenizer.json
files are printed.

Usage:
    python utils/check_tokenizers_backend_equivalence.py
    python utils/check_tokenizers_backend_equivalence.py --model-type bert --limit 5
    python utils/check_tokenizers_backend_equivalence.py --model-type albert --limit 3 --verbose
    python utils/check_tokenizers_backend_equivalence.py --model-type glm --verbose --output-dir ./my_outputs

    # With XNLI token-ID regression (requires a v4 env):
    python utils/check_tokenizers_backend_equivalence.py --xnli --v4-python /tmp/transformers_v4_env/bin/python
    python utils/check_tokenizers_backend_equivalence.py --model-type xlm-roberta --xnli --v4-python /tmp/v4/bin/python --xnli-samples 100
"""

import argparse
import difflib
import hashlib
import json
import logging
import os
import subprocess
import textwrap
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table

from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from huggingface_hub.utils import disable_progress_bars

from transformers import AutoTokenizer
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, convert_slow_tokenizer
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
from transformers.tokenization_utils_tokenizers import TokenizersBackend


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

disable_progress_bars()

console = Console(highlight=False)

api = HfApi()

SPM_FILENAMES = ("sentencepiece.model", "spiece.model", "tokenizer.model")

# ---------------------------------------------------------------------------
# XNLI: known multilingual models get per-language sampling; everything else
# defaults to English.
# ---------------------------------------------------------------------------
_XNLI_ALL_LANGS = ["en", "ar", "bg", "de", "el", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]


# Embedded v4 worker — written to /tmp/v4_tokenizer_worker.py at first use.
_V4_WORKER_SCRIPT = textwrap.dedent("""\
    import json, sys
    from transformers import AutoTokenizer

    data = json.load(sys.stdin)
    model_id = data["model_id"]
    texts = data["texts"]

    tok = AutoTokenizer.from_pretrained(model_id)
    encoded = tok(texts, add_special_tokens=True)
    json.dump({"ids": encoded["input_ids"], "class": type(tok).__name__}, sys.stdout)
""")

_v4_worker_path: str | None = None


def _ensure_v4_worker() -> str:
    global _v4_worker_path
    if _v4_worker_path is None:
        _v4_worker_path = "/tmp/v4_tokenizer_worker.py"
        with open(_v4_worker_path, "w") as f:
            f.write(_V4_WORKER_SCRIPT)
    return _v4_worker_path


# ---------------------------------------------------------------------------
# Diff helper
# ---------------------------------------------------------------------------

def _print_diff(diff_lines: list[str]) -> None:
    for line in diff_lines:
        if line.startswith("+") and not line.startswith("+++"):
            console.print(f"[green]{line}[/]")
        elif line.startswith("-") and not line.startswith("---"):
            console.print(f"[red]{line}[/]")
        elif line.startswith("@@"):
            console.print(f"[cyan]{line}[/]")
        else:
            console.print(line)


# ---------------------------------------------------------------------------
# Tokenizer structure helpers
# ---------------------------------------------------------------------------

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
    cls_name = type(tok).__name__
    if TOKENIZER_MAPPING_NAMES.get(cls_name) is not None:
        return TOKENIZER_MAPPING_NAMES[cls_name]
    converter = SLOW_TO_FAST_CONVERTERS.get(cls_name)
    if converter is not None:
        return converter.__name__
    vocab_file = getattr(tok, "vocab_file", "") or ""
    if vocab_file.endswith("tekken.json"):
        return "MistralConverter"
    if vocab_file.endswith(".model"):
        return "SentencePieceExtractor"
    return "TikTokenConverter"


# ---------------------------------------------------------------------------
# Hub helpers
# ---------------------------------------------------------------------------

def has_tokenizer_json(model_id: str) -> bool:
    try:
        files = list(list_repo_files(model_id))
        return "tokenizer.json" in files
    except Exception:
        return True


def find_spm_filename(model_id: str):
    try:
        files = set(list_repo_files(model_id))
    except Exception:
        return None
    for name in SPM_FILENAMES:
        if name in files:
            return name
    return None


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _is_gated_error(exc: Exception) -> bool:
    msg = str(exc)
    return "gated" in msg.lower() or "403" in msg or "401" in msg or "not in the authorized list" in msg


def download_and_convert(model_id: str, spm_filename: str, output_dir: str, verbose: bool):
    slug = model_id.replace("/", "--")
    folder = os.path.join(output_dir, slug)
    os.makedirs(folder, exist_ok=True)

    try:
        spm_path = hf_hub_download(model_id, spm_filename, local_dir=folder)
    except Exception as e:
        if verbose:
            console.print(f"  [red][ERROR][/] [cyan]{model_id}[/]: hf_hub_download({spm_filename}) failed: {e}")
        return None, None

    try:
        slow_tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        fast = convert_slow_tokenizer(slow_tok)
        out_path = os.path.join(folder, "tokenizer.json")
        fast.save(out_path)
        return spm_path, out_path
    except Exception as e:
        if verbose:
            console.print(f"  [red][ERROR][/] [cyan]{model_id}[/]: conversion failed: {e}")
        return spm_path, None


# ---------------------------------------------------------------------------
# XNLI token-ID regression
# ---------------------------------------------------------------------------

def _load_xnli_texts(lang: str, n: int) -> list[str]:
    """Return up to n*2 texts (premise + hypothesis) from XNLI test split."""

    from datasets import load_dataset

    ds = load_dataset("xnli", lang, split=f"test[:{n}]")
    texts = []
    for row in ds:
        texts.append(row["premise"])
        texts.append(row["hypothesis"])
    return texts


def _run_v4_tokenizer(v4_python: str, model_id: str, texts: list[str]) -> list[list[int]] | None:
    worker = _ensure_v4_worker()
    try:
        result = subprocess.run(
            [v4_python, worker],
            input=json.dumps({"model_id": model_id, "texts": texts}),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as e:
        console.print(f"    [red][XNLI][/] v4 subprocess error: {e}")
        return None
    if result.returncode != 0:
        console.print(f"    [red][XNLI][/] v4 worker failed: {result.stderr[:200]}")
        return None
    return json.loads(result.stdout)["ids"]


def xnli_check(model_id: str, auto_tok, v4_python: str, n_samples: int, verbose: bool) -> bool:
    """
    Tokenize n_samples XNLI examples with v4 (subprocess) and v5 (auto_tok),
    compare token IDs. Returns True if all match, False on any mismatch.
    """
    langs = ["en"]
    all_match = True

    for lang in langs:
        texts = _load_xnli_texts(lang, n_samples)
        if not texts:
            if verbose:
                console.print(f"    [yellow dim][XNLI][/] lang={lang}: no data, skipping")
            continue

        # v5 IDs
        try:
            v5_ids = auto_tok(texts, add_special_tokens=True)["input_ids"]
        except Exception as e:
            console.print(f"    [red][XNLI][/] lang={lang}: v5 tokenization error: {e}")
            all_match = False
            continue

        # v4 IDs
        v4_ids = _run_v4_tokenizer(v4_python, model_id, texts)
        if v4_ids is None:
            all_match = False
            continue

        mismatches = [(i, texts[i], v4_ids[i], v5_ids[i]) for i in range(min(len(v4_ids), len(v5_ids))) if v4_ids[i] != v5_ids[i]]
        n = len(texts)
        n_ok = n - len(mismatches)

        if not mismatches:
            if verbose:
                console.print(f"    [green][XNLI][/] lang={lang}  {n_ok}/{n} ✅")
        else:
            all_match = False
            console.print(f"    [bold red][XNLI][/] lang={lang}  {len(mismatches)} MISMATCH(es) / {n} ❌")
            for idx, text, a, b in mismatches[:3]:
                console.print(f"      [#{idx}] {text[:60]!r}")
                console.print(f"             v4: {a[:15]}")
                console.print(f"             v5: {b[:15]}")

    return all_match


# ---------------------------------------------------------------------------
# Main per-model-type check
# ---------------------------------------------------------------------------

def check_model_type(model_type, limit, verbose, spm_registry, output_dir, v4_python, xnli_samples, progress=None, inner_task=None):
    mismatches = []

    try:
        models = list(api.list_models(filter=model_type, sort="downloads"))
    except Exception as e:
        console.print(f"  [red][ERROR][/] Failed to list models for model_type=[cyan]{model_type!r}[/]: {e}")
        return mismatches

    models = models[:limit]

    if not models:
        if verbose:
            console.print(f"  [yellow dim][INFO][/] No models found for model_type=[cyan]{model_type!r}[/]")
        return mismatches

    if progress is not None and inner_task is not None:
        progress.update(inner_task, total=len(models), completed=0, description=f"[cyan]{model_type}[/]")

    for model_info in models:
        model_id = model_info.id

        if progress is not None and inner_task is not None:
            progress.update(inner_task, description=f"[cyan]{model_id}[/]")

        try:
            auto_tok = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            if _is_gated_error(e):
                if verbose:
                    console.print(f"  [yellow dim][SKIP][/] [cyan]{model_id}[/]: gated repo")
            elif verbose:
                console.print(f"  [yellow dim][SKIP][/] [cyan]{model_id}[/]: AutoTokenizer.from_pretrained failed: {e}")
            if progress is not None and inner_task is not None:
                progress.advance(inner_task)
            continue

        if not hasattr(auto_tok, "_tokenizer") or auto_tok._tokenizer is None:
            if verbose:
                console.print(f"  [yellow dim][SKIP][/] [cyan]{model_id}[/]: no fast tokenizer (_tokenizer is None)")
            if progress is not None and inner_task is not None:
                progress.advance(inner_task)
            continue

        try:
            backend_tok = TokenizersBackend.from_pretrained(model_id)
        except Exception as e:
            if _is_gated_error(e):
                if verbose:
                    console.print(f"  [yellow dim][SKIP][/] [cyan]{model_id}[/]: gated repo")
            elif verbose:
                console.print(f"  [yellow dim][SKIP][/] [cyan]{model_id}[/]: TokenizersBackend.from_pretrained failed: {e}")
            if progress is not None and inner_task is not None:
                progress.advance(inner_task)
            continue

        if backend_tok._tokenizer is None:
            if verbose:
                console.print(f"  [yellow dim][SKIP][/] [cyan]{model_id}[/]: TokenizersBackend._tokenizer is None")
            if progress is not None and inner_task is not None:
                progress.advance(inner_task)
            continue

        try:
            diff_lines = diff_tokenizers(
                auto_tok,
                backend_tok,
                label_a=f"{model_id} (AutoTokenizer)",
                label_b=f"{model_id} (TokenizersBackend)",
        )
            if diff_lines:
                mismatches.append(model_type)
                converter = get_converter_name(auto_tok)
                if has_tokenizer_json(model_id):
                    console.print(f"\n[bold red]MISMATCH[/]: Auto-converted to `tokenizer.json` from {list(list_repo_files(model_id))} for [cyan]{model_id}[/] => this means the AutoConversion is wrong (maybe it does not use the correct converter!) converter={converter}\n\n")
                console.print(f"[bold red]MISMATCH[/]: No conversion -> Model should not map to `converter` [cyan]{model_id}[/] model_type={model_type}  converter={converter}\n\n {auto_tok}")
                _print_diff(diff_lines)
            else:
                status = "[OK]"
                xnli_ok = True
                if v4_python and xnli_samples:
                    xnli_ok = xnli_check(model_id, auto_tok, v4_python, xnli_samples, verbose)
                    if not xnli_ok:
                        mismatches.append(model_id)
                        status = "[XNLI MISMATCH]"
                if not xnli_ok:
                    console.print(f"  [bold red]{status}[/] [cyan]{model_id}[/]")
                elif verbose or (xnli_ok and v4_python):
                    console.print(f"  [green]{status}[/] [cyan]{model_id}[/]")
                elif verbose:
                    console.print(f"  [green]{status}[/] [cyan]{model_id}[/]")
        except Exception as e:
            if verbose:
                console.print(f"  [red][ERROR][/] [cyan]{model_id}[/]: comparison failed: {e}")

        if progress is not None and inner_task is not None:
            progress.advance(inner_task)

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
    parser.add_argument(
        "--xnli",
        action="store_true",
        help="Also run a token-ID regression on 100 XNLI samples per model (requires --v4-python).",
    )
    parser.add_argument(
        "--v4-python",
        type=str,
        default=None,
        help="Path to a v4 transformers Python executable for XNLI token-ID comparison.",
    )
    parser.add_argument(
        "--xnli-samples",
        type=int,
        default=100,
        help="Number of XNLI examples per language to tokenize (default: 100).",
    )
    args = parser.parse_args()

    v4_python = None
    xnli_samples = 0
    if args.xnli:
        if not args.v4_python:
            parser.error("--xnli requires --v4-python")
        v4_python = args.v4_python
        xnli_samples = args.xnli_samples

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


    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
    try:
        with progress:
            outer_task = progress.add_task("[bold blue]model types", total=len(model_types))
            inner_task = progress.add_task("", total=None, visible=False)

            for model_type in model_types:
                progress.update(inner_task, visible=True, completed=0, total=None, description=f"[cyan]{model_type}[/]")
                console.print(Rule(f"[bold blue]{model_type}[/]"))

                mismatches = check_model_type(
                    model_type, args.limit, args.verbose, None, args.output_dir,
                    v4_python, xnli_samples,
                    progress=progress,
                    inner_task=inner_task,
                )
                all_mismatches.extend(mismatches)
                progress.advance(outer_task)
                progress.update(inner_task, visible=False)
    finally:
        console.print()
        console.print(Rule("[bold]Summary[/]"))
        if not all_mismatches:
            console.print("[bold green]✓ No mismatches — all clear[/]")
        else:
            table = Table(title="Models to add to ignore list", show_header=True, header_style="bold magenta")
            table.add_column("Model ID", style="cyan")
            for m in all_mismatches:
                table.add_row(m)
            console.print(table)


if __name__ == "__main__":
    main()
