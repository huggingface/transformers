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
    python utils/check_tokenizers_backend_equivalence.py --model-type albert --model-type bert --limit 3

    # With XNLI token-ID regression (requires a v4 env):
    python utils/check_tokenizers_backend_equivalence.py --xnli --v4-python /tmp/transformers_v4_env/bin/python
    python utils/check_tokenizers_backend_equivalence.py --model-type xlm-roberta --xnli --v4-python /tmp/v4/bin/python --xnli-samples 100
"""
import re
import argparse
import difflib
import hashlib
import json
import logging
import os
import subprocess
import textwrap
import threading
import warnings
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

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
_v4_worker_lock = threading.Lock()


def _ensure_v4_worker() -> str:
    global _v4_worker_path
    with _v4_worker_lock:
        if _v4_worker_path is None:
            _v4_worker_path = "/tmp/v4_tokenizer_worker.py"
            with open(_v4_worker_path, "w") as f:
                f.write(_V4_WORKER_SCRIPT)
    return _v4_worker_path


# ---------------------------------------------------------------------------
# Diff helper — returns lines of markup (caller decides when to flush)
# ---------------------------------------------------------------------------

def _format_diff(diff_lines: list[str]) -> list[str]:
    """Convert raw unified-diff lines into Rich markup strings."""
    out = []
    for line in diff_lines:
        if line.startswith("+") and not line.startswith("+++"):
            out.append(f"[green]{line}[/]")
        elif line.startswith("-") and not line.startswith("---"):
            out.append(f"[red]{line}[/]")
        elif line.startswith("@@"):
            out.append(f"[cyan]{line}[/]")
        else:
            out.append(line)
    return out


def _print_diff(diff_lines: list[str]) -> None:
    for line in _format_diff(diff_lines):
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


def _normalize(d):
    """Recursively normalize values so that None and '' are treated as equivalent (both → None)."""
    if isinstance(d, dict):
        return {k: _normalize(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_normalize(i) for i in d]
    if d == "":
        return None
    return d


def _prepare(tok_obj):
    return _normalize(_strip_vocab(json.loads(tok_obj._tokenizer.to_str())))


def equal_tokenizers(t1, t2):
    """Compare two tokenizer objects by their JSON representation, ignoring vocab/merges."""
    return _prepare(t1) == _prepare(t2)


def diff_tokenizers(t1, t2, label_a="auto", label_b="backend") -> list:
    """Return unified diff lines between two tokenizer objects (vocab/merges stripped)."""
    j1 = json.dumps(_prepare(t1), indent=2).splitlines()
    j2 = json.dumps(_prepare(t2), indent=2).splitlines()
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


def has_non_tokenizer_json(model_id: str):
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


def print_diffs(spm_registry: dict):
    entries = list(spm_registry.values())
    if len(entries) < 2:
        return
    console.print("\n[bold blue]=== Diffs between distinct converted tokenizer.json files ===[/]")
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
                console.print(f"  [red][ERROR][/] Could not read files for diff ([cyan]{id_a}[/] vs [cyan]{id_b}[/]): {e}")
                continue
            diff = list(difflib.unified_diff(text_a, text_b, fromfile=id_a, tofile=id_b, lineterm=""))
            if diff:
                console.print(f"\n[bold]--- [cyan]{id_a}[/]  vs  [cyan]{id_b}[/] ---[/]")
                _print_diff(diff)
            else:
                console.print(f"\n[green][IDENTICAL (post-strip)][/] [cyan]{id_a}[/]  vs  [cyan]{id_b}[/]")


# ---------------------------------------------------------------------------
# XNLI token-ID regression
# ---------------------------------------------------------------------------

def _load_xnli_texts(lang: str, n: int) -> list[str]:
    """Return up to n*2 texts (premise + hypothesis) from XNLI test split."""
    from datasets import load_dataset

    ds = load_dataset("facebook/xnli", lang, split=f"test[:{n}]")
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
        return None, [f"    [red][XNLI][/] v4 subprocess error: {e}"]
    if result.returncode != 0:
        return None, [f"    [red][XNLI][/] v4 worker failed: {result.stderr[:200]}"]
    return json.loads(result.stdout)["ids"], []


def xnli_check_buffered(model_id: str, auto_tok, v4_python: str, n_samples: int, verbose: bool) -> tuple[bool, list[str]]:
    """
    Like xnli_check but returns (all_match, output_lines) instead of printing directly.
    """
    langs = ["en"]
    all_match = True
    out = []

    for lang in langs:
        texts = _load_xnli_texts(lang, n_samples)
        if not texts:
            if verbose:
                out.append(f"    [yellow dim][XNLI][/] lang={lang}: no data, skipping")
            continue

        try:
            v5_ids = auto_tok(texts, add_special_tokens=True)["input_ids"]
        except Exception as e:
            out.append(f"    [red][XNLI][/] lang={lang}: v5 tokenization error: {e}")
            all_match = False
            continue

        v4_ids, errs = _run_v4_tokenizer(v4_python, model_id, texts)
        out.extend(errs)
        if v4_ids is None:
            all_match = False
            continue

        mismatches = [
            (i, texts[i], v4_ids[i], v5_ids[i])
            for i in range(min(len(v4_ids), len(v5_ids)))
            if v4_ids[i] != v5_ids[i]
        ]
        n = len(texts)
        n_ok = n - len(mismatches)

        if not mismatches:
            if verbose:
                out.append(f"    [green][XNLI][/] lang={lang}  {n_ok}/{n} ✅")
        else:
            all_match = False
            out.append(f"    [bold red][XNLI][/] lang={lang}  {len(mismatches)} MISMATCH(es) / {n} ❌")
            for idx, text, a, b in mismatches[:3]:
                out.append(f"      [#{idx}] {text[:60]!r}")
                out.append(f"             v4: {a[:15]}")
                out.append(f"             v5: {b[:15]}")

    return all_match, out


# ---------------------------------------------------------------------------
# Hub fetch helper (runs in thread pool, Phase A)
# ---------------------------------------------------------------------------

def _fetch_model_list(model_type: str, limit: int) -> list:
    """Fetch and return up to `limit` models for the given model_type."""
    models = list(api.list_models(filter=model_type, sort="downloads"))
    return models[:limit]


# ---------------------------------------------------------------------------
# Per-model worker — runs in a thread, returns buffered output
# ---------------------------------------------------------------------------

def check_one_model(
    model_id: str,
    model_type: str,
    verbose: bool,
    spm_registry: dict,
    spm_lock: threading.Lock,
    output_dir: str,
    v4_python: str | None,
    xnli_samples: int,
) -> tuple[str | None, list[str]]:
    """
    Check a single model.  Returns (mismatch_id_or_None, output_lines).
    Nothing is printed directly — all output is buffered so the caller can
    flush it atomically (no interleaving between concurrent models).
    Only mismatches and (when verbose) errors/skips produce output.
    """
    out: list[str] = []
    mismatch: str | None = None

    try:
        auto_tok = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        if _is_gated_error(e):
            if verbose:
                out.append(f"  [yellow dim][SKIP][/] [cyan]{model_id}[/]: gated repo")
        elif verbose:
            out.append(f"  [yellow dim][SKIP][/] [cyan]{model_id}[/]: AutoTokenizer.from_pretrained failed: {e}")
        return mismatch, out

    if not hasattr(auto_tok, "_tokenizer") or auto_tok._tokenizer is None:
        if verbose:
            out.append(f"  [yellow dim][SKIP][/] [cyan]{model_id}[/]: no fast tokenizer (_tokenizer is None)")
        return mismatch, out

    try:
        backend_tok = TokenizersBackend.from_pretrained(model_id)
    except Exception as e:
        if _is_gated_error(e):
            if verbose:
                out.append(f"  [yellow dim][SKIP][/] [cyan]{model_id}[/]: gated repo")
        elif verbose:
            out.append(f"  [yellow dim][SKIP][/] [cyan]{model_id}[/]: TokenizersBackend.from_pretrained failed: {e}")
        return mismatch, out

    if backend_tok._tokenizer is None:
        if verbose:
            out.append(f"  [yellow dim][SKIP][/] [cyan]{model_id}[/]: TokenizersBackend._tokenizer is None")
        return mismatch, out

    try:
        diff_lines = diff_tokenizers(
            auto_tok,
            backend_tok,
            label_a=f"{model_id} (AutoTokenizer)",
            label_b=f"{model_id} (TokenizersBackend)",
        )
        if diff_lines:
            mismatch = model_id
            converter = get_converter_name(auto_tok)
            list_repo = set(list_repo_files(model_id))
            patt = re.compile(r"tokenizer\.json|sentencepiece\.model|spiece\.model|tokenizer\.model|tiktoken\.model", re.IGNORECASE)
            if "tokenizer.json" in list_repo and patt.search("".join(list_repo)) is not None:
                out.append(
                    f"\n[bold red]MISMATCH[/]: Auto-converted to `tokenizer.json` from "
                    f"{list_repo} for [cyan]{model_id}[/] => "
                    f"AutoConversion is wrong (maybe incorrect converter!) converter={converter}\n"
                )
            out.append(
                f"[bold red]MISMATCH[/]: No conversion -> Model should not map to `converter` "
                f"[cyan]{model_id}[/] model_type={model_type}  converter={converter}\n\n"
            )
            out.extend(_format_diff(diff_lines))
        else:
            xnli_ok = True
            if v4_python and xnli_samples:
                xnli_ok, xnli_out = xnli_check_buffered(model_id, auto_tok, v4_python, xnli_samples, verbose)
                out.extend(xnli_out)
                if not xnli_ok:
                    mismatch = model_id
                    out.append(f"  [bold red][XNLI MISMATCH][/] [cyan]{model_id}[/]")
    except Exception as e:
        if verbose:
            out.append(f"  [red][ERROR][/] [cyan]{model_id}[/]: comparison failed: {e}")

    return mismatch, out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Check TokenizersBackend equivalence across hub models."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        action="append",
        dest="model_type",
        default=None,
        help="Restrict to one or more model types (e.g. --model-type bert --model-type gpt2). Defaults to all types.",
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
        "--workers",
        type=int,
        default=32,
        help="Number of parallel worker threads (default: 32).",
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
        default=50,
        help="Number of XNLI examples per language to tokenize (default: 50).",
    )
    args = parser.parse_args()

    v4_python = None
    xnli_samples = 0
    if args.xnli:
        if not args.v4_python:
            parser.error("--xnli requires --v4-python")
        v4_python = args.v4_python
        xnli_samples = args.xnli_samples
        _ensure_v4_worker()  # write the script once before threads start

    if args.model_type:
        model_types = args.model_type
    else:
        seen = set()
        model_types = []
        for model_type in TOKENIZER_MAPPING_NAMES:
            if model_type is None:
                continue
            if model_type not in seen:
                seen.add(model_type)
                model_types.append(model_type)

    all_mismatches: dict[str, set[str]] = {}
    spm_registry: dict = {}
    spm_lock = threading.Lock()

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
            check_task = progress.add_task("[bold]models", total=None)

            check_total = 0

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                # Phase A: submit list_models fetches for all model types in parallel
                list_futures: dict[Future, str] = {
                    executor.submit(_fetch_model_list, model_type, args.limit): model_type
                    for model_type in model_types
                }

                check_futures: dict[Future, tuple[str, str]] = {}  # future → (model_id, model_type)

                # As each list arrives, immediately fan out model checks
                for list_future in as_completed(list_futures):
                    model_type = list_futures[list_future]
                    try:
                        models = list_future.result()
                    except Exception as e:
                        console.print(f"  [red][ERROR][/] list_models({model_type!r}): {e}")
                        progress.advance(outer_task)
                        continue

                    if not models:
                        if args.verbose:
                            console.print(f"  [yellow dim][INFO][/] No models found for model_type=[cyan]{model_type!r}[/]")
                        progress.advance(outer_task)
                        continue

                    check_total += len(models)
                    progress.update(check_task, total=check_total)

                    for model_info in models:
                        f = executor.submit(
                            check_one_model,
                            model_info.id,
                            model_type,
                            args.verbose,
                            spm_registry,
                            spm_lock,
                            args.output_dir,
                            v4_python,
                            xnli_samples,
                        )
                        check_futures[f] = (model_info.id, model_type)
                        f.add_done_callback(lambda _: progress.advance(check_task))

                    progress.advance(outer_task)

                # Phase B: drain model-check futures
                for future in as_completed(check_futures):
                    model_id, model_type = check_futures[future]
                    try:
                        mismatch, output_lines = future.result()
                    except Exception as e:
                        output_lines = [f"  [red][ERROR][/] [cyan]{model_id}[/]: unexpected exception: {e}"]
                        mismatch = None

                    for line in output_lines:
                        console.print(line)

                    if mismatch is not None:
                        all_mismatches.setdefault(model_type, set()).add(mismatch)

    finally:
        print_diffs(spm_registry)

        console.print()
        console.print(Rule("[bold]Summary[/]"))
        if not all_mismatches:
            console.print("[bold green]✓ No mismatches — all clear[/]")
        else:
            table = Table(title="Models to add to ignore list", show_header=True, header_style="bold magenta")
            table.add_column("Model ID", style="cyan")
            for model_type, mismatched_models in all_mismatches.items():
                table.add_row(f"{model_type} : {mismatched_models}")
            console.print(table)


if __name__ == "__main__":
    main()
