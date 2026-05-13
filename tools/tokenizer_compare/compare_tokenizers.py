#!/usr/bin/env python3
"""Simple tokenizer comparison tool (tests-only).

Scans `tests/models/test_modeling_*.py` for `.from_pretrained(...)` usages and
compares tokenizers loaded via `AutoTokenizer.from_pretrained` vs
`TokenizersBackend.from_pretrained`.

Resolves literal strings and simple variable assignments inside each test file.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import constants as hf_constants
from transformers import AutoTokenizer
from transformers.tokenization_utils_tokenizers import TokenizersBackend


def collect_from_tests(tests_root: Path) -> dict[str, set[str]]:
    """Return mapping model_id -> set(of file paths where found)."""
    found: dict[str, set[str]] = {}
    for p in tests_root.rglob("test_modeling_*.py"):
        try:
            src = p.read_text(encoding="utf-8")
            tree = ast.parse(src, p.name)
        except Exception:
            continue
        assign_map: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                val = _str_from_node(node.value)
                for t in node.targets:
                    if isinstance(t, ast.Name) and val is not None:
                        assign_map[t.id] = val

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "from_pretrained":
                    file_str = str(p)
                    if node.args:
                        model = _str_from_node(node.args[0])
                        if isinstance(model, str) and "/" in model:
                            found.setdefault(model, set()).add(file_str)
                            continue
                        if isinstance(node.args[0], ast.Name):
                            name = node.args[0].id
                            if name in assign_map:
                                found.setdefault(assign_map[name], set()).add(file_str)
                                continue
                    for kw in node.keywords:
                        if kw.arg in ("pretrained_model_name_or_path", "pretrained", "repo_or_path"):
                            model = _str_from_node(kw.value)
                            if isinstance(model, str) and "/" in model:
                                found.setdefault(model, set()).add(file_str)
                            elif isinstance(kw.value, ast.Name):
                                name = kw.value.id
                                if name in assign_map:
                                    found.setdefault(assign_map[name], set()).add(file_str)
    return found


def _str_from_node(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if hasattr(ast, "Str") and isinstance(node, ast.Str):
        return node.s
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a = _str_from_node(node.left)
        b = _str_from_node(node.right)
        if a is not None and b is not None:
            return a + b
    return None


def _serialize_tokenizer(tok) -> dict:
    if tok is None:
        return {}
    def safe_getattr(o, name):
        try:
            return getattr(o, name, None)
        except Exception:
            return None

    def safe_repr(x):
        try:
            return repr(x)
        except Exception:
            return "<unreprable>"

    out = {
        "type": type(tok).__name__,
        "normalizer": safe_repr(safe_getattr(tok, "normalizer")),
        "pre_tokenizer": safe_repr(safe_getattr(tok, "pre_tokenizer")),
        "decoder": safe_repr(safe_getattr(tok, "decoder")),
        "post_processor": safe_repr(safe_getattr(tok, "post_processor")),
        "padding": safe_repr(safe_getattr(tok, "padding")),
        "truncation": safe_repr(safe_getattr(tok, "truncation")),
    }
    # vocab size if available
    try:
        out["vocab_size"] = tok.get_vocab_size()
    except Exception:
        out["vocab_size"] = None
    return out


def compare_tokenizers(model_id: str, found_in: list[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"model_id": model_id, "found_in": found_in or [], "errors": []}
    try:
        auto = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        result["errors"].append({"loader": "AutoTokenizer", "error": str(e)})
        return result
    try:
        backend = TokenizersBackend.from_pretrained(model_id)
    except Exception as e:
        result["errors"].append({"loader": "TokenizersBackend", "error": str(e)})
        return result

    a_tok = getattr(auto, "_tokenizer", None)
    b_tok = getattr(backend, "_tokenizer", None)

    result["tokenizer_auto"] = _serialize_tokenizer(a_tok)
    result["tokenizer_backend"] = _serialize_tokenizer(b_tok)

    # compute diff per field
    diffs = {}
    fields = ("type", "normalizer", "pre_tokenizer", "decoder", "post_processor", "padding", "truncation", "vocab_size")
    for f in fields:
        a_val = result["tokenizer_auto"].get(f)
        b_val = result["tokenizer_backend"].get(f)
        if a_val != b_val:
            diffs[f] = {"auto": a_val, "backend": b_val}

    result["match"] = len(diffs) == 0
    result["diff"] = diffs
    return result


def cleanup_cache(repo: str) -> None:
    home = Path(hf_constants.HF_HOME)
    pattern = f"models--{repo.replace('/','--')}"
    removed = 0
    if not home.exists():
        return
    for p in home.rglob(f"*{pattern}*"):
        if p.is_dir():
            try:
                shutil.rmtree(p)
                removed += 1
            except Exception:
                pass
    if removed:
        print(f"cleaned {removed} cache dirs for {repo}")


def main() -> int:
    script_dir = Path(__file__).resolve().parent

    p = argparse.ArgumentParser()
    p.add_argument("--models-file")
    p.add_argument("--models", nargs="*")
    p.add_argument("--output", default="tokenizer_compare_results.json")
    p.add_argument("--no-cleanup", action="store_true")
    p.add_argument("--list", action="store_true", help="Only list discovered model IDs and the files where they were found")
    args = p.parse_args()

    root = Path(".")
    models_map: dict[str, set[str]] = {}
    if args.models_file:
        for l in Path(args.models_file).read_text(encoding="utf-8").splitlines():
            v = l.strip()
            if not v:
                continue
            models_map.setdefault(v, set()).add("<models-file>")
    if args.models:
        for v in args.models:
            models_map.setdefault(v, set()).add("<cli>")
    # collect from tests only
    tests_map = collect_from_tests(root / "tests/models")
    for k, s in tests_map.items():
        models_map.setdefault(k, set()).update(s)

    if args.list:
        for m in sorted(models_map.keys()):
            files = ", ".join(sorted(models_map[m]))
            print(f"{m}  -> {files}")
        return 0

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = script_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    # load existing results and extract processed model IDs
    results = []
    processed: set[str] = set()
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text(encoding="utf-8"))
            processed = {r.get("model_id") for r in results if r.get("model_id")}
            print(f"Loaded {len(results)} existing results. Resuming from {len(processed)} processed models.")
        except Exception as e:
            print(f"Warning: failed to load existing results: {e}")

    models_to_process = sorted(set(models_map.keys()) - processed)
    for m in models_to_process:
        files = sorted(models_map[m])
        print(f"Checking: {m}  (found in: {', '.join(files)})")
        res = compare_tokenizers(m, found_in=files)
        results.append(res)
        # write incremental results atomically
        try:
            tmp_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            tmp_path.replace(out_path)
        except Exception as e:
            print("Warning: failed to write incremental results:", e)

        if not args.no_cleanup:
            try:
                cleanup_cache(m)
            except Exception as e:
                print("cleanup failed for", m, e)

    if not models_to_process:
        print("All models already processed. No new results to write.")
    else:
        print(f"Wrote {out_path} with {len(results)} total results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
