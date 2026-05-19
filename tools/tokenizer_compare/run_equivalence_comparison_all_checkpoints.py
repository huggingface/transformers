#!/usr/bin/env python3
"""Compare tokenizers loaded via AutoTokenizer vs TokenizersBackend.

Scans tests/models/test_modeling_*.py for .from_pretrained(...) usages,
compares the two backends field-by-field, writes results to JSON, and
generates an HTML diff report grouped by unique diff pattern.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from huggingface_hub import constants as hf_constants
from huggingface_hub.utils import disable_progress_bars
from tqdm import tqdm

from transformers import AutoTokenizer
from transformers.tokenization_utils_tokenizers import TokenizersBackend
from transformers.utils import logging as transformers_logging


def collect_from_tests(tests_root: Path) -> dict[str, set[str]]:
    """Return model_id -> set of file paths where it appears in from_pretrained() calls."""

    def str_from_node(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if hasattr(ast, "Str") and isinstance(node, ast.Str):
            return node.s
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            a, b = str_from_node(node.left), str_from_node(node.right)
            return a + b if a is not None and b is not None else None
        return None

    found: dict[str, set[str]] = {}
    for p in tests_root.rglob("test_modeling_*.py"):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"), p.name)
        except Exception:
            continue

        assign_map: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                val = str_from_node(node.value)
                if val is not None:
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            assign_map[t.id] = val

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_pretrained"
            ):
                continue
            model = None
            if node.args:
                arg = node.args[0]
                model = str_from_node(arg) or (assign_map.get(arg.id) if isinstance(arg, ast.Name) else None)
            if model is None:
                for kw in node.keywords:
                    if kw.arg in ("pretrained_model_name_or_path", "pretrained", "repo_or_path"):
                        model = str_from_node(kw.value) or (
                            assign_map.get(kw.value.id) if isinstance(kw.value, ast.Name) else None
                        )
                        break
            if isinstance(model, str) and "/" in model:
                found.setdefault(model, set()).add(str(p))
    return found


def should_ignore_diff(diff: dict) -> bool:
    """Return True if all differences are trivial ByteLevel flags, post_processor ordering, or vocab_size."""
    if not diff:
        return False
    
    # Skip if only difference is vocab_size
    if diff.keys() == {"vocab_size"}:
        return True
    
    if any(f not in ("decoder", "pre_tokenizer", "post_processor", "vocab_size") for f in diff):
        return False

    trivial = r"[,\s]*(add_prefix_space|trim_offsets|use_regex)=[^,\)]*"

    def normalize_post_processor(value: str) -> str:
        def sort_bracket(m):
            parts, cur, depth = [], "", 0
            for ch in m.group(1):
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                elif ch == "," and depth == 0:
                    parts.append(cur.strip())
                    cur = ""
                    continue
                cur += ch
            if cur.strip():
                parts.append(cur.strip())
            return "[" + ", ".join(sorted(parts)) + "]"

        return re.sub(r"\[(.*?)\](?=[,\)])", sort_bracket, value)

    for field, pair in diff.items():
        auto, backend = pair.get("auto", ""), pair.get("backend", "")
        if field in ("decoder", "pre_tokenizer"):
            if re.sub(trivial, "", auto) != re.sub(trivial, "", backend):
                return False
        elif normalize_post_processor(auto) != normalize_post_processor(backend):
            return False
    return True


def compare_tokenizers(model_id: str, found_in: list[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"model_id": model_id, "found_in": found_in or [], "errors": []}

    def serialize(tok) -> dict:
        if tok is None:
            return {}

        def safe_repr(attr):
            try:
                return repr(getattr(tok, attr, None))
            except Exception:
                return "<unrepr>"

        out = {
            k: safe_repr(k)
            for k in ("normalizer", "pre_tokenizer", "decoder", "post_processor", "padding", "truncation")
        }
        out["type"] = type(tok).__name__
        try:
            out["vocab_size"] = tok.get_vocab_size()
        except Exception:
            out["vocab_size"] = None
        return out

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

    a_info = serialize(getattr(auto, "_tokenizer", None))
    b_info = serialize(getattr(backend, "_tokenizer", None))
    result["tokenizer_auto"] = a_info
    result["tokenizer_backend"] = b_info

    fields = (
        "type",
        "normalizer",
        "pre_tokenizer",
        "decoder",
        "post_processor",
        "padding",
        "truncation",
        "vocab_size",
    )
    result["diff"] = {
        f: {"auto": a_info.get(f), "backend": b_info.get(f)} for f in fields if a_info.get(f) != b_info.get(f)
    }
    result["match"] = not result["diff"]
    return result


def cleanup_cache(repo: str) -> None:
    home = Path(hf_constants.HF_HOME)
    if not home.exists():
        return
    pattern = f"models--{repo.replace('/', '--')}"
    removed = 0
    for p in home.rglob(f"*{pattern}*"):
        if p.is_dir():
            try:
                shutil.rmtree(p)
                removed += 1
            except Exception:
                pass
    if removed:
        tqdm.write(f"cleaned {removed} cache dirs for {repo}")


_CSS = """
body { font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; padding:24px; background:#f8f9fa }
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:24px}
.card{background:#fff;border-radius:8px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.06);display:flex;flex-direction:column;min-width:0;max-height:520px}
pre.diff{background:#0f1720;color:#d8eefe;padding:12px;border-radius:6px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;font-size:0.8rem;flex:1;min-height:120px;max-height:280px}
.del{color:#ffb4a2}
.add{color:#a7f3d0}
.muted{color:#6b7280}
.models{font-size:0.85rem;margin-top:12px;max-height:180px;overflow:auto}
.models ul{margin:8px 0 0 20px;padding:0}
.models li{margin:4px 0}
.models a{overflow-wrap:anywhere;word-break:break-word}
.badge{display:inline-block;padding:3px 8px;border-radius:999px;background:#0d6efd;color:#fff;margin-left:8px;font-size:0.8rem}
h2{margin:0 0 12px 0;font-size:1rem}
@media(max-width:1200px){.grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:768px){.grid{grid-template-columns:1fr}}
"""


def make_report(data: list[dict], out_path: Path, ignore_useless_flags: bool = True) -> None:
    """Write an HTML diff report grouping models by identical diff pattern."""

    def truncate_charsmap(value: object, max_len: int = 220) -> str:
        text = "" if value is None else str(value)

        def repl(m: re.Match) -> str:
            s = m.group(1)
            if len(s) <= max_len:
                return m.group(0)
            return f'precompiled_charsmap="{s[:160]}...[{len(s) - 200} chars omitted]...{s[-40:]}"'

        return re.sub(r'precompiled_charsmap="([^"]*)"', repl, text)

    def diff_to_text(diff: dict) -> str:
        lines = []
        for field in sorted(diff):
            pair = diff[field]
            lines.append(f"-{field}:\t{truncate_charsmap(pair.get('auto'))}")
            lines.append(f"+{field}:\t{truncate_charsmap(pair.get('backend'))}")
        return "\n".join(lines)

    def colorize(diff_text: str) -> str:
        escaped = html.escape(diff_text)
        escaped = escaped.replace("-", '<span class="del">-', 1).replace("+", '</span><span class="add">+', 1)
        escaped = escaped.replace("\n-", '\n</span><span class="del">-')
        escaped = escaped.replace("\n+", '\n</span><span class="add">+')
        return escaped if escaped.endswith("</span>") else escaped + "</span>"

    mismatches = [
        d
        for d in data
        if not d.get("match", False)
        and d.get("diff")
        and (not ignore_useless_flags or not should_ignore_diff(d["diff"]))
    ]

    groups: dict[str, dict] = {}
    for entry in mismatches:
        key = json.dumps(entry["diff"], sort_keys=True, ensure_ascii=False)
        groups.setdefault(key, {"diff": entry["diff"], "models": []})
        groups[key]["models"].append({"model_id": entry.get("model_id"), "found_in": entry.get("found_in", [])})

    parts = [
        '<!doctype html><html><head><meta charset="utf-8"><title>Tokenizer diff report</title>',
        f"<style>{_CSS}</style></head><body>",
        "<h1>Tokenizer backend equivalence report</h1>",
        f'<p class="muted">Models checked: {len(data)} — mismatches: {len(mismatches)} — unique patterns: {len(groups)}</p>',
        '<div class="grid">',
    ]

    for idx, (_, info) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]["models"])), 1):
        diff_html = colorize(diff_to_text(info["diff"]) or "(empty)")
        parts += [
            f'<div class="card" id="pattern-{idx}"><h2>Pattern #{idx} <span class="badge">{len(info["models"])} models</span></h2>',
            f'<pre class="diff">{diff_html}</pre>',
            '<div class="models"><strong>Affected models:</strong><ul>',
        ]
        for m in sorted(info["models"], key=lambda x: x.get("model_id") or ""):
            mid = html.escape(m.get("model_id") or "")
            files = ", ".join(html.escape(f) for f in m.get("found_in", []))
            parts.append(
                f'<li><a href="https://huggingface.co/{mid}" target="_blank">{mid}</a> <span class="muted">{files}</span></li>'
            )
        parts.append("</ul></div></div>")

    parts.append("</div>\n</body></html>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    script_dir = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(description="Compare AutoTokenizer vs TokenizersBackend for HF models.")
    p.add_argument("--models-file", help="Text file with one model ID per line.")
    p.add_argument("--models", nargs="*", help="Model IDs to compare.")
    p.add_argument("--rerun-from-results", help="Path to a prior results JSON; only failed entries are rerun.")
    p.add_argument("--output", default="tokenizer_compare_results.json")
    p.add_argument("--no-cleanup", action="store_true", help="Keep downloaded model files in HF cache.")
    p.add_argument("--list", action="store_true", help="Print discovered model IDs and exit.")
    p.add_argument("--show-all", action="store_true", help="Include trivial flag-only diffs in the HTML report.")
    p.add_argument("--workers", type=int, default=1, help="Number of parallel download/compare workers (default: 1).")
    args = p.parse_args()

    models_map: dict[str, set[str]] = {}
    if args.models_file:
        for line in Path(args.models_file).read_text(encoding="utf-8").splitlines():
            if v := line.strip():
                models_map.setdefault(v, set()).add("<models-file>")
    if args.models:
        for v in args.models:
            models_map.setdefault(v, set()).add("<cli>")
    if args.rerun_from_results:
        for entry in json.loads(Path(args.rerun_from_results).read_text(encoding="utf-8")):
            if entry.get("match") is False and (mid := entry.get("model_id")):
                models_map.setdefault(mid, set()).update(entry.get("found_in") or ["<rerun>"])
    else:
        for k, s in collect_from_tests(Path("tests/models")).items():
            models_map.setdefault(k, set()).update(s)

    if args.list:
        for m in sorted(models_map):
            print(f"{m}  -> {', '.join(sorted(models_map[m]))}")
        return 0

    disable_progress_bars()
    transformers_logging.set_verbosity_error()

    out_path = script_dir / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    results, processed = [], set()
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text(encoding="utf-8"))
            processed = {r["model_id"] for r in results if r.get("model_id")}
            print(f"Resuming: {len(processed)} already done, {len(models_map) - len(processed)} remaining.")
        except Exception as e:
            print(f"Warning: could not load existing results: {e}")

    models_to_process = sorted(set(models_map) - processed)
    write_lock = threading.Lock()

    def process_one(m: str) -> None:
        result = compare_tokenizers(m, found_in=sorted(models_map[m]))
        with write_lock:
            results.append(result)
            try:
                tmp_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
                tmp_path.replace(out_path)
            except Exception as e:
                tqdm.write(f"Warning: failed to write results: {e}")
        if not args.no_cleanup:
            try:
                cleanup_cache(m)
            except Exception as e:
                tqdm.write(f"cleanup failed for {m}: {e}")

    pbar = tqdm(total=len(models_to_process), desc="Comparing tokenizers", unit="model")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, m): m for m in models_to_process}
        for future in as_completed(futures):
            m = futures[future]
            try:
                future.result()
            except Exception as e:
                tqdm.write(f"Error processing {m}: {e}")
            pbar.set_postfix_str(m, refresh=True)
            pbar.update(1)
    pbar.close()

    if models_to_process:
        print(f"Wrote {out_path} ({len(results)} total results)")
    else:
        print("Nothing new to process.")

    html_path = out_path.with_suffix(".html")
    make_report(results, html_path, ignore_useless_flags=not args.show_all)
    print("Wrote", html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
