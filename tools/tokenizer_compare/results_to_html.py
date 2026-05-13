#!/usr/bin/env python3
"""Generate an HTML report grouping tokenizer mismatches by diff pattern.

Reads the JSON produced by `compare_tokenizers.py` (default
`tokenizer_compare_results.json`) and writes a simple HTML report grouping
models by identical `diff` content. Output is similar in spirit to
sample_report.html but intentionally small and self-contained.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Dict, List


CSS = """
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


def should_ignore_diff(diff: Dict) -> bool:
    """Check if diff should be ignored (only useless ByteLevel flags).
    
    Ignores diffs where ALL differences are in decoder and/or pre_tokenizer fields,
    and within those fields, the only differences are add_prefix_space, trim_offsets,
    and use_regex (implementation details that don't affect tokenization).
    """
    if not diff:
        return False

    # Check if only decoder/pre_tokenizer fields differ
    non_trivial_fields = [f for f in diff.keys() if f not in ("decoder", "pre_tokenizer")]
    if non_trivial_fields:
        return False

    # Pattern to match add_prefix_space, trim_offsets, use_regex
    pattern = r"[,\s]*(add_prefix_space|trim_offsets|use_regex)=[^,\)]*"

    for field, pair in diff.items():
        if field not in ("decoder", "pre_tokenizer"):
            return False

        auto_val = pair.get("auto", "")
        backend_val = pair.get("backend", "")

        # Remove the useless flags from both values
        auto_clean = re.sub(pattern, "", auto_val)
        backend_clean = re.sub(pattern, "", backend_val)

        # If the cleaned versions differ, there's a significant diff
        if auto_clean != backend_clean:
            return False

    return True


def _truncate_precompiled_charsmap(value: object, max_len: int = 220) -> str:
    text = "" if value is None else str(value)
    pattern = r'precompiled_charsmap="([^"]*)"'

    def _repl(match: re.Match) -> str:
        charsmap = match.group(1)
        if len(charsmap) <= max_len:
            return match.group(0)
        head = charsmap[:160]
        tail = charsmap[-40:]
        omitted = len(charsmap) - len(head) - len(tail)
        return f'precompiled_charsmap="{head}...[{omitted} chars omitted]...{tail}"'

    return re.sub(pattern, _repl, text)


def diff_to_text(diff: Dict) -> str:
    """Create a predictable textual representation of a diff mapping."""
    lines: List[str] = []
    for field in sorted(diff.keys()):
        pair = diff[field]
        a = _truncate_precompiled_charsmap(pair.get("auto"))
        b = _truncate_precompiled_charsmap(pair.get("backend"))
        lines.append(f"-{field}:\t{a}")
        lines.append(f"+{field}:\t{b}")
    return "\n".join(lines)


def make_report(data: List[Dict], out_path: Path, ignore_useless_flags: bool = True) -> None:
    total = len(data)
    # collect mismatches that have diffs (excluding ignored diffs if enabled)
    mismatches = [
        d
        for d in data
        if (not d.get("match", False))
        and d.get("diff")
        and (not ignore_useless_flags or not should_ignore_diff(d.get("diff", {})))
    ]

    groups: Dict[str, Dict] = {}
    for entry in mismatches:
        diff = entry.get("diff", {})
        # canonical key
        key = json.dumps(diff, sort_keys=True, ensure_ascii=False)
        groups.setdefault(key, {"diff": diff, "models": []})
        groups[key]["models"].append({"model_id": entry.get("model_id"), "found_in": entry.get("found_in", [])})

    uniq_patterns = len(groups)

    # build html
    parts: List[str] = []
    parts.append("<!doctype html><html><head><meta charset=\"utf-8\"><title>Tokenizer diff report</title>")
    parts.append(f"<style>{CSS}</style></head><body>")
    parts.append(f"<h1>Tokenizer backend equivalence report</h1>")
    parts.append(f"<p class=\"muted\">Models checked: {total} — mismatches with diffs: {len(mismatches)} — unique diff patterns: {uniq_patterns}</p>")

    # pattern cards
    parts.append("<div class=\"grid\">")
    idx = 1
    for key, info in sorted(groups.items(), key=lambda kv: -len(kv[1]["models"])):
        diff_text = diff_to_text(info["diff"]) or "(empty)"
        # escape for HTML and add span classes for +/- at line starts
        escaped = html.escape(diff_text)
        # add spans for + and -
        escaped = escaped.replace("-", "<span class=\"del\">-", 1).replace("+", "</span><span class=\"add\">+", 1)
        # fix remaining line prefixes
        escaped = escaped.replace("\n-", "\n</span><span class=\"del\">-")
        escaped = escaped.replace("\n+", "\n</span><span class=\"add\">+")
        # close last span
        if not escaped.endswith("</span>"):
            escaped = escaped + "</span>"

        parts.append(f"<div class=\"card\" id=\"pattern-{idx}\"><h2>Pattern #{idx} <span class=\"badge\">{len(info['models'])} models</span></h2>")
        parts.append(f"<pre class=\"diff\">{escaped}</pre>")
        # model list
        parts.append("<div class=\"models\"><strong>Affected models:</strong>")
        parts.append("<ul>")
        for m in sorted(info["models"], key=lambda x: x.get("model_id") or ""):
            mid = html.escape(m.get("model_id") or "")
            url = f"https://huggingface.co/{mid}"
            files = ", ".join(html.escape(f) for f in m.get("found_in", []))
            parts.append(f"<li><a href=\"{url}\" target=\"_blank\">{mid}</a> <span class=\"muted\">{files}</span></li>")
        parts.append("</ul></div></div>")
        idx += 1

    parts.append("</div>\n</body></html>")

    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    script_dir = Path(__file__).resolve().parent

    p = argparse.ArgumentParser()
    p.add_argument("--input", default="tokenizer_compare_results.json")
    p.add_argument("--output", default="tokenizer_diff_report.html")
    p.add_argument("--show-all", action="store_true", help="Show all diffs, including useless flag differences")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = script_dir / input_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    make_report(data, output_path, ignore_useless_flags=not args.show_all)
    print("Wrote", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
