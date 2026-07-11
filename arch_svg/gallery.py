"""Batch driver: render every model, write per-model SVGs, an index.html contact sheet,
and a report.json. Keeps going on per-model failure -- errors are collected, never fatal.
"""

from __future__ import annotations

import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from html import escape

from .discover import ModelEntry, discover_models, model_type_for
from .introspect import introspect
from .layout import build_diff, build_full
from .modular import diff
from .render import render


def render_one(model: str, model_type: str | None, mode: str) -> tuple[str, dict]:
    """Render a single model to an SVG string + a report record. Never raises."""
    record: dict = {"model": model, "mode": mode, "status": "ok"}
    try:
        am = introspect(model, model_type)
        record["build_status"] = am.status
        record["decoder_kind"] = am.decoder_kind
        record["layer_summary"] = am.layer_summary
        if am.error:
            record["build_error"] = am.error
        if mode in ("diff", "both"):
            ad = diff(model)
            record["is_modular"] = ad.is_modular
            record["parent_model"] = ad.parent_model
            record["parent_models"] = ad.parent_models
            record["diff_totals"] = ad.totals
            record["note"] = ad.note
            svg = render(build_diff(am, ad))
        else:
            svg = render(build_full(am))
        if am.status == "failed":
            record["status"] = "failed"
        elif am.status == "config-only":
            record["status"] = "built-from-config-only"
        return svg, record
    except Exception:
        record["status"] = "failed"
        record["traceback"] = traceback.format_exc()[-1500:]
        # emit a minimal error placeholder svg so the gallery link still works
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="480" height="120" viewBox="0 0 480 120">'
            f'<rect width="480" height="120" fill="#fff"/>'
            f'<text x="16" y="40" font-family="sans-serif" font-size="16" fill="#dc2626">{escape(model)} — failed</text>'
            f'<text x="16" y="66" font-family="monospace" font-size="11" fill="#6b7280">{escape(mode)} mode</text>'
            f"</svg>"
        )
        return svg, record


def _write(out, fname, svg):
    path = os.path.join(out, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    return os.path.basename(path)


def _worker(args):
    model, model_type, mode, out = args
    if mode == "both":
        from .modular import _modular_path

        # the detailed full view is always emitted
        full_svg, record = render_one(model, model_type, "full")
        record["svg"] = _write(out, f"{model}.svg", full_svg)
        # the diff view is only meaningful for modular models; standalone -> full only
        if _modular_path(model) is not None:
            diff_svg, diff_rec = render_one(model, model_type, "diff")
            record["diff_svg"] = _write(out, f"{model}.diff.svg", diff_svg)
            for key in ("is_modular", "parent_model", "parent_models", "diff_totals", "note"):
                if key in diff_rec:
                    record[key] = diff_rec[key]
            if diff_rec.get("status") == "failed" and record.get("status") != "failed":
                record["diff_status"] = "failed"
        else:
            record["is_modular"] = False
            record["parent_model"] = None
            record["note"] = "standalone (no modular parent)"
        return record
    svg, record = render_one(model, model_type, mode)
    record["svg"] = _write(out, f"{model}.svg", svg)
    return record


def run_all(out: str, mode: str = "both", jobs: int = 4, limit: int | None = None) -> dict:
    os.makedirs(out, exist_ok=True)
    entries: list[ModelEntry] = discover_models()
    if limit:
        entries = entries[:limit]
    tasks = [(e.name, model_type_for(e), mode, out) for e in entries]

    records: list[dict] = []
    if jobs and jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futures = {ex.submit(_worker, t): t[0] for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                name = futures[fut]
                try:
                    records.append(fut.result())
                except Exception:
                    records.append(
                        {
                            "model": name,
                            "mode": mode,
                            "status": "failed",
                            "traceback": traceback.format_exc()[-800:],
                            "svg": f"{name}.svg",
                        }
                    )
                print(f"[{i}/{len(tasks)}] {name}", flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            records.append(_worker(t))
            print(f"[{i}/{len(tasks)}] {t[0]}", flush=True)

    records.sort(key=lambda r: r["model"])
    report = _summarize(records, mode)
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"mode": mode, "summary": report, "models": records}, f, indent=2)
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(_index_html(records, mode, report))
    return {"summary": report, "out": out}


def _summarize(records: list[dict], mode: str) -> dict:
    s = {"total": len(records), "ok": 0, "config_only": 0, "failed": 0}
    if mode in ("diff", "both"):
        s["modular"] = 0
        s["standalone"] = 0
        s["largest_diffs"] = []
    for r in records:
        st = r.get("status")
        if st == "failed":
            s["failed"] += 1
        elif st == "built-from-config-only":
            s["config_only"] += 1
        else:
            s["ok"] += 1
        if mode in ("diff", "both"):
            if r.get("is_modular"):
                s["modular"] += 1
            elif r.get("is_modular") is False:
                s["standalone"] += 1
    if mode in ("diff", "both"):
        scored = []
        for r in records:
            t = r.get("diff_totals")
            if r.get("is_modular") and t:
                size = t["overridden"] + t["added"] + t["deleted"] + 3 * t["new_classes"]
                scored.append((size, r["model"], r.get("parent_model"), t))
        scored.sort(reverse=True)
        s["largest_diffs"] = [{"model": m, "parent": p, "diff_size": sz, "totals": t} for sz, m, p, t in scored[:20]]
    return s


def _index_html(records: list[dict], mode: str, summary: dict) -> str:
    cards = []
    for r in records:
        st = r.get("status", "ok")
        badge = {"ok": "#16a34a", "built-from-config-only": "#d97706", "failed": "#dc2626"}.get(st, "#6b7280")
        sub = ""
        if mode in ("diff", "both"):
            if r.get("is_modular"):
                t = r.get("diff_totals", {})
                sub = f"↳ {escape(str(r.get('parent_model')))} · ovr {t.get('overridden', 0)} / add {t.get('added', 0)} / del {t.get('deleted', 0)} / new {t.get('new_classes', 0)}"
            else:
                sub = "standalone"
        else:
            sub = escape(str(r.get("layer_summary") or r.get("decoder_kind") or ""))
        if mode == "both" and r.get("diff_svg"):
            thumbs = (
                f'<div class="thumbs">'
                f'<a href="{escape(r["svg"])}" target="_blank" class="tw">'
                f'<object data="{escape(r["svg"])}" type="image/svg+xml" class="thumb"></object>'
                f'<span class="tlabel">full</span></a>'
                f'<a href="{escape(r["diff_svg"])}" target="_blank" class="tw">'
                f'<object data="{escape(r["diff_svg"])}" type="image/svg+xml" class="thumb"></object>'
                f'<span class="tlabel">diff</span></a></div>'
            )
        else:
            thumbs = (
                f'<a href="{escape(r["svg"])}" target="_blank" class="tw">'
                f'<object data="{escape(r["svg"])}" type="image/svg+xml" class="thumb"></object></a>'
            )
        cards.append(
            f'<div class="card">'
            f'<div class="card-h"><span class="name">{escape(r["model"])}</span>'
            f'<span class="dot" style="background:{badge}"></span></div>'
            f"{thumbs}"
            f'<div class="sub">{sub}</div></div>'
        )

    summary_html = f"ok {summary['ok']} · config-only {summary['config_only']} · failed {summary['failed']}"
    if mode in ("diff", "both"):
        summary_html += f" · modular {summary.get('modular', 0)} · standalone {summary.get('standalone', 0)}"
        rows = "".join(
            f"<tr><td>{escape(d['model'])}</td><td>{escape(str(d['parent']))}</td>"
            f"<td>{d['diff_size']}</td><td>{d['totals']['overridden']}</td>"
            f"<td>{d['totals']['added']}</td><td>{d['totals']['deleted']}</td>"
            f"<td>{d['totals']['new_classes']}</td></tr>"
            for d in summary.get("largest_diffs", [])
        )
        big = (
            "<h2>Largest diffs (least modular)</h2>"
            '<table class="lb"><tr><th>model</th><th>parent</th><th>size</th>'
            "<th>ovr</th><th>add</th><th>del</th><th>new</th></tr>"
            f"{rows}</table>"
        )
    else:
        big = ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>transformers architecture gallery — {escape(mode)}</title>
<style>
:root {{ color-scheme: light dark; --bg:#fff; --fg:#111; --muted:#666; --card:#f6f8fa; --bd:#e5e7eb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d1117; --fg:#e6edf3; --muted:#8b949e; --card:#161b22; --bd:#30363d; }} }}
body {{ background:var(--bg); color:var(--fg); font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; margin:0; padding:28px; }}
h1 {{ margin:0 0 4px; }} .meta {{ color:var(--muted); margin-bottom:18px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:16px; }}
.card {{ background:var(--card); border:1px solid var(--bd); border-radius:12px; padding:12px; color:inherit; display:flex; flex-direction:column; }}
.card:hover {{ border-color:#3b82f6; }}
.card-h {{ display:flex; justify-content:space-between; align-items:center; }}
.name {{ font-weight:700; }} .dot {{ width:10px; height:10px; border-radius:50%; }}
.thumbs {{ display:flex; gap:8px; }}
.tw {{ position:relative; flex:1; text-decoration:none; }}
.tlabel {{ position:absolute; top:10px; left:6px; font-size:10px; font-weight:700; color:var(--muted); background:var(--bg); padding:1px 5px; border-radius:4px; opacity:.85; }}
.thumb {{ width:100%; height:260px; pointer-events:none; margin:8px 0; background:var(--bg); border-radius:6px; border:1px solid var(--bd); }}
.sub {{ font-size:12px; color:var(--muted); }}
table.lb {{ border-collapse:collapse; margin:8px 0 24px; font-size:13px; }}
table.lb td, table.lb th {{ border:1px solid var(--bd); padding:4px 10px; text-align:left; }}
input {{ padding:8px 12px; border-radius:8px; border:1px solid var(--bd); background:var(--card); color:var(--fg); width:260px; margin-bottom:16px; }}
</style></head>
<body>
<h1>transformers architecture gallery</h1>
<div class="meta">mode: <b>{escape(mode)}</b> · {summary_html}</div>
{big}
<input id="q" placeholder="filter models…" oninput="for(const c of document.querySelectorAll('.card')){{c.style.display=c.querySelector('.name').textContent.includes(this.value)?'':'none'}}">
<div class="grid">
{"".join(cards)}
</div>
</body></html>"""
