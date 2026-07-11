"""Positioned boxes + arrows -> SVG string.

Theming is a one-line swap: all colors are CSS variables on ``:root``; flipping to the
``.dark`` block (or honoring ``prefers-color-scheme``) reskins the whole gallery. Output is
deterministic given a ``Diagram``.
"""

from __future__ import annotations

from html import escape

from .engine import fit_text
from .layout import Diagram


# palette: each component kind gets a stable fill/stroke variable pair
_STYLE = """
:root {
  --bg: #ffffff; --fg: #1b1f24; --muted: #6b7280; --panel: #f6f8fa; --grid: #e5e7eb;
  --embed: #dbeafe; --embed-s: #3b82f6;
  --attn: #cffafe; --attn-s: #06b6d4;
  --mamba: #dcfce7; --mamba-s: #22c55e;
  --linattn: #fce7f3; --linattn-s: #ec4899;
  --recur: #ede9fe; --recur-s: #8b5cf6;
  --moe: #ffedd5; --moe-s: #f97316;
  --mlp: #ede9fe; --mlp-s: #8b5cf6;
  --norm: #e5e7eb; --norm-s: #9ca3af;
  --head: #fee2e2; --head-s: #ef4444;
  --config: #f1f5f9; --config-s: #64748b;
  --rope: #fef9c3; --rope-s: #eab308;
  --layer: #f8fafc; --layer-s: #cbd5e1;
  --io: #f1f5f9; --io-s: #94a3b8;
  --soft: #fae8ff; --soft-s: #c026d3;
  --add: #ffffff; --add-s: #475569;
  --block-s: #94a3b8;
  --residual: #f59e0b;
  --added: #16a34a; --over: #d97706; --deleted: #dc2626;
  --lt-full: #06b6d4; --lt-sliding: #3b82f6; --lt-chunked: #8b5cf6;
  --lt-compressed: #f97316; --lt-heavy: #dc2626; --lt-linear: #ec4899; --lt-mamba: #22c55e;
  --cell-on: #0ea5e9; --cell-off: #e5e7eb;
  --vision: #dcfce7; --vision-s: #16a34a; --audio: #fae8ff; --audio-s: #c026d3;
  --proj: #fef3c7; --proj-s: #d97706; --xattn: #db2777;
  --conv: #d1fae5; --conv-s: #10b981; --act: #ecfccb; --act-s: #65a30d;
  --pool: #e0f2fe; --pool-s: #0284c7; --quant: #fae8ff; --quant-s: #c026d3;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --fg: #e6edf3; --muted: #8b949e; --panel: #161b22; --grid: #30363d;
    --embed: #172554; --attn: #083344; --mamba: #052e16; --linattn: #500724;
    --recur: #2e1065; --moe: #431407; --mlp: #2e1065; --norm: #21262d; --head: #450a0a;
    --config: #1e293b; --rope: #422006; --layer: #161b22;
    --conv: #022c22; --act: #1a2e05; --pool: #082f49; --quant: #3b0764; --proj: #422006;
    --io: #1e293b; --soft: #3b0764; --add: #0d1117; --cell-off: #21262d;
  }
}
.bg { fill: var(--bg); }
text { font-family: ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif; fill: var(--fg); }
.title { font-size: 22px; font-weight: 700; }
.subtitle { font-size: 13px; fill: var(--muted); }
.box-label { font-size: 14px; font-weight: 600; }
.box-label.sm { font-size: 12.5px; }
.box-sub { font-size: 11px; fill: var(--muted); }
.glyph { font-size: 18px; font-weight: 700; fill: var(--add-s); }
.badge { font-size: 12px; font-weight: 700; fill: var(--fg); }
.facts-k { font-size: 11.5px; fill: var(--muted); }
.facts-v { font-size: 11.5px; font-weight: 600; }
.legend-t { font-size: 11.5px; fill: var(--fg); }
.panel { fill: var(--panel); stroke: var(--grid); }
rect.b { rx: 9; stroke-width: 1.6; }
.c-embed { fill: var(--embed); stroke: var(--embed-s); }
.c-attn  { fill: var(--attn);  stroke: var(--attn-s); }
.c-mamba { fill: var(--mamba); stroke: var(--mamba-s); }
.c-linattn { fill: var(--linattn); stroke: var(--linattn-s); }
.c-recur { fill: var(--recur); stroke: var(--recur-s); }
.c-moe   { fill: var(--moe);   stroke: var(--moe-s); }
.c-mlp   { fill: var(--mlp);   stroke: var(--mlp-s); }
.c-norm  { fill: var(--norm);  stroke: var(--norm-s); }
.c-head  { fill: var(--head);  stroke: var(--head-s); }
.c-config{ fill: var(--config);stroke: var(--config-s); }
.c-rope  { fill: var(--rope);  stroke: var(--rope-s); }
.c-proj  { fill: var(--proj);  stroke: var(--proj-s); }
.c-conv  { fill: var(--conv);  stroke: var(--conv-s); }
.c-act   { fill: var(--act);   stroke: var(--act-s); }
.c-pool  { fill: var(--pool);  stroke: var(--pool-s); }
.c-quant { fill: var(--quant); stroke: var(--quant-s); }
.c-layer { fill: var(--layer); stroke: var(--layer-s); }
.c-io    { fill: var(--io);    stroke: var(--io-s); }
.c-soft  { fill: var(--soft);  stroke: var(--soft-s); }
.c-add   { fill: var(--add);   stroke: var(--add-s); }
.c-block { fill: none; stroke: var(--block-s); stroke-width: 1.6; stroke-dasharray: 7 5; }
.c-lt-full { fill: var(--lt-full); stroke: var(--lt-full); }
.c-lt-sliding { fill: var(--lt-sliding); stroke: var(--lt-sliding); }
.c-lt-chunked { fill: var(--lt-chunked); stroke: var(--lt-chunked); }
.c-lt-compressed { fill: var(--lt-compressed); stroke: var(--lt-compressed); }
.c-lt-heavy { fill: var(--lt-heavy); stroke: var(--lt-heavy); }
.c-lt-linear { fill: var(--lt-linear); stroke: var(--lt-linear); }
.c-lt-mamba { fill: var(--lt-mamba); stroke: var(--lt-mamba); }
.cell-on { fill: var(--cell-on); }
.cell-off { fill: var(--cell-off); }
.grid-frame { fill: none; stroke: var(--grid); stroke-width: 1; }
.mask-bg { fill: var(--cell-off); }
.mask-on { fill: #22c55e; }
.mask-div { stroke: var(--fg); stroke-width: 1.5; stroke-dasharray: 3 2; }
.c-vision { fill: var(--vision); stroke: var(--vision-s); }
.c-audio  { fill: var(--audio);  stroke: var(--audio-s); }
.c-proj   { fill: var(--proj);   stroke: var(--proj-s); }
.c-sub    { fill: var(--bg); stroke: var(--block-s); stroke-width: 1.2; }
.sec-h { font-size: 12px; font-weight: 700; }
.sec-hbar { fill: var(--bg); opacity: 0.82; }
.residual.xattn { stroke: var(--xattn); stroke-width: 2.4; }
.ghost { opacity: 0.32; stroke-dasharray: 4 3; }
.ch-added rect.b, rect.b.ch-added { stroke: var(--added); stroke-width: 3.2; }
.ch-over rect.b, rect.b.ch-over { stroke: var(--over); stroke-width: 3.2; }
.ch-deleted rect.b, rect.b.ch-deleted { stroke: var(--deleted); stroke-width: 3.2; }
.edge { stroke: var(--grid); stroke-width: 2; }
.flow { stroke: var(--grid); stroke-width: 2; fill: none; }
.residual { stroke: var(--residual); stroke-width: 2; fill: none; }
.rope { stroke: var(--rope-s); stroke-width: 2.2; fill: none; }
.xattn { stroke: var(--xattn); stroke-width: 2.4; fill: none; }
.cell-idx { font-size: 9px; fill: #ffffff; font-weight: 600; }
.sky { fill: #bae6fd; } .sun { fill: #fde047; } .hill { fill: #4ade80; }
"""

_CHANGE_SWATCH = {"added": "var(--added)", "overridden": "var(--over)", "deleted": "var(--deleted)"}
_LEGEND_SWATCH = {
    "c-attn": "var(--attn-s)",
    "c-moe": "var(--moe-s)",
    "c-mlp": "var(--mlp-s)",
    "c-mamba": "var(--mamba-s)",
    "c-norm": "var(--norm-s)",
    "c-embed": "var(--embed-s)",
    "c-head": "var(--head-s)",
    "c-rope": "var(--rope-s)",
    "c-vision": "var(--vision-s)",
    "c-audio": "var(--audio-s)",
    "c-proj": "var(--proj-s)",
    "xattn": "var(--xattn)",
    "c-lt-full": "var(--lt-full)",
    "c-lt-sliding": "var(--lt-sliding)",
    "c-lt-chunked": "var(--lt-chunked)",
    "c-lt-compressed": "var(--lt-compressed)",
    "c-lt-heavy": "var(--lt-heavy)",
    "c-lt-linear": "var(--lt-linear)",
    "c-lt-mamba": "var(--lt-mamba)",
    "residual": "var(--residual)",
    "ch-added": "var(--added)",
    "ch-over": "var(--over)",
    "ch-deleted": "var(--deleted)",
    "ghost": "var(--grid)",
}


def _box_svg(b) -> str:
    change_cls = {"added": "ch-added", "overridden": "ch-over", "deleted": "ch-deleted"}.get(b.change, "")
    group_cls = "ghost" if b.ghost else ""
    parts = [f'<g class="{group_cls}">']
    if b.title:
        parts.append(f"<title>{escape(str(b.title))}</title>")
    cx = b.x + b.w / 2

    if b.shape == "cell":  # a single layer-schedule tick with its index
        parts.append(f'<rect class="{b.cls}" x="{b.x}" y="{b.y}" width="{b.w}" height="{b.h}" rx="2"/>')
        if b.label and b.h >= 11:
            parts.append(
                f'<text class="cell-idx" x="{b.x + b.w / 2}" y="{b.y + b.h / 2 + 3.5}" text-anchor="middle">{escape(b.label)}</text>'
            )
        parts.append("</g>")
        return "".join(parts)

    if b.shape == "grid" and b.grid:
        rows = len(b.grid)
        cols = len(b.grid[0]) if rows else 0
        cell = b.w / max(cols, 1)  # square cells; width fixed, height follows rows
        parts.append(
            f'<text class="box-sub" x="{b.x}" y="{b.y - 6}">{escape(fit_text(b.label, max(b.w, 130), 11))}</text>'
        )
        parts.append(
            f'<rect class="mask-bg" x="{b.x}" y="{b.y}" width="{cell * cols:.1f}" height="{cell * rows:.1f}"/>'
        )
        for i, row in enumerate(b.grid):
            for j, c in enumerate(row):
                if c:
                    parts.append(
                        f'<rect class="mask-on" x="{b.x + j * cell:.1f}" y="{b.y + i * cell:.1f}" '
                        f'width="{cell:.1f}" height="{cell:.1f}"/>'
                    )
        if b.grid_split is not None and 0 < b.grid_split < cols:  # sliding | compressed divider
            dx = b.x + b.grid_split * cell
            parts.append(
                f'<line class="mask-div" x1="{dx:.1f}" y1="{b.y}" x2="{dx:.1f}" y2="{b.y + cell * rows:.1f}"/>'
            )
        parts.append(
            f'<rect class="grid-frame" x="{b.x}" y="{b.y}" width="{cell * cols:.1f}" height="{cell * rows:.1f}"/>'
        )
        parts.append("</g>")
        return "".join(parts)

    if b.shape == "circle":
        r = b.h // 2
        parts.append(f'<circle class="b {b.cls} {change_cls}" cx="{cx}" cy="{b.y + r}" r="{r}"/>')
        if b.glyph:
            parts.append(
                f'<text class="glyph" x="{cx}" y="{b.y + r + 6}" text-anchor="middle">{escape(b.glyph)}</text>'
            )
        parts.append("</g>")
        return "".join(parts)

    if b.shape == "container":
        parts.append(f'<rect class="{b.cls}" x="{b.x}" y="{b.y}" width="{b.w}" height="{b.h}" rx="14"/>')
        parts.append(f'<text class="box-sub" x="{b.x + 12}" y="{b.y + 18}">{escape(b.label)}</text>')
        if b.badge:
            parts.append(
                f'<text class="badge" x="{b.x + b.w - 12}" y="{b.y + 20}" text-anchor="end">{escape(b.badge)}</text>'
            )
        parts.append("</g>")
        return "".join(parts)

    if b.shape == "section":  # filled rounded panel with a header; inner chips drawn on top
        change_cls = {"added": "ch-added", "overridden": "ch-over", "deleted": "ch-deleted"}.get(b.change, "")
        parts.append(
            f'<rect class="b {b.cls} {change_cls}" x="{b.x}" y="{b.y}" width="{b.w}" height="{b.h}" rx="11"/>'
        )
        # white header strip so the class name stays readable on saturated section fills
        parts.append(f'<rect class="sec-hbar" x="{b.x + 5}" y="{b.y + 4}" width="{b.w - 10}" height="18" rx="5"/>')
        hdr = fit_text(b.label, b.w - 24 - (40 if b.badge else 0), 12)
        parts.append(f'<text class="sec-h" x="{b.x + 12}" y="{b.y + 17}">{escape(hdr)}</text>')
        if b.badge:
            parts.append(
                f'<text class="badge" x="{b.x + b.w - 12}" y="{b.y + 17}" text-anchor="end">{escape(b.badge)}</text>'
            )
        parts.append("</g>")
        return "".join(parts)

    if b.shape == "image":  # tiny schematic example image (for pixel_values)
        parts.append(f'<rect class="b {b.cls}" x="{b.x}" y="{b.y}" width="{b.w}" height="{b.h}" rx="6"/>')
        ix, iy, iw, ih = b.x + 6, b.y + 6, b.w - 12, b.h - 12
        parts.append(
            f'<clipPath id="ic{b.x}{b.y}"><rect x="{ix}" y="{iy}" width="{iw}" height="{ih}" rx="3"/></clipPath>'
        )
        g = f'<g clip-path="url(#ic{b.x}{b.y})">'
        g += f'<rect class="sky" x="{ix}" y="{iy}" width="{iw}" height="{ih}"/>'
        g += f'<circle class="sun" cx="{ix + iw * 0.72:.0f}" cy="{iy + ih * 0.32:.0f}" r="{ih * 0.16:.0f}"/>'
        g += f'<path class="hill" d="M{ix},{iy + ih} L{ix + iw * 0.35:.0f},{iy + ih * 0.5:.0f} L{ix + iw * 0.6:.0f},{iy + ih:.0f} Z"/>'
        g += f'<path class="hill" d="M{ix + iw * 0.45:.0f},{iy + ih} L{ix + iw * 0.75:.0f},{iy + ih * 0.55:.0f} L{ix + iw:.0f},{iy + ih:.0f} Z"/>'
        g += "</g>"
        parts.append(g)
        parts.append(f'<rect class="grid-frame" x="{ix}" y="{iy}" width="{iw}" height="{ih}" rx="3"/>')
        if b.label:
            parts.append(
                f'<text class="box-sub" x="{cx}" y="{b.y + b.h + 12}" text-anchor="middle">{escape(b.label)}</text>'
            )
        parts.append("</g>")
        return "".join(parts)

    rx = b.h // 2 if b.shape == "io" else 9
    rect_cls = f"b {b.cls} {change_cls}".strip()
    parts.append(f'<rect class="{rect_cls}" x="{b.x}" y="{b.y}" width="{b.w}" height="{b.h}" rx="{rx}"/>')
    label_cls = "box-label sm" if b.small else "box-label"
    lpx = 12.5 if b.small else 14
    inner = b.w - 14  # text must fit inside the box with a little padding
    # vertically centre the label + sublabels block within the box
    subs = b.sublabels or []
    n_sub = len(subs)
    if n_sub == 0:
        ty = b.y + b.h / 2 + 4.5
    else:
        block = 13 + 15 * n_sub
        ty = b.y + (b.h - block) / 2 + 12
    lab = fit_text(b.label, inner, lpx)
    parts.append(f'<text class="{label_cls}" x="{cx}" y="{ty:.1f}" text-anchor="middle">{escape(lab)}</text>')
    sy = ty + 15
    for s in subs:
        parts.append(
            f'<text class="box-sub" x="{cx}" y="{sy:.1f}" text-anchor="middle">{escape(fit_text(s, inner, 11))}</text>'
        )
        sy += 15
    if b.badge:
        parts.append(
            f'<text class="badge" x="{b.x + b.w - 12}" y="{b.y + 20}" text-anchor="end">{escape(b.badge)}</text>'
        )
    if b.change:
        sw = _CHANGE_SWATCH.get(b.change, "")
        parts.append(f'<circle cx="{b.x + 12}" cy="{b.y + 12}" r="5" fill="{sw}"/>')
    parts.append("</g>")
    return "".join(parts)


def _arrow_svg(a) -> str:
    pts = " ".join(f"{x},{y}" for x, y in a.points)
    dash = ' stroke-dasharray="6 4"' if a.dashed else ""
    return f'<polyline class="{a.cls}" points="{pts}"{dash} marker-end="url(#ah-{a.cls})"/>'


def _marker(name: str, color: str) -> str:
    return (
        f'<marker id="ah-{name}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>'
    )


def render(diagram: Diagram) -> str:
    d = diagram
    panel_w = 264
    panel_x = d.width - panel_w - 24
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {d.width} {d.height}" '
        f'width="{d.width}" height="{d.height}" font-size="14">',
        f"<style>{_STYLE}</style>",
        f"<defs>{_marker('flow', 'var(--grid)')}{_marker('residual', 'var(--residual)')}"
        f"{_marker('rope', 'var(--rope-s)')}{_marker('xattn', 'var(--xattn)')}</defs>",
        f'<rect class="bg" x="0" y="0" width="{d.width}" height="{d.height}"/>',
        f'<text class="title" x="24" y="34">{escape(d.title)}</text>',
        f'<text class="subtitle" x="24" y="54">{escape(d.subtitle)}</text>',
    ]

    # connecting spine behind the central column only (use the modal box-center x)
    if d.spine:
        col = [b for b in d.boxes if b.shape not in ("container", "cell", "grid")]
        if col:
            from collections import Counter

            centers = Counter(round(b.x + b.w / 2) for b in col)
            xs = centers.most_common(1)[0][0]
            on_axis = [b for b in col if abs((b.x + b.w / 2) - xs) < 6]
            top = min(b.y for b in on_axis)
            bot = max(b.y + b.h for b in on_axis)
            parts.append(f'<line class="edge" x1="{xs}" y1="{top}" x2="{xs}" y2="{bot}"/>')

    # containers + sections first (behind), then arrows, then boxes on top.
    # sections are drawn LARGEST-first so a nested (inner) section paints on top of its
    # parent instead of being hidden by the parent's opaque fill.
    for b in sorted((b for b in d.boxes if b.shape == "container"), key=lambda b: -b.w * b.h):
        parts.append(_box_svg(b))
    for b in sorted((b for b in d.boxes if b.shape == "section"), key=lambda b: -b.w * b.h):
        parts.append(_box_svg(b))
    for a in d.arrows:
        parts.append(_arrow_svg(a))
    for b in d.boxes:
        if b.shape not in ("container", "section"):
            parts.append(_box_svg(b))

    # side panel: facts
    py = 88
    parts.append(
        f'<rect class="panel" x="{panel_x}" y="{py}" width="{panel_w}" height="{len(d.facts) * 22 + 20}" rx="8"/>'
    )
    fy = py + 24
    for k, v in d.facts:
        parts.append(f'<text class="facts-k" x="{panel_x + 14}" y="{fy}">{escape(k)}</text>')
        parts.append(
            f'<text class="facts-v" x="{panel_x + panel_w - 14}" y="{fy}" text-anchor="end">{escape(str(v))}</text>'
        )
        fy += 22
    fy += 16

    # side panel: legend
    parts.append(f'<text class="legend-t" x="{panel_x + 2}" y="{fy}" font-weight="700">legend</text>')
    fy += 12
    _seen_leg = set()
    legend = [(c, l) for c, l in d.legend if not ((c, l) in _seen_leg or _seen_leg.add((c, l)))]
    for cls, label in legend:
        sw = _LEGEND_SWATCH.get(cls, "var(--grid)")
        is_change = cls.startswith("ch-") or cls in ("ghost", "residual")
        if is_change:
            parts.append(
                f'<rect x="{panel_x + 4}" y="{fy - 1}" width="16" height="12" rx="2" fill="none" stroke="{sw}" stroke-width="3"/>'
            )
        else:
            parts.append(f'<rect x="{panel_x + 4}" y="{fy - 1}" width="16" height="12" rx="2" fill="{sw}"/>')
        parts.append(f'<text class="legend-t" x="{panel_x + 28}" y="{fy + 9}">{escape(label)}</text>')
        fy += 18

    # side panel: per-class changes (diff mode)
    if d.changes:
        fy += 14
        parts.append(f'<text class="legend-t" x="{panel_x + 2}" y="{fy}" font-weight="700">changes by class</text>')
        fy += 16
        for ch, name, detail in d.changes:
            sw = _CHANGE_SWATCH.get(ch, "var(--grid)")
            parts.append(f'<circle cx="{panel_x + 8}" cy="{fy - 3}" r="4" fill="{sw}"/>')
            parts.append(f'<text class="facts-v" x="{panel_x + 20}" y="{fy}">{escape(name)}</text>')
            fy += 13
            if detail:
                d_short = detail if len(detail) <= 46 else detail[:44] + "…"
                parts.append(f'<text class="box-sub" x="{panel_x + 20}" y="{fy}">{escape(d_short)}</text>')
                fy += 15

    parts.append("</svg>")
    return "\n".join(parts)
