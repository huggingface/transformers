"""Visualise the per-layer-type attention mask DeepSeek-V4 actually feeds to
`eager_attention_forward`, for the three attention layer types (sliding, CSA, HCA).
Generates the SVGs embedded in `docs/source/en/model_doc/deepseek_v4.md`.

We build a tiny V4 model (small enough that 16 source tokens × all compressed
slots fits on screen), forward a dummy batch through it, wrap each attention
layer's forward with a thin shim that replays the mask-extension logic
(`cat([sliding_mask, block_bias])`) and captures the exact post-cat mask, then
render each layer's mask in either:

  * the `■` / `⬚` ANSI-coloured style of `transformers.utils.attention_visualizer`
    (default; suppressed when stdout isn't a TTY or `NO_COLOR=1` is set), or
  * an SVG grid suitable for embedding in markdown (with `--svg <DIR>`).

For CSA layers the visualizer remaps the literal `[S, S·k]` flat-slot `block_bias`
back to the more readable `[S, T_entries]` entry-visibility view (each compressor
column is then a compressed *entry* rather than a gather slot), and shades cells
red when an entry is causally available but the indexer's top-k did not pick it.

Run from the repository root::

    python docs/source/en/imgs/deepseek_v4/visualize_attention_masks.py
    python docs/source/en/imgs/deepseek_v4/visualize_attention_masks.py \\
        --svg docs/source/en/imgs/deepseek_v4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACompressor,
    DeepseekV4HCACompressor,
    DeepseekV4Indexer,
    DeepseekV4PreTrainedModel,
)
from transformers.utils.output_capturing import OutputRecorder


# Tiny config so the visualised matrices fit in a terminal. Compress rates are dialled
# down (m_csa=4, m_hca=8 — vs the real 4/128) so HCA actually emits an entry at S=16.
CFG = DeepseekV4Config(
    vocab_size=64,
    hidden_size=64,
    moe_intermediate_size=64,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=32,
    partial_rotary_factor=8 / 32,
    q_lora_rank=32,
    o_groups=2,
    o_lora_rank=16,
    num_experts_per_tok=2,
    n_routed_experts=4,
    n_shared_experts=1,
    mlp_layer_types=["moe", "moe", "moe"],
    layer_types=["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"],
    compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
    sliding_window=8,
    hc_mult=2,
    hc_sinkhorn_iters=3,
    hc_eps=1.0e-6,
    index_n_heads=2,
    index_head_dim=16,
    index_topk=2,
    num_nextn_predict_layers=0,
    swiglu_limit=10.0,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    max_position_embeddings=64,
)


# ANSI colours, matching `transformers.utils.attention_visualizer`.
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[90m"
RESET = "\033[0m"
BLACK_SQUARE = "■"
WHITE_SQUARE = "⬚"


def _meta_subtitle(meta: dict | None) -> str:
    """One-liner subtitle describing the layer's compressor config."""
    if meta is None:
        return ""
    parts: list[str] = []
    if meta.get("compress_rate") is not None:
        parts.append(f"compress_rate (m) = {meta['compress_rate']}")
    if meta.get("index_topk") is not None:
        parts.append(f"index_topk (k) = {meta['index_topk']}")
    if meta.get("T_entries"):
        parts.append(f"total compressed entries = {meta['T_entries']}")
    return "   ".join(parts)


def _render(
    mask_2d: torch.Tensor,
    query_labels: list[str],
    kv_labels: list[str],
    title: str,
    *,
    color: bool = True,
    meta: dict | None = None,
) -> str:
    """Render a 2D additive mask (0 = visible, -inf = masked) as a ■ / ⬚ grid.

    `mask_2d` is `[S_q, S_kv]`. Columns past `len(query_labels)` are the
    compressor / indexer entries the caller cat'd into the mask via
    `cat([sliding_causal_mask, block_bias], dim=-1)`. When `color=False`,
    diagonal / compressor slots are marked with the distinguishing glyphs
    `◆` and `▲` instead of ANSI colours so the output renders in markdown.
    """
    visible = (mask_2d > -1e30).bool()
    n_q, n_kv = visible.shape
    sliding_kv = len(query_labels)  # by construction kv labels begin with the sliding tokens

    if color:
        g, y, d, r = GREEN, YELLOW, DIM, RESET
        visible_glyph = f"{g}{BLACK_SQUARE}{r}"
        compr_glyph = f"{y}{BLACK_SQUARE}{r}"
    else:
        g = y = d = r = ""
        visible_glyph = "▣"
        compr_glyph = "▲"

    max_q_label = max(len(q) for q in query_labels)
    out: list[str] = []
    out.append(title)
    out.append("=" * len(title))
    out.append(
        f"  shape=[{n_q}, {n_kv}]   "
        f"{visible_glyph} visible (sliding KV)   {compr_glyph} visible (compressor entry)   {WHITE_SQUARE} masked"
    )
    subtitle = _meta_subtitle(meta)
    if subtitle:
        out.append(f"  {subtitle}")
    if meta and meta.get("topk_picks") is not None:
        # Append per-query indexer picks so warm-up sentinels and entry choices are auditable.
        picks_lines = [f"  indexer topk picks (entry id, -1 = warm-up sentinel):"]
        for i, picks in enumerate(meta["topk_picks"][0]):  # B=1
            picks_lines.append(f"    q{i:>2}: {picks}")
        out.append("\n".join(picks_lines))

    # Column headers (kv indices, stacked vertically across `width` lines).
    width = max(2, len(str(n_kv - 1)))
    for line in range(width):
        header_chars = []
        for j in range(n_kv):
            label = str(j).rjust(width)
            ch = label[line] if line < len(label) else " "
            # Tint compressor-section header so it visually separates from the sliding section.
            header_chars.append(f"{d}{ch}{r}" if j >= sliding_kv else ch)
        prefix = " " * (max_q_label + 6)
        out.append(prefix + " ".join(header_chars))

    # Body rows.
    for i in range(n_q):
        row_cells = []
        for j in range(n_kv):
            if not visible[i, j]:
                row_cells.append(WHITE_SQUARE)
            elif j >= sliding_kv:
                row_cells.append(compr_glyph)
            else:
                row_cells.append(visible_glyph)
        out.append(f"{query_labels[i].rjust(max_q_label)} : {str(i).rjust(2)}  " + " ".join(row_cells))

    out.append(f"  {compr_glyph} columns past index {sliding_kv - 1} are compressor / indexer slots (block_bias).")
    return "\n".join(out)


# SVG palette + geometry. Bigger cells so we can fit token-string labels comfortably,
# and compressor columns are drawn `m` cells wide so the visual encodes the fact that
# one compressed entry summarizes m source tokens (rather than a 1:1 KV slot like the
# sliding section).
SVG_CELL = 24
SVG_GAP = 2
SVG_PAD_LEFT = 120     # roomy enough for the longest row-label token ("morning") at 11pt
SVG_PAD_TOP = 340      # enough for vertical compressor-token lists (HCA m=8 = ~250px tall)
SVG_PAD_RIGHT = 48
SVG_PAD_BOTTOM = 64
SVG_COLORS = {
    # Dark-mode palette: black background, light text.
    "background": "#0b1220",    # near-black with a hint of blue
    "visible": "#22c55e",       # green-500, pops against black
    "masked": "#1e293b",        # slate-800 — dim cells that are visible against bg but recede
    "indexer_masked": "#ef4444",# red-500, also pops
    "compressor": "#22c55e",    # same green as visible (single-color attended-to)
    "separator": "#475569",     # slate-600
    "text": "#f8fafc",          # slate-50 — main text on black
    "muted": "#94a3b8",         # slate-400 — secondary text
}

# Static word list used as token labels on the SVG axes. Real token semantics don't
# affect the mask (which depends only on positions), so these labels are decorative.
SVG_TOKENS = [
    "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy",
    "dog", "ate", "a", "small", "red", "fish", "this", "morning",
]


def _render_svg(mask_2d: torch.Tensor, title: str, sliding_kv: int, meta: dict | None = None) -> str:
    """Render the mask as a standalone SVG document.

    Rows = queries (top → bottom). The sliding K=V section is one cell per
    source position (key index). The compressor section is `T_entries`
    blocks each `m·cell` wide — the wide rectangle is the visual hint that
    one compressed entry summarizes `m` source tokens (rather than 1:1).

    Cells are coloured: green = visible (the query attends to this slot),
    light gray = masked. Compressor entries are tinted amber when visible
    to a given query and pale amber when masked, so the compressor section
    reads as a distinct "compression band" even at a glance.
    """
    visible = (mask_2d > -1e30).bool()
    n_q, n_kv = visible.shape
    n_compr = n_kv - sliding_kv
    m = (meta or {}).get("compress_rate") or 1
    stride = SVG_CELL + SVG_GAP
    compr_block_w = m * SVG_CELL + (m - 1) * SVG_GAP

    # Per-column x positions and widths. Sliding cells: 1×cell. Compressor entries:
    # `m`×cell wide each, with a small gap between sections.
    section_gap = SVG_GAP * 4
    col_x: list[float] = []
    col_w: list[float] = []
    cursor = float(SVG_PAD_LEFT)
    for j in range(sliding_kv):
        col_x.append(cursor)
        col_w.append(SVG_CELL)
        cursor += stride
    if n_compr > 0:
        cursor += section_gap
    for _w in range(n_compr):
        col_x.append(cursor)
        col_w.append(compr_block_w)
        cursor += compr_block_w + stride
    grid_w = cursor - SVG_PAD_LEFT - stride
    grid_h = n_q * stride - SVG_GAP
    w = SVG_PAD_LEFT + grid_w + SVG_PAD_RIGHT
    h = SVG_PAD_TOP + grid_h + SVG_PAD_BOTTOM

    subtitle = _meta_subtitle(meta)
    # Explicit `width` / `height` (matching viewBox) so renderers don't fall back to the
    # 150×150 SVG default when the file is embedded without sizing CSS. `preserveAspectRatio`
    # keeps it sharp when scaled.
    elems: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" preserveAspectRatio="xMinYMin meet" '
        f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="11">',
        f'  <rect x="0" y="0" width="{w}" height="{h}" fill="{SVG_COLORS["background"]}" />',
        f'  <text x="{SVG_PAD_LEFT}" y="20" font-size="15" font-weight="600" fill="{SVG_COLORS["text"]}">{title}</text>',
        f'  <text x="{SVG_PAD_LEFT}" y="40" fill="{SVG_COLORS["muted"]}">'
        f'shape=[{n_q}, {n_kv}]   keys →   queries ↓</text>',
    ]
    if subtitle:
        elems.append(
            f'  <text x="{SVG_PAD_LEFT}" y="60" fill="{SVG_COLORS["compressor"]}">{subtitle}</text>'
        )

    # Column headers, top-down within the `SVG_PAD_TOP` gutter (above the grid line):
    #   y =  78  : `C_w` (centered over the compressor block, bold green)
    #   y =  92  : `pos a–b` source-position range for the compressor block
    #   y = SVG_PAD_TOP - 70 = 270 : numeric column index (sliding) — just above the
    #                                tallest rotated token so it reads as a tick label.
    #   y = SVG_PAD_TOP - 60 .. SVG_PAD_TOP : rotated (-90°) per-position token labels
    #                                          (sliding + per-source-position inside blocks)
    #   y = SVG_PAD_TOP                       : grid begins.
    idx_y = SVG_PAD_TOP - 70
    token_baseline_y = SVG_PAD_TOP - 8
    # Vertical extent of the rotated-token area we want the dashed separators to span —
    # just the token region, not the full title/subtitle gutter.
    token_top_y = SVG_PAD_TOP - 60
    for j in range(sliding_kv):
        cx = col_x[j] + col_w[j] / 2
        elems.append(
            f'  <text x="{cx:.1f}" y="{idx_y}" text-anchor="middle" '
            f'fill="{SVG_COLORS["muted"]}" font-size="10">{j}</text>'
        )
        tok = SVG_TOKENS[j] if j < len(SVG_TOKENS) else f"t{j}"
        elems.append(
            f'  <text x="{cx:.1f}" y="{token_baseline_y}" text-anchor="start" '
            f'transform="rotate(-90 {cx:.1f} {token_baseline_y})" '
            f'fill="{SVG_COLORS["text"]}" font-size="12" font-style="italic">{tok}</text>'
        )
    for w_idx in range(n_compr):
        j = sliding_kv + w_idx
        cx = col_x[j] + col_w[j] / 2
        block_x = col_x[j]
        a, b = w_idx * m, (w_idx + 1) * m - 1
        # `C_w` + `pos a–b` sit on the same row as the sliding-section indices, hugging
        # the top of the rotated-token area — same vertical band as the column ticks.
        elems.append(
            f'  <text x="{cx:.1f}" y="{idx_y - 14}" text-anchor="middle" '
            f'fill="{SVG_COLORS["visible"]}" font-size="13" font-weight="700">C{w_idx}</text>'
        )
        elems.append(
            f'  <text x="{cx:.1f}" y="{idx_y}" text-anchor="middle" '
            f'fill="{SVG_COLORS["muted"]}" font-size="10">pos {a}–{b}</text>'
        )
        # One rotated label per source position, dashed vertical separators between
        # sub-cells inside the block (only as tall as the rotated-token area, not the
        # whole gutter — they're a hint that the wide green block bundles m positions).
        for sub_i in range(m):
            pos = a + sub_i
            sub_cx = block_x + sub_i * stride + SVG_CELL / 2
            tok = SVG_TOKENS[pos] if pos < len(SVG_TOKENS) else f"t{pos}"
            elems.append(
                f'  <text x="{sub_cx:.1f}" y="{token_baseline_y}" text-anchor="start" '
                f'transform="rotate(-90 {sub_cx:.1f} {token_baseline_y})" '
                f'fill="{SVG_COLORS["text"]}" font-size="12" font-style="italic">{tok}</text>'
            )
            if sub_i > 0:
                sep_x = block_x + sub_i * stride - SVG_GAP / 2
                elems.append(
                    f'  <line x1="{sep_x:.1f}" y1="{token_top_y}" '
                    f'x2="{sep_x:.1f}" y2="{SVG_PAD_TOP}" '
                    f'stroke="{SVG_COLORS["separator"]}" stroke-width="0.7" stroke-dasharray="3 3" />'
                )

    # Row labels: query index (small, muted) + token (italic, dark) on the left.
    # `SVG_PAD_LEFT` is sized to fit the longest token in the static word list.
    for i in range(n_q):
        cy = SVG_PAD_TOP + i * stride + SVG_CELL / 2 + 4
        elems.append(
            f'  <text x="6" y="{cy:.1f}" fill="{SVG_COLORS["muted"]}" font-size="10">q{i:>2}</text>'
        )
        tok = SVG_TOKENS[i] if i < len(SVG_TOKENS) else f"t{i}"
        elems.append(
            f'  <text x="{SVG_PAD_LEFT - 8}" y="{cy:.1f}" text-anchor="end" '
            f'fill="{SVG_COLORS["text"]}" font-size="12" font-style="italic">{tok}</text>'
        )

    # Dashed vertical separator between sliding K=V and compressor sections.
    if 0 < sliding_kv < n_kv:
        sep_x = col_x[sliding_kv - 1] + col_w[sliding_kv - 1] + section_gap / 2
        elems.append(
            f'  <line x1="{sep_x:.1f}" y1="{SVG_PAD_TOP - 28}" x2="{sep_x:.1f}" '
            f'y2="{SVG_PAD_TOP + grid_h:.1f}" stroke="{SVG_COLORS["separator"]}" '
            f'stroke-width="1" stroke-dasharray="4 3" />'
        )

    # Indexer-rejected mask (CSA only): causally available compressor entry that the
    # indexer's top-k chose not to pick. Rendered red to distinguish from the plain
    # "not yet causally ready" masking (light gray).
    rejected = (meta or {}).get("indexer_rejected")
    if rejected is not None:
        if rejected.ndim == 4:
            rejected = rejected[0, 0]
        elif rejected.ndim == 3:
            rejected = rejected[0]

    # Cells.
    for i in range(n_q):
        for j in range(n_kv):
            x = col_x[j]
            y = SVG_PAD_TOP + i * stride
            in_compressor = j >= sliding_kv
            is_rejected = rejected is not None and bool(rejected[i, j])
            if visible[i, j]:
                fill = SVG_COLORS["compressor"] if in_compressor else SVG_COLORS["visible"]
            elif is_rejected:
                fill = SVG_COLORS["indexer_masked"]
            else:
                fill = SVG_COLORS["masked"]
            elems.append(
                f'  <rect x="{x:.1f}" y="{y}" width="{col_w[j]:.1f}" height="{SVG_CELL}" '
                f'rx="3" ry="3" fill="{fill}" />'
            )

    # Legend at the bottom. The red legend item is only meaningful for CSA (the only
    # layer type that has an indexer + top-k pick step), so we include it only when
    # the captured mask actually has rejected cells.
    legend_y = SVG_PAD_TOP + grid_h + 30
    legend_items = [
        ("attended-to", SVG_COLORS["visible"]),
        ("masked (not causally ready)", SVG_COLORS["masked"]),
    ]
    if rejected is not None and bool(rejected.any()):
        legend_items.append(("indexer-masked (top-k did not pick)", SVG_COLORS["indexer_masked"]))
    x_cursor = SVG_PAD_LEFT
    for label, color in legend_items:
        elems.append(
            f'  <rect x="{x_cursor}" y="{legend_y - 12}" width="14" height="14" rx="3" ry="3" fill="{color}" />'
        )
        elems.append(
            f'  <text x="{x_cursor + 20}" y="{legend_y}" fill="{SVG_COLORS["text"]}">{label}</text>'
        )
        x_cursor += 20 + len(label) * 6.5 + 24

    elems.append("</svg>")
    return "\n".join(elems)


def _sliding_causal_mask(seq_len: int, sliding_window: int) -> torch.Tensor:
    """Standard `[1, 1, S, S]` sliding-window-causal additive mask (0 visible, -inf masked).
    Identical to what `create_sliding_window_causal_mask` builds for the sliding section
    of every V4 attention layer — we synthesise it here so the visualiser stays decoupled
    from the model's internal mask-building."""
    q = torch.arange(seq_len)
    k = torch.arange(seq_len)
    diff = q.view(-1, 1) - k.view(1, -1)
    visible = (diff >= 0) & (diff < sliding_window)
    return torch.where(visible, torch.zeros((), dtype=torch.float32), torch.full((), float("-inf"))).view(1, 1, seq_len, seq_len)


def main() -> None:
    torch.manual_seed(0)

    # ────────────────────────────────────────────────────────────────────────────────
    # Extend `DeepseekV4PreTrainedModel._can_record_outputs` with the compressor /
    # indexer outputs we need to reconstruct the visualised mask. The base class
    # already registers `router_logits` / `hidden_states` / `attentions`; we add three
    # more keys, each routed by `OutputRecorder(target_class=…, index=…)`:
    #
    #   * `csa_block_bias`  ← `DeepseekV4CSACompressor.forward()` returns `(gathered, block_bias)`,
    #                          we grab index 1 (the per-query bias `[B, 1, S, S*k]`).
    #   * `hca_block_bias`  ← `DeepseekV4HCACompressor.forward()` likewise (already entry-resolved).
    #   * `indexer_topk`    ← `DeepseekV4Indexer.forward()` returns a single tensor `[B, S, k]`.
    #
    # Setting this *before* instantiating the model is what `PreTrainedModel.__init__`
    # picks up to populate `_CAN_RECORD_REGISTRY`. The `@capture_outputs` decorator on
    # `DeepseekV4Model.forward` then installs the corresponding `register_forward_hook`
    # calls lazily on first request and stashes the captures on the returned
    # `ModelOutput`. We just flip `config.output_<key>=True` to switch them on.
    # ────────────────────────────────────────────────────────────────────────────────
    DeepseekV4PreTrainedModel._can_record_outputs = {
        **(DeepseekV4PreTrainedModel._can_record_outputs or {}),
        "csa_block_bias": OutputRecorder(DeepseekV4CSACompressor, index=1),
        "hca_block_bias": OutputRecorder(DeepseekV4HCACompressor, index=1),
        "indexer_topk": OutputRecorder(DeepseekV4Indexer, index=0),
    }

    cfg = CFG
    for key in ("csa_block_bias", "hca_block_bias", "indexer_topk"):
        setattr(cfg, f"output_{key}", True)
    full_model = DeepseekV4ForCausalLM(cfg).eval()
    # `@capture_outputs` decorates `DeepseekV4Model.forward` (the inner model). The outer
    # `DeepseekV4ForCausalLM.forward` builds a `CausalLMOutput` that only forwards logits
    # and last_hidden_state, so our extra capture keys disappear if we go via the outer.
    # Call the inner directly — we don't need logits to visualise masks.
    inner_model = full_model.model

    seq_len = 16
    input_ids = torch.arange(seq_len).unsqueeze(0) % cfg.vocab_size  # [1, S]
    with torch.no_grad():
        out = inner_model(input_ids=input_ids)

    # Tuples (one entry per layer of the matching module class). With our 3-layer
    # config: 1 CSA layer + 1 HCA layer + 1 sliding-only layer.
    csa_block_biases = getattr(out, "csa_block_bias", ()) or ()
    hca_block_biases = getattr(out, "hca_block_bias", ()) or ()
    indexer_topks = getattr(out, "indexer_topk", ()) or ()

    # ────────────────────────────────────────────────────────────────────────────────
    # Build the display mask + metadata per layer from the captured outputs.
    # ────────────────────────────────────────────────────────────────────────────────
    sliding_mask = _sliding_causal_mask(seq_len, cfg.sliding_window)  # [1, 1, S, S]
    captured: dict[int, tuple[str, torch.Tensor, dict]] = {}
    csa_iter = iter(csa_block_biases)
    hca_iter = iter(hca_block_biases)
    indexer_iter = iter(indexer_topks)
    for layer_idx, layer in enumerate(inner_model.layers):
        attn = layer.self_attn
        layer_type = attn.layer_type
        meta = {
            "compress_rate": (attn.compressor.compress_rate if attn.compressor is not None else None),
            "index_topk": (
                attn.compressor.indexer.index_topk
                if attn.compressor is not None and getattr(attn.compressor, "indexer", None) is not None
                else None
            ),
            "T_entries": seq_len // attn.compressor.compress_rate if attn.compressor is not None else 0,
            "topk_picks": None,
            "indexer_rejected": None,
        }
        if attn.compressor is None:
            captured[layer_idx] = (layer_type, sliding_mask, meta)
            continue

        if layer_type == "heavily_compressed_attention":
            block_bias = next(hca_iter).detach().cpu()
            display_mask = torch.cat([sliding_mask, block_bias.to(sliding_mask.dtype)], dim=-1)
        elif layer_type == "compressed_sparse_attention":
            _ = next(csa_iter)  # consume the (literal) `[B, 1, S, S*k]` flat-slot bias
            topk = next(indexer_iter).detach().cpu()  # [B, S, k]
            meta["topk_picks"] = topk.tolist()
            # Remap the flat-slot view to the more readable entry-visibility view via
            # the indexer's per-query top-k picks (one-hot + OR-reduce sidesteps the
            # duplicate-index clobber that `scatter_` would hit on warm-up sentinels).
            T_entries = meta["T_entries"]
            valid = topk >= 0
            safe = topk.clamp(min=0)
            oh = F.one_hot(safe, num_classes=T_entries).bool() & valid.unsqueeze(-1)
            em = oh.any(dim=-2)  # [B, S, T_entries]
            entry_bias = torch.where(em, torch.zeros((), dtype=torch.float32), torch.full((), float("-inf")))
            display_mask = torch.cat([sliding_mask, entry_bias.unsqueeze(1)], dim=-1)
            # Red-highlight cells: causally available entry that the indexer rejected.
            causal_threshold = torch.arange(seq_len).add(1).floor_divide(attn.compressor.compress_rate)
            entry_idx = torch.arange(T_entries)
            causally_available = entry_idx.view(1, -1) < causal_threshold.view(-1, 1)
            rejected = causally_available & ~em[0]
            meta["indexer_rejected"] = torch.cat(
                [torch.zeros((seq_len, seq_len), dtype=torch.bool), rejected], dim=-1
            )
        else:
            display_mask = sliding_mask

        captured[layer_idx] = (layer_type, display_mask, meta)

    color = os.environ.get("NO_COLOR") is None and os.isatty(1)

    svg_dir = main.svg_dir if hasattr(main, "svg_dir") else None
    if svg_dir is not None:
        svg_dir.mkdir(parents=True, exist_ok=True)

    print()
    for layer_idx in sorted(captured):
        layer_type, mask, meta = captured[layer_idx]
        if mask is None:
            continue
        if mask.ndim == 4:
            mask_2d = mask[0, 0]
        elif mask.ndim == 3:
            mask_2d = mask[0]
        else:
            mask_2d = mask

        q_labels = [f"q{i}" for i in range(mask_2d.shape[0])]
        kv_labels_sliding = [f"k{i}" for i in range(seq_len)]
        n_compressor = mask_2d.shape[1] - seq_len
        kv_labels_compressor = [f"c{i}" for i in range(n_compressor)]
        kv_labels = kv_labels_sliding + kv_labels_compressor

        title = f"Layer {layer_idx}: {layer_type}"
        print(_render(mask_2d, q_labels, kv_labels, title, color=color, meta=meta))
        print()
        if svg_dir is not None:
            svg_path = svg_dir / f"deepseek_v4_mask_layer{layer_idx}_{layer_type}.svg"
            svg_path.write_text(_render_svg(mask_2d, title, sliding_kv=seq_len, meta=meta))
            print(f"  wrote {svg_path}")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--svg",
        type=Path,
        default=None,
        help="Directory to write one SVG per layer (in addition to the ANSI / plain-text terminal output).",
    )
    args = parser.parse_args()
    main.svg_dir = args.svg  # type: ignore[attr-defined]
    main()
