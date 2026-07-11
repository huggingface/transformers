"""Turn an ``ArchModel`` / ``ArchDiff`` into positioned boxes.

Layout is deterministic (stable ordering, integer coordinates) so the emitted SVG is
diff-able in git. The visual vocabulary is *standardized*: a given component kind always
maps to the same shape class and palette slot, mirroring the library's "standardize, don't
abstract" tenet -- an attention block looks the same across every model's diagram.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .engine import text_width
from .introspect import ArchModel
from .modular import ArchDiff, ClassChange


_PARENT_CACHE: dict = {}  # model_type -> ArchModel of the modular parent (for diff comparison)


def _parent_arch(model_type):
    if model_type not in _PARENT_CACHE:
        try:
            from .introspect import introspect

            _PARENT_CACHE[model_type] = introspect(model_type)
        except Exception:
            _PARENT_CACHE[model_type] = None
    return _PARENT_CACHE[model_type]


def _collect_submodule_names(nodes, out: set):
    for nd in nodes:
        out.add(nd["name"])
        if nd.get("children"):
            _collect_submodule_names(nd["children"], out)


# component kind -> css class used by render.py (the standardized vocabulary)
KIND_CLASS = {
    "embedding": "c-embed",
    "attention": "c-attn",
    "mamba": "c-mamba",
    "linear_attention": "c-linattn",
    "recurrent": "c-recur",
    "moe": "c-moe",
    "mlp": "c-mlp",
    "norm": "c-norm",
    "head": "c-head",
    "config": "c-config",
    "rope": "c-rope",
    "layer": "c-layer",
}

# diff change -> css class
CHANGE_CLASS = {"added": "ch-added", "overridden": "ch-over", "deleted": "ch-deleted", None: ""}


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    label: str
    cls: str = ""
    sublabels: list[str] = field(default_factory=list)
    badge: str | None = None  # small corner tag, e.g. "×32"
    ghost: bool = False
    change: str | None = None  # added | overridden | deleted | None
    title: str | None = None  # SVG <title> tooltip
    shape: str = "rect"  # rect | circle | container | op | io | cell | grid
    small: bool = False  # render with smaller fonts (inner ops, Q/K/V chips)
    glyph: str | None = None  # single char drawn centered (e.g. "+", "×")
    grid: list[list[int]] | None = None  # for shape == "grid": 0/1 attention-mask matrix
    grid_split: int | None = None  # column index dividing sliding K/V | compressed cache


@dataclass
class Arrow:
    """A poly-line connector. ``points`` are (x, y) pairs; rendered with an arrowhead."""

    points: list[tuple[int, int]]
    cls: str = "flow"  # flow | residual
    label: str | None = None
    dashed: bool = False


@dataclass
class Diagram:
    width: int
    height: int
    boxes: list[Box]
    title: str
    subtitle: str = ""
    legend: list[tuple[str, str]] = field(default_factory=list)  # (css-class, text)
    facts: list[tuple[str, str]] = field(default_factory=list)  # side panel key/value
    changes: list[tuple[str, str, str]] = field(default_factory=list)  # (type, class, detail)
    arrows: list[Arrow] = field(default_factory=list)
    mode: str = "full"
    spine: bool = True  # draw the central connector spine


# geometry constants
W = 920
COL_X = 150  # left edge of the central stack
COL_W = 420  # width of the central stack
PAD = 28
ROW_H = 30


def _fmt_val(v) -> str:
    """Compactly format a config value -- including sequence-valued fields.

    Some (often hybrid) models carry per-layer lists for ``intermediate_size``,
    ``head_dim`` or ``sliding_window``. We collapse those to ``[x]×N`` (all equal) or
    ``[a … z] (N)`` rather than dumping the whole list. No per-model handling.
    """
    if v is None:
        return "—"
    if isinstance(v, (list, tuple)):
        if not v:
            return "[]"
        if len(set(v)) == 1:
            return f"[{_fmt_val(v[0])}]×{len(v)}"
        return f"[{_fmt_val(v[0])} … {_fmt_val(v[-1])}] ({len(v)})"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _facts(am: ArchModel) -> list[tuple[str, str]]:
    fmt = _fmt_val

    mid = am.checkpoint or am.model
    if len(mid) > 30:
        mid = mid[:29] + "…"
    rows = [
        ("model id", mid),
        ("model_type", am.model_type or "—"),
        ("config", am.config_class or "—"),
        ("decoder", am.decoder_kind),
        ("layers", fmt(am.num_layers)),
        ("hidden", fmt(am.hidden_size)),
        ("intermediate", fmt(am.intermediate_size)),
        ("heads / kv", f"{am.num_attention_heads} / {am.num_kv_heads}"),
        ("attention", am.attn_variant or "—"),
        ("norm", am.norm_type or "—"),
        ("positional", am.positional or "—"),
        ("vocab", fmt(am.vocab_size)),
        ("max_pos", fmt(am.max_position_embeddings)),
    ]
    if am.sliding_window:
        rows.append(("sliding_window", fmt(am.sliding_window)))
    if am.is_moe:
        rows.append(("experts (top-k)", f"{am.num_experts} (top-{am.experts_per_token})"))
    rows.append(("tie_embeddings", str(am.tie_word_embeddings)))
    if am.status != "ok":
        rows.append(("status", am.status))
    return rows


def _reflow(boxes: list[Box], start_y: int = PAD + 70, gap: int = 16) -> int:
    """Re-stack boxes vertically in list order using their (possibly grown) heights.

    Called after any box height changes so diff annotations never overlap neighbours.
    Returns the y coordinate just below the last box.
    """
    y = start_y
    for b in boxes:
        b.y = y
        y += b.h + gap
    return y


def _build_compact(am: ArchModel) -> Diagram:
    """Top-down view from CONFIG alone — used when the model can't be instantiated on meta (so
    there's no module tree to decompose) or has no transformer block (e.g. conv backbones).
    Shows input → embedding → the per-config layer blocks (collapsed, ×N) → final norm → head,
    family-aware, with the layer-type schedule. Far cleaner than the old bottom-up stack."""
    boxes: list[Box] = []
    x, w = COL_X, COL_W
    y = PAD + 78
    modality = (
        "vision"
        if am.family == "image_classification"
        else ("audio" if am.family == "audio_classification" else "text")
    )
    enc = am.view == "encoder"

    def add(label, cls, subs=None, h=None, shape="op", badge=None):
        nonlocal y
        b = Box(
            x,
            y,
            w,
            h if h is not None else (30 + 15 * len(subs or [])),
            label,
            cls,
            sublabels=subs or [],
            shape=shape,
            small=True,
            badge=badge,
        )
        boxes.append(b)
        y = y + b.h + 16
        return b

    # input + embedding (family-aware)
    if modality == "vision":
        add("input image", "c-io", ["pixel_values [1, 3, H, W]"], h=30, shape="io")
        add("Patch Embedding", "c-embed", [f"→ [1, S, {am.hidden_size or '?'}]"], h=40)
    elif modality == "audio":
        add("input audio", "c-io", ["input_features"], h=30, shape="io")
        add("Feature Projection", "c-embed", [f"→ [1, S, {am.hidden_size or '?'}]"], h=40)
    else:
        add("input_ids", "c-io", ["[1, S]"], h=30, shape="io")
        add("Token Embedding", "c-embed", [f"[{am.vocab_size or '?'} × {am.hidden_size or '?'}]"], h=40)

    # per-config layer blocks (collapsed ×N), coloured by kind
    for b in am.layer_blocks:
        inner = [f"{b.norm or 'Norm'} → {(b.attn_variant or b.kind.replace('_', '-'))}"]
        inner.append(
            f"{b.norm or 'Norm'} → {('MoE ' + str(am.num_experts) + 'E·top-' + str(am.experts_per_token)) if b.mlp_kind == 'moe' else 'MLP'}"
        )
        cls = KIND_CLASS["moe"] if b.mlp_kind == "moe" else KIND_CLASS.get(b.kind, KIND_CLASS["layer"])
        add(b.layer_class, cls, inner, badge=f"×{b.count}", h=30 + 15 * len(inner))

    add(f"Final {am.norm_type or 'Norm'}", "c-norm", h=28)
    if enc:
        add("Pool + Classifier head", "c-head", [f"→ class logits [1, {am.num_labels or '?'}]"], h=40)
    else:
        add("LM Head", "c-head", [f"→ logits [1, S, {am.vocab_size or '?'}]"], h=40)

    facts = _facts(am)
    height = max(y + PAD, PAD + 78 + len(facts) * 22 + 120)
    legend = _STD_LEGEND
    note = (
        "config-only — built from config (weights not instantiable on meta)"
        if am.status != "ok"
        else "no transformer block (conv backbone) — shown from config"
    )
    return Diagram(
        width=W,
        height=height,
        boxes=boxes,
        title=am.model,
        subtitle=f"⚠ {note}  ·  {am.decoder_kind} · {am.layer_summary or ''}",
        legend=legend,
        facts=facts,
        mode="full",
        spine=True,
    )


def _build_generic(am: ArchModel) -> Diagram:
    """Universal top-down render from the real (collapsed) module tree — for any model that is
    not a standard attention/SSM transformer: conv backbones (ResNet/ConvNeXt/MobileNet),
    audio codecs (EnCodec/DAC), FFT mixers (FNet), detectors, pose/segmentation heads, ...
    Every ``ModuleList``/``Sequential`` of identical blocks is collapsed to a ``×N`` section."""
    boxes: list[Box] = []
    GX, GW = 40, 850  # tree column (facts panel sits at width-288)
    y = PAD + 78
    PFX = (am.config_class or "").replace("Config", "")

    def add(label, cls, subs=None, h=30, shape="op"):
        nonlocal y
        b = Box(GX, y, GW, h, label, cls, sublabels=subs or [], shape=shape, small=True)
        boxes.append(b)
        y = b.y + b.h + 14
        return b

    # input (family-aware) — what the model consumes
    modality = (
        "vision" if am.family == "image_classification" else ("audio" if am.family == "audio_classification" else None)
    )
    if "pixel_values" in (am.modal_inputs or []) or modality == "vision":
        add("input image", "c-io", ["pixel_values [1, 3, H, W]"], shape="io")
    elif "input_features" in (am.modal_inputs or []) or modality == "audio":
        add("input audio", "c-io", ["input_values / input_features"], shape="io")
    else:
        add("inputs", "c-io", ["input_ids / pixel_values / input_values"], shape="io")

    # the real module hierarchy
    y = _emit_tree(boxes, am.generic_tree, GX, GW, y, gp=8, prefix=PFX)

    facts = _facts(am)
    height = max(y + PAD, PAD + 78 + len(facts) * 22 + 120)
    legend = _STD_LEGEND
    return Diagram(
        width=WF,
        height=height,
        boxes=boxes,
        title=am.model,
        subtitle=f"{am.top_class or ''}  ·  full module tree (no attention — shown structurally)",
        legend=legend,
        facts=facts,
        mode="full",
        spine=False,
    )


def _struct_hint(node: dict | None) -> str | None:
    """A one-line structural summary of a stage's children, e.g. '4× DacEncoderBlock'."""
    if not node:
        return None
    reps = [c["cls"] for c in node.get("children", []) if "×" in c.get("cls", "")]
    if reps:
        return " · ".join(reps[:2])
    kids = node.get("children", [])
    if kids:
        return f"{len(kids)} submodules"
    return None


def _shape_str(s) -> str:
    return "×".join(str(d) for d in s) if s else "?"


def _build_flow(am: ArchModel) -> Diagram:
    """Clean forward data-flow: the real input tensor flows top-to-bottom through each stage (in
    call order, with the shape it produces) to the output tensor. Shapes come from one dummy
    forward pass (meta, or a tiny CPU model). Used for codecs / conv backbones / detectors where
    the data path — not an attention block — is the story (DAC, EnCodec, BLT, ResNet, ...)."""
    from .introspect import _classify_mod

    boxes: list[Box] = []
    arrows: list[Arrow] = []
    GX, GW = 300, 580  # flow column (facts panel sits at width-288)
    y = PAD + 80
    PFX = (am.config_class or "").replace("Config", "")
    tree_by = {n["name"]: n for n in am.generic_tree}

    def place(label, cls, subs, shape="op"):
        nonlocal y
        b = Box(GX, y, GW, 34 + 15 * len(subs), label, cls, sublabels=subs, shape=shape, small=True)
        boxes.append(b)
        y = b.y + b.h
        return b

    # stage list + shapes: prefer the captured forward (true call order + shapes); otherwise fall
    # back to the structural top-level stages (definition order, no shapes) — always memory-free.
    shapes = {s["name"]: s.get("out_shape") for s in am.flow}
    if am.flow:
        stages = [(s["name"], s["cls"]) for s in am.flow]
        has_shapes = True
    else:
        stages = [(n["name"], n["cls"]) for n in am.generic_tree]
        has_shapes = False

    # input box
    if am.flow_input:
        in_label, in_sub = am.flow_input, []
    else:
        mods = am.modal_inputs or []
        if "pixel_values" in mods or am.family == "image_classification":
            in_label, in_sub = "input image", ["pixel_values [1, 3, H, W]"]
        elif "input_features" in mods or am.family == "audio_classification":
            in_label, in_sub = "input audio", ["input_values / input_features"]
        else:
            in_label, in_sub = "inputs", ["input_ids / pixel_values / input_values"]
    prev = place(in_label, "c-io", in_sub, shape="io")

    for name, cls in stages:
        y += 24  # arrow gap
        kind = _classify_mod(name, cls)
        node = tree_by.get(name)
        shp = shapes.get(name)
        shape_lbl = f"   →  [{_shape_str(shp)}]" if shp else ""
        hdr = f"{name}  ·  {_short_cls(cls, PFX)}{shape_lbl}"
        sec_top = y
        if node and node.get("children"):  # expose the stage's internal structure (×N blocks)
            y += 28  # header strip
            y = _emit_tree(boxes, node["children"], GX + 10, GW - 20, y, gp=7, prefix=PFX)
            anchor = Box(GX, sec_top, GW, y - sec_top + 6, hdr, _NODE_CLS.get(kind, "c-layer"), shape="section")
            boxes.append(anchor)
            y += 6
        else:
            anchor = place(hdr, _NODE_CLS.get(kind, "c-layer"), [])
        arrows.append(Arrow([(prev.x + prev.w / 2, prev.y + prev.h), (anchor.x + anchor.w / 2, sec_top)], cls="flow"))
        prev = anchor
    if am.flow_output:
        y += 24
        b = place(f"output  [{_shape_str(am.flow_output)}]", "c-io", [], shape="io")
        arrows.append(Arrow([(prev.x + prev.w / 2, prev.y + prev.h), (b.x + b.w / 2, b.y)], cls="flow"))

    facts = _facts(am)
    height = max(y + PAD, PAD + 80 + len(facts) * 22 + 120)
    sub_note = "forward data flow  (shapes from a dummy input pass)" if has_shapes else "staged data flow  (structure)"
    return Diagram(
        width=WF,
        height=height,
        boxes=boxes,
        arrows=arrows,
        title=am.model,
        subtitle=f"{am.top_class or ''}  ·  {sub_note}",
        legend=_STD_LEGEND,
        facts=facts,
        mode="full",
        spine=False,
    )


# --------------------------------------------------------------- detailed full (Raschka)

# geometry for the detailed view (wider canvas: schedule strip · column · masks · facts)
WF = 1200
STRIP_X = 40  # layer-schedule strip
DCOL_X = 320  # left edge of central op column
DCOL_W = 330  # width of central op column
RAIL_X = 286  # residual rail x (left of the column)
MASK_X = 700  # attention-mask grids column
SEQ = 12  # dummy batch/seq for displayed tensor shapes: [1, 12]

# layer_type -> color class (see render._STYLE)
_LT_CLASS = {
    "full_attention": "c-lt-full",
    "sliding_attention": "c-lt-sliding",
    "chunked_attention": "c-lt-chunked",
    "compressed_sparse_attention": "c-lt-compressed",
    "heavily_compressed_attention": "c-lt-heavy",
    "linear_attention": "c-lt-linear",
}


def _lt_class(t: str) -> str:
    if t in _LT_CLASS:
        return _LT_CLASS[t]
    n = t.lower()
    if "linear" in n or "delta" in n or "gated" in n:
        return "c-lt-linear"
    if "mamba" in n or "ssm" in n:
        return "c-lt-mamba"
    if "sliding" in n:
        return "c-lt-sliding"
    if "chunk" in n:
        return "c-lt-chunked"
    if "compress" in n:
        return "c-lt-compressed"
    return "c-lt-full"


def _dim(p) -> str:
    if p is None:
        return ""
    return f"[{p.in_f or '?'}→{p.out_f or '?'}]"


# module-tree node kind -> css class (for the recursive decomposition)
# standardized node kind -> css class, shared by every view (attention internals, SSM mixer,
# MoE, and the generic conv/codec/FFT module tree) so a given component is always the same colour.
_NODE_CLS = {
    "embedding": "c-embed",
    "linear": "c-proj",
    "conv": "c-conv",
    "norm": "c-norm",
    "act": "c-act",
    "pool": "c-pool",
    "dropout": "c-sub",
    "rope": "c-rope",
    "attention": "c-attn",
    "mamba": "c-mamba",
    "recurrent": "c-recur",
    "quantizer": "c-quant",
    "compressor": "c-lt-compressed",
    "indexer": "c-lt-heavy",
    "router": "c-moe",
    "experts": "c-moe",
    "head": "c-head",
    "other": "c-sub",
}

# one standardized legend shared by all full-view layouts (only the colours that can appear)
_STD_LEGEND = [
    ("c-embed", "embedding"),
    ("c-attn", "attention"),
    ("c-mamba", "Mamba / SSM"),
    ("c-moe", "MoE / experts"),
    ("c-proj", "linear / proj"),
    ("c-conv", "convolution"),
    ("c-norm", "normalization"),
    ("c-act", "activation"),
    ("c-pool", "pooling"),
    ("c-head", "head"),
]


def _short_cls(cls: str, prefix: str = "") -> str:
    """Drop the model prefix and the redundant 'Activation' suffix: DeepseekV4RMSNorm→RMSNorm,
    SiLUActivation→SiLU."""
    s = cls
    if prefix and s.startswith(prefix):
        s = s[len(prefix) :]
    if s.endswith("Activation") and len(s) > 10:
        s = s[: -len("Activation")]
    return s or cls


def _leaf_label(nd, prefix: str) -> str:
    """One compact line per leaf: name + dims; class shown only when it isn't a plain Linear."""
    sc = _short_cls(nd["cls"], prefix)
    dim = nd.get("dim")
    if sc == "Linear":  # 'Linear' is implied by the [in→out] arrow
        return f"{nd['name']}  {dim}" if dim else nd["name"]
    return f"{nd['name']}  {sc}" + (f" {dim}" if dim else "")


def _emit_tree(boxes, nodes, x, w, y, gp=7, depth=0, prefix=""):
    """Recursively lay out a module tree: leaves are single-line chips, branches become titled
    sub-sections containing their own children. Returns the y below the laid-out tree."""
    i = 0
    while i < len(nodes):
        nd = nodes[i]
        if nd.get("children"):  # a branch -> nested titled sub-section
            sub_top = y
            y += 25  # clear the white header strip
            y = _emit_tree(boxes, nd["children"], x + 8, w - 16, y, gp, depth + 1, prefix)
            label = f"{nd['name']} · {nd['cls']}" + (f"  ({nd['dim']})" if nd.get("dim") else "")
            boxes.append(
                Box(x, sub_top, w, y - sub_top + 6, label, _NODE_CLS.get(nd["kind"], "c-sub"), shape="section")
            )
            y += 6 + gp
            i += 1
        else:  # a run of leaf chips — content-sized: columns chosen by width, 2-line when needed
            run = []
            while i < len(nodes) and not nodes[i].get("children"):
                run.append(nodes[i])
                i += 1
            # how many columns fit? each chip wants ~its text width; clamp 1..3 by available w
            widest = max((text_width(_leaf_label(nd2, prefix), 12.5) for nd2 in run), default=80) + 18
            per = max(1, min(3, int(w // max(widest, 120))))
            for r in range(0, len(run), per):
                rown = run[r : r + per]
                cw = (w - (len(rown) - 1) * gp) / len(rown)
                # a chip is 2 lines (name / class-dim) when the one-line label doesn't fit its cell
                two = [text_width(_leaf_label(nd2, prefix), 12.5) > cw - 12 for nd2 in rown]
                rh = 34 if any(two) else 22
                for j, nd2 in enumerate(rown):
                    ttl = f"{nd2['name']}: {nd2['cls']}" + (f" {nd2['dim']}" if nd2.get("dim") else "")
                    if two[j]:  # split onto two lines instead of ellipsizing the dims away
                        sc = _short_cls(nd2["cls"], prefix)
                        sub = (
                            (sc + (f" {nd2['dim']}" if nd2.get("dim") else ""))
                            if sc != "Linear"
                            else (nd2.get("dim") or "")
                        )
                        boxes.append(
                            Box(
                                int(x + j * (cw + gp)),
                                y,
                                int(cw),
                                rh,
                                nd2["name"],
                                _NODE_CLS.get(nd2["kind"], "c-sub"),
                                sublabels=[sub] if sub else [],
                                shape="op",
                                small=True,
                                title=ttl,
                            )
                        )
                    else:
                        boxes.append(
                            Box(
                                int(x + j * (cw + gp)),
                                y,
                                int(cw),
                                rh,
                                _leaf_label(nd2, prefix),
                                _NODE_CLS.get(nd2["kind"], "c-sub"),
                                shape="op",
                                small=True,
                                title=ttl,
                            )
                        )
                y += rh + gp
    return y


def _node_sig(node):
    """Structural signature of a module node (class + dims + recursive children)."""
    return (node.get("cls"), node.get("dim"), tuple(_node_sig(c) for c in (node.get("children") or [])))


def _emit_merged(boxes, variants, types, x, w, y, gp=7, prefix=""):
    """Render N aligned module trees, sharing identical sub-trees and branching side-by-side
    ONLY where they actually differ. For DeepSeek-V4 attention this shows q/k/v/o once and the
    differing `compressor` (HCA vs CSA+Indexer) side by side -- the real subtlety."""
    base = variants[0]
    names = [n["name"] for n in base]
    if any([n["name"] for n in v] != names for v in variants):  # misaligned -> full side-by-side
        nn = len(variants)
        cw = (w - (nn - 1) * 18) / nn
        mb = y
        for k, v in enumerate(variants):
            mb = max(mb, _emit_tree(boxes, v, int(x + k * (cw + 18)), int(cw), y, gp, prefix=prefix))
        return mb
    for idx in range(len(names)):
        nodes = [v[idx] for v in variants]
        if all(_node_sig(nd) == _node_sig(nodes[0]) for nd in nodes):  # identical -> render once
            y = _emit_tree(boxes, [nodes[0]], x, w, y, gp, prefix=prefix)
        else:  # diverges -> side by side, tagged by layer type
            nn = len(nodes)
            cw = (w - (nn - 1) * gp) / nn
            top = y
            mb = y
            for k, nd in enumerate(nodes):
                cx = int(x + k * (cw + gp))
                boxes.append(Box(cx, top, int(cw), 18, f"▼ {types[k]}", "c-sub", shape="op", small=True))
                mb = max(mb, _emit_tree(boxes, [nd], cx, int(cw), top + 22, gp, prefix=prefix))
            y = mb
    return y


def _short_proj(name: str) -> str:
    return name.replace("_proj", "").replace("_with_mqa", "·mqa").replace("kv_a", "kv↓").replace("kv_b", "kv↑")


def _chips(boxes, items, x, w, y, per_row, cls="c-sub", gp=8, h=34):
    """Place `items` = list of (label, sublabel) as a grid of small boxes; return new y."""
    rows = [items[i : i + per_row] for i in range(0, len(items), per_row)]
    for r in rows:
        cw = (w - (len(r) - 1) * gp) / len(r)
        for j, (lb, sub) in enumerate(r):
            boxes.append(
                Box(
                    int(x + j * (cw + gp)),
                    y,
                    int(cw),
                    h,
                    lb,
                    cls,
                    sublabels=([sub] if sub else []),
                    shape="op",
                    small=True,
                )
            )
        y += h + gp
    return y


def _shape3(h) -> str:
    return f"[1, {SEQ}, {h or '?'}]"


def build_full(am: ArchModel) -> Diagram:
    """Dispatch: multimodal models get the multi-tower view, everything else the LLM column."""
    if am.is_multimodal and am.towers and am.block is not None:
        return build_multimodal(am)
    # multi-stage pipelines (BLT, codecs, ...) read best as a data flow even when they use attention
    if am.is_pipeline and am.generic_tree:
        return _build_flow(am)
    return _build_column(am)


def _build_column(am: ArchModel) -> Diagram:
    """Detailed, Raschka-style view, read **top-to-bottom like the code**: input_ids at the
    top flow down through the embedding, the expanded transformer block (pre-norm →
    self-attention with RoPE/Q/K/V/O and real dims → ⊕ residual → pre-norm → MLP or Sparse
    MoE → ⊕ residual), final norm, LM head and softmax to the logits at the bottom. A
    left-side strip shows the per-layer schedule (config.layer_types); attention-mask grids
    for each distinct layer type are shown on the right. Tensor shapes use a dummy [1, 12]
    input. Falls back to the compact view when block internals can't be extracted."""
    if am.block is None or (am.block.attention is None and am.block.mixer is None and am.block.mlp is None):
        # a real module tree → staged data-flow view (with shapes where a meta forward succeeded);
        # only models that couldn't be built on meta fall back to the config-only schematic.
        return _build_flow(am) if am.generic_tree else _build_compact(am)

    blk = am.block
    blk_is_ssm = blk.attention is None and blk.mixer is not None  # Mamba/SSM token-mixer block
    GAP = 14
    boxes: list[Box] = []
    arrows: list[Arrow] = []
    y = PAD + 80
    H = am.hidden_size
    PFX = (am.config_class or "").replace("Config", "")  # model prefix to strip from class names

    def add(label, cls, subs=None, h=None, shape="op", glyph=None, w=DCOL_W, x=DCOL_X, title=None, grid=None):
        nonlocal y
        b = Box(
            x,
            y,
            w,
            h if h is not None else (30 + 15 * len(subs or [])),
            label,
            cls,
            sublabels=subs or [],
            shape=shape,
            small=True,
            glyph=glyph,
            title=title,
            grid=grid,
        )
        boxes.append(b)
        y = y + b.h + GAP
        return b

    # which kind of model -> what the input / embedding / head look like
    modality = (
        "vision"
        if am.family == "image_classification"
        else ("audio" if am.family == "audio_classification" else "text")
    )
    enc = am.view == "encoder"
    seq = len(am.tokens or [1] * 6) or 6

    def _shapeN(h):
        return f"[1, {seq}, {h or '?'}]"

    # ---------- top: input ----------
    if modality == "vision":
        add("input image", "c-io", ["pixel_values [1, 3, H, W]"], h=32, shape="io")
        add("Patch Embedding", "c-embed", [f"conv patches → {_shapeN(H)}", "+ position embeddings"], h=60)
    elif modality == "audio":
        add("input audio", "c-io", ["input_features [1, n_mels, T]"], h=32, shape="io")
        add("Feature Projection", "c-embed", [f"conv/linear → {_shapeN(H)}", "+ position embeddings"], h=60)
    else:
        toks = am.tokens or ["Hey", ",", "␣how", "␣are", "␣you", "?"]
        seq = len(toks)
        add("input_ids", "c-io", [f'tokenize("Hey, how are you?")  →  [1, {seq}]'], h=32, shape="io")
        tgp = 5
        tw = (DCOL_W - (seq - 1) * tgp) / seq
        for j, t in enumerate(toks):
            label = t if len(t) <= 6 else t[:6] + "…"
            boxes.append(
                Box(
                    int(DCOL_X + j * (tw + tgp)),
                    y,
                    max(14, int(tw)),
                    26,
                    label,
                    "c-sub",
                    sublabels=[str(j)],
                    shape="op",
                    small=True,
                    title=f"token {j}: {t!r}",
                )
            )
        y += 26 + 18
        pos_note = [f"+ {am.positional} positions"] if (am.positional and am.positional not in ("RoPE", "n/a")) else []
        add(
            "Token Embedding",
            "c-embed",
            [f"weight [{am.vocab_size or '?'} × {H or '?'}]   →   {_shapeN(H)}"] + pos_note,
            h=30 + 15 * (1 + len(pos_note)),
        )
    body_top = boxes[-1].y  # the embedding box top — start of the "<Model>" body wrapper

    # inner-chip geometry shared by the attention / MoE / MLP sections
    IX, IW, GP = DCOL_X + 12, DCOL_W - 24, 8

    def section(label, cls, body, badge=None):
        """Draw a filled section panel containing inner chips produced by ``body(y)``."""
        nonlocal y
        sec_top = y
        y += 26  # header room
        y = body(y)
        sec = Box(DCOL_X, sec_top, DCOL_W, y - sec_top + 6, label, cls, shape="section", badge=badge)
        boxes.append(sec)
        y = y + 6 + GAP
        return sec

    # ---------- transformer block ----------
    y += 22  # headroom for the dashed container's "decoder block ×N" label
    block_top = y
    entry_y = block_top  # residual stream entry
    add(
        f"{blk.pre_attn_norm or 'Norm'}",
        "c-norm",
        [("pre-mixer  " if blk_is_ssm else "pre-attention  ") + _shapeN(H)],
        h=28,
    )

    # ----- self-attention: ONE section; shared sub-modules rendered once, and only the parts
    # ----- that actually differ across layer types (e.g. the compressor) branch side by side.
    a = blk.attention
    from .introspect import _short_layer_type

    attn_right = DCOL_X + DCOL_W
    variants = am.attention_variants
    if variants:
        n = min(len(variants), 3)
        types = [_short_layer_type(v["type"]) for v in variants[:n]]
        sc_by_type = {sc["type"]: sc for sc in am.sparse_components}
        diverges = n > 1
        Wsec = DCOL_W if not diverges else max(DCOL_W, n * 300)
        gqa = ""
        if a and a.n_heads and a.n_kv and a.n_kv < a.n_heads:
            gqa = f" · GQA {a.n_heads}:{a.n_kv}"
        elif a and a.n_kv == 1:
            gqa = " · MQA"
        vtop = y
        vy = vtop + 26
        if n == 1:
            vy = _emit_tree(boxes, variants[0]["children"], DCOL_X + 10, Wsec - 20, vy, gp=GP)
        else:
            vy = _emit_merged(boxes, [v["children"] for v in variants[:n]], types, DCOL_X + 10, Wsec - 20, vy, gp=GP)
        # core op footer (shared): SDPA for attention, selective scan for SSM/Mamba mixers
        is_mixer = a is None and blk.mixer is not None
        boxes.append(
            Box(
                DCOL_X + 10,
                vy,
                Wsec - 20,
                30,
                "selective state-space scan (SSM)" if is_mixer else "scaled dot-product attention",
                "c-sub",
                sublabels=([f"{a.n_heads or '?'} heads · head_dim {a.head_dim or '?'}{gqa}"] if a else []),
                shape="op",
                small=True,
            )
        )
        vy += 30 + GP
        # masks: one per layer type, side by side (this is where the compressors differ in effect)
        mask_items = [
            (v["type"], (sc_by_type.get(v["type"], {}).get("mask") or am.attn_patterns.get(v["type"])))
            for v in variants[:n]
        ]
        mask_items = [(t, g) for t, g in mask_items if g]
        if not mask_items:  # encoders attend bidirectionally; decoders are causal
            sq = 14
            if enc:
                grid = [[1] * sq for _ in range(sq)]
                mask_items = [("bidirectional", grid)]
            else:
                grid = [[1 if j <= i else 0 for j in range(sq)] for i in range(sq)]
                mask_items = [("causal", grid)]
        if mask_items:
            mw = (Wsec - 20 - (len(mask_items) - 1) * 16) / len(mask_items)
            mtop = vy + 4
            mb = mtop
            for k, (lt, grid) in enumerate(mask_items):
                sc = sc_by_type.get(lt)
                cols = len(grid[0])
                cellp = max(3, min(10, int((mw - 6) // cols)))
                gx = int(DCOL_X + 10 + k * (mw + 16) + (mw - cellp * cols) / 2)
                cap = (
                    f"{_short_layer_type(lt)} mask  m={sc['display_m']}·{sc['n_comp']}c"
                    if sc
                    else f"{_short_layer_type(lt)} mask"
                )
                boxes.append(
                    Box(
                        gx,
                        mtop + 6,
                        cellp * cols,
                        cellp * len(grid),
                        cap,
                        _lt_class(lt),
                        shape="grid",
                        grid=grid,
                        grid_split=(sc.get("mask_split") if sc else None),
                        title=f"{lt} attention mask (q↓ × k→)",
                    )
                )
                mb = max(mb, mtop + 6 + cellp * len(grid))
            vy = mb
        note = ""
        if diverges:
            sigs = [tuple(_node_sig(c) for c in v["children"]) for v in variants[:n]]
            struct_diff = any(s != sigs[0] for s in sigs)
            note = (
                f"  ({n} layer types — modules differ ▼)"
                if struct_diff
                else f"  ({n} layer types — same modules, mask differs)"
            )
        if is_mixer:
            hdr = f"Token Mixer (SSM) · {blk.mixer.cls}{note}"
        else:
            hdr = f"Self-Attention · {a.cls if a else 'attention'}{note}"
        boxes.append(Box(DCOL_X, vtop, Wsec, vy - vtop + 6, hdr, "c-attn", shape="section"))
        attn_right = DCOL_X + Wsec
        y = vy + 6 + GAP
        if a and a.rope:
            ry = vtop + 30
            rope_label = f"RoPE  θ={int(am.rope_theta):,}" if am.rope_theta else "RoPE"
            rnode = Box(
                DCOL_X - 168,
                ry - 4,
                132,
                40,
                rope_label,
                "c-rope",
                sublabels=["rotary positions → Q, K"],
                shape="op",
                small=True,
            )
            boxes.append(rnode)
            arrows.append(Arrow([(rnode.x + rnode.w, ry + 14), (DCOL_X, ry + 14)], cls="rope"))
    else:
        add("Token Mixing  (linear-attention / SSM)", "c-linattn", [blk.layer_class], h=44)
    plus_attn = add("", "c-add", h=26, shape="circle", glyph="+", title="residual add")

    # ----- MLP / Sparse MoE: FULL recursive decomposition, variants side by side -----
    # (Mamba/SSM blocks are norm → mixer → residual with NO feed-forward, so skip the FFN entirely)
    mm = blk.mlp
    mvars = am.mlp_variants if len(am.mlp_variants) > 1 else None
    has_mlp = bool(mm or (am.mlp_tree and am.mlp_tree.get("children")) or mvars)
    if has_mlp:
        add(f"{blk.post_attn_norm or 'Norm'}", "c-norm", ["pre-FFN  " + _shapeN(H)], h=28)
    if mvars:
        # distinct FFN types (e.g. DeepSeek-V4 moe / hash_moe): shared sub-modules once, only
        # the differing part (the router) branches side by side.
        nm = min(len(mvars), 3)
        Wm = max(DCOL_W, nm * 300)
        mtypes = [mv["type"] for mv in mvars[:nm]]
        vtop = y
        my = vtop + 26
        my = _emit_merged(boxes, [mv["children"] for mv in mvars[:nm]], mtypes, DCOL_X + 10, Wm - 20, my, gp=GP)
        cls0 = mvars[0]["cls"]
        boxes.append(
            Box(
                DCOL_X,
                vtop,
                Wm,
                my - vtop + 6,
                f"Sparse MoE · {cls0}  ({nm} FFN types ▼)",
                "c-moe",
                shape="section",
                badge="MoE",
            )
        )
        attn_right = max(attn_right, DCOL_X + Wm)
        y = my + 6 + GAP
    elif am.mlp_tree and am.mlp_tree.get("children"):
        cls = am.mlp_tree["cls"]
        is_moe = bool(mm and mm.is_moe)
        section(
            ("Sparse MoE · " if is_moe else "MLP · ") + cls,
            "c-moe" if is_moe else "c-mlp",
            lambda yy: _emit_tree(boxes, am.mlp_tree["children"], IX, IW, yy, gp=GP),
            badge="MoE" if is_moe else None,
        )
    elif mm:

        def mlp_body(y):
            items = []
            if mm.gate:
                items.append(("gate_proj", _dim(mm.gate)))
            if mm.up:
                items.append(("up_proj", _dim(mm.up)))
            per = len(items) if items else 1
            if items:
                y = _chips(boxes, items, IX, IW, y, per, gp=GP, h=32)
            boxes.append(
                Box(IX, y, IW, 28, f"{mm.act}(gate) ⊙ up" if mm.gate else mm.act, "c-sub", shape="op", small=True)
            )
            y += 28 + GP
            boxes.append(Box(IX, y, IW, 32, f"down_proj  {_dim(mm.down)}", "c-sub", shape="op", small=True))
            y += 32
            return y

        section(f"MLP · {mm.cls}", "c-mlp", mlp_body)
    elif has_mlp:
        add("MLP", "c-mlp", h=30)
    # second residual add only exists when there is an FFN; SSM blocks have a single residual
    plus_mlp = add("", "c-add", h=26, shape="circle", glyph="+", title="residual add") if has_mlp else plus_attn
    block_bot = y - GAP

    container_w = max(DCOL_W, attn_right - DCOL_X) + 52
    container = Box(
        DCOL_X - 26,
        block_top - 22,
        container_w,
        block_bot - block_top + 30,
        blk.layer_class,  # the real layer class, e.g. LlamaDecoderLayer / ASTLayer
        "c-block",
        shape="container",
        badge=f"× {am.num_layers or '?'}",
        title=f"{am.num_layers} × {blk.layer_class}",
    )

    # ---------- bottom: final norm, then the task head (depends on family) ----------
    add(f"Final {am.norm_type or 'Norm'}", "c-norm", [_shapeN(H)], h=28)
    body_bot = boxes[-1].y + boxes[-1].h  # end of the "<Model>" body (everything above = base model)
    head_title = am.head_class  # e.g. ViTForImageClassification / GemmaForCausalLM (the For wrapper)
    if enc:  # encoder / classifier head
        nl = am.num_labels or "?"
        add("Pool  (CLS / mean)", "c-head", [f"{_shapeN(H)} → [1, {H or '?'}]"], h=40, title=head_title)
        add("Classifier head", "c-head", [f"Linear [{H or '?'}→{nl}]"], h=40, title=head_title)
        add("Softmax", "c-soft", h=26)
        add("class logits", "c-io", [f"[1, {nl}]  ({nl} classes)"], h=30, shape="io")
    else:  # decoder LM head
        head_sub = [f"Linear [{H or '?'}→{am.vocab_size or '?'}]" + ("  · tied" if am.tie_word_embeddings else "")]
        add("LM Head", "c-head", head_sub, h=42, title=head_title)
        add("Softmax", "c-soft", h=26)
        add("logits", "c-io", [f"[1, {seq}, {am.vocab_size or '?'}]"], h=30, shape="io")

    # ---------- wrapper hierarchy: base <Model> body and <ForXxx> task head, STACKED (adjacent,
    # not nested) so the two labels never overlap ----------
    ww = (attn_right - DCOL_X) + 68
    if am.top_class:
        boxes.append(
            Box(
                DCOL_X - 34,
                body_top - 14,
                ww,
                body_bot - body_top + 20,
                f"{am.top_class}  ·  base model",
                "c-block",
                shape="container",
            )
        )
    if head_title:
        boxes.append(
            Box(
                DCOL_X - 34,
                body_bot + 6,
                ww,
                (y - GAP) - (body_bot + 6) + 8,
                f"{head_title}  ·  task head",
                "c-block",
                shape="container",
            )
        )

    # ---------- residual skip arrows (flow downward) ----------
    def midy(b):
        return b.y + b.h // 2

    arrows.append(
        Arrow(
            [(DCOL_X, entry_y), (RAIL_X, entry_y), (RAIL_X, midy(plus_attn)), (plus_attn.x, midy(plus_attn))],
            cls="residual",
            dashed=True,
        )
    )
    if has_mlp:
        arrows.append(
            Arrow(
                [
                    (plus_attn.x, midy(plus_attn)),
                    (RAIL_X - 16, midy(plus_attn)),
                    (RAIL_X - 16, midy(plus_mlp)),
                    (plus_mlp.x, midy(plus_mlp)),
                ],
                cls="residual",
                dashed=True,
            )
        )

    boxes.insert(0, container)

    # ---------- layer-schedule strip (config.layer_types) — one labelled cell per layer ----
    legend = [("c-attn", "self-attention")]
    strip_bottom = block_bot
    # only show the per-layer schedule when layers actually differ (≥2 distinct types)
    if am.layer_types and len(set(am.layer_types)) >= 2:
        from .introspect import _short_layer_type

        n = len(am.layer_types)
        SW = 46  # strip width
        cell_h = max(15, (block_bot - block_top) / n)  # at least 15px so the index fits
        boxes.append(Box(STRIP_X - 4, block_top - 24, SW + 12, 16, "layers (idx)", "c-block", shape="container"))
        for i, lt in enumerate(am.layer_types):
            cy = int(block_top + i * cell_h)
            boxes.append(
                Box(
                    STRIP_X,
                    cy,
                    SW,
                    max(13, int(cell_h) - 2),
                    str(i),
                    _lt_class(lt),
                    shape="cell",
                    title=f"layer {i}: {lt}",
                )
            )
        strip_bottom = int(block_top + n * cell_h)
        seen = []
        for lt in am.layer_types:
            if lt not in seen:
                seen.append(lt)
        for lt in seen:
            legend.append((_lt_class(lt), _short_layer_type(lt)))

    # (attention-mask grids are now drawn INSIDE each variant card, so no separate column)
    gy = 0
    legend += [
        ("c-proj", "linear / proj"),
        ("c-moe", "MoE / experts") if am.is_moe else ("c-mlp", "MLP / feed-forward"),
        ("c-rope", "RoPE"),
        ("c-norm", "normalization"),
        ("residual", "residual / skip"),
    ]

    facts = _facts(am)
    mask_bottom = gy
    height = max(y + PAD, strip_bottom + PAD, mask_bottom + PAD, PAD + 80 + len(facts) * 22 + len(legend) * 18 + 120)
    sub = f"{am.decoder_kind} · {am.layer_summary or ''}"
    width = max(WF, attn_right + 320)  # widen for side-by-side attention variants + facts panel
    return Diagram(
        width=width,
        height=height,
        boxes=boxes,
        arrows=arrows,
        title=am.model,
        subtitle=sub,
        legend=legend,
        facts=facts,
        mode="full",
        spine=True,
    )


# --------------------------------------------------------- multimodal multi-tower full view


def build_multimodal(am: ArchModel) -> Diagram:
    """VLM / audio view: encoder tower(s) → projector → the LLM (the detailed column).

    The LLM column is built by ``_build_column`` and shifted right; the modality towers are
    drawn in a left lane, with a fusion arrow into the LLM (merge at modality tokens), or
    cross-attention edges into specific LLM layers when the text config declares them.
    """
    d = _build_column(am)
    dx = 300
    for b in d.boxes:
        b.x += dx
    for a in d.arrows:
        a.points = [(x + dx, y) for (x, y) in a.points]
    d.width += dx

    emb = next((b for b in d.boxes if b.cls == "c-embed"), None)
    container = next((b for b in d.boxes if b.shape == "container" and b.cls == "c-block"), None)

    TX, TW = 36, 250
    ty = PAD + 92
    encoders = [t for t in am.towers if t["role"] in ("vision", "audio")]
    proj = next((t for t in am.towers if t["role"] == "projector"), None)

    mmpfx = (am.config_class or "").replace("Config", "")

    def tb(label, cls, subs, h, badge=None, shape="op"):
        nonlocal ty
        b = Box(TX, ty, TW, h, label, cls, sublabels=subs or [], shape=shape, small=True, badge=badge)
        d.boxes.append(b)
        ty += h + 16
        return b

    def tower(t, label, cls, summary, badge=None):
        """Render a tower: fully decomposed when small; a representative block ×N for big
        encoder stacks; else a summary box. So every tower shows what's inside it."""
        nonlocal ty
        ch = t.get("children")
        if not ch and t.get("block_children"):
            # big encoder: outer section + one representative layer (decomposed) with ×N
            top = ty
            inner_top = top + 26
            blabel = f"{t['block_class']}"
            by = _emit_tree(d.boxes, t["block_children"], TX + 18, TW - 36, inner_top + 22, gp=6, prefix=mmpfx)
            d.boxes.append(
                Box(
                    TX + 10,
                    inner_top,
                    TW - 20,
                    by - inner_top + 6,
                    blabel,
                    cls,
                    shape="section",
                    badge=f"×{t['block_n']}",
                )
            )
            b = Box(TX, top, TW, (by + 6) - top + 6, label, cls, shape="section", badge=badge)
            d.boxes.append(b)
            ty = by + 6 + 6 + 16
            return b
        if ch:
            top = ty
            yy = _emit_tree(d.boxes, ch, TX + 10, TW - 20, top + 26, gp=6, prefix=mmpfx)
            b = Box(TX, top, TW, yy - top + 6, label, cls, shape="section", badge=badge)
            d.boxes.append(b)
            ty = yy + 6 + 16
            return b
        return tb(label, cls, summary, 52, badge=badge)

    enc_boxes = []
    for t in encoders:
        if t["role"] == "vision":
            inp, ishape, cls, role = "pixel_values", "[1, 3, 336, 336]", "c-vision", "Vision encoder"
            img = Box(TX + (TW - 76) // 2, ty, 76, 64, "example image", "c-vision", shape="image")
            d.boxes.append(img)
            ty += 64 + 22
        else:
            inp, ishape, cls, role = "input_features", "[1, 128, 3000]", "c-audio", "Audio encoder"
            img = None
        ib = tb(inp, "c-io", [ishape], 30, shape="io")
        if img is not None:
            d.arrows.append(Arrow([(img.x + img.w // 2, img.y + img.h), (ib.x + ib.w // 2, ib.y)], cls="flow"))
        cls_line = t["cls"] + ("  · via AutoModel" if am.auto_classes else "")
        eb = tower(
            t,
            f"{role} · {t['cls']}",
            cls,
            [cls_line, f"{t['model_type'] or ''} · {t['layers'] or '?'} layers · h={t['hidden'] or '?'}"],
            badge=(f"×{t['layers']}" if t["layers"] else None),
        )
        d.arrows.append(Arrow([(ib.x + ib.w // 2, ib.y + ib.h), (eb.x + eb.w // 2, eb.y)], cls="flow"))
        enc_boxes.append(eb)

    pj = tower(
        proj or {}, f"Projector · {proj['cls'] if proj else 'Linear'}", "c-proj", ["align features → text hidden dim"]
    )
    for eb in enc_boxes:
        d.arrows.append(Arrow([(eb.x + eb.w // 2, eb.y + eb.h), (pj.x + pj.w // 2, pj.y)], cls="flow"))

    # fusion into the LLM
    px = pj.x + pj.w
    pmid = pj.y + pj.h // 2
    if am.cross_attention_layers and container is not None:
        n = len(am.cross_attention_layers)
        target_x, target_y = container.x, container.y + container.h // 2
        d.arrows.append(
            Arrow(
                [(px, pmid), ((px + target_x) // 2, pmid), ((px + target_x) // 2, target_y), (target_x, target_y)],
                cls="xattn",
            )
        )
        d.boxes.append(
            Box(
                int((px + target_x) // 2) - 70,
                target_y - 34,
                150,
                18,
                f"cross-attention @ {n} layers",
                "c-proj",
                shape="op",
                small=True,
            )
        )
    elif emb is not None:
        target_x, target_y = emb.x, emb.y + emb.h // 2
        d.arrows.append(
            Arrow(
                [(px, pmid), ((px + target_x) // 2, pmid), ((px + target_x) // 2, target_y), (target_x, target_y)],
                cls="flow",
            )
        )
        d.boxes.append(
            Box(
                int((px + target_x) // 2) - 78,
                target_y - 30,
                168,
                18,
                "merge at modality tokens",
                "c-proj",
                shape="op",
                small=True,
            )
        )

    # image ↔ text attention mask (the cross-modal mask the LLM actually uses)
    if any(t["role"] == "vision" for t in encoders):
        from .masks import image_text_mask

        n_img, n_text = 5, 7
        grid, split = image_text_mask(n_img, n_text, image_bidirectional=am.image_bidirectional)
        cellp = 13
        ty += 16  # gap below the projector
        mx = TX + (TW - cellp * len(grid)) // 2
        my = ty + 22
        kind = "prefix-LM: image bidirectional" if am.image_bidirectional else "fully causal"
        d.boxes.append(Box(TX - 4, ty, TW + 8, 16, "image ⊕ text attention", "c-block", shape="container"))
        d.boxes.append(
            Box(
                mx,
                my,
                cellp * len(grid),
                cellp * len(grid),
                f"{n_img} img + {n_text} text · {kind}",
                "c-vision",
                shape="grid",
                grid=grid,
                grid_split=split,
                title="image↓text query × image|text key",
            )
        )
        ty = my + cellp * len(grid) + 16

    # legend additions
    if any(t["role"] == "vision" for t in encoders):
        d.legend.insert(0, ("c-vision", "vision encoder"))
    if any(t["role"] == "audio" for t in encoders):
        d.legend.insert(0, ("c-audio", "audio encoder"))
    d.legend.append(("c-proj", "projector / fusion"))
    if am.cross_attention_layers:
        d.legend.append(("xattn", "cross-attention"))

    d.subtitle = f"multimodal · {'+'.join(t['role'] for t in encoders)} → projector → LLM · {am.decoder_kind}"
    d.facts.insert(1, ("inputs", ", ".join(am.modal_inputs)))
    if am.auto_classes:
        d.facts.insert(2, ("built via", ", ".join(am.auto_classes[:2])))
    d.height = max(d.height, ty + PAD)
    return d


# ----------------------------------------------------------------------------- diff mode


# map a modular class name to which architectural slot it touches.
# We match on SUFFIXES, not substrings: model names themselves often contain component
# keywords (e.g. "Qwen2Moe...") which would otherwise mis-route every class to "moe".
def _slot_for_class(name: str) -> str:
    n = name.lower()
    if n.endswith("config"):
        return "config"
    if n.endswith("rotaryembedding") or n.endswith("rotaryembeddings"):
        return "rope"
    if n.endswith(("router", "experts", "expert", "moeblock", "sparsemoeblock", "moe", "moemlp")):
        return "moe"
    if n.endswith(("mlp", "feedforward", "ffn")):
        return "mlp"
    if n.endswith(("attention", "attn", "sdpaattention", "flashattention2")):
        return "attention"
    if n.endswith(("rmsnorm", "layernorm", "norm")):
        return "norm"
    if (
        n.endswith(("pretrainedmodel",))
        or "forcausallm" in n
        or "forsequence" in n
        or "fortoken" in n
        or "forquestion" in n
        or "forconditional" in n
        or "forretrieval" in n
        or "withlmhead" in n
    ):
        return "head"
    if n.endswith(("decoderlayer", "encoderlayer", "layer", "block", "mixer")):
        return "layer"
    if n.endswith(("embedding", "embeddings")):
        return "embedding"
    if n.endswith("model"):
        return "layer"  # the decoder/model wrapper -> the stack
    return "layer"


def _change_for(cc: ClassChange) -> str | None:
    if cc.relation == "new":
        return "added"
    if cc.deleted_methods or cc.deleted_attrs:
        return "deleted"
    if cc.n_changes == 0:
        return None  # trivial / unchanged
    if cc.added_methods or cc.added_attrs:
        return "added"
    return "overridden"


def _change_detail(cc: ClassChange) -> str:
    bits = []
    if cc.overridden_methods:
        bits.append("ovr " + ",".join(cc.overridden_methods))
    if cc.added_methods:
        bits.append("add " + ",".join(cc.added_methods))
    if cc.deleted_methods:
        bits.append("del " + ",".join(cc.deleted_methods))
    na = len(cc.added_attrs) + len(cc.overridden_attrs)
    if na:
        bits.append(f"{na} attr")
    return "; ".join(bits) if bits else "rename only"


def build_diff(am: ArchModel, ad: ArchDiff) -> Diagram:
    """Diff over the *real architecture*: render the full detailed view, then colour each
    class-box by how it relates to the parent it inherits from -- added (green), overridden
    (amber), deleted (red). Everything inherited unchanged / copy-pasted is greyed (ghosted),
    so at a glance you see what the model is made of AND exactly what it changed."""
    d = build_full(am)
    d.mode = "diff"
    d.title = am.model

    if not ad.is_modular:
        # standalone: nothing inherited -> show the full architecture, note it
        d.subtitle = (ad.note or "standalone (no modular parent)") + " · shown in full"
        return d

    # change type + detail per modular class (skip trivial → treated as inherited/ghost).
    # Box colour: a class redefined in the modular file (even a "new" subclass like
    # GemmaRMSNorm, which reimplements the inherited norm) is a CHANGE -> amber, not green.
    # Green is reserved for genuinely net-new *submodules* (added below from the parent diff).
    by_cls: dict[str, str] = {}
    detail: dict[str, ClassChange] = {}
    changes_list: list[tuple[str, str, str]] = []
    rank = {"deleted": 0, "overridden": 1, "added": 2}
    for cc in ad.changes:
        ch = _change_for(cc)
        detail[cc.name] = cc
        if ch is not None:
            by_cls[cc.name] = "deleted" if ch == "deleted" else "overridden"
            changes_list.append((ch, cc.name, _change_detail(cc)))
    changes_list.sort(key=lambda t: (rank.get(t[0], 9), t[1]))

    # longest class names first so we match e.g. GemmaAttention before Gemma
    changed_names = sorted(by_cls, key=len, reverse=True)

    def match(text: str) -> str | None:
        for nm in changed_names:
            if nm in text:
                return by_cls[nm]
        return None

    def by_keyword(*kws: str) -> str | None:
        # the embedding/head boxes have generic labels; match their changed class by keyword
        for nm in changed_names:
            low = nm.lower()
            if any(k in low for k in kws):
                return by_cls[nm]
        return None

    # colour each box; ghost everything that didn't change (inherited / copy-pasted)
    for b in d.boxes:
        if b.shape in ("container", "io", "circle", "cell", "grid"):
            continue
        ch = match(f"{b.label or ''}  {b.title or ''}")
        if ch is None and b.cls == "c-embed":
            ch = by_keyword("embedding", "embed")
        if ch is None and b.cls == "c-head":
            ch = by_keyword("for", "lmhead", "classifier", "classification", "head")
        if ch:
            b.change = ch
            b.ghost = False
        else:
            b.ghost = True  # inherited unchanged / not a (changed) modular class

    # compare against the PARENT architecture itself, so overridden blocks reveal WHAT changed
    pam = _parent_arch(ad.parent_model) if ad.parent_model else None
    if pam is not None:
        pnames: set = set()
        for v in pam.attention_variants:
            _collect_submodule_names(v["children"], pnames)
        if pam.mlp_tree:
            _collect_submodule_names(pam.mlp_tree.get("children") or [], pnames)
        # a submodule present in the child but NOT in the parent is an addition (e.g. qwen3's
        # q_norm / k_norm vs qwen2) -> light it green even inside an "overridden" block
        if pnames:
            for b in d.boxes:
                if b.shape == "op" and b.title and ": " in b.title and not b.change:
                    nm, _, rest = b.title.partition(": ")
                    # only real submodule chips (title "name: ClassName"), not token/io chips
                    if not rest[:1].isupper():
                        continue
                    if nm and nm not in pnames:
                        b.change = "added"
                        b.ghost = False
        # inherited (ghosted) section headers: show the PARENT class they came from
        cpfx = (am.config_class or "").replace("Config", "")
        ppfx = (pam.config_class or "").replace("Config", "")
        if cpfx and ppfx and cpfx != ppfx:
            for b in d.boxes:
                if b.ghost and b.shape == "section" and cpfx in (b.label or ""):
                    b.label = b.label.replace(cpfx, ppfx) + "  ↩ inherited"

    t = ad.totals
    extra = f" (+{','.join(p for p in ad.parent_models if p != ad.parent_model)})" if len(ad.parent_models) > 1 else ""
    d.subtitle = (
        f"diff vs {ad.parent_model}{extra}  ·  {t['overridden']} overridden · {t['added']} added · "
        f"{t['deleted']} deleted · {t['new_classes']} new · {t['trivial']} inherited-as-is"
    )
    d.changes = changes_list
    d.legend = [
        ("ch-added", "new submodule (vs parent)"),
        ("ch-over", "changed / redefined"),
        ("ch-deleted", "deleted"),
        ("ghost", "inherited / copy-pasted"),
    ]
    d.facts = [
        ("model id", (am.checkpoint or am.model)[:30]),
        ("parent", ad.parent_model or "—"),
        ("classes", str(len(ad.changes))),
        ("overridden", str(t["overridden"])),
        ("added", str(t["added"])),
        ("new classes", str(t["new_classes"])),
        ("inherited as-is", str(t["trivial"])),
    ]
    # ensure height fits the changes panel
    panel_h = len(d.facts) * 22 + len(d.legend) * 18 + len(d.changes) * 28 + 200
    d.height = max(d.height, PAD + 80 + panel_h)
    return d
