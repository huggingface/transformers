# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Block-variant facet extraction for `transformers blocks`.

Every attention-bearing block in the library is described by an ordered vector of *facets*, split
into two tiers:

- **tier 1** changes the `forward` pass. An exact tier-1 match means `forward` is inheritable, so
  tier 1 alone decides block identity (the "variant") and therefore the tag.
- **tier 2** only changes `__init__` (bias flags, `head_dim`, eps, the `ACT2FN` key). It never
  gates a match, but it is always reported, because people routinely fork a whole class over a
  single tier-2 bit.

Stdlib only, on purpose: this module is imported by a repo-consistency checker and must not drag
torch into `make check-repo`. The two axis vocabularies that already exist in the library
(`LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING`, `ROPE_INIT_FUNCTIONS`) are read out of their source
files with `ast` rather than imported, so they cannot drift and cost nothing.
"""

import ast
import hashlib
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = REPO_ROOT / "src" / "transformers" / "models"
MODEL_DOC_ROOT = REPO_ROOT / "docs" / "source" / "en" / "model_doc"

# `utils/add_dates.py` maintains a "contributed to Hugging Face Transformers on YYYY-MM-DD" line in
# every model card. That is the date a variant entered *this codebase*, which is exactly the
# ordering a canonical owner needs -- not the paper date, which the same line also carries.
# Some cards use unicode dashes, so dates are matched after normalising them.
_CONTRIBUTED_RE = re.compile(r"contributed to Hugging Face Transformers on (\d{4}-\d{2}-\d{2})")
_RELEASED_RE = re.compile(r"was released on (\d{4}-\d{2}-\d{2})")
_DASHES = str.maketrans(dict.fromkeys("‐‑‒–—―", "-"))

# Module-level helpers worth fingerprinting: high copy count, low semantic variation.
TRACKED_HELPERS = ("repeat_kv", "rotate_half", "apply_rotary_pos_emb", "eager_attention_forward")

# Axis order == the "transformers format". Ordered by override cost, expensive first.
# `blocks compile --fit-order` re-derives this from real modular override sizes.
ATTENTION_AXES = ("mix", "qkv", "qk_norm", "window", "rope", "extras")
MLP_AXES = ("gating",)
MOE_AXES = ("router", "router_bias", "topk_norm", "shared", "weights", "grouping")
NORM_AXES = ("norm_kind",)
ROTARY_AXES = ("rope_kind",)

TIER1_AXES = {
    "attention": ATTENTION_AXES,
    "mlp": MLP_AXES,
    "moe": MOE_AXES,
    "norm": NORM_AXES,
    "rotary": ROTARY_AXES,
    "layer": ("topology",),
}


# --------------------------------------------------------------------------------------------------
# Vocabularies read out of the library source (never imported: these files pull in torch)
# --------------------------------------------------------------------------------------------------
def _dict_keys_from_source(path: Path, name: str) -> tuple[str, ...]:
    """Return the string keys of the module-level dict literal assigned to `name` in `path`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        if isinstance(node.value, ast.Dict):
            return tuple(k.value for k in node.value.keys if isinstance(k, ast.Constant))
    return ()


@cache
def layer_pattern_vocabulary() -> tuple[str, ...]:
    """The 11 `config.layer_types` values, straight from `masking_utils`."""
    return _dict_keys_from_source(
        REPO_ROOT / "src" / "transformers" / "masking_utils.py", "LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING"
    )


@cache
def rope_scaling_vocabulary() -> tuple[str, ...]:
    """The rope-scaling types, straight from `modeling_rope_utils`. `default` is implicit."""
    keys = _dict_keys_from_source(REPO_ROOT / "src" / "transformers" / "modeling_rope_utils.py", "ROPE_INIT_FUNCTIONS")
    return ("default",) + tuple(k for k in keys if k != "default")


# --------------------------------------------------------------------------------------------------
# Release dates -- the historic ordering that decides a variant's canonical owner (lint rule R3).
# --------------------------------------------------------------------------------------------------
def _git_first_commit_date(model: str) -> str | None:
    """Fallback for models with no dated card: when their directory first appeared in git."""
    try:
        out = subprocess.run(
            [
                "git",
                "log",
                "--diff-filter=A",
                "--format=%ad",
                "--date=short",
                "--",
                f"src/transformers/models/{model}",
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    # `git log` is newest-first, so the model's first appearance is the last line.
    lines = [line for line in out.splitlines() if line.strip()]
    return lines[-1] if lines else None


@cache
def build_date_data() -> dict[str, str]:
    """
    Build `{model: YYYY-MM-DD}`, the date each model entered the codebase.

    Primary source is the model card line that `utils/add_dates.py` already maintains and CI keeps
    fresh, so this costs one glob and no network. Models whose card has no date (or no card at all)
    fall back to the first commit that added their directory.
    """
    dates: dict[str, str] = {}
    for md_path in sorted(MODEL_DOC_ROOT.glob("*.md")):
        try:
            text = md_path.read_text(encoding="utf-8", errors="ignore").translate(_DASHES)
        except OSError:
            continue
        match = _CONTRIBUTED_RE.search(text) or _RELEASED_RE.search(text)
        if match:
            # Doc stems use dashes where model directories use underscores (`dab-detr` / `dab_detr`).
            dates[md_path.stem.replace("-", "_")] = match.group(1)

    for model_dir in sorted(p for p in MODELS_ROOT.iterdir() if p.is_dir()):
        if model_dir.name not in dates:
            found = _git_first_commit_date(model_dir.name)
            if found:
                dates[model_dir.name] = found
    return dates


# --------------------------------------------------------------------------------------------------
# Config-declared facts
# --------------------------------------------------------------------------------------------------
CONFIG_ATTENTION_KEYS = ("sliding_window", "attention_chunk_size", "layer_types")


@cache
def config_flags(model: str) -> frozenset[str]:
    """
    Which attention-pattern keys the model's config actually declares with a non-`None` default.

    This cannot be read off the attention class. Nearly every Llama descendant threads a
    `getattr(config, "sliding_window", None)` through `forward`, so the class body mentions sliding
    whether or not the architecture slides -- `OlmoeAttention` does, and olmoe's config has no
    `sliding_window` at all. Trusting the class body made a non-sliding model the canonical owner of
    the sliding variant, which is precisely the wrong-parent bug this tool exists to find.
    """
    model_dir = MODELS_ROOT / model
    found: set[str] = set()
    for path in sorted(model_dir.glob("configuration_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            # Modern configs declare `sliding_window: int | None = 4096` in the class body; older
            # ones take it as an `__init__` keyword.
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name, default = node.target.id, node.value
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name, default = node.targets[0].id, node.value
            elif isinstance(node, ast.FunctionDef) and node.name == "__init__":
                for arg, arg_default in zip(
                    node.args.args[len(node.args.args) - len(node.args.defaults) :], node.args.defaults
                ):
                    if arg.arg in CONFIG_ATTENTION_KEYS and not (
                        isinstance(arg_default, ast.Constant) and arg_default.value is None
                    ):
                        found.add(arg.arg)
                continue
            else:
                continue
            if name in CONFIG_ATTENTION_KEYS and not (isinstance(default, ast.Constant) and default.value is None):
                found.add(name)
    # `layer_types` defaults to None and is filled in by `__post_init__`; its mere presence means
    # the architecture mixes patterns, so treat a declaration as meaningful.
    for path in sorted(model_dir.glob("configuration_*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "layer_types" in text:
            found.add("layer_types")
    return frozenset(found)


# --------------------------------------------------------------------------------------------------
# Block classification
# --------------------------------------------------------------------------------------------------
# Order matters: the first pattern that matches wins. Mixer families are checked before
# `*Attention` because a linear-attention mixer is a different block, not an attention variant.
_KIND_PATTERNS = (
    ("mixer", r"(Mixer|Mamba|SSM|DeltaNet|LinearAttention|Recurrent|RWKV|Retention|Lightning)$"),
    ("moe", r"(SparseMoeBlock|MoeBlock|MoE|Moe|Experts|Expert)$"),
    ("attention", r"Attention$"),
    ("layer", r"(DecoderLayer|EncoderLayer)$"),
    ("mlp", r"(MLP|Mlp|FeedForward|FFN)$"),
    ("rotary", r"RotaryEmbedding$"),
    ("norm", r"(RMSNorm|LayerNorm)$"),
    ("layer_other", r"(Layer|Block)$"),
)


def classify(class_name: str) -> str | None:
    """Map a class name to a block kind, or `None` if it is not an attention-bearing block."""
    for kind, pattern in _KIND_PATTERNS:
        if re.search(pattern, class_name):
            return kind
    return None


# --------------------------------------------------------------------------------------------------
# Naming variance the extractor has to absorb, or the variant count inflates.
# --------------------------------------------------------------------------------------------------
BIAS_FLAGS = ("attention_bias", "use_bias", "qkv_bias", "use_qkv_bias", "enable_bias", "add_bias_linear", "mlp_bias")

_has = lambda src, *needles: any(n in src for n in needles)  # noqa: E731
_re = lambda src, pattern: bool(re.search(pattern, src))  # noqa: E731


def _bias_source(src: str) -> str:
    """Where a block's bias comes from. Tier 2: it never blocks inheriting `forward`."""
    if _has(src, *(f"config.{flag}" for flag in BIAS_FLAGS)) or _has(src, *BIAS_FLAGS):
        return "config"
    if _re(src, r"bias\s*=\s*True"):
        return "true"
    if _re(src, r"bias\s*=\s*False"):
        return "false"
    return "unknown"


# --------------------------------------------------------------------------------------------------
# Per-kind facet extraction
# --------------------------------------------------------------------------------------------------
def is_container(src: str) -> bool:
    """
    True for a class that owns no parameters and only wires sub-modules together.

    The BERT family splits attention into `XAttention` (holding `self.self` and `self.output`),
    `XSelfAttention` and `XSelfOutput`. Only the middle one is a real block; treating the wrapper
    as one invents a variant whose every facet is unknown.
    """
    return not _has(src, "nn.Linear", "nn.Parameter", "Conv1D", "nn.Conv1d", "nn.Embedding")


def _attention_facets(src: str, flags: frozenset[str] = frozenset()) -> tuple[dict, dict]:
    if _has(src, "sampling_offsets"):
        # Deformable attention samples a few learned offsets instead of scoring all keys. It is a
        # different mixing type, not a projection layout, so it never has a qkv triple.
        mix = "deformable"
    elif _has(src, "kv_a_proj_with_mqa", "kv_lora_rank", "q_a_proj", "kv_b_proj"):
        mix = "mla"
    elif _has(src, "num_key_value_heads", "num_kv_heads", "n_kv_heads"):
        mix = "gqa"
    else:
        mix = "mha"
    if _has(src, "encoder_hidden_states", "is_cross_attention", "key_value_states"):
        mix += "+cross"

    if mix.startswith("mla"):
        # MLA's projection set *is* the latent layout; its names vary per model (`q_a_proj`,
        # `q_lora_rank`, `kv_b_proj`) and carry no extra information once mix is known.
        qkv = "latent"
    elif mix == "deformable":
        qkv = "sampled"
    elif _has(src, "qkv_proj", "query_key_value", "c_attn", "in_proj_weight", "Wqkv") or _re(src, r"self\.qkv\s*="):
        qkv = "fused_qkv"
    elif _has(src, "kv_proj"):
        qkv = "fused_kv"
    # Vision and audio models spell the same three projections a dozen ways; anchor on `= ` so
    # `self.q` does not also match `self.qkv`.
    elif _has(src, "q_proj", "self.query") or _re(src, r"self\.(q|to_q|linear_q|query_proj|q_lin|q_content_proj)\s*="):
        qkv = "split"
    elif _has(src, "MultiheadAttention"):
        qkv = "fused_qkv"
    else:
        # A bespoke projection layout (mostly detection heads with positional/content splits).
        # Named rather than "unknown" so it groups honestly instead of merging unrelated blocks.
        qkv = "other"

    if _re(src, r"self\.(q_norm|query_norm)\s*=\s*\w*RMSNorm"):
        qk_norm = "rms"
    elif _re(src, r"self\.(q_norm|q_layernorm|query_layernorm|query_norm)\s*="):
        qk_norm = "layernorm"
    else:
        qk_norm = "none"

    # The window axis speaks `config.layer_types`' own vocabulary, and is decided by what the
    # config declares -- not by what the class body happens to mention.
    patterns = layer_pattern_vocabulary()
    window = "full_attention"
    if "attention_chunk_size" in flags and "chunked_attention" in patterns:
        window = "chunked_attention"
    elif ("sliding_window" in flags or "layer_types" in flags) and "sliding_attention" in patterns:
        window = "sliding_attention" if _has(src, "sliding_window") else "full_attention"

    if _has(src, "alibi"):
        rope = "alibi"
    elif _has(src, "rotate_every_two", "apply_rotary_emb_interleaved") or _re(src, r"\[\.\.\., 0::2\]"):
        rope = "interleaved"
    elif _has(src, "rotate_half", "apply_rotary_pos_emb", "position_embeddings", "rotary_emb"):
        rope = "half_split"
    else:
        rope = "none"

    extras = tuple(
        name
        for name, present in (
            ("sink", _has(src, "sink")),
            ("softcap", _has(src, "softcap", "logit_capping")),
            ("out_gate", _re(src, r"self\.(out_gate|attn_gate|g_proj|q_gate_proj)\s*=")),
        )
        if present
    )

    tier1 = {
        "mix": mix,
        "qkv": qkv,
        "qk_norm": qk_norm,
        "window": window,
        "rope": rope,
        "extras": "+".join(extras) or "plain",
    }
    tier2 = {
        "bias": _bias_source(src),
        "head_dim": "config" if _has(src, "config.head_dim") else "derived",
        "dropout": "yes" if _has(src, "attention_dropout", "attn_pdrop") else "no",
        # The forward threads a window through but the architecture never sets one: inheritable
        # either way, worth knowing when comparing against a model that does slide.
        "sliding_capable": "yes" if _has(src, "sliding_window") and window == "full_attention" else "no",
    }
    return tier1, tier2


def _mlp_facets(src: str) -> tuple[dict, dict]:
    if _has(src, "pointwise_conv", "depthwise_conv"):
        # Conformer-style convolutional feed-forward: not a linear MLP at all.
        gating = "conv"
    elif _has(src, "gate_up_proj"):
        gating = "fused_gate_up"
    elif _has(src, "gate_proj") or (_has(src, "self.w1") and _has(src, "self.w3")):
        gating = "gated"
    elif _has(
        src,
        "fc1",
        "c_fc",
        "dense_h_to_4h",
        "up_proj",
        "wi",
        "fc_in",
        "linear_in",
        "in_proj",
        "w_in",
        "w_1",
        "intermediate_dense",
        "proj_in",
        "linear1",
        "lin1",
        "layer1",
        "ffw_layer_1",
        "linear_start",
        "linear_1",
        "self.layers",
        "self.intermediate",
        "self.mlp",
    ):
        gating = "ungated"
    else:
        # A single projection plus a norm: a multimodal connector, not a transformer FFN.
        gating = "projector"
    tier2 = {"act": "config" if _has(src, "ACT2FN") else "literal", "bias": _bias_source(src)}
    return {"gating": gating}, tier2


def _moe_facets(src: str) -> tuple[dict, dict]:
    tier1 = {
        "router": "sigmoid" if _has(src, "sigmoid") else "softmax" if _has(src, "softmax") else "unknown",
        "router_bias": "yes" if _has(src, "e_score_correction_bias", "router_bias", "expert_bias") else "no",
        "topk_norm": "yes" if _has(src, "norm_topk_prob", "renormalize") else "no",
        "shared": "yes" if _has(src, "shared_expert") else "no",
        "weights": "grouped_3d" if _has(src, "nn.Parameter") else "module_list",
        "grouping": "grouped" if _has(src, "n_group", "topk_group", "expert_group") else "flat",
    }
    tier2 = {
        "aux_loss": "yes" if _has(src, "router_aux_loss", "load_balancing") else "no",
        "jitter": "yes" if _has(src, "jitter") else "no",
    }
    return tier1, tier2


def _norm_facets(src: str) -> tuple[dict, dict]:
    if _re(src, r"\(1\.?0?\s*\+\s*self\.weight") or _re(src, r"self\.weight\s*\+\s*1"):
        kind = "rms_one_plus_weight"
    elif _has(src, "rsqrt", "pow(2)", "variance"):
        kind = "rms"
    else:
        kind = "layernorm"
    return {"norm_kind": kind}, {"eps": "config" if _has(src, "config.") else "literal"}


def _rotary_facets(src: str) -> tuple[dict, dict]:
    if _has(src, "layer_types"):
        kind = "per_layer_type"
    elif _has(src, "long_factor", "short_factor"):
        kind = "longrope_buffers"
    else:
        kind = "standard"
    scalings = [s for s in rope_scaling_vocabulary() if s != "default" and s in src]
    return {"rope_kind": kind}, {"scalings": "+".join(sorted(scalings)) or "default"}


_FACET_EXTRACTORS = {
    "attention": _attention_facets,
    "mlp": _mlp_facets,
    "moe": _moe_facets,
    "norm": _norm_facets,
    "rotary": _rotary_facets,
}


# --------------------------------------------------------------------------------------------------
# Layer topology: the forward event string
# --------------------------------------------------------------------------------------------------
_EVENT_PATTERNS = (
    ("N", r"self\.\w*(norm|_ln|ln_)\w*\s*\("),
    ("X", r"self\.(cross_attn|encoder_attn|crossattention|cross_attention)\s*\("),
    ("A", r"self\.(self_attn|self_attention|attention|attn|mixer|token_mixer|temporal_block|linear_attn)\s*\("),
    ("M", r"self\.(mlp|feed_forward|ffn|block_sparse_moe|moe|feedforward|mlp_block|channel_mixer)\s*\("),
)
_SCALED_RESIDUAL_RE = re.compile(r"residual\s*\*|\*\s*residual|residual_multiplier|residual_scale")
_RESIDUAL_RE = re.compile(r"residual\s*\+|\+\s*residual")


def forward_topology(class_node: ast.ClassDef, file_source: str) -> str | None:
    """
    Summarise a layer's `forward` as an event string: `N` norm, `A` self-attention, `X`
    cross-attention, `M` mlp/moe, `R` residual add, `R*` scaled residual add.

    `N A R N M R` is classic pre-norm; `N A N R N M N R` is a Gemma2 sandwich; `N A R* N M R*`
    carries a residual multiplier. One token captures sandwich-vs-not and residual scaling, which
    is why this is the layer's whole tier-1 identity.
    """
    forward = next((n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name == "forward"), None)
    if forward is None:
        return None
    body = "\n".join(file_source.splitlines()[forward.lineno - 1 : forward.end_lineno])
    events: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.split("#")[0]
        # A norm and a call can share a line (`self.mlp(self.norm(x))`), so these are not exclusive;
        # attention vs cross-attention are, since the cross pattern is the more specific one.
        if re.search(_EVENT_PATTERNS[0][1], line):
            events.append("N")
        if re.search(_EVENT_PATTERNS[1][1], line):
            events.append("X")
        elif re.search(_EVENT_PATTERNS[2][1], line):
            events.append("A")
        if re.search(_EVENT_PATTERNS[3][1], line):
            events.append("M")
        if "residual" in line and _SCALED_RESIDUAL_RE.search(line):
            events.append("R*")
        elif _RESIDUAL_RE.search(line):
            events.append("R")
    return " ".join(events) or None


# --------------------------------------------------------------------------------------------------
# Helper canonicalisation (module-level functions such as `repeat_kv`)
# --------------------------------------------------------------------------------------------------
def canonical_source(node: ast.AST) -> str:
    """
    Unparse `node` with docstrings and the symbol's own name removed, for body-hash comparison.

    Equality here is *conservative*: an equal hash proves two bodies behave identically, but two
    bodies that behave identically can still hash differently (`x.reshape(...).unbind()` versus
    `x[..., 0::2]` compute the same interleaved rotation). The linter is therefore allowed to miss a
    duplicate but can never claim two different implementations are the same, which is the safe
    direction for something that tells people to delete code.
    """
    node = ast.parse(ast.unparse(node)).body[0]
    for sub in ast.walk(node):
        body = getattr(sub, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
        ):
            if isinstance(body[0].value.value, str):
                sub.body = body[1:] or [ast.Pass()]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        node.name = "_"
    return ast.unparse(node)


def canonical_method(class_node: ast.ClassDef, file_source: str, model: str, method: str = "forward") -> str | None:
    """
    The named method of a class, canonicalised for cross-model comparison.

    Model-specific identifiers are replaced with a placeholder so `Qwen3Attention.forward` and
    `LlamaAttention.forward` compare on structure rather than on naming.
    """
    node = next((n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name == method), None)
    if node is None:
        return None
    try:
        body = canonical_source(node)
    except (SyntaxError, ValueError):
        return None
    # `gpt_neox` -> `GptNeox`, `Gpt_Neox`, `GPTNeoX`-ish. Strip the squashed forms case-insensitively
    # rather than trying to reproduce each model's exact capitalisation.
    squashed = model.replace("_", "")
    return re.sub(rf"\b{re.escape(squashed)}", "X", body, flags=re.IGNORECASE)


# --------------------------------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------------------------------
@dataclass
class Block:
    """One block class found in one modeling file."""

    model: str
    path: Path
    class_name: str
    kind: str
    tier1: dict = field(default_factory=dict)
    tier2: dict = field(default_factory=dict)
    lineno: int = 0

    @property
    def variant(self) -> str:
        """The tag: the tier-1 facet values in axis order. Identical variant == inheritable forward."""
        axes = TIER1_AXES.get(self.kind, tuple(self.tier1))
        return "|".join(str(self.tier1.get(axis, "?")) for axis in axes)

    @property
    def tag(self) -> str:
        return f"{self.kind}:{self.variant}"

    def tier2_delta(self, other: "Block") -> dict[str, tuple[str, str]]:
        """Tier-2 facets that differ from `other` -- the init-rewrite hint on a suggestion."""
        return {k: (v, other.tier2.get(k, "?")) for k, v in self.tier2.items() if other.tier2.get(k, "?") != v}


@dataclass
class Helper:
    model: str
    path: Path
    name: str
    body: str

    @property
    def variant(self) -> str:
        # Content hash, not `hash()`: str hashing is salted per process, and this value is
        # written into a committed artifact.
        return hashlib.sha1(self.body.encode()).hexdigest()[:7]


def scan_file(path: Path, model: str) -> tuple[list[Block], list[Helper]]:
    """Extract every block and tracked helper from one `modeling_*.py`."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    blocks: list[Block] = []
    helpers: list[Helper] = []
    moe_nodes: list[tuple[ast.ClassDef, str]] = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in TRACKED_HELPERS:
            helpers.append(Helper(model, path, node.name, canonical_source(node)))
            continue
        if not isinstance(node, ast.ClassDef):
            continue
        kind = classify(node.name)
        if kind is None:
            continue
        class_source = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        if kind == "moe":
            # A model's MoE design is spread over three classes -- `*Experts` holds the weights,
            # `*SparseMoeBlock` the wiring, `*TopkRouter` the routing -- so no single class sees
            # the whole thing. Collect the candidates and emit one block per file below.
            moe_nodes.append((node, class_source))
            continue
        if kind in ("attention", "mlp") and is_container(class_source):
            continue
        if kind in ("layer", "layer_other"):
            topology = forward_topology(node, source)
            if topology is None:
                continue
            tier1, tier2 = {"topology": topology}, {}
        elif kind == "attention":
            tier1, tier2 = _attention_facets(class_source, config_flags(model))
        elif kind in _FACET_EXTRACTORS:
            tier1, tier2 = _FACET_EXTRACTORS[kind](class_source)
        else:
            continue
        blocks.append(Block(model, path, node.name, kind, tier1, tier2, node.lineno))

    if moe_nodes:
        # Facets come from the whole file so the router is visible; the block is named after the
        # wiring class rather than the expert-weight or router class.
        # ponytail: one MoE design per file. A file with two distinct designs would union them --
        # split per enclosing decoder layer if that ever shows up.
        primary = next(
            (n for n, _ in moe_nodes if re.search(r"(SparseMoeBlock|MoeBlock|MoE|Moe)$", n.name)), moe_nodes[0][0]
        )
        tier1, tier2 = _moe_facets(source)
        blocks.append(Block(model, path, primary.name, "moe", tier1, tier2, primary.lineno))
    return blocks, helpers


def scan_repo(models_root: Path = MODELS_ROOT) -> tuple[list[Block], list[Helper]]:
    """Scan every `modeling_*.py` under `models_root`."""
    blocks: list[Block] = []
    helpers: list[Helper] = []
    for model_dir in sorted(p for p in models_root.iterdir() if p.is_dir()):
        for path in sorted(model_dir.glob("modeling_*.py")):
            found_blocks, found_helpers = scan_file(path, model_dir.name)
            blocks.extend(found_blocks)
            helpers.extend(found_helpers)
    return blocks, helpers


# --------------------------------------------------------------------------------------------------
# The modular DAG, for "is this model already an ancestor of that one"
# --------------------------------------------------------------------------------------------------
@cache
def modular_parents(models_root: Path = MODELS_ROOT) -> dict[str, frozenset[str]]:
    """`{model: direct modular parents}`, from the `from ..parent.modeling_parent import X` lines."""
    if str(Path(__file__).parent) not in sys.path:
        sys.path.append(str(Path(__file__).parent))
    from create_dependency_mapping import extract_model_imports_from_file

    parents: dict[str, set[str]] = defaultdict(set)
    for model_dir in sorted(p for p in models_root.iterdir() if p.is_dir()):
        for path in sorted(model_dir.glob("modular_*.py")):
            for module in extract_model_imports_from_file(path):
                parent = module.split(".")[0]
                if parent != model_dir.name:
                    parents[model_dir.name].add(parent)
    return {model: frozenset(found) for model, found in parents.items()}


def ancestors(model: str, parents: dict[str, frozenset[str]] | None = None) -> set[str]:
    """Every model transitively reachable from `model` through modular inheritance."""
    parents = modular_parents() if parents is None else parents
    seen: set[str] = set()
    stack = list(parents.get(model, ()))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(parents.get(current, ()))
    return seen


# --------------------------------------------------------------------------------------------------
# Variant table
# --------------------------------------------------------------------------------------------------
@dataclass
class Variant:
    kind: str
    variant: str
    blocks: list[Block] = field(default_factory=list)

    @property
    def owners(self) -> list[str]:
        return sorted({b.model for b in self.blocks})

    @property
    def canonical(self) -> str | None:
        """The oldest owner: inheritance should follow history, so lineage stays stable."""
        dates = build_date_data()
        dated = [(dates[m], m) for m in self.owners if m in dates]
        return min(dated)[1] if dated else (self.owners[0] if self.owners else None)

    @property
    def tag(self) -> str:
        return f"{self.kind}:{self.variant}"


def build_variants(blocks: list[Block]) -> dict[str, Variant]:
    """Group blocks into tier-1 variants, keyed by tag."""
    variants: dict[str, Variant] = {}
    for block in blocks:
        variant = variants.setdefault(block.tag, Variant(block.kind, block.variant))
        variant.blocks.append(block)
    return variants


def _selfcheck() -> None:
    """Assert the facts the plan pinned down as regression tests."""
    assert layer_pattern_vocabulary(), "layer pattern vocabulary came back empty"
    assert "sliding_attention" in layer_pattern_vocabulary()
    assert "yarn" in rope_scaling_vocabulary() and "default" in rope_scaling_vocabulary()

    blocks, helpers = scan_repo()
    by_model_kind = defaultdict(list)
    for block in blocks:
        by_model_kind[(block.model, block.kind)].append(block)

    # `repeat_kv` is copied verbatim into ~100 files; it must collapse to a single variant.
    repeat_kv = {h.variant for h in helpers if h.name == "repeat_kv"}
    assert len(repeat_kv) == 1, f"repeat_kv should have 1 canonical body, got {len(repeat_kv)}"

    llama = next(b for b in by_model_kind[("llama", "attention")] if b.class_name == "LlamaAttention")
    assert llama.tier1["mix"] == "gqa", llama.tier1
    assert llama.tier1["qk_norm"] == "none", llama.tier1
    assert llama.tier1["window"] == "full_attention", llama.tier1
    assert llama.tier1["rope"] == "half_split", llama.tier1
    assert llama.tier2["bias"] == "config", llama.tier2

    qwen3 = next(b for b in by_model_kind[("qwen3", "attention")] if b.class_name == "Qwen3Attention")
    assert qwen3.tier1["qk_norm"] == "rms", qwen3.tier1

    # olmoe threads `getattr(config, "sliding_window", None)` but declares no window anywhere, so
    # it must not be classified as -- let alone become the canonical owner of -- a sliding variant.
    assert "sliding_window" not in config_flags("olmoe"), config_flags("olmoe")
    olmoe = next(b for b in by_model_kind[("olmoe", "attention")] if b.class_name == "OlmoeAttention")
    assert olmoe.tier1["window"] == "full_attention", olmoe.tier1
    assert olmoe.tier2["sliding_capable"] == "yes", olmoe.tier2
    mistral = next(b for b in by_model_kind[("mistral", "attention")] if b.class_name == "MistralAttention")
    assert mistral.tier1["window"] == "sliding_attention", mistral.tier1

    gemma2_layer = next(b for b in by_model_kind[("gemma2", "layer")])
    assert gemma2_layer.tier1["topology"].count("N") == 4, gemma2_layer.tier1

    # Every facet must resolve: an "unknown" either merges variants that differ or splits one
    # that does not, and both corrupt the tag.
    unknown = defaultdict(set)
    bespoke = 0
    for block in blocks:
        for axis, value in block.tier1.items():
            if value == "unknown":
                unknown[f"{block.kind}.{axis}"].add(f"{block.model}/{block.class_name}")
            elif value in ("other", "projector"):
                bespoke += 1
    assert not unknown, "unresolved facets: " + "; ".join(
        f"{axis}={len(models)} ({sorted(models)[:3]})" for axis, models in sorted(unknown.items())
    )
    # Bespoke layouts are legitimate but rare. If this ratio climbs, a real facet is being missed
    # and blocks that differ are being merged under one tag.
    assert bespoke / len(blocks) < 0.03, f"{bespoke}/{len(blocks)} blocks fell through to a bespoke value"

    # 5 textual bodies covering 3 behaviours: the half-split rotation (146 models), the interleaved
    # one written three different ways (8 + 6 + 1), and nanochat's flipped sign convention. Merging
    # the three interleaved spellings needs temp-variable inlining and list/tuple equivalence, which
    # is not worth it while equality stays conservative. This bound catches a regression that splits
    # them further.
    rotate_half = {h.variant for h in helpers if h.name == "rotate_half"}
    assert len(rotate_half) <= 5, f"rotate_half should collapse to at most 5 bodies, got {len(rotate_half)}"
    assert len({h.variant for h in helpers if h.name == "repeat_kv"}) == 1

    # Every axis must draw from a closed vocabulary. This is the real guard against the extractor
    # splitting a variant on cosmetics: a new spelling shows up as an unexpected value here rather
    # than quietly inflating the variant count.
    expected = {
        ("attention", "mix"): {"mha", "gqa", "mla", "deformable"},
        ("attention", "qkv"): {"split", "fused_qkv", "fused_kv", "latent", "sampled", "other"},
        ("attention", "qk_norm"): {"none", "rms", "layernorm"},
        ("attention", "window"): set(layer_pattern_vocabulary()),
        ("attention", "rope"): {"none", "half_split", "interleaved", "alibi"},
        ("mlp", "gating"): {"gated", "ungated", "fused_gate_up", "conv", "projector"},
        ("moe", "router"): {"softmax", "sigmoid", "unknown"},
        ("norm", "norm_kind"): {"rms", "rms_one_plus_weight", "layernorm"},
        ("rotary", "rope_kind"): {"standard", "per_layer_type", "longrope_buffers"},
    }
    for block in blocks:
        for axis, value in block.tier1.items():
            allowed = expected.get((block.kind, axis))
            if allowed is None:
                continue
            # `mix` carries an optional "+cross" suffix.
            base = value.split("+")[0] if axis == "mix" else value
            assert base in allowed, f"{block.model}/{block.class_name}: {axis}={value!r} is outside {sorted(allowed)}"

    print(f"selfcheck ok: {len(blocks)} blocks, {len(helpers)} helpers, {len(build_variants(blocks))} variants")


if __name__ == "__main__":
    _selfcheck()
