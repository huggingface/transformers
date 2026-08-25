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
import difflib
import hashlib
import re
import statistics
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

# Axis order == the "transformers format". Ordered by override cost, expensive first, so that
# picking a parent by longest common prefix forces agreement on the expensive axes and leaves any
# divergence in the cheap tail.
#
# Fitted, not guessed: `blocks_cli.py fit-order` measures each axis's cost as the median size of a
# real modular override that differs on that axis alone, then scores all 720 permutations. Measured
# costs (LoC): mix 72, extras 62, rope 58, qkv 52, qk_norm 50, window 50. This descending-cost order
# scores 26 591 against the exhaustive optimum's 26 584 -- 8 LoC apart, i.e. noise -- while the
# hand-picked order it replaced scored 30 571, 15% worse.
ATTENTION_AXES = ("mix", "extras", "rope", "qkv", "qk_norm", "window")
MLP_AXES = ("gating",)
# Left in semantic order on purpose: every MoE axis had fewer than 3 single-axis overrides to
# measure, so `fit-order` falls back to the kind median for all six and its "best" permutation is
# fitting noise. Revisit once more MoE models land.
MOE_AXES = ("router", "router_bias", "topk_norm", "shared", "weights", "grouping")
NORM_AXES = ("norm_kind",)
ROTARY_AXES = ("rope_kind",)
MIXER_AXES = ("mechanism",)

TIER1_AXES = {
    "attention": ATTENTION_AXES,
    "mlp": MLP_AXES,
    "moe": MOE_AXES,
    "norm": NORM_AXES,
    "rotary": ROTARY_AXES,
    "mixer": MIXER_AXES,
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
    # Mechanism names only, and anchored. A bare substring match over-captures model *names*:
    # `Recurrent` hits `RecurrentGemmaAttention` and `Mixer` hits `PatchTSMixerBlock`, which silently
    # removed mamba, mamba2, falcon_mamba, recurrent_gemma and patchtsmixer from the census entirely.
    # The `*Attention`-suffixed forms are listed explicitly because they are linear/recurrent mixers
    # wearing an attention name (`MiniMaxLightningAttention`).
    ("mixer", r"(Mixer|Mamba|SSM|DeltaNet|RWKV|Retention)$"),
    ("mixer", r"(Lightning|Linear|GatedDelta|Delta|SSM|Mamba)Attention$"),
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
        return "bias_from_config"
    if _re(src, r"bias\s*=\s*True"):
        return "bias_true"
    if _re(src, r"bias\s*=\s*False"):
        return "bias_false"
    return "bias_unknown"


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
        mix += "_cross"

    if mix.startswith("mla"):
        # MLA's projection set *is* the latent layout; its names vary per model (`q_a_proj`,
        # `q_lora_rank`, `kv_b_proj`) and carry no extra information once mix is known.
        qkv = "kv_latent"
    elif mix == "deformable":
        qkv = "qkv_sampled"
    elif _has(src, "qkv_proj", "query_key_value", "c_attn", "in_proj_weight", "Wqkv") or _re(src, r"self\.qkv\s*="):
        qkv = "qkv_fused"
    elif _has(src, "kv_proj"):
        qkv = "kv_fused"
    # Vision and audio models spell the same three projections a dozen ways; anchor on `= ` so
    # `self.q` does not also match `self.qkv`.
    elif _has(src, "q_proj", "self.query") or _re(src, r"self\.(q|to_q|linear_q|query_proj|q_lin|q_content_proj)\s*="):
        qkv = "qkv_split"
    elif _has(src, "MultiheadAttention"):
        qkv = "qkv_fused"
    else:
        # A bespoke projection layout (mostly detection heads with positional/content splits).
        # Named rather than "unknown" so it groups honestly instead of merging unrelated blocks.
        qkv = "qkv_custom"

    if _re(src, r"self\.(q_norm|query_norm)\s*=\s*\w*RMSNorm"):
        qk_norm = "qk_rmsnorm"
    elif _re(src, r"self\.(q_norm|q_layernorm|query_layernorm|query_norm)\s*="):
        qk_norm = "qk_layernorm"
    else:
        qk_norm = "no_qk_norm"

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
        rope = "rope_interleaved"
    elif _has(src, "rotate_half", "apply_rotary_pos_emb", "position_embeddings", "rotary_emb"):
        rope = "rope_half"
    else:
        rope = "no_pos_emb"

    extras = tuple(
        name
        for name, present in (
            ("attn_sink", _has(src, "sink")),
            ("logit_softcap", _has(src, "softcap", "logit_capping")),
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
        "extras": "+".join(extras) or "no_extras",
    }
    tier2 = {
        "bias": _bias_source(src),
        "head_dim": "head_dim_from_config" if _has(src, "config.head_dim") else "head_dim_derived",
        "dropout": "attn_dropout" if _has(src, "attention_dropout", "attn_pdrop") else "no_attn_dropout",
        # The forward threads a window through but the architecture never sets one: inheritable
        # either way, worth knowing when comparing against a model that does slide.
        "sliding_capable": "sliding_capable"
        if _has(src, "sliding_window") and window == "full_attention"
        else "not_sliding_capable",
    }
    return tier1, tier2


def _mlp_facets(src: str) -> tuple[dict, dict]:
    if _has(src, "pointwise_conv", "depthwise_conv"):
        # Conformer-style convolutional feed-forward: not a linear MLP at all.
        gating = "conv_ffn"
    elif _has(src, "gate_up_proj"):
        gating = "fused_gate_up_mlp"
    elif _has(src, "gate_proj") or (_has(src, "self.w1") and _has(src, "self.w3")):
        gating = "gated_mlp"
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
        gating = "ungated_mlp"
    else:
        # A single projection plus a norm: a multimodal connector, not a transformer FFN.
        gating = "linear_projector"
    tier2 = {"act": "act_from_config" if _has(src, "ACT2FN") else "act_literal", "bias": _bias_source(src)}
    return {"gating": gating}, tier2


def _moe_facets(src: str) -> tuple[dict, dict]:
    tier1 = {
        "router": "sigmoid_router"
        if _has(src, "sigmoid")
        else "softmax_router"
        if _has(src, "softmax")
        else "unknown",
        "router_bias": "router_bias"
        if _has(src, "e_score_correction_bias", "router_bias", "expert_bias")
        else "no_router_bias",
        "topk_norm": "norm_topk" if _has(src, "norm_topk_prob", "renormalize") else "no_norm_topk",
        "shared": "shared_expert" if _has(src, "shared_expert") else "no_shared_expert",
        "weights": "grouped_expert_weights" if _has(src, "nn.Parameter") else "expert_module_list",
        "grouping": "grouped_routing" if _has(src, "n_group", "topk_group", "expert_group") else "flat_routing",
    }
    tier2 = {
        "aux_loss": "aux_loss" if _has(src, "router_aux_loss", "load_balancing") else "no_aux_loss",
        "jitter": "jitter" if _has(src, "jitter") else "no_jitter",
    }
    return tier1, tier2


def _norm_facets(src: str) -> tuple[dict, dict]:
    # RMSNorm is distinguished from LayerNorm by what it does *not* do: no mean subtraction and no
    # bias. Matching on `variance`/`pow(2)` alone called `DebertaLayerNorm` an RMSNorm -- it computes
    # a variance but subtracts the mean and adds a bias -- which made deberta (2020) the canonical
    # owner of `rmsnorm` for 155 models.
    centred = _re(src, r"-\s*mean") or _re(src, r"mean\s*=\s*\w+\.mean\(")
    biased = _has(src, "self.bias")
    if _re(src, r"\(1\.?0?\s*\+\s*self\.weight") or _re(src, r"self\.weight\s*\+\s*1"):
        kind = "rmsnorm_one_plus_weight"
    elif _has(src, "rsqrt", "pow(2)", "variance") and not centred and not biased:
        kind = "rmsnorm"
    else:
        kind = "layernorm"
    return {"norm_kind": kind}, {"eps": "eps_from_config" if _has(src, "config.") else "eps_literal"}


def _mixer_facets(src: str) -> tuple[dict, dict]:
    """
    Linear / recurrent token mixers. Coarse on purpose: these are the frontier of the library and a
    detailed facet set would be guesswork. Recording the mechanism beats dropping the block.
    """
    if _has(src, "A_log", "dt_bias", "ssm_state"):
        mechanism = "ssm"
    elif _has(src, "beta", "g_norm", "gated_delta"):
        mechanism = "gated_delta"
    elif _has(src, "decay", "slope_rate"):
        mechanism = "decay_linear"
    else:
        mechanism = "custom_mixer"
    return {"mechanism": mechanism}, {"conv": "depthwise_conv" if _has(src, "conv1d") else "no_conv"}


def _rotary_facets(src: str) -> tuple[dict, dict]:
    if _has(src, "layer_types"):
        kind = "rope_per_layer_type"
    elif _has(src, "long_factor", "short_factor"):
        kind = "longrope_buffers"
    else:
        kind = "standard_rope"
    scalings = [s for s in rope_scaling_vocabulary() if s != "default" and s in src]
    return {"rope_kind": kind}, {"scalings": "+".join(sorted(scalings)) or "default_rope_scaling"}


_FACET_EXTRACTORS = {
    "attention": _attention_facets,
    "mlp": _mlp_facets,
    "moe": _moe_facets,
    "norm": _norm_facets,
    "rotary": _rotary_facets,
    "mixer": _mixer_facets,
}


# --------------------------------------------------------------------------------------------------
# Layer topology: the forward event string
# --------------------------------------------------------------------------------------------------
# Spelled out rather than initialled: `norm-attn-residual-norm-mlp-residual` says what it is,
# where `N A R N M R` needed a legend.
_EVENT_PATTERNS = (
    ("norm", r"self\.\w*(norm|_ln|ln_)\w*\s*\("),
    ("cross_attn", r"self\.(cross_attn|encoder_attn|crossattention|cross_attention)\s*\("),
    ("attn", r"self\.(self_attn|self_attention|attention|attn|mixer|token_mixer|temporal_block|linear_attn)\s*\("),
    ("mlp", r"self\.(mlp|feed_forward|ffn|block_sparse_moe|moe|feedforward|mlp_block|channel_mixer)\s*\("),
)
_SCALED_RESIDUAL_RE = re.compile(r"residual\s*\*|\*\s*residual|residual_multiplier|residual_scale")
_RESIDUAL_RE = re.compile(r"residual\s*\+|\+\s*residual")


def forward_topology(class_node: ast.ClassDef, file_source: str) -> str | None:
    """
    Summarise a layer's `forward` as the sequence of things it does, joined by `-`.

    `norm-attn-residual-norm-mlp-residual` is classic pre-norm;
    `norm-attn-norm-residual-norm-mlp-norm-residual` is a Gemma2 sandwich;
    `norm-attn-scaled_residual-norm-mlp-scaled_residual` carries a residual multiplier. One string
    captures sandwich-vs-not and residual scaling, which is why it is the layer's whole tier-1
    identity.
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
            events.append("norm")
        if re.search(_EVENT_PATTERNS[1][1], line):
            events.append("cross_attn")
        elif re.search(_EVENT_PATTERNS[2][1], line):
            events.append("attn")
        if re.search(_EVENT_PATTERNS[3][1], line):
            events.append("mlp")
        if "residual" in line and _SCALED_RESIDUAL_RE.search(line):
            events.append("scaled_residual")
        elif _RESIDUAL_RE.search(line):
            events.append("residual")
    return "-".join(events) or None


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
    # The canonicalised `forward`. Facets are a *lossy* summary: `DebertaV2DisentangledSelfAttention`
    # and `LlamaAttention` reduce to the same facet vector but compute entirely different things. So
    # facets only generate candidates, and this body is what confirms a match.
    forward: str | None = None

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
        blocks.append(
            Block(model, path, node.name, kind, tier1, tier2, node.lineno, canonical_method(node, source, model))
        )

    if moe_nodes:
        # Facets come from the whole file so the router is visible; the block is named after the
        # wiring class rather than the expert-weight or router class.
        # ponytail: one MoE design per file. A file with two distinct designs would union them --
        # split per enclosing decoder layer if that ever shows up.
        primary = next(
            (n for n, _ in moe_nodes if re.search(r"(SparseMoeBlock|MoeBlock|MoE|Moe)$", n.name)), moe_nodes[0][0]
        )
        tier1, tier2 = _moe_facets(source)
        blocks.append(
            Block(
                model,
                path,
                primary.name,
                "moe",
                tier1,
                tier2,
                primary.lineno,
                canonical_method(primary, source, model),
            )
        )
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
# The banner the modular converter stamps into every file it writes.
_GENERATED_MARKER = "This file was automatically generated from"


@cache
def generates_modeling(path: Path) -> bool:
    """
    Whether this specific `modeling_*.py` is produced by a modular file.

    Judged per *file*, not per directory. Having a `modular_*.py` in the directory is not enough:
    `modular_yolos.py` declares only image-processor classes, so `modeling_yolos.py` is hand-written,
    and adding a block subclass to that modular made the converter emit a modeling file containing
    only that block -- 565 of 655 lines gone. Nor is one generated file enough: `data2vec` ships
    audio, text and vision modeling files and they are not all generated. A finding on a
    hand-written file cannot be applied by editing a modular, and saying so up front is the
    difference between a one-line fix and a wasted afternoon.
    """
    try:
        return _GENERATED_MARKER in path.read_text(encoding="utf-8")[:2000]
    except OSError:
        return False


def tier2_mismatch(a: Block, b: Block) -> int:
    """How many init-only facets differ. Used to prefer the candidate needing the smallest `__init__`."""
    return sum(1 for k, v in a.tier2.items() if b.tier2.get(k) != v)


def parent_from_module(module: str | None) -> str | None:
    """
    The model a modular import refers to, e.g. `llama` for `..llama.modeling_llama`.

    Two spellings exist. Almost every modular file uses the level-2 form
    `from ..llama.modeling_llama import ...`, whose module is `llama.modeling_llama`. A couple use
    the level-3 form `from ...models.jamba.modeling_jamba import ...`, whose module is
    `models.jamba.modeling_jamba` -- and naively taking the first segment there yields `models`,
    a model that does not exist. That silently emptied those files' ancestry, so every block they
    already inherited correctly was reported as duplicated. Strip the `models.` prefix instead of
    normalising the source files, so both spellings resolve.
    """
    if not module:
        return None
    parts = module.split(".")
    # Strip whatever prefix the spelling carries: `models.jamba.modeling_jamba` (level 3) and
    # `transformers.models.aya_vision...` (absolute) both hide the real model further along.
    while parts and parts[0] in ("transformers", "models", "src"):
        parts = parts[1:]
    return parts[0] if parts else None


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
                parent = parent_from_module(module)
                if parent and parent != model_dir.name:
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
# Measured override cost -- the ground truth for the axis order
# --------------------------------------------------------------------------------------------------
@dataclass
class Override:
    """One `class Child(Parent):` in a modular file, and how many lines it actually spends."""

    child_model: str
    parent_model: str
    kind: str
    child_class: str
    parent_class: str
    loc: int


def modular_overrides(models_root: Path = MODELS_ROOT) -> list[Override]:
    """Every cross-model block subclass declared in a `modular_*.py`, with its size in lines."""
    found: list[Override] = []
    for model_dir in sorted(p for p in models_root.iterdir() if p.is_dir()):
        for path in sorted(model_dir.glob("modular_*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            # `from ..llama.modeling_llama import LlamaAttention` -> {"LlamaAttention": "llama"}.
            # Levels 2 and 3 both occur; see `parent_from_module`.
            owner: dict[str, str] = {}
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level not in (2, 3) or not node.module:
                    continue
                parent = parent_from_module(node.module)
                if parent is None or parent == model_dir.name:
                    continue
                for alias in node.names:
                    owner[alias.asname or alias.name] = parent
            for node in tree.body:
                if not isinstance(node, ast.ClassDef) or not node.bases:
                    continue
                base = node.bases[0]
                base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
                if base_name not in owner:
                    continue
                kind = classify(node.name)
                if kind is None:
                    continue
                found.append(
                    Override(
                        model_dir.name,
                        owner[base_name],
                        kind,
                        node.name,
                        base_name,
                        node.end_lineno - node.lineno + 1,
                    )
                )
    return found


# `# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->BlenderbotSmall`
_COPIED_FROM_RE = re.compile(r"#\s*Copied from transformers\.models\.(\w+)\.\w+\.(\w+)")


@cache
def copied_from_sources(models_root: Path = MODELS_ROOT) -> dict[tuple[str, str], str]:
    """
    `{(model, symbol): source model}` for every `# Copied from` marker in the library.

    This is the library's *third* reuse mechanism, alongside modular inheritance and plain
    duplication, and `utils/check_copies.py` keeps it consistent in CI. A block carrying a marker is
    therefore already managed reuse: `BlenderbotSmallAttention` is copied from `BartAttention` on
    purpose, and reporting it as unmanaged duplication would send someone to author a modular file
    for a model that already tracks its source.
    """
    sources: dict[tuple[str, str], str] = {}
    for model_dir in sorted(p for p in models_root.iterdir() if p.is_dir()):
        for path in sorted(model_dir.glob("modeling_*.py")):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            pending: str | None = None
            for line in lines:
                marker = _COPIED_FROM_RE.search(line)
                if marker:
                    pending = marker.group(1)
                    continue
                declared = re.match(r"(?:class|def)\s+(\w+)", line)
                if declared:
                    if pending:
                        sources[(model_dir.name, declared.group(1))] = pending
                    pending = None
    return sources


@cache
def modular_class_edges(models_root: Path = MODELS_ROOT) -> dict[tuple[str, str], int]:
    """
    `{(child model, base model): how many classes the child subclasses from it}`.

    Counts *every* class, not only blocks. Whether two models have an established relationship is
    decided by everything they share -- `glm4` subclasses `GlmForCausalLM` and
    `GlmForSequenceClassification` as well as `GlmAttention`, so taking its attention from glm is
    descent. Counting blocks alone made that look like a one-off reach.
    """
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for model_dir in sorted(p for p in models_root.iterdir() if p.is_dir()):
        for path in sorted(model_dir.glob("modular_*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            owner: dict[str, str] = {}
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level not in (2, 3) or not node.module:
                    continue
                parent = parent_from_module(node.module)
                if parent is None or parent == model_dir.name:
                    continue
                for alias in node.names:
                    owner[alias.asname or alias.name] = parent
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                for base in node.bases:
                    name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
                    if name in owner:
                        counts[(model_dir.name, owner[name])] += 1
    return dict(counts)


MIN_SAMPLES_PER_AXIS = 3


def forward_similarity(a: Block, b: Block) -> float:
    """How alike two blocks' canonicalised `forward` bodies are, in [0, 1]. 0 if either is missing."""
    if not a.forward or not b.forward:
        return 0.0
    return difflib.SequenceMatcher(None, a.forward, b.forward).ratio()


def forwards_match(a: Block, b: Block) -> bool:
    """
    Whether `a` could inherit `b`'s `forward` without overriding it.

    Requires the canonicalised bodies to be **identical**, and that threshold is measured rather
    than chosen: across 489 real modular overrides where the child shares the parent's variant and
    spends <=5 lines (i.e. reuse that actually happened and was reviewed), forward similarity is
    1.000 at the 5th percentile -- every single one. Overrides that had to be written instead sit at
    a median of 0.832 and reach 0.988, so anything short of equality is a block someone would have
    had to rewrite. Facets only nominate candidates; this is what decides.
    """
    return bool(a.forward) and a.forward == b.forward


def measure_axis_costs(blocks: list[Block]) -> tuple[dict[tuple[str, str], float], dict[str, list[int]]]:
    """
    Measure what differing on each axis actually costs, in lines of override.

    Ground truth is every `class Child(Parent)` in a modular file: we know the axes on which the
    child's variant differs from the parent's, and we can count the lines the child spends. Cases
    differing on exactly one axis measure that axis directly. Axes with too few samples fall back to
    their block kind's median so a lucky single observation cannot dominate the ordering.

    Returns the per-axis costs and, per kind, the sizes of the overrides whose variant *matches*
    the parent -- the baseline this whole design rests on (it should be ~2 lines).
    """
    exact = {(b.model, b.class_name): b for b in blocks}
    per_axis: dict[tuple[str, str], list[int]] = defaultdict(list)
    per_kind: dict[str, list[int]] = defaultdict(list)
    baseline: dict[str, list[int]] = defaultdict(list)

    for override in modular_overrides():
        child = exact.get((override.child_model, override.child_class))
        parent = exact.get((override.parent_model, override.parent_class))
        if child is None or parent is None or child.kind != parent.kind:
            continue
        axes = TIER1_AXES.get(child.kind, ())
        delta = [
            axis
            for axis, mine, theirs in zip(axes, child.variant.split("|"), parent.variant.split("|"))
            if mine != theirs
        ]
        if not delta:
            baseline[child.kind].append(override.loc)
        elif len(delta) == 1:
            per_axis[(child.kind, delta[0])].append(override.loc)
            per_kind[child.kind].append(override.loc)

    costs: dict[tuple[str, str], float] = {}
    for kind, axes in TIER1_AXES.items():
        kind_median = statistics.median(per_kind[kind]) if per_kind.get(kind) else 0.0
        for axis in axes:
            samples = per_axis.get((kind, axis), [])
            costs[(kind, axis)] = statistics.median(samples) if len(samples) >= MIN_SAMPLES_PER_AXIS else kind_median
    return costs, baseline


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
    assert llama.tier1["qk_norm"] == "no_qk_norm", llama.tier1
    assert llama.tier1["window"] == "full_attention", llama.tier1
    assert llama.tier1["rope"] == "rope_half", llama.tier1
    assert llama.tier2["bias"] == "bias_from_config", llama.tier2

    qwen3 = next(b for b in by_model_kind[("qwen3", "attention")] if b.class_name == "Qwen3Attention")
    assert qwen3.tier1["qk_norm"] == "qk_rmsnorm", qwen3.tier1

    # olmoe threads `getattr(config, "sliding_window", None)` but declares no window anywhere, so
    # it must not be classified as -- let alone become the canonical owner of -- a sliding variant.
    assert "sliding_window" not in config_flags("olmoe"), config_flags("olmoe")
    olmoe = next(b for b in by_model_kind[("olmoe", "attention")] if b.class_name == "OlmoeAttention")
    assert olmoe.tier1["window"] == "full_attention", olmoe.tier1
    assert olmoe.tier2["sliding_capable"] == "sliding_capable", olmoe.tier2
    # DebertaLayerNorm computes a variance but centres and biases: it is a LayerNorm.
    deberta_norm = next(b for b in by_model_kind[("deberta", "norm")] if b.class_name == "DebertaLayerNorm")
    assert deberta_norm.tier1["norm_kind"] == "layernorm", deberta_norm.tier1
    llama_norm = next(b for b in by_model_kind[("llama", "norm")] if b.class_name == "LlamaRMSNorm")
    assert llama_norm.tier1["norm_kind"] == "rmsnorm", llama_norm.tier1

    mistral = next(b for b in by_model_kind[("mistral", "attention")] if b.class_name == "MistralAttention")
    assert mistral.tier1["window"] == "sliding_attention", mistral.tier1

    # No model may resolve to a parent that is not a real model directory. `nemotron_h` uses the
    # level-3 `...models.jamba.modeling_jamba` import form and used to resolve to a phantom
    # "models" parent, which emptied its ancestry and reported every correctly inherited block as a
    # duplicate.
    real = {p.name for p in MODELS_ROOT.iterdir() if p.is_dir()}
    for model, found in modular_parents().items():
        bogus = found - real
        assert not bogus, f"{model} resolves to non-existent parents {sorted(bogus)}"
    assert "jamba" in modular_parents()["nemotron_h"], modular_parents()["nemotron_h"]

    gemma2_layer = next(b for b in by_model_kind[("gemma2", "layer")])
    assert gemma2_layer.tier1["topology"].split("-").count("norm") == 4, gemma2_layer.tier1

    # Every facet must resolve: an "unknown" either merges variants that differ or splits one
    # that does not, and both corrupt the tag.
    unknown = defaultdict(set)
    bespoke = 0
    for block in blocks:
        for axis, value in block.tier1.items():
            if value == "unknown":
                unknown[f"{block.kind}.{axis}"].add(f"{block.model}/{block.class_name}")
            elif value in ("qkv_custom", "linear_projector"):
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
        ("attention", "qkv"): {"qkv_split", "qkv_fused", "kv_fused", "kv_latent", "qkv_sampled", "qkv_custom"},
        ("attention", "qk_norm"): {"no_qk_norm", "qk_rmsnorm", "qk_layernorm"},
        ("attention", "window"): set(layer_pattern_vocabulary()),
        ("attention", "rope"): {"no_pos_emb", "rope_half", "rope_interleaved", "alibi"},
        ("mlp", "gating"): {"gated_mlp", "ungated_mlp", "fused_gate_up_mlp", "conv_ffn", "linear_projector"},
        ("moe", "router"): {"softmax_router", "sigmoid_router", "unknown"},
        ("moe", "router_bias"): {"router_bias", "no_router_bias"},
        ("moe", "topk_norm"): {"norm_topk", "no_norm_topk"},
        ("moe", "shared"): {"shared_expert", "no_shared_expert"},
        ("moe", "weights"): {"grouped_expert_weights", "expert_module_list"},
        ("moe", "grouping"): {"grouped_routing", "flat_routing"},
        ("mixer", "mechanism"): {"ssm", "gated_delta", "decay_linear", "custom_mixer"},
        ("norm", "norm_kind"): {"rmsnorm", "rmsnorm_one_plus_weight", "layernorm"},
        ("rotary", "rope_kind"): {"standard_rope", "rope_per_layer_type", "longrope_buffers"},
    }
    for block in blocks:
        for axis, value in block.tier1.items():
            allowed = expected.get((block.kind, axis))
            if allowed is None:
                continue
            # `mix` carries an optional "+cross" suffix.
            base = value.split("_cross")[0] if axis == "mix" else value
            assert base in allowed, f"{block.model}/{block.class_name}: {axis}={value!r} is outside {sorted(allowed)}"

    print(f"selfcheck ok: {len(blocks)} blocks, {len(helpers)} helpers, {len(build_variants(blocks))} variants")


if __name__ == "__main__":
    _selfcheck()
