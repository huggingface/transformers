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
import textwrap
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

# Axis order == the "transformers format". Ordered so that picking a parent by longest common
# prefix forces agreement on the expensive axes and leaves any divergence in the cheap tail.
#
# Fitted, not guessed: `blocks_cli.py fit-order` measures each axis's cost as the median size of a
# real modular override that differs on that axis alone, then scores all 5040 permutations by the
# total override LoC the library would pay. Re-fitted after `mix` collapsed MHA/MQA into GQA and
# `cross` split off it. Measured costs (LoC): cross 92, rope 58, extras 58, mix 55, qkv 55,
# layer_typing 55, qk_norm 50.
#
# Descending cost is *not* the optimum any more: it scores 28 128 against the optimum's 26 293, and
# this order is one of the 14 permutations that reach it -- specifically the one closest to
# descending cost (2 inversions). The cost column is only weakly determined now: `mix` has 0
# single-axis samples, `layer_typing` 0 and `qkv` 1, so all three fall back to the attention kind
# median of 55 and are effectively tied. The exhaustive score, which sees every attention variant,
# is what decides the order; the cost column only breaks its ties.
#
# `cross` leads because it is the one axis with a decisively higher *measured* cost, from three real
# overrides of 48, 92 and 102 LoC: turning self-attention into cross-attention rewrites `forward`.
# Splitting it out of `mix` is what exposed that. The old `mix` cost of 80 LoC was the median of two
# unrelated populations -- of its 5 single-axis overrides, 2 (exaone4_5, jina_embeddings_v3) were
# pure MHA-vs-GQA relabelling and now match their parent exactly, at no cost at all, while the other
# 3 (lightglue, moonshine, t5gemma2) were self-vs-cross rewrites. One axis was pricing both, so it
# mis-weighted both. Collapsing the relabelling took the attention overrides that inherit `forward`
# unchanged from 178 to 180.
#
# `layer_typing` stays tier 1, but only just, and only because it is read off `forward`: three blocks
# genuinely index by it (gemma4 and gemma4_unified take `shared_kv_states[self.layer_type]`,
# deepseek_v4 takes `position_embeddings[self.rope_layer_type]`). The 50 blocks that merely set it in
# `__init__` are tier 2 as `layer_typing_init` -- which is what lets `qwen3` and `qwen3_moe` share a
# variant, as their byte-identical forwards require, while `del self.layer_type` stays visible.
#
# `mix` is the mixing *mechanism* only: grouped softmax attention, MLA's shared low-rank latent, or
# deformable sampling. It deliberately does not encode the kv head count. GQA is the general
# formulation and MHA and MQA are the degenerate cases you get by setting
# `num_key_value_heads` to `num_attention_heads` or to 1, which sizes an `nn.Linear` in `__init__`
# and changes nothing in `forward`. That is a config fact, and it is reported as tier-2
# `kv_sharing`, read out of `configuration_*.py` by `config_kv_sharing`.
ATTENTION_AXES = ("cross", "mix", "rope", "extras", "qkv", "layer_typing", "qk_norm")
MLP_AXES = ("gating",)
# Left in semantic order on purpose: every MoE axis had fewer than 3 single-axis overrides to
# measure, so `fit-order` falls back to the kind median for all six and its "best" permutation is
# fitting noise. Revisit once more MoE models land.
MOE_AXES = ("router", "router_bias", "topk_norm", "shared", "weights", "grouping")
NORM_AXES = ("norm_kind",)
ROTARY_AXES = ("rope_kind",)
MIXER_AXES = ("mechanism", "gating", "conv")
ROUTER_AXES = ("scoring", "selection", "router_bias", "topk_norm", "scaling")
INDEXER_AXES = ("query_source", "scoring", "key_norm", "output")

TIER1_AXES = {
    "attention": ATTENTION_AXES,
    "mlp": MLP_AXES,
    "moe": MOE_AXES,
    "norm": NORM_AXES,
    "rotary": ROTARY_AXES,
    "mixer": MIXER_AXES,
    "router": ROUTER_AXES,
    "indexer": INDEXER_AXES,
    "layer": ("topology",),
    "conv_block": ("conv",),
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


# The two head counts that decide *which* degenerate case of grouped attention a model instantiates.
# The kv spellings are exactly the ones the class body used to be sniffed for, so config and class
# can never disagree about what "kv heads" means.
CONFIG_QUERY_HEAD_KEYS = (
    "num_attention_heads",
    "num_heads",
    "n_head",
    "n_heads",
    "encoder_attention_heads",
    # moonshine sizes its two towers separately and aliases `num_key_value_heads` to the decoder's
    # via `attribute_map`, so the unprefixed name never appears as a default.
    "decoder_num_attention_heads",
    "encoder_num_attention_heads",
)
CONFIG_KV_HEAD_KEYS = (
    "num_key_value_heads",
    "num_kv_heads",
    "n_kv_heads",
    "decoder_num_key_value_heads",
    "encoder_num_key_value_heads",
)


def _class_defaults(node: ast.ClassDef) -> dict[str, ast.AST]:
    """`{name: default-expression}` for one config class, from its body and its `__init__` keywords."""
    out: dict[str, ast.AST] = {}
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            out[stmt.target.id] = stmt.value
        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            out[stmt.targets[0].id] = stmt.value
        elif isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
            args = stmt.args
            pairs = list(zip(args.args[len(args.args) - len(args.defaults) :], args.defaults))
            pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
            for arg, default in pairs:
                # A class-body annotation wins: it is the modern spelling of the same default.
                out.setdefault(arg.arg, default)
    return out


def _int_default(node: ast.AST | None) -> int | None:
    """The literal `int` a default expression holds, or `None` for `None`/computed/absent."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    return None


def _kv_sharing_of_config(defaults: dict[str, ast.AST]) -> str | None:
    """Which case of grouped attention one config class instantiates, or `None` if it says nothing."""
    query_heads = next(
        (n for n in (_int_default(defaults.get(k)) for k in CONFIG_QUERY_HEAD_KEYS) if n is not None), None
    )
    kv_key = next((k for k in CONFIG_KV_HEAD_KEYS if k in defaults), None)
    if kv_key is None:
        # gpt_bigcode -- the library's original MQA model -- spells multi-query as a boolean rather
        # than as `num_key_value_heads=1`. Same degenerate case, different notation. Only consulted
        # when there is no head count at all: falcon carries both, and its head count is the more
        # precise statement of the two.
        if isinstance(defaults.get("multi_query"), ast.Constant) and defaults["multi_query"].value is True:
            return "mqa_degenerate"
        # No kv-head knob at all: every query head carries its own key and value, by construction.
        return "no_kv_heads" if query_heads is not None else None
    kv_heads = _int_default(defaults[kv_key])
    if kv_heads is None:
        # `num_key_value_heads=None`, filled from `num_attention_heads` in `__init__`. Still the
        # general formulation -- it just defaults to the degenerate one.
        return "mha_degenerate_by_default"
    if kv_heads == 1:
        return "mqa_degenerate"
    if query_heads is not None and kv_heads == query_heads:
        return "mha_degenerate"
    return "grouped"


@cache
def config_kv_sharing(model: str) -> str:
    """
    Which case of grouped attention the model's configs ask for -- read from the config, not the class.

    GQA is the general formulation: `nn.Linear(hidden, num_key_value_heads * head_dim)` is
    multi-head attention when `num_key_value_heads == num_attention_heads` and multi-query when it
    is `1`. Which one you get is a *config* fact, so it belongs here and in tier 2, not in the
    tier-1 vector -- the class body and the `forward` are identical either way.

    A model with several attention towers (a text config and a vision config) legitimately asks for
    several, so the labels are joined the way `extras` joins its flags rather than collapsed to one.
    """
    labels: set[str] = set()
    for path in sorted((MODELS_ROOT / model).glob("configuration_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.endswith("Config"):
                label = _kv_sharing_of_config(_class_defaults(node))
                if label is not None:
                    labels.add(label)
    return "+".join(sorted(labels)) or "unknown_kv_sharing"


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
    # Sparse-attention indexers (DeepSeek DSA and friends) own their own projections and top-k
    # selection; they matched nothing at all before, so every model with one was invisible.
    ("indexer", r"Indexer\w*$"),
    # Routing is a separate decision from the expert stack it feeds -- `*TopkRouter` is where topk,
    # group-limiting and the score-correction bias actually live.
    ("router", r"Router$"),
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


# A projection's *role* is a structural fact; the attribute holding it is a spelling. The library
# spells the query projection at least eleven ways -- `q_proj`, `query`, `q`, `to_q`, `to_query`,
# `linear_q`, `q_lin`, `query_proj`, `Wq`, `project_q`, `self_attn_query_content_proj` -- and an
# extractor that keys off one preferred spelling turns that into architecture: `cpmant`
# (`project_q`), `ctrl` (`Wq`) and both `dab_detr` decoder attentions fell through to `qkv_custom`,
# and dab_detr's two singleton variants existed only because the same model spells the same
# projection two ways in two classes.
#
# So roles are read off the *words* an attribute name contains, after the decoration words that
# carry no role (`proj`, `lin`, `to`, `w`, `self`, `attn`, ...) have been split away. Word equality,
# not substring: `qkv` must not read as `q`, and `kv_a_proj_with_mqa` must not read as `k` + `v`.
_ROLE_WORDS = {
    "q": ("q", "query"),
    "k": ("k", "key"),
    "v": ("v", "value", "values"),
    # `dense` and `final` are the BERT and swiftformer spellings of the output projection.
    "o": ("o", "out", "output", "dense", "final"),
}
_PROJ_ASSIGN_RE = re.compile(r"self\.([A-Za-z_0-9]+)\s*=\s*(?:nn\.Linear|nn\.Conv1d|Conv1D|nn\.Parameter)\b")
# Split on separators *and* on camel-case boundaries, so `Wq` yields `w` + `q`.
_WORD_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])")


_ROLE_QKV = frozenset({"q", "k", "v"})


def _projection_roles(src: str) -> set[str]:
    """Which of the q/k/v/o projection roles a class assigns, regardless of what it calls them."""
    roles: set[str] = set()
    for name in _PROJ_ASSIGN_RE.findall(src):
        words = {w.lower() for w in _WORD_SPLIT_RE.split(name) if w}
        # `Wq`/`Wk`/`Wv` (ctrl) hide the role behind a single-letter weight prefix that no
        # camel-case boundary splits: there is no case change between `W` and `q`.
        words |= {w[1:] for w in words if len(w) == 2 and w[0] == "w"}
        for role, spellings in _ROLE_WORDS.items():
            if not words.isdisjoint(spellings):
                roles.add(role)
    return roles


# One projection three times as wide as the model, unpacked into three tensors, is a fused qkv
# whatever the attribute is called -- `att_proj` in bark, a bare `proj` in esmfold. Both halves are
# required: the width alone would catch an unrelated 3x projection, and a three-way unpack alone
# would catch a class that unpacks something else.
_FUSED_TRIPLE_WIDTH_RE = re.compile(r"nn\.Linear\(\s*[^,()]*,\s*(?:3\s*\*|[\w.]+\s*\*\s*3\b)")
_TRIPLE_UNPACK_RE = re.compile(r"\w+\s*,\s*\w+\s*,\s*\w+\s*=\s*self\.\w+\(|\.chunk\(\s*3\b")


def _is_fused_triple(src: str) -> bool:
    """True when the class projects q, k and v with one 3x-wide layer and splits the result."""
    return bool(_FUSED_TRIPLE_WIDTH_RE.search(src) and _TRIPLE_UNPACK_RE.search(src))


def _mlp_projections(src: str) -> set[str]:
    """The projection attributes an MLP owns, for counting them instead of naming them."""
    return set(_PROJ_ASSIGN_RE.findall(src))


# --------------------------------------------------------------------------------------------------
# Per-kind facet extraction
# --------------------------------------------------------------------------------------------------
def is_conv_block(src: str) -> bool:
    """
    True for a `*Layer`/`*Block` class that convolves and never attends.

    Name-based classification alone put 73 of these in with the transformer layers -- `BeitConvLayer`,
    `Data2VecAudioConvLayer`, `EfficientNetDepthwiseLayer`. They have no residual-and-norm topology
    to report, so they made the layer census look far more degenerate than it was. They are a
    different kind of block, not a badly-written transformer layer.
    """
    convolves = _has(src, "nn.Conv1d", "nn.Conv2d", "nn.Conv3d", "nn.ConvTranspose")
    attends = _has(src, "self_attn", "self.attention", "self.attn", "self_attention", "Attention(")
    return convolves and not attends


def is_container(src: str) -> bool:
    """
    True for a class that owns no parameters and only wires sub-modules together.

    The BERT family splits attention into `XAttention` (holding `self.self` and `self.output`),
    `XSelfAttention` and `XSelfOutput`. Only the middle one is a real block; treating the wrapper
    as one invents a variant whose every facet is unknown.
    """
    return not _has(src, "nn.Linear", "nn.Parameter", "Conv1D", "nn.Conv1d", "nn.Embedding")


def _forward_source(src: str) -> str:
    """Just the `forward` method of a class source, for facets that must key off `forward` only."""
    try:
        tree = ast.parse(textwrap.dedent(src))
    except (SyntaxError, ValueError):
        return src
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            try:
                return ast.unparse(node)
            except (AttributeError, ValueError):
                return src
    return ""


def _attention_facets(
    src: str, flags: frozenset[str] = frozenset(), kv_sharing: str = "unknown_kv_sharing"
) -> tuple[dict, dict]:
    if _has(src, "sampling_offsets"):
        # Deformable attention samples a few learned offsets instead of scoring all keys. It is a
        # different mixing type, not a projection layout, so it never has a qkv triple.
        #
        # Splitting `cross` out of `mix` also un-broke the `qkv_sampled` branch below. Every
        # deformable block in the library attends to encoder features, so `mix` was always suffixed
        # to `deformable_cross` before that branch tested `mix == "deformable"` -- it never matched,
        # and all 14 of them fell through to `qkv_custom`.
        mix = "deformable"
    elif _has(src, "kv_a_proj_with_mqa", "kv_lora_rank", "q_a_proj", "kv_b_proj"):
        # MLA projects keys and values through a shared low-rank latent. That is a genuinely
        # different mechanism -- not grouped attention with a different head count -- so it stays.
        mix = "mla"
    else:
        # GQA is the *general* formulation, and MHA and MQA are its degenerate cases:
        # `nn.Linear(hidden, num_key_value_heads * head_dim)` is multi-head attention when
        # `num_key_value_heads == num_attention_heads` and multi-query when it is 1. Same code, same
        # `forward`, one mechanism parameterised by config -- so which one you get is a *config*
        # fact, reported as `kv_sharing` in tier 2, not a tier-1 value.
        #
        # The old split asked whether the class body happened to mention a kv-head count, which is
        # not a forward difference at all. It kept `MoonshineStreamingEncoderAttention` in a
        # different variant from `ViTAttention`, `BeitAttention`, `DeiTAttention`, `ASTAttention`
        # and five more whose canonicalised forwards are byte-identical to it -- every one of the
        # 10 over-split pairs in the attention census was this one distinction. `repeat_kv`, the
        # only thing a kv-head count actually changes, appears in exactly one forward in the
        # library.
        mix = "gqa"
    # Orthogonal to the mixing mechanism: whether keys and values come from somewhere else. The
    # suffix form (`mha_cross`) put two independent facts on one axis, so `fit-order` could only
    # weight them together and the closed-vocabulary check had to special-case the suffix.
    cross = (
        "cross_attention"
        if _has(src, "encoder_hidden_states", "is_cross_attention", "key_value_states")
        else "self_attention"
    )

    if mix == "mla":
        # MLA's projection set *is* the latent layout; its names vary per model (`q_a_proj`,
        # `q_lora_rank`, `kv_b_proj`) and carry no extra information once mix is known.
        qkv = "kv_latent"
    elif mix == "deformable":
        qkv = "qkv_sampled"
    elif (
        _has(src, "qkv_proj", "query_key_value", "c_attn", "in_proj_weight", "Wqkv")
        or _re(src, r"self\.qkv\s*=")
        # Structural, not by name: bark's `att_proj` and esmfold's `proj` are fused qkv triples.
        or _is_fused_triple(src)
    ):
        qkv = "qkv_fused"
    elif _has(src, "kv_proj"):
        qkv = "kv_fused"
    # Three separate projections, whatever they are called. The role test is what makes this
    # name-agnostic; the spelling list stays as a union so a class that only *mentions* `q_proj`
    # (esmfold's invariant-point attention, evolla's sequence compressor, granite_speech's
    # conformer) cannot regress out of `qkv_split`.
    elif (
        _ROLE_QKV.issubset(_projection_roles(src))
        or _has(src, "q_proj", "self.query")
        or _re(src, r"self\.(q|to_q|linear_q|query_proj|q_lin|q_content_proj)\s*=")
    ):
        qkv = "qkv_split"
    # `nn.` qualified on purpose: the bare substring matched `FunnelRelMultiheadAttention`'s own
    # class name and called a block with three real `q_head`/`k_head`/`v_head` projections fused.
    elif _has(src, "nn.MultiheadAttention"):
        qkv = "qkv_fused"
    else:
        # A bespoke projection layout (mostly detection heads with positional/content splits).
        # Named rather than "unknown" so it groups honestly instead of merging unrelated blocks.
        qkv = "qkv_custom"

    # Binary on purpose. `forward` calls `self.q_norm(...)` whatever class the attribute holds, so
    # the class cannot gate it: cohere (`CohereLayerNorm`) and glm4_moe (`Glm4MoeRMSNorm`) have
    # byte-identical forwards. Keying on the class name also mislabelled 5 of 13 -- hunyuan and lfm2
    # hold an RMSNorm under a `*_layernorm` attribute -- and called `nn.Identity()` a norm. The class
    # is a norm-block fact, already censused under `NORM_AXES`, so it lands in tier 2.
    # `q_layer_norm` (idefics) and `layernorm_q` are the same projection norm under two more
    # spellings. `q_a_layernorm` is deliberately *not* in this set: it normalises MLA's compressed
    # query latent, not the per-head query, and is already accounted for by `mix == "mla"`. Folding
    # it in here would relabel all 11 kv_latent blocks and claim they share a facet with qwen3.
    _qk = re.search(
        r"self\.(q_norm|q_layernorm|query_layernorm|query_norm|q_layer_norm|layernorm_q)\s*=\s*(?P<cls>[\w.]+)", src
    )
    qk_norm = "no_qk_norm" if _qk is None else "qk_norm"
    if _qk is None:
        qk_norm_class = "none"
    elif _qk.group("cls").endswith("Identity"):
        qk_norm_class = "identity"
    elif "RMSNorm" in _qk.group("cls"):
        qk_norm_class = "rmsnorm"
    else:
        qk_norm_class = "layernorm"

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

    # Reading `config.layer_types[layer_idx]` means the model mixes attention patterns per layer;
    # deleting it (as `Qwen3MoeAttention` does) means one pattern throughout.
    # Tier 1 means "changes `forward`", so read it off `forward` -- not off `__init__`. Only 3 of the
    # 53 blocks the class-body test called `per_layer_type` actually use the layer type in `forward`
    # (gemma4/gemma4_unified index `shared_kv_states[self.layer_type]`, deepseek_v4 indexes
    # `position_embeddings[self.rope_layer_type]`). For the other 50 it is an `__init__` fact.
    _fwd = _forward_source(src)
    layer_typing = "per_layer_type" if _has(_fwd, "layer_type") else "uniform_layer"
    layer_typing_init = "per_layer_type" if _has(src, "layer_types[", "self.layer_type") else "uniform_layer"

    extras = tuple(
        name
        for name, present in (
            ("attn_sink", _has(src, "sink")),
            ("logit_softcap", _has(src, "softcap", "logit_capping")),
            # Name-agnostic: `attn_output * sigmoid(self.<gate>(x))` is the same forward whether the
            # gate is called `out_gate`, `g_proj`, `gate_proj` (afmoe, hrm_text, muse_glimmer,
            # esmfold2) or `gate_attention` (evolla). Keying off four chosen spellings left six
            # gated attentions sharing a variant with ungated ones, which is an under-split: their
            # `forward` multiplies by a gate and is not inheritable from one that does not.
            (
                "out_gate",
                # `g_proj` (esmfold) and `attn_gate` do not both contain the word "gate", so the
                # explicit list stays as a union with the name-agnostic test rather than being
                # replaced by it.
                _re(src, r"self\.(out_gate|attn_gate|g_proj|q_gate_proj)\s*=")
                or _re(src, r"self\.\w*gate\w*\s*=\s*(?:nn\.Linear|nn\.Parameter|Conv1D|nn\.Conv1d)"),
            ),
        )
        if present
    )

    tier1 = {
        "mix": mix,
        "cross": cross,
        "qkv": qkv,
        "qk_norm": qk_norm,
        "rope": rope,
        "extras": "+".join(extras) or "no_extras",
        "layer_typing": layer_typing,
    }
    tier2 = {
        "qk_norm_class": qk_norm_class,
        "layer_typing_init": layer_typing_init,
        # Which degenerate case of grouped attention the config asks for. Read from the config, and
        # tier 2 on purpose: `num_key_value_heads` sizes a `nn.Linear` in `__init__` and changes
        # nothing in `forward`, so it must never split a variant.
        "kv_sharing": kv_sharing,
        # Config-declared, therefore a *model* fact, not a class-body fact: two byte-identical
        # attention classes get different values when only their configs differ (mistral/mixtral).
        # Kept in tier 2 so canonical-owner selection can still refuse to let a non-sliding model
        # own a sliding parent, without that fact splitting the variant.
        "window": window,
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
    elif _has(src, "gate_up_proj") or (
        # Structural, not by attribute name: one projection widened to 2x, split in two, and one half
        # gating the other *is* a fused SwiGLU whatever the tensors are called. Keying off names alone
        # labelled 14 of these `ungated_mlp` and one `linear_projector`, and split byte-identical
        # classes (`Dinov2SwiGLUFFN` vs `VideomtGatedMLP`) into different variants.
        _re(src, r"\.chunk\(\s*2\b")
        # any activation spelling: silu(x)*y, self.act_fn(x)*y, self.activation(x)*y, ACT2FN[c](x)*y
        and _re(src, r"(?:silu|gelu|glu|\w*[Aa]ct\w*)(?:\[[^\]]*\])?\s*\([^()]*\)\s*\*")
    ):
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
    elif len(_mlp_projections(src)) == 2:
        # Two projections in sequence with nothing gating between them *is* an ungated FFN, whatever
        # the two layers are called. The name list above is 19 spellings of "the first of two" and
        # still missed `layer_1`/`layer_2` (xlnet) and `conv_1`/`conv_2` (vits), which were reported
        # as multimodal connectors. Conv1d rather than Linear makes it the convolutional FFN that
        # `conv_ffn` already names -- vits is what that value was always for.
        gating = "conv_ffn" if _has(src, "nn.Conv1d") else "ungated_mlp"
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


def _mixer_facets(src: str, class_name: str = "") -> tuple[dict, dict]:
    """
    Linear / recurrent token mixers. Coarse on purpose: these are the frontier of the library and a
    detailed facet set would be guesswork. Recording the mechanism beats dropping the block.

    Order matters, and getting it wrong is not hypothetical: a gated DeltaNet also carries `A_log`
    and `dt_bias`, so testing for SSM first silently folded every `*GatedDeltaNet` into mamba's
    variant and the delta-rule mixers vanished from the census. The class name is the reliable
    discriminator, so it is checked before any source sniffing.
    """
    if _has(class_name, "GatedDelta", "DeltaNet") or _has(src, "gated_delta_rule", "delta_rule"):
        mechanism = "gated_delta"
    elif _has(src, "ssm_state", "selective_scan", "A_log", "dt_bias"):
        mechanism = "ssm"
    elif _has(src, "slope_rate", "decay"):
        mechanism = "decay_linear"
    else:
        mechanism = "custom_mixer"
    return {
        "mechanism": mechanism,
        # the output gate is what separates a gated delta rule from a plain one
        "gating": "output_gate" if _has(src, "g_norm", "gate_proj", "self.gate") else "no_output_gate",
        "conv": "depthwise_conv" if _has(src, "conv1d") else "no_conv",
    }, {}


def _router_facets(src: str, class_name: str = "") -> tuple[dict, dict]:
    """
    Expert routing, kept apart from the expert stack it feeds. The two really are independent
    choices: DeepSeek pairs group-limited sigmoid routing with a score-correction bias, Mixtral
    pairs plain softmax top-k with the same grouped expert weights.
    """
    if _has(src, "scoring_func", "ACT2FN["):
        scoring = "config_scoring"
    elif _has(src, "sigmoid"):
        scoring = "sigmoid_router"
    elif _has(src, "softmax"):
        scoring = "softmax_router"
    else:
        scoring = "custom_router"
    if _has(src, "tid2eid") or _has(class_name, "Hash"):
        # DeepSeek-V4 hash routing: a frozen token-id -> expert-id table decides *which* experts
        # run, so selection is static and only the weighting is learned.
        selection = "hash_table"
    elif _has(src, "sparsemixer"):
        selection = "sparsemixer"
    elif _has(src, "topk_group", "n_group", "num_group"):
        # experts are bucketed into groups and only the best groups can be selected from
        selection = "group_limited_topk"
    elif _has(class_name, "Top1"):
        selection = "top1"
    elif _has(class_name, "Top2"):
        selection = "top2"
    else:
        # arity comes from the class name, never the body: `top_1_mask` appears inside a top-2
        # router and used to mislabel it.
        selection = "flat_topk"
    return {
        "scoring": scoring,
        "selection": selection,
        "router_bias": "score_correction_bias"
        if _has(src, "e_score_correction_bias", "expert_bias", "correction_bias")
        else "no_router_bias",
        "topk_norm": "norm_topk" if _has(src, "norm_topk") else "no_norm_topk",
        "scaling": "routed_scaling" if _has(src, "routed_scaling_factor") else "no_scaling",
    }, {"jitter": "jitter_noise" if _has(src, "jitter") else "no_jitter"}


def _indexer_facets(src: str) -> tuple[dict, dict]:
    """
    Sparse-attention indexers: the lightweight scorer that picks which tokens the real attention is
    allowed to see. Separate projections from the attention they gate, which is exactly why they
    need their own entry rather than being read off the attention class.
    """
    return {
        "query_source": "lora_query" if _has(src, "wq_b", "q_lora", "q_b_proj") else "hidden_query",
        "scoring": "weighted_head_sum" if _has(src, "weights_proj") else "dot_score",
        "key_norm": "key_norm" if _has(src, "k_norm", "key_norm") else "no_key_norm",
        "output": "additive_mask" if _has(src, "masked_fill", "-inf") else "topk_indices",
    }, {"scale": "learned_scale" if _has(src, "softmax_scale") else "fixed_scale"}


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
    "router": _router_facets,
    "indexer": _indexer_facets,
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
# `norm(x + sublayer(x))` or `x = x + sublayer(x)`: an add whose left operand is the value being
# threaded through the layer, written without naming it `residual`.
_INLINE_RESIDUAL_RE = re.compile(
    r"(?:self\.\w*(?:norm|_ln|ln_)\w*\s*\([^)]*\+|(\w+)\s*=\s*\1\s*\+|hidden_states\s*\+\s*self\.)"
)


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
    lines = file_source.splitlines()
    body = "\n".join(lines[forward.lineno - 1 : forward.end_lineno])

    # Follow `apply_chunking_to_forward(self.feed_forward_chunk, ...)` into the method it calls.
    # BERT and its family put the whole feed-forward half of the layer behind that helper, so
    # reading `forward` alone reported `attn-norm-cross_attn` and hid the residual and the MLP
    # entirely -- the topology looked degenerate when it was merely indirect.
    # Inline any of the class's own methods that `forward` hands off to, whether by name in
    # `apply_chunking_to_forward(self.feed_forward_chunk, ...)` or by calling it directly.
    methods = {n.name: n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name != "forward"}
    nodes = [forward]
    for name in re.findall(r"self\.(\w+)", body):
        target = methods.get(name)
        if target is not None and target not in nodes:
            nodes.append(target)
            body += "\n" + "\n".join(lines[target.lineno - 1 : target.end_lineno])

    # Which lines perform a residual add? Regex cannot chase every naming convention -- `residual`,
    # `hidden_states`, `x`, `inputs` -- so take it from the syntax: an `Add` where one side is a bare
    # name is the layer threading a value past a sublayer. `ctrl` writes `x + attn_output`, which no
    # name-based pattern would ever catch.
    residual_lines: set[int] = set()
    for node in nodes:
        for sub in ast.walk(node):
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Add):
                if isinstance(sub.left, ast.Name) or isinstance(sub.right, ast.Name):
                    residual_lines.add(sub.lineno)

    events: list[str] = []
    numbered = [
        (forward.lineno + i, raw)
        for i, raw in enumerate("\n".join(lines[forward.lineno - 1 : forward.end_lineno]).splitlines())
    ]
    for node in nodes[1:]:
        numbered += [
            (node.lineno + i, raw)
            for i, raw in enumerate("\n".join(lines[node.lineno - 1 : node.end_lineno]).splitlines())
        ]
    for lineno, raw_line in numbered:
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
        elif lineno in residual_lines:
            # A post-norm layer writes the add inline, with no variable called `residual`:
            # `x = self.post_attention_layernorm(x + self.self_attn(x))`. Looking only for the name
            # `residual` reported those layers as having no residual at all -- 23 models read as
            # `attn-cross_attn` when they were ordinary post-norm blocks. The add precedes the norm,
            # so it is emitted before the `norm` that this same line already produced.
            # A post-norm layer adds before it normalises, and both happen on one line, so the
            # residual is placed before the `norm` this same line already emitted.
            events.insert(len(events) - 1 if events and events[-1] == "norm" else len(events), "residual")
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
    tree = ast.parse(source)
    # An indexer that another indexer instantiates is a component of it, not an indexer in its own
    # right -- the same reason a `*SelfAttention` inside an `*Attention` wrapper is skipped.
    indexer_nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and classify(n.name) == "indexer"]
    nested_indexers = {
        other.name
        for node in indexer_nodes
        for other in indexer_nodes
        if other is not node
        and re.search(rf"=\s*{re.escape(other.name)}\s*\(", "\n".join(lines[node.lineno - 1 : node.end_lineno]))
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in TRACKED_HELPERS:
            helpers.append(Helper(model, path, node.name, canonical_source(node)))
            continue
        if not isinstance(node, ast.ClassDef):
            continue
        kind = classify(node.name)
        if kind is None:
            continue
        class_source = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        if kind == "indexer" and node.name in nested_indexers:
            # `DeepseekV4IndexerScorer` is built by `DeepseekV4Indexer` (`self.scorer = ...`): it is
            # part of an indexer, not one of its own. Counting it separately made one model the
            # owner of two indexer variants and pushed the real indexer out to a suffixed key.
            continue
        if kind == "moe":
            # A model's MoE design is spread over three classes -- `*Experts` holds the weights,
            # `*SparseMoeBlock` the wiring, `*TopkRouter` the routing -- so no single class sees
            # the whole thing. Collect the candidates and emit one block per file below.
            moe_nodes.append((node, class_source))
            continue
        if kind in ("attention", "mlp") and is_container(class_source):
            continue
        if kind in ("layer", "layer_other") and is_conv_block(class_source):
            kind = "conv_block"
        if kind == "conv_block":
            tier1 = {"conv": "depthwise" if _has(class_source, "groups=") else "plain"}
            tier2 = {"norm": "yes" if _has(class_source, "Norm") else "no"}
        elif kind in ("layer", "layer_other"):
            topology = forward_topology(node, source)
            if topology is None:
                continue
            tier1, tier2 = {"topology": topology}, {}
        elif kind == "attention":
            tier1, tier2 = _attention_facets(class_source, config_flags(model), config_kv_sharing(model))
        elif kind == "mixer":
            tier1, tier2 = _mixer_facets(class_source, node.name)
        elif kind == "router":
            tier1, tier2 = _router_facets(class_source, node.name)
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
            try:
                modules = extract_model_imports_from_file(path)
            except (OSError, SyntaxError):
                # A modular file can be transiently unparseable while someone is editing it. The
                # registry must degrade rather than take the whole tool down.
                continue
            for module in modules:
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
                if not is_cross_model_import(node):
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
_COPIED_FROM_RE = re.compile(r"#\s*copied from transformers\.models\.(\w+)\.\w+\.(\w+)", re.IGNORECASE)


def is_cross_model_import(node: ast.AST) -> bool:
    """
    Whether an `ImportFrom` pulls from another model, in any of the three spellings in use.

    `from ..llama.modeling_llama import X`            (level 2, ~805 lines -- the convention)
    `from ...models.jamba.modeling_jamba import X`    (level 3, 3 models)
    `from transformers.models.blip_2.modeling_blip_2 import X`  (absolute, 13 models)

    Accepting only the relative forms made 25 declared inheritances invisible, so the registry
    reported classes as written-from-scratch when they were already inheriting correctly.
    """
    if not isinstance(node, ast.ImportFrom) or not node.module:
        return False
    if node.level in (2, 3):
        return True
    return node.level == 0 and node.module.startswith("transformers.models.")


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
                if not is_cross_model_import(node):
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
    # Still `gqa`, but the value changed meaning: it is now the *general* formulation rather than
    # "this class mentions num_key_value_heads". `mha` is no longer a member of the vocabulary.
    assert llama.tier1["mix"] == "gqa", llama.tier1
    assert llama.tier1["cross"] == "self_attention", llama.tier1
    # llama declares `num_key_value_heads=None` and fills it from `num_attention_heads`, so its
    # default really is the degenerate MHA case -- expressed in tier 2, where it cannot split it
    # from any of the 141 models that write a real kv-head count.
    assert llama.tier2["kv_sharing"] == "mha_degenerate_by_default", llama.tier2
    assert llama.tier1["qk_norm"] == "no_qk_norm", llama.tier1
    assert llama.tier2["window"] == "full_attention", llama.tier2
    assert llama.tier1["rope"] == "rope_half", llama.tier1
    assert llama.tier2["bias"] == "bias_from_config", llama.tier2

    qwen3 = next(b for b in by_model_kind[("qwen3", "attention")] if b.class_name == "Qwen3Attention")
    assert qwen3.tier1["qk_norm"] == "qk_norm", qwen3.tier1
    assert qwen3.tier2["qk_norm_class"] == "rmsnorm", qwen3.tier2
    assert qwen3.tier1["layer_typing"] == "uniform_layer", qwen3.tier1
    assert qwen3.tier2["layer_typing_init"] == "per_layer_type", qwen3.tier2

    # qwen3_moe's attention forward is byte-identical to qwen3's, but its `__init__` deletes
    # `layer_type`. Without this axis the two read as one variant with `init_diff=0`, which is how a
    # model came to inherit the MoE flavour of an attention it did not want.
    qwen3_moe = next(b for b in by_model_kind[("qwen3_moe", "attention")] if b.class_name == "Qwen3MoeAttention")
    assert qwen3_moe.tier1["layer_typing"] == "uniform_layer", qwen3_moe.tier1
    assert qwen3.variant == qwen3_moe.variant, "byte-identical forwards must share a variant"
    assert qwen3.tier2["layer_typing_init"] != qwen3_moe.tier2["layer_typing_init"], "init delta must stay visible"

    # olmoe threads `getattr(config, "sliding_window", None)` but declares no window anywhere, so
    # it must not be classified as -- let alone become the canonical owner of -- a sliding variant.
    assert "sliding_window" not in config_flags("olmoe"), config_flags("olmoe")
    olmoe = next(b for b in by_model_kind[("olmoe", "attention")] if b.class_name == "OlmoeAttention")
    assert olmoe.tier2["window"] == "full_attention", olmoe.tier2
    assert olmoe.tier2["sliding_capable"] == "sliding_capable", olmoe.tier2
    # DebertaLayerNorm computes a variance but centres and biases: it is a LayerNorm.
    deberta_norm = next(b for b in by_model_kind[("deberta", "norm")] if b.class_name == "DebertaLayerNorm")
    assert deberta_norm.tier1["norm_kind"] == "layernorm", deberta_norm.tier1
    llama_norm = next(b for b in by_model_kind[("llama", "norm")] if b.class_name == "LlamaRMSNorm")
    assert llama_norm.tier1["norm_kind"] == "rmsnorm", llama_norm.tier1

    # A gated DeltaNet also carries `A_log`/`dt_bias`; testing SSM first folded all four of them
    # into mamba's variant, so the delta-rule mixers did not appear anywhere in the census.
    delta = next(b for b in by_model_kind[("qwen3_next", "mixer")] if b.class_name == "Qwen3NextGatedDeltaNet")
    assert delta.tier1["mechanism"] == "gated_delta", delta.tier1
    mamba = next(b for b in by_model_kind[("mamba", "mixer")] if b.class_name == "MambaMixer")
    assert mamba.tier1["mechanism"] == "ssm", mamba.tier1
    assert delta.variant != mamba.variant, "gated delta and mamba must be distinguishable"

    # Sparse-attention indexers matched no kind pattern at all, so every model with one was absent.
    assert by_model_kind[("deepseek_v32", "indexer")], "deepseek_v32 indexer went missing"
    # ... but the scorer *inside* deepseek_v4's indexer is a component of it, not a second indexer.
    v4_indexers = {b.class_name for b in by_model_kind[("deepseek_v4", "indexer")]}
    assert v4_indexers == {"DeepseekV4Indexer"}, v4_indexers

    # Routing is its own decision, separate from the expert stack it feeds.
    dsr = next(b for b in by_model_kind[("deepseek_v3", "router")] if b.class_name == "DeepseekV3TopkRouter")
    assert dsr.tier1["scoring"] == "sigmoid_router", dsr.tier1
    assert dsr.tier1["selection"] == "group_limited_topk", dsr.tier1
    assert dsr.tier1["router_bias"] == "score_correction_bias", dsr.tier1
    hashr = next(b for b in by_model_kind[("deepseek_v4", "router")] if b.class_name == "DeepseekV4HashRouter")
    assert hashr.tier1["selection"] == "hash_table", hashr.tier1
    nllb = next(b for b in by_model_kind[("nllb_moe", "router")] if b.class_name == "NllbMoeTop2Router")
    assert nllb.tier1["selection"] == "top2", nllb.tier1

    # `Dinov2SwiGLUFFN` and `VideomtGatedMLP` are byte-identical fused SwiGLUs. Naming-based gating
    # detection gave them two different labels and called neither one gated.
    dino_ffn = next(b for b in by_model_kind[("dinov2", "mlp")] if b.class_name == "Dinov2SwiGLUFFN")
    videomt_ffn = next(b for b in by_model_kind[("videomt", "mlp")] if b.class_name == "VideomtGatedMLP")
    assert dino_ffn.tier1["gating"] == "fused_gate_up_mlp", dino_ffn.tier1
    assert dino_ffn.variant == videomt_ffn.variant, (dino_ffn.variant, videomt_ffn.variant)

    # ---- naming must not create variants -------------------------------------------------------
    # Three separate projections are `qkv_split` whatever they are called. `project_q` (cpmant),
    # `Wq` (ctrl) and `self_attn_query_content_proj` (dab_detr) all used to fall through to
    # `qkv_custom`; dab_detr's two singleton variants existed only because the same model spells the
    # same projection two ways in two of its own classes.
    for model, class_name in (
        ("cpmant", "CpmAntAttention"),
        ("ctrl", "MultiHeadAttention"),
        ("dab_detr", "DabDetrDecoderLayerSelfAttention"),
        ("dab_detr", "DabDetrDecoderLayerCrossAttention"),
    ):
        block = next(b for b in by_model_kind[(model, "attention")] if b.class_name == class_name)
        assert block.tier1["qkv"] == "qkv_split", (model, class_name, block.tier1)
    # ... and one 3x-wide projection that gets unpacked into three is `qkv_fused`, whether it is
    # called `att_proj` (bark) or a bare `proj` (esmfold).
    bark = next(b for b in by_model_kind[("bark", "attention")] if b.class_name == "BarkSelfAttention")
    assert bark.tier1["qkv"] == "qkv_fused", bark.tier1
    # DeBERTa's `in_proj` is genuinely one 3x projection chunked three ways. It used to read as
    # `qkv_split` only because `pos_q_proj` happens to contain the substring `q_proj`.
    deberta_attn = next(
        b for b in by_model_kind[("deberta", "attention")] if b.class_name == "DisentangledSelfAttention"
    )
    assert deberta_attn.tier1["qkv"] == "qkv_fused", deberta_attn.tier1
    # The reverse mistake: `FunnelRelMultiheadAttention` owns three real `q_head`/`k_head`/`v_head`
    # projections and was called fused because its own *class name* contains "MultiheadAttention".
    funnel = next(b for b in by_model_kind[("funnel", "attention")] if b.class_name == "FunnelRelMultiheadAttention")
    assert funnel.tier1["qkv"] == "qkv_split", funnel.tier1

    # An output gate is an output gate whatever it is called. These six multiply the attention
    # output by `sigmoid(self.gate_proj(x))` (or a tanh gate, for evolla) and used to share a
    # variant with ungated attention -- an under-split, since that `forward` is not inheritable.
    for model, class_name in (
        ("afmoe", "AfmoeAttention"),
        ("hrm_text", "HrmTextAttention"),
        ("muse_glimmer", "MuseGlimmerTextAttention"),
        ("esmfold2", "EsmFold2AtomAttention"),
        ("esmfold2", "EsmFold2DiffusionAttention"),
        ("evolla", "EvollaSequenceAlignerCrossAttention"),
    ):
        block = next(b for b in by_model_kind[(model, "attention")] if b.class_name == class_name)
        assert "out_gate" in block.tier1["extras"], (model, class_name, block.tier1)
    # `g_proj` does not contain the word "gate", so the explicit spellings must survive too.
    esmfold = next(b for b in by_model_kind[("esm", "attention")] if b.class_name == "EsmFoldSelfAttention")
    assert "out_gate" in esmfold.tier1["extras"], esmfold.tier1

    # idefics spells the query norm `q_layer_norm`. `q_a_layernorm` is deliberately *not* an alias:
    # it normalises MLA's compressed query latent, which `mix == "mla"` already reports.
    idefics = next(b for b in by_model_kind[("idefics", "attention")] if b.class_name == "IdeficsAttention")
    assert idefics.tier1["qk_norm"] == "qk_norm", idefics.tier1
    dsv3 = next(b for b in by_model_kind[("deepseek_v3", "attention")] if b.class_name == "DeepseekV3Attention")
    assert dsv3.tier1["qk_norm"] == "no_qk_norm" and dsv3.tier1["mix"] == "mla", dsv3.tier1

    # Two projections in sequence is an ungated FFN whatever the two layers are called; `conv_1` /
    # `conv_2` makes it the convolutional FFN that `conv_ffn` was always for. Both were reported as
    # multimodal connectors because their names were not on a 19-entry spelling list.
    xlnet_ffn = next(b for b in by_model_kind[("xlnet", "mlp")] if b.class_name == "XLNetFeedForward")
    assert xlnet_ffn.tier1["gating"] == "ungated_mlp", xlnet_ffn.tier1
    vits_ffn = next(b for b in by_model_kind[("vits", "mlp")] if b.class_name == "VitsFeedForward")
    assert vits_ffn.tier1["gating"] == "conv_ffn", vits_ffn.tier1

    mistral = next(b for b in by_model_kind[("mistral", "attention")] if b.class_name == "MistralAttention")
    assert mistral.tier2["window"] == "sliding_attention", mistral.tier2
    # mistral writes a real kv-head count (32 query heads, 8 kv heads): genuinely grouped, and it
    # must still share a variant with llama, whose forward is the same.
    assert mistral.tier2["kv_sharing"] == "grouped", mistral.tier2
    assert mistral.variant == llama.variant, (mistral.variant, llama.variant)

    # The pair this whole axis change exists for. These forwards are byte-identical, and the old
    # `mha`/`gqa` sniff was the only thing keeping them in different variants -- an over-split, and
    # the canonical example the audit found. `MoonshineStreamingEncoderAttention` sizes its
    # projections with a kv-head count; `ViTAttention` does not. Same formulation, same forward.
    vit = next(b for b in by_model_kind[("vit", "attention")] if b.class_name == "ViTAttention")
    moonshine = next(
        b
        for b in by_model_kind[("moonshine_streaming", "attention")]
        if b.class_name == "MoonshineStreamingEncoderAttention"
    )
    assert vit.forward == moonshine.forward, "these two forwards are byte-identical"
    assert vit.variant == moonshine.variant, (vit.variant, moonshine.variant)
    assert vit.tier2["kv_sharing"] != moonshine.tier2["kv_sharing"], "the config delta must stay visible"

    # `cross` is its own axis now. bart's attention takes `key_value_states`, so it is the
    # cross-capable formulation; llama's is not. Same `mix`, different `cross`.
    bart = next(b for b in by_model_kind[("bart", "attention")] if b.class_name == "BartAttention")
    assert bart.tier1["cross"] == "cross_attention", bart.tier1
    assert bart.tier1["mix"] == llama.tier1["mix"] == "gqa", (bart.tier1, llama.tier1)
    assert bart.variant != llama.variant, "self- and cross-attention are still different variants"

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
        # `mha` is gone on purpose: it is GQA with `num_key_value_heads == num_attention_heads`,
        # which is a config fact (tier-2 `kv_sharing`), not a mixing mechanism.
        ("attention", "mix"): {"gqa", "mla", "deformable"},
        ("attention", "cross"): {"self_attention", "cross_attention"},
        ("attention", "qkv"): {"qkv_split", "qkv_fused", "kv_fused", "kv_latent", "qkv_sampled", "qkv_custom"},
        ("attention", "qk_norm"): {"no_qk_norm", "qk_norm"},
        ("attention", "rope"): {"no_pos_emb", "rope_half", "rope_interleaved", "alibi"},
        ("attention", "layer_typing"): {"per_layer_type", "uniform_layer"},
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
            # No suffix stripping any more: `mix` used to carry an optional "_cross" suffix, which
            # is now the `cross` axis and is checked against its own vocabulary like everything else.
            assert value in allowed, f"{block.model}/{block.class_name}: {axis}={value!r} is outside {sorted(allowed)}"

    print(f"selfcheck ok: {len(blocks)} blocks, {len(helpers)} helpers, {len(build_variants(blocks))} variants")


if __name__ == "__main__":
    _selfcheck()
