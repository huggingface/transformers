"""Build a normalized ``ArchModel`` from a model's default config + meta-device module tree.

No network, no weights: we instantiate the *config class* with its defaults and build the
model under ``torch.device("meta")`` so allocation is free. If the meta build fails we fall
back to a config-only ``ArchModel`` (status ``"config-only"``). A registry of detectors maps
module class names to normalized component kinds so unknown modules degrade to their class
name rather than crashing.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field


warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------------- model


@dataclass
class LayerBlock:
    """A run of consecutive identical decoder layers, collapsed into one box."""

    layer_class: str
    count: int
    kind: str  # "attention" | "moe" | "mamba" | "linear_attention" | "recurrent" | "other"
    attn_variant: str | None = None  # MHA | GQA | MQA | MLA | sliding | None
    mlp_kind: str = "dense"  # dense | moe
    norm: str | None = None


# ---- structured "inside the block" specs (for the Raschka-style expanded full view) ----


@dataclass
class Proj:
    """A single projection (Linear) inside attention / MLP, with meta-tensor shapes."""

    name: str
    in_f: int | None
    out_f: int | None
    bias: bool = False


@dataclass
class AttentionSpec:
    cls: str
    variant: str | None = None  # MHA | GQA | MQA | MLA
    n_heads: int | None = None
    n_kv: int | None = None
    head_dim: int | None = None
    projs: list[Proj] = field(default_factory=list)
    qk_norm: bool = False
    rope: bool = False
    sliding_window: int | None = None


@dataclass
class MLPSpec:
    cls: str
    is_moe: bool = False
    act: str = "SiLU"
    gate: Proj | None = None
    up: Proj | None = None
    down: Proj | None = None
    n_experts: int | None = None
    top_k: int | None = None
    expert_dim: int | None = None
    n_shared: int | None = None
    projs: list[Proj] = field(default_factory=list)  # generic fallback when gate/up/down naming differs


@dataclass
class BlockSpec:
    """The canonical pre-norm transformer block, decomposed for detailed rendering."""

    layer_class: str
    pre_attn_norm: str | None = None
    attention: AttentionSpec | None = None
    mixer: AttentionSpec | None = None  # token mixer for non-attention blocks (Mamba/SSM/RWKV/linear-attn)
    post_attn_norm: str | None = None
    mlp: MLPSpec | None = None
    extra: list[str] = field(default_factory=list)  # e.g. sandwich-norm note


@dataclass
class ArchModel:
    model: str
    model_type: str | None = None
    config_class: str | None = None
    status: str = "ok"  # ok | config-only | failed
    error: str | None = None

    # scalar config facts
    hidden_size: int | None = None
    vocab_size: int | None = None
    num_layers: int | None = None
    num_attention_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    intermediate_size: int | None = None
    max_position_embeddings: int | None = None
    sliding_window: int | None = None

    # normalized architecture facts
    norm_type: str | None = None
    positional: str | None = None
    attn_variant: str | None = None
    decoder_kind: str = "dense"  # dense | sparse-moe | hybrid | ssm | other
    is_moe: bool = False
    num_experts: int | None = None
    experts_per_token: int | None = None
    tie_word_embeddings: bool | None = None

    layer_blocks: list[LayerBlock] = field(default_factory=list)
    layer_summary: str | None = None  # e.g. "36× DeltaNet + 12× Attention"
    module_kinds: dict[str, int] = field(default_factory=dict)  # class name -> count (top kinds)
    top_class: str | None = None  # outer model class name
    block: BlockSpec | None = None  # decomposed decoder block internals (full view)
    generic_tree: list[dict] = field(default_factory=list)  # full module tree for non-transformer models
    flow: list[dict] = field(default_factory=list)  # per-stage forward I/O shapes (data-flow view)
    flow_input: str | None = None  # input tensor label, e.g. "input_values [1, 1, 8000]"
    flow_output: list[int] | None = None  # final output shape
    is_pipeline: bool = False  # ≥3 distinct top-level stage sub-networks (e.g. BLT, codecs)
    rope_theta: float | None = None
    hidden_act: str | None = None
    layer_types: list[str] = field(default_factory=list)  # config.layer_types schedule (per layer)
    alt_blocks: list[str] = field(default_factory=list)  # other layer classes (hybrid) not shown in detail
    checkpoint: str | None = None  # example HF checkpoint id (from @auto_docstring), offline
    attn_patterns: dict = field(default_factory=dict)  # layer_type -> 0/1 mask grid
    tokens: list[str] = field(default_factory=list)  # example tokenized input shown at the top
    # sparse/compressed attention components (e.g. DeepSeek-V4 HCA/CSA), per distinct layer_type
    sparse_components: list[dict] = field(default_factory=list)
    # fully-decomposed module trees (recursive) of each distinct attention variant + the FFN
    attention_variants: list[dict] = field(default_factory=list)
    mlp_tree: dict | None = None
    mlp_variants: list[dict] = field(default_factory=list)  # per distinct mlp_layer_type (moe/hash_moe/dense)
    mlp_layer_types: list[str] = field(default_factory=list)
    # multimodal (VLM / audio) structure
    is_multimodal: bool = False
    towers: list[dict] = field(default_factory=list)  # encoder towers (vision/audio) + projector
    cross_attention_layers: list | None = None  # LLM layer indices that cross-attend to features
    modal_inputs: list[str] = field(default_factory=list)  # e.g. ["input_ids", "pixel_values"]
    auto_classes: list[str] = field(default_factory=list)  # Auto* classes used to build sub-models
    image_bidirectional: bool = False  # VLM: image tokens attend bidirectionally (prefix-LM)
    family: str | None = None  # causal_lm | image_classification | audio_classification | seq2seq | ...
    head_class: str | None = None  # the task wrapper class, e.g. ViTForImageClassification
    view: str = "decoder"  # decoder | encoder | enc_dec | multimodal — which layout to use
    num_labels: int | None = None


# ----------------------------------------------------------------------------- detectors

_MOE_RE = re.compile(r"(expert|router|moe|sparsemoe|sparse_moe)", re.IGNORECASE)
_MAMBA_RE = re.compile(r"(mamba|mixer|ssm|\bs4\b|selectivescan)", re.IGNORECASE)
_LINEAR_ATTN_RE = re.compile(r"(deltanet|linearattention|lineattn|gateddelta|gla\b|retention|rwkv)", re.IGNORECASE)
_ATTN_RE = re.compile(r"attention$", re.IGNORECASE)
_NORM_RE = re.compile(r"norm", re.IGNORECASE)


def _kind_for_layer(class_name: str) -> str:
    n = class_name.lower()
    if _MAMBA_RE.search(n):
        return "mamba"
    if _LINEAR_ATTN_RE.search(n):
        return "linear_attention"
    if "recurrent" in n or "rnn" in n:
        return "recurrent"
    return "attention"


def layer_type_kind(t: str) -> str:
    """Classify a ``config.layer_types`` entry (e.g. 'sliding_attention') into a kind.

    Sliding/full/compressed are all *attention* (they differ only in the mask); linear /
    delta / gated / mamba / recurrent are structurally different layers.
    """
    n = t.lower()
    if _MAMBA_RE.search(n):
        return "mamba"
    if _LINEAR_ATTN_RE.search(n) or "linear" in n or "delta" in n or "gated" in n:
        return "linear_attention"
    if "recurrent" in n:
        return "recurrent"
    return "attention"


def _cfg(config, *names, default=None):
    for n in names:
        if hasattr(config, n):
            v = getattr(config, n)
            if v is not None:
                return v
    return default


def _attn_variant(config) -> str | None:
    heads = _cfg(config, "num_attention_heads", "n_head", "num_heads")
    kv = _cfg(config, "num_key_value_heads", "num_kv_heads", default=heads)
    # Multi-head Latent Attention (DeepSeek style)
    if _cfg(config, "kv_lora_rank", "q_lora_rank") is not None:
        return "MLA"
    if heads is None:
        return None
    if kv is None:
        return "MHA"
    if kv == 1:
        return "MQA"
    if kv < heads:
        return "GQA"
    return "MHA"


def _positional(config, module_classes: set[str]) -> str:
    if any("rotary" in c.lower() or "rope" in c.lower() for c in module_classes):
        return "RoPE"
    if _cfg(config, "rope_parameters", "rope_theta", "rope_scaling") is not None:
        return "RoPE"
    if _cfg(config, "alibi") or any("alibi" in c.lower() for c in module_classes):
        return "ALiBi"
    if any(c in ("PositionEmbedding", "LearnedPositionalEmbedding") for c in module_classes):
        return "learned-absolute"
    # presence of a position embedding nn.Embedding sized to max_position_embeddings
    return "RoPE" if _cfg(config, "rotary_emb_base") is not None else "n/a"


def _norm_type(module_classes: set[str]) -> str | None:
    norms = sorted({c for c in module_classes if _NORM_RE.search(c)})
    if not norms:
        return None
    # prefer the model-specific norm name; normalize common ones
    for c in norms:
        if "rms" in c.lower():
            return "RMSNorm"
    for c in norms:
        if "layernorm" in c.lower():
            return "LayerNorm"
    return norms[0]


# ----------------------------------------------------------------------- module-tree walk


def _find_layer_list(model, num_layers):
    """Find the nn.ModuleList holding the decoder/encoder layers."""
    import torch.nn as nn

    best = None
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            # prefer one whose length matches num_layers, else the longest
            if num_layers and len(module) == num_layers:
                return name, module
            if best is None or len(module) > len(best[1]):
                best = (name, module)
    return best if best else (None, None)


def _collapse(class_names: list[str]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for name in class_names:
        if out and out[-1][0] == name:
            out[-1] = (name, out[-1][1] + 1)
        else:
            out.append((name, 1))
    return out


def _all_configs(config):
    """Yield the config and every nested sub-config (``*_config``)."""
    yield config
    for k in list(vars(config)):
        if k.endswith("_config"):
            sc = getattr(config, k, None)
            if sc is not None and hasattr(sc, "model_type"):
                yield from _all_configs(sc)


def _sanitize_config(config) -> bool:
    """Fix invalid default-config combos that break a meta build (generic, no per-model code).

    The common one: ``hidden_size % num_attention_heads != 0`` in a sub-config (e.g. some
    vision towers). We snap ``num_attention_heads`` to the largest divisor ≤ the original and
    make ``num_key_value_heads`` divide it. Returns True if anything changed.
    """
    changed = False
    for c in _all_configs(config):
        h = getattr(c, "hidden_size", None)
        nh = getattr(c, "num_attention_heads", None)
        if isinstance(h, int) and isinstance(nh, int) and nh > 0 and h % nh != 0:
            d = next((x for x in range(nh, 0, -1) if h % x == 0), 1)
            c.num_attention_heads = d
            changed = True
            kv = getattr(c, "num_key_value_heads", None)
            if isinstance(kv, int) and (kv == 0 or d % kv != 0):
                c.num_key_value_heads = d
    return changed


def _build_meta_model(config, device: str = "meta"):
    import torch

    import transformers as tf

    # try AutoModel first, then common task factories, then EVERY remaining AutoModelFor*
    # (so depth / segmentation / time-series / retrieval wrappers all build without hardcoding)
    preferred = [
        "AutoModel",
        "AutoModelForImageTextToText",
        "AutoModelForCausalLM",
        "AutoModelForImageClassification",
        "AutoModelForDepthEstimation",
        "AutoModelForSemanticSegmentation",
        "AutoModelForPreTraining",
        "AutoModelForSeq2SeqLM",
    ]
    rest = sorted(n for n in dir(tf) if n.startswith("AutoModel") and n not in preferred)
    factories, seen = [], set()
    for n in preferred + rest:
        f = getattr(tf, n, None)
        if f is not None and hasattr(f, "from_config") and n not in seen:
            seen.add(n)
            factories.append(f)
    with torch.device(device):
        last = None
        for sanitize in (False, True):
            if sanitize and not _sanitize_config(config):
                break
            for fac in factories:
                try:
                    return fac.from_config(config)
                except Exception as e:
                    last = e
        raise last


_CKPT_RE = re.compile(r"""checkpoint\s*=\s*["']([\w\-./]+/[\w\-.]+)["']""")


def _example_checkpoint(model: str, mt: str) -> str | None:
    """Best-effort example checkpoint id from ``@auto_docstring(checkpoint=...)`` in the
    model's source (offline; we just read the files). Returns e.g. 'google/gemma-7b'."""
    import os

    from .discover import models_root

    d = os.path.join(models_root(), model)
    if not os.path.isdir(d):
        return None
    for f in sorted(os.listdir(d)):
        if f.startswith(("modeling_", "configuration_", "modular_")) and f.endswith(".py"):
            try:
                m = _CKPT_RE.search(open(os.path.join(d, f), encoding="utf-8").read())
            except Exception:
                continue
            if m:
                return m.group(1)
    return None


_ACT_NAMES = {"gelu", "relu", "silu", "tanh", "sigmoid", "swish", "mish", "quickgelu", "newgelu", "geglu"}


def _classify_mod(name: str, cls: str) -> str:
    """Map a submodule to a *standardized* semantic kind used for colouring across every view."""
    n = (name + " " + cls).lower()
    c = cls.lower()
    if "rotary" in n or "rope" in n:
        return "rope"
    if "norm" in c:
        return "norm"
    if "conv" in c:  # Conv1d/2d/3d, ParametrizedConv*, depthwise conv, ...
        return "conv"
    if any(k in c for k in ("lstm", "gru", "rnn")):
        return "recurrent"
    if "embedding" in c or "embed" in c:
        return "embedding"
    if "pool" in c:
        return "pool"
    if "dropout" in c or c == "identity" or "drop_path" in n:
        return "dropout"
    if "activation" in c or c in _ACT_NAMES or name in ("act", "activation", "act_fn"):
        return "act"
    if "quantiz" in c or "codebook" in c:
        return "quantizer"
    if "indexer" in n:
        return "indexer"
    if "compress" in n:
        return "compressor"
    if name == "gate" or "router" in n:
        return "router"
    if "expert" in n:
        return "experts"
    if any(k in n for k in ("classifier", "lm_head", "score")) or name == "head":
        return "head"
    if "linear" in c or "proj" in name:
        return "linear"
    return "other"


def _decompose(module, depth: int) -> list[dict]:
    """Recursively decompose a module into a tree of ``{name, cls, kind, dim, children}``."""
    import torch.nn as nn

    nodes = []
    for n, c in module.named_children():
        cls = type(c).__name__
        kind = _classify_mod(n, cls)
        dim = None
        if isinstance(c, nn.Linear):
            dim = f"[{c.in_features}→{c.out_features}]"
        else:
            w = getattr(c, "weight", None)
            if w is not None and hasattr(w, "shape") and 1 <= len(w.shape) <= 3:
                dim = "×".join(str(s) for s in tuple(w.shape))
        is_leaf = isinstance(c, nn.Linear) or kind in ("norm", "rope")
        children = _decompose(c, depth - 1) if (depth > 0 and not is_leaf) else []
        nodes.append({"name": n, "cls": cls, "kind": kind, "dim": dim, "children": children})
    return nodes


def _mod_dim(c) -> str | None:
    """A compact dimension label for any leaf module (Linear / Conv / Embedding / parametrized)."""
    import torch.nn as nn

    if isinstance(c, nn.Linear):
        return f"[{c.in_features}→{c.out_features}]"
    if isinstance(c, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        k = "×".join(str(x) for x in c.kernel_size)
        g = f" g{c.groups}" if c.groups != 1 else ""
        return f"{c.in_channels}→{c.out_channels} k{k}{g}"
    if isinstance(c, nn.Embedding):
        return f"[{c.num_embeddings}×{c.embedding_dim}]"
    w = getattr(c, "weight", None)
    if w is not None and hasattr(w, "shape") and 1 <= len(w.shape) <= 2:
        return "×".join(str(s) for s in tuple(w.shape))
    return None


def _is_leaf_mod(c) -> bool:
    import torch.nn as nn

    return isinstance(c, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Embedding)) or bool(
        _NORM_RE.search(type(c).__name__)
    )


def _node_for(name, c, depth) -> dict:
    leaf = _is_leaf_mod(c) or not list(c.named_children())
    return {
        "name": name,
        "cls": type(c).__name__,
        "kind": _classify_mod(name, type(c).__name__),
        "dim": _mod_dim(c),
        "children": _module_tree(c, depth - 1) if (depth > 0 and not leaf) else [],
    }


def _collapse_sequence(kids, depth: int) -> list[dict]:
    """Collapse a sequence of children by detecting the smallest repeating *period*: a run like
    ``(Conv1d, ResnetBlock) ×4`` becomes one ``×4`` group, ``[Stage, Stage, Stage] ×3`` becomes
    ``×3 Stage``. Non-repeating elements (an input stem, a trailing LSTM, ...) stay as singles."""
    sigs = [type(c).__name__ for _, c in kids]
    n = len(kids)
    out: list[dict] = []
    i = 0
    while i < n:
        best = None
        for p in range(1, (n - i) // 2 + 1):  # smallest period that repeats ≥ twice wins
            k = 1
            while i + (k + 1) * p <= n and sigs[i + k * p : i + (k + 1) * p] == sigs[i : i + p]:
                k += 1
            if k >= 2:
                best = (p, k)
                break
        if best:
            p, k = best
            members = kids[i : i + p]
            if p == 1:  # a homogeneous run → "×k ClassName" with the block's internals
                nm, c = members[0]
                g = _node_for(nm, c, depth)
                g["name"], g["cls"] = f"[{i}:{i + k}]", f"{k}× {type(c).__name__}"
                out.append(g)
            else:  # a periodic pattern → "×k (A, B, …)" containing one copy of the pattern
                out.append(
                    {
                        "name": f"[{i}:{i + p * k}]",
                        "cls": f"{k}× (" + ", ".join(sigs[i : i + p]) + ")",
                        "kind": "other",
                        "dim": None,
                        "children": [_node_for(nm, c, depth) for nm, c in members],
                    }
                )
            i += p * k
        else:
            out.append(_node_for(*kids[i], depth))
            i += 1
    return out


def _module_tree(module, depth: int) -> list[dict]:
    """Recursive module hierarchy with repeating ``ModuleList``/``Sequential`` runs collapsed to
    ``×N`` (see ``_collapse_sequence``). This renders *any* architecture — conv backbones, codecs,
    FFT mixers, detectors — from its real structure, never a config-only schematic."""
    import torch.nn as nn

    nodes: list[dict] = []
    for n, c in module.named_children():
        cls = type(c).__name__
        kids = list(c.named_children())
        if isinstance(c, (nn.ModuleList, nn.Sequential)) and len(kids) > 1:
            groups = _collapse_sequence(kids, depth) if depth > 0 else []
            if len(groups) == 1 and groups[0]["cls"].startswith(f"{len(kids)}× "):
                # whole list is one homogeneous repeat → inline it (e.g. "stages · 4× ResNetStage")
                g = groups[0]
                nodes.append({"name": n, "cls": g["cls"], "kind": g["kind"], "dim": None, "children": g["children"]})
            else:
                nodes.append(
                    {
                        "name": n,
                        "cls": f"{cls} [{len(kids)}]",
                        "kind": _classify_mod(n, cls),
                        "dim": None,
                        "children": groups,
                    }
                )
            continue
        nodes.append(_node_for(n, c, depth))
    return nodes


def _out_shape(x) -> list[int] | None:
    """Best-effort tensor shape of a module's input/output: tensor, tuple/list, or ModelOutput."""
    import torch

    if torch.is_tensor(x):
        return list(x.shape)
    if isinstance(x, (list, tuple)):
        for e in x:
            s = _out_shape(e)
            if s:
                return s
        return None
    for attr in ("last_hidden_state", "logits", "sample", "audio_values", "audio_codes", "hidden_states"):
        v = getattr(x, attr, None)
        s = _out_shape(v) if v is not None else None
        if s:
            return s
    return None


def _dummy_inputs(model, config, device: str) -> dict:
    """A minimal forward input matching the model's modality (no tokenizer / processor needed)."""
    import torch

    mi = getattr(model, "main_input_name", None) or "input_ids"
    if mi in ("input_values", "input_features"):
        return {mi: torch.zeros(1, 1, 8000, device=device)}
    if mi == "pixel_values":
        sz = getattr(config, "image_size", None) or 64
        sz = sz if isinstance(sz, int) else (sz[0] if isinstance(sz, (list, tuple)) else 64)
        ch = getattr(config, "num_channels", None) or 3
        return {mi: torch.zeros(1, ch, sz, sz, device=device)}
    return {"input_ids": torch.ones(1, 8, dtype=torch.long, device=device)}


class _Deadline:
    """Hard wall-clock timeout (SIGALRM) so a slow/looping forward can never hang the build.
    Works in the main thread of each ProcessPoolExecutor worker; a no-op where SIGALRM is absent."""

    def __init__(self, seconds: float):
        self.seconds = seconds

    def __enter__(self):
        import signal

        self._sig = getattr(signal, "SIGALRM", None)
        if self._sig is not None:

            def _raise(*_):
                raise TimeoutError("forward exceeded time budget")

            self._old = signal.signal(self._sig, _raise)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *exc):
        import signal

        if self._sig is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(self._sig, self._old)
        return False


def _run_flow(model, config, device: str) -> tuple[str, list[dict], list[int] | None] | None:
    """Run one forward pass capturing the in→out shape of each top-level stage (in call order),
    via forward hooks. Returns (input_label, stages, output_shape) or None on failure."""
    import torch

    model.eval()
    inp = _dummy_inputs(model, config, device)
    in_name = next(iter(inp))
    in_shape = list(next(iter(inp.values())).shape)
    flow: list[dict] = []
    hooks = []
    order = {id(m): i for i, (_, m) in enumerate(model.named_children())}

    def mk(nm):
        def hook(mod, i, o):
            flow.append(
                {
                    "name": nm,
                    "cls": type(mod).__name__,
                    "in_shape": _out_shape(i),
                    "out_shape": _out_shape(o),
                    "_ord": order.get(id(mod), 999),
                }
            )

        return hook

    for nm, mod in model.named_children():
        hooks.append(mod.register_forward_hook(mk(nm)))
    out = None
    try:
        with torch.no_grad(), _Deadline(8):
            out = model(**inp)
    except BaseException:  # includes TimeoutError from the deadline
        for h in hooks:
            h.remove()
        return None
    for h in hooks:
        h.remove()
    if not flow:
        return None
    # de-dup (a stage hit more than once keeps its first call) and keep call order
    seen, uniq = set(), []
    for f in flow:
        if f["name"] in seen:
            continue
        seen.add(f["name"])
        uniq.append(f)
    return (f"{in_name} {in_shape}", uniq, _out_shape(out))


def _shrink_config(config):
    """Shrink a config in-place so it can be instantiated cheaply on CPU for a real forward."""
    for k in (
        "num_hidden_layers",
        "num_layers",
        "decoder_layers",
        "encoder_layers",
        "n_layers",
        "num_local_experts",
        "n_routed_experts",
    ):
        if isinstance(getattr(config, k, None), int):
            setattr(config, k, min(2, getattr(config, k)))
    # cap the big allocations (vocab × hidden embeddings, position tables) so a CPU build is cheap
    for k, cap in (("vocab_size", 256), ("max_position_embeddings", 64), ("max_target_positions", 64)):
        if isinstance(getattr(config, k, None), int) and getattr(config, k) > cap:
            setattr(config, k, cap)
    for sub in ("encoder", "decoder", "text_config", "vision_config", "audio_config"):
        c = getattr(config, sub, None)
        if hasattr(c, "__dict__"):
            _shrink_config(c)
    return config


def _stage_children(model_obj) -> list:
    """Top-level children that are substantial *stage* sub-networks (own ModuleList or ≥2 kids),
    i.e. a real pipeline rather than a single backbone + head. Excludes embeddings/norms/heads."""
    import torch.nn as nn

    stages = []
    for n, c in model_obj.named_children():
        if _is_leaf_mod(c):
            continue
        kids = list(c.named_children())
        if not kids:
            continue
        has_list = any(isinstance(m, (nn.ModuleList, nn.Sequential)) for m in c.modules())
        if has_list or len(kids) >= 2:
            stages.append((n, c))
    return stages


def _capture_flow(model_obj, config, model_type) -> tuple[str, list[dict], list[int] | None] | None:
    """Capture data-flow shapes via a forward on the ALREADY-BUILT *meta* model — meta tensors
    carry shapes but allocate ~no real memory, so this is safe to run for every model in parallel.
    Models whose forward can't run on meta (data-dependent ops) simply don't get the flow view;
    they fall back to the structural module-tree view. We never build a real model here."""
    try:
        f = _run_flow(model_obj, config, "meta")
        if f and len(f[1]) >= 2:
            return f
    except BaseException:
        pass
    return None


_HEAD_PRIORITY = [
    "IMAGE_TEXT_TO_TEXT",
    "SEQ_TO_SEQ_CAUSAL_LM",
    "CAUSAL_LM",
    "IMAGE_CLASSIFICATION",
    "AUDIO_CLASSIFICATION",
    "MASKED_LM",
    "SEQUENCE_CLASSIFICATION",
    "TOKEN_CLASSIFICATION",
    "PRETRAINING",
]


def _detect_family(mt: str, config):
    """Return (family, head_class) by scanning the Auto MODEL_FOR_* mappings."""
    try:
        import transformers.models.auto.modeling_auto as ma
    except Exception:
        return None, None
    for key in _HEAD_PRIORITY:
        m = getattr(ma, f"MODEL_FOR_{key}_MAPPING_NAMES", {})
        if mt in m:
            hc = m[mt]
            if isinstance(hc, (tuple, list)):  # some types map to several heads; take the first
                hc = hc[0] if hc else None
            return key.lower(), hc
    return None, None


def _attn_child(layer):
    for n, c in layer.named_children():
        if "attention" in type(c).__name__.lower() or n in ("self_attn", "attn", "attention", "self_attention"):
            return c
    return None


def _mixer_child(layer):
    """The token-mixing submodule of a non-attention block (Mamba/SSM/RWKV/linear-attention)."""
    # prefer the conventionally-named mixer child
    for n, c in layer.named_children():
        if n in ("mixer", "ssm", "temporal_mixer", "time_mixer"):
            return c
    # else a non-norm child whose class name names a known mixer family
    for n, c in layer.named_children():
        cls = type(c).__name__
        if _NORM_RE.search(cls):
            continue
        low = cls.lower()
        if any(k in low for k in ("mamba", "mixer", "ssm", "rwkv", "retention", "lineardelta")):
            return c
    return None


def _block_core(layer):
    """The token-mixing module of a block: real attention if present, else an SSM/recurrent mixer."""
    return _attn_child(layer) or _mixer_child(layer)


def _annotate_moe(nodes: list[dict], am, ed) -> None:
    """Fill experts/router counts (fused experts have no per-expert submodules on meta)."""
    for nd in nodes:
        if nd["kind"] == "experts" and not nd.get("dim"):
            nd["dim"] = f"×{am.num_experts or '?'} · dim {ed or '?'}"
        if nd["kind"] == "router":
            nd["dim"] = f"top-{am.experts_per_token or '?'} of {am.num_experts or '?'}"


def _mlp_child(layer):
    for n, c in layer.named_children():
        cl = type(c).__name__.lower()
        if n in ("mlp", "feed_forward", "ffn", "block_sparse_moe", "moe") or cl.endswith(
            ("mlp", "moe", "moeblock", "sparsemoeblock", "feedforward")
        ):
            return c
    return None


def _linear_proj(name, module) -> Proj | None:
    import torch.nn as nn

    if isinstance(module, nn.Linear):
        return Proj(name=name, in_f=module.in_features, out_f=module.out_features, bias=module.bias is not None)
    # some models use Conv1D (gpt2) with weight shape (in, out)
    w = getattr(module, "weight", None)
    if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
        return Proj(
            name=name, in_f=int(w.shape[1]), out_f=int(w.shape[0]), bias=getattr(module, "bias", None) is not None
        )
    return None


def _extract_block(layer, tcfg, am) -> BlockSpec | None:
    """Decompose one decoder layer into a canonical pre-norm block for detailed rendering."""

    spec = BlockSpec(layer_class=type(layer).__name__)
    norms: list[tuple[str, str]] = []  # (child_name, class_name)
    attn_mod = mlp_mod = None
    for cname, child in layer.named_children():
        cls = type(child).__name__
        low = (cname + " " + cls).lower()
        if _NORM_RE.search(cls):
            norms.append((cname, cls))
        elif low.find("attention") >= 0 or cname in ("self_attn", "attn", "attention", "self_attention"):
            attn_mod = (cname, child)
        elif cname in ("mlp", "feed_forward", "ffn", "block_sparse_moe", "moe", "mlp_layer") or cls.lower().endswith(
            ("mlp", "moe", "moeblock", "sparsemoeblock", "feedforward")
        ):
            mlp_mod = (cname, child)

    # ---- attention ----
    if attn_mod is not None:
        _, am_mod = attn_mod
        aspec = AttentionSpec(cls=type(am_mod).__name__, variant=am.attn_variant)
        aspec.n_heads = am.num_attention_heads
        aspec.n_kv = am.num_kv_heads
        aspec.head_dim = am.head_dim
        aspec.sliding_window = am.sliding_window if isinstance(am.sliding_window, int) else None
        for n, c in am_mod.named_children():
            p = _linear_proj(n, c)
            if p is not None:  # every Linear/Conv1D in attention is a projection (q/k/v/o, fused qkv, lora)
                aspec.projs.append(p)
            if _NORM_RE.search(type(c).__name__) and ("q_norm" in n or "k_norm" in n or n.endswith("norm")):
                aspec.qk_norm = True
            if "rotary" in type(c).__name__.lower():
                aspec.rope = True
        aspec.rope = aspec.rope or (am.positional == "RoPE")
        spec.attention = aspec

    # ---- mlp / moe ----
    if mlp_mod is not None:
        mname, mm = mlp_mod
        mcls = type(mm).__name__
        # MoE if this module (or any descendant) is a router/experts/sparse block
        is_moe = bool(_MOE_RE.search(mcls)) or any(_MOE_RE.search(type(c).__name__) for c in mm.modules())
        mspec = MLPSpec(cls=mcls, is_moe=is_moe, act=(am.hidden_act or "SiLU"))
        if is_moe:
            mspec.n_experts = am.num_experts
            mspec.top_k = am.experts_per_token
            mspec.expert_dim = _cfg(tcfg, "moe_intermediate_size", "expert_intermediate_size", "intermediate_size")
            mspec.n_shared = _cfg(tcfg, "n_shared_experts", "num_shared_experts")  # counts only
        else:
            projs = {}
            for n, c in mm.named_children():
                p = _linear_proj(n, c)
                if p is not None:
                    projs[n.lower()] = p
            mspec.projs = list(projs.values())
            mspec.gate = next((projs[k] for k in projs if "gate" in k and "up" not in k), None)
            mspec.up = next(
                (projs[k] for k in projs if k.startswith("up") or "up_proj" in k or "fc" in k or "c_fc" in k), None
            )
            mspec.down = next(
                (
                    projs[k]
                    for k in projs
                    if "down" in k
                    or "c_proj" in k
                    or k.endswith(("proj2", "_proj"))
                    and "up" not in k
                    and "gate" not in k
                ),
                None,
            )
        spec.mlp = mspec

    # ---- norms (canonical pre-norm: input -> attn, post-attn -> mlp) ----
    if norms:
        pre = next((c for n, c in norms if "input" in n or "pre" in n or n == norms[0][0]), norms[0][1])
        spec.pre_attn_norm = pre
        post = next((c for n, c in norms if "post" in n or "attention" in n), None)
        spec.post_attn_norm = post or (norms[1][1] if len(norms) > 1 else pre)
        if len(norms) > 2:
            spec.extra.append(f"{len(norms)} norms (sandwich/extra)")

    # ---- token mixer (Mamba/SSM/RWKV) when there is no attention ----
    if spec.attention is None:
        mix = _mixer_child(layer)
        if mix is not None:
            mspec2 = AttentionSpec(cls=type(mix).__name__, variant="SSM")
            for n, c in mix.named_children():
                p = _linear_proj(n, c)
                if p is not None:
                    mspec2.projs.append(p)
            spec.mixer = mspec2

    if spec.attention is None and spec.mixer is None and spec.mlp is None:
        return None
    return spec


_EXAMPLE_PROMPT = "Hey, how are you?"
_FALLBACK_TOKENS = ["Hey", ",", "␣how", "␣are", "␣you", "?"]


def _example_tokens(checkpoint: str | None) -> list[str]:
    """Tokenize a fixed example prompt with the model's tokenizer (offline / cached only).

    Falls back to a fixed illustrative token list when no cached tokenizer is available.
    Token strings are cleaned: the SentencePiece ``▁`` and BPE ``Ġ`` space markers become a
    visible ``␣`` so the per-token boxes read naturally.
    """
    if not checkpoint:
        return list(_FALLBACK_TOKENS)
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        ids = tok(_EXAMPLE_PROMPT, add_special_tokens=False)["input_ids"]
        pieces = tok.convert_ids_to_tokens(ids)
        out = []
        for p in pieces[:10]:
            out.append(p.replace("▁", "␣").replace("Ġ", "␣").replace("Ċ", "⏎"))
        return out or list(_FALLBACK_TOKENS)
    except Exception:
        return list(_FALLBACK_TOKENS)


_AUTO_RE = re.compile(r"\b(AutoModel(?:ForCausalLM|ForConditionalGeneration|ForImageTextToText)?|AutoModelFor\w+)\b")


def _auto_classes(model: str) -> list[str]:
    """Which Auto* classes the modeling file uses to instantiate sub-models (offline grep).

    Composition models build their towers with e.g. ``AutoModel.from_config(vision_config)``;
    surfacing this is important — the vision tower or LLM is often an AutoModel.
    """
    import os

    from .discover import models_root

    d = os.path.join(models_root(), model)
    if not os.path.isdir(d):
        return []
    found: set[str] = set()
    for f in sorted(os.listdir(d)):
        if f.startswith(("modeling_", "modular_")) and f.endswith(".py"):
            try:
                src = open(os.path.join(d, f), encoding="utf-8").read()
            except Exception:
                continue
            for m in _AUTO_RE.findall(src):
                found.add(m)
    return sorted(found)


_VISION_RE = re.compile(r"(vision|visual|image|vit|clip|siglip|patch)", re.IGNORECASE)
_AUDIO_RE = re.compile(r"(audio|speech|wav|mel|acoustic|whisper)", re.IGNORECASE)
_PROJ_RE = re.compile(r"(projector|connector|multi_modal|mm_proj|adapter|merger|aligner)", re.IGNORECASE)
_LLM_RE = re.compile(r"(language_model|text_model|^model$|decoder|thinker)", re.IGNORECASE)


def _subconfig(config, *names):
    for n in names:
        sc = getattr(config, n, None)
        if sc is not None and hasattr(sc, "model_type"):
            return sc
    return None


def _tower_summary(child_name, child, config) -> dict:
    """Summarize one top-level modality tower (encoder/projector) for the multi-tower view."""
    cls = type(child).__name__
    role = "other"
    nl = child_name.lower() + " " + cls.lower()
    if _PROJ_RE.search(nl):
        role = "projector"
    elif _VISION_RE.search(nl):
        role = "vision"
    elif _AUDIO_RE.search(nl):
        role = "audio"
    elif _LLM_RE.search(child_name.lower()):
        role = "text"
    sc = None
    if role == "vision":
        sc = _subconfig(config, "vision_config")
    elif role == "audio":
        sc = _subconfig(config, "audio_config")
    layers = hidden = mtype = None
    if sc is not None:
        layers = _cfg(sc, "num_hidden_layers", "depth", "num_layers")
        hidden = _cfg(sc, "hidden_size", "d_model", "embed_dim")
        mtype = getattr(sc, "model_type", None)
    out = {"role": role, "name": child_name, "cls": cls, "model_type": mtype, "layers": layers, "hidden": hidden}
    # decompose small towers (projector / connector) fully so their inside is visible.
    n_sub = sum(1 for _ in child.modules())
    if role == "projector" or (role in ("vision", "audio") and n_sub <= 12):
        out["children"] = _decompose(child, 2)
    elif role in ("vision", "audio"):
        # big encoder stack: show ONE representative layer (decomposed) ×N, like the LLM block
        import contextlib

        with contextlib.suppress(Exception):
            _, elayers = _find_layer_list(child, layers)
            if elayers is not None and len(elayers) > 0:
                rep = elayers[0]
                if _attn_child(rep) is None:  # hierarchical (e.g. swin stage) -> descend
                    rep = next((m for m in rep.modules() if _attn_child(m) is not None), rep)
                out["block_children"] = _decompose(rep, 2)
                out["block_class"] = type(rep).__name__
                out["block_n"] = len(elayers)
    return out


def _detect_towers(model_obj, config) -> list[dict]:
    """Detect modality towers from the top-level children of a (multimodal) model."""
    towers = []
    for n, c in model_obj.named_children():
        cls = type(c).__name__
        # skip plain containers like embeddings/heads at the very top
        if cls in ("Embedding", "Linear") and not _PROJ_RE.search(n.lower()):
            continue
        t = _tower_summary(n, c, config)
        if t["role"] in ("vision", "audio", "projector", "text"):
            towers.append(t)
    return towers


def introspect(model: str, model_type: str | None = None) -> ArchModel:
    """Introspect ``model`` into an ``ArchModel``. Never raises -- failures are recorded."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    am = ArchModel(model=model, model_type=model_type)
    mt = model_type or model
    try:
        config_cls = CONFIG_MAPPING[mt]
    except Exception:
        # try to recover model_type from the directory name
        config_cls = None
        for k, cls in CONFIG_MAPPING.items():
            if k.replace("-", "_") == model:
                config_cls, mt = cls, k
                break
        if config_cls is None:
            am.status = "failed"
            am.error = f"no config class for model_type={mt!r}"
            return am

    am.model_type = mt
    am.config_class = config_cls.__name__
    try:
        config = config_cls()
    except Exception as e:  # some configs require args
        am.status = "failed"
        am.error = f"config instantiation failed: {e!r}"[:300]
        return am

    # decoder/text config (multimodal configs nest a text_config)
    tcfg = getattr(config, "get_text_config", lambda: config)()
    am.cross_attention_layers = _cfg(tcfg, "cross_attention_layers")
    has_vision_cfg = _subconfig(config, "vision_config") is not None
    has_audio_cfg = _subconfig(config, "audio_config") is not None

    # task family + which layout view to use (decoder LM / encoder-classifier / enc-dec / VLM)
    am.family, am.head_class = _detect_family(mt, config)
    am.num_labels = _cfg(config, "num_labels", "num_classes")
    if getattr(config, "is_encoder_decoder", False):
        am.view = "enc_dec"
    elif am.family in ("image_classification", "audio_classification", "masked_lm"):
        am.view = "encoder"
    else:
        am.view = "decoder"  # multimodal is set later once towers are detected

    am.hidden_size = _cfg(tcfg, "hidden_size", "d_model", "n_embd", "dim")
    am.vocab_size = _cfg(tcfg, "vocab_size")
    am.num_layers = _cfg(tcfg, "num_hidden_layers", "num_layers", "n_layer", "n_layers")
    am.num_attention_heads = _cfg(tcfg, "num_attention_heads", "n_head", "num_heads")
    am.num_kv_heads = _cfg(tcfg, "num_key_value_heads", "num_kv_heads", default=am.num_attention_heads)
    am.head_dim = _cfg(tcfg, "head_dim")
    # hierarchical vision models (swin, dinat, hiera, ...) carry per-stage tuples; only the
    # plain int case has a well-defined single head_dim.
    if am.head_dim is None and isinstance(am.hidden_size, int) and isinstance(am.num_attention_heads, int):
        am.head_dim = am.hidden_size // am.num_attention_heads
    am.intermediate_size = _cfg(tcfg, "intermediate_size", "ffn_dim", "d_ff")
    am.max_position_embeddings = _cfg(tcfg, "max_position_embeddings", "n_positions")
    am.sliding_window = _cfg(tcfg, "sliding_window")
    am.num_experts = _cfg(tcfg, "num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
    am.experts_per_token = _cfg(tcfg, "num_experts_per_tok", "moe_topk", "num_experts_per_token", "top_k")
    am.tie_word_embeddings = _cfg(config, "tie_word_embeddings")
    am.attn_variant = _attn_variant(tcfg)
    am.hidden_act = _cfg(tcfg, "hidden_act", "activation_function", "hidden_activation")
    rt = _cfg(tcfg, "rope_theta")
    am.rope_theta = rt if isinstance(rt, (int, float)) else None
    lts = _cfg(tcfg, "layer_types")
    if isinstance(lts, (list, tuple)) and lts and all(isinstance(x, str) for x in lts):
        am.layer_types = list(lts)
    am.checkpoint = _example_checkpoint(model, mt)
    am.tokens = _example_tokens(am.checkpoint)
    # sparse/compressed attention components, keyed by distinct attention layer_type
    compress_rates = _cfg(tcfg, "compress_rates")
    if isinstance(compress_rates, dict) and am.layer_types:
        index_topk = _cfg(tcfg, "index_topk")
        from .masks import concat_compressed_mask

        for lt in dict.fromkeys(am.layer_types):
            if lt in compress_rates:
                comp = {"type": lt, "compress_rate": compress_rates[lt]}
                if "sparse" in lt and index_topk:
                    comp["index_topk"] = index_topk
                # the real mask passed to attention: sliding K/V cache ⊕ compressed cache.
                # display m is capped so the figure is legible (true rate kept in the label).
                m = min(int(compress_rates[lt]), 8)
                grid, split, n_comp = concat_compressed_mask(16, m)
                comp.update(display_m=m, mask=grid, mask_split=split, n_comp=n_comp)
                am.sparse_components.append(comp)

    module_classes: set[str] = set()
    layer_classes: list[str] = []

    try:
        model_obj = _build_meta_model(config)
        am.top_class = type(model_obj).__name__
        counts: dict[str, int] = {}
        for _, mod in model_obj.named_modules():
            cn = type(mod).__name__
            counts[cn] = counts.get(cn, 0) + 1
            module_classes.add(cn)
        am.module_kinds = dict(sorted(counts.items(), key=lambda kv: -kv[1])[:25])

        name, layers = _find_layer_list(model_obj, am.num_layers)

        # multimodal towers (vision/audio encoders + projector + LLM)
        towers = _detect_towers(model_obj, config)
        if (has_vision_cfg or has_audio_cfg) and any(t["role"] in ("vision", "audio") for t in towers):
            am.is_multimodal = True
            am.view = "multimodal"
            am.towers = towers
            am.modal_inputs = ["input_ids"]
            if any(t["role"] == "vision" for t in towers):
                am.modal_inputs.append("pixel_values")
            if any(t["role"] == "audio" for t in towers):
                am.modal_inputs.append("input_features")
            am.auto_classes = _auto_classes(model)
            # prefix-LM image attention (bidirectional over image tokens, e.g. PaliGemma/Gemma3)
            import os as _os

            from .discover import models_root as _mr

            _d = _os.path.join(_mr(), model)
            for _f in sorted(_os.listdir(_d)) if _os.path.isdir(_d) else []:
                if _f.startswith(("modeling_", "modular_")) and _f.endswith(".py"):
                    try:
                        _src = open(_os.path.join(_d, _f), encoding="utf-8").read()
                    except Exception:
                        continue
                    if "token_type_ids" in _src or "bidirectional" in _src.lower():
                        am.image_bidirectional = True
                        break
            # for the LLM block, restrict the layer search to the language_model subtree
            llm = next((getattr(model_obj, t["name"]) for t in towers if t["role"] == "text"), None)
            if llm is not None:
                _, llm_layers = _find_layer_list(llm, am.num_layers)
                if llm_layers is not None:
                    layers = llm_layers

        # norm/positional must be known before block extraction (rope detection uses them)
        am.norm_type = _norm_type(module_classes)
        am.positional = _positional(tcfg, module_classes)
        if layers is not None:
            layer_classes = [type(layer).__name__ for layer in layers]
            if not am.num_layers:
                am.num_layers = len(layers)
            # decompose a representative decoder layer into block internals (full view).
            # MoE models often keep the first few layers dense (e.g. DeepSeek); prefer a
            # layer that actually contains the MoE block so the diagram is representative.
            try:
                import torch.nn as nn

                def _score(layer):
                    has_attn = any(
                        "attention" in type(c).__name__.lower() and any(isinstance(g, nn.Linear) for g in c.children())
                        for c in layer.children()
                    )
                    has_moe = any(_MOE_RE.search(type(c).__name__) for c in layer.modules())
                    # prefer a layer with real attention; among those prefer one with MoE
                    return (1 if has_attn else 0, 1 if (has_moe and am.num_experts) else 0)

                rep = layers[0]
                if len(layers) > 1:
                    rep = max(layers, key=_score)
                # hierarchical/staged models (Swin/Donut/NAT): the "layer" is a Stage with no
                # direct attention -> descend to the innermost module that has real attention
                if _attn_child(rep) is None:
                    deep = next(
                        (m for m in rep.modules() if _attn_child(m) is not None and m is not rep),
                        None,
                    )
                    if deep is None:  # search the whole model
                        deep = next((m for m in model_obj.modules() if _attn_child(m) is not None), None)
                    if deep is not None:
                        rep = deep
                        layers = [rep]
                am.block = _extract_block(rep, tcfg, am)
                # record any structurally different layer classes (hybrid) not shown in detail
                kinds = {_kind_for_layer(type(layer).__name__) for layer in layers}
                am.alt_blocks = sorted(k for k in kinds if k != "attention")

                # FULL recursive decomposition of each DISTINCT attention variant (the real
                # module tree -- e.g. DeepSeek-V4 HCA vs CSA-with-Indexer), so all blocks show.
                variants, seen = [], set()
                for i, layer in enumerate(layers):
                    at = _block_core(layer)
                    if at is None:
                        continue
                    lt = getattr(at, "layer_type", None) or (
                        am.layer_types[i] if i < len(am.layer_types) else type(at).__name__
                    )
                    if lt in seen:
                        continue
                    seen.add(lt)
                    variants.append({"type": lt, "cls": type(at).__name__, "children": _decompose(at, 2)})
                    if len(variants) >= 3:
                        break
                am.attention_variants = variants
                # decompose the FFN/MoE of the representative layer too
                for n, c in rep.named_children():
                    if n in ("mlp", "feed_forward", "ffn", "block_sparse_moe", "moe") or type(
                        c
                    ).__name__.lower().endswith(("mlp", "moe", "moeblock", "sparsemoeblock")):
                        am.mlp_tree = {"cls": type(c).__name__, "children": _decompose(c, 2)}
                        # annotate router/experts nodes with counts (fused experts have no
                        # per-expert submodules on meta, so surface the count + dims here)
                        ed = _cfg(tcfg, "moe_intermediate_size", "expert_intermediate_size", "intermediate_size")
                        _annotate_moe(am.mlp_tree["children"], am, ed)
                        break

                # distinct FFN variants per config.mlp_layer_types (moe / hash_moe / dense)
                mlts = _cfg(tcfg, "mlp_layer_types")
                if isinstance(mlts, (list, tuple)) and mlts and all(isinstance(x, str) for x in mlts):
                    am.mlp_layer_types = list(mlts)
                    ed = _cfg(tcfg, "moe_intermediate_size", "expert_intermediate_size", "intermediate_size")
                    mseen = set()
                    for i, layer in enumerate(layers):
                        if i >= len(am.mlp_layer_types):
                            break
                        mt2 = am.mlp_layer_types[i]
                        if mt2 in mseen:
                            continue
                        mc = _mlp_child(layer)
                        if mc is None:
                            continue
                        mseen.add(mt2)
                        ch = _decompose(mc, 2)
                        _annotate_moe(ch, am, ed)
                        am.mlp_variants.append({"type": mt2, "cls": type(mc).__name__, "children": ch})
                        if len(am.mlp_variants) >= 3:
                            break
            except Exception:
                am.block = None

        # Universal fallback: any model without a recognized attention/SSM/MLP transformer block
        # (conv backbones, audio codecs, FFT mixers, detectors, ...) still gets a real top-down
        # render from its actual (collapsed) module tree instead of a config-only schematic.
        no_block = am.block is None or (am.block.attention is None and am.block.mixer is None and am.block.mlp is None)
        generic = no_block and not am.attention_variants
        try:
            am.is_pipeline = len(_stage_children(model_obj)) >= 3
        except Exception:
            am.is_pipeline = False
        # A model gets the data-flow view when it has no single transformer block (conv backbone,
        # codec, ...) OR when it is a multi-stage pipeline (BLT: patcher → encoder → transformer →
        # decoder), where the data path is the story rather than one attention block.
        if (generic or am.is_pipeline) and not am.is_multimodal:
            try:
                am.generic_tree = am.generic_tree or _module_tree(model_obj, depth=4)
            except Exception:
                am.generic_tree = []
            try:
                cap = _capture_flow(model_obj, config, model)
                if cap:
                    am.flow_input, am.flow, am.flow_output = cap
            except Exception:
                am.flow = []
    except Exception as e:
        am.status = "config-only"
        am.error = f"meta build failed: {e!r}"[:300]

    # MoE / decoder kind
    moe_modules = {c for c in module_classes if _MOE_RE.search(c)}
    am.is_moe = bool(moe_modules) or (am.num_experts is not None and am.num_experts > 1)
    has_mamba = any(_MAMBA_RE.search(c) for c in module_classes)
    has_linear = any(_LINEAR_ATTN_RE.search(c) for c in module_classes)

    if am.norm_type is None:
        am.norm_type = _norm_type(module_classes)
    if am.positional is None:
        am.positional = _positional(tcfg, module_classes)

    # layer blocks: prefer config.layer_types (the real per-layer schedule) when present,
    # since hybrid models (qwen3_next: linear+full, gemma3: sliding+full) reuse ONE layer
    # class so the module tree alone hides the mix.
    blocks: list[LayerBlock] = []
    if am.layer_types:
        for lt, n in _collapse(am.layer_types):
            kind = layer_type_kind(lt)
            blocks.append(
                LayerBlock(
                    layer_class=lt,
                    count=n,
                    kind=kind,
                    attn_variant=(am.attn_variant if kind == "attention" else None),
                    mlp_kind="moe" if (am.is_moe and kind in ("attention", "linear_attention")) else "dense",
                    norm=am.norm_type,
                )
            )
    elif layer_classes:
        for cls_name, n in _collapse(layer_classes):
            kind = _kind_for_layer(cls_name)
            blocks.append(
                LayerBlock(
                    layer_class=cls_name,
                    count=n,
                    kind=kind,
                    attn_variant=am.attn_variant if kind == "attention" else None,
                    mlp_kind="moe" if (am.is_moe and kind == "attention") else "dense",
                    norm=am.norm_type,
                )
            )
    elif am.num_layers:
        # config-only fallback: one synthetic block. With no module tree to walk, infer the
        # kind from the synthetic layer name (e.g. "MambaDecoderLayer" -> mamba) and config.
        synth_name = f"{am.config_class.replace('Config', '') if am.config_class else mt}DecoderLayer"
        kind = "mamba" if has_mamba else ("linear_attention" if has_linear else _kind_for_layer(synth_name))
        blocks.append(
            LayerBlock(
                layer_class=synth_name,
                count=am.num_layers,
                kind=kind,
                attn_variant=am.attn_variant,
                mlp_kind="moe" if am.is_moe else "dense",
                norm=am.norm_type,
            )
        )
    am.layer_blocks = blocks

    # decoder kind + summary
    distinct_kinds = {b.kind for b in blocks}
    if len(distinct_kinds) > 1:
        am.decoder_kind = "hybrid"
    elif has_mamba and not any(b.kind == "attention" for b in blocks):
        am.decoder_kind = "ssm"
    elif am.is_moe:
        am.decoder_kind = "sparse-moe"
    elif distinct_kinds and distinct_kinds != {"attention"}:
        am.decoder_kind = "other"
    else:
        am.decoder_kind = "dense"

    if blocks:
        # aggregate by kind-label totals (not per-run) so the summary stays short
        totals: dict[str, int] = {}
        for b in blocks:
            label = _pretty_kind(b)
            totals[label] = totals.get(label, 0) + b.count
        am.layer_summary = " + ".join(f"{n}× {label}" for label, n in totals.items())

    # attention-mask patterns per distinct layer type (offline, real masking_utils)
    if am.layer_types:
        try:
            from .masks import attention_pattern_grids

            am.attn_patterns = attention_pattern_grids(tcfg, am.layer_types, seq=24)
        except Exception:
            am.attn_patterns = {}

    return am


_LT_SHORT = {
    "full_attention": "full",
    "sliding_attention": "sliding",
    "chunked_attention": "chunked",
    "compressed_sparse_attention": "compressed-sparse",
    "heavily_compressed_attention": "heavily-compressed",
    "linear_attention": "linear",
}


def _short_layer_type(name: str) -> str:
    if name in _LT_SHORT:
        return _LT_SHORT[name]
    n = name.lower()
    for key, short in _LT_SHORT.items():
        if n == key:
            return short
    return name.replace("_attention", "").replace("_", " ")


def _pretty_kind(b: LayerBlock) -> str:
    # when the layer_class is actually a layer_type (sliding/full/...), show that label
    is_layer_type = b.layer_class.endswith("attention") or b.layer_class in _LT_SHORT
    if b.kind == "attention":
        base = _short_layer_type(b.layer_class) if is_layer_type else (b.attn_variant or "Attention")
        return base + (" +MoE" if b.mlp_kind == "moe" else "")
    if b.kind == "linear_attention":
        base = _short_layer_type(b.layer_class) if is_layer_type else "linear-attn"
        return base + (" +MoE" if b.mlp_kind == "moe" else "")
    return {"mamba": "Mamba", "recurrent": "Recurrent"}.get(b.kind, b.layer_class)
