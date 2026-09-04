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
Processor-variant facet extraction, the `blocks_facets.py` method applied to `processing_*.py`.

`blocks_facets.py` asks "what does this block's `forward` do, and whose `forward` could it
inherit". The processor equivalent is not `forward` but `__call__`, and on this branch that
question has a sharp answer, because `ProcessorMixin.__call__` is already a complete generic
pipeline:

    prepare_inputs_layout -> validate_inputs -> _merge_kwargs(valid_processor_kwargs)
      -> _process_{images,videos,audio}  (each calls replace_{image,video,audio}_token per item)
      -> get_text_with_replacements -> tokenizer -> create_mm_token_type_ids -> BatchFeature

So a processor is fully described by *which sub-processors it composes* and *what its
`replace_*_token` hooks return*. Everything else the mixin does for it. That is why the tier line
falls where it does:

- **tier 1 is the token-stream contract**: the composition, whether `__call__` is the inherited
  generic one or a hand-written override, how each modality's placeholder count is computed, and
  what shape the replacement string takes. These decide what ids and tensors a given
  `(text, media)` pair maps to, so they decide identity and therefore the tag.
- **tier 2 is everything reached by another entry point, or that only reshapes and validates what
  the caller already passed**: chat templating, `decode`/`post_process_*`, kwargs `_defaults`, the
  `_get_num_multimodal_tokens` planning API, and the small pre-flight hooks
  (`prepare_inputs_layout`, `validate_inputs`, `create_mm_token_type_ids`). The test applied to
  each is "would differing here force a different `__call__`?" -- for all of these the answer is
  no, because each is its own overridable hook, so two processors that differ only there still
  share one `__call__`.

That line is measured, not asserted. Tier 1 alone yields 53 variants over 148 processors, 30 of
them singletons. Folding all of tier 2 into the tag yields 111 variants and 88 singletons -- 60% of
the library would become a variant of one, without a single processor emitting a different token.
The worst single offender is `kwargs_defaults` on its own (53 -> 76), which is why the `_defaults`
groups are reported next to a variant and never inside it. A facet that shatters the table while
changing no behaviour is the definition of tier 2 here; `_selfcheck` pins that this stays true.

The same lesson as the blocks registry, in a different costume: **facets nominate, source
decides.** Facet equality is a lossy proxy, and measurably so: the single largest variant holds 26
processors and contains 25 *distinct* implementations, because "composes an image processor and a
tokenizer, hand-writes `__call__`, expands nothing" describes every pre-multimodal vision-text
processor in the library without saying what any of them does. So `sources_match` compares the
canonicalised `__call__` and hooks on top of the facets, and a facet-equal / code-different pair is
reported as a near miss (572 of them) rather than a duplicate (15).

Vocabularies are read out of the library rather than invented, with `ast`, never by importing
(`processing_utils` pulls in torch):

- the kwargs group names come from `_merge_kwargs` itself, which is the only thing that decides
  whether a `_defaults` entry is live -- and they have to be gathered from *two* code paths there,
  which is exactly the trap that reading source instead of hardcoding is meant to avoid; see
  `kwargs_group_names` and `dead_default_groups`;
- the overridable-hook set is just `ProcessorMixin`'s own method names, so "which hooks does this
  processor override" cannot drift from what the mixin actually offers;
- release dates come from `blocks_facets.build_date_data()`, unchanged, so canonical owners are
  ordered the same way in both registries.

Stdlib only, for the same reason as `blocks_facets`: this must be importable from a
repo-consistency checker without dragging torch into `make check-repo`.
"""

import ast
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = REPO_ROOT / "src" / "transformers" / "models"
PROCESSING_UTILS = REPO_ROOT / "src" / "transformers" / "processing_utils.py"

if str(Path(__file__).parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent))

# Reused verbatim rather than reimplemented, so both registries agree on history and on the
# modular DAG. `generates_modeling` is named for modeling files but only looks for the converter's
# banner in the first 2 KB of whatever path it is handed, so it works unchanged on a
# `processing_*.py` -- 38 of the 148 are generated, and a finding on a generated file has to be
# applied to the modular that produced it.
from blocks_facets import (  # noqa: E402, I001
    ancestors,
    build_date_data,
    canonical_source,
    generates_modeling as is_generated_file,
)


# --------------------------------------------------------------------------------------------------
# Axis order == the "processors format". Ordered by override cost, expensive first, mirroring
# `blocks_facets.ATTENTION_AXES`: agreeing on the expensive axes first means picking a parent by
# longest common prefix leaves any divergence in the cheap tail.
#
# Not fitted the way the block axes were -- there is no equivalent ground truth to fit against,
# because processors are almost never subclassed across models in a modular file (91 of the 148
# processor-owning models have a `modular_*.py`, but processor-to-processor overrides are rare
# enough that a per-axis median would be fitting noise). Ordered by argument instead:
#   composition       -- changing it changes the constructor, the saved config and every branch of
#                        `__call__` that fires. Nothing is more expensive.
#   call_style        -- a hand-written `__call__` is unbounded; it can invalidate every axis below.
#   image/video/audio -- the placeholder arithmetic, one axis per modality. Kept separate rather
#                        than joined into one string because a `|`-joined tag has to read without a
#                        legend, and `count_from_grid_thw+count_from_subprocessor` does not say
#                        which modality got which.
#   replacement_shape -- cheapest: the same count, wrapped differently.
PROCESSOR_AXES = (
    "composition",
    "call_style",
    "image_expansion",
    "video_expansion",
    "audio_expansion",
    "replacement_shape",
)

# The three modalities the mixin's `__call__` expands placeholders for, in the order it calls them.
MODALITIES = ("image", "video", "audio")

# A parameter name is a composed sub-processor if it contains one of these. This mirrors
# `ProcessorMixin.get_attributes`, which does the same substring test against the auto-processor
# mapping -- so `qformer_tokenizer`, `protein_tokenizer` and `char_tokenizer` all count, as the
# library itself counts them.
#
# One deliberate divergence: `get_attributes` explicitly skips `audio_tokenizer` (it is not
# saved as an attribute), but for the census question "what does this processor compose" a
# discrete audio codec plainly is a composed sub-processor, so it is kept here.
SUBPROCESSOR_HINTS = ("tokenizer", "image_processor", "video_processor", "feature_extractor", "audio_processor")


# --------------------------------------------------------------------------------------------------
# Vocabularies read out of the library source (never imported: processing_utils pulls in torch)
# --------------------------------------------------------------------------------------------------
@cache
def kwargs_group_names() -> frozenset[str]:
    """
    Every kwarg group name `_merge_kwargs` actually looks up, so a `_defaults` key outside this set
    is dead config (see `dead_default_groups`).

    Two code paths have to be read, not one, and getting this wrong is exactly the kind of error
    reading the vocabulary from source is supposed to prevent. Four names come from the
    `output_kwargs` literal, which `_merge_kwargs` then iterates with
    `_defaults.get(modality, {})`. The fifth, `common_kwargs`, never appears in that literal -- it
    is fetched by a separate hardcoded `.get("common_kwargs", {})` and merged into all four. Taking
    the literal alone declared `common_kwargs` dead in 32 models that use it correctly.
    """
    tree = ast.parse(PROCESSING_UTILS.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "_merge_kwargs":
            continue
        for sub in ast.walk(node):
            targets = sub.targets if isinstance(sub, ast.Assign) else []
            if any(isinstance(t, ast.Name) and t.id == "output_kwargs" for t in targets) and isinstance(
                sub.value, ast.Dict
            ):
                names.update(k.value for k in sub.value.keys if isinstance(k, ast.Constant))
            # `<something>._defaults.get("common_kwargs", {})`: a group resolved outside the loop.
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "get"
                and isinstance(sub.func.value, ast.Attribute)
                and sub.func.value.attr == "_defaults"
                and sub.args
                and isinstance(sub.args[0], ast.Constant)
                and isinstance(sub.args[0].value, str)
            ):
                names.add(sub.args[0].value)
    return frozenset(names)


@cache
def mixin_hooks() -> frozenset[str]:
    """
    Every method name on `ProcessorMixin`, i.e. the set a processor can override.

    Derived rather than listed so that "which hooks does this processor override" cannot drift as
    the mixin grows. The generic `__call__` landing in the mixin is the whole reason this registry
    is interesting, and it arrived by someone adding a method here.
    """
    tree = ast.parse(PROCESSING_UTILS.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ProcessorMixin":
            return frozenset(s.name for s in node.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)))
    return frozenset()


# --------------------------------------------------------------------------------------------------
# Processor classification
# --------------------------------------------------------------------------------------------------
def is_processor_class(name: str) -> bool:
    """
    Whether a class declared in a `processing_*.py` is the model's processor.

    Two pieces of naming variance have to be absorbed here or the census is wrong in both
    directions. `Florence2PostProcessor` is a detection-output decoder that happens to end in
    `Processor`, and counting it gave florence2 two processors. `Wav2Vec2ProcessorWithLM` is a real
    processor whose name does not end in `Processor` at all, and missing it dropped
    `wav2vec2_with_lm` from the registry entirely.

    Note what is deliberately *not* excluded: `GlmImageProcessor` and `LlavaNextVideoProcessor` end
    in `ImageProcessor`/`VideoProcessor` but are the top-level processors of `glm_image` and
    `llava_next_video`. A blanket suffix exclusion loses eight real processors to their models'
    names.
    """
    if "Kwargs" in name or name.endswith("PostProcessor"):
        return False
    return name.endswith("Processor") or name.endswith("ProcessorWithLM")


# --------------------------------------------------------------------------------------------------
# Placeholder expansion: the axis with the most genuine variety
# --------------------------------------------------------------------------------------------------
# How the number of placeholder tokens for one media item is computed. Ordered most-specific
# first, and every value names its *source of truth* rather than its arithmetic, because the
# arithmetic is where models differ cosmetically and the source is where they differ in substance.
_EXPANSION_RULES = (
    # The sub-processor already returned the count. The cleanest possible arrangement -- the
    # component that did the resizing is the one that knows how many tokens it produced -- and the
    # target every other value on this axis should eventually collapse into.
    ("count_from_subprocessor", ('num_image_tokens"]', 'num_video_tokens"]', 'num_audio_tokens"]')),
    # Qwen-style: the image processor returns a `(t, h, w)` grid and the processor divides by
    # `spatial_merge_size ** 2`. The single most common expansion in the library.
    ("count_from_grid_thw", ("grid_thw",)),
    # Idefics/SmolVLM tiling: `rows` x `cols` sub-images, each worth `image_seq_len` tokens, with
    # per-tile marker tokens interleaved.
    ("count_from_tile_grid", ('"rows"', '"cols"', "num_rows", "num_cols", "image_rows", "image_cols")),
    # Pixtral-style: original (height, width) per image, floor-divided by an effective patch size.
    ("count_from_image_sizes", ("image_sizes",)),
    # A count of tiles/crops/patches the image processor chose (`num_crops`, `num_patches`), each
    # worth a fixed-size sequence.
    ("count_from_patch_count", ("num_patches", "num_crops", "num_tiles", "tokens_per_tile", "batch_num_images")),
    # Llava-style: read the *processed* tensor's own shape and divide by `patch_size`. Fragile --
    # it re-derives downstream of the resize that already knew the answer.
    ("count_from_pixel_shape", ("get_image_size", "pixel_values", ".shape")),
    # A constant from the processor config: `image_seq_length`, `num_soft_tokens`, a resampler's
    # query count. Independent of the image.
    (
        "count_fixed",
        (
            "image_seq_length",
            "image_seq_len",
            "num_soft_tokens",
            "num_query_tokens",
            "seq_length",
            "num_image_tokens",
            "num_audio_tokens",
        ),
    ),
)

# Audio counts come from frame arithmetic, and the mel/frame names have to be checked before the
# generic `.shape` rule or every audio processor reads as `count_from_pixel_shape`.
_AUDIO_FRAME_NEEDLES = (
    "num_mel",
    "feature_attention_mask",
    "audio_embed_sizes",
    "input_features",
    "audio_lengths",
    "audio_length",
)


def _returns_bare_token(node: ast.FunctionDef) -> bool:
    """
    True when the hook returns a single `self.*_token` with no repetition.

    Detected on the syntax rather than by looking for a `*` in the text, because every hook's
    signature carries `**kwargs` and a textual test for `*` therefore never fires. mllama is the
    only processor in this state: it has an `image_token` but attends to images by cross-attention,
    so it must emit exactly one token and expand nothing. That is a different behaviour from having
    no hook at all, and collapsing the two would put a cross-attention model in the same variant as
    a text-only processor.
    """
    returns = [n for n in ast.walk(node) if isinstance(n, ast.Return) and n.value is not None]
    if not returns:
        return False
    return all(
        isinstance(r.value, ast.Attribute)
        and isinstance(r.value.value, ast.Name)
        and r.value.value.id == "self"
        and r.value.attr.endswith("_token")
        for r in returns
    )


def expansion_facet(node: ast.FunctionDef, body: str, modality: str) -> str:
    """Classify how one `replace_<modality>_token` computes its token count."""
    if _returns_bare_token(node):
        return "single_token"
    if modality == "audio" and any(n in body for n in _AUDIO_FRAME_NEEDLES):
        return "count_from_audio_frames"
    for value, needles in _EXPANSION_RULES:
        if any(n in body for n in needles):
            return value
    # Named rather than "unknown": it groups honestly instead of merging unrelated processors.
    return "count_custom"


# The shape of the replacement string, independent of the count. Tier 1 because it changes the
# emitted id sequence, but the cheapest tier-1 axis: same count, different scaffolding.
_ROW_SEPARATED_RE = re.compile(r"break_token|_row|separator|newline")
_BOUNDARY_RE = re.compile(r"start_token|bos_token|boi_token|vision_start|_begin_token")


def replacement_shape(body: str) -> str:
    if _ROW_SEPARATED_RE.search(body):
        return "row_separated"
    if _BOUNDARY_RE.search(body):
        return "boundary_wrapped"
    return "bare_repeat"


# --------------------------------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------------------------------
# The methods whose source is compared when a facet match has to be confirmed. `__call__` is the
# contract; the hooks are what the generic `__call__` calls into; the rest are the pre/post hooks
# that a would-be parent must also match before its `__call__` is genuinely reusable.
KEY_METHODS = (
    "__call__",
    "replace_image_token",
    "replace_video_token",
    "replace_audio_token",
    "prepare_inputs_layout",
    "validate_inputs",
    "create_mm_token_type_ids",
    "model_input_names",
    "unused_input_names",
)

# Tier-2 conveniences, grouped so the facet reads as a set rather than a bitfield.
_DECODE_METHODS = (
    "batch_decode",
    "decode",
    "post_process_image_text_to_text",
    "post_process_multimodal_output",
    "parse_response",
)


@dataclass
class Processor:
    """One processor class found in one `processing_*.py`."""

    model: str
    path: Path
    class_name: str
    tier1: dict = field(default_factory=dict)
    tier2: dict = field(default_factory=dict)
    lineno: int = 0
    subprocessors: tuple[str, ...] = ()
    # Canonicalised source of each `KEY_METHODS` entry the class actually defines. Facets are a
    # lossy summary -- 41 processors share `no_image_expansion|...|no_replacement` while doing
    # entirely different things in `__call__` -- so facets only nominate candidates and this is
    # what confirms a match.
    sources: dict = field(default_factory=dict)
    generated: bool = False

    @property
    def variant(self) -> str:
        """The tag: tier-1 values in axis order. Identical variant == same token-stream contract."""
        return "|".join(str(self.tier1.get(axis, "?")) for axis in PROCESSOR_AXES)

    @property
    def inherits_generic_call(self) -> bool:
        return self.tier1.get("call_style") == "inherited_generic_call"

    def tier2_delta(self, other: "Processor") -> dict[str, tuple[str, str]]:
        """Tier-2 facets that differ from `other` -- the rewrite hint on a suggestion."""
        return {k: (v, other.tier2.get(k, "?")) for k, v in self.tier2.items() if other.tier2.get(k, "?") != v}


def _canonical(node: ast.FunctionDef, model: str) -> str | None:
    """`node` canonicalised for cross-model comparison, with the model's own name neutralised."""
    try:
        body = canonical_source(node)
    except (SyntaxError, ValueError):
        return None
    # `qwen2_vl` -> `Qwen2VL`, `Qwen2_VL`, `QWEN2VL`. Strip the squashed form case-insensitively
    # rather than trying to reproduce each model's exact capitalisation.
    return re.sub(rf"\b{re.escape(model.replace('_', ''))}", "X", body, flags=re.IGNORECASE)


def _kwargs_defaults(tree: ast.Module) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    """
    `({kwargs class: declared group names}, all group names)` for one processing module.

    Read from every `*Kwargs` class in the file, not just the one wired to
    `valid_processor_kwargs`, because a file can declare several and only one is live.
    """
    per_class: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or "Kwargs" not in node.name:
            continue
        for stmt in node.body:
            targets = stmt.targets if isinstance(stmt, ast.Assign) else []
            if not any(isinstance(t, ast.Name) and t.id == "_defaults" for t in targets):
                continue
            if isinstance(stmt.value, ast.Dict):
                per_class[node.name] = tuple(
                    sorted(
                        k.value for k in stmt.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)
                    )
                )
    everything = tuple(sorted({g for groups in per_class.values() for g in groups}))
    return per_class, everything


def scan_file(path: Path, model: str) -> list[Processor]:
    """Extract every processor class from one `processing_*.py`."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    generated = is_generated_file(path)
    hooks = mixin_hooks()
    _, default_groups = _kwargs_defaults(tree)

    # `valid_processor_kwargs = XProcessorKwargs` at class level says the processor declares its
    # own kwargs schema; absent, it inherits the mixin's bare `ProcessingKwargs`.
    found: list[Processor] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not is_processor_class(node.name):
            continue

        methods = {s.name: s for s in node.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))}
        bodies = {name: "\n".join(lines[s.lineno - 1 : s.end_lineno]) for name, s in methods.items()}
        declares_kwargs = any(
            isinstance(s, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "valid_processor_kwargs" for t in s.targets)
            for s in node.body
        )

        init = methods.get("__init__")
        params = [a.arg for a in init.args.args[1:]] if isinstance(init, ast.FunctionDef) else []
        # Sorted, not source-ordered. 21 processors compose the same three sub-processors and
        # declare them in three different orders (`image+tokenizer+video`,
        # `image+video+tokenizer`, `video+image+tokenizer`); source order would have split one
        # variant into three on pure cosmetics.
        subs = tuple(sorted(p for p in params if any(h in p for h in SUBPROCESSOR_HINTS)))

        tier1 = {
            "composition": "+".join(subs) or "no_subprocessors",
            # The migration axis. 60 of 148 already inherit the mixin's generic pipeline; the other
            # 88 hand-write a `__call__` that mostly reimplements it.
            "call_style": "custom_call" if "__call__" in methods else "inherited_generic_call",
        }
        shapes: set[str] = set()
        for modality in MODALITIES:
            hook = f"replace_{modality}_token"
            if hook not in methods:
                tier1[f"{modality}_expansion"] = f"no_{modality}_expansion"
                continue
            body = bodies[hook]
            # A hook that delegates to a private helper says nothing on its own, so inline the
            # class's own methods it calls before classifying -- smolvlm's whole tile-grid
            # arithmetic lives in a module helper reached that way.
            for called in set(re.findall(r"self\.(\w+)\(", body)):
                if called in bodies and called != hook:
                    body += "\n" + bodies[called]
            tier1[f"{modality}_expansion"] = expansion_facet(methods[hook], body, modality)
            shapes.add(replacement_shape(body))
        tier1["replacement_shape"] = "+".join(sorted(shapes)) or "no_replacement"

        overridden = sorted(set(methods) & hooks)
        tier2 = {
            # A data artifact, not a code path: it changes what `apply_chat_template` renders, never
            # what `__call__` maps a given input to. No chat template ships in this repo -- all 0 of
            # them live on the Hub -- so the only thing observable statically is whether the
            # constructor accepts one.
            "chat_template": "chat_template_arg" if "chat_template" in params else "no_chat_template_arg",
            "kwargs_schema": "own_processing_kwargs" if declares_kwargs else "inherited_processing_kwargs",
            "kwargs_defaults": "+".join(default_groups) or "no_kwargs_defaults",
            "input_layout": "custom_prepare_inputs_layout"
            if "prepare_inputs_layout" in methods
            else "generic_prepare_inputs_layout",
            "input_validation": "custom_validate_inputs"
            if "validate_inputs" in methods
            else "generic_validate_inputs",
            "token_type_ids": "custom_mm_token_type_ids"
            if "create_mm_token_type_ids" in methods
            else "generic_mm_token_type_ids",
            # The vLLM-facing planning API: predicts token counts without running the processor.
            "num_tokens_api": "declares_get_num_multimodal_tokens"
            if "_get_num_multimodal_tokens" in methods
            else "no_num_tokens_api",
            "decode_surface": "+".join(m for m in _DECODE_METHODS if m in methods) or "no_decode_overrides",
            "overridden_mixin_hooks": str(len(overridden)),
        }

        sources = {
            name: _canonical(methods[name], model)
            for name in KEY_METHODS
            if isinstance(methods.get(name), ast.FunctionDef)
        }
        found.append(
            Processor(
                model=model,
                path=path,
                class_name=node.name,
                tier1=tier1,
                tier2=tier2,
                lineno=node.lineno,
                subprocessors=subs,
                sources={k: v for k, v in sources.items() if v is not None},
                generated=generated,
            )
        )
    return found


def scan_repo(models_root: Path = MODELS_ROOT) -> list[Processor]:
    """Scan every `processing_*.py` under `models_root`."""
    found: list[Processor] = []
    for model_dir in sorted(p for p in models_root.iterdir() if p.is_dir()):
        for path in sorted(model_dir.glob("processing_*.py")):
            found.extend(scan_file(path, model_dir.name))
    return found


# --------------------------------------------------------------------------------------------------
# `# Copied from` markers in processing files
# --------------------------------------------------------------------------------------------------
_COPIED_FROM_RE = re.compile(r"#\s*copied from transformers\.models\.(\w+)\.\w+\.(\w+)", re.IGNORECASE)


@cache
def copied_from_processor_sources(models_root: Path = MODELS_ROOT) -> dict[tuple[str, str], str]:
    """
    `{(model, symbol): source model}` for every `# Copied from` marker in a `processing_*.py`.

    `blocks_facets.copied_from_sources` globs `modeling_*.py` only, so it cannot answer this; the
    scan is duplicated here rather than generalising that function, to keep the blocks registry's
    behaviour bit-identical.

    Unlike in the blocks registry a marker is *not* treated as an exemption. `# Copied from` is the
    legacy reuse mechanism being removed, so a marked duplicate is a duplicate whose source is
    already documented -- easier to act on, not out of scope. There are only 11 markers across 6
    processing files anyway, so essentially all processor duplication here is unmanaged.
    """
    sources: dict[tuple[str, str], str] = {}
    for model_dir in sorted(p for p in models_root.iterdir() if p.is_dir()):
        for path in sorted(model_dir.glob("processing_*.py")):
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
                declared = re.match(r"(?:class|def)\s+(\w+)", line.strip())
                if declared:
                    if pending:
                        sources[(model_dir.name, declared.group(1))] = pending
                    pending = None
    return sources


# --------------------------------------------------------------------------------------------------
# Lint: `_defaults` groups that no code reads
# --------------------------------------------------------------------------------------------------
def dead_default_groups(models_root: Path = MODELS_ROOT) -> list[tuple[str, str, tuple[str, ...]]]:
    """
    `_defaults` entries keyed by a name `_merge_kwargs` never looks up, i.e. silently dead config.

    This falls out of reading the vocabulary from source instead of hardcoding it. `_merge_kwargs`
    resolves `ModelProcessorKwargs._defaults.get(modality, {})` for exactly the five names in
    `kwargs_group_names()`, so `{"image_kwargs": ...}` (singular) is never read at all -- the
    defaults look configured and do nothing.
    """
    valid = kwargs_group_names()
    findings: list[tuple[str, str, tuple[str, ...]]] = []
    for model_dir in sorted(p for p in models_root.iterdir() if p.is_dir()):
        for path in sorted(model_dir.glob("processing_*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            per_class, _ = _kwargs_defaults(tree)
            for cls, groups in per_class.items():
                dead = tuple(g for g in groups if g not in valid)
                if dead:
                    findings.append((model_dir.name, cls, dead))
    return findings


# --------------------------------------------------------------------------------------------------
# Variant table
# --------------------------------------------------------------------------------------------------
@dataclass
class Variant:
    variant: str
    processors: list[Processor] = field(default_factory=list)

    @property
    def owners(self) -> list[str]:
        return sorted({p.model for p in self.processors})

    @property
    def canonical(self) -> str | None:
        """The oldest owner: inheritance should follow history, so lineage stays stable."""
        dates = build_date_data()
        dated = [(dates[m], m) for m in self.owners if m in dates]
        return min(dated)[1] if dated else (self.owners[0] if self.owners else None)

    @property
    def tag(self) -> str:
        return f"processor:{self.variant}"


def build_variants(processors: list[Processor]) -> dict[str, Variant]:
    """Group processors into tier-1 variants, keyed by variant string."""
    variants: dict[str, Variant] = {}
    for processor in processors:
        variant = variants.setdefault(processor.variant, Variant(processor.variant))
        variant.processors.append(processor)
    return variants


def sources_match(a: Processor, b: Processor) -> bool:
    """
    Whether `a` could take `b`'s `__call__` and hooks unchanged.

    Requires the canonicalised bodies of *the same set* of key methods to be identical, for the
    same reason `blocks_facets.forwards_match` requires equality: equality proves behaviour is
    shared, while similarity only suggests it, and this decides whether someone is told to delete
    code. A processor defining a key method the other does not is not a match, since the extra
    method is behaviour the parent cannot supply.

    Two processors that both define nothing do match: that is the mixin supplying everything to
    both, not a failure to read either of them.
    """
    return a.sources == b.sources


def related(a: str, b: str) -> bool:
    """
    Whether two models already have a declared relationship, in either direction.

    Checked both ways and through the full transitive closure, because duplication between a model
    and its own modular ancestor is not a finding -- it is the converter doing its job.
    """
    if a == b:
        return True
    return b in ancestors(a) or a in ancestors(b)


@dataclass
class Duplicate:
    """Two unrelated processors with one variant and byte-identical key methods."""

    variant: str
    canonical: str
    model: str
    class_name: str
    methods: tuple[str, ...]
    copied_from: str | None
    generated: bool

    @property
    def kind(self) -> str:
        """
        Which of the two shapes of duplication this is.

        `identical_bodies` is duplicated hand-written code: someone can delete a `__call__` and
        inherit. `no_bodies` is a processor that defines none of the key methods at all, so it is
        already nothing but `ProcessorMixin` plus an `__init__` -- interchangeable with its peers,
        but there is no code to delete, only a class that never needed to exist separately. Both
        satisfy "same variant, identical key methods"; they call for opposite actions, so the
        report must not merge them.
        """
        return "identical_bodies" if self.methods else "no_bodies"


def find_duplicates(processors: list[Processor]) -> list[Duplicate]:
    """
    Processors sharing a variant *and* identical key-method sources, not related by modular descent.

    Each group's canonical owner is the oldest model holding the variant, matching the blocks
    registry's rule (lint rule R3): inheritance follows history, so lineage stays stable as models
    are added.

    Processors that define *no* key method are deliberately included, tagged `no_bodies` -- see
    `Duplicate.kind`. The blocks registry's `forwards_match` refuses an empty body because there a
    missing `forward` means "unparseable or not a block"; here defining nothing is a positive
    statement that the mixin does everything, and it is the single largest cluster in the library.
    """
    markers = copied_from_processor_sources()
    dates = build_date_data()
    duplicates: list[Duplicate] = []
    for variant in build_variants(processors).values():
        # Bucket by identical source so a variant with several distinct implementations reports
        # each cluster separately rather than claiming they are all one.
        buckets: dict[tuple, list[Processor]] = defaultdict(list)
        for processor in variant.processors:
            buckets[tuple(sorted(processor.sources.items()))].append(processor)
        for members in buckets.values():
            if len(members) < 2:
                continue
            owner = min(members, key=lambda p: (dates.get(p.model, "9999"), p.model))
            for member in members:
                if member is owner or related(member.model, owner.model):
                    continue
                # Does this model already *declare* that it tracks the canonical owner? A
                # `# Copied from` marker may sit on the class or on any single method, and in
                # processing files it is nearly always a method that is not the duplicated one:
                # musicgen_melody marks `batch_decode`, `_decode_audio` and `get_decoder_prompt_ids`
                # from musicgen but leaves the `__call__` it also copied unmarked. Keying only on
                # the duplicated symbol therefore reports "unmarked" for a relationship the file
                # states three times, so the whole file's markers are consulted. Recording the
                # relationship rather than the exact symbol is what makes the finding actionable --
                # it says a modular parent is already the documented answer.
                copied = markers.get((member.model, member.class_name)) or next(
                    (
                        source
                        for (model, _), source in markers.items()
                        if model == member.model and source == owner.model
                    ),
                    None,
                )
                duplicates.append(
                    Duplicate(
                        variant=variant.variant,
                        canonical=owner.model,
                        model=member.model,
                        class_name=member.class_name,
                        methods=tuple(sorted(member.sources)),
                        copied_from=copied,
                        generated=member.generated,
                    )
                )
    return duplicates


def near_misses(processors: list[Processor]) -> list[tuple[str, str, str]]:
    """
    `(variant, model, model)` pairs that share a variant but whose key methods differ.

    The honest accounting of how lossy the facet vector is. In the blocks registry this gap was 38
    false matches; naming it keeps facet equality from being mistaken for proof.
    """
    out: list[tuple[str, str, str]] = []
    for variant in build_variants(processors).values():
        members = variant.processors
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                if not sources_match(a, b) and not related(a.model, b.model):
                    out.append((variant.variant, a.model, b.model))
    return out


def _selfcheck() -> None:
    """Assert the facts hand-verified by reading these processors' source."""
    groups = kwargs_group_names()
    assert groups, "kwargs group vocabulary came back empty"
    assert {"text_kwargs", "images_kwargs", "videos_kwargs", "audio_kwargs"} <= groups, sorted(groups)
    hooks = mixin_hooks()
    # The premise of the whole registry: the mixin owns a generic `__call__` and the three
    # placeholder hooks it dispatches to. If any of these vanish, the tier-1 split is wrong.
    assert {"__call__", "replace_image_token", "replace_video_token", "replace_audio_token"} <= hooks, sorted(hooks)

    # Naming variance, both directions (see `is_processor_class`).
    assert is_processor_class("Wav2Vec2ProcessorWithLM")
    assert not is_processor_class("Florence2PostProcessor")
    assert not is_processor_class("LlavaProcessorKwargs")
    assert is_processor_class("LlavaNextVideoProcessor")

    processors = scan_repo()
    by_model = {p.model: p for p in processors}
    assert len(processors) == len({(p.model, p.class_name) for p in processors})

    def facets(model: str) -> dict:
        return by_model[model].tier1

    # clip: the minimal processor. Two sub-processors, a two-line `__init__`, and nothing else --
    # it already rides the mixin's generic `__call__` end to end.
    clip = by_model["clip"]
    assert facets("clip")["composition"] == "image_processor+tokenizer", facets("clip")
    assert clip.inherits_generic_call, facets("clip")
    assert facets("clip")["image_expansion"] == "no_image_expansion", facets("clip")
    assert facets("clip")["replacement_shape"] == "no_replacement", facets("clip")
    assert "__call__" not in clip.sources, sorted(clip.sources)

    # llava: re-derives the token count from the *processed* pixel tensor's shape.
    assert facets("llava")["image_expansion"] == "count_from_pixel_shape", facets("llava")
    assert by_model["llava"].inherits_generic_call, facets("llava")

    # paligemma: a constant `image_seq_length`, yet still a hand-written `__call__`. The pairing
    # that makes `call_style` tier 1 -- the simplest possible expansion behind a custom pipeline.
    assert facets("paligemma")["image_expansion"] == "count_fixed", facets("paligemma")
    assert facets("paligemma")["call_style"] == "custom_call", facets("paligemma")

    # qwen2_vl: the fully migrated shape. Three sub-processors, no `__call__`, both image and video
    # counts taken from the grid the image/video processor returns.
    assert facets("qwen2_vl")["composition"] == "image_processor+tokenizer+video_processor", facets("qwen2_vl")
    assert by_model["qwen2_vl"].inherits_generic_call, facets("qwen2_vl")
    assert facets("qwen2_vl")["image_expansion"] == "count_from_grid_thw", facets("qwen2_vl")
    assert facets("qwen2_vl")["video_expansion"] == "count_from_grid_thw", facets("qwen2_vl")

    # idefics3: `rows` x `cols` tiles, each worth `image_seq_len` tokens, with `<row_i_col_j>`
    # markers between them -- hence row_separated, not bare_repeat.
    assert facets("idefics3")["image_expansion"] == "count_from_tile_grid", facets("idefics3")
    assert facets("idefics3")["replacement_shape"] == "row_separated", facets("idefics3")
    # smolvlm computes the same thing through a module-level helper, which is only visible because
    # `scan_file` inlines the methods a hook delegates to.
    assert facets("smolvlm")["image_expansion"] == "count_from_tile_grid", facets("smolvlm")

    # gemma3: `image_seq_length` per Pan-and-Scan crop, so the count follows `num_crops` from the
    # image processor -- not a fixed sequence, despite the constant in its `__init__`.
    assert facets("gemma3")["image_expansion"] == "count_from_patch_count", facets("gemma3")

    # mllama attends to images by cross-attention: it owns an `image_token` but must expand it to
    # exactly one token. Distinct from having no hook at all.
    assert facets("mllama")["image_expansion"] == "single_token", facets("mllama")

    # The two audio processors that predate the multimodal era: both compose
    # `feature_extractor+tokenizer`, both hand-write `__call__`, neither expands anything.
    for model in ("whisper", "wav2vec2"):
        assert facets(model)["composition"] == "feature_extractor+tokenizer", facets(model)
        assert facets(model)["call_style"] == "custom_call", facets(model)
        assert facets(model)["audio_expansion"] == "no_audio_expansion", facets(model)
    # and the class whose name breaks the suffix convention is present at all
    assert "wav2vec2_with_lm" in by_model

    # pixtral emits `[IMG]` runs separated by `[IMG_BREAK]` and terminated by `[IMG_END]`.
    assert facets("pixtral")["replacement_shape"] == "row_separated", facets("pixtral")
    assert facets("pixtral")["image_expansion"] == "count_from_image_sizes", facets("pixtral")

    # Every facet must resolve to a closed vocabulary: an unexpected value either merges variants
    # that differ or splits one that does not, and both corrupt the tag.
    counts = {
        "single_token",
        "count_from_subprocessor",
        "count_from_grid_thw",
        "count_from_tile_grid",
        "count_from_image_sizes",
        "count_from_patch_count",
        "count_from_pixel_shape",
        "count_from_audio_frames",
        "count_fixed",
        "count_custom",
    }
    shapes = {"bare_repeat", "boundary_wrapped", "row_separated", "no_replacement"}
    for processor in processors:
        assert processor.tier1["call_style"] in ("custom_call", "inherited_generic_call"), processor.tier1
        for modality in MODALITIES:
            value = processor.tier1[f"{modality}_expansion"]
            assert value == f"no_{modality}_expansion" or value in counts, (
                f"{processor.model}: {modality}_expansion={value!r}"
            )
        for part in processor.tier1["replacement_shape"].split("+"):
            assert part in shapes, f"{processor.model}: replacement_shape={processor.tier1['replacement_shape']!r}"

    # `count_custom` is the fall-through. It is legitimate but must stay rare: if it climbs, a real
    # expansion strategy is being missed and processors that differ are merging under one tag.
    custom = sum(1 for p in processors for m in MODALITIES if p.tier1[f"{m}_expansion"] == "count_custom")
    assert custom <= 3, f"{custom} hooks fell through to count_custom"

    # The tier line, measured rather than asserted: folding a tier-2 facet into the tag must
    # fragment the table without changing behaviour. If tier 2 ever stops fragmenting it, it was
    # tier 1 all along.
    variants = build_variants(processors)
    with_tier2 = {(p.variant, p.tier2["kwargs_defaults"]) for p in processors}
    assert len(with_tier2) > len(variants), (
        f"kwargs_defaults no longer fragments the table ({len(with_tier2)} vs {len(variants)})"
    )

    # Three models key `_defaults` by a group name `_merge_kwargs` never reads, so those defaults
    # are dead. Pinned so the count cannot silently grow.
    dead = dead_default_groups()
    assert len(dead) == 3, f"expected 3 dead `_defaults` groups, got {len(dead)}: {dead}"
    assert {m for m, _, _ in dead} == {"aria", "mllama", "videoprism"}, dead

    # Hand-verified duplication. `Speech2TextProcessor.__call__`, `WhisperProcessor.__call__` and
    # `Wav2Vec2ProcessorWithLM.__call__` are the same 23 lines of pre-`_merge_kwargs` positional
    # shuffling, in three models with no modular relationship and no `# Copied from` marker between
    # them. speech_to_text is the oldest, so it owns the variant.
    duplicates = find_duplicates(processors)
    by_model = defaultdict(list)
    for duplicate in duplicates:
        by_model[duplicate.model].append(duplicate)
    for model in ("whisper", "wav2vec2_with_lm"):
        found = [d for d in by_model[model] if d.kind == "identical_bodies"]
        assert found and found[0].canonical == "speech_to_text", (model, by_model[model])
        assert found[0].methods == ("__call__",), found[0]
    # musicgen_melody is the marked case, and the reason the marker lookup is file-wide: it carries
    # three `# Copied from ...MusicgenProcessor` markers, none of them on the `__call__` it also
    # copied verbatim.
    melody = [d for d in by_model["musicgen_melody"] if d.kind == "identical_bodies"]
    assert melody and melody[0].canonical == "musicgen", by_model["musicgen_melody"]
    assert melody[0].copied_from == "musicgen", melody[0]
    # ...and whisper is the unmarked case, so the two are genuinely distinguishable.
    assert [d for d in by_model["whisper"] if d.kind == "identical_bodies"][0].copied_from is None

    inherit = sum(1 for p in processors if p.inherits_generic_call)
    print(
        f"selfcheck ok: {len(processors)} processors, {len(variants)} variants, "
        f"{sum(1 for v in variants.values() if len(v.processors) == 1)} singletons, "
        f"{inherit} inherit the generic __call__, {len(find_duplicates(processors))} duplicates"
    )


if __name__ == "__main__":
    _selfcheck()
