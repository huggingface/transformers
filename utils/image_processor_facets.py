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
Image-processor variant facet extraction, the sibling of `utils/blocks_facets.py`.

Every image processor in the library is described by an ordered vector of *facets*, split into two
tiers, exactly as blocks are. The split had to be re-derived, because an image processor is not
shaped like a block:

- a block's behaviour lives in `forward`, so blocks split on "changes `forward`" (tier 1) versus
  "changes `__init__` only" (tier 2);
- an image processor on this branch is *declarative*. `ViTImageProcessor` is seven class attributes
  and no method bodies at all; the backend's `_preprocess` does the work. Splitting on "overrides a
  method" would therefore put 26 processors' entire behaviour in tier 2 and collapse clip, vit and
  siglip into one indistinguishable variant -- which is precisely the question this tool exists to
  answer.

So the line is drawn one level down, at **control flow versus data**:

- **tier 1 is anything that selects which code runs.** A `do_*` flag gates a call in
  `_preprocess`; the *key set* of the `size` dict picks a branch in `resize()` (read it: it tests
  `shortest_edge`/`longest_edge`, then `max_height`/`max_width`, then `height`/`width`); a tiling
  attribute pulls in a whole patching path; an overridden `_preprocess` replaces the pipeline
  outright. Tier 1 alone decides the variant, and therefore the tag.
- **tier 2 is data fed into a path tier 1 already selected.** `size = {"height": 224}` versus
  `{"height": 384}`, `BICUBIC` versus `BILINEAR`, `OPENAI_CLIP_MEAN` versus
  `IMAGENET_STANDARD_MEAN`, torchvision versus PIL. It never gates a match, but it is always
  reported, because -- as with blocks -- people routinely fork a whole processor class over a
  single tier-2 bit. That is the headline finding this tool produces.

Two consequences of that line are worth stating up front, because they are the useful ones:

- **The backend is tier 2.** `CLIPImageProcessor` and `CLIPImageProcessorPil` implement one policy
  against two libraries. Making the backend part of the identity would double the variant count and
  hide the actual question, which is "do the two backends agree?". Because it is tier 2, a
  disagreement shows up as *the same model appearing in two different variants* -- a parity bug,
  reported by `backend_parity_gaps()`.
- **Facets only nominate candidates; a body hash decides.** This is the lesson `blocks_facets.py`
  learned the hard way (38 attention classes had identical facets and genuinely different code), and
  it transfers: 141 distinct `_preprocess` bodies live behind far fewer facet vectors. So the
  canonicalised source of every non-trivial overridden method is compared too, and facet-equal but
  code-different is a *near miss*, never a match.

Vocabularies are read out of the library rather than invented, with `ast`, never by importing
(`image_utils` pulls in PIL and torchvision):

- `utils/constants.py` gives the six named mean/std constants. Reading them lets a literal
  `[0.48145466, 0.4578275, 0.40821073]` resolve to `openai_clip` instead of `custom_stats` -- four
  processors and `IDEFICS_STANDARD_MEAN` spell out CLIP's numbers by hand, and without resolution
  they would look like five bespoke normalisations.
- `pil_torch_interpolation_mapping` in `image_utils.py` gives the set of resample filters the library
  actually supports, which is the vocabulary this axis speaks. Their integer values are the one thing
  that genuinely cannot be read here (`PILImageResampling` is an alias for `PIL.Image.Resampling`),
  so `_PIL_RESAMPLING_VALUES` carries them with a selfcheck tying it back to the library's own list;
  five processors declare the bare int (`resample = 2`) and would otherwise each get a bucket.
- `AnnotationFormat` in `image_utils.py` gives the detection annotation vocabulary.

Stdlib only, on purpose, same as `blocks_facets.py`: importable from a repo-consistency checker
without dragging torch, torchvision or PIL into `make check-repo`. The historical primitives
(`build_date_data`, `modular_parents`, `ancestors`, `canonical_source`) are imported from
`blocks_facets` rather than reimplemented.
"""

import ast
import hashlib
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path


if str(Path(__file__).parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent))

from blocks_facets import (  # noqa: E402
    MODELS_ROOT,
    REPO_ROOT,
    ancestors,
    build_date_data,
    canonical_source,
    is_cross_model_import,
    modular_parents,
    parent_from_module,
)


# The backend a processor targets is written on the filename, not just the base class:
# `image_processing_clip.py` is the torchvision one, `image_processing_pil_clip.py` the PIL one.
# Using the filename means the backend is known before the class body is parsed, which matters for
# the two processors that inherit from another model's processor rather than from a backend
# (`Ernie4_5_VL_MoeImageProcessor`) and so never name a backend at all.
_PIL_FILE_PREFIX = "image_processing_pil_"
BACKEND_BASES = {"TorchvisionBackend": "torchvision_backend", "PilBackend": "pil_backend"}
# `BaseImageProcessor` itself is a legitimate base for the handful of processors that own their whole
# pipeline (`timm_wrapper`, `vivit`, `video_llava`, `emu3`) and therefore need neither backend.
PROCESSOR_BASES = frozenset({*BACKEND_BASES, "BaseImageProcessor"})

# Axis order == the "transformers format" for image processors. Same principle as the block axes:
# ordered by override cost, expensive first, so that picking a parent by longest common prefix forces
# agreement on the expensive axes and leaves divergence in the cheap tail.
#
# Left in *semantic* order, not fitted -- and this is a measurement, not an omission.
# `measure_axis_costs()` is the same instrument `blocks_facets` uses, and running it here finds only
# 75 cross-model image-processor overrides, of which 45 share the parent's variant outright and just
# 7 differ on exactly one axis. Only `outputs` clears the 3-sample bar; every other axis falls back
# to the global median, so a "fitted" order would be fitting noise. This is the same call
# `blocks_facets` makes for its six MoE axes, for the same reason.
#
# The semantic ordering is by what differing on the axis forces you to write: `pipeline` and `tiling`
# lead because differing on either means writing a `_preprocess` from scratch, `resize` next because
# it means an overridden `resize`, and `photometric` and `channels` trail because differing on either
# is a one-line flag. Re-fit once the modular image-processor population grows.
IMAGE_PROCESSOR_AXES = (
    "pipeline",
    "tiling",
    "resize",
    "outputs",
    "annotations",
    "pad",
    "crop",
    "photometric",
    "channels",
)


# --------------------------------------------------------------------------------------------------
# Vocabularies read out of the library source (never imported: these files pull in PIL/torchvision)
# --------------------------------------------------------------------------------------------------
@cache
def normalization_vocabulary() -> dict[tuple[float, ...], str]:
    """
    `{(0.5, 0.5, 0.5): "imagenet_standard", ...}` from `src/transformers/utils/constants.py`.

    Keyed by *value*, not by name, because that is what makes the axis a real dedup axis: a
    processor that writes `image_mean = [0.5, 0.5, 0.5]` is using ImageNet-standard normalisation
    whether or not it says so, and `IDEFICS_STANDARD_MEAN` is CLIP's numbers under another name.
    Keying by name would have reported those as bespoke and hidden the duplication.
    """
    path = REPO_ROOT / "src" / "transformers" / "utils" / "constants.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    by_value: dict[tuple[float, ...], str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        # `IMAGENET_DEFAULT_MEAN` and `IMAGENET_DEFAULT_STD` both map to `imagenet_default`: the
        # facet names the *stat set*, and a processor that mixes one set's mean with another's std
        # should read as bespoke rather than silently picking a side.
        family = re.sub(r"_(MEAN|STD)$", "", name).lower()
        if family == name.lower():
            continue
        try:
            by_value[tuple(ast.literal_eval(node.value))] = family
        except (ValueError, SyntaxError, TypeError):
            continue
    return by_value


# The integer value of each `PIL.Image.Resampling` member. This is the one table that cannot be read
# out of this repo: `image_utils.py` does `PILImageResampling = PIL.Image.Resampling`, so the members
# live in PIL, and importing PIL is exactly what this module must not do. It is needed because five
# processors declare the bare int (`resample = 2`) -- `slanext` even annotates it
# `# PILImageResampling.BILINEAR` -- and without the table they would each land in a bucket of their
# own instead of joining the 85 that spell out `BILINEAR`. The *names* are still read from the
# library, and `_selfcheck` asserts this table covers exactly the filters the library supports, so a
# new filter shows up as a failure rather than as a silent gap.
_PIL_RESAMPLING_VALUES = {"NEAREST": 0, "LANCZOS": 1, "BILINEAR": 2, "BICUBIC": 3, "BOX": 4, "HAMMING": 5}


@cache
def resample_vocabulary() -> tuple[dict[str, str], dict[int, str]]:
    """
    The resample filter names from `pil_torch_interpolation_mapping`, plus the int -> name map.

    The mapping dict in `image_utils.py` enumerates every filter the library actually supports, which
    is the vocabulary this axis should speak. It is declared twice -- once under
    `if is_torchvision_available()` and once as `{}` in the `else` -- so the first *non-empty* one is
    the real one. Taking the last match instead emptied the vocabulary and turned every resample
    facet into `custom_resample`.
    """
    path = REPO_ROOT / "src" / "transformers" / "image_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not any(
            isinstance(t, ast.Name) and t.id == "pil_torch_interpolation_mapping" for t in node.targets
        ):
            continue
        if isinstance(node.value, ast.Dict):
            found = [key.attr for key in node.value.keys if isinstance(key, ast.Attribute)]
            if found:
                names = found
                break
    by_name = {name: name.lower() for name in names}
    by_int = {_PIL_RESAMPLING_VALUES[name]: name.lower() for name in names if name in _PIL_RESAMPLING_VALUES}
    return by_name, by_int


@cache
def annotation_format_vocabulary() -> tuple[str, ...]:
    """The `AnnotationFormat` members (`coco_detection`, `coco_panoptic`) from `image_utils`."""
    path = REPO_ROOT / "src" / "transformers" / "image_utils.py"
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ClassDef) and node.name == "AnnotationFormat":
            return tuple(
                s.value.value
                for s in node.body
                if isinstance(s, ast.Assign) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)
            )
    return ()


# --------------------------------------------------------------------------------------------------
# Naming variance the extractor has to absorb, or the variant count inflates
# --------------------------------------------------------------------------------------------------
_has = lambda names, *needles: any(n in names for n in needles)  # noqa: E731


# A method whose whole body is `return super().<same name>(...)` carries no behaviour. 130 of the
# 142 `preprocess` overrides in the library are exactly that -- an `@auto_docstring` peg so the
# processor's own signature shows up in the docs -- and 123 of the 175 `__init__` overrides are the
# same shape. Counting them as behaviour would have made almost every processor look like it had a
# custom pipeline, and would have made 130 pure-boilerplate bodies part of the duplicate hash.
def is_trivial_override(node: ast.FunctionDef) -> bool:
    """
    True if the method's whole body is one call to `super()`, so it changes nothing.

    Deliberately strict: *exactly* one statement. An earlier version allowed two, on the theory that
    a guard clause plus a `super()` call was still boilerplate -- and it swallowed
    `CLIPImageProcessor.__init__`, whose two statements are a KOSMOS-2 `use_square_size` shim that
    rewrites `size` and then the `super()` call. Calling that trivial reported clip as a
    pure-attribute processor and would have offered it as a parent whose extra behaviour the child
    silently inherits.
    """
    body = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if len(body) != 1:
        return False
    statement = body[0]
    if isinstance(statement, ast.Return) and statement.value is not None:
        call = statement.value
    elif isinstance(statement, ast.Expr):
        call = statement.value
    else:
        return False
    return isinstance(call, ast.Call) and ast.unparse(call.func) == f"super().{node.name}"


def _truthy(value: str | None) -> bool:
    """Whether a class attribute is declared with a value that enables the operation.

    Absent, `None` and `False` all mean off: `BaseImageProcessor` declares no `do_*` default, so an
    unset flag resolves to `None` at runtime and the pipeline skips the step. Treating "unset" as
    unknown rather than off would have split `no_center_crop` into two variants for no reason -- 164
    processors simply never mention `do_center_crop`.
    """
    return value is not None and value not in ("None", "False")


def _size_keys(literal: str | None) -> frozenset[str]:
    """The key set of a `size`-like dict literal, which is what `resize()` actually branches on."""
    if not literal:
        return frozenset()
    return frozenset(re.findall(r"['\"](\w+)['\"]\s*:", literal))


# --------------------------------------------------------------------------------------------------
# Tier-1 facet extraction
# --------------------------------------------------------------------------------------------------
# Each rung is (facet value, required attributes, required methods). The first rung whose signals are
# all present wins, so the ladder runs most-specific first. Every rung was read off real code rather
# than guessed, and the model that motivated it is named -- these are the families a reader needs to
# be able to tell apart, and they are the axis with the most genuine variety by a wide margin.
_TILING_LADDER: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    # qwen2_vl and its 25 descendants: resize to a patch grid, then flatten into
    # `(temporal x spatial)` patch tokens and report the grid as `image_grid_thw`.
    ("patch_grid_tokens", ("merge_size", "temporal_patch_size"), ()),
    # pix2struct: no pixel grid at all, a fixed budget of flattened patches plus an attention mask.
    ("flattened_patches", ("max_patches",), ("extract_flattened_patches",)),
    # got_ocr2 / cohere2_vision / deepseek_ocr2: pick a tile count between `min_patches` and
    # `max_patches` by aspect ratio, then crop the resized image into that grid.
    ("dynamic_aspect_tiles", ("crop_to_patches", "min_patches"), ()),
    ("dynamic_aspect_tiles", ("dynamic_hd",), ()),
    # llava_next / llava_onevision: choose from an explicit list of allowed resolutions.
    ("pinpoint_tiles", ("image_grid_pinpoints",), ()),
    # mllama: aspect-ratio buckets over a fixed maximum number of tiles, with a tile mask.
    ("aspect_ratio_tiles", ("max_image_tiles",), ()),
    ("aspect_ratio_tiles", ("max_num_tiles",), ()),
    # gemma3: keep the whole image *and* add crops of it when the aspect ratio is extreme.
    ("pan_and_scan", ("do_pan_and_scan",), ()),
    # minicpm: slice into a grid whose shape is solved for per image.
    ("slice_grid", ("max_slice_nums",), ()),
    # idefics2 / idefics3 / smolvlm: split into a fixed grid of `max_image_size` tiles and keep the
    # thumbnail, reporting rows and columns.
    ("grid_split_tiles", ("do_image_splitting",), ()),
    ("grid_split_tiles", ("do_split_image",), ()),
    ("grid_split_tiles", ("split_image",), ()),
    # siglip2: a *budget* of patches rather than a fixed resolution, so every image gets a different
    # patch count and needs a `pixel_attention_mask` plus `spatial_shapes`.
    ("variable_patch_budget", ("max_num_patches",), ()),
    # Anything that patches or splits by a route none of the above describes. Named rather than
    # "unknown" so it groups honestly instead of merging unrelated processors.
    ("custom_tiling", (), ("patchify",)),
    ("custom_tiling", (), ("crop_image_to_patches",)),
    ("custom_tiling", (), ("get_image_patches",)),
    ("custom_tiling", (), ("_get_image_patches",)),
    ("custom_tiling", (), ("split_images",)),
)


def _tiling_facet(attrs: dict[str, str], methods: set[str]) -> str:
    """
    Which VLM tiling strategy the processor uses, or `no_tiling`.

    `patch_size` alone deliberately does *not* count. Seven processors declare it purely so the
    downstream model can compute a token count, and reading it as a tiling signal made them look
    like members of the qwen2_vl patch-grid family.
    """
    for value, needed_attrs, needed_methods in _TILING_LADDER:
        if all(a in attrs for a in needed_attrs) and all(m in methods for m in needed_methods):
            return value
    return "no_tiling"


def _resize_facet(attrs: dict[str, str], methods: set[str]) -> str:
    """
    The resize *policy*: what geometry comes out, not what numbers went in.

    The values mirror the branches in `TorchvisionBackend.resize`, which is the point of putting the
    size dict's key set in tier 1 and its values in tier 2: the key set picks the branch, the values
    only parameterise it. `resize_shortest_edge_capped` (`{"shortest_edge": 800, "longest_edge":
    1333}`, 46 processors) is a genuinely different output shape from `resize_shortest_edge`
    (32 processors), which is different again from `resize_to_square` (102).
    """
    if not _truthy(attrs.get("do_resize")):
        return "no_resize"
    # A patch-grid resize solves for a size that is a whole number of patches, so it overrides the
    # size dict entirely -- qwen2_vl declares `{"shortest_edge": 56*56, "longest_edge": 28*28*1280}`
    # but those are *pixel-count* bounds fed to `smart_resize`, not edge lengths. Reading them as
    # edges put the whole qwen family in detr's variant.
    if "merge_size" in attrs or "smart_resize" in methods:
        return "resize_patch_grid"
    if _has(attrs, "size_divisor", "ensure_multiple_of"):
        return "resize_multiple_of"
    # `crop_pct` lives on the *resize* axis, not the crop axis, and that placement is the whole
    # reason it is worth a facet. convnext resizes the shortest edge to `size / crop_pct` and then
    # takes a plain fixed centre crop -- the crop is ordinary, the resize is not. Putting it on the
    # crop axis said "the crop is unusual", which is false, and it hid the fact that `levit` does the
    # identical thing with `int((256 / 224) * shortest_edge)` hardcoded inside an overridden
    # `resize`: same policy, expressed imperatively, and therefore a different facet value. That
    # disagreement is the finding.
    if "crop_pct" in attrs:
        return "resize_shortest_edge_upscaled"
    keys = _size_keys(attrs.get("size"))
    if {"shortest_edge", "longest_edge"} <= keys:
        return "resize_shortest_edge_capped"
    if "shortest_edge" in keys:
        return "resize_shortest_edge"
    if "longest_edge" in keys:
        return "resize_longest_edge"
    if {"max_height", "max_width"} <= keys:
        return "resize_max_height_width"
    if {"height", "width"} <= keys:
        return "resize_to_square"
    # `do_resize` is on but no size is declared on the class: the size arrives from the checkpoint's
    # `preprocessor_config.json`. Named so it cannot be mistaken for a policy.
    return "resize_size_from_config"


def _pad_facet(attrs: dict[str, str], methods: set[str]) -> str:
    """
    The padding strategy, which decides whether the batch is ragged and needs a mask.

    `pad_to_batch_max` is the interesting one: 58 processors set `do_pad = True` and only 8 declare a
    `pad_size`, so the overwhelming majority pad every image up to the largest in the batch -- the
    reason detr-family processors return `pixel_mask` at all.
    """
    if "pad_to_square" in methods or "background_color" in attrs:
        return "pad_to_square"
    if not _truthy(attrs.get("do_pad")):
        # Some processors pad unconditionally from an overridden `_preprocess` without a flag.
        return "pad_unflagged" if _has(methods, "pad_to_max_num_crops", "_pad_for_batching") else "no_pad"
    if _has(attrs, "pad_size", "mask_pad_size"):
        return "pad_to_fixed_size"
    if _has(attrs, "size_divisor", "ensure_multiple_of", "patch_size") and "pad" in methods:
        return "pad_to_multiple"
    return "pad_to_batch_max"


def _crop_facet(attrs: dict[str, str]) -> str:
    """Whether a centre crop runs. Deliberately two-valued: see `_resize_facet` for `crop_pct`."""
    return "center_crop_fixed" if _truthy(attrs.get("do_center_crop")) else "no_center_crop"


def _photometric_facet(attrs: dict[str, str]) -> str:
    rescale, normalize = _truthy(attrs.get("do_rescale")), _truthy(attrs.get("do_normalize"))
    if rescale and normalize:
        return "rescale_and_normalize"
    if rescale:
        return "rescale_only"
    if normalize:
        return "normalize_only"
    return "no_photometric"


def _channels_facet(attrs: dict[str, str], methods: set[str]) -> str:
    """
    What happens to the channel axis before the pipeline runs.

    Split out from the rest because it is decided in `process_image`, upstream of `_preprocess`, and
    it is a near-perfect 50/50 in the library (102 processors convert to RGB, 106 leave the input
    alone) -- an axis that genuinely halves the population is worth its own slot.
    """
    if _truthy(attrs.get("do_flip_channel_order")) or "flip_channel_order" in methods:
        return "flip_channel_order"
    if _truthy(attrs.get("do_grayscale")):
        return "to_grayscale"
    if _truthy(attrs.get("do_convert_rgb")):
        return "convert_rgb"
    return "keep_input_channels"


def _annotations_facet(attrs: dict[str, str], methods: set[str]) -> str:
    """Detection / segmentation side-channels, in `AnnotationFormat`'s own vocabulary."""
    formats = annotation_format_vocabulary()
    declared = attrs.get("format", "")
    for name in formats:
        if name in declared.lower():
            return f"{name}_annotations"
    if _has(methods, "prepare_annotation", "resize_annotation", "normalize_annotation"):
        return f"{formats[0]}_annotations" if formats else "detection_annotations"
    if _has(methods, "convert_segmentation_map_to_binary_masks", "encode_inputs"):
        return "binary_mask_annotations"
    if _truthy(attrs.get("do_reduce_labels")) or "reduce_label" in methods:
        return "segmentation_maps_reduced"
    if "segmentation_maps" in methods:
        return "segmentation_maps"
    return "no_annotations"


def _pipeline_facet(methods: set[str]) -> str:
    """
    Whether the processor uses the backend's pipeline, replaces it, or replaces the orchestration.

    This is the most expensive axis to differ on and therefore leads the vector: `standard_pipeline`
    means the class is *only* class attributes and can inherit everything, `custom_preprocess` means
    someone wrote the batch loop, and `custom_orchestration` means they took over
    `_preprocess_image_like_inputs` too, usually to thread a second input (segmentation maps,
    prompt images) through.
    """
    if "_preprocess_image_like_inputs" in methods:
        return "custom_orchestration"
    if _has(methods, "_preprocess", "preprocess"):
        return "custom_preprocess"
    return "standard_pipeline"


def _outputs_facet(output_keys: frozenset[str]) -> str:
    """What the processor returns beyond `pixel_values`, `+`-joined and sorted."""
    extras = sorted(output_keys - {"pixel_values"})
    return "pixel_values+" + "+".join(extras) if extras else "pixel_values_only"


def image_processor_facets(
    attrs: dict[str, str],
    methods: set[str],
    output_keys: frozenset[str],
    module_constants: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Both tiers for one processor, from its resolved attributes, methods and output keys."""
    tier1 = {
        "pipeline": _pipeline_facet(methods),
        "tiling": _tiling_facet(attrs, methods),
        "resize": _resize_facet(attrs, methods),
        "outputs": _outputs_facet(output_keys),
        "annotations": _annotations_facet(attrs, methods),
        "pad": _pad_facet(attrs, methods),
        "crop": _crop_facet(attrs),
        "photometric": _photometric_facet(attrs),
        "channels": _channels_facet(attrs, methods),
    }
    tier2 = {
        "resample": _resample_value(attrs.get("resample")),
        "norm_stats": _norm_stats_value(attrs.get("image_mean"), attrs.get("image_std"), module_constants or {}),
        "size_values": _size_values(attrs.get("size")),
        "crop_values": _size_values(attrs.get("crop_size")) if "crop_size" in attrs else "no_crop_size",
        "rescale": _rescale_value(attrs.get("rescale_factor")),
        "kwargs": "custom_images_kwargs"
        if attrs.get("valid_kwargs", "ImagesKwargs") != "ImagesKwargs"
        else "standard_images_kwargs",
    }
    return tier1, tier2


def _resample_value(literal: str | None) -> str:
    """`PILImageResampling.BICUBIC`, `3` and `InterpolationMode.BICUBIC` all reduce to `bicubic`."""
    if not literal:
        return "resample_from_config"
    by_name, by_int = resample_vocabulary()
    for name, value in by_name.items():
        if name in literal:
            return value
    if literal.strip().isdigit():
        return by_int.get(int(literal.strip()), f"resample_{literal.strip()}")
    return "custom_resample"


def _norm_stats_value(mean: str | None, std: str | None, module_constants: dict[str, str]) -> str:
    """
    Which named stat set the normalisation uses, resolving literals through `utils/constants.py`.

    Resolution follows three hops, because the library spells the same six numbers three ways:
    the canonical name (`OPENAI_CLIP_MEAN`, 53 processors), a file-local alias for it
    (`IDEFICS_STANDARD_MEAN = [0.48145466, ...]`, and `FLAVA_IMAGE_MEAN = OPENAI_CLIP_MEAN`, which
    is a name pointing at a name), or the numbers written out inline (`[0.5, 0.5, 0.5]`, 8
    processors). Stopping at the first hop reported the aliases and the inline literals as bespoke,
    which is exactly the duplication this axis exists to surface.

    A mean and a std from different families reads as `mixed_stats` rather than picking one, since
    that combination is neither of the named sets and pretending otherwise would suggest a parent
    whose numbers are wrong.
    """
    if mean is None and std is None:
        return "no_stats"
    by_value = normalization_vocabulary()
    families = set(by_value.values())

    def resolve(literal: str | None, depth: int = 0) -> str | None:
        if not literal or depth > 4:  # depth guard: an alias chain cannot be circular, but be safe.
            return None
        literal = literal.strip()
        family = re.sub(r"_(MEAN|STD)$", "", literal).lower()
        if family in families:
            return family
        # A file-local alias: follow it to whatever it points at, name or literal.
        if literal in module_constants:
            return resolve(module_constants[literal], depth + 1)
        try:
            parsed = ast.literal_eval(literal)
        except (ValueError, SyntaxError, TypeError):
            return None
        values = tuple(parsed) if isinstance(parsed, (list, tuple)) else (parsed,) * 3
        try:
            return by_value.get(tuple(float(v) for v in values))
        except (TypeError, ValueError):
            return None

    resolved = {resolve(mean), resolve(std)}
    if len(resolved) == 1:
        return resolved.pop() or "custom_stats"
    return "mixed_stats" if None not in resolved else "custom_stats"


def _size_values(literal: str | None) -> str:
    """A readable rendering of the declared size, e.g. `224x224` or `shortest_edge_800_1333`."""
    if not literal or literal == "None":
        return "size_from_config"
    pairs = re.findall(r"['\"](\w+)['\"]\s*:\s*([^,}]+)", literal)
    if not pairs:
        return "size_from_config"
    numbers = {k: v.strip().replace(" ", "") for k, v in pairs}
    if {"height", "width"} <= set(numbers):
        return f"{numbers['height']}x{numbers['width']}"
    return "_".join(f"{k}_{v}" for k, v in numbers.items())


def _rescale_value(literal: str | None) -> str:
    if not literal or literal == "None":
        # `BaseImageProcessor.rescale_factor` is `1 / 255`; not declaring it means taking that.
        return "rescale_1_over_255"
    return "rescale_1_over_255" if literal.replace(" ", "") == "1/255" else f"rescale_{literal.replace(' ', '')}"


# --------------------------------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------------------------------
# Keys the processor builds into its `BatchFeature`, in the two spellings that occur.
_BATCH_DATA_RE = re.compile(r"BatchFeature\(\s*data=\{(.*?)\}", re.DOTALL)
_BATCH_KEY_RE = re.compile(r"['\"](\w+)['\"]\s*:")
_BATCH_SETITEM_RE = re.compile(r"(?:batch_feature|encoded_outputs|data|encoded_inputs)\[['\"](\w+)['\"]\]\s*=")
_MODEL_INPUT_NAMES_RE = re.compile(r"model_input_names\s*=\s*\[([^\]]*)\]")


@dataclass
class Processor:
    """One image-processor class found in one `image_processing_*.py`."""

    model: str
    path: Path
    class_name: str
    backend: str
    tier1: dict = field(default_factory=dict)
    tier2: dict = field(default_factory=dict)
    lineno: int = 0
    # Every non-trivial overridden method, canonicalised and concatenated. Facets are a *lossy*
    # summary -- the library has 141 distinct `_preprocess` bodies behind far fewer facet vectors --
    # so this is what confirms a match, exactly as `Block.forward` does for blocks.
    body: str = ""
    # Which methods this class actually implements, kept for reporting: it is the difference between
    # "these two are the same" and "these two are the same *and* here is the code to delete".
    methods: frozenset[str] = frozenset()

    @property
    def is_declarative(self) -> bool:
        """No method bodies at all: the class is a bag of defaults over the backend's pipeline."""
        return not self.body

    @property
    def variant(self) -> str:
        """The tag: tier-1 facet values in axis order. Identical variant == same observable policy."""
        return "|".join(self.tier1.get(axis, "?") for axis in IMAGE_PROCESSOR_AXES)

    @property
    def body_hash(self) -> str:
        # Content hash, not `hash()`: str hashing is salted per process, and this value ends up in
        # a report people diff between runs.
        return "declarative" if self.is_declarative else hashlib.sha1(self.body.encode()).hexdigest()[:10]

    def tier2_delta(self, other: "Processor") -> dict[str, tuple[str, str]]:
        """Tier-2 facets that differ from `other` -- the attribute-rewrite hint on a suggestion."""
        return {k: (v, other.tier2.get(k, "?")) for k, v in self.tier2.items() if other.tier2.get(k, "?") != v}


def _class_backend(path: Path, bases: list[str]) -> str:
    for base in bases:
        if base in BACKEND_BASES:
            return BACKEND_BASES[base]
    return "pil_backend" if path.name.startswith(_PIL_FILE_PREFIX) else "torchvision_backend"


def scan_file(path: Path, model: str) -> list[Processor]:
    """
    Extract every image-processor class from one `image_processing_*.py`.

    Attributes and methods are resolved through the *in-file* base chain, because a handful of
    processors subclass another processor in the same file (`Ernie4_5_VL_MoeImageProcessor`). Not
    following that chain reported those classes as declaring nothing at all, which put them in the
    same variant as a processor that really does nothing.
    """
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    found: list[Processor] = []
    # `{class name: (attrs, methods, output keys)}` for classes declared earlier in this file.
    local: dict[str, tuple[dict[str, str], set[str], set[str]]] = {}
    # Module-level `NAME = ...` in this file, so a file-local mean/std alias can be followed to the
    # constant it really is. See `_norm_stats_value`.
    module_constants: dict[str, str] = {}
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            module_constants[node.targets[0].id] = ast.unparse(node.value)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [ast.unparse(b) for b in node.bases]
        # A `*Kwargs` TypedDict is a parameter declaration, not a processor; `FlavaMaskingGenerator`
        # and `AutoImageProcessor` are neither. Requiring a real processor base excludes all three
        # without a name blocklist.
        if not (set(bases) & PROCESSOR_BASES or any(b in local for b in bases)):
            continue

        attrs: dict[str, str] = {}
        methods: set[str] = set()
        output_keys: set[str] = set()
        for base in bases:
            if base in local:
                inherited_attrs, inherited_methods, inherited_keys = local[base]
                attrs.update(inherited_attrs)
                methods |= inherited_methods
                output_keys |= inherited_keys

        class_source = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        canonical_parts: list[str] = []
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                attrs[stmt.targets[0].id] = ast.unparse(stmt.value)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                attrs[stmt.target.id] = ast.unparse(stmt.value)
            elif isinstance(stmt, ast.FunctionDef) and not is_trivial_override(stmt):
                methods.add(stmt.name)
                try:
                    canonical_parts.append(canonical_source(stmt))
                except (SyntaxError, ValueError):
                    # A method the canonicaliser cannot round-trip still counts as behaviour; drop
                    # only its body, so the class can never be reported as a duplicate by accident.
                    canonical_parts.append(f"<unparseable {stmt.name}>")

        for blob in _BATCH_DATA_RE.findall(class_source):
            output_keys |= set(_BATCH_KEY_RE.findall(blob))
        output_keys |= set(_BATCH_SETITEM_RE.findall(class_source))
        declared = _MODEL_INPUT_NAMES_RE.search(attrs.get("model_input_names", "") or "")
        if "model_input_names" in attrs:
            output_keys |= set(re.findall(r"['\"](\w+)['\"]", attrs["model_input_names"]))
        elif declared:
            output_keys |= set(re.findall(r"['\"](\w+)['\"]", declared.group(1)))

        local[node.name] = (dict(attrs), set(methods), set(output_keys))
        tier1, tier2 = image_processor_facets(attrs, methods, frozenset(output_keys), module_constants)
        body = "\n".join(canonical_parts)
        # Strip the model's own name so two models' bodies compare on structure, not on naming --
        # the same normalisation `blocks_facets.canonical_method` applies.
        body = re.sub(rf"\b{re.escape(model.replace('_', ''))}", "X", body, flags=re.IGNORECASE)
        found.append(
            Processor(
                model=model,
                path=path,
                class_name=node.name,
                backend=_class_backend(path, bases),
                tier1=tier1,
                tier2=tier2,
                lineno=node.lineno,
                body=body,
                methods=frozenset(methods),
            )
        )
    return found


def scan_repo(models_root: Path = MODELS_ROOT) -> list[Processor]:
    """Scan every `image_processing_*.py` under `models_root`. `auto` holds only the dispatcher."""
    processors: list[Processor] = []
    for model_dir in sorted(p for p in models_root.iterdir() if p.is_dir()):
        if model_dir.name == "auto":
            continue
        for path in sorted(model_dir.glob("image_processing_*.py")):
            processors.extend(scan_file(path, model_dir.name))
    return processors


# --------------------------------------------------------------------------------------------------
# Legacy reuse markers and declared inheritance
# --------------------------------------------------------------------------------------------------
_COPIED_FROM_RE = re.compile(r"#\s*copied from transformers\.models\.(\w+)\.\w+\.(\w+)", re.IGNORECASE)


@cache
def copied_from_sources(models_root: Path = MODELS_ROOT) -> dict[tuple[str, str], str]:
    """
    `{(model, class): source model}` for `# Copied from` markers in image-processing files.

    `blocks_facets.copied_from_sources` globs `modeling_*.py` only, so this is the image-processing
    twin rather than a reimplementation. A marker is the *legacy* reuse mechanism the modular work is
    replacing, so unlike in the block registry it does not excuse a finding -- it makes one easier to
    act on, because `utils/check_copies.py` already proves the two bodies are in sync.
    """
    sources: dict[tuple[str, str], str] = {}
    for model_dir in sorted(p for p in models_root.iterdir() if p.is_dir()):
        for path in sorted(model_dir.glob("image_processing_*.py")):
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


@dataclass
class Override:
    """One `class Child(ParentImageProcessor):` in a modular file, and the lines it spends."""

    child_model: str
    parent_model: str
    child_class: str
    parent_class: str
    loc: int


def modular_overrides(models_root: Path = MODELS_ROOT) -> list[Override]:
    """Every cross-model image-processor subclass declared in a `modular_*.py`, with its size."""
    found: list[Override] = []
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
                if not isinstance(node, ast.ClassDef) or not node.bases:
                    continue
                base = node.bases[0]
                base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
                # Only image processors, and not their `*Kwargs` TypedDicts.
                if base_name not in owner or "ImageProcessor" not in node.name or node.name.endswith("Kwargs"):
                    continue
                found.append(
                    Override(model_dir.name, owner[base_name], node.name, base_name, node.end_lineno - node.lineno + 1)
                )
    return found


MIN_SAMPLES_PER_AXIS = 3


def measure_axis_costs(processors: list[Processor]) -> tuple[dict[str, float], list[int]]:
    """
    Measure what differing on each axis costs, in lines of override -- the ground truth for the order.

    Same construction as `blocks_facets.measure_axis_costs`: every `class Child(Parent)` in a modular
    file is a case where we know which axes differ and can count the lines the child spent. Cases
    differing on exactly one axis measure that axis; axes with too few samples fall back to the
    global median so one lucky observation cannot dominate the ordering.

    Also returns the sizes of the overrides whose variant *matches* the parent. For blocks that
    baseline is ~2 lines and confirms the design; here it is 45 overrides with a median of 62 lines,
    which does *not* confirm it -- it is the headline finding instead. Ten of the 45 are the ideal
    (byte-identical body, 2-9 lines of attribute overrides); twelve are 95%-or-more identical to the
    parent and still spend 1306 lines between them; the rest genuinely diverge, which is the honest
    admission that nine axes cannot fully describe a detection or segmentation post-processing suite.
    """
    exact = {(p.model, p.class_name): p for p in processors}
    per_axis: dict[str, list[int]] = defaultdict(list)
    single_axis: list[int] = []
    baseline: list[int] = []
    for override in modular_overrides():
        child = exact.get((override.child_model, override.child_class))
        parent = exact.get((override.parent_model, override.parent_class))
        if child is None or parent is None:
            continue
        delta = [
            axis
            for axis, mine, theirs in zip(IMAGE_PROCESSOR_AXES, child.variant.split("|"), parent.variant.split("|"))
            if mine != theirs
        ]
        if not delta:
            baseline.append(override.loc)
        elif len(delta) == 1:
            per_axis[delta[0]].append(override.loc)
            single_axis.append(override.loc)
    fallback = statistics.median(single_axis) if single_axis else 0.0
    costs = {
        axis: statistics.median(per_axis[axis]) if len(per_axis.get(axis, [])) >= MIN_SAMPLES_PER_AXIS else fallback
        for axis in IMAGE_PROCESSOR_AXES
    }
    return costs, baseline


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
        return f"image_processor:{self.variant}"


def build_variants(processors: list[Processor]) -> dict[str, Variant]:
    """Group processors into tier-1 variants, keyed by variant string."""
    variants: dict[str, Variant] = {}
    for processor in processors:
        variant = variants.setdefault(processor.variant, Variant(processor.variant))
        variant.processors.append(processor)
    return variants


@dataclass
class Duplicate:
    """Two processors with the same variant *and* the same key-method bodies, unrelated by modular."""

    variant: str
    backend: str
    members: list[Processor]
    body_hash: str

    @property
    def models(self) -> list[str]:
        return sorted({p.model for p in self.members})


def _related(a: str, b: str) -> bool:
    """
    Whether two models already share a modular lineage in either direction.

    Deliberately *model*-level rather than class-level, using the same DAG the block registry uses.
    Both were tried: restricting the relation to image-processor subclass edges (from
    `modular_overrides`) produced the identical answer on the current tree, because every
    body-identical pair here is related both ways. The model-level relation is the more conservative
    of the two -- it forgives more -- which is the right direction for a tool that tells people to
    delete code, so it is the one kept. `modular_parents()` is consulted directly for the one-hop
    case so a finding is never blamed on a transitive path that does not exist.
    """
    if a == b:
        return True
    parents = modular_parents()
    if b in parents.get(a, ()) or a in parents.get(b, ()):
        return True
    return b in ancestors(a, parents) or a in ancestors(b, parents)


def duplicates(processors: list[Processor], include_declarative: bool = False) -> list[Duplicate]:
    """
    Groups of processors that are the same variant, have identical bodies, and are unrelated.

    Grouped *per backend*, because a torchvision body and a PIL body of the same policy are written
    against different libraries and can never hash equal -- comparing across backends would find
    nothing and hide the real within-backend duplication.

    `include_declarative` is off by default and is the guard against the degenerate comparison: a
    class with no method bodies hashes to the same empty string as every other such class, so
    "identical bodies" would be trivially true for the 39 pure-attribute processors. They are still a
    real (and easier) finding, but they are a different one, so `declarative_clusters` reports them
    separately.
    """
    groups: dict[tuple[str, str, str], list[Processor]] = defaultdict(list)
    for processor in processors:
        if processor.is_declarative and not include_declarative:
            continue
        groups[(processor.variant, processor.backend, processor.body_hash)].append(processor)

    found: list[Duplicate] = []
    for (variant, backend, body_hash), members in sorted(groups.items()):
        models = sorted({p.model for p in members})
        if len(models) < 2:
            continue
        # Keep only the members that are not already related to the oldest holder: a child that
        # declares `class ChildImageProcessor(ParentImageProcessor)` in a modular file is *already*
        # reusing the code, and the generated file repeating it is the converter doing its job.
        dates = build_date_data()
        oldest = min(models, key=lambda m: (dates.get(m, "9999-99-99"), m))
        unrelated = [p for p in members if p.model == oldest or not _related(p.model, oldest)]
        if len({p.model for p in unrelated}) < 2:
            continue
        found.append(Duplicate(variant, backend, unrelated, body_hash))
    return sorted(found, key=lambda d: -len(d.models))


def declarative_clusters(processors: list[Processor]) -> dict[str, list[Processor]]:
    """
    `{variant: pure-attribute processors}` for variants held by two or more *unrelated* models.

    These are the cheapest dedup in the library: nothing but class attributes differ, so the whole
    class can become `class XImageProcessor(YImageProcessor)` plus the tier-2 lines that changed.

    Related models are dropped for the same reason as in `duplicates`: a child that already declares
    `class ChildImageProcessor(ParentImageProcessor)` in a modular file is reusing the parent
    correctly, and the generated file repeating the attributes is the converter doing its job.
    """
    groups: dict[str, list[Processor]] = defaultdict(list)
    for processor in processors:
        if processor.is_declarative:
            groups[processor.variant].append(processor)
    clusters: dict[str, list[Processor]] = {}
    dates = build_date_data()
    for variant, members in groups.items():
        models = sorted({p.model for p in members})
        if len(models) < 2:
            continue
        oldest = min(models, key=lambda m: (dates.get(m, "9999-99-99"), m))
        unrelated = [p for p in members if p.model == oldest or not _related(p.model, oldest)]
        if len({p.model for p in unrelated}) > 1:
            clusters[variant] = unrelated
    return clusters


def body_variant_disagreements(processors: list[Processor]) -> list[tuple[str, list[Processor]]]:
    """
    Groups whose *bodies* are byte-identical but whose *variants* differ -- the facets over-split.

    The mirror image of `near_misses`, and the more actionable one. Identical code with a different
    facet vector means the two classes declare policies their shared implementation does not
    distinguish, i.e. one of the two declarations is wrong. `glm_image` is the live example: its
    `_preprocess` is byte-identical to `qwen2_vl`'s, but it declares an extra `images_per_sample`
    output that the shared code never builds.
    """
    groups: dict[tuple[str, str], list[Processor]] = defaultdict(list)
    for processor in processors:
        if not processor.is_declarative:
            groups[(processor.backend, processor.body_hash)].append(processor)
    found = []
    for key, members in sorted(groups.items()):
        if len({p.model for p in members}) > 1 and len({p.variant for p in members}) > 1:
            found.append((key[1], members))
    return found


def near_misses(processors: list[Processor]) -> dict[tuple[str, str], int]:
    """
    `{(variant, backend): distinct bodies}` where one variant hides several implementations.

    The facet-equal-but-code-different case. It is not a duplication finding -- it is the honest
    admission that facets are a lossy proxy, and a shortlist of variants whose axis set is too
    coarse to describe what the code does.
    """
    bodies: dict[tuple[str, str], set[str]] = defaultdict(set)
    for processor in processors:
        if not processor.is_declarative:
            bodies[(processor.variant, processor.backend)].add(processor.body_hash)
    return {key: len(values) for key, values in bodies.items() if len(values) > 1}


def backend_parity_gaps(processors: list[Processor]) -> dict[str, dict[str, str]]:
    """
    `{model: {backend: variant}}` for models whose two backends implement different policies.

    This is the payoff for putting the backend in tier 2. The torchvision and PIL classes are meant
    to be one policy expressed twice, so any model appearing under two variants is a parity bug --
    a user switching backend gets different pixels.
    """
    by_model: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for processor in processors:
        by_model[processor.model][processor.backend].add(processor.variant)
    gaps: dict[str, dict[str, str]] = {}
    for model, backends in sorted(by_model.items()):
        if len(backends) < 2:
            continue
        # A model can ship several processors per backend (hybrid/high-res twins); only flag a model
        # where no variant is shared, i.e. the two backends agree on nothing.
        torchvision, pil = backends.get("torchvision_backend", set()), backends.get("pil_backend", set())
        if torchvision and pil and not (torchvision & pil):
            gaps[model] = {
                "torchvision_backend": "; ".join(sorted(torchvision)),
                "pil_backend": "; ".join(sorted(pil)),
            }
    return gaps


def _selfcheck() -> None:
    """Assert the facts that were hand-verified by reading the processors named below."""
    # Vocabularies must actually resolve; an empty one silently turns every facet into "custom".
    assert normalization_vocabulary(), "mean/std vocabulary came back empty"
    assert normalization_vocabulary()[(0.5, 0.5, 0.5)] == "imagenet_standard"
    assert normalization_vocabulary()[(0.48145466, 0.4578275, 0.40821073)] == "openai_clip"
    by_name, by_int = resample_vocabulary()
    assert "BICUBIC" in by_name and "BILINEAR" in by_name, sorted(by_name)
    # The int table must cover every filter the library supports. If a new one is added to
    # `pil_torch_interpolation_mapping` this fails rather than silently dropping bare-int users.
    assert set(by_name) <= set(_PIL_RESAMPLING_VALUES), set(by_name) - set(_PIL_RESAMPLING_VALUES)
    # `slanext` writes `resample = 2  # PILImageResampling.BILINEAR`: the one place the repo states
    # the int meaning in prose, and the anchor for the hardcoded table above.
    assert by_int[2] == "bilinear" and by_int[3] == "bicubic", by_int
    assert "coco_detection" in annotation_format_vocabulary(), annotation_format_vocabulary()

    processors = scan_repo()
    at = {(p.model, p.class_name): p for p in processors}

    # clip: shortest-edge resize then a fixed square centre crop, CLIP's own stats, bicubic.
    clip = at[("clip", "CLIPImageProcessor")]
    assert clip.tier1["pipeline"] == "standard_pipeline", clip.tier1
    assert clip.tier1["resize"] == "resize_shortest_edge", clip.tier1
    assert clip.tier1["crop"] == "center_crop_fixed", clip.tier1
    assert clip.tier1["channels"] == "convert_rgb", clip.tier1
    assert clip.tier1["outputs"] == "pixel_values_only", clip.tier1
    assert clip.tier2["norm_stats"] == "openai_clip", clip.tier2
    assert clip.tier2["resample"] == "bicubic", clip.tier2
    # clip is *not* declarative: its `__init__` carries a KOSMOS-2 `use_square_size` shim. That shim
    # is the reason `is_trivial_override` insists on a single statement.
    assert not clip.is_declarative and clip.methods == frozenset({"__init__"}), clip.methods
    assert at[("vit", "ViTImageProcessor")].is_declarative, "ViTImageProcessor is pure attributes"

    # siglip: same square resize as vit, but converts to RGB and uses bicubic. The three of them are
    # the canonical demonstration that tier 2 is where the library's real duplication lives.
    siglip = at[("siglip", "SiglipImageProcessor")]
    vit = at[("vit", "ViTImageProcessor")]
    assert siglip.tier1["resize"] == vit.tier1["resize"] == "resize_to_square", (siglip.tier1, vit.tier1)
    assert siglip.tier1["crop"] == vit.tier1["crop"] == "no_center_crop"
    assert vit.tier2["norm_stats"] == "imagenet_standard" and vit.tier2["resample"] == "bilinear", vit.tier2
    assert siglip.variant != vit.variant, "siglip converts to RGB, vit does not: tier 1 must see it"

    # detr: aspect-preserving resize capped by a longest edge, pad to the batch maximum, and return
    # a pixel mask -- the three facets that make the whole detection family one variant.
    detr = at[("detr", "DetrImageProcessor")]
    assert detr.tier1["resize"] == "resize_shortest_edge_capped", detr.tier1
    assert detr.tier1["pad"] == "pad_to_batch_max", detr.tier1
    assert "pixel_mask" in detr.tier1["outputs"], detr.tier1
    assert detr.tier1["annotations"] == "coco_detection_annotations", detr.tier1

    # qwen2_vl: resize onto a patch grid, then flatten to patch tokens and report the grid.
    qwen2_vl = at[("qwen2_vl", "Qwen2VLImageProcessor")]
    assert qwen2_vl.tier1["tiling"] == "patch_grid_tokens", qwen2_vl.tier1
    assert qwen2_vl.tier1["resize"] == "resize_patch_grid", qwen2_vl.tier1
    assert "image_grid_thw" in qwen2_vl.tier1["outputs"], qwen2_vl.tier1
    assert qwen2_vl.tier2["norm_stats"] == "openai_clip", qwen2_vl.tier2

    # idefics3: splits into a grid of tiles and keeps a pixel attention mask.
    idefics3 = at[("idefics3", "Idefics3ImageProcessor")]
    assert idefics3.tier1["tiling"] == "grid_split_tiles", idefics3.tier1
    assert "pixel_attention_mask" in idefics3.tier1["outputs"], idefics3.tier1

    # llava: plain square resize, CLIP stats, centre crop -- deliberately *not* a tiling processor,
    # unlike llava_next, which picks a resolution from `image_grid_pinpoints`.
    llava = at[("llava", "LlavaImageProcessor")]
    assert llava.tier1["tiling"] == "no_tiling", llava.tier1
    llava_next = at[("llava_next", "LlavaNextImageProcessor")]
    assert llava_next.tier1["tiling"] == "pinpoint_tiles", llava_next.tier1

    # sam: pads to a fixed square and carries segmentation-style post-processing, so it must not
    # land in detr's pad-to-batch-max variant.
    sam = at[("sam", "SamImageProcessor")]
    assert sam.tier1["pad"] == "pad_to_fixed_size", sam.tier1
    assert sam.tier1["resize"] == "resize_longest_edge", sam.tier1

    # siglip2 is the only processor with a *patch budget* rather than a resolution, which is why it
    # is the one that needs `spatial_shapes` on the way out.
    siglip2 = at[("siglip2", "Siglip2ImageProcessor")]
    assert siglip2.tier1["tiling"] == "variable_patch_budget", siglip2.tier1

    # pix2struct returns no `pixel_values` at all: flattened patches plus their attention mask.
    pix2struct = at[("pix2struct", "Pix2StructImageProcessor")]
    assert pix2struct.tier1["tiling"] == "flattened_patches", pix2struct.tier1
    assert "flattened_patches" in pix2struct.tier1["outputs"], pix2struct.tier1

    # convnext declares `crop_pct`, so its resize target is derived; levit does the identical thing
    # with `int((256 / 224) * shortest_edge)` hardcoded in an overridden `resize` and declares
    # nothing. The two must land on different resize facets -- that disagreement is the finding, and
    # collapsing it would hide the one processor that should simply declare `crop_pct` instead.
    convnext = at[("convnext", "ConvNextImageProcessor")]
    assert convnext.tier1["resize"] == "resize_shortest_edge_upscaled", convnext.tier1
    levit = at[("levit", "LevitImageProcessor")]
    assert levit.tier1["resize"] == "resize_shortest_edge", levit.tier1
    assert levit.methods == frozenset({"resize"}), levit.methods

    # Every facet must resolve to a value from a closed vocabulary. An unexpected value means the
    # extractor is either merging variants that differ or splitting one that does not.
    expected = {
        "pipeline": {"standard_pipeline", "custom_preprocess", "custom_orchestration"},
        "tiling": {
            "no_tiling",
            "patch_grid_tokens",
            "flattened_patches",
            "dynamic_aspect_tiles",
            "pinpoint_tiles",
            "aspect_ratio_tiles",
            "pan_and_scan",
            "slice_grid",
            "grid_split_tiles",
            "variable_patch_budget",
            "custom_tiling",
        },
        "resize": {
            "no_resize",
            "resize_patch_grid",
            "resize_multiple_of",
            "resize_shortest_edge_capped",
            "resize_shortest_edge",
            "resize_longest_edge",
            "resize_max_height_width",
            "resize_to_square",
            "resize_shortest_edge_upscaled",
            "resize_size_from_config",
        },
        "annotations": {"no_annotations", "binary_mask_annotations", "segmentation_maps", "segmentation_maps_reduced"}
        | {f"{name}_annotations" for name in annotation_format_vocabulary()},
        "pad": {
            "no_pad",
            "pad_unflagged",
            "pad_to_square",
            "pad_to_fixed_size",
            "pad_to_multiple",
            "pad_to_batch_max",
        },
        "crop": {"no_center_crop", "center_crop_fixed"},
        "photometric": {"rescale_and_normalize", "rescale_only", "normalize_only", "no_photometric"},
        "channels": {"convert_rgb", "keep_input_channels", "to_grayscale", "flip_channel_order"},
    }
    for processor in processors:
        for axis, allowed in expected.items():
            value = processor.tier1[axis]
            assert value in allowed, f"{processor.model}/{processor.class_name}: {axis}={value!r} outside vocabulary"
        assert processor.tier1["outputs"].startswith("pixel_values"), processor.tier1["outputs"]
        # No facet may be the string "unknown": it would merge processors that differ.
        assert "unknown" not in processor.variant, f"{processor.model}/{processor.class_name}: {processor.variant}"

    # `custom_*` is a legitimate escape hatch but must stay rare; if it climbs, a real facet is being
    # missed and processors that differ are being merged under one tag.
    bespoke = sum(1 for p in processors if p.tier1["tiling"] == "custom_tiling")
    assert bespoke / len(processors) < 0.10, f"{bespoke}/{len(processors)} processors fell through to custom_tiling"

    # Literal stats must resolve through `utils/constants.py`, not read as bespoke. idefics writes
    # CLIP's numbers out by hand under a local alias; if this stops resolving, the biggest dedup
    # axis in the library goes quiet.
    idefics = at[("idefics", "IdeficsImageProcessor")]
    assert idefics.tier2["norm_stats"] == "openai_clip", idefics.tier2
    custom_stats = sum(1 for p in processors if p.tier2["norm_stats"] == "custom_stats")
    assert custom_stats < 0.15 * len(processors), f"{custom_stats} processors have unresolvable normalisation stats"

    # Both backends must be represented, or the parity check is measuring nothing.
    backends = {p.backend for p in processors}
    assert backends == {"torchvision_backend", "pil_backend"}, backends

    # The two findings this tool exists to produce, pinned so a facet change cannot silence them.
    #
    # `bit` and `chinese_clip` declare the same policy *and* the same constants as clip, with no code
    # of their own -- the strongest duplication in the library, and the case the whole tier-2 report
    # is built to surface. chinese_clip is the canonical owner by date.
    bit, chinese_clip = at[("bit", "BitImageProcessor")], at[("chinese_clip", "ChineseCLIPImageProcessor")]
    assert bit.variant == chinese_clip.variant == clip.variant, (bit.variant, chinese_clip.variant)
    assert bit.tier2 == chinese_clip.tier2 == clip.tier2, (bit.tier2, chinese_clip.tier2)
    assert bit.is_declarative and chinese_clip.is_declarative
    clusters = declarative_clusters(processors)
    assert any({"bit", "chinese_clip"} <= {p.model for p in members} for members in clusters.values()), sorted(
        {p.model for members in clusters.values() for p in members}
    )

    # fuyu's torchvision processor puts `image_sizes` in its `BatchFeature` and its PIL processor does
    # not, so switching backend changes the dict the downstream `FuyuProcessor` receives. That is the
    # bug that putting the backend in tier 2 was designed to expose.
    assert "fuyu" in backend_parity_gaps(processors), sorted(backend_parity_gaps(processors))

    # `glm_image` declares an `images_per_sample` output but shares qwen2_vl's byte-identical
    # `_preprocess`, which never builds that key: identical code, disagreeing declarations.
    disagreements = {p.model for _, members in body_variant_disagreements(processors) for p in members}
    assert "glm_image" in disagreements, sorted(disagreements)

    variants = build_variants(processors)
    singletons = sum(1 for v in variants.values() if len(v.owners) == 1)
    print(
        f"selfcheck ok: {len(processors)} image processors "
        f"({sum(1 for p in processors if p.backend == 'torchvision_backend')} torchvision / "
        f"{sum(1 for p in processors if p.backend == 'pil_backend')} pil), "
        f"{len(variants)} variants, {singletons} singletons, "
        f"{len(duplicates(processors))} duplicate groups, {len(backend_parity_gaps(processors))} parity gaps"
    )


if __name__ == "__main__":
    _selfcheck()
