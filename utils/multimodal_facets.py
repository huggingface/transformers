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
Composition-variant facet extraction for the library's multimodal models.

`blocks_facets.py` answers "which attention/mlp/layer does each model use". It stops one level
below the question people actually ask about a VLM, which is: *which vision tower, which connector,
and how do the image features get into the text stream*. This module is that layer. It deliberately
reuses `blocks_facets`' machinery (`scan_repo`, `build_date_data`, `modular_parents`, `ancestors`,
`copied_from_sources`, `canonical_source`, `forwards_match`) rather than re-deriving any of it: the
block-level facets of a vision tower are *already* computed there, so a tower is identified here by
provenance -- who wrote it -- and its internals are looked up, not recomputed.

Three axes, because a multimodal model is three decisions:

- **tower**: which model provides the image (or audio) encoder.
- **connector**: the mechanism that maps encoder features into the text embedding space.
- **merge**: how the projected features enter the text stream.

Tiers follow `blocks_facets`' rule -- tier 1 changes behaviour and decides identity, tier 2 is
incidental and reported but never gating -- applied to the *composition*:

- **tier 1** is `(tower, connector, merge)`. Each is a decision a model author makes in code and
  cannot walk back with a config flag. Swapping the tower rewrites `get_image_features` (siglip is
  called `self.vision_tower(pixel_values).last_hidden_state`, qwen2_vl's is
  `self.visual(pixel_values, grid_thw)` and returns already-merged patches); swapping the connector
  mechanism replaces a module; swapping the merge replaces the contract with the text model.
- **tier 2** is `binding`, `tower_output`, `feature_select` and the connector's internal shape
  (`connector_norm`, `connector_act`, `connector_bias`, `connector_params`). Each is either
  `__init__`-only or *read from config at runtime*: `vision_feature_layer` and
  `vision_feature_select_strategy` are config attributes in every model that has them, so one
  `forward` already serves both settings and they cannot gate inheritance. `binding` -- whether the
  tower arrives via `AutoModel.from_config`, a modular subclass, a `# Copied from` fork or bespoke
  code -- is the most interesting tier-2 facet in the file and the reason it is reported loudly:
  it is exactly the axis on which "we reimplemented siglip again" hides.

Facets nominate, source decides. Facet equality is a lossy proxy here for the same reason it was for
blocks: two models can agree on all three tier-1 axes and still have different image paths (aria and
llava_onevision both read "siglip + mlp + masked_scatter"). So a claimed match is confirmed by
comparing the canonicalised *image path* -- `get_image_features` plus the merge statements lifted out
of `forward` -- with attribute names normalised first. Without that normalisation the comparison is
worthless: the same code is spelled `self.multi_modal_projector`, `self.connector`,
`self.mm_projector`, `self.merger`, `self.aligner`, `self.projector` and `self.vision_projection`
across the library, and the tower is `self.vision_tower`, `self.vision_model`, `self.visual`,
`self.image_tower` or `self.vision_encoder`. Seven connector spellings and five tower spellings for
one concept: absorbing that variance is not cosmetic, it is the measurement.

Canonical owner is the oldest holder, from `blocks_facets.build_date_data()`, so lineage advice
points backwards in history and stays stable.

Stdlib only, same as `blocks_facets`: nothing here may import torch, because both files are meant to
be callable from `make check-repo`.
"""

import ast
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path


if str(Path(__file__).parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent))

from blocks_facets import (  # noqa: E402
    MODELS_ROOT,
    ancestors,
    build_date_data,
    canonical_source,
    copied_from_sources,
    generates_modeling,
    is_cross_model_import,
    modular_parents,
    parent_from_module,
    scan_repo,
)


AUTO_ROOT = MODELS_ROOT / "auto"

# The two authoritative lists. Parsed with `ast`, never imported: `modeling_auto` pulls in torch and
# this module is meant to be cheap enough to run in a consistency check.
MAPPING_NAMES = (
    "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",
    "MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES",
)

# Axis order. `blocks_facets` fits its order against measured override cost; there is no equivalent
# corpus here (a modular file that overrides *only* the connector and nothing else is rare enough
# that any fit would be noise, the same reason `MOE_AXES` is left in semantic order there). So this
# is semantic order, expensive first: the tower is a whole encoder, the merge is the contract with
# the text model, the connector is the smallest of the three to rewrite.
COMPOSITION_AXES = ("tower", "connector", "merge")
# Reported, never gating. See the module docstring for why each one is tier 2.
COMPOSITION_TIER2_AXES = (
    "binding",
    "tower_output",
    "feature_select",
    "connector_norm",
    "connector_act",
    "connector_bias",
    "connector_params",
)

# The connector is also registered on its own, because "which projector designs exist" is a
# question with a different answer from "which compositions exist" -- one connector design is shared
# by models with different towers.
CONNECTOR_AXES = ("mechanism", "norm", "downsample")
CONNECTOR_TIER2_AXES = ("act", "bias", "depth", "params")


# --------------------------------------------------------------------------------------------------
# The authoritative model lists
# --------------------------------------------------------------------------------------------------
def _mapping_pairs(node: ast.AST, known: dict[str, list[tuple[str, str]]]) -> list[tuple[str, str]]:
    """
    The `(model_type, class_name)` pairs of one `OrderedDict([...])` mapping literal.

    `MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES` starts with `*list(MODEL_FOR_IMAGE_TEXT_TO_TEXT_...
    .items())`, so a plain literal walk sees 12 entries instead of 95. The `Starred` branch splices
    in whatever earlier mapping the star refers to, which is why `known` is threaded through.
    """
    if not (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.List)):
        return []
    pairs: list[tuple[str, str]] = []
    for element in node.args[0].elts:
        if isinstance(element, ast.Tuple) and len(element.elts) == 2 and isinstance(element.elts[0], ast.Constant):
            pairs.append((element.elts[0].value, ast.unparse(element.elts[1]).strip("'\"")))
        elif isinstance(element, ast.Starred):
            for sub in ast.walk(element):
                if isinstance(sub, ast.Name) and sub.id in known:
                    pairs.extend(known[sub.id])
    return pairs


@cache
def _auto_mappings() -> dict[str, list[tuple[str, str]]]:
    """Every `*_MAPPING_NAMES` literal in `modeling_auto.py`, in declaration order."""
    tree = ast.parse((AUTO_ROOT / "modeling_auto.py").read_text(encoding="utf-8"))
    known: dict[str, list[tuple[str, str]]] = {}
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        for target in targets:
            if isinstance(target, ast.Name) and target.id.endswith("_MAPPING_NAMES"):
                known[target.id] = _mapping_pairs(node.value, known)
    return known


@cache
def multimodal_models() -> dict[str, str]:
    """`{model_type: head class name}` for the union of the two multimodal mappings."""
    known = _auto_mappings()
    found: dict[str, str] = {}
    for name in MAPPING_NAMES:
        found.update(dict(known.get(name, [])))
    return found


@cache
def automodel_classes() -> dict[str, str]:
    """
    `{model_type: base model class}` from `MODEL_MAPPING_NAMES` -- the registry `AutoModel` consults.

    Needed because `AutoModel.from_config(config.vision_config)` names no class at all, and a
    connector that lives *inside* the tower (qwen2_vl's merger, glm4v's downsample) is unreachable
    until the tower has been resolved to a concrete class. Reading the same table `AutoModel` reads
    means the resolution cannot disagree with what actually gets built.
    """
    return dict(_auto_mappings().get("MODEL_MAPPING_NAMES", []))


# --------------------------------------------------------------------------------------------------
# Where classes and `model_type` strings live
# --------------------------------------------------------------------------------------------------
@cache
def _parsed(path: Path) -> tuple[ast.Module, str] | None:
    try:
        source = path.read_text(encoding="utf-8")
        return ast.parse(source), source
    except (OSError, SyntaxError):
        return None


@cache
def class_index() -> dict[str, tuple[str, ...]]:
    """`{class name: directories whose modeling files define it}`, in directory order."""
    index: dict[str, list[str]] = defaultdict(list)
    for model_dir in sorted(p for p in MODELS_ROOT.iterdir() if p.is_dir()):
        for path in sorted(model_dir.glob("modeling_*.py")):
            parsed = _parsed(path)
            if parsed is None:
                continue
            for node in parsed[0].body:
                if isinstance(node, ast.ClassDef) and model_dir.name not in index[node.name]:
                    index[node.name].append(model_dir.name)
    return {name: tuple(dirs) for name, dirs in index.items()}


@cache
def model_type_owners() -> dict[str, str]:
    """
    `{model_type: directory}`, read off the `model_type = "..."` line in every config class.

    Taken from the config classes rather than `CONFIG_MAPPING_NAMES` because that mapping is
    assembled at import time out of a generated `auto_mappings` module plus two manual `.update()`
    calls, and one of them re-binds the name. The declaration on the class cannot drift.
    """
    owners: dict[str, str] = {}
    for model_dir in sorted(p for p in MODELS_ROOT.iterdir() if p.is_dir()):
        for path in sorted(model_dir.glob("configuration_*.py")):
            parsed = _parsed(path)
            if parsed is None:
                continue
            for node in parsed[0].body:
                if not isinstance(node, ast.ClassDef):
                    continue
                for statement in node.body:
                    if (
                        isinstance(statement, ast.Assign)
                        and len(statement.targets) == 1
                        and isinstance(statement.targets[0], ast.Name)
                        and statement.targets[0].id == "model_type"
                        and isinstance(statement.value, ast.Constant)
                    ):
                        owners.setdefault(statement.value.value, model_dir.name)
    return owners


@cache
def config_class_model_types() -> dict[str, str]:
    """
    `{config class name: its declared model_type}`.

    Resolves the other `sub_configs` spelling: `"vision_config": InternVLVisionConfig` names a class,
    not a `model_type`, and the tower has to be identified the same way in both cases or the two
    spellings count as two different towers.
    """
    types: dict[str, str] = {}
    for model_dir in sorted(p for p in MODELS_ROOT.iterdir() if p.is_dir()):
        for path in sorted(model_dir.glob("configuration_*.py")):
            parsed = _parsed(path)
            if parsed is None:
                continue
            for node in parsed[0].body:
                if not isinstance(node, ast.ClassDef):
                    continue
                for statement in node.body:
                    if (
                        isinstance(statement, ast.Assign)
                        and len(statement.targets) == 1
                        and isinstance(statement.targets[0], ast.Name)
                        and statement.targets[0].id == "model_type"
                        and isinstance(statement.value, ast.Constant)
                    ):
                        types.setdefault(node.name, statement.value.value)
    return types


@cache
def modular_class_bases() -> dict[tuple[str, str], tuple[str, str]]:
    """
    `{(model, class): (base model, base class)}` for every cross-model base in a modular file.

    `blocks_facets.modular_class_edges` counts these per model pair; here the individual edge is
    needed, because the question is which model a *specific* tower class descends from.
    """
    edges: dict[tuple[str, str], tuple[str, str]] = {}
    for model_dir in sorted(p for p in MODELS_ROOT.iterdir() if p.is_dir()):
        for path in sorted(model_dir.glob("modular_*.py")):
            parsed = _parsed(path)
            if parsed is None:
                continue
            imported: dict[str, str] = {}
            for node in ast.walk(parsed[0]):
                if not is_cross_model_import(node):
                    continue
                parent = parent_from_module(node.module)
                # `from ...modeling_utils import PreTrainedModel` is a level-3 import too, so
                # `is_cross_model_import` accepts it and `parent_from_module` hands back
                # `modeling_utils` -- which is not a model. Without this guard emu3's tower family
                # came out as `modeling_utils`, declared `modular_subclass`.
                if parent is None or not (MODELS_ROOT / parent).is_dir():
                    continue
                for alias in node.names:
                    imported[alias.asname or alias.name] = parent
            for node in parsed[0].body:
                if not isinstance(node, ast.ClassDef):
                    continue
                for base in node.bases:
                    name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
                    if name in imported and imported[name] != model_dir.name:
                        edges.setdefault((model_dir.name, node.name), (imported[name], name))
    return edges


# --------------------------------------------------------------------------------------------------
# Naming variance the extractor has to absorb, or the variant count inflates
# --------------------------------------------------------------------------------------------------
# Five spellings of "the encoder" and one concept. `encoder` is in the list because pix2struct,
# udop, granite_speech and pp_formulanet use exactly that, but it is also what every text stack
# calls its own body, so it only counts when the right-hand side names a vision/audio class -- see
# `_TOWER_RHS_RE`. `image_tower`/`video_tower` are video_llava's, which is the only model that
# splits the two.
_TOWER_ATTRS = (
    "vision_tower",
    "vision_model",
    "visual",
    "vision_encoder",
    "image_encoder",
    "image_tower",
    "video_tower",
    "audio_tower",
    "audio_encoder",
    "audio_model",
    "vision_embed_tokens",
    "embed_vision",
    "high_res_vision_model",
    "sam_encoder",
    "vqmodel",
    "vision_backbone",
    "protein_encoder",
    "encoder",
)
_TOWER_ATTR_RE = re.compile(rf"^({'|'.join(_TOWER_ATTRS)})$")
# Evidence from the right-hand side, used both to rescue unconventional attribute names and to veto
# `self.encoder = SomeTextStack(...)`.
_TOWER_RHS_RE = re.compile(r"(Vision|Visual|Image|Audio|Speech|Acoustic|Sam|VQVAE|Protein)", re.IGNORECASE)

# Seven spellings of "the connector". Missing any one of them re-counts a shared design as unique,
# which is the specific way this measurement goes wrong.
_CONNECTOR_ATTRS = (
    "multi_modal_projector",
    "multimodal_projector",
    "mm_projector",
    "connector",
    "perceiver_resampler",
    "resampler_model",
    "sequence_compressor_resampler",
    "vision_projection",
    "visual_projection",
    "language_projection",
    "image_to_text_projection",
    "modality_projection",
    "projector",
    "layerwise_projectors",
    "spatial_projectors",
    "merger",
    "merger_list",
    "deepstack_merger_list",
    "vit_merger",
    "vision_resampler",
    "aligner",
    "generation_aligner",
    "vision_adapter",
    "vision_proj",
    "patch_merger",
    "enc_to_dec_proj",
    "head_linear",
    # blip-2 and instructblip put a whole Q-Former between the tower and the text model. Reading
    # only their `language_projection` reported `linear_single`, i.e. the last 5% of the connector.
    "qformer",
    "multimodal_embedder",
    "embed_audio",
    # The gemma3n/gemma4 family calls its projector an *embedder* and files it under
    # `embed_vision`; fuyu calls its patch projection `vision_embed_tokens`. Both names are also in
    # `_TOWER_ATTRS`, deliberately: for gemma4_unified and fuyu the module genuinely is both the
    # tower and the connector, and the tower search takes the first non-audio entry in `__init__`
    # order so gemma3n still files under its real `vision_tower`.
    "embed_vision",
    "vision_embed_tokens",
    "downsample",
)
_CONNECTOR_ATTR_RE = re.compile(rf"^({'|'.join(_CONNECTOR_ATTRS)})$")
# The projections that belong to an attention block, never to a connector.
_BLOCK_PROJ_RE = re.compile(r"^(q|k|v|o|out|up|down|gate|gate_up|in|c)_proj$")
# A tower's *input* stage, which reads like a connector and is not one. This is why the connector
# test stays an explicit name list rather than the two-sided name-or-right-hand-side test `_is_tower`
# uses: `Gemma4VisionPatchEmbedder`, `Gemma4VisionPooler`, `Gemma3nAudioSubSampleConvProjection` and
# `VoxtralRealtimeEmbedder` all instantiate classes whose names read exactly like connectors, and no
# right-hand-side pattern can tell them from one. They sit at the *front* of the tower, so accepting
# them would report a model's patch embedding as its projector into text space.
_TOWER_STAGE_ATTRS = ("patch_embedder", "patch_embed", "pooler", "subsample_conv_projection", "embedder")

_TEXT_ATTRS = ("language_model", "text_model", "model", "decoder", "thinker", "llm", "text_decoder", "layers")

# Attribute-name normalisation, applied to source before any byte comparison. Without this the
# library's seven connector spellings make every model's image path look unique.
_ATTR_NORMALISATION = (
    (rf"self\.({'|'.join(_TOWER_ATTRS)})\b", "self.TOWER"),
    (rf"self\.({'|'.join(_CONNECTOR_ATTRS)})\b", "self.CONNECTOR"),
    (rf"self\.({'|'.join(_TEXT_ATTRS)})\b", "self.TEXT"),
)


def normalise_attributes(source: str) -> str:
    """Rewrite every spelling of tower/connector/text-model to one canonical name."""
    for pattern, replacement in _ATTR_NORMALISATION:
        source = re.sub(pattern, replacement, source)
    return source


# --------------------------------------------------------------------------------------------------
# Finding the composite class
# --------------------------------------------------------------------------------------------------
def _self_assignments(class_node: ast.ClassDef) -> list[tuple[str, str]]:
    """`(attribute, unparsed value)` for every `self.x = ...` anywhere in the class's `__init__`."""
    init = next((n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    if init is None:
        return []
    found: list[tuple[str, str]] = []
    for node in ast.walk(init):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "self"
        ):
            found.append((node.targets[0].attr, ast.unparse(node.value)))
    return found


def _is_tower(attr: str, rhs: str) -> bool:
    """
    Whether `self.<attr> = <rhs>` installs an encoder for a non-text modality.

    Two-sided on purpose. The attribute name alone accepts `self.encoder = UdopStack(...)` in every
    text model that happens to be listed; the right-hand side alone accepts
    `self.visual_projection = nn.Linear(...)`, a connector. Requiring the name to be a known tower
    spelling *and* the value to name a vision/audio class keeps `encoder` usable for the four models
    that really do call their image encoder that.
    """
    if not _TOWER_ATTR_RE.match(attr):
        return False
    if attr == "encoder":
        return bool(_TOWER_RHS_RE.search(rhs))
    return True


def _is_connector(attr: str, rhs: str) -> bool:
    if _BLOCK_PROJ_RE.match(attr) or attr in _TOWER_STAGE_ATTRS:
        return False
    if not _CONNECTOR_ATTR_RE.match(attr):
        return False
    # `self.downsample` is a connector only when it is a real module; several towers use the name for
    # an integer stride.
    return attr != "downsample" or "nn." in rhs


@dataclass
class Composite:
    """The class that owns both an encoder for another modality and a text stream."""

    model: str
    path: Path
    class_name: str
    node: ast.ClassDef
    source: str
    towers: list[tuple[str, str]]
    connectors: list[tuple[str, str]]


def _as_composite(model: str, path: Path, source: str, node: ast.ClassDef) -> Composite | None:
    """`node` as a `Composite` if it wires a non-text encoder to a text stream, else `None`."""
    assignments = _self_assignments(node)
    towers = [(a, r) for a, r in assignments if _is_tower(a, r)]
    if not towers or not any(a in _TEXT_ATTRS for a, _ in assignments):
        return None
    connectors = [(a, r) for a, r in assignments if _is_connector(a, r)]
    return Composite(model, path, node.name, node, source, towers, connectors)


def find_composite(model: str) -> Composite | None:
    """
    The class in `model`'s own directory that wires an encoder to a text stream.

    Anchored on the auto mapping's head class and then walked *down* through `self.model` /
    `self.thinker`, because that is the library's actual layout (`XForConditionalGeneration` holds
    `self.model`, and `XModel` holds the tower and the connector). A purely structural search finds
    the right class for most models but picks `Emu3VQVAE` for emu3 -- a VQ-VAE has an `encoder` and a
    `decoder`, which satisfies "an encoder plus a text-ish attribute" by coincidence. Anchoring
    removes that whole class of false positive.

    The structural search is kept as the fallback, because the head class's name is not a reliable
    entry point either: it is `LlavaModel`, `Llama4ForConditionalGeneration`,
    `Qwen2_5OmniThinkerForConditionalGeneration`, `T5Gemma2Encoder` and
    `Pix2StructForConditionalGeneration` across five models. In the fallback, preference goes to the
    class holding the most towers, so gemma3n's three-modality `Gemma3nModel` wins over a
    single-tower wrapper in the same file.
    """
    directory = MODELS_ROOT / model
    if not directory.is_dir():
        return None
    classes: dict[str, tuple[Path, str, ast.ClassDef]] = {}
    for path in sorted(directory.glob("modeling_*.py")):
        parsed = _parsed(path)
        if parsed is None:
            continue
        tree, source = parsed
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.setdefault(node.name, (path, source, node))

    head = multimodal_models().get(model)
    seen: set[str] = set()
    frontier = [head] if head in classes else []
    while frontier:
        name = frontier.pop(0)
        if name in seen:
            continue
        seen.add(name)
        path, source, node = classes[name]
        composite = _as_composite(model, path, source, node)
        if composite is not None:
            return composite
        for attr, rhs in _self_assignments(node):
            if attr not in ("model", "thinker", "language_model"):
                continue
            call = re.match(r"(\w+)", rhs)
            if call and call.group(1) in classes:
                frontier.append(call.group(1))

    best: Composite | None = None
    for name, (path, source, node) in classes.items():
        candidate = _as_composite(model, path, source, node)
        if candidate is not None and (best is None or len(candidate.towers) > len(best.towers)):
            best = candidate
    return best


def head_class_location(model: str) -> tuple[str, str] | None:
    """
    `(class name, directory that defines it)` for the model's entry in the auto mapping.

    A model whose head class is defined in *another* directory has no code of its own at all --
    shieldgemma2's entry is `Gemma3Model` and pp_chart2table's is `GotOcr2Model`. That is the
    strongest form of reuse in the library and it would otherwise be invisible.
    """
    class_name = multimodal_models().get(model)
    if class_name is None:
        return None
    dirs = class_index().get(class_name, ())
    if not dirs:
        return None
    return class_name, (model if model in dirs else dirs[0])


# --------------------------------------------------------------------------------------------------
# A. The tower: which model provides it, and how it is bound
# --------------------------------------------------------------------------------------------------
@cache
def sub_config_defaults(model: str) -> dict[str, str]:
    """
    `{sub-config key: default `model_type`}` for one composite config.

    Three spellings, and all three have to work or reuse is undercounted:

    1. `sub_configs = {"vision_config": InternVLVisionConfig}` -- a concrete class, whose own
       `model_type` declaration identifies the tower.
    2. `self.vision_config.get("model_type", "siglip_vision_model")` -- the default a dict config
       falls back to. This is the *only* spelling in deepseek_vl and diffusion_gemma, and the reason
       their towers first read `config_driven`: they never subscript `CONFIG_MAPPING` with a literal.
    3. `CONFIG_MAPPING["clip_vision_model"](...)` -- the `vision_config is None` branch.

    Spelling 2 is also what makes llava's tower swappable at load time, so this is not pedantry:
    reading only `sub_configs` there yields `AutoConfig`, i.e. nothing.
    """
    directory = MODELS_ROOT / model
    resolved: dict[str, str] = {}
    if not directory.is_dir():
        return resolved
    owners, class_types = model_type_owners(), config_class_model_types()
    for path in sorted(directory.glob("configuration_*.py")):
        parsed = _parsed(path)
        if parsed is None:
            continue
        for node in parsed[0].body:
            if not isinstance(node, ast.ClassDef):
                continue
            for statement in node.body:
                if not (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == "sub_configs"
                    and isinstance(statement.value, ast.Dict)
                ):
                    continue
                for key, value in zip(statement.value.keys, statement.value.values):
                    if not isinstance(key, ast.Constant):
                        continue
                    name = ast.unparse(value)
                    if name != "AutoConfig" and name in class_types:
                        resolved.setdefault(key.value, class_types[name])
            # Both `__init__` and `__post_init__` are read: composite configs were migrated to
            # dataclass-style bodies with a `__post_init__`, and looking only for `__init__` found
            # nothing at all in llava, paligemma and every other migrated config.
            statements = [
                statement
                for method in node.body
                if isinstance(method, ast.FunctionDef) and method.name in ("__init__", "__post_init__")
                for statement in method.body
            ]
            if not statements:
                continue
            block = ast.Module(body=statements, type_ignores=[])
            for sub in ast.walk(block):
                model_type = _declared_model_type(sub, owners)
                if model_type is None:
                    continue
                # Attribute the default to the sub-config key whose name it is nearest to in the
                # source; in practice each default sits inside an `if isinstance(self.<key>, dict)`
                # branch, so scan enclosing statements.
                for key in _enclosing_sub_config_keys(statements, sub):
                    resolved.setdefault(key, model_type)
    return resolved


def _declared_model_type(node: ast.AST, owners: dict[str, str]) -> str | None:
    """A literal `model_type` named by `CONFIG_MAPPING["x"]` or by `.get("model_type", "x")`."""
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        if node.value.id == "CONFIG_MAPPING" and isinstance(node.slice, ast.Constant):
            value = node.slice.value
            return value if isinstance(value, str) and value in owners else None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "model_type"
        and isinstance(node.args[1], ast.Constant)
    ):
        value = node.args[1].value
        return value if isinstance(value, str) and value in owners else None
    return None


def _enclosing_sub_config_keys(statements: list[ast.stmt], target: ast.AST) -> list[str]:
    """
    Which `self.<x>_config` the statement containing `target` is about.

    A composite config's initialiser is a run of near-identical `if isinstance(self.vision_config,
    dict): ... CONFIG_MAPPING[...]` blocks, one per modality. The `CONFIG_MAPPING` key alone does not
    say which modality it belongs to, so the enclosing top-level statement is searched for the
    `self.<something>_config` it mentions.
    """
    for statement in statements:
        if target not in set(ast.walk(statement)):
            continue
        keys = []
        for sub in ast.walk(statement):
            if isinstance(sub, ast.Attribute) and sub.attr.endswith("_config"):
                keys.append(sub.attr)
        # Nearest-first: the branch's own subject is the attribute it tests, i.e. the first one.
        return list(dict.fromkeys(keys))
    return []


_TOWER_INTERNAL_RE = re.compile(r"(Vision|Visual|Image|Audio|Speech)")


@cache
def has_own_tower(model: str) -> bool:
    """Whether a model defines any vision/audio module class of its own -- i.e. can *be* a tower."""
    return any(_TOWER_INTERNAL_RE.search(name) for name, dirs in class_index().items() if model in dirs)


def _copied_from_provenance(model: str, tower_class: str) -> str | None:
    """
    The model a directly-declared tower was forked from, via its parts' `# Copied from` markers.

    A tower's wrapper class usually carries no marker -- `Idefics3VisionTransformer` has none -- but
    its attention, MLP and encoder-layer do, and they point at siglip. Voting over the parts is the
    only way to see that idefics3's "own" tower is a siglip fork, which is precisely the kind of
    reuse this registry exists to surface.

    Candidates are restricted to models that own a vision or audio module themselves. Without that
    filter, `# Copied from transformers.models.t5...Pix2StructVisionMlp` made **t5** the tower family
    of pix2struct and kosmos-2.5 -- true at the level of one gated MLP, and nonsense at the level of
    "which vision tower does this model use". Borrowing a block from a text model is a block-level
    fact, and `blocks_facets` is where it belongs.
    """
    sources = copied_from_sources()
    votes: dict[str, int] = defaultdict(int)
    for (owner, symbol), source_model in sources.items():
        if owner != model or source_model == model or not has_own_tower(source_model):
            continue
        if _TOWER_INTERNAL_RE.search(symbol) or symbol.startswith(tower_class):
            votes[source_model] += 1
    if not votes:
        return None
    return max(sorted(votes), key=lambda m: votes[m])


def tower_provenance(model: str, attr: str, rhs: str) -> tuple[str, str]:
    """
    `(tower family, binding)` for one `self.<attr> = <rhs>`.

    Binding values are ordered strongest reuse first:
    `automodel_pluggable`  -- `AutoModel.from_config(config.vision_config)`: the tower is whatever
                              the checkpoint says, and the default comes from the config.
    `imported_class`       -- the tower class is defined in another model's directory outright.
    `modular_subclass`     -- declared in a `modular_*.py` as a subclass of another model's tower.
    `copied_from`          -- a fork whose parts carry `# Copied from` markers.
    `bespoke`              -- no traceable relationship: a genuinely new encoder, or an untracked
                              reimplementation. The two are indistinguishable from source alone,
                              which is the point.
    """
    call = re.match(r"(\w+)\.(from_config|_from_config)\(", rhs) or re.match(r"(\w+)\(", rhs)
    symbol = call.group(1) if call else ""

    if symbol in ("AutoModel", "AutoModelForCausalLM", "AutoModelForVision2Seq"):
        # `AutoModel.from_config(config.<key>)` -- the key names the sub-config that decides.
        key = next((m for m in re.findall(r"config\.(\w+_config)", rhs)), None)
        model_type = sub_config_defaults(model).get(key or "", "")
        family = model_type_owners().get(model_type, "") or "config_driven"
        return family, "automodel_pluggable"

    if not symbol or symbol.startswith("nn."):
        # `self.vision_embed_tokens = nn.Linear(...)`: fuyu has no tower at all, it projects raw
        # patches. Named rather than left unknown so it groups honestly.
        return "raw_patch_projection", "bespoke"

    owners = class_index().get(symbol, ())
    if owners and model not in owners:
        return owners[0], "imported_class"

    edge = modular_class_bases().get((model, symbol))
    if edge is not None:
        return edge[0], "modular_subclass"

    copied = _copied_from_provenance(model, symbol)
    if copied is not None:
        return copied, "copied_from"
    return model, "bespoke"


# --------------------------------------------------------------------------------------------------
# B. The connector
# --------------------------------------------------------------------------------------------------
# Config attributes that name a token-count reduction. The value is the *attribute*, not a number,
# because the number lives in a checkpoint -- and the attribute name is self-describing anyway.
_DOWNSAMPLE_KEYS = (
    "scale_factor",
    "downsample_ratio",
    "spatial_merge_size",
    "spatial_pool_stride",
    "pool_size",
    "mm_tokens_per_image",
    "pixel_shuffle_factor",
    "merge_size",
    "projector_patch_to_query_dict",
    "resampler_n_latents",
)
# Config attribute -> normalised token in the facet value, so `spatial_merge_size` and `merge_size`
# do not read as two different mechanisms.
_DOWNSAMPLE_NAMES = {
    "scale_factor": "shuffle_scale_factor",
    "pixel_shuffle_factor": "shuffle_scale_factor",
    "downsample_ratio": "shuffle_downsample_ratio",
    "spatial_merge_size": "spatial_merge",
    "merge_size": "spatial_merge",
    "spatial_pool_stride": "pool_stride",
    "pool_size": "pool_stride",
    "mm_tokens_per_image": "pool_to_fixed_token_count",
    "projector_patch_to_query_dict": "resample_to_learned_queries",
    "resampler_n_latents": "resample_to_learned_queries",
}

_CONFIG_SHORTHAND = (
    (r"config\.vision_config\.hidden_size", "vision_hidden"),
    (r"config\.vision_config\.\w+", "vision_cfg"),
    (r"config\.text_config\.hidden_size", "text_hidden"),
    (r"config\.text_config\.\w+", "text_cfg"),
    (r"config\.audio_config\.\w+", "audio_cfg"),
    (r"self\.\w+", "attr"),
    (r"config\.\w+", "cfg"),
)


def _shorthand(expression: str) -> str:
    """Collapse a layer's size expression to a readable shape, e.g. `vision_hidden*4 -> text_hidden`."""
    for pattern, replacement in _CONFIG_SHORTHAND:
        expression = re.sub(pattern, replacement, expression)
    return re.sub(r"\s+", "", expression)


def connector_facets(source: str) -> tuple[dict, dict]:
    """
    Facet one connector class body.

    Tier 1 is `(mechanism, norm, downsample)`: all three change the module's `forward`. Tier 2 is
    the activation, the bias source, the layer count and the parameter shape -- `__init__` detail
    that a model routinely forks a whole class over without changing what the class computes.
    """
    linears = re.findall(r"nn\.Linear\(([^)]*)\)", source, re.DOTALL)
    has = lambda *needles: any(n in source for n in needles)  # noqa: E731

    # Mechanism, most specific first. A resampler that also has an MLP is a resampler; a
    # pixel-shuffle that also has a linear is a pixel-shuffle. Ordering by specificity is what keeps
    # these from collapsing into "has a linear".
    if has("QFormer", "query_length", "cross_attention_frequency"):
        # blip-2 / instructblip. A Q-Former is a stack of cross-attending blocks driven by learned
        # query tokens, but the tokens are declared on the *composite* (`self.query_tokens`) and the
        # `nn.Linear`s are two class levels down, so neither the learned-query test nor the linear
        # count sees it from the Q-Former's own body. Its name and its two distinctive config
        # attributes do, and they are unambiguous.
        mechanism = "qformer_cross_attention"
    elif has("latents", "query_tokens") or re.search(r"self\.quer(y|ies)\s*=\s*nn\.Parameter", source):
        mechanism = "perceiver_resampler_learned_queries"
    elif has("pixel_shuffle", "pixel_unshuffle"):
        mechanism = "pixel_shuffle"
    elif has("functional.unfold", "F.unfold", "nn.Unfold"):
        mechanism = "unfold_patch_merge"
    elif has("nn.AvgPool2d", "adaptive_avg_pool", "avg_pool2d", "nn.MaxPool2d", "spatial_pool_mode"):
        mechanism = "pooling"
    elif has("nn.Conv2d", "nn.Conv1d"):
        mechanism = "conv_downsample"
    elif re.search(r"self\.\w*(gate_proj|up_proj)\s*=", source):
        mechanism = "gated_mlp"
    elif re.search(r"nn\.Parameter", source) and "matmul" in source:
        mechanism = "weight_matmul"
    elif has("nn.Sequential") or len(linears) >= 3:
        mechanism = "mlp_deep"
    elif len(linears) == 2:
        mechanism = "mlp_2layer"
    elif len(linears) == 1:
        mechanism = "linear_single"
    else:
        # A wrapper that only calls sub-modules; its mechanism is whatever they are, and this value
        # says so rather than pretending the connector does nothing.
        mechanism = "submodule_wrapper"

    if re.search(r"nn\.LayerNorm|LayerNorm\(", source):
        norm = "layernorm"
    elif re.search(r"RMSNorm\(", source):
        norm = "rmsnorm"
    else:
        norm = "no_norm"

    keys = [k for k in _DOWNSAMPLE_KEYS if k in source]
    downsample = _DOWNSAMPLE_NAMES.get(keys[0], keys[0]) if keys else "no_downsample"
    if downsample == "no_downsample" and mechanism in ("pooling", "conv_downsample", "unfold_patch_merge"):
        # The mechanism is intrinsically a reduction even when the stride is not read from a
        # recognised config key.
        downsample = "implicit_" + mechanism

    tier1 = {"mechanism": mechanism, "norm": norm, "downsample": downsample}
    activations = re.findall(r"ACT2FN\[([^\]]+)\]|nn\.(GELU|SiLU|ReLU|Tanh)\(", source)
    act = next((a or b for a, b in activations if a or b), "")
    tier2 = {
        "act": _shorthand(act) if act else "no_act",
        "bias": "bias_from_config"
        if "bias=config" in source.replace(" ", "") or re.search(r"bias=\w*config", source)
        else ("bias_false" if "bias=False" in source else ("bias_true" if "bias=True" in source else "bias_default")),
        "depth": f"{len(linears)}_linear" if linears else "0_linear",
        "params": " -> ".join(
            _shorthand(a.split(",")[0]) + "," + _shorthand(a.split(",")[1]) for a in linears[:2] if a.count(",") >= 1
        )
        or "no_linear",
    }
    return tier1, tier2


# --------------------------------------------------------------------------------------------------
# C. The merge: how features enter the text stream
# --------------------------------------------------------------------------------------------------
# Ordered most specific first. `masked_scatter` on a placeholder mask is the library's convention and
# has to be tested before the plain-index form, because a model that does both (a compile-friendly
# branch plus a fallback) should read as the convention it follows.
_MERGE_PATTERNS = (
    ("masked_scatter_on_placeholder", r"masked_scatter"),
    ("index_assign_on_placeholder", r"(inputs_embeds|hidden_states)\[[^\]]*(mask|indices|idx)[^\]]*\]\s*="),
    ("index_put_on_placeholder", r"index_put|index_copy|index_fill|\.scatter_?\("),
    ("where_on_placeholder", r"torch\.where\([^)]*(mask|special)"),
    # Cross-attention keeps the image out of the text sequence entirely and feeds it as keys/values.
    # Two spellings: the generic `encoder_hidden_states=` and mllama's `cross_attention_states=`.
    #
    # `image_hidden_states=` is deliberately NOT here even though idefics2/3 use that name. It is
    # their variable for the projected features, so `image_hidden_states = image_hidden_states.to(...)`
    # -- a dtype cast one line above an ordinary `masked_scatter` -- made both models, and smolvlm
    # with them, read as cross-attention. Three of the library's most-copied VLMs mislabelled by one
    # over-general alternative in one regex.
    (
        "cross_attention",
        r"(encoder_hidden_states|cross_attention_states)\s*=\s*"
        r"(image|vision|visual|audio|projected|cross|\w*_features|\w*_hidden_states)",
    ),
    ("concat_prefix", r"torch\.cat\(\s*\[?\s*[\(\[]?\s*(image|vision|visual|audio|projected|\w*_features)"),
)
# An encoder-decoder VLM has no merge site at all: the image encoder's output *is* the encoder
# output, and the decoder cross-attends to it. Detected from the wiring rather than from a
# statement, because there is no statement to find -- which is exactly why pix2struct,
# vision-encoder-decoder and pp_formulanet first read `no_explicit_merge`.
_ENCODER_DECODER_RE = re.compile(r"self\.decoder\s*\(|encoder_outputs\s*=|encoder_hidden_states\s*=\s*hidden_states")


def merge_facets(image_path: str, whole_class: str, has_connector: bool) -> tuple[dict, dict]:
    """
    How the composite writes projected features into the text embedding stream.

    Read from the merge region of the composite's `forward` first and only then from the whole
    class: several models keep a legacy `_merge_input_ids_with_image_features` around that is no
    longer called, and letting the whole class decide would report the dead path.

    The discrete case is tested *first*, before any pattern. chameleon and emu3 quantise the image to
    token ids and let the text embedding table do the lookup -- they do call `masked_scatter`, but on
    *ids*, not on features, so reporting them as ordinary masked-scatter models hides a genuinely
    fourth strategy. The `has_connector` guard is what keeps janus out of this branch: janus has a
    VQ-VAE for generation *and* a continuous aligner for understanding, and it is the aligner that
    describes its image path.
    """
    if not has_connector and ("vocabulary_mapping" in whole_class or "get_image_tokens" in whole_class):
        return {"merge": "discrete_token_ids"}, {}
    for name, pattern in _MERGE_PATTERNS:
        if re.search(pattern, image_path):
            return {"merge": name}, {}
    for name, pattern in _MERGE_PATTERNS:
        if re.search(pattern, whole_class):
            return {"merge": name}, {}
    if _ENCODER_DECODER_RE.search(whole_class):
        return {"merge": "encoder_decoder_cross_attention"}, {}
    return {"merge": "no_explicit_merge"}, {}


# Every method that is part of the image path, by name. Taken from a census of the method names
# actually in use rather than guessed: `get_image_features` (147 definitions), `get_placeholder_mask`
# (76), `get_video_features` (53), `get_audio_features` (28), plus idefics2/3/smolvlm/aria's
# `inputs_merger` and the two surviving `_merge_input_ids_with_*` helpers. `get_text_features` is
# excluded on purpose: it is the text half.
_FEATURE_METHOD_RE = re.compile(
    r"^(get_(?!text_)\w*(features|mask)|get_placeholder_mask|inputs_merger|_merge_input_ids_with_\w+)$"
)


def image_path_source(composite: Composite) -> str:
    """
    The canonicalised image path of a composite: its feature getters plus the merge lines.

    This is the "source decides" half of the design. It is deliberately *not* the whole `forward`:
    a composite's forward also threads cache, position ids and generation kwargs, and two models
    that merge identically routinely differ there. Restricting the comparison to the feature
    getters and the statements that touch the features is what makes a byte-identical verdict mean
    "these two merge the same way" instead of "these two are the same file".
    """
    chunks: list[str] = []
    for node in composite.node.body:
        if isinstance(node, ast.FunctionDef) and _FEATURE_METHOD_RE.match(node.name):
            try:
                chunks.append(canonical_source(node))
            except (SyntaxError, ValueError):
                continue
    forward = next((n for n in composite.node.body if isinstance(n, ast.FunctionDef) and n.name == "forward"), None)
    if forward is not None:
        lines = composite.source.splitlines()
        body = "\n".join(lines[forward.lineno - 1 : forward.end_lineno])
        # Only the lines that mention features, a mask or the connector: the merge, and nothing else.
        chunks.append(
            "\n".join(
                line.strip()
                for line in body.splitlines()
                if re.search(r"features|special_\w*_mask|placeholder_mask|masked_scatter|CONNECTOR|projector", line)
            )
        )
    joined = normalise_attributes("\n".join(chunks))
    # Strip the model's own name so `LlavaModel` and `LlavaNextModel` compare on structure, exactly
    # as `blocks_facets.canonical_method` does for blocks.
    squashed = composite.model.replace("_", "")
    return re.sub(rf"\b{re.escape(squashed)}", "X", joined, flags=re.IGNORECASE)


# --------------------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------------------
@dataclass
class Connector:
    """One connector module found in one model."""

    model: str
    class_name: str
    attr: str
    tier1: dict = field(default_factory=dict)
    tier2: dict = field(default_factory=dict)
    forward: str | None = None

    @property
    def variant(self) -> str:
        return "|".join(str(self.tier1.get(axis, "?")) for axis in CONNECTOR_AXES)

    @property
    def tag(self) -> str:
        return f"connector:{self.variant}"


@dataclass
class Composition:
    """One multimodal model's (tower, connector, merge) triple."""

    model: str
    head_class: str
    head_owner: str
    composite_class: str
    modality: str
    tier1: dict = field(default_factory=dict)
    tier2: dict = field(default_factory=dict)
    connectors: list[Connector] = field(default_factory=list)
    # The canonicalised image path. Facets nominate; this confirms.
    image_path: str | None = None

    @property
    def variant(self) -> str:
        return "|".join(str(self.tier1.get(axis, "?")) for axis in COMPOSITION_AXES)

    @property
    def tag(self) -> str:
        return f"composition:{self.variant}"

    def tier2_delta(self, other: "Composition") -> dict[str, tuple[str, str]]:
        return {k: (v, other.tier2.get(k, "?")) for k, v in self.tier2.items() if other.tier2.get(k, "?") != v}


_MODALITY_RE = re.compile(r"(audio|speech|acoustic|semantic_tokenizer)", re.IGNORECASE)


def _modality(towers: list[tuple[str, str]]) -> str:
    """`vision`, `audio` or `vision+audio`, from the tower attributes present."""
    kinds = set()
    for attr, rhs in towers:
        kinds.add("audio" if _MODALITY_RE.search(attr) or _MODALITY_RE.search(rhs) else "vision")
    return "+".join(sorted(kinds)) or "none"


def _tower_output(image_path: str, whole_class: str) -> str:
    """
    How the composite gets features out of the image path -- the shape of the plumbing, not the
    tower's own return type.

    Three shapes are in use, and the distinction is worth reporting: 64 models follow the modern
    convention of routing everything through `get_image_features(...)` and reading `.pooler_output`
    off the `BaseModelOutputWithPooling` it returns; 15 select a layer with
    `config.vision_feature_layer`; 8 read the tower's `.last_hidden_state` directly.

    Tier 2, and reported rather than gating, because it co-varies with the tower -- knowing the
    family already tells you the call shape -- and because the dominant value is a library-wide
    convention rather than an architectural choice.
    """
    text = image_path + whole_class
    if re.search(r"vision_feature_layer|hidden_states\[[^\]]*layer", text):
        return "feature_layer_from_config"
    if "pooler_output" in text:
        return "pooler_output"
    if "last_hidden_state" in text:
        return "last_hidden_state"
    return "direct_call"


def build_compositions() -> list[Composition]:
    """One `Composition` per model in the two authoritative mappings, skipping those with no tower."""
    compositions: list[Composition] = []
    for model in sorted(multimodal_models()):
        head = head_class_location(model)
        if head is None:
            continue
        head_class, head_owner = head
        composite = find_composite(model)
        if composite is None and head_owner != model:
            # The model has no composite of its own because its head class *is* another model's.
            composite = find_composite(head_owner)
        if composite is None:
            continue

        towers = composite.towers
        # The primary tower is the first non-audio one where both exist, so a vision+audio model
        # files under its vision tower -- the axis the question is about.
        primary = next((t for t in towers if not _MODALITY_RE.search(t[0])), towers[0])
        family, binding = tower_provenance(composite.model, *primary)

        located = [(composite.model, attr, rhs) for attr, rhs in composite.connectors]
        if not located:
            # qwen2_vl, glm4v and their descendants put the merger *inside* the tower, so a search
            # that stops at the composite reports "no connector" for a whole family. Descend.
            located = _connectors_inside(composite.model, primary[1])
        connectors = [_connector_record(directory, attr, rhs) for directory, attr, rhs in located]
        connectors = [c for c in connectors if c is not None]

        image_path = image_path_source(composite)
        merge_tier1, _ = merge_facets(image_path, composite.source, bool(connectors))
        mechanism = connectors[0].tier1["mechanism"] if connectors else "no_connector"
        first = connectors[0] if connectors else None

        compositions.append(
            Composition(
                model=model,
                head_class=head_class,
                head_owner=head_owner,
                composite_class=f"{composite.model}.{composite.class_name}",
                modality=_modality(towers),
                tier1={"tower": family, "connector": mechanism, **merge_tier1},
                tier2={
                    "binding": binding,
                    "tower_output": _tower_output(image_path, composite.source),
                    "feature_select": "strips_cls_token"
                    if "vision_feature_select_strategy" in composite.source
                    else "keeps_all_tokens",
                    "connector_norm": first.tier1["norm"] if first else "no_connector",
                    "connector_act": first.tier2["act"] if first else "no_connector",
                    "connector_bias": first.tier2["bias"] if first else "no_connector",
                    "connector_params": first.tier2["params"] if first else "no_connector",
                },
                connectors=connectors,
                image_path=image_path,
            )
        )
    return compositions


_NOT_MODELLING_RE = re.compile(
    r"^(configuration_|processing_|image_processing_|video_processing_|tokenization_|feature_extraction_|convert_|__init__)"
)


def _model_sources(model: str) -> list[Path]:
    """
    Every file in a model's directory that can define a module class.

    Not `modeling_*.py`: idefics keeps `IdeficsPerceiverResampler` in `perceiver.py` and its vision
    tower in `vision.py`, so globbing only `modeling_*` reported idefics as having no connector when
    it has the library's first perceiver resampler.
    """
    return [p for p in sorted((MODELS_ROOT / model).glob("*.py")) if not _NOT_MODELLING_RE.match(p.name)]


def _find_class(model: str, class_name: str) -> tuple[ast.ClassDef, str] | None:
    for path in _model_sources(model):
        parsed = _parsed(path)
        if parsed is None:
            continue
        tree, source = parsed
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node, source
    return None


def _class_body(model: str, class_name: str) -> str | None:
    found = _find_class(model, class_name)
    if found is None:
        return None
    node, source = found
    return "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])


def tower_class(model: str, tower_rhs: str) -> tuple[str, str] | None:
    """
    `(directory, class name)` of the tower module a composite actually builds.

    `AutoModel.from_config(config.vision_config)` names no class, so the concrete tower is looked up
    through the same `MODEL_MAPPING_NAMES` table `AutoModel` itself consults. Skipping this step
    reported "no connector" for qwen3_vl, glm46v, cosmos3_omni, gemma3n and every other model that
    binds its tower through `AutoModel` *and* keeps the merger inside it -- 21 models with a
    connector that the registry could not see.
    """
    call = re.match(r"([\w.]+)\.(from_config|_from_config)\(", tower_rhs) or re.match(r"([\w.]+)\(", tower_rhs)
    if not call:
        return None
    symbol = call.group(1)
    if symbol.startswith("nn."):
        return None
    if symbol.startswith("AutoModel"):
        key = next((m for m in re.findall(r"config\.(\w+_config)", tower_rhs)), None)
        model_type = sub_config_defaults(model).get(key or "", "")
        resolved = automodel_classes().get(model_type)
        family = model_type_owners().get(model_type)
        return (family, resolved) if family and resolved else None
    owners = class_index().get(symbol, ())
    if not owners:
        return None
    return (model if model in owners else owners[0]), symbol


def _connectors_inside(model: str, tower_rhs: str) -> list[tuple[str, str, str]]:
    """
    `(directory, attribute, value)` for connectors declared inside the tower class.

    The directory travels with the result because the tower can belong to another model: cosmos3_omni
    binds qwen3_vl's tower, so its merger class is `Qwen3VLVisionPatchMerger` and lives in qwen3_vl's
    directory, not in cosmos3_omni's.
    """
    located = tower_class(model, tower_rhs)
    if located is None:
        return []
    directory, symbol = located
    for path in sorted((MODELS_ROOT / directory).glob("modeling_*.py")):
        parsed = _parsed(path)
        if parsed is None:
            continue
        for node in parsed[0].body:
            if isinstance(node, ast.ClassDef) and node.name == symbol:
                return [(directory, a, r) for a, r in _self_assignments(node) if _is_connector(a, r)]
    return []


def _referenced_classes(model: str, body: str) -> list[str]:
    """Classes from `model`'s own directory that `body` instantiates -- one level, no recursion."""
    local = {name for name, dirs in class_index().items() if model in dirs}
    return sorted({symbol for symbol in re.findall(r"\b([A-Z]\w+)\(", body) if symbol in local})


def _connector_record(model: str, attr: str, rhs: str) -> Connector | None:
    """Facet the class named on the right-hand side of a connector assignment."""
    call = re.match(r"(\w+)\.(from_config|_from_config)\(", rhs) or re.match(r"([\w.]+)\(", rhs)
    symbol = call.group(1) if call else ""
    if symbol.startswith("nn."):
        # `nn.ModuleList([Granite4VisionWindowQFormerDownsampler(...)])`: the container says nothing,
        # the element says everything. Reach through it before falling back to facetting the
        # assignment text, which is right for a plain inline `nn.Linear(...)` -- mllama, step3p7 and
        # deepseek_ocr2 all declare their projector that way.
        inner = next(iter(_referenced_classes(model, rhs)), None)
        if inner is not None:
            symbol = inner
        else:
            tier1, tier2 = connector_facets(rhs)
            return Connector(model, symbol, attr, tier1, tier2, forward=None)
    body = _class_body(model, symbol)
    if body is None:
        return None
    tier1, tier2 = connector_facets(body)
    if tier1["mechanism"] == "submodule_wrapper":
        # `Idefics2Connector` owns no parameters: it holds a `modality_projection` and a
        # `perceiver_resampler`, and facetting its own body says only "this class calls things". The
        # mechanism is whatever its parts are, so read them in and re-facet. Two models
        # (idefics2, minicpmv4_6) reported `submodule_wrapper` before this, which is a legend, not
        # a self-describing value.
        # Two levels: a wrapper's parts are often themselves wrappers (`Blip2QFormerModel` holds an
        # encoder which holds the layers that hold the projections).
        expanded, seen = body, {model}
        for _ in range(2):
            parts = [_class_body(model, s) or "" for s in _referenced_classes(model, expanded) if s not in seen]
            seen |= set(_referenced_classes(model, expanded))
            if not parts:
                break
            expanded = "\n".join([expanded, *parts])
        tier1, tier2 = connector_facets(expanded)
    forward = None
    found = _find_class(model, symbol)
    if found is not None:
        method = next((n for n in found[0].body if isinstance(n, ast.FunctionDef) and n.name == "forward"), None)
        if method is not None:
            try:
                squashed = model.replace("_", "")
                forward = re.sub(rf"\b{re.escape(squashed)}", "X", canonical_source(method), flags=re.IGNORECASE)
            except (SyntaxError, ValueError):
                forward = None
    return Connector(model, symbol, attr, tier1, tier2, forward)


# --------------------------------------------------------------------------------------------------
# Variant tables
# --------------------------------------------------------------------------------------------------
@dataclass
class Variant:
    """A set of records sharing one tier-1 vector, with the oldest holder as canonical owner."""

    kind: str
    variant: str
    records: list = field(default_factory=list)

    @property
    def owners(self) -> list[str]:
        return sorted({r.model for r in self.records})

    @property
    def canonical(self) -> str | None:
        """The oldest owner: inheritance should follow history, so lineage stays stable."""
        dates = build_date_data()
        dated = [(dates[m], m) for m in self.owners if m in dates]
        return min(dated)[1] if dated else (self.owners[0] if self.owners else None)

    @property
    def tag(self) -> str:
        return f"{self.kind}:{self.variant}"


def build_variants(records: list, kind: str) -> dict[str, Variant]:
    """Group records into tier-1 variants, keyed by tag. Mirrors `blocks_facets.build_variants`."""
    variants: dict[str, Variant] = {}
    for record in records:
        variant = variants.setdefault(record.tag, Variant(kind, record.variant))
        variant.records.append(record)
    return variants


def related(a: str, b: str) -> bool:
    """
    Whether two models are ancestor and descendant through modular inheritance, either direction.

    Uses `blocks_facets.ancestors`, so "related" means the same thing here as it does for blocks:
    reachable through `from ..parent.modeling_parent import ...` in a `modular_*.py`.
    """
    parents = modular_parents()
    return a == b or a in ancestors(b, parents) or b in ancestors(a, parents)


def siblings(a: str, b: str) -> bool:
    """
    Whether two models share a modular ancestor without descending from each other.

    Worth separating from `related`, because the two cases call for different action. glm4v_moe and
    glm_ocr both descend from glm4v: their duplication is already *tracked*, and the fix is to lift
    the shared code into glm4v. Two models with no common ancestor duplicating each other is a
    different problem -- nothing in the repo records that they are the same thing.
    """
    parents = modular_parents()
    return not related(a, b) and bool(ancestors(a, parents) & ancestors(b, parents))


def tracked_copy(left_model: str, left_class: str, right_model: str, right_class: str) -> bool:
    """
    Whether a `# Copied from` marker already records that one of these two classes copies the other.

    `blocks_facets` treats copy markers as the library's third reuse mechanism, and so must this:
    `utils/check_copies.py` keeps a marked class in sync with its source in CI, so a marked class is
    *managed* reuse. Reporting it as an unnoticed duplicate would send someone to write a modular
    file for a model that already tracks its source.
    """
    sources = copied_from_sources()
    return (
        sources.get((left_model, left_class)) == right_model or sources.get((right_model, right_class)) == left_model
    )


def _facet_equal_pairs(compositions: list[Composition]) -> list[tuple[Composition, Composition]]:
    """
    Every pair sharing a tier-1 variant, excluding pairs that are literally the same class.

    The exclusion is not cosmetic. Several `model_type` entries in the auto mappings point at a class
    another entry already owns -- `qwen2_5_omni` and `qwen2_5_omni_thinker` are one
    `Qwen2_5OmniThinkerForConditionalGeneration`, shieldgemma2's entry *is* `Gemma3Model`, glmga's
    *is* `Glm46VModel` -- so comparing them finds a byte-identical image path because it is the same
    bytes. Reporting that as duplicated code would be the tool lying about its own subject.
    """
    pairs: list[tuple[Composition, Composition]] = []
    by_variant: dict[str, list[Composition]] = defaultdict(list)
    for composition in compositions:
        # An *alias* -- a mapping entry whose head class belongs to another model's directory
        # (shieldgemma2 -> gemma3, pp_chart2table -> got_ocr2, glmga -> glm46v, the two omni
        # "thinker" entries) -- has no code of its own, so it can neither duplicate nor be
        # duplicated. Leaving them in produced pairs like shieldgemma2 <-> t5gemma2, which is really
        # gemma3 <-> t5gemma2 and is already tracked by t5gemma2's modular file.
        if composition.head_owner == composition.model:
            by_variant[composition.tag].append(composition)
    for group in by_variant.values():
        for i, left in enumerate(group):
            for right in group[i + 1 :]:
                if left.composite_class != right.composite_class:
                    pairs.append((left, right))
    return pairs


def blind_spots(compositions: list[Composition]) -> list[tuple[Composition, Composition]]:
    """
    Pairs sharing a tier-1 variant, not ancestor/descendant, with byte-identical image paths.

    The byte comparison is the same conservative test `blocks_facets.forwards_match` applies to
    blocks, and for the same reason: facet equality was a lossy proxy there (38 false matches) and it
    is a lossy proxy here too. An equal image path proves the two models build and merge image
    features identically; anything short of equality is a near miss and is reported separately.

    Use `siblings()` on each pair to sort the results: a sibling pair is duplication the modular DAG
    already knows about, a non-sibling pair is duplication nothing in the repo records.
    """
    return [
        (left, right)
        for left, right in _facet_equal_pairs(compositions)
        if not related(left.model, right.model) and left.image_path and left.image_path == right.image_path
    ]


def near_misses(compositions: list[Composition]) -> list[tuple[Composition, Composition]]:
    """Facet-equal, not ancestor/descendant, but *not* byte-identical -- the residue of the proxy."""
    return [
        (left, right)
        for left, right in _facet_equal_pairs(compositions)
        if not related(left.model, right.model) and left.image_path != right.image_path
    ]


def connector_blind_spots(compositions: list[Composition]) -> list[tuple[Connector, Connector]]:
    """
    Connector classes with byte-identical `forward` bodies in models that are not related.

    The composition-level test asks whether two models' whole image path is identical, which is a
    high bar: it fails as soon as one of them also handles video, or patches, or a second tower. This
    asks the narrower and more actionable question -- is this specific projector class a verbatim
    copy of one that already exists elsewhere? A `yes` is a class that could be imported instead of
    re-declared, and it is the smallest unit of the finding that someone can act on today.
    """
    records = [c for composition in compositions for c in composition.connectors if c.forward]
    pairs: list[tuple[Connector, Connector]] = []
    by_forward: dict[str, list[Connector]] = defaultdict(list)
    for record in records:
        by_forward[record.forward].append(record)
    for group in by_forward.values():
        unique: dict[str, Connector] = {}
        for record in group:
            unique.setdefault(f"{record.model}.{record.class_name}", record)
        ordered = sorted(unique.values(), key=lambda r: r.model)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                if left.model == right.model or related(left.model, right.model):
                    continue
                if tracked_copy(left.model, left.class_name, right.model, right.class_name):
                    continue
                pairs.append((left, right))
    return pairs


@dataclass
class Duplication:
    """
    One cluster of byte-identical implementations, split by whether the repo records the link.

    Pairwise reporting was the first shape of this and it was noisy: `fast_vlm` inherits llava's
    projector through a modular file and `llava_next` carries a `# Copied from` marker for the same
    class, so the pair (fast_vlm, llava_next) looked like an unrecorded duplicate when both members
    are correctly tracked -- just to different parents by different mechanisms. Clustering against
    the *oldest* holder and asking "does this member have any recorded link to that holder" removes
    that whole class of false positive and states the finding the way someone can act on it.
    """

    kind: str
    canonical: str
    canonical_class: str
    tracked: list[str] = field(default_factory=list)
    untracked: list[str] = field(default_factory=list)

    @property
    def members(self) -> list[str]:
        return sorted({self.canonical, *self.tracked, *self.untracked})


def _oldest(models: list[str]) -> str:
    dates = build_date_data()
    return min(sorted(models), key=lambda m: (dates.get(m, "9999-99-99"), m))


def _class_path(model: str, class_name: str) -> Path | None:
    """The file in `model`'s directory that declares `class_name`."""
    for path in _model_sources(model):
        parsed = _parsed(path)
        if parsed is None:
            continue
        if any(isinstance(n, ast.ClassDef) and n.name == class_name for n in parsed[0].body):
            return path
    return None


def is_tracked(model: str, class_name: str, holders: set[str]) -> bool:
    """
    Whether the repo already records that `model.class_name` is a copy of something in `holders`.

    Three mechanisms count, which is the same list `blocks_facets` works from:

    1. The file is generated from a `modular_*.py` (`blocks_facets.generates_modeling`). Every class
       in a generated file is either declared in the modular or pulled in by the converter from a
       declared dependency, so the duplication is reproduced from a single source on every
       `make fix-repo`. It is not duplication anyone has to notice.
    2. A `# Copied from` marker naming one of the holders, which `utils/check_copies.py` keeps in
       sync in CI.
    3. Modular ancestry reaching a holder -- the code came down the DAG.

    Getting this list wrong in either direction ruins the finding. Omitting (1) reported
    llava_onevision as having reinvented llava's projector when its whole modeling file is generated;
    omitting (3) reported every sibling pair twice.
    """
    path = _class_path(model, class_name)
    if path is not None and generates_modeling(path):
        return True
    if copied_from_sources().get((model, class_name)) in holders:
        return True
    reachable = ancestors(model, modular_parents())
    return bool(reachable & holders)


def _cluster(groups: dict[str, list[tuple[str, str]]], kind: str) -> list[Duplication]:
    """
    Turn `{identical body: [(model, class), ...]}` into duplication clusters.

    The canonical owner is the oldest holder, the same rule `Variant.canonical` uses, so advice
    points backwards in history. Every other holder is asked whether the repo records the link; the
    ones that do not are the finding.
    """
    found: list[Duplication] = []
    for members in groups.values():
        by_model: dict[str, str] = {}
        for model, class_name in members:
            by_model.setdefault(model, class_name)
        if len(by_model) < 2:
            continue
        canonical = _oldest(list(by_model))
        holders = set(by_model)
        duplication = Duplication(kind, canonical, by_model[canonical])
        for model, class_name in sorted(by_model.items()):
            if model == canonical:
                continue
            linked = related(model, canonical) or is_tracked(model, class_name, holders - {model})
            (duplication.tracked if linked else duplication.untracked).append(model)
        if duplication.untracked:
            found.append(duplication)
    return sorted(found, key=lambda d: (-len(d.untracked), d.canonical))


def connector_duplications(compositions: list[Composition]) -> list[Duplication]:
    """
    Connector classes whose `forward` is byte-identical across models, clustered by that body.

    The narrowest and most actionable form of the blindness finding: this specific projector class
    already exists, verbatim, somewhere else, and could be imported instead of re-declared.
    """
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for composition in compositions:
        for connector in composition.connectors:
            # Keyed on `connector.model` -- the directory the class was *found* in -- not on
            # `composition.model`. shieldgemma2's projector IS `gemma3.Gemma3MultiModalProjector`,
            # and keying on the composition made shieldgemma2 an untracked duplicate of the class it
            # literally imports. Same for the two omni "thinker" aliases.
            if connector.forward:
                groups[connector.forward].append((connector.model, connector.class_name))
    return _cluster(groups, "connector")


def composition_duplications(compositions: list[Composition]) -> list[Duplication]:
    """
    Whole image paths that are byte-identical across models -- tower call, projection and merge.

    A stronger claim than a shared connector: these models do the same thing end to end.
    """
    # Keyed on the *owner* of the composite class, so two mapping entries pointing at one class are
    # one implementation, not two; see `_facet_equal_pairs` for why that matters.
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for composition in compositions:
        if composition.image_path:
            owner, _, class_name = composition.composite_class.partition(".")
            groups[composition.image_path].setdefault(owner, class_name)
    return _cluster({body: sorted(members.items()) for body, members in groups.items()}, "composition")


_VISION_BLOCK_RE = re.compile(r"Vision|Visual|Image|Audio|Speech")
# A block in a *text* stack that happens to sit in a multimodal model's file. Excluded explicitly,
# because `Granite4VisionTextAttention` matches `Vision` and is the language model's attention.
_TEXT_BLOCK_RE = re.compile(r"Text|Language|Decoder(?!Layer)|Talker|Thinker")


@cache
def tower_families(scan: tuple[Composition, ...] | None = None) -> frozenset[str]:
    """Every model that supplies a tower to some multimodal model, plus the multimodal models."""
    compositions = scan if scan is not None else tuple(build_compositions())
    return frozenset({c.tier1["tower"] for c in compositions} | {c.model for c in compositions})


@cache
def _tower_blocks() -> dict[str, list]:
    """
    `{model: its tower blocks}`, filtered out of one `blocks_facets.scan_repo()` pass.

    Two filters, both needed. The name filter keeps `Idefics3VisionAttention` and drops
    `Granite4VisionTextAttention`. The *scope* filter keeps only models that actually supply a tower
    to something, or are multimodal themselves; without it `HiggsAudioV2Attention` -- an ordinary
    text attention in a model with "Audio" in its name -- was reported as the canonical owner of a
    vision tower's attention, which is nonsense dressed as a finding. Models that are pure encoders
    (siglip, clip, dinov2) contribute all their attention blocks, since every block they own is a
    tower block.
    """
    blocks, _ = scan_repo()
    scope = tower_families()
    pure_encoders = scope - set(multimodal_models())
    found: dict[str, list] = defaultdict(list)
    for block in blocks:
        if block.model not in scope or _TEXT_BLOCK_RE.search(block.class_name):
            continue
        if _VISION_BLOCK_RE.search(block.class_name) or block.model in pure_encoders:
            found[block.model].append(block)
    return dict(found)


def tower_block_facets(family: str) -> dict[str, list[str]]:
    """
    The block-level facets of a tower family, straight out of `blocks_facets.scan_repo()`.

    Not recomputed: a vision tower's attention and MLP are ordinary blocks and are already
    registered there, so this is a lookup that keeps one source of truth for block facets.
    """
    found: dict[str, list[str]] = defaultdict(list)
    for block in _tower_blocks().get(family, []):
        found[block.kind].append(block.variant)
    return {kind: sorted(set(variants)) for kind, variants in found.items()}


def tower_reimplementations(compositions: list[Composition]) -> list[tuple[str, str, str, str]]:
    """
    `(model, its tower class, the model it duplicates, that model's class)` for every tower whose
    vision attention `forward` is byte-identical to an unrelated model's.

    This is the tower half of the blindness question, and it is answered entirely with
    `blocks_facets`: the towers' attention blocks are already scanned and canonicalised there, so
    the test is `forwards_match` on two `Block`s. A tower declared `bespoke` whose attention is
    byte-for-byte a siglip attention is not a new encoder, it is an untracked fork -- and unlike the
    `copied_from` case, nothing in the repo says so.
    """
    declared = {c.model: c for c in compositions}
    by_forward: dict[str, list] = defaultdict(list)
    for model, blocks in _tower_blocks().items():
        for block in blocks:
            if block.kind == "attention" and block.forward:
                by_forward[block.forward].append(block)

    findings: list[tuple[str, str, str, str]] = []
    dates = build_date_data()
    for group in by_forward.values():
        models = sorted({b.model for b in group})
        if len(models) < 2:
            continue
        # The oldest holder is the one the others duplicate, same rule as `Variant.canonical`.
        oldest = min(models, key=lambda m: (dates.get(m, "9999-99-99"), m))
        holders = set(models)
        for model in models:
            if model == oldest or model not in declared or related(model, oldest):
                continue
            mine = next(b for b in group if b.model == model)
            theirs = next(b for b in group if b.model == oldest)
            # Same tracking test as `connector_duplications`: a class in a modular-generated file or
            # carrying a `# Copied from` marker is managed reuse. `Ovis2VisionAttention` is
            # byte-identical to `Siglip2Attention` and ovis2 has no relationship to siglip2 -- but
            # ovis2's file is generated from a modular that inherits from *siglip*, whose attention
            # happens to be the same body. Without this test that reads as a reimplementation.
            if is_tracked(model, mine.class_name, holders - {model}):
                continue
            findings.append((model, mine.class_name, oldest, theirs.class_name))
    return sorted(set(findings))


# --------------------------------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------------------------------
def _selfcheck() -> None:
    """Assert facts hand-verified by reading the eight models named in the plan."""
    models = multimodal_models()
    assert len(models) >= 90, f"expected ~95 multimodal models, got {len(models)}"
    assert "llava" in models and "qwen2_vl" in models and "gemma3" in models

    # The `model_type` -> owner map is what resolves `AutoModel.from_config`; if this drifts, every
    # `automodel_pluggable` tower silently becomes `config_driven`.
    owners = model_type_owners()
    assert owners["clip_vision_model"] == "clip", owners.get("clip_vision_model")
    assert owners["siglip_vision_model"] == "siglip"
    assert owners["siglip2_vision_model"] == "siglip2"

    compositions = {c.model: c for c in build_compositions()}

    # llava: clip tower via AutoModel, 2-layer MLP, masked_scatter. The reference composition.
    llava = compositions["llava"]
    assert llava.tier1 == {
        "tower": "clip",
        "connector": "mlp_2layer",
        "merge": "masked_scatter_on_placeholder",
    }, llava.tier1
    assert llava.tier2["binding"] == "automodel_pluggable", llava.tier2

    # llava_next: same triple as llava; it differs only in how many patches it feeds.
    assert compositions["llava_next"].tier1 == llava.tier1, compositions["llava_next"].tier1

    # paligemma: siglip tower, one bare Linear, masked_scatter.
    paligemma = compositions["paligemma"]
    assert paligemma.tier1["tower"] == "siglip", paligemma.tier1
    assert paligemma.tier1["connector"] == "linear_single", paligemma.tier1
    assert paligemma.tier1["merge"] == "masked_scatter_on_placeholder", paligemma.tier1

    # gemma3: siglip tower, but the projector average-pools to a fixed token count, RMS-norms and
    # multiplies by a raw Parameter -- not an MLP.
    gemma3 = compositions["gemma3"]
    assert gemma3.tier1["tower"] == "siglip", gemma3.tier1
    assert gemma3.tier1["connector"] == "pooling", gemma3.tier1
    assert gemma3.tier2["connector_norm"] == "rmsnorm", gemma3.tier2

    # idefics3 / smolvlm: pixel-shuffle connectors. Their towers are siglip forks carried by
    # `# Copied from`, not by modular inheritance -- which is the finding, not an accident.
    idefics3 = compositions["idefics3"]
    assert idefics3.tier1["connector"] == "pixel_shuffle", idefics3.tier1
    assert idefics3.tier1["tower"] == "siglip", idefics3.tier1
    assert idefics3.tier2["binding"] == "copied_from", idefics3.tier2
    smolvlm = compositions["smolvlm"]
    assert smolvlm.tier1["connector"] == "pixel_shuffle", smolvlm.tier1
    assert smolvlm.tier2["binding"] == "modular_subclass", smolvlm.tier2
    # smolvlm descends from idefics3, so however identical they are they are not a blind spot.
    assert related("smolvlm", "idefics3")

    # internvl: its own tower (`InternVLVisionConfig` is internvl's), pixel-shuffle in the
    # composite, and a LayerNorm ahead of the MLP.
    internvl = compositions["internvl"]
    assert internvl.tier1["tower"] == "internvl", internvl.tier1
    assert internvl.tier2["connector_norm"] == "layernorm", internvl.tier2

    # qwen2_vl: bespoke tower, and the connector lives *inside* it. If the descend-into-tower path
    # regresses, this reads `no_connector`.
    qwen2_vl = compositions["qwen2_vl"]
    assert qwen2_vl.tier1["tower"] == "qwen2_vl", qwen2_vl.tier1
    assert qwen2_vl.tier1["connector"] != "no_connector", qwen2_vl.tier1
    assert qwen2_vl.tier2["binding"] == "bespoke", qwen2_vl.tier2

    # The naming-variance absorption has to be load-bearing: llava spells the connector
    # `multi_modal_projector` and idefics3 spells it `connector`, and both must normalise.
    assert "self.CONNECTOR" in normalise_attributes("self.multi_modal_projector(x)")
    assert "self.CONNECTOR" in normalise_attributes("self.connector(x)")
    assert "self.TOWER" in normalise_attributes("self.visual(pixel_values)")
    assert "self.TOWER" in normalise_attributes("self.vision_tower(pixel_values)")

    # aria's tower is idefics3's: `CONFIG_MAPPING["idefics3_vision"]`, which only the
    # `.get("model_type", ...)` / `CONFIG_MAPPING[...]` reading finds. A nice check on the whole
    # config-resolution path, since nothing in aria's *modeling* file names idefics3.
    assert compositions["aria"].tier1["tower"] == "idefics3", compositions["aria"].tier1

    # video_llava spells its tower `image_tower` -- the only model that does -- and llava_next_video
    # keeps a second connector (`vision_resampler`) beside the projector. Both were invisible at
    # some point during development, so both are pinned.
    assert compositions["video_llava"].tier1["connector"] == "mlp_2layer", compositions["video_llava"].tier1
    assert len(compositions["llava_next_video"].connectors) == 2, compositions["llava_next_video"].connectors

    # chameleon quantises the image to token ids: a fourth merge strategy, and it must not be
    # reported as ordinary masked-scatter just because it calls `masked_scatter` on *ids*.
    assert compositions["chameleon"].tier1["merge"] == "discrete_token_ids", compositions["chameleon"].tier1
    # mllama cross-attends instead of splicing into the sequence.
    assert compositions["mllama"].tier1["merge"] == "cross_attention", compositions["mllama"].tier1

    # Every composition must carry a readable tag: no `?`, and no value that needs a legend.
    for composition in compositions.values():
        assert "?" not in composition.variant, (composition.model, composition.variant)
        assert not re.search(r"type_\d|kind_\d|variant_\d", composition.variant), composition.variant

    # The registry's own claims have to survive the tracking filter: every byte-identical connector
    # in the library traces to a modular file or a `# Copied from` marker, so an untracked
    # duplication report means either a genuine new find or a regression in `is_tracked`.
    records = list(compositions.values())
    for duplication in connector_duplications(records):
        assert duplication.untracked, duplication
    assert not tower_reimplementations(records), tower_reimplementations(records)

    # One tensor operation, eight bodies. If this ever drops to 1 the redundancy was fixed; if the
    # count moves for any other reason the helper scan has drifted.
    shuffles = helper_implementations("pixel_shuffle")
    assert len(shuffles) >= 5, f"expected several distinct pixel_shuffle bodies, got {len(shuffles)}"
    assert sum(len(owners) for owners in shuffles.values()) > len(shuffles), "no pixel_shuffle body is shared"

    print(f"selfcheck ok ({len(compositions)} compositions, {len(models)} models in the mappings)")


# The reshape operations connectors implement inline instead of calling a shared helper. Each is one
# well-defined tensor operation with one correct implementation, which is what makes counting the
# distinct bodies meaningful: every body beyond the first is a place the same operation was written
# again.
CONNECTOR_HELPERS = ("pixel_shuffle", "pixel_unshuffle", "space_to_depth", "unfold_patches")


def helper_implementations(helper: str = "pixel_shuffle") -> dict[str, list[str]]:
    """
    `{canonicalised body: models that define it}` for one connector helper, library-wide.

    The sharpest form of the duplication question, and the one with the clearest fix. Unlike a whole
    connector class -- where a model may legitimately want a different projection -- a pixel shuffle
    is one tensor operation. Counting how many *distinct* bodies implement it measures redundancy
    directly, with no judgement call about whether two models "should" share code.
    """
    bodies: dict[str, list[str]] = defaultdict(list)
    for model_dir in sorted(p for p in MODELS_ROOT.iterdir() if p.is_dir()):
        for path in _model_sources(model_dir.name):
            parsed = _parsed(path)
            if parsed is None:
                continue
            for node in ast.walk(parsed[0]):
                if not (isinstance(node, ast.FunctionDef) and re.fullmatch(rf"_?{helper}\w*", node.name)):
                    continue
                try:
                    squashed = model_dir.name.replace("_", "")
                    body = re.sub(rf"\b{re.escape(squashed)}", "X", canonical_source(node), flags=re.IGNORECASE)
                except (SyntaxError, ValueError):
                    continue
                if model_dir.name not in bodies[body]:
                    bodies[body].append(model_dir.name)
    return dict(bodies)


def census() -> None:
    """Print the registry: counts, the combination table, and the blindness findings."""
    compositions = build_compositions()
    models = multimodal_models()
    variants = build_variants(compositions, "composition")
    connectors = [c for composition in compositions for c in composition.connectors]

    aliases = [c.model for c in compositions if c.head_owner != c.model]
    print(f"models in the two mappings   {len(models)}")
    print(f"  with a tower + text stream {len(compositions)}")
    print(f"  no code of their own       {len(aliases)}  ({', '.join(sorted(aliases))})")
    print(f"distinct towers              {len({c.tier1['tower'] for c in compositions})}")
    print(
        f"distinct connectors          {len({c.tier1['connector'] for c in compositions})} mechanisms, "
        f"{len(build_variants(connectors, 'connector'))} full variants"
    )
    print(f"distinct merges              {len({c.tier1['merge'] for c in compositions})}")
    print(f"distinct combinations        {len(variants)}")

    print(f"\n{'n':>3}  {'owner':<18} {'tower':<18} {'connector':<32} {'merge':<32} members")
    for variant in sorted(variants.values(), key=lambda v: (-len(v.records), v.variant)):
        tower, connector, merge = variant.variant.split("|")
        members = ", ".join(sorted(r.model for r in variant.records))
        print(
            f"{len(variant.records):3d}  {str(variant.canonical):<18} {tower:<18} {connector:<32} {merge:<32} {members}"
        )

    print("\nuntracked duplication (byte-identical, no modular/copy link):")
    for duplication in connector_duplications(compositions) or [None]:
        print(f"  connector: {duplication}" if duplication else "  connector: none")
    for finding in tower_reimplementations(compositions) or [None]:
        print(f"  tower: {finding}" if finding else "  tower: none")

    print("\nsame combination, no ancestor/descendant link (facet-equal, code differs):")
    for left, right in near_misses(compositions):
        print(
            f"  {left.model:22s} {right.model:22s} {'sibling' if siblings(left.model, right.model) else '-':8s} {left.variant}"
        )

    print("\none operation, many implementations:")
    for helper in CONNECTOR_HELPERS:
        bodies = helper_implementations(helper)
        if not bodies:
            continue
        holders = {model for owners in bodies.values() for model in owners}
        print(f"  {helper}: {len(bodies)} distinct bodies across {len(holders)} models")
        for owners in sorted(bodies.values(), key=lambda v: (-len(v), v)):
            print(f"      {sorted(owners)}")


if __name__ == "__main__":
    if "census" in sys.argv[1:]:
        census()
    else:
        _selfcheck()
