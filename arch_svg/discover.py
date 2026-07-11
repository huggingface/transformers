"""Enumerate models under ``transformers/models/*`` -- no imports of the models themselves.

Discovery is cheap and import-free: we only stat the directory tree and read the
``model_type`` from the config mapping. Heavy work (building the model on meta) happens
later, per-model, so a single broken model can never crash the whole batch.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from functools import lru_cache


# Subpackages under models/ that are not actual models (utilities, tokenizers-only, etc.).
_NON_MODELS = {"auto", "__pycache__"}


@dataclass
class ModelEntry:
    """A discovered model subpackage. Purely path-derived; nothing is imported yet."""

    name: str  # directory name, e.g. "gemma"
    path: str  # absolute path to the model subpackage
    modeling_files: list[str]  # modeling_*.py present
    modular_file: str | None  # modular_*.py if present
    config_files: list[str]  # configuration_*.py present

    @property
    def has_modular(self) -> bool:
        return self.modular_file is not None


@lru_cache(maxsize=1)
def models_root() -> str:
    """Locate ``transformers/models`` inside the *installed* source tree."""
    spec = importlib.util.find_spec("transformers")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Could not locate the installed `transformers` package.")
    return os.path.join(list(spec.submodule_search_locations)[0], "models")


def discover_models(root: str | None = None) -> list[ModelEntry]:
    """Return every model subpackage under ``transformers/models/`` sorted by name."""
    root = root or models_root()
    entries: list[ModelEntry] = []
    for name in sorted(os.listdir(root)):
        if name in _NON_MODELS or name.startswith("_") or name.startswith("."):
            continue
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        files = os.listdir(path)
        modeling = sorted(f for f in files if f.startswith("modeling_") and f.endswith(".py"))
        configs = sorted(f for f in files if f.startswith("configuration_") and f.endswith(".py"))
        modular = next((f for f in sorted(files) if f.startswith("modular_") and f.endswith(".py")), None)
        if not modeling and not modular and not configs:
            continue  # tokenizer/processor-only subpackage
        entries.append(
            ModelEntry(
                name=name,
                path=path,
                modeling_files=modeling,
                modular_file=modular,
                config_files=configs,
            )
        )
    return entries


@lru_cache(maxsize=1)
def _config_mapping_keys() -> dict[str, str]:
    """Map model directory name heuristics to model_type via the config mapping.

    Returns a dict {model_type: model_type}; the keys ARE the model_types registered in
    the library. Kept import-light: only imports the auto config module.
    """
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    return dict(CONFIG_MAPPING_NAMES)


def model_type_for(entry: ModelEntry) -> str | None:
    """Best-effort resolution of a directory name to a registered ``model_type``.

    The directory name usually equals the model_type; when it does not (e.g. dir
    ``kosmos2`` -> ``kosmos-2``, ``x_clip`` -> ``xclip``), fall back to normalized matching
    against the config mapping (strip ``-``/``_`` and compare). For dirs that back several
    model_types (``data2vec`` -> ``data2vec-{audio,text,vision}``) the first is chosen.
    """
    keys = list(_config_mapping_keys())
    name = entry.name
    if name in keys:
        return name
    # exact match after hyphen/underscore unification
    for mt in keys:
        if mt.replace("-", "_") == name:
            return mt

    def norm(s: str) -> str:
        return s.replace("-", "").replace("_", "")

    target = norm(name)
    candidates = [mt for mt in keys if norm(mt) == target]
    if not candidates:
        # dir name is a prefix of exactly one (or more) model_types, e.g. lasr -> lasr_ctc
        candidates = [mt for mt in keys if norm(mt).startswith(target)]
    if candidates:
        # avoid obviously auxiliary types; prefer the shortest / alphabetically first
        candidates = [c for c in candidates if "privacy" not in c and "filter" not in c] or candidates
        return sorted(candidates, key=lambda c: (len(c), c))[0]
    return None
