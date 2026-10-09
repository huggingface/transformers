"""Select the contracts a Transformers change can affect, from the files it changes.

A contract's code is found from its fixtures, not declared by hand: the
model_type and architectures in each fixture's config.json, and the tokenizer,
feature extractor, image processor, and processor classes in its files or in
the Auto mappings for that model_type, each resolved to its directory under
src/transformers/models/. A changed file then selects:

  src/transformers/models/<dir>/...        the contracts whose code lives in <dir>
  any other file under src/transformers/   every contract (shared code, models/auto included)
  setup.py, pyproject.toml                 every contract (dependencies)
  anything else (tests, docs, utils, ...)  nothing
"""

import json
from pathlib import Path

SHARED = ("setup.py", "pyproject.toml")
PROCESSOR_FILES = ("tokenizer_config.json", "preprocessor_config.json", "processor_config.json")
CLASS_KEYS = ("tokenizer_class", "processor_class", "feature_extractor_type", "image_processor_type")


def module_dir(name):
    """models/<dir> holding a Transformers class, or None if it lives in shared code."""
    import transformers

    cls = getattr(transformers, name, None) or getattr(transformers, name.removesuffix("Fast"), None)
    if cls is None:
        return None
    parts = cls.__module__.split(".")
    return parts[2] if len(parts) > 3 and parts[1] == "models" else None


def code_dirs(fixture_folder):
    """The model directories whose code a fixture's usage path runs."""
    from transformers.models.auto import (
        configuration_auto,
        feature_extraction_auto,
        image_processing_auto,
        processing_auto,
        tokenization_auto,
    )

    folder = Path(fixture_folder)
    config = json.loads((folder / "config.json").read_text())
    model_type = config["model_type"]
    names = set(config.get("architectures") or [])
    names.add(configuration_auto.CONFIG_MAPPING_NAMES[model_type])
    for mapping in (
        tokenization_auto.TOKENIZER_MAPPING_NAMES,
        feature_extraction_auto.FEATURE_EXTRACTOR_MAPPING_NAMES,
        image_processing_auto.IMAGE_PROCESSOR_MAPPING_NAMES,
        processing_auto.PROCESSOR_MAPPING_NAMES,
    ):
        value = mapping.get(model_type)
        names.update(v for v in (value if isinstance(value, (tuple, list)) else [value]) if isinstance(v, str))
    for file in PROCESSOR_FILES:
        if (folder / file).exists():
            values = json.loads((folder / file).read_text())
            names.update(values[k] for k in CLASS_KEYS if isinstance(values.get(k), str))
    return sorted({d for d in map(module_dir, names) if d})


def select(changed_files, contract_dirs):
    """{contract: reason} for the contracts the changed files can affect."""
    selected = {}
    for path in changed_files:
        if path in SHARED:
            return {name: f"dependencies ({path})" for name in contract_dirs}
        if not path.startswith("src/transformers/"):
            continue
        parts = path.split("/")
        if len(parts) > 4 and parts[2] == "models" and parts[3] != "auto":
            for name, dirs in contract_dirs.items():
                if parts[3] in dirs:
                    selected.setdefault(name, f"models/{parts[3]} ({path})")
        else:
            return {name: f"shared code ({path})" for name in contract_dirs}
    return selected
