# Copyright 2026 the HuggingFace Team. All rights reserved.
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

import argparse
import json
import shutil
from pathlib import Path

from transformers import AutoConfig, Gliner2Config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.gliner2.configuration_gliner2 import Gliner2BoundaryConfig


_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spm.model",
    "tokenizer.model",
)

_CARD = """---
library_name: transformers
pipeline_tag: schema-extraction
license: apache-2.0
---

# {name}

Native Transformers conversion of a GLiNER2 checkpoint. Span and boundary heads keep their original weight names.

```python
from transformers import pipeline

extractor = pipeline("schema-extraction", model="{name}")
print(extractor("Ada Lovelace wrote notes about the analytical engine.", schema={{"entities": {{"person": {{}}, "work": {{}}}}}}))
```
"""


def _config_from_payload(payload: dict):
    """Build an encoder config from an inline config dict."""
    model_type = payload.get("model_type")
    if model_type not in CONFIG_MAPPING:
        raise ValueError(f"unsupported encoder model_type {model_type!r}")
    return CONFIG_MAPPING[model_type].from_dict(payload)


def _encoder_config(source: Path, raw: dict):
    """Load the encoder config from the sidecar, an inline dict, or model_name."""
    nested = source / "encoder_config" / "config.json"
    if nested.is_file():
        return _config_from_payload(json.loads(nested.read_text()))
    inline = raw.get("encoder_config")
    if isinstance(inline, dict) and inline.get("model_type"):
        return _config_from_payload(inline)
    name = raw.get("model_name")
    if not name:
        raise ValueError(f"{source} has no encoder_config and no model_name")
    return AutoConfig.from_pretrained(name, local_files_only=True)


def _published_head(raw: dict) -> dict:
    """Read boundary_head, or boundary_config when the source is already converted."""
    head = raw.get("boundary_head")
    if isinstance(head, dict) and head:
        return head
    nested = raw.get("boundary_config")
    return nested if isinstance(nested, dict) else {}


def _boundary_config(raw: dict):
    """Keep boundary fields that the native config defines."""
    head = _published_head(raw)
    if raw.get("architecture") != "boundary" and not head:
        return None
    known = set(Gliner2BoundaryConfig.__dataclass_fields__)
    return {key: value for key, value in head.items() if key in known}


def _max_len(raw: dict):
    """Keep a published null ``max_len`` instead of rewriting it to an int."""
    if "max_len" not in raw:
        return 2048
    value = raw["max_len"]
    return None if value is None else int(value)


def _max_width(raw: dict) -> int:
    """Read span width from the top-level config or the published span head."""
    if raw.get("max_width") is not None:
        return int(raw["max_width"])
    span_head = raw.get("span_head") or {}
    if isinstance(span_head, dict) and span_head.get("max_width") is not None:
        return int(span_head["max_width"])
    return 8


def convert(source: Path, dest: Path) -> None:
    """Write config.json with an inline encoder_config and copy weights unchanged."""
    source = Path(source)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    raw = json.loads((source / "config.json").read_text())
    encoder = _encoder_config(source, raw)
    boundary = _boundary_config(raw)
    architecture = raw.get("architecture", "boundary" if boundary else "span")
    span_head = raw.get("span_head")
    config = Gliner2Config(
        encoder_config=encoder,
        architecture=architecture,
        max_width=_max_width(raw),
        counting_layer=raw.get("counting_layer", "count_lstm"),
        token_pooling=raw.get("token_pooling", "first"),
        max_len=_max_len(raw),
        model_name=raw.get("model_name"),
        span_head=span_head if isinstance(span_head, dict) else None,
        architecture_version=raw.get("architecture_version"),
        config_version=raw.get("config_version"),
        boundary_config=boundary,
        classification_temperature=float(raw.get("classification_temperature", 1.0)),
    )
    config.architectures = ["Gliner2ForSchemaExtraction"]
    config.save_pretrained(dest)
    saved = json.loads((dest / "config.json").read_text())
    if "encoder_config" not in saved:
        raise ValueError("converted config.json is missing inline encoder_config")

    weight = source / "model.safetensors"
    if not weight.is_file():
        weight = source / "pytorch_model.bin"
    if not weight.is_file():
        raise FileNotFoundError(f"no weights in {source}")
    shutil.copy(weight, dest / weight.name)
    for name in _TOKENIZER_FILES:
        path = source / name
        if path.is_file():
            shutil.copy(path, dest / name)
    (dest / "README.md").write_text(_CARD.format(name=dest.name))


def main():
    """Convert one checkpoint directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    convert(args.source, args.dest)


if __name__ == "__main__":
    main()
