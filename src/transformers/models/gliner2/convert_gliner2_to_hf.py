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

"""Convert a GLiNER2 checkpoint into a native Transformers checkpoint."""

import argparse
import json
import shutil
from pathlib import Path

from transformers import AutoConfig, Gliner2Config
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


def _encoder_config(source: Path, raw: dict):
    """Load the encoder config shipped beside the checkpoint."""
    nested = source / "encoder_config" / "config.json"
    if nested.is_file():
        return AutoConfig.from_pretrained(nested.parent)
    name = raw.get("model_name")
    if not name:
        raise ValueError(f"{source} has no encoder_config and no model_name")
    return AutoConfig.from_pretrained(name)


def _boundary_config(raw: dict):
    """Keep boundary fields that the native config defines."""
    head = raw.get("boundary_head") or {}
    if raw.get("architecture") != "boundary" and not head:
        return None
    known = set(Gliner2BoundaryConfig.__dataclass_fields__)
    return {key: value for key, value in head.items() if key in known}


def convert(source: Path, dest: Path) -> None:
    """Write a Transformers checkpoint. Tokenizer files are copied unchanged."""
    source = Path(source)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    raw = json.loads((source / "config.json").read_text())
    encoder = _encoder_config(source, raw)
    encoder._attn_implementation = "eager"
    boundary = _boundary_config(raw)
    architecture = raw.get("architecture", "boundary" if boundary else "span")
    config = Gliner2Config(
        encoder_config=encoder,
        architecture=architecture,
        max_width=int(raw.get("max_width", 8)),
        counting_layer=raw.get("counting_layer", "count_lstm"),
        token_pooling=raw.get("token_pooling", "first"),
        max_len=int(raw.get("max_len", 2048)),
        boundary_config=boundary,
        classification_temperature=float(
            (raw.get("boundary_head") or {}).get(
                "classification_temperature", raw.get("classification_temperature", 1.0)
            )
        ),
    )
    config.architectures = ["Gliner2ForSchemaExtraction"]
    config.save_pretrained(dest)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    convert(args.source, args.dest)


if __name__ == "__main__":
    main()
