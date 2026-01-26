# Copyright 2022 The HuggingFace Inc. team.
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
This script is responsible for ensuring that all model docs are part of the `_toctree.yml` and cleaning the model
section of the table of content by removing duplicates and sorting the entries in alphabetical order.

Usage (from the root of the repo):

Check that the table of content is properly sorted (used in `make check-repo`):

```bash
python utils/check_doc_toc.py
```

Auto-sort the table of content if it is not properly sorted (used in `make fix-repo`):

```bash
python utils/check_doc_toc.py --fix_and_overwrite
```
"""

import argparse
import os
from collections import defaultdict

import yaml


ROOT = os.path.dirname(os.path.dirname(__file__))
TOCTREE_PATH = os.path.join(ROOT, "docs", "source", "en", "_toctree.yml")
DOC_PATH = os.path.join(ROOT, "docs", "source", "en", "model_doc")


def clean_model_doc_toc(model_doc: list[dict]) -> list[dict]:
    """
    Cleans a section of the table of content of the model documentation (one specific modality) by removing duplicates
    and sorting models alphabetically.

    Args:
        model_doc (`List[dict]`):
            The list of dictionaries extracted from the `_toctree.yml` file for this specific modality.

    Returns:
        `List[dict]`: List of dictionaries like the input, but cleaned up and sorted.
    """
    counts = defaultdict(int)
    for doc in model_doc:
        counts[doc["local"]] += 1
    duplicates = [key for key, value in counts.items() if value > 1]

    new_doc = []
    for duplicate_key in duplicates:
        titles = list({doc["title"] for doc in model_doc if doc["local"] == duplicate_key})
        if len(titles) > 1:
            raise ValueError(
                f"{duplicate_key} is present several times in the documentation table of content at "
                "`docs/source/en/_toctree.yml` with different *Title* values. Choose one of those and remove the "
                "others."
            )
        # Only add this once
        new_doc.append({"local": duplicate_key, "title": titles[0]})

    # Add none duplicate-keys
    new_doc.extend([doc for doc in model_doc if counts[doc["local"]] == 1])

    # Sort
    return sorted(new_doc, key=lambda s: s["title"].lower())


def ensure_all_models_in_toctree(model_doc: list[dict]):
    """Make sure that all models in `model_doc` folder are also part of the `_toctree.yml`. Raise if it's not
    the case."""
    all_documented_models = {model_doc_file.removesuffix(".md") for model_doc_file in os.listdir(DOC_PATH)} - {"auto"}
    all_models_in_toctree = {
        model_entry["local"].removeprefix("model_doc/") for section in model_doc for model_entry in section["sections"]
    }

    # everything alright
    if all_documented_models == all_models_in_toctree:
        return

    documented_but_not_in_toctree = all_documented_models - all_models_in_toctree
    in_toctree_but_not_documented = all_models_in_toctree - all_documented_models

    error_msg = ""
    if len(documented_but_not_in_toctree) > 0:
        error_msg += (
            f"{documented_but_not_in_toctree} appear(s) inside the folder `model_doc`, but not in the `_toctree.yml`. "
            "Please add it/them in their corresponding section inside the `_toctree.yml`."
        )
    if len(in_toctree_but_not_documented) > 0:
        if len(error_msg) > 0:
            error_msg += "\n"
        error_msg += (
            f"{in_toctree_but_not_documented} appear(s) in the `_toctree.yml`, but not inside the folder `model_doc`. "
            "Please add a corresponding `model.md` in `model_doc`."
        )

    raise ValueError(error_msg)


def check_model_doc(overwrite: bool = False):
    """
    Check that the content of the table of content in `_toctree.yml` is up-to-date (i.e. it contains all models) and
    clean (no duplicates and sorted for the model API doc) and potentially auto-cleans it.

    Args:
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether to just check if the TOC is clean or to auto-clean it (when `overwrite=True`).
    """
    with open(TOCTREE_PATH, encoding="utf-8") as f:
        content = yaml.safe_load(f.read())

    # Get to the API doc
    api_idx = 0
    while content[api_idx]["title"] != "API":
        api_idx += 1
    api_doc = content[api_idx]["sections"]

    # Then to the model doc
    model_idx = 0
    while api_doc[model_idx]["title"] != "Models":
        model_idx += 1

    model_doc = api_doc[model_idx]["sections"]

    # Make sure the toctree contains all models
    ensure_all_models_in_toctree(model_doc)

    # Extract the modalities and clean them one by one.
    modalities_docs = [(idx, section) for idx, section in enumerate(model_doc) if "sections" in section]
    diff = False
    for idx, modality_doc in modalities_docs:
        old_modality_doc = modality_doc["sections"]
        new_modality_doc = clean_model_doc_toc(old_modality_doc)

        if old_modality_doc != new_modality_doc:
            diff = True
            if overwrite:
                model_doc[idx]["sections"] = new_modality_doc

    if diff:
        if overwrite:
            api_doc[model_idx]["sections"] = model_doc
            content[api_idx]["sections"] = api_doc
            with open(TOCTREE_PATH, "w", encoding="utf-8") as f:
                f.write(yaml.dump(content, allow_unicode=True))
        else:
            raise ValueError(
                "The model doc part of the table of content is not properly sorted, run `make fix-repo` to fix this."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    check_model_doc(args.fix_and_overwrite)
