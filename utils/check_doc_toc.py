# coding=utf-8
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
This script is responsible for cleaning the model section of the table of content by removing duplicates and sorting
the entries in alphabetical order.

Usage (from the root of the repo):

Check that the table of content is properly sorted (used in `make quality`):

```bash
python utils/check_doc_toc.py
```

Auto-sort the table of content if it is not properly sorted (used in `make style`):

```bash
python utils/check_doc_toc.py --fix_and_overwrite
```
"""

import argparse
from collections import defaultdict
from typing import List

import yaml


PATH_TO_TOC = "docs/source/en/_toctree.yml"


def clean_model_doc_toc(model_doc: List[dict]) -> List[dict]:
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


def check_model_doc(overwrite: bool = False):
    """
    Check that the content of the table of content in `_toctree.yml` is clean (no duplicates and sorted for the model
    API doc) and potentially auto-cleans it.

    Args:
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether to just check if the TOC is clean or to auto-clean it (when `overwrite=True`).
    """
    with open(PATH_TO_TOC, encoding="utf-8") as f:
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
            with open(PATH_TO_TOC, "w", encoding="utf-8") as f:
                f.write(yaml.dump(content, allow_unicode=True))
        else:
            raise ValueError(
                "The model doc part of the table of content is not properly sorted, run `make style` to fix this."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    check_model_doc(args.fix_and_overwrite)
