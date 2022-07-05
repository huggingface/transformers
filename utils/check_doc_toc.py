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

import argparse
from collections import defaultdict

import yaml


PATH_TO_TOC = "docs/source/en/_toctree.yml"


def clean_model_doc_toc(model_doc):
    """
    Cleans the table of content of the model documentation by removing duplicates and sorting models alphabetically.
    """
    counts = defaultdict(int)
    for doc in model_doc:
        counts[doc["local"]] += 1
    duplicates = [key for key, value in counts.items() if value > 1]

    new_doc = []
    for duplicate_key in duplicates:
        titles = list(set(doc["title"] for doc in model_doc if doc["local"] == duplicate_key))
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


def check_model_doc(overwrite=False):
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

    old_model_doc = api_doc[model_idx]["sections"]
    new_model_doc = clean_model_doc_toc(old_model_doc)

    if old_model_doc != new_model_doc:
        if overwrite:
            api_doc[model_idx]["sections"] = new_model_doc
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
