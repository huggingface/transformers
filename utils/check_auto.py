# Copyright 2026 The HuggingFace Inc. team.
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
import ast
import glob
import json
import os
from collections import OrderedDict


def build_config_mapping_names() -> tuple[dict, dict]:
    model_type_map = OrderedDict()
    special_mappings = OrderedDict()

    # root_path = Path(__file__).resolve().parents[2]
    all_files = glob.glob("src/transformers/models/**/configuration_*.py", recursive=True)
    for config_path in all_files:
        module_name = config_path.split("/")[-2]
        with open(config_path, "r") as f:
            content = f.read()

        tree = ast.parse(content)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and any(
                base.id == "PreTrainedConfig" for base in node.bases if isinstance(base, ast.Name)
            ):
                config_cls_name = node.name
                model_type = None
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        model_types = [
                            stmt.value.value
                            for target in stmt.targets
                            if isinstance(target, ast.Name) and target.id == "model_type"
                        ]
                        if model_types:
                            model_type = model_types[0]
                            break

                if not model_type:
                    continue

                if model_type != module_name:
                    special_mappings[model_type] = module_name
                model_type_map[model_type] = config_cls_name

    return model_type_map, special_mappings


def build_image_processor_mapping(
    config_mapping: dict[str, str],
    special_mapping: dict[str, str],
) -> OrderedDict[str, dict[str, str | None]]:
    processor_mapping = OrderedDict()
    for model_type in config_mapping:
        if model_type in special_mapping:
            model_type = special_mapping[model_type]

        module = model_type.replace("-", "_")
        fast_processor_name = slow_processor_name = None
        if os.path.exists(f"src/transformers/models/{module}/image_processing_pil_{module}.py"):
            with open(f"src/transformers/models/{module}/image_processing_pil_{module}.py", "r") as f:
                content = f.read()

            tree = ast.parse(content)
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and any(
                    base.id == "PilBackend" for base in node.bases if isinstance(base, ast.Name)
                ):
                    slow_processor_name = node.name

        if os.path.exists(f"src/transformers/models/{module}/image_processing_{module}.py"):
            with open(f"src/transformers/models/{module}/image_processing_{module}.py", "r") as f:
                content = f.read()

            tree = ast.parse(content)
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and any(
                    base.id == "TorchvisionBackend" for base in node.bases if isinstance(base, ast.Name)
                ):
                    fast_processor_name = node.name

        processor_mapping[model_type] = {}

        if slow_processor_name is not None:
            processor_mapping[model_type]["pil"] = slow_processor_name

        if fast_processor_name is not None:
            processor_mapping[model_type]["torchvision"] = fast_processor_name

    return processor_mapping


def main(overwrite: bool):
    config_mapping, special_mapping = build_config_mapping_names()
    image_processor_mapping = build_image_processor_mapping(config_mapping, special_mapping)

    try:
        with open("src/transformers/models/auto/auto_mappings.json", "r") as f:
            old_mappings = json.load(f)
    except json.decoder.JSONDecodeError as e:
        old_mappings = {}
        print(f"Could not open the mapping due to {e}, defaulting to an empty mapping!")

    new_mappings = {
        "CONFIG_MAPPING_NAMES": config_mapping,
        "SPECIAL_MODEL_TYPE_TO_MODULE_NAME": special_mapping,
        "IMAGE_PROCESSOR_MAPPING_NAMES": image_processor_mapping,
    }

    if old_mappings != new_mappings:
        if not overwrite:
            raise Exception(
                "Generated auto-mapping is not consistent with the contents of `models/auto/auto_mappings.json`:\n"
                + "\nRun `make fix-repo` or `python utils/check_auto.py --fix_and_overwrite` to fix them."
            )
        else:
            with open("src/transformers/models/auto/auto_mappings.json", "w") as f:
                json.dump(new_mappings, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()
    main(overwrite=args.fix_and_overwrite)
