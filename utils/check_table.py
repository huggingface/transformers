# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
import collections
import importlib.util
import os
import re


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_table.py
TRANSFORMERS_PATH = "src/transformers"
PATH_TO_DOCS = "docs/source"
REPO_PATH = "."


def _find_text_in_file(filename, start_prompt, end_prompt):
    """
    Find the text in `filename` between a line beginning with `start_prompt` and before `end_prompt`, removing empty
    lines.
    """
    with open(filename, "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()
    # Find the start prompt.
    start_index = 0
    while not lines[start_index].startswith(start_prompt):
        start_index += 1
    start_index += 1

    end_index = start_index
    while not lines[end_index].startswith(end_prompt):
        end_index += 1
    end_index -= 1

    while len(lines[start_index]) <= 1:
        start_index += 1
    while len(lines[end_index]) <= 1:
        end_index -= 1
    end_index += 1
    return "".join(lines[start_index:end_index]), start_index, end_index, lines


# Add here suffixes that are used to identify models, seperated by |
ALLOWED_MODEL_SUFFIXES = "Model|Encoder|Decoder|ForConditionalGeneration"
# Regexes that match TF/Flax/PT model names.
_re_tf_models = re.compile(r"TF(.*)(?:Model|Encoder|Decoder|ForConditionalGeneration)")
_re_flax_models = re.compile(r"Flax(.*)(?:Model|Encoder|Decoder|ForConditionalGeneration)")
# Will match any TF or Flax model too so need to be in an else branch afterthe two previous regexes.
_re_pt_models = re.compile(r"(.*)(?:Model|Encoder|Decoder|ForConditionalGeneration)")


# This is to make sure the transformers module imported is the one in the repo.
spec = importlib.util.spec_from_file_location(
    "transformers",
    os.path.join(TRANSFORMERS_PATH, "__init__.py"),
    submodule_search_locations=[TRANSFORMERS_PATH],
)
transformers_module = spec.loader.load_module()


# Thanks to https://stackoverflow.com/questions/29916065/how-to-do-camelcase-split-in-python
def camel_case_split(identifier):
    "Split a camelcased `identifier` into words."
    matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", identifier)
    return [m.group(0) for m in matches]


def _center_text(text, width):
    text_length = 2 if text == "✅" or text == "❌" else len(text)
    left_indent = (width - text_length) // 2
    right_indent = width - text_length - left_indent
    return " " * left_indent + text + " " * right_indent


def get_model_table_from_auto_modules():
    """Generates an up-to-date model table from the content of the auto modules."""
    # Dictionary model names to config.
    config_maping_names = transformers_module.models.auto.configuration_auto.CONFIG_MAPPING_NAMES
    model_name_to_config = {
        name: config_maping_names[code]
        for code, name in transformers_module.MODEL_NAMES_MAPPING.items()
        if code in config_maping_names
    }
    model_name_to_prefix = {name: config.replace("Config", "") for name, config in model_name_to_config.items()}

    # Dictionaries flagging if each model prefix has a slow/fast tokenizer, backend in PT/TF/Flax.
    slow_tokenizers = collections.defaultdict(bool)
    fast_tokenizers = collections.defaultdict(bool)
    pt_models = collections.defaultdict(bool)
    tf_models = collections.defaultdict(bool)
    flax_models = collections.defaultdict(bool)

    # Let's lookup through all transformers object (once).
    for attr_name in dir(transformers_module):
        lookup_dict = None
        if attr_name.endswith("Tokenizer"):
            lookup_dict = slow_tokenizers
            attr_name = attr_name[:-9]
        elif attr_name.endswith("TokenizerFast"):
            lookup_dict = fast_tokenizers
            attr_name = attr_name[:-13]
        elif _re_tf_models.match(attr_name) is not None:
            lookup_dict = tf_models
            attr_name = _re_tf_models.match(attr_name).groups()[0]
        elif _re_flax_models.match(attr_name) is not None:
            lookup_dict = flax_models
            attr_name = _re_flax_models.match(attr_name).groups()[0]
        elif _re_pt_models.match(attr_name) is not None:
            lookup_dict = pt_models
            attr_name = _re_pt_models.match(attr_name).groups()[0]

        if lookup_dict is not None:
            while len(attr_name) > 0:
                if attr_name in model_name_to_prefix.values():
                    lookup_dict[attr_name] = True
                    break
                # Try again after removing the last word in the name
                attr_name = "".join(camel_case_split(attr_name)[:-1])

    # Let's build that table!
    model_names = list(model_name_to_config.keys())
    model_names.sort(key=str.lower)
    columns = ["Model", "Tokenizer slow", "Tokenizer fast", "PyTorch support", "TensorFlow support", "Flax Support"]
    # We'll need widths to properly display everything in the center (+2 is to leave one extra space on each side).
    widths = [len(c) + 2 for c in columns]
    widths[0] = max([len(name) for name in model_names]) + 2

    # Build the table per se
    table = "|" + "|".join([_center_text(c, w) for c, w in zip(columns, widths)]) + "|\n"
    # Use ":-----:" format to center-aligned table cell texts
    table += "|" + "|".join([":" + "-" * (w - 2) + ":" for w in widths]) + "|\n"

    check = {True: "✅", False: "❌"}
    for name in model_names:
        prefix = model_name_to_prefix[name]
        line = [
            name,
            check[slow_tokenizers[prefix]],
            check[fast_tokenizers[prefix]],
            check[pt_models[prefix]],
            check[tf_models[prefix]],
            check[flax_models[prefix]],
        ]
        table += "|" + "|".join([_center_text(l, w) for l, w in zip(line, widths)]) + "|\n"
    return table


def check_model_table(overwrite=False):
    """Check the model table in the index.rst is consistent with the state of the lib and maybe `overwrite`."""
    current_table, start_index, end_index, lines = _find_text_in_file(
        filename=os.path.join(PATH_TO_DOCS, "index.mdx"),
        start_prompt="<!--This table is updated automatically from the auto modules",
        end_prompt="<!-- End table-->",
    )
    new_table = get_model_table_from_auto_modules()

    if current_table != new_table:
        if overwrite:
            with open(os.path.join(PATH_TO_DOCS, "index.mdx"), "w", encoding="utf-8", newline="\n") as f:
                f.writelines(lines[:start_index] + [new_table] + lines[end_index:])
        else:
            raise ValueError(
                "The model table in the `index.mdx` has not been updated. Run `make fix-copies` to fix this."
            )


def has_onnx(model_type):
    """
    Returns whether `model_type` is supported by ONNX (by checking if there is an ONNX config) or not.
    """
    config_mapping = transformers_module.models.auto.configuration_auto.CONFIG_MAPPING
    if model_type not in config_mapping:
        return False
    config = config_mapping[model_type]
    config_module = config.__module__
    module = transformers_module
    for part in config_module.split(".")[1:]:
        module = getattr(module, part)
    config_name = config.__name__
    onnx_config_name = config_name.replace("Config", "OnnxConfig")
    return hasattr(module, onnx_config_name)


def get_onnx_model_list():
    """
    Return the list of models supporting ONNX.
    """
    config_mapping = transformers_module.models.auto.configuration_auto.CONFIG_MAPPING
    model_names = config_mapping = transformers_module.models.auto.configuration_auto.MODEL_NAMES_MAPPING
    onnx_model_types = [model_type for model_type in config_mapping.keys() if has_onnx(model_type)]
    onnx_model_names = [model_names[model_type] for model_type in onnx_model_types]
    onnx_model_names.sort(key=lambda x: x.upper())
    return "\n".join([f"- {name}" for name in onnx_model_names]) + "\n"


def check_onnx_model_list(overwrite=False):
    """Check the model list in the serialization.mdx is consistent with the state of the lib and maybe `overwrite`."""
    current_list, start_index, end_index, lines = _find_text_in_file(
        filename=os.path.join(PATH_TO_DOCS, "serialization.mdx"),
        start_prompt="<!--This table is automatically generated by make style, do not fill manually!-->",
        end_prompt="The ONNX conversion is supported for the PyTorch versions of the models.",
    )
    new_list = get_onnx_model_list()

    if current_list != new_list:
        if overwrite:
            with open(os.path.join(PATH_TO_DOCS, "serialization.mdx"), "w", encoding="utf-8", newline="\n") as f:
                f.writelines(lines[:start_index] + [new_list] + lines[end_index:])
        else:
            raise ValueError("The list of ONNX-supported models needs an update. Run `make fix-copies` to fix this.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    check_model_table(args.fix_and_overwrite)
    check_onnx_model_list(args.fix_and_overwrite)
