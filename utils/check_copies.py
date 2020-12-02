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
import glob
import importlib
import os
import re
import tempfile


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_copies.py
TRANSFORMERS_PATH = "src/transformers"
PATH_TO_DOCS = "docs/source"
REPO_PATH = "."


def find_code_in_transformers(object_name):
    """ Find and return the code source code of `object_name`."""
    parts = object_name.split(".")
    i = 0

    # First let's find the module where our object lives.
    module = parts[i]
    while i < len(parts) and not os.path.isfile(os.path.join(TRANSFORMERS_PATH, f"{module}.py")):
        i += 1
        module = os.path.join(module, parts[i])
    if i >= len(parts):
        raise ValueError(
            f"`object_name` should begin with the name of a module of transformers but got {object_name}."
        )

    with open(os.path.join(TRANSFORMERS_PATH, f"{module}.py"), "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()

    # Now let's find the class / func in the code!
    indent = ""
    line_index = 0
    for name in parts[i + 1 :]:
        while line_index < len(lines) and re.search(fr"^{indent}(class|def)\s+{name}", lines[line_index]) is None:
            line_index += 1
        indent += "    "
        line_index += 1

    if line_index >= len(lines):
        raise ValueError(f" {object_name} does not match any function or class in {module}.")

    # We found the beginning of the class / func, now let's find the end (when the indent diminishes).
    start_index = line_index
    while line_index < len(lines) and (lines[line_index].startswith(indent) or len(lines[line_index]) <= 1):
        line_index += 1
    # Clean up empty lines at the end (if any).
    while len(lines[line_index - 1]) <= 1:
        line_index -= 1

    code_lines = lines[start_index:line_index]
    return "".join(code_lines)


_re_copy_warning = re.compile(r"^(\s*)#\s*Copied from\s+transformers\.(\S+\.\S+)\s*($|\S.*$)")
_re_replace_pattern = re.compile(r"with\s+(\S+)->(\S+)(?:\s|$)")


def blackify(code):
    """
    Applies the black part of our `make style` command to `code`.
    """
    has_indent = code.startswith("    ")
    if has_indent:
        code = f"class Bla:\n{code}"
    with tempfile.TemporaryDirectory() as d:
        fname = os.path.join(d, "tmp.py")
        with open(fname, "w", encoding="utf-8", newline="\n") as f:
            f.write(code)
        os.system(f"black -q --line-length 119 --target-version py35 {fname}")
        with open(fname, "r", encoding="utf-8", newline="\n") as f:
            result = f.read()
            return result[len("class Bla:\n") :] if has_indent else result


def is_copy_consistent(filename, overwrite=False):
    """
    Check if the code commented as a copy in `filename` matches the original.

    Return the differences or overwrites the content depending on `overwrite`.
    """
    with open(filename, "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()
    diffs = []
    line_index = 0
    # Not a for loop cause `lines` is going to change (if `overwrite=True`).
    while line_index < len(lines):
        search = _re_copy_warning.search(lines[line_index])
        if search is None:
            line_index += 1
            continue

        # There is some copied code here, let's retrieve the original.
        indent, object_name, replace_pattern = search.groups()
        theoretical_code = find_code_in_transformers(object_name)
        theoretical_indent = re.search(r"^(\s*)\S", theoretical_code).groups()[0]

        start_index = line_index + 1 if indent == theoretical_indent else line_index + 2
        indent = theoretical_indent
        line_index = start_index

        # Loop to check the observed code, stop when indentation diminishes or if we see a End copy comment.
        should_continue = True
        while line_index < len(lines) and should_continue:
            line_index += 1
            if line_index >= len(lines):
                break
            line = lines[line_index]
            should_continue = (len(line) <= 1 or line.startswith(indent)) and re.search(
                f"^{indent}# End copy", line
            ) is None
        # Clean up empty lines at the end (if any).
        while len(lines[line_index - 1]) <= 1:
            line_index -= 1

        observed_code_lines = lines[start_index:line_index]
        observed_code = "".join(observed_code_lines)

        # Before comparing, use the `replace_pattern` on the original code.
        if len(replace_pattern) > 0:
            search_patterns = _re_replace_pattern.search(replace_pattern)
            if search_patterns is not None:
                obj1, obj2 = search_patterns.groups()
                theoretical_code = re.sub(obj1, obj2, theoretical_code)

        # Test for a diff and act accordingly.
        if observed_code != theoretical_code:
            diffs.append([object_name, start_index])
            if overwrite:
                lines = lines[:start_index] + [theoretical_code] + lines[line_index:]
                line_index = start_index + 1

    if overwrite and len(diffs) > 0:
        # Warn the user a file has been modified.
        print(f"Detected changes, rewriting {filename}.")
        with open(filename, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(lines)
    return diffs


def check_copies(overwrite: bool = False):
    all_files = glob.glob(os.path.join(TRANSFORMERS_PATH, "**/*.py"), recursive=True)
    diffs = []
    for filename in all_files:
        new_diffs = is_copy_consistent(filename, overwrite)
        diffs += [f"- {filename}: copy does not match {d[0]} at line {d[1]}" for d in new_diffs]
    if not overwrite and len(diffs) > 0:
        diff = "\n".join(diffs)
        raise Exception(
            "Found the following copy inconsistencies:\n"
            + diff
            + "\nRun `make fix-copies` or `python utils/check_copies.py --fix_and_overwrite` to fix them."
        )
    check_model_list_copy(overwrite=overwrite)


def get_model_list():
    """ Extracts the model list from the README. """
    # If the introduction or the conclusion of the list change, the prompts may need to be updated.
    _start_prompt = "🤗 Transformers currently provides the following architectures"
    _end_prompt = "1. Want to contribute a new model?"
    with open(os.path.join(REPO_PATH, "README.md"), "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()
    # Find the start of the list.
    start_index = 0
    while not lines[start_index].startswith(_start_prompt):
        start_index += 1
    start_index += 1

    result = []
    current_line = ""
    end_index = start_index

    while not lines[end_index].startswith(_end_prompt):
        if lines[end_index].startswith("1."):
            if len(current_line) > 1:
                result.append(current_line)
            current_line = lines[end_index]
        elif len(lines[end_index]) > 1:
            current_line = f"{current_line[:-1]} {lines[end_index].lstrip()}"
        end_index += 1
    if len(current_line) > 1:
        result.append(current_line)

    return "".join(result)


def split_long_line_with_indent(line, max_per_line, indent):
    """ Split the `line` so that it doesn't go over `max_per_line` and adds `indent` to new lines. """
    words = line.split(" ")
    lines = []
    current_line = words[0]
    for word in words[1:]:
        if len(f"{current_line} {word}") > max_per_line:
            lines.append(current_line)
            current_line = " " * indent + word
        else:
            current_line = f"{current_line} {word}"
    lines.append(current_line)
    return "\n".join(lines)


def convert_to_rst(model_list, max_per_line=None):
    """ Convert `model_list` to rst format. """
    # Convert **[description](link)** to `description <link>`__
    def _rep_link(match):
        title, link = match.groups()
        # Keep hard links for the models not released yet
        if "master" in link or not link.startswith("https://huggingface.co/transformers"):
            return f"`{title} <{link}>`__"
        # Convert links to relative links otherwise
        else:
            link = link[len("https://huggingface.co/transformers/") : -len(".html")]
            return f":doc:`{title} <{link}>`"

    model_list = re.sub(r"\*\*\[([^\]]*)\]\(([^\)]*)\)\*\*", _rep_link, model_list)

    # Convert [description](link) to `description <link>`__
    model_list = re.sub(r"\[([^\]]*)\]\(([^\)]*)\)", r"`\1 <\2>`__", model_list)

    # Enumerate the lines properly
    lines = model_list.split("\n")
    result = []
    for i, line in enumerate(lines):
        line = re.sub(r"^\s*(\d+)\.", f"{i+1}.", line)
        # Split the lines that are too long
        if max_per_line is not None and len(line) > max_per_line:
            prompt = re.search(r"^(\s*\d+\.\s+)\S", line)
            indent = len(prompt.groups()[0]) if prompt is not None else 0
            line = split_long_line_with_indent(line, max_per_line, indent)

        result.append(line)
    return "\n".join(result)


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


def check_model_list_copy(overwrite=False, max_per_line=119):
    """ Check the model lists in the README and index.rst are consistent and maybe `overwrite`. """
    rst_list, start_index, end_index, lines = _find_text_in_file(
        filename=os.path.join(PATH_TO_DOCS, "index.rst"),
        start_prompt="    This list is updated automatically from the README",
        end_prompt=".. _bigtable:",
    )
    md_list = get_model_list()
    converted_list = convert_to_rst(md_list, max_per_line=max_per_line)

    if converted_list != rst_list:
        if overwrite:
            with open(os.path.join(PATH_TO_DOCS, "index.rst"), "w", encoding="utf-8", newline="\n") as f:
                f.writelines(lines[:start_index] + [converted_list] + lines[end_index:])
        else:
            raise ValueError(
                "The model list in the README changed and the list in `index.rst` has not been updated. Run "
                "`make fix-copies` to fix this."
            )


# Add here suffixes that are used to identify models, seperated by |
ALLOWED_MODEL_SUFFIXES = "Model|Encoder|Decoder|ForConditionalGeneration"
# Regexes that match TF/Flax/PT model names.
_re_tf_models = re.compile(r"TF(.*)(?:Model|Encoder|Decoder|ForConditionalGeneration)")
_re_flax_models = re.compile(r"Flax(.*)(?:Model|Encoder|Decoder|ForConditionalGeneration)")
# Will match any TF or Flax model too so need to be in an else branch afterthe two previous regexes.
_re_pt_models = re.compile(r"(.*)(?:Model|Encoder|Decoder|ForConditionalGeneration)")


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
    # This is to make sure the transformers module imported is the one in the repo.
    spec = importlib.util.spec_from_file_location(
        "transformers",
        os.path.join(TRANSFORMERS_PATH, "__init__.py"),
        submodule_search_locations=[TRANSFORMERS_PATH],
    )
    transformers = spec.loader.load_module()

    # Dictionary model names to config.
    model_name_to_config = {
        name: transformers.CONFIG_MAPPING[code] for code, name in transformers.MODEL_NAMES_MAPPING.items()
    }
    model_name_to_prefix = {
        name: config.__name__.replace("Config", "") for name, config in model_name_to_config.items()
    }

    # Dictionaries flagging if each model prefix has a slow/fast tokenizer, backend in PT/TF/Flax.
    slow_tokenizers = collections.defaultdict(bool)
    fast_tokenizers = collections.defaultdict(bool)
    pt_models = collections.defaultdict(bool)
    tf_models = collections.defaultdict(bool)
    flax_models = collections.defaultdict(bool)

    # Let's lookup through all transformers object (once).
    for attr_name in dir(transformers):
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
    model_names.sort()
    columns = ["Model", "Tokenizer slow", "Tokenizer fast", "PyTorch support", "TensorFlow support", "Flax Support"]
    # We'll need widths to properly display everything in the center (+2 is to leave one extra space on each side).
    widths = [len(c) + 2 for c in columns]
    widths[0] = max([len(name) for name in model_names]) + 2

    # Rst table per se
    table = ".. rst-class:: center-aligned-table\n\n"
    table += "+" + "+".join(["-" * w for w in widths]) + "+\n"
    table += "|" + "|".join([_center_text(c, w) for c, w in zip(columns, widths)]) + "|\n"
    table += "+" + "+".join(["=" * w for w in widths]) + "+\n"

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
        table += "+" + "+".join(["-" * w for w in widths]) + "+\n"
    return table


def check_model_table(overwrite=False):
    """ Check the model table in the index.rst is consistent with the state of the lib and maybe `overwrite`. """
    current_table, start_index, end_index, lines = _find_text_in_file(
        filename=os.path.join(PATH_TO_DOCS, "index.rst"),
        start_prompt="    This table is updated automatically from the auto module",
        end_prompt=".. toctree::",
    )
    new_table = get_model_table_from_auto_modules()

    if current_table != new_table:
        if overwrite:
            with open(os.path.join(PATH_TO_DOCS, "index.rst"), "w", encoding="utf-8", newline="\n") as f:
                f.writelines(lines[:start_index] + [new_table] + lines[end_index:])
        else:
            raise ValueError(
                "The model table in the `index.rst` has not been updated. Run `make fix-copies` to fix this."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    check_copies(args.fix_and_overwrite)
    check_model_table(args.fix_and_overwrite)
