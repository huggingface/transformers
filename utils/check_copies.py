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
import glob
import os
import re

import black
from doc_builder.style_doc import style_docstrings_in_code


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_copies.py
TRANSFORMERS_PATH = "src/transformers"
PATH_TO_DOCS = "docs/source/en"
REPO_PATH = "."

# Mapping for files that are full copies of others (keys are copies, values the file to keep them up to data with)
FULL_COPIES = {
    "examples/tensorflow/question-answering/utils_qa.py": "examples/pytorch/question-answering/utils_qa.py",
    "examples/flax/question-answering/utils_qa.py": "examples/pytorch/question-answering/utils_qa.py",
}


LOCALIZED_READMES = {
    # If the introduction or the conclusion of the list change, the prompts may need to be updated.
    "README.md": {
        "start_prompt": "🤗 Transformers currently provides the following architectures",
        "end_prompt": "1. Want to contribute a new model?",
        "format_model_list": "**[{title}]({model_link})** (from {paper_affiliations}) released with the paper {paper_title_link} by {paper_authors}.{supplements}",
    },
    "README_zh-hans.md": {
        "start_prompt": "🤗 Transformers 目前支持如下的架构",
        "end_prompt": "1. 想要贡献新的模型？",
        "format_model_list": "**[{title}]({model_link})** (来自 {paper_affiliations}) 伴随论文 {paper_title_link} 由 {paper_authors} 发布。{supplements}",
    },
    "README_zh-hant.md": {
        "start_prompt": "🤗 Transformers 目前支援以下的架構",
        "end_prompt": "1. 想要貢獻新的模型？",
        "format_model_list": "**[{title}]({model_link})** (from {paper_affiliations}) released with the paper {paper_title_link} by {paper_authors}.{supplements}",
    },
    "README_ko.md": {
        "start_prompt": "🤗 Transformers는 다음 모델들을 제공합니다",
        "end_prompt": "1. 새로운 모델을 올리고 싶나요?",
        "format_model_list": "**[{title}]({model_link})** (from {paper_affiliations}) released with the paper {paper_title_link} by {paper_authors}.{supplements}",
    },
}


def _should_continue(line, indent):
    return line.startswith(indent) or len(line) <= 1 or re.search(r"^\s*\)(\s*->.*:|:)\s*$", line) is not None


def find_code_in_transformers(object_name):
    """Find and return the code source code of `object_name`."""
    parts = object_name.split(".")
    i = 0

    # First let's find the module where our object lives.
    module = parts[i]
    while i < len(parts) and not os.path.isfile(os.path.join(TRANSFORMERS_PATH, f"{module}.py")):
        i += 1
        if i < len(parts):
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
        while (
            line_index < len(lines) and re.search(rf"^{indent}(class|def)\s+{name}(\(|\:)", lines[line_index]) is None
        ):
            line_index += 1
        indent += "    "
        line_index += 1

    if line_index >= len(lines):
        raise ValueError(f" {object_name} does not match any function or class in {module}.")

    # We found the beginning of the class / func, now let's find the end (when the indent diminishes).
    start_index = line_index
    while line_index < len(lines) and _should_continue(lines[line_index], indent):
        line_index += 1
    # Clean up empty lines at the end (if any).
    while len(lines[line_index - 1]) <= 1:
        line_index -= 1

    code_lines = lines[start_index:line_index]
    return "".join(code_lines)


_re_copy_warning = re.compile(r"^(\s*)#\s*Copied from\s+transformers\.(\S+\.\S+)\s*($|\S.*$)")
_re_replace_pattern = re.compile(r"^\s*(\S+)->(\S+)(\s+.*|$)")


def get_indent(code):
    lines = code.split("\n")
    idx = 0
    while idx < len(lines) and len(lines[idx]) == 0:
        idx += 1
    if idx < len(lines):
        return re.search(r"^(\s*)\S", lines[idx]).groups()[0]
    return ""


def blackify(code):
    """
    Applies the black part of our `make style` command to `code`.
    """
    has_indent = len(get_indent(code)) > 0
    if has_indent:
        code = f"class Bla:\n{code}"
    mode = black.Mode(target_versions={black.TargetVersion.PY35}, line_length=119)
    result = black.format_str(code, mode=mode)
    result, _ = style_docstrings_in_code(result)
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
        theoretical_indent = get_indent(theoretical_code)

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
            should_continue = _should_continue(line, indent) and re.search(f"^{indent}# End copy", line) is None
        # Clean up empty lines at the end (if any).
        while len(lines[line_index - 1]) <= 1:
            line_index -= 1

        observed_code_lines = lines[start_index:line_index]
        observed_code = "".join(observed_code_lines)

        # Before comparing, use the `replace_pattern` on the original code.
        if len(replace_pattern) > 0:
            patterns = replace_pattern.replace("with", "").split(",")
            patterns = [_re_replace_pattern.search(p) for p in patterns]
            for pattern in patterns:
                if pattern is None:
                    continue
                obj1, obj2, option = pattern.groups()
                theoretical_code = re.sub(obj1, obj2, theoretical_code)
                if option.strip() == "all-casing":
                    theoretical_code = re.sub(obj1.lower(), obj2.lower(), theoretical_code)
                    theoretical_code = re.sub(obj1.upper(), obj2.upper(), theoretical_code)

            # Blackify after replacement. To be able to do that, we need the header (class or function definition)
            # from the previous line
            theoretical_code = blackify(lines[start_index - 1] + theoretical_code)
            theoretical_code = theoretical_code[len(lines[start_index - 1]) :]

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


def check_full_copies(overwrite: bool = False):
    diffs = []
    for target, source in FULL_COPIES.items():
        with open(source, "r", encoding="utf-8") as f:
            source_code = f.read()
        with open(target, "r", encoding="utf-8") as f:
            target_code = f.read()
        if source_code != target_code:
            if overwrite:
                with open(target, "w", encoding="utf-8") as f:
                    print(f"Replacing the content of {target} by the one of {source}.")
                    f.write(source_code)
            else:
                diffs.append(f"- {target}: copy does not match {source}.")

    if not overwrite and len(diffs) > 0:
        diff = "\n".join(diffs)
        raise Exception(
            "Found the following copy inconsistencies:\n"
            + diff
            + "\nRun `make fix-copies` or `python utils/check_copies.py --fix_and_overwrite` to fix them."
        )


def get_model_list(filename, start_prompt, end_prompt):
    """Extracts the model list from the README."""
    with open(os.path.join(REPO_PATH, filename), "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()
    # Find the start of the list.
    start_index = 0
    while not lines[start_index].startswith(start_prompt):
        start_index += 1
    start_index += 1

    result = []
    current_line = ""
    end_index = start_index

    while not lines[end_index].startswith(end_prompt):
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


def convert_to_localized_md(model_list, localized_model_list, format_str):
    """Convert `model_list` to each localized README."""

    def _rep(match):
        title, model_link, paper_affiliations, paper_title_link, paper_authors, supplements = match.groups()
        return format_str.format(
            title=title,
            model_link=model_link,
            paper_affiliations=paper_affiliations,
            paper_title_link=paper_title_link,
            paper_authors=paper_authors,
            supplements=" " + supplements.strip() if len(supplements) != 0 else "",
        )

    # This regex captures metadata from an English model description, including model title, model link,
    # affiliations of the paper, title of the paper, authors of the paper, and supplemental data (see DistilBERT for example).
    _re_capture_meta = re.compile(
        r"\*\*\[([^\]]*)\]\(([^\)]*)\)\*\* \(from ([^)]*)\)[^\[]*([^\)]*\)).*?by (.*?[A-Za-z\*]{2,}?)\. (.*)$"
    )
    # This regex is used to synchronize link.
    _re_capture_title_link = re.compile(r"\*\*\[([^\]]*)\]\(([^\)]*)\)\*\*")

    num_models_equal = True

    if len(localized_model_list) == 0:
        localized_model_index = {}
    else:
        try:
            localized_model_index = {
                re.search(r"\*\*\[([^\]]*)", line).groups()[0]: line
                for line in localized_model_list.strip().split("\n")
            }
        except AttributeError:
            raise AttributeError("A model name in localized READMEs cannot be recognized.")

    for model in model_list.strip().split("\n"):
        title, model_link = _re_capture_title_link.search(model).groups()
        if title not in localized_model_index:
            num_models_equal = False
            # Add an anchor white space behind a model description string for regex.
            # If metadata cannot be captured, the English version will be directly copied.
            localized_model_index[title] = _re_capture_meta.sub(_rep, model + " ")
        else:
            # Synchronize link
            localized_model_index[title] = _re_capture_title_link.sub(
                f"**[{title}]({model_link})**", localized_model_index[title], count=1
            )

    sorted_index = sorted(localized_model_index.items(), key=lambda x: x[0].lower())

    return num_models_equal, "\n".join(map(lambda x: x[1], sorted_index)) + "\n"


def convert_readme_to_index(model_list):
    model_list = model_list.replace("https://huggingface.co/docs/transformers/main/", "")
    return model_list.replace("https://huggingface.co/docs/transformers/", "")


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
    """Check the model lists in the README and index.rst are consistent and maybe `overwrite`."""
    # Fix potential doc links in the README
    with open(os.path.join(REPO_PATH, "README.md"), "r", encoding="utf-8", newline="\n") as f:
        readme = f.read()
    new_readme = readme.replace("https://huggingface.co/transformers", "https://huggingface.co/docs/transformers")
    new_readme = readme.replace(
        "https://huggingface.co/docs/main/transformers", "https://huggingface.co/docs/transformers/main"
    )
    if new_readme != readme:
        if overwrite:
            with open(os.path.join(REPO_PATH, "README.md"), "w", encoding="utf-8", newline="\n") as f:
                f.write(new_readme)
        else:
            raise ValueError(
                "The main README contains wrong links to the documentation of Transformers. Run `make fix-copies` to "
                "automatically fix them."
            )

    # If the introduction or the conclusion of the list change, the prompts may need to be updated.
    index_list, start_index, end_index, lines = _find_text_in_file(
        filename=os.path.join(PATH_TO_DOCS, "index.mdx"),
        start_prompt="<!--This list is updated automatically from the README",
        end_prompt="### Supported frameworks",
    )
    md_list = get_model_list(
        filename="README.md",
        start_prompt=LOCALIZED_READMES["README.md"]["start_prompt"],
        end_prompt=LOCALIZED_READMES["README.md"]["end_prompt"],
    )

    converted_md_lists = []
    for filename, value in LOCALIZED_READMES.items():
        _start_prompt = value["start_prompt"]
        _end_prompt = value["end_prompt"]
        _format_model_list = value["format_model_list"]

        localized_md_list = get_model_list(filename, _start_prompt, _end_prompt)
        num_models_equal, converted_md_list = convert_to_localized_md(md_list, localized_md_list, _format_model_list)

        converted_md_lists.append((filename, num_models_equal, converted_md_list, _start_prompt, _end_prompt))

    converted_md_list = convert_readme_to_index(md_list)
    if converted_md_list != index_list:
        if overwrite:
            with open(os.path.join(PATH_TO_DOCS, "index.mdx"), "w", encoding="utf-8", newline="\n") as f:
                f.writelines(lines[:start_index] + [converted_md_list] + lines[end_index:])
        else:
            raise ValueError(
                "The model list in the README changed and the list in `index.mdx` has not been updated. Run "
                "`make fix-copies` to fix this."
            )

    for converted_md_list in converted_md_lists:
        filename, num_models_equal, converted_md, _start_prompt, _end_prompt = converted_md_list

        if filename == "README.md":
            continue
        if overwrite:
            _, start_index, end_index, lines = _find_text_in_file(
                filename=os.path.join(REPO_PATH, filename), start_prompt=_start_prompt, end_prompt=_end_prompt
            )
            with open(os.path.join(REPO_PATH, filename), "w", encoding="utf-8", newline="\n") as f:
                f.writelines(lines[:start_index] + [converted_md] + lines[end_index:])
        elif not num_models_equal:
            raise ValueError(
                f"The model list in the README changed and the list in `{filename}` has not been updated. Run "
                "`make fix-copies` to fix this."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    check_copies(args.fix_and_overwrite)
    check_full_copies(args.fix_and_overwrite)
