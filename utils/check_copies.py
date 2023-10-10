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
"""
Utility that checks whether the copies defined in the library match the original or not. This includes:
- All code commented with `# Copied from` comments,
- The list of models in the main README.md matches the ones in the localized READMEs,
- Files that are registered as full copies of one another in the `FULL_COPIES` constant of this script.

This also checks the list of models in the README is complete (has all models) and add a line to complete if there is
a model missing.

Use from the root of the repo with:

```bash
python utils/check_copies.py
```

for a check that will error in case of inconsistencies (used by `make repo-consistency`) or

```bash
python utils/check_copies.py --fix_and_overwrite
```

for a check that will fix all inconsistencies automatically (used by `make fix-copies`).
"""

import argparse
import glob
import os
import re
from typing import List, Optional, Tuple

import black
from doc_builder.style_doc import style_docstrings_in_code

from transformers.utils import direct_transformers_import


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_copies.py
TRANSFORMERS_PATH = "src/transformers"
MODEL_TEST_PATH = "tests/models"
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
        "format_model_list": (
            "**[{title}]({model_link})** (from {paper_affiliations}) released with the paper {paper_title_link} by"
            " {paper_authors}.{supplements}"
        ),
    },
    "README_zh-hans.md": {
        "start_prompt": "🤗 Transformers 目前支持如下的架构",
        "end_prompt": "1. 想要贡献新的模型？",
        "format_model_list": (
            "**[{title}]({model_link})** (来自 {paper_affiliations}) 伴随论文 {paper_title_link} 由 {paper_authors}"
            " 发布。{supplements}"
        ),
    },
    "README_zh-hant.md": {
        "start_prompt": "🤗 Transformers 目前支援以下的架構",
        "end_prompt": "1. 想要貢獻新的模型？",
        "format_model_list": (
            "**[{title}]({model_link})** (from {paper_affiliations}) released with the paper {paper_title_link} by"
            " {paper_authors}.{supplements}"
        ),
    },
    "README_ko.md": {
        "start_prompt": "🤗 Transformers는 다음 모델들을 제공합니다",
        "end_prompt": "1. 새로운 모델을 올리고 싶나요?",
        "format_model_list": (
            "**[{title}]({model_link})** ({paper_affiliations} 에서 제공)은 {paper_authors}.{supplements}의"
            " {paper_title_link}논문과 함께 발표했습니다."
        ),
    },
    "README_es.md": {
        "start_prompt": "🤗 Transformers actualmente proporciona las siguientes arquitecturas",
        "end_prompt": "1. ¿Quieres aportar un nuevo modelo?",
        "format_model_list": (
            "**[{title}]({model_link})** (from {paper_affiliations}) released with the paper {paper_title_link} by"
            " {paper_authors}.{supplements}"
        ),
    },
    "README_ja.md": {
        "start_prompt": "🤗Transformersは現在、以下のアーキテクチャを提供しています",
        "end_prompt": "1. 新しいモデルを投稿したいですか？",
        "format_model_list": (
            "**[{title}]({model_link})** ({paper_affiliations} から) {paper_authors}.{supplements} から公開された研究論文"
            " {paper_title_link}"
        ),
    },
    "README_hd.md": {
        "start_prompt": "🤗 ट्रांसफॉर्मर वर्तमान में निम्नलिखित आर्किटेक्चर का समर्थन करते हैं",
        "end_prompt": "1. एक नए मॉडल में योगदान देना चाहते हैं?",
        "format_model_list": (
            "**[{title}]({model_link})** ({paper_affiliations} से) {paper_authors}.{supplements} द्वारा"
            "अनुसंधान पत्र {paper_title_link} के साथ जारी किया गया"
        ),
    },
}


# This is to make sure the transformers module imported is the one in the repo.
transformers_module = direct_transformers_import(TRANSFORMERS_PATH)


def _should_continue(line: str, indent: str) -> bool:
    # Helper function. Returns `True` if `line` is empty, starts with the `indent` or is the end parenthesis of a
    # function definition
    return line.startswith(indent) or len(line.strip()) == 0 or re.search(r"^\s*\)(\s*->.*:|:)\s*$", line) is not None


def find_code_in_transformers(object_name: str, base_path: str = None) -> str:
    """
    Find and return the source code of an object.

    Args:
        object_name (`str`):
            The name of the object we want the source code of.
        base_path (`str`, *optional*):
            The path to the base folder where files are checked. If not set, it will be set to `TRANSFORMERS_PATH`.

    Returns:
        `str`: The source code of the object.
    """
    parts = object_name.split(".")
    i = 0

    # We can't set this as the default value in the argument, otherwise `CopyCheckTester` will fail, as it uses a
    # patched temp directory.
    if base_path is None:
        base_path = TRANSFORMERS_PATH

    # Detail: the `Copied from` statement is originally designed to work with the last part of `TRANSFORMERS_PATH`,
    # (which is `transformers`). The same should be applied for `MODEL_TEST_PATH`. However, its last part is `models`
    # (to only check and search in it) which is a bit confusing. So we keep the copied statement staring with
    # `tests.models.` and change it to `tests` here.
    if base_path == MODEL_TEST_PATH:
        base_path = "tests"

    # First let's find the module where our object lives.
    module = parts[i]
    while i < len(parts) and not os.path.isfile(os.path.join(base_path, f"{module}.py")):
        i += 1
        if i < len(parts):
            module = os.path.join(module, parts[i])
    if i >= len(parts):
        raise ValueError(
            f"`object_name` should begin with the name of a module of transformers but got {object_name}."
        )

    with open(os.path.join(base_path, f"{module}.py"), "r", encoding="utf-8", newline="\n") as f:
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
    start_index = line_index - 1
    while line_index < len(lines) and _should_continue(lines[line_index], indent):
        line_index += 1
    # Clean up empty lines at the end (if any).
    while len(lines[line_index - 1]) <= 1:
        line_index -= 1

    code_lines = lines[start_index:line_index]
    return "".join(code_lines)


_re_copy_warning = re.compile(r"^(\s*)#\s*Copied from\s+transformers\.(\S+\.\S+)\s*($|\S.*$)")
_re_copy_warning_for_test_file = re.compile(r"^(\s*)#\s*Copied from\s+tests\.(\S+\.\S+)\s*($|\S.*$)")
_re_replace_pattern = re.compile(r"^\s*(\S+)->(\S+)(\s+.*|$)")
_re_fill_pattern = re.compile(r"<FILL\s+[^>]*>")


def get_indent(code: str) -> str:
    """
    Find the indent in the first non empty line in a code sample.

    Args:
        code (`str`): The code to inspect.

    Returns:
        `str`: The indent looked at (as string).
    """
    lines = code.split("\n")
    idx = 0
    while idx < len(lines) and len(lines[idx]) == 0:
        idx += 1
    if idx < len(lines):
        return re.search(r"^(\s*)\S", lines[idx]).groups()[0]
    return ""


def blackify(code: str) -> str:
    """
    Applies the black part of our `make style` command to some code.

    Args:
        code (`str`): The code to format.

    Returns:
        `str`: The formatted code.
    """
    has_indent = len(get_indent(code)) > 0
    if has_indent:
        code = f"class Bla:\n{code}"
    mode = black.Mode(target_versions={black.TargetVersion.PY37}, line_length=119)
    result = black.format_str(code, mode=mode)
    result, _ = style_docstrings_in_code(result)
    return result[len("class Bla:\n") :] if has_indent else result


def check_codes_match(observed_code: str, theoretical_code: str) -> Optional[int]:
    """
    Checks if two version of a code match with the exception of the class/function name.

    Args:
        observed_code (`str`): The code found.
        theoretical_code (`str`): The code to match.

    Returns:
        `Optional[int]`: The index of the first line where there is a difference (if any) and `None` if the codes
        match.
    """
    observed_code_header = observed_code.split("\n")[0]
    theoretical_code_header = theoretical_code.split("\n")[0]

    # Catch the function/class name: it is expected that those do not match.
    _re_class_match = re.compile(r"class\s+([^\(:]+)(?:\(|:)")
    _re_func_match = re.compile(r"def\s+([^\(]+)\(")
    for re_pattern in [_re_class_match, _re_func_match]:
        if re_pattern.match(observed_code_header) is not None:
            observed_obj_name = re_pattern.search(observed_code_header).groups()[0]
            theoretical_name = re_pattern.search(theoretical_code_header).groups()[0]
            theoretical_code_header = theoretical_code_header.replace(theoretical_name, observed_obj_name)

    # Find the first diff. Line 0 is special since we need to compare with the function/class names ignored.
    diff_index = 0
    if theoretical_code_header != observed_code_header:
        return 0

    diff_index = 1
    for observed_line, theoretical_line in zip(observed_code.split("\n")[1:], theoretical_code.split("\n")[1:]):
        if observed_line != theoretical_line:
            return diff_index
        diff_index += 1


def is_copy_consistent(filename: str, overwrite: bool = False) -> Optional[List[Tuple[str, int]]]:
    """
    Check if the code commented as a copy in a file matches the original.

    Args:
        filename (`str`):
            The name of the file to check.
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to overwrite the copies when they don't match.

    Returns:
        `Optional[List[Tuple[str, int]]]`: If `overwrite=False`, returns the list of differences as tuples `(str, int)`
        with the name of the object having a diff and the line number where theere is the first diff.
    """
    with open(filename, "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()
    diffs = []
    line_index = 0
    # Not a for loop cause `lines` is going to change (if `overwrite=True`).
    while line_index < len(lines):
        search_re = _re_copy_warning
        if filename.startswith("tests"):
            search_re = _re_copy_warning_for_test_file

        search = search_re.search(lines[line_index])
        if search is None:
            line_index += 1
            continue

        # There is some copied code here, let's retrieve the original.
        indent, object_name, replace_pattern = search.groups()

        base_path = TRANSFORMERS_PATH if not filename.startswith("tests") else MODEL_TEST_PATH
        theoretical_code = find_code_in_transformers(object_name, base_path=base_path)
        theoretical_indent = get_indent(theoretical_code)

        start_index = line_index + 1 if indent == theoretical_indent else line_index
        line_index = start_index + 1

        subcode = "\n".join(theoretical_code.split("\n")[1:])
        indent = get_indent(subcode)
        # Loop to check the observed code, stop when indentation diminishes or if we see a End copy comment.
        should_continue = True
        while line_index < len(lines) and should_continue:
            line_index += 1
            if line_index >= len(lines):
                break
            line = lines[line_index]
            # There is a special pattern `# End copy` to stop early. It's not documented cause it shouldn't really be
            # used.
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

            theoretical_code = blackify(theoretical_code)

        # Test for a diff and act accordingly.
        diff_index = check_codes_match(observed_code, theoretical_code)
        if diff_index is not None:
            diffs.append([object_name, diff_index + start_index + 1])
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
    """
    Check every file is copy-consistent with the original. Also check the model list in the main README and other
    READMEs are consistent.

    Args:
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to overwrite the copies when they don't match.
    """
    all_files = glob.glob(os.path.join(TRANSFORMERS_PATH, "**/*.py"), recursive=True)
    all_test_files = glob.glob(os.path.join(MODEL_TEST_PATH, "**/*.py"), recursive=True)
    all_files = list(all_files) + list(all_test_files)

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
    """
    Check the files that are full copies of others (as indicated in `FULL_COPIES`) are copy-consistent.

    Args:
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to overwrite the copies when they don't match.
    """
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


def get_model_list(filename: str, start_prompt: str, end_prompt: str) -> str:
    """
    Extracts the model list from a README.

    Args:
        filename (`str`): The name of the README file to check.
        start_prompt (`str`): The string to look for that introduces the model list.
        end_prompt (`str`): The string to look for that ends the model list.

    Returns:
        `str`: The model list.
    """
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

    # Keep going until the end of the list.
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


def convert_to_localized_md(model_list: str, localized_model_list: str, format_str: str) -> Tuple[bool, str]:
    """
    Compare the model list from the main README to the one in a localized README.

    Args:
        model_list (`str`): The model list in the main README.
        localized_model_list (`str`): The model list in one of the localized README.
        format_str (`str`):
            The template for a model entry in the localized README (look at the `format_model_list` in the entries of
            `LOCALIZED_READMES` for examples).

    Returns:
        `Tuple[bool, str]`: A tuple where the first value indicates if the READMEs match or not, and the second value
        is the correct localized README.
    """

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
    # affiliations of the paper, title of the paper, authors of the paper, and supplemental data (see DistilBERT for
    # example).
    _re_capture_meta = re.compile(
        r"\*\*\[([^\]]*)\]\(([^\)]*)\)\*\* \(from ([^)]*)\)[^\[]*([^\)]*\)).*?by (.*?[A-Za-z\*]{2,}?)\. (.*)$"
    )
    # This regex is used to synchronize link.
    _re_capture_title_link = re.compile(r"\*\*\[([^\]]*)\]\(([^\)]*)\)\*\*")

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

    model_keys = [re.search(r"\*\*\[([^\]]*)", line).groups()[0] for line in model_list.strip().split("\n")]

    # We exclude keys in localized README not in the main one.
    readmes_match = not any(k not in model_keys for k in localized_model_index)
    localized_model_index = {k: v for k, v in localized_model_index.items() if k in model_keys}

    for model in model_list.strip().split("\n"):
        title, model_link = _re_capture_title_link.search(model).groups()
        if title not in localized_model_index:
            readmes_match = False
            # Add an anchor white space behind a model description string for regex.
            # If metadata cannot be captured, the English version will be directly copied.
            localized_model_index[title] = _re_capture_meta.sub(_rep, model + " ")
        elif _re_fill_pattern.search(localized_model_index[title]) is not None:
            update = _re_capture_meta.sub(_rep, model + " ")
            if update != localized_model_index[title]:
                readmes_match = False
                localized_model_index[title] = update
        else:
            # Synchronize link
            localized_model_index[title] = _re_capture_title_link.sub(
                f"**[{title}]({model_link})**", localized_model_index[title], count=1
            )

    sorted_index = sorted(localized_model_index.items(), key=lambda x: x[0].lower())

    return readmes_match, "\n".join((x[1] for x in sorted_index)) + "\n"


def _find_text_in_file(filename: str, start_prompt: str, end_prompt: str) -> Tuple[str, int, int, List[str]]:
    """
    Find the text in a file between two prompts.

    Args:
        filename (`str`): The name of the file to look into.
        start_prompt (`str`): The string to look for that introduces the content looked for.
        end_prompt (`str`): The string to look for that ends the content looked for.

    Returns:
        Tuple[str, int, int, List[str]]: The content between the two prompts, the index of the start line in the
        original file, the index of the end line in the original file and the list of lines of that file.
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


def check_model_list_copy(overwrite: bool = False):
    """
    Check the model lists in the README is consistent with the ones in the other READMES and also with `index.nmd`.

    Args:
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to overwrite the copies when they don't match.
    """
    # Fix potential doc links in the README
    with open(os.path.join(REPO_PATH, "README.md"), "r", encoding="utf-8", newline="\n") as f:
        readme = f.read()
    new_readme = readme.replace("https://huggingface.co/transformers", "https://huggingface.co/docs/transformers")
    new_readme = new_readme.replace(
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

    md_list = get_model_list(
        filename="README.md",
        start_prompt=LOCALIZED_READMES["README.md"]["start_prompt"],
        end_prompt=LOCALIZED_READMES["README.md"]["end_prompt"],
    )

    # Build the converted Markdown.
    converted_md_lists = []
    for filename, value in LOCALIZED_READMES.items():
        _start_prompt = value["start_prompt"]
        _end_prompt = value["end_prompt"]
        _format_model_list = value["format_model_list"]

        localized_md_list = get_model_list(filename, _start_prompt, _end_prompt)
        readmes_match, converted_md_list = convert_to_localized_md(md_list, localized_md_list, _format_model_list)

        converted_md_lists.append((filename, readmes_match, converted_md_list, _start_prompt, _end_prompt))

    # Compare the converted Markdowns
    for converted_md_list in converted_md_lists:
        filename, readmes_match, converted_md, _start_prompt, _end_prompt = converted_md_list

        if filename == "README.md":
            continue
        if overwrite:
            _, start_index, end_index, lines = _find_text_in_file(
                filename=os.path.join(REPO_PATH, filename), start_prompt=_start_prompt, end_prompt=_end_prompt
            )
            with open(os.path.join(REPO_PATH, filename), "w", encoding="utf-8", newline="\n") as f:
                f.writelines(lines[:start_index] + [converted_md] + lines[end_index:])
        elif not readmes_match:
            raise ValueError(
                f"The model list in the README changed and the list in `{filename}` has not been updated. Run "
                "`make fix-copies` to fix this."
            )


# Map a model name with the name it has in the README for the check_readme check
SPECIAL_MODEL_NAMES = {
    "Bert Generation": "BERT For Sequence Generation",
    "BigBird": "BigBird-RoBERTa",
    "Data2VecAudio": "Data2Vec",
    "Data2VecText": "Data2Vec",
    "Data2VecVision": "Data2Vec",
    "DonutSwin": "Swin Transformer",
    "Marian": "MarianMT",
    "MaskFormerSwin": "Swin Transformer",
    "OpenAI GPT-2": "GPT-2",
    "OpenAI GPT": "GPT",
    "Perceiver": "Perceiver IO",
    "SAM": "Segment Anything",
    "ViT": "Vision Transformer (ViT)",
}

# Update this list with the models that shouldn't be in the README. This only concerns modular models or those who do
# not have an associated paper.
MODELS_NOT_IN_README = [
    "BertJapanese",
    "Encoder decoder",
    "FairSeq Machine-Translation",
    "HerBERT",
    "RetriBERT",
    "Speech Encoder decoder",
    "Speech2Text",
    "Speech2Text2",
    "TimmBackbone",
    "Vision Encoder decoder",
    "VisionTextDualEncoder",
]

# Template for new entries to add in the main README when we have missing models.
README_TEMPLATE = (
    "1. **[{model_name}](https://huggingface.co/docs/main/transformers/model_doc/{model_type})** (from "
    "<FILL INSTITUTION>) released with the paper [<FILL PAPER TITLE>](<FILL ARKIV LINK>) by <FILL AUTHORS>."
)


def check_readme(overwrite: bool = False):
    """
    Check if the main README contains all the models in the library or not.

    Args:
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to add an entry for the missing models using `README_TEMPLATE`.
    """
    info = LOCALIZED_READMES["README.md"]
    models, start_index, end_index, lines = _find_text_in_file(
        os.path.join(REPO_PATH, "README.md"),
        info["start_prompt"],
        info["end_prompt"],
    )
    models_in_readme = [re.search(r"\*\*\[([^\]]*)", line).groups()[0] for line in models.strip().split("\n")]

    model_names_mapping = transformers_module.models.auto.configuration_auto.MODEL_NAMES_MAPPING
    absents = [
        (key, name)
        for key, name in model_names_mapping.items()
        if SPECIAL_MODEL_NAMES.get(name, name) not in models_in_readme
    ]
    # Remove exceptions
    absents = [(key, name) for key, name in absents if name not in MODELS_NOT_IN_README]
    if len(absents) > 0 and not overwrite:
        print(absents)
        raise ValueError(
            "The main README doesn't contain all models, run `make fix-copies` to fill it with the missing model(s)"
            " then complete the generated entries.\nIf the model is not supposed to be in the main README, add it to"
            " the list `MODELS_NOT_IN_README` in utils/check_copies.py.\nIf it has a different name in the repo than"
            " in the README, map the correspondence in `SPECIAL_MODEL_NAMES` in utils/check_copies.py."
        )

    new_models = [README_TEMPLATE.format(model_name=name, model_type=key) for key, name in absents]

    all_models = models.strip().split("\n") + new_models
    all_models = sorted(all_models, key=lambda x: re.search(r"\*\*\[([^\]]*)", x).groups()[0].lower())
    all_models = "\n".join(all_models) + "\n"

    if all_models != models:
        if overwrite:
            print("Fixing the main README.")
            with open(os.path.join(REPO_PATH, "README.md"), "w", encoding="utf-8", newline="\n") as f:
                f.writelines(lines[:start_index] + [all_models] + lines[end_index:])
        else:
            raise ValueError("The main README model list is not properly sorted. Run `make fix-copies` to fix this.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    check_readme(args.fix_and_overwrite)
    check_copies(args.fix_and_overwrite)
    check_full_copies(args.fix_and_overwrite)
