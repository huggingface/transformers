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
"""Style utils for the .rst and the docstrings."""

import argparse
import os
import re


# Regexes
# Re pattern that catches list introduction (with potential indent)
_re_list = re.compile(r"^(\s*-\s+|\s*\*\s+|\s*\d+\.\s+)")
# Re pattern that catches code block introduction (with potentinal indent)
_re_code = re.compile(r"^(\s*)```(.*)$")
# Re pattern that catches rst args blocks of the form `Parameters:`.
_re_args = re.compile("^\s*(Args?|Arguments?|Params?|Parameters?):\s*$")
# Re pattern that catches return blocks of the form `Return:`.
_re_returns = re.compile("^\s*Returns?:\s*$")
# Matches the special tag to ignore some paragraphs.
_re_doc_ignore = re.compile(r"(\.\.|#)\s*docstyle-ignore")
# Re pattern that matches <Tip>, </Tip> and <Tip warning={true}> blocks.
_re_tip = re.compile("^\s*</?Tip(>|\s+warning={true}>)\s*$")

DOCTEST_PROMPTS = [">>>", "..."]


def is_empty_line(line):
    return len(line) == 0 or line.isspace()


def find_indent(line):
    """
    Returns the number of spaces that start a line indent.
    """
    search = re.search("^(\s*)(?:\S|$)", line)
    if search is None:
        return 0
    return len(search.groups()[0])


def parse_code_example(code_lines):
    """
    Parses a code example

    Args:
        code_lines (`List[str]`): The code lines to parse.

    Returns:
        (List[`str`], List[`str`]): The list of code samples and the list of outputs.
    """
    has_doctest = code_lines[0][:3] in DOCTEST_PROMPTS

    code_samples = []
    outputs = []
    in_code = True
    current_bit = []

    for line in code_lines:
        if in_code and has_doctest and not is_empty_line(line) and line[:3] not in DOCTEST_PROMPTS:
            code_sample = "\n".join(current_bit)
            code_samples.append(code_sample.strip())
            in_code = False
            current_bit = []
        elif not in_code and line[:3] in DOCTEST_PROMPTS:
            output = "\n".join(current_bit)
            outputs.append(output.strip())
            in_code = True
            current_bit = []

        # Add the line without doctest prompt
        if line[:3] in DOCTEST_PROMPTS:
            line = line[4:]
        current_bit.append(line)

    # Add last sample
    if in_code:
        code_sample = "\n".join(current_bit)
        code_samples.append(code_sample.strip())
    else:
        output = "\n".join(current_bit)
        outputs.append(output.strip())

    return code_samples, outputs


def style_docstring(docstring):
    """
    Style a docstring by making sure there is no useless whitespace and the maximum horizontal space is used.

    Args:
        docstring (`str`): The docstring to style.

    Returns:
        `str`: The styled docstring
    """
    lines = docstring.split("\n")
    new_lines = []

    # Initialization
    current_paragraph = None
    current_indent = -1
    in_code = False
    param_indent = -1
    prefix = ""

    # Special case for docstrings that begin with continuation of Args with no Args block.
    idx = 0
    while idx < len(lines) and is_empty_line(lines[idx]):
        idx += 1
    if (
        len(lines[idx]) > 1
        and lines[idx].rstrip().endswith(":")
        and find_indent(lines[idx + 1]) > find_indent(lines[idx])
    ):
        param_indent = find_indent(lines[idx])

    for idx, line in enumerate(lines):
        # Doing all re searches once for the one we need to repeat.
        list_search = _re_list.search(line)
        code_search = _re_code.search(line)

        # Are we starting a new paragraph?
        # New indentation or new line:
        new_paragraph = find_indent(line) != current_indent or is_empty_line(line)
        # List item
        new_paragraph = new_paragraph or list_search is not None
        # Code block beginning
        new_paragraph = new_paragraph or code_search is not None
        # Beginning/end of tip
        new_paragraph = new_paragraph or _re_tip.search(line)

        if not in_code and new_paragraph and current_paragraph is not None and len(current_paragraph) > 0:
            paragraph = " ".join(current_paragraph)
            new_lines.append(paragraph)
            current_paragraph = None

        if code_search is not None:
            if not in_code:
                current_paragraph = []
                current_indent = len(code_search.groups()[0])
                current_code = code_search.groups()[1]
                prefix = ""
                if current_indent < param_indent:
                    param_indent = -1
            else:
                current_indent = -1
                code = "\n".join(current_paragraph)
                if current_code in ["py", "python"]:
                    # add empty space in the end
                    formatted_code = code + "\n"
                    new_lines.append(formatted_code)
                else:
                    new_lines.append(code)
                current_paragraph = None
            new_lines.append(line)
            in_code = not in_code

        elif in_code:
            current_paragraph.append(line)
        elif is_empty_line(line):
            current_paragraph = None
            current_indent = -1
            prefix = ""
            new_lines.append(line)
        elif list_search is not None:
            prefix = list_search.groups()[0]
            current_indent = len(prefix)
            current_paragraph = [line[current_indent:]]
        elif _re_args.search(line):
            new_lines.append(line)
            param_indent = find_indent(lines[idx + 1])
        elif _re_tip.search(line):
            # Add a new line before if not present
            if not is_empty_line(new_lines[-1]):
                new_lines.append("")
            new_lines.append(line)
            # Add a new line after if not present
            if idx < len(lines) - 1 and not is_empty_line(lines[idx + 1]):
                new_lines.append("")
        elif current_paragraph is None or find_indent(line) != current_indent:
            indent = find_indent(line)
            # Special behavior for parameters intros.
            if indent == param_indent:
                # Special rules for some docstring where the Returns blocks has the same indent as the parameters.
                if _re_returns.search(line) is not None:
                    param_indent = -1
                    new_lines.append(line)
                else:
                    new_lines.append(line)
            else:
                # Check if we have exited the parameter block
                if indent < param_indent:
                    param_indent = -1

                current_paragraph = [line.strip()]
                current_indent = find_indent(line)
                prefix = ""
        elif current_paragraph is not None:
            current_paragraph.append(line.lstrip())

    if current_paragraph is not None and len(current_paragraph) > 0:
        paragraph = " ".join(current_paragraph)
        new_lines.append(paragraph)

    return "\n".join(new_lines)


def style_docstrings_in_code(code):
    """
    Style all docstrings in some code.

    Args:
        code (`str`): The code in which we want to style the docstrings.
        max_len (`int`): The maximum number of characters per line.

    Returns:
        `Tuple[str, str]`: A tuple with the clean code and the black errors (if any)
    """
    # fmt: off
    splits = code.split('\"\"\"')
    splits = [
        s if i % 2 == 0 or _re_doc_ignore.search(splits[i - 1]) is not None else style_docstring(s)
        for i, s in enumerate(splits)
    ]
    splits = [s[0] if isinstance(s, tuple) else s for s in splits]
    clean_code = '\"\"\"'.join(splits)
    # fmt: on

    return clean_code


def style_file_docstrings(code_file):
    """
    Style all docstrings in a given file.

    Args:
        code_file (`str` or `os.PathLike`): The file in which we want to style the docstring.
        max_len (`int`): The maximum number of characters per line.
        check_only (`bool`, *optional*, defaults to `False`):
            Whether to restyle file or just check if they should be restyled.

    Returns:
        `bool`: Whether or not the file was or should be restyled.
    """
    with open(code_file, "r", encoding="utf-8", newline="\n") as f:
        code = f.read()

    clean_code = style_docstrings_in_code(code)

    diff = clean_code != code
    if diff:
        print(f"Overwriting content of {code_file}.")
        with open(code_file, "w", encoding="utf-8", newline="\n") as f:
            f.write(clean_code)


def process_doc_files(*files):
    """
    Applies doc styling or checks everything is correct in a list of files.

    Args:
        files (several `str` or `os.PathLike`): The files to treat.

    Returns:
        List[`str`]: The list of files changed or that should be restyled.
    """
    for file in files:
        # Treat folders
        if os.path.isdir(file):
            files = [os.path.join(file, f) for f in os.listdir(file)]
            files = [f for f in files if os.path.isdir(f) or f.endswith(".mdx") or f.endswith(".py")]
            process_doc_files(*files)
        else:
            try:
                style_file_docstrings(file)
            except Exception:
                print(f"There is a problem in {file}.")
                raise


def main(*files):
    process_doc_files(*files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="The file(s) or folder(s) to restyle.")
    args = parser.parse_args()

    main(*args.files)
