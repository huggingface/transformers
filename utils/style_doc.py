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
import warnings

import black


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


DOCTEST_PROMPTS = [">>> ", "... "]


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


def format_text(text, max_len, prefix="", min_indent=None):
    """
    Format a text in the biggest lines possible with the constraint of a maximum length and an indentation.

    Args:
        text (`str`): The text to format
        max_len (`int`): The maximum length per line to use
        prefix (`str`, *optional*, defaults to `""`): A prefix that will be added to the text.
            The prefix doesn't count toward the indent (like a - introducing a list).
        min_indent (`int`, *optional*): The minimum indent of the text.
            If not set, will default to the length of the `prefix`.

    Returns:
        `str`: The formatted text.
    """
    text = re.sub(r"\s+", " ", text)
    if min_indent is not None:
        if len(prefix) < min_indent:
            prefix = " " * (min_indent - len(prefix)) + prefix

    indent = " " * len(prefix)
    new_lines = []
    words = text.split(" ")
    current_line = f"{prefix}{words[0]}"
    for word in words[1:]:
        try_line = f"{current_line} {word}"
        if len(try_line) > max_len:
            new_lines.append(current_line)
            current_line = f"{indent}{word}"
        else:
            current_line = try_line
    new_lines.append(current_line)
    return "\n".join(new_lines)


def parse_code_example(code_lines):
    """
    Parses a code example

    Args:
        code_lines (`List[str]`): The code lines to parse.
        max_len (`int`): The maximum lengh per line.

    Returns:
        (List[`str`], List[`str`]): The list of code samples and the list of outputs.
    """
    has_doctest = code_lines[0][:4] in DOCTEST_PROMPTS

    code_samples = []
    outputs = []
    in_code = True
    current_bit = []

    for line in code_lines:
        if in_code and has_doctest and not is_empty_line(line) and line[:4] not in DOCTEST_PROMPTS:
            code_sample = "\n".join(current_bit)
            code_samples.append(code_sample.strip())
            in_code = False
            current_bit = []
        elif not in_code and line[:4] in DOCTEST_PROMPTS:
            output = "\n".join(current_bit)
            outputs.append(output.strip())
            in_code = True
            current_bit = []

        # Add the line without doctest prompt
        if line[:4] in DOCTEST_PROMPTS:
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


def format_code_example(code: str, max_len: int):
    """
    Format a code example using black. Will take into account the doctest syntax as well as any initial indentation in
    the code provided.

    Args:
        code (`str`): The code example to format.
        max_len (`int`): The maximum lengh per line.

    Returns:
        `str`: The formatted code.
    """
    code_lines = code.split("\n")

    # Find initial indent
    idx = 0
    while idx < len(code_lines) and is_empty_line(code_lines[idx]):
        idx += 1
    if idx >= len(code_lines):
        return ""
    indent = find_indent(code_lines[idx])
    has_doctest = code_lines[0][:4] in DOCTEST_PROMPTS

    # Remove the initial indent for now, we will had it back after styling.
    # Note that l[indent:] works for empty lines
    code_lines = [l[indent:] for l in code_lines[idx:]]

    code_samples, outputs = parse_code_example(code_lines)

    # Let's blackify the code! We put everything in one big text to go faster.
    delimiter = "\n\n### New code sample ###\n"
    full_code = delimiter.join(code_samples)
    line_length = max_len - indent
    if has_doctest:
        line_length -= 4
    formatted_code = black.format_str(
        full_code, mode=black.FileMode([black.TargetVersion.PY37], line_length=line_length)
    )

    # Let's get back the formatted code samples
    code_samples = formatted_code.split(delimiter)
    # We can have one output less than code samples
    if len(outputs) == len(code_samples) - 1:
        outputs.append("")

    formatted_lines = []
    for code_sample, output in zip(code_samples, outputs):
        # black may have added some new lines, we remove them
        code_sample = code_sample.strip()
        for line in code_sample.strip().split("\n"):
            if has_doctest and not is_empty_line(line):
                prefix = "... " if line.startswith(" ") else ">>> "
            else:
                prefix = ""
            formatted_lines.append(" " * indent + prefix + line)

        formatted_lines.extend([" " * indent + line for line in output.split("\n")])
        formatted_lines.append("")

    result = "\n".join(formatted_lines)
    return result.rstrip()


def split_line_on_first_colon(line):
    splits = line.split(":")
    return splits[0], ":".join(splits[1:])


def style_docstring(docstring, max_len):
    """
    Style a docstring by making sure there is no useless whitespace and the maximum horizontal space is used.

    Args:
        docstring (`str`): The docstring to style.
        max_len (`int`): The maximum length of each line.

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

        # In this case, we treat the current paragraph
        if not in_code and new_paragraph and current_paragraph is not None and len(current_paragraph) > 0:
            paragraph = " ".join(current_paragraph)
            new_lines.append(format_text(paragraph, max_len, prefix=prefix, min_indent=current_indent))
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
                    new_lines.append(format_code_example(code, max_len))
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
        elif current_paragraph is None or find_indent(line) != current_indent:
            indent = find_indent(line)
            # Special behavior for parameters intros.
            if indent == param_indent:
                # Special rules for some docstring where the Returns blocks has the same indent as the parameters.
                if _re_returns.search(line) is not None:
                    param_indent = -1
                    new_lines.append(line)
                elif len(line) < max_len:
                    new_lines.append(line)
                else:
                    intro, description = split_line_on_first_colon(line)
                    new_lines.append(intro + ":")
                    if len(description) != 0:
                        if find_indent(lines[idx + 1]) > indent:
                            current_indent = find_indent(lines[idx + 1])
                        else:
                            current_indent = indent + 4
                        current_paragraph = [description.strip()]
                        prefix = ""
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
        new_lines.append(format_text(paragraph, max_len, prefix=prefix, min_indent=current_indent))

    return "\n".join(new_lines)


def style_file_docstrings(code_file, max_len=119, check_only=False):
    """
    Style all docstrings in  a given file.

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
    # fmt: off
    splits = code.split('\"\"\"')
    splits = [
        (s if i % 2 == 0 or _re_doc_ignore.search(splits[i - 1]) is not None else style_docstring(s, max_len=max_len))
        for i, s in enumerate(splits)
    ]
    clean_code = '\"\"\"'.join(splits)
    # fmt: on

    diff = clean_code != code
    if not check_only and diff:
        print(f"Overwriting content of {code_file}.")
        with open(code_file, "w", encoding="utf-8", newline="\n") as f:
            f.write(clean_code)

    return diff


def style_doc_files(*files, max_len=119, check_only=False):
    """
    Applies doc styling or checks everything is correct in a list of files.

    Args:
        files (several `str` or `os.PathLike`): The files to treat.
        max_len (`int`): The maximum number of characters per line.
        check_only (`bool`, *optional*, defaults to `False`):
            Whether to restyle file or just check if they should be restyled.

    Returns:
        List[`str`]: The list of files changed or that should be restyled.
    """
    changed = []
    for file in files:
        # Treat folders
        if os.path.isdir(file):
            files = [os.path.join(file, f) for f in os.listdir(file)]
            files = [f for f in files if os.path.isdir(f) or f.endswith(".rst") or f.endswith(".py")]
            changed += style_doc_files(*files, max_len=max_len, check_only=check_only)
        # Treat mdx
        elif file.endswith(".mdx"):
            pass
            # Will add code samples black styling in the mdx files.
            # if style_mdx_file(file, max_len=max_len, check_only=check_only):
            #     changed.append(file)
        # Treat python files
        elif file.endswith(".py"):
            if style_file_docstrings(file, max_len=max_len, check_only=check_only):
                changed.append(file)
        else:
            warnings.warn(f"Ignoring {file} because it's not a py or an mdx file or a folder.")
    return changed


def main(*files, max_len=119, check_only=False):
    changed = style_doc_files(*files, max_len=max_len, check_only=check_only)
    if check_only and len(changed) > 0:
        raise ValueError(f"{len(changed)} files should be restyled!")
    elif len(changed) > 0:
        print(f"Cleaned {len(changed)} files!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="The file(s) or folder(s) to restyle.")
    parser.add_argument("--max_len", type=int, help="The maximum length of lines.")
    parser.add_argument("--check_only", action="store_true", help="Whether to only check and not fix styling issues.")
    args = parser.parse_args()

    main(*args.files, max_len=args.max_len, check_only=args.check_only)
