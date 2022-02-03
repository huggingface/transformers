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


BLACK_AVOID_PATTERNS = {
    "===PT-TF-SPLIT===": "### PT-TF-SPLIT",
    "{processor_class}": "FakeProcessorClass",
    "{model_class}": "FakeModelClass",
    "{object_class}": "FakeObjectClass",
}


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
        max_len (`int`): The maximum lengh per line.

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


def format_code_example(code: str, max_len: int, in_docstring: bool = False):
    """
    Format a code example using black. Will take into account the doctest syntax as well as any initial indentation in
    the code provided.

    Args:
        code (`str`): The code example to format.
        max_len (`int`): The maximum lengh per line.
        in_docstring (`bool`, *optional*, defaults to `False`): Whether or not the code example is inside a docstring.

    Returns:
        `str`: The formatted code.
    """
    code_lines = code.split("\n")

    # Find initial indent
    idx = 0
    while idx < len(code_lines) and is_empty_line(code_lines[idx]):
        idx += 1
    if idx >= len(code_lines):
        return "", ""
    indent = find_indent(code_lines[idx])

    # Remove the initial indent for now, we will had it back after styling.
    # Note that l[indent:] works for empty lines
    code_lines = [l[indent:] for l in code_lines[idx:]]
    has_doctest = code_lines[0][:3] in DOCTEST_PROMPTS

    code_samples, outputs = parse_code_example(code_lines)

    # Let's blackify the code! We put everything in one big text to go faster.
    delimiter = "\n\n### New code sample ###\n"
    full_code = delimiter.join(code_samples)
    line_length = max_len - indent
    if has_doctest:
        line_length -= 4

    for k, v in BLACK_AVOID_PATTERNS.items():
        full_code = full_code.replace(k, v)
    try:
        formatted_code = black.format_str(
            full_code, mode=black.FileMode([black.TargetVersion.PY37], line_length=line_length)
        )
        error = ""
    except Exception as e:
        formatted_code = full_code
        error = f"Code sample:\n{full_code}\n\nError message:\n{e}"

    # Let's get back the formatted code samples
    for k, v in BLACK_AVOID_PATTERNS.items():
        formatted_code = formatted_code.replace(v, k)
    # Triple quotes will mess docstrings.
    if in_docstring:
        formatted_code = formatted_code.replace('"""', "'''")

    code_samples = formatted_code.split(delimiter)
    # We can have one output less than code samples
    if len(outputs) == len(code_samples) - 1:
        outputs.append("")

    formatted_lines = []
    for code_sample, output in zip(code_samples, outputs):
        # black may have added some new lines, we remove them
        code_sample = code_sample.strip()
        in_triple_quotes = False
        for line in code_sample.strip().split("\n"):
            if has_doctest and not is_empty_line(line):
                prefix = "... " if line.startswith(" ") or line in [")", "]", "}"] or in_triple_quotes else ">>> "
            else:
                prefix = ""
            indent_str = "" if is_empty_line(line) else (" " * indent)
            formatted_lines.append(indent_str + prefix + line)

            if '"""' in line:
                in_triple_quotes = not in_triple_quotes

        formatted_lines.extend([" " * indent + line for line in output.split("\n")])
        if not output.endswith("===PT-TF-SPLIT==="):
            formatted_lines.append("")

    result = "\n".join(formatted_lines)
    return result.rstrip(), error


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
    black_errors = []

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
                    formatted_code, error = format_code_example(code, max_len, in_docstring=True)
                    new_lines.append(formatted_code)
                    if len(error) > 0:
                        black_errors.append(error)
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

    return "\n".join(new_lines), "\n\n".join(black_errors)


def style_docstrings_in_code(code, max_len=119):
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
        (s if i % 2 == 0 or _re_doc_ignore.search(splits[i - 1]) is not None else style_docstring(s, max_len=max_len))
        for i, s in enumerate(splits)
    ]
    black_errors = "\n\n".join([s[1] for s in splits if isinstance(s, tuple) and len(s[1]) > 0])
    splits = [s[0] if isinstance(s, tuple) else s for s in splits]
    clean_code = '\"\"\"'.join(splits)
    # fmt: on

    return clean_code, black_errors


def style_file_docstrings(code_file, max_len=119, check_only=False):
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

    clean_code, black_errors = style_docstrings_in_code(code, max_len=max_len)

    diff = clean_code != code
    if not check_only and diff:
        print(f"Overwriting content of {code_file}.")
        with open(code_file, "w", encoding="utf-8", newline="\n") as f:
            f.write(clean_code)

    return diff, black_errors


def style_mdx_file(mdx_file, max_len=119, check_only=False):
    """
    Style a MDX file by formatting all Python code samples.

    Args:
        mdx_file (`str` or `os.PathLike`): The file in which we want to style the examples.
        max_len (`int`): The maximum number of characters per line.
        check_only (`bool`, *optional*, defaults to `False`):
            Whether to restyle file or just check if they should be restyled.

    Returns:
        `bool`: Whether or not the file was or should be restyled.
    """
    with open(mdx_file, "r", encoding="utf-8", newline="\n") as f:
        content = f.read()

    lines = content.split("\n")
    current_code = []
    current_language = ""
    in_code = False
    new_lines = []
    black_errors = []

    for line in lines:
        if _re_code.search(line) is not None:
            in_code = not in_code
            if in_code:
                current_language = _re_code.search(line).groups()[1]
                current_code = []
            else:
                code = "\n".join(current_code)
                if current_language in ["py", "python"]:
                    code, error = format_code_example(code, max_len)
                    if len(error) > 0:
                        black_errors.append(error)
                new_lines.append(code)

            new_lines.append(line)
        elif in_code:
            current_code.append(line)
        else:
            new_lines.append(line)

    if in_code:
        raise ValueError(f"There was a problem when styling {mdx_file}. A code block is opened without being closed.")

    clean_content = "\n".join(new_lines)
    diff = clean_content != content
    if not check_only and diff:
        print(f"Overwriting content of {mdx_file}.")
        with open(mdx_file, "w", encoding="utf-8", newline="\n") as f:
            f.write(clean_content)

    return diff, "\n\n".join(black_errors)


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
    black_errors = []
    for file in files:
        # Treat folders
        if os.path.isdir(file):
            files = [os.path.join(file, f) for f in os.listdir(file)]
            files = [f for f in files if os.path.isdir(f) or f.endswith(".mdx") or f.endswith(".py")]
            changed += style_doc_files(*files, max_len=max_len, check_only=check_only)
        # Treat mdx
        elif file.endswith(".mdx"):
            try:
                diff, black_error = style_mdx_file(file, max_len=max_len, check_only=check_only)
                if diff:
                    changed.append(file)
                if len(black_error) > 0:
                    black_errors.append(
                        f"There was a problem while formatting an example in {file} with black:\m{black_error}"
                    )
            except Exception:
                print(f"There is a problem in {file}.")
                raise
        # Treat python files
        elif file.endswith(".py"):
            try:
                diff, black_error = style_file_docstrings(file, max_len=max_len, check_only=check_only)
                if diff:
                    changed.append(file)
                if len(black_error) > 0:
                    black_errors.append(
                        f"There was a problem while formatting an example in {file} with black:\m{black_error}"
                    )
            except Exception:
                print(f"There is a problem in {file}.")
                raise
        else:
            warnings.warn(f"Ignoring {file} because it's not a py or an mdx file or a folder.")
    if len(black_errors) > 0:
        black_message = "\n\n".join(black_errors)
        raise ValueError(
            "Some code examples can't be interpreted by black, which means they aren't regular python:\n\n"
            + black_message
            + "\n\nMake sure to fix the corresponding docstring or doc file, or remove the py/python after ``` if it "
            + "was not supposed to be a Python code sample."
        )
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
