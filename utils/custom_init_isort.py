# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
Utility that sorts the imports in the custom inits of Transformers. Transformers uses init files that delay the
import of an object to when it's actually needed. This is to avoid the main init importing all models, which would
make the line `import transformers` very slow when the user has all optional dependencies installed. The inits with
delayed imports have two halves: one definining a dictionary `_import_structure` which maps modules to the name of the
objects in each module, and one in `TYPE_CHECKING` which looks like a normal init for type-checkers. `isort` or `ruff`
properly sort the second half which looks like traditionl imports, the goal of this script is to sort the first half.

Use from the root of the repo with:

```bash
python utils/custom_init_isort.py
```

which will auto-sort the imports (used in `make style`).

For a check only (as used in `make quality`) run:

```bash
python utils/custom_init_isort.py --check_only
```
"""

import argparse
import os
import re
from typing import Any, Callable, List, Optional


# Path is defined with the intent you should run this script from the root of the repo.
PATH_TO_TRANSFORMERS = "src/transformers"

# Pattern that looks at the indentation in a line.
_re_indent = re.compile(r"^(\s*)\S")
# Pattern that matches `"key":" and puts `key` in group 0.
_re_direct_key = re.compile(r'^\s*"([^"]+)":')
# Pattern that matches `_import_structure["key"]` and puts `key` in group 0.
_re_indirect_key = re.compile(r'^\s*_import_structure\["([^"]+)"\]')
# Pattern that matches `"key",` and puts `key` in group 0.
_re_strip_line = re.compile(r'^\s*"([^"]+)",\s*$')
# Pattern that matches any `[stuff]` and puts `stuff` in group 0.
_re_bracket_content = re.compile(r"\[([^\]]+)\]")


def get_indent(line: str) -> str:
    """Returns the indent in  given line (as string)."""
    search = _re_indent.search(line)
    return "" if search is None else search.groups()[0]


def split_code_in_indented_blocks(
    code: str, indent_level: str = "", start_prompt: Optional[str] = None, end_prompt: Optional[str] = None
) -> List[str]:
    """
    Split some code into its indented blocks, starting at a given level.

    Args:
        code (`str`): The code to split.
        indent_level (`str`): The indent level (as string) to use for identifying the blocks to split.
        start_prompt (`str`, *optional*): If provided, only starts splitting at the line where this text is.
        end_prompt (`str`, *optional*): If provided, stops splitting at a line where this text is.

    Warning:
        The text before `start_prompt` or after `end_prompt` (if provided) is not ignored, just not split. The input `code`
        can thus be retrieved by joining the result.

    Returns:
        `List[str]`: The list of blocks.
    """
    # Let's split the code into lines and move to start_index.
    index = 0
    lines = code.split("\n")
    if start_prompt is not None:
        while not lines[index].startswith(start_prompt):
            index += 1
        blocks = ["\n".join(lines[:index])]
    else:
        blocks = []

    # This variable contains the block treated at a given time.
    current_block = [lines[index]]
    index += 1
    # We split into blocks until we get to the `end_prompt` (or the end of the file).
    while index < len(lines) and (end_prompt is None or not lines[index].startswith(end_prompt)):
        # We have a non-empty line with the proper indent -> start of a new block
        if len(lines[index]) > 0 and get_indent(lines[index]) == indent_level:
            # Store the current block in the result and rest. There are two cases: the line is part of the block (like
            # a closing parenthesis) or not.
            if len(current_block) > 0 and get_indent(current_block[-1]).startswith(indent_level + " "):
                # Line is part of the current block
                current_block.append(lines[index])
                blocks.append("\n".join(current_block))
                if index < len(lines) - 1:
                    current_block = [lines[index + 1]]
                    index += 1
                else:
                    current_block = []
            else:
                # Line is not part of the current block
                blocks.append("\n".join(current_block))
                current_block = [lines[index]]
        else:
            # Just add the line to the current block
            current_block.append(lines[index])
        index += 1

    # Adds current block if it's nonempty.
    if len(current_block) > 0:
        blocks.append("\n".join(current_block))

    # Add final block after end_prompt if provided.
    if end_prompt is not None and index < len(lines):
        blocks.append("\n".join(lines[index:]))

    return blocks


def ignore_underscore_and_lowercase(key: Callable[[Any], str]) -> Callable[[Any], str]:
    """
    Wraps a key function (as used in a sort) to lowercase and ignore underscores.
    """

    def _inner(x):
        return key(x).lower().replace("_", "")

    return _inner


def sort_objects(objects: List[Any], key: Optional[Callable[[Any], str]] = None) -> List[Any]:
    """
    Sort a list of objects following the rules of isort (all uppercased first, camel-cased second and lower-cased
    last).

    Args:
        objects (`List[Any]`):
            The list of objects to sort.
        key (`Callable[[Any], str]`, *optional*):
            A function taking an object as input and returning a string, used to sort them by alphabetical order.
            If not provided, will default to noop (so a `key` must be provided if the `objects` are not of type string).

    Returns:
        `List[Any]`: The sorted list with the same elements as in the inputs
    """

    # If no key is provided, we use a noop.
    def noop(x):
        return x

    if key is None:
        key = noop
    # Constants are all uppercase, they go first.
    constants = [obj for obj in objects if key(obj).isupper()]
    # Classes are not all uppercase but start with a capital, they go second.
    classes = [obj for obj in objects if key(obj)[0].isupper() and not key(obj).isupper()]
    # Functions begin with a lowercase, they go last.
    functions = [obj for obj in objects if not key(obj)[0].isupper()]

    # Then we sort each group.
    key1 = ignore_underscore_and_lowercase(key)
    return sorted(constants, key=key1) + sorted(classes, key=key1) + sorted(functions, key=key1)


def sort_objects_in_import(import_statement: str) -> str:
    """
    Sorts the imports in a single import statement.

    Args:
        import_statement (`str`): The import statement in which to sort the imports.

    Returns:
        `str`: The same as the input, but with objects properly sorted.
    """

    # This inner function sort imports between [ ].
    def _replace(match):
        imports = match.groups()[0]
        # If there is one import only, nothing to do.
        if "," not in imports:
            return f"[{imports}]"
        keys = [part.strip().replace('"', "") for part in imports.split(",")]
        # We will have a final empty element if the line finished with a comma.
        if len(keys[-1]) == 0:
            keys = keys[:-1]
        return "[" + ", ".join([f'"{k}"' for k in sort_objects(keys)]) + "]"

    lines = import_statement.split("\n")
    if len(lines) > 3:
        # Here we have to sort internal imports that are on several lines (one per name):
        # key: [
        #     "object1",
        #     "object2",
        #     ...
        # ]

        # We may have to ignore one or two lines on each side.
        idx = 2 if lines[1].strip() == "[" else 1
        keys_to_sort = [(i, _re_strip_line.search(line).groups()[0]) for i, line in enumerate(lines[idx:-idx])]
        sorted_indices = sort_objects(keys_to_sort, key=lambda x: x[1])
        sorted_lines = [lines[x[0] + idx] for x in sorted_indices]
        return "\n".join(lines[:idx] + sorted_lines + lines[-idx:])
    elif len(lines) == 3:
        # Here we have to sort internal imports that are on one separate line:
        # key: [
        #     "object1", "object2", ...
        # ]
        if _re_bracket_content.search(lines[1]) is not None:
            lines[1] = _re_bracket_content.sub(_replace, lines[1])
        else:
            keys = [part.strip().replace('"', "") for part in lines[1].split(",")]
            # We will have a final empty element if the line finished with a comma.
            if len(keys[-1]) == 0:
                keys = keys[:-1]
            lines[1] = get_indent(lines[1]) + ", ".join([f'"{k}"' for k in sort_objects(keys)])
        return "\n".join(lines)
    else:
        # Finally we have to deal with imports fitting on one line
        import_statement = _re_bracket_content.sub(_replace, import_statement)
        return import_statement


def sort_imports(file: str, check_only: bool = True):
    """
    Sort the imports defined in the `_import_structure` of a given init.

    Args:
        file (`str`): The path to the init to check/fix.
        check_only (`bool`, *optional*, defaults to `True`): Whether or not to just check (and not auto-fix) the init.
    """
    with open(file, encoding="utf-8") as f:
        code = f.read()

    # If the file is not a custom init, there is nothing to do.
    if "_import_structure" not in code or "define_import_structure" in code:
        return

    # Blocks of indent level 0
    main_blocks = split_code_in_indented_blocks(
        code, start_prompt="_import_structure = {", end_prompt="if TYPE_CHECKING:"
    )

    # We ignore block 0 (everything untils start_prompt) and the last block (everything after end_prompt).
    for block_idx in range(1, len(main_blocks) - 1):
        # Check if the block contains some `_import_structure`s thingy to sort.
        block = main_blocks[block_idx]
        block_lines = block.split("\n")

        # Get to the start of the imports.
        line_idx = 0
        while line_idx < len(block_lines) and "_import_structure" not in block_lines[line_idx]:
            # Skip dummy import blocks
            if "import dummy" in block_lines[line_idx]:
                line_idx = len(block_lines)
            else:
                line_idx += 1
        if line_idx >= len(block_lines):
            continue

        # Ignore beginning and last line: they don't contain anything.
        internal_block_code = "\n".join(block_lines[line_idx:-1])
        indent = get_indent(block_lines[1])
        # Slit the internal block into blocks of indent level 1.
        internal_blocks = split_code_in_indented_blocks(internal_block_code, indent_level=indent)
        # We have two categories of import key: list or _import_structure[key].append/extend
        pattern = _re_direct_key if "_import_structure = {" in block_lines[0] else _re_indirect_key
        # Grab the keys, but there is a trap: some lines are empty or just comments.
        keys = [(pattern.search(b).groups()[0] if pattern.search(b) is not None else None) for b in internal_blocks]
        # We only sort the lines with a key.
        keys_to_sort = [(i, key) for i, key in enumerate(keys) if key is not None]
        sorted_indices = [x[0] for x in sorted(keys_to_sort, key=lambda x: x[1])]

        # We reorder the blocks by leaving empty lines/comments as they were and reorder the rest.
        count = 0
        reorderded_blocks = []
        for i in range(len(internal_blocks)):
            if keys[i] is None:
                reorderded_blocks.append(internal_blocks[i])
            else:
                block = sort_objects_in_import(internal_blocks[sorted_indices[count]])
                reorderded_blocks.append(block)
                count += 1

        # And we put our main block back together with its first and last line.
        main_blocks[block_idx] = "\n".join(block_lines[:line_idx] + reorderded_blocks + [block_lines[-1]])

    if code != "\n".join(main_blocks):
        if check_only:
            return True
        else:
            print(f"Overwriting {file}.")
            with open(file, "w", encoding="utf-8") as f:
                f.write("\n".join(main_blocks))


def sort_imports_in_all_inits(check_only=True):
    """
    Sort the imports defined in the `_import_structure` of all inits in the repo.

    Args:
        check_only (`bool`, *optional*, defaults to `True`): Whether or not to just check (and not auto-fix) the init.
    """
    failures = []
    for root, _, files in os.walk(PATH_TO_TRANSFORMERS):
        if "__init__.py" in files:
            result = sort_imports(os.path.join(root, "__init__.py"), check_only=check_only)
            if result:
                failures = [os.path.join(root, "__init__.py")]
    if len(failures) > 0:
        raise ValueError(f"Would overwrite {len(failures)} files, run `make style`.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_only", action="store_true", help="Whether to only check or fix style.")
    args = parser.parse_args()

    sort_imports_in_all_inits(check_only=args.check_only)
