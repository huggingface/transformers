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

import argparse
import os
import re


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


def get_indent(line):
    """Returns the indent in `line`."""
    search = _re_indent.search(line)
    return "" if search is None else search.groups()[0]


def split_code_in_indented_blocks(code, indent_level="", start_prompt=None, end_prompt=None):
    """
    Split `code` into its indented blocks, starting at `indent_level`. If provided, begins splitting after
    `start_prompt` and stops at `end_prompt` (but returns what's before `start_prompt` as a first block and what's
    after `end_prompt` as a last block, so `code` is always the same as joining the result of this function).
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

    # We split into blocks until we get to the `end_prompt` (or the end of the block).
    current_block = [lines[index]]
    index += 1
    while index < len(lines) and (end_prompt is None or not lines[index].startswith(end_prompt)):
        if len(lines[index]) > 0 and get_indent(lines[index]) == indent_level:
            if len(current_block) > 0 and get_indent(current_block[-1]).startswith(indent_level + " "):
                current_block.append(lines[index])
                blocks.append("\n".join(current_block))
                if index < len(lines) - 1:
                    current_block = [lines[index + 1]]
                    index += 1
                else:
                    current_block = []
            else:
                blocks.append("\n".join(current_block))
                current_block = [lines[index]]
        else:
            current_block.append(lines[index])
        index += 1

    # Adds current block if it's nonempty.
    if len(current_block) > 0:
        blocks.append("\n".join(current_block))

    # Add final block after end_prompt if provided.
    if end_prompt is not None and index < len(lines):
        blocks.append("\n".join(lines[index:]))

    return blocks


def ignore_underscore(key):
    "Wraps a `key` (that maps an object to string) to lower case and remove underscores."

    def _inner(x):
        return key(x).lower().replace("_", "")

    return _inner


def sort_objects(objects, key=None):
    "Sort a list of `objects` following the rules of isort. `key` optionally maps an object to a str."
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

    key1 = ignore_underscore(key)
    return sorted(constants, key=key1) + sorted(classes, key=key1) + sorted(functions, key=key1)


def sort_objects_in_import(import_statement):
    """
    Return the same `import_statement` but with objects properly sorted.
    """
    # This inner function sort imports between [ ].
    def _replace(match):
        imports = match.groups()[0]
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


def sort_imports(file, check_only=True):
    """
    Sort `_import_structure` imports in `file`, `check_only` determines if we only check or overwrite.
    """
    with open(file, encoding="utf-8") as f:
        code = f.read()

    if "_import_structure" not in code:
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
        # We have two categories of import key: list or _import_structu[key].append/extend
        pattern = _re_direct_key if "_import_structure" in block_lines[0] else _re_indirect_key
        # Grab the keys, but there is a trap: some lines are empty or jsut comments.
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
