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
Utility that checks the custom inits of Transformers are well-defined: Transformers uses init files that delay the
import of an object to when it's actually needed. This is to avoid the main init importing all models, which would
make the line `import transformers` very slow when the user has all optional dependencies installed. The inits with
delayed imports have two halves: one defining a dictionary `_import_structure` which maps modules to the name of the
objects in each module, and one in `TYPE_CHECKING` which looks like a normal init for type-checkers. The goal of this
script is to check the objects defined in both halves are the same.

This also checks the main init properly references all submodules, even if it doesn't import anything from them: every
submodule should be defined as a key of `_import_structure`, with an empty list as value potentially, or the submodule
won't be importable.

Use from the root of the repo with:

```bash
python utils/check_inits.py
```

for a check that will error in case of inconsistencies (used by `make repo-consistency`).

There is no auto-fix possible here sadly :-(
"""

import collections
import os
import re
from pathlib import Path
from typing import Optional


# Path is set with the intent you should run this script from the root of the repo.
PATH_TO_TRANSFORMERS = "src/transformers"


# Matches is_xxx_available()
_re_backend = re.compile(r"is\_([a-z_]*)_available()")
# Catches a one-line _import_struct = {xxx}
_re_one_line_import_struct = re.compile(r"^_import_structure\s+=\s+\{([^\}]+)\}")
# Catches a line with a key-values pattern: "bla": ["foo", "bar"]
_re_import_struct_key_value = re.compile(r'\s+"\S*":\s+\[([^\]]*)\]')
# Catches a line if not is_foo_available
_re_test_backend = re.compile(r"^\s*if\s+not\s+is\_[a-z_]*\_available\(\)")
# Catches a line _import_struct["bla"].append("foo")
_re_import_struct_add_one = re.compile(r'^\s*_import_structure\["\S*"\]\.append\("(\S*)"\)')
# Catches a line _import_struct["bla"].extend(["foo", "bar"]) or _import_struct["bla"] = ["foo", "bar"]
_re_import_struct_add_many = re.compile(r"^\s*_import_structure\[\S*\](?:\.extend\(|\s*=\s+)\[([^\]]*)\]")
# Catches a line with an object between quotes and a comma:     "MyModel",
_re_quote_object = re.compile(r'^\s+"([^"]+)",')
# Catches a line with objects between brackets only:    ["foo", "bar"],
_re_between_brackets = re.compile(r"^\s+\[([^\]]+)\]")
# Catches a line with from foo import bar, bla, boo
_re_import = re.compile(r"\s+from\s+\S*\s+import\s+([^\(\s].*)\n")
# Catches a line with try:
_re_try = re.compile(r"^\s*try:")
# Catches a line with else:
_re_else = re.compile(r"^\s*else:")


def find_backend(line: str) -> Optional[str]:
    """
    Find one (or multiple) backend in a code line of the init.

    Args:
        line (`str`): A code line of the main init.

    Returns:
        Optional[`str`]: If one (or several) backend is found, returns it. In the case of multiple backends (the line
        contains `if is_xxx_available() and `is_yyy_available()`) returns all backends joined on `_and_` (so
        `xxx_and_yyy` for instance).
    """
    if _re_test_backend.search(line) is None:
        return None
    backends = [b[0] for b in _re_backend.findall(line)]
    backends.sort()
    return "_and_".join(backends)


def parse_init(init_file) -> Optional[tuple[dict[str, list[str]], dict[str, list[str]]]]:
    """
    Read an init_file and parse (per backend) the `_import_structure` objects defined and the `TYPE_CHECKING` objects
    defined.

    Args:
        init_file (`str`): Path to the init file to inspect.

    Returns:
        `Optional[Tuple[Dict[str, List[str]], Dict[str, List[str]]]]`: A tuple of two dictionaries mapping backends to list of
        imported objects, one for the `_import_structure` part of the init and one for the `TYPE_CHECKING` part of the
        init. Returns `None` if the init is not a custom init.
    """
    with open(init_file, "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()

    # Get the to `_import_structure` definition.
    line_index = 0
    while line_index < len(lines) and not lines[line_index].startswith("_import_structure = {"):
        line_index += 1

    # If this is a traditional init, just return.
    if line_index >= len(lines):
        return None

    # First grab the objects without a specific backend in _import_structure
    objects = []
    while not lines[line_index].startswith("if TYPE_CHECKING") and find_backend(lines[line_index]) is None:
        line = lines[line_index]
        # If we have everything on a single line, let's deal with it.
        if _re_one_line_import_struct.search(line):
            content = _re_one_line_import_struct.search(line).groups()[0]
            imports = re.findall(r"\[([^\]]+)\]", content)
            for imp in imports:
                objects.extend([obj[1:-1] for obj in imp.split(", ")])
            line_index += 1
            continue
        single_line_import_search = _re_import_struct_key_value.search(line)
        if single_line_import_search is not None:
            imports = [obj[1:-1] for obj in single_line_import_search.groups()[0].split(", ") if len(obj) > 0]
            objects.extend(imports)
        elif line.startswith(" " * 8 + '"'):
            objects.append(line[9:-3])
        line_index += 1

    # Those are stored with the key "none".
    import_dict_objects = {"none": objects}

    # Let's continue with backend-specific objects in _import_structure
    while not lines[line_index].startswith("if TYPE_CHECKING"):
        # If the line is an if not is_backend_available, we grab all objects associated.
        backend = find_backend(lines[line_index])
        # Check if the backend declaration is inside a try block:
        if _re_try.search(lines[line_index - 1]) is None:
            backend = None

        if backend is not None:
            line_index += 1

            # Scroll until we hit the else block of try-except-else
            while _re_else.search(lines[line_index]) is None:
                line_index += 1

            line_index += 1

            objects = []
            # Until we unindent, add backend objects to the list
            while len(lines[line_index]) <= 1 or lines[line_index].startswith(" " * 4):
                line = lines[line_index]
                if _re_import_struct_add_one.search(line) is not None:
                    objects.append(_re_import_struct_add_one.search(line).groups()[0])
                elif _re_import_struct_add_many.search(line) is not None:
                    imports = _re_import_struct_add_many.search(line).groups()[0].split(", ")
                    imports = [obj[1:-1] for obj in imports if len(obj) > 0]
                    objects.extend(imports)
                elif _re_between_brackets.search(line) is not None:
                    imports = _re_between_brackets.search(line).groups()[0].split(", ")
                    imports = [obj[1:-1] for obj in imports if len(obj) > 0]
                    objects.extend(imports)
                elif _re_quote_object.search(line) is not None:
                    objects.append(_re_quote_object.search(line).groups()[0])
                elif line.startswith(" " * 8 + '"'):
                    objects.append(line[9:-3])
                elif line.startswith(" " * 12 + '"'):
                    objects.append(line[13:-3])
                line_index += 1

            import_dict_objects[backend] = objects
        else:
            line_index += 1

    # At this stage we are in the TYPE_CHECKING part, first grab the objects without a specific backend
    objects = []
    while (
        line_index < len(lines)
        and find_backend(lines[line_index]) is None
        and not lines[line_index].startswith("else")
    ):
        line = lines[line_index]
        single_line_import_search = _re_import.search(line)
        if single_line_import_search is not None:
            objects.extend(single_line_import_search.groups()[0].split(", "))
        elif line.startswith(" " * 8):
            objects.append(line[8:-2])
        line_index += 1

    type_hint_objects = {"none": objects}

    # Let's continue with backend-specific objects
    while line_index < len(lines):
        # If the line is an if is_backend_available, we grab all objects associated.
        backend = find_backend(lines[line_index])
        # Check if the backend declaration is inside a try block:
        if _re_try.search(lines[line_index - 1]) is None:
            backend = None

        if backend is not None:
            line_index += 1

            # Scroll until we hit the else block of try-except-else
            while _re_else.search(lines[line_index]) is None:
                line_index += 1

            line_index += 1

            objects = []
            # Until we unindent, add backend objects to the list
            while len(lines[line_index]) <= 1 or lines[line_index].startswith(" " * 8):
                line = lines[line_index]
                single_line_import_search = _re_import.search(line)
                if single_line_import_search is not None:
                    objects.extend(single_line_import_search.groups()[0].split(", "))
                elif line.startswith(" " * 12):
                    objects.append(line[12:-2])
                line_index += 1

            type_hint_objects[backend] = objects
        else:
            line_index += 1

    return import_dict_objects, type_hint_objects


def analyze_results(import_dict_objects: dict[str, list[str]], type_hint_objects: dict[str, list[str]]) -> list[str]:
    """
    Analyze the differences between _import_structure objects and TYPE_CHECKING objects found in an init.

    Args:
        import_dict_objects (`Dict[str, List[str]]`):
            A dictionary mapping backend names (`"none"` for the objects independent of any specific backend) to
            list of imported objects.
        type_hint_objects (`Dict[str, List[str]]`):
            A dictionary mapping backend names (`"none"` for the objects independent of any specific backend) to
            list of imported objects.

    Returns:
        `List[str]`: The list of errors corresponding to mismatches.
    """

    def find_duplicates(seq):
        return [k for k, v in collections.Counter(seq).items() if v > 1]

    # If one backend is missing from the other part of the init, error early.
    if list(import_dict_objects.keys()) != list(type_hint_objects.keys()):
        return ["Both sides of the init do not have the same backends!"]

    errors = []
    # Find all errors.
    for key in import_dict_objects:
        # Duplicate imports in any half.
        duplicate_imports = find_duplicates(import_dict_objects[key])
        if duplicate_imports:
            errors.append(f"Duplicate _import_structure definitions for: {duplicate_imports}")
        duplicate_type_hints = find_duplicates(type_hint_objects[key])
        if duplicate_type_hints:
            errors.append(f"Duplicate TYPE_CHECKING objects for: {duplicate_type_hints}")

        # Missing imports in either part of the init.
        if sorted(set(import_dict_objects[key])) != sorted(set(type_hint_objects[key])):
            name = "base imports" if key == "none" else f"{key} backend"
            errors.append(f"Differences for {name}:")
            for a in type_hint_objects[key]:
                if a not in import_dict_objects[key]:
                    errors.append(f"  {a} in TYPE_HINT but not in _import_structure.")
            for a in import_dict_objects[key]:
                if a not in type_hint_objects[key]:
                    errors.append(f"  {a} in _import_structure but not in TYPE_HINT.")
    return errors


def get_transformers_submodules() -> list[str]:
    """
    Returns the list of Transformers submodules.
    """
    submodules = []
    for path, directories, files in os.walk(PATH_TO_TRANSFORMERS):
        for folder in directories:
            # Ignore private modules
            if folder.startswith("_"):
                directories.remove(folder)
                continue
            # Ignore leftovers from branches (empty folders apart from pycache)
            if len(list((Path(path) / folder).glob("*.py"))) == 0:
                continue
            short_path = str((Path(path) / folder).relative_to(PATH_TO_TRANSFORMERS))
            submodule = short_path.replace(os.path.sep, ".")
            submodules.append(submodule)
        for fname in files:
            if fname == "__init__.py":
                continue
            short_path = str((Path(path) / fname).relative_to(PATH_TO_TRANSFORMERS))
            submodule = short_path.replace(".py", "").replace(os.path.sep, ".")
            if len(submodule.split(".")) == 1:
                submodules.append(submodule)
    return submodules


IGNORE_SUBMODULES = [
    "convert_pytorch_checkpoint_to_tf2",
    "modeling_flax_pytorch_utils",
    "models.esm.openfold_utils",
    "modeling_attn_mask_utils",
    "safetensors_conversion",
    "modeling_gguf_pytorch_utils",
    "kernels.falcon_mamba",
    "kernels",
]


def check_submodules():
    """
    Check all submodules of Transformers are properly registered in the main init. Error otherwise.
    """
    # This is to make sure the transformers module imported is the one in the repo.
    from transformers.utils import direct_transformers_import

    transformers = direct_transformers_import(PATH_TO_TRANSFORMERS)

    import_structure_keys = set(transformers._import_structure.keys())
    # This contains all the base keys of the _import_structure object defined in the init, but if the user is missing
    # some optional dependencies, they may not have all of them. Thus we read the init to read all additions and
    # (potentiall re-) add them.
    with open(os.path.join(PATH_TO_TRANSFORMERS, "__init__.py"), "r") as f:
        init_content = f.read()
    import_structure_keys.update(set(re.findall(r"import_structure\[\"([^\"]*)\"\]", init_content)))

    module_not_registered = [
        module
        for module in get_transformers_submodules()
        if module not in IGNORE_SUBMODULES and module not in import_structure_keys
    ]

    if len(module_not_registered) > 0:
        list_of_modules = "\n".join(f"- {module}" for module in module_not_registered)
        raise ValueError(
            "The following submodules are not properly registered in the main init of Transformers:\n"
            f"{list_of_modules}\n"
            "Make sure they appear somewhere in the keys of `_import_structure` with an empty list as value."
        )


if __name__ == "__main__":
    # This entire files needs an overhaul
    pass
