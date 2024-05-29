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
This script is responsible for making sure the dummies in utils/dummies_xxx.py are up to date with the main init.

Why dummies? This is to make sure that a user can always import all objects from `transformers`, even if they don't
have the necessary extra libs installed. Those objects will then raise helpful error message whenever the user tries
to access one of their methods.

Usage (from the root of the repo):

Check that the dummy files are up to date (used in `make repo-consistency`):

```bash
python utils/check_dummies.py
```

Update the dummy files if needed (used in `make fix-copies`):

```bash
python utils/check_dummies.py --fix_and_overwrite
```
"""

import argparse
import os
import re
from typing import Dict, List, Optional


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_dummies.py
PATH_TO_TRANSFORMERS = "src/transformers"

# Matches is_xxx_available()
_re_backend = re.compile(r"is\_([a-z_]*)_available()")
# Matches from xxx import bla
_re_single_line_import = re.compile(r"\s+from\s+\S*\s+import\s+([^\(\s].*)\n")
# Matches if not is_xxx_available()
_re_test_backend = re.compile(r"^\s+if\s+not\s+\(?is\_[a-z_]*\_available\(\)")


# Template for the dummy objects.
DUMMY_CONSTANT = """
{0} = None
"""


DUMMY_CLASS = """
class {0}(metaclass=DummyObject):
    _backends = {1}

    def __init__(self, *args, **kwargs):
        requires_backends(self, {1})
"""


DUMMY_FUNCTION = """
def {0}(*args, **kwargs):
    requires_backends({0}, {1})
"""


def find_backend(line: str) -> Optional[str]:
    """
    Find one (or multiple) backend in a code line of the init.

    Args:
        line (`str`): A code line in an init file.

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


def read_init() -> Dict[str, List[str]]:
    """
    Read the init and extract backend-specific objects.

    Returns:
        Dict[str, List[str]]: A dictionary mapping backend name to the list of object names requiring that backend.
    """
    with open(os.path.join(PATH_TO_TRANSFORMERS, "__init__.py"), "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()

    # Get to the point we do the actual imports for type checking
    line_index = 0
    while not lines[line_index].startswith("if TYPE_CHECKING"):
        line_index += 1

    backend_specific_objects = {}
    # Go through the end of the file
    while line_index < len(lines):
        # If the line is an if is_backend_available, we grab all objects associated.
        backend = find_backend(lines[line_index])
        if backend is not None:
            while not lines[line_index].startswith("    else:"):
                line_index += 1
            line_index += 1

            objects = []
            # Until we unindent, add backend objects to the list
            while len(lines[line_index]) <= 1 or lines[line_index].startswith(" " * 8):
                line = lines[line_index]
                single_line_import_search = _re_single_line_import.search(line)
                if single_line_import_search is not None:
                    # Single-line imports
                    objects.extend(single_line_import_search.groups()[0].split(", "))
                elif line.startswith(" " * 12):
                    # Multiple-line imports (with 3 indent level)
                    objects.append(line[12:-2])
                line_index += 1

            backend_specific_objects[backend] = objects
        else:
            line_index += 1

    return backend_specific_objects


def create_dummy_object(name: str, backend_name: str) -> str:
    """
    Create the code for a dummy object.

    Args:
        name (`str`): The name of the object.
        backend_name (`str`): The name of the backend required for that object.

    Returns:
        `str`: The code of the dummy object.
    """
    if name.isupper():
        return DUMMY_CONSTANT.format(name)
    elif name.islower():
        return DUMMY_FUNCTION.format(name, backend_name)
    else:
        return DUMMY_CLASS.format(name, backend_name)


def create_dummy_files(backend_specific_objects: Optional[Dict[str, List[str]]] = None) -> Dict[str, str]:
    """
    Create the content of the dummy files.

    Args:
        backend_specific_objects (`Dict[str, List[str]]`, *optional*):
            The mapping backend name to list of backend-specific objects. If not passed, will be obtained by calling
            `read_init()`.

    Returns:
        `Dict[str, str]`: A dictionary mapping backend name to code of the corresponding backend file.
    """
    if backend_specific_objects is None:
        backend_specific_objects = read_init()

    dummy_files = {}

    for backend, objects in backend_specific_objects.items():
        backend_name = "[" + ", ".join(f'"{b}"' for b in backend.split("_and_")) + "]"
        dummy_file = "# This file is autogenerated by the command `make fix-copies`, do not edit.\n"
        dummy_file += "from ..utils import DummyObject, requires_backends\n\n"
        dummy_file += "\n".join([create_dummy_object(o, backend_name) for o in objects])
        dummy_files[backend] = dummy_file

    return dummy_files


def check_dummies(overwrite: bool = False):
    """
    Check if the dummy files are up to date and maybe `overwrite` with the right content.

    Args:
        overwrite (`bool`, *optional*, default to `False`):
            Whether or not to overwrite the content of the dummy files. Will raise an error if they are not up to date
            when `overwrite=False`.
    """
    dummy_files = create_dummy_files()
    # For special correspondence backend name to shortcut as used in utils/dummy_xxx_objects.py
    short_names = {"torch": "pt"}

    # Locate actual dummy modules and read their content.
    path = os.path.join(PATH_TO_TRANSFORMERS, "utils")
    dummy_file_paths = {
        backend: os.path.join(path, f"dummy_{short_names.get(backend, backend)}_objects.py")
        for backend in dummy_files.keys()
    }

    actual_dummies = {}
    for backend, file_path in dummy_file_paths.items():
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8", newline="\n") as f:
                actual_dummies[backend] = f.read()
        else:
            actual_dummies[backend] = ""

    # Compare actual with what they should be.
    for backend in dummy_files.keys():
        if dummy_files[backend] != actual_dummies[backend]:
            if overwrite:
                print(
                    f"Updating transformers.utils.dummy_{short_names.get(backend, backend)}_objects.py as the main "
                    "__init__ has new objects."
                )
                with open(dummy_file_paths[backend], "w", encoding="utf-8", newline="\n") as f:
                    f.write(dummy_files[backend])
            else:
                raise ValueError(
                    "The main __init__ has objects that are not present in "
                    f"transformers.utils.dummy_{short_names.get(backend, backend)}_objects.py. Run `make fix-copies` "
                    "to fix this."
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    check_dummies(args.fix_and_overwrite)
