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

import collections
import importlib.util
import os
import re
from pathlib import Path


PATH_TO_TRANSFORMERS = "src/transformers"


# Matches is_xxx_available()
_re_backend = re.compile(r"is\_([a-z_]*)_available()")
# Catches a line with a key-values pattern: "bla": ["foo", "bar"]
_re_import_struct_key_value = re.compile(r'\s+"\S*":\s+\[([^\]]*)\]')
# Catches a line if is_foo_available
_re_test_backend = re.compile(r"^\s*if\s+is\_[a-z_]*\_available\(\)")
# Catches a line _import_struct["bla"].append("foo")
_re_import_struct_add_one = re.compile(r'^\s*_import_structure\["\S*"\]\.append\("(\S*)"\)')
# Catches a line _import_struct["bla"].extend(["foo", "bar"]) or _import_struct["bla"] = ["foo", "bar"]
_re_import_struct_add_many = re.compile(r"^\s*_import_structure\[\S*\](?:\.extend\(|\s*=\s+)\[([^\]]*)\]")
# Catches a line with an object between quotes and a comma:     "MyModel",
_re_quote_object = re.compile('^\s+"([^"]+)",')
# Catches a line with objects between brackets only:    ["foo", "bar"],
_re_between_brackets = re.compile("^\s+\[([^\]]+)\]")
# Catches a line with from foo import bar, bla, boo
_re_import = re.compile(r"\s+from\s+\S*\s+import\s+([^\(\s].*)\n")


def find_backend(line):
    """Find one (or multiple) backend in a code line of the init."""
    if _re_test_backend.search(line) is None:
        return None
    backends = [b[0] for b in _re_backend.findall(line)]
    backends.sort()
    return "_and_".join(backends)


def parse_init(init_file):
    """
    Read an init_file and parse (per backend) the _import_structure objects defined and the TYPE_CHECKING objects
    defined
    """
    with open(init_file, "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()

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
        single_line_import_search = _re_import_struct_key_value.search(line)
        if single_line_import_search is not None:
            imports = [obj[1:-1] for obj in single_line_import_search.groups()[0].split(", ") if len(obj) > 0]
            objects.extend(imports)
        elif line.startswith(" " * 8 + '"'):
            objects.append(line[9:-3])
        line_index += 1

    import_dict_objects = {"none": objects}
    # Let's continue with backend-specific objects in _import_structure
    while not lines[line_index].startswith("if TYPE_CHECKING"):
        # If the line is an if is_backend_available, we grab all objects associated.
        backend = find_backend(lines[line_index])
        if backend is not None:
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
        # If the line is an if is_backemd_available, we grab all objects associated.
        backend = find_backend(lines[line_index])
        if backend is not None:
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


def analyze_results(import_dict_objects, type_hint_objects):
    """
    Analyze the differences between _import_structure objects and TYPE_CHECKING objects found in an init.
    """

    def find_duplicates(seq):
        return [k for k, v in collections.Counter(seq).items() if v > 1]

    if list(import_dict_objects.keys()) != list(type_hint_objects.keys()):
        return ["Both sides of the init do not have the same backends!"]

    errors = []
    for key in import_dict_objects.keys():
        duplicate_imports = find_duplicates(import_dict_objects[key])
        if duplicate_imports:
            errors.append(f"Duplicate _import_structure definitions for: {duplicate_imports}")
        duplicate_type_hints = find_duplicates(type_hint_objects[key])
        if duplicate_type_hints:
            errors.append(f"Duplicate TYPE_CHECKING objects for: {duplicate_type_hints}")

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


def check_all_inits():
    """
    Check all inits in the transformers repo and raise an error if at least one does not define the same objects in
    both halves.
    """
    failures = []
    for root, _, files in os.walk(PATH_TO_TRANSFORMERS):
        if "__init__.py" in files:
            fname = os.path.join(root, "__init__.py")
            objects = parse_init(fname)
            if objects is not None:
                errors = analyze_results(*objects)
                if len(errors) > 0:
                    errors[0] = f"Problem in {fname}, both halves do not define the same objects.\n{errors[0]}"
                    failures.append("\n".join(errors))
    if len(failures) > 0:
        raise ValueError("\n\n".join(failures))


def get_transformers_submodules():
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
            submodule = short_path.replace(os.path.sep, ".").replace(".py", "")
            if len(submodule.split(".")) == 1:
                submodules.append(submodule)
    return submodules


IGNORE_SUBMODULES = [
    "convert_pytorch_checkpoint_to_tf2",
    "modeling_flax_pytorch_utils",
]


def check_submodules():
    # This is to make sure the transformers module imported is the one in the repo.
    spec = importlib.util.spec_from_file_location(
        "transformers",
        os.path.join(PATH_TO_TRANSFORMERS, "__init__.py"),
        submodule_search_locations=[PATH_TO_TRANSFORMERS],
    )
    transformers = spec.loader.load_module()

    module_not_registered = [
        module
        for module in get_transformers_submodules()
        if module not in IGNORE_SUBMODULES and module not in transformers._import_structure.keys()
    ]
    if len(module_not_registered) > 0:
        list_of_modules = "\n".join(f"- {module}" for module in module_not_registered)
        raise ValueError(
            "The following submodules are not properly registed in the main init of Transformers:\n"
            f"{list_of_modules}\n"
            "Make sure they appear somewhere in the keys of `_import_structure` with an empty list as value."
        )


if __name__ == "__main__":
    check_all_inits()
    check_submodules()
