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
Welcome to tests_fetcher V2.
This util is designed to fetch tests to run on a PR so that only the tests impacted by the modifications are run, and
when too many models are being impacted, only run the tests of a subset of core models. It works like this.

Stage 1: Identify the modified files. This takes all the files from the branching point to the current commit (so
all modifications in a PR, not just the last commit) but excludes modifications that are on docstrings or comments
only.

Stage 2: Extract the tests to run. This is done by looking at the imports in each module and test file: if module A
imports module B, then changing module B impacts module A, so the tests using module A should be run. We thus get the
dependencies of each model and then recursively builds the 'reverse' map of dependencies to get all modules and tests
impacted by a given file. We then only keep the tests (and only the code models tests if there are too many modules).

Caveats:
  - This module only filters tests by files (not individual tests) so it's better to have tests for different things
    in different files.
  - This module assumes inits are just importing things, not really building objects, so it's better to structure
    them this way and move objects building in separate submodules.
"""

import argparse
import collections
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

from git import Repo


PATH_TO_REPO = Path(__file__).parent.parent.resolve()
PATH_TO_EXAMPLES = PATH_TO_REPO / "examples"
PATH_TO_TRANFORMERS = PATH_TO_REPO / "src/transformers"
PATH_TO_TESTS = PATH_TO_REPO / "tests"

# List here the models to always test.
IMPORTANT_MODELS = [
    "auto",
    # Most downloaded models
    "bert",
    "clip",
    "t5",
    "xlm-roberta",
    "gpt2",
    "bart",
    "mpnet",
    "gpt-j",
    "wav2vec2",
    "deberta-v2",
    "layoutlm",
    "opt",
    "longformer",
    "vit",
    # Pipeline-specific model (to be sure each pipeline has one model in this list)
    "tapas",
    "vilt",
    "clap",
    "detr",
    "owlvit",
    "dpt",
    "videomae",
]


@contextmanager
def checkout_commit(repo, commit_id):
    """
    Context manager that checks out a commit in the repo.
    """
    current_head = repo.head.commit if repo.head.is_detached else repo.head.ref

    try:
        repo.git.checkout(commit_id)
        yield

    finally:
        repo.git.checkout(current_head)


def clean_code(content):
    """
    Remove docstrings, empty line or comments from `content`.
    """
    # fmt: off
    # Remove docstrings by splitting on triple " then triple ':
    splits = content.split('\"\"\"')
    content = "".join(splits[::2])
    splits = content.split("\'\'\'")
    # fmt: on
    content = "".join(splits[::2])

    # Remove empty lines and comments
    lines_to_keep = []
    for line in content.split("\n"):
        # remove anything that is after a # sign.
        line = re.sub("#.*$", "", line)
        if len(line) == 0 or line.isspace():
            continue
        lines_to_keep.append(line)
    return "\n".join(lines_to_keep)


def keep_doc_examples_only(content):
    """
    Remove code, docstring that is not code example, empty line or comments from `content`.
    """
    # Keep doc examples only by splitting on triple "`"
    splits = content.split("```")
    # Add leading and trailing "```" so the navigation is easier when compared to the original input `content`
    content = "```" + "```".join(splits[1::2]) + "```"

    # Remove empty lines and comments
    lines_to_keep = []
    for line in content.split("\n"):
        # remove anything that is after a # sign.
        line = re.sub("#.*$", "", line)
        if len(line) == 0 or line.isspace():
            continue
        lines_to_keep.append(line)
    return "\n".join(lines_to_keep)


def get_all_tests():
    """
    Return a list of paths to all test folders and files under `tests`. All paths are rooted at `tests`.

    - folders under `tests`: `tokenization`, `pipelines`, etc. The folder `models` is excluded.
    - folders under `tests/models`: `bert`, `gpt2`, etc.
    - test files under `tests`: `test_modeling_common.py`, `test_tokenization_common.py`, etc.
    """

    # test folders/files directly under `tests` folder
    tests = os.listdir(PATH_TO_TESTS)
    tests = [f"tests/{f}" for f in tests if "__pycache__" not in f]
    tests = sorted([f for f in tests if (PATH_TO_REPO / f).is_dir() or f.startswith("tests/test_")])

    # model specific test folders
    model_test_folders = os.listdir(PATH_TO_TESTS / "models")
    model_test_folders = [f"tests/models/{f}" for f in model_test_folders if "__pycache__" not in f]
    model_test_folders = sorted([f for f in model_test_folders if (PATH_TO_REPO / f).is_dir()])

    tests.remove("tests/models")
    # Sagemaker tests are not meant to be run on the CI.
    if "tests/sagemaker" in tests:
        tests.remove("tests/sagemaker")
    tests = model_test_folders + tests

    return tests


def diff_is_docstring_only(repo, branching_point, filename):
    """
    Check if the diff is only in docstrings in a filename.
    """
    folder = Path(repo.working_dir)
    with checkout_commit(repo, branching_point):
        with open(folder / filename, "r", encoding="utf-8") as f:
            old_content = f.read()

    with open(folder / filename, "r", encoding="utf-8") as f:
        new_content = f.read()

    old_content_clean = clean_code(old_content)
    new_content_clean = clean_code(new_content)

    return old_content_clean == new_content_clean


def diff_contains_doc_examples(repo, branching_point, filename):
    """
    Check if the diff is only in code in a filename.
    """
    folder = Path(repo.working_dir)
    with checkout_commit(repo, branching_point):
        with open(folder / filename, "r", encoding="utf-8") as f:
            old_content = f.read()

    with open(folder / filename, "r", encoding="utf-8") as f:
        new_content = f.read()

    old_content_clean = keep_doc_examples_only(old_content)
    new_content_clean = keep_doc_examples_only(new_content)

    return old_content_clean != new_content_clean


def get_diff(repo, base_commit, commits):
    """
    Get's the diff between one or several commits and the head of the repository.
    """
    print("\n### DIFF ###\n")
    code_diff = []
    for commit in commits:
        for diff_obj in commit.diff(base_commit):
            # We always add new python files
            if diff_obj.change_type == "A" and diff_obj.b_path.endswith(".py"):
                code_diff.append(diff_obj.b_path)
            # We check that deleted python files won't break corresponding tests.
            elif diff_obj.change_type == "D" and diff_obj.a_path.endswith(".py"):
                code_diff.append(diff_obj.a_path)
            # Now for modified files
            elif diff_obj.change_type in ["M", "R"] and diff_obj.b_path.endswith(".py"):
                # In case of renames, we'll look at the tests using both the old and new name.
                if diff_obj.a_path != diff_obj.b_path:
                    code_diff.extend([diff_obj.a_path, diff_obj.b_path])
                else:
                    # Otherwise, we check modifications are in code and not docstrings.
                    if diff_is_docstring_only(repo, commit, diff_obj.b_path):
                        print(f"Ignoring diff in {diff_obj.b_path} as it only concerns docstrings or comments.")
                    else:
                        code_diff.append(diff_obj.a_path)

    return code_diff


def get_modified_python_files(diff_with_last_commit=False):
    """
    Return a list of python files that have been modified between:

    - the current head and the main branch if `diff_with_last_commit=False` (default)
    - the current head and its parent commit otherwise.
    """
    repo = Repo(PATH_TO_REPO)

    if not diff_with_last_commit:
        print(f"main is at {repo.refs.main.commit}")
        print(f"Current head is at {repo.head.commit}")

        branching_commits = repo.merge_base(repo.refs.main, repo.head)
        for commit in branching_commits:
            print(f"Branching commit: {commit}")
        return get_diff(repo, repo.head.commit, branching_commits)
    else:
        print(f"main is at {repo.head.commit}")
        parent_commits = repo.head.commit.parents
        for commit in parent_commits:
            print(f"Parent commit: {commit}")
        return get_diff(repo, repo.head.commit, parent_commits)


def get_diff_for_doctesting(repo, base_commit, commits):
    """
    Get's the diff between one or several commits and the head of the repository where some doc example(s) are changed.
    """
    print("\n### DIFF ###\n")
    code_diff = []
    for commit in commits:
        for diff_obj in commit.diff(base_commit):
            # We always add new python/md files
            if diff_obj.change_type in ["A"] and (diff_obj.b_path.endswith(".py") or diff_obj.b_path.endswith(".md")):
                code_diff.append(diff_obj.b_path)
            # Now for modified files
            elif (
                diff_obj.change_type in ["M", "R"]
                and diff_obj.b_path.endswith(".py")
                or diff_obj.b_path.endswith(".md")
            ):
                # In case of renames, we'll look at the tests using both the old and new name.
                if diff_obj.a_path != diff_obj.b_path:
                    code_diff.extend([diff_obj.a_path, diff_obj.b_path])
                else:
                    # Otherwise, we check modifications contain some doc example(s).
                    if diff_contains_doc_examples(repo, commit, diff_obj.b_path):
                        code_diff.append(diff_obj.a_path)
                    else:
                        print(f"Ignoring diff in {diff_obj.b_path} as it doesn't contain any doc example.")

    return code_diff


def get_doctest_files(diff_with_last_commit=False):
    """
    Return a list of python and mdx files where some doc example(s) in them have been modified between:

    - the current head and the main branch if `diff_with_last_commit=False` (default)
    - the current head and its parent commit otherwise.
    """
    repo = Repo(PATH_TO_REPO)

    test_files_to_run = []  # noqa
    if not diff_with_last_commit:
        print(f"main is at {repo.refs.main.commit}")
        print(f"Current head is at {repo.head.commit}")

        branching_commits = repo.merge_base(repo.refs.main, repo.head)
        for commit in branching_commits:
            print(f"Branching commit: {commit}")
        test_files_to_run = get_diff_for_doctesting(repo, repo.head.commit, branching_commits)
    else:
        print(f"main is at {repo.head.commit}")
        parent_commits = repo.head.commit.parents
        for commit in parent_commits:
            print(f"Parent commit: {commit}")
        test_files_to_run = get_diff_for_doctesting(repo, repo.head.commit, parent_commits)

    # This is the full list of doctest tests
    with open("utils/documentation_tests.txt") as fp:
        documentation_tests = set(fp.read().strip().split("\n"))
    # Not to run slow doctest tests
    with open("utils/slow_documentation_tests.txt") as fp:
        slow_documentation_tests = set(fp.read().strip().split("\n"))

    # So far we don't have 100% coverage for doctest. This line will be removed once we achieve 100%.
    test_files_to_run = [
        x for x in test_files_to_run if x in documentation_tests and x not in slow_documentation_tests
    ]
    # Make sure we did not end up with a test file that was removed
    test_files_to_run = [f for f in test_files_to_run if (PATH_TO_REPO / f).exists()]

    return test_files_to_run


# (:?^|\n) -> Non-catching group for the beginning of the doc or a new line.
# \s*from\s+(\.+\S+)\s+import\s+([^\n]+) -> Line only contains from .xxx import yyy and we catch .xxx and yyy
# (?=\n) -> Look-ahead to a new line. We can't just put \n here or using find_all on this re will only catch every
#           other import.
_re_single_line_relative_imports = re.compile(r"(?:^|\n)\s*from\s+(\.+\S+)\s+import\s+([^\n]+)(?=\n)")
# (:?^|\n) -> Non-catching group for the beginning of the doc or a new line.
# \s*from\s+(\.+\S+)\s+import\s+\(([^\)]+)\) -> Line continues with from .xxx import (yyy) and we catch .xxx and yyy
# yyy will take multiple lines otherwise there wouldn't be parenthesis.
_re_multi_line_relative_imports = re.compile(r"(?:^|\n)\s*from\s+(\.+\S+)\s+import\s+\(([^\)]+)\)")
# (:?^|\n) -> Non-catching group for the beginning of the doc or a new line.
# \s*from\s+transformers(\S*)\s+import\s+([^\n]+) -> Line only contains from transformers.xxx import yyy and we catch
#           .xxx and yyy
# (?=\n) -> Look-ahead to a new line. We can't just put \n here or using find_all on this re will only catch every
#           other import.
_re_single_line_direct_imports = re.compile(r"(?:^|\n)\s*from\s+transformers(\S*)\s+import\s+([^\n]+)(?=\n)")
# (:?^|\n) -> Non-catching group for the beginning of the doc or a new line.
# \s*from\s+transformers(\S*)\s+import\s+\(([^\)]+)\) -> Line continues with from transformers.xxx import (yyy) and we
# catch .xxx and yyy. yyy will take multiple lines otherwise there wouldn't be parenthesis.
_re_multi_line_direct_imports = re.compile(r"(?:^|\n)\s*from\s+transformers(\S*)\s+import\s+\(([^\)]+)\)")


def extract_imports(module_fname, cache=None):
    """
    Get the imports a given module makes. This takes a module filename and returns the list of module filenames
    imported in the module with the objects imported in that module filename.
    """
    if cache is not None and module_fname in cache:
        return cache[module_fname]

    with open(PATH_TO_REPO / module_fname, "r", encoding="utf-8") as f:
        content = f.read()

    # Filter out all docstrings to not get imports in code examples.
    # fmt: off
    splits = content.split('\"\"\"')
    # fmt: on
    content = "".join(splits[::2])

    module_parts = str(module_fname).split(os.path.sep)
    imported_modules = []

    # Let's start with relative imports
    relative_imports = _re_single_line_relative_imports.findall(content)
    relative_imports = [
        (mod, imp) for mod, imp in relative_imports if "# tests_ignore" not in imp and imp.strip() != "("
    ]
    multiline_relative_imports = _re_multi_line_relative_imports.findall(content)
    relative_imports += [(mod, imp) for mod, imp in multiline_relative_imports if "# tests_ignore" not in imp]

    for module, imports in relative_imports:
        level = 0
        while module.startswith("."):
            module = module[1:]
            level += 1

        if len(module) > 0:
            dep_parts = module_parts[: len(module_parts) - level] + module.split(".")
        else:
            dep_parts = module_parts[: len(module_parts) - level]
        imported_module = os.path.sep.join(dep_parts)
        imported_modules.append((imported_module, [imp.strip() for imp in imports.split(",")]))

    # Let's continue with direct imports
    direct_imports = _re_single_line_direct_imports.findall(content)
    direct_imports = [(mod, imp) for mod, imp in direct_imports if "# tests_ignore" not in imp and imp.strip() != "("]
    multiline_direct_imports = _re_multi_line_direct_imports.findall(content)
    direct_imports += [(mod, imp) for mod, imp in multiline_direct_imports if "# tests_ignore" not in imp]

    for module, imports in direct_imports:
        import_parts = module.split(".")[1:]  # ignore the first .
        dep_parts = ["src", "transformers"] + import_parts
        imported_module = os.path.sep.join(dep_parts)
        imported_modules.append((imported_module, [imp.strip() for imp in imports.split(",")]))

    result = []
    for module_file, imports in imported_modules:
        if (PATH_TO_REPO / f"{module_file}.py").is_file():
            module_file = f"{module_file}.py"
        elif (PATH_TO_REPO / module_file).is_dir() and (PATH_TO_REPO / module_file / "__init__.py").is_file():
            module_file = os.path.sep.join([module_file, "__init__.py"])
        imports = [imp for imp in imports if len(imp) > 0 and re.match("^[A-Za-z0-9_]*$", imp)]
        if len(imports) > 0:
            result.append((module_file, imports))

    if cache is not None:
        cache[module_fname] = result

    return result


def get_module_dependencies(module_fname, cache=None):
    """
    Get the dependencies of a module from the module filename as a list of module filenames. This will resolve any
    __init__ we pass: if we import from a submodule utils, the dependencies will be utils/foo.py and utils/bar.py (if
    the objects imported actually come from utils.foo and utils.bar) not utils/__init__.py.
    """
    dependencies = []
    imported_modules = extract_imports(module_fname, cache=cache)
    # The while loop is to recursively traverse all inits we may encounter.
    while len(imported_modules) > 0:
        new_modules = []
        for module, imports in imported_modules:
            # If we end up in an __init__ we are often not actually importing from this init (except in the case where
            # the object is fully defined in the __init__)
            if module.endswith("__init__.py"):
                # So we get the imports from that init then try to find where our objects come from.
                new_imported_modules = extract_imports(module, cache=cache)
                for new_module, new_imports in new_imported_modules:
                    if any(i in new_imports for i in imports):
                        if new_module not in dependencies:
                            new_modules.append((new_module, [i for i in new_imports if i in imports]))
                        imports = [i for i in imports if i not in new_imports]
                if len(imports) > 0:
                    # If there are any objects lefts, they may be a submodule
                    path_to_module = PATH_TO_REPO / module.replace("__init__.py", "")
                    dependencies.extend(
                        [
                            os.path.join(module.replace("__init__.py", ""), f"{i}.py")
                            for i in imports
                            if (path_to_module / f"{i}.py").is_file()
                        ]
                    )
                    imports = [i for i in imports if not (path_to_module / f"{i}.py").is_file()]
                    if len(imports) > 0:
                        # Then if there are still objects left, they are fully defined in the init, so we keep it as a
                        # dependency.
                        dependencies.append(module)
            else:
                dependencies.append(module)

        imported_modules = new_modules
    return dependencies


def create_reverse_dependency_tree():
    """
    Create a list of all edges (a, b) which mean that modifying a impacts b with a going over all module and test files.
    """
    cache = {}
    all_modules = list(PATH_TO_TRANFORMERS.glob("**/*.py")) + list(PATH_TO_TESTS.glob("**/*.py"))
    all_modules = [str(mod.relative_to(PATH_TO_REPO)) for mod in all_modules]
    edges = [(dep, mod) for mod in all_modules for dep in get_module_dependencies(mod, cache=cache)]

    return list(set(edges))


def get_tree_starting_at(module, edges):
    """
    Returns the tree starting at a given module following all edges in the following format: [module, [list of edges
    starting at module], [list of edges starting at the preceding level], ...]
    """
    vertices_seen = [module]
    new_edges = [edge for edge in edges if edge[0] == module and edge[1] != module and "__init__.py" not in edge[1]]
    tree = [module]
    while len(new_edges) > 0:
        tree.append(new_edges)
        final_vertices = list({edge[1] for edge in new_edges})
        vertices_seen.extend(final_vertices)
        new_edges = [
            edge
            for edge in edges
            if edge[0] in final_vertices and edge[1] not in vertices_seen and "__init__.py" not in edge[1]
        ]

    return tree


def print_tree_deps_of(module, all_edges=None):
    """
    Prints the tree of modules depending on a given module.
    """
    if all_edges is None:
        all_edges = create_reverse_dependency_tree()
    tree = get_tree_starting_at(module, all_edges)

    # The list of lines is a list of tuples (line_to_be_printed, module)
    # Keeping the modules lets us know where to insert each new lines in the list.
    lines = [(tree[0], tree[0])]
    for index in range(1, len(tree)):
        edges = tree[index]
        start_edges = {edge[0] for edge in edges}

        for start in start_edges:
            end_edges = {edge[1] for edge in edges if edge[0] == start}
            # We will insert all those edges just after the line showing start.
            pos = 0
            while lines[pos][1] != start:
                pos += 1
            lines = lines[: pos + 1] + [(" " * (2 * index) + end, end) for end in end_edges] + lines[pos + 1 :]

    for line in lines:
        # We don't print the refs that where just here to help build lines.
        print(line[0])


def init_test_examples_dependencies():
    """
    The test examples do not import from the examples (which are just scripts, not modules) so we need som extra
    care initializing the dependency map there.
    """
    test_example_deps = {}
    all_examples = []
    for framework in ["flax", "pytorch", "tensorflow"]:
        test_files = list((PATH_TO_EXAMPLES / framework).glob("test_*.py"))
        all_examples.extend(test_files)
        examples = [
            f for f in (PATH_TO_EXAMPLES / framework).glob("**/*.py") if f.parent != PATH_TO_EXAMPLES / framework
        ]
        all_examples.extend(examples)
        for test_file in test_files:
            with open(test_file, "r", encoding="utf-8") as f:
                content = f.read()
            test_example_deps[str(test_file.relative_to(PATH_TO_REPO))] = [
                str(e.relative_to(PATH_TO_REPO)) for e in examples if e.name in content
            ]
            test_example_deps[str(test_file.relative_to(PATH_TO_REPO))].append(
                str(test_file.relative_to(PATH_TO_REPO))
            )
    return test_example_deps, all_examples


def create_reverse_dependency_map():
    """
    Create the dependency map from module/test filename to the list of modules/tests that depend on it (even
    recursively).
    """
    cache = {}
    example_deps, examples = init_test_examples_dependencies()
    all_modules = list(PATH_TO_TRANFORMERS.glob("**/*.py")) + list(PATH_TO_TESTS.glob("**/*.py")) + examples
    all_modules = [str(mod.relative_to(PATH_TO_REPO)) for mod in all_modules]
    direct_deps = {m: get_module_dependencies(m, cache=cache) for m in all_modules}
    direct_deps.update(example_deps)

    # This recurses the dependencies
    something_changed = True
    while something_changed:
        something_changed = False
        for m in all_modules:
            for d in direct_deps[m]:
                if d.endswith("__init__.py"):
                    continue
                if d not in direct_deps:
                    raise ValueError(f"KeyError:{d}. From {m}")
                new_deps = set(direct_deps[d]) - set(direct_deps[m])
                if len(new_deps) > 0:
                    direct_deps[m].extend(list(new_deps))
                    something_changed = True

    # Finally we can build the reverse map.
    reverse_map = collections.defaultdict(list)
    for m in all_modules:
        for d in direct_deps[m]:
            reverse_map[d].append(m)

    for m in [f for f in all_modules if f.endswith("__init__.py")]:
        direct_deps = get_module_dependencies(m, cache=cache)
        deps = sum([reverse_map[d] for d in direct_deps if not d.endswith("__init__.py")], direct_deps)
        reverse_map[m] = list(set(deps) - {m})

    return reverse_map


def create_module_to_test_map(reverse_map=None, filter_models=False):
    """
    Extract the tests from the reverse_dependency_map and potentially filters the model tests.
    """
    if reverse_map is None:
        reverse_map = create_reverse_dependency_map()

    def is_test(fname):
        if fname.startswith("tests"):
            return True
        if fname.startswith("examples") and fname.split(os.path.sep)[-1].startswith("test"):
            return True
        return False

    test_map = {module: [f for f in deps if is_test(f)] for module, deps in reverse_map.items()}

    if not filter_models:
        return test_map

    num_model_tests = len(list(PATH_TO_TESTS.glob("models/*")))

    def has_many_models(tests):
        model_tests = {Path(t).parts[2] for t in tests if t.startswith("tests/models/")}
        return len(model_tests) > num_model_tests // 2

    def filter_tests(tests):
        return [t for t in tests if not t.startswith("tests/models/") or Path(t).parts[2] in IMPORTANT_MODELS]

    return {module: (filter_tests(tests) if has_many_models(tests) else tests) for module, tests in test_map.items()}


def check_imports_all_exist():
    """
    Isn't used per se by the test fetcher but might be used later as a quality check. Putting this here for now so the
    code is not lost.
    """
    cache = {}
    all_modules = list(PATH_TO_TRANFORMERS.glob("**/*.py")) + list(PATH_TO_TESTS.glob("**/*.py"))
    all_modules = [str(mod.relative_to(PATH_TO_REPO)) for mod in all_modules]
    direct_deps = {m: get_module_dependencies(m, cache=cache) for m in all_modules}

    for module, deps in direct_deps.items():
        for dep in deps:
            if not (PATH_TO_REPO / dep).is_file():
                print(f"{module} has dependency on {dep} which does not exist.")


def _print_list(l):
    return "\n".join([f"- {f}" for f in l])


def create_json_map(test_files_to_run, json_output_file):
    if json_output_file is None:
        return

    test_map = {}
    for test_file in test_files_to_run:
        # `test_file` is a path to a test folder/file, starting with `tests/`. For example,
        #   - `tests/models/bert/test_modeling_bert.py` or `tests/models/bert`
        #   - `tests/trainer/test_trainer.py` or `tests/trainer`
        #   - `tests/test_modeling_common.py`
        names = test_file.split(os.path.sep)
        if names[1] == "models":
            # take the part like `models/bert` for modeling tests
            key = os.path.sep.join(names[1:3])
        elif len(names) > 2 or not test_file.endswith(".py"):
            # test folders under `tests` or python files under them
            # take the part like tokenization, `pipeline`, etc. for other test categories
            key = os.path.sep.join(names[1:2])
        else:
            # common test files directly under `tests/`
            key = "common"

        if key not in test_map:
            test_map[key] = []
        test_map[key].append(test_file)

    # sort the keys & values
    keys = sorted(test_map.keys())
    test_map = {k: " ".join(sorted(test_map[k])) for k in keys}
    with open(json_output_file, "w", encoding="UTF-8") as fp:
        json.dump(test_map, fp, ensure_ascii=False)


def infer_tests_to_run(output_file, diff_with_last_commit=False, filter_models=True, json_output_file=None):
    modified_files = get_modified_python_files(diff_with_last_commit=diff_with_last_commit)
    print(f"\n### MODIFIED FILES ###\n{_print_list(modified_files)}")

    # Create the map that will give us all impacted modules.
    reverse_map = create_reverse_dependency_map()
    impacted_files = modified_files.copy()
    for f in modified_files:
        if f in reverse_map:
            impacted_files.extend(reverse_map[f])

    # Remove duplicates
    impacted_files = sorted(set(impacted_files))
    print(f"\n### IMPACTED FILES ###\n{_print_list(impacted_files)}")

    # Grab the corresponding test files:
    if "setup.py" in modified_files:
        test_files_to_run = ["tests"]
        repo_utils_launch = True
    else:
        # All modified tests need to be run.
        test_files_to_run = [
            f for f in modified_files if f.startswith("tests") and f.split(os.path.sep)[-1].startswith("test")
        ]
        # Then we grab the corresponding test files.
        test_map = create_module_to_test_map(reverse_map=reverse_map, filter_models=filter_models)
        for f in modified_files:
            if f in test_map:
                test_files_to_run.extend(test_map[f])
        test_files_to_run = sorted(set(test_files_to_run))
        # Remove repo utils tests
        test_files_to_run = [f for f in test_files_to_run if not f.split(os.path.sep)[1] == "repo_utils"]
        # Remove SageMaker tests
        test_files_to_run = [f for f in test_files_to_run if not f.split(os.path.sep)[1] == "sagemaker"]
        # Make sure we did not end up with a test file that was removed
        test_files_to_run = [f for f in test_files_to_run if (PATH_TO_REPO / f).exists()]

        repo_utils_launch = any(f.split(os.path.sep)[0] == "utils" for f in modified_files)

    if repo_utils_launch:
        repo_util_file = Path(output_file).parent / "test_repo_utils.txt"
        with open(repo_util_file, "w", encoding="utf-8") as f:
            f.write("tests/repo_utils")

    examples_tests_to_run = [f for f in test_files_to_run if f.startswith("examples")]
    test_files_to_run = [f for f in test_files_to_run if not f.startswith("examples")]
    print(f"\n### TEST TO RUN ###\n{_print_list(test_files_to_run)}")
    if len(test_files_to_run) > 0:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(" ".join(test_files_to_run))

        # Create a map that maps test categories to test files, i.e. `models/bert` -> [...test_modeling_bert.py, ...]

        # Get all test directories (and some common test files) under `tests` and `tests/models` if `test_files_to_run`
        # contains `tests` (i.e. when `setup.py` is changed).
        if "tests" in test_files_to_run:
            test_files_to_run = get_all_tests()

        create_json_map(test_files_to_run, json_output_file)

    print(f"\n### EXAMPLES TEST TO RUN ###\n{_print_list(examples_tests_to_run)}")
    if len(examples_tests_to_run) > 0:
        example_file = Path(output_file).parent / "examples_test_list.txt"
        with open(example_file, "w", encoding="utf-8") as f:
            f.write(" ".join(examples_tests_to_run))

    doctest_list = get_doctest_files()

    print(f"\n### DOCTEST TO RUN ###\n{_print_list(doctest_list)}")
    if len(doctest_list) > 0:
        doctest_file = Path(output_file).parent / "doctest_list.txt"
        with open(doctest_file, "w", encoding="utf-8") as f:
            f.write(" ".join(doctest_list))


def filter_tests(output_file, filters):
    """
    Reads the content of the output file and filters out all the tests in a list of given folders.

    Args:
        output_file (`str` or `os.PathLike`): The path to the output file of the tests fetcher.
        filters (`List[str]`): A list of folders to filter.
    """
    if not os.path.isfile(output_file):
        print("No test file found.")
        return
    with open(output_file, "r", encoding="utf-8") as f:
        test_files = f.read().split(" ")

    if len(test_files) == 0 or test_files == [""]:
        print("No tests to filter.")
        return

    if test_files == ["tests"]:
        test_files = [os.path.join("tests", f) for f in os.listdir("tests") if f not in ["__init__.py"] + filters]
    else:
        test_files = [f for f in test_files if f.split(os.path.sep)[1] not in filters]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(" ".join(test_files))


def parse_commit_message(commit_message):
    """
    Parses the commit message to detect if a command is there to skip, force all or part of the CI.

    Returns a dictionary of strings to bools with keys skip, test_all_models and test_all.
    """
    if commit_message is None:
        return {"skip": False, "no_filter": False, "test_all": False}

    command_search = re.search(r"\[([^\]]*)\]", commit_message)
    if command_search is not None:
        command = command_search.groups()[0]
        command = command.lower().replace("-", " ").replace("_", " ")
        skip = command in ["ci skip", "skip ci", "circleci skip", "skip circleci"]
        no_filter = set(command.split(" ")) == {"no", "filter"}
        test_all = set(command.split(" ")) == {"test", "all"}
        return {"skip": skip, "no_filter": no_filter, "test_all": test_all}
    else:
        return {"skip": False, "no_filter": False, "test_all": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file", type=str, default="test_list.txt", help="Where to store the list of tests to run"
    )
    parser.add_argument(
        "--json_output_file",
        type=str,
        default="test_map.json",
        help="Where to store the tests to run in a dictionary format mapping test categories to test files",
    )
    parser.add_argument(
        "--diff_with_last_commit",
        action="store_true",
        help="To fetch the tests between the current commit and the last commit",
    )
    parser.add_argument(
        "--filter_tests",
        action="store_true",
        help="Will filter the pipeline/repo utils tests outside of the generated list of tests.",
    )
    parser.add_argument(
        "--print_dependencies_of",
        type=str,
        help="Will only print the tree of modules depending on the file passed.",
        default=None,
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        help="The commit message (which could contain a command to force all tests or skip the CI).",
        default=None,
    )
    args = parser.parse_args()
    if args.print_dependencies_of is not None:
        print_tree_deps_of(args.print_dependencies_of)
    elif args.filter_tests:
        filter_tests(args.output_file, ["pipelines", "repo_utils"])
    else:
        repo = Repo(PATH_TO_REPO)
        commit_message = repo.head.commit.message
        commit_flags = parse_commit_message(commit_message)
        if commit_flags["skip"]:
            print("Force-skipping the CI")
            quit()
        if commit_flags["no_filter"]:
            print("Running all tests fetched without filtering.")
        if commit_flags["test_all"]:
            print("Force-launching all tests")

        diff_with_last_commit = args.diff_with_last_commit
        if not diff_with_last_commit and not repo.head.is_detached and repo.head.ref == repo.refs.main:
            print("main branch detected, fetching tests against last commit.")
            diff_with_last_commit = True

        if not commit_flags["test_all"]:
            try:
                infer_tests_to_run(
                    args.output_file,
                    diff_with_last_commit=diff_with_last_commit,
                    json_output_file=args.json_output_file,
                    filter_models=not commit_flags["no_filter"],
                )
                filter_tests(args.output_file, ["repo_utils"])
            except Exception as e:
                print(f"\nError when trying to grab the relevant tests: {e}\n\nRunning all tests.")
                commit_flags["test_all"] = True

        if commit_flags["test_all"]:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write("tests")
            example_file = Path(args.output_file).parent / "examples_test_list.txt"
            with open(example_file, "w", encoding="utf-8") as f:
                f.write("all")

            test_files_to_run = get_all_tests()
            create_json_map(test_files_to_run, args.json_output_file)
