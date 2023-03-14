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
import collections
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

from git import Repo


# This script is intended to be run from the root of the repo but you can adapt this constant if you need to.
PATH_TO_TRANFORMERS = "."

# A temporary way to trigger all pipeline tests contained in model test files after PR #21516
all_model_test_files = [str(x) for x in Path("tests/models/").glob("**/**/test_modeling_*.py")]

all_pipeline_test_files = [str(x) for x in Path("tests/pipelines/").glob("**/test_pipelines_*.py")]


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


def get_all_tests():
    """
    Return a list of paths to all test folders and files under `tests`. All paths are rooted at `tests`.

    - folders under `tests`: `tokenization`, `pipelines`, etc. The folder `models` is excluded.
    - folders under `tests/models`: `bert`, `gpt2`, etc.
    - test files under `tests`: `test_modeling_common.py`, `test_tokenization_common.py`, etc.
    """
    test_root_dir = os.path.join(PATH_TO_TRANFORMERS, "tests")

    # test folders/files directly under `tests` folder
    tests = os.listdir(test_root_dir)
    tests = sorted(filter(lambda x: os.path.isdir(x) or x.startswith("tests/test_"), [f"tests/{x}" for x in tests]))

    # model specific test folders
    model_tests_folders = os.listdir(os.path.join(test_root_dir, "models"))
    model_test_folders = sorted(filter(os.path.isdir, [f"tests/models/{x}" for x in model_tests_folders]))

    tests.remove("tests/models")
    tests = model_test_folders + tests

    return tests


def diff_is_docstring_only(repo, branching_point, filename):
    """
    Check if the diff is only in docstrings in a filename.
    """
    with checkout_commit(repo, branching_point):
        with open(filename, "r", encoding="utf-8") as f:
            old_content = f.read()

    with open(filename, "r", encoding="utf-8") as f:
        new_content = f.read()

    old_content_clean = clean_code(old_content)
    new_content_clean = clean_code(new_content)

    return old_content_clean == new_content_clean


def get_modified_python_files(diff_with_last_commit=False):
    """
    Return a list of python files that have been modified between:

    - the current head and the main branch if `diff_with_last_commit=False` (default)
    - the current head and its parent commit otherwise.
    """
    repo = Repo(PATH_TO_TRANFORMERS)

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


def get_module_dependencies(module_fname):
    """
    Get the dependencies of a module.
    """
    with open(os.path.join(PATH_TO_TRANFORMERS, module_fname), "r", encoding="utf-8") as f:
        content = f.read()

    module_parts = module_fname.split(os.path.sep)
    imported_modules = []

    # Let's start with relative imports
    relative_imports = re.findall(r"from\s+(\.+\S+)\s+import\s+([^\n]+)\n", content)
    relative_imports = [mod for mod, imp in relative_imports if "# tests_ignore" not in imp]
    for imp in relative_imports:
        level = 0
        while imp.startswith("."):
            imp = imp[1:]
            level += 1

        if len(imp) > 0:
            dep_parts = module_parts[: len(module_parts) - level] + imp.split(".")
        else:
            dep_parts = module_parts[: len(module_parts) - level] + ["__init__.py"]
        imported_module = os.path.sep.join(dep_parts)
        # We ignore the main init import as it's only for the __version__ that it's done
        # and it would add everything as a dependency.
        if not imported_module.endswith("transformers/__init__.py"):
            imported_modules.append(imported_module)

    # Let's continue with direct imports
    # The import from the transformers module are ignored for the same reason we ignored the
    # main init before.
    direct_imports = re.findall(r"from\s+transformers\.(\S+)\s+import\s+([^\n]+)\n", content)
    direct_imports = [mod for mod, imp in direct_imports if "# tests_ignore" not in imp]
    for imp in direct_imports:
        import_parts = imp.split(".")
        dep_parts = ["src", "transformers"] + import_parts
        imported_modules.append(os.path.sep.join(dep_parts))

    # Now let's just check that we have proper module files, or append an init for submodules
    dependencies = []
    for imported_module in imported_modules:
        if os.path.isfile(os.path.join(PATH_TO_TRANFORMERS, f"{imported_module}.py")):
            dependencies.append(f"{imported_module}.py")
        elif os.path.isdir(os.path.join(PATH_TO_TRANFORMERS, imported_module)) and os.path.isfile(
            os.path.sep.join([PATH_TO_TRANFORMERS, imported_module, "__init__.py"])
        ):
            dependencies.append(os.path.sep.join([imported_module, "__init__.py"]))
    return dependencies


def get_test_dependencies(test_fname):
    """
    Get the dependencies of a test file.
    """
    with open(os.path.join(PATH_TO_TRANFORMERS, test_fname), "r", encoding="utf-8") as f:
        content = f.read()

    # Tests only have relative imports for other test files
    # TODO Sylvain: handle relative imports cleanly
    relative_imports = re.findall(r"from\s+(\.\S+)\s+import\s+([^\n]+)\n", content)
    relative_imports = [test for test, imp in relative_imports if "# tests_ignore" not in imp]

    def _convert_relative_import_to_file(relative_import):
        level = 0
        while relative_import.startswith("."):
            level += 1
            relative_import = relative_import[1:]

        directory = os.path.sep.join(test_fname.split(os.path.sep)[:-level])
        return os.path.join(directory, f"{relative_import.replace('.', os.path.sep)}.py")

    dependencies = [_convert_relative_import_to_file(relative_import) for relative_import in relative_imports]
    return [f for f in dependencies if os.path.isfile(os.path.join(PATH_TO_TRANFORMERS, f))]


def create_reverse_dependency_tree():
    """
    Create a list of all edges (a, b) which mean that modifying a impacts b with a going over all module and test files.
    """
    modules = [
        str(f.relative_to(PATH_TO_TRANFORMERS))
        for f in (Path(PATH_TO_TRANFORMERS) / "src/transformers").glob("**/*.py")
    ]
    module_edges = [(d, m) for m in modules for d in get_module_dependencies(m)]

    tests = [str(f.relative_to(PATH_TO_TRANFORMERS)) for f in (Path(PATH_TO_TRANFORMERS) / "tests").glob("**/*.py")]
    test_edges = [(d, t) for t in tests for d in get_test_dependencies(t)]

    return module_edges + test_edges


def get_tree_starting_at(module, edges):
    """
    Returns the tree starting at a given module following all edges in the following format: [module, [list of edges
    starting at module], [list of edges starting at the preceding level], ...]
    """
    vertices_seen = [module]
    new_edges = [edge for edge in edges if edge[0] == module and edge[1] != module]
    tree = [module]
    while len(new_edges) > 0:
        tree.append(new_edges)
        final_vertices = list({edge[1] for edge in new_edges})
        vertices_seen.extend(final_vertices)
        new_edges = [edge for edge in edges if edge[0] in final_vertices and edge[1] not in vertices_seen]

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


def create_reverse_dependency_map():
    """
    Create the dependency map from module/test filename to the list of modules/tests that depend on it (even
    recursively).
    """
    modules = [
        str(f.relative_to(PATH_TO_TRANFORMERS))
        for f in (Path(PATH_TO_TRANFORMERS) / "src/transformers").glob("**/*.py")
    ]
    # We grab all the dependencies of each module.
    direct_deps = {m: get_module_dependencies(m) for m in modules}

    # We add all the dependencies of each test file
    tests = [str(f.relative_to(PATH_TO_TRANFORMERS)) for f in (Path(PATH_TO_TRANFORMERS) / "tests").glob("**/*.py")]
    direct_deps.update({t: get_test_dependencies(t) for t in tests})

    all_files = modules + tests

    # This recurses the dependencies
    something_changed = True
    while something_changed:
        something_changed = False
        for m in all_files:
            for d in direct_deps[m]:
                if d not in direct_deps:
                    raise ValueError(f"KeyError:{d}. From {m}")
                for dep in direct_deps[d]:
                    if dep not in direct_deps[m]:
                        direct_deps[m].append(dep)
                        something_changed = True

    # Finally we can build the reverse map.
    reverse_map = collections.defaultdict(list)
    for m in all_files:
        if m.endswith("__init__.py"):
            reverse_map[m].extend(direct_deps[m])
        for d in direct_deps[m]:
            reverse_map[d].append(m)

    return reverse_map


# Any module file that has a test name which can't be inferred automatically from its name should go here. A better
# approach is to (re-)name the test file accordingly, and second best to add the correspondence map here.
SPECIAL_MODULE_TO_TEST_MAP = {
    "commands/add_new_model_like.py": "utils/test_add_new_model_like.py",
    "configuration_utils.py": "test_configuration_common.py",
    "convert_graph_to_onnx.py": "onnx/test_onnx.py",
    "data/data_collator.py": "trainer/test_data_collator.py",
    "deepspeed.py": "deepspeed/",
    "feature_extraction_sequence_utils.py": "test_sequence_feature_extraction_common.py",
    "feature_extraction_utils.py": "test_feature_extraction_common.py",
    "file_utils.py": ["utils/test_file_utils.py", "utils/test_model_output.py"],
    "image_processing_utils.py": ["test_image_processing_common.py", "utils/test_image_processing_utils.py"],
    "image_transforms.py": "test_image_transforms.py",
    "utils/generic.py": ["utils/test_file_utils.py", "utils/test_model_output.py", "utils/test_generic.py"],
    "utils/hub.py": "utils/test_hub_utils.py",
    "modelcard.py": "utils/test_model_card.py",
    "modeling_flax_utils.py": "test_modeling_flax_common.py",
    "modeling_tf_utils.py": ["test_modeling_tf_common.py", "utils/test_modeling_tf_core.py"],
    "modeling_utils.py": ["test_modeling_common.py", "utils/test_offline.py"],
    "models/auto/modeling_auto.py": [
        "models/auto/test_modeling_auto.py",
        "models/auto/test_modeling_tf_pytorch.py",
        "models/bort/test_modeling_bort.py",
        "models/dit/test_modeling_dit.py",
    ],
    "models/auto/modeling_flax_auto.py": "models/auto/test_modeling_flax_auto.py",
    "models/auto/modeling_tf_auto.py": [
        "models/auto/test_modeling_tf_auto.py",
        "models/auto/test_modeling_tf_pytorch.py",
        "models/bort/test_modeling_tf_bort.py",
    ],
    "models/gpt2/modeling_gpt2.py": [
        "models/gpt2/test_modeling_gpt2.py",
        "models/megatron_gpt2/test_modeling_megatron_gpt2.py",
    ],
    "models/dpt/modeling_dpt.py": [
        "models/dpt/test_modeling_dpt.py",
        "models/dpt/test_modeling_dpt_hybrid.py",
    ],
    "optimization.py": "optimization/test_optimization.py",
    "optimization_tf.py": "optimization/test_optimization_tf.py",
    "pipelines/__init__.py": all_pipeline_test_files + all_model_test_files,
    "pipelines/base.py": all_pipeline_test_files + all_model_test_files,
    "pipelines/text2text_generation.py": [
        "pipelines/test_pipelines_text2text_generation.py",
        "pipelines/test_pipelines_summarization.py",
        "pipelines/test_pipelines_translation.py",
    ],
    "pipelines/zero_shot_classification.py": "pipelines/test_pipelines_zero_shot.py",
    "testing_utils.py": "utils/test_skip_decorators.py",
    "tokenization_utils.py": ["test_tokenization_common.py", "tokenization/test_tokenization_utils.py"],
    "tokenization_utils_base.py": ["test_tokenization_common.py", "tokenization/test_tokenization_utils.py"],
    "tokenization_utils_fast.py": [
        "test_tokenization_common.py",
        "tokenization/test_tokenization_utils.py",
        "tokenization/test_tokenization_fast.py",
    ],
    "trainer.py": [
        "trainer/test_trainer.py",
        "extended/test_trainer_ext.py",
        "trainer/test_trainer_distributed.py",
        "trainer/test_trainer_tpu.py",
    ],
    "train_pt_utils.py": "trainer/test_trainer_utils.py",
    "utils/versions.py": "utils/test_versions_utils.py",
}


def module_to_test_file(module_fname):
    """
    Returns the name of the file(s) where `module_fname` is tested.
    """
    splits = module_fname.split(os.path.sep)

    # Special map has priority
    short_name = os.path.sep.join(splits[2:])
    if short_name in SPECIAL_MODULE_TO_TEST_MAP:
        test_file = SPECIAL_MODULE_TO_TEST_MAP[short_name]
        if isinstance(test_file, str):
            return f"tests/{test_file}"
        return [f"tests/{f}" for f in test_file]

    module_name = splits[-1]
    # Fast tokenizers are tested in the same file as the slow ones.
    if module_name.endswith("_fast.py"):
        module_name = module_name.replace("_fast.py", ".py")

    # Special case for pipelines submodules
    if len(splits) >= 2 and splits[-2] == "pipelines":
        default_test_file = f"tests/pipelines/test_pipelines_{module_name}"
        return [default_test_file] + all_model_test_files
    # Special case for benchmarks submodules
    elif len(splits) >= 2 and splits[-2] == "benchmark":
        return ["tests/benchmark/test_benchmark.py", "tests/benchmark/test_benchmark_tf.py"]
    # Special case for commands submodules
    elif len(splits) >= 2 and splits[-2] == "commands":
        return "tests/utils/test_cli.py"
    # Special case for onnx submodules
    elif len(splits) >= 2 and splits[-2] == "onnx":
        return ["tests/onnx/test_features.py", "tests/onnx/test_onnx.py", "tests/onnx/test_onnx_v2.py"]
    # Special case for utils (not the one in src/transformers, the ones at the root of the repo).
    elif len(splits) > 0 and splits[0] == "utils":
        default_test_file = f"tests/repo_utils/test_{module_name}"
    elif len(splits) > 4 and splits[2] == "models":
        default_test_file = f"tests/models/{splits[3]}/test_{module_name}"
    elif len(splits) > 2 and splits[2].startswith("generation"):
        default_test_file = f"tests/generation/test_{module_name}"
    elif len(splits) > 2 and splits[2].startswith("trainer"):
        default_test_file = f"tests/trainer/test_{module_name}"
    else:
        default_test_file = f"tests/utils/test_{module_name}"

    if os.path.isfile(default_test_file):
        return default_test_file

    # Processing -> processor
    if "processing" in default_test_file:
        test_file = default_test_file.replace("processing", "processor")
        if os.path.isfile(test_file):
            return test_file


# This list contains the list of test files we expect never to be launched from a change in a module/util. Those are
# launched separately.
EXPECTED_TEST_FILES_NEVER_TOUCHED = [
    "tests/generation/test_framework_agnostic.py",  # Mixins inherited by actual test classes
    "tests/mixed_int8/test_mixed_int8.py",  # Mixed-int8 bitsandbytes test
    "tests/pipelines/test_pipelines_common.py",  # Actually checked by the pipeline based file
    "tests/sagemaker/test_single_node_gpu.py",  # SageMaker test
    "tests/sagemaker/test_multi_node_model_parallel.py",  # SageMaker test
    "tests/sagemaker/test_multi_node_data_parallel.py",  # SageMaker test
    "tests/test_pipeline_mixin.py",  # Contains no test of its own (only the common tester class)
    "tests/utils/test_doc_samples.py",  # Doc tests
]


def _print_list(l):
    return "\n".join([f"- {f}" for f in l])


def sanity_check():
    """
    Checks that all test files can be touched by a modification in at least one module/utils. This test ensures that
    newly-added test files are properly mapped to some module or utils, so they can be run by the CI.
    """
    # Grab all module and utils
    all_files = [
        str(p.relative_to(PATH_TO_TRANFORMERS))
        for p in (Path(PATH_TO_TRANFORMERS) / "src/transformers").glob("**/*.py")
    ]
    all_files += [
        str(p.relative_to(PATH_TO_TRANFORMERS)) for p in (Path(PATH_TO_TRANFORMERS) / "utils").glob("**/*.py")
    ]

    # Compute all the test files we get from those.
    test_files_found = []
    for f in all_files:
        test_f = module_to_test_file(f)
        if test_f is not None:
            if isinstance(test_f, str):
                test_files_found.append(test_f)
            else:
                test_files_found.extend(test_f)

    # Some of the test files might actually be subfolders so we grab the tests inside.
    test_files = []
    for test_f in test_files_found:
        if os.path.isdir(os.path.join(PATH_TO_TRANFORMERS, test_f)):
            test_files.extend(
                [
                    str(p.relative_to(PATH_TO_TRANFORMERS))
                    for p in (Path(PATH_TO_TRANFORMERS) / test_f).glob("**/test*.py")
                ]
            )
        else:
            test_files.append(test_f)

    # Compare to existing test files
    existing_test_files = [
        str(p.relative_to(PATH_TO_TRANFORMERS)) for p in (Path(PATH_TO_TRANFORMERS) / "tests").glob("**/test*.py")
    ]
    not_touched_test_files = [f for f in existing_test_files if f not in test_files]

    should_be_tested = set(not_touched_test_files) - set(EXPECTED_TEST_FILES_NEVER_TOUCHED)
    if len(should_be_tested) > 0:
        raise ValueError(
            "The following test files are not currently associated with any module or utils files, which means they "
            f"will never get run by the CI:\n{_print_list(should_be_tested)}\n. Make sure the names of these test "
            "files match the name of the module or utils they are testing, or adapt the constant "
            "`SPECIAL_MODULE_TO_TEST_MAP` in `utils/tests_fetcher.py` to add them. If your test file is triggered "
            "separately and is not supposed to be run by the regular CI, add it to the "
            "`EXPECTED_TEST_FILES_NEVER_TOUCHED` constant instead."
        )


def infer_tests_to_run(output_file, diff_with_last_commit=False, filters=None, json_output_file=None):
    modified_files = get_modified_python_files(diff_with_last_commit=diff_with_last_commit)
    print(f"\n### MODIFIED FILES ###\n{_print_list(modified_files)}")

    # Create the map that will give us all impacted modules.
    impacted_modules_map = create_reverse_dependency_map()
    impacted_files = modified_files.copy()
    for f in modified_files:
        if f in impacted_modules_map:
            impacted_files.extend(impacted_modules_map[f])

    # Remove duplicates
    impacted_files = sorted(set(impacted_files))
    print(f"\n### IMPACTED FILES ###\n{_print_list(impacted_files)}")

    # Grab the corresponding test files:
    if "setup.py" in impacted_files:
        test_files_to_run = ["tests"]
        repo_utils_launch = True
    else:
        # Grab the corresponding test files:
        test_files_to_run = []
        for f in impacted_files:
            # Modified test files are always added
            if f.startswith("tests/"):
                test_files_to_run.append(f)
            # Example files are tested separately
            elif f.startswith("examples/pytorch"):
                test_files_to_run.append("examples/pytorch/test_pytorch_examples.py")
                test_files_to_run.append("examples/pytorch/test_accelerate_examples.py")
            elif f.startswith("examples/tensorflow"):
                test_files_to_run.append("examples/tensorflow/test_tensorflow_examples.py")
            elif f.startswith("examples/flax"):
                test_files_to_run.append("examples/flax/test_flax_examples.py")
            else:
                new_tests = module_to_test_file(f)
                if new_tests is not None:
                    if isinstance(new_tests, str):
                        test_files_to_run.append(new_tests)
                    else:
                        test_files_to_run.extend(new_tests)

        # Remove duplicates
        test_files_to_run = sorted(set(test_files_to_run))
        # Make sure we did not end up with a test file that was removed
        test_files_to_run = [f for f in test_files_to_run if os.path.isfile(f) or os.path.isdir(f)]
        if filters is not None:
            filtered_files = []
            for filter in filters:
                filtered_files.extend([f for f in test_files_to_run if f.startswith(filter)])
            test_files_to_run = filtered_files
        repo_utils_launch = any(f.split(os.path.sep)[1] == "repo_utils" for f in test_files_to_run)

    if repo_utils_launch:
        repo_util_file = Path(output_file).parent / "test_repo_utils.txt"
        with open(repo_util_file, "w", encoding="utf-8") as f:
            f.write("tests/repo_utils")

    print(f"\n### TEST TO RUN ###\n{_print_list(test_files_to_run)}")
    if len(test_files_to_run) > 0:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(" ".join(test_files_to_run))

        # Create a map that maps test categories to test files, i.e. `models/bert` -> [...test_modeling_bert.py, ...]

        # Get all test directories (and some common test files) under `tests` and `tests/models` if `test_files_to_run`
        # contains `tests` (i.e. when `setup.py` is changed).
        if "tests" in test_files_to_run:
            test_files_to_run = get_all_tests()

        if json_output_file is not None:
            test_map = {}
            for test_file in test_files_to_run:
                # `test_file` is a path to a test folder/file, starting with `tests/`. For example,
                #   - `tests/models/bert/test_modeling_bert.py` or `tests/models/bert`
                #   - `tests/trainer/test_trainer.py` or `tests/trainer`
                #   - `tests/test_modeling_common.py`
                names = test_file.split(os.path.sep)
                if names[1] == "models":
                    # take the part like `models/bert` for modeling tests
                    key = "/".join(names[1:3])
                elif len(names) > 2 or not test_file.endswith(".py"):
                    # test folders under `tests` or python files under them
                    # take the part like tokenization, `pipeline`, etc. for other test categories
                    key = "/".join(names[1:2])
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sanity_check", action="store_true", help="Only test that all tests and modules are accounted for."
    )
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
        "--filters",
        type=str,
        nargs="*",
        default=["tests"],
        help="Only keep the test files matching one of those filters.",
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
    args = parser.parse_args()
    if args.print_dependencies_of is not None:
        print_tree_deps_of(args.print_dependencies_of)
    elif args.sanity_check:
        sanity_check()
    elif args.filter_tests:
        filter_tests(args.output_file, ["pipelines", "repo_utils"])
    else:
        repo = Repo(PATH_TO_TRANFORMERS)

        diff_with_last_commit = args.diff_with_last_commit
        if not diff_with_last_commit and not repo.head.is_detached and repo.head.ref == repo.refs.main:
            print("main branch detected, fetching tests against last commit.")
            diff_with_last_commit = True

        try:
            infer_tests_to_run(
                args.output_file,
                diff_with_last_commit=diff_with_last_commit,
                filters=args.filters,
                json_output_file=args.json_output_file,
            )
            filter_tests(args.output_file, ["repo_utils"])
        except Exception as e:
            print(f"\nError when trying to grab the relevant tests: {e}\n\nRunning all tests.")
            with open(args.output_file, "w", encoding="utf-8") as f:
                if args.filters is None:
                    f.write("./tests/")
                else:
                    f.write(" ".join(args.filters))
