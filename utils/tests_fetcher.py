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
import os
import re
from contextlib import contextmanager
from pathlib import Path

from git import Repo


# This script is intended to be run from the root of the repo but you can adapt this constant if you need to.
PATH_TO_TRANFORMERS = "."


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

    - the current head and the master branch if `diff_with_last_commit=False` (default)
    - the current head and its parent commit otherwise.
    """
    repo = Repo(PATH_TO_TRANFORMERS)

    if not diff_with_last_commit:
        print(f"Master is at {repo.refs.master.commit}")
        print(f"Current head is at {repo.head.commit}")

        branching_commits = repo.merge_base(repo.refs.master, repo.head)
        for commit in branching_commits:
            print(f"Branching commit: {commit}")
        return get_diff(repo, repo.head.commit, branching_commits)
    else:
        print(f"Master is at {repo.head.commit}")
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
    relative_imports = re.findall(r"from\s+\.(\S+)\s+import\s+([^\n]+)\n", content)
    relative_imports = [test for test, imp in relative_imports if "# tests_ignore" not in imp]
    return [os.path.join("tests", f"{test}.py") for test in relative_imports]


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
    "configuration_utils.py": "test_configuration_common.py",
    "convert_graph_to_onnx.py": "test_onnx.py",
    "data/data_collator.py": "test_data_collator.py",
    "deepspeed.py": "deepspeed/",
    "feature_extraction_sequence_utils.py": "test_sequence_feature_extraction_common.py",
    "feature_extraction_utils.py": "test_feature_extraction_common.py",
    "file_utils.py": ["test_file_utils.py", "test_model_output.py"],
    "modelcard.py": "test_model_card.py",
    "modeling_flax_utils.py": "test_modeling_flax_common.py",
    "modeling_tf_utils.py": ["test_modeling_tf_common.py", "test_modeling_tf_core.py"],
    "modeling_utils.py": ["test_modeling_common.py", "test_offline.py"],
    "models/auto/modeling_auto.py": ["test_modeling_auto.py", "test_modeling_tf_pytorch.py", "test_modeling_bort.py"],
    "models/auto/modeling_flax_auto.py": "test_flax_auto.py",
    "models/auto/modeling_tf_auto.py": [
        "test_modeling_tf_auto.py",
        "test_modeling_tf_pytorch.py",
        "test_modeling_tf_bort.py",
    ],
    "models/blenderbot_small/tokenization_blenderbot_small.py": "test_tokenization_small_blenderbot.py",
    "models/blenderbot_small/tokenization_blenderbot_small_fast.py": "test_tokenization_small_blenderbot.py",
    "models/gpt2/modeling_gpt2.py": ["test_modeling_gpt2.py", "test_modeling_megatron_gpt2.py"],
    "pipelines/base.py": "test_pipelines_*.py",
    "pipelines/text2text_generation.py": [
        "test_pipelines_text2text_generation.py",
        "test_pipelines_summarization.py",
        "test_pipelines_translation.py",
    ],
    "pipelines/zero_shot_classification.py": "test_pipelines_zero_shot.py",
    "testing_utils.py": "test_skip_decorators.py",
    "tokenization_utils.py": "test_tokenization_common.py",
    "tokenization_utils_base.py": "test_tokenization_common.py",
    "tokenization_utils_fast.py": "test_tokenization_fast.py",
    "trainer.py": [
        "test_trainer.py",
        "extended/test_trainer_ext.py",
        "test_trainer_distributed.py",
        "test_trainer_tpu.py",
    ],
    "train_pt_utils.py": "test_trainer_utils.py",
    "utils/versions.py": "test_versions_utils.py",
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
        default_test_file = f"tests/test_pipelines_{module_name}"
    # Special case for benchmarks submodules
    elif len(splits) >= 2 and splits[-2] == "benchmark":
        return ["tests/test_benchmark.py", "tests/test_benchmark_tf.py"]
    # Special case for commands submodules
    elif len(splits) >= 2 and splits[-2] == "commands":
        return "tests/test_cli.py"
    # Special case for onnx submodules
    elif len(splits) >= 2 and splits[-2] == "onnx":
        return ["tests/test_onnx.py", "tests/test_onnx_v2.py"]
    # Special case for utils (not the one in src/transformers, the ones at the root of the repo).
    elif len(splits) > 0 and splits[0] == "utils":
        default_test_file = f"tests/test_utils_{module_name}"
    else:
        default_test_file = f"tests/test_{module_name}"

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
    "tests/test_doc_samples.py",  # Doc tests
    "tests/test_pipelines_common.py",  # Actually checked by the pipeline based file
    "tests/sagemaker/test_single_node_gpu.py",  # SageMaker test
    "tests/sagemaker/test_multi_node_model_parallel.py",  # SageMaker test
    "tests/sagemaker/test_multi_node_data_parallel.py",  # SageMaker test
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


def infer_tests_to_run(output_file, diff_with_last_commit=False, filters=None):
    modified_files = get_modified_python_files(diff_with_last_commit=diff_with_last_commit)
    print(f"\n### MODIFIED FILES ###\n{_print_list(modified_files)}")

    # Create the map that will give us all impacted modules.
    impacted_modules_map = create_reverse_dependency_map()
    impacted_files = modified_files.copy()
    for f in modified_files:
        if f in impacted_modules_map:
            impacted_files.extend(impacted_modules_map[f])

    # Remove duplicates
    impacted_files = sorted(list(set(impacted_files)))
    print(f"\n### IMPACTED FILES ###\n{_print_list(impacted_files)}")

    # Grab the corresponding test files:
    if "setup.py" in impacted_files:
        test_files_to_run = ["tests"]
    else:
        # Grab the corresponding test files:
        test_files_to_run = []
        for f in impacted_files:
            # Modified test files are always added
            if f.startswith("tests/"):
                test_files_to_run.append(f)
            # Example files are tested separately
            elif f.startswith("examples/pytorch"):
                test_files_to_run.append("examples/pytorch/test_examples.py")
            elif f.startswith("examples/flax"):
                test_files_to_run.append("examples/flax/test_examples.py")
            else:
                new_tests = module_to_test_file(f)
                if new_tests is not None:
                    if isinstance(new_tests, str):
                        test_files_to_run.append(new_tests)
                    else:
                        test_files_to_run.extend(new_tests)

        # Remove duplicates
        test_files_to_run = sorted(list(set(test_files_to_run)))
        # Make sure we did not end up with a test file that was removed
        test_files_to_run = [f for f in test_files_to_run if os.path.isfile(f) or os.path.isdir(f)]
        if filters is not None:
            filtered_files = []
            for filter in filters:
                filtered_files.extend([f for f in test_files_to_run if f.startswith(filter)])
            test_files_to_run = filtered_files

    print(f"\n### TEST TO RUN ###\n{_print_list(test_files_to_run)}")
    if len(test_files_to_run) > 0:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(" ".join(test_files_to_run))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sanity_check", action="store_true", help="Only test that all tests and modules are accounted for."
    )
    parser.add_argument(
        "--output_file", type=str, default="test_list.txt", help="Where to store the list of tests to run"
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
    args = parser.parse_args()
    if args.sanity_check:
        sanity_check()
    else:
        repo = Repo(PATH_TO_TRANFORMERS)

        diff_with_last_commit = args.diff_with_last_commit
        if not diff_with_last_commit and not repo.head.is_detached and repo.head.ref == repo.refs.master:
            print("Master branch detected, fetching tests against last commit.")
            diff_with_last_commit = True

        try:
            infer_tests_to_run(args.output_file, diff_with_last_commit=diff_with_last_commit, filters=args.filters)
        except Exception as e:
            print(f"\nError when trying to grab the relevant tests: {e}\n\nRunning all tests.")
            with open(args.output_file, "w", encoding="utf-8") as f:
                if args.filters is None:
                    f.write("./tests/")
                else:
                    f.write(" ".join(args.filters))
