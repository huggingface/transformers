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

import transformers


PATH_TO_TRANFORMERS = Path(transformers.__path__[0])


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
    # Remove docstrings by splitting on """ then ''':
    splits = content.split('"""')
    content = "".join(splits[::2])
    splits = content.split("'''")
    content = "".join(splits[::2])

    # Remove empty lines and comments
    lines_to_keep = []
    for line in content.split("\n"):
        # remove anything that is after a # sign.
        line = re.sub("#.*$", "", line)
        if line.isspace():
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


def get_modified_python_files():
    """
    Return a list of python files that have been modified between the current head and the master branch.
    """
    repo = Repo(PATH_TO_TRANFORMERS)

    print(f"Master is at {repo.refs.master.commit}")
    print(f"Current head is at {repo.head.commit}")

    branching_commits = repo.merge_base(repo.refs.master, repo.head)
    for commit in branching_commits:
        print(f"Branching commit: {commit}")

    print("\n### DIFF ###\n")
    code_diff = []
    for commit in branching_commits:
        for diff_obj in commit.diff(repo.head.commit):
            # We always add new python files
            if diff_obj.change_type == "A" and diff_obj.b_path.endswith(".py"):
                code_diff.append(diff_obj.b_path)
            # We check that deleted python files won't break correspondping tests.
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
    with open(module_fname, "r", encoding="utf-8") as f:
        content = f.read()

    imports = re.findall(r"from\s+(\.+\S+)\s+import\s+\S+\s", content)
    module_parts = module_fname.split(os.path.sep)
    dependencies = []
    for imp in imports:
        level = 0
        while imp.startswith("."):
            imp = imp[1:]
            level += 1

        dep_parts = module_parts[: len(module_parts) - level] + imp.split(".")
        if os.path.isfile(os.path.sep.join(dep_parts) + ".py"):
            dependencies.append(os.path.sep.join(dep_parts) + ".py")
    return dependencies


def create_reverse_dependency_map():
    """
    Create the dependency map from module filename to the list of modules that depend on it.
    """
    modules = [
        str(f.relative_to(PATH_TO_TRANFORMERS))
        for f in (Path(PATH_TO_TRANFORMERS) / "src/transformers").glob("**/*.py")
    ]
    direct_deps = {m: get_module_dependencies(m) for m in modules}

    something_changed = True
    while something_changed:
        something_changed = False
        for m in modules:
            for d in direct_deps[m]:
                for dep in direct_deps[d]:
                    if dep not in direct_deps[m]:
                        direct_deps[m].append(dep)
                        something_changed = True

    reverse_map = collections.defaultdict(list)
    for m in modules:
        for d in direct_deps[m]:
            reverse_map[d].append(m)

    return reverse_map


# Any module file that has a test name which is not obviously inferred from its name should go there (better name
# your test file appropriately than add a new entry in this list if you can!)
SPECIAL_MODULE_TO_TEST_MAP = {
    "configuration_utils.py": "test_configuration_common.py",
    "convert_graph_to_onnx.py": "test_onnx.py",
    "data/data_collator.py": "test_data_collator.py",
    "deepspeed.py": "deepspeed/test_deepspeed.py",
    "feature_extraction_sequence_utils.py": "test_sequence_feature_extraction_common.py",
    "feature_extraction_utils.py": "test_feature_extraction_common.py",
    "file_utils.py": ["test_file_utils.py", "test_model_output.py"],
    "modelcard.py": "test_model_card.py",
    "modeling_flax_utils.py": "test_modeling_flax_common.py",
    "modeling_tf_utils.py": "test_modeling_tf_common.py",
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
    "pipelines/base.py": "test_pipelines_common.py",
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
    if splits[-2] == "pipelines":
        default_test_file = f"tests/test_pipelines_{module_name}"
    # Special case for benchmarks submodules
    elif splits[-2] == "benchmark":
        return ["tests/test_benchmark.py", "tests/test_benchmark_tf.py"]
    # Special case for commands submodules
    elif splits[-2] == "commands":
        return "tests/test_cli.py"
    # Special case for utils (not the one in src/transformers, the ones at the root of the repo).
    elif splits[0] == "utils":
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

    # Compare to existing test files
    existing_test_files = [
        str(p.relative_to(PATH_TO_TRANFORMERS)) for p in (Path(PATH_TO_TRANFORMERS) / "tests").glob("**/test*.py")
    ]
    not_touched_test_files = [f for f in existing_test_files if f not in test_files_found]

    should_be_tested = set(not_touched_test_files) - set(EXPECTED_TEST_FILES_NEVER_TOUCHED)
    if len(should_be_tested) > 0:
        raise ValueError(
            "The following test files are not currently associated with any module or utils files, which means they "
            f"will never get run by the CI:\n{_print_list(should_be_tested)}\n. Make sure the names of these test "
            "files match the name of the module or utils they are testing, or adapt the constant "
            "`SPECIAL_MODULE_TO_TEST_MAP` in `utils/git_diff.py` to add them. If your test file is triggered "
            "separately and is not supposed to be run by the regular CI, add it to the "
            "`EXPECTED_TEST_FILES_NEVER_TOUCHED` constant instead."
        )


def infer_tests_to_run():
    modified_files = get_modified_python_files()
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
    test_files_to_run = []
    for f in impacted_files:
        # Modified test files are always added
        if f.startswith("tests/"):
            test_files_to_run.append(f)
        else:
            new_tests = module_to_test_file(f)
            if new_tests is not None:
                if isinstance(new_tests, str):
                    test_files_to_run.append(new_tests)
                else:
                    test_files_to_run.extend(new_tests)

    # Remove duplicates
    test_files_to_run = sorted(list(set(test_files_to_run)))
    print(f"\n### TEST TO RUN ###\n{_print_list(test_files_to_run)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity_check", action="store_true", help="Whether to just perform a sanity check.")
    args = parser.parse_args()
    if args.sanity_check:
        sanity_check()
    else:
        infer_tests_to_run()
