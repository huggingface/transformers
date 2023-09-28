# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
import copy
import glob
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml


COMMON_ENV_VARIABLES = {
    "OMP_NUM_THREADS": 1,
    "TRANSFORMERS_IS_CI": True,
    "PYTEST_TIMEOUT": 120,
    "RUN_PIPELINE_TESTS": False,
    "RUN_PT_TF_CROSS_TESTS": False,
    "RUN_PT_FLAX_CROSS_TESTS": False,
}
# Disable the use of {"s": None} as the output is way too long, causing the navigation on CircleCI impractical
COMMON_PYTEST_OPTIONS = {"max-worker-restart": 0, "dist": "loadfile"}
DEFAULT_DOCKER_IMAGE = [{"image": "cimg/python:3.8.12"}]


class EmptyJob:
    job_name = "empty"

    def to_dict(self):
        return {
            "working_directory": "~/transformers",
            "docker": copy.deepcopy(DEFAULT_DOCKER_IMAGE),
            "steps":["checkout"],
        }


@dataclass
class CircleCIJob:
    name: str
    additional_env: Dict[str, Any] = None
    cache_name: str = None
    cache_version: str = "0.7"
    docker_image: List[Dict[str, str]] = None
    install_steps: List[str] = None
    marker: Optional[str] = None
    parallelism: Optional[int] = 1
    pytest_num_workers: int = 8
    pytest_options: Dict[str, Any] = None
    resource_class: Optional[str] = "xlarge"
    tests_to_run: Optional[List[str]] = None
    working_directory: str = "~/transformers"
    # This should be only used for doctest job!
    command_timeout: Optional[int] = None

    def __post_init__(self):
        # Deal with defaults for mutable attributes.
        if self.additional_env is None:
            self.additional_env = {}
        if self.cache_name is None:
            self.cache_name = self.name
        if self.docker_image is None:
            # Let's avoid changing the default list and make a copy.
            self.docker_image = copy.deepcopy(DEFAULT_DOCKER_IMAGE)
        if self.install_steps is None:
            self.install_steps = []
        if self.pytest_options is None:
            self.pytest_options = {}
        if isinstance(self.tests_to_run, str):
            self.tests_to_run = [self.tests_to_run]
        if self.parallelism is None:
            self.parallelism = 1

    def to_dict(self):
        env = COMMON_ENV_VARIABLES.copy()
        env.update(self.additional_env)

        cache_branch_prefix = os.environ.get("CIRCLE_BRANCH", "pull")
        if cache_branch_prefix != "main":
            cache_branch_prefix = "pull"

        job = {
            "working_directory": self.working_directory,
            "docker": self.docker_image,
            "environment": env,
        }
        if self.resource_class is not None:
            job["resource_class"] = self.resource_class
        if self.parallelism is not None:
            job["parallelism"] = self.parallelism
        steps = [
            "checkout",
            {"attach_workspace": {"at": "~/transformers/test_preparation"}},
            {
                "restore_cache": {
                    "keys": [
                        # check the fully-matched cache first
                        f"v{self.cache_version}-{self.cache_name}-{cache_branch_prefix}-pip-" + '{{ checksum "setup.py" }}',
                        # try the partially-matched cache from `main`
                        f"v{self.cache_version}-{self.cache_name}-main-pip-",
                        # try the general partially-matched cache
                        f"v{self.cache_version}-{self.cache_name}-{cache_branch_prefix}-pip-",
                    ]
                }
            },
            {
                "restore_cache": {
                    "keys": [
                        f"v{self.cache_version}-{self.cache_name}-{cache_branch_prefix}-site-packages-" + '{{ checksum "setup.py" }}',
                        f"v{self.cache_version}-{self.cache_name}-main-site-packages-",
                        f"v{self.cache_version}-{self.cache_name}-{cache_branch_prefix}-site-packages-",
                    ]
                }
            },
        ]
        steps.extend([{"run": l} for l in self.install_steps])
        steps.append(
            {
                "save_cache": {
                    "key": f"v{self.cache_version}-{self.cache_name}-{cache_branch_prefix}-pip-" + '{{ checksum "setup.py" }}',
                    "paths": ["~/.cache/pip"],
                }
            }
        )
        steps.append(
            {
                "save_cache": {
                    "key": f"v{self.cache_version}-{self.cache_name}-{cache_branch_prefix}-site-packages-" + '{{ checksum "setup.py" }}',
                    "paths": ["~/.pyenv/versions/"],
                }
            }
        )
        steps.append({"run": {"name": "Show installed libraries and their versions", "command": "pip freeze | tee installed.txt"}})
        steps.append({"store_artifacts": {"path": "~/transformers/installed.txt"}})

        all_options = {**COMMON_PYTEST_OPTIONS, **self.pytest_options}
        pytest_flags = [f"--{key}={value}" if (value is not None or key in ["doctest-modules"]) else f"-{key}" for key, value in all_options.items()]
        pytest_flags.append(
            f"--make-reports={self.name}" if "examples" in self.name else f"--make-reports=tests_{self.name}"
        )

        steps.append({"run": {"name": "Create `test-results` directory", "command": "mkdir test-results"}})

        test_command = ""
        if self.command_timeout:
            test_command = f"timeout {self.command_timeout} "
        test_command += f"python -m pytest --junitxml=test-results/junit.xml -n {self.pytest_num_workers} " + " ".join(pytest_flags)

        if self.parallelism == 1:
            if self.tests_to_run is None:
                test_command += " << pipeline.parameters.tests_to_run >>"
            else:
                test_command += " " + " ".join(self.tests_to_run)
        else:
            # We need explicit list instead of `pipeline.parameters.tests_to_run` (only available at job runtime)
            tests = self.tests_to_run
            if tests is None:
                folder = os.environ["test_preparation_dir"]
                test_file = os.path.join(folder, "filtered_test_list.txt")
                if os.path.exists(test_file):
                    with open(test_file) as f:
                        tests = f.read().split(" ")

            # expand the test list
            if tests == ["tests"]:
                tests = [os.path.join("tests", x) for x in os.listdir("tests")]
            expanded_tests = []
            for test in tests:
                if test.endswith(".py"):
                    expanded_tests.append(test)
                elif test == "tests/models":
                    expanded_tests.extend([os.path.join(test, x) for x in os.listdir(test)])
                elif test == "tests/pipelines":
                    expanded_tests.extend([os.path.join(test, x) for x in os.listdir(test)])
                else:
                    expanded_tests.append(test)
            # Avoid long tests always being collected together
            random.shuffle(expanded_tests)
            tests = " ".join(expanded_tests)

            # Each executor to run ~10 tests
            n_executors = max(len(tests) // 10, 1)
            # Avoid empty test list on some executor(s) or launching too many executors
            if n_executors > self.parallelism:
                n_executors = self.parallelism
            job["parallelism"] = n_executors

            # Need to be newline separated for the command `circleci tests split` below
            command = f'echo {tests} | tr " " "\\n" >> tests.txt'
            steps.append({"run": {"name": "Get tests", "command": command}})

            command = 'TESTS=$(circleci tests split tests.txt) && echo $TESTS > splitted_tests.txt'
            steps.append({"run": {"name": "Split tests", "command": command}})

            steps.append({"store_artifacts": {"path": "~/transformers/tests.txt"}})
            steps.append({"store_artifacts": {"path": "~/transformers/splitted_tests.txt"}})

            test_command = ""
            if self.timeout:
                test_command = f"timeout {self.timeout} "
            test_command += f"python -m pytest -n {self.pytest_num_workers} " + " ".join(pytest_flags)
            test_command += " $(cat splitted_tests.txt)"
        if self.marker is not None:
            test_command += f" -m {self.marker}"

        if self.name == "pr_documentation_tests":
            # can't use ` | tee tee tests_output.txt` as usual
            test_command += " > tests_output.txt"
            # Save the return code, so we can check if it is timeout in the next step.
            test_command += '; touch "$?".txt'
            # Never fail the test step for the doctest job. We will check the results in the next step, and fail that
            # step instead if the actual test failures are found. This is to avoid the timeout being reported as test
            # failure.
            test_command = f"({test_command}) || true"
        else:
            test_command += " || true"
        steps.append({"run": {"name": "Run tests", "command": test_command}})

        # Deal with errors
        check_test_command = f'if [ -s reports/{self.job_name}/errors.txt ]; '
        check_test_command += 'then echo "Some tests errored out!"; echo ""; '
        check_test_command += f'cat reports/{self.job_name}/errors.txt; '
        check_test_command += 'echo ""; echo ""; '

        py_command = f'import os; fp = open("reports/{self.job_name}/summary_short.txt"); failed = os.linesep.join([x for x in fp.read().split(os.linesep) if x.startswith("ERROR ")]); fp.close(); fp = open("summary_short.txt", "w"); fp.write(failed); fp.close()'
        check_test_command += f"$(python3 -c '{py_command}'); "
        check_test_command += f'cat summary_short.txt; echo ""; exit -1; '

        # Deeal with failed tests
        check_test_command += f'elif [ -s reports/{self.job_name}/failures_short.txt ]; '
        check_test_command += 'then echo "Some tests failed!"; echo ""; '
        check_test_command += f'cat reports/{self.job_name}/failures_short.txt; '
        check_test_command += 'echo ""; echo ""; '

        py_command = f'import os; fp = open("reports/{self.job_name}/summary_short.txt"); failed = os.linesep.join([x for x in fp.read().split(os.linesep) if x.startswith("FAILED ")]); fp.close(); fp = open("summary_short.txt", "w"); fp.write(failed); fp.close()'
        check_test_command += f"$(python3 -c '{py_command}'); "
        check_test_command += f'cat summary_short.txt; echo ""; exit -1; '

        check_test_command += f'elif [ -s reports/{self.job_name}/stats.txt ]; then echo "All tests pass!"; '

        # return code `124` means the previous (pytest run) step is timeout
        if self.name == "pr_documentation_tests":
            check_test_command += 'elif [ -f 124.txt ]; then echo "doctest timeout!"; '

        check_test_command += 'else echo "other fatal error"; echo ""; exit -1; fi;'

        steps.append({"run": {"name": "Check test results", "command": check_test_command}})

        steps.append({"store_test_results": {"path": "test-results"}})

        steps.append({"store_artifacts": {"path": "~/transformers/tests_output.txt"}})
        steps.append({"store_artifacts": {"path": "~/transformers/reports"}})
        job["steps"] = steps
        return job

    @property
    def job_name(self):
        return self.name if "examples" in self.name else f"tests_{self.name}"


# JOBS
torch_and_tf_job = CircleCIJob(
    "torch_and_tf",
    additional_env={"RUN_PT_TF_CROSS_TESTS": True},
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install -y libsndfile1-dev espeak-ng git-lfs cmake",
        "git lfs install",
        "pip install --upgrade --upgrade-strategy eager pip",
        "pip install -U --upgrade-strategy eager .[sklearn,tf-cpu,torch,testing,sentencepiece,torch-speech,vision]",
        "pip install -U --upgrade-strategy eager tensorflow_probability",
        "pip install -U --upgrade-strategy eager git+https://github.com/huggingface/accelerate",
    ],
    marker="is_pt_tf_cross_test",
    pytest_options={"rA": None, "durations": 0},
)


torch_and_flax_job = CircleCIJob(
    "torch_and_flax",
    additional_env={"RUN_PT_FLAX_CROSS_TESTS": True},
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install -y libsndfile1-dev espeak-ng",
        "pip install -U --upgrade-strategy eager --upgrade pip",
        "pip install -U --upgrade-strategy eager .[sklearn,flax,torch,testing,sentencepiece,torch-speech,vision]",
        "pip install -U --upgrade-strategy eager git+https://github.com/huggingface/accelerate",
    ],
    marker="is_pt_flax_cross_test",
    pytest_options={"rA": None, "durations": 0},
)


torch_job = CircleCIJob(
    "torch",
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install -y libsndfile1-dev espeak-ng time",
        "pip install --upgrade --upgrade-strategy eager pip",
        "pip install -U --upgrade-strategy eager .[sklearn,torch,testing,sentencepiece,torch-speech,vision,timm]",
        "pip install -U --upgrade-strategy eager git+https://github.com/huggingface/accelerate",
    ],
    parallelism=1,
    pytest_num_workers=8,
)


tf_job = CircleCIJob(
    "tf",
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install -y libsndfile1-dev espeak-ng cmake",
        "pip install --upgrade --upgrade-strategy eager pip",
        "pip install -U --upgrade-strategy eager .[sklearn,tf-cpu,testing,sentencepiece,tf-speech,vision]",
        "pip install -U --upgrade-strategy eager tensorflow_probability",
    ],
    parallelism=1,
)


flax_job = CircleCIJob(
    "flax",
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install -y libsndfile1-dev espeak-ng",
        "pip install --upgrade --upgrade-strategy eager pip",
        "pip install -U --upgrade-strategy eager .[flax,testing,sentencepiece,flax-speech,vision]",
    ],
    parallelism=1,
)


pipelines_torch_job = CircleCIJob(
    "pipelines_torch",
    additional_env={"RUN_PIPELINE_TESTS": True},
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install -y libsndfile1-dev espeak-ng",
        "pip install --upgrade --upgrade-strategy eager pip",
        "pip install -U --upgrade-strategy eager .[sklearn,torch,testing,sentencepiece,torch-speech,vision,timm,video]",
    ],
    marker="is_pipeline_test",
)


pipelines_tf_job = CircleCIJob(
    "pipelines_tf",
    additional_env={"RUN_PIPELINE_TESTS": True},
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install -y cmake",
        "pip install --upgrade --upgrade-strategy eager pip",
        "pip install -U --upgrade-strategy eager .[sklearn,tf-cpu,testing,sentencepiece,vision]",
        "pip install -U --upgrade-strategy eager tensorflow_probability",
    ],
    marker="is_pipeline_test",
)


custom_tokenizers_job = CircleCIJob(
    "custom_tokenizers",
    additional_env={"RUN_CUSTOM_TOKENIZERS": True},
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install -y cmake",
        {
            "name": "install jumanpp",
            "command":
                "wget https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc3/jumanpp-2.0.0-rc3.tar.xz\n"
                "tar xvf jumanpp-2.0.0-rc3.tar.xz\n"
                "mkdir jumanpp-2.0.0-rc3/bld\n"
                "cd jumanpp-2.0.0-rc3/bld\n"
                "sudo cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local\n"
                "sudo make install\n",
        },
        "pip install --upgrade --upgrade-strategy eager pip",
        "pip install -U --upgrade-strategy eager .[ja,testing,sentencepiece,jieba,spacy,ftfy,rjieba]",
        "python -m unidic download",
    ],
    parallelism=None,
    resource_class=None,
    tests_to_run=[
        "./tests/models/bert_japanese/test_tokenization_bert_japanese.py",
        "./tests/models/openai/test_tokenization_openai.py",
        "./tests/models/clip/test_tokenization_clip.py",
    ],
)


examples_torch_job = CircleCIJob(
    "examples_torch",
    cache_name="torch_examples",
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install -y libsndfile1-dev espeak-ng",
        "pip install --upgrade --upgrade-strategy eager pip",
        "pip install -U --upgrade-strategy eager .[sklearn,torch,sentencepiece,testing,torch-speech]",
        "pip install -U --upgrade-strategy eager -r examples/pytorch/_tests_requirements.txt",
    ],
)


examples_tensorflow_job = CircleCIJob(
    "examples_tensorflow",
    cache_name="tensorflow_examples",
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install -y cmake",
        "pip install --upgrade --upgrade-strategy eager pip",
        "pip install -U --upgrade-strategy eager .[sklearn,tensorflow,sentencepiece,testing]",
        "pip install -U --upgrade-strategy eager -r examples/tensorflow/_tests_requirements.txt",
    ],
)


examples_flax_job = CircleCIJob(
    "examples_flax",
    cache_name="flax_examples",
    install_steps=[
        "pip install --upgrade --upgrade-strategy eager pip",
        "pip install -U --upgrade-strategy eager .[flax,testing,sentencepiece]",
        "pip install -U --upgrade-strategy eager -r examples/flax/_tests_requirements.txt",
    ],
)


hub_job = CircleCIJob(
    "hub",
    additional_env={"HUGGINGFACE_CO_STAGING": True},
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install git-lfs",
        'git config --global user.email "ci@dummy.com"',
        'git config --global user.name "ci"',
        "pip install --upgrade --upgrade-strategy eager pip",
        "pip install -U --upgrade-strategy eager .[torch,sentencepiece,testing,vision]",
    ],
    marker="is_staging_test",
    pytest_num_workers=1,
)


onnx_job = CircleCIJob(
    "onnx",
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install -y cmake",
        "pip install --upgrade --upgrade-strategy eager pip",
        "pip install -U --upgrade-strategy eager .[torch,tf,testing,sentencepiece,onnxruntime,vision,rjieba]",
    ],
    pytest_options={"k onnx": None},
    pytest_num_workers=1,
)


exotic_models_job = CircleCIJob(
    "exotic_models",
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install -y libsndfile1-dev",
        "pip install --upgrade --upgrade-strategy eager pip",
        "pip install -U --upgrade-strategy eager .[torch,testing,vision]",
        "pip install -U --upgrade-strategy eager torchvision",
        "pip install -U --upgrade-strategy eager scipy",
        "pip install -U --upgrade-strategy eager 'git+https://github.com/facebookresearch/detectron2.git'",
        "sudo apt install tesseract-ocr",
        "pip install -U --upgrade-strategy eager pytesseract",
        "pip install -U --upgrade-strategy eager natten",
        "pip install -U --upgrade-strategy eager python-Levenshtein",
        "pip install -U --upgrade-strategy eager opencv-python",
        "pip install -U --upgrade-strategy eager nltk",
    ],
    tests_to_run=[
        "tests/models/*layoutlmv*",
        "tests/models/*nat",
        "tests/models/deta",
        "tests/models/nougat",
    ],
    pytest_num_workers=1,
    pytest_options={"durations": 100},
)


repo_utils_job = CircleCIJob(
    "repo_utils",
    install_steps=[
        "pip install --upgrade --upgrade-strategy eager pip",
        "pip install -U --upgrade-strategy eager .[quality,testing,torch]",
    ],
    parallelism=None,
    pytest_num_workers=1,
    resource_class="large",
    tests_to_run="tests/repo_utils",
)


# We also include a `dummy.py` file in the files to be doc-tested to prevent edge case failure. Otherwise, the pytest
# hangs forever during test collection while showing `collecting 0 items / 21 errors`. (To see this, we have to remove
# the bash output redirection.)
py_command = 'from utils.tests_fetcher import get_doctest_files; to_test = get_doctest_files() + ["dummy.py"]; to_test = " ".join(to_test); print(to_test)'
py_command = f"$(python3 -c '{py_command}')"
command = f'echo "{py_command}" > pr_documentation_tests_temp.txt'
doc_test_job = CircleCIJob(
    "pr_documentation_tests",
    additional_env={"TRANSFORMERS_VERBOSITY": "error", "DATASETS_VERBOSITY": "error", "SKIP_CUDA_DOCTEST": "1"},
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install -y libsndfile1-dev espeak-ng time ffmpeg",
        "pip install --upgrade --upgrade-strategy eager pip",
        "pip install -U --upgrade-strategy eager -e .[dev]",
        "pip install -U --upgrade-strategy eager git+https://github.com/huggingface/accelerate",
        "pip install --upgrade --upgrade-strategy eager pytest pytest-sugar",
        "pip install -U --upgrade-strategy eager natten",
        "find -name __pycache__ -delete",
        "find . -name \*.pyc -delete",
        # Add an empty file to keep the test step running correctly even no file is selected to be tested.
        "touch dummy.py",
        {
            "name": "Get files to test",
            "command": command,
        },
        {
            "name": "Show information in `Get files to test`",
            "command":
                "cat pr_documentation_tests_temp.txt"
        },
        {
            "name": "Get the last line in `pr_documentation_tests.txt`",
            "command":
                "tail -n1 pr_documentation_tests_temp.txt | tee pr_documentation_tests.txt"
        },
    ],
    tests_to_run="$(cat pr_documentation_tests.txt)",  # noqa
    pytest_options={"-doctest-modules": None, "doctest-glob": "*.md", "dist": "loadfile", "rvsA": None},
    command_timeout=1200,  # test cannot run longer than 1200 seconds
    pytest_num_workers=1,
)

REGULAR_TESTS = [
    torch_and_tf_job,
    torch_and_flax_job,
    torch_job,
    tf_job,
    flax_job,
    custom_tokenizers_job,
    hub_job,
    onnx_job,
    exotic_models_job,
]
EXAMPLES_TESTS = [
    examples_torch_job,
    examples_tensorflow_job,
    examples_flax_job,
]
PIPELINE_TESTS = [
    pipelines_torch_job,
    pipelines_tf_job,
]
REPO_UTIL_TESTS = [repo_utils_job]
DOC_TESTS = [doc_test_job]


def create_circleci_config(folder=None):
    if folder is None:
        folder = os.getcwd()
    # Used in CircleCIJob.to_dict() to expand the test list (for using parallelism)
    os.environ["test_preparation_dir"] = folder
    jobs = []
    all_test_file = os.path.join(folder, "test_list.txt")
    if os.path.exists(all_test_file):
        with open(all_test_file) as f:
            all_test_list = f.read()
    else:
        all_test_list = []
    if len(all_test_list) > 0:
        jobs.extend(PIPELINE_TESTS)

    test_file = os.path.join(folder, "filtered_test_list.txt")
    if os.path.exists(test_file):
        with open(test_file) as f:
            test_list = f.read()
    else:
        test_list = []
    if len(test_list) > 0:
        jobs.extend(REGULAR_TESTS)

        extended_tests_to_run = set(test_list.split())
        # Extend the test files for cross test jobs
        for job in jobs:
            if job.job_name in ["tests_torch_and_tf", "tests_torch_and_flax"]:
                for test_path in copy.copy(extended_tests_to_run):
                    dir_path, fn = os.path.split(test_path)
                    if fn.startswith("test_modeling_tf_"):
                        fn = fn.replace("test_modeling_tf_", "test_modeling_")
                    elif fn.startswith("test_modeling_flax_"):
                        fn = fn.replace("test_modeling_flax_", "test_modeling_")
                    else:
                        if job.job_name == "test_torch_and_tf":
                            fn = fn.replace("test_modeling_", "test_modeling_tf_")
                        elif job.job_name == "test_torch_and_flax":
                            fn = fn.replace("test_modeling_", "test_modeling_flax_")
                    new_test_file = str(os.path.join(dir_path, fn))
                    if os.path.isfile(new_test_file):
                        if new_test_file not in extended_tests_to_run:
                            extended_tests_to_run.add(new_test_file)
        extended_tests_to_run = sorted(extended_tests_to_run)
        for job in jobs:
            if job.job_name in ["tests_torch_and_tf", "tests_torch_and_flax"]:
                job.tests_to_run = extended_tests_to_run
        fn = "filtered_test_list_cross_tests.txt"
        f_path = os.path.join(folder, fn)
        with open(f_path, "w") as fp:
            fp.write(" ".join(extended_tests_to_run))

    example_file = os.path.join(folder, "examples_test_list.txt")
    if os.path.exists(example_file) and os.path.getsize(example_file) > 0:
        with open(example_file, "r", encoding="utf-8") as f:
            example_tests = f.read()
        for job in EXAMPLES_TESTS:
            framework = job.name.replace("examples_", "").replace("torch", "pytorch")
            if example_tests == "all":
                job.tests_to_run = [f"examples/{framework}"]
            else:
                job.tests_to_run = [f for f in example_tests.split(" ") if f.startswith(f"examples/{framework}")]

            if len(job.tests_to_run) > 0:
                jobs.append(job)

    doctest_file = os.path.join(folder, "doctest_list.txt")
    if os.path.exists(doctest_file):
        with open(doctest_file) as f:
            doctest_list = f.read()
    else:
        doctest_list = []
    if len(doctest_list) > 0:
        jobs.extend(DOC_TESTS)

    repo_util_file = os.path.join(folder, "test_repo_utils.txt")
    if os.path.exists(repo_util_file) and os.path.getsize(repo_util_file) > 0:
        jobs.extend(REPO_UTIL_TESTS)

    if len(jobs) == 0:
        jobs = [EmptyJob()]
    config = {"version": "2.1"}
    config["parameters"] = {
        # Only used to accept the parameters from the trigger
        "nightly": {"type": "boolean", "default": False},
        "tests_to_run": {"type": "string", "default": test_list},
    }
    config["jobs"] = {j.job_name: j.to_dict() for j in jobs}
    config["workflows"] = {"version": 2, "run_tests": {"jobs": [j.job_name for j in jobs]}}
    with open(os.path.join(folder, "generated_config.yml"), "w") as f:
        f.write(yaml.dump(config, indent=2, width=1000000, sort_keys=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fetcher_folder", type=str, default=None, help="Only test that all tests and modules are accounted for."
    )
    args = parser.parse_args()

    create_circleci_config(args.fetcher_folder)
