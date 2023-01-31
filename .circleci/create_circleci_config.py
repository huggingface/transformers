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


COMMON_ENV_VARIABLES = {"OMP_NUM_THREADS": 1, "TRANSFORMERS_IS_CI": True, "PYTEST_TIMEOUT": 120}
COMMON_PYTEST_OPTIONS = {"max-worker-restart": 0, "dist": "loadfile", "s": None}
DEFAULT_DOCKER_IMAGE = [{"image": "cimg/python:3.7.12"}]


@dataclass
class CircleCIJob:
    name: str
    additional_env: Dict[str, Any] = None
    cache_name: str = None
    cache_version: str = "0.5"
    docker_image: List[Dict[str, str]] = None
    install_steps: List[str] = None
    marker: Optional[str] = None
    parallelism: Optional[int] = 1
    pytest_num_workers: int = 8
    pytest_options: Dict[str, Any] = None
    resource_class: Optional[str] = "xlarge"
    tests_to_run: Optional[List[str]] = None
    working_directory: str = "~/transformers"

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
        job = {
            "working_directory": self.working_directory,
            "docker": self.docker_image,
            "environment": {**COMMON_ENV_VARIABLES, **self.additional_env},
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
                        f"v{self.cache_version}-{self.cache_name}-" + '{{ checksum "setup.py" }}',
                        f"v{self.cache_version}-{self.cache_name}-",
                    ]
                }
            },
        ]
        steps.extend([{"run": l} for l in self.install_steps])
        steps.append(
            {
                "save_cache": {
                    "key": f"v{self.cache_version}-{self.cache_name}-" + '{{ checksum "setup.py" }}',
                    "paths": ["~/.cache/pip"],
                }
            }
        )
        steps.append({"run": {"name": "Show installed libraries and their versions", "command": "pip freeze | tee installed.txt"}})
        steps.append({"store_artifacts": {"path": "~/transformers/installed.txt"}})

        all_options = {**COMMON_PYTEST_OPTIONS, **self.pytest_options}
        pytest_flags = [f"--{key}={value}" if value is not None else f"-{key}" for key, value in all_options.items()]
        pytest_flags.append(
            f"--make-reports={self.name}" if "examples" in self.name else f"--make-reports=tests_{self.name}"
        )
        test_command = f"python -m pytest -n {self.pytest_num_workers} " + " ".join(pytest_flags)
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

            test_command = f"python -m pytest -n {self.pytest_num_workers} " + " ".join(pytest_flags)
            test_command += " $(cat splitted_tests.txt)"
        if self.marker is not None:
            test_command += f" -m {self.marker}"
        test_command += " | tee tests_output.txt"
        steps.append({"run": {"name": "Run tests", "command": test_command}})
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
        "sudo apt-get -y update && sudo apt-get install -y libsndfile1-dev espeak-ng git-lfs",
        "git lfs install",
        "pip install --upgrade pip",
        "pip install .[sklearn,tf-cpu,torch,testing,sentencepiece,torch-speech,vision]",
        "pip install tensorflow_probability",
        "pip install git+https://github.com/huggingface/accelerate",
    ],
    marker="is_pt_tf_cross_test",
    pytest_options={"rA": None, "durations": 0},
)


torch_and_flax_job = CircleCIJob(
    "torch_and_flax",
    additional_env={"RUN_PT_FLAX_CROSS_TESTS": True},
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install -y libsndfile1-dev espeak-ng",
        "pip install --upgrade pip",
        "pip install .[sklearn,flax,torch,testing,sentencepiece,torch-speech,vision]",
        "pip install git+https://github.com/huggingface/accelerate",
    ],
    marker="is_pt_flax_cross_test",
    pytest_options={"rA": None, "durations": 0},
)


torch_job = CircleCIJob(
    "torch",
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install -y libsndfile1-dev espeak-ng time",
        "pip install --upgrade pip",
        "pip install .[sklearn,torch,testing,sentencepiece,torch-speech,vision,timm]",
        "pip install git+https://github.com/huggingface/accelerate",
    ],
    parallelism=1,
    pytest_num_workers=3,
)


tf_job = CircleCIJob(
    "tf",
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install -y libsndfile1-dev espeak-ng",
        "pip install --upgrade pip",
        "pip install .[sklearn,tf-cpu,testing,sentencepiece,tf-speech,vision]",
        "pip install tensorflow_probability",
    ],
    parallelism=1,
    pytest_options={"rA": None},
)


flax_job = CircleCIJob(
    "flax",
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install -y libsndfile1-dev espeak-ng",
        "pip install --upgrade pip",
        "pip install .[flax,testing,sentencepiece,flax-speech,vision]",
    ],
    parallelism=1,
    pytest_options={"rA": None},
)


pipelines_torch_job = CircleCIJob(
    "pipelines_torch",
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install -y libsndfile1-dev espeak-ng",
        "pip install --upgrade pip",
        "pip install .[sklearn,torch,testing,sentencepiece,torch-speech,vision,timm,video]",
    ],
    pytest_options={"rA": None},
    tests_to_run="tests/pipelines/"
)


pipelines_tf_job = CircleCIJob(
    "pipelines_tf",
    install_steps=[
        "pip install --upgrade pip",
        "pip install .[sklearn,tf-cpu,testing,sentencepiece,vision]",
        "pip install tensorflow_probability",
    ],
    pytest_options={"rA": None},
    tests_to_run="tests/pipelines/"
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
        "pip install --upgrade pip",
        "pip install .[ja,testing,sentencepiece,jieba,spacy,ftfy,rjieba]",
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
        "pip install --upgrade pip",
        "pip install .[sklearn,torch,sentencepiece,testing,torch-speech]",
        "pip install -r examples/pytorch/_tests_requirements.txt",
    ],
    tests_to_run="./examples/pytorch/",
)


examples_tensorflow_job = CircleCIJob(
    "examples_tensorflow",
    cache_name="tensorflow_examples",
    install_steps=[
        "pip install --upgrade pip",
        "pip install .[sklearn,tensorflow,sentencepiece,testing]",
        "pip install -r examples/tensorflow/_tests_requirements.txt",
    ],
    tests_to_run="./examples/tensorflow/",
)


examples_flax_job = CircleCIJob(
    "examples_flax",
    cache_name="flax_examples",
    install_steps=[
        "pip install --upgrade pip",
        "pip install .[flax,testing,sentencepiece]",
        "pip install -r examples/flax/_tests_requirements.txt",
    ],
    tests_to_run="./examples/flax/",
)


hub_job = CircleCIJob(
    "hub",
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install git-lfs",
        'git config --global user.email "ci@dummy.com"',
        'git config --global user.name "ci"',
        "pip install --upgrade pip",
        "pip install .[torch,sentencepiece,testing]",
    ],
    marker="is_staging_test",
    pytest_num_workers=1,
)


onnx_job = CircleCIJob(
    "onnx",
    install_steps=[
        "pip install --upgrade pip",
        "pip install .[torch,tf,testing,sentencepiece,onnxruntime,vision,rjieba]",
    ],
    pytest_options={"k onnx": None},
    pytest_num_workers=1,
)


exotic_models_job = CircleCIJob(
    "exotic_models",
    install_steps=[
        "sudo apt-get -y update && sudo apt-get install -y libsndfile1-dev",
        "pip install --upgrade pip",
        "pip install .[torch,testing,vision]",
        "pip install torchvision",
        "pip install scipy",
        "pip install 'git+https://github.com/facebookresearch/detectron2.git'",
        "sudo apt install tesseract-ocr",
        "pip install pytesseract",
        "pip install natten",
    ],
    tests_to_run=[
        "tests/models/*layoutlmv*",
        "tests/models/*nat",
        "tests/models/deta",
    ],
    pytest_num_workers=1,
    pytest_options={"durations": 100},
)


repo_utils_job = CircleCIJob(
    "repo_utils",
    install_steps=[
        "pip install --upgrade pip",
        "pip install .[quality,testing]",
    ],
    parallelism=None,
    pytest_num_workers=1,
    resource_class=None,
    tests_to_run="tests/repo_utils",
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

    example_file = os.path.join(folder, "examples_test_list.txt")
    if os.path.exists(example_file) and os.path.getsize(example_file) > 0:
        jobs.extend(EXAMPLES_TESTS)

    repo_util_file = os.path.join(folder, "test_repo_utils.txt")
    if os.path.exists(repo_util_file) and os.path.getsize(repo_util_file) > 0:
        jobs.extend(REPO_UTIL_TESTS)

    if len(jobs) > 0:
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
