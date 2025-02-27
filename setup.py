# Copyright 2021 The HuggingFace Team. All rights reserved.
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
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/main/setup.py

To create the package for pypi.

1. Create the release branch named: v<RELEASE>-release, for example v4.19-release. For a patch release checkout the
   current release branch.

   If releasing on a special branch, copy the updated README.md on the main branch for the commit you will make
   for the post-release and run `make fix-copies` on the main branch as well.

2. Run `make pre-release` (or `make pre-patch` for a patch release) and commit these changes with the message:
   "Release: <VERSION>" and push.

3. Go back to the main branch and run `make post-release` then `make fix-copies`. Commit these changes with the
   message "v<NEXT_VERSION>.dev.0" and push to main.

# If you were just cutting the branch in preparation for a release, you can stop here for now.

4. Wait for the tests on the release branch to be completed and be green (otherwise revert and fix bugs)

5. On the release branch, add a tag in git to mark the release: "git tag v<VERSION> -m 'Adds tag v<VERSION> for pypi' "
   Push the tag to git: git push --tags origin v<RELEASE>-release

6. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   Run `make build-release`. This will build the release and do some sanity checks for you. If this ends with an error
   message, you need to fix things before going further.

   You should now have a /dist directory with both .whl and .tar.gz source versions.

7. Check that everything looks correct by uploading the package to the pypi test server:

   twine upload dist/* -r testpypi
   (pypi suggest using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r testpypi --repository-url=https://test.pypi.org/legacy/

   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi transformers

   Check you can run the following commands:
   python -c "from transformers import pipeline; classifier = pipeline('text-classification'); print(classifier('What a nice release'))"
   python -c "from transformers import *"
   python utils/check_build.py --check_lib

   If making a patch release, double check the bug you are patching is indeed resolved.

8. Upload the final version to actual pypi:
   twine upload dist/* -r pypi

9. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.
"""

import os
import re

from setuptools import Command, find_packages, setup


# IMPORTANT:
# 1. all dependencies should be listed here with their version requirements if any
# 2. once modified, run: `make deps_table_update` to update src/transformers/dependency_versions_table.py
_deps = [
    "Pillow>=10.0.1,<=15.0",
    "accelerate>=0.26.0",
    "av",
    "beautifulsoup4",
    "blobfile",
    "codecarbon>=2.8.1",
    "cookiecutter==1.7.3",
    "datasets!=2.5.0",
    "deepspeed>=0.9.3",
    "diffusers",
    "evaluate>=0.2.0",
    "faiss-cpu",
    "fastapi",
    "filelock",
    "flax>=0.4.1,<=0.7.0",
    "ftfy",
    "fugashi>=1.0",
    "GitPython",
    "hf-doc-builder>=0.3.0",
    "huggingface-hub>=0.26.0,<1.0",
    "importlib_metadata",
    "ipadic>=1.0.0,<2.0",
    "isort>=5.5.4",
    "jax>=0.4.1,<=0.4.13",
    "jaxlib>=0.4.1,<=0.4.13",
    "jieba",
    "jinja2>=3.1.0",
    "kenlm",
    # Keras pin - this is to make sure Keras 3 doesn't destroy us. Remove or change when we have proper support.
    "keras>2.9,<2.16",
    "keras-nlp>=0.3.1,<0.14.0",  # keras-nlp 0.14 doesn't support keras 2, see pin on keras.
    "libcst",
    "librosa",
    "natten>=0.14.6",
    "nltk<=3.8.1",
    "num2words",
    "numpy>=1.17",
    "onnxruntime>=1.4.0",
    "opencv-python",
    "optimum-benchmark>=0.3.0",
    "optuna",
    "optax>=0.0.8,<=0.1.4",
    "packaging>=20.0",
    "parameterized",
    "phonemizer",
    "protobuf",
    "psutil",
    "pyyaml>=5.1",
    "pydantic",  # TODO (joao) can be removed, only used in the old serving command
    "pytest>=7.2.0",
    "pytest-asyncio",
    "pytest-timeout",
    "pytest-xdist",
    "python>=3.9.0",
    "ray[tune]>=2.7.0",
    "regex!=2019.12.17",
    "requests",
    "rhoknp>=1.1.0,<1.3.1",
    "rich",
    "rjieba",
    "ruff==0.5.1",
    "sacrebleu>=1.4.12,<2.0.0",
    "sacremoses",
    "safetensors>=0.4.1",
    "sagemaker>=2.31.0",
    "schedulefree>=1.2.6",
    "scikit-learn",
    "scipy<1.13.0",  # SciPy >= 1.13.0 is not supported with the current jax pin (`jax>=0.4.1,<=0.4.13`)
    "sentencepiece>=0.1.91,!=0.1.92",
    "sigopt",
    "starlette",
    "sudachipy>=0.6.6",
    "sudachidict_core>=20220729",
    "tensorboard",
    # TensorFlow pin. When changing this value, update examples/tensorflow/_tests_requirements.txt accordingly
    "tensorflow-cpu>2.9,<2.16",
    "tensorflow>2.9,<2.16",
    "tensorflow-text<2.16",
    "tensorflow-probability<0.24",
    "tf2onnx",
    "timeout-decorator",  # TODO (joao) can be removed, redundant with pytest-timeout
    "tiktoken",
    "timm<=1.0.11",
    "tokenizers>=0.21,<0.22",
    "torch>=2.0",
    "torchaudio",
    "torchvision",
    "pyctcdecode>=0.4.0",
    "tqdm>=4.27",
    "unidic>=1.0.2",
    "unidic_lite>=1.0.7",
    "urllib3<2.0.0",
    "uvicorn",
]


# this is a lookup table with items like:
#
# tokenizers: "tokenizers==0.9.4"
# packaging: "packaging"
#
# some of the values are versioned whereas others aren't.
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ ]+)(?:[!=<>~ ].*)?$)", x)[0] for x in _deps)}

# since we save this data in src/transformers/dependency_versions_table.py it can be easily accessed from
# anywhere. If you need to quickly access the data from this table in a shell, you can do so easily with:
#
# python -c 'import sys; from transformers.dependency_versions_table import deps; \
# print(" ".join([ deps[x] for x in sys.argv[1:]]))' tokenizers datasets
#
# Just pass the desired package names to that script as it's shown with 2 packages above.
#
# If transformers is not yet installed and the work is done from the cloned repo remember to add `PYTHONPATH=src` to the script above
#
# You can then feed this for example to `pip`:
#
# pip install -U $(python -c 'import sys; from transformers.dependency_versions_table import deps; \
# print(" ".join([deps[x] for x in sys.argv[1:]]))' tokenizers datasets)
#


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


class DepsTableUpdateCommand(Command):
    """
    A custom distutils command that updates the dependency table.
    usage: python setup.py deps_table_update
    """

    description = "build runtime dependency table"
    user_options = [
        # format: (long option, short option, description).
        ("dep-table-update", None, "updates src/transformers/dependency_versions_table.py"),
    ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        entries = "\n".join([f'    "{k}": "{v}",' for k, v in deps.items()])
        content = [
            "# THIS FILE HAS BEEN AUTOGENERATED. To update:",
            "# 1. modify the `_deps` dict in setup.py",
            "# 2. run `make deps_table_update``",
            "deps = {",
            entries,
            "}",
            "",
        ]
        target = "src/transformers/dependency_versions_table.py"
        print(f"updating {target}")
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(content))


extras = {}

extras["ja"] = deps_list("fugashi", "ipadic", "unidic_lite", "unidic", "sudachipy", "sudachidict_core", "rhoknp")
extras["sklearn"] = deps_list("scikit-learn")

extras["tf"] = deps_list("tensorflow", "tf2onnx", "tensorflow-text", "keras-nlp")
extras["tf-cpu"] = deps_list(
    "keras", "tensorflow-cpu", "tf2onnx", "tensorflow-text", "keras-nlp", "tensorflow-probability"
)

extras["torch"] = deps_list("torch", "accelerate")
extras["accelerate"] = deps_list("accelerate")

if os.name == "nt":  # windows
    extras["retrieval"] = deps_list("datasets")  # faiss is not supported on windows
    extras["flax"] = []  # jax is not supported on windows
else:
    extras["retrieval"] = deps_list("faiss-cpu", "datasets")
    extras["flax"] = deps_list("jax", "jaxlib", "flax", "optax", "scipy")

extras["tokenizers"] = deps_list("tokenizers")
extras["ftfy"] = deps_list("ftfy")
extras["onnxruntime"] = deps_list("onnxruntime")
extras["onnx"] = deps_list("tf2onnx") + extras["onnxruntime"]
extras["modelcreation"] = deps_list("cookiecutter")

extras["sagemaker"] = deps_list("sagemaker")
extras["deepspeed"] = deps_list("deepspeed") + extras["accelerate"]
extras["optuna"] = deps_list("optuna")
extras["ray"] = deps_list("ray[tune]")
extras["sigopt"] = deps_list("sigopt")

extras["integrations"] = extras["optuna"] + extras["ray"] + extras["sigopt"]

extras["serving"] = deps_list("pydantic", "uvicorn", "fastapi", "starlette")
extras["audio"] = deps_list("librosa", "pyctcdecode", "phonemizer", "kenlm")
extras["torch-speech"] = deps_list("torchaudio") + extras["audio"]
extras["tf-speech"] = extras["audio"]
extras["flax-speech"] = extras["audio"]
extras["vision"] = deps_list("Pillow")
extras["timm"] = deps_list("timm")
extras["torch-vision"] = deps_list("torchvision") + extras["vision"]
extras["natten"] = deps_list("natten")
extras["codecarbon"] = deps_list("codecarbon")
extras["video"] = deps_list("av")
extras["num2words"] = deps_list("num2words")
extras["sentencepiece"] = deps_list("sentencepiece", "protobuf")
extras["tiktoken"] = deps_list("tiktoken", "blobfile")
extras["testing"] = (
    deps_list(
        "pytest",
        "pytest-asyncio",
        "pytest-xdist",
        "timeout-decorator",
        "parameterized",
        "psutil",
        "datasets",
        "evaluate",
        "pytest-timeout",
        "ruff",
        "sacrebleu",
        "nltk",
        "GitPython",
        "sacremoses",
        "rjieba",
        "beautifulsoup4",
        "tensorboard",
        "pydantic",
        "sentencepiece",
    )
    + extras["retrieval"]
    + extras["modelcreation"]
    + extras["tiktoken"]
)

extras["deepspeed-testing"] = extras["deepspeed"] + extras["testing"] + extras["optuna"] + extras["sentencepiece"]
extras["ruff"] = deps_list("ruff")
extras["quality"] = deps_list("datasets", "isort", "ruff", "GitPython", "urllib3", "libcst", "rich")

extras["all"] = (
    extras["tf"]
    + extras["torch"]
    + extras["flax"]
    + extras["sentencepiece"]
    + extras["tokenizers"]
    + extras["torch-speech"]
    + extras["vision"]
    + extras["integrations"]
    + extras["timm"]
    + extras["torch-vision"]
    + extras["codecarbon"]
    + extras["accelerate"]
    + extras["video"]
    + extras["num2words"]
)


extras["dev-torch"] = (
    extras["testing"]
    + extras["torch"]
    + extras["sentencepiece"]
    + extras["tokenizers"]
    + extras["torch-speech"]
    + extras["vision"]
    + extras["integrations"]
    + extras["timm"]
    + extras["torch-vision"]
    + extras["codecarbon"]
    + extras["quality"]
    + extras["ja"]
    + extras["sklearn"]
    + extras["modelcreation"]
    + extras["onnxruntime"]
    + extras["num2words"]
)
extras["dev-tensorflow"] = (
    extras["testing"]
    + extras["tf"]
    + extras["sentencepiece"]
    + extras["tokenizers"]
    + extras["vision"]
    + extras["quality"]
    + extras["sklearn"]
    + extras["modelcreation"]
    + extras["onnx"]
    + extras["tf-speech"]
)
extras["dev"] = (
    extras["all"] + extras["testing"] + extras["quality"] + extras["ja"] + extras["sklearn"] + extras["modelcreation"]
)

extras["torchhub"] = deps_list(
    "filelock",
    "huggingface-hub",
    "importlib_metadata",
    "numpy",
    "packaging",
    "protobuf",
    "regex",
    "requests",
    "sentencepiece",
    "torch",
    "tokenizers",
    "tqdm",
)

# NOTE: when we remove `agents` from `transformers`, review entries in `_deps`.
extras["agents"] = deps_list(
    "diffusers", "accelerate", "datasets", "torch", "sentencepiece", "opencv-python", "Pillow"
)

extras["benchmark"] = deps_list("optimum-benchmark")

# when modifying the following list, make sure to update src/transformers/dependency_versions_check.py
install_requires = [
    deps["filelock"],  # filesystem locks, e.g., to prevent parallel downloads
    deps["huggingface-hub"],
    deps["numpy"],
    deps["packaging"],  # utilities from PyPA to e.g., compare versions
    deps["pyyaml"],  # used for the model cards metadata
    deps["regex"],
    deps["requests"],  # for downloading models over HTTPS
    deps["tokenizers"],
    deps["safetensors"],
    deps["tqdm"],  # progress bars in model download and training scripts
]

setup(
    name="transformers",
    version="4.50.0.dev0",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    author="The Hugging Face team (past and future) with the help of all our contributors (https://github.com/huggingface/transformers/graphs/contributors)",
    author_email="transformers@huggingface.co",
    description="State-of-the-art Machine Learning for JAX, PyTorch and TensorFlow",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP vision speech deep learning transformer pytorch tensorflow jax BERT GPT-2 Wav2Vec2 ViT",
    license="Apache 2.0 License",
    url="https://github.com/huggingface/transformers",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    package_data={"": ["**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.pyx"]},
    zip_safe=False,
    extras_require=extras,
    entry_points={"console_scripts": ["transformers-cli=transformers.commands.transformers_cli:main"]},
    python_requires=">=3.9.0",
    install_requires=list(install_requires),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    cmdclass={"deps_table_update": DepsTableUpdateCommand},
)

extras["tests_torch"] = deps_list()
extras["tests_tf"] = deps_list()
extras["tests_flax"] = deps_list()
extras["tests_hub"] = deps_list()
extras["tests_pipelines_torch"] = deps_list()
extras["tests_pipelines_tf"] = deps_list()
extras["tests_onnx"] = deps_list()
extras["tests_examples_torch"] = deps_list()
extras["tests_examples_tf"] = deps_list()
extras["tests_custom_tokenizers"] = deps_list()
extras["tests_exotic_models"] = deps_list()
extras["consistency"] = deps_list()
