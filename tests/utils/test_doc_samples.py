# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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
import doctest
import logging
import os
import unittest
from glob import glob
from pathlib import Path
from typing import List, Union

import transformers
from transformers.testing_utils import require_tf, require_torch, slow


logger = logging.getLogger()


@require_torch
class TestDocLists(unittest.TestCase):
    def test_flash_support_list(self):
        with open("./docs/source/en/perf_infer_gpu_one.md", "r") as f:
            doctext = f.read()

            doctext = doctext.split("FlashAttention-2 is currently supported for the following architectures:")[1]
            doctext = doctext.split("You can request to add FlashAttention-2 support")[0]

        patterns = glob("./src/transformers/models/**/modeling_*.py")
        patterns_tf = glob("./src/transformers/models/**/modeling_tf_*.py")
        patterns_flax = glob("./src/transformers/models/**/modeling_flax_*.py")
        patterns = list(set(patterns) - set(patterns_tf) - set(patterns_flax))
        archs_supporting_fa2 = []
        for filename in patterns:
            with open(filename, "r") as f:
                text = f.read()

                if "_supports_flash_attn_2 = True" in text:
                    model_name = os.path.basename(filename).replace(".py", "").replace("modeling_", "")
                    archs_supporting_fa2.append(model_name)

        for arch in archs_supporting_fa2:
            if arch not in doctext:
                raise ValueError(
                    f"{arch} should be in listed in the flash attention documentation but is not. Please update the documentation."
                )

    def test_sdpa_support_list(self):
        with open("./docs/source/en/perf_infer_gpu_one.md", "r") as f:
            doctext = f.read()

            doctext = doctext.split(
                "For now, Transformers supports inference and training through SDPA for the following architectures:"
            )[1]
            doctext = doctext.split("Note that FlashAttention can only be used for models using the")[0]

        patterns = glob("./src/transformers/models/**/modeling_*.py")
        patterns_tf = glob("./src/transformers/models/**/modeling_tf_*.py")
        patterns_flax = glob("./src/transformers/models/**/modeling_flax_*.py")
        patterns = list(set(patterns) - set(patterns_tf) - set(patterns_flax))
        archs_supporting_sdpa = []
        for filename in patterns:
            with open(filename, "r") as f:
                text = f.read()

                if "_supports_sdpa = True" in text:
                    model_name = os.path.basename(filename).replace(".py", "").replace("modeling_", "")
                    archs_supporting_sdpa.append(model_name)

        for arch in archs_supporting_sdpa:
            if arch not in doctext:
                raise ValueError(
                    f"{arch} should be in listed in the SDPA documentation but is not. Please update the documentation."
                )


@unittest.skip("Temporarily disable the doc tests.")
@require_torch
@require_tf
@slow
class TestCodeExamples(unittest.TestCase):
    def analyze_directory(
        self,
        directory: Path,
        identifier: Union[str, None] = None,
        ignore_files: Union[List[str], None] = None,
        n_identifier: Union[str, List[str], None] = None,
        only_modules: bool = True,
    ):
        """
        Runs through the specific directory, looking for the files identified with `identifier`. Executes
        the doctests in those files

        Args:
            directory (`Path`): Directory containing the files
            identifier (`str`): Will parse files containing this
            ignore_files (`List[str]`): List of files to skip
            n_identifier (`str` or `List[str]`): Will not parse files containing this/these identifiers.
            only_modules (`bool`): Whether to only analyze modules
        """
        files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

        if identifier is not None:
            files = [file for file in files if identifier in file]

        if n_identifier is not None:
            if isinstance(n_identifier, List):
                for n_ in n_identifier:
                    files = [file for file in files if n_ not in file]
            else:
                files = [file for file in files if n_identifier not in file]

        ignore_files = ignore_files or []
        ignore_files.append("__init__.py")
        files = [file for file in files if file not in ignore_files]

        for file in files:
            # Open all files
            print("Testing", file)

            if only_modules:
                module_identifier = file.split(".")[0]
                try:
                    module_identifier = getattr(transformers, module_identifier)
                    suite = doctest.DocTestSuite(module_identifier)
                    result = unittest.TextTestRunner().run(suite)
                    self.assertIs(len(result.failures), 0)
                except AttributeError:
                    logger.info(f"{module_identifier} is not a module.")
            else:
                result = doctest.testfile(str(".." / directory / file), optionflags=doctest.ELLIPSIS)
                self.assertIs(result.failed, 0)

    def test_modeling_examples(self):
        transformers_directory = Path("src/transformers")
        files = "modeling"
        ignore_files = [
            "modeling_ctrl.py",
            "modeling_tf_ctrl.py",
        ]
        self.analyze_directory(transformers_directory, identifier=files, ignore_files=ignore_files)

    def test_tokenization_examples(self):
        transformers_directory = Path("src/transformers")
        files = "tokenization"
        self.analyze_directory(transformers_directory, identifier=files)

    def test_configuration_examples(self):
        transformers_directory = Path("src/transformers")
        files = "configuration"
        self.analyze_directory(transformers_directory, identifier=files)

    def test_remaining_examples(self):
        transformers_directory = Path("src/transformers")
        n_identifiers = ["configuration", "modeling", "tokenization"]
        self.analyze_directory(transformers_directory, n_identifier=n_identifiers)

    def test_doc_sources(self):
        doc_source_directory = Path("docs/source")
        ignore_files = ["favicon.ico"]
        self.analyze_directory(doc_source_directory, ignore_files=ignore_files, only_modules=False)
