# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import os
import shutil
import tempfile
import textwrap
import unittest
from datetime import date

import transformers.commands.add_new_model_like
from transformers.commands.add_new_model_like import ModelInfos, create_new_model_like
from transformers.testing_utils import require_torch


REPO_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_TO_COPY = ("auto", "llama", "phi4_multimodal")
CURRENT_YEAR = date.today().year


@require_torch
class TestAddNewModelLike(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Create a temporary repo with the same structure as Transformers, with just 2 models.
        """
        cls.FAKE_REPO = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(cls.FAKE_REPO, "src", "transformers", "models"), exist_ok=True)
        os.makedirs(os.path.join(cls.FAKE_REPO, "tests", "models"), exist_ok=True)
        os.makedirs(os.path.join(cls.FAKE_REPO, "docs", "source", "en"), exist_ok=True)

        # Copy the __init__ files
        main_init = os.path.join(REPO_PATH, "src", "transformers", "__init__.py")
        shutil.copy(main_init, main_init.replace(REPO_PATH, cls.FAKE_REPO))
        model_init = os.path.join(REPO_PATH, "src", "transformers", "models", "__init__.py")
        shutil.copy(model_init, model_init.replace(REPO_PATH, cls.FAKE_REPO))
        doc_toc = os.path.join(REPO_PATH, "docs", "source", "en", "_toctree.yml")
        shutil.copy(doc_toc, doc_toc.replace(REPO_PATH, cls.FAKE_REPO))
        # Copy over all the specific model files
        for model in MODELS_TO_COPY:
            model_src = os.path.join(REPO_PATH, "src", "transformers", "models", model)
            shutil.copytree(model_src, model_src.replace(REPO_PATH, cls.FAKE_REPO))

            test_src = os.path.join(REPO_PATH, "tests", "models", model)
            shutil.copytree(test_src, test_src.replace(REPO_PATH, cls.FAKE_REPO))

            if model != "auto":
                doc_src = os.path.join(REPO_PATH, "docs", "source", "en", f"{model}.md")
                shutil.copy(doc_src, doc_src.replace(REPO_PATH, cls.FAKE_REPO))

        # Replace the globals
        cls.ORIGINAL_REPO = transformers.commands.add_new_model_like.REPO_PATH
        cls.ORIGINAL_TRANSFORMERS_REPO = transformers.commands.add_new_model_like.TRANSFORMERS_PATH
        transformers.commands.add_new_model_like.REPO_PATH = cls.FAKE_REPO
        transformers.commands.add_new_model_like.TRANSFORMERS_PATH = os.path.join(cls.FAKE_REPO, "src", "transformers")

        # For convenience
        cls.MODEL_PATH = os.path.join(cls.FAKE_REPO, "src", "transformers", "models")
        cls.TESTS_MODEL_PATH = os.path.join(cls.FAKE_REPO, "tests", "models")
        cls.DOC_PATH = os.path.join(cls.FAKE_REPO, "docs", "source", "en")

    @classmethod
    def tearDownClass(cls):
        transformers.commands.add_new_model_like.REPO_PATH = cls.ORIGINAL_REPO
        transformers.commands.add_new_model_like.TRANSFORMERS_PATH = cls.ORIGINAL_TRANSFORMERS_REPO
        del cls.FAKE_REPO

    def assertFileIsEqual(self, text: str, filepath: str):
        with open(filepath, "r") as f:
            file_text = f.read()
        self.assertEqual(file_text.strip(), text.strip())

    def assertInFile(self, text: str, filepath: str):
        with open(filepath, "r") as f:
            file_text = f.read()
        self.assertTrue(text in file_text)

    def test_llama_without_tokenizers(self):
        # This is the structure without adding the tokenizers
        filenames_to_add = (
            ("configuration_llama.py", True),
            ("modeling_llama.py", True),
            ("tokenization_llama.py", False),
            ("tokenization_llama_fast.py", False),
            ("image_processing_llama.py", False),
            ("image_processing_llama_fast.py", False),
            ("video_processing_llama.py", False),
            ("feature_extraction_llama.py", False),
            ("processing_llama.py", False),
        )
        # Run the command
        create_new_model_like(
            old_model_infos=ModelInfos("llama"),
            new_lowercase_name="my_test",
            new_model_paper_name="MyTest",
            filenames_to_add=filenames_to_add,
            create_fast_image_processor=False,
        )

        # First assert that all files were created correctly
        model_repo = os.path.join(self.MODEL_PATH, "my_test")
        tests_repo = os.path.join(self.TESTS_MODEL_PATH, "my_test")
        self.assertTrue(os.path.isfile(os.path.join(model_repo, "modular_my_test.py")))
        self.assertTrue(os.path.isfile(os.path.join(model_repo, "modeling_my_test.py")))
        self.assertTrue(os.path.isfile(os.path.join(model_repo, "configuration_my_test.py")))
        self.assertTrue(os.path.isfile(os.path.join(model_repo, "__init__.py")))
        self.assertTrue(os.path.isfile(os.path.join(self.DOC_PATH, "models", "my_test.md")))
        self.assertTrue(os.path.isfile(os.path.join(tests_repo, "__init__.py")))
        self.assertTrue(os.path.isfile(os.path.join(tests_repo, "test_modeling_my_test.py")))

        # Now assert the correct imports/auto mappings/toctree were added
        self.assertInFile("from .my_test import *\n", os.path.join(self.MODEL_PATH, "__init__.py"))
        self.assertInFile('("my_test", "MyTestConfig"),\n', os.path.join(model_repo, "configuration_my_test.py"))
        self.assertInFile('("my_test", "MyTest"),\n', os.path.join(model_repo, "configuration_my_test.py"))
        self.assertInFile('("my_test", "MyTestModel"),\n', os.path.join(model_repo, "modeling_my_test.py"))
        self.assertInFile('("my_test", "MyTestForCausalLM"),\n', os.path.join(model_repo, "modeling_my_test.py"))
        self.assertInFile(
            '("my_test", "MyTestForSequenceClassification"),\n', os.path.join(model_repo, "modeling_my_test.py")
        )
        self.assertInFile(
            '("my_test", "MyTestForQuestionAnswering"),\n', os.path.join(model_repo, "modeling_my_test.py")
        )
        self.assertInFile(
            '("my_test", "MyTestForTokenClassification"),\n', os.path.join(model_repo, "modeling_my_test.py")
        )
        self.assertInFile(
            "- local: model_doc/my_test\n        title: MyTest\n", os.path.join(self.DOC_PATH, "_toctree.yml")
        )

        # Check some exact file creation. For model definition, only check modular as modeling/config/etc... are created
        # directly from it
        EXPECTED_MODULAR = textwrap.dedent(
            f"""
            # coding=utf-8
            # Copyright {CURRENT_YEAR} the HuggingFace Team. All rights reserved.
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

            from ..llama.configuration_llama import LlamaConfig
            from ..llama.modeling_llama import (
                LlamaAttention,
                LlamaDecoderLayer,
                LlamaForCausalLM,
                LlamaForQuestionAnswering,
                LlamaForSequenceClassification,
                LlamaForTokenClassification,
                LlamaMLP,
                LlamaModel,
                LlamaPreTrainedModel,
                LlamaRMSNorm,
                LlamaRotaryEmbedding,
            )


            class MyTestConfig(LlamaConfig):
                pass


            class MyTestRMSNorm(LlamaRMSNorm):
                pass


            class MyTestRotaryEmbedding(LlamaRotaryEmbedding):
                pass


            class MyTestMLP(LlamaMLP):
                pass


            class MyTestAttention(LlamaAttention):
                pass


            class MyTestDecoderLayer(LlamaDecoderLayer):
                pass


            class MyTestPreTrainedModel(LlamaPreTrainedModel):
                pass


            class MyTestModel(LlamaModel):
                pass


            class MyTestForCausalLM(LlamaForCausalLM):
                pass


            class MyTestForSequenceClassification(LlamaForSequenceClassification):
                pass


            class MyTestForQuestionAnswering(LlamaForQuestionAnswering):
                pass


            class MyTestForTokenClassification(LlamaForTokenClassification):
                pass


            __all__ = [
                "MyTestConfig",
                "MyTestForCausalLM",
                "MyTestModel",
                "MyTestPreTrainedModel",
                "MyTestForSequenceClassification",
                "MyTestForQuestionAnswering",
                "MyTestForTokenClassification",
            ]
            """
        )
        self.assertFileIsEqual(os.path.join(model_repo, "modular_my_test.py"), EXPECTED_MODULAR)

        EXPECTED_INIT = textwrap.dedent(
            f"""
            # coding=utf-8
            # Copyright {CURRENT_YEAR} the HuggingFace Team. All rights reserved.
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

            from typing import TYPE_CHECKING

            from ...utils import _LazyModule
            from ...utils.import_utils import define_import_structure


            if TYPE_CHECKING:
                from .configuration_my_test import *
                from .modeling_my_test import *
            else:
                import sys

                _file = globals()["__file__"]
                sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)

            """
        )
        self.assertFileIsEqual(os.path.join(model_repo, "__init__.py"), EXPECTED_INIT)

        EXPECTED_DOC = textwrap.dedent(
            f"""
            <!--Copyright {CURRENT_YEAR} the HuggingFace Team. All rights reserved.
            #
            Licensed under the Apache License, Version 2.0 (the "License");
            you may not use this file except in compliance with the License.
            You may obtain a copy of the License at
            #
                http://www.apache.org/licenses/LICENSE-2.0
            #
            Unless required by applicable law or agreed to in writing, software
            distributed under the License is distributed on an "AS IS" BASIS,
            WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
            See the License for the specific language governing permissions and
            limitations under the License.


            ⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

            -->


            # MyTest

            ## Overview

            The MyTest model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>) by <INSERT AUTHORS HERE>.
            <INSERT SHORT SUMMARY HERE>

            The abstract from the paper is the following:

            <INSERT PAPER ABSTRACT HERE>

            Tips:

            <INSERT TIPS ABOUT MODEL HERE>

            This model was contributed by [INSERT YOUR HF USERNAME HERE](https://huggingface.co/<INSERT YOUR HF USERNAME HERE>).
            The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).

            ## Usage examples

            <INSERT SOME NICE EXAMPLES HERE>

            ## MyTestConfig

            [[autodoc]] MyTestConfig

            ## MyTestForCausalLM

            [[autodoc]] MyTestForCausalLM

            ## MyTestModel

            [[autodoc]] MyTestModel
                - forward

            ## MyTestPreTrainedModel

            [[autodoc]] MyTestPreTrainedModel
                - forward

            ## MyTestForSequenceClassification

            [[autodoc]] MyTestForSequenceClassification

            ## MyTestForQuestionAnswering

            [[autodoc]] MyTestForQuestionAnswering

            ## MyTestForTokenClassification

            [[autodoc]] MyTestForTokenClassification
            """
        )
        self.assertFileIsEqual(os.path.join(self.DOC_PATH, "models", "my_test.md"), EXPECTED_DOC)
