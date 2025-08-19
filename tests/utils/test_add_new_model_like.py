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
from pathlib import Path

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
        cls.FAKE_REPO = tempfile.TemporaryDirectory().name
        os.makedirs(os.path.join(cls.FAKE_REPO, "src", "transformers", "models"), exist_ok=True)
        os.makedirs(os.path.join(cls.FAKE_REPO, "tests", "models"), exist_ok=True)
        os.makedirs(os.path.join(cls.FAKE_REPO, "docs", "source", "en", "model_doc"), exist_ok=True)

        # We need to copy the utils to run the cleanup commands
        utils_src = os.path.join(REPO_PATH, "utils")
        shutil.copytree(utils_src, utils_src.replace(REPO_PATH, cls.FAKE_REPO))
        # Copy the __init__ files
        model_init = os.path.join(REPO_PATH, "src", "transformers", "models", "__init__.py")
        shutil.copy(model_init, model_init.replace(REPO_PATH, cls.FAKE_REPO))
        doc_toc = os.path.join(REPO_PATH, "docs", "source", "en", "_toctree.yml")
        shutil.copy(doc_toc, doc_toc.replace(REPO_PATH, cls.FAKE_REPO))
        # We need the pyproject for ruff as well
        pyproject = os.path.join(REPO_PATH, "pyproject.toml")
        shutil.copy(pyproject, pyproject.replace(REPO_PATH, cls.FAKE_REPO))
        # Copy over all the specific model files
        for model in MODELS_TO_COPY:
            model_src = os.path.join(REPO_PATH, "src", "transformers", "models", model)
            shutil.copytree(model_src, model_src.replace(REPO_PATH, cls.FAKE_REPO))

            test_src = os.path.join(REPO_PATH, "tests", "models", model)
            shutil.copytree(test_src, test_src.replace(REPO_PATH, cls.FAKE_REPO))

            if model != "auto":
                doc_src = os.path.join(REPO_PATH, "docs", "source", "en", "model_doc", f"{model}.md")
                shutil.copy(doc_src, doc_src.replace(REPO_PATH, cls.FAKE_REPO))

        # Replace the globals
        cls.ORIGINAL_REPO = transformers.commands.add_new_model_like.REPO_PATH
        cls.ORIGINAL_TRANSFORMERS_REPO = transformers.commands.add_new_model_like.TRANSFORMERS_PATH
        transformers.commands.add_new_model_like.REPO_PATH = Path(cls.FAKE_REPO)
        transformers.commands.add_new_model_like.TRANSFORMERS_PATH = Path(cls.FAKE_REPO) / "src" / "transformers"

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
        self.assertTrue(os.path.isfile(os.path.join(self.DOC_PATH, "model_doc", "my_test.md")))
        self.assertTrue(os.path.isfile(os.path.join(tests_repo, "__init__.py")))
        self.assertTrue(os.path.isfile(os.path.join(tests_repo, "test_modeling_my_test.py")))

        # Now assert the correct imports/auto mappings/toctree were added
        self.assertInFile(
            "from .my_test import *\n",
            os.path.join(self.MODEL_PATH, "__init__.py"),
        )
        self.assertInFile(
            '("my_test", "MyTestConfig"),\n',
            os.path.join(self.MODEL_PATH, "auto", "configuration_auto.py"),
        )
        self.assertInFile(
            '("my_test", "MyTest"),\n',
            os.path.join(self.MODEL_PATH, "auto", "configuration_auto.py"),
        )
        self.assertInFile(
            '("my_test", "MyTestModel"),\n',
            os.path.join(self.MODEL_PATH, "auto", "modeling_auto.py"),
        )
        self.assertInFile(
            '("my_test", "MyTestForCausalLM"),\n',
            os.path.join(self.MODEL_PATH, "auto", "modeling_auto.py"),
        )
        self.assertInFile(
            '("my_test", "MyTestForSequenceClassification"),\n',
            os.path.join(self.MODEL_PATH, "auto", "modeling_auto.py"),
        )
        self.assertInFile(
            '("my_test", "MyTestForQuestionAnswering"),\n',
            os.path.join(self.MODEL_PATH, "auto", "modeling_auto.py"),
        )
        self.assertInFile(
            '("my_test", "MyTestForTokenClassification"),\n',
            os.path.join(self.MODEL_PATH, "auto", "modeling_auto.py"),
        )
        self.assertInFile(
            "- local: model_doc/my_test\n        title: MyTest\n",
            os.path.join(self.DOC_PATH, "_toctree.yml"),
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
        self.assertFileIsEqual(EXPECTED_MODULAR, os.path.join(model_repo, "modular_my_test.py"))

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
        self.assertFileIsEqual(EXPECTED_INIT, os.path.join(model_repo, "__init__.py"))

        EXPECTED_DOC = textwrap.dedent(
            f"""
            <!--Copyright {CURRENT_YEAR} the HuggingFace Team. All rights reserved.

            Licensed under the Apache License, Version 2.0 (the "License");
            you may not use this file except in compliance with the License.
            You may obtain a copy of the License at

                http://www.apache.org/licenses/LICENSE-2.0

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
        self.assertFileIsEqual(EXPECTED_DOC, os.path.join(self.DOC_PATH, "model_doc", "my_test.md"))

    def test_phi4_with_all_processors(self):
        # This is the structure without adding the tokenizers
        filenames_to_add = (
            ("configuration_phi4_multimodal.py", True),
            ("modeling_phi4_multimodal.py", True),
            ("tokenization_phi4_multimodal.py", False),
            ("tokenization_phi4_multimodal_fast.py", False),
            ("image_processing_phi4_multimodal.py", False),
            ("image_processing_phi4_multimodal_fast.py", True),
            ("video_processing_phi4_multimodal.py", False),
            ("feature_extraction_phi4_multimodal.py", True),
            ("processing_phi4_multimodal.py", True),
        )
        # Run the command
        create_new_model_like(
            old_model_infos=ModelInfos("phi4_multimodal"),
            new_lowercase_name="my_test2",
            new_model_paper_name="MyTest2",
            filenames_to_add=filenames_to_add,
            create_fast_image_processor=False,
        )

        # First assert that all files were created correctly
        model_repo = os.path.join(self.MODEL_PATH, "my_test2")
        tests_repo = os.path.join(self.TESTS_MODEL_PATH, "my_test2")
        self.assertTrue(os.path.isfile(os.path.join(model_repo, "modular_my_test2.py")))
        self.assertTrue(os.path.isfile(os.path.join(model_repo, "modeling_my_test2.py")))
        self.assertTrue(os.path.isfile(os.path.join(model_repo, "configuration_my_test2.py")))
        self.assertTrue(os.path.isfile(os.path.join(model_repo, "image_processing_my_test2_fast.py")))
        self.assertTrue(os.path.isfile(os.path.join(model_repo, "feature_extraction_my_test2.py")))
        self.assertTrue(os.path.isfile(os.path.join(model_repo, "processing_my_test2.py")))
        self.assertTrue(os.path.isfile(os.path.join(model_repo, "__init__.py")))
        self.assertTrue(os.path.isfile(os.path.join(self.DOC_PATH, "model_doc", "my_test2.md")))
        self.assertTrue(os.path.isfile(os.path.join(tests_repo, "__init__.py")))
        self.assertTrue(os.path.isfile(os.path.join(tests_repo, "test_modeling_my_test2.py")))
        self.assertTrue(os.path.isfile(os.path.join(tests_repo, "test_feature_extraction_my_test2.py")))
        self.assertTrue(os.path.isfile(os.path.join(tests_repo, "test_image_processing_my_test2.py")))

        # Now assert the correct imports/auto mappings/toctree were added
        self.assertInFile(
            "from .my_test2 import *\n",
            os.path.join(self.MODEL_PATH, "__init__.py"),
        )
        self.assertInFile(
            '("my_test2", "MyTest2Config"),\n',
            os.path.join(self.MODEL_PATH, "auto", "configuration_auto.py"),
        )
        self.assertInFile(
            '("my_test2", "MyTest2"),\n',
            os.path.join(self.MODEL_PATH, "auto", "configuration_auto.py"),
        )
        self.assertInFile(
            '("my_test2", "MyTest2Model"),\n',
            os.path.join(self.MODEL_PATH, "auto", "modeling_auto.py"),
        )
        self.assertInFile(
            '("my_test2", "MyTest2ForCausalLM"),\n',
            os.path.join(self.MODEL_PATH, "auto", "modeling_auto.py"),
        )
        self.assertInFile(
            '("my_test2", (None, "MyTest2ImageProcessorFast")),\n',
            os.path.join(self.MODEL_PATH, "auto", "image_processing_auto.py"),
        )
        self.assertInFile(
            '("my_test2", "MyTest2FeatureExtractor"),\n',
            os.path.join(self.MODEL_PATH, "auto", "feature_extraction_auto.py"),
        )
        self.assertInFile(
            '("my_test2", "MyTest2Processor"),\n',
            os.path.join(self.MODEL_PATH, "auto", "processing_auto.py"),
        )
        self.assertInFile(
            "- local: model_doc/my_test2\n        title: MyTest2\n",
            os.path.join(self.DOC_PATH, "_toctree.yml"),
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

            from ..phi4_multimodal.configuration_phi4_multimodal import (
                Phi4MultimodalAudioConfig,
                Phi4MultimodalConfig,
                Phi4MultimodalVisionConfig,
            )
            from ..phi4_multimodal.feature_extraction_phi4_multimodal import Phi4MultimodalFeatureExtractor
            from ..phi4_multimodal.image_processing_phi4_multimodal_fast import (
                Phi4MultimodalFastImageProcessorKwargs,
                Phi4MultimodalImageProcessorFast,
            )
            from ..phi4_multimodal.modeling_phi4_multimodal import (
                Phi4MultimodalAttention,
                Phi4MultimodalAudioAttention,
                Phi4MultimodalAudioConformerEncoderLayer,
                Phi4MultimodalAudioConvModule,
                Phi4MultimodalAudioDepthWiseSeperableConv1d,
                Phi4MultimodalAudioEmbedding,
                Phi4MultimodalAudioGluPointWiseConv,
                Phi4MultimodalAudioMeanVarianceNormLayer,
                Phi4MultimodalAudioMLP,
                Phi4MultimodalAudioModel,
                Phi4MultimodalAudioNemoConvSubsampling,
                Phi4MultimodalAudioPreTrainedModel,
                Phi4MultimodalAudioRelativeAttentionBias,
                Phi4MultimodalDecoderLayer,
                Phi4MultimodalFeatureEmbedding,
                Phi4MultimodalForCausalLM,
                Phi4MultimodalImageEmbedding,
                Phi4MultimodalMLP,
                Phi4MultimodalModel,
                Phi4MultimodalPreTrainedModel,
                Phi4MultimodalRMSNorm,
                Phi4MultimodalRotaryEmbedding,
                Phi4MultimodalVisionAttention,
                Phi4MultimodalVisionEmbeddings,
                Phi4MultimodalVisionEncoder,
                Phi4MultimodalVisionEncoderLayer,
                Phi4MultimodalVisionMLP,
                Phi4MultimodalVisionModel,
                Phi4MultimodalVisionMultiheadAttentionPoolingHead,
                Phi4MultimodalVisionPreTrainedModel,
            )
            from ..phi4_multimodal.processing_phi4_multimodal import Phi4MultimodalProcessor, Phi4MultimodalProcessorKwargs


            class MyTest2VisionConfig(Phi4MultimodalVisionConfig):
                pass


            class MyTest2AudioConfig(Phi4MultimodalAudioConfig):
                pass


            class MyTest2Config(Phi4MultimodalConfig):
                pass


            class MyTest2VisionMLP(Phi4MultimodalVisionMLP):
                pass


            class MyTest2VisionAttention(Phi4MultimodalVisionAttention):
                pass


            class MyTest2VisionEncoderLayer(Phi4MultimodalVisionEncoderLayer):
                pass


            class MyTest2VisionEncoder(Phi4MultimodalVisionEncoder):
                pass


            class MyTest2VisionPreTrainedModel(Phi4MultimodalVisionPreTrainedModel):
                pass


            class MyTest2VisionEmbeddings(Phi4MultimodalVisionEmbeddings):
                pass


            class MyTest2VisionMultiheadAttentionPoolingHead(Phi4MultimodalVisionMultiheadAttentionPoolingHead):
                pass


            class MyTest2VisionModel(Phi4MultimodalVisionModel):
                pass


            class MyTest2ImageEmbedding(Phi4MultimodalImageEmbedding):
                pass


            class MyTest2AudioMLP(Phi4MultimodalAudioMLP):
                pass


            class MyTest2AudioAttention(Phi4MultimodalAudioAttention):
                pass


            class MyTest2AudioDepthWiseSeperableConv1d(Phi4MultimodalAudioDepthWiseSeperableConv1d):
                pass


            class MyTest2AudioGluPointWiseConv(Phi4MultimodalAudioGluPointWiseConv):
                pass


            class MyTest2AudioConvModule(Phi4MultimodalAudioConvModule):
                pass


            class MyTest2AudioConformerEncoderLayer(Phi4MultimodalAudioConformerEncoderLayer):
                pass


            class MyTest2AudioNemoConvSubsampling(Phi4MultimodalAudioNemoConvSubsampling):
                pass


            class MyTest2AudioRelativeAttentionBias(Phi4MultimodalAudioRelativeAttentionBias):
                pass


            class MyTest2AudioMeanVarianceNormLayer(Phi4MultimodalAudioMeanVarianceNormLayer):
                pass


            class MyTest2AudioPreTrainedModel(Phi4MultimodalAudioPreTrainedModel):
                pass


            class MyTest2AudioModel(Phi4MultimodalAudioModel):
                pass


            class MyTest2AudioEmbedding(Phi4MultimodalAudioEmbedding):
                pass


            class MyTest2RMSNorm(Phi4MultimodalRMSNorm):
                pass


            class MyTest2MLP(Phi4MultimodalMLP):
                pass


            class MyTest2Attention(Phi4MultimodalAttention):
                pass


            class MyTest2DecoderLayer(Phi4MultimodalDecoderLayer):
                pass


            class MyTest2FeatureEmbedding(Phi4MultimodalFeatureEmbedding):
                pass


            class MyTest2RotaryEmbedding(Phi4MultimodalRotaryEmbedding):
                pass


            class MyTest2PreTrainedModel(Phi4MultimodalPreTrainedModel):
                pass


            class MyTest2Model(Phi4MultimodalModel):
                pass


            class MyTest2ForCausalLM(Phi4MultimodalForCausalLM):
                pass


            class MyTest2FastImageProcessorKwargs(Phi4MultimodalFastImageProcessorKwargs):
                pass


            class MyTest2ImageProcessorFast(Phi4MultimodalImageProcessorFast):
                pass


            class MyTest2FeatureExtractor(Phi4MultimodalFeatureExtractor):
                pass


            class MyTest2ProcessorKwargs(Phi4MultimodalProcessorKwargs):
                pass


            class MyTest2Processor(Phi4MultimodalProcessor):
                pass


            __all__ = [
                "MyTest2VisionConfig",
                "MyTest2AudioConfig",
                "MyTest2Config",
                "MyTest2AudioPreTrainedModel",
                "MyTest2AudioModel",
                "MyTest2VisionPreTrainedModel",
                "MyTest2VisionModel",
                "MyTest2PreTrainedModel",
                "MyTest2Model",
                "MyTest2ForCausalLM",
                "MyTest2ImageProcessorFast",
                "MyTest2FeatureExtractor",
                "MyTest2Processor",
            ]
            """
        )
        self.assertFileIsEqual(EXPECTED_MODULAR, os.path.join(model_repo, "modular_my_test2.py"))

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
                from .configuration_my_test2 import *
                from .feature_extraction_my_test2 import *
                from .image_processing_my_test2_fast import *
                from .modeling_my_test2 import *
                from .processing_my_test2 import *
            else:
                import sys

                _file = globals()["__file__"]
                sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
            """
        )
        self.assertFileIsEqual(EXPECTED_INIT, os.path.join(model_repo, "__init__.py"))

        EXPECTED_DOC = textwrap.dedent(
            f"""
            <!--Copyright {CURRENT_YEAR} the HuggingFace Team. All rights reserved.

            Licensed under the Apache License, Version 2.0 (the "License");
            you may not use this file except in compliance with the License.
            You may obtain a copy of the License at

                http://www.apache.org/licenses/LICENSE-2.0

            Unless required by applicable law or agreed to in writing, software
            distributed under the License is distributed on an "AS IS" BASIS,
            WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
            See the License for the specific language governing permissions and
            limitations under the License.


            ⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

            -->


            # MyTest2

            ## Overview

            The MyTest2 model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>) by <INSERT AUTHORS HERE>.
            <INSERT SHORT SUMMARY HERE>

            The abstract from the paper is the following:

            <INSERT PAPER ABSTRACT HERE>

            Tips:

            <INSERT TIPS ABOUT MODEL HERE>

            This model was contributed by [INSERT YOUR HF USERNAME HERE](https://huggingface.co/<INSERT YOUR HF USERNAME HERE>).
            The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).

            ## Usage examples

            <INSERT SOME NICE EXAMPLES HERE>

            ## MyTest2VisionConfig

            [[autodoc]] MyTest2VisionConfig

            ## MyTest2AudioConfig

            [[autodoc]] MyTest2AudioConfig

            ## MyTest2Config

            [[autodoc]] MyTest2Config

            ## MyTest2AudioPreTrainedModel

            [[autodoc]] MyTest2AudioPreTrainedModel
                - forward

            ## MyTest2AudioModel

            [[autodoc]] MyTest2AudioModel
                - forward

            ## MyTest2VisionPreTrainedModel

            [[autodoc]] MyTest2VisionPreTrainedModel
                - forward

            ## MyTest2VisionModel

            [[autodoc]] MyTest2VisionModel
                - forward

            ## MyTest2PreTrainedModel

            [[autodoc]] MyTest2PreTrainedModel
                - forward

            ## MyTest2Model

            [[autodoc]] MyTest2Model
                - forward

            ## MyTest2ForCausalLM

            [[autodoc]] MyTest2ForCausalLM

            ## MyTest2ImageProcessorFast

            [[autodoc]] MyTest2ImageProcessorFast

            ## MyTest2FeatureExtractor

            [[autodoc]] MyTest2FeatureExtractor

            ## MyTest2Processor

            [[autodoc]] MyTest2Processor
            """
        )
        self.assertFileIsEqual(EXPECTED_DOC, os.path.join(self.DOC_PATH, "model_doc", "my_test2.md"))
