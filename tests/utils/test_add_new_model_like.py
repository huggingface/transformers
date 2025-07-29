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
import unittest

import transformers.commands.add_new_model_like
from transformers.commands.add_new_model_like import ModelInfos, create_new_model_like
from transformers.testing_utils import require_torch


REPO_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODELS_TO_COPY = ("auto", "llama", "phi4_multimodal")


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

    @classmethod
    def tearDownClass(cls):
        transformers.commands.add_new_model_like.REPO_PATH = cls.ORIGINAL_REPO
        transformers.commands.add_new_model_like.TRANSFORMERS_PATH = cls.ORIGINAL_TRANSFORMERS_REPO
        del cls.FAKE_REPO

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
