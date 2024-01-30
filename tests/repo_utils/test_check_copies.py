# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, "utils"))

import check_copies  # noqa: E402
from check_copies import convert_to_localized_md, find_code_in_transformers, is_copy_consistent  # noqa: E402


# This is the reference code that will be used in the tests.
# If BertLMPredictionHead is changed in modeling_bert.py, this code needs to be manually updated.
REFERENCE_CODE = """    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
"""

MOCK_BERT_CODE = """from ...modeling_utils import PreTrainedModel

def bert_function(x):
    return x


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__()
        self.bert = BertEncoder(config)

    @add_docstring(BERT_DOCSTRING)
    def forward(self, x):
        return self.bert(x)
"""

MOCK_BERT_COPY_CODE = """from ...modeling_utils import PreTrainedModel

# Copied from transformers.models.bert.modeling_bert.bert_function
def bert_copy_function(x):
    return x


# Copied from transformers.models.bert.modeling_bert.BertAttention
class BertCopyAttention(nn.Module):
    def __init__(self, config):
        super().__init__()


# Copied from transformers.models.bert.modeling_bert.BertModel with Bert->BertCopy all-casing
class BertCopyModel(BertCopyPreTrainedModel):
    def __init__(self, config):
        super().__init__()
        self.bertcopy = BertCopyEncoder(config)

    @add_docstring(BERTCOPY_DOCSTRING)
    def forward(self, x):
        return self.bertcopy(x)
"""


MOCK_DUMMY_BERT_CODE_MATCH = """
class BertDummyModel:
    attr_1 = 1
    attr_2 = 2

    def __init__(self, a=1, b=2):
        self.a = a
        self.b = b

    # Copied from transformers.models.dummy_gpt2.modeling_dummy_gpt2.GPT2DummyModel.forward
    def forward(self, c):
        return 1

    def existing_common(self, c):
        return 4

    def existing_diff_to_be_ignored(self, c):
        return 9
"""


MOCK_DUMMY_ROBERTA_CODE_MATCH = """
# Copied from transformers.models.dummy_bert_match.modeling_dummy_bert_match.BertDummyModel with BertDummy->RobertaBertDummy
class RobertaBertDummyModel:

    attr_1 = 1
    attr_2 = 2

    def __init__(self, a=1, b=2):
        self.a = a
        self.b = b

    # Ignore copy
    def only_in_roberta_to_be_ignored(self, c):
        return 3

    # Copied from transformers.models.dummy_gpt2.modeling_dummy_gpt2.GPT2DummyModel.forward
    def forward(self, c):
        return 1

    def existing_common(self, c):
        return 4

    # Ignore copy
    def existing_diff_to_be_ignored(self, c):
        return 6
"""


MOCK_DUMMY_BERT_CODE_NO_MATCH = """
class BertDummyModel:
    attr_1 = 1
    attr_2 = 2

    def __init__(self, a=1, b=2):
        self.a = a
        self.b = b

    # Copied from transformers.models.dummy_gpt2.modeling_dummy_gpt2.GPT2DummyModel.forward
    def forward(self, c):
        return 1

    def only_in_bert(self, c):
        return 7

    def existing_common(self, c):
        return 4

    def existing_diff_not_ignored(self, c):
        return 8

    def existing_diff_to_be_ignored(self, c):
        return 9
"""


MOCK_DUMMY_ROBERTA_CODE_NO_MATCH = """
# Copied from transformers.models.dummy_bert_no_match.modeling_dummy_bert_no_match.BertDummyModel with BertDummy->RobertaBertDummy
class RobertaBertDummyModel:

    attr_1 = 1
    attr_2 = 3

    def __init__(self, a=1, b=2):
        self.a = a
        self.b = b

    # Ignore copy
    def only_in_roberta_to_be_ignored(self, c):
        return 3

    # Copied from transformers.models.dummy_gpt2.modeling_dummy_gpt2.GPT2DummyModel.forward
    def forward(self, c):
        return 1

    def only_in_roberta_not_ignored(self, c):
        return 2

    def existing_common(self, c):
        return 4

    def existing_diff_not_ignored(self, c):
        return 5

    # Ignore copy
    def existing_diff_to_be_ignored(self, c):
        return 6
"""


EXPECTED_REPLACED_CODE = """
# Copied from transformers.models.dummy_bert_no_match.modeling_dummy_bert_no_match.BertDummyModel with BertDummy->RobertaBertDummy
class RobertaBertDummyModel:
    attr_1 = 1
    attr_2 = 2

    def __init__(self, a=1, b=2):
        self.a = a
        self.b = b

    # Copied from transformers.models.dummy_gpt2.modeling_dummy_gpt2.GPT2DummyModel.forward
    def forward(self, c):
        return 1

    def only_in_bert(self, c):
        return 7

    def existing_common(self, c):
        return 4

    def existing_diff_not_ignored(self, c):
        return 8

    # Ignore copy
    def existing_diff_to_be_ignored(self, c):
        return 6

    # Ignore copy
    def only_in_roberta_to_be_ignored(self, c):
        return 3
"""


def replace_in_file(filename, old, new):
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    content = content.replace(old, new)

    with open(filename, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def create_tmp_repo(tmp_dir):
    """
    Creates a mock repository in a temporary folder for testing.
    """
    tmp_dir = Path(tmp_dir)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(exist_ok=True)

    model_dir = tmp_dir / "src" / "transformers" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "bert": MOCK_BERT_CODE,
        "bertcopy": MOCK_BERT_COPY_CODE,
        "dummy_bert_match": MOCK_DUMMY_BERT_CODE_MATCH,
        "dummy_roberta_match": MOCK_DUMMY_ROBERTA_CODE_MATCH,
        "dummy_bert_no_match": MOCK_DUMMY_BERT_CODE_NO_MATCH,
        "dummy_roberta_no_match": MOCK_DUMMY_ROBERTA_CODE_NO_MATCH,
    }
    for model, code in models.items():
        model_subdir = model_dir / model
        model_subdir.mkdir(exist_ok=True)
        with open(model_subdir / f"modeling_{model}.py", "w", encoding="utf-8", newline="\n") as f:
            f.write(code)


@contextmanager
def patch_transformer_repo_path(new_folder):
    """
    Temporarily patches the variables defines in `check_copies` to use a different location for the repo.
    """
    old_repo_path = check_copies.REPO_PATH
    old_doc_path = check_copies.PATH_TO_DOCS
    old_transformer_path = check_copies.TRANSFORMERS_PATH
    repo_path = Path(new_folder).resolve()
    check_copies.REPO_PATH = str(repo_path)
    check_copies.PATH_TO_DOCS = str(repo_path / "docs" / "source" / "en")
    check_copies.TRANSFORMERS_PATH = str(repo_path / "src" / "transformers")
    try:
        yield
    finally:
        check_copies.REPO_PATH = old_repo_path
        check_copies.PATH_TO_DOCS = old_doc_path
        check_copies.TRANSFORMERS_PATH = old_transformer_path


class CopyCheckTester(unittest.TestCase):
    def test_find_code_in_transformers(self):
        with tempfile.TemporaryDirectory() as tmp_folder:
            create_tmp_repo(tmp_folder)
            with patch_transformer_repo_path(tmp_folder):
                code = find_code_in_transformers("models.bert.modeling_bert.BertAttention")

        reference_code = (
            "class BertAttention(nn.Module):\n    def __init__(self, config):\n        super().__init__()\n"
        )
        self.assertEqual(code, reference_code)

    def test_is_copy_consistent(self):
        path_to_check = ["src", "transformers", "models", "bertcopy", "modeling_bertcopy.py"]
        with tempfile.TemporaryDirectory() as tmp_folder:
            # Base check
            create_tmp_repo(tmp_folder)
            with patch_transformer_repo_path(tmp_folder):
                file_to_check = os.path.join(tmp_folder, *path_to_check)
                diffs = is_copy_consistent(file_to_check)
                self.assertEqual(diffs, [])

            # Base check with an inconsistency
            create_tmp_repo(tmp_folder)
            with patch_transformer_repo_path(tmp_folder):
                file_to_check = os.path.join(tmp_folder, *path_to_check)

                replace_in_file(file_to_check, "self.bertcopy(x)", "self.bert(x)")
                diffs = is_copy_consistent(file_to_check)
                self.assertEqual(diffs, [["models.bert.modeling_bert.BertModel", 22]])

                _ = is_copy_consistent(file_to_check, overwrite=True)

                with open(file_to_check, "r", encoding="utf-8") as f:
                    self.assertEqual(f.read(), MOCK_BERT_COPY_CODE)

    def test_is_copy_consistent_with_ignored_match(self):
        path_to_check = ["src", "transformers", "models", "dummy_roberta_match", "modeling_dummy_roberta_match.py"]
        with tempfile.TemporaryDirectory() as tmp_folder:
            # Base check
            create_tmp_repo(tmp_folder)
            with patch_transformer_repo_path(tmp_folder):
                file_to_check = os.path.join(tmp_folder, *path_to_check)
                diffs = is_copy_consistent(file_to_check)
                self.assertEqual(diffs, [])

    def test_is_copy_consistent_with_ignored_no_match(self):
        path_to_check = [
            "src",
            "transformers",
            "models",
            "dummy_roberta_no_match",
            "modeling_dummy_roberta_no_match.py",
        ]
        with tempfile.TemporaryDirectory() as tmp_folder:
            # Base check with an inconsistency
            create_tmp_repo(tmp_folder)
            with patch_transformer_repo_path(tmp_folder):
                file_to_check = os.path.join(tmp_folder, *path_to_check)

                diffs = is_copy_consistent(file_to_check)
                # line 6: `attr_2 = 3` in `MOCK_DUMMY_ROBERTA_CODE_NO_MATCH`.
                # (which has a leading `\n`.)
                self.assertEqual(
                    diffs, [["models.dummy_bert_no_match.modeling_dummy_bert_no_match.BertDummyModel", 6]]
                )

                _ = is_copy_consistent(file_to_check, overwrite=True)

                with open(file_to_check, "r", encoding="utf-8") as f:
                    self.assertEqual(f.read(), EXPECTED_REPLACED_CODE)

    def test_convert_to_localized_md(self):
        localized_readme = check_copies.LOCALIZED_READMES["README_zh-hans.md"]

        md_list = (
            "1. **[ALBERT](https://huggingface.co/transformers/model_doc/albert.html)** (from Google Research and the"
            " Toyota Technological Institute at Chicago) released with the paper [ALBERT: A Lite BERT for"
            " Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942), by Zhenzhong"
            " Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.\n1."
            " **[DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html)** (from HuggingFace),"
            " released together with the paper [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and"
            " lighter](https://arxiv.org/abs/1910.01108) by Victor Sanh, Lysandre Debut and Thomas Wolf. The same"
            " method has been applied to compress GPT2 into"
            " [DistilGPT2](https://github.com/huggingface/transformers/tree/main/examples/distillation), RoBERTa into"
            " [DistilRoBERTa](https://github.com/huggingface/transformers/tree/main/examples/distillation),"
            " Multilingual BERT into"
            " [DistilmBERT](https://github.com/huggingface/transformers/tree/main/examples/distillation) and a German"
            " version of DistilBERT.\n1. **[ELECTRA](https://huggingface.co/transformers/model_doc/electra.html)**"
            " (from Google Research/Stanford University) released with the paper [ELECTRA: Pre-training text encoders"
            " as discriminators rather than generators](https://arxiv.org/abs/2003.10555) by Kevin Clark, Minh-Thang"
            " Luong, Quoc V. Le, Christopher D. Manning."
        )
        localized_md_list = (
            "1. **[ALBERT](https://huggingface.co/transformers/model_doc/albert.html)** (来自 Google Research and the"
            " Toyota Technological Institute at Chicago) 伴随论文 [ALBERT: A Lite BERT for Self-supervised Learning of"
            " Language Representations](https://arxiv.org/abs/1909.11942), 由 Zhenzhong Lan, Mingda Chen, Sebastian"
            " Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut 发布。\n"
        )
        converted_md_list_sample = (
            "1. **[ALBERT](https://huggingface.co/transformers/model_doc/albert.html)** (来自 Google Research and the"
            " Toyota Technological Institute at Chicago) 伴随论文 [ALBERT: A Lite BERT for Self-supervised Learning of"
            " Language Representations](https://arxiv.org/abs/1909.11942), 由 Zhenzhong Lan, Mingda Chen, Sebastian"
            " Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut 发布。\n1."
            " **[DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html)** (来自 HuggingFace) 伴随论文"
            " [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and"
            " lighter](https://arxiv.org/abs/1910.01108) 由 Victor Sanh, Lysandre Debut and Thomas Wolf 发布。 The same"
            " method has been applied to compress GPT2 into"
            " [DistilGPT2](https://github.com/huggingface/transformers/tree/main/examples/distillation), RoBERTa into"
            " [DistilRoBERTa](https://github.com/huggingface/transformers/tree/main/examples/distillation),"
            " Multilingual BERT into"
            " [DistilmBERT](https://github.com/huggingface/transformers/tree/main/examples/distillation) and a German"
            " version of DistilBERT.\n1. **[ELECTRA](https://huggingface.co/transformers/model_doc/electra.html)** (来自"
            " Google Research/Stanford University) 伴随论文 [ELECTRA: Pre-training text encoders as discriminators rather"
            " than generators](https://arxiv.org/abs/2003.10555) 由 Kevin Clark, Minh-Thang Luong, Quoc V. Le,"
            " Christopher D. Manning 发布。\n"
        )

        num_models_equal, converted_md_list = convert_to_localized_md(
            md_list, localized_md_list, localized_readme["format_model_list"]
        )

        self.assertFalse(num_models_equal)
        self.assertEqual(converted_md_list, converted_md_list_sample)

        num_models_equal, converted_md_list = convert_to_localized_md(
            md_list, converted_md_list, localized_readme["format_model_list"]
        )

        # Check whether the number of models is equal to README.md after conversion.
        self.assertTrue(num_models_equal)

        link_changed_md_list = (
            "1. **[ALBERT](https://huggingface.co/transformers/model_doc/albert.html)** (from Google Research and the"
            " Toyota Technological Institute at Chicago) released with the paper [ALBERT: A Lite BERT for"
            " Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942), by Zhenzhong"
            " Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut."
        )
        link_unchanged_md_list = (
            "1. **[ALBERT](https://huggingface.co/transformers/main/model_doc/albert.html)** (来自 Google Research and"
            " the Toyota Technological Institute at Chicago) 伴随论文 [ALBERT: A Lite BERT for Self-supervised Learning of"
            " Language Representations](https://arxiv.org/abs/1909.11942), 由 Zhenzhong Lan, Mingda Chen, Sebastian"
            " Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut 发布。\n"
        )
        converted_md_list_sample = (
            "1. **[ALBERT](https://huggingface.co/transformers/model_doc/albert.html)** (来自 Google Research and the"
            " Toyota Technological Institute at Chicago) 伴随论文 [ALBERT: A Lite BERT for Self-supervised Learning of"
            " Language Representations](https://arxiv.org/abs/1909.11942), 由 Zhenzhong Lan, Mingda Chen, Sebastian"
            " Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut 发布。\n"
        )

        num_models_equal, converted_md_list = convert_to_localized_md(
            link_changed_md_list, link_unchanged_md_list, localized_readme["format_model_list"]
        )

        # Check if the model link is synchronized.
        self.assertEqual(converted_md_list, converted_md_list_sample)
