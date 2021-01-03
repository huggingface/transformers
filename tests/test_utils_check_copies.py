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
import re
import shutil
import sys
import tempfile
import unittest


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(git_repo_path, "utils"))

import check_copies  # noqa: E402


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


class CopyCheckTester(unittest.TestCase):
    def setUp(self):
        self.transformer_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.transformer_dir, "models/bert/"))
        check_copies.TRANSFORMER_PATH = self.transformer_dir
        shutil.copy(
            os.path.join(git_repo_path, "src/transformers/models/bert/modeling_bert.py"),
            os.path.join(self.transformer_dir, "models/bert/modeling_bert.py"),
        )

    def tearDown(self):
        check_copies.TRANSFORMER_PATH = "src/transformers"
        shutil.rmtree(self.transformer_dir)

    def check_copy_consistency(self, comment, class_name, class_code, overwrite_result=None):
        code = comment + f"\nclass {class_name}(nn.Module):\n" + class_code
        if overwrite_result is not None:
            expected = comment + f"\nclass {class_name}(nn.Module):\n" + overwrite_result
        fname = os.path.join(self.transformer_dir, "new_code.py")
        with open(fname, "w") as f:
            f.write(code)
        if overwrite_result is None:
            self.assertTrue(len(check_copies.is_copy_consistent(fname)) == 0)
        else:
            check_copies.is_copy_consistent(f.name, overwrite=True)
            with open(fname, "r") as f:
                self.assertTrue(f.read(), expected)

    def test_find_code_in_transformers(self):
        code = check_copies.find_code_in_transformers("models.bert.modeling_bert.BertLMPredictionHead")
        self.assertEqual(code, REFERENCE_CODE)

    def test_is_copy_consistent(self):
        # Base copy consistency
        self.check_copy_consistency(
            "# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead",
            "BertLMPredictionHead",
            REFERENCE_CODE + "\n",
        )

        # With no empty line at the end
        self.check_copy_consistency(
            "# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead",
            "BertLMPredictionHead",
            REFERENCE_CODE,
        )

        # Copy consistency with rename
        self.check_copy_consistency(
            "# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->TestModel",
            "TestModelLMPredictionHead",
            re.sub("Bert", "TestModel", REFERENCE_CODE),
        )

        # Copy consistency with a really long name
        long_class_name = "TestModelWithAReallyLongNameBecauseSomePeopleLikeThatForSomeReasonIReallyDontUnderstand"
        self.check_copy_consistency(
            f"# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->{long_class_name}",
            f"{long_class_name}LMPredictionHead",
            re.sub("Bert", long_class_name, REFERENCE_CODE),
        )

        # Copy consistency with overwrite
        self.check_copy_consistency(
            "# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->TestModel",
            "TestModelLMPredictionHead",
            REFERENCE_CODE,
            overwrite_result=re.sub("Bert", "TestModel", REFERENCE_CODE),
        )
