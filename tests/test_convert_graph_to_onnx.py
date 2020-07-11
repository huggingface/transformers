# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.tokenization_bert import VOCAB_FILES_NAMES

from transformers.convert_graph_to_onnx import convert, verify
from .test_modeling_bert import BertModelTester
from .test_tokenization_common import TokenizerTesterMixin
if is_torch_available():
    from transformers import (
        BertModel,
        BertTokenizer
    )

@require_torch
class ONNXExportTest(TokenizerTesterMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.model_tester = BertModelTester(self)

        # Setting up model files dir
        output_dir = f'{self.tmpdirname}/model_files'

        # Setting up vocab files
        vocab_tokens = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
            "want",
            "##want",
            "##ed",
            "wa",
            "un",
            "runn",
            "##ing",
            ",",
            "low",
            "lowest",
        ]
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

        # Creation of model files
        config, input_ids, _, _, _, _, _ = self.model_tester.prepare_config_and_inputs()
        model = BertModel(config=config)
        tokenizer = BertTokenizer(self.vocab_file)
        self.model_path = f'{self.tmpdirname}/model'
        self.tokenizer_path = f'{self.tmpdirname}/bert'

        model.save_pretrained(self.model_path)
        tokenizer.save_pretrained(self.tokenizer_path)

        # Necessary since AutoTokenizer looks for config.json by default
        os.rename(f'{self.tokenizer_path}/tokenizer_config.json', f'{self.tokenizer_path}/config.json')

        # Conversion to ONNX
        self.out_dir = f'{output_dir}/out'

    def test_bert_export_pt(self):
        convert(
            'pt',
            self.model_path,
            self.out_dir,
            11,
            self.tokenizer_path,
            False,
        )

        verify(self.out_dir)

    def test_bert_export_tf(self):
        convert(
            'tf',
            self.model_path,
            self.out_dir,
            11,
            self.tokenizer_path,
            False,
        )

        verify(self.out_dir)

    def test_bert_export_large(self):
        convert(
            'pt',
            self.model_path,
            self.out_dir,
            11,
            self.tokenizer_path,
            True,
        )

        verify(self.out_dir)
