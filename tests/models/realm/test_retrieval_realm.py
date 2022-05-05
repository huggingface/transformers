# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from datasets import Dataset

from transformers.models.realm.configuration_realm import RealmConfig
from transformers.models.realm.retrieval_realm import _REALM_BLOCK_RECORDS_FILENAME, RealmRetriever
from transformers.models.realm.tokenization_realm import VOCAB_FILES_NAMES, RealmTokenizer


class RealmRetrieverTest(TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        self.num_block_records = 5

        # Realm tok
        vocab_tokens = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
            "test",
            "question",
            "this",
            "is",
            "the",
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "record",
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
        realm_tokenizer_path = os.path.join(self.tmpdirname, "realm_tokenizer")
        os.makedirs(realm_tokenizer_path, exist_ok=True)
        self.vocab_file = os.path.join(realm_tokenizer_path, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

        realm_block_records_path = os.path.join(self.tmpdirname, "realm_block_records")
        os.makedirs(realm_block_records_path, exist_ok=True)

    def get_tokenizer(self) -> RealmTokenizer:
        return RealmTokenizer.from_pretrained(os.path.join(self.tmpdirname, "realm_tokenizer"))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def get_config(self):
        config = RealmConfig(num_block_records=self.num_block_records)
        return config

    def get_dummy_dataset(self):
        dataset = Dataset.from_dict(
            {
                "id": ["0", "1"],
                "question": ["foo", "bar"],
                "answers": [["Foo", "Bar"], ["Bar"]],
            }
        )
        return dataset

    def get_dummy_block_records(self):
        block_records = np.array(
            [
                b"This is the first record",
                b"This is the second record",
                b"This is the third record",
                b"This is the fourth record",
                b"This is the fifth record",
                b"This is a longer longer longer record",
            ],
            dtype=np.object,
        )
        return block_records

    def get_dummy_retriever(self):
        retriever = RealmRetriever(
            block_records=self.get_dummy_block_records(),
            tokenizer=self.get_tokenizer(),
        )
        return retriever

    def test_retrieve(self):
        config = self.get_config()
        retriever = self.get_dummy_retriever()
        tokenizer = retriever.tokenizer

        retrieved_block_ids = np.array([0, 3], dtype=np.long)
        question_input_ids = tokenizer(["Test question"]).input_ids
        answer_ids = tokenizer(
            ["the fourth"],
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        ).input_ids
        max_length = config.reader_seq_len

        has_answers, start_pos, end_pos, concat_inputs = retriever(
            retrieved_block_ids, question_input_ids, answer_ids=answer_ids, max_length=max_length, return_tensors="np"
        )

        self.assertEqual(len(has_answers), 2)
        self.assertEqual(len(start_pos), 2)
        self.assertEqual(len(end_pos), 2)
        self.assertEqual(concat_inputs.input_ids.shape, (2, 10))
        self.assertEqual(concat_inputs.attention_mask.shape, (2, 10))
        self.assertEqual(concat_inputs.token_type_ids.shape, (2, 10))
        self.assertEqual(concat_inputs.special_tokens_mask.shape, (2, 10))
        self.assertEqual(
            tokenizer.convert_ids_to_tokens(concat_inputs.input_ids[0]),
            ["[CLS]", "test", "question", "[SEP]", "this", "is", "the", "first", "record", "[SEP]"],
        )
        self.assertEqual(
            tokenizer.convert_ids_to_tokens(concat_inputs.input_ids[1]),
            ["[CLS]", "test", "question", "[SEP]", "this", "is", "the", "fourth", "record", "[SEP]"],
        )

    def test_block_has_answer(self):
        config = self.get_config()
        retriever = self.get_dummy_retriever()
        tokenizer = retriever.tokenizer

        retrieved_block_ids = np.array([0, 3, 5], dtype=np.long)
        question_input_ids = tokenizer(["Test question"]).input_ids
        answer_ids = tokenizer(
            ["the fourth", "longer longer"],
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        ).input_ids
        max_length = config.reader_seq_len

        has_answers, start_pos, end_pos, _ = retriever(
            retrieved_block_ids, question_input_ids, answer_ids=answer_ids, max_length=max_length, return_tensors="np"
        )

        self.assertEqual([False, True, True], has_answers)
        self.assertEqual([[-1, -1, -1], [6, -1, -1], [6, 7, 8]], start_pos)
        self.assertEqual([[-1, -1, -1], [7, -1, -1], [7, 8, 9]], end_pos)

    def test_save_load_pretrained(self):
        retriever = self.get_dummy_retriever()
        retriever.save_pretrained(os.path.join(self.tmpdirname, "realm_block_records"))

        # Test local path
        retriever = retriever.from_pretrained(os.path.join(self.tmpdirname, "realm_block_records"))
        self.assertEqual(retriever.block_records[0], b"This is the first record")

        # Test mocked remote path
        with patch("transformers.models.realm.retrieval_realm.hf_hub_download") as mock_hf_hub_download:
            mock_hf_hub_download.return_value = os.path.join(
                os.path.join(self.tmpdirname, "realm_block_records"), _REALM_BLOCK_RECORDS_FILENAME
            )
            retriever = RealmRetriever.from_pretrained("google/realm-cc-news-pretrained-openqa")

        self.assertEqual(retriever.block_records[0], b"This is the first record")
