import os
import shutil
import tempfile
from unittest import TestCase

import numpy as np
from datasets import Dataset

from transformers.models.realm.configuration_realm import RealmConfig
from transformers.models.realm.retrieval_realm import RealmRetriever
from transformers.models.realm.tokenization_realm import RealmTokenizer, VOCAB_FILES_NAMES
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch


class RealmRetrieverTest(TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        self.num_block_records = 10

        # Realm tok
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
        realm_tokenizer_path = os.path.join(self.tmpdirname, "realm_tokenizer")
        os.makedirs(realm_tokenizer_path, exist_ok=True)
        self.vocab_file = os.path.join(realm_tokenizer_path, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def get_tokenizer(self) -> RealmTokenizer:
        return RealmTokenizer.from_pretrained(os.path.join(self.tmpdirname, "realm_tokenizer"))
    
    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

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
        np_block_records = np.array(
            [
                "This is the first record",
                "This is the second record",
            ],
            np.object,
        )
        return np_block_records

    def get_dummy_retriever(self):
        config = RealmConfig(
            num_block_records=self.num_block_records
        )
        retriever = RealmRetriever(
            config,
            block_records=self.get_dummy_block_records,
            tokenizer=self.get_tokenizer(),
        )
        return retriever

    def test_retrieve(self):
        pass

    def test_block_has_answer(self):
        pass

    def test_from_pretrained(self):
        pass

    def test_save_pretrained(self):
        pass
