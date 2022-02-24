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

import json
import os
import shutil
import tempfile
from unittest import TestCase

from transformers import BartTokenizer, BartTokenizerFast, DPRQuestionEncoderTokenizer, DPRQuestionEncoderTokenizerFast
from transformers.file_utils import is_datasets_available, is_faiss_available, is_torch_available
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bert.tokenization_bert import VOCAB_FILES_NAMES as DPR_VOCAB_FILES_NAMES
from transformers.models.dpr.configuration_dpr import DPRConfig
from transformers.models.roberta.tokenization_roberta import VOCAB_FILES_NAMES as BART_VOCAB_FILES_NAMES
from transformers.testing_utils import require_faiss, require_tokenizers, require_torch, slow


if is_torch_available() and is_datasets_available() and is_faiss_available():
    from transformers.models.rag.configuration_rag import RagConfig
    from transformers.models.rag.tokenization_rag import RagTokenizer


@require_faiss
@require_torch
class RagTokenizerTest(TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        self.retrieval_vector_size = 8

        # DPR tok
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
        dpr_tokenizer_path = os.path.join(self.tmpdirname, "dpr_tokenizer")
        os.makedirs(dpr_tokenizer_path, exist_ok=True)
        self.vocab_file = os.path.join(dpr_tokenizer_path, DPR_VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

        # BART tok
        vocab = [
            "l",
            "o",
            "w",
            "e",
            "r",
            "s",
            "t",
            "i",
            "d",
            "n",
            "\u0120",
            "\u0120l",
            "\u0120n",
            "\u0120lo",
            "\u0120low",
            "er",
            "\u0120lowest",
            "\u0120newer",
            "\u0120wider",
            "<unk>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        bart_tokenizer_path = os.path.join(self.tmpdirname, "bart_tokenizer")
        os.makedirs(bart_tokenizer_path, exist_ok=True)
        self.vocab_file = os.path.join(bart_tokenizer_path, BART_VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(bart_tokenizer_path, BART_VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_dpr_tokenizer(self) -> DPRQuestionEncoderTokenizer:
        return DPRQuestionEncoderTokenizer.from_pretrained(os.path.join(self.tmpdirname, "dpr_tokenizer"))

    def get_bart_tokenizer(self) -> BartTokenizer:
        return BartTokenizer.from_pretrained(os.path.join(self.tmpdirname, "bart_tokenizer"))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    @require_tokenizers
    def test_save_load_pretrained_with_saved_config(self):

        save_dir = os.path.join(self.tmpdirname, "rag_tokenizer")
        rag_config = RagConfig(question_encoder=DPRConfig().to_dict(), generator=BartConfig().to_dict())
        rag_tokenizer = RagTokenizer(question_encoder=self.get_dpr_tokenizer(), generator=self.get_bart_tokenizer())
        rag_config.save_pretrained(save_dir)
        rag_tokenizer.save_pretrained(save_dir)
        new_rag_tokenizer = RagTokenizer.from_pretrained(save_dir, config=rag_config)
        self.assertIsInstance(new_rag_tokenizer.question_encoder, DPRQuestionEncoderTokenizerFast)
        self.assertEqual(new_rag_tokenizer.question_encoder.get_vocab(), rag_tokenizer.question_encoder.get_vocab())
        self.assertIsInstance(new_rag_tokenizer.generator, BartTokenizerFast)
        self.assertEqual(new_rag_tokenizer.generator.get_vocab(), rag_tokenizer.generator.get_vocab())

    @slow
    def test_pretrained_token_nq_tokenizer(self):
        tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        input_strings = [
            "who got the first nobel prize in physics",
            "when is the next deadpool movie being released",
            "which mode is used for short wave broadcast service",
            "who is the owner of reading football club",
            "when is the next scandal episode coming out",
            "when is the last time the philadelphia won the superbowl",
            "what is the most current adobe flash player version",
            "how many episodes are there in dragon ball z",
            "what is the first step in the evolution of the eye",
            "where is gall bladder situated in human body",
            "what is the main mineral in lithium batteries",
            "who is the president of usa right now",
            "where do the greasers live in the outsiders",
            "panda is a national animal of which country",
            "what is the name of manchester united stadium",
        ]
        input_dict = tokenizer(input_strings)
        self.assertIsNotNone(input_dict)

    @slow
    def test_pretrained_sequence_nq_tokenizer(self):
        tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
        input_strings = [
            "who got the first nobel prize in physics",
            "when is the next deadpool movie being released",
            "which mode is used for short wave broadcast service",
            "who is the owner of reading football club",
            "when is the next scandal episode coming out",
            "when is the last time the philadelphia won the superbowl",
            "what is the most current adobe flash player version",
            "how many episodes are there in dragon ball z",
            "what is the first step in the evolution of the eye",
            "where is gall bladder situated in human body",
            "what is the main mineral in lithium batteries",
            "who is the president of usa right now",
            "where do the greasers live in the outsiders",
            "panda is a national animal of which country",
            "what is the name of manchester united stadium",
        ]
        input_dict = tokenizer(input_strings)
        self.assertIsNotNone(input_dict)
