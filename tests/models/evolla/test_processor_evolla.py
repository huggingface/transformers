# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import random
import shutil
import tempfile
import unittest

from transformers import (
    AutoProcessor,
    EsmTokenizer,
    EvollaProcessor,
    LlamaTokenizerFast,
)
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch

if is_vision_available():
    pass


EVOLLA_VALID_AA = list("ACDEFGHIKLMNPQRSTVWY#")
EVOLLA_VALID_FS = list("pynwrqhgdlvtmfsaeikc#")


@require_torch
class EvollaProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = EvollaProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        protein_tokenizer = EsmTokenizer.from_pretrained("/zhouxibin/workspaces/ProteinQA/Models/SaProt_35M_AF2")
        tokenizer = LlamaTokenizerFast.from_pretrained(
            "/zhouxibin/workspaces/ProteinQA/Models/meta-llama_Meta-Llama-3-8B-Instruct"
        )

        processor = EvollaProcessor(protein_tokenizer, tokenizer)

        processor.save_pretrained(self.tmpdirname)

        self.input_keys = ["protein_input_ids", "protein_attention_mask", "input_ids", "attention_mask"]

    def prepare_input_and_expected_output(self):
        amino_acid_sequence = "AAAA"
        foldseek_sequence = "dddd"
        question = "What is the function of this protein?"

        expected_output = {
            "protein_input_ids": torch.tensor([[0, 13, 13, 13, 13, 2]]),
            "protein_attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
            "input_ids": torch.tensor(
                [
                    [
                        128000,
                        128006,
                        9125,
                        128007,
                        271,
                        2675,
                        527,
                        459,
                        15592,
                        6335,
                        430,
                        649,
                        4320,
                        904,
                        4860,
                        922,
                        13128,
                        13,
                        128009,
                        128006,
                        882,
                        128007,
                        271,
                        3923,
                        374,
                        279,
                        734,
                        315,
                        420,
                        13128,
                        30,
                        128009,
                        128006,
                        78191,
                        128007,
                        271,
                    ]
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    [
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                    ]
                ]
            ),
        }
        protein_dict = {"aa_seq": amino_acid_sequence, "foldseek": foldseek_sequence}
        message = [
            {"role": "system", "content": "You are an AI expert that can answer any questions about protein."},
            {"role": "user", "content": question},
        ]
        return protein_dict, message, expected_output

    def test_processor(self):
        protein_tokenizer = self.get_protein_tokenizer()
        tokenizer = self.get_tokenizer()

        processor = EvollaProcessor(protein_tokenizer, tokenizer)

        protein_dict, message, expected_output = self.prepare_input_and_expected_output()
        inputs = processor(proteins=[protein_dict], messages_list=[message])

        # check if the input is correct
        for key, value in expected_output.items():
            self.assertTrue(
                torch.equal(inputs[key], value),
                f"inputs[key] is {inputs[key]} and expected_output[key] is {expected_output[key]}",
            )

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_protein_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).protein_tokenizer

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_inputs_single(self):
        proteins = {
            "aa_seq": "".join(random.choices(EVOLLA_VALID_AA, k=100)),
            "foldseek": "".join(random.choices(EVOLLA_VALID_FS, k=100)),
        }
        return proteins

    def prepare_inputs_pair(self):
        proteins = [
            {
                "aa_seq": "".join(random.choices(EVOLLA_VALID_AA, k=100)),
                "foldseek": "".join(random.choices(EVOLLA_VALID_FS, k=100)),
            },
            {
                "aa_seq": "".join(random.choices(EVOLLA_VALID_AA, k=100)),
                "foldseek": "".join(random.choices(EVOLLA_VALID_FS, k=100)),
            },
        ]
        return proteins

    def prepare_inputs_long(self):
        proteins = [
            {
                "aa_seq": "".join(random.choices(EVOLLA_VALID_AA, k=100)),
                "foldseek": "".join(random.choices(EVOLLA_VALID_FS, k=100)),
            },
            {
                "aa_seq": "".join(random.choices(EVOLLA_VALID_AA, k=2000)),
                "foldseek": "".join(random.choices(EVOLLA_VALID_FS, k=2000)),
            },
        ]
        return proteins

    def prepare_inputs_short(self):
        proteins = [
            {
                "aa_seq": "".join(random.choices(EVOLLA_VALID_AA, k=1)),
                "foldseek": "".join(random.choices(EVOLLA_VALID_FS, k=1)),
            },
            {
                "aa_seq": "".join(random.choices(EVOLLA_VALID_AA, k=100)),
                "foldseek": "".join(random.choices(EVOLLA_VALID_FS, k=100)),
            },
        ]
        return proteins

    def prepare_inputs_empty(self):
        proteins = [
            {
                "aa_seq": "",
                "foldseek": "",
            },
            {
                "aa_seq": "".join(random.choices(EVOLLA_VALID_AA, k=100)),
                "foldseek": "".join(random.choices(EVOLLA_VALID_FS, k=100)),
            },
        ]
        return proteins

    def prepare_inputs(self, protein_types="pair"):
        r"""
        Prepare inputs for the test.

        Args:
            protein_types (`str`): the types of proteins to prepare.
                - "single": a single correct protein.
                - "pair": a pair of correct proteins.
                - "long": a long sequence of correct proteins and a correct protein.
                - "short": a short sequence of correct proteins (only have 1 aa) and a correct protein.
                - "empty": an empty sequence of proteins and a correct protein.
        """
        if protein_types == "single":
            proteins = self.prepare_inputs_single()
        elif protein_types == "pair":
            proteins = self.prepare_inputs_pair()
        elif protein_types == "long":
            proteins = self.prepare_inputs_long()
        elif protein_types == "short":
            proteins = self.prepare_inputs_short()
        elif protein_types == "empty":
            proteins = self.prepare_inputs_empty()
        else:
            raise ValueError(
                f"protein_types should be one of 'single', 'pair', 'long','short', 'empty', but got {protein_types}"
            )

        questions = ["What is the function of the protein?"] * len(proteins)
        messages_list = []
        for question in questions:
            messages = [
                {"role": "system", "content": "You are an AI expert that can answer any questions about protein."},
                {"role": "user", "content": question},
            ]
            messages_list.append(messages)
        return proteins, messages_list

    def test_tokenizer_decode(self):
        protein_tokenizer = self.get_protein_tokenizer()
        tokenizer = self.get_tokenizer()

        processor = EvollaProcessor(tokenizer=tokenizer, protein_tokenizer=protein_tokenizer, return_tensors="pt")

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        protein_tokenizer = self.get_protein_tokenizer()
        tokenizer = self.get_tokenizer()

        processor = EvollaProcessor(tokenizer=tokenizer, protein_tokenizer=protein_tokenizer)
        proteins, messages_list = self.prepare_inputs()

        inputs = processor(messages_list=messages_list, proteins=proteins, padding="longest", return_tensors="pt")

        # For now the processor supports only ['pixel_values', 'input_ids', 'attention_mask']
        self.assertSetEqual(set(inputs.keys()), set(self.input_keys))
