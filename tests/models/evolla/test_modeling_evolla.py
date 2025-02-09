# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Evolla model."""

import inspect
import unittest

import pytest
from parameterized import parameterized

from transformers import BitsAndBytesConfig, EvollaConfig, is_torch_available, is_vision_available
from transformers.testing_utils import (
    TestCasePlus,
    is_pt_tf_cross_test,
    require_bitsandbytes,
    require_torch,
    require_torch_sdpa,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import cached_property
from transformers import EsmTokenizer, LlamaTokenizerFast

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import EvollaForVisionText2Text, EvollaModel, EvollaProcessor
    from transformers.models.evolla.configuration_evolla import EvollaPerceiverConfig, EvollaVisionConfig

if is_vision_available():
    from PIL import Image

@require_torch
class EvollaModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (EvollaModel) if is_torch_available() else ()

    def setUp(self):
        self.model = EvollaModel(EvollaConfig())
        protein_tokenizer = EsmTokenizer.from_pretrained("/zhouxibin/workspaces/ProteinQA/Models/SaProt_35M_AF2")
        tokenizer = LlamaTokenizerFast.from_pretrained("/zhouxibin/workspaces/ProteinQA/Models/meta-llama_Meta-Llama-3-8B-Instruct")
        self.processor = EvollaProcessor(protein_tokenizer, tokenizer)

    def prepare_input_and_expected_output(self):
        amino_acid_sequence = "AAAA"
        foldseek_sequence = "dddd"
        question = "What is the function of this protein?"
        answer = "I don't know"

        expected_output = {
            "protein_input_ids": torch.tensor([[0, 13, 13, 13, 13, 2]]),
            "protein_attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
            "text_input_ids": torch.tensor([[128000, 128006,   9125, 128007,    271,   2675,    527,    459,  15592,
            6335,    430,    649,   4320,    904,   4860,    922,  13128,     13,
          128009, 128006,    882, 128007,    271,   3923,    374,    279,    734,
             315,    420,  13128,     30, 128009, 128006,  78191, 128007,    271]]),
            "text_attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
        }
        protein_dict = {
            "aa_seq": amino_acid_sequence,
            "foldseek": foldseek_sequence
        }
        message = [{"role": "system", "content": "You are an AI expert that can answer any questions about protein."},
                   {"role": "user", "content": question}]
        return protein_dict, message, expected_output

    def test_processor(self):
        protein_dict, message, expected_output = self.prepare_input_and_expected_output()
        inputs = self.processor(proteins=[protein_dict],
                                messages_list=[message])
        
        # check if the input is correct
        for key, value in expected_output.items():
            self.assertTrue(torch.equal(inputs[key], value))

    def test_saprot_output(self):
        protein_dict, message, expected_output = self.prepare_input_and_expected_output()
        inputs = self.processor(proteins=[protein_dict],
                                messages_list=[message])
        
        protein_informations = {
            "input_ids": inputs["protein_input_ids"],
            "attention_mask": inputs["protein_attention_mask"],
        }
        saprot_outputs = self.model.protein_encoder.sequence_encode(**protein_informations, return_dict=True, output_hidden_states=False)
        # TODO: check accuracy
        print(saprot_outputs)
        print(saprot_outputs.last_hidden_state.shape)
        print(saprot_outputs.pooler_output.shape)

    def test_protein_encoder_output(self):
        protein_dict, message, expected_output = self.prepare_input_and_expected_output()
        inputs = self.processor(proteins=[protein_dict],
                                messages_list=[message])
        
        protein_informations = {
            "input_ids": inputs["protein_input_ids"],
            "attention_mask": inputs["protein_attention_mask"],
        }
        protein_encoder_outputs = self.model.protein_encoder(**protein_informations, return_dict=True)
        # TODO: check accuracy
        print(protein_encoder_outputs)
    
    def test_single_forward(self):
        protein_dict, message, expected_output = self.prepare_input_and_expected_output()
        inputs = self.processor(proteins=[protein_dict],
                                messages_list=[message])
        
        outputs = self.model(**inputs)
        # TODO: check accuracy
        print(outputs)

@require_torch
@require_vision
class EvollaModelIntegrationTest(TestCasePlus):
    @cached_property
    def default_processor(self):
        return (
            EvollaProcessor.from_pretrained("westlake-repl/Evolla-10B", revision="refs/pr/11")
            if is_vision_available()
            else None
        )

    @require_bitsandbytes
    @slow
    def test_inference_natural_language_visual_reasoning(self):
        cat_image_path = self.tests_dir / "fixtures/tests_samples/COCO/000000039769.png"
        cats_image_obj = Image.open(cat_image_path)  # 2 cats
        dogs_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_nlvr2/raw/main/image1.jpeg"

        prompts = [
            [
                "User:",
                dogs_image_url,
                "Describe this image.\nAssistant: An image of two dogs.\n",
                "User:",
                cats_image_obj,
                "Describe this image.\nAssistant:",
            ],
            [
                "User:",
                cats_image_obj,
                "Describe this image.\nAssistant: An image of two kittens.\n",
                "User:",
                dogs_image_url,
                "Describe this image.\nAssistant:",
            ],
        ]

        # the CI gpu is small so using quantization to fit
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
        )
        model = EvollaForVisionText2Text.from_pretrained(
            "westlake-repl/Evolla-10B", quantization_config=quantization_config, device_map="auto"
        )
        processor = self.default_processor
        inputs = processor(text=prompts, return_tensors="pt", padding="longest").to(torch_device)
        generated_ids = model.generate(**inputs, max_length=100)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # keep for debugging
        for i, t in enumerate(generated_text):
            t = bytes(t, "utf-8").decode("unicode_escape")
            print(f"{i}:\n{t}\n")

        self.assertIn("image of two cats", generated_text[0])
        self.assertIn("image of two dogs", generated_text[1])
