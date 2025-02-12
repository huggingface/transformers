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
from pytest import mark
from parameterized import parameterized
import tempfile

from transformers import BitsAndBytesConfig, EvollaConfig, is_torch_available, is_vision_available
from transformers.testing_utils import (
    TestCasePlus,
    is_pt_tf_cross_test,
    require_bitsandbytes,
    require_accelerate,
    require_torch,
    require_torch_gpu,
    require_torch_sdpa,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import (
    is_accelerate_available,
)

from transformers.utils import cached_property
from transformers import EsmTokenizer, LlamaTokenizerFast

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_accelerate_available():
    from accelerate.utils import compute_module_sizes

if is_torch_available():
    import torch

    from transformers import EvollaForVisionText2Text, EvollaModel, EvollaProcessor

if is_vision_available():
    from PIL import Image


class EvollaModelTester:
    def __init__(
        self,
        parent,
        batch_size=1,
        is_training=True,
        text_seq_length=20,
        text_vocab_size=100,
        protein_seq_length=10,
        protein_vocab_size=20,
        hidden_size=4, # llama hidden size
        num_hidden_layers=1, # llama hidden layers
        num_attention_heads=2, # llama attention heads
        num_key_value_heads=2, # llama key value heads
        protein_hidden_size=8, # protein encoder hidden size
        protein_num_hidden_layers=1, # protein encoder hidden layers
        protein_num_attention_heads=4, # protein encoder attention heads
        sequence_compressor_num_latents=7, # sequence compressor num latents
        sequence_compressor_ff_mult=1, # sequence compressor ff mult
        sequence_compressor_depth=2, # sequence compressor depth
        sequence_aligner_num_add_layers=1, # sequence aligner num add layers
        use_input_mask=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.protein_seq_length = protein_seq_length
        self.protein_vocab_size = protein_vocab_size
        self.text_seq_length = text_seq_length
        self.text_vocab_size = text_vocab_size
        self.seq_length = text_seq_length
        
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.protein_hidden_size = protein_hidden_size
        self.protein_num_hidden_layers = protein_num_hidden_layers
        self.protein_num_attention_heads = protein_num_attention_heads

        self.sequence_compressor_num_latents = sequence_compressor_num_latents
        self.sequence_compressor_ff_mult = sequence_compressor_ff_mult
        self.sequence_compressor_depth = sequence_compressor_depth

        self.sequence_aligner_num_add_layers = sequence_aligner_num_add_layers

        self.use_input_mask = use_input_mask
        self.is_training = is_training

    @property
    def is_encoder_decoder(self):
        return False
    
    def prepare_config_and_inputs(self):
        text_input_ids = ids_tensor([self.batch_size, self.text_seq_length], self.text_vocab_size)

        protein_input_ids = ids_tensor([self.batch_size, self.protein_seq_length], self.protein_vocab_size)

        if self.use_input_mask:
            text_input_mask = random_attention_mask([self.batch_size, self.text_seq_length])
            protein_input_mask = random_attention_mask([self.batch_size, self.protein_seq_length])

        config = self.get_config()
        return (config, text_input_ids, text_input_mask, protein_input_ids, protein_input_mask)


    def get_config(self):
        return EvollaConfig(
            llm_config={
                "llama_config": {
                    "num_hidden_layers": self.num_hidden_layers,
                    "hidden_size": self.hidden_size,
                    "num_attention_heads": self.num_attention_heads,
                    "num_key_value_heads": self.num_key_value_heads,
                },
                "sequence_aligner_config": {
                    "num_add_layers": self.sequence_aligner_num_add_layers,
                }
            },
            protein_config={
                "protein_encoder_config": {
                    "num_hidden_layers": self.protein_num_hidden_layers,
                    "hidden_size": self.protein_hidden_size,
                    "num_attention_heads": self.protein_num_attention_heads,
                },
                "resampler_config": {
                    "num_latents": self.sequence_compressor_num_latents,
                    "ff_mult": self.sequence_compressor_ff_mult,
                    "depth": self.sequence_compressor_depth,
                }
            }
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        input_mask,
        pixel_values,
        image_attention_mask,
        interpolate_pos_encoding,
    ):
        model = EvollaModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            pixel_values=pixel_values,
            image_attention_mask=image_attention_mask,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, input_ids.shape[1], self.hidden_size)
        )

    
    def create_and_check_model_gen(
        self,
        config,
        input_ids,
        input_mask,
        pixel_values,
        image_attention_mask,
        interpolate_pos_encoding,
    ):
        model = EvollaForVisionText2Text(config)
        model.to(torch_device)
        model.eval()
        model.generate(
            input_ids,
            attention_mask=input_mask,
            pixel_values=pixel_values,
            image_attention_mask=image_attention_mask,
            interpolate_pos_encoding=interpolate_pos_encoding,
            max_length=self.seq_length + 2,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config, 
            text_input_ids,
            text_input_mask, 
            protein_input_ids, 
            protein_input_mask
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": text_input_ids,
            "attention_mask": text_input_mask,
            "protein_input_ids": protein_input_ids,
            "protein_attention_mask": protein_input_mask,
        }
        return config, inputs_dict
@require_torch
class EvollaModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (EvollaModel,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": EvollaModel}
        if is_torch_available()
        else {}
    )
    test_pruning = False
    test_headmasking = False
    test_torchscript = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = EvollaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EvollaConfig, hidden_size=37)
        self.is_encoder_decoder = self.model_tester.is_encoder_decoder
        # protein_tokenizer = EsmTokenizer.from_pretrained("/zhouxibin/workspaces/ProteinQA/Models/SaProt_35M_AF2")
        # tokenizer = LlamaTokenizerFast.from_pretrained("/zhouxibin/workspaces/ProteinQA/Models/meta-llama_Meta-Llama-3-8B-Instruct")
        # self.processor = EvollaProcessor(protein_tokenizer, tokenizer)
        self.processor = EvollaProcessor.from_pretrained("/zhouxibin/workspaces/transformers/src/transformers/models/evolla/evolla-base")


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
            self.assertTrue(torch.equal(inputs[key], value), f"inputs[key] is {inputs[key]} and expected_output[key] is {expected_output[key]}")

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


    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        seq_len = getattr(self.model_tester, "seq_length", None)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions

            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            self.assertEqual(out_len + 2, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )

    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @require_torch_sdpa
    @unittest.skip("Evolla requires both text and protein inputs which is currently not done in this test.")
    def test_eager_matches_sdpa_inference(self):
        pass

    # @unittest.skip(reason="We cannot configure to output a smaller model.")
    # def test_model_is_small(self):
    #     pass

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
