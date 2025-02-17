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

import unittest

from parameterized import parameterized
from pytest import mark

from transformers import BitsAndBytesConfig, EvollaConfig, is_torch_available, is_vision_available
from transformers.testing_utils import (
    TestCasePlus,
    require_accelerate,
    require_bitsandbytes,
    require_torch,
    require_torch_gpu,
    require_torch_sdpa,
    slow,
    torch_device,
)
from transformers.utils import (
    cached_property,
    is_accelerate_available,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_accelerate_available():
    pass

if is_torch_available():
    import torch

    from transformers import EvollaForProteinText2Text, EvollaModel, EvollaProcessor

if is_vision_available():
    pass


class EvollaModelTester:
    def __init__(
        self,
        parent,
        batch_size=1,
        is_training=False,
        text_seq_length=20,
        text_vocab_size=100,
        protein_seq_length=10,
        protein_vocab_size=20,
        hidden_size=4,  # llama hidden size
        intermediate_size=7,  # llama intermediate size
        num_hidden_layers=1,  # llama hidden layers
        num_attention_heads=2,  # llama attention heads
        num_key_value_heads=2,  # llama key value heads
        protein_hidden_size=8,  # protein encoder hidden size
        protein_num_hidden_layers=1,  # protein encoder hidden layers
        protein_num_attention_heads=4,  # protein encoder attention heads
        protein_intermediate_size=11,  # protein encoder intermediate size
        resampler_num_latents=7,  # sequence compressor num latents
        resampler_ff_mult=1,  # sequence compressor ff mult
        resampler_depth=2,  # sequence compressor depth
        resampler_dim_head=4,  # sequence compressor dim head
        resampler_heads=2,  # sequence compressor heads
        aligner_num_add_layers=1,  # sequence aligner num add layers
        aligner_ffn_mult=1,  # sequence aligner ffn mult
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
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.protein_hidden_size = protein_hidden_size
        self.protein_num_hidden_layers = protein_num_hidden_layers
        self.protein_num_attention_heads = protein_num_attention_heads
        self.protein_intermediate_size = protein_intermediate_size

        self.resampler_num_latents = resampler_num_latents
        self.resampler_ff_mult = resampler_ff_mult
        self.resampler_depth = resampler_depth
        self.resampler_dim_head = resampler_dim_head
        self.resampler_heads = resampler_heads

        self.aligner_num_add_layers = aligner_num_add_layers
        self.aligner_ffn_mult = aligner_ffn_mult

        self.use_input_mask = use_input_mask
        self.is_training = is_training

    @property
    def is_encoder_decoder(self):
        return False

    def prepare_config_and_inputs(self, num_proteins=None):
        batch_size = num_proteins if num_proteins is not None else self.batch_size
        text_input_ids = ids_tensor([batch_size, self.text_seq_length], self.text_vocab_size)

        protein_input_ids = ids_tensor([batch_size, self.protein_seq_length], self.protein_vocab_size)

        if self.use_input_mask:
            text_input_mask = random_attention_mask([batch_size, self.text_seq_length])
            protein_input_mask = random_attention_mask([batch_size, self.protein_seq_length])

        config = self.get_config()
        return (config, text_input_ids, text_input_mask, protein_input_ids, protein_input_mask)

    def get_config(self):
        return EvollaConfig(
            vocab_size=self.text_vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            aligner_ffn_mult=self.aligner_ffn_mult,
            aligner_num_add_layers=self.aligner_num_add_layers,
            protein_vocab_size=self.protein_vocab_size,
            protein_hidden_size=self.protein_hidden_size,
            protein_num_hidden_layers=self.protein_num_hidden_layers,
            protein_num_attention_heads=self.protein_num_attention_heads,
            protein_intermediate_size=self.protein_intermediate_size,
            resampler_depth=self.resampler_depth,
            resampler_dim_head=self.resampler_dim_head,
            resampler_heads=self.resampler_heads,
            resampler_num_latents=self.resampler_num_latents,
            resampler_ff_mult=self.resampler_ff_mult,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        input_mask,
        protein_input_ids,
        protein_input_mask,
        batch_size=None,
    ):
        batch_size = batch_size if batch_size is not None else self.batch_size
        model = EvollaModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_input_mask,
        )
        self.parent.assertEqual(result.last_hidden_state.shape, (batch_size, input_ids.shape[1], self.hidden_size))

    def create_and_check_model_gen(
        self,
        config,
        input_ids,
        input_mask,
        protein_input_ids,
        protein_input_mask,
    ):
        model = EvollaForProteinText2Text(config)
        model.to(torch_device)
        model.eval()
        model.generate(
            input_ids,
            attention_mask=input_mask,
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_input_mask,
            max_length=self.seq_length + 2,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, text_input_ids, text_input_mask, protein_input_ids, protein_input_mask) = config_and_inputs
        inputs_dict = {
            "input_ids": text_input_ids,
            "attention_mask": text_input_mask,
            "protein_input_ids": protein_input_ids,
            "protein_attention_mask": protein_input_mask,
        }
        return config, inputs_dict


@require_torch
class EvollaModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (EvollaModel, EvollaForProteinText2Text) if is_torch_available() else ()
    pipeline_model_mapping = {"feature-extraction": EvollaModel} if is_torch_available() else {}
    test_pruning = False
    test_headmasking = False
    test_torchscript = False
    test_resize_embeddings = False
    maxDiff = None

    def setUp(self):
        self.model_tester = EvollaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EvollaConfig, hidden_size=37)

    @property
    def is_encoder_decoder(self):
        return self.model_tester.is_encoder_decoder

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        # XXX: EvollaForProteinText2Text has no MODEL_FOR group yet, but it should be the same
        # as MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, so for now manually changing to do the right thing
        # as super won't do it
        if return_labels:
            inputs_dict["labels"] = torch.zeros(
                (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
            )

        return inputs_dict

    def test_model_outputs_equivalence(self):
        try:
            orig = self.all_model_classes
            # EvollaModel.forward doesn't have labels input arg - only EvollaForProteinText2Text does
            self.all_model_classes = (EvollaForProteinText2Text,) if is_torch_available() else ()
            super().test_model_outputs_equivalence()
        finally:
            self.all_model_classes = orig

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_single_protein(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(num_proteins=1)
        self.model_tester.create_and_check_model(*config_and_inputs, batch_size=1)

    def test_model_multiple_proteins(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(num_proteins=2)
        self.model_tester.create_and_check_model(*config_and_inputs, batch_size=2)

    def test_generate_single_protein(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(num_proteins=1)
        self.model_tester.create_and_check_model_gen(*config_and_inputs)

    def test_generate_multiple_proteins(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(num_proteins=2)
        self.model_tester.create_and_check_model_gen(*config_and_inputs)

    def test_saprot_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        protein_informations = {
            "input_ids": inputs_dict["protein_input_ids"],
            "attention_mask": inputs_dict["protein_attention_mask"],
        }
        for model_class in self.all_model_classes:
            if model_class is not EvollaModel:
                continue
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            protein_encoder_outputs = model.protein_encoder.sequence_encode(**protein_informations, return_dict=True)
            print(model_class, protein_encoder_outputs)

    def test_protein_encoder_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        protein_informations = {
            "input_ids": inputs_dict["protein_input_ids"],
            "attention_mask": inputs_dict["protein_attention_mask"],
        }
        for model_class in self.all_model_classes:
            if model_class is not EvollaModel:
                continue
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            protein_encoder_outputs = model.protein_encoder(**protein_informations, return_dict=True)
            print(model_class, protein_encoder_outputs)

    def test_single_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            print(outputs)

    def test_initialization(self):
        # we skip the latents initialization test
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # skip latents
                    if name.endswith("latents"):
                        print(f"Skipping latents {name}")
                        continue
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
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
class EvollaModelIntegrationTest(TestCasePlus):
    @cached_property
    def default_processor(self):
        return EvollaProcessor.from_pretrained("/zhouxibin/workspaces/transformers/evolla-base", revision="refs/pr/11")

    @require_bitsandbytes
    @slow
    def test_inference_natural_language_visual_reasoning(self):
        protein_information = {"aa_seq": "AAAA", "foldseek": "dddd"}
        proteins = [protein_information]

        message = [
            {"role": "system", "content": "You are an AI expert that can answer any questions about protein."},
            {"role": "user", "content": "What is the function of this protein?"},
        ]
        messages_list = [message]
        processor = self.default_processor
        inputs = processor(messages_list=messages_list, proteins=proteins, return_tensors="pt", padding="longest").to(
            torch_device
        )

        # the CI gpu is small so using quantization to fit
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
        )
        model = EvollaForProteinText2Text.from_pretrained(
            "/zhouxibin/workspaces/transformers/evolla-base",
            quantization_config=quantization_config,
            device_map="auto",
        )
        generated_ids = model.generate(**inputs, max_length=100)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # keep for debugging
        for i, t in enumerate(generated_text):
            t = bytes(t, "utf-8").decode("unicode_escape")
            print(f"{i}:\n{t}\n")

        self.assertIn("image of two cats", generated_text[0])
        self.assertIn("image of two dogs", generated_text[1])
