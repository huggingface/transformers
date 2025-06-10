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
"""Testing suite for the PyTorch Dia model."""

import tempfile
import unittest
from typing import Tuple

from transformers.models.dia import DiaConfig, DiaDecoderConfig, DiaEncoderConfig
from transformers.testing_utils import (
    require_torch,
    torch_device,
)
from transformers.utils import is_torch_available, is_torchaudio_available
from transformers.utils.import_utils import is_datasets_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_datasets_available():
    pass

if is_torch_available():
    from transformers import (
        DiaForConditionalGeneration,
        DiaModel,
    )
    from transformers.models.dia.modeling_dia import DiaDecoder, DiaEncoder


if is_torchaudio_available():
    pass


@require_torch
class DiaModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,  # need batch_size != num_hidden_layers
        seq_length=7,
        max_length=50,
        is_training=False,
        vocab_size=100,
        hidden_size=16,
        intermediate_size=37,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=8,
        decoder_hidden_size=32,  # typically larger than encoder
        hidden_act="silu",
        eos_token_id=97,  # special tokens all occur after eos
        pad_token_id=98,
        bos_token_id=99,
        delay_pattern=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.max_length = max_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.decoder_hidden_size = decoder_hidden_size
        self.hidden_act = hidden_act
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        # Set default delay pattern if not provided
        self.delay_pattern = delay_pattern if delay_pattern is not None else [0, 1, 2]
        self.num_channels = len(self.delay_pattern)

    def get_config(self):
        encoder_config = DiaEncoderConfig(
            max_length=self.max_length,
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_attention_heads,  # same as num_attention_heads for testing
            head_dim=self.head_dim,
            intermediate_size=self.intermediate_size,
            vocab_size=self.vocab_size,
            hidden_act=self.hidden_act,
        )

        decoder_config = DiaDecoderConfig(
            max_length=self.max_length,
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.decoder_hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=1,  # GQA
            head_dim=self.head_dim,
            cross_num_attention_heads=self.num_attention_heads,
            cross_head_dim=self.head_dim,
            cross_num_key_value_heads=1,  # GQA
            cross_hidden_size=self.hidden_size,  # match encoder hidden size
            vocab_size=self.vocab_size,
            hidden_act=self.hidden_act,
            num_channels=self.num_channels,
        )

        config = DiaConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            delay_pattern=self.delay_pattern,
        )

        return config

    def prepare_config_and_inputs(self) -> Tuple[DiaConfig, dict]:
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = input_ids.ne(self.pad_token_id)

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length, self.num_channels], self.vocab_size)
        decoder_attention_mask = decoder_input_ids[..., 0].ne(self.pad_token_id)

        config = self.get_config()
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self) -> Tuple[DiaConfig, dict]:
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_model_forward(self, config, inputs_dict):
        model = DiaModel(config=config).to(torch_device).eval()

        input_ids = inputs_dict["input_ids"]
        decoder_input_ids = inputs_dict["decoder_input_ids"]

        # first forward pass
        last_hidden_state = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).last_hidden_state

        self.parent.assertTrue(
            last_hidden_state.shape, (self.batch_size, self.seq_length, config.decoder_config.hidden_size)
        )

    def check_encoder_decoder_model_standalone(self, config, inputs_dict):
        model = DiaModel(config=config).to(torch_device).eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = DiaEncoder.from_pretrained(tmpdirname).to(torch_device)

        encoder_last_hidden_state_2 = encoder(
            input_ids=inputs_dict["input_ids"], attention_mask=inputs_dict["attention_mask"]
        )[0]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 2e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = DiaDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            input_ids=inputs_dict["decoder_input_ids"],
            attention_mask=inputs_dict["decoder_attention_mask"],
            encoder_hidden_states=encoder_last_hidden_state,
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 2e-3)


@require_torch
class DiaModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (DiaModel, DiaForConditionalGeneration) if is_torch_available() else ()
    # We only allow greedy search / sampling with one sequence; see `skip_non_greedy_generate`
    all_generative_model_classes = (DiaForConditionalGeneration,)
    # TODO: needs processor for pipeline
    # pipeline_model_mapping = {"text-to-audio": DiaForConditionalGeneration} if is_torch_available() else {}
    pipeline_model_mapping = {}
    fx_compatible = False
    test_pruning = False
    test_head_masking = False
    is_encoder_decoder = True
    # Indicates VLMs usually but there are many audio models which are also composite
    _is_composite = True

    def setUp(self):
        self.model_tester = DiaModelTester(self)
        # Skipping `has_text_modality` but manually testing down below
        self.config_tester = ConfigTester(self, has_text_modality=False, config_class=DiaConfig)
        self.skip_non_greedy_generate()

    def skip_non_greedy_generate(self):
        skippable_tests = [
            "test_sample_generate_dict_output",  # return sequences > 1
            "test_beam",
            "test_group_beam",
            "test_constrained_beam",
            "test_contrastive",
            "test_assisted",
            "test_dola",
            "test_prompt_lookup",
            "test_model_parallel_beam_search",
            "test_generate_without_input_ids",
            "test_generate_with_head_masking",
        ]

        for test in skippable_tests:
            if self._testMethodName.startswith(test):
                self.skipTest(reason="Dia only supports greedy search / sampling with one sequence.")

    def test_config(self):
        self.config_tester.run_common_tests()

        # Manual testing because of composite configs
        config = self.model_tester.prepare_config_and_inputs()[0]
        self.assertTrue(hasattr(config.encoder_config, "vocab_size"), msg="Encoder `vocab_size` does not exist")
        self.assertTrue(hasattr(config.decoder_config, "vocab_size"), msg="Decoder `vocab_size` does not exist")

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_encoder_decoder_model_standalone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.check_encoder_decoder_model_standalone(*config_and_inputs)

    def _check_scores(self, batch_size, scores, generated_length, config):
        # -- Overriden as Dia internally meshes 3D with 2D
        # Special case where Dia keeps score in a 2D mesh of (bsz * channels)
        vocab_size = config.get_text_config(decoder=True).vocab_size
        expected_shape = (batch_size * len(config.delay_pattern), vocab_size)
        self.assertIsInstance(scores, tuple)
        self.assertEqual(len(scores), generated_length)
        self.assertListEqual([iter_scores.shape for iter_scores in scores], [expected_shape] * len(scores))

    # training is not supported yet
    """@unittest.skip(reason="Training is not supported yet")
    def test_training(self):
        pass

    @unittest.skip(reason="Training is not supported yet")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @parameterized.expand([("offloaded",)])
    @pytest.mark.generate
    @unittest.skip(reason="Dia doesnt work with offloaded cache implementation yet")
    def test_offloaded_cache_implementation(self, cache_implementation):
        pass"""
