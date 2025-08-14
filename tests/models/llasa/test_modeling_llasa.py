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
"""Testing suite for the PyTorch Llasa model."""

import copy
import pathlib
import tempfile
import unittest

import pytest

from transformers.models.llasa import LlasaConfig
from transformers.testing_utils import (
    cleanup,
    is_flaky,
    require_torch,
    require_torch_accelerator,
    require_torch_sdpa,
    slow,
    torch_device,
)
from transformers.utils import is_soundfile_available, is_torch_available, is_torchaudio_available
from transformers.utils.import_utils import is_datasets_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_datasets_available():
    from datasets import Audio, load_dataset

if is_torch_available():
    import torch

    from transformers import LlasaModel, LlasaForCausalLM

if is_torchaudio_available():
    import torchaudio

if is_soundfile_available():
    import soundfile as sf


@require_torch
class LlasaModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,  # need batch_size != num_hidden_layers
        seq_length=7,
        max_length=50,
        is_training=True,    # TODO, making True work
        vocab_size=100,
        hidden_size=16,
        intermediate_size=37,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=8,
        hidden_act="silu",
        eos_token_id=97,  # special tokens all occur after eos
        pad_token_id=98,
        bos_token_id=99,
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
        self.hidden_act = hidden_act
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

    def get_config(self):
        return LlasaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            hidden_act=self.hidden_act,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
        )

    def prepare_config_and_inputs(self) -> tuple[LlasaConfig, dict]:
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = input_ids.ne(self.pad_token_id)

        config = self.get_config()
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self) -> tuple[LlasaConfig, dict]:
        return self.prepare_config_and_inputs()
        

@require_torch
class LlasaModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (LlasaModel, LlasaForCausalLM,) if is_torch_available() else ()
    all_generative_model_classes = (LlasaForCausalLM,)
    pipeline_model_mapping = (
        {
            "feature-extraction": LlasaModel,
            # "text-generation": LlasaForCausalLM,
            "text-to-speech": LlasaForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    pipeline_model_mapping = {}     # to skip `resize_embeddings` tests
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = LlasaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LlasaConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        """
        Overrides [ModelTesterMixin._prepare_for_class] to handle third input_ids dimension (namely adding labels).
        """
        inputs_dict = copy.deepcopy(inputs_dict)

        if return_labels:
            inputs_dict["labels"] = torch.zeros(
                (
                    self.model_tester.batch_size,
                    self.model_tester.seq_length,
                ),
                dtype=torch.long,
                device=torch_device,
            )

        return inputs_dict


class LlasaForCausalLMIntegrationTest(unittest.TestCase):

    def setUp(self):
        # TODO exchange with official checkpoint
        self.model_checkpoint = "bezzam/Llasa-1B"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)
