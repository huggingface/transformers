# coding=utf-8
# Copyright 2021, The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch MBART model. """
import copy
import unittest

from transformers import MusicgenDecoderConfig, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        MusicgenForCausalLM,
        MusicgenModel,
    )


def prepare_musicgen_decoder_inputs_dict(
    config,
    input_ids,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    cross_attn_head_mask=None,
):
    if attention_mask is None:
        attention_mask = input_ids[:, 0, :].ne(config.pad_token_id)
    if head_mask is None:
        head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads, device=torch_device)
    if encoder_attention_mask is None and encoder_hidden_states is not None:
        encoder_attention_mask = torch.ones(encoder_hidden_states.shape[:2], device=torch_device)
    if cross_attn_head_mask is None:
        cross_attn_head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads, device=torch_device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "head_mask": head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
    }


class MusicgenDecoderTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        is_training=False,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=100,
        pad_token_id=0,
        bos_token_id=0,
        num_codebooks=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.num_codebooks = num_codebooks

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.num_codebooks, self.seq_length], self.vocab_size)
        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])

        config = self.get_config()
        inputs_dict = prepare_musicgen_decoder_inputs_dict(
            config, input_ids, encoder_hidden_states=encoder_hidden_states
        )
        return config, inputs_dict

    def get_config(self):
        config = MusicgenDecoderConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            d_ff=self.intermediate_size,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            num_codebooks=self.num_codebooks,
            tie_word_embeddings=False,
        )
        return config

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict


@require_torch
class MusicgenModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (MusicgenModel, MusicgenForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (MusicgenForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "text-generation": MusicgenForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    def setUp(self):
        self.model_tester = MusicgenDecoderTester(self)
        self.config_tester = ConfigTester(self, config_class=MusicgenDecoderConfig, hidden_size=16)

    def test_config(self):
        self.config_tester.run_common_tests()

    def _get_input_ids_and_config(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict["input_ids"]

        # cut to half length & take max batch_size 3
        _, codebooks, sequence_length = input_ids.shape
        input_ids = input_ids[:batch_size, :codebooks, : sequence_length // 2]

        # generate max 3 tokens
        max_length = input_ids.shape[-1] + 3
        attention_mask = torch.ones((batch_size, sequence_length // 2), dtype=torch.long)
        return config, input_ids, attention_mask, max_length

    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            input_ids = inputs["input_ids"]
            del inputs["input_ids"]

            embed_tokens = model.get_input_embeddings()

            inputs["inputs_embeds"] = sum(
                [embed_tokens[codebook](input_ids[:, codebook]) for codebook in range(config.num_codebooks)]
            )

            with torch.no_grad():
                model(**inputs)[0]
