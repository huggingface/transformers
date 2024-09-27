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
"""Testing suite for the PyTorch Moshi model."""

import copy
import math
import unittest

import numpy as np

from transformers import (
    MoshiConfig,
    PretrainedConfig,
)
from transformers.testing_utils import (
    is_torch_available,
    require_torch,
    torch_device,
)
from transformers.utils import cached_property

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        AutoFeatureExtractor,
        AutoTokenizer,
        MoshiForCausalLM,
        MoshiForConditionalGeneration,
        MoshiModel,
    )


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if "_range" in key or "_std" in key or "initializer_factor" in key or "layer_scale" in key:
            setattr(configs_no_init, key, 1e-10)
        if isinstance(getattr(configs_no_init, key, None), PretrainedConfig):
            no_init_subconfig = _config_zero_init(getattr(configs_no_init, key))
            setattr(configs_no_init, key, no_init_subconfig)
    return configs_no_init


class MoshiDecoderTester:
    def __init__(
        self,
        parent,
        batch_size=4,  # need batch_size != num_hidden_layers
        seq_length=7,
        is_training=True,
        vocab_size=99,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=100,
        pad_token_id=99,
        bos_token_id=99,
        num_codebooks=4,
        audio_encoder_type="mimi",
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
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
        self.audio_encoder_type = audio_encoder_type

    def prepare_config_and_inputs(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        input_ids = ids_tensor([batch_size, self.seq_length], self.vocab_size)
        config = self.get_config()

        attention_mask = input_ids.ne(config.pad_token_id)

        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict

    def get_config(self):
        config = MoshiConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            d_ff=self.intermediate_size,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.bos_token_id,
            bos_token_id=self.bos_token_id,
            num_codebooks=self.num_codebooks,
            tie_word_embeddings=False,
            audio_encoder={"model_type": self.audio_encoder_type},
        )
        return config

    def prepare_config_and_inputs_for_common(self, batch_size=None):
        config, inputs_dict = self.prepare_config_and_inputs(batch_size)
        return config, inputs_dict


@require_torch
class MoshiDecoderTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (MoshiModel, MoshiForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (
        (MoshiForCausalLM,) if is_torch_available() else ()
    )  # we don't want to run all the generation tests, only a specific subset
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    pipeline_model_mapping = (
        {
            "feature-extraction": MoshiModel,
            "text-generation": MoshiForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    def setUp(self):
        self.model_tester = MoshiDecoderTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=MoshiConfig,
            hidden_size=16,
            audio_encoder={"model_type": self.model_tester.audio_encoder_type},
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="The MimiModel does not have support dynamic compile yet")
    def test_sdpa_can_compile_dynamic(self):
        pass

    def _get_input_ids_and_config(self, batch_size=1):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common(batch_size)
        input_ids = inputs_dict.pop("input_ids").to(torch_device)
        attention_mask = inputs_dict.pop("attention_mask").to(torch_device)

        return config, input_ids, attention_mask, inputs_dict

    def _get_logits_processor_kwargs(self, do_sample=False, config=None):
        logits_processor_kwargs = {}
        return logits_processor_kwargs


def prepare_moshi_inputs_dict(
    config,
    input_ids,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
    labels=None,
):
    if decoder_attention_mask is None:
        decoder_attention_mask = decoder_input_ids.reshape(
            -1, config.decoder.num_codebooks, decoder_input_ids.shape[-1]
        )[:, 0, :]
        decoder_attention_mask = decoder_attention_mask.ne(config.decoder.pad_token_id)
    if head_mask is None:
        head_mask = torch.ones(
            config.text_encoder.num_hidden_layers, config.text_encoder.num_attention_heads, device=torch_device
        )
    if decoder_head_mask is None:
        decoder_head_mask = torch.ones(
            config.decoder.num_hidden_layers, config.decoder.num_attention_heads, device=torch_device
        )
    if cross_attn_head_mask is None:
        cross_attn_head_mask = torch.ones(
            config.decoder.num_hidden_layers, config.decoder.num_attention_heads, device=torch_device
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
        "labels": labels,
    }


class MoshiTester:
    def __init__(
        self,
        parent,
        batch_size=4,  # need batch_size != num_hidden_layers
        seq_length=7,
        is_training=True,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=100,
        pad_token_id=99,
        bos_token_id=99,
        num_codebooks=4,
        num_filters=4,
        codebook_size=128,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
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
        self.num_filters = num_filters
        self.codebook_size = codebook_size

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        decoder_input_ids = ids_tensor([self.batch_size * self.num_codebooks, self.seq_length], self.vocab_size)

        config = self.get_config()
        inputs_dict = prepare_moshi_inputs_dict(config, input_ids, decoder_input_ids=decoder_input_ids)
        return config, inputs_dict

    def get_config(self):
        config = MoshiConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            ffn_dim=self.intermediate_size,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.bos_token_id,
            bos_token_id=self.bos_token_id,
            num_codebooks=self.num_codebooks,
            tie_word_embeddings=False,
        )
        return config

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict


@require_torch
class MoshiTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (MoshiForConditionalGeneration,) if is_torch_available() else ()
    greedy_sample_model_classes = (MoshiForConditionalGeneration,) if is_torch_available() else ()
    # TODO: test generation
    test_pruning = False  # training is not supported yet for Moshi
    test_headmasking = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = MoshiTester(self)

    # special case for labels
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            inputs_dict["labels"] = torch.zeros(
                (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_codebooks),
                dtype=torch.long,
                device=torch_device,
            )
        return inputs_dict


def get_bip_bip(bip_duration=0.125, duration=0.5, sample_rate=32000):
    """Produces a series of 'bip bip' sounds at a given frequency."""
    timesteps = np.arange(int(duration * sample_rate)) / sample_rate
    wav = np.cos(2 * math.pi * 440 * timesteps)
    time_period = (timesteps % (2 * bip_duration)) / (2 * bip_duration)
    envelope = time_period >= 0.5
    return wav * envelope


def place_dict_on_device(dict_to_place, device):
    for key in dict_to_place:
        if dict_to_place[key] is not None and isinstance(dict_to_place[key], torch.Tensor):
            dict_to_place[key] = dict_to_place[key].to(device)
    return dict_to_place


@require_torch
class MoshiIntegrationTests(unittest.TestCase):
    @cached_property
    def model(self):
        return MoshiForConditionalGeneration.from_pretrained("kmhf/hf-moshiko").to(torch_device)

    @cached_property
    def feature_extractor(self):
        return AutoFeatureExtractor.from_pretrained("kmhf/hf-moshiko")

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained("kmhf/hf-moshiko")

    # TODO: also test moshika
