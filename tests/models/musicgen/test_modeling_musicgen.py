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
import tempfile
import unittest

from transformers import MusicgenConfig, is_torch_available, MusicgenEncoderConfig, MusicgenDecoderConfig
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device
from transformers.utils import cached_property

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        MusicgenForConditionalGeneration,
        MusicgenModel,
    )
    from transformers.models.musicgen.modeling_musicgen import MusicgenDecoder, MusicgenEncoder


def prepare_musicgen_inputs_dict(
    config,
    input_ids,
    decoder_input_ids,
    attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if attention_mask is None:
        attention_mask = input_ids.ne(config.decoder_config.pad_token_id)
    if head_mask is None:
        head_mask = torch.ones(config.encoder_config.num_layers, config.encoder_config.num_heads, device=torch_device)
    if decoder_head_mask is None:
        decoder_head_mask = torch.ones(config.decoder_config.num_layers, config.decoder_config.num_heads, device=torch_device)
    if cross_attn_head_mask is None:
        cross_attn_head_mask = torch.ones(config.decoder_config.num_layers, config.decoder_config.num_heads, device=torch_device)
    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
    }


class MusicgenModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        is_training=False,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=100,
        pad_token_id=0,
        bos_token_id=0,
        num_codebooks=4,
        relative_attention_num_buckets=8,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.num_codebooks = num_codebooks
        self.relative_attention_num_buckets = relative_attention_num_buckets

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        decoder_input_ids = ids_tensor([self.batch_size, self.num_codebooks, self.seq_length], self.vocab_size)

        config = self.get_config()
        inputs_dict = prepare_musicgen_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def get_config(self):
        encoder_config = MusicgenEncoderConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.intermediate_size,
            d_kv=self.hidden_size // self.num_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
        )
        decoder_config = MusicgenDecoderConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.intermediate_size,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            num_codebooks=self.num_codebooks,
        )
        return MusicgenConfig.from_encoder_decoder_configs(encoder_config, decoder_config)

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def check_encoder_decoder_model_standalone(self, config, inputs_dict):
        model = MusicgenModel(config=config).to(torch_device).eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = MusicgenEncoder.from_pretrained(tmpdirname).to(torch_device)

        encoder_last_hidden_state_2 = encoder(inputs_dict["input_ids"], attention_mask=inputs_dict["attention_mask"])[
            0
        ]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = MusicgenDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            input_ids=inputs_dict["decoder_input_ids"],
            attention_mask=inputs_dict["decoder_attention_mask"],
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=inputs_dict["attention_mask"],
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)


@require_torch
class MusicgenModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (MusicgenModel, MusicgenForConditionalGeneration)
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (MusicgenForConditionalGeneration,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = MusicgenModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=MusicgenConfig, has_text_modality=False, hidden_size=16
        )

    def _get_input_ids_and_config(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict["decoder_input_ids"]

        # cut to half length & take max batch_size 3
        _, codebooks, sequence_length = input_ids.shape
        input_ids = input_ids[:batch_size, :codebooks, :sequence_length // 2]

        # generate max 3 tokens
        max_length = input_ids.shape[-1] + 3
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)[:batch_size, :sequence_length]
        return config, input_ids, attention_mask, max_length
