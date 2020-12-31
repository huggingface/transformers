#!/usr/bin/env python3
# coding=utf-8
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
"""Tests for BlenderBot"""
import unittest

from transformers import is_torch_available
from transformers.file_utils import cached_property
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        BlenderbotConfig,
        BlenderbotForConditionalGeneration,
        BlenderbotModel,
        BlenderbotSmallTokenizer,
        BlenderbotTokenizer,
    )

TOK_DECODE_KW = dict(skip_special_tokens=True, clean_up_tokenization_spaces=True)
FASTER_GEN_KWARGS = dict(num_beams=1, early_stopping=True, min_length=15, max_length=25)


@require_torch
class BlenderbotModelTester:
    # Required attributes
    vocab_size = 99
    batch_size = 13
    seq_length = 7
    num_hidden_layers = 2
    hidden_size = 16
    num_attention_heads = 4
    is_training = True

    def __init__(self, parent):
        torch.manual_seed(0)
        self.parent = parent
        self.config = BlenderbotConfig(
            d_model=self.hidden_size,
            dropout=0.0,
            activation_function="gelu",
            vocab_size=self.vocab_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            attention_dropout=0.0,
            encoder_ffn_dim=4,
            decoder_ffn_dim=4,
            do_blenderbot_90_layernorm=False,
            normalize_before=True,
            max_position_embeddings=50,
            static_position_embeddings=False,
            scale_embedding=True,
            bos_token_id=0,
            eos_token_id=2,
            pad_token_id=1,
            num_beams=1,
            min_length=3,
            max_length=10,
        )

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return self.config, inputs_dict


@require_torch
class BlenderbotTesterMixin(ModelTesterMixin, unittest.TestCase):
    if is_torch_available():
        all_generative_model_classes = (BlenderbotForConditionalGeneration,)
        all_model_classes = (BlenderbotForConditionalGeneration, BlenderbotModel)
    else:
        all_generative_model_classes = ()
        all_model_classes = ()
    is_encoder_decoder = True
    test_head_masking = False
    test_pruning = False
    test_missing_keys = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = BlenderbotModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BlenderbotConfig)

    def test_initialization_module(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = BlenderbotForConditionalGeneration(config).model
        model.to(torch_device)
        model.eval()
        enc_embeds = model.encoder.embed_tokens.weight
        assert (enc_embeds == model.shared.weight).all().item()
        self.assertAlmostEqual(torch.std(enc_embeds).item(), config.init_std, 2)

    def test_embed_pos_shape(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = BlenderbotForConditionalGeneration(config)
        expected_shape = (config.max_position_embeddings + config.extra_pos_embeddings, config.d_model)
        assert model.model.encoder.embed_positions.weight.shape == expected_shape
        model.model.decoder.embed_positions.weight.shape == expected_shape

    @unittest.skip("This test is flaky")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip("TODO: Decoder embeddings cannot be resized at the moment")
    def test_resize_embeddings_untied(self):
        pass


@unittest.skipUnless(torch_device != "cpu", "3B test too slow on CPU.")
@require_torch
@require_sentencepiece
@require_tokenizers
class Blenderbot3BIntegrationTests(unittest.TestCase):
    ckpt = "facebook/blenderbot-3B"

    @cached_property
    def tokenizer(self):
        return BlenderbotTokenizer.from_pretrained(self.ckpt)

    @slow
    def test_generation_from_short_input_same_as_parlai_3B(self):
        torch.cuda.empty_cache()
        model = BlenderbotForConditionalGeneration.from_pretrained(self.ckpt).half().to(torch_device)

        src_text = ["Sam"]
        model_inputs = self.tokenizer(src_text, return_tensors="pt").to(torch_device)

        generated_utterances = model.generate(**model_inputs, **FASTER_GEN_KWARGS)
        tgt_text = 'Sam is a great name. It means "sun" in Gaelic.'

        generated_txt = self.tokenizer.batch_decode(generated_utterances, **TOK_DECODE_KW)
        assert generated_txt[0].strip() == tgt_text

        src_text = "Social anxiety\nWow, I am never shy. Do you have anxiety?\nYes. I end up sweating and blushing and feel like i'm going to throw up.\nand why is that?"

        model_inputs = self.tokenizer([src_text], return_tensors="pt").to(torch_device)

        generated_ids = model.generate(**model_inputs, **FASTER_GEN_KWARGS)[0]
        reply = self.tokenizer.decode(generated_ids, **TOK_DECODE_KW)

        assert "I think it's because we are so worried about what people think of us." == reply.strip()
        del model


@require_torch
class Blenderbot90MIntegrationTests(unittest.TestCase):
    ckpt = "facebook/blenderbot-90M"

    @cached_property
    def model(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.ckpt).to(torch_device)
        if torch_device == "cuda":
            model = model.half()
        return model

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.ckpt)

    @slow
    def test_90_generation_from_long_input(self):

        src_text = [
            "Social anxiety\nWow, I am never shy. Do you have anxiety?\nYes. I end up sweating and blushing and feel like\
       i'm going to throw up.\nand why is that?"
        ]

        model_inputs = self.tokenizer(src_text, return_tensors="pt").to(torch_device)

        # model does not have "token_type_ids"
        model_inputs.pop("token_type_ids")
        assert isinstance(self.tokenizer, BlenderbotSmallTokenizer)
        generated_ids = self.model.generate(**model_inputs)[0]
        reply = self.tokenizer.decode(generated_ids, **TOK_DECODE_KW)

        assert reply in (
            "i don't know. i just feel like i'm going to throw up. it's not fun.",
            "i'm not sure. i just feel like i've been feeling like i have to be in a certain place",
        )

    def test_90_generation_from_short_input(self):
        model_inputs = self.tokenizer(["sam"], return_tensors="pt").to(torch_device)

        # model does not have "token_type_ids"
        model_inputs.pop("token_type_ids")
        generated_utterances = self.model.generate(**model_inputs)

        clean_txt = self.tokenizer.decode(generated_utterances[0], **TOK_DECODE_KW)
        assert clean_txt in (
            "have you ever been to a sam club? it's a great club in the south.",
            "have you ever heard of sam harris? he's an american singer, songwriter, and actor.",
        )
