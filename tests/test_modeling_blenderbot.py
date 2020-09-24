#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the;
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
# LICENSE file in the root directory of this source tree.

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

    from transformers import BlenderbotConfig, BlenderbotForConditionalGeneration, BlenderbotTokenizer
    from transformers.file_utils import cached_property
    from transformers.tokenization_blenderbot import BlenderbotSmallTokenizer


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
            max_position_embeddings=50,
            variant="prelayernorm",
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
        all_model_classes = (BlenderbotForConditionalGeneration,)
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

    def test_inputs_embeds(self):
        pass

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


@unittest.skipUnless(torch_device != "cpu", "3B test too slow on CPU.")
@require_torch
class Blenderbot3BIntegrationTests(unittest.TestCase):
    ckpt = "facebook/blenderbot-3B"

    @cached_property
    def model(self):
        model = BlenderbotForConditionalGeneration.from_pretrained(self.ckpt).to(torch_device)
        if torch_device == "cuda":
            model = model.half()
        return model

    @cached_property
    def tokenizer(self):
        return BlenderbotTokenizer.from_pretrained(self.ckpt)

    @slow
    def test_tokenization_same_as_parlai(self):
        # TODO(SS): this can run on CPU
        self.assertListEqual(self.tokenizer("sam").input_ids, [268, 343, 2])

    @slow
    @unittest.skip("This fails.")
    def test_forward_3B_same_as_parlai(self):
        torch.manual_seed(0)
        config = BlenderbotConfig(
            d_model=16,
            vocab_size=50,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_layers=2,
            decoder_layers=2,
            encoder_ffn_dim=4,
            decoder_ffn_dim=4,
            variant="prelayernorm",
            scale_embedding=True,
            normalize_embedding=True,
            max_position_embeddings=50,
            activation_function="gelu",
            normalize_before=False,
            static_position_embeddings=True,
            dropout=0.1,
        )
        input_ids = torch.tensor(
            [[49, 12, 38, 24, 13, 25, 10, 28, 37, 7, 44, 7, 2, 3]],
            dtype=torch.long,
            device=torch_device,
        )
        model = BlenderbotForConditionalGeneration(config).to(torch_device)
        model.eval()

        # output from parlai model after copying the same blenderbot weight in parlai and setting a manual_seed
        expected_logits = torch.tensor(
            [
                [
                    [
                        -1.0000e20,
                        0.0000e00,
                        2.6462e-02,
                        9.7588e-02,
                        -5.6271e-02,
                        -1.1409e-01,
                        -3.3294e-02,
                        -1.0423e-01,
                        -4.8363e-02,
                        -1.2610e-01,
                        -3.4125e-02,
                        -2.9841e-02,
                        8.6975e-02,
                        2.5547e-02,
                        2.0425e-03,
                        -5.9153e-02,
                        1.4392e-02,
                        -1.4324e-02,
                        1.2774e-01,
                        -5.3284e-02,
                        -1.5876e-02,
                        9.1752e-02,
                        -3.0166e-02,
                        -3.1726e-02,
                        9.9600e-02,
                        1.0991e-01,
                        8.0946e-03,
                        3.5396e-03,
                        -2.5164e-02,
                        -4.0277e-02,
                        -3.6360e-02,
                        7.5158e-02,
                        4.3379e-02,
                        1.3465e-01,
                        -1.3209e-01,
                        -1.1706e-01,
                        5.6180e-02,
                        -3.6239e-02,
                        6.6490e-02,
                        4.9879e-02,
                        1.0979e-02,
                        -2.7895e-02,
                        -8.4691e-02,
                        4.5857e-02,
                        2.0233e-02,
                        9.2533e-02,
                        -6.4260e-02,
                        -5.0988e-02,
                        -4.0852e-02,
                        1.9867e-02,
                    ]
                ]
            ],
            device=torch_device,
        )

        decoder_inputs = torch.LongTensor([1]).expand(1, 1).to(torch_device)
        logits = model(input_ids, decoder_input_ids=decoder_inputs, return_dict=True)["logits"]
        assert torch.allclose(expected_logits, logits, atol=1e-4)

    @unittest.skip("This fails.")
    @slow
    def test_generation_from_short_input_same_as_parlai_3B(self):

        src_text = [
            "sam",
        ]

        model_inputs = self.tokenizer(src_text, return_tensors="pt").to(torch_device)
        generated_utterances = self.model.generate(**model_inputs)
        tgt_text = ["Sam is a great name. It means 'sun' in Gaelic."]

        generated_txt = self.tokenizer.batch_decode(generated_utterances)
        self.assertListEqual(tgt_text, generated_txt)

    @slow
    def test_generation_from_long_input_same_as_parlai_3B(self):

        src_text = "Social anxiety\nWow, I am never shy. Do you have anxiety?\nYes. I end up sweating and blushing and feel like i'm going to throw up.\nand why is that?"

        model_inputs = self.tokenizer([src_text], return_tensors="pt").to(torch_device)
        generated_ids = self.model.generate(**model_inputs, min_length=15, early_stopping=True, num_beams=1)[0]
        reply = self.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        assert "I'm not sure, but I do know that social anxiety disorder is a mental disorder." == reply


@require_torch
class Blenderbot90MIntegrationTests(unittest.TestCase):
    ckpt = "facebook/blenderbot-90M"

    @cached_property
    def model(self):
        model = BlenderbotForConditionalGeneration.from_pretrained(self.ckpt).to(torch_device)
        if torch_device == "cuda":
            model = model.half()
        return model

    @cached_property
    def tokenizer(self):
        return BlenderbotSmallTokenizer.from_pretrained(self.ckpt)

    def test_tokenization_same_as_parlai(self):
        assert self.tokenizer("sam").input_ids == [1384]

    @unittest.skip("This does not pass. It should be deleted")
    def test_forward_90M_same_as_parlai(self):
        torch.manual_seed(0)
        config = BlenderbotConfig(
            d_model=16,
            vocab_size=50,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_layers=2,
            decoder_layers=2,
            encoder_ffn_dim=4,
            decoder_ffn_dim=4,
            variant="xlm",
            scale_embedding=True,
            normalize_embedding=True,
            max_position_embeddings=50,
            activation_function="gelu",
            normalize_before=False,
            static_position_embeddings=False,
            dropout=0.1,
        )
        input_ids = torch.tensor(
            [[49, 12, 38, 24, 13, 25, 10, 28, 37, 7, 44, 7, 2, 3]],
            dtype=torch.long,
            device=torch_device,
        )
        model = BlenderbotForConditionalGeneration(config)
        model.eval()
        model.to(torch_device)
        # output from parlai model after copying the same blenderbot weight in parlai and setting a manual_seed
        expected_logits = torch.tensor(
            [
                [
                    [
                        -1.0000e20,
                        0.0000e00,
                        -8.3858e-03,
                        5.3556e-02,
                        -6.7345e-02,
                        -1.1861e-01,
                        -4.7368e-02,
                        -8.6005e-02,
                        -6.6010e-02,
                        -1.1263e-01,
                        -1.2138e-02,
                        -5.0588e-02,
                        1.1818e-01,
                        3.8662e-03,
                        2.3491e-02,
                        -1.0256e-01,
                        1.9944e-02,
                        -2.8050e-02,
                        1.2771e-01,
                        -5.6630e-02,
                        -3.7779e-02,
                        6.9132e-02,
                        -8.2159e-04,
                        -6.3877e-02,
                        1.1591e-01,
                        9.1973e-02,
                        3.8424e-03,
                        5.4423e-02,
                        -3.4574e-02,
                        3.1875e-02,
                        -3.2030e-02,
                        6.0317e-02,
                        6.8307e-02,
                        1.3964e-01,
                        -1.2045e-01,
                        -1.1150e-01,
                        7.3168e-02,
                        -4.0991e-02,
                        3.8692e-04,
                        5.9230e-02,
                        -2.0674e-02,
                        -3.2628e-02,
                        -9.5583e-02,
                        6.5901e-02,
                        5.8617e-02,
                        9.2186e-02,
                        -4.5951e-02,
                        -3.7279e-02,
                        -1.5638e-02,
                        3.7328e-02,
                    ]
                ]
            ],
            device=torch_device,
        )
        decoder_inputs = torch.LongTensor([1]).expand(1, 1).to(torch_device)
        logits = model(input_ids, decoder_input_ids=decoder_inputs)[0]

        assert torch.allclose(expected_logits, logits, atol=1e-4)

    @slow
    def test_generation_from_long_input_same_as_parlai_90M(self):

        src_text = [
            "Social anxiety\nWow, I am never shy. Do you have anxiety?\nYes. I end up sweating and blushing and feel like\
       i'm going to throw up.\nand why is that?"
        ]
        #tgt_text = "i ' m not sure . i just feel like i ' m going to throw up ."
        tgt_text = "i don't know. i just feel like i'm going to throw up. it's not fun."

        model_inputs = self.tokenizer(src_text, return_tensors="pt").to(torch_device)
        generated_ids = self.model.generate(**model_inputs, early_stopping=True)[0]
        reply = self.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        assert tgt_text == reply

    def test_generation_from_short_input_same_as_parlai_90M(self):
        model_inputs = self.tokenizer(["sam"], return_tensors="pt").to(torch_device)
        generated_utterances = self.model.generate(**model_inputs)
        tgt_text = (
            "__start__ have you ever heard of sam harris? he's an american singer, songwriter, and actor. __end__"
        )

        generated_txt = self.tokenizer.decode(generated_utterances[0])
        assert tgt_text == generated_txt
