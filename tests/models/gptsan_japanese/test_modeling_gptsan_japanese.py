# coding=utf-8
# Copyright 2023 Google GPTSANJapanese Authors and HuggingFace Inc. team.
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


import unittest

import numpy as np

from transformers import (
    GPTSANJapaneseConfig,
    GPTSANJapaneseForConditionalGeneration,
    GPTSANJapaneseModel,
    GPTSANJapaneseTokenizer,
    is_torch_available,
)
from transformers.testing_utils import require_torch, slow, tooslow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor


class GPTSANJapaneseTester:
    def __init__(
        self,
        parent,
        vocab_size=36000,
        batch_size=13,
        num_contexts=7,
        # For common tests
        is_training=True,
        hidden_size=32,
        ext_size=42,
        num_hidden_layers=5,
        num_ext_layers=2,
        num_attention_heads=4,
        num_experts=2,
        d_ff=32,
        d_ext=80,
        d_spout=33,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        expert_capacity=100,
        router_jitter_noise=0.0,
    ):

        self.vocab_size = vocab_size
        self.parent = parent
        self.batch_size = batch_size
        self.num_contexts = num_contexts
        # For common tests
        self.seq_length = self.num_contexts
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_ext_layers = num_ext_layers
        self.ext_size = ext_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_experts = num_experts
        self.d_ff = d_ff
        self.d_ext = d_ext
        self.d_spout = d_spout
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.expert_capacity = expert_capacity
        self.router_jitter_noise = router_jitter_noise

    def get_large_model_config(self):
        return GPTSANJapaneseConfig.from_pretrained("Tanrei/GPTSAN-japanese")

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()

        return (config, input_ids)

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()

        return (config, {"input_ids": input_ids})

    def get_config(self):
        return GPTSANJapaneseConfig(
            vocab_size=36000,
            num_contexts=self.seq_length,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_ext=self.d_ext,
            d_spout=self.d_spout,
            num_switch_layers=self.num_hidden_layers - self.num_ext_layers,
            num_ext_layers=self.num_ext_layers,
            num_heads=self.num_attention_heads,
            num_experts=self.num_experts,
            expert_capacity=self.expert_capacity,
            dropout_rate=self.dropout_rate,
            layer_norm_epsilon=self.layer_norm_epsilon,
            router_jitter_noise=self.router_jitter_noise,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
    ):
        model = GPTSANJapaneseForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids=input_ids,
        )
        assert result


@require_torch
class GPTSANJapaneseTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (GPTSANJapaneseModel,) if is_torch_available() else ()
    fx_compatible = False
    is_encoder_decoder = False
    test_pruning = False
    test_headmasking = False
    test_cpu_offload = False
    test_disk_offload = False
    test_save_load_fast_init_to_base = False
    test_training = False
    # The small GPTSAN_JAPANESE model needs higher percentages for CPU/MP tests
    model_split_percents = [0.8, 0.9]

    def setUp(self):
        self.model_tester = GPTSANJapaneseTester(self)
        self.config_tester = ConfigTester(self, config_class=GPTSANJapaneseConfig, d_model=37)

    def test_config(self):
        GPTSANJapaneseConfig()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)


@require_torch
class GPTSANJapaneseForConditionalGenerationTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):

    all_model_classes = (GPTSANJapaneseForConditionalGeneration,) if is_torch_available() else ()
    fx_compatible = False
    is_encoder_decoder = False
    test_pruning = False
    test_headmasking = False
    test_cpu_offload = False
    test_disk_offload = False
    # The small GPTSAN_JAPANESE model needs higher percentages for CPU/MP tests
    model_split_percents = [0.8, 0.9]

    def setUp(self):
        self.model_tester = GPTSANJapaneseTester(self)
        self.config_tester = ConfigTester(self, config_class=GPTSANJapaneseConfig, d_model=37)

    def test_config(self):
        GPTSANJapaneseConfig()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_logits(self):
        model = GPTSANJapaneseForConditionalGeneration.from_pretrained("Tanrei/GPTSAN-japanese")
        tokenizer = GPTSANJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        input_ids = tokenizer.encode("武田信玄は", return_tensors="pt")
        outputs = model(input_ids)
        output_logits = outputs.logits.detach().cpu().numpy()
        # Output of original model created with mesh-tensoflow
        target = [
            [
                -12.037839889526367,
                -12.433061599731445,
                -14.333840370178223,
                -12.450345993041992,
                -11.1661376953125,
                -11.930137634277344,
                -10.659740447998047,
                -12.909574508666992,
                -13.241043090820312,
                -13.398579597473145,
                -11.107524871826172,
                -12.3685941696167,
                -22.97943115234375,
                -10.481067657470703,
                -12.484030723571777,
                -12.807360649108887,
                -14.769700050354004,
                -12.233579635620117,
                -13.428145408630371,
                -22.624177932739258,
            ],
            [
                -7.511149883270264,
                -8.281851768493652,
                -7.943127155303955,
                -7.55021333694458,
                -6.49869966506958,
                -7.586796283721924,
                -6.978085994720459,
                -7.839145183563232,
                -8.21964168548584,
                -8.695091247558594,
                -6.706910610198975,
                -6.6585798263549805,
                -19.565698623657227,
                -5.353842735290527,
                -8.350686073303223,
                -8.039388656616211,
                -10.856569290161133,
                -7.75154447555542,
                -8.819022178649902,
                -19.51532745361328,
            ],
            [
                -9.73066234588623,
                -10.223922729492188,
                -9.932981491088867,
                -11.857836723327637,
                -7.662626266479492,
                -11.13529109954834,
                -7.765097618103027,
                -11.472923278808594,
                -9.543149948120117,
                -11.905633926391602,
                -9.366164207458496,
                -11.5734281539917,
                -23.699003219604492,
                -9.429590225219727,
                -10.42839241027832,
                -10.585240364074707,
                -10.94771957397461,
                -11.095416069030762,
                -10.390240669250488,
                -23.769372940063477,
            ],
            [
                -9.728265762329102,
                -9.859712600708008,
                -10.09729290008545,
                -9.678522109985352,
                -6.879519939422607,
                -9.68487548828125,
                -4.2803425788879395,
                -10.018914222717285,
                -9.308445930480957,
                -10.63394546508789,
                -8.083646774291992,
                -9.06301498413086,
                -21.904266357421875,
                -8.90160846710205,
                -8.841876029968262,
                -11.856719970703125,
                -12.079398155212402,
                -11.233753204345703,
                -10.177338600158691,
                -21.87256622314453,
            ],
            [
                -9.669764518737793,
                -9.614198684692383,
                -9.814510345458984,
                -9.996501922607422,
                -11.375690460205078,
                -10.113405227661133,
                -10.546867370605469,
                -10.04369068145752,
                -10.907809257507324,
                -10.504216194152832,
                -11.129199028015137,
                -10.151124000549316,
                -21.96586799621582,
                -9.086349487304688,
                -11.730339050292969,
                -10.460667610168457,
                -10.298049926757812,
                -10.784148216247559,
                -10.840693473815918,
                -22.03152847290039,
            ],
        ]
        target = np.array(target).flatten()
        predict = output_logits[0, :, :20].flatten()

        def check(a, b, epsilon=5e-4):
            return abs(a - b) < epsilon * max(abs(a), abs(b))

        assert np.all([check(target[i], predict[i]) for i in range(len(target))])

    @tooslow
    def test_sample(self):
        model = GPTSANJapaneseForConditionalGeneration.from_pretrained("Tanrei/GPTSAN-japanese")
        tokenizer = GPTSANJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        # Output of original model created with mesh-tensoflow
        target = [
            ("武田信玄は", 35675),
            ("武田信玄は、", 45),
            ("武田信玄は、この", 29),
            ("武田信玄は、このよう", 30642),
            ("武田信玄は、このような", 35680),
            ("武田信玄は、このような「", 8640),
            ("武田信玄は、このような「武田", 31617),
            ("武田信玄は、このような「武田家", 30646),
            ("武田信玄は、このような「武田家の", 31617),
            ("武田信玄は、このような「武田家の家", 31381),
        ]
        for input, output in target:
            input_ids = tokenizer.encode(input, return_tensors="pt")
            outputs = model(input_ids)
            output_logits = outputs.logits.detach().cpu().numpy()[0]
            output_id = np.argmax(output_logits[-1])
            assert output_id == output
