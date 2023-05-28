# coding=utf-8
# Copyright 2023 Toshiyuki Sakamoto(tanreinama) and HuggingFace Inc. team.
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
    GPTSanJapaneseConfig,
    GPTSanJapaneseForConditionalGeneration,
    GPTSanJapaneseModel,
    GPTSanJapaneseTokenizer,
    is_torch_available,
)
from transformers.generation import GenerationConfig
from transformers.testing_utils import require_torch, slow, tooslow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


class GPTSanJapaneseTester:
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
        return GPTSanJapaneseConfig.from_pretrained("Tanrei/GPTSAN-japanese")

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()

        return (config, input_ids)

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()

        return (config, {"input_ids": input_ids})

    def get_config(self):
        return GPTSanJapaneseConfig(
            vocab_size=self.vocab_size,
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
        model = GPTSanJapaneseForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids=input_ids,
        )
        self.parent.assertIsNotNone(result)


@require_torch
class GPTSanJapaneseTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (GPTSanJapaneseModel,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "conversational": GPTSanJapaneseForConditionalGeneration,
            "feature-extraction": GPTSanJapaneseForConditionalGeneration,
            "summarization": GPTSanJapaneseForConditionalGeneration,
            "text2text-generation": GPTSanJapaneseForConditionalGeneration,
            "translation": GPTSanJapaneseForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
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

    # TODO: Fix the failed tests when this model gets more usage
    def is_pipeline_test_to_skip(
        self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name
    ):
        if pipeline_test_casse_name == "SummarizationPipelineTests":
            # TODO: fix `_reorder_cache` is not implemented for this model
            return True
        elif pipeline_test_casse_name == "Text2TextGenerationPipelineTests":
            # TODO: check this.
            return True

        return False

    def setUp(self):
        self.model_tester = GPTSanJapaneseTester(self)
        self.config_tester = ConfigTester(self, config_class=GPTSanJapaneseConfig, d_model=37)

    def test_config(self):
        GPTSanJapaneseConfig()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(
        reason="skip for now as the computed `max_memory` by `model_split_percents` in the test method will be changed inside `from_pretrained`"
    )
    def test_model_parallelism(self):
        super().test_model_parallelism()


@require_torch
class GPTSanJapaneseForConditionalGenerationTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (GPTSanJapaneseForConditionalGeneration,) if is_torch_available() else ()
    fx_compatible = False
    is_encoder_decoder = False
    test_pruning = False
    test_headmasking = False
    test_cpu_offload = False
    test_disk_offload = False
    # The small GPTSAN_JAPANESE model needs higher percentages for CPU/MP tests
    model_split_percents = [0.8, 0.9]

    def setUp(self):
        self.model_tester = GPTSanJapaneseTester(self)
        self.config_tester = ConfigTester(self, config_class=GPTSanJapaneseConfig, d_model=37)

    def test_config(self):
        GPTSanJapaneseConfig()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(
        reason="skip for now as the computed `max_memory` by `model_split_percents` in the test method will be changed inside `from_pretrained`"
    )
    def test_model_parallelism(self):
        super().test_model_parallelism()

    @slow
    def test_logits(self):
        model = GPTSanJapaneseForConditionalGeneration.from_pretrained("Tanrei/GPTSAN-japanese")
        tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        input_ids = tokenizer.encode("武田信玄は", return_tensors="pt")
        outputs = model(input_ids)
        output_logits = outputs.logits.detach().cpu().numpy()
        # Output of original model created with mesh-tensoflow
        target = [
            # fmt: off
            [-12.037839889526367, -12.433061599731445, -14.333840370178223, -12.450345993041992, -11.1661376953125,
            -11.930137634277344, -10.659740447998047, -12.909574508666992, -13.241043090820312, -13.398579597473145,
            -11.107524871826172, -12.3685941696167, -22.97943115234375, -10.481067657470703, -12.484030723571777,
            -12.807360649108887, -14.769700050354004, -12.233579635620117, -13.428145408630371, -22.624177932739258],
            [-7.511149883270264, -8.281851768493652, -7.943127155303955, -7.55021333694458, -6.49869966506958,
            -7.586796283721924, -6.978085994720459, -7.839145183563232, -8.21964168548584, -8.695091247558594,
            -6.706910610198975, -6.6585798263549805, -19.565698623657227, -5.353842735290527, -8.350686073303223,
            -8.039388656616211, -10.856569290161133, -7.75154447555542, -8.819022178649902, -19.51532745361328],
            [-9.73066234588623, -10.223922729492188, -9.932981491088867, -11.857836723327637, -7.662626266479492,
            -11.13529109954834, -7.765097618103027, -11.472923278808594, -9.543149948120117, -11.905633926391602,
            -9.366164207458496, -11.5734281539917, -23.699003219604492, -9.429590225219727, -10.42839241027832,
            -10.585240364074707, -10.94771957397461, -11.095416069030762, -10.390240669250488, -23.769372940063477],
            [-9.728265762329102, -9.859712600708008, -10.09729290008545, -9.678522109985352, -6.879519939422607,
            -9.68487548828125, -4.2803425788879395, -10.018914222717285, -9.308445930480957, -10.63394546508789,
            -8.083646774291992, -9.06301498413086, -21.904266357421875, -8.90160846710205, -8.841876029968262,
            -11.856719970703125, -12.079398155212402, -11.233753204345703, -10.177338600158691, -21.87256622314453],
            [-9.669764518737793, -9.614198684692383, -9.814510345458984, -9.996501922607422, -11.375690460205078,
            -10.113405227661133, -10.546867370605469, -10.04369068145752, -10.907809257507324, -10.504216194152832,
            -11.129199028015137, -10.151124000549316, -21.96586799621582, -9.086349487304688, -11.730339050292969,
            -10.460667610168457, -10.298049926757812, -10.784148216247559, -10.840693473815918, -22.03152847290039],
            # fmt: on
        ]
        target = np.array(target).flatten()
        predict = output_logits[0, :, :20].flatten()

        def check(a, b, epsilon=5e-4):
            return abs(a - b) < epsilon * max(abs(a), abs(b))

        self.assertTrue(np.all([check(target[i], predict[i]) for i in range(len(target))]))

    @slow
    def test_batch_generation(self):
        model = GPTSanJapaneseForConditionalGeneration.from_pretrained("Tanrei/GPTSAN-japanese")
        tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        model.to(torch_device)

        # set deterministically
        generation_config = GenerationConfig.from_pretrained("Tanrei/GPTSAN-japanese")
        generation_config.top_k = 1

        # use different length sentences to test batching
        sentences = [
            "甲斐なら武田と言うほど",
            "織田信長は、",
        ]

        tokenizer.padding_side = "left"
        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(torch_device)

        self.assertNotEqual(inputs["attention_mask"][0].numpy().tolist(), inputs["attention_mask"][1].numpy().tolist())

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"].to(torch_device),
            max_new_tokens=3,
            generation_config=generation_config,
        )

        inputs_non_padded = tokenizer(sentences[0], return_tensors="pt").input_ids.to(torch_device)
        output_non_padded = model.generate(
            input_ids=inputs_non_padded, max_new_tokens=3, generation_config=generation_config
        )

        inputs_padded = tokenizer(sentences[1], return_tensors="pt").input_ids.to(torch_device)
        output_padded = model.generate(input_ids=inputs_padded, max_new_tokens=3, generation_config=generation_config)

        self.assertNotEqual(inputs_non_padded.shape, inputs_padded.shape)

        batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        non_padded_sentence = tokenizer.decode(output_non_padded[0], skip_special_tokens=True)
        padded_sentence = tokenizer.decode(output_padded[0], skip_special_tokens=True)

        expected_output_sentence = [
            "甲斐なら武田と言うほど甲斐の武田",
            "織田信長は、このような",
        ]
        self.assertListEqual(expected_output_sentence, batch_out_sentence)
        self.assertListEqual(batch_out_sentence, [non_padded_sentence, padded_sentence])

    @tooslow
    def test_sample(self):
        model = GPTSanJapaneseForConditionalGeneration.from_pretrained("Tanrei/GPTSAN-japanese")
        tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
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
            self.assertEqual(output_id, output)

    @slow
    def test_spout_generation(self):
        model = GPTSanJapaneseForConditionalGeneration.from_pretrained("Tanrei/GPTSAN-japanese")
        tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        model.to(torch_device)

        # set deterministically
        generation_config = GenerationConfig.from_pretrained("Tanrei/GPTSAN-japanese")
        generation_config.top_k = 1

        input_text = "武田信玄は、"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(torch_device)
        input_ids_batch = tokenizer([input_text, input_text], return_tensors="pt").input_ids.to(torch_device)

        # spout from uniform and one-hot
        spouts = [
            # fmt: off
            [0.87882208, 0.38426396, 0.33220248, 0.43890406, 0.16562252,
            0.04803985, 0.211572  , 0.23188473, 0.37153068, 0.7836377 ,
            0.02160172, 0.38761719, 0.75290772, 0.90198857, 0.34365777,
            0.64168169, 0.44318471, 0.14575746, 0.92562881, 0.40812148,
            0.29019122, 0.88861599, 0.65524846, 0.43563456, 0.38177187,
            0.70832965, 0.81527892, 0.68832812, 0.38833192, 0.4561522 ,
            0.14828817, 0.47248213, 0.54357335, 0.82009566, 0.1338884 ,
            0.02755417, 0.19764677, 0.2422084 , 0.04757674, 0.65409606,
            0.0824589 , 0.03304383, 0.94387689, 0.98764509, 0.82433901,
            0.27646741, 0.64907493, 0.76009406, 0.30087915, 0.17904689,
            0.41601714, 0.67046398, 0.10422822, 0.08447374, 0.07354344,
            0.61423565, 0.70284866, 0.7532333 , 0.1972038 , 0.29575659,
            0.90583886, 0.29265307, 0.50000175, 0.70407655, 0.889363  ,
            0.81904418, 0.66829128, 0.64468815, 0.56563723, 0.85601875,
            0.94924672, 0.00166762, 0.25220643, 0.74540219, 0.67993247,
            0.1549675 , 0.39385352, 0.92153607, 0.63745931, 0.27759043,
            0.84702295, 0.65904271, 0.58676614, 0.8666936 , 0.39607438,
            0.79954983, 0.42220697, 0.39650381, 0.7849864 , 0.56150201,
            0.15678925, 0.14746032, 0.34542114, 0.47026783, 0.11956489,
            0.25421435, 0.33788901, 0.68934842, 0.36424685, 0.71737898,
            0.38983449, 0.94393779, 0.39575588, 0.36616553, 0.87104665,
            0.64630203, 0.22516905, 0.88270804, 0.15031338, 0.75144345,
            0.46459025, 0.85396454, 0.86355643, 0.65139851, 0.70266061,
            0.30241389, 0.81056497, 0.88865969, 0.38773807, 0.70635849,
            0.90718459, 0.43245789, 0.28000654, 0.45935562, 0.08773519,
            0.9552151 , 0.93901511, 0.22489288], # uniform
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0.],
            # fmt: on
        ]

        output1 = model.generate(
            input_ids=input_ids,
            spout=spouts[0],
            max_new_tokens=20,
            generation_config=generation_config,
        )

        output2 = model.generate(
            input_ids=input_ids,
            spout=spouts[1],
            max_new_tokens=20,
            generation_config=generation_config,
        )

        output3 = model.generate(
            input_ids=input_ids_batch,
            spout=spouts,
            max_new_tokens=20,
            generation_config=generation_config,
        )

        out1_sentence = tokenizer.decode(output1[0])
        out2_sentence = tokenizer.decode(output2[0])
        batch_out_sentence = tokenizer.batch_decode(output3)

        expected_output_sentence = [
            "武田信玄は、武田氏の滅亡後、武田氏の居城であった甲斐武田氏の居城である",
            "武田信玄は、武田家の滅亡を防ぐため、武田家の家臣である武田信虎を討",
        ]
        self.assertListEqual(expected_output_sentence, batch_out_sentence)
        self.assertListEqual(batch_out_sentence, [out1_sentence, out2_sentence])

    @slow
    def test_prefix_lm_generation(self):
        model = GPTSanJapaneseForConditionalGeneration.from_pretrained("Tanrei/GPTSAN-japanese")
        tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        model.to(torch_device)

        # set deterministically
        generation_config = GenerationConfig.from_pretrained("Tanrei/GPTSAN-japanese")
        generation_config.top_k = 1

        prefix_text_1 = "武田信玄"
        prefix_text_2 = "織田信長"
        input_text_1 = "は、"
        input_text_2 = "が、"
        input_tok_1 = tokenizer(input_text_1, prefix_text=prefix_text_1, return_tensors="pt")
        input_tok_2 = tokenizer(input_text_2, prefix_text=prefix_text_2, return_tensors="pt")
        input_tok_3 = tokenizer([[prefix_text_1, input_text_1], [prefix_text_2, input_text_2]], return_tensors="pt")

        output1 = model.generate(
            input_ids=input_tok_1.input_ids.to(torch_device),
            token_type_ids=input_tok_1.token_type_ids.to(torch_device),
            max_new_tokens=20,
            generation_config=generation_config,
        )

        output2 = model.generate(
            input_ids=input_tok_2.input_ids.to(torch_device),
            token_type_ids=input_tok_2.token_type_ids.to(torch_device),
            max_new_tokens=20,
            generation_config=generation_config,
        )

        output3 = model.generate(
            input_ids=input_tok_3.input_ids.to(torch_device),
            token_type_ids=input_tok_3.token_type_ids.to(torch_device),
            attention_mask=input_tok_3.attention_mask.to(torch_device),
            max_new_tokens=20,
            generation_config=generation_config,
        )

        out1_sentence = tokenizer.decode(output1[0])
        out2_sentence = tokenizer.decode(output2[0])
        batch_out_sentence = tokenizer.batch_decode(output3)

        expected_output_sentence = [
            "武田信玄は、武田氏の祖である武田信虎を、その子・武田信友を擁して",
            "織田信長が、織田信長の妻・お市の方を妻として迎えたという逸話が残",
        ]
        self.assertListEqual(expected_output_sentence, batch_out_sentence)
        self.assertListEqual(batch_out_sentence, [out1_sentence, out2_sentence])
