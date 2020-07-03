# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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


import random
import unittest

from transformers import TransfoXLConfig, is_tf_available
from transformers.testing_utils import require_tf, slow

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor


if is_tf_available():
    import tensorflow as tf
    from transformers import (
        TFTransfoXLModel,
        TFTransfoXLLMHeadModel,
        TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
    )


class TFTransfoXLModelTester:
    def __init__(
        self, parent,
    ):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.mem_len = 30
        self.key_length = self.seq_length + self.mem_len
        self.clamp_len = 15
        self.is_training = True
        self.use_labels = True
        self.vocab_size = 99
        self.cutoffs = [10, 50, 80]
        self.hidden_size = 32
        self.d_embed = 32
        self.num_attention_heads = 4
        self.d_head = 8
        self.d_inner = 128
        self.div_val = 2
        self.num_hidden_layers = 5
        self.scope = None
        self.seed = 1
        self.eos_token_id = 0

    def prepare_config_and_inputs(self):
        input_ids_1 = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_ids_2 = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = TransfoXLConfig(
            vocab_size=self.vocab_size,
            mem_len=self.mem_len,
            clamp_len=self.clamp_len,
            cutoffs=self.cutoffs,
            d_model=self.hidden_size,
            d_embed=self.d_embed,
            n_head=self.num_attention_heads,
            d_head=self.d_head,
            d_inner=self.d_inner,
            div_val=self.div_val,
            n_layer=self.num_hidden_layers,
            eos_token_id=self.eos_token_id,
        )

        return (config, input_ids_1, input_ids_2, lm_labels)

    def set_seed(self):
        random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def create_and_check_transfo_xl_model(self, config, input_ids_1, input_ids_2, lm_labels):
        model = TFTransfoXLModel(config)

        hidden_states_1, mems_1 = model(input_ids_1)

        inputs = {"input_ids": input_ids_2, "mems": mems_1}

        hidden_states_2, mems_2 = model(inputs)

        result = {
            "hidden_states_1": hidden_states_1.numpy(),
            "mems_1": [mem.numpy() for mem in mems_1],
            "hidden_states_2": hidden_states_2.numpy(),
            "mems_2": [mem.numpy() for mem in mems_2],
        }

        self.parent.assertListEqual(
            list(result["hidden_states_1"].shape), [self.batch_size, self.seq_length, self.hidden_size]
        )
        self.parent.assertListEqual(
            list(result["hidden_states_2"].shape), [self.batch_size, self.seq_length, self.hidden_size]
        )
        self.parent.assertListEqual(
            list(list(mem.shape) for mem in result["mems_1"]),
            [[self.mem_len, self.batch_size, self.hidden_size]] * self.num_hidden_layers,
        )
        self.parent.assertListEqual(
            list(list(mem.shape) for mem in result["mems_2"]),
            [[self.mem_len, self.batch_size, self.hidden_size]] * self.num_hidden_layers,
        )

    def create_and_check_transfo_xl_lm_head(self, config, input_ids_1, input_ids_2, lm_labels):
        model = TFTransfoXLLMHeadModel(config)

        lm_logits_1, mems_1 = model(input_ids_1)

        inputs = {"input_ids": input_ids_1, "labels": lm_labels}
        _, mems_1 = model(inputs)

        lm_logits_2, mems_2 = model([input_ids_2, mems_1])

        inputs = {"input_ids": input_ids_1, "mems": mems_1, "labels": lm_labels}

        _, mems_2 = model(inputs)

        result = {
            "mems_1": [mem.numpy() for mem in mems_1],
            "lm_logits_1": lm_logits_1.numpy(),
            "mems_2": [mem.numpy() for mem in mems_2],
            "lm_logits_2": lm_logits_2.numpy(),
        }

        self.parent.assertListEqual(
            list(result["lm_logits_1"].shape), [self.batch_size, self.seq_length, self.vocab_size]
        )
        self.parent.assertListEqual(
            list(list(mem.shape) for mem in result["mems_1"]),
            [[self.mem_len, self.batch_size, self.hidden_size]] * self.num_hidden_layers,
        )

        self.parent.assertListEqual(
            list(result["lm_logits_2"].shape), [self.batch_size, self.seq_length, self.vocab_size]
        )
        self.parent.assertListEqual(
            list(list(mem.shape) for mem in result["mems_2"]),
            [[self.mem_len, self.batch_size, self.hidden_size]] * self.num_hidden_layers,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids_1, input_ids_2, lm_labels) = config_and_inputs
        inputs_dict = {"input_ids": input_ids_1}
        return config, inputs_dict


@require_tf
class TFTransfoXLModelTest(TFModelTesterMixin, unittest.TestCase):

    all_model_classes = (TFTransfoXLModel, TFTransfoXLLMHeadModel) if is_tf_available() else ()
    all_generative_model_classes = () if is_tf_available() else ()
    # TODO: add this test when TFTransfoXLLMHead has a linear output layer implemented
    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = TFTransfoXLModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TransfoXLConfig, d_embed=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_transfo_xl_model(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_transfo_xl_model(*config_and_inputs)

    def test_transfo_xl_lm_head(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_transfo_xl_lm_head(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFTransfoXLModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_tf
class TFTransfoXLModelLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_transfo_xl_wt103(self):
        model = TFTransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103")
        input_ids = tf.convert_to_tensor(
            [
                [
                    33,
                    1297,
                    2,
                    1,
                    1009,
                    4,
                    1109,
                    11739,
                    4762,
                    358,
                    5,
                    25,
                    245,
                    22,
                    1706,
                    17,
                    20098,
                    5,
                    3215,
                    21,
                    37,
                    1110,
                    3,
                    13,
                    1041,
                    4,
                    24,
                    603,
                    490,
                    2,
                    71477,
                    20098,
                    104447,
                    2,
                    20961,
                    1,
                    2604,
                    4,
                    1,
                    329,
                    3,
                    6224,
                    831,
                    16002,
                    2,
                    8,
                    603,
                    78967,
                    29546,
                    23,
                    803,
                    20,
                    25,
                    416,
                    5,
                    8,
                    232,
                    4,
                    277,
                    6,
                    1855,
                    4601,
                    3,
                    29546,
                    54,
                    8,
                    3609,
                    5,
                    57211,
                    49,
                    4,
                    1,
                    277,
                    18,
                    8,
                    1755,
                    15691,
                    3,
                    341,
                    25,
                    416,
                    693,
                    42573,
                    71,
                    17,
                    401,
                    94,
                    31,
                    17919,
                    2,
                    29546,
                    7873,
                    18,
                    1,
                    435,
                    23,
                    11011,
                    755,
                    5,
                    5167,
                    3,
                    7983,
                    98,
                    84,
                    2,
                    29546,
                    3267,
                    8,
                    3609,
                    4,
                    1,
                    4865,
                    1075,
                    2,
                    6087,
                    71,
                    6,
                    346,
                    8,
                    5854,
                    3,
                    29546,
                    824,
                    1400,
                    1868,
                    2,
                    19,
                    160,
                    2,
                    311,
                    8,
                    5496,
                    2,
                    20920,
                    17,
                    25,
                    15097,
                    3,
                    24,
                    24,
                    0,
                ]
            ],
            dtype=tf.int32,
        )
        #  In 1991 , the remains of Russian Tsar Nicholas II and his family
        #  ( except for Alexei and Maria ) are discovered .
        #  The voice of Nicholas's young son , Tsarevich Alexei Nikolaevich , narrates the
        #  remainder of the story . 1883 Western Siberia ,
        #  a young Grigori Rasputin is asked by his father and a group of men to perform magic .
        #  Rasputin has a vision and denounces one of the men as a horse thief . Although his
        #  father initially slaps him for making such an accusation , Rasputin watches as the
        #  man is chased outside and beaten . Twenty years later , Rasputin sees a vision of
        #  the Virgin Mary , prompting him to become a priest . Rasputin quickly becomes famous ,
        #  with people , even a bishop , begging for his blessing . <eod> </s> <eos>

        expected_output_ids = [
            33,
            1297,
            2,
            1,
            1009,
            4,
            1109,
            11739,
            4762,
            358,
            5,
            25,
            245,
            22,
            1706,
            17,
            20098,
            5,
            3215,
            21,
            37,
            1110,
            3,
            13,
            1041,
            4,
            24,
            603,
            490,
            2,
            71477,
            20098,
            104447,
            2,
            20961,
            1,
            2604,
            4,
            1,
            329,
            3,
            6224,
            831,
            16002,
            2,
            8,
            603,
            78967,
            29546,
            23,
            803,
            20,
            25,
            416,
            5,
            8,
            232,
            4,
            277,
            6,
            1855,
            4601,
            3,
            29546,
            54,
            8,
            3609,
            5,
            57211,
            49,
            4,
            1,
            277,
            18,
            8,
            1755,
            15691,
            3,
            341,
            25,
            416,
            693,
            42573,
            71,
            17,
            401,
            94,
            31,
            17919,
            2,
            29546,
            7873,
            18,
            1,
            435,
            23,
            11011,
            755,
            5,
            5167,
            3,
            7983,
            98,
            84,
            2,
            29546,
            3267,
            8,
            3609,
            4,
            1,
            4865,
            1075,
            2,
            6087,
            71,
            6,
            346,
            8,
            5854,
            3,
            29546,
            824,
            1400,
            1868,
            2,
            19,
            160,
            2,
            311,
            8,
            5496,
            2,
            20920,
            17,
            25,
            15097,
            3,
            24,
            24,
            0,
            33,
            1,
            1857,
            2,
            1,
            1009,
            4,
            1109,
            11739,
            4762,
            358,
            5,
            25,
            245,
            28,
            1110,
            3,
            13,
            1041,
            4,
            24,
            603,
            490,
            2,
            71477,
            20098,
            104447,
            2,
            20961,
            1,
            2604,
            4,
            1,
            329,
            3,
            0,
        ]
        #  In 1991, the remains of Russian Tsar Nicholas II and his family (
        #  except for Alexei and Maria ) are discovered. The voice of young son,
        #  Tsarevich Alexei Nikolaevich, narrates the remainder of the story.
        #  1883 Western Siberia, a young Grigori Rasputin is asked by his father
        #  and a group of men to perform magic. Rasputin has a vision and
        #  denounces one of the men as a horse thief. Although his father initially
        #  slaps him for making such an accusation, Rasputin watches as the man
        #  is chased outside and beaten. Twenty years later, Rasputin sees a vision
        #  of the Virgin Mary, prompting him to become a priest.
        #  Rasputin quickly becomes famous, with people, even a bishop, begging for
        #  his blessing. <unk> <unk> <eos> In the 1990s, the remains of Russian Tsar
        # Nicholas II and his family were discovered. The voice of <unk> young son,
        # Tsarevich Alexei Nikolaevich, narrates the remainder of the story.<eos>

        output_ids = model.generate(input_ids, max_length=200, do_sample=False)
        self.assertListEqual(output_ids[0].numpy().tolist(), expected_output_ids)
