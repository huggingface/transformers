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

from transformers import XLNetConfig, is_tf_available

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor
from .utils import CACHE_DIR, require_tf, slow


if is_tf_available():
    import tensorflow as tf

    from transformers.modeling_tf_xlnet import (
        TFXLNetModel,
        TFXLNetLMHeadModel,
        TFXLNetForSequenceClassification,
        TFXLNetForTokenClassification,
        TFXLNetForQuestionAnsweringSimple,
        TF_XLNET_PRETRAINED_MODEL_ARCHIVE_MAP,
    )


@require_tf
class TFXLNetModelTest(TFModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            TFXLNetModel,
            TFXLNetLMHeadModel,
            TFXLNetForSequenceClassification,
            TFXLNetForTokenClassification,
            TFXLNetForQuestionAnsweringSimple,
        )
        if is_tf_available()
        else ()
    )
    all_generative_model_classes = (
        (TFXLNetLMHeadModel,) if is_tf_available() else ()
    )  # TODO (PVP): Check other models whether language generation is also applicable
    test_pruning = False

    class TFXLNetModelTester(object):
        def __init__(
            self,
            parent,
            batch_size=13,
            seq_length=7,
            mem_len=10,
            clamp_len=-1,
            reuse_len=15,
            is_training=True,
            use_labels=True,
            vocab_size=99,
            cutoffs=[10, 50, 80],
            hidden_size=32,
            num_attention_heads=4,
            d_inner=128,
            num_hidden_layers=5,
            type_sequence_label_size=2,
            untie_r=True,
            bi_data=False,
            same_length=False,
            initializer_range=0.05,
            seed=1,
            type_vocab_size=2,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=5,
        ):
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.mem_len = mem_len
            # self.key_len = seq_length + mem_len
            self.clamp_len = clamp_len
            self.reuse_len = reuse_len
            self.is_training = is_training
            self.use_labels = use_labels
            self.vocab_size = vocab_size
            self.cutoffs = cutoffs
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads
            self.d_inner = d_inner
            self.num_hidden_layers = num_hidden_layers
            self.bi_data = bi_data
            self.untie_r = untie_r
            self.same_length = same_length
            self.initializer_range = initializer_range
            self.seed = seed
            self.type_vocab_size = type_vocab_size
            self.type_sequence_label_size = type_sequence_label_size
            self.bos_token_id = bos_token_id
            self.pad_token_id = pad_token_id
            self.eos_token_id = eos_token_id

        def prepare_config_and_inputs(self):
            input_ids_1 = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            input_ids_2 = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            segment_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)
            input_mask = ids_tensor([self.batch_size, self.seq_length], 2, dtype=tf.float32)

            input_ids_q = ids_tensor([self.batch_size, self.seq_length + 1], self.vocab_size)
            perm_mask = tf.zeros((self.batch_size, self.seq_length + 1, self.seq_length), dtype=tf.float32)
            perm_mask_last = tf.ones((self.batch_size, self.seq_length + 1, 1), dtype=tf.float32)
            perm_mask = tf.concat([perm_mask, perm_mask_last], axis=-1)
            # perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
            target_mapping = tf.zeros((self.batch_size, 1, self.seq_length), dtype=tf.float32)
            target_mapping_last = tf.ones((self.batch_size, 1, 1), dtype=tf.float32)
            target_mapping = tf.concat([target_mapping, target_mapping_last], axis=-1)
            # target_mapping[:, 0, -1] = 1.0  # predict last token

            sequence_labels = None
            lm_labels = None
            is_impossible_labels = None
            if self.use_labels:
                lm_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
                sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
                is_impossible_labels = ids_tensor([self.batch_size], 2, dtype=tf.float32)

            config = XLNetConfig(
                vocab_size=self.vocab_size,
                d_model=self.hidden_size,
                n_head=self.num_attention_heads,
                d_inner=self.d_inner,
                n_layer=self.num_hidden_layers,
                untie_r=self.untie_r,
                mem_len=self.mem_len,
                clamp_len=self.clamp_len,
                same_length=self.same_length,
                reuse_len=self.reuse_len,
                bi_data=self.bi_data,
                initializer_range=self.initializer_range,
                num_labels=self.type_sequence_label_size,
                bos_token_id=self.bos_token_id,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )

            return (
                config,
                input_ids_1,
                input_ids_2,
                input_ids_q,
                perm_mask,
                input_mask,
                target_mapping,
                segment_ids,
                lm_labels,
                sequence_labels,
                is_impossible_labels,
            )

        def set_seed(self):
            random.seed(self.seed)
            tf.random.set_seed(self.seed)

        def create_and_check_xlnet_base_model(
            self,
            config,
            input_ids_1,
            input_ids_2,
            input_ids_q,
            perm_mask,
            input_mask,
            target_mapping,
            segment_ids,
            lm_labels,
            sequence_labels,
            is_impossible_labels,
        ):
            model = TFXLNetModel(config)

            inputs = {"input_ids": input_ids_1, "input_mask": input_mask, "token_type_ids": segment_ids}

            _, _ = model(inputs)

            inputs = [input_ids_1, input_mask]

            outputs, mems_1 = model(inputs)

            result = {
                "mems_1": [mem.numpy() for mem in mems_1],
                "outputs": outputs.numpy(),
            }

            config.mem_len = 0
            model = TFXLNetModel(config)
            no_mems_outputs = model(inputs)
            self.parent.assertEqual(len(no_mems_outputs), 1)

            self.parent.assertListEqual(
                list(result["outputs"].shape), [self.batch_size, self.seq_length, self.hidden_size]
            )
            self.parent.assertListEqual(
                list(list(mem.shape) for mem in result["mems_1"]),
                [[self.seq_length, self.batch_size, self.hidden_size]] * self.num_hidden_layers,
            )

        def create_and_check_xlnet_lm_head(
            self,
            config,
            input_ids_1,
            input_ids_2,
            input_ids_q,
            perm_mask,
            input_mask,
            target_mapping,
            segment_ids,
            lm_labels,
            sequence_labels,
            is_impossible_labels,
        ):
            model = TFXLNetLMHeadModel(config)

            inputs_1 = {"input_ids": input_ids_1, "token_type_ids": segment_ids}

            all_logits_1, mems_1 = model(inputs_1)

            inputs_2 = {"input_ids": input_ids_2, "mems": mems_1, "token_type_ids": segment_ids}

            all_logits_2, mems_2 = model(inputs_2)

            inputs_3 = {"input_ids": input_ids_q, "perm_mask": perm_mask, "target_mapping": target_mapping}

            logits, _ = model(inputs_3)

            result = {
                "mems_1": [mem.numpy() for mem in mems_1],
                "all_logits_1": all_logits_1.numpy(),
                "mems_2": [mem.numpy() for mem in mems_2],
                "all_logits_2": all_logits_2.numpy(),
            }

            self.parent.assertListEqual(
                list(result["all_logits_1"].shape), [self.batch_size, self.seq_length, self.vocab_size]
            )
            self.parent.assertListEqual(
                list(list(mem.shape) for mem in result["mems_1"]),
                [[self.seq_length, self.batch_size, self.hidden_size]] * self.num_hidden_layers,
            )

            self.parent.assertListEqual(
                list(result["all_logits_2"].shape), [self.batch_size, self.seq_length, self.vocab_size]
            )
            self.parent.assertListEqual(
                list(list(mem.shape) for mem in result["mems_2"]),
                [[self.mem_len, self.batch_size, self.hidden_size]] * self.num_hidden_layers,
            )

        def create_and_check_xlnet_qa(
            self,
            config,
            input_ids_1,
            input_ids_2,
            input_ids_q,
            perm_mask,
            input_mask,
            target_mapping,
            segment_ids,
            lm_labels,
            sequence_labels,
            is_impossible_labels,
        ):
            model = TFXLNetForQuestionAnsweringSimple(config)

            inputs = {"input_ids": input_ids_1, "attention_mask": input_mask, "token_type_ids": segment_ids}
            start_logits, end_logits, mems = model(inputs)

            result = {
                "start_logits": start_logits.numpy(),
                "end_logits": end_logits.numpy(),
                "mems": [m.numpy() for m in mems],
            }

            self.parent.assertListEqual(list(result["start_logits"].shape), [self.batch_size, self.seq_length])
            self.parent.assertListEqual(list(result["end_logits"].shape), [self.batch_size, self.seq_length])
            self.parent.assertListEqual(
                list(list(mem.shape) for mem in result["mems"]),
                [[self.seq_length, self.batch_size, self.hidden_size]] * self.num_hidden_layers,
            )

        def create_and_check_xlnet_sequence_classif(
            self,
            config,
            input_ids_1,
            input_ids_2,
            input_ids_q,
            perm_mask,
            input_mask,
            target_mapping,
            segment_ids,
            lm_labels,
            sequence_labels,
            is_impossible_labels,
        ):
            model = TFXLNetForSequenceClassification(config)

            logits, mems_1 = model(input_ids_1)

            result = {
                "mems_1": [mem.numpy() for mem in mems_1],
                "logits": logits.numpy(),
            }

            self.parent.assertListEqual(list(result["logits"].shape), [self.batch_size, self.type_sequence_label_size])
            self.parent.assertListEqual(
                list(list(mem.shape) for mem in result["mems_1"]),
                [[self.seq_length, self.batch_size, self.hidden_size]] * self.num_hidden_layers,
            )

        def create_and_check_xlnet_for_token_classification(
            self,
            config,
            input_ids_1,
            input_ids_2,
            input_ids_q,
            perm_mask,
            input_mask,
            target_mapping,
            segment_ids,
            lm_labels,
            sequence_labels,
            is_impossible_labels,
        ):
            config.num_labels = input_ids_1.shape[1]
            model = TFXLNetForTokenClassification(config)
            inputs = {
                "input_ids": input_ids_1,
                "attention_mask": input_mask,
                # 'token_type_ids': token_type_ids
            }
            logits, mems_1 = model(inputs)
            result = {
                "mems_1": [mem.numpy() for mem in mems_1],
                "logits": logits.numpy(),
            }
            self.parent.assertListEqual(
                list(result["logits"].shape), [self.batch_size, self.seq_length, config.num_labels]
            )
            self.parent.assertListEqual(
                list(list(mem.shape) for mem in result["mems_1"]),
                [[self.seq_length, self.batch_size, self.hidden_size]] * self.num_hidden_layers,
            )

        def prepare_config_and_inputs_for_common(self):
            config_and_inputs = self.prepare_config_and_inputs()
            (
                config,
                input_ids_1,
                input_ids_2,
                input_ids_q,
                perm_mask,
                input_mask,
                target_mapping,
                segment_ids,
                lm_labels,
                sequence_labels,
                is_impossible_labels,
            ) = config_and_inputs
            inputs_dict = {"input_ids": input_ids_1}
            return config, inputs_dict

    def setUp(self):
        self.model_tester = TFXLNetModelTest.TFXLNetModelTester(self)
        self.config_tester = ConfigTester(self, config_class=XLNetConfig, d_inner=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_xlnet_base_model(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_base_model(*config_and_inputs)

    def test_xlnet_lm_head(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_lm_head(*config_and_inputs)

    def test_xlnet_sequence_classif(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_sequence_classif(*config_and_inputs)

    def test_xlnet_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_for_token_classification(*config_and_inputs)

    def test_xlnet_qa(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_qa(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(TF_XLNET_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            model = TFXLNetModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
            self.assertIsNotNone(model)


class TFXLNetModelLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_xlnet_base_cased(self):
        model = TFXLNetLMHeadModel.from_pretrained("xlnet-base-cased")
        input_ids = tf.convert_to_tensor(
            [
                [
                    67,
                    2840,
                    19,
                    18,
                    1484,
                    20,
                    965,
                    29077,
                    8719,
                    1273,
                    21,
                    45,
                    273,
                    17,
                    10,
                    15048,
                    28,
                    27511,
                    21,
                    4185,
                    11,
                    41,
                    2444,
                    9,
                    32,
                    1025,
                    20,
                    8719,
                    26,
                    23,
                    673,
                    966,
                    19,
                    29077,
                    20643,
                    27511,
                    20822,
                    20643,
                    19,
                    17,
                    6616,
                    17511,
                    18,
                    8978,
                    20,
                    18,
                    777,
                    9,
                    19233,
                    1527,
                    17669,
                    19,
                    24,
                    673,
                    17,
                    28756,
                    150,
                    12943,
                    4354,
                    153,
                    27,
                    442,
                    37,
                    45,
                    668,
                    21,
                    24,
                    256,
                    20,
                    416,
                    22,
                    2771,
                    4901,
                    9,
                    12943,
                    4354,
                    153,
                    51,
                    24,
                    3004,
                    21,
                    28142,
                    23,
                    65,
                    20,
                    18,
                    416,
                    34,
                    24,
                    2958,
                    22947,
                    9,
                    1177,
                    45,
                    668,
                    3097,
                    13768,
                    23,
                    103,
                    28,
                    441,
                    148,
                    48,
                    20522,
                    19,
                    12943,
                    4354,
                    153,
                    12860,
                    34,
                    18,
                    326,
                    27,
                    17492,
                    684,
                    21,
                    6709,
                    9,
                    8585,
                    123,
                    266,
                    19,
                    12943,
                    4354,
                    153,
                    6872,
                    24,
                    3004,
                    20,
                    18,
                    9225,
                    2198,
                    19,
                    12717,
                    103,
                    22,
                    401,
                    24,
                    6348,
                    9,
                    12943,
                    4354,
                    153,
                    1068,
                    2768,
                    2286,
                    19,
                    33,
                    104,
                    19,
                    176,
                    24,
                    9313,
                    19,
                    20086,
                    28,
                    45,
                    10292,
                    9,
                    4,
                    3,
                ]
            ],
            dtype=tf.int32,
        )
        #  In 1991, the remains of Russian Tsar Nicholas II and his family
        #  (except for Alexei and Maria) are discovered.
        #  The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
        #  remainder of the story. 1883 Western Siberia,
        #  a young Grigori Rasputin is asked by his father and a group of men to perform magic.
        #  Rasputin has a vision and denounces one of the men as a horse thief. Although his
        #  father initially slaps him for making such an accusation, Rasputin watches as the
        #  man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
        #  the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
        #  with people, even a bishop, begging for his blessing. """

        expected_output_ids = [
            67,
            2840,
            19,
            18,
            1484,
            20,
            965,
            29077,
            8719,
            1273,
            21,
            45,
            273,
            17,
            10,
            15048,
            28,
            27511,
            21,
            4185,
            11,
            41,
            2444,
            9,
            32,
            1025,
            20,
            8719,
            26,
            23,
            673,
            966,
            19,
            29077,
            20643,
            27511,
            20822,
            20643,
            19,
            17,
            6616,
            17511,
            18,
            8978,
            20,
            18,
            777,
            9,
            19233,
            1527,
            17669,
            19,
            24,
            673,
            17,
            28756,
            150,
            12943,
            4354,
            153,
            27,
            442,
            37,
            45,
            668,
            21,
            24,
            256,
            20,
            416,
            22,
            2771,
            4901,
            9,
            12943,
            4354,
            153,
            51,
            24,
            3004,
            21,
            28142,
            23,
            65,
            20,
            18,
            416,
            34,
            24,
            2958,
            22947,
            9,
            1177,
            45,
            668,
            3097,
            13768,
            23,
            103,
            28,
            441,
            148,
            48,
            20522,
            19,
            12943,
            4354,
            153,
            12860,
            34,
            18,
            326,
            27,
            17492,
            684,
            21,
            6709,
            9,
            8585,
            123,
            266,
            19,
            12943,
            4354,
            153,
            6872,
            24,
            3004,
            20,
            18,
            9225,
            2198,
            19,
            12717,
            103,
            22,
            401,
            24,
            6348,
            9,
            12943,
            4354,
            153,
            1068,
            2768,
            2286,
            19,
            33,
            104,
            19,
            176,
            24,
            9313,
            19,
            20086,
            28,
            45,
            10292,
            9,
            4,
            3,
            19,
            12943,
            4354,
            153,
            27,
            442,
            22,
            2771,
            4901,
            9,
            69,
            27,
            50,
            551,
            22,
            2771,
            4901,
            19,
            21,
            45,
            668,
            21,
            18,
            416,
            41,
            1499,
            22,
            755,
            18,
            14285,
            9,
            12943,
            4354,
            153,
            27,
            1499,
            22,
            642,
            22,
        ]
        #  In 1991, the remains of Russian Tsar Nicholas II and his family (except for Alexei and Maria)
        #  are discovered. The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich,
        #  narrates the remainder of the story. 1883 Western Siberia, a young Grigori Rasputin
        #  is asked by his father and a group of men to perform magic. Rasputin has a vision and
        #  denounces one of the men as a horse thief. Although his father initially slaps
        #  him for making such an accusation, Rasputin watches as the man is chased outside and beaten.
        #  Twenty years later, Rasputin sees a vision of the Virgin Mary, prompting him to become a priest.
        #  Rasputin quickly becomes famous, with people, even a bishop, begging for his blessing.
        #  <sep><cls>, Rasputin is asked to perform magic.
        #  He is not able to perform magic, and his father and
        # the men are forced to leave the monastery. Rasputin is forced to return to

        output_ids = model.generate(input_ids, max_length=200, do_sample=False)

        self.assertListEqual(output_ids[0].numpy().tolist(), expected_output_ids)
