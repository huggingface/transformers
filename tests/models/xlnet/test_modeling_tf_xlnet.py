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


from __future__ import annotations

import inspect
import random
import unittest

from transformers import XLNetConfig, is_tf_available
from transformers.testing_utils import require_tf, slow

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers.models.xlnet.modeling_tf_xlnet import (
        TFXLNetForMultipleChoice,
        TFXLNetForQuestionAnsweringSimple,
        TFXLNetForSequenceClassification,
        TFXLNetForTokenClassification,
        TFXLNetLMHeadModel,
        TFXLNetModel,
    )


class TFXLNetModelTester:
    def __init__(
        self,
        parent,
    ):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.mem_len = 10
        # self.key_len = seq_length + mem_len
        self.clamp_len = -1
        self.reuse_len = 15
        self.is_training = True
        self.use_labels = True
        self.vocab_size = 99
        self.cutoffs = [10, 50, 80]
        self.hidden_size = 32
        self.num_attention_heads = 4
        self.d_inner = 128
        self.num_hidden_layers = 2
        self.type_sequence_label_size = 2
        self.untie_r = True
        self.bi_data = False
        self.same_length = False
        self.initializer_range = 0.05
        self.seed = 1
        self.type_vocab_size = 2
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 5
        self.num_choices = 4

    def prepare_config_and_inputs(self):
        input_ids_1 = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_ids_2 = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        segment_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)
        input_mask = random_attention_mask([self.batch_size, self.seq_length], dtype=tf.float32)

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
        result = model(inputs)

        inputs = [input_ids_1, input_mask]
        result = model(inputs)

        config.use_mems_eval = False
        model = TFXLNetModel(config)
        no_mems_outputs = model(inputs)
        self.parent.assertEqual(len(no_mems_outputs), 1)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertListEqual(
            [mem.shape for mem in result.mems],
            [(self.seq_length, self.batch_size, self.hidden_size)] * self.num_hidden_layers,
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
        all_logits_1, mems_1 = model(inputs_1).to_tuple()

        inputs_2 = {"input_ids": input_ids_2, "mems": mems_1, "token_type_ids": segment_ids}
        all_logits_2, mems_2 = model(inputs_2).to_tuple()

        inputs_3 = {"input_ids": input_ids_q, "perm_mask": perm_mask, "target_mapping": target_mapping}
        logits, _ = model(inputs_3).to_tuple()

        self.parent.assertEqual(all_logits_1.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertListEqual(
            [mem.shape for mem in mems_1],
            [(self.seq_length, self.batch_size, self.hidden_size)] * self.num_hidden_layers,
        )
        self.parent.assertEqual(all_logits_2.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertListEqual(
            [mem.shape for mem in mems_2],
            [(self.mem_len, self.batch_size, self.hidden_size)] * self.num_hidden_layers,
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
        result = model(inputs)

        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertListEqual(
            [mem.shape for mem in result.mems],
            [(self.seq_length, self.batch_size, self.hidden_size)] * self.num_hidden_layers,
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

        result = model(input_ids_1)

        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))
        self.parent.assertListEqual(
            [mem.shape for mem in result.mems],
            [(self.seq_length, self.batch_size, self.hidden_size)] * self.num_hidden_layers,
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
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, config.num_labels))
        self.parent.assertListEqual(
            [mem.shape for mem in result.mems],
            [(self.seq_length, self.batch_size, self.hidden_size)] * self.num_hidden_layers,
        )

    def create_and_check_xlnet_for_multiple_choice(
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
        config.num_choices = self.num_choices
        model = TFXLNetForMultipleChoice(config=config)
        multiple_choice_inputs_ids = tf.tile(tf.expand_dims(input_ids_1, 1), (1, self.num_choices, 1))
        multiple_choice_input_mask = tf.tile(tf.expand_dims(input_mask, 1), (1, self.num_choices, 1))
        multiple_choice_token_type_ids = tf.tile(tf.expand_dims(segment_ids, 1), (1, self.num_choices, 1))
        inputs = {
            "input_ids": multiple_choice_inputs_ids,
            "attention_mask": multiple_choice_input_mask,
            "token_type_ids": multiple_choice_token_type_ids,
        }
        result = model(inputs)

        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))
        self.parent.assertListEqual(
            [mem.shape for mem in result.mems],
            [(self.seq_length, self.batch_size * self.num_choices, self.hidden_size)] * self.num_hidden_layers,
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


@require_tf
class TFXLNetModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            TFXLNetModel,
            TFXLNetLMHeadModel,
            TFXLNetForSequenceClassification,
            TFXLNetForTokenClassification,
            TFXLNetForQuestionAnsweringSimple,
            TFXLNetForMultipleChoice,
        )
        if is_tf_available()
        else ()
    )
    all_generative_model_classes = (
        (TFXLNetLMHeadModel,) if is_tf_available() else ()
    )  # TODO (PVP): Check other models whether language generation is also applicable
    pipeline_model_mapping = (
        {
            "feature-extraction": TFXLNetModel,
            "question-answering": TFXLNetForQuestionAnsweringSimple,
            "text-classification": TFXLNetForSequenceClassification,
            "text-generation": TFXLNetLMHeadModel,
            "token-classification": TFXLNetForTokenClassification,
            "zero-shot": TFXLNetForSequenceClassification,
        }
        if is_tf_available()
        else {}
    )
    test_head_masking = False
    test_onnx = False

    # Note that `TFXLNetModelTest` is not a subclass of `GenerationTesterMixin`, so no contrastive generation tests
    # from there is run against `TFXLNetModel`.
    @unittest.skip("XLNet has special cache mechanism and is currently not working with contrastive generation")
    def test_xla_generate_contrastive(self):
        super().test_xla_generate_contrastive()

    # TODO: Fix the failed tests
    def is_pipeline_test_to_skip(
        self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name
    ):
        # Exception encountered when calling layer '...'
        return True

    def setUp(self):
        self.model_tester = TFXLNetModelTester(self)
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

    def test_xlnet_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_for_multiple_choice(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "xlnet/xlnet-base-cased"
        model = TFXLNetModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @unittest.skip("Some of the XLNet models misbehave with flexible input shapes.")
    def test_compile_tf_model(self):
        pass

    # overwrite since `TFXLNetLMHeadModel` doesn't cut logits/labels
    def test_loss_computation(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            if getattr(model, "hf_compute_loss", None):
                # The number of elements in the loss should be the same as the number of elements in the label
                prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
                added_label = prepared_for_class[
                    sorted(prepared_for_class.keys() - inputs_dict.keys(), reverse=True)[0]
                ]
                expected_loss_size = added_label.shape.as_list()[:1]

                # `TFXLNetLMHeadModel` doesn't cut logits/labels
                # if model.__class__ in get_values(TF_MODEL_FOR_CAUSAL_LM_MAPPING):
                #     # if loss is causal lm loss, labels are shift, so that one label per batch
                #     # is cut
                #     loss_size = loss_size - self.model_tester.batch_size

                # Test that model correctly compute the loss with kwargs
                prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
                input_name = "input_ids" if "input_ids" in prepared_for_class else "pixel_values"
                input_ids = prepared_for_class.pop(input_name)

                loss = model(input_ids, **prepared_for_class)[0]
                self.assertTrue(loss.shape.as_list() == expected_loss_size or loss.shape.as_list() == [1])

                # Test that model correctly compute the loss with a dict
                prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
                loss = model(prepared_for_class)[0]
                self.assertTrue(loss.shape.as_list() == expected_loss_size or loss.shape.as_list() == [1])

                # Test that model correctly compute the loss with a tuple
                prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)

                # Get keys that were added with the _prepare_for_class function
                label_keys = prepared_for_class.keys() - inputs_dict.keys()
                signature = inspect.signature(model.call).parameters
                signature_names = list(signature.keys())

                # Create a dictionary holding the location of the tensors in the tuple
                tuple_index_mapping = {0: input_name}
                for label_key in label_keys:
                    label_key_index = signature_names.index(label_key)
                    tuple_index_mapping[label_key_index] = label_key
                sorted_tuple_index_mapping = sorted(tuple_index_mapping.items())
                # Initialize a list with their default values, update the values and convert to a tuple
                list_input = []

                for name in signature_names:
                    if name != "kwargs":
                        list_input.append(signature[name].default)

                for index, value in sorted_tuple_index_mapping:
                    list_input[index] = prepared_for_class[value]

                tuple_input = tuple(list_input)

                # Send to model
                loss = model(tuple_input[:-1])[0]

                self.assertTrue(loss.shape.as_list() == expected_loss_size or loss.shape.as_list() == [1])


@require_tf
class TFXLNetModelLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_xlnet_base_cased(self):
        model = TFXLNetLMHeadModel.from_pretrained("xlnet/xlnet-base-cased")
        # fmt: off
        input_ids = tf.convert_to_tensor(
            [
                [
                    67, 2840, 19, 18, 1484, 20, 965, 29077, 8719, 1273, 21, 45, 273, 17, 10, 15048, 28, 27511, 21, 4185, 11, 41, 2444, 9, 32, 1025, 20, 8719, 26, 23, 673, 966, 19, 29077, 20643, 27511, 20822, 20643, 19, 17, 6616, 17511, 18, 8978, 20, 18, 777, 9, 19233, 1527, 17669, 19, 24, 673, 17, 28756, 150, 12943, 4354, 153, 27, 442, 37, 45, 668, 21, 24, 256, 20, 416, 22, 2771, 4901, 9, 12943, 4354, 153, 51, 24, 3004, 21, 28142, 23, 65, 20, 18, 416, 34, 24, 2958, 22947, 9, 1177, 45, 668, 3097, 13768, 23, 103, 28, 441, 148, 48, 20522, 19, 12943, 4354, 153, 12860, 34, 18, 326, 27, 17492, 684, 21, 6709, 9, 8585, 123, 266, 19, 12943, 4354, 153, 6872, 24, 3004, 20, 18, 9225, 2198, 19, 12717, 103, 22, 401, 24, 6348, 9, 12943, 4354, 153, 1068, 2768, 2286, 19, 33, 104, 19, 176, 24, 9313, 19, 20086, 28, 45, 10292, 9, 4, 3,
                ]
            ],
            dtype=tf.int32,
        )
        # fmt: on

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

        # fmt: off
        expected_output_ids = [
            67, 2840, 19, 18, 1484, 20, 965, 29077, 8719, 1273, 21, 45, 273, 17, 10, 15048, 28, 27511, 21, 4185, 11, 41, 2444, 9, 32, 1025, 20, 8719, 26, 23, 673, 966, 19, 29077, 20643, 27511, 20822, 20643, 19, 17, 6616, 17511, 18, 8978, 20, 18, 777, 9, 19233, 1527, 17669, 19, 24, 673, 17, 28756, 150, 12943, 4354, 153, 27, 442, 37, 45, 668, 21, 24, 256, 20, 416, 22, 2771, 4901, 9, 12943, 4354, 153, 51, 24, 3004, 21, 28142, 23, 65, 20, 18, 416, 34, 24, 2958, 22947, 9, 1177, 45, 668, 3097, 13768, 23, 103, 28, 441, 148, 48, 20522, 19, 12943, 4354, 153, 12860, 34, 18, 326, 27, 17492, 684, 21, 6709, 9, 8585, 123, 266, 19, 12943, 4354, 153, 6872, 24, 3004, 20, 18, 9225, 2198, 19, 12717, 103, 22, 401, 24, 6348, 9, 12943, 4354, 153, 1068, 2768, 2286, 19, 33, 104, 19, 176, 24, 9313, 19, 20086, 28, 45, 10292, 9, 4, 3, 19, 12943, 4354, 153, 27, 442, 22, 2771, 4901, 9, 69, 27, 442, 22, 2771, 24, 11335, 20, 18, 9225, 2198, 9, 69, 27, 442, 22, 2771, 24, 11335, 20, 18, 9225, 2198, 9, 69, 27, 442, 22, 2771,
        ]
        # fmt: on
        #  In 1991, the remains of Russian Tsar Nicholas II and his family (except for Alexei and Maria)
        #  are discovered. The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich,
        #  narrates the remainder of the story. 1883 Western Siberia, a young Grigori Rasputin
        #  is asked by his father and a group of men to perform magic. Rasputin has a vision and
        #  denounces one of the men as a horse thief. Although his father initially slaps
        #  him for making such an accusation, Rasputin watches as the man is chased outside and beaten.
        #  Twenty years later, Rasputin sees a vision of the Virgin Mary, prompting him to become a priest.
        #  Rasputin quickly becomes famous, with people, even a bishop, begging for his blessing.
        #  <sep><cls>, Rasputin is asked to perform magic. He is asked to perform a ritual of the Virgin Mary.
        #  He is asked to perform a ritual of the Virgin Mary. He is asked to perform

        output_ids = model.generate(input_ids, max_length=200, do_sample=False)

        self.assertListEqual(output_ids[0].numpy().tolist(), expected_output_ids)
