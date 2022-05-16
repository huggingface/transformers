# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from transformers import XGLMConfig, XGLMTokenizer, is_tf_available
from transformers.testing_utils import require_tf, slow

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_tf_available():
    import tensorflow as tf

    from transformers.models.xglm.modeling_tf_xglm import (
        TF_XGLM_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFXGLMForCausalLM,
        TFXGLMModel,
        shape_list,
    )


class TFXGLMModelTester:
    def __init__(
        self,
        parent,
        batch_size=14,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        d_model=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        ffn_dim=37,
        activation_function="gelu",
        activation_dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = d_model
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.ffn_dim = ffn_dim
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = None
        self.bos_token_id = 0
        self.eos_token_id = 2
        self.pad_token_id = 1

    def get_large_model_config(self):
        return XGLMConfig.from_pretrained("facebook/xglm-564M")

    def prepare_config_and_inputs(self):
        input_ids = tf.clip_by_value(
            ids_tensor([self.batch_size, self.seq_length], self.vocab_size), clip_value_min=0, clip_value_max=3
        )

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        head_mask = floats_tensor([self.num_hidden_layers, self.num_attention_heads], 2)

        return (
            config,
            input_ids,
            input_mask,
            head_mask,
        )

    def get_config(self):
        return XGLMConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            num_layers=self.num_hidden_layers,
            attention_heads=self.num_attention_heads,
            ffn_dim=self.ffn_dim,
            activation_function=self.activation_function,
            activation_dropout=self.activation_dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            use_cache=True,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            return_dict=True,
        )

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            input_mask,
            head_mask,
        ) = self.prepare_config_and_inputs()

        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            input_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def create_and_check_xglm_model(self, config, input_ids, input_mask, head_mask, *args):
        model = TFXGLMModel(config=config)

        result = model(input_ids, head_mask=head_mask)
        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(len(result.past_key_values), config.num_hidden_layers)

    def create_and_check_xglm_model_past(self, config, input_ids, input_mask, head_mask, *args):
        model = TFXGLMModel(config=config)

        # first forward pass
        outputs = model(input_ids, use_cache=True)
        outputs_no_past = model(input_ids, use_cache=False)

        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        output, past = outputs.to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # append to next input_ids and token_type_ids
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)

        output_from_no_past = model(next_input_ids)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past)["last_hidden_state"]

        # select random slice
        random_slice_idx = int(ids_tensor((1,), shape_list(output_from_past)[-1]))
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx]
        output_from_past_slice = output_from_past[:, 0, random_slice_idx]

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-6)

    def create_and_check_xglm_model_attention_mask_past(self, config, input_ids, input_mask, head_mask, *args):
        model = TFXGLMModel(config=config)

        # create attention mask
        half_seq_length = self.seq_length // 2
        attn_mask_begin = tf.ones((self.batch_size, half_seq_length), dtype=tf.int32)
        attn_mask_end = tf.zeros((self.batch_size, self.seq_length - half_seq_length), dtype=tf.int32)
        attn_mask = tf.concat([attn_mask_begin, attn_mask_end], axis=1)

        # first forward pass
        output, past = model(input_ids, attention_mask=attn_mask).to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # append to next input_ids and attn_mask
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        attn_mask = tf.concat([attn_mask, tf.ones((shape_list(attn_mask)[0], 1), dtype=tf.int32)], axis=1)

        # get two different outputs
        output_from_no_past = model(next_input_ids, attention_mask=attn_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past, attention_mask=attn_mask)["last_hidden_state"]

        # select random slice
        random_slice_idx = int(ids_tensor((1,), shape_list(output_from_past)[-1]))
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx]
        output_from_past_slice = output_from_past[:, 0, random_slice_idx]

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-12)

    def create_and_check_xglm_model_past_large_inputs(self, config, input_ids, input_mask, head_mask, *args):
        model = TFXGLMModel(config=config)

        # first forward pass
        outputs = model(input_ids, attention_mask=input_mask, use_cache=True)

        output, past = outputs.to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=1)

        # append to next input_ids
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = tf.concat([input_mask, next_mask], axis=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past)[
            "last_hidden_state"
        ]
        self.parent.assertTrue(output_from_past.shape[1] == next_tokens.shape[1])

        # select random slice
        random_slice_idx = int(ids_tensor((1,), shape_list(output_from_past)[-1]))
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx]
        output_from_past_slice = output_from_past[:, :, random_slice_idx]

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-3)

    def create_and_check_lm_head_model(self, config, input_ids, input_mask, head_mask, *args):
        model = TFXGLMForCausalLM(config)

        result = model(input_ids)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()

        (
            config,
            input_ids,
            input_mask,
            head_mask,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "head_mask": head_mask,
        }

        return config, inputs_dict


@require_tf
class TFXGLMModelTest(TFModelTesterMixin, unittest.TestCase):

    all_model_classes = (TFXGLMModel, TFXGLMForCausalLM) if is_tf_available() else ()
    all_generative_model_classes = (TFXGLMForCausalLM,) if is_tf_available() else ()
    test_onnx = False
    test_missing_keys = False
    test_pruning = False

    def setUp(self):
        self.model_tester = TFXGLMModelTester(self)
        self.config_tester = ConfigTester(self, config_class=XGLMConfig, n_embd=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_xglm_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xglm_model(*config_and_inputs)

    def test_xglm_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xglm_model_past(*config_and_inputs)

    def test_xglm_model_att_mask_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xglm_model_attention_mask_past(*config_and_inputs)

    def test_xglm_model_past_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xglm_model_past_large_inputs(*config_and_inputs)

    def test_xglm_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_model_common_attributes(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            assert isinstance(model.get_input_embeddings(), tf.keras.layers.Layer)

            if model_class in self.all_generative_model_classes:
                x = model.get_output_embeddings()
                assert isinstance(x, tf.keras.layers.Layer)
                name = model.get_bias()
                assert name is None
            else:
                x = model.get_output_embeddings()
                assert x is None
                name = model.get_bias()
                assert name is None

    @slow
    def test_batch_generation(self):
        model = TFXGLMForCausalLM.from_pretrained("facebook/xglm-564M", from_pt=True)
        tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-564M")

        tokenizer.padding_side = "left"

        # use different length sentences to test batching
        sentences = [
            "Hello, my dog is a little",
            "Today, I",
        ]

        inputs = tokenizer(sentences, return_tensors="tf", padding=True)

        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        inputs_non_padded = tokenizer(sentences[0], return_tensors="tf").input_ids
        output_non_padded = model.generate(input_ids=inputs_non_padded)

        num_paddings = (
            inputs_non_padded.shape[-1]
            - tf.math.reduce_sum(tf.cast(inputs["attention_mask"][-1], dtype=tf.int64)).numpy()
        )
        inputs_padded = tokenizer(sentences[1], return_tensors="tf").input_ids
        output_padded = model.generate(input_ids=inputs_padded, max_length=model.config.max_length - num_paddings)

        batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        non_padded_sentence = tokenizer.decode(output_non_padded[0], skip_special_tokens=True)
        padded_sentence = tokenizer.decode(output_padded[0], skip_special_tokens=True)

        expected_output_sentence = [
            "Hello, my dog is a little bit of a shy one, but he is very friendly",
            "Today, I am going to share with you a few of my favorite things",
        ]
        self.assertListEqual(expected_output_sentence, batch_out_sentence)
        self.assertListEqual(expected_output_sentence, [non_padded_sentence, padded_sentence])

    @slow
    def test_model_from_pretrained(self):
        for model_name in TF_XGLM_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFXGLMModel.from_pretrained(model_name, from_pt=True)
            self.assertIsNotNone(model)

    @unittest.skip(reason="Currently, model embeddings are going to undergo a major refactor.")
    def test_resize_token_embeddings(self):
        super().test_resize_token_embeddings()


@require_tf
class TFXGLMModelLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_xglm(self, verify_outputs=True):
        model = TFXGLMForCausalLM.from_pretrained("facebook/xglm-564M", from_pt=True)
        input_ids = tf.convert_to_tensor([2, 268, 9865], dtype=tf.int32)  # The dog
        # </s> The dog is a very friendly dog. He is very affectionate and loves to play with other
        # fmt: off
        expected_output_ids = [2, 268, 9865, 67, 11, 1988, 57252, 9865, 5, 984, 67, 1988, 213838, 1658, 53, 70446, 33, 6657, 278, 1581]
        # fmt: on
        output_ids = model.generate(input_ids, do_sample=False, num_beams=1)
        if verify_outputs:
            self.assertListEqual(output_ids[0].tolist(), expected_output_ids)

    @slow
    def test_xglm_sample(self):
        tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-564M")
        model = TFXGLMForCausalLM.from_pretrained("facebook/xglm-564M", from_pt=True)

        tf.random.set_seed(0)
        tokenized = tokenizer("Today is a nice day and", return_tensors="tf")
        input_ids = tokenized.input_ids
        output_ids = model.generate(input_ids, do_sample=True, num_beams=1)
        output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        EXPECTED_OUTPUT_STR = "Today is a nice day and the sun is shining. A nice day with warm rainy"
        self.assertEqual(output_str, EXPECTED_OUTPUT_STR)
