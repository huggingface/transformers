# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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

from transformers import OPTConfig, is_tf_available
from transformers.testing_utils import require_sentencepiece, require_tf, slow, tooslow

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers import GPT2Tokenizer, TFOPTForCausalLM, TFOPTModel


def prepare_opt_inputs_dict(config, input_ids, attention_mask=None, head_mask=None):
    if attention_mask is None:
        attention_mask = tf.cast(tf.math.not_equal(input_ids, config.pad_token_id), tf.int8)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


@require_tf
class TFOPTModelTester:
    config_cls = OPTConfig
    config_updates = {}
    hidden_act = "gelu"

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
        embed_dim=16,
        word_embed_proj_dim=16,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.embed_dim = embed_dim
        self.word_embed_proj_dim = word_embed_proj_dim
        self.is_encoder_decoder = False

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length - 1], self.vocab_size)
        eos_tensor = tf.expand_dims(tf.constant([self.eos_token_id] * self.batch_size), 1)
        input_ids = tf.concat([input_ids, eos_tensor], axis=1)

        config = self.config_cls(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            embed_dim=self.embed_dim,
            word_embed_proj_dim=self.word_embed_proj_dim,
            is_encoder_decoder=False,
            **self.config_updates,
        )
        inputs_dict = prepare_opt_inputs_dict(config, input_ids)
        return config, inputs_dict

    def check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = TFOPTModel(config=config)
        input_ids = inputs_dict["input_ids"]

        input_ids = input_ids[:1, :]
        attention_mask = inputs_dict["attention_mask"][:1, :]
        self.batch_size = 1

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_attn_mask = tf.cast(ids_tensor((self.batch_size, 3), 2), tf.int8)

        # append to next input_ids and
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = tf.concat([attention_mask, next_attn_mask], axis=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)[0]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values)[0]

        self.parent.assertEqual(next_tokens.shape[1], output_from_past.shape[1])

        # select random slice
        random_slice_idx = int(ids_tensor((1,), output_from_past.shape[-1]))
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx]
        output_from_past_slice = output_from_past[:, :, random_slice_idx]

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-3)


@require_tf
class TFOPTModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (TFOPTModel, TFOPTForCausalLM) if is_tf_available() else ()
    all_generative_model_classes = (TFOPTForCausalLM,) if is_tf_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": TFOPTModel, "text-generation": TFOPTForCausalLM} if is_tf_available() else {}
    )
    is_encoder_decoder = False
    test_pruning = False
    test_onnx = False
    onnx_min_opset = 10

    def setUp(self):
        self.model_tester = TFOPTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=OPTConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_decoder_model_past_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_model_common_attributes(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            assert isinstance(model.get_input_embeddings(), tf.keras.layers.Layer)

            if model_class in self.all_generative_model_classes:
                x = model.get_output_embeddings()
                assert isinstance(x, tf.keras.layers.Layer)
            else:
                x = model.get_output_embeddings()
                assert x is None

    def test_resize_token_embeddings(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def _get_word_embedding_weight(model, embedding_layer):
            if hasattr(embedding_layer, "weight"):
                return embedding_layer.weight
            else:
                # Here we build the word embeddings weights if not exists.
                # And then we retry to get the attribute once built.
                model(model.dummy_inputs)
                if hasattr(embedding_layer, "weight"):
                    return embedding_layer.weight
                else:
                    return None

        for model_class in self.all_model_classes:
            for size in [config.vocab_size - 10, config.vocab_size + 10]:
                # build the embeddings
                model = model_class(config=config)
                old_input_embeddings = _get_word_embedding_weight(model, model.get_input_embeddings())
                old_output_embeddings = _get_word_embedding_weight(model, model.get_output_embeddings())

                # reshape the embeddings
                model.resize_token_embeddings(size)
                new_input_embeddings = _get_word_embedding_weight(model, model.get_input_embeddings())
                new_output_embeddings = _get_word_embedding_weight(model, model.get_output_embeddings())

                # check that the resized embeddings size matches the desired size.
                assert_size = size if size is not None else config.vocab_size

                self.assertEqual(new_input_embeddings.shape[0], assert_size)

                # check that weights remain the same after resizing
                models_equal = True
                for p1, p2 in zip(old_input_embeddings.value(), new_input_embeddings.value()):
                    if tf.math.reduce_sum(tf.math.abs(p1 - p2)) > 0:
                        models_equal = False
                self.assertTrue(models_equal)

                if old_output_embeddings is not None and new_output_embeddings is not None:
                    self.assertEqual(new_output_embeddings.shape[0], assert_size)

                    models_equal = True
                    for p1, p2 in zip(old_output_embeddings.value(), new_output_embeddings.value()):
                        if tf.math.reduce_sum(tf.math.abs(p1 - p2)) > 0:
                            models_equal = False
                    self.assertTrue(models_equal)

    @tooslow
    def test_saved_model_creation(self):
        pass


def _long_tensor(tok_lst):
    return tf.constant(tok_lst, dtype=tf.int32)


@require_tf
class TFOPTHeadTests(unittest.TestCase):
    vocab_size = 99

    def _get_config_and_data(self):
        eos_column_vector = tf.ones((4, 1), dtype=tf.int32) * 2
        input_ids = tf.concat([ids_tensor((4, 6), self.vocab_size - 3) + 3, eos_column_vector], axis=1)
        batch_size = input_ids.shape[0]
        config = OPTConfig(
            vocab_size=self.vocab_size,
            hidden_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            ffn_dim=32,
            max_position_embeddings=48,
            eos_token_id=2,
            pad_token_id=1,
            bos_token_id=0,
        )
        return config, input_ids, batch_size


@require_sentencepiece
@require_tf
class OPTModelIntegrationTests(unittest.TestCase):
    @slow
    def test_inference_no_head(self):
        model = TFOPTModel.from_pretrained("facebook/opt-350m")
        input_ids = _long_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        attention_mask = tf.not_equal(input_ids, model.config.pad_token_id)
        with tf.GradientTape():
            output = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        expected_shape = (1, 11, 512)
        self.assertEqual(output.shape, expected_shape)
        expected_slice = tf.constant(
            [[-0.2873, -1.9218, -0.3033], [-1.2710, -0.1338, -0.1902], [0.4095, 0.1214, -1.3121]]
        )
        self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=4e-3))

        xla_generate = tf.function(model, jit_compile=True)
        output = xla_generate(input_ids, attention_mask)[0]
        self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=4e-2))


@require_tf
@slow
class TFOPTEmbeddingsTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.path_model = "facebook/opt-350m"

    def test_logits(self):
        model = TFOPTForCausalLM.from_pretrained(self.path_model)
        tokenizer = GPT2Tokenizer.from_pretrained(self.path_model)

        prompts = [
            "Today is a beautiful day and I want to",
            "In the city of",
            "Paris is the capital of France and",
            "Computers and mobile phones have taken",
        ]
        # verify that prompt without BOS token is identical to Metaseq -> add_special_tokens=False
        inputs = tokenizer(prompts, return_tensors="tf", padding=True, add_special_tokens=False)
        logits = tf.math.reduce_mean(model(inputs.input_ids, attention_mask=inputs.attention_mask)[0], axis=-1)
        logits_meta = tf.constant(
            [
                [1.3851, -13.8923, -10.5229, -10.7533, -0.2309, -10.2384, -0.5365, -9.0947, -5.1670],
                [-4.7073, -10.6276, -3.9415, -21.5242, -0.2822, -0.2822, -0.2822, -0.2822, -0.2822],
                [0.6247, -3.4229, -8.9179, -1.4297, -14.1650, 1.4146, -9.0218, -0.2703, -0.2703],
                [6.4783, -1.9913, -10.7926, -2.3336, 1.5092, -0.9974, -6.8213, 1.3477, 1.3477],
            ]
        )
        self.assertTrue(np.allclose(logits, logits_meta, atol=1e-4))

        xla_generate = tf.function(model, jit_compile=True)
        logits = tf.math.reduce_mean(xla_generate(inputs.input_ids, attention_mask=inputs.attention_mask)[0], axis=-1)
        self.assertTrue(np.allclose(logits, logits_meta, atol=1e-4))


@require_tf
@slow
class TFOPTGenerationTest(unittest.TestCase):
    @property
    def prompts(self):
        return [
            "Today is a beautiful day and I want",
            "In the city of",
            "Paris is the capital of France and",
            "Computers and mobile phones have taken",
        ]

    def test_generation_pre_attn_layer_norm(self):
        model_id = "facebook/opt-125m"

        EXPECTED_OUTPUTS = [
            "Today is a beautiful day and I want to",
            "In the city of New York, the city",
            "Paris is the capital of France and the capital",
            "Computers and mobile phones have taken over the",
        ]

        predicted_outputs = []
        tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        model = TFOPTForCausalLM.from_pretrained(model_id)

        for prompt in self.prompts:
            input_ids = tokenizer(prompt, return_tensors="tf").input_ids

            generated_ids = model.generate(input_ids, max_length=10)

            generated_string = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predicted_outputs += generated_string

        self.assertListEqual(predicted_outputs, EXPECTED_OUTPUTS)

    def test_batch_generation(self):
        model_id = "facebook/opt-350m"

        tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        model = TFOPTForCausalLM.from_pretrained(model_id)

        tokenizer.padding_side = "left"

        # use different length sentences to test batching
        sentences = [
            "Hello, my dog is a little",
            "Today, I",
        ]

        inputs = tokenizer(sentences, return_tensors="tf", padding=True)
        input_ids = inputs["input_ids"]

        outputs = model.generate(input_ids=input_ids, attention_mask=inputs["attention_mask"])

        inputs_non_padded = tokenizer(sentences[0], return_tensors="tf").input_ids
        output_non_padded = model.generate(input_ids=inputs_non_padded)

        num_paddings = inputs_non_padded.shape[-1] - tf.math.reduce_sum(
            tf.cast(inputs["attention_mask"][-1], tf.int64)
        )
        inputs_padded = tokenizer(sentences[1], return_tensors="tf").input_ids
        output_padded = model.generate(input_ids=inputs_padded, max_length=model.config.max_length - num_paddings)

        batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        non_padded_sentence = tokenizer.decode(output_non_padded[0], skip_special_tokens=True)
        padded_sentence = tokenizer.decode(output_padded[0], skip_special_tokens=True)

        expected_output_sentence = [
            "Hello, my dog is a little bit of a dork.\nI'm a little bit",
            "Today, I was in the middle of a conversation with a friend about the",
        ]
        self.assertListEqual(expected_output_sentence, batch_out_sentence)
        self.assertListEqual(batch_out_sentence, [non_padded_sentence, padded_sentence])

    def test_generation_post_attn_layer_norm(self):
        model_id = "facebook/opt-350m"

        EXPECTED_OUTPUTS = [
            "Today is a beautiful day and I want to",
            "In the city of San Francisco, the city",
            "Paris is the capital of France and the capital",
            "Computers and mobile phones have taken over the",
        ]

        predicted_outputs = []
        tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        model = TFOPTForCausalLM.from_pretrained(model_id)

        for prompt in self.prompts:
            input_ids = tokenizer(prompt, return_tensors="tf").input_ids

            generated_ids = model.generate(input_ids, max_length=10)

            generated_string = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predicted_outputs += generated_string

        self.assertListEqual(predicted_outputs, EXPECTED_OUTPUTS)
