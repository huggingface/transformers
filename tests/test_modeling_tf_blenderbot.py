# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

from transformers import BlenderbotConfig, BlenderbotTokenizer, is_tf_available
from transformers.file_utils import cached_property
from transformers.testing_utils import require_tf, require_tokenizers, slow

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor


if is_tf_available():
    import tensorflow as tf

    from transformers import TFAutoModelForSeq2SeqLM, TFBlenderbotForConditionalGeneration, TFBlenderbotModel


@require_tf
class TFBlenderbotModelTester:
    config_cls = BlenderbotConfig
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
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
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

        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length - 1], self.vocab_size)
        eos_tensor = tf.expand_dims(tf.constant([self.eos_token_id] * self.batch_size), 1)
        input_ids = tf.concat([input_ids, eos_tensor], axis=1)

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.config_cls(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_ids=[2],
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.pad_token_id,
            **self.config_updates,
        )
        inputs_dict = prepare_blenderbot_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = TFBlenderbotModel(config=config).get_decoder()
        input_ids = inputs_dict["input_ids"]

        input_ids = input_ids[:1, :]
        attention_mask = inputs_dict["attention_mask"][:1, :]
        head_mask = inputs_dict["head_mask"]
        self.batch_size = 1

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, head_mask=head_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()
        past_key_values = past_key_values[1]

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


def prepare_blenderbot_inputs_dict(
    config,
    input_ids,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if attention_mask is None:
        attention_mask = tf.cast(tf.math.not_equal(input_ids, config.pad_token_id), tf.int8)
    if decoder_attention_mask is None:
        decoder_attention_mask = tf.concat(
            [
                tf.ones(decoder_input_ids[:, :1].shape, dtype=tf.int8),
                tf.cast(tf.math.not_equal(decoder_input_ids[:, 1:], config.pad_token_id), tf.int8),
            ],
            axis=-1,
        )
    if head_mask is None:
        head_mask = tf.ones((config.encoder_layers, config.encoder_attention_heads))
    if decoder_head_mask is None:
        decoder_head_mask = tf.ones((config.decoder_layers, config.decoder_attention_heads))
    if cross_attn_head_mask is None:
        cross_attn_head_mask = tf.ones((config.decoder_layers, config.decoder_attention_heads))
    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
    }


@require_tf
class TFBlenderbotModelTest(TFModelTesterMixin, unittest.TestCase):
    all_model_classes = (TFBlenderbotForConditionalGeneration, TFBlenderbotModel) if is_tf_available() else ()
    all_generative_model_classes = (TFBlenderbotForConditionalGeneration,) if is_tf_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFBlenderbotModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BlenderbotConfig)

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
                name = model.get_bias()
                assert isinstance(name, dict)
                for k, v in name.items():
                    assert isinstance(v, tf.Variable)
            else:
                x = model.get_output_embeddings()
                assert x is None
                name = model.get_bias()
                assert name is None

    def test_saved_model_creation(self):
        # This test is too long (>30sec) and makes fail the CI
        pass

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
            for size in [config.vocab_size - 10, config.vocab_size + 10, None]:
                # build the embeddings
                model = model_class(config=config)
                old_input_embeddings = _get_word_embedding_weight(model, model.get_input_embeddings())
                old_output_embeddings = _get_word_embedding_weight(model, model.get_output_embeddings())
                old_final_logits_bias = model.get_bias()

                # reshape the embeddings
                model.resize_token_embeddings(size)
                new_input_embeddings = _get_word_embedding_weight(model, model.get_input_embeddings())
                new_output_embeddings = _get_word_embedding_weight(model, model.get_output_embeddings())
                new_final_logits_bias = model.get_bias()

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

                if old_final_logits_bias is not None and new_final_logits_bias is not None:
                    old_final_logits_bias = old_final_logits_bias["final_logits_bias"]
                    new_final_logits_bias = new_final_logits_bias["final_logits_bias"]
                    self.assertEqual(new_final_logits_bias.shape[0], 1)
                    self.assertEqual(new_final_logits_bias.shape[1], assert_size)

                    models_equal = True
                    for old, new in zip(old_final_logits_bias.value(), new_final_logits_bias.value()):
                        for p1, p2 in zip(old, new):
                            if tf.math.reduce_sum(tf.math.abs(p1 - p2)) > 0:
                                models_equal = False
                    self.assertTrue(models_equal)


def _assert_tensors_equal(a, b, atol=1e-12, prefix=""):
    """If tensors not close, or a and b arent both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if tf.debugging.assert_near(a, b, atol=atol):
            return True
        raise
    except Exception:
        if len(prefix) > 0:
            prefix = f"{prefix}: "
        raise AssertionError(f"{prefix}{a} != {b}")


def _long_tensor(tok_lst):
    return tf.constant(tok_lst, dtype=tf.int32)


@require_tokenizers
@require_tf
class TFBlenderbot400MIntegrationTests(unittest.TestCase):
    src_text = ["My friends are cool but they eat too many carbs."]
    model_name = "facebook/blenderbot-400M-distill"

    @cached_property
    def tokenizer(self):
        return BlenderbotTokenizer.from_pretrained(self.model_name)

    @cached_property
    def model(self):
        model = TFAutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        return model

    @slow
    def test_generation_from_long_input(self):
        model_inputs = self.tokenizer(self.src_text, return_tensors="tf")
        generated_ids = self.model.generate(
            model_inputs.input_ids,
        )
        generated_words = self.tokenizer.batch_decode(generated_ids.numpy(), skip_special_tokens=True)[0]
        assert (
            generated_words
            == " That's unfortunate. Are they trying to lose weight or are they just trying to be healthier?"
        )
