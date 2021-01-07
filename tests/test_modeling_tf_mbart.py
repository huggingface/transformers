# coding=utf-8
# Copyright 2020 HuggingFace Inc. team.
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
import tempfile
import unittest

from tests.test_configuration_common import ConfigTester
from tests.test_modeling_tf_bart import TFBartModelTester
from tests.test_modeling_tf_common import TFModelTesterMixin
from transformers import AutoTokenizer, MBartConfig, is_tf_available
from transformers.file_utils import cached_property
from transformers.testing_utils import is_pt_tf_cross_test, require_sentencepiece, require_tf, require_tokenizers, slow


if is_tf_available():

    import tensorflow as tf

    from transformers import TFAutoModelForSeq2SeqLM, TFMBartForConditionalGeneration


class ModelTester(TFBartModelTester):
    config_updates = dict(normalize_before=True, add_final_layer_norm=True)
    config_cls = MBartConfig


@require_tf
class TFMBartModelTest(TFModelTesterMixin, unittest.TestCase):
    all_model_classes = (TFMBartForConditionalGeneration,) if is_tf_available() else ()
    all_generative_model_classes = (TFMBartForConditionalGeneration,) if is_tf_available() else ()
    model_tester_cls = ModelTester
    is_encoder_decoder = True
    test_pruning = False

    def setUp(self):
        self.model_tester = self.model_tester_cls(self)
        self.config_tester = ConfigTester(self, config_class=MBartConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_inputs_embeds(self):
        # inputs_embeds not supported
        pass

    def test_compile_tf_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")

        model_class = self.all_generative_model_classes[0]
        input_ids = {
            "decoder_input_ids": tf.keras.Input(batch_shape=(2, 2000), name="decoder_input_ids", dtype="int32"),
            "input_ids": tf.keras.Input(batch_shape=(2, 2000), name="input_ids", dtype="int32"),
        }

        # Prepare our model
        model = model_class(config)
        model(self._prepare_for_class(inputs_dict, model_class))  # Model must be called before saving.
        # Let's load it from the disk to be sure we can use pretrained weights
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            model = model_class.from_pretrained(tmpdirname)

        outputs_dict = model(input_ids)
        hidden_states = outputs_dict[0]

        # Add a dense layer on top to test integration with other keras modules
        outputs = tf.keras.layers.Dense(2, activation="softmax", name="outputs")(hidden_states)

        # Compile extended model
        extended_model = tf.keras.Model(inputs=[input_ids], outputs=[outputs])
        extended_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

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


@is_pt_tf_cross_test
@require_sentencepiece
@require_tokenizers
class TestMBartEnRO(unittest.TestCase):
    src_text = [
        " UN Chief Says There Is No Military Solution in Syria",
    ]
    expected_text = [
        "Şeful ONU declară că nu există o soluţie militară în Siria",
    ]
    model_name = "facebook/mbart-large-en-ro"

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    @cached_property
    def model(self):
        model = TFAutoModelForSeq2SeqLM.from_pretrained(self.model_name, from_pt=True)
        return model

    def _assert_generated_batch_equal_expected(self, **tokenizer_kwargs):
        generated_words = self.translate_src_text(**tokenizer_kwargs)
        self.assertListEqual(self.expected_text, generated_words)

    def translate_src_text(self, **tokenizer_kwargs):
        model_inputs = self.tokenizer.prepare_seq2seq_batch(
            src_texts=self.src_text, **tokenizer_kwargs, return_tensors="tf"
        )
        generated_ids = self.model.generate(
            model_inputs.input_ids, attention_mask=model_inputs.attention_mask, num_beams=2
        )
        generated_words = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_words

    @slow
    def test_batch_generation_en_ro(self):
        self._assert_generated_batch_equal_expected()
