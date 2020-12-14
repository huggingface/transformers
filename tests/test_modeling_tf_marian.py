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
import warnings

from transformers import AutoTokenizer, MarianConfig, MarianTokenizer, TranslationPipeline, is_tf_available
from transformers.file_utils import cached_property
from transformers.testing_utils import is_pt_tf_cross_test, require_sentencepiece, require_tf, require_tokenizers, slow

from .test_configuration_common import ConfigTester
from .test_modeling_tf_bart import TFBartModelTester
from .test_modeling_tf_common import TFModelTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers import TFAutoModelForSeq2SeqLM, TFMarianMTModel


class ModelTester(TFBartModelTester):
    config_updates = dict(static_position_embeddings=True, add_bias_logits=True)
    config_cls = MarianConfig


@require_tf
class TestTFMarianCommon(TFModelTesterMixin, unittest.TestCase):
    all_model_classes = (TFMarianMTModel,) if is_tf_available() else ()
    all_generative_model_classes = (TFMarianMTModel,) if is_tf_available() else ()
    model_tester_cls = ModelTester
    is_encoder_decoder = True
    test_pruning = False

    def setUp(self):
        self.model_tester = self.model_tester_cls(self)
        self.config_tester = ConfigTester(self, config_class=MarianConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_inputs_embeds(self):
        # inputs_embeds not supported
        pass

    def test_saved_model_with_hidden_states_output(self):
        # Should be uncommented during patrick TF refactor
        pass

    def test_saved_model_with_attentions_output(self):
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
        # Let's load it from the disk to be sure we can use pre-trained weights
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
            x = model.get_output_layer_with_bias()
            assert x is None
            name = model.get_prefix_bias_name()
            assert name is None


class AbstractMarianIntegrationTest(unittest.TestCase):
    maxDiff = 1000  # show more chars for failing integration tests

    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = f"Helsinki-NLP/opus-mt-{cls.src}-{cls.tgt}"
        return cls

    @cached_property
    def tokenizer(self) -> MarianTokenizer:
        return AutoTokenizer.from_pretrained(self.model_name)

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @cached_property
    def model(self):
        warnings.simplefilter("error")
        model: TFMarianMTModel = TFAutoModelForSeq2SeqLM.from_pretrained(self.model_name, from_pt=True)
        assert isinstance(model, TFMarianMTModel)
        c = model.config
        self.assertListEqual(c.bad_words_ids, [[c.pad_token_id]])
        self.assertEqual(c.max_length, 512)
        self.assertEqual(c.decoder_start_token_id, c.pad_token_id)
        return model

    def _assert_generated_batch_equal_expected(self, **tokenizer_kwargs):
        generated_words = self.translate_src_text(**tokenizer_kwargs)
        self.assertListEqual(self.expected_text, generated_words)

    def translate_src_text(self, **tokenizer_kwargs):
        model_inputs = self.tokenizer.prepare_seq2seq_batch(
            src_texts=self.src_text, **tokenizer_kwargs, return_tensors="tf"
        )
        generated_ids = self.model.generate(
            model_inputs.input_ids, attention_mask=model_inputs.attention_mask, num_beams=2, max_length=128
        )
        generated_words = self.tokenizer.batch_decode(generated_ids.numpy(), skip_special_tokens=True)
        return generated_words


@require_sentencepiece
@require_tokenizers
@is_pt_tf_cross_test
class TestMarian_MT_EN(AbstractMarianIntegrationTest):
    """Cover low resource/high perplexity setting. This breaks if pad_token_id logits not set to LARGE_NEGATIVE."""

    src = "mt"
    tgt = "en"
    src_text = ["Billi messu b'mod ġentili, Ġesù fejjaq raġel li kien milqut bil - marda kerha tal - ġdiem."]
    expected_text = ["Touching gently, Jesus healed a man who was affected by the sad disease of leprosy."]

    @slow
    def test_batch_generation_mt_en(self):
        self._assert_generated_batch_equal_expected()


@is_pt_tf_cross_test
@require_sentencepiece
@require_tokenizers
class TestMarian_en_zh(AbstractMarianIntegrationTest):
    src = "en"
    tgt = "zh"
    src_text = ["My name is Wolfgang and I live in Berlin"]
    expected_text = ["我叫沃尔夫冈 我住在柏林"]

    @slow
    def test_batch_generation_en_zh(self):
        self._assert_generated_batch_equal_expected()


@is_pt_tf_cross_test
@require_sentencepiece
@require_tokenizers
class TestMarian_en_ROMANCE(AbstractMarianIntegrationTest):
    """Multilingual on target side."""

    src = "en"
    tgt = "ROMANCE"
    src_text = [
        ">>fr<< Don't spend so much time watching TV.",
        ">>pt<< Your message has been sent.",
        ">>es<< He's two years older than me.",
    ]
    expected_text = [
        "Ne passez pas autant de temps à regarder la télé.",
        "A sua mensagem foi enviada.",
        "Es dos años más viejo que yo.",
    ]

    @slow
    def test_batch_generation_en_ROMANCE_multi(self):
        self._assert_generated_batch_equal_expected()

    @slow
    def test_pipeline(self):
        pipeline = TranslationPipeline(self.model, self.tokenizer, framework="tf")
        output = pipeline(self.src_text)
        self.assertEqual(self.expected_text, [x["translation_text"] for x in output])
