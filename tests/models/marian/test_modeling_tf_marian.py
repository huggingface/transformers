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


import tempfile
import unittest
import warnings

from transformers import AutoTokenizer, MarianConfig, MarianTokenizer, TranslationPipeline, is_tf_available
from transformers.testing_utils import require_sentencepiece, require_tf, require_tokenizers, slow, tooslow
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers import TFAutoModelForSeq2SeqLM, TFMarianModel, TFMarianMTModel


@require_tf
class TFMarianModelTester:
    config_cls = MarianConfig
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
        inputs_dict = prepare_marian_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = TFMarianModel(config=config).get_decoder()
        input_ids = inputs_dict["input_ids"]

        input_ids = input_ids[:1, :]
        attention_mask = inputs_dict["attention_mask"][:1, :]
        head_mask = inputs_dict["head_mask"]
        self.batch_size = 1

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, head_mask=head_mask, use_cache=True)

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


def prepare_marian_inputs_dict(
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
class TFMarianModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (TFMarianMTModel, TFMarianModel) if is_tf_available() else ()
    all_generative_model_classes = (TFMarianMTModel,) if is_tf_available() else ()
    pipeline_model_mapping = (
        {
            "conversational": TFMarianMTModel,
            "feature-extraction": TFMarianModel,
            "summarization": TFMarianMTModel,
            "text2text-generation": TFMarianMTModel,
            "translation": TFMarianMTModel,
        }
        if is_tf_available()
        else {}
    )
    is_encoder_decoder = True
    test_pruning = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFMarianModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MarianConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_decoder_model_past_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.check_decoder_model_past_large_inputs(*config_and_inputs)

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

    @tooslow
    def test_saved_model_creation(self):
        pass


@require_tf
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
        model: TFMarianMTModel = TFAutoModelForSeq2SeqLM.from_pretrained(self.model_name)
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
        model_inputs = self.tokenizer(self.src_text, **tokenizer_kwargs, padding=True, return_tensors="tf")
        generated_ids = self.model.generate(
            model_inputs.input_ids, attention_mask=model_inputs.attention_mask, num_beams=2, max_length=128
        )
        generated_words = self.tokenizer.batch_decode(generated_ids.numpy(), skip_special_tokens=True)
        return generated_words


@require_sentencepiece
@require_tokenizers
@require_tf
class TestMarian_MT_EN(AbstractMarianIntegrationTest):
    """Cover low resource/high perplexity setting. This breaks if pad_token_id logits not set to LARGE_NEGATIVE."""

    src = "mt"
    tgt = "en"
    src_text = ["Billi messu b'mod ġentili, Ġesù fejjaq raġel li kien milqut bil - marda kerha tal - ġdiem."]
    expected_text = ["Touching gently, Jesus healed a man who was affected by the sad disease of leprosy."]

    @unittest.skip("Skipping until #12647 is resolved.")
    @slow
    def test_batch_generation_mt_en(self):
        self._assert_generated_batch_equal_expected()


@require_sentencepiece
@require_tokenizers
@require_tf
class TestMarian_en_zh(AbstractMarianIntegrationTest):
    src = "en"
    tgt = "zh"
    src_text = ["My name is Wolfgang and I live in Berlin"]
    expected_text = ["我叫沃尔夫冈 我住在柏林"]

    @unittest.skip("Skipping until #12647 is resolved.")
    @slow
    def test_batch_generation_en_zh(self):
        self._assert_generated_batch_equal_expected()


@require_sentencepiece
@require_tokenizers
@require_tf
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

    @unittest.skip("Skipping until #12647 is resolved.")
    @slow
    def test_batch_generation_en_ROMANCE_multi(self):
        self._assert_generated_batch_equal_expected()

    @unittest.skip("Skipping until #12647 is resolved.")
    @slow
    def test_pipeline(self):
        pipeline = TranslationPipeline(self.model, self.tokenizer, framework="tf")
        output = pipeline(self.src_text)
        self.assertEqual(self.expected_text, [x["translation_text"] for x in output])
