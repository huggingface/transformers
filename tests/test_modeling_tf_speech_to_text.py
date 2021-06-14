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
import copy
import inspect
import tempfile
import unittest

import numpy as np

from transformers import Speech2TextConfig
from transformers.file_utils import cached_property
from transformers.testing_utils import is_tf_available, require_sentencepiece, require_tf, require_tokenizers, slow

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor


if is_tf_available():
    import tensorflow as tf

    from transformers import Speech2TextProcessor, TFSpeech2TextForConditionalGeneration, TFSpeech2TextModel


@require_tf
def prepare_speech_to_text_inputs_dict(
    config,
    input_features,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if attention_mask is None:
        attention_mask = tf.cast(tf.math.not_equal(input_features, 0), tf.float32)
    if decoder_attention_mask is None:
        decoder_attention_mask = tf.cast(tf.math.not_equal(decoder_input_ids, config.pad_token_id), tf.float32)
    if head_mask is None:
        head_mask = tf.ones((config.encoder_layers, config.encoder_attention_heads))
    if decoder_head_mask is None:
        decoder_head_mask = tf.ones((config.decoder_layers, config.decoder_attention_heads))
    if cross_attn_head_mask is None:
        cross_attn_head_mask = tf.ones((config.decoder_layers, config.decoder_attention_heads))
    return {
        # "input_ids": input_features,
        "input_features": input_features,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
    }


@require_tf
class TFSpeech2TextModelTester:
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
        num_conv_layers=2,
        conv_kernel_sizes=(5, 5),
        conv_channels=32,
        input_feat_per_channel=24,
        input_channels=1,
        hidden_act="relu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        max_source_positions=20,
        max_target_positions=20,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
        scope=None,
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
        self.num_conv_layers = num_conv_layers
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_channels = conv_channels
        self.input_feat_per_channel = input_feat_per_channel
        self.input_channels = input_channels
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_features = tf.cast(
            ids_tensor([self.batch_size, self.seq_length, self.input_feat_per_channel], self.vocab_size), tf.float32
        )
        attention_mask = tf.ones((self.batch_size, self.seq_length))
        decoder_input_ids = tf.clip_by_value(ids_tensor([self.batch_size, self.seq_length], self.vocab_size), 0, 2)

        config = self.get_config()
        inputs_dict = prepare_speech_to_text_inputs_dict(
            config,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
        )
        return config, inputs_dict

    def get_config(self):
        return Speech2TextConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            num_conv_layers=self.num_conv_layers,
            conv_kernel_sizes=self.conv_kernel_sizes,
            conv_channels=self.conv_channels,
            input_feat_per_channel=self.input_feat_per_channel,
            input_channels=self.input_channels,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            max_source_positions=self.max_source_positions,
            max_target_positions=self.max_target_positions,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
        )

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_subsampled_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """

        for i in range(self.num_conv_layers):
            input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = TFSpeech2TextModel(config=config).get_decoder()
        input_ids = inputs_dict["decoder_input_ids"]
        attention_mask = inputs_dict["decoder_attention_mask"]

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical next token and extent to next_input_features
        next_tokens = tf.clip_by_value(ids_tensor((self.batch_size, 3), config.vocab_size), 0, 2)
        next_attn_mask = tf.cast(ids_tensor((self.batch_size, 3), 2), tf.float32)

        # append to next input_features and
        next_input_features = tf.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = tf.concat([attention_mask, next_attn_mask], axis=-1)

        output_from_no_past = model(next_input_features, attention_mask=next_attention_mask)[0]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values)[0]

        # select random slice
        random_slice_idx = int(ids_tensor((1,), output_from_past.shape[-1]))
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx]
        output_from_past_slice = output_from_past[:, :, random_slice_idx]

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-2, atol=1e-2)


@require_tf
class TFSpeech2TextModelTest(TFModelTesterMixin, unittest.TestCase):
    all_model_classes = (TFSpeech2TextForConditionalGeneration, TFSpeech2TextModel) if is_tf_available() else ()
    all_generative_model_classes = (TFSpeech2TextForConditionalGeneration,) if is_tf_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFSpeech2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Speech2TextConfig)
        self.maxDiff = 3000

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_save_load_strict(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model2, info = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info["missing_keys"], [])

    def test_decoder_model_past_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_inputs_embeds(self):
        pass

    def test_training(self):
        pass

    def test_generate_fp16(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.call)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["input_features"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
            else:
                seq_length = self.model_tester.seq_length

            subsampled_seq_length = model.model._get_subsampled_output_lengths(seq_length)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [subsampled_seq_length, self.model_tester.hidden_size],
            )

            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                seq_len = getattr(self.model_tester, "seq_length", None)
                decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)

                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [decoder_seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)

            subsampled_encoder_seq_length = model.model._get_subsampled_output_lengths(encoder_seq_length)
            subsampled_encoder_key_length = model.model._get_subsampled_output_lengths(encoder_key_length)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, subsampled_encoder_seq_length, subsampled_encoder_key_length],
            )
            out_len = len(outputs)

            correct_outlen = 5

            # loss is at first position
            if "labels" in inputs_dict:
                correct_outlen += 1  # loss is added to beginning
            if "past_key_values" in outputs:
                correct_outlen += 1  # past_key_values have been returned

            self.assertEqual(out_len, correct_outlen)

            # decoder attentions
            decoder_attentions = outputs.decoder_attentions
            self.assertIsInstance(decoder_attentions, (list, tuple))
            self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(decoder_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
            )

            # cross attentions
            cross_attentions = outputs.cross_attentions
            self.assertIsInstance(cross_attentions, (list, tuple))
            self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(cross_attentions[0].shape[-3:]),
                [
                    self.model_tester.num_attention_heads,
                    decoder_seq_length,
                    subsampled_encoder_key_length,
                ],
            )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            added_hidden_states = 2
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, subsampled_encoder_seq_length, subsampled_encoder_key_length],
            )

    def test_compile_tf_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")

        for model_class in self.all_model_classes:
            input_ids = {
                "input_features": tf.keras.Input(
                    batch_shape=(
                        self.model_tester.batch_size,
                        config.max_source_positions,
                        config.input_feat_per_channel,
                    ),
                    name="input_features",
                    dtype="float32",
                ),
                "decoder_input_ids": tf.keras.Input(
                    batch_shape=(self.model_tester.batch_size, config.max_target_positions),
                    name="decoder_input_ids",
                    dtype="int32",
                ),
            }
            # Prepare our model
            model = model_class(config)
            model(self._prepare_for_class(inputs_dict, model_class))  # Model must be called before saving.
            # Let's load it from the disk to be sure we can use pretrained weights
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, saved_model=False)
                model = model_class.from_pretrained(tmpdirname)

            outputs_dict = model(input_ids)
            hidden_states = outputs_dict[0]

            # Add a dense layer on top to test integration with other keras modules
            outputs = tf.keras.layers.Dense(2, activation="softmax", name="outputs")(hidden_states)

            # Compile extended model
            extended_model = tf.keras.Model(inputs=[input_ids], outputs=[outputs])
            extended_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    def test_keyword_and_dict_args(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            inputs = self._prepare_for_class(inputs_dict, model_class)

            outputs_dict = model(inputs)

            inputs_keywords = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))
            input_values = inputs_keywords.pop("input_features", None)
            outputs_keywords = model(input_values, **inputs_keywords)
            output_dict = outputs_dict[0].numpy()
            output_keywords = outputs_keywords[0].numpy()

            self.assertLess(np.sum(np.abs(output_dict - output_keywords)), 1e-6)

    def test_resize_token_embeddings(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            return

        def _get_word_embedding_weight(model, embedding_layer):
            embeds = getattr(embedding_layer, "weight", None)
            if embeds is not None:
                return embeds

            embeds = getattr(embedding_layer, "decoder", None)
            if embeds is not None:
                return embeds

            model(model.dummy_inputs)

            embeds = getattr(embedding_layer, "weight", None)
            if embeds is not None:
                return embeds

            embeds = getattr(embedding_layer, "decoder", None)
            if embeds is not None:
                return embeds

            return None

        for model_class in self.all_model_classes:
            for size in [config.vocab_size - 10, config.vocab_size + 10, None]:
                # build the embeddings
                model = model_class(config=config)
                old_input_embeddings = _get_word_embedding_weight(model, model.get_input_embeddings())
                old_bias = model.get_bias()
                old_output_embeddings = _get_word_embedding_weight(model, model.get_output_embeddings())
                # reshape the embeddings
                model.resize_token_embeddings(size)
                new_input_embeddings = _get_word_embedding_weight(model, model.get_input_embeddings())
                new_bias = model.get_bias()
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

                if old_bias is not None and new_bias is not None:
                    for old_weight, new_weight in zip(old_bias.values(), new_bias.values()):
                        self.assertEqual(new_weight.shape[0], assert_size)

                        models_equal = True
                        for p1, p2 in zip(old_weight.value(), new_weight.value()):
                            if tf.math.reduce_sum(tf.math.abs(p1 - p2)) > 0:
                                models_equal = False
                        self.assertTrue(models_equal)

                if old_output_embeddings is not None and new_output_embeddings is not None:
                    self.assertEqual(new_output_embeddings.shape[0], assert_size)
                    self.assertEqual(new_output_embeddings.shape[1], old_output_embeddings.shape[1])

                    models_equal = True
                    for p1, p2 in zip(old_output_embeddings.value(), new_output_embeddings.value()):
                        if tf.math.reduce_sum(tf.math.abs(p1 - p2)) > 0:
                            models_equal = False
                    self.assertTrue(models_equal)

    def test_resize_embeddings_untied(self):
        pass

    def test_generate_without_input_ids(self):
        pass

    @staticmethod
    def _get_encoder_outputs(
        model, input_ids, attention_mask, output_attentions=None, output_hidden_states=None, num_interleave=1
    ):
        pass

    def _check_outputs(self, output, input_ids, config, use_cache=False, num_return_sequences=1):
        pass

    def test_loss_computation(self):
        pass


@slow
@require_sentencepiece
@require_tokenizers
@require_tf
class TFSpeech2TextModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

    @staticmethod
    def _load_datasamples(num_samples):
        from datasets import load_dataset

        import soundfile as sf

        # map files to raw
        def map_to_array(batch):
            speech, _ = sf.read(batch["file"])
            batch["speech"] = speech
            return batch

        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.select(range(num_samples)).map(map_to_array)

        return ds["speech"][:num_samples]

    def test_generation_librispeech(self):
        model = TFSpeech2TextForConditionalGeneration.from_pretrained(
            "facebook/s2t-small-librispeech-asr", from_pt=True
        )
        processor = self.default_processor

        input_speech = self._load_datasamples(1)

        input_features = processor(input_speech, return_tensors="tf").input_features

        generated_ids = model.generate(input_features)
        generated_transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)

        EXPECTED_TRANSCRIPTIONS = ["a man said to the universe sir i exist"]
        self.assertListEqual(generated_transcript, EXPECTED_TRANSCRIPTIONS)

    def test_generation_librispeech_batched(self):
        model = TFSpeech2TextForConditionalGeneration.from_pretrained(
            "facebook/s2t-small-librispeech-asr", from_pt=True
        )
        processor = self.default_processor

        input_speech = self._load_datasamples(4)

        inputs = processor(input_speech, return_tensors="tf", padding=True)

        input_features = inputs.input_features
        attention_mask = inputs.attention_mask

        generated_ids = model.generate(input_features, attention_mask=attention_mask)
        generated_transcripts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe sir i exist",
            "sweat covered brion's body trickling into the titleing cloth that was the only garment he wore",
            "the cut on his chest still dripping blood the ache of his overstrained eyes even the soaring arena around him with the thousands of spectators were trivialities not worth thinking about",
            "his instant of panic was followed by a small sharp blow high on his chest",
        ]

        self.assertListEqual(generated_transcripts, EXPECTED_TRANSCRIPTIONS)
