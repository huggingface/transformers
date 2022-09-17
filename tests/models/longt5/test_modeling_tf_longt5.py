# coding=utf-8
# Copyright 2022 Google LongT5 Authors and HuggingFace Inc. team.
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

from transformers import LongT5Config, is_tf_available
from transformers.testing_utils import require_sentencepiece, require_tf, require_tokenizers, slow, tooslow
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor, random_attention_mask


if is_tf_available():
    import tensorflow as tf

    from transformers import AutoTokenizer, TFLongT5EncoderModel, TFLongT5ForConditionalGeneration, TFLongT5Model


class TFLongT5ModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        encoder_seq_length=7,
        decoder_seq_length=9,
        n_positions=14,
        local_radius=5,
        encoder_attention_type="local",
        global_block_size=3,
        # For common tests
        is_training=True,
        use_input_mask=True,
        use_attention_mask=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        d_ff=37,
        relative_attention_num_buckets=8,
        dropout_rate=0.1,
        initializer_factor=0.002,
        eos_token_id=1,
        pad_token_id=0,
        decoder_start_token_id=0,
        scope=None,
        decoder_layers=None,
        large_model_config_path="google/long-t5-local-large",
    ):

        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        self.n_positions = n_positions
        self.local_radius = local_radius
        self.block_len = local_radius + 1
        self.encoder_attention_type = encoder_attention_type
        self.global_block_size = global_block_size
        # For common tests
        self.seq_length = self.decoder_seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.initializer_factor = initializer_factor
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.scope = None
        self.decoder_layers = decoder_layers
        self.large_model_config_path = large_model_config_path

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_labels = None
        if self.use_labels:
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = LongT5Config(
            vocab_size=self.vocab_size,
            n_positions=self.n_positions,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.pad_token_id,
            local_radius=self.local_radius,
            encoder_attention_type=self.encoder_attention_type,
        )

        return (config, input_ids, input_mask, token_labels)

    def create_and_check_longt5_model(self, config, input_ids, input_mask, token_labels):
        model = TFLongT5Model(config=config)
        inputs = {
            "input_ids": input_ids,
            "decoder_input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        result = model(inputs)

        result = model(input_ids, decoder_attention_mask=input_mask, decoder_input_ids=input_ids)
        decoder_output = result.last_hidden_state
        decoder_past = result.past_key_values
        encoder_output = result.encoder_last_hidden_state
        self.parent.assertListEqual(list(encoder_output.shape), [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertListEqual(list(decoder_output.shape), [self.batch_size, self.seq_length, self.hidden_size])
        # There should be `num_layers` key value embeddings stored in decoder_past[1]
        self.parent.assertEqual(len(decoder_past), config.num_layers)
        # There should be a self attn key, a self attn value, a cross attn key and a cross attn value stored in each decoder_past[1] tuple
        self.parent.assertEqual(len(decoder_past[0]), 4)

    def create_and_check_longt5_with_lm_head(self, config, input_ids, input_mask, token_labels):
        model = TFLongT5ForConditionalGeneration(config=config)
        inputs_dict = {
            "input_ids": input_ids,
            "decoder_input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }

        result = model(inputs_dict)

        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_longt5_decoder_model_past(self, config, input_ids, decoder_input_ids, attention_mask):
        model = TFLongT5Model(config=config).get_decoder()

        input_ids = input_ids[:1, :]
        self.batch_size = 1

        # first forward pass
        outputs = model(input_ids, use_cache=True)

        outputs_use_cache_conf = model(input_ids)
        outputs_no_past = model(input_ids, use_cache=False)

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # append to next input_ids and
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)

        output_from_no_past = model(next_input_ids)[0]
        output_from_past = model(next_tokens, past_key_values=outputs.past_key_values)[0]

        # select random slice
        random_slice_idx = int(ids_tensor((1,), output_from_past.shape[-1]))
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx]
        output_from_past_slice = output_from_past[:, 0, random_slice_idx]

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-3)

    def create_and_check_longt5_decoder_model_attention_mask_past(
        self, config, input_ids, decoder_input_ids, attention_mask
    ):
        model = TFLongT5Model(config=config).get_decoder()

        # create attention mask
        half_seq_length = self.seq_length // 2
        attn_mask_begin = tf.ones((self.batch_size, half_seq_length), dtype=tf.int32)
        attn_mask_end = tf.zeros((self.batch_size, self.seq_length - half_seq_length), dtype=tf.int32)
        attn_mask = tf.concat([attn_mask_begin, attn_mask_end], axis=1)

        # first forward pass
        outputs = model(input_ids, attention_mask=attn_mask, use_cache=True)

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # change a random masked slice from input_ids
        random_seq_idx_to_change = ids_tensor((1,), half_seq_length).numpy() + 1
        random_other_next_tokens = ids_tensor((self.batch_size, self.seq_length), config.vocab_size)
        vector_condition = tf.range(self.seq_length) == (self.seq_length - random_seq_idx_to_change)
        condition = tf.transpose(
            tf.broadcast_to(tf.expand_dims(vector_condition, -1), (self.seq_length, self.batch_size))
        )
        input_ids = tf.where(condition, random_other_next_tokens, input_ids)

        # append to next input_ids and attn_mask
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        attn_mask = tf.concat(
            [attn_mask, tf.ones((attn_mask.shape[0], 1), dtype=tf.int32)],
            axis=1,
        )

        # get two different outputs
        output_from_no_past = model(next_input_ids, attention_mask=attn_mask)[0]
        output_from_past = model(next_tokens, past_key_values=outputs.past_key_values, attention_mask=attn_mask)[0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).numpy().item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx]
        output_from_past_slice = output_from_past[:, 0, random_slice_idx]

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-3)

    def create_and_check_longt5_decoder_model_past_large_inputs(
        self, config, input_ids, decoder_input_ids, attention_mask
    ):
        model = TFLongT5Model(config=config).get_decoder()

        input_ids = input_ids[:1, :]
        attention_mask = attention_mask[:1, :]
        self.batch_size = 1

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)

        # append to next input_ids and
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = tf.concat([attention_mask, next_attn_mask], axis=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)[0]
        output_from_past = model(
            next_tokens, attention_mask=next_attention_mask, past_key_values=outputs.past_key_values
        )[0]

        self.parent.assertEqual(next_tokens.shape[1], output_from_past.shape[1])

        # select random slice
        random_slice_idx = int(ids_tensor((1,), output_from_past.shape[-1]))
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx]
        output_from_past_slice = output_from_past[:, :, random_slice_idx]

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-3)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, input_mask, token_labels) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "decoder_input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return config, inputs_dict


@require_tf
class TFLongT5ModelTest(TFModelTesterMixin, unittest.TestCase):

    is_encoder_decoder = True
    all_model_classes = (TFLongT5Model, TFLongT5ForConditionalGeneration) if is_tf_available() else ()
    all_generative_model_classes = (TFLongT5ForConditionalGeneration,) if is_tf_available() else ()
    test_onnx = False

    def setUp(self):
        self.model_tester = TFLongT5ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LongT5Config, d_model=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_longt5_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_longt5_model(*config_and_inputs)

    def test_with_lm_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_longt5_with_lm_head(*config_and_inputs)

    def test_longt5_decoder_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_longt5_decoder_model_past(*config_and_inputs)

    def test_longt5_decoder_model_past_with_attn_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_longt5_decoder_model_attention_mask_past(*config_and_inputs)

    def test_longt5_decoder_model_past_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()

        # `create_and_check_t5_decoder_model_past_large_inputs` has special inputs:
        #     (config, input_ids, decoder_input_ids, attention_mask)
        # and we have to prepare it correctly here.
        config, input_ids, input_mask, token_labels = config_and_inputs
        config_and_inputs = (config, input_ids, None, input_mask)

        self.model_tester.create_and_check_longt5_decoder_model_past_large_inputs(*config_and_inputs)

    def test_attention_outputs(self):
        if not self.has_attentions:
            return

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", self.model_tester.seq_length)
        decoder_key_length = getattr(self.model_tester, "key_length", decoder_seq_length)
        block_len = getattr(self.model_tester, "block_len", None)

        def check_decoder_attentions_output(outputs):
            out_len = len(outputs)
            self.assertEqual(min(out_len % 2, out_len % 5), 0)  # differentiation due to newly added cross_attentions
            decoder_attentions = outputs.decoder_attentions
            self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(decoder_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
            )

        def check_encoder_attentions_output(outputs):
            attentions = [
                t.numpy() for t in (outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions)
            ]
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, block_len, 3 * block_len],
            )

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            config.output_hidden_states = False
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            out_len = len(outputs)
            self.assertEqual(config.output_hidden_states, False)
            check_encoder_attentions_output(outputs)

            if self.is_encoder_decoder:
                model = model_class(config)
                outputs = model(self._prepare_for_class(inputs_dict, model_class))
                self.assertEqual(config.output_hidden_states, False)
                check_decoder_attentions_output(outputs)

            # Check that output attentions can also be changed via the config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(config.output_hidden_states, False)
            check_encoder_attentions_output(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            config.output_hidden_states = True
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))

            self.assertEqual(out_len + (2 if self.is_encoder_decoder else 1), len(outputs))
            self.assertEqual(model.config.output_hidden_states, True)
            check_encoder_attentions_output(outputs)
    
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

    @tooslow
    def test_saved_model_creation(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = TFLongT5Model.from_pretrained("google/long-t5-local-base", from_pt=True)
        self.assertIsNotNone(model)

    def test_generate_with_headmasking(self):
        attention_names = ["encoder_attentions", "decoder_attentions", "cross_attentions"]
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        max_length = inputs_dict["input_ids"].shape[-1] + 3
        model = TFLongT5ForConditionalGeneration(config)

        head_masking = {
            "head_mask": tf.zeros((config.num_layers, config.num_heads)),
            "decoder_head_mask": tf.zeros((config.num_decoder_layers, config.num_heads)),
            "cross_attn_head_mask": tf.zeros((config.num_decoder_layers, config.num_heads)),
        }

        for attn_name, (name, mask) in zip(attention_names, head_masking.items()):
            head_masks = {name: mask}
            # Explicitly pass decoder_head_mask as it is required from LONGT5 model when head_mask specified
            if name == "head_mask":
                head_masks["decoder_head_mask"] = tf.ones((config.num_decoder_layers, config.num_heads))

            out = model.generate(
                inputs_dict["input_ids"],
                num_beams=1,
                max_length=max_length,
                output_attentions=True,
                return_dict_in_generate=True,
                **head_masks
            )
            # We check the state of decoder_attentions just from the last step
            # TF generate does not return `cross_attentions`
            if attn_name != "cross_attentions":
                attn_weights = out[attn_name] if attn_name == attention_names[0] else out[attn_name][-1]
                self.assertEqual(sum([tf.reduce_sum(w).numpy() for w in attn_weights]), 0.0)

    @slow
    def test_resize_embeddings(self):
        model = TFLongT5ForConditionalGeneration.from_pretrained("google/long-t5-local-base", from_pt=True)
        original_vocab_size = model.get_input_embeddings().weight.shape[0]
        # the vocab size is defined in the model config
        self.assertEqual(original_vocab_size, model.config.vocab_size)

        tokenizer = AutoTokenizer.from_pretrained("google/long-t5-local-base")
        tokenizer.add_special_tokens({"bos_token": "", "eos_token": ""})
        model._resize_token_embeddings(len(tokenizer))
        # the vocab size is now resized to the length of the tokenizer, which is different from the original size
        self.assertEqual(model.get_input_embeddings().weight.shape[0], len(tokenizer))
        self.assertNotEqual(model.get_input_embeddings().weight.shape[0], original_vocab_size)

    # This test is run in `TFLongT5EncoderOnlyModelTest`, where the main layer has the same inputs as the model
    @unittest.skip(reason="The inputs of the Main Layer are different.")
    def test_keras_save_load(self):
        pass
