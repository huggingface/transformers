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
"""Testing suite for the TensorFlow Speech2Text model."""

from __future__ import annotations

import inspect
import unittest

from transformers import Speech2TextConfig
from transformers.testing_utils import require_sentencepiece, require_tf, require_tokenizers, slow
from transformers.utils import cached_property, is_tf_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers import Speech2TextProcessor, TFSpeech2TextForConditionalGeneration, TFSpeech2TextModel


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
        attention_mask = tf.math.not_equal(input_features, 0)
    if decoder_attention_mask is None:
        decoder_attention_mask = tf.math.not_equal(decoder_input_ids, config.pad_token_id)
    if head_mask is None:
        head_mask = tf.ones((config.encoder_layers, config.encoder_attention_heads))
    if decoder_head_mask is None:
        decoder_head_mask = tf.ones((config.decoder_layers, config.decoder_attention_heads))
    if cross_attn_head_mask is None:
        cross_attn_head_mask = tf.ones((config.decoder_layers, config.decoder_attention_heads))
    return {
        "input_features": input_features,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": attention_mask,
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
        scale_embedding=False,
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
        self.scale_embedding = scale_embedding

    def prepare_config_and_inputs(self):
        input_features = floats_tensor(
            [self.batch_size, self.seq_length, self.input_feat_per_channel], self.vocab_size
        )
        attention_mask = tf.ones([self.batch_size, self.seq_length], dtype=tf.int64)
        decoder_input_ids = tf.math.maximum(ids_tensor([self.batch_size, self.seq_length], self.vocab_size), 2)

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
            scale_embedding=self.scale_embedding,
        )

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_subsampled_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """

        for _ in range(self.num_conv_layers):
            input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = TFSpeech2TextModel(config=config).get_decoder()
        input_ids = inputs_dict["decoder_input_ids"]
        attention_mask = inputs_dict["decoder_attention_mask"]

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

        _, past_key_values = outputs.to_tuple()

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = tf.math.maximum(ids_tensor((self.batch_size, 3), config.vocab_size), 2)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2, dtype=tf.int64)

        # append to next input_ids and
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = tf.concat([attention_mask, next_attn_mask], axis=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = int(ids_tensor((1,), output_from_past.shape[-1]))
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx]
        output_from_past_slice = output_from_past[:, :, random_slice_idx]

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, atol=1e-2)


@require_tf
class TFSpeech2TextModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (TFSpeech2TextModel, TFSpeech2TextForConditionalGeneration) if is_tf_available() else ()
    all_generative_model_classes = (TFSpeech2TextForConditionalGeneration,) if is_tf_available() else ()
    pipeline_model_mapping = {"feature-extraction": TFSpeech2TextModel} if is_tf_available() else {}
    is_encoder_decoder = True
    test_pruning = False
    test_missing_keys = False
    test_onnx = False

    input_name = "input_ids"

    def setUp(self):
        self.model_tester = TFSpeech2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Speech2TextConfig)
        self.maxDiff = 3000

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    # not implemented currently
    def test_inputs_embeds(self):
        pass

    # training is not supported yet
    def test_training(self):
        pass

    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    def test_generate_fp16(self):
        pass

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

            subsampled_seq_length = model._get_feat_extract_output_lengths(seq_length)

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

            subsampled_encoder_seq_length = model._get_feat_extract_output_lengths(encoder_seq_length)
            subsampled_encoder_key_length = model._get_feat_extract_output_lengths(encoder_key_length)

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

    def test_resize_token_embeddings(self):
        # Overwritten method from parent; see `test_resize_embeddings_untied`
        pass

    def test_resize_tokens_embeddings(self):
        # see `test_resize_embeddings_untied`
        pass

    def test_resize_embeddings_untied(self):
        # TODO: copy test from PT. Not working at the moment because the test relies on `model.resize_token_embeddings`,
        # whose TF implementation assumes the use of `TFWrappedEmbeddings`. But with a `TFWrappedEmbeddings` we can't
        # load the weights from PT (also, it induces TF1 behavior, so we might want to rework how
        # `model.resize_token_embeddings` operates).
        pass

    def test_generate_without_input_ids(self):
        pass

    @staticmethod
    def _get_encoder_outputs(
        model, input_ids, attention_mask, output_attentions=None, output_hidden_states=None, num_interleave=1
    ):
        encoder = model.get_encoder()
        encoder_outputs = encoder(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        encoder_outputs["last_hidden_state"] = tf.repeat(encoder_outputs.last_hidden_state, num_interleave, axis=0)

        input_ids = input_ids[:, :, 0]
        input_ids = tf.zeros_like(input_ids[:, :1], dtype=tf.int64) + model._get_decoder_start_token_id()
        attention_mask = None
        return encoder_outputs, input_ids, attention_mask

    def _check_outputs(self, output, input_ids, config, use_cache=False, num_return_sequences=1):
        batch_size, seq_length = input_ids.shape[:2]
        subsampled_seq_length = self.model_tester.get_subsampled_output_lengths(seq_length)
        num_sequences_in_output = batch_size * num_return_sequences
        gen_len = (
            output.sequences.shape[-1] - 1 if config.is_encoder_decoder else output.sequences.shape[-1] - seq_length
        )

        # scores
        self._check_scores(num_sequences_in_output, output.scores, length=gen_len, config=config)

        # Attentions
        # encoder
        self._check_encoder_attention_for_generate(
            output.encoder_attentions, batch_size, config, subsampled_seq_length
        )
        # decoder
        self._check_attentions_for_generate(
            num_sequences_in_output,
            output.decoder_attentions,
            min_length=1,
            max_length=output.sequences.shape[-1],
            config=config,
            use_cache=use_cache,
        )

        # Hidden States
        # encoder
        self._check_encoder_hidden_states_for_generate(
            output.encoder_hidden_states, batch_size, config, subsampled_seq_length
        )

        # decoder
        self._check_hidden_states_for_generate(
            num_sequences_in_output,
            output.decoder_hidden_states,
            min_length=1,
            max_length=output.sequences.shape[-1],
            config=config,
            use_cache=use_cache,
        )

    # overwritten from parent due to the inability to work when non-text inputs are not passed AND because the input is
    # `input_features`
    def test_lm_head_model_random_no_beam_search_generate(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_features = inputs_dict.get("input_features", None)

        # iterate over all generative models
        for model_class in self.all_generative_model_classes:
            model = model_class(config)

            if config.bos_token_id is None:
                # if bos token id is not defined model needs input_features
                with self.assertRaises(AssertionError):
                    model.generate(do_sample=True, max_length=5)
                # num_return_sequences = 1
                self._check_generated_ids(model.generate(input_features, do_sample=True))

            with self.assertRaises(ValueError):
                # generating multiple sequences when no beam search generation
                # is not allowed as it would always generate the same sequences
                model.generate(input_features, do_sample=False, num_return_sequences=2)

            # num_return_sequences > 1, sample
            self._check_generated_ids(model.generate(input_features, do_sample=True, num_return_sequences=2))

            # check bad words tokens language generation
            # create list of 1-seq bad token and list of 2-seq of bad tokens
            bad_words_ids = [self._generate_random_bad_tokens(1, model), self._generate_random_bad_tokens(2, model)]
            output_tokens = model.generate(
                input_features, do_sample=True, bad_words_ids=bad_words_ids, num_return_sequences=2
            )
            # only count generated tokens
            generated_ids = output_tokens[:, input_features.shape[-1] :]
            self.assertFalse(self._check_match_tokens(generated_ids.numpy().tolist(), bad_words_ids))

    # overwritten from parent due to the inability to work when non-text inputs are not passed AND because the input is
    # `input_features`
    def test_lm_head_model_random_beam_search_generate(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_features = inputs_dict.get("input_features", None)

        for model_class in self.all_generative_model_classes:
            model = model_class(config)

            if config.bos_token_id is None:
                # if bos token id is not defined model needs input_ids, num_return_sequences = 1
                self._check_generated_ids(model.generate(input_features, do_sample=True, num_beams=2))

            with self.assertRaises(ValueError):
                # generating more sequences than having beams leads is not possible
                model.generate(input_features, do_sample=False, num_return_sequences=3, num_beams=2)

            # num_return_sequences > 1, sample
            self._check_generated_ids(
                model.generate(
                    input_features,
                    do_sample=True,
                    num_beams=2,
                    num_return_sequences=2,
                )
            )
            # num_return_sequences > 1, greedy
            self._check_generated_ids(
                model.generate(input_features, do_sample=False, num_beams=2, num_return_sequences=2)
            )

            # check bad words tokens language generation
            # create list of 1-seq bad token and list of 2-seq of bad tokens
            bad_words_ids = [self._generate_random_bad_tokens(1, model), self._generate_random_bad_tokens(2, model)]
            output_tokens = model.generate(
                input_features, do_sample=False, bad_words_ids=bad_words_ids, num_beams=2, num_return_sequences=2
            )
            # only count generated tokens
            generated_ids = output_tokens[:, input_features.shape[-1] :]
            self.assertFalse(self._check_match_tokens(generated_ids.numpy().tolist(), bad_words_ids))

    # overwritten from parent -- the input is `input_features`, not `input_ids`
    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.call)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "input_features",
                "attention_mask",
                "decoder_input_ids",
                "decoder_attention_mask",
            ]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    def test_pt_tf_model_equivalence(self, allow_missing_keys=True):
        # Allow missing keys since TF doesn't cache the sinusoidal embeddings in an attribute
        super().test_pt_tf_model_equivalence(allow_missing_keys=allow_missing_keys)


@require_tf
@require_sentencepiece
@require_tokenizers
@slow
class TFSpeech2TextModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

    def _load_datasamples(self, num_samples):
        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def test_generation_librispeech(self):
        model = TFSpeech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
        processor = self.default_processor

        input_speech = self._load_datasamples(1)

        input_features = processor(input_speech, return_tensors="tf").input_features

        generated_ids = model.generate(input_features)
        generated_transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)

        EXPECTED_TRANSCRIPTIONS = [
            "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel"
        ]
        self.assertListEqual(generated_transcript, EXPECTED_TRANSCRIPTIONS)

    def test_generation_librispeech_batched(self):
        model = TFSpeech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
        processor = self.default_processor

        input_speech = self._load_datasamples(4)

        inputs = processor(input_speech, return_tensors="tf", padding=True)
        generated_ids = model.generate(inputs.input_features, attention_mask=inputs.attention_mask)
        generated_transcripts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        EXPECTED_TRANSCRIPTIONS = [
            "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel",
            "nor is mister cultar's manner less interesting than his matter",
            "he tells us that at this festive season of the year with christmas and roast beef looming before us"
            " similes drawn from eating and its results occur most readily to the mind",
            "he has grave doubts whether sir frederick leyton's work is really greek after all and can discover in it"
            " but little of rocky ithaca",
        ]
        self.assertListEqual(generated_transcripts, EXPECTED_TRANSCRIPTIONS)
