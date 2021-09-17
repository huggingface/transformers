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
import math
import unittest

import numpy as np
import pytest

from transformers import is_tf_available
from transformers.testing_utils import require_datasets, require_soundfile, require_tf, slow

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor


if is_tf_available():
    import tensorflow as tf

    from transformers import HubertConfig, TFHubertForCTC, TFHubertModel, Wav2Vec2Processor
    from transformers.models.hubert.modeling_tf_hubert import _compute_mask_indices


@require_tf
class TFHubertModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=1024,
        is_training=False,
        hidden_size=16,
        feat_extract_norm="group",
        feat_extract_dropout=0.0,
        feat_extract_activation="gelu",
        conv_dim=(32, 32, 32),
        conv_stride=(4, 4, 4),
        conv_kernel=(8, 8, 8),
        conv_bias=False,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        num_hidden_layers=4,
        num_attention_heads=2,
        hidden_dropout_prob=0.1,  # this is most likely not correctly set yet
        intermediate_size=20,
        layer_norm_eps=1e-5,
        hidden_act="gelu",
        initializer_range=0.02,
        vocab_size=32,
        do_stable_layer_norm=False,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.feat_extract_norm = feat_extract_norm
        self.feat_extract_dropout = feat_extract_dropout
        self.feat_extract_activation = feat_extract_activation
        self.conv_dim = conv_dim
        self.conv_stride = conv_stride
        self.conv_kernel = conv_kernel
        self.conv_bias = conv_bias
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.do_stable_layer_norm = do_stable_layer_norm
        self.scope = scope

        output_seq_length = self.seq_length
        for kernel, stride in zip(self.conv_kernel, self.conv_stride):
            output_seq_length = (output_seq_length - (kernel - 1)) / stride
        self.output_seq_length = int(math.ceil(output_seq_length))
        self.encoder_seq_length = self.output_seq_length

    def prepare_config_and_inputs(self):
        input_values = tf.cast(ids_tensor([self.batch_size, self.seq_length], 32768), tf.float32) / 32768.0
        attention_mask = tf.ones_like(input_values)

        config = HubertConfig(
            hidden_size=self.hidden_size,
            feat_extract_norm=self.feat_extract_norm,
            feat_extract_dropout=self.feat_extract_dropout,
            feat_extract_activation=self.feat_extract_activation,
            conv_dim=self.conv_dim,
            conv_stride=self.conv_stride,
            conv_kernel=self.conv_kernel,
            conv_bias=self.conv_bias,
            num_conv_pos_embeddings=self.num_conv_pos_embeddings,
            num_conv_pos_embedding_groups=self.num_conv_pos_embedding_groups,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_dropout_prob=self.hidden_dropout_prob,
            intermediate_size=self.intermediate_size,
            layer_norm_eps=self.layer_norm_eps,
            hidden_act=self.hidden_act,
            initializer_range=self.initializer_range,
            vocab_size=self.vocab_size,
            do_stable_layer_norm=self.do_stable_layer_norm,
        )

        return config, input_values, attention_mask

    def create_and_check_model(self, config, input_values, attention_mask):
        model = TFHubertModel(config)
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size)
        )

    def create_and_check_batch_inference(self, config, input_values, *args):
        # test does not pass for models making use of `group_norm`
        # check: https://github.com/pytorch/fairseq/issues/3227
        config.layerdrop = 0.0
        model = TFHubertModel(config)

        input_values = input_values[:3]
        attention_mask = tf.ones_like(input_values)

        input_lengths = tf.constant([input_values.shape[-1] // i for i in [4, 2, 1]])
        length_mask = tf.sequence_mask(input_lengths, dtype=tf.float32)

        # convert values that are over input_lengths to padding
        input_values = input_values * length_mask
        attention_mask = attention_mask * length_mask

        batch_outputs = model(input_values, attention_mask=attention_mask, training=False).last_hidden_state

        for i in range(input_values.shape[0]):
            input_slice = input_values[i : i + 1, : input_lengths[i]]
            output = model(input_slice, training=False).last_hidden_state

            batch_output = batch_outputs[i : i + 1, : output.shape[1]]
            self.parent.assertTrue(np.allclose(output, batch_output, atol=1e-3))

    def check_ctc_loss(self, config, input_values, *args):
        model = TFHubertForCTC(config)

        input_values = input_values[:3]
        attention_mask = tf.ones_like(input_values)

        input_lengths = tf.constant([input_values.shape[-1] // i for i in [4, 2, 1]])
        max_length_labels = model.hubert._get_feat_extract_output_lengths(input_lengths)
        labels = ids_tensor((input_values.shape[0], min(max_length_labels) - 1), model.config.vocab_size)

        length_mask = tf.sequence_mask(input_lengths, dtype=tf.float32)

        # convert values that are over input_lengths to padding
        input_values = input_values * length_mask
        attention_mask = attention_mask * length_mask

        model.config.ctc_loss_reduction = "sum"
        sum_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss

        model.config.ctc_loss_reduction = "mean"
        mean_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss

        self.parent.assertTrue(abs(labels.shape[0] * mean_loss - sum_loss) < 1e-2)

    def check_training(self, config, input_values, *args):
        model = TFHubertForCTC(config)

        # freeze feature encoder
        model.freeze_feature_extractor()

        input_values = input_values[:3]

        input_lengths = tf.constant([input_values.shape[-1] // i for i in [4, 2, 1]])
        max_length_labels = model.hubert._get_feat_extract_output_lengths(input_lengths)
        labels = ids_tensor((input_values.shape[0], max(max_length_labels) - 2), model.config.vocab_size)

        length_mask = tf.sequence_mask(input_lengths, dtype=tf.float32)

        input_values = input_values * length_mask

        pad_size = max(max_length_labels) - labels.shape[1]
        labels = tf.pad(labels, ((0, 0), (0, pad_size)), constant_values=-100)

        loss = model(input_values, labels=labels, training=True).loss

        self.parent.assertFalse(tf.math.is_inf(loss))

    def check_labels_out_of_vocab(self, config, input_values, *args):
        model = TFHubertForCTC(config)
        input_lengths = tf.constant([input_values.shape[-1] // i for i in [4, 2, 1]])
        max_length_labels = model.hubert._get_feat_extract_output_lengths(input_lengths)
        labels = ids_tensor((input_values.shape[0], min(max_length_labels) - 1), model.config.vocab_size + 100)
        with pytest.raises(ValueError):
            model(input_values, labels=labels)

    def prepare_config_and_inputs_for_common(self):
        config, input_values, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_values": input_values, "attention_mask": attention_mask}
        return config, inputs_dict


@require_tf
class TFHubertModelTest(TFModelTesterMixin, unittest.TestCase):

    all_model_classes = (TFHubertModel, TFHubertForCTC) if is_tf_available() else ()
    test_resize_embeddings = False
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFHubertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=HubertConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    # overwrite because input_values != input_ids
    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.call)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["input_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    # overwrite because input_values != input_ids
    def test_keyword_and_dict_args(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            inputs = self._prepare_for_class(inputs_dict, model_class)

            outputs_dict = model(inputs)

            inputs_keywords = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))
            input_values = inputs_keywords.pop("input_values", None)
            outputs_keywords = model(input_values, **inputs_keywords)
            output_dict = outputs_dict[0].numpy()
            output_keywords = outputs_keywords[0].numpy()

            self.assertLess(np.sum(np.abs(output_dict - output_keywords)), 1e-6)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_hidden_states_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_hidden_states_output(config, inputs_dict, model_class):
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )

            hidden_states = outputs.hidden_states
            self.assertEqual(config.output_attentions, False)
            self.assertEqual(len(hidden_states), expected_num_layers)
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [self.model_tester.output_seq_length, self.model_tester.hidden_size],
            )

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(config, inputs_dict, model_class)

            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            check_hidden_states_output(config, inputs_dict, model_class)

    def test_ctc_loss_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_loss(*config_and_inputs)

    def test_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_training(*config_and_inputs)

    def test_labels_out_of_vocab(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_labels_out_of_vocab(*config_and_inputs)

    # Hubert has no inputs_embeds
    def test_inputs_embeds(self):
        pass

    # Hubert cannot resize token embeddings
    # since it has no tokens embeddings
    def test_resize_tokens_embeddings(self):
        pass

    # Hubert has no inputs_embeds
    # and thus the `get_input_embeddings` fn
    # is not implemented
    def test_model_common_attributes(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = TFHubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.assertIsNotNone(model)


@require_tf
class TFHubertRobustModelTest(TFModelTesterMixin, unittest.TestCase):
    all_model_classes = (TFHubertModel, TFHubertForCTC) if is_tf_available() else ()
    test_resize_embeddings = False
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFHubertModelTester(
            self,
            conv_stride=(3, 3, 3),
            feat_extract_norm="layer",
            do_stable_layer_norm=True,
            scope="robust",
        )
        self.config_tester = ConfigTester(self, config_class=HubertConfig, hidden_size=37)

    # overwrite because input_values != input_ids
    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.call)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["input_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    # overwrite because input_values != input_ids
    def test_keyword_and_dict_args(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            inputs = self._prepare_for_class(inputs_dict, model_class)

            outputs_dict = model(inputs)

            inputs_keywords = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))
            input_values = inputs_keywords.pop("input_values", None)
            outputs_keywords = model(input_values, **inputs_keywords)
            output_dict = outputs_dict[0].numpy()
            output_keywords = outputs_keywords[0].numpy()

            self.assertLess(np.sum(np.abs(output_dict - output_keywords)), 1e-6)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_hidden_states_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_hidden_states_output(config, inputs_dict, model_class):
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )

            hidden_states = outputs.hidden_states
            self.assertEqual(config.output_attentions, False)
            self.assertEqual(len(hidden_states), expected_num_layers)
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [self.model_tester.output_seq_length, self.model_tester.hidden_size],
            )

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(config, inputs_dict, model_class)

            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            check_hidden_states_output(config, inputs_dict, model_class)

    def test_batched_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_batch_inference(*config_and_inputs)

    def test_ctc_loss_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_loss(*config_and_inputs)

    def test_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_training(*config_and_inputs)

    def test_labels_out_of_vocab(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_labels_out_of_vocab(*config_and_inputs)

    # Hubert has no inputs_embeds
    def test_inputs_embeds(self):
        pass

    # Hubert cannot resize token embeddings
    # since it has no tokens embeddings
    def test_resize_tokens_embeddings(self):
        pass

    # Hubert has no inputs_embeds
    # and thus the `get_input_embeddings` fn
    # is not implemented
    def test_model_common_attributes(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = TFHubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        self.assertIsNotNone(model)


@require_tf
class TFHubertUtilsTest(unittest.TestCase):
    def test_compute_mask_indices(self):
        batch_size = 4
        sequence_length = 60
        mask_prob = 0.5
        mask_length = 1

        mask = _compute_mask_indices((batch_size, sequence_length), mask_prob, mask_length)

        self.assertListEqual(
            tf.reduce_sum(mask, -1).numpy().tolist(), [mask_prob * sequence_length for _ in range(batch_size)]
        )

    def test_compute_mask_indices_overlap(self):
        batch_size = 4
        sequence_length = 80
        mask_prob = 0.5
        mask_length = 4

        mask = _compute_mask_indices((batch_size, sequence_length), mask_prob, mask_length)

        # because of overlap mask don't have to add up exactly to `mask_prob * sequence_length`, but have to be smaller or equal
        for batch_sum in tf.reduce_sum(mask, -1):
            self.assertTrue(int(batch_sum) <= mask_prob * sequence_length)


@require_tf
@slow
@require_datasets
@require_soundfile
class TFHubertModelIntegrationTest(unittest.TestCase):
    def _load_datasamples(self, num_samples):
        from datasets import load_dataset

        import soundfile as sf

        ids = [f"1272-141231-000{i}" for i in range(num_samples)]

        # map files to raw
        def map_to_array(batch):
            speech, _ = sf.read(batch["file"])
            batch["speech"] = speech
            return batch

        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

        ds = ds.filter(lambda x: x["id"] in ids).sort("id").map(map_to_array)

        return ds["speech"][:num_samples]

    def test_inference_ctc_normal(self):
        model = TFHubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
        processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft", do_lower_case=True)
        input_speech = self._load_datasamples(1)

        input_values = processor(input_speech, return_tensors="tf", sampling_rate=16000).input_values

        logits = model(input_values).logits

        predicted_ids = tf.argmax(logits, axis=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = ["a man said to the universe sir i exist"]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_ctc_normal_batched(self):
        model = TFHubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
        processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft", do_lower_case=True)

        input_speech = self._load_datasamples(2)

        input_values = processor(input_speech, return_tensors="tf", padding=True, sampling_rate=16000).input_values

        logits = model(input_values).logits

        predicted_ids = tf.argmax(logits, axis=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe sir i exist",
            "sweat covered brion's body trickling into the tight loin cloth that was the only garment he wore",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_ctc_robust_batched(self):
        model = TFHubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
        processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft", do_lower_case=True)

        input_speech = self._load_datasamples(4)

        inputs = processor(input_speech, return_tensors="tf", padding=True, sampling_rate=16000)

        input_values = inputs.input_values
        attention_mask = inputs.attention_mask

        logits = model(input_values, attention_mask=attention_mask).logits

        predicted_ids = tf.argmax(logits, axis=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe sir i exist",
            "sweat covered brion's body trickling into the tight loin cloth that was the only garment he wore",
            "the cut on his chest still dripping blood the ache of his overstrained eyes even the soaring arena around him with the thousands of spectators were trivialities not worth thinking about",
            "his instant of panic was followed by a small sharp blow high on his chest",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)
