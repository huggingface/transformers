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


from __future__ import annotations

import copy
import gc
import glob
import inspect
import math
import multiprocessing
import os
import tempfile
import traceback
import unittest

import numpy as np
import pytest
from datasets import load_dataset
from huggingface_hub import snapshot_download

from transformers import Wav2Vec2Config, is_tf_available
from transformers.testing_utils import (
    CaptureLogger,
    is_flaky,
    is_pt_tf_cross_test,
    require_librosa,
    require_pyctcdecode,
    require_tf,
    run_test_in_subprocess,
    slow,
)
from transformers.utils import is_librosa_available, is_pyctcdecode_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers import (
        AutoFeatureExtractor,
        TFWav2Vec2ForCTC,
        TFWav2Vec2ForSequenceClassification,
        TFWav2Vec2Model,
        Wav2Vec2Processor,
    )
    from transformers.models.wav2vec2.modeling_tf_wav2vec2 import _compute_mask_indices


if is_pyctcdecode_available():
    import pyctcdecode.decoder

    from transformers import Wav2Vec2ProcessorWithLM
    from transformers.models.wav2vec2_with_lm import processing_wav2vec2_with_lm


if is_librosa_available():
    import librosa


def _test_wav2vec2_with_lm_invalid_pool(in_queue, out_queue, timeout):
    error = None
    try:
        _ = in_queue.get(timeout=timeout)

        downloaded_folder = snapshot_download("patrickvonplaten/common_voice_es_sample")
        file_path = glob.glob(downloaded_folder + "/*")[0]
        sample = librosa.load(file_path, sr=16_000)[0]

        model = TFWav2Vec2ForCTC.from_pretrained("patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm")
        processor = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm")

        input_values = processor(sample, return_tensors="tf").input_values

        logits = model(input_values).logits

        # use a spawn pool, which should trigger a warning if different than fork
        with CaptureLogger(pyctcdecode.decoder.logger) as cl, multiprocessing.get_context("spawn").Pool(1) as pool:
            transcription = processor.batch_decode(logits.numpy(), pool).text

        unittest.TestCase().assertIn("Falling back to sequential decoding.", cl.out)
        unittest.TestCase().assertEqual(transcription[0], "el libro ha sido escrito por cervantes")

        # force batch_decode to internally create a spawn pool, which should trigger a warning if different than fork
        multiprocessing.set_start_method("spawn", force=True)
        with CaptureLogger(processing_wav2vec2_with_lm.logger) as cl:
            transcription = processor.batch_decode(logits.numpy()).text

        unittest.TestCase().assertIn("Falling back to sequential decoding.", cl.out)
        unittest.TestCase().assertEqual(transcription[0], "el libro ha sido escrito por cervantes")
    except Exception:
        error = f"{traceback.format_exc()}"

    results = {"error": error}
    out_queue.put(results, timeout=timeout)
    out_queue.join()


@require_tf
class TFWav2Vec2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
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
        num_hidden_layers=2,
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

        config = Wav2Vec2Config(
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
        model = TFWav2Vec2Model(config)
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size)
        )

    def create_and_check_batch_inference(self, config, input_values, *args):
        # test does not pass for models making use of `group_norm`
        # check: https://github.com/pytorch/fairseq/issues/3227
        config.layerdrop = 0.0
        model = TFWav2Vec2Model(config)

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
        model = TFWav2Vec2ForCTC(config)

        input_values = input_values[:3]
        attention_mask = tf.ones_like(input_values)

        input_lengths = tf.constant([input_values.shape[-1] // i for i in [4, 2, 1]])
        max_length_labels = model.wav2vec2._get_feat_extract_output_lengths(input_lengths)
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

    def check_seq_classifier_loss(self, loss, config, input_values, *args):
        model = TFWav2Vec2ForSequenceClassification(config)

        input_values = input_values[:3]
        attention_mask = tf.ones(input_values.shape, dtype=tf.int32)

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        labels = tf.random.uniform((input_values.shape[0],), maxval=len(model.config.id2label), dtype=tf.int32)

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0
        training = False
        masked_loss = (
            model(input_values, attention_mask=attention_mask, labels=labels, training=training).loss.numpy().item()
        )
        unmasked_loss = model(input_values, labels=labels, training=training).loss.numpy().item()

        assert isinstance(masked_loss, float)
        assert isinstance(unmasked_loss, float)
        assert masked_loss != unmasked_loss

    def check_training(self, config, input_values, *args):
        model = TFWav2Vec2ForCTC(config)

        # freeze feature encoder
        model.freeze_feature_encoder()

        input_values = input_values[:3]

        input_lengths = tf.constant([input_values.shape[-1] // i for i in [4, 2, 1]])
        max_length_labels = model.wav2vec2._get_feat_extract_output_lengths(input_lengths)
        labels = ids_tensor((input_values.shape[0], max(max_length_labels) - 2), model.config.vocab_size)

        length_mask = tf.sequence_mask(input_lengths, dtype=tf.float32)

        input_values = input_values * length_mask

        pad_size = max(max_length_labels) - labels.shape[1]
        labels = tf.pad(labels, ((0, 0), (0, pad_size)), constant_values=-100)

        loss = model(input_values, labels=labels, training=True).loss

        self.parent.assertFalse(tf.math.is_inf(loss))

    def check_labels_out_of_vocab(self, config, input_values, *args):
        model = TFWav2Vec2ForCTC(config)
        input_lengths = tf.constant([input_values.shape[-1] // i for i in [4, 2, 1]])
        max_length_labels = model.wav2vec2._get_feat_extract_output_lengths(input_lengths)
        labels = ids_tensor((input_values.shape[0], min(max_length_labels) - 1), model.config.vocab_size + 100)
        with pytest.raises(ValueError):
            model(input_values, labels=labels)

    def prepare_config_and_inputs_for_common(self):
        config, input_values, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_values": input_values, "attention_mask": attention_mask}
        return config, inputs_dict


@require_tf
class TFWav2Vec2ModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (TFWav2Vec2Model, TFWav2Vec2ForCTC, TFWav2Vec2ForSequenceClassification) if is_tf_available() else ()
    )
    pipeline_model_mapping = (
        {"audio-classification": TFWav2Vec2ForSequenceClassification, "feature-extraction": TFWav2Vec2Model}
        if is_tf_available()
        else {}
    )
    test_resize_embeddings = False
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFWav2Vec2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Wav2Vec2Config, hidden_size=37)

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

    @is_flaky()
    def test_labels_out_of_vocab(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_labels_out_of_vocab(*config_and_inputs)

    def test_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_training(*config_and_inputs)

    @unittest.skip(reason="Wav2Vec2 has no input embeddings")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Wav2Vec2 has no tokens embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Wav2Vec2 has no input embeddings")
    def test_model_common_attributes(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.assertIsNotNone(model)

    @unittest.skip(reason="Fix me! Wav2Vec2 hits OOM errors when loss is computed on full batch")
    def test_dataset_conversion(self):
        # TODO: (Amy) - check whether skipping CTC model resolves this issue and possible resolutions for CTC
        pass

    @unittest.skip(reason="Fix me! Wav2Vec2 hits OOM errors when loss is computed on full batch")
    def test_keras_fit(self):
        # TODO: (Amy) - check whether skipping CTC model resolves this issue and possible resolutions for CTC
        pass

    @is_pt_tf_cross_test
    def test_pt_tf_model_equivalence(self, allow_missing_keys=False):
        # We override the base test here to skip loss calculation for Wav2Vec2 models because the loss is massive with
        # the default labels and frequently overflows to inf or exceeds numerical tolerances between TF/PT
        import torch

        import transformers

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            # Output all for aggressive testing
            config.output_hidden_states = True
            config.output_attentions = self.has_attentions

            # Make sure no sequence has all zeros as attention mask, otherwise some tests fail due to the inconsistency
            # of the usage `1e-4`, `1e-9`, `1e-30`, `-inf`.
            # TODO: Use a uniform value for all models, make sure all tests pass without this processing, and remove it.
            self._make_attention_mask_non_null(inputs_dict)

            pt_model_class_name = model_class.__name__[2:]  # Skip the "TF" at the beginning
            pt_model_class = getattr(transformers, pt_model_class_name)

            tf_model = model_class(config)
            pt_model = pt_model_class(config)

            tf_inputs_dict = self._prepare_for_class(inputs_dict, model_class)

            # Check we can load pt model in tf and vice-versa with model => model functions
            tf_model = transformers.load_pytorch_model_in_tf2_model(
                tf_model, pt_model, tf_inputs=tf_inputs_dict, allow_missing_keys=allow_missing_keys
            )
            pt_model = transformers.load_tf2_model_in_pytorch_model(
                pt_model, tf_model, allow_missing_keys=allow_missing_keys
            )

            # Original test: check without `labels`
            self.check_pt_tf_models(tf_model, pt_model, tf_inputs_dict)

            # Check we can load pt model in tf and vice-versa with checkpoint => model functions
            with tempfile.TemporaryDirectory() as tmpdirname:
                pt_checkpoint_path = os.path.join(tmpdirname, "pt_model.bin")
                torch.save(pt_model.state_dict(), pt_checkpoint_path)
                tf_model = transformers.load_pytorch_checkpoint_in_tf2_model(
                    tf_model, pt_checkpoint_path, allow_missing_keys=allow_missing_keys
                )

                tf_checkpoint_path = os.path.join(tmpdirname, "tf_model.h5")
                tf_model.save_weights(tf_checkpoint_path)
                pt_model = transformers.load_tf2_checkpoint_in_pytorch_model(
                    pt_model, tf_checkpoint_path, allow_missing_keys=allow_missing_keys
                )

            # Original test: check without `labels`
            self.check_pt_tf_models(tf_model, pt_model, tf_inputs_dict)


@require_tf
class TFWav2Vec2RobustModelTest(TFModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (TFWav2Vec2Model, TFWav2Vec2ForCTC, TFWav2Vec2ForSequenceClassification) if is_tf_available() else ()
    )
    test_resize_embeddings = False
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFWav2Vec2ModelTester(
            self,
            conv_stride=(3, 3, 3),
            feat_extract_norm="layer",
            do_stable_layer_norm=True,
            scope="robust",
        )
        self.config_tester = ConfigTester(self, config_class=Wav2Vec2Config, hidden_size=37)

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

    # TODO (Joao): fix me
    @unittest.skip("Broke with TF 2.10")
    def test_labels_out_of_vocab(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_labels_out_of_vocab(*config_and_inputs)

    def test_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_training(*config_and_inputs)

    @unittest.skip(reason="Wav2Vec2 has no input embeddings")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Wav2Vec2 has no tokens embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Wav2Vec2 has no input embeddings")
    def test_model_common_attributes(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.assertIsNotNone(model)

    @unittest.skip(reason="Fix me! Wav2Vec2 hits OOM errors when loss is computed on full batch")
    def test_dataset_conversion(self):
        # TODO: (Amy) - check whether skipping CTC model resolves this issue and possible resolutions for CTC
        pass

    @unittest.skip(reason="Fix me! Wav2Vec2 hits OOM errors when loss is computed on full batch")
    def test_keras_fit(self):
        # TODO: (Amy) - check whether skipping CTC model resolves this issue and possible resolutions for CTC
        pass

    @is_pt_tf_cross_test
    def test_pt_tf_model_equivalence(self, allow_missing_keys=False):
        # We override the base test here to skip loss calculation for Wav2Vec2 models because the loss is massive with
        # the default labels and frequently overflows to inf or exceeds numerical tolerances between TF/PT
        import torch

        import transformers

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            # Output all for aggressive testing
            config.output_hidden_states = True
            config.output_attentions = self.has_attentions

            # Make sure no sequence has all zeros as attention mask, otherwise some tests fail due to the inconsistency
            # of the usage `1e-4`, `1e-9`, `1e-30`, `-inf`.
            # TODO: Use a uniform value for all models, make sure all tests pass without this processing, and remove it.
            self._make_attention_mask_non_null(inputs_dict)

            pt_model_class_name = model_class.__name__[2:]  # Skip the "TF" at the beginning
            pt_model_class = getattr(transformers, pt_model_class_name)

            tf_model = model_class(config)
            pt_model = pt_model_class(config)

            tf_inputs_dict = self._prepare_for_class(inputs_dict, model_class)

            # Check we can load pt model in tf and vice-versa with model => model functions
            tf_model = transformers.load_pytorch_model_in_tf2_model(
                tf_model, pt_model, tf_inputs=tf_inputs_dict, allow_missing_keys=allow_missing_keys
            )
            pt_model = transformers.load_tf2_model_in_pytorch_model(
                pt_model, tf_model, allow_missing_keys=allow_missing_keys
            )

            # Original test: check without `labels`
            self.check_pt_tf_models(tf_model, pt_model, tf_inputs_dict)

            # Check we can load pt model in tf and vice-versa with checkpoint => model functions
            with tempfile.TemporaryDirectory() as tmpdirname:
                pt_checkpoint_path = os.path.join(tmpdirname, "pt_model.bin")
                torch.save(pt_model.state_dict(), pt_checkpoint_path)
                tf_model = transformers.load_pytorch_checkpoint_in_tf2_model(
                    tf_model, pt_checkpoint_path, allow_missing_keys=allow_missing_keys
                )

                tf_checkpoint_path = os.path.join(tmpdirname, "tf_model.h5")
                tf_model.save_weights(tf_checkpoint_path)
                pt_model = transformers.load_tf2_checkpoint_in_pytorch_model(
                    pt_model, tf_checkpoint_path, allow_missing_keys=allow_missing_keys
                )

            # Original test: check without `labels`
            self.check_pt_tf_models(tf_model, pt_model, tf_inputs_dict)


@require_tf
class TFWav2Vec2UtilsTest(unittest.TestCase):
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
class TFWav2Vec2ModelIntegrationTest(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        # clean-up as much as possible GPU memory occupied by PyTorch
        gc.collect()

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").filter(
            lambda x: x["id"] in [f"1272-141231-000{i}" for i in range(num_samples)]
        )[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def _load_superb(self, task, num_samples):
        ds = load_dataset("anton-l/superb_dummy", task, split="test")

        return ds[:num_samples]

    def test_inference_ctc_normal(self):
        model = TFWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", do_lower_case=True)
        input_speech = self._load_datasamples(1)

        input_values = processor(input_speech, return_tensors="tf", sampling_rate=16000).input_values

        logits = model(input_values).logits

        predicted_ids = tf.argmax(logits, axis=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = ["a man said to the universe sir i exist"]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_ctc_normal_batched(self):
        model = TFWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", do_lower_case=True)

        input_speech = self._load_datasamples(2)

        input_values = processor(input_speech, return_tensors="tf", padding=True, sampling_rate=16000).input_values

        logits = model(input_values).logits

        predicted_ids = tf.argmax(logits, axis=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe sir i exist",
            "sweat covered brion's body trickling into the tight lowing cloth that was the only garment he wore",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_ctc_robust_batched(self):
        model = TFWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", do_lower_case=True)

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
            "the cut on his chest still dripping blood the ache of his overstrained eyes even the soaring arena around"
            " him with the thousands of spectators were trivialities not worth thinking about",
            "his instant panic was followed by a small sharp blow high on his chest",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    @require_pyctcdecode
    @require_librosa
    def test_wav2vec2_with_lm(self):
        downloaded_folder = snapshot_download("patrickvonplaten/common_voice_es_sample")
        file_path = glob.glob(downloaded_folder + "/*")[0]
        sample = librosa.load(file_path, sr=16_000)[0]

        model = TFWav2Vec2ForCTC.from_pretrained("patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm")
        processor = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm")

        input_values = processor(sample, return_tensors="tf").input_values

        logits = model(input_values).logits

        transcription = processor.batch_decode(logits.numpy()).text

        self.assertEqual(transcription[0], "el libro ha sido escrito por cervantes")

    @require_pyctcdecode
    @require_librosa
    def test_wav2vec2_with_lm_pool(self):
        downloaded_folder = snapshot_download("patrickvonplaten/common_voice_es_sample")
        file_path = glob.glob(downloaded_folder + "/*")[0]
        sample = librosa.load(file_path, sr=16_000)[0]

        model = TFWav2Vec2ForCTC.from_pretrained("patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm")
        processor = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm")

        input_values = processor(sample, return_tensors="tf").input_values

        logits = model(input_values).logits

        # test user-managed pool
        with multiprocessing.get_context("fork").Pool(2) as pool:
            transcription = processor.batch_decode(logits.numpy(), pool).text

        self.assertEqual(transcription[0], "el libro ha sido escrito por cervantes")

        # user-managed pool + num_processes should trigger a warning
        with CaptureLogger(processing_wav2vec2_with_lm.logger) as cl, multiprocessing.get_context("fork").Pool(
            2
        ) as pool:
            transcription = processor.batch_decode(logits.numpy(), pool, num_processes=2).text

        self.assertIn("num_process", cl.out)
        self.assertIn("it will be ignored", cl.out)

        self.assertEqual(transcription[0], "el libro ha sido escrito por cervantes")

    @require_pyctcdecode
    @require_librosa
    def test_wav2vec2_with_lm_invalid_pool(self):
        run_test_in_subprocess(test_case=self, target_func=_test_wav2vec2_with_lm_invalid_pool, inputs=None)

    def test_inference_keyword_spotting(self):
        model = TFWav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-ks", from_pt=True)
        processor = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-ks")
        input_data = self._load_superb("ks", 4)
        inputs = processor(input_data["speech"], return_tensors="tf", padding=True)
        input_values = inputs.input_values
        attention_mask = inputs.attention_mask
        outputs = model(input_values, attention_mask)
        predicted_logits, predicted_ids = tf.math.reduce_max(outputs.logits, axis=-1), tf.argmax(
            outputs.logits, axis=-1
        )
        expected_labels = [7, 6, 10, 9]
        expected_logits = tf.convert_to_tensor([6.1186, 11.8961, 10.2931, 6.0898])
        self.assertListEqual(predicted_ids.numpy().tolist(), expected_labels)
        self.assertTrue(np.allclose(predicted_logits, expected_logits, atol=1e-2))

    def test_inference_intent_classification(self):
        model = TFWav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-ic", from_pt=True)
        processor = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-ic")
        input_data = self._load_superb("ic", 4)
        inputs = processor(input_data["speech"], return_tensors="tf", padding=True)
        input_values = inputs.input_values
        attention_mask = inputs.attention_mask
        outputs = model(input_values, attention_mask=attention_mask)
        predicted_logits_action, predicted_ids_action = tf.math.reduce_max(outputs.logits[:, :6], axis=-1), tf.argmax(
            outputs.logits[:, :6], axis=-1
        )
        predicted_logits_object, predicted_ids_object = tf.math.reduce_max(
            outputs.logits[:, 6:20], axis=-1
        ), tf.argmax(outputs.logits[:, 6:20], axis=-1)
        predicted_logits_location, predicted_ids_location = tf.math.reduce_max(
            outputs.logits[:, 20:24], axis=-1
        ), tf.argmax(outputs.logits[:, 20:24], axis=-1)
        expected_labels_action = [0, 0, 2, 3]
        expected_logits_action = tf.convert_to_tensor([0.4568, 11.0848, 1.6621, 9.3841])
        expected_labels_object = [3, 10, 3, 4]
        expected_logits_object = tf.convert_to_tensor([1.5322, 10.7094, 5.2469, 22.1318])
        expected_labels_location = [0, 0, 0, 1]
        expected_logits_location = tf.convert_to_tensor([1.5335, 6.5096, 10.5704, 11.0569])

        self.assertListEqual(predicted_ids_action.numpy().tolist(), expected_labels_action)
        self.assertListEqual(predicted_ids_object.numpy().tolist(), expected_labels_object)
        self.assertListEqual(predicted_ids_location.numpy().tolist(), expected_labels_location)

        self.assertTrue(np.allclose(predicted_logits_action, expected_logits_action, atol=1e-2))
        self.assertTrue(np.allclose(predicted_logits_object, expected_logits_object, atol=1e-2))
        self.assertTrue(np.allclose(predicted_logits_location, expected_logits_location, atol=1e-2))

    def test_inference_speaker_identification(self):
        model = TFWav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid", from_pt=True)
        processor = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")
        input_data = self._load_superb("si", 4)
        output_logits = []
        for example in input_data["speech"]:
            input = processor(example, return_tensors="tf", padding=True)
            output = model(input.input_values, attention_mask=None)
            output_logits.append(output.logits[0])
        output_logits = tf.stack(output_logits)
        predicted_logits, predicted_ids = tf.math.reduce_max(output_logits, axis=-1), tf.argmax(output_logits, axis=-1)
        expected_labels = [251, 1, 1, 3]
        expected_logits = tf.convert_to_tensor([37.5627, 71.6362, 64.2419, 31.7778])
        self.assertListEqual(predicted_ids.numpy().tolist(), expected_labels)
        self.assertTrue(np.allclose(predicted_logits, expected_logits, atol=1e-2))

    def test_inference_emotion_recognition(self):
        model = TFWav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er", from_pt=True)
        processor = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
        input_data = self._load_superb("er", 4)
        inputs = processor(input_data["speech"], return_tensors="tf", padding=True)

        input_values = inputs.input_values
        attention_mask = inputs.attention_mask
        outputs = model(input_values, attention_mask=attention_mask)
        predicted_logits, predicted_ids = tf.math.reduce_max(outputs.logits, axis=-1), tf.argmax(
            outputs.logits, axis=-1
        )

        expected_labels = [1, 1, 2, 2]
        # s3prl logits for the same batch
        expected_logits = tf.convert_to_tensor([2.1722, 3.0779, 8.0287, 6.6797])

        self.assertListEqual(predicted_ids.numpy().tolist(), expected_labels)
        self.assertTrue(np.allclose(predicted_logits, expected_logits, atol=1e-2))
