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


import math
import unittest

import numpy as np

from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, is_tf_available, is_torch_available
from transformers.testing_utils import require_datasets, require_soundfile, require_tf, require_torch, slow

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor


if is_tf_available():
    import tensorflow as tf

    from transformers import TFWav2Vec2ForCTC, TFWav2Vec2Model, Wav2Vec2Processor

if is_torch_available():
    import torch

MODEL_PATH = "wrice/wav2vec2-base-960h"


@require_torch
@require_tf
class TFWav2Vec2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=1024,
        is_training=False,
        hidden_size=16,
        feat_extract_norm="layer",
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

        self.pt_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.tf_model = TFWav2Vec2ForCTC.from_pretrained(MODEL_PATH)

    def prepare_config_and_inputs(self):
        input_ids = tf.cast(ids_tensor([self.batch_size, self.seq_length], 32768), tf.float32) / 32768.0

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
        )

        inputs_dict = {"input_ids": input_ids}
        return config, inputs_dict

    def create_and_check_model(self, config, input_dict):
        model = TFWav2Vec2Model(config)
        result = model(input_dict)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size)
        )

    def check_pos_embedding(self, config, input_dict):

        tf_layer = self.tf_model.wav2vec2.encoder.pos_conv_embed
        pt_layer = self.pt_model.wav2vec2.encoder.pos_conv_embed

        input_ids = np.random.rand(1, 49, 768).astype(np.float32)

        tf_output = tf_layer(input_ids).numpy()
        pt_output = pt_layer(torch.from_numpy(input_ids)).detach().numpy()

        tf.testing.assert_near(tf_output, pt_output, atol=1e-3)

    def check_attention(self, config, input_dict):

        for i in range(config.num_hidden_layers):
            tf_layer = self.tf_model.wav2vec2.encoder.layer[i].attention
            pt_layer = self.pt_model.wav2vec2.encoder.layers[i].attention

            test = np.random.rand(1, 768, 768).astype(np.float32)

            tf_output = tf_layer(test)[0].numpy()
            pt_output = pt_layer(torch.from_numpy(test))[0].detach().numpy()

            tf.testing.assert_near(tf_output, pt_output, atol=1e-3)

    def check_feature_projector(self):
        tf_layer = self.tf_model.wav2vec2.feature_projection
        pt_layer = self.pt_model.wav2vec2.feature_projection

        test = np.random.rand(1, 292, 768).astype(np.float32)

        tf_input = test
        pt_input = torch.from_numpy(test)

        tf_output = tf_layer(tf_input).last_hidden_state.numpy()
        pt_output = pt_layer(pt_input).last_hidden_state.detach().numpy()

        tf.testing.assert_near(tf_output, pt_output, atol=1e-3)

    def check_feature_extractor(self):
        tf_layer = self.tf_model.wav2vec2.feature_projection
        pt_layer = self.pt_model.wav2vec2.feature_projection

        test = np.random.rand(1, 16000).astype(np.float32)

        tf_input = test
        pt_input = torch.from_numpy(test)

        tf_output = tf_layer(tf_input).last_hidden_state.numpy()
        pt_output = pt_layer(pt_input).last_hidden_state.detach().numpy()

        tf.testing.assert_near(tf_output, pt_output, atol=1e-3)

    def check_feed_forward(self):
        config = Wav2Vec2Config()
        for i in range(config.num_hidden_layers):
            tf_layer = self.tf_model.wav2vec2.encoder.layer[i].feed_forward
            pt_layer = self.pt_model.wav2vec2.encoder.layers[i].feed_forward

            test = np.random.rand(1, 292, 768).astype(np.float32)

            tf_input = test
            pt_input = torch.from_numpy(test)

            tf_output = tf_layer(tf_input)[0].numpy()
            pt_output = pt_layer(pt_input)[0].detach().numpy()

            tf.testing.assert_near(tf_output, pt_output, atol=1e-3)

    def check_ctc_head(self):
        tf_layer = self.tf_model.lm_head
        pt_layer = self.pt_model.lm_head

        test = np.random.rand(1, 292, 768).astype(np.float32)

        tf_input = test
        pt_input = torch.from_numpy(test)

        tf_output = tf_layer(tf_input).numpy()
        pt_output = pt_layer(pt_input).detach().numpy()

        tf.testing.assert_near(tf_output, pt_output, atol=1e-3)

    def check_encoder(self):
        tf_layer = self.tf_model.wav2vec2.encoder
        pt_layer = self.pt_model.wav2vec2.encoder

        test = np.random.rand(1, 292, 768).astype(np.float32)

        tf_input = test
        pt_input = torch.from_numpy(test)

        tf_output = tf_layer(tf_input).last_hidden_state.numpy()
        pt_output = pt_layer(pt_input).last_hidden_state.detach().numpy()

        tf.testing.assert_near(tf_output, pt_output, atol=1e-3)

    def create_and_check_batch_inference(self, config, input_values, *args):
        # test does not pass for models making use of `group_norm`
        # check: https://github.com/pytorch/fairseq/issues/3227
        pass

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict


@require_torch
@require_tf
class TFWav2Vec2ModelTest(TFModelTesterMixin, unittest.TestCase):

    all_model_classes = (TFWav2Vec2Model, TFWav2Vec2ForCTC) if is_tf_available() else ()
    test_resize_embeddings = False
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFWav2Vec2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Wav2Vec2Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_hidden_states_output(self):
        pass

    # Wav2Vec2 has no inputs_embeds
    def test_inputs_embeds(self):
        pass

    # Wav2Vec2 cannot resize token embeddings
    # since it has no tokens embeddings
    def test_resize_tokens_embeddings(self):
        pass

    # Wav2Vec2 has no inputs_embeds
    # and thus the `get_input_embeddings` fn
    # is not implemented
    def test_model_common_attributes(self):
        pass

    def test_pt_tf_model_equivalence(self):
        return super().test_pt_tf_model_equivalence()

    @slow
    def test_model_from_pretrained(self):
        model = TFWav2Vec2Model.from_pretrained(MODEL_PATH)
        self.assertIsNotNone(model)


@require_tf
@require_datasets
@require_soundfile
class TFWav2Vec2ModelIntegrationTest(unittest.TestCase):
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
        model = TFWav2Vec2ForCTC.from_pretrained(MODEL_PATH)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", do_lower_case=True)
        input_speech = self._load_datasamples(1)

        input_values = processor(input_speech, return_tensors="tf", sampling_rate=16000).input_values

        logits = model(input_values).logits

        predicted_ids = tf.argmax(logits, axis=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = ["a man said to the universe sir i exist"]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_ctc_normal_batched(self):
        model = TFWav2Vec2ForCTC.from_pretrained(MODEL_PATH)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", do_lower_case=True)

        input_speech = self._load_datasamples(2)

        input_values = processor(
            input_speech, return_tensors="tf", padding=True, truncation=True, sampling_rate=16000
        ).input_values

        logits = model(input_values).logits

        predicted_ids = tf.argmax(logits, axis=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe sir i exist",
            "sweat covered brion's body trickling into the tight lowing cloth that was the only garment he wore",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)
