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
""" Testing suite for the PyTorch Wav2Vec2 model. """


import math
import unittest

from tests.test_modeling_common import floats_tensor, random_attention_mask
from transformers import is_torch_available
from transformers.testing_utils import require_datasets, require_soundfile, require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, _config_zero_init


if is_torch_available():
    import torch

    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2ForMaskedLM, Wav2Vec2Model, Wav2Vec2Tokenizer


class Wav2Vec2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=1024,  # speech is longer
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
        input_values = floats_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

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

        return config, input_values, attention_mask

    def create_and_check_model(self, config, input_values, attention_mask):
        model = Wav2Vec2Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size)
        )

    def create_and_check_batch_inference(self, config, input_values, *args):
        # Not sure how to make this test pass at the moment. Batched input yields
        # same results as official fairseq implementation, but gives different results
        # depending on whether batched input is used or not
        # check: https://github.com/pytorch/fairseq/issues/3227
        model = Wav2Vec2Model(config=config)
        model.to(torch_device)
        model.eval()

        input_values = input_values[:3]
        attention_mask = torch.ones(input_values.shape, device=torch_device, dtype=torch.bool)

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0.0

        batch_outputs = model(input_values, attention_mask=attention_mask).last_hidden_state

        for i in range(input_values.shape[0]):
            input_slice = input_values[i : i + 1, : input_lengths[i]]
            output = model(input_slice).last_hidden_state

            batch_output = batch_outputs[i : i + 1, : output.shape[1]]
            self.parent.assertTrue(torch.allclose(output, batch_output, atol=1e-3))

    def prepare_config_and_inputs_for_common(self):
        config, input_values, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_values": input_values, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class Wav2Vec2ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            Wav2Vec2Model,
            Wav2Vec2ForMaskedLM,
        )
        if is_torch_available()
        else ()
    )
    test_pruning = False
    test_headmasking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = Wav2Vec2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Wav2Vec2Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    # Wav2Vec2 has no inputs_embeds
    def test_inputs_embeds(self):
        pass

    # `input_ids` is renamed to `input_values`
    def test_forward_signature(self):
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

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if "conv.weight" in name:
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg="Parameter {} of model {} seems not properly initialized".format(name, model_class),
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg="Parameter {} of model {} seems not properly initialized".format(name, model_class),
                        )

    @slow
    def test_model_from_pretrained(self):
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.assertIsNotNone(model)


@require_torch
class Wav2Vec2RobustModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Wav2Vec2Model, Wav2Vec2ForMaskedLM, Wav2Vec2ForCTC) if is_torch_available() else ()
    test_pruning = False
    test_headmasking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = Wav2Vec2ModelTester(
            self, conv_stride=(3, 3, 3), feat_extract_norm="layer", do_stable_layer_norm=True
        )
        self.config_tester = ConfigTester(self, config_class=Wav2Vec2Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_batched_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_batch_inference(*config_and_inputs)

    # Wav2Vec2 has no inputs_embeds
    def test_inputs_embeds(self):
        pass

    # `input_ids` is renamed to `input_values`
    def test_forward_signature(self):
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

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if "conv.weight" in name:
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg="Parameter {} of model {} seems not properly initialized".format(name, model_class),
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg="Parameter {} of model {} seems not properly initialized".format(name, model_class),
                        )

    @slow
    def test_model_from_pretrained(self):
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.assertIsNotNone(model)


@require_torch
@slow
@require_datasets
@require_soundfile
class Wav2Vec2ModelIntegrationTest(unittest.TestCase):
    def _load_datasamples(self, num_samples):
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

    def test_inference_ctc_normal(self):
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        model.to(torch_device)
        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h", do_lower_case=True)

        input_speech = self._load_datasamples(1)

        input_values = tokenizer(input_speech, return_tensors="pt").input_values.to(torch_device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = tokenizer.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = ["a man said to the universe sir i exist"]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_ctc_normal_batched(self):
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        model.to(torch_device)
        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h", do_lower_case=True)

        input_speech = self._load_datasamples(2)

        inputs = tokenizer(input_speech, return_tensors="pt", padding=True, truncation=True)

        input_values = inputs.input_values.to(torch_device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = tokenizer.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe sir i exist",
            "sweat covered brion's body trickling into the tight lowing cloth that was the only garment he wore",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_ctc_robust_batched(self):
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(torch_device)
        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", do_lower_case=True)

        input_speech = self._load_datasamples(4)

        inputs = tokenizer(input_speech, return_tensors="pt", padding=True, truncation=True)

        input_values = inputs.input_values.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)

        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = tokenizer.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe sir i exist",
            "sweat covered brion's body trickling into the tight loin cloth that was the only garment he wore",
            "the cut on his chest still dripping blood the ache of his overstrained eyes even the soaring arena around him with the thousands of spectators were trivialities not worth thinking about",
            "his instant panic was followed by a small sharp blow high on his chest",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)
