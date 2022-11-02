# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch MCTCT model. """

import inspect
import math
import unittest

from datasets import load_dataset

from transformers import MCTCTConfig, is_torch_available
from transformers.testing_utils import require_soundfile, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import MCTCTForCTC, MCTCTModel, MCTCTProcessor


class MCTCTModelTester:
    def __init__(
        self,
        parent,
        batch_size=10,
        seq_length=40,  # speech is longer
        is_training=False,
        vocab_size=32,
        hidden_size=128 * 4,
        num_hidden_layers=4,
        intermediate_size=20,
        num_attention_heads=4,
        attention_head_dim=128,
        max_position_embeddings=920,
        layer_norm_eps=1e-5,
        layerdrop=0.3,
        hidden_act="relu",
        initializer_range=0.02,
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.3,
        conv_glu_dim=1,
        conv_dropout=0.3,
        num_conv_layers=1,
        conv_kernel=(7,),
        conv_stride=(3,),
        input_feat_per_channel=80,
        input_channels=1,
        conv_channels=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length  # speech is longer
        self.is_training = is_training

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads

        self.attention_head_dim = attention_head_dim
        self.max_position_embeddings = max_position_embeddings

        self.layer_norm_eps = layer_norm_eps
        self.layerdrop = layerdrop
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        self.conv_glu_dim = conv_glu_dim
        self.conv_dropout = conv_dropout
        self.num_conv_layers = num_conv_layers
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.input_feat_per_channel = input_feat_per_channel
        self.input_channels = input_channels
        self.conv_channels = conv_channels

        output_seq_length = self.seq_length
        dilation = 1
        for _, kernel_sz, stride in zip(range(self.num_conv_layers), self.conv_kernel, self.conv_stride):
            padding = kernel_sz // 2
            output_seq_length = output_seq_length + 2 * padding - dilation * (kernel_sz - 1) - 1
            output_seq_length = torch.div(output_seq_length, stride, rounding_mode="trunc") + 1

        self.output_seq_length = int(math.ceil(output_seq_length))
        self.encoder_seq_length = self.output_seq_length

    def prepare_config_and_inputs(self):
        input_features = floats_tensor(
            [self.batch_size, self.seq_length, self.input_feat_per_channel], self.vocab_size
        )
        attention_mask = torch.ones([self.batch_size, self.seq_length], dtype=torch.long, device=torch_device)

        config = self.get_config()

        return config, input_features, attention_mask

    def get_config(self):
        return MCTCTConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            attention_head_dim=self.attention_head_dim,
            max_position_embeddings=self.max_position_embeddings,
            layer_norm_eps=self.layer_norm_eps,
            layerdrop=self.layerdrop,
            hidden_act=self.hidden_act,
            initializer_range=self.initializer_range,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            conv_glu_dim=self.conv_glu_dim,
            conv_dropout=self.conv_dropout,
            num_conv_layers=self.num_conv_layers,
            conv_kernel=self.conv_kernel,
            conv_stride=self.conv_stride,
            input_feat_per_channel=self.input_feat_per_channel,
            input_channels=self.input_channels,
            conv_channels=self.conv_channels,
        )

    def create_and_check_model(self, config, input_features, attention_mask):
        model = MCTCTModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_features, attention_mask=attention_mask)

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size)
        )

    def create_and_check_model_for_ctc(self, config, input_features, attention_mask):
        config.add_adapter = True
        config.output_hidden_size = 2 * config.hidden_size
        model = MCTCTForCTC(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_features, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.adapter_output_seq_length, self.vocab_size)
        )

    def create_and_check_batch_inference(self, config, input_features, *args):
        # test does not pass for models making use of `group_norm`
        # check: https://github.com/pytorch/fairseq/issues/3227
        model = MCTCTModel(config=config)
        model.to(torch_device)
        model.eval()

        input_features = input_features[:3]
        attention_mask = torch.ones(input_features.shape[:-1], device=torch_device, dtype=torch.bool)

        input_lengths = [input_features.shape[-1] // i for i in [2, 2, 1]]

        # pad input
        for i in range(len(input_lengths)):
            input_features[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0.0

        batch_outputs = model(input_features, attention_mask=attention_mask).last_hidden_state

        for i in range(input_features.shape[0]):
            input_slice = input_features[i : i + 1, : input_lengths[i]]
            output = model(input_slice).last_hidden_state

            batch_output = batch_outputs[i : i + 1, : output.shape[1]]
            self.parent.assertTrue(torch.allclose(output, batch_output, atol=1e-3))

    def check_ctc_loss(self, config, input_features, *args):
        model = MCTCTForCTC(config=config)
        model.to(torch_device)

        # make sure that dropout is disabled
        model.eval()

        input_features = input_features[:3]

        # input_features is a 2D window for each sequence
        attention_mask = torch.ones(input_features.shape[:-1], device=torch_device, dtype=torch.long)

        # -2 since input_features is a 2D window for each sequence in batch
        input_lengths = [input_features.shape[-2] // i for i in [2, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_features.shape[0], min(max_length_labels) - 1), model.config.vocab_size)
        # pad input
        for i in range(len(input_lengths)):
            input_features[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0

        model.config.ctc_loss_reduction = "sum"
        sum_loss = model(input_features, attention_mask=attention_mask, labels=labels).loss.item()

        model.config.ctc_loss_reduction = "mean"
        mean_loss = model(input_features, attention_mask=attention_mask, labels=labels).loss.item()

        self.parent.assertTrue(isinstance(sum_loss, float))
        self.parent.assertTrue(isinstance(mean_loss, float))

    def check_ctc_training(self, config, input_features, *args):
        config.ctc_zero_infinity = True
        model = MCTCTForCTC(config=config)
        model.to(torch_device)
        model.train()

        input_features = input_features[:3]

        input_lengths = [input_features.shape[-2] // i for i in [2, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_features.shape[0], max(max_length_labels) - 1), model.config.vocab_size)

        # pad input
        for i in range(len(input_lengths)):
            input_features[i, input_lengths[i] :] = 0.0

            if max_length_labels[i] < labels.shape[-1]:
                # it's important that we make sure that target lenghts are at least
                # one shorter than logit lenghts to prevent -inf
                labels[i, max_length_labels[i] - 1 :] = -100

        loss = model(input_features, labels=labels).loss
        self.parent.assertFalse(torch.isinf(loss).item())

        loss.backward()

    def check_labels_out_of_vocab(self, config, input_features, *args):
        model = MCTCTForCTC(config)
        model.to(torch_device)
        model.train()

        input_features = input_features[:3]

        input_lengths = [input_features.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_features.shape[0], max(max_length_labels) - 2), model.config.vocab_size + 100)

        with self.parent.assertRaises(ValueError):
            model(input_features, labels=labels)

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_features": input_features, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class MCTCTModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (MCTCTForCTC, MCTCTModel) if is_torch_available() else ()
    test_pruning = False
    test_headmasking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = MCTCTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MCTCTConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_ctc_loss_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_loss(*config_and_inputs)

    def test_ctc_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_training(*config_and_inputs)

    def test_labels_out_of_vocab(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_labels_out_of_vocab(*config_and_inputs)

    # MCTCT has no inputs_embeds
    def test_inputs_embeds(self):
        pass

    # `input_ids` is renamed to `input_features`
    def test_forward_signature(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "input_features",
                "attention_mask",
                "head_mask",
                "output_attentions",
                "output_hidden_states",
                "return_dict",
            ]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    # MCTCT cannot resize token embeddings
    # since it has no tokens embeddings
    def test_resize_tokens_embeddings(self):
        pass

    # MCTCT has no inputs_embeds
    def test_model_common_attributes(self):
        pass

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True
        config.layerdrop = 0.0

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        input_features = inputs_dict["input_features"]

        input_lengths = torch.tensor(
            [input_features.shape[1] for _ in range(input_features.shape[0])], dtype=torch.long, device=torch_device
        )
        output_lengths = model._get_feat_extract_output_lengths(input_lengths)

        labels = ids_tensor((input_features.shape[0], output_lengths[0] - 2), self.model_tester.vocab_size)
        inputs_dict["attention_mask"] = torch.ones_like(inputs_dict["attention_mask"])
        inputs_dict["labels"] = labels

        outputs = model(**inputs_dict)

        output = outputs[0]

        # Encoder-/Decoder-only models
        hidden_states = outputs.hidden_states[0]
        attentions = outputs.attentions[0]

        hidden_states.retain_grad()
        attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_states.grad)
        self.assertIsNotNone(attentions.grad)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = [
                    "conv.weight",
                    "masked_spec_embed",
                    "codevectors",
                    "quantizer.weight_proj.weight",
                    "project_hid.weight",
                    "project_hid.bias",
                    "project_q.weight",
                    "project_q.bias",
                    "feature_projection.projection.weight",
                    "feature_projection.projection.bias",
                    "objective.weight",
                ]
                if param.requires_grad:
                    if any([x in name for x in uniform_init_parms]):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    # overwrite from test_modeling_common
    def _mock_init_weights(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data.fill_(3)
        if hasattr(module, "weight_g") and module.weight_g is not None:
            module.weight_g.data.fill_(3)
        if hasattr(module, "weight_v") and module.weight_v is not None:
            module.weight_v.data.fill_(3)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.fill_(3)
        if hasattr(module, "codevectors") and module.codevectors is not None:
            module.codevectors.data.fill_(3)
        if hasattr(module, "masked_spec_embed") and module.masked_spec_embed is not None:
            module.masked_spec_embed.data.fill_(3)

    @slow
    def test_model_from_pretrained(self):
        model = MCTCTModel.from_pretrained("speechbrain/m-ctc-t-large")
        self.assertIsNotNone(model)


@require_torch
class MCTCTRobustModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (MCTCTForCTC, MCTCTModel) if is_torch_available() else ()
    test_pruning = False
    test_headmasking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = MCTCTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MCTCTConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_batched_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_batch_inference(*config_and_inputs)

    def test_ctc_loss_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_loss(*config_and_inputs)

    def test_ctc_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_training(*config_and_inputs)

    def test_labels_out_of_vocab(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_labels_out_of_vocab(*config_and_inputs)

    # MCTCT has no inputs_embeds
    def test_inputs_embeds(self):
        pass

    # `input_ids` is renamed to `input_features`
    def test_forward_signature(self):
        pass

    # MCTCT cannot resize token embeddings
    # since it has no tokens embeddings
    def test_resize_tokens_embeddings(self):
        pass

    # MCTCT has no inputs_embeds
    # and thus the `get_input_embeddings` fn
    # is not implemented
    def test_model_common_attributes(self):
        pass

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        # set layer drop to 0
        model.config.layerdrop = 0.0

        input_features = inputs_dict["input_features"]

        input_lengths = torch.tensor(
            [input_features.shape[1] for _ in range(input_features.shape[0])], dtype=torch.long, device=torch_device
        )
        output_lengths = model._get_feat_extract_output_lengths(input_lengths)

        labels = ids_tensor((input_features.shape[0], output_lengths[0] - 2), self.model_tester.vocab_size)
        inputs_dict["attention_mask"] = torch.ones_like(inputs_dict["attention_mask"])
        inputs_dict["labels"] = labels

        outputs = model(**inputs_dict)

        output = outputs[0]

        # Encoder-/Decoder-only models
        hidden_states = outputs.hidden_states[0]
        attentions = outputs.attentions[0]

        hidden_states.retain_grad()
        attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_states.grad)
        self.assertIsNotNone(attentions.grad)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = [
                    "conv.weight",
                    "masked_spec_embed",
                    "codevectors",
                    "quantizer.weight_proj.weight",
                    "project_hid.weight",
                    "project_hid.bias",
                    "project_q.weight",
                    "project_q.bias",
                    "feature_projection.projection.weight",
                    "feature_projection.projection.bias",
                    "objective.weight",
                ]
                if param.requires_grad:
                    if any([x in name for x in uniform_init_parms]):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    # overwrite from test_modeling_common
    def _mock_init_weights(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data.fill_(3)
        if hasattr(module, "weight_g") and module.weight_g is not None:
            module.weight_g.data.fill_(3)
        if hasattr(module, "weight_v") and module.weight_v is not None:
            module.weight_v.data.fill_(3)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.fill_(3)
        if hasattr(module, "codevectors") and module.codevectors is not None:
            module.codevectors.data.fill_(3)
        if hasattr(module, "masked_spec_embed") and module.masked_spec_embed is not None:
            module.masked_spec_embed.data.fill_(3)

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = MCTCTModel.from_pretrained("speechbrain/m-ctc-t-large")
        self.assertIsNotNone(model)


@require_torch
@require_soundfile
@slow
class MCTCTModelIntegrationTest(unittest.TestCase):
    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").filter(
            lambda x: x["id"] in [f"1272-141231-000{i}" for i in range(num_samples)]
        )[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def test_inference_ctc_normal(self):
        model = MCTCTForCTC.from_pretrained("speechbrain/m-ctc-t-large")
        model.to(torch_device)
        processor = MCTCTProcessor.from_pretrained("speechbrain/m-ctc-t-large", do_lower_case=True)
        input_speech = self._load_datasamples(1)

        input_features = processor(input_speech, return_tensors="pt").input_features.to(torch_device)

        with torch.no_grad():
            logits = model(input_features).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = ["a man said to the universe, sir, i exist."]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_ctc_normal_batched(self):
        model = MCTCTForCTC.from_pretrained("speechbrain/m-ctc-t-large")
        model.to(torch_device)
        processor = MCTCTProcessor.from_pretrained("speechbrain/m-ctc-t-large", do_lower_case=True)

        input_speech = self._load_datasamples(2)

        inputs = processor(input_speech, return_tensors="pt", padding=True)

        input_features = inputs.input_features.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)

        with torch.no_grad():
            logits = model(input_features, attention_mask=attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe, sir, i exist.",
            '"sweat-covered brion\'s body, trickling into the tight-lowing clossa was the only germent huor."',
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_ctc_robust_batched(self):
        model = MCTCTForCTC.from_pretrained("speechbrain/m-ctc-t-large").to(torch_device)
        processor = MCTCTProcessor.from_pretrained("speechbrain/m-ctc-t-large", do_lower_case=True)

        input_speech = self._load_datasamples(4)

        inputs = processor(input_speech, return_tensors="pt", padding=True, return_attention_mask=True)

        input_features = inputs.input_features.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)

        with torch.no_grad():
            logits = model(input_features, attention_mask=attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe, sir, i exist.",
            '"sweat-covered brion\'s body, trickling into the tight-lowing clossa was the only germent huor." "',
            "\"the cadona's chest still-dripping bloodthe acofis overstrained eyes, even the soring arena around him"
            " with thousands of spectators retrivialities not worth-thinking about.",
            "his instant panic was followed by a small sharp blow high on his chestr.",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)
