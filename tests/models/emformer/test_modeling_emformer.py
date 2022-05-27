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
""" Testing suite for the PyTorch Emformer model. """

import unittest

from datasets import load_dataset

from transformers import EmformerConfig, is_torch_available
from transformers.file_utils import is_torchaudio_available
from transformers.testing_utils import is_pt_flax_cross_test, require_soundfile, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


if is_torch_available():
    import torch

    from transformers import EmformerForRNNT, EmformerModel

if is_torchaudio_available():
    from transformers import EmformerProcessor


class EmformerModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=32,  # Mel spectrogram length
        feature_size=80,
        is_training=False,
        time_reduction_input_dim=4,
        time_reduction_stride=4,
        right_context_length=4,
        segment_length=16,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        num_hidden_layers=4,
        num_attention_heads=2,
        hidden_dropout_prob=0.1,
        intermediate_size=20,
        output_dim=16,
        layer_norm_eps=1e-5,
        hidden_act="gelu",
        initializer_range=0.02,
        vocab_size=32,
        blank_token_id=31,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.is_training = is_training
        self.time_reduction_input_dim = time_reduction_input_dim
        self.time_reduction_stride = time_reduction_stride
        self.right_context_length = right_context_length
        self.segment_length = segment_length
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.intermediate_size = intermediate_size
        self.output_dim = output_dim
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.blank_token_id = blank_token_id
        self.scope = scope

        self.hidden_size = time_reduction_input_dim * time_reduction_stride
        right_context_length = right_context_length // time_reduction_stride
        self.encoder_seq_length = seq_length // time_reduction_stride - right_context_length
        self.output_seq_length = self.encoder_seq_length

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.seq_length, self.feature_size], self.vocab_size)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        return config, input_features, attention_mask

    def get_config(self):
        return EmformerConfig(
            hidden_size=self.hidden_size,
            time_reduction_input_dim=self.time_reduction_input_dim,
            time_reduction_stride=self.time_reduction_stride,
            right_context_length=self.right_context_length,
            segment_length=self.segment_length,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_dropout_prob=self.hidden_dropout_prob,
            intermediate_size=self.intermediate_size,
            output_dim=self.output_dim,
            layer_norm_eps=self.layer_norm_eps,
            hidden_act=self.hidden_act,
            initializer_range=self.initializer_range,
            vocab_size=self.vocab_size,
            blank_token_id=self.blank_token_id,
        )

    def create_and_check_model(self, config, input_features, attention_mask):
        model = EmformerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_features, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.output_dim)
        )

    def create_and_check_batch_inference(self, config, input_features, *args):
        # test does not pass for models making use of `group_norm`
        # check: https://github.com/pytorch/fairseq/issues/3227
        model = EmformerModel(config=config)
        model.to(torch_device)
        model.eval()

        input_features = input_features[:3]
        attention_mask = torch.ones(input_features.shape, device=torch_device, dtype=torch.bool)

        input_lengths = [input_features.shape[-1] // i for i in [4, 2, 1]]

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

    def check_rnnt_loss(self, config, input_features, *args):
        model = EmformerForRNNT(config=config)
        model.to(torch_device)

        # make sure that dropout is disabled
        model.eval()

        input_features = input_features[:3]
        attention_mask = torch.ones(input_features.shape, device=torch_device, dtype=torch.long)

        input_lengths = [input_features.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_time_reduced_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_features.shape[0], min(max_length_labels) - 1), model.config.vocab_size)

        # pad input
        for i in range(len(input_lengths)):
            input_features[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0

        loss = model(input_features, attention_mask=attention_mask, labels=labels).loss.item()

        self.parent.assertTrue(isinstance(loss, float))

    def check_labels_out_of_vocab(self, config, input_features, *args):
        model = EmformerForRNNT(config)
        model.to(torch_device)
        model.train()

        input_features = input_features[:3]

        input_lengths = [input_features.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_time_reduced_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_features.shape[0], max(max_length_labels) - 2), model.config.vocab_size + 100)

        with self.parent.assertRaises(ValueError):
            model(input_features, labels=labels)

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_features": input_features, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class EmformerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (EmformerModel, EmformerForRNNT) if is_torch_available() else ()
    test_pruning = False
    test_headmasking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = EmformerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EmformerConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_labels_out_of_vocab(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_labels_out_of_vocab(*config_and_inputs)

    # Emformer has no inputs_embeds
    def test_inputs_embeds(self):
        pass

    # `input_ids` is renamed to `input_features`
    def test_forward_signature(self):
        pass

    # Emformer cannot resize token embeddings
    # since it has no tokens embeddings
    def test_resize_tokens_embeddings(self):
        pass

    # Emformer has no inputs_embeds
    # and thus the `get_input_embeddings` fn
    # is not implemented
    def test_model_common_attributes(self):
        pass

    @is_pt_flax_cross_test
    # not implemented in Flax
    def test_equivalence_flax_to_pt(self):
        pass

    @is_pt_flax_cross_test
    # not implemented in Flax
    def test_equivalence_pt_to_flax(self):
        pass

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[1]
        model = model_class(config)
        model.to(torch_device)

        input_features = inputs_dict["input_features"]

        input_lengths = torch.tensor(
            [input_features.shape[1] for _ in range(input_features.shape[0])], dtype=torch.long, device=torch_device
        )
        output_lengths = model._get_time_reduced_output_lengths(input_lengths)

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

        with torch.autograd.set_detect_anomaly(True):
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
                    "predictor.embedding.weight",
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

    @unittest.skip(reason="Gradient checkpointing is not implemented")
    def test_gradient_checkpointing_backward_compatibility(self):
        pass

    @unittest.skip(reason="Gradient checkpointing is not implemented")
    def test_gradient_checkpointing_enable_disable(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = EmformerModel.from_pretrained("anton-l/emformer-base-librispeech")
        self.assertIsNotNone(model)


@require_torch
@require_soundfile
@slow
class EmformerModelIntegrationTest(unittest.TestCase):
    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        sample_ids = [f"1272-141231-000{i}" for i in range(num_samples)]
        speech_samples = ds.sort("id").filter(lambda x: x["id"] in sample_ids)
        speech_samples = speech_samples["audio"]

        return [x["array"] for x in speech_samples]

    def test_inference_rnnt(self):
        model = EmformerForRNNT.from_pretrained("anton-l/emformer-base-librispeech")
        model.to(torch_device)
        processor = EmformerProcessor.from_pretrained(
            "anton-l/emformer-base-librispeech",
            do_lower_case=True,
        )
        input_speech = self._load_datasamples(1)

        input_features = processor(input_speech, return_tensors="pt").input_features.to(torch_device)

        with torch.no_grad():
            logits = model(input_features).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = ["a man said to the universe sir i exist"]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_rnnt_batched(self):
        model = EmformerForRNNT.from_pretrained("anton-l/emformer-base-librispeech")
        model.to(torch_device)
        processor = EmformerProcessor.from_pretrained(
            "anton-l/emformer-base-librispeech",
            do_lower_case=True,
        )

        input_speech = self._load_datasamples(2)

        inputs = processor(input_speech, return_tensors="pt", padding=True)

        input_features = inputs.input_features.to(torch_device)

        with torch.no_grad():
            logits = model(input_features).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe sir i exist",
            "sweat covered brion's body trickling into the tait loingleness that was the only garment he wore",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)
