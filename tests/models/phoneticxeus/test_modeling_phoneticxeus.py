# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch PhoneticXeus model."""

import math
import tempfile
import unittest

from transformers import PhoneticXeusConfig, is_torch_available
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        PhoneticXeusForCTC,
        PhoneticXeusModel,
        Wav2Vec2FeatureExtractor,
    )


class PhoneticXeusModelTester:
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
        conv_bias=True,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        intermediate_size=20,
        layer_norm_eps=1e-5,
        hidden_act="swish",
        initializer_range=0.02,
        vocab_size=32,
        cgmlp_linear_units=20,
        cgmlp_conv_kernel=3,
        merge_conv_kernel=3,
        use_ffn=True,
        macaron_ffn=True,
        interctc_layer_idx=(1,),
        interctc_use_conditioning=True,
        normalize_audio=True,
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
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.cgmlp_linear_units = cgmlp_linear_units
        self.cgmlp_conv_kernel = cgmlp_conv_kernel
        self.merge_conv_kernel = merge_conv_kernel
        self.use_ffn = use_ffn
        self.macaron_ffn = macaron_ffn
        self.interctc_layer_idx = interctc_layer_idx
        self.interctc_use_conditioning = interctc_use_conditioning
        self.normalize_audio = normalize_audio
        self.scope = scope

        output_seq_length = self.seq_length
        for kernel, stride in zip(self.conv_kernel, self.conv_stride):
            output_seq_length = (output_seq_length - (kernel - 1)) / stride
        self.output_seq_length = int(math.ceil(output_seq_length))
        self.encoder_seq_length = self.output_seq_length

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        return config, input_values, attention_mask

    def get_config(self):
        return PhoneticXeusConfig(
            hidden_size=self.hidden_size,
            feat_extract_norm=self.feat_extract_norm,
            feat_extract_activation=self.feat_extract_activation,
            conv_dim=self.conv_dim,
            conv_stride=self.conv_stride,
            conv_kernel=self.conv_kernel,
            conv_bias=self.conv_bias,
            num_conv_pos_embeddings=self.num_conv_pos_embeddings,
            num_conv_pos_embedding_groups=self.num_conv_pos_embedding_groups,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_dropout=self.hidden_dropout,
            attention_dropout=self.attention_dropout,
            intermediate_size=self.intermediate_size,
            layer_norm_eps=self.layer_norm_eps,
            hidden_act=self.hidden_act,
            initializer_range=self.initializer_range,
            vocab_size=self.vocab_size,
            cgmlp_linear_units=self.cgmlp_linear_units,
            cgmlp_conv_kernel=self.cgmlp_conv_kernel,
            merge_conv_kernel=self.merge_conv_kernel,
            use_ffn=self.use_ffn,
            macaron_ffn=self.macaron_ffn,
            interctc_layer_idx=self.interctc_layer_idx,
            interctc_use_conditioning=self.interctc_use_conditioning,
            normalize_audio=self.normalize_audio,
        )

    def create_and_check_model(self, config, input_values, attention_mask):
        model = PhoneticXeusModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size)
        )

    def create_and_check_model_float16(self, config, input_values, attention_mask):
        model = PhoneticXeusModel(config=config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            model = PhoneticXeusModel.from_pretrained(tmpdirname, dtype=torch.float16)

        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            result = model(input_values.type(dtype=torch.float16), attention_mask=attention_mask)

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size)
        )

    def check_ctc_loss(self, config, input_values, *args):
        model = PhoneticXeusForCTC(config=config)
        model.to(torch_device)

        # make sure that dropout is disabled
        model.eval()

        input_values = input_values[:3]
        attention_mask = torch.ones(input_values.shape, device=torch_device, dtype=torch.long)

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_values.shape[0], min(max_length_labels) - 1), model.config.vocab_size)

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0

        model.config.ctc_loss_reduction = "sum"
        sum_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss.item()

        model.config.ctc_loss_reduction = "mean"
        mean_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss.item()

        self.parent.assertTrue(isinstance(sum_loss, float))
        self.parent.assertTrue(isinstance(mean_loss, float))

    def check_ctc_training(self, config, input_values, *args):
        config.ctc_zero_infinity = True
        model = PhoneticXeusForCTC(config=config)
        model.to(torch_device)
        model.train()

        # freeze feature encoder
        model.freeze_feature_encoder()

        input_values = input_values[:3]

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_values.shape[0], max(max_length_labels) - 2), model.config.vocab_size)

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0

            if max_length_labels[i] < labels.shape[-1]:
                # it's important that we make sure that target lengths are at least
                # one shorter than logit lengths to prevent -inf
                labels[i, max_length_labels[i] - 1 :] = -100

        loss = model(input_values, labels=labels).loss
        self.parent.assertFalse(torch.isinf(loss).item())

        loss.backward()

    def check_labels_out_of_vocab(self, config, input_values, *args):
        model = PhoneticXeusForCTC(config)
        model.to(torch_device)
        model.train()

        input_values = input_values[:3]

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_values.shape[0], max(max_length_labels) - 2), model.config.vocab_size + 100)

        with self.parent.assertRaises(ValueError):
            model(input_values, labels=labels)

    def prepare_config_and_inputs_for_common(self):
        config, input_values, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_values": input_values, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class PhoneticXeusModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            PhoneticXeusForCTC,
            PhoneticXeusModel,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "automatic-speech-recognition": PhoneticXeusForCTC,
            "feature-extraction": PhoneticXeusModel,
        }
        if is_torch_available()
        else {}
    )

    test_resize_embeddings = False

    def test_batching_equivalence(self, atol=1e-3, rtol=1e-3):
        super().test_batching_equivalence(atol=atol, rtol=rtol)

    def setUp(self):
        self.model_tester = PhoneticXeusModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PhoneticXeusConfig, hidden_size=32)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @require_torch_accelerator
    @require_torch_fp16
    def test_model_float16(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_float16(*config_and_inputs)

    def test_ctc_loss_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_loss(*config_and_inputs)

    def test_ctc_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_training(*config_and_inputs)

    def test_labels_out_of_vocab(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_labels_out_of_vocab(*config_and_inputs)

    @unittest.skip(reason="PhoneticXeus has no inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="PhoneticXeus has input_values instead of input_ids")
    def test_forward_signature(self):
        pass

    @unittest.skip(reason="PhoneticXeus has no token embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
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

        input_values = inputs_dict["input_values"]

        input_lengths = torch.tensor(
            [input_values.shape[1] for _ in range(input_values.shape[0])], dtype=torch.long, device=torch_device
        )
        output_lengths = model._get_feat_extract_output_lengths(input_lengths)

        labels = ids_tensor((input_values.shape[0], output_lengths[0] - 2), self.model_tester.vocab_size)
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

    # overwrite from test_modeling_common
    def _mock_init_weights(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.fill_(3)
        if hasattr(module, "weight_g") and module.weight_g is not None:
            module.weight_g.data.fill_(3)
        if hasattr(module, "weight_v") and module.weight_v is not None:
            module.weight_v.data.fill_(3)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.fill_(3)

    @slow
    def test_model_from_pretrained(self):
        model = PhoneticXeusModel.from_pretrained("changelinglab/PhoneticXeus-hf")
        self.assertIsNotNone(model)


@require_torch
@slow
class PhoneticXeusModelIntegrationTest(unittest.TestCase):
    def _load_datasamples(self, num_samples):
        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        speech_samples = ds.sort("id").filter(lambda x: x["id"] in [f"1272-141231-000{i}" for i in range(num_samples)])
        speech_samples = speech_samples[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def test_inference_ctc_batched(self):
        model = PhoneticXeusForCTC.from_pretrained("changelinglab/PhoneticXeus-hf")
        model.to(torch_device)
        model.eval()

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("changelinglab/PhoneticXeus-hf")

        input_speech = self._load_datasamples(2)

        inputs = feature_extractor(input_speech, return_tensors="pt", padding=True, sampling_rate=16000)

        input_values = inputs.input_values.to(torch_device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)

        # Verify output shape: batch=2, time steps, vocab=428
        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[-1], 428)

        # Verify predictions are valid token ids
        self.assertTrue((predicted_ids >= 0).all())
        self.assertTrue((predicted_ids < 428).all())
