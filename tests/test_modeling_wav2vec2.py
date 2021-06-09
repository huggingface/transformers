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

from tests.test_modeling_common import floats_tensor, ids_tensor, random_attention_mask
from transformers import is_torch_available
from transformers.testing_utils import require_datasets, require_soundfile, require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, _config_zero_init


if is_torch_available():
    import torch

    from transformers import (
        Wav2Vec2Config,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForCTC,
        Wav2Vec2ForMaskedLM,
        Wav2Vec2ForPreTraining,
        Wav2Vec2Model,
        Wav2Vec2Processor,
    )
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2GumbelVectorQuantizer, _compute_mask_indices


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
        # test does not pass for models making use of `group_norm`
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

    def check_ctc_loss(self, config, input_values, *args):
        model = Wav2Vec2ForCTC(config=config)
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
        sum_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss

        model.config.ctc_loss_reduction = "mean"
        mean_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss

        self.parent.assertTrue(abs(labels.shape[0] * labels.shape[1] * mean_loss.item() - sum_loss.item()) < 1e-3)

    def check_training(self, config, input_values, *args):
        config.ctc_zero_infinity = True
        model = Wav2Vec2ForCTC(config=config)
        model.to(torch_device)
        model.train()

        # freeze feature encoder
        model.freeze_feature_extractor()

        input_values = input_values[:3]

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_values.shape[0], max(max_length_labels) - 2), model.config.vocab_size)

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0

            if max_length_labels[i] < labels.shape[-1]:
                # it's important that we make sure that target lenghts are at least
                # one shorter than logit lenghts to prevent -inf
                labels[i, max_length_labels[i] - 1 :] = -100

        loss = model(input_values, labels=labels).loss
        self.parent.assertFalse(torch.isinf(loss).item())

        loss.backward()

    def prepare_config_and_inputs_for_common(self):
        config, input_values, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_values": input_values, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class Wav2Vec2ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2ForMaskedLM, Wav2Vec2ForPreTraining) if is_torch_available() else ()
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

    def test_ctc_loss_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_loss(*config_and_inputs)

    def test_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_training(*config_and_inputs)

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
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.assertIsNotNone(model)


@require_torch
class Wav2Vec2RobustModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2ForMaskedLM, Wav2Vec2ForPreTraining) if is_torch_available() else ()
    )
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

    def test_ctc_loss_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_loss(*config_and_inputs)

    def test_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_training(*config_and_inputs)

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

    def test_model_for_pretraining(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Wav2Vec2ForPreTraining(config).to(torch_device)

        features_shape = (
            inputs_dict["input_values"].shape[0],
            model._get_feat_extract_output_lengths(torch.tensor(inputs_dict["input_values"].shape[1])),
        )

        mask_time_indices = _compute_mask_indices(
            features_shape,
            model.config.mask_time_prob,
            model.config.mask_time_length,
            device=inputs_dict["input_values"].device,
            min_masks=2,
        ).to(torch_device)

        loss = model(
            inputs_dict["input_values"],
            attention_mask=inputs_dict["attention_mask"],
            mask_time_indices=mask_time_indices,
        ).loss

        mask_time_indices[:, : mask_time_indices.shape[-1] // 2] = True
        loss_more_masked = model(
            inputs_dict["input_values"],
            attention_mask=inputs_dict["attention_mask"],
            mask_time_indices=mask_time_indices,
        ).loss

        # loss_more_masked has to be bigger or equal loss since more masked inputs have to be predicted
        self.assertTrue(loss.detach().item() <= loss_more_masked.detach().item())

    @slow
    def test_model_from_pretrained(self):
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.assertIsNotNone(model)


@require_torch
class Wav2Vec2UtilsTest(unittest.TestCase):
    def test_compute_mask_indices(self):
        batch_size = 4
        sequence_length = 60
        mask_prob = 0.5
        mask_length = 1

        mask = _compute_mask_indices((batch_size, sequence_length), mask_prob, mask_length, torch_device)

        self.assertListEqual(mask.sum(axis=-1).tolist(), [mask_prob * sequence_length for _ in range(batch_size)])

    def test_compute_mask_indices_overlap(self):
        batch_size = 4
        sequence_length = 80
        mask_prob = 0.5
        mask_length = 4

        mask = _compute_mask_indices((batch_size, sequence_length), mask_prob, mask_length, torch_device)

        # because of overlap mask don't have to add up exactly to `mask_prob * sequence_length`, but have to be smaller or equal
        for batch_sum in mask.sum(axis=-1):
            self.assertTrue(int(batch_sum) <= mask_prob * sequence_length)

    def test_compute_perplexity(self):
        probs = torch.arange(100, device=torch_device).reshape(2, 5, 10) / 100

        ppl = Wav2Vec2GumbelVectorQuantizer._compute_perplexity(probs)
        self.assertTrue(abs(ppl.item() - 141.4291) < 1e-3)

        # mask half of the input
        mask = torch.ones((2,), device=torch_device, dtype=torch.bool)
        mask[0] = 0

        ppl = Wav2Vec2GumbelVectorQuantizer._compute_perplexity(probs, mask)
        self.assertTrue(abs(ppl.item() - 58.6757) < 1e-3)

    def test_sample_negatives(self):
        batch_size = 2
        sequence_length = 10
        hidden_size = 4
        num_negatives = 3

        features = (torch.arange(sequence_length * hidden_size, device=torch_device) // hidden_size).view(
            sequence_length, hidden_size
        )  # each value in vector consits of same value
        features = features[None, :].expand(batch_size, sequence_length, hidden_size).contiguous()

        negatives = Wav2Vec2ForPreTraining._sample_negatives(features, num_negatives)

        self.assertTrue(negatives.shape == (num_negatives, batch_size, sequence_length, hidden_size))

        # make sure no negatively sampled vector is actually a positive one
        for negative in negatives:
            self.assertTrue(((negative - features) == 0).sum() == 0.0)

        # make sure that full vectors are sampled and not values of vectors => this means that `unique()` yields a single value for `hidden_size` dim
        self.assertTrue(negatives.unique(dim=-1).shape, (num_negatives, batch_size, sequence_length, 1))


@require_torch
@require_datasets
@require_soundfile
@slow
class Wav2Vec2ModelIntegrationTest(unittest.TestCase):
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
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        model.to(torch_device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", do_lower_case=True)
        input_speech = self._load_datasamples(1)

        input_values = processor(input_speech, return_tensors="pt").input_values.to(torch_device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = ["a man said to the universe sir i exist"]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_ctc_normal_batched(self):
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        model.to(torch_device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", do_lower_case=True)

        input_speech = self._load_datasamples(2)

        inputs = processor(input_speech, return_tensors="pt", padding=True, truncation=True)

        input_values = inputs.input_values.to(torch_device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe sir i exist",
            "sweat covered brion's body trickling into the tight lowing cloth that was the only garment he wore",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_ctc_robust_batched(self):
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(torch_device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", do_lower_case=True)

        input_speech = self._load_datasamples(4)

        inputs = processor(input_speech, return_tensors="pt", padding=True, truncation=True)

        input_values = inputs.input_values.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)

        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe sir i exist",
            "sweat covered brion's body trickling into the tight loin cloth that was the only garment he wore",
            "the cut on his chest still dripping blood the ache of his overstrained eyes even the soaring arena around him with the thousands of spectators were trivialities not worth thinking about",
            "his instant panic was followed by a small sharp blow high on his chest",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_integration(self):
        model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")
        model.to(torch_device)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base", return_attention_mask=True
        )
        input_speech = self._load_datasamples(2)

        inputs_dict = feature_extractor(input_speech, return_tensors="pt", padding=True)

        features_shape = (
            inputs_dict["input_values"].shape[0],
            model._get_feat_extract_output_lengths(torch.tensor(inputs_dict["input_values"].shape[1])),
        )

        torch.manual_seed(0)
        mask_time_indices = _compute_mask_indices(
            features_shape,
            model.config.mask_time_prob,
            model.config.mask_time_length,
            device=inputs_dict["input_values"].device,
            min_masks=2,
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(
                inputs_dict.input_values.to(torch_device),
                attention_mask=inputs_dict.attention_mask.to(torch_device),
                mask_time_indices=mask_time_indices,
            )

        # compute cosine similarity
        cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)

        # retrieve cosine sim of masked features
        cosine_sim_masked = cosine_sim[mask_time_indices]

        # fmt: off
        expected_cosine_sim_masked = torch.tensor(
            [0.7458, 0.7188, 0.6418, 0.3729, 0.3741, 0.3694, 0.3110, 0.2257, 0.4403, 0.5415, 0.3950, 0.3701, 0.8831, 0.8613, 0.5229, 0.6696, 0.7206, 0.7877, 0.6758, 0.8746, 0.6596, 0.6282, 0.6178, 0.5839, 0.5926, 0.6651, 0.4635, 0.6332, 0.6572, 0.8776, 0.4999, 0.7001, 0.7257, 0.5098, 0.6229, 0.4566, 0.5261, 0.6363, 0.5371, 0.6997],
            device=torch_device,
        )
        # fmt: on

        self.assertTrue(torch.allclose(cosine_sim_masked, expected_cosine_sim_masked, atol=1e-3))

    def test_inference_pretrained(self):
        model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")
        model.to(torch_device)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base", return_attention_mask=True
        )
        input_speech = self._load_datasamples(2)

        inputs_dict = feature_extractor(input_speech, return_tensors="pt", padding=True)

        features_shape = (
            inputs_dict["input_values"].shape[0],
            model._get_feat_extract_output_lengths(torch.tensor(inputs_dict["input_values"].shape[1])),
        )

        torch.manual_seed(0)
        mask_time_indices = _compute_mask_indices(
            features_shape,
            model.config.mask_time_prob,
            model.config.mask_time_length,
            device=inputs_dict["input_values"].device,
            min_masks=2,
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(
                inputs_dict.input_values.to(torch_device),
                attention_mask=inputs_dict.attention_mask.to(torch_device),
                mask_time_indices=mask_time_indices,
            )

        # compute cosine similarity
        cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)

        # retrieve cosine sim of masked features
        cosine_sim_masked = cosine_sim[mask_time_indices]

        # ... now compare to randomly initialized model

        config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
        model_rand = Wav2Vec2ForPreTraining(config).to(torch_device).eval()

        with torch.no_grad():
            outputs_rand = model_rand(
                inputs_dict.input_values.to(torch_device),
                attention_mask=inputs_dict.attention_mask.to(torch_device),
                mask_time_indices=mask_time_indices,
            )

        # compute cosine similarity
        cosine_sim_rand = torch.cosine_similarity(
            outputs_rand.projected_states, outputs_rand.projected_quantized_states, dim=-1
        )

        # retrieve cosine sim of masked features
        cosine_sim_masked_rand = cosine_sim_rand[mask_time_indices]

        # a pretrained wav2vec2 model has learned to predict the quantized latent states
        # => the cosine similarity between quantized states and predicted states > 0.5
        # a random wav2vec2 model has not learned to predict the quantized latent states
        # => the cosine similarity between quantized states and predicted states is very likely < 0.1
        self.assertTrue(cosine_sim_masked.mean().item() - 5 * cosine_sim_masked_rand.mean().item() > 0)

    @unittest.skipIf(torch_device != "cpu", "cannot make deterministic on GPU")
    def test_loss_pretraining(self):
        model = Wav2Vec2ForPreTraining.from_pretrained(
            "facebook/wav2vec2-base",
            attention_dropout=0.0,
            feat_proj_dropout=0.0,
            hidden_dropout=0.0,
            layerdrop=0.0,
        )
        model.to(torch_device).train()

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base", return_attention_mask=True
        )
        input_speech = self._load_datasamples(2)

        inputs_dict = feature_extractor(input_speech, return_tensors="pt", padding=True)

        features_shape = (
            inputs_dict["input_values"].shape[0],
            model._get_feat_extract_output_lengths(inputs_dict["input_values"].shape[1]),
        )

        torch.manual_seed(0)
        mask_time_indices = _compute_mask_indices(
            features_shape,
            model.config.mask_time_prob,
            model.config.mask_time_length,
            device=inputs_dict["input_values"].device,
            min_masks=2,
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(
                inputs_dict.input_values.to(torch_device),
                attention_mask=inputs_dict.attention_mask.to(torch_device),
                mask_time_indices=mask_time_indices,
            )

        # check diversity loss
        num_codevectors = model.config.num_codevectors_per_group * model.config.num_codevector_groups
        diversity_loss = (num_codevectors - outputs.codevector_perplexity) / num_codevectors
        self.assertTrue(abs(diversity_loss.item() - 0.8859) < 1e-3)

        # check overall loss (contrastive loss + diversity loss)
        expected_loss = 62.5170

        self.assertTrue(abs(outputs.loss.item() - expected_loss) < 1e-3)
