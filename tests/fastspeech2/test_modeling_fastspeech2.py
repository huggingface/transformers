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
""" Testing suite for the PyTorch FastSpeech2 model. """

import math
import unittest

import numpy as np
from datasets import load_dataset

from transformers import FastSpeech2Config, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ..test_configuration_common import ConfigTester
from ..test_modeling_common import ModelTesterMixin, _config_zero_init, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import FastSpeech2Model, FastSpeech2Tokenizer


class FastSpeech2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=False,
        # use_input_mask=True,
        # use_token_type_ids=True,
        # use_labels=True,
        n_frames_per_step=1,
        output_frame_dim=80,
        encoder_embed_dim=256,
        speaker_embed_dim=64,
        dropout=0.2,
        max_source_positions=1024,
        encoder_attention_heads=2,
        fft_hidden_dim=1024,
        fft_kernel_size=9,
        attention_dropout=0,
        encoder_layers=4,
        decoder_embed_dim=256,
        decoder_attention_heads=2,
        decoder_layers=4,
        add_postnet=False,
        postnet_conv_dim=512,
        postnet_conv_kernel_size=5,
        postnet_layers=5,
        postnet_dropout=0.5,
        vocab_size=75,
        num_speakers=1,
        var_pred_n_bins=256,
        var_pred_hidden_dim=256,
        var_pred_kernel_size=3,
        var_pred_dropout=0.5,
        pitch_max=5.733940816898645,
        pitch_min=-4.660287183665281,
        energy_max=3.2244551181793213,
        energy_min=-4.9544901847839355,
        initializer_range=0.0625,
        mean=True,
        std=True,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.n_frames_per_step = n_frames_per_step
        self.output_frame_dim = output_frame_dim
        self.encoder_embed_dim = encoder_embed_dim
        self.speaker_embed_dim = speaker_embed_dim
        self.encoder_embed_dim = encoder_embed_dim
        self.dropout = dropout
        self.max_source_positions = max_source_positions
        self.encoder_attention_heads = encoder_attention_heads
        self.fft_hidden_dim = fft_hidden_dim
        self.fft_kernel_size = fft_kernel_size
        self.attention_dropout = attention_dropout
        self.encoder_layers = encoder_layers
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_layers = decoder_layers
        self.add_postnet = add_postnet
        self.postnet_conv_dim = postnet_conv_dim
        self.postnet_conv_kernel_size = postnet_conv_kernel_size
        self.postnet_layers = postnet_layers
        self.postnet_dropout = postnet_dropout
        self.vocab_size = vocab_size
        self.num_speakers = num_speakers
        self.var_pred_n_bins = var_pred_n_bins
        self.var_pred_hidden_dim = var_pred_hidden_dim
        self.var_pred_kernel_size = var_pred_kernel_size
        self.var_pred_dropout = var_pred_dropout
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.initializer_range = initializer_range
        self.mean = mean
        self.std = std
        self.initializer_range = initializer_range
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_values = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        return config, input_values, attention_mask

    def get_config(self):
        return FastSpeech2Config(
            n_frames_per_step=self.n_frames_per_step,
            output_frame_dim=self.output_frame_dim,
            encoder_embed_dim=self.encoder_embed_dim,
            speaker_embed_dim=self.speaker_embed_dim,
            dropout=self.dropout,
            max_source_positions=self.max_source_positions,
            encoder_attention_heads=self.encoder_attention_heads,
            fft_hidden_dim=self.fft_hidden_dim,
            fft_kernel_size=self.fft_kernel_size,
            attention_dropout=self.attention_dropout,
            encoder_layers=self.encoder_layers,
            decoder_embed_dim=self.decoder_embed_dim,
            decoder_attention_heads=self.decoder_attention_heads,
            decoder_layers=self.decoder_layers,
            add_postnet=self.add_postnet,
            postnet_conv_dim=self.postnet_conv_dim,
            postnet_conv_kernel_size=self.postnet_conv_kernel_size,
            postnet_layer=self.postnet_layers,
            postnet_dropout=self.postnet_dropout,
            vocab_size=self.vocab_size,
            num_speakers=self.num_speakers,
            var_pred_n_bins=self.var_pred_n_bins,
            var_pred_hidden_dim=self.var_pred_hidden_dim,
            var_pred_kernel_size=self.var_pred_kernel_size,
            var_pred_dropout=self.var_pred_dropout,
            pitch_min=self.pitch_min,
            pitch_max=self.pitch_max,
            energy_min=self.energy_min,
            energy_max=self.energy_max,
            initializer_range=self.initializer_range,
            mean=self.mean,
            std=self.std,
        )

    def create_and_check_model(self, config, input_values, *args):
        model = FastSpeech2Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values)

        # total of 5 keys in result
        self.assertEqual(len(result), 5)
        # check batch sizes match
        for value in result:
            self.parent.assertEqual(value.size(0), self.batch_size)
        # check log_duration, pitch, and energy have same shape
        for i in range(2, 6):
            self.parent.assertEqual(result[i].shape, result[i + 1].shape)
        # check predicted mel-spectrogram has correct dimension
        self.parent.assertEqual(result.mel_spectrogram.size(2), self.output_frame_dim)

    def prepare_config_and_inputs_for_common(self):
        config, input_values, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_values": input_values, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class FastSpeech2ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (FastSpeech2Model,) if is_torch_available() else ()
    test_pruning = False
    test_headmasking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = FastSpeech2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=FastSpeech2Config)

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

    # def test_retain_grad_hidden_states_attentions(self):
    #     config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
    #     config.output_hidden_states = True
    #     config.output_attentions = True

    #     # no need to test all models as different heads yield the same functionality
    #     model_class = self.all_model_classes[0]
    #     model = model_class(config)
    #     model.to(torch_device)

    #     # set layer drop to 0
    #     model.config.layerdrop = 0.0

    #     input_values = inputs_dict["input_values"]

    #     input_lengths = torch.tensor(
    #         [input_values.shape[1] for _ in range(input_values.shape[0])], dtype=torch.long, device=torch_device
    #     )
    #     output_lengths = model._get_feat_extract_output_lengths(input_lengths)

    #     labels = ids_tensor((input_values.shape[0], output_lengths[0] - 2), self.model_tester.vocab_size)
    #     inputs_dict["attention_mask"] = torch.ones_like(inputs_dict["attention_mask"])
    #     inputs_dict["labels"] = labels

    #     outputs = model(**inputs_dict)

    #     output = outputs[0]

    #     # Encoder-/Decoder-only models
    #     hidden_states = outputs.hidden_states[0]
    #     attentions = outputs.attentions[0]

    #     hidden_states.retain_grad()
    #     attentions.retain_grad()

    #     output.flatten()[0].backward(retain_graph=True)

    #     self.assertIsNotNone(hidden_states.grad)
    #     self.assertIsNotNone(attentions.grad)

    # def test_initialization(self):
    #     config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

    #     configs_no_init = _config_zero_init(config)
    #     for model_class in self.all_model_classes:
    #         model = model_class(config=configs_no_init)
    #         for name, param in model.named_parameters():
    #             uniform_init_parms = [
    #                 "conv.weight",
    #                 "masked_spec_embed",
    #                 "codevectors",
    #                 "quantizer.weight_proj.weight",
    #                 "project_hid.weight",
    #                 "project_hid.bias",
    #                 "project_q.weight",
    #                 "project_q.bias",
    #                 "feature_projection.projection.weight",
    #                 "feature_projection.projection.bias",
    #                 "objective.weight",
    #             ]
    #             if param.requires_grad:
    #                 if any([x in name for x in uniform_init_parms]):
    #                     self.assertTrue(
    #                         -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
    #                         msg=f"Parameter {name} of model {model_class} seems not properly initialized",
    #                     )
    #                 else:
    #                     self.assertIn(
    #                         ((param.data.mean() * 1e9).round() / 1e9).item(),
    #                         [0.0, 1.0],
    #                         msg=f"Parameter {name} of model {model_class} seems not properly initialized",
    #                     )

    # # overwrite from test_modeling_common
    # def _mock_init_weights(self, module):
    #     if hasattr(module, "weight") and module.weight is not None:
    #         module.weight.data.fill_(3)
    #     if hasattr(module, "weight_g") and module.weight_g is not None:
    #         module.weight_g.data.fill_(3)
    #     if hasattr(module, "weight_v") and module.weight_v is not None:
    #         module.weight_v.data.fill_(3)
    #     if hasattr(module, "bias") and module.bias is not None:
    #         module.bias.data.fill_(3)
    #     if hasattr(module, "codevectors") and module.codevectors is not None:
    #         module.codevectors.data.fill_(3)
    #     if hasattr(module, "masked_spec_embed") and module.masked_spec_embed is not None:
    #         module.masked_spec_embed.data.fill_(3)

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = FastSpeech2Model.from_pretrained("jaketae/fastspeech2-commonvoice")
        self.assertIsNotNone(model)


# @require_torch
# @slow
# class Wav2Vec2ModelIntegrationTest(unittest.TestCase):
#     def _load_datasamples(self, num_samples):
#         ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
#         # automatic decoding with librispeech
#         speech_samples = ds.sort("id").filter(
#             lambda x: x["id"] in [f"1272-141231-000{i}" for i in range(num_samples)]
#         )[:num_samples]["audio"]

#         return [x["array"] for x in speech_samples]

#     def _load_superb(self, task, num_samples):
#         ds = load_dataset("anton-l/superb_dummy", task, split="test")

#         return ds[:num_samples]

#     # @unittest.skipIf(torch_device != "cpu", "cannot make deterministic on GPU")
#     def test_inference_integration(self):
#         model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")
#         model.to(torch_device)
#         feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
#         input_speech = self._load_datasamples(2)

#         inputs_dict = feature_extractor(input_speech, return_tensors="pt", padding=True)

#         batch_size = inputs_dict["input_values"].shape[0]
#         feature_seq_length = int(model._get_feat_extract_output_lengths(inputs_dict["input_values"].shape[1]))

#         features_shape = (batch_size, feature_seq_length)

#         np.random.seed(4)
#         mask_time_indices = _compute_mask_indices(
#             features_shape,
#             model.config.mask_time_prob,
#             model.config.mask_time_length,
#             min_masks=2,
#         )
#         mask_time_indices = torch.from_numpy(mask_time_indices).to(torch_device)

#         with torch.no_grad():
#             outputs = model(
#                 inputs_dict.input_values.to(torch_device),
#                 mask_time_indices=mask_time_indices,
#             )

#         # compute cosine similarity
#         cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)

#         # retrieve cosine sim of masked features
#         cosine_sim_masked = cosine_sim[mask_time_indices]

#         # cosine similarity of model is all > 0.5 as model is
#         # pre-trained on contrastive loss
#         # fmt: off
#         expected_cosine_sim_masked = torch.tensor([
#             0.8523, 0.5860, 0.6905, 0.5557, 0.7456, 0.5249, 0.6639, 0.7654, 0.7565,
#             0.8167, 0.8222, 0.7960, 0.8034, 0.8166, 0.8310, 0.8263, 0.8274, 0.8258,
#             0.8179, 0.8412, 0.8536, 0.5098, 0.4728, 0.6461, 0.4498, 0.6002, 0.5774,
#             0.6457, 0.7123, 0.5668, 0.6866, 0.4960, 0.6293, 0.7423, 0.7419, 0.7526,
#             0.7768, 0.4898, 0.5393, 0.8183
#         ], device=torch_device)
#         # fmt: on

#         self.assertTrue(torch.allclose(cosine_sim_masked, expected_cosine_sim_masked, atol=1e-3))

#     def test_inference_pretrained(self):
#         model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")
#         model.to(torch_device)
#         feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
#             "facebook/wav2vec2-base", return_attention_mask=True
#         )
#         input_speech = self._load_datasamples(2)

#         inputs_dict = feature_extractor(input_speech, return_tensors="pt", padding=True)

#         batch_size = inputs_dict["input_values"].shape[0]
#         feature_seq_length = int(model._get_feat_extract_output_lengths(inputs_dict["input_values"].shape[1]))

#         features_shape = (batch_size, feature_seq_length)

#         torch.manual_seed(0)
#         mask_time_indices = _compute_mask_indices(
#             features_shape,
#             model.config.mask_time_prob,
#             model.config.mask_time_length,
#             min_masks=2,
#         )
#         mask_time_indices = torch.from_numpy(mask_time_indices).to(torch_device)

#         with torch.no_grad():
#             outputs = model(
#                 inputs_dict.input_values.to(torch_device),
#                 attention_mask=inputs_dict.attention_mask.to(torch_device),
#                 mask_time_indices=mask_time_indices,
#             )

#         # compute cosine similarity
#         cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)

#         # retrieve cosine sim of masked features
#         cosine_sim_masked = cosine_sim[mask_time_indices]

#         # ... now compare to randomly initialized model

#         config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
#         model_rand = Wav2Vec2ForPreTraining(config).to(torch_device).eval()

#         with torch.no_grad():
#             outputs_rand = model_rand(
#                 inputs_dict.input_values.to(torch_device),
#                 attention_mask=inputs_dict.attention_mask.to(torch_device),
#                 mask_time_indices=mask_time_indices,
#             )

#         # compute cosine similarity
#         cosine_sim_rand = torch.cosine_similarity(
#             outputs_rand.projected_states, outputs_rand.projected_quantized_states, dim=-1
#         )

#         # retrieve cosine sim of masked features
#         cosine_sim_masked_rand = cosine_sim_rand[mask_time_indices]

#         # a pretrained wav2vec2 model has learned to predict the quantized latent states
#         # => the cosine similarity between quantized states and predicted states > 0.5
#         # a random wav2vec2 model has not learned to predict the quantized latent states
#         # => the cosine similarity between quantized states and predicted states is very likely < 0.1
#         self.assertTrue(cosine_sim_masked.mean().item() - 5 * cosine_sim_masked_rand.mean().item() > 0)
