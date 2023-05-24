



# # # # # # # # # # # # # # # # # # # # # # #
#  WIP - Currently has old draft FS2 tests  #
# # # # # # # # # # # # # # # # # # # # # # #



# # coding=utf-8
# # Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """ Testing suite for the PyTorch FastSpeech2Conformer model. """

# import unittest

# import numpy as np

# from transformers import FastSpeech2ConformerConfig, is_torch_available
# from transformers.testing_utils import require_torch, slow, torch_device

# from ...test_configuration_common import ConfigTester
# from ...test_modeling_common import ModelTesterMixin, ids_tensor


# if is_torch_available():
#     import torch

#     from transformers import FastSpeech2ConformerModel, FastSpeech2ConformerTokenizer


# class FastSpeech2ConformerModelTester:
#     def __init__(
#         self,
#         parent,
#         batch_size=13,
#         seq_length=7,
#         is_training=False,
#         encoder_embed_dim=256,
#         speaker_embed_dim=64,
#         max_source_positions=1024,
#         encoder_attention_heads=2,
#         fft_hidden_dim=1024,
#         fft_kernel_size=9,
#         fft_dropout=0.2,
#         attention_dropout=0,
#         encoder_layers=4,
#         decoder_embed_dim=256,
#         decoder_attention_heads=2,
#         decoder_layers=4,
#         add_postnet=False,
#         postnet_conv_dim=512,
#         postnet_conv_kernel_size=5,
#         postnet_layers=5,
#         postnet_dropout=0.5,
#         vocab_size=75,
#         num_speakers=1,
#         var_pred_hidden_dim=256,
#         var_pred_kernel_size=3,
#         var_pred_dropout=0.5,
#         pitch_max=5.733940816898645,
#         pitch_min=-4.660287183665281,
#         energy_max=3.2244551181793213,
#         energy_min=-4.9544901847839355,
#         initializer_range=0.0625,
#         use_mean=True,
#         use_standard_deviation=True,
#     ):
#         self.parent = parent
#         self.batch_size = batch_size
#         self.seq_length = seq_length
#         self.is_training = is_training
#         self.encoder_embed_dim = encoder_embed_dim
#         self.speaker_embed_dim = speaker_embed_dim
#         self.encoder_embed_dim = encoder_embed_dim
#         self.max_source_positions = max_source_positions
#         self.encoder_attention_heads = encoder_attention_heads
#         self.fft_hidden_dim = fft_hidden_dim
#         self.fft_kernel_size = fft_kernel_size
#         self.fft_dropout = fft_dropout
#         self.attention_dropout = attention_dropout
#         self.encoder_layers = encoder_layers
#         self.decoder_embed_dim = decoder_embed_dim
#         self.decoder_attention_heads = decoder_attention_heads
#         self.decoder_layers = decoder_layers
#         self.add_postnet = add_postnet
#         self.postnet_conv_dim = postnet_conv_dim
#         self.postnet_conv_kernel_size = postnet_conv_kernel_size
#         self.postnet_layers = postnet_layers
#         self.postnet_dropout = postnet_dropout
#         self.vocab_size = vocab_size
#         self.num_speakers = num_speakers
#         self.var_pred_hidden_dim = var_pred_hidden_dim
#         self.var_pred_kernel_size = var_pred_kernel_size
#         self.var_pred_dropout = var_pred_dropout
#         self.pitch_min = pitch_min
#         self.pitch_max = pitch_max
#         self.energy_min = energy_min
#         self.energy_max = energy_max
#         self.initializer_range = initializer_range
#         self.use_mean = use_mean
#         self.use_standard_deviation = use_standard_deviation
#         self.initializer_range = initializer_range

#     def prepare_config_and_inputs(self):
#         config = self.get_config()
#         input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
#         return config, input_ids

#     def get_config(self):
#         return FastSpeech2ConformerConfig(
#             encoder_embed_dim=self.encoder_embed_dim,
#             speaker_embed_dim=self.speaker_embed_dim,
#             max_source_positions=self.max_source_positions,
#             encoder_attention_heads=self.encoder_attention_heads,
#             fft_dropout=self.fft_dropout,
#             fft_hidden_dim=self.fft_hidden_dim,
#             fft_kernel_size=self.fft_kernel_size,
#             attention_dropout=self.attention_dropout,
#             encoder_layers=self.encoder_layers,
#             decoder_embed_dim=self.decoder_embed_dim,
#             decoder_attention_heads=self.decoder_attention_heads,
#             decoder_layers=self.decoder_layers,
#             add_postnet=self.add_postnet,
#             postnet_conv_dim=self.postnet_conv_dim,
#             postnet_conv_kernel_size=self.postnet_conv_kernel_size,
#             postnet_layer=self.postnet_layers,
#             postnet_dropout=self.postnet_dropout,
#             vocab_size=self.vocab_size,
#             num_speakers=self.num_speakers,
#             var_pred_hidden_dim=self.var_pred_hidden_dim,
#             var_pred_kernel_size=self.var_pred_kernel_size,
#             var_pred_dropout=self.var_pred_dropout,
#             pitch_min=self.pitch_min,
#             pitch_max=self.pitch_max,
#             energy_min=self.energy_min,
#             energy_max=self.energy_max,
#             initializer_range=self.initializer_range,
#             use_mean=self.use_mean,
#             use_standard_deviation=self.use_standard_deviation,
#         )

#     def create_and_check_model(self, config, input_values, *args):
#         model = FastSpeech2ConformerModel(config=config)
#         model.to(torch_device)
#         model.eval()
#         result = model(input_values, return_dict=True)

#         # total of 5 keys in result
#         self.parent.assertEqual(len(result), 5)
#         # check batch sizes match
#         for value in result.values():
#             self.parent.assertEqual(value.size(0), self.batch_size)
#         # check log_duration, pitch, and energy have same shape
#         for i in range(2, 4):
#             self.parent.assertEqual(result[i].shape, result[i + 1].shape)
#         # check predicted mel-spectrogram has correct dimension
#         self.parent.assertEqual(result.mel_spectrogram.size(2), model.config.mel_dim)

#     def prepare_config_and_inputs_for_common(self):
#         config, input_ids = self.prepare_config_and_inputs()
#         inputs_dict = {"input_ids": input_ids}
#         return config, inputs_dict


# @require_torch
# class FastSpeech2ConformerModelTest(ModelTesterMixin, unittest.TestCase):
#     all_model_classes = (FastSpeech2ConformerModel,) if is_torch_available() else ()
#     test_pruning = False
#     test_headmasking = False
#     test_torchscript = False

#     def setUp(self):
#         self.model_tester = FastSpeech2ConformerModelTester(self)
#         self.config_tester = ConfigTester(self, config_class=FastSpeech2ConformerConfig)
#         # FastSpeech2ConformerConfig does not have `hidden_size`
#         self.config_tester.create_and_test_config_common_properties = lambda: None

#     def test_config(self):
#         self.config_tester.run_common_tests()

#     def test_model(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs()
#         self.model_tester.create_and_check_model(*config_and_inputs)

#     @unittest.skip(reason="FastSpeech2 does not output attentions")
#     def test_attention_outputs(self):
#         pass

#     @unittest.skip(reason="FastSpeech2 does not output attentions")
#     def test_retain_grad_hidden_states_attentions(sef):
#         pass

#     @unittest.skip(reason="FastSpeech2 does not output hidden_states")
#     def test_hidden_states_output(self):
#         pass

#     @unittest.skip(reason="FastSpeech2 does not accept inputs_embeds")
#     def test_inputs_embeds(self):
#         pass

#     @unittest.skip(reason="Feed forward chunking is not implemented")
#     def test_feed_forward_chunking(self):
#         pass

#     def test_initialization(self):
#         # TODO
#         pass

#     @slow
#     def test_model_from_pretrained(self):
#         model = FastSpeech2ConformerModel.from_pretrained("jaketae/fastspeech2-commonvoice")
#         self.assertIsNotNone(model)


# @require_torch
# @slow
# class FastSpeech2ConformerModelIntegrationTest(unittest.TestCase):
#     @unittest.skipIf(torch_device != "cpu", "cannot make deterministic on GPU")
#     def test_inference_integration(self):
#         model = FastSpeech2ConformerModel.from_pretrained("jaketae/fastspeech2-ljspeech")
#         model.to(torch_device)
#         tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("jaketae/fastspeech2-ljspeech")
#         text = ["This is a test sentence."]
#         inputs = tokenizer(text, return_tensors="pt", padding=True).to(torch_device)

#         torch.manual_seed(0)
#         np.random.seed(0)

#         with torch.no_grad():
#             output = model(**inputs, return_dict=True)

#         # mel-spectrogram is too large (123, 80),
#         # so only check top-left 100 elements
#         expected_mel_spectrogram = torch.tensor(
#             [
#                 [-7.4015, -7.0025, -6.5533, -6.4955, -6.4318, -5.9519, -5.6957, -5.7323, -5.8078, -5.7491],
#                 [-6.8250, -6.3284, -5.7400, -5.3538, -4.1562, -2.9194, -2.1966, -2.1588, -2.4447, -2.5317],
#                 [-6.2405, -5.6605, -4.9551, -4.7267, -3.7683, -2.1692, -1.1314, -1.4534, -1.9853, -1.8161],
#                 [-6.2821, -5.5036, -4.6862, -4.6298, -4.2279, -2.4826, -0.7308, -1.0276, -2.2183, -2.4050],
#                 [-6.2990, -5.5917, -4.8082, -4.9588, -4.6967, -2.7377, -0.3793, -0.4573, -2.6168, -3.7818],
#                 [-6.1249, -5.5540, -4.8532, -4.9064, -4.3390, -2.4323, -0.0720, -0.0985, -2.3689, -3.8561],
#                 [-5.9463, -5.5087, -4.8651, -4.7078, -3.5536, -1.7139, -0.0421, -0.4204, -2.4050, -3.7703],
#                 [-5.7945, -5.4198, -4.8480, -4.6813, -3.1621, -1.3546, -0.5719, -1.4292, -2.9163, -3.7725],
#                 [-5.7746, -5.4186, -4.8243, -4.6667, -3.5592, -2.2771, -1.8510, -2.6992, -3.8929, -4.3657],
#                 [-5.9439, -5.5150, -4.9092, -4.8813, -4.5146, -3.5995, -2.8243, -3.0832, -4.2484, -4.7525],
#             ],
#             device=torch_device,
#         )

#         self.assertTrue(torch.allclose(output.mel_spectrogram[0, :10, :10], expected_mel_spectrogram, atol=1e-3))
