# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import tempfile
import unittest

from transformers import (
    YuE2AcousticTokenizerConfig,
    YuE2AcousticTokenizerDecoderConfig,
    YuE2AcousticTokenizerEncoderConfig,
)
from transformers.testing_utils import is_torch_available, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import (
        YuE2AcousticTokenizerDecoderModel,
        YuE2AcousticTokenizerEncoderModel,
        YuE2AcousticTokenizerModel,
    )


class YuE2AcousticTokenizerModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_frames=5,
        audio_channels=2,
        hidden_sizes=[4, 4, 8],
        downsampling_ratios=[2, 3],
        latent_dim=6,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.audio_channels = audio_channels
        self.hidden_sizes = hidden_sizes
        self.downsampling_ratios = downsampling_ratios
        self.latent_dim = latent_dim
        self.is_training = is_training

    def get_config(self):
        return YuE2AcousticTokenizerConfig(
            audio_channels=self.audio_channels,
            hidden_sizes=self.hidden_sizes,
            downsampling_ratios=self.downsampling_ratios,
            latent_dim=self.latent_dim,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_values = floats_tensor([self.batch_size, self.audio_channels, self.num_frames * config.hop_length])
        return config, {"input_values": input_values}

    def prepare_config_and_inputs_for_common(self):
        return self.prepare_config_and_inputs()

    def create_and_check_model_forward(self, config, inputs_dict):
        model = YuE2AcousticTokenizerModel(config=config).to(torch_device).eval()
        with torch.no_grad():
            result = model(**inputs_dict)
        self.parent.assertEqual(result.latents.shape, (self.batch_size, self.num_frames, self.latent_dim))
        # the transposed convolutions of the decoder crop `kernel_size - stride` samples per block
        self.parent.assertEqual(result.audio_values.shape[:2], (self.batch_size, self.audio_channels))


@require_torch
class YuE2AcousticTokenizerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (YuE2AcousticTokenizerModel,) if is_torch_available() else ()
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = YuE2AcousticTokenizerModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=YuE2AcousticTokenizerConfig, common_properties=[], has_text_modality=False
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_encode_sample(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = YuE2AcousticTokenizerModel(config).to(torch_device).eval()
        with torch.no_grad():
            mean = model.encode(inputs_dict["input_values"]).latents
            sampled = model.encode(inputs_dict["input_values"], sample=True).latents
        self.assertEqual(sampled.shape, mean.shape)
        self.assertFalse(torch.allclose(sampled, mean))

    def test_config_without_encoder_or_decoder(self):
        config = self.model_tester.get_config()
        self.assertNotIn("encoder_config", config.to_diff_dict())

        config.encoder_config = None
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            config = YuE2AcousticTokenizerConfig.from_pretrained(tmp_dir)
        self.assertIsNone(config.encoder_config)
        self.assertIsInstance(config.decoder_config, YuE2AcousticTokenizerDecoderConfig)

        with self.assertRaises(ValueError):
            config.decoder_config = None
        with self.assertRaises(ValueError):
            config.encoder_config = YuE2AcousticTokenizerEncoderConfig()

    def test_model_without_encoder_or_decoder(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_values = inputs_dict["input_values"]
        model = YuE2AcousticTokenizerModel(config).to(torch_device).eval()
        with torch.no_grad():
            latents = model.encode(input_values).latents
            audio_values = model.decode(latents).audio_values

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            encoder_only, encoder_info = YuE2AcousticTokenizerModel.from_pretrained(
                tmp_dir, decoder_config=None, output_loading_info=True
            )
            decoder_only, decoder_info = YuE2AcousticTokenizerModel.from_pretrained(
                tmp_dir, encoder_config=None, output_loading_info=True
            )
            # each half also loads on its own from the full checkpoint
            encoder_half, encoder_half_info = YuE2AcousticTokenizerEncoderModel.from_pretrained(
                tmp_dir, output_loading_info=True
            )
            decoder_half, decoder_half_info = YuE2AcousticTokenizerDecoderModel.from_pretrained(
                tmp_dir, output_loading_info=True
            )
        for loading_info in (encoder_info, decoder_info, encoder_half_info, decoder_half_info):
            self.assertFalse(any(loading_info.values()), loading_info)

        for submodel in (encoder_only, decoder_only, encoder_half, decoder_half):
            submodel.to(torch_device).eval()
        with torch.no_grad():
            torch.testing.assert_close(encoder_only.encode(input_values).latents, latents)
            torch.testing.assert_close(decoder_only.decode(latents).audio_values, audio_values)
            torch.testing.assert_close(encoder_half(input_values).latents, latents)
            torch.testing.assert_close(decoder_half(latents).audio_values, audio_values)
        with self.assertRaises(ValueError):
            encoder_only.decode(latents)
        with self.assertRaises(ValueError):
            decoder_only.encode(input_values)

    @unittest.skip(reason="The acoustic tokenizer has no token embeddings.")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="The acoustic tokenizer is convolutional: it has no attention.")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="The acoustic tokenizer is convolutional: it records no hidden states.")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="The acoustic tokenizer is convolutional: it has no attention nor recorded hidden states.")
    def test_retain_grad_hidden_states_attentions(self):
        pass


@require_torch
class YuE2AcousticTokenizerIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    """
    Expected values from the original implementation (`yue2.modeling_vae.YuE2VAE` of the `yue2_infer` 0.1.5 wheel
    shipped with https://huggingface.co/m-a-p/YuE2-3B), on a CUDA device with TF32 disabled, as the original pipeline runs:

    ```python
    import torch
    from yue2.modeling_vae import YuE2VAE

    torch.backends.cudnn.allow_tf32 = False
    generator = torch.Generator().manual_seed(0)
    audio = 0.1 * torch.randn(1, 2, 1920 * 25, generator=generator)
    latents = torch.randn(1, 64, 25, generator=generator)
    model = YuE2VAE.from_pretrained("m-a-p/YuE2-Vae", device="cuda")
    with torch.inference_mode():
        print(model.encode(audio.cuda())[0, :3, :5].T)  # EXPECTED_LATENTS
        print(model.decode(latents.cuda())[0, :, 1000:1006])  # EXPECTED_AUDIO
    ```
    """

    checkpoint = "bezzam/yue2-acoustic-tokenizer"

    def setUp(self):
        super().setUp()
        generator = torch.Generator().manual_seed(0)
        self.audio = 0.1 * torch.randn(1, 2, 1920 * 25, generator=generator)
        self.latents = torch.randn(1, 64, 25, generator=generator).transpose(1, 2)

    @slow
    def test_encode(self):
        model = YuE2AcousticTokenizerModel.from_pretrained(self.checkpoint, device_map=torch_device)
        # TF32 convolutions, PyTorch's default on recent GPUs, move the latents by up to 6e-4
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
            latents = model.encode(self.audio.to(torch_device)).latents
        # fmt: off
        EXPECTED_LATENTS = torch.tensor([
            [0.798269, 0.673941, -0.226476],
            [1.905051, -0.001175, 0.167258],
            [1.279705, 0.309271, 0.173560],
            [1.841689, -0.261596, 0.271040],
            [1.466523, 0.176592, 0.295676],
        ])
        # fmt: on
        self.assertEqual(latents.shape, (1, 25, 64))
        torch.testing.assert_close(latents[0, :5, :3].cpu(), EXPECTED_LATENTS, rtol=0, atol=1e-5)

    @slow
    def test_decode(self):
        model = YuE2AcousticTokenizerModel.from_pretrained(self.checkpoint, device_map=torch_device)
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
            audio = model.decode(self.latents.to(torch_device)).audio_values
        # fmt: off
        EXPECTED_AUDIO = torch.tensor([
            [-0.044364, -0.046931, -0.042471, -0.033022, -0.035489, -0.044477],
            [-0.045536, -0.048896, -0.041945, -0.032035, -0.044850, -0.054282],
        ])
        # fmt: on
        # the decoder's transposed convolutions crop 64 samples overall
        self.assertEqual(audio.shape, (1, 2, 1920 * 25 - 64))
        torch.testing.assert_close(audio[0, :, 1000:1006].cpu(), EXPECTED_AUDIO, rtol=0, atol=1e-5)
