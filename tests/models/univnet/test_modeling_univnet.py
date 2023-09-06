# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import gc
import inspect
import random
import unittest

import numpy as np
from datasets import Audio, load_dataset

from transformers import UnivNetFeatureExtractor, UnivNetGanConfig
from transformers.testing_utils import (
    is_torch_available,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
)


if is_torch_available():
    import torch

    from transformers import UnivNetGan


class UnivNetGanTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        in_channels=8,
        hidden_channels=8,
        num_mel_bins=20,
        kernel_predictor_hidden_channels=8,
        seed=0,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_mel_bins = num_mel_bins
        self.kernel_predictor_hidden_channels = kernel_predictor_hidden_channels
        self.seed = seed
        self.is_training = is_training

    def prepare_noise_sequence(self):
        generator = torch.manual_seed(self.seed)
        noise_shape = (self.seq_length, self.in_channels)
        # Create noise on CPU for reproducibility
        noise_sequence = torch.randn(noise_shape, generator=generator, dtype=torch.float)
        return noise_sequence

    def prepare_config_and_inputs(self):
        spectrogram = floats_tensor([self.seq_length, self.num_mel_bins], scale=1.0)
        noise_sequence = self.prepare_noise_sequence()
        noise_sequence = noise_sequence.to(spectrogram.device)
        config = self.get_config()
        return config, spectrogram, noise_sequence

    def get_config(self):
        return UnivNetGanConfig(
            model_in_channels=self.in_channels,
            model_hidden_channels=self.hidden_channels,
            num_mel_channels=self.num_mel_bins,
            kernel_predictor_hidden_channels=self.kernel_predictor_hidden_channels,
        )

    def create_and_check_model(self, config, spectrogram, noise_sequence):
        model = UnivNetGan(config=config).to(torch_device).eval()
        result = model(spectrogram, noise_sequence)
        self.parent.assertEqual(result.shape, (self.seq_length * 256,))

    def prepare_config_and_inputs_for_common(self):
        config, spectrogram, noise_sequence = self.prepare_config_and_inputs()
        inputs_dict = {"spectrogram": spectrogram, "noise_sequence": noise_sequence}
        return config, inputs_dict


@require_torch
class UnivNetGanTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (UnivNetGan,) if is_torch_available() else ()
    test_torchscript = False
    test_pruning = False
    test_resize_embeddings = False
    test_resize_position_embeddings = False
    test_head_masking = False
    test_mismatched_shapes = False
    test_missing_keys = False
    test_model_parallel = False
    is_encoder_decoder = False
    has_attentions = False

    input_name = "spectrogram"

    def setUp(self):
        self.model_tester = UnivNetGanTester(self)
        self.config_tester = ConfigTester(self, config_class=UnivNetGanConfig)

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_from_and_save_pretrained_subfolder()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "spectrogram",
            ]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    @unittest.skip(reason="UnivNetGan does not output hidden_states.")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="UnivNetGan.forward does not accept an inputs_embeds argument.")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="UnivNetGan does not use input embeddings and thus has no get_input_embeddings method.")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="UnivNetGan does not support all arguments tested, such as output_hidden_states.")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="UnivNetGan does not output hidden_states.")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    def test_batched_inputs_outputs(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            batched_spectrogram = inputs["spectrogram"].unsqueeze(0).repeat(2, 1, 1)
            batched_noise_sequence = inputs["noise_sequence"].unsqueeze(0).repeat(2, 1, 1)
            with torch.no_grad():
                batched_outputs = model(
                    batched_spectrogram.to(torch_device),
                    batched_noise_sequence.to(torch_device),
                )

            self.assertEqual(
                batched_spectrogram.shape[0],
                batched_outputs.shape[0],
                msg="Got different batch dims for input and output",
            )

    def test_unbatched_inputs_outputs(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(inputs["spectrogram"].to(torch_device), inputs["noise_sequence"].to(torch_device))
            self.assertTrue(outputs.dim() == 1, msg="Got un-batched inputs but batched output")

    def test_unbatched_batched_outputs_consistency(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            unbatched_spectrogram = inputs["spectrogram"].detach().clone()
            unbatched_noise_sequence = inputs["noise_sequence"].detach().clone()
            batched_spectrogram = inputs["spectrogram"].unsqueeze(0)
            batched_noise_sequence = inputs["noise_sequence"].unsqueeze(0)

            with torch.no_grad():
                unbatched_outputs = model(
                    unbatched_spectrogram.to(torch_device),
                    unbatched_noise_sequence.to(torch_device),
                )
                print(f"Unbatched outputs shape: {unbatched_outputs.shape}")

                batched_outputs = model(
                    batched_spectrogram.to(torch_device),
                    batched_noise_sequence.to(torch_device),
                )
                print(f"Batched outputs shape: {batched_outputs.shape}")

            self.assertTrue(
                torch.allclose(unbatched_outputs, batched_outputs),
                msg="Got different batch dims for input and output",
            )


@require_torch_gpu
@slow
class UnivNetGanIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def _load_datasamples(self, num_samples, sampling_rate=24000):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return [x["array"] for x in speech_samples], [x["sampling_rate"] for x in speech_samples]

    def get_inputs(self, device, num_samples: int = 3, noise_length: int = 10, seed: int = 0):
        generator = torch.manual_seed(seed)
        # Note: hardcode model_in_channels -> 64
        if num_samples == 1:
            noise_sequence_shape = (64, noise_length)
        else:
            noise_sequence_shape = (num_samples, 64, noise_length)
        # Explicity generate noise_sequence on CPU for consistency.
        noise_sequence = torch.randn(noise_sequence_shape, generator=generator, dtype=torch.float32, device="cpu")
        # Put noise_sequence on the desired device.
        noise_sequence = noise_sequence.to(device)

        # Note: hardcode num_mel_channels -> 100
        if num_samples == 1:
            spectrogram_shape = [100, noise_length]
        else:
            spectrogram_shape = [num_samples, 100, noise_length]
        spectrogram = floats_tensor(spectrogram_shape, scale=1.0, rng=random.Random(seed))
        # Note: spectrogram should already be on torch_device

        # Permute to match diffusers implementation
        if num_samples == 1:
            noise_sequence = noise_sequence.transpose(1, 0)
            spectrogram = spectrogram.transpose(1, 0)
        else:
            noise_sequence = noise_sequence.transpose(2, 1)
            spectrogram = spectrogram.transpose(2, 1)

        inputs = {
            "spectrogram": spectrogram,
            "noise_sequence": noise_sequence,
            "generator": generator,
        }

        return inputs

    @torch.no_grad()
    def test_model_inference_batched(self):
        # Load sample checkpoint from Tortoise TTS
        model = UnivNetGan.from_pretrained("dg845/univnet-dev")
        model.to(torch_device)

        # Get batched noise and spectrogram inputs.
        input_speech = self.get_inputs(torch_device, num_samples=3)

        waveform = model(**input_speech)
        waveform_slice = waveform[-1, -9:].detach().cpu().flatten().numpy()

        expected_slice = np.array([-0.3408, -0.6045, -0.5052, 0.1160, -0.1556, -0.0405, -0.3024, -0.5290, -0.5019])

        self.assertTrue(np.allclose(waveform_slice, expected_slice, atol=5e-4))

    @torch.no_grad()
    def test_model_inference_unbatched(self):
        # Load sample checkpoint from Tortoise TTS
        model = UnivNetGan.from_pretrained("dg845/univnet-dev")
        model.to(torch_device)

        # Get unbatched noise and spectrogram inputs.
        input_speech = self.get_inputs(torch_device, num_samples=1)

        waveform = model(**input_speech)
        waveform_slice = waveform[-9:].detach().cpu().flatten().numpy()

        expected_slice = np.array([-0.3276, -0.5504, -0.3484, 0.3574, -0.0373, -0.1826, -0.4880, -0.6431, -0.5162])

        self.assertTrue(np.allclose(waveform_slice, expected_slice, atol=5e-4))

    @torch.no_grad()
    def test_integration(self):
        feature_extractor = UnivNetFeatureExtractor.from_pretrained("dg845/univnet-dev")
        model = UnivNetGan.from_pretrained("dg845/univnet-dev")
        model.to(torch_device)

        audio, sr = self._load_datasamples(1, sampling_rate=feature_extractor.sampling_rate)

        input_features = feature_extractor(audio, sampling_rate=sr[0], return_tensors="pt").input_features
        input_features = input_features.to(device=torch_device, dtype=torch.float)

        input_speech = self.get_inputs(torch_device, num_samples=1, noise_length=input_features.shape[1])
        input_speech["spectrogram"] = input_features

        waveform = model(**input_speech)
        waveform_slice = waveform[-9:].detach().cpu().flatten().numpy()

        expected_slice = np.array(
            [
                -4.3934e-04,
                -1.8203e-04,
                -3.3033e-04,
                -3.8716e-04,
                -1.6125e-04,
                3.5389e-06,
                -3.3149e-04,
                -3.7613e-04,
                -2.3331e-04,
            ]
        )

        self.assertTrue(np.allclose(waveform_slice, expected_slice, atol=5e-6))
