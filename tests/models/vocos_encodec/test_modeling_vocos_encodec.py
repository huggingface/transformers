# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Vocos model."""

import copy
import inspect
import json
import tempfile
import unittest

from transformers.testing_utils import (
    require_torch,
    require_torch_gpu,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    torch_device,
)
from transformers.utils import is_datasets_available, is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor


if is_datasets_available():
    from datasets import Audio, load_dataset

if is_torch_available():
    import torch

    from transformers import VocosEncodecModel, VocosEncodecProcessor


from transformers import VocosEncodecConfig


class VocosEncodecModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 2
        self.codebook_dim = 128
        self.hidden_size = 16
        self.intermediate_size = 32
        self.num_layers = 2
        self.kernel_size = 3
        self.padding = 1
        self.layer_scale_init_value = 0.1
        self.use_adaptive_norm = False
        self.num_bandwidths = 1
        self.layer_norm_eps = 1e-6
        self.n_fft = 16
        self.hop_length = 8
        self.istft_padding = "same"
        self.seq_length = 10

    def get_config(self):
        return VocosEncodecConfig(
            codebook_dim=self.codebook_dim,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_layers=self.num_layers,
            kernel_size=self.kernel_size,
            padding=self.padding,
            layer_scale_init_value=self.layer_scale_init_value,
            use_adaptive_norm=self.use_adaptive_norm,
            num_bandwidths=self.num_bandwidths,
            layer_norm_eps=self.layer_norm_eps,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            istft_padding=self.istft_padding,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_values = floats_tensor([self.batch_size, self.codebook_dim, self.seq_length])
        bandwidth = torch.tensor(1.5)
        return config, input_values, bandwidth

    def prepare_config_and_inputs_for_common(self):
        config, input_features, bandwidth = self.prepare_config_and_inputs()
        return config, {"input_features": input_features, "bandwidth": bandwidth}

    def create_and_check_model(self, config, features, bandwidth):
        model = VocosEncodecModel(config=config).to(torch_device).eval()
        with torch.no_grad():
            output = model(features.to(torch_device), bandwidth=bandwidth.to(torch_device))

        if config.istft_padding == "center":
            # when padding is `center`,  output is computed using PyTorch's ISTFT
            expected_len = (self.seq_length - 1) * config.hop_length

        elif config.istft_padding == "same":
            # when padding is `same`, output is computed using custom ISTFT implementation in `custom_istft`
            pad = (config.n_fft - config.hop_length) // 2
            expected_len = (self.seq_length - 1) * config.hop_length + config.n_fft - 2 * pad

        self.parent.assertEqual(output.audio.shape, (self.batch_size, expected_len))


@require_torch
class VocosEncodecModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (VocosEncodecModel,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_torchscript = False
    test_resize_embeddings = False
    test_attention_outputs = False
    has_attentions = False
    test_missing_keys = False
    test_can_init_all_missing_weights = False

    def setUp(self):
        self.model_tester = VocosEncodecModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=VocosEncodecConfig, common_properties=[], has_text_modality=False
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = VocosEncodecModel(config)
        signature = inspect.signature(model.forward)
        arg_names = list(signature.parameters.keys())
        self.assertListEqual(arg_names, ["input_features", "attention_mask", "bandwidth", "kwargs"])

    @unittest.skip(
        reason="The VocosEncodecModel is not transformers based, thus it does not have the usual `hidden_states` logic"
    )
    def test_save_load(self):
        pass

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = [
                    "embed.weight",
                    "dwconv.weight",
                    "pwconv1.weight",
                    "pwconv2.weight",
                    "out.weight",
                ]
                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
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

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

    @unittest.skip(
        reason="The VocosEncodecModel is not transformers based, thus it does not have the usual `hidden_states` logic"
    )
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="We cannot configure to output a smaller model.")
    def test_model_is_small(self):
        pass

    @unittest.skip(reason="The VocosEncodecModel does not have `inputs_embeds` logics")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="VocosEncodecModel does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="VocosEncodecModel does not output any loss term in the forward pass")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="VocosEncodecModel does not output any loss term in the forward pass")
    def test_training(self):
        pass

    @unittest.skip(reason="VocosEncodecModel does not output any loss term in the forward pass")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="VocosEncodecModel does not output any loss term in the forward pass")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="VocosEncodecModel does not output any loss term in the forward pass")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="VocosEncodecModel has no attention layers")
    def test_torchscript_output_attentions(self):
        pass

    @unittest.skip("VocosEncodecModel cannot be tested with meta device")
    def test_can_load_with_meta_device_context_manager(self):
        pass

    @unittest.skip(reason="The VocosEncodecModel does not have `hidden_states`")
    def test_torchscript_output_hidden_state(self):
        pass

    def test_save_load_strict(self):
        config, _, _ = self.model_tester.prepare_config_and_inputs()
        model = VocosEncodecModel(config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            _, info = VocosEncodecModel.from_pretrained(tmpdirname, output_loading_info=True)
        self.assertEqual(info["missing_keys"], set())

    # override because `bandwidth` is passed as a float is not batch dependent.
    def test_batching_equivalence(self, atol=1e-5, rtol=1e-5):
        def recursive_check(batched_object, single_row_object, model_name, key):
            if isinstance(batched_object, (list, tuple)):
                for batched_object_value, single_row_object_value in zip(batched_object, single_row_object):
                    recursive_check(batched_object_value, single_row_object_value, model_name, key)
            elif isinstance(batched_object, dict):
                for batched_object_value, single_row_object_value in zip(
                    batched_object.values(), single_row_object.values()
                ):
                    recursive_check(batched_object_value, single_row_object_value, model_name, key)
            # do not compare returned loss (0-dim tensor) / codebook ids (int) / caching objects
            elif batched_object is None or not isinstance(batched_object, torch.Tensor):
                return
            elif batched_object.dim() == 0:
                return
            # do not compare int or bool outputs as they are mostly computed with max/argmax/topk methods which are
            # very sensitive to the inputs (e.g. tiny differences may give totally different results)
            elif not torch.is_floating_point(batched_object):
                return
            else:
                # indexing the first element does not always work
                # e.g. models that output similarity scores of size (N, M) would need to index [0, 0]
                slice_ids = [slice(0, index) for index in single_row_object.shape]
                batched_row = batched_object[slice_ids]
                self.assertFalse(
                    torch.isnan(batched_row).any(), f"Batched output has `nan` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isinf(batched_row).any(), f"Batched output has `inf` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isnan(single_row_object).any(), f"Single row output has `nan` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isinf(single_row_object).any(), f"Single row output has `inf` in {model_name} for key={key}"
                )
                try:
                    torch.testing.assert_close(batched_row, single_row_object, atol=atol, rtol=rtol)
                except AssertionError as e:
                    msg = f"Batched and Single row outputs are not equal in {model_name} for key={key}.\n\n"
                    msg += str(e)
                    raise AssertionError(msg)

        config, batched_input = self.model_tester.prepare_config_and_inputs_for_common()
        set_config_for_less_flaky_test(config)

        for model_class in self.all_model_classes:
            config.output_hidden_states = True

            model_name = model_class.__name__
            if hasattr(self.model_tester, "prepare_config_and_inputs_for_model_class"):
                config, batched_input = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
            batched_input_prepared = self._prepare_for_class(batched_input, model_class)
            model = model_class(copy.deepcopy(config)).to(torch_device).eval()
            set_model_for_less_flaky_test(model)

            batch_size = self.model_tester.batch_size
            single_row_input = {}
            for key, value in batched_input_prepared.items():
                if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] % batch_size == 0:
                    # e.g. musicgen has inputs of size (bs*codebooks). in most cases value.shape[0] == batch_size
                    single_batch_shape = value.shape[0] // batch_size
                    single_row_input[key] = value[:single_batch_shape]
                else:
                    single_row_input[key] = value

            with torch.no_grad():
                model_batched_output = model(**batched_input_prepared)
                model_row_output = model(**single_row_input)

            if isinstance(model_batched_output, torch.Tensor):
                model_batched_output = {"model_output": model_batched_output}
                model_row_output = {"model_output": model_row_output}

            for key in model_batched_output:
                # DETR starts from zero-init queries to decoder, leading to cos_similarity = `nan`
                if hasattr(self, "zero_init_hidden_state") and "decoder_hidden_states" in key:
                    model_batched_output[key] = model_batched_output[key][1:]
                    model_row_output[key] = model_row_output[key][1:]
                recursive_check(model_batched_output[key], model_row_output[key], model_name, key)


@require_torch
class VocosEncodecModelIntegrationTest(unittest.TestCase):
    """
    See code for reproducing expected outputs: https://gist.github.com/Manalelaidouni/340704dfef4d0aaec5a515e1af543b9c
    """

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.cast_column("audio", Audio(sampling_rate=24000))
        speech_samples = ds.sort("id")[:num_samples]["audio"]
        return [x["array"] for x in speech_samples]

    def setUp(self):
        with open("tests/fixtures/vocos/vocos_encodec_integration.json", "r") as f:
            self.encodec_expected = json.load(f)
        with open("tests/fixtures/vocos/vocos_encodec_batch_integration.json", "r") as f:
            self.encodec_batch_expected = json.load(f)

    @require_torch_gpu
    def test_inference(self):
        hf_repo_id = "Manel/vocos-encodec-24khz"
        model = VocosEncodecModel.from_pretrained(hf_repo_id).to(torch_device).eval()
        processor = VocosEncodecProcessor.from_pretrained(hf_repo_id)

        audio_np = self._load_datasamples(1)[0]
        audio = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0).to(torch_device)

        for entry in self.encodec_expected:
            # resconstruct audio from audio input
            inputs = processor(audio=audio, bandwidth=entry["bandwidth"], return_tensors="pt").to(torch_device)
            with torch.no_grad():
                output_from_audio = model(**inputs).audio

            EXPECTED_AUDIO = torch.tensor(entry["reconstructed_from_audio"], dtype=torch.float32).to(torch_device)

            torch.testing.assert_close(
                output_from_audio.squeeze(0)[: EXPECTED_AUDIO.shape[0]],
                EXPECTED_AUDIO,
                rtol=1e-5,
                atol=1e-5,
            )

            # resconstructing audio from quantized codes
            codes = torch.tensor(entry["input_codes"], dtype=torch.long)
            inputs = processor(codes=codes, bandwidth=entry["bandwidth"], return_tensors="pt")

            with torch.no_grad():
                output_from_codes = model(**inputs.to(torch_device)).audio

            EXPECTED_AUDIO_FROM_CODES = torch.tensor(entry["reconstructed_from_codes"], dtype=torch.float32).to(
                torch_device
            )

            torch.testing.assert_close(
                output_from_codes.squeeze(0)[: EXPECTED_AUDIO_FROM_CODES.shape[0]],
                EXPECTED_AUDIO_FROM_CODES,
                rtol=1e-5,
                atol=1e-5,
            )

    @require_torch_gpu
    def test_batch(self):
        repo_id = "Manel/vocos-encodec-24khz"
        processor = VocosEncodecProcessor.from_pretrained(repo_id)
        model = VocosEncodecModel.from_pretrained(repo_id).to(torch_device).eval()

        # audios reconstruction from batch of audios
        audios = self._load_datasamples(3)

        for entry in self.encodec_batch_expected:
            if "reconstructed_from_audio" not in entry:
                continue
            bandwidth = entry["bandwidth"]
            inputs = processor(audio=audios, bandwidth=bandwidth, return_tensors="pt").to(torch_device)
            hf_batch = model(**inputs).audio

            for idx, saved in enumerate(entry["reconstructed_from_audio"]):
                expected = torch.tensor(saved, dtype=torch.float32, device=torch_device)
                torch.testing.assert_close(
                    hf_batch[idx, : expected.shape[0]],
                    expected,
                    rtol=1e-4,
                    atol=1e-4,
                )

        # reconstruction from batch of quantized codes
        for entry in self.encodec_batch_expected:
            if "audio_codes" not in entry:
                continue
            codes = torch.tensor(entry["audio_codes"], dtype=torch.long, device=torch_device)
            bandwidth = entry["bandwidth"]
            inputs = processor(codes=codes, bandwidth=bandwidth, return_tensors="pt").to(torch_device)
            hf_batch = model(**inputs).audio

            for idx, saved in enumerate(entry["reconstructed_from_codes"]):
                expected = torch.tensor(saved, dtype=torch.float32, device=torch_device)
                torch.testing.assert_close(
                    hf_batch[idx, : expected.shape[0]],
                    expected,
                    rtol=1e-4,
                    atol=1e-4,
                )
