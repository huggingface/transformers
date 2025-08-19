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
"""Testing suite for the PyTorch Xcodec model."""

import inspect
import math
import os
import tempfile
import unittest

import numpy as np
from datasets import Audio, load_dataset
from parameterized import parameterized
from pytest import mark

from tests.test_configuration_common import ConfigTester
from tests.test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from transformers import AutoFeatureExtractor, XcodecConfig
from transformers.testing_utils import (
    is_flaky,
    is_torch_available,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import XcodecModel


@require_torch
class XcodecModelTester:
    def __init__(
        self,
        parent,
        batch_size=4,
        num_channels=1,
        sample_rate=16000,
        codebook_size=1024,
        num_samples=400,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.codebook_size = codebook_size
        self.is_training = is_training
        self.num_samples = num_samples

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.num_channels, self.num_samples], scale=1.0)
        config = self.get_config()
        inputs_dict = {"input_values": input_values}
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def prepare_config_and_inputs_for_model_class(self, model_class):
        config, inputs_dict = self.prepare_config_and_inputs()
        codes_length = math.ceil(self.num_samples / config.hop_length)
        inputs_dict["audio_codes"] = ids_tensor(
            [self.batch_size, config.num_quantizers, codes_length], config.codebook_size
        )

        return config, inputs_dict

    def get_config(self):
        return XcodecConfig(
            sample_rate=self.sample_rate,
            audio_channels=self.num_channels,
            codebook_size=self.codebook_size,
        )

    def create_and_check_model_forward(self, config, inputs_dict):
        model = XcodecModel(config=config).to(torch_device).eval()
        input_values = inputs_dict["input_values"]
        result = model(input_values)
        self.parent.assertEqual(result.audio_values.shape, (self.batch_size, self.num_channels, self.num_samples))


@require_torch
class XcodecModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (XcodecModel,) if is_torch_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False
    test_torchscript = False
    test_can_init_all_missing_weights = False

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        # model does not support returning hidden states
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if "output_attentions" in inputs_dict:
            inputs_dict.pop("output_attentions")
        if "output_hidden_states" in inputs_dict:
            inputs_dict.pop("output_hidden_states")
        return inputs_dict

    def setUp(self):
        self.model_tester = XcodecModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=XcodecConfig, common_properties=[], has_text_modality=False
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["input_values", "audio_codes", "bandwidth", "return_dict"]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    def test_gradient_checkpointing_backward_compatibility(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue

            config.text_encoder.gradient_checkpointing = True
            config.audio_encoder.gradient_checkpointing = True
            config.decoder.gradient_checkpointing = True
            model = model_class(config)
            self.assertTrue(model.is_gradient_checkpointing)

    @unittest.skip("XcodecModel cannot be tested with meta device")
    def test_can_load_with_meta_device_context_manager(self):
        pass

    @unittest.skip(reason="We cannot configure to output a smaller model.")
    def test_model_is_small(self):
        pass

    @unittest.skip(reason="The XcodecModel does not have `inputs_embeds` logics")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="The XcodecModel does not have `inputs_embeds` logics")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="The XcodecModel does not have the usual `attention` logic")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="The XcodecModel does not have the usual `attention` logic")
    def test_torchscript_output_attentions(self):
        pass

    @unittest.skip(reason="The XcodecModel does not have the usual `hidden_states` logic")
    def test_torchscript_output_hidden_state(self):
        pass

    # Copied from transformers.tests.encodec.test_modeling_encodec.XcodecModelTest._create_and_check_torchscript
    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            self.skipTest(reason="test_torchscript is set to False")

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        configs_no_init.return_dict = False
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class)

            main_input_name = model_class.main_input_name

            try:
                main_input = inputs[main_input_name]
                model(main_input)
                traced_model = torch.jit.trace(model, main_input)
            except RuntimeError:
                self.fail("Couldn't trace module.")

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                try:
                    torch.jit.save(traced_model, pt_file_name)
                except Exception:
                    self.fail("Couldn't save module.")

                try:
                    loaded_model = torch.jit.load(pt_file_name)
                except Exception:
                    self.fail("Couldn't load module.")

            model.to(torch_device)
            model.eval()

            loaded_model.to(torch_device)
            loaded_model.eval()

            model_state_dict = model.state_dict()
            loaded_model_state_dict = loaded_model.state_dict()

            non_persistent_buffers = {}
            for key in loaded_model_state_dict.keys():
                if key not in model_state_dict.keys():
                    non_persistent_buffers[key] = loaded_model_state_dict[key]

            loaded_model_state_dict = {
                key: value for key, value in loaded_model_state_dict.items() if key not in non_persistent_buffers
            }

            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

            model_buffers = list(model.buffers())
            for non_persistent_buffer in non_persistent_buffers.values():
                found_buffer = False
                for i, model_buffer in enumerate(model_buffers):
                    if torch.equal(non_persistent_buffer, model_buffer):
                        found_buffer = True
                        break

                self.assertTrue(found_buffer)
                model_buffers.pop(i)

            model_buffers = list(model.buffers())
            for non_persistent_buffer in non_persistent_buffers.values():
                found_buffer = False
                for i, model_buffer in enumerate(model_buffers):
                    if torch.equal(non_persistent_buffer, model_buffer):
                        found_buffer = True
                        break

                self.assertTrue(found_buffer)
                model_buffers.pop(i)

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                if layer_name in loaded_model_state_dict:
                    p2 = loaded_model_state_dict[layer_name]
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False

            self.assertTrue(models_equal)

            # Avoid memory leak. Without this, each call increase RAM usage by ~20MB.
            # (Even with this call, there are still memory leak by ~0.04MB)
            self.clear_torch_jit_class_registry()

    @unittest.skip(reason="The XcodecModel does not have the usual `attention` logic")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="The XcodecModel does not have the usual `hidden_states` logic")
    def test_hidden_states_output(self):
        pass

    # Copied from transformers.tests.encodec.test_modeling_encodecEncodecModelTest.test_determinism
    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_determinism(first, second):
            # outputs are not tensors but list (since each sequence don't have the same frame_length)
            out_1 = first.cpu().numpy()
            out_2 = second.cpu().numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]
                second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)

    # Copied from transformers.tests.encodec.test_modeling_encodecEncodecModelTest.test_model_outputs_equivalence
    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs)

                self.assertTrue(isinstance(tuple_output, tuple))
                self.assertTrue(isinstance(dict_output, dict))

                for tuple_value, dict_value in zip(tuple_output, dict_output.values()):
                    self.assertTrue(
                        torch.allclose(
                            set_nan_tensor_to_zero(tuple_value), set_nan_tensor_to_zero(dict_value), atol=1e-5
                        ),
                        msg=(
                            "Tuple and dict output are not equal. Difference:"
                            f" {torch.max(torch.abs(tuple_value - dict_value))}. Tuple has `nan`:"
                            f" {torch.isnan(tuple_value).any()} and `inf`: {torch.isinf(tuple_value)}. Dict has"
                            f" `nan`: {torch.isnan(dict_value).any()} and `inf`: {torch.isinf(dict_value)}."
                        ),
                    )

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                # skipping the parametrizations original0 tensor
                if name == "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0":
                    continue

                uniform_init_parms = ["conv"]

                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of {model_class.__name__} seems not properly initialized",
                        )

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    @is_flaky()
    def test_flash_attn_2_inference_equivalence(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.bfloat16)
                model.to(torch_device)

                dummy_input = inputs_dict[model.main_input_name][:1]
                if dummy_input.dtype in [torch.float32, torch.float16]:
                    dummy_input = dummy_input.to(torch.bfloat16)

                outputs = model(dummy_input)
                outputs_fa = model_fa(dummy_input)

                logits = outputs[1]
                logits_fa = outputs_fa[1]

                assert torch.allclose(logits_fa, logits, atol=4e-2, rtol=4e-2)

    @unittest.skip(reason="The XcodecModel does not support right padding")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @unittest.skip(reason="The XcodecModel does not have support dynamic compile yet")
    def test_sdpa_can_compile_dynamic(self):
        pass


# Copied from transformers.tests.encodec.test_modeling_encodec.normalize
def normalize(arr):
    norm = np.linalg.norm(arr)
    normalized_arr = arr / norm
    return normalized_arr


# Copied from transformers.tests.encodec.test_modeling_encodec.compute_rmse
def compute_rmse(arr1, arr2):
    arr1_np = arr1.cpu().numpy().squeeze()
    arr2_np = arr2.cpu().numpy().squeeze()
    max_length = min(arr1.shape[-1], arr2.shape[-1])
    arr1_np = arr1_np[..., :max_length]
    arr2_np = arr2_np[..., :max_length]
    arr1_normalized = normalize(arr1_np)
    arr2_normalized = normalize(arr2_np)
    return np.sqrt(((arr1_normalized - arr2_normalized) ** 2).mean())


"""
Integration tests for XCodec

Code for reproducing expected outputs can be found here:
https://gist.github.com/ebezzam/cdaf8c223e59e7677b2ea6bc2dc8230b

"""

# fmt: off
EXPECTED_OUTPUTS = {
    "xcodec-hubert-librispeech": {
        0.5: {
            "code": torch.tensor([[[ 590,  590,  306,  306,  590, 1006,  826,  916,  590,  826,  826,
                844,  844,  746,  392]]]),
            "decoded": torch.tensor([[[-6.4479e-04,  1.3059e-04,  6.7415e-04,  3.4384e-04, -1.0848e-04,
                1.4183e-04,  6.2488e-05, -3.3248e-04, -5.2672e-04, -1.0176e-03,
                -1.1252e-03, -6.4466e-04, -4.7831e-04, -2.3266e-04, -1.2719e-04,
                -1.8236e-05, -5.3274e-05, -1.7652e-04, -4.2407e-04, -2.9935e-04,
                -2.6427e-04, -3.1354e-04, -2.6051e-04, -2.6807e-04, -2.2152e-06,
                2.0787e-04,  1.7594e-04, -1.6461e-04, -3.9851e-04, -2.5608e-04,
                -1.3758e-04, -6.3655e-05, -2.0918e-04, -2.4738e-04, -1.4894e-04,
                1.3322e-04,  4.0269e-04,  2.6470e-04,  7.3273e-05, -2.5838e-04,
                -7.7150e-04, -1.2102e-03, -9.5020e-04, -1.7247e-04,  3.3870e-04,
                3.1896e-04,  2.1004e-04,  1.0997e-04,  2.4434e-04,  3.2173e-05]]]),
            "codec_error": 0.005472081247717142
        },
        1.0: {
            "code": torch.tensor([[[ 590,  590,  306,  306,  590, 1006,  826,  916,  590,  826,  826,
                844,  844,  746,  392],
                [ 748,  375,  156,  327,  327,  156,  448,  448,  327,  346, 1023,
                346,  346, 1023, 1023]]]),
            "decoded": torch.tensor([[[-2.2301e-04,  1.8041e-04,  3.4872e-04,  1.0567e-04, -2.2225e-04,
                -1.2607e-04, -7.7315e-05, -1.2137e-04, -1.7624e-04, -7.7369e-04,
                -1.0080e-03, -4.6426e-04, -7.9643e-05, -3.2846e-05, -1.0335e-05,
                8.4996e-05, -1.0046e-04, -2.9116e-04, -4.4626e-04, -1.4261e-04,
                -3.6265e-04, -5.8392e-04, -2.4049e-04,  1.8860e-05,  5.6159e-06,
                1.7515e-04,  4.1015e-04,  1.7480e-04, -3.0472e-04, -3.2516e-04,
                -2.1712e-04, -2.5662e-04, -2.2619e-04,  8.0582e-05,  2.9059e-04,
                7.3141e-04,  9.4131e-04,  5.3816e-04, -7.7466e-05, -5.0398e-04,
                -8.5594e-04, -1.0778e-03, -5.7206e-04,  4.6507e-04,  9.3888e-04,
                7.7739e-04,  5.6054e-04,  3.0491e-04,  1.0391e-04, -4.8576e-04]]]),
            "codec_error": 0.005680468864738941
        },
        1.5: {
            "code": torch.tensor([[[ 590,  590,  306,  306,  590, 1006,  826,  916,  590,  826,  826,
                844,  844,  746,  392],
                [ 748,  375,  156,  327,  327,  156,  448,  448,  327,  346, 1023,
                346,  346, 1023, 1023],
                [ 351,  835,  351,  992,  992,  817,  127,  832,  855,  177,  817,
                177,  177,   97,  472]]]),
            "decoded": torch.tensor([[[-8.1314e-05,  3.7229e-04,  5.2648e-04,  2.4720e-04, -1.2782e-04,
                -6.0210e-05,  2.0436e-06, -9.9719e-06, -5.3663e-05, -7.1501e-04,
                -9.9129e-04, -3.8447e-04,  5.8557e-05,  1.2949e-04,  1.7357e-04,
                2.4178e-04, -4.8278e-05, -3.2059e-04, -4.9153e-04, -1.3682e-04,
                -2.8308e-04, -3.9494e-04,  5.2281e-05,  3.7370e-04,  3.4041e-04,
                4.9286e-04,  6.6506e-04,  2.3967e-04, -4.3357e-04, -5.3537e-04,
                -3.8190e-04, -3.2748e-04, -1.9068e-04,  2.7354e-04,  6.7566e-04,
                1.2528e-03,  1.3770e-03,  7.2869e-04, -1.9324e-04, -9.0314e-04,
                -1.4543e-03, -1.7141e-03, -1.0317e-03,  3.2189e-04,  1.0539e-03,
                1.0152e-03,  8.7432e-04,  6.2338e-04,  2.9369e-04, -4.9140e-04]]]),
            "codec_error": 0.005860992707312107
        },
        2.0: {
            "code": torch.tensor([[[ 590,  590,  306,  306,  590, 1006,  826,  916,  590,  826,  826,
                844,  844,  746,  392],
                [ 748,  375,  156,  327,  327,  156,  448,  448,  327,  346, 1023,
                346,  346, 1023, 1023],
                [ 351,  835,  351,  992,  992,  817,  127,  832,  855,  177,  817,
                177,  177,   97,  472],
                [ 156, 1008,  546,  266,  856,  116,  173,  531,  894,  938,   87,
                531,  745,  747,  747]]]),
            "decoded": torch.tensor([[[ 6.8767e-05,  5.1738e-04,  6.6016e-04,  4.0730e-04,  2.4633e-05,
                6.3171e-05,  1.1770e-04,  1.0633e-04,  6.6064e-05, -6.3992e-04,
                -8.8675e-04, -2.6030e-04,  1.5681e-04,  2.1466e-04,  2.6819e-04,
                3.9723e-04,  1.0386e-04, -1.1724e-04, -2.9744e-04,  6.5359e-05,
                -1.4087e-04, -2.6657e-04,  2.2346e-04,  5.5748e-04,  5.3625e-04,
                6.5261e-04,  7.9668e-04,  3.3301e-04, -4.4982e-04, -6.8057e-04,
                -5.9004e-04, -5.2670e-04, -3.6594e-04,  1.6948e-04,  6.8814e-04,
                1.3760e-03,  1.4863e-03,  8.1168e-04, -8.7532e-05, -8.3853e-04,
                -1.4866e-03, -1.7615e-03, -9.9449e-04,  4.1615e-04,  1.1418e-03,
                1.1300e-03,  1.0678e-03,  7.8835e-04,  4.4182e-04, -3.3208e-04]]]),
            "codec_error": 0.005977352149784565
        },
        4.0: {
            "code": None,
            # "code": torch.tensor([[[ 590,  590,  306,  306,  590, 1006,  826,  916,  590,  826,  826,
            #     844,  844,  746,  392],
            #     [ 748,  375,  156,  327,  327,  156,  448,  448,  327,  346, 1023,
            #     346,  346, 1023, 1023],
            #     [ 351,  835,  351,  992,  992,  817,  127,  832,  855,  177,  817,
            #     177,  177,   97,  472],
            #     [ 156, 1008,  546,  266,  856,  116,  173,  531,  894,  938,   87,
            #     531,  745,  747,  747],
            #     [ 901,  623,  523,  273,  654, 1013, 1013,  220,  886,  586,  370,
            #     861,  861,  628,  644],
            #     [ 977,  322,  977,  341,  606,  350,  532,   83,   83,  706,   83,
            #     159,  695,  514,  172],
            #     [ 360,  257, 1002,  532, 1015,  849,   37,  645,  527,  648,  851,
            #         84,  520,  271,   84],
            #     [ 759,  747,  877,  310,   20,  486,  445,  374,  453,  760,  194,
            #     194,  340,   88,  853]]]),
            "decoded": torch.tensor([[[ 4.9193e-05,  4.4999e-04,  5.1708e-04,  3.8594e-04,  1.2237e-04,
                6.0459e-05,  7.3040e-05,  2.3535e-04,  2.8712e-04, -4.9805e-04,
                -7.8073e-04, -1.9005e-04,  2.0040e-04,  2.2385e-04,  2.8799e-04,
                5.5226e-04,  3.2119e-04,  5.3692e-05, -2.6665e-04,  3.6196e-05,
                -3.0041e-04, -6.2983e-04, -1.7190e-04,  4.7767e-04,  7.1321e-04,
                7.8250e-04,  1.0266e-03,  7.2837e-04, -1.1806e-04, -6.0261e-04,
                -5.6294e-04, -3.4018e-04, -1.8106e-04,  2.3291e-04,  7.5652e-04,
                1.5197e-03,  1.6511e-03,  9.6738e-04,  1.3117e-04, -4.4446e-04,
                -1.0821e-03, -1.4284e-03, -6.3103e-04,  7.6366e-04,  1.3156e-03,
                1.2162e-03,  1.2933e-03,  1.0688e-03,  5.4352e-04, -3.2686e-04]]]),
            "codec_error": 0.00608655484393239
        },
    },
    "xcodec-hubert-general": {
        0.5: {
            "code": torch.tensor([[[935, 433, 126, 803, 850, 448, 917, 387, 387, 592, 592, 855, 917, 572,
                28]]]),
            "decoded": torch.tensor([[[-2.1018e-04,  4.9652e-04,  1.3865e-04, -2.0842e-04, -8.9698e-05,
                -2.7716e-04, -2.7933e-04,  5.5042e-05,  2.6330e-04,  1.7548e-04,
                3.8894e-04,  5.5395e-04,  1.4358e-04, -3.1661e-04, -5.8826e-04,
                -4.4183e-04, -1.4901e-04, -5.8116e-05,  4.0458e-05, -7.7961e-05,
                3.5779e-06,  1.7821e-04,  9.8606e-05,  1.0031e-04,  1.2427e-05,
                -1.0456e-04, -9.7690e-05, -5.8267e-05,  1.7144e-04,  7.1903e-06,
                -1.9244e-04,  5.3309e-05,  1.9154e-04,  2.9334e-05, -9.6718e-05,
                -3.8430e-04, -5.9647e-04, -5.4143e-04, -3.4540e-04, -1.2945e-04,
                -1.9762e-04, -1.1440e-04,  1.8424e-05,  5.5768e-05,  1.2666e-04,
                -9.9905e-06, -8.8416e-05, -7.7985e-05, -1.2243e-04, -9.8607e-05]]]),
            "codec_error": 0.0038952771574258804
        },
        1.0: {
            "code": torch.tensor([[[935, 433, 126, 803, 850, 448, 917, 387, 387, 592, 592, 855, 917, 572,
                28],
                [739, 882, 882,  49, 459,  64, 189, 459, 459, 459, 143, 551, 550, 760,
                808]]]),
            "decoded": torch.tensor([[[-2.0465e-04,  4.6923e-04,  1.1031e-04, -2.9965e-04, -2.1691e-04,
                -3.9182e-04, -3.8641e-04, -4.2736e-05,  2.2560e-04,  1.9318e-04,
                3.6341e-04,  4.7189e-04,  5.8798e-05, -3.5422e-04, -5.8790e-04,
                -4.7569e-04, -2.4429e-04, -8.3336e-05,  9.2228e-05, -2.7093e-05,
                2.5268e-05,  2.0193e-04,  1.0741e-04,  1.9980e-05, -1.2808e-04,
                -2.5810e-04, -2.4647e-04, -1.8511e-04,  6.0304e-05, -3.3718e-05,
                -1.4680e-04,  5.0458e-05,  3.8814e-05, -2.0871e-04, -2.8215e-04,
                -5.0108e-04, -7.1238e-04, -6.8506e-04, -5.0637e-04, -3.0805e-04,
                -3.5870e-04, -2.4597e-04, -1.2729e-04, -1.4050e-04, -9.5578e-05,
                -2.4283e-04, -3.3660e-04, -2.3544e-04, -1.1086e-04, -1.4352e-05]]]),
            "codec_error": 0.003511926392093301
        },
        1.5: {
            "code": torch.tensor([[[935, 433, 126, 803, 850, 448, 917, 387, 387, 592, 592, 855, 917, 572,
                28],
                [739, 882, 882,  49, 459,  64, 189, 459, 459, 459, 143, 551, 550, 760,
                808],
                [710, 536, 176, 531, 623, 626, 100, 833, 796, 311, 457, 382, 360, 176,
                410]]]),
            "decoded": torch.tensor([[[-1.3057e-04,  7.1128e-04,  3.3386e-04, -6.9387e-05,  1.1096e-04,
                -7.8544e-05, -5.8596e-05,  4.3661e-04,  7.9219e-04,  7.4244e-04,
                9.2735e-04,  1.0654e-03,  5.3728e-04, -4.1218e-05, -3.3019e-04,
                -1.2553e-04,  1.8500e-04,  4.0498e-04,  6.7218e-04,  5.5694e-04,
                5.7880e-04,  7.3124e-04,  5.4757e-04,  3.8789e-04,  2.4302e-04,
                1.4217e-04,  1.4891e-04,  2.6071e-04,  5.8686e-04,  4.2715e-04,
                1.9053e-04,  4.0223e-04,  4.0457e-04,  8.7184e-05,  8.3866e-07,
                -2.1423e-04, -4.3250e-04, -3.4649e-04, -9.3173e-05,  1.2368e-04,
                -5.1642e-06,  4.8767e-05,  1.0705e-04,  2.7181e-05,  1.1900e-04,
                4.6864e-05,  7.4937e-06,  1.6385e-04,  2.6516e-04,  2.7671e-04]]]),
            "codec_error": 0.0034364312887191772
        },
        2.0: {
            "code": torch.tensor([[[935, 433, 126, 803, 850, 448, 917, 387, 387, 592, 592, 855, 917, 572,
                28],
                [739, 882, 882,  49, 459,  64, 189, 459, 459, 459, 143, 551, 550, 760,
                808],
                [710, 536, 176, 531, 623, 626, 100, 833, 796, 311, 457, 382, 360, 176,
                410],
                [791, 515, 953, 596, 454, 753, 295, 454, 454, 115, 515, 816,  36,  75,
                226]]]),
            "decoded": torch.tensor([[[-1.4006e-04,  7.9890e-04,  4.6538e-04,  8.5670e-05,  2.6646e-04,
                5.3687e-05,  4.9512e-05,  6.0565e-04,  1.0515e-03,  1.1005e-03,
                1.3407e-03,  1.4766e-03,  8.7940e-04,  1.9451e-04, -1.7620e-04,
                1.8656e-05,  3.5232e-04,  6.2158e-04,  9.4826e-04,  8.7659e-04,
                9.1280e-04,  1.0658e-03,  8.6510e-04,  6.9313e-04,  5.2745e-04,
                4.0837e-04,  3.8283e-04,  4.5729e-04,  8.1393e-04,  7.0759e-04,
                4.9867e-04,  7.3805e-04,  7.4706e-04,  3.7274e-04,  2.0862e-04,
                -4.4480e-05, -2.6486e-04, -1.4659e-04,  1.4993e-04,  3.9470e-04,
                2.4374e-04,  2.5463e-04,  2.7235e-04,  1.7000e-04,  2.7363e-04,
                2.2305e-04,  1.8746e-04,  3.6476e-04,  4.7830e-04,  4.9256e-04]]]),
            "codec_error": 0.003246477572247386
        },
        4.0: {
            "code": None,
            # "code": torch.tensor([[[ 935,  433,  126,  803,  850,  448,  917,  387,  387,  592,  592,
            #     855,  917,  572,   28],
            #     [ 739,  882,  882,   49,  459,   64,  189,  459,  459,  459,  143,
            #     551,  550,  760,  808],
            #     [ 710,  536,  176,  531,  623,  626,  100,  833,  796,  311,  457,
            #     382,  360,  176,  410],
            #     [ 791,  515,  953,  596,  454,  753,  295,  454,  454,  115,  515,
            #     816,   36,   75,  226],
            #     [ 976,  816,  652,  544,  721,  572,  334, 1017,  201,  201,  160,
            #     744,  201,   10,  906],
            #     [ 653,  738,  320,  467,  233,  845,  768,  656,   11,   11,  618,
            #     383,   52,  327,  327],
            #     [ 411,  698,  872,  491,  210,   38,  191,  613,  819,  221,  958,
            #     450,  326,  981,  930],
            #     [ 954,  302,  491, 1004,    8,  334,  658,  838,  648,  280,  587,
            #     449,  106,  280,  992]]]),
            "decoded": torch.tensor([[[-0.0002,  0.0008,  0.0005,  0.0002,  0.0005,  0.0003,  0.0002,
                0.0007,  0.0012,  0.0012,  0.0014,  0.0017,  0.0011,  0.0005,
                0.0001,  0.0003,  0.0006,  0.0008,  0.0011,  0.0011,  0.0011,
                0.0014,  0.0011,  0.0009,  0.0008,  0.0006,  0.0004,  0.0005,
                0.0009,  0.0009,  0.0008,  0.0011,  0.0010,  0.0004,  0.0001,
                -0.0001, -0.0004, -0.0002,  0.0002,  0.0005,  0.0003,  0.0005,
                0.0006,  0.0004,  0.0004,  0.0003,  0.0001,  0.0004,  0.0006,
                0.0006]]]),
            "codec_error": 0.0029708717484027147
        },
    },
    "xcodec-hubert-general-balanced": {
        0.5: {
            "code": torch.tensor([[[361, 327, 361, 220, 296, 448, 794, 794, 220, 215, 215, 523, 794, 572,
          837]]]),
            "decoded": torch.tensor([[[-3.4494e-04,  4.8887e-04,  5.2613e-05, -3.1542e-04, -3.7401e-04,
          -6.6790e-04, -4.4689e-04,  1.4481e-04,  4.1871e-04,  2.4440e-04,
           5.5280e-04,  7.3242e-04,  1.9533e-04, -1.8689e-04, -3.9331e-04,
          -1.5876e-05,  3.0878e-04,  2.3550e-04,  2.6388e-04, -1.3994e-05,
           5.0710e-05,  2.5054e-04,  1.7873e-04,  4.5949e-04,  5.8807e-04,
           2.0285e-04, -1.7900e-04, -4.2233e-04, -5.6768e-04, -9.2936e-04,
          -1.0679e-03, -5.0414e-04,  4.1774e-05,  3.0837e-04,  2.2762e-04,
          -1.8441e-04, -3.5402e-04, -4.8435e-04, -5.0981e-04, -4.7943e-05,
           3.3328e-04,  6.6242e-04,  7.2671e-04,  5.3310e-04,  5.1023e-04,
           2.6526e-04, -1.1996e-04, -3.4912e-04, -3.5247e-04, -1.1564e-04]]]),
            "codec_error": 0.0033109495416283607,
            "codec_tol": 1e-4,
        },
        1.0: {
            "code": torch.tensor([[[361, 327, 361, 220, 296, 448, 794, 794, 220, 215, 215, 523, 794, 572,
                837],
                [558, 561,  17, 689, 341,  17,  17, 746, 995, 203, 203, 626,  44, 930,
                137]]]),
            "decoded": torch.tensor([[[ 3.0808e-04,  1.0459e-03,  7.5420e-04,  3.3998e-04,  3.4536e-04,
                9.4889e-05,  1.3063e-04,  6.0099e-04,  8.5338e-04,  6.9592e-04,
                1.0017e-03,  1.1683e-03,  7.2195e-04,  4.5067e-04,  3.8395e-04,
                6.0960e-04,  6.7787e-04,  6.2032e-04,  6.7532e-04,  4.1310e-04,
                4.9198e-04,  7.9509e-04,  7.7787e-04,  9.3789e-04,  1.0353e-03,
                6.2544e-04,  2.1485e-04,  1.2941e-04,  1.5907e-04, -7.8654e-05,
                -4.9406e-05,  5.3553e-04,  8.4295e-04,  9.2303e-04,  9.1781e-04,
                5.7923e-04,  3.1794e-04,  2.3509e-04,  2.5939e-04,  4.7500e-04,
                5.9297e-04,  8.2354e-04,  9.4886e-04,  9.4531e-04,  1.1052e-03,
                9.6752e-04,  5.9950e-04,  4.3947e-04,  4.5157e-04,  4.1881e-04]]]),
            "codec_error": 0.0031247916631400585,
            "codec_tol": 1e-4,
        },
        1.5: {
            "code": torch.tensor([[[ 361,  327,  361,  220,  296,  448,  794,  794,  220,  215,  215,
                523,  794,  572,  837],
                [ 558,  561,   17,  689,  341,   17,   17,  746,  995,  203,  203,
                626,   44,  930,  137],
                [ 551,  408,  315,  232,  209,  733,  935,   96,  644,  260,  204,
                1005,  360,  220,  566]]]),
            "decoded": torch.tensor([[[0.0005, 0.0013, 0.0010, 0.0006, 0.0005, 0.0002, 0.0002, 0.0008,
            0.0012, 0.0011, 0.0016, 0.0018, 0.0013, 0.0010, 0.0009, 0.0011,
            0.0010, 0.0009, 0.0010, 0.0007, 0.0009, 0.0014, 0.0015, 0.0017,
            0.0017, 0.0011, 0.0005, 0.0003, 0.0004, 0.0002, 0.0003, 0.0012,
            0.0016, 0.0016, 0.0016, 0.0012, 0.0008, 0.0006, 0.0006, 0.0009,
            0.0010, 0.0013, 0.0016, 0.0016, 0.0018, 0.0017, 0.0013, 0.0010,
            0.0010, 0.0009]]]),
            "codec_error": 0.0029652512166649103
        },
        2.0: {
            "code": torch.tensor([[[ 361,  327,  361,  220,  296,  448,  794,  794,  220,  215,  215,
                523,  794,  572,  837],
                [ 558,  561,   17,  689,  341,   17,   17,  746,  995,  203,  203,
                626,   44,  930,  137],
                [ 551,  408,  315,  232,  209,  733,  935,   96,  644,  260,  204,
                1005,  360,  220,  566],
                [  14,   14,  407,  656,  472,  407,  472,  365,  444,  521,  162,
                128,  575,  340,  407]]]),
            "decoded": torch.tensor([[[0.0005, 0.0014, 0.0011, 0.0007, 0.0007, 0.0003, 0.0004, 0.0008,
                0.0012, 0.0010, 0.0015, 0.0018, 0.0013, 0.0010, 0.0009, 0.0011,
                0.0011, 0.0010, 0.0011, 0.0008, 0.0009, 0.0014, 0.0013, 0.0015,
                0.0017, 0.0012, 0.0006, 0.0005, 0.0005, 0.0003, 0.0003, 0.0011,
                0.0015, 0.0015, 0.0016, 0.0013, 0.0010, 0.0008, 0.0008, 0.0009,
                0.0009, 0.0012, 0.0013, 0.0013, 0.0016, 0.0016, 0.0013, 0.0011,
                0.0011, 0.0009]]]),
            "codec_error": 0.002776096574962139
        },
        4.0: {
            "code": None,
            "decoded": None,
            "codec_error": 0.002473212545737624,
            "codec_tol": 1e-4,
        },
    },
    "xcodec-wavlm-mls": {
        0.5: {
            "code": None,
        #     "code": torch.tensor([[[837, 619, 704, 704, 704, 619, 977, 619, 704, 704, 704, 885, 437, 864,
        #   471]]]),
            "decoded": torch.tensor([[[-2.4717e-04, -7.3656e-04, -2.9662e-04, -1.4877e-04, -2.5510e-04,
          -8.9435e-06,  1.6782e-04,  1.9963e-04, -4.0840e-05, -3.5881e-04,
          -3.7214e-04, -3.1695e-04, -3.1708e-04, -3.3144e-04, -4.3814e-05,
           3.9877e-04,  5.6475e-04,  6.1846e-04,  4.4193e-04, -8.9463e-05,
          -4.1933e-04, -4.4411e-04, -2.3402e-04, -3.5529e-05,  1.4211e-04,
           2.4672e-04,  1.5604e-04,  3.1932e-05, -1.3763e-04, -3.5282e-04,
          -5.2977e-04, -6.5176e-04, -6.0771e-04, -2.0285e-04,  3.4686e-04,
           6.8113e-04,  6.7921e-04,  5.0776e-04,  3.8163e-04,  2.1450e-04,
          -1.6965e-04, -1.6414e-04,  7.1620e-05,  2.9065e-04,  7.5659e-04,
           1.0058e-03,  1.2552e-03,  1.0699e-03,  9.3360e-04,  6.6751e-04]]]),
            "codec_error": 0.002784556010738015
        },
        1.0: {
            "code": None,
        #     "code": torch.tensor([[[ 837,  619,  704,  704,  704,  619,  977,  619,  704,  704,  704,
        #    885,  437,  864,  471],
        #  [ 848,  335,  335,  961,  134,  134,  553,  553,  170,  496,  961,
        #    553,  961, 1011,  961]]]),
            "decoded": torch.tensor([[[-2.1897e-04, -6.7073e-04, -3.1986e-04, -2.0197e-04, -3.1223e-04,
          -1.5073e-04,  1.3110e-05,  1.0109e-04, -4.4848e-05, -2.6842e-04,
          -2.1817e-04, -2.9384e-04, -5.0452e-04, -5.7566e-04, -2.5194e-04,
           1.7382e-04,  3.4506e-04,  4.6400e-04,  3.2735e-04, -2.9985e-04,
          -7.3240e-04, -6.3583e-04, -1.9679e-04,  9.1550e-05,  2.3105e-04,
           2.6949e-04,  5.3934e-05, -1.7367e-04, -3.7033e-04, -4.5389e-04,
          -3.7428e-04, -3.1831e-04, -3.1072e-04, -7.1176e-05,  3.2563e-04,
           5.1735e-04,  3.5740e-04,  1.5329e-04,  8.0183e-05, -8.0111e-05,
          -4.1296e-04, -2.1683e-04,  1.6200e-04,  3.4918e-04,  6.2400e-04,
           7.1023e-04,  6.8606e-04,  2.1982e-04, -7.1069e-05, -1.8566e-04]]]),
            "codec_error": 0.0025421823374927044
        },
        1.5: {
            "code": None,
        #     "code": torch.tensor([[[ 837,  619,  704,  704,  704,  619,  977,  619,  704,  704,  704,
        #    885,  437,  864,  471],
        #  [ 848,  335,  335,  961,  134,  134,  553,  553,  170,  496,  961,
        #    553,  961, 1011,  961],
        #  [ 335,  222,  222,  656,  650,  801,   95,  712,   95,  986,  948,
        #    724,  948,  986,  966]]]),
            "decoded": torch.tensor([[[-2.8944e-04, -7.2795e-04, -3.7402e-04, -2.3793e-04, -3.4111e-04,
          -1.5954e-04,  4.3848e-05,  3.8975e-05, -1.5777e-04, -3.8873e-04,
          -3.2729e-04, -3.0824e-04, -5.2032e-04, -6.4972e-04, -3.2806e-04,
           9.0036e-05,  2.5077e-04,  3.8552e-04,  3.1532e-04, -2.8256e-04,
          -7.8011e-04, -6.7990e-04, -1.8713e-04,  9.0754e-05,  2.0593e-04,
           2.5207e-04,  8.5912e-05, -6.9033e-05, -2.2177e-04, -3.1648e-04,
          -2.8968e-04, -3.4715e-04, -4.5017e-04, -1.4488e-04,  4.2452e-04,
           7.1036e-04,  5.3200e-04,  3.0199e-04,  1.7572e-04, -7.3444e-05,
          -4.1257e-04, -2.2480e-04,  1.0712e-04,  2.1271e-04,  4.7596e-04,
           7.0453e-04,  8.2431e-04,  4.2970e-04,  2.1766e-04,  3.3507e-05]]]),
            "codec_error": 0.0024348432198166847
        },
        2.0: {
            "code": None,
        #     "code": torch.tensor([[[ 837,  619,  704,  704,  704,  619,  977,  619,  704,  704,  704,
        #    885,  437,  864,  471],
        #  [ 848,  335,  335,  961,  134,  134,  553,  553,  170,  496,  961,
        #    553,  961, 1011,  961],
        #  [ 335,  222,  222,  656,  650,  801,   95,  712,   95,  986,  948,
        #    724,  948,  986,  966],
        #  [ 601,  541,  234,  684,  747,  960,  601,  963,  601,  601,  747,
        #    388,   13,  963,  201]]]),
            "decoded": torch.tensor([[[-1.0455e-04, -6.1305e-04, -1.9683e-04, -1.0099e-04, -2.1099e-04,
          -2.6221e-06,  2.0872e-04,  1.8668e-04, -1.0649e-05, -1.8443e-04,
          -9.1624e-05, -1.0016e-04, -3.7102e-04, -4.8930e-04, -1.1564e-04,
           3.2124e-04,  4.6699e-04,  6.2269e-04,  5.6839e-04, -1.2995e-04,
          -7.3447e-04, -6.5225e-04, -7.3695e-05,  2.3017e-04,  2.9707e-04,
           4.0724e-04,  3.1984e-04,  1.0261e-04, -1.7222e-04, -2.6103e-04,
          -1.5820e-04, -2.7220e-04, -4.7698e-04, -1.3688e-04,  4.7703e-04,
           7.2367e-04,  5.7554e-04,  5.3053e-04,  4.9716e-04,  9.5859e-05,
          -4.4039e-04, -2.5911e-04,  8.8823e-05,  1.0468e-04,  4.4652e-04,
           8.7012e-04,  1.1334e-03,  6.0353e-04,  3.2270e-04,  1.8620e-04]]]),
            "codec_error": 0.0022935366723686457
        },
        4.0: {
            "code": None,
            "decoded": torch.tensor([[[ 2.6091e-04, -5.5686e-04, -8.9853e-06,  8.9663e-05, -7.4791e-05,
           2.0765e-04,  3.4825e-04,  3.0706e-04,  1.9627e-04, -3.6874e-05,
          -7.0256e-06, -1.2438e-04, -3.2755e-04, -4.1191e-04, -8.8219e-05,
           3.7177e-04,  4.7274e-04,  6.2578e-04,  5.6791e-04, -1.4851e-04,
          -6.2424e-04, -5.3526e-04, -1.3237e-06,  3.0456e-04,  4.1467e-04,
           5.9324e-04,  4.4341e-04,  2.3843e-04,  8.9017e-05, -1.0392e-05,
           1.0842e-04,  2.1005e-04,  8.2285e-05,  3.8514e-04,  8.2333e-04,
           8.8926e-04,  7.8599e-04,  6.9101e-04,  5.6542e-04,  3.4487e-04,
          -1.2858e-04, -3.4345e-05,  3.0926e-04,  3.4923e-04,  6.0201e-04,
           8.2463e-04,  1.0069e-03,  3.9763e-04,  1.7323e-04,  1.2473e-04]]]),
            "codec_error": 0.002119641751050949
        },
    },
    "xcodec-wavlm-more-data": {
        0.5: {
            "code": torch.tensor([[[ 44, 881, 344, 344, 344, 881,  44, 881, 571, 813, 107, 950, 437, 950,
          437]]]),
            "decoded": torch.tensor([[[ 8.6080e-05, -3.0602e-04,  9.0437e-05,  1.4830e-04, -2.1000e-05,
           1.7922e-04,  2.3896e-04,  2.4265e-04,  9.4004e-05, -1.3934e-04,
          -1.7325e-04, -1.0025e-04, -9.4449e-05, -2.2923e-04, -1.1478e-06,
           3.7331e-04,  4.8061e-04,  4.4257e-04,  3.0929e-04, -2.4855e-05,
          -1.5137e-04, -4.7809e-05,  2.0679e-04,  3.3284e-04,  3.7892e-04,
           4.1557e-04,  1.9657e-04, -6.5251e-05, -1.4414e-04, -2.4506e-04,
          -2.7640e-04, -2.5284e-04, -2.7842e-04, -5.5240e-05,  3.4663e-04,
           5.0817e-04,  3.4517e-04,  1.7949e-04,  1.8056e-04,  5.4157e-05,
          -2.9087e-04, -2.2547e-04, -2.8170e-06,  8.7965e-05,  3.6707e-04,
           4.8741e-04,  6.2959e-04,  4.8174e-04,  4.2524e-04,  2.6322e-04]]]),
            "codec_error": 0.002914232201874256
        },
        1.0: {
            "code": torch.tensor([[[ 44, 881, 344, 344, 344, 881,  44, 881, 571, 813, 107, 950, 437, 950,
          437],
         [659, 335, 335, 801,  30, 726, 647, 721, 562, 421, 421, 797, 797, 797,
          797]]]),
            "decoded": torch.tensor([[[-1.3854e-04, -6.0011e-04, -1.4206e-04, -5.9195e-06, -1.0433e-04,
           8.3727e-05,  1.9300e-04,  2.3585e-04, -3.9955e-05, -2.9311e-04,
          -2.6569e-04, -4.3283e-04, -6.5565e-04, -6.9341e-04, -2.2642e-04,
           2.5325e-04,  5.7375e-04,  7.9662e-04,  5.2056e-04, -1.5709e-04,
          -4.7744e-04, -3.2604e-04,  6.2916e-05,  2.6513e-04,  3.1191e-04,
           3.7558e-04,  2.4415e-04,  7.2527e-05, -6.4866e-05, -1.6354e-04,
          -3.5271e-05, -1.1909e-04, -4.9999e-04, -4.1618e-04,  2.2183e-04,
           4.9358e-04,  2.8011e-04,  3.2458e-04,  7.0085e-04,  6.5556e-04,
           7.3185e-05,  1.4355e-04,  3.7439e-04,  1.7988e-04,  2.2015e-04,
           3.7314e-04,  5.4426e-04,  2.8907e-04,  2.5519e-04,  3.8737e-04]]]),
            "codec_error": 0.002613937947899103
        },
        1.5: {
            "code": None,
            "decoded": torch.tensor([[[ 6.2799e-05, -3.3690e-04,  1.1250e-04,  1.9458e-04,  1.2857e-04,
           3.2529e-04,  3.9155e-04,  3.0269e-04,  2.7159e-04,  1.4779e-04,
           1.1753e-04,  5.8236e-05, -1.9056e-04, -3.7417e-04,  6.2791e-05,
           7.3489e-04,  1.0395e-03,  1.0661e-03,  7.7554e-04, -2.9395e-05,
          -5.7636e-04, -2.7807e-04,  3.7756e-04,  6.3580e-04,  6.6073e-04,
           5.5544e-04,  1.5077e-04, -7.9727e-05, -7.0684e-05, -1.8670e-04,
          -1.8936e-04, -1.2967e-04, -3.8470e-04, -4.1005e-04,  2.5019e-04,
           7.8431e-04,  5.6394e-04,  4.6740e-04,  7.5482e-04,  5.2776e-04,
          -1.5600e-05,  2.9396e-04,  6.6701e-04,  5.3045e-04,  8.2493e-04,
           1.2162e-03,  1.1341e-03,  7.2962e-04,  8.0423e-04,  5.6797e-04]]]),
            "codec_error": 0.002474759239703417
        },
        2.0: {
            "code": None,
            "decoded": torch.tensor([[[ 2.5080e-04, -1.7259e-04,  2.5968e-04,  3.6472e-04,  2.6965e-04,
           4.1702e-04,  5.7311e-04,  5.5453e-04,  4.9474e-04,  3.1295e-04,
           2.8015e-04,  1.7707e-04, -1.1514e-04, -2.6510e-04,  2.1959e-04,
           8.1657e-04,  1.1201e-03,  1.2912e-03,  1.1668e-03,  5.3093e-04,
           8.7590e-06,  2.1711e-04,  7.1509e-04,  8.0455e-04,  7.0224e-04,
           7.4241e-04,  6.1524e-04,  4.5919e-04,  3.9249e-04,  2.6367e-04,
           1.7672e-04,  4.0699e-05, -1.5962e-04, -7.8742e-05,  4.4577e-04,
           8.7299e-04,  7.4678e-04,  7.1037e-04,  1.0272e-03,  8.9605e-04,
           4.2782e-04,  6.7035e-04,  8.9649e-04,  6.1876e-04,  7.4221e-04,
           1.0012e-03,  8.7497e-04,  4.5577e-04,  5.8786e-04,  4.6868e-04]]]),
            "codec_error": 0.0023411870934069157
        },
        4.0: {
            "code": None,
            "decoded": torch.tensor([[[ 5.9435e-04,  1.7470e-04,  7.0983e-04,  8.1596e-04,  8.1736e-04,
           1.0675e-03,  1.1203e-03,  9.7972e-04,  8.5006e-04,  6.0132e-04,
           4.8908e-04,  4.7585e-04,  4.3114e-04,  4.4707e-04,  8.8117e-04,
           1.4071e-03,  1.5383e-03,  1.4448e-03,  1.1609e-03,  4.9698e-04,
          -1.3974e-06,  1.6933e-04,  7.6717e-04,  1.0725e-03,  1.0862e-03,
           1.0590e-03,  7.5759e-04,  3.8924e-04, -1.2118e-05, -3.6682e-04,
          -4.3772e-04, -3.8845e-04, -3.6490e-04, -5.0594e-05,  5.7642e-04,
           1.0160e-03,  9.5409e-04,  9.2827e-04,  1.0461e-03,  7.9255e-04,
           2.6879e-04,  4.8614e-04,  9.1454e-04,  9.6337e-04,  1.2745e-03,
           1.6484e-03,  1.6050e-03,  8.4984e-04,  5.4461e-04,  1.5283e-04]]]),
            "codec_error": 0.0021570040844380856
        },
    },
}
# fmt: on


@slow
@require_torch
class XcodecIntegrationTest(unittest.TestCase):
    @parameterized.expand(
        [
            (f"{model_id}_{str(bandwidth).replace('.', 'p')}", model_id, bandwidth)
            for model_id, v in EXPECTED_OUTPUTS.items()
            for bandwidth in v
        ]
    )
    def test_integration(self, name, model_id, bandwidth):
        # load model
        repo_id = f"hf-audio/{model_id}"
        model = XcodecModel.from_pretrained(repo_id).to(torch_device).eval()
        feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)

        # load audio example
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        librispeech_dummy = librispeech_dummy.cast_column(
            "audio", Audio(sampling_rate=feature_extractor.sampling_rate)
        )
        audio_array = librispeech_dummy[0]["audio"]["array"]
        inputs = feature_extractor(
            raw_audio=audio_array, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt"
        ).to(torch_device)
        x = inputs["input_values"]

        with torch.no_grad():
            ENC_TOL = 0
            audio_codes = model.encode(x, bandwidth=bandwidth, return_dict=False)
            if EXPECTED_OUTPUTS[model_id][bandwidth]["code"] is not None:
                torch.testing.assert_close(
                    audio_codes[..., : EXPECTED_OUTPUTS[model_id][bandwidth]["code"].shape[-1]],
                    EXPECTED_OUTPUTS[model_id][bandwidth]["code"].to(torch_device),
                    rtol=ENC_TOL,
                    atol=ENC_TOL,
                )

            DEC_TOL = 1e-3
            input_values_dec = model.decode(audio_codes).audio_values
            if EXPECTED_OUTPUTS[model_id][bandwidth]["decoded"] is not None:
                torch.testing.assert_close(
                    input_values_dec[..., : EXPECTED_OUTPUTS[model_id][bandwidth]["decoded"].shape[-1]],
                    EXPECTED_OUTPUTS[model_id][bandwidth]["decoded"].to(torch_device),
                    rtol=DEC_TOL,
                    atol=DEC_TOL,
                )

            # compute codec error
            codec_tol = EXPECTED_OUTPUTS[model_id][bandwidth].get("codec_tol", 1e-5)
            codec_err = compute_rmse(input_values_dec, x)
            torch.testing.assert_close(
                codec_err, EXPECTED_OUTPUTS[model_id][bandwidth]["codec_error"], rtol=codec_tol, atol=codec_tol
            )

            # make sure forward and decode gives same result
            audio_values_enc_dec = model(x, bandwidth=bandwidth).audio_values
            torch.testing.assert_close(input_values_dec, audio_values_enc_dec, rtol=1e-6, atol=1e-6)
