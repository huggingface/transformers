# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Mimi model."""

import inspect
import os
import tempfile
import unittest

import numpy as np
from datasets import Audio, load_dataset
from pytest import mark

from transformers import AutoFeatureExtractor, MimiConfig
from transformers.testing_utils import (
    is_flaky,
    is_torch_available,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import MimiModel


# Copied from transformers.tests.encodec.test_modeling_encodec.prepare_inputs_dict
def prepare_inputs_dict(
    config,
    input_ids=None,
    input_values=None,
    decoder_input_ids=None,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if input_ids is not None:
        encoder_dict = {"input_ids": input_ids}
    else:
        encoder_dict = {"input_values": input_values}

    decoder_dict = {"decoder_input_ids": decoder_input_ids} if decoder_input_ids is not None else {}

    return {**encoder_dict, **decoder_dict}


@require_torch
class MimiModelTester:
    def __init__(
        self,
        parent,
        batch_size=5,
        num_channels=1,
        is_training=False,
        intermediate_size=40,
        hidden_size=32,
        num_filters=8,
        num_residual_layers=1,
        upsampling_ratios=[8, 4],
        codebook_size=64,
        vector_quantization_hidden_dimension=64,
        codebook_dim=64,
        upsample_groups=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        sliding_window=4,
        use_cache=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.num_residual_layers = num_residual_layers
        self.upsampling_ratios = upsampling_ratios
        self.codebook_size = codebook_size
        self.vector_quantization_hidden_dimension = vector_quantization_hidden_dimension
        self.codebook_dim = codebook_dim
        self.upsample_groups = upsample_groups
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.sliding_window = sliding_window
        self.use_cache = use_cache

    def prepare_config_and_inputs(self, input_values_length=None):
        input_values = floats_tensor(
            [
                self.batch_size,
                self.num_channels,
                self.intermediate_size if input_values_length is None else input_values_length,
            ],
            scale=1.0,
        )
        config = self.get_config()
        inputs_dict = {"input_values": input_values}
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self, input_values_length=None):
        config, inputs_dict = self.prepare_config_and_inputs(input_values_length=input_values_length)
        return config, inputs_dict

    def prepare_config_and_inputs_for_model_class(self, model_class):
        config, inputs_dict = self.prepare_config_and_inputs()
        inputs_dict["audio_codes"] = ids_tensor([self.batch_size, 1, self.num_channels], self.codebook_size).type(
            torch.int32
        )

        return config, inputs_dict

    def get_config(self):
        return MimiConfig(
            audio_channels=self.num_channels,
            chunk_in_sec=None,
            hidden_size=self.hidden_size,
            num_filters=self.num_filters,
            num_residual_layers=self.num_residual_layers,
            upsampling_ratios=self.upsampling_ratios,
            codebook_size=self.codebook_size,
            vector_quantization_hidden_dimension=self.vector_quantization_hidden_dimension,
            upsample_groups=self.upsample_groups,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            sliding_window=self.sliding_window,
            codebook_dim=self.codebook_dim,
            use_cache=self.use_cache,
        )

    def create_and_check_model_forward(self, config, inputs_dict):
        model = MimiModel(config=config).to(torch_device).eval()

        input_values = inputs_dict["input_values"]
        result = model(input_values)
        self.parent.assertEqual(
            result.audio_values.shape, (self.batch_size, self.num_channels, self.intermediate_size)
        )


@require_torch
class MimiModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (MimiModel,) if is_torch_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False
    test_torchscript = False

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        # model does support returning hidden states
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if "output_attentions" in inputs_dict:
            inputs_dict.pop("output_attentions")
        if "output_hidden_states" in inputs_dict:
            inputs_dict.pop("output_hidden_states")
        return inputs_dict

    def setUp(self):
        self.model_tester = MimiModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=MimiConfig, hidden_size=37, common_properties=[], has_text_modality=False
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

            expected_arg_names = ["input_values", "padding_mask", "num_quantizers"]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    @unittest.skip(reason="The MimiModel does not have `inputs_embeds` logics")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="The MimiModel does not have `inputs_embeds` logics")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="The MimiModel does not have the usual `attention` logic")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="The MimiModel does not have the usual `attention` logic")
    def test_torchscript_output_attentions(self):
        pass

    @unittest.skip(reason="The MimiModel does not have the usual `hidden_states` logic")
    def test_torchscript_output_hidden_state(self):
        pass

    # Copied from transformers.tests.encodec.test_modeling_encodec.MimiModelTest._create_and_check_torchscript
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
            for key in loaded_model_state_dict:
                if key not in model_state_dict:
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

    @unittest.skip(reason="The MimiModel does not have the usual `attention` logic")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="The MimiModel does not have the usual `hidden_states` logic")
    def test_hidden_states_output(self):
        pass

    # Copied from transformers.tests.encodec.test_modeling_encodec.MimiModelTest.test_determinism
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

    # Copied from transformers.tests.encodec.test_modeling_encodec.MimiModelTest.test_model_outputs_equivalence
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
                uniform_init_parms = ["conv", "input_proj", "output_proj"]
                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    # Copied from transformers.tests.encodec.test_modeling_encodec.MimiModelTest.test_identity_shortcut
    def test_identity_shortcut(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        config.use_conv_shortcut = False
        self.model_tester.create_and_check_model_forward(config, inputs_dict)

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
                    tmpdirname, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, dtype=torch.bfloat16)
                model.to(torch_device)

                dummy_input = inputs_dict[model.main_input_name][:1]
                if dummy_input.dtype in [torch.float32, torch.float16]:
                    dummy_input = dummy_input.to(torch.bfloat16)

                outputs = model(dummy_input)
                outputs_fa = model_fa(dummy_input)

                logits = outputs[1]
                logits_fa = outputs_fa[1]

                assert torch.allclose(logits_fa, logits, atol=4e-2, rtol=4e-2)

    @unittest.skip(reason="The MimiModel does not support right padding")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @unittest.skip(reason="The MimiModel does not have support dynamic compile yet")
    def test_sdpa_can_compile_dynamic(self):
        pass


# Copied from transformers.tests.encodec.test_modeling_encodec.normalize
def normalize(arr):
    norm = np.linalg.norm(arr)
    normalized_arr = arr / norm
    return normalized_arr


# Copied from transformers.tests.encodec.test_modeling_encodec.compute_rmse
def compute_rmse(arr1, arr2):
    arr1_normalized = normalize(arr1)
    arr2_normalized = normalize(arr2)
    return np.sqrt(((arr1_normalized - arr2_normalized) ** 2).mean())


@slow
@require_torch
class MimiIntegrationTest(unittest.TestCase):
    def test_integration_using_cache_decode(self):
        expected_rmse = {
            "8": 0.0018785292,
            "32": 0.0012330565,
        }

        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        model_id = "kyutai/mimi"

        model = MimiModel.from_pretrained(model_id, use_cache=True).to(torch_device)
        processor = AutoFeatureExtractor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        audio_sample = librispeech_dummy[-1]["audio"]["array"]

        inputs = processor(
            raw_audio=audio_sample,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
        ).to(torch_device)

        for num_codebooks, expected_rmse in expected_rmse.items():
            with torch.no_grad():
                # use max bandwidth for best possible reconstruction
                encoder_outputs = model.encode(inputs["input_values"], num_quantizers=int(num_codebooks))

                audio_codes = encoder_outputs[0]

                decoder_outputs_first_part = model.decode(audio_codes[:, :, : audio_codes.shape[2] // 2])
                decoder_outputs_second_part = model.decode(
                    audio_codes[:, :, audio_codes.shape[2] // 2 :],
                    decoder_past_key_values=decoder_outputs_first_part.decoder_past_key_values,
                )

                audio_output_entire_context = model.decode(audio_codes)[0]
                audio_output_concat_context = torch.cat(
                    [decoder_outputs_first_part[0], decoder_outputs_second_part[0]], dim=2
                )

            # make sure audios are more or less equal
            # the RMSE of two random gaussian noise vectors with ~N(0, 1) is around 1.0
            rmse = compute_rmse(
                audio_output_concat_context.squeeze().cpu().numpy(),
                audio_output_entire_context.squeeze().cpu().numpy(),
            )
            self.assertTrue(rmse < 1e-3)

    def test_integration_encode_with_padding_cache(self):
        """
        We test here the possibility to run Mimi in a streaming manner, i.e. chunk by chunk.
        1. we encode a first time the entire audio
        2. we encode the audio chunk by chunk, each chunk being the smallest size possible for the model (i.e. the frame size)

        This test must be run on CPU since GPU floating point operations accumulate rounding errors that cause test failures.
        """
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        model_id = "kyutai/mimi"

        model = MimiModel.from_pretrained(model_id, use_cache=True).to("cpu")
        processor = AutoFeatureExtractor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        audio_sample = librispeech_dummy[-1]["audio"]["array"]

        inputs = processor(
            raw_audio=audio_sample,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
        ).to("cpu")

        frame_size = model.config.frame_size
        audio_codes = model.encode(inputs["input_values"]).audio_codes

        # streaming chunk by chunk
        encoder_past_key_values = None
        padding_cache = None
        encoded_frames_list = []

        for start in range(0, inputs["input_values"].shape[-1], frame_size):
            input_values_chunk = inputs["input_values"][:, :, start : start + frame_size]
            encoder_outputs = model.encode(
                input_values_chunk,
                padding_cache=padding_cache,
                encoder_past_key_values=encoder_past_key_values,
                use_streaming=True,
            )
            encoder_past_key_values = encoder_outputs.encoder_past_key_values
            padding_cache = encoder_outputs.padding_cache
            encoded_frames_list.append(encoder_outputs.audio_codes)

        streamed_audio_codes = torch.cat(encoded_frames_list, dim=-1)

        torch.testing.assert_close(streamed_audio_codes, audio_codes)

    def test_integration(self):
        expected_rmses = {
            "8": 0.0018785292,
            "32": 0.0012330565,
        }
        expected_codesums = {
            "8": 426176,
            "32": 1795819,
        }
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        model_id = "kyutai/mimi"

        processor = AutoFeatureExtractor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        audio_sample = librispeech_dummy[-1]["audio"]["array"]

        inputs = processor(
            raw_audio=audio_sample,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
        ).to(torch_device)

        for use_cache in [False, True]:
            model = MimiModel.from_pretrained(model_id, use_cache=use_cache).to(torch_device)
            for num_codebooks, expected_rmse in expected_rmses.items():
                with torch.no_grad():
                    # use max bandwidth for best possible reconstruction
                    encoder_outputs = model.encode(inputs["input_values"], num_quantizers=int(num_codebooks))

                    audio_code_sums = encoder_outputs[0].sum().item()

                    # make sure audio encoded codes are correct
                    # assert relative difference less than a threshold, because `audio_code_sums` varies a bit
                    # depending on torch version
                    self.assertTrue(
                        np.abs(audio_code_sums - expected_codesums[num_codebooks]) <= (3e-3 * audio_code_sums)
                    )

                    input_values_dec = model.decode(encoder_outputs[0], padding_mask=inputs["padding_mask"])[0]
                    input_values_enc_dec = model(
                        inputs["input_values"], inputs["padding_mask"], num_quantizers=int(num_codebooks)
                    )[1]

                # make sure forward and decode gives same result
                torch.testing.assert_close(input_values_dec, input_values_enc_dec)

                # make sure shape matches
                self.assertTrue(inputs["input_values"].shape == input_values_enc_dec.shape)

                arr = inputs["input_values"][0].cpu().numpy()
                arr_enc_dec = input_values_enc_dec[0].cpu().numpy()

                # make sure audios are more or less equal
                # the RMSE of two random gaussian noise vectors with ~N(0, 1) is around 1.0
                rmse = compute_rmse(arr, arr_enc_dec)
                self.assertTrue(np.abs(rmse - expected_rmse) < 1e-5)
