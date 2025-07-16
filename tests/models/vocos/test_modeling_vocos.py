# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import tempfile
import unittest

from datasets import Audio, load_dataset

from transformers.testing_utils import (
    require_torch,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    set_model_tester_for_less_flaky_test,
    torch_device,
)
from transformers.utils import is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch

    from transformers import VocosModel, VocosWithEncodecModel


from transformers import VocosConfig, VocosFeatureExtractor, VocosWithEncodecConfig


class VocosModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 2
        self.input_channels = 8
        self.hidden_dim = 16
        self.intermediate_dim = 32
        self.num_layers = 2
        self.kernel_size = 3
        self.padding = 1
        self.layer_scale_init_value = 0.1
        self.use_adaptive_norm = False
        self.num_bandwidths = 1
        self.layer_norm_eps = 1e-6
        self.n_fft = 16
        self.hop_length = 8
        self.spec_padding = "center"
        self.seq_length = 10

    def get_config(self):
        return VocosConfig(
            input_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            kernel_size=self.kernel_size,
            padding=self.padding,
            layer_scale_init_value=self.layer_scale_init_value,
            use_adaptive_norm=self.use_adaptive_norm,
            num_bandwidths=self.num_bandwidths,
            layer_norm_eps=self.layer_norm_eps,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            spec_padding=self.spec_padding,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_values = floats_tensor([self.batch_size, self.input_channels, self.seq_length])
        return config, input_values

    def prepare_config_and_inputs_for_common(self):
        config, features = self.prepare_config_and_inputs()
        return config, {"features": features}

    def create_and_check_model(self, config, features):
        model = VocosModel(config=config).to(torch_device).eval()
        with torch.no_grad():
            audio = model(features.to(torch_device))
        if config.spec_padding == "center":
            # the expected output using PyTorch's ISTFT
            expected_len = (self.seq_length - 1) * config.hop_length
        else:
            # when padding is same "same" padding, the expected output using the custom ISTFT implementation
            pad = (config.n_fft - config.hop_length) // 2
            expected_len = (self.seq_length - 1) * config.hop_length + config.n_fft - 2 * pad
        self.parent.assertEqual(audio.shape, (self.batch_size, expected_len))


@require_torch
class VocosModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (VocosModel,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_torchscript = False
    test_resize_embeddings = False
    test_attention_outputs = False
    has_attentions = False
    test_missing_keys = False
    test_can_init_all_missing_weights = False

    def setUp(self):
        self.model_tester = VocosModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=VocosConfig, common_properties=[], has_text_modality=False
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(
        reason="The VocosModel is not transformers based, thus it does not have the usual `hidden_states` logic"
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
                    "out_proj.weight",
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

    @unittest.skip(
        reason="The VocosModel is not transformers based, thus it does not have the usual `hidden_states` logic"
    )
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="We cannot configure to output a smaller model.")
    def test_model_is_small(self):
        pass

    @unittest.skip(reason="The VocosModel does not have `inputs_embeds` logics")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="VocosModel only has one output format.")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="VocosModel does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="VocosModel does not output any loss term in the forward pass")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="VocosModel does not output any loss term in the forward pass")
    def test_training(self):
        pass

    @unittest.skip(reason="VocosModel does not output any loss term in the forward pass")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="The VocosModel does not have the usual `attention` logic")
    def test_torchscript_output_attentions(self):
        pass

    @unittest.skip("VocosModel cannot be tested with meta device")
    def test_can_load_with_meta_device_context_manager(self):
        pass

    @unittest.skip(reason="The VocosModel does not have the usual `hidden_states` logic")
    def test_torchscript_output_hidden_state(self):
        pass

    def test_save_load_strict(self):
        config, _ = self.model_tester.prepare_config_and_inputs()
        model = VocosModel(config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            _, info = VocosModel.from_pretrained(tmpdirname, output_loading_info=True)
        self.assertEqual(info["missing_keys"], [])


class VocosWithEncodecModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 2
        self.audio_length = 512
        self.seq_length = 12

    def get_config(self):
        return VocosWithEncodecConfig()

    def prepare_config_and_inputs(self):
        config = self.get_config()
        model = VocosWithEncodecModel(config).to(torch_device).eval()
        codebook_size = model.encodec_model.quantizer.layers[0].codebook.embed.shape[0]
        codes = ids_tensor([model.num_quantizers, self.batch_size, self.seq_length], codebook_size).to(torch_device)
        audio = floats_tensor([self.batch_size, self.audio_length]).to(torch_device)
        bandwidth_id = torch.tensor(0, dtype=torch.long, device=torch_device)
        return config, codes, audio, bandwidth_id

    def prepare_config_and_inputs_for_common(self):
        config, codes, audio, bandwidth_id = self.prepare_config_and_inputs()
        return config, {
            "audio": audio,
            "codes": codes,
            "bandwidth_id": bandwidth_id,
        }

    def create_and_check_model(self, config, codes, audio, bandwidth_id):
        model = VocosWithEncodecModel(config=config).to(torch_device).eval()
        with torch.no_grad():
            audio_from_codes = model(codes=codes, bandwidth_id=bandwidth_id)
        if config.spec_padding == "center":
            expected_len_codes = (self.seq_length - 1) * config.hop_length + config.n_fft
        elif config.spec_padding == "same":
            expected_len_codes = self.seq_length * config.hop_length
        self.parent.assertEqual(audio_from_codes.shape, (self.batch_size, expected_len_codes))
        with torch.no_grad():
            audio_from_audio, out_codes = model(audio=audio, bandwidth_id=bandwidth_id, return_codes=True)
        actual_seq_length = out_codes.shape[-1]
        if config.spec_padding == "center":
            expected_len = (actual_seq_length - 1) * config.hop_length + config.n_fft
        elif config.spec_padding == "same":
            expected_len = actual_seq_length * config.hop_length
        self.parent.assertEqual(audio_from_audio.shape, (self.batch_size, expected_len))


@require_torch
class VocosWithEncodecModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (VocosWithEncodecModel,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_torchscript = False
    test_resize_embeddings = False
    test_attention_outputs = False
    has_attentions = False
    test_missing_keys = False
    test_can_init_all_missing_weights = False

    def setUp(self):
        self.model_tester = VocosWithEncodecModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=VocosWithEncodecConfig, common_properties=[], has_text_modality=False
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, codes, audio, bandwidth_id = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, codes, audio, bandwidth_id)

    def test_save_load_strict(self):
        config, codes, audio, bandwidth_id = self.model_tester.prepare_config_and_inputs()
        model = VocosWithEncodecModel(config=config)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            _, info = VocosWithEncodecModel.from_pretrained(tmpdir, output_loading_info=True)
        self.assertListEqual(info["missing_keys"], [])

    @unittest.skip(reason="The VocosWithEncodecModel does not support hidden_states")
    def test_save_load(self):
        pass

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = [
                    "conv.bias",
                    "parametrizations.weight.original",
                    "embed.weight",
                    "dwconv.weight",
                    "pwconv1.weight",
                    "pwconv2.weight",
                    "out_proj.weight",
                    "lstm.",
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

    @unittest.skip(
        reason="The VocosWithEncodecModel is not transformers based, thus it does not have the usual `hidden_states` logic"
    )
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="We cannot configure to output a smaller model.")
    def test_model_is_small(self):
        pass

    @unittest.skip(reason="The VocosWithEncodecModel does not have `inputs_embeds` logics")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="VocosWithEncodecModel only has one output format.")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="VocosWithEncodecModel does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="VocosWithEncodecModel does not output any loss term in the forward pass")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="VocosWithEncodecModel does not output any loss term in the forward pass")
    def test_training(self):
        pass

    @unittest.skip(reason="VocosWithEncodecModel does not output any loss term in the forward pass")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="The VocosWithEncodecModel does not have the usual `attention` logic")
    def test_torchscript_output_attentions(self):
        pass

    @unittest.skip("VocosWithEncodecModel cannot be tested with meta device")
    def test_can_load_with_meta_device_context_manager(self):
        pass

    @unittest.skip(reason="The VocosWithEncodecModel does not have the usual `hidden_states` logic")
    def test_torchscript_output_hidden_state(self):
        pass

    # Overwrite to prevent splitting a scalar `bandwidth_id` by checking tensor.dim() before batch splitting.
    def test_batching_equivalence(self, atol=1e-5, rtol=1e-5):
        """
        Tests that the model supports batching and that the output is the nearly the same for the same input in
        different batch sizes.
        (Why "nearly the same" not "exactly the same"? Batching uses different matmul shapes, which often leads to
        different results: https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535)
        """

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

        set_model_tester_for_less_flaky_test(self)

        config, batched_input = self.model_tester.prepare_config_and_inputs_for_common()
        set_config_for_less_flaky_test(config)

        for model_class in self.all_model_classes:
            config.output_hidden_states = True

            model_name = model_class.__name__
            if hasattr(self.model_tester, "prepare_config_and_inputs_for_model_class"):
                config, batched_input = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
            batched_input_prepared = self._prepare_for_class(batched_input, model_class)
            model = model_class(config).to(torch_device).eval()
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
class VocosModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.feature_extractor = VocosFeatureExtractor.from_pretrained("Manel/Vocos", resume_download=True)
        self.model = VocosModel.from_pretrained("Manel/Vocos", resume_download=True).to(torch_device).eval()
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.cast_column("audio", Audio(sampling_rate=self.feature_extractor.sampling_rate))
        self.speech = ds[0]["audio"]["array"]

    def test_inference(self):
        EXPECTED = torch.tensor(
            [
                0.0001700431457720697,
                0.00010000158363254741,
                -5.997690459480509e-05,
                -8.697436715010554e-05,
                3.8385427615139633e-05,
                0.0001993452024180442,
                0.00026118403184227645,
                0.00024136024876497686,
                0.0002001010434469208,
                0.000260183762293309,
                0.000239697823417373,
                1.3868119822291192e-05,
                -6.546344957314432e-05,
                2.3145852537709288e-05,
                0.0001909736020024866,
                0.00043056777212768793,
                0.00040265079587697983,
                -7.634644862264395e-05,
                -0.0007267086184583604,
                -0.0012220395728945732,
            ],
            dtype=torch.float32,
        )

        inputs = self.feature_extractor(
            self.speech, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
        ).to(torch_device)

        with torch.no_grad():
            audio = self.model(inputs.input_features)

        expected_shape = torch.Size([1, 140544])
        self.assertEqual(audio.shape, expected_shape)

        torch.testing.assert_close(audio[0][: EXPECTED.shape[0]], EXPECTED, rtol=1e-4, atol=1e-4)


@require_torch
class VocosWithEncodecModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model = (
            VocosWithEncodecModel.from_pretrained("Manel/Vocos-Encodec", resume_download=True).to(torch_device).eval()
        )
        self.config = VocosWithEncodecConfig.from_pretrained("Manel/Vocos-Encodec", resume_download=True)
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.cast_column("audio", Audio(sampling_rate=24000))
        self.speech = ds[0]["audio"]["array"]

    def test_inference_audio_and_codes(self):
        EXPECTED_AUDIO = torch.tensor(
            [
                -0.00015610073751304299,
                0.0006738820229656994,
                0.0014662687899544835,
                0.0019666007719933987,
                0.0018747239373624325,
                0.0016342204762622714,
                0.0013575436314567924,
                0.0010286348406225443,
                0.00036631093826144934,
                -7.642315176781267e-05,
                -0.0005207710200920701,
                -0.0007273774244822562,
                -0.0006747262086719275,
                -6.980449688853696e-05,
                0.0008167537162080407,
                0.0008955168887041509,
                0.0011381119256839156,
                0.0012689086142927408,
                0.0016888295067474246,
                0.001389320706948638,
            ],
            dtype=torch.float32,
        )

        EXPECTED_AUDIO_FROM_CODES = torch.tensor(
            [
                -0.00015610073751304299,
                0.0006738820229656994,
                0.0014662687899544835,
                0.0019666007719933987,
                0.0018747239373624325,
                0.0016342204762622714,
                0.0013575436314567924,
                0.0010286348406225443,
                0.00036631093826144934,
                -7.642315176781267e-05,
                -0.0005207710200920701,
                -0.0007273774244822562,
                -0.0006747262086719275,
                -6.980449688853696e-05,
                0.0008167537162080407,
                0.0008955168887041509,
                0.0011381119256839156,
                0.0012689086142927408,
                0.0016888295067474246,
                0.001389320706948638,
            ]
        )

        audio_tensor = torch.tensor(self.speech, dtype=torch.float32, device=torch_device).unsqueeze(0)
        bandwidth_id = torch.tensor([0], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            output_from_audio = self.model(audio=audio_tensor, bandwidth_id=bandwidth_id)

        expected_shape = torch.Size((1, 140800))
        self.assertEqual(output_from_audio.shape, expected_shape)
        torch.testing.assert_close(
            output_from_audio[0, : EXPECTED_AUDIO.shape[0]], EXPECTED_AUDIO, rtol=1e-4, atol=1e-4
        )

        with torch.no_grad():
            codes = self.model.encodec_model.quantizer.encode(
                self.model.encodec_model.encoder(audio_tensor.unsqueeze(1)),
                bandwidth=self.config.encodec_config.target_bandwidths[0],
            )
            output_from_codes = self.model(codes=codes, bandwidth_id=bandwidth_id)

        self.assertEqual(output_from_codes.shape, output_from_audio.shape)
        self.assertEqual(output_from_codes.shape, expected_shape)
        torch.testing.assert_close(
            output_from_codes[0, : EXPECTED_AUDIO_FROM_CODES.shape[0]], EXPECTED_AUDIO_FROM_CODES, rtol=1e-4, atol=1e-4
        )
