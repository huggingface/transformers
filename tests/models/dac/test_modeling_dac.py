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
"""Testing suite for the PyTorch Dac model."""

import inspect
import json
import unittest
from pathlib import Path

import numpy as np
from datasets import Audio, load_dataset
from parameterized import parameterized

from tests.utils.test_audio_utils import compute_rmse
from transformers import AutoProcessor, DacConfig, DacModel
from transformers.testing_utils import (
    is_torch_available,
    require_deterministic_for_xpu,
    require_torch,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch


@require_torch
# Copied from transformers.tests.encodec.test_modeling_encodec.EncodecModelTester with Encodec->Dac
class DacModelTester:
    # Ignore copy
    def __init__(
        self,
        parent,
        batch_size=3,
        num_channels=1,
        is_training=False,
        intermediate_size=1024,
        encoder_hidden_size=16,
        downsampling_ratios=[2, 4, 4],
        decoder_hidden_size=16,
        n_codebooks=6,
        codebook_size=512,
        codebook_dim=4,
        quantizer_dropout=0.0,
        commitment_loss_weight=0.25,
        codebook_loss_weight=1.0,
        sample_rate=16000,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.intermediate_size = intermediate_size
        self.sample_rate = sample_rate

        self.encoder_hidden_size = encoder_hidden_size
        self.downsampling_ratios = downsampling_ratios
        self.decoder_hidden_size = decoder_hidden_size
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout
        self.commitment_loss_weight = commitment_loss_weight
        self.codebook_loss_weight = codebook_loss_weight

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.num_channels, self.intermediate_size], scale=1.0)
        config = self.get_config()
        inputs_dict = {"input_values": input_values}
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def prepare_config_and_inputs_for_model_class(self, model_class):
        input_values = floats_tensor([self.batch_size, self.num_channels, self.intermediate_size], scale=1.0)
        config = self.get_config()
        inputs_dict = {"input_values": input_values}

        return config, inputs_dict

    # Ignore copy
    def get_config(self):
        return DacConfig(
            encoder_hidden_size=self.encoder_hidden_size,
            downsampling_ratios=self.downsampling_ratios,
            decoder_hidden_size=self.decoder_hidden_size,
            n_codebooks=self.n_codebooks,
            codebook_size=self.codebook_size,
            codebook_dim=self.codebook_dim,
            quantizer_dropout=self.quantizer_dropout,
            commitment_loss_weight=self.commitment_loss_weight,
            codebook_loss_weight=self.codebook_loss_weight,
        )

    # Ignore copy
    def create_and_check_model_forward(self, config, inputs_dict):
        model = DacModel(config=config).to(torch_device).eval()

        input_values = inputs_dict["input_values"]
        result = model(input_values)
        self.parent.assertEqual(result.audio_values.shape, (self.batch_size, self.intermediate_size))


@require_torch
# Copied from transformers.tests.encodec.test_modeling_encodec.EncodecModelTest with Encodec->Dac
class DacModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (DacModel,) if is_torch_available() else ()
    is_encoder_decoder = True

    test_resize_embeddings = False
    pipeline_model_mapping = {"feature-extraction": DacModel} if is_torch_available() else {}

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        # model does not have attention and does not support returning hidden states
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if "output_attentions" in inputs_dict:
            inputs_dict.pop("output_attentions")
        if "output_hidden_states" in inputs_dict:
            inputs_dict.pop("output_hidden_states")
        return inputs_dict

    def setUp(self):
        self.model_tester = DacModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=DacConfig, hidden_size=32, common_properties=[], has_text_modality=False
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    # TODO (ydshieh): Although we have a potential cause, it's still strange that this test fails all the time with large differences
    @unittest.skip(reason="Might be caused by `indices` computed with `max()` in `decode_latents`")
    def test_batching_equivalence(self):
        super().test_batching_equivalence()

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            # Ignore copy
            expected_arg_names = ["input_values", "n_quantizers", "return_dict"]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    @unittest.skip("The DacModel is not transformers based, thus it does not have `inputs_embeds` logics")
    def test_inputs_embeds(self):
        pass

    @unittest.skip("The DacModel is not transformers based, thus it does not have `inputs_embeds` logics")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip("The DacModel is not transformers based, thus it does not have the usual `attention` logic")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip("The DacModel is not transformers based, thus it does not have the usual `attention` logic")
    def test_attention_outputs(self):
        pass

    @unittest.skip("The DacModel is not transformers based, thus it does not have the usual `hidden_states` logic")
    def test_hidden_states_output(self):
        pass

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

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, (list, tuple)):
                        for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, dict):
                        for tuple_iterable_value, dict_iterable_value in zip(
                            tuple_object.values(), dict_object.values()
                        ):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif tuple_object is None:
                        return
                    else:
                        self.assertTrue(
                            torch.allclose(
                                set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
                            ),
                            msg=(
                                "Tuple and dict output are not equal. Difference:"
                                f" {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                                f" {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has"
                                f" `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}."
                            ),
                        )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

    def test_identity_shortcut(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        config.use_conv_shortcut = False
        self.model_tester.create_and_check_model_forward(config, inputs_dict)

    def test_quantizer_from_latents(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = DacModel(config=config).to(torch_device).eval()
        self.assertTrue(
            all(hasattr(quantizer, "codebook_dim") for quantizer in model.quantizer.quantizers),
            msg="All quantizers should have the attribute codebook_dim",
        )
        with torch.no_grad():
            encoder_outputs = model.encode(inputs_dict["input_values"])
            latents = encoder_outputs.projected_latents
            quantizer_representation, quantized_latents = model.quantizer.from_latents(latents=latents)

        self.assertIsInstance(quantizer_representation, torch.Tensor)
        self.assertIsInstance(quantized_latents, torch.Tensor)
        self.assertEqual(quantized_latents.shape[0], latents.shape[0])
        self.assertEqual(quantized_latents.shape[1], latents.shape[1])


"""
Integration tests for DAC.

Code for reproducing expected outputs can be found here:
- test_integration: https://gist.github.com/ebezzam/bb315efa7a416db6336a6b2a2d424ffa#file-test_dac-py
- test_batch: https://gist.github.com/ebezzam/bb315efa7a416db6336a6b2a2d424ffa#file-test_dac_batch-py
NOTE (ebezzam): had to run reproducers from CI for expected outputs to match, cf PR which modified CI torch settings: https://github.com/huggingface/transformers/pull/39885

See https://github.com/huggingface/transformers/pull/39313 for reason behind large tolerance between for encoder
and decoder outputs (1e-3). In summary, original model uses weight normalization, while Transformers does not. This
leads to accumulating error. However, this does not affect the quantizer codes, thanks to discretization being
robust to precision errors. Moreover, codec error is similar between Transformers and original.

Moreover, here is a script to debug outputs and weights layer-by-layer:
https://gist.github.com/ebezzam/bb315efa7a416db6336a6b2a2d424ffa#file-dac_layer_by_layer_debugging-py
"""

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures/dac"

with open(FIXTURES_DIR / "expected_integration.json") as f:
    EXPECTED_INTEGRATION = json.load(f)

with open(FIXTURES_DIR / "expected_integration_batch.json") as f:
    EXPECTED_INTEGRATION_BATCH = json.load(f)


@slow
@require_torch
class DacIntegrationTest(unittest.TestCase):
    @parameterized.expand([(model_name,) for model_name in EXPECTED_INTEGRATION.keys()])
    @require_deterministic_for_xpu
    def test_integration(self, model_name):
        expected = EXPECTED_INTEGRATION[model_name]

        # load model and processor
        model_id = f"descript/{model_name}"
        model = DacModel.from_pretrained(model_id, force_download=True).to(torch_device).eval()
        processor = AutoProcessor.from_pretrained(model_id)

        # load audio sample
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        audio_sample = librispeech_dummy[0]["audio"]["array"]

        # check on processor audio shape
        inputs = processor(
            raw_audio=audio_sample,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
        ).to(torch_device)
        torch.equal(torch.tensor(inputs["input_values"].shape), torch.tensor(expected["preproc_shape"]))

        with torch.no_grad():
            # compare encoder loss
            encoder_outputs = model.encode(inputs["input_values"])
            torch.testing.assert_close(encoder_outputs[0].squeeze().item(), expected["enc_loss"], rtol=1e-3, atol=1e-3)

            # compare quantizer outputs
            expected_quant_codes = torch.tensor(expected["quant_codes"]).to(torch_device)
            quantizer_outputs = model.quantizer(encoder_outputs[1])
            torch.testing.assert_close(
                quantizer_outputs[1][..., : expected_quant_codes.shape[-1]],
                expected_quant_codes,
                rtol=1e-6,
                atol=1e-6,
            )
            torch.testing.assert_close(
                quantizer_outputs[4].squeeze().item(), expected["quant_codebook_loss"], rtol=1e-4, atol=1e-4
            )

            # compare decoder outputs
            expected_dec_outputs = torch.tensor(expected["dec_outputs"]).to(torch_device)
            decoded_outputs = model.decode(encoder_outputs[1])
            torch.testing.assert_close(
                decoded_outputs["audio_values"][..., : expected_dec_outputs.shape[-1]],
                expected_dec_outputs,
                rtol=1e-3,
                atol=1e-3,
            )

            # compare codec error / lossiness
            codec_err = compute_rmse(decoded_outputs["audio_values"], inputs["input_values"])
            torch.testing.assert_close(codec_err, expected["codec_error"], rtol=1e-5, atol=1e-5)

            # make sure forward and decode gives same result
            enc_dec = model(inputs["input_values"])[1]
            torch.testing.assert_close(decoded_outputs["audio_values"], enc_dec, rtol=1e-6, atol=1e-6)

    @parameterized.expand([(model_name,) for model_name in EXPECTED_INTEGRATION_BATCH.keys()])
    def test_integration_batch(self, model_name):
        expected = EXPECTED_INTEGRATION_BATCH[model_name]

        # load model and processor
        model_id = f"descript/{model_name}"
        model = DacModel.from_pretrained(model_id).to(torch_device)
        processor = AutoProcessor.from_pretrained(model_id)

        # load audio samples
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        audio_samples = [np.array([audio_sample["array"]])[0] for audio_sample in librispeech_dummy[-2:]["audio"]]

        # check on processor audio shape
        inputs = processor(
            raw_audio=audio_samples,
            sampling_rate=processor.sampling_rate,
            truncation=False,
            return_tensors="pt",
        ).to(torch_device)
        torch.equal(torch.tensor(inputs["input_values"].shape), torch.tensor(expected["preproc_shape"]))

        with torch.no_grad():
            # compare encoder loss
            encoder_outputs = model.encode(inputs["input_values"])
            torch.testing.assert_close(encoder_outputs[0].mean().item(), expected["enc_loss"], rtol=1e-3, atol=1e-3)

            # compare quantizer outputs
            expected_quant_codes = torch.tensor(expected["quant_codes"]).to(torch_device)
            quantizer_outputs = model.quantizer(encoder_outputs[1])
            torch.testing.assert_close(
                quantizer_outputs[1][..., : expected_quant_codes.shape[-1]],
                expected_quant_codes,
                rtol=1e-6,
                atol=1e-6,
            )
            torch.testing.assert_close(
                quantizer_outputs[4].mean().item(),
                expected["quant_codebook_loss"],
                rtol=1e-4,
                atol=1e-4,
            )

            # compare decoder outputs
            expected_dec_outputs = torch.tensor(expected["dec_outputs"]).to(torch_device)
            decoded_outputs = model.decode(encoder_outputs[1])
            torch.testing.assert_close(
                expected_dec_outputs,
                decoded_outputs["audio_values"][..., : expected_dec_outputs.shape[-1]],
                rtol=1e-3,
                atol=1e-3,
            )

            # compare codec error / lossiness
            codec_err = compute_rmse(decoded_outputs["audio_values"], inputs["input_values"])
            torch.testing.assert_close(codec_err, expected["codec_error"], rtol=1e-6, atol=1e-6)

            # make sure forward and decode gives same result
            enc_dec = model(inputs["input_values"])[1]
            torch.testing.assert_close(decoded_outputs["audio_values"], enc_dec, rtol=1e-6, atol=1e-6)

    @parameterized.expand([(model_name,) for model_name in EXPECTED_INTEGRATION_BATCH.keys()])
    def test_quantizer_from_latents_integration(self, model_name):
        model_id = f"descript/{model_name}"
        model = DacModel.from_pretrained(model_id).to(torch_device)
        processor = AutoProcessor.from_pretrained(model_id)

        # load audio sample
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        audio_sample = librispeech_dummy[0]["audio"]["array"]

        # check on processor audio shape
        inputs = processor(
            raw_audio=audio_sample,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
        ).to(torch_device)

        input_values = inputs["input_values"]
        with torch.no_grad():
            encoder_outputs = model.encode(input_values)
            latents = encoder_outputs.projected_latents
            original_quantizer_representation = encoder_outputs.quantized_representation

            # reconstruction using from_latents
            quantizer_representation, quantized_latents = model.quantizer.from_latents(latents=latents)
            reconstructed = model.decode(quantized_representation=quantizer_representation).audio_values

            # forward pass
            original_reconstructed = model(input_values).audio_values

        # ensure quantizer representations match
        self.assertTrue(
            torch.allclose(quantizer_representation, original_quantizer_representation, atol=1e-6),
            msg="Quantizer representation from from_latents should match original quantizer forward pass",
        )
        # ensure forward and decode are the same
        self.assertTrue(
            torch.allclose(reconstructed, original_reconstructed, atol=1e-6),
            msg="Reconstructed codes from latents should match original quantized codes",
        )
