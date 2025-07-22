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
import os
import tempfile
import unittest

import numpy as np
from datasets import Audio, load_dataset

from transformers import AutoProcessor, DacConfig, DacModel
from transformers.testing_utils import is_torch_available, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor
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
    test_pruning = False
    test_headmasking = False
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
            self, config_class=DacConfig, hidden_size=37, common_properties=[], has_text_modality=False
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
    def test_torchscript_output_attentions(self):
        pass

    @unittest.skip("The DacModel is not transformers based, thus it does not have the usual `hidden_states` logic")
    def test_torchscript_output_hidden_state(self):
        pass

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            return

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

    # Ignore copy
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = ["conv", "in_proj", "out_proj", "codebook"]
                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    def test_identity_shortcut(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        config.use_conv_shortcut = False
        self.model_tester.create_and_check_model_forward(config, inputs_dict)


def normalize(arr):
    norm = np.linalg.norm(arr)
    normalized_arr = arr / norm
    return normalized_arr


def compute_rmse(arr1, arr2):
    arr1_np = arr1.cpu().numpy().squeeze()
    arr2_np = arr2.cpu().numpy().squeeze()
    max_length = min(arr1.shape[-1], arr2.shape[-1])
    arr1_np = arr1_np[..., :max_length]
    arr2_np = arr2_np[..., :max_length]
    arr1_normalized = normalize(arr1_np)
    arr2_normalized = normalize(arr2_np)
    return np.sqrt(((arr1_normalized - arr2_normalized) ** 2).mean())


@slow
@require_torch
class DacIntegrationTest(unittest.TestCase):
    """
    Integration tests for DAC.

    Code for reproducing expected outputs can be found here:
    - Single file: https://gist.github.com/ebezzam/bb315efa7a416db6336a6b2a2d424ffa#file-dac_integration_single-py
    - Batched: https://gist.github.com/ebezzam/bb315efa7a416db6336a6b2a2d424ffa#file-dac_integration-py

    See https://github.com/huggingface/transformers/pull/39313 for reason behind large tolerance between for encoder
    and decoder outputs (1e-3). In summary, original model uses weight normalization, while Transformers does not. This
    leads to accumulating error. However, this does not affect the quantizer codes, thanks to discretization being
    robust to precision errors. Moreover, codec error is similar between Transformers and original.

    Moreover, here is a script to debug outputs and weights layer-by-layer:
    https://gist.github.com/ebezzam/bb315efa7a416db6336a6b2a2d424ffa#file-dac_layer_by_layer_debugging-py
    """

    def test_integration_16khz(self):
        model_name = "dac_16khz"

        # expected values
        EXPECTED_PREPROC_SHAPE = torch.tensor([1, 1, 93760])
        EXPECTED_ENC_LOSS = 24.84908103942871
        EXPECTED_QUANT_CODES = torch.tensor(
            [
                [
                    [804, 25, 977, 52, 68, 867, 388, 653, 315, 706, 301, 305, 140, 25, 40],
                    [77, 955, 532, 601, 431, 375, 967, 56, 54, 261, 871, 552, 735, 341, 228],
                    [355, 908, 77, 927, 617, 443, 790, 149, 403, 707, 511, 226, 995, 883, 644],
                    [184, 162, 611, 54, 211, 890, 906, 253, 677, 1007, 302, 577, 378, 330, 778],
                    [763, 322, 6, 321, 116, 228, 911, 865, 1000, 234, 6, 901, 10, 174, 895],
                    [454, 1, 622, 622, 487, 668, 749, 833, 382, 900, 372, 959, 232, 418, 964],
                    [203, 43, 173, 307, 961, 593, 318, 1011, 386, 949, 343, 899, 536, 824, 38],
                    [82, 810, 692, 83, 131, 866, 483, 362, 519, 531, 853, 121, 1010, 512, 710],
                    [1003, 691, 530, 460, 827, 903, 81, 76, 629, 298, 168, 177, 368, 613, 762],
                    [571, 752, 544, 394, 198, 479, 952, 437, 222, 992, 934, 316, 741, 123, 538],
                    [686, 421, 393, 635, 246, 330, 908, 384, 962, 873, 92, 254, 912, 496, 83],
                    [721, 977, 148, 204, 993, 660, 176, 395, 901, 323, 342, 849, 474, 8, 513],
                ]
            ]
        ).to(torch_device)
        # fmt: off
        EXPECTED_DEC_OUTPUTS = torch.tensor([[ 7.2661e-05,  5.9626e-04,  1.0609e-03,  1.4515e-03,  1.6704e-03,
            1.0837e-03,  4.6979e-04, -1.3811e-04, -2.7733e-04,  2.0613e-04,
            4.0715e-04,  8.4999e-04,  1.7112e-03,  2.7275e-03,  2.5560e-03,
            1.6202e-03,  1.4603e-03,  1.1447e-03,  7.4274e-04,  7.6758e-04,
            1.5931e-03,  2.5598e-03,  2.6844e-03,  2.9216e-03,  3.6430e-03,
            3.0532e-03,  2.1169e-03,  2.3657e-03,  2.0313e-03,  8.8282e-04,
            -1.6314e-04,  2.0697e-05,  9.0119e-04,  1.5815e-03,  2.1719e-03,
            2.2010e-03,  1.4089e-03, -9.8639e-05, -7.1111e-04, -2.1185e-04,
            3.3837e-04,  5.2177e-04,  1.0538e-03,  2.2637e-03,  1.9972e-03,
            1.6396e-03,  1.6282e-03,  1.1689e-03,  2.7550e-04, -4.4859e-04]]).to(torch_device)
        # fmt: on
        EXPECTED_QUANT_CODEBOOK_LOSS = 20.5806350708007
        EXPECTED_CODEC_ERROR = 0.0038341842591762543

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
        torch.equal(torch.tensor(inputs["input_values"].shape), EXPECTED_PREPROC_SHAPE)

        with torch.no_grad():
            # compare encoder loss
            encoder_outputs = model.encode(inputs["input_values"])
            torch.testing.assert_close(EXPECTED_ENC_LOSS, encoder_outputs[0].squeeze().item(), rtol=1e-3, atol=1e-3)

            # compare quantizer outputs
            quantizer_outputs = model.quantizer(encoder_outputs[1])
            torch.testing.assert_close(
                EXPECTED_QUANT_CODES, quantizer_outputs[1][..., : EXPECTED_QUANT_CODES.shape[-1]], rtol=1e-6, atol=1e-6
            )
            torch.testing.assert_close(
                EXPECTED_QUANT_CODEBOOK_LOSS, quantizer_outputs[4].squeeze().item(), rtol=1e-6, atol=1e-6
            )

            # compare decoder outputs
            decoded_outputs = model.decode(encoder_outputs[1])
            torch.testing.assert_close(
                EXPECTED_DEC_OUTPUTS,
                decoded_outputs["audio_values"][..., : EXPECTED_DEC_OUTPUTS.shape[-1]],
                rtol=1e-3,
                atol=1e-3,
            )

            # compare codec error / lossiness
            codec_err = compute_rmse(decoded_outputs["audio_values"], inputs["input_values"])
            torch.testing.assert_close(EXPECTED_CODEC_ERROR, codec_err, rtol=1e-6, atol=1e-6)

            # make sure forward and decode gives same result
            enc_dec = model(inputs["input_values"])[1]
            torch.testing.assert_close(decoded_outputs["audio_values"], enc_dec, rtol=1e-6, atol=1e-6)

    def test_integration_24khz(self):
        model_name = "dac_24khz"

        # expected values
        EXPECTED_PREPROC_SHAPE = torch.tensor([1, 1, 140800])
        EXPECTED_ENC_LOSS = 28.112096786499023
        EXPECTED_QUANT_CODES = torch.tensor(
            [
                [
                    [160, 360, 826, 204, 239, 360, 90, 160, 851, 234, 252, 690, 360, 160, 665],
                    [189, 496, 717, 74, 847, 692, 496, 549, 847, 78, 669, 440, 9, 243, 117],
                    [497, 562, 161, 827, 408, 330, 562, 152, 80, 84, 320, 745, 1023, 544, 944],
                    [261, 140, 271, 843, 179, 239, 150, 211, 788, 343, 333, 760, 217, 243, 623],
                    [487, 846, 919, 947, 417, 787, 140, 186, 567, 129, 633, 328, 927, 932, 901],
                    [862, 953, 929, 184, 85, 433, 545, 672, 382, 666, 694, 382, 572, 38, 134],
                    [835, 260, 975, 144, 621, 800, 341, 1017, 28, 889, 521, 287, 805, 231, 474],
                    [470, 803, 475, 208, 574, 679, 382, 71, 413, 79, 571, 330, 408, 759, 79],
                    [452, 272, 257, 101, 76, 540, 378, 933, 83, 350, 334, 539, 808, 975, 860],
                    [450, 704, 839, 811, 705, 304, 895, 340, 979, 53, 573, 80, 241, 110, 571],
                    [801, 523, 138, 939, 729, 417, 588, 9, 501, 304, 820, 271, 497, 719, 141],
                    [579, 741, 42, 811, 561, 630, 528, 945, 1009, 637, 109, 702, 1005, 911, 748],
                    [96, 581, 853, 817, 256, 592, 23, 1014, 309, 3, 846, 780, 704, 481, 138],
                    [162, 193, 808, 498, 128, 949, 103, 928, 277, 599, 375, 718, 893, 388, 532],
                    [318, 498, 5, 696, 953, 1018, 442, 97, 573, 179, 850, 353, 548, 1002, 279],
                    [962, 911, 712, 684, 214, 240, 290, 467, 812, 588, 232, 588, 922, 101, 768],
                    [969, 785, 514, 168, 106, 423, 37, 683, 882, 657, 516, 819, 535, 50, 988],
                    [299, 914, 787, 584, 582, 449, 444, 366, 666, 721, 1022, 1015, 700, 752, 710],
                    [926, 669, 287, 618, 806, 309, 368, 502, 704, 573, 319, 562, 355, 994, 873],
                    [513, 75, 447, 290, 16, 370, 185, 43, 1015, 346, 450, 24, 490, 299, 231],
                    [616, 506, 867, 444, 648, 987, 6, 301, 556, 128, 898, 352, 657, 616, 798],
                    [382, 353, 420, 424, 107, 256, 163, 113, 832, 247, 415, 541, 893, 922, 918],
                    [135, 775, 363, 14, 603, 311, 346, 722, 746, 207, 695, 48, 821, 428, 53],
                    [626, 72, 220, 524, 256, 736, 86, 64, 618, 780, 607, 799, 734, 506, 868],
                    [310, 913, 13, 707, 177, 19, 856, 463, 400, 141, 959, 904, 910, 818, 734],
                    [948, 105, 835, 842, 802, 117, 340, 466, 774, 726, 389, 599, 558, 491, 420],
                    [916, 440, 167, 177, 842, 450, 744, 820, 906, 739, 702, 158, 745, 546, 636],
                    [135, 675, 544, 64, 955, 904, 1017, 862, 167, 564, 362, 1023, 774, 78, 914],
                    [216, 218, 494, 28, 605, 962, 212, 649, 249, 710, 83, 94, 437, 613, 54],
                    [611, 109, 743, 56, 493, 294, 364, 514, 980, 524, 474, 978, 35, 724, 767],
                    [719, 752, 343, 171, 776, 414, 217, 656, 717, 73, 955, 516, 582, 559, 241],
                    [821, 641, 740, 272, 468, 847, 699, 842, 20, 330, 216, 703, 581, 306, 137],
                ]
            ]
        ).to(torch_device)
        # fmt: off
        EXPECTED_DEC_OUTPUTS = torch.tensor([[ 4.2660e-04,  4.0129e-04,  1.5403e-04,  5.0874e-05,  2.9436e-04,
            1.0682e-03,  1.9777e-03,  1.9081e-03,  1.5145e-03,  1.2959e-03,
            1.1858e-03,  8.6308e-04,  7.6199e-05, -6.2039e-04, -2.8909e-04,
            7.2902e-04,  9.6803e-04,  3.5680e-04, -1.4637e-04,  7.8926e-05,
            7.9285e-04,  1.3313e-03,  1.1692e-03,  5.7410e-04,  7.0640e-04,
            1.5462e-03,  1.9182e-03,  1.3498e-03,  5.0153e-04,  1.5142e-04,
            2.1018e-04,  4.2771e-04,  7.4621e-04,  1.1082e-03,  1.5289e-03,
            1.9526e-03,  2.3434e-03,  2.6424e-03,  2.8369e-03,  2.7632e-03,
            2.3256e-03,  1.8973e-03,  1.8191e-03,  1.9133e-03,  1.7674e-03,
            1.0398e-03,  2.6915e-04,  1.3725e-04,  2.8598e-04,  2.5875e-04]]).to(torch_device)
        # fmt: on
        EXPECTED_QUANT_CODEBOOK_LOSS = 22.581758499145508
        EXPECTED_CODEC_ERROR = 0.002570481738075614

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
        torch.equal(torch.tensor(inputs["input_values"].shape), EXPECTED_PREPROC_SHAPE)

        with torch.no_grad():
            # compare encoder loss
            encoder_outputs = model.encode(inputs["input_values"])
            torch.testing.assert_close(EXPECTED_ENC_LOSS, encoder_outputs[0].squeeze().item(), rtol=1e-3, atol=1e-3)

            # compare quantizer outputs
            quantizer_outputs = model.quantizer(encoder_outputs[1])
            torch.testing.assert_close(
                EXPECTED_QUANT_CODES, quantizer_outputs[1][..., : EXPECTED_QUANT_CODES.shape[-1]], rtol=1e-6, atol=1e-6
            )
            torch.testing.assert_close(
                EXPECTED_QUANT_CODEBOOK_LOSS, quantizer_outputs[4].squeeze().item(), rtol=1e-6, atol=1e-6
            )

            # compare decoder outputs
            decoded_outputs = model.decode(encoder_outputs[1])
            torch.testing.assert_close(
                EXPECTED_DEC_OUTPUTS,
                decoded_outputs["audio_values"][..., : EXPECTED_DEC_OUTPUTS.shape[-1]],
                rtol=1e-3,
                atol=1e-3,
            )

            # compare codec error / lossiness
            codec_err = compute_rmse(decoded_outputs["audio_values"], inputs["input_values"])
            torch.testing.assert_close(EXPECTED_CODEC_ERROR, codec_err, rtol=1e-6, atol=1e-6)

            # make sure forward and decode gives same result
            enc_dec = model(inputs["input_values"])[1]
            torch.testing.assert_close(decoded_outputs["audio_values"], enc_dec, rtol=1e-6, atol=1e-6)

    def test_integration_44khz(self):
        model_name = "dac_44khz"

        # expected values
        EXPECTED_PREPROC_SHAPE = torch.tensor([1, 1, 258560])
        EXPECTED_ENC_LOSS = 23.78483772277832
        EXPECTED_QUANT_CODES = torch.tensor(
            [
                [
                    [332, 315, 105, 315, 616, 105, 494, 698, 315, 481, 330, 93, 105, 315, 105],
                    [670, 350, 249, 27, 232, 365, 311, 881, 186, 402, 311, 521, 527, 778, 254],
                    [569, 300, 361, 530, 1002, 419, 285, 501, 456, 471, 180, 615, 419, 491, 764],
                    [605, 436, 641, 291, 901, 556, 715, 780, 502, 410, 858, 125, 562, 174, 746],
                    [854, 706, 242, 294, 346, 88, 527, 961, 559, 664, 314, 963, 278, 90, 682],
                    [175, 152, 706, 884, 986, 457, 567, 176, 49, 535, 851, 417, 533, 349, 779],
                    [913, 710, 628, 162, 770, 254, 247, 6, 397, 264, 233, 704, 577, 111, 916],
                    [999, 693, 512, 884, 38, 223, 29, 744, 497, 123, 972, 120, 47, 301, 90],
                    [490, 163, 368, 507, 253, 283, 745, 65, 295, 935, 811, 587, 801, 255, 105],
                ]
            ]
        ).to(torch_device)
        # fmt: off
        EXPECTED_DEC_OUTPUTS = torch.tensor([[ 8.3748e-04,  3.7760e-04,  4.7135e-04,  8.2829e-04,  1.3677e-03,
            1.7487e-03,  1.8883e-03,  1.7437e-03,  1.4828e-03,  1.2284e-03,
            1.0894e-03,  1.0442e-03,  1.0558e-03,  1.0136e-03,  8.4781e-04,
            4.8677e-04, -2.0375e-05, -5.2144e-04, -8.6839e-04, -9.8977e-04,
            -8.0130e-04, -3.6122e-04,  1.8086e-04,  6.4340e-04,  9.1103e-04,
            9.6243e-04,  8.6814e-04,  7.7186e-04,  7.5613e-04,  8.1264e-04,
            9.0747e-04,  9.5464e-04,  9.5436e-04,  8.7902e-04,  7.6080e-04,
            6.2870e-04,  5.5878e-04,  5.7444e-04,  6.6622e-04,  7.9741e-04,
            8.7610e-04,  8.4571e-04,  6.7909e-04,  4.2059e-04,  1.5131e-04,
            -7.1465e-05, -1.8646e-04, -1.8300e-04, -1.2542e-04, -7.1933e-05]]).to(torch_device)
        # fmt: on
        EXPECTED_QUANT_CODEBOOK_LOSS = 16.2640438079834
        EXPECTED_CODEC_ERROR = 0.0007429996621794999

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
        torch.equal(torch.tensor(inputs["input_values"].shape), EXPECTED_PREPROC_SHAPE)

        with torch.no_grad():
            # compare encoder loss
            encoder_outputs = model.encode(inputs["input_values"])
            torch.testing.assert_close(EXPECTED_ENC_LOSS, encoder_outputs[0].squeeze().item(), rtol=1e-3, atol=1e-3)

            # compare quantizer outputs
            quantizer_outputs = model.quantizer(encoder_outputs[1])
            torch.testing.assert_close(
                EXPECTED_QUANT_CODES, quantizer_outputs[1][..., : EXPECTED_QUANT_CODES.shape[-1]], rtol=1e-6, atol=1e-6
            )
            torch.testing.assert_close(
                EXPECTED_QUANT_CODEBOOK_LOSS, quantizer_outputs[4].squeeze().item(), rtol=1e-6, atol=1e-6
            )

            # compare decoder outputs
            decoded_outputs = model.decode(encoder_outputs[1])
            torch.testing.assert_close(
                EXPECTED_DEC_OUTPUTS,
                decoded_outputs["audio_values"][..., : EXPECTED_DEC_OUTPUTS.shape[-1]],
                rtol=1e-3,
                atol=1e-3,
            )

            # compare codec error / lossiness
            codec_err = compute_rmse(decoded_outputs["audio_values"], inputs["input_values"])
            torch.testing.assert_close(EXPECTED_CODEC_ERROR, codec_err, rtol=1e-6, atol=1e-6)

            # make sure forward and decode gives same result
            enc_dec = model(inputs["input_values"])[1]
            torch.testing.assert_close(decoded_outputs["audio_values"], enc_dec, rtol=1e-6, atol=1e-6)

    def test_integration_batch_16khz(self):
        model_name = "dac_16khz"

        # expected values
        EXPECTED_PREPROC_SHAPE = torch.tensor([2, 1, 113920])
        EXPECTED_ENC_LOSS = 20.370271682739258
        EXPECTED_QUANT_CODES = torch.tensor(
            [
                [
                    [490, 664, 726, 166, 55, 379, 367, 664, 661, 726, 592, 301, 130, 198, 129],
                    [1020, 734, 23, 53, 134, 648, 549, 589, 790, 1000, 449, 271, 1021, 740, 36],
                    [701, 344, 955, 19, 927, 212, 212, 667, 212, 627, 453, 954, 777, 706, 496],
                    [526, 805, 444, 474, 870, 920, 394, 823, 814, 1021, 763, 677, 251, 485, 1021],
                    [721, 134, 280, 439, 287, 77, 175, 902, 973, 412, 739, 953, 130, 75, 543],
                    [675, 316, 285, 341, 783, 850, 131, 487, 701, 150, 749, 730, 900, 481, 498],
                    [377, 37, 237, 489, 55, 246, 427, 456, 755, 1011, 712, 631, 695, 576, 804],
                    [601, 557, 681, 52, 10, 299, 284, 216, 869, 276, 424, 364, 955, 41, 497],
                    [465, 553, 697, 59, 701, 195, 335, 225, 896, 804, 776, 928, 392, 192, 332],
                    [807, 306, 977, 801, 77, 172, 760, 747, 445, 38, 731, 31, 924, 724, 835],
                    [903, 561, 205, 421, 231, 873, 931, 361, 679, 854, 471, 884, 1011, 857, 248],
                    [490, 993, 122, 787, 178, 307, 141, 468, 652, 786, 879, 885, 226, 343, 501],
                ],
                [
                    [140, 320, 210, 489, 444, 388, 210, 73, 821, 1004, 388, 686, 405, 563, 407],
                    [725, 449, 802, 85, 36, 532, 620, 28, 620, 418, 146, 532, 418, 453, 565],
                    [695, 725, 600, 371, 829, 237, 911, 927, 181, 707, 306, 337, 254, 577, 289],
                    [51, 648, 186, 129, 781, 570, 737, 563, 400, 839, 674, 689, 544, 767, 577],
                    [1007, 234, 145, 966, 734, 748, 68, 272, 473, 973, 414, 586, 618, 6, 909],
                    [410, 566, 507, 756, 943, 736, 269, 349, 549, 320, 303, 729, 507, 741, 76],
                    [172, 102, 548, 714, 225, 723, 149, 423, 307, 527, 844, 102, 747, 76, 586],
                    [656, 144, 407, 245, 140, 409, 48, 197, 126, 418, 112, 674, 582, 916, 223],
                    [776, 971, 291, 781, 833, 296, 817, 261, 937, 467, 352, 463, 530, 804, 683],
                    [1009, 284, 427, 907, 900, 630, 279, 285, 878, 315, 734, 751, 337, 699, 966],
                    [389, 748, 203, 585, 609, 474, 555, 64, 154, 443, 16, 139, 905, 172, 86],
                    [884, 34, 477, 1013, 335, 306, 724, 202, 356, 199, 728, 552, 755, 223, 371],
                ],
            ]
        ).to(torch_device)
        # fmt: off
        EXPECTED_DEC_OUTPUTS = torch.tensor([[-1.9181e-04,  1.9380e-04,  3.1524e-04,  2.0670e-04, -2.8026e-05,
            -3.3014e-04, -4.6584e-04, -4.3935e-04, -2.8362e-04,  2.7245e-04,
            8.8112e-04,  1.1195e-03,  1.6224e-03,  1.9368e-03,  1.7803e-03,
            5.9601e-04, -4.4178e-04, -1.3736e-03, -1.9979e-03, -2.0477e-03,
            -1.5583e-03, -4.1277e-04,  6.2742e-04,  1.2409e-03,  1.3380e-03,
            1.2884e-03,  6.0346e-04,  8.9812e-05, -6.1626e-04, -1.3760e-03,
            -1.4970e-03, -9.8225e-04, -3.9102e-04,  5.3190e-04,  1.8696e-03,
            2.3731e-03,  2.1139e-03,  1.4220e-03,  7.3644e-04, -2.4944e-04,
            -9.8294e-04, -1.3858e-03, -1.6684e-03, -1.0482e-03, -6.1834e-04,
            -5.3312e-04, -2.1345e-04,  4.1917e-04,  7.7653e-04,  8.0206e-04],
            [ 3.1081e-05,  4.7076e-04, -1.5066e-03, -1.7006e-05, -3.3131e-04,
            -1.1786e-03,  8.2880e-04, -1.2492e-03,  4.6135e-04, -8.7780e-04,
            -8.5493e-04,  3.2979e-04,  1.1218e-03, -1.8018e-03,  2.2795e-04,
            2.4981e-04, -3.1100e-03,  1.0356e-03,  1.1427e-03,  2.1378e-03,
            -7.0038e-04,  1.6522e-03, -3.3599e-04, -2.3893e-03, -5.2286e-04,
            2.9462e-04,  1.2429e-03, -1.8078e-03,  3.3687e-03,  1.3336e-03,
            -1.5815e-03, -1.5836e-04, -5.4054e-04, -7.2660e-04, -2.2980e-03,
            -5.3254e-04,  1.4890e-03, -1.0853e-03,  1.0333e-03,  8.1283e-04,
            -1.6996e-03,  6.0168e-05, -2.6916e-03,  3.7072e-04, -1.0729e-03,
            2.7891e-04,  3.3514e-03, -1.8029e-03,  5.5011e-04, -1.1905e-03]]).to(torch_device)
            # fmt: on
        EXPECTED_QUANT_CODEBOOK_LOSS = 20.61562156677246
        EXPECTED_CODEC_ERROR = 0.001973195234313607

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
        torch.equal(torch.tensor(inputs["input_values"].shape), EXPECTED_PREPROC_SHAPE)

        with torch.no_grad():
            # compare encoder loss
            encoder_outputs = model.encode(inputs["input_values"])
            torch.testing.assert_close(EXPECTED_ENC_LOSS, encoder_outputs[0].mean().item(), rtol=1e-3, atol=1e-3)

            # compare quantizer outputs
            quantizer_outputs = model.quantizer(encoder_outputs[1])
            torch.testing.assert_close(
                EXPECTED_QUANT_CODES, quantizer_outputs[1][..., : EXPECTED_QUANT_CODES.shape[-1]], rtol=1e-6, atol=1e-6
            )
            torch.testing.assert_close(
                EXPECTED_QUANT_CODEBOOK_LOSS, quantizer_outputs[4].mean().item(), rtol=1e-6, atol=1e-6
            )

            # compare decoder outputs
            decoded_outputs = model.decode(encoder_outputs[1])
            torch.testing.assert_close(
                EXPECTED_DEC_OUTPUTS,
                decoded_outputs["audio_values"][..., : EXPECTED_DEC_OUTPUTS.shape[-1]],
                rtol=1e-3,
                atol=1e-3,
            )

            # compare codec error / lossiness
            codec_err = compute_rmse(decoded_outputs["audio_values"], inputs["input_values"])
            torch.testing.assert_close(EXPECTED_CODEC_ERROR, codec_err, rtol=1e-6, atol=1e-6)

            # make sure forward and decode gives same result
            enc_dec = model(inputs["input_values"])[1]
            torch.testing.assert_close(decoded_outputs["audio_values"], enc_dec, rtol=1e-6, atol=1e-6)

    def test_integration_batch_24khz(self):
        model_name = "dac_24khz"

        # expected values
        EXPECTED_PREPROC_SHAPE = torch.tensor([2, 1, 170880])
        EXPECTED_ENC_LOSS = 24.505210876464844
        EXPECTED_QUANT_CODES = torch.tensor(
            [
                [
                    [234, 826, 826, 360, 204, 716, 766, 766, 360, 252, 919, 999, 360, 772, 668],
                    [117, 496, 229, 267, 9, 663, 1002, 629, 756, 372, 781, 496, 23, 780, 781],
                    [559, 712, 401, 423, 290, 27, 674, 340, 762, 410, 877, 558, 516, 5, 197],
                    [914, 8, 186, 766, 622, 547, 724, 101, 355, 634, 252, 517, 986, 348, 449],
                    [636, 148, 671, 232, 374, 24, 925, 118, 561, 760, 748, 964, 117, 126, 589],
                    [950, 825, 985, 600, 771, 949, 24, 629, 284, 398, 361, 893, 345, 840, 721],
                    [18, 263, 904, 778, 348, 839, 603, 447, 468, 117, 840, 631, 574, 898, 711],
                    [455, 359, 188, 148, 878, 246, 376, 509, 906, 759, 799, 991, 797, 833, 116],
                    [786, 275, 343, 492, 578, 952, 854, 833, 720, 730, 949, 72, 630, 305, 943],
                    [476, 696, 254, 283, 913, 407, 45, 408, 387, 904, 207, 206, 931, 621, 115],
                    [517, 73, 1019, 268, 238, 754, 188, 670, 923, 930, 110, 992, 870, 210, 953],
                    [311, 31, 371, 819, 949, 52, 650, 557, 573, 388, 222, 510, 908, 343, 559],
                    [405, 355, 520, 986, 179, 171, 49, 349, 706, 16, 439, 700, 704, 852, 759],
                    [854, 745, 982, 727, 466, 71, 530, 23, 125, 639, 254, 450, 397, 171, 766],
                    [863, 439, 415, 421, 463, 789, 551, 717, 641, 161, 882, 246, 576, 238, 464],
                    [331, 416, 322, 794, 416, 187, 689, 880, 29, 570, 283, 92, 310, 327, 748],
                    [149, 338, 105, 63, 848, 995, 824, 497, 792, 375, 745, 321, 914, 597, 101],
                    [588, 361, 77, 311, 483, 461, 889, 132, 724, 352, 187, 338, 72, 235, 761],
                    [434, 882, 522, 153, 462, 62, 725, 265, 597, 9, 161, 613, 576, 654, 1006],
                    [697, 927, 617, 1011, 561, 19, 181, 402, 830, 318, 248, 521, 645, 386, 111],
                    [787, 604, 809, 223, 21, 569, 817, 550, 253, 484, 718, 292, 358, 704, 556],
                    [821, 935, 743, 973, 982, 801, 799, 614, 988, 186, 337, 606, 166, 488, 116],
                    [789, 555, 32, 57, 671, 538, 712, 732, 524, 52, 869, 646, 91, 766, 516],
                    [481, 31, 464, 774, 756, 612, 619, 771, 372, 615, 697, 337, 28, 891, 706],
                    [293, 676, 468, 515, 777, 479, 625, 882, 725, 975, 491, 599, 594, 563, 235],
                    [170, 373, 462, 102, 335, 616, 880, 542, 989, 68, 154, 918, 716, 897, 33],
                    [228, 480, 610, 886, 733, 16, 924, 366, 490, 417, 790, 909, 88, 344, 351],
                    [243, 987, 683, 814, 104, 47, 173, 591, 376, 570, 181, 556, 955, 771, 464],
                    [1010, 62, 490, 536, 440, 174, 263, 849, 934, 544, 231, 908, 586, 558, 670],
                    [757, 604, 828, 519, 968, 862, 62, 182, 971, 627, 655, 518, 153, 666, 903],
                    [720, 192, 470, 262, 404, 920, 755, 138, 614, 245, 458, 182, 920, 398, 761],
                    [570, 527, 276, 994, 124, 174, 561, 150, 139, 988, 935, 327, 174, 1020, 383],
                ],
                [
                    [851, 110, 668, 103, 826, 360, 919, 160, 826, 160, 204, 110, 360, 910, 160],
                    [325, 846, 245, 722, 664, 594, 1002, 130, 859, 261, 260, 496, 846, 146, 23],
                    [529, 465, 354, 408, 597, 710, 450, 460, 980, 1011, 577, 392, 631, 453, 861],
                    [344, 645, 255, 327, 101, 1017, 474, 296, 513, 903, 363, 823, 85, 83, 760],
                    [415, 208, 656, 878, 751, 798, 240, 326, 137, 393, 511, 253, 369, 110, 590],
                    [514, 639, 623, 632, 163, 77, 911, 168, 811, 314, 928, 365, 886, 571, 692],
                    [768, 700, 408, 359, 937, 540, 1018, 570, 401, 746, 541, 166, 813, 492, 659],
                    [141, 802, 880, 55, 557, 13, 440, 550, 250, 640, 92, 691, 671, 266, 707],
                    [539, 706, 445, 343, 984, 280, 667, 414, 525, 987, 272, 727, 247, 834, 383],
                    [668, 94, 376, 890, 975, 337, 178, 839, 449, 863, 980, 35, 929, 913, 661],
                    [489, 430, 874, 230, 318, 714, 732, 491, 460, 681, 897, 124, 653, 990, 203],
                    [352, 625, 110, 636, 618, 691, 976, 249, 165, 584, 92, 487, 940, 907, 83],
                    [168, 518, 471, 139, 693, 101, 761, 185, 415, 338, 330, 557, 1013, 530, 163],
                    [282, 355, 539, 464, 725, 808, 607, 691, 374, 502, 898, 960, 822, 680, 233],
                    [599, 15, 236, 918, 475, 45, 16, 631, 409, 662, 961, 868, 589, 820, 943],
                    [398, 238, 897, 395, 502, 972, 125, 219, 748, 1000, 310, 664, 371, 867, 163],
                    [415, 685, 758, 452, 615, 491, 298, 645, 180, 659, 137, 895, 158, 780, 803],
                    [14, 138, 789, 848, 203, 360, 66, 589, 842, 597, 296, 763, 157, 259, 176],
                    [432, 65, 342, 488, 399, 259, 869, 214, 490, 975, 349, 894, 691, 87, 850],
                    [20, 524, 1019, 333, 926, 632, 41, 1002, 75, 282, 319, 426, 513, 368, 241],
                    [252, 292, 705, 578, 937, 800, 861, 548, 732, 57, 914, 493, 415, 76, 626],
                    [1004, 799, 467, 438, 656, 397, 547, 882, 873, 675, 900, 360, 941, 25, 63],
                    [695, 7, 446, 799, 900, 821, 859, 760, 740, 398, 236, 936, 974, 305, 27],
                    [977, 58, 979, 294, 514, 525, 768, 381, 920, 147, 264, 675, 6, 318, 619],
                    [539, 315, 574, 938, 208, 454, 869, 220, 1007, 964, 906, 133, 247, 14, 357],
                    [555, 968, 337, 468, 767, 805, 991, 266, 620, 653, 882, 720, 592, 920, 1016],
                    [320, 824, 133, 631, 861, 176, 607, 5, 686, 187, 186, 982, 453, 479, 849],
                    [247, 191, 164, 884, 292, 289, 579, 996, 332, 480, 965, 856, 628, 522, 652],
                    [142, 388, 533, 548, 600, 1, 504, 663, 140, 246, 1, 80, 555, 739, 672],
                    [909, 361, 285, 925, 509, 358, 219, 725, 476, 626, 651, 511, 3, 456, 620],
                    [731, 421, 150, 573, 598, 936, 796, 57, 442, 821, 162, 359, 912, 139, 659],
                    [588, 398, 945, 404, 804, 494, 572, 124, 47, 809, 775, 266, 9, 596, 435],
                ],
            ]
        ).to(torch_device)
        # fmt: off
        EXPECTED_DEC_OUTPUTS = torch.tensor([[ 2.9611e-04,  5.0039e-05, -5.4961e-04, -7.9769e-04, -6.9696e-04,
            -5.6013e-04, -4.7665e-04, -3.8039e-04, -6.8090e-05,  6.5704e-05,
            1.3205e-05,  1.3519e-04,  1.4002e-04,  4.3348e-05,  2.9029e-04,
            5.1533e-04,  1.4072e-04, -1.8430e-04,  6.3313e-05,  4.6729e-04,
            5.5076e-04,  5.6079e-04,  5.6557e-04,  3.2839e-04,  2.6326e-04,
            3.9028e-04,  3.1820e-04,  5.1251e-05, -7.0745e-05, -2.0471e-04,
            -7.0736e-04, -1.2458e-03, -1.4124e-03, -1.3991e-03, -1.4890e-03,
            -1.4013e-03, -1.0092e-03, -5.4982e-04, -3.5847e-05,  5.3150e-04,
            9.2390e-04,  1.0131e-03,  1.0362e-03,  1.0253e-03,  8.1528e-04,
            3.7854e-04, -1.3280e-05, -2.6982e-04, -4.8256e-04, -7.0810e-04],
            [-4.3881e-04,  3.3771e-04,  1.0076e-03,  1.2748e-03,  1.4132e-03,
            1.0326e-03,  7.5779e-04,  5.3942e-04, -2.8545e-04, -2.0953e-03,
            -2.2058e-03,  1.1152e-04,  5.6744e-04, -1.7912e-03, -1.4614e-03,
            1.8420e-03,  1.5202e-03, -1.0541e-03,  1.9058e-04,  1.3378e-03,
            -2.0335e-03, -2.5633e-03,  2.4959e-03,  2.4356e-03, -3.1333e-03,
            -2.8208e-03,  9.7969e-04, -1.0972e-03, -3.0217e-03,  4.1109e-04,
            2.3006e-04, -2.8686e-03,  1.2978e-03,  5.9192e-03,  7.3619e-04,
            -3.9734e-03, -2.6965e-04,  1.3701e-03, -1.7230e-03, -9.4332e-04,
            4.2128e-04, -2.6123e-03, -1.8240e-03,  3.3554e-03,  1.7732e-03,
            -3.2838e-03, -8.2577e-04,  3.1959e-03,  1.1458e-03, -2.4608e-04]]).to(torch_device)
        # fmt: on
        EXPECTED_QUANT_CODEBOOK_LOSS = 23.9102783203125
        EXPECTED_CODEC_ERROR = 0.0012980918399989605

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
        torch.equal(torch.tensor(inputs["input_values"].shape), EXPECTED_PREPROC_SHAPE)

        with torch.no_grad():
            # compare encoder loss
            encoder_outputs = model.encode(inputs["input_values"])
            torch.testing.assert_close(EXPECTED_ENC_LOSS, encoder_outputs[0].mean().item(), rtol=1e-3, atol=1e-3)

            # compare quantizer outputs
            quantizer_outputs = model.quantizer(encoder_outputs[1])
            torch.testing.assert_close(
                EXPECTED_QUANT_CODES, quantizer_outputs[1][..., : EXPECTED_QUANT_CODES.shape[-1]], rtol=1e-6, atol=1e-6
            )
            torch.testing.assert_close(
                EXPECTED_QUANT_CODEBOOK_LOSS, quantizer_outputs[4].mean().item(), rtol=1e-6, atol=1e-6
            )

            # compare decoder outputs
            decoded_outputs = model.decode(encoder_outputs[1])
            torch.testing.assert_close(
                EXPECTED_DEC_OUTPUTS,
                decoded_outputs["audio_values"][..., : EXPECTED_DEC_OUTPUTS.shape[-1]],
                rtol=1e-3,
                atol=1e-3,
            )

            # compare codec error / lossiness
            codec_err = compute_rmse(decoded_outputs["audio_values"], inputs["input_values"])
            torch.testing.assert_close(EXPECTED_CODEC_ERROR, codec_err, rtol=1e-6, atol=1e-6)

            # make sure forward and decode gives same result
            enc_dec = model(inputs["input_values"])[1]
            torch.testing.assert_close(decoded_outputs["audio_values"], enc_dec, rtol=1e-6, atol=1e-6)

    def test_integration_batch_44khz(self):
        model_name = "dac_44khz"

        # expected values
        EXPECTED_PREPROC_SHAPE = torch.tensor([2, 1, 313856])
        EXPECTED_ENC_LOSS = 19.557754516601562
        EXPECTED_QUANT_CODES = torch.tensor(
            [
                [
                    [330, 315, 315, 619, 481, 315, 197, 315, 315, 105, 481, 481, 481, 481, 481],
                    [718, 1007, 309, 6, 906, 35, 402, 750, 396, 854, 962, 115, 609, 224, 329],
                    [417, 266, 150, 335, 300, 812, 325, 780, 1022, 605, 480, 342, 939, 150, 456],
                    [813, 811, 897, 334, 200, 852, 723, 497, 678, 922, 396, 333, 918, 548, 285],
                    [832, 315, 165, 106, 902, 326, 32, 572, 610, 170, 395, 223, 193, 807, 585],
                    [91, 941, 81, 684, 34, 340, 362, 946, 157, 640, 888, 215, 577, 483, 371],
                    [676, 859, 446, 664, 473, 815, 860, 640, 514, 385, 73, 201, 701, 78, 825],
                    [326, 426, 347, 970, 605, 997, 534, 111, 559, 538, 526, 208, 372, 709, 167],
                    [776, 315, 179, 232, 140, 456, 318, 155, 191, 674, 105, 992, 721, 406, 267],
                ],
                [
                    [578, 592, 330, 330, 330, 330, 330, 801, 330, 330, 330, 698, 330, 330, 330],
                    [501, 204, 514, 215, 615, 580, 567, 684, 478, 905, 208, 32, 495, 84, 1000],
                    [141, 458, 489, 125, 691, 471, 522, 60, 978, 30, 125, 480, 424, 67, 1],
                    [908, 192, 865, 878, 137, 698, 965, 969, 565, 216, 535, 488, 441, 503, 181],
                    [850, 635, 993, 391, 500, 122, 365, 850, 905, 449, 586, 451, 840, 811, 797],
                    [307, 408, 497, 294, 24, 396, 417, 922, 161, 268, 100, 753, 778, 1014, 259],
                    [178, 918, 568, 28, 187, 375, 301, 889, 834, 406, 665, 7, 889, 909, 387],
                    [935, 566, 315, 13, 490, 37, 436, 801, 484, 62, 476, 551, 557, 232, 533],
                    [1017, 89, 585, 401, 13, 238, 744, 1017, 774, 872, 850, 468, 640, 833, 854],
                ],
            ]
        ).to(torch_device)
        # fmt: off
        EXPECTED_DEC_OUTPUTS = torch.tensor([[-3.7834e-04, -1.0849e-04,  1.1856e-04,  2.6852e-04,  3.7313e-04,
            5.0301e-04,  6.4261e-04,  8.0797e-04,  9.0969e-04,  9.9720e-04,
            1.0807e-03,  1.1217e-03,  1.1229e-03,  1.1208e-03,  1.0862e-03,
            9.5098e-04,  7.5477e-04,  5.2319e-04,  2.7449e-04,  2.4389e-05,
            -1.9138e-04, -3.2046e-04, -4.0629e-04, -4.4804e-04, -5.0271e-04,
            -5.8324e-04, -6.6573e-04, -6.9545e-04, -6.8046e-04, -6.1640e-04,
            -5.3542e-04, -4.2302e-04, -3.0829e-04, -1.8475e-04, -3.9555e-05,
            9.0104e-05,  1.9291e-04,  2.7445e-04,  3.6738e-04,  4.7454e-04,
            6.0626e-04,  7.5514e-04,  8.5390e-04,  8.8749e-04,  8.5473e-04,
            7.5550e-04,  6.2329e-04,  4.9771e-04,  3.8809e-04,  3.0741e-04],
            [ 1.1130e-04,  4.6536e-04,  1.0524e-04, -6.1460e-04, -1.1777e-03,
            -1.0661e-03, -3.7962e-04,  5.3627e-04,  1.0481e-03,  8.7734e-04,
            1.3513e-04, -6.6297e-04, -9.5284e-04, -4.6333e-04,  5.5780e-04,
            1.4526e-03,  1.6264e-03,  1.0852e-03,  3.3766e-04,  1.0960e-04,
            7.7973e-04,  2.0579e-03,  3.0206e-03,  2.9674e-03,  1.8141e-03,
            3.1059e-04, -5.7140e-04, -3.4386e-04,  4.8406e-04,  8.6931e-04,
            2.1745e-05, -1.7647e-03, -3.2787e-03, -3.3368e-03, -1.7466e-03,
            4.3745e-04,  1.6595e-03,  1.1171e-03, -6.3018e-04, -2.0979e-03,
            -2.1286e-03, -6.8752e-04,  1.1514e-03,  2.1590e-03,  1.9204e-03,
            1.0659e-03,  5.3295e-04,  6.6817e-04,  9.2716e-04,  5.3240e-04]]).to(torch_device)
        # fmt: on
        EXPECTED_QUANT_CODEBOOK_LOSS = 16.177066802978516
        EXPECTED_CODEC_ERROR = 0.00037737112143076956

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
        torch.equal(torch.tensor(inputs["input_values"].shape), EXPECTED_PREPROC_SHAPE)

        with torch.no_grad():
            # compare encoder loss
            encoder_outputs = model.encode(inputs["input_values"])
            torch.testing.assert_close(EXPECTED_ENC_LOSS, encoder_outputs[0].mean().item(), rtol=1e-3, atol=1e-3)

            # compare quantizer outputs
            quantizer_outputs = model.quantizer(encoder_outputs[1])
            torch.testing.assert_close(
                EXPECTED_QUANT_CODES, quantizer_outputs[1][..., : EXPECTED_QUANT_CODES.shape[-1]], rtol=1e-6, atol=1e-6
            )
            torch.testing.assert_close(
                EXPECTED_QUANT_CODEBOOK_LOSS, quantizer_outputs[4].mean().item(), rtol=1e-6, atol=1e-6
            )

            # compare decoder outputs
            decoded_outputs = model.decode(encoder_outputs[1])
            torch.testing.assert_close(
                EXPECTED_DEC_OUTPUTS,
                decoded_outputs["audio_values"][..., : EXPECTED_DEC_OUTPUTS.shape[-1]],
                rtol=1e-3,
                atol=1e-3,
            )

            # compare codec error / lossiness
            codec_err = compute_rmse(decoded_outputs["audio_values"], inputs["input_values"])
            torch.testing.assert_close(EXPECTED_CODEC_ERROR, codec_err, rtol=1e-6, atol=1e-6)

            # make sure forward and decode gives same result
            enc_dec = model(inputs["input_values"])[1]
            torch.testing.assert_close(decoded_outputs["audio_values"], enc_dec, rtol=1e-6, atol=1e-6)
