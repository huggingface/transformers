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
import unittest

import numpy as np
from datasets import Audio, load_dataset
from parameterized import parameterized

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
    pipeline_model_mapping = {"text-embedding": DacModel} if is_torch_available() else {}

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

# fmt: off
# -- test_integration
EXPECTED_PREPROC_SHAPE = {
    "dac_16khz": torch.tensor([1, 1, 93760]),
    "dac_24khz": torch.tensor([1, 1, 140800]),
    "dac_44khz": torch.tensor([1, 1, 258560]),
}
EXPECTED_ENC_LOSS = {
    "dac_16khz": 24.888723373413086,
    "dac_24khz": 27.65193748474121,
    "dac_44khz": 23.874713897705078,
}
EXPECTED_QUANT_CODES = {
    "dac_16khz": torch.tensor([[[ 804,   25,  536,   52,  834,  867,  388,  653,  484,  706,  301,
           305,  752,   25,   40],
         [  77,  955,  134,  601,  162,  375,  967,   56,  684,  261,  871,
           552,  232,  341,  228],
         [ 355,  701,  172,  927,  785,  765,  790,  149,  117,  707,  511,
           226,  254,  883,  644],
         [ 184,   85,  828,   54,  154, 1007,  906,  253,  406, 1007,  302,
           577,  644,  330,  601],
         [ 763,  865,  586,  321,  966,  357,  911,  865,  234,  234,    6,
           630,    6,  174,  895],
         [ 454,  241,   67,  622,   41,  426,  749,  833,  639,  900,  372,
           481,  622,  418,  964],
         [ 203,  609,  730,  307,  874,  609,  318, 1011,  747,  949,  343,
           548,  657,  824,   21],
         [  82,   92,  692,   83,  421,  866,  483,  362,  596,  531,  853,
           121,  404,  512,  373],
         [1003,  260,  431,  460,   29,  927,   81,   76,  444,  298,  168,
           673,  466,  613,  383],
         [ 571,  203,  594,  394,   73,  560,  952,  437,  343,  992,  934,
           316,  497,  123,  305],
         [ 686,  715,  393,  635,  832,  716,  908,  384,   98,  873,   92,
           878,  592,  496,  104],
         [ 721,  502,  606,  204,  490,  428,  176,  395,  617,  323,  342,
           530,  226,    8,  600]]]
    ).to(torch_device),
    "dac_24khz": torch.tensor([[[ 252,  851,  919,  204,  239,  360,   90,  103,  851,  876,  160,
           160,  103,  234,  665],
         [ 908,  658,  479,  556,  847,  265,  496,   32,  847,  773,  623,
           375,    9,  497,  117],
         [ 385,  278,  221,  778,  408,  330,  562,  215,   80,   84,  320,
           728,  931,  470,  944],
         [ 383,  134,  271,  494,  179,  304,  150,  804,  788,  780,  356,
           416,  297,  903,  623],
         [ 487,  263,  414,  947,  608,  810,  140,   74,  372,  129,  417,
           592,  671,  479,  901],
         [ 692,  953,  508,  359,   85,  396,  545,  375,  382,  382,  511,
           382,  383,  643,  134],
         [ 652,  213,  210,  385,  326,  899,  341,  925,  908,   68,  216,
            21,  568, 1008,  635],
         [ 938,  848,  570,  515,  574,  693,  382,   71,   42,  742,  603,
           109,  193,  629,   79],
         [ 847,  101,  874,  894,  384,  832,  378,  658,    1,  487,  976,
           993,  932,  886,  860],
         [ 220,  344,  307,   69,  705,  974,  895,  438,    8,  806,  573,
           690,  543,  709,  303],
         [ 394,  594,  144,   10,  832,    4,  588,  659,  501,  218,  351,
           861,  915,  148,  141],
         [ 447,  763,  930,  894,  196,  668,  528,  862,   70,  598,  136,
           119,  395,  474, 1000],
         [ 677,  178,  637,  874,  471,  113,   23,  534,  333,    6,  821,
           777,  635,  932,  475],
         [ 932,  345,  436,  335,  555,  355,  103,  436,  277,  816,  400,
           356,   73,   23,  450],
         [ 592,  402,  177,   31,  693,  459,  442,  193,  615,  940,  927,
           917,  676,  327,  658],
         [ 192,  458,  540,  808,  626,  340,  290,  700,  190,  345,  381,
           137,  280,  611,  794],
         [ 834,    5,  522,  685,  146,  754,   37,  580,   78,    2, 1008,
           808,  281,  375,  366],
         [ 892,  790,  948,  662,  355,  437,  444,  790,  450,  850,  316,
           529,  385,  480,  178],
         [  36,  696,  125,  753,  143,  562,  368,  824,  491,  507,  892,
           880,  355,  152,  253],
         [ 934,  829,  457,  261,  668, 1014,  185,  464,   78,  332,  374,
           869,  530,   67,  884],
         [ 567,  914,  334,   38,  313,  744,    6,  210,  489,  867,  200,
           799,  540,  318,  706],
         [ 178,  882,  776,  992,  651,  800,  163,  470,  687,  906,  508,
           260,   36,  783,   64],
         [ 169,   66,  179,  711,  598,  938,  346,  251,  773,  108,  873,
           813,  479,  425,  669],
         [ 981,  692,  143,  589,  224,  282,   86,  712,  689,  907,  586,
           595,  444,  265,  198],
         [ 856,  540,  556,  302,  883,   96,  856,  560,  529,   91,  707,
           286,  142,  553,  252],
         [ 103,  868,  879,  779,  882,   34,  340,  603,  186,  808,  397,
           673,  919,  989,  626],
         [ 933,  215,  775,  747,  842,  836,  744,  272,  604,  202,  288,
           164,  242,  542,  207],
         [ 969,  373,  999,  524,  927,  879, 1017,   14,  526,  385,  478,
           690,  347,  589,   10],
         [ 716,  503,  781,  119,  176,  316,  212,  836,  850,   26,  685,
           973,  606,  796,  593],
         [ 164,  418,  929,  523,  571,  917,  364,  964,  480, 1021,    0,
           994,  876,  887,  379],
         [ 416,  957,  819,  478,  640,  479,  217,  842,  926,  771,  129,
           537,  899,  680,  547],
         [ 623,  596,  332,  517,  947,  376,  699,  918, 1012,  995,  858,
           516,   56,   43,  268]]]
    ).to(torch_device),
    "dac_44khz": torch.tensor([[[ 698,  315,  105,  315,  330,  105,  105,  698,  315,  481,  330,
            93,  629,  315,  105],
         [  30,  232,  249,  881,  962,  365,   56,  881,  186,  402,  311,
           521,  558,  778,  254],
         [1022,   22,  361,  491,  233,  419,  909,  456,  456,  471,  420,
           569,  455,  491,   16],
         [ 599,  143,  641,  352,   40,  556,  860,  780,  138,  137,  304,
           563,  863,  174,  370],
         [ 485,  350,  242,  555,  174,  581,  666,  744,  559,  810,  127,
           558,  453,   90,  124],
         [ 851,  423,  706,  178,   36,  564,  650,  539,  733,  720,   18,
           265,  619,  545,  581],
         [ 755,  891,  628,  674,  724,  764,  420,   51,  566,  315,  178,
           881,  461,  111,  675],
         [  52,  995,  512,  139,  538,  666, 1017,  868,  619,    0,  449,
          1005,  982,  106,  139],
         [ 357,  180,  368,  892,  856,  567,  960,  148,   36,  708,  945,
           285,  531,  331,  440]]]
    ).to(torch_device),
}
EXPECTED_DEC_OUTPUTS = {
    "dac_16khz": torch.tensor([[ 1.8940e-04,  6.8451e-04,  1.1393e-03,  1.4752e-03,  1.6592e-03,
          1.0343e-03,  3.7672e-04, -2.1513e-04, -3.7062e-04,  1.1900e-04,
          5.1029e-04,  1.1605e-03,  1.8881e-03,  2.8023e-03,  2.4951e-03,
          1.4668e-03,  1.4306e-03,  1.3172e-03,  9.2493e-04,  1.0286e-03,
          1.7709e-03,  2.5561e-03,  2.7497e-03,  3.1355e-03,  3.8951e-03,
          3.0081e-03,  2.1188e-03,  2.3982e-03,  1.9411e-03,  9.4039e-04,
          6.7362e-05,  6.3032e-05,  6.2965e-04,  1.4908e-03,  2.3690e-03,
          2.3852e-03,  1.6764e-03,  1.8238e-04, -7.1753e-04, -3.3184e-04,
          2.9475e-04,  5.3457e-04,  1.1068e-03,  2.2653e-03,  1.9302e-03,
          1.4867e-03,  1.4196e-03,  1.0963e-03,  4.4992e-04, -3.3099e-04]]).to(torch_device),
    "dac_24khz": torch.tensor([[ 1.6667e-04,  1.8821e-04, -2.7001e-05, -5.4563e-05,  2.2055e-04,
          8.4348e-04,  1.5988e-03,  1.6767e-03,  1.5461e-03,  1.4022e-03,
          1.1126e-03,  6.1560e-04, -1.2618e-04, -5.7430e-04, -1.7778e-04,
          6.1698e-04,  6.7644e-04,  1.7771e-04, -1.0049e-05,  3.2456e-04,
          6.4919e-04,  7.2769e-04,  6.4367e-04,  5.7299e-04,  1.1143e-03,
          1.9033e-03,  1.9752e-03,  1.2789e-03,  5.7600e-04,  3.9365e-04,
          3.9031e-04,  3.0397e-04,  3.8265e-04,  7.6303e-04,  1.3043e-03,
          1.7859e-03,  2.1733e-03,  2.5245e-03,  2.9150e-03,  3.0501e-03,
          2.7420e-03,  2.2311e-03,  1.8259e-03,  1.6864e-03,  1.6260e-03,
          1.0522e-03,  3.6211e-04,  2.8836e-04,  3.9427e-04,  2.4493e-04]]).to(torch_device),
    "dac_44khz": torch.tensor([[ 1.3247e-03,  1.4762e-03,  1.6968e-03,  1.8309e-03,  1.8860e-03,
          1.9468e-03,  1.9114e-03,  1.7796e-03,  1.5217e-03,  1.2451e-03,
          1.0056e-03,  8.3350e-04,  7.6910e-04,  7.7483e-04,  7.7547e-04,
          6.9667e-04,  5.2119e-04,  3.1329e-04,  1.6479e-04,  1.5293e-04,
          2.9349e-04,  5.4231e-04,  8.1284e-04,  1.0286e-03,  1.1453e-03,
          1.1638e-03,  1.1177e-03,  1.0757e-03,  1.0826e-03,  1.1571e-03,
          1.3236e-03,  1.5490e-03,  1.7671e-03,  1.9077e-03,  1.9214e-03,
          1.7885e-03,  1.5424e-03,  1.2386e-03,  9.3116e-04,  6.4010e-04,
          3.5748e-04,  8.3612e-05, -1.7643e-04, -4.0232e-04, -5.8362e-04,
         -7.0310e-04, -7.1898e-04, -5.8100e-04, -2.7705e-04,  1.6211e-04]]).to(torch_device),
}
EXPECTED_QUANT_CODEBOOK_LOSS = {
    "dac_16khz": 20.653074264526367,
    "dac_24khz": 22.438047409057617,
    "dac_44khz": 16.226943969726562,
}
EXPECTED_CODEC_ERROR = {
    "dac_16khz": 0.003834083443507552,
    "dac_24khz": 0.0025610385928303003,
    "dac_44khz": 0.000743341282941401,
}
# -- test_batch
EXPECTED_PREPROC_SHAPE_BATCH = {
    "dac_16khz": torch.tensor([2, 1, 113920]),
    "dac_24khz": torch.tensor([2, 1, 170880]),
    "dac_44khz": torch.tensor([2, 1, 313856]),
}
EXPECTED_ENC_LOSS_BATCH = {
    "dac_16khz": 20.345306396484375,
    "dac_24khz": 23.542919158935547,
    "dac_44khz": 19.58289909362793,
}
EXPECTED_QUANT_CODES_BATCH = {
    "dac_16khz": torch.tensor([[[ 490,  664,  726,  166,   55,  379,  367,  664,  661,  726,  592,
           301,  130,  198,  129],
         [1020,  734,   23,   53,  134,  648,  549,  589,  790, 1000,  420,
           271, 1021,  740,   36],
         [ 701,  344,  955,   19,  927,  212,  212,  667,  212,  627,  837,
           954,  777,  706,  496],
         [ 526,  805,  444,  474,  870,  920,  394,  823,  814, 1021,  319,
           677,  251,  485, 1021],
         [ 721,  134,  280,  439,  287,   77,  175,  902,  973,  412,  548,
           953,  130,   75,  543],
         [ 675,  316,  285,  341,  783,  850,  131,  487,  701,  150,  674,
           730,  900,  481,  498],
         [ 377,   37,  237,  489,   55,  246,  427,  456,  755, 1011,  171,
           631,  695,  576,  804],
         [ 601,  557,  681,   52,   10,  299,  284,  216,  869,  276,  907,
           364,  955,   41,  497],
         [ 465,  553,  697,   59,  701,  195,  335,  225,  896,  804,  240,
           928,  392,  192,  332],
         [ 807,  306,  977,  801,   77,  172,  760,  747,  445,   38,  395,
            31,  924,  724,  835],
         [ 903,  561,  205,  421,  231,  873,  931,  361,  679,  854,  248,
           884, 1011,  857,  248],
         [ 490,  993,  122,  787,  178,  307,  141,  468,  652,  786,  959,
           885,  226,  343,  501]],
        [[ 140,  320,  140,  489,  444,  320,  210,   73,  821, 1004,  388,
           686,  405,  563,  517],
         [ 725,  449,  715,   85,  761,  532,  620,   28,  620,  418,  146,
           532,  418,  453,  565],
         [ 695,  725,  994,  371,  829, 1008,  911,  927,  181,  707,  306,
           337,  254,  577,  857],
         [  51,  648,  474,  129,  781,  968,  737,  718,  400,  839,  674,
           689,  544,  767,  540],
         [1007,  234,  865,  966,  734,  748,   68,  454,  473,  973,  414,
           586,  618,    6,  612],
         [ 410,  566,  692,  756,  307, 1008,  269,  743,  549,  320,  303,
           729,  507,  741,  362],
         [ 172,  102,  959,  714,  292,  173,  149,  308,  307,  527,  844,
           102,  747,   76,  295],
         [ 656,  144,  994,  245,  686,  925,   48,  356,  126,  418,  112,
           674,  582,  916,  296],
         [ 776,  971,  967,  781,  174,  688,  817,  278,  937,  467,  352,
           463,  530,  804,  207],
         [1009,  284,  966,  907,  397,  875,  279,  643,  878,  315,  734,
           751,  337,  699,  382],
         [ 389,  748,   50,  585,   69,  565,  555,  931,  154,  443,   16,
           139,  905,  172,  496],
         [ 884,   34,  945, 1013,  212,  493,  724,  775,  356,  199,  728,
           552,  755,  223,  397]]]).to(torch_device),
    "dac_24khz": torch.tensor([[[ 234,  322,  826,  360,  204,  208,  766,  826,  458,  322,  919,
           999,  360,  772,  204],
         [ 780,  201,  229,  497,    9,  663, 1002,  243,  556,  300,  781,
           496,   77,  780,  781],
         [ 714,  342,  401,  553,  728,  196,  181,  109,  949,  528,  177,
           558,  180,    5,  197],
         [ 112,  408,  186,  933,  543,  829,  724, 1001,  425,   39,  163,
           517,  986,  348,  653],
         [1001,  207,  671,  551,  742,  231,  870,  577,  353, 1016,  497,
           282,  247,  126,   63],
         [ 924,   59,  799,  739,  771,  568,  280,  673,  639, 1002,   35,
           143,  270,  749,  571],
         [ 310,  982,  904,  666,  819,   67,  161,  373,  945,  871,  117,
           466,  388,  898,  584],
         [  69,  357,  188,  969,  213,  162,  376,   35,  638,  657,  676,
           991,  625,  833,  801],
         [ 333,  885,  343,  621,  752,  319,  292,  389,  947,  776,  958,
           585,  193,  834,  622],
         [ 958,  144,  680,  819,  303,  832,   56,  683,  366,  996,    8,
           784,  305,  621,   36],
         [ 561,  766,   69,  768,  219,  126,  945,  798,  568,  554,  539,
           245,   31,  384,  167],
         [ 727,  684,  371,  447,   50,  309,  407,  121,  839, 1019,  747,
           423,  604,  489,  738],
         [ 598,  490,  578,  353,  517,  283,  927,  432,  464,  608,  967,
            32,  240,  852,  326],
         [ 337,  226,  450,  862,  549,  799,  887,  925,  392,  841,  886,
           633,  351,    7,  386],
         [ 668,  497,  586,  937,  516,  898,  768, 1014,  420,  173,  855,
           602,  786,  940,   56],
         [ 575,  927,  322,  885,  367,  175,  691,  337,   21,  796,  595,
           826,  109,  604,   54],
         [  50,  854,  118,  231,  567,  332,  827,  422,  339,  958,  969,
            63,  992,  597,  428],
         [ 480,  619,  605,  598,  912, 1012,  365,  926,  538,  915,  644,
           675,  460,  667,  255],
         [ 578,  373,  355,   92,  920,  454,  979,  536,  645,  442,  247,
           956,  693,  457,  842],
         [1019,    0,  998,  958,  159,  159,  332,   94,  886,    1,  455,
           981,  418,  758,  358],
         [ 698,  843, 1008,  626,  776,  342,   53,  518,  636,  997,  956,
            36,  997,   12,  374],
         [ 904,  408,  802,  456,  645,  899,   15,  447,  857,  265,  258,
           983, 1018,  282,  607],
         [ 459,  467,  461,  358,  389,  792,  385,  678,   50,  888,  721,
             3,  792,  588,  972],
         [ 877,  180,  212,  656,   60,   73,  261,  644,  755,  496,  381,
           948,  879,  361,  863],
         [ 172,  588,  948,  452,  297, 1009,   49,  426,  853,  843,  896,
           957, 1008,  730,  860],
         [ 677,  125,  519,  975,  686,  404,  321,  310,   38,  138,  667,
           457,   98,  736, 1004],
         [ 784,  262,  289,  299, 1022,  170,  865,  869,  951,  839,  524,
           301,  828,   62,  511],
         [ 726,  693,  235,  208,  668,  777,  284,   61,  376,  203,  265,
           101,  344,  587,  736],
         [ 851,   83,  484,  951,  839,  180,  801,  525,  890,  373,   10,
           467,  524,  572,  614],
         [  48,  297,  674,  895,  740,  179,  782,  242,  721,  815,  238,
            74,  179,  650,  554],
         [ 336,  166,  203, 1021,   89,  991,  410,  518, 1019,  742,  718,
           810,  782,  623,  176],
         [ 110,  999,  360,  260,  278,  582,  921,  470,  242,  667,  757,
           463,  335,  566,  897]],
        [[ 851,  160,  851,  877,  665,  110,  581,  936,  826,  910,  110,
           110,  160,  103,  160],
         [ 325,  342,  722,  260,  549,  617,  508,    0,  965,  631,  846,
           446,  457,  124,   23],
         [ 529,  921,  767,  408,  628,  980,   80,  460,  980,  209,  768,
           255,  773,  759,  861],
         [ 344,  600,  255,  271,  402,  228,  805,  662,  497,   94,  852,
           337,  812,  140,  760],
         [ 415,  423,  322,  337,  599,  703,  520,  332,  811,  539,  511,
           511,  124,  110,  638],
         [ 514,  501,  660, 1014,  678,   77,  563,  793,  520,  464,  405,
            24,  630,  176,  692],
         [ 768,  497,  276,  353,  968,  214,  527,  447,  552,  746,  281,
           972,  681,  708,  907],
         [ 461,  802,   81,  411,  271,  186,  530,  670,  250, 1001,  828,
           270,  568,   74,  606],
         [ 539,  178,  451,  343,  235,  336,  346,  272,  291,  958,  924,
            91,  606,  408,  104],
         [ 668,  629,  817,  872,  526,  369,  889,  265,  580,  140,  229,
           240,  360,  811,  189],
         [ 973,  419,  164,  855,  767,  168,  378,  294,  350,   10,  610,
           297,  236,  976,  668],
         [ 162,  291,   66,   67,  749,  433,  428,  573,  209,  467,  202,
           838,  125,  452,  873],
         [   5,  949,  393,  322,  563,  679,  306,  467,   58,  326,  624,
            27,  447,  142,  965],
         [ 981,  105,  116,   51,  674,  584,  351,  824,  123,  320,  476,
           527,  668,  212,  944],
         [ 813,  156, 1013,  675,  964,  788,  137,  475,  906,  109,  400,
           899,  599,  820,  746],
         [ 398,   21,   63,  720,  304, 1017, 1009,  889,  704,  619,  684,
           571,  430,  642,   69],
         [ 405,  140,  531,  526,  657,  991,  624,   14,   45,  256,  300,
          1013,  255,  567,    0],
         [ 153,  469,   23,  553,  210,  812,  327,  778,  536,  406,   38,
           893,  974,  777,   58],
         [ 324,  399,    4,  563,  703,  499,  256,  136,  549,  164,  979,
           524,  975,  596,  520],
         [ 792,  511,  224,  225,  229,  424,  436,  124,  291,  267,  806,
             8,  657,  914,  808],
         [ 595,  491,  993,  961,  722,  756,  937,  585,   23,  991,  436,
           392,  464,  837,  604],
         [ 918,  647,  931,  658,  594,  677,  106,  963,  868,   92,  728,
           575,  302,  864,  930],
         [ 672,  685,  997,   36,  344,  956,  260,  365,  127,  348,  755,
           142,   65,  754,  284],
         [ 327,  987,  859,  525,  115,  551,  384,  289,  884,  669,   84,
           481,  193,  392,  246],
         [ 206,  432, 1018,  954,  534,  350,  902,  631,  459,  701,  913,
           408,  456,  135,  726],
         [ 483,  953,  684,  843,  478,  406,  931,  493,  386,  596,  459,
            34,  306,  140,   22],
         [ 508,  990,  988,  862,  265,  437,  277,  490,  633,  301,  759,
           759,  989,   85,  292],
         [ 586,  487,  860,  525,   90,  436,   15,  884,  727,  714,  697,
           180,  453,  279,  524],
         [ 639,  844,  513,  487,  853,  185,  690,  865,  562,  842,  439,
          1002,  468,  745,  298],
         [ 551,  764,  383,  422,  768,  760,  244,  177,  325,  567,  352,
           654,  579, 1019,  787],
         [ 207,  365,  766,  423,  792,  470,  582,  139,  363,  408,  573,
            19,  314,  471,  587],
         [ 776,  854,  529,  113,  927,  187,  362,  410,  596,  570,  559,
            61,  763,   83, 1015]]]).to(torch_device),
    "dac_44khz": torch.tensor([[[ 330,  315,  315,  619,  481,  315,  197,  315,  315,  105,  481,
           315,  481,  481,  481],
         [ 718, 1007,  929,    6,  906,  944,  402,  750,  675,  854,  336,
           426,  609,  356,  329],
         [ 417,  266,  697,  456,  300,  941,  325,  923, 1022,  605,  991,
             7,  939,  329,  456],
         [ 813,  811,  271,  148,  184,  838,  723,  497,  330,  922,   12,
           333,  918,  963,  285],
         [ 832,  307,  635,  794,  334,  114,   32,  505,  344,  170,  161,
           907,  193,  180,  585],
         [  91,  941,  912, 1001,  507,  486,  362, 1006,  228,  640,  760,
           215,  577,  633,  371],
         [ 676,   27,  903,  472,  473,  219,  860,  477,  969,  385,  533,
           911,  701,  241,  825],
         [ 326,  399,  116,  443,  605,  373,  534,  199,  748,  538,  516,
           983,  372,  565,  167],
         [ 776,  843,  185,  326,  723,  756,  318,   34,  818,  674,  728,
           554,  721,  369,  267]],
        [[ 578,  698,  330,  330,  330,  578,  330,  801,  330,  330,  330,
           330,  330,  330,  330],
         [ 171,  503,  725,  215,  814,  861,  139,  684,  880,  905,  937,
           418,  359,  190,  823],
         [ 141,  482,  780,  489,  845,  499,   59,  480,  296,   30,  631,
           540,  399,   23,  385],
         [ 402,  837,  216,  116,  535,  456, 1006,  969,  994,  125, 1011,
           285,  851,  832,  197],
         [  46,  950,  728,  645,  850,  839,  527,  850,   81,  205,  590,
           166,   22,  148,  402],
         [  98,  758,  474,  941,  217,  667,  681,  109,  719,  824,  162,
           160,  329,  627,  716],
         [ 999,  228,  752,  639,  404,  333,  993,  177,  888,  158,  644,
           221, 1011,  302,   79],
         [ 669,  535,  164,  665,  809,  798,  448,  800,  123,  936,  639,
           361,  353,  402,  160],
         [ 345,  355,  940,  261,   71,  946,  750,  120,  565,  164,  813,
           976,  946,   50,  516]]]).to(torch_device),
}
EXPECTED_DEC_OUTPUTS_BATCH = {
    "dac_16khz": torch.tensor([[-1.9496e-04,  1.8703e-04,  3.2085e-04,  2.1353e-04, -2.9954e-05,
         -3.3594e-04, -4.6374e-04, -4.3778e-04, -2.8602e-04,  2.7734e-04,
          8.8930e-04,  1.1189e-03,  1.6160e-03,  1.9375e-03,  1.7888e-03,
          5.9822e-04, -4.4124e-04, -1.3748e-03, -2.0023e-03, -2.0485e-03,
         -1.5615e-03, -4.1984e-04,  6.3778e-04,  1.2580e-03,  1.3390e-03,
          1.2830e-03,  5.9607e-04,  9.5532e-05, -6.1828e-04, -1.3873e-03,
         -1.4950e-03, -9.8374e-04, -3.8628e-04,  5.3108e-04,  1.8674e-03,
          2.3877e-03,  2.1173e-03,  1.4175e-03,  7.4522e-04, -2.4308e-04,
         -9.8757e-04, -1.3877e-03, -1.6685e-03, -1.0587e-03, -6.2359e-04,
         -5.2869e-04, -2.1441e-04,  4.1749e-04,  7.7953e-04,  7.9138e-04],
        [ 6.3088e-05,  3.4278e-04, -1.4322e-03, -2.2803e-04, -3.7853e-04,
         -1.3376e-03,  1.0602e-03, -1.4524e-03,  2.1785e-04, -3.2819e-04,
         -1.3297e-03,  4.8561e-04,  8.6668e-04, -1.7512e-03,  4.4856e-05,
          2.0326e-04, -2.9777e-03,  8.6695e-04,  1.3459e-03,  2.0098e-03,
         -5.5258e-04,  1.3641e-03, -4.5632e-05, -2.6290e-03, -6.7004e-04,
          6.1164e-04,  8.3981e-04, -1.6069e-03,  3.3123e-03,  1.3866e-03,
         -1.7855e-03, -3.5581e-05, -5.5376e-04, -9.3256e-04, -2.3831e-03,
         -5.4240e-04,  1.5906e-03, -1.3903e-03,  1.2177e-03,  6.1323e-04,
         -1.7830e-03,  3.3165e-05, -3.0913e-03,  4.9273e-04, -1.1230e-03,
          1.1301e-04,  3.3335e-03, -1.7503e-03,  5.2264e-04, -1.3666e-03]]).to(torch_device),
    "dac_24khz": torch.tensor([[ 2.6454e-04,  9.1854e-05, -4.1192e-04, -6.1339e-04, -5.8966e-04,
         -5.6627e-04, -5.2073e-04, -4.3783e-04, -1.5260e-04, -5.9512e-05,
         -7.9432e-05,  7.0958e-05,  8.1968e-05,  1.3918e-05,  2.0052e-04,
          4.1790e-04,  1.1061e-04, -1.7492e-04,  5.6043e-05,  4.1358e-04,
          4.5141e-04,  4.0811e-04,  4.1412e-04,  2.4054e-04,  2.5673e-04,
          4.4426e-04,  3.9844e-04,  1.3728e-04, -3.9132e-05, -2.7411e-04,
         -8.5156e-04, -1.4007e-03, -1.5820e-03, -1.5349e-03, -1.5199e-03,
         -1.4401e-03, -1.0491e-03, -5.1940e-04,  3.2038e-05,  5.5414e-04,
          8.9546e-04,  1.0130e-03,  1.0392e-03,  9.4535e-04,  6.9895e-04,
          3.2545e-04, -7.5281e-05, -3.8828e-04, -5.6601e-04, -7.2890e-04],
        [-4.8100e-04,  3.8518e-04,  4.0440e-04,  3.6149e-04,  1.4942e-03,
          1.2861e-03, -1.7561e-04, -7.2232e-05,  6.3749e-04, -1.1513e-03,
         -2.7382e-03, -1.5372e-03, -8.3539e-04, -1.6908e-03, -1.4055e-05,
          2.3753e-03, -2.4103e-04, -2.9636e-03,  3.0217e-04,  2.7415e-03,
         -3.6650e-04, -2.1928e-03, -3.5845e-04, -6.6671e-04, -2.0204e-03,
         -8.6126e-05,  5.4914e-04, -3.3885e-03, -3.9277e-03,  5.7712e-04,
          1.1305e-03, -1.0921e-03,  1.1022e-03,  2.9793e-03, -4.0440e-04,
         -1.8317e-03,  1.0773e-03,  2.3741e-04, -3.4544e-03, -2.0132e-03,
          5.8320e-04, -1.3169e-03, -1.3552e-03,  1.8405e-03,  4.7396e-04,
         -2.6800e-03, -1.6327e-05,  2.8485e-03,  1.2113e-04, -1.7437e-03]]).to(torch_device),
    "dac_44khz": torch.tensor([[-4.8096e-04, -2.2681e-04,  7.1221e-06,  1.6016e-04,  2.5950e-04,
          3.9612e-04,  5.2983e-04,  6.9538e-04,  8.0269e-04,  9.1193e-04,
          1.0201e-03,  1.0611e-03,  1.0619e-03,  1.0377e-03,  9.7943e-04,
          8.4063e-04,  6.4808e-04,  4.2628e-04,  1.9633e-04, -6.3365e-06,
         -1.6062e-04, -2.4469e-04, -2.7976e-04, -2.7269e-04, -2.9232e-04,
         -3.5925e-04, -4.6551e-04, -5.6719e-04, -6.5320e-04, -7.0686e-04,
         -7.1884e-04, -6.8951e-04, -6.1897e-04, -4.9569e-04, -3.2152e-04,
         -1.3526e-04,  2.5438e-05,  1.5100e-04,  2.6975e-04,  4.1167e-04,
          6.0325e-04,  8.1468e-04,  9.7458e-04,  1.0553e-03,  1.0614e-03,
          1.0112e-03,  9.2461e-04,  8.1784e-04,  6.9947e-04,  5.8702e-04],
        [ 7.1763e-04,  8.2121e-04,  3.6971e-04, -3.9159e-04, -8.7189e-04,
         -6.0987e-04,  2.0028e-04,  1.0584e-03,  1.3271e-03,  7.7182e-04,
         -3.3962e-04, -1.3513e-03, -1.6492e-03, -1.0778e-03,  2.4176e-05,
          9.7890e-04,  1.2389e-03,  7.5767e-04, -2.8469e-05, -4.7786e-04,
         -1.4488e-04,  8.8599e-04,  2.0020e-03,  2.4978e-03,  2.0565e-03,
          1.0179e-03,  1.4521e-04,  9.3082e-05,  7.8215e-04,  1.3156e-03,
          7.2480e-04, -1.1225e-03, -3.1963e-03, -3.9686e-03, -2.6454e-03,
         -1.5142e-05,  1.9848e-03,  1.7642e-03, -5.8603e-04, -3.2934e-03,
         -4.2989e-03, -2.8547e-03, -7.1620e-05,  2.0387e-03,  2.2099e-03,
          8.9436e-04, -3.5793e-04, -3.9508e-04,  6.0126e-04,  1.4234e-03]]).to(torch_device),
}
EXPECTED_QUANT_CODEBOOK_LOSS_BATCH = {
    "dac_16khz": 20.716419219970703,
    "dac_24khz": 23.65462875366211,
    "dac_44khz": 16.124454498291016,
}
EXPECTED_CODEC_ERROR_BATCH = {
    "dac_16khz": 0.001972666708752513,
    "dac_24khz": 0.001301625743508339,
    "dac_44khz": 0.00038262043381109834,
}
# fmt: on


@slow
@require_torch
class DacIntegrationTest(unittest.TestCase):
    @parameterized.expand([(model_name,) for model_name in EXPECTED_PREPROC_SHAPE.keys()])
    @require_deterministic_for_xpu
    def test_integration(self, model_name):
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
        torch.equal(torch.tensor(inputs["input_values"].shape), EXPECTED_PREPROC_SHAPE[model_name])

        with torch.no_grad():
            # compare encoder loss
            encoder_outputs = model.encode(inputs["input_values"])
            torch.testing.assert_close(
                encoder_outputs[0].squeeze().item(), EXPECTED_ENC_LOSS[model_name], rtol=1e-3, atol=1e-3
            )

            # compare quantizer outputs
            quantizer_outputs = model.quantizer(encoder_outputs[1])
            torch.testing.assert_close(
                quantizer_outputs[1][..., : EXPECTED_QUANT_CODES[model_name].shape[-1]],
                EXPECTED_QUANT_CODES[model_name],
                rtol=1e-6,
                atol=1e-6,
            )
            torch.testing.assert_close(
                quantizer_outputs[4].squeeze().item(), EXPECTED_QUANT_CODEBOOK_LOSS[model_name], rtol=1e-4, atol=1e-4
            )

            # compare decoder outputs
            decoded_outputs = model.decode(encoder_outputs[1])
            torch.testing.assert_close(
                decoded_outputs["audio_values"][..., : EXPECTED_DEC_OUTPUTS[model_name].shape[-1]],
                EXPECTED_DEC_OUTPUTS[model_name],
                rtol=1e-3,
                atol=1e-3,
            )

            # compare codec error / lossiness
            codec_err = compute_rmse(decoded_outputs["audio_values"], inputs["input_values"])
            torch.testing.assert_close(codec_err, EXPECTED_CODEC_ERROR[model_name], rtol=1e-5, atol=1e-5)

            # make sure forward and decode gives same result
            enc_dec = model(inputs["input_values"])[1]
            torch.testing.assert_close(decoded_outputs["audio_values"], enc_dec, rtol=1e-6, atol=1e-6)

    @parameterized.expand([(model_name,) for model_name in EXPECTED_PREPROC_SHAPE_BATCH.keys()])
    def test_integration_batch(self, model_name):
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
        torch.equal(torch.tensor(inputs["input_values"].shape), EXPECTED_PREPROC_SHAPE_BATCH[model_name])

        with torch.no_grad():
            # compare encoder loss
            encoder_outputs = model.encode(inputs["input_values"])
            torch.testing.assert_close(
                encoder_outputs[0].mean().item(), EXPECTED_ENC_LOSS_BATCH[model_name], rtol=1e-3, atol=1e-3
            )

            # compare quantizer outputs
            quantizer_outputs = model.quantizer(encoder_outputs[1])
            torch.testing.assert_close(
                quantizer_outputs[1][..., : EXPECTED_QUANT_CODES_BATCH[model_name].shape[-1]],
                EXPECTED_QUANT_CODES_BATCH[model_name],
                rtol=1e-6,
                atol=1e-6,
            )
            torch.testing.assert_close(
                quantizer_outputs[4].mean().item(),
                EXPECTED_QUANT_CODEBOOK_LOSS_BATCH[model_name],
                rtol=1e-4,
                atol=1e-4,
            )

            # compare decoder outputs
            decoded_outputs = model.decode(encoder_outputs[1])
            torch.testing.assert_close(
                EXPECTED_DEC_OUTPUTS_BATCH[model_name],
                decoded_outputs["audio_values"][..., : EXPECTED_DEC_OUTPUTS_BATCH[model_name].shape[-1]],
                rtol=1e-3,
                atol=1e-3,
            )

            # compare codec error / lossiness
            codec_err = compute_rmse(decoded_outputs["audio_values"], inputs["input_values"])
            torch.testing.assert_close(codec_err, EXPECTED_CODEC_ERROR_BATCH[model_name], rtol=1e-6, atol=1e-6)

            # make sure forward and decode gives same result
            enc_dec = model(inputs["input_values"])[1]
            torch.testing.assert_close(decoded_outputs["audio_values"], enc_dec, rtol=1e-6, atol=1e-6)

    @parameterized.expand([(model_name,) for model_name in EXPECTED_PREPROC_SHAPE_BATCH.keys()])
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
