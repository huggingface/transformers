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

# fmt: off
# -- test_integration
EXPECTED_PREPROC_SHAPE = {
    "dac_16khz": torch.tensor([1, 1, 93760]),
    "dac_24khz": torch.tensor([1, 1, 140800]),
    "dac_44khz": torch.tensor([1, 1, 258560]),
}
EXPECTED_ENC_LOSS = {
    "dac_16khz": 24.876689910888672,
    "dac_24khz": 27.683147430419922,
    "dac_44khz": 23.88699722290039,
}
EXPECTED_QUANT_CODES = {
    "dac_16khz": torch.tensor([[[ 804,   25,  536,   52,   68,  867,  388,  653,  315,  706,  301,
           305,  752,   25,   40],
         [ 955,  955,  134,  601,  431,  375,  967,   56,   54,  261,  871,
           552,  232,  341,  228],
         [ 977,  701,  172,  927,  617,  765,  790,  149,  403,  707,  511,
           226,  254,  883,  644],
         [ 467,   85,  828,   54,  211, 1007,  906,  253,  677, 1007,  302,
           577,  644,  330,  778],
         [ 189,  865,  586,  321,  116,  357,  911,  865, 1000,  234,    6,
           901,    6,  470,  895],
         [ 454,  241,   67,  622,  487,  426,  749,  833,  382,  900,  372,
           959,  622,  305,  964],
         [ 175,  609,  730,  307,  961,  609,  318, 1011,  386,  949,  343,
           899,  657,  609,   38],
         [  82,   92,  692,   83,  131,  866,  483,  362,  519,  531,  853,
           121,  404,  224,  710],
         [1003,  260,  431,  460,  827,  927,   81,   76,  629,  298,  168,
           177,  466,  741,  762],
         [ 196,  203,  594,  394,  198,  560,  952,  437,  222,  992,  934,
           316,  497,   31,  538],
         [ 129,  715,  393,  635,  246,  716,  908,  384,  962,  873,   92,
           254,  592,  496,   83],
         [ 257,  502,  606,  204,  993,  428,  176,  395,  901,  323,  342,
           849,  226,  453,  513]]]
    ).to(torch_device),
    "dac_24khz": torch.tensor([[[ 252,  851,  919,  204,  239,  360,  160,  103,  851,  876,  160,
           793,  103,  234,  665],
         [ 908,  658,  479,  556,  847,  738,  395,  124,  847,  496,  623,
            77,    9,  497,  117],
         [ 385,  278,  221, 1020,  408,  330,   70,  215,   80,   84,  320,
           998,  931,  470,  944],
         [ 383,  259,  271,  348,  179,  304,  634,  282,  788,  542,  356,
           760,  297,  903,  623],
         [ 487,  159,  414,  947,  608,  685,  101,   74,  372,  823,  417,
           866,  671,  589,  901],
         [ 692,  175,  508,   54,   85,  383,  787,  629,  844,    7,  511,
           382,  383,  643,  134],
         [ 652,  895,  846,  766,  326,  640,  852,  365,  887,  126,  216,
           224,  568, 1008,  635],
         [ 938,  285,  570,  515,  574,  515,  862,  644,  845,  207,  603,
           830,  193,  158,   79],
         [ 847,  825,  874,  991,  384,  509, 1008,  308,  579,  487,  976,
           651,  932,  692,  860],
         [ 220,  392,  307,  397,  705,  876,  273,  438,  411,  449,  573,
           393,  543,  709,  303],
         [ 394,  773,  144,  254,  832,  586,  790,  941,  501,  502,  351,
           907,  915,  148,  141],
         [ 447,  985,  930,  175,  196,  854,  968,  494,  899,  637,  136,
           937,  395,  364, 1000],
         [ 677,  690,  428,  756,  471,  225,  763,  124,  333,   23,  821,
           644,  635,  130,  475],
         [ 932,  589,  436,  548,  555,   53,  466,  280,  598,  689,  400,
           194,   73,  619,  450],
         [ 592,  402,  177,  731,  693,  618,  871,  177,  761,  222,  927,
           986,  676,  197,  658],
         [ 192,  560,  368,  729,  626,  656,  174,  271,  383,  345,  381,
           567,  467,  970,  794],
         [ 834,   92,  990,  380,  146,  286,  644,  929,  173,  292, 1008,
           948,  281,  973,  366],
         [ 892,  533,  350,  589,  355,  163,  561,  229,  655,  240,  316,
           926,  385,  624,  178],
         [  36,  385,  589,  342,  143,  517,  648,   94,  457,  217,  892,
            60,  355,   46,  253],
         [ 934,  939,  457,    5,  668,  323,  312,  825,  448,  697,  374,
           199,   98,  955,  884],
         [ 567,  297,   40,  498,  313,   86,  832,  270,   21,  609,  200,
           688,  168,  616,  706],
         [ 178,  559,  922,  627,  651,   19,  589,  475,  312,  898,  508,
           969,   36,  783,   64],
         [ 169,  981,   86,    4,  598,  988,  670,  480,   68,  235,  873,
           130,  479,  543,  669],
         [ 981,  575,  827,  149,  224,  572,  470,  265,  504,  654,  586,
           835,  444,  497,  198],
         [ 856,  913,  658,  664,  883,  771,  646,   56,  440,  482,  707,
           229,  864,  286,  252],
         [ 103,  568,   68,  904,  882,  239,   67,  112,  941,  457,  397,
           412,  634, 1018,  626],
         [ 933,  908,   96,  316,  842,  842,  241,  600,  504,  765,  288,
           520,  312,  847,  207],
         [ 969,  255,  492,  868,  927,  951,  170,  607,  720,  234,  478,
           482,  119,  376,   10],
         [ 716,  727,  375,  904,  176,  667,  729,  590,  391,  364,  685,
           975,  186,  195,  593],
         [ 164,  923,  485,  139,  571,  968,  718,  305,   62,  828,    0,
           177,  827,  368,  379],
         [ 416,  151,   83,  822,  640,  414,  969,  128,  667,  297,  129,
           907,  938,  142,  547],
         [ 623,  263,  408,  922,  947,  916,  705,  475,  360,   68,  858,
           679,  601,  737,  268]]]
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
    "dac_16khz": torch.tensor([[ 2.8727e-04,  7.5153e-04,  1.1655e-03,  1.4741e-03,  1.7629e-03,
          1.2686e-03,  6.8926e-04,  9.9498e-05, -1.2343e-04,  9.0252e-05,
          4.6127e-04,  1.2541e-03,  2.1357e-03,  3.2235e-03,  2.7988e-03,
          1.6963e-03,  1.6913e-03,  1.8761e-03,  1.5943e-03,  1.6859e-03,
          1.9988e-03,  2.0419e-03,  2.1781e-03,  2.8659e-03,  4.0749e-03,
          3.3388e-03,  2.6116e-03,  2.6826e-03,  1.5272e-03,  4.6868e-04,
          2.7379e-05,  4.6056e-04,  1.0661e-03,  1.6188e-03,  2.4102e-03,
          2.1339e-03,  1.2963e-03,  4.1525e-04,  1.2672e-04,  1.5717e-04,
         -1.8527e-04, -1.8147e-04,  4.4626e-04,  1.7926e-03,  2.0294e-03,
          2.0505e-03,  1.9239e-03,  1.2841e-03,  2.8655e-04, -7.2832e-04]]).to(torch_device),
    "dac_24khz": torch.tensor([[ 2.2079e-04,  2.4399e-04,  4.3078e-06, -7.5914e-05,  1.4212e-04,
          7.6769e-04,  1.6150e-03,  1.7733e-03,  1.6170e-03,  1.3906e-03,
          1.0458e-03,  5.5205e-04, -1.3534e-04, -5.3017e-04, -1.3089e-04,
          6.0571e-04,  6.0875e-04,  9.8797e-05, -4.1255e-05,  3.9135e-04,
          7.6663e-04,  7.9177e-04,  5.9919e-04,  4.7265e-04,  1.0604e-03,
          1.9382e-03,  2.0611e-03,  1.3375e-03,  5.6962e-04,  3.6642e-04,
          4.1219e-04,  3.6118e-04,  3.8400e-04,  6.6042e-04,  1.1633e-03,
          1.7126e-03,  2.2300e-03,  2.6731e-03,  3.0229e-03,  3.0357e-03,
          2.6539e-03,  2.1634e-03,  1.8445e-03,  1.7679e-03,  1.6686e-03,
          1.0201e-03,  3.2503e-04,  3.0262e-04,  4.5186e-04,  3.1867e-04]]).to(torch_device),
    "dac_44khz": torch.tensor([[ 1.3242e-03,  1.4757e-03,  1.6963e-03,  1.8309e-03,  1.8861e-03,
          1.9472e-03,  1.9124e-03,  1.7809e-03,  1.5221e-03,  1.2444e-03,
          1.0050e-03,  8.3391e-04,  7.7057e-04,  7.7590e-04,  7.7512e-04,
          6.9593e-04,  5.2162e-04,  3.1479e-04,  1.6659e-04,  1.5419e-04,
          2.9440e-04,  5.4320e-04,  8.1338e-04,  1.0288e-03,  1.1451e-03,
          1.1634e-03,  1.1173e-03,  1.0755e-03,  1.0824e-03,  1.1565e-03,
          1.3220e-03,  1.5470e-03,  1.7653e-03,  1.9072e-03,  1.9218e-03,
          1.7886e-03,  1.5425e-03,  1.2397e-03,  9.3327e-04,  6.4261e-04,
          3.5974e-04,  8.4780e-05, -1.7630e-04, -4.0178e-04, -5.8225e-04,
         -7.0167e-04, -7.1810e-04, -5.8053e-04, -2.7654e-04,  1.6288e-04]]).to(torch_device),
}
EXPECTED_QUANT_CODEBOOK_LOSS = {
    "dac_16khz": 20.72989845275879,
    "dac_24khz": 22.66023063659668,
    "dac_44khz": 16.216812133789062,
}
EXPECTED_CODEC_ERROR = {
    "dac_16khz": 0.0038333097472786903,
    "dac_24khz": 0.002560485154390335,
    "dac_44khz": 0.0007433565333485603,
}
# -- test_batch
EXPECTED_PREPROC_SHAPE_BATCH = {
    "dac_16khz": torch.tensor([2, 1, 113920]),
    "dac_24khz": torch.tensor([2, 1, 170880]),
    "dac_44khz": torch.tensor([2, 1, 313856]),
}
EXPECTED_ENC_LOSS_BATCH = {
    "dac_16khz": 20.375171661376953,
    "dac_24khz": 23.56614112854004,
    "dac_44khz": 19.585765838623047,
}
EXPECTED_QUANT_CODES_BATCH = {
    "dac_16khz": torch.tensor([[[ 490,  664,  726,  166,   55,  379,  367,  664,  661,  726,  592,
           301,  130,  198,  129],
         [1020,  734,   23,   53,  134,  648,  549,  589,  790, 1000,  449,
           271, 1021,  740,   36],
         [ 701,  344,  955,   19,  927,  212,  212,  667,  212,  627,  453,
           954,  777,  706,  496],
         [ 526,  805,  444,  474,  870,  920,  394,  823,  814, 1021,  763,
           677,  251,  485, 1021],
         [ 721,  134,  280,  439,  287,   77,  175,  902,  973,  412,  739,
           953,  130,   75,  543],
         [ 675,  316,  285,  341,  783,  850,  131,  487,  701,  150,  749,
           730,  900,  481,  498],
         [ 377,   37,  237,  489,   55,  246,  427,  456,  755, 1011,  712,
           631,  695,  576,  804],
         [ 601,  557,  681,   52,   10,  299,  284,  216,  869,  276,  424,
           364,  955,   41,  497],
         [ 465,  553,  697,   59,  701,  195,  335,  225,  896,  804,  776,
           928,  392,  192,  332],
         [ 807,  306,  977,  801,   77,  172,  760,  747,  445,   38,  731,
            31,  924,  724,  835],
         [ 903,  561,  205,  421,  231,  873,  931,  361,  679,  854,  471,
           884, 1011,  857,  248],
         [ 490,  993,  122,  787,  178,  307,  141,  468,  652,  786,  879,
           885,  226,  343,  501]],
        [[ 140,  320,  210,  489,  444,  320,  210,   73,  821, 1004,  388,
           686,  405,  563,  407],
         [ 725,  449,  802,   85,   36,  532,  620,   28,  620,  418,  146,
           532,  418,  453,  565],
         [ 695,  725,  600,  371,  829, 1008,  911,  927,  181,  707,  306,
           337,  254,  577,  289],
         [  51,  648,  186,  129,  781,  968,  737,  563,  400,  839,  674,
           689,  544,  767,  577],
         [1007,  234,  145,  966,  734,  748,   68,  272,  473,  973,  414,
           586,  618,    6,  909],
         [ 410,  566,  507,  756,  943, 1008,  269,  349,  549,  320,  303,
           729,  507,  741,   76],
         [ 172,  102,  548,  714,  225,  173,  149,  423,  307,  527,  844,
           102,  747,   76,  586],
         [ 656,  144,  407,  245,  140,  925,   48,  197,  126,  418,  112,
           674,  582,  916,  223],
         [ 776,  971,  291,  781,  833,  688,  817,  261,  937,  467,  352,
           463,  530,  804,  683],
         [1009,  284,  427,  907,  900,  875,  279,  285,  878,  315,  734,
           751,  337,  699,  966],
         [ 389,  748,  203,  585,  609,  565,  555,   64,  154,  443,   16,
           139,  905,  172,   86],
         [ 884,   34,  477, 1013,  335,  493,  724,  202,  356,  199,  728,
           552,  755,  223,  371]]]).to(torch_device),
    "dac_24khz": torch.tensor([[[ 234,  322,  826,  360,  204,  208,  766,  826,  458,  322,  919,
           999,  360,  772,  204],
         [ 117,  201,  229,  497,    9,  663, 1002,  243,  556,  300,  781,
           496,   77,  780,  781],
         [ 554,  342,  401,  553,  728,  196,  181,  109,  949,  528,   39,
           558,  180,    5,  197],
         [ 112,  408,  186,  933,  543,  829,  724, 1001,  425,   39,  163,
           517,  986,  348,  653],
         [  88,  207,  671,  551,  742,  231,  870,  577,  353, 1016,  259,
           282,  247,  126,   63],
         [ 924,   59,  799,  739,  771,  568,  280,  673,  639, 1002,   35,
           143,  270,  749,  571],
         [ 214,  982,  904,  666,  819,   67,  161,  373,  945,  871,  597,
           466,  388,  898,  584],
         [ 696,  357,  188,  969,  213,  162,  376,   35,  638,  657,  731,
           991,  625,  833,  801],
         [ 559,  885,  343,  621,  752,  319,  292,  389,  947,  776,   78,
           585,  193,  834,  622],
         [ 642,  144,  680,  819,  303,  832,   56,  683,  366,  996,  609,
           784,  305,  621,   36],
         [ 517,  766,   69,  768,  219,  126,  945,  798,  568,  554,  115,
           245,   31,  384,  167],
         [ 424,  684,  371,  447,   50,  309,  407,  121,  839, 1019,  816,
           423,  604,  489,  738],
         [ 274,  490,  578,  353,  517,  283,  927,  432,  464,  608,  927,
            32,  240,  852,  326],
         [ 737,  226,  450,  862,  549,  799,  887,  925,  392,  841,  539,
           633,  351,    7,  386],
         [ 624,  497,  586,  937,  516,  898,  768,  188,  420,  173,  116,
           602,  786,  940,   56],
         [ 430,  927,  322,  885,  367,  175,  691,  337,   21,  796,  317,
           826,  109,  604,   54],
         [ 917,  854,  118,  231,  567,  332,  827,  422,  339,  958,  529,
            63,  992,  597,  428],
         [ 468,  619,  605,  598,  912, 1012,  365,   60,  538,  915,   22,
           675,  460,  667,  255],
         [ 912,  373,  355,   92,  920,  454,  979,  414,  645,  442,  783,
           956,  693,  457,  842],
         [ 230,    0,  998,  958,  159,  159,  332,   94,  886,    1,  455,
           981,  418,  758,  358],
         [ 132,  843, 1008,  626,  776,  342,   53,  362,  636,  997,   22,
            36,  997,   12,  374],
         [ 135,  408,  802,  456,  645,  899,   15,  447,  857,  265,  185,
           983, 1018,  282,  607],
         [ 272,  467,  461,  358,  389,  792,  385,  339,   50,  888,   63,
             3,  792,  588,  972],
         [ 179,  180,  212,  656,   60,   73,  261,  644,  755,  496,  137,
           948,  879,  361,  863],
         [ 739,  588,  948,  452,  297, 1009,   49,  725,  853,  843,  249,
           957, 1008,  730,  860],
         [ 174,  125,  519,  975,  686,  404,  321,  668,   38,  138,  424,
           457,   98,  736, 1004],
         [  68,  262,  289,  299, 1022,  170,  865,  869,  951,  839,  100,
           301,  828,   62,  511],
         [ 509,  693,  235,  208,  668,  777,  284,  832,  376,  203,  784,
           101,  344,  587,  736],
         [ 121,   83,  484,  951,  839,  180,  801,  363,  890,  373,  206,
           467,  524,  572,  614],
         [ 146,  297,  674,  895,  740,  179,  782,  521,  721,  815,   85,
            74,  179,  650,  554],
         [ 708,  166,  203, 1021,   89,  991,  410,  117, 1019,  742,  235,
           810,  782,  623,  176],
         [ 358,  999,  360,  260,  278,  582,  921,  314,  242,  667,   21,
           463,  335,  566,  897]],
        [[ 851,  360,  851,  877,  665,  322,  581,  936,  826,  910,  110,
           110,  160,  103,  204],
         [ 325,  260,  722,  260,  549,   20,  508,  455,  221,  631,  846,
           658,  457,  124,  496],
         [ 529,  367,  767,  408,  628,  190,   80,  460,  351,  209,  768,
           255,  655,  759,  605],
         [ 344,  192,  255,  271,  402,  930,  805,  939,  497,   94,  843,
            38,   96,  140,  760],
         [ 415,   65,  953,  337,  599,  358,  520,  477,  602,  539,  443,
           703,  124,  110,   92],
         [ 514,  847,  606, 1014,  678,  806,  563,  408,  520,    4,  208,
            83,  630,  176,  423],
         [ 768,  741,  546,  353,  968,  371,  527,  447,   21,  746,  343,
           100,  286,  708,  781],
         [ 461,  499,  836,  411,  271,  279,  530,  882,  345, 1001,  828,
           270,  733,   74,  709],
         [ 539,  706,  278,  343,  235,  754,  346,  272,   52,  987,  151,
            74,  757,  408,  623],
         [ 668,  754,  817,  872,  526,  479,  889,   24,  297,  482,  162,
           414,  128,  811,  488],
         [ 973,  938,  874,  855,  767,  419,  378,  832,  745,  820,  957,
           364,  389,  976,  301],
         [ 162,  174,  830,   67,  749,  433,  428,  405,   63,  632,  391,
           750,  518,  452,  743],
         [   5,  694,  393,  322,  563,  425,  306,  211,  870,  302,  491,
           694,  324,  142,  997],
         [ 981,  953,  116,   51,  674,  451,  351,  335,  285,   44,  591,
           147,  124,  212,  957],
         [ 813,   80,  700,  675,  964,  355,  137,  104,  679,  151,   88,
           553,  815,  820,   21],
         [ 398,  102,  563,  720,  304,  299, 1009,  606,  186,   52, 1012,
           807,  999,  642,  901],
         [ 405,  522,  668,  526,  657,  762,  624,  636,  358,  570,  572,
           169,  580,  567,  939],
         [ 153,  712,  786,  553,  210,  472,  327,  759,   51,  153,  833,
            22,  800,  777,  283],
         [ 324,   45,  757,  563,  703,  888,  256,  447,  515,  313,   94,
           345,  295,  596,  132],
         [ 792,  242,  242,  225,  229, 1004,  436,   61,  869,  757,  945,
          1004,  122,  914,  989],
         [ 595,  902,   56,  961,  722,  731,  937,  332,  706,   30,  372,
           479, 1023,  837,  513],
         [ 918,  972,  772,  658,  594,   12,  106,  225,  678,  920,  971,
           724,  181,  864,  837],
         [ 672,  237,   87,   36,  344,  866,  260,  473,  915,  203,  385,
            23,  561,  754,   71],
         [ 327,   65,  330,  525,  115,  837,  384,  734,  113,  178,  982,
           285,  678,  392,   50],
         [ 206,  317,  201,  954,  534,  692,  902,  773,  399,  215,  766,
           143,   35,  135,  672],
         [ 483,  984,  864,  843,  478,  811,  931,  656,  561,  636,  638,
           326,  141,  140,  632],
         [ 508,  315,  204,  862,  265,  444,  277,  658,  281, 1009,  453,
           283,  387,   85,  677],
         [ 586,  992,  528,  525,   90,  288,   15,  370,  939,  894,  791,
           819,  879,  279,  222],
         [ 639,  896,  792,  487,  853,  852,  690,  886,  141,  988,  889,
            29,  899,  745,  864],
         [ 551,  167,  982,  422,  768,  495,  244,  956,  991,  242,  353,
           622,  168, 1019,  735],
         [ 207,  155,  674,  423,  792,  755,  582,  541,  612,  429,  460,
           947,  173,  471,   79],
         [ 776,  304,  401,  113,  927,  439,  362,  612,  527,  343,  845,
           326,  708,   83,  473]]]).to(torch_device),
    "dac_44khz": torch.tensor([[[ 330,  315,  315,  619,  481,  315,  197,  315,  315,  105,  481,
           315,  481,  481,  481],
         [ 718, 1007,  929,    6,  906,  944,  402,  750,  396,  854,  336,
           426,  609,  356,  329],
         [ 417,  266,  697,  456,  300,  941,  325,  923, 1022,  605,  991,
             7,  939,  217,  456],
         [ 813,  811,  271,  148,  184,  838,  723,  497,  678,  922,   12,
           333,  918,  842,  285],
         [ 832,  307,  635,  794,  334,  828,   32,  505,  610,  170,  161,
           907,  193,  372,  585],
         [  91,  941,  912, 1001,  507,  486,  362, 1006,  157,  640,  760,
           215,  577,  256,  371],
         [ 676,   27,  903,  472,  473,  881,  860,  477,  514,  385,  533,
           911,  701,  102,  825],
         [ 326,  399,  116,  443,  605,  807,  534,  199,  559,  538,  516,
           983,  372,  861,  167],
         [ 776,  843,  185,  326,  723,  390,  318,   34,  191,  674,  728,
           554,  721,  354,  267]],
        [[ 578,  698,  330,  330,  330,  578,  330,  801,  330,  330,  330,
           330,  330,  330,  330],
         [ 171,  503,  725,  215,  814,  861,  139,  684,  880,  905,  937,
           418,  359,  190,  823],
         [ 141,  482,  780,  489,  845,  499,   59,  480,  296,   30,  631,
           540,  399,   23,  385],
         [ 402,  837,  216,  116,  535,  456, 1006,  969,  994,  125, 1011,
           285,  851,  832,  197],
         [  46,  950,  728,  645,  850,  839,  527,  850,   81,  449,  590,
           166,   22,  148,  402],
         [  98,  758,  474,  941,  217,  667,  681,  109,  719,  233,  162,
           160,  329,  627,  716],
         [ 999,  228,  752,  639,  404,  333,  993,  177,  888,  158,  644,
           221, 1011,  302,   79],
         [ 669,  535,  164,  665,  809,  798,  448,  800,  123,  936,  639,
           361,  353,  402,  160],
         [ 345,  355,  940,  261,   71,  946,  750,  120,  565,  692,  813,
           976,  946,   50,  516]]]).to(torch_device),
}
EXPECTED_DEC_OUTPUTS_BATCH = {
    "dac_16khz": torch.tensor([[-1.9467e-04,  1.8777e-04,  3.2151e-04,  2.1486e-04, -2.9929e-05,
         -3.3721e-04, -4.6482e-04, -4.3892e-04, -2.8615e-04,  2.7902e-04,
          8.9113e-04,  1.1197e-03,  1.6152e-03,  1.9360e-03,  1.7877e-03,
          5.9848e-04, -4.4067e-04, -1.3753e-03, -2.0031e-03, -2.0505e-03,
         -1.5628e-03, -4.1970e-04,  6.3706e-04,  1.2562e-03,  1.3377e-03,
          1.2818e-03,  5.9528e-04,  9.5545e-05, -6.1904e-04, -1.3879e-03,
         -1.4964e-03, -9.8743e-04, -3.9004e-04,  5.2811e-04,  1.8659e-03,
          2.3861e-03,  2.1142e-03,  1.4150e-03,  7.4383e-04, -2.4484e-04,
         -9.8938e-04, -1.3888e-03, -1.6679e-03, -1.0585e-03, -6.2462e-04,
         -5.2981e-04, -2.1624e-04,  4.1613e-04,  7.7746e-04,  7.9018e-04],
        [ 5.6791e-06,  4.7036e-04, -1.4851e-03, -4.5138e-05, -3.5936e-04,
         -1.1597e-03,  8.7787e-04, -1.3094e-03,  4.6284e-04, -8.3225e-04,
         -8.9304e-04,  3.2897e-04,  1.1206e-03, -1.7839e-03,  1.9769e-04,
          2.5722e-04, -3.0796e-03,  1.0220e-03,  1.1142e-03,  2.1819e-03,
         -7.3865e-04,  1.6388e-03, -2.9485e-04, -2.4126e-03, -5.0188e-04,
          2.9123e-04,  1.2642e-03, -1.8195e-03,  3.3691e-03,  1.3383e-03,
         -1.5909e-03, -1.7411e-04, -5.3695e-04, -6.9192e-04, -2.3446e-03,
         -5.2493e-04,  1.5174e-03, -1.0849e-03,  9.9574e-04,  8.2603e-04,
         -1.6866e-03,  2.7005e-05, -2.7066e-03,  4.2188e-04, -1.0763e-03,
          2.5058e-04,  3.3699e-03, -1.8019e-03,  5.5643e-04, -1.2191e-03]]).to(torch_device),
    "dac_24khz": torch.tensor([[ 3.2435e-04,  1.7916e-04, -3.3709e-04, -5.6194e-04, -5.2839e-04,
         -4.7647e-04, -3.6741e-04, -2.0843e-04,  9.9571e-05,  1.6874e-04,
          8.8567e-05,  1.9617e-04,  1.8428e-04,  1.1912e-04,  3.4829e-04,
          5.9329e-04,  2.6147e-04, -6.2153e-05,  1.6157e-04,  5.2102e-04,
          5.5313e-04,  5.4360e-04,  5.7747e-04,  3.5971e-04,  3.0730e-04,
          4.5132e-04,  3.8683e-04,  1.2163e-04, -3.3234e-06, -1.6039e-04,
         -7.1165e-04, -1.2626e-03, -1.4309e-03, -1.3981e-03, -1.4285e-03,
         -1.3813e-03, -1.0371e-03, -5.6944e-04, -3.7733e-05,  5.5460e-04,
          9.8616e-04,  1.1260e-03,  1.1337e-03,  1.0434e-03,  7.9634e-04,
          3.9289e-04,  1.3979e-05, -2.1139e-04, -3.5446e-04, -5.8127e-04],
        [-5.4007e-04,  3.4078e-04,  5.9058e-04,  5.5237e-04,  1.5243e-03,
          1.3597e-03, -3.3368e-05, -5.5780e-05,  6.9357e-04, -8.9678e-04,
         -2.7849e-03, -2.0006e-03, -8.1877e-04, -1.1219e-03,  9.4584e-05,
          2.0848e-03, -1.7615e-04, -2.5922e-03,  6.0989e-04,  2.8621e-03,
         -4.9042e-04, -2.5334e-03, -4.9112e-04, -2.1532e-04, -1.5235e-03,
         -2.4817e-04,  1.7170e-05, -3.4833e-03, -3.3743e-03,  1.0406e-03,
          9.7970e-04, -1.3182e-03,  1.1609e-03,  3.0964e-03, -7.4387e-05,
         -1.4777e-03,  7.0038e-04, -4.5195e-04, -3.2168e-03, -1.0643e-03,
          8.4545e-04, -1.7978e-03, -1.5020e-03,  2.0434e-03,  5.6250e-04,
         -2.4668e-03,  1.7021e-04,  2.5531e-03, -8.0111e-05, -1.2472e-03]]).to(torch_device),
    "dac_44khz": torch.tensor([[-4.7262e-04, -2.2317e-04,  7.4245e-06,  1.6209e-04,  2.6290e-04,
          4.0085e-04,  5.3670e-04,  7.0190e-04,  8.0829e-04,  9.1402e-04,
          1.0151e-03,  1.0510e-03,  1.0495e-03,  1.0252e-03,  9.6937e-04,
          8.3693e-04,  6.4934e-04,  4.3139e-04,  2.0274e-04, -5.0105e-07,
         -1.5536e-04, -2.4031e-04, -2.7451e-04, -2.6790e-04, -2.8830e-04,
         -3.5545e-04, -4.6174e-04, -5.6426e-04, -6.5037e-04, -7.0368e-04,
         -7.1532e-04, -6.8602e-04, -6.1577e-04, -4.9298e-04, -3.1922e-04,
         -1.3299e-04,  2.6576e-05,  1.4860e-04,  2.6291e-04,  4.0380e-04,
          5.9641e-04,  8.1077e-04,  9.7452e-04,  1.0567e-03,  1.0606e-03,
          1.0059e-03,  9.1518e-04,  8.0621e-04,  6.8698e-04,  5.7436e-04],
        [ 7.1654e-04,  8.2036e-04,  3.7017e-04, -3.8982e-04, -8.6940e-04,
         -6.0764e-04,  2.0173e-04,  1.0585e-03,  1.3268e-03,  7.7415e-04,
         -3.3407e-04, -1.3455e-03, -1.6446e-03, -1.0762e-03,  2.2359e-05,
          9.7529e-04,  1.2350e-03,  7.5504e-04, -2.9908e-05, -4.7744e-04,
         -1.4328e-04,  8.8642e-04,  2.0004e-03,  2.4950e-03,  2.0541e-03,
          1.0173e-03,  1.4635e-04,  9.4588e-05,  7.8387e-04,  1.3183e-03,
          7.2871e-04, -1.1185e-03, -3.1932e-03, -3.9675e-03, -2.6470e-03,
         -1.7819e-05,  1.9833e-03,  1.7660e-03, -5.8018e-04, -3.2851e-03,
         -4.2912e-03, -2.8509e-03, -7.3051e-05,  2.0344e-03,  2.2056e-03,
          8.9169e-04, -3.5831e-04, -3.9292e-04,  6.0499e-04,  1.4272e-03]]).to(torch_device),
}
EXPECTED_QUANT_CODEBOOK_LOSS_BATCH = {
    "dac_16khz": 20.647245407104492,
    "dac_24khz": 23.597030639648438,
    "dac_44khz": 16.138029098510742,
}
EXPECTED_CODEC_ERROR_BATCH = {
    "dac_16khz": 0.0019725316669791937,
    "dac_24khz": 0.0013007941888645291,
    "dac_44khz": 0.0003823760780505836,
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
