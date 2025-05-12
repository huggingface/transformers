# coding=utf-8
# Copyright 2024, The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch ConversationalSpeechModel model."""

import collections
import copy
import re
import unittest

import pytest
from parameterized import parameterized

from transformers import (
    AutoProcessor,
    CsmConfig,
    CsmForConditionalGeneration,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch_gpu,
    slow,
    torch_device,
)
from transformers.utils.import_utils import is_datasets_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    ids_tensor,
)


if is_datasets_available():
    from datasets import load_dataset

if is_torch_available():
    import torch

    from transformers.pytorch_utils import id_tensor_storage


class CsmModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        batch_size=3,
        seq_length=7,
        is_training=True,
        depth_decoder_config={
            "num_codebooks": 10,
            "backbone_hidden_size": 64,
            "vocab_size": 6,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_act": "silu",
            "max_position_embeddings": 10,
        },
        codec_config={
            "model_type": "mimi",
            "audio_channels": 1,
            "chunk_in_sec": None,
            "hidden_size": 32,
            "num_filters": 8,
            "num_residual_layers": 1,
            "upsampling_ratios": [8, 4],
            "codebook_size": 64,
            "vector_quantization_hidden_dimension": 64,
            "upsample_groups": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "sliding_window": 4,
            "codebook_dim": 64,
            "use_cache": False,
        },
        config={
            "num_codebooks": 10,
            "vocab_size": 6,
            "text_vocab_size": 99,
            "hidden_size": 64,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_act": "silu",
            "max_position_embeddings": 10,
            "bos_token_id": 1,
            "pad_token_id": 2,
            "eos_token_id": 3,
            "codebook_pad_token_id": 2,
            "codebook_eos_token_id": 3,
        },
    ):
        self.parent = parent
        self.is_training = is_training
        self.ignore_index = ignore_index
        self.depth_decoder_config = depth_decoder_config
        self.codec_config = codec_config
        self.config = config
        self.seq_length = seq_length
        self.batch_size = batch_size

        self.num_hidden_layers = config["num_hidden_layers"]
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.pad_token_id = config["pad_token_id"]

    def get_config(self):
        return CsmConfig(
            depth_decoder_config=self.depth_decoder_config,
            codec_config=self.codec_config,
            **self.config,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length, config.num_codebooks], config.vocab_size - 1) + 1
        attention_mask = input_ids[..., -1].ne(1).to(torch_device)
        return config, input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict


class CsmForConditionalGenerationTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (CsmForConditionalGeneration,) if is_torch_available() else ()
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False
    test_resize_embeddings_untied = False
    test_torch_exportable = True

    def setUp(self):
        self.model_tester = CsmModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CsmConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        """
        Overrides [ModelTesterMixin._prepare_for_class] to handle third input_ids dimension.
        """
        inputs_dict = copy.deepcopy(inputs_dict)

        if return_labels:
            inputs_dict["labels"] = torch.zeros(
                (
                    self.model_tester.batch_size,
                    self.model_tester.seq_length,
                    self.model_tester.config["num_codebooks"],
                ),
                dtype=torch.long,
                device=torch_device,
            )

        return inputs_dict

    def _get_logits_processor_kwargs(self, do_sample=False, config=None):
        """
        Overrides [GenerationTesterMixin._get_logits_processor_kwargs] to restrict to top_k, top_p, and temperature sampling.
        """
        logits_processor_kwargs = {}
        if do_sample:
            logits_processor_kwargs.update(
                {
                    "top_k": 10,
                    "top_p": 0.7,
                    "temperature": 0.7,
                }
            )

        return logits_processor_kwargs

    def test_initialization(self):
        """
        Overrides [ModelTesterMixin.test_initialization] because of specificities of Mimi codec model.
        See https://github.com/huggingface/transformers/blob/1077603410cd73ba71d64a522033574d66d64b55/tests/models/mimi/test_modeling_mimi.py#L384-L397
        """
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

    def _check_similar_generate_outputs(self, output_1, output_2, atol=1e-5, rtol=1e-5):
        """
        Overrides [GenerationTesterMixin._check_similar_generate_outputs] to handle third input_ids dimension.
        Here we only look a the first codebook (index 0 on last dimension of the generated sequences) since returned scores
        are for this token.
        """
        # scores doesn't include data regarding decoder input tokens
        decoder_input_length = output_1.sequences.shape[1] - len(output_1.scores)
        output_matches = output_1.sequences[..., 0] == output_2.sequences[..., 0]
        has_matching_outputs = output_matches.all()
        has_matching_scores = None
        if not has_matching_outputs:
            for batch_idx in range(output_1.sequences.shape[0]):
                batch_matches = output_matches[batch_idx]
                if batch_matches.all():
                    continue
                first_mismatch_idx = batch_matches.int().argmin()  # gets the index of the first False
                first_mismatch_idx -= decoder_input_length
                output_1_first_mismatch_scores = output_1.scores[first_mismatch_idx][batch_idx]
                output_2_first_mismatch_scores = output_2.scores[first_mismatch_idx][batch_idx]
                has_matching_scores = torch.allclose(
                    output_1_first_mismatch_scores, output_2_first_mismatch_scores, rtol=atol, atol=rtol
                )
                if not has_matching_scores:
                    break
        self.assertTrue(has_matching_outputs or has_matching_scores)

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support assisted decoding.")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support assisted decoding.")
    def test_assisted_decoding_sample(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support Dola decoding.")
    def test_dola_decoding_sample(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support beam search.")
    def test_beam_sample_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support beam search.")
    def test_beam_search_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support beam search.")
    def test_beam_search_generate_dict_output(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support beam search.")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support beam search.")
    def test_beam_sample_generate_dict_output(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support group beam search.")
    def test_group_beam_search_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support group beam search.")
    def test_group_beam_search_generate_dict_output(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support constrained beam search.")
    def test_constrained_beam_search_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support constrained beam search.")
    def test_constrained_beam_search_generate_dict_output(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support contrastive search.")
    def test_contrastive_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support contrastive search.")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support contrastive search.")
    def test_contrastive_generate_low_memory(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support prompt lookup decoding.")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support prompt lookup decoding.")
    def test_prompt_lookup_decoding_stops_at_eos(self):
        pass

    @pytest.mark.skip(reason="CSM has custom embedding approach (text and audio embeddings).")
    def test_model_get_set_embeddings(self):
        pass

    @pytest.mark.skip(reason="CSM has custom embedding approach (text and audio embeddings).")
    def test_tie_model_weights(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support beam search.")
    def test_generate_from_inputs_embeds_1_beam_search(self, _, num_beams):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support beam search.")
    def test_model_parallel_beam_search(self):
        pass

    def test_tied_weights_keys(self):
        """
        Overrides [ModelTesterMixin.test_tied_weights_keys] to not test for text config (not applicable to CSM).
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model_tied = model_class(config)

            ptrs = collections.defaultdict(list)
            for name, tensor in model_tied.state_dict().items():
                ptrs[id_tensor_storage(tensor)].append(name)

            # These are all the pointers of shared tensors.
            tied_params = [names for _, names in ptrs.items() if len(names) > 1]

            tied_weight_keys = model_tied._tied_weights_keys if model_tied._tied_weights_keys is not None else []
            # Detect we get a hit for each key
            for key in tied_weight_keys:
                is_tied_key = any(re.search(key, p) for group in tied_params for p in group)
                self.assertTrue(is_tied_key, f"{key} is not a tied weight key for {model_class}.")

            # Removed tied weights found from tied params -> there should only be one left after
            for key in tied_weight_keys:
                for i in range(len(tied_params)):
                    tied_params[i] = [p for p in tied_params[i] if re.search(key, p) is None]

            tied_params = [group for group in tied_params if len(group) > 1]
            self.assertListEqual(
                tied_params,
                [],
                f"Missing `_tied_weights_keys` for {model_class}: add all of {tied_params} except one.",
            )

    def _get_custom_4d_mask_test_data(self):
        """
        Overrides [ModelTesterMixin._get_custom_4d_mask_test_data] to handle third input_ids dimension.
        """
        # Sequence in which all but the last token is the same
        input_ids = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]], device=torch_device, dtype=torch.int64)
        input_ids = input_ids.unsqueeze(-1).expand(-1, -1, self.model_tester.config["num_codebooks"])
        position_ids = torch.tensor([[0, 1, 2, 3]] * 3, device=torch_device, dtype=torch.int64)

        # Combining common prefix with the unique ending tokens:
        input_ids_shared_prefix = torch.cat([input_ids[0][:-1], input_ids[:, -1]]).unsqueeze(0)

        # Creating a 4D mask where each of the last 3 tokens do not attend to each other.
        mask_shared_prefix = torch.tensor(
            [
                [
                    [
                        [1, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 0, 1, 0],
                        [1, 1, 1, 0, 0, 1],
                    ]
                ]
            ],
        )
        # inverting the attention mask
        mask_dtype = torch.float32
        min_dtype = torch.finfo(mask_dtype).min
        mask_shared_prefix = (mask_shared_prefix.eq(0.0)).to(dtype=mask_dtype, device=torch_device) * min_dtype

        # Creating a position_ids tensor. note the repeating figures in the end.
        position_ids_shared_prefix = torch.tensor([[0, 1, 2, 3, 3, 3]], device=torch_device, dtype=torch.int64)

        return input_ids, position_ids, input_ids_shared_prefix, mask_shared_prefix, position_ids_shared_prefix


class CsmForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        # TODO: @eustlb, update with correct sesame's repo
        self.model_checkpoint = "eustlb/csm-1b"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def _load_conversation(self):
        ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
        ds = ds.filter(lambda x: x["conversation_id"] == 0)
        ds = ds.sort("turn_id")
        return ds[0]

    @slow
    @require_torch_gpu
    def test_1b_model_integration_generate(self):
        """
        Tests the generated tokens match the ones from the original model implementation.
        Such tokens are to be retreived using https://gist.github.com/eustlb/d25577a357ddcf8f4a8cd0d00baca551, which is a script that infers the original model.
        """
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)
        prompt = "<|begin_of_text|>[0]What are you working on?<|end_of_text|><|AUDIO|><|audio_eos|><|begin_of_text|>[1]I'm figuring out my budget.<|end_of_text|>"

        ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
        audio = ds[0]["audio"]["array"]
        inputs = processor(text=prompt, audio=audio, return_tensors="pt").to(torch_device)

        model = CsmForConditionalGeneration.from_pretrained(self.model_checkpoint, device_map=torch_device)
        output_tokens = model.generate(**inputs, do_sample=False, depth_decoder_do_sample=False)

        # fmt: off
        EXPECTED_OUTPUT_TOKENS = torch.tensor([[
            [1140, 10, 37, 1180, 1100, 1319, 601, 1482, 1918, 1739, 372, 856, 674, 1, 854, 459, 1843, 1191, 347, 349, 1087, 846, 759, 1690, 947, 1280, 580, 1909, 1192, 487, 1302, 1601],
            [1494, 1412, 1824, 1852, 150, 928, 91, 326, 623, 1632, 1163, 1221, 1949, 999, 1779, 248, 693, 1149, 1423, 1503, 598, 80, 223, 1798, 251, 385, 1391, 1692, 1228, 1631, 1101, 866],
            [778, 645, 830, 1812, 524, 1704, 1805, 1289, 74, 1069, 243, 1622, 1755, 1281, 1397, 620, 1962, 1995, 253, 1124, 1007, 518, 89, 559, 1304, 1482, 523, 1747, 1979, 1003, 1707, 1578],
            [1356, 481, 642, 989, 287, 1819, 171, 1115, 824, 1253, 1488, 1074, 1019, 342, 279, 513, 1275, 1364, 893, 2007, 553, 407, 882, 1170, 1586, 485, 762, 559, 100, 542, 911, 1460],
            [1860, 593, 1944, 404, 575, 545, 862, 830, 1002, 125, 2010, 268, 1779, 804, 811, 809, 255, 373, 387, 1756, 259, 822, 1191, 700, 1686, 390, 1676, 844, 2006, 286, 1376, 719],
            [1165, 1047, 848, 212, 1018, 1470, 93, 1709, 1487, 1691, 1190, 275, 1278, 2018, 121, 1023, 485, 463, 39, 1825, 1936, 1817, 569, 209, 1553, 1599, 1137, 769, 968, 558, 1957, 265],
            [902, 1608, 719, 850, 371, 1920, 75, 1917, 2005, 1238, 562, 1743, 713, 95, 1107, 1463, 696, 840, 8, 487, 1950, 1171, 1004, 1516, 1130, 303, 1866, 1728, 2046, 238, 265, 153],
            [1932, 839, 334, 1167, 134, 2025, 40, 505, 1244, 1238, 1840, 800, 697, 72, 216, 486, 940, 1312, 510, 361, 549, 583, 1364, 844, 397, 1181, 1779, 962, 457, 1782, 1316, 465],
            [31, 1558, 1048, 404, 354, 7, 827, 414, 1082, 807, 243, 1517, 801, 1364, 99, 1276, 1655, 1488, 1313, 464, 828, 1612, 774, 1558, 745, 1496, 960, 1874, 995, 1943, 255, 213],
            [355, 1270, 413, 1519, 1659, 1904, 690, 552, 1279, 1821, 2022, 458, 1779, 2003, 604, 832, 661, 1295, 305, 1701, 173, 869, 230, 539, 1188, 669, 117, 692, 250, 388, 1995, 294],
            [629, 199, 1899, 1123, 1070, 344, 578, 1795, 1451, 1257, 168, 1410, 1120, 1270, 316, 983, 1245, 1870, 165, 471, 966, 1337, 308, 1118, 746, 67, 1767, 1480, 1517, 1585, 871, 1110],
            [1281, 1173, 784, 404, 368, 403, 580, 526, 853, 1692, 792, 895, 1286, 573, 1368, 896, 931, 1958, 1912, 644, 583, 1706, 1176, 1262, 1637, 315, 524, 1629, 795, 1211, 915, 533],
            [9, 1783, 621, 1954, 1212, 993, 197, 977, 1662, 1340, 618, 1997, 1689, 1001, 74, 1765, 1865, 797, 1219, 1609, 671, 1491, 950, 1849, 1301, 2031, 875, 323, 203, 1063, 1490, 1538],
            [1944, 1578, 1256, 1169, 790, 1444, 1382, 1616, 1100, 1264, 214, 1646, 488, 573, 1333, 285, 1954, 74, 1333, 674, 1303, 266, 622, 1290, 402, 109, 1331, 1666, 1347, 780, 106, 605],
            [221, 161, 1322, 1, 565, 1507, 1403, 1091, 1557, 932, 1664, 1165, 1828, 1647, 2008, 1616, 648, 1113, 1870, 22, 734, 1458, 1940, 1756, 1689, 925, 1318, 1095, 985, 473, 604, 1974],
            [1178, 597, 1804, 747, 1383, 360, 1497, 406, 1053, 1023, 1901, 56, 1221, 628, 75, 1729, 575, 1681, 840, 410, 650, 794, 1171, 1889, 187, 54, 1364, 1390, 505, 1285, 1814, 90],
            [1432, 1221, 1800, 1873, 1255, 627, 41, 9, 630, 896, 1469, 1195, 1098, 145, 442, 1460, 13, 57, 2039, 1015, 149, 461, 1084, 1288, 1099, 910, 63, 157, 906, 111, 1394, 460],
            [1352, 593, 307, 780, 1614, 1675, 1491, 1253, 723, 1793, 1032, 1486, 1805, 1904, 777, 398, 1791, 951, 770, 499, 1858, 244, 1372, 1514, 1858, 1200, 69, 181, 673, 1144, 1938, 1191],
            [905, 403, 1626, 1529, 581, 1443, 976, 754, 1561, 1370, 1048, 253, 194, 1271, 853, 959, 1532, 30, 286, 1594, 1255, 1135, 1410, 1699, 1423, 2002, 260, 69, 941, 1640, 895, 722],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]])
        # fmt: on

        torch.testing.assert_close(output_tokens.cpu(), EXPECTED_OUTPUT_TOKENS)

    @slow
    @require_torch_gpu
    def test_1b_model_integration_generate_no_audio(self):
        """
        Tests the generated tokens match the ones from the original model implementation.
        Such tokens are to be retreived using https://gist.github.com/eustlb/aed822f765e928b9612e01b0d8836d69, which is a script that infers the original model.
        """

        processor = AutoProcessor.from_pretrained(self.model_checkpoint)

        conversation = [
            {"role": "0", "content": [{"type": "text", "text": "The past is just a story we tell ourselves."}]},
        ]

        inputs = processor.apply_chat_template(conversation, tokenize=True, return_dict=True).to(torch_device)

        model = CsmForConditionalGeneration.from_pretrained(self.model_checkpoint, device_map=torch_device)
        output_tokens = model.generate(**inputs, do_sample=False, depth_decoder_do_sample=False)

        print(output_tokens)
        # fmt: off
        EXPECTED_OUTPUT_TOKENS = torch.tensor([[
            [1656, 629, 723, 1785, 206, 1873, 1059, 1190, 1833, 240, 618, 350, 156, 109, 2010, 452, 435, 1764, 77, 654, 1133, 908, 1095, 74, 804, 494, 1760, 1343, 1312, 1464, 1657, 324],
            [366, 1532, 1945, 21, 145, 1428, 1417, 1987, 1793, 1444, 356, 1491, 849, 333, 788, 426, 1423, 1004, 414, 1823, 1169, 257, 1892, 696, 1572, 998, 1098, 523, 390, 1977, 546, 1692],
            [1343, 1382, 1288, 1744, 1685, 1154, 1837, 1156, 1680, 1641, 1479, 1548, 632, 824, 694, 2010, 671, 1251, 1822, 343, 638, 1372, 696, 1272, 144, 125, 1332, 579, 936, 77, 159, 357],
            [456, 1534, 349, 274, 1956, 1502, 1268, 1038, 1911, 523, 1360, 1159, 761, 293, 718, 1143, 63, 705, 168, 550, 413, 1372, 1771, 787, 631, 693, 784, 1789, 2039, 1131, 1601, 918],
            [456, 829, 2026, 1108, 1649, 207, 1308, 1440, 1192, 1394, 426, 546, 590, 36, 1682, 1827, 1387, 1425, 1909, 1500, 1438, 1297, 5, 888, 948, 1745, 1304, 1364, 1692, 131, 300, 1908],
            [2027, 1431, 1037, 1789, 1296, 1264, 1331, 1787, 1235, 1902, 1161, 1591, 590, 561, 1633, 1218, 510, 148, 1962, 118, 212, 608, 565, 1869, 583, 598, 532, 658, 1416, 9, 1172, 493],
            [1215, 460, 1722, 317, 1423, 716, 1589, 1177, 1927, 1860, 1756, 1552, 1674, 643, 74, 1256, 587, 1742, 771, 2028, 469, 1070, 1683, 1614, 699, 494, 2020, 139, 1365, 1171, 171, 904],
            [1615, 339, 323, 317, 469, 714, 104, 2015, 1407, 278, 468, 77, 2007, 650, 1630, 269, 168, 934, 1544, 58, 1487, 1373, 705, 874, 1252, 2031, 1995, 254, 1334, 1171, 1911, 1607],
            [1259, 693, 666, 1700, 1115, 607, 982, 769, 1106, 1500, 101, 88, 1698, 1864, 1358, 1594, 192, 153, 1868, 1654, 604, 1948, 526, 778, 172, 1664, 1966, 99, 1334, 1030, 1349, 1209],
            [1211, 579, 1369, 492, 1725, 203, 1125, 778, 701, 1982, 1420, 155, 736, 1145, 2018, 609, 658, 561, 1147, 923, 1794, 1753, 116, 1374, 612, 956, 1587, 392, 1062, 2047, 901, 1931],
            [460, 1093, 1346, 1917, 1223, 470, 271, 390, 547, 112, 143, 1633, 1030, 643, 96, 1759, 920, 1959, 75, 1280, 1630, 999, 333, 853, 1110, 1291, 1911, 57, 171, 1658, 1704, 1508],
            [908, 500, 393, 184, 1437, 482, 2008, 1834, 356, 1435, 1550, 1407, 1236, 109, 1167, 452, 1141, 934, 207, 957, 660, 670, 28, 1066, 1252, 1932, 669, 906, 1904, 1820, 2043, 881],
            [1599, 1031, 1474, 336, 1540, 571, 437, 1440, 1616, 1365, 1412, 1246, 400, 405, 1776, 96, 296, 38, 1597, 466, 1630, 1256, 1940, 887, 1769, 294, 285, 842, 1756, 1619, 451, 1529],
            [1615, 339, 1722, 525, 942, 105, 1365, 670, 785, 1316, 465, 1860, 438, 968, 547, 1938, 1816, 1429, 1065, 1942, 660, 1446, 1093, 1066, 931, 121, 688, 1033, 1178, 754, 1783, 94],
            [912, 1354, 598, 254, 341, 1980, 1166, 585, 1302, 473, 554, 242, 174, 2030, 2011, 325, 978, 1690, 258, 396, 1831, 1768, 1291, 1699, 2001, 433, 1414, 2012, 1045, 511, 533, 1104],
            [80, 1791, 1062, 1136, 391, 568, 1651, 101, 959, 2043, 1683, 760, 794, 181, 570, 540, 1599, 20, 1017, 973, 1654, 396, 586, 778, 2044, 1664, 1911, 929, 66, 897, 510, 643],
            [1161, 1093, 161, 1296, 589, 54, 906, 981, 1927, 605, 516, 1731, 1461, 1204, 1902, 920, 1488, 177, 805, 1402, 610, 1446, 1154, 1067, 2025, 645, 762, 1715, 415, 1658, 1713, 1607],
            [374, 1444, 1577, 792, 1450, 628, 604, 1729, 322, 514, 1725, 540, 1070, 575, 653, 800, 250, 187, 569, 349, 354, 1573, 176, 793, 897, 359, 536, 276, 1224, 23, 145, 1287],
            [1184, 415, 1644, 1737, 1788, 385, 784, 1861, 1172, 1118, 367, 1156, 234, 1946, 1742, 981, 828, 1798, 1821, 361, 1148, 670, 518, 1288, 761, 1050, 1642, 1006, 1747, 840, 1599, 720],
            [1141, 1731, 1670, 1542, 1347, 1907, 683, 753, 1347, 68, 2031, 153, 556, 719, 736, 1759, 1131, 1073, 1747, 1730, 1487, 1137, 1869, 1624, 699, 1900, 748, 49, 1312, 735, 726, 1268],
            [1141, 1383, 405, 1033, 490, 488, 1102, 471, 713, 1630, 447, 703, 1495, 1001, 1855, 354, 456, 411, 786, 853, 168, 407, 116, 699, 605, 128, 532, 1076, 208, 447, 1448, 1071],
            [345, 1013, 948, 1728, 1837, 337, 930, 1226, 1643, 1729, 983, 1688, 2009, 435, 1358, 721, 42, 1779, 1332, 1077, 1873, 128, 1327, 125, 1226, 1704, 705, 1459, 1449, 862, 155, 1870],
            [336, 904, 684, 184, 1542, 714, 1752, 1180, 1373, 1816, 504, 1716, 1066, 1086, 1212, 530, 1413, 1278, 75, 1347, 82, 1623, 1307, 1717, 1861, 494, 888, 1589, 670, 1999, 905, 1430],
            [578, 554, 14, 523, 1016, 300, 1589, 1017, 356, 1583, 1654, 414, 449, 376, 1413, 58, 706, 963, 388, 1626, 131, 352, 1024, 1054, 2025, 1561, 77, 1589, 1486, 431, 1249, 1508],
            [184, 2043, 169, 1673, 580, 162, 1752, 397, 1119, 2009, 697, 150, 1475, 157, 1523, 1402, 575, 86, 1373, 1230, 1564, 1308, 626, 1093, 1603, 1446, 1390, 1543, 1778, 1142, 1357, 1831],
            [1484, 1987, 932, 1728, 1504, 1618, 291, 1865, 1151, 460, 1792, 141, 234, 2043, 829, 513, 435, 791, 1037, 1541, 65, 424, 1589, 1711, 312, 1306, 212, 686, 673, 984, 1914, 1549],
            [513, 1536, 1844, 1319, 572, 1069, 121, 735, 1949, 1211, 1362, 1027, 105, 1379, 315, 1782, 706, 1658, 1510, 1989, 1443, 1690, 822, 1614, 1194, 1460, 992, 2040, 1178, 1474, 1110, 1326],
            [1858, 194, 1594, 1935, 1622, 1892, 1577, 137, 1907, 2015, 757, 414, 1823, 836, 496, 530, 1385, 1503, 1065, 1554, 664, 525, 1031, 433, 69, 466, 1016, 1846, 1609, 1658, 911, 94],
            [1134, 1744, 323, 691, 1837, 347, 1871, 172, 811, 91, 1883, 436, 1912, 23, 1336, 1684, 519, 1612, 1219, 1402, 728, 1953, 1658, 641, 27, 1340, 436, 139, 2008, 1030, 159, 324],
            [1270, 1536, 1639, 414, 1387, 1170, 1067, 1701, 1414, 505, 1122, 36, 1731, 350, 1552, 1214, 1444, 30, 107, 172, 480, 1858, 655, 168, 1107, 691, 1272, 797, 1656, 548, 1407, 1375],
            [1270, 286, 1371, 1552, 1622, 1739, 1348, 2018, 345, 1537, 1941, 2024, 1423, 740, 284, 513, 91, 1228, 2015, 385, 992, 39, 813, 803, 2025, 497, 663, 462, 1609, 334, 927, 1470],
            [1718, 994, 265, 1421, 1622, 1098, 845, 1868, 832, 459, 447, 619, 1970, 929, 513, 63, 1448, 1509, 1219, 1942, 285, 1373, 1259, 1004, 11, 1040, 1984, 57, 188, 1687, 1475, 805],
            [1157, 832, 480, 1225, 1019, 347, 326, 999, 125, 1542, 118, 1383, 1343, 1077, 1821, 1602, 1978, 1642, 618, 808, 692, 1953, 1353, 963, 619, 1291, 1016, 1458, 1995, 1688, 1872, 1718],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]])
        # fmt: on

        torch.testing.assert_close(output_tokens.cpu(), EXPECTED_OUTPUT_TOKENS)

    @slow
    @require_torch_gpu
    def test_1b_model_integration_generate_multiple_audio(self):
        """
        Test the generated tokens match the ones from the original model implementation.
        Such tokens are to be retreived using https://gist.github.com/eustlb/0c94de002e1325abb61d32217f74c0f8, which is a script that infers the original model.
        """
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)

        ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
        conversation = []

        # context
        for text, audio, speaker_id in zip(ds[:4]["text"], ds[:4]["audio"], ds[:4]["speaker_id"]):
            conversation.append(
                {
                    "role": f"{speaker_id}",
                    "content": [{"type": "text", "text": text}, {"type": "audio", "path": audio["array"]}],
                }
            )

        # text prompt
        conversation.append({"role": f"{ds[4]['speaker_id']}", "content": [{"type": "text", "text": ds[4]["text"]}]})

        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        ).to(torch_device)

        model = CsmForConditionalGeneration.from_pretrained(self.model_checkpoint, device_map=torch_device)
        output_tokens = model.generate(**inputs, do_sample=False, depth_decoder_do_sample=False)

        # fmt: off
        EXPECTED_OUTPUT_TOKENS = torch.tensor([[
            [420, 1189, 1311, 318, 359, 694, 1550, 1044, 1614, 1437, 1978, 537, 554, 1681, 147, 1225, 422, 1357, 1681, 1619, 165, 641, 1132, 1975, 1568, 406, 756, 503, 1673, 1428, 762, 781],
            [1848, 1412, 957, 1656, 871, 540, 1999, 175, 711, 1383, 1814, 104, 742, 1285, 733, 1251, 1165, 1915, 1392, 645, 1804, 913, 1772, 632, 376, 1507, 1132, 725, 716, 1121, 1769, 1509],
            [429, 1138, 895, 1018, 1099, 257, 1395, 1015, 576, 1599, 497, 19, 1858, 1437, 282, 357, 1143, 828, 1481, 70, 985, 551, 935, 278, 1102, 1453, 1902, 755, 526, 498, 1441, 1733],
            [546, 343, 1547, 879, 2039, 692, 1999, 1150, 1969, 1866, 1178, 199, 1913, 1738, 1530, 1728, 1193, 74, 695, 612, 1095, 1597, 1381, 683, 1385, 2045, 1069, 865, 438, 70, 1437, 318],
            [1741, 1621, 733, 1580, 1006, 1790, 1031, 1563, 569, 1822, 1229, 854, 142, 1554, 792, 741, 147, 552, 731, 772, 908, 831, 1291, 1819, 296, 290, 1871, 100, 1904, 1420, 1903, 1653],
            [1264, 1576, 963, 12, 1403, 453, 259, 1359, 1270, 466, 1744, 1579, 1081, 1691, 1495, 1293, 110, 1020, 2042, 189, 1358, 955, 784, 1317, 2, 1794, 388, 376, 327, 511, 866, 1308],
            [1407, 1412, 1665, 1683, 284, 874, 1859, 326, 1491, 1343, 777, 695, 1424, 396, 274, 202, 178, 747, 470, 1805, 1414, 2000, 127, 1884, 531, 215, 1322, 1098, 1674, 1227, 1092, 204],
            [584, 637, 1665, 1683, 1136, 1201, 212, 310, 1441, 1619, 190, 1611, 1629, 2011, 1754, 1587, 413, 1287, 1251, 1382, 1904, 444, 1665, 1047, 1982, 1169, 1200, 809, 117, 327, 958, 1877],
            [471, 1469, 1679, 1184, 343, 974, 1442, 897, 1888, 1468, 1092, 1398, 1714, 963, 1577, 1797, 766, 565, 403, 920, 1806, 466, 1193, 446, 825, 775, 1886, 1095, 159, 1085, 858, 504],
            [28, 1511, 1510, 1580, 447, 1934, 1031, 1439, 202, 1435, 474, 1731, 724, 1080, 1121, 421, 625, 1410, 95, 605, 815, 1825, 127, 785, 900, 1673, 178, 1242, 2033, 1230, 350, 139],
            [20, 1215, 253, 955, 871, 1689, 1986, 24, 1648, 423, 562, 1937, 1146, 26, 1266, 346, 188, 318, 179, 1164, 1100, 1978, 478, 1192, 715, 392, 1837, 425, 1492, 766, 1651, 822],
            [1879, 1401, 1444, 723, 1754, 732, 1307, 702, 1768, 2013, 1284, 577, 1287, 1532, 647, 189, 903, 587, 800, 152, 898, 182, 2016, 639, 1074, 1220, 1934, 264, 250, 745, 1652, 536],
            [1874, 1526, 232, 1580, 1980, 988, 1623, 341, 1768, 956, 1430, 1667, 1687, 1289, 826, 1378, 173, 1466, 479, 835, 1786, 1671, 328, 131, 815, 871, 379, 1329, 440, 1117, 392, 272],
            [1762, 426, 1350, 1590, 314, 190, 1514, 344, 1926, 822, 534, 523, 703, 36, 379, 494, 464, 1886, 1555, 1318, 1654, 1469, 1976, 304, 218, 655, 1826, 958, 502, 326, 1898, 861],
            [1577, 386, 503, 1492, 698, 405, 1031, 349, 1804, 2012, 1450, 996, 1140, 26, 449, 33, 1917, 354, 702, 1255, 1942, 1184, 864, 2045, 514, 744, 466, 54, 37, 486, 362, 525],
            [1109, 1920, 445, 1719, 1670, 1220, 745, 40, 171, 1921, 999, 104, 489, 1911, 883, 306, 649, 1751, 762, 1183, 1085, 1112, 1912, 2035, 1940, 1129, 1592, 1276, 1570, 1236, 738, 209],
            [1837, 990, 1063, 318, 1398, 1838, 1678, 906, 754, 802, 562, 353, 1389, 207, 1319, 1188, 2013, 1079, 888, 1706, 1042, 657, 482, 953, 94, 2007, 871, 485, 1596, 275, 410, 1855],
            [872, 974, 1344, 1798, 655, 805, 1604, 1913, 455, 615, 1827, 966, 1330, 1826, 1285, 359, 544, 221, 1538, 1658, 374, 1352, 1714, 1925, 235, 65, 350, 931, 1009, 1164, 218, 736],
            [1547, 617, 1622, 740, 655, 265, 1324, 1265, 1449, 482, 1037, 105, 1128, 701, 1866, 1674, 1999, 1302, 985, 1942, 663, 449, 1881, 698, 805, 1446, 1742, 1192, 1623, 605, 948, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]])
        # fmt: on

        torch.testing.assert_close(output_tokens.cpu(), EXPECTED_OUTPUT_TOKENS)

    @slow
    @require_torch_gpu
    def test_1b_model_integration_generate_batched(self):
        """
        Test the generated tokens match the ones from the original model implementation.
        Such tokens are to be retreived using https://gist.github.com/eustlb/bcc532b53161bc31da3d66cb07ae193f, which is a script that infers the original model.
        """
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)

        ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
        conversation = [
            [
                {
                    "role": f"{ds[0]['speaker_id']}",
                    "content": [
                        {"type": "text", "text": ds[0]["text"]},
                        {"type": "audio", "path": ds[0]["audio"]["array"]},
                    ],
                },
                {
                    "role": f"{ds[1]['speaker_id']}",
                    "content": [
                        {"type": "text", "text": ds[1]["text"]},
                    ],
                },
            ],
            [
                {
                    "role": f"{ds[0]['speaker_id']}",
                    "content": [
                        {"type": "text", "text": ds[0]["text"]},
                    ],
                }
            ],
        ]

        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        ).to(torch_device)

        model = CsmForConditionalGeneration.from_pretrained(self.model_checkpoint, device_map=torch_device)
        output_tokens = model.generate(**inputs, do_sample=False, depth_decoder_do_sample=False)

        # fmt: off
        EXPECTED_OUTPUT_TOKENS = torch.tensor([
            [
                [1140, 10, 37, 1180, 1100, 1319, 601, 1482, 1918, 1739, 372, 856, 674, 1, 854, 459, 1843, 1191, 347, 349, 1087, 846, 759, 1690, 947, 1280, 580, 1909, 1192, 487, 1302, 1601],
                [1494, 1412, 1824, 1852, 150, 928, 91, 326, 623, 1632, 1163, 1221, 1949, 999, 1779, 248, 693, 1149, 1423, 1503, 1656, 80, 1947, 1666, 933, 1950, 1544, 1577, 1612, 1791, 1883, 765],
                [778, 645, 830, 1051, 524, 1704, 1805, 1438, 211, 906, 691, 814, 1798, 1642, 1042, 284, 1906, 1513, 520, 137, 1052, 1548, 423, 1564, 330, 873, 1381, 188, 317, 1503, 1707, 1744],
                [1416, 864, 242, 1653, 604, 1577, 202, 1808, 926, 1867, 204, 134, 1096, 1765, 496, 1680, 268, 1796, 2024, 1989, 583, 183, 952, 105, 765, 1534, 669, 895, 2008, 11, 1199, 195],
                [1356, 796, 25, 1580, 15, 344, 1730, 99, 1330, 315, 955, 1964, 1731, 543, 1159, 1860, 671, 732, 63, 382, 143, 395, 1749, 1421, 1640, 1340, 650, 100, 171, 1346, 41, 806],
                [1860, 1835, 823, 388, 254, 1734, 1135, 324, 1508, 983, 937, 1703, 1541, 875, 1319, 799, 1259, 1175, 1295, 807, 261, 760, 1916, 1606, 1616, 1894, 1605, 441, 387, 167, 2016, 222],
                [1165, 919, 1318, 54, 1727, 1766, 777, 1128, 623, 353, 1840, 241, 977, 424, 1055, 898, 395, 655, 1695, 1084, 1346, 616, 1028, 1927, 603, 858, 758, 1539, 0, 1655, 1853, 1661],
                [902, 1746, 1318, 298, 1982, 1184, 775, 328, 1676, 871, 133, 1374, 1927, 1984, 698, 1037, 100, 1884, 1596, 429, 1794, 2046, 105, 2037, 1767, 178, 176, 1293, 1893, 1780, 1832, 1382],
                [1932, 714, 1084, 1167, 624, 509, 1213, 651, 1000, 1686, 1537, 555, 461, 623, 1433, 1089, 1212, 1628, 834, 1111, 943, 1816, 1947, 1063, 354, 1843, 1741, 2015, 404, 928, 1488, 168],
                [1437, 314, 1356, 404, 1274, 2016, 998, 1350, 155, 553, 368, 1501, 1431, 1563, 1105, 1353, 535, 908, 1305, 1214, 1656, 65, 1469, 1517, 480, 252, 1289, 696, 302, 632, 246, 72],
                [724, 848, 1140, 927, 1669, 296, 447, 1708, 1898, 685, 1041, 1685, 708, 1510, 1623, 876, 11, 99, 43, 586, 1705, 1753, 1477, 1191, 583, 1249, 1613, 992, 1319, 677, 418, 668],
                [925, 54, 1810, 674, 1306, 848, 573, 1772, 105, 301, 1753, 989, 440, 1057, 823, 1313, 1663, 750, 1477, 102, 1437, 1114, 399, 1440, 319, 118, 1827, 295, 1429, 139, 1594, 55],
                [629, 149, 784, 838, 984, 604, 685, 1229, 1432, 859, 1526, 1336, 1949, 281, 988, 1260, 52, 6, 1216, 1542, 1426, 1938, 253, 280, 1319, 794, 901, 843, 615, 437, 814, 20],
                [1281, 502, 1237, 404, 625, 1444, 397, 1999, 2016, 1686, 533, 1785, 1152, 1245, 579, 1906, 1204, 549, 1334, 536, 1351, 1979, 208, 111, 2011, 751, 677, 1948, 1772, 1525, 2038, 419],
                [9, 490, 869, 2026, 1928, 1489, 587, 549, 1241, 460, 1458, 1636, 924, 222, 1246, 480, 706, 398, 75, 1717, 604, 1446, 333, 237, 805, 1446, 421, 1343, 78, 1260, 1872, 1116],
                [1944, 755, 375, 332, 1464, 828, 1273, 579, 1457, 353, 1510, 1910, 1609, 705, 400, 1666, 227, 1544, 1270, 136, 1857, 1975, 1762, 2006, 1102, 221, 1965, 151, 2041, 198, 1830, 287],
                [221, 502, 440, 247, 181, 1912, 42, 357, 1883, 596, 919, 953, 1774, 772, 915, 188, 438, 1226, 544, 1313, 726, 1298, 85, 677, 566, 1581, 30, 341, 878, 1732, 591, 1446],
                [1178, 1690, 320, 1746, 1798, 685, 1941, 666, 832, 623, 1907, 128, 337, 1779, 824, 923, 1041, 287, 1165, 437, 1803, 1222, 870, 646, 358, 220, 2009, 735, 468, 1908, 1349, 1603],
                [1432, 1286, 540, 1687, 1741, 951, 299, 1233, 1061, 1128, 985, 953, 1917, 198, 2031, 1559, 1096, 1455, 780, 437, 163, 1268, 649, 1029, 1081, 1518, 304, 1638, 814, 364, 140, 1385],
                [905, 463, 1739, 1063, 351, 936, 1652, 101, 1323, 1731, 298, 1193, 266, 1554, 1837, 1659, 409, 1739, 1012, 725, 851, 1909, 213, 1918, 1759, 1561, 1250, 970, 1571, 352, 911, 195],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
            [
                [1375, 203, 265, 164, 200, 1867, 976, 924, 1972, 1637, 1048, 271, 1912, 1430, 853, 1942, 260, 1642, 400, 57, 1376, 1626, 1821, 1163, 619, 777, 1076, 951, 389, 1820, 84, 1417],
                [914, 527, 286, 968, 305, 1314, 805, 1703, 87, 559, 1980, 1124, 1726, 36, 1139, 618, 1628, 519, 1943, 781, 400, 1265, 438, 113, 87, 856, 465, 162, 1099, 352, 1141, 274],
                [1408, 6, 126, 2009, 90, 996, 934, 134, 1857, 126, 602, 876, 1092, 1962, 1205, 828, 707, 1063, 393, 1533, 123, 1086, 1749, 1324, 1, 1763, 1707, 1191, 34, 1323, 1017, 1787],
                [1000, 683, 1630, 703, 1574, 587, 25, 1049, 213, 1270, 1641, 1072, 1892, 1634, 1603, 90, 867, 2037, 1021, 715, 206, 507, 1138, 959, 1822, 1785, 280, 1100, 1660, 251, 1903, 988],
                [1657, 1981, 246, 1048, 1952, 451, 305, 423, 2000, 416, 756, 1748, 7, 748, 1866, 1795, 1682, 1832, 338, 212, 1685, 518, 154, 1407, 416, 765, 776, 25, 55, 458, 612, 262],
                [1034, 564, 667, 1474, 1212, 350, 712, 941, 1151, 1182, 1280, 640, 924, 1722, 1816, 458, 226, 359, 1518, 102, 1203, 459, 676, 1788, 1110, 393, 1974, 1721, 795, 1459, 798, 1723],
                [742, 1616, 119, 653, 441, 679, 246, 1432, 486, 1615, 1191, 500, 650, 223, 687, 1765, 1875, 963, 1385, 863, 151, 1771, 458, 1170, 737, 1932, 785, 1954, 1067, 16, 1986, 2029],
                [1437, 1078, 1767, 1452, 1392, 45, 2010, 1664, 245, 2015, 1416, 1055, 457, 985, 740, 1594, 1562, 1838, 258, 1431, 701, 604, 1813, 352, 792, 632, 21, 895, 70, 609, 850, 1599],
                [983, 1961, 54, 135, 846, 711, 473, 1630, 1373, 1094, 251, 525, 632, 1014, 1594, 1594, 1752, 398, 1266, 1357, 942, 1680, 191, 874, 483, 1291, 381, 1873, 1964, 1278, 1477, 122],
                [1663, 1969, 1887, 113, 145, 251, 1133, 156, 245, 1641, 209, 1322, 2037, 836, 539, 667, 940, 797, 1758, 1357, 191, 1137, 587, 1699, 27, 701, 395, 99, 1682, 876, 762, 839],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ])
        # fmt: on

        torch.testing.assert_close(output_tokens.cpu(), EXPECTED_OUTPUT_TOKENS)
