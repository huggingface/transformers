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
    require_read_token,
    require_torch_accelerator,
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


@require_read_token
class CsmForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        # TODO: @eustlb, update with correct sesame's repo
        self.model_checkpoint = "sesame/csm-1b"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def _load_conversation(self):
        ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
        ds = ds.filter(lambda x: x["conversation_id"] == 0)
        ds = ds.sort("turn_id")
        return ds[0]

    @slow
    @require_torch_accelerator
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
            [1140, 1818, 86, 1072, 1029, 1010, 796, 577, 1523, 1599, 902, 1308, 817, 232, 1860, 56, 327, 1399, 1069, 1014, 1980, 53, 407, 1841, 1559, 928, 972, 1432, 832, 1007, 1325, 371],
            [955, 1390, 1503, 861, 265, 1753, 91, 1690, 389, 1025, 1086, 495, 1192, 1334, 773, 1277, 957, 1388, 513, 1110, 539, 349, 1865, 1515, 806, 1514, 237, 1424, 1783, 1928, 523, 1925],
            [1925, 190, 654, 1538, 19, 37, 1923, 100, 1909, 1156, 1847, 1901, 975, 982, 2002, 544, 1933, 311, 79, 850, 238, 1034, 428, 1231, 764, 313, 973, 269, 1669, 1058, 1641, 891],
            [1721, 92, 1298, 989, 1868, 154, 386, 1115, 347, 384, 853, 1439, 970, 1369, 238, 1279, 268, 595, 2010, 1861, 723, 999, 578, 1612, 69, 121, 306, 1647, 1609, 1185, 1786, 1268],
            [1356, 1419, 1199, 1575, 418, 53, 1140, 805, 355, 324, 633, 199, 343, 1176, 784, 41, 268, 366, 1478, 466, 1591, 305, 1298, 1335, 1866, 1563, 1503, 1558, 1468, 852, 1244, 312],
            [1860, 1603, 546, 1805, 607, 160, 1528, 191, 1867, 1830, 861, 661, 1740, 1276, 218, 954, 1286, 1216, 1727, 1637, 983, 597, 1857, 65, 797, 947, 427, 476, 739, 978, 107, 1394],
            [1165, 1775, 177, 823, 100, 370, 521, 200, 2007, 434, 1444, 1205, 819, 1278, 31, 912, 150, 1546, 2035, 1147, 559, 1995, 639, 35, 1812, 56, 1485, 2003, 1573, 1693, 1762, 1313],
            [1932, 704, 907, 897, 56, 1587, 990, 1905, 2007, 256, 671, 868, 282, 1731, 460, 1055, 1309, 1880, 584, 1849, 1643, 1198, 310, 361, 789, 1657, 905, 1564, 1354, 110, 915, 1011],
            [1437, 1958, 1483, 313, 79, 28, 859, 397, 1783, 1693, 633, 1424, 1128, 1831, 605, 1123, 1496, 739, 1177, 498, 781, 1756, 1288, 890, 224, 1875, 279, 800, 1999, 1740, 348, 1420],
            [724, 870, 1344, 861, 429, 522, 1877, 1689, 771, 1468, 1952, 156, 856, 462, 18, 834, 33, 840, 1136, 2012, 1766, 1891, 2034, 1731, 624, 108, 1469, 653, 1344, 1682, 407, 515],
            [355, 26, 36, 1700, 1032, 293, 1799, 978, 944, 296, 1333, 1377, 664, 1249, 421, 516, 1178, 531, 1587, 899, 1, 1449, 934, 942, 1604, 1208, 1889, 710, 825, 2012, 1563, 1299],
            [629, 15, 551, 861, 310, 918, 149, 1689, 1464, 1950, 1900, 1502, 1503, 615, 477, 1090, 1556, 1393, 1143, 1112, 1934, 416, 1604, 1470, 1501, 1594, 903, 1400, 972, 199, 1075, 1643],
            [1281, 106, 1162, 1313, 115, 429, 1792, 1379, 1535, 1311, 743, 484, 333, 498, 547, 699, 1075, 1861, 1038, 1352, 166, 622, 759, 1398, 241, 138, 1330, 481, 1254, 1365, 985, 423],
            [9, 520, 323, 25, 1873, 716, 1414, 1413, 266, 1449, 1265, 290, 1341, 836, 674, 411, 913, 911, 637, 1038, 1097, 1158, 1009, 803, 737, 154, 1388, 938, 466, 725, 1216, 1549],
            [1944, 15, 62, 332, 540, 689, 106, 1805, 1303, 1787, 1724, 1011, 1515, 1442, 1197, 496, 2026, 1820, 906, 372, 322, 1413, 1305, 1674, 443, 1733, 828, 905, 1116, 1850, 1870, 786],
            [221, 220, 1093, 1790, 759, 1266, 1169, 1379, 572, 1859, 1155, 596, 1398, 412, 1788, 1963, 167, 89, 1011, 1489, 714, 73, 486, 780, 1136, 254, 983, 138, 386, 800, 1819, 1857],
            [1178, 1939, 107, 1605, 582, 1256, 420, 637, 648, 1023, 1809, 978, 1703, 278, 1668, 2044, 1599, 1321, 1670, 1716, 1155, 56, 602, 877, 886, 220, 910, 797, 1028, 1226, 869, 811],
            [1432, 1926, 1197, 1687, 540, 1815, 658, 1080, 1162, 192, 315, 1713, 422, 586, 65, 947, 493, 1536, 13, 505, 1269, 456, 1042, 645, 512, 1394, 1124, 590, 1058, 1896, 1055, 1537],
            [905, 564, 1739, 1594, 1201, 1773, 738, 994, 239, 1686, 1528, 368, 1791, 1924, 607, 44, 1320, 552, 1862, 1578, 591, 1434, 330, 1576, 1946, 1233, 113, 445, 669, 2041, 1242, 1406],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]])
        # fmt: on

        torch.testing.assert_close(output_tokens.cpu(), EXPECTED_OUTPUT_TOKENS)

    @slow
    @require_torch_accelerator
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
    @require_torch_accelerator
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
            [1741, 1621, 733, 1580, 1006, 482, 1508, 1722, 1529, 1822, 745, 552, 142, 1568, 704, 480, 214, 552, 321, 1858, 1902, 1042, 1249, 1328, 1730, 1218, 1755, 597, 670, 738, 1056, 762],
            [1264, 1561, 1307, 730, 1403, 688, 212, 949, 1871, 994, 1174, 674, 858, 293, 1577, 1221, 1024, 1535, 1224, 872, 509, 1971, 46, 440, 1531, 1100, 1466, 732, 964, 381, 1933, 1612],
            [1407, 982, 1665, 1247, 1636, 1546, 939, 882, 1999, 618, 484, 1632, 66, 430, 290, 327, 351, 1236, 687, 504, 1973, 1073, 1233, 1972, 82, 1655, 361, 1612, 861, 1085, 880, 1407],
            [584, 637, 304, 1805, 1683, 1381, 404, 862, 1278, 916, 1695, 370, 316, 1049, 237, 1187, 1389, 300, 680, 135, 1068, 1368, 810, 1392, 103, 1459, 1051, 644, 38, 1517, 790, 646],
            [471, 1984, 1333, 553, 193, 319, 1604, 1546, 153, 513, 990, 839, 1714, 1998, 984, 1882, 1055, 476, 1821, 1476, 1522, 1817, 949, 1923, 1416, 1885, 1832, 1368, 1782, 1229, 436, 918],
            [28, 1238, 489, 1580, 596, 1232, 840, 835, 297, 762, 474, 1106, 1761, 483, 1165, 923, 1184, 1181, 1724, 398, 1484, 860, 1945, 665, 1925, 14, 67, 1693, 1853, 1283, 1822, 1973],
            [20, 637, 253, 1254, 738, 188, 593, 1239, 1768, 1047, 1703, 1512, 1398, 464, 13, 161, 651, 1844, 666, 210, 1510, 1798, 614, 1649, 1751, 341, 808, 915, 1965, 840, 778, 950],
            [1879, 2028, 1405, 694, 432, 2036, 612, 387, 1843, 1204, 1044, 8, 1538, 542, 1198, 598, 1131, 760, 1217, 901, 800, 1046, 136, 639, 1320, 618, 606, 707, 574, 1288, 1254, 198],
            [1874, 937, 1063, 1341, 254, 13, 359, 888, 1837, 1246, 980, 818, 2046, 1258, 1290, 1470, 2028, 1701, 228, 1766, 51, 93, 296, 991, 1094, 1694, 156, 1207, 401, 967, 867, 211],
            [1762, 426, 1749, 2004, 314, 903, 1254, 220, 1330, 1813, 534, 102, 658, 1460, 603, 1046, 402, 2005, 783, 973, 1764, 210, 1458, 803, 605, 369, 669, 352, 1964, 1549, 632, 1375],
            [1577, 386, 503, 1492, 604, 405, 1329, 349, 180, 875, 329, 196, 514, 1854, 925, 159, 1428, 1300, 1510, 329, 76, 1682, 1036, 854, 695, 1097, 816, 382, 1417, 697, 1693, 194],
            [1109, 848, 1385, 126, 1136, 979, 687, 130, 2045, 140, 562, 361, 921, 1706, 1060, 1723, 165, 1304, 203, 1067, 158, 692, 980, 313, 1896, 1812, 839, 837, 985, 116, 866, 1049],
            [1810, 1092, 1534, 1730, 773, 2044, 1098, 1326, 85, 249, 455, 1728, 860, 443, 1841, 1885, 1698, 864, 1747, 1083, 1591, 1785, 1577, 1001, 1025, 1837, 1504, 1839, 1900, 1932, 230, 968],
            [1547, 1465, 896, 794, 613, 1383, 1806, 1984, 526, 671, 100, 519, 2037, 1631, 1724, 633, 824, 994, 893, 1448, 1793, 1237, 1855, 699, 349, 143, 270, 535, 1550, 101, 22, 1311],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]])
        # fmt: on

        torch.testing.assert_close(output_tokens.cpu(), EXPECTED_OUTPUT_TOKENS)

    @slow
    @require_torch_accelerator
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
                [1140, 1818, 1713, 1072, 1029, 1185, 697, 358, 220, 481, 1127, 1779, 817, 891, 958, 1058, 672, 495, 426, 1135, 236, 1440, 829, 2023, 1097, 94, 926, 1830, 114, 307, 235, 1190],
                [955, 968, 696, 676, 52, 618, 0, 1818, 1285, 143, 1733, 1268, 1317, 1510, 1027, 2033, 1276, 1744, 790, 638, 1179, 1125, 650, 266, 1180, 364, 1015, 1604, 1152, 154, 178, 284],
                [1925, 274, 433, 273, 1391, 1528, 1683, 1120, 976, 944, 357, 1681, 847, 1783, 546, 857, 1662, 1695, 40, 152, 2039, 1076, 994, 1743, 265, 1751, 602, 981, 483, 981, 538, 1381],
                [1908, 1625, 1975, 729, 1067, 1844, 837, 1849, 224, 1223, 1037, 1188, 1428, 1977, 317, 530, 990, 1670, 766, 1411, 811, 154, 433, 1645, 1565, 1291, 1390, 49, 1160, 1464, 1911, 1961],
                [1908, 566, 175, 1387, 1437, 1873, 1785, 1536, 961, 414, 406, 1753, 835, 284, 764, 1522, 1889, 1816, 840, 440, 756, 860, 1753, 516, 601, 1498, 280, 1425, 1904, 1540, 1074, 314],
                [1860, 296, 1766, 361, 1155, 1675, 528, 1975, 1286, 113, 1656, 237, 372, 580, 1571, 1958, 502, 893, 1300, 261, 313, 455, 693, 1658, 654, 1585, 1723, 721, 178, 679, 908, 1077],
                [1165, 1787, 1877, 1904, 85, 609, 1007, 1724, 1959, 245, 645, 463, 1321, 1695, 192, 711, 1892, 1193, 302, 1835, 69, 940, 148, 913, 110, 108, 1244, 1510, 165, 726, 745, 1746],
                [1405, 1410, 186, 1569, 1214, 1920, 1946, 1907, 990, 1152, 1401, 1713, 541, 115, 423, 616, 1191, 1149, 1122, 9, 303, 195, 906, 566, 1718, 668, 1637, 1975, 51, 2005, 1260, 1672],
                [1932, 780, 143, 110, 286, 1460, 1136, 1366, 1788, 446, 645, 587, 1708, 189, 1295, 526, 1667, 735, 707, 1215, 27, 834, 1865, 182, 1776, 1130, 528, 1523, 1156, 316, 492, 1666],
                [1437, 364, 314, 432, 575, 1640, 529, 1128, 973, 789, 1820, 808, 1317, 1681, 347, 471, 737, 1626, 1386, 75, 433, 517, 365, 1982, 1434, 1378, 1059, 56, 1475, 653, 1507, 861],
                [724, 538, 1140, 1853, 76, 402, 0, 397, 330, 1787, 1382, 682, 1134, 296, 377, 997, 705, 627, 1700, 17, 1791, 1000, 1271, 1019, 1552, 1521, 668, 534, 433, 344, 1007, 1046],
                [925, 1297, 1017, 1785, 1403, 520, 1603, 1908, 665, 1827, 951, 1588, 1526, 414, 1945, 1153, 1933, 1571, 1821, 104, 179, 769, 619, 117, 56, 790, 721, 992, 1284, 1495, 1459, 823],
                [629, 1208, 689, 924, 1617, 1100, 1028, 1231, 1708, 1582, 200, 2011, 1611, 1966, 1153, 1326, 2036, 1515, 884, 1790, 581, 549, 1491, 701, 973, 836, 2031, 1249, 1411, 365, 1946, 1552],
                [1281, 1305, 610, 1666, 676, 544, 1788, 315, 159, 809, 1333, 1785, 1159, 1084, 1356, 318, 1933, 854, 475, 638, 1616, 1801, 1816, 1921, 283, 1745, 814, 974, 1056, 1316, 1509, 2031],
                [9, 212, 1590, 163, 1289, 923, 2046, 1620, 632, 127, 963, 405, 850, 471, 1430, 108, 1845, 1196, 1928, 143, 1717, 1054, 1288, 1351, 1340, 1294, 831, 480, 1562, 2004, 483, 1776],
                [221, 142, 1555, 1434, 1481, 1371, 1873, 1607, 207, 631, 1042, 1084, 472, 465, 1772, 1002, 1761, 1912, 1298, 1918, 685, 1053, 1635, 1536, 497, 55, 1432, 1394, 1512, 365, 2026, 1210],
                [1741, 1923, 930, 1423, 1258, 1227, 879, 1217, 1999, 422, 420, 1832, 1660, 1542, 92, 2000, 1790, 1909, 56, 695, 704, 1752, 371, 792, 625, 328, 567, 1397, 1557, 390, 1424, 14],
                [1178, 812, 577, 895, 1386, 339, 1467, 844, 235, 703, 551, 2021, 1592, 1042, 353, 621, 1672, 653, 2029, 103, 766, 182, 2016, 1921, 556, 1092, 1579, 626, 1950, 70, 1467, 850],
                [1352, 472, 577, 351, 1126, 1943, 52, 2028, 430, 1017, 1136, 645, 820, 2028, 723, 1385, 1922, 323, 106, 267, 438, 1064, 202, 1249, 244, 1962, 625, 1380, 476, 924, 1221, 1854],
                [905, 811, 374, 2021, 1067, 675, 927, 427, 416, 1521, 663, 77, 457, 1849, 1362, 262, 1669, 1238, 286, 102, 555, 1809, 1585, 1918, 972, 1446, 688, 523, 1904, 943, 17, 904],
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
