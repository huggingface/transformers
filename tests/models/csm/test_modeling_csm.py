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

import unittest

import pytest
import inspect
import copy
import tempfile
import requests
import collections
import re
from parameterized import parameterized

from transformers import (
    CsmConfig,
    CsmBackboneConfig,
    CsmDepthDecoderConfig,
    CsmBackboneModel,
    CsmDepthDecoderModel,
    CsmDepthDecoderForCausalLM,
    CsmForCausalLM,
    AutoProcessor,
    is_torch_available,
    is_vision_available,
)
from transformers.cache_utils import Cache
from transformers.models.mllama.configuration_mllama import MllamaTextConfig
from transformers.testing_utils import (
    cleanup,
    require_bitsandbytes,
    require_read_token,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
    require_torch_sdpa,
    is_flaky,
)
from transformers.utils.import_utils import is_datasets_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION

if is_datasets_available():
    import datasets
    from datasets import Audio, load_dataset

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
        backbone_config={
            "num_codebooks": 10,
            "codebook_vocab_size": 6,
            "vocab_size": 99,
            "hidden_size": 64,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_act": "silu",
            "max_position_embeddings": 10,
            "pad_token_id": 2,
            "codebook_pad_token_id": 2,
            "codebook_eos_token_id": 3,
        },
    ):
        self.parent = parent
        self.is_training = is_training
        self.ignore_index = ignore_index
        self.depth_decoder_config = depth_decoder_config
        self.backbone_config = backbone_config
        self.seq_length = seq_length
        self.batch_size = batch_size

        self.num_hidden_layers = backbone_config["num_hidden_layers"]
        self.vocab_size = backbone_config["vocab_size"]
        self.hidden_size = backbone_config["hidden_size"]
        self.num_attention_heads = backbone_config["num_attention_heads"]
        self.pad_token_id = self.backbone_config["pad_token_id"] 

    def get_config(self):
        return CsmConfig(
            depth_decoder_config=self.depth_decoder_config,
            backbone_config=self.backbone_config,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()

        num_codebooks = self.backbone_config["num_codebooks"]
        text_input_ids = ids_tensor([self.batch_size, self.seq_length], config.backbone_config.vocab_size - 1) + 1
        codebook_input_ids = ids_tensor([self.batch_size, self.seq_length, num_codebooks], config.backbone_config.codebook_vocab_size - 1) + 1
        input_ids = torch.cat([codebook_input_ids, text_input_ids.unsqueeze(-1)], dim=-1)

        attention_mask = input_ids[..., -1].ne(1).to(torch_device)

        return config, input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict


class CsmForCausalLMTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (CsmForCausalLM,) if is_torch_available() else ()
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
                (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.backbone_config["num_codebooks"] + 1), dtype=torch.long, device=torch_device
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


    @pytest.mark.generate
    def test_left_padding_compatibility(self):
        """
        Overrides [GenerationTesterMixin.test_left_padding_compatibility] to handle third input_ids dimension.
        """
        # NOTE: left-padding results in small numerical differences. This is expected.
        # See https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535

        # First, filter out models that don't support left padding
        # - The model must have generative capabilities
        if len(self.all_generative_model_classes) == 0:
            self.skipTest(reason="No generative architecture available for this model.")

        # - The model must support padding
        if not self.has_attentions:
            self.skipTest(reason="This model doesn't support padding.")

        # - The model must be a decoder-only architecture (encoder-based architectures use right-padding)
        decoder_only_classes = []
        for model_class in self.all_generative_model_classes:
            config, _ = self.prepare_config_and_inputs_for_generate()
            if config.is_encoder_decoder:
                continue
            else:
                decoder_only_classes.append(model_class)
        if len(decoder_only_classes) == 0:
            self.skipTest(reason="No decoder-only architecture available for this model.")

        # - Decoder-only architectures derived from encoder-decoder models could support it in theory, but we haven't
        #   added support for it yet. We skip these models for now.
        has_encoder_attributes = any(
            attr_name
            for attr_name in config.to_dict().keys()
            if attr_name.startswith("encoder") and attr_name != "encoder_no_repeat_ngram_size"
        )
        if has_encoder_attributes:
            self.skipTest(
                reason="The decoder-only derived from encoder-decoder models are not expected to support left-padding."
            )

        # Then, test left-padding
        def _prepare_model_kwargs(input_ids, attention_mask, signature):
            model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "position_ids" in signature:
                position_ids = torch.cumsum(attention_mask, dim=-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                model_kwargs["position_ids"] = position_ids
            if "cache_position" in signature:
                cache_position = torch.arange(input_ids.shape[1], device=torch_device)
                model_kwargs["cache_position"] = cache_position
            return model_kwargs

        for model_class in decoder_only_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            input_ids = inputs_dict["input_ids"]
            attention_mask = inputs_dict.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            model = model_class(config).to(torch_device).eval()
            signature = inspect.signature(model.forward).parameters.keys()

            # no cache as some models require special cache classes to be init outside forward
            model.generation_config.use_cache = False

            # Without padding
            model_kwargs = _prepare_model_kwargs(input_ids, attention_mask, signature)
            next_logits_wo_padding = model(**model_kwargs).logits[:, -1, :]

            # With left-padding (length 32)
            # can hardcode pad_token to be 0 as we'll do attn masking anyway
            pad_token_id = (
                config.get_text_config().pad_token_id if config.get_text_config().pad_token_id is not None else 0
            )
            pad_size = (input_ids.shape[0], 32, model.config.num_codebooks + 1)
            pad_size_attention_mask = (input_ids.shape[0], 32)
            padding = torch.ones(pad_size, dtype=input_ids.dtype, device=torch_device) * pad_token_id
            padded_input_ids = torch.cat((padding, input_ids), dim=1)
            padded_attention_mask = torch.cat(
                (torch.zeros(pad_size_attention_mask, dtype=attention_mask.dtype, device=torch_device), attention_mask),
                dim=1
            )
            model_kwargs = _prepare_model_kwargs(padded_input_ids, padded_attention_mask, signature)
            next_logits_with_padding = model(**model_kwargs).logits[:, -1, :]

            # They should result in very similar logits
            torch.testing.assert_close(next_logits_wo_padding, next_logits_with_padding, rtol=1e-5, atol=1e-5)

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
        input_ids = torch.tensor(
            [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]], device=torch_device, dtype=torch.int64
        )
        input_ids = input_ids.unsqueeze(-1).expand(-1, -1, self.model_tester.backbone_config["num_codebooks"] + 1)
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


class CsmForCausalLMIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_checkpoint = "/home/eustache_lebihan/add-sesame/eustlb/csm-1b"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def _load_conversation(self):
        ds = load_dataset("eustlb/dailytalk-dummy", split="train")
        ds = ds.filter(lambda x: x["conversation_id"] == 0)
        ds = ds.sort('turn_id')
        return ds[0]

    @slow
    @require_torch_gpu
    def test_1b_model_integration_generate(self):
        """
        Test the generated tokens match the ones from the original model implementation.
        Such tokens are to be retreived using https://gist.github.com/eustlb/d25577a357ddcf8f4a8cd0d00baca551, which is a script that infers the original model.
        """
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)
        prompt = "<|begin_of_text|>[0]What are you working on?<|end_of_text|><|AUDIO|><|audio_eos|><|begin_of_text|>[1]I'm figuring out my budget.<|end_of_text|>"
        
        ds = load_dataset("eustlb/dailytalk-dummy", split="train")
        audio = ds[0]["audio"]["array"]
        inputs = processor(text=prompt, audio=audio, return_tensors="pt").to(torch_device)

        model = CsmForCausalLM.from_pretrained(self.model_checkpoint, device_map=torch_device)
        output_tokens = model.generate(**inputs, do_sample=False)

        # skip input_ids
        output_tokens = output_tokens[:, inputs["input_ids"].shape[1]:, :].cpu()

        # fmt: off
        EXPECTED_OUTPUT_TOKENS = torch.tensor([
            [[1140, 10, 37, 1180, 1100, 1319, 601, 1482, 1918, 1739, 372, 856, 674, 1, 854, 459, 1843, 1191, 347, 349, 1087, 846, 759, 1690, 947, 1280, 580, 1909, 1192, 487, 1302, 1601, 128002],
            [1494, 1412, 1824, 1852, 150, 928, 91, 326, 623, 1632, 1163, 1221, 1949, 999, 1779, 248, 693, 1149, 1423, 1503, 598, 80, 223, 1798, 251, 385, 1391, 1692, 1228, 1631, 1101, 866, 128002],
            [778, 645, 830, 1812, 524, 1704, 1805, 1289, 74, 1069, 243, 1622, 1755, 1281, 1397, 620, 1962, 1995, 253, 1124, 1007, 518, 89, 559, 1304, 1482, 523, 1747, 1979, 1003, 1707, 1578, 128002],
            [1356, 481, 642, 989, 287, 1819, 171, 1115, 824, 1253, 1488, 1074, 1019, 342, 279, 513, 1275, 1364, 893, 2007, 553, 407, 882, 1170, 1586, 485, 762, 559, 100, 542, 911, 1460, 128002],
            [1860, 593, 1944, 404, 575, 545, 862, 830, 1002, 125, 2010, 268, 1779, 804, 811, 809, 255, 373, 387, 1756, 259, 822, 1191, 700, 1686, 390, 1676, 844, 2006, 286, 1376, 719, 128002],
            [1165, 1047, 848, 212, 1018, 1470, 93, 1709, 1487, 1691, 1190, 275, 1278, 2018, 121, 1023, 485, 463, 39, 1825, 1936, 1817, 569, 209, 1553, 1599, 1137, 769, 968, 558, 1957, 265, 128002],
            [902, 1608, 719, 850, 371, 1920, 75, 1917, 2005, 1238, 562, 1743, 713, 95, 1107, 1463, 696, 840, 8, 487, 1950, 1171, 1004, 1516, 1130, 303, 1866, 1728, 2046, 238, 265, 153, 128002],
            [1932, 839, 334, 1167, 134, 2025, 40, 505, 1244, 1238, 1840, 800, 697, 72, 216, 486, 940, 1312, 510, 361, 549, 583, 1364, 844, 397, 1181, 1779, 962, 457, 1782, 1316, 465, 128002],
            [31, 1558, 1048, 404, 354, 7, 827, 414, 1082, 807, 243, 1517, 801, 1364, 99, 1276, 1655, 1488, 1313, 464, 828, 1612, 774, 1558, 745, 1496, 960, 1874, 995, 1943, 255, 213, 128002],
            [355, 1270, 413, 1519, 1659, 1904, 690, 552, 1279, 1821, 2022, 458, 1779, 2003, 604, 832, 661, 1295, 305, 1701, 173, 869, 230, 539, 1188, 669, 117, 692, 250, 388, 1995, 294, 128002],
            [629, 199, 1899, 1123, 1070, 344, 578, 1795, 1451, 1257, 168, 1410, 1120, 1270, 316, 983, 1245, 1870, 165, 471, 966, 1337, 308, 1118, 746, 67, 1767, 1480, 1517, 1585, 871, 1110, 128002],
            [1281, 1173, 784, 404, 368, 403, 580, 526, 853, 1692, 792, 895, 1286, 573, 1368, 896, 931, 1958, 1912, 644, 583, 1706, 1176, 1262, 1637, 315, 524, 1629, 795, 1211, 915, 533, 128002],
            [9, 1783, 621, 1954, 1212, 993, 197, 977, 1662, 1340, 618, 1997, 1689, 1001, 74, 1765, 1865, 797, 1219, 1609, 671, 1491, 950, 1849, 1301, 2031, 875, 323, 203, 1063, 1490, 1538, 128002],
            [1944, 1578, 1256, 1169, 790, 1444, 1382, 1616, 1100, 1264, 214, 1646, 488, 573, 1333, 285, 1954, 74, 1333, 674, 1303, 266, 622, 1290, 402, 109, 1331, 1666, 1347, 780, 106, 605, 128002],
            [221, 161, 1322, 1, 565, 1507, 1403, 1091, 1557, 932, 1664, 1165, 1828, 1647, 2008, 1616, 648, 1113, 1870, 22, 734, 1458, 1940, 1756, 1689, 925, 1318, 1095, 985, 473, 604, 1974, 128002],
            [1178, 597, 1804, 747, 1383, 360, 1497, 406, 1053, 1023, 1901, 56, 1221, 628, 75, 1729, 575, 1681, 840, 410, 650, 794, 1171, 1889, 187, 54, 1364, 1390, 505, 1285, 1814, 90, 128002],
            [1432, 1221, 1800, 1873, 1255, 627, 41, 9, 630, 896, 1469, 1195, 1098, 145, 442, 1460, 13, 57, 2039, 1015, 149, 461, 1084, 1288, 1099, 910, 63, 157, 906, 111, 1394, 460, 128002],
            [1352, 593, 307, 780, 1614, 1675, 1491, 1253, 723, 1793, 1032, 1486, 1805, 1904, 777, 398, 1791, 951, 770, 499, 1858, 244, 1372, 1514, 1858, 1200, 69, 181, 673, 1144, 1938, 1191, 128002],
            [905, 403, 1626, 1529, 581, 1443, 976, 754, 1561, 1370, 1048, 253, 194, 1271, 853, 959, 1532, 30, 286, 1594, 1255, 1135, 1410, 1699, 1423, 2002, 260, 69, 941, 1640, 895, 722, 128002],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128002]]
        ])
        # fmt: on

        torch.testing.assert_close(output_tokens, EXPECTED_OUTPUT_TOKENS)

    # TODO: test batched generation ? 
        


