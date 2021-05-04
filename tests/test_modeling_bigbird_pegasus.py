# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch BigBirdPegasus model. """


import copy
import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_generation_utils import GenerationTesterMixin
from .test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        BigBirdPegasusConfig,
        BigBirdPegasusForCausalLM,
        BigBirdPegasusForConditionalGeneration,
        BigBirdPegasusForQuestionAnswering,
        BigBirdPegasusForSequenceClassification,
        BigBirdPegasusModel,
        PegasusTokenizer,
    )
    from transformers.models.bigbird_pegasus.modeling_bigbird_pegasus import (
        BigBirdPegasusDecoder,
        BigBirdPegasusEncoder,
    )

MODEL_ID = "google/bigbird-pegasus-large-pubmed"
# TODO
# bigbird fast tests will have problem of attention type switching


def prepare_bigbird_pegasus_inputs_dict(
    config,
    input_ids,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
):
    if attention_mask is None:
        attention_mask = input_ids.ne(config.pad_token_id)
    if decoder_attention_mask is None:
        decoder_attention_mask = decoder_input_ids.ne(config.pad_token_id)
    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": attention_mask,
    }


@require_torch
class BigBirdPegasusModelTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        seq_length=256,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=31,
        hidden_act="gelu_fast",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=260,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
        attention_type="original_full",
        use_bias=True,
        block_size=16,
        num_random_blocks=3,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

        self.attention_type = attention_type
        self.use_bias = use_bias
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).clamp(
            3,
        )
        input_ids[:, -1] = self.eos_token_id  # Eos Token

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = BigBirdPegasusConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            attention_type=self.attention_type,
            use_bias=self.use_bias,
            block_size=self.block_size,
            num_random_blocks=self.num_random_blocks,
        )
        inputs_dict = prepare_bigbird_pegasus_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = BigBirdPegasusModel(config=config).get_decoder().to(torch_device).eval()
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([attention_mask, next_attn_mask], dim=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-2))

    def check_encoder_decoder_model_standalone(self, config, inputs_dict):
        model = BigBirdPegasusModel(config=config).to(torch_device).eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = BigBirdPegasusEncoder.from_pretrained(tmpdirname).to(torch_device)

        encoder_last_hidden_state_2 = encoder(inputs_dict["input_ids"], attention_mask=inputs_dict["attention_mask"])[
            0
        ]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = BigBirdPegasusDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            input_ids=inputs_dict["decoder_input_ids"],
            attention_mask=inputs_dict["decoder_attention_mask"],
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=inputs_dict["attention_mask"],
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)

    def create_and_check_model(self, config, inputs_dict):
        model = BigBirdPegasusModel(config=config).to(torch_device).eval()
        input_ids = inputs_dict["input_ids"]
        decoder_input_ids = inputs_dict["decoder_input_ids"]
        result = model(input_ids, decoder_input_ids=decoder_input_ids, use_cache=True)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    # def create_and_check_for_auto_padding(
    #     self,
    #     config,
    #     input_ids,
    #     token_type_ids,
    #     input_mask,
    #     sequence_labels,
    #     token_labels,
    #     choice_labels,
    # ):
    #     model = BigBirdModel(config)
    #     model.to(torch_device)
    #     model.eval()
    #     result = model(input_ids)
    #     self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    # def create_and_check_for_change_to_full_attn(
    #     self,
    #     config,
    #     input_ids,
    #     token_type_ids,
    #     input_mask,
    #     sequence_labels,
    #     token_labels,
    #     choice_labels,
    # ):
    #     model = BigBirdModel(config)
    #     model.to(torch_device)
    #     model.eval()
    #     result = model(input_ids)
    #     self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
    #     # the config should not be changed
    #     self.parent.assertTrue(model.config.attention_type == "block_sparse")


@require_torch
class BigBirdPegasusModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            BigBirdPegasusModel,
            BigBirdPegasusForConditionalGeneration,
            BigBirdPegasusForSequenceClassification,
            BigBirdPegasusForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (BigBirdPegasusForConditionalGeneration,) if is_torch_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_head_masking = False
    test_missing_keys = True

    def setUp(self):
        self.model_tester = BigBirdPegasusModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BigBirdPegasusConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_save_load_strict(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model2, info = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info["missing_keys"], [])

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_encoder_decoder_model_standalone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.check_encoder_decoder_model_standalone(*config_and_inputs)

    def test_model_various_attn_type(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["original_full", "block_sparse"]:
            config_and_inputs[0].attention_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    # BigBirdPegasusForSequenceClassification does not support inputs_embeds
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in (
            BigBirdPegasusModel,
            BigBirdPegasusForConditionalGeneration,
            BigBirdPegasusForQuestionAnswering,
        ):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)

            wte = model.get_input_embeddings()
            if not self.is_encoder_decoder:
                inputs["inputs_embeds"] = wte(input_ids)
            else:
                inputs["inputs_embeds"] = wte(encoder_input_ids)
                inputs["decoder_inputs_embeds"] = wte(decoder_input_ids)

            with torch.no_grad():
                model(**inputs)[0]

    def test_generate_fp16(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = BigBirdPegasusForConditionalGeneration(config).eval().to(torch_device)
        if torch_device == "cuda":
            model.half()
        model.generate(input_ids, attention_mask=attention_mask)
        model.generate(num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)

    # def test_auto_padding(self):
    #     self.model_tester.seq_length = 241
    #     config_and_inputs = self.model_tester.prepare_config_and_inputs()
    #     self.model_tester.create_and_check_for_auto_padding(*config_and_inputs)

    # def test_for_change_to_full_attn(self):
    #     self.model_tester.seq_length = 9
    #     config_and_inputs = self.model_tester.prepare_config_and_inputs()
    #     self.model_tester.create_and_check_for_change_to_full_attn(*config_and_inputs)


@require_torch
@require_sentencepiece
@require_tokenizers
@slow
class BigBirdPegasusModelIntegrationTests(unittest.TestCase):
    def _get_dummy_input_ids(self):
        # fmt: off
        ids = torch.tensor(
            [[685, 560, 630, 193, 836, 764, 708, 360, 10, 724, 278, 755, 805, 600, 71, 473, 601, 397, 315, 706, 487, 552, 88, 175, 601, 850, 678, 538, 846, 73, 778, 917, 116, 977, 756, 710, 1023, 848, 432, 449, 851, 100, 985, 178, 756, 798, 660, 148, 911, 424, 289, 962, 266, 698, 640, 545, 544, 715, 245, 152, 676, 511, 460, 883, 184, 29, 803, 129, 129, 933, 54, 902, 551, 489, 757, 274, 336, 389, 618, 43, 443, 544, 889, 258, 322, 1000, 938, 58, 292, 871, 120, 780, 431, 83, 92, 897, 399, 612, 566, 909, 634, 939, 85, 204, 325, 775, 965, 48, 640, 1013, 132, 973, 869, 181, 1001, 847, 144, 661, 228, 955, 792, 720, 910, 374, 854, 561, 306, 582, 170, 676, 449, 96, 198, 607, 257, 882, 691, 293, 931, 817, 862, 388, 611, 555, 974, 369, 1000, 918, 202, 384, 513, 907, 371, 556, 955, 384, 24, 700, 131, 378, 99, 575, 932, 735, 124, 964, 595, 943, 740, 149, 210, 563, 412, 783, 42, 59, 706, 37, 779, 87, 44, 873, 12, 771, 308, 81, 33, 183, 129, 807, 276, 175, 555, 372, 185, 445, 489, 590, 287, 281, 638, 771, 516, 95, 227, 876, 270, 881, 297, 329, 20, 608, 841, 411, 451, 249, 181, 324, 1005, 830, 783, 865, 261, 964, 750, 140, 1021, 599, 462, 890, 622, 844, 697, 529, 153, 926, 150, 111, 26, 465, 957, 890, 887, 118, 446, 596, 674, 873, 929, 229, 508, 764, 122, 327, 470, 288, 526, 840, 697, 153, 592, 42, 275, 553, 439, 208, 780, 167, 112, 350, 1018, 130, 736, 887, 813, 217, 382, 25, 68, 979, 1008, 772, 235, 717, 999, 292, 727, 1023, 702, 710, 728, 556, 33, 12, 617, 213, 139, 695, 1004, 422, 638, 669, 624, 489, 771, 540, 980, 218, 664, 822, 308, 175, 149, 950, 542, 580, 548, 808, 394, 74, 298, 920, 900, 815, 731, 947, 877, 772, 800, 778, 395, 540, 430, 200, 424, 62, 342, 866, 45, 803, 931, 89, 34, 646, 233, 768, 37, 769, 460, 291, 198, 895, 950, 255, 81, 447, 137, 190, 130, 210, 369, 292, 377, 348, 169, 885, 805, 177, 538, 324, 872, 509, 804, 115, 799, 30, 754, 290, 147, 274, 222, 341, 510, 515, 70, 358, 909, 557, 886, 766, 323, 624, 92, 342, 424, 552, 972, 663, 415, 658, 711, 968, 275, 861, 44, 84, 434, 810, 94, 175, 406, 202, 858, 499, 481, 988, 330, 541, 1004, 210, 618, 955, 897, 983, 576, 17, 107, 165, 607, 537, 629, 192, 196, 308, 137, 953, 860, 94, 892, 751, 88, 161, 148, 585, 456, 88, 14, 315, 594, 121, 885, 952, 833, 716, 733, 933, 282, 801, 427, 783, 471, 285, 277, 979, 325, 535, 228, 891, 596, 648, 969, 574, 654, 518, 257, 137, 208, 464, 950, 140, 5, 424, 349, 942, 283, 587, 821, 1007, 434, 220, 820, 740, 874, 787, 374, 291, 564, 671, 438, 827, 940, 824, 509, 1021, 787, 942, 856, 450, 327, 491, 54, 817, 95, 60, 337, 667, 637, 164, 571, 946, 107, 202, 301, 782, 890, 839, 551, 680, 649, 14, 1017, 904, 721, 1017, 535, 505, 848, 986, 777, 740, 775, 210, 456, 469, 474, 963, 573, 401, 57, 883, 750, 664, 281, 5, 613, 1005, 306, 344, 543, 567, 154, 789, 354, 358, 698, 408, 412, 30, 930, 372, 822, 632, 948, 855, 503, 8, 618, 1010, 138, 695, 897, 852, 377, 933, 722, 149, 886, 1009, 260, 127, 811, 578, 533, 805, 325, 977, 113, 944, 651, 238, 361, 991, 860, 556, 64, 928, 917, 455, 266, 445, 604, 624, 420, 340, 845, 275, 370, 843, 227, 226, 940, 644, 909, 229, 827, 898, 370, 129, 808, 25, 699, 293, 356, 838, 135, 4, 227, 890, 681, 445, 418, 285, 837, 27, 737, 249, 366, 948, 202, 438, 198, 930, 648, 638, 607, 73, 247, 853, 136, 708, 214, 476, 621, 324, 103, 853, 328, 596, 224, 257, 646, 348, 108, 927, 970, 980, 520, 150, 998, 477, 393, 684, 559, 1, 361, 692, 551, 90, 75, 500, 739, 636, 344, 97, 852, 283, 719, 33, 116, 455, 866, 429, 828, 826, 691, 174, 746, 133, 442, 94, 348, 402, 420, 707, 405, 942, 186, 976, 376, 677, 874, 703, 517, 498, 499, 206, 415, 366, 856, 739, 420, 586, 219, 952, 539, 375, 23, 461, 720, 355, 603, 52, 999, 815, 721, 574, 445, 816, 1019, 105, 641, 395, 972, 910, 328, 607, 519, 686, 246, 415, 528, 170, 167, 310, 940, 595, 392, 221, 834, 682, 835, 115, 861, 335, 742, 220, 247, 101, 416, 222, 179, 509, 175, 606, 627, 674, 781, 737, 746, 849, 67, 457, 1012, 126, 139, 625, 731, 156, 697, 121, 322, 449, 710, 857, 291, 976, 4, 701, 239, 678, 172, 724, 857, 583, 661, 903, 797, 628, 903, 835, 605, 989, 615, 870, 380, 710, 110, 330, 101, 695, 846, 918, 508, 672, 594, 36, 238, 244, 251, 393, 767, 282, 22, 430, 230, 983, 401, 154, 1007, 120, 678, 896, 386, 390, 711, 397, 347, 587, 1020, 951, 79, 831, 585, 200, 814, 134, 560, 700, 171, 452, 139, 755, 314, 476, 346, 388, 126, 719, 851, 198, 699, 901, 18, 710, 448, 351, 665, 644, 326, 425, 165, 571, 178, 440, 665, 674, 915, 866, 463, 754, 136, 950, 748, 47, 497, 1013, 640, 930, 338, 158, 525, 631, 815, 887, 289, 803, 116, 600, 637, 410, 175, 499, 876, 565, 1002, 623, 577, 333, 887, 586, 147, 773, 776, 644, 49, 77, 294, 117, 494, 561, 110, 979, 180, 562, 72, 859, 434, 1007, 286, 516, 75, 597, 491, 322, 888, 533, 209, 43, 499, 29, 411, 856, 181, 305, 963, 615, 778, 259, 373, 877, 746, 858, 381, 886, 613, 91, 69, 618, 523, 13, 617, 226, 422, 168, 929, 379, 290, 923, 100, 218, 307, 345, 211, 789, 735, 669, 585, 275, 410, 921, 552, 235, 636, 285, 665, 659, 708, 173, 724, 302, 823, 1, 139, 708, 903, 732, 868, 442, 967, 916, 163, 51, 243, 871]],  # noqa: E231
            dtype=torch.long,
            device=torch_device,
        )
        # fmt: on
        return ids

    def _get_dummy_target_ids(self):
        # fmt: off
        ids = torch.tensor(
            [[13, 6, 1, 4, 12, 4, 8, 10, 4, 6, 3, 5, 8, 7, 9, 9]],  # noqa: E231
            dtype=torch.long,
            device=torch_device,
        )
        # fmt: on
        return ids

    def test_inference_block_sparse(self):
        model = BigBirdPegasusForConditionalGeneration.from_pretrained(
            MODEL_ID, attention_type="block_sparse", block_size=16, num_random_blocks=3
        )
        model.to(torch_device)

        input_ids = self._get_dummy_input_ids()
        target_ids = self._get_dummy_target_ids()

        outputs = model(input_ids, labels=target_ids)
        prediction_logits = outputs.logits

        self.assertEqual(prediction_logits.shape, torch.Size((1, 16, 96103)))
        # fmt: off
        expected_prediction_logits_slice = torch.tensor(
            [[1.7769, 5.8479, 6.2375, 2.2745, 8.6157, 4.7483, 5.0647, 6.5358, 2.3393, 7.8333, 3.8403, 0.0255, 7.219, 5.2759, 3.097, 6.387, 4.9341, 7.1409, 5.1179, 0.1144, 6.8268, 0.7598, 0.6258, 2.373, 0.4627, -1.9919, 1.8422, 3.4578], [1.8026, 5.9604, 5.954, 2.8642, 9.0608, 4.394, 5.3779, 7.0216, 1.543, 7.8744, 4.4231, -0.0398, 7.6091, 5.6611, 3.3536, 6.8624, 4.7699, 6.5241, 4.8893, 0.5791, 6.8368, 0.1034, 0.0338, 2.9393, 0.5034, -2.5509, 2.0172, 3.2858], [1.8426, 5.9151, 5.5374, 3.0426, 9.1762, 3.6287, 5.3916, 7.4621, 1.2582, 7.9244, 4.694, -0.1308, 7.4725, 5.5385, 3.4598, 7.0422, 4.2455, 5.797, 4.5927, 0.7478, 6.7467, -0.2695, -0.3207, 3.0269, 0.4714, -2.8134, 2.0406, 3.1089], [1.6527, 5.8416, 5.4558, 3.0044, 9.3478, 3.2607, 5.3887, 7.52, 0.9362, 7.8877, 4.8465, -0.1705, 7.3932, 5.6352, 3.5744, 7.2623, 4.0485, 5.2788, 4.5859, 0.8325, 6.6088, -0.3676, -0.6287, 3.1731, 0.4483, -3.1573, 2.0522, 2.8868]],  # noqa: E231
            device=torch_device,
        )
        # fmt: on
        self.assertTrue(
            torch.allclose(prediction_logits[0, 4:8, 128:156], expected_prediction_logits_slice, atol=1e-4)
        )

    def test_inference_full_attn(self):
        model = BigBirdPegasusForConditionalGeneration.from_pretrained(MODEL_ID, attention_type="original_full")
        model.to(torch_device)

        input_ids = self._get_dummy_input_ids()
        target_ids = self._get_dummy_target_ids()

        outputs = model(input_ids, labels=target_ids)
        prediction_logits = outputs.logits

        self.assertEqual(prediction_logits.shape, torch.Size((1, 16, 96103)))
        # fmt: off
        expected_prediction_logits_slice = torch.tensor(
            [[1.3418, 5.8304, 6.5662, 2.0448, 8.7702, 4.6579, 4.9947, 6.429, 2.4296, 7.9431, 4.217, 0.0672, 7.334, 5.1966, 2.9603, 6.0814, 4.6756, 7.5522, 5.076, 0.213, 6.6638, 0.6577, 0.244, 2.1221, 0.7531, -2.4076, 1.8731, 3.5594], [1.5525, 6.0524, 6.309, 2.6245, 9.229, 4.5213, 5.0913, 7.0622, 1.7992, 8.0962, 4.7994, -0.0248, 7.7168, 5.5878, 3.0883, 6.5248, 4.7895, 6.9974, 4.8787, 0.5445, 6.6686, 0.0102, -0.1659, 2.6195, 0.7389, -2.8956, 1.9928, 3.3777], [1.6407, 6.2104, 6.0331, 2.8076, 9.4074, 3.9772, 5.0574, 7.5316, 1.4201, 8.3035, 5.0212, -0.1031, 7.553, 5.5023, 3.1427, 6.7674, 4.4409, 6.457, 4.525, 0.728, 6.5422, -0.6234, -0.4726, 2.7486, 0.6985, -3.0804, 1.9669, 3.2365], [1.5065, 6.1271, 5.8296, 2.8405, 9.5649, 3.6834, 5.1214, 7.546, 0.9758, 8.3335, 5.1952, -0.1395, 7.4348, 5.6893, 3.2942, 7.0356, 4.1665, 5.9695, 4.3898, 0.8931, 6.3988, -0.8957, -0.7522, 2.8924, 0.6498, -3.4358, 1.8654, 2.9735]],  # noqa: E231
            device=torch_device,
        )
        # fmt: on
        self.assertTrue(
            torch.allclose(prediction_logits[0, 4:8, 128:156], expected_prediction_logits_slice, atol=1e-4)
        )

    def test_seq_to_seq_generation(self):

        model = BigBirdPegasusForConditionalGeneration.from_pretrained(
            MODEL_ID, attention_type="block_sparse", block_size=16, num_random_blocks=3
        )
        model.to(torch_device)
        tokenizer = PegasusTokenizer.from_pretrained(MODEL_ID)

        # fmt: off
        batch_input = [
            'for about 20 years the problem of properties of short - term changes of solar activity has been considered extensively .\nmany investigators studied the short - term periodicities of the various indices of solar activity .\nseveral periodicities were detected , but the periodicities about 155 days and from the interval of @xmath3 $ ] days ( @xmath4 $ ] years ) are mentioned most often .\nfirst of them was discovered by @xcite in the occurence rate of gamma - ray flares detected by the gamma - ray spectrometer aboard the _ solar maximum mission ( smm ) .\nthis periodicity was confirmed for other solar flares data and for the same time period @xcite .\nit was also found in proton flares during solar cycles 19 and 20 @xcite , but it was not found in the solar flares data during solar cycles 22 @xcite .\n_    several autors confirmed above results for the daily sunspot area data . @xcite studied the sunspot data from 18741984 .\nshe found the 155-day periodicity in data records from 31 years .\nthis periodicity is always characteristic for one of the solar hemispheres ( the southern hemisphere for cycles 1215 and the northern hemisphere for cycles 1621 ) .\nmoreover , it is only present during epochs of maximum activity ( in episodes of 13 years ) .\nsimilarinvestigationswerecarriedoutby + @xcite .\nthey applied the same power spectrum method as lean , but the daily sunspot area data ( cycles 1221 ) were divided into 10 shorter time series .\nthe periodicities were searched for the frequency interval 57115 nhz ( 100200 days ) and for each of 10 time series .\nthe authors showed that the periodicity between 150160 days is statistically significant during all cycles from 16 to 21 .\nthe considered peaks were remained unaltered after removing the 11-year cycle and applying the power spectrum analysis .\n@xcite used the wavelet technique for the daily sunspot areas between 1874 and 1993 .\nthey determined the epochs of appearance of this periodicity and concluded that it presents around the maximum activity period in cycles 16 to 21 .\nmoreover , the power of this periodicity started growing at cycle 19 , decreased in cycles 20 and 21 and disappered after cycle 21 .\nsimilaranalyseswerepresentedby + @xcite , but for sunspot number , solar wind plasma , interplanetary magnetic field and geomagnetic activity index @xmath5 .\nduring 1964 - 2000 the sunspot number wavelet power of periods less than one year shows a cyclic evolution with the phase of the solar cycle.the 154-day period is prominent and its strenth is stronger around the 1982 - 1984 interval in almost all solar wind parameters .\nthe existence of the 156-day periodicity in sunspot data were confirmed by @xcite .\nthey considered the possible relation between the 475-day ( 1.3-year ) and 156-day periodicities .\nthe 475-day ( 1.3-year ) periodicity was also detected in variations of the interplanetary magnetic field , geomagnetic activity helioseismic data and in the solar wind speed @xcite .\n@xcite concluded that the region of larger wavelet power shifts from 475-day ( 1.3-year ) period to 620-day ( 1.7-year ) period and then back to 475-day ( 1.3-year ) .\nthe periodicities from the interval @xmath6 $ ] days ( @xmath4 $ ] years ) have been considered from 1968 .\n@xcite mentioned a 16.3-month ( 490-day ) periodicity in the sunspot numbers and in the geomagnetic data .\n@xcite analysed the occurrence rate of major flares during solar cycles 19 .\nthey found a 18-month ( 540-day ) periodicity in flare rate of the norhern hemisphere .\n@xcite confirmed this result for the @xmath7 flare data for solar cycles 20 and 21 and found a peak in the power spectra near 510540 days .\n@xcite found a 17-month ( 510-day ) periodicity of sunspot groups and their areas from 1969 to 1986 .\nthese authors concluded that the length of this period is variable and the reason of this periodicity is still not understood .\n@xcite and + @xcite obtained statistically significant peaks of power at around 158 days for daily sunspot data from 1923 - 1933 ( cycle 16 ) . in this paper the problem of the existence of this periodicity for sunspot data from cycle 16 is considered .\nthe daily sunspot areas , the mean sunspot areas per carrington rotation , the monthly sunspot numbers and their fluctuations , which are obtained after removing the 11-year cycle are analysed . in section 2 the properties of the power spectrum methods are described . in section 3 a new approach to the problem of aliases in the power spectrum analysis\nis presented . in section 4 numerical results of the new method of the diagnosis of an echo - effect for sunspot area data are discussed . in section 5 the problem of the existence of the periodicity of about 155 days during the maximum activity period for sunspot data from the whole solar disk and from each solar hemisphere separately is considered .'  # noqa: E231
        ]
        # fmt: on

        dct = tokenizer(
            batch_input,
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        hypotheses_batch = model.generate(input_ids=dct["input_ids"].to(torch_device))
        generated = tokenizer.batch_decode(hypotheses_batch.tolist())

        # fmt: off
        expected = [
            '<s> the properties of the short - term periodicities of various indices of solar activity are studied.<n> one of them is the periodicity between 150 and 160 days.<n> this periodicity was discovered by @xcite in the occurence rate of gamma - ray flares detected by the gamma - ray spectrometer during epochs of maximum activity ( in episodes of 13 years ).<n> it was also found in proton flares during solar cycles 19 and 20 @xcite, and in data records from 31 years.<n> it is only present during epochs of maximum activity ( in episodes of 13 years ).<n> another property of short - term periodicities is the 18-month ( 540-day ) periodicity in flare rate of the norhern hemisphere.<n> it was discovered by @xcite in the occurence rate of gamma - ray flares detected by the gamma - ray spectrometer during solar cycles 19 and 20.<n> it was also found in proton flares during solar cycles 19 and 20 @xcite, and in data records from 31 years. in this paper<n> the problem of the existence of this periodicity for sunspot data from 1923 - 1933 is considered. the daily sunspot areas, the mean sunspot',  # noqa: E231
        ]
        # fmt: on
        self.assertTrue(generated == expected)


class BigBirdPegasusStandaloneDecoderModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=7,
        d_model=32,
        decoder_seq_length=7,
        is_training=True,
        is_decoder=True,
        use_attention_mask=True,
        use_cache=False,
        use_labels=True,
        decoder_start_token_id=2,
        decoder_ffn_dim=32,
        decoder_layers=4,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        max_position_embeddings=30,
        is_encoder_decoder=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        scope=None,
        attention_type="original_full",
        use_bias=True,
        block_size=16,
        num_random_blocks=3,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.decoder_seq_length = decoder_seq_length
        # For common tests
        self.seq_length = self.decoder_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_size = d_model
        self.num_hidden_layers = decoder_layers
        self.decoder_layers = decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.num_attention_heads = decoder_attention_heads
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.is_encoder_decoder = is_encoder_decoder

        self.scope = None
        self.decoder_key_length = decoder_seq_length
        self.base_model_out_len = 2
        self.decoder_attention_idx = 1

        self.attention_type = attention_type
        self.use_bias = use_bias
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.decoder_seq_length], vocab_size=2)

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

        config = BigBirdPegasusConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            decoder_layers=self.decoder_layers,
            decoder_ffn_dim=self.decoder_ffn_dim,
            encoder_attention_heads=self.encoder_attention_heads,
            decoder_attention_heads=self.decoder_attention_heads,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
            max_position_embeddings=self.max_position_embeddings,
            is_encoder_decoder=self.is_encoder_decoder,
            attention_type=self.attention_type,
            use_bias=self.use_bias,
            block_size=self.block_size,
            num_random_blocks=self.num_random_blocks,
        )

        return (
            config,
            input_ids,
            attention_mask,
            lm_labels,
        )

    def create_and_check_decoder_model_past(
        self,
        config,
        input_ids,
        attention_mask,
        lm_labels,
    ):
        config.use_cache = True
        model = BigBirdPegasusDecoder(config=config).to(torch_device).eval()
        # first forward pass
        outputs = model(input_ids, use_cache=True)
        outputs_use_cache_conf = model(input_ids)
        outputs_no_past = model(input_ids, use_cache=False)

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        past_key_values = outputs["past_key_values"]

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        output_from_no_past = model(next_input_ids)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past_key_values)["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, next_input_ids.shape[-1] - 1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        assert torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3)

    def create_and_check_decoder_model_attention_mask_past(
        self,
        config,
        input_ids,
        attention_mask,
        lm_labels,
    ):
        model = BigBirdPegasusDecoder(config=config).to(torch_device).eval()

        # create attention mask
        attn_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        half_seq_length = input_ids.shape[-1] // 2
        attn_mask[:, half_seq_length:] = 0

        # first forward pass
        past_key_values = model(input_ids, attention_mask=attn_mask, use_cache=True)["past_key_values"]

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # change a random masked slice from input_ids
        random_seq_idx_to_change = ids_tensor((1,), half_seq_length).item() + 1
        random_other_next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size).squeeze(-1)
        input_ids[:, -random_seq_idx_to_change] = random_other_next_tokens

        # append to next input_ids and attn_mask
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        attn_mask = torch.cat(
            [attn_mask, torch.ones((attn_mask.shape[0], 1), dtype=torch.long, device=torch_device)],
            dim=1,
        )

        # get two different outputs
        output_from_no_past = model(next_input_ids)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past_key_values)["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, next_input_ids.shape[-1] - 1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        assert torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-2)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
            lm_labels,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class BigBirdPegasusStandaloneDecoderModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (BigBirdPegasusDecoder, BigBirdPegasusForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (BigBirdPegasusForCausalLM,) if is_torch_available() else ()
    test_pruning = False
    is_encoder_decoder = False

    def setUp(
        self,
    ):
        self.model_tester = BigBirdPegasusStandaloneDecoderModelTester(self, is_training=False)
        self.config_tester = ConfigTester(self, config_class=BigBirdPegasusConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_decoder_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past(*config_and_inputs)

    def test_decoder_model_attn_mask_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_attention_mask_past(*config_and_inputs)

    def test_retain_grad_hidden_states_attentions(self):
        # decoder cannot keep gradients
        return
