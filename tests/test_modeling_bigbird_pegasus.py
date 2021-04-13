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
from transformers.file_utils import cached_property
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
        BigBirdPegasusTokenizer,
    )
    from transformers.models.bigbird_pegasus.modeling_bigbird_pegasus import (
        BigBirdPegasusDecoder,
        BigBirdPegasusEncoder,
    )


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


# bigbird fast tests will have problem of attention type switching


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
    test_missing_keys = False

    # torchscript should be possible, but takes prohibitively long to test.
    # Also torchscript is not an important feature to have in the beginning.
    # test_torchscript = False

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


def assert_tensors_close(a, b, atol=1e-12, prefix=""):
    """If tensors have different shapes, different values or a and b are not both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if torch.allclose(a, b, atol=atol):
            return True
        raise
    except Exception:
        pct_different = (torch.gt((a - b).abs(), atol)).float().mean().item()
        if a.numel() > 100:
            msg = f"tensor values are {pct_different:.1%} percent different."
        else:
            msg = f"{a} != {b}"
        if prefix:
            msg = prefix + ": " + msg
        raise AssertionError(msg)


TOLERANCE = 1e-4
MODEL_ID = "vasudevgupta/bigbird-pegasus-large-pubmed"


@require_torch
@require_sentencepiece
@require_tokenizers
@slow
class BigBirdPegasusModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_tokenizer(self):
        return BigBirdPegasusTokenizer.from_pretrained(MODEL_ID)

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
            [[2.785, 2.134, 6.9243, -1.6229, 1.3514, 1.6717, 3.7153, 4.046, 7.7103, 2.0608, 2.0849, 0.26425, 1.3922, 3.2817, 6.7929, 5.1377, -0.88223, 8.061, 3.504, -0.67016, 3.1645, 5.9968, 7.1979, 4.675, -1.7048, 3.6679, 6.0426, 2.2554], [2.8886, 2.1618, 7.3298, -1.6505, 1.4547, 1.5609, 4.0607, 4.0344, 7.9283, 2.0122, 2.2935, 0.64506, 1.3994, 2.8479, 6.8885, 5.2957, -1.0617, 7.596, 3.5451, -0.28749, 3.0484, 5.8572, 7.0605, 4.7069, -1.7569, 3.5513, 5.7782, 2.2574], [3.0022, 2.1422, 7.5003, -1.8414, 1.5542, 1.528, 4.2919, 4.2379, 8.0021, 2.0778, 2.3393, 0.77081, 1.4541, 2.8147, 6.9367, 5.5263, -0.9241, 7.3092, 3.5381, 0.0074557, 3.0441, 5.8372, 7.0485, 4.8198, -1.715, 3.52, 5.7461, 2.2033], [3.0866, 2.1803, 7.5962, -1.9698, 1.6234, 1.4729, 4.237, 4.2311, 8.0083, 2.1442, 2.2766, 0.79028, 1.3316, 2.8024, 6.8416, 5.7381, -0.95507, 7.1824, 3.6925, -0.022659, 2.9421, 6.0023, 7.0864, 4.9817, -1.6959, 3.4832, 5.6923, 2.2286]],  # noqa: E231
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
            [[3.7736, 0.6459, 5.9393, -2.055, 1.3957, 1.6994, 1.7002, 4.3194, 6.727, 0.8877, 2.7457, 0.3128, -0.3091, 3.7636, 6.4191, 3.2155, -0.9953, 7.4407, 3.8938, 0.407, 3.7436, 3.7248, 5.6073, 3.8378, -1.94, 5.2315, 4.6829, 2.0397], [3.8075, 0.5993, 5.9881, -1.9268, 1.7395, 1.9801, 1.4785, 4.404, 6.9427, 0.6825, 2.8742, 0.7088, -0.6241, 3.3309, 6.5836, 3.2848, -1.1375, 7.2144, 4.1101, 0.8657, 3.952, 3.5079, 5.4696, 3.9301, -2.2243, 5.2562, 4.651, 1.9688], [3.9393, 0.5811, 6.1118, -1.9829, 1.9584, 2.0622, 1.6118, 4.5815, 7.1832, 0.6703, 2.9474, 0.8766, -0.7241, 3.309, 6.772, 3.4544, -1.0948, 7.0197, 4.2286, 1.1543, 4.0334, 3.4939, 5.5613, 4.1545, -2.2169, 5.4238, 4.7881, 1.9614], [4.0004, 0.5768, 6.1671, -2.1092, 2.0556, 2.0222, 1.5487, 4.5812, 7.2975, 0.7099, 3.0134, 0.9069, -0.8406, 3.2854, 6.756, 3.5334, -1.2231, 6.8766, 4.3854, 1.1062, 3.9816, 3.6632, 5.6403, 4.3885, -2.2414, 5.4234, 4.7948, 1.9947]],  # noqa: E231
            device=torch_device,
        )
        # fmt: on
        self.assertTrue(
            torch.allclose(prediction_logits[0, 4:8, 128:156], expected_prediction_logits_slice, atol=1e-4)
        )

    def test_seq_to_seq_generation(self):

        hf = BigBirdPegasusForConditionalGeneration.from_pretrained(MODEL_ID).to(torch_device)
        tok = BigBirdPegasusTokenizer.from_pretrained(MODEL_ID)

        batch_input = [
            # string 1,
            # string 2,
            # string 3,
            # string 4,
        ]

        # The below article tests that we don't add any hypotheses outside of the top n_beams
        dct = tok.batch_encode_plus(
            batch_input,
            max_length=512,
            padding="max_length",
            truncation_strategy="only_first",
            truncation=True,
            return_tensors="pt",
        )

        hypotheses_batch = hf.generate(
            input_ids=dct["input_ids"].to(torch_device),
            attention_mask=dct["attention_mask"].to(torch_device),
            num_beams=2,
        )

        EXPECTED = [
            # here expected 1,
            # here expected 2,
            # here expected 3,
            # here expected 4,
        ]

        generated = tok.batch_decode(
            hypotheses_batch.tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True
        )
        assert generated == EXPECTED


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
