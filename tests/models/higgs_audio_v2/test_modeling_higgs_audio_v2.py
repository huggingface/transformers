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
from parameterized import parameterized

from transformers import (
    AutoProcessor,
    HiggsAudioV2Config,
    HiggsAudioV2ForConditionalGeneration,
    HiggsAudioV2Model,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin, has_similar_generate_outputs
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, global_rng, ids_tensor, random_attention_mask


if is_torch_available():
    import torch


class HiggsAudioV2ModelTester:
    base_model_class = HiggsAudioV2Model
    causal_lm_class = HiggsAudioV2ForConditionalGeneration

    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=14,
        audio_seq_length=10,
        is_training=True,
        main_input_name_for_generation="audio_input_ids",
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=500000.0,
        rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=16,
        num_codebooks=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.audio_seq_length = audio_seq_length
        self.is_training = is_training
        self.main_input_name_for_generation = main_input_name_for_generation

        self.config_kwargs = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "hidden_act": hidden_act,
            "max_position_embeddings": max_position_embeddings,
            "initializer_range": initializer_range,
            "rms_norm_eps": rms_norm_eps,
            "use_cache": use_cache,
            "pad_token_id": pad_token_id,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "pretraining_tp": pretraining_tp,
            "tie_word_embeddings": tie_word_embeddings,
            "rope_theta": rope_theta,
            "rope_scaling": rope_scaling,
            "attention_bias": attention_bias,
            "attention_dropout": attention_dropout,
            "mlp_bias": mlp_bias,
            "head_dim": head_dim,
            "num_codebooks": num_codebooks,
            "codebook_size": hidden_size // num_codebooks,
            "audio_token_id": vocab_size - 1,
            "audio_bos_token_id": vocab_size - 2,
            "audio_delay_token_id": vocab_size - 3,
            "audio_stream_bos_id": hidden_size // num_codebooks - 1,
            "audio_stream_eos_id": hidden_size // num_codebooks - 2,
        }

        # also set them as attributes
        for key, value in self.config_kwargs.items():
            setattr(self, key, value)

    def prepare_config_and_inputs(self):
        # let's make sure we don't sample audio_token_id, audio_bos_token_id, audio_delay_token_id
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 3)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        # for audio token positions, we use a second random attention mask to set idxs
        audio_token_mask = random_attention_mask([self.batch_size, self.audio_seq_length])

        # let's ensure at least one batch_idx has audio_seq_length audio tokens
        batch_idx = global_rng.randint(0, self.batch_size - 1)
        audio_token_mask[batch_idx, :] = 1

        # audio_token_mask should have the shape of input_ids
        audio_token_mask = torch.cat(
            [audio_token_mask.new_zeros([self.batch_size, self.seq_length - self.audio_seq_length]), audio_token_mask],
            dim=1,
        )

        input_ids[audio_token_mask.bool()] = self.audio_token_id

        audio_seq_lengths = audio_token_mask.sum(-1)
        max_audio_seq_length = audio_seq_lengths.max()
        audio_input_ids = ids_tensor([self.batch_size, max_audio_seq_length, self.num_codebooks], self.codebook_size)
        audio_input_ids_mask = (
            torch.arange(max_audio_seq_length, device=torch_device)[None, :] < audio_seq_lengths[:, None]
        )

        # TODO: @eustlb, should il really be bool?
        audio_input_ids_mask = audio_input_ids_mask.bool()
        config = self.get_config()

        return config, input_ids, attention_mask, audio_input_ids, audio_input_ids_mask

    def get_config(self):
        return HiggsAudioV2Config(**self.config_kwargs)

    def create_and_check_model(self, config, input_ids):
        model = config
        model.to(torch_device)
        model.eval()
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask, audio_input_ids, audio_input_ids_mask = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_input_ids": audio_input_ids,
            "audio_input_ids_mask": audio_input_ids_mask,
        }
        return config, inputs_dict


@require_torch
class HiggsAudioV2ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (HiggsAudioV2ForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {"text-to-speech": HiggsAudioV2ForConditionalGeneration} if is_torch_available() else {}
    test_pruning = False

    def setUp(self):
        self.model_tester = HiggsAudioV2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=HiggsAudioV2Config)

    def test_config(self):
        self.config_tester.run_common_tests()

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    @pytest.mark.skip(reason="HiggsAudioV2 does not support assisted decoding.")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @pytest.mark.generate
    @pytest.mark.skip(reason="HiggsAudioV2 does not support assisted decoding.")
    def test_assisted_decoding_sample(self):
        pass

    @pytest.mark.generate
    @pytest.mark.skip(reason="HiggsAudioV2 does not support beam search.")
    def test_beam_sample_generate(self):
        pass

    @pytest.mark.generate
    @pytest.mark.skip(reason="HiggsAudioV2 does not support beam search.")
    def test_beam_search_generate(self):
        pass

    @pytest.mark.generate
    @pytest.mark.skip(reason="HiggsAudioV2 does not support beam search.")
    def test_beam_search_generate_dict_output(self):
        pass

    @pytest.mark.generate
    @pytest.mark.skip(reason="HiggsAudioV2 does not support beam search.")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @pytest.mark.generate
    @pytest.mark.skip(reason="HiggsAudioV2 does not support beam search.")
    def test_beam_sample_generate_dict_output(self):
        pass

    @pytest.mark.generate
    @pytest.mark.skip(reason="HiggsAudioV2 does not support beam search.")
    def test_generate_from_inputs_embeds_1_beam_search(self, _, num_beams):
        pass

    @pytest.mark.generate
    @pytest.mark.skip(reason="HiggsAudioV2 does not support beam search.")
    def test_model_parallel_beam_search(self):
        pass

    @pytest.mark.generate
    @pytest.mark.skip(reason="HiggsAudioV2 does not support prompt lookup decoding.")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @pytest.mark.generate
    @pytest.mark.skip(reason="HiggsAudioV2 does not support prompt lookup decoding.")
    def test_prompt_lookup_decoding_stops_at_eos(self):
        pass

    @pytest.mark.skip(reason="HiggsAudioV2 has custom embedding approach (text and audio embeddings).")
    def test_model_get_set_embeddings(self):
        pass

    @pytest.mark.skip(reason="HiggsAudioV2 has custom embedding approach (text and audio embeddings).")
    def test_tie_model_weights(self):
        pass

    @pytest.mark.skip(reason="HiggsAudioV2 has custom embedding approach (text and audio embeddings).")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @pytest.mark.skip(reason="HiggsAudioV2 has custom embedding approach (text and audio embeddings).")
    def test_resize_tokens_embeddings(self):
        pass

    @pytest.mark.skip(reason="HiggsAudioV2 has special embeddings that can never be tied")
    def test_tied_weights_keys(self):
        pass

    def _check_scores(self, batch_size, scores, generated_length, config):
        expected_shape = (batch_size, config.num_codebooks, config.codebook_size)
        self.assertIsInstance(scores, tuple)
        self.assertEqual(len(scores), generated_length)
        self.assertListEqual([iter_scores.shape for iter_scores in scores], [expected_shape] * len(scores))

    def _check_logits(self, batch_size, logits, config):
        self.assertIsInstance(logits, tuple)
        self.assertListEqual([iter_logits.shape[0] for iter_logits in logits], [batch_size] * len(logits))
        # Check that the shape matches expected codebook dimensions
        expected_last_dim = config.num_codebooks * config.codebook_size
        self.assertListEqual([iter_logits.shape[-1] for iter_logits in logits], [expected_last_dim] * len(logits))

    @pytest.mark.generate
    def test_greedy_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).to(torch_device).eval()
            output_generate = self._greedy_generate(model=model, inputs_dict=inputs_dict)
            self.assertTrue(output_generate.shape[1] == self.max_new_tokens + inputs_dict["audio_input_ids"].shape[1])

    @pytest.mark.generate
    def test_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).to(torch_device).eval()
            output_generate = self._sample_generate(model=model, inputs_dict=inputs_dict, num_return_sequences=1)
            self.assertTrue(output_generate.shape[1] == self.max_new_tokens + inputs_dict["audio_input_ids"].shape[1])

    def test_forward_with_logits_to_keep(self):
        for model_class in self.all_generative_model_classes:
            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
            batch_size, sequence_length = inputs["input_ids"].shape[:2]
            model = model_class(config).to(device=torch_device).eval()
            # some models have labels but `logits_to_keep` should not be used in train mode
            _ = inputs.pop("labels", None)

            # logits_to_keep=0 is a special case meaning "keep all logits"
            all_logits = model(**inputs, logits_to_keep=0).logits
            last_token_logits = model(**inputs, logits_to_keep=1).logits

            # Assert all shapes are correct
            self.assertEqual(
                tuple(all_logits.shape), (batch_size, sequence_length, config.num_codebooks * config.codebook_size)
            )
            self.assertEqual(
                tuple(last_token_logits.shape), (batch_size, 1, config.num_codebooks * config.codebook_size)
            )

            # Assert the last tokens are actually the same (except for the natural fluctuation due to order of FP ops)
            torch.testing.assert_close(all_logits[:, -1:, :], last_token_logits, rtol=1e-5, atol=1e-5)

    @pytest.mark.generate
    def test_generate_continue_from_past_key_values(self):
        # Tests that we can continue generating from past key values, returned from a previous `generate` call
        for model_class in self.all_generative_model_classes:
            if any(model_name in model_class.__name__.lower() for model_name in ["imagegpt", "mllama"]):
                self.skipTest(reason="Won't fix: old model with unique inputs/caches/other")
            if any(model_name in model_class.__name__.lower() for model_name in ["umt5"]):
                self.skipTest(reason="TODO: needs modeling or test input preparation fixes for compatibility")

            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            if not hasattr(config.get_text_config(), "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")

            # Let's make it always:
            # 1. use cache (for obvious reasons)
            # 2. generate to max length (which can be achieved by setting the eos token to an invalid value), which
            #    would make the test flaky (e.g. EOS is generated on iteration 1 on both generations, but the
            #    continuation would force it to generate beyond an EOS token)
            # 3. ignore `token_type_ids` for simplicity
            # 4. ignore `forced_eos_token_id`, which requires further manipulation of the continuation inputs and is
            #    active by default on some models
            # 5. ignore `encoder_no_repeat_ngram_size`, which is set by default in some encoder-decoder models. When
            #    we use their decoder as a stand-alone model, `encoder_no_repeat_ngram_size` actually prevents
            #    repetition exclusively from the prompt. This test relies on comparing one call vs 2 calls
            #    with cache, what is considered a prompt is different in the two cases.

            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            model = model_class(config).to(torch_device)
            model.eval()

            # If "past_key_values" is not returned, skip the test (e.g. RWKV uses a different cache name and format)
            outputs = model(**inputs)
            if "past_key_values" not in outputs:
                self.skipTest(reason="This model doesn't return `past_key_values`")

            generate_kwargs = {
                "pad_token_id": -1,
                "eos_token_id": -1,
                "forced_eos_token_id": None,
                "encoder_no_repeat_ngram_size": 0,
                "use_cache": True,
                "do_sample": False,
                "return_dict_in_generate": True,
                "output_scores": True,
            }

            # Traditional way of generating text, with `return_dict_in_generate` to return the past key values
            outputs = model.generate(**inputs, **generate_kwargs, max_new_tokens=4)

            # Let's generate again, but passing the past key values in between (3 + 1 = 4 tokens). Note that the
            # inputs may need to be tweaked across `generate` calls (like the attention mask).
            outputs_cached = model.generate(**inputs, **generate_kwargs, max_new_tokens=3)

            # Continue from the tokens generated above, preparing the inputs accordingly
            inputs["past_key_values"] = outputs_cached.past_key_values
            new_attention_len = outputs_cached.sequences.shape[1]
            new_audio_input_ids_len = outputs_cached.audio_sequences.shape[1]
            inputs["input_ids"] = outputs_cached.sequences
            inputs["audio_input_ids"] = outputs_cached.audio_sequences
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.nn.functional.pad(
                    inputs["attention_mask"],
                    (0, new_attention_len - inputs["attention_mask"].shape[1]),
                    mode="constant",
                    value=1,
                )
            if "audio_input_ids_mask" in inputs:
                num_gen_audio_inputs_ids = new_audio_input_ids_len - inputs["audio_input_ids_mask"].shape[1]
                inputs["audio_input_ids_mask"] = torch.nn.functional.pad(
                    inputs["audio_input_ids_mask"],
                    (0, num_gen_audio_inputs_ids),
                    mode="constant",
                    value=1,
                )
                mask = inputs["input_ids"][:, -num_gen_audio_inputs_ids:] == config.eos_token_id
                inputs["audio_input_ids_mask"][:, -num_gen_audio_inputs_ids:][mask] = False

            first_caches_scores = outputs_cached.scores
            outputs_cached = model.generate(**inputs, **generate_kwargs, max_new_tokens=1)
            full_cached_scores = first_caches_scores + outputs_cached.scores
            outputs_cached.scores = full_cached_scores

            # The two sets of generated text and past kv should be equal to each other
            self.assertTrue(has_similar_generate_outputs(outputs, outputs_cached))
            self._check_caches_are_equal(outputs.past_key_values, outputs_cached.past_key_values)


class HiggsAudioV2ForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint_name = "eustlb/higgs-v2"
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_name, device_map=torch_device)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_torch_accelerator
    def test_single_speaker_smart_voice(self):
        model = HiggsAudioV2ForConditionalGeneration.from_pretrained(self.checkpoint_name, device_map=torch_device)

        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Generate audio following instruction."
                    }
                ],
            },
            {
                "role": "scene",
                "content": [
                    {
                        "type": "text",
                        "text": "Audio is recorded from a quiet room."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "The sun rises in the east and sets in the west."
                    }
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            sampling_rate=24000,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=1000, do_sample=False)

        # fmt: off
        EXPECTED_OUTPUT_TOKENS = torch.tensor(
            [
                [
                    [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [244, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [498, 537, 1024, 1024, 1024, 1024, 1024, 1024],
                    [430, 851, 977, 1024, 1024, 1024, 1024, 1024],
                    [950, 986, 39, 130, 1024, 1024, 1024, 1024],
                    [196, 212, 784, 392, 283, 1024, 1024, 1024],
                    [196, 367, 242, 1022, 686, 325, 1024, 1024],
                    [776, 562, 971, 196, 932, 645, 53, 1024],
                    [480, 214, 971, 75, 709, 157, 326, 669],
                    [650, 299, 97, 339, 406, 434, 245, 655],
                    [1000, 89, 570, 400, 958, 75, 417, 794],
                    [651, 391, 129, 106, 9, 318, 417, 298],
                    [945, 535, 770, 918, 935, 117, 330, 409],
                    [113, 299, 328, 882, 594, 699, 176, 790],
                    [649, 344, 178, 563, 996, 784, 368, 146],
                    [287, 454, 555, 164, 46, 464, 86, 223],
                    [134, 94, 684, 966, 244, 464, 920, 898],
                    [63, 862, 683, 776, 914, 568, 1007, 799],
                    [165, 18, 107, 881, 176, 1007, 930, 557],
                    [822, 89, 18, 158, 1016, 821, 256, 430],
                    [27, 89, 564, 750, 127, 171, 728, 458],
                    [27, 94, 232, 781, 624, 0, 433, 550],
                    [898, 137, 1002, 738, 751, 525, 852, 454],
                    [541, 853, 1002, 556, 339, 174, 540, 566],
                    [419, 449, 69, 576, 886, 208, 803, 12],
                    [196, 515, 976, 29, 744, 333, 828, 11],
                    [419, 393, 784, 637, 577, 1000, 700, 667],
                    [473, 958, 843, 816, 686, 121, 152, 203],
                    [708, 462, 21, 31, 304, 129, 730, 667],
                    [343, 759, 832, 995, 581, 228, 116, 766],
                    [405, 225, 784, 977, 2, 861, 942, 834],
                    [778, 532, 207, 228, 359, 189, 636, 992],
                    [40, 265, 290, 296, 674, 755, 979, 195],
                    [442, 18, 666, 495, 96, 778, 763, 179],
                    [740, 132, 720, 465, 196, 399, 829, 969],
                    [740, 284, 460, 941, 539, 7, 203, 369],
                    [265, 662, 467, 576, 469, 65, 971, 761],
                    [265, 22, 5, 181, 118, 275, 283, 256],
                    [419, 420, 46, 507, 988, 472, 577, 924],
                    [300, 539, 967, 813, 239, 134, 573, 720],
                    [300, 248, 829, 739, 903, 176, 379, 668],
                    [778, 746, 691, 531, 1016, 166, 205, 322],
                    [778, 153, 276, 496, 445, 361, 531, 800],
                    [740, 153, 398, 914, 789, 415, 932, 157],
                    [675, 532, 398, 226, 95, 455, 950, 853],
                    [196, 196, 784, 214, 556, 799, 644, 57],
                    [697, 825, 308, 432, 155, 5, 924, 474],
                    [645, 60, 535, 269, 1020, 815, 801, 671],
                    [771, 832, 565, 409, 67, 1018, 1021, 375],
                    [238, 198, 56, 925, 269, 950, 255, 467],
                    [787, 423, 912, 345, 215, 747, 186, 458],
                    [338, 216, 720, 469, 428, 890, 441, 88],
                    [733, 954, 490, 549, 630, 588, 201, 962],
                    [196, 406, 415, 862, 492, 729, 621, 770],
                    [317, 580, 623, 446, 2, 194, 776, 891],
                    [880, 853, 184, 673, 659, 397, 73, 393],
                    [921, 789, 279, 130, 1006, 25, 804, 607],
                    [578, 461, 570, 19, 96, 890, 22, 949],
                    [733, 728, 575, 456, 1008, 612, 765, 229],
                    [776, 954, 576, 510, 827, 525, 981, 158],
                    [950, 828, 960, 805, 664, 898, 401, 312],
                    [787, 543, 739, 485, 445, 895, 310, 490],
                    [372, 464, 318, 830, 743, 63, 909, 1016],
                    [238, 36, 487, 465, 288, 812, 469, 321],
                    [4, 517, 487, 649, 539, 1021, 602, 803],
                    [261, 561, 46, 912, 325, 20, 742, 932],
                    [261, 90, 575, 694, 352, 54, 81, 176],
                    [882, 570, 912, 50, 575, 5, 315, 261],
                    [565, 651, 688, 617, 257, 983, 826, 255],
                    [1000, 36, 31, 221, 703, 744, 663, 683],
                    [0, 584, 459, 579, 1010, 805, 807, 79],
                    [717, 111, 798, 752, 914, 799, 470, 206],
                    [715, 18, 220, 787, 759, 904, 529, 732],
                    [338, 517, 339, 19, 883, 282, 257, 355],
                    [196, 617, 52, 667, 659, 276, 708, 395],
                    [783, 825, 649, 171, 392, 947, 861, 264],
                    [645, 641, 0, 804, 803, 287, 386, 493],
                    [553, 27, 903, 925, 269, 1014, 559, 803],
                    [891, 954, 73, 654, 758, 189, 558, 408],
                    [800, 569, 508, 128, 324, 895, 652, 235],
                    [989, 202, 270, 460, 789, 764, 391, 152],
                    [989, 873, 468, 161, 709, 860, 595, 962],
                    [800, 873, 553, 351, 968, 162, 306, 1021],
                    [989, 354, 553, 467, 709, 488, 296, 964],
                    [989, 503, 468, 343, 433, 552, 986, 620],
                    [989, 503, 468, 467, 433, 653, 595, 229],
                    [989, 503, 468, 351, 433, 653, 602, 858],
                    [989, 873, 468, 351, 433, 552, 926, 858],
                    [1025, 528, 427, 351, 799, 552, 926, 858],
                    [1025, 1025, 468, 343, 421, 764, 926, 669],
                    [1025, 1025, 1025, 584, 433, 764, 926, 419],
                    [1025, 1025, 1025, 1025, 433, 552, 926, 390],
                    [1025, 1025, 1025, 1025, 1025, 342, 44, 858],
                    [1025, 1025, 1025, 1025, 1025, 1025, 926, 419],
                    [1025, 1025, 1025, 1025, 1025, 1025, 1025, 419],
                    [1025, 1025, 1025, 1025, 1025, 1025, 1025, 1025]
                ]
            ]
        )
        # fmt: on

        torch.testing.assert_close(outputs.cpu(), EXPECTED_OUTPUT_TOKENS)

    @slow
    @require_torch_accelerator
    def test_multi_speaker_smart_voice(self):
        system_message = """You are an AI assistant designed to convert text into speech.
        If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
        If no speaker tag is present, select a suitable voice on your own."""

        user_message = """[SPEAKER0] I can't believe you did that without even asking me first!
        [SPEAKER1] Oh, come on! It wasn't a big deal, and I knew you would overreact like this.
        [SPEAKER0] Overreact? You made a decision that affects both of us without even considering my opinion!
        [SPEAKER1] Because I didn't have time to sit around waiting for you to make up your mind! Someone had to act."""

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {
                "role": "scene",
                "content": [
                    {"type": "text", "text": "Audio is recorded from a quiet room."},
                    {"type": "text", "text": "SPEAKER0: feminine"},
                    {"type": "text", "text": "SPEAKER1: masculine"},
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": user_message}]},
        ]

        model = HiggsAudioV2ForConditionalGeneration.from_pretrained(self.checkpoint_name, device_map=torch_device)

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            sampling_rate=24000,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=50)

        # fmt: off
        EXPECTED_OUTPUT_TOKENS = torch.tensor(
            [
                [
                    [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [127, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [296, 713, 1024, 1024, 1024, 1024, 1024, 1024],
                    [252, 477, 872, 1024, 1024, 1024, 1024, 1024],
                    [569, 477, 142, 954, 1024, 1024, 1024, 1024],
                    [252, 644, 57, 623, 283, 1024, 1024, 1024],
                    [743, 746, 869, 313, 786, 809, 1024, 1024],
                    [642, 644, 142, 805, 185, 608, 442, 1024],
                    [412, 334, 297, 890, 982, 100, 248, 727],
                    [492, 518, 108, 169, 108, 169, 367, 201],
                    [171, 239, 834, 1014, 53, 429, 959, 96],
                    [386, 212, 598, 388, 903, 573, 904, 224],
                    [187, 484, 815, 173, 919, 586, 165, 276],
                    [725, 257, 137, 75, 772, 245, 304, 850],
                    [725, 224, 975, 756, 765, 12, 867, 68],
                    [173, 840, 328, 95, 1022, 481, 163, 282],
                    [187, 644, 316, 899, 578, 80, 201, 63],
                    [356, 909, 869, 277, 133, 328, 446, 966],
                    [569, 904, 149, 477, 985, 735, 815, 825],
                    [594, 706, 208, 362, 178, 190, 772, 966],
                    [840, 644, 475, 655, 568, 120, 308, 388],
                    [353, 127, 142, 633, 937, 107, 836, 895],
                    [569, 839, 517, 579, 399, 201, 1015, 236],
                    [693, 569, 834, 337, 623, 183, 682, 1013],
                    [263, 644, 834, 673, 656, 777, 97, 349],
                    [440, 477, 917, 619, 223, 548, 1022, 308],
                    [272, 334, 142, 298, 351, 56, 1015, 893],
                    [31, 502, 482, 28, 743, 286, 412, 324],
                    [386, 48, 316, 271, 53, 407, 579, 743],
                    [693, 257, 475, 61, 891, 978, 328, 569],
                    [487, 477, 143, 933, 676, 169, 734, 159],
                    [487, 875, 1014, 96, 608, 923, 839, 328],
                    [412, 777, 721, 498, 298, 999, 328, 49],
                    [534, 447, 316, 966, 562, 85, 589, 216],
                    [173, 309, 289, 904, 670, 446, 836, 664],
                    [263, 368, 328, 377, 169, 348, 913, 612],
                    [679, 501, 475, 555, 358, 140, 632, 210],
                    [252, 997, 147, 240, 109, 671, 41, 977],
                    [440, 771, 917, 362, 911, 450, 326, 633],
                    [263, 552, 834, 488, 357, 964, 1007, 785],
                    [440, 477, 517, 688, 227, 597, 655, 458],
                    [675, 311, 517, 672, 270, 919, 387, 30],
                    [979, 484, 917, 240, 281, 735, 1021, 30],
                    [569, 347, 910, 503, 594, 923, 803, 120],
                    [390, 477, 205, 218, 562, 597, 944, 740],
                    [534, 904, 142, 388, 975, 482, 744, 187],
                    [226, 69, 364, 61, 22, 42, 879, 4],
                    [226, 108, 234, 754, 776, 42, 271, 652],
                    [679, 1008, 107, 756, 578, 482, 729, 829],
                    [695, 363, 234, 977, 190, 340, 684, 684]
                ]
            ]
        )
        # fmt: on

        torch.testing.assert_close(outputs.cpu(), EXPECTED_OUTPUT_TOKENS)

    @slow
    @require_torch_accelerator
    def test_zero_shot_voice_cloning(self):
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "Generate audio following instruction."}]},
            {"role": "scene", "content": [{"type": "text", "text": "Audio is recorded from a quiet room."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Twas the night before my birthday. Hooray! It's almost here! It may not be a holiday, but it's the best day of the year.",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "audio",
                        "url": "https://huggingface.co/datasets/eustlb/dummy-audio-samples-higgs/resolve/main/belinda.wav",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
                    }
                ],
            },
        ]

        model = HiggsAudioV2ForConditionalGeneration.from_pretrained(self.checkpoint_name, device_map=torch_device)

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            sampling_rate=24000,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=50)
        outputs = outputs[:, inputs.audio_input_ids.shape[1]:, :]

        # fmt: off
        EXPECTED_OUTPUT_TOKENS = torch.tensor(
            [
                [
                    [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [800, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [800, 354, 1024, 1024, 1024, 1024, 1024, 1024],
                    [800, 74, 427, 1024, 1024, 1024, 1024, 1024],
                    [800, 74, 468, 351, 1024, 1024, 1024, 1024],
                    [989, 202, 468, 467, 433, 1024, 1024, 1024],
                    [800, 202, 468, 467, 433, 552, 1024, 1024],
                    [321, 478, 553, 351, 916, 552, 926, 1024],
                    [800, 202, 553, 675, 324, 764, 926, 858],
                    [989, 74, 468, 343, 324, 552, 926, 858],
                    [989, 74, 52, 865, 283, 552, 405, 669],
                    [989, 873, 427, 467, 433, 764, 602, 419],
                    [321, 528, 553, 467, 433, 552, 233, 419],
                    [321, 74, 478, 351, 433, 325, 44, 669],
                    [800, 641, 468, 550, 324, 653, 926, 858],
                    [357, 998, 16, 467, 283, 904, 405, 419],
                    [1012, 337, 478, 584, 283, 552, 926, 708],
                    [267, 212, 91, 913, 324, 342, 442, 708],
                    [752, 297, 299, 921, 588, 890, 505, 858],
                    [752, 357, 70, 229, 1012, 904, 505, 390],
                    [299, 267, 986, 923, 546, 885, 986, 669],
                    [299, 203, 986, 696, 336, 182, 426, 217],
                    [414, 849, 91, 696, 324, 450, 11, 150],
                    [483, 313, 1010, 260, 513, 702, 64, 769],
                    [414, 911, 969, 315, 9, 254, 90, 417],
                    [353, 440, 279, 159, 172, 778, 624, 229],
                    [353, 187, 204, 767, 351, 441, 31, 450],
                    [373, 282, 204, 1023, 244, 152, 744, 150],
                    [572, 282, 277, 826, 267, 128, 240, 1010],
                    [263, 717, 277, 359, 460, 458, 32, 743],
                    [683, 708, 35, 64, 811, 595, 27, 618],
                    [487, 693, 91, 164, 811, 160, 475, 541],
                    [487, 820, 871, 204, 595, 221, 616, 842],
                    [135, 184, 273, 939, 624, 786, 616, 795],
                    [483, 2, 76, 663, 231, 145, 977, 116],
                    [721, 820, 279, 267, 172, 299, 247, 735],
                    [318, 12, 507, 658, 431, 198, 313, 136],
                    [114, 352, 112, 290, 39, 778, 459, 665],
                    [840, 653, 256, 747, 921, 724, 434, 806],
                    [588, 274, 986, 442, 483, 953, 656, 388],
                    [602, 33, 507, 0, 196, 691, 766, 709],
                    [114, 555, 2, 763, 208, 254, 513, 832],
                    [752, 946, 528, 1016, 338, 212, 970, 883],
                    [644, 278, 932, 119, 497, 925, 60, 401],
                    [128, 774, 1006, 381, 271, 373, 38, 523],
                    [637, 940, 25, 64, 624, 764, 122, 544],
                    [637, 479, 377, 172, 111, 746, 559, 891],
                    [679, 596, 332, 848, 651, 292, 297, 741],
                    [721, 849, 283, 899, 641, 142, 785, 743],
                    [840, 492, 91, 769, 906, 480, 329, 401]
                ]
            ]
        )
        # fmt: on

        torch.testing.assert_close(outputs.cpu(), EXPECTED_OUTPUT_TOKENS)

    @slow
    @require_torch_accelerator
    def test_multi_speaker_voice_cloning(self):
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "Generate audio following instruction."}]},
            {
                "role": "scene",
                "content": [
                    {"type": "text", "text": "Audio is recorded from a quiet room."},
                    {"type": "text", "text": "SPEAKER0:"},
                    {
                        "type": "audio",
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav",
                    },
                    {"type": "text", "text": "SPEAKER1:"},
                    {
                        "type": "audio",
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[SPEAKER0] I can't believe you did that without even asking me first!"}
                ],
            },
        ]

        model = HiggsAudioV2ForConditionalGeneration.from_pretrained(self.checkpoint_name, device_map=torch_device)

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            sampling_rate=24000,
            return_tensors="pt",
        ).to(model.device)  

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=50)
        outputs = outputs[:, inputs.audio_input_ids.shape[1]:, :]

        # fmt: off
        EXPECTED_OUTPUT_TOKENS = torch.tensor(
            [
                [
                    [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [633, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [242, 908, 1024, 1024, 1024, 1024, 1024, 1024],
                    [242, 176, 1018, 1024, 1024, 1024, 1024, 1024],
                    [463, 978, 386, 647, 1024, 1024, 1024, 1024],
                    [703, 978, 386, 713, 93, 1024, 1024, 1024],
                    [703, 834, 816, 88, 272, 812, 1024, 1024],
                    [26, 834, 227, 88, 554, 227, 680, 1024],
                    [846, 278, 227, 444, 807, 279, 538, 843],
                    [846, 499, 960, 692, 626, 258, 663, 895],
                    [856, 537, 960, 256, 362, 330, 1023, 407],
                    [83, 770, 468, 351, 984, 330, 602, 729],
                    [1016, 716, 553, 351, 513, 904, 602, 166],
                    [440, 184, 861, 51, 968, 904, 963, 239],
                    [160, 792, 367, 77, 968, 153, 760, 590],
                    [588, 122, 227, 261, 588, 342, 118, 588],
                    [1016, 352, 693, 635, 803, 342, 38, 215],
                    [20, 972, 164, 278, 742, 557, 770, 588],
                    [774, 394, 231, 959, 338, 739, 27, 293],
                    [603, 186, 231, 876, 645, 621, 842, 253],
                    [851, 12, 587, 1009, 313, 860, 331, 532],
                    [936, 321, 543, 842, 746, 239, 853, 106],
                    [900, 321, 207, 784, 846, 986, 906, 678],
                    [956, 337, 973, 436, 705, 1015, 476, 99],
                    [198, 491, 277, 406, 785, 405, 875, 849],
                    [470, 423, 50, 280, 352, 356, 72, 501],
                    [401, 66, 794, 950, 8, 194, 205, 393],
                    [431, 189, 232, 594, 787, 810, 929, 211],
                    [572, 25, 255, 321, 544, 394, 822, 433],
                    [572, 938, 139, 568, 525, 40, 532, 852],
                    [995, 78, 573, 835, 944, 884, 871, 823],
                    [30, 601, 25, 201, 32, 751, 84, 444],
                    [12, 987, 293, 532, 1020, 669, 158, 178],
                    [737, 952, 573, 339, 835, 111, 782, 836],
                    [723, 119, 967, 213, 591, 215, 633, 359],
                    [946, 575, 164, 815, 543, 715, 238, 765],
                    [286, 119, 164, 267, 176, 739, 147, 1003],
                    [817, 637, 860, 320, 946, 330, 678, 997],
                    [307, 571, 176, 243, 917, 551, 551, 81],
                    [390, 220, 661, 844, 393, 495, 124, 161],
                    [177, 99, 843, 289, 318, 814, 221, 279],
                    [981, 1001, 910, 807, 590, 383, 629, 468],
                    [879, 220, 824, 67, 0, 357, 484, 425],
                    [749, 601, 30, 650, 877, 291, 443, 1016],
                    [12, 521, 729, 406, 276, 185, 36, 733],
                    [1006, 533, 609, 486, 837, 766, 226, 167],
                    [292, 435, 969, 413, 1005, 413, 331, 310],
                    [391, 289, 80, 525, 77, 177, 374, 580],
                    [908, 2, 474, 864, 236, 413, 923, 732],
                    [12, 96, 750, 885, 545, 209, 740, 654]
                ]
            ]
        )
        # fmt: on

        torch.testing.assert_close(outputs.cpu(), EXPECTED_OUTPUT_TOKENS)

    @slow
    @require_torch_accelerator
    def test_batched_inference(self):
        """
        Testing batched inference for zero shot voice cloning.
        """
        conversation1 = [
            {"role": "system", "content": [{"type": "text", "text": "Generate audio following instruction."}]},
            {"role": "scene", "content": [{"type": "text", "text": "Audio is recorded from a quiet room."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Twas the night before my birthday. Hooray! It's almost here! It may not be a holiday, but it's the best day of the year.",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "audio",
                        "url": "https://huggingface.co/datasets/eustlb/dummy-audio-samples-higgs/resolve/main/belinda.wav",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
                    }
                ],
            },
        ]

        conversation2 = [
            {"role": "system", "content": [{"type": "text", "text": "Generate audio following instruction."}]},
            {"role": "scene", "content": [{"type": "text", "text": "Audio is recorded from a quiet room."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": " It's super important to assess fairly the fact that our former model is over. And this is not a question of adjustment. This is not the same world, 2024, 2025. And on top of that, we are making the same mistakes, on top of the key elements I mentioned. We are over-regulating and under-investing. So just if, in the two to three years to come, if we follow our classical agenda, we will be out of the market. I have no doubts.",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "audio",
                        "url": "https://huggingface.co/datasets/eustlb/dummy-audio-samples-higgs/resolve/main/macron.wav",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Hey, here is a clone from the given voice."}]},
        ]

        model = HiggsAudioV2ForConditionalGeneration.from_pretrained(self.checkpoint_name, device_map=torch_device)

        inputs = self.processor.apply_chat_template(
            [conversation1, conversation2],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            sampling_rate=24000,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=50)
        outputs = outputs[:, inputs.audio_input_ids.shape[1]:, :]

        # fmt: off
        EXPECTED_OUTPUT_TOKENS = torch.tensor(
            [
                [
                    [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [800, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [800, 354, 1024, 1024, 1024, 1024, 1024, 1024],
                    [989, 74, 427, 1024, 1024, 1024, 1024, 1024],
                    [989, 202, 468, 351, 1024, 1024, 1024, 1024],
                    [800, 202, 553, 467, 421, 1024, 1024, 1024],
                    [800, 74, 553, 467, 433, 764, 1024, 1024],
                    [800, 74, 468, 351, 433, 764, 926, 1024],
                    [989, 202, 468, 467, 433, 552, 926, 858],
                    [321, 202, 468, 351, 433, 552, 926, 858],
                    [989, 74, 553, 161, 433, 552, 44, 419],
                    [989, 150, 468, 467, 283, 764, 602, 419],
                    [321, 537, 136, 351, 433, 904, 926, 858],
                    [349, 442, 873, 391, 433, 764, 926, 669],
                    [560, 288, 478, 467, 799, 764, 926, 858],
                    [637, 917, 862, 293, 283, 764, 44, 669],
                    [752, 462, 435, 552, 843, 552, 926, 390],
                    [752, 917, 25, 894, 588, 904, 474, 407],
                    [752, 540, 91, 923, 391, 974, 505, 708],
                    [299, 199, 198, 894, 744, 192, 986, 165],
                    [483, 233, 986, 696, 513, 451, 83, 988],
                    [993, 313, 534, 260, 838, 254, 683, 275],
                    [414, 911, 251, 329, 940, 254, 296, 447],
                    [667, 137, 377, 519, 686, 323, 742, 335],
                    [642, 750, 377, 465, 336, 778, 375, 217],
                    [431, 568, 769, 842, 841, 152, 967, 929],
                    [373, 282, 474, 589, 741, 110, 536, 340],
                    [727, 282, 474, 835, 434, 989, 553, 323],
                    [128, 8, 848, 64, 895, 458, 512, 723],
                    [487, 185, 48, 586, 162, 477, 646, 345],
                    [455, 429, 462, 193, 368, 908, 28, 15],
                    [455, 781, 515, 927, 624, 985, 414, 695],
                    [487, 595, 273, 727, 203, 427, 373, 758],
                    [286, 696, 256, 30, 846, 460, 311, 574],
                    [314, 703, 5, 1018, 143, 930, 975, 866],
                    [873, 579, 618, 849, 344, 703, 447, 809],
                    [752, 946, 764, 648, 614, 659, 129, 421],
                    [588, 946, 91, 839, 582, 259, 359, 333],
                    [588, 33, 703, 517, 434, 606, 728, 949],
                    [114, 584, 653, 104, 493, 203, 834, 865],
                    [1008, 278, 38, 759, 381, 549, 966, 588],
                    [568, 282, 30, 751, 475, 212, 792, 676],
                    [679, 35, 377, 574, 212, 212, 127, 378],
                    [637, 187, 711, 709, 1015, 530, 765, 804],
                    [534, 570, 44, 323, 34, 573, 751, 49],
                    [721, 355, 332, 76, 370, 10, 958, 427],
                    [785, 492, 711, 281, 16, 790, 1002, 788],
                    [489, 583, 44, 442, 592, 566, 37, 862],
                    [386, 426, 929, 818, 823, 771, 873, 148],
                    [418, 583, 696, 119, 895, 819, 826, 129]
                ],
                [
                    [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [244, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [624, 518, 1024, 1024, 1024, 1024, 1024, 1024],
                    [497, 945, 427, 1024, 1024, 1024, 1024, 1024],
                    [207, 963, 270, 841, 1024, 1024, 1024, 1024],
                    [207, 172, 468, 865, 799, 1024, 1024, 1024],
                    [497, 914, 270, 514, 362, 243, 1024, 1024],
                    [843, 1021, 427, 550, 609, 653, 602, 1024],
                    [850, 893, 427, 880, 362, 552, 53, 981],
                    [497, 415, 66, 865, 609, 890, 581, 364],
                    [666, 1021, 468, 343, 984, 890, 798, 340],
                    [533, 415, 725, 584, 843, 552, 986, 229],
                    [515, 172, 270, 351, 984, 243, 765, 727],
                    [207, 983, 468, 550, 984, 904, 53, 419],
                    [207, 802, 0, 157, 916, 882, 209, 981],
                    [497, 1021, 270, 584, 609, 870, 581, 727],
                    [526, 518, 52, 550, 843, 764, 677, 606],
                    [675, 763, 265, 584, 614, 764, 986, 447],
                    [6, 269, 905, 161, 769, 552, 798, 364],
                    [497, 1021, 586, 161, 916, 270, 22, 340],
                    [497, 878, 434, 974, 222, 765, 53, 562],
                    [207, 21, 270, 934, 843, 870, 209, 562],
                    [207, 914, 265, 550, 609, 904, 53, 419],
                    [207, 1021, 427, 584, 984, 890, 765, 25],
                    [497, 415, 270, 880, 916, 325, 83, 1021],
                    [515, 873, 553, 467, 984, 270, 22, 407],
                    [207, 21, 270, 584, 445, 890, 652, 727],
                    [83, 415, 265, 467, 344, 552, 83, 727],
                    [463, 198, 468, 550, 916, 342, 485, 719],
                    [748, 856, 66, 293, 916, 325, 233, 418],
                    [404, 643, 621, 433, 764, 342, 22, 562],
                    [418, 421, 194, 493, 362, 764, 405, 1023],
                    [419, 775, 78, 270, 122, 243, 725, 419],
                    [637, 471, 263, 87, 345, 651, 53, 418],
                    [998, 784, 716, 506, 777, 129, 439, 132],
                    [207, 306, 99, 466, 1015, 60, 874, 374],
                    [207, 1020, 371, 434, 715, 962, 630, 843],
                    [998, 37, 649, 124, 348, 44, 430, 336],
                    [843, 996, 649, 852, 745, 6, 642, 938],
                    [990, 346, 628, 935, 120, 990, 532, 906],
                    [814, 21, 270, 446, 777, 146, 169, 165],
                    [497, 872, 270, 496, 916, 331, 532, 58],
                    [497, 21, 270, 550, 734, 431, 381, 25],
                    [205, 963, 265, 433, 468, 431, 101, 936],
                    [207, 254, 468, 351, 468, 590, 44, 696],
                    [538, 914, 434, 467, 916, 550, 83, 606],
                    [207, 172, 427, 157, 808, 870, 83, 447],
                    [497, 415, 427, 880, 1012, 243, 201, 32],
                    [538, 873, 434, 880, 984, 764, 296, 669],
                    [497, 172, 270, 584, 916, 890, 559, 606]
                ]
            ]
        )
        # fmt: on

        torch.testing.assert_close(outputs.cpu(), EXPECTED_OUTPUT_TOKENS)
