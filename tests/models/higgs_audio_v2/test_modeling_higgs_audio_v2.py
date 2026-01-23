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
        torch.manual_seed(0)

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Generate audio following instruction."}],
            },
            {"role": "scene", "content": [{"type": "text", "text": "Audio is recorded from a quiet room."}]},
            {"role": "user", "content": [{"type": "text", "text": "The sun rises in the east and sets in the west."}]},
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

        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

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
                    [196, 562, 971, 196, 932, 645, 53, 1024],
                    [239, 432, 971, 75, 709, 157, 326, 669],
                    [668, 935, 243, 339, 406, 434, 245, 655],
                    [273, 974, 466, 400, 297, 809, 417, 794],
                    [431, 219, 568, 999, 126, 83, 677, 104],
                    [852, 797, 915, 809, 270, 720, 201, 962],
                    [988, 280, 179, 370, 647, 500, 862, 790],
                    [988, 673, 75, 651, 879, 931, 670, 446],
                    [945, 112, 102, 338, 354, 276, 770, 880],
                    [287, 18, 555, 6, 53, 357, 716, 794],
                    [300, 474, 801, 55, 377, 595, 1022, 820],
                    [169, 276, 762, 173, 743, 987, 422, 625],
                    [363, 974, 104, 886, 581, 25, 99, 249],
                    [1006, 89, 630, 197, 668, 101, 627, 197],
                    [363, 955, 961, 290, 275, 529, 242, 127],
                    [444, 192, 721, 711, 689, 778, 352, 901],
                    [300, 853, 363, 402, 217, 51, 75, 464],
                    [343, 304, 961, 833, 289, 374, 890, 682],
                    [343, 962, 784, 911, 289, 583, 463, 974],
                    [473, 227, 450, 926, 586, 957, 920, 550],
                    [300, 212, 965, 969, 659, 699, 846, 837],
                    [409, 440, 307, 995, 144, 435, 34, 510],
                    [343, 559, 812, 850, 621, 684, 72, 726],
                    [965, 227, 612, 19, 396, 627, 711, 448],
                    [740, 1019, 450, 869, 207, 751, 6, 862],
                    [740, 542, 784, 68, 400, 239, 62, 886],
                    [40, 18, 889, 414, 532, 620, 698, 43],
                    [486, 170, 152, 714, 538, 865, 1, 300],
                    [473, 153, 784, 1016, 755, 727, 700, 73],
                    [95, 305, 595, 226, 849, 333, 985, 245],
                    [221, 261, 50, 117, 42, 697, 808, 326],
                    [300, 261, 290, 966, 538, 567, 929, 518],
                    [473, 69, 458, 993, 97, 956, 99, 276],
                    [885, 560, 409, 649, 686, 377, 5, 857],
                    [676, 662, 784, 975, 674, 473, 487, 242],
                    [780, 867, 782, 926, 931, 895, 428, 86],
                    [815, 1010, 398, 637, 512, 62, 47, 49],
                    [338, 829, 784, 443, 512, 157, 596, 360],
                    [338, 590, 242, 73, 533, 298, 15, 564],
                    [338, 962, 186, 371, 462, 298, 835, 894],
                    [95, 829, 579, 984, 333, 504, 963, 451],
                    [815, 367, 300, 608, 333, 243, 78, 338],
                    [360, 393, 816, 317, 271, 488, 233, 60]
                ]
            ]
        )
        # fmt: on

        torch.testing.assert_close(outputs.cpu(), EXPECTED_OUTPUT_TOKENS)

    @slow
    @require_torch_accelerator
    def test_multi_speaker_smart_voice(self):
        torch.manual_seed(0)

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
        torch.manual_seed(0)

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
        outputs = outputs[:, inputs.audio_input_ids.shape[1] :, :]

        # fmt: off
        EXPECTED_OUTPUT_TOKENS = torch.tensor(
            [
                [
                    [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [800, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [800, 354, 1024, 1024, 1024, 1024, 1024, 1024],
                    [800, 74, 427, 1024, 1024, 1024, 1024, 1024],
                    [800, 74, 468, 351, 1024, 1024, 1024, 1024],
                    [800, 74, 468, 467, 433, 1024, 1024, 1024],
                    [800, 74, 136, 467, 433, 552, 1024, 1024],
                    [800, 354, 427, 467, 433, 764, 926, 1024],
                    [800, 202, 11, 467, 799, 764, 926, 858],
                    [800, 74, 468, 865, 433, 552, 926, 858],
                    [321, 74, 998, 351, 799, 552, 83, 669],
                    [321, 641, 468, 584, 433, 904, 926, 1023],
                    [846, 641, 16, 467, 609, 153, 581, 419],
                    [29, 717, 478, 584, 609, 890, 405, 215],
                    [683, 142, 490, 664, 799, 764, 847, 858],
                    [187, 204, 507, 608, 1012, 890, 474, 858],
                    [752, 134, 251, 759, 719, 764, 505, 669],
                    [752, 541, 507, 557, 308, 89, 374, 981],
                    [299, 267, 986, 210, 918, 324, 820, 151],
                    [114, 203, 986, 696, 493, 945, 569, 250],
                    [775, 282, 91, 696, 203, 412, 815, 6],
                    [422, 251, 830, 260, 513, 476, 127, 976],
                    [913, 696, 595, 555, 475, 342, 698, 556],
                    [25, 184, 350, 758, 753, 392, 46, 53],
                    [470, 487, 554, 159, 738, 193, 660, 694],
                    [110, 381, 363, 525, 442, 739, 316, 302],
                    [470, 288, 950, 699, 678, 929, 50, 376],
                    [136, 278, 1008, 939, 678, 797, 802, 611],
                    [263, 282, 277, 939, 337, 193, 811, 647],
                    [792, 238, 784, 212, 1021, 673, 990, 256],
                    [414, 245, 986, 567, 299, 667, 528, 11],
                    [120, 781, 515, 714, 940, 242, 939, 884],
                    [667, 184, 273, 1006, 660, 949, 772, 565],
                    [651, 113, 124, 937, 948, 242, 241, 173],
                    [861, 786, 76, 19, 142, 930, 676, 477],
                    [431, 438, 600, 737, 780, 840, 616, 811],
                    [114, 533, 239, 594, 1015, 149, 24, 54],
                    [71, 498, 611, 930, 814, 48, 925, 510],
                    [58, 278, 986, 491, 519, 189, 9, 313],
                    [431, 584, 524, 448, 746, 852, 876, 755],
                    [114, 255, 832, 115, 208, 1021, 449, 158],
                    [752, 238, 956, 832, 837, 778, 40, 398],
                    [568, 834, 286, 593, 816, 187, 1011, 1023],
                    [881, 37, 91, 382, 196, 276, 24, 446],
                    [725, 786, 934, 79, 536, 554, 817, 277],
                    [725, 849, 969, 695, 181, 802, 496, 427],
                    [572, 708, 443, 869, 250, 723, 90, 54],
                    [583, 725, 189, 151, 131, 184, 966, 176],
                    [785, 99, 91, 19, 922, 1000, 961, 423],
                    [489, 795, 123, 442, 639, 771, 462, 528]
                ]
            ]
        )
        # fmt: on

        torch.testing.assert_close(outputs.cpu(), EXPECTED_OUTPUT_TOKENS)

    @slow
    @require_torch_accelerator
    def test_multi_speaker_voice_cloning(self):
        torch.manual_seed(0)

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
        outputs = outputs[:, inputs.audio_input_ids.shape[1] :, :]

        # fmt: off
        EXPECTED_OUTPUT_TOKENS = torch.tensor(
            [
                [
                    [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [633, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [242, 908, 1024, 1024, 1024, 1024, 1024, 1024],
                    [242, 176, 1018, 1024, 1024, 1024, 1024, 1024],
                    [280, 978, 386, 647, 1024, 1024, 1024, 1024],
                    [703, 792, 386, 289, 93, 1024, 1024, 1024],
                    [703, 792, 886, 88, 272, 812, 1024, 1024],
                    [242, 808, 639, 424, 807, 599, 680, 1024],
                    [846, 808, 639, 138, 617, 334, 737, 165],
                    [846, 442, 446, 838, 117, 473, 604, 902],
                    [749, 537, 446, 184, 252, 1021, 239, 865],
                    [321, 537, 287, 214, 252, 1021, 350, 350],
                    [160, 712, 981, 865, 843, 342, 788, 274],
                    [160, 895, 862, 880, 609, 153, 869, 29],
                    [721, 465, 30, 312, 421, 890, 652, 616],
                    [888, 554, 139, 84, 39, 890, 83, 314],
                    [749, 360, 110, 565, 408, 342, 474, 391],
                    [252, 839, 367, 327, 292, 505, 242, 932],
                    [867, 99, 231, 758, 719, 1023, 443, 964],
                    [21, 194, 848, 395, 787, 776, 447, 913],
                    [913, 792, 139, 711, 155, 398, 35, 476],
                    [603, 465, 609, 64, 310, 926, 651, 365],
                    [391, 321, 158, 317, 338, 252, 593, 26],
                    [912, 391, 446, 623, 1021, 408, 219, 153],
                    [956, 389, 333, 64, 679, 139, 999, 693],
                    [798, 928, 277, 939, 25, 323, 94, 1020],
                    [12, 288, 770, 247, 410, 256, 242, 417],
                    [401, 984, 277, 735, 771, 351, 5, 593],
                    [852, 189, 285, 725, 1021, 937, 830, 249],
                    [9, 613, 108, 939, 258, 632, 884, 173],
                    [995, 129, 452, 767, 740, 326, 255, 410],
                    [945, 435, 358, 304, 467, 521, 604, 436],
                    [945, 952, 25, 782, 779, 933, 267, 952],
                    [737, 114, 746, 44, 905, 441, 450, 912],
                    [12, 173, 680, 229, 99, 105, 450, 710],
                    [723, 119, 164, 327, 215, 734, 347, 367],
                    [760, 818, 254, 997, 68, 688, 792, 555],
                    [115, 1013, 746, 626, 817, 543, 644, 522],
                    [817, 191, 236, 112, 675, 852, 223, 266],
                    [652, 868, 132, 315, 383, 845, 312, 250],
                    [730, 87, 661, 936, 626, 674, 889, 87],
                    [529, 99, 573, 82, 31, 489, 853, 266],
                    [1006, 78, 394, 207, 991, 316, 630, 990],
                    [179, 579, 139, 17, 991, 986, 973, 115],
                    [652, 235, 294, 241, 975, 124, 189, 321],
                    [514, 827, 230, 682, 736, 876, 939, 819],
                    [707, 533, 914, 737, 302, 584, 276, 319],
                    [437, 66, 500, 131, 632, 1023, 65, 824],
                    [116, 96, 232, 378, 702, 651, 421, 875],
                    [116, 471, 750, 112, 612, 932, 951, 856]
                ]
            ]
        )
        # fmt: on

        torch.testing.assert_close(outputs.cpu(), EXPECTED_OUTPUT_TOKENS)

    @slow
    @require_torch_accelerator
    def test_batched_inference(self):
        torch.manual_seed(0)

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
        outputs = outputs[:, inputs.audio_input_ids.shape[1] :, :]

        # fmt: off
        EXPECTED_OUTPUT_TOKENS = torch.tensor(
            [
                [
                    [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [800, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [800, 354, 1024, 1024, 1024, 1024, 1024, 1024],
                    [800, 74, 427, 1024, 1024, 1024, 1024, 1024],
                    [800, 74, 427, 351, 1024, 1024, 1024, 1024],
                    [800, 74, 468, 467, 421, 1024, 1024, 1024],
                    [800, 74, 468, 467, 433, 764, 1024, 1024],
                    [244, 354, 427, 467, 433, 552, 926, 1024],
                    [989, 528, 553, 467, 609, 764, 926, 858],
                    [800, 202, 553, 880, 421, 552, 926, 419],
                    [321, 74, 553, 584, 283, 904, 405, 858],
                    [989, 88, 468, 351, 433, 552, 405, 858],
                    [357, 100, 136, 467, 799, 764, 442, 708],
                    [1009, 360, 663, 974, 609, 890, 926, 708],
                    [752, 496, 715, 128, 799, 904, 926, 151],
                    [752, 360, 195, 835, 188, 764, 405, 390],
                    [114, 267, 415, 478, 391, 666, 986, 669],
                    [299, 1011, 986, 478, 298, 950, 426, 419],
                    [487, 152, 286, 927, 607, 985, 652, 299],
                    [292, 820, 830, 696, 818, 944, 183, 724],
                    [707, 137, 783, 779, 588, 254, 201, 313],
                    [524, 907, 858, 802, 902, 254, 90, 644],
                    [418, 950, 141, 887, 342, 425, 296, 104],
                    [1009, 946, 474, 521, 649, 505, 284, 0],
                    [28, 8, 474, 829, 434, 625, 645, 40],
                    [78, 473, 332, 164, 888, 124, 307, 203],
                    [908, 968, 986, 586, 110, 79, 112, 207],
                    [908, 781, 871, 212, 334, 771, 1014, 187],
                    [292, 781, 486, 690, 230, 569, 369, 998],
                    [246, 289, 884, 690, 296, 211, 311, 850],
                    [1012, 820, 615, 135, 547, 961, 293, 405],
                    [1012, 289, 703, 803, 991, 329, 584, 348],
                    [114, 341, 788, 175, 120, 550, 790, 619],
                    [29, 946, 279, 238, 339, 833, 961, 199],
                    [702, 106, 332, 90, 921, 257, 21, 933],
                    [682, 249, 91, 517, 374, 626, 241, 831],
                    [299, 274, 653, 878, 956, 445, 877, 770],
                    [29, 565, 91, 614, 711, 918, 735, 150],
                    [455, 894, 256, 867, 534, 440, 508, 340],
                    [524, 696, 564, 959, 1010, 74, 604, 733],
                    [878, 618, 332, 0, 208, 819, 119, 206],
                    [1012, 946, 474, 198, 607, 690, 434, 637],
                    [92, 194, 618, 64, 849, 212, 419, 508],
                    [644, 413, 332, 586, 566, 537, 163, 723],
                    [644, 57, 15, 37, 566, 346, 45, 431],
                    [386, 650, 394, 805, 943, 221, 585, 147],
                    [418, 793, 258, 973, 840, 897, 740, 95],
                    [418, 492, 197, 81, 769, 4, 716, 369],
                    [330, 492, 955, 957, 351, 114, 683, 255],
                    [418, 533, 630, 543, 834, 306, 355, 592]
                ],
                [
                    [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [244, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [624, 518, 1024, 1024, 1024, 1024, 1024, 1024],
                    [497, 945, 427, 1024, 1024, 1024, 1024, 1024],
                    [207, 963, 270, 841, 1024, 1024, 1024, 1024],
                    [207, 172, 468, 865, 799, 1024, 1024, 1024],
                    [497, 914, 270, 514, 362, 243, 1024, 1024],
                    [533, 1021, 427, 550, 609, 653, 602, 1024],
                    [693, 172, 468, 880, 362, 1021, 53, 981],
                    [497, 963, 632, 343, 609, 890, 581, 364],
                    [497, 1021, 905, 351, 984, 890, 595, 340],
                    [207, 21, 265, 467, 513, 764, 233, 719],
                    [207, 415, 265, 351, 916, 904, 22, 719],
                    [497, 415, 270, 161, 916, 585, 505, 340],
                    [497, 21, 270, 584, 983, 342, 296, 407],
                    [526, 478, 42, 584, 984, 342, 53, 438],
                    [526, 624, 16, 161, 68, 75, 798, 419],
                    [82, 802, 270, 467, 421, 75, 233, 932],
                    [766, 14, 270, 550, 609, 552, 1018, 727],
                    [754, 438, 350, 862, 984, 552, 426, 438],
                    [328, 1009, 870, 741, 482, 826, 22, 418],
                    [16, 746, 7, 231, 157, 826, 83, 719],
                    [637, 163, 661, 755, 336, 812, 53, 418],
                    [850, 421, 674, 145, 253, 497, 201, 804],
                    [207, 127, 297, 996, 409, 380, 350, 932],
                    [207, 1020, 649, 455, 294, 380, 363, 151],
                    [998, 1020, 958, 262, 874, 157, 976, 438],
                    [207, 37, 1013, 984, 177, 568, 215, 944],
                    [51, 37, 371, 135, 553, 957, 46, 802],
                    [990, 21, 649, 184, 151, 503, 742, 285],
                    [240, 1021, 1, 547, 882, 503, 180, 507],
                    [497, 21, 270, 187, 916, 861, 462, 569],
                    [497, 1021, 881, 880, 984, 377, 823, 610],
                    [497, 21, 265, 584, 916, 962, 760, 1023],
                    [497, 983, 265, 351, 912, 493, 417, 165],
                    [497, 21, 270, 467, 83, 153, 474, 419],
                    [538, 503, 73, 161, 984, 12, 379, 563],
                    [207, 172, 270, 161, 368, 270, 53, 563],
                    [515, 415, 427, 550, 609, 552, 53, 418],
                    [497, 1021, 434, 351, 609, 904, 22, 669],
                    [207, 503, 468, 584, 799, 552, 209, 562],
                    [240, 415, 468, 819, 609, 50, 22, 419],
                    [515, 21, 52, 351, 609, 342, 474, 1023],
                    [497, 21, 434, 584, 799, 552, 474, 1023],
                    [515, 1021, 427, 584, 843, 243, 602, 669],
                    [207, 873, 427, 467, 1012, 342, 44, 215],
                    [497, 415, 346, 865, 283, 890, 233, 390],
                    [403, 1021, 270, 293, 421, 77, 485, 364],
                    [497, 503, 468, 675, 188, 890, 405, 419],
                    [497, 873, 270, 351, 799, 653, 581, 390]
                ]
            ]
        )
        # fmt: on

        torch.testing.assert_close(outputs.cpu(), EXPECTED_OUTPUT_TOKENS)
