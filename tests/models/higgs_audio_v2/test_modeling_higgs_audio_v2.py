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
        self.checkpoint_name = "eustlb/higgs-audio-v2-generation-3B-base"
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
                    [470, 644, 142, 805, 185, 608, 442, 1024],
                    [662, 334, 297, 890, 982, 100, 248, 727],
                    [492, 582, 697, 658, 856, 169, 367, 201],
                    [867, 51, 322, 586, 929, 897, 959, 96],
                    [470, 566, 934, 188, 594, 308, 881, 385],
                    [874, 63, 189, 859, 443, 980, 48, 94],
                    [860, 51, 5, 290, 719, 484, 537, 136],
                    [646, 347, 446, 978, 793, 152, 909, 112],
                    [646, 253, 104, 277, 663, 792, 626, 568],
                    [646, 309, 148, 448, 973, 482, 604, 664],
                    [173, 114, 702, 701, 50, 446, 976, 30],
                    [874, 855, 677, 273, 227, 351, 859, 652],
                    [821, 363, 834, 901, 19, 805, 135, 328],
                    [874, 452, 517, 805, 47, 588, 452, 646],
                    [412, 809, 702, 998, 595, 503, 423, 582],
                    [569, 648, 208, 0, 353, 724, 141, 507],
                    [95, 473, 548, 483, 903, 280, 888, 528],
                    [259, 477, 733, 26, 889, 748, 452, 569],
                    [549, 382, 845, 421, 417, 305, 101, 663],
                    [253, 334, 524, 333, 662, 644, 207, 742],
                    [569, 107, 446, 160, 109, 12, 348, 1012],
                    [736, 160, 5, 879, 25, 819, 781, 636],
                    [849, 895, 840, 898, 227, 739, 658, 988],
                    [422, 582, 547, 115, 888, 856, 178, 495],
                    [446, 107, 507, 160, 1009, 145, 741, 351],
                    [31, 582, 835, 879, 947, 169, 452, 136],
                    [273, 466, 189, 845, 326, 94, 11, 973],
                    [861, 64, 315, 776, 594, 482, 630, 940],
                    [422, 346, 984, 931, 299, 435, 331, 832],
                    [944, 691, 283, 185, 461, 731, 1008, 206],
                    [854, 582, 835, 425, 458, 56, 438, 302],
                    [65, 582, 820, 713, 298, 187, 835, 652],
                    [549, 245, 466, 716, 710, 381, 10, 179],
                    [874, 452, 394, 623, 595, 349, 881, 859],
                    [979, 309, 507, 33, 171, 316, 354, 326],
                    [422, 741, 517, 357, 554, 482, 496, 883],
                    [874, 311, 719, 142, 554, 616, 50, 652],
                    [902, 277, 548, 505, 581, 226, 537, 100],
                    [472, 895, 835, 0, 595, 967, 437, 130],
                    [176, 356, 673, 700, 745, 627, 877, 714],
                    [273, 582, 517, 366, 50, 980, 790, 454],
                    [854, 582, 295, 380, 175, 268, 452, 752],
                    [65, 64, 835, 623, 1009, 548, 568, 746]
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
                    [800, 478, 427, 467, 433, 764, 926, 1024],
                    [800, 202, 52, 467, 433, 764, 926, 858],
                    [800, 74, 468, 343, 433, 552, 926, 858],
                    [321, 74, 136, 550, 513, 552, 83, 669],
                    [321, 74, 468, 161, 433, 890, 602, 858],
                    [846, 641, 468, 467, 799, 653, 926, 419],
                    [151, 717, 478, 161, 362, 890, 602, 419],
                    [29, 878, 490, 862, 799, 904, 485, 981],
                    [71, 212, 677, 723, 283, 342, 926, 858],
                    [114, 357, 299, 486, 646, 764, 926, 669],
                    [299, 1004, 91, 894, 208, 440, 986, 981],
                    [299, 203, 91, 224, 793, 649, 986, 867],
                    [487, 282, 619, 517, 250, 571, 42, 209],
                    [246, 218, 830, 260, 838, 576, 181, 736],
                    [487, 907, 216, 323, 9, 778, 53, 958],
                    [784, 739, 93, 719, 374, 472, 924, 765],
                    [367, 984, 554, 757, 120, 1016, 995, 819],
                    [14, 950, 204, 266, 672, 557, 598, 1013],
                    [808, 947, 474, 543, 120, 906, 327, 317],
                    [614, 282, 277, 769, 895, 198, 909, 417],
                    [135, 185, 276, 649, 895, 658, 81, 360],
                    [487, 968, 462, 188, 824, 740, 286, 723],
                    [487, 879, 394, 212, 682, 824, 546, 397],
                    [487, 96, 273, 517, 1018, 211, 113, 261],
                    [614, 251, 75, 414, 969, 1016, 645, 76],
                    [286, 643, 141, 281, 672, 446, 578, 107],
                    [114, 351, 44, 517, 769, 924, 688, 334],
                    [71, 677, 769, 440, 665, 794, 793, 864],
                    [759, 278, 286, 972, 635, 794, 129, 1012],
                    [263, 392, 123, 67, 365, 651, 117, 62],
                    [114, 274, 780, 661, 911, 20, 219, 12],
                    [946, 677, 394, 808, 544, 671, 795, 230],
                    [314, 555, 286, 238, 363, 242, 29, 38],
                    [784, 156, 910, 567, 96, 573, 643, 541],
                    [784, 708, 474, 425, 624, 314, 405, 1012],
                    [721, 541, 474, 223, 260, 449, 474, 910],
                    [386, 99, 44, 823, 495, 212, 84, 4],
                    [644, 793, 823, 385, 823, 566, 120, 460],
                    [386, 403, 6, 48, 991, 93, 514, 456],
                    [330, 341, 258, 902, 229, 212, 597, 656],
                    [1012, 513, 21, 902, 704, 430, 373, 62],
                    [330, 513, 903, 659, 719, 691, 701, 536],
                    [330, 57, 903, 957, 49, 309, 992, 187]
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
                    [242, 808, 639, 424, 807, 654, 680, 1024],
                    [846, 808, 639, 138, 617, 334, 737, 165],
                    [846, 442, 446, 838, 995, 473, 419, 479],
                    [749, 537, 446, 184, 337, 1021, 509, 440],
                    [321, 537, 287, 214, 252, 1021, 350, 453],
                    [160, 712, 981, 865, 843, 890, 64, 453],
                    [160, 818, 862, 880, 609, 153, 805, 29],
                    [721, 642, 283, 312, 421, 890, 721, 1021],
                    [888, 186, 82, 983, 283, 325, 417, 215],
                    [749, 357, 110, 956, 161, 243, 182, 1023],
                    [252, 839, 824, 609, 484, 49, 505, 620],
                    [867, 99, 231, 534, 322, 990, 443, 964],
                    [21, 194, 726, 395, 886, 692, 354, 532],
                    [25, 352, 139, 766, 887, 855, 921, 391],
                    [851, 486, 30, 37, 482, 456, 19, 740],
                    [912, 465, 158, 583, 607, 610, 565, 639],
                    [12, 389, 80, 956, 450, 471, 532, 130],
                    [798, 928, 277, 788, 642, 946, 353, 83],
                    [401, 288, 277, 979, 879, 323, 491, 268],
                    [30, 314, 877, 190, 626, 335, 630, 906],
                    [14, 538, 80, 461, 790, 348, 106, 779],
                    [723, 938, 255, 12, 863, 632, 302, 855],
                    [177, 924, 737, 745, 825, 94, 302, 609],
                    [725, 521, 500, 22, 104, 473, 566, 462],
                    [721, 194, 983, 743, 1005, 324, 119, 764],
                    [12, 811, 112, 1023, 611, 370, 960, 334],
                    [727, 1007, 164, 517, 1005, 902, 342, 828],
                    [683, 186, 681, 180, 4, 803, 151, 327],
                    [286, 601, 164, 543, 113, 72, 210, 448],
                    [652, 341, 600, 737, 891, 86, 412, 1003],
                    [842, 600, 241, 488, 499, 781, 457, 626],
                    [66, 719, 1003, 419, 813, 415, 990, 35],
                    [165, 39, 754, 276, 399, 615, 556, 448],
                    [29, 240, 609, 1012, 368, 202, 643, 384],
                    [413, 408, 956, 64, 748, 626, 204, 1012],
                    [842, 601, 232, 906, 612, 291, 189, 762],
                    [707, 180, 729, 462, 673, 803, 366, 860],
                    [747, 579, 500, 428, 997, 948, 33, 158],
                    [851, 96, 983, 565, 633, 444, 630, 738],
                    [851, 138, 507, 195, 428, 739, 921, 663],
                    [12, 96, 750, 897, 379, 810, 740, 648],
                    [12, 682, 590, 156, 370, 86, 178, 436],
                    [12, 164, 383, 979, 229, 865, 860, 325]
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
                    [800, 74, 468, 351, 1024, 1024, 1024, 1024],
                    [800, 74, 468, 467, 433, 1024, 1024, 1024],
                    [800, 74, 136, 467, 433, 552, 1024, 1024],
                    [800, 354, 427, 467, 433, 552, 926, 1024],
                    [800, 202, 11, 467, 799, 764, 926, 858],
                    [800, 74, 468, 865, 433, 764, 926, 858],
                    [321, 74, 998, 351, 799, 552, 22, 669],
                    [321, 641, 468, 584, 433, 904, 926, 1023],
                    [846, 641, 998, 467, 609, 153, 581, 419],
                    [29, 717, 300, 584, 609, 890, 405, 215],
                    [683, 142, 490, 664, 324, 904, 926, 708],
                    [187, 204, 91, 483, 1012, 552, 44, 708],
                    [114, 878, 251, 371, 143, 890, 442, 390],
                    [752, 755, 332, 177, 711, 440, 442, 606],
                    [752, 267, 462, 470, 277, 521, 986, 719],
                    [114, 267, 986, 555, 434, 617, 464, 727],
                    [299, 1011, 986, 436, 838, 412, 685, 574],
                    [414, 152, 286, 436, 513, 254, 418, 296],
                    [414, 313, 826, 51, 513, 254, 83, 1021],
                    [270, 911, 496, 779, 10, 778, 624, 539],
                    [9, 435, 189, 543, 607, 771, 624, 852],
                    [927, 408, 96, 491, 841, 566, 49, 77],
                    [450, 849, 509, 955, 605, 929, 552, 407],
                    [861, 790, 474, 64, 608, 25, 327, 762],
                    [727, 282, 164, 522, 759, 458, 256, 701],
                    [614, 8, 848, 17, 94, 673, 612, 126],
                    [135, 185, 903, 591, 256, 561, 1003, 970],
                    [487, 968, 462, 517, 149, 968, 618, 207],
                    [487, 52, 394, 193, 62, 478, 641, 145],
                    [797, 696, 51, 79, 1018, 472, 466, 488],
                    [314, 785, 884, 146, 812, 483, 311, 547],
                    [721, 355, 279, 266, 322, 446, 913, 375],
                    [873, 429, 745, 266, 780, 758, 332, 252],
                    [299, 910, 909, 223, 80, 833, 806, 465],
                    [572, 26, 398, 4, 921, 280, 1013, 397],
                    [583, 693, 337, 953, 698, 795, 307, 844],
                    [318, 710, 844, 132, 208, 848, 515, 186],
                    [299, 894, 715, 823, 208, 441, 183, 596],
                    [845, 32, 394, 192, 843, 383, 142, 476],
                    [881, 355, 91, 517, 202, 437, 436, 199],
                    [679, 792, 930, 555, 684, 568, 662, 280],
                    [679, 849, 892, 763, 760, 254, 827, 707],
                    [356, 596, 474, 198, 374, 341, 722, 415],
                    [588, 274, 283, 195, 484, 314, 959, 488],
                    [785, 69, 91, 759, 956, 945, 719, 363],
                    [644, 793, 1010, 784, 796, 362, 147, 663]
                ],
                [
                    [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [244, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                    [624, 518, 1024, 1024, 1024, 1024, 1024, 1024],
                    [497, 945, 427, 1024, 1024, 1024, 1024, 1024],
                    [207, 963, 270, 841, 1024, 1024, 1024, 1024],
                    [207, 172, 468, 865, 799, 1024, 1024, 1024],
                    [497, 914, 270, 514, 362, 243, 1024, 1024],
                    [666, 1021, 427, 550, 609, 653, 579, 1024],
                    [998, 415, 270, 880, 362, 552, 53, 981],
                    [515, 9, 65, 351, 609, 890, 581, 364],
                    [497, 415, 265, 351, 984, 890, 798, 340],
                    [207, 873, 0, 550, 799, 890, 22, 229],
                    [272, 172, 427, 433, 916, 243, 798, 719],
                    [207, 1003, 270, 161, 843, 342, 760, 1023],
                    [497, 983, 52, 974, 984, 342, 118, 708],
                    [526, 1021, 982, 584, 609, 552, 296, 418],
                    [82, 349, 265, 157, 433, 666, 53, 340],
                    [717, 14, 270, 675, 421, 270, 494, 364],
                    [792, 438, 350, 550, 609, 347, 83, 419],
                    [883, 820, 672, 741, 483, 75, 798, 438],
                    [609, 662, 321, 196, 574, 826, 417, 544],
                    [1012, 657, 653, 430, 921, 812, 53, 981],
                    [442, 438, 669, 745, 253, 584, 201, 438],
                    [498, 800, 431, 23, 214, 602, 668, 1006],
                    [207, 306, 838, 542, 247, 503, 850, 151],
                    [207, 1020, 93, 283, 835, 37, 860, 976],
                    [272, 37, 265, 928, 935, 889, 613, 571],
                    [911, 810, 265, 113, 156, 243, 363, 946],
                    [280, 624, 270, 149, 912, 537, 201, 860],
                    [758, 576, 321, 865, 916, 636, 663, 1022],
                    [179, 136, 854, 595, 541, 937, 737, 802],
                    [106, 359, 98, 28, 916, 544, 330, 25],
                    [213, 52, 321, 970, 403, 544, 405, 341],
                    [16, 457, 134, 408, 622, 883, 53, 320],
                    [1012, 45, 15, 608, 345, 1023, 53, 729],
                    [815, 45, 299, 227, 814, 851, 422, 589],
                    [317, 545, 817, 670, 814, 68, 275, 10],
                    [708, 204, 817, 138, 540, 781, 624, 943],
                    [526, 136, 817, 444, 348, 95, 248, 193],
                    [15, 681, 497, 68, 550, 788, 294, 989],
                    [412, 229, 567, 478, 196, 1021, 743, 458],
                    [444, 933, 265, 984, 589, 168, 996, 727],
                    [165, 135, 56, 10, 253, 754, 349, 584],
                    [326, 795, 412, 663, 877, 168, 905, 925],
                    [1013, 860, 800, 520, 128, 20, 472, 651],
                    [355, 434, 299, 85, 891, 626, 272, 80],
                    [150, 70, 1016, 72, 819, 521, 670, 536],
                    [150, 623, 621, 541, 577, 763, 505, 906],
                    [517, 944, 586, 207, 147, 248, 843, 243],
                    [907, 77, 726, 75, 745, 746, 620, 653]
                ]
            ]
        )
        # fmt: on

        torch.testing.assert_close(outputs.cpu(), EXPECTED_OUTPUT_TOKENS)
