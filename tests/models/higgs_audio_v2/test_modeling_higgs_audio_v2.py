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
    Expectations,
    cleanup,
    require_deterministic_for_xpu,
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

    def _get_logits_processor_kwargs(self, do_sample=False, config=None):
        # The base `GenerationTesterMixin` adds logits processors (e.g. repetition_penalty, bad_words_ids)
        # that index into scores using `input_ids`. HiggsAudioV2 outputs audio codebook logits with shape
        # (batch, num_codebooks, codebook_size), so those processors cause index-out-of-bounds errors.
        # Override to return an empty dict (or only sampling params) to skip incompatible processors.
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

    @pytest.mark.skip(
        reason="This test does not apply to HiggsAudioV2 since audio_input_ids must be provided along input_ids"
    )
    def test_flash_attention_2_continue_generate_with_position_ids(self):
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
    @require_deterministic_for_xpu
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
        EXPECTED_OUTPUT_TOKENS = Expectations(
            {
                ("xpu", 3): torch.tensor(
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
                            [431, 219, 568, 999, 126, 83, 677, 260],
                            [852, 797, 915, 809, 270, 318, 201, 962],
                            [988, 280, 179, 591, 647, 500, 862, 790],
                            [988, 673, 75, 651, 879, 931, 801, 770],
                            [945, 896, 102, 338, 539, 219, 100, 880],
                            [287, 728, 555, 927, 377, 357, 335, 794],
                            [300, 84, 684, 159, 377, 376, 1022, 786],
                            [63, 618, 68, 645, 667, 831, 256, 918],
                            [707, 936, 104, 198, 667, 191, 746, 522],
                            [292, 942, 302, 756, 341, 555, 979, 243],
                            [822, 138, 814, 919, 202, 413, 891, 879],
                            [300, 614, 466, 519, 852, 340, 213, 103],
                            [682, 307, 762, 882, 954, 365, 249, 375],
                            [473, 550, 1000, 357, 205, 692, 248, 473],
                            [196, 550, 587, 779, 207, 583, 292, 637],
                            [343, 962, 784, 514, 505, 687, 328, 787],
                            [473, 227, 478, 296, 83, 851, 882, 204],
                            [300, 984, 478, 997, 913, 5, 422, 786],
                            [343, 697, 325, 995, 532, 78, 827, 63],
                            [409, 225, 812, 92, 276, 347, 509, 431],
                            [300, 829, 949, 797, 40, 862, 53, 693],
                            [1013, 486, 794, 69, 53, 927, 801, 744],
                            [740, 728, 784, 386, 1016, 515, 966, 978],
                            [740, 543, 231, 68, 715, 641, 426, 222],
                            [300, 284, 68, 429, 452, 611, 965, 630],
                            [265, 305, 450, 820, 423, 979, 135, 567],
                            [265, 834, 221, 611, 401, 72, 607, 428],
                            [758, 834, 8, 126, 73, 901, 947, 202],
                            [1013, 526, 97, 969, 56, 415, 895, 536],
                            [740, 69, 490, 768, 393, 983, 359, 432],
                            [787, 153, 46, 357, 731, 473, 619, 176],
                            [773, 560, 784, 536, 393, 1009, 387, 200],
                            [338, 865, 398, 732, 106, 5, 662, 187],
                            [360, 829, 490, 536, 138, 434, 997, 209],
                            [490, 367, 271, 849, 435, 688, 288, 998],
                            [95, 405, 242, 136, 42, 933, 714, 88],
                            [645, 264, 191, 547, 197, 473, 603, 545],
                            [360, 829, 324, 608, 246, 890, 561, 235],
                            [283, 464, 49, 665, 930, 108, 123, 632],
                        ]
                    ]
                ),
                (None, None): torch.tensor(
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
                            [360, 393, 816, 317, 271, 488, 233, 60],
                        ]
                    ]
                ),
            }
        ).get_expectation()
        # fmt: on

        torch.testing.assert_close(outputs.cpu(), EXPECTED_OUTPUT_TOKENS)

    @slow
    @require_torch_accelerator
    @require_deterministic_for_xpu
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
        EXPECTED_OUTPUT_TOKENS = Expectations(
            {
                ("xpu", 3): torch.tensor(
                    [
                        [
                            [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                            [244, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                            [171, 537, 1024, 1024, 1024, 1024, 1024, 1024],
                            [744, 908, 977, 1024, 1024, 1024, 1024, 1024],
                            [49, 923, 812, 485, 1024, 1024, 1024, 1024],
                            [360, 649, 522, 137, 324, 1024, 1024, 1024],
                            [715, 304, 991, 776, 101, 325, 1024, 1024],
                            [940, 758, 522, 928, 59, 491, 817, 1024],
                            [464, 567, 867, 505, 799, 762, 829, 669],
                            [464, 842, 187, 300, 73, 832, 557, 12],
                            [49, 225, 958, 864, 493, 794, 482, 414],
                            [208, 722, 370, 336, 828, 57, 488, 964],
                            [918, 454, 828, 350, 712, 384, 260, 372],
                            [943, 63, 835, 735, 694, 886, 128, 905],
                            [470, 84, 731, 955, 174, 942, 154, 283],
                            [646, 270, 285, 484, 202, 344, 185, 666],
                            [646, 114, 302, 111, 569, 778, 422, 308],
                            [470, 348, 867, 748, 194, 68, 445, 870],
                            [821, 471, 190, 283, 266, 63, 133, 475],
                            [320, 583, 115, 249, 600, 326, 987, 402],
                            [995, 960, 278, 179, 990, 779, 489, 608],
                            [30, 467, 555, 386, 651, 252, 239, 474],
                            [995, 527, 440, 62, 746, 550, 392, 202],
                            [208, 989, 18, 191, 409, 690, 695, 155],
                            [968, 990, 187, 116, 561, 407, 268, 730],
                            [431, 1016, 429, 802, 739, 12, 790, 712],
                            [21, 194, 867, 997, 631, 418, 499, 466],
                            [821, 606, 144, 352, 851, 501, 1017, 911],
                            [208, 874, 190, 674, 608, 24, 181, 176],
                            [499, 558, 522, 748, 390, 240, 853, 411],
                            [273, 291, 398, 404, 198, 314, 349, 111],
                            [667, 471, 797, 771, 713, 533, 360, 56],
                            [400, 989, 169, 479, 457, 772, 741, 538],
                            [281, 582, 279, 501, 852, 63, 970, 1002],
                            [139, 990, 148, 73, 531, 759, 551, 628],
                            [572, 732, 312, 62, 885, 797, 301, 796],
                            [318, 313, 166, 367, 3, 972, 174, 897],
                            [667, 146, 440, 267, 719, 590, 997, 68],
                            [996, 313, 984, 191, 157, 235, 26, 163],
                            [667, 356, 522, 590, 941, 384, 138, 217],
                            [944, 299, 932, 592, 595, 568, 661, 212],
                            [728, 427, 283, 757, 286, 441, 900, 145],
                            [995, 960, 522, 295, 584, 71, 993, 331],
                            [30, 39, 762, 495, 180, 332, 834, 123],
                            [642, 11, 215, 26, 375, 656, 399, 971],
                            [318, 203, 420, 695, 281, 932, 191, 886],
                            [1000, 953, 701, 356, 281, 961, 916, 680],
                            [725, 300, 867, 938, 190, 924, 820, 30],
                            [48, 779, 18, 838, 527, 374, 196, 1018],
                            [171, 268, 166, 143, 606, 55, 816, 142],
                        ]
                    ]
                ),
                (None, None): torch.tensor(
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
                            [65, 64, 835, 623, 1009, 548, 568, 746],
                        ]
                    ]
                ),
            }
        ).get_expectation()
        # fmt: on

        torch.testing.assert_close(outputs.cpu(), EXPECTED_OUTPUT_TOKENS)

    @slow
    @require_deterministic_for_xpu
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
        EXPECTED_OUTPUT_TOKENS = Expectations(
            {
                ("xpu", 3): torch.tensor(
                    [
                        [
                            [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                            [800, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                            [800, 74, 1024, 1024, 1024, 1024, 1024, 1024],
                            [800, 74, 427, 1024, 1024, 1024, 1024, 1024],
                            [800, 202, 427, 467, 1024, 1024, 1024, 1024],
                            [800, 74, 468, 467, 433, 1024, 1024, 1024],
                            [800, 74, 468, 865, 433, 552, 1024, 1024],
                            [244, 74, 427, 351, 433, 764, 926, 1024],
                            [244, 528, 468, 467, 433, 552, 926, 419],
                            [800, 354, 553, 351, 421, 764, 44, 858],
                            [321, 74, 553, 550, 700, 904, 83, 419],
                            [321, 88, 553, 880, 799, 653, 926, 858],
                            [560, 730, 136, 161, 843, 764, 926, 669],
                            [1012, 288, 862, 961, 362, 552, 926, 858],
                            [752, 196, 191, 360, 799, 904, 581, 696],
                            [752, 917, 955, 375, 588, 243, 233, 858],
                            [114, 540, 1003, 312, 198, 552, 986, 669],
                            [299, 203, 986, 880, 250, 537, 670, 981],
                            [414, 152, 286, 51, 403, 596, 561, 708],
                            [483, 223, 545, 164, 9, 703, 474, 791],
                            [400, 444, 140, 779, 895, 25, 760, 949],
                            [572, 435, 256, 933, 374, 211, 505, 36],
                            [852, 849, 321, 499, 635, 883, 69, 56],
                            [9, 282, 1003, 521, 28, 907, 41, 280],
                            [808, 282, 204, 771, 1008, 990, 1011, 896],
                            [297, 473, 848, 356, 888, 458, 49, 913],
                            [760, 118, 986, 1004, 629, 678, 916, 717],
                            [135, 968, 394, 877, 260, 733, 116, 129],
                            [993, 444, 515, 567, 880, 573, 630, 451],
                            [483, 59, 642, 659, 899, 949, 842, 725],
                            [414, 2, 336, 61, 454, 483, 527, 845],
                            [588, 867, 380, 326, 374, 930, 578, 69],
                            [572, 702, 44, 637, 873, 686, 371, 735],
                            [114, 351, 560, 874, 120, 431, 463, 965],
                            [71, 553, 782, 406, 254, 886, 593, 767],
                            [9, 278, 123, 129, 878, 821, 637, 822],
                            [373, 132, 373, 207, 329, 710, 211, 454],
                            [114, 1001, 871, 859, 1015, 294, 177, 229],
                            [239, 498, 337, 1007, 534, 457, 485, 768],
                            [568, 910, 91, 669, 191, 925, 252, 888],
                            [356, 452, 930, 973, 778, 471, 95, 714],
                            [834, 911, 930, 207, 923, 212, 679, 418],
                            [725, 473, 4, 353, 120, 703, 309, 958],
                            [588, 653, 138, 316, 1020, 902, 792, 387],
                            [386, 353, 711, 661, 969, 879, 229, 388],
                            [211, 793, 123, 327, 193, 651, 466, 626],
                            [386, 12, 910, 382, 457, 561, 617, 834],
                            [534, 583, 53, 281, 943, 386, 740, 593],
                            [330, 492, 884, 346, 872, 440, 276, 11],
                            [330, 492, 479, 169, 129, 842, 930, 367],
                        ]
                    ]
                ),
                (None, None): torch.tensor(
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
                            [330, 57, 903, 957, 49, 309, 992, 187],
                        ]
                    ]
                ),
            }
        ).get_expectation()
        # fmt: on

        torch.testing.assert_close(outputs.cpu(), EXPECTED_OUTPUT_TOKENS)

    @slow
    @require_deterministic_for_xpu
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
        EXPECTED_OUTPUT_TOKENS = Expectations(
            {
                ("xpu", 3): torch.tensor(
                    [
                        [
                            [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                            [633, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                            [242, 269, 1024, 1024, 1024, 1024, 1024, 1024],
                            [703, 172, 715, 1024, 1024, 1024, 1024, 1024],
                            [703, 468, 386, 309, 1024, 1024, 1024, 1024],
                            [891, 468, 508, 224, 926, 1024, 1024, 1024],
                            [242, 184, 508, 46, 272, 664, 1024, 1024],
                            [846, 337, 446, 741, 117, 654, 623, 1024],
                            [749, 389, 886, 100, 117, 1021, 664, 485],
                            [846, 442, 886, 841, 469, 833, 440, 943],
                            [749, 442, 816, 865, 252, 833, 831, 962],
                            [749, 537, 287, 675, 515, 243, 719, 390],
                            [83, 537, 65, 550, 916, 243, 532, 578],
                            [703, 947, 981, 161, 609, 653, 787, 562],
                            [160, 229, 375, 51, 916, 653, 986, 409],
                            [721, 122, 231, 969, 588, 890, 849, 981],
                            [888, 134, 2, 734, 962, 552, 417, 981],
                            [737, 513, 139, 378, 404, 738, 451, 229],
                            [390, 127, 367, 732, 132, 246, 201, 229],
                            [831, 186, 854, 414, 631, 727, 552, 1023],
                            [296, 69, 628, 254, 824, 269, 826, 424],
                            [874, 492, 204, 187, 159, 891, 358, 189],
                            [25, 101, 848, 242, 579, 904, 465, 214],
                            [936, 675, 708, 640, 773, 258, 15, 991],
                            [12, 321, 158, 892, 63, 617, 475, 1020],
                            [12, 2, 973, 413, 530, 994, 154, 615],
                            [749, 928, 848, 81, 75, 786, 1004, 606],
                            [401, 1001, 80, 143, 280, 833, 223, 168],
                            [30, 1017, 720, 105, 78, 190, 11, 208],
                            [879, 727, 666, 113, 771, 326, 132, 497],
                            [182, 938, 573, 321, 582, 326, 509, 557],
                            [106, 419, 810, 785, 78, 902, 408, 328],
                            [330, 554, 826, 358, 352, 157, 794, 722],
                            [286, 220, 711, 25, 523, 96, 313, 624],
                            [998, 492, 164, 82, 821, 550, 737, 73],
                            [12, 818, 164, 81, 290, 180, 635, 241],
                            [796, 811, 481, 1015, 1019, 33, 104, 281],
                            [179, 99, 277, 200, 582, 776, 132, 459],
                            [1012, 892, 162, 1018, 808, 944, 144, 394],
                            [652, 811, 945, 659, 90, 763, 239, 532],
                            [44, 988, 573, 0, 360, 315, 128, 113],
                            [254, 361, 1003, 998, 183, 781, 960, 887],
                            [672, 718, 969, 163, 76, 860, 728, 188],
                            [692, 333, 609, 141, 51, 105, 158, 257],
                            [413, 533, 277, 571, 877, 439, 246, 809],
                            [307, 232, 737, 969, 302, 810, 43, 52],
                            [137, 521, 139, 813, 91, 584, 276, 571],
                            [437, 96, 956, 1012, 997, 6, 911, 953],
                            [986, 642, 80, 617, 22, 601, 266, 172],
                            [116, 79, 1003, 565, 230, 950, 520, 65],
                        ]
                    ]
                ),
                (None, None): torch.tensor(
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
                            [12, 164, 383, 979, 229, 865, 860, 325],
                        ]
                    ]
                ),
            }
        ).get_expectation()
        # fmt: on

        torch.testing.assert_close(outputs.cpu(), EXPECTED_OUTPUT_TOKENS)

    @slow
    @require_deterministic_for_xpu
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
        EXPECTED_OUTPUT_TOKENS = Expectations(
            {
                ("xpu", 3): torch.tensor(
                    [
                        [
                            [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                            [800, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                            [800, 74, 1024, 1024, 1024, 1024, 1024, 1024],
                            [800, 74, 427, 1024, 1024, 1024, 1024, 1024],
                            [800, 202, 427, 467, 1024, 1024, 1024, 1024],
                            [800, 74, 468, 467, 433, 1024, 1024, 1024],
                            [800, 74, 468, 351, 433, 552, 1024, 1024],
                            [244, 74, 427, 351, 433, 764, 926, 1024],
                            [244, 528, 468, 467, 433, 764, 926, 419],
                            [800, 354, 553, 467, 433, 552, 233, 858],
                            [321, 74, 553, 550, 433, 764, 83, 390],
                            [321, 88, 553, 351, 799, 904, 926, 858],
                            [817, 730, 136, 467, 433, 552, 926, 151],
                            [128, 408, 715, 961, 362, 904, 926, 669],
                            [791, 523, 816, 841, 799, 904, 926, 858],
                            [187, 396, 394, 704, 916, 653, 83, 419],
                            [752, 647, 140, 544, 649, 45, 579, 669],
                            [299, 203, 751, 199, 805, 626, 505, 981],
                            [299, 203, 91, 26, 1002, 505, 109, 390],
                            [650, 282, 91, 204, 335, 929, 924, 25],
                            [120, 440, 954, 260, 686, 278, 362, 749],
                            [400, 196, 373, 555, 122, 254, 874, 214],
                            [583, 440, 5, 296, 231, 809, 640, 562],
                            [945, 675, 124, 274, 921, 209, 31, 174],
                            [572, 282, 409, 442, 867, 869, 18, 931],
                            [808, 947, 514, 627, 426, 680, 532, 852],
                            [614, 238, 609, 26, 22, 458, 100, 949],
                            [177, 185, 615, 927, 470, 291, 662, 356],
                            [993, 299, 975, 877, 595, 114, 229, 998],
                            [176, 797, 858, 714, 304, 225, 341, 155],
                            [400, 559, 273, 691, 692, 551, 772, 326],
                            [270, 2, 124, 308, 479, 449, 1007, 357],
                            [668, 153, 554, 71, 931, 155, 176, 256],
                            [995, 12, 788, 282, 254, 262, 740, 681],
                            [114, 355, 843, 289, 701, 884, 416, 961],
                            [71, 1011, 587, 661, 961, 938, 264, 540],
                            [58, 278, 211, 842, 745, 456, 317, 755],
                            [431, 392, 524, 945, 64, 604, 1016, 912],
                            [114, 79, 924, 169, 285, 441, 183, 415],
                            [239, 717, 255, 74, 828, 671, 523, 1018],
                            [568, 903, 91, 148, 40, 304, 913, 996],
                            [199, 452, 1006, 902, 196, 763, 160, 76],
                            [834, 78, 630, 220, 409, 551, 102, 993],
                            [725, 278, 769, 962, 442, 671, 736, 991],
                            [288, 717, 446, 356, 1007, 533, 321, 405],
                            [881, 750, 983, 321, 814, 820, 567, 743],
                            [321, 403, 983, 823, 618, 494, 568, 221],
                            [644, 426, 34, 873, 358, 771, 374, 240],
                            [330, 424, 21, 965, 350, 221, 544, 920],
                            [418, 394, 433, 805, 321, 204, 565, 173],
                        ],
                        [
                            [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                            [244, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                            [451, 518, 1024, 1024, 1024, 1024, 1024, 1024],
                            [497, 478, 52, 1024, 1024, 1024, 1024, 1024],
                            [497, 503, 11, 841, 1024, 1024, 1024, 1024],
                            [533, 503, 468, 478, 799, 1024, 1024, 1024],
                            [497, 624, 270, 675, 283, 764, 1024, 1024],
                            [533, 1021, 270, 467, 609, 153, 579, 1024],
                            [497, 624, 265, 550, 362, 552, 53, 438],
                            [515, 1021, 65, 351, 609, 890, 426, 419],
                            [207, 415, 468, 550, 916, 75, 798, 1023],
                            [240, 415, 0, 351, 843, 270, 798, 1023],
                            [533, 21, 427, 880, 916, 890, 118, 727],
                            [207, 1021, 262, 584, 650, 342, 53, 229],
                            [497, 624, 468, 584, 984, 342, 742, 563],
                            [49, 893, 725, 293, 609, 270, 290, 229],
                            [6, 461, 265, 550, 968, 764, 233, 390],
                            [250, 533, 403, 184, 843, 75, 201, 719],
                            [766, 982, 265, 804, 700, 890, 798, 151],
                            [766, 536, 672, 875, 436, 826, 83, 867],
                            [928, 536, 318, 954, 724, 37, 986, 407],
                            [213, 518, 318, 741, 916, 270, 840, 229],
                            [692, 236, 392, 786, 372, 146, 999, 966],
                            [419, 996, 687, 692, 671, 278, 417, 438],
                            [40, 342, 518, 621, 921, 177, 726, 910],
                            [866, 342, 560, 410, 93, 53, 732, 435],
                            [515, 624, 571, 948, 901, 414, 53, 356],
                            [207, 127, 434, 404, 640, 421, 487, 398],
                            [207, 749, 649, 236, 397, 775, 622, 403],
                            [911, 236, 725, 396, 356, 610, 500, 964],
                            [497, 624, 607, 256, 482, 827, 245, 468],
                            [207, 1021, 345, 215, 733, 432, 426, 964],
                            [497, 1021, 265, 934, 984, 632, 581, 696],
                            [497, 21, 468, 584, 984, 75, 624, 821],
                            [497, 873, 265, 467, 843, 544, 677, 229],
                            [240, 873, 270, 351, 613, 342, 296, 229],
                            [497, 21, 468, 467, 916, 342, 83, 165],
                            [497, 1021, 262, 161, 916, 270, 83, 419],
                            [207, 983, 434, 675, 609, 270, 986, 419],
                            [403, 415, 270, 351, 609, 552, 22, 544],
                            [207, 503, 270, 161, 641, 414, 44, 340],
                            [666, 415, 427, 584, 984, 72, 798, 438],
                            [207, 1021, 270, 467, 984, 890, 53, 669],
                            [207, 914, 270, 584, 916, 890, 22, 669],
                            [240, 914, 427, 467, 823, 890, 233, 562],
                            [497, 21, 16, 934, 700, 342, 559, 70],
                            [207, 873, 262, 880, 799, 342, 485, 390],
                            [515, 415, 468, 865, 843, 243, 485, 719],
                            [207, 873, 66, 161, 362, 552, 296, 390],
                            [497, 415, 468, 584, 609, 414, 83, 566],
                        ],
                    ]
                ),
                (None, None): torch.tensor(
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
                            [644, 793, 1010, 784, 796, 362, 147, 663],
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
                            [907, 77, 726, 75, 745, 746, 620, 653],
                        ],
                    ]
                ),
            }
        ).get_expectation()
        # fmt: on

        torch.testing.assert_close(outputs.cpu(), EXPECTED_OUTPUT_TOKENS)
