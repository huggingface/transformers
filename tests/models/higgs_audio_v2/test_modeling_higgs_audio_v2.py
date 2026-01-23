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
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Generate audio following instruction."}],
            },
            {"role": "scene", "content": [{"type": "text", "text": "Audio is recorded from a quiet room."}]},
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
            conversation, tokenize=True, return_dict=True, sampling_rate=24000, return_tensors="pt"
        )
        inputs = inputs.to(torch_device)

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

        EXPECTED_OUTPUT_TOKENS = torch.tensor()
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
        )
        inputs = inputs.to(torch_device)

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

        EXPECTED_OUTPUT_TOKENS = torch.tensor()
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
        )
        inputs = inputs.to(torch_device)

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

        EXPECTED_OUTPUT_TOKENS = torch.tensor()
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
        )
        inputs = inputs.to(torch_device)

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

        EXPECTED_OUTPUT_TOKENS = torch.tensor()
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
        )
        inputs = inputs.to(torch_device)

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

        EXPECTED_OUTPUT_TOKENS = torch.tensor()
        torch.testing.assert_close(outputs.cpu(), EXPECTED_OUTPUT_TOKENS)
