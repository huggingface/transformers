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

import copy
import json
import unittest
from pathlib import Path

import pytest
from parameterized import parameterized

from transformers import (
    AutoProcessor,
    VibeVoiceConfig,
    VibeVoiceForConditionalGeneration,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_diffusers,
    slow,
    torch_device,
)
from transformers.trainer_utils import set_seed

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
)


if is_torch_available():
    import torch


class DummyNoiseScheduler:
    """A simple dummy noise scheduler for testing purposes."""

    def __init__(self):
        self.num_inference_steps = None
        self.timesteps = None

    def step(self, eps, timestep, sample):
        # Simple dummy step: just subtract a fraction of the noise estimate
        if self.num_inference_steps is None:
            step_size = 0.1
        else:
            step_size = 1.0 / self.num_inference_steps

        # Return an object with prev_sample attribute like real schedulers
        class StepOutput:
            def __init__(self, prev_sample):
                self.prev_sample = prev_sample

        prev_sample = sample - step_size * eps
        return StepOutput(prev_sample)

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        # Create timesteps as torch tensors going from high to low (typical for diffusion)
        self.timesteps = torch.linspace(1000, 1, num_inference_steps).long()


class VibeVoiceModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=3,
        is_training=True,
        use_cache=True,
        num_head_layers=2,
        text_config={
            "model_type": "qwen2",
            "intermediate_size": 36,
            "initializer_range": 0.02,
            "hidden_size": 32,
            "max_position_embeddings": 52,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "use_labels": True,
            "use_mrope": False,
            "vocab_size": 99,
            "pad_token_id": 0,
            "eos_token_id": 0,  # same as pad_token for Vibevoice
            "bos_token_id": None,
        },
        acoustic_tokenizer_config={
            "model_type": "vibevoice_acoustic_tokenizer",
            "hidden_size": 16,
            "kernel_size": 3,
            "n_filters": 4,
            "downsampling_ratios": [2],
            "depths": [1, 1],
        },
        semantic_tokenizer_config={
            "model_type": "vibevoice_semantic_tokenizer",
            "channels": 1,
            "hidden_size": 32,
            "kernel_size": 3,
            "n_filters": 4,
            "downsampling_ratios": [2],
            "depths": [1, 1],
        },
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_cache = use_cache
        self.text_config = text_config
        self.acoustic_tokenizer_config = acoustic_tokenizer_config
        self.semantic_tokenizer_config = semantic_tokenizer_config
        self.num_head_layers = num_head_layers

        # Extract common attributes for testing
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.pad_token_id = text_config["pad_token_id"]

    def get_config(self):
        return VibeVoiceConfig(
            text_config=self.text_config,
            acoustic_tokenizer_config=self.acoustic_tokenizer_config,
            semantic_tokenizer_config=self.semantic_tokenizer_config,
            num_head_layers=self.num_head_layers,
            use_cache=self.use_cache,
            pad_token_id=self.text_config["pad_token_id"],
            eos_token_id=self.text_config["eos_token_id"],
            # Use token IDs that exist in our test vocabulary (vocab_size=99)
            audio_bos_token_id=3,  # Instead of default 151652
            audio_eos_token_id=4,  # Instead of default 151653
            audio_diffusion_token_id=5,  # Instead of default 151654
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones([self.batch_size, self.seq_length], dtype=torch.long, device=torch_device)
        return config, input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict

    def create_and_check_model(self, config, input_ids, attention_mask):
        model = VibeVoiceForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            result = model(input_ids=input_ids, attention_mask=attention_mask)

        # Check that the model returns expected outputs
        self.parent.assertIsNotNone(result.logits)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))


class VibeVoiceForConditionalGenerationTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (VibeVoiceForConditionalGeneration,) if is_torch_available() else ()

    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = VibeVoiceModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VibeVoiceConfig, has_text_modality=True)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        """
        VibeVoice uses standard input format.
        """
        inputs_dict = copy.deepcopy(inputs_dict)

        if return_labels:
            inputs_dict["labels"] = torch.zeros(
                (
                    self.model_tester.batch_size,
                    self.model_tester.seq_length,
                ),
                dtype=torch.long,
                device=torch_device,
            )

        return inputs_dict

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        self.skipTest("VibeVoice generation has unique generation")

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation has unique generation")
    def test_assisted_decoding_sample(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation has unique generation")
    def test_beam_sample_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation has unique generation")
    def test_beam_search_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation has unique generation")
    def test_beam_search_generate_dict_output(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation has unique generation")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation has unique generation")
    def test_beam_sample_generate_dict_output(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation has unique generation")
    def test_eager_matches_sdpa_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation has unique generation")
    def test_generate_continue_from_past_key_values(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation has unique generation")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation has unique generation")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation has unique generation")
    def test_prompt_lookup_decoding_stops_at_eos(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice uses diffusion process instead of traditional token sampling")
    def test_sample_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice uses diffusion process instead of traditional token sampling")
    def test_sample_generate_dict_output(self):
        pass

    @pytest.mark.skip(reason="VibeVoice has composite model structure.")
    def test_model_get_set_embeddings(self):
        pass

    @pytest.mark.skip(reason="VibeVoice has composite model structure.")
    def test_tie_model_weights(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation requires specific audio/text setup.")
    def test_generate_from_inputs_embeds_1_beam_search(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation requires specific audio/text setup.")
    def test_model_parallel_beam_search(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation requires noise_scheduler parameter.")
    def test_generate_continue_from_inputs_embeds(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation requires noise_scheduler parameter.")
    def test_generate_from_random_inputs_embeds(self):
        pass

    @parameterized.expand([("greedy", 1), ("beam search", 2)])
    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation performs different type of generation (diffusion process).")
    def test_generate_from_inputs_embeds(self, _, num_beams):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation returns audio output, not text tokens.")
    def test_generate_methods_with_logits_to_keep(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation returns audio output, not standard token sequences.")
    def test_greedy_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation has attention dimension issues during generation.")
    def test_greedy_generate_dict_outputs(self):
        pass

    @unittest.skip(reason="VibeVoice has composite model structure.")
    def test_tied_weights_keys(self):
        pass

    @pytest.mark.generate
    def test_vibevoice_generate_max_new_tokens(self):
        """
        Test VibeVoice-specific generation to ensure sequences output has correct length.
        This test verifies that the returned sequences include the original input_ids
        plus the newly generated tokens as specified by max_new_tokens.
        """
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config, input_ids, attention_mask = config_and_inputs

        model = VibeVoiceForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()

        max_new_tokens = 5
        original_length = input_ids.shape[1]
        expected_length = original_length + max_new_tokens

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                noise_scheduler=DummyNoiseScheduler(),
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                cfg_scale=1.3,
                n_diffusion_steps=10,
            )
        self.assertIsNotNone(output.sequences)
        self.assertEqual(output.sequences.shape[0], self.model_tester.batch_size)
        self.assertEqual(output.sequences.shape[1], expected_length)
        torch.testing.assert_close(
            output.sequences[:, :original_length],
            input_ids,
            msg="Original input_ids should be preserved at the beginning of sequences",
        )
        self.assertIsNotNone(output.audio)
        self.assertEqual(len(output.audio), self.model_tester.batch_size)


class VibeVoiceForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_checkpoint = "bezzam/VibeVoice-1.5B"
        self.sampling_rate = 24000

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_diffusers
    def test_1b5_inference_no_voice(self):
        """
        Reproducer which generates JSON expected outputs for acoustic/semantic tokenizers and main model:
        https://gist.github.com/ebezzam/507dfd544e0a0f12402966503cbc73e6#file-reproducer-py
        diffusers library is needed (ran with `diffusers==0.35.2`)
        """
        set_seed(42)
        fixtures_path = Path(__file__).parent.parent.parent / "fixtures/vibevoice/expected_results_single_noaudio.json"
        max_new_tokens = 32

        # Load model and processor
        model = VibeVoiceForConditionalGeneration.from_pretrained(
            self.model_checkpoint,
            dtype=torch.float32,
            device_map=torch_device,
        ).eval()
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)

        # Prepare input
        conversation = [
            {
                "role": "0",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello everyone, and welcome to the VibeVoice podcast. I'm your host, Linda, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Thomas here to talk about it with me.",
                    },
                ],
            },
            {
                "role": "1",
                "content": [
                    {
                        "type": "text",
                        "text": "Thanks so much for having me, Linda. You're absolutely right—this question always brings out some seriously strong feelings.",
                    },
                ],
            },
        ]
        inputs = processor.apply_chat_template(conversation, tokenize=True, return_dict=True).to(
            torch_device, dtype=next(model.parameters()).dtype
        )

        # Generate audio
        generated_speech = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=False,
        )
        generated_speech = generated_speech[0].cpu().float()

        # Compare against expected results
        with open(fixtures_path, "r") as f:
            expected_results = json.load(f)
        expected_speech = torch.tensor(expected_results["speech_outputs"])
        generated_speech = generated_speech[..., : expected_speech.shape[-1]]
        torch.testing.assert_close(generated_speech, expected_speech, rtol=1e-5, atol=1e-5)

    @slow
    @require_diffusers
    def test_1b5_inference(self):
        """
        Reproducer which generates JSON expected outputs for acoustic/semantic tokenizers and main model:
        https://gist.github.com/ebezzam/507dfd544e0a0f12402966503cbc73e6#file-reproducer-py
        diffusers library is needed (ran with `diffusers==0.35.2`)
        """
        set_seed(42)
        fixtures_path = Path(__file__).parent.parent.parent / "fixtures/vibevoice/expected_results_single.json"
        max_new_tokens = 32

        # Load model and processor
        model = VibeVoiceForConditionalGeneration.from_pretrained(
            self.model_checkpoint,
            dtype=torch.float32,
            device_map=torch_device,
        ).eval()
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)

        # Prepare inputs
        conversation = [
            {
                "role": "0",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello everyone, and welcome to the VibeVoice podcast. I'm your host, Linda, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Thomas here to talk about it with me.",
                    },
                    {
                        "type": "audio",
                        "url": "https://hf.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav",
                    },
                ],
            },
            {
                "role": "1",
                "content": [
                    {
                        "type": "text",
                        "text": "Thanks so much for having me, Linda. You're absolutely right—this question always brings out some seriously strong feelings.",
                    },
                    {
                        "type": "audio",
                        "url": "https://hf.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Frank_man.wav",
                    },
                ],
            },
        ]
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True, sampling_rate=self.sampling_rate
        ).to(torch_device, dtype=next(model.parameters()).dtype)

        # Generate audio
        generated_speech = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=False,
        )
        generated_speech = generated_speech[0].cpu().float()

        # Compare against expected results
        with open(fixtures_path, "r") as f:
            expected_results = json.load(f)
        expected_speech = torch.tensor(expected_results["speech_outputs"])
        generated_speech = generated_speech[..., : expected_speech.shape[-1]]
        torch.testing.assert_close(generated_speech, expected_speech, rtol=1e-5, atol=1e-5)
