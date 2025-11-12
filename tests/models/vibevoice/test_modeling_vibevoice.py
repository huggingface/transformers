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
from transformers.trainer_utils import set_seed

import pytest

from transformers import (
    AutoProcessor,
    VibeVoiceForConditionalGeneration,
    VibeVoiceConfig,
    is_torch_available,
)
from transformers.audio_utils import load_audio_librosa
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
    ids_tensor,
)


if is_datasets_available():
    from datasets import load_dataset
    from huggingface_hub import snapshot_download


if is_torch_available():
    import torch


# TODO (ebezzam) best way to do this?
# if is_diffusers_available():
import diffusers


class VibeVoiceModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        is_training=True,
        use_cache=True,
        text_config={
            "model_type": "qwen2",
            "vocab_size": 100,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_act": "silu",
            "max_position_embeddings": 512,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "bos_token_id": 2,
            "tie_word_embeddings": False,
        },
        acoustic_tokenizer_config={
            "model_type": "vibevoice_acoustic_tokenizer", 
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
        },
        semantic_tokenizer_config={
            "model_type": "vibevoice_semantic_tokenizer",
            "hidden_size": 32, 
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
        },
        diffusion_head_config={
            "model_type": "vibevoice_diffusion_head",
            "hidden_size": 64,
            "num_head_layers": 2,
            "head_ffn_ratio": 3,
            "latent_size": 32,
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
        self.diffusion_head_config = diffusion_head_config

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
            diffusion_head_config=self.diffusion_head_config,
            use_cache=self.use_cache,
            pad_token_id=self.text_config["pad_token_id"],
            eos_token_id=self.text_config["eos_token_id"],
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        # For VibeVoice, we need text input_ids (not codebook-style)
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
        self.parent.assertEqual(
            result.logits.shape, 
            (self.batch_size, self.seq_length, self.vocab_size)
        )


class VibeVoiceForConditionalGenerationTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (VibeVoiceForConditionalGeneration,) if is_torch_available() else ()

    test_resize_embeddings = False
    test_resize_embeddings_untied = False

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

    # Skip tests that are not applicable to VibeVoice
    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation requires specific audio/text setup.")
    def test_assisted_decoding_matches_greedy_search(self):
        pass

    @pytest.mark.generate  
    @unittest.skip(reason="VibeVoice generation requires specific audio/text setup.")
    def test_assisted_decoding_sample(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation requires specific audio/text setup.")
    def test_beam_sample_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation requires specific audio/text setup.")
    def test_beam_search_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation requires specific audio/text setup.")
    def test_beam_search_generate_dict_output(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation requires specific audio/text setup.")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation requires specific audio/text setup.")
    def test_beam_sample_generate_dict_output(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation requires specific audio/text setup.")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="VibeVoice generation requires specific audio/text setup.")
    def test_prompt_lookup_decoding_stops_at_eos(self):
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

    @unittest.skip(reason="VibeVoice has composite model structure.")
    def test_tied_weights_keys(self):
        pass


@require_read_token
class VibeVoiceForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_checkpoint = "bezzam/VibeVoice-1.5B"
        self.sampling_rate = 24000

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_1b5_inference(self):
        """
        reproducer that generates JSON with expected output: https://gist.github.com/ebezzam/507dfd544e0a0f12402966503cbc73e6#file-reproducer-py
        standalone script for this test: https://gist.github.com/ebezzam/507dfd544e0a0f12402966503cbc73e6#file-test_integration_single-py
        """
        set_seed(42)
        fixtures_path = Path(__file__).parent.parent.parent / "fixtures/vibevoice/expected_results_single.json"
        example_files_repo = "bezzam/vibevoice_samples"
        audio_fn = ["voices/en-Alice_woman.wav", "voices/en-Frank_man.wav"]
        max_new_tokens = 32
        cfg_scale = 1.3

        # Load model and processor
        model = VibeVoiceForConditionalGeneration.from_pretrained(
            self.model_checkpoint,
            dtype=torch.float32,
            device_map=torch_device,
        ).eval()
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)

        # Prepare inputs
        repo_dir = snapshot_download(repo_id=example_files_repo, repo_type="dataset")
        audio_paths = [f"{repo_dir}/{fn}" for fn in audio_fn]
        conversation = [
            {"role": "0", "content": [
                {"type": "text", "text": "Hello everyone, and welcome to the VibeVoice podcast. I'm your host, Linda, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Thomas here to talk about it with me."},
                {"type": "audio", "path": load_audio_librosa(audio_paths[0], sampling_rate=self.sampling_rate)}
            ]},
            {"role": "1", "content": [
                {"type": "text", "text": "Thanks so much for having me, Linda. You're absolutely right—this question always brings out some seriously strong feelings."},
                {"type": "audio", "path": load_audio_librosa(audio_paths[1], sampling_rate=self.sampling_rate)}
            ]},
        ]
        inputs = processor.apply_chat_template(
            conversation, 
            tokenize=True,
            return_dict=True
        ).to(torch_device, dtype=next(model.parameters()).dtype)

        # Generate audio
        noise_scheduler = getattr(diffusers, model.generation_config.noise_scheduler)(
            **model.generation_config.noise_scheduler_config
        )
        generated_speech = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            cfg_scale=cfg_scale,
            do_sample=False,
            noise_scheduler=noise_scheduler,
            return_dict_in_generate=False,
        )[0].cpu().float()

        # Compare against expected results
        with open(fixtures_path, "r") as f:
            expected_results = json.load(f)
        expected_speech = torch.tensor(expected_results["speech_outputs"])
        generated_speech = generated_speech[..., :expected_speech.shape[-1]]
        torch.testing.assert_close(generated_speech, expected_speech, rtol=1e-5, atol=1e-5)
