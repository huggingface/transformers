# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import json
import tempfile
import unittest
from pathlib import Path

import pytest
from parameterized import parameterized

from transformers import (
    VibeVoiceAsrConfig,
    VibeVoiceAsrForConditionalGeneration,
    is_datasets_available,
    is_torch_available,
)
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)
from transformers.trainer_utils import set_seed

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_datasets_available():
    from datasets import Audio, load_dataset

if is_torch_available():
    import torch


class VibeVoiceAsrModelTester:
    """
    Builds a tiny VibeVoice ASR config and synthetic inputs for testing.
    """

    def __init__(
        self,
        parent,
        audio_token_id=0,
        seq_length=25,
        audio_samples=24000,  # 1 second at 24kHz
        text_config={
            "model_type": "qwen2",
            "intermediate_size": 36,
            "initializer_range": 0.02,
            "hidden_size": 32,
            "max_position_embeddings": 52,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": 99,
            "pad_token_id": 1,  # Ensure pad token != audio token
        },
        acoustic_tokenizer_encoder_config={
            "model_type": "vibevoice_acoustic_tokenizer_encoder",
            "hidden_size": 16,
            "kernel_size": 3,
            "n_filters": 4,
            "downsampling_ratios": [2],
            "depths": [1, 1],
        },
        semantic_tokenizer_encoder_config={
            "model_type": "vibevoice_acoustic_tokenizer_encoder",
            "channels": 1,
            "hidden_size": 32,  # 2x acoustic hidden size
            "kernel_size": 3,
            "n_filters": 4,
            "downsampling_ratios": [2],
            "depths": [1, 1],
        },
        is_training=True,
    ):
        self.parent = parent
        self.audio_token_id = audio_token_id
        self.seq_length = seq_length
        self.audio_samples = audio_samples
        self.is_training = is_training
        self.text_config = text_config
        self.acoustic_tokenizer_encoder_config = acoustic_tokenizer_encoder_config
        self.semantic_tokenizer_encoder_config = semantic_tokenizer_encoder_config
        self.batch_size = 2
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.encoder_seq_length = seq_length

    def get_config(self):
        return VibeVoiceAsrConfig(
            acoustic_tokenizer_encoder_config=self.acoustic_tokenizer_encoder_config,
            semantic_tokenizer_encoder_config=self.semantic_tokenizer_encoder_config,
            text_config=self.text_config,
            audio_token_id=self.audio_token_id,
        )

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.text_config["vocab_size"])
        attention_mask = torch.ones([self.batch_size, self.seq_length], dtype=torch.long, device=torch_device)
        input_values = floats_tensor([self.batch_size, 1, self.audio_samples])
        padding_mask = torch.ones([self.batch_size, self.audio_samples], dtype=torch.bool, device=torch_device)
        config = self.get_config()
        return config, input_ids, attention_mask, input_values, padding_mask

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, input_values, padding_mask = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_values": input_values,
            "padding_mask": padding_mask,
        }

        return config, inputs_dict


@require_torch
class VibeVoiceAsrForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (VibeVoiceAsrForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"audio-text-to-text": VibeVoiceAsrForConditionalGeneration} if is_torch_available() else {}
    )
    _is_composite = True

    def setUp(self):
        self.model_tester = VibeVoiceAsrModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VibeVoiceAsrConfig, has_text_modality=False)

    @unittest.skip(
        reason="This test does not apply to VibeVoiceAsr since inputs_embeds corresponding to audio tokens are replaced when input features are provided."
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="Compile not yet supported for VibeVoiceAsr models")
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Compile not yet supported for VibeVoiceAsr models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="VibeVoiceAsr tests avoid right-padding equivalence; fusion is in-place.")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @unittest.skip(reason="VibeVoiceAsr has no separate base model without a head.")
    def test_model_base_model_prefix(self):
        pass

    @unittest.skip(reason="VibeVoiceAsr audio components do not use attention.")
    def test_get_audio_features_attentions(self):
        pass

    @unittest.skip(reason="VibeVoiceAsr has unique audio processing with acoustic and semantic tokenizers.")
    def test_get_audio_features_hidden_states(self):
        pass

    @unittest.skip(reason="VibeVoiceAsr has slight randomness due to VAE sampling.")
    def test_determinism(self):
        pass

    @unittest.skip(reason="VibeVoiceAsr has slight randomness due to VAE sampling.")
    def test_batching_equivalence(self):
        pass

    @unittest.skip(reason="VibeVoiceAsr has slight randomness due to VAE sampling.")
    def test_save_load(self):
        pass

    @unittest.skip(reason="VibeVoiceAsr has slight randomness due to VAE sampling.")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip(reason="VibeVoiceAsr has slight randomness due to VAE sampling.")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="VibeVoiceAsr has slight randomness due to VAE sampling.")
    def test_left_padding_compatibility(self):
        pass

    def test_sdpa_can_dispatch_composite_models(self):
        # VibeVoiceAsr is audio+text composite; but audio components do not use attention
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                # SDPA (default)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                text_attn = "sdpa" if model.language_model._supports_sdpa else "eager"

                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
                self.assertTrue(model.language_model.config._attn_implementation == text_attn)

                # Eager
                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(model_eager.config._attn_implementation == "eager")
                self.assertTrue(model_eager.language_model.config._attn_implementation == "eager")

                for _, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        raise ValueError("The eager model should not have SDPA attention layers")

    @parameterized.expand([True, False, None])
    def test_get_audio_features_output(self, return_dict: bool | None):
        for model_class in self.all_model_classes:
            config, inputs_dict = self._audio_features_prepare_config_and_inputs()
            if return_dict is not None:
                config.return_dict = return_dict

            model = model_class(config).eval()
            model = model.to(torch_device)

            torch.manual_seed(0)
            with torch.no_grad():
                outputs = model.get_audio_features(**inputs_dict)

            if return_dict in (True, None):
                last_hidden_state_shape = outputs.last_hidden_state.shape
                batch_size = inputs_dict["input_values"].shape[0]
                self.assertEqual(
                    last_hidden_state_shape[0],
                    batch_size,
                    f"batch_size mismatch, full shape: {last_hidden_state_shape}",
                )
                audio_config = config.acoustic_tokenizer_encoder_config
                hidden_size = audio_config.hidden_size
                self.assertEqual(
                    last_hidden_state_shape[-1],
                    hidden_size,
                    f"hidden_size mismatch, full shape: {last_hidden_state_shape}",
                )

            else:
                self.assertIsInstance(outputs, tuple, "get_audio_features() must return a tuple if return_dict=False")


@require_torch
class VibeVoiceAsrForConditionalGenerationIntegrationTest(unittest.TestCase):
    _dataset = None

    @classmethod
    def setUp(cls):
        from transformers import AutoProcessor
        from transformers.testing_utils import cleanup

        cleanup(torch_device, gc_collect=True)
        cls.checkpoint = "microsoft/VibeVoice-ASR-HF"
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint)

    def tearDown(self):
        from transformers.testing_utils import cleanup

        cleanup(torch_device, gc_collect=True)

    @classmethod
    def _load_dataset(cls):
        # Lazy loading of the dataset. Because it is a class method, it will only be loaded once per pytest process.
        if cls._dataset is None:
            cls._dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
            cls._dataset = cls._dataset.cast_column(
                "audio", Audio(sampling_rate=cls.processor.feature_extractor.sampling_rate)
            )

    def _load_datasamples(self, num_samples):
        self._load_dataset()
        ds = self._dataset
        speech_samples = ds.sort("id")[:num_samples]["audio"]
        return [x["array"] for x in speech_samples]

    @slow
    def test_single(self):
        """
        reproducer: https://gist.github.com/ebezzam/e1200bcecdc29e87dadd9d8423ae7ecb#file-reproducer_vibevoice_asr-py
        """
        set_seed(42)

        path = Path(__file__).parent.parent.parent / "fixtures/vibevoice_asr/expected_results_single.json"
        with open(path, "r", encoding="utf-8") as f:
            expected_outputs = json.load(f)

        samples = self._load_datasamples(1)
        conversation = [{"role": "user", "content": [{"type": "audio", "audio": samples[0]}]}]

        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map=torch_device, dtype=torch.bfloat16
        )

        inputs = self.processor.apply_chat_template(conversation, tokenize=True, return_dict=True).to(
            model.device, dtype=model.dtype
        )
        torch.testing.assert_close(inputs["input_ids"].cpu(), torch.tensor(expected_outputs["input_ids"]))

        output = model.generate(**inputs)
        gen_ids = output[:, inputs["input_ids"].shape[1] :]
        torch.testing.assert_close(gen_ids.cpu(), torch.tensor(expected_outputs["generated_ids"]))
        txt = self.processor.decode(gen_ids, skip_special_tokens=True)
        self.assertListEqual(txt, expected_outputs["transcriptions"])

    @slow
    def test_batch(self):
        """
        reproducer: https://gist.github.com/ebezzam/e1200bcecdc29e87dadd9d8423ae7ecb#file-reproducer_vibevoice_asr_batch-py
        """
        set_seed(42)

        path = Path(__file__).parent.parent.parent / "fixtures/vibevoice_asr/expected_results_batch.json"
        with open(path, "r", encoding="utf-8") as f:
            expected_outputs = json.load(f)

        samples = self._load_datasamples(2)
        conversation = [
            [{"role": "user", "content": [{"type": "audio", "audio": samples[0]}]}],
            [{"role": "user", "content": [{"type": "audio", "audio": samples[1]}]}],
        ]

        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map=torch_device, dtype=torch.bfloat16
        )
        inputs = self.processor.apply_chat_template(conversation, tokenize=True, return_dict=True).to(
            model.device, dtype=model.dtype
        )

        output = model.generate(**inputs)
        gen_ids = output[:, inputs["input_ids"].shape[1] :]
        for i, exp_gen in enumerate(expected_outputs["generated_ids"]):
            actual_gen = gen_ids[i, : len(exp_gen)]
            torch.testing.assert_close(actual_gen.cpu(), torch.tensor(exp_gen))
        txt = self.processor.decode(gen_ids, skip_special_tokens=True)
        self.assertListEqual(txt, expected_outputs["transcriptions"])

    @slow
    def test_single_with_context(self):
        """
        reproducer: tests/models/vibevoice_asr/reproducer_vibevoice_asr_with_context.py
        """
        set_seed(42)

        path = Path(__file__).parent.parent.parent / "fixtures/vibevoice_asr/expected_results_with_context.json"
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "About VibeVoice",
                    },
                    {
                        "type": "audio",
                        "path": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/realtime_model/vibevoice_tts_german.wav",
                    },
                ],
            }
        ]

        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map=torch_device, dtype=torch.bfloat16
        )

        inputs = self.processor.apply_chat_template(conversation, tokenize=True, return_dict=True).to(
            model.device, dtype=model.dtype
        )
        torch.testing.assert_close(inputs["input_ids"].cpu(), torch.tensor(raw["input_ids"]))

        output = model.generate(**inputs)
        gen_ids = output[:, inputs["input_ids"].shape[1] :]
        torch.testing.assert_close(gen_ids.cpu(), torch.tensor(raw["generated_ids"]))
        txt = self.processor.decode(gen_ids, skip_special_tokens=True)
        self.assertListEqual(txt, raw["transcriptions"])
