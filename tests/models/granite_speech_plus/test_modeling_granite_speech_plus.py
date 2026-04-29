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
"""Testing suite for the IBM Granite Speech Plus model."""

import unittest

from parameterized import parameterized

from transformers import AutoProcessor, GraniteSpeechPlusConfig, GraniteSpeechPlusForConditionalGeneration
from transformers.testing_utils import cleanup, require_torch, slow, torch_device
from transformers.utils import ModelOutput, is_datasets_available, is_torch_available

from ...test_configuration_common import ConfigTester
from ..granite_speech.test_modeling_granite_speech import (
    GraniteSpeechForConditionalGenerationModelTest as _GraniteSpeechModelTestBase,
)
from ..granite_speech.test_modeling_granite_speech import (
    GraniteSpeechForConditionalGenerationModelTester as _GraniteSpeechModelTesterBase,
)


if is_torch_available():
    import torch
if is_datasets_available():
    from datasets import load_dataset

from transformers import set_seed


class GraniteSpeechPlusForConditionalGenerationModelTester(_GraniteSpeechModelTesterBase):
    """
    Plus variant that exercises the ``encoder_hidden_layers`` concat path. The projector's
    ``encoder_hidden_size`` is scaled to match ``encoder_config.hidden_dim * (len(encoder_hidden_layers) + 1)``.
    """

    def __init__(self, parent, encoder_hidden_layers=(0,), **kwargs):
        projector_config = kwargs.pop(
            "projector_config",
            {
                "attention_probs_dropout_prob": 0.1,
                "cross_attention_frequency": 1,
                "encoder_hidden_size": 64,  # 32 (hidden_dim) * (1 intermediate + 1 last) = 64
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "hidden_size": 32,
                "initializer_range": 0.02,
                "intermediate_size": 256,
                "layer_norm_eps": 1e-12,
                "max_position_embeddings": 2048,
                "model_type": "blip_2_qformer",
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "use_qformer_text_input": False,
                "vocab_size": 30522,
            },
        )
        super().__init__(parent=parent, projector_config=projector_config, **kwargs)
        self.encoder_hidden_layers = list(encoder_hidden_layers)
        self.encoder_config["cat_hidden_layers"] = self.encoder_hidden_layers

    def get_config(self):
        return GraniteSpeechPlusConfig(
            encoder_config=self.encoder_config,
            text_config=self.text_config,
            projector_config=self.projector_config,
            audio_token_index=self.audio_token_index,
            tie_word_embeddings=self.tie_word_embeddings,
            initializer_range=self.initializer_range,
            has_lora_adapter=self.has_lora_adapter,
        )


@require_torch
class GraniteSpeechPlusForConditionalGenerationModelTest(_GraniteSpeechModelTestBase):
    """
    Model tester for `GraniteSpeechPlusForConditionalGeneration`.
    """

    all_model_classes = (GraniteSpeechPlusForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {"any-to-any": GraniteSpeechPlusForConditionalGeneration} if is_torch_available() else {}

    def setUp(self):
        self.model_tester = GraniteSpeechPlusForConditionalGenerationModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=GraniteSpeechPlusConfig,
            has_text_modality=False,
        )

    def test_encoder_hidden_layers_concat_shape(self):
        """With ``encoder_hidden_layers`` set, get_audio_features concatenates the selected intermediate
        hidden states with the final hidden state before the projector."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = GraniteSpeechPlusForConditionalGeneration(config).to(
            self.model_tester.parent.device if hasattr(self.model_tester.parent, "device") else "cpu"
        )
        model.eval()
        with torch.no_grad():
            out = model.get_audio_features(inputs_dict["input_features"].to(next(model.parameters()).device))
        self.assertEqual(out.pooler_output.shape[0], inputs_dict["input_features"].shape[0])

    @parameterized.expand([True, False, None])
    def test_get_audio_features_output(self, return_dict: bool | None):
        for model_class in self.all_model_classes:
            if not hasattr(model_class, "get_audio_features"):
                continue

            config, inputs_dict = self._audio_features_prepare_config_and_inputs()
            if return_dict is not None:
                config.return_dict = return_dict

            model = model_class(config).eval()
            model = model.to(torch_device)

            set_seed(42)
            with torch.no_grad():
                outputs = model.get_audio_features(**inputs_dict)

            if return_dict in (True, None):
                self.assertTrue(
                    isinstance(outputs, ModelOutput), "get_audio_features() must return a BaseModelOutputWithPooling"
                )
                self.assertTrue(
                    hasattr(outputs, "last_hidden_state"),
                    "get_audio_features() must return a BaseModelOutputWithPooling with last_hidden_state",
                )
                self.assertTrue(
                    hasattr(outputs, "pooler_output"),
                    "get_audio_features() must return a BaseModelOutputWithPooling with pooler_output",
                )
                self.assertTrue(
                    hasattr(outputs, "hidden_states"),
                    "get_audio_features() must return a BaseModelOutputWithPooling with hidden_states",
                )
                if self.has_attentions:
                    self.assertTrue(
                        hasattr(outputs, "attentions"),
                        "get_audio_features() must return a BaseModelOutputWithPooling with attentions",
                    )

                if getattr(self, "skip_test_audio_features_output_shape", False):
                    return

                last_hidden_state_shape = outputs.last_hidden_state.shape

                if "input_features" in inputs_dict:
                    batch_size = inputs_dict["input_features"].shape[0]
                else:
                    batch_size = inputs_dict["input_values"].shape[0]
                self.assertEqual(
                    last_hidden_state_shape[0],
                    batch_size,
                    f"batch_size mismatch, full shape: {last_hidden_state_shape}",
                )

                audio_config = config.audio_config if hasattr(config, "audio_config") else config
                hidden_size = None
                if hasattr(audio_config, "projection_dim"):
                    hidden_size = audio_config.projection_dim
                elif hasattr(audio_config, "hidden_size"):
                    hidden_size = audio_config.hidden_size
                elif hasattr(audio_config, "encoder_config"):
                    hidden_size = audio_config.encoder_config.hidden_dim * (
                        len(audio_config.encoder_config.cat_hidden_layers) + 1
                    )
                elif hasattr(audio_config, "encoder_ffn_dim"):
                    hidden_size = audio_config.encoder_ffn_dim
                self.assertEqual(
                    last_hidden_state_shape[-1],
                    hidden_size,
                    f"hidden_size mismatch, full shape: {last_hidden_state_shape}",
                )

            else:
                self.assertIsInstance(outputs, tuple, "get_audio_features() must return a tuple if return_dict=False")


class GraniteSpeechPlusForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_path = "ibm-granite/granite-speech-4.1-2b-plus"
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.prompt = self._get_prompt(self.processor.tokenizer)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def _get_prompt(self, tokenizer):
        chat = [
            {
                "role": "system",
                "content": "Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant",
            },
            {
                "role": "user",
                "content": "<|audio|> can you transcribe the speech into a written format?",
            },
        ]
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id")[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    @slow
    def test_small_model_integration_test_single(self):
        model = GraniteSpeechPlusForConditionalGeneration.from_pretrained(self.model_path).to(torch_device)
        input_speech = self._load_datasamples(1)

        # Verify feature sizes; note that the feature mask refers to the size of
        # features that are masked into the LLM, not the output of the processor,
        # which is why we inspect the mask instead of the `num_features` tensor.
        inputs = self.processor(self.prompt, input_speech, return_tensors="pt").to(torch_device)

        num_computed_features = self.processor.audio_processor._get_num_audio_features(
            [speech_arr.shape[-1] for speech_arr in input_speech],
        )[0]
        num_actual_features = torch.sum(inputs["input_features_mask"]).item()
        assert num_actual_features == num_computed_features

        # verify generation
        output = model.generate(**inputs, max_new_tokens=32)
        EXPECTED_DECODED_TEXT = "systemKnowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant\nuser can you transcribe the speech into a written format?\nassistantmister quiltor is the apostle of the middle classes and we are glad to welcome his gospel"  # fmt: skip

        self.assertEqual(
            self.processor.tokenizer.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch(self):
        model = GraniteSpeechPlusForConditionalGeneration.from_pretrained(self.model_path).to(torch_device)
        input_speech = self._load_datasamples(2)
        prompts = [self.prompt, self.prompt]

        # Verify feature sizes & padding
        inputs = self.processor(prompts, input_speech, return_tensors="pt").to(model.device)
        num_computed_features = self.processor.audio_processor._get_num_audio_features(
            [speech_arr.shape[-1] for speech_arr in input_speech],
        )
        num_actual_features = torch.sum(inputs["input_features_mask"], dim=-1)
        for e_feats, a_feats in zip(num_computed_features, num_actual_features):
            assert e_feats == a_feats.item()

        # verify generation
        output = model.generate(**inputs, max_new_tokens=32)

        EXPECTED_DECODED_TEXT = [
            "systemKnowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant\nuser can you transcribe the speech into a written format?\nassistantmister quiltor is the apostle of the middle classes and we are glad to welcome his gospel",
            "systemKnowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant\nuser can you transcribe the speech into a written format?\nassistantnor is mister quilter's manner less interesting than his matter"
        ]  # fmt: skip

        self.assertEqual(
            self.processor.tokenizer.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
