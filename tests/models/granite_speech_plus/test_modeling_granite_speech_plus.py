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

from transformers import (
    AutoProcessor,
    GraniteSpeechPlusConfig,
    GraniteSpeechPlusEncoderConfig,
    GraniteSpeechPlusForConditionalGeneration,
)
from transformers.testing_utils import cleanup, require_torch, slow, torch_device
from transformers.utils import is_datasets_available, is_torch_available

from ..granite_speech.test_modeling_granite_speech import (
    GraniteSpeechForConditionalGenerationModelTest,
    GraniteSpeechModelTester,
)


if is_torch_available():
    import torch
if is_datasets_available():
    from datasets import load_dataset


class GraniteSpeechPlusForConditionalGenerationModelTester(GraniteSpeechModelTester):
    """
    Plus variant that exercises the ``cat_hidden_layers`` concat path. The projector's
    ``encoder_hidden_size`` is scaled to match ``encoder_config.hidden_dim * (len(cat_hidden_layers) + 1)``.
    """

    config_class = GraniteSpeechPlusConfig
    conditional_generation_class = GraniteSpeechPlusForConditionalGeneration
    audio_config_class = GraniteSpeechPlusEncoderConfig

    def __init__(self, parent, cat_hidden_layers=(0,), **kwargs):
        super().__init__(parent, **kwargs)
        self.cat_hidden_layers = list(cat_hidden_layers)
        # Projector encoder_hidden_size must equal hidden_dim * (len(cat_hidden_layers) + 1).
        self.projector_config = {
            "model_type": "blip_2_qformer",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 256,
            "encoder_hidden_size": 32 * (len(self.cat_hidden_layers) + 1),
        }


@require_torch
class GraniteSpeechPlusForConditionalGenerationModelTest(GraniteSpeechForConditionalGenerationModelTest):
    """
    Model tester for `GraniteSpeechPlusForConditionalGeneration`.
    """

    model_tester_class = GraniteSpeechPlusForConditionalGenerationModelTester
    pipeline_model_mapping = {"any-to-any": GraniteSpeechPlusForConditionalGeneration} if is_torch_available() else {}

    # The cat path changes the encoder output feature dim, so the generic shape assertion in
    # `test_get_audio_features_output` (which assumes hidden_dim) does not apply.
    skip_test_audio_features_output_shape = True

    def test_encoder_hidden_layers_concat_shape(self):
        """``encoder_config.cat_hidden_layers`` concatenates selected intermediate hidden states with the final
        hidden state along the feature dim before the projector."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = GraniteSpeechPlusForConditionalGeneration(config).to(torch_device).eval()
        with torch.no_grad():
            out = model.get_audio_features(inputs_dict["input_features"].to(next(model.parameters()).device))
        cat_factor = len(config.encoder_config.cat_hidden_layers) + 1
        expected_hidden_size = config.encoder_config.hidden_dim * cat_factor
        self.assertEqual(out.last_hidden_state.shape[0], inputs_dict["input_features"].shape[0])
        self.assertEqual(out.last_hidden_state.shape[-1], expected_hidden_size)


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
