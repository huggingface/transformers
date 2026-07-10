# Copyright 2025 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.
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
"""Tests for Fun-ASR-Nano model."""

import inspect
import os
import unittest

from transformers import FunAsrNanoConfig, FunAsrNanoEncoderConfig, Qwen3Config
from transformers.testing_utils import require_torch, require_torch_gpu, slow

from ...alm_tester import ALMModelTest, ALMModelTester
from ...test_modeling_common import is_torch_available, torch_device


if is_torch_available():
    import torch

    from transformers import AutoProcessor, FunAsrNanoEncoder, FunAsrNanoForConditionalGeneration, FunAsrNanoModel
    from transformers.modeling_layers import GradientCheckpointingLayer
    from transformers.models.audioflamingo3.modeling_audioflamingo3 import (
        AudioFlamingo3ForConditionalGeneration,
        AudioFlamingo3Model,
        AudioFlamingo3PreTrainedModel,
    )
    from transformers.models.fun_asr_nano import modeling_fun_asr_nano
    from transformers.models.fun_asr_nano.convert_fun_asr_nano_to_hf import convert_key
    from transformers.models.fun_asr_nano.modular_fun_asr_nano import (
        FunAsrNanoAdaptorAttention,
        FunAsrNanoAdaptorLayer,
        FunAsrNanoEncoderLayer,
        FunAsrNanoFeedForward,
        FunAsrNanoMultiModalProjector,
        FunAsrNanoSANMAttention,
    )
    from transformers.models.fun_asr_nano.modular_fun_asr_nano import (
        FunAsrNanoForConditionalGeneration as ModularFunAsrNanoForConditionalGeneration,
    )
    from transformers.models.fun_asr_nano.modular_fun_asr_nano import (
        FunAsrNanoModel as ModularFunAsrNanoModel,
    )
    from transformers.models.fun_asr_nano.modular_fun_asr_nano import (
        FunAsrNanoPreTrainedModel as ModularFunAsrNanoPreTrainedModel,
    )
    from transformers.models.llama.modeling_llama import LlamaAttention
    from transformers.models.parakeet.modeling_parakeet import ParakeetEncoderFeedForward
    from transformers.models.voxtral.modeling_voxtral import VoxtralMultiModalProjector


class FunAsrNanoModelTester(ALMModelTester):
    config_class = FunAsrNanoConfig
    base_model_class = FunAsrNanoModel
    conditional_generation_class = FunAsrNanoForConditionalGeneration
    text_config_class = Qwen3Config
    audio_config_class = FunAsrNanoEncoderConfig
    audio_config_key = "audio_encoder_config"
    audio_mask_key = "input_features_mask"

    def __init__(self, parent, **kwargs):
        # Fun-ASR-Nano audio features are (batch, time, feature_dim) after LFR.
        kwargs.setdefault("feat_seq_length", 20)
        kwargs.setdefault("num_mel_bins", 80)
        kwargs.setdefault("vocab_size", 100)
        kwargs.setdefault("audio_token_id", 0)
        kwargs.setdefault("audio_token_index", 0)
        kwargs.setdefault("is_training", False)

        # Small Qwen3 text config for common tests.
        kwargs.setdefault("hidden_size", 32)
        kwargs.setdefault("intermediate_size", 64)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 8)
        kwargs.setdefault("max_position_embeddings", 128)

        # Small encoder config.
        kwargs.setdefault("input_size", 80)
        kwargs.setdefault("output_dim", 32)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("intermediate_size", 64)
        kwargs.setdefault("encoder_layers", 2)
        kwargs.setdefault("tp_blocks", 1)
        kwargs.setdefault("kernel_size", 5)
        kwargs.setdefault("sanm_shift", 0)
        kwargs.setdefault("dropout", 0.0)
        kwargs.setdefault("attention_dropout", 0.0)
        kwargs.setdefault("adaptor_intermediate_size", 64)
        kwargs.setdefault("adaptor_num_hidden_layers", 1)
        kwargs.setdefault("adaptor_num_attention_heads", 4)
        kwargs.setdefault("adaptor_dropout", 0.0)

        super().__init__(parent, **kwargs)

    def create_audio_features(self):
        """Fun-ASR-Nano audio features are (batch, time, feature_dim) after LFR."""
        from ...test_modeling_common import floats_tensor

        return floats_tensor([self.batch_size, self.feat_seq_length, self.num_mel_bins])

    def get_audio_embeds_mask(self, audio_mask):
        """Fun-ASR-Nano encoder and adaptor preserve sequence length in the test config."""
        return audio_mask

    def _prepare_modality_inputs(self, input_ids, config):
        input_features = self.create_audio_features()
        feature_lengths = torch.full((self.batch_size,), self.feat_seq_length, dtype=torch.long, device=torch_device)
        positions = torch.arange(self.feat_seq_length, device=torch_device)[None, :]
        audio_mask = positions < feature_lengths[:, None]
        audio_embeds_mask = self.get_audio_embeds_mask(audio_mask)
        num_audio_tokens = audio_embeds_mask.sum(dim=1)
        input_ids = self.place_audio_tokens(input_ids, config, num_audio_tokens)
        return input_ids, {"input_features": input_features, "input_features_mask": audio_mask}


@require_torch
class FunAsrNanoForConditionalGenerationModelTest(ALMModelTest, unittest.TestCase):
    """Model tester for `FunAsrNanoForConditionalGeneration`."""

    model_tester_class = FunAsrNanoModelTester
    pipeline_model_mapping = {}
    model_split_percents = [0.5, 0.8, 0.9]

    # The adaptor pools/flattens the audio embeddings, so `get_audio_features().pooler_output` is not the standard
    # `(batch, seq, hidden)` shape the common test expects.
    skip_test_audio_features_output_shape = True

    def test_reuses_audioflamingo3_model_wrappers(self):
        self.assertTrue(issubclass(ModularFunAsrNanoPreTrainedModel, AudioFlamingo3PreTrainedModel))
        self.assertTrue(issubclass(ModularFunAsrNanoModel, AudioFlamingo3Model))
        self.assertTrue(issubclass(ModularFunAsrNanoForConditionalGeneration, AudioFlamingo3ForConditionalGeneration))

    def test_generated_model_has_single_causal_lm_output_class(self):
        self.assertTrue(hasattr(modeling_fun_asr_nano, "FunAsrNanoCausalLMOutputWithPast"))
        self.assertFalse(hasattr(modeling_fun_asr_nano, "FunAsrNanoCausalLMOutput"))

    def test_public_get_audio_features_accepts_omitted_mask(self):
        parameter = inspect.signature(FunAsrNanoForConditionalGeneration.get_audio_features).parameters[
            "input_features_mask"
        ]
        self.assertIsNone(parameter.default)

    def test_checkpoint_key_mapping_matches_reused_component_names(self):
        self.assertEqual(
            convert_key("audio_encoder.encoders.0.self_attn.linear_q_k_v.weight"),
            "model.audio_tower.encoders.0.self_attn.linear_q_k_v.weight",
        )
        self.assertEqual(
            convert_key("audio_encoder.encoders.0.feed_forward.w_1.weight"),
            "model.audio_tower.encoders.0.feed_forward.linear1.weight",
        )
        self.assertEqual(
            convert_key("audio_adaptor.blocks.0.feed_forward.w_2.bias"),
            "model.multi_modal_projector.blocks.0.feed_forward.linear2.bias",
        )
        self.assertEqual(
            convert_key("audio_adaptor.linear1.weight"),
            "model.multi_modal_projector.linear_1.weight",
        )
        self.assertEqual(
            convert_key("audio_adaptor.linear2.bias"),
            "model.multi_modal_projector.linear_2.bias",
        )

    def test_forward_docstring_uses_fun_asr_nano_checkpoint(self):
        docstring = FunAsrNanoForConditionalGeneration.forward.__doc__
        self.assertIn("FunAudioLLM/Fun-ASR-Nano-2512-hf", docstring)
        self.assertNotIn("nvidia/audio-flamingo-3-hf", docstring)

    def test_audio_layers_use_shared_gradient_checkpointing(self):
        self.assertTrue(issubclass(FunAsrNanoEncoderLayer, GradientCheckpointingLayer))
        self.assertTrue(issubclass(FunAsrNanoAdaptorLayer, GradientCheckpointingLayer))

    def test_reuses_shared_audio_components(self):
        self.assertTrue(issubclass(FunAsrNanoSANMAttention, LlamaAttention))
        self.assertTrue(issubclass(FunAsrNanoAdaptorAttention, LlamaAttention))
        self.assertTrue(issubclass(FunAsrNanoFeedForward, ParakeetEncoderFeedForward))
        self.assertTrue(issubclass(FunAsrNanoMultiModalProjector, VoxtralMultiModalProjector))

    def test_encoder_uses_standard_layer_norm(self):
        config = self.model_tester.get_config().audio_encoder_config
        encoder = FunAsrNanoEncoder(config)
        norms = [encoder.after_norm, encoder.tp_norm]
        for layer in [*encoder.encoders0, *encoder.encoders, *encoder.tp_encoders]:
            norms.extend([layer.norm1, layer.norm2])
        for norm in norms:
            self.assertIs(type(norm), torch.nn.LayerNorm)

    @require_torch_gpu
    def test_standard_layer_norm_matches_float32_accumulation_in_bfloat16(self):
        hidden_size = self.model_tester.get_config().audio_encoder_config.output_dim
        layer_norm = torch.nn.LayerNorm(hidden_size, device=torch_device, dtype=torch.bfloat16)
        hidden_states = torch.randn(2, 17, hidden_size, device=torch_device, dtype=torch.bfloat16)

        expected = torch.nn.functional.layer_norm(
            hidden_states.float(),
            layer_norm.normalized_shape,
            layer_norm.weight.float(),
            layer_norm.bias.float(),
            layer_norm.eps,
        ).to(torch.bfloat16)
        actual = layer_norm(hidden_states)

        torch.testing.assert_close(actual, expected, rtol=0, atol=5e-3)

    @unittest.skip(reason="inputs_embeds is the audio-fused path; can't match raw token-only embeddings.")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="FunAsrNanoForConditionalGeneration has no separate base model without a generation head.")
    def test_model_base_model_prefix(self):
        pass

    @unittest.skip(reason="The tiny Fun-ASR-Nano test config is not split across two GPUs by auto device-map.")
    def test_model_parallelism(self):
        pass

    @unittest.skip(
        reason="The SAN-M encoder uses a custom attention path (encoders0/encoders/tp_encoders) that does not expose "
        "per-layer attentions in the standard way."
    )
    def test_get_audio_features_attentions(self):
        pass

    @unittest.skip(
        reason="The SAN-M encoder stacks heterogeneous blocks (encoders0/encoders/tp_encoders), so the per-layer "
        "hidden-state count does not match the single `num_hidden_layers` the common test assumes."
    )
    def test_get_audio_features_hidden_states(self):
        pass


@slow
@require_torch_gpu
class FunAsrNanoIntegrationTest(unittest.TestCase):
    """Integration tests with real checkpoint (run with RUN_SLOW=1).

    Expected outputs from original FunASR:
    - ZH (example/zh.mp3): "开饭时间早上九点至下午五点。"
    - EN (example/en.mp3): "The tribal chieftain called for the boy, and presented him with fifty pieces of gold."
    """

    # Allow pointing at a local checkpoint for pre-upload verification via env override.
    model_id = os.environ.get("FUN_ASR_NANO_MODEL_ID", "FunAudioLLM/Fun-ASR-Nano-2512-hf")

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(cls.model_id)
        cls.processor.tokenizer.padding_side = "left"
        cls.model = FunAsrNanoForConditionalGeneration.from_pretrained(
            cls.model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )

    def _load_audio(self, filename):
        """Load audio from the model repo."""
        import librosa
        from huggingface_hub import hf_hub_download

        audio_path = hf_hub_download("FunAudioLLM/Fun-ASR-Nano-2512", filename)
        audio, _ = librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)
        return audio

    def _decode_generated(self, generated_ids, inputs):
        """Decode only newly generated tokens, excluding the prompt."""
        generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

    def _prepare_inputs(self, audio, prompt_text):
        """Prepare model inputs using apply_chat_template."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "audio"},
                ],
            },
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audio=audio, sampling_rate=16000, return_tensors="pt")
        return inputs.to(self.model.device)

    def test_generate_chinese(self):
        """Test Chinese transcription matches expected output."""
        audio = self._load_audio("example/zh.mp3")
        inputs = self._prepare_inputs(audio, "语音转写成中文：")

        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        text = self._decode_generated(generated_ids, inputs)[0].strip()

        EXPECTED_TRANSCRIPT = "开饭时间早上九点至下午五点。"
        self.assertEqual(text, EXPECTED_TRANSCRIPT)

    def test_generate_english(self):
        """Test English transcription matches expected output."""
        audio = self._load_audio("example/en.mp3")
        inputs = self._prepare_inputs(audio, "Transcribe the audio:")

        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        text = self._decode_generated(generated_ids, inputs)[0].strip()

        EXPECTED_TRANSCRIPT = "The tribal chieftain called for the boy, and presented him with fifty pieces of gold."
        self.assertEqual(text, EXPECTED_TRANSCRIPT)

    def test_generate_batch(self):
        """Test batch inference with both Chinese and English audio."""
        audio_zh = self._load_audio("example/zh.mp3")
        audio_en = self._load_audio("example/en.mp3")

        messages_zh = [
            {"role": "user", "content": [{"type": "text", "text": "语音转写成中文："}, {"type": "audio"}]},
        ]
        messages_en = [
            {"role": "user", "content": [{"type": "text", "text": "Transcribe the audio:"}, {"type": "audio"}]},
        ]

        text_zh = self.processor.apply_chat_template(messages_zh, add_generation_prompt=True, tokenize=False)
        text_en = self.processor.apply_chat_template(messages_en, add_generation_prompt=True, tokenize=False)

        inputs = self.processor(
            text=[text_zh, text_en],
            audio=[audio_zh, audio_en],
            sampling_rate=16000,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        transcripts = [text.strip() for text in self._decode_generated(generated_ids, inputs)]

        EXPECTED_TRANSCRIPTS = [
            "开饭时间早上九点至下午五点。",
            "The tribal chieftain called for the boy, and presented him with fifty pieces of gold.",
        ]
        self.assertListEqual(transcripts, EXPECTED_TRANSCRIPTS)


if __name__ == "__main__":
    unittest.main()
