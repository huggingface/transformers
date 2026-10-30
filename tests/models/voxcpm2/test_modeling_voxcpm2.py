# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import unittest
from unittest.mock import patch

import numpy as np

from transformers import GenerationConfig, VoxCPM2Config, is_torch_available, set_seed
from transformers.testing_utils import require_torch, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from .test_processing_voxcpm2 import get_tiny_voxcpm2_processor


if is_torch_available():
    import torch

    from transformers import TextToAudioPipeline, VoxCPM2Model
    from transformers.models.voxcpm2.modeling_voxcpm2 import VoxCPM2GenerationOutput


class VoxCPM2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=3,
        is_training=True,
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        feat_dim=4,
        patch_size=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.feat_dim = feat_dim
        self.patch_size = patch_size

    def get_config(self):
        return VoxCPM2Config(
            lm_config={
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": 1,
                "head_dim": self.hidden_size // self.num_attention_heads,
                "max_position_embeddings": 32,
            },
            encoder_config={
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "head_dim": self.hidden_size // self.num_attention_heads,
            },
            dit_config={
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "head_dim": self.hidden_size // self.num_attention_heads,
                "cfm_config": {"training_cfg_rate": 0.0},
            },
            audio_vae_config={
                "encoder_dim": 4,
                "encoder_rates": [2],
                "latent_dim": self.feat_dim,
                "decoder_dim": 8,
                "decoder_rates": [2],
                "sr_bin_boundaries": [10000, 20000],
            },
            feat_dim=self.feat_dim,
            patch_size=self.patch_size,
            residual_lm_num_layers=1,
            scalar_quantization_latent_dim=4,
            audio_start_token_id=13,
            audio_end_token_id=14,
            reference_audio_start_token_id=15,
            reference_audio_end_token_id=16,
        )

    def prepare_config_and_inputs_for_common(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        text_mask = torch.ones_like(input_ids)
        text_mask[:, -1] = 0
        audio_features = torch.randn(
            self.batch_size, self.seq_length, self.patch_size, self.feat_dim, device=torch_device
        )
        return config, {
            "input_ids": input_ids,
            "text_mask": text_mask,
            "audio_features": audio_features,
            "audio_mask": 1 - text_mask,
        }


@require_torch
class VoxCPM2ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (VoxCPM2Model,) if is_torch_available() else ()
    _is_composite = True
    has_attentions = False
    test_resize_embeddings = False

    @property
    def all_generative_model_classes(self):
        # The waveform generator has no token sequences; its public paths are tested below.
        return ()

    def setUp(self):
        self.model_tester = VoxCPM2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VoxCPM2Config, has_text_modality=False)
        if self._testMethodName.startswith("test_eager_matches_sdpa_inference"):
            self.skipTest(
                "The common comparison reads hidden states; test_attention_backends_match checks stop logits."
            )

    def get_model(self, processor=None):
        config = self.model_tester.get_config()
        if processor is not None:
            config.lm_config.vocab_size = len(processor.tokenizer)
            config.audio_start_token_id = processor.audio_start_token_id
            config.reference_audio_start_token_id = processor.reference_audio_start_token_id
            config.reference_audio_end_token_id = processor.reference_audio_end_token_id
        return VoxCPM2Model(config).to(torch_device).eval()

    def get_generation_inputs(self):
        _, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        return {key: value[:1] for key, value in inputs.items()}

    def get_stop_outputs(self, stop=True):
        return (
            torch.tensor([[0.0, 1.0]], device=torch_device),
            torch.tensor([stop], device=torch_device),
        )

    @unittest.skip("VoxCPM2 exposes stop logits and audio latents, without top-level hidden states.")
    def test_hidden_states_output(self):
        pass

    @unittest.skip("VoxCPM2 does not return top-level hidden states or attentions to retain gradients on.")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip("The four backbones use derived configurations with different dimensions and attention modes.")
    def test_internal_model_config_and_subconfig_are_same(self):
        pass

    def test_model_base_model_prefix(self):
        # The public model contains the complete speech stack, rather than a separate LM head wrapper.
        model = self.get_model()
        self.assertIs(model.base_model, model)

    def test_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = VoxCPM2Model(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(**inputs)
        self.assertEqual(outputs.stop_logits.shape, (self.model_tester.batch_size, self.model_tester.seq_length, 2))
        self.assertEqual(
            outputs.latent_features.shape,
            (
                self.model_tester.batch_size,
                self.model_tester.feat_dim,
                self.model_tester.seq_length * self.model_tester.patch_size,
            ),
        )

    def test_training(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = VoxCPM2Model(config).to(torch_device).train()
        inputs["audio_features"].requires_grad_()
        outputs = model(
            **inputs,
            loss_mask=inputs["audio_mask"].float(),
            labels=inputs["audio_mask"],
            sample_generate=True,
            num_inference_steps=2,
        )
        self.assertEqual(outputs.loss.ndim, 0)
        self.assertTrue(torch.isfinite(outputs.loss))
        torch.testing.assert_close(outputs.loss, outputs.diffusion_loss + outputs.stop_loss)
        self.assertEqual(outputs.generated_latent_features.shape, outputs.latent_features.shape)
        outputs.loss.backward()
        self.assertIsNotNone(model.stop_head.weight.grad)
        self.assertIsNotNone(inputs["audio_features"].grad)

    def test_attention_backends_match(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        config._attn_implementation = "eager"
        eager = VoxCPM2Model(config).to(torch_device).eval()
        sdpa_config = copy.deepcopy(config)
        sdpa_config._attn_implementation = "sdpa"
        sdpa = VoxCPM2Model(sdpa_config).to(torch_device).eval()
        sdpa.load_state_dict(eager.state_dict())
        with torch.no_grad():
            expected = eager(**inputs)
            actual = sdpa(**inputs)
        torch.testing.assert_close(actual.stop_logits, expected.stop_logits, rtol=1e-5, atol=1e-6)

    def test_generate(self):
        model = self.get_model()
        inputs = self.get_generation_inputs()
        generation_config = GenerationConfig(min_new_tokens=2, max_new_tokens=2, return_dict_in_generate=True)
        kwargs = {"generation_config": generation_config, "num_inference_steps": 2, "decoder_context_patches": 1}
        with patch.object(model, "_get_stop_flags", return_value=self.get_stop_outputs()):
            set_seed(7)
            outputs = model.generate(**inputs, **kwargs)
            set_seed(7)
            repeated = model.generate(**inputs, **kwargs)
            set_seed(7)
            waveform = model.generate(**inputs, **{**kwargs, "return_dict_in_generate": False})
        self.assertIsInstance(outputs, VoxCPM2GenerationOutput)
        self.assertEqual(outputs.num_generated_patches, 2)
        self.assertEqual(outputs.audio.shape, (1, 2 * model.patch_size * model._decode_chunk_size))
        self.assertEqual(outputs.audio_features.shape, (1, 2, model.patch_size, model.feat_dim))
        torch.testing.assert_close(repeated.audio, outputs.audio, rtol=0, atol=0)
        torch.testing.assert_close(waveform, outputs.audio, rtol=0, atol=0)

    def test_generate_stopping(self):
        model = self.get_model()
        inputs = self.get_generation_inputs()
        for stop, minimum, maximum, expected in [(True, 3, 5, 3), (False, 0, 2, 2)]:
            with (
                self.subTest(stop=stop),
                patch.object(model, "_get_stop_flags", return_value=self.get_stop_outputs(stop)),
            ):
                outputs = model.generate(
                    **inputs,
                    min_new_audio_patches=minimum,
                    max_new_audio_patches=maximum,
                    num_inference_steps=1,
                    return_dict_in_generate=True,
                )
                self.assertEqual(outputs.num_generated_patches, expected)

    def test_raw_and_precomputed_conditioning_match(self):
        processor = get_tiny_voxcpm2_processor()
        model = self.get_model(processor)
        inputs = processor(
            text="A",
            audio=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            prompt_text="B",
            reference_audio=np.array([0.4, 0.5, 0.6], dtype=np.float32),
            sampling_rate=16000,
            return_tensors="pt",
        ).to(torch_device)
        aligned = model._prepare_generation_audio_features(
            inputs.input_ids,
            inputs.audio_mask,
            prompt_input_values=inputs.prompt_input_values,
            prompt_attention_mask=inputs.prompt_attention_mask,
            reference_input_values=inputs.reference_input_values,
            reference_attention_mask=inputs.reference_attention_mask,
        )
        kwargs = {"min_new_audio_patches": 2, "max_new_audio_patches": 2, "num_inference_steps": 1}
        set_seed(7)
        expected = model.generate(
            input_ids=inputs.input_ids,
            text_mask=inputs.text_mask,
            audio_features=aligned,
            audio_mask=inputs.audio_mask,
            **kwargs,
        )
        set_seed(7)
        actual = model.generate(**inputs, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_streaming_matches_full_generation(self):
        model = self.get_model()
        inputs = self.get_generation_inputs()
        kwargs = {
            "min_new_audio_patches": 2,
            "max_new_audio_patches": 2,
            "num_inference_steps": 1,
            "decoder_context_patches": 1,
        }
        set_seed(7)
        expected = model.generate(**inputs, **kwargs)
        set_seed(7)
        chunks = list(model.generate_streaming(**inputs, return_dict_in_generate=True, **kwargs))
        self.assertEqual([chunk.num_generated_patches for chunk in chunks], [1, 2])
        self.assertTrue(all(isinstance(chunk, VoxCPM2GenerationOutput) for chunk in chunks))
        torch.testing.assert_close(
            torch.cat([chunk.audio for chunk in chunks], dim=-1), expected, rtol=1e-5, atol=1e-6
        )

    def test_generation_caches_match_full_context(self):
        model = self.get_model()
        inputs = self.get_generation_inputs()
        next_patch = torch.randn(1, model.patch_size, model.feat_dim, device=torch_device)
        _, _, _, base_cache, residual_cache = model._prefill_generation(**inputs)
        actual_base, actual_residual, base_cache, residual_cache = model._update_generation_cache(
            next_patch, base_cache, residual_cache
        )

        encoded_audio = model.enc_to_lm_proj(model.feat_encoder(inputs["audio_features"]))
        text_embeddings = model.base_lm.embed_tokens(inputs["input_ids"])
        text_mask = inputs["text_mask"].unsqueeze(-1)
        audio_mask = inputs["audio_mask"].unsqueeze(-1)
        prompt = text_mask * text_embeddings + audio_mask * encoded_audio
        encoded_patch = model.enc_to_lm_proj(model.feat_encoder(next_patch.unsqueeze(1)))
        base_inputs = torch.cat((prompt, encoded_patch), dim=1)
        base_states = model.base_lm(inputs_embeds=base_inputs).last_hidden_state
        expected_base = model.fsq_layer(base_states[:, -1])
        prompt_states = text_mask * base_states[:, :-1] + audio_mask * model.fsq_layer(base_states[:, :-1])
        residual_prompt = model.fusion_concat_proj(torch.cat((prompt_states, audio_mask * encoded_audio), dim=-1))
        residual_patch = model.fusion_concat_proj(torch.cat((expected_base, encoded_patch[:, 0]), dim=-1)).unsqueeze(1)
        residual_inputs = torch.cat((residual_prompt, residual_patch), dim=1)
        expected_residual = model.residual_lm(inputs_embeds=residual_inputs).last_hidden_state[:, -1]

        torch.testing.assert_close(actual_base, expected_base, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_residual, expected_residual, rtol=1e-5, atol=1e-6)
        self.assertEqual(base_cache.get_seq_length(), inputs["input_ids"].shape[1] + 1)
        self.assertEqual(residual_cache.get_seq_length(), inputs["input_ids"].shape[1] + 1)

    def test_generation_input_validation(self):
        model = self.get_model()
        inputs = self.get_generation_inputs()
        with self.assertRaisesRegex(ValueError, "batch size of 1"):
            model.generate(**{key: value.repeat_interleave(2, dim=0) for key, value in inputs.items()})
        with self.assertRaisesRegex(ValueError, "greater than"):
            model.generate(**inputs, min_new_audio_patches=3, max_new_audio_patches=2)
        with self.assertRaisesRegex(ValueError, "Unsupported generation arguments"):
            model.generate(**inputs, unsupported_argument=True)
        with self.assertRaisesRegex(ValueError, "patches with shape"):
            model.generate(**{**inputs, "audio_features": inputs["audio_features"][:, :, :1]})

    def test_audio_vae_streaming(self):
        audio_vae = self.get_model().audio_vae
        values = torch.randn(2, 13, device=torch_device)
        latents = audio_vae.encode(values, sampling_rate=16000)
        self.assertEqual(latents.shape, (2, self.model_tester.feat_dim, 7))
        expected = audio_vae.decode(latents)
        self.assertEqual(expected.shape, (2, 1, 14))
        with audio_vae.streaming_decode() as decoder:
            actual = torch.cat([decoder.decode_chunk(chunk) for chunk in latents.split([3, 4], dim=-1)], dim=-1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "expects 16000 Hz"):
            audio_vae.encode(values, sampling_rate=8000)

    def test_flow_euler_constant_velocity(self):
        class ConstantVelocity(torch.nn.Module):
            def forward(self, sample, mu, timestep, conditioning, delta_timestep):
                return torch.full_like(sample, 0.25)

        flow = self.get_model().feat_decoder
        flow.estimator = ConstantVelocity()
        sample = torch.ones(1, self.model_tester.feat_dim, self.model_tester.patch_size, device=torch_device)
        output = flow.solve_euler(
            sample,
            torch.tensor([1.0, 0.5, 0.0], device=torch_device),
            torch.zeros(1, 2 * self.model_tester.hidden_size, device=torch_device),
            torch.zeros_like(sample),
            use_cfg_zero_star=False,
        )
        torch.testing.assert_close(output, torch.full_like(output, 0.75), rtol=0, atol=0)

    def test_text_to_audio_pipeline(self):
        processor = get_tiny_voxcpm2_processor()
        model = self.get_model(processor)
        speech_generator = TextToAudioPipeline(model=model, processor=processor, device=torch_device)
        outputs = speech_generator(
            "A",
            generate_kwargs={"min_new_audio_patches": 1, "max_new_audio_patches": 1, "num_inference_steps": 1},
        )
        self.assertEqual(outputs["sampling_rate"], model.sample_rate)
        self.assertEqual(outputs["audio"].shape[-1], model.patch_size * model._decode_chunk_size)
