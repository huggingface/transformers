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

import math

import numpy as np
import pytest

from transformers import (
    AutoModel,
    AutoModelForTextToWaveform,
    GenerationConfig,
    TextToAudioPipeline,
    VoxCPM2AudioVAEConfig,
    VoxCPM2Config,
    VoxCPM2TextConfig,
    is_torch_available,
)
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers.models.voxcpm2.modeling_voxcpm2 import (
        VoxCPM2Attention,
        VoxCPM2AudioDecoder,
        VoxCPM2AudioEncoder,
        VoxCPM2AudioVAE,
        VoxCPM2BackboneModel,
        VoxCPM2CausalConv1d,
        VoxCPM2CausalConvTranspose1d,
        VoxCPM2CausalDecoderBlock,
        VoxCPM2CausalEncoderBlock,
        VoxCPM2CausalResidualUnit,
        VoxCPM2ConditionalFlowMatching,
        VoxCPM2DecoderLayer,
        VoxCPM2GenerationOutput,
        VoxCPM2LocalDiT,
        VoxCPM2LocalEncoder,
        VoxCPM2Model,
        VoxCPM2ModelOutput,
        VoxCPM2NoiseBlock,
        VoxCPM2PreTrainedModel,
        VoxCPM2RMSNorm,
        VoxCPM2RotaryEmbedding,
        VoxCPM2SampleRateConditionLayer,
        VoxCPM2ScalarQuantizationLayer,
        VoxCPM2SinusoidalPositionEmbedding,
        VoxCPM2Snake1d,
        VoxCPM2TimestepEmbedding,
    )

    all_model_classes = (VoxCPM2Model,)

from .test_processing_voxcpm2 import get_tiny_voxcpm2_processor


def get_tiny_voxcpm2_config() -> VoxCPM2Config:
    return VoxCPM2Config(
        lm_config={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "kv_channels": 4,
            "no_rope": True,
            "rope_parameters": None,
        },
        encoder_config={"hidden_dim": 8, "ffn_dim": 16, "num_heads": 2, "num_layers": 1, "kv_channels": 4},
        dit_config={
            "hidden_dim": 8,
            "ffn_dim": 16,
            "num_heads": 2,
            "num_layers": 1,
            "kv_channels": 4,
            "cfm_config": {"training_cfg_rate": 0.0, "t_scheduler": "uniform"},
        },
        audio_vae_config={
            "encoder_dim": 4,
            "encoder_rates": (2,),
            "latent_dim": 4,
            "decoder_dim": 8,
            "decoder_rates": (2,),
            "depthwise": True,
            "sr_bin_boundaries": (10000, 20000),
        },
        feat_dim=4,
        patch_size=2,
        residual_lm_num_layers=1,
        scalar_quantization_latent_dim=4,
    )


@require_torch
def test_pretrained_model_metadata():
    model = VoxCPM2PreTrainedModel(VoxCPM2Config())

    assert model.main_input_name == "input_ids"
    assert model.input_modalities == ("audio", "text")
    assert model._supports_sdpa
    assert not model.supports_gradient_checkpointing
    assert "VoxCPM2DecoderLayer" in model._no_split_modules
    assert model._get_dtype_plan(torch.float16)["audio_vae"] == torch.float32
    assert model._get_dtype_plan(torch.bfloat16)["audio_vae"] == torch.float32


@require_torch
def test_pretrained_model_initialization():
    model = VoxCPM2PreTrainedModel(VoxCPM2Config())
    convolution = torch.nn.utils.parametrizations.weight_norm(VoxCPM2CausalConv1d(2, 3, 3))
    with torch.no_grad():
        convolution.parametrizations.weight.original0.zero_()
        convolution.parametrizations.weight.original1.zero_()

    model._init_weights(convolution)

    weight_magnitude = convolution.parametrizations.weight.original0
    weight_vector = convolution.parametrizations.weight.original1
    expected_magnitude = torch.linalg.vector_norm(weight_vector, dim=(1, 2), keepdim=True)
    torch.testing.assert_close(weight_magnitude, expected_magnitude)
    assert torch.count_nonzero(weight_vector)

    condition = VoxCPM2SampleRateConditionLayer(input_dim=4, num_sample_rate_buckets=3)
    model._init_weights(condition)
    torch.testing.assert_close(condition.scale_embed.weight, torch.ones_like(condition.scale_embed.weight))
    torch.testing.assert_close(condition.bias_embed.weight, torch.zeros_like(condition.bias_embed.weight))


@require_torch
def test_model_constructor_and_state_dict_layout():
    config = get_tiny_voxcpm2_config()
    model = VoxCPM2Model(config)

    assert model.base_lm.config._attn_implementation == "sdpa"
    assert model.residual_lm.config._attn_implementation == "sdpa"
    assert model.feat_encoder.encoder.config._attn_implementation == "sdpa"
    assert model.feat_decoder.estimator.decoder.config._attn_implementation == "sdpa"
    assert model.base_model is model
    assert model.can_generate()
    assert model.generation_config is not None
    assert model.chunk_size == 2
    assert model._decode_chunk_size == 2
    assert model.audio_start_token == 101
    assert model.ref_audio_end_token == 104

    state_keys = set(model.state_dict())
    expected_keys = {
        "base_lm.embed_tokens.weight",
        "residual_lm.norm.weight",
        "feat_encoder.special_token",
        "feat_decoder.estimator.in_proj.weight",
        "fsq_layer.in_proj.weight",
        "audio_vae.encoder.fc_mu.parametrizations.weight.original0",
    }
    assert expected_keys.issubset(state_keys)
    assert not any(key.startswith("model.") for key in state_keys)

    assert model.get_input_embeddings() is model.base_lm.embed_tokens
    replacement_embeddings = torch.nn.Embedding(32, 8)
    model.set_input_embeddings(replacement_embeddings)
    assert model.get_input_embeddings() is replacement_embeddings


@require_torch
def test_auto_model_registration():
    config = get_tiny_voxcpm2_config()

    assert isinstance(AutoModel.from_config(config), VoxCPM2Model)
    assert isinstance(AutoModelForTextToWaveform.from_config(config), VoxCPM2Model)


@require_torch
def test_text_to_audio_pipeline_zero_shot():
    model = VoxCPM2Model(get_tiny_voxcpm2_config()).eval()
    processor = get_tiny_voxcpm2_processor()
    speech_generator = TextToAudioPipeline(
        model=model,
        processor=processor,
        device=-1,
        min_new_tokens=1,
        max_new_tokens=1,
    )

    output = speech_generator(
        "A",
        generate_kwargs={"num_inference_steps": 1},
    )

    assert speech_generator.generation_config.min_new_tokens == 1
    assert speech_generator.generation_config.max_new_tokens == 1
    assert isinstance(output["audio"], np.ndarray)
    assert output["audio"].shape == (model.patch_size * model._decode_chunk_size,)
    assert output["sampling_rate"] == model.config.sample_rate
    assert "'output_modalities': ('audio',)" in repr(speech_generator)


@require_torch
def test_model_training_forward_and_diagnostic_sampling():
    model = VoxCPM2Model(get_tiny_voxcpm2_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    text_mask = torch.tensor([[1, 1, 0]])
    audio_mask = 1 - text_mask
    audio_features = torch.randn(1, 3, 2, 4, requires_grad=True)
    loss_mask = torch.tensor([[0, 0, 1]], dtype=torch.float32)
    labels = torch.tensor([[0, 0, 1]])

    torch.manual_seed(7)
    output = model(
        input_ids=input_ids,
        text_mask=text_mask,
        audio_features=audio_features,
        audio_mask=audio_mask,
        loss_mask=loss_mask,
        labels=labels,
        sample_generate=True,
        num_inference_steps=2,
    )

    assert isinstance(output, VoxCPM2ModelOutput)
    assert output.loss.ndim == 0
    assert output.diffusion_loss.ndim == 0
    assert output.stop_loss.ndim == 0
    torch.testing.assert_close(output.loss, output.diffusion_loss + output.stop_loss)
    assert output.stop_logits.shape == (1, 3, 2)
    assert output.latent_features.shape == (1, 4, 6)
    assert output.generated_latent_features.shape == (1, 4, 6)
    expected_latent_features = audio_features.reshape(1, 6, 4).transpose(1, 2).contiguous()
    torch.testing.assert_close(output.latent_features, expected_latent_features)

    output.loss.backward()
    assert audio_features.grad is not None
    assert model.stop_head.weight.grad is not None

    tuple_output = model(
        input_ids=input_ids,
        text_mask=text_mask,
        audio_features=audio_features.detach(),
        audio_mask=audio_mask,
        return_dict=False,
    )
    assert isinstance(tuple_output, tuple)
    assert tuple_output[0].shape == (1, 3, 2)
    assert tuple_output[1].shape == (1, 4, 6)

    with pytest.raises(ValueError, match="loss_mask"):
        model(input_ids, text_mask, audio_features.detach(), audio_mask, labels=labels)


@require_torch
def test_generation_input_validation():
    model = VoxCPM2Model(get_tiny_voxcpm2_config())
    input_ids = torch.ones(1, 3, dtype=torch.long)
    text_mask = torch.ones_like(input_ids)
    audio_features = torch.zeros(1, 3, 2, 4)
    audio_mask = torch.zeros_like(input_ids)

    model._validate_generation_inputs(
        input_ids,
        text_mask,
        audio_features,
        audio_mask,
        min_new_audio_patches=2,
        max_new_audio_patches=2,
        num_inference_steps=1,
    )

    with pytest.raises(ValueError, match="batch size of 1"):
        model._validate_generation_inputs(
            input_ids.repeat(2, 1),
            text_mask.repeat(2, 1),
            audio_features.repeat(2, 1, 1, 1),
            audio_mask.repeat(2, 1),
            min_new_audio_patches=0,
            max_new_audio_patches=2,
            num_inference_steps=1,
        )
    with pytest.raises(ValueError, match="patches with shape"):
        model._validate_generation_inputs(
            input_ids,
            text_mask,
            torch.zeros(1, 3, 3, 4),
            audio_mask,
            min_new_audio_patches=0,
            max_new_audio_patches=2,
            num_inference_steps=1,
        )
    with pytest.raises(ValueError, match="greater than"):
        model._validate_generation_inputs(
            input_ids,
            text_mask,
            audio_features,
            audio_mask,
            min_new_audio_patches=3,
            max_new_audio_patches=2,
            num_inference_steps=1,
        )


@require_torch
def test_generation_audio_extraction_uses_attention_mask():
    model = VoxCPM2Model(get_tiny_voxcpm2_config())
    input_values = torch.tensor([[[0.0, 0.0, 2.0, 0.0, 3.0, 0.0]]])

    left_padded = model._extract_generation_audio(input_values, torch.tensor([[0, 0, 1, 1, 1, 1]]))
    right_padded = model._extract_generation_audio(input_values, torch.tensor([[1, 1, 1, 1, 0, 0]]))

    assert left_padded.tolist() == [[[2.0, 0.0, 3.0, 0.0]]]
    assert right_padded.tolist() == [[[0.0, 0.0, 2.0, 0.0]]]
    torch.testing.assert_close(model._extract_generation_audio(input_values[:, 0]), input_values)

    with pytest.raises(ValueError, match="contiguous"):
        model._extract_generation_audio(input_values, torch.tensor([[1, 0, 1, 0, 0, 0]]))
    with pytest.raises(ValueError, match="zeros and ones"):
        model._extract_generation_audio(input_values, torch.tensor([[1, 2, 1, 0, 0, 0]]))
    with pytest.raises(ValueError, match="unmasked sample"):
        model._extract_generation_audio(input_values, torch.zeros(1, 6))
    with pytest.raises(ValueError, match="shape"):
        model._extract_generation_audio(torch.zeros(1, 2, 6))
    with pytest.raises(ValueError, match="one audio sample"):
        model._extract_generation_audio(torch.zeros(2, 1, 6))


@require_torch
def test_generation_audio_encoding_applies_role_specific_padding():
    model = VoxCPM2Model(get_tiny_voxcpm2_config()).eval()
    input_values = torch.tensor([[[1.0, 2.0, 3.0, 0.0]]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])

    left_padded_features = model._encode_generation_audio_features(input_values, attention_mask, "left")
    right_padded_features = model._encode_generation_audio_features(input_values, attention_mask, "right")

    manual_left_padding = torch.nn.functional.pad(input_values[..., :3], (1, 0))
    expected_left_features = model.audio_vae.encode(manual_left_padding, sampling_rate=16000)
    expected_left_features = expected_left_features.transpose(1, 2).reshape(1, 1, 2, 4)
    manual_right_padding = torch.nn.functional.pad(input_values[..., :3], (0, 1))
    expected_right_features = model.audio_vae.encode(manual_right_padding, sampling_rate=16000)
    expected_right_features = expected_right_features.transpose(1, 2).reshape(1, 1, 2, 4)

    torch.testing.assert_close(left_padded_features, expected_left_features, rtol=0, atol=0)
    torch.testing.assert_close(right_padded_features, expected_right_features, rtol=0, atol=0)
    assert not torch.equal(left_padded_features, right_padded_features)

    with pytest.raises(ValueError, match="padding_side"):
        model._encode_generation_audio_features(input_values, attention_mask, "middle")


@require_torch
def test_generation_audio_feature_alignment_preserves_prompt_order():
    model = VoxCPM2Model(get_tiny_voxcpm2_config())
    input_ids = torch.ones(1, 5, dtype=torch.long)
    reference_features = torch.stack((torch.ones(2, 4), torch.full((2, 4), 2.0))).unsqueeze(0)
    prompt_features = torch.full((1, 1, 2, 4), 3.0)

    aligned_features = model._align_generation_audio_features(
        input_ids,
        audio_mask=torch.tensor([[0, 1, 1, 0, 1]]),
        reference_features=reference_features,
        prompt_features=prompt_features,
    )

    torch.testing.assert_close(aligned_features[0, 1], reference_features[0, 0])
    torch.testing.assert_close(aligned_features[0, 2], reference_features[0, 1])
    torch.testing.assert_close(aligned_features[0, 4], prompt_features[0, 0])
    assert torch.count_nonzero(aligned_features[0, [0, 3]]) == 0

    zero_shot_features = model._align_generation_audio_features(
        input_ids[:, :2],
        audio_mask=torch.zeros(1, 2),
    )
    assert zero_shot_features.shape == (1, 2, 2, 4)
    assert torch.count_nonzero(zero_shot_features) == 0

    with pytest.raises(ValueError, match="received 2 patches"):
        model._align_generation_audio_features(
            input_ids,
            audio_mask=torch.tensor([[0, 1, 0, 0, 0]]),
            reference_features=reference_features,
        )


@require_torch
def test_generation_audio_feature_preparation_supports_raw_and_precomputed_inputs():
    model = VoxCPM2Model(get_tiny_voxcpm2_config()).eval()
    input_ids = torch.ones(1, 4, dtype=torch.long)
    audio_mask = torch.tensor([[0, 1, 0, 1]])
    reference_values = torch.tensor([[[1.0, 2.0, 3.0, 0.0]]])
    prompt_values = torch.tensor([[[4.0, 5.0, 6.0, 0.0]]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])

    prepared_features = model._prepare_generation_audio_features(
        input_ids,
        audio_mask,
        prompt_input_values=prompt_values,
        prompt_attention_mask=attention_mask,
        reference_input_values=reference_values,
        reference_attention_mask=attention_mask,
    )
    reference_features = model._encode_generation_audio_features(reference_values, attention_mask, "right")
    prompt_features = model._encode_generation_audio_features(prompt_values, attention_mask, "left")
    expected_features = model._align_generation_audio_features(
        input_ids,
        audio_mask,
        reference_features=reference_features,
        prompt_features=prompt_features,
    )
    torch.testing.assert_close(prepared_features, expected_features, rtol=0, atol=0)

    precomputed_features = torch.randn(1, 4, 2, 4)
    returned_features = model._prepare_generation_audio_features(input_ids, audio_mask, precomputed_features)
    assert returned_features is precomputed_features

    zero_shot_features = model._prepare_generation_audio_features(input_ids[:, :2], torch.zeros(1, 2))
    assert torch.count_nonzero(zero_shot_features) == 0

    with pytest.raises(ValueError, match="cannot be combined"):
        model._prepare_generation_audio_features(
            input_ids,
            audio_mask,
            precomputed_features,
            prompt_input_values=prompt_values,
        )
    with pytest.raises(ValueError, match="requires"):
        model._prepare_generation_audio_features(
            input_ids,
            audio_mask,
            prompt_attention_mask=attention_mask,
        )


@require_torch
def test_generation_prefill_matches_full_backbones():
    model = VoxCPM2Model(get_tiny_voxcpm2_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    text_mask = torch.tensor([[1, 1, 0]])
    audio_mask = 1 - text_mask
    audio_features = torch.randn(1, 3, 2, 4)

    lm_hidden_states, residual_hidden_states, conditioning_features, base_cache, residual_cache = (
        model._prefill_generation(input_ids, text_mask, audio_features, audio_mask)
    )

    encoded_features = model.enc_to_lm_proj(model.feat_encoder(audio_features))
    text_embeddings = model.base_lm.embed_tokens(input_ids)
    inputs_embeds = text_mask.unsqueeze(-1) * text_embeddings + audio_mask.unsqueeze(-1) * encoded_features
    base_states = model.base_lm(inputs_embeds=inputs_embeds).last_hidden_state
    expected_encoded_states = model.fsq_layer(base_states) * audio_mask.unsqueeze(-1)
    expected_encoded_states += base_states * text_mask.unsqueeze(-1)
    residual_inputs = model.fusion_concat_proj(
        torch.cat((expected_encoded_states, audio_mask.unsqueeze(-1) * encoded_features), dim=-1)
    )
    expected_residual_states = model.residual_lm(inputs_embeds=residual_inputs).last_hidden_state

    torch.testing.assert_close(lm_hidden_states, expected_encoded_states[:, -1], rtol=0, atol=0)
    torch.testing.assert_close(residual_hidden_states, expected_residual_states[:, -1], rtol=0, atol=0)
    torch.testing.assert_close(conditioning_features, audio_features[:, -1], rtol=0, atol=0)
    assert base_cache is not residual_cache
    assert base_cache.get_seq_length() == input_ids.shape[1]
    assert residual_cache.get_seq_length() == input_ids.shape[1]


@require_torch
def test_generation_audio_patch_sampling():
    model = VoxCPM2Model(get_tiny_voxcpm2_config()).eval()

    class RecordingFlowMatcher(torch.nn.Module):
        def forward(self, mu, num_inference_steps, patch_size, conditioning, **kwargs):
            self.call = (mu, num_inference_steps, patch_size, conditioning, kwargs)
            return conditioning

    flow_matcher = RecordingFlowMatcher()
    model.feat_decoder = flow_matcher
    lm_hidden_states = torch.randn(1, 8)
    residual_hidden_states = torch.randn(1, 8)
    conditioning_features = torch.randn(1, 2, 4)

    generated_features = model._sample_audio_patch(
        lm_hidden_states,
        residual_hidden_states,
        conditioning_features,
        num_inference_steps=3,
        guidance_scale=2.5,
        temperature=0.8,
        sway_sampling_coefficient=0.4,
        use_cfg_zero_star=False,
    )

    mu, num_inference_steps, patch_size, conditioning, kwargs = flow_matcher.call
    expected_mu = torch.cat(
        (model.lm_to_dit_proj(lm_hidden_states), model.res_to_dit_proj(residual_hidden_states)), dim=-1
    )
    torch.testing.assert_close(mu, expected_mu)
    torch.testing.assert_close(conditioning, conditioning_features.transpose(1, 2))
    torch.testing.assert_close(generated_features, conditioning_features)
    assert num_inference_steps == 3
    assert patch_size == 2
    assert kwargs == {
        "temperature": 0.8,
        "cfg_value": 2.5,
        "sway_sampling_coefficient": 0.4,
        "use_cfg_zero_star": False,
    }

    generator = torch.Generator().manual_seed(7)
    model._sample_audio_patch(
        lm_hidden_states,
        residual_hidden_states,
        conditioning_features,
        num_inference_steps=3,
        guidance_scale=2.5,
        temperature=0.8,
        sway_sampling_coefficient=0.4,
        use_cfg_zero_star=False,
        generator=generator,
    )
    assert flow_matcher.call[-1]["generator"] is generator


@require_torch
def test_generation_cache_update_matches_full_recomputation():
    model = VoxCPM2Model(get_tiny_voxcpm2_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    text_mask = torch.tensor([[1, 1, 0]])
    audio_mask = 1 - text_mask
    audio_features = torch.randn(1, 3, 2, 4)
    generated_features = torch.randn(1, 2, 4)

    _, _, _, base_cache, residual_cache = model._prefill_generation(input_ids, text_mask, audio_features, audio_mask)
    lm_hidden_states, residual_hidden_states, base_cache, residual_cache = model._update_generation_cache(
        generated_features, base_cache, residual_cache
    )

    encoded_features = model.enc_to_lm_proj(model.feat_encoder(audio_features))
    text_embeddings = model.base_lm.embed_tokens(input_ids)
    prompt_embeddings = text_mask.unsqueeze(-1) * text_embeddings + audio_mask.unsqueeze(-1) * encoded_features
    base_prompt_states = model.base_lm(inputs_embeds=prompt_embeddings).last_hidden_state
    mixed_base_prompt_states = model.fsq_layer(base_prompt_states) * audio_mask.unsqueeze(-1)
    mixed_base_prompt_states += base_prompt_states * text_mask.unsqueeze(-1)
    encoded_patch = model.enc_to_lm_proj(model.feat_encoder(generated_features.unsqueeze(1)))
    full_base_inputs = torch.cat((prompt_embeddings, encoded_patch), dim=1)
    expected_lm_hidden_states = model.base_lm(inputs_embeds=full_base_inputs).last_hidden_state[:, -1]
    expected_lm_hidden_states = model.fsq_layer(expected_lm_hidden_states)

    residual_prompt_inputs = model.fusion_concat_proj(
        torch.cat((mixed_base_prompt_states, audio_mask.unsqueeze(-1) * encoded_features), dim=-1)
    )
    residual_patch_input = model.fusion_concat_proj(
        torch.cat((expected_lm_hidden_states, encoded_patch[:, 0]), dim=-1)
    ).unsqueeze(1)
    full_residual_inputs = torch.cat((residual_prompt_inputs, residual_patch_input), dim=1)
    expected_residual_hidden_states = model.residual_lm(inputs_embeds=full_residual_inputs).last_hidden_state[:, -1]

    torch.testing.assert_close(lm_hidden_states, expected_lm_hidden_states, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(residual_hidden_states, expected_residual_hidden_states, rtol=1e-5, atol=1e-6)
    assert base_cache.get_seq_length() == input_ids.shape[1] + 1
    assert residual_cache.get_seq_length() == input_ids.shape[1] + 1


@require_torch
def test_generation_stop_prediction():
    model = VoxCPM2Model(get_tiny_voxcpm2_config()).eval()
    hidden_states = torch.randn(3, 8)

    stop_logits, stop_flags = model._get_stop_flags(hidden_states)

    expected_logits = model.stop_head(model.stop_actn(model.stop_proj(hidden_states)))
    torch.testing.assert_close(stop_logits, expected_logits)
    torch.testing.assert_close(stop_flags, expected_logits.argmax(dim=-1).bool())
    assert stop_logits.shape == (3, 2)
    assert stop_flags.dtype == torch.bool


@require_torch
def test_generation_decoder_context_uses_trailing_audio_only():
    model = VoxCPM2Model(get_tiny_voxcpm2_config())
    audio_features = torch.arange(1 * 5 * 2 * 4).reshape(1, 5, 2, 4)

    context = model._prepare_decoder_context(
        audio_features, audio_mask=torch.tensor([[1, 0, 0, 1, 1]]), decoder_context_patches=3
    )
    torch.testing.assert_close(context, audio_features[:, -2:])

    empty_context = model._prepare_decoder_context(
        audio_features, audio_mask=torch.tensor([[1, 0, 0, 1, 0]]), decoder_context_patches=3
    )
    assert empty_context.shape == (1, 0, 2, 4)

    zero_context = model._prepare_decoder_context(
        audio_features, audio_mask=torch.ones(1, 5), decoder_context_patches=0
    )
    assert zero_context.shape == (1, 0, 2, 4)

    with pytest.raises(ValueError, match="non-negative"):
        model._prepare_decoder_context(audio_features, torch.ones(1, 5), decoder_context_patches=-1)


@require_torch
def test_autoregressive_audio_feature_generation():
    model = VoxCPM2Model(get_tiny_voxcpm2_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    text_mask = torch.ones_like(input_ids)
    audio_mask = torch.zeros_like(input_ids)
    audio_features = torch.zeros(1, 3, 2, 4)
    model._get_stop_flags = lambda hidden_states: (
        torch.tensor([[0.0, 1.0]], device=hidden_states.device),
        torch.tensor([True], device=hidden_states.device),
    )

    output = model._generate_audio_features(
        input_ids,
        text_mask,
        audio_features,
        audio_mask,
        min_new_audio_patches=4,
        max_new_audio_patches=5,
        num_inference_steps=1,
        generator=torch.Generator().manual_seed(7),
    )

    assert isinstance(output, VoxCPM2GenerationOutput)
    assert output.num_generated_patches == 4
    assert output.audio_features.shape == (1, 4, 2, 4)
    assert output.latent_features.shape == (1, 4, 8)
    assert output.stop_logits.shape == (1, 4, 2)
    expected_latent_features = output.audio_features.reshape(1, 8, 4).transpose(1, 2).contiguous()
    torch.testing.assert_close(output.latent_features, expected_latent_features)

    repeated_output = model._generate_audio_features(
        input_ids,
        text_mask,
        audio_features,
        audio_mask,
        min_new_audio_patches=4,
        max_new_audio_patches=5,
        num_inference_steps=1,
        generator=torch.Generator().manual_seed(7),
    )
    different_output = model._generate_audio_features(
        input_ids,
        text_mask,
        audio_features,
        audio_mask,
        min_new_audio_patches=4,
        max_new_audio_patches=5,
        num_inference_steps=1,
        generator=torch.Generator().manual_seed(8),
    )
    torch.testing.assert_close(output.audio_features, repeated_output.audio_features, rtol=0, atol=0)
    assert not torch.equal(output.audio_features, different_output.audio_features)

    with pytest.raises(ValueError, match="max_cache_length"):
        model._generate_audio_features(
            input_ids,
            text_mask,
            audio_features,
            audio_mask,
            min_new_audio_patches=4,
            max_new_audio_patches=model.config.max_cache_length,
            num_inference_steps=1,
        )


@require_torch
def test_waveform_generation_and_decoder_context_crop():
    model = VoxCPM2Model(get_tiny_voxcpm2_config()).eval()
    input_ids = torch.tensor([[1, 0, 0]])
    text_mask = torch.tensor([[1, 0, 0]])
    audio_mask = 1 - text_mask
    audio_features = torch.randn(1, 3, 2, 4)
    model._get_stop_flags = lambda hidden_states: (
        torch.tensor([[0.0, 1.0]], device=hidden_states.device),
        torch.tensor([True], device=hidden_states.device),
    )
    generation_config = GenerationConfig(min_new_tokens=2, max_new_tokens=2, return_dict_in_generate=True)

    output = model.generate(
        input_ids,
        text_mask,
        audio_features,
        audio_mask,
        generation_config=generation_config,
        num_inference_steps=1,
        decoder_context_patches=1,
        generator=torch.Generator().manual_seed(7),
    )

    assert isinstance(output, VoxCPM2GenerationOutput)
    assert "audio" in output
    assert output.audio.shape == (1, 8)
    decoder_context = audio_features[:, -1:]
    decoder_features = torch.cat((decoder_context, output.audio_features), dim=1)
    decoder_features = decoder_features.reshape(1, 6, 4).transpose(1, 2).contiguous()
    expected_audio = model.audio_vae.decode(decoder_features).squeeze(1)
    expected_audio = expected_audio[:, model.patch_size * model._decode_chunk_size :]
    torch.testing.assert_close(output.audio, expected_audio)

    waveform = model.generate(
        input_ids,
        text_mask,
        audio_features,
        audio_mask,
        min_new_audio_patches=2,
        max_new_audio_patches=2,
        num_inference_steps=1,
        generator=torch.Generator().manual_seed(7),
    )
    assert isinstance(waveform, torch.Tensor)
    assert waveform.shape == (1, 8)


@require_torch
def test_raw_prompt_generation_matches_precomputed_audio_features():
    model = VoxCPM2Model(get_tiny_voxcpm2_config()).eval()
    processor = get_tiny_voxcpm2_processor()
    model._get_stop_flags = lambda hidden_states: (
        torch.tensor([[0.0, 1.0]], device=hidden_states.device),
        torch.tensor([True], device=hidden_states.device),
    )
    model_inputs = processor(
        text="A",
        audio=np.array([4.0, 5.0, 6.0], dtype=np.float32),
        prompt_text="B",
        reference_audio=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        sampling_rate=16000,
        return_tensors="pt",
    )
    aligned_features = model._prepare_generation_audio_features(
        model_inputs.input_ids,
        model_inputs.audio_mask,
        prompt_input_values=model_inputs.prompt_input_values,
        prompt_attention_mask=model_inputs.prompt_attention_mask,
        reference_input_values=model_inputs.reference_input_values,
        reference_attention_mask=model_inputs.reference_attention_mask,
    )
    generation_kwargs = {
        "min_new_audio_patches": 2,
        "max_new_audio_patches": 2,
        "num_inference_steps": 1,
        "decoder_context_patches": 1,
    }

    precomputed_waveform = model.generate(
        model_inputs.input_ids,
        model_inputs.text_mask,
        aligned_features,
        model_inputs.audio_mask,
        generator=torch.Generator().manual_seed(7),
        **generation_kwargs,
    )
    raw_waveform = model.generate(
        **model_inputs,
        generator=torch.Generator().manual_seed(7),
        **generation_kwargs,
    )

    torch.testing.assert_close(raw_waveform, precomputed_waveform, rtol=0, atol=0)


@require_torch
def test_raw_prompt_streaming_matches_precomputed_audio_features():
    model = VoxCPM2Model(get_tiny_voxcpm2_config()).eval()
    processor = get_tiny_voxcpm2_processor()
    model._get_stop_flags = lambda hidden_states: (
        torch.tensor([[0.0, 1.0]], device=hidden_states.device),
        torch.tensor([True], device=hidden_states.device),
    )
    model_inputs = processor(
        text="A",
        audio=np.array([4.0, 5.0, 6.0], dtype=np.float32),
        prompt_text="B",
        reference_audio=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        sampling_rate=16000,
        return_tensors="pt",
    )
    aligned_features = model._prepare_generation_audio_features(
        model_inputs.input_ids,
        model_inputs.audio_mask,
        prompt_input_values=model_inputs.prompt_input_values,
        prompt_attention_mask=model_inputs.prompt_attention_mask,
        reference_input_values=model_inputs.reference_input_values,
        reference_attention_mask=model_inputs.reference_attention_mask,
    )
    generation_kwargs = {
        "min_new_audio_patches": 2,
        "max_new_audio_patches": 2,
        "num_inference_steps": 1,
        "decoder_context_patches": 1,
    }

    precomputed_chunks = list(
        model.generate_streaming(
            model_inputs.input_ids,
            model_inputs.text_mask,
            aligned_features,
            model_inputs.audio_mask,
            generator=torch.Generator().manual_seed(7),
            **generation_kwargs,
        )
    )
    raw_chunks = list(
        model.generate_streaming(
            **model_inputs,
            generator=torch.Generator().manual_seed(7),
            **generation_kwargs,
        )
    )

    assert len(raw_chunks) == len(precomputed_chunks) == 2
    for raw_chunk, precomputed_chunk in zip(raw_chunks, precomputed_chunks):
        torch.testing.assert_close(raw_chunk, precomputed_chunk, rtol=0, atol=0)


@require_torch
def test_streaming_waveform_generation_matches_non_streaming_generation():
    model = VoxCPM2Model(get_tiny_voxcpm2_config()).eval()
    input_ids = torch.tensor([[1, 0, 0]])
    text_mask = torch.tensor([[1, 0, 0]])
    audio_mask = 1 - text_mask
    audio_features = torch.randn(1, 3, 2, 4)
    model._get_stop_flags = lambda hidden_states: (
        torch.tensor([[0.0, 1.0]], device=hidden_states.device),
        torch.tensor([True], device=hidden_states.device),
    )
    generation_kwargs = {
        "min_new_audio_patches": 2,
        "max_new_audio_patches": 2,
        "num_inference_steps": 1,
        "decoder_context_patches": 1,
    }

    waveform = model.generate(
        input_ids,
        text_mask,
        audio_features,
        audio_mask,
        generator=torch.Generator().manual_seed(7),
        **generation_kwargs,
    )
    streamed_chunks = list(
        model.generate_streaming(
            input_ids,
            text_mask,
            audio_features,
            audio_mask,
            generator=torch.Generator().manual_seed(7),
            **generation_kwargs,
        )
    )

    assert len(streamed_chunks) == 2
    assert all(chunk.shape == (1, 4) for chunk in streamed_chunks)
    torch.testing.assert_close(torch.cat(streamed_chunks, dim=-1), waveform, rtol=1e-5, atol=1e-6)

    structured_chunks = list(
        model.generate_streaming(
            input_ids,
            text_mask,
            audio_features,
            audio_mask,
            generator=torch.Generator().manual_seed(7),
            return_dict_in_generate=True,
            **generation_kwargs,
        )
    )
    assert all(isinstance(chunk, VoxCPM2GenerationOutput) for chunk in structured_chunks)
    assert [chunk.num_generated_patches for chunk in structured_chunks] == [1, 2]
    assert all(chunk.audio.shape == (1, 4) for chunk in structured_chunks)


@require_torch
def test_streaming_audio_features_match_non_streaming_generation():
    model = VoxCPM2Model(get_tiny_voxcpm2_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    text_mask = torch.ones_like(input_ids)
    audio_mask = torch.zeros_like(input_ids)
    audio_features = torch.zeros(1, 3, 2, 4)
    model._get_stop_flags = lambda hidden_states: (
        torch.tensor([[0.0, 1.0]], device=hidden_states.device),
        torch.tensor([True], device=hidden_states.device),
    )

    chunks = list(
        model._generate_audio_features_streaming(
            input_ids,
            text_mask,
            audio_features,
            audio_mask,
            min_new_audio_patches=4,
            max_new_audio_patches=5,
            num_inference_steps=1,
            generator=torch.Generator().manual_seed(7),
        )
    )
    full_output = model._generate_audio_features(
        input_ids,
        text_mask,
        audio_features,
        audio_mask,
        min_new_audio_patches=4,
        max_new_audio_patches=5,
        num_inference_steps=1,
        generator=torch.Generator().manual_seed(7),
    )

    assert len(chunks) == 4
    assert [chunk.num_generated_patches for chunk in chunks] == [1, 2, 3, 4]
    assert all(chunk.audio_features.shape == (1, 1, 2, 4) for chunk in chunks)
    streamed_features = torch.cat([chunk.audio_features for chunk in chunks], dim=1)
    torch.testing.assert_close(streamed_features, full_output.audio_features, rtol=0, atol=0)


@require_torch
def test_generation_parameter_resolution():
    model = VoxCPM2Model(get_tiny_voxcpm2_config())

    defaults = model._resolve_generation_parameters(None, None, None, None, None, None, {})
    assert defaults == (4, 2000, 2.0, 1.0, False)

    generation_config = GenerationConfig(
        min_new_tokens=2,
        max_new_tokens=3,
        guidance_scale=1.5,
        temperature=0.7,
        return_dict_in_generate=True,
    )
    resolved = model._resolve_generation_parameters(generation_config, None, None, None, None, None, {})
    assert resolved == (2, 3, 1.5, 0.7, True)

    updated = model._resolve_generation_parameters(None, None, None, None, None, None, {"max_new_tokens": 6})
    assert updated[1] == 6
    with pytest.raises(ValueError, match="Unsupported generation arguments"):
        model._resolve_generation_parameters(None, None, None, None, None, None, {"not_supported": True})


@require_torch
def test_scalar_quantization_matches_reference():
    config = VoxCPM2Config()
    config.lm_config.hidden_size = 2
    config.scalar_quantization_latent_dim = 3
    config.scalar_quantization_scale = 4
    layer = VoxCPM2ScalarQuantizationLayer(config)

    with torch.no_grad():
        layer.in_proj.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]]))
        layer.in_proj.bias.zero_()
        layer.out_proj.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]))
        layer.out_proj.bias.zero_()

    assert set(layer.state_dict()) == {"in_proj.weight", "in_proj.bias", "out_proj.weight", "out_proj.bias"}
    assert layer.in_proj.weight.shape == (3, 2)
    assert layer.out_proj.weight.shape == (2, 3)

    hidden_states = torch.tensor([[[-1.0, 0.25], [0.5, 1.0]]])
    projected_states = torch.tanh(layer.in_proj(hidden_states))
    quantized_states = torch.round(projected_states * layer.scale) / layer.scale
    expected_output = layer.out_proj(quantized_states)

    layer.eval()
    torch.testing.assert_close(layer(hidden_states), expected_output, rtol=0, atol=0)

    layer.train()
    training_input = hidden_states.clone().requires_grad_()
    training_output = layer(training_input)
    torch.testing.assert_close(training_output, expected_output, rtol=0, atol=0)
    training_output.sum().backward()

    reference_input = hidden_states.clone().requires_grad_()
    layer.out_proj(torch.tanh(layer.in_proj(reference_input))).sum().backward()
    torch.testing.assert_close(training_input.grad, reference_input.grad, rtol=0, atol=0)


@require_torch
def test_snake_activation_matches_reference():
    layer = VoxCPM2Snake1d(3)
    with torch.no_grad():
        layer.alpha.copy_(torch.tensor([[[0.5], [1.0], [2.0]]]))

    assert list(layer.state_dict()) == ["alpha"]
    assert layer.alpha.shape == (1, 3, 1)

    hidden_states = torch.randn(2, 3, 4, 5, requires_grad=True)
    output = layer(hidden_states)
    reshaped_states = hidden_states.reshape(2, 3, -1)
    expected_output = reshaped_states + (layer.alpha + 1e-9).reciprocal() * torch.sin(
        layer.alpha * reshaped_states
    ).pow(2)
    expected_output = expected_output.reshape_as(hidden_states)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

    output.sum().backward()
    output_gradient = hidden_states.grad.clone()
    hidden_states.grad = None
    expected_output.sum().backward()
    torch.testing.assert_close(output_gradient, hidden_states.grad, rtol=0, atol=0)


@require_torch
def test_causal_convolution_matches_reference():
    layer = VoxCPM2CausalConv1d(2, 4, kernel_size=6, stride=3, padding=2, output_padding=1)
    assert list(layer.state_dict()) == ["weight", "bias"]
    assert layer.padding == (0,)
    assert layer.causal_padding == 3

    hidden_states = torch.randn(2, 2, 19, requires_grad=True)
    output = layer(hidden_states)

    reference_input = hidden_states.detach().clone().requires_grad_()
    padded_states = torch.nn.functional.pad(reference_input, (layer.causal_padding, 0))
    expected_output = torch.nn.functional.conv1d(
        padded_states,
        layer.weight,
        layer.bias,
        stride=layer.stride,
        dilation=layer.dilation,
        groups=layer.groups,
    )
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

    output.sum().backward()
    expected_output.sum().backward()
    torch.testing.assert_close(hidden_states.grad, reference_input.grad, rtol=0, atol=0)


@require_torch
def test_causal_residual_unit_matches_reference():
    layer = VoxCPM2CausalResidualUnit(hidden_dim=4, dilation=3, groups=2)
    assert set(layer.state_dict()) == {
        "block.0.alpha",
        "block.1.bias",
        "block.1.parametrizations.weight.original0",
        "block.1.parametrizations.weight.original1",
        "block.2.alpha",
        "block.3.bias",
        "block.3.parametrizations.weight.original0",
        "block.3.parametrizations.weight.original1",
    }

    hidden_states = torch.randn(2, 4, 19, requires_grad=True)
    output = layer(hidden_states)
    expected_output = hidden_states + layer.block(hidden_states)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

    output.sum().backward()
    assert hidden_states.grad is not None


@require_torch
def test_causal_encoder_block_matches_reference():
    layer = VoxCPM2CausalEncoderBlock(output_dim=8, input_dim=4, stride=3, groups=2)
    state_keys = set(layer.state_dict())
    assert len(state_keys) == 28
    assert "block.0.block.1.parametrizations.weight.original0" in state_keys
    assert "block.4.parametrizations.weight.original1" in state_keys

    hidden_states = torch.randn(2, 4, 31, requires_grad=True)
    output = layer(hidden_states)
    expected_output = layer.block(hidden_states)
    assert output.shape == (2, 8, 10)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

    output.sum().backward()
    assert hidden_states.grad is not None


@require_torch
def test_audio_encoder_matches_reference():
    config = VoxCPM2AudioVAEConfig(
        encoder_dim=4,
        encoder_rates=(2, 3),
        latent_dim=3,
        decoder_dim=16,
        decoder_rates=(2, 2),
        depthwise=True,
    )
    model = VoxCPM2AudioEncoder(config)
    assert len(model.state_dict()) == 65
    assert model.encoder_dim == 16
    assert "block.0.parametrizations.weight.original0" in model.state_dict()
    assert "fc_mu.parametrizations.weight.original1" in model.state_dict()

    input_values = torch.randn(2, 1, 36, requires_grad=True)
    output = model(input_values)
    hidden_states = model.block(input_values)
    assert output["hidden_state"].shape == (2, 16, 6)
    assert output["mu"].shape == (2, 3, 6)
    assert output["logvar"].shape == (2, 3, 6)
    torch.testing.assert_close(output["hidden_state"], hidden_states, rtol=0, atol=0)
    torch.testing.assert_close(output["mu"], model.fc_mu(hidden_states), rtol=0, atol=0)
    torch.testing.assert_close(output["logvar"], model.fc_logvar(hidden_states), rtol=0, atol=0)

    output["mu"].sum().backward()
    assert input_values.grad is not None


@require_torch
def test_decoder_noise_block_matches_reference():
    layer = VoxCPM2NoiseBlock(4)
    assert set(layer.state_dict()) == {
        "linear.parametrizations.weight.original0",
        "linear.parametrizations.weight.original1",
    }

    hidden_states = torch.randn(2, 4, 9)
    torch.manual_seed(7)
    noise = torch.randn(2, 1, 9)
    expected_output = hidden_states + noise * layer.linear(hidden_states)
    torch.manual_seed(7)
    output = layer(hidden_states)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)


@require_torch
def test_causal_decoder_block_matches_reference():
    layer = VoxCPM2CausalDecoderBlock(input_dim=8, output_dim=4, stride=3, groups=2)
    state_keys = set(layer.state_dict())
    assert len(state_keys) == 28
    assert "block.1.parametrizations.weight.original0" in state_keys
    assert "block.2.block.1.parametrizations.weight.original1" in state_keys

    hidden_states = torch.randn(2, 8, 10, requires_grad=True)
    output = layer(hidden_states)
    expected_output = layer.block(hidden_states)
    assert output.shape == (2, 4, 30)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

    output.sum().backward()
    assert hidden_states.grad is not None

    noise_layer = VoxCPM2CausalDecoderBlock(8, 4, stride=3, groups=2, use_noise_block=True)
    assert len(noise_layer.state_dict()) == 30


@require_torch
def test_sample_rate_conditioning_matches_reference():
    hidden_states = torch.randn(2, 4, 5)
    sample_rate_ids = torch.tensor([0, 2])

    scale_bias_layer = VoxCPM2SampleRateConditionLayer(4, 3)
    assert set(scale_bias_layer.state_dict()) == {"scale_embed.weight", "bias_embed.weight"}
    torch.testing.assert_close(scale_bias_layer(hidden_states, sample_rate_ids), hidden_states, rtol=0, atol=0)

    add_layer = VoxCPM2SampleRateConditionLayer(4, 3, conditioning_type="add")
    expected_add_output = hidden_states + add_layer.cond_embed(sample_rate_ids).unsqueeze(-1)
    torch.testing.assert_close(add_layer(hidden_states, sample_rate_ids), expected_add_output, rtol=0, atol=0)

    concat_layer = VoxCPM2SampleRateConditionLayer(
        4, 3, conditioning_type="concat", conditioning_dim=2, use_output_layer=True
    )
    assert concat_layer(hidden_states, sample_rate_ids).shape == hidden_states.shape
    with pytest.raises(ValueError, match="use_output_layer"):
        VoxCPM2SampleRateConditionLayer(4, 3, conditioning_type="concat")


@require_torch
def test_audio_decoder_matches_reference():
    config = VoxCPM2AudioVAEConfig(
        encoder_dim=4,
        encoder_rates=(2, 3),
        latent_dim=3,
        decoder_dim=16,
        decoder_rates=(3, 2),
        depthwise=True,
        sr_bin_boundaries=(10, 20),
    )
    model = VoxCPM2AudioDecoder(config)
    assert len(model.state_dict()) == 71
    torch.testing.assert_close(model.state_dict()["sr_bin_boundaries"], torch.tensor([10, 20], dtype=torch.int32))

    sample_rate = torch.tensor([8, 25], dtype=torch.int32)
    torch.testing.assert_close(model.get_sample_rate_ids(sample_rate), torch.tensor([0, 2]))
    hidden_states = torch.randn(2, 3, 4)
    output = model(hidden_states, sample_rate)
    expected_output = hidden_states
    for layer in model.model:
        expected_output = layer(expected_output)
    assert output.shape == (2, 1, 24)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

    unconditioned_config = VoxCPM2AudioVAEConfig(**{**config.to_dict(), "sr_bin_boundaries": None})
    unconditioned_model = VoxCPM2AudioDecoder(unconditioned_config)
    assert isinstance(unconditioned_model.model, torch.nn.Sequential)
    assert unconditioned_model(hidden_states).shape == (2, 1, 24)


@require_torch
def test_audio_vae_encode_decode():
    config = VoxCPM2AudioVAEConfig(
        encoder_dim=4,
        encoder_rates=(2, 3),
        latent_dim=3,
        decoder_dim=16,
        decoder_rates=(3, 2),
        depthwise=True,
        sr_bin_boundaries=(10000, 20000),
        sample_rate=16000,
        out_sample_rate=24000,
    )
    model = VoxCPM2AudioVAE(config)
    assert len(model.state_dict()) == 136
    assert model.chunk_size == 6
    assert model.decode_chunk_size == 6

    input_values = torch.randn(2, 13)
    latent_features = model.encode(input_values, sampling_rate=16000)
    assert latent_features.shape == (2, 3, 3)
    output_values = model.decode(latent_features)
    assert output_values.shape == (2, 1, 18)

    with pytest.raises(ValueError, match="expects 16000 Hz"):
        model.encode(input_values, sampling_rate=8000)


@require_torch
def test_audio_vae_streaming_decode_matches_full_decode():
    config = VoxCPM2AudioVAEConfig(
        encoder_dim=4,
        encoder_rates=(2, 3),
        latent_dim=3,
        decoder_dim=16,
        decoder_rates=(3, 2),
        depthwise=True,
        sr_bin_boundaries=(10000, 20000),
        out_sample_rate=24000,
    )
    model = VoxCPM2AudioVAE(config).eval()
    latent_chunks = [torch.randn(2, 3, 4), torch.randn(2, 3, 4)]
    expected_output = model.decode(torch.cat(latent_chunks, dim=-1))

    streaming_decoder = model.streaming_decode()
    with streaming_decoder:
        output = torch.cat([streaming_decoder.decode_chunk(chunk) for chunk in latent_chunks], dim=-1)

    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    assert not streaming_decoder.states
    torch.testing.assert_close(model.decode(torch.cat(latent_chunks, dim=-1)), expected_output, rtol=0, atol=0)


@require_torch
def test_causal_transposed_convolution_matches_reference():
    layer = VoxCPM2CausalConvTranspose1d(4, 2, kernel_size=6, stride=3, padding=2, output_padding=1)
    assert list(layer.state_dict()) == ["weight", "bias"]
    assert layer.padding == (0,)
    assert layer.output_padding == (0,)
    assert layer.causal_trim == 3

    hidden_states = torch.randn(2, 4, 7, requires_grad=True)
    output = layer(hidden_states)

    reference_input = hidden_states.detach().clone().requires_grad_()
    expected_output = torch.nn.functional.conv_transpose1d(
        reference_input,
        layer.weight,
        layer.bias,
        stride=layer.stride,
        groups=layer.groups,
        dilation=layer.dilation,
    )
    expected_output = expected_output[..., : -layer.causal_trim]
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

    output.sum().backward()
    expected_output.sum().backward()
    torch.testing.assert_close(hidden_states.grad, reference_input.grad, rtol=0, atol=0)

    zero_trim_layer = VoxCPM2CausalConvTranspose1d(2, 3, kernel_size=3)
    zero_trim_input = torch.randn(1, 2, 5)
    zero_trim_output = zero_trim_layer(zero_trim_input)
    expected_zero_trim_output = torch.nn.functional.conv_transpose1d(
        zero_trim_input, zero_trim_layer.weight, zero_trim_layer.bias
    )
    torch.testing.assert_close(zero_trim_output, expected_zero_trim_output, rtol=0, atol=0)


@require_torch
def test_sinusoidal_timestep_embedding_matches_reference():
    layer = VoxCPM2SinusoidalPositionEmbedding(8)
    assert not layer.state_dict()

    for timesteps in (torch.tensor(0.25), torch.tensor([0.0, 0.25, 1.0])):
        output = layer(timesteps)
        normalized_timesteps = timesteps.reshape(-1)
        exponent = math.log(10000) / 3
        frequencies = torch.exp(torch.arange(4, dtype=timesteps.dtype) * -exponent)
        embeddings = 1000.0 * normalized_timesteps.unsqueeze(1) * frequencies.unsqueeze(0)
        expected_output = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

    assert layer(torch.tensor(0.5)).shape == (1, 8)
    assert layer(torch.tensor([0.5, 1.0])).shape == (2, 8)


@require_torch
def test_timestep_projection_matches_reference():
    layer = VoxCPM2TimestepEmbedding(4, 6, output_dim=5)
    assert list(layer.state_dict()) == [
        "linear_1.weight",
        "linear_1.bias",
        "linear_2.weight",
        "linear_2.bias",
    ]
    assert layer.linear_1.weight.shape == (6, 4)
    assert layer.linear_2.weight.shape == (5, 6)

    hidden_states = torch.randn(3, 4, requires_grad=True)
    output = layer(hidden_states)

    reference_input = hidden_states.detach().clone().requires_grad_()
    expected_output = torch.nn.functional.linear(reference_input, layer.linear_1.weight, layer.linear_1.bias)
    expected_output = torch.nn.functional.silu(expected_output)
    expected_output = torch.nn.functional.linear(expected_output, layer.linear_2.weight, layer.linear_2.bias)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

    output.sum().backward()
    expected_output.sum().backward()
    torch.testing.assert_close(hidden_states.grad, reference_input.grad, rtol=0, atol=0)

    default_output_layer = VoxCPM2TimestepEmbedding(4, 6)
    assert default_output_layer(torch.randn(2, 4)).shape == (2, 6)


@require_torch
def test_attention_matches_reference():
    config = VoxCPM2TextConfig(
        vocab_size=32,
        hidden_size=6,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        kv_channels=4,
        use_mup=False,
        rope_parameters=None,
    )
    config._attn_implementation = "sdpa"
    layer = VoxCPM2Attention(config, layer_idx=0).eval()

    assert list(layer.state_dict()) == ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"]
    assert layer.q_proj.weight.shape == (8, 6)
    assert layer.k_proj.weight.shape == (4, 6)
    assert layer.o_proj.weight.shape == (6, 8)

    hidden_states = torch.randn(1, 4, 6)
    angles = torch.randn(1, 4, 4)
    position_embeddings = (angles.cos(), angles.sin())
    for is_causal, rotary_embeddings in ((True, None), (False, None), (True, position_embeddings)):
        output, _ = layer(hidden_states, position_embeddings=rotary_embeddings, is_causal=is_causal)

        query_states = layer.q_proj(hidden_states).view(1, 4, 2, 4).transpose(1, 2)
        key_states = layer.k_proj(hidden_states).view(1, 4, 1, 4).transpose(1, 2)
        value_states = layer.v_proj(hidden_states).view(1, 4, 1, 4).transpose(1, 2)
        if rotary_embeddings is not None:
            cos, sin = (embedding.unsqueeze(1) for embedding in rotary_embeddings)
            rotated_query = torch.cat((-query_states[..., 2:], query_states[..., :2]), dim=-1)
            rotated_key = torch.cat((-key_states[..., 2:], key_states[..., :2]), dim=-1)
            query_states = query_states * cos + rotated_query * sin
            key_states = key_states * cos + rotated_key * sin

        expected_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=is_causal,
            enable_gqa=True,
            scale=layer.scaling,
        )
        expected_output = expected_output.transpose(1, 2).reshape(1, 4, 8)
        expected_output = layer.o_proj(expected_output)
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)


@require_torch
def test_decoder_layer_residual_scaling():
    for use_mup in (False, True):
        config = VoxCPM2TextConfig(
            vocab_size=32,
            hidden_size=6,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            kv_channels=4,
            scale_depth=1.4,
            use_mup=use_mup,
            rope_parameters=None,
        )
        layer = VoxCPM2DecoderLayer(config, layer_idx=0)
        expected_keys = {
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        }
        assert set(layer.state_dict()) == expected_keys

        layer.input_layernorm = torch.nn.Identity()
        layer.post_attention_layernorm = torch.nn.Identity()
        layer.self_attn = torch.nn.Identity()
        layer.self_attn.forward = lambda hidden_states, **kwargs: (hidden_states, None)
        layer.mlp = torch.nn.Identity()

        hidden_states = torch.randn(1, 3, 6)
        output = layer(hidden_states, position_embeddings=None, is_causal=False)
        expected_scale = 1.4 / math.sqrt(2) if use_mup else 1.0
        expected_output = hidden_states + hidden_states * expected_scale
        expected_output = expected_output + expected_output * expected_scale
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
        assert layer.residual_scale == expected_scale


@require_torch
def test_rms_normalization_matches_reference():
    layer = VoxCPM2RMSNorm(6, eps=1e-5)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([0.5, 0.75, 1.0, 1.25, 1.5, 2.0]))

    assert list(layer.state_dict()) == ["weight"]
    hidden_states = torch.randn(2, 3, 6)
    variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
    expected_output = layer.weight * (hidden_states * torch.rsqrt(variance + layer.variance_epsilon))
    torch.testing.assert_close(layer(hidden_states), expected_output, rtol=0, atol=0)


@require_torch
def test_rotary_embedding_matches_reference():
    config = VoxCPM2TextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        kv_channels=4,
        max_position_embeddings=8,
        rope_parameters={
            "rope_type": "longrope",
            "rope_theta": 10000.0,
            "long_factor": [1.0, 2.0],
            "short_factor": [1.0, 2.0],
            "original_max_position_embeddings": 8,
        },
    )
    layer = VoxCPM2RotaryEmbedding(config)
    assert not layer.state_dict()

    position_ids = torch.tensor([[0, 1, 3]])
    cosine, sine = layer(torch.zeros(1, 3, 8), position_ids)
    inverse_frequencies = torch.tensor([1.0, 0.01]) / torch.tensor([1.0, 2.0])
    frequencies = position_ids.float().unsqueeze(-1) * inverse_frequencies
    expected_angles = torch.cat((frequencies, frequencies), dim=-1)
    torch.testing.assert_close(cosine, expected_angles.cos(), rtol=0, atol=0)
    torch.testing.assert_close(sine, expected_angles.sin(), rtol=0, atol=0)


@require_torch
def test_transformer_backbone_embeddings_and_cache():
    config = VoxCPM2TextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        kv_channels=4,
        use_mup=False,
        no_rope=True,
        rope_parameters=None,
    )
    config._attn_implementation = "sdpa"
    model = VoxCPM2BackboneModel(config).eval()

    layer_keys = {
        f"layers.{layer_index}.{name}"
        for layer_index in range(2)
        for name in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        )
    }
    expected_keys = {"embed_tokens.weight", "norm.weight", *layer_keys}
    assert set(model.state_dict()) == expected_keys

    input_ids = torch.tensor([[1, 2, 3, 4]])
    inputs_embeds = model.embed_tokens(input_ids)
    token_output = model(input_ids=input_ids).last_hidden_state
    embedding_output = model(inputs_embeds=inputs_embeds).last_hidden_state
    torch.testing.assert_close(token_output, embedding_output, rtol=0, atol=0)

    changed_embeds = inputs_embeds.clone()
    changed_embeds[:, -1] += 10
    changed_causal_output = model(inputs_embeds=changed_embeds).last_hidden_state
    torch.testing.assert_close(changed_causal_output[:, :-1], embedding_output[:, :-1], rtol=0, atol=0)
    bidirectional_output = model(inputs_embeds=inputs_embeds, is_causal=False).last_hidden_state
    changed_bidirectional_output = model(inputs_embeds=changed_embeds, is_causal=False).last_hidden_state
    assert not torch.allclose(changed_bidirectional_output[:, :-1], bidirectional_output[:, :-1])

    prefill = model(inputs_embeds=inputs_embeds[:, :3], use_cache=True)
    cached_output = model(
        inputs_embeds=inputs_embeds[:, 3:], past_key_values=prefill.past_key_values, use_cache=True
    ).last_hidden_state
    torch.testing.assert_close(cached_output, embedding_output[:, -1:], rtol=1e-5, atol=1e-6)

    residual_config = VoxCPM2TextConfig(**{**config.to_dict(), "vocab_size": 0})
    residual_config._attn_implementation = "sdpa"
    residual_model = VoxCPM2BackboneModel(residual_config)
    assert set(residual_model.state_dict()) == expected_keys - {"embed_tokens.weight"}
    with pytest.raises(ValueError, match="inputs_embeds.*vocab_size"):
        residual_model(input_ids=input_ids)


@require_torch
def test_local_encoder_matches_reference_layout():
    config = VoxCPM2Config(
        lm_config={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "kv_channels": 4,
            "max_position_embeddings": 8,
            "use_mup": False,
            "rope_parameters": {
                "rope_type": "longrope",
                "rope_theta": 10000.0,
                "long_factor": [1.0, 1.0],
                "short_factor": [1.0, 1.0],
                "original_max_position_embeddings": 8,
            },
        },
        encoder_config={"hidden_dim": 8, "ffn_dim": 16, "num_heads": 2, "num_layers": 2, "kv_channels": 4},
        feat_dim=4,
        audio_vae_config={"latent_dim": 4},
    )
    config.lm_config._attn_implementation = "sdpa"
    model = VoxCPM2LocalEncoder(config)

    layer_keys = {
        f"encoder.layers.{layer_index}.{name}"
        for layer_index in range(2)
        for name in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        )
    }
    expected_keys = {"special_token", "in_proj.weight", "in_proj.bias", "encoder.norm.weight", *layer_keys}
    assert set(model.state_dict()) == expected_keys

    audio_features = torch.randn(2, 3, 4, 4)
    output = model(audio_features)
    projected_features = model.in_proj(audio_features)
    special_tokens = model.special_token.expand(2, 3, 1, -1)
    encoder_input = torch.cat((special_tokens, projected_features), dim=2).reshape(6, 5, 8)
    expected_output = model.encoder(inputs_embeds=encoder_input, is_causal=False).last_hidden_state[:, 0]
    expected_output = expected_output.reshape(2, 3, 8)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

    output.sum().backward()
    assert model.special_token.grad is not None


@require_torch
def test_local_dit_matches_reference_layout():
    config = VoxCPM2Config(
        lm_config={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "kv_channels": 4,
            "use_mup": False,
            "no_rope": True,
            "rope_parameters": None,
        },
        dit_config={"hidden_dim": 8, "ffn_dim": 16, "num_heads": 2, "num_layers": 2, "kv_channels": 4},
        feat_dim=4,
        audio_vae_config={"latent_dim": 4},
    )
    config.lm_config._attn_implementation = "sdpa"
    model = VoxCPM2LocalDiT(config)

    layer_keys = {
        f"decoder.layers.{layer_index}.{name}"
        for layer_index in range(2)
        for name in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        )
    }
    projection_keys = {
        f"{name}.{parameter}"
        for name in (
            "in_proj",
            "cond_proj",
            "out_proj",
            "time_mlp.linear_1",
            "time_mlp.linear_2",
            "delta_time_mlp.linear_1",
            "delta_time_mlp.linear_2",
        )
        for parameter in ("weight", "bias")
    }
    assert set(model.state_dict()) == {"decoder.norm.weight", *projection_keys, *layer_keys}

    sample = torch.randn(2, 4, 3)
    mu = torch.randn(2, 16)
    timestep = torch.tensor([0.2, 0.7])
    conditioning = torch.randn(2, 4, 2)
    delta_timestep = torch.tensor([0.1, 0.1])
    output = model(sample, mu, timestep, conditioning, delta_timestep)

    sample_hidden_states = model.in_proj(sample.transpose(1, 2))
    conditioning_hidden_states = model.cond_proj(conditioning.transpose(1, 2))
    timestep_hidden_states = model.time_mlp(model.time_embeddings(timestep))
    timestep_hidden_states += model.delta_time_mlp(model.time_embeddings(delta_timestep))
    mu_hidden_states = mu.reshape(2, 2, 8)
    decoder_input = torch.cat(
        (mu_hidden_states, timestep_hidden_states.unsqueeze(1), conditioning_hidden_states, sample_hidden_states),
        dim=1,
    )
    assert decoder_input.shape == (2, 8, 8)
    decoder_output = model.decoder(inputs_embeds=decoder_input, is_causal=False).last_hidden_state
    expected_output = model.out_proj(decoder_output[:, 5:]).transpose(1, 2).contiguous()
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)


@require_torch
def test_conditional_flow_matching_euler_steps():
    config = VoxCPM2Config(
        lm_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "kv_channels": 4,
            "no_rope": True,
            "rope_parameters": None,
        },
        dit_config={"hidden_dim": 8, "ffn_dim": 16, "num_heads": 2, "num_layers": 1, "kv_channels": 4},
        feat_dim=4,
        audio_vae_config={"latent_dim": 4},
    )
    model = VoxCPM2ConditionalFlowMatching(config)
    assert len(model.state_dict()) == 24
    assert all(key.startswith("estimator.") for key in model.state_dict())

    estimator = torch.nn.Identity()
    estimator.forward = (
        lambda sample, mu, timestep, conditioning, delta_timestep: mu[:, :4].unsqueeze(-1).expand_as(sample)
    )
    model.estimator = estimator

    sample = torch.randn(2, 4, 3)
    mu = torch.randn(2, 4)
    conditioning = torch.randn(2, 4, 3)
    timestep_span = torch.tensor([1.0, 0.5, 0.0])
    derivative = mu.unsqueeze(-1).expand_as(sample)

    output = model.solve_euler(sample.clone(), timestep_span, mu, conditioning, cfg_value=1.5, use_cfg_zero_star=False)
    torch.testing.assert_close(output, sample - 1.5 * derivative, rtol=1e-6, atol=1e-6)

    zero_star_output = model.solve_euler(
        sample.clone(), timestep_span, mu, conditioning, cfg_value=1.5, use_cfg_zero_star=True
    )
    torch.testing.assert_close(zero_star_output, sample - 0.75 * derivative, rtol=1e-6, atol=1e-6)

    torch.manual_seed(7)
    initial_sample = torch.randn_like(sample) * 0.5
    torch.manual_seed(7)
    generated_sample = model(
        mu,
        num_inference_steps=2,
        patch_size=3,
        conditioning=conditioning,
        temperature=0.5,
        cfg_value=1.5,
        sway_sampling_coefficient=0.0,
        use_cfg_zero_star=False,
    )
    torch.testing.assert_close(generated_sample, initial_sample - 1.5 * derivative, rtol=1e-6, atol=1e-6)


@require_torch
def test_conditional_flow_matching_generator():
    model = VoxCPM2ConditionalFlowMatching(get_tiny_voxcpm2_config()).eval()
    mu = torch.randn(2, 16)
    conditioning = torch.randn(2, 4, 2)

    first_output = model(
        mu,
        num_inference_steps=1,
        patch_size=2,
        conditioning=conditioning,
        generator=torch.Generator().manual_seed(7),
    )
    repeated_output = model(
        mu,
        num_inference_steps=1,
        patch_size=2,
        conditioning=conditioning,
        generator=torch.Generator().manual_seed(7),
    )
    different_output = model(
        mu,
        num_inference_steps=1,
        patch_size=2,
        conditioning=conditioning,
        generator=torch.Generator().manual_seed(8),
    )

    torch.testing.assert_close(first_output, repeated_output, rtol=0, atol=0)
    assert not torch.equal(first_output, different_output)


@require_torch
def test_conditional_flow_matching_loss():
    config = VoxCPM2Config(
        lm_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "kv_channels": 4,
            "no_rope": True,
            "rope_parameters": None,
        },
        dit_config={
            "hidden_dim": 8,
            "ffn_dim": 16,
            "num_heads": 2,
            "num_layers": 1,
            "kv_channels": 4,
            "cfm_config": {
                "t_scheduler": "uniform",
                "training_cfg_rate": 0.0,
                "noise_cond_prob_range": (0.0, 0.0),
                "noise_cond_scale": 0.0,
            },
        },
        feat_dim=4,
        audio_vae_config={"latent_dim": 4},
    )
    model = VoxCPM2ConditionalFlowMatching(config)
    estimator = torch.nn.Identity()
    estimator.forward = lambda sample, mu, timestep, conditioning, delta_timestep: torch.zeros_like(sample)
    model.estimator = estimator

    target = torch.randn(2, 4, 3)
    mu = torch.randn(2, 4)
    conditioning = torch.randn(2, 4, 3)
    target_mask = torch.tensor([[[1.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]]])

    torch.manual_seed(11)
    torch.rand(2)
    torch.randn_like(conditioning)
    torch.rand(2)
    torch.rand(2)
    torch.rand(2)
    noise = torch.randn_like(target)
    target_velocity = noise - target
    losses = torch.nn.functional.mse_loss(torch.zeros_like(target_velocity), target_velocity, reduction="none").mean(
        dim=1
    )
    expected_loss = (losses * target_mask.squeeze(1)).sum() / target_mask.sum()

    torch.manual_seed(11)
    loss = model.compute_loss(target, mu, conditioning, target_mask=target_mask)
    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
