# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import re
import tempfile
import unittest
from pathlib import Path

import pytest
import requests
from parameterized import parameterized

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
    Phi4MultimodalAudioConfig,
    Phi4MultimodalConfig,
    Phi4MultimodalForCausalLM,
    Phi4MultimodalModel,
    Phi4MultimodalVisionConfig,
    is_torch_available,
    is_vision_available,
)
from transformers.audio_utils import load_audio
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_deterministic_for_xpu,
    require_peft,
    require_torch,
    require_torch_large_accelerator,
    require_torchcodec,
    skip_tpu,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...model_tester_config import ModelTesterConfigMixin
from ...test_configuration_common import ConfigTester
from ...test_image_processing_common import load_test_image
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image


class Phi4MultimodalModelTester(ModelTesterConfigMixin):
    defaults = {}

    small_config_overrides = {
        "batch_size": 2,
        # text tokens only: `get_small_config_defaults` adds the image and audio tokens
        "seq_length": 12,
        "image_seq_length": 13,
        "audio_seq_length": 8,
        "is_training": True,
        "num_hidden_layers": 2,
        "vocab_size": 49,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 4096,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "pad_token_id": 0,
        "image_token_id": 1,
        "audio_token_id": 2,
        "image_size": 16,
        "audio_size": 12,
        "num_channels": 3,
    }

    def __init__(
        self,
        parent,
        audio_config=Phi4MultimodalAudioConfig(
            num_blocks=2,
            hidden_size=32,
            num_attention_heads=8,
            intermediate_size=48,
            ext_pw_out_channel=32,
            depthwise_separable_out_channel=32,
            nemo_conv_channels=32,
            audio_token_id=2,
            initializer_range=1e-5,
        ),
        vision_config=Phi4MultimodalVisionConfig(
            num_hidden_layers=2,
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=8,
            image_size=16,
            patch_size=4,
            crop_size=16,
            image_token_id=1,
            initializer_range=1e-5,
        ),
        **kwargs,
    ):
        self.parent = parent
        for key, value in self.get_small_config_defaults(**kwargs).items():
            setattr(self, key, value)
        self.audio_config = copy.deepcopy(audio_config)
        self.vision_config = copy.deepcopy(vision_config)

    @classmethod
    def get_small_config_defaults(cls, **kwargs):
        values = super().get_small_config_defaults(**kwargs)
        values["seq_length"] = values["seq_length"] + values["image_seq_length"] + values["audio_seq_length"]
        return values

    def get_config(self):
        return Phi4MultimodalConfig(
            max_position_embeddings=self.max_position_embeddings,
            num_hidden_layers=self.num_hidden_layers,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            vision_config=self.vision_config,
            audio_config=self.audio_config,
        )

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 3) + 3

        # 16x16 image with patch_size=4 -> 4x4 patches -> 2x2 pooled
        image_pixel_values = floats_tensor([self.batch_size, 2, self.num_channels, self.image_size, self.image_size])
        image_attention_mask = torch.ones(self.batch_size, 2, 4, 4, device=torch_device)
        image_sizes = torch.tensor(
            [[self.image_size, self.image_size]] * self.batch_size, dtype=torch.long, device=torch_device
        )

        # 61 frames -> ceil(61/8)=8 audio tokens
        audio_input_features = floats_tensor([self.batch_size, 61, 80])
        audio_embed_sizes = torch.tensor(
            [self.audio_seq_length] * self.batch_size, dtype=torch.long, device=torch_device
        )

        input_ids[-1, 0] = self.pad_token_id
        image_start = 2
        audio_start = image_start + self.image_seq_length
        input_ids[:, image_start:audio_start] = self.image_token_id
        input_ids[:, audio_start : audio_start + self.audio_seq_length] = self.audio_token_id

        attention_mask = torch.ones_like(input_ids)
        attention_mask[-1, 0] = 0
        config = self.get_config()

        return (
            config,
            input_ids,
            attention_mask,
            image_pixel_values,
            image_attention_mask,
            image_sizes,
            audio_input_features,
            audio_embed_sizes,
        )

    def prepare_config_and_inputs_for_common(self):
        (
            config,
            input_ids,
            attention_mask,
            image_pixel_values,
            image_attention_mask,
            image_sizes,
            audio_input_features,
            audio_embed_sizes,
        ) = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_pixel_values": image_pixel_values,
            "image_attention_mask": image_attention_mask,
            "image_sizes": image_sizes,
            "audio_input_features": audio_input_features,
            "audio_embed_sizes": audio_embed_sizes,
        }
        return config, inputs_dict


@require_torch
class Phi4MultimodalModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `Phi4Multimodal`.
    """

    model_tester_class = Phi4MultimodalModelTester

    all_model_classes = (Phi4MultimodalForCausalLM, Phi4MultimodalModel) if is_torch_available() else ()

    _is_composite = True
    test_torch_exportable = False  # data-dependent multimodal placeholder mask

    def setUp(self):
        self.model_tester = Phi4MultimodalModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Phi4MultimodalConfig)

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing(self):
        super().test_training_gradient_checkpointing()

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        super().test_training_gradient_checkpointing_use_reentrant_false()

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        super().test_training_gradient_checkpointing_use_reentrant_true()

    @unittest.skip(reason="Test is only for old attention format")
    def test_sdpa_can_dispatch_composite_models(self):
        pass

    @unittest.skip(reason="Static cache supported only for text-only inputs (not images or audios)")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip(reason="Static cache supported only for text-only inputs (not images or audios)")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip(
        reason="Supported only for text-only inputs (otherwise dynamic control flows for multimodal inputs)"
    )
    def test_generate_compilation_all_outputs(self):
        pass

    @unittest.skip(
        reason="Supported only for text-only inputs (otherwise dynamic control flows for multimodal inputs)"
    )
    @pytest.mark.torch_compile_test
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip(reason="`image_attention_mask` has a specific shape")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip(reason="`image_attention_mask` has a specific shape")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip(reason="`image_attention_mask` has a specific shape")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @unittest.skip(reason="Cannot unpad inputs for all modalities so easily")
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(reason="Dynamo error")
    def test_flex_attention_with_grads(self):
        pass

    def test_attention_backends_and_audio_masking(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        eager_config = copy.deepcopy(config)
        eager_config._attn_implementation = "eager"
        sdpa_config = copy.deepcopy(config)
        sdpa_config._attn_implementation = "sdpa"

        eager_model = Phi4MultimodalForCausalLM(eager_config).to(torch_device).eval()
        sdpa_model = Phi4MultimodalForCausalLM(sdpa_config).to(torch_device).eval()
        sdpa_model.load_state_dict(eager_model.state_dict())

        audio_features = floats_tensor([2, 600, 80])
        audio_attention_mask = torch.zeros((2, 600), dtype=torch.bool, device=torch_device)
        audio_attention_mask[0, :600] = True
        audio_attention_mask[1, :320] = True
        audio_embed_sizes = torch.tensor([75, 40], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            eager_img = eager_model.get_image_features(
                inputs_dict["image_pixel_values"],
                image_sizes=inputs_dict["image_sizes"],
                image_attention_mask=inputs_dict["image_attention_mask"],
            )
            sdpa_img = sdpa_model.get_image_features(
                inputs_dict["image_pixel_values"],
                image_sizes=inputs_dict["image_sizes"],
                image_attention_mask=inputs_dict["image_attention_mask"],
            )
            for e_feat, s_feat in zip(eager_img.pooler_output, sdpa_img.pooler_output):
                torch.testing.assert_close(e_feat, s_feat, atol=1e-4, rtol=1e-4)

            eager_aud = eager_model.get_audio_features(
                audio_features,
                audio_embed_sizes=audio_embed_sizes,
                audio_attention_mask=audio_attention_mask,
            )
            sdpa_aud = sdpa_model.get_audio_features(
                audio_features,
                audio_embed_sizes=audio_embed_sizes,
                audio_attention_mask=audio_attention_mask,
            )
            self.assertEqual([t.shape[0] for t in eager_aud.pooler_output], [75, 40])
            for e_feat, s_feat in zip(eager_aud.pooler_output, sdpa_aud.pooler_output):
                torch.testing.assert_close(e_feat, s_feat, atol=1e-4, rtol=1e-4)

            perturbed_audio = audio_features.clone()
            perturbed_audio[1, 320:] = 100.0
            perturbed_aud = sdpa_model.get_audio_features(
                perturbed_audio,
                audio_embed_sizes=audio_embed_sizes,
                audio_attention_mask=audio_attention_mask,
            )
            torch.testing.assert_close(perturbed_aud.pooler_output[1], sdpa_aud.pooler_output[1], atol=1e-4, rtol=1e-4)

            eager_out = eager_model(**inputs_dict)
            sdpa_out = sdpa_model(**inputs_dict)
            valid_mask = inputs_dict["attention_mask"].bool()
            torch.testing.assert_close(eager_out.logits[valid_mask], sdpa_out.logits[valid_mask], atol=1e-4, rtol=1e-4)

        audio_encoder = eager_model.model.get_encoder("audio")
        self.assertFalse(audio_encoder._supports_flash_attn)
        self.assertFalse(audio_encoder._supports_flex_attn)
        self.assertTrue(audio_encoder._supports_sdpa)

        mixed_config = copy.deepcopy(config)
        mixed_config._attn_implementation = {"": "sdpa", "vision_config": "eager", "audio_config": "sdpa"}
        mixed_model = Phi4MultimodalForCausalLM(mixed_config).to(torch_device).eval()
        self.assertEqual(mixed_model.config._attn_implementation, "sdpa")
        self.assertEqual(mixed_model.config.vision_config._attn_implementation, "eager")
        self.assertEqual(mixed_model.config.audio_config._attn_implementation, "sdpa")

    def test_precomputed_mm_encoder_outputs_and_audio_projection_mode(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.audio_config.initializer_range = 0.05
        model = Phi4MultimodalForCausalLM(config).to(torch_device).eval()

        with torch.no_grad():
            image_outputs = model.get_image_features(
                inputs_dict["image_pixel_values"],
                image_sizes=inputs_dict["image_sizes"],
                image_attention_mask=inputs_dict["image_attention_mask"],
            )
            audio_vision_outputs = model.get_audio_features(
                inputs_dict["audio_input_features"],
                audio_embed_sizes=inputs_dict["audio_embed_sizes"],
                audio_projection_mode="vision",
            )
            audio_speech_outputs = model.get_audio_features(
                inputs_dict["audio_input_features"],
                audio_embed_sizes=inputs_dict["audio_embed_sizes"],
                audio_projection_mode="speech",
            )

            raw_mixed = model(**inputs_dict)
            precomp_mixed = model(
                input_ids=inputs_dict["input_ids"],
                attention_mask=inputs_dict["attention_mask"],
                mm_encoder_outputs={"image": image_outputs, "audio": audio_vision_outputs},
            )
            torch.testing.assert_close(raw_mixed.logits, precomp_mixed.logits, atol=1e-5, rtol=1e-5)

            raw_mixed_speech = model(**inputs_dict, audio_projection_mode="speech")
            precomp_mixed_speech = model(
                input_ids=inputs_dict["input_ids"],
                attention_mask=inputs_dict["attention_mask"],
                mm_encoder_outputs={"image": image_outputs, "audio": audio_speech_outputs},
            )
            torch.testing.assert_close(raw_mixed_speech.logits, precomp_mixed_speech.logits, atol=1e-5, rtol=1e-5)
            self.assertFalse(torch.allclose(raw_mixed.logits, raw_mixed_speech.logits))

            audio_only_ids = inputs_dict["input_ids"].clone()
            audio_only_ids[audio_only_ids == config.image_token_id] = 5
            raw_audio = model(
                input_ids=audio_only_ids,
                attention_mask=inputs_dict["attention_mask"],
                audio_input_features=inputs_dict["audio_input_features"],
                audio_embed_sizes=inputs_dict["audio_embed_sizes"],
            )
            precomp_audio = model(
                input_ids=audio_only_ids,
                attention_mask=inputs_dict["attention_mask"],
                mm_encoder_outputs={"audio": audio_speech_outputs},
            )
            torch.testing.assert_close(raw_audio.logits, precomp_audio.logits, atol=1e-5, rtol=1e-5)

            raw_gen = model.generate(**inputs_dict, max_new_tokens=4, do_sample=False)
            precomp_gen = model.generate(
                input_ids=inputs_dict["input_ids"],
                attention_mask=inputs_dict["attention_mask"],
                mm_encoder_outputs={"image": image_outputs, "audio": audio_vision_outputs},
                max_new_tokens=4,
                do_sample=False,
            )
            torch.testing.assert_close(raw_gen, precomp_gen)

            # Multi-audio prompt (2 clips of different lengths in 1 prompt) under beam search
            multi_audio_feats = floats_tensor([2, 61, 80])
            multi_audio_mask = torch.zeros((2, 61), dtype=torch.bool, device=torch_device)
            multi_audio_mask[0, :61] = True
            multi_audio_mask[1, :35] = True
            multi_audio_sizes = torch.tensor([8, 5], dtype=torch.long, device=torch_device)
            multi_audio_ids = torch.full((1, 18), 5, dtype=torch.long, device=torch_device)
            multi_audio_ids[0, 2:15] = config.audio_token_id
            multi_audio_outputs = model.get_audio_features(
                multi_audio_feats,
                audio_embed_sizes=multi_audio_sizes,
                audio_attention_mask=multi_audio_mask,
                audio_projection_mode="speech",
            )
            beam_kwargs = {"max_new_tokens": 3, "num_beams": 2, "do_sample": False}
            raw_beam = model.generate(
                input_ids=multi_audio_ids,
                attention_mask=torch.ones_like(multi_audio_ids),
                audio_input_features=multi_audio_feats,
                audio_embed_sizes=multi_audio_sizes,
                audio_attention_mask=multi_audio_mask,
                **beam_kwargs,
            )
            precomp_beam = model.generate(
                input_ids=multi_audio_ids,
                attention_mask=torch.ones_like(multi_audio_ids),
                mm_encoder_outputs={"audio": multi_audio_outputs},
                **beam_kwargs,
            )
            torch.testing.assert_close(raw_beam, precomp_beam)
            self.assertEqual(len(multi_audio_outputs.pooler_output), 2)

            orig_img_len = len(image_outputs.pooler_output)
            orig_aud_len = len(audio_vision_outputs.pooler_output)
            model.generate(
                input_ids=inputs_dict["input_ids"],
                attention_mask=inputs_dict["attention_mask"],
                mm_encoder_outputs={"image": image_outputs, "audio": audio_vision_outputs},
                **beam_kwargs,
            )
            self.assertEqual(len(image_outputs.pooler_output), orig_img_len)
            self.assertEqual(len(audio_vision_outputs.pooler_output), orig_aud_len)

            model.config.return_dict = False
            try:
                tuple_out = model(**inputs_dict)
                self.assertIsInstance(tuple_out, tuple)
            finally:
                model.config.return_dict = True

        with self.assertRaises(ValueError):
            model(**inputs_dict, mm_encoder_outputs={"image": image_outputs})
        with self.assertRaises(ValueError):
            model(**inputs_dict, mm_encoder_outputs={"audio": audio_vision_outputs})

    def test_nested_output_capturing_and_embedding_dropout_semantics(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.embd_pdrop = 0.5
        config._attn_implementation = "eager"
        model = Phi4MultimodalForCausalLM(config).to(torch_device).eval()

        with torch.no_grad():
            img_before = model.get_image_features(
                inputs_dict["image_pixel_values"],
                image_sizes=inputs_dict["image_sizes"],
                image_attention_mask=inputs_dict["image_attention_mask"],
                output_hidden_states=True,
                output_attentions=True,
            )
            aud_before = model.get_audio_features(
                inputs_dict["audio_input_features"],
                audio_embed_sizes=inputs_dict["audio_embed_sizes"],
                output_hidden_states=True,
                output_attentions=True,
            )
            full_1 = model(**inputs_dict, output_hidden_states=True, output_attentions=True)
            full_2 = model(**inputs_dict, output_hidden_states=True, output_attentions=True)
            img_after = model.get_image_features(
                inputs_dict["image_pixel_values"],
                image_sizes=inputs_dict["image_sizes"],
                image_attention_mask=inputs_dict["image_attention_mask"],
                output_hidden_states=True,
                output_attentions=True,
            )
            aud_after = model.get_audio_features(
                inputs_dict["audio_input_features"],
                audio_embed_sizes=inputs_dict["audio_embed_sizes"],
                output_hidden_states=True,
                output_attentions=True,
            )

        self.assertEqual(len(img_after.hidden_states), config.vision_config.num_hidden_layers + 1)
        self.assertEqual(len(aud_after.hidden_states), config.audio_config.num_blocks + 1)
        self.assertEqual(len(full_2.hidden_states), config.num_hidden_layers + 1)
        torch.testing.assert_close(full_1.logits, full_2.logits)
        torch.testing.assert_close(img_before.pooler_output[0], img_after.pooler_output[0])
        torch.testing.assert_close(aud_before.pooler_output[0], aud_after.pooler_output[0])

        # Default get_image_features skips the unused final vision layer, post_layernorm, and head;
        # requesting hidden_states or attentions preserves the full vision tower
        vision_model = model.get_encoder("image")
        calls = {"last_layer": 0, "post_layernorm": 0, "head": 0}
        handles = [
            vision_model.encoder.layers[-1].register_forward_hook(
                lambda *_: calls.__setitem__("last_layer", calls["last_layer"] + 1)
            ),
            vision_model.post_layernorm.register_forward_hook(
                lambda *_: calls.__setitem__("post_layernorm", calls["post_layernorm"] + 1)
            ),
            vision_model.head.register_forward_hook(lambda *_: calls.__setitem__("head", calls["head"] + 1)),
        ]
        try:
            with torch.no_grad():
                fast_img = model.get_image_features(
                    inputs_dict["image_pixel_values"],
                    image_sizes=inputs_dict["image_sizes"],
                    image_attention_mask=inputs_dict["image_attention_mask"],
                )
                self.assertEqual(calls, {"last_layer": 0, "post_layernorm": 0, "head": 0})
                self.assertIsNone(fast_img.hidden_states)
                self.assertIsNone(fast_img.attentions)
                torch.testing.assert_close(fast_img.pooler_output[0], img_after.pooler_output[0])

                attn_img = model.get_image_features(
                    inputs_dict["image_pixel_values"],
                    image_sizes=inputs_dict["image_sizes"],
                    image_attention_mask=inputs_dict["image_attention_mask"],
                    output_attentions=True,
                )
                self.assertEqual(calls, {"last_layer": 1, "post_layernorm": 1, "head": 1})
                self.assertEqual(len(attn_img.attentions), config.vision_config.num_hidden_layers)
                self.assertIsNone(attn_img.hidden_states)
                torch.testing.assert_close(attn_img.pooler_output[0], img_after.pooler_output[0])
        finally:
            for handle in handles:
                handle.remove()

        # Preserve upstream semantics: text-only and raw inputs_embeds skip embed_dropout, multimodal applies it
        model.train()
        text_ids = torch.tensor([[3, 4, 5, 6]], device=torch_device)
        raw_embeds = model.get_input_embeddings()(text_ids).detach()
        with torch.no_grad():
            text_1 = model(input_ids=text_ids).logits
            text_2 = model(input_ids=text_ids).logits
            emb_1 = model(inputs_embeds=raw_embeds).logits
            emb_2 = model(inputs_embeds=raw_embeds).logits
            mm_1 = model(**inputs_dict).logits
            mm_2 = model(**inputs_dict).logits
        torch.testing.assert_close(text_1, text_2)
        torch.testing.assert_close(emb_1, emb_2)
        self.assertFalse(torch.allclose(mm_1, mm_2))

    def test_longrope_frequency_switch_and_generate_boundary(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        seq_len = inputs_dict["input_ids"].shape[1]
        config.original_max_position_embeddings = seq_len + 1
        config.max_position_embeddings = seq_len + 32
        config.rope_parameters = {
            "rope_type": "longrope",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.75,
            "short_factor": [1.0, 1.0, 1.0],
            "long_factor": [1.0, 2.0, 4.0],
            "original_max_position_embeddings": seq_len + 1,
        }
        model = Phi4MultimodalForCausalLM(config).to(torch_device).eval()

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": 4,
                "do_sample": False,
                "suppress_tokens": [config.image_token_id, config.audio_token_id],
            }
            gen_cached = model.generate(**inputs_dict, **gen_kwargs, use_cache=True)
            gen_uncached = model.generate(**inputs_dict, **gen_kwargs, use_cache=False)
            torch.testing.assert_close(gen_cached, gen_uncached)

    @require_peft
    def test_peft_independent_vision_speech_adapters(self):
        from safetensors.torch import save_file

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Phi4MultimodalForCausalLM(config).to(torch_device).eval()
        target_regex = r"model.layers.\d+.((self_attn.(qkv|o)_proj)|(mlp.(gate_up|down)_proj))"

        def write_adapter(adapter_dir: Path, fill_value: float):
            adapter_dir.mkdir(parents=True, exist_ok=True)
            adapter_cfg = {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": target_regex,
            }
            (adapter_dir / "adapter_config.json").write_text(json.dumps(adapter_cfg))
            sd = {}
            for name, param in model.named_parameters():
                if re.match(target_regex + r"\.weight$", name):
                    out_f, in_f = param.shape
                    prefix = name[: -len(".weight")]
                    sd[f"{prefix}.lora_A.weight"] = torch.full((4, in_f), fill_value)
                    sd[f"{prefix}.lora_B.weight"] = torch.full((out_f, 4), fill_value)
            save_file(sd, str(adapter_dir / "adapter_model.safetensors"))

        with torch.no_grad():
            base_logits = model(**inputs_dict).logits.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            write_adapter(tmp_path / "vision-lora", 0.05)
            write_adapter(tmp_path / "speech-lora", -0.04)

            model.load_adapter(str(tmp_path / "vision-lora"), adapter_name="vision")
            model.load_adapter(str(tmp_path / "speech-lora"), adapter_name="speech")

        for name, _ in model.named_parameters():
            if "audio_embed" in name:
                self.assertNotIn("lora_", name)

        with torch.no_grad():
            model.set_adapter("vision")
            vision_logits = model(**inputs_dict).logits.clone()
            model.set_adapter("speech")
            speech_logits = model(**inputs_dict).logits.clone()
            model.disable_adapters()
            disabled_logits = model(**inputs_dict).logits.clone()
            model.enable_adapters()
            model.set_adapter("vision")
            vision_logits_again = model(**inputs_dict).logits.clone()

        torch.testing.assert_close(disabled_logits, base_logits, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(vision_logits, vision_logits_again, atol=1e-5, rtol=1e-5)
        self.assertFalse(torch.allclose(vision_logits, base_logits))
        self.assertFalse(torch.allclose(speech_logits, base_logits))
        self.assertFalse(torch.allclose(vision_logits, speech_logits))


@require_torch
@slow
class Phi4MultimodalIntegrationTest(unittest.TestCase):
    checkpoint_path = "microsoft/Phi-4-multimodal-instruct"
    revision = "refs/pr/70"
    image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/australia.jpg"
    audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/f2641_0_throatclearing.wav"

    def setUp(self):
        # Currently, the Phi-4 checkpoint on the hub is not working with the latest Phi-4 code, so the slow integration tests
        # won't pass without using the correct revision (refs/pr/70)
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_path, revision=self.revision)
        self.generation_config = GenerationConfig(max_new_tokens=20, do_sample=False)
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"
        self.end_token = "<|end|>"
        self.image = Image.open(requests.get(self.image_url, stream=True).raw)
        self.sampling_rate = 16000
        self.audio = load_audio(self.audio_url, sampling_rate=self.sampling_rate)

        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_text_only_generation(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path, revision=self.revision, dtype=torch.float16, device_map=torch_device
        )

        prompt = f"{self.user_token}What is the answer for 1+1? Explain it.{self.end_token}{self.assistant_token}"
        inputs = self.processor(text=prompt, images=None, return_tensors="pt").to(torch_device)

        output = model.generate(
            **inputs,
            generation_config=self.generation_config,
        )
        output = output[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        EXPECTED_RESPONSE = "The answer for 1+1 is 2. This is because when you add one to another"

        self.assertEqual(response, EXPECTED_RESPONSE)

    @require_deterministic_for_xpu
    @skip_tpu("test aborts")
    def test_vision_text_generation(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path, revision=self.revision, dtype=torch.float16, device_map=torch_device
        )

        prompt = f"{self.user_token}<|image|>What is shown in this image?{self.end_token}{self.assistant_token}"
        inputs = self.processor(text=prompt, images=self.image, return_tensors="pt").to(torch_device)

        output = model.generate(
            **inputs,
            generation_config=self.generation_config,
        )
        output = output[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        EXPECTED_RESPONSES = Expectations(
            {
                ("cuda", 7): 'The image shows a vibrant scene at a traditional Chinese-style street entrance, known as a "gate"',
                ("cuda", 8): 'The image shows a vibrant scene at a street intersection in a city with a Chinese-influenced architectural',
                ("xpu", 5): 'The image shows a vibrant scene at a street intersection in a city with a Chinese-influenced architectural',
            }
        )  # fmt: skip
        EXPECTED_RESPONSE = EXPECTED_RESPONSES.get_expectation()

        self.assertEqual(response, EXPECTED_RESPONSE)

    @require_torch_large_accelerator
    def test_multi_image_vision_text_generation(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path, revision=self.revision, dtype=torch.float16, device_map=torch_device
        )

        images = []
        placeholder = ""
        for i in range(1, 5):
            url = f"https://huggingface.co/datasets/hf-internal-testing/transformers-synthetic-assets/resolve/main/images/azure_slide_{i}.jpg"
            images.append(load_test_image(url))
            placeholder += "<|image|>"

        prompt = f"{self.user_token}{placeholder}Summarize the deck of slides.{self.end_token}{self.assistant_token}"
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(torch_device)

        output = model.generate(
            **inputs,
            generation_config=self.generation_config,
        )
        output = output[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        EXPECTED_RESPONSE = "This presentation provides an introduction to cloud computing, focusing on Microsoft Azure services. It outlines a four-part"

        self.assertEqual(response, EXPECTED_RESPONSE)

    @require_torchcodec
    def test_audio_text_generation(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path, revision=self.revision, dtype=torch.float16, device_map=torch_device
        )

        prompt = f"{self.user_token}<|audio|>What is happening in this audio?{self.end_token}{self.assistant_token}"
        inputs = self.processor(
            text=prompt, audio=self.audio, sampling_rate=self.sampling_rate, return_tensors="pt"
        ).to(torch_device)

        output = model.generate(
            **inputs,
            generation_config=self.generation_config,
        )
        output = output[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Yes, it is truly the expected response... Even though the model correctly treats the audio file
        EXPECTED_RESPONSE = "I'm sorry, but I can't listen to audio. However, if you describe the audio to me,"

        self.assertEqual(response, EXPECTED_RESPONSE)
