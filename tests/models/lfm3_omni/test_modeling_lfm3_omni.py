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
"""Mixed-modality, checkpoint and training contracts for LFM3-Omni."""

import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from huggingface_hub.errors import StrictDataclassClassValidationError
from PIL import Image
from safetensors import safe_open

from transformers import (
    AutoConfig,
    AutoModelForMultimodalLM,
    AutoProcessor,
    Lfm3OmniConfig,
    Lfm3OmniForConditionalGeneration,
    Lfm3OmniModel,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    require_accelerate,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_modeling_common import ModelTesterMixin
from ..lfm3_audio.test_modeling_lfm3_audio import Boost, Lfm3AudioBehaviorTestMixin
from ..lfm3_audio.test_modeling_lfm3_audio import tiny_config as audio_config


if is_torch_available():
    import torch

    from transformers import LogitsProcessorList


def tiny_config():
    values = audio_config().to_dict()
    values.pop("model_type")
    values["vision_config"] = {
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "layer_types": ["conv", "full_attention"],
        "patch_size": 2,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
    }
    values.update(image_token_id=249, video_token_id=248, projector_hidden_size=32)
    return Lfm3OmniConfig(**values)


def mixed_inputs():
    return {
        "input_ids": torch.tensor([[1, 4, 249, 250, 250, 8, 251, 251, 9]], device=torch_device),
        "pixel_values": torch.randn(4, 24, device=torch_device),
        "image_grid_thw": torch.tensor([[1, 2, 2]], device=torch_device),
        "input_features": torch.randn(1, 16, 9, device=torch_device),
        "feature_attention_mask": torch.ones(1, 9, device=torch_device, dtype=torch.long),
        "audio_out_codes": torch.randint(0, 6, (2, 4), device=torch_device),
    }


@require_torch
class Lfm3OmniModelTest(Lfm3AudioBehaviorTestMixin, unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = Lfm3OmniForConditionalGeneration(tiny_config()).to(torch_device).eval()
        self.inputs = mixed_inputs()
        # the released checkpoints ask for the output object, which carries the speech frames
        self.model.generation_config.return_dict_in_generate = True

    def test_modalities_match_explicit_embedding_assembly(self):
        model, inputs = self.model, self.inputs
        with torch.no_grad():
            embeds = model.get_input_embeddings()(inputs["input_ids"])
            embeds[:, 2] = torch.cat(
                model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"]).pooler_output
            )
            embeds[:, 3:5] = model.get_audio_features(
                inputs["input_features"], inputs["feature_attention_mask"]
            ).pooler_output
            embeds[:, 6:8] = model.model.get_audio_out_embeds(inputs["audio_out_codes"])
            expected = model.lm_head(model.language_model(inputs_embeds=embeds, use_cache=False).last_hidden_state)
            actual = model(**inputs, use_cache=False).logits
        torch.testing.assert_close(actual, expected)

    def test_mixed_prefill_cached_decode_matches_full_forward(self):
        with torch.no_grad():
            prefill = self.model(**self.inputs)
            tokens = torch.tensor([[11, 12, 13]], device=torch_device)
            decoded = self.model(input_ids=tokens, past_key_values=prefill.past_key_values)
            full = {**self.inputs, "input_ids": torch.cat((self.inputs["input_ids"], tokens), dim=1)}
            expected = self.model(**full, use_cache=False).logits[:, -3:]
        torch.testing.assert_close(decoded.logits, expected, rtol=1e-4, atol=1e-5)

    def test_generation_encodes_images_once_and_matches_cache_free(self):
        config = self.model.config
        encoder = self.model.model.get_image_features
        with (
            torch.no_grad(),
            # `generate` reads the encoder's signature to pick its inputs, which a bare mock would hide
            patch.object(self.model.model, "get_image_features", autospec=True, side_effect=encoder) as encode,
        ):
            output = self.model.generate(**self.inputs, max_new_tokens=4, generation_mode="text")
            # `generate` pre-encodes the images once and replays them from the cache
            self.assertEqual(encode.call_count, 1)
            encoded = self.model._prepare_multimodal_encoder_kwargs_for_generation(dict(self.inputs))
            with self.assertRaisesRegex(ValueError, "You cannot pass both: raw pixels and pre-computed embeddings"):
                self.model.generate(**self.inputs, mm_encoder_outputs=encoded["mm_encoder_outputs"], max_new_tokens=1)
            ids = self.inputs["input_ids"].clone()
            for _ in range(4):
                logits = self.model(**{**self.inputs, "input_ids": ids}, use_cache=False).logits[:, -1]
                # the text mode forbids the speech markers, and the media placeholders are never sampled as text
                logits[
                    :, [config.audio_start_token_id, config.mixed_start_token_id, *self.model._placeholder_token_ids]
                ] = -float("inf")
                ids = torch.cat((ids, logits.argmax(-1, keepdim=True)), dim=-1)
                if ids[0, -1].item() in (config.text_end_token_id, config.text_config.eos_token_id):
                    break
        torch.testing.assert_close(output.sequences, ids)

    def test_canonical_checkpoint_roundtrip(self):
        with torch.no_grad():
            expected = self.model(**self.inputs, use_cache=False).logits
        with tempfile.TemporaryDirectory() as directory:
            self.model.save_pretrained(directory)
            with safe_open(f"{directory}/model.safetensors", framework="pt") as checkpoint:
                keys = set(checkpoint.keys())
            # the checkpoint is written in the released layout: LFM2-named encoder attention, per-codebook embeddings
            attention = "model.audio_encoder.lfm.layers.1.self_attn.{}.weight"
            module_only = {"lm_head.weight", "model.depth_decoder.codebook_embedding"}
            module_only |= {"model.depth_decoder.codebook_norm.weight"}
            module_only |= {attention.format(name) for name in ("o_proj", "q_norm", "k_norm")}
            released = {attention.format(name) for name in ("out_proj", "q_layernorm", "k_layernorm")}
            released |= {
                f"model.depth_decoder.depth_embeddings.{index}.{name}.weight"
                for index in range(self.model.config.num_codebooks)
                for name in ("embedding", "embedding_norm")
            }
            self.assertEqual(keys, (set(self.model.state_dict()) - module_only) | released)
            self.assertIsInstance(AutoConfig.from_pretrained(directory), Lfm3OmniConfig)
            model, info = AutoModelForMultimodalLM.from_pretrained(directory, output_loading_info=True)
            self.assertFalse(any(info.values()), info)
            for name, value in self.model.state_dict().items():
                torch.testing.assert_close(model.state_dict()[name].to(value.device), value, rtol=0, atol=0)
            with torch.no_grad():
                actual = model.to(torch_device).eval()(**self.inputs, use_cache=False).logits
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_input_modalities_are_declared(self):
        self.assertEqual(self.model.input_modalities, ("image", "video", "audio", "text"))

    def test_video_patches_reach_the_language_model(self):
        config = self.model.config
        inputs = dict(self.inputs)
        # one video of two temporal patches on a 2x2 grid: 2 placeholder groups of one token after merging
        video_ids = torch.tensor([[config.video_token_id, config.video_token_id]], device=torch_device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], video_ids], 1)
        inputs["pixel_values_videos"] = torch.randn(8, 24, device=torch_device)
        inputs["video_grid_thw"] = torch.tensor([[2, 2, 2]], device=torch_device)
        model = self.model
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits
            features = torch.cat(
                model.get_video_features(inputs["pixel_values_videos"], inputs["video_grid_thw"]).pooler_output
            )
            self.assertEqual(tuple(features.shape), (2, config.text_config.hidden_size))
            # the video features must land on their placeholders, like the other modalities
            embeds = model.get_input_embeddings()(inputs["input_ids"])
            embeds[:, 2] = torch.cat(
                model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"]).pooler_output
            )
            embeds[:, 3:5] = model.get_audio_features(
                inputs["input_features"], inputs["feature_attention_mask"]
            ).pooler_output
            embeds[:, 6:8] = model.model.get_audio_out_embeds(inputs["audio_out_codes"])
            embeds[:, -2:] = features
            expected = model.lm_head(model.language_model(inputs_embeds=embeds, use_cache=False).last_hidden_state)
        torch.testing.assert_close(logits, expected)
        with self.assertRaises(ValueError):
            self.model(**{**inputs, "video_grid_thw": None}, use_cache=False)

    @require_flash_attn
    @require_torch_gpu
    def test_flash_attention_runs_the_vision_tower(self):
        with tempfile.TemporaryDirectory() as directory:
            self.model.save_pretrained(directory)
            model = Lfm3OmniForConditionalGeneration.from_pretrained(
                directory, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
            ).to(torch_device)
        self.assertEqual(model.model.vision_tower.config._attn_implementation, "flash_attention_2")
        with torch.no_grad():
            features = model.model.get_image_features(
                self.inputs["pixel_values"].to(torch.bfloat16), self.inputs["image_grid_thw"]
            ).last_hidden_state
        self.assertTrue(torch.isfinite(features).all())

    def test_mixed_training_reaches_every_component(self):
        labels = self.inputs["input_ids"].clone()
        labels[labels >= 249] = -100
        self.model.train()
        output = self.model(**self.inputs, labels=labels, audio_labels=self.inputs["audio_out_codes"], use_cache=False)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        for name in (
            "model.language_model.layers.0.conv.in_proj.weight",
            "model.audio_encoder.downsample.conv_in.weight",
            "model.audio_projector.linear_1.weight",
            "model.vision_tower.patchifier.weight",
            "model.multi_modal_projector.linear_1.weight",
            "model.codebook_embedding.weight",
            "model.depth_decoder.depth_linear.weight",
        ):
            gradient = self.model.get_parameter(name).grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)

    def test_config_rejects_out_of_range_speech_fields(self):
        values = tiny_config().to_dict()
        vocab_size = values["text_config"]["vocab_size"]
        for kwargs in (
            {"audio_eos_code": 7},
            {"audio_eos_code": -1},
            {"interleaved_n_audio": 0},
            {"image_token_id": vocab_size + 7},
            {"video_token_id": vocab_size},
            {"audio_token_id": vocab_size},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Lfm3OmniConfig(**{**values, **kwargs})
        # the checks are class validators: a later edit fails `validate()` and is never written
        for name, value in (("audio_eos_code", 7), ("image_token_id", vocab_size)):
            config = tiny_config()
            setattr(config, name, value)
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(StrictDataclassClassValidationError):
                    config.validate()
                with self.assertRaises(StrictDataclassClassValidationError):
                    config.save_pretrained(tmp)

    def test_media_placeholders_without_their_features_are_rejected(self):
        for dropped, message in (
            (("input_features", "feature_attention_mask"), "input_features"),
            (("audio_out_codes",), "audio_out_codes"),
        ):
            with self.subTest(dropped=dropped), self.assertRaisesRegex(ValueError, message):
                self.model(**{**self.inputs, **dict.fromkeys(dropped)})
        # the embedding form is checked the way `get_placeholder_mask` finds placeholders
        embeds = self.model.get_input_embeddings()(self.inputs["input_ids"])
        media = {name: self.inputs[name] for name in ("pixel_values", "image_grid_thw")}
        with self.assertRaisesRegex(ValueError, "input_features"):
            self.model(inputs_embeds=embeds, **media)
        with self.assertRaisesRegex(ValueError, "input_features"):
            self.model(inputs_embeds=embeds, audio_out_codes=self.inputs["audio_out_codes"], **media)
        # the live audio arrives as features, so a prompt placeholder would never have any behind it
        features = torch.randn(1, 16, 96, device=torch_device)
        for token_id in (250, 251):
            prompt = torch.tensor([[1, token_id, 11]], device=torch_device)
            with self.subTest(token_id=token_id), self.assertRaisesRegex(ValueError, "speech placeholders"):
                self.model.transcribe_live(features, input_ids=prompt)

    def test_media_placeholder_ids_are_never_sampled_as_text(self):
        prompt = torch.tensor([[1, 10, 11]], device=torch_device)
        features = torch.randn(1, 16, 96, device=torch_device)
        for boosted_id in (249, 248, 250, 251):
            with self.subTest(boosted_id=boosted_id), torch.no_grad():
                boost = LogitsProcessorList([Boost(lambda step, row: boosted_id)])
                outputs = self.model.generate(
                    prompt, max_new_tokens=3, generation_mode="text", do_sample=True, logits_processor=boost
                )
                self.assertNotIn(boosted_id, outputs.sequences[0, 3:].tolist())
                head, original = self.model.lm_head, self.model.lm_head.forward

                def boosted(hidden_states):
                    logits = original(hidden_states)
                    logits[..., boosted_id] += 50.0
                    return logits

                head.forward = boosted
                try:
                    output = self.model.transcribe_live(features, input_ids=prompt, is_final=True, max_new_tokens=3)
                    self.assertNotIn(boosted_id, output.sequences[0].tolist())
                finally:
                    head.forward = original

    def test_image_mismatch_and_unknown_generation_arguments_raise(self):
        bad = {**self.inputs, "image_grid_thw": None}
        with self.assertRaisesRegex(ValueError, "supplied together"):
            self.model(**bad)
        bad = {**self.inputs, "input_ids": self.inputs["input_ids"].masked_fill(self.inputs["input_ids"] == 249, 3)}
        with self.assertRaisesRegex(ValueError, "Placeholder"):
            self.model(**bad)
        with self.assertRaisesRegex(ValueError, "not used by the model"):
            self.model.generate(**self.inputs, max_new_tokens=1, misspelled_argument=True)

    def test_audio_only_supervision_and_empty_supervision_are_finite(self):
        labels = torch.full_like(self.inputs["input_ids"], -100)
        for audio_labels in (
            self.inputs["audio_out_codes"],
            torch.full_like(self.inputs["audio_out_codes"], -100),
        ):
            with self.subTest(supervised=bool((audio_labels != -100).any())):
                output = self.model(**self.inputs, labels=labels, audio_labels=audio_labels, use_cache=False)
                self.assertTrue(torch.isfinite(output.loss))
                self.assertEqual(output.text_loss.item(), 0)
                output.loss.backward()
                self.model.zero_grad(set_to_none=True)

    def test_loss_weights_text_tokens_and_audio_frames_like_training(self):
        labels = self.inputs["input_ids"].clone()
        labels[labels >= 249] = -100
        output = self.model(**self.inputs, labels=labels, audio_labels=self.inputs["audio_out_codes"])
        text_count = (labels[:, 1:] != -100).sum()
        expected = (text_count * output.text_loss + 2 * output.audio_loss) / (text_count + 2 + 1e-6)
        torch.testing.assert_close(output.loss, expected)

    def test_audio_and_interleaved_generation_with_images(self):
        for mode in ("audio", "interleaved"):
            with self.subTest(mode=mode), torch.no_grad():
                torch.manual_seed(7)
                output = self.model.generate(
                    **self.inputs,
                    generation_mode=mode,
                    max_new_tokens=10,
                    depth_decoder_temperature=0.8,
                    depth_decoder_top_k=3,
                )
                self.assertTrue(torch.isfinite(output.audio_codes).all())
                # the prompt did not end on the mode's marker, so generate appended it before the budget
                prompt_length = self.inputs["input_ids"].shape[1]
                marker = (
                    self.model.config.audio_start_token_id
                    if mode == "audio"
                    else self.model.config.mixed_start_token_id
                )
                self.assertEqual(output.sequences[0, prompt_length].item(), marker)
                self.assertLessEqual(output.sequences.shape[1], prompt_length + 11)

    def test_maskgit_sampling_is_reproducible_for_batches(self):
        hidden = torch.randn(2, 32, device=torch_device)
        depth = self.model.depth_decoder.generation_config
        depth.update(temperature=0.8, top_k=3)
        try:
            with torch.no_grad():
                torch.manual_seed(123)
                first = self.model.sample_audio_frames(hidden)
                torch.manual_seed(123)
                second = self.model.sample_audio_frames(hidden)
        finally:
            depth.update(temperature=1.0, top_k=None)
        torch.testing.assert_close(first, second)
        self.assertEqual(tuple(first.shape), (2, 4))
        self.assertTrue(((first >= 0) & (first < 7)).all())

    def test_bfloat16_training_round_trips_bitwise(self):
        with tempfile.TemporaryDirectory() as directory:
            self.model.save_pretrained(directory)
            model = Lfm3OmniForConditionalGeneration.from_pretrained(directory, dtype=torch.bfloat16).to(torch_device)
            # the vision tower follows the language model's dtype, as in LFM3-VL
            self.assertTrue(all(parameter.dtype == torch.bfloat16 for parameter in model.parameters()))
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            inputs = {key: value.to(torch_device) for key, value in self.inputs.items()}
            labels = inputs["input_ids"].clone()
            labels[labels >= 249] = -100
            model(**inputs, labels=labels, audio_labels=inputs["audio_out_codes"], use_cache=False).loss.backward()
            optimizer.step()
            model.save_pretrained(directory)
            restored = Lfm3OmniForConditionalGeneration.from_pretrained(directory, dtype=torch.bfloat16).to(
                torch_device
            )
        for name, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)

    @require_accelerate
    def test_trainer_preserves_joint_loss_and_accumulates_gradients(self):
        import math

        from transformers import Trainer, TrainingArguments

        inputs = dict(
            self.inputs, labels=self.inputs["input_ids"].clone(), audio_labels=self.inputs["audio_out_codes"]
        )
        inputs["labels"][inputs["labels"] >= 249] = -100
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=self.model,
                args=TrainingArguments(
                    output_dir=directory,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=2,
                    max_steps=2,
                    learning_rate=1e-3,
                    save_strategy="no",
                    report_to=[],
                    disable_tqdm=True,
                    use_cpu=torch_device == "cpu",
                ),
                train_dataset=[{key: value.cpu() for key, value in inputs.items()}] * 4,
                data_collator=lambda rows: rows[0],
            )
            torch.manual_seed(123)
            expected = self.model(**inputs).loss
            torch.manual_seed(123)
            actual = trainer.compute_loss(self.model, dict(inputs), num_items_in_batch=1000)
            torch.testing.assert_close(actual, expected)
            before = self.model.model.audio_projector.linear_1.weight.detach().clone()
            result = trainer.train()
        self.assertEqual(result.global_step, 2)
        self.assertTrue(math.isfinite(result.training_loss))
        self.assertFalse(torch.equal(before, self.model.model.audio_projector.linear_1.weight))


class Lfm3OmniModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 2
        self.seq_length = 12
        self.is_training = True
        self.num_hidden_layers = 2
        self.hidden_size = 32
        self.num_attention_heads = 4
        self.vocab_size = 256

    def get_config(self):
        config = tiny_config().to_dict()
        config["text_config"].update(num_hidden_layers=2, layer_types=["conv", "full_attention"])
        return Lfm3OmniConfig(**config)

    def prepare_config_and_inputs_for_common(self):
        # text only: the shared attention tests feed `input_ids` alone, and a placeholder without its features is
        # refused; the media paths are pinned by the behaviour tests above
        input_ids = torch.randint(3, 200, (self.batch_size, self.seq_length), device=torch_device)
        input_ids[:, 0] = 1
        return self.get_config(), {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


@require_torch
class Lfm3OmniCommonTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """The common model and generation tests; the mixed-modality contracts are covered by the behaviour tests above."""

    all_model_classes = (Lfm3OmniModel, Lfm3OmniForConditionalGeneration) if is_torch_available() else ()
    pipeline_model_mapping = {}
    _is_composite = True
    # the common inputs carry no media; `test_mixed_training_reaches_every_component` covers the media parameters
    test_all_params_have_gradient = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_torch_exportable = False

    def setUp(self):
        self.model_tester = Lfm3OmniModelTester(self)

    def _get_conv_state_shape(self, batch_size: int, config):
        text_config = config.get_text_config()
        return (batch_size, text_config.hidden_size, text_config.conv_L_cache)

    def _audio_features_get_expected_num_attentions(self, model_tester=None):
        layer_types = self.model_tester.get_config().audio_encoder_config.text_config.layer_types
        return sum(layer_type != "conv" for layer_type in layer_types)

    def _audio_features_get_expected_num_hidden_states(self, model_tester=None):
        return self.model_tester.get_config().audio_encoder_config.text_config.num_hidden_layers + 1

    def _audio_features_prepare_config_and_inputs(self):
        config = self.model_tester.get_config()
        input_features = torch.randn(2, 16, 40, device=torch_device)
        feature_attention_mask = torch.ones(2, 40, dtype=torch.long, device=torch_device)
        feature_attention_mask[1, 24:] = 0
        return config, {"input_features": input_features, "feature_attention_mask": feature_attention_mask}

    @unittest.skip("LFM3-Omni generates with greedy search or sampling only")
    def test_generate_from_inputs_embeds_1_beam_search(self):
        pass

    @unittest.skip("the common inputs carry no media; the behaviour test of the pre-encoded images covers it")
    def test_generate_from_multimodal_encoder_outputs_and_raw_data(self):
        pass

    @unittest.skip("the conv layers record no attention weights")
    def test_attention_outputs(self):
        pass

    @unittest.skip("the conv layers keep a conv state, not key/value pairs")
    def test_past_key_values_format(self):
        pass

    def test_reverse_loading_mapping(self):
        # the base model has no reverse-renamed keys of its own
        super().test_reverse_loading_mapping(skip_base_model=True)

    @unittest.skip("LFM3 keys `rope_parameters` by attention layer type, unsupported by this test")
    def test_model_rope_scaling_frequencies(self):
        pass

    @unittest.skip("LFM3 keys `rope_parameters` by attention layer type, unsupported by this test")
    def test_model_rope_scaling_from_config_0_linear(self):
        pass

    @unittest.skip("LFM3 keys `rope_parameters` by attention layer type, unsupported by this test")
    def test_model_rope_scaling_from_config_1_dynamic(self):
        pass

    @unittest.skip("LFM3 keys `rope_parameters` by attention layer type, unsupported by this test")
    def test_model_rope_scaling_from_config_2_yarn(self):
        pass

    @unittest.skip("LFM3-Omni generates with greedy search or sampling only")
    def test_beam_search_generate(self):
        pass

    @unittest.skip("LFM3-Omni generates with greedy search or sampling only")
    def test_beam_search_generate_dict_output(self):
        pass

    @unittest.skip("LFM3-Omni generates with greedy search or sampling only")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("LFM3-Omni generates with greedy search or sampling only")
    def test_beam_sample_generate(self):
        pass

    @unittest.skip("LFM3-Omni generates with greedy search or sampling only")
    def test_beam_sample_generate_dict_output(self):
        pass

    @unittest.skip("LFM3-Omni generates with greedy search or sampling only")
    def test_constrained_beam_search_generate(self):
        pass

    @unittest.skip("LFM3-Omni generates with greedy search or sampling only")
    def test_constrained_beam_search_generate_dict_output(self):
        pass

    @unittest.skip("LFM3-Omni generates with greedy search or sampling only")
    def test_group_beam_search_generate(self):
        pass

    @unittest.skip("LFM3-Omni generates with greedy search or sampling only")
    def test_group_beam_search_generate_dict_output(self):
        pass

    @unittest.skip("LFM3-Omni generates with greedy search or sampling only")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("LFM3-Omni generates with greedy search or sampling only")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("LFM3-Omni generates with greedy search or sampling only")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip("LFM3-Omni generates with greedy search or sampling only")
    def test_assisted_decoding_matches_greedy_search_0_random(self):
        pass

    @unittest.skip("LFM3-Omni generates with greedy search or sampling only")
    def test_assisted_decoding_matches_greedy_search_1_same(self):
        pass

    @unittest.skip("LFM3-Omni generates with greedy search or sampling only")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("LFM3-Omni generates with greedy search or sampling only")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @unittest.skip("LFM3-Omni generates with greedy search or sampling only")
    def test_dola_decoding_sample(self):
        pass

    @unittest.skip("LFM3-Omni generates with greedy search or sampling only")
    def test_assisted_decoding_in_different_gpu(self):
        pass

    @unittest.skip("LFM3-Omni generates with greedy search or sampling only")
    def test_assisted_decoding_model_in_gpu_assistant_in_cpu(self):
        pass

    def _image_features_prepare_config_and_inputs(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        n = self.model_tester.batch_size
        return config, {
            "pixel_values": torch.randn(4 * n, 24, device=torch_device),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * n, device=torch_device),
        }

    def _video_features_prepare_config_and_inputs(self):
        config, inputs = self._image_features_prepare_config_and_inputs()
        return config, {"pixel_values_videos": inputs["pixel_values"], "video_grid_thw": inputs["image_grid_thw"]}

    def _image_features_get_expected_num_attentions(self, model_tester=None):
        return tiny_config().vision_config.layer_types.count("full_attention")

    def _image_features_get_expected_num_hidden_states(self, model_tester=None):
        return tiny_config().vision_config.num_hidden_layers + 1

    def _video_features_get_expected_num_attentions(self, model_tester=None):
        return self._image_features_get_expected_num_attentions()

    def _video_features_get_expected_num_hidden_states(self, model_tester=None):
        return self._image_features_get_expected_num_hidden_states()


@require_torch_accelerator
@slow
class Lfm3OmniIntegrationTest(unittest.TestCase):
    model_id = "LiquidAI/LFM3-Omni-2B-Preview"
    revision = "b241d3710b88bc504276604378ad3fc922dbd614"

    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
        from transformers.image_utils import load_image

        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/coco_sample.png"
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ]
        self.inputs = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

    def test_model_logits(self):
        model = Lfm3OmniForConditionalGeneration.from_pretrained(
            self.model_id, revision=self.revision, device_map=torch_device, dtype=torch.float32
        )
        self.assertEqual(
            int((self.inputs["input_ids"] == self.processor.image_token_id).sum()), LFM3_OMNI_IMAGE_TOKENS
        )
        with torch.no_grad():
            logits = model(**self.inputs).logits.float()

        expectations = Expectations({(None, None): LFM3_OMNI_EXPECTED_LOGITS_SLICE})
        expected_slice = torch.tensor(expectations.get_expectation(), device=torch_device)
        torch.testing.assert_close(logits[0, -1, :16], expected_slice, rtol=1e-3, atol=1e-3)

    def test_model_greedy_generation(self):
        model = Lfm3OmniForConditionalGeneration.from_pretrained(
            self.model_id, revision=self.revision, device_map=torch_device, dtype=torch.bfloat16
        )
        output = model.generate(**self.inputs, max_new_tokens=24, do_sample=False)
        text = self.processor.batch_decode(
            output.sequences[:, self.inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]

        expectations = Expectations({(None, None): LFM3_OMNI_EXPECTED_GENERATION})
        self.assertEqual(text, expectations.get_expectation())

    golden_revision = "2bc8893e4e022c980dea61369089006323456971"

    def _golden_processor_and_model(self, dtype):
        processor = AutoProcessor.from_pretrained(self.model_id, revision=self.golden_revision)
        model = Lfm3OmniForConditionalGeneration.from_pretrained(
            self.model_id, revision=self.golden_revision, device_map=torch_device, dtype=dtype
        ).eval()
        return processor, model

    def test_image_and_audio_question_greedy_ids_match_the_recorded_goldens(self):
        """Greedy reply to a synthetic gradient image + clip 0 of hf-internal-testing/librispeech_asr_dummy, exact ids
        of LiquidAI/LFM3-Omni-2B-Preview @ 2bc8893e4e022c980dea61369089006323456971 converted once with
        pr11-review-plan/hub_rewrites/H4/convert_audio_export.py --omni (config only, weights unchanged) plus the
        staged H3/H7 processor configs; point `model_id` at that directory until the uploads land. The ids were
        recorded 2026-10-27 on a B300 in fp32 from the OLD implementation (9a5bed21c9) and reproduced bitwise, in
        fp32 and bf16, by the refactored tree lfm3/stack-merge @ 22ac0dff95 (M5 merged; the pre-merge stack raised
        `pixel_values and image_grid_thw must be supplied together` on the first decode step of an image prompt).
        """
        x = np.linspace(0, 255, 448, dtype=np.float32)
        red, green = np.tile(x[None, :], (448, 1)), np.tile(x[:, None], (1, 448))
        image = Image.fromarray(np.stack([red, green, (red + green) / 2], -1).round().astype(np.uint8))
        from datasets import load_dataset

        clip = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")[0]
        audio = np.asarray(clip["audio"]["array"], dtype=np.float32)
        processor, model = self._golden_processor_and_model(torch.float32)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "audio"},
                    {"type": "text", "text": "What is in the image, and what is said in the audio? Answer briefly."},
                ],
            }
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, images=[image], audio=[audio], sampling_rate=16000, return_tensors="pt")
        inputs = inputs.to(torch_device)
        self.assertEqual(int((inputs["input_ids"] == processor.image_token_id).sum()), 196)
        output = model.generate(
            **inputs, max_new_tokens=32, do_sample=False, generation_mode="text", return_dict_in_generate=True
        )
        expected = Expectations({(None, None): LFM3_OMNI_GOLDEN_QA_IDS}).get_expectation()
        self.assertEqual(output.sequences[0, inputs["input_ids"].shape[1] :].tolist(), expected)

    def test_speech_frames_match_the_recorded_codes(self):
        """12 greedy speech frames (`depth_decoder_do_sample=False`, seeded MaskGIT order) of the same checkpoint,
        recorded 2026-10-27 on an NVIDIA B300 (torch 2.12.0+cu130, transformers 5.21.0.dev0) with the refactored tree (lfm3/stack @ a6b781c242) by pr11-review-plan/goldens/golden_omni.py; OLD = the pre-refactor PR #11 code merged on upstream main (9a5bed21c9), compared in goldens/REPORT.md. The fp32 codes equal the OLD implementation's bitwise (bf16 ones diverge from
        the fifth codebook of the first frame on, fp32 RoPE cos/sin)."""
        processor, model = self._golden_processor_and_model(torch.float32)
        conversation = [
            {"role": "system", "content": "Perform TTS in American English using a female voice."},
            {"role": "user", "content": "The quick brown fox jumps over the lazy dog."},
            {"role": "assistant", "content": "<|audio_start|>"},
        ]
        inputs = processor.apply_chat_template(
            conversation, continue_final_message=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)
        torch.manual_seed(0)
        output = model.generate(
            **inputs, max_new_tokens=12, do_sample=False, depth_decoder_do_sample=False, return_dict_in_generate=True
        )
        self.assertEqual(output.audio_lengths.tolist(), [12])
        expected = Expectations({(None, None): LFM3_OMNI_GOLDEN_TTS_CODES}).get_expectation()
        self.assertEqual(output.audio_codes[0].tolist(), expected)


LFM3_OMNI_GOLDEN_QA_IDS = [
    597,
    4145,
    355,
    267,
    10225,
    3599,
    23266,
    9320,
    299,
    487,
    4752,
    383,
    278,
    2690,
    1125,
    3698,
    6776,
    8824,
    834,
    310,
    3789,
    383,
    278,
    1583,
    22,
    440,
    13551,
    2997,
    34,
    496,
    22054,
    22,
]
LFM3_OMNI_GOLDEN_TTS_CODES = [
    [647, 583, 927, 392, 1059, 1273, 213, 1790, 1217, 1355, 1574, 1185],
    [603, 204, 148, 1777, 161, 154, 1404, 194, 1545, 888, 892, 1562],
    [1912, 1296, 632, 1279, 1464, 1777, 1623, 1160, 1672, 1076, 1074, 850],
    [689, 688, 680, 1430, 1039, 175, 758, 1028, 622, 254, 1797, 355],
    [1200, 818, 755, 1854, 426, 477, 163, 916, 902, 1086, 628, 407],
    [399, 62, 406, 1482, 1615, 1223, 1098, 1199, 1679, 1818, 891, 1099],
    [147, 250, 249, 1362, 1789, 177, 1713, 1724, 434, 90, 1847, 1342],
    [1621, 1997, 1948, 1612, 632, 1430, 803, 506, 1985, 628, 1483, 1584],
    [736, 1425, 1144, 1582, 858, 772, 1420, 1417, 1411, 1534, 13, 1515],
    [1044, 755, 1082, 1075, 1196, 1236, 794, 1173, 1443, 1518, 1432, 1016],
    [1460, 1130, 801, 471, 1527, 1780, 1997, 737, 1240, 454, 1150, 757],
    [1753, 1761, 1769, 865, 777, 860, 1522, 1170, 707, 1467, 1588, 1220],
]


LFM3_OMNI_IMAGE_TOKENS = 300
LFM3_OMNI_EXPECTED_LOGITS_SLICE = [
    -12.286387,
    -11.071152,
    -12.899508,
    -10.642688,
    -10.852482,
    -10.18991,
    -11.388788,
    -9.495357,
    -9.535076,
    -8.705992,
    -4.582078,
    -5.84231,
    -7.200204,
    -7.905283,
    -8.593463,
    -7.693958,
]
LFM3_OMNI_EXPECTED_GENERATION = (
    "Two tabby cats are sprawled out asleep on a bright pink couch, with two remote controls resting on the cushion"
)
