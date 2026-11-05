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
"""Testing suite for the PyTorch LFM3-Omni model."""

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from transformers import (
    AutoProcessor,
    Lfm3OmniConfig,
    Lfm3OmniForConditionalGeneration,
    Lfm3OmniModel,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import ModelTesterMixin
from ..lfm3_audio.test_modeling_lfm3_audio import (
    AUDIO_OUT_TOKEN_ID,
    AUDIO_TOKEN_ID,
    CODEBOOK_SIZE,
    NUM_CODEBOOKS,
    TEXT_ONLY,
    Boost,
    Lfm3AudioBehaviorTestMixin,
)
from ..lfm3_audio.test_modeling_lfm3_audio import tiny_config as audio_config


if is_torch_available():
    import torch

    from transformers import LogitsProcessorList


IMAGE_TOKEN_ID, VIDEO_TOKEN_ID = 249, 248


def tiny_config():
    values = audio_config().to_dict()
    values.pop("model_type")
    values["vision_config"] = {
        # head_dim 16: the FA4 kernels need at least 16
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "layer_types": ["conv", "full_attention"],
        "patch_size": 2,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
    }
    values.update(image_token_id=IMAGE_TOKEN_ID, video_token_id=VIDEO_TOKEN_ID, projector_hidden_size=32)
    return Lfm3OmniConfig(**values)


def mixed_inputs():
    return {
        "input_ids": torch.tensor(
            [[1, 4, IMAGE_TOKEN_ID, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 8, AUDIO_OUT_TOKEN_ID, AUDIO_OUT_TOKEN_ID, 9]],
            device=torch_device,
        ),
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

    def test_generation_encodes_images_once_and_matches_cache_free(self):
        config = self.model.config
        encoder = self.model.model.get_image_features
        with (
            torch.no_grad(),
            # `generate` reads the encoder's signature to pick its inputs, which a bare mock would hide
            patch.object(self.model.model, "get_image_features", autospec=True, side_effect=encoder) as encode,
        ):
            # no early stop either way: the random tiny model may otherwise end on its first token
            suppress = [config.audio_start_token_id, config.mixed_start_token_id, config.text_end_token_id, 2]
            output = self.model.generate(**self.inputs, max_new_tokens=4, suppress_tokens=suppress)
            # `generate` pre-encodes the images once and replays them from the cache
            self.assertEqual(encode.call_count, 1)
            encoded = self.model._prepare_multimodal_encoder_kwargs_for_generation(dict(self.inputs))
            with self.assertRaisesRegex(ValueError, "You cannot pass both: raw pixels and pre-computed embeddings"):
                self.model.generate(**self.inputs, mm_encoder_outputs=encoded["mm_encoder_outputs"], max_new_tokens=1)
            ids = self.inputs["input_ids"].clone()
            for _ in range(4):
                logits = self.model(**{**self.inputs, "input_ids": ids}, use_cache=False).logits[:, -1]
                # the media placeholders are never sampled as text
                logits[:, [*suppress, *self.model._placeholder_token_ids]] = -float("inf")
                ids = torch.cat((ids, logits.argmax(-1, keepdim=True)), dim=-1)
        torch.testing.assert_close(output.sequences, ids)

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

    def test_mixed_training_reaches_every_component(self):
        labels = self.inputs["input_ids"].clone()
        labels[labels >= VIDEO_TOKEN_ID] = -100
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
        for token_id in (AUDIO_TOKEN_ID, AUDIO_OUT_TOKEN_ID):
            prompt = torch.tensor([[1, token_id, 11]], device=torch_device)
            with self.subTest(token_id=token_id), self.assertRaisesRegex(ValueError, "speech placeholders"):
                self.model.transcribe_live(features, input_ids=prompt)

    def test_media_placeholder_ids_are_never_sampled_as_text(self):
        prompt = torch.tensor([[1, 10, 11]], device=torch_device)
        features = torch.randn(1, 16, 96, device=torch_device)
        for boosted_id in (IMAGE_TOKEN_ID, VIDEO_TOKEN_ID, AUDIO_TOKEN_ID, AUDIO_OUT_TOKEN_ID):
            with self.subTest(boosted_id=boosted_id), torch.no_grad():
                boost = LogitsProcessorList([Boost(lambda step, row: boosted_id)])
                outputs = self.model.generate(
                    prompt, max_new_tokens=3, do_sample=True, logits_processor=boost, **TEXT_ONLY
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

    def test_audio_and_interleaved_generation_with_images(self):
        config = self.model.config
        for marker in (config.audio_start_token_id, config.mixed_start_token_id):
            inputs = dict(self.inputs)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([[marker]], device=torch_device)], 1)
            with self.subTest(marker=marker), torch.no_grad():
                torch.manual_seed(7)
                output = self.model.generate(
                    **inputs, max_new_tokens=10, depth_decoder_temperature=0.8, depth_decoder_top_k=3
                )
                self.assertTrue(torch.isfinite(output.audio_codes).all())
                generated = output.sequences[0, inputs["input_ids"].shape[1] :]
                self.assertLessEqual(generated.shape[0], 10)
                self.assertEqual(int((generated == config.audio_out_token_id).sum()), output.audio_lengths.item())
                self.assertGreater(output.audio_lengths.item(), 0)


class Lfm3OmniModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 7
        self.seq_length = 12
        self.is_training = True
        self.num_hidden_layers = 3
        self.hidden_size = 32
        self.num_attention_heads = 4
        self.vocab_size = 256

    def get_config(self):
        config = tiny_config().to_dict()
        # a window past every prompt, as for the other sliding-window models
        config["text_config"].update(
            num_hidden_layers=3,
            layer_types=["conv", "sliding_attention", "full_attention"],
            sliding_window=4096,
            max_position_embeddings=512,
        )
        # the SDPA flash backend takes no explicit mask, so the encoder window covers every clip too
        config["audio_encoder_config"]["text_config"]["sliding_window"] = 4096
        return Lfm3OmniConfig(**config)

    def prepare_config_and_inputs_for_common(self):
        # one image, one input clip and one prompt speech frame per row behind matching placeholders; the last clip
        # is shorter (24 of 40 spectrogram frames: 3 encoder tokens instead of 5)
        input_ids = torch.randint(3, 200, (self.batch_size, self.seq_length), device=torch_device)
        input_ids[:, 0] = 1
        input_ids[:, 1] = IMAGE_TOKEN_ID
        input_ids[:, 2:7] = AUDIO_TOKEN_ID
        input_ids[-1, 5:7] = 10
        input_ids[:, 7] = AUDIO_OUT_TOKEN_ID
        feature_attention_mask = torch.ones(self.batch_size, 40, dtype=torch.long, device=torch_device)
        feature_attention_mask[-1, 24:] = 0
        return self.get_config(), {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": torch.randn(4 * self.batch_size, 24, device=torch_device),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * self.batch_size, device=torch_device),
            "input_features": torch.randn(self.batch_size, 16, 40, device=torch_device),
            "feature_attention_mask": feature_attention_mask,
            "audio_out_codes": torch.randint(
                0, CODEBOOK_SIZE - 1, (self.batch_size, NUM_CODEBOOKS), device=torch_device
            ),
        }


@require_torch
class Lfm3OmniCommonTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """The common model and generation tests; the mixed-modality contracts are covered by the behaviour tests above."""

    all_model_classes = (Lfm3OmniModel, Lfm3OmniForConditionalGeneration)
    _is_composite = True
    # the shared tests pad these to a fixed batch size, which the patch-major vision inputs cannot take
    additional_model_inputs = ["input_features", "feature_attention_mask", "audio_out_codes"]
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = Lfm3OmniModelTester(self)
        # `generate` only runs greedy search or sampling, like Dia
        if any(mode in self._testMethodName for mode in ("beam", "assisted_decoding", "prompt_lookup")):
            self.skipTest("LFM3-Omni generates with greedy search or sampling only")
        if "rope_scaling_from_config" in self._testMethodName:
            self.skipTest("random `input_ids` hit the media placeholders, whose features the test does not pass")

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if return_labels and model_class is Lfm3OmniForConditionalGeneration:
            # end-of-audio frames: every codebook is masked, so the loss draws no random mask and stays deterministic
            num_frames = int((inputs_dict["input_ids"] == AUDIO_OUT_TOKEN_ID).sum())
            inputs_dict["audio_labels"] = torch.full(
                (num_frames, NUM_CODEBOOKS), CODEBOOK_SIZE - 1, dtype=torch.long, device=torch_device
            )
        return inputs_dict

    @unittest.skip("the common prompts carry media placeholders whose features travel beside the ids")
    def test_generate_from_random_inputs_embeds(self):
        pass

    @unittest.skip("the common prompts carry media placeholders whose features travel beside the ids")
    def test_flash_attention_2_continue_generate_with_position_ids(self):
        pass

    @unittest.skip("the common prompts carry media placeholders whose features travel beside the ids")
    def test_recurrent_layers_mask_padding_on_continued_forward(self):
        pass

    def test_generate_methods_with_logits_to_keep(self):
        # a sampled `<|audio_start|>` makes speech frames, whose MaskGIT reveal order each call draws anew
        generate = Lfm3OmniForConditionalGeneration.generate

        def seeded_generate(model, *args, **kwargs):
            torch.manual_seed(0)
            return generate(model, *args, **kwargs)

        with patch.object(Lfm3OmniForConditionalGeneration, "generate", seeded_generate):
            super().test_generate_methods_with_logits_to_keep()

    def test_generate_from_multimodal_encoder_outputs(self):
        # a sampled `<|text_end|>` ends its row, which this test then pads with -1: keep the draw reproducible
        torch.manual_seed(0)
        super().test_generate_from_multimodal_encoder_outputs()

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

    @unittest.skip("builds its inputs from ids with speech placeholders, without the features behind them")
    def test_flash_attn_2_from_config(self):
        pass

    @unittest.skip("builds its inputs from ids with speech placeholders, without the features behind them")
    def test_flash_attn_4_from_config(self):
        pass

    @unittest.skip("The vision tower splits frames by host-side sizes, which `torch.export` can't trace")
    def test_torch_export_dynamic(self):
        pass

    @unittest.skip("The vision tower splits frames by host-side sizes, which `torch.export` can't trace")
    def test_torch_export_static(self):
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
class Lfm3OmniIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    model_id = "LiquidAI/LFM3-Omni-2B-Preview"

    def setUp(self):
        super().setUp()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
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
            self.model_id, device_map=torch_device, dtype=torch.float32
        )
        self.assertEqual(int((self.inputs["input_ids"] == self.processor.image_token_id).sum()), 300)
        with torch.no_grad():
            logits = model(**self.inputs).logits.float()

        expectations = Expectations(
            {
                (None, None): [
                    -11.675359,
                    -9.37761,
                    -11.801423,
                    -10.33584,
                    -11.593874,
                    -10.714907,
                    -12.667312,
                    -10.655792,
                    -10.47758,
                    -10.292186,
                    -4.483351,
                    -6.508935,
                    -7.688687,
                    -8.485828,
                    -9.174135,
                    -9.053003,
                ]
            }
        )
        expected_slice = torch.tensor(expectations.get_expectation(), device=torch_device)
        torch.testing.assert_close(logits[0, -1, :16], expected_slice, rtol=1e-3, atol=1e-3)

    def test_model_greedy_generation(self):
        model = Lfm3OmniForConditionalGeneration.from_pretrained(
            self.model_id, device_map=torch_device, dtype=torch.bfloat16
        )
        output = model.generate(**self.inputs, max_new_tokens=24, do_sample=False)
        text = self.processor.batch_decode(
            output.sequences[:, self.inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]

        expectations = Expectations(
            {
                (
                    None,
                    None,
                ): "Two tabby cats are lying side by side, asleep, on a bright pink fabric surface with two remote controls resting near"
            }
        )
        self.assertEqual(text, expectations.get_expectation())

    def _golden_processor_and_model(self, dtype):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = Lfm3OmniForConditionalGeneration.from_pretrained(
            self.model_id, device_map=torch_device, dtype=dtype
        ).eval()
        return processor, model

    def test_image_and_audio_question_greedy_ids_match_the_recorded_goldens(self):
        # a synthetic gradient image and clip 0 of hf-internal-testing/librispeech_asr_dummy, greedy in fp32
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
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            suppress_tokens=[model.config.audio_start_token_id, model.config.mixed_start_token_id],
            return_dict_in_generate=True,
        )
        expected = Expectations(
            {
                (None, None): [
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
            }
        ).get_expectation()
        self.assertEqual(output.sequences[0, inputs["input_ids"].shape[1] :].tolist(), expected)

    def test_speech_frames_match_the_recorded_codes(self):
        # 12 greedy speech frames (`depth_decoder_do_sample=False`, seeded MaskGIT reveal order) in fp32
        processor, model = self._golden_processor_and_model(torch.float32)
        conversation = [
            {"role": "system", "content": "Perform TTS in American English using a female voice."},
            {"role": "user", "content": "The quick brown fox jumps over the lazy dog."},
            {"role": "assistant", "content": "<|audio_start|>"},
        ]
        inputs = processor.apply_chat_template(
            conversation, continue_final_message=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)
        torch.manual_seed(0)  # the order the MaskGIT passes reveal the codebooks in
        output = model.generate(
            **inputs, max_new_tokens=12, do_sample=False, depth_decoder_do_sample=False, return_dict_in_generate=True
        )
        self.assertEqual(output.audio_lengths.tolist(), [12])
        expected = Expectations(
            {
                (None, None): [
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
            }
        ).get_expectation()
        self.assertEqual(output.audio_codes[0].tolist(), expected)
