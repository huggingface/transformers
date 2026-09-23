# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch EmbeddingGemma2 model."""

import tempfile
import unittest

from parameterized import parameterized

from transformers import (
    EmbeddingGemma2Config,
    EmbeddingGemma2TextConfig,
    is_torch_available,
    set_seed,
)
from transformers.testing_utils import (
    CaptureLogger,
    require_torch,
    torch_device,
)
from transformers.utils import ModelOutput, logging

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        EmbeddingGemma2Model,
        EmbeddingGemma2TextModel,
    )


class EmbeddingGemma2TextModelTester:
    """Builds a tiny `EmbeddingGemma2TextConfig` and matching text-only inputs."""

    config_class = EmbeddingGemma2TextConfig
    if is_torch_available():
        base_model_class = EmbeddingGemma2TextModel

    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=32,
        intermediate_size=37,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        sliding_window=4,
        hidden_size_per_layer_input=16,
        embedding_dim=24,
        pad_token_id=0,
        initializer_range=0.02,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.sliding_window = sliding_window
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range

        # Both attention flavours are exercised; the last layer is always forced to `full_attention`.
        self.layer_types = ["sliding_attention", "full_attention"]
        # Gemma 4 (and therefore EmbeddingGemma 2) uses a wider `head_dim` on the full attention layers.
        # Without an explicit override the config falls back to `global_head_dim=512`, which would blow
        # up the size of this tiny model.
        self.per_layer_config = {
            layer_idx: {"head_dim": 2 * self.head_dim}
            for layer_idx, layer_type in enumerate(self.layer_types)
            if layer_type == "full_attention"
        }
        self.encoder_seq_length = seq_length

    def get_config(self):
        return EmbeddingGemma2TextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            sliding_window=self.sliding_window,
            hidden_size_per_layer_input=self.hidden_size_per_layer_input,
            embedding_dim=self.embedding_dim,
            pad_token_id=self.pad_token_id,
            initializer_range=self.initializer_range,
            layer_types=self.layer_types,
            per_layer_config=self.per_layer_config,
        )

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 1) + 1
        attention_mask = None
        if self.use_input_mask:
            attention_mask = torch.ones_like(input_ids).to(torch_device)
        return self.get_config(), input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}


@require_torch
class EmbeddingGemma2TextModelTest(ModelTesterMixin, unittest.TestCase):
    # EmbeddingGemma 2 is an encoder: there is no `ForCausalLM` / `ForConditionalGeneration`, hence no
    # `GenerationTesterMixin` and no generative classes.
    all_model_classes = (EmbeddingGemma2TextModel,) if is_torch_available() else ()
    all_generative_model_classes = ()

    def setUp(self):
        self.model_tester = EmbeddingGemma2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EmbeddingGemma2TextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()


class EmbeddingGemma2ModelTester:
    """Builds a tiny composite config: text backbone + reused Gemma 4 vision and audio towers."""

    def __init__(
        self,
        parent,
        mm_tokens_per_image=2,
        image_token_id=4,
        video_token_id=7,
        audio_token_id=8,
        boi_token_id=5,
        eoi_token_id=6,
        seq_length=25,
        is_training=True,
        vision_config={
            "use_labels": True,
            "image_size": 20,
            "patch_size": 5,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "num_key_value_heads": 1,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
        audio_config={
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "output_proj_dims": 32,
            "subsampling_conv_channels": [8, 8],
        },
    ):
        self.parent = parent
        self.mm_tokens_per_image = mm_tokens_per_image
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.llm_tester = EmbeddingGemma2TextModelTester(self.parent)
        self.text_config = self.llm_tester.get_config()
        self.vision_config = vision_config
        self.audio_config = audio_config
        self.seq_length = seq_length
        self.pad_token_id = self.text_config.pad_token_id

        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size
        self.embedding_dim = self.text_config.embedding_dim
        self.num_attention_heads = self.text_config.num_attention_heads
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = vision_config["num_channels"]
        self.image_size = vision_config["image_size"]
        self.encoder_seq_length = seq_length

    def get_config(self):
        return EmbeddingGemma2Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            audio_config=self.audio_config,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            audio_token_id=self.audio_token_id,
            boi_token_id=self.boi_token_id,
            eoi_token_id=self.eoi_token_id,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        config.vision_config.pooling_kernel_size = 2

        # (num_images, max_num_patches, patch_size * patch_size * num_channels)
        patch_size = config.vision_config.patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.vision_config["image_size"],
                patch_size * patch_size * self.vision_config["num_channels"],
            ]
        )

        # create (h*w, 2) grid of (x, y) coords for a non-square input image
        num_patches = self.vision_config["image_size"]
        h = int(num_patches**0.5)
        w = num_patches // h

        xs = torch.arange(w).repeat(h)
        ys = torch.arange(h).repeat_interleave(w)
        pixel_position_ids = torch.stack([xs, ys], dim=-1).to(device=torch_device)
        pixel_position_ids = pixel_position_ids.unsqueeze(0).repeat(self.batch_size, 1, 1)

        return config, pixel_values, pixel_position_ids

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, pixel_position_ids = self.prepare_config_and_inputs()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(self.pad_token_id).to(torch_device)

        # Ensure no tokens accidentally match special token IDs
        for token_id in [config.image_token_id, config.video_token_id, config.audio_token_id]:
            input_ids[input_ids == token_id] = self.pad_token_id
        input_ids[:, :5] = config.image_token_id

        inputs_dict = {
            "pixel_values": pixel_values,
            "image_position_ids": pixel_position_ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class EmbeddingGemma2ModelTest(ModelTesterMixin, unittest.TestCase):
    # No generative classes: EmbeddingGemma 2 only ever produces embeddings.
    all_model_classes = (EmbeddingGemma2Model,) if is_torch_available() else ()
    all_generative_model_classes = ()
    additional_model_inputs = ["image_position_ids"]
    model_split_percents = [0.85, 0.9]

    def setUp(self):
        self.model_tester = EmbeddingGemma2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EmbeddingGemma2Config, hidden_size=37)
        self.skip_flash_attn_inference_equivalence_tests()

    def skip_flash_attn_inference_equivalence_tests(self):
        skippable_tests = [
            "test_flash_attn_2_inference_equivalence",
            "test_flash_attn_3_inference_equivalence",
            "test_flash_attn_4_inference_equivalence",
        ]
        for test in skippable_tests:
            if self._testMethodName.startswith(test):
                self.skipTest(reason="The base test does not pass the image_position_ids required by EmbeddingGemma 2")

    # NOTE: no `test_config` here (unlike the text-model test): `ConfigTester.run_common_tests`
    # asserts a `vocab_size` attribute, which on a composite config only lives on `text_config`.
    # Gemma 4's multimodal test leaves its config tester unused for the same reason.

    def test_vision_axial_rope(self):
        # Override: the inherited Gemma 4 vision rope takes position IDs with a batch dim, which the
        # common test omits. Same override as `Gemma4Vision2TextModelTest`.
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        rope_class = None
        base_model = EmbeddingGemma2Model(config)
        for _, module in base_model.named_modules():
            if hasattr(module, "compute_axial_rope_parameters"):
                rope_class = type(module)
                vision_config = module.config
                break

        if rope_class is None:
            self.skipTest("Couldn't infer RoPE layer for this model class.")

        # First make sure that validation on default config raises no rope-related warnings
        logger = logging.get_logger("transformers.modeling_rope_utils")
        with CaptureLogger(logger) as cl:
            vision_config.validate_rope()
        self.assertEqual("", cl.out)
        logger.warning_once.cache_clear()

        # Axial rope type expects only `rope_theta`, otherwise raises warning
        vision_config.rope_parameters["factor"] = 0.25
        logger = logging.get_logger("transformers.modeling_rope_utils")
        with CaptureLogger(logger) as cl:
            vision_config.validate_rope()
        self.assertEqual("Unrecognized keys in `rope_parameters` for 'rope_type'='axial': {'factor'}\n", cl.out)
        del vision_config.rope_parameters["factor"]
        logger.warning_once.cache_clear()

        inv_freq, attention_scale = rope_class.compute_axial_rope_parameters(config=vision_config)
        rope_module = rope_class(vision_config).to(device=torch_device)

        self.assertTrue(hasattr(rope_module, "inv_freq"))
        self.assertTrue(hasattr(rope_module, "attention_scaling"))
        self.assertEqual(attention_scale, 1.0)  # attention scale is always 1
        torch.testing.assert_close(inv_freq, rope_module.inv_freq.cpu())

        # create 2D position IDs for a single grid of one row and 10 cols `size=(10, 2)`
        position_ids = torch.stack(
            [
                torch.arange(10, dtype=torch.long, device=torch_device),
                torch.zeros(10, dtype=torch.long, device=torch_device),
            ]
        ).transpose(0, 1)
        position_ids = position_ids[None, ...].repeat(3, 1, 1)  # batch size of `3`
        # and an empty hidden states used only to infer device/dtype
        hidden_states = torch.empty(1, dtype=torch.float32, device=torch_device)
        cos, sin = rope_module(hidden_states, position_ids)
        self.assertEqual(cos.shape[-1], inv_freq.shape[-1] * 4)  # the freq are `//4` of head dim
        self.assertEqual(cos.shape[0], 3)  # angles preserve batch

    def _audio_features_prepare_config_and_inputs(self):
        config = self.model_tester.get_config()
        input_features = floats_tensor([self.model_tester.batch_size, 16, 8])
        input_features_mask = torch.ones(self.model_tester.batch_size, 16, dtype=torch.bool, device=torch_device)
        return config, {"input_features": input_features, "input_features_mask": input_features_mask}

    def _video_features_prepare_config_and_inputs(self):
        config, _, _ = self.model_tester.prepare_config_and_inputs()
        # Use a (2, 2) patch grid per frame with pooling_kernel_size=2 so each frame produces 1 pooled patch row,
        # matching `ModelTesterMixin.test_get_video_features_output`'s `last_hidden_state.shape[0] == batch_size`.
        pixel_values_videos, video_position_ids = self._ragged_video_inputs(
            config, [(2, 2)] * self.model_tester.batch_size
        )
        num_frames_per_video = torch.ones(self.model_tester.batch_size, dtype=torch.long, device=torch_device)
        return config, {
            "pixel_values_videos": pixel_values_videos,
            "video_position_ids": video_position_ids,
            "num_frames_per_video": num_frames_per_video,
        }

    @parameterized.expand([True, False, None])
    def test_get_image_features_output(self, return_dict: bool | None):
        "Override to infer last hidden states' `batch_size` from image position ids"
        for model_class in self.all_model_classes:
            if not hasattr(model_class, "get_image_features"):
                continue

            config, inputs_dict = self._image_features_prepare_config_and_inputs()
            if return_dict is not None:
                config.return_dict = return_dict

            model = model_class(config).eval()
            model = model.to(torch_device)

            set_seed(42)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs_dict)

            if return_dict in (True, None):
                self.assertTrue(isinstance(outputs, ModelOutput), "get_image_features() must return a BaseModelOutput")
                self.assertTrue(
                    hasattr(outputs, "last_hidden_state"),
                    "get_image_features() must return a BaseModelOutput with last_hidden_state",
                )
                self.assertTrue(
                    hasattr(outputs, "pooler_output"),
                    "get_image_features() must return a BaseModelOutput with pooler_output",
                )
                self.assertTrue(
                    hasattr(outputs, "hidden_states"),
                    "get_image_features() must return a BaseModelOutput with hidden_states",
                )
                if self.has_attentions:
                    self.assertTrue(
                        hasattr(outputs, "attentions"),
                        "get_image_features() must return a BaseModelOutput with attentions",
                    )

                last_hidden_state_shape = outputs.last_hidden_state.shape
                batch_size = inputs_dict["pixel_values"].shape[0]
                output_length = inputs_dict["pixel_values"].shape[-2] // (
                    model.config.vision_config.pooling_kernel_size**2
                )
                k_squared = int((inputs_dict["image_position_ids"].shape[1] // output_length) ** 0.5) ** 2
                batch_size *= inputs_dict["image_position_ids"].shape[1] // k_squared

                self.assertEqual(
                    last_hidden_state_shape[0],
                    batch_size,
                    f"batch_size mismatch, full shape: {last_hidden_state_shape}",
                )
                self.assertEqual(
                    last_hidden_state_shape[-1],
                    config.vision_config.hidden_size,
                    f"hidden_size mismatch, full shape: {last_hidden_state_shape}",
                )

                self.assertEqual(
                    len(outputs.pooler_output),
                    self.model_tester.batch_size,
                    f"batch_size mismatch for `pooler_output`: {len(outputs.pooler_output)} != "
                    f"{self.model_tester.batch_size}",
                )
                self.assertEqual(
                    outputs.pooler_output[0].ndim,
                    2,
                    f"each sample in `pooler_output` should be a 2D array but got {outputs.pooler_output[0].ndim}",
                )
            else:
                self.assertIsInstance(outputs, tuple, "get_image_features() must return a tuple if return_dict=False")

    def test_model_outputs_embedding_dim(self):
        """The composite model returns the projected embedding, not `hidden_size` states."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = EmbeddingGemma2Model(config).to(torch_device).eval()

        with torch.no_grad():
            outputs = model(**inputs_dict)

        self.assertEqual(
            outputs.last_hidden_state.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, config.text_config.embedding_dim),
        )
        self.assertNotEqual(config.text_config.embedding_dim, config.text_config.hidden_size)

    def _ragged_video_inputs(self, config, frame_grids):
        """Builds a flat video batch: frames of all videos concatenated along dim 0.

        `frame_grids` gives each frame's `(height, width)` patch grid. Both dims must be multiples of
        `pooling_kernel_size` — [`get_image_size_for_max_num_patches`] guarantees that, and the pooler's token
        count only agrees with `valid_patches // k**2` when it holds. Everything past a frame's grid is padded:
        patches to zero, position ids to `(-1, -1)`, exactly as `pad_to_max_patches` does.
        """
        max_patches = self.model_tester.vision_config["image_size"]
        patch_dim = config.vision_config.patch_size**2 * self.model_tester.vision_config["num_channels"]

        pixel_values_videos = floats_tensor([len(frame_grids), max_patches, patch_dim])
        video_position_ids = torch.full((len(frame_grids), max_patches, 2), -1, dtype=torch.long)

        for frame_index, (h, w) in enumerate(frame_grids):
            xs = torch.arange(w).repeat(h)
            ys = torch.arange(h).repeat_interleave(w)
            video_position_ids[frame_index, : h * w] = torch.stack([xs, ys], dim=-1)
            pixel_values_videos[frame_index, h * w :] = 0.0

        return pixel_values_videos, video_position_ids.to(torch_device)

    def _expected_video_split_sizes(self, config, video_position_ids, num_frames_per_video):
        k_squared = config.vision_config.pooling_kernel_size**2
        tokens_per_frame = (video_position_ids != -1).all(dim=-1).sum(dim=-1) // k_squared
        split_sizes, offset = [], 0
        for num_frames in num_frames_per_video:
            split_sizes.append(int(tokens_per_frame[offset : offset + num_frames].sum()))
            offset += num_frames
        return split_sizes

    def test_get_video_features_ragged_batch(self):
        """Videos of different lengths are split back apart using `num_frames_per_video`."""
        config, _, _ = self.model_tester.prepare_config_and_inputs()
        num_frames_per_video = [2, 3]
        pixel_values_videos, video_position_ids = self._ragged_video_inputs(config, [(4, 4)] * 5)

        model = EmbeddingGemma2Model(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model.get_video_features(
                pixel_values_videos=pixel_values_videos,
                video_position_ids=video_position_ids,
                num_frames_per_video=torch.tensor(num_frames_per_video, device=torch_device),
            )

        expected = self._expected_video_split_sizes(config, video_position_ids, num_frames_per_video)
        self.assertEqual(len(outputs.pooler_output), len(num_frames_per_video))
        self.assertEqual([len(video) for video in outputs.pooler_output], expected)
        # The longer video must own more soft tokens, or a wrong split could pass unnoticed
        self.assertLess(expected[0], expected[1])

    def test_get_video_features_ragged_frames_and_patches(self):
        """Ragged along both axes: different frame counts *and* different patch grids per frame."""
        config, _, _ = self.model_tester.prepare_config_and_inputs()
        num_frames_per_video = [2, 3]
        frame_grids = [(4, 4), (2, 4), (4, 4), (2, 2), (4, 4)]
        pixel_values_videos, video_position_ids = self._ragged_video_inputs(config, frame_grids)

        model = EmbeddingGemma2Model(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model.get_video_features(
                pixel_values_videos=pixel_values_videos,
                video_position_ids=video_position_ids,
                num_frames_per_video=torch.tensor(num_frames_per_video, device=torch_device),
            )

        expected = self._expected_video_split_sizes(config, video_position_ids, num_frames_per_video)
        self.assertEqual([len(video) for video in outputs.pooler_output], expected)
        # Frames within a video must not all contribute the same count, or raggedness is untested
        self.assertNotEqual(expected[0], expected[1])
        # Every soft token is accounted for by exactly one video
        self.assertEqual(sum(len(video) for video in outputs.pooler_output), sum(expected))

    def test_get_video_features_requires_num_frames_per_video(self):
        """Without the frame counts the flat batch cannot be split, so it must fail loudly."""
        config, _, _ = self.model_tester.prepare_config_and_inputs()
        pixel_values_videos, video_position_ids = self._ragged_video_inputs(config, [(4, 4)] * 5)

        model = EmbeddingGemma2Model(config).to(torch_device).eval()
        with self.assertRaisesRegex(ValueError, "num_frames_per_video"):
            model.get_video_features(pixel_values_videos=pixel_values_videos, video_position_ids=video_position_ids)

    def test_dynamic_tower_loading_and_guards(self):
        """Disabling vision_config or audio_config (or loading EmbeddingGemma2TextModel directly) loads cleanly with zero unexpected/missing keys and guards disabled feature extractors."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        full_model = EmbeddingGemma2Model(config).to(torch_device).eval()

        text_input_ids = ids_tensor([2, 6], config.text_config.vocab_size - 1) + 1
        for token_id in [config.image_token_id, config.video_token_id, config.audio_token_id]:
            text_input_ids[text_input_ids == token_id] = self.model_tester.pad_token_id

        with torch.no_grad():
            full_text_out = full_model(input_ids=text_input_ids).last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdir:
            full_model.save_pretrained(tmpdir)

            # 1. Text + vision only (audio_config=None)
            vision_only_model, info = EmbeddingGemma2Model.from_pretrained(
                tmpdir, audio_config=None, output_loading_info=True
            )
            self.assertEqual(len(info["missing_keys"]), 0)
            self.assertEqual(len(info["unexpected_keys"]), 0)
            self.assertIsNotNone(vision_only_model.vision_tower)
            self.assertIsNone(vision_only_model.audio_tower)
            self.assertIsNone(vision_only_model.embed_audio)
            with self.assertRaisesRegex(ValueError, "without an audio"):
                vision_only_model.get_audio_features(
                    input_features=torch.zeros(1, 16, 8),
                    input_features_mask=torch.ones(1, 16, dtype=torch.bool),
                )

            # 2. Text + audio only (vision_config=None)
            audio_only_model, info = EmbeddingGemma2Model.from_pretrained(
                tmpdir, vision_config=None, output_loading_info=True
            )
            self.assertEqual(len(info["missing_keys"]), 0)
            self.assertEqual(len(info["unexpected_keys"]), 0)
            self.assertIsNone(audio_only_model.vision_tower)
            self.assertIsNone(audio_only_model.embed_vision)
            self.assertIsNotNone(audio_only_model.audio_tower)
            with self.assertRaisesRegex(ValueError, "without a vision"):
                audio_only_model.get_image_features(
                    pixel_values=inputs_dict["pixel_values"],
                    image_position_ids=inputs_dict["image_position_ids"],
                )
            with self.assertRaisesRegex(ValueError, "without a vision"):
                audio_only_model.get_video_features(
                    pixel_values_videos=inputs_dict["pixel_values"],
                    video_position_ids=inputs_dict["image_position_ids"],
                    num_frames_per_video=torch.tensor([self.model_tester.batch_size]),
                )

            # 3. Text only via EmbeddingGemma2Model (vision_config=None, audio_config=None)
            text_via_composite, info = EmbeddingGemma2Model.from_pretrained(
                tmpdir, vision_config=None, audio_config=None, output_loading_info=True
            )
            self.assertEqual(len(info["missing_keys"]), 0)
            self.assertEqual(len(info["unexpected_keys"]), 0)
            self.assertIsNone(text_via_composite.vision_tower)
            self.assertIsNone(text_via_composite.audio_tower)
            text_via_composite = text_via_composite.to(torch_device).eval()
            with torch.no_grad():
                composite_text_out = text_via_composite(input_ids=text_input_ids).last_hidden_state
            torch.testing.assert_close(composite_text_out, full_text_out)

            # 4. Text only via EmbeddingGemma2TextModel directly
            text_model, info = EmbeddingGemma2TextModel.from_pretrained(
                tmpdir, config=config.text_config, output_loading_info=True
            )
            self.assertEqual(len(info["missing_keys"]), 0)
            self.assertEqual(len(info["unexpected_keys"]), 0)
            text_model = text_model.to(torch_device).eval()
            with torch.no_grad():
                direct_text_out = text_model(input_ids=text_input_ids).last_hidden_state
            torch.testing.assert_close(direct_text_out, full_text_out)
