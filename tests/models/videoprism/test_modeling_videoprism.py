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
"""Testing suite for the PyTorch VideoPrism model."""

import copy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from transformers import VideoPrismConfig, VideoPrismTextConfig, VideoPrismVisionConfig
from transformers.testing_utils import (
    Expectations,
    require_tokenizers,
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import (
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_processing_common import url_to_local_path


if is_torch_available():
    import torch
    import torch.nn.functional as F
    from torch import nn

    from transformers import (
        VideoPrismClipModel,
        VideoPrismForVideoClassification,
        VideoPrismTextModel,
        VideoPrismVideoModel,
        VideoPrismVisionModel,
    )
    from transformers.models.videoprism.modeling_videoprism import VideoPrismLayerNorm
if is_vision_available():
    from transformers import LlavaOnevisionVideoProcessor
if is_tokenizers_available():
    from transformers import VideoPrismTokenizer
torch.set_printoptions(precision=10)

TENNIS_VIDEO_URL = "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"
INTEGRATION_NUM_FRAMES = 16
INTEGRATION_FRAME_SIZE = 288
VIDEO_PRISM_LVT_CHECKPOINT = "MHRDYN7/videoprism-lvt-base-f16r288"


class VideoPrismFlashAttentionTesterMixin:
    def flash_attn_inference_equivalence(
        self, attn_implementation: str, padding_side: str, atol: float = 4e-2, rtol: float = 4e-2
    ) -> None:
        """Override: custom LayerNorm (gamma+1) amplifies eager vs flash differences."""

        def standard_layernorm_forward(self, hidden_states):
            return F.layer_norm(hidden_states, self.normalized_shape, self.weight, self.bias, self.eps)

        with patch.object(VideoPrismLayerNorm, "forward", standard_layernorm_forward):
            super().flash_attn_inference_equivalence(attn_implementation, padding_side, atol, rtol)


@require_vision
class VideoPrismVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_size=8,
        num_frames=3,
        tubelet_size=[1, 4, 4],
        num_channels=3,
        hidden_size=32,
        num_spatial_layers=3,
        num_temporal_layers=2,
        num_attention_heads=4,
        intermediate_size=64,  # a multiple of hidden size so that intermediate_size / num_attention_heads is integer
        hidden_act="gelu_python",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-06,
        qkv_bias=True,
        attn_logit_softcapping=50.0,
        num_auxiliary_layers=2,
        apply_l2norm=True,
        is_training=False,
        **kwargs,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_spatial_layers = num_spatial_layers
        self.num_temporal_layers = num_temporal_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.attn_logit_softcapping = attn_logit_softcapping
        self.num_auxiliary_layers = num_auxiliary_layers
        self.apply_l2norm = apply_l2norm
        self.is_training = is_training

        patch_size = (self.tubelet_size[1], self.tubelet_size[2])
        image_size = (self.image_size, self.image_size)
        self.num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.spatial_seq_length = self.num_patches
        self.temporal_seq_length = self.num_frames

        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [self.batch_size, self.num_frames, self.num_channels, self.image_size, self.image_size]
        )
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        config = VideoPrismVisionConfig(
            image_size=self.image_size,
            num_frames=self.num_frames,
            tubelet_size=self.tubelet_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_spatial_layers=self.num_spatial_layers,
            num_temporal_layers=self.num_temporal_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            qkv_bias=self.qkv_bias,
            attn_logit_softcapping=self.attn_logit_softcapping,
            num_auxiliary_layers=self.num_auxiliary_layers,
            apply_l2norm=self.apply_l2norm,
        )
        return config

    def create_and_check_model(self, config, pixel_values):
        model = VideoPrismVisionModel._from_config(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.tubelet_size[1], self.tubelet_size[2])
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, num_patches * self.num_frames, self.hidden_size)
        )
        self.parent.assertEqual(
            result.spatial_hidden_state.shape, (self.batch_size * self.num_frames, num_patches, self.hidden_size)
        )
        self.parent.assertEqual(
            result.temporal_hidden_state.shape, (self.batch_size * num_patches, self.num_frames, self.hidden_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values_videos": pixel_values}
        return config, inputs_dict


@require_vision
class VideoPrismVisionModelTest(VideoPrismFlashAttentionTesterMixin, ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as VideoPrismVisionModel does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (VideoPrismVisionModel, VideoPrismVideoModel) if is_torch_available() else ()

    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = VideoPrismVisionModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=VideoPrismVisionConfig,
            has_text_modality=False,
            hidden_size=37,
            common_properties=["num_channels", "hidden_size", "num_attention_heads"],
        )

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), nn.Module)
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_attention_outputs(self):
        """ViViT-style attention test for the spatial then temporal VideoPrismVisionModel stack."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        model_class = VideoPrismVisionModel

        num_spatial_layers = self.model_tester.num_spatial_layers
        num_temporal_layers = self.model_tester.num_temporal_layers
        num_patches = self.model_tester.num_patches
        num_frames = self.model_tester.num_frames
        num_attention_heads = self.model_tester.num_attention_heads

        inputs_dict["output_attentions"] = True
        inputs_dict["output_hidden_states"] = False
        model = model_class._from_config(config, attn_implementation="eager")
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
        attentions = outputs.attentions
        self.assertEqual(len(attentions), num_spatial_layers + num_temporal_layers)

        del inputs_dict["output_attentions"]
        config.output_attentions = True
        model = model_class._from_config(config, attn_implementation="eager")
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
        attentions = outputs.attentions
        self.assertEqual(len(attentions), num_spatial_layers + num_temporal_layers)

        for layer_idx in range(num_spatial_layers):
            self.assertListEqual(
                list(attentions[layer_idx].shape[-3:]),
                [num_attention_heads, num_patches, num_patches],
            )
        for layer_idx in range(num_spatial_layers, num_spatial_layers + num_temporal_layers):
            self.assertListEqual(
                list(attentions[layer_idx].shape[-3:]),
                [num_attention_heads, num_frames, num_frames],
            )

        inputs_dict["output_attentions"] = True
        inputs_dict["output_hidden_states"] = True
        model = model_class._from_config(config, attn_implementation="eager")
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
        self.assertIsNotNone(outputs.attentions)
        self.assertIsNotNone(outputs.hidden_states)
        self.assertEqual(len(outputs.attentions), num_spatial_layers + num_temporal_layers)
        self.assertEqual(len(outputs.hidden_states), 1 + num_spatial_layers + num_temporal_layers)

    def test_hidden_states_output(self):
        """Hidden states: spatial tokens, then temporal tokens; last entry is last_hidden_state."""

        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class._from_config(config, attn_implementation="eager")
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states
            num_spatial_layers = self.model_tester.num_spatial_layers
            num_temporal_layers = self.model_tester.num_temporal_layers
            expected_num_layers = 1 + num_spatial_layers + num_temporal_layers
            self.assertEqual(len(hidden_states), expected_num_layers)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [self.model_tester.num_patches, self.model_tester.hidden_size],
            )
            self.assertListEqual(
                list(hidden_states[num_spatial_layers].shape[-2:]),
                [self.model_tester.num_patches, self.model_tester.hidden_size],
            )
            self.assertListEqual(
                list(hidden_states[num_spatial_layers + 1].shape[-2:]),
                [self.model_tester.num_frames, self.model_tester.hidden_size],
            )
            self.assertListEqual(
                list(hidden_states[-1].shape[-2:]),
                [
                    self.model_tester.num_patches * self.model_tester.num_frames,
                    self.model_tester.hidden_size,
                ],
            )
            torch.testing.assert_close(hidden_states[-1], outputs.last_hidden_state)

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model_class = VideoPrismVisionModel

        inputs_dict["output_hidden_states"] = True
        check_hidden_states_output(inputs_dict, config, model_class)

        del inputs_dict["output_hidden_states"]
        config.output_hidden_states = True
        check_hidden_states_output(inputs_dict, config, model_class)

    @unittest.skip(
        reason="VideoPrismVisionModel does not expose common hidden_states/attentions fields for retain-grad checks."
    )
    def test_retain_grad_hidden_states_attentions(self):
        pass

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "MHRDYN7/videoprism-base-f16r288"
        model = VideoPrismVisionModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


@require_vision
class VideoPrismTextModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        hidden_size=32,  # should be same as the hidden_size of the vision model tester
        intermediate_size=37,
        num_attention_heads=4,
        num_hidden_layers=2,
        vocab_size=99,
        apply_l2norm=True,
        hidden_act="relu",
        attention_probs_dropout_prob=0.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        layer_norm_eps=1e-06,
        initializer_range=0.02,
        attn_logit_softcapping=50.0,
        seq_length=7,
        is_training=False,
        use_input_mask=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.apply_l2norm = apply_l2norm
        self.hidden_act = hidden_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.attn_logit_softcapping = attn_logit_softcapping
        self.seq_length = seq_length
        self.encoder_seq_length = seq_length + 1
        self.key_length = seq_length + 1
        self.is_training = is_training
        self.use_input_mask = use_input_mask

    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTester.prepare_config_and_inputs
    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return VideoPrismTextConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            num_text_layers=self.num_hidden_layers,
            vocab_size=self.vocab_size,
            apply_l2norm=self.apply_l2norm,
            hidden_act=self.hidden_act,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            qkv_bias=self.qkv_bias,
            hidden_dropout_prob=self.hidden_dropout_prob,
            layer_norm_eps=self.layer_norm_eps,
            initializer_range=self.initializer_range,
            attn_logit_softcapping=self.attn_logit_softcapping,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = VideoPrismTextModel._from_config(config=config).to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.encoder_seq_length, self.hidden_size)
        )
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTester.prepare_config_and_inputs_for_common
    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_vision
class VideoPrismTextModelTest(VideoPrismFlashAttentionTesterMixin, ModelTesterMixin, unittest.TestCase):
    all_model_classes = (VideoPrismTextModel,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = VideoPrismTextModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=VideoPrismTextConfig,
            hidden_size=37,
            common_properties=["hidden_size", "num_attention_heads"],
        )

    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTest.test_config
    def test_config(self):
        self.config_tester.run_common_tests()

    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTest.test_model
    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "MHRDYN7/videoprism-lvt-base-f16r288"
        model = VideoPrismTextModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


@require_vision
class VideoPrismClipModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = VideoPrismTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = VideoPrismVisionModelTester(parent, **vision_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.is_training = is_training

    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTester.prepare_config_and_inputs
    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return VideoPrismConfig(
            text_config=self.text_model_tester.get_config().to_dict(),
            vision_config=self.vision_model_tester.get_config().to_dict(),
        )

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = VideoPrismClipModel(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(pixel_values, input_ids, attention_mask)
        self.parent.assertEqual(
            result.logits_per_video.shape, (self.vision_model_tester.batch_size, self.text_model_tester.batch_size)
        )
        self.parent.assertEqual(
            result.logits_per_text.shape, (self.text_model_tester.batch_size, self.vision_model_tester.batch_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values_videos": pixel_values,
        }
        return config, inputs_dict


@require_vision
class VideoPrismClipModelTest(VideoPrismFlashAttentionTesterMixin, ModelTesterMixin, unittest.TestCase):
    _is_composite = True
    test_attention_outputs = False
    additional_model_inputs = ["input_ids", "attention_mask"]
    test_resize_embeddings = False

    all_model_classes = (VideoPrismClipModel,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = VideoPrismClipModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=VideoPrismConfig,
            has_text_modality=False,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTest.test_model
    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTest.test_hidden_states_output
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTest.test_retain_grad_hidden_states_attentions
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(
        reason="VideoPrismClipModel normalizes exp(similarity) across the batch, so logits are batch-dependent by design."
    )
    def test_batching_equivalence(self):
        pass

    @unittest.skip(reason="SDPA is turned off for this model.")
    def test_can_set_attention_dynamically_composite_model(self):
        pass

    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTest.test_load_vision_text_config with CLIP->VideoPrism
    def test_load_vision_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save VideoPrismConfig and check if we can load VideoPrismVisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = VideoPrismVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save VideoPrismConfig and check if we can load VideoPrismTextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = VideoPrismTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        model_name = "MHRDYN7/videoprism-lvt-base-f16r288"
        model = VideoPrismClipModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def _test_get_text_features_output(self, return_dict):
        config, inputs_dict = self._text_features_prepare_config_and_inputs()
        if return_dict is not None:
            config.return_dict = return_dict

        model = VideoPrismClipModel(config).eval().to(torch_device)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs_dict)

        if return_dict in (True, None):
            expected_shape = (
                inputs_dict["input_ids"].shape[0],
                self.model_tester.text_model_tester.encoder_seq_length,
                config.text_config.hidden_size,
            )
            self.assertEqual(outputs.last_hidden_state.shape, expected_shape)
        else:
            self.assertIsInstance(outputs, tuple)

    def test_get_text_features_output_0(self):
        self._test_get_text_features_output(return_dict=True)

    def test_get_text_features_output_1(self):
        self._test_get_text_features_output(return_dict=False)

    def test_get_text_features_output_2(self):
        self._test_get_text_features_output(return_dict=None)

    def _video_features_expected_num_layers(self):
        vision_tester = self.model_tester.vision_model_tester
        return vision_tester.num_spatial_layers + vision_tester.num_temporal_layers

    def test_get_video_features_hidden_states(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(copy.deepcopy(config))
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model.get_video_features(**inputs_dict)

            hidden_states = outputs.hidden_states
            expected_num_hidden_states = self._video_features_expected_num_layers() + 1
            self.assertIsNotNone(hidden_states)
            self.assertEqual(len(hidden_states), expected_num_hidden_states)

        config, inputs_dict = self._video_features_prepare_config_and_inputs()

        inputs_dict["output_hidden_states"] = True
        check_hidden_states_output(inputs_dict, config, VideoPrismClipModel)

        del inputs_dict["output_hidden_states"]
        config.output_hidden_states = True
        for k in config.sub_configs:
            if getattr(config, k) is not None:
                getattr(config, k).output_hidden_states = True

        check_hidden_states_output(inputs_dict, config, VideoPrismClipModel)

    def test_get_video_features_attentions(self):
        def check_attentions_output(inputs_dict, config, model_class):
            model = model_class(copy.deepcopy(config))
            model.set_attn_implementation("eager")
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model.get_video_features(**inputs_dict)

            attentions = outputs.attentions
            expected_num_attentions = self._video_features_expected_num_layers()
            self.assertIsNotNone(attentions)
            self.assertEqual(len(attentions), expected_num_attentions)

        if not self.has_attentions:
            return

        config, inputs_dict = self._video_features_prepare_config_and_inputs()
        inputs_dict["output_hidden_states"] = False
        inputs_dict["output_attentions"] = True
        check_attentions_output(inputs_dict, config, VideoPrismClipModel)

        del inputs_dict["output_attentions"]
        config.output_attentions = True
        for k in config.sub_configs:
            if getattr(config, k) is not None:
                getattr(config, k).output_attentions = True

        check_attentions_output(inputs_dict, config, VideoPrismClipModel)


@require_vision
class VideoPrismForVideoClassificationModelTester(ModelTesterMixin, VideoPrismVisionModelTester):
    def __init__(self, parent, vision_kwargs=None, is_training=True):
        if vision_kwargs is None:
            vision_kwargs = {}
        super().__init__(parent, **vision_kwargs)

    def get_config(self):
        config = super().get_config()
        config.num_labels = self.num_labels
        return config

    def prepare_config_and_inputs(self):
        config, pixel_values = super().prepare_config_and_inputs()
        labels = ids_tensor([self.batch_size], self.num_labels) if self.use_labels else None
        return config, pixel_values, labels

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, _ = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values_videos": pixel_values}
        return config, inputs_dict

    def create_and_check_model(self, config, pixel_values, labels):
        model = VideoPrismForVideoClassification._from_config(config=config)
        model.to(torch_device)
        pixel_values = pixel_values.to(torch_device)
        labels = labels.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values, labels=labels)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.tubelet_size[1], self.tubelet_size[2])
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.loss.shape, torch.Size([]))
        self.parent.assertEqual(result.logits.shape, (self.batch_size, 1, self.num_labels))
        self.parent.assertEqual(
            result.hidden_states.shape, (self.batch_size, num_patches * self.num_frames, self.hidden_size)
        )


@require_vision
class VideoPrismForVideoClassificationTest(VideoPrismFlashAttentionTesterMixin, ModelTesterMixin, unittest.TestCase):
    all_model_classes = (VideoPrismForVideoClassification,) if is_torch_available() else ()
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = VideoPrismForVideoClassificationModelTester(
            self, vision_kwargs={"use_labels": True, "num_labels": 10}
        )
        self.config_tester = ConfigTester(
            self,
            config_class=VideoPrismVisionConfig,
            has_text_modality=False,
            hidden_size=37,
            common_properties=["num_channels", "hidden_size", "num_attention_heads"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), nn.Module)
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    @unittest.skip(reason="VideoPrismForVideoClassification does not expose top-level attentions")
    def test_attention_outputs(self):
        pass

    @unittest.skip(
        reason="VideoPrismForVideoClassification returns a single hidden_states tensor, not layer-wise hidden states"
    )
    def test_hidden_states_output(self):
        pass

    @unittest.skip(
        reason="VideoPrismForVideoClassification does not expose common hidden_states/attentions fields for retain-grad checks"
    )
    def test_retain_grad_hidden_states_attentions(self):
        pass


def prepare_tennis_frames():
    tennis_video = url_to_local_path(TENNIS_VIDEO_URL)
    video_processor = LlavaOnevisionVideoProcessor(
        size={"height": INTEGRATION_FRAME_SIZE, "width": INTEGRATION_FRAME_SIZE},
        do_normalize=False,
    )
    return tennis_video, video_processor(
        videos=tennis_video,
        return_tensors="pt",
        do_sample_frames=True,
        num_frames=INTEGRATION_NUM_FRAMES,
    )["pixel_values_videos"]


def prepare_texts():
    text_query_csv = "playing drums,sitting,playing flute,playing at playground,concert"
    prompt_template = "a video of {}."

    text_queries = text_query_csv.split(",")
    text_queries = [prompt_template.format(t) for t in text_queries]
    tokenizer = VideoPrismTokenizer.from_pretrained(VIDEO_PRISM_LVT_CHECKPOINT)
    return tokenizer, text_queries


@require_vision
@require_torch
@require_tokenizers
class VideoPrismModelIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tennis_video, cls.tennis_frames = prepare_tennis_frames()
        cls.tokenizer, cls.text_queries = prepare_texts()

    @slow
    def test_videoprism_vision_model(self):
        model = VideoPrismVisionModel.from_pretrained("MHRDYN7/videoprism-base-f16r288").to(torch_device)
        input_vids = torch.cat([self.tennis_frames, self.tennis_frames], dim=0).to(torch_device)
        model.eval()
        with torch.inference_mode():
            outputs = model(input_vids).last_hidden_state

        self.assertListEqual(
            outputs[0].cpu().tolist(),
            outputs[1].cpu().tolist(),
            "Outputs of the batches are not identical for identical input batches",
        )
        expectations = Expectations(
            {
                (None, None): [
                    [0.4354448914527893, 0.4073091447353363, -0.29193729162216187],
                    [0.21557867527008057, 0.24542216956615448, 0.25062084197998047],
                    [0.16283036768436432, 0.11620327830314636, 0.008987100794911385],
                ],
                ("cuda", 8): [
                    [0.43544602394104004, 0.4073105454444885, -0.2919350862503052],
                    [0.21557006239891052, 0.24542272090911865, 0.2506211996078491],
                    [0.16283579170703888, 0.11620290577411652, 0.008984619751572609],
                ],
            }
        )
        expected_values = torch.tensor(expectations.get_expectation(), device=torch_device)
        output_slice = outputs[0, :3, :3]
        torch.testing.assert_close(output_slice, expected_values, rtol=2e-4, atol=2e-4)

    @slow
    def test_videoprism_clip_model(self):
        model = VideoPrismClipModel.from_pretrained("MHRDYN7/videoprism-lvt-base-f16r288").to(torch_device)
        input_vids = torch.cat([self.tennis_frames, self.tennis_frames], dim=0).to(torch_device)
        tokens = self.tokenizer(self.text_queries, max_length=64, padding="max_length", return_tensors="pt").to(
            torch_device
        )
        model.eval()
        with torch.inference_mode():
            outputs = model(input_vids, **tokens)
        torch.testing.assert_close(outputs.video_embeds[0], outputs.video_embeds[1], rtol=2e-4, atol=2e-4)

        self.assertEqual(
            outputs.logits_per_video.shape,
            torch.Size((input_vids.shape[0], tokens.input_ids.shape[0])),
        )
        self.assertEqual(
            outputs.logits_per_text.shape,
            torch.Size((tokens.input_ids.shape[0], input_vids.shape[0])),
        )

        video_expectation = Expectations(
            {
                (None, None): [
                    -0.0022147062700241804,
                    -0.015442248433828354,
                    0.026582615450024605,
                    0.024988114833831787,
                    0.023289205506443977,
                    0.03686177730560303,
                    -0.016299977898597717,
                    0.010566001757979393,
                    -0.016186168417334557,
                ],
                ("cuda", 8): [
                    -0.002214818261563778,
                    -0.015442408621311188,
                    0.026582621037960052,
                    0.02498835325241089,
                    0.023289136588573456,
                    0.03686164692044258,
                    -0.016299953684210777,
                    0.010566022247076035,
                    -0.016185807064175606,
                ],
            }
        )
        text_expectation = Expectations(
            {
                (None, None): [
                    [-0.008009868673980236, 0.009317189455032349, 0.015544884838163853],
                    [0.022461067885160446, 9.54712595557794e-05, -0.01074187271296978],
                    [-0.022578040137887, 0.001339073060080409, -0.015561817213892937],
                    [0.010591105557978153, 0.018359515815973282, -0.015389746055006981],
                    [-0.003638886846601963, 0.0036980013828724623, 0.007990806363523006],
                ],
                ("cuda", 8): [
                    [-0.00800985936075449, 0.009317193180322647, 0.015544882975518703],
                    [0.022461047396063805, 9.546728688292205e-05, -0.010741823352873325],
                    [-0.022578010335564613, 0.0013390942476689816, -0.015561779029667377],
                    [0.01059112511575222, 0.018359506502747536, -0.015389740467071533],
                    [-0.0036388880107551813, 0.003698008367791772, 0.007990810088813305],
                ],
            }
        )

        video_expected_values = torch.tensor(video_expectation.get_expectation(), device=torch_device)
        text_expected_values = torch.tensor(text_expectation.get_expectation(), device=torch_device)
        video_logits = outputs.video_embeds[0, :9]
        text_logits = outputs.text_embeds[:, :3]
        torch.testing.assert_close(video_logits, video_expected_values, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(text_logits, text_expected_values, rtol=2e-4, atol=2e-4)

    @slow
    def test_videoprism_interpolate_pos_encoding(self):
        model_name = "MHRDYN7/videoprism-base-f16r288"
        model = VideoPrismVisionModel.from_pretrained(model_name).to(torch_device)
        processor = LlavaOnevisionVideoProcessor.from_pretrained(model_name)
        kwargs = {
            "num_frames": 10,
            "size": {"height": 144, "width": 144},
            "do_resize": True,
        }
        inputs = processor(videos=self.tennis_video, return_tensors="pt", **kwargs).to(torch_device)
        model.eval()
        with torch.inference_mode():
            outputs = model(**inputs, interpolate_pos_encoding=True)

        expected_shape = torch.Size([1, int((144 / 18) * (144 / 18) * 10), model.config.hidden_size])
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)
