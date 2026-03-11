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
"""Testing suite for the PyTorch VideoPrism model."""

import tempfile
import unittest

import numpy as np
from huggingface_hub import HfApi
from parameterized import parameterized

from transformers import VideoPrismConfig, VideoPrismTextConfig, VideoPrismVisionConfig
from transformers.testing_utils import (
    require_sentencepiece,
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import (
    is_sentencepiece_available,
    is_torch_available,
    is_vision_available,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        VideoPrismClipModel,
        VideoPrismForVideoClassification,
        VideoPrismTextModel,
        VideoPrismVideoModel,
        VideoPrismVisionModel,
    )
if is_vision_available():
    from transformers import VideoPrismVideoProcessor
if is_sentencepiece_available():
    from transformers import VideoPrismTokenizer


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
        apply_l2_norm=True,
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
        self.apply_l2_norm = apply_l2_norm
        self.is_training = is_training

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
            apply_l2_norm=self.apply_l2_norm,
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
class VideoPrismVisionModelTest(ModelTesterMixin, unittest.TestCase):
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

    @unittest.skip(
        reason="VideoPrismVisionModel exposes spatial/temporal backbone states, not a single hidden_states tuple."
    )
    def test_hidden_states_output(self):
        pass

    @unittest.skip(
        reason="VideoPrismVisionModel does not expose a single attentions tuple compatible with ModelTesterMixin."
    )
    def test_attention_outputs(self):
        pass

    @unittest.skip(
        reason="VideoPrismVisionModel does not expose common hidden_states/attentions fields for retain-grad checks."
    )
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip(reason="VideoPrism reference outputs are validated only with eager attention.")
    def test_eager_matches_sdpa_inference(
        self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
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
        num_attention_heads=2,
        num_hidden_layers=2,
        vocab_size=32,
        apply_l2_norm=True,
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
        self.apply_l2_norm = apply_l2_norm
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
            vocab_size=self.vocab_size,
            apply_l2_norm=self.apply_l2_norm,
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
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.hidden_size))

    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTester.prepare_config_and_inputs_for_common
    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_vision
class VideoPrismTextModelTest(ModelTesterMixin, unittest.TestCase):
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

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip(reason="VideoPrism reference outputs are validated only with eager attention.")
    def test_eager_matches_sdpa_inference(
        self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        pass

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
class VideoPrismClipModelTest(ModelTesterMixin, unittest.TestCase):
    _is_composite = True
    test_attention_outputs = False

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

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip(reason="VideoPrism reference outputs are validated only with eager attention.")
    def test_eager_matches_sdpa_inference(
        self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        pass

    @unittest.skip(reason="VideoPrism composite model is only validated with eager attention.")
    def test_sdpa_can_dispatch_composite_models(self):
        pass

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


@require_vision
class VideoPrismForVideoClassificationModelTester(ModelTesterMixin, VideoPrismVisionModelTester):
    def __init__(self, parent, vision_kwargs=None, is_training=True):
        if vision_kwargs is None:
            vision_kwargs = {}
        super().__init__(parent, **vision_kwargs)

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
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
class VideoPrismForVideoClassificationTest(ModelTesterMixin, unittest.TestCase):
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

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip(reason="VideoPrism reference outputs are validated only with eager attention.")
    def test_eager_matches_sdpa_inference(
        self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        pass


def prepare_video(video_type="water_bottle_drumming"):
    """
    Returns the input video array preprocessed using the original repo's processor if frames=True, else returns the original video file.
    """

    api = HfApi()
    if video_type == "water_bottle_drumming":
        filename = "water_bottle_drumming.mp4"
    elif video_type == "water_bottle_drumming_frames":
        filename = "frames_16_288.npy"
    elif video_type == "basketball_dunk":
        filename = "v_BasketballDunk_g14_c06.avi"
    else:
        raise ValueError(
            "The `video_type` should be one of ['water_bottle_drumming', 'water_bottle_drumming_frames', 'basketball_dunk']."
        )

    file = api.hf_hub_download(repo_id="MHRDYN7/videoprism_assets", filename=filename, repo_type="dataset")
    if video_type == "water_bottle_drumming_frames":
        file = np.load(file)
    return file


def prepare_texts():
    text_query_csv = "playing drums,sitting,playing flute,playing at playground,concert"
    prompt_template = "a video of {}."

    text_queries = text_query_csv.split(",")
    text_queries = [prompt_template.format(t) for t in text_queries]
    tokenizer = VideoPrismTokenizer.from_pretrained("MHRDYN7/videoprism-lvt-base-f16r288")
    return tokenizer, text_queries


@require_vision
@require_torch
@require_sentencepiece
class VideoPrismModelIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.water_bottle_drumming_frames = (
            torch.tensor(prepare_video("water_bottle_drumming_frames")).unsqueeze(0).permute(0, 1, 4, 2, 3)
        )
        cls.water_bottle_drumming_video = prepare_video("water_bottle_drumming")
        cls.basketball_dunk_video = prepare_video("basketball_dunk")
        cls.tokenizer, cls.text_queries = prepare_texts()

    @slow
    def test_videoprism_vision_model(self):
        model = VideoPrismVisionModel.from_pretrained(
            "MHRDYN7/videoprism-base-f16r288", attn_implementation="eager"
        ).to(torch_device)
        input_vids = torch.cat([self.water_bottle_drumming_frames, self.water_bottle_drumming_frames], dim=0).to(
            torch_device
        )
        model.eval()
        with torch.inference_mode():
            outputs = model(input_vids).last_hidden_state

        self.assertListEqual(
            outputs[0].cpu().tolist(),
            outputs[1].cpu().tolist(),
            "Outputs of the batches are not identical for identical input batches",
        )
        expectations = torch.tensor(
            [
                [0.11648951, 0.4568253, 0.19288044],
                [0.28420594, -0.04224018, 0.377879],
                [0.24594213, -0.3914095, -0.30516925],
            ]
        )
        expected_slice = outputs[0, :3, :3].cpu()
        torch.testing.assert_close(expected_slice, expectations, rtol=1e-5, atol=1e-5)

    @slow
    def test_videoprism_clip_model(self):
        model = VideoPrismClipModel.from_pretrained("MHRDYN7/videoprism-lvt-base-f16r288").to(torch_device)
        model.config._attn_implementation = "eager"
        input_vids = torch.cat([self.water_bottle_drumming_frames, self.water_bottle_drumming_frames], dim=0).to(
            torch_device
        )
        tokens = self.tokenizer(self.text_queries, max_length=64, padding="max_length", return_tensors="pt").to(
            torch_device
        )
        model.eval()
        with torch.inference_mode():
            outputs = model(input_vids, **tokens)
        torch.testing.assert_close(outputs.video_embeds[0], outputs.video_embeds[1], rtol=1e-5, atol=1e-5)

        self.assertEqual(
            outputs.logits_per_video.shape,
            torch.Size((input_vids.shape[0], tokens.input_ids.shape[0])),
        )
        self.assertEqual(
            outputs.logits_per_text.shape,
            torch.Size((tokens.input_ids.shape[0], input_vids.shape[0])),
        )

        video_expectation = torch.tensor(
            [
                -0.01940615,
                -0.04830061,
                0.0069022,
                0.02915299,
                -0.05897291,
                0.02168823,
                -0.01471708,
                -0.00971614,
                -0.00220576,
            ]
        )
        text_expectation = torch.tensor(
            [
                [-0.00802545, 0.00931361, 0.01555958],
                [0.02245245, 0.00010197, -0.01073526],
                [-0.02258418, 0.00133927, -0.01555064],
                [0.01056228, 0.01835608, -0.01539922],
                [-0.00366718, 0.00370416, 0.00800336],
            ]
        )

        video_logits = outputs.video_embeds[0, :9].cpu()
        text_logits = outputs.text_embeds[:, :3].cpu()
        torch.testing.assert_close(video_logits, video_expectation, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(text_logits, text_expectation, rtol=1e-5, atol=1e-5)

    @slow
    def test_videoprism_interpolate_pos_encoding(self):
        model_name = "MHRDYN7/videoprism-base-f16r288"
        model = VideoPrismVisionModel.from_pretrained(model_name).to(torch_device)
        processor = VideoPrismVideoProcessor.from_pretrained(model_name)
        kwargs = {
            "num_frames": 10,
            "size": {"height": 144, "width": 144},
            "do_resize": True,
        }
        inputs = processor(videos=self.water_bottle_drumming_video, return_tensors="pt", **kwargs).to(torch_device)
        model.eval()
        with torch.inference_mode():
            outputs = model(**inputs, interpolate_pos_encoding=True)

        expected_shape = torch.Size([1, int((144 / 18) * (144 / 18) * 10), model.config.hidden_size])
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

    @slow
    def test_videoprism_classification_model(self):
        model_name = "MHRDYN7/videoprism-base-f16r288-finetuned-ucf101"
        model = VideoPrismForVideoClassification.from_pretrained(model_name).to(torch_device)
        processor = VideoPrismVideoProcessor.from_pretrained(model_name)
        inputs = processor(videos=self.basketball_dunk_video, return_tensors="pt")["pixel_values_videos"].to(
            torch_device
        )
        label = torch.tensor([8], dtype=torch.long, device=torch_device)
        model.eval()
        with torch.inference_mode():
            outputs = model(inputs, labels=label)

        expected_logits = torch.tensor(
            [
                [
                    [-18.5863, -12.8547, -4.8901, -8.7695, 15.0777, 15.0308, -0.2944, 0.5263, 22.7533, 5.9714],
                ]
            ]
        )
        torch.testing.assert_close(outputs.logits.cpu(), expected_logits, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(outputs.loss.cpu(), torch.tensor(0.0009), rtol=1e-4, atol=1e-4)
