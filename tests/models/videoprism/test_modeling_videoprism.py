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


import inspect
import torch
import torch.nn as nn
import unittest
from huggingface_hub import HfApi
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_configuration_common import ConfigTester
from transformers import VideoPrismVisionConfig, VideoPrismTextConfig, VideoPrismConfig
from transformers.testing_utils import (
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import (
    is_torch_available,
    is_vision_available,
    is_sentencepiece_available,
)

if is_torch_available():
    from transformers import VideoPrismVisionModel, VideoPrismVideoModel, VideoPrismTextModel, VideoPrismClipModel

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
        intermediate_size=37,
        hidden_act="gelu_python",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-06,
        qkv_bias=True,
        attn_logit_softcapping=50.0,
        num_auxiliary_layers=2,
        apply_l2_norm=True,
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
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches * self.num_frames, self.hidden_size))
        self.parent.assertEqual(result.spatial_hidden_state.shape, (self.batch_size * self.num_frames, num_patches, self.hidden_size))
        self.parent.assertEqual(result.temporal_hidden_state.shape, (self.batch_size * num_patches, self.num_frames, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values_videos": pixel_values}
        return config, inputs_dict


@require_vision
class VideoPrismVisionModelTest(unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as VideoPrismVisionModel does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (VideoPrismVisionModel,) if is_torch_available() else ()
    pipeline_model_mapping = ()

    def setUp(self):
        self.model_tester = VideoPrismVisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VideoPrismVisionConfig, has_text_modality=False, hidden_size=37)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)

        if return_labels:
            if model_class in get_values(MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING):
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )

        return inputs_dict
    @unittest.skip(reason="VideoPrism does not use common configs")
    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="VideoPrism does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            self.assertEqual(arg_names[0], "pixel_values_videos")

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    # @unittest.skip(reason="VideoPrismVisionModel does not support standalone training")
    # def test_training(self):
    #     pass

    # @unittest.skip(reason="VideoPrismVisionModel does not support standalone training")
    # def test_training_gradient_checkpointing(self):
    #     pass

    # @unittest.skip(reason="VideoPrismVisionModel does not support standalone training")
    # def test_training_gradient_checkpointing_use_reentrant(self):
    #     pass

    # @unittest.skip(reason="VideoPrismVisionModel does not support standalone training")
    # def test_training_gradient_checkpointing_use_reentrant_false(self):
    #     pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "MHRDYN7/videoprism-base-f16r288"
        model = VideoPrismVisionModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


@require_vision
class VideoPrismTextModelTester:
    pass

@require_vision
class VideoPrismTextModelTest(unittest.TestCase):
    pass

@require_vision
class VideoPrismVideoModelTester:
    pass

@require_vision
class VideoPrismVideoModelTest(unittest.TestCase):
    pass

@require_vision
class VideoPrismClipModelTester:
    pass

@require_vision
class VideoPrismClipModelTest(unittest.TestCase):
    pass

@require_torch
class VideoPrismImageClassificationModelTester:
    pass

@require_torch
class VideoPrismImageClassificationModelTest(unittest.TestCase):
    pass



def prepare_video():
    """
    Input video tensor proprocessed using the original repo's processor
    """
    import numpy as np
    api = HfApi()
    frames = api.hf_hub_download(
        repo_id="MHRDYN7/water_bottle_drumming_video",
        filename="frames_16_288.npy",
        repo_type="dataset"
    )
    return np.load(frames)

def prepare_texts():
    TEXT_QUERY_CSV = "playing drums,sitting,playing flute,playing at playground,concert"  # @param {type: "string"}
    PROMPT_TEMPLATE = "a video of {}."

    text_queries = TEXT_QUERY_CSV.split(",")
    text_queries = [PROMPT_TEMPLATE.format(t) for t in text_queries]

    tokenizer = VideoPrismTokenizer.from_pretrained("MHRDYN7/videoprism-lvt-base-f16r288")

    return tokenizer, text_queries
    

@require_vision
@require_torch
class VideoPrismModelIntegrationTest(unittest.TestCase):
    @slow
    def test_vision_model(self):
        model = VideoPrismVisionModel.from_pretrained("MHRDYN7/videoprism-base-f16r288")
        model.config._attn_implementation = "eager"
        frames = torch.tensor(prepare_video()).unsqueeze(0).permute(0, 1, 4, 2, 3)
        input_vids = torch.cat([frames, frames], dim=0)  # batch size 2
        outputs = model(input_vids).last_hidden_state
        assert torch.equal(outputs[0], outputs[1]), "Outputs of the batches are not identical for identical input batches"
        expectations = torch.tensor(
            [
                [0.11648951, 0.4568253, 0.19288044],
                [0.28420594, -0.04224018, 0.377879],
                [0.24594213, -0.3914095, -0.30516925],
            ]
        )
        expected_slice = outputs[0, :3, :3]
        torch.testing.assert_close(expected_slice, expectations, atol=1e-5)
        return

    @slow
    def test_clip_model(self):
        model = VideoPrismClipModel.from_pretrained("MHRDYN7/videoprism-lvt-base-f16r288")
        model.config._attn_implementation = "eager"
        frames = torch.tensor(prepare_video()).unsqueeze(0).permute(0, 1, 4, 2, 3)
        input_vids = torch.cat([frames, frames], dim=0)
        tokenizer, text_queries = prepare_texts()
        tokens = tokenizer(text_queries, max_length=64, padding="max_length", return_tensors="pt")
        outputs = model(input_vids, **tokens)
        torch.testing.assert_close(outputs.video_embeds[0], outputs.video_embeds[1], atol=1e-5)
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
        ),
        text_expectation = torch.tensor(
            [
                [-0.00802545, 0.00931361, 0.01555958],
                [0.02245245, 0.00010197, -0.01073526],
                [-0.02258418, 0.00133927, -0.01555064],
                [0.01056228, 0.01835608, -0.01539922],
                [-0.00366718, 0.00370416, 0.00800336],
            ]
        ),

        video_logits = outputs.video_embeds[0, :9]
        text_logits = outputs.text_embeds[:, :3]
        torch.testing.assert_close(video_logits, video_expectation, atol=1e-5)
        torch.testing.assert_close(text_logits, text_expectation, atol=1e-5)

