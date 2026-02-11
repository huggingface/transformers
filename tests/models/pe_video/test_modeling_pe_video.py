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
import unittest

from transformers import PeVideoConfig, PeVideoEncoderConfig
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


if is_torch_available():
    import torch

    from transformers import (
        ModernBertConfig,
        PeVideoEncoder,
        PeVideoModel,
    )


class PeVideoEncoderTester:
    def __init__(
        self,
        parent,
        config_kwargs={
            "vision_config": {
                "architecture": "vit_pe_core_large_patch14_336",
                "model_args": {
                    "embed_dim": 64,
                    "img_size": (14, 14),
                    "depth": 2,
                },
                "num_classes": 4,
            },
            "hidden_size": 32,
            "intermediate_size": 37,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "hidden_act": "silu",
            "max_position_embeddings": 512,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-5,
            "use_cache": True,
            "rope_theta": 20000,
            "rope_scaling": None,
            "attention_bias": False,
            "max_window_layers": 28,
            "attention_dropout": 0.0,
        },
        batch_size=4,
        num_frames=8,
        num_channels=3,
        is_training=True,
    ):
        self.parent = parent

        self.config_kwargs = config_kwargs
        for key, value in config_kwargs.items():
            setattr(self, key, value)

        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.is_training = is_training

    @property
    def seq_length(self):
        # seq_length is what gets fed to the transformer
        # we add 1 because we add the class token
        return self.num_frames + 1

    def prepare_config_and_inputs(self):
        pixel_values_videos = floats_tensor(
            [
                self.batch_size,
                self.num_frames,
                self.num_channels,
                self.config_kwargs["vision_config"]["model_args"]["img_size"][0],
                self.config_kwargs["vision_config"]["model_args"]["img_size"][1],
            ]
        )
        # Generate valid_lengths in range [1, num_frames] to ensure at least one valid frame
        valid_lengths = ids_tensor([self.batch_size], self.num_frames - 1) + 1
        padding_mask_videos = torch.arange(self.num_frames, device=torch_device).unsqueeze(0) < valid_lengths[:, None]
        padding_mask_videos = padding_mask_videos.int()
        config = self.get_config()

        return config, pixel_values_videos, padding_mask_videos

    def get_config(self):
        return PeVideoEncoderConfig(**self.config_kwargs)

    def create_and_check_model(self, config, pixel_values_videos, padding_mask_videos):
        model = PeVideoEncoder(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values_videos, padding_mask_videos=padding_mask_videos)
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values_videos, padding_mask_videos = config_and_inputs
        inputs_dict = {"pixel_values_videos": pixel_values_videos, "padding_mask_videos": padding_mask_videos}
        return config, inputs_dict


@require_torch
class PeVideoEncoderTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (PeVideoEncoder,)
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = PeVideoEncoderTester(self)
        self.config_tester = ConfigTester(
            self, config_class=PeVideoEncoderConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Timm Eva (PE) weights cannot be fully constructed in _init_weights")
    def test_can_init_all_missing_weights(self):
        pass

    @unittest.skip(reason="PeVideoEncoder does not have usual input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip("Cannot set `output_attentions` for timm models.")
    def test_attention_outputs(self):
        pass

    @unittest.skip("TimmWrapperModel cannot be tested with meta device")
    def test_can_be_initialized_on_meta(self):
        pass

    @unittest.skip("TimmWrapperModel cannot be tested with meta device")
    def test_can_load_with_meta_device_context_manager(self):
        pass

    @unittest.skip("Cannot set `output_attentions` for timm models.")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Timm Eva (PE) weights cannot be fully constructed in _init_weights")
    def test_initialization(self):
        pass

    @unittest.skip(reason="PeVideoEncoder does not support feedforward chunking yet")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="PeAudioModel uses some timm stuff not compatible")
    def test_save_load(self):
        pass

    @unittest.skip(reason="TimmWrapperModel does not support model parallelism")
    def test_model_parallelism(self):
        pass

    @unittest.skip(reason="@eustlb this is not really expected")
    def test_batching_equivalence(self):
        pass


class PeVideoTextModelTester:
    """
    Only a ModelTester and no PeVideoTextModelTest since text model is ModernBertModel that is already tested.
    """

    def __init__(
        self,
        parent,
        config_kwargs={
            "vocab_size": 99,
            "pad_token_id": 0,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "hidden_activation": "gelu",
            "mlp_dropout": 0.0,
            "attention_dropout": 0.0,
            "embedding_dropout": 0.0,
            "classifier_dropout": 0.0,
            "max_position_embeddings": 512,
            "type_vocab_size": 16,
            "is_decoder": False,
            "initializer_range": 0.02,
        },
        batch_size=4,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
    ):
        self.parent = parent

        self.config_kwargs = config_kwargs
        for key, value in config_kwargs.items():
            setattr(self, key, value)

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return ModernBertConfig(**self.config_kwargs)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


class PeVideoModelTester:
    def __init__(self, parent, text_kwargs=None, video_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if video_kwargs is None:
            video_kwargs = {}

        self.parent = parent
        self.text_model_tester = PeVideoTextModelTester(parent, **text_kwargs)
        self.video_model_tester = PeVideoEncoderTester(parent, **video_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        _, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        _, pixel_values_videos, padding_mask_videos = self.video_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values_videos, padding_mask_videos

    def get_config(self):
        text_config = self.text_model_tester.get_config()
        video_config = self.video_model_tester.get_config()
        return PeVideoConfig(
            text_config=text_config.to_dict(),
            video_config=video_config.to_dict(),
            projection_dim=32,
        )

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values_videos, padding_mask_videos):
        model = PeVideoModel(config).to(torch_device).eval()
        with torch.no_grad():
            _ = model(input_ids, pixel_values_videos, attention_mask, padding_mask_videos)

        # TODO: there is no logits per video for now
        # self.parent.assertEqual(result.logits_per_video.shape, (self.video_model_tester.batch_size, self.text_model_tester.batch_size))
        # self.parent.assertEqual(result.logits_per_text.shape, (self.text_model_tester.batch_size, self.video_model_tester.batch_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values_videos, padding_mask_videos = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values_videos": pixel_values_videos,
            "padding_mask_videos": padding_mask_videos,
        }
        return config, inputs_dict


@require_torch
class PeVideoModelTest(ModelTesterMixin, unittest.TestCase):
    # TODO: add PipelineTesterMixin
    all_model_classes = (PeVideoModel,)
    additional_model_inputs = ["pixel_values_videos", "padding_mask_videos"]
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False
    _is_composite = True

    def setUp(self):
        self.model_tester = PeVideoModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=PeVideoConfig, has_text_modality=False, common_properties=[], hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="PeVideoModel does not have usual input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="PeVideoModel does not support feed forward chunking yet")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip("#TODO @eustlb this should be fixed tho")
    def test_save_load(self):
        pass

    @unittest.skip(reason="@eustlb this is not really expected")
    def test_batching_equivalence(self):
        pass

    @unittest.skip(reason="@eustlb this is not really expected")
    def test_can_init_all_missing_weights(self):
        pass


@require_torch
class PeVideoIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        # TODO: Add integration test when pretrained model is available
        pass
