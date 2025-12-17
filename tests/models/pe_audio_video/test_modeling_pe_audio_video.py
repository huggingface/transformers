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

from huggingface_hub import hf_hub_download

from transformers import PeAudioVideoEncoderConfig, PeAudioVideoProcessor
from transformers.testing_utils import (
    cleanup,
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
)


if is_torch_available():
    import torch

    from transformers import (
        PeAudioVideoEncoder,
        PeAudioVideoModel,
    )


class PeAudioVideoEncoderTester:
    def __init__(
        self,
        parent,
        config_kwargs={
            "audio_config": {
                "dac_config": {
                    "encoder_hidden_size": 16,
                    "downsampling_ratios": [2, 4, 4],
                    "decoder_hidden_size": 16,
                    "n_codebooks": 6,
                    "codebook_size": 512,
                    "codebook_dim": 32,
                    "quantizer_dropout": 0.0,
                    "commitment_loss_weight": 0.25,
                    "codebook_loss_weight": 1.0,
                },
                "hidden_size": 32,
                "intermediate_size": 37,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 128,
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
            "video_config": {
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
                "head_dim": 128,
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
            "hidden_size": 32,
            "intermediate_size": 37,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 128,
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
        batch_size=12,
        num_audio_channels=1,
        num_video_channels=3,
        audio_seq_length=160,
        num_frames=24,
        is_training=True,
    ):
        self.parent = parent

        self.config_kwargs = config_kwargs
        for key, value in config_kwargs.items():
            setattr(self, key, value)

        self.batch_size = batch_size
        self.num_audio_channels = num_audio_channels
        self.num_video_channels = num_video_channels
        self.audio_seq_length = audio_seq_length
        self.num_frames = num_frames
        self.is_training = is_training

    @property
    def seq_length(self):
        config = self.get_config()
        # seq_length is what gets feeded to the transformer
        # we first have to divide by hop_length to get the number of frames
        # then we add 1 because we add the class token
        return self.audio_seq_length // config.audio_config.dac_config.hop_length + 1

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.num_audio_channels, self.audio_seq_length])
        valid_audio_lengths = ids_tensor([self.batch_size], self.audio_seq_length)
        padding_mask = torch.arange(self.audio_seq_length, device=torch_device)[None, :] < valid_audio_lengths[:, None]
        padding_mask = padding_mask.int()

        pixel_values_videos = floats_tensor(
            [
                self.batch_size,
                self.num_frames,
                self.num_video_channels,
                self.config_kwargs["video_config"]["vision_config"]["model_args"]["img_size"][0],
                self.config_kwargs["video_config"]["vision_config"]["model_args"]["img_size"][1],
            ]
        )
        valid_video_lengths = ids_tensor([self.batch_size], self.num_frames)
        padding_mask_videos = (
            torch.arange(self.num_frames, device=torch_device)[None, :] < valid_video_lengths[:, None]
        )
        padding_mask_videos = padding_mask_videos.int()

        config = self.get_config()

        return config, input_values, padding_mask, pixel_values_videos, padding_mask_videos

    def get_config(self):
        if not hasattr(self, "_config"):
            self._config = PeAudioVideoEncoderConfig(**self.config_kwargs)
        return self._config

    def create_and_check_model(self, config, input_values, padding_mask, pixel_values_videos, padding_mask_videos):
        model = PeAudioVideoEncoder(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(
                input_values,
                padding_mask=padding_mask,
                pixel_values_videos=pixel_values_videos,
                padding_mask_videos=padding_mask_videos,
            )
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_values, padding_mask, pixel_values_videos, padding_mask_videos = config_and_inputs
        inputs_dict = {
            "input_values": input_values,
            "padding_mask": padding_mask,
            "pixel_values_videos": pixel_values_videos,
            "padding_mask_videos": padding_mask_videos,
        }
        return config, inputs_dict


@require_torch
class PeAudioVideoEncoderTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (PeAudioVideoEncoder,)
    additional_model_inputs = ["pixel_values_videos", "padding_mask_videos"]
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = PeAudioVideoEncoderTester(self)
        self.config_tester = ConfigTester(
            self, config_class=PeAudioVideoEncoderConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="PeAudioVideoEncoder does not have usual input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Timm Eva (PE) weights cannot be fully constructed in _init_weights")
    def test_initialization(self):
        pass

    @unittest.skip("PeAudioVideoEncoder does not have language_model, vision_tower, multi_modal_projector.")
    def test_sdpa_can_dispatch_composite_models(self):
        pass

    @unittest.skip(
        "TimmWrapperForImageClassification does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention yet."
    )
    def test_can_set_attention_dynamically_composite_model(self):
        pass

    @unittest.skip("ViT PE / TimmWrapperModel cannot be tested with meta device")
    def test_can_be_initialized_on_meta(self):
        pass

    @unittest.skip("ViT PE / TimmWrapperModel cannot be tested with meta device")
    def test_can_load_with_meta_device_context_manager(self):
        pass

    @unittest.skip("PeAudioVideoEncoder does not support feed forward chunking")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip("#TODO @eustlb this should be fixed tho")
    def test_save_load(self):
        pass

    @unittest.skip(reason="@eustlb this is not really expected")
    def test_batching_equivalence(self):
        pass

    @unittest.skip(reason="@eustlb this is not really expected just the class embedding!")
    def test_can_init_all_missing_weights(self):
        pass


@require_torch
class PeAudioVideoModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint_name = "/raid/eustache/sam-audio/converted"
        self.dtype = torch.float32
        self.processor = PeAudioVideoProcessor.from_pretrained("facebook/pe-av-large")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    @unittest.skip(reason="TODO when released")
    def test(self):
        video_path = hf_hub_download(
            repo_id="eustlb/dummy-video-dataset", filename="audiobox.mp4", repo_type="dataset"
        )
        audio_path = hf_hub_download(
            repo_id="eustlb/dummy-video-dataset", filename="audiobox.mp4", repo_type="dataset"
        )

        inputs = self.processor(
            text=["A woman and a man speaking", "A woman speaking"],
            audio=[audio_path, "/home/eustache_lebihan/add-sam-audio/audiobox_first5sec.mp4"],
            videos=[video_path, "/home/eustache_lebihan/add-sam-audio/audiobox_first5sec.mp4"],
            return_tensors="pt",
            padding=True,
        ).to(torch_device)
        model = PeAudioVideoModel.from_pretrained(
            self.checkpoint_name, dtype=self.dtype, device_map=torch_device, attn_implementation="eager"
        )

        with torch.no_grad():
            outputs = model(**inputs)
            print(outputs)
