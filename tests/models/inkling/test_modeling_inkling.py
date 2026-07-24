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
"""Testing suite for the PyTorch Inkling model."""

import os
import tempfile
import unittest

from huggingface_hub import download_bucket_files
from parameterized import parameterized
from safetensors.torch import load_file

from transformers import (
    AutoProcessor,
    InklingConfig,
    InklingTextConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTester
from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        InklingForConditionalGeneration,
        InklingModel,
        InklingTextModel,
    )


GEMMA4_RANDOM_MOE_FA2_SKIP_REASON = (
    "Randomly initialized Inkling MoE routers are too sensitive to tiny eager/FA2 input differences"
)


class InklingTextModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = InklingTextConfig
        base_model_class = InklingTextModel
        causal_lm_class = InklingForConditionalGeneration

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hidden_layers = 2
        # we want to test sharing on both types
        self.layer_types = ["hybrid_sliding", "hybrid"]
        self.mlp_layer_types = ["dense", "sparse"]
        self.swa_num_attention_heads = self.num_attention_heads
        self.swa_num_key_value_heads = self.num_key_value_heads
        self.swa_head_dim = self.head_dim

        # To activate moe blocks
        self.enable_moe_block = True
        self.moe_intermediate_size = 16


class InklingAudio2TextModelTester:
    def __init__(
        self,
        parent,
        image_token_id=4,
        boi_token_id=5,
        eoi_token_id=6,
        audio_token_id=7,
        video_token_id=10,
        seq_length=50,
        audio_num_frames=4,
        n_mel_bins=4,
        mel_vocab_size=8,
        is_training=True,
    ):
        self.parent = parent
        self.image_token_id = image_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.audio_token_id = audio_token_id
        self.video_token_id = video_token_id
        self.llm_tester = InklingTextModelTester(self.parent)
        self.llm_tester.use_bidirectional_attention = None
        self.text_config = self.llm_tester.get_config()
        self.audio_num_frames = audio_num_frames
        self.n_mel_bins = n_mel_bins
        self.mel_vocab_size = mel_vocab_size
        self.audio_config = {
            "hidden_size": self.text_config.hidden_size,
            "n_mel_bins": n_mel_bins,
            "mel_vocab_size": mel_vocab_size,
        }
        self.seq_length = seq_length
        self.pad_token_id = self.text_config.pad_token_id

        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size
        self.num_attention_heads = self.text_config.num_attention_heads
        self.is_training = is_training

        self.batch_size = 3
        self.encoder_seq_length = seq_length

    def get_config(self):
        config = InklingConfig(
            text_config=self.text_config,
            vision_config={"patch_size": 5, "num_hidden_layers": 2, "num_channels": 3},
            audio_config=self.audio_config,
            image_token_id=self.image_token_id,
            boi_token_id=self.boi_token_id,
            eoi_token_id=self.eoi_token_id,
            audio_token_id=self.audio_token_id,
            video_token_id=self.video_token_id,
        )
        config.num_hidden_layers = config.text_config.num_hidden_layers
        return config

    def prepare_config_and_inputs(self):
        audio_input_ids = ids_tensor([self.batch_size, self.audio_num_frames, self.n_mel_bins], self.mel_vocab_size)
        audio_input_ids_mask = torch.ones(self.batch_size, self.audio_num_frames, dtype=torch.bool)
        config = self.get_config()
        return config, audio_input_ids, audio_input_ids_mask

    def prepare_config_and_inputs_for_common(self):
        config, audio_input_ids, audio_input_ids_mask = self.prepare_config_and_inputs()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(self.pad_token_id).to(torch_device)

        # Ensure no tokens accidentally match special token IDs
        for token_id in [config.image_token_id, config.video_token_id, config.audio_token_id]:
            input_ids[input_ids == token_id] = self.pad_token_id

        # One audio embedding is produced per valid frame; place that many audio placeholders per sequence
        input_ids[:, : self.audio_num_frames] = config.audio_token_id

        inputs_dict = {
            "audio_input_ids": audio_input_ids,
            "audio_input_ids_mask": audio_input_ids_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class InklingAudio2TextModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (InklingModel, InklingForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (InklingForConditionalGeneration,) if is_torch_available() else ()
    test_all_params_have_gradient = False  # e-score correction bias is only used for expert routing
    # Audio embeddings are packed per valid frame, so last_hidden_state[0] is the total frame count, not batch size
    skip_test_audio_features_output_shape = True

    def setUp(self):
        self.model_tester = InklingAudio2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=InklingConfig, hidden_size=37)

    @unittest.skip(
        "Inkling chains tower namespace and internal renames, so intermediate source keys are absent after reverse mapping"
    )
    def test_reverse_loading_mapping(self):
        pass

    @unittest.skip("Inkling's audio tower is an embedding+norm module with no attention or hidden-state layers")
    def test_get_audio_features_hidden_states(self):
        pass

    @unittest.skip("Inkling's audio tower is an embedding+norm module with no attention or hidden-state layers")
    def test_get_audio_features_attentions(self):
        pass

    @unittest.skip("The tester has no image in input dict")
    def test_get_image_features_hidden_states(self):
        pass

    @unittest.skip("The tester has no image in input dict")
    def test_get_image_features_attentions(self):
        pass

    @parameterized.expand([True, False, None])
    @unittest.skip("The tester has no image in input dict")
    def test_get_image_features_output(self, return_dict: bool | None):
        pass

    @unittest.skip("The tester has no videos in input dict")
    def test_get_video_features_hidden_states(self):
        pass

    @unittest.skip("The tester has no videos in input dict")
    def test_get_video_features_attentions(self):
        pass

    @parameterized.expand([True, False, None])
    @unittest.skip("The tester has no videos in input dict")
    def test_get_video_features_output(self, return_dict: bool | None):
        pass

    @unittest.skip("We need 4 layers to correctly test cache sharing.")
    def test_num_layers_is_small(self):
        pass

    @unittest.skip("Inkling needs correct embeddings for per-layer-input computation, random won't work!")
    def test_generate_from_random_inputs_embeds(self):
        pass

    @unittest.skip("Inkling requires an explicit prompt for generation")
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip("Audio placeholder embeddings are replaced when audio inputs are provided")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip("Accelerate does not create a device map when the entire tiny model fits on CPU")
    def test_cpu_offload(self):
        pass

    @unittest.skip("Accelerate maps the entire tiny model to disk instead of producing a split device map")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip("Accelerate maps the entire tiny model to disk instead of producing a split device map")
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip(GEMMA4_RANDOM_MOE_FA2_SKIP_REASON)
    def test_flash_attn_2_inference_equivalence(self):
        pass

    @unittest.skip(GEMMA4_RANDOM_MOE_FA2_SKIP_REASON)
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass


class InklingVision2TextModelTester:
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
    ):
        self.parent = parent
        # `image_token_id` is set to 0 to pass "resize_embeddings" test, do not modify
        self.mm_tokens_per_image = mm_tokens_per_image
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.llm_tester = InklingTextModelTester(self.parent)
        self.text_config = self.llm_tester.get_config()
        self.vision_config = vision_config
        self.seq_length = seq_length
        self.pad_token_id = self.text_config.pad_token_id

        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size
        self.num_attention_heads = self.text_config.num_attention_heads
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = vision_config["num_channels"]
        self.image_size = vision_config["image_size"]
        self.encoder_seq_length = seq_length

    def get_config(self):
        config = InklingConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            audio_config={"hidden_size": self.text_config.hidden_size, "n_mel_bins": 4, "mel_vocab_size": 8},
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            audio_token_id=self.audio_token_id,
            boi_token_id=self.boi_token_id,
            eoi_token_id=self.eoi_token_id,
            mm_tokens_per_image=self.mm_tokens_per_image,
        )
        config.num_hidden_layers = config.text_config.num_hidden_layers
        return config

    def prepare_config_and_inputs(self):
        config = self.get_config()
        config.vision_config.pooling_kernel_size = 2

        # One packed patch per image placeholder: (num_patches, time, height, width, channels)
        patch_size = config.vision_config.patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size,
                config.vision_config.temporal_patch_size,
                patch_size,
                patch_size,
                self.vision_config["num_channels"],
            ]
        )
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(self.pad_token_id).to(torch_device)

        # Ensure no tokens accidentally match special token IDs
        for token_id in [config.image_token_id, config.video_token_id, config.audio_token_id]:
            input_ids[input_ids == token_id] = self.pad_token_id
        input_ids[:, :1] = config.image_token_id

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class InklingVision2TextModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (InklingModel, InklingForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (InklingForConditionalGeneration,) if is_torch_available() else ()
    test_all_params_have_gradient = False  # e-score correction bias is only used for expert routing
    model_split_percents = [0.85, 0.9]

    def setUp(self):
        self.model_tester = InklingVision2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=InklingConfig, hidden_size=37)

    @unittest.skip(
        "Inkling chains tower namespace and internal renames, so intermediate source keys are absent after reverse mapping"
    )
    def test_reverse_loading_mapping(self):
        pass

    def test_training(self):
        # Overwrite to test training with text-only samples, should not raise errors
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        model = InklingForConditionalGeneration(config)
        model.to(torch_device)
        model.train()
        inputs = self._prepare_for_class(inputs_dict, InklingForConditionalGeneration, return_labels=True)
        loss = model(**inputs).loss
        loss.backward()

        # pop out image-related inputs and try to run forward
        inputs.pop("pixel_values", None)
        loss = model(**inputs).loss
        loss.backward()

    @unittest.skip("The tester has no audios in input dict")
    def test_get_audio_features_hidden_states(self):
        pass

    @unittest.skip("The tester has no audios in input dict")
    def test_get_audio_features_attentions(self):
        pass

    @parameterized.expand([True, False, None])
    @unittest.skip("The tester has no audios in input dict")
    def test_get_audio_features_output(self, return_dict: bool | None):
        pass

    @unittest.skip("Inkling's HMLP vision tower has no attention or hidden-state outputs")
    def test_get_image_features_hidden_states(self):
        pass

    @unittest.skip("Inkling's HMLP vision tower has no attention or hidden-state outputs")
    def test_get_image_features_attentions(self):
        pass

    @unittest.skip("The tester has no videos in input dict")
    def test_get_video_features_hidden_states(self):
        pass

    @unittest.skip("The tester has no videos in input dict")
    def test_get_video_features_attentions(self):
        pass

    @parameterized.expand([True, False, None])
    @unittest.skip("The tester has no videos in input dict")
    def test_get_video_features_output(self, return_dict: bool | None):
        pass

    @unittest.skip("We need 4 layers to correctly test cache sharing.")
    def test_num_layers_is_small(self):
        pass

    @unittest.skip("Inkling needs correct embeddings for per-layer-input computation, random won't work!")
    def test_generate_from_random_inputs_embeds(self):
        pass

    @unittest.skip("Inkling requires an explicit prompt for generation")
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip("Image placeholder embeddings are replaced when pixel values are provided")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(
        "Randomly starts failing after module order changed in the __init__ because accelertate is not robust enough"
    )
    def test_cpu_offload(self):
        pass

    @unittest.skip(
        "Randomly starts failing after module order changed in the __init__ because accelertate is not robust enough"
    )
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(
        "Randomly starts failing after module order changed in the __init__ because accelertate is not robust enough"
    )
    def test_disk_offload_safetensors(self):
        pass


@slow
@require_torch_accelerator
class InklingIntegrationTest(unittest.TestCase):
    """Validate the Inkling next-token distribution against sglang, across every input modality.

    reproducer (single sglang Engine, all cases, uploads the golden to
    ``hf://buckets/hf-internal-testing/tml-integration-tests/<case>/expected_next_token_logprobs.safetensors``):
        ~/tml/reproducers/reproducer_logits.py
    gist: https://gist.github.com/eustlb/cb2a5df1676911fa0eb07d0a76a38ae7
    """

    IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
    IMAGE_URL_2 = "http://images.cocodataset.org/val2017/000000000139.jpg"
    AUDIO_URL = (
        "https://huggingface.co/datasets/adarshxs/voxcpm2-native-generated-audio-user-ref/resolve/main/zs_medium.wav"
    )
    AUDIO_URL_2 = (
        "https://huggingface.co/datasets/adarshxs/voxcpm2-native-generated-audio-user-ref/resolve/main/zs_short.wav"
    )

    @classmethod
    def setUpClass(cls):
        cls.checkpoint_name = "hf-internal-testing/tiny-inkling"
        cls.bucket = "hf-internal-testing/inkling-integration-test"
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint_name)
        cls.model = InklingForConditionalGeneration.from_pretrained(cls.checkpoint_name, device_map=torch_device)

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cleanup(torch_device, gc_collect=True)

    def _load_expected_logprobs(self, case: str):
        remote = f"{case}/expected_next_token_logprobs.safetensors"
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "expected_next_token_logprobs.safetensors")
            download_bucket_files(self.bucket, files=[(remote, local)])
            return load_file(local)["next_token_logprobs"]

    def _assert_next_token_logprobs(self, case: str, messages: list):
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.model.dtype)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1].float().cpu()
        logprobs = torch.log_softmax(logits, dim=-1)

        expected_logprobs = self._load_expected_logprobs(case)

        self.assertEqual(tuple(logprobs.shape), tuple(expected_logprobs.shape))
        torch.testing.assert_close(logprobs.exp(), expected_logprobs.exp(), rtol=1e-3, atol=1e-4)

    def test_text_next_token_logprobs(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}]
        self._assert_next_token_logprobs("text", messages)

    def test_image_next_token_logprobs(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is shown in this image?"},
                    {"type": "image", "url": self.IMAGE_URL},
                ],
            }
        ]
        self._assert_next_token_logprobs("image", messages)

    def test_audio_next_token_logprobs(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is said in this clip?"},
                    {"type": "audio", "url": self.AUDIO_URL},
                ],
            }
        ]
        self._assert_next_token_logprobs("audio", messages)

    def test_image_audio_next_token_logprobs(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image and tell me what is said in the clip."},
                    {"type": "image", "url": self.IMAGE_URL},
                    {"type": "audio", "url": self.AUDIO_URL},
                ],
            }
        ]
        self._assert_next_token_logprobs("image_audio", messages)

    def test_multi_image_next_token_logprobs(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these two images."},
                    {"type": "image", "url": self.IMAGE_URL},
                    {"type": "image", "url": self.IMAGE_URL_2},
                ],
            }
        ]
        self._assert_next_token_logprobs("multi_image", messages)

    def test_multi_audio_next_token_logprobs(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is said in these two clips?"},
                    {"type": "audio", "url": self.AUDIO_URL},
                    {"type": "audio", "url": self.AUDIO_URL_2},
                ],
            }
        ]
        self._assert_next_token_logprobs("multi_audio", messages)
