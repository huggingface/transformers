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
    InklingAudioConfig,
    InklingConfig,
    InklingTextConfig,
    InklingVisionConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...alm_tester import ALMModelTest, ALMModelTester
from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import floats_tensor, ids_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import InklingForCausalLM, InklingForConditionalGeneration, InklingModel, InklingTextModel


class InklingTextModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = InklingTextConfig
        base_model_class = InklingTextModel
        causal_lm_class = InklingForCausalLM

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
        self.moe_intermediate_size = 16
        self.n_routed_experts = 16


class InklingTextModelTests(CausalLMModelTest, unittest.TestCase):
    model_tester_class = InklingTextModelTester
    _torch_compile_train_cls = InklingForCausalLM if is_torch_available() else None
    model_split_percents = [0.5, 0.8, 0.9]

    @unittest.skip("MoE routing on a tiny randomly-initialized model makes the overfit target unstable.")
    def test_training_overfit(self):
        pass


class InklingAudio2TextModelTester(ALMModelTester):
    base_model_class = InklingModel
    conditional_generation_class = InklingForConditionalGeneration
    config_class = InklingConfig
    text_config_class = InklingTextConfig
    audio_config_class = InklingAudioConfig
    audio_mask_key = "audio_input_ids_mask"

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 4)
        kwargs.setdefault("audio_token_id", 7)
        kwargs.setdefault("pad_token_id", 0)
        kwargs.setdefault("seq_length", 50)
        kwargs.setdefault("feat_seq_length", 4)
        kwargs.setdefault("n_mel_bins", 4)
        kwargs.setdefault("mel_vocab_size", 8)
        kwargs.setdefault("layer_types", ["hybrid_sliding", "hybrid"])
        kwargs.setdefault("mlp_layer_types", ["dense", "sparse"])
        kwargs.setdefault("moe_intermediate_size", 16)
        kwargs.setdefault("n_routed_experts", 16)
        kwargs.setdefault("head_dim", 16)
        kwargs.setdefault("swa_num_attention_heads", 2)
        kwargs.setdefault("swa_num_key_value_heads", 2)
        kwargs.setdefault("swa_head_dim", 16)
        super().__init__(parent, **kwargs)

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {self.image_token_id}

    def create_attention_mask(self, input_ids):
        return input_ids.ne(self.pad_token_id).to(torch_device)

    def create_audio_features(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        # Quantized mel frames: (num_audios, num_frames, n_mel_bins)
        return ids_tensor([batch_size, self.feat_seq_length, self.n_mel_bins], self.mel_vocab_size)

    def get_audio_embeds_mask(self, audio_mask):
        return audio_mask

    def get_audio_feature_key(self):
        return "audio_input_ids"

    def _build_modality_sub_configs(self):
        return {
            "audio_config": self.get_audio_config(),
            "vision_config": InklingVisionConfig(patch_size=5, num_hidden_layers=2, num_channels=3),
        }


@require_torch
class InklingAudio2TextModelTest(ALMModelTest, unittest.TestCase):
    model_tester_class = InklingAudio2TextModelTester
    test_all_params_have_gradient = False  # e-score correction bias is only used for expert routing
    # Audio embeddings are packed per valid frame, so last_hidden_state[0] is the total frame count, not batch size
    skip_test_audio_features_output_shape = True

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

    @unittest.skip("Accelerate does not create a device map when the entire tiny model fits on CPU")
    def test_cpu_offload(self):
        pass

    @unittest.skip("Accelerate maps the entire tiny model to disk instead of producing a split device map")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip("Accelerate maps the entire tiny model to disk instead of producing a split device map")
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip("Randomly initialized Inkling MoE routers are too sensitive to tiny eager/FA2 input differences")
    def test_flash_attn_2_inference_equivalence(self):
        pass

    @unittest.skip("Randomly initialized Inkling MoE routers are too sensitive to tiny eager/FA2 input differences")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @unittest.skip(
        reason="Inkling attention always adds a relative position bias, which requires a float additive mask that is incompatible with the SDPA flash backend"
    )
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(
        reason="The audio tower and embeddings are non-splittable and hold almost all of the weights, so device_map='auto' can't split the model across GPUs"
    )
    def test_model_parallelism(self):
        pass


class InklingVision2TextModelTester(VLMModelTester):
    base_model_class = InklingModel
    conditional_generation_class = InklingForConditionalGeneration
    config_class = InklingConfig
    text_config_class = InklingTextConfig
    vision_config_class = InklingVisionConfig

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 4)
        kwargs.setdefault("audio_token_id", 8)
        kwargs.setdefault("seq_length", 25)
        kwargs.setdefault("num_image_tokens", 1)
        kwargs.setdefault("patch_size", 5)
        kwargs.setdefault("temporal_patch_size", 2)
        kwargs.setdefault("layer_types", ["hybrid_sliding", "hybrid"])
        kwargs.setdefault("mlp_layer_types", ["dense", "sparse"])
        kwargs.setdefault("moe_intermediate_size", 16)
        kwargs.setdefault("n_routed_experts", 16)
        kwargs.setdefault("swa_num_attention_heads", 2)
        kwargs.setdefault("swa_num_key_value_heads", 2)
        kwargs.setdefault("swa_head_dim", 16)
        super().__init__(parent, **kwargs)

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {self.audio_token_id}

    def create_attention_mask(self, input_ids):
        return input_ids.ne(self.pad_token_id).to(torch_device)

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        # One packed patch per image placeholder: (num_patches, time, height, width, channels)
        return floats_tensor(
            [batch_size, self.temporal_patch_size, self.patch_size, self.patch_size, self.num_channels]
        )

    def _build_modality_sub_configs(self):
        return {
            "vision_config": self.get_vision_config(),
            "audio_config": InklingAudioConfig(n_mel_bins=4, mel_vocab_size=8),
        }


@require_torch
class InklingVision2TextModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = InklingVision2TextModelTester
    test_all_params_have_gradient = False  # e-score correction bias is only used for expert routing
    test_torch_exportable = False  # data-dependent control flow in the HMLP vision tower (time/space folding)
    model_split_percents = [0.85, 0.9]

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

    @unittest.skip(
        reason="Inkling attention always adds a relative position bias, which requires a float additive mask that is incompatible with the SDPA flash backend"
    )
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(
        reason="The vision tower and embeddings are non-splittable and hold almost all of the weights, so device_map='auto' can't split the model across GPUs"
    )
    def test_model_parallelism(self):
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

    IMAGE_URL = (
        "https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/000000039769.jpg"
    )
    IMAGE_URL_2 = (
        "https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/000000000139.jpg"
    )
    AUDIO_URL = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/zs_medium.wav"
    AUDIO_URL_2 = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/zs_short.wav"

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
