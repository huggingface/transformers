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

import pytest
import requests
from parameterized import parameterized

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
    Phi4MultimodalAudioConfig,
    Phi4MultimodalConfig,
    Phi4MultimodalForCausalLM,
    Phi4MultimodalModel,
    Phi4MultimodalVisionConfig,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    require_torch_large_accelerator,
    require_torchcodec,
    slow,
    torch_device,
)
from transformers.utils import is_torchcodec_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image


if is_torchcodec_available():
    import torchcodec


class Phi4MultimodalModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=12,
        image_seq_length=275,
        audio_seq_length=8,
        is_training=True,
        num_hidden_layers=2,
        vocab_size=49,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=8,
        num_key_value_heads=4,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        image_token_id=1,
        audio_token_id=2,
        image_size=16,
        audio_size=12,
        audio_config=Phi4MultimodalAudioConfig(
            num_blocks=2,
            hidden_size=32,
            num_attention_heads=8,
            intermediate_size=48,
            depthwise_seperable_out_channel=128,
            nemo_conv_channels=128,
            initializer_range=1e-5,
        ),
        vision_config=Phi4MultimodalVisionConfig(
            num_hidden_layers=2,
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=8,
            crop_size=16,
            initializer_range=1e-5,
        ),
    ):
        self.parent = parent
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id
        self.audio_config = audio_config
        self.vision_config = vision_config

        self.is_training = is_training
        self.batch_size = batch_size
        self.seq_length = seq_length + image_seq_length + audio_seq_length
        self.image_seq_length = image_seq_length
        self.audio_seq_length = audio_seq_length
        self.image_size = image_size
        self.audio_size = audio_size
        self.num_channels = 3

    def get_config(self):
        return Phi4MultimodalConfig(
            num_hidden_layers=self.num_hidden_layers,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            vision_config=self.vision_config,
            audio_config=self.audio_config,
        )

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        # The shapes corresponds to the inputs for image of size 16x16
        image_pixel_values = floats_tensor([self.batch_size, 2, self.num_channels, self.image_size, self.image_size])
        image_attention_mask = torch.ones(self.batch_size, 2, 1, 1)
        image_sizes = torch.tensor(
            [[self.image_size, self.image_size]] * self.batch_size, dtype=torch.long, device=torch_device
        )

        # Feature sizes returned by an audio of size 10000
        audio_input_features = floats_tensor([self.batch_size, 61, 80])
        audio_embed_sizes = torch.tensor([self.audio_seq_length] * self.batch_size, dtype=torch.long)

        input_ids[input_ids == self.pad_token_id] = self.pad_token_id + 1  # random value but not pad token
        input_ids[-1, 0] = self.pad_token_id  # mask the last text token
        input_ids[:, -self.image_seq_length - self.audio_seq_length : -self.audio_seq_length] = self.image_token_id
        input_ids[:, -self.audio_seq_length :] = self.audio_token_id

        attention_mask = torch.ones_like(input_ids)
        attention_mask[-1, 0] = 0  # mask the last text token
        config = self.get_config()

        return (
            config,
            input_ids,
            attention_mask,
            image_pixel_values,
            image_attention_mask,
            image_sizes,
            audio_input_features,
            audio_embed_sizes,
        )

    def prepare_config_and_inputs_for_common(self):
        (
            config,
            input_ids,
            attention_mask,
            image_pixel_values,
            image_attention_mask,
            image_sizes,
            audio_input_features,
            audio_embed_sizes,
        ) = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_pixel_values": image_pixel_values,
            "image_attention_mask": image_attention_mask,
            "image_sizes": image_sizes,
            "audio_input_features": audio_input_features,
            "audio_embed_sizes": audio_embed_sizes,
        }
        return config, inputs_dict


@require_torch
class Phi4MultimodalModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `Phi4Multimodal`.
    """

    all_model_classes = (Phi4MultimodalForCausalLM, Phi4MultimodalModel) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = Phi4MultimodalModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Phi4MultimodalConfig)

    @unittest.skip(reason="Unstable test")
    def test_initialization(self):
        pass

    @unittest.skip(reason="Depending on input modalities, some params may not have gradients")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Depending on input modalities, some params may not have gradients")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="Depending on input modalities, some params may not have gradients")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Test tries to instantiate dynamic cache with an arg")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip(reason="Test is only for old attention format")
    def test_sdpa_can_dispatch_composite_models(self):
        pass

    @unittest.skip(reason="Static cache supported only for text-only inputs (not images or audios)")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip(reason="Static cache supported only for text-only inputs (not images or audios)")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip(
        reason="Supported only for text-only inputs (otherwise dynamic control flows for multimodal inputs)"
    )
    def test_generate_compilation_all_outputs(self):
        pass

    @unittest.skip(
        reason="Supported only for text-only inputs (otherwise dynamic control flows for multimodal inputs)"
    )
    @pytest.mark.torch_compile_test
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip(reason="`image_attention_mask` has a specific shape")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip(reason="`image_attention_mask` has a specific shape")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip(reason="`image_attention_mask` has a specific shape")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @unittest.skip(reason="Cannot unpad inputs for all modalities so easily")
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(reason="Dynamo error")
    def test_flex_attention_with_grads(self):
        pass


@require_torch
@slow
class Phi4MultimodalIntegrationTest(unittest.TestCase):
    checkpoint_path = "microsoft/Phi-4-multimodal-instruct"
    revision = "refs/pr/70"
    image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"

    def setUp(self):
        # Currently, the Phi-4 checkpoint on the hub is not working with the latest Phi-4 code, so the slow integration tests
        # won't pass without using the correct revision (refs/pr/70)
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_path, revision=self.revision)
        self.generation_config = GenerationConfig(max_new_tokens=20, do_sample=False)
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"
        self.end_token = "<|end|>"
        self.image = Image.open(requests.get(self.image_url, stream=True).raw)
        audio_bytes = requests.get(self.audio_url, stream=True).raw.data
        samples = torchcodec.decoders.AudioDecoder(audio_bytes).get_all_samples()
        self.audio, self.sampling_rate = samples.data, samples.sample_rate

        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_text_only_generation(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path, revision=self.revision, dtype=torch.float16, device_map=torch_device
        )

        prompt = f"{self.user_token}What is the answer for 1+1? Explain it.{self.end_token}{self.assistant_token}"
        inputs = self.processor(prompt, images=None, return_tensors="pt").to(torch_device)

        output = model.generate(
            **inputs,
            generation_config=self.generation_config,
        )
        output = output[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        EXPECTED_RESPONSE = "The answer for 1+1 is 2. This is because when you add one to another"

        self.assertEqual(response, EXPECTED_RESPONSE)

    def test_vision_text_generation(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path, revision=self.revision, dtype=torch.float16, device_map=torch_device
        )

        prompt = f"{self.user_token}<|image|>What is shown in this image?{self.end_token}{self.assistant_token}"
        inputs = self.processor(prompt, images=self.image, return_tensors="pt").to(torch_device)

        output = model.generate(
            **inputs,
            generation_config=self.generation_config,
        )
        output = output[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        EXPECTED_RESPONSES = Expectations(
            {
                ("cuda", 7): 'The image shows a vibrant scene at a traditional Chinese-style street entrance, known as a "gate"',
                ("cuda", 8): 'The image shows a vibrant scene at a street intersection in a city with a Chinese-influenced architectural',
            }
        )  # fmt: skip
        EXPECTED_RESPONSE = EXPECTED_RESPONSES.get_expectation()

        self.assertEqual(response, EXPECTED_RESPONSE)

    @require_torch_large_accelerator
    def test_multi_image_vision_text_generation(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path, revision=self.revision, dtype=torch.float16, device_map=torch_device
        )

        images = []
        placeholder = ""
        for i in range(1, 5):
            url = f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg"
            images.append(Image.open(requests.get(url, stream=True).raw))
            placeholder += "<|image|>"

        prompt = f"{self.user_token}{placeholder}Summarize the deck of slides.{self.end_token}{self.assistant_token}"
        inputs = self.processor(prompt, images, return_tensors="pt").to(torch_device)

        output = model.generate(
            **inputs,
            generation_config=self.generation_config,
        )
        output = output[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        EXPECTED_RESPONSE = "The presentation provides an overview of Microsoft Azure, a cloud computing platform by Microsoft, and its various services"

        self.assertEqual(response, EXPECTED_RESPONSE)

    @require_torchcodec
    def test_audio_text_generation(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path, revision=self.revision, dtype=torch.float16, device_map=torch_device
        )

        prompt = f"{self.user_token}<|audio|>What is happening in this audio?{self.end_token}{self.assistant_token}"
        inputs = self.processor(prompt, audio=self.audio, sampling_rate=self.sampling_rate, return_tensors="pt").to(
            torch_device
        )

        output = model.generate(
            **inputs,
            generation_config=self.generation_config,
        )
        output = output[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Yes, it is truly the expected response... Even though the model correctly treats the audio file
        EXPECTED_RESPONSE = "I'm sorry, but I can't listen to audio. However, if you describe the audio to me,"

        self.assertEqual(response, EXPECTED_RESPONSE)
