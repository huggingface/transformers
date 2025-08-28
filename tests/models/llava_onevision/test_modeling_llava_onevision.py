# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Llava-NeXT model."""

import unittest

import numpy as np
import requests
from huggingface_hub import hf_hub_download
from parameterized import parameterized

from transformers import (
    AutoProcessor,
    LlavaOnevisionConfig,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionModel,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_bitsandbytes,
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image


class LlavaOnevisionVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_index=1,
        video_token_index=2,
        projector_hidden_act="gelu",
        seq_length=7,
        vision_feature_select_strategy="full",
        vision_feature_layer=-1,
        text_config={
            "model_type": "qwen2",
            "seq_length": 7,
            "is_training": True,
            "use_input_mask": True,
            "use_token_type_ids": False,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 37,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 580,
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 0,
        },
        is_training=True,
        vision_config={
            "image_size": 16,
            "patch_size": 8,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "projection_dim": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.text_config = text_config
        self.vision_config = vision_config
        self.pad_token_id = text_config["pad_token_id"]
        self.num_image_tokens = 10
        self.seq_length = seq_length + self.num_image_tokens

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = 3
        self.image_size = 30
        self.image_grid_pinpoints = [[16, 16]]

    def get_config(self):
        return LlavaOnevisionConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            ignore_index=self.ignore_index,
            image_token_index=self.image_token_index,
            video_token_index=self.video_token_index,
            projector_hidden_act=self.projector_hidden_act,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            vision_feature_layer=self.vision_feature_layer,
            image_grid_pinpoints=self.image_grid_pinpoints,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                3,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 2) + 2
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)

        input_ids[input_ids == config.image_token_index] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = config.image_token_index

        labels = torch.zeros((self.batch_size, self.seq_length), dtype=torch.long, device=torch_device)
        labels[:, : self.num_image_tokens] == self.ignore_index

        inputs_dict = {
            "pixel_values": pixel_values,
            "image_sizes": torch.tensor([[45, 45]] * self.batch_size),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return config, inputs_dict


@require_torch
class LlavaOnevisionForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `LlavaOnevisionForConditionalGeneration`.
    """

    all_model_classes = (
        (
            LlavaOnevisionModel,
            LlavaOnevisionForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {"image-text-to-text": LlavaOnevisionForConditionalGeneration} if is_torch_available() else {}
    )
    test_pruning = False
    test_head_masking = False
    # MP works but offload doesn't work when the MultiheadAttention is offloaded
    # TODO: One potential solution would be to add to set preload_module_classes = ["Siglip2MultiheadAttentionPoolingHead"]
    # in the dispatch_model function
    test_cpu_offload = False
    test_disk_offload_safetensors = False
    test_disk_offload_bin = False
    _is_composite = True

    def setUp(self):
        self.model_tester = LlavaOnevisionVisionText2TextModelTester(self)
        common_properties = ["image_token_index", "video_token_index", "vision_feature_layer"]
        self.config_tester = ConfigTester(
            self, config_class=LlavaOnevisionConfig, has_text_modality=False, common_properties=common_properties
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                # LLaVa Onevision has SigLIP backbone which init weights differently from CLIP
                if "image_newline" in name or "vision_tower" in name:
                    continue
                elif param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    def test_odd_sized_image(self):
        # prepare model configuration
        config = self.model_tester.get_config()

        # prepare input
        num_image_tokens = 10
        pixel_values = floats_tensor([1, 2, 3, config.vision_config.image_size, config.vision_config.image_size])
        input_ids = ids_tensor([1, 64], config.text_config.vocab_size - 2) + 2
        input_ids[:, :num_image_tokens] = config.image_token_index
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)
        inputs_dict = {
            "pixel_values": pixel_values,
            "image_sizes": torch.tensor([[13, 16]]),  # odd-sized image
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # forward with odd-sized image input
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model(**inputs_dict)

    @parameterized.expand(
        [
            (-1,),
            ([-1],),
            ([-1, -2],),
        ],
    )
    def test_vision_feature_layers(self, vision_feature_layer):
        """
        Test that we can use either one vision feature layer, or a list of
        vision feature layers.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.vision_feature_layer = vision_feature_layer

        num_feature_layers = 1 if isinstance(vision_feature_layer, int) else len(vision_feature_layer)
        hidden_size = config.vision_config.hidden_size
        expected_features = hidden_size * num_feature_layers

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            # We should have the right number of input features,
            # and should be able to run a forward pass without exploding
            base_model = getattr(model, "model", model)
            assert base_model.multi_modal_projector.linear_1.in_features == expected_features
            model(**input_dict)

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, SiglipVisionModel does not support standalone training"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, SiglipVisionModel does not support standalone training"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, SiglipVisionModel does not support standalone training"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(
        "VLMs need lots of steps to prepare images/mask correctly to get pad-free inputs. Can be tested as part of LLM test"
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass


@require_torch
class LlavaOnevisionForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", padding_side="left"
        )
        image_file = hf_hub_download(
            repo_id="raushan-testing-hf/images_test", filename="llava_v1_5_radar.jpg", repo_type="dataset"
        )
        video_file = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="video_demo.npy", repo_type="dataset"
        )
        self.image = Image.open(image_file)
        self.video = np.load(video_file)
        self.prompt_image = "user\n<image>\nWhat do you see in this image?<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_video = "user\n<video>\nWhat do you see in this video?<|im_end|>\n<|im_start|>assistant\n"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test(self):
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", dtype="float16", device_map=torch_device
        )

        inputs = self.processor(images=self.image, text=self.prompt_image, return_tensors="pt").to(
            torch_device, torch.float16
        )
        self.assertTrue(inputs.input_ids.shape[1] == 6567)  # should expand num-image-tokens times
        self.assertTrue(inputs.pixel_values.shape == torch.Size([1, 10, 3, 384, 384]))
        self.assertTrue(inputs.image_sizes.tolist() == [[899, 1024]])

        # verify single forward pass
        inputs = inputs.to(torch_device)

        # verify generation
        output = model.generate(**inputs, max_new_tokens=100)

        EXPECTED_DECODED_TEXTS = Expectations(
            {
                ("xpu", 3): 'user\n\nWhat do you see in this image?\nassistant\nThe image is a radar chart that compares the performance of different models in a specific task, likely related to natural language processing or machine learning. The chart is divided into several axes, each representing a different model or method. The models are color-coded and labeled with their respective names. The axes are labeled with terms such as "VQA," "GQA," "MQA," "VQAv2," "MM-Vet," "LLaVA-Bench," "LLaVA-1',
                ("cuda", 7): 'user\n\nWhat do you see in this image?\nassistant\nThe image is a radar chart that compares the performance of different models in a specific task, likely related to natural language processing or machine learning. The chart is divided into several axes, each representing a different model or method. The models are color-coded and labeled with their respective names. The axes are labeled with terms such as "VQA," "GQA," "MQA," "VQAv2," "MM-Vet," "LLaVA-Bench," "LLaVA-1',
                ("cuda", 8): 'user\n\nWhat do you see in this image?\nassistant\nThe image is a radar chart that compares the performance of different models in a specific task, likely related to natural language processing or machine learning. The chart is divided into several axes, each representing a different model or method. The models are color-coded and labeled with their respective names. The axes are labeled with terms such as "VQA," "GQA," "MQA," "VIZ," "TextVQA," "SQA-IMG," and "MQE." The radar chart shows',
            }
        )  # fmt: skip
        EXPECTED_DECODED_TEXT = EXPECTED_DECODED_TEXTS.get_expectation()
        DECODED_TEXT = self.processor.decode(output[0], skip_special_tokens=True)

        self.assertEqual(DECODED_TEXT, EXPECTED_DECODED_TEXT)

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_batch(self):
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", dtype="float16", device_map=torch_device
        )

        inputs = self.processor(
            text=[self.prompt_image, self.prompt_video],
            images=self.image,
            videos=self.video,
            return_tensors="pt",
            padding=True,
        ).to(torch_device, torch.float16)

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = ['user\n\nWhat do you see in this image?\nassistant\nThe image is a radar chart that compares the performance of different models in a specific task, likely related', 'user\n\nWhat do you see in this video?\nassistant\nA child wearing a light blue sleeveless top and pink pants is seen sitting on a bed, eng']  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_video(self):
        # related to (#29835)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            dtype="float16",
            device_map=torch_device,
        )

        inputs = self.processor(text=self.prompt_video, videos=self.video, return_tensors="pt").to(
            torch_device, torch.float16
        )

        # verify generation
        output = model.generate(**inputs, max_new_tokens=40)
        EXPECTED_DECODED_TEXT = 'user\n\nWhat do you see in this video?\nassistant\nA child wearing a light blue sleeveless top and pink pants is seen sitting on a bed, engrossed in reading a book.'  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_multi_image(self):
        # related to (#29835)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            dtype="float16",
            device_map=torch_device,
        )

        url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        prompt = (
            "user\n<image><image>\nWhat is the difference between these images?<|im_end|>\n<|im_start|>assistant\n"
        )
        inputs = self.processor(text=prompt, images=[self.image, image], return_tensors="pt").to(
            torch_device, torch.float16
        )

        # verify generation
        output = model.generate(**inputs, max_new_tokens=40)
        EXPECTED_DECODED_TEXT = "user\n\nWhat is the difference between these images?\nassistant\nThe images you've provided appear to be related to a graphical representation of a radar chart, which is a type of data visualization used to show the distribution of a particular variable across a geographic area. The"  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_multi_image_nested(self):
        # related to (#34585)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            dtype="float16",
            device_map=torch_device,
        )

        url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        prompts = [
            "user\nTell me about the french revolution.<|im_end|>\n<|im_start|>assistant\n",  # text-only case
            "user\n<image><image>\nWhat is the difference between these images?<|im_end|>\n<|im_start|>assistant\n",
            self.prompt_image,
        ]
        images_nested = [[], [image, self.image], [self.image]]
        inputs = self.processor(
            text=prompts,
            images=images_nested,
            return_tensors="pt",
            padding=True,
        ).to(torch_device, torch.float16)

        # verify generation
        output = model.generate(**inputs, max_new_tokens=40)
        EXPECTED_DECODED_TEXT = ["user\nTell me about the french revolution.\nassistant\nThe French Revolution! A pivotal event in modern history that had a profound impact on the course of Western civilization. Here's a brief overview:\n\n**Background**\n\nIn the late 18th century,", "user\n\nWhat is the difference between these images?\nassistant\nThe first image shows a stop sign with a traditional Chinese architectural background, while the second image displays a radar chart with various algorithms and models, including BLIP-2, InstructBLIP, Q", "user\n\nWhat do you see in this image?\nassistant\nThe image is a radar chart that compares the performance of different models in a specific task, likely related to natural language processing or machine learning. The chart is divided into several axes, each representing a different"]  # fmt: skip
        DECODED_TEXT = self.processor.batch_decode(output, skip_special_tokens=True)

        self.assertListEqual(DECODED_TEXT, EXPECTED_DECODED_TEXT)

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_multi_video(self):
        # related to (#29835)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            dtype="float16",
            device_map=torch_device,
        )

        prompt = "user\n<video><video>\nAre these videos identical?<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.processor(text=prompt, videos=[self.video, self.video], return_tensors="pt").to(
            torch_device, torch.float16
        )

        # verify generation
        output = model.generate(**inputs, max_new_tokens=40)
        EXPECTED_DECODED_TEXT = "user\n\nAre these videos identical?\nassistant\nNo, the video is not identical; it shows slight variations in the child's actions and the background."  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_batch_different_resolutions(self):
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", dtype="float16", device_map=torch_device
        )

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        lowres_url = "https://4.img-dpreview.com/files/p/TS560x560~forums/56876524/03975b28741443319e9a94615e35667e"
        cats_image = Image.open(requests.get(url, stream=True).raw)
        lowres_img = Image.open(requests.get(lowres_url, stream=True).raw)

        inputs = self.processor(
            text=[self.prompt_image, self.prompt_image],
            images=[lowres_img, cats_image],
            return_tensors="pt",
            padding=True,
        ).to(torch_device, torch.float16)

        # verify generation
        output = model.generate(**inputs, max_new_tokens=50)
        EXPECTED_DECODED_TEXT = [
            'user\n\nWhat do you see in this image?\nassistant\nThe image shows a scene of two deer in a grassy area with trees in the background. The weather appears to be foggy, giving the scene a misty and somewhat mysterious atmosphere. The deer are standing close to each other, possibly grazing or',
            'user\n\nWhat do you see in this image?\nassistant\nIn the tranquil setting of this image, two cats are enjoying a peaceful nap on a vibrant pink blanket. The cat on the left, with its gray and black striped fur, is lying on its side, its head comfortably resting on the blanket. Its',
        ]  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_batch_matches_single(self):
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            dtype="float16",
            device_map=torch_device,
        )

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        lowres_url = "https://4.img-dpreview.com/files/p/TS560x560~forums/56876524/03975b28741443319e9a94615e35667e"
        cats_image = Image.open(requests.get(url, stream=True).raw)
        lowres_img = Image.open(requests.get(lowres_url, stream=True).raw)

        inputs_batched = self.processor(
            text=[self.prompt_image, self.prompt_image],
            images=[lowres_img, cats_image],
            return_tensors="pt",
            padding=True,
        ).to(torch_device, torch.float16)

        inputs_single = self.processor(
            text=self.prompt_image, images=lowres_img, return_tensors="pt", padding=True
        ).to(torch_device, torch.float16)

        # verify generation
        output_batched = model.generate(**inputs_batched, max_new_tokens=50)
        output_single = model.generate(**inputs_single, max_new_tokens=50)

        self.assertEqual(
            self.processor.decode(output_batched[0], skip_special_tokens=True),
            self.processor.decode(output_single[0], skip_special_tokens=True),
        )
