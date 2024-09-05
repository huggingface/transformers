# coding=utf-8
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
"""Testing suite for the PyTorch Llava-NeXT-Video model."""

import gc
import unittest

import numpy as np
from huggingface_hub import hf_hub_download

from transformers import (
    AutoProcessor,
    LlavaNextVideoConfig,
    LlavaNextVideoForConditionalGeneration,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
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

else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image


class LlavaNextVideoVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_index=0,
        video_token_index=1,
        projector_hidden_act="gelu",
        seq_length=7,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        text_config={
            "model_type": "llama",
            "seq_length": 7,
            "is_training": True,
            "use_input_mask": True,
            "use_token_type_ids": False,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
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
            "patch_size": 2,
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
        self.seq_length = seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = 3
        self.image_size = 30
        self.encoder_seq_length = 469
        self.image_grid_pinpoints = [[32, 32]]

    def get_config(self):
        return LlavaNextVideoConfig(
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
                5,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        pixel_values_videos = floats_tensor(
            [
                self.batch_size,
                8,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values, pixel_values_videos

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, pixel_values_videos = self.prepare_config_and_inputs()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 2) + 2
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)
        # we are giving 3 images and videos let's make sure we pass in 3 special tokens
        input_ids[:, 1] = config.image_token_index
        input_ids[:, 2] = config.video_token_index
        labels = torch.zeros((self.batch_size, self.seq_length), dtype=torch.long, device=torch_device)
        # maskout where the image/video token is
        labels[:, 1] == self.ignore_index
        labels[:, 2] == self.ignore_index
        inputs_dict = {
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_sizes": torch.tensor(
                [[self.vision_config["image_size"], self.vision_config["image_size"]]] * self.batch_size
            ),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return config, inputs_dict

    def create_and_check_llava_next_video_model_fp16_forward(
        self, config, input_ids, pixel_values, pixel_values_videos, attention_mask, image_sizes
    ):
        model = LlavaNextVideoForConditionalGeneration(config=config)
        model.to(torch_device)
        model.half()
        model.eval()
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_sizes=image_sizes,
            pixel_values=pixel_values.to(torch.bfloat16),
            pixel_values_videos=pixel_values_videos.to(torch.bfloat16),
            return_dict=True,
        )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())

    def create_and_check_llava_next_video_model_fp16_autocast_forward(
        self, config, input_ids, pixel_values, pixel_values_videos, attention_mask, image_sizes
    ):
        config.torch_dtype = torch.float16
        model = LlavaNextVideoForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_sizes=image_sizes,
                pixel_values=pixel_values.to(torch.bfloat16),
                pixel_values_videos=pixel_values_videos.to(torch.bfloat16),
                return_dict=True,
            )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())


@require_torch
class LlavaNextVideoForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `LlavaNextVideoForConditionalGeneration`.
    """

    all_model_classes = (LlavaNextVideoForConditionalGeneration,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = LlavaNextVideoVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LlavaNextVideoConfig, has_text_modality=False)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if "image_newline" in name:
                    continue
                elif param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    # overwrite inputs_embeds tests because we need to delete "pixel values" for LVLMs
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)

            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values"]
            del inputs["pixel_values_videos"]

            wte = model.get_input_embeddings()
            inputs["inputs_embeds"] = wte(input_ids)

            with torch.no_grad():
                model(**inputs)

    # overwrite inputs_embeds tests because we need to delete "pixel values" for LVLMs
    # while some other models require pixel_values to be present
    def test_inputs_embeds_matches_input_ids(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)
            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values"]
            del inputs["pixel_values_videos"]

            inputs_embeds = model.get_input_embeddings()(input_ids)

            with torch.no_grad():
                out_ids = model(input_ids=input_ids, **inputs)[0]
                out_embeds = model(inputs_embeds=inputs_embeds, **inputs)[0]
            self.assertTrue(torch.allclose(out_embeds, out_ids))

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Feedforward chunking is not yet supported")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="CPU offload is not yet supported")
    def test_cpu_offload(self):
        pass

    @unittest.skip(
        reason="Compile not yet supported because in LLava models (https://github.com/huggingface/transformers/issues/29891)"
    )
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(
        reason="Compile not yet supported because in LLava models (https://github.com/huggingface/transformers/issues/29891)"
    )
    def test_sdpa_can_dispatch_on_flash(self):
        pass


@require_torch
class LlavaNextVideoForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
        image_file = hf_hub_download(
            repo_id="raushan-testing-hf/images_test", filename="llava_v1_5_radar.jpg", repo_type="dataset"
        )
        video_file = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="video_demo.npy", repo_type="dataset"
        )
        self.image = Image.open(image_file)
        self.video = np.load(video_file)
        self.prompt_image = "USER: <image>\nWhat is shown in this image? ASSISTANT:"
        self.prompt_video = "USER: <video>\nWhy is this video funny? ASSISTANT:"

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test(self):
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            "llava-hf/LLaVA-NeXT-Video-7B-hf", load_in_4bit=True, cache_dir="./"
        )

        inputs = self.processor(self.prompt_video, videos=self.video, return_tensors="pt")
        expected_input_ids = [
            1,
            3148,
            1001,
            29901,
            29871,
            32000,
            13,
            11008,
            338,
            445,
            4863,
            2090,
            1460,
            29973,
            319,
            1799,
            9047,
            13566,
            29901,
        ]
        self.assertListEqual(expected_input_ids, inputs.input_ids[0].tolist())

        # verify single forward pass
        inputs = inputs.to(torch_device)
        with torch.no_grad():
            output = model(**inputs)

        # verify generation
        output = model.generate(**inputs, do_sample=False, max_new_tokens=40)
        EXPECTED_DECODED_TEXT = 'USER: \nWhy is this video funny? ASSISTANT: The humor in this video comes from the unexpected and exaggerated reactions of the child to the book. The child appears to be reading a book, but instead of a calm and focused reading experience'  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_batch(self):
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            "llava-hf/LLaVA-NeXT-Video-7B-hf", load_in_4bit=True, cache_dir="./"
        )

        inputs = self.processor(
            [self.prompt_video, self.prompt_video],
            videos=[self.video, self.video],
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        output = model.generate(**inputs, do_sample=False, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = ['USER: \nWhy is this video funny? ASSISTANT: The humor in this video comes from the unexpected and exaggerated reactions of the child to the', 'USER: \nWhy is this video funny? ASSISTANT: The humor in this video comes from the unexpected and exaggerated reactions of the child to the']  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_batch_different_vision_types(self):
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            "llava-hf/LLaVA-NeXT-Video-7B-hf",
            load_in_4bit=True,
            cache_dir="./",
        )

        inputs = self.processor(
            [self.prompt_image, self.prompt_video],
            images=self.image,
            videos=self.video,
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        # check loss when labels are passed
        inputs["labels"] = inputs["input_ids"].clone()
        with torch.no_grad():
            output = model(**inputs)
        self.assertTrue(output.loss is not None)

        # verify generation
        output = model.generate(**inputs, do_sample=False, max_new_tokens=50)
        EXPECTED_DECODED_TEXT = 'USER: \nWhat is shown in this image? ASSISTANT: The image appears to be a graphical representation of a benchmark test for a machine learning model. It shows the performance of various models on a task, with the x-axis representing the number of parameters (measured in millions) and the y'  # fmt: skip
        self.assertEqual(self.processor.decode(output[0], skip_special_tokens=True), EXPECTED_DECODED_TEXT)

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_batch_matches_single(self):
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            "llava-hf/LLaVA-NeXT-Video-7B-hf", load_in_4bit=True, cache_dir="./"
        )

        inputs_batched = self.processor(
            [self.prompt_video, self.prompt_image],
            images=[self.image],
            videos=[self.video],
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        inputs_single = self.processor(self.prompt_video, videos=[self.video], return_tensors="pt").to(torch_device)

        # verify generation
        output_batched = model.generate(**inputs_batched, do_sample=False, max_new_tokens=50)
        output_single = model.generate(**inputs_single, do_sample=False, max_new_tokens=50)
        self.assertEqual(
            self.processor.decode(output_batched[0], skip_special_tokens=True),
            self.processor.decode(output_single[0], skip_special_tokens=True),
        )

    @slow
    @require_bitsandbytes
    def test_padding_side_when_merging_inputs(self):
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            "llava-hf/LLaVA-NeXT-Video-7B-hf", load_in_4bit=True
        )

        inputs_batched = self.processor(
            [self.prompt_video, self.prompt_image],
            images=[self.image],
            videos=[self.video],
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        # model is in eval mode by default so we should get pad on the left side
        # we can check the first hidden-states (aka inputs embeds)
        # the first element was lo-res image and we expect the first 1482 tokens to be all pads
        output_eval = model(**inputs_batched, output_hidden_states=True)
        self.assertTrue((output_eval.hidden_states[0][0, :1482, ...] == 0).all().item())

        # otherwise padding is on the right side, so it's last 1482 tokens
        self.processor.padding_side = "right"
        inputs_batched = self.processor(
            [self.prompt_video, self.prompt_image],
            images=[self.image],
            videos=[self.video],
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        model.train()
        with torch.no_grad():
            output_train = model(**inputs_batched, output_hidden_states=True)
        self.assertTrue((output_train.hidden_states[0][0, -1482:, ...] == 0).all().item())

        with self.assertLogs("transformers", level="WARNING") as logs:
            model.padding_side = "left"
            model.train()
            model(**inputs_batched, output_hidden_states=True)

            self.assertIn(
                "Padding side is set to 'left' but the model is in training mode. For training", logs.output[0]
            )

        with self.assertLogs("transformers", level="WARNING") as logs:
            model.padding_side = "right"
            model.eval()
            model(**inputs_batched, output_hidden_states=True)

            self.assertIn(
                "Padding side is set to 'right' but the model is in inference mode. For correct", logs.output[0]
            )

    @slow
    @require_bitsandbytes
    def test_expansion_in_processing(self):
        model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            "llava-hf/LLaVA-NeXT-Video-7B-hf", load_in_4bit=True
        )
        processor = AutoProcessor.from_pretrained(model_id)

        # check processing with expansion of inputs
        processor.vision_feature_select_strategy = "default"
        processor.patch_size = 14
        inputs_expanded = processor(self.prompt_video, videos=[self.video], return_tensors="pt").to(torch_device)
        self.assertTrue(inputs_expanded.input_ids.shape[-1] == 1170)

        # check processing without expansion of inputs (legacy behavior)
        processor.vision_feature_select_strategy = None
        processor.patch_size = None
        inputs = processor(self.prompt_video, videos=[self.video], return_tensors="pt").to(torch_device)
        self.assertTrue(inputs.input_ids.shape[-1] == 19)

        # generate exactly 20 tokens
        output = model.generate(**inputs, min_new_tokens=20, max_new_tokens=20)
        output_expanded = model.generate(**inputs_expanded, min_new_tokens=20, max_new_tokens=20)

        # check that both inputs are handled correctly and generate the same output
        self.assertListEqual(output_expanded[:, -20:].tolist(), output[:, -20:].tolist())
