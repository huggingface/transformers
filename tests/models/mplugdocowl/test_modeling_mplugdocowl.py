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
"""Testing suite for the PyTorch MPLUGDocOwl model."""

import gc
import unittest

import requests
from parameterized import parameterized

from transformers import (
    MPLUGDocOwlConfig,
    MPLUGDocOwlForConditionalGeneration,
    MPLUGDocOwlProcessor,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    require_torch,
    require_torch_sdpa,
    require_vision,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor


if is_torch_available():
    import torch
else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image


class MPLUGDocOwlVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_index=0,
        projector_hidden_act="gelu",
        seq_length=7,
        vision_feature_select_strategy="default",
        hreducer_hidden_size=32,
        hreducer_initializer_range=0.02,
        hreducer_layer_norm=1e-6,
        hreducer_conv_shape="1x2",
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
            "max_position_embeddings": 512,
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 0,
        },
        is_training=True,
        vision_config={
            "image_size": 30,
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
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.text_config = text_config
        self.vision_config = vision_config
        self.seq_length = seq_length
        self.hreducer_hidden_size = hreducer_hidden_size
        self.hreducer_initializer_range = hreducer_initializer_range
        self.hreducer_layer_norm = hreducer_layer_norm
        self.hreducer_conv_shape = hreducer_conv_shape

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = 3
        self.image_size = 336
        self.encoder_seq_length = 112

    def get_config(self):
        return MPLUGDocOwlConfig(
            hreducer_conv_shape=self.hreducer_conv_shape,
            hreducer_hidden_size=self.hreducer_hidden_size,
            hreducer_initializer_range=self.hreducer_initializer_range,
            hreducer_layer_norm=self.hreducer_layer_norm,
            text_config=self.text_config,
            vision_config=self.vision_config,
            ignore_index=self.ignore_index,
            image_token_index=self.image_token_index,
            projector_hidden_act=self.projector_hidden_act,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
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
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(1).to(torch_device)
        # we are giving 3 images let's make sure we pass in 3 image tokens
        input_ids[:, 1] = config.image_token_index
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_mplugdocowl_model_fp16_forward(self, config, input_ids, pixel_values, attention_mask):
        model = MPLUGDocOwlForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values.to(torch.bfloat16),
                return_dict=True,
            )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())


@require_torch
class MPLUGDocOwlForConditionalGenerationModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `MPLUGDocOwlForConditionalGeneration`.
    """

    all_model_classes = (MPLUGDocOwlForConditionalGeneration,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_attention_outputs = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = MPLUGDocOwlVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MPLUGDocOwlConfig, has_text_modality=False)

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

    @unittest.skip(reason="input_embeds cannot be passed in without input_ids")
    def test_inputs_embeds():
        pass

    @require_torch_sdpa
    @slow
    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        self.skipTest(reason="This model does not support SDPA")

    @unittest.skip(reason="MPLUGDocOwl1.5 does not use feedforward chunking.")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="Compile not yet supported in MPLUGDocOwl1.5")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Compile not yet supported in MPLUGDocOwl1.5")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)

            # Ensure all parameters are initialized to 0.0 or 1.0
            for name, param in model.named_parameters():
                if "embeddings" not in name and param.requires_grad:
                    # Explicitly initialize parameters
                    with torch.no_grad():
                        param.fill_(0.0)  # or param.fill_(1.0) based on your requirements

                    # Calculate the rounded mean of the parameter data
                    param_mean = ((param.data.mean() * 1e9).round() / 1e9).item()

                    # Check if the mean is either 0.0 or 1.0
                    try:
                        self.assertIn(
                            param_mean,
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized: found {param_mean}, expected 0.0 or 1.0",
                        )
                    except AssertionError as e:
                        print(f"Initialization error: {e}")
                        raise

    @unittest.skip(
        reason="MPLUGDocOwlVisionModel does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention yet. Thus, cannot be created with no checkpoint."
    )
    def test_from_pretrained_no_checkpoint(self):
        pass


@require_vision
@require_torch
class MPLUGDocOwlForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = MPLUGDocOwlProcessor.from_pretrained("danaaubakirova/mplugdocowl1.5-Chat-hf")

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    @slow
    def test_small_model_integration_test(self):
        model = MPLUGDocOwlForConditionalGeneration.from_pretrained(
            "danaaubakirova/mplugdocowl1.5-Chat-hf", load_in_4bit=False
        )

        prompt = "<image>What's the value of the Very well bar in the 65+ age group? Answer the question with detailed explanation."
        raw_image = Image.open(
            requests.get(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/test_image.png",
                stream=True,
            ).raw
        )
        inputs = self.processor(prompt, raw_image, return_tensors="pt")

        output = model.generate(**inputs, max_new_tokens=500)
        EXPECTED_DECODED_TEXT = """ 68%\nIn the image, which appears to be a chart from a Pew Research Center report, the bar representing the percentage of Republicans and Republican leaners who believe "very well" describes how fights for what they believe in describe Trump is at 68% for the 65+ age group."""

        self.assertEqual(
            self.processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_single(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = MPLUGDocOwlForConditionalGeneration.from_pretrained(
            "danaaubakirova/mplugdocowl1.5-Chat-hf", load_in_4bit=False
        )

        prompt = "<image>What is the name of the movie in the poster? Provide detailed explanation."
        raw_image = Image.open(
            requests.get(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/examples_Rebecca_(1939_poster)_Small.jpeg",
                stream=True,
            ).raw
        )
        inputs = self.processor(prompt, raw_image, return_tensors="pt", do_add_global_image=True)
        output = model.generate(**inputs, max_new_tokens=500)
        EXPECTED_DECODED_TEXT = 'Rebecca\nThe name of the movie in the poster is "Rebecca," as indicated by the large title at the top of the poster. The poster also includes the names of the stars, Laurence Olivier and Joan Fontaine, suggesting that they are the lead actors in the film. The poster features a classic Hollywood style with a focus on the two main characters and the title.'  # fmt: skip
        self.assertEqual(
            self.processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_mplugdocowl_single(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "danaaubakirova/mplugdocowl1.5-Chat-hf"

        model = MPLUGDocOwlForConditionalGeneration.from_pretrained(
            "danaaubakirova/mplugdocowl1.5-Chat-hf", load_in_4bit=False
        )
        processor = MPLUGDocOwlProcessor.from_pretrained(model_id)

        prompt = "<image>Recognize text in the image."
        raw_image = Image.open(
            requests.get(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/test_image.tif",
                stream=True,
            ).raw
        )

        inputs = processor(prompt, raw_image, return_tensors="pt")  # .to(torch_device, torch.float16)

        output = model.generate(**inputs, max_new_tokens=500, do_sample=False)

        EXPECTED_DECODED_TEXT = "PHILIP MORRIS MANAGEMENT CORP."
        self.assertEqual(
            processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    # @require_bitsandbytes
    def test_small_model_integration_test_llama_batched(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "danaaubakirova/mplugdocowl1.5-Chat-hf"

        model = MPLUGDocOwlForConditionalGeneration.from_pretrained(
            "danaaubakirova/mplugdocowl1.5-Chat-hf", load_in_4bit=False
        )
        processor = MPLUGDocOwlProcessor.from_pretrained(model_id)

        prompts = [
            "<image>What is the name of the movie in the poster? Provide detailed explanation.",
            "<image>What is unusual about this image? Provide detailed explanation.",
        ]
        image1 = Image.open(
            requests.get(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/examples_Rebecca_(1939_poster)_Small.jpeg",
                stream=True,
            ).raw
        )
        image2 = Image.open(
            requests.get(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/extreme_ironing.jpg",
                stream=True,
            ).raw
        )

        inputs = processor(text=prompts, images=[image1, image2], return_tensors="pt")

        output = model.generate(**inputs, max_new_tokens=512, do_sample=False, use_cache=True)

        EXPECTED_DECODED_TEXT = [
            'USER: <global_img><crop_img_row0_col0><crop_img_row0_col1><crop_img_row1_col0><crop_img_row1_col1><crop_img_row2_col0><crop_img_row2_col1>What is the name of the movie in the poster? Provide detailed explanation. ASSISTANT: Rebecca\nThe name of the movie in the poster is "Rebecca," as indicated by the large title at the top of the poster. The poster also includes the names of the stars, Laurence Olivier and Joan Fontaine, suggesting that they are the lead actors in the film. The poster features a classic Hollywood style with a focus on the two main characters and the title.',
            "USER: <global_img><crop_img_row0_col0><crop_img_row0_col1><crop_img_row0_col2><crop_img_row1_col0><crop_img_row1_col1><crop_img_row1_col2>What is unusual about this image? Provide detailed explanation. ASSISTANT:\nThe unusual aspect of this image is that the man is ironing clothes on the back of a taxi, which is not a common sight. It is not typical to see someone ironing on the back of a vehicle, especially in an urban setting where such activities are generally not practical due to the lack of space and the potential for disruption to traffic. The presence of a taxi with a man ironing on its back adds an element of surprise and novelty to the scene.",
        ]
        self.assertEqual(
            processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
