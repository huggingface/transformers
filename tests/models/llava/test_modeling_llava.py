# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch Llava model. """

import unittest

import requests

from transformers import AutoProcessor, LlavaConfig, LlavaForCausalLM, is_torch_available, is_vision_available
from transformers.testing_utils import (
    require_torch,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch
else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image


class LlavaVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_index=1,
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
            "batch_size": 12,
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

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = 3
        self.image_size = 336

    def get_config(self):
        return LlavaConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            ignore_index=self.ignore_index,
            image_token_index=self.image_token_index,
            projector_hidden_act=self.projector_hidden_act,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            vision_feature_layer=self.vision_feature_layer,
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
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size)
        attention_mask = input_ids.ne(1).to(torch_device)
        input_ids[:,1] = config.image_token_index
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class LlavaForCausalLMModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as CLIP does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (LlavaForCausalLM,) if is_torch_available() else ()
    fx_compatible = True
    test_pruning = False
    test_resize_embeddings = True  # TODO we need to support resizing
    test_head_masking = False

    def setUp(self):
        self.model_tester = LlavaVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LlavaConfig, has_text_modality=False)

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


@require_torch
class LlavaForCausalLMIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("ArthurZ/llava-1.5-7b")

    def test_large_model_integration_test(self):
        # Let' s make sure we test the preprocessing to replace what is used
        inputs = self.processor.tokenizer(
            ["Hey how are you", "This seems<s>Odd not?</s>."], return_tensors="pt", padding=True
        )
        torch.testing.assert_close(
            inputs["input_ids"],
            torch.tensor(
                [[1, 18637, 920, 526, 366, 0, 0, 0, 0, 0], [1, 910, 2444, 1, 29949, 1289, 451, 29973, 2, 29889]]
            ),
        )
        torch.testing.assert_close(
            inputs["attention_mask"], torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        )

        # model.prepare_inputs_labels_for_multimodal(input_ids = inputs["input_ids"], past_key_values = None, attention_mask = inputs["attention_mask"], position_ids=None, labels=None,images=None)
        prepared_inputs = self.model.prepare_inputs_for_generation(**inputs)
        torch.testing.assert_close(
            prepared_inputs,
            (
                torch.tensor(
                    [[1, 18637, 920, 526, 366, 0, 0, 0, 0, 0], [1, 910, 2444, 1, 29949, 1289, 451, 29973, 2, 29889]]
                ),
                None,
                torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
                None,
                None,
                None,
            ),
        )

    def test_small_model_integration_test(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = LlavaForCausalLM.from_pretrained("ArthurZ/llava-1.5-7b")
        EXPECTED_OUTPUTS = torch.tensor([[1, -200, 29871, 13, 11889, 29901, 1724, 526, 278, 2712, 306, 881, 367, 274, 1300, 2738, 1048, 746, 306, 6493, 445, 2058, 29973, 13, 22933, 9047, 13566, 29901, 1932, 6493, 292, 445, 2058, 29892, 607, 5692, 304, 367, 263, 9307, 470, 23630, 23771, 964, 263, 3573, 310, 4094, 29892, 727, 526, 263, 2846, 2712, 304, 367, 274, 1300, 2738, 1048, 29889, 3824, 29892, 367, 9543, 310, 278, 4094, 10809, 322, 738, 7037, 447, 29920, 3163, 29892, 1316, 408, 1014, 1050, 3192, 23150, 470, 316, 1182, 275, 29889, 6440, 29892, 367, 3458, 1319, 310, 278, 14826, 5855, 29892, 408, 8327, 3620, 297, 14826, 508, 1207, 278, 9307, 470, 23630, 25110, 304, 6686, 373, 29889, 18008, 29892, 367, 274, 1300, 2738, 310, 278, 18830, 5177, 29892, 408, 727, 1795, 367, 8775, 19264, 470, 916, 7037, 447, 29920, 3163, 297, 278, 4038, 29889, 9208, 368, 29892, 367, 9543, 310, 738, 1887, 1072, 8250, 470, 1410, 10652, 1475, 363, 278, 671, 310, 278, 9307, 470, 23630, 29892, 408, 777, 10161, 1122, 505, 2702, 6865, 470, 25091, 297, 2058, 29889, 2,]])  # fmt: skip
        prompt = "<image>\nUSER: What are the things I should be cautious about when I visit this place?\nASSISTANT:"
        image_file = "https://llava-vl.github.io/static/images/view.jpg"
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = self.processor(prompt, raw_image, return_tensors="pt")

        EXPECTED_INPUT_IDS = torch.tensor([[1, -200, 29871, 13, 11889, 29901, 1724, 526, 278, 2712, 306, 881, 367, 274, 1300, 2738, 1048, 746, 306, 6493, 445, 2058, 29973, 13, 22933, 9047, 13566, 29901,]])  # fmt: skip
        torch.testing.assert_close(inputs["input_ids"], EXPECTED_INPUT_IDS)

        output = model.generate(**inputs, max_new_tokens=200)
        torch.testing.assert_close(output, EXPECTED_OUTPUTS)

        EXPECTED_DECODED_TEXT = """\nUSER: What are the things I should be cautious about when I visit this place?\nASSISTANT: When visiting this place, which appears to be a pier or dock extending into a body of water, there are a few things to be cautious about. First, be aware of the water depth and any potential hazards, such as submerged rocks or debris. Second, be mindful of the weather conditions, as sudden changes in weather can make the pier or dock unsafe to walk on. Third, be cautious of the surrounding environment, as there might be wildlife or other potential hazards in the area. Lastly, be aware of any local regulations or guidelines for the use of the pier or dock, as some areas may have specific rules or restrictions in place."""  # fmt: skip
        self.assertEqual(self.processor.decode(output[0][2:], skip_special_tokens=True), EXPECTED_DECODED_TEXT)

    def test_small_model_integration_test_batch(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = LlavaForCausalLM.from_pretrained("ArthurZ/llava-1.5-7b")
        EXPECTED_OUTPUTS = torch.tensor([[1, -200, 29871, 13, 11889, 29901, 1724, 526, 278, 2712, 306, 881, 367, 274, 1300, 2738, 1048, 746, 306, 6493, 445, 2058, 29973, 13, 22933, 9047, 13566, 29901, 1932, 6493, 292, 445, 2058, 29892, 607, 5692, 304, 367, 263, 9307, 470, 23630, 23771, 964, 263, 3573, 310, 4094, 29892, 727, 526, 263, 2846, 2712, 304, 367, 274, 1300, 2738, 1048, 29889, 3824, 29892, 367, 9543, 310, 278, 4094, 10809, 322, 738, 7037, 447, 29920, 3163, 29892, 1316, 408, 1014, 1050, 3192, 23150, 470, 316, 1182, 275, 29889, 6440, 29892, 367, 3458, 1319, 310, 278, 14826, 5855, 29892, 408, 8327, 3620, 297, 14826, 508, 1207, 278, 9307, 470, 23630, 25110, 304, 6686, 373, 29889, 18008, 29892, 367, 274, 1300, 2738, 310, 278, 18830, 5177, 29892, 408, 727, 1795, 367, 8775, 19264, 470, 916, 7037, 447, 29920, 3163, 297, 278, 4038, 29889, 9208, 368, 29892, 367, 9543, 310, 738, 1887, 1072, 8250, 470, 1410, 10652, 1475, 363, 278, 671, 310, 278, 9307, 470, 23630, 29892, 408, 777, 10161, 1122, 505, 2702, 6865, 470, 25091, 297, 2058, 29889, 2,]])  # fmt: skip
        # The first batch is longer in terms of text, but only has 1 image. The second batch will be padded in text, but the first will be padded because images take more space!.
        prompts = [
            "<image>\nUSER: What are the things I should be cautious about when I visit this place? What should I bring with me\nASSISTANT:",
            "<image>\nUSER: What is this?\nASSISTANT: Two cats lying on a bed!\nUSER:\nAnd this?<image>",
        ]
        image_file = "https://llava-vl.github.io/static/images/view.jpg"
        raw_image = Image.open(requests.get(image_file, stream=True).raw)

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        raw_image_bis = Image.open(requests.get(url, stream=True).raw)

        inputs = self.processor(prompts, [raw_image, raw_image_bis, raw_image], return_tensors="pt")

        # EXPECTED_INPUT_IDS = torch.tensor([[1, -200, 29871, 13, 11889, 29901, 1724, 526, 278, 2712, 306, 881, 367, 274, 1300, 2738, 1048, 746, 306, 6493, 445, 2058, 29973, 13, 22933, 9047, 13566, 29901,]]) # fmt: skip
        # torch.testing.assert_close(inputs["input_ids"], EXPECTED_INPUT_IDS)

        output = model.generate(**inputs, max_new_tokens=200)
        torch.testing.assert_close(output, EXPECTED_OUTPUTS)

        # EXPECTED_DECODED_TEXT = """\nUSER: What are the things I should be cautious about when I visit this place?\nASSISTANT: When visiting this place, which appears to be a pier or dock extending into a body of water, there are a few things to be cautious about. First, be aware of the water depth and any potential hazards, such as submerged rocks or debris. Second, be mindful of the weather conditions, as sudden changes in weather can make the pier or dock unsafe to walk on. Third, be cautious of the surrounding environment, as there might be wildlife or other potential hazards in the area. Lastly, be aware of any local regulations or guidelines for the use of the pier or dock, as some areas may have specific rules or restrictions in place.""" # fmt: skip
        # self.assertEqual(self.processor.decode(output[0][2:], skip_special_tokens=True), EXPECTED_DECODED_TEXT)

    def test_small_model_logits_test(self):
        # Let' s make sure we test the preprocessing to replace what is used
        pass

    def test_image_text_pre_processing(self):
        # Let' s make sure we test the preprocessing to replace what is used
        pass

    def test_prepare_inputs_labels_for_multimodal(self):
        # Pretty much the most important function
        pass

    def test_tokenizer_image_token(self):
        # prompt = "Make sure to always answer in a truthful manner"
        pass
