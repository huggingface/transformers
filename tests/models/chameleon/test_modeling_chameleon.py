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
"""Testing suite for the PyTorch chameleon model."""

import copy
import unittest

import requests
from parameterized import parameterized

from transformers import ChameleonConfig, is_torch_available, is_vision_available, set_seed
from transformers.testing_utils import (
    Expectations,
    require_bitsandbytes,
    require_read_token,
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_vision_available():
    from PIL import Image

if is_torch_available():
    import torch

    from transformers import (
        ChameleonForConditionalGeneration,
        ChameleonModel,
        ChameleonProcessor,
    )


class ChameleonModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=35,
        is_training=False,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        image_token_id=4,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        vq_num_embeds=5,
        vq_embed_dim=5,
        vq_channel_multiplier=[1, 4],
        vq_img_token_start_id=10,  # has to be less than vocab size when added with vq_num_embeds
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.image_token_id = image_token_id
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope
        self.vq_num_embeds = vq_num_embeds
        self.vq_embed_dim = vq_embed_dim
        self.vq_channel_multiplier = vq_channel_multiplier
        self.vq_img_token_start_id = vq_img_token_start_id

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        # create dummy vocab map for image2bpe mapping if it needs remapping
        # we assume that vocab size is big enough to account for image tokens somewhere in the beginning
        # same way as in real ckpt, when img tokens are in first half of embeds
        # we will need "vq_num_embeds" amount of tokens

        vocab_map = {i: chr(i) for i in range(self.vocab_size)}
        vocab_map[self.image_token_id] = "<image>"
        start = self.vq_img_token_start_id
        end = self.vq_img_token_start_id + self.vq_num_embeds
        for i in range(start, end):
            image_token_infix = "".join(chr(ord("A") + int(c)) for c in str(i))
            # dummy str for each image token, anything starting with IMGIMG
            vocab_map[i] = f"IMGIMG{image_token_infix}Z"

        return ChameleonConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            vocabulary_map={v: k for k, v in vocab_map.items()},
            vq_config=self.get_vq_config(),
        )

    def get_vq_config(self):
        return {
            "embed_dim": self.vq_embed_dim,
            "num_embeddings": self.vq_num_embeds,
            "latent_channels": self.vq_embed_dim,
            "in_channels": 3,
            "base_channels": 32,  # we have a GroupNorm of 32 groups, so can't do less
            "channel_multiplier": self.vq_channel_multiplier,
        }

    def create_and_check_model(self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels):
        model = ChameleonModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class ChameleonModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (ChameleonModel, ChameleonForConditionalGeneration) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": ChameleonModel,
            "text-generation": ChameleonForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False

    def setUp(self):
        self.model_tester = ChameleonModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ChameleonConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @parameterized.expand([("linear",), ("dynamic",)])
    def test_model_rope_scaling(self, scaling_type):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        original_model = ChameleonModel(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        config.rope_scaling = {"type": scaling_type, "factor": 10.0}
        scaled_model = ChameleonModel(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state

        # Dynamic scaling does not change the RoPE embeddings until it receives an input longer than the original
        # maximum sequence length, so the outputs for the short input should match.
        if scaling_type == "dynamic":
            torch.testing.assert_close(original_short_output, scaled_short_output, rtol=1e-5, atol=1e-5)
        else:
            self.assertFalse(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))

        # The output should be different for long inputs
        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-5))

    @unittest.skip("Chameleon forces some token ids to be -inf!")
    def test_batching_equivalence(self):
        pass

    @unittest.skip("Chameleon VQ model cannot be squishes more due to hardcoded layer params in model code")
    def test_model_is_small(self):
        pass


class ChameleonVision2SeqModelTester(ChameleonModelTester):
    def __init__(self, parent, image_size=10, **kwargs):
        super().__init__(parent, **kwargs)
        self.image_size = image_size
        self.image_seq_length = 25

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[:, : self.image_seq_length] = self.image_token_id
        attention_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))
        pixel_values = floats_tensor([self.batch_size, 3, self.image_size, self.image_size])

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class ChameleonVision2SeqModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (ChameleonModel, ChameleonForConditionalGeneration) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "image-text-to-text": ChameleonForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False

    def setUp(self):
        self.model_tester = ChameleonVision2SeqModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ChameleonConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip("Chameleon forces some token ids to be -inf!")
    def test_batching_equivalence(self):
        pass

    @unittest.skip("Chameleon cannot do offload because it uses `self.linear.weight` in forward")
    def test_cpu_offload(self):
        pass

    @unittest.skip("Chameleon cannot do offload because it uses `self.linear.weight` in forward")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip("Chameleon cannot do offload because it uses `self.linear.weight` in forward")
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip("Chameleon VQ model cannot be squishes more due to hardcoded layer params in model code")
    def test_model_is_small(self):
        pass

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs through an error with explicit message saying what is wrong
        when number of images don't match number of image tokens in the text.
        Also we need to test multi-image cases when one prompr has multiple image tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            curr_input_dict = copy.deepcopy(input_dict)  # the below tests modify dict in-place
            _ = model(**curr_input_dict)  # successful forward with no modifications

            # remove one image but leave the image token in text
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            input_ids = curr_input_dict["input_ids"][:1]
            pixel_values = curr_input_dict["pixel_values"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(input_ids=input_ids, pixel_values=pixel_values)

            # two images and two image tokens don't raise an error
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            _ = model(input_ids=input_ids, pixel_values=pixel_values)

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

            inputs_embeds = model.get_input_embeddings()(input_ids)

            with torch.no_grad():
                out_ids = model(input_ids=input_ids, **inputs)[0]
                out_embeds = model(inputs_embeds=inputs_embeds, **inputs)[0]
            torch.testing.assert_close(out_embeds, out_ids)


@require_torch
class ChameleonIntegrationTest(unittest.TestCase):
    @slow
    @require_bitsandbytes
    @require_read_token
    def test_model_7b(self):
        model = ChameleonForConditionalGeneration.from_pretrained(
            "facebook/chameleon-7b", load_in_4bit=True, device_map="auto"
        )
        processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")

        image = Image.open(
            requests.get("https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg", stream=True).raw
        )
        prompt = "<image>Describe what do you see here and tell me about the history behind it?"

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)

        # greedy generation outputs
        EXPECTED_TEXT_COMPLETIONS = Expectations(
            {
                ("xpu", 3): ['Describe what do you see here and tell me about the history behind it?The image depicts a star map, with a bright blue dot in the center representing the star Altair. The star map is set against a black background, with the constellations visible in the night'],
                ("cuda", 7): ['Describe what do you see here and tell me about the history behind it?The image depicts a star map, with a bright blue dot in the center representing the star Alpha Centauri. The star map is a representation of the night sky, showing the positions of stars in'],
                ("cuda", 8): ['Describe what do you see here and tell me about the history behind it?The image depicts a star map, with a bright blue dot representing the position of the star Alpha Centauri. Alpha Centauri is the brightest star in the constellation Centaurus and is located'],
            }
        )  # fmt: skip
        EXPECTED_TEXT_COMPLETION = EXPECTED_TEXT_COMPLETIONS.get_expectation()

        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    @require_bitsandbytes
    @require_read_token
    def test_model_7b_batched(self):
        model = ChameleonForConditionalGeneration.from_pretrained(
            "facebook/chameleon-7b", load_in_4bit=True, device_map="auto"
        )
        processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")

        image = Image.open(
            requests.get("https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg", stream=True).raw
        )
        image_2 = Image.open(
            requests.get("https://www.kxan.com/wp-content/uploads/sites/40/2020/10/ORION.jpg", stream=True).raw
        )
        prompts = [
            "<image>Describe what do you see here and tell me about the history behind it?",
            "What constellation is this image showing?<image>",
        ]

        inputs = processor(images=[image, image_2], text=prompts, padding=True, return_tensors="pt").to(
            model.device, torch.float16
        )

        # greedy generation outputs
        EXPECTED_TEXT_COMPLETIONS = Expectations(
            {
                ("xpu", 3): [
                    'Describe what do you see here and tell me about the history behind it?The image depicts a star map, with a bright blue dot in the center representing the star Altair. The star map is set against a black background, with the constellations visible in the night',
                    'What constellation is this image showing?The image shows the constellation of Orion.The image shows the constellation of Orion.The image shows the constellation of Orion.The image shows the constellation of Orion.',
                ],
                ("cuda", 7): [
                    'Describe what do you see here and tell me about the history behind it?The image depicts a star map, with a bright blue dot representing the position of the star Alpha Centauri. Alpha Centauri is the brightest star in the constellation Centaurus and is located',
                    'What constellation is this image showing?The image shows the constellation of Orion.The image shows the constellation of Orion.The image shows the constellation of Orion.The image shows the constellation of Orion.',
                ],
                ("cuda", 8): [
                    'Describe what do you see here and tell me about the history behind it?The image depicts a star map, with a bright blue dot representing the position of the star Alpha Centauri. Alpha Centauri is the brightest star in the constellation Centaurus and is located',
                    'What constellation is this image showing?The image shows the constellation of Orion.The image shows the constellation of Orion.The image shows the constellation of Orion.The image shows the constellation of Orion.',
                ],
            }
        )  # fmt: skip
        EXPECTED_TEXT_COMPLETION = EXPECTED_TEXT_COMPLETIONS.get_expectation()

        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    @require_bitsandbytes
    @require_read_token
    def test_model_7b_multi_image(self):
        model = ChameleonForConditionalGeneration.from_pretrained(
            "facebook/chameleon-7b", load_in_4bit=True, device_map="auto"
        )
        processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")

        image = Image.open(
            requests.get("https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg", stream=True).raw
        )
        image_2 = Image.open(
            requests.get("https://www.kxan.com/wp-content/uploads/sites/40/2020/10/ORION.jpg", stream=True).raw
        )
        prompt = "What do these two images have in common?<image><image>"

        inputs = processor(images=[image, image_2], text=prompt, return_tensors="pt").to(model.device, torch.float16)

        # greedy generation outputs
        EXPECTED_TEXT_COMPLETION = ['What do these two images have in common?The two images show a connection between the night sky and the internet. The first image shows a starry night sky, with the stars arranged in a pattern that resembles the structure of the internet. The']  # fmt: skip
        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
