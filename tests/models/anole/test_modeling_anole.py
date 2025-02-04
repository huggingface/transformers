# coding=utf-8
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
"""Testing suite for the PyTorch anole model."""

import unittest

import requests
from parameterized import parameterized

from transformers import AnoleConfig, AnoleVQVAEConfig, is_torch_available, is_vision_available, set_seed
from transformers.testing_utils import (
    require_bitsandbytes,
    require_read_token,
    require_torch,
    require_torch_multi_gpu,
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
        AnoleForConditionalGeneration,
        AnoleModel,
        AnoleVQVAE,
        ChameleonProcessor,
    )


class AnoleModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=False,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        image_token_index=1,
        boi_token_id=97,
        eoi_token_id=96,
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
        image_size=10,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        vq_num_embeds=12,
        vq_embed_dim=12,
        vq_channel_multiplier=[1, 2],
        vq_img_token_start_id=10,  # has to be less than vocab size when added with vq_num_embeds
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.image_token_index = image_token_index
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
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
        self.image_size = image_size
        self.vq_num_embeds = vq_num_embeds
        self.vq_embed_dim = vq_embed_dim
        self.vq_channel_multiplier = vq_channel_multiplier
        self.vq_img_token_start_id = vq_img_token_start_id
        self.image_seq_length = 25
        self.seq_length = seq_length + self.image_seq_length

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_ids[input_ids == self.image_token_index] = self.pad_token_id
        input_ids[:, : self.image_seq_length] = self.image_token_index
        pixel_values = floats_tensor([self.batch_size, 3, self.image_size, self.image_size])

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

        return config, input_ids, input_mask, pixel_values, sequence_labels, token_labels, choice_labels

    def get_config(self):
        # create dummy vocab map for image2bpe mapping if it needs remapping
        # we assume that vocab size is big enough to accoun for image tokens somewhere in the beginning
        # same way as in real ckpt, when img tokens are in first half of embeds
        # we will need "vq_num_embeds" amount of tokens

        vocab_map = {i: chr(i) for i in range(self.vocab_size)}
        vocab_map[self.image_token_index] = "<image>"
        start = self.vq_img_token_start_id
        end = self.vq_img_token_start_id + self.vq_num_embeds
        for i in range(start, end):
            image_token_infix = "".join(chr(ord("A") + int(c)) for c in str(i))
            # dummy str for each image token, anything starting with IMGIMG
            vocab_map[i] = f"IMGIMG{image_token_infix}Z"

        return AnoleConfig(
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
            image_token_index=self.image_token_index,
            boi_token_id=self.boi_token_id,
            eoi_token_id=self.eoi_token_id,
        )

    def get_vq_config(self):
        return {
            "embed_dim": self.vq_embed_dim,
            "num_embeddings": self.vq_num_embeds,
            "latent_channels": self.vq_embed_dim,
            "in_channels": 3,
            "base_channels": 32,  # we have a GroupNorm of 32 groups, so can't do less
            "channel_multiplier": self.vq_channel_multiplier,
            "initializer_range": self.initializer_range,
        }

    def create_and_check_model(
        self, config, input_ids, input_mask, pixel_values, sequence_labels, token_labels, choice_labels
    ):
        model = AnoleModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, pixel_values=pixel_values)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        model = AnoleForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.is_decoder = True
        model = AnoleForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            pixel_values,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask, "pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class AnoleModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (AnoleModel, AnoleForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (AnoleForConditionalGeneration,) if is_torch_available() else ()
    test_head_masking = False
    test_pruning = False
    fx_compatible = False

    def setUp(self):
        self.model_tester = AnoleModelTester(self)
        self.config_tester = ConfigTester(self, config_class=AnoleConfig, hidden_size=37)

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
        original_model = AnoleModel(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        config.rope_scaling = {"type": scaling_type, "factor": 10.0}
        scaled_model = AnoleModel(config)
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
            self.assertTrue(torch.allclose(out_embeds, out_ids))

    @unittest.skip("Anole forces some token ids to be -inf!")
    def test_batching_equivalence(self):
        pass


class AnoleVQModelTester:
    def __init__(
        self,
        parent,
        batch_size=5,
        is_training=False,
        initializer_range=0.02,
        image_size=30,
        num_embeds=12,
        base_channels=32,  # we have a GroupNorm of 32 groups, so can't do less
        embed_dim=12,
        channel_multiplier=[1, 2],
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.initializer_range = initializer_range
        self.image_size = image_size
        self.base_channels = base_channels
        self.num_embeds = num_embeds
        self.embed_dim = embed_dim
        self.channel_multiplier = channel_multiplier

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, 3, self.image_size, self.image_size])
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return AnoleVQVAEConfig(
            embed_dim=self.embed_dim,
            num_embeddings=self.num_embeds,
            latent_channels=self.embed_dim,
            in_channels=3,
            base_channels=self.base_channels,
            channel_multiplier=self.channel_multiplier,
            initializer_range=self.initializer_range,
            resolution=self.image_size,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class AnoleVQModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (AnoleVQVAE,) if is_torch_available() else ()
    test_head_masking = False
    test_pruning = False
    fx_compatible = False
    has_attentions = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = AnoleVQModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=AnoleVQVAEConfig,
            has_text_modality=False,
            common_properties=["embed_dim", "num_embeddings"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip("Anole VQ module cannot offload due to using `self.weight` directly")
    def test_cpu_offload(self):
        pass

    @unittest.skip("Anole VQ module cannot offload due to using `self.weight` directly")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip("Anole VQ module cannot offload due to using `self.weight` directly")
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip("Anole VQ module has no hidden states")
    def test_hidden_states_output(self):
        pass

    @unittest.skip("Anole VQ module has no hidden states")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip("Anole VQ module has no get/set embeddings method")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip("Anole VQ module has no hidden states")
    def test_retain_grad_hidden_states_attentions(self):
        pass


@require_torch
class AnoleIntegrationTest(unittest.TestCase):
    @slow
    @require_bitsandbytes
    @require_read_token
    def test_model_7b(self):
        model = AnoleForConditionalGeneration.from_pretrained(
            "facebook/anole-7b", load_in_4bit=True, device_map="auto"
        )
        processor = ChameleonProcessor.from_pretrained("facebook/anole-7b")

        image = Image.open(
            requests.get("https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg", stream=True).raw
        )
        prompt = "<image>Describe what do you see here and tell me about the history behind it?"

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)

        # greedy generation outputs
        EXPECTED_TEXT_COMPLETION = ['Describe what do you see here and tell me about the history behind it?The image depicts a star map, with a bright blue dot in the center representing the star Alpha Centauri. The star map is a representation of the night sky, showing the positions of stars in']  # fmt: skip
        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    @require_bitsandbytes
    @require_read_token
    def test_model_7b_batched(self):
        model = AnoleForConditionalGeneration.from_pretrained(
            "facebook/anole-7b", load_in_4bit=True, device_map="auto"
        )
        processor = ChameleonProcessor.from_pretrained("facebook/anole-7b")

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
        EXPECTED_TEXT_COMPLETION = [
            'Describe what do you see here and tell me about the history behind it?The image depicts a star map, with a bright blue dot in the center representing the star Alpha Centauri. The star map is a representation of the night sky, showing the positions of stars in',
            'What constellation is this image showing?The image shows the constellation of Orion.The image shows the constellation of Orion.The image shows the constellation of Orion.The image shows the constellation of Orion.'
            ]  # fmt: skip
        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    @require_bitsandbytes
    @require_read_token
    def test_model_7b_multi_image(self):
        model = AnoleForConditionalGeneration.from_pretrained(
            "facebook/anole-7b", load_in_4bit=True, device_map="auto"
        )
        processor = ChameleonProcessor.from_pretrained("facebook/anole-7b")

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

    @slow
    @require_bitsandbytes
    @require_read_token
    @require_torch_multi_gpu
    def test_model_7b_multi_gpu(self):
        model = AnoleForConditionalGeneration.from_pretrained(
            "facebook/anole-7b",
            load_in_4bit=True,
            device_map="auto",
            max_memory={0: "1GB"},
        )
        processor = ChameleonProcessor.from_pretrained("facebook/anole-7b")

        image = Image.open(
            requests.get("https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg", stream=True).raw
        )
        prompt = "<image>Describe what do you see here and tell me about the history behind it?"

        inputs = processor(prompt, images=image, return_tensors="pt").to(model.device, torch.float16)

        # greedy generation outputs
        EXPECTED_TEXT_COMPLETION = ['Describe what do you see here and tell me about the history behind it?The image depicts a star map, with a bright blue line extending across the center of the image. The line is labeled "390 light years" and is accompanied by a small black and']  # fmt: skip
        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
