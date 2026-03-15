# Copyright 2026 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Qwen3.5 model."""

import copy
import unittest

from transformers import AutoProcessor, AutoTokenizer, is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch

    from transformers import (
        Qwen3_5Config,
        Qwen3_5ForCausalLM,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5ForSequenceClassification,
        Qwen3_5Model,
        Qwen3_5TextConfig,
        Qwen3_5TextModel,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache


class Qwen3_5TextModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Qwen3_5TextModel
        causal_lm_class = Qwen3_5ForCausalLM
        sequence_classification_class = Qwen3_5ForSequenceClassification

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.layer_types = ["full_attention", "linear_attention"]
        self.linear_conv_kernel_dim = 2
        self.linear_key_head_dim = 16
        self.linear_value_head_dim = 16
        self.linear_num_key_heads = 4
        self.linear_num_value_heads = 8


@require_torch
class Qwen3_5TextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Qwen3_5TextModelTester
    config_class = Qwen3_5TextConfig
    model_split_percents = [0.5, 0.8, 0.9]

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        "Qwen3.5 has a special Cache as it alternates with gated deltanet layers"
        self.assertIsInstance(past_key_values, Qwen3_5DynamicCache)

        # (batch, kv heads, seq_length, head_dim)
        num_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        expected_shape = (batch_size, num_heads, seq_length, head_dim)

        attention_layer_indices = past_key_values.transformer_layers
        self.assertListEqual(
            [past_key_values.key_cache[idx].shape for idx in attention_layer_indices],
            [expected_shape] * len(attention_layer_indices),
        )
        self.assertListEqual(
            [past_key_values.value_cache[idx].shape for idx in attention_layer_indices],
            [expected_shape] * len(attention_layer_indices),
        )

    def _check_caches_are_equal(self, cache1, cache2):
        "Qwen3.5 has a special Cache as it alternates with gated deltanet layers"
        if not len(cache1) == len(cache2):
            raise ValueError("Both caches do not have the same number of layers.")

        num_layers = len(cache1)
        for idx in range(num_layers):
            if cache1.key_cache[idx] is not None:
                torch.testing.assert_close(cache1.key_cache[idx], cache2.key_cache[idx])
                torch.testing.assert_close(cache1.value_cache[idx], cache2.value_cache[idx])

    def test_attention_outputs(self):
        "Needs to be overwritten as Qwen3.5 alternates between attention layers and gated deltanet layers."
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        # force eager attention to support output attentions
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                self_attentions = outputs.attentions

            self.assertEqual(out_len + 1, len(outputs))
            self.assertEqual(len(self_attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])

    @unittest.skip("The specific cache format cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip("Intentionally not reversable (no changes) as only load time within a VLM depends on this")
    def test_reverse_loading_mapping(self, check_keys_were_modified=True):
        pass


class Qwen3_5VisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_channels=3,
        ignore_index=-100,
        image_size=16,
        text_config={
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 2,
            "hidden_act": "silu",
            "head_dim": 8,
            "hidden_size": 32,
            "vocab_size": 99,
            "intermediate_size": 37,
            "max_position_embeddings": 512,
            "model_type": "qwen3_vl",
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "layer_types": ["full_attention", "linear_attention"],
            "num_key_value_heads": 2,
            "rope_theta": 10000,
            "tie_word_embeddings": True,
            "rope_parameters": {"rope_type": "default", "mrope_section": [16, 8, 8], "mrope_interleaved": True},
            "linear_conv_kernel_dim": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 8,
        },
        vision_config={
            "depth": 2,
            "in_chans": 3,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 32,
            "out_hidden_size": 32,
            "hidden_size": 32,
            "num_heads": 4,
            "patch_size": 16,
            "spatial_merge_size": 1,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
        },
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
        tie_word_embeddings=True,
        is_training=True,
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.is_training = is_training

        self.vision_config = vision_config
        self.text_config = text_config

        self.vocab_size = text_config["vocab_size"]
        self.bos_token_id = text_config["bos_token_id"]
        self.eos_token_id = text_config["eos_token_id"]
        self.pad_token_id = text_config["pad_token_id"]
        self.head_dim = text_config["head_dim"]
        self.hidden_size = text_config["hidden_size"]
        self.intermediate_size = text_config["intermediate_size"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.num_key_value_heads = text_config["num_key_value_heads"]
        self.rope_theta = text_config["rope_theta"]
        self.rope_parameters = text_config["rope_parameters"]
        self.hidden_act = text_config["hidden_act"]
        self.max_position_embeddings = text_config["max_position_embeddings"]
        self.model_type = text_config["model_type"]

        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.tie_word_embeddings = tie_word_embeddings

        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.num_image_tokens = 32
        self.seq_length = seq_length + self.num_image_tokens

    def get_config(self):
        return Qwen3_5Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vision_config.patch_size
        temporal_patch_size = config.vision_config.temporal_patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (patch_size**2),
                self.num_channels * (patch_size**2) * temporal_patch_size,
            ]
        )

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.vision_start_token_id] = self.pad_token_id
        input_ids[:, self.num_image_tokens] = self.image_token_id
        input_ids[:, self.num_image_tokens - 1] = self.vision_start_token_id

        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[:, self.num_image_tokens] = 1

        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, 1, 1]] * self.batch_size, device=torch_device),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
        }
        return config, inputs_dict


@require_torch
class Qwen3_5ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `Qwen3_5ForConditionalGeneration`.
    """

    all_model_classes = (
        (
            Qwen3_5Model,
            Qwen3_5ForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    model_split_percents = [0.5, 0.8, 0.9]

    def setUp(self):
        self.model_tester = Qwen3_5VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Qwen3_5Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        "Qwen3.5 has a special Cache as it alternates with gated deltanet layers"
        self.assertIsInstance(past_key_values, Qwen3_5DynamicCache)

        # (batch, kv heads, seq_length, head_dim)
        num_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        expected_shape = (batch_size, num_heads, seq_length, head_dim)

        attention_layer_indices = past_key_values.transformer_layers
        self.assertListEqual(
            [past_key_values.key_cache[idx].shape for idx in attention_layer_indices],
            [expected_shape] * len(attention_layer_indices),
        )
        self.assertListEqual(
            [past_key_values.value_cache[idx].shape for idx in attention_layer_indices],
            [expected_shape] * len(attention_layer_indices),
        )

    def _check_caches_are_equal(self, cache1, cache2):
        "Qwen3.5 has a special Cache as it alternates with gated deltanet layers"
        if not len(cache1) == len(cache2):
            raise ValueError("Both caches do not have the same number of layers.")

        num_layers = len(cache1)
        for idx in range(num_layers):
            if cache1.key_cache[idx] is not None:
                torch.testing.assert_close(cache1.key_cache[idx], cache2.key_cache[idx])
                torch.testing.assert_close(cache1.value_cache[idx], cache2.value_cache[idx])

    def test_attention_outputs(self):
        "Needs to be overwritten as Qwen3.5 alternates between attention layers and gated deltanet layers."
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        # force eager attention to support output attentions
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(
                len(attentions), sum(layer == "full_attention" for layer in config.text_config.layer_types)
            )

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.text_config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(
                len(attentions), sum(layer == "full_attention" for layer in config.text_config.layer_types)
            )
            self.assertListEqual(
                list(attentions[0].shape[-3:]), [config.text_config.num_attention_heads, seq_len, seq_len]
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                self_attentions = outputs.attentions

            self.assertEqual(out_len + 1, len(outputs))
            self.assertEqual(
                len(self_attentions), sum(layer == "full_attention" for layer in config.text_config.layer_types)
            )
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]), [config.text_config.num_attention_heads, seq_len, seq_len]
            )

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs throw an explicit error when image count mismatches image-token count in text.
        Also checks multi-image cases where one prompt has multiple image tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            _ = model(**input_dict)
            curr_input_dict = copy.deepcopy(input_dict)

            patch_size = config.vision_config.patch_size
            one_img_length = (self.model_tester.image_size**2) // (patch_size**2)
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-one_img_length:, ...]
            curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            if hasattr(model.base_model, "rope_deltas"):
                model.base_model.rope_deltas = None

            input_ids = curr_input_dict["input_ids"][:1]
            mm_token_type_ids = curr_input_dict["mm_token_type_ids"][:1]
            pixel_values = curr_input_dict["pixel_values"][:one_img_length]
            image_grid_thw = curr_input_dict["image_grid_thw"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)
            mm_token_type_ids = torch.cat([mm_token_type_ids, mm_token_type_ids], dim=0)

            with self.assertRaises(ValueError):
                _ = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    mm_token_type_ids=mm_token_type_ids,
                )

            if hasattr(model.base_model, "rope_deltas"):
                model.base_model.rope_deltas = None

            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            image_grid_thw = torch.cat([image_grid_thw, image_grid_thw], dim=0)
            _ = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )

    def test_image_forward(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        bsz = self.model_tester.batch_size
        channels = config.vision_config.in_chans
        temporal_patch = config.vision_config.temporal_patch_size
        patch_size = config.vision_config.patch_size
        num_images = 2

        input_ids = ids_tensor([bsz, self.model_tester.seq_length], self.model_tester.vocab_size)
        input_ids[:, -1] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.video_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.image_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_start_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_end_token_id] = self.model_tester.pad_token_id

        patches_per_image = 1
        pixel_values = floats_tensor(
            [
                bsz * num_images * patches_per_image,
                channels * temporal_patch * (patch_size**2),
            ]
        )
        image_grid_thw = torch.tensor([[1, 1, 1]] * (bsz * num_images))
        self.assertEqual(pixel_values.shape[0], image_grid_thw.prod(dim=1).sum().item())

        insertion_point = 0
        tokens_per_image = 3  # vision_start + image_token + vision_end
        required_seq_length = insertion_point + num_images * tokens_per_image
        self.assertLessEqual(required_seq_length, input_ids.shape[1])

        for b in range(bsz):
            for image_idx in range(num_images):
                image_start = insertion_point + image_idx * tokens_per_image
                input_ids[b, image_start] = self.model_tester.vision_start_token_id
                input_ids[b, image_start + 1] = self.model_tester.image_token_id
                input_ids[b, image_start + 2] = self.model_tester.vision_end_token_id

        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.model_tester.image_token_id] = 1

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.assertIsNotNone(outputs)

    def test_video_forward(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        bsz = self.model_tester.batch_size
        channels = config.vision_config.in_chans
        temporal_patch = config.vision_config.temporal_patch_size
        patch_size = config.vision_config.patch_size

        input_ids = ids_tensor([bsz, self.model_tester.seq_length], self.model_tester.vocab_size)

        frames = 4
        num_video = 2
        frame_timestamp_tokens = 5
        patch_h = self.model_tester.image_size // patch_size
        patch_w = self.model_tester.image_size // patch_size
        patch_t = frames // temporal_patch
        patches_per_video = patch_t * patch_h * patch_w
        patches_per_frame = patch_h * patch_w
        pixel_values_videos = floats_tensor(
            [
                bsz * num_video * patches_per_video,
                channels * temporal_patch * (patch_size**2),
            ]
        )

        video_grid_thw = torch.tensor([[patch_t, patch_h, patch_w]] * (bsz * num_video))
        self.assertEqual(pixel_values_videos.shape[0], video_grid_thw.prod(dim=1).sum().item())

        input_ids[:, -1] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.video_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.image_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_start_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_end_token_id] = self.model_tester.pad_token_id

        insertion_point = 0
        tokens_per_frame = frame_timestamp_tokens + 1 + patches_per_frame + 1
        tokens_per_video = patch_t * tokens_per_frame
        required_seq_length = insertion_point + num_video * tokens_per_video
        if required_seq_length > input_ids.shape[1]:
            pad_extension = torch.full(
                (bsz, required_seq_length - input_ids.shape[1]),
                self.model_tester.pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, pad_extension], dim=1)

        timestamp_start_token_id = self.model_tester.vision_end_token_id + 1
        self.assertLessEqual(timestamp_start_token_id + frame_timestamp_tokens, self.model_tester.vocab_size)
        timestamp_token_ids = torch.arange(
            timestamp_start_token_id,
            timestamp_start_token_id + frame_timestamp_tokens,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )

        self.assertLessEqual(required_seq_length, input_ids.shape[1])
        for b in range(bsz):
            for video_idx in range(num_video):
                video_start = insertion_point + video_idx * tokens_per_video
                for frame_idx in range(patch_t):
                    frame_start = video_start + frame_idx * tokens_per_frame
                    input_ids[b, frame_start : frame_start + frame_timestamp_tokens] = timestamp_token_ids

                    vision_start_pos = frame_start + frame_timestamp_tokens
                    input_ids[b, vision_start_pos] = self.model_tester.vision_start_token_id

                    frame_token_start = vision_start_pos + 1
                    frame_token_end = frame_token_start + patches_per_frame
                    input_ids[b, frame_token_start:frame_token_end] = self.model_tester.video_token_id
                    input_ids[b, frame_token_end] = self.model_tester.vision_end_token_id

        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.model_tester.video_token_id] = 2

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            outputs = model(
                input_ids=input_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.assertIsNotNone(outputs)

    @unittest.skip("The specific cache format cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass


@require_torch
class Qwen3_5IntegrationTest(unittest.TestCase):
    model_id = "Qwen/Qwen3.5-0.8B"

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = Qwen3_5ForCausalLM.from_pretrained(self.model_id, device_map="auto")
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)

        with torch.no_grad():
            logits = model(input_ids).logits.float().cpu()

        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], len(input_ids[0]))
        self.assertTrue(torch.isfinite(logits).all().item())

        # Greedy token picks on each position should remain stable for this checkpoint.
        expected_argmax = torch.tensor([[198, 74, 9230, 198, 1, 264, 1, 198]])
        torch.testing.assert_close(logits.argmax(-1), expected_argmax)

    @slow
    def test_model_generation(self):
        expected_text_completion = "My favourite condiment is 100% real.\nThe 100% real is the only one that is"
        prompt = "My favourite condiment is "

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=False)
        model = Qwen3_5ForCausalLM.from_pretrained(self.model_id, device_map="auto")
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        generated_ids = model.generate(
            **prompt_inputs,
            max_new_tokens=20,
            do_sample=False,
        )
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        self.assertEqual(expected_text_completion, text)

    @slow
    def test_model_vision_generation(self):
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "What kind of animal is this?"},
                ],
            }
        ]

        processor = AutoProcessor.from_pretrained(self.model_id)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(self.model_id, dtype="auto", device_map="auto")

        inputs = processor.apply_chat_template(
            message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )

        expected_input_ids = [248045, 846, 198, 248053, 248056, 248056, 248056, 248056, 248056, 248056]
        self.assertListEqual(expected_input_ids, inputs.input_ids[0].tolist()[:10])

        expected_pixel_slice = torch.tensor(
            [
                [-0.0902, -0.0824, -0.0824],
                [-0.2627, -0.2627, -0.2627],
                [-0.0824, -0.0902, -0.0902],
                [-0.0118, -0.0510, -0.1137],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        self.assertListEqual(
            [round(x, 3) for x in expected_pixel_slice.flatten().tolist()],
            [round(x, 3) for x in inputs.pixel_values[:4, :3].flatten().tolist()],
        )

        inputs = inputs.to(model.device)
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        decoded_text = processor.decode(output[0], skip_special_tokens=True)
        self.assertIn("What kind of animal is this?", decoded_text)
        self.assertIn("cat", decoded_text.lower())

    @slow
    def test_model_video_generation(self):
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "url": "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
                    },
                    {"type": "text", "text": "Describe the video in short."},
                ],
            }
        ]

        processor = AutoProcessor.from_pretrained(self.model_id, max_image_size={"longest_edge": 50176})
        model = Qwen3_5ForConditionalGeneration.from_pretrained(self.model_id, dtype="auto", device_map="auto")

        inputs = processor.apply_chat_template(
            message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        expected_input_ids = [248045, 846, 198, 27, 15, 13, 18, 6283, 29, 248053]
        self.assertListEqual(expected_input_ids, inputs.input_ids[0].tolist()[:10])

        expected_video_slice = torch.tensor(
            [
                [-0.757, -0.757, -0.757],
                [-0.694, -0.639, -0.498],
                [-0.773, -0.773, -0.773],
                [-0.373, -0.357, -0.373],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        self.assertListEqual(
            [round(x, 3) for x in expected_video_slice.flatten().tolist()],
            [round(x, 3) for x in inputs.pixel_values_videos[:4, :3].flatten().tolist()],
        )

        inputs = inputs.to(model.device)
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        decoded_text = processor.decode(output[0], skip_special_tokens=True)
        self.assertIn("Describe the video in short.", decoded_text)
        self.assertIn("seconds>", decoded_text)
        self.assertIn("assistant", decoded_text)

    @slow
    def test_model_video_generation_batch(self):
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "url": "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
                    },
                    {"type": "text", "text": "Describe the video in short."},
                ],
            }
        ]
        batch_messages = [message, message]

        processor = AutoProcessor.from_pretrained(self.model_id, max_image_size={"longest_edge": 50176})
        model = Qwen3_5ForConditionalGeneration.from_pretrained(self.model_id, dtype="auto", device_map="auto")

        inputs = processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        expected_input_ids = [248045, 846, 198, 27, 15, 13, 18, 6283, 29, 248053]
        self.assertListEqual(expected_input_ids, inputs.input_ids[0].tolist()[:10])
        self.assertListEqual(expected_input_ids, inputs.input_ids[1].tolist()[:10])

        inputs = inputs.to(model.device)
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        decoded_text = processor.batch_decode(output, skip_special_tokens=True)

        expected_decoded_text = [
            "user\n<0.3 seconds><1.3 seconds><2.4 seconds><3.5 seconds><4.6 seconds><5.6 seconds><6.7 seconds><7.8 seconds><8.9 seconds><9.7 seconds>Describe the video in short.\nassistant\n<think>\n\n</think>\n\nA toddler is sitting on a bed and reading a book.\n",
            "user\n<0.3 seconds><1.3 seconds><2.4 seconds><3.5 seconds><4.6 seconds><5.6 seconds><6.7 seconds><7.8 seconds><8.9 seconds><9.7 seconds>Describe the video in short.\nassistant\n<think>\n\n</think>\n\nA toddler is sitting on a bed and reading a book.\n",
        ]
        self.assertEqual(expected_decoded_text, decoded_text)

    @slow
    def test_model_video_generation_batch_mixed(self):
        video_message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "url": "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
                    },
                    {"type": "text", "text": "Describe the video in short."},
                ],
            }
        ]
        text_only_message = [{"role": "user", "content": [{"type": "text", "text": "Who are you?"}]}]
        batch_messages = [video_message, text_only_message]

        processor = AutoProcessor.from_pretrained(self.model_id, max_image_size={"longest_edge": 50176})
        processor.tokenizer.padding_side = "left"
        model = Qwen3_5ForConditionalGeneration.from_pretrained(self.model_id, dtype="auto", device_map="auto")

        inputs = processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        decoded_text = processor.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(len(decoded_text), 2)
        self.assertIn("Describe the video in short.", decoded_text[0])
        self.assertIn("seconds>", decoded_text[0])
        self.assertIn("toddler", decoded_text[0].lower())

        self.assertIn("Who are you?", decoded_text[1])
        self.assertIn("qwen", decoded_text[1].lower())

    @slow
    def test_model_video_generation_batch_different_videos(self):
        message_1 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "url": "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
                    },
                    {"type": "text", "text": "Describe the video in short."},
                ],
            }
        ]
        message_2 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4",
                    },
                    {"type": "text", "text": "Describe the video in short."},
                ],
            }
        ]
        batch_messages = [message_1, message_2]

        processor = AutoProcessor.from_pretrained(self.model_id, max_image_size={"longest_edge": 50176})
        processor.tokenizer.padding_side = "left"
        model = Qwen3_5ForConditionalGeneration.from_pretrained(self.model_id, dtype="auto", device_map="auto")

        inputs = processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        decoded_text = processor.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(len(decoded_text), 2)
        self.assertIn("Describe the video in short.", decoded_text[0])
        self.assertIn("seconds>", decoded_text[0])
        self.assertIn("Describe the video in short.", decoded_text[1])
        self.assertIn("seconds>", decoded_text[1])
        self.assertIn("tennis", decoded_text[1].lower())
