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
import tempfile
import unittest

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import (
    require_torch,
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
        AutoModelForCausalLM,
        Qwen3_5MoeConfig,
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3_5MoeModel,
        Qwen3_5MoeTextConfig,
        Qwen3_5MoeTextModel,
    )


class Qwen3_5MoeTextModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Qwen3_5MoeTextModel
        causal_lm_class = Qwen3_5MoeForCausalLM

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.layer_types = ["full_attention", "linear_attention"]
        self.linear_conv_kernel_dim = 2
        self.linear_key_head_dim = 16
        self.linear_value_head_dim = 16
        self.linear_num_key_heads = 4
        self.linear_num_value_heads = 8


@require_torch
class Qwen3_5MoeTextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Qwen3_5MoeTextModelTester
    config_class = Qwen3_5MoeTextConfig

    def _get_conv_state_shape(self, batch_size: int, config):
        num_v_heads = config.linear_num_value_heads
        num_k_heads = config.linear_num_key_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim
        intermediate_size = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim

        return (batch_size, intermediate_size, config.linear_conv_kernel_dim)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        num_v_heads = config.linear_num_value_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim

        return (batch_size, num_v_heads, head_k_dim, head_v_dim)

    def test_attention_outputs(self):
        "Needs to be overwritten as Qwen3.5 Moe alternates between attention layers and gated deltanet layers."
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

    @unittest.skip("Intentionally not reversable (no changes) as only load time within a VLM depends on this")
    def test_reverse_loading_mapping(self, check_keys_were_modified=True):
        pass

    @unittest.skip("The specific cache format cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass


class Qwen3_5MoeVisionText2TextModelTester:
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
            "model_type": "qwen3_5_moe_text",
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
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 36,
            "num_experts_per_tok": 2,
            "num_experts": 8,
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
        return Qwen3_5MoeConfig(
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
        mm_token_type_ids[input_ids == self.image_token_id] = 1
        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, 1, 1]] * self.batch_size, device=torch_device),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
        }
        return config, inputs_dict


@require_torch
class Qwen3_5MoeModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `Qwen3_5MoeForConditionalGeneration`.
    """

    all_model_classes = (
        (
            Qwen3_5MoeModel,
            Qwen3_5MoeForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )

    def setUp(self):
        self.model_tester = Qwen3_5MoeVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Qwen3_5MoeConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @parameterized.expand([("from_pretrained",), ("from_config",)])
    def test_automodelforcausallm(self, loader: str) -> None:
        """`AutoModelForCausalLM` must unwrap the text sub-config for composite-to-text-only mappings."""
        config = self.model_tester.get_config()
        self.assertIsInstance(config, Qwen3_5MoeConfig, msg="Test setup expects the composite Qwen3_5MoeConfig.")

        if loader == "from_config":
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(config)
        else:
            full_model = Qwen3_5MoeForConditionalGeneration(config)
            with tempfile.TemporaryDirectory() as tmp_dir:
                full_model.save_pretrained(tmp_dir)
                model = AutoModelForCausalLM.from_pretrained(tmp_dir)

        self.assertIsInstance(model, Qwen3_5MoeForCausalLM)
        self.assertIsInstance(model.config, Qwen3_5MoeTextConfig)

    @unittest.skip(
        "Conversion only for the `CausalLM` loading from saved `ConditionalLM`, doesn't apply to simple VLM"
    )
    def test_reverse_loading_mapping(self, check_keys_were_modified=True):
        pass

    def _get_conv_state_shape(self, batch_size: int, config):
        num_v_heads = config.linear_num_value_heads
        num_k_heads = config.linear_num_key_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim
        intermediate_size = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim

        return (batch_size, intermediate_size, config.linear_conv_kernel_dim)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        num_v_heads = config.linear_num_value_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim

        return (batch_size, num_v_heads, head_k_dim, head_v_dim)

    def test_attention_outputs(self):
        "Needs to be overwritten as Qwen3.5 Moe alternates between attention layers and gated deltanet layers."
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
            pixel_values = curr_input_dict["pixel_values"][:one_img_length]
            image_grid_thw = curr_input_dict["image_grid_thw"][:1]
            mm_token_type_ids = curr_input_dict["mm_token_type_ids"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            with self.assertRaises(ValueError):
                _ = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    mm_token_type_ids=torch.cat([mm_token_type_ids, mm_token_type_ids], dim=0),
                )

            if hasattr(model.base_model, "rope_deltas"):
                model.base_model.rope_deltas = None

            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            image_grid_thw = torch.cat([image_grid_thw, image_grid_thw], dim=0)
            mm_token_type_ids = torch.cat(
                [curr_input_dict["mm_token_type_ids"][:1], curr_input_dict["mm_token_type_ids"][:1]], dim=0
            )
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
