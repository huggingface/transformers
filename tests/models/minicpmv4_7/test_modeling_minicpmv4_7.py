# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch MiniCPM-V 4.7 model."""

import unittest

import pytest

from transformers import (
    AutoProcessor,
    MiniCPMV4_7Config,
    is_torch_available,
)
from transformers.models.minicpmv4_7.configuration_minicpmv4_7 import MiniCPMV4_7VisionConfig
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_modeling_common import floats_tensor
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import DynamicCache, MiniCPMV4_7ForConditionalGeneration, MiniCPMV4_7Model
    from transformers.models.minicpmv4_7.modeling_minicpmv4_7 import MiniCPMV4_7ViTWindowAttentionMerger
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


class MiniCPMV4_7VisionText2TextModelTester(VLMModelTester):
    base_model_class = MiniCPMV4_7Model if is_torch_available() else None
    config_class = MiniCPMV4_7Config
    text_config_class = Qwen3_5TextConfig if is_torch_available() else None
    vision_config_class = MiniCPMV4_7VisionConfig
    conditional_generation_class = MiniCPMV4_7ForConditionalGeneration if is_torch_available() else None

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("batch_size", 2)
        kwargs.setdefault("image_token_id", 100)
        # patch_size=8, image_size=32 → 4×4 grid → vit_merger [2×2] → merger [1×1] = 1 token
        kwargs.setdefault("image_size", 32)
        kwargs.setdefault("patch_size", 8)
        kwargs.setdefault("num_image_tokens", 1)
        kwargs.setdefault("vocab_size", 256)
        kwargs.setdefault("hidden_size", 64)
        kwargs.setdefault("intermediate_size", 37)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 32)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("max_position_embeddings", 512)
        kwargs.setdefault("rope_parameters", {"type": "default", "rope_theta": 10_000, "mrope_section": [2, 1, 1]})
        kwargs.setdefault("tie_word_embeddings", True)
        kwargs.setdefault("bos_token_id", 0)
        kwargs.setdefault("eos_token_id", 1)
        kwargs.setdefault("pad_token_id", 2)
        # Qwen3.5 hybrid attention
        kwargs.setdefault("layer_types", ["full_attention", "linear_attention"])
        kwargs.setdefault("linear_conv_kernel_dim", 2)
        kwargs.setdefault("linear_key_head_dim", 16)
        kwargs.setdefault("linear_value_head_dim", 16)
        kwargs.setdefault("linear_num_key_heads", 4)
        kwargs.setdefault("linear_num_value_heads", 8)
        # Vision config overrides
        kwargs.setdefault("vision_hidden_act", "gelu_pytorch_tanh")
        kwargs.setdefault("vision_intermediate_size", 128)
        # MiniCPM-V 4.6 specific
        kwargs.setdefault("insert_layer_id", 0)
        super().__init__(parent, **kwargs)

    def _navit_pixel_values(self, batch_size):
        """Build NaViT-packed pixel_values: (1, C, patch_size, total_L)."""
        C = self.num_channels
        P = self.patch_size
        h_patches = self.image_size // self.patch_size
        w_patches = self.image_size // self.patch_size
        total_L = batch_size * h_patches * w_patches * P
        return floats_tensor([1, C, P, total_L])

    def _target_sizes(self, batch_size):
        h_patches = self.image_size // self.patch_size
        w_patches = self.image_size // self.patch_size
        return torch.tensor([[h_patches, w_patches]] * batch_size, dtype=torch.int32, device=torch_device)

    def create_pixel_values(self):
        return self._navit_pixel_values(self.batch_size)

    def get_additional_inputs(self, config, input_ids, pixel_values):
        return {"target_sizes": self._target_sizes(self.batch_size)}

    def get_config(self):
        text_config = {
            "model_type": "qwen3_5_text",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "hidden_act": "silu",
            "max_position_embeddings": self.max_position_embeddings,
            "rope_parameters": self.rope_parameters,
            "tie_word_embeddings": self.tie_word_embeddings,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "layer_types": self.layer_types,
            "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_head_dim": self.linear_value_head_dim,
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
        }
        vision_config = {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.vision_intermediate_size,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_channels": self.num_channels,
            "hidden_act": self.vision_hidden_act,
        }
        return MiniCPMV4_7Config(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=self.image_token_id,
            image_size=self.image_size,
            drop_vision_last_layer=False,
            insert_layer_id=self.insert_layer_id,
        )


@require_torch
class MiniCPMV4_7ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = MiniCPMV4_7VisionText2TextModelTester

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = super().prepare_config_and_inputs_for_generate(batch_size=batch_size)
        inputs_dict["pixel_values"] = self.model_tester._navit_pixel_values(batch_size)
        inputs_dict["target_sizes"] = self.model_tester._target_sizes(batch_size)
        return config, inputs_dict

    def _image_features_prepare_config_and_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict = {
            key: value
            for key, value in inputs_dict.items()
            if ("pixel" in key or "image" in key or key == "target_sizes") and "video" not in key
        }
        return config, inputs_dict

    def _video_features_prepare_config_and_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        return config, {
            "pixel_values_videos": inputs_dict["pixel_values"],
            "target_sizes_videos": inputs_dict["target_sizes"],
        }

    @unittest.skip(
        "NaViT packing puts all images in a single tensor with dim-0 = 1; "
        "the default test cannot correctly simulate image count mismatches"
    )
    def test_mismatching_num_image_tokens(self):
        pass

    @unittest.skip(reason="MiniCPM-V uses custom pixel_values format (list-of-list), skipping common input tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="MiniCPM-V uses custom pixel_values format (list-of-list), skipping common input tests")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="Compile not yet supported for MiniCPM-V models")
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip("FlashAttention only supports fp16 and bf16 data type")
    def test_flash_attn_2_fp32_ln(self):
        pass

    @unittest.skip(reason="MiniCPM-V 4.6 uses Qwen3.5 hybrid cache layers that are incompatible with QuantizedCache.")
    def test_generate_with_quant_cache(self):
        pass

    @unittest.skip(reason="Conversion only for CausalLM loading from saved ConditionalLM")
    def test_reverse_loading_mapping(self, check_keys_were_modified=True):
        pass

    @unittest.skip(
        reason="NaViT packs all images into a single tensor (batch dim=1); "
        "generic batch-splitting logic cannot separate individual samples"
    )
    def test_batching_equivalence(self):
        pass

    @unittest.skip(
        reason="NaViT packs all images into a single tensor (batch dim=1); "
        "generic batch-splitting logic cannot separate individual samples"
    )
    def test_model_forward_default_config_values(self):
        pass

    @unittest.skip(
        reason="get_image_features uses a custom pipeline (vision_tower -> vit_merger -> merger) "
        "that does not accept output_attentions/output_hidden_states kwargs"
    )
    def test_get_image_features_attentions(self):
        pass

    @unittest.skip(
        reason="get_image_features uses a custom pipeline (vision_tower -> vit_merger -> merger) "
        "that does not accept output_attentions/output_hidden_states kwargs"
    )
    def test_get_image_features_hidden_states(self):
        pass

    @unittest.skip(
        reason="get_video_features uses a custom pipeline that does not accept "
        "output_attentions/output_hidden_states kwargs"
    )
    def test_get_video_features_attentions(self):
        pass

    @unittest.skip(
        reason="get_video_features uses a custom pipeline that does not accept "
        "output_attentions/output_hidden_states kwargs"
    )
    def test_get_video_features_hidden_states(self):
        pass

    @unittest.skip(
        "MiniCPM-V generate creates vision-aware embeddings via _build_vlm_inputs; "
        "text-only get_input_embeddings bypass produces different outputs"
    )
    def test_generate_from_inputs_embeds(self):
        pass

    @unittest.skip(reason="Same as test_generate_from_inputs_embeds: vision-aware vs text-only embeddings mismatch")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip(
        "Manual left-padding in test does not adjust image_bound offsets, "
        "causing vision features to be placed at wrong positions"
    )
    def test_left_padding_compatibility(self):
        pass

    @unittest.skip(reason="Batch splitting in compile test incompatible with list-of-list pixel_values")
    @pytest.mark.torch_compile_test
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @unittest.skip(reason="Batch splitting in compile test incompatible with list-of-list pixel_values")
    @pytest.mark.torch_compile_test
    def test_generate_compilation_all_outputs(self):
        pass

    @unittest.skip(reason="FA works on generate test, inference needs override to pass target sizes")
    def test_flash_attn_2_inference_equivalence(self):
        pass

    @unittest.skip(reason="FA works on generate, inference needs override to pass target sizes")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
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
        """Overwritten: Qwen3.5 alternates between full attention and gated deltanet layers."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
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

    # Canvas M-RoPE. `get_rope_index` is cheap and pure, so it is covered here with the same tiny
    # model the rest of the suite uses. Tokens 100/101 are the image/video placeholders; 10/11 wrap
    # an image, 12/13 wrap a slice. `target_sizes_mrope` carries one `(h, w)` patch grid per visual
    # crop -- images, slices and video frames all go through this single list.
    @staticmethod
    def _mm_token_type_ids(input_ids):
        """What the processor emits: 0 for text, 1 for image tokens, 2 for video tokens."""
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == 100] = 1
        mm_token_type_ids[input_ids == 101] = 2
        return mm_token_type_ids

    def _mrope_model(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.image_start_id = 10
        config.image_end_id = 11
        config.slice_start_id = 12
        config.slice_end_id = 13
        config.newline_id = 14
        return self.model_tester.base_model_class(config).to(torch_device).eval()

    def _get_rope_index(self, input_ids, grids=None):
        model = self._mrope_model()
        input_ids = torch.tensor(input_ids, device=torch_device)
        attention_mask = torch.ones_like(input_ids)
        grids = None if grids is None else [torch.tensor(grids, dtype=torch.int32, device=torch_device)]
        return model.get_rope_index(
            input_ids,
            attention_mask=attention_mask,
            target_sizes_mrope=grids,
            mm_token_type_ids=self._mm_token_type_ids(input_ids),
        )

    def test_get_rope_index_image_lays_out_canvas(self):
        """A single 2x2 image: time is frozen over the span while H/W walk the patch grid."""
        # [bos, im_start, 4 visual patches, im_end, eos]
        position_ids, rope_deltas = self._get_rope_index([[1, 10, 100, 100, 100, 100, 11, 2]], grids=[[8, 8]])

        self.assertEqual(tuple(position_ids.shape), (3, 1, 8))
        temporal, height, width = position_ids[:, 0]
        self.assertEqual(temporal[2:6].unique().numel(), 1)
        self.assertEqual(height[2:6].tolist(), [1, 1, 2, 2])
        self.assertEqual(width[2:6].tolist(), [1, 2, 1, 2])
        # Text after the image restarts one step past the whole canvas, on every channel.
        self.assertEqual(position_ids[:, 0, -1].tolist(), [4, 4, 4])
        # rope_deltas is the usual `max_position + 1 - real_length`.
        self.assertTrue(torch.equal(rope_deltas, torch.tensor([[-3]], device=torch_device)))

    def test_get_rope_index_slice_shares_image_timestep(self):
        """Slices belong to the same picture, so they reuse its timestep and restart H/W."""
        # [bos, im_start, 4 patches, im_end, slice_start, 4 patches, slice_end, eos]
        input_ids = [[1, 10, 100, 100, 100, 100, 11, 12, 100, 100, 100, 100, 13, 2]]
        sliced, _ = self._get_rope_index(input_ids, grids=[[8, 8], [8, 8]])
        unsliced, _ = self._get_rope_index([[1, 10, 100, 100, 100, 100, 11, 2]], grids=[[8, 8]])

        self.assertEqual(tuple(sliced.shape), (3, 1, 14))
        temporal, height, width = sliced[:, 0]
        self.assertEqual(temporal[1:13].unique().numel(), 1)
        # The slice re-uses the very same canvas coordinates as the global view.
        self.assertEqual(height[8:12].tolist(), unsliced[1, 0, 2:6].tolist())
        self.assertEqual(width[8:12].tolist(), unsliced[2, 0, 2:6].tolist())

    def test_get_rope_index_separate_images_advance_time(self):
        """Two crops in their own im_start/im_end spans are different timesteps, unlike slices."""
        input_ids = [[1, 10, 100, 100, 100, 100, 11, 10, 100, 100, 100, 100, 11, 2]]
        position_ids, _ = self._get_rope_index(input_ids, grids=[[8, 8], [8, 8]])

        temporal = position_ids[0, 0]
        self.assertEqual(temporal[2:6].unique().numel(), 1)
        self.assertEqual(temporal[8:12].unique().numel(), 1)
        self.assertLess(int(temporal[2]), int(temporal[8]))

    def test_get_rope_index_left_padding_matches_unpadded(self):
        """Left padding must shift nothing: the canvas is built on the unpadded tokens."""
        model = self._mrope_model()
        grids = [torch.tensor([[8, 8]], dtype=torch.int32, device=torch_device)]

        unpadded = torch.tensor([[1, 10, 100, 100, 100, 100, 11, 2]], device=torch_device)
        baseline, _ = model.get_rope_index(
            unpadded,
            attention_mask=torch.ones_like(unpadded),
            target_sizes_mrope=grids,
            mm_token_type_ids=self._mm_token_type_ids(unpadded),
        )

        padded = torch.tensor([[0, 0, 1, 10, 100, 100, 100, 100, 11, 2]], device=torch_device)
        padded_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]], device=torch_device)
        padded_positions, _ = model.get_rope_index(
            padded,
            attention_mask=padded_mask,
            target_sizes_mrope=grids,
            mm_token_type_ids=self._mm_token_type_ids(padded),
        )

        self.assertTrue(torch.equal(padded_positions[:, 0, 2:], baseline[:, 0]))
        self.assertTrue(
            torch.equal(padded_positions[:, 0, :2], torch.zeros(3, 2, device=torch_device, dtype=torch.long))
        )

    def test_compute_3d_position_ids_keeps_decoding_on_the_canvas(self):
        """Every decoding step must get the cached-delta positions back, not `None`."""
        model = self._mrope_model()
        model.rope_deltas = torch.tensor([[-3]], device=torch_device)

        past_key_values = DynamicCache()
        num_kv_heads = model.config.text_config.num_key_value_heads
        head_dim = model.config.text_config.head_dim
        prefilled = torch.zeros(1, num_kv_heads, 8, head_dim, device=torch_device)
        past_key_values.update(prefilled, prefilled, 0)

        inputs_embeds = torch.zeros(1, 1, model.config.text_config.hidden_size, device=torch_device)
        position_ids = model.compute_3d_position_ids(
            input_ids=None, inputs_embeds=inputs_embeds, past_key_values=past_key_values
        )

        self.assertIsNotNone(position_ids)
        self.assertEqual(tuple(position_ids.shape), (3, 1, 1))
        # The 9th token sits at 1-D position 8, shifted onto the canvas by the cached delta.
        self.assertEqual(position_ids.unique().tolist(), [5])

    def test_text_only_generation_keeps_1d_positions(self):
        """A text-only batch still carries `mm_token_type_ids`, but there is no canvas to build."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = self.model_tester.conditional_generation_class(config).to(torch_device).eval()

        input_ids = torch.tensor([[1, 2, 3, 2]], device=torch_device)
        model_kwargs = {
            "attention_mask": torch.ones_like(input_ids),
            "mm_token_type_ids": torch.zeros_like(input_ids),
        }
        position_ids = model._prepare_position_ids_for_generation(input_ids, model_kwargs)

        self.assertEqual(position_ids.tolist(), [[0, 1, 2, 3]])

    # Golden canvas layouts, one entry per shape the processor can emit. The tests above assert
    # canvas *properties*; these pin the exact `(3, batch, seq)` coordinates so the canvas internals
    # stay refactorable without silently moving a single position. Ids: 1 bos, 2/3 text, 10/11 wrap
    # an image, 12/13 wrap a slice, 14 is the "\n" between slice rows, 100 is an image token and 101
    # a video token. `grids` are patch grids -- the default 16x downsample merges 4x4 patches into
    # one LLM token, so an 8x8 patch grid is a 2x2 LLM grid worth 4 visual tokens.
    GOLDEN_CANVAS_LAYOUTS = {
        "single image, no slices": {
            "input_ids": [[1, 10, 100, 100, 100, 100, 11, 2]],
            "grids": [[[8, 8]]],
            "positions": [
                [[0, 1, 1, 1, 1, 1, 1, 4]],
                [[0, 0, 1, 1, 2, 2, 3, 4]],
                [[0, 0, 1, 2, 1, 2, 3, 4]],
            ],
            "deltas": [[-3]],
        },
        "image with a 2x2 slice grid, rows split by a newline": {
            "input_ids": [[1, 10, 100, 100, 100, 100, 11, 12, 100, 13, 12, 100, 13, 14, 12, 100, 13, 12, 100, 13, 2]],
            "grids": [[[8, 8], [4, 4], [4, 4], [4, 4], [4, 4]]],
            "positions": [
                [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4]],
                [[0, 0, 1, 1, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4]],
                [[0, 0, 1, 2, 1, 2, 3, 1, 1, 1, 2, 2, 2, 3, 1, 1, 1, 2, 2, 2, 4]],
            ],
            "deltas": [[-16]],
        },
        "two images separated by text": {
            "input_ids": [[1, 10, 100, 100, 100, 100, 11, 2, 10, 100, 100, 100, 100, 11, 3]],
            "grids": [[[8, 8], [8, 8]]],
            "positions": [
                [[0, 1, 1, 1, 1, 1, 1, 4, 5, 5, 5, 5, 5, 5, 8]],
                [[0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8]],
                [[0, 0, 1, 2, 1, 2, 3, 4, 4, 5, 6, 5, 6, 7, 8]],
            ],
            "deltas": [[-6]],
        },
        "one video, two frames, no separator between them": {
            "input_ids": [[1, 10, 101, 101, 101, 101, 11, 10, 101, 101, 101, 101, 11, 2]],
            "grids": [[[8, 8], [8, 8]]],
            "positions": [
                [[0, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 7]],
                [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7]],
                [[0, 0, 1, 2, 1, 2, 3, 3, 4, 5, 4, 5, 6, 7]],
            ],
            "deltas": [[-6]],
        },
        "an image followed by a two-frame video": {
            "input_ids": [
                [1, 10, 100, 100, 100, 100, 11, 2, 10, 101, 101, 101, 101, 11, 10, 101, 101, 101, 101, 11, 3]
            ],
            "grids": [[[8, 8], [8, 8], [8, 8]]],
            "positions": [
                [[0, 1, 1, 1, 1, 1, 1, 4, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 11]],
                [[0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 11]],
                [[0, 0, 1, 2, 1, 2, 3, 4, 4, 5, 6, 5, 6, 7, 7, 8, 9, 8, 9, 10, 11]],
            ],
            "deltas": [[-9]],
        },
        "left-padded batch, mixed layouts": {
            "input_ids": [
                [0, 0, 0, 1, 10, 100, 100, 100, 100, 11, 2],
                [1, 10, 100, 100, 100, 100, 11, 12, 100, 13, 2],
            ],
            "attention_mask": [[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
            "grids": [[[8, 8]], [[8, 8], [4, 4]]],
            "positions": [
                [[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 4], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3]],
                [[0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 4], [0, 0, 1, 1, 1, 1, 2, 1, 1, 1, 3]],
                [[0, 0, 0, 0, 0, 1, 2, 1, 2, 3, 4], [0, 0, 1, 1, 1, 1, 2, 1, 1, 1, 3]],
            ],
            "deltas": [[-3], [-7]],
        },
    }

    def test_get_rope_index_golden_canvas_layouts(self):
        """Exact canvas coordinates for every layout the processor can emit."""
        model = self._mrope_model()
        for layout, case in self.GOLDEN_CANVAS_LAYOUTS.items():
            with self.subTest(layout=layout):
                input_ids = torch.tensor(case["input_ids"], device=torch_device)
                if "attention_mask" in case:
                    attention_mask = torch.tensor(case["attention_mask"], device=torch_device)
                else:
                    attention_mask = torch.ones_like(input_ids)
                grids = [torch.tensor(grid, dtype=torch.int32, device=torch_device) for grid in case["grids"]]

                position_ids, rope_deltas = model.get_rope_index(
                    input_ids,
                    attention_mask=attention_mask,
                    target_sizes_mrope=grids,
                    mm_token_type_ids=self._mm_token_type_ids(input_ids),
                )

                self.assertEqual(position_ids.tolist(), case["positions"])
                self.assertEqual(rope_deltas.tolist(), case["deltas"])


@require_torch
class MiniCPMV4_7ViTWindowAttentionMergerTest(unittest.TestCase):
    """The merger runs over a packed batch whose per-image patch grids need not agree.

    Slicing emits a global thumbnail grid next to the slice grids, and the two differ for any
    image that is not square-ish, e.g. 896x448 at `max_slice_nums=9` gives
    `[[24, 44], [32, 32], [32, 32]]`.
    """

    def _build_merger(self):
        config = MiniCPMV4_7VisionConfig(
            hidden_size=32,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            window_kernel_size=[2, 2],
            layer_norm_eps=1e-6,
        )
        return config, MiniCPMV4_7ViTWindowAttentionMerger(config).eval()

    def _run(self, target_sizes):
        config, merger = self._build_merger()
        num_patches = sum(height * width for height, width in target_sizes)
        hidden_states = floats_tensor([1, num_patches, config.hidden_size])
        with torch.no_grad():
            merged = merger(hidden_states, torch.tensor(target_sizes, dtype=torch.int32))
        return merged

    def test_uniform_grids(self):
        merged = self._run([[32, 32], [32, 32], [32, 32]])
        self.assertEqual(merged.shape[:2], torch.Size([1, 3 * 16 * 16]))

    def test_non_uniform_grids(self):
        """A thumbnail grid that disagrees with the slice grids used to raise in `view()`."""
        target_sizes = [[24, 44], [32, 32], [32, 32]]
        merged = self._run(target_sizes)
        expected = sum((height // 2) * (width // 2) for height, width in target_sizes)
        self.assertEqual(expected, 776)
        self.assertEqual(merged.shape[:2], torch.Size([1, expected]))

    def test_single_grid(self):
        merged = self._run([[16, 64]])
        self.assertEqual(merged.shape[:2], torch.Size([1, 8 * 32]))

    def test_merges_each_image_against_its_own_grid(self):
        """Merging image-by-image must not depend on the order images are packed in."""
        config, merger = self._build_merger()
        target_sizes = [[24, 44], [32, 32]]
        num_patches = sum(height * width for height, width in target_sizes)
        hidden_states = floats_tensor([1, num_patches, config.hidden_size])

        with torch.no_grad():
            merged = merger(hidden_states, torch.tensor(target_sizes, dtype=torch.int32))
            first_only = merger(hidden_states[:, : 24 * 44, :], torch.tensor(target_sizes[:1], dtype=torch.int32))

        torch.testing.assert_close(merged[:, : 12 * 22, :], first_only)


@slow
@require_torch_accelerator
class MiniCPMV4_7IntegrationTest(unittest.TestCase):
    model_id = "openbmb/MiniCPM-V-4_6"

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_small_model_logits(self):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = MiniCPMV4_7ForConditionalGeneration.from_pretrained(
            self.model_id, device_map="auto", dtype=torch.bfloat16
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits.float().cpu()

        self.assertEqual(logits.shape[0], 1)
        self.assertTrue(torch.isfinite(logits).all().item())

    @slow
    def test_small_model_vision_generation(self):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = MiniCPMV4_7ForConditionalGeneration.from_pretrained(
            self.model_id, device_map="auto", dtype=torch.bfloat16
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/pipeline-cat-chonk.jpeg"
                        ),
                    },
                    {"type": "text", "text": "What kind of animal is this?"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        decoded_text = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        # fmt: off
        EXPECTED_TEXT = Expectations(
            {
                ("cuda", (8, 6)): "The animal in the image is a Pystylus, also known as a Pystylus cat or Eurasian pystylus. It",
                ("cuda", (10, 0)): "The animal in the image is a Pystylus, also known as a Eurasian pystylus or snow leopard cat. It's a",
            }
        ).get_expectation()
        # fmt: on
        self.assertEqual(EXPECTED_TEXT, decoded_text)

    @slow
    def test_small_model_video_generation(self):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = MiniCPMV4_7ForConditionalGeneration.from_pretrained(
            self.model_id, device_map="auto", dtype=torch.bfloat16
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"
                        ),
                    },
                    {"type": "text", "text": "What is shown in this video?"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        decoded_text = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        expected_texts = Expectations(
            {
                ("cuda", None): "The video shows two tennis players engaged in a match or practice session on an indoor tennis court. The player in the foreground is positioned at the net,",
            }
        )  # fmt: skip
        EXPECTED_TEXT = expected_texts.get_expectation()

        self.assertEqual(EXPECTED_TEXT, decoded_text)

    @slow
    def test_small_model_vision_generation_batch(self):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = MiniCPMV4_7ForConditionalGeneration.from_pretrained(
            self.model_id, device_map="auto", dtype=torch.bfloat16
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/pipeline-cat-chonk.jpeg"
                        ),
                    },
                    {"type": "text", "text": "What kind of animal is this?"},
                ],
            }
        ]
        batch_messages = [messages, messages]

        inputs = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device, dtype=torch.bfloat16)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        decoded_texts = processor.batch_decode(output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        expected_texts = Expectations(
            {
                ("cuda", (8, 6)): [
                    "The animal in the image is a Pystylus, also known as a Pystylus cat or Eurasian pystylus. It",
                ] * 2,
                ("cuda", (10, 0)): [
                    "The animal in the image is a Pystylus, also known as a Eurasian pystylus or snow leopard cat. It's a",
                ] * 2,
            }
        )  # fmt: skip
        EXPECTED_TEXT = expected_texts.get_expectation()
        self.assertListEqual(decoded_texts, EXPECTED_TEXT)

    @slow
    def test_small_model_vision_generation_batch_mixed(self):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = MiniCPMV4_7ForConditionalGeneration.from_pretrained(
            self.model_id, device_map="auto", dtype=torch.bfloat16
        )

        image_message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/pipeline-cat-chonk.jpeg"
                        ),
                    },
                    {"type": "text", "text": "What kind of animal is this?"},
                ],
            }
        ]
        text_only_message = [{"role": "user", "content": [{"type": "text", "text": "Who are you?"}]}]
        batch_messages = [image_message, text_only_message]

        inputs = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device, dtype=torch.bfloat16)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        decoded_texts = processor.batch_decode(output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        expected_texts = Expectations(
            {
                ("cuda", (8, 6)): [
                    "The animal in the image is a Pystylus, also known as a Pystylus cat or Eurasian pystylus. It",
                    "I'm a model from the MiniCPM series, developed by Modelbest and OpenBMB. For more details, you can visit https://github",
                ],
                ("cuda", (10, 0)): [
                    "The animal in the image is a Pystylus, also known as a Eurasian pystylus or snow leopard cat. It's a",
                    "I'm a model from the MiniCPM series, developed by Modelbest and OpenBMB. For more details, you can visit https://github",
                ],
            }
        )  # fmt: skip
        EXPECTED_TEXT = expected_texts.get_expectation()
        self.assertListEqual(decoded_texts, EXPECTED_TEXT)
