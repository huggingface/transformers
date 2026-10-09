# Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch HY4-VL model, using local tiny configurations only."""

import copy
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from huggingface_hub.errors import StrictDataclassClassValidationError
from parameterized import parameterized

from transformers import HYV4VLConfig, HYV4VLTextConfig, HYV4VLVisionConfig, is_torch_available
from transformers.testing_utils import require_torch, require_torch_multi_gpu, torch_device

from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForImageTextToText,
        DynamicCache,
        HYV4VLForConditionalGeneration,
        HYV4VLModel,
        HYV4VLTextModel,
        HYV4VLVisionTransformer,
    )
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key
    from transformers.distributed.tensor_parallel import apply_tensor_parallelism
    from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
    from transformers.models.hy_v4_vl.modeling_hy_v4_vl import (
        HYV4VLVisionPatchMerger,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )


def _check_sharded_attention_sinks(rank, rendezvous):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=2, device_id=device, timeout=timedelta(seconds=120)
    )
    try:
        mesh = init_device_mesh("cuda", (2,))
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(17)
            query, key, value = [torch.randn(1, 4, 3, 8, device=device, dtype=dtype) for _ in range(3)]
            reference_sinks = torch.nn.Parameter(torch.linspace(-1, 1, 4, device=device))
            module = torch.nn.Module()
            module.num_key_value_groups = 1
            module.sinks = torch.nn.Parameter(reference_sinks.detach().clone())
            module.tp_plan = {"sinks": "colwise"}
            apply_tensor_parallelism(module, mesh)
            module.other = torch.nn.Parameter(distribute_tensor(torch.ones(4, device=device), mesh, [Replicate()]))
            sink, other = module.sinks, module.other
            assert isinstance(sink, DTensor) and sink.placements == (Shard(0),)
            head_slice = slice(rank * 2, (rank + 1) * 2)
            torch.testing.assert_close(sink.to_local(), reference_sinks[head_slice], atol=0, rtol=0)

            mask = torch.full((1, 1, 3, 3), torch.finfo(dtype).min, device=device, dtype=dtype).triu(1)
            scores = query @ key.transpose(-1, -2) * 8**-0.5 + mask
            combined = torch.cat([scores, reference_sinks.view(1, -1, 1, 1).expand(1, -1, 3, 1)], dim=-1)
            weights = (combined - combined.max(-1, keepdim=True).values).softmax(-1)[..., :-1].to(dtype)
            expected = (weights @ value).transpose(1, 2).contiguous()
            expected.float().square().sum().backward()

            original_cat = torch.cat

            def checked_cat(*args, **kwargs):
                assert module.sinks is sink and module.other is other
                assert module.state_dict(keep_vars=True)["sinks"] is sink
                return original_cat(*args, **kwargs)

            with patch("torch.cat", side_effect=checked_cat):
                actual, _ = eager_attention_forward(
                    module, query[:, head_slice], key[:, head_slice], value[:, head_slice], mask, scaling=8**-0.5
                )
            torch.testing.assert_close(actual, expected[:, :, head_slice], atol=0, rtol=0)
            actual.float().square().sum().backward()
            torch.testing.assert_close(sink.grad.to_local(), reference_sinks.grad[head_slice], atol=1e-6, rtol=1e-5)
            assert torch.isfinite(sink.grad.to_local()).all() and sink.grad.to_local().abs().sum() > 0
            assert module.sinks is sink and module.other is other
            assert sink.placements == (Shard(0),) and other.placements == (Replicate(),)
            torch.testing.assert_close(sink.full_tensor(), reference_sinks, atol=0, rtol=0)
            with torch.no_grad():
                repeated, _ = eager_attention_forward(
                    module, query[:, head_slice], key[:, head_slice], value[:, head_slice], mask, scaling=8**-0.5
                )
            torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
            assert module.sinks is sink
    finally:
        dist.destroy_process_group()


@require_torch_multi_gpu
class HYV4VLSinkTensorParallelTest(unittest.TestCase):
    def test_sharded_sinks_forward_backward_without_parameter_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            torch.multiprocessing.spawn(_check_sharded_attention_sinks, args=(rendezvous,), nprocs=2, join=True)


class HYV4VLVisionText2TextModelTester(VLMModelTester):
    config_class = HYV4VLConfig
    text_config_class = HYV4VLTextConfig
    vision_config_class = HYV4VLVisionConfig
    if is_torch_available():
        base_model_class = HYV4VLModel
        conditional_generation_class = HYV4VLForConditionalGeneration

    def __init__(self, parent, **kwargs):
        defaults = {
            "batch_size": 2,
            "vocab_size": 256,
            "hidden_size": 48,
            "intermediate_size": 96,
            "moe_intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 16,
            "v_head_dim": 16,
            "q_lora_rank": 32,
            "kv_lora_rank": 16,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "index_head_dim": 16,
            "index_n_heads": 2,
            "index_topk": 16,
            "hc_mult": 2,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [4, 2, 2],
            },
            "video_token_id": 4,
            "image_size": 16,
            "spatial_merge_size": 2,
            "spatial_patch_size": 1,
            "num_image_tokens": 6,
        }
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        super().__init__(parent, **kwargs)
        self.grid_hw = self.image_size // self.patch_size
        self.num_image_patches = self.grid_hw**2

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {self.video_token_id}

    def get_config(self):
        config = super().get_config()
        config._attn_implementation = "eager"
        return config

    def get_vision_config(self):
        return HYV4VLVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=2,
            num_hidden_layers=2,
            patch_size=self.patch_size,
            spatial_merge_size=self.spatial_merge_size,
            spatial_patch_size=self.spatial_patch_size,
            max_image_size=64,
            out_hidden_size=self.hidden_size,
        )

    def create_attention_mask(self, input_ids):
        return torch.ones_like(input_ids)

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return floats_tensor([batch_size * self.num_image_patches, self.num_channels * self.patch_size**2]).to(
            torch_device
        )

    def get_additional_inputs(self, config, input_ids, modality_inputs, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return {
            "image_grid_thw": torch.tensor([[1, self.grid_hw, self.grid_hw]] * batch_size, device=torch_device),
            "mm_token_type_ids": (input_ids == self.image_token_id).long(),
        }

    def prepare_config_and_inputs(self):
        return self.prepare_config_and_inputs_for_common()

    def prepare_multimodal_inputs(self, image_grid=None, video_grid=None):
        input_ids, token_types = [10, 11, 12], [0, 0, 0]
        inputs = {}
        for grid, token_id, modality, pixel_key, grid_key in (
            (image_grid, self.image_token_id, 1, "pixel_values", "image_grid_thw"),
            (video_grid, self.video_token_id, 2, "pixel_values_videos", "video_grid_thw"),
        ):
            if grid is None:
                continue
            frames, height, width = grid
            merge = self.spatial_merge_size * (self.spatial_patch_size if modality == 2 else 1)
            num_tokens = frames * (height // merge) * (width // merge + 1)
            input_ids.extend([token_id] * num_tokens + [13])
            token_types.extend([modality] * num_tokens + [0])
            inputs[pixel_key] = floats_tensor([frames * height * width, self.num_channels * self.patch_size**2]).to(
                torch_device
            )
            inputs[grid_key] = torch.tensor([grid], device=torch_device)
        inputs["input_ids"] = torch.tensor([input_ids], device=torch_device)
        inputs["mm_token_type_ids"] = torch.tensor([token_types], device=torch_device)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        return inputs


@require_torch
class HYV4VLModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = HYV4VLVisionText2TextModelTester
    # Routing only visits a subset of experts; unused experts need not receive a gradient.
    test_all_params_have_gradient = False
    # Packed vision features have a leading singleton, not the pixel stream's patch count.
    # Exact packed feature shapes and newline values are checked below instead.
    skip_test_image_features_output_shape = True
    skip_test_video_features_output_shape = True

    @unittest.skip("GenerationMixin._split_model_outputs does not support four-dimensional iHC hidden states.")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("GenerationMixin._split_model_outputs does not support four-dimensional iHC hidden states.")
    def test_assisted_decoding_matches_greedy_search_0_random(self):
        pass

    @unittest.skip("GenerationMixin._split_model_outputs does not support four-dimensional iHC hidden states.")
    def test_assisted_decoding_matches_greedy_search_1_same(self):
        pass

    def test_can_set_attention_dynamically_composite_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config)).to(torch_device).eval()
            model.set_attn_implementation({"text_config": "eager", "vision_config": "sdpa"})
            self.assertEqual(model.config.text_config._attn_implementation, "eager")
            self.assertEqual(model.config.vision_config._attn_implementation, "sdpa")
            with torch.no_grad():
                sdpa = model(**inputs)[0]
            model.set_attn_implementation({"text_config": "eager", "vision_config": "eager"})
            with torch.no_grad():
                eager = model(**inputs)[0]
            torch.testing.assert_close(sdpa, eager, atol=1e-5, rtol=1e-5)
            with self.assertRaises(ValueError):
                model.set_attn_implementation({"text_config": "sdpa"})

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
    ):
        self.assertIsInstance(hidden_states, tuple)
        self.assertEqual(len(hidden_states), output_length - prompt_length)
        for step, layer_states in enumerate(hidden_states):
            self.assertIsInstance(layer_states, tuple)
            self.assertEqual(len(layer_states), config.num_hidden_layers + 1)
            length = 1 if use_cache and step > 0 else prompt_length + step
            for state in layer_states[:-1]:
                self.assertEqual(state.shape, (batch_size, length, config.hc_mult, config.hidden_size))
            self.assertEqual(layer_states[-1].shape, (batch_size, length, config.hidden_size))

    def test_router_logits_follow_explicit_and_config_defaults(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config)).to(torch_device).eval()
            with torch.no_grad():
                self.assertIsNone(model(**inputs).router_logits)
                explicit = model(**inputs, output_router_logits=True).router_logits
                model.config.text_config.output_router_logits = True
                configured = model(**inputs).router_logits
                self.assertIsNone(model(**inputs, output_router_logits=False).router_logits)
            self.assertEqual(len(explicit), config.text_config.mlp_layer_types.count("sparse"))
            for actual, expected in zip(configured, explicit):
                self.assertEqual(actual.shape[-1], config.text_config.n_routed_experts)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_hidden_states_output(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config)).to(torch_device).eval()
            with torch.no_grad():
                hidden_states = model(**inputs, output_hidden_states=True).hidden_states
            self.assertIsNotNone(hidden_states)
            self.assertEqual(len(hidden_states), config.text_config.num_hidden_layers + 1)
            # iHC keeps residual channels until the final head collapses them.
            for state in hidden_states[:-1]:
                self.assertEqual(
                    state.shape,
                    (*inputs["input_ids"].shape, config.text_config.hc_mult, config.text_config.hidden_size),
                )
            self.assertEqual(hidden_states[-1].shape, (*inputs["input_ids"].shape, config.text_config.hidden_size))

    def test_reverse_loading_mapping(self, check_keys_were_modified=True, skip_base_model=True):
        # Source prefixes describe the complete checkpoint, including the model wrapper.
        return super().test_reverse_loading_mapping(check_keys_were_modified, skip_base_model)

    @parameterized.expand([(False,), (True,)])
    def test_checkpoint_projection_and_text_renaming(self, with_lm_head):
        model_class = HYV4VLForConditionalGeneration if with_lm_head else HYV4VLModel
        model = model_class(self.model_tester.get_config()).to(torch_device)
        transforms = get_model_conversion_mapping(model)
        renamings = [item for item in transforms if isinstance(item, WeightRenaming)]
        converters = [
            item for item in transforms if isinstance(item, WeightConverter) and not isinstance(item, WeightRenaming)
        ]
        state = model.state_dict()
        mappings = {
            "vit.perceive.proj.0.weight": "model.vision_tower.patch_merger.proj_conv.weight",
            "vit.perceive.proj.0.bias": "model.vision_tower.patch_merger.proj_conv.bias",
            "vit.perceive.proj.2.weight": "model.vision_tower.patch_merger.proj_out.weight",
            "vit.perceive.proj.2.bias": "model.vision_tower.patch_merger.proj_out.bias",
            "vit.perceive.norm.weight": "model.vision_tower.patch_merger.before_rms.weight",
            "model.vision_tower.patch_merger.norm.weight": "model.vision_tower.patch_merger.before_rms.weight",
            "model.vision_tower.patch_merger.before_rms.weight": "model.vision_tower.patch_merger.before_rms.weight",
            "vit.layers.0.input_layernorm.weight": "model.vision_tower.layers.0.layer_norm1.weight",
            "model.layers.0.self_attn.learnable_sink_param": "model.language_model.layers.0.self_attn.sinks",
            "model.layers.0.hc_attn_layer.hc_pre.hc_fn": "model.language_model.layers.0.attn_hc.fn",
            "model.layers.0.hc_mlp_layer.hc_pre.hc_fn": "model.language_model.layers.0.ffn_hc.fn",
            "model.hc_head.hc_head_fn": "model.language_model.hc_head.hc_fn",
            "model.language_model.norm.weight": "model.language_model.norm.weight",
            "model.vision_tower.layers.0.layer_norm1.weight": "model.vision_tower.layers.0.layer_norm1.weight",
        }
        for source, target in mappings.items():
            if not with_lm_head:
                target = target.removeprefix("model.")
                if source.startswith(("model.vision_tower.", "model.language_model.")):
                    source = source.removeprefix("model.")
            with self.subTest(source=source):
                actual, _ = rename_source_key(
                    source, renamings, converters, base_model_prefix=model.base_model_prefix, meta_state_dict=state
                )
                self.assertEqual(actual, target)
                self.assertIn(actual, state)

    def test_standalone_text_checkpoint_renaming(self):
        model = HYV4VLTextModel(self.model_tester.get_text_config()).to(torch_device)
        transforms = get_model_conversion_mapping(model)
        renamings = [item for item in transforms if isinstance(item, WeightRenaming)]
        converters = [item for item in transforms if isinstance(item, WeightConverter)]
        mappings = {
            "layers.0.self_attn.learnable_sink_param": "layers.0.self_attn.sinks",
            "layers.0.hc_attn_layer.hc_pre.hc_fn": "layers.0.attn_hc.fn",
            "layers.0.hc_mlp_layer.hc_pre.hc_fn": "layers.0.ffn_hc.fn",
            "hc_head.hc_head_fn": "hc_head.hc_fn",
            "norm.weight": "norm.weight",
        }
        state = model.state_dict()
        for source, target in mappings.items():
            with self.subTest(source=source):
                actual, _ = rename_source_key(
                    source, renamings, converters, base_model_prefix=model.base_model_prefix, meta_state_dict=state
                )
                self.assertEqual(actual, target)
                self.assertIn(actual, state)

    def test_auto_classes_load_legacy_merger_weights(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.assertIsInstance(AutoConfig.for_model("hy_v4_vl"), HYV4VLConfig)
        for auto_class, model_class in (
            (AutoModel, HYV4VLModel),
            (AutoModelForImageTextToText, HYV4VLForConditionalGeneration),
        ):
            with self.subTest(model_class=model_class.__name__):
                model = auto_class.from_config(copy.deepcopy(config)).to(torch_device).eval()
                self.assertIsInstance(model, model_class)
                with torch.no_grad():
                    model.base_model.vision_tower.patch_merger.before_rms.weight.uniform_(0.5, 1.5)
                    for layer in model.base_model.language_model.layers:
                        layer.self_attn.sinks.uniform_(-1, 1)
                    expected = model(**inputs)[0]
                legacy_state = {
                    name.replace("vision_tower.patch_merger.before_rms.", "vision_tower.patch_merger.norm.").replace(
                        ".self_attn.sinks", ".self_attn.learnable_sink_param"
                    ): tensor
                    for name, tensor in model.state_dict().items()
                }
                with tempfile.TemporaryDirectory() as directory:
                    model.save_pretrained(directory, state_dict=legacy_state, save_original_format=False)
                    restored_config = AutoConfig.from_pretrained(directory, local_files_only=True)
                    self.assertIsInstance(restored_config.text_config, HYV4VLTextConfig)
                    self.assertIsInstance(restored_config.vision_config, HYV4VLVisionConfig)
                    self.assertEqual(restored_config.text_config.rope_parameters["mrope_section"], [4, 2, 2])
                    restored, loading_info = auto_class.from_pretrained(
                        directory, local_files_only=True, output_loading_info=True
                    )
                self.assertFalse(loading_info["missing_keys"])
                self.assertFalse(loading_info["unexpected_keys"])
                self.assertFalse(loading_info["mismatched_keys"])
                restored = restored.to(torch_device).eval()
                expected_merger = model.base_model.vision_tower.patch_merger.state_dict()
                actual_merger = restored.base_model.vision_tower.patch_merger.state_dict()
                self.assertEqual(actual_merger.keys(), expected_merger.keys())
                for name, tensor in expected_merger.items():
                    torch.testing.assert_close(actual_merger[name], tensor, rtol=0, atol=0)
                for layer, restored_layer in zip(
                    model.base_model.language_model.layers, restored.base_model.language_model.layers
                ):
                    torch.testing.assert_close(layer.self_attn.sinks, restored_layer.self_attn.sinks, atol=0, rtol=0)
                with torch.no_grad():
                    torch.testing.assert_close(restored(**inputs)[0], expected)

    def test_subconfig_specs_initialize_and_round_trip_without_ocr_side_effects(self):
        config = HYV4VLConfig()
        self.assertIsInstance(config.text_config, HYV4VLTextConfig)
        self.assertIsInstance(config.vision_config, HYV4VLVisionConfig)
        self.assertFalse(config.tie_word_embeddings)
        self.assertFalse(hasattr(config.vision_config, "text_hidden_size"))
        self.assertEqual(config.image_token_id, 120120)
        self.assertEqual(config.video_token_id, 120124)
        text_config = self.model_tester.get_text_config()
        vision_config = self.model_tester.get_vision_config()
        for text, vision in (
            (text_config, vision_config),
            (text_config.to_dict(), vision_config.to_dict()),
            (None, None),
        ):
            with self.subTest(text_type=type(text).__name__):
                restored = HYV4VLConfig(text_config=text, vision_config=vision)
                self.assertIsInstance(restored.text_config, HYV4VLTextConfig)
                self.assertIsInstance(restored.vision_config, HYV4VLVisionConfig)
                self.assertFalse(restored.tie_word_embeddings)
                round_trip = HYV4VLConfig.from_dict(restored.to_dict())
                self.assertEqual(round_trip.to_dict(), restored.to_dict())

    @parameterized.expand(
        [
            ([2, 2, 2, 2], "three.*mrope_section"),
            ([2, 2, 2], "must sum"),
            ([4.5, 2, 2], "non-negative integers"),
            ([4.0, 2, 2], "non-negative integers"),
            ([-1, 5, 4], "non-negative integers"),
            ([True, 5, 2], "non-negative integers"),
            ([2, 4, 2], "interleaved frequency slots"),
            ([3, 2, 3], "interleaved frequency slots"),
        ]
    )
    def test_mrope_configuration_rejects_invalid_partitions(self, sections, message):
        config_dict = self.model_tester.get_text_config().to_dict()
        config_dict["rope_parameters"]["mrope_section"] = sections
        with self.assertRaisesRegex(StrictDataclassClassValidationError, message):
            HYV4VLTextConfig.from_dict(config_dict)

    @parameterized.expand([(0,), (-2,), (15,)])
    def test_mrope_configuration_requires_positive_even_dimension(self, dimension):
        config_dict = self.model_tester.get_text_config().to_dict()
        config_dict["qk_rope_head_dim"] = dimension
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "positive even integer"):
            HYV4VLTextConfig.from_dict(config_dict)

    @parameterized.expand([([8, 0, 0],), ([3, 3, 2],)])
    def test_mrope_configuration_accepts_frequency_slot_boundaries(self, sections):
        config_dict = self.model_tester.get_text_config().to_dict()
        config_dict["rope_parameters"]["mrope_section"] = sections
        config = HYV4VLTextConfig.from_dict(config_dict)
        self.assertEqual(config.rope_parameters["mrope_section"], sections)

    @parameterized.expand([(None, None), ((1, 4, 6), None), (None, (2, 4, 4)), ((1, 4, 6), (2, 4, 4))])
    def test_text_image_video_and_mixed_forward(self, image_grid, video_grid):
        config = self.model_tester.get_config()
        inputs = self.model_tester.prepare_multimodal_inputs(image_grid, video_grid)
        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config)).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**inputs)
            width = (
                config.text_config.vocab_size
                if model_class is HYV4VLForConditionalGeneration
                else config.text_config.hidden_size
            )
            output_class = (
                MoeCausalLMOutputWithPast if model_class is HYV4VLForConditionalGeneration else MoeModelOutputWithPast
            )
            self.assertIsInstance(outputs, output_class)
            self.assertNotIn("rope_deltas", outputs)
            self.assertFalse(hasattr(outputs, "rope_deltas"))
            self.assertEqual(outputs[0].shape, (*inputs["input_ids"].shape, width))
            self.assertTrue(torch.isfinite(outputs[0]).all())

    def test_mixed_modality_left_padded_batch_matches_single(self):
        config = self.model_tester.get_config()
        rows = [
            self.model_tester.prepare_multimodal_inputs(),
            self.model_tester.prepare_multimodal_inputs((1, 4, 6)),
            self.model_tester.prepare_multimodal_inputs(video_grid=(2, 4, 4)),
        ]
        max_length = max(row["input_ids"].shape[-1] for row in rows)
        batched = {}
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            batched[key] = torch.cat(
                [torch.nn.functional.pad(row[key], (max_length - row[key].shape[-1], 0)) for row in rows]
            )
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            batched[key] = torch.cat([row[key] for row in rows if key in row])
        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config)).to(torch_device).eval()
            with torch.no_grad():
                batch_output = model(**batched)[0]
                for index, row in enumerate(rows):
                    single_output = model(**row)[0]
                    torch.testing.assert_close(
                        batch_output[index : index + 1, -row["input_ids"].shape[-1] :],
                        single_output,
                        atol=1e-5,
                        rtol=1e-5,
                    )

    def test_eager_attention_without_sinks(self):
        module = torch.nn.Module()
        module.num_key_value_groups = 1
        query, key, value = [torch.randn(1, 2, 3, 8, device=torch_device) for _ in range(3)]
        expected_weights = (query @ key.transpose(-1, -2) * 8**-0.5).softmax(-1)
        actual, weights = eager_attention_forward(module, query, key, value, None, scaling=8**-0.5)
        torch.testing.assert_close(weights, expected_weights)
        torch.testing.assert_close(actual, (expected_weights @ value).transpose(1, 2))

    def test_mixed_loss_and_visual_gradients(self):
        config = self.model_tester.get_config()
        inputs = self.model_tester.prepare_multimodal_inputs((1, 4, 6), (2, 4, 4))
        model = HYV4VLForConditionalGeneration(config).to(torch_device).train()
        labels = inputs["input_ids"].clone()
        labels[inputs["mm_token_type_ids"] != 0] = -100
        outputs = model(**inputs, labels=labels, use_cache=False)
        expected_loss = torch.nn.functional.cross_entropy(
            outputs.logits[:, :-1].reshape(-1, config.text_config.vocab_size).float(), labels[:, 1:].reshape(-1)
        )
        torch.testing.assert_close(outputs.loss, expected_loss)
        self.assertTrue(torch.isfinite(outputs.loss))
        outputs.loss.backward()
        for parameter in (
            model.lm_head.weight,
            model.model.vision_tower.patch_merger.image_newline,
            model.model.language_model.layers[0].self_attn.sinks,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_three_axis_mrope_with_newline_and_video_time(self):
        model = HYV4VLModel(self.model_tester.get_config()).to(torch_device)
        inputs = self.model_tester.prepare_multimodal_inputs((1, 4, 6), (2, 4, 4))
        positions, deltas = model.get_rope_index(
            inputs["input_ids"],
            inputs["mm_token_type_ids"],
            inputs["image_grid_thw"],
            inputs["video_grid_thw"],
            inputs["attention_mask"],
        )
        self.assertEqual(positions.shape, (3, 1, 25))
        expected = torch.tensor(
            [
                [0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 7, 8, 8, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 14],
                [0, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4, 7, 8, 8, 8, 9, 9, 9, 11, 11, 11, 12, 12, 12, 14],
                [0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 8, 9, 10, 8, 9, 10, 11, 12, 13, 11, 12, 13, 14],
            ],
            device=torch_device,
        ).unsqueeze(1)
        torch.testing.assert_close(positions, expected)
        torch.testing.assert_close(deltas, torch.tensor([[-10]], device=torch_device))

    @parameterized.expand([((1, 4, 6), 2), ((3, 4, 6), 2), ((2, 12, 4), 2), ((2, 1, 3), 1)])
    def test_vision_position_ids_include_all_frames_and_newlines(self, grid, spatial_merge_size):
        model = HYV4VLModel(self.model_tester.get_config()).to(torch_device)
        start = 7
        frames, height, width = grid
        height //= spatial_merge_size
        width //= spatial_merge_size
        positions = model.get_vision_position_ids(
            start_position=start,
            grid_thw=torch.tensor(grid, device=torch_device),
            spatial_merge_size=spatial_merge_size,
            device=torch_device,
        )
        expected = [
            (offset, offset + row, offset + column)
            for frame in range(frames)
            for offset in [start + frame * max(height, width + 1)]
            for row in range(height)
            for column in range(width + 1)
        ]
        self.assertEqual(positions.shape, (3, frames * height * (width + 1)))
        torch.testing.assert_close(positions, torch.tensor(expected, device=torch_device).T)

    def test_mrope_left_padding_does_not_shift_valid_positions(self):
        model = HYV4VLModel(self.model_tester.get_config()).to(torch_device)
        inputs = self.model_tester.prepare_multimodal_inputs((1, 4, 6))
        positions, deltas = model.get_rope_index(
            inputs["input_ids"],
            inputs["mm_token_type_ids"],
            inputs["image_grid_thw"],
            attention_mask=inputs["attention_mask"],
        )
        padded = {
            key: torch.nn.functional.pad(value, (3, 0))
            for key, value in inputs.items()
            if key in ("input_ids", "mm_token_type_ids", "attention_mask")
        }
        padded_positions, padded_deltas = model.get_rope_index(**padded, image_grid_thw=inputs["image_grid_thw"])
        torch.testing.assert_close(padded_positions[:, :, 3:], positions)
        torch.testing.assert_close(padded_deltas, deltas)

    @parameterized.expand([(1, 2, 2, 2), (1, 2, 6, 2), (3, 6, 4, 2), (2, 1, 3, 1)])
    def test_patch_merger_matches_conv_reference(self, frames, height, width, merge_size):
        config = self.model_tester.get_vision_config()
        config.spatial_merge_size = merge_size
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                merger = HYV4VLVisionPatchMerger(config).to(device=torch_device, dtype=dtype)
                torch.nn.init.normal_(merger.image_newline, std=0.1)
                channels = config.hidden_size
                merged_channels = channels * merge_size**2
                shapes = {
                    "before_rms.weight": (channels,),
                    "proj_conv.weight": (merged_channels, channels, merge_size, merge_size),
                    "proj_conv.bias": (merged_channels,),
                    "proj_out.weight": (config.out_hidden_size, merged_channels, 1, 1),
                    "proj_out.bias": (config.out_hidden_size,),
                    "image_newline": (config.out_hidden_size,),
                }
                self.assertEqual({key: tuple(value.shape) for key, value in merger.state_dict().items()}, shapes)
                reference = copy.deepcopy(merger)
                hidden = torch.randn(1, frames * height * width, channels * 2, device=torch_device, dtype=dtype)
                hidden = hidden[..., ::2].detach().requires_grad_(True)
                reference_hidden = hidden.detach().clone().requires_grad_(True)
                expected = reference.before_rms(reference_hidden)
                expected = expected.reshape(-1, height, width, channels).permute(0, 3, 1, 2)
                expected = reference.proj_out(reference.proj_act(reference.proj_conv(expected)))
                expected = expected.permute(0, 2, 3, 1)
                newline = reference.image_newline.view(1, 1, 1, -1).expand(frames, height // merge_size, 1, -1)
                expected = torch.cat((expected, newline), dim=2).reshape(1, -1, config.out_hidden_size)
                actual = merger(hidden, size=(height, width))
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                gradient = torch.randn_like(actual)
                actual.backward(gradient)
                expected.backward(gradient)
                torch.testing.assert_close(hidden.grad, reference_hidden.grad)
                for name, parameter in merger.named_parameters():
                    torch.testing.assert_close(parameter.grad, reference.get_parameter(name).grad)

    def test_packed_features_have_one_newline_per_merged_row(self):
        config = self.model_tester.get_config()
        model = HYV4VLModel(config).to(torch_device).eval()
        inputs = self.model_tester.prepare_multimodal_inputs((1, 4, 6), (2, 4, 4))
        for pixel_key, grid_key, get_features in (
            ("pixel_values", "image_grid_thw", model.get_image_features),
            ("pixel_values_videos", "video_grid_thw", model.get_video_features),
        ):
            with torch.no_grad():
                features = get_features(inputs[pixel_key], inputs[grid_key], return_dict=True)
            frames, height, width = inputs[grid_key][0].tolist()
            merged_height, merged_width = height // 2, width // 2
            self.assertEqual(features.last_hidden_state.shape, (1, frames * height * width, 32))
            (pooled,) = features.pooler_output
            self.assertEqual(pooled.shape, (frames * merged_height * (merged_width + 1), 48))
            rows = pooled.reshape(frames, merged_height, merged_width + 1, 48)
            expected = model.vision_tower.patch_merger.image_newline.expand(frames, merged_height, -1)
            torch.testing.assert_close(rows[:, :, -1], expected, atol=0, rtol=0)

    def test_mismatching_num_image_tokens(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config)).to(torch_device).eval()
            with torch.no_grad():
                model(**inputs)
                missing_image = dict(inputs)
                missing_image["pixel_values"] = inputs["pixel_values"][: self.model_tester.num_image_patches]
                missing_image["image_grid_thw"] = inputs["image_grid_thw"][:1]
                with self.assertRaisesRegex(ValueError, "features and (image|video) tokens do not match"):
                    model(**missing_image)
                missing_token = dict(inputs)
                missing_token["input_ids"] = inputs["input_ids"].clone()
                missing_token["input_ids"][0, 0] = 10
                with self.assertRaisesRegex(ValueError, "features and (image|video) tokens do not match"):
                    model(**missing_token)

    def test_mixed_inputs_embeds_matches_explicit_mrope(self):
        config = self.model_tester.get_config()
        inputs = self.model_tester.prepare_multimodal_inputs((1, 4, 6), (2, 4, 4))
        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config)).to(torch_device).eval()
            positions, deltas = model.base_model.get_rope_index(**inputs)
            embeds_inputs = dict(inputs)
            input_ids = embeds_inputs.pop("input_ids")
            with torch.no_grad():
                embeds_inputs["inputs_embeds"] = model.get_input_embeddings()(input_ids)
                self.assertIsNone(model.base_model.rope_deltas)
                actual = model(**embeds_inputs)[0]
                torch.testing.assert_close(model.base_model.rope_deltas, deltas)
                expected = model(**embeds_inputs, position_ids=positions)[0]
                expected_ids = model(**inputs)[0]
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual, expected_ids)

    def test_multimodal_inputs_embeds_requires_types_or_explicit_positions(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config)).to(torch_device).eval()
            positions, _ = model.base_model.get_rope_index(**inputs)
            embeds_inputs = dict(inputs)
            input_ids = embeds_inputs.pop("input_ids")
            embeds_inputs.pop("mm_token_type_ids")
            with torch.no_grad():
                embeds_inputs["inputs_embeds"] = model.get_input_embeddings()(input_ids)
                with self.assertRaisesRegex(ValueError, "mm_token_type_ids or explicit position_ids"):
                    model(**embeds_inputs)
                actual = model(**embeds_inputs, position_ids=positions)[0]
                expected = model(**inputs)[0]
            torch.testing.assert_close(actual, expected)

    def test_rotary_embeddings_are_built_once_per_forward(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = HYV4VLForConditionalGeneration(config).to(torch_device).eval()
        rotary = model.model.language_model.rotary_emb
        with patch.object(rotary, "forward", wraps=rotary.forward) as rotary_forward, torch.no_grad():
            model(**inputs)
        self.assertEqual(rotary_forward.call_count, 1)

    @parameterized.expand(
        [(prefill, decode, padding) for prefill in (False, True) for decode in (False, True) for padding in (0, 3)]
    )
    def test_cached_decode_matches_full_multimodal_forward(self, explicit_prefill, explicit_positions, left_padding):
        config = self.model_tester.get_config()
        inputs = self.model_tester.prepare_multimodal_inputs((1, 4, 6), (2, 4, 4))
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            inputs[key] = torch.nn.functional.pad(inputs[key], (left_padding, 0))
        next_tokens = torch.tensor([[14, 15]], device=torch_device)
        extended = dict(inputs)
        extended["input_ids"] = torch.cat([inputs["input_ids"], next_tokens], dim=-1)
        extended["attention_mask"] = torch.nn.functional.pad(inputs["attention_mask"], (0, 2), value=1)
        extended["mm_token_type_ids"] = torch.nn.functional.pad(inputs["mm_token_type_ids"], (0, 2))
        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config)).to(torch_device).eval()
            base_model = model.base_model
            expected_positions, _ = base_model.get_rope_index(**extended)
            with torch.no_grad():
                full_output = model(**extended, use_cache=False)[0][:, -2:]
                base_model.rope_deltas = None
                prefill_kwargs = {"position_ids": expected_positions[:, :, :-2]} if explicit_prefill else {}
                prefill = model(**inputs, use_cache=True, **prefill_kwargs)
                self.assertEqual(prefill.past_key_values.get_seq_length(), inputs["input_ids"].shape[-1])
                decode_kwargs = {}
                if explicit_positions:
                    past_length = prefill.past_key_values.get_seq_length() - left_padding
                    positions = torch.arange(past_length, past_length + 2, device=torch_device)[None, None]
                    decode_kwargs["position_ids"] = (positions + base_model.rope_deltas).expand(3, -1, -1)
                with patch.object(
                    base_model.language_model, "forward", wraps=base_model.language_model.forward
                ) as forward:
                    decode = model(
                        input_ids=next_tokens,
                        attention_mask=extended["attention_mask"],
                        past_key_values=prefill.past_key_values,
                        use_cache=True,
                        **decode_kwargs,
                    )
                decode_positions = forward.call_args.kwargs["position_ids"]
                self.assertEqual(decode_positions.shape, (3, 1, next_tokens.shape[-1]))
                torch.testing.assert_close(decode_positions, expected_positions[:, :, -2:])
            self.assertEqual(decode.past_key_values.get_seq_length(), extended["input_ids"].shape[-1])
            torch.testing.assert_close(decode[0], full_output, atol=1e-5, rtol=1e-5)

    @parameterized.expand([(False, False), (False, True), (True, False), (True, True)])
    def test_fresh_text_request_resets_multimodal_rope_deltas(self, use_inputs_embeds, use_empty_cache):
        config = self.model_tester.get_config()
        multimodal = self.model_tester.prepare_multimodal_inputs((1, 4, 6), (2, 4, 4))
        input_ids = torch.tensor([[10, 11, 12], [0, 13, 14]], device=torch_device)
        attention_mask = (input_ids != config.text_config.pad_token_id).long()
        expected_positions = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
        next_tokens = torch.tensor([[15, 16], [17, 18]], device=torch_device)
        extended_mask = torch.nn.functional.pad(attention_mask, (0, 2), value=1)
        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config)).to(torch_device).eval()
            base_model = model.base_model
            with torch.no_grad():
                expected = model(
                    input_ids=torch.cat([input_ids, next_tokens], dim=-1),
                    attention_mask=extended_mask,
                    use_cache=False,
                )[0][:, -2:]
                model(**multimodal)
                self.assertTrue((base_model.rope_deltas != 0).any())
                text_inputs = (
                    {"inputs_embeds": model.get_input_embeddings()(input_ids)}
                    if use_inputs_embeds
                    else {"input_ids": input_ids}
                )
                cache = DynamicCache(config=config.text_config) if use_empty_cache else None
                with patch.object(
                    base_model.language_model, "forward", wraps=base_model.language_model.forward
                ) as forward:
                    prefill = model(
                        **text_inputs, attention_mask=attention_mask, past_key_values=cache, use_cache=True
                    )
                self.assertIsNone(base_model.rope_deltas)
                torch.testing.assert_close(
                    forward.call_args.kwargs["position_ids"], expected_positions.unsqueeze(0).expand(3, -1, -1)
                )
                decode = model(
                    input_ids=next_tokens,
                    attention_mask=extended_mask,
                    past_key_values=prefill.past_key_values,
                    use_cache=True,
                )
            torch.testing.assert_close(decode[0], expected, atol=1e-5, rtol=1e-5)

    def test_generation_injects_image_and_video_only_at_prefill(self):
        config = self.model_tester.get_config()
        model = HYV4VLForConditionalGeneration(config).to(torch_device).eval()
        inputs = self.model_tester.prepare_multimodal_inputs((1, 4, 6), (2, 4, 4))
        with (
            patch.object(
                model.model, "get_image_features", autospec=True, side_effect=model.model.get_image_features
            ) as image_features,
            patch.object(
                model.model, "get_video_features", autospec=True, side_effect=model.model.get_video_features
            ) as video_features,
            torch.no_grad(),
        ):
            generated = model.generate(**inputs, max_new_tokens=3, min_new_tokens=3, do_sample=False)
        self.assertEqual(image_features.call_count, 1)
        self.assertEqual(video_features.call_count, 1)
        self.assertEqual(generated.shape, (1, inputs["input_ids"].shape[1] + 3))
        torch.testing.assert_close(generated[:, : inputs["input_ids"].shape[1]], inputs["input_ids"])
        with torch.no_grad():
            without_cache = model.generate(
                **inputs, max_new_tokens=3, min_new_tokens=3, do_sample=False, use_cache=False
            )
        torch.testing.assert_close(generated, without_cache)

    def test_prepare_inputs_for_generation_drops_both_pixel_streams(self):
        config = self.model_tester.get_config()
        model = HYV4VLForConditionalGeneration(config).to(torch_device).eval()
        inputs = self.model_tester.prepare_multimodal_inputs((1, 4, 6), (2, 4, 4))
        positions, _ = model.model.get_rope_index(
            inputs["input_ids"], inputs["mm_token_type_ids"], inputs["image_grid_thw"], inputs["video_grid_thw"]
        )
        prefill = model.prepare_inputs_for_generation(**inputs, position_ids=positions, is_first_iteration=True)
        for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
            self.assertIs(prefill[key], inputs[key])
        decode = model.prepare_inputs_for_generation(
            **inputs, position_ids=positions, is_first_iteration=False, use_cache=True, next_sequence_length=1
        )
        self.assertNotIn("pixel_values", decode)
        self.assertNotIn("pixel_values_videos", decode)
        self.assertEqual(decode["position_ids"].shape, (3, 1, 1))

    def _prepare_beam_batch(self, modalities):
        rows = []
        token_keys = ("input_ids", "attention_mask", "mm_token_type_ids")
        visual_keys = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
        for sample in modalities:
            segments = [
                self.model_tester.prepare_multimodal_inputs(**{"image_grid" if modality == 1 else "video_grid": grid})
                for modality, grid in sample
            ] or [self.model_tester.prepare_multimodal_inputs()]
            row = {key: torch.cat([segment[key] for segment in segments], dim=1) for key in token_keys}
            for key in visual_keys:
                if any(key in segment for segment in segments):
                    row[key] = torch.cat([segment[key] for segment in segments if key in segment])
            rows.append(row)
        length = max(row["input_ids"].shape[-1] for row in rows)
        inputs = {
            key: torch.cat([torch.nn.functional.pad(row[key], (length - row[key].shape[-1], 0)) for row in rows])
            for key in token_keys
        }
        for key in visual_keys:
            if any(key in row for row in rows):
                inputs[key] = torch.cat([row[key] for row in rows if key in row])
        return inputs, rows

    @parameterized.expand([(False, 1), (False, 2), (False, 3), (True, 1), (True, 2), (True, 3)])
    def test_beam_expansion_preserves_multimodal_sample_order(self, use_inputs_embeds, expand_size):
        config = self.model_tester.get_config()
        model = HYV4VLForConditionalGeneration(config).to(torch_device).eval()
        inputs, rows = self._prepare_beam_batch(
            [
                [(1, (1, 4, 6)), (2, (2, 2, 4)), (1, (1, 2, 4))],
                [],
                [(2, (1, 4, 2)), (1, (1, 6, 2)), (1, (1, 2, 6)), (2, (2, 2, 2))],
                [(2, (2, 4, 2))],
            ]
        )
        batch_size, length = inputs["input_ids"].shape
        positions = torch.arange(3 * batch_size * length, device=torch_device).reshape(3, batch_size, length)
        inputs["position_ids"] = positions
        inputs["past_key_values"] = DynamicCache(config=config.text_config)
        rope_deltas = torch.arange(batch_size, device=torch_device).unsqueeze(1)
        model.model.rope_deltas = rope_deltas
        if use_inputs_embeds:
            with torch.no_grad():
                inputs["inputs_embeds"] = model.get_input_embeddings()(inputs.pop("input_ids"))
        with torch.no_grad():
            inputs = model._prepare_multimodal_encoder_kwargs_for_generation(inputs)
        expected_features = {}
        for modality in ("image", "video"):
            features = inputs["mm_encoder_outputs"][modality].pooler_output
            expected_features[modality] = []
            start = 0
            for row in rows:
                count = len(row.get(f"{modality}_grid_thw", []))
                for _ in range(expand_size):
                    expected_features[modality].extend(features[start : start + count])
                start += count
        expanded_ids, expanded = model._expand_inputs_for_generation(expand_size=expand_size, **inputs)
        self.assertIs(expanded["past_key_values"], inputs["past_key_values"])
        self.assertEqual(expanded["past_key_values"].get_seq_length(), 0)
        if use_inputs_embeds:
            self.assertIsNone(expanded_ids)
            torch.testing.assert_close(
                expanded["inputs_embeds"],
                inputs["inputs_embeds"].repeat_interleave(expand_size, dim=0),
                rtol=0,
                atol=0,
            )
        else:
            torch.testing.assert_close(expanded_ids, inputs["input_ids"].repeat_interleave(expand_size, dim=0))
        for key in ("attention_mask", "mm_token_type_ids"):
            torch.testing.assert_close(expanded[key], inputs[key].repeat_interleave(expand_size, dim=0))
        self.assertEqual(expanded["position_ids"].shape, (3, batch_size * expand_size, length))
        torch.testing.assert_close(expanded["position_ids"], positions.repeat_interleave(expand_size, dim=1))
        # Qwen2-VL leaves the cached deltas at the original batch size and broadcasts them lazily.
        self.assertIs(model.model.rope_deltas, rope_deltas)
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            self.assertNotIn(key, expanded)
        for modality, expected in expected_features.items():
            actual = expanded["mm_encoder_outputs"][modality].pooler_output
            self.assertEqual(len(actual), len(expected))
            for actual_feature, expected_feature in zip(actual, expected):
                torch.testing.assert_close(actual_feature, expected_feature, rtol=0, atol=0)
        if expand_size == 1:
            for key, value in inputs.items():
                self.assertIs(expanded_ids if key == "input_ids" else expanded[key], value)

    @parameterized.expand([(None,), (2,), (3,)])
    def test_beam_expansion_text_only(self, position_dims):
        model = HYV4VLForConditionalGeneration(self.model_tester.get_config()).to(torch_device).eval()
        input_ids = torch.tensor([[10, 11, 12], [13, 14, 15]], device=torch_device)
        inputs = {"attention_mask": torch.ones_like(input_ids)}
        if position_dims is not None:
            positions = torch.arange(6, device=torch_device).reshape(2, 3)
            if position_dims == 3:
                positions = torch.stack([positions, positions + 10, positions + 20])
            inputs["position_ids"] = positions
        expanded_ids, expanded = model._expand_inputs_for_generation(expand_size=2, input_ids=input_ids, **inputs)
        torch.testing.assert_close(expanded_ids, input_ids.repeat_interleave(2, dim=0))
        torch.testing.assert_close(expanded["attention_mask"], inputs["attention_mask"].repeat_interleave(2, dim=0))
        self.assertNotIn("pixel_values", expanded)
        self.assertNotIn("image_grid_thw", expanded)
        self.assertIsNone(model.model.rope_deltas)
        if position_dims is None:
            self.assertNotIn("position_ids", expanded)
        else:
            torch.testing.assert_close(
                expanded["position_ids"], positions.repeat_interleave(2, dim=1 if position_dims == 3 else 0)
            )

    @parameterized.expand([(False,), (True,)])
    def test_greedy_and_beam_generation_multimodal_batch(self, use_inputs_embeds):
        config = self.model_tester.get_config()
        model = HYV4VLForConditionalGeneration(config).to(torch_device).eval()
        inputs, _ = self._prepare_beam_batch(
            [[(1, (1, 4, 6)), (1, (1, 2, 4)), (2, (2, 2, 4))], [], [(2, (1, 4, 2)), (1, (1, 2, 6))]]
        )
        input_ids = inputs["input_ids"]
        if use_inputs_embeds:
            positions, _ = model.model.get_rope_index(
                input_ids,
                inputs["mm_token_type_ids"],
                inputs["image_grid_thw"],
                inputs["video_grid_thw"],
                inputs["attention_mask"],
            )
            inputs["position_ids"] = positions
            with torch.no_grad():
                inputs["inputs_embeds"] = model.get_input_embeddings()(inputs.pop("input_ids"))
        prompt_length = 0 if use_inputs_embeds else input_ids.shape[-1]
        for num_beams in (1, 2):
            with self.subTest(num_beams=num_beams), torch.no_grad():
                generated = model.generate(
                    **inputs,
                    num_beams=num_beams,
                    do_sample=False,
                    min_new_tokens=3,
                    max_new_tokens=3,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            self.assertEqual(generated.sequences.shape, (input_ids.shape[0], prompt_length + 3))
            self.assertEqual(generated.sequences.dtype, torch.long)
            self.assertEqual(len(generated.scores), 3)
            for scores in generated.scores:
                self.assertEqual(scores.shape, (input_ids.shape[0] * num_beams, config.text_config.vocab_size))
                self.assertTrue(torch.isfinite(scores).any(dim=-1).all())
            if not use_inputs_embeds:
                torch.testing.assert_close(generated.sequences[:, :prompt_length], input_ids)

    @unittest.skip("Packed vision attention returns no attention matrix; text attention is tested separately.")
    def test_get_image_features_attentions(self):
        pass

    @unittest.skip("Packed vision attention returns no attention matrix; text attention is tested separately.")
    def test_get_video_features_attentions(self):
        pass


@require_torch
class HYV4VLComponentTest(unittest.TestCase):
    all_model_classes = (HYV4VLTextModel, HYV4VLVisionTransformer) if is_torch_available() else ()

    def component_inputs(self, model_class):
        tester = HYV4VLVisionText2TextModelTester(self)
        config = tester.get_config()
        if model_class is HYV4VLTextModel:
            return (
                config.text_config,
                {
                    "input_ids": torch.tensor([[10, 11, 12, 13]], device=torch_device),
                    "attention_mask": torch.ones(1, 4, dtype=torch.long, device=torch_device),
                    "use_cache": False,
                },
                (1, 4, config.text_config.hidden_size),
            )
        inputs = tester.prepare_multimodal_inputs((1, 4, 6))
        return (
            config.vision_config,
            {
                "pixel_values": inputs["pixel_values"],
                "grid_thw": inputs["image_grid_thw"],
            },
            (1, 24, config.vision_config.hidden_size),
        )

    def test_component_forward_and_save_load(self):
        for model_class in self.all_model_classes:
            with self.subTest(model_class=model_class.__name__):
                config, inputs, expected_shape = self.component_inputs(model_class)
                model = model_class(config).to(torch_device).eval()
                with torch.no_grad():
                    if model_class is HYV4VLTextModel:
                        for layer in model.layers:
                            layer.self_attn.sinks.uniform_(-1, 1)
                    expected = model(**inputs).last_hidden_state
                self.assertEqual(expected.shape, expected_shape)
                self.assertTrue(torch.isfinite(expected).all())
                state = {
                    name.replace(".self_attn.sinks", ".self_attn.learnable_sink_param"): tensor
                    for name, tensor in model.state_dict().items()
                }
                with tempfile.TemporaryDirectory() as directory:
                    model.save_pretrained(directory, state_dict=state, save_original_format=False)
                    restored, loading = model_class.from_pretrained(
                        directory, local_files_only=True, output_loading_info=True
                    )
                self.assertFalse(loading["missing_keys"])
                self.assertFalse(loading["unexpected_keys"])
                restored.to(torch_device).eval()
                if model_class is HYV4VLTextModel:
                    for layer, restored_layer in zip(model.layers, restored.layers):
                        torch.testing.assert_close(
                            layer.self_attn.sinks, restored_layer.self_attn.sinks, atol=0, rtol=0
                        )
                with torch.no_grad():
                    actual = restored(**inputs).last_hidden_state
                torch.testing.assert_close(actual, expected)

    def test_rotary_matches_interleaved_checkpoint_channels(self):
        torch.manual_seed(7)
        for dtype in (torch.float32, torch.bfloat16):
            for unsqueeze_dim in (1, 2):
                with self.subTest(dtype=dtype, unsqueeze_dim=unsqueeze_dim):
                    shape = (1, 2, 3, 8) if unsqueeze_dim == 1 else (1, 3, 2, 8)
                    q, k = [torch.randn(shape, device=torch_device, dtype=dtype) for _ in range(2)]
                    angles = (
                        torch.tensor([0.0, 1.0, 31.0], device=torch_device)[:, None]
                        * torch.tensor([1.0, 0.1, 0.01, 0.001], device=torch_device)[None, :]
                    )
                    cos = torch.cat([angles.cos(), angles.cos()], -1).unsqueeze(0).to(dtype)
                    sin = torch.cat([angles.sin(), angles.sin()], -1).unsqueeze(0).to(dtype)
                    actual = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
                    half_cos = cos[..., :4].float().unsqueeze(unsqueeze_dim)
                    half_sin = sin[..., :4].float().unsqueeze(unsqueeze_dim)
                    for value, result in zip((q, k), actual):
                        real, imaginary = value[..., ::2].float(), value[..., 1::2].float()
                        # Each adjacent (real, imag) pair is rotated in place, in fp32
                        expected = (
                            torch.stack(
                                [real * half_cos - imaginary * half_sin, imaginary * half_cos + real * half_sin], -1
                            )
                            .flatten(-2)
                            .to(dtype)
                        )
                        torch.testing.assert_close(result, expected, atol=0, rtol=0)

    def test_mla_latent_norms_use_checkpoint_epsilon(self):
        config = HYV4VLVisionText2TextModelTester(self).get_config().text_config
        config.rms_norm_eps = 1e-5
        model = HYV4VLTextModel(config).to(torch_device)
        for layer in model.layers:
            self.assertEqual(layer.self_attn.q_a_layernorm.variance_epsilon, config.rms_norm_eps)
            self.assertEqual(layer.self_attn.kv_a_layernorm.variance_epsilon, config.rms_norm_eps)

    def test_bfloat16_loading_preserves_float32_lm_head(self):
        config = HYV4VLVisionText2TextModelTester(self).get_config()
        model = HYV4VLForConditionalGeneration(config).to(torch_device).eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = (
                HYV4VLForConditionalGeneration.from_pretrained(directory, dtype=torch.bfloat16, local_files_only=True)
                .to(torch_device)
                .eval()
            )
        self.assertEqual(loaded.lm_head.weight.dtype, torch.float32)
        input_ids = torch.tensor([[10, 11, 12]], device=torch_device)
        with torch.no_grad():
            logits = loaded(input_ids=input_ids, use_cache=False).logits
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(logits).all())

    def test_component_backward(self):
        for model_class in self.all_model_classes:
            with self.subTest(model_class=model_class.__name__):
                config, inputs, _ = self.component_inputs(model_class)
                model = model_class(config).to(torch_device).train()
                outputs = model(**inputs)
                result = outputs.pooler_output if model_class is HYV4VLVisionTransformer else outputs.last_hidden_state
                result.square().mean().backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
