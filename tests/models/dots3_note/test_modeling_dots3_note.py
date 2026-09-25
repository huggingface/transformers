# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for the PyTorch Dots 3 Note Preview model."""

import json
import tempfile
import unittest
from unittest.mock import patch

from parameterized import parameterized
from safetensors import safe_open

from transformers import (
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    Dots3NoteConfig,
    Dots3NoteForConditionalGeneration,
    Dots3NoteModel,
    Dots3NoteTextConfig,
    is_torch_available,
)
from transformers.cache_utils import (
    StaticCache,
    StaticIndexedLayer,
    StaticSlidingWindowLayer,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.testing_utils import require_torch

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION,
    _test_eager_matches_batched_and_grouped_inference,
)


if is_torch_available():
    import torch

    from transformers.models.dots3_note.modeling_dots3_note import (
        Dots3NoteAudioModel,
        Dots3NoteForCausalLM,
        Dots3NoteTextAttention,
        Dots3NoteTextModel,
        Dots3NoteVisionModel,
        eager_attention_forward,
    )


def get_tiny_config(use_dsa=False):
    vision_config = {
        "embed_dim": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "moe_intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_channels": 3,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "pyramid_num_routed": [-1, 2],
        "capacity_factor": 2,
        "adapter_in_dim": 32,
        "adapter_out_dim": 32,
        "adapter_merge_size": 2,
    }
    audio_config = {
        "whisper_config": {
            "d_model": 32,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 64,
            "encoder_layers": 2,
            "num_mel_bins": 8,
            "max_source_positions": 32,
            "activation_function": "swiglu",
        },
        "feature_size": 8,
        "hop_length": 4,
        "downsample_hidden_size": 4,
        "adapter_input_size": 32,
        "adapter_output_size": 32,
    }
    config = Dots3NoteConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        head_dim=16,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window_size=4,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=4,
        swa_q_lora_rank=16,
        swa_kv_lora_rank=16,
        swa_head_dim=16,
        swa_qk_nope_head_dim=8,
        swa_qk_rope_head_dim=8,
        swa_v_head_dim=8,
        index_n_heads=2,
        index_head_dim=128,
        index_topk=4,
        use_dsa=use_dsa,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_experts_intermediate_size=16,
        first_k_dense_replace=1,
        image_token_id=120,
        video_token_id=122,
        audio_token_id=121,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        vision_config=vision_config,
        audio_config=audio_config,
    )
    config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    return config


class Dots3NoteTextModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Dots3NoteTextModel
        config_class = Dots3NoteTextConfig
        causal_lm_class = Dots3NoteForCausalLM

    def __init__(self, parent):
        super().__init__(
            parent=parent,
            batch_size=2,
            seq_length=7,
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=64,
            max_position_embeddings=128,
        )
        # Initial support is inference-focused, so common coverage targets forward, cache, and generation.
        self.is_training = False

    def get_config(self):
        config = get_tiny_config().text_config
        # Common cache assertions assume uniform KV widths and no sliding-window eviction.
        # The dedicated cache/batching tests exercise the released asymmetric MLA layout.
        config.v_head_dim = config.head_dim
        config.per_layer_config["sliding_attention"].v_head_dim = config.head_dim
        config.sliding_window = config.max_position_embeddings
        return config


@require_torch
class Dots3NoteTextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Dots3NoteTextModelTester
    all_model_classes = (Dots3NoteForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = {"text-generation": Dots3NoteForCausalLM} if is_torch_available() else {}
    _is_stateful = True

    def setUp(self):
        super().setUp()
        # DSA and SWA have different head counts, so no global num_attention_heads exists.
        self.config_tester.common_properties = ["hidden_size", "num_hidden_layers"]

    @unittest.skip(reason="Initial Dots 3 Note Preview support is inference-only.")
    def test_gradient_checkpointing_enable_disable(self):
        pass

    @unittest.skip(reason="The model-specific DSA/SWA cache cannot be reconstructed by torch.nn.DataParallel.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_batched_and_grouped_inference(self, name, dtype):
        # SonicMoE and DeepGEMM are optional integrations. This model's required expert
        # implementations are eager, batched_mm, and grouped_mm.
        with (
            patch("tests.test_modeling_common.is_sonicmoe_loadable", return_value=False),
            patch("tests.test_modeling_common.is_deepgemm_loadable", return_value=False),
        ):
            _test_eager_matches_batched_and_grouped_inference(self, name, dtype)


@require_torch
class Dots3NoteModelTest(unittest.TestCase):
    def test_vision_head_override_roundtrip(self):
        config = get_tiny_config().vision_config
        config_class = type(config)
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            config = config_class.from_pretrained(directory, num_attention_heads=8)
            self.assertEqual(Dots3NoteVisionModel(config).blocks[0].attn.num_heads, 8)
            self.assertNotIn("num_heads", config.to_dict())
            config.save_pretrained(directory)
            self.assertEqual(config_class.from_pretrained(directory).num_attention_heads, 8)
        legacy = config.to_dict()
        legacy.pop("num_attention_heads")
        legacy["num_heads"] = 4
        converted = config_class(**legacy)
        self.assertEqual(converted.num_attention_heads, 4)
        self.assertNotIn("num_heads", converted.to_dict())

    def test_heterogeneous_attention_dimensions(self):
        config = get_tiny_config(use_dsa=True).text_config
        values = config.to_dict()
        values["per_layer_config"]["1"] = {
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "q_lora_rank": 8,
            "kv_lora_rank": 8,
        }
        config = Dots3NoteTextConfig(**values)
        model = Dots3NoteForCausalLM(config).eval()
        attention = model.model.layers[1].self_attn
        self.assertEqual(attention.num_heads, 2)
        self.assertEqual(attention.q_a_proj.out_features, 8)
        self.assertEqual(attention.kv_lora_rank, 8)
        with torch.no_grad():
            output = model(torch.tensor([[1, 7, 2]]), use_cache=True)
        self.assertEqual(output.logits.shape, (1, 3, config.vocab_size))

    def test_legacy_text_config_loading(self):
        config = get_tiny_config()
        legacy = config.to_dict()
        legacy.update(legacy.pop("text_config"))
        legacy["model_type"] = "dots3_note"
        quantization_config = {"quant_method": "fp8", "weight_block_size": [128, 128]}
        quantized = Dots3NoteConfig(**(legacy | {"quantization_config": quantization_config}))
        self.assertEqual(quantized.quantization_config, quantization_config)
        self.assertEqual(quantized.text_config.model_type, "dots3_note_text")
        model = AutoModelForCausalLM.from_config(Dots3NoteConfig(**legacy)).eval()
        input_ids = torch.tensor([[1, 7, 2]])
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            Dots3NoteConfig(**legacy).save_pretrained(directory)
            loaded = AutoModelForCausalLM.from_pretrained(directory).eval()
            with torch.no_grad():
                torch.testing.assert_close(model(input_ids).logits, loaded(input_ids).logits)

    all_model_classes = (
        (
            Dots3NoteAudioModel,
            Dots3NoteForCausalLM,
            Dots3NoteForConditionalGeneration,
            Dots3NoteModel,
            Dots3NoteVisionModel,
        )
        if is_torch_available()
        else ()
    )

    @parameterized.expand([True, False])
    def test_dsa_shared_mask_reaches_indexer_and_attention(self, boolean_mask):
        config = get_tiny_config(use_dsa=True).text_config
        config._attn_implementation = "eager"
        attention = Dots3NoteTextAttention(config, layer_idx=0).eval()
        allowed = torch.ones(1, 1, 5, 5, dtype=torch.bool).tril()
        allowed[..., :2] = False
        mask = allowed if boolean_mask else torch.zeros_like(allowed, dtype=torch.float).masked_fill(~allowed, -10000)
        indices = torch.tensor([[[0, 3, 4]] * 5], dtype=torch.int32)
        received = []

        def attention_spy(module, query, key, value, attention_mask, **kwargs):
            received.append(attention_mask)
            return eager_attention_forward(module, query, key, value, attention_mask, **kwargs)

        hidden = torch.randn(1, 5, config.hidden_size)
        cos = torch.ones(1, 5, config.qk_rope_head_dim)
        position_ids = torch.tensor([[0, 0, 0, 1, 2]])
        with (
            patch.object(attention.indexer, "forward", return_value=indices) as indexer,
            patch.object(ALL_ATTENTION_FUNCTIONS, "get_interface", return_value=attention_spy),
            torch.no_grad(),
        ):
            attention(hidden, (cos, torch.zeros_like(cos)), mask, position_ids=position_ids)
        torch.testing.assert_close(indexer.call_args.args[3], mask[:, 0])
        self.assertIs(indexer.call_args.args[4], position_ids)
        torch.testing.assert_close(received[0][0, 0, -1] == 0, torch.tensor([False, False, False, True, True]))
        with self.assertRaisesRegex(ValueError, "per-head"):
            attention(hidden, (cos, torch.zeros_like(cos)), mask.expand(1, 2, 5, 5))

    def test_dsa_chunked_prefill_matches_one_shot(self):
        config = get_tiny_config(use_dsa=True).text_config
        # Keep every key in this cache-parity test. Randomly initialized tiny indexers can produce exact
        # score ties, for which `topk` is allowed to pick different tied keys as the cached key width grows.
        # Sparse top-k math and sparse cached generation are covered independently below.
        config.index_topk = config.max_position_embeddings
        model = Dots3NoteForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 7, 11, 9, 6, 13, 12, 5, 8, 10, 2]])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            expected = model(input_ids, attention_mask=attention_mask, use_cache=False).logits
            past_key_values = None
            chunks = []
            for start in range(0, input_ids.shape[1], 3):
                stop = min(start + 3, input_ids.shape[1])
                outputs = model(
                    input_ids[:, start:stop],
                    attention_mask=attention_mask[:, :stop],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                chunks.append(outputs.logits)

        actual = torch.cat(chunks, dim=1)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_dsa_left_padded_batch_cache_matches_unpadded_decode(self):
        torch.manual_seed(0)
        config = get_tiny_config(use_dsa=True).text_config
        config.layer_types = ["deepseek_sparse_attention"] * config.num_hidden_layers
        # Select every key so this test targets physical cache/padding coordinates rather than
        # the numerical discontinuity of hard top-k selection on a randomly initialized indexer.
        config.index_topk = config.max_position_embeddings
        model = Dots3NoteForCausalLM(config).eval()
        model.set_experts_implementation("eager")

        sequences = [
            torch.tensor([[1, 7, 11, 9, 2]]),
            torch.tensor([[1, 8, 6, 13, 12, 5, 10, 2]]),
        ]
        next_tokens = torch.tensor([[14], [15]])
        max_length = max(sequence.shape[1] for sequence in sequences)
        input_ids = torch.cat(
            [torch.nn.functional.pad(sequence, (max_length - sequence.shape[1], 0)) for sequence in sequences]
        )
        attention_mask = input_ids.ne(config.pad_token_id).long()
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask.eq(0), 0)

        with torch.no_grad():
            prefill = model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
            batched = model(
                next_tokens,
                attention_mask=torch.cat((attention_mask, torch.ones_like(next_tokens)), dim=-1),
                position_ids=attention_mask.sum(-1, keepdim=True),
                past_key_values=prefill.past_key_values,
                use_cache=True,
            ).logits[:, -1]

            unpadded = []
            for sequence, next_token in zip(sequences, next_tokens):
                sequence_mask = torch.ones_like(sequence)
                sequence_prefill = model(
                    sequence,
                    attention_mask=sequence_mask,
                    position_ids=torch.arange(sequence.shape[1]).unsqueeze(0),
                    use_cache=True,
                )
                unpadded.append(
                    model(
                        next_token.view(1, 1),
                        attention_mask=torch.ones(1, sequence.shape[1] + 1, dtype=torch.long),
                        position_ids=torch.tensor([[sequence.shape[1]]]),
                        past_key_values=sequence_prefill.past_key_values,
                        use_cache=True,
                    ).logits[:, -1]
                )

        torch.testing.assert_close(batched, torch.cat(unpadded), rtol=1e-4, atol=1e-4)

    def test_dsa_static_cache_matches_dynamic_cache(self):
        torch.manual_seed(0)
        config = get_tiny_config(use_dsa=True).text_config
        model = Dots3NoteForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 7, 11, 9]])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            dynamic = model(input_ids, attention_mask=attention_mask, use_cache=True)
            next_token = dynamic.logits[:, -1:].argmax(dim=-1)
            decode_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=-1)
            dynamic_decode = model(
                next_token,
                attention_mask=decode_mask,
                past_key_values=dynamic.past_key_values,
                use_cache=True,
            )

            static_cache = StaticCache(config=config, max_cache_len=16)
            static = model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=static_cache,
                use_cache=True,
            )
            static_decode = model(
                next_token,
                attention_mask=decode_mask,
                past_key_values=static.past_key_values,
                use_cache=True,
            )

        self.assertIsInstance(static_cache.layers[0], StaticIndexedLayer)
        self.assertIsInstance(static_cache.layers[1], StaticSlidingWindowLayer)
        torch.testing.assert_close(static_decode.logits, dynamic_decode.logits, rtol=1e-4, atol=1e-4)

    def test_dsa_precomputed_mask_uses_dispatched_layer_type(self):
        model = Dots3NoteForCausalLM(get_tiny_config(use_dsa=True).text_config).eval()
        full_mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
        sliding_mask = torch.eye(2, dtype=torch.bool).view(1, 1, 2, 2)

        received = []
        handles = [
            layer.register_forward_pre_hook(
                lambda module, args, kwargs: received.append(kwargs["attention_mask"]), with_kwargs=True
            )
            for layer in model.model.layers
        ]
        try:
            with torch.no_grad():
                model(
                    torch.tensor([[1, 2]]),
                    attention_mask={"deepseek_sparse_attention": full_mask, "sliding_attention": sliding_mask},
                    use_cache=False,
                )
        finally:
            for handle in handles:
                handle.remove()
        self.assertIs(received[0], full_mask)
        self.assertIs(received[1], sliding_mask)

    def test_vision_forward(self):
        config = get_tiny_config().vision_config
        config.is_causal = False
        model = Dots3NoteVisionModel(config).eval()
        pixel_values = torch.randn(4, 3 * config.patch_size**2)
        grid_thw = torch.tensor([[1, 2, 2]])
        with torch.no_grad():
            outputs = model(pixel_values, grid_thw, output_hidden_states=True)
        self.assertEqual(outputs.last_hidden_state.shape, (4, config.embed_dim))
        self.assertEqual(outputs.pooler_output.shape, (1, config.adapter_out_dim))
        self.assertEqual(len(outputs.hidden_states), config.num_hidden_layers + 1)

    def test_audio_forward(self):
        config = get_tiny_config().audio_config
        config._attn_implementation = "eager"
        model = Dots3NoteAudioModel(config).eval()
        inputs = {
            "input_features": torch.randn(3, config.feature_size, 32),
            "chunk_sample_lengths": torch.tensor([33, 128, 65]),
        }
        with (
            torch.no_grad(),
            patch.object(model.audio_adapter, "forward", wraps=model.audio_adapter.forward) as adapter,
        ):
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
        adapter.assert_called_once()
        self.assertEqual(outputs.last_hidden_state.shape, (9, config.adapter_output_size))
        self.assertEqual(len(outputs.hidden_states), config.num_hidden_layers + 1)
        self.assertEqual(len(outputs.attentions), config.num_hidden_layers)

    @parameterized.expand(["base", "conditional_generation"])
    def test_multimodal_forward(self, variant):
        config = get_tiny_config()
        model_class = {
            "base": Dots3NoteModel,
            "conditional_generation": Dots3NoteForConditionalGeneration,
        }[variant]
        model = model_class(config).eval()
        input_ids = torch.tensor([[1, 120, 5, 121, 121, 122, 2]])
        pixel_values = torch.randn(4, config.vision_config.num_channels * config.vision_config.patch_size**2)
        media_inputs = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "pixel_values_videos": pixel_values,
            "video_grid_thw": torch.tensor([[1, 2, 2]]),
            "input_features": torch.randn(1, config.audio_config.feature_size, 16),
            "chunk_sample_lengths": torch.tensor([64]),
        }

        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False, **media_inputs)
            embedded_outputs = model(
                inputs_embeds=model.get_input_embeddings()(input_ids), use_cache=False, **media_inputs
            )
            torch.testing.assert_close(embedded_outputs[0], outputs[0])
            backbone = model if variant == "base" else model.model
            for modality, args in (
                ("image", (pixel_values, media_inputs["image_grid_thw"])),
                ("video", (pixel_values, media_inputs["video_grid_thw"])),
                ("audio", (media_inputs["input_features"], media_inputs["chunk_sample_lengths"])),
            ):
                with self.subTest(modality=modality):
                    get_features = getattr(backbone, f"get_{modality}_features")
                    expected = get_features(*args, return_dict=True).to_tuple()
                    actual = get_features(*args, return_dict=False)
                    self.assertIsInstance(actual, tuple)
                    torch.testing.assert_close(actual, expected)
            config.return_dict = False
            tuple_outputs = model(input_ids=input_ids, use_cache=False, return_dict=False, **media_inputs)
            torch.testing.assert_close(tuple_outputs[0], outputs[0])
            config.return_dict = True
            if variant == "conditional_generation":
                generated = model.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=2,
                    min_new_tokens=2,
                    do_sample=False,
                    **media_inputs,
                )
                self.assertEqual(generated.shape, (1, input_ids.shape[1] + 2))
        output = outputs.last_hidden_state if variant == "base" else outputs.logits
        width = config.text_config.hidden_size if variant == "base" else config.text_config.vocab_size
        self.assertEqual(output.shape, (1, input_ids.shape[1], width))
        self.assertTrue(torch.isfinite(output).all())

    def test_multimodal_chunked_prefill_processes_cached_media(self):
        torch.manual_seed(0)
        config = get_tiny_config(use_dsa=True)
        config.text_config.index_topk = config.text_config.max_position_embeddings
        model = Dots3NoteForConditionalGeneration(config).eval()
        prefix_ids = torch.tensor([[1, 5]])
        patch_width = config.vision_config.num_channels * config.vision_config.patch_size**2
        cases = {
            "image": (
                torch.tensor([[config.image_token_id, 7, 2]]),
                {
                    "pixel_values": torch.randn(4, patch_width),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                },
            ),
            "video": (
                torch.tensor([[config.video_token_id, 7, 2]]),
                {
                    "pixel_values_videos": torch.randn(4, patch_width),
                    "video_grid_thw": torch.tensor([[1, 2, 2]]),
                },
            ),
            "audio": (
                torch.tensor([[config.audio_token_id, config.audio_token_id, 7, 2]]),
                {
                    "input_features": torch.randn(1, config.audio_config.feature_size, 16),
                    "chunk_sample_lengths": torch.tensor([64]),
                },
            ),
        }

        for modality, (suffix_ids, media_inputs) in cases.items():
            with self.subTest(modality=modality):
                input_ids = torch.cat((prefix_ids, suffix_ids), dim=-1)
                attention_mask = torch.ones_like(input_ids)

                with torch.no_grad():
                    expected = model(
                        input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        **media_inputs,
                    ).logits[:, -suffix_ids.shape[1] :]
                    prefix = model(prefix_ids, attention_mask=torch.ones_like(prefix_ids), use_cache=True)
                    actual = model(
                        suffix_ids,
                        attention_mask=attention_mask,
                        past_key_values=prefix.past_key_values,
                        use_cache=True,
                        **media_inputs,
                    ).logits

                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_expert_checkpoint_conversion_roundtrip(self):
        torch.manual_seed(13)
        model = Dots3NoteForConditionalGeneration(get_tiny_config()).eval()
        input_ids = torch.tensor([[1, 5, 6, 2]])
        with torch.no_grad():
            expected = model(input_ids, use_cache=False).logits

        with tempfile.TemporaryDirectory() as tmpdirname:
            # Released checkpoints also contain an MTP prediction layer outside the text decoder.
            state_dict = model.state_dict() | {"model.language_model.layers.46.eh_proj.weight": torch.zeros(32, 64)}
            model.save_pretrained(tmpdirname, state_dict=state_dict)
            with safe_open(f"{tmpdirname}/model.safetensors", framework="pt") as checkpoint:
                checkpoint_keys = set(checkpoint.keys())
            self.assertIn("model.layers.1.mlp.experts.0.gate_proj.weight", checkpoint_keys)
            self.assertIn("model.layers.1.mlp.experts.0.up_proj.weight", checkpoint_keys)
            self.assertIn("model.layers.1.mlp.experts.0.down_proj.weight", checkpoint_keys)
            self.assertNotIn("model.layers.1.mlp.experts.gate_up_proj", checkpoint_keys)
            self.assertIn("model.vision_encoder.blocks.1.mlp.experts.0.fc1.weight", checkpoint_keys)
            self.assertIn("model.vision_encoder.blocks.1.mlp.gate_weight", checkpoint_keys)

            model.config.architectures = ["Dots3NoteForCausalLM"]
            legacy_config = model.config.to_dict() | {"rope_scaling": None}
            text_config = legacy_config.pop("text_config")
            legacy_config.update(text_config)
            legacy_config["model_type"] = "dots3_note"
            legacy_config["architectures"] = ["Dots3NoteForCausalLM"]
            with open(f"{tmpdirname}/config.json", "w") as config_file:
                json.dump(legacy_config, config_file)
            reloaded, info = AutoModelForMultimodalLM.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertIsInstance(reloaded, Dots3NoteForConditionalGeneration)
            self.assertFalse(info["missing_keys"])
            self.assertFalse(info["unexpected_keys"])
            torch.testing.assert_close(reloaded.state_dict(), model.state_dict(), rtol=0, atol=0)

        with torch.no_grad():
            actual = reloaded(input_ids, use_cache=False).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
