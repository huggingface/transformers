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
import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

from huggingface_hub.errors import StrictDataclassClassValidationError
from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_accelerate, require_torch, torch_device


if is_torch_available():
    import torch
    from safetensors.torch import load_file, save_file

    from transformers import (
        DeepseekV41Config,
        DeepseekV41ForCausalLM,
        DeepseekV41Model,
        DeepseekV41TextConfig,
        FineGrainedFP8Config,
    )
    from transformers.integrations import FP8Embedding, FP8Linear
    from transformers.models.deepseek_v41.modeling_deepseek_v41 import (
        DeepseekV41Compressor,
        DeepseekV41NgramHasher,
        fake_quantize,
        select_candidate_blocks,
    )
    from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


# The quantization config of deepseek-ai/DeepSeek-V4.1-Flash, as it is on the Hub
CHECKPOINT_QUANTIZATION_CONFIG = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}

INPUTS_EMBEDS_SKIP_REASON = (
    "The Engram layers hash n-grams of token ids, so the model needs `input_ids` and cannot run on `inputs_embeds` "
    "alone."
)


class DeepseekV41ModelTester(VLMModelTester):
    if is_torch_available():
        base_model_class = DeepseekV41Model
        config_class = DeepseekV41Config
        conditional_generation_class = DeepseekV41ForCausalLM

    small_config_overrides = {
        "batch_size": 3,
        "seq_length": 11,
        "vocab_size": 100,
        # Ids 0 to 3 are the pad, BOS, EOS and DSpark noise tokens
        "image_token_id": 4,
        "hidden_size": 64,
        "num_hidden_layers": 5,
        "num_attention_heads": 4,
        # All the attention heads share a single KV head
        "num_key_value_heads": 1,
        "hidden_act": "silu",
        "moe_intermediate_size": 32,
        "head_dim": 32,
        # RoPE on 8 of the 32 head dims
        "partial_rotary_factor": 0.25,
        "q_lora_rank": 32,
        "o_groups": 2,
        "o_lora_rank": 16,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "max_position_embeddings": 64,
        "sliding_window": 4,
        # One layer of each kind: sliding only, compressing (2-to-1), sharing that KV, compressing again (1-to-1),
        # then sharing the KV but indexing it on its own inside the candidate blocks of the layer before
        "compress_ratios": [0, 2, 2, 1, 1],
        "kv_source_layer_ids": [1, 3],
        "index_source_layer_ids": [1, 3, 4],
        "candidate_source_layer_id": 3,
        "candidate_topk_blocks": 2,
        "candidate_block_size": 2,
        # Enough heads that two entries never get the exact same index score. With a couple of heads, the ReLU
        # leaves many scores at exactly 0, and how top-k breaks those ties depends on the shape of its input.
        "index_n_heads": 16,
        "index_head_dim": 32,
        "index_topk": 3,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 3,
        "engram_layer_ids": [0, 2],
        "engram_max_ngram_size": 3,
        "engram_vocab_size": 50,
        "engram_n_heads": 2,
        "engram_head_dim": 8,
        "engram_pad_token_id": 2,
        "engram_num_embeddings": None,
        "num_nextn_predict_layers": 0,
        "dspark_noise_token_id": 3,
        "rms_norm_eps": 1e-6,
        # The vision tower. The fields that it shares with the language model take a `vision_` prefix.
        "vision_hidden_size": 32,
        "vision_num_hidden_layers": 2,
        "vision_intermediate_size": 48,
        "patch_size": 2,
        "downsample_ratio": 3,
        # Every prompt starts with one image of 3 x 6 patches. They are merged 3 x 3 into 1 x 2 tokens, so with
        # the newline of the row and the two delimiters, the image takes 1 + 1 * (2 + 1) + 1 positions.
        "grid_height": 3,
        "grid_width": 6,
        "num_image_tokens": 5,
    }

    @classmethod
    def get_small_config_defaults(cls, **kwargs):
        values = super().get_small_config_defaults(**kwargs)
        # As large as the vocab, so that no tokens are merged and the model works without a tokenizer
        values["engram_compressed_vocab_size"] = values["vocab_size"]
        return values

    def create_pixel_values(self, batch_size: int | None = None):
        # `pixel_values` has a row per patch rather than per image, with the images one after the other
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_patches = batch_size * self.grid_height * self.grid_width
        return floats_tensor([num_patches, self.num_channels * self.patch_size**2])

    def get_additional_inputs(self, config, input_ids, modality_inputs, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        grid = [1, self.grid_height, self.grid_width]
        return {"image_grid_thw": torch.tensor([grid] * batch_size, device=torch_device)}

    def prepare_text_ids(self, batch_size, seq_length):
        # Starts after the special tokens and the image token, so that none of them turn up in the text
        low = max(self._special_token_ids) + 1
        return torch.randint(low, self.vocab_size, (batch_size, seq_length), device=torch_device)


@require_torch
class DeepseekV41ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = DeepseekV41ModelTester

    # The indexer parameters only influence which compressed entries are selected, which is not differentiable
    test_all_params_have_gradient = False
    # The Engram token map and n-gram tables are fixed to the vocabulary, so `resize_token_embeddings` raises
    test_resize_embeddings = False
    # Tolerance for our tests that compute the same logits in two ways. The model rounds its KV to FP8, so when
    # float noise tips a value over a rounding boundary, the two ways differ by a rounding step instead of by
    # the noise: up to 8e-3 on the logits over 300 random models, against 5e-7 without the rounding.
    rounded_logits_tolerance = 1e-2

    def is_pipeline_test_to_skip(self, *args, **kwargs):
        return True

    def _image_features_get_expected_num_attentions(self, model_tester=None):
        # The common version reads `num_hidden_layers` from the tester, which is the depth of the language model
        return self.model_tester.vision_num_hidden_layers

    def _check_attentions_for_generate(
        self, batch_size, attentions, prompt_length, output_length, config, decoder_cache
    ):
        # The layers that attend to compressed KV have those entries after the window on the KV axis, one per
        # complete group of `compress_ratio` tokens. So the KV length differs between layers.
        config = config.get_text_config()
        self.assertIsInstance(attentions, tuple)
        self.assertEqual(len(attentions), output_length - prompt_length)
        for generated_length, iter_attentions in enumerate(attentions):
            self.assertIsInstance(iter_attentions, tuple)
            seq_length = prompt_length + generated_length
            if decoder_cache is not None and generated_length > 0:
                # Decoding one token from the cache, which only keeps the rest of the window
                query_length = 1
                window_length = min(seq_length, config.sliding_window)
            else:
                # The whole sequence goes in, and the mask is what limits each query to its window
                query_length = window_length = seq_length
            expected_shapes = [
                (
                    batch_size,
                    config.num_attention_heads,
                    query_length,
                    window_length + (seq_length // ratio if ratio else 0),
                )
                for ratio in config.compress_ratios
            ]
            self.assertListEqual([layer_attention.shape for layer_attention in iter_attentions], expected_shapes)

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
    ):
        # The residual is `hc_mult` parallel streams, which are only collapsed for the final norm. So every
        # hidden state has an extra dim, except for the last one.
        config = config.get_text_config()
        self.assertIsInstance(hidden_states, tuple)
        self.assertEqual(len(hidden_states), output_length - prompt_length)
        for generated_length, iter_hidden_states in enumerate(hidden_states):
            self.assertIsInstance(iter_hidden_states, tuple)
            input_length = 1 if use_cache and generated_length > 0 else prompt_length + generated_length
            streams_shape = (batch_size, input_length, config.hc_mult, config.hidden_size)
            self.assertListEqual(
                [layer_hidden_states.shape for layer_hidden_states in iter_hidden_states],
                [streams_shape] * config.num_hidden_layers + [(batch_size, input_length, config.hidden_size)],
            )

    def _check_cache_for_generate(self, batch_size, cache, seq_length, config):
        # Every layer keeps the part of its window that the next token can still see, and there is a single
        # KV head
        config = config.get_text_config()
        self.assertEqual(len(cache.layers), config.num_hidden_layers)
        window_shape = (batch_size, 1, min(seq_length, config.sliding_window - 1), config.head_dim)
        for layer, attention_type, ratio in zip(cache.layers, config.attention_layer_types, config.compress_ratios):
            self.assertEqual(layer.keys.shape, window_shape)
            self.assertEqual(layer.values.shape, window_shape)
            if attention_type == "compressed_attention":
                # Only the layer that compresses the KV stores it. The KV-sharing layers read it from there.
                self.assertEqual(layer.compressed_kv.shape, (batch_size, seq_length // ratio, config.head_dim))
            else:
                self.assertIsNone(layer.compressed_kv)

    def test_hidden_states_output(self):
        # Same as the common test, but the hidden states are `hc_mult` parallel streams until the final norm
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        batch_size, seq_length = inputs_dict["input_ids"].shape
        text_config = config.text_config
        streams_shape = (batch_size, seq_length, text_config.hc_mult, text_config.hidden_size)
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                hidden_states = model(**inputs_dict, output_hidden_states=True).hidden_states
            self.assertListEqual(
                [layer_hidden_states.shape for layer_hidden_states in hidden_states],
                [streams_shape] * text_config.num_hidden_layers + [(batch_size, seq_length, text_config.hidden_size)],
            )

    def test_left_padding_compatibility(self):
        # The common test allows 1e-5, which is a tolerance for float noise. With the activations rounded to FP8
        # and FP4, the noise now and then tips a value over a rounding boundary, and the logits move by a
        # rounding step instead (2 of 200 random models on a GPU).
        tolerance = self.rounded_logits_tolerance
        super().test_left_padding_compatibility(atol=tolerance, rtol=tolerance)

    def test_reverse_loading_mapping(self):
        # The checkpoint is flat and has the LM head, so the renames that nest it under `model.` only apply to
        # the model with a head. The base model on its own is saved without that prefix.
        super().test_reverse_loading_mapping(skip_base_model=True)

    @unittest.skip(
        reason=(
            "The attention of the language model has sinks and reads selected compressed KV, so it is eager only, "
            "and the top-level model cannot switch to SDPA."
        )
    )
    def test_can_set_attention_dynamically_composite_model(self):
        pass

    @unittest.skip(reason=INPUTS_EMBEDS_SKIP_REASON)
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason=INPUTS_EMBEDS_SKIP_REASON)
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @parameterized.expand([("greedy", 1), ("beam search", 2)])
    @unittest.skip(reason=INPUTS_EMBEDS_SKIP_REASON)
    def test_generate_from_inputs_embeds(self, _, num_beams):
        pass

    @unittest.skip(reason=INPUTS_EMBEDS_SKIP_REASON)
    def test_generate_from_random_inputs_embeds(self):
        pass

    def test_training_gradient_checkpointing_use_reentrant_true(self):
        # Reentrant checkpointing only tracks the gradient of positional tensor arguments, so the compressed KV
        # that the layers share would silently lose theirs. The model refuses it instead.
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = DeepseekV41ForCausalLM(config)
        with self.assertRaisesRegex(ValueError, "use_reentrant=True"):
            model.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": True})

    def test_checkpointing_keeps_the_gradients(self):
        """
        The KV-sharing layers read the compressed KV of an earlier layer, and checkpointing must not change the
        gradient that flows back through it. The common tests only check which parameters get a gradient.
        """
        config, model = self._get_model()
        model.train()
        input_ids = self.model_tester.prepare_text_ids(2, 23)
        model(input_ids, labels=input_ids).loss.backward()
        expected = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
        self.assertTrue(any(".compressor." in name for name in expected))

        model.zero_grad()
        model.activation_checkpointing_enable()
        model(input_ids, labels=input_ids).loss.backward()
        for name, expected_grad in expected.items():
            torch.testing.assert_close(model.get_parameter(name).grad, expected_grad, msg=lambda msg: f"{name}: {msg}")

    def _get_model(self):
        config = self.model_tester.get_config()
        torch.manual_seed(0)
        model = DeepseekV41ForCausalLM(config).to(torch_device).eval()
        # The default init leaves the attention sinks and hyper-connection biases at zero, and we want them to matter
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.endswith(("sinks", ".base", ".scale")):
                    param.normal_(mean=1.0 if name.endswith(".scale") else 0.0, std=0.3)
        return config, model

    def test_cached_decoding_matches_full_forward(self):
        """Decoding one token at a time with a cache gives the logits of a single forward over the whole sequence."""
        config, model = self._get_model()
        input_ids = self.model_tester.prepare_text_ids(2, 19)
        with torch.no_grad():
            expected = model(input_ids=input_ids, use_cache=False).logits
            outputs = model(input_ids=input_ids[:, :7], use_cache=True)
            logits = [outputs.logits]
            for position in range(7, input_ids.shape[1]):
                outputs = model(input_ids=input_ids[:, position : position + 1], cache=outputs.cache, use_cache=True)
                logits.append(outputs.logits)
        tolerance = self.rounded_logits_tolerance
        torch.testing.assert_close(torch.cat(logits, dim=1), expected, atol=tolerance, rtol=tolerance)

    def test_chunked_prefill_matches_full_forward(self):
        """The cache carries a pooling group that is still filling up across calls of any length."""
        config, model = self._get_model()
        input_ids = self.model_tester.prepare_text_ids(2, 19)
        with torch.no_grad():
            expected = model(input_ids=input_ids, use_cache=False).logits
            outputs = model(input_ids=input_ids[:, :5], use_cache=True)
            logits = [outputs.logits]
            for start, end in ((5, 12), (12, 13), (13, 19)):
                outputs = model(input_ids=input_ids[:, start:end], cache=outputs.cache, use_cache=True)
                logits.append(outputs.logits)
        tolerance = self.rounded_logits_tolerance
        torch.testing.assert_close(torch.cat(logits, dim=1), expected, atol=tolerance, rtol=tolerance)

    def test_left_padding_matches_unpadded_with_cache(self):
        """Each row of a left-padded batch gets the logits of that row on its own, with a cache as well."""
        config, model = self._get_model()
        # 0, 3 and 5 tokens of padding. The pooling groups are counted from the first real token of a row, so
        # the first two rows never complete a 2-token group on the same step.
        rows = [self.model_tester.prepare_text_ids(1, length) for length in (19, 16, 14)]
        input_ids = torch.full((3, 19), config.text_config.pad_token_id, device=torch_device)
        attention_mask = torch.zeros((3, 19), dtype=torch.long, device=torch_device)
        for row_idx, row in enumerate(rows):
            input_ids[row_idx, -row.shape[1] :] = row[0]
            attention_mask[row_idx, -row.shape[1] :] = 1
        # The positions that `generate` makes: each row counts from its own first real token
        position_ids = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)

        # The first chunk is nothing but padding for the last row, then comes a longer chunk, then single tokens
        cache, logits = None, []
        with torch.no_grad():
            for start, end in ((0, 5), (5, 12), *((position, position + 1) for position in range(12, 19))):
                outputs = model(
                    input_ids=input_ids[:, start:end],
                    attention_mask=attention_mask[:, :end],
                    position_ids=position_ids[:, start:end],
                    cache=cache,
                    use_cache=True,
                )
                cache = outputs.cache
                logits.append(outputs.logits)
            logits = torch.cat(logits, dim=1)
            tolerance = self.rounded_logits_tolerance
            for row_idx, row in enumerate(rows):
                expected = model(input_ids=row, use_cache=False).logits[0]
                torch.testing.assert_close(
                    logits[row_idx, -row.shape[1] :], expected, atol=tolerance, rtol=tolerance, msg=f"Row {row_idx}"
                )

    def test_compressed_attention_is_used(self):
        """Tokens outside the sliding window still reach the output, through the compressed KV."""
        config, model = self._get_model()
        input_ids = self.model_tester.prepare_text_ids(1, 19)
        changed = input_ids.clone()
        # Far outside the window of the last position (4 tokens), and too old for the 2-token Engram look-back
        changed[0, 2] = (changed[0, 2] + 1) % self.model_tester.vocab_size
        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1]
            changed_logits = model(input_ids=changed, use_cache=False).logits[0, -1]
        self.assertGreater((logits - changed_logits).abs().max().item(), 1e-4)

    @require_accelerate
    def test_compressor_with_offloaded_weights(self):
        """Offloaded weights are only loaded while the module that owns them runs, and the pooling must cope."""
        from accelerate import cpu_offload

        config = self.model_tester.get_config().text_config
        # Layer 1 pools 2-to-1, which is the path that promotes its weights to float32
        compressor = DeepseekV41Compressor(config, layer_idx=1).to(torch_device).eval()
        offloaded_compressor = copy.deepcopy(compressor)
        cpu_offload(offloaded_compressor, execution_device=torch.device(torch_device))
        hidden_states = torch.randn(2, 6, config.hidden_size, device=torch_device)
        # No padding and no cache, and the RoPE of the latents plays no part in the pooling
        inputs = (
            hidden_states,
            (torch.ones(1, 6, 4, device=torch_device), torch.zeros(1, 6, 4, device=torch_device)),
            torch.ones(2, 6, dtype=torch.bool, device=torch_device),
            torch.zeros(2, dtype=torch.long, device=torch_device),
        )
        with torch.no_grad():
            # The offloaded one goes first. A weight that was not loaded is uninitialized memory, which could
            # otherwise still hold the result we are comparing with.
            offloaded, _, _ = offloaded_compressor(*inputs)
            expected, _, num_groups = compressor(*inputs)
        self.assertEqual(expected.shape, (2, 3, config.head_dim))
        self.assertEqual(num_groups.tolist(), [3, 3])
        torch.testing.assert_close(offloaded, expected)

    def test_attention_layer_types(self):
        config = self.model_tester.get_config().text_config
        self.assertEqual(
            config.attention_layer_types,
            [
                "sliding_attention",
                "compressed_attention",
                "kv_shared_attention",
                "compressed_attention",
                "kv_shared_indexed_attention",
            ],
        )
        text_config = config.to_dict()
        # The layer types are derived from the fields that we change below, so they have to be derived again
        del text_config["attention_layer_types"]
        # A layer cannot read compressed KV that no layer before it has made
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "kv_source_layer_ids"):
            DeepseekV41TextConfig.from_dict(
                {**text_config, "kv_source_layer_ids": [3], "index_source_layer_ids": [3, 4]}
            )
        # ... and the index keys are made from the compressed KV, so a KV source must index too
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "index_source_layer_ids"):
            DeepseekV41TextConfig.from_dict({**text_config, "index_source_layer_ids": [3, 4]})

    def test_engram_hash_ids(self):
        """The n-gram hash ids match the ones of the reference implementation, with and without a masked token."""
        config = self.model_tester.get_config().text_config
        hasher = DeepseekV41NgramHasher(config)
        self.assertEqual(hasher.bucket_sizes.tolist(), [[53, 59, 61, 67], [71, 73, 79, 83]])
        input_ids = torch.tensor([[5, 9, 2, 7, 7, 3]])
        # fmt: off
        expected = [
            [[0, 56, 118, 234], [29, 105, 167, 257]],
            [[16, 61, 163, 235], [4, 132, 174, 274]],
            [[15, 88, 113, 194], [56, 98, 189, 267]],
            [[27, 59, 167, 201], [14, 96, 179, 253]],
            [[23, 80, 118, 228], [16, 92, 199, 242]],
            [[13, 76, 118, 200], [58, 88, 146, 227]],
        ]
        expected_masked = [
            [[0, 56, 118, 234], [29, 105, 167, 257]],
            [[16, 61, 163, 235], [4, 132, 174, 274]],
            [[15, 88, 113, 194], [56, 98, 189, 267]],
            [[41, 58, 115, 232], [51, 106, 201, 234]],
            [[27, 59, 116, 203], [14, 96, 209, 224]],
            [[13, 76, 139, 237], [58, 88, 202, 287]],
        ]
        # fmt: on
        self.assertEqual(hasher(input_ids).tolist(), [expected])
        token_mask = torch.tensor([[True, True, True, False, True, True]])
        self.assertEqual(hasher(input_ids, token_mask).tolist(), [expected_masked])

    def test_engram_token_map_must_be_set(self):
        """With merged tokens, the map comes from the tokenizer, and the model says so instead of guessing."""
        config = self.model_tester.get_config()
        config.text_config.engram_compressed_vocab_size = 60
        model = DeepseekV41ForCausalLM(config).to(torch_device).eval()
        input_ids = self.model_tester.prepare_text_ids(1, 6)
        with self.assertRaisesRegex(ValueError, "set_engram_token_map"):
            model(input_ids=input_ids)

        token_map = torch.arange(config.text_config.vocab_size) % 60
        with self.assertRaisesRegex(ValueError, "compressed ids"):
            model.set_engram_token_map(token_map % 50)
        model.set_engram_token_map(token_map)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
        self.assertEqual(logits.shape, (1, 6, config.text_config.vocab_size))

    def test_resize_token_embeddings_is_not_supported(self):
        """A new token would have no entry in the Engram token map, so resizing raises instead."""
        config, model = self._get_model()
        with self.assertRaisesRegex(NotImplementedError, "placeholder tokens"):
            model.resize_token_embeddings(config.text_config.vocab_size + 10)
        self.assertEqual(model.get_input_embeddings().weight.shape[0], config.text_config.vocab_size)

    def test_fake_quantize(self):
        # First, FP4 with a scale of 1, which the largest value of 6 gives. FP4 holds 0, 0.5, 1, 1.5, 2, 3, 4 and 6
        # times the scale, and a value halfway between two of them goes to the one with the even mantissa.
        values = [6.0, 5.0, 3.5, 2.5, 1.75, 1.25, 0.75, 0.25, 0.2, -0.3, -2.6, -5.5, 0.0, 1.0, 1.5, 3.0]
        expected = [6.0, 4.0, 4.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, -0.5, -3.0, -6.0, 0.0, 1.0, 1.5, 3.0]
        values = torch.tensor([values], device=torch_device)
        self.assertEqual(fake_quantize(values, "fp4", block_size=16).tolist(), [expected])
        # Each block has its own scale: an all-zero block stays at zero, next to the block from before
        two_blocks = torch.cat([values, torch.zeros_like(values)], dim=-1)
        self.assertEqual(fake_quantize(two_blocks, "fp4", block_size=16).tolist(), [expected + [0.0] * 16])

        # Next, the scale is a power of two. Scaling the block by one changes nothing else, but a largest value
        # just over 6 doubles the scale, and everything is rounded twice as coarsely.
        scaled = fake_quantize(values * 2.0**-20, "fp4", block_size=16)
        self.assertEqual((scaled * 2.0**20).tolist(), [expected])
        coarse = fake_quantize(torch.tensor([[6.5, 5.0, 3.0, 1.4]], device=torch_device), "fp4", block_size=4)
        self.assertEqual(coarse.tolist(), [[6.0, 4.0, 3.0, 1.0]])

        # With an FP8 scale, the scale is the largest value over 6, rounded to FP8: 5 / 6 becomes 0.8125. The
        # largest value itself is then cut off at 6 times the scale.
        fine = fake_quantize(torch.tensor([[5.0, 2.5, 1.0, 0.0]], device=torch_device), "fp4", 4, scale_format="fp8")
        self.assertEqual(fine.tolist(), [[4.875, 2.4375, 0.8125, 0.0]])

        # Finally, FP8 against the cast to FP8 of PyTorch, with the power-of-two scale of each block. The scale
        # has a floor: a block whose largest value is under 1e-4 is scaled as if it were 1e-4.
        values = torch.randn(64, 32, device=torch_device) * torch.logspace(-6, 6, 64, device=torch_device)[:, None]
        largest = values.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
        scale = torch.exp2(torch.ceil(torch.log2(largest.double() / 448))).float()
        expected = (values / scale).to(torch.float8_e4m3fn).float() * scale
        torch.testing.assert_close(fake_quantize(values, "fp8", block_size=32), expected, rtol=0, atol=0)

    def test_fake_quantize_passes_the_gradient_through(self):
        values = torch.randn(3, 64, device=torch_device, requires_grad=True)
        rounded = fake_quantize(values, "fp8", block_size=32)
        rounded.sum().backward()
        self.assertTrue(torch.equal(values.grad, torch.ones_like(values)))
        # ... and the values are the rounded ones to the bit, as without a gradient
        self.assertTrue(torch.equal(rounded, fake_quantize(values.detach(), "fp8", block_size=32)))

    def test_cache_holds_rounded_kv(self):
        """The KV and the index keys are rounded before they go in the cache, so rounding them again is a no-op."""
        config, model = self._get_model()
        input_ids = self.model_tester.prepare_text_ids(2, 19)
        with torch.no_grad():
            cache = model(input_ids=input_ids, use_cache=True).cache
        for layer, attention_type in zip(cache.layers, config.text_config.attention_layer_types):
            self.assertTrue(torch.equal(fake_quantize(layer.keys, "fp8", block_size=32), layer.keys))
            if attention_type == "compressed_attention":
                self.assertTrue(torch.equal(fake_quantize(layer.index_keys, "fp4", block_size=32), layer.index_keys))

    def test_select_candidate_blocks(self):
        scores = torch.tensor([[[0.1, 0.9, 0.3, 0.2, 0.8, float("-inf")]]])
        # The best block is the first one, and the block with the newest visible entry is always kept
        keep = select_candidate_blocks(scores, num_visible=torch.tensor([[5]]), topk_blocks=2, block_size=2)
        self.assertEqual(keep.tolist(), [[[True, True, False, False, True, True]]])
        # With one visible block, the second pick is not reachable and is dropped
        scores = torch.tensor([[[0.1, 0.9, float("-inf"), float("-inf")]]])
        keep = select_candidate_blocks(scores, num_visible=torch.tensor([[2]]), topk_blocks=2, block_size=2)
        self.assertEqual(keep.tolist(), [[[True, True, False, False]]])

    def test_image_and_text_forward(self):
        """An image takes a start token, its rows of tokens with a newline after each, and an end token."""
        config, model = self._get_model()
        # 5 x 7 patches are merged 3 x 3 into 2 x 3 tokens, so the image takes 1 + 2 * (3 + 1) + 1 positions
        height, width, num_image_tokens = 5, 7, 10
        pixel_values = torch.randn(height * width, 3 * config.vision_config.patch_size**2, device=torch_device)
        image_grid_thw = torch.tensor([[1, height, width]], device=torch_device)
        text = self.model_tester.prepare_text_ids(1, 12)
        image = torch.full((1, num_image_tokens), config.image_token_id, device=torch_device)
        input_ids = torch.cat([text[:, :4], image, text[:, 4:]], dim=1)
        with torch.no_grad():
            image_features = model.model.get_image_features(pixel_values, image_grid_thw).pooler_output[0]
            logits = model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits
            other_image = model(
                input_ids=input_ids, pixel_values=pixel_values + 1, image_grid_thw=image_grid_thw
            ).logits
        self.assertEqual(image_features.shape, (num_image_tokens, config.text_config.hidden_size))
        torch.testing.assert_close(image_features[0], model.model.image_start)
        torch.testing.assert_close(image_features[4], model.model.image_newline)
        torch.testing.assert_close(image_features[-1], model.model.image_end)
        self.assertEqual(logits.shape, (1, input_ids.shape[1], config.text_config.vocab_size))
        # The text before the image cannot see it, and the text after it can
        torch.testing.assert_close(logits[:, :4], other_image[:, :4])
        self.assertGreater((logits[:, -1] - other_image[:, -1]).abs().max().item(), 1e-4)

        # A prompt whose image span is one token short of what the image takes up
        short_input_ids = torch.cat([input_ids[:, :5], input_ids[:, 6:]], dim=1)
        with self.assertRaisesRegex(ValueError, "image tokens"):
            model(input_ids=short_input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

        # An image is one frame, so a grid with a temporal size other than 1 is not an image
        video_grid_thw = torch.tensor([[2, height, width]], device=torch_device)
        with self.assertRaisesRegex(ValueError, "temporal size"):
            model(input_ids=input_ids, pixel_values=pixel_values.repeat(2, 1), image_grid_thw=video_grid_thw)

    def test_beam_search_keeps_each_image_whole(self):
        """
        `generate` repeats its inputs once per beam. `pixel_values` has a row per patch, so repeating its rows
        would hand every beam the patches of a scrambled image. The common beam tests only check output shapes.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = DeepseekV41ForCausalLM(config).to(torch_device).eval()
        batch_size, num_beams = inputs_dict["input_ids"].shape[0], 2
        with torch.no_grad():
            expected = model(**inputs_dict).logits[:, -1]
            outputs = model.generate(
                **inputs_dict,
                num_beams=num_beams,
                max_new_tokens=2,
                do_sample=False,
                output_logits=True,
                return_dict_in_generate=True,
            )
        # On the first step the beams of a prompt are still copies of it, so each one gets the logits of the prompt
        first_step_logits = outputs.logits[0].view(batch_size, num_beams, -1)
        tolerance = self.rounded_logits_tolerance
        for beam in range(num_beams):
            torch.testing.assert_close(first_step_logits[:, beam], expected, atol=tolerance, rtol=tolerance)

    def test_image_features_of_several_images(self):
        """Images are encoded independently, whether or not they come in together."""
        config, model = self._get_model()
        model.set_attn_implementation("eager")
        image_grid_thw = torch.tensor([[1, 5, 7], [1, 3, 6]], device=torch_device)
        pixel_values = torch.randn(5 * 7 + 3 * 6, 3 * config.vision_config.patch_size**2, device=torch_device)
        with torch.no_grad():
            outputs = model.model.get_image_features(
                pixel_values, image_grid_thw, output_hidden_states=True, output_attentions=True
            )
            first = model.model.get_image_features(pixel_values[:35], image_grid_thw[:1], output_attentions=True)
            second = model.model.get_image_features(pixel_values[35:], image_grid_thw[1:], output_attentions=True)

        # 2 x 3 and 1 x 2 tokens, each with a newline per row and the two delimiters
        self.assertEqual([features.shape[0] for features in outputs.pooler_output], [10, 5])
        torch.testing.assert_close(outputs.pooler_output[0], first.pooler_output[0])
        torch.testing.assert_close(outputs.pooler_output[1], second.pooler_output[0])
        # The hidden states and the attentions cover the patches of both images, like `pixel_values`
        self.assertEqual(len(outputs.hidden_states), config.vision_config.num_hidden_layers + 1)
        self.assertEqual(outputs.last_hidden_state.shape, (53, config.vision_config.hidden_size))
        for layer_attention, first_attention, second_attention in zip(
            outputs.attentions, first.attentions, second.attentions
        ):
            self.assertEqual(layer_attention.shape, (config.vision_config.num_attention_heads, 53, 53))
            torch.testing.assert_close(layer_attention[:, :35, :35], first_attention)
            torch.testing.assert_close(layer_attention[:, 35:, 35:], second_attention)
            # ... and no patch attends to the other image
            self.assertEqual(layer_attention[:, :35, 35:].abs().max().item(), 0.0)
            self.assertEqual(layer_attention[:, 35:, :35].abs().max().item(), 0.0)

    def test_fp8_quantizer_conversion(self):
        """
        The quantization config of the checkpoint only describes its FP8 linears. The model config adds the Engram
        tables, which are stored in FP8 as well, and the linears that are stored in BF16.
        """
        config = self.model_tester.get_config().to_dict()
        # A scale of an Engram table covers 32 values, so the rows have to be at least that wide
        config["text_config"]["engram_head_dim"] = 32
        config = DeepseekV41Config.from_dict({**config, "quantization_config": CHECKPOINT_QUANTIZATION_CONFIG})
        with torch.device("meta"):
            model = DeepseekV41ForCausalLM(config)
        FineGrainedFP8HfQuantizer(FineGrainedFP8Config(**config.quantization_config)).preprocess_model(model)

        bf16_linears = (
            "lm_head",
            "vision_tower",
            "multi_modal_projector",
            "compressor",
            "indexer.k_proj",
            "indexer.weights_proj",
        )
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                is_bf16 = any(part in name for part in bf16_linears)
                self.assertEqual(isinstance(module, FP8Linear), not is_bf16, name)
        for layer_idx in config.text_config.engram_layer_ids:
            table = model.model.language_model.layers[layer_idx].engram.embed
            self.assertIsInstance(table, FP8Embedding)
            # One scale per block of 32 values, and the rows are a single block wide here
            self.assertEqual(table.weight_scale.shape, (table.num_embeddings, 1))
        # The token embeddings are not a quantized table
        self.assertIs(type(model.model.language_model.embed_tokens), torch.nn.Embedding)

    @require_accelerate
    def test_loading_the_quantized_checkpoint_layout(self):
        """
        The checkpoint stores its Engram tables and most of its linears in FP8, and its experts in FP4, each with
        power-of-two scales next to it. Loading that layout with `dequantize=True` gives the model that those
        tensors stand for, with the Engram tables still in FP8.
        """
        config = self.model_tester.get_config()
        # A scale of an Engram table covers 32 values, so the rows have to be at least that wide
        config.text_config.engram_head_dim = 32
        torch.manual_seed(0)
        model = DeepseekV41ForCausalLM(config)
        fp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        # Extracts the linears that the checkpoint stores in FP8, by their names in the checkpoint
        fp8_linears = re.compile(
            r"\.(attn\.(indexer\.)?wq_b|attn\.(wq_a|wkv|wo_a|wo_b)|engram\.wkv|shared_experts\.w[123])\.weight$"
        )

        with tempfile.TemporaryDirectory() as quantized_dir, tempfile.TemporaryDirectory() as plain_dir:
            # `save_pretrained` writes the names of the original checkpoint. We quantize the tensors that the
            # checkpoint quantizes, and keep the values they stand for to build the model we compare with.
            model.save_pretrained(quantized_dir)
            model.save_pretrained(plain_dir)
            quantized, dequantized = {}, {}
            for name, tensor in load_file(f"{quantized_dir}/model.safetensors").items():
                if ".ffn.experts." in name:
                    # FP4 experts, with a scale for every 32 values of a row
                    block_rows, max_value = 1, 6.0
                elif ".engram.embed." in name:
                    # FP8 tables, with a scale for every 32 values of a row too
                    block_rows, max_value = 1, 448.0
                elif fp8_linears.search(name):
                    # FP8 linears, with a scale for every block of 32 x 32 values
                    block_rows, max_value = 32, 448.0
                else:
                    # Everything else is stored as it is
                    quantized[name] = dequantized[name] = tensor
                    continue
                blocks = tensor.abs().unflatten(0, (-1, block_rows)).unflatten(-1, (-1, 32))
                scales = torch.exp2(torch.ceil(torch.log2(blocks.amax(dim=(1, 3)).clamp_min(1e-4) / max_value)))
                scale_of_each_value = scales.repeat_interleave(block_rows, dim=0).repeat_interleave(32, dim=1)
                scaled = tensor / scale_of_each_value
                if max_value == 6.0:
                    # FP4 has 8 magnitudes and a sign bit, and a byte holds two values, the first one in its low bits
                    codes = (scaled.abs().unsqueeze(-1) - fp4_values).abs().argmin(-1)
                    values = fp4_values[codes] * scaled.sign()
                    codes = codes + 8 * (scaled < 0)
                    quantized[name] = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8).view(torch.int8)
                else:
                    quantized[name] = scaled.to(torch.float8_e4m3fn)
                    values = quantized[name].float()
                quantized[name.removesuffix("weight") + "scale"] = scales.to(torch.float8_e8m0fnu)
                dequantized[name] = values * scale_of_each_value
            save_file(quantized, f"{quantized_dir}/model.safetensors", metadata={"format": "pt"})
            save_file(dequantized, f"{plain_dir}/model.safetensors", metadata={"format": "pt"})
            config_path = Path(quantized_dir, "config.json")
            config_dict = json.loads(config_path.read_text(encoding="utf-8"))
            config_dict["quantization_config"] = CHECKPOINT_QUANTIZATION_CONFIG
            config_path.write_text(json.dumps(config_dict), encoding="utf-8")

            expected_model = DeepseekV41ForCausalLM.from_pretrained(plain_dir).to(torch_device).eval()
            loaded_model = DeepseekV41ForCausalLM.from_pretrained(
                quantized_dir, quantization_config=FineGrainedFP8Config(dequantize=True)
            )
            loaded_model = loaded_model.to(torch_device).eval()

        for layer_idx in config.text_config.engram_layer_ids:
            self.assertIsInstance(loaded_model.model.language_model.layers[layer_idx].engram.embed, FP8Embedding)
        input_ids = self.model_tester.prepare_text_ids(2, 9)
        with torch.no_grad():
            expected = expected_model(input_ids=input_ids).logits
            logits = loaded_model(input_ids=input_ids).logits
        torch.testing.assert_close(logits, expected)
