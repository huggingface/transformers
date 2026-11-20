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
import unittest
import unittest.mock

from huggingface_hub.errors import StrictDataclassClassValidationError
from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_accelerate, require_torch, torch_device


if is_torch_available():
    import torch

    from transformers import DeepseekV41Config, DeepseekV41ForCausalLM, DeepseekV41Model, DeepseekV41TextConfig
    from transformers.models.deepseek_v41.modeling_deepseek_v41 import (
        DeepseekV41Compressor,
        DeepseekV41NgramHasher,
        select_candidate_blocks,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_configuration_common import ConfigTester


INPUTS_EMBEDS_SKIP_REASON = (
    "The Engram layers hash n-grams of token ids, so the model needs `input_ids` and cannot run on `inputs_embeds` "
    "alone."
)


class DeepseekV41ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = DeepseekV41Model
        config_class = DeepseekV41Config
        causal_lm_class = DeepseekV41ForCausalLM

    small_config_overrides = {
        "batch_size": 3,
        "seq_length": 11,
        "hidden_size": 64,
        "num_hidden_layers": 5,
        "num_attention_heads": 4,
        "text_config": {
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
            "index_head_dim": 16,
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
        },
        "vision_config": {
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 48,
            "patch_size": 2,
            "downsample_ratio": 3,
        },
    }

    @classmethod
    def get_small_config_defaults(cls, **kwargs):
        values = super().get_small_config_defaults(**kwargs)
        # The inputs are text only, so the image token must not show up in the random ids
        values["image_token_id"] = values["vocab_size"]
        # The text config takes its sizes and special tokens from the tester, so that the common tests can
        # read them there. Its vocab has one more id than the tester's, for the image token.
        values["text_config"] = {
            **values["text_config"],
            "vocab_size": values["vocab_size"] + 1,
            "hidden_size": values["hidden_size"],
            "num_hidden_layers": values["num_hidden_layers"],
            "num_attention_heads": values["num_attention_heads"],
            # As large as the vocab, so that no tokens are merged and the model works without a tokenizer
            "engram_compressed_vocab_size": values["vocab_size"] + 1,
            "pad_token_id": values["pad_token_id"],
            "bos_token_id": values["bos_token_id"],
            "eos_token_id": values["eos_token_id"],
        }
        return values

    def get_config(self):
        return DeepseekV41Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )


@require_torch
class DeepseekV41ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = DeepseekV41ModelTester

    # The indexer parameters only influence which compressed entries are selected, which is not differentiable
    test_all_params_have_gradient = False
    # The Engram token map and n-gram tables are fixed to the vocabulary, so `resize_token_embeddings` raises
    test_resize_embeddings = False

    def setUp(self):
        super().setUp()
        # The config is a composite, so the text fields like `vocab_size` are on its `text_config`
        self.config_tester = ConfigTester(
            self, config_class=DeepseekV41Config, has_text_modality=False, common_properties=["image_token_id"]
        )

    def is_pipeline_test_to_skip(self, *args, **kwargs):
        return True

    def _image_features_prepare_config_and_inputs(self):
        # The common inputs are text only, so we make the image inputs here: two images with different grids
        config = self.model_tester.get_config()
        image_grid_thw = torch.tensor([[1, 5, 7], [1, 3, 6]], device=torch_device)
        num_patches = int(image_grid_thw.prod(-1).sum())
        pixel_values = torch.randn(num_patches, 3 * config.vision_config.patch_size**2, device=torch_device)
        return config, {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

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

    @unittest.skip(
        reason=(
            "The compressor pools consecutive tokens before the attention mask applies, so left padding shifts "
            "the groups and folds pad tokens into the compressed KV. Same limitation as DeepSeek-V4, not fixed yet."
        )
    )
    def test_left_padding_compatibility(self):
        pass

    def test_reverse_loading_mapping(self):
        # The checkpoint is flat and has the LM head, so the renames that nest it under `model.` only apply to
        # the model with a head. The base model on its own is saved without that prefix.
        super().test_reverse_loading_mapping(skip_base_model=True)

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

    def check_training_gradient_checkpointing(self, gradient_checkpointing_kwargs=None):
        # The common check watches the first checkpointing layer of the model, which is a vision layer. The
        # common inputs are text only, so we put an image at the start of the first row to make that layer run.
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        # 4 x 5 patches are merged 3 x 3 into 2 x 2 tokens, so the image takes 1 + 2 * (2 + 1) + 1 positions
        height, width, num_image_tokens = 4, 5, 8
        inputs_dict["input_ids"][0, :num_image_tokens] = config.image_token_id
        inputs_dict["pixel_values"] = torch.randn(
            height * width, 3 * config.vision_config.patch_size**2, device=torch_device
        )
        inputs_dict["image_grid_thw"] = torch.tensor([[1, height, width]], device=torch_device)
        with unittest.mock.patch.object(
            self.model_tester, "prepare_config_and_inputs_for_common", return_value=(config, inputs_dict)
        ):
            super().check_training_gradient_checkpointing(gradient_checkpointing_kwargs)

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
        input_ids = torch.randint(0, self.model_tester.vocab_size, (2, 23), device=torch_device)
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
        input_ids = torch.randint(0, self.model_tester.vocab_size, (2, 19), device=torch_device)
        with torch.no_grad():
            expected = model(input_ids=input_ids, use_cache=False).logits
            outputs = model(input_ids=input_ids[:, :7], use_cache=True)
            logits = [outputs.logits]
            for position in range(7, input_ids.shape[1]):
                outputs = model(input_ids=input_ids[:, position : position + 1], cache=outputs.cache, use_cache=True)
                logits.append(outputs.logits)
        torch.testing.assert_close(torch.cat(logits, dim=1), expected, atol=1e-4, rtol=1e-4)

    def test_chunked_prefill_matches_full_forward(self):
        """The cache carries a pooling group that is still filling up across calls of any length."""
        config, model = self._get_model()
        input_ids = torch.randint(0, self.model_tester.vocab_size, (2, 19), device=torch_device)
        with torch.no_grad():
            expected = model(input_ids=input_ids, use_cache=False).logits
            outputs = model(input_ids=input_ids[:, :5], use_cache=True)
            logits = [outputs.logits]
            for start, end in ((5, 12), (12, 13), (13, 19)):
                outputs = model(input_ids=input_ids[:, start:end], cache=outputs.cache, use_cache=True)
                logits.append(outputs.logits)
        torch.testing.assert_close(torch.cat(logits, dim=1), expected, atol=1e-4, rtol=1e-4)

    def test_compressed_attention_is_used(self):
        """Tokens outside the sliding window still reach the output, through the compressed KV."""
        config, model = self._get_model()
        input_ids = torch.randint(0, self.model_tester.vocab_size, (1, 19), device=torch_device)
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
        with torch.no_grad():
            # The offloaded one goes first. A weight that was not loaded is uninitialized memory, which could
            # otherwise still hold the result we are comparing with.
            offloaded = offloaded_compressor(hidden_states)
            expected = compressor(hidden_states)
        self.assertEqual(expected.shape, (2, 3, config.head_dim))
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
        text_config = self.model_tester.text_config
        # A layer cannot read compressed KV that no layer before it has made
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "kv_source_layer_ids"):
            DeepseekV41TextConfig(**{**text_config, "kv_source_layer_ids": [3], "index_source_layer_ids": [3, 4]})
        # ... and the index keys are made from the compressed KV, so a KV source must index too
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "index_source_layer_ids"):
            DeepseekV41TextConfig(**{**text_config, "index_source_layer_ids": [3, 4]})

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
        text_config = {**self.model_tester.text_config, "engram_compressed_vocab_size": 60}
        config = DeepseekV41Config(text_config=text_config, vision_config=self.model_tester.vision_config)
        model = DeepseekV41ForCausalLM(config).to(torch_device).eval()
        input_ids = torch.randint(0, self.model_tester.vocab_size, (1, 6), device=torch_device)
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
        text = torch.randint(0, self.model_tester.vocab_size, (1, 12), device=torch_device)
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
