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
import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device


if is_torch_available():
    import torch

    from transformers import DeepseekV41Config, DeepseekV41ForCausalLM, DeepseekV41Model, DeepseekV41TextConfig
    from transformers.models.deepseek_v41.modeling_deepseek_v41 import (
        DeepseekV41NgramHasher,
        select_candidate_blocks,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class DeepseekV41ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = DeepseekV41Model
        config_class = DeepseekV41Config
        causal_lm_class = DeepseekV41ForCausalLM

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.hidden_size = 64
        self.num_attention_heads = 4
        self.num_hidden_layers = 5
        self.seq_length = 11
        self.batch_size = 3
        # The inputs are text only, so the image token must not show up in the random ids
        self.image_token_id = self.vocab_size
        self.text_config = {
            "vocab_size": self.vocab_size + 1,
            "hidden_size": self.hidden_size,
            "moe_intermediate_size": 32,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "head_dim": 32,
            "partial_rotary_factor": 8 / 32,
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
            # As large as the vocab, so that no tokens are merged and the model works without a tokenizer
            "engram_compressed_vocab_size": self.vocab_size + 1,
            "engram_num_embeddings": None,
            "num_nextn_predict_layers": 0,
            "dspark_noise_token_id": 3,
            "rms_norm_eps": 1e-6,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }
        self.vision_config = {
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 48,
            "patch_size": 2,
            "downsample_ratio": 3,
        }

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

    def is_pipeline_test_to_skip(self, *args, **kwargs):
        return True

    @unittest.skip(
        reason=(
            "The compressor pools consecutive tokens before the attention mask applies, so left padding shifts "
            "the groups and folds pad tokens into the compressed KV. Same limitation as DeepSeek-V4, not fixed yet."
        )
    )
    def test_left_padding_compatibility(self):
        pass

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
        image_grid_hw = torch.tensor([[height, width]], device=torch_device)
        text = torch.randint(0, self.model_tester.vocab_size, (1, 12), device=torch_device)
        image = torch.full((1, num_image_tokens), config.image_token_id, device=torch_device)
        input_ids = torch.cat([text[:, :4], image, text[:, 4:]], dim=1)
        with torch.no_grad():
            image_features = model.model.get_image_features(pixel_values, image_grid_hw)
            logits = model(input_ids=input_ids, pixel_values=pixel_values, image_grid_hw=image_grid_hw).logits
            other_image = model(input_ids=input_ids, pixel_values=pixel_values + 1, image_grid_hw=image_grid_hw).logits
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
            model(input_ids=short_input_ids, pixel_values=pixel_values, image_grid_hw=image_grid_hw)
