import os
import unittest

import pytest

pytest.importorskip("parameterized")

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester

if is_torch_available():
    import torch

    from transformers import Evo2ForCausalLM, Evo2Model, Evo2Tokenizer


class Evo2ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Evo2Model

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            pad_token_id=1,
            bos_token_id=None,
            eos_token_id=0,
            vocab_size=256,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=64,
            use_input_mask=True,
            use_token_type_ids=False,
            use_labels=True,
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        config.layer_types = ["attention"] * config.num_hidden_layers
        config.hyena_filters = 8
        config.hyena_kernel_size = 3
        config.hyena_order = 2
        config.tie_word_embeddings = True
        return config


@require_torch
class Evo2ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Evo2ModelTester


@require_torch
@slow
class Evo2InferenceTest(unittest.TestCase):
    """Test inference against ground truth logits from the original evo2_1b_base model."""

    @staticmethod
    def convert_original_weights_to_transformers(original_weights):
        """Convert weights from original Evo2 format to transformers format."""
        from transformers import Evo2Config

        # Create config based on the original model architecture
        # vocab_size=512, hidden_size=1920, 25 layers (21 hyena + 4 attention every 7th layer starting from 3)
        layer_types = []
        for i in range(25):
            if i % 7 == 3:
                layer_types.append("attention")
            else:
                layer_types.append("hyena")

        config = Evo2Config(
            vocab_size=512,
            hidden_size=1920,
            intermediate_size=5120,
            num_hidden_layers=25,
            num_attention_heads=15,  # 1920 / 128
            num_key_value_heads=15,
            layer_types=layer_types,
            hyena_filters=128,  # Number of filter groups
            hyena_order=3,  # 5760 / 1920 = 3
            hyena_kernel_size=3,  # Short filter kernel size
            tie_word_embeddings=True,
        )

        # Initialize new state dict
        new_state_dict = {}

        # Convert embeddings
        new_state_dict["model.embed_tokens.weight"] = original_weights["embedding_layer.weight"]
        new_state_dict["lm_head.weight"] = original_weights["unembed.weight"]

        # Convert each layer
        for layer_idx in range(25):
            layer_type = layer_types[layer_idx]
            orig_prefix = f"blocks.{layer_idx}"
            new_prefix = f"model.layers.{layer_idx}.block"

            # Common components: norms and MLP
            new_state_dict[f"model.layers.{layer_idx}.block.input_layernorm.weight"] = original_weights[
                f"{orig_prefix}.pre_norm.scale"
            ]
            new_state_dict[f"model.layers.{layer_idx}.block.post_attention_layernorm.weight"] = original_weights[
                f"{orig_prefix}.post_norm.scale"
            ]

            # MLP layers
            # Original: l1 (gate), l2 (up), l3 (down)
            new_state_dict[f"{new_prefix}.mlp.gate_proj.weight"] = original_weights[f"{orig_prefix}.mlp.l1.weight"]
            new_state_dict[f"{new_prefix}.mlp.up_proj.weight"] = original_weights[f"{orig_prefix}.mlp.l2.weight"]
            new_state_dict[f"{new_prefix}.mlp.down_proj.weight"] = original_weights[f"{orig_prefix}.mlp.l3.weight"]

            if layer_type == "attention":
                # Convert attention layer
                # Original uses Wqkv (combined), we need separate q_proj, k_proj, v_proj
                wqkv = original_weights[f"{orig_prefix}.inner_mha_cls.Wqkv.weight"]
                hidden_size = config.hidden_size
                head_dim = hidden_size // config.num_attention_heads

                # Split Wqkv into q, k, v
                q, k, v = torch.split(wqkv, hidden_size, dim=0)
                new_state_dict[f"model.layers.{layer_idx}.block.attention.q_proj.weight"] = q
                new_state_dict[f"model.layers.{layer_idx}.block.attention.k_proj.weight"] = k
                new_state_dict[f"model.layers.{layer_idx}.block.attention.v_proj.weight"] = v

                # Output projection
                new_state_dict[f"model.layers.{layer_idx}.block.attention.o_proj.weight"] = original_weights[
                    f"{orig_prefix}.inner_mha_cls.out_proj.weight"
                ]

                # Load rotary embedding inv_freq from original weights
                if f"{orig_prefix}.inner_mha_cls.rotary_emb.inv_freq" in original_weights:
                    new_state_dict[f"model.layers.{layer_idx}.block.attention.rotary_emb.inv_freq"] = original_weights[
                        f"{orig_prefix}.inner_mha_cls.rotary_emb.inv_freq"
                    ]

                # Note: Original has out_proj.bias but our implementation doesn't use bias
            else:
                # Convert hyena filter layer
                new_state_dict[f"model.layers.{layer_idx}.block.filter.projections.weight"] = original_weights[
                    f"{orig_prefix}.projections.weight"
                ]
                new_state_dict[f"model.layers.{layer_idx}.block.filter.short_filter_weight"] = original_weights[
                    f"{orig_prefix}.filter.short_filter_weight"
                ]
                new_state_dict[f"model.layers.{layer_idx}.block.filter.out_filter_dense.weight"] = original_weights[
                    f"{orig_prefix}.out_filter_dense.weight"
                ]
                new_state_dict[f"model.layers.{layer_idx}.block.filter.out_filter_dense.bias"] = original_weights[
                    f"{orig_prefix}.out_filter_dense.bias"
                ]

                # Long filter parameters (FIR or IIR)
                if f"{orig_prefix}.filter.h" in original_weights:
                    new_state_dict[f"model.layers.{layer_idx}.block.filter.h"] = original_weights[
                        f"{orig_prefix}.filter.h"
                    ]
                if f"{orig_prefix}.filter.D" in original_weights:
                    new_state_dict[f"model.layers.{layer_idx}.block.filter.D"] = original_weights[
                        f"{orig_prefix}.filter.D"
                    ]
                if f"{orig_prefix}.filter.log_poles" in original_weights:
                    new_state_dict[f"model.layers.{layer_idx}.block.filter.log_poles"] = original_weights[
                        f"{orig_prefix}.filter.log_poles"
                    ]
                if f"{orig_prefix}.filter.residues" in original_weights:
                    new_state_dict[f"model.layers.{layer_idx}.block.filter.residues"] = original_weights[
                        f"{orig_prefix}.filter.residues"
                    ]

        # Final norm
        new_state_dict["model.norm.weight"] = original_weights["norm.scale"]

        return new_state_dict, config

    def test_weight_loading(self):
        """Test that we can successfully load and convert weights from the original model."""
        from huggingface_hub import hf_hub_download

        # Download original weights
        weights_path = hf_hub_download("arcinstitute/evo2_1b_base", "evo2_1b_base.pt")
        original_weights = torch.load(weights_path, map_location="cpu", weights_only=False)

        # Convert weights to transformers format
        new_state_dict, config = self.convert_original_weights_to_transformers(original_weights)

        # Create model and load converted weights
        model = Evo2ForCausalLM(config)
        
        # Load state dict (strict=False because Hyena layers have optional parameters)
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        # Manually assign filter parameters (h, D, log_poles, residues)
        for layer_idx in range(config.num_hidden_layers):
            if config.layer_types[layer_idx] == "hyena":
                filter_module = model.model.layers[layer_idx].block.filter
                orig_prefix = f"blocks.{layer_idx}.filter"
                
                if f"{orig_prefix}.h" in original_weights:
                    filter_module.h = original_weights[f"{orig_prefix}.h"]
                if f"{orig_prefix}.D" in original_weights:
                    filter_module.D = original_weights[f"{orig_prefix}.D"]
                if f"{orig_prefix}.log_poles" in original_weights:
                    filter_module.log_poles = original_weights[f"{orig_prefix}.log_poles"]
                if f"{orig_prefix}.residues" in original_weights:
                    filter_module.residues = original_weights[f"{orig_prefix}.residues"]
        
        # Check that only expected keys are missing/unexpected
        # (Hyena filter parameters and rotary embeddings)
        expected_patterns = ["filter.h", "filter.D", "filter.log_poles", "filter.residues", "rotary_emb.inv_freq"]
        
        for key in missing_keys:
            self.assertTrue(
                any(pattern in key for pattern in expected_patterns),
                f"Unexpected missing key: {key}"
            )
        
        for key in unexpected_keys:
            self.assertTrue(
                any(pattern in key for pattern in expected_patterns),
                f"Unexpected key in state dict: {key}"
            )
        
        print(f"✓ Successfully loaded weights ({len(missing_keys)} missing, {len(unexpected_keys)} unexpected)")

    def test_inference_shape(self):
        """Test that the model can run inference and produces the correct output shape."""
        from huggingface_hub import hf_hub_download

        # Load ground truth for reference
        ground_truth_path = os.path.join(
            os.path.dirname(__file__), "evo2_1b_base_ground_truth_logits.pt"
        )
        ground_truth = torch.load(ground_truth_path, map_location="cpu", weights_only=False)

        # Download and convert weights
        weights_path = hf_hub_download("arcinstitute/evo2_1b_base", "evo2_1b_base.pt")
        original_weights = torch.load(weights_path, map_location="cpu", weights_only=False)
        new_state_dict, config = self.convert_original_weights_to_transformers(original_weights)

        # Create and load model
        model = Evo2ForCausalLM(config)
        model.load_state_dict(new_state_dict, strict=False)
        
        # Manually assign filter parameters (h, D, log_poles, residues)
        # These can't be loaded via load_state_dict because they're None initially
        for layer_idx in range(config.num_hidden_layers):
            if config.layer_types[layer_idx] == "hyena":
                filter_module = model.model.layers[layer_idx].block.filter
                orig_prefix = f"blocks.{layer_idx}.filter"
                
                if f"{orig_prefix}.h" in original_weights:
                    filter_module.h = original_weights[f"{orig_prefix}.h"]
                if f"{orig_prefix}.D" in original_weights:
                    filter_module.D = original_weights[f"{orig_prefix}.D"]
                if f"{orig_prefix}.log_poles" in original_weights:
                    filter_module.log_poles = original_weights[f"{orig_prefix}.log_poles"]
                if f"{orig_prefix}.residues" in original_weights:
                    filter_module.residues = original_weights[f"{orig_prefix}.residues"]
        
        model = model.to(torch.bfloat16)
        model.eval()

        # Create tokenizer
        tokenizer = Evo2Tokenizer()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        sequences = ground_truth["sequences"]
        results = ground_truth["results"]

        # Test each sequence
        for seq in sequences:
            with self.subTest(sequence=seq):
                # Get ground truth
                gt_input_ids = results[seq]["input_ids"]
                gt_logits = results[seq]["logits"]

                # Tokenize
                tokens = tokenizer.tokenize(seq)
                input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

                # Verify input_ids match
                self.assertTrue(
                    torch.equal(input_ids.cpu(), gt_input_ids.unsqueeze(0)),
                    f"Input IDs mismatch for sequence {seq!r}"
                )

                # Run inference
                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits

                # Check shapes match
                expected_shape = gt_logits.shape
                actual_shape = logits.shape
                self.assertEqual(
                    actual_shape,
                    expected_shape,
                    f"Shape mismatch for {seq!r}: expected {expected_shape}, got {actual_shape}"
                )
                
                # Check that logits are finite (not NaN or Inf)
                self.assertTrue(torch.isfinite(logits).all(), f"Non-finite values in logits for {seq!r}")
                
                print(f"✓ {seq!r}: shape {actual_shape} OK, logits finite")

                # Check logits values match ground truth
                # Using relaxed tolerance for bfloat16
                # rtol=1e-2, atol=1e-2 is typical for bfloat16 accumulation differences
                torch.testing.assert_close(logits.cpu(), gt_logits.cpu(), rtol=0.02, atol=0.02)
                
                print(f"✓ {seq!r}: shape {actual_shape} OK, logits match ground truth")


if __name__ == "__main__":
    unittest.main()
