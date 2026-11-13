# Copyright 2025 Arc Institute and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Evo 2 model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch
    from torch import nn

    from transformers import Evo2ForCausalLM, Evo2Model, Evo2Tokenizer
    from transformers.models.evo2.convert_evo2_weights import convert_original_weights_to_transformers


class Evo2ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Evo2Model

    def __init__(
        self,
        parent,
        mixer_types=["hcs", "attention"],
        short_conv_kernel_size=3,
        hcs_filter_length=4,
        hcm_filter_length=8,
        filter_groups=4,
        state_size=4,
        **kwargs,
    ):
        super().__init__(
            parent,
            mixer_types=mixer_types,
            short_conv_kernel_size=short_conv_kernel_size,
            hcs_filter_length=hcs_filter_length,
            hcm_filter_length=hcm_filter_length,
            filter_groups=filter_groups,
            state_size=state_size,
            **kwargs,
        )


@require_torch
class Evo2ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Evo2ModelTester
    _torch_compile_train_cls = Evo2ForCausalLM if is_torch_available() else None

    def _get_conv_state_shape(self, batch_size: int, config):
        return [
            (batch_size, 3 * config.hidden_size, config.short_conv_kernel_size - 1),
            (batch_size, config.hidden_size, config.hcs_filter_length - 1),
        ]

    def _get_recurrent_state_shape(self, batch_size: int, config):
        return (batch_size, config.hidden_size, config.state_size)

    def test_attention_outputs(self):
        """Evo2 alternates between Hyena conv/recurrence layers and full attention layers."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device).eval()
            config = model.config
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                self_attentions = outputs.attentions

            self.assertEqual(out_len + 1, len(outputs))
            self.assertEqual(len(self_attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])

    def test_mlp_activation_only_on_first_layer(self):
        config = self.model_tester.get_config()
        config.num_hidden_layers = 4
        config.mixer_types = ["hcs", "hcm", "hcl", "attention"]
        model = Evo2Model(config)
        self.assertNotIsInstance(model.layers[0].feed_forward.act_fn, nn.Identity)
        for layer in model.layers[1:]:
            self.assertIsInstance(layer.feed_forward.act_fn, nn.Identity)

    def test_cached_decoding_equivalence(self):
        """Verify that full-sequence forward matches prefill + token-by-token cached decoding across HCS, HCM, HCL, and attention layers."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_hidden_layers = 4
        config.mixer_types = ["hcs", "hcm", "hcl", "attention"]
        model = Evo2ForCausalLM(config).to(torch_device).eval()
        # Give HCL poles negative real parts so modal recurrence is stable
        with torch.no_grad():
            for layer in model.model.layers:
                if not layer.is_attention_layer and layer.conv.mixer_type == "hcl":
                    layer.conv.log_poles.copy_(-torch.abs(layer.conv.log_poles) - 0.1)

        input_ids = inputs_dict["input_ids"]
        with torch.no_grad():
            full_logits = model(input_ids, use_cache=False).logits

            prefill_len = 4
            prefill_out = model(input_ids[:, :prefill_len], use_cache=True)
            cache = prefill_out.cache
            step_logits = [prefill_out.logits]
            for t in range(prefill_len, input_ids.shape[1]):
                step_out = model(input_ids[:, t : t + 1], cache=cache, use_cache=True)
                cache = step_out.cache
                step_logits.append(step_out.logits)
            cached_logits = torch.cat(step_logits, dim=1)

        torch.testing.assert_close(full_logits, cached_logits, rtol=1e-4, atol=1e-4)

    def test_weight_conversion(self):
        """Verify that original Vortex state dict keys convert and load with strict=True across all mixer types."""
        hidden_size = 32
        intermediate_size = 64
        vocab_size = 64
        mixer_types = ["hcs", "hcm", "hcl", "attention"]
        orig = {
            "embedding_layer.weight": torch.randn(vocab_size, hidden_size),
            "unembed.weight": torch.randn(vocab_size, hidden_size),
            "norm.scale": torch.ones(hidden_size),
        }
        for i, mixer in enumerate(mixer_types):
            orig[f"blocks.{i}.pre_norm.scale"] = torch.ones(hidden_size)
            orig[f"blocks.{i}.post_norm.scale"] = torch.ones(hidden_size)
            orig[f"blocks.{i}.mlp.l1.weight"] = torch.randn(intermediate_size, hidden_size)
            orig[f"blocks.{i}.mlp.l2.weight"] = torch.randn(intermediate_size, hidden_size)
            orig[f"blocks.{i}.mlp.l3.weight"] = torch.randn(hidden_size, intermediate_size)
            if mixer == "attention":
                orig[f"blocks.{i}.inner_mha_cls.Wqkv.weight"] = torch.randn(3 * hidden_size, hidden_size)
                orig[f"blocks.{i}.inner_mha_cls.out_proj.weight"] = torch.randn(hidden_size, hidden_size)
                orig[f"blocks.{i}.inner_mha_cls.out_proj.bias"] = torch.randn(hidden_size)
                orig[f"blocks.{i}.inner_mha_cls.rotary_emb.inv_freq"] = torch.randn(4)
            else:
                orig[f"blocks.{i}.projections.weight"] = torch.randn(3 * hidden_size, hidden_size)
                orig[f"blocks.{i}.filter.short_filter_weight"] = torch.randn(3 * hidden_size, 1, 3)
                orig[f"blocks.{i}.out_filter_dense.weight"] = torch.randn(hidden_size, hidden_size)
                orig[f"blocks.{i}.out_filter_dense.bias"] = torch.randn(hidden_size)
                if mixer == "hcs":
                    orig[f"blocks.{i}.filter.h"] = torch.randn(4, 1, 4)
                elif mixer == "hcm":
                    orig[f"blocks.{i}.filter.h"] = torch.randn(4, 1, 8)
                    orig[f"blocks.{i}.filter.D"] = torch.randn(hidden_size)
                elif mixer == "hcl":
                    orig[f"blocks.{i}.filter.log_poles"] = torch.randn(hidden_size, 4, 1)
                    orig[f"blocks.{i}.filter.residues"] = torch.randn(hidden_size, 4)
                    orig[f"blocks.{i}.filter.D"] = torch.randn(hidden_size)

        new_state_dict, config = convert_original_weights_to_transformers(orig)
        self.assertEqual(config.mixer_types, mixer_types)
        model = Evo2ForCausalLM(config)
        model.load_state_dict(new_state_dict, strict=True)


@require_torch_accelerator
@slow
class Evo2InferenceTest(unittest.TestCase):
    """Test inference against ground truth logits from the original evo2_1b_base model."""

    def tearDown(self):
        super().tearDown()
        cleanup(torch_device, gc_collect=True)

    def test_inference_ground_truth(self):
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download("arcinstitute/evo2_1b_base", "evo2_1b_base.pt")
        original_weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        new_state_dict, config = convert_original_weights_to_transformers(original_weights)

        model = Evo2ForCausalLM(config)
        model.load_state_dict(new_state_dict, strict=True)
        model = model.to(device=torch_device, dtype=torch.bfloat16).eval()

        tokenizer = Evo2Tokenizer()
        nucleotide_indices = [65, 67, 71, 84]  # 'A', 'C', 'G', 'T'

        expected_nucleotide_logits = {
            "ACGT": torch.tensor(
                [
                    [0.91796875, 1.0234375, 0.828125, 0.8046875],
                    [0.984375, 0.8515625, 0.6953125, 0.90625],
                    [0.8828125, 0.84375, 0.7734375, 0.9140625],
                    [0.6171875, 0.9921875, 0.8203125, 1.0078125],
                ]
            ),
            "ACGTACGTACGT": torch.tensor(
                [
                    [0.921875, 1.0234375, 0.828125, 0.8046875],
                    [0.98828125, 0.8515625, 0.6953125, 0.90625],
                    [0.890625, 0.84375, 0.765625, 0.91015625],
                    [0.62109375, 0.9921875, 0.8203125, 1.0078125],
                ]
            ),
            "AAAAAAAAAA": torch.tensor(
                [
                    [0.921875, 1.015625, 0.828125, 0.8046875],
                    [1.0859375, 0.73828125, 0.66796875, 0.94921875],
                    [1.125, 0.66796875, 0.63671875, 0.99609375],
                    [1.1328125, 0.68359375, 0.62109375, 0.984375],
                ]
            ),
        }

        for seq, expected_slice in expected_nucleotide_logits.items():
            input_ids = tokenizer(seq, return_tensors="pt").input_ids.to(torch_device)
            with torch.no_grad():
                logits = model(input_ids).logits
            self.assertEqual(logits.shape, (1, len(seq), config.vocab_size))
            actual_slice = logits[0, :4, nucleotide_indices].float().cpu()
            torch.testing.assert_close(actual_slice, expected_slice, rtol=0.02, atol=0.02)


if __name__ == "__main__":
    unittest.main()
