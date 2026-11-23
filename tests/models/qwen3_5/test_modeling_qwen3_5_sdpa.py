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
from unittest.mock import patch

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch, require_torch_gpu, torch_device


if is_torch_available():
    import torch

    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig, Qwen3_5TextModel
    from transformers.integrations import sdpa_attention
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling


@require_torch
class Qwen3_5PaddingSDPATest(unittest.TestCase):
    def config(self, hybrid=False, **kwargs):
        return Qwen3_5TextConfig(
            vocab_size=97,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            layer_types=["full_attention", "linear_attention" if hybrid else "full_attention"],
            linear_conv_kernel_dim=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            rope_parameters={"rope_type": "default", "rope_theta": 10000, "mrope_section": [2, 1, 1]},
            use_cache=False,
            **kwargs,
        )

    def mask_mapping(self, model, embeds, mask):
        args = {"config": model.config, "inputs_embeds": embeds, "attention_mask": mask, "cache": None}
        return {
            "full_attention": create_causal_mask(**args),
            "linear_attention": create_recurrent_attention_mask(**args),
        }

    @parameterized.expand([(1,), (7,), (15,), (16,)])
    def test_attention_outputs_and_gradients(self, prefix_length):
        self.check_attention(prefix_length, torch.float32)

    @parameterized.expand([(1,), (7,), (15,), (16,)])
    @require_torch_gpu
    def test_attention_outputs_and_gradients_bfloat16(self, prefix_length):
        self.check_attention(prefix_length, torch.bfloat16)

    def check_attention(self, prefix_length, dtype):
        torch.manual_seed(42)
        config = self.config()
        config._attn_implementation = "sdpa"
        reference = modeling.Qwen3_5Attention(config, 0).to(device=torch_device, dtype=dtype)
        optimized = copy.deepcopy(reference)
        x = torch.randn(1, 16, 64, device=torch_device, dtype=dtype, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        angles = torch.randn(1, 16, 16, device=torch_device, dtype=dtype)
        positions = (angles.cos(), angles.sin())
        mask = torch.ones(1, 16, device=torch_device, dtype=torch.long)
        mask[:, prefix_length:] = 0
        causal_mask = create_causal_mask(config=config, inputs_embeds=x, attention_mask=mask, cache=None)
        ref = reference(x, positions, causal_mask)[0]
        out = optimized(y, positions, None, _sdpa_prefix_length=prefix_length)[0]
        # Include padded queries in the loss: discarding them would hide incorrect K/V gradients.
        upstream_grad = torch.randn_like(ref)
        ref.backward(upstream_grad)
        out.backward(upstream_grad)
        atol, rtol = (1e-5, 1e-4) if dtype == torch.float32 else (2e-2, 3e-2)
        for actual, expected in [(out, ref), (y.grad, x.grad)]:
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        for (name, actual), (_, expected) in zip(optimized.named_parameters(), reference.named_parameters()):
            self.assertIsNotNone(actual.grad, name)
            torch.testing.assert_close(actual.grad, expected.grad, atol=atol, rtol=rtol, msg=name)

    @parameterized.expand([(False,), (True,), ("reentrant",)])
    def test_model_loss_gradients_and_update(self, checkpointing):
        self.check_model(checkpointing, hybrid=False, dtype=torch.float32)

    @parameterized.expand([(False,), (True,), ("reentrant",)])
    @require_torch_gpu
    def test_hybrid_model_loss_gradients_and_update(self, checkpointing):
        self.check_model(checkpointing, hybrid=True, dtype=torch.bfloat16)

    def check_model(self, checkpointing, hybrid, dtype):
        torch.manual_seed(42)
        reference = Qwen3_5ForCausalLM(self.config(hybrid=hybrid)).to(device=torch_device, dtype=dtype).train()
        optimized = copy.deepcopy(reference)
        if checkpointing:
            for model in (reference, optimized):
                model.activation_checkpointing_enable(
                    activation_checkpointing_kwargs={"use_reentrant": checkpointing == "reentrant"}
                )
        ref_loss = out_loss = 0
        for prefix_length in (7, 13):
            ids = torch.randint(0, 97, (1, 16), device=torch_device)
            mask = torch.ones_like(ids)
            mask[:, prefix_length:] = 0
            labels = ids.masked_fill(mask == 0, -100)
            mapping = self.mask_mapping(reference, reference.get_input_embeddings()(ids), mask)
            ref = reference(ids, attention_mask=mapping, labels=labels)
            out = optimized(ids, attention_mask=mask, labels=labels)
            atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (3e-3, 3e-2)
            torch.testing.assert_close(out.logits, ref.logits, atol=atol, rtol=rtol)
            torch.testing.assert_close(out.loss, ref.loss, atol=atol, rtol=rtol)
            ref_loss = ref_loss + ref.loss
            out_loss = out_loss + out.loss
        # Two outstanding forwards with distinct lengths exercise checkpoint recomputation.
        ref_loss.backward()
        out_loss.backward()
        for (name, actual), (_, expected) in zip(optimized.named_parameters(), reference.named_parameters()):
            self.assertEqual(actual.grad is None, expected.grad is None, name)
            if actual.grad is not None:
                torch.testing.assert_close(actual.grad, expected.grad, atol=atol, rtol=rtol, msg=name)
        for model in (reference, optimized):
            torch.optim.SGD(model.parameters(), lr=0.01).step()
        for (name, actual), (_, expected) in zip(optimized.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=name)

    @parameterized.expand(
        [
            ("left",),
            ("holes",),
            ("empty",),
            ("nonbinary",),
            ("batch",),
            ("mapping",),
            ("dropout",),
            ("cache",),
            ("eager",),
            ("kwargs",),
            ("attentions",),
            ("compiling",),
        ]
    )
    def test_unsupported_inputs_keep_original_attention(self, case):
        config = self.config(attention_dropout=0.1 if case == "dropout" else 0.0)
        config._attn_implementation = "eager" if case == "eager" else "sdpa"
        model = Qwen3_5TextModel(config).to(torch_device).eval()
        ids = torch.ones(2 if case == "batch" else 1, 8, device=torch_device, dtype=torch.long)
        mask = torch.ones_like(ids)
        mask[:, 5:] = 0
        kwargs = {}
        if case == "left":
            mask = mask.flip(-1)
        elif case == "holes":
            mask[:, 2] = 0
        elif case == "empty":
            mask.zero_()
        elif case == "nonbinary":
            mask[:, 2] = 2
        elif case == "mapping":
            mask = self.mask_mapping(model, model.embed_tokens(ids), mask)
        elif case == "cache":
            kwargs["use_cache"] = True
        elif case == "kwargs":
            kwargs["is_causal"] = False
        elif case == "attentions":
            kwargs["output_attentions"] = True
        attention = model.layers[0].self_attn
        with (
            patch.object(attention, "forward", wraps=attention.forward) as forward,
            patch.object(sdpa_attention, "is_torchdynamo_compiling", return_value=case == "compiling"),
            torch.no_grad(),
        ):
            model(ids, attention_mask=mask, **kwargs)
        self.assertIsNone(forward.call_args.kwargs.get("_sdpa_prefix_length"))

    def test_fast_path_preserves_recurrent_mask(self):
        model = Qwen3_5TextModel(self.config()).to(torch_device).eval()
        ids = torch.ones(1, 8, device=torch_device, dtype=torch.long)
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], device=torch_device)
        attention = model.layers[0].self_attn
        with (
            patch.object(attention, "forward", wraps=attention.forward) as forward,
            patch.object(modeling, "create_causal_mask", wraps=modeling.create_causal_mask) as causal,
            patch.object(
                modeling, "create_recurrent_attention_mask", wraps=modeling.create_recurrent_attention_mask
            ) as recurrent,
            torch.no_grad(),
        ):
            model(ids, attention_mask=mask)
        self.assertEqual(forward.call_args.kwargs["_sdpa_prefix_length"], 5)
        causal.assert_not_called()
        recurrent.assert_called_once()
