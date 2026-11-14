# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from transformers.testing_utils import (
    is_torch_available,
    require_flex_attention,
    require_torch_accelerator,
    torch_device,
)


if is_torch_available():
    import torch
    from torch.nn.attention.flex_attention import create_block_mask

    from transformers.integrations.flex_attention import flex_attention_forward


@require_flex_attention
@require_torch_accelerator
class FlexAttentionForwardTest(unittest.TestCase):
    def test_head_dim_512_forward_and_backward(self):
        # Gemma 4 global layers use head_dim 512, where the default flex block sizes fault on Blackwell
        torch.manual_seed(0)
        batch_size, num_heads, num_kv_heads, seq_len, head_dim = 1, 8, 2, 1024, 512
        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=torch_device, dtype=torch.bfloat16)
        key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=torch_device, dtype=torch.bfloat16)
        value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=torch_device, dtype=torch.bfloat16)
        query.requires_grad_()
        block_mask = create_block_mask(lambda b, h, q, kv: q >= kv, None, None, seq_len, seq_len, device=torch_device)

        output, _ = flex_attention_forward(torch.nn.Module(), query, key, value, block_mask)
        output.float().sum().backward()

        reference_query = query.detach().float().requires_grad_()
        reference = torch.nn.functional.scaled_dot_product_attention(
            reference_query, key.float(), value.float(), is_causal=True, enable_gqa=True
        )
        reference.sum().backward()

        torch.testing.assert_close(output.float(), reference.transpose(1, 2), rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(query.grad.float(), reference_query.grad, rtol=2e-2, atol=2e-2)
