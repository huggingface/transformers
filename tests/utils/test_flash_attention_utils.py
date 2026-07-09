# Copyright 2025 HuggingFace Inc. team.
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
import torch

from transformers.modeling_flash_attention_utils import _get_unpad_data, _unpad_input
from transformers.utils.import_utils import is_torchdynamo_compiling


class FlashAttentionUtilsTest(unittest.TestCase):
    def test_get_unpad_data_returns_scalar_when_not_compiling(self):
        """
        Regression test for #46693: _get_unpad_data should return max_seqlen_in_batch
        as a Python int (not a 0-dim tensor) when not under torch.compile,
        to avoid performance regression on large KV caches.
        """
        # Skip if we're somehow compiling during test collection
        if is_torchdynamo_compiling():
            self.skipTest("Test not valid under torch.compile")
        
        batch_size = 4
        seq_len = 1024
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create attention mask (all ones = no padding)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int32, device=device)
        
        indices, cu_seqlens, max_seqlen_in_batch = _get_unpad_data(attention_mask)
        
        # max_seqlen_in_batch should be a Python int, not a tensor
        self.assertIsInstance(max_seqlen_in_batch, int, 
            f"Expected max_seqlen_in_batch to be int, got {type(max_seqlen_in_batch)}")
        self.assertEqual(max_seqlen_in_batch, seq_len)
        
        # indices and cu_seqlens should still be tensors
        self.assertIsInstance(indices, torch.Tensor)
        self.assertIsInstance(cu_seqlens, torch.Tensor)

    def test_unpad_input_returns_scalar_when_not_compiling(self):
        """
        Regression test for #46693: _unpad_input should return max_seqlen_in_batch
        as a Python int when not under torch.compile.
        """
        if is_torchdynamo_compiling():
            self.skipTest("Test not valid under torch.compile")
        
        batch_size = 2
        seq_len = 512
        hidden_dim = 64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int32, device=device)
        
        _, _, _, max_seqlen_in_batch, _ = _unpad_input(hidden_states, attention_mask)
        
        self.assertIsInstance(max_seqlen_in_batch, int,
            f"Expected max_seqlen_in_batch to be int, got {type(max_seqlen_in_batch)}")
        self.assertEqual(max_seqlen_in_batch, seq_len)

    def test_get_unpad_data_with_padding(self):
        """Test _get_unpad_data with variable sequence lengths (padding)."""
        if is_torchdynamo_compiling():
            self.skipTest("Test not valid under torch.compile")
        
        batch_size = 4
        seq_len = 1024
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create attention mask with padding (last 256 tokens masked)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int32, device=device)
        attention_mask[:, -256:] = 0
        
        indices, cu_seqlens, max_seqlen_in_batch = _get_unpad_data(attention_mask)
        
        self.assertIsInstance(max_seqlen_in_batch, int)
        self.assertEqual(max_seqlen_in_batch, seq_len - 256)
        self.assertEqual(cu_seqlens.shape[0], batch_size + 1)
        self.assertEqual(cu_seqlens[-1].item(), batch_size * (seq_len - 256))


if __name__ == "__main__":
    unittest.main()