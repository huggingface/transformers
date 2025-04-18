import unittest
from typing import Any, Dict, Optional, Tuple

import torch

from transformers.cache_utils import Cache


class TestCache(unittest.TestCase):
    def setUp(self):
        # Create a simple Cache subclass for testing
        class TestCacheImpl(Cache):
            def __init__(self):
                super().__init__()
                self.key_cache = []
                self.value_cache = []
                self._seen_tokens = 0
                self.update_cache = True

            def update(
                self,
                key_states: torch.Tensor,
                value_states: torch.Tensor,
                layer_idx: int,
                cache_kwargs: Optional[Dict[str, Any]] = None,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                # Update the number of seen tokens
                if layer_idx == 0:
                    self._seen_tokens += key_states.shape[-2]

                # Update the cache
                if key_states is not None:
                    if len(self.key_cache) <= layer_idx:
                        # There may be skipped layers, fill them with empty lists
                        for _ in range(len(self.key_cache), layer_idx):
                            self.key_cache.append([])
                            self.value_cache.append([])
                        self.key_cache.append(key_states)
                        self.value_cache.append(value_states)
                    elif (
                        len(self.key_cache[layer_idx]) == 0 and self.update_cache
                    ):  # fills previously skipped layers; checking for tensor causes errors
                        self.key_cache[layer_idx] = key_states
                        self.value_cache[layer_idx] = value_states
                    elif (
                        len(self.key_cache[layer_idx]) == 0 and not self.update_cache
                    ):  # fills previously skipped layers; checking for tensor causes errors
                        return key_states, value_states
                    elif self.update_cache:
                        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                    else:
                        new_keys = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                        new_values = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                        return new_keys, new_values

                return self.key_cache[layer_idx], self.value_cache[layer_idx]

        self.cache = TestCacheImpl()

    def test_update_first_layer(self):
        # Test updating the first layer
        batch_size = 2
        num_heads = 4
        seq_len = 3
        head_dim = 8

        key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)

        updated_keys, updated_values = self.cache.update(key_states, value_states, layer_idx=0)

        # Check if cache is updated correctly
        self.assertEqual(len(self.cache.key_cache), 1)
        self.assertEqual(len(self.cache.value_cache), 1)
        self.assertEqual(self.cache._seen_tokens, seq_len)
        self.assertTrue(torch.allclose(updated_keys, key_states))
        self.assertTrue(torch.allclose(updated_values, value_states))

    def test_update_multiple_layers(self):
        # Test updating multiple layers
        batch_size = 2
        num_heads = 4
        seq_len = 3
        head_dim = 8

        # Update layer 0
        key_states_0 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value_states_0 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        self.cache.update(key_states_0, value_states_0, layer_idx=0)

        # Update layer 2 (skip layer 1)
        key_states_2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value_states_2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        updated_keys, updated_values = self.cache.update(key_states_2, value_states_2, layer_idx=2)

        # Check if cache is updated correctly
        self.assertEqual(len(self.cache.key_cache), 3)  # Should have 3 layers (0,1,2)
        self.assertEqual(len(self.cache.value_cache), 3)
        self.assertTrue(isinstance(self.cache.key_cache[1], list))  # Layer 1 should be an empty list
        self.assertTrue(isinstance(self.cache.value_cache[1], list))
        self.assertTrue(torch.allclose(updated_keys, key_states_2))
        self.assertTrue(torch.allclose(updated_values, value_states_2))

    def test_update_with_concatenation(self):
        # Test tensor concatenation
        batch_size = 2
        num_heads = 4
        seq_len = 3
        head_dim = 8

        # First update
        key_states_1 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value_states_1 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        self.cache.update(key_states_1, value_states_1, layer_idx=0)

        # Second update
        key_states_2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value_states_2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        updated_keys, updated_values = self.cache.update(key_states_2, value_states_2, layer_idx=0)

        # Check if concatenation is correct
        expected_keys = torch.cat([key_states_1, key_states_2], dim=-2)
        expected_values = torch.cat([value_states_1, value_states_2], dim=-2)
        self.assertTrue(torch.allclose(updated_keys, expected_keys))
        self.assertTrue(torch.allclose(updated_values, expected_values))

    def test_update_without_update_cache(self):
        # Test behavior when update_cache=False
        self.cache.update_cache = False

        batch_size = 2
        num_heads = 4
        seq_len = 3
        head_dim = 8

        # First update
        key_states_1 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value_states_1 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        self.cache.update(key_states_1, value_states_1, layer_idx=0)

        # Second update
        key_states_2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value_states_2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        updated_keys, updated_values = self.cache.update(key_states_2, value_states_2, layer_idx=0)

        # Check if new tensors are correctly returned but cache remains unchanged
        expected_keys = torch.cat([key_states_1, key_states_2], dim=-2)
        expected_values = torch.cat([value_states_1, value_states_2], dim=-2)
        self.assertTrue(torch.allclose(updated_keys, expected_keys))
        self.assertTrue(torch.allclose(updated_values, expected_values))
        self.assertTrue(torch.allclose(self.cache.key_cache[0], key_states_1))  # Cache should remain unchanged
        self.assertTrue(torch.allclose(self.cache.value_cache[0], value_states_1))
