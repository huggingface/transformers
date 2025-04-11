import unittest
import torch
from transformers.cache_utils import Cache
from typing import Optional, Dict, Tuple, Any

class TestCache(unittest.TestCase):
    def setUp(self):
        # 创建一个简单的 Cache 子类用于测试
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
        # 测试第一层的更新
        batch_size = 2
        num_heads = 4
        seq_len = 3
        head_dim = 8

        key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)

        updated_keys, updated_values = self.cache.update(key_states, value_states, layer_idx=0)

        # 检查缓存是否正确更新
        self.assertEqual(len(self.cache.key_cache), 1)
        self.assertEqual(len(self.cache.value_cache), 1)
        self.assertEqual(self.cache._seen_tokens, seq_len)
        self.assertTrue(torch.allclose(updated_keys, key_states))
        self.assertTrue(torch.allclose(updated_values, value_states))

    def test_update_multiple_layers(self):
        # 测试多层更新
        batch_size = 2
        num_heads = 4
        seq_len = 3
        head_dim = 8

        # 更新第0层
        key_states_0 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value_states_0 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        self.cache.update(key_states_0, value_states_0, layer_idx=0)

        # 更新第2层（跳过第1层）
        key_states_2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value_states_2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        updated_keys, updated_values = self.cache.update(key_states_2, value_states_2, layer_idx=2)

        # 检查缓存是否正确更新
        self.assertEqual(len(self.cache.key_cache), 3)  # 应该有3层（0,1,2）
        self.assertEqual(len(self.cache.value_cache), 3)
        self.assertTrue(isinstance(self.cache.key_cache[1], list))  # 第1层应该是空列表
        self.assertTrue(isinstance(self.cache.value_cache[1], list))
        self.assertTrue(torch.allclose(updated_keys, key_states_2))
        self.assertTrue(torch.allclose(updated_values, value_states_2))

    def test_update_with_concatenation(self):
        # 测试张量连接
        batch_size = 2
        num_heads = 4
        seq_len = 3
        head_dim = 8

        # 第一次更新
        key_states_1 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value_states_1 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        self.cache.update(key_states_1, value_states_1, layer_idx=0)

        # 第二次更新
        key_states_2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value_states_2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        updated_keys, updated_values = self.cache.update(key_states_2, value_states_2, layer_idx=0)

        # 检查是否正确连接
        expected_keys = torch.cat([key_states_1, key_states_2], dim=-2)
        expected_values = torch.cat([value_states_1, value_states_2], dim=-2)
        self.assertTrue(torch.allclose(updated_keys, expected_keys))
        self.assertTrue(torch.allclose(updated_values, expected_values))

    def test_update_without_update_cache(self):
        # 测试当 update_cache=False 时的行为
        self.cache.update_cache = False
        
        batch_size = 2
        num_heads = 4
        seq_len = 3
        head_dim = 8

        # 第一次更新
        key_states_1 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value_states_1 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        self.cache.update(key_states_1, value_states_1, layer_idx=0)

        # 第二次更新
        key_states_2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value_states_2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        updated_keys, updated_values = self.cache.update(key_states_2, value_states_2, layer_idx=0)

        # 检查是否正确返回连接后的新张量，但不更新缓存
        expected_keys = torch.cat([key_states_1, key_states_2], dim=-2)
        expected_values = torch.cat([value_states_1, value_states_2], dim=-2)
        self.assertTrue(torch.allclose(updated_keys, expected_keys))
        self.assertTrue(torch.allclose(updated_values, expected_values))
        self.assertTrue(torch.allclose(self.cache.key_cache[0], key_states_1))  # 缓存应该保持不变
        self.assertTrue(torch.allclose(self.cache.value_cache[0], value_states_1))

if __name__ == '__main__':
    unittest.main() 