import unittest
import torch

from transformers import ZagrosNextConfig, ZagrosNextModel, ZagrosNextForCausalLM
from transformers.models.zagros_next.modeling_zagros_next import ZagrosNextDynamicCache

class TestZagrosNextModeling(unittest.TestCase):
    def setUp(self):
        self.config = ZagrosNextConfig(
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            vocab_size=1000,
            pad_token_id=0,
            num_experts=4,
            num_experts_per_tok=2,
            rms_norm_eps=1e-6,
            max_position_embeddings=128,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=3,
            layer_types=["full_attention", "linear_attention", "full_attention", "linear_attention"],
        )
        self.batch_size = 2
        self.seq_length = 10

    def test_zagros_next_model_initialization(self):
        model = ZagrosNextModel(self.config)
        self.assertIsNotNone(model)
        self.assertEqual(model.config, self.config)

    def test_zagros_next_model_forward_pass(self):
        model = ZagrosNextModel(self.config)
        input_ids = torch.ones((self.batch_size, self.seq_length), dtype=torch.long)
        attention_mask = torch.ones((self.batch_size, self.seq_length), dtype=torch.long)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        self.assertIsNotNone(outputs)
        self.assertEqual(outputs.last_hidden_state.shape, (self.batch_size, self.seq_length, self.config.hidden_size))

    def test_zagros_next_causal_lm_initialization(self):
        model = ZagrosNextForCausalLM(self.config)
        self.assertIsNotNone(model)
        self.assertEqual(model.config, self.config)

    def test_zagros_next_causal_lm_forward_pass(self):
        model = ZagrosNextForCausalLM(self.config)
        input_ids = torch.ones((self.batch_size, self.seq_length), dtype=torch.long)
        attention_mask = torch.ones((self.batch_size, self.seq_length), dtype=torch.long)
        labels = torch.zeros((self.batch_size, self.seq_length), dtype=torch.long)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        self.assertIsNotNone(outputs)
        self.assertEqual(outputs.logits.shape, (self.batch_size, self.seq_length, self.config.vocab_size))
        self.assertIsNotNone(outputs.loss)

    def test_rotary_embedding(self):
        from transformers.models.zagros_next.modeling_zagros_next import ZagrosNextRotaryEmbedding
        rotary_emb = ZagrosNextRotaryEmbedding(self.config)
        position_ids = torch.arange(self.seq_length).expand((self.batch_size, -1))
        cos, sin = rotary_emb(torch.zeros((self.batch_size, self.seq_length, self.config.hidden_size)), position_ids)
        self.assertEqual(cos.shape, (self.batch_size, self.seq_length, self.config.hidden_size // self.config.num_attention_heads))
        self.assertEqual(sin.shape, (self.batch_size, self.seq_length, self.config.hidden_size // self.config.num_attention_heads))

    def test_attention_layer(self):
        from transformers.models.zagros_next.modeling_zagros_next import ZagrosNextAttention
        attention = ZagrosNextAttention(self.config, layer_idx=0)
        hidden_states = torch.zeros((self.batch_size, self.seq_length, self.config.hidden_size))
        position_embeddings = (torch.ones((self.batch_size, self.seq_length, self.config.head_dim)), torch.zeros((self.batch_size, self.seq_length, self.config.head_dim)))
        outputs, attn_weights = attention(hidden_states, position_embeddings, attention_mask=None)
        self.assertEqual(outputs.shape, (self.batch_size, self.seq_length, self.config.hidden_size))
        self.assertIsNotNone(attn_weights)

    def test_gated_delta_net_layer(self):
        from transformers.models.zagros_next.modeling_zagros_next import ZagrosNextGatedDeltaNet
        delta_net = ZagrosNextGatedDeltaNet(self.config, layer_idx=1)
        hidden_states = torch.zeros((self.batch_size, self.seq_length, self.config.hidden_size))
        cache = ZagrosNextDynamicCache(self.config)
        outputs = delta_net(hidden_states, cache, cache_position=torch.arange(self.seq_length))
        self.assertEqual(outputs.shape, (self.batch_size, self.seq_length, self.config.hidden_size))

if __name__ == "__main__":
    unittest.main()
    