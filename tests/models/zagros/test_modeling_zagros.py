import unittest
import torch

from transformers import ZagrosConfig, ZagrosModel, ZagrosForCausalLM

class TestZagrosModeling(unittest.TestCase):
    def setUp(self):
        self.config = ZagrosConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            vocab_size=1000,
            pad_token_id=0,
            num_experts=4,
            num_experts_per_tok=2,
            rms_norm_eps=1e-6,
            max_position_embeddings=128,
        )
        self.batch_size = 2
        self.seq_length = 10

    def test_zagros_model_initialization(self):
        model = ZagrosModel(self.config)
        self.assertIsNotNone(model)
        self.assertEqual(model.config, self.config)

    def test_zagros_model_forward_pass(self):
        model = ZagrosModel(self.config)
        input_ids = torch.ones((self.batch_size, self.seq_length), dtype=torch.long)
        attention_mask = torch.ones((self.batch_size, self.seq_length), dtype=torch.long)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        self.assertIsNotNone(outputs)
        self.assertEqual(outputs.last_hidden_state.shape, (self.batch_size, self.seq_length, self.config.hidden_size))

    def test_zagros_causal_lm_initialization(self):
        model = ZagrosForCausalLM(self.config)
        self.assertIsNotNone(model)
        self.assertEqual(model.config, self.config)

    def test_zagros_causal_lm_forward_pass(self):
        model = ZagrosForCausalLM(self.config)
        input_ids = torch.ones((self.batch_size, self.seq_length), dtype=torch.long)
        attention_mask = torch.ones((self.batch_size, self.seq_length), dtype=torch.long)
        labels = torch.zeros((self.batch_size, self.seq_length), dtype=torch.long)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        self.assertIsNotNone(outputs)
        self.assertEqual(outputs.logits.shape, (self.batch_size, self.seq_length, self.config.vocab_size))
        self.assertIsNotNone(outputs.loss)

    def test_rotary_embedding(self):
        from transformers.models.zagros.modeling_zagros import ZagrosRotaryEmbedding
        rotary_emb = ZagrosRotaryEmbedding(self.config)
        position_ids = torch.arange(self.seq_length).expand((self.batch_size, -1))
        cos, sin = rotary_emb(torch.zeros((self.batch_size, self.seq_length, self.config.hidden_size)), position_ids)
        self.assertEqual(cos.shape, (self.batch_size, self.seq_length, self.config.hidden_size // self.config.num_attention_heads))
        self.assertEqual(sin.shape, (self.batch_size, self.seq_length, self.config.hidden_size // self.config.num_attention_heads))

    def test_attention_layer(self):
        from transformers.models.zagros.modeling_zagros import ZagrosAttention
        attention = ZagrosAttention(self.config, layer_idx=0)
        hidden_states = torch.zeros((self.batch_size, self.seq_length, self.config.hidden_size))
        position_embeddings = (torch.ones((self.batch_size, self.seq_length, self.config.head_dim)), torch.zeros((self.batch_size, self.seq_length, self.config.head_dim)))
        outputs, attn_weights = attention(hidden_states, position_embeddings, attention_mask=None)
        self.assertEqual(outputs.shape, (self.batch_size, self.seq_length, self.config.hidden_size))
        self.assertIsNotNone(attn_weights)

if __name__ == "__main__":
    unittest.main()
