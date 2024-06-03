import torch
from transformers.models.nemotron.modeling_nemotron import NemotronLayerNorm, NemotronMLP
from transformers.models.nemotron.configuration_nemotron import NemotronConfig

def get_nemotron_config():
   config = NemotronConfig(
        vocab_size=256000, # TODO
        hidden_size=4096,
        intermediate_size=16384,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="squared-relu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        normalization='layernorm1p',
        norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None, #TODO
        bos_token_id=1, # TODO
        eos_token_id=2, # TODO
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        rope_percentage=0.5,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
   )
   return config
       

def test_layernorm():
    hidden_size = 1024
    batch_size = 8
    norm = NemotronLayerNorm(hidden_size, normalization='layernorm1p')
    input = torch.ones((batch_size, hidden_size))
    output = norm(input)
    assert output.shape == input.shape

def test_rope():
    pass

def test_nemotron_mlp():
    

if __name__ == '__main__':
    test_layernorm()
    