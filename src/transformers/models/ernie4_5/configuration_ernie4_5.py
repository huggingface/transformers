{
  "architectures": [
    "Ernie4_5_ForCausalLM"
  ],
  "auto_map": {
    "AutoConfig": "configuration_ernie4_5.Ernie4_5_Config",
    "AutoModel": "modeling_ernie4_5.Ernie4_5_Model",
    "AutoModelForCausalLM": "modeling_ernie4_5.Ernie4_5_ForCausalLM"
  },
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 1024,
  "intermediate_size": 3072,
  "max_position_embeddings": 131072,
  "model_type": "ernie4_5",
  "num_attention_heads": 16,
  "num_key_value_heads": 2,
  "head_dim": 128,
  "num_hidden_layers": 18,
  "pad_token_id": 0,
  "rms_norm_eps": 1e-05,
  "use_cache": false,
  "vocab_size": 103424,
  "rope_theta": 500000,
  "use_bias": false,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16"
}
