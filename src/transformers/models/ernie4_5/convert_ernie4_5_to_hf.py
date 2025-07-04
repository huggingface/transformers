from transformers import Ernie4_5ForCausalLM


def convert_ernie_4_5_model_to_hf(checkpoint_path):
    model = Ernie4_5ForCausalLM.from_pretrained(checkpoint_path)
    model_dict = model.state_dict()

    # meta info for RoPE conversion
    head_dim = model.config.head_dim
    hidden_size = model.config.hidden_size

    num_heads = model.config.num_attention_heads
    dim_q = num_heads * head_dim
    num_kv_heads = model.config.num_key_value_heads
    dim_kv = num_kv_heads * head_dim

    def permute(w, n_heads, dim1=dim_q, dim2=hidden_size):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    converted_state_dict = {}
    for key, tensor in model_dict.items():
        if "q_proj" in key:
            converted_state_dict[key] = permute(tensor, n_heads=num_heads)
        elif "k_proj" in key:
            converted_state_dict[key] = permute(tensor, n_heads=num_kv_heads, dim1=dim_kv)
        else:
            converted_state_dict[key] = tensor

    # Load converted weights
    model.load_state_dict(converted_state_dict, assign=True)

    return model


model = convert_ernie_4_5_model_to_hf("baidu/ERNIE-4.5-0.3B-PT")
model.save_pretrained("AntonV/ERNIE-4.5-0.3B-PT")
