from transformers import LlavaConfig, LlavaForConditionalGeneration, AutoTokenizer, MistralConfig, PixtralConfig

import torch
from safetensors.torch import load_file as safe_load_file
import regex as re

tokenizer = AutoTokenizer.from_pretrained("leafspark/Pixtral-12B-2409-hf", )


text_config = MistralConfig(
    attention_dropout=0.0,
    bos_token_id=1,
    eos_token_id=2,
    head_dim=128,
    hidden_act="silu",
    hidden_size=5120,
    initializer_range=0.02,
    intermediate_size=14336,
    max_position_embeddings=1024000,
    model_type="mistral",
    num_attention_heads=32,
    num_hidden_layers=40,
    num_key_value_heads=8,
    rms_norm_eps=1e-05,
    rope_theta=1000000000.0,
    sliding_window=None,
    tie_word_embeddings=False,
    vocab_size=131072
)

vision_config = PixtralConfig()
config = LlavaConfig(vision_config, text_config)
config.architectures = ["LlavaForConditionalGeneration"]
config.save_pretrained("/Users/arthurzucker/Work/pixtral")

        
original_state_dict = safe_load_file("/Users/arthurzucker/Work/pixtral/model.safetensors")


OLD_KEY_TO_NEW_KEY_MAPPING = {
    # Layer Normalization Weights
    r"vision_encoder.transformer.layers.(\d+).input_layernorm.weight":  r"vision_tower.transformer.layers.\1.attention_norm.weight",
    r"vision_encoder.transformer.layers.(\d+).ffn_norm.weight":         r"vision_tower.transformer.layers.\1.ffn_norm.weight",
    
    # Self Attention Projections
    r"vision_encoder.transformer.layers.(\d+).attention.wq.weight":     r"vision_tower.transformer.layers.\1.attention.q_proj.weight",
    r"vision_encoder.transformer.layers.(\d+).attention.wk.weight":     r"vision_tower.transformer.layers.\1.attention.k_proj.weight",
    r"vision_encoder.transformer.layers.(\d+).attention.wv.weight":     r"vision_tower.transformer.layers.\1.attention.v_proj.weight",
    r"vision_encoder.transformer.layers.(\d+).attention.wo.weight":     r"vision_tower.transformer.layers.\1.attention.o_proj.weight",
    
    # MLP Projections
    r"vision_encoder.transformer.layers.(\d+).feed_forward.w1.weight":  r"vision_tower.transformer.layers.\1.feed_forward.gate_proj.weight",
    r"vision_encoder.transformer.layers.(\d+).feed_forward.w2.weight":  r"vision_tower.transformer.layers.\1.feed_forward.down_proj.weight",
    r"vision_encoder.transformer.layers.(\d+).feed_forward.w3.weight":  r"vision_tower.transformer.layers.\1.feed_forward.up_proj.weight",
    
    # Additional mappings
    r"vision_encoder":                                  r"vision_tower",
    r"vision_language_adapter.w_in":                    r"multi_modal_projector.linear_1",
    r"vision_language_adapter.w_out":                   r"multi_modal_projector.linear_2",
    r"layers.(\d+).attention.wq.weight":                r"language_model.model.layers.\1.self_attn.q_proj.weight",
    r"layers.(\d+).attention.wk.weight":                r"language_model.model.layers.\1.self_attn.k_proj.weight",
    r"layers.(\d+).attention.wv.weight":                r"language_model.model.layers.\1.self_attn.v_proj.weight",
    r"layers.(\d+).attention.wo.weight":                r"language_model.model.layers.\1.self_attn.o_proj.weight",
    r"layers.(\d+).feed_forward.w1.weight":             r"language_model.model.layers.\1.mlp.gate_proj.weight",
    r"layers.(\d+).feed_forward.w2.weight":             r"language_model.model.layers.\1.mlp.down_proj.weight",
    r"layers.(\d+).feed_forward.w3.weight":             r"language_model.model.layers.\1.mlp.up_proj.weight",
    r"layers.(\d+).ffn_norm.weight":                    r"language_model.model.layers.\1.post_attention_layernorm.weight",
    r"layers.(\d+).attention_norm.weight":              r"language_model.model.layers.\1.input_layernorm.weight",
    r"tok_embeddings.weight":                           r"language_model.model.embed_tokens.weight",
    r"output.weight":                                   r"language_model.lm_head.weight",
    r"norm.weight":                                     r"language_model.model.norm.weight"

}

new_state_dict = {} 
all_keys = "\n".join(original_state_dict.keys())
old_keys = all_keys
for old, new in OLD_KEY_TO_NEW_KEY_MAPPING.items():
    all_keys = re.sub(r"\n"+ old,r"\n"+new,all_keys)

OLD_TO_NEW = dict(zip(old_keys.split("\n"), all_keys.split("\n")))

new_dict={}

def permute_for_rope(value,n_heads, config):
        dim1 = config.head_dim
        dim2 = config.hidden_size
        return value.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2) 

for key, value in original_state_dict.items():

    if "vision_encoder" in key:
        _config = vision_config
    else:
        _config = text_config
        # convert the text model (basically mistral model)

    if "q_proj" in key:
        value = permute_for_rope(value,_config.num_attention_heads)
    if "k_proj" in key:
        value = permute_for_rope(value,_config.num_key_value_heads)

    new_key = OLD_TO_NEW[key]
    new_dict[new_key] = value

with torch.device("meta"):
    model = LlavaForConditionalGeneration(config)

model.load_state_dict(new_dict, strict=True, assign=True)