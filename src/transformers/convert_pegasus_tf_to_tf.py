import tensorflow as tf


def convert_pegasus_to_bart(tf_weights_dict, bart_model_state_dict):
    pass



#layer_keys = {'self_attn.k_proj.weight', 'self_attn.k_proj.bias', 'self_attn.v_proj.weight', 'self_attn.v_proj.bias', 'self_attn.q_proj.weight', 'self_attn.q_proj.bias', 'self_attn.out_proj.weight', 'self_attn.out_proj.bias', 'self_attn_layer_norm.weight', 'self_attn_layer_norm.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'final_layer_norm.weight', 'final_layer_norm.bias'}

['self_attn.k_proj.weight',
 'self_attn.k_proj.bias',
 'self_attn.v_proj.weight',
 'self_attn.v_proj.bias',
 'self_attn.q_proj.weight',
 'self_attn.q_proj.bias',
 'self_attn.out_proj.weight',
 'self_attn.out_proj.bias',
 'self_attn_layer_norm.weight',
 'self_attn_layer_norm.bias',
 'encoder_attn.k_proj.weight',
 'encoder_attn.k_proj.bias',
 'encoder_attn.v_proj.weight',
 'encoder_attn.v_proj.bias',
 'encoder_attn.q_proj.weight',
 'encoder_attn.q_proj.bias',
 'encoder_attn.out_proj.weight',
 'encoder_attn.out_proj.bias',
 'encoder_attn_layer_norm.weight',
 'encoder_attn_layer_norm.bias',
 'fc1.weight',
 'fc1.bias',
 'fc2.weight',
 'fc2.bias',
 'final_layer_norm.weight',
 'final_layer_norm.bias']
