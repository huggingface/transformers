"""Convert Flax checkpoints from original paper to PyTorch"""
import argparse
from collections import OrderedDict
from pathlib import Path

import torch
import numpy as np

from flax.training.checkpoints import restore_checkpoint

def transform_attention(current: np.ndarray):
    if np.ndim(current) == 2:
        return transform_attention_bias(current)

    elif np.ndim(current) == 3:
        return transform_attention_kernel(current)

    else:
        raise Exception(f"Invalid number of dimesions: {np.ndim(current)}")


def transform_attention_bias(current: np.ndarray):
    return current.flatten()

def transform_attention_kernel(current: np.ndarray):
    return np.reshape(current, (current.shape[0], current.shape[1] * current.shape[2])).T

def transform_attention_output_weight(current: np.ndarray):
    return np.reshape(current, (current.shape[0] * current.shape[1], current.shape[2])).T


def transform_state_encoder_block(state_dict, i):
    state = state_dict['optimizer']['target']['Transformer'][f'encoderblock_{i}']

    new_state = OrderedDict()
    prefix = f'encoder.layer.{i}.'
    new_state = {
        prefix+'intermediate.dense.bias': state['MlpBlock_0']['Dense_0']['bias'],
        prefix+'intermediate.dense.weight': np.transpose(state['MlpBlock_0']['Dense_0']['kernel']),
        prefix+'output.dense.bias': state['MlpBlock_0']['Dense_1']['bias'],
        prefix+'output.dense.weight': np.transpose(state['MlpBlock_0']['Dense_1']['kernel']),
        prefix+'layernorm_before.bias': state['LayerNorm_0']['bias'],
        prefix+'layernorm_before.weight': state['LayerNorm_0']['scale'],
        prefix+'layernorm_after.bias': state['LayerNorm_1']['bias'],
        prefix+'layernorm_after.weight': state['LayerNorm_1']['scale'],

        prefix+'attention.attention.query.bias': transform_attention(state['MultiHeadDotProductAttention_0']['query']['bias']),
        prefix+'attention.attention.query.weight': transform_attention(state['MultiHeadDotProductAttention_0']['query']['kernel']),

        prefix+'attention.attention.key.bias': transform_attention(state['MultiHeadDotProductAttention_0']['key']['bias']),
        prefix+'attention.attention.key.weight': transform_attention(state['MultiHeadDotProductAttention_0']['key']['kernel']),

        prefix+'attention.attention.value.bias': transform_attention(state['MultiHeadDotProductAttention_0']['value']['bias']),
        prefix+'attention.attention.value.weight': transform_attention(state['MultiHeadDotProductAttention_0']['value']['kernel']),

        prefix+'attention.output.dense.bias': state['MultiHeadDotProductAttention_0']['out']['bias'],
        prefix+'attention.output.dense.weight': transform_attention_output_weight(state['MultiHeadDotProductAttention_0']['out']['kernel']),
    }

    return new_state


def transform_state(state_dict, transformer_layers=12):
    new_state = OrderedDict()

    new_state['layernorm.bias'] = state_dict['optimizer']['target']['Transformer']['encoder_norm']['bias']
    new_state['layernorm.weight'] = state_dict['optimizer']['target']['Transformer']['encoder_norm']['scale']

    new_state['embeddings.patch_embeddings.projection.weight'] = np.transpose(state_dict['optimizer']['target']['embedding']['kernel'], (4, 3, 0, 1, 2))
    new_state['embeddings.patch_embeddings.projection.bias'] = state_dict['optimizer']['target']['embedding']['bias']

    new_state['embeddings.cls_token'] = state_dict['optimizer']['target']['cls']
    new_state['embeddings.position_embeddings'] = state_dict['optimizer']['target']['Transformer']['posembed_input']['pos_embedding']

    for i in range(transformer_layers):
        new_state.update(transform_state_encoder_block(state_dict, i))

    
    return {k: torch.tensor(v) for k,v in new_state.items()}


def get_n_layers(state_dict):
    return sum([1 if 'encoderblock_' in k else 0 for k in state_dict['optimizer']['target']['Transformer'].keys()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--flax_model', type=str, help='Path to flax model')
    parser.add_argument('--output_model_name', type=str, help='Name of the outputed file')
    
    args = parser.parse_args()
    
    state_dict = restore_checkpoint(args.flax_model, None)
    
    n_layers = get_n_layers(state_dict)
    new_state = transform_state(state_dict, n_layers)
    
    out_path = Path(args.flax_model).parent.absolute()
    
    if '.pt' in args.output_model_name:
        out_path = out_path / args.output_model_name
        
    else:
        out_path = out_path / (args.output_model_name + '.pt')
    
    torch.save(new_state, out_path)
