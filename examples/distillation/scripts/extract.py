# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocessing script before training the distilled model.
Specific to RoBERTa -> DistilRoBERTa and GPT2 -> DistilGPT2.
"""
from transformers import BertForMaskedLM, RobertaForMaskedLM, GPT2LMHeadModel
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extraction some layers of the full RobertaForMaskedLM or GPT2LMHeadModel for Transfer Learned Distillation")
    parser.add_argument("--model_type", default="roberta", choices=["roberta", "gpt2"])
    parser.add_argument("--model_name", default='roberta-large', type=str)
    parser.add_argument("--dump_checkpoint", default='serialization_dir/tf_roberta_048131723.pth', type=str)
    parser.add_argument("--vocab_transform", action='store_true')
    args = parser.parse_args()


    if args.model_type == 'roberta':
        model = RobertaForMaskedLM.from_pretrained(args.model_name)
        prefix = 'roberta'
    elif args.model_type == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(args.model_name)
        prefix = 'transformer'

    state_dict = model.state_dict()
    compressed_sd = {}

    ### Embeddings ###
    if args.model_type == 'gpt2':
        for param_name in ['wte.weight', 'wpe.weight']:
            compressed_sd[f'{prefix}.{param_name}'] = state_dict[f'{prefix}.{param_name}']
    else:
        for w in ['word_embeddings', 'position_embeddings', 'token_type_embeddings']:
            param_name = f'{prefix}.embeddings.{w}.weight'
            compressed_sd[param_name] = state_dict[param_name]
        for w in ['weight', 'bias']:
            param_name = f'{prefix}.embeddings.LayerNorm.{w}'
            compressed_sd[param_name] = state_dict[param_name]

    ### Transformer Blocks ###
    std_idx = 0
    for teacher_idx in [0, 2, 4, 7, 9, 11]:
        if args.model_type == 'gpt2':
            for layer in ['ln_1', 'attn.c_attn', 'attn.c_proj', 'ln_2', 'mlp.c_fc', 'mlp.c_proj']:
                for w in ['weight', 'bias']:
                    compressed_sd[f'{prefix}.h.{std_idx}.{layer}.{w}'] = \
                        state_dict[f'{prefix}.h.{teacher_idx}.{layer}.{w}']
            compressed_sd[f'{prefix}.h.{std_idx}.attn.bias'] = state_dict[f'{prefix}.h.{teacher_idx}.attn.bias']
        else:
            for layer in ['attention.self.query', 'attention.self.key', 'attention.self.value',
                        'attention.output.dense', 'attention.output.LayerNorm',
                        'intermediate.dense', 'output.dense', 'output.LayerNorm']:
                for w in ['weight', 'bias']:
                    compressed_sd[f'{prefix}.encoder.layer.{std_idx}.{layer}.{w}'] = \
                        state_dict[f'{prefix}.encoder.layer.{teacher_idx}.{layer}.{w}']
        std_idx += 1

    ### Language Modeling Head ###s
    if args.model_type == 'roberta':
        for layer in ['lm_head.decoder.weight', 'lm_head.bias']:
            compressed_sd[f'{layer}'] = state_dict[f'{layer}']
        if args.vocab_transform:
            for w in ['weight', 'bias']:
                compressed_sd[f'lm_head.dense.{w}'] = state_dict[f'lm_head.dense.{w}']
                compressed_sd[f'lm_head.layer_norm.{w}'] = state_dict[f'lm_head.layer_norm.{w}']
    elif args.model_type == 'gpt2':
        for w in ['weight', 'bias']:
            compressed_sd[f'{prefix}.ln_f.{w}'] = state_dict[f'{prefix}.ln_f.{w}']
        compressed_sd[f'lm_head.weight'] = state_dict[f'lm_head.weight']

    print(f'N layers selected for distillation: {std_idx}')
    print(f'Number of params transfered for distillation: {len(compressed_sd.keys())}')

    print(f'Save transfered checkpoint to {args.dump_checkpoint}.')
    torch.save(compressed_sd, args.dump_checkpoint)
