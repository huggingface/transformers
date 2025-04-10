# NOTE link: https://github.com/huggingface/transformers/issues/36536

# NOTE possible references
# https://github.com/huggingface/transformers/issues/35276
# https://discuss.huggingface.co/t/transformer-kv-cache-produces-worse-output-than-normal-generation-why/143839


from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
import torch

import numpy as np
import random

import pdb
import argparse

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def str2bool(v):
    """
    Convert string input to bool for argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cache', type=str2bool)
    parser.add_argument('--output_attentions', type=str2bool)
    args = parser.parse_args()
    return args

def generate(model, tokenizer, inputs, use_cache, output_attentions=False, device='cpu'):
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_new_tokens=2,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            return_dict_in_generate=True,
            use_cache=use_cache,
            output_scores=True,
            # output_hidden_states=True,
            output_attentions=output_attentions,
        )
    # pdb.set_trace()
    return outputs

def main(args):
    
    # MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    MODEL = "/scratch/yx3038/model_ckpt/Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32) # , cache_dir="/workspace/fmrai/")

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()

    # device = 'cpu' 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    # Example text to generate from
    prompt = "Tell me something that is very exciting"

    # Format the prompt using the chat template
    formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)

    # Tokenize the input
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)

    outputs_cache = generate(model, tokenizer, inputs, use_cache=True, device=device)
    outputs_no_cache = generate(model, tokenizer, inputs, use_cache=False, device=device)
    outputs_cache_attentions = generate(model, tokenizer, inputs, use_cache=True, output_attentions=True, device=device)
    outputs_no_cache_attentions = generate(model, tokenizer, inputs, use_cache=False, output_attentions=True, device=device)
    # outputs = generate(model, tokenizer, inputs, use_cache=args.use_cache, output_attentions=args.output_attentions, device=device)

    # NOTE
    # outputs: the logits of the i-th new token (1 x 128256)
    # outputs_attentions: the attentions of the i-th new token (1 x 128256)
    for i in range(2):
        print(f"Cache {i}:        {outputs_cache.scores[i]}")
        print(f"No Cache {i}:     {outputs_no_cache.scores[i]}")
        print(f"Cache Att {i}:    {outputs_cache_attentions.scores[i]}")
        print(f"No Cache Att {i}: {outputs_no_cache_attentions.scores[i]}")
    
if __name__ == '__main__':
    set_seed(42)
    args = create_argparser()
    main(args)