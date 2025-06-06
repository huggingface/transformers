from huggingface_hub import hf_hub_download
import json

import logging
import os

import torch

from transformers.models.blt_wip.modeling_blt_wip import ByteLatentTransformer
from transformers.models.blt_wip.configuration_blt import BLTConfig
from transformers.models.blt_wip.tokenizers.blt_tokenizer import BltTokenizer

logger = logging.getLogger()

import os
os.environ["BLT_SUPPRESS_ATTN_ERROR"] = "1"

def get_generation_range(
    prompt_tokens: list[list[int]] | None, max_gen_len: int
) -> tuple[int, int]:
    batch_min_prompt_length = min([len(t) for t in prompt_tokens])
    batch_max_prompt_length = max([len(t) for t in prompt_tokens])
    return batch_min_prompt_length, batch_max_prompt_length + max_gen_len


def sample_top_k(probs, k):
    topk_value, _ = torch.topk(probs, k)  # batch_sz x topk
    min_value_top_k = topk_value[:, [-1]]
    probs[probs < min_value_top_k] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


@torch.inference_mode()
def generate(
    prompts: list[str] | None,
    *,
    model: ByteLatentTransformer,
    tokenizer: BltTokenizer,
    max_prompt_len: int = 256,
    max_gen_len: int = 256,
    use_sampling: bool = False,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    remove_prompts: bool = True,
) -> list[list[int]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert (
        model.patch_in_forward
    ), "generate requires model.patch_in_forward=True"
    model.eval()
    prompt_tokens = [tokenizer.encode(t, add_eos=False) for t in prompts]
    # Truncation
    prompt_tokens = [
        t if len(t) < max_prompt_len else t[len(t) - max_prompt_len :]
        for t in prompt_tokens
    ]
    start_pos, end_pos = get_generation_range(prompt_tokens, max_gen_len)
    batch_size = len(prompt_tokens)
    tokens = torch.full((batch_size, end_pos), tokenizer.pad_id, dtype=torch.long, device=device)

    # Copy inputs to tensor for generated tokens
    for i, row_tokens in enumerate(prompt_tokens):
        tokens[i, : len(row_tokens)] = torch.tensor(row_tokens).long()
    input_text_mask = tokens != tokenizer.pad_id

    for i, curr_pos in enumerate(range(start_pos, end_pos)):
        current_tokens = tokens[:, :curr_pos]
        patch_lengths, _ = model.patch(current_tokens, include_next_token=True)
        logits = model(current_tokens, patch_lengths=patch_lengths)[:, -1]

        if use_sampling:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_token = sample_top_p(probs, top_p)
            elif top_k > 0:
                next_token = sample_top_k(probs, top_k)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1)

        next_token = torch.where(
            input_text_mask[:, curr_pos], tokens[:, curr_pos], next_token
        )
        tokens[:, curr_pos] = next_token

    if remove_prompts:
        generated_tokens = [
            t[len(prompt_tokens[i]) : len(prompt_tokens[i]) + max_gen_len].tolist()
            for i, t in enumerate(tokens)
        ]
    else:
        generated_tokens = [
            t[: len(prompt_tokens[i]) + max_gen_len].tolist()
            for i, t in enumerate(tokens)
        ]
    return generated_tokens



def main(prompt: str = "my name is", model_name: str = "blt-1b"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #HF
    blt_repo = "facebook/blt-1b"
    
    # Get the model's default configuration and entropy model params
    print("Loading model configuration...")
    config_path = hf_hub_download(repo_id=blt_repo, filename="config.json")
    entropy_params_path = hf_hub_download(repo_id=blt_repo, filename="entropy_model/params.json")
    entropy_checkpoint_path = hf_hub_download(repo_id=blt_repo, filename="entropy_model/consolidated.pth")
    entropy_dir = os.path.dirname(entropy_checkpoint_path)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    with open(entropy_params_path, 'r') as f:
        entropy_params = json.load(f)
    
    config['args']['attn_bias_type'] = 'causal'
    config['args']['attn_impl'] = 'sdpa'

    model_config = BLTConfig(**config["args"])
    
    patcher_args = entropy_params["data"]["patcher_args"]
    model_config.patch_in_forward = True
    model_config.realtime_patching = True  # Enable realtime patching
    model_config.patch_size = patcher_args["patch_size"]
    model_config.patching_mode = "entropy"  #patcher_args["patching_mode"] #TODO: we need to pass "entropy" to run through the Patcher / "entropy model", which is the LMTransformer
    model_config.patching_threshold = patcher_args["threshold"]
    model_config.patching_threshold_add = patcher_args["threshold_add"]
    model_config.max_patch_length = patcher_args["max_patch_length"]
    model_config.patching_batch_size = patcher_args["patching_batch_size"]
    model_config.patching_device = patcher_args["patching_device"]
    model_config.monotonicity = patcher_args["monotonicity"]
    model_config.entropy_model_checkpoint_dir = entropy_dir #original config on the hub don't set this


    model = ByteLatentTransformer.from_pretrained(blt_repo, config=model_config).to(device)

    tokenizer = BltTokenizer(
        vocab_size_unit_1=model_config.vocab_size,
        add_bos=True,
        add_eos=True
    )

    prompts = [prompt]
    outputs = generate(
        prompts, 
        model=model, 
        tokenizer=tokenizer,
        max_gen_len=100
    )
    
    text_outputs = [tokenizer.decode(t) for t in outputs]
    for p, t in zip(prompts, text_outputs):
        print(f'Prompt: "{p}"')
        print(f'Completion: "{t}"')
        print()

if __name__ == "__main__":
    main()

