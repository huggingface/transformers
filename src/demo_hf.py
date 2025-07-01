import logging
import os

import torch

from transformers.models.blt_wip.modeling_blt import BLTModel
from transformers.models.blt_wip.tokenization_blt import BLTTokenizer


logger = logging.getLogger()

os.environ["BLT_SUPPRESS_ATTN_ERROR"] = "1"

import gc
gc.collect()
torch.cuda.empty_cache()

def get_generation_range(prompt_tokens: list[list[int]] | None, max_gen_len: int) -> tuple[int, int]:
    batch_min_prompt_length = min([len(t) for t in prompt_tokens])
    batch_max_prompt_length = max([len(t) for t in prompt_tokens])
    return batch_min_prompt_length, batch_max_prompt_length + max_gen_len


@torch.inference_mode()
def generate(
    prompts: list[str] | None,
    *,
    model: BLTModel,
    tokenizer: BLTTokenizer,
    max_prompt_len: int = 256,
    max_gen_len: int = 256,
    device: torch.device = torch.device("cpu"),
) -> list[list[int]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    prompt_tokens = [tokenizer.encode(t, add_eos=False) for t in prompts]
    # Truncation
    prompt_tokens = [t if len(t) < max_prompt_len else t[len(t) - max_prompt_len :] for t in prompt_tokens]
    start_pos, end_pos = get_generation_range(prompt_tokens, max_gen_len)
    batch_size = len(prompt_tokens)
    tokens = torch.full((batch_size, end_pos), tokenizer.pad_id, dtype=torch.long, device=device)

    # Copy inputs to tensor for generated tokens
    for i, row_tokens in enumerate(prompt_tokens):
        tokens[i, : len(row_tokens)] = torch.tensor(row_tokens).long()
    input_text_mask = tokens != tokenizer.pad_id

    for i, curr_pos in enumerate(range(start_pos, end_pos)):
        current_tokens = tokens[:, :curr_pos]
        logits = model(current_tokens)[:, -1]

        next_token = torch.argmax(logits, dim=-1)

        next_token = torch.where(input_text_mask[:, curr_pos], tokens[:, curr_pos], next_token)
        tokens[:, curr_pos] = next_token

    generated_tokens = [t[: len(prompt_tokens[i]) + max_gen_len].tolist() for i, t in enumerate(tokens)]
    return generated_tokens


def main(prompt: str = "my name is", model_name: str = "blt-1b"):
    device = "cuda"

    blt_repo = "itazap/blt-1b-converted" 

    model = BLTModel.from_pretrained(blt_repo).to(device)
    tokenizer = BLTTokenizer(add_bos_token=True, add_eos_token=True)

    prompts = [prompt]

    outputs = generate(prompts, model=model, tokenizer=tokenizer, max_gen_len=200, device=device)

    text_outputs = [tokenizer.decode(t) for t in outputs]
    
    for p, t in zip(prompts, text_outputs):
        print(f'Prompt: "{p}"')
        print(f'Completion: "{t}"')
        print()

    print('here')


if __name__ == "__main__":
    main()
