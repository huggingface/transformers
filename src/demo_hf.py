import logging
import os

import torch

from transformers.models.blt_wip.modeling_blt_wip import BLTModel
from transformers.models.blt_wip.tokenizers.blt_tokenizer import BltTokenizer


logger = logging.getLogger()

os.environ["BLT_SUPPRESS_ATTN_ERROR"] = "1"


def get_generation_range(prompt_tokens: list[list[int]] | None, max_gen_len: int) -> tuple[int, int]:
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
    model: BLTModel,
    tokenizer: BltTokenizer,
    max_prompt_len: int = 256,
    max_gen_len: int = 256,
    use_sampling: bool = False,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    remove_prompts: bool = True,
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

        next_token = torch.where(input_text_mask[:, curr_pos], tokens[:, curr_pos], next_token)
        tokens[:, curr_pos] = next_token

    if remove_prompts:
        generated_tokens = [
            t[len(prompt_tokens[i]) : len(prompt_tokens[i]) + max_gen_len].tolist() for i, t in enumerate(tokens)
        ]
    else:
        generated_tokens = [t[: len(prompt_tokens[i]) + max_gen_len].tolist() for i, t in enumerate(tokens)]
    return generated_tokens


def main(prompt: str = "my name is", model_name: str = "blt-1b"):
    device = "cuda"

    blt_repo = "itazap/blt-1b"

    model = BLTModel.from_pretrained(blt_repo).to(device)
    tokenizer = BltTokenizer(vocab_size_unit_1=model.config.vocab_size, add_bos=True, add_eos=True)

    prompts = [prompt]

    outputs = generate(prompts, model=model, tokenizer=tokenizer, max_gen_len=200, device=device)

    text_outputs = [tokenizer.decode(t) for t in outputs]
    for p, t in zip(prompts, text_outputs):
        print(f'Prompt: "{p}"')
        print(f'Completion: "{t}"')
        print()


if __name__ == "__main__":
    main()
