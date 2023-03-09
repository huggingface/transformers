#!/usr/bin/env python3
import os

import torch

# from megatron.initialize import initialize_megatron
from fairseq import checkpoint_utils, models

from transformers import NllbMoeForConditionalGeneration, NllbTokenizer


path = "/home/arthur_huggingface_co/fairseq/weights/checkpoints/model_moe_54b"
hf_path = "/home/arthur/facebook/nllb-moe"

tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer.save_pretrained(path)

# load the rank-0, which will merge all the states
state = checkpoint_utils.load_checkpoint_to_cpu(
    os.path.join(path, "checkpoint_2_300000-rank-0.pt"),
    is_moe=True,
)
cfg = state["cfg"]
cfg.model.ddp_backend = ""
# 1. build the task to make sure that the embedding layers will be built?


# build the model
model = models.build_model(cfg.model, None, from_checkpoint=True)
# model = quantization_utils.quantize_model_scalar(model, args)

# load the merged state dict in the built model.
model.load_state_dict(state["model"], strict=True, model_cfg=cfg.model)

checkpoint = checkpoint_utils.load_model_ensemble_and_task(
    [os.path.join(path, "checkpoint_2_300000-shared.pt")],
    is_moe=True,
)

model = checkpoint[0][0].eval()
model = model

hf_model = NllbMoeForConditionalGeneration.from_pretrained(hf_path)


# forward passes
def single_batch_forward_logits(prompts):
    input_ids = tokenizer(prompts, return_tensors="pt").input_ids
    input_ids = torch.cat([torch.tensor([[0]]), input_ids], dim=-1)
    input_ids = input_ids
    with torch.no_grad():
        logits = model(input_ids)[0]
    return logits


# forward hf
def forward_hf(prompts):
    input_ids = tokenizer(prompts, return_tensors="pt").input_ids
    input_ids = torch.cat([torch.tensor([[0]]), input_ids], dim=-1)
    input_ids = input_ids
    with torch.no_grad():
        logits = hf_model(input_ids)[0]
    return logits


prompts = [
    "Today is a beautiful day and I want to",
    "In the city of",
    "Paris is the capital of France and",
    "Computers and mobile phones have taken",
]

print("Next word generation")
for prompt in prompts:
    print("-------------")
    print(f"Prompt: {prompt}...\n")
    logits_fsq = single_batch_forward_logits(prompt)
    pred_next_token = torch.argmax(logits_fsq[0, -1], -1)
    next_token = tokenizer.convert_ids_to_tokens([pred_next_token])
    next_token = next_token[0].replace("Ġ", "")
    print(f"Next word: {next_token}")
    print("-------------")
    logits = forward_hf(prompt)
    pred_next_token = torch.argmax(logits[0, -1], -1)
    next_token = tokenizer.convert_ids_to_tokens([pred_next_token])
    next_token = next_token[0].replace("Ġ", "")
    print(f"Next word: {next_token}")
    print("-------------")


print("Is equal:", torch.allclose(logits_fsq.cpu(), logits.cpu(), atol=1e-3))
