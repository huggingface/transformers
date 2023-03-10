#!/usr/bin/env python3
import os

import torch
import torch.distributed as dist

# from megatron.initialize import initialize_megatron
from fairseq import checkpoint_utils, models
from fairseq.models.transformer_lm import TransformerLanguageModel

from transformers import NllbMoeForConditionalGeneration, NllbTokenizer


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

# initialize the process group
dist.init_process_group("gloo", rank=0, world_size=1)


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
# cfg.model.moe_expert_count=256, the checkpoint has more experts than the configuration available with the
# `state`. This is strange
# cfg.model.ddp_backend = ""
# 1. build the task to make sure that the embedding layers will be built?
# There are 256 experts, not 128


# build the model
model = models.build_model(cfg.model, None, from_checkpoint=True)
# model = quantization_utils.quantize_model_scalar(model, args)

# load the merged state dict in the built model.
model.load_state_dict(state["model"], strict=False, model_cfg=cfg.model)
model = model.eval()


tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="fra_Latn")

src_text = "Life is like a box of chocolates."
tgt_text = "La vie est comme une boîte de chocolat."

model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt").input_ids
with torch.no_grad():
    logits = model(model_inputs, len(model_inputs), torch.tensor([[2, tokenizer.lang_code_to_id["fra_Latn"]]]))[0]
pred_next_token = torch.argmax(logits[0, -1], -1)
next_token = tokenizer.decode(pred_next_token)
print(f"Next word: {next_token}")
print("-------------")


# forward passes
def single_batch_forward_logits(prompts):
    input_ids = tokenizer(prompts, return_tensors="pt").input_ids
    input_ids = torch.cat([torch.tensor([[0]]), input_ids], dim=-1)
    input_ids = input_ids
    with torch.no_grad():
        logits = model(input_ids, len(input_ids), input_ids)[0]
    return logits


# Generating with fairseq:


custom_lm = TransformerLanguageModel.from_pretrained(
    "/path/to/model/dir", "checkpoint100.pt", tokenizer="moses", bpe="fastbpe"
)
custom_lm.sample("Barack Obama", beam=5)

# myabe use the /home/arthur_huggingface_co/fairseq/examples/nllb/modeling/evaluation/conf/generate_multi.yaml
# and the generate multi.py script
# There is also generate.py which contains all of the generation methods.
# Let's hack through this!


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


exit(0)

hf_model = NllbMoeForConditionalGeneration.from_pretrained(hf_path)


# forward hf
def forward_hf(prompts):
    input_ids = tokenizer(prompts, return_tensors="pt").input_ids
    input_ids = torch.cat([torch.tensor([[0]]), input_ids], dim=-1)
    input_ids = input_ids
    with torch.no_grad():
        logits = hf_model(input_ids)[0]
    return logits


print("Next word generation")
for prompt in prompts:
    logits = forward_hf(prompt)
    pred_next_token = torch.argmax(logits[0, -1], -1)
    next_token = tokenizer.convert_ids_to_tokens([pred_next_token])
    next_token = next_token[0].replace("Ġ", "")
    print(f"Next word: {next_token}")
    print("-------------")

print("Is equal:", torch.allclose(logits_fsq.cpu(), logits.cpu(), atol=1e-3))
