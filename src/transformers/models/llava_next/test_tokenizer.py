import torch
from huggingface_hub import hf_hub_download

from transformers import AddedToken, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("NousResearch/Nous-Hermes-2-Yi-34B", use_fast=False)
tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)

prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant\n"

filepath = hf_hub_download(repo_id="nielsr/test-image", filename="llava_1_6_34b_input_ids.pt", repo_type="dataset")
original_input_ids = torch.load(filepath, map_location="cpu")

inputs = tokenizer(prompt, return_tensors="pt")

original_input_ids[original_input_ids == -200] = 64000

# TODO figure out why "system", "user" and "assistant" are tokenized differently
assert original_input_ids[0].tolist() == inputs.input_ids[0].tolist()

# for original_id, hf_id in zip(original_input_ids[0].tolist(), input_ids[0].tolist()):
#     if original_id != hf_id:
#         print(original_id, hf_id, tokenizer.decode(original_id), tokenizer.decode(hf_id))

# print(tokenizer.decode([id for id in original_input_ids if id != -200]))

# print(tokenizer.decode(input_ids["input_ids"][0].tolist()))
