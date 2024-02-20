import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", padding_side="left", pad_token = "<s>")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", attn_implementation="sdpa").to(device)
model = model.eval()

#inputs = tokenizer(["A short text about dragons:", "Can you please write me a very long and entertaining text about dragons? Text:"], padding=True, return_tensors="pt")
inputs = tokenizer(["Can you please write me a very long and entertaining text about dragons? Text:"], return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

out = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.batch_decode(out))

