import unittest

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.generation_beam_constraints import (
    PhrasalConstraint
)
device = "cuda"

model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

force_text = "talk"
force_text_2 = "forceful manner"
force_tokens = tokenizer.encode(force_text, return_tensors="pt").to(device)[0]
force_tokens_2 = tokenizer.encode(force_text_2, return_tensors="pt").to(device)[0]

constraints = [
    PhrasalConstraint(force_tokens),
    PhrasalConstraint(force_tokens_2)
]

input_text = ["He always"] * 2

model_inputs = tokenizer(input_text, return_tensors="pt")

for key, value in model_inputs.items():
    model_inputs[key] = value.to(device)

print("model_inputs", model_inputs)
k = model.generate(
    **model_inputs,
    constraints=constraints,
    num_beams=4,
    num_return_sequences=3
)

for out in k:
    print(tokenizer.decode(out))

assert False
