import unittest

from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.generation_beam_constraints import (
    PhrasalConstraint
)
device = "cuda"

model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

force_text = "forced a little"
force_tokens = tokenizer.encode(force_text, return_tensors="pt").to(device)
print("force_tokens", force_tokens[0][1:-1])
constraints = [PhrasalConstraint(force_tokens[0][1:-1])]

input_text = ["this feels very"] * 10

model_inputs = tokenizer(input_text, return_tensors="pt")

for key, value in model_inputs.items():
    model_inputs[key] = value.to(device)

print("model_inputs", model_inputs)
k = model.generate(
    **model_inputs,
    constraints=constraints
)

for out in k:
    print(tokenizer.decode(out))

assert False
