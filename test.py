import unittest

from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.generation_beam_constraints import (
    PhrasalConstraint
)
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

force_text = "forced"
force_tokens = tokenizer.encode(force_text, return_tensors="pt")
print("force_tokens", force_tokens)
constraints = [PhrasalConstraint(force_tokens[1:])]

input_text = ["This feels a little"]

model_inputs = tokenizer(input_text, return_tensors="pt")

print("input_ids", input_ids)
k = model.generate(
    **model_inputs,
    constraints=constraints
)

print(k)

assert False
