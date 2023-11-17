import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers.generation.grammar_utils import IncrementalGrammarConstraint

import logging


if __name__ == '__main__':
    torch.manual_seed(2)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    with open("examples/grammars/json.gbnf", "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)

    prefix1= "This is a valid json string for email:"
    prefix2= "This is a valid json string for shopping cart:"
    input_ids = tokenizer([prefix1, prefix2],add_special_tokens=False, return_tensors="pt", padding=True)["input_ids"]

    output = model.generate(input_ids, do_sample=False, max_length=30, num_beams=2, grammar=grammar,
                            num_return_sequences=2)
    # decode output
    generations = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(generations)

    """
    'This is a valid json string for email:{ "title": "Theory", "text": "Theory", "type": "text", "text": "Theory", "type',
    'This is a valid json string for shopping cart:{ "name": "MyCart", "price": "10", "price": "10", "price": "10", "price": "'
    """