import torch

from transformers import BloomForCausalLM, BloomForPrefixLM
from transformers import AutoTokenizer

model_path = "bigscience/bloom-350m"
model_causal = BloomForCausalLM.from_pretrained(model_path)
model_prefix = BloomForPrefixLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

input_string = [
    "This is an input sentence.",
    "This is an input sentence. This is a following sentence.",
    "This is an alternative input sequence.",
    "A clause, another clause, and a closing clause"
    ]

for i, _model in enumerate([model_causal, model_prefix]):
    print("\n######")
    print("Test generation on {}".format(["model_causal", "model_prefix"][i]))
    print("######\n")

    print("## Single input generation\n")
    tokenized_inputs = tokenizer(
        input_string[0],
        return_tensors="pt"
        )

    output = _model.generate(**tokenized_inputs)
    print(tokenizer.decode(output[0]))

    print("\n## Batched generation\n")
    tokenized_inputs = tokenizer(
        input_string,
        return_tensors="pt",
        padding=True
        )

    output = _model.generate(**tokenized_inputs)
    for _output in output:
        print(tokenizer.decode(_output))