from transformers import SiglipTokenizer, SiglipTokenizerFast


slow_tokenizer = SiglipTokenizer.from_pretrained("google/siglip-so400m-patch14-384")

fast_tokenizer = SiglipTokenizerFast.from_pretrained("google/siglip-so400m-patch14-384")

text = "hello world"

inputs = slow_tokenizer(text, return_tensors="pt")

fast_inputs = fast_tokenizer(text, return_tensors="pt")

for k, v in inputs.items():
    assert (v == fast_inputs[k]).all()
