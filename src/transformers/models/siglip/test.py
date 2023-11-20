from transformers import SiglipTokenizer


tokenizer = SiglipTokenizer(vocab_file="/Users/nielsrogge/Documents/SigLIP/sentencepiece.model", model_max_length=2)

print(tokenizer.eos_token_id)
print(tokenizer.pad_token_id)

text = "a"

print(tokenizer(text, padding="max_length", truncation=True))

text = "aaa"

print(tokenizer(text, padding="max_length", truncation=True))

text = "aaaa"

print(tokenizer(text, padding="max_length", truncation=True))
