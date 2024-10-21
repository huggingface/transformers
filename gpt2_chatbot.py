# ## FROM VIDEO
# from transformers import pipeline

# classifier = pipeline("sentiment-analysis")
# res = classifier("I've been waiting for a HuggingFace course my whole life.")
# print(res)

## FROM CHAT GPT
print("starting imports")
from transformers import GPT2Tokenizer, GPT2LMHeadModel

print("Finished imports")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print("Got GPT2Tokenizer")
model = GPT2LMHeadModel.from_pretrained("gpt2")
print("Got GPT2LMHeadModel")

inputs = tokenizer("Hello world!", return_tensors="pt")
print("inputs")
outputs = model(**inputs)
print("outputs")

print("hello")

##FROM README FILE
# from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = AutoModel.from_pretrained("gpt2")

# inputs = tokenizer("Hello world!", return_tensors="pt")
# outputs = model(**inputs)
