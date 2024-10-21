# ## FROM VIDEO
# from transformers import pipeline

# classifier = pipeline("sentiment-analysis")
# res = classifier("I've been waiting for a HuggingFace course my whole life.")
# print(res)

## FROM CHAT GPT
print("starting imports")
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)

print("hello")

##FROM README FILE
# from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = AutoModel.from_pretrained("gpt2")

# inputs = tokenizer("Hello world!", return_tensors="pt")
# outputs = model(**inputs)
