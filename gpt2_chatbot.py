## FROM VIDEO
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
res = classifier("I've been waiting for a HuggingFace course my whole life.")
print(res)

## FROM CHAT GPT
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# # Load pre-trained model and tokenizer
# model_name = "gpt2"  # You can use "gpt2", "gpt2-medium", "gpt2-large", or "gpt2-xl"
# model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# # Start interactive loop
# while True:
#     input_text = input("You: ")
#     if input_text.lower() in ["exit", "quit"]:
#         break

#     # Encode input and generate a response
#     input_ids = tokenizer.encode(input_text, return_tensors="pt")
#     output = model.generate(input_ids, max_length=100, num_return_sequences=1)

#     # Decode and print response
#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#     print(f"GPT-2: {response}")



##FROM README FILE
# from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = AutoModel.from_pretrained("gpt2")

# inputs = tokenizer("Hello world!", return_tensors="pt")
# outputs = model(**inputs)
