import torch

from transformers import OpenAIGPTTokenizer, TFOpenAIGPTLMHeadModel

tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
model = TFOpenAIGPTLMHeadModel.from_pretrained('openai-gpt')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, labels=input_ids)
loss, logits = outputs[:2]
