import datasets
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16)

train_dataset = datasets.load_dataset('imdb', split='train')

def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=128, truncation=True, padding="max_length")

tokenized_datasets = train_dataset.map(tokenize_function, batched=True)

for batch in model.fast_generate(tokenized_datasets):
    print(batch)



