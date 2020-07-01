from transformers.modeling_t5 import T5ForConditionalGeneration
from transformers.tokenization_t5 import T5Tokenizer
from pathlib import Path

cache_dir = Path('E:/Coding/cache/tokenizers')

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

inputs = tokenizer.encode(
    "translate English to German: Hugging Face is a technology company based in New York and Paris",
    return_tensors="pt")
outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
print(tokenizer.decode(outputs[0]))
