## Advanced Text-Generation: Batch Inference with Post-Processing

This example demonstrates **running multiple prompts** and **applying custom post-processing** on outputs.

```python
from transformers import pipeline,AutoModelForCasualLM,AutoTokenizer
import torch

#load model and tokenizer

model_name = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCasualLM.from_pretrained(model_name)

generator - pipeline("text-generation",model=model,tokenizer=tokenizer,device=0)

#Multiple prompts

prompts = [
    "The secrets to a perfect cake is ",
    " The future of AI is "
]

#Custom post-processing funtcion
def clean_output(text):
    # Keep only the first sentence
    return text.split('.')[0] + '.'

# Generate and process outputs
outputs = generator(prompts,max_lenght=50)
processed = [clean_output(out[0]['generated_text']) for out in outputs]

for prompt, text in zip(prompts, processed):
    print(f"Prompt: {prompt}")
    print(f"Processed output: {text}\n")
