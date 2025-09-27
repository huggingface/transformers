# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="google/gemma-3-270m")

text = pipe("Booker, are you afraid of God?", do_sample=True, top_k=50, top_p=0.95, min_p=0.05, temperature=0.95)

print(text)
