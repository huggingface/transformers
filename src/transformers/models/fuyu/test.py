from transformers import pipeline

pipe = pipeline(task="image-to-text", model="adept/fuyu-8b")

outputs = pipe("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
print(outputs)