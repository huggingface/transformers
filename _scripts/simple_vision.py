from transformers import pipeline

pipe = pipeline(model="CohereLabs/command-a-vision-07-2025", task="image-text-to-text", device_map="auto")

# Format message with the Command-A-Vision chat template
messages = [
    {"role": "user",
     "content": [
       {"type": "image", "url": "https://media.istockphoto.com/id/458012057/photo/istanbul-turkey.jpg?s=612x612&w=0&k=20&c=qogAOVvkpfUyqLUMr_XJQyq-HkACXyYUSZbKhBlPrxo="},
        {"type": "text", "text": "Where was this taken ?"},
    ]},
]

outputs = pipe(text=messages, max_new_tokens=300, return_full_text=False)
print(outputs[0]['generated_text'])