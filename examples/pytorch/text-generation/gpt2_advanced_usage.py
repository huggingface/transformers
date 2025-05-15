from transformers import pipeline


# Load GPT2 model using pipeline
generator = pipeline("text-generation", model="gpt2")

prompt = "The future of AI is"
output = generator(
    prompt,
    max_length=50,
    num_return_sequences=2,
    temperature=0.7,  # Balanced creativity
    top_k=50,  # Limits sampling to top 50 tokens
    truncation=True,  # Ensures proper handling of length
)

for i, o in enumerate(output):
    print(f"--- Output {i + 1} ---")
    print(o["generated_text"])
