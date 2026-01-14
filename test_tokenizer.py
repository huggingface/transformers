from transformers import AutoTokenizer

model_id = "Qwen/Qwen3-4B-Instruct-2507"

# First run
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)
print("First run tokenizer:", type(tokenizer))

# Second run (to test cached behavior)
tokenizer2 = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)
print("Second run tokenizer:", type(tokenizer2))
