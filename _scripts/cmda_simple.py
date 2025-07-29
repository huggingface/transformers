from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "CohereLabs/c4ai-command-a-03-2025"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# Format message with the c4ai-command-a-03-2025 chat template
messages = [{"role": "user", "content": "Hello, how are you?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

gen_tokens = model.generate(
    input_ids, 
    max_new_tokens=100, 
    do_sample=True, 
    temperature=0.3,
)

gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)