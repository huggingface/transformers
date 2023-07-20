from transformers import AutoModelForCausalLM, AutoTokenizer


model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)

inputs = tokenizer(["I'm not going to"], return_tensors="pt")

summary_ids = model.generate(inputs["input_ids"], max_length=20, top_p=10.1, filter_value=2.0)
print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
