from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "../../models/gqa-1b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

x = "Hi how"
x = tokenizer(x, return_tensors="pt")

y = model.generate(**x, max_new_tokens=100)
y = tokenizer.batch_decode(y)
for i in y:
    print(i)
