from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration

model_name = "HFLAY/switch_base_8"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = SwitchTransformersForConditionalGeneration.from_pretrained(model_name)


input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
labels = tokenizer("Das Haus ist wunderbar.", return_tensors="pt").input_ids

model(input_ids=input_ids, labels=labels, return_dict=True)