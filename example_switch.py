from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("t5-small")
text = "A <extra_id_0> walks into a bar a orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>."
model = SwitchTransformersForConditionalGeneration.from_pretrained("HFLAY/switch_base_8")

input_ids = tokenizer(text, return_tensors="pt").input_ids
out = model.generate(input_ids, decoder_start_token_id=0, output_router_logits=True)
print(tokenizer.decode(out[0]))

# Loss

input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
outputs = model(input_ids=input_ids, labels=labels, output_router_logits=True)