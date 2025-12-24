from transformers import AutoProcessor, AutoModelForCausalLM

# processor = AutoProcessor.from_pretrained("/glm/1231/GLM-Lite-Opensource")
#
# model = AutoModelForCausalLM.from_pretrained("/glm/1231/GLM-Lite-Opensource", device_map="cuda:1")

processor = AutoProcessor.from_pretrained("/cloud/oss_checkpoints/zai-org/GLM-4.5-Air")
model = AutoModelForCausalLM.from_pretrained("/cloud/oss_checkpoints/zai-org/GLM-4.5-Air", device_map="auto")
messages = [
    {
        "role": "user",
        "content": "What animal is on the candy?"
    },
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

inputs.pop("token_type_ids")
outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1] :]))
