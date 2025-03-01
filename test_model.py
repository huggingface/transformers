from transformers import AutoModel, AutoProcessor, AyaVisionProcessor, AyaVisionModel
import torch
import base64
from src.transformers.feature_extraction_utils import BatchFeature
import numpy as np

template = """
{%- if preamble != "" -%}
<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{ preamble }}<|END_OF_TURN_TOKEN|>
{%- endif -%}
{%- for message in messages -%}
    <|START_OF_TURN_TOKEN|>{{ message.role | replace("User", "<|USER_TOKEN|>") | replace("Chatbot", "<|CHATBOT_TOKEN|><|START_RESPONSE|>") | replace("System", "<|SYSTEM_TOKEN|>") }}
    {%- if message.content is defined -%}
        {%- if message.content is string -%}
{{ message.content }}
        {%- else -%}
            {%- for item in message.content | selectattr('type', 'equalto', 'image') -%}
<image>
            {%- endfor -%}
            {%- for item in message.content | selectattr('type', 'equalto', 'text') -%}
{{ item.text }}
            {%- endfor -%}
        {%- endif -%}
    {%- elif message.message is defined -%}
        {%- if message.message is string -%}
{{ message.message }}
        {%- else -%}
            {%- for item in message.message | selectattr('type', 'equalto', 'image') -%}
<image>
            {%- endfor -%}
            {%- for item in message.message | selectattr('type', 'equalto', 'text') -%}
{{ item.text }}
            {%- endfor -%}
        {%- endif -%}
    {%- endif -%}
    {%- if message.role == "Chatbot" -%}
<|END_RESPONSE|>
    {%- endif -%}
<|END_OF_TURN_TOKEN|>
{%- endfor -%}
<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
"""

def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        byte_string = image_file.read()
        base64_encoded = base64.b64encode(byte_string)
        return f"data:image/jpeg;base64,{base64_encoded.decode('utf-8')}"

conversation1 = [
    # {"role": "System", "message": [{"text": f"{STD_PREAMBLE}"}]},
    {"role": "User", "content": [
        {"type": "image", "url": image_to_base64("tests/test_images/image_resized.png")},
        {"type": "text", "text": "Describe this image"}
    ]},
]

conversation3 = [
    {"role": "User", "content": [
        {"type": "image", "url": image_to_base64("tests/test_images/v1_93.jpg")},
        {"type": "text", "text": "Print the exact text of this image"}
    ]},
]

model_name = "Cohere-hf/aya-vision-early-ckpt"

processor = AyaVisionProcessor.from_pretrained(model_name)

device = "cuda:0"

inputs = processor.apply_chat_template(
    [conversation1, conversation3],
    chat_template=template,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(device)

model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

output = model.generate(**inputs, max_new_tokens=200, do_sample=True, top_k=1, num_return_sequences=1)
for i in range(len(output)):
    print("\n")
    print(processor.tokenizer.decode(output[i][inputs.input_ids.shape[1]:], skip_special_tokens=True))

