import requests
from PIL import Image

import torch
from transformers import MllamaProcessor, MllamaForConditionalGeneration, EncoderDecoderCache, DynamicCache

# torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)

device = torch.device("cuda")

url = "https://www.ilankelman.org/stopsigns/australia.jpg"
stop_image = Image.open(requests.get(url, stream=True).raw)

url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
snowman_image = Image.open(requests.get(url, stream=True).raw)

processor = MllamaProcessor.from_pretrained("s0409/model-3")
model = MllamaForConditionalGeneration.from_pretrained("s0409/model-3", torch_dtype=torch.bfloat16, device_map=device)

texts = ["<|begin_of_text|>My name is"]
inputs = processor(text=texts, return_tensors="pt").to(device)
output = model.generate(
    **inputs,
    do_sample=False,
    max_new_tokens=40,
    use_cache=False,
    output_logits=True,
    return_dict_in_generate=True
)
print("Full text generate text-only:", processor.batch_decode(output.sequences, skip_special_tokens=False))




conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What’s shown in this image?"},
        
        ]
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "This image shows a red stop sign."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What about this one"}
        ],
    },
]

template = "{% for message in messages %}" \
                "{% if loop.index0 == 0 %}" \
                    "{{bos_token}}" \
                "{% endif %}" \
                "{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}" \
                "{# Render all images first #}" \
                "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}" \
                    "{{ '<|image|>' }}" \
                "{% endfor %}" \
                "{# Render all text next #}" \
                "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}" \
                    "{{content['text'] | trim + '<|eot_id|>' }}" \
                "{% endfor %}" \
            "{% endfor %}" \
            "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}" \

prompt = processor.apply_chat_template(conversation, chat_template=template, add_generation_prompt=True)
inputs = processor(text=prompt, images=[[stop_image, snowman_image]], return_tensors="pt").to(device, torch.bfloat16)
inputs.pop("aspect_ratios")

output = model.generate(
    **inputs,
    do_sample=False,
    max_new_tokens=40,
    use_cache=True,
    output_logits=True,
    return_dict_in_generate=True
)
print("Full text our generate:", processor.batch_decode(output.sequences, skip_special_tokens=False))

# -------------------------------------------------------------------------------------------------------------

texts = ["<|image|><|begin_of_text|>This image demonstrates us a"]
inputs = processor(text=texts, images=[[snowman_image]], return_tensors="pt").to(device, torch.bfloat16)
inputs.pop("aspect_ratios")

output = model.generate(
    **inputs,
    do_sample=False,
    max_new_tokens=40,
    use_cache=True,
    output_logits=True,
    num_beams=3,
    return_dict_in_generate=True
)
print("Full text our generate:", processor.batch_decode(output.sequences, skip_special_tokens=False))


# ----------------------------------------------------------------------------------------------------------------------------------


# msg = ""
# past_kv = DynamicCache(model.config.text_config)
# user_prompts = [("What’s shown in this image?", stop_image), ("What about this image, what do you see here?", snowman_image)]
# for prompt, image in user_prompts:
#     msg += f"<|image|><|begin_of_text|>{prompt}"
#     inputs = processor(text=msg, images=[[image]], return_tensors="pt").to(device, torch.bfloat16)
#     inputs.pop("aspect_ratios", None)
# 
#     print(inputs.cross_attention_mask.shape)
#     outputs = model.generate(**inputs, do_sample=False, max_new_tokens=10, past_key_values=past_kv, return_dict_in_generate=True)
#     past_kv = outputs.past_key_values
#     msg += processor.batch_decode(outputs.sequences)[0]
#     print(processor.batch_decode(outputs.sequences))
