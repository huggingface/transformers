from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_path = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).to("cuda")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "/raid/raushan/karate.mp4"},
            {"type": "text", "text": "Can you describe this video?"},            
        ]
    },
]

messages_2 = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "/raid/raushan/Skiing.mp4"},
            {"type": "text", "text": "What do you see here in the video?"},            
        ]
    },
]

# chat_template = "<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% elif line['type'] == 'video' %}{{ '<video>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"

inputs = processor.apply_chat_template(
    [messages_2],
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    padding=True,
    return_tensors="pt",
    # chat_template=chat_template,
    do_sample_frames=True,
).to(model.device, dtype=torch.bfloat16)
print(inputs.pixel_values.shape)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=50)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts)

