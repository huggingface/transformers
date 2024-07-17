from transformers import AutoProcessor, AutoTokenizer


# {{SYSTEM_PROMPT}} USER: <image>\n{{PROMPT}} ASSISTANT:" assistant end with "</s> "
chat_vicuna = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ message['content'][0]['text'] }}"
    "{% else %}"
    "{{ message['role'].upper() + ': '}}"
    "{% endif %}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>\n' }}"
    "{% endfor %}"
    "{# Render all text next #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ content['text'] + ' '}}"
    "{% endfor %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ 'ASSISTANT:' }}"
    "{% endif %}"
)

# {{SYSTEM_PROMPT}}###Human:: <image>\n{{PROMPT}}###Assistant:"
chat_llama = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ message['content'][0]['text'] }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '###Human: '}}"
    "{% else %}"
    "{{ '###' + message['role'].title() + ': '}}"
    "{% endif %}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>\n' }}"
    "{% endfor %}"
    "{# Render all text next #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ content['text'] }}"
    "{% endfor %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '###Assistant:' }}"
    "{% endif %}"
)


# "[INST] <image>\nWhat is shown in this image? [/INST]" assistant end with "</s> "
chat_mistral = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '[INST] ' }}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>\n' }}"
    "{% endfor %}"
    "{# Render all text next #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ content['text'] }}"
    "{% endfor %}"
    "{{' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' ' + message['content'][0]['text'] + '<\s> '}}"
    "{% else %}"
    "{{ raise_exception('Only user and assistant roles are supported!') }}"
    "{% endif %}"
    "{% endfor %}"
)

# "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant\n"
chat_yi = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>\n' }}"
    "{% endfor %}"
    "{# Render all text next #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ content['text'] }}"
    "{% endfor %}"
    "{{'<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

model2template = {
    "llava-hf/vip-llava-13b-hf": chat_llama,
    "llava-hf/vip-llava-7b-hf": chat_llama,
    "llava-hf/llava-1.5-7b-hf": chat_vicuna,
    "llava-hf/llava-1.5-13b-hf": chat_vicuna,
    "llava-hf/bakLlava-v1-hf": chat_mistral,
    "llava-hf/llava-v1.6-mistral-7b-hf": chat_mistral,
    "llava-hf/llava-v1.6-vicuna-7b-hf": chat_vicuna,
    "llava-hf/llava-v1.6-vicuna-13b-hf": chat_vicuna,
    "llava-hf/llava-v1.6-34b-hf": chat_yi,
}

models = [
    # "llava-hf/vip-llava-13b-hf",
    # "llava-hf/vip-llava-7b-hf",
    # "llava-hf/llava-1.5-7b-hf",
    # "llava-hf/llava-1.5-13b-hf",
    # "llava-hf/bakLlava-v1-hf",
    # "llava-hf/llava-v1.6-mistral-7b-hf",
    # "llava-hf/llava-v1.6-vicuna-7b-hf",
    # "llava-hf/llava-v1.6-vicuna-13b-hf",
    "llava-hf/llava-v1.6-34b-hf",
]


from transformers import AddedToken

for model_id in models:
    processor = AutoProcessor.from_pretrained(model_id)
    print(processor.chat_template)
    # processor.push_to_hub(model_id)

# TEST IN VIPLLAVA
messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s the content of this image?"},
                {"type": "image"},
                ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "This picture shows a red stop sign."},]
        }
    ]
# processor = AutoProcessor.from_pretrained("llava-hf/vip-llava-7b-hf")
# convo = processor.apply_chat_template(messages)
# print(convo)
