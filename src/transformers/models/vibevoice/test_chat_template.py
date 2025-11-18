from transformers import AutoTokenizer


model_path = "bezzam/VibeVoice-1.5B"

# Original approach:
# 1. System prompt: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/vibevoice_processor.py#L41
# 2. Adding voice input: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/vibevoice_processor.py#L403
# 3. Adding conversation script: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/vibevoice_processor.py#L271

# below chat template is already default of `bezzam/VibeVoice-1.5B`
chat_template = """{%- set system_prompt = system_prompt | default(" Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n") -%}
{{ system_prompt -}}
{%- set speech_start_token = speech_start_token | default("<|vision_start|>") %}
{%- set speech_end_token = speech_end_token | default("<|vision_end|>") %}
{%- set speech_diffusion_token = speech_diffusion_token | default("<|vision_pad|>") %}
{%- set ns = namespace(speakers_with_audio="") %}
{%- for message in messages %}
    {%- set role = message['role'] %}
    {%- set content = message['content'] %}
    {%- set has_audio = content | selectattr('type', 'equalto', 'audio') | list | length > 0 %}
    {%- if has_audio and role not in ns.speakers_with_audio %}
        {%- set ns.speakers_with_audio = ns.speakers_with_audio + role + "," %}
    {%- endif %}
{%- endfor %}

{%- if ns.speakers_with_audio %}
{{ " Voice input:\n" }}
{%- for speaker in ns.speakers_with_audio.rstrip(',').split(',') %}
{%- if speaker %}
 Speaker {{ speaker }}:{{ speech_start_token }}{{ speech_diffusion_token }}{{ speech_end_token }}{{ "\n" }}
{%- endif %}
{%- endfor %}
{%- endif %}
 Text input:{{ "\n" }}

{%- for message in messages %}
    {%- set role = message['role'] %}
    {%- set text_items = message['content'] | selectattr('type', 'equalto', 'text') | list %}
    {%- for item in text_items %}
 Speaker {{ role }}: {{ item['text'] }}{{ "\n" }}
    {%- endfor %}
{%- endfor %}
 Speech output:{{ "\n" }}{{ speech_start_token }}"""

conversation = [
    [   # Script 1
        {"role": "0", "content": [
            {"type": "text", "text": "Hello, how are you?"},
            {"type": "audio", "path": "src/transformers/models/vibevoice/voices/en-Alice_woman.wav"}  # first time for speaker 0
        ]},
        {"role": "1", "content": [
            {"type": "text", "text": "I'm fine, thank you! And you?"},
            {"type": "audio", "path": "src/transformers/models/vibevoice/voices/en-Frank_man.wav"}  # first time for speaker 1
        ]},
        {"role": "0", "content": [{"type": "text", "text": "Nice weather today, right?"}]},
        {"role": "1", "content": [{"type": "text", "text": "Absolutely, it's beautiful."}]},
    ],
    [   # Script 2
        {"role": "0", "content": [{"type": "text", "text": "I'm doing well, thanks for asking."}]},
        {"role": "1", "content": [{"type": "text", "text": "That's great to hear."}]},
    ],
]

tokenizer = AutoTokenizer.from_pretrained(model_path)


print("=== DEBUG: Script 1 ===")
for i, message in enumerate(conversation[0]):
    role = message['role']
    content = message['content']
    has_audio = any(item['type'] == 'audio' for item in content)
    print(f"Message {i}: role={role}, has_audio={has_audio}")


rendered_batch = tokenizer.apply_chat_template(
    conversation, 
    tokenize=False,
    # chat_template=chat_template,  # Comment to use default from the model
)

print("=== Rendered Batch ===")
for i, script in enumerate(rendered_batch):
    print(f"--- Script {i+1} ---")
    print(script)
