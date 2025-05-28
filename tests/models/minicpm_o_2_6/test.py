from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
import torch

model_path = "xxx"

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    attn_implementation='sdpa',
    torch_dtype=torch.bfloat16,
    init_vision=True,
    init_audio=True,
    init_tts=True
)


tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    local_files_only=True,
    )

# device = torch.device("mps")
# model.to(device)

from PIL import Image

image = Image.open('xxx.jpeg').convert('RGB')
question = 'What is in the image?'
msgs = [{'role': 'user', 'content': [image, question]}]
res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(res)

res = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    stream=True
)
generated_text = ""
for new_text in res:
    generated_text += new_text
    print(new_text, flush=True, end='')