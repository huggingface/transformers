import requests
from PIL import Image

from transformers import AutoProcessor


processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)

url = "https://upload.wikimedia.org/wikipedia/commons/f/f3/Zinedine_Zidane_by_Tasnim_03.jpg"
test_image = Image.open(requests.get(url, stream=True).raw)

# prepare image and prompt for the model
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract JSON."},
            {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[test_image], return_tensors="pt")
for k, v in inputs.items():
    print(k, v.shape)

print(processor.batch_decode(inputs.input_ids))
