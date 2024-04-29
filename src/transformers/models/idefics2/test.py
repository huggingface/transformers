import requests
from PIL import Image

from transformers import Idefics2ForConditionalGeneration, Idefics2Processor


url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
url_2 = "http://images.cocodataset.org/val2017/000000219578.jpg"

image_1 = Image.open(requests.get(url_1, stream=True).raw)
image_2 = Image.open(requests.get(url_2, stream=True).raw)
images = [image_1, image_2]

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What’s the difference between these two images?"},
            {"type": "image"},
            {"type": "image"},
        ],
    }
]

processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)
model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b", device_map="auto")

# at inference time, one needs to pass `add_generation_prompt=True` in order to make sure the model completes the prompt
text = processor.apply_chat_template(messages, add_generation_prompt=True)

inputs = processor(images=images, text=text, return_tensors="pt").to("cuda")

for k, v in inputs.items():
    print(k, v.shape)

generated_text = model.generate(**inputs, max_new_tokens=500)
generated_text = processor.batch_decode(generated_text, skip_special_tokens=True)[0]
print("Generated text:", generated_text)
