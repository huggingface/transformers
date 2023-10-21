import io

import requests
from PIL import Image

from transformers import FuyuForCausalLM, FuyuProcessor


processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b")

url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bus.png"
image = Image.open(io.BytesIO(requests.get(url).content))

text_prompt_coco_captioning = "Generate a coco-style caption.\n"

inputs = processor(text=text_prompt_coco_captioning, images=image, return_tensors="pt")
generated_tokens = model.generate(**inputs, max_new_tokens=10)

text = processor.batch_decode(generated_tokens)
end_sequence = text[0].split("\x04")[1]
clean_sequence = (
    end_sequence[: end_sequence.find("|ENDOFTEXT|") + len("|ENDOFTEXT|")]
    if "|ENDOFTEXT|" in end_sequence
    else end_sequence
)

print(clean_sequence)
