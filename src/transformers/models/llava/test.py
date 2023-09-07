from transformers.models.llava.modeling_llava import LlavaLlamaForCausalLM
from transformers import LlamaForCausalLM, AutoTokenizer
import torch
from transformers.models.llava.processing_llava import LlavaProcessor
from transformers import CLIPImageProcessor, CLIPVisionModel
from PIL import Image

import requests

#processor = AutoTokenizer.from_pretrained("shauray/llva-llama-2-7B")
url = "https://llava-vl.github.io/static/images/view.jpg"
#url = "https://pbs.twimg.com/media/F5WP8mKXUAADlQH?format=jpg&name=small"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
text = "How would you best describe this image?"

model = LlavaLlamaForCausalLM.from_pretrained("shauray/llva-llama-2-7B",torch_dtype=torch.float16, device_map="cuda",
low_cpu_mem_usage=True).to("cuda")

processor = LlavaProcessor.from_pretrained("shauray/llva-llama-2-7B",device="cuda",torch_dtype=torch.float16)

inputs = processor(text=text,images=image, return_tensors="pt").to("cuda",dtype=torch.float16)
outputs = model.generate(
        **inputs,
        do_sample=True,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=.2,
)

print(processor.decode(outputs[0, inputs["input_ids"].shape[1]:]).strip())



