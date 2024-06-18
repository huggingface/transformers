from PIL import Image
import requests
import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from time import perf_counter

torch._logging.set_logs(recompiles=True)

model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)


processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

prompt = "[INST] <image>\nWhat is shown in the bottom right section of this collage? [/INST]"
pdf_url = "https://pdfclown.org/wp-content/uploads/2011/04/texthighlight.jpg"
quote_url  ="https://i.sstatic.net/IvV2y.png"
collage_url = "https://i0.wp.com/mirthandmotivation.com/wp-content/uploads/2017/07/super-collage2a.jpg?resize=600%2C600&ssl=1"

image = Image.open(requests.get(collage_url, stream=True).raw)
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

init_mem = torch.cuda.memory_allocated()
torch.cuda.reset_max_memory_allocated()

# Generate
start = perf_counter()
generate_ids = model.generate(**inputs, min_new_tokens=100, max_new_tokens=100)
out = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(out)

print(f"Time: {(perf_counter() - start):.05f}")
print(f"Mem: {(torch.cuda.max_memory_allocated() - init_mem) // 1024 ** 2}")
print(f"Mem: {(torch.cuda.max_memory_reserved() - init_mem) // 1024 ** 2}")
