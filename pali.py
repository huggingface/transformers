from PIL import Image
import requests
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch



from transformers import AutoTokenizer, AutoModelForCausalLM

# model_id = "meta-llama/Llama-2-7b-chat-hf"
# tok = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager")
# inputs = tok("Hello", return_tensors="pt").to(model.device)
# 
# model.generate(**inputs, max_new_tokens=4)



model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-pt-224", attn_implementation="eager")
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

prompt = "answer en What color is the cat?"
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate
generate_ids = model.generate(**inputs, do_sample=False, min_new_tokens=6, max_new_tokens=30)
out = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(out)

