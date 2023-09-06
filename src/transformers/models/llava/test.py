from transformers.models.llava.modeling_llava import LlavaLlamaForCausalLM
from transformers import LlamaForCausalLM, AutoTokenizer
import torch
from transformers.models.llava.processing_llava import LlavaProcessor
from transformers import CLIPImageProcessor, CLIPVisionModel
from PIL import Image
CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

import requests

#processor = AutoTokenizer.from_pretrained("shauray/llva-llama-2-7B")
url = "https://llava-vl.github.io/static/images/view.jpg"
#url = "https://images.pexels.com/photos/18138908/pexels-photo-18138908/free-photo-of-cute-cat-on-window.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
text = "what are the things I should be cautious about while travelling to this place alone"

#imager = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
#vision = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda",dtype=torch.float16)
#vision.requires_grad_(False)

def tokenizer_image_token(prompt, tokenizer, image_token_index, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def feature_select(image_forward_outs):
  image_features = image_forward_outs.hidden_states[-2]
  image_features = image_features[:, 1:]
  return image_features

#acc = {}
#text = DEFAULT_IMAGE_TOKEN + '\n' + text
#processor.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)

#another = processor(text,return_tensors="pt").to("cuda")
#input_ids = tokenizer_image_token(prompt=text,
#tokenizer=processor,
#image_token_index=200,
#return_tensors="pt").unsqueeze(0).to("cuda")
#print(input_ids)
#acc["input_ids"] = input_ids
#another["attention_mask"] = None
#encoding.update(tokens)
#output = vision(imager.preprocess(image,return_tensors="pt")["pixel_values"].half().cuda(),
#output_hidden_states=True)
#image_features = feature_select(output).to(torch.float16)
#acc["pixel_values"] = image_features
#another["pixel_values"] = image_features

text = "what are the things I should be cautious about while travelling to this place alone"
processor = LlavaProcessor.from_pretrained("shauray/llva-llama-2-7B",device="cuda",torch_dtype=torch.float16)
inputs = processor(text=text,images=image, return_tensors="pt").to("cuda",dtype=torch.float16)
#print(inputs["pixel_values"])
#print(acc,inputs)
import gc
gc.collect()
model = LlavaLlamaForCausalLM.from_pretrained("shauray/llva-llama-2-7B",torch_dtype=torch.float16, device_map="cuda",
low_cpu_mem_usage=True).to("cuda")



#inputs = processor(images=image, return_tensors="pt")
#tokenizer = AutoTokenizer.from_pretrained("shauray/llva-llama-2-7B")
#tokens = tokenizer("what soes this image describe?",return_tensors="pt").to("cuda")
#tokens["pixel_values"] = inpu
outputs = model.generate(
        **acc,
        do_sample=True,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
)

print(processor.decode(outputs[0, input_ids.shape[1]:]).strip())



