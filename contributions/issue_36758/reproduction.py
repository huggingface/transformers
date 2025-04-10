# NOTE issue: https://github.com/huggingface/transformers/issues/36758 

import pdb
from transformers import AutoModelForCausalLM
model_path = "/scratch/yx3038/model_ckpt/Llama-2-7b-hf"
# model_path = "/scratch/yx3038/model_ckpt/Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu"
)

# for name, _ in model.named_parameters():
    # print(name)
    
# pdb.set_trace()