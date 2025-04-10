# import torch
# import transformers
# model = transformers.AutoModel.from_pretrained("/scratch/yx3038/model_ckpt/Llama-3.1-8B-Instruct/", torch_dtype=torch.bfloat16)
# # model.push_to_hub('wizeng23/Llama-test')
# print(type(model))
# import pdb; pdb.set_trace()
# tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
# tokenizer.push_to_hub('wizeng23/Llama-test')

import os
import torch
import transformers

model_path = "/scratch/yx3038/model_ckpt/Llama-3.1-8B-Instruct/"

llama_model = transformers.AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16)
# check if lm_head is in the named_parameters of LlamaForCausalLM
print(any("lm_head" in name for name, _ in llama_model.named_parameters()))
print(type(llama_model))
# save LlamaModel and check the size of the safetensors
# llama_model.save_pretrained('./ckpt/llama_model')
print('model-00004-of-00004.safetensors size: {:.2e} bytes'.format(
    os.path.getsize('./ckpt/llama_model/model-00004-of-00004.safetensors')
))

llama_for_causal_lm = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
# check if lm_head is in the named_parameters of LlamaForCausalLM
print(any("lm_head" in name for name, _ in llama_for_causal_lm.named_parameters()))
print([name for name, _ in llama_for_causal_lm.named_parameters() if "lm_head" in name])
# check if lm_head is in the state_dict of LlamaForCausalLM
print(any("lm_head" in key for key in llama_for_causal_lm.state_dict().keys()))
print([key for key in llama_for_causal_lm.state_dict().keys() if "lm_head" in key])
# save LlamaForCausalLM and check the size of the safetensors
# llama_for_causal_lm.save_pretrained('./ckpt/llama_for_causal_lm')
print('model-00004-of-00004.safetensors size: {:.2e} bytes'.format(
    os.path.getsize('./ckpt/llama_for_causal_lm/model-00004-of-00004.safetensors')
))

# load LlamaModel and LlamaForCausalLM from the save paths
llama_model = transformers.AutoModel.from_pretrained('./ckpt/llama_model', torch_dtype=torch.bfloat16)
llama_for_causal_lm = transformers.AutoModelForCausalLM.from_pretrained('./ckpt/llama_model', torch_dtype=torch.bfloat16)




