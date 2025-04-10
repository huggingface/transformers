# NOTE issue: https://github.com/huggingface/transformers/issues/36598

from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoModelForCausalLM

import pdb
import torch

model_name = "/scratch/yx3038/model_ckpt/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# model_name = "/scratch/yx3038/model_ckpt/Llama-3.1-8B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )

# pdb.set_trace()

# Check named_parameters for lm_head
has_lm_head_in_named_params = any("lm_head" in name for name, _ in model.named_parameters())
print(f"lm_head in named_parameters(): {has_lm_head_in_named_params}")

# Check state_dict for lm_head
has_lm_head_in_state_dict = any("lm_head" in key for key in model.state_dict().keys())
print(f"lm_head in state_dict(): {has_lm_head_in_state_dict}")

pdb.set_trace()

# Check named_parameters for embed_token
embed_token_in_named_params = any("model.embed_tokens.weight" in name for name, _ in model.named_parameters())
print(f"embed_token weights in amed_parameters(): {embed_token_in_named_params}")

# Check embed_token is equal to lm_head
embed_token_eq_lm_head = torch.all(model.state_dict()['model.embed_tokens.weight'] == model.state_dict()['lm_head.weight'])
print(f"embed_token weights equal to lm_head weights: {embed_token_eq_lm_head}")