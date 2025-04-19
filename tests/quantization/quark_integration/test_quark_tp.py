import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig, GenerationConfig, QuarkConfig
import transformers
from torch.distributed.tensor import DTensor, Placement, Replicate

model_id = "amd/Llama-3.1-8B-Instruct-w-int8-a-int8-sym-test"

config = AutoConfig.from_pretrained(model_id)

quant_config = QuarkConfig(**config.quantization_config)
print("ak mode:", quant_config.custom_mode)

rank = None
world_size = -1
tp_plan = None
device = None

try:
   rank = int(os.environ["RANK"])
   world_size = int(os.environ["WORLD_SIZE"])
   tp_plan = "auto"
except Exception as e:
   print(e)

if rank is not None and rank == 1:
   print("torch: ", torch.__version__)
   print("tf: ", transformers.__version__)
   print("tf: ", transformers.__file__)

if rank is not None:
   device = torch.device(f"cuda:{rank}")
   torch.cuda.set_device(device)
   torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

dtype = "auto"
dtype = torch.float16

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, tp_plan=tp_plan,
                                             #quantization_config=quant_config,
                                             device_map = None)

if rank is not None:
   torch.distributed.barrier()
   device_mesh = None
   has_dtensor = 0
   for name, parameter in model.named_parameters():
      if isinstance(parameter.data, torch.distributed.tensor.DTensor):
         has_dtensor = 1
         break
   assert has_dtensor == 1, "TP model must has DTensor"

tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt = "Can I help"

if device is None:
   device  = model.device

inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
outputs = model(inputs)

next_token_logits = outputs[0][:, -1, :]
next_token = torch.argmax(next_token_logits, dim=-1)
response = tokenizer.decode(next_token)

print("\n response = ",response)

if rank is not None:
   torch.distributed.barrier()
   torch.distributed.destroy_process_group()
