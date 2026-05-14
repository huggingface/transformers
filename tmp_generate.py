import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device(f"cuda:{rank}")
torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=rank)

if __name__ == "__main__":
    distributed_config = DistributedConfig(tp_size=8, tp_plan="auto")
    model = AutoModelForCausalLM.from_pretrained(model_id, distributed_config=distributed_config, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    messages = [
        {"role": "user", "content": "What do you think about life?"},
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    input_size = inputs.input_ids.shape[-1]

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50, do_sample=False)

    text = tokenizer.batch_decode(output[:, input_size:])[0]
    if rank == 0:
        print("message:")
        print(messages)
        print("========================")
        print("response:")
        print(text)

torch.distributed.destroy_process_group()