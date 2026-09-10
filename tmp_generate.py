import argparse
import os

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_id = "Qwen/Qwen3-14B"
# model_id = "Qwen/Qwen3-0.6B"
# model_id = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
# model_id = "Qwen/Qwen3-30B-A3B-Instruct-2507"

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device(f"cuda:{rank}")
# Need to be initialized explicitly to use the `barrier` before loading
torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=rank)

@record
def main(args):

    distributed_config = DistributedConfig(tp_size=4, tp_plan="auto")
    model = AutoModelForCausalLM.from_pretrained(model_id, distributed_config=distributed_config, dtype=torch.bfloat16)
    # model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    messages = [
        {"role": "user", "content": "What do you think about life?"},
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    input_size = inputs.input_ids.shape[-1]

    if args.profile:
        # Warmup
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            output = model.generate(**inputs, max_new_tokens=2, do_sample=False)

        if rank == 0:
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
            prof.export_chrome_trace("trace.json")
    else:
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)

    text = tokenizer.batch_decode(output[:, input_size:])[0]
    if rank == 0:
        print(text)

parser = argparse.ArgumentParser()
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()

main(args)

torch.distributed.destroy_process_group()