# python verify_loading.py --mode single_gpu
# torchrun --nproc_per_node=4 verify_loading.py --mode fsdp
# torchrun --nproc_per_node=4 verify_loading.py --mode tp
# torchrun --nproc_per_node=4 verify_loading.py --mode tp_fsdp
import argparse, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["single_gpu", "fsdp", "tp", "tp_fsdp"], required=True)
args = parser.parse_args()

if args.mode != "single_gpu":
    torch.distributed.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    world_size = torch.distributed.get_world_size()
else:
    rank = 0
    torch.cuda.set_device(0)
    world_size = 1 # single GPU

configs = {
    "single_gpu": None,
    "fsdp": DistributedConfig(fsdp_size=world_size, fsdp_plan="auto"),
    "tp": DistributedConfig(tp_size=world_size, tp_plan="auto", enable_sequence_parallel=True),
    "tp_fsdp": DistributedConfig(tp_size=2, tp_plan="auto", fsdp_size=world_size // 2, fsdp_plan="auto", enable_sequence_parallel=True),
}

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", distributed_config=configs[args.mode], dtype=torch.float32)
if args.mode == "single_gpu":
    model = model.to("cuda:0")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
text = "The capital of France is Paris and the capital of Germany is Berlin and the capital of Italy is Rome"
inputs = tokenizer(text, return_tensors="pt").to(f"cuda:{rank}")

model.eval()
with torch.no_grad():
    loss = model(**inputs, labels=inputs["input_ids"].clone()).loss

if rank == 0:
    print(f"{args.mode}: loss = {loss.item():.4f}")

if args.mode != "single_gpu":
    torch.distributed.destroy_process_group()
