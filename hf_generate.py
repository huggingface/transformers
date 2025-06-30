"""
torchrun --nproc_per_node=2 hf_generate.py 
"""

from transformers import AutoTokenizer, OpenAIMoeForCausalLM
import torch
import os
import logging
import torch.distributed as dist
from torch.distributed.tensor.experimental import context_parallel
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.distributed.device_mesh import DeviceMesh

model_id = "ft-hf-o-c/random-checkpoint-converted-20b"


# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    tp_size = int(os.environ.get("TP_SIZE", 4))

    # Initialize distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        mesh = torch.arange(world_size).reshape(tp_size)
        world_mesh = DeviceMesh(device_type="cuda", mesh=mesh, mesh_dim_names=("tp",))
        tp_mesh = world_mesh["tp"]
        logger.info(f"Created DeviceMesh: {world_mesh}")
        logger.info(
            f"Distributed setup - Rank: {rank}, World size: {world_size}, Local rank: {local_rank}, TP: {tp_mesh.get_local_rank()}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # messages = [
    #     {"role": "user", "content": "Who are you?"},
    # ]
    # inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    inputs = tokenizer("Hello! How are you?", return_tensors="pt")
    model = OpenAIMoeForCausalLM.from_pretrained(
        model_id,
        device_mesh=tp_mesh if dist.is_initialized() else None,
        tp_plan="auto",
        tp_size=tp_size,
        torch_dtype=torch.bfloat16,
        # torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    logger.info(f"Model loaded onto device mesh: {tp_mesh}")
    device = torch.device(f"cuda:{local_rank}")
    logger.info(f"Using device: {device} for non-model tensors")
    model.eval()

    outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
    outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])
    print(outputs[0])

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()