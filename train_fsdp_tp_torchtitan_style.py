# torchrun --nproc_per_node=4 train_fsdp_tp_torchtitan_style.py
# LOAD_PRETRAINED=1 torchrun --nproc_per_node=4 train_fsdp_tp_torchtitan_style.py
#
# Minimal standalone training script that reuses torchtitan's components
# (model wrapper, parallelization, loss, optimizer, grad clipping) directly.
# This is the same code path as `./run_train.sh` but without the config system.

import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from torch.distributed.checkpoint import HuggingFaceStorageReader

# ---------- torchtitan imports ----------
from torchtitan.distributed import ParallelDims
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.transformers_modeling_backend.infra.parallelize import (
    apply_fsdp,
    apply_non_moe_tp,
    disable_fsdp_gradient_division,
)
from torchtitan.experiments.transformers_modeling_backend.model.args import (
    HFTransformerModelArgs,
    TitanDenseModelArgs,
)
from torchtitan.experiments.transformers_modeling_backend.model.model import (
    HFTransformerModel,
)

# ---------- transformers imports ----------
from transformers import AutoConfig, AutoTokenizer

IGNORE_INDEX = -100


def build_model_args(hf_model_name: str, seq_len: int) -> HFTransformerModelArgs:
    """Build HFTransformerModelArgs from a HuggingFace model name."""
    hf_config = AutoConfig.from_pretrained(
        hf_model_name, attn_implementation="sdpa", trust_remote_code=True
    )
    hf_config_dict = hf_config.to_dict()

    model_args = HFTransformerModelArgs(titan_dense_args=TitanDenseModelArgs())

    # Map TorchTitan attr names → HF attr names
    for titan_name, hf_name in model_args._tt_to_hf_attribute_map.items():
        if hasattr(hf_config, hf_name):
            setattr(model_args, titan_name, getattr(hf_config, hf_name))

    # Copy all HF config attributes
    for key, value in hf_config_dict.items():
        setattr(model_args, key, value)

    # Override with training-specific settings
    model_args.max_seq_len = seq_len
    model_args.deterministic = False
    model_args.attention_bias = False
    model_args.mlp_bias = False
    model_args.use_cache = False
    model_args.initializer_range = 1.0
    model_args.pruned_heads = getattr(hf_config, "pruned_heads", {})

    if "head_dim" not in hf_config_dict:
        model_args.head_dim = model_args.dim // model_args.num_attention_heads

    return model_args


if __name__ == "__main__":
    # ── Config ──────────────────────────────────────────────────────────
    model_name = "Qwen/Qwen3-0.6B"
    seq_len = 512
    num_steps = 50
    lr = 3e-4
    max_norm = 1.0
    tp_degree = 2
    dp_degree = 2  # FSDP shard degree
    batch_size = 4

    # ── Distributed init ────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    parallel_dims = ParallelDims(
        dp_shard=dp_degree,
        dp_replicate=1,
        tp=tp_degree,
        pp=1,
        ep=1,
        etp=1,
        cp=1,
        world_size=world_size,
    )
    world_mesh = parallel_dims.build_mesh()

    # ── C4 dataset (same as torchtitan) ─────────────────────────────────
    from torchtitan.hf_datasets.text_datasets import build_text_dataloader
    from torchtitan.components.tokenizer import build_hf_tokenizer
    from torchtitan.config.job_config import JobConfig as TTJobConfig
    from types import SimpleNamespace

    tt_tokenizer = build_hf_tokenizer(
        SimpleNamespace(
            model=SimpleNamespace(
                hf_assets_path=snapshot_download(model_name),
                name="transformers_modeling_backend",
                tokenizer_path="",
            )
        )
    )
    dp_rank = parallel_dims.get_mesh("fsdp").get_local_rank()
    dp_world_size = parallel_dims.get_mesh("fsdp").size()
    tt_job_config = TTJobConfig()
    tt_job_config.training.dataset = "c4"
    tt_job_config.training.dataset_path = None
    tt_job_config.training.local_batch_size = batch_size
    tt_job_config.training.seq_len = seq_len
    dataloader = build_text_dataloader(
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tt_tokenizer,
        job_config=tt_job_config,
        infinite=True,
    )

    # ── Model ───────────────────────────────────────────────────────────
    model_args = build_model_args(model_name, seq_len)

    with torch.device("meta"):
        model = HFTransformerModel(model_args)

    # ── Parallelize (same as torchtitan's parallelize_hf_transformers) ──
    tp_mesh = parallel_dims.get_mesh("tp")
    apply_non_moe_tp(
        model,
        tp_mesh,
        loss_parallel=True,  # lm_head output → Shard(-1)
        enable_float8_tensorwise_tp=False,
    )

    dp_mesh = parallel_dims.get_mesh("fsdp")
    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        pp_enabled=False,
    )
    disable_fsdp_gradient_division(model)

    # ── Materialize + init weights ──────────────────────────────────────
    model.to_empty(device=device)
    with torch.no_grad():
        model.init_weights()
    model.train()

    # ── (Optional) Load pretrained weights via DCP ──────────────────────
    # Set LOAD_PRETRAINED=1 to load HF weights. Default: train from random init
    # (matching what torchtitan's run_train.sh does without a checkpoint).
    if os.environ.get("LOAD_PRETRAINED", "0") == "1":
        checkpoint_path = snapshot_download(model_name)
        state_dict = model.state_dict()
        PREFIX = "model."
        hf_keyed = {k[len(PREFIX):]: v for k, v in state_dict.items() if k.startswith(PREFIX)}
        dcp.load(hf_keyed, storage_reader=HuggingFaceStorageReader(checkpoint_path))
        model.load_state_dict({PREFIX + k: v for k, v in hf_keyed.items()})
        if rank == 0:
            print("Pretrained weights loaded via DCP.")
    else:
        if rank == 0:
            print("Training from random init (no pretrained weights).")

    # ── Optimizer ───────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,
    )

    # ── loss_parallel context (logits are Shard(-1) on TP mesh) ─────────
    loss_parallel_enabled = parallel_dims.tp_enabled
    train_context = dist_utils.get_train_context(loss_parallel_enabled)

    # ── Training loop ───────────────────────────────────────────────────
    data_iterator = iter(dataloader)
    for step in range(num_steps):
        optimizer.zero_grad()

        # torchtitan dataloader yields ({"input": input_ids}, labels)
        # both of shape (batch, seq_len) — already shifted, no padding.
        input_dict, labels = next(data_iterator)
        input_ids = input_dict["input"].to(device)
        labels = labels.to(device)

        # No padding in C4 stream — all tokens are valid
        local_valid_tokens = (labels != IGNORE_INDEX).sum().to(device)
        global_valid_tokens = dist_utils.dist_sum(
            local_valid_tokens, parallel_dims.get_mesh("batch")
        )

        # Forward + loss under train_context (enables loss_parallel if TP)
        # input_ids and labels are same length (seq_len), already shifted by dataloader.
        # pred aligns directly with labels — no slicing needed.
        with train_context():
            pred = model(input_ids)  # (batch, seq_len, vocab) as Shard(-1) DTensor
            loss_sum = F.cross_entropy(
                pred.flatten(0, 1).float(),
                labels.flatten(0, 1),
                reduction="sum",
                ignore_index=IGNORE_INDEX,
            )
            loss = loss_sum / global_valid_tokens
            del pred
            loss.backward()

        # Gradient clipping (torchtitan's implementation)
        grad_norm = dist_utils.clip_grad_norm_(
            list(model.parameters()), max_norm, foreach=True
        )

        optimizer.step()

        if rank == 0:
            print(
                f"Step {step:>4d} | Loss: {loss.item():.4f} | "
                f"Grad norm: {grad_norm.item():.4f}"
            )

    dist.destroy_process_group()
