# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Dummy testing script to test transformers + FSDP + CPU offloading
# works correctly
from functools import partial
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

# verify we have FSDP activation support ready by importing:
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from transformers.models.llama.modeling_llama import LlamaDecoderLayer

model_id = "HuggingFaceM4/tiny-random-Llama3ForCausalLM"

model = AutoModelForCausalLM.from_pretrained(model_id)

model.train()
model.gradient_checkpointing_enable()

accelerator = Accelerator()
model = accelerator.prepare(model)

check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    offload_to_cpu=False,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

apply_activation_checkpointing(
    model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
)

rand_input = torch.LongTensor([[0, 1, 0, 1]]).to(0)

_ = model(rand_input)
