# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import textwrap
from typing import Any, Callable

from transformers import is_torch_available, is_torch_xpu_available
from transformers.testing_utils import (
    TestCasePlus,
    backend_device_count,
    backend_torch_accelerator_module,
    execute_subprocess_async,
    get_torch_dist_unique_port,
    require_torch_multi_accelerator,
    torch_device,
    torchrun,
)
from transformers.utils import is_ccl_available, is_ipex_available


if is_torch_available():
    import functools

    import torch

    if is_torch_xpu_available():
        if is_ipex_available():
            import intel_extension_for_pytorch  # noqa: F401
        if is_ccl_available():
            import oneccl_bindings_for_pytorch  # noqa: F401
    import torch.distributed
    from torch.distributed._composable.fsdp import fully_shard, register_fsdp_forward_method
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block

    data = 4 * [
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
    ]

    def manage_process_group(func: Callable[..., Any]) -> Callable[..., Any]:
        """Manage the creation and destruction of the distributed process group for the wrapped function."""

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            device_count = backend_device_count(torch_device)
            torch.distributed.init_process_group(world_size=device_count)
            try:
                return func(*args, **kwargs)
            finally:
                torch.distributed.destroy_process_group()

        return wrapped

    @manage_process_group
    def fsdp_generate():
        torch_accelerator_module = backend_torch_accelerator_module(torch_device)
        torch_accelerator_module.set_device(device := torch.device(rank := torch.distributed.get_rank()))

        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(device)

        fsdp_model = FullyShardedDataParallel(
            model,
            auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={GPT2Block}),
            limit_all_gathers=True,
            use_orig_params=True,
        )

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        batch = tokenizer(data[rank], return_tensors="pt", return_attention_mask=True).to(device)

        with FullyShardedDataParallel.summon_full_params(fsdp_model):
            _ = fsdp_model.module.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=30,
            )

    @manage_process_group
    def fsdp2_generate():
        torch_accelerator_module = backend_torch_accelerator_module(torch_device)
        torch_accelerator_module.set_device(device := torch.device(rank := torch.distributed.get_rank()))

        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(device)

        mesh = init_device_mesh(device.type, (torch.distributed.get_world_size(),))
        for submodule in model.modules():
            if isinstance(submodule, GPT2Block):
                fully_shard(submodule, mesh=mesh)
        fully_shard(model, mesh=mesh)

        register_fsdp_forward_method(model, "generate")

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        batch = tokenizer(data[rank], return_tensors="pt", return_attention_mask=True).to(device)

        _ = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=30,
        )


class TestFSDPGeneration(TestCasePlus):
    @require_torch_multi_accelerator
    def test_fsdp_generate(self):
        device_count = backend_device_count(torch_device)
        distributed_args = f"""--nproc_per_node={device_count}
            --master_port={get_torch_dist_unique_port()}
            {self.test_file_dir}/test_fsdp.py
        """.split()
        args = ["--fsdp"]
        cmd = ["torchrun"] + distributed_args + args
        execute_subprocess_async(cmd, env=self.get_env())
        # successful return here == success - any errors would have caused an error in the sub-call

    @require_torch_multi_accelerator
    def test_fsdp2_generate(self):
        device_count = backend_device_count(torch_device)

        distributed_args = f"""--nproc_per_node={device_count}
            --master_port={get_torch_dist_unique_port()}
            {self.test_file_dir}/test_fsdp.py
        """.split()
        args = ["--fsdp2"]
        cmd = ["torchrun"] + distributed_args + args
        execute_subprocess_async(cmd, env=self.get_env())
        # successful return here == success - any errors would have caused an error in the sub-call


class TestFSDPGenericTaskModel(TestCasePlus):
    nproc_per_node = 2

    def test_generic_task_model_can_be_sharded(self):
        script_to_run = textwrap.dedent(
            """
            import torch
            from torch.distributed.fsdp import fully_shard
            from transformers import AutoModelForTokenClassification

            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://"
            )
            rank = torch.distributed.get_rank()
            if torch.cuda.is_available():
                torch.cuda.set_device(rank)

            # Make sure it works
            model = AutoModelForTokenClassification.from_pretrained("Qwen/Qwen2-0.5B")
            module = fully_shard(model)

            torch.distributed.destroy_process_group()
            """
        )
        torchrun(script_to_run, self.nproc_per_node, env=self.get_env())


if __name__ == "__main__":
    # The script below is meant to be run under torch.distributed, on a machine with multiple GPUs:
    #
    # PYTHONPATH="src" python -m torch.distributed.run --nproc_per_node 2 --output_dir output_dir ./tests/generation/test_fsdp.py --fsdp

    class CLIArgs(argparse.Namespace):
        fsdp: bool
        fsdp2: bool

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fsdp", action="store_true")
    group.add_argument("--fsdp2", action="store_true")
    args = parser.parse_args(namespace=CLIArgs())

    if args.fsdp:
        fsdp_generate()
    elif args.fsdp2:
        fsdp2_generate()
    else:
        raise ValueError("Missing test selection")
