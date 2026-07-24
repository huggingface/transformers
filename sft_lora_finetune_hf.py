# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

import os
import socket

import torch
import torch_neuronx
from datasets import load_dataset
from accelerate import ParallelismConfig
from torch.profiler import profile, ProfilerActivity
from torch_neuronx.profiling import NeuronConfig, ProfileMode, NeuronProfiler
from transformers import AutoModelForCausalLM, AutoConfig, TrainerCallback
from transformers.modeling_outputs import CausalLMOutputWithPast

from peft import get_peft_model
from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser, get_peft_config

# PyTorch/Neuron profiler window (disabled unless PROFILE_NUM_STEPS > 0), mirroring
# --profile_step_start/--profile_num_steps/--profile_output_dir in sft_lora_finetune_custom.py.
PROFILE_STEP_START = int(os.environ.get("PROFILE_STEP_START", "0"))
PROFILE_NUM_STEPS = int(os.environ.get("PROFILE_NUM_STEPS", "0"))
PROFILE_OUTPUT_DIR = os.environ.get("PROFILE_OUTPUT_DIR", "./pt-profile")
PROFILE_MAX_EVENTS_PER_NC = int(os.environ.get("PROFILE_MAX_EVENTS_PER_NC", "4000000"))


class NeuronProfilerCallback(TrainerCallback):
    def __init__(self, start_step, num_steps, output_dir, max_events_per_nc):
        self.start_step = start_step
        self.end_step = start_step + num_steps - 1
        self.neuron_config = NeuronConfig(
            modes=[
                ProfileMode.DEVICE, 
                ProfileMode.RUNTIME,
                ProfileMode.CPU_UTIL, 
                ProfileMode.HOST_MEMORY
            ],
            profile_output_dir=output_dir,
            capture_enabled_for_nc="0,1",
            max_events_per_nc=max_events_per_nc,
        )
        self.exporter = NeuronProfiler(self.neuron_config)
        self.profiler_ctx = None

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step + 1 == self.start_step:
            self.profiler_ctx = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                experimental_config=self.neuron_config,
                on_trace_ready=self.exporter.export_trace,
                with_stack=True,
            )
            self.profiler_ctx.__enter__()

    def on_step_end(self, args, state, control, **kwargs):
        if self.profiler_ctx is not None:
            # torch.neuron.synchronize() 
            torch_neuronx.synchronize()
            if state.global_step == self.end_step:
                self.profiler_ctx.__exit__(None, None, None)
                self.profiler_ctx = None


def main(script_args, training_args, model_args):
    # ---------------------------------------------------------------------------
    # Tensor Parallelism
    # ---------------------------------------------------------------------------
    tp_size = int(os.environ.get("TP_SIZE", "1"))

    tp_plan = {
        "model.layers.*.self_attn.q_proj": "colwise",
        "model.layers.*.self_attn.k_proj": "colwise",
        "model.layers.*.self_attn.v_proj": "colwise",
        "model.layers.*.self_attn.o_proj": "rowwise",
        "model.layers.*.mlp.gate_proj": "colwise",
        "model.layers.*.mlp.up_proj": "colwise",
        "model.layers.*.mlp.down_proj": "rowwise",
    }

    kwargs = {}
    if tp_size > 1:
        training_args.parallelism_config = ParallelismConfig(tp_size=tp_size)
        kwargs["tp_plan"] = tp_plan
        kwargs["tp_size"] = tp_size

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=True,
    )
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        revision=model_args.model_revision,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        **kwargs,
    )

    peft_config = get_peft_config(model_args)

    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    callbacks = []
    # if PROFILE_NUM_STEPS > 0:
    #     callbacks.append(
    #         NeuronProfilerCallback(
    #             PROFILE_STEP_START, PROFILE_NUM_STEPS, PROFILE_OUTPUT_DIR, PROFILE_MAX_EVENTS_PER_NC
    #         )
    #     )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    main(script_args, training_args, model_args)
