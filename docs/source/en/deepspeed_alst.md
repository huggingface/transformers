<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# DeepSpeed ALST

```text
     [░░░░░░░░│▒▒▒▒▒▒▒▒│▓▓▓▓▓▓▓▓│████████]  ← sequence
      │           │          │          │
      ▼           ▼          ▼          ▼
  ┌───────┐   ┌───────┐  ┌───────┐  ┌───────┐
  │   M   │   │   M   │  │   M   │  │   M   │  ← same model
  └───────┘   └───────┘  └───────┘  └───────┘
```

DeepSpeed’s ALST/Ulysses sequence parallelism enables training with very long sequences by splitting the sequence across multiple GPUs. This is particularly useful for training large language models with very long sequence lengths.

Arctic Long Sequence Training (ALST) uses a combination of sharding inputs along the sequence dimension and attention head parallelism. With this approach, you can train models with sequence lengths up to 500K tokens on a single H100 GPU, 3.7M on a single node, or 15M tokens on just four nodes with Llama-8B. The implementation described here enables one component of the full ALST system. For additional optimizations like TiledMLP and activation checkpoint offloading, refer to the DeepSpeed ALST tutorial.

For more detailed information about sequence parallelism, see the Accelerate Sequence Parallelism guide.

To enable ALST/Ulysses sequence parallelism with Trainer, configure parallelism_config in TrainingArguments. Sequence parallelism is configured via Accelerate’s ParallelismConfig and requires an Accelerate version higher than 1.12.0.

```py
from accelerate.utils import ParallelismConfig, DeepSpeedSequenceParallelConfig

# Example: 4 GPUs with sp_size=4, dp_replicate_size=1 (no data parallelism)
# Ensure total_size = dp_replicate_size * dp_shard_size * sp_size = 1 * 1 * 4 = 4 GPUs
parallelism_config = ParallelismConfig(
    sp_backend="deepspeed",
    sp_size=4,  # Number of GPUs to split sequence across
    dp_replicate_size=1,  # Explicit: no data parallelism
    sp_handler=DeepSpeedSequenceParallelConfig(
        sp_seq_length_is_variable=True,
        sp_attn_implementation="sdpa",
    ),
)

training_args = TrainingArguments(
    ...,
    deepspeed="path/to/deepspeed_config.json",
    parallelism_config=parallelism_config,
)
```

You can also configure sequence parallelism using an Accelerate config file.

```yaml
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: path/to/ds_config.json
machine_rank: 0
num_machines: 1
num_processes: 4  # Total number of processes
parallelism_config:
  parallelism_config_sp_size: 4  # Sequence parallel size
  parallelism_config_dp_replicate_size: 1  # Must be: dp_replicate_size * dp_shard_size * sp_size = num_processes
  parallelism_config_sp_backend: deepspeed
  parallelism_config_sp_seq_length_is_variable: true
  parallelism_config_sp_attn_implementation: sdpa
```

Important configuration parameters include the following.

sp_backend must be set to "deepspeed" to use ALST/Ulysses sequence parallelism.
sp_size is the degree of sequence parallelism. For example, sp_size=4 means 4 GPUs will process a single sequence in parallel. You need at least 2 GPUs to enable sequence parallelism. Data feeding: Each rank receives a unique data stream from the DataLoader (like DP). Batch size calculation: The effective dp_world_size = world_size / sp_size. So with 4 GPUs and sp_size=4, each of the 4 ranks gets different samples from the DataLoader, but dp_world_size=1 for total batch size calculations
sp_seq_length_is_variable determines how sequence lengths are handled. When set to True (recommended), the implementation adapts to varying sequence lengths between batches. When False, all sequences must be padded to a fixed length specified by sp_seq_length.
sp_attn_implementation specifies the attention implementation to use. Supported values are "sdpa", "flash_attention_2", or "flash_attention_3". Flash Attention is recommended for best performance, especially with multiple samples in a batch, because SDPA may incorrectly attend across sample boundaries.

Sequence parallelism requires your model to use one of the supported attention implementations (sdpa, flash_attention_2, or flash_attention_3). The eager attention implementation is not supported because it doesn’t properly handle position_ids.

When using sequence parallelism, ensure your sequences are properly padded. Use pad_to_multiple_of in your data collator to ensure sequences are divisible by sp_size. For example, with sp_size=4, set pad_to_multiple_of=4 or higher.

```py
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=4,  # Ensure sequences are divisible by sp_size
)
```

When using sp_size with multiple GPUs, you must explicitly set dp_replicate_size or dp_shard_size to ensure total_size = dp_replicate_size * dp_shard_size * sp_size equals your total number of GPUs. For example, with 8 GPUs and sp_size=4, you must set dp_replicate_size=2 (since 2 × 1 × 4 = 8):

```py
parallelism_config = ParallelismConfig(
    sp_backend="deepspeed",
    sp_size=4,
    dp_replicate_size=2,
    sp_handler=DeepSpeedSequenceParallelConfig(
        sp_seq_length_is_variable=True,
        sp_attn_implementation="flash_attention_2",
    ),
)
```

Trainer automatically handles the special requirements for sequence parallelism including:

Adapting the data loader via DeepSpeed’s UlyssesSPDataLoaderAdapter to shard sequences across GPUs. Important: Unlike Tensor Parallelism (where all ranks must receive identical data), each rank with SP receives a unique data stream from the DataLoader (similar to DP). The adapter handles distributing sequence chunks across SP ranks internally, so your DataLoader should continue feeding different samples to each rank.
Generating position_ids when not provided
Creating shift_labels for causal language modeling
Aggregating loss across sequence parallel ranks with proper masking for -100 labels
You can launch training with sequence parallelism using the accelerate launch command.

```shell
accelerate launch --config_file alst_config.yaml your_training_script.py \
--output_dir output_dir \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1
```