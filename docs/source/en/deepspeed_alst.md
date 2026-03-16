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

# Ulysses sequence parallelism

Ulysses sequence parallelism (SP) trains on very long sequences by splitting them across multiple GPUs. To compute attention correctly, an all-to-all collective swaps the sharding dimension from sequence to attention heads. Each GPU then has the full sequence and computes attention locally over a subset of heads. A second all-to-all returns to the sequence-sharded layout so the rest of the forward pass continues locally on each chunk.

```text
  G0  G1  G2  G3           G0  G1  G2  G3           G0  G1  G2  G3
  ┌──┬──┬──┬──┐            ┌──────────────┐          ┌──┬──┬──┬──┐
h │░░│▒▒│▓▓│██│         G0 │░░  ░░  ░░  ░░│        h │░░│▒▒│▓▓│██│
e │░░│▒▒│▓▓│██│ ──────► G1 │▒▒  ▒▒  ▒▒  ▒▒│ ─────► e │░░│▒▒│▓▓│██│
a │░░│▒▒│▓▓│██│         G2 │▓▓  ▓▓  ▓▓  ▓▓│        a │░░│▒▒│▓▓│██│
d │░░│▒▒│▓▓│██│         G3 │██  ██  ██  ██│        d │░░│▒▒│▓▓│██│
  └──┴──┴──┴──┘            └──────────────┘          └──┴──┴──┴──┘
  seq sharded              heads sharded              seq sharded

```

> [!NOTE]
> This guide covers the Ulysses sequence parallelism component of [ALST](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/) (Arctic Long Sequence Training). The full ALST system also includes TiledMLP and activation checkpoint offloading, which aren't available in Transformers. See the [DeepSpeed ALST tutorial](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/) for the complete system.

## Configure

Sequence parallelism requires Accelerate v1.12.0 and at least 2 GPUs. Configure sequence parallelism in Accelerate's [`~accelerate.ParallelismConfig`] and pass it to [`~TrainingArguments.parallelism_config`] or an [Accelerate config file](./accelerate#accelerate-config-file).

<hfoptions id="launch">
<hfoption id="parallelism_config">

```py
from accelerate.utils import ParallelismConfig, DeepSpeedSequenceParallelConfig

parallelism_config = ParallelismConfig(
    sp_backend="deepspeed",
    sp_size=4,
    dp_replicate_size=1,
    sp_handler=DeepSpeedSequenceParallelConfig(
        sp_seq_length_is_variable=True,
        sp_attn_implementation="flash_attention_2",
    ),
)

training_args = TrainingArguments(
    ...,
    deepspeed="path/to/deepspeed_config.json",
    parallelism_config=parallelism_config,
)
```

Run [accelerate launch](https://huggingface.co/docs/accelerate/en/package_reference/cli#accelerate-launch) with a [`Trainer`]-based script.

```shell
accelerate launch --num_processes 4 train.py \
--output_dir output_dir \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1
```

</hfoption>
<hfoption id="Accelerate config file">

Run the [accelerate config](https://huggingface.co/docs/accelerate/en/package_reference/cli#accelerate-config) command and answer questions about your hardware and training setup to create a `default_config.yaml` file in your cache.

```yaml
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: path/to/ds_config.json
machine_rank: 0
num_machines: 1
num_processes: 4
parallelism_config:
  parallelism_config_sp_size: 4
  parallelism_config_dp_replicate_size: 1
  parallelism_config_sp_backend: deepspeed
  parallelism_config_sp_seq_length_is_variable: true
  parallelism_config_sp_attn_implementation: flash_attention_2
```

Run [accelerate launch](https://huggingface.co/docs/accelerate/en/package_reference/cli#accelerate-launch) with a [`Trainer`]-based script.

```shell
accelerate launch --config_file alst_config.yaml train.py \
--output_dir output_dir \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1
```

</hfoption>
</hfoptions>

The following fields are important for configuring sequence parallelism.

> [!TIP]
> The [`Trainer`] automatically handles DataLoader sharding, `position_ids` generation, label shifting, and loss aggregation across SP ranks. If you're writing a custom training loop, see the Accelerate [Sequence Parallelism](https://huggingface.co/docs/accelerate/concept_guides/sequence_parallelism) guide instead.

- `sp_backend` must be set to `"deepspeed"` to use Ulysses sequence parallelism.

- `sp_size` is the number of GPUs that process a single sequence in parallel. Each SP rank receives a unique data stream from the DataLoader, unlike tensor parallelism where all ranks receive identical data. The effective `dp_world_size = world_size / sp_size`, so with 4 GPUs and `sp_size=4`, `dp_world_size=1` for batch size calculations. Sequences must also be padded to a multiple of `sp_size`. Set `pad_to_multiple_of` in your data collator accordingly.

    > [!WARNING]
    > The number of attention heads must be divisible by `sp_size`. A model with 32 heads supports `sp_size` of 1, 2, 4, 8, 16, or 32.

    ```py
    from transformers import DataCollatorForLanguageModeling

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=sp_size,
    )
    ```

- `sp_seq_length_is_variable` controls variable sequence length handling. Set it to `True` (recommended) for varying lengths between batches. Set it to `False` when all sequences pad to a fixed length specified by `sp_seq_length`.

- `sp_attn_implementation` sets the attention backend. Supported values are `"sdpa"`, `"flash_attention_2"`, or `"flash_attention_3"`. FlashAttention is recommended, especially when packing multiple samples in a batch. SDPA can attend incorrectly across sample boundaries when samples are packed. Eager attention isn't supported because it doesn't handle `position_ids` correctly.

### Combining with data parallelism

Sequence parallelism and data parallelism use the same GPUs, and SP doesn't require additional hardware. To run both, set `dp_replicate_size` or `dp_shard_size` so that `dp_replicate_size × dp_shard_size × sp_size` equals your total GPU count.

For example, with 8 GPUs and `sp_size=4`, set `dp_replicate_size=2` (2 × 1 × 4 = 8).

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

## Next steps

- The Accelerate [Sequence Parallelism](https://huggingface.co/docs/accelerate/concept_guides/sequence_parallelism) guide covers the Ulysses implementation in more depth and shows how to write a custom training loop.
- The [DeepSpeed ALST tutorial](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/) covers the full ALST system, including TiledMLP and activation checkpoint offloading.
- The [parallelism methods](./perf_train_gpu_many) guide shows how to combine sequence parallelism with other strategies like ZeRO.
- The [Ulysses Sequence Parallelism: Training with Million-Token Contexts](https://huggingface.co/blog/ulysses-sp) blog post explains how Ulysses works and how it's integrated in Accelerate, Trainer, and SFTTrainer.
