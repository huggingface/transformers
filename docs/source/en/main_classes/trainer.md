<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Trainer

The [`Trainer`] class provides an API for feature-complete training in PyTorch, and it supports distributed training on multiple GPUs/TPUs, mixed precision for [NVIDIA GPUs](https://nvidia.github.io/apex/), [AMD GPUs](https://rocm.docs.amd.com/en/latest/rocm.html), and [`torch.amp`](https://pytorch.org/docs/stable/amp.html) for PyTorch. [`Trainer`] goes hand-in-hand with the [`TrainingArguments`] class, which offers a wide range of options to customize how a model is trained. Together, these two classes provide a complete training API.

[`Seq2SeqTrainer`] and [`Seq2SeqTrainingArguments`] inherit from the [`Trainer`] and [`TrainingArgument`] classes and they're adapted for training models for sequence-to-sequence tasks such as summarization or translation.

<Tip warning={true}>

The [`Trainer`] class is optimized for 🤗 Transformers models and can have surprising behaviors
when used with other models. When using it with your own model, make sure:

- your model always return tuples or subclasses of [`~utils.ModelOutput`]
- your model can compute the loss if a `labels` argument is provided and that loss is returned as the first
  element of the tuple (if your model returns tuples)
- your model can accept multiple label arguments (use `label_names` in [`TrainingArguments`] to indicate their name to the [`Trainer`]) but none of them should be named `"label"`

</Tip>

## Trainer[[api-reference]]

[[autodoc]] Trainer
    - all

## Seq2SeqTrainer

[[autodoc]] Seq2SeqTrainer
    - evaluate
    - predict

## TrainingArguments

[[autodoc]] TrainingArguments
    - all

## Seq2SeqTrainingArguments

[[autodoc]] Seq2SeqTrainingArguments
    - all

## Specific GPUs Selection

Let's discuss how you can tell your program which GPUs are to be used and in what order.

When using [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) to use only a subset of your GPUs, you simply specify the number of GPUs to use. For example, if you have 4 GPUs, but you wish to use the first 2 you can do:

```bash
torchrun --nproc_per_node=2  trainer-program.py ...
```

if you have either [`accelerate`](https://github.com/huggingface/accelerate) or [`deepspeed`](https://github.com/microsoft/DeepSpeed) installed you can also accomplish the same by using one of:

```bash
accelerate launch --num_processes 2 trainer-program.py ...
```

```bash
deepspeed --num_gpus 2 trainer-program.py ...
```

You don't need to use the Accelerate or [the Deepspeed integration](deepspeed) features to use these launchers.


Until now you were able to tell the program how many GPUs to use. Now let's discuss how to select specific GPUs and control their order.

The following environment variables help you control which GPUs to use and their order.

**`CUDA_VISIBLE_DEVICES`**

If you have multiple GPUs and you'd like to use only 1 or a few of those GPUs, set the environment variable `CUDA_VISIBLE_DEVICES` to a list of the GPUs to be used.

For example, let's say you have 4 GPUs: 0, 1, 2 and 3. To run only on the physical GPUs 0 and 2, you can do:

```bash
CUDA_VISIBLE_DEVICES=0,2 torchrun trainer-program.py ...
```

So now pytorch will see only 2 GPUs, where your physical GPUs 0 and 2 are mapped to `cuda:0` and `cuda:1` correspondingly.

You can even change their order:

```bash
CUDA_VISIBLE_DEVICES=2,0 torchrun trainer-program.py ...
```

Here your physical GPUs 0 and 2 are mapped to `cuda:1` and `cuda:0` correspondingly.

The above examples were all for `DistributedDataParallel` use pattern, but the same method works for [`DataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) as well:

```bash
CUDA_VISIBLE_DEVICES=2,0 python trainer-program.py ...
```

To emulate an environment without GPUs simply set this environment variable to an empty value like so:

```bash
CUDA_VISIBLE_DEVICES= python trainer-program.py ...
```

As with any environment variable you can, of course, export those instead of adding these to the command line, as in:


```bash
export CUDA_VISIBLE_DEVICES=0,2
torchrun trainer-program.py ...
```

but this approach can be confusing since you may forget you set up the environment variable earlier and not understand why the wrong GPUs are used. Therefore, it's a common practice to set the environment variable just for a specific run on the same command line as it's shown in most examples of this section.

**`CUDA_DEVICE_ORDER`**

There is an additional environment variable `CUDA_DEVICE_ORDER` that controls how the physical devices are ordered. The two choices are:

1. ordered by PCIe bus IDs (matches `nvidia-smi` and `rocm-smi`'s order) - this is the default.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

2. ordered by GPU compute capabilities

```bash
export CUDA_DEVICE_ORDER=FASTEST_FIRST
```

Most of the time you don't need to care about this environment variable, but it's very helpful if you have a lopsided setup where you have an old and a new GPUs physically inserted in such a way so that the slow older card appears to be first. One way to fix that is to swap the cards. But if you can't swap the cards (e.g., if the cooling of the devices gets impacted) then setting `CUDA_DEVICE_ORDER=FASTEST_FIRST` will always put the newer faster card first. It'll be somewhat confusing though since `nvidia-smi` (or `rocm-smi`) will still report them in the PCIe order.

The other solution to swapping the order is to use:

```bash
export CUDA_VISIBLE_DEVICES=1,0
```
In this example we are working with just 2 GPUs, but of course the same would apply to as many GPUs as your computer has.

Also if you do set this environment variable it's the best to set it in your `~/.bashrc` file or some other startup config file and forget about it.

## Trainer Integrations

The [`Trainer`] has been extended to support libraries that may dramatically improve your training
time and fit much bigger models.

Currently it supports third party solutions, [DeepSpeed](https://github.com/microsoft/DeepSpeed) and [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html), which implement parts of the paper [ZeRO: Memory Optimizations
Toward Training Trillion Parameter Models, by Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He](https://arxiv.org/abs/1910.02054).

This provided support is new and experimental as of this writing. While the support for DeepSpeed and PyTorch FSDP is active and we welcome issues around it, we don't support the FairScale integration anymore since it has been integrated in PyTorch main (see the [PyTorch FSDP integration](#pytorch-fully-sharded-data-parallel))

<a id='zero-install-notes'></a>

### CUDA Extension Installation Notes

As of this writing, Deepspeed require compilation of CUDA C++ code, before it can be used.

While all installation issues should be dealt with through the corresponding GitHub Issues of [Deepspeed](https://github.com/microsoft/DeepSpeed/issues), there are a few common issues that one may encounter while building
any PyTorch extension that needs to build CUDA extensions.

Therefore, if you encounter a CUDA-related build issue while doing the following:

```bash
pip install deepspeed
```

please, read the following notes first.

In these notes we give examples for what to do when `pytorch` has been built with CUDA `10.2`. If your situation is
different remember to adjust the version number to the one you are after.

#### Possible problem #1

While, Pytorch comes with its own CUDA toolkit, to build these two projects you must have an identical version of CUDA
installed system-wide.

For example, if you installed `pytorch` with `cudatoolkit==10.2` in the Python environment, you also need to have
CUDA `10.2` installed system-wide.

The exact location may vary from system to system, but `/usr/local/cuda-10.2` is the most common location on many
Unix systems. When CUDA is correctly set up and added to the `PATH` environment variable, one can find the
installation location by doing:

```bash
which nvcc
```

If you don't have CUDA installed system-wide, install it first. You will find the instructions by using your favorite
search engine. For example, if you're on Ubuntu you may want to search for: [ubuntu cuda 10.2 install](https://www.google.com/search?q=ubuntu+cuda+10.2+install).

#### Possible problem #2

Another possible common problem is that you may have more than one CUDA toolkit installed system-wide. For example you
may have:

```bash
/usr/local/cuda-10.2
/usr/local/cuda-11.0
```

Now, in this situation you need to make sure that your `PATH` and `LD_LIBRARY_PATH` environment variables contain
the correct paths to the desired CUDA version. Typically, package installers will set these to contain whatever the
last version was installed. If you encounter the problem, where the package build fails because it can't find the right
CUDA version despite you having it installed system-wide, it means that you need to adjust the 2 aforementioned
environment variables.

First, you may look at their contents:

```bash
echo $PATH
echo $LD_LIBRARY_PATH
```

so you get an idea of what is inside.

It's possible that `LD_LIBRARY_PATH` is empty.

`PATH` lists the locations of where executables can be found and `LD_LIBRARY_PATH` is for where shared libraries
are to looked for. In both cases, earlier entries have priority over the later ones. `:` is used to separate multiple
entries.

Now, to tell the build program where to find the specific CUDA toolkit, insert the desired paths to be listed first by
doing:

```bash
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
```

Note that we aren't overwriting the existing values, but prepending instead.

Of course, adjust the version number, the full path if need be. Check that the directories you assign actually do
exist. `lib64` sub-directory is where the various CUDA `.so` objects, like `libcudart.so` reside, it's unlikely
that your system will have it named differently, but if it is adjust it to reflect your reality.


#### Possible problem #3

Some older CUDA versions may refuse to build with newer compilers. For example, you my have `gcc-9` but it wants
`gcc-7`.

There are various ways to go about it.

If you can install the latest CUDA toolkit it typically should support the newer compiler.

Alternatively, you could install the lower version of the compiler in addition to the one you already have, or you may
already have it but it's not the default one, so the build system can't see it. If you have `gcc-7` installed but the
build system complains it can't find it, the following might do the trick:

```bash
sudo ln -s /usr/bin/gcc-7  /usr/local/cuda-10.2/bin/gcc
sudo ln -s /usr/bin/g++-7  /usr/local/cuda-10.2/bin/g++
```

Here, we are making a symlink to `gcc-7` from `/usr/local/cuda-10.2/bin/gcc` and since
`/usr/local/cuda-10.2/bin/` should be in the `PATH` environment variable (see the previous problem's solution), it
should find `gcc-7` (and `g++7`) and then the build will succeed.

As always make sure to edit the paths in the example to match your situation.


### PyTorch Fully Sharded Data parallel

To accelerate training huge models on larger batch sizes, we can use a fully sharded data parallel model.
This type of data parallel paradigm enables fitting more data and larger models by sharding the optimizer states, gradients and parameters.
To read more about it and the benefits, check out the [Fully Sharded Data Parallel blog](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/).
We have integrated the latest PyTorch's Fully Sharded Data Parallel (FSDP) training feature.
All you need to do is enable it through the config.

**Required PyTorch version for FSDP support**: PyTorch >=2.1.0

**Usage**:

- Make sure you have added the distributed launcher
`-m torch.distributed.launch --nproc_per_node=NUMBER_OF_GPUS_YOU_HAVE` if you haven't been using it already.

- **Sharding Strategy**: 
  - FULL_SHARD : Shards optimizer states + gradients + model parameters across data parallel workers/GPUs.
    For this, add `--fsdp full_shard` to the command line arguments. 
  - SHARD_GRAD_OP : Shards optimizer states + gradients across data parallel workers/GPUs.
    For this, add `--fsdp shard_grad_op` to the command line arguments.
  - NO_SHARD : No sharding. For this, add `--fsdp no_shard` to the command line arguments.
  - HYBRID_SHARD : No sharding. For this, add `--fsdp hybrid_shard` to the command line arguments.
  - HYBRID_SHARD_ZERO2 : No sharding. For this, add `--fsdp hybrid_shard_zero2` to the command line arguments.
- To offload the parameters and gradients to the CPU, 
  add `--fsdp "full_shard offload"` or `--fsdp "shard_grad_op offload"` to the command line arguments.
- To automatically recursively wrap layers with FSDP using `default_auto_wrap_policy`, 
  add `--fsdp "full_shard auto_wrap"` or `--fsdp "shard_grad_op auto_wrap"` to the command line arguments.
- To enable both CPU offloading and auto wrapping, 
  add `--fsdp "full_shard offload auto_wrap"` or `--fsdp "shard_grad_op offload auto_wrap"` to the command line arguments.
- Remaining FSDP config is passed via `--fsdp_config <path_to_fsdp_config.json>`. It is either a location of
  FSDP json config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`. 
  - If auto wrapping is enabled, you can either use transformer based auto wrap policy or size based auto wrap policy.
    - For transformer based auto wrap policy, it is recommended to specify `transformer_layer_cls_to_wrap` in the config file. If not specified, the default value is `model._no_split_modules` when available.
      This specifies the list of transformer layer class name (case-sensitive) to wrap ,e.g, [`BertLayer`], [`GPTJBlock`], [`T5Block`] ....
      This is important because submodules that share weights (e.g., embedding layer) should not end up in different FSDP wrapped units.
      Using this policy, wrapping happens for each block containing Multi-Head Attention followed by couple of MLP layers. 
      Remaining layers including the shared embeddings are conveniently wrapped in same outermost FSDP unit.
      Therefore, use this for transformer based models.
    - For size based auto wrap policy, please add `min_num_params` in the config file. 
      It specifies FSDP's minimum number of parameters for auto wrapping.
  - `backward_prefetch` can be specified in the config file. It controls when to prefetch next set of parameters. 
    `backward_pre` and `backward_pos` are available options. 
    For more information refer `torch.distributed.fsdp.fully_sharded_data_parallel.BackwardPrefetch`
  - `forward_prefetch` can be specified in the config file. It controls when to prefetch next set of parameters. 
    If `"True"`, FSDP explicitly prefetches the next upcoming all-gather while executing in the forward pass. 
  - `limit_all_gathers` can be specified in the config file. 
    If `"True"`, FSDP explicitly synchronizes the CPU thread to prevent too many in-flight all-gathers.
  - `activation_checkpointing` can be specified in the config file.
    If `"True"`, FSDP activation checkpointing is a technique to reduce memory usage by clearing activations of
    certain layers and recomputing them during a backward pass. Effectively, this trades extra computation time
    for reduced memory usage.
  - `use_orig_params` can be specified in the config file. 
    If True, allows non-uniform `requires_grad` during init, which means support for interspersed frozen and trainable paramteres. Useful in cases such as parameter-efficient fine-tuning. This also enables to have different optimizer param groups. This should be `True` when creating optimizer object before preparing/wrapping the model with FSDP.
    Please refer this [blog](https://dev-discuss.pytorch.org/t/rethinking-pytorch-fully-sharded-data-parallel-fsdp-from-first-principles/1019). 

**Saving and loading**
Saving entire intermediate checkpoints using `FULL_STATE_DICT` state_dict_type with CPU offloading on rank 0 takes a lot of time and often results in NCCL Timeout errors due to indefinite hanging during broadcasting. However, at the end of training, we want the whole model state dict instead of the sharded state dict which is only compatible with FSDP. Use `SHARDED_STATE_DICT` (default) state_dict_type to save the intermediate checkpoints and optimizer states in this format recommended by the PyTorch team. 

Saving the final checkpoint in transformers format using default `safetensors` format requires below changes.
```python
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

trainer.save_model(script_args.output_dir)
```

**Few caveats to be aware of**
- it is incompatible with `generate`, thus is incompatible with `--predict_with_generate` 
  in all seq2seq/clm scripts (translation/summarization/clm etc.).  
  Please refer issue [#21667](https://github.com/huggingface/transformers/issues/21667)

### PyTorch/XLA Fully Sharded Data parallel

For all the TPU users, great news! PyTorch/XLA now supports FSDP.
All the latest Fully Sharded Data Parallel (FSDP) training are supported.
For more information refer to the [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/) and [PyTorch/XLA implementation of FSDP](https://github.com/pytorch/xla/tree/master/torch_xla/distributed/fsdp)
All you need to do is enable it through the config.

**Required PyTorch/XLA version for FSDP support**: >=2.0

**Usage**:

Pass `--fsdp "full shard"` along with following changes to be made in `--fsdp_config <path_to_fsdp_config.json>`:
- `xla` should be set to `True` to enable PyTorch/XLA FSDP.
- `xla_fsdp_settings` The value is a dictionary which stores the XLA FSDP wrapping parameters.
  For a complete list of options, please see [here](
  https://github.com/pytorch/xla/blob/master/torch_xla/distributed/fsdp/xla_fully_sharded_data_parallel.py).
- `xla_fsdp_grad_ckpt`. When `True`, uses gradient checkpointing over each nested XLA FSDP wrapped layer. 
  This setting can only be used when the xla flag is set to true, and an auto wrapping policy is specified through
  `min_num_params` or `transformer_layer_cls_to_wrap`. 
- You can either use transformer based auto wrap policy or size based auto wrap policy.
  - For transformer based auto wrap policy, it is recommended to specify `transformer_layer_cls_to_wrap` in the config file. If not specified, the default value is `model._no_split_modules` when available.
    This specifies the list of transformer layer class name (case-sensitive) to wrap ,e.g, [`BertLayer`], [`GPTJBlock`], [`T5Block`] ....
    This is important because submodules that share weights (e.g., embedding layer) should not end up in different FSDP wrapped units.
    Using this policy, wrapping happens for each block containing Multi-Head Attention followed by couple of MLP layers. 
    Remaining layers including the shared embeddings are conveniently wrapped in same outermost FSDP unit.
    Therefore, use this for transformer based models.
  - For size based auto wrap policy, please add `min_num_params` in the config file. 
    It specifies FSDP's minimum number of parameters for auto wrapping.

Sections that were moved:

[ <a href="./deepspeed#deepspeed-trainer-integration">DeepSpeed</a><a id="deepspeed"></a>
| <a href="./deepspeed#deepspeed-installation">Installation</a><a id="installation"></a>
| <a href="./deepspeed#deepspeed-multi-gpu">Deployment with multiple GPUs</a><a id="deployment-with-multiple-gpus"></a>
| <a href="./deepspeed#deepspeed-one-gpu">Deployment with one GPU</a><a id="deployment-with-one-gpu"></a>
| <a href="./deepspeed#deepspeed-notebook">Deployment in Notebooks</a><a id="deployment-in-notebooks"></a>
| <a href="./deepspeed#deepspeed-config">Configuration</a><a id="configuration"></a>
| <a href="./deepspeed#deepspeed-config-passing">Passing Configuration</a><a id="passing-configuration"></a>
| <a href="./deepspeed#deepspeed-config-shared">Shared Configuration</a><a id="shared-configuration"></a>
| <a href="./deepspeed#deepspeed-zero">ZeRO</a><a id="zero"></a>
| <a href="./deepspeed#deepspeed-zero2-config">ZeRO-2 Config</a><a id="zero-2-config"></a>
| <a href="./deepspeed#deepspeed-zero3-config">ZeRO-3 Config</a><a id="zero-3-config"></a>
| <a href="./deepspeed#deepspeed-nvme">NVMe Support</a><a id="nvme-support"></a>
| <a href="./deepspeed#deepspeed-zero2-zero3-performance">ZeRO-2 vs ZeRO-3 Performance</a><a id="zero-2-vs-zero-3-performance"></a>
| <a href="./deepspeed#deepspeed-zero2-example">ZeRO-2 Example</a><a id="zero-2-example"></a>
| <a href="./deepspeed#deepspeed-zero3-example">ZeRO-3 Example</a><a id="zero-3-example"></a>
| <a href="./deepspeed#deepspeed-optimizer">Optimizer</a><a id="optimizer"></a>
| <a href="./deepspeed#deepspeed-scheduler">Scheduler</a><a id="scheduler"></a>
| <a href="./deepspeed#deepspeed-fp32">fp32 Precision</a><a id="fp32-precision"></a>
| <a href="./deepspeed#deepspeed-amp">Automatic Mixed Precision</a><a id="automatic-mixed-precision"></a>
| <a href="./deepspeed#deepspeed-bs">Batch Size</a><a id="batch-size"></a>
| <a href="./deepspeed#deepspeed-grad-acc">Gradient Accumulation</a><a id="gradient-accumulation"></a>
| <a href="./deepspeed#deepspeed-grad-clip">Gradient Clipping</a><a id="gradient-clipping"></a>
| <a href="./deepspeed#deepspeed-weight-extraction">Getting The Model Weights Out</a><a id="getting-the-model-weights-out"></a>
]
