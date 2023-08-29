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

The [`Trainer`] class provides an API for feature-complete training in PyTorch for most standard use cases. It's used in most of the [example scripts](https://github.com/huggingface/transformers/tree/main/examples).

Before instantiating your [`Trainer`], create a [`TrainingArguments`] to access all the points of customization during training.

The API supports distributed training on multiple GPUs/TPUs, mixed precision through [NVIDIA Apex](https://github.com/NVIDIA/apex) and Native AMP for PyTorch.

The [`Trainer`] contains the basic training loop which supports the above features. To inject custom behavior you can subclass them and override the following methods:

- **get_train_dataloader** -- Creates the training DataLoader.
- **get_eval_dataloader** -- Creates the evaluation DataLoader.
- **get_test_dataloader** -- Creates the test DataLoader.
- **log** -- Logs information on the various objects watching training.
- **create_optimizer_and_scheduler** -- Sets up the optimizer and learning rate scheduler if they were not passed at
  init. Note, that you can also subclass or override the `create_optimizer` and `create_scheduler` methods
  separately.
- **create_optimizer** -- Sets up the optimizer if it wasn't passed at init.
- **create_scheduler** -- Sets up the learning rate scheduler if it wasn't passed at init.
- **compute_loss** - Computes the loss on a batch of training inputs.
- **training_step** -- Performs a training step.
- **prediction_step** -- Performs an evaluation/test step.
- **evaluate** -- Runs an evaluation loop and returns metrics.
- **predict** -- Returns predictions (with metrics if labels are available) on a test set.

<Tip warning={true}>

The [`Trainer`] class is optimized for 🤗 Transformers models and can have surprising behaviors
when you use it on other models. When using it on your own model, make sure:

- your model always return tuples or subclasses of [`~utils.ModelOutput`].
- your model can compute the loss if a `labels` argument is provided and that loss is returned as the first
  element of the tuple (if your model returns tuples)
- your model can accept multiple label arguments (use the `label_names` in your [`TrainingArguments`] to indicate their name to the [`Trainer`]) but none of them should be named `"label"`.

</Tip>

Here is an example of how to customize [`Trainer`] to use a weighted loss (useful when you have an unbalanced training set):

```python
from torch import nn
from transformers import Trainer


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```

Another way to customize the training loop behavior for the PyTorch [`Trainer`] is to use [callbacks](callback) that can inspect the training loop state (for progress reporting, logging on TensorBoard or other ML platforms...) and take decisions (like early stopping).


## Trainer

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

## Checkpoints

By default, [`Trainer`] will save all checkpoints in the `output_dir` you set in the
[`TrainingArguments`] you are using. Those will go in subfolder named `checkpoint-xxx` with xxx
being the step at which the training was at.

Resuming training from a checkpoint can be done when calling [`Trainer.train`] with either:

- `resume_from_checkpoint=True` which will resume training from the latest checkpoint
- `resume_from_checkpoint=checkpoint_dir` which will resume training from the specific checkpoint in the directory
  passed.

In addition, you can easily save your checkpoints on the Model Hub when using `push_to_hub=True`. By default, all
the models saved in intermediate checkpoints are saved in different commits, but not the optimizer state. You can adapt
the `hub-strategy` value of your [`TrainingArguments`] to either:

- `"checkpoint"`: the latest checkpoint is also pushed in a subfolder named last-checkpoint, allowing you to
  resume training easily with `trainer.train(resume_from_checkpoint="output_dir/last-checkpoint")`.
- `"all_checkpoints"`: all checkpoints are pushed like they appear in the output folder (so you will get one
  checkpoint folder per folder in your final repository)


## Logging

By default [`Trainer`] will use `logging.INFO` for the main process and `logging.WARNING` for the replicas if any.

These defaults can be overridden to use any of the 5 `logging` levels with [`TrainingArguments`]'s
arguments:

- `log_level` - for the main process
- `log_level_replica` - for the replicas

Further, if [`TrainingArguments`]'s `log_on_each_node` is set to `False` only the main node will
use the log level settings for its main process, all other nodes will use the log level settings for replicas.

Note that [`Trainer`] is going to set `transformers`'s log level separately for each node in its
[`Trainer.__init__`]. So you may want to set this sooner (see the next example) if you tap into other
`transformers` functionality before creating the [`Trainer`] object.

Here is an example of how this can be used in an application:

```python
[...]
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# set the main code and the modules it uses to the same log-level according to the node
log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

trainer = Trainer(...)
```

And then if you only want to see warnings on the main node and all other nodes to not print any most likely duplicated
warnings you could run it as:

```bash
my_app.py ... --log_level warning --log_level_replica error
```

In the multi-node environment if you also don't want the logs to repeat for each node's main process, you will want to
change the above to:

```bash
my_app.py ... --log_level warning --log_level_replica error --log_on_each_node 0
```

and then only the main process of the first node will log at the "warning" level, and all other processes on the main
node and all processes on other nodes will log at the "error" level.

If you need your application to be as quiet as possible you could do:

```bash
my_app.py ... --log_level error --log_level_replica error --log_on_each_node 0
```

(add `--log_on_each_node 0` if on multi-node environment)


## Randomness

When resuming from a checkpoint generated by [`Trainer`] all efforts are made to restore the
_python_, _numpy_ and _pytorch_ RNG states to the same states as they were at the moment of saving that checkpoint,
which should make the "stop and resume" style of training as close as possible to non-stop training.

However, due to various default non-deterministic pytorch settings this might not fully work. If you want full
determinism please refer to [Controlling sources of randomness](https://pytorch.org/docs/stable/notes/randomness). As explained in the document, that some of those settings
that make things deterministic (.e.g., `torch.backends.cudnn.deterministic`) may slow things down, therefore this
can't be done by default, but you can enable those yourself if needed.


## Specific GPUs Selection

Let's discuss how you can tell your program which GPUs are to be used and in what order.

When using [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) to use only a subset of your GPUs, you simply specify the number of GPUs to use. For example, if you have 4 GPUs, but you wish to use the first 2 you can do:

```bash
python -m torch.distributed.launch --nproc_per_node=2  trainer-program.py ...
```

if you have either [`accelerate`](https://github.com/huggingface/accelerate) or [`deepspeed`](https://github.com/microsoft/DeepSpeed) installed you can also accomplish the same by using one of:
```bash
accelerate launch --num_processes 2 trainer-program.py ...
```

```bash
deepspeed --num_gpus 2 trainer-program.py ...
```

You don't need to use the Accelerate or [the Deepspeed integration](Deepspeed) features to use these launchers.


Until now you were able to tell the program how many GPUs to use. Now let's discuss how to select specific GPUs and control their order.

The following environment variables help you control which GPUs to use and their order.

**`CUDA_VISIBLE_DEVICES`**

If you have multiple GPUs and you'd like to use only 1 or a few of those GPUs, set the environment variable `CUDA_VISIBLE_DEVICES` to a list of the GPUs to be used.

For example, let's say you have 4 GPUs: 0, 1, 2 and 3. To run only on the physical GPUs 0 and 2, you can do:

```bash
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch trainer-program.py ...
```

So now pytorch will see only 2 GPUs, where your physical GPUs 0 and 2 are mapped to `cuda:0` and `cuda:1` correspondingly.

You can even change their order:

```bash
CUDA_VISIBLE_DEVICES=2,0 python -m torch.distributed.launch trainer-program.py ...
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
python -m torch.distributed.launch trainer-program.py ...
```

but this approach can be confusing since you may forget you set up the environment variable earlier and not understand why the wrong GPUs are used. Therefore, it's a common practice to set the environment variable just for a specific run on the same command line as it's shown in most examples of this section.

**`CUDA_DEVICE_ORDER`**

There is an additional environment variable `CUDA_DEVICE_ORDER` that controls how the physical devices are ordered. The two choices are:

1. ordered by PCIe bus IDs (matches `nvidia-smi`'s order) - this is the default.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

2. ordered by GPU compute capabilities

```bash
export CUDA_DEVICE_ORDER=FASTEST_FIRST
```

Most of the time you don't need to care about this environment variable, but it's very helpful if you have a lopsided setup where you have an old and a new GPUs physically inserted in such a way so that the slow older card appears to be first. One way to fix that is to swap the cards. But if you can't swap the cards (e.g., if the cooling of the devices gets impacted) then setting `CUDA_DEVICE_ORDER=FASTEST_FIRST` will always put the newer faster card first. It'll be somewhat confusing though since `nvidia-smi` will still report them in the PCIe order.

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

**Required PyTorch version for FSDP support**: PyTorch Nightly (or 1.12.0 if you read this after it has been released)
as the model saving with FSDP activated is only available with recent fixes.

**Usage**:

- Make sure you have added the distributed launcher
`-m torch.distributed.launch --nproc_per_node=NUMBER_OF_GPUS_YOU_HAVE` if you haven't been using it already.

- **Sharding Strategy**: 
  - FULL_SHARD : Shards optimizer states + gradients + model parameters across data parallel workers/GPUs.
    For this, add `--fsdp full_shard` to the command line arguments. 
  - SHARD_GRAD_OP : Shards optimizer states + gradients across data parallel workers/GPUs.
    For this, add `--fsdp shard_grad_op` to the command line arguments.
  - NO_SHARD : No sharding. For this, add `--fsdp no_shard` to the command line arguments.
- To offload the parameters and gradients to the CPU, 
  add `--fsdp "full_shard offload"` or `--fsdp "shard_grad_op offload"` to the command line arguments.
- To automatically recursively wrap layers with FSDP using `default_auto_wrap_policy`, 
  add `--fsdp "full_shard auto_wrap"` or `--fsdp "shard_grad_op auto_wrap"` to the command line arguments.
- To enable both CPU offloading and auto wrapping, 
  add `--fsdp "full_shard offload auto_wrap"` or `--fsdp "shard_grad_op offload auto_wrap"` to the command line arguments.
- Remaining FSDP config is passed via `--fsdp_config <path_to_fsdp_config.json>`. It is either a location of
  FSDP json config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`. 
  - If auto wrapping is enabled, you can either use transformer based auto wrap policy or size based auto wrap policy.
    - For transformer based auto wrap policy, it is recommended to specify `fsdp_transformer_layer_cls_to_wrap` in the config file. If not specified, the default value is `model._no_split_modules` when available.
      This specifies the list of transformer layer class name (case-sensitive) to wrap ,e.g, [`BertLayer`], [`GPTJBlock`], [`T5Block`] ....
      This is important because submodules that share weights (e.g., embedding layer) should not end up in different FSDP wrapped units.
      Using this policy, wrapping happens for each block containing Multi-Head Attention followed by couple of MLP layers. 
      Remaining layers including the shared embeddings are conveniently wrapped in same outermost FSDP unit.
      Therefore, use this for transformer based models.
    - For size based auto wrap policy, please add `fsdp_min_num_params` in the config file. 
      It specifies FSDP's minimum number of parameters for auto wrapping.
  - `fsdp_backward_prefetch` can be specified in the config file. It controls when to prefetch next set of parameters. 
    `backward_pre` and `backward_pos` are available options. 
    For more information refer `torch.distributed.fsdp.fully_sharded_data_parallel.BackwardPrefetch`
  - `fsdp_forward_prefetch` can be specified in the config file. It controls when to prefetch next set of parameters. 
    If `"True"`, FSDP explicitly prefetches the next upcoming all-gather while executing in the forward pass. 
  - `limit_all_gathers` can be specified in the config file. 
    If `"True"`, FSDP explicitly synchronizes the CPU thread to prevent too many in-flight all-gathers.
  - `activation_checkpointing` can be specified in the config file.
    If `"True"`, FSDP activation checkpointing is a technique to reduce memory usage by clearing activations of
    certain layers and recomputing them during a backward pass. Effectively, this trades extra computation time
    for reduced memory usage.

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
  `fsdp_min_num_params` or `fsdp_transformer_layer_cls_to_wrap`. 
- You can either use transformer based auto wrap policy or size based auto wrap policy.
  - For transformer based auto wrap policy, it is recommended to specify `fsdp_transformer_layer_cls_to_wrap` in the config file. If not specified, the default value is `model._no_split_modules` when available.
    This specifies the list of transformer layer class name (case-sensitive) to wrap ,e.g, [`BertLayer`], [`GPTJBlock`], [`T5Block`] ....
    This is important because submodules that share weights (e.g., embedding layer) should not end up in different FSDP wrapped units.
    Using this policy, wrapping happens for each block containing Multi-Head Attention followed by couple of MLP layers. 
    Remaining layers including the shared embeddings are conveniently wrapped in same outermost FSDP unit.
    Therefore, use this for transformer based models.
  - For size based auto wrap policy, please add `fsdp_min_num_params` in the config file. 
    It specifies FSDP's minimum number of parameters for auto wrapping.


### Using Trainer for accelerated PyTorch Training on Mac 

With PyTorch v1.12 release, developers and researchers can take advantage of Apple silicon GPUs for significantly faster model training. 
This unlocks the ability to perform machine learning workflows like prototyping and fine-tuning locally, right on Mac.
Apple's Metal Performance Shaders (MPS) as a backend for PyTorch enables this and can be used via the new `"mps"` device. 
This will map computational graphs and primitives on the MPS Graph framework and tuned kernels provided by MPS.
For more information please refer official documents [Introducing Accelerated PyTorch Training on Mac](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)
and [MPS BACKEND](https://pytorch.org/docs/stable/notes/mps.html). 

<Tip warning={false}>

We strongly recommend to install PyTorch >= 1.13 (nightly version at the time of writing) on your MacOS machine. 
It has major fixes related to model correctness and performance improvements for transformer based models.
Please refer to https://github.com/pytorch/pytorch/issues/82707 for more details.

</Tip>

**Benefits of Training and Inference using Apple Silicon Chips**

1. Enables users to train larger networks or batch sizes locally
2. Reduces data retrieval latency and provides the GPU with direct access to the full memory store due to unified memory architecture. 
Therefore, improving end-to-end performance.
3. Reduces costs associated with cloud-based development or the need for additional local GPUs.

**Pre-requisites**: To install torch with mps support, 
please follow this nice medium article [GPU-Acceleration Comes to PyTorch on M1 Macs](https://medium.com/towards-data-science/gpu-acceleration-comes-to-pytorch-on-m1-macs-195c399efcc1).

**Usage**:
`mps` device will be used by default if available similar to the way `cuda` device is used.
Therefore, no action from user is required. 
For example, you can run the official Glue text classififcation task (from the root folder) using Apple Silicon GPU with below command:

```bash
export TASK_NAME=mrpc

python examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir
```

**A few caveats to be aware of**

1. Some PyTorch operations have not been implemented in mps and will throw an error. 
One way to get around that is to set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1`, 
which will fallback to CPU for these operations. It still throws a UserWarning however.
2. Distributed setups `gloo` and `nccl` are not working with `mps` device. 
This means that currently only single GPU of `mps` device type can be used.

Finally, please, remember that, 🤗 `Trainer` only integrates MPS backend, therefore if you
have any problems or questions with regards to MPS backend usage, please, 
file an issue with [PyTorch GitHub](https://github.com/pytorch/pytorch/issues).


## Using Accelerate Launcher with Trainer

Accelerate now powers Trainer. In terms of what users should expect:
- They can keep using the Trainer ingterations such as FSDP, DeepSpeed vis trainer arguments without any changes on their part.
- They can now use Accelerate Launcher with Trainer (recommended).

Steps to use Accelerate Launcher with Trainer:
1. Make sure 🤗 Accelerate is installed, you can't use the `Trainer` without it anyway. If not `pip install accelerate`. You may also need to update your version of Accelerate: `pip install accelerate --upgrade`
2. Run `accelerate config` and fill the questionnaire. Below are example accelerate configs:
  a. DDP Multi-node Multi-GPU config:
    ```yaml
    compute_environment: LOCAL_MACHINE                                                                                             
    distributed_type: MULTI_GPU                                                                                                    
    downcast_bf16: 'no'
    gpu_ids: all
    machine_rank: 0 #change rank as per the node
    main_process_ip: 192.168.20.1
    main_process_port: 9898
    main_training_function: main
    mixed_precision: fp16
    num_machines: 2
    num_processes: 8
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false
    ```

  b. FSDP config:
    ```yaml
    compute_environment: LOCAL_MACHINE
    distributed_type: FSDP
    downcast_bf16: 'no'
    fsdp_config:
      fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
      fsdp_backward_prefetch_policy: BACKWARD_PRE
      fsdp_forward_prefetch: true
      fsdp_offload_params: false
      fsdp_sharding_strategy: 1
      fsdp_state_dict_type: FULL_STATE_DICT
      fsdp_sync_module_states: true
      fsdp_transformer_layer_cls_to_wrap: BertLayer
      fsdp_use_orig_params: true
    machine_rank: 0
    main_training_function: main
    mixed_precision: bf16
    num_machines: 1
    num_processes: 2
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false
    ```
  c. DeepSpeed config pointing to a file:
    ```yaml
    compute_environment: LOCAL_MACHINE
    deepspeed_config:
      deepspeed_config_file: /home/user/configs/ds_zero3_config.json
      zero3_init_flag: true
    distributed_type: DEEPSPEED
    downcast_bf16: 'no'
    machine_rank: 0
    main_training_function: main
    num_machines: 1
    num_processes: 4
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false
    ```

  d. DeepSpeed config using accelerate plugin:
    ```yaml
    compute_environment: LOCAL_MACHINE                                                                                             
    deepspeed_config:                                                                                                              
      gradient_accumulation_steps: 1
      gradient_clipping: 0.7
      offload_optimizer_device: cpu
      offload_param_device: cpu
      zero3_init_flag: true
      zero_stage: 2
    distributed_type: DEEPSPEED
    downcast_bf16: 'no'
    machine_rank: 0
    main_training_function: main
    mixed_precision: bf16
    num_machines: 1
    num_processes: 4
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false
    ```

3. Run the Trainer script with args other than the ones handled above by accelerate config or launcher args.
Below is an example to run `run_glue.py` using `accelerate launcher` with FSDP config from above. 

```bash
cd transformers

accelerate launch \
./examples/pytorch/text-classification/run_glue.py \
--model_name_or_path bert-base-cased \
--task_name $TASK_NAME \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 16 \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--output_dir /tmp/$TASK_NAME/ \
--overwrite_output_dir
```

4. You can also directly use the cmd args for `accelerate launch`. Above example would map to:

```bash
cd transformers

accelerate launch --num_processes=2 \
--use_fsdp \
--mixed_precision=bf16 \
--fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP  \
--fsdp_transformer_layer_cls_to_wrap="BertLayer" \
--fsdp_sharding_strategy=1 \
--fsdp_state_dict_type=FULL_STATE_DICT \
./examples/pytorch/text-classification/run_glue.py
--model_name_or_path bert-base-cased \
--task_name $TASK_NAME \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 16 \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--output_dir /tmp/$TASK_NAME/ \
--overwrite_output_dir
```

For more information, please refer the 🤗 Accelerate CLI guide: [Launching your 🤗 Accelerate scripts](https://huggingface.co/docs/accelerate/basic_tutorials/launch).

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
