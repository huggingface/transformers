<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Debugging

Training on multiple GPUs can be a tricky endeavor whether you're running into installation issues or communication problems between your GPUs. This debugging guide covers some issues you may run into and how to resolve them.

## DeepSpeed CUDA installation

If you're using DeepSpeed, you've probably already installed it with the following command.

```bash
pip install deepspeed
```

DeepSpeed compiles CUDA C++ code and it can be a potential source of errors when building PyTorch extensions that require CUDA. These errors depend on how CUDA is installed on your system, and this section focuses on PyTorch built with *CUDA 10.2*.

<Tip>

For any other installation issues, please [open an issue](https://github.com/deepspeedai/DeepSpeed/issues) with the DeepSpeed team.

</Tip>

### Non-identical CUDA toolkits

PyTorch comes with its own CUDA toolkit, but to use DeepSpeed with PyTorch, you need to have an identical version of CUDA installed system-wide. For example, if you installed PyTorch with `cudatoolkit==10.2` in your Python environment, then you'll also need to have CUDA 10.2 installed system-wide. If you don't have CUDA installed system-wide, you should install it first.

The exact location may vary from system to system, but `usr/local/cuda-10.2` is the most common location on many Unix systems. When CUDA is correctly setup and added to your `PATH` environment variable, you can find the installation location with the following command:

```bash
which nvcc
```

### Multiple CUDA toolkits

You may also have more than one CUDA toolkit installed system-wide.

```bash
/usr/local/cuda-10.2
/usr/local/cuda-11.0
```

Typically, package installers set the paths to whatever the last version was installed. If the package build fails because it can't find the right CUDA version (despite it being installed system-wide already), then you need to configure the `PATH` and `LD_LIBRARY_PATH` environment variables to point to the correct path.

Take a look at the contents of these environment variables first:

```bash
echo $PATH
echo $LD_LIBRARY_PATH
```

`PATH` lists the locations of the executables and `LD_LIBRARY_PATH` lists where to look for shared libraries. Earlier entries are prioritized over later ones, and `:` is used to separate multiple entries. To tell the build program where to find the specific CUDA toolkit you want, insert the correct path to list first. This command prepends rather than overwrites the existing values.

```bash
# adjust the version and full path if needed
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
```

In addition, you should also check the directories you assign actually exist. The `lib64` sub-directory contains various CUDA `.so` objects (like `libcudart.so`) and while it is unlikely your system names them differently, you should check the actual names and change them accordingly.

### Older CUDA versions

Sometimes, older CUDA versions may refuse to build with newer compilers. For example, if you have `gcc-9` but CUDA wants `gcc-7`. Usually, installing the latest CUDA toolkit enables support for the newer compiler.

You could also install an older version of the compiler in addition to the one you're currently using (or it may already be installed but it's not used by default and the build system can't see it). To resolve this, you can create a symlink to give the build system visibility to the older compiler.

```bash
# adapt the path to your system
sudo ln -s /usr/bin/gcc-7  /usr/local/cuda-10.2/bin/gcc
sudo ln -s /usr/bin/g++-7  /usr/local/cuda-10.2/bin/g++
```

### Prebuild

If you're still having issues with installing DeepSpeed or if you're building DeepSpeed at run time, you can try to prebuild the DeepSpeed modules before installing them. To make a local build for DeepSpeed:

```bash
git clone https://github.com/deepspeedai/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log
```

<Tip>

To use NVMe offload, add the `DS_BUILD_AIO=1` parameter to the build command and make sure you install the libaio-dev package system-wide.

</Tip>

Next, you'll have to specify your GPU's architecture by editing the `TORCH_CUDA_ARCH_LIST` variable (find a complete list of NVIDIA GPUs and their corresponding architectures on this [page](https://developer.nvidia.com/cuda-gpus)). To check the PyTorch version that corresponds to your architecture, run the following command:

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

Find the architecture for a GPU with the following command:

<hfoptions id="arch">
<hfoption id="same GPUs">

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())"
```

</hfoption>
<hfoption id="specific GPU">

To find the architecture for GPU `0`:

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import torch; \
print(torch.cuda.get_device_properties(torch.device('cuda')))
"_CudaDeviceProperties(name='GeForce RTX 3090', major=8, minor=6, total_memory=24268MB, multi_processor_count=82)"
```

This means your GPU architecture is `8.6`.

</hfoption>
</hfoptions>

If you get `8, 6`, then you can set `TORCH_CUDA_ARCH_LIST="8.6"`. For multiple GPUs with different architectures, list them like `TORCH_CUDA_ARCH_LIST="6.1;8.6"`.

It is also possible to not specify `TORCH_CUDA_ARCH_LIST` and the build program automatically queries the GPU architecture of the build. However, it may or may not match the actual GPU on the target machine which is why it is better to explicitly specify the correct architecture.

For training on multiple machines with the same setup, you'll need to make a binary wheel:

```bash
git clone https://github.com/deepspeedai/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 \
python setup.py build_ext -j8 bdist_wheel
```

This command generates a binary wheel that'll look something like `dist/deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl`. Now you can install this wheel locally or on another machine.

```bash
pip install deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl
```

## Multi-GPU Network Issues Debug

When training or inferencing with `DistributedDataParallel` and multiple GPU, if you run into issue of inter-communication between processes and/or nodes, you can use the following script to diagnose network issues.

```bash
wget https://raw.githubusercontent.com/huggingface/transformers/main/scripts/distributed/torch-distributed-gpu-test.py
```

For example to test how 2 GPUs interact do:

```bash
python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```
If both processes can talk to each and allocate GPU memory each will print an OK status.

For more GPUs or nodes adjust the arguments in the script.

You will find a lot more details inside the diagnostics script and even a recipe to how you could run it in a SLURM environment.

An additional level of debug is to add `NCCL_DEBUG=INFO` environment variable as follows:

```bash
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

This will dump a lot of NCCL-related debug information, which you can then search online if you find that some problems are reported. Or if you're not sure how to interpret the output you can share the log file in an Issue.



## Underflow and Overflow Detection

<Tip>

This feature is currently available for PyTorch-only.

</Tip>

<Tip>

For multi-GPU training it requires DDP (`torch.distributed.launch`).

</Tip>

<Tip>

This feature can be used with any `nn.Module`-based model.

</Tip>

If you start getting `loss=NaN` or the model exhibits some other abnormal behavior due to `inf` or `nan` in
activations or weights one needs to discover where the first underflow or overflow happens and what led to it. Luckily
you can accomplish that easily by activating a special module that will do the detection automatically.

If you're using [`Trainer`], you just need to add:

```bash
--debug underflow_overflow
```

to the normal command line arguments, or pass `debug="underflow_overflow"` when creating the
[`TrainingArguments`] object.

If you're using your own training loop or another Trainer you can accomplish the same with:

```python
from transformers.debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model)
```

[`~debug_utils.DebugUnderflowOverflow`] inserts hooks into the model that immediately after each
forward call will test input and output variables and also the corresponding module's weights. As soon as `inf` or
`nan` is detected in at least one element of the activations or weights, the program will assert and print a report
like this (this was caught with `google/mt5-small` under fp16 mixed precision):

```
Detected inf/nan during batch_number=0
Last 21 forward frames:
abs min  abs max  metadata
                  encoder.block.1.layer.1.DenseReluDense.dropout Dropout
0.00e+00 2.57e+02 input[0]
0.00e+00 2.85e+02 output
[...]
                  encoder.block.2.layer.0 T5LayerSelfAttention
6.78e-04 3.15e+03 input[0]
2.65e-04 3.42e+03 output[0]
             None output[1]
2.25e-01 1.00e+04 output[2]
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.dropout Dropout
0.00e+00 8.76e+03 input[0]
0.00e+00 9.74e+03 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

The example output has been trimmed in the middle for brevity.

The second column shows the value of the absolute largest element, so if you have a closer look at the last few frames,
the inputs and outputs were in the range of `1e4`. So when this training was done under fp16 mixed precision the very
last step overflowed (since under `fp16` the largest number before `inf` is `64e3`). To avoid overflows under
`fp16` the activations must remain way below `1e4`, because `1e4 * 1e4 = 1e8` so any matrix multiplication with
large activations is going to lead to a numerical overflow condition.

At the very start of the trace you can discover at which batch number the problem occurred (here `Detected inf/nan during batch_number=0` means the problem occurred on the first batch).

Each reported frame starts by declaring the fully qualified entry for the corresponding module this frame is reporting
for. If we look just at this frame:

```
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
```

Here, `encoder.block.2.layer.1.layer_norm` indicates that it was a layer norm for the first layer, of the second
block of the encoder. And the specific calls of the `forward` is `T5LayerNorm`.

Let's look at the last few frames of that report:

```
Detected inf/nan during batch_number=0
Last 21 forward frames:
abs min  abs max  metadata
[...]
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

The last frame reports for `Dropout.forward` function with the first entry for the only input and the second for the
only output. You can see that it was called from an attribute `dropout` inside `DenseReluDense` class. We can see
that it happened during the first layer, of the 2nd block, during the very first batch. Finally, the absolute largest
input elements was `6.27e+04` and same for the output was `inf`.

You can see here, that `T5DenseGatedGeluDense.forward` resulted in output activations, whose absolute max value was
around 62.7K, which is very close to fp16's top limit of 64K. In the next frame we have `Dropout` which renormalizes
the weights, after it zeroed some of the elements, which pushes the absolute max value to more than 64K, and we get an
overflow (`inf`).

As you can see it's the previous frames that we need to look into when the numbers start going into very large for fp16
numbers.

Let's match the report to the code from `models/t5/modeling_t5.py`:

```python
class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
```

Now it's easy to see the `dropout` call, and all the previous calls as well.

Since the detection is happening in a forward hook, these reports are printed immediately after each `forward`
returns.

Going back to the full report, to act on it and to fix the problem, we need to go a few frames up where the numbers
started to go up and most likely switch to the `fp32` mode here, so that the numbers don't overflow when multiplied
or summed up. Of course, there might be other solutions. For example, we could turn off `amp` temporarily if it's
enabled, after moving the original `forward` into a helper wrapper, like so:

```python
def _forward(self, hidden_states):
    hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
    hidden_linear = self.wi_1(hidden_states)
    hidden_states = hidden_gelu * hidden_linear
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.wo(hidden_states)
    return hidden_states


import torch


def forward(self, hidden_states):
    if torch.is_autocast_enabled():
        with torch.cuda.amp.autocast(enabled=False):
            return self._forward(hidden_states)
    else:
        return self._forward(hidden_states)
```

Since the automatic detector only reports on inputs and outputs of full frames, once you know where to look, you may
want to analyse the intermediary stages of any specific `forward` function as well. In such a case you can use the
`detect_overflow` helper function to inject the detector where you want it, for example:

```python
from debug_utils import detect_overflow


class T5LayerFF(nn.Module):
    [...]

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        detect_overflow(forwarded_states, "after layer_norm")
        forwarded_states = self.DenseReluDense(forwarded_states)
        detect_overflow(forwarded_states, "after DenseReluDense")
        return hidden_states + self.dropout(forwarded_states)
```

You can see that we added 2 of these and now we track if `inf` or `nan` for `forwarded_states` was detected
somewhere in between.

Actually, the detector already reports these because each of the calls in the example above is a `nn.Module`, but
let's say if you had some local direct calculations this is how you'd do that.

Additionally, if you're instantiating the debugger in your own code, you can adjust the number of frames printed from
its default, e.g.:

```python
from transformers.debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model, max_frames_to_save=100)
```

### Specific batch absolute min and max value tracing

The same debugging class can be used for per-batch tracing with the underflow/overflow detection feature turned off.

Let's say you want to watch the absolute min and max values for all the ingredients of each `forward` call of a given
batch, and only do that for batches 1 and 3. Then you instantiate this class as:

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3])
```

And now full batches 1 and 3 will be traced using the same format as the underflow/overflow detector does.

Batches are 0-indexed.

This is helpful if you know that the program starts misbehaving after a certain batch number, so you can fast-forward
right to that area. Here is a sample truncated output for such configuration:

```
                  *** Starting batch number=1 ***
abs min  abs max  metadata
                  shared Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.47e+04 input[0]
5.36e-05 7.92e+02 output
[...]
                  decoder.dropout Dropout
1.60e-07 2.27e+01 input[0]
0.00e+00 2.52e+01 output
                  decoder T5Stack
     not a tensor output
                  lm_head Linear
1.01e-06 7.92e+02 weight
0.00e+00 1.11e+00 input[0]
6.06e-02 8.39e+01 output
                   T5ForConditionalGeneration
     not a tensor output

                  *** Starting batch number=3 ***
abs min  abs max  metadata
                  shared Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.78e+04 input[0]
5.36e-05 7.92e+02 output
[...]
```

Here you will get a huge number of frames dumped - as many as there were forward calls in your model, so it may or may
not what you want, but sometimes it can be easier to use for debugging purposes than a normal debugger. For example, if
a problem starts happening at batch number 150. So you can dump traces for batches 149 and 150 and compare where
numbers started to diverge.

You can also specify the batch number after which to stop the training, with:

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3], abort_after_batch_num=3)
```
