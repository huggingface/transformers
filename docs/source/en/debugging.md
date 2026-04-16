<!--Copyright 2024 The HuggingFace Team. All rights reserved.

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

Debugging distributed training problems typically falls into one of these categories: numerical issues, communication failures, runtime errors, and build errors.

## Underflow and overflow detection

Underflow and overflow occur when activations or weights reach `inf` or `nan`, or when `loss=NaN`. To detect these, enable the `DebugUnderflowOverflow` module in [`TrainingArguments.debug`], or import and add it to your own training loop.

<hfoptions id="overflow">
<hfoption id="Trainer">

```py
from transformers import TrainingArguments

args = TrainingArguments(
    debug="underflow_overflow",
    ...
)
```

</hfoption>
<hfoption id="PyTorch training loop">

```py
from transformers.debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model)
```

</hfoption>
</hfoptions>

[`~debug_utils.DebugUnderflowOverflow`] inserts hooks into the model to test input and output variables and the corresponding model weights after each forward call. When `inf` or `nan` is detected in at least one element of the activations or weights, the module prints a report like the one below.

The example below is for fp16 mixed precision training with [google/mt5-small](https://huggingface.co/google/mt5-small).

```shell
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

The first line shows the batch number where the error occurred. In this case, it occurred on batch 0.

Each frame describes the module it reports on. For example, the frame below reports on `encoder.block.2.layer.1.layer_norm`, the layer norm in the first layer of the encoder's second block. The forward calls are to `T5LayerNorm`.

```shell
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
```

The last frame reports on the `Dropout.forward` function, which calls the `dropout` attribute inside the `DenseReluDense` class. The overflow (`inf`) occurred in the encoder's second block on the first batch. The largest input element was 6.27e+04.

```shell
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

`T5DenseGatedGeluDense.forward` output activations reached a maximum of 6.27e+04, which is close to fp16's maximum of 6.4e+04. In the next step, `Dropout` renormalizes the weights after zeroing some elements, pushing the maximum above 6.4e+04 and causing the overflow.

Now that you know where the error is happening, investigate the modeling code in [modeling_t5.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py).

```py
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

One fix is to switch to fp32 a few steps before the values grew too large, so numbers don't overflow when multiplied or summed. Another option is to disable mixed precision training (`amp`) temporarily.

```py
import torch

def forward(self, hidden_states):
    device_type = hidden_states.device.type
    if torch.is_autocast_enabled(device_type):
        with torch.amp.autocast(device_type, enabled=False):
            return self._forward(hidden_states)
    else:
        return self._forward(hidden_states)
```

The report only covers inputs and outputs of full frames. To analyze intermediate values inside any `forward` function, add `detect_overflow` after each forward call to track `inf` or `nan` in `forwarded_states`.

```py
from transformers.debug_utils import detect_overflow

class T5LayerFF(nn.Module):
    [...]

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        detect_overflow(forwarded_states, "after layer_norm")
        forwarded_states = self.DenseReluDense(forwarded_states)
        detect_overflow(forwarded_states, "after DenseReluDense")
        return hidden_states + self.dropout(forwarded_states)
```

Configure the number of frames printed by [`~debug_utils.DebugUnderflowOverflow`].

```py
from transformers.debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model, max_frames_to_save=100)
```

### Batch tracing

[`~debug_utils.DebugUnderflowOverflow`] can also trace the absolute minimum and maximum values in each batch with underflow and overflow detection disabled. This helps you locate where values start diverging in your model.

The example below traces batches 1 and 3 (batches are zero-indexed).

```py
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3])
```

```shell
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

[`~debug_utils.DebugUnderflowOverflow`] reports many frames, which makes it easier to spot where values diverge. If you know the problem is around batch 150, focus the trace on batches 149 and 150 to compare where the numbers start to differ.

You can also stop the trace after a specific batch number, for example batch 3.

```py
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3], abort_after_batch_num=3)
```

## Communication

Distributed training requires inter-process and inter-node communication, which is a common source of errors.

Download the script below to diagnose network issues, then run it to test GPU communication. The command below tests two GPUs. Adjust `--nproc_per_node` and `--nnodes` for your system.

```bash
wget https://raw.githubusercontent.com/huggingface/transformers/main/scripts/distributed/torch-distributed-gpu-test.py
python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

The script prints `OK` if both GPUs communicate and allocate memory successfully. See the diagnostic script for more details and a recipe for running it in a SLURM environment.

Set `NCCL_DEBUG=INFO` to get detailed NCCL debugging output.

```bash
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

## DeepSpeed

When you hit an error, first check whether DeepSpeed is the cause. Retry your setup without DeepSpeed, and if the error persists, report the issue. For issues unrelated to the Transformers integration, open an issue on the DeepSpeed [repository](https://github.com/microsoft/DeepSpeed).

For issues related to the Transformers integration, include the following information.

* The full DeepSpeed config file.
* The command line arguments for [`Trainer`] or the [`TrainingArguments`] if you're scripting the [`Trainer`] setup yourself (don't dump the entire [`TrainingArguments`] which contains many irrelevant entries).
* The outputs of these commands.

    ```bash
    python -c 'import torch; print(f"torch: {torch.__version__}")'
    python -c 'import transformers; print(f"transformers: {transformers.__version__}")'
    python -c 'import deepspeed; print(f"deepspeed: {deepspeed.__version__}")'
    ```

* A link to a Google Colab notebook to reproduce the issue.
* A standard or non-custom dataset or an existing example to reproduce the issue.

### Process killed at startup

If the DeepSpeed process is killed during launch without a traceback, the program tried to allocate more CPU memory than is available or allowed. The OS kernel terminates the process in either case.

Check whether your config file has `offload_optimizer`, `offload_param`, or both configured to offload to the CPU.

If you have NVMe and ZeRO-3 set up, try offloading to the NVMe instead. [Estimate](https://deepspeed.readthedocs.io/en/latest/memory.html) the memory requirements of your model first.

### NaN loss

NaN loss often occurs when a model is pretrained in bf16 and then it is used with fp16 (this is especially common with TPU-trained models). Use fp32 or bf16 if your hardware supports it (TPUs, Ampere GPUs or newer).

fp16 can also cause overflow. If your config file looks like the one below, you may see overflow errors in the logs.

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

The `OVERFLOW!` error below means the DeepSpeed loss scaler couldn't find a scaling coefficient to overcome the loss overflow. Try a higher `initial_scale_power` value (32 usually works).

```bash
0%|                                                                                                                             | 0/189 [00:00<?, ?it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 262144
  1%|▌                                                                                                                    | 1/189 [00:00<01:26,  2.17it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 131072.0
  1%|█▏
 [...]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 14%|████████████████▌                                                                                                   | 27/189 [00:14<01:13,  2.21it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 15%|█████████████████▏                                                                                                  | 28/189 [00:14<01:13,  2.18it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 15%|█████████████████▊                                                                                                  | 29/189 [00:15<01:13,  2.18it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
[...]
```

## DeepSpeed CUDA

DeepSpeed compiles CUDA C++ code, which is a common source of build errors for PyTorch extensions that require CUDA. These errors depend on how CUDA is installed on your system.

```bash
pip install deepspeed
```

> [!TIP]
> For any other installation issues, [open an issue](https://github.com/microsoft/DeepSpeed/issues) with the DeepSpeed team.

### Non-identical toolkits

PyTorch ships with its own CUDA toolkit, but DeepSpeed requires an identical CUDA version installed system-wide. If you installed PyTorch with `cudatoolkit==10.2` in your Python environment, you'll also need CUDA 10.2 installed everywhere.

The exact location varies by system, but `/usr/local/cuda-10.2` is the most common path on Unix systems. Once CUDA is set up and added to your `PATH`, find the installation location with this command.

```bash
which nvcc
```

### Multiple toolkits

Your system may have more than one CUDA toolkit installed.

```text
/usr/local/cuda-10.2
/usr/local/cuda-11.0
```

Package installers typically set paths to the last installed version. If the build fails because it can't find the right CUDA version, configure `PATH` and `LD_LIBRARY_PATH` to point to the correct path.

Check these environment variables first.

```bash
echo $PATH
echo $LD_LIBRARY_PATH
```

`PATH` lists executable locations. `LD_LIBRARY_PATH` lists shared library locations. Earlier entries take priority, and `:` separates multiple entries. Prepend the correct CUDA path to prioritize it.

```bash
# adjust the version and full path if needed
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
```

Also verify the assigned directories exist. The `lib64` sub-directory contains CUDA `.so` objects like `libcudart.so`. Check the actual filenames and update accordingly.

### Older versions

Older CUDA versions sometimes require older compiler versions. For example, if CUDA requires `gcc-7` but your system only has `gcc-9`, the build will fail. Install the required older compiler and create a symlink so the CUDA build system can find it.

```bash
# adjust the path to your system
sudo ln -s /usr/bin/gcc-7  /usr/local/cuda-10.2/bin/gcc
sudo ln -s /usr/bin/g++-7  /usr/local/cuda-10.2/bin/g++
```

### Prebuild

If you're still having trouble installing DeepSpeed or building it at runtime, prebuild the DeepSpeed modules first. Run the commands below for a local build.

```bash
git clone https://github.com/deepspeedai/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log
```

> [!TIP]
> Add `DS_BUILD_AIO=1` to the build command to use NVMe offload. Make sure you install the libaio-dev package system-wide.

Next, set your GPU architecture in `TORCH_CUDA_ARCH_LIST`. A complete list of NVIDIA GPUs and their architectures is on the [CUDA GPUs page](https://developer.nvidia.com/cuda-gpus). To check the PyTorch version that corresponds to your architecture, run the command below.

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

Find the architecture for a GPU with the following command.

<hfoptions id="arch">
<hfoption id="same GPUs">

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())"
```

</hfoption>
<hfoption id="specific GPU">

Run the following command to find the architecture for GPU `0`. The output shows `major` and `minor` values that together form the GPU architecture. The example below shows architecture `8.6`.

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import torch; \
print(torch.cuda.get_device_properties(torch.device('cuda')))
"_CudaDeviceProperties(name='GeForce RTX 3090', major=8, minor=6, total_memory=24268MB, multi_processor_count=82)"
```

</hfoption>
</hfoptions>

For a result of `8, 6`, set `TORCH_CUDA_ARCH_LIST="8.6"`. For multiple GPUs with different architectures, list them like `TORCH_CUDA_ARCH_LIST="6.1;8.6"`.

You can omit `TORCH_CUDA_ARCH_LIST` and let the build program detect the GPU architecture automatically, but it might not match the actual GPU on the target machine. Explicitly setting the architecture is more reliable.

For training on multiple machines with the same setup, build a binary wheel.

```bash
git clone https://github.com/deepspeedai/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 \
python setup.py build_ext -j8 bdist_wheel
```

This generates a binary wheel like `dist/deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl`. Install it locally or on another machine.

```bash
pip install deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl
```
