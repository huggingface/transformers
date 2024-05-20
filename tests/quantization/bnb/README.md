# Testing mixed int8 quantization

![HFxbitsandbytes.png](https://cdn-uploads.huggingface.co/production/uploads/1660567705337-62441d1d9fdefb55a0b7d12c.png)

The following is the recipe on how to effectively debug `bitsandbytes` integration on Hugging Face `transformers`.

## Library requirements

+ `transformers>=4.22.0`
+ `accelerate>=0.12.0` 
+ `bitsandbytes>=0.31.5`.
## Hardware requirements

The following instructions are tested with 2 NVIDIA-Tesla T4 GPUs. To run successfully `bitsandbytes` you would need a 8-bit core tensor supported GPU. Note that Turing, Ampere or newer architectures - e.g. T4, RTX20s RTX30s, A40-A100, A6000 should be supported. 

## Virutal envs

```bash
conda create --name int8-testing python==3.8
pip install bitsandbytes>=0.31.5
pip install accelerate>=0.12.0
pip install transformers>=4.23.0
```
if `transformers>=4.23.0` is not released yet, then use:
```bash
pip install git+https://github.com/huggingface/transformers.git
```

## Troubleshooting

A list of common errors:

### Torch does not correctly do the operations on GPU

First check that:

```py
import torch

vec = torch.randn(1, 2, 3).to(0)
```

Works without any error. If not, install torch using `conda` like:

```bash
conda create --name int8-testing python==3.8
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install bitsandbytes>=0.31.5
pip install accelerate>=0.12.0
pip install transformers>=4.23.0
```
For the latest pytorch instructions please see [this](https://pytorch.org/get-started/locally/)

and the snippet above should work.

### ` bitsandbytes operations are not supported under CPU!`

This happens when some Linear weights are set to the CPU when using `accelerate`. Please check carefully `model.hf_device_map` and make sure that there is no `Linear` module that is assigned to CPU. It is fine to have the last module (usually the Lm_head) set on CPU.

### `To use the type as a Parameter, please correct the detach() semantics defined by __torch_dispatch__() implementation.`

Use the latest version of `accelerate` with a command such as: `pip install -U accelerate` and the problem should be solved.

### `Parameter has no attribue .CB` 

Same solution as above.

### `RuntimeError: CUDA error: an illegal memory access was encountered ... consider passing CUDA_LAUNCH_BLOCKING=1`

Run your script by pre-pending `CUDA_LAUNCH_BLOCKING=1` and you should observe an error as described in the next section.

### `CUDA illegal memory error: an illegal memory access at line...`:

Check the CUDA verisons with:
```bash
nvcc --version
```
and confirm it is the same version as the one detected by `bitsandbytes`. If not, run:
```bash
ls -l $CONDA_PREFIX/lib/libcudart.so
```
or 
```bash
ls -l $LD_LIBRARY_PATH
```
Check if `libcudart.so` has a correct symlink that is set. Sometimes `nvcc` detects the correct CUDA version but `bitsandbytes` doesn't. You have to make sure that the symlink that is set for the file `libcudart.so` is redirected to the correct CUDA file. 

Here is an example of a badly configured CUDA installation:

`nvcc --version` gives:

![Screenshot 2022-08-15 at 15.12.23.png](https://cdn-uploads.huggingface.co/production/uploads/1660569220888-62441d1d9fdefb55a0b7d12c.png)

which means that the detected CUDA version is 11.3 but `bitsandbytes` outputs:

![image.png](https://cdn-uploads.huggingface.co/production/uploads/1660569284243-62441d1d9fdefb55a0b7d12c.png)

First check:

```bash
echo $LD_LIBRARY_PATH
```

If this contains multiple paths separated by `:`. Then you have to make sure that the correct CUDA version is set. By doing:

```bash
ls -l $path/libcudart.so
```

On each path (`$path`) separated by `:`.
If not, simply run
```bash
ls -l $LD_LIBRARY_PATH/libcudart.so
```

and you can see

![Screenshot 2022-08-15 at 15.12.33.png](https://cdn-uploads.huggingface.co/production/uploads/1660569176504-62441d1d9fdefb55a0b7d12c.png)

If you see that the file is linked to the wrong CUDA version (here 10.2), find the correct location for `libcudart.so` (`find --name libcudart.so`) and replace the environment variable `LD_LIBRARY_PATH` with the one containing the correct `libcudart.so` file.