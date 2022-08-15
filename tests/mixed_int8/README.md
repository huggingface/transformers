# Testing mixed int8 quantization

![HFxbitsandbytes.png](https://s3.amazonaws.com/moonup/production/uploads/1660567705337-62441d1d9fdefb55a0b7d12c.png)

Hi there, this is a recipe on how to effectively debug `bitsandbytes` integration on HuggingFace `transformers`.

## Library requirements

+ `transformers==4.22.0.dev`
+ `accelerate==0.12.0` 
+ `0.31.8>=bitsandbytes>=0.31.5`.
## Hardware requirements

I am using a setup of 2 GPUs that are NVIDIA-Tesla T4 15GB - `younes-testing-multi-gpu` on GCP. To run successfully `bitsandbytes` you would need a 8-bit core tensor supported GPU. Note that Ampere architectures should be supported. Here is an exhaustive list of the supported GPU types at the time of writing:

- RTX 20s & RTX 30s
- A40-A100
- T4 + (e.g. Google Colab GPUs) 

## Virutal envs

```conda create --name int8-testing python==3.8```
```pip install bitsandbytes```
```pip install accelerate```
```pip install git+https://github.com/huggingface/transformers.git```

## Trobleshooting

A list of common errors:

### Torch does not correctly do the operations on GPU

First check that:

```py
import torch

vec = torch.randn(1, 2, 3).to(0)
```

Works without any error. If not install torch using `conda` like:

```conda create --name int8-testing python==3.8```
```pip install bitsandbytes```
```conda install pytorch torchvision torchaudio -c pytorch```
```pip install git+https://github.com/huggingface/transformers.git```
```pip install accelerate```

And the snippet above should work

### ` `bitsandbytes` operations are not supported under CPU!`

This happens when some Linear weights are set to the CPU when using `accelerate`. Please check carefully `model.hf_device_map` and make sure that there is no `Linear` module that is assigned to CPU. It is fine to have the last module (usually the Lm_head) set on CPU.

### `To use the type as a Parameter, please correct the detach() semantics defined by __torch_dispatch__() implementation.`

Use the latest version of `accelerate` with a command such as: `pip install --force accelerate` and the problem should be solved.

### `Parameter has no attribue .CB` 

Same comment as above

### `RuntimeError: CUDA error: an illegal memory access was encountered ... consider passing CUDA_LAUNCH_BLOCKING=1`

Run your script by pre-pending `CUDA_LAUNCH_BLOCKING=1` and you should observe an error as below:
### `CUDA illegal memory error: an illegal memory access at line...`:

Check the CUDA verisons with:
```
nvcc --version
```
And confirm it is the same version than the one detected by `bitsandbytes`. If not, run:
```
ls -l $CONDA_PREFIX/lib/libcudart.so
```
or 
```
ls -l $LD_LIBRARY_PATH
```
And check if `libcudart.so` has a correct simlink that is set. Sometimes `nvcc` detects the correct CUDA version but `bitsandbytes` doesn't. You have to make sure that the simlink that is set for the file `libcudart.so` is redirected to the correct CUDA file. 

Here is an example of a badly configured CUDA installation:

`nvcc --version` gives:

![Screenshot 2022-08-15 at 15.12.23.png](https://s3.amazonaws.com/moonup/production/uploads/1660569220888-62441d1d9fdefb55a0b7d12c.png)

Which means that the detected CUDA version is 11.3 but `bitsandbytes` outputs:

![image.png](https://s3.amazonaws.com/moonup/production/uploads/1660569284243-62441d1d9fdefb55a0b7d12c.png)

Therefore check:

```
ls -l /opt/conda/envs/py37/lib/libcudart.so
```

And you can see that:

![Screenshot 2022-08-15 at 15.12.33.png](https://s3.amazonaws.com/moonup/production/uploads/1660569176504-62441d1d9fdefb55a0b7d12c.png)

If you see that the file is linked to the wrong CUDA version (here 10.2), find the correct location for `libcudart.so` (`find --name ...`) and replace the environment variable `LD_LIBRARY_PATH` with the one containing the correct `libcudart.so` file.
### If `bitsandbytes` installation breaks everything:

It happened in a previous version that after installing `bitsandbytes` and running this script:

```py
import bitsandbytes as bnb
```

You get an error:

```
major, minor, revision = ...
Too many values to unpack...
```

Re-install `bitsandbytes==0.31.8` or `bitsandbytes==0.31.5` as everything worked fine on our Docker image with those versions. In the worst case remove the [line that installs bitsandbytes on the Dockerfile](https://github.com/huggingface/transformers/blob/d6eeb871706db0d64ab9ffd79f9545d95286b536/docker/transformers-all-latest-gpu/Dockerfile#L49)