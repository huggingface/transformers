# Testing mixed int8 quantization

## Hardware requirements

I am using a setup of 2 GPUs that are NVIDIA-Tesla T4 15GB

## Virutal envs

```conda create --name int8-testing python==3.8```
```git clone https://github.com/younesbelkada/transformers.git && git checkout integration-8bit```
```pip install -e ".[dev]"```
```pip install -i https://test.pypi.org/simple/ bitsandbytes-cuda114```
```pip install git+https://github.com/huggingface/accelerate.git@e0212893ea6098cc0a7a3c7a6eb286a9104214c1```

## Trouble shooting

### Check driver settings:

```
nvcc --version
```

```
ls -l $CONDA_PREFIX/lib/libcudart.so
```