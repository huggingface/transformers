# Testing mixed int8 quantization

## Hardware requirements

I am using a setup of 2 GPUs that are NVIDIA-Tesla T4 15GB

## Virutal envs

```conda create --name int8-testing python==3.8```
```git clone https://github.com/younesbelkada/transformers.git && git checkout integration-8bit```
```pip install -e ".[dev]"```
```pip install -i https://test.pypi.org/simple/ bitsandbytes```
```pip install git+https://github.com/huggingface/accelerate.git@e0212893ea6098cc0a7a3c7a6eb286a9104214c1```


## Trobleshooting

```conda create --name int8-testing python==3.8```
```pip install -i https://test.pypi.org/simple/ bitsandbytes```
```conda install pytorch torchvision torchaudio -c pytorch```
```git clone https://github.com/younesbelkada/transformers.git && git checkout integration-8bit```
```pip install -e ".[dev]"```
```pip install git+https://github.com/huggingface/accelerate.git@b52b793ea8bac108ba61192eead3cf11ca02433d```

### Check driver settings:

```
nvcc --version
```

```
ls -l $CONDA_PREFIX/lib/libcudart.so
```

### Recurrent bugs

Sometimes you have to run a "dummy" inference pass when dealing with a multi-GPU setup. Checkout the ```test_multi_gpu_loading``` and the ```test_pipeline``` functions.