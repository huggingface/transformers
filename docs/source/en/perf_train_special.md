<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# PyTorch training on Apple silicon

Previously, training models on a Mac was limited to the CPU only. With the release of PyTorch v1.12, you can take advantage of training models with Apple's silicon GPUs for significantly faster performance and training. This is powered in PyTorch by integrating Apple's Metal Performance Shaders (MPS) as a backend. The [MPS backend](https://pytorch.org/docs/stable/notes/mps.html) implements PyTorch operations as custom Metal shaders and places these modules on a `mps` device.

<Tip warning={true}>

Some PyTorch operations are not implemented in MPS yet and will throw an error. To avoid this, you should set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU kernels instead (you'll still see a `UserWarning`).

<br>

If you run into any other errors, please open an issue in the [PyTorch](https://github.com/pytorch/pytorch/issues) repository because the [`Trainer`] only integrates the MPS backend.

</Tip>

With the `mps` device set, you can:

* train larger networks or batch sizes locally
* reduce data retrieval latency because the GPU's unified memory architecture allows direct access to the full memory store
* reduce costs because you don't need to train on cloud-based GPUs or add additional local GPUs

Get started by making sure you have PyTorch installed. MPS acceleration is supported on macOS 12.3+.

```bash
pip install torch torchvision torchaudio
```

[`TrainingArguments`] uses the `mps` device by default if it's available which means you don't need to explicitly set the device. For example, you can run the [run_glue.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py) script with the MPS backend automatically enabled without making any changes.

```diff
export TASK_NAME=mrpc

python examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path google-bert/bert-base-cased \
  --task_name $TASK_NAME \
- --use_mps_device \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir
```

Backends for [distributed setups](https://pytorch.org/docs/stable/distributed.html#backends) like `gloo` and `nccl` are not supported by the `mps` device which means you can only train on a single GPU with the MPS backend.

You can learn more about the MPS backend in the [Introducing Accelerated PyTorch Training on Mac](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/) blog post.
