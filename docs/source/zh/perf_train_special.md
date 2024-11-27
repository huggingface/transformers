<!--Copyright 2022 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 在Apple芯片(Apple silicon)上进行PyTorch训练

之前，在Mac上训练模型仅限于使用CPU。不过从PyTorch v1.12开始，通过将Apple的Metal性能着色器(MPS)作为后端集成到PyTorch中，[MPS后端](https://pytorch.org/docs/stable/notes/mps.html)能够把PyTorch操作视为自定义的Metal着色器来实现，并部署到`mps`设备上。所以您可以利用Apple芯片的GPU进行模型训练，大幅提升性能和训练速度。

<Tip warning={true}>

某些PyTorch操作目前还未在MPS上实现，可能会抛出错误提示。可以通过设置环境变量`PYTORCH_ENABLE_MPS_FALLBACK=1`来使用CPU内核以避免这种情况发生（您仍然会看到一个`UserWarning`）。

<br>

如果您遇到任何其他错误，请在[PyTorch库](https://github.com/pytorch/pytorch/issues)中创建一个issue，因为[`Trainer`]类中只集成了MPS后端.

</Tip>

配置好`mps`设备后，您可以：

* 在本地训练更大的网络或更大的批量大小
* 降低数据获取延迟，因为GPU的统一内存架构允许直接访问整个内存存储
* 降低成本，因为您不需要再在云端GPU上训练或增加额外的本地GPU

在确保已安装PyTorch后就可以开始使用了。MPS加速支持macOS 12.3及以上版本。

```bash
pip install torch torchvision torchaudio
```

[`TrainingArguments`]类默认使用`mps`设备(如果可用)因此无需显式设置设备。例如，您可以直接运行[run_glue.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py)脚本，在无需进行任何修改的情况下自动启用 MPS 后端。

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

用于[分布式设置](https://pytorch.org/docs/stable/distributed.html#backends)的后端(如`gloo`和`nccl`)不支持`mps`设备，这也意味着使用MPS后端时只能在单个GPU上进行训练。

您可以在[Introducing Accelerated PyTorch Training on Mac](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)博客文章中了解有关MPS后端的更多信息。