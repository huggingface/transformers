<!--版权所有 2022 年 The HuggingFace 团队。保留所有权利。
根据 Apache License，Version 2.0（“许可证”）获得许可；除非符合许可证，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“原样”分发，不附带任何形式的保证或条件。有关许可证的详细信息请参阅许可证。
⚠️ 请注意，此文件是 Markdown 格式，但包含特定于我们文档构建器（类似于 MDX）的语法，可能无法正确在您的 Markdown 查看器中呈现。
-->

# 在 CPU 上高效训练

本指南侧重于在 CPU 上高效训练大型模型。

## 使用 IPEX 进行混合精度训练

IPEX 针对支持 AVX-512 或更高版本的 CPU 进行了优化，对于仅支持 AVX2 的 CPU 也可以正常工作。因此，在支持 AVX-512 或更高版本的英特尔 CPU 代
用户体验。

请查看更详细的 [自动混合精度信息](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/amp.html)。

### IPEX 安装：

IPEX 的发布遵循 PyTorch，在 pip 上安装：
| PyTorch 版本   | IPEX 版本   || :---------------: | :----------:   || 1.13              |  1.13.0+cpu    || 1.12              |  1.12.300+cpu  || 1.11              |  1.11.200+cpu  || 1.10              |  1.10.100+cpu  |
```
pip install intel_extension_for_pytorch==<version_name> -f https://developer.intel.com/ipex-whl-stable-cpu
```

请查看更多 [IPEX 安装方法](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/installation.html)。

### 在训练器中的使用

要在训练器中启用 IPEX 的自动混合精度，用户应在训练命令参数中添加 `use_ipex`、`bf16` 和 `no_cuda`。
以 [Transformers question-answering](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) 为例子

- 在 CPU 上使用 BF16 自动混合精度进行训练：
<pre> python run_qa.py \
--model_name_or_path bert-base-uncased \
--dataset_name squad \
--do_train \
--do_eval \
--per_device_train_batch_size 12 \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/debug_squad/ \
<b>--use_ipex \</b>
<b>--bf16 --no_cuda </b> </pre> 

### 实践示例

博客：[加速 PyTorch Transformers 与 Intel Sapphire Rapids](https://huggingface.co/blog/intel-sapphire-rapids)