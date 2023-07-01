<!---版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非符合许可证，否则您无法使用此文件。您可以在以下位置获取许可证的副本
    http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，否则根据许可证分发的软件分发在“按原样”基础上，没有任何明示或暗示的保证或条件。有关特定语言的权限和限制详见许可证。
⚠️ 请注意，此文件是 Markdown 格式，但包含我们的文档生成器的特定语法（类似于 MDX），可能无法在您的 Markdown 查看器中正确呈现。
-->

# 安装

根据您使用的深度学习库安装 🤗 Transformers，设置缓存，并可选择配置 🤗 Transformers 以离线运行。

🤗 Transformers 在 Python 3.6+、PyTorch 1.1.0+、TensorFlow 2.0+ 和 Flax 上进行了测试。按照下面的安装说明安装您正在使用的深度学习库：

* [PyTorch](https://pytorch.org/get-started/locally/) 安装说明。
* [TensorFlow 2.0](https://www.tensorflow.org/install/pip) 安装说明。
* [Flax](https://flax.readthedocs.io/en/latest/) 安装说明。

## 使用 pip 安装

您应该在 [虚拟环境](https://docs.python.org/3/library/venv.html) 中安装 🤗 Transformers。如果您对 Python 虚拟环境不熟悉，请参阅此 [指南](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)。虚拟环境可以更轻松地管理不同的项目，并避免依赖项之间的兼容性问题。

首先，在您的项目目录中创建一个虚拟环境：
```bash
python -m venv .env
```

激活虚拟环境。在 Linux 和 MacOS 上：
```bash
source .env/bin/activate
```
在 Windows 上激活虚拟环境
```bash
.env/Scripts/activate
```

现在，您可以使用以下命令安装 🤗 Transformers：
```bash
pip install transformers
```

仅支持 CPU 的情况下，您可以通过一行命令方便地安装 🤗 Transformers 和深度学习库。例如，使用以下命令安装 🤗 Transformers 和 PyTorch：
```bash
pip install 'transformers[torch]'
```

🤗 Transformers 和 TensorFlow 2.0：
```bash
pip install 'transformers[tf-cpu]'
```

<Tip warning={true}>

M1 / ARM 用户    
在安装 TensorFlow 2.0 之前，您需要安装以下内容 
```
brew install cmake
brew install pkg-config
```

</Tip>

🤗 Transformers 和 Flax：
```bash
pip install 'transformers[flax]'
```

最后，通过运行以下命令检查 🤗 Transformers 是否已正确安装。它将下载一个预训练模型：
```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

然后打印标签和分数：
```bash
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

## 从源代码安装
使用以下命令从源代码安装 🤗 Transformers：
```bash
pip install git+https://github.com/huggingface/transformers
```

此命令安装最新的“main”版本，而不是最新的“stable”版本。“main”版本对于保持与最新的开发一致非常有用。例如，如果自上次正式发布以来修复了错误但尚未发布新版本。但是，这意味着“main”版本可能并不总是稳定的。我们努力使“main”版本正常运行，大多数问题通常在几个小时或一天内解决。如果遇到问题，请打开一个 [Issue](https://github.com/huggingface/transformers/issues)，以便我们可以更快地修复。

通过运行以下命令检查 🤗 Transformers 是否已正确安装：

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love you'))"
```

## 可编辑安装

如果您想要执行以下操作，您将需要一个可编辑安装：

* 使用源代码的“main”版本。

* 对 🤗 Transformers 进行贡献并需要在代码中进行测试更改。

使用以下命令克隆存储库并安装 🤗 Transformers：

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

这些命令将连接您克隆的存储库的文件夹和 Python 库路径。Python 现在将在正常库路径之外的文件夹中搜索：`~/transformers/`。例如，如果您的 Python 包通常安装在 `~/anaconda3/envs/main/lib/python3.7/site-packages/` 中，Python 也将搜索您克隆到的文件夹。
<Tip warning={true}>

如果要继续使用该库，您必须保留 `transformers` 文件夹。
</Tip>

现在，您可以使用以下命令轻松更新到最新版本的 🤗 Transformers：
```bash
cd ~/transformers/
git pull
```

您的 Python 环境将在下一次运行时找到 `main` 版本的 🤗 Transformers。

## 使用 conda 安装

从 conda 渠道 `huggingface` 安装：
```bash
conda install -c huggingface transformers
```

## 缓存设置

预训练模型在 `~/.cache/huggingface/hub` 下载并本地缓存。这是 shell 环境变量 `TRANSFORMERS_CACHE` 给出的默认目录。在 Windows 上，默认目录由 `C:\Users\username\.cache\huggingface\hub` 给出。您可以按照下面优先级的 shell 环境变量更改这些变量来指定不同的缓存目录：

1. Shell 环境变量（默认）：`HUGGINGFACE_HUB_CACHE` 或 `TRANSFORMERS_CACHE`。2. Shell 环境变量：`HF_HOME`。3. Shell 环境变量：`XDG_CACHE_HOME` + `/huggingface`。

<Tip>

如果您是从此库的早期版本转换而来并且已设置了 shell 环境变量 `PYTORCH_TRANSFORMERS_CACHE` 或 `PYTORCH_PRETRAINED_BERT_CACHE`，则🤗 Transformers 将使用这些环境变量，除非您指定 shell 环境变量 `TRANSFORMERS_CACHE`。
</Tip>

## 离线模式

🤗 Transformers 可以在防火墙或离线环境中运行，只需使用本地文件即可。设置环境变量 `TRANSFORMERS_OFFLINE=1` 以启用此行为。

<Tip>

通过设置环境变量 `HF_DATASETS_OFFLINE=1`，将 [🤗 Datasets](https://huggingface.co/docs/datasets/) 添加到离线训练工作流程中。
</Tip>

例如，您通常会在普通网络上的防火墙上运行程序，以下是命令：
```bash
python examples/pytorch/translation/run_translation.py --model_name_or_path t5-small --dataset_name wmt16 --dataset_config ro-en ...
```

在离线实例中运行相同的程序：
```bash
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python examples/pytorch/translation/run_translation.py --model_name_or_path t5-small --dataset_name wmt16 --dataset_config ro-en ...
```

现在，脚本应该可以运行而无需挂起或等待超时，因为它知道它只应查找本地文件。

### 获取用于离线使用的模型和分词器 (Tokenizer)
🤗 Transformers 的另一种离线使用选项是提前下载文件，然后在需要离线使用时指向其本地路径。有三种方法可以做到这一点：

* 通过 [模型中心](https://huggingface.co/models) 上的用户界面下载文件，单击 ↓ 图标。
    ![download-icon](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/download-icon.png)
* 使用 [`PreTrainedModel.from_pretrained`] 和 [`PreTrainedModel.save_pretrained`] 工作流程：

    1. 使用 [`PreTrainedModel.from_pretrained`] 提前下载您的文件：

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")
    ```
    2. Save your files to a specified directory with [`PreTrainedModel.save_pretrained`]:

    ```py    
    >>> tokenizer.save_pretrained("./your/path/bigscience_t0")
    >>> model.save_pretrained("./your/path/bigscience_t0")
    ```
    3. Now when you're offline, reload your files with [`PreTrainedModel.from_pretrained`] from the specified directory:

    ```py    
    >>> tokenizer = AutoTokenizer.from_pretrained("./your/path/bigscience_t0")
    >>> model = AutoModel.from_pretrained("./your/path/bigscience_t0")
    ```
* 使用 [huggingface_hub](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub) 库以编程方式下载文件：

   1. 在虚拟环境中安装 `huggingface_hub` 库：

```bash    
python -m pip install huggingface_hub
```
    2. 使用 [`hf_hub_download`](https://huggingface.co/docs/hub/adding-a-library#download-files-from-the-hub) 函数将文件下载到指定路径。例如，下面的命令将从 [T0](https://huggingface.co/bigscience/T0_3B) 模型中下载 `config.json` 文件到你指定的路径：


    ```py
    >>> from huggingface_hub import hf_hub_download

    >>> hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="./your/path/bigscience_t0")
    ```
一旦文件下载并且在本地缓存，你可以指定它的本地路径来加载和使用它：

```py
>>> from transformers import AutoConfig

>>> config = AutoConfig.from_pretrained("./your/path/bigscience_t0/config.json")
```

<Tip>

查看 [如何从 Hub 下载文件](https://huggingface.co/docs/hub/how-to-downstream) 部分，了解有关从 Hub 下载文件的更多详细信息。

</Tip>
