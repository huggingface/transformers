<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<!---
A useful guide for English-Chinese translation of Hugging Face documentation
- Add space around English words and numbers when they appear between Chinese characters. E.g., 共 100 多种语言; 使用 transformers 库。
- Use square quotes, e.g.,「引用」

Dictionary

Hugging Face: Hugging Face（不翻译）
token: 词符（并用括号标注原英文）
tokenize: 词符化（并用括号标注原英文）
tokenizer: 词符化器（并用括号标注原英文）
transformer: transformer（不翻译）
pipeline: pipeline（不翻译）
API: API (不翻译）
inference: 推理
Trainer: 训练器。当作为类名出现时不翻译。
pretrained/pretrain: 预训练
finetune: 微调
community: 社区
example: 当特指仓库中 example 目录时翻译为「用例」
Python data structures (e.g., list, set, dict): 翻译为列表，集合，词典，并用括号标注原英文
NLP/Natural Language Processing: 以 NLP 出现时不翻译，以 Natural Language Processing 出现时翻译为自然语言处理
checkpoint: 检查点
-->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://huggingface.co/models"><img alt="Checkpoints on Hub" src="https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen"></a>
    <a href="https://circleci.com/gh/huggingface/transformers"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/transformers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <a href="https://github.com/huggingface/transformers/blob/main/README.md">English</a> |
        <b>简体中文</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hant.md">繁體中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_es.md">Español</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ja.md">日本語</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_hd.md">हिन्दी</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ru.md">Русский</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_pt-br.md">Português</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_it.md">Italiano</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_bn.md">বাংলা</a> |
    </p>
</h4>

<h3 align="center">
    <p>为文本、视觉、音频、视频与多模态提供推理与训练的先进预训练模型</p>
</h3>

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

Transformers 充当跨文本、计算机视觉、音频、视频与多模态的最先进机器学习模型的「模型定义框架」，同时覆盖推理与训练。

它将模型的定义集中化，使整个生态系统对该定义达成一致。`transformers` 是跨框架的枢纽：一旦某模型定义被支持，它通常就能兼容多数训练框架（如 Axolotl、Unsloth、DeepSpeed、FSDP、PyTorch‑Lightning 等）、推理引擎（如 vLLM、SGLang、TGI 等），以及依赖 `transformers` 模型定义的相关库（如 llama.cpp、mlx 等）。

我们的目标是持续支持新的最先进模型，并通过让模型定义保持简单、可定制且高效来普及其使用。

目前在 [Hugging Face Hub](https://huggingface.com/models) 上有超过 1M+ 使用 `transformers` 的[模型检查点](https://huggingface.co/models?library=transformers&sort=trending)，可随取随用。
 
今天就去探索 Hub，找到一个模型，并用 Transformers 立刻开始吧。

## 安装

Transformers 支持 Python 3.10+，以及 [PyTorch](https://pytorch.org/get-started/locally/) 2.4+。

使用 [venv](https://docs.python.org/3/library/venv.html) 或 [uv](https://docs.astral.sh/uv/)（一个基于 Rust 的快速 Python 包与项目管理器）创建并激活虚拟环境：

```py
# venv
python -m venv .my-env
source .my-env/bin/activate
# uv
uv venv .my-env
source .my-env/bin/activate
```

在虚拟环境中安装 Transformers：

```py
# pip
pip install "transformers[torch]"

# uv
uv pip install "transformers[torch]"
```

如果你需要库中的最新改动或计划参与贡献，可从源码安装（注意：最新版可能不稳定；如遇错误，欢迎在 [issues](https://github.com/huggingface/transformers/issues) 中反馈）：

```shell
git clone https://github.com/huggingface/transformers.git
cd transformers

# pip
pip install '.[torch]'

# uv
uv pip install '.[torch]'
```

## 快速上手

使用 [Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) API 一步上手。`Pipeline` 是一个高级推理类，支持文本、音频、视觉与多模态任务，负责输入预处理并返回适配的输出。

实例化一个用于文本生成的 pipeline，指定使用的模型。模型会被下载并缓存，方便复用。最后传入文本作为提示：

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1) to use the right ingredients and 2) to follow the recipe exactly. the recipe for the cake is as follows: 1 cup of sugar, 1 cup of flour, 1 cup of milk, 1 cup of butter, 1 cup of eggs, 1 cup of chocolate chips. if you want to make 2 cakes, how much sugar do you need? To make 2 cakes, you will need 2 cups of sugar.'}]
```

要与模型进行「聊天」，用法也一致。唯一不同是需要构造一段「聊天历史」（即 `Pipeline` 的输入）：

> [!TIP]
> 你也可以直接在命令行与模型聊天：
> ```shell
> transformers chat Qwen/Qwen2.5-0.5B-Instruct
> ```

```py
import torch
from transformers import pipeline

chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

展开下方示例，查看 `Pipeline` 在不同模态与任务中的用法。

<details>
<summary>自动语音识别</summary>

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

</details>

<details>
<summary>图像分类</summary>

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"></a>
</h3>

```py
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="facebook/dinov2-small-imagenet1k-1-layer")
pipeline("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
[{"label": "macaw", "score": 0.997848391532898},
 {"label": "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita",
  "score": 0.0016551691805943847},
 {"label": "lorikeet", "score": 0.00018523589824326336},
 {"label": "African grey, African gray, Psittacus erithacus",
  "score": 7.85409429227002e-05},
 {"label": "quail", "score": 5.502637941390276e-05}]
```

</details>

<details>
<summary>视觉问答</summary>

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg"></a>
</h3>

```py
from transformers import pipeline

pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
pipeline(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="What is in the image?",
)
[{"answer": "statue of liberty"}]
```

</details>

## 为什么要用 Transformers？

1. 易于使用的最先进模型：
    - 在自然语言理解与生成、计算机视觉、音频、视频与多模态任务上表现优越。
    - 对研究者、工程师与开发者友好且低门槛。
    - 少量用户侧抽象，仅需学习三个类。
    - 统一的 API，使用所有预训练模型体验一致。

1. 更低计算开销与更小碳足迹：
    - 共享已训练的模型，而非每次从零开始训练。
    - 减少计算时间与生产环境成本。
    - 覆盖数十种模型架构，跨所有模态提供 1M+ 预训练检查点。

1. 在模型生命周期的每个阶段都可以选用合适的框架：
    - 3 行代码即可训练最先进模型。
    - 在 PyTorch/JAX/TF2.0 间自由迁移同一个模型。
    - 为训练、评估与生产挑选最合适的框架。

1. 轻松定制模型或用例：
    - 为每个架构提供示例以复现原论文结果。
    - 尽可能一致地暴露模型内部。
    - 模型文件可独立于库使用，便于快速实验。

<a target="_blank" href="https://huggingface.co/enterprise">
    <img alt="Hugging Face Enterprise Hub" src="https://github.com/user-attachments/assets/247fb16d-d251-4583-96c4-d3d76dda4925">
</a><br>

## 为什么我不该用 Transformers？

- 该库不是一个可自由拼搭的神经网络模块化工具箱。模型文件中的代码刻意减少额外抽象，以便研究者能快速在各个模型上迭代，而无需深入更多抽象或文件跳转。
- 训练 API 优化用于 Transformers 提供的 PyTorch 模型。若需要通用的机器学习训练循环，请使用其它库，如 [Accelerate](https://huggingface.co/docs/accelerate)。
- [示例脚本](https://github.com/huggingface/transformers/tree/main/examples)只是「示例」。它们不一定能直接适配你的具体用例，需要你进行必要的改动。


## 100 个使用 Transformers 的项目

Transformers 不止是一个使用预训练模型的工具包，它还是围绕 Hugging Face Hub 构建的项目社区。我们希望 Transformers 能助力开发者、研究人员、学生、老师、工程师与任何人构建理想项目。

为庆祝 Transformers 获得 100,000 颗星，我们制作了 [awesome-transformers](https://github.com/huggingface/transformers/blob/main/awesome-transformers.md) 页面，展示了 100 个由社区构建的优秀项目。

如果你拥有或使用某个项目，认为它应该在列表中出现，欢迎提交 PR 添加它！

## 示例模型

你可以直接在它们的 [Hub 模型页](https://huggingface.co/models) 上测试我们的多数模型。

展开每个模态以查看不同用例中的部分示例模型。

<details>
<summary>音频</summary>

- 使用 [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo) 进行音频分类
- 使用 [Moonshine](https://huggingface.co/UsefulSensors/moonshine) 进行自动语音识别
- 使用 [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks) 进行关键词检索
- 使用 [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16) 进行语音到语音生成
- 使用 [MusicGen](https://huggingface.co/facebook/musicgen-large) 文本到音频生成
- 使用 [Bark](https://huggingface.co/suno/bark) 文本到语音生成

</details>

<details>
<summary>计算机视觉</summary>

- 使用 [SAM](https://huggingface.co/facebook/sam-vit-base) 自动生成掩码
- 使用 [DepthPro](https://huggingface.co/apple/DepthPro-hf) 进行深度估计
- 使用 [DINO v2](https://huggingface.co/facebook/dinov2-base) 进行图像分类
- 使用 [SuperPoint](https://huggingface.co/magic-leap-community/superpoint) 进行关键点检测
- 使用 [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor) 进行关键点匹配
- 使用 [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd) 进行目标检测
- 使用 [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple) 进行姿态估计
- 使用 [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large) 进行通用分割
- 使用 [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large) 进行视频分类

</details>

<details>
<summary>多模态</summary>

- 使用 [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B) 实现音频或文本到文本
- 使用 [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base) 进行文档问答
- 使用 [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) 实现图像或文本到文本
- 使用 [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b) 进行图文描述
- 使用 [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf) 进行基于 OCR 的文档理解
- 使用 [TAPAS](https://huggingface.co/google/tapas-base) 进行表格问答
- 使用 [Emu3](https://huggingface.co/BAAI/Emu3-Gen) 进行统一的多模态理解与生成
- 使用 [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) 视觉到文本
- 使用 [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf) 进行视觉问答
- 使用 [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224) 进行视觉指代表达分割

</details>

<details>
<summary>NLP</summary>

- 使用 [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) 进行掩码词填充
- 使用 [Gemma](https://huggingface.co/google/gemma-2-2b) 进行命名实体识别（NER）
- 使用 [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) 进行问答
- 使用 [BART](https://huggingface.co/facebook/bart-large-cnn) 进行摘要
- 使用 [T5](https://huggingface.co/google-t5/t5-base) 进行翻译
- 使用 [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B) 进行文本生成
- 使用 [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B) 进行文本分类

</details>

## 引用

我们已将此库的[论文](https://www.aclweb.org/anthology/2020.emnlp-demos.6/)正式发表，如果你使用了 🤗 Transformers 库，请引用:
```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```
