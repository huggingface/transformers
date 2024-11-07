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

# 为 🤗 Transformers 做贡献

欢迎所有人为 🤗 Transformers 做出贡献，我们重视每个人的贡献。代码贡献并不是帮助社区的唯一途径。回答问题、帮助他人和改进文档也非常有价值。

宣传 🤗 Transformers 也会帮助我们！比如在博客文章里介绍一下这个库是如何帮助你完成了很棒的项目，每次它帮助你时都在 Twitter 上大声宣传，或者给这个代码仓库点⭐️来表示感谢。

无论你选择以哪种方式做出贡献，请注意并尊重我们的[行为准则](https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md)。

**本指南的灵感来源于 [scikit-learn贡献指南](https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md) ，它令人印象深刻.**

## 做贡献的方法

有多种方法可以为 🤗 Transformers 做贡献：

* 修复现有代码中尚未解决的问题。
* 提交与 bug 或所需新功能相关的 issue。
* 实现新的模型。
* 为示例或文档做贡献。

如果你不知道从哪里开始，有一个特别的 [Good First Issue](https://github.com/huggingface/transformers/contribute) 列表。它会列出一些适合初学者的开放的 issues，并帮助你开始为开源项目做贡献。只需要在你想要处理的 issue 下发表评论就行。 

如果想要稍微更有挑战性的内容，你也可以查看 [Good Second Issue](https://github.com/huggingface/transformers/labels/Good%20Second%20Issue) 列表。总的来说，如果你觉得自己知道该怎么做，就去做吧，我们会帮助你达到目标的！🚀

> 所有的贡献对社区来说都同样宝贵。🥰

## 修复尚未解决的问题

如果你发现现有代码中存在问题，并且已经想到了解决方法，请随时[开始贡献](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md/#create-a-pull-request) 并创建一个 Pull Request！

## 提交与 bug 相关的 issue 或功能请求

在提交与错误相关的 issue 或功能请求时，请尽量遵循下面的指南。这能让我们更容易迅速回复你，并提供良好的反馈意见。

### 你发现了 bug 吗？

🤗 Transformers 之所以强大可靠，要感谢用户报告了他们遇到的问题。

在提出issue之前，请你**确认该 bug 尚未被报告**（使用 GitHub 的 Issues 下面的搜索栏）。issue 也应该是与库本身的 bug 有关，而不是与你的代码有关。如果不确定 bug 是在你的代码中还是在库中，请先在[论坛](https://discuss.huggingface.co/)中询问。这有助于我们更快地解决与库相关的问题。

一旦你确认该 bug 尚未被报告，请在你的 issue 中包含以下信息，以便我们快速解决：

* 使用的**操作系统类型和版本**，以及 **Python**、**PyTorch** 和 **TensorFlow** 的版本。
* 一个简短、独立的代码片段，可以让我们在不到30秒内重现这个问题。
* 如果发生异常，请提供*完整的* traceback。
* 附上你认为可能有帮助的任何其他附加信息，如屏幕截图。

想要自动获取操作系统和软件版本，请运行以下命令：

```bash
transformers-cli env
```

你也可以从代码仓库的根目录下运行相同的命令：

```bash
python src/transformers/commands/transformers_cli.py env
```

### 你想要新功能吗？

如果你希望在 🤗 Transformers 中看到新功能，请提出一个 issue 并包含以下内容：

1. 这个新功能的*动机*是什么呢？是因为使用这个库时遇到了问题或者感到了某种不满吗？是因为你的项目需要这个功能吗？或者是你自己开发了某项内容，并且认为它可能会对社区有所帮助？

   不管是什么，我们都很想听！

2. 请尽可能详细地描述你想要的功能。你告诉我们的越多，我们就能更好地帮助你。
3. 请提供一个*代码片段*，演示该功能的使用方法。
4. 如果这个功能与某篇论文相关，请包含链接。

如果你描述得足够清晰，那么在你创建 issue 时，我们已经完成了80%的工作。

我们已经添加了[模板](https://github.com/huggingface/transformers/tree/main/templates)，可能有助于你提出 issue。

## 你想要实现一个新模型吗？

我们会持续发布新模型，如果你想要实现一个新模型，请提供以下信息:

* 模型的简要描述和论文链接。
* 如果实现是开源的，请提供实现的链接。
* 如果模型权重可用，请提供模型权重的链接。

如果你想亲自贡献模型，请告诉我们。让我们帮你把它添加到 🤗 Transformers！

我们还有一个更技术性的指南，告诉你[如何将模型添加到 🤗 Transformers](https://huggingface.co/docs/transformers/add_new_model)。

## 你想要添加文档吗？

我们始终在寻求改进文档，使其更清晰准确。请告诉我们如何改进文档，比如拼写错误以及任何缺失、不清楚或不准确的内容。我们非常乐意进行修改，如果你有兴趣，我们也可以帮助你做出贡献！

有关如何生成、构建和编写文档的更多详细信息，请查看文档 [README](https://github.com/huggingface/transformers/tree/main/docs)。

## 创建 Pull Request

在开始编写任何代码之前，我们强烈建议你先搜索现有的 PR(Pull Request) 或 issue，以确保没有其他人已经在做同样的事情。如果你不确定，提出 issue 来获取反馈意见是一个好办法。

要为 🤗 Transformers 做贡献，你需要基本的 `git` 使用技能。虽然 `git` 不是一个很容易使用的工具，但它提供了非常全面的手册，在命令行中输入 `git --help` 并享受吧！如果你更喜欢书籍，[Pro Git](https://git-scm.com/book/en/v2)是一本很好的参考书。

要为 🤗 Transformers 做贡献，你需要 **[Python 3.9](https://github.com/huggingface/transformers/blob/main/setup.py#L426)** 或更高版本。请按照以下步骤开始贡献：

1. 点击[仓库](https://github.com/huggingface/transformers)页面上的 **[Fork](https://github.com/huggingface/transformers/fork)** 按钮，这会在你的 GitHub 账号下拷贝一份代码。

2. 把派生仓库克隆到本地磁盘，并将基础仓库添加为远程仓库：

   ```bash
   git clone git@github.com:<your Github handle>/transformers.git
   cd transformers
   git remote add upstream https://github.com/huggingface/transformers.git
   ```

3. 创建一个新的分支来保存你的更改：

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

   🚨 **不要**在 `main` 分支工作!

4. 在虚拟环境中运行以下命令来设置开发环境：

   ```bash
   pip install -e ".[dev]"
   ```

   如果在虚拟环境中已经安装了 🤗 Transformers，请先使用 `pip uninstall transformers` 卸载它，然后再用 `-e` 参数以可编辑模式重新安装。
   
   根据你的操作系统，以及 Transformers 的可选依赖项数量的增加，可能会在执行此命令时出现失败。如果出现这种情况，请确保已经安装了你想使用的深度学习框架（PyTorch, TensorFlow 和 Flax），然后执行以下操作：

   ```bash
   pip install -e ".[quality]"
   ```

   大多数情况下，这些应该够用了。

5. 在你的分支上开发相关功能。

   在编写代码时，请确保测试套件通过。用下面的方式运行受你的更改影响的测试：

   ```bash
   pytest tests/<TEST_TO_RUN>.py
   ```

   想了解更多关于测试的信息，请阅读[测试](https://huggingface.co/docs/transformers/testing)指南。

   🤗 Transformers 使用 `black` 和 `ruff` 来保持代码风格的一致性。进行更改后，使用以下命令自动执行格式更正和代码验证：

   ```bash
   make fixup
   ```

   它已经被优化为仅适用于你创建的 PR 所修改过的文件。

   如果想要逐个运行检查，可以使用以下命令：

   ```bash
   make style
   ```

   🤗 Transformers 还使用了 `ruff` 和一些自定义脚本来检查编码错误。虽然质量管理是通过 CI 进行的，但你也可以使用以下命令来运行相同的检查：

   ```bash
   make quality
   ```

   最后，我们有许多脚本来确保在添加新模型时不会忘记更新某些文件。你可以使用以下命令运行这些脚本：

   ```bash
   make repo-consistency
   ```

   想要了解有关这些检查及如何解决相关问题的更多信息，请阅读 [检查 Pull Request](https://huggingface.co/docs/transformers/pr_checks) 指南。

   如果你修改了 `docs/source` 目录下的文档，请确保文档仍然能够被构建。这个检查也会在你创建 PR 时在 CI 中运行。如果要进行本地检查，请确保安装了文档构建工具：
   
   ```bash
   pip install ".[docs]"
   ```

   在仓库的根目录下运行以下命令：

   ```bash
   doc-builder build transformers docs/source/en --build_dir ~/tmp/test-build
   ```

   这将会在 `~/tmp/test-build` 文件夹中构建文档，你可以使用自己喜欢的编辑器查看生成的 Markdown 文件。当你创建 PR 时，也可以在GitHub上预览文档。

   当你对修改满意后，使用 `git add` 把修改的文件添加到暂存区，然后使用 `git commit` 在本地记录你的更改:

   ```bash
   git add modified_file.py
   git commit
   ```

   请记得写一个[好的提交信息](https://chris.beams.io/posts/git-commit/)来清晰地传达你所做的更改！

   为了保持你的代码副本与原始仓库的最新状态一致，在你创建 PR *之前*或者在管理员要求的情况下，把你的分支在 `upstream/branch` 上进行 rebase：

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   把你的更改推送到你的分支：

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

   如果你已经创建了一个 PR，你需要使用 `--force` 参数进行强制推送。如果 PR 还没有被创建，你可以正常推送你的更改。

6. 现在你可以转到 GitHub 上你的账号下的派生仓库，点击 **Pull Request** 来创建一个 PR。 请确保勾选我们 [checklist](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md/#pull-request-checklist) 下的所有项目。准备好这些后，可以将你的更改发送给项目管理员进行审查。

7. 如果管理员要求你进行更改，别气馁，我们的核心贡献者也会经历相同的事情！请在你的本地分支上进行工作，并将更改推送到派生仓库，以便于每个人都可以在 PR 中看到你的更改。这样它们会自动出现在 PR 中。

### Pull request 的检查清单

☐ Pull request 的标题应该总结你的贡献内容。<br>
☐ 如果你的 Pull request 解决了一个issue，请在 Pull request 描述中提及该 issue 的编号，以确保它们被关联起来（这样查看 issue 的人就知道你正在处理它）。<br>
☐ 如果是正在进行中的工作，请在标题前加上 [WIP]。这有助于避免重复工作和区分哪些 PR 可以合并。<br>
☐ 确保可以通过现有的测试。<br>
☐ 如果添加了新功能，请同时添加对应的测试。<br>
   - 如果添加一个新模型，请使用 `ModelTester.all_model_classes = (MyModel, MyModelWithLMHead,...)` 来触发通用测试。
   - 如果你正在添加新的 `@slow` 测试，请确保通过以下检查：`RUN_SLOW=1 python -m pytest tests/models/my_new_model/test_my_new_model.py`
   - 如果你正在添加一个新的分词器，请编写测试并确保通过以下检查：`RUN_SLOW=1 python -m pytest tests/models/{your_model_name}/test_tokenization_{your_model_name}.py`
   - CircleCI 不会运行时间较长的测试，但 GitHub Actions 每晚会运行所有测试！<br>

☐ 所有公共 method 必须具有信息文档（比如 [`modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py)）。<br>
☐ 由于代码仓库的体积正在迅速增长，请避免添加图像、视频和其他非文本文件，它们会增加仓库的负担。请使用 [`hf-internal-testing`](https://huggingface.co/hf-internal-testing) 等 Hub 仓库来托管这些文件，并通过 URL 引用它们。我们建议将与文档相关的图片放置在以下仓库中：[huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images)。你可以在这个数据集仓库上创建一个 PR，并请求 Hugging Face 成员进行合并。

要了解更多有关在 Pull request 上运行的检查的信息，请查看我们的 [检查 Pull Request](https://huggingface.co/docs/transformers/pr_checks) 指南。

### 测试

包含了广泛的测试套件来测试库的行为和一些示例。库测试可以在 [tests](https://github.com/huggingface/transformers/tree/main/tests) 文件夹中找到，示例测试可以在 [examples](https://github.com/huggingface/transformers/tree/main/examples) 文件夹中找到。

我们喜欢使用 `pytest` 和 `pytest-xdist`，因为它运行更快。在仓库的根目录，指定一个*子文件夹的路径或测试文件*来运行测试：

```bash
python -m pytest -n auto --dist=loadfile -s -v ./tests/models/my_new_model
```

同样地，在 `examples` 目录，指定一个*子文件夹的路径或测试文件* 来运行测试。例如，以下命令会测试 PyTorch `examples` 目录中的文本分类子文件夹：

```bash
pip install -r examples/xxx/requirements.txt  # 仅在第一次需要
python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/text-classification
```

实际上这就是我们的 `make test` 和 `make test-examples` 命令的实现方式（不包括 `pip install`）！

你也可以指定一个较小的测试集来仅测试特定功能。

默认情况下，会跳过时间较长的测试，但你可以将 `RUN_SLOW` 环境变量设置为 `yes` 来运行它们。这将下载以 GB 为单位的模型文件，所以确保你有足够的磁盘空间、良好的网络连接和足够的耐心！

<Tip warning={true}>

记得指定一个*子文件夹的路径或测试文件*来运行测试。否则你将会运行 `tests` 或 `examples` 文件夹中的所有测试，它会花费很长时间！

</Tip>

```bash
RUN_SLOW=yes python -m pytest -n auto --dist=loadfile -s -v ./tests/models/my_new_model
RUN_SLOW=yes python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/text-classification
```

和时间较长的测试一样，还有其他环境变量在测试过程中，在默认情况下是未启用的：
- `RUN_CUSTOM_TOKENIZERS`: 启用自定义分词器的测试。
- `RUN_PT_FLAX_CROSS_TESTS`: 启用 PyTorch + Flax 整合的测试。
- `RUN_PT_TF_CROSS_TESTS`: 启用 TensorFlow + PyTorch 整合的测试。

更多环境变量和额外信息可以在 [testing_utils.py](src/transformers/testing_utils.py) 中找到。

🤗 Transformers 只是使用 `pytest` 作为测试运行程序，但测试套件本身没用任何与 `pytest` 相关的功能。

这意味着完全支持 `unittest` 。以下是如何使用 `unittest` 运行测试的方法：

```bash
python -m unittest discover -s tests -t . -v
python -m unittest discover -s examples -t examples -v
```

### 风格指南

🤗 Transformers 的文档遵循 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)。请查看我们的 [文档编写指南](https://github.com/huggingface/transformers/tree/main/docs#writing-documentation---specification) 来获取更多信息。

### 在 Windows 上开发

在 Windows 上（除非你正在使用 [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/) 或 WSL），你需要配置 git 将 Windows 的 `CRLF` 行结束符转换为 Linux 的 `LF` 行结束符：

```bash
git config core.autocrlf input
```

在 Windows 上有一种方法可以运行 `make` 命令，那就是使用 MSYS2：

1. [下载 MSYS2](https://www.msys2.org/)，假设已经安装在 `C:\msys64`。
2. 从命令行打开 `C:\msys64\msys2.exe` （可以在 **开始** 菜单中找到）。
3. 在 shell 中运行： `pacman -Syu` ，并使用 `pacman -S make` 安装 `make`。
4. 把 `C:\msys64\usr\bin` 添加到你的 PATH 环境变量中。

现在你可以在任何终端（PowerShell、cmd.exe 等）中使用 `make` 命令了！ 🎉

### 将派生仓库与上游主仓库（Hugging Face 仓库）同步

更新派生仓库的主分支时，请按照以下步骤操作。这是为了避免向每个上游 PR 添加参考注释，同时避免向参与这些 PR 的开发人员发送不必要的通知。

1. 可以的话，请避免使用派生仓库上的分支和 PR 来与上游进行同步，而是直接合并到派生仓库的主分支。
2. 如果确实需要一个 PR，在检查你的分支后，请按照以下步骤操作：

   ```bash
   git checkout -b your-branch-for-syncing
   git pull --squash --no-commit upstream main
   git commit -m '<your message without GitHub references>'
   git push --set-upstream origin your-branch-for-syncing
   ```
