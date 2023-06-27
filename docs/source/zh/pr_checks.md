<!---版权所有 2020 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可。除非符合许可证的规定，否则不得使用此文件。您可以在以下位置获取许可证的副本
    http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，否则按 "原样" 分发的软件不附带任何形式的明示或暗示的任何保证或条件。有关特定语言的权限和限制的许可证请参阅许可证。无论如何，此文件是 Markdown 文件，但包含我们 doc-builder 的特定语法（类似于 MDX），在您的 Markdown 查看器中可能无法正确渲染。
渲染完整。
-->

# 拉取请求的检查

当您在🤗 Transformers 上打开拉取请求时，将运行大量检查，以确保您添加的补丁不会破坏现有功能。这些检查有四种类型：
- 常规测试
- 文档构建
- 代码和文档样式
- 一般存储库一致性

在本文档中，我们将尝试解释这些各种检查以及它们背后的原因，以及如果其中一个检查在您的 PR 上失败，如何在本地调试它们。

请注意，理想情况下，它们要求您进行开发安装：
```bash
pip install transformers[dev]
```

或可编辑安装：
```bash
pip install -e .[dev]
```

在 Transformers 存储库内部。由于 Transformers 的可选依赖项数量已经增加了很多，您可能无法全部获取到它们。如果开发安装失败，请确保安装您正在使用的深度学习框架（PyTorch、TensorFlow 和/或 Flax），然后执行
```bash
pip install transformers[quality]
```

或可编辑安装：
```bash
pip install -e .[quality]
```


## 测试
以 `ci/circleci: run_tests_` 开头的所有作业都运行 Transformers 测试套件的各个部分。每个作业都侧重于特定环境中的库的一部分：例如，`ci/circleci: run_tests_pipelines_tf` 在仅安装 TensorFlow 的环境中运行 pipelines 测试。

请注意，为了避免在测试的模块中没有真正的变化时运行测试，每次只运行测试套件的一部分：运行一个实用程序来确定在 PR 之前和之后库中的差异（GitHub 在 "文件更改" 选项卡中显示的内容）并选择受该差异影响的测试。可以在本地运行此实用程序：
```bash
python utils/tests_fetcher.py
```

从 Transformers 存储库的根目录运行。它将：
1. 对差异中的每个文件检查更改是在代码中还是仅在注释或文档字符串中。只保留具有实际代码更改的文件。2. 构建一个内部映射，为库的源代码中的每个文件提供所有递归受其影响的文件。

如果模块 B 导入模块 A，则说模块 A 对模块 B 造成影响。对于递归影响，我们需要一个从模块 A 到模块 B 的模块链，其中每个模块都导入前一个模块。3. 将步骤 1 中收集的文件应用于此映射，从而得到受 PR 影响的模型文件列表。4. 将每个文件映射到其相应的测试文件，并获取要运行的测试列表。

在本地执行脚本时，您应该得到步骤 1、3 和 4 的结果打印出来，从而知道运行哪些测试。该脚本还将创建一个名为 `test_list.txt` 的文件，其中包含要运行的测试列表，您可以使用以下命令在本地运行它们：
```bash
python -m pytest -n 8 --dist=loadfile -rA -s $(cat test_list.txt)
```

以防万一有什么遗漏，还会每天运行完整的测试套件。

## 文档构建

`build_pr_documentation` 作业会构建和生成文档的预览，以确保在合并您的 PR 后一切正常。机器人将在您的 PR 中添加预览文档的链接。您对 PR 所做的任何更改都会自动更新到预览中。如果文档构建失败，请单击失败作业旁边的 **详细信息** 以查看出错的位置。通常，错误可能只是 `toctree` 中缺少文件。

如果您有兴趣在本地构建或预览文档，请查看文档文件夹中的 [`README.md`](https://github.com/huggingface/transformers/tree/main/docs)。

## 代码和文档样式

所有源文件、示例和测试都使用 `black` 和 `ruff` 进行代码格式化。我们还有一个自定义工具，负责格式化 docstrings 和 `rst` 文件（`utils/style_doc.py`），以及 Transformers `__init__.py` 文件中延迟导入的顺序（`utils/custom_init_isort.py`）。

所有这些都可以通过执行以下命令进行启动

```bash
make style
```

CI 检查这些是否已应用在 `ci/circleci: check_code_quality` 检查中。它还运行 `ruff`，它会基本检查您的代码，并在找到未定义变量或未使用变量时发出警告。要在本地运行该检查，请使用
```bash
make quality
```

这可能需要很长时间，因此要仅在当前分支修改的文件上运行相同的操作，请运行
```bash
make fixup
```

此命令还将运行存储库一致性的所有其他检查。让我们来看看它们。

## 存储库一致性

这将对所有测试进行分组，以确保您的 PR 使存储库保持良好状态，并由 `ci/circleci: check_repository_consistency` 检查执行。您可以通过执行以下命令在本地运行该检查：

```bash
make repo-consistency
```

这将检查以下内容：
- 所有添加到 init 的对象都已记录（由 `utils/check_repo.py` 执行）- 所有 `__init__.py` 文件在其两个部分中具有相同的内容（由 `utils/check_inits.py` 执行）- 从另一个模块复制的所有代码与原始代码一致（由 `utils/check_copies.py` 执行）- 所有配置类在其文档字符串中至少提到一个有效的检查点（由 `utils/check_config_docstrings.py` 执行）- 所有配置类仅包含在相应建模文件中使用的属性（由 `utils/check_config_attributes.py` 执行）- README 和文档索引的翻译与主 README 中的模型列表相同（由 `utils/check_copies.py` 执行）- 文档中的自动生成表格已更新（由 `utils/check_table.py` 执行）- 即使没有安装所有可选依赖项，库仍具有所有可用对象（由 `utils/check_dummies.py` 执行）

如果此检查失败，前两个项目需要手动修复，后四个项目可以通过运行以下命令自动修复

```bash
make fix-copies
```

其他检查涉及添加新模型的 PR，主要包括：
- 所有添加的模型都在自动映射中（由 `utils/check_repo.py` 执行）

<!-- TODO Sylvain,添加一个检查，确保实现了常用测试。-->

- 所有模型都经过了适当的测试（由 `utils/check_repo.py` 执行）
<!-- TODO Sylvain, 添加以下内容- 所有模型都添加到主 README 中，位于主文档内- 使用的所有检查点实际上存在于 Hub 上
-->