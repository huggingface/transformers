<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可；除非符合许可证的规定，否则您不得使用此文件。您可以在下面获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”基础分发的，不附带任何明示或暗示的担保或条件。有关详细信息，请参阅许可证
⚠️ 请注意，此文件是 Markdown 格式，但包含特定于我们的文档构建器（类似于 MDX）的语法，可能无法在您的 Markdown 查看器中正确渲染。

-->

# 如何将🤗 Transformers 模型转换为 TensorFlow？

在使用🤗 Transformers 时，有多个可用的框架可以发挥其优势，以便在设计应用程序时充分利用它们的优势，但这意味着必须对每个模型进行兼容性添加。

好消息是，将现有模型添加到 TensorFlow 中比 [从头开始添加新模型](add_new_model) 要简单。无论您是希望更深入地了解大型 TensorFlow 模型，做出重大的开源贡献，还是为您选择的模型启用 TensorFlow，本指南都适用于您。

本指南旨在使您作为我们社区的一员，能够在最少的 Hugging Face 团队监督下，为🤗 Transformers 贡献 TensorFlow 模型权重和/或架构。

编写一个新模型并不是一件小事，但希望本指南能使它变得不那么曲折，而更像在公园里散步一样轻松愉快。利用我们的集体经验对于使这个过程变得更加容易至关重要，因此我们强烈建议您对本指南提出改进意见！

在深入研究之前，建议您查看以下资源，如果您是第一次使用🤗 Transformers：

- [🤗 Transformers 的总体概述](add_new_model#general-overview-of-transformers)
- [Hugging Face 的 TensorFlow 哲学](https://huggingface.co/blog/tensorflow-philosophy)

在本指南的其余部分，您将了解添加新的 TensorFlow 模型架构所需的内容，将 PyTorch 转换为 TensorFlow 模型权重的过程，以及如何有效地调试不同的 ML 框架之间的不匹配。让我们开始吧！

<Tip>

您是否不确定您希望使用的模型是否已经有相应的 TensorFlow 架构？
&nbsp;

通过检查所选择模型的 `config.json` 中的 `model_type` 字段来检查（[示例](https://huggingface.co/bert-base-uncased/blob/main/config.json#L14)）。

如果🤗 Transformers 中的相应模型文件夹
中有一个以 "modeling_tf" 开头的文件，那么它意味着它有一个相应的 TensorFlow 架构([example](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert)).

</Tip>

## 逐步指南：添加 TensorFlow 模型架构代码


有很多方法可以设计一个大型模型架构，并且有多种实现这种设计的方式。然而，正如我们在 [🤗 Transformers 的概述](add_new_model#general-overview-of-transformers) 中提到的那样（[示例](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert)），我们可以告诉您一些关于添加 TensorFlow 模型的重要事项:

- 不要重复造轮子！往往情况下，您至少应该检查两个参考实现：您正在实现的模型的 PyTorch 等效版本和同类问题的其他 TensorFlow 模型。
- 优秀的模型实现经得住时间的检验。这不是因为代码很漂亮，而是因为代码清晰、易于调试和构建。如果您通过复制其他 TensorFlow 模型中的相同模式并尽量减少与 PyTorch 实现的不匹配，使维护者的生活变得轻松，您可以确保您的贡献将长寿。
- 在遇到困难时寻求帮助！🤗 Transformers 团队在这里帮助您，我们可能已经找到了您面临的相同问题的解决办法。

以下是添加 TensorFlow 模型架构所需的步骤概述：
1. 选择要转换的模型
2. 设置 transformers 开发环境。
3. （可选）了解理论方面和现有实现的内容。
4. 实现模型架构
5. 实现模型测试
6. 提交拉取请求
7. （可选）构建演示并与世界分享

### 1.-3. 准备您的模型贡献

**1. 选择要转换的模型**

让我们从基础知识开始：您需要了解您想要转换的架构。如果您还没有针对特定的架构，请向🤗 Transformers 团队寻求建议，这是最大化影响力的好方法-我们将指导您选择在 TensorFlow 方面缺失的最重要的架构。如果您想要与 TensorFlow 一起使用的特定模型在🤗 Transformers 中已经拥有 TensorFlow 架构的实现，只是缺少权重，请直接跳转到本页面的 [权重转换部分](#adding-tensorflow-weights-to-hub)。

为了简单起见，本指南的其余部分假设您已决定使用 TensorFlow 版本的 *BrandNewBert*（与 [指南](add_new_model) 中添加新模型的示例相同）进行贡献。

<Tip>

在开始工作于 TensorFlow 模型架构之前，请仔细检查是否有正在进行的相关工作。您可以在 [pull request GitHub 页面](https://github.com/huggingface/transformers/pulls?q=is%3Apr) 上搜索“BrandNewBert”以确认没有与 TensorFlow 相关的 pull request。

</Tip>


**2. 准备 transformers 开发环境**

选择了模型架构后，打开一个草案 PR 表示您打算在其中进行工作。按照以下说明设置您的环境并打开一个草稿 PR。

1. 单击存储库页面上的“Fork”按钮，将该存储库 fork 到您的 GitHub 用户帐户下。这将在您的 GitHub 用户帐户下创建代码副本。

2. 将您的 `transformers` fork 克隆到本地磁盘，并将基本存储库添加为远程存储库：
```bash
git clone https://github.com/[your Github handle]/transformers.git
cd transformers
git remote add upstream https://github.com/huggingface/transformers.git
```

3. 设置开发环境，例如运行以下命令：
```bash
python -m venv .env
source .env/bin/activate
pip install -e ".[dev]"
```

根据您的操作系统，并且由于 Transformers 的可选依赖项数量不断增加，您可能会在此命令中遇到错误。如果是这种情况，请确保安装了 TensorFlow，然后执行以下操作：
```bash
pip install -e ".[quality]"
```

**注意：** 您不需要安装 CUDA。使新模型在 CPU 上工作就足够了。
4. 从主分支创建一个具有描述性名称的分支
```bash
git checkout -b add_tf_brand_new_bert
```

5. 获取并将当前主分支合并到您的分支上
```bash
git fetch upstream
git rebase upstream/main
```

6. 在 `transformers/src/models/brandnewbert/` 中添加一个名为 `modeling_tf_brandnewbert.py` 的空的 `.py` 文件。这将成为您的 TensorFlow 模型文件。
7. 使用以下命令将更改推送到您的账户：
```bash
git add .
git commit -m "initial commit"
git push -u origin add_tf_brand_new_bert
```

8. 一旦您满意了，转到 GitHub 上您 fork 的网页。点击“Pull request”。确保添加 Hugging Face 团队成员的 GitHub 账号作为审查者，这样 Hugging Face 团队就会收到有关未来更改的通知。  
9. 单击 GitHub 拉取请求网页右侧的“Convert to draft”将 PR 更改为草稿状态。

现在您已经在🤗 Transformers 中设置了一个开发环境，以将 *BrandNewBert* 移植到 TensorFlow。




**3. （可选）了解理论方面和现有实现**
如果有关 *BrandNewBert* 的论文存在，您应该花些时间阅读。可能有些部分的论文很难理解。如果是这样，没关系-不用担心！目标不是深入理解论文，而是提取在使用 TensorFlow 有效重新实现模型所需的必要信息。话虽如此，您不必花太多时间在理论方面，而是更多关注实践方面，即现有模型文档页面（例如 [model docs for BERT](model_doc/bert)）。


一旦掌握了要实现的模型的基础知识，了解现有的实现是很重要的。
这是确认工作中的实现是否符合模型预期的好机会，也可以预见 TensorFlow 方面的技术挑战。

您可能会因刚吸收的大量信息而感到不知所措。在这个阶段，您不必要求完全理解模型的所有方面。尽管如此，我们强烈建议您在我们的 [论坛](https://discuss.huggingface.co/) 上解决任何紧迫的问题。
### 4. 模型实现

现在是时候开始编码了。我们建议从 PyTorch 文件本身开始：将 `src/transformers/models/brand_new_bert/` 目录下 `modeling_brand_new_bert.py` 文件的内容复制到 `modeling_tf_brand_new_bert.py` 文件中。本节的目标是修改文件并更新🤗 Transformers 的导入结构，以便您可以成功导入 `TFBrandNewBert` 和 `TFBrandNewBert.from_pretrained(model_repo, from_pt=True)`，并成功加载一个可用的 TensorFlow *BrandNewBert* 模型。

遗憾的是，没有将 PyTorch 模型转换为 TensorFlow 的准则。但是，您可以遵循我们的一些建议，使过程尽可能顺利：
- 将所有类的名称前加上 `TF`（例如，`BrandNewBert` 变为 `TFBrandNewBert`）。
- 大多数 PyTorch 操作都有直接对应的 TensorFlow 替代品。例如，`torch.nn.Linear` 对应于 `tf.keras.layers.Dense`，`torch.nn.Dropout` 对应于 `tf.keras.layers.Dropout` 等等。如果您对特定操作不确定，可以参考 [TensorFlow 文档](https://www.tensorflow.org/api_docs/python/tf) 或 [PyTorch 文档](https://pytorch.org/docs/stable/)。
- 在🤗 Transformers 代码库中查找模式。如果遇到某个操作没有直接替代的情况，很有可能其他人已经遇到过同样的问题。
- 默认情况下，保持与 PyTorch 中相同的变量名称和结构。这将使调试、跟踪问题和之后添加修复更加容易。
- 一些层在每个框架中的默认值不同。一个显著的例子是批标准化层的 epsilon（在 [PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d) 中为 `1e-5`，在 [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) 中为 `1e-3`）。一定要仔细检查文档！
- PyTorch 的 `nn.Parameter` 变量通常需要在 TF Layer 的 `build()` 函数内进行初始化。请参考以下示例：[PyTorch](https://github.com/huggingface/transformers/blob/655f72a6896c0533b1bdee519ed65a059c2425ac/src/transformers/models/vit_mae/modeling_vit_mae.py#L212) / [TensorFlow](https://github.com/huggingface/transformers/blob/655f72a6896c0533b1bdee519ed65a059c2425ac/src/transformers/models/vit_mae/modeling_tf_vit_mae.py#L220)
- 如果 PyTorch 模型的函数顶部有 `#copied from ...`，很有可能您的 TensorFlow 模型也可以从它被复制的架构中借用该函数，假设该架构具有 TensorFlow 架构。
- 在 TensorFlow 函数中正确设置 `name` 属性对于执行 `from_pt=True` 的权重交叉加载非常重要。`name` 几乎总是与 PyTorch 代码中相应变量的名称相同。如果 `name` 没有正确设置，加载模型权重时将看到错误消息。
- 基础模型类 `BrandNewBertModel` 的逻辑实际上位于 `TFBrandNewBertMainLayer` 中，这是一个 Keras 层子类（[示例](https://github.com/huggingface/transformers/blob/4fd32a1f499e45f009c2c0dea4d81c321cba7e02/src/transformers/models/bert/modeling_tf_bert.py#L719)）。`TFBrandNewBertModel` 只是该层的一个包装。
- Keras 模型需要进行构建才能加载预训练权重。因此，`TFBrandNewBertPreTrainedModel` 将需要保存模型输入的示例 `dummy_inputs`（[示例](https://github.com/huggingface/transformers/blob/4fd32a1f499e45f009c2c0dea4d81c321cba7e02/src/transformers/models/bert/modeling_tf_bert.py#L916)）。
- 如果遇到困难，请寻求帮助-我们在这里帮助您！🤗
除了模型文件本身之外，您还需要添加模型类和相关文档页面的指针。您可以完全按照其他 PR 的模式完成此部分（[示例](https://github.com/huggingface/transformers/pull/18020/files)）。以下是所需手动更改的列表：
- 在 `src/transformers/__init__.py` 中包含 *BrandNewBert* 的所有公共类
- 将 *BrandNewBert* 的类添加到 `src/transformers/models/auto/modeling_tf_auto.py` 中对应的 Auto 类中
- 在 `utils/documentation_tests.txt` 中将建模文件包含在文档测试文件列表中
- 在 `src/transformers/utils/dummy_tf_objects.py` 中添加与 *BrandNewBert* 相关的延迟加载类
- 更新 `src/transformers/models/brand_new_bert/__init__.py` 中公共类的导入结构
- 在 `docs/source/en/model_doc/brand_new_bert.md` 中添加 *BrandNewBert* 的公共方法的文档指针
- 在 `docs/source/en/model_doc/brand_new_bert.md` 中将自己添加到 *BrandNewBert* 的贡献者列表中
- 最后，在 `docs/source/en/index.md` 中 *BrandNewBert* 的 TensorFlow 栏中添加一个绿色的✅
当您对您的实现感到满意时，运行以下检查列表以确认您的模型架构已经准备就绪：
1. 所有在训练时行为不同的层（例如 Dropout）都使用了一个 `training` 参数进行调用，并且该参数从顶层类一直传递下去
2. 尽可能使用了 `#copied from ...`
3. `TFBrandNewBertMainLayer` 和使用它的所有类的 `call` 函数都使用 `@unpack_inputs` 装饰
4. `TFBrandNewBertMainLayer` 使用 `@keras_serializable` 进行了装饰
5. 可以使用 `TFBrandNewBert.from_pretrained(model_repo, from_pt=True)` 从 PyTorch 权重加载 TensorFlow 模型
6. 您可以使用预期的输入格式调用 TensorFlow 模型
### 5. 添加模型测试
太棒了，您已经实现了一个 TensorFlow 模型！现在是时候添加测试，以确保您的模型的行为符合预期。
与上一节一样，我们建议您首先将 `tests/models/brand_new_bert/` 目录中的 `test_modeling_brand_new_bert.py` 文件复制到 `test_modeling_tf_brand_new_bert.py`，然后进行必要的 TensorFlow 替换。目前，在所有的 `.from_pretrained()` 调用中，应该使用 `from_pt=True` 标志来加载现有的 PyTorch 权重。预期。然后，在所有的 `.from_pretrained()` 调用中，应该使用 `from_pt=True` 标志来加载现有的 PyTorch 权重。
然后，在所有的 `.from_pretrained()` 调用中，应该使用 `from_pt=True` 标志来加载现有的 PyTorch 权重。现有的 PyTorch 权重。

完成之后，现在是真相的时刻：运行测试！ 😬
```bash
NVIDIA_TF32_OVERRIDE=0 RUN_SLOW=1 RUN_PT_TF_CROSS_TESTS=1 \
py.test -vv tests/models/brand_new_bert/test_modeling_tf_brand_new_bert.py
```

最有可能的结果是您会看到一堆错误。不用担心，这是正常的！调试机器学习模型非常困难，成功的关键是耐心（和 `breakpoint()`）。根据我们的经验，最困难的问题是不同机器学习框架之间的细微差异，我们在本指南的末尾给出了一些建议。在其他情况下，一般的测试可能无法直接适用于您的模型，这种情况下我们建议在模型测试类级别进行覆盖。无论出现什么问题，如果您陷入困境，请毫不犹豫地在您的草稿拉取请求中寻求帮助。如果您陷入困境，请毫不犹豫地在您的草稿拉取请求中寻求帮助。在其他情况下，一般的测试可能无法直接适用于您的模型，这种情况下我们建议在模型测试类级别进行覆盖。无论出现什么问题，如果您陷入困境，请毫不犹豫地在您的草稿拉取请求中寻求帮助。如果您陷入困境，请毫不犹豫地在您的草稿拉取请求中寻求帮助。如果您陷入困境，请毫不犹豫地在您的草稿拉取请求中寻求帮助。

当所有的测试都通过时，恭喜您，您的模型几乎已经准备好添加到🤗 Transformers 库中了！🎉

### 6.-7. 确保每个人都能使用您的模型

**6. 提交拉取请求**

完成实现和测试后，现在是提交拉取请求的时候了。在推送代码之前，请运行我们的代码格式化工具 `make fixup` 🪄。这将自动修复任何格式问题，否则会导致自动检查失败。

现在是时候将您的草稿拉取请求转换为真正的拉取请求了。为此，点击 "Ready for review" 按钮，并将 Joao（`@gante`）和 Matt（`@Rocketknight1`）添加为审核者。模型拉取请求需要至少 3 个审核者，但他们将负责为您的模型找到适当的其他审核者。

当所有审核者对您的 PR 状态满意时，最后一个操作是在 `.from_pretrained()` 调用中删除 `from_pt=True` 标志。由于没有 TensorFlow 权重，您将需要添加它们！请参考下面的说明。

最后，当 TensorFlow 权重合并后，您获得了至少 3 个审核者的批准，并且所有 CI 检查都通过时，最后再次在本地检查测试。

```bash
NVIDIA_TF32_OVERRIDE=0 RUN_SLOW=1 RUN_PT_TF_CROSS_TESTS=1 \
py.test -vv tests/models/brand_new_bert/test_modeling_tf_brand_new_bert.py
```

### 7. （可选）构建演示并与世界共享
开源的最大困难之一是发现性。其他用户如何了解您出色的 TensorFlow 贡献的存在？当然是通过适当的沟通！📣
有两种主要的方式与社区共享您的模型：
- 构建演示。这包括 Gradio 演示、笔记本和其他有趣的方式来展示您的模型。我们强烈建议您在我们的 [社区驱动的演示](https://huggingface.co/docs/transformers/community) 中添加一个笔记本。
- 在 Twitter 和 LinkedIn 等社交媒体上分享您的故事。您应该为自己的工作感到自豪，并与社区分享您的成就-您的模型现在可以被全世界的数千名工程师和研究人员使用。我们将很乐意转发您的帖子，并帮助您与社区分享您的工作。


## 向🤗 Hub 添加 TensorFlow 权重

假设🤗 Transformers 中提供了 TensorFlow 模型架构，将 PyTorch 权重转换为 TensorFlow 权重非常简单！具体操作如下：

请按以下步骤操作：
1. 确保您已在终端中登录到您的 Hugging Face 帐户。您可以使用命令 `huggingface-cli login` 登录（您可以在 [此处](https://huggingface.co/settings/tokens) 找到您的访问令牌）。
2. 运行 `transformers-cli pt-to-tf --model-name foo/bar` 命令，其中 `foo/bar` 是要转换的包含 PyTorch 权重的模型存储库的名称。
3. 在🤗 Hub PR 中标记 `@joaogante` 和 `@Rocketknight1`，这是刚刚创建的命令的 PR。

就是这样！🎉

## 调试跨机器学习框架的不匹配问题🐛

在添加新架构或为现有架构创建 TensorFlow 权重时，您可能会遇到关于 PyTorch 和 TensorFlow 之间不匹配的错误。

您甚至可能决定打开两个框架的模型架构代码，发现它们看起来是相同的。到底发生了什么？🤔

首先，让我们谈谈为什么理解这些不匹配问题很重要。许多社区成员将直接使用🤗 Transformers 模型，并相信我们的模型的行为符合预期。

当两个框架之间存在较大的不匹配时，这意味着模型至少在一个框架中没有遵循参考实现。这可能会导致静默失败，即模型运行但性能较差。这比完全无法运行的模型更糟糕！

因此，我们的目标是在模型的所有阶段中，框架之间的不匹配小于 `1e-5`。

与其他数值问题一样，细节决定成败。在任何注重细节的工艺中，关键因素是耐心。以下是我们建议在遇到此类问题时使用的工作流程：
1. 找出不匹配问题的源头。您要转换的模型可能具有几乎相同的内部变量，直到某个点为止。在两个框架的架构中放置 `breakpoint()` 语句，以自上而下的方式比较数字变量的值，直到找到问题的源头。
2. 现在，您已经找到了问题的源头，请与🤗 Transformers 团队联系。我们可能之前遇到了类似的问题，并且可以立即提供解决方案。作为备选方案，请浏览像 StackOverflow 和 GitHub 问题这样的热门页面。
3. 如果没有解决方案，这意味着您需要进一步深入。好消息是您已经找到了问题的源头，因此可以专注于有问题的指令，将模型的其余部分抽象出来！坏消息是您将不得不深入到该指令的源代码实现中。在某些情况下，您可能会发现参考实现存在问题-不要不敢提出问题。

在与🤗 Transformers 团队讨论后，我们可能会发现修复不匹配是不可行的。当输出层中的不匹配非常小（但在隐藏状态中可能很大）时，我们认为这是较小的问题可能忽略它，而选择分发模型。上面提到的 `pt-to-tf` CLI 工具有一个 `--max-error` 标志来在权重转换时覆盖错误消息。