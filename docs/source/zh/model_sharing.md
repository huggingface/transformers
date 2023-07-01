<!-- 版权所有2022年HuggingFace团队保留所有权利。
根据Apache许可证第2.0版（“许可证”）获得许可；除非符合许可证，否则您不得使用此文件。您可以在以下位置获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或以书面形式达成协议，否则根据许可证分发的软件是在“按原样”基础上分发的，不附带任何形式的任何保证或条件。请参阅许可证以了解特定语言下的权限和限制。

⚠️ 特别提示：此文件是Markdown文件，但包含特定语法以适应我们的文档构建器（类似于MDX），在您的Markdown查看器中可能无法正确显示。
-->

# 共享模型

前两篇教程展示了如何使用PyTorch、Keras和🤗 Accelerate进行模型微调和分布式设置。下一步是与社区分享您的模型！在Hugging Face，我们相信公开分享知识和资源，以使人工智能为每个人所用。我们鼓励您考虑与社区分享您的模型，以帮助他人节省时间和资源。

在本教程中，您将学习两种在[模型中心](https://huggingface.co/models)上共享已训练或微调模型的方法：

- 通过编程方式将文件推送到模型中心。- 使用Web界面将文件拖放到模型中心。

<iframe width="560" height="315" src="https://www.youtube.com/embed/XvSGPZFEjDY" title="YouTube video player"
frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope;
picture-in-picture" allowfullscreen></iframe>

<Tip>

要与社区共享模型，您需要在[huggingface.co](https://huggingface.co/join)上拥有一个帐户。您还可以加入现有组织或创建一个新组织。
</Tip>

## 存储库功能 Repository features

模型中心上的每个存储库都像一个典型的GitHub存储库一样运作。我们的存储库提供版本控制、提交历史记录以及可视化差异的功能。

模型中心内置的版本控制基于git和[git-lfs](https://git-lfs.github.com/)。换句话说，您可以将一个模型视为一个存储库，从而实现更大的访问控制和可伸缩性。版本控制允许对模型进行*修订*，即使用提交哈希、标签或分支固定特定版本的方法。

因此，您可以使用`revision`参数加载特定的模型版本:
```py
>>> model = AutoModel.from_pretrained(
...     "julien-c/EsperBERTo-small", revision="v2.0.1"  # tag name, or branch name, or commit hash
... )
```

在存储库中还可以轻松编辑文件，您可以查看提交历史记录以及差异:
![vis_diff](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vis_diff.png)

## 设置

在将模型共享到模型中心之前，您需要使用Hugging Face凭据。如果您可以访问终端，请在安装了🤗 Transformers的虚拟环境中运行以下命令。这将在您的Hugging Face缓存文件夹（默认为`~/.cache/`）中存储您的访问令牌:
```bash
huggingface-cli login
```

如果您正在使用Jupyter或Colaboratory等笔记本，请确保已安装[`huggingface_hub`](https://huggingface.co/docs/hub/adding-a-library)库。此库允许您以编程方式与模型中心进行交互。

```bash
pip install huggingface_hub
```

然后使用`notebook_login`登录模型中心，并在此处[生成一个令牌](https://huggingface.co/settings/token)以进行登录:
```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 将模型转换为所有框架

为确保他人可以在使用不同框架的情况下使用您的模型，我们建议您转换并上传PyTorch和TensorFlow的检查点。如果跳过此步骤，用户仍然可以从不同框架加载您的模型，但速度会较慢，因为🤗 Transformers需要即时转换检查点。
将检查点转换为另一个框架非常简单。

确保已安装PyTorch和TensorFlow（安装说明请参见[此处](installation)），然后在另一个框架中找到适合您任务的特定模型。

<frameworkcontent>
<pt>
将`from_tf=True`指定为从TensorFlow转换为PyTorch的检查点:
```py
>>> pt_model = DistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_tf=True)
>>> pt_model.save_pretrained("path/to/awesome-name-you-picked")
```
</pt>
<tf>
将`from_pt=True`指定为从PyTorch转换为TensorFlow的检查点:
```py
>>> tf_model = TFDistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_pt=True)
```

然后，您可以保存具有新检查点的新的TensorFlow模型:
```py
>>> tf_model.save_pretrained("path/to/awesome-name-you-picked")
```
</tf>
<jax>

如果某个模型在Flax中可用，您还可以将检查点从PyTorch转换为Flax:

```py
>>> flax_model = FlaxDistilBertForSequenceClassification.from_pretrained(
...     "path/to/awesome-name-you-picked", from_pt=True
... )
```
</jax>
</frameworkcontent>

## 在训练过程中推送模型

<frameworkcontent>
<pt>
<Youtube id="Z1-XMy-GNLQ"/>

将模型推送到模型中心就像添加一个额外的参数或回调一样简单。请记住在[微调教程](training)中，[`TrainingArguments`]类是您指定超参数和其他训练选项的位置。其中一个训练选项包括直接将模型推送到模型中心。在[`TrainingArguments`]中设置`push_to_hub=True`:
```py
>>> training_args = TrainingArguments(output_dir="my-awesome-model", push_to_hub=True)
```

像往常一样将您的训练参数传递给[`Trainer`]:
```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

在微调模型后，调用[`~transformers.Trainer.push_to_hub`]将训练后的模型推送到模型中心。🤗 Transformers甚至会自动将训练超参数、训练结果和框架版本添加到您的模型卡中！
```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
使用[`PushToHubCallback`]将模型共享到模型中心。在[`PushToHubCallback`]函数中添加以下内容:
- 用于您的模型的输出目录。- 一个标记器。- `hub_model_id`，即您的模型中心用户名和模型名称。
```py
>>> from transformers import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="./your_model_save_path", tokenizer=tokenizer, hub_model_id="your-username/my-awesome-model"
... )
```

将回调添加到[`fit`](https://keras.io/api/models/model_training_apis/)中，🤗 Transformers将会将训练后的模型推送到模型中心:
```py
>>> model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3, callbacks=push_to_hub_callback)
```
</tf>
</frameworkcontent>

## 使用`push_to_hub`函数

您还可以直接在模型上调用`push_to_hub`将其上传到模型中心。
在`push_to_hub`中指定您的模型名称:
```py
>>> pt_model.push_to_hub("my-awesome-model")
```

这将在您的用户名下创建一个名为`my-awesome-model`的存储库。现在用户可以使用`from_pretrained`函数加载您的模型了:
```py
>>> from transformers import AutoModel

>>> model = AutoModel.from_pretrained("your_username/my-awesome-model")
```

如果您属于某个组织，并希望将您的模型推送到组织名称下，请将其添加到`repo_id`中:
```py
>>> pt_model.push_to_hub("my-awesome-org/my-awesome-model")
```

`push_to_hub`函数还可用于将其他文件添加到模型存储库。例如，将一个标记器添加到模型存储库:
```py
>>> tokenizer.push_to_hub("my-awesome-model")
```

或者，您可能希望将您的微调的PyTorch模型的TensorFlow版本添加到模型存储库中:
```py
>>> tf_model.push_to_hub("my-awesome-model")
```

现在当您导航到您的Hugging Face个人资料时，您应该会看到您新创建的模型存储库。单击**文件**选项卡将显示您上传到存储库的所有文件。

有关如何创建和上传文件到存储库的更多详细信息，请参阅[此处](https://huggingface.co/docs/hub/how-to-upstream)的模型中心文档。

## 使用Web界面上传

偏好无代码方法的用户可以通过模型中心的Web界面上传模型。访问[huggingface.co/new](https://huggingface.co/new)创建一个新的存储库:

![new_model_repo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/new_model_repo.png)

从这里，添加有关您的模型的一些信息:

- 选择存储库的**所有者**。这可以是您自己或您所属的任何组织。
- 为您的模型选择一个名称，这也将成为存储库的名称。
- 选择您的模型是公开的还是私有的。- 指定您的模型的许可使用情况。

现在点击 **文件** 选项卡，然后点击 **添加文件** 按钮将新文件上传到您的存储库。然后拖放一个文件进行上传并添加提交消息。

![upload_file](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/upload_file.png)

## 添加模型卡片
为了确保用户了解您的模型的能力、限制、潜在偏见和道德考虑，请在存储库中添加一个模型卡片。模型卡片在 `README.md` 文件中定义。您可以通过以下方式添加模型卡片：
* 手动创建并上传 `README.md` 文件。
* 在您的模型存储库中点击 **编辑模型卡片** 按钮。

查看 DistilBert 的 [模型卡片](https://huggingface.co/distilbert-base-uncased) 以获得模型卡片应包含的信息类型的良好示例。有关您可以在 `README.md` 文件中控制的其他选项的更多详细信息，例如模型的碳足迹或小部件示例，请参阅[此处](https://huggingface.co/docs/hub/models-cards)的文档。