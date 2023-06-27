<!--版权所有2020年的HuggingFace小组。保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”）进行许可；除非符合许可证，否则您不得使用此文件。您可以在以下位置获得许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以 "按原样" 的方式分发，不提供任何明示或暗示的保证或条件。请参阅许可证以了解具体的语言权限和限制。an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
⚠️ 请注意，此文件是 Markdown 格式的，但包含我们的文档构建器的特定语法（类似于 MDX），可能无法在您的 Markdown 查看器中正确呈现。


-->

# Sharing custom models

🤗 Transformers 库旨在易于扩展。每个模型都是在给定子文件夹中完全编码的，没有抽象，因此您可以轻松复制建模文件并根据需要进行调整。

如果您正在编写全新的模型，从头开始可能更容易。在本教程中，我们将向您展示如何编写自定义模型及其配置，以便可以在 Transformers 内部使用它，并且可以与社区共享它（以及它所依赖的代码），以便任何人都可以使用它，即使它不在 🤗 Transformers 库中。

我们将以 ResNet 模型为例说明所有这些，将 ResNet 类包装到 [timm 库](https://github.com/rwightman/pytorch-image-models) 中的 [`PreTrainedModel`] 中。[timm library](https://github.com/rwightman/pytorch-image-models) into a [`PreTrainedModel`].

### 编写自定义配置

在深入研究模型之前，让我们首先编写其配置。模型的配置是一个包含构建模型所需的所有必要信息的对象。正如我们将在下一节中看到的，模型只能 will contain all the necessary information to build the model. As we will see in the next section, the model can only
接受 `config` 进行初始化，因此我们确实需要该对象尽可能完整。

在我们的示例中，我们将采用可能需要调整的 ResNet 类的一对参数。不同的配置将为我们提供不同类型的 ResNet。在检查其中一些参数的有效性之后，我们只需存储这些参数。
```python
from transformers import PretrainedConfig
from typing import List


class ResnetConfig(PretrainedConfig):
    model_type = "resnet"

    def __init__(
        self,
        block_type="bottleneck",
        layers: List[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        input_channels: int = 3,
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        avg_down: bool = False,
        **kwargs,
    ):
        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(f"`block_type` must be 'basic' or bottleneck', got {block_type}.")
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {stem_type}.")

        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down
        super().__init__(**kwargs)
```

编写自己的配置时需要记住的三个重要事项如下:
- 您必须继承自 `PretrainedConfig`，
- 您的 `PretrainedConfig` 的 `__init__` 必须接受任何 kwargs，- 这些 `kwargs` 需要传递给超类的 `__init__`。

继承是为了确保您获得🤗 Transformers 库的所有功能，而另外两个约束来自于 PretrainedConfig 具有比您设置的字段更多的字段。

当使用 `from_pretrained` 方法重新加载配置时，这些字段需要被您的配置接受，然后发送给超类。
为您的配置定义 `model_type`（这里是 `model_type="resnet"`）是可选的，除非您想将您的模型注册到自动类中（请参见最后一节）。

完成后，您可以像使用库中的任何其他模型配置一样轻松创建和保存您的配置。下面是如何创建一个 resnet50d 配置并保存它的示例：
```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d_config.save_pretrained("custom-resnet")
```

这将在 `custom-resnet` 文件夹内保存一个名为 `config.json` 的文件。然后，您可以使用 `from_pretrained` 方法重新加载配置：

```py
resnet50d_config = ResnetConfig.from_pretrained("custom-resnet")
```

您还可以使用 [`PretrainedConfig`] 类的任何其他方法，例如 [`~PretrainedConfig.push_to_hub`] 直接将配置上传到 Hub。directly upload your config to the Hub.

### 编写自定义模型
现在我们有了 ResNet 配置，我们可以继续编写模型。实际上，我们将编写两个模型：一个从图像批次中提取隐藏特征的模型（类似于 [`BertModel`]）
一个适用于图像分类的模型（类似于 [`BertForSequenceClassification`]）。
如前所述，我们只会编写一个松散的模型包装器，以保持本示例的简单性。在编写此类之前，我们需要建立块类型与实际块类之间的映射。然后，通过将所有内容传递给 `ResNet` 类来定义模型：
```py
from transformers import PreTrainedModel
from timm.models.resnet import BasicBlock, Bottleneck, ResNet
from .configuration_resnet import ResnetConfig


BLOCK_MAPPING = {"basic": BasicBlock, "bottleneck": Bottleneck}


class ResnetModel(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor):
        return self.model.forward_features(tensor)
```

对于将对图像进行分类的模型，我们只需更改前向方法：
```py
import torch


class ResnetModelForImageClassification(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        if labels is not None:
            loss = torch.nn.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
```

在这两种情况下，请注意我们继承自 `PreTrainedModel` 并使用 `config` 调用超类初始化（有点像编写常规的 `torch.nn.Module`）。

设置 `config_class` 的行不是必需的，除非您想将您的模型注册到自动类中（请参见最后一节）。
<Tip>
如果您的模型与库中的模型非常相似，您可以重用相同的配置作为此模型。
</Tip>

您的模型可以返回任何您想要的内容，但是像我们为 `ResnetModelForImageClassification` 所做的那样返回一个字典，当传递标签时，
包括损失，将使您的模型能够直接在 [`Trainer`] 类中使用。使用其他输出格式是可以的，只要您计划使用自己的训练循环或其他训练库。

现在我们有了模型类，让我们创建一个：
```py
resnet50d = ResnetModelForImageClassification(resnet50d_config)
```

同样，您可以使用 [`PreTrainedModel`] 的任何方法，例如 [`~PreTrainedModel.save_pretrained`] 或 [`~PreTrainedModel.push_to_hub`]。

我们将在下一节中使用第二种方法，查看如何将模型权重与模型的代码一起推送。但是首先，让我们在我们的模型内加载一些预训练权重。

在您自己的用例中，您可能会对自己的数据进行自定义模型训练。为了快速进行本教程，我们将使用 resnet50d 的预训练版本。

由于我们的模型只是它的包装器，因此很容易传输这些权重：

```py
import timm

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

现在让我们看看如何确保在执行 [`~PreTrainedModel.save_pretrained`] 或 [`~PreTrainedModel.push_to_hub`] 时保存模型的代码。

## 将代码发送到 Hub

<Tip  warning={true}>
此 API 是实验性的，可能在后续版本中会有一些轻微的破坏性更改。
</Tip>

首先，请确保您的模型在 `.py` 文件中完全定义。它可以依赖于某些其他文件的相对导入，只要所有文件都在同一个目录中（我们尚不支持子模块的此功能）。

对于我们的示例，我们将在当前工作目录的一个名为 `resnet_model` 的文件夹中定义一个 `modeling_resnet.py` 文件和一个 `configuration_resnet.py` 文件。配置文件包含 `ResnetConfig` 的代码，建模文件包含 `ResnetModel` 和 `ResnetModelForImageClassification` 的代码。

```
.
└── resnet_model
    ├── __init__.py
    ├── configuration_resnet.py
    └── modeling_resnet.py
```

`__init__.py` 文件可以为空，只是为了让 Python 检测到 `resnet_model` 可以用作模块。

<Tip  warning={true}>
如果从库中复制建模文件，则需要将文件顶部的所有相对导入替换为从 `transformers` 包导入。to import from the `transformers` package.

</Tip>


请注意，您可以重用（或子类化）现有的配置/模型。要与社区共享您的模型，请按照以下步骤操作：首先从新创建的文件中导入 ResNet 模型和配置：
```py
from resnet_model.configuration_resnet import ResnetConfig
from resnet_model.modeling_resnet import ResnetModel, ResnetModelForImageClassification
```

然后，当使用 `save_pretrained` 方法时，您需要告诉库您想要复制这些对象的代码文件，并将其正确注册到给定的 Auto 类中（特别是对于模型），只需运行以下命令：请注意，对于配置文件，无需指定自动类（它们只有一个自动类，即 [`AutoConfig`]），但模型不同。您的自定义模型可能适用于许多不同的任务，因此您需要指定哪个自动类是适合您的模型的正确类。

```py
ResnetConfig.register_for_auto_class()
ResnetModel.register_for_auto_class("AutoModel")
ResnetModelForImageClassification.register_for_auto_class("AutoModelForImageClassification")
```

接下来，让我们像之前一样创建配置和模型：现在，要将模型发送到 Hub，请确保已登录。您可以在终端运行以下命令：或者在笔记本中运行：
然后，您可以像这样将其推送到您自己的命名空间（或您是其成员的组织）：
```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d = ResnetModelForImageClassification(resnet50d_config)

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

除了建模权重和 json 格式的配置之外，这还复制了文件夹 `custom-resnet50d` 中的建模和配置 `.py` 文件，并将结果上传到了 Hub。您可以在此 [model repo](https://huggingface.co/sgugger/custom-resnet50d) 中查看结果。

```bash
huggingface-cli login
```

有关推送到 Hub 方法的更多信息，请参阅 [共享教程](model_sharing)。

```py
from huggingface_hub import notebook_login

notebook_login()
```

You can then push to your own namespace (or an organization you are a member of) like this:

```py
resnet50d.push_to_hub("custom-resnet50d")
```

除了模型权重和以 JSON 格式保存的配置之外，这个过程还复制了模型和配置的 `.py` 文件到 `custom-resnet50d` 文件夹，并将结果上传到了 Hub。您可以在[此模型存储库](https://huggingface.co/sgugger/custom-resnet50d)中查看结果。

有关 `push_to_hub` 方法的更多信息，请参阅[分享教程](model_sharing)。

## 使用带有自定义代码的模型

您可以使用任何带有其代码文件的配置、模型或分词器 (Tokenizer)，并使用自动类和 `from_pretrained` 方法。所有上传到 Hub 的文件和代码都会进行恶意软件扫描（有关详细信息，请参阅 [Hub 安全性](https://huggingface.co/docs/hub/security#malware-scanning) 文档），但您仍应查看模型代码和作者，以避免在您的设备上执行恶意代码。

设置 `trust_remote_code=True` 以使用带有自定义代码的模型：

```py
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained("sgugger/custom-resnet50d", trust_remote_code=True)
```

强烈建议通过传递提交哈希作为 `revision` 来确保模型的作者没有使用一些恶意的新代码（除非您完全信任模型的作者）。

```py
commit_hash = "ed94a7c6247d8aedce4647f00f20de6875b5b292"
model = AutoModelForImageClassification.from_pretrained(
    "sgugger/custom-resnet50d", trust_remote_code=True, revision=commit_hash
)
```

请注意，在 Hub 上浏览模型仓库的提交历史记录时，有一个按钮可以轻松复制任何提交的提交哈希。

## 将带有自定义代码的模型注册到自动类
如果您正在编写扩展🤗 Transformers 的库，可能希望将自己的模型添加到自动类中。这与将代码推送到 Hub 不同，因为用户需要导入您的库才能获取自定义模型（与自动从 Hub 下载模型代码相反）。

只要您的配置具有与现有模型类型不同的 `model_type` 属性，并且您的模型类具有正确的 `config_class` 属性，您只需像这样将它们添加到自动类中：

```py
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

AutoConfig.register("resnet", ResnetConfig)
AutoModel.register(ResnetConfig, ResnetModel)
AutoModelForImageClassification.register(ResnetConfig, ResnetModelForImageClassification)
```

请注意，将自定义配置注册到 [`AutoConfig`] 时使用的第一个参数必须与您的自定义配置的 `model_type` 匹配，将自定义模型注册到任何自动模型类时使用的第一个参数必须匹配那些模型的 `config_class`。
