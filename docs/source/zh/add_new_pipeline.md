<!--版权所有 2020 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可; 除非符合许可证，否则您不能使用此文件。您可以在
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，否则根据许可证分发的软件将按“按原样” BASIS，无论是明示还是暗示，都没有任何形式的保证或条件。有关许可证的详细信息请参阅
⚠️请注意，此文件是 Markdown 格式，但包含特定于我们的文档生成器（类似于 MDX）的语法，可能在您的 Markdown 查看器中无法正确呈现。
-->
# 如何创建自定义管道？
在本指南中，我们将看到如何创建自定义管道并在 [Hub](hf.co/models) 上共享它或将其添加到🤗 Transformers 库中。
首先，您需要决定管道将能够接受的原始输入。它可以是字符串，原始字节, 字典或任何可能是最可能的期望输入的内容。尽量保持这些输入尽可能纯粹的 Python，因为它使得兼容性更容易（甚至通过 JSON 通过其他语言）。这些将是管道的 `inputs`。（`preprocess`）。
然后定义 `outputs`。与 `inputs` 相同的策略。越简单越好。这些将是 `postprocess` 方法的输出。
首先通过继承基类 `Pipeline` 来实现 `preprocess`，`_forward`，`postprocess` 和 `_sanitize_parameters` 这四种方法。

```python
from transformers import Pipeline


class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        model_input = Tensor(inputs["input_ids"])
        return {"model_input": model_input}

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        outputs = self.model(**model_inputs)
        # Maybe {"logits": Tensor(...)}
        return outputs

    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"].softmax(-1)
        return best_class
```

此分解的结构是为了支持相对无缝支持 CPU / GPU，同时支持在不同线程上在 CPU 上进行预处理/后处理
`preprocess` 将采用最初定义的输入，并将其转换为可供模型使用的内容。它可能包含更多的信息，通常是一个 `Dict`。
`_forward` 是实现细节，不应直接调用。`forward` 是首选的调用方法，因为它包含了确保一切都在预期设备上工作的保护措施。如果有任何与真实模型相关联的内容属于 `_forward` 方法，其他任何内容都在预处理/后处理中。
`postprocess` 方法将采用 `_forward` 的输出，并将其转换为之前决定的最终输出。
`_sanitize_parameters` 存在的目的是允许用户在任何时候传递任何参数，无论是在初始化时 `pipeline（....，maybe_arg = 4）` 还是在调用时 `pipe = pipeline（....，maybe_arg = 4）`。
`_sanitize_parameters` 的返回值是将直接传递给 `preprocess` 的 3 个 kwargs 字典，`_forward` 和 `postprocess`。如果调用者没有使用任何额外的参数，则不填写任何内容。这允许在函数定义中保留默认参数，这总是更加“自然”。
在分类任务的后处理中，经典示例是 `top_k` 参数。
```python
>>> pipe = pipeline("my-new-task")
>>> pipe("This is a test")
[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}, {"label": "3-star", "score": 0.05}
{"label": "4-star", "score": 0.025}, {"label": "5-star", "score": 0.025}]

>>> pipe("This is a test", top_k=2)
[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}]
```

为了实现这一点，我们将使用默认参数 `5` 来更新我们的 `postprocess` 方法，并编辑 `_sanitize_parameters` 以允许此新参数。

```python
def postprocess(self, model_outputs, top_k=5):
    best_class = model_outputs["logits"].softmax(-1)
    # Add logic to handle top_k
    return best_class


def _sanitize_parameters(self, **kwargs):
    preprocess_kwargs = {}
    if "maybe_arg" in kwargs:
        preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]

    postprocess_kwargs = {}
    if "top_k" in kwargs:
        postprocess_kwargs["top_k"] = kwargs["top_k"]
    return preprocess_kwargs, {}, postprocess_kwargs
```

请尽量保持输入/输出非常简单，并且理想情况下是可 JSON 序列化的，因为这使得使用管道非常容易，而无需用户了解新类型的对象。支持许多不同类型的参数以便使用方便也是相对常见的（音频文件可以是文件名，URL 或纯字节）。


## 将其添加到支持的任务列表
要将您的 `new-task` 注册到支持的任务列表中，您必须将其添加到 `PIPELINE_REGISTRY` 中：
```python
from transformers.pipelines import PIPELINE_REGISTRY

PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
)
```

如果需要，您可以指定一个默认模型，其中应包含特定的修订版本（可以是分支名称或提交哈希，这里我们使用了 `"abcdef"`）以及类型：

```python
brand_new_bert.push_to_hub("brand_new_bert", model_type="brand_new_bert", revision="abcdef")
```

这将把 `brand_new_bert` 模型上传到模型中心，并指定了特定的修订版本和类型。
```python
PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
    default={"pt": ("user/awesome_model", "abcdef")},
    type="text",  # current support type: text, audio, image, multimodal
)
```

## 在 Hub 上共享您的管道
要在 Hub 上共享您的自定义管道，您只需将 `Pipeline` 子类的自定义代码保存在 python 文件中。例如，假设我们想要使用一个自定义的句子对分类管道，如下所示：
```py
import numpy as np

from transformers import Pipeline


def softmax(outputs):
    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class PairClassificationPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "second_text" in kwargs:
            preprocess_kwargs["second_text"] = kwargs["second_text"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text, second_text=None):
        return self.tokenizer(text, text_pair=second_text, return_tensors=self.framework)

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs):
        logits = model_outputs.logits[0].numpy()
        probabilities = softmax(logits)

        best_class = np.argmax(probabilities)
        label = self.model.config.id2label[best_class]
        score = probabilities[best_class].item()
        logits = logits.tolist()
        return {"label": label, "score": score, "logits": logits}
```

该实现与框架无关，并且适用于 PyTorch 和 TensorFlow 模型。如果我们将其保存在名为 `pair_classification.py` 的文件中，然后可以像这样导入并注册它：
```py
from pair_classification import PairClassificationPipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification

PIPELINE_REGISTRY.register_pipeline(
    "pair-classification",
    pipeline_class=PairClassificationPipeline,
    pt_model=AutoModelForSequenceClassification,
    tf_model=TFAutoModelForSequenceClassification,
)
```

完成后，我们可以将其与预训练模型一起使用。例如，`sgugger/finetuned-bert-mrpc` 已经在 MRPC 数据集上进行了微调，用于将句子对分类为释义或非释义。
```py
from transformers import pipeline

classifier = pipeline("pair-classification", model="sgugger/finetuned-bert-mrpc")
```

然后，我们可以使用 `Repository` 中的 `save_pretrained` 方法将其共享到 Hub 上：
```py
from huggingface_hub import Repository

repo = Repository("test-dynamic-pipeline", clone_from="{your_username}/test-dynamic-pipeline")
classifier.save_pretrained("test-dynamic-pipeline")
repo.push_to_hub()
```

这将与您定义 `PairClassificationPipeline` 的文件一起复制到文件夹 `"test-dynamic-pipeline"` 中，同时保存管道的模型和标记器，然后将所有内容推送到存储库 `{your_username}/test-dynamic-pipeline`。之后，任何人都可以使用它，只要他们提供选项 `trust_remote_code=True`：
```py
from transformers import pipeline

classifier = pipeline(model="{your_username}/test-dynamic-pipeline", trust_remote_code=True)
```

## 将管道添加到🤗 Transformers
如果您想将您的管道贡献给🤗 Transformers，您需要在 `pipelines` 子模块中添加一个新模块其中包含您的管道的代码，然后将其添加到 `pipelines/__init__.py` 中定义的任务列表中。
然后，您需要添加测试。创建一个名为 `tests/test_pipelines_MY_PIPELINE.py` 的新文件，与其他测试示例一起。
`run_pipeline_test` 函数将非常通用，并在每个可能的情况下在小型随机模型上运行由 `model_mapping` 和 `tf_model_mapping` 定义的体系结构。
这对于测试未来的兼容性非常重要，这意味着如果有人为 `XXXForQuestionAnswering` 添加了一个新模型，那么管道测试将尝试在其上运行。由于模型是随机的，所以无法检查实际值，这就是为什么有一个帮助程序 `ANY`，它将尝试匹配管道类型的输出。
您还需要实施 2 个（理想情况下为 4 个）测试。
- `test_small_model_pt`：为此管道定义 1 个小模型（结果无关紧要）  并测试管道输出。结果应与 `test_small_model_tf` 相同。- `test_small_model_tf`：为此管道定义 1 个小模型（结果无关紧要）  并测试管道输出。结果应与 `test_small_model_pt` 相同。- `test_large_model_pt`（`可选`）：在实际管道上测试结果应该  有意义。这些测试速度较慢，应标记为这样。在这里，目标是展示管道并  确保未来的发布中没有漂移。- `test_large_model_tf`（`可选`）：在实际管道上测试结果应该  有意义。这些测试速度较慢，应标记为这样。在这里，目标是展示管道并  确保未来的发布中没有漂移。