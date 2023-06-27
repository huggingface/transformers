<!--版权所有 2020 年 HuggingFace 团队保留所有权利。
根据 Apache License，Version 2.0（“许可证”）许可；您除非符合许可证的规定，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的保证或条件。请参阅许可证以了解特定语言许可权限和限制。注意：此文件是 Markdown 格式，但包含了我们的文档构建器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确
呈现。
-->

# 流程 Pipelines
流程是使用模型进行推理的一种简单有效的方法。

这些流程是抽象了库中大部分复杂代码的对象，提供了专用于多个任务的简单 API，包括命名实体识别、遮蔽语言建模、情感分析、特征提取和问答。请参阅 [任务概述](../task_summary) 以获取使用示例。

有两种流程抽象类别需要注意：
- [`pipeline`] 是封装了所有其他流程的最强大的对象。

- 针对音频、计算机视觉、自然语言处理和多模态任务提供了特定任务的流程。

## 流程抽象 The pipeline abstraction
*流程* 抽象是对所有其他可用流程的包装器。它与任何其他流程一样实例化。但是它能提供额外的便利。
简单调用一个项目：
```python
>>> pipe = pipeline("text-classification")
>>> pipe("This restaurant is awesome")
[{'label': 'POSITIVE', 'score': 0.9998743534088135}]
```

如果您想使用 [hub](https://huggingface.co) 上的特定模型，如果 hub 上的模型已经定义了它，您可以忽略任务：

```python
>>> pipe = pipeline(model="roberta-large-mnli")
>>> pipe("This restaurant is awesome")
[{'label': 'NEUTRAL', 'score': 0.7313136458396912}]
```

要在多个项目上调用流程，可以使用 *列表* 调用它。
```python
>>> pipe = pipeline("text-classification")
>>> pipe(["This restaurant is awesome", "This restaurant is awful"])
[{'label': 'POSITIVE', 'score': 0.9998743534088135},
 {'label': 'NEGATIVE', 'score': 0.9996669292449951}]
```

要遍历整个数据集，建议直接使用 `dataset`。这意味着您不需要一次性分配整个数据集，也不需要自己进行批处理。这应该与在 GPU 上自定义循环一样快。如果不是，请创建一个问题。GPU. 

```python
import datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0)
dataset = datasets.load_dataset("superb", name="asr", split="test")

# KeyDataset (only *pt*) will simply return the item in the dict returned by the dataset item
# as we're not interested in the *target* part of the dataset. For sentence pair use KeyPairDataset
for out in tqdm(pipe(KeyDataset(dataset, "file"))):
    print(out)
    # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
    # {"text": ....}
    # ....
```

为了方便使用，还可以使用生成器：

```python
from transformers import pipeline

pipe = pipeline("text-classification")


def data():
    while True:
        # This could come from a dataset, a database, a queue or HTTP request
        # in a server
        # Caveat: because this is iterative, you cannot use `num_workers > 1` variable
        # to use multiple threads to preprocess data. You can still have 1 thread that
        # does the preprocessing while the main runs the big inference
        yield "This is a test"


for out in pipe(data()):
    print(out)
    # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
    # {"text": ....}
    # ....
```

[[autodoc]] pipeline

## 流程批处理

所有流程都可以使用批处理。这将在流程使用其流式传输功能时起作用（因此当传递列表或 `Dataset` 或 `generator` 时）。
```python
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipe = pipeline("text-classification", device=0)
for out in pipe(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
    # [{'label': 'POSITIVE', 'score': 0.9998743534088135}]
    # Exactly the same output as before, but the content are passed
    # as batches to the model
```

<Tip warning={true}>

然而，这不是自动提高性能。它可能是 10 倍加速或 5 倍减速，这取决于硬件、数据和实际使用的模型。大多数情况下加速的示例：
</Tip>
```python
from transformers import pipeline
from torch.utils.data import Dataset
from tqdm.auto import tqdm

pipe = pipeline("text-classification", device=0)


class MyDataset(Dataset):
    def __len__(self):
        return 5000

    def __getitem__(self, i):
        return "This is a test"


dataset = MyDataset()

for batch_size in [1, 8, 64, 256]:
    print("-" * 30)
    print(f"Streaming batch_size={batch_size}")
    for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset)):
        pass
```

```
# On GTX 970
------------------------------
Streaming no batching
100%|██████████████████████████████████████████████████████████████████████| 5000/5000 [00:26<00:00, 187.52it/s]
------------------------------
Streaming batch_size=8
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:04<00:00, 1205.95it/s]
------------------------------
Streaming batch_size=64
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:02<00:00, 2478.24it/s]
------------------------------
Streaming batch_size=256
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:01<00:00, 2554.43it/s]
(diminishing returns, saturated the GPU)
```

大多数情况下减速的示例：
```python
class MyDataset(Dataset):
    def __len__(self):
        return 5000

    def __getitem__(self, i):
        if i % 64 == 0:
            n = 100
        else:
            n = 1
        return "This is a test" * n
```

这是一个与其他句子相比较长的句子。在这种情况下，**整个** 批次将需要 400 个标记长度，因此整个批次将是 [64，400] 而不是 [64，4]，导致严重减速。更糟糕的是，对于更大的批次，程序会直接崩溃。


```
------------------------------
Streaming no batching
100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:05<00:00, 183.69it/s]
------------------------------
Streaming batch_size=8
100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:03<00:00, 265.74it/s]
------------------------------
Streaming batch_size=64
100%|██████████████████████████████████████████████████████████████████████| 1000/1000 [00:26<00:00, 37.80it/s]
------------------------------
Streaming batch_size=256
  0%|                                                                                 | 0/1000 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/home/nicolas/src/transformers/test.py", line 42, in <module>
    for out in tqdm(pipe(dataset, batch_size=256), total=len(dataset)):
....
    q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
RuntimeError: CUDA out of memory. Tried to allocate 376.00 MiB (GPU 0; 3.95 GiB total capacity; 1.72 GiB already allocated; 354.88 MiB free; 2.46 GiB reserved in total by PyTorch)
```

对于这个问题，没有好的（一般性）解决方案，您的情况可能会有所不同。以下是一些经验法则：对于用户而言，一个经验法则是：

- **通过您的负载、硬件进行性能测量。测量、测量和不断测量。真实数据才是最重要的。**

- 如果您受到延迟限制（实时产品进行推理），请不要批处理- 如果您使用的是 CPU，请不要批处理。
- 如果您使用吞吐量（您希望在一堆静态数据上运行模型），使用 GPU，则：
- 如果您对 sequence_length 的大小没有概念（"自然" 数据），默认情况下不要批处理，进行测量并尝试适当添加  它，添加 OOM 检查以在失败时（如果您不控制 sequence_length，它将在某些时候失败）进行恢复。    control the sequence_length.)
- 如果您的 sequence_length 非常规则，则批处理更有可能非常有趣，请测量并不断优化。    it until you get OOMs.
- GPU 越大，批处理更有可能非常有趣- 一旦启用批处理，请确保您可以很好地处理 OOM。

## 流程分块批处理 Pipeline chunk batching
`zero-shot-classification` 和 `question-answering` 在某种程度上是具体的，因为单个输入可能会触发模型的多个前向传递。在正常情况下，这将导致 `batch_size` 参数的问题。
为了解决这个问题，这两个流程都是稍微特殊的，它们是 `ChunkPipeline` 而不是常规的 `Pipeline`。简而言之：regular `Pipeline`. 


```python
preprocessed = pipe.preprocess(inputs)
model_outputs = pipe.forward(preprocessed)
outputs = pipe.postprocess(model_outputs)
```

现在变成了：

```python
all_model_outputs = []
for preprocessed in pipe.preprocess(inputs):
    model_outputs = pipe.forward(preprocessed)
    all_model_outputs.append(model_outputs)
outputs = pipe.postprocess(all_model_outputs)
```

这对您的代码来说应该是非常透明的，因为流程的使用方式不变。

这只是一个简化的视图，因为流程可以自动处理批处理！这意味着您不必关心输入将触发多少次前向传递，您可以独立优化 `batch_size`，而不受输入的影响。前一节中的注意事项仍然适用。
## 流程自定义代码
如果您想要覆盖特定的流程。
请随时为您的任务创建一个问题，流程的目标是易于使用并支持大多数情况，因此 `transformers` 可能支持您的用例。

如果您要尝试简单的自定义代码，可以：

- 子类化您选择的流程
```python
class MyPipeline(TextClassificationPipeline):
    def postprocess():
        # Your code goes here
        scores = scores * 100
        # And here


my_pipeline = MyPipeline(model=model, tokenizer=tokenizer, ...)
# or if you use *pipeline* function, then:
my_pipeline = pipeline(model="xxxx", pipeline_class=MyPipeline)
```

这将使您能够编写所有所需的自定义代码。

## 实现流程

[实现新流程](../add_new_pipeline)

## 音频

音频任务可用的流程包括以下内容。

### 音频分类流程

[[autodoc]] 音频分类流程    - __call__    - all

### 自动语音识别流程

## Audio

Pipelines available for audio tasks include the following.

### AudioClassificationPipeline

[[autodoc]] AudioClassificationPipeline
    - __call__
    - all

### AutomaticSpeechRecognitionPipeline

[[autodoc]] AutomaticSpeechRecognitionPipeline
    - __call__
    - all

### ZeroShotAudioClassificationPipeline

[[autodoc]] ZeroShotAudioClassificationPipeline
    - __call__
    - all

## Computer vision

Pipelines available for computer vision tasks include the following.

### DepthEstimationPipeline
[[autodoc]] DepthEstimationPipeline
    - __call__
    - all

### ImageClassificationPipeline

[[autodoc]] ImageClassificationPipeline
    - __call__
    - all

### ImageSegmentationPipeline

[[autodoc]] ImageSegmentationPipeline
    - __call__
    - all

### ObjectDetectionPipeline

[[autodoc]] ObjectDetectionPipeline
    - __call__
    - all

### VideoClassificationPipeline

[[autodoc]] VideoClassificationPipeline
    - __call__
    - all

### ZeroShotImageClassificationPipeline

[[autodoc]] ZeroShotImageClassificationPipeline
    - __call__
    - all

### ZeroShotObjectDetectionPipeline

[[autodoc]] ZeroShotObjectDetectionPipeline
    - __call__
    - all

## Natural Language Processing

Pipelines available for natural language processing tasks include the following.

### ConversationalPipeline

[[autodoc]] Conversation

[[autodoc]] ConversationalPipeline
    - __call__
    - all

### FillMaskPipeline

[[autodoc]] FillMaskPipeline
    - __call__
    - all

### NerPipeline

[[autodoc]] NerPipeline

See [`TokenClassificationPipeline`] for all details.

### QuestionAnsweringPipeline

[[autodoc]] QuestionAnsweringPipeline
    - __call__
    - all

### SummarizationPipeline

[[autodoc]] SummarizationPipeline
    - __call__
    - all

### TableQuestionAnsweringPipeline

[[autodoc]] TableQuestionAnsweringPipeline
    - __call__

### TextClassificationPipeline

[[autodoc]] TextClassificationPipeline
    - __call__
    - all

### TextGenerationPipeline

[[autodoc]] TextGenerationPipeline
    - __call__
    - all

### Text2TextGenerationPipeline

[[autodoc]] Text2TextGenerationPipeline
    - __call__
    - all

### TokenClassificationPipeline

[[autodoc]] TokenClassificationPipeline
    - __call__
    - all

### TranslationPipeline

[[autodoc]] TranslationPipeline
    - __call__
    - all

### ZeroShotClassificationPipeline

[[autodoc]] ZeroShotClassificationPipeline
    - __call__
    - all

## Multimodal

Pipelines available for multimodal tasks include the following.

### DocumentQuestionAnsweringPipeline

[[autodoc]] DocumentQuestionAnsweringPipeline
    - __call__
    - all

### FeatureExtractionPipeline

[[autodoc]] FeatureExtractionPipeline
    - __call__
    - all

### ImageToTextPipeline

[[autodoc]] ImageToTextPipeline
    - __call__
    - all

### VisualQuestionAnsweringPipeline

[[autodoc]] VisualQuestionAnsweringPipeline
    - __call__
    - all

## Parent class: `Pipeline`

[[autodoc]] Pipeline
