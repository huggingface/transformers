<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Pipelines

pipelines是使用模型进行推理的一种简单方法。这些pipelines是抽象了库中大部分复杂代码的对象，提供了一个专用于多个任务的简单API，包括专名识别、掩码语言建模、情感分析、特征提取和问答等。请参阅[任务摘要](../task_summary)以获取使用示例。

有两种pipelines抽象类需要注意：

- [`pipeline`]，它是封装所有其他pipelines的最强大的对象。
- 针对特定任务pipelines，适用于[音频](#audio)、[计算机视觉](#computer-vision)、[自然语言处理](#natural-language-processing)和[多模态](#multimodal)任务。

## pipeline抽象类

*pipeline*抽象类是对所有其他可用pipeline的封装。它可以像任何其他pipeline一样实例化，但进一步提供额外的便利性。

简单调用一个项目：


```python
>>> pipe = pipeline("text-classification")
>>> pipe("This restaurant is awesome")
[{'label': 'POSITIVE', 'score': 0.9998743534088135}]
```

如果您想使用 [hub](https://huggingface.co) 上的特定模型，可以忽略任务，如果hub上的模型已经定义了该任务：

```python
>>> pipe = pipeline(model="FacebookAI/roberta-large-mnli")
>>> pipe("This restaurant is awesome")
[{'label': 'NEUTRAL', 'score': 0.7313136458396912}]
```

要在多个项目上调用pipeline，可以使用*列表*调用它。

```python
>>> pipe = pipeline("text-classification")
>>> pipe(["This restaurant is awesome", "This restaurant is awful"])
[{'label': 'POSITIVE', 'score': 0.9998743534088135},
 {'label': 'NEGATIVE', 'score': 0.9996669292449951}]
```

为了遍历整个数据集，建议直接使用 `dataset`。这意味着您不需要一次性分配整个数据集，也不需要自己进行批处理。这应该与GPU上的自定义循环一样快。如果不是，请随时提出issue。

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

为了方便使用，也可以使用生成器：


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

## Pipeline batching

所有pipeline都可以使用批处理。这将在pipeline使用其流处理功能时起作用（即传递列表或 `Dataset` 或 `generator` 时）。

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

然而，这并不自动意味着性能提升。它可能是一个10倍的加速或5倍的减速，具体取决于硬件、数据和实际使用的模型。

主要是加速的示例：

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

主要是减速的示例：

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

与其他句子相比，这是一个非常长的句子。在这种情况下，**整个**批次将需要400个tokens的长度，因此整个批次将是 [64, 400] 而不是 [64, 4]，从而导致较大的减速。更糟糕的是，在更大的批次上，程序会崩溃。

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

对于这个问题，没有好的（通用）解决方案，效果可能因您的用例而异。经验法则如下：

对于用户，一个经验法则是：

- **使用硬件测量负载性能。测量、测量、再测量。真实的数字是唯一的方法。**
- 如果受到延迟的限制（进行推理的实时产品），不要进行批处理。
- 如果使用CPU，不要进行批处理。
- 如果您在GPU上处理的是吞吐量（您希望在大量静态数据上运行模型），则：
  - 如果对序列长度的大小没有概念（"自然"数据），默认情况下不要进行批处理，进行测试并尝试逐渐添加，添加OOM检查以在失败时恢复（如果您不能控制序列长度，它将在某些时候失败）。
  - 如果您的序列长度非常规律，那么批处理更有可能非常有趣，进行测试并推动它，直到出现OOM。
  - GPU越大，批处理越有可能变得更有趣
- 一旦启用批处理，确保能够很好地处理OOM。

## Pipeline chunk batching

`zero-shot-classification` 和 `question-answering` 在某种意义上稍微特殊，因为单个输入可能会导致模型的多次前向传递。在正常情况下，这将导致 `batch_size` 参数的问题。

为了规避这个问题，这两个pipeline都有点特殊，它们是 `ChunkPipeline` 而不是常规的 `Pipeline`。简而言之：


```python
preprocessed = pipe.preprocess(inputs)
model_outputs = pipe.forward(preprocessed)
outputs = pipe.postprocess(model_outputs)
```

现在变成：


```python
all_model_outputs = []
for preprocessed in pipe.preprocess(inputs):
    model_outputs = pipe.forward(preprocessed)
    all_model_outputs.append(model_outputs)
outputs = pipe.postprocess(all_model_outputs)
```

这对您的代码应该是非常直观的，因为pipeline的使用方式是相同的。

这是一个简化的视图，因为Pipeline可以自动处理批次！这意味着您不必担心您的输入实际上会触发多少次前向传递，您可以独立于输入优化 `batch_size`。前面部分的注意事项仍然适用。

## Pipeline自定义

如果您想要重载特定的pipeline。

请随时为您手头的任务创建一个issue，Pipeline的目标是易于使用并支持大多数情况，因此 `transformers` 可能支持您的用例。

如果您想简单地尝试一下，可以：

- 继承您选择的pipeline

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

这样就可以让您编写所有想要的自定义代码。


## 实现一个pipeline

[实现一个新的pipeline](../add_new_pipeline)

## 音频

可用于音频任务的pipeline包括以下几种。

### AudioClassificationPipeline

[[autodoc]] AudioClassificationPipeline
    - __call__
    - all

### AutomaticSpeechRecognitionPipeline

[[autodoc]] AutomaticSpeechRecognitionPipeline
    - __call__
    - all

### TextToAudioPipeline

[[autodoc]] TextToAudioPipeline
    - __call__
    - all


### ZeroShotAudioClassificationPipeline

[[autodoc]] ZeroShotAudioClassificationPipeline
    - __call__
    - all

## 计算机视觉

可用于计算机视觉任务的pipeline包括以下几种。

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

### ImageToImagePipeline

[[autodoc]] ImageToImagePipeline
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

## 自然语言处理

可用于自然语言处理任务的pipeline包括以下几种。

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

## 多模态

可用于多模态任务的pipeline包括以下几种。

### DocumentQuestionAnsweringPipeline

[[autodoc]] DocumentQuestionAnsweringPipeline
    - __call__
    - all

### FeatureExtractionPipeline

[[autodoc]] FeatureExtractionPipeline
    - __call__
    - all

### ImageFeatureExtractionPipeline

[[autodoc]] ImageFeatureExtractionPipeline
    - __call__
    - all

### ImageToTextPipeline

[[autodoc]] ImageToTextPipeline
    - __call__
    - all

### MaskGenerationPipeline

[[autodoc]] MaskGenerationPipeline
    - __call__
    - all

### VisualQuestionAnsweringPipeline

[[autodoc]] VisualQuestionAnsweringPipeline
    - __call__
    - all

## Parent class: `Pipeline`

[[autodoc]] Pipeline
