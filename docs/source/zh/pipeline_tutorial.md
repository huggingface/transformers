<!--版权所有 2022 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可。除非符合许可证，否则您不得使用此文件。您可以在
http://www.apache.org/licenses/LICENSE-2.0
适用法律要求或书面同意的情况下，根据许可证分发的软件是基于“按原样” BASIS，无论是明示还是暗示，都没有任何保证或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️请注意，此文件是 Markdown 格式的，但包含了我们的 doc-builder（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。
-->
# 推理流程
[`pipeline`] 使得在任何语言、计算机视觉、语音和多模态任务上使用 [Hub](https://huggingface.co/models) 中的任何模型都变得简单。即使您没有使用特定模态的经验或不熟悉模型背后的代码，您仍然可以使用 [`pipeline`] 进行推理！本教程将教您：
* 使用 [`pipeline`] 进行推理。* 使用特定的分词器 (Tokenizer)或模型。* 使用 [`pipeline`] 进行音频、视觉和多模态任务。
<Tip>
查看 [`pipeline`] 文档，了解完整的受支持任务和可用参数列表。
</Tip>
## 使用流程
尽管每个任务都有一个相关的 [`pipeline`]，但使用通用的 [`pipeline`] 抽象更简单，它包含了所有特定任务的流程。[`pipeline`] 会自动加载默认模型和一个能够进行推理的预处理类。
1. 首先创建一个 [`pipeline`] 并指定一个推理任务：
```py
>>> from transformers import pipeline

>>> generator = pipeline(task="automatic-speech-recognition")
```

2. 将输入文本传递给 [`pipeline`]：
```py
>>> generator("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': 'I HAVE A DREAM BUT ONE DAY THIS NATION WILL RISE UP LIVE UP THE TRUE MEANING OF ITS TREES'}
```

结果与您预期的不一样吗？在 Hub 上查看一些 [下载量最多的自动语音识别模型](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=downloads)，看看是否可以获得更好的转录。让我们尝试一下 [openai/whisper-large](https://huggingface.co/openai/whisper-large)：
```py
>>> generator = pipeline(model="openai/whisper-large")
>>> generator("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

现在这个结果看起来更准确了！我们真的鼓励您在 Hub 上查看不同语言的模型、专业领域的模型等等。您可以直接从浏览器在 Hub 上查看和比较模型结果，看看它是否比其他模型更适合或能够处理特殊情况。如果您找不到适用于您的用例的模型，您始终可以 [开始训练](training) 您自己的模型！And if you don't find a model for your use case, you can always start [training](training) your own!

如果您有多个输入，可以将输入作为列表传递给 [`pipeline`]：
```py
generator(
    [
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
    ]
)
```

如果您想遍历整个数据集，或者想在 Web 服务器上进行推理，请查看专门的部分
[在数据集上使用流程](#using-pipelines-on-a-dataset)
[在 Web 服务器上使用流程](./pipeline_webserver)
## 参数
[`pipeline`] 支持许多参数；一些参数是特定于任务的，一些参数适用于所有流程。通常您可以在任何地方指定参数：
```py
generator = pipeline(model="openai/whisper-large", my_parameter=1)
out = generator(...)  # This will use `my_parameter=1`.
out = generator(..., my_parameter=2)  # This will override and use `my_parameter=2`.
out = generator(...)  # This will go back to using `my_parameter=1`.
```

让我们看看其中的 3 个重要参数：
### 设备
如果使用 `device=n`，[`pipeline`] 会自动将模型放在指定的设备上。无论您是使用 PyTorch 还是 Tensorflow，都可以这样做。
```py
generator = pipeline(model="openai/whisper-large", device=0)
```

如果模型对于单个 GPU 来说太大，您可以设置 `device_map="auto"`，以允许🤗 [Accelerate](https://huggingface.co/docs/accelerate) 自动确定如何加载和存储模型权重。
```py
#!pip install accelerate
generator = pipeline(model="openai/whisper-large", device_map="auto")
```

请注意，如果传递了 `device_map="auto"`，在实例化您的 `pipeline` 时无需添加 `device=device` 参数，因为可能会遇到一些意外行为！
### 批处理大小
默认情况下，流程不会对推理进行批处理，原因可以在此处详细解释 [here](https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching)。原因是批处理不一定更快，实际上在某些情况下可能更慢。
但是如果您的用例可以使用批处理，您可以使用：
```py
generator = pipeline(model="openai/whisper-large", device=0, batch_size=2)
audio_filenames = [f"audio_{i}.flac" for i in range(10)]
texts = generator(audio_filenames)
```

这将在提供的 10 个音频文件上运行流程，但会将它们分批处理为 2 个传递给模型（模型在 GPU 上，批处理更有可能有所帮助），而无需您编写更多的代码。输出应始终与您在不进行批处理时收到的结果相匹配。它只是一种帮助您从流程中获得更高速度的方法。
流程还可以减轻一些批处理的复杂性，因为对于某些流程，需要将单个项目（如长音频文件）分成多个部分以便由模型处理。流程会为您执行此 [*chunk batching*](./main_classes/pipelines#pipeline-chunk-batching)。
### 任务特定参数
所有任务都提供任务特定参数，这些参数允许额外的灵活性和选项，以帮助您完成工作。例如，[`transformers.AutomaticSpeechRecognitionPipeline.__call__`] 方法具有一个 `return_timestamps` 参数，对于制作字幕视频可能很有用：

```py
>>> # Not using whisper, as it cannot provide timestamps.
>>> generator = pipeline(model="facebook/wav2vec2-large-960h-lv60-self", return_timestamps="word")
>>> generator("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': 'I HAVE A DREAM BUT ONE DAY THIS NATION WILL RISE UP AND LIVE OUT THE TRUE MEANING OF ITS CREED', 'chunks': [{'text': 'I', 'timestamp': (1.22, 1.24)}, {'text': 'HAVE', 'timestamp': (1.42, 1.58)}, {'text': 'A', 'timestamp': (1.66, 1.68)}, {'text': 'DREAM', 'timestamp': (1.76, 2.14)}, {'text': 'BUT', 'timestamp': (3.68, 3.8)}, {'text': 'ONE', 'timestamp': (3.94, 4.06)}, {'text': 'DAY', 'timestamp': (4.16, 4.3)}, {'text': 'THIS', 'timestamp': (6.36, 6.54)}, {'text': 'NATION', 'timestamp': (6.68, 7.1)}, {'text': 'WILL', 'timestamp': (7.32, 7.56)}, {'text': 'RISE', 'timestamp': (7.8, 8.26)}, {'text': 'UP', 'timestamp': (8.38, 8.48)}, {'text': 'AND', 'timestamp': (10.08, 10.18)}, {'text': 'LIVE', 'timestamp': (10.26, 10.48)}, {'text': 'OUT', 'timestamp': (10.58, 10.7)}, {'text': 'THE', 'timestamp': (10.82, 10.9)}, {'text': 'TRUE', 'timestamp': (10.98, 11.18)}, {'text': 'MEANING', 'timestamp': (11.26, 11.58)}, {'text': 'OF', 'timestamp': (11.66, 11.7)}, {'text': 'ITS', 'timestamp': (11.76, 11.88)}, {'text': 'CREED', 'timestamp': (12.0, 12.38)}]}
```

正如您所见，模型推断出文本并输出了各个单词的 **发音时间**。
每个任务都有许多可用参数，因此请查看每个任务的 API 参考，了解您可以进行哪些调整！例如，[`~transformers.AutomaticSpeechRecognitionPipeline`] 有一个 `chunk_length_s` 参数，对于处理非常长的音频文件（例如，为整部电影或一小时长的视频制作字幕）非常有帮助。这些是模型通常无法单独处理的文件。

如果找不到真正有帮助的参数，请随时 [提出请求](https://github.com/huggingface/transformers/issues/new?assignees=&labels=feature&template=feature-request.yml)！

## 在数据集上使用流程
流程还可以对大型数据集进行推理。我们推荐的最简单方法是使用迭代器：
```py
def data():
    for i in range(1000):
        yield f"My example {i}"


pipe = pipeline(model="gpt2", device=0)
generated_characters = 0
for out in pipe(data()):
    generated_characters += len(out[0]["generated_text"])
```

迭代器 `data()` 产生每个结果，而流程会自动识别到输入是可迭代的，并在继续处理的同时在 GPU 上处理数据（这在底层使用 [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)）。这很重要，因为您不需要为整个数据集分配内存并且可以尽可能快地将数据传递给 GPU。
由于批处理可能加快速度，因此在这里尝试调整 `batch_size` 参数可能很有用。
遍历数据集的最简单方法是只需从🤗 [Datasets](https://github.com/huggingface/datasets/) 加载一个数据集：
```py
# KeyDataset is a util that will just output the item we're interested in.
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset

pipe = pipeline(model="hf-internal-testing/tiny-random-wav2vec2", device=0)
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:10]")

for out in pipe(KeyDataset(dataset, "audio")):
    print(out)
```


## 在 Web 服务器上使用流程
<Tip> 创建推理引擎是一个复杂的主题，值得拥有自己的页面。</Tip>
[链接](./pipeline_webserver)
## 视觉流程
对于视觉任务，使用 [`pipeline`] 几乎是相同的。
指定您的任务并将图像传递给分类器。图像可以是链接或图像的本地路径。例如，下面显示了哪种猫的物种？
![pipeline-cat-chonk](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg)
```py
>>> from transformers import pipeline

>>> vision_classifier = pipeline(model="google/vit-base-patch16-224")
>>> preds = vision_classifier(
...     images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.4335, 'label': 'lynx, catamount'}, {'score': 0.0348, 'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor'}, {'score': 0.0324, 'label': 'snow leopard, ounce, Panthera uncia'}, {'score': 0.0239, 'label': 'Egyptian cat'}, {'score': 0.0229, 'label': 'tiger cat'}]
```

## 文本流程
使用 NLP 任务的 [`pipeline`] 实际上是完全相同的。
```py
>>> from transformers import pipeline

>>> # This model is a `zero-shot-classification` model.
>>> # It will classify text, except you are free to choose any label you might imagine
>>> classifier = pipeline(model="facebook/bart-large-mnli")
>>> classifier(
...     "I have a problem with my iphone that needs to be resolved asap!!",
...     candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
... )
{'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'], 'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}
```

## 多模态管道
[`pipeline`] 支持多个模态。例如，视觉问答（VQA）任务结合了文本和图像。请随意使用您喜欢的任何图像链接和想要询问图像的问题。图像可以是 URL 或图像的本地路径。
例如，如果您使用这个 [发票图像](https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png)：
```py
>>> from transformers import pipeline

>>> vqa = pipeline(model="impira/layoutlm-document-qa")
>>> vqa(
...     image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
...     question="What is the invoice number?",
... )
[{'score': 0.42515, 'answer': 'us-001', 'start': 16, 'end': 16}]
```

<Tip>
要运行上面的示例，您需要额外安装 [`pytesseract`](https://pypi.org/project/pytesseract/) 并且安装 🤗 Transformers:
```bash
sudo apt install -y tesseract-ocr
pip install pytesseract
```

</Tip>
## 使用 🤗 `accelerate` 运行 `pipeline` 的大模型:
您可以使用 🤗 `accelerate` 轻松运行大型模型上的 `pipeline`！首先确保您已经安装了 `accelerate`，可以通过 `pip install accelerate` 进行安装。
使用 `device_map="auto"` 首先加载您的模型！我们将在示例中使用 `facebook/opt-1.3b`。
```py
# pip install accelerate
import torch
from transformers import pipeline

pipe = pipeline(model="facebook/opt-1.3b", torch_dtype=torch.bfloat16, device_map="auto")
output = pipe("This is a cool example!", do_sample=True, top_p=0.95)
```

如果您安装了 `bitsandbytes` 并添加了参数 `load_in_8bit=True`，您还可以传递 8 位加载的模型
```py
# pip install accelerate bitsandbytes
import torch
from transformers import pipeline

pipe = pipeline(model="facebook/opt-1.3b", device_map="auto", model_kwargs={"load_in_8bit": True})
output = pipe("This is a cool example!", do_sample=True, top_p=0.95)
```

请注意，您可以将检查点替换为任何支持大型模型加载的 Hugging Face 模型，例如 BLOOM！