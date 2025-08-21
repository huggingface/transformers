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

パイプラインは、推論にモデルを使うための簡単で優れた方法である。パイプラインは、複雑なコードのほとんどを抽象化したオブジェクトです。
パイプラインは、ライブラリから複雑なコードのほとんどを抽象化したオブジェクトで、名前付き固有表現認識、マスク言語モデリング、感情分析、特徴抽出、質問応答などのタスクに特化したシンプルなAPIを提供します。
Recognition、Masked Language Modeling、Sentiment Analysis、Feature Extraction、Question Answeringなどのタスクに特化したシンプルなAPIを提供します。以下を参照のこと。
[タスク概要](../task_summary)を参照してください。


パイプラインの抽象化には2つのカテゴリーがある：

- [`pipeline`] は、他のすべてのパイプラインをカプセル化する最も強力なオブジェクトです。
- タスク固有のパイプラインは、[オーディオ](#audio)、[コンピューター ビジョン](#computer-vision)、[自然言語処理](#natural-language-processing)、および [マルチモーダル](#multimodal) タスクで使用できます。

## The pipeline abstraction

*パイプライン* 抽象化は、他のすべての利用可能なパイプラインのラッパーです。他のものと同様にインスタンス化されます
パイプラインですが、さらなる生活の質を提供できます。

1 つの項目に対する単純な呼び出し:

```python
>>> pipe = pipeline("text-classification")
>>> pipe("This restaurant is awesome")
[{'label': 'POSITIVE', 'score': 0.9998743534088135}]
```

[ハブ](https://huggingface.co) の特定のモデルを使用したい場合は、モデルがオンになっている場合はタスクを無視できます。
ハブはすでにそれを定義しています。

```python
>>> pipe = pipeline(model="FacebookAI/roberta-large-mnli")
>>> pipe("This restaurant is awesome")
[{'label': 'NEUTRAL', 'score': 0.7313136458396912}]
```

多くの項目に対してパイプラインを呼び出すには、*list* を使用してパイプラインを呼び出すことができます。

```python
>>> pipe = pipeline("text-classification")
>>> pipe(["This restaurant is awesome", "This restaurant is awful"])
[{'label': 'POSITIVE', 'score': 0.9998743534088135},
 {'label': 'NEGATIVE', 'score': 0.9996669292449951}]
```

完全なデータセットを反復するには、`Dataset`を直接使用することをお勧めします。これは、割り当てる必要がないことを意味します
データセット全体を一度に処理することも、自分でバッチ処理を行う必要もありません。これはカスタムループと同じくらい速く動作するはずです。
GPU。それが問題でない場合は、ためらわずに問題を作成してください。

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

使いやすくするために、ジェネレーターを使用することもできます。

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


すべてのパイプラインでバッチ処理を使用できます。これはうまくいきます
パイプラインがストリーミング機能を使用するときは常に (つまり、リスト、`dataset`、または `generator`を渡すとき)。

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


ただし、これによってパフォーマンスが自動的に向上するわけではありません。状況に応じて、10 倍の高速化または 5 倍の低速化のいずれかになります。
ハードウェア、データ、使用されている実際のモデルについて。

主に高速化である例:


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

最も速度が低下する例:


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

これは、他の文に比べて非常に長い文が時折あります。その場合、**全体**のバッチは 400 である必要があります。
トークンが長いため、バッチ全体が [64, 4] ではなく [64, 400] になり、速度が大幅に低下します。さらに悪いことに、
バッチが大きくなると、プログラムは単純にクラッシュします。

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

この問題に対する適切な (一般的な) 解決策はなく、使用できる距離はユースケースによって異なる場合があります。のルール
親指：

ユーザーにとっての経験則は次のとおりです。

- **ハードウェアを使用して、負荷に対するパフォーマンスを測定します。測って、測って、測り続ける。実数というのは、
  進むべき唯一の方法。**
- レイテンシに制約がある場合 (実際の製品が推論を実行している場合)、バッチ処理を行わないでください。
- CPU を使用している場合は、バッチ処理を行わないでください。
- GPU でスループットを使用している場合 (大量の静的データでモデルを実行したい場合)、次のようにします。

  - sequence_length (「自然な」データ) のサイズについてまったくわからない場合は、デフォルトではバッチ処理や測定を行わず、
    暫定的に追加してみます。失敗した場合に回復するために OOM チェックを追加します (失敗した場合は、ある時点で回復します)。
    sequence_length を制御します。)
  - sequence_length が非常に規則的である場合、バッチ処理は非常に興味深いものとなる可能性が高く、測定してプッシュしてください。
    OOM が発生するまで続けます。
  - GPU が大きいほど、バッチ処理がより興味深いものになる可能性が高くなります。
- バッチ処理を有効にしたらすぐに、OOM を適切に処理できることを確認してください。


## Pipeline chunk batching

`zero-shot-classification` と `question-answering` は、単一の入力で結果が得られる可能性があるという意味で、少し特殊です。
モデルの複数の前方パス。通常の状況では、これにより `batch_size` 引数に関する問題が発生します。

この問題を回避するために、これらのパイプラインはどちらも少し特殊になっており、代わりに `ChunkPipeline` になっています。
通常の `Pipeline`。要するに：

```python
preprocessed = pipe.preprocess(inputs)
model_outputs = pipe.forward(preprocessed)
outputs = pipe.postprocess(model_outputs)
```

今は次のようになります:

```python
all_model_outputs = []
for preprocessed in pipe.preprocess(inputs):
    model_outputs = pipe.forward(preprocessed)
    all_model_outputs.append(model_outputs)
outputs = pipe.postprocess(all_model_outputs)
```

パイプラインは以下で使用されるため、これはコードに対して非常に透過的である必要があります。
同じ方法。

パイプラインはバッチを自動的に処理できるため、これは簡略化されたビューです。気にする必要はないという意味です
入力が実際にトリガーする前方パスの数については、`batch_size` を最適化できます。
入力とは独立して。前のセクションの注意事項が引き続き適用されます。

## Pipeline custom code

特定のパイプラインをオーバーライドする場合。

目の前のタスクに関する問題を作成することを躊躇しないでください。パイプラインの目標は、使いやすく、ほとんどのユーザーをサポートすることです。
したがって、`transformers`があなたのユースケースをサポートする可能性があります。


単純に試してみたい場合は、次のことができます。

- 選択したパイプラインをサブクラス化します

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

これにより、必要なカスタム コードをすべて実行できるようになります。

## Implementing a pipeline

[Implementing a new pipeline](../add_new_pipeline)

## Audio

オーディオ タスクに使用できるパイプラインには次のものがあります。

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

## Computer vision

コンピューター ビジョン タスクに使用できるパイプラインには次のものがあります。

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

## Natural Language Processing

自然言語処理タスクに使用できるパイプラインには次のものがあります。

### FillMaskPipeline

[[autodoc]] FillMaskPipeline
    - __call__
    - all

### NerPipeline

[[autodoc]] NerPipeline

詳細については、[`TokenClassificationPipeline`] を参照してください。

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

マルチモーダル タスクに使用できるパイプラインには次のものがあります。

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

### ImageTextToTextPipeline

[[autodoc]] ImageTextToTextPipeline
    - __call__
    - all

### VisualQuestionAnsweringPipeline

[[autodoc]] VisualQuestionAnsweringPipeline
    - __call__
    - all

## Parent class: `Pipeline`

[[autodoc]] Pipeline
