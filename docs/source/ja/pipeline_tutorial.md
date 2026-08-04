<!--
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ このファイルはMarkdown形式ですが、当社のdoc-builder（MDXに似た構文）を含むため、Markdownビューアで正しく表示されないことがあります。

-->

# Pipelines for inference

[`pipeline`]を使用することで、[Hub](https://huggingface.co/models)からの任意のモデルを言語、コンピュータビジョン、音声、およびマルチモーダルタスクの推論に簡単に使用できます。
特定のモダリティに関する経験がない場合や、モデルの背後にあるコードに精通していない場合でも、[`pipeline`]を使用して推論できます！
このチュートリアルでは、次のことを学びます：

- 推論のための[`pipeline`]の使用方法。
- 特定のトークナイザやモデルの使用方法。
- オーディオ、ビジョン、マルチモーダルタスクのための[`pipeline`]の使用方法。

<Tip>

サポートされているタスクと利用可能なパラメータの完全な一覧については、[`pipeline`]のドキュメンテーションをご覧ください。

</Tip>

## Pipeline usage

各タスクには関連する[`pipeline`]がありますが、タスク固有の[`pipeline`]を使用する代わりに、すべてのタスク固有のパイプラインを含む一般的な[`pipeline`]の抽象化を使用すると、より簡単です。[`pipeline`]は自動的にデフォルトのモデルと、タスクの推論が可能な前処理クラスを読み込みます。

1. [`pipeline`]を作成し、推論タスクを指定して始めます：

```py
>>> from transformers import pipeline

>>> generator = pipeline(task="automatic-speech-recognition")
```

2. [`pipeline`]に入力テキストを渡します：

```python
>>> generator("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': 'I HAVE A DREAM BUT ONE DAY THIS NATION WILL RISE UP LIVE UP THE TRUE MEANING OF ITS TREES'}
```

チェックアウトできなかったか？ [Hubの最もダウンロードされた自動音声認識モデル](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=downloads) のいくつかを見て、より良い転写を得ることができるかどうかを確認してみてください。
[openai/whisper-large](https://huggingface.co/openai/whisper-large) を試してみましょう：

```python
>>> generator = pipeline(model="openai/whisper-large")
>>> generator("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

この結果はより正確に見えますね！
異なる言語、専門分野に特化したモデル、その他のモデルについては、Hubをチェックすることを強くお勧めします。
Hubでは、ブラウザから直接モデルの結果をチェックして、他のモデルよりも適しているか、特殊なケースをよりよく処理できるかを確認できます。
そして、あなたのユースケースに適したモデルが見つからない場合、いつでも[トレーニング](training)を開始できます！

複数の入力がある場合、入力をリストとして渡すことができます：

```py
generator(
    [
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
    ]
)
```

データセット全体を繰り返し処理したり、ウェブサーバーで推論に使用したい場合は、専用の部分を確認してください。

[データセットでパイプラインを使用する](#using-pipeline-in-a-dataset)

[ウェブサーバーでパイプラインを使用する](./pipeline_webserver)



## パラメータ

[`pipeline`]は多くのパラメータをサポートしており、一部はタスク固有であり、一部はすべてのパイプラインに共通です。
一般的には、どこでもパラメータを指定できます：

```py
generator = pipeline(model="openai/whisper-large", my_parameter=1)
out = generator(...)  # これは `my_parameter=1` を使用します。
out = generator(..., my_parameter=2)  # これは上書きして `my_parameter=2` を使用します。
out = generator(...)  # これは再び `my_parameter=1` を使用します。
```

3つの重要なものを確認しましょう：

### Device

`device=n` を使用すると、パイプラインはモデルを指定したデバイスに自動的に配置します。
これは、PyTorchまたはTensorflowを使用しているかどうかに関係なく機能します。

```py
generator = pipeline(model="openai/whisper-large", device=0)
```

もしモデルが単一のGPUには大きすぎる場合、`device_map="auto"`を設定して、🤗 [Accelerate](https://huggingface.co/docs/accelerate) にモデルの重みをどのようにロードし、保存するかを自動的に決定させることができます。

```python
#!pip install accelerate
generator = pipeline(model="openai/whisper-large", device_map="auto")
```

注意: `device_map="auto"` が渡された場合、`pipeline` をインスタンス化する際に `device=device` 引数を追加する必要はありません。そうしないと、予期しない動作に遭遇する可能性があります！

### Batch size

デフォルトでは、パイプラインは詳細について[こちら](https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching)で説明されている理由から、推論をバッチ処理しません。その理由は、バッチ処理が必ずしも速くないためであり、実際にはいくつかのケースでかなり遅くなることがあるからです。

ただし、あなたのユースケースで機能する場合は、次のように使用できます：

```py
generator = pipeline(model="openai/whisper-large", device=0, batch_size=2)
audio_filenames = [f"audio_{i}.flac" for i in range(10)]
texts = generator(audio_filenames)
```

これにより、パイプラインは提供された10個のオーディオファイルでパイプラインを実行しますが、
モデルにはバッチ処理がより効果的であるGPU上にあり、バッチ処理を行うための追加のコードは必要ありません。
出力は常にバッチ処理なしで受け取ったものと一致するはずです。これは単にパイプラインからより高速な処理を得るための方法として提供されています。

パイプラインは、バッチ処理のいくつかの複雑さを軽減することもできます。なぜなら、一部のパイプラインでは、
モデルで処理するために1つのアイテム（長いオーディオファイルのようなもの）を複数の部分に分割する必要がある場合があるからです。
パイプラインはこれをあなたのために実行します。[*チャンクバッチ処理*](./main_classes/pipelines#pipeline-chunk-batching)として知られるものを実行します。

### Task specific parameters

すべてのタスクは、タスク固有のパラメータを提供し、追加の柔軟性とオプションを提供して、作業をスムーズに進めるのに役立ちます。
たとえば、[`transformers.AutomaticSpeechRecognitionPipeline.__call__`]メソッドには、ビデオの字幕作成に有用な`return_timestamps`パラメータがあります。

```py
>>> # Not using whisper, as it cannot provide timestamps.
>>> generator = pipeline(model="facebook/wav2vec2-large-960h-lv60-self", return_timestamps="word")
>>> generator("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': 'I HAVE A DREAM BUT ONE DAY THIS NATION WILL RISE UP AND LIVE OUT THE TRUE MEANING OF ITS CREED', 'chunks': [{'text': 'I', 'timestamp': (1.22, 1.24)}, {'text': 'HAVE', 'timestamp': (1.42, 1.58)}, {'text': 'A', 'timestamp': (1.66, 1.68)}, {'text': 'DREAM', 'timestamp': (1.76, 2.14)}, {'text': 'BUT', 'timestamp': (3.68, 3.8)}, {'text': 'ONE', 'timestamp': (3.94, 4.06)}, {'text': 'DAY', 'timestamp': (4.16, 4.3)}, {'text': 'THIS', 'timestamp': (6.36, 6.54)}, {'text': 'NATION', 'timestamp': (6.68, 7.1)}, {'text': 'WILL', 'timestamp': (7.32, 7.56)}, {'text': 'RISE', 'timestamp': (7.8, 8.26)}, {'text': 'UP', 'timestamp': (8.38, 8.48)}, {'text': 'AND', 'timestamp': (10.08, 10.18)}, {'text': 'LIVE', 'timestamp': (10.26, 10.48)}, {'text': 'OUT', 'timestamp': (10.58, 10.7)}, {'text': 'THE', 'timestamp': (10.82, 10.9)}, {'text': 'TRUE', 'timestamp': (10.98, 11.18)}, {'text': 'MEANING', 'timestamp': (11.26, 11.58)}, {'text': 'OF', 'timestamp': (11.66, 11.7)}, {'text': 'ITS', 'timestamp': (11.76, 11.88)}, {'text': 'CREED', 'timestamp': (12.0, 12.38)}]}
```

モデルは、テキストを推測し、文の中で各単語がいつ発音されたかを出力しました。

各タスクごとに利用可能な多くのパラメータがありますので、何を調整できるかを確認するために各タスクのAPIリファレンスを確認してください！
たとえば、[`~transformers.AutomaticSpeechRecognitionPipeline`]には、モデル単体では処理できない非常に長いオーディオファイル（たとえば、映画全体や1時間のビデオの字幕付けなど）で役立つ`chunk_length_s`パラメータがあります。

<!--役立つパラメータが見つからない場合は、[リクエスト](https://github.com/huggingface/transformers/issues/new?assignees=&labels=feature&template=feature-request.yml)してください！-->

役立つパラメータが見つからない場合は、[リクエスト](https://github.com/huggingface/transformers/issues/new?assignees=&labels=feature&template=feature-request.yml)してください！

## Using pipeline in a dataset

パイプラインは大規模なデータセット上で推論を実行することもできます。これを行う最も簡単な方法は、イテレータを使用することです：

```py
def data():
    for i in range(1000):
        yield f"My example {i}"


pipe = pipeline(model="openai-community/gpt2", device=0)
generated_characters = 0
for out in pipe(data()):
    generated_characters += len(out[0]["generated_text"])
```

イテレーター `data()` は各結果を生成し、パイプラインは自動的に入力が反復可能であることを認識し、データを取得し続けながらGPU上で処理を行います（これは[DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)を内部で使用しています）。
これは、データセット全体にメモリを割り当てる必要がなく、GPUにできるだけ速くデータを供給できるため重要です。

バッチ処理は処理を高速化できる可能性があるため、ここで`batch_size`パラメータを調整して試すことが役立つかもしれません。

データセットを反復処理する最も簡単な方法は、🤗 [Datasets](https://github.com/huggingface/datasets/)からデータセットを読み込むことです：

```py
# KeyDataset is a util that will just output the item we're interested in.
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset

pipe = pipeline(model="hf-internal-testing/tiny-random-wav2vec2", device=0)
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:10]")

for out in pipe(KeyDataset(dataset, "audio")):
    print(out)
```

## Using pipelines for a webserver

<Tip>
推論エンジンを作成することは複雑なトピックで、独自のページが必要です。
</Tip>

[リンク](./pipeline_webserver)

## Vision pipeline

ビジョンタスク用の[`pipeline`]を使用する方法はほぼ同じです。

タスクを指定し、画像をクラシファイアに渡します。画像はリンク、ローカルパス、またはBase64エンコードされた画像であることができます。例えば、以下の画像はどの種類の猫ですか？

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

## Text pipeline

[`pipeline`]を使用することは、NLPタスクに対してほぼ同じです。

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

## Multimodal pipeline

[`pipeline`]は、1つ以上のモダリティをサポートしています。たとえば、視覚的な質問応答（VQA）タスクはテキストと画像を組み合わせています。
好きな画像リンクと画像に関する質問を自由に使ってください。画像はURLまたは画像のローカルパスで指定できます。

例えば、この[請求書画像](https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png)を使用する場合：

```py
>>> from transformers import pipeline

>>> vqa = pipeline(model="impira/layoutlm-document-qa")
>>> output = vqa(
...     image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
...     question="What is the invoice number?",
... )
>>> output[0]["score"] = round(output[0]["score"], 3)
>>> output
[{'score': 0.425, 'answer': 'us-001', 'start': 16, 'end': 16}]
```

<Tip>

上記の例を実行するには、🤗 Transformersに加えて [`pytesseract`](https://pypi.org/project/pytesseract/) がインストールされている必要があります。

```bash
sudo apt install -y tesseract-ocr
pip install pytesseract
```

</Tip>

## Using `pipeline` on large models with 🤗 `accelerate`:

まず、`accelerate` を`pip install accelerate` でインストールしていることを確認してください。

次に、`device_map="auto"` を使用してモデルをロードします。この例では `facebook/opt-1.3b` を使用します。

```python
# pip install accelerate
import torch
from transformers import pipeline

pipe = pipeline(model="facebook/opt-1.3b", dtype=torch.bfloat16, device_map="auto")
output = pipe("これは素晴らしい例です！", do_sample=True, top_p=0.95)
```

もし `bitsandbytes` をインストールし、`load_in_8bit=True` 引数を追加すれば、8ビットで読み込まれたモデルを渡すこともできます。

```py
# pip install accelerate bitsandbytes
import torch
from transformers import pipeline

pipe = pipeline(model="facebook/opt-1.3b", device_map="auto", model_kwargs={"load_in_8bit": True})
output = pipe("This is a cool example!", do_sample=True, top_p=0.95)
```

注意: BLOOMなどの大規模モデルのロードをサポートするHugging Faceモデルのいずれかで、チェックポイントを置き換えることができます！
