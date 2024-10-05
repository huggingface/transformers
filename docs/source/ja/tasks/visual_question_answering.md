<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Visual Question Answering

[[open-in-colab]]

Visual Question Answering (VQA) は、画像に基づいて自由形式の質問に答えるタスクです。
このタスクをサポートするモデルへの入力は通常、画像と質問の組み合わせであり、出力は
自然言語で表現された答え。

VQA の注目すべき使用例には次のようなものがあります。
* 視覚障害者向けのアクセシビリティ アプリケーション。
* 教育: 講義や教科書で示されている視覚的な資料について質問を投げかけること。 VQA は、インタラクティブな博物館の展示物や史跡でも利用できます。
* カスタマー サービスと電子商取引: VQA は、ユーザーが製品について質問できるようにすることでユーザー エクスペリエンスを向上させます。
* 画像検索: VQA モデルを使用して、特定の特徴を持つ画像を検索できます。たとえば、ユーザーは「犬はいますか?」と尋ねることができます。一連の画像から犬が写っているすべての画像を検索します。

このガイドでは、次の方法を学びます。

- [`Graphcore/vqa` データセット](https://huggingface.co/datasets/Graphcore/vqa) 上で分類 VQA モデル、特に [ViLT](../model_doc/vilt) を微調整します。
- 微調整された ViLT を推論に使用します。
- BLIP-2 などの生成モデルを使用してゼロショット VQA 推論を実行します。

## Fine-tuning ViLT

ViLT モデルは、Vision Transformer (ViT) にテキスト埋め込みを組み込んでおり、最小限の設計を可能にします。
視覚と言語の事前トレーニング (VLP)。このモデルは、いくつかの下流タスクに使用できます。 VQA タスクの場合、分類子
head は最上部 (`[CLS]` トークンの最終的な非表示状態の最上部にある線形層) に配置され、ランダムに初期化されます。
したがって、視覚的質問応答は **分類問題** として扱われます。

BLIP、BLIP-2、InstructBLIP などの最近のモデルは、VQA を生成タスクとして扱います。このガイドの後半では、
ゼロショット VQA 推論にそれらを使用する方法を示します。

始める前に、必要なライブラリがすべてインストールされていることを確認してください。

```bash
pip install -q transformers datasets
```
モデルをコミュニティと共有することをお勧めします。 Hugging Face アカウントにログインして、🤗 ハブにアップロードします。
プロンプトが表示されたら、トークンを入力してログインします。

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

モデルのチェックポイントをグローバル変数として定義しましょう。

```py
>>> model_checkpoint = "dandelin/vilt-b32-mlm"
```

## Load the data

説明の目的で、このガイドでは、注釈付きの視覚的な質問に答える「Graphcore/vqa」データセットの非常に小さなサンプルを使用します。
完全なデータセットは [🤗 Hub](https://huggingface.co/datasets/Graphcore/vqa) で見つけることができます。

[`Graphcore/vqa` データセット](https://huggingface.co/datasets/Graphcore/vqa) の代わりに、
公式 [VQA データセット ページ](https://visualqa.org/download.html) から同じデータを手動で取得します。フォローしたい場合は、
カスタム データを使用したチュートリアルでは、[画像データセットを作成する](https://huggingface.co/docs/datasets/image_dataset#loading-script) 方法を確認してください。
🤗 データセットのドキュメントのガイド。

検証分割から最初の 200 個の例をロードし、データセットの機能を調べてみましょう。

```python
>>> from datasets import load_dataset

>>> dataset = load_dataset("Graphcore/vqa", split="validation[:200]")
>>> dataset
Dataset({
    features: ['question', 'question_type', 'question_id', 'image_id', 'answer_type', 'label'],
    num_rows: 200
})
```
データセットの特徴を理解するために例を見てみましょう。

```py
>>> dataset[0]
{'question': 'Where is he looking?',
 'question_type': 'none of the above',
 'question_id': 262148000,
 'image_id': '/root/.cache/huggingface/datasets/downloads/extracted/ca733e0e000fb2d7a09fbcc94dbfe7b5a30750681d0e965f8e0a23b1c2f98c75/val2014/COCO_val2014_000000262148.jpg',
 'answer_type': 'other',
 'label': {'ids': ['at table', 'down', 'skateboard', 'table'],
  'weights': [0.30000001192092896,
   1.0,
   0.30000001192092896,
   0.30000001192092896]}}
```

このタスクに関連する機能には次のものがあります。
* `question`: 画像から回答する質問
* `image_id`: 質問が参照する画像へのパス
* `label`: 注釈

残りの機能は必要ないので削除できます。


```py
>>> dataset = dataset.remove_columns(['question_type', 'question_id', 'answer_type'])
```

ご覧のとおり、`label`機能には、さまざまなヒューマン・アノテーターによって収集された、同じ質問に対する複数の回答 (ここでは`id`と呼びます) が含まれています。
質問に対する答えは主観的なものになる可能性があるためです。この場合、問題は "彼はどこを見ているのか？"ということです。一部の人々
これには "ダウン" という注釈が付けられ、他のものには "テーブルで" という注釈が付けられ、別の注釈には "スケートボード" という注釈が付けられました。

画像を見て、どの答えを出すかを考えてください。

```python
>>> from PIL import Image

>>> image = Image.open(dataset[0]['image_id'])
>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/vqa-example.png" alt="VQA Image Example"/>
</div>


質問と回答のあいまいさのため、このようなデータセットはマルチラベル分類問題として扱われます (
複数の回答が有効である可能性があります)。さらに、ワンホット エンコードされたベクトルを作成するだけではなく、
注釈内に特定の回答が出現した回数に基づくソフト エンコーディング。

たとえば、上の例では、"down"という回答が他の回答よりも頻繁に選択されるため、
スコア (データセットでは`weight`と呼ばれます) は 1.0 で、残りの回答のスコアは 1.0 未満です。

後で適切な分類ヘッドを使用してモデルをインスタンス化するために、2 つの辞書を作成しましょう。
ラベル名を整数に変換する、またはその逆:

```py
>>> import itertools

>>> labels = [item['ids'] for item in dataset['label']]
>>> flattened_labels = list(itertools.chain(*labels))
>>> unique_labels = list(set(flattened_labels))

>>> label2id = {label: idx for idx, label in enumerate(unique_labels)}
>>> id2label = {idx: label for label, idx in label2id.items()}
```

マッピングができたので、文字列の回答をその ID に置き換え、さらに前処理をより便利にするためにデータセットをフラット化することができます。

```python
>>> def replace_ids(inputs):
...   inputs["label"]["ids"] = [label2id[x] for x in inputs["label"]["ids"]]
...   return inputs


>>> dataset = dataset.map(replace_ids)
>>> flat_dataset = dataset.flatten()
>>> flat_dataset.features
{'question': Value(dtype='string', id=None),
 'image_id': Value(dtype='string', id=None),
 'label.ids': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None),
 'label.weights': Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None)}
```

## Preprocessing data

次のステップでは、ViLT プロセッサをロードして、モデルの画像データとテキスト データを準備します。
[`ViltProcessor`] は、BERT トークナイザーと ViLT 画像プロセッサを便利な単一プロセッサにラップします。

```py
>>> from transformers import ViltProcessor

>>> processor = ViltProcessor.from_pretrained(model_checkpoint)
```

データを前処理するには、[`ViltProcessor`] を使用して画像と質問をエンコードする必要があります。プロセッサーは使用します
[`BertTokenizerFast`] を使用してテキストをトークン化し、テキスト データの `input_ids`、`attention_mask`、および `token_type_ids` を作成します。
画像に関しては、プロセッサは [`ViltImageProcessor`] を利用して画像のサイズ変更と正規化を行い、`pixel_values` と `pixel_mask` を作成します。

これらの前処理ステップはすべて内部で行われ、`processor`を呼び出すだけで済みます。ただし、それでも必要なのは、
対象のラベルを準備します。この表現では、各要素は考えられる答え (ラベル) に対応します。正解の場合、要素は保持されます。
それぞれのスコア (重み) が設定され、残りの要素は 0 に設定されます。

次の関数は、画像と質問に `processor` を適用し、上で説明したようにラベルをフォーマットします。

```py
>>> import torch

>>> def preprocess_data(examples):
...     image_paths = examples['image_id']
...     images = [Image.open(image_path) for image_path in image_paths]
...     texts = examples['question']

...     encoding = processor(images, texts, padding="max_length", truncation=True, return_tensors="pt")

...     for k, v in encoding.items():
...           encoding[k] = v.squeeze()

...     targets = []

...     for labels, scores in zip(examples['label.ids'], examples['label.weights']):
...         target = torch.zeros(len(id2label))

...         for label, score in zip(labels, scores):
...             target[label] = score

...         targets.append(target)

...     encoding["labels"] = targets

...     return encoding
```

データセット全体に前処理関数を適用するには、🤗 Datasets [`~datasets.map`] 関数を使用します。 `map` を高速化するには、次のようにします。
データセットの複数の要素を一度に処理するには、`batched=True` を設定します。この時点で、不要な列は自由に削除してください。


```py
>>> processed_dataset = flat_dataset.map(preprocess_data, batched=True, remove_columns=['question','question_type',  'question_id', 'image_id', 'answer_type', 'label.ids', 'label.weights'])
>>> processed_dataset
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'pixel_values', 'pixel_mask', 'labels'],
    num_rows: 200
})
```

最後のステップとして、[`DefaultDataCollat​​or`] を使用してサンプルのバッチを作成します。


```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

## Train the model

これでモデルのトレーニングを開始する準備が整いました。 [`ViltForQuestionAnswering`] で ViLT をロードします。ラベルの数を指定します
ラベルマッピングとともに:

```py
>>> from transformers import ViltForQuestionAnswering

>>> model = ViltForQuestionAnswering.from_pretrained(model_checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)
```

この時点で残っているステップは 3 つだけです。

1. [`TrainingArguments`] でトレーニング ハイパーパラメータを定義します。

```py
>>> from transformers import TrainingArguments

>>> repo_id = "MariaK/vilt_finetuned_200"

>>> training_args = TrainingArguments(
...     output_dir=repo_id,
...     per_device_train_batch_size=4,
...     num_train_epochs=20,
...     save_steps=200,
...     logging_steps=50,
...     learning_rate=5e-5,
...     save_total_limit=2,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )
```

2. トレーニング引数をモデル、データセット、プロセッサー、データ照合器とともに [`Trainer`] に渡します。

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=processed_dataset,
...     processing_class=processor,
... )
```

3. [`~Trainer.train`] を呼び出してモデルを微調整します。

```py
>>> trainer.train()
```

トレーニングが完了したら、 [`~Trainer.push_to_hub`] メソッドを使用してモデルをハブに共有し、🤗 ハブで最終モデルを共有します。

```py
>>> trainer.push_to_hub()
```

## Inference

ViLT モデルを微調整し、🤗 Hub にアップロードしたので、それを推論に使用できます。もっとも単純な
推論用に微調整されたモデルを試す方法は、それを [`pipeline`] で使用することです。

```py
>>> from transformers import pipeline

>>> pipe = pipeline("visual-question-answering", model="MariaK/vilt_finetuned_200")
```

このガイドのモデルは 200 の例でのみトレーニングされているため、多くを期待しないでください。少なくともそれがあるかどうか見てみましょう
データから何かを学習し、推論を説明するためにデータセットから最初の例を取り出します。

```py
>>> example = dataset[0]
>>> image = Image.open(example['image_id'])
>>> question = example['question']
>>> print(question)
>>> pipe(image, question, top_k=1)
"Where is he looking?"
[{'score': 0.5498199462890625, 'answer': 'down'}]
```

あまり自信がありませんが、モデルは確かに何かを学習しました。より多くの例とより長いトレーニングを行うと、はるかに良い結果が得られます。

必要に応じて、パイプラインの結果を手動で複製することもできます。
1. 画像と質問を取得し、モデルのプロセッサを使用してモデル用に準備します。
2. モデルを通じて結果または前処理を転送します。
3. ロジットから、最も可能性の高い回答の ID を取得し、`id2label` で実際の回答を見つけます。

```py
>>> processor = ViltProcessor.from_pretrained("MariaK/vilt_finetuned_200")

>>> image = Image.open(example['image_id'])
>>> question = example['question']

>>> # prepare inputs
>>> inputs = processor(image, question, return_tensors="pt")

>>> model = ViltForQuestionAnswering.from_pretrained("MariaK/vilt_finetuned_200")

>>> # forward pass
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> logits = outputs.logits
>>> idx = logits.argmax(-1).item()
>>> print("Predicted answer:", model.config.id2label[idx])
Predicted answer: down
```

## Zero-shot VQA

以前のモデルでは、VQA を分類タスクとして扱いました。 BLIP、BLIP-2、InstructBLIP アプローチなどの一部の最近のモデル
生成タスクとしての VQA。 [BLIP-2](../model_doc/blip-2) を例として考えてみましょう。新しいビジュアル言語の事前トレーニングを導入しました
事前にトレーニングされたビジョン エンコーダーと LLM を任意に組み合わせて使用​​できるパラダイム (詳細については、[BLIP-2 ブログ投稿](https://huggingface.co/blog/blip-2) を参照)。
これにより、視覚的な質問応答を含む複数の視覚言語タスクで最先端の結果を達成することができます。

このモデルを VQA に使用する方法を説明しましょう。まず、モデルをロードしましょう。ここではモデルを明示的に送信します。
GPU (利用可能な場合)。これは [`Trainer`] が自動的に処理するため、トレーニング時に事前に行う必要はありませんでした。


```py
>>> from transformers import AutoProcessor, Blip2ForConditionalGeneration
>>> import torch

>>> processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
>>> model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)
```

モデルは画像とテキストを入力として受け取るため、VQA データセットの最初の例とまったく同じ画像と質問のペアを使用してみましょう。


```py
>>> example = dataset[0]
>>> image = Image.open(example['image_id'])
>>> question = example['question']
```

視覚的な質問応答タスクに BLIP-2 を使用するには、テキスト プロンプトが特定の形式 (`Question: {} Answer:`) に従う必要があります。


```py
>>> prompt = f"Question: {question} Answer:"
```

次に、モデルのプロセッサで画像/プロンプトを前処理し、処理された入力をモデルに渡し、出力をデコードする必要があります。

```py
>>> inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

>>> generated_ids = model.generate(**inputs, max_new_tokens=10)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
>>> print(generated_text)
"He is looking at the crowd"
```

ご覧のとおり、モデルは群衆と顔の向き (下を向いている) を認識しましたが、見逃しているようです。
観客がスケーターの後ろにいるという事実。それでも、人間が注釈を付けたデータセットを取得することが不可能な場合には、これは
このアプローチにより、有用な結果がすぐに得られます。
