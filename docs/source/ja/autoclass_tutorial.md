<!--
著作権 2023 The HuggingFace Team。全著作権所有。

Apache License、Version 2.0（以下「ライセンス」と呼びます）に基づくライセンスで、
ライセンスに従わない限り、このファイルを使用できません。
ライセンスのコピーは以下から入手できます：

http://www.apache.org/licenses/LICENSE-2.0

適用法に従うか、書面による同意がある限り、ライセンスの下でソフトウェアは配布されます。
ライセンスに基づく特定の言語での条件を確認するか、ライセンスを参照してください。
このファイルはMarkdown形式ですが、doc-builder（MDXに類似したもの）の特定の構文を含んでおり、
お使いのMarkdownビューアで正しくレンダリングされない場合があります。

-->

# AutoClassを使用して事前学習済みインスタンスをロードする

さまざまなTransformerアーキテクチャが存在するため、自分のタスクに合ったモデルを作成するのは難しいことがあります。
🤗 Transformersのコア哲学の一環として、ライブラリを使用しやすく、シンプルで柔軟にするために、
`AutoClass`は与えられたチェックポイントから正しいアーキテクチャを自動的に推論してロードします。
`from_pretrained()`メソッドを使用すると、事前学習済みモデルを素早くロードできるため、モデルをゼロからトレーニングするために時間とリソースを費やす必要がありません。
この種のチェックポイントに依存しないコードを生成することは、
コードが1つのチェックポイントで動作すれば、アーキテクチャが異なっていても、同じタスクに向けてトレーニングされた場合は別のチェックポイントでも動作することを意味します。

<Tip>

アーキテクチャはモデルの骨格を指し、チェックポイントは特定のアーキテクチャの重みです。
たとえば、[BERT](https://huggingface.co/bert-base-uncased)はアーキテクチャであり、`bert-base-uncased`はチェックポイントです。
モデルはアーキテクチャまたはチェックポイントのどちらを指す一般的な用語です。

</Tip>

このチュートリアルでは、以下を学習します：

* 事前学習済みトークナイザをロードする。
* 事前学習済み画像プロセッサをロードする。
* 事前学習済み特徴量抽出器をロードする。
* 事前学習済みプロセッサをロードする。
* 事前学習済みモデルをロードする。

## AutoTokenizer

ほとんどのNLPタスクはトークナイザで始まります。トークナイザは入力をモデルで処理できる形式に変換します。

[`AutoTokenizer.from_pretrained`]を使用してトークナイザをロードします：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```


次に、以下のように入力をトークナイズします：

```py
>>> sequence = "In a hole in the ground there lived a hobbit."
>>> print(tokenizer(sequence))
{'input_ids': [101, 1999, 1037, 4920, 1999, 1996, 2598, 2045, 2973, 1037, 7570, 10322, 4183, 1012, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

## AutoImageProcessor

ビジョンタスクの場合、画像プロセッサが画像を正しい入力形式に変換します。

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

## AutoFeatureExtractor

オーディオタスクの場合、特徴量抽出器がオーディオ信号を正しい入力形式に変換します。

[`AutoFeatureExtractor.from_pretrained`]を使用して特徴量抽出器をロードします.

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained(
...     "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
... )
```

## AutoProcessor

マルチモーダルタスクの場合、2つの前処理ツールを組み合わせるプロセッサが必要です。たとえば、
[LayoutLMV2](model_doc/layoutlmv2)モデルは画像を処理するための画像プロセッサとテキストを処理するためのトークナイザが必要です。
プロセッサはこれらの両方を組み合わせます。

[`AutoProcessor.from_pretrained`]を使用してプロセッサをロードします：

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
```

## AutoModel

<frameworkcontent>
<pt>
最後に、`AutoModelFor`クラスは特定のタスクに対して事前学習済みモデルをロードできます（使用可能なタスクの完全な一覧については[こちら](model_doc/auto)を参照）。
たとえば、[`AutoModelForSequenceClassification.from_pretrained`]を使用してシーケンス分類用のモデルをロードできます：

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

同じチェックポイントを再利用して異なるタスクのアーキテクチャをロードできます：

```py
>>> from transformers import AutoModelForTokenClassification

>>> model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased")
```

<Tip warning={true}>

PyTorchモデルの場合、  `from_pretrained()`メソッドは内部で`torch.load()`を使用し、内部的には`pickle`を使用しており、セキュリティの問題が知られています。
一般的には、信頼性のないソースから取得した可能性があるモデルや改ざんされた可能性のあるモデルをロードしないでください。
このセキュリティリスクは、`Hugging Face Hub`でホストされている公開モデルに対して部分的に緩和されており、各コミットでマルウェアのスキャンが行われています。
GPGを使用した署名済みコミットの検証などのベストプラクティスについては、Hubのドキュメンテーションを参照してください。

TensorFlowおよびFlaxのチェックポイントには影響がなく、`from_pretrained`メソッドの`from_tf`および`from_flax`引数を使用してPyTorchアーキテクチャ内でロードできます。

</Tip>

一般的に、事前学習済みモデルのインスタンスをロードするために`AutoTokenizer`クラスと`AutoModelFor`クラスの使用をお勧めします。
これにより、常に正しいアーキテクチャをロードできます。
次の[tutorial](preprocessing)では、新しくロードしたトークナイザ、画像プロセッサ、特徴量抽出器、およびプロセッサを使用して、ファインチューニング用にデータセットを前処理する方法を学びます。
</pt>
<tf>
最後に、`TFAutoModelFor`クラスは特定のタスクに対して事前学習済みモデルをロードできます（使用可能なタスクの完全な一覧についてはこちらを参照）。
たとえば、[`TFAutoModelForSequenceClassification.from_pretrained`]を使用してシーケンス分類用のモデルをロードできます：

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

同じチェックポイントを再利用して異なるタスクのアーキテクチャをロードできます：

```py
>>> from transformers import TFAutoModelForTokenClassification

>>> model = TFAutoModelForTokenClassification.from_pretrained("distilbert-base-uncased")
```

一般的には、事前学習済みモデルのインスタンスをロードするために`AutoTokenizer`クラスと`TFAutoModelFor`クラスの使用をお勧めします。
これにより、常に正しいアーキテクチャをロードできます。
次の[tutorial](preproccesing)では、新しくロードしたトークナイザ、画像プロセッサ、特徴量抽出器、およびプロセッサを使用して、ファインチューニング用にデータセットを前処理する方法を学びます。
</tf>
</frameworkcontent>