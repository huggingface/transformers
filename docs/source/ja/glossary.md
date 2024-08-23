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

# Glossary

この用語集は、一般的な機械学習と 🤗 トランスフォーマーの用語を定義し、ドキュメンテーションをより理解するのに役立ちます。

## A

### attention mask

アテンション マスクは、シーケンスをバッチ処理する際に使用されるオプションの引数です。

<Youtube id="M6adb1j2jPI"/>

この引数は、モデルにどのトークンを注視すべきか、どのトークンを注視しないかを示します。

例えば、次の2つのシーケンスを考えてみてください：

```python
>>> from transformers import BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

>>> sequence_a = "This is a short sequence."
>>> sequence_b = "This is a rather long sequence. It is at least longer than the sequence A."

>>> encoded_sequence_a = tokenizer(sequence_a)["input_ids"]
>>> encoded_sequence_b = tokenizer(sequence_b)["input_ids"]
```

The encoded versions have different lengths:

```python
>>> len(encoded_sequence_a), len(encoded_sequence_b)
(8, 19)
```

したがって、これらのシーケンスをそのまま同じテンソルに配置することはできません。最初のシーケンスは、
2番目のシーケンスの長さに合わせてパディングする必要があります。または、2番目のシーケンスは、最初のシーケンスの
長さに切り詰める必要があります。

最初の場合、IDのリストはパディングインデックスで拡張されます。トークナイザにリストを渡し、次のようにパディングするように
依頼できます:


```python
>>> padded_sequences = tokenizer([sequence_a, sequence_b], padding=True)
```

0sが追加されて、最初の文が2番目の文と同じ長さになるのがわかります：

```python
>>> padded_sequences["input_ids"]
[[101, 1188, 1110, 170, 1603, 4954, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 1188, 1110, 170, 1897, 1263, 4954, 119, 1135, 1110, 1120, 1655, 2039, 1190, 1103, 4954, 138, 119, 102]]
```

これは、PyTorchまたはTensorFlowでテンソルに変換できます。注意マスクは、モデルがそれらに注意を払わないように、埋め込まれたインデックスの位置を示すバイナリテンソルです。[`BertTokenizer`]では、`1`は注意を払う必要がある値を示し、`0`は埋め込まれた値を示します。この注意マスクは、トークナイザが返す辞書のキー「attention_mask」の下にあります。



```python
>>> padded_sequences["attention_mask"]
[[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
```

### autoencoding models

[エンコーダーモデル](#encoder-models) および [マスク言語モデリング](#masked-language-modeling-mlm) を参照してください。

### autoregressive models

[因果言語モデリング](#causal-language-modeling) および [デコーダーモデル](#decoder-models) を参照してください。

## B

### backbone

バックボーンは、生の隠れた状態や特徴を出力するネットワーク（埋め込みと層）です。通常、特徴を入力として受け取るために [ヘッド](#head) に接続されており、予測を行います。たとえば、[`ViTModel`] は特定のヘッドが上にないバックボーンです。他のモデルも [`VitModel`] をバックボーンとして使用できます、例えば [DPT](model_doc/dpt) です。

## C

### causal language modeling

モデルがテキストを順番に読み、次の単語を予測する事前トレーニングタスクです。通常、モデルは文全体を読み取りますが、特定のタイムステップで未来のトークンを隠すためにモデル内でマスクを使用します。

### channel

カラー画像は、赤、緑、青（RGB）の3つのチャネルの値の組み合わせから成り立っており、グレースケール画像は1つのチャネルしか持ちません。🤗 Transformers では、チャネルは画像のテンソルの最初または最後の次元になることがあります：[`n_channels`, `height`, `width`] または [`height`, `width`, `n_channels`]。

### connectionist temporal classification (CTC)

入力と出力が正確にどのように整列するかを正確に知らなくてもモデルを学習させるアルゴリズム。CTC は、特定の入力に対してすべての可能な出力の分布を計算し、その中から最も可能性の高い出力を選択します。CTC は、スピーカーの異なる発話速度など、さまざまな理由で音声がトランスクリプトと完全に整合しない場合に、音声認識タスクで一般的に使用されます。

### convolution

ニューラルネットワークの一種で、入力行列が要素ごとに小さな行列（カーネルまたはフィルター）と乗算され、値が新しい行列に合計されるレイヤーのタイプ。これは入力行列全体に対して繰り返される畳み込み操作として知られ、各操作は入力行列の異なるセグメントに適用されます。畳み込みニューラルネットワーク（CNN）は、コンピュータビジョンで一般的に使用されています。

## D

### decoder input IDs

この入力はエンコーダーデコーダーモデルに特有であり、デコーダーに供給される入力IDを含みます。これらの入力は、翻訳や要約などのシーケンスツーシーケンスタスクに使用され、通常、各モデルに固有の方法で構築されます。

ほとんどのエンコーダーデコーダーモデル（BART、T5）は、`labels` から独自に `decoder_input_ids` を作成します。このようなモデルでは、`labels` を渡すことがトレーニングを処理する優れた方法です。

シーケンスツーシーケンストレーニングにおけるこれらの入力IDの処理方法を確認するために、各モデルのドキュメントを確認してください。

### decoder models

オートリグレッションモデルとも呼ばれ、モデルがテキストを順番に読み、次の単語を予測する事前トレーニングタスク（因果言語モデリング）に関与します。通常、モデルは文全体を読み取り、特定のタイムステップで未来のトークンを隠すマスクを使用して行われます。

<Youtube id="d_ixlCubqQw"/>


### deep learning (DL)

ニューラルネットワークを使用する機械学習アルゴリズムで、複数の層を持っています。

## E

### encoder models

オートエンコーディングモデルとしても知られており、エンコーダーモデルは入力（テキストや画像など）を、埋め込みと呼ばれる簡略化された数値表現に変換します。エンコーダーモデルは、しばしば[マスクされた言語モデリング（#masked-language-modeling-mlm）](#masked-language-modeling-mlm)などの技術を使用して事前にトレーニングされ、入力シーケンスの一部をマスクし、モデルにより意味のある表現を作成することが強制されます。

<Youtube id="H39Z_720T5s"/>

## F

### feature extraction

生データをより情報豊かで機械学習アルゴリズムにとって有用な特徴のセットに選択および変換するプロセス。特徴抽出の例には、生のテキストを単語埋め込みに変換したり、画像/ビデオデータからエッジや形状などの重要な特徴を抽出したりすることが含まれます。

### feed forward chunking

トランスフォーマー内の各残差注意ブロックでは、通常、自己注意層の後に2つのフィードフォワード層が続きます。
フィードフォワード層の中間埋め込みサイズは、モデルの隠れたサイズよりも大きいことがよくあります（たとえば、`bert-base-uncased`の場合）。

入力サイズが `[batch_size、sequence_length]` の場合、中間フィードフォワード埋め込み `[batch_size、sequence_length、config.intermediate_size]` を保存するために必要なメモリは、メモリの大部分を占めることがあります。[Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451)の著者は、計算が `sequence_length` 次元に依存しないため、両方のフィードフォワード層の出力埋め込み `[batch_size、config.hidden_size]_0、...、[batch_size、config.hidden_size]_n` を個別に計算し、後で `[batch_size、sequence_length、config.hidden_size]` に連結することは数学的に等価であると気付きました。これにより、増加した計算時間とメモリ使用量のトレードオフが生じますが、数学的に等価な結果が得られます。

[`apply_chunking_to_forward`] 関数を使用するモデルの場合、`chunk_size` は並列に計算される出力埋め込みの数を定義し、メモリと時間の複雑さのトレードオフを定義します。`chunk_size` が 0 に設定されている場合、フィードフォワードのチャンキングは行われません。

### finetuned models

ファインチューニングは、事前にトレーニングされたモデルを取り、その重みを固定し、新しく追加された[model head](#head)で出力レイヤーを置き換える形式の転移学習です。モデルヘッドは対象のデータセットでトレーニングされます。

詳細については、[Fine-tune a pretrained model](https://huggingface.co/docs/transformers/training) チュートリアルを参照して、🤗 Transformersを使用したモデルのファインチューニング方法を学びましょう。

## H

### head

モデルヘッドは、ニューラルネットワークの最後のレイヤーを指し、生の隠れた状態を受け入れて異なる次元に射影します。各タスクに対して異なるモデルヘッドがあります。例えば：

  * [`GPT2ForSequenceClassification`] は、ベースの[`GPT2Model`]の上にあるシーケンス分類ヘッド（線形層）です。
  * [`ViTForImageClassification`] は、ベースの[`ViTModel`]の`CLS`トークンの最終隠れた状態の上にある画像分類ヘッド（線形層）です。
  * [`Wav2Vec2ForCTC`] は、[CTC](#connectionist-temporal-classification-(CTC))を持つベースの[`Wav2Vec2Model`]の言語モデリングヘッドです。

## I

### image patch

ビジョンベースのトランスフォーマーモデルは、画像をより小さなパッチに分割し、それらを線形に埋め込み、モデルにシーケンスとして渡します。モデルの

### inference

推論は、トレーニングが完了した後に新しいデータでモデルを評価するプロセスです。 🤗 Transformers を使用して推論を実行する方法については、[推論のパイプライン](https://huggingface.co/docs/transformers/pipeline_tutorial) チュートリアルを参照してください。

### input IDs

入力IDは、モデルへの入力として渡す必要があるパラメーターの中で最も一般的なものです。これらはトークンのインデックスであり、モデルによって入力として使用されるシーケンスを構築するトークンの数値表現です。

<Youtube id="VFp38yj8h3A"/>

各トークナイザーは異なる方法で動作しますが、基本的なメカニズムは同じです。以下はBERTトークナイザーを使用した例です。BERTトークナイザーは[WordPiece](https://arxiv.org/pdf/1609.08144.pdf)トークナイザーです。


```python
>>> from transformers import BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

>>> sequence = "A Titan RTX has 24GB of VRAM"
```

トークナイザーは、シーケンスをトークナイザー語彙で使用可能なトークンに分割します。

```python
>>> tokenized_sequence = tokenizer.tokenize(sequence)
```

トークンは単語またはサブワードです。 たとえば、ここでは "VRAM" はモデルの語彙に含まれていなかったため、"V"、"RA"、"M" に分割されました。
これらのトークンが別々の単語ではなく、同じ単語の一部であることを示すために、"RA" と "M" にはダブルハッシュのプレフィックスが追加されます。

```python
>>> print(tokenized_sequence)
['A', 'Titan', 'R', '##T', '##X', 'has', '24', '##GB', 'of', 'V', '##RA', '##M']
```

これらのトークンは、モデルが理解できるようにIDに変換できます。これは、文をトークナイザーに直接供給して行うことができます。トークナイザーは、パフォーマンスの向上のために[🤗 Tokenizers](https://github.com/huggingface/tokenizers)のRust実装を活用しています。

```python
>>> inputs = tokenizer(sequence)
```

トークナイザーは、対応するモデルが正しく動作するために必要なすべての引数を含む辞書を返します。トークンのインデックスは、キー `input_ids` の下にあります。

```python
>>> encoded_sequence = inputs["input_ids"]
>>> print(encoded_sequence)
[101, 138, 18696, 155, 1942, 3190, 1144, 1572, 13745, 1104, 159, 9664, 2107, 102]
```

注意：トークナイザは、関連するモデルがそれらを必要とする場合に自動的に「特別なトークン」を追加します。これらは、モデルが時折使用する特別なIDです。

前のIDシーケンスをデコードする場合、


```python
>>> decoded_sequence = tokenizer.decode(encoded_sequence)
```

私たちは見ます

```python
>>> print(decoded_sequence)
[CLS] A Titan RTX has 24GB of VRAM [SEP]
```

これは[`BertModel`]がその入力を期待する方法です。


## L

### Labels

ラベルは、モデルが損失を計算するために渡すことができるオプションの引数です。これらのラベルは、モデルの予測の期待値であるべきです。モデルは、通常の損失を使用して、その予測と期待値（ラベル）との間の損失を計算します。

これらのラベルはモデルのヘッドに応じて異なります。たとえば：

- シーケンス分類モデル（[`BertForSequenceClassification`]）の場合、モデルは次元が `(batch_size)` のテンソルを期待し、バッチ内の各値がシーケンス全体の予測ラベルに対応します。
- トークン分類モデル（[`BertForTokenClassification`]）の場合、モデルは次元が `(batch_size, seq_length)` のテンソルを期待し、各値が各個々のトークンの予測ラベルに対応します。
- マスク言語モデリングの場合（[`BertForMaskedLM`]）、モデルは次元が `(batch_size, seq_length)` のテンソルを期待し、各値が各個々のトークンの予測ラベルに対応します。ここでのラベルはマスクされたトークンのトークンIDであり、他のトークンには通常 -100 などの値が設定されます。
- シーケンス間のタスクの場合（[`BartForConditionalGeneration`]、[`MBartForConditionalGeneration`]）、モデルは次元が `(batch_size, tgt_seq_length)` のテンソルを期待し、各値が各入力シーケンスに関連付けられたターゲットシーケンスに対応します。トレーニング中、BARTとT5の両方は適切な `decoder_input_ids` とデコーダーのアテンションマスクを内部で生成します。通常、これらを提供する必要はありません。これはエンコーダーデコーダーフレームワークを利用するモデルには適用されません。
- 画像分類モデルの場合（[`ViTForImageClassification`]）、モデルは次元が `(batch_size)` のテンソルを期待し、バッチ内の各値が各個々の画像の予測ラベルに対応します。
- セマンティックセグメンテーションモデルの場合（[`SegformerForSemanticSegmentation`]）、モデルは次元が `(batch_size, height, width)` のテンソルを期待し、バッチ内の各値が各個々のピクセルの予測ラベルに対応します。
- 物体検出モデルの場合（[`DetrForObjectDetection`]）、モデルは各個々の画像の予測ラベルと境界ボックスの数に対応する `class_labels` と `boxes` キーを持つ辞書のリストを期待します。
- 自動音声認識モデルの場合（[`Wav2Vec2ForCTC`]）、モデルは次元が `(batch_size, target_length)` のテンソルを期待し、各値が各個々のトークンの予測ラベルに対応します。

<Tip>

各モデルのラベルは異なる場合があるため、常に各モデルのドキュメントを確認して、それらの特定のラベルに関する詳細情報を確認してください！

</Tip>

ベースモデル（[`BertModel`]）はラベルを受け入れません。これらはベースのトランスフォーマーモデルであり、単に特徴を出力します。


### large language models (LLM)

大量のデータでトレーニングされた変換器言語モデル（GPT-3、BLOOM、OPT）を指す一般的な用語です。これらのモデルは通常、多くの学習可能なパラメータを持っています（たとえば、GPT-3の場合、1750億個）。

## M

### masked language modeling (MLM)

モデルはテキストの破損バージョンを見る事前トレーニングタスクで、通常はランダムに一部のトークンをマスキングして元のテキストを予測する必要があります。

### multimodal

テキストと別の種類の入力（たとえば画像）を組み合わせるタスクです。

## N

### Natural language generation (NLG)

テキストを生成する関連するすべてのタスク（たとえば、[Transformersで書く](https://transformer.huggingface.co/)、翻訳など）。

### Natural language processing (NLP)

テキストを扱う方法を一般的に表現したものです。

### Natural language understanding (NLU)

テキスト内に何があるかを理解する関連するすべてのタスク（たとえば、テキスト全体の分類、個々の単語の分類など）。

## P

### pipeline

🤗 Transformersのパイプラインは、データの前処理と変換を特定の順序で実行してデータを処理し、モデルから予測を返す一連のステップを指す抽象化です。パイプラインに見られるいくつかのステージの例には、データの前処理、特徴抽出、正規化などがあります。

詳細については、[推論のためのパイプライン](https://huggingface.co/docs/transformers/pipeline_tutorial)を参照してください。

### pixel values

モデルに渡される画像の数値表現のテンソルです。ピクセル値は、形状が [`バッチサイズ`, `チャネル数`, `高さ`, `幅`] の行列で、画像プロセッサから生成されます。

### pooling

行列を小さな行列に縮小する操作で、プール対象の次元の最大値または平均値を取ることが一般的です。プーリングレイヤーは一般的に畳み込みレイヤーの間に見られ、特徴表現をダウンサンプリングします。

### position IDs

トークンごとの位置が埋め込まれているRNNとは異なり、トランスフォーマーは各トークンの位置を把握していません。したがって、モデルはトークンの位置を識別するために位置ID（`position_ids`）を使用します。

これはオプションのパラメータです。モデルに `position_ids` が渡されない場合、IDは自動的に絶対的な位置埋め込みとして作成されます。

絶対的な位置埋め込みは範囲 `[0、config.max_position_embeddings - 1]` から選択されます。一部のモデルは、正弦波位置埋め込みや相対位置埋め込みなど、他のタイプの位置埋め込みを使用することがあります。


### preprocessing

生データを機械学習モデルで簡単に処理できる形式に準備するタスクです。例えば、テキストは通常、トークン化によって前処理されます。他の入力タイプに対する前処理の具体的な方法を知りたい場合は、[Preprocess](https://huggingface.co/docs/transformers/preprocessing) チュートリアルをご覧ください。

### pretrained model

あるデータ（たとえば、Wikipedia全体など）で事前に学習されたモデルです。事前学習の方法には、自己教師ありの目的が含まれ、テキストを読み取り、次の単語を予測しようとするもの（[因果言語モデリング](#因果言語モデリング)を参照）や、一部の単語をマスクし、それらを予測しようとするもの（[マスク言語モデリング](#マスク言語モデリング-mlm)を参照）があります。

音声とビジョンモデルには独自の事前学習の目的があります。たとえば、Wav2Vec2は音声モデルで、モデルに対して「真の」音声表現を偽の音声表現のセットから識別する必要がある対比的なタスクで事前学習されています。一方、BEiTはビジョンモデルで、一部の画像パッチをマスクし、モデルにマスクされたパッチを予測させるタスク（マスク言語モデリングの目的と似ています）で事前学習されています。

## R

### recurrent neural network (RNN)

テキストを処理するために層をループさせるモデルの一種です。

### representation learning

生データの意味のある表現を学習する機械学習のサブフィールドです。表現学習の技術の一部には単語埋め込み、オートエンコーダー、Generative Adversarial Networks（GANs）などがあります。

## S

### sampling rate

秒ごとに取られるサンプル（オーディオ信号など）の数をヘルツ単位で測定したものです。サンプリングレートは音声などの連続信号を離散化する結果です。

### self-attention

入力の各要素は、どの他の要素に注意を払うべきかを検出します。

### self-supervised learning 

モデルがラベルのないデータから自分自身の学習目標を作成する機械学習技術のカテゴリです。これは[教師なし学習](#教師なし学習)や[教師あり学習](#教師あり学習)とは異なり、学習プロセスはユーザーからは明示的には監督されていない点が異なります。

自己教師あり学習の1つの例は[マスク言語モデリング](#マスク言語モデリング-mlm)で、モデルには一部のトークンが削除された文が与えられ、欠落したトークンを予測するように学習します。

### semi-supervised learning

ラベル付きデータの少量とラベルのないデータの大量を組み合わせてモデルの精度を向上させる広範な機械学習トレーニング技術のカテゴリです。[教師あり学習](#教師あり学習)や[教師なし学習](#教師なし学習)とは異なり、半教師あり学習のアプローチの1つは「セルフトレーニング」であり、モデルはラベル付きデータでトレーニングされ、次にラベルのないデータで予測を行います。モデルが最も自信を持って予測する部分がラベル付きデータセットに追加され、モデルの再トレーニングに使用されます。

### sequence-to-sequence (seq2seq)

入力から新しいシーケンスを生成するモデルです。翻訳モデルや要約モデル（[Bart](model_doc/bart)や[T5](model_doc/t5)など）などがこれに該当します。

### stride

[畳み込み](#畳み込み)または[プーリング](#プーリング)において、ストライドはカーネルが行列上で移動する距離を指します。ストライドが1の場合、カーネルは1ピクセルずつ移動し、ストライドが2の場合、カーネルは2ピクセルずつ移動します。

### supervised learning

モデルのトレーニング方法の一つで、直接ラベル付きデータを使用してモデルの性能を修正し指導します。データがトレーニングされているモデルに供給され、その予測が既知のラベルと比較されます。モデルは予測がどれだけ誤っていたかに基づいて重みを更新し、プロセスはモデルの性能を最適化するために繰り返されます。

## T

### token

文の一部であり、通常は単語ですが、サブワード（一般的でない単語はしばしばサブワードに分割されることがあります）または句読点の記号であることもあります。

### token Type IDs

一部のモデルは、文のペアの分類や質問応答を行うことを目的としています。

<Youtube id="0u3ioSwev3s"/>

これには異なる2つのシーケンスを単一の「input_ids」エントリに結合する必要があり、通常は分類子（`[CLS]`）や区切り記号（`[SEP]`）などの特別なトークンの助けを借りて実行されます。例えば、BERTモデルは次のように2つのシーケンス入力を構築します：

日本語訳を提供していただきたいです。Markdown形式で記述してください。


```python
>>> # [CLS] SEQUENCE_A [SEP] SEQUENCE_B [SEP]
```

我々は、前述のように、2つのシーケンスを2つの引数として `tokenizer` に渡すことで、このような文を自動的に生成することができます（以前のようにリストではなく）。以下のように：

```python
>>> from transformers import BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
>>> sequence_a = "HuggingFace is based in NYC"
>>> sequence_b = "Where is HuggingFace based?"

>>> encoded_dict = tokenizer(sequence_a, sequence_b)
>>> decoded = tokenizer.decode(encoded_dict["input_ids"])
```

これに対応するコードは以下です：

```python
>>> print(decoded)
[CLS] HuggingFace is based in NYC [SEP] Where is HuggingFace based? [SEP]
```

一部のモデルでは、1つのシーケンスがどこで終わり、別のシーケンスがどこで始まるかを理解するのに十分な情報が備わっています。ただし、BERTなどの他のモデルでは、トークンタイプID（セグメントIDとも呼ばれる）も使用されています。これは、モデル内の2つのシーケンスを識別するバイナリマスクとして表されます。

トークナイザは、このマスクを「token_type_ids」として返します。

```python
>>> encoded_dict["token_type_ids"]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

最初のシーケンス、つまり質問のために使用される「コンテキスト」は、すべてのトークンが「0」で表されています。一方、2番目のシーケンス、質問に対応するものは、すべてのトークンが「1」で表されています。

一部のモデル、例えば [`XLNetModel`] のように、追加のトークンが「2」で表されます。


### transfer learning

事前に学習されたモデルを取り、それをタスク固有のデータセットに適応させる技術。ゼロからモデルを訓練する代わりに、既存のモデルから得た知識を出発点として活用できます。これにより学習プロセスが加速し、必要な訓練データの量が減少します。

### transformer

自己注意ベースの深層学習モデルアーキテクチャ。

## U

### unsupervised learning

モデルに提供されるデータがラベル付けされていないモデルトレーニングの形態。教師なし学習の技術は、タスクに役立つパターンを見つけるためにデータ分布の統計情報を活用します。
