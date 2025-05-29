<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# BROS

## Overview

BROS モデルは、Teakgyu Hon、Donghyun Kim、Mingi Ji, Wonseok Hwang, Daehyun Nam, Sungrae Park によって [BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents](https://arxiv.org/abs/2108.04539) で提案されました。 

BROS は *BERT Relying On Spatality* の略です。これは、一連のトークンとその境界ボックスを入力として受け取り、一連の隠れ状態を出力するエンコーダー専用の Transformer モデルです。 BROS は、絶対的な空間情報を使用する代わりに、相対的な空間情報をエンコードします。

BERT で使用されるトークンマスク言語モデリング目標 (TMLM) と新しいエリアマスク言語モデリング目標 (AMLM) の 2 つの目標で事前トレーニングされています。
TMLM では、トークンはランダムにマスクされ、モデルは空間情報と他のマスクされていないトークンを使用してマスクされたトークンを予測します。
AMLM は TMLM の 2D バージョンです。テキスト トークンをランダムにマスクし、TMLM と同じ情報で予測しますが、テキスト ブロック (領域) をマスクします。

`BrosForTokenClassification`には、BrosModel の上に単純な線形層があります。各トークンのラベルを予測します。
`BrosSpadeEEForTokenClassification`には、BrosModel の上に`initial_token_classifier`と`subsequent_token_classifier`があります。 `initial_token_classifier` は各エンティティの最初のトークンを予測するために使用され、`subsequent_token_classifier` はエンティティ内の次のトークンを予測するために使用されます。 `BrosSpadeELForTokenClassification`には BrosModel の上に`entity_linker`があります。 `entity_linker` は 2 つのエンティティ間の関係を予測するために使用されます。

`BrosForTokenClassification`と`BrosSpadeEEForTokenClassification`は基本的に同じジョブを実行します。ただし、`BrosForTokenClassification`は入力トークンが完全にシリアル化されていることを前提としています (トークンは 2D 空間に存在するため、これは非常に困難な作業です)。一方、`BrosSpadeEEForTokenClassification`は 1 つのトークンから次の接続トークンを予測するため、シリアル化エラーの処理をより柔軟に行うことができます。

`BrosSpadeELForTokenClassification` はエンティティ内のリンク タスクを実行します。これら 2 つのエンティティが何らかの関係を共有する場合、(あるエンティティの) 1 つのトークンから (別のエンティティの) 別のトークンへの関係を予測します。

BROS は、明示的な視覚機能に依存せずに、FUNSD、SROIE、CORD、SciTSR などの Key Information Extraction (KIE) ベンチマークで同等以上の結果を達成します。

論文の要約は次のとおりです。

*文書画像からの重要情報抽出 (KIE) には、2 次元 (2D) 空間におけるテキストの文脈的および空間的意味論を理解する必要があります。最近の研究の多くは、文書画像の視覚的特徴とテキストおよびそのレイアウトを組み合わせることに重点を置いた事前トレーニング済み言語モデルを開発することで、この課題を解決しようとしています。一方、このペーパーでは、テキストとレイアウトの効果的な組み合わせという基本に立ち返ってこの問題に取り組みます。具体的には、BROS (BERT Relying On Spatality) という名前の事前トレーニング済み言語モデルを提案します。この言語モデルは、2D 空間内のテキストの相対位置をエンコードし、エリア マスキング戦略を使用してラベルのないドキュメントから学習します。 2D 空間内のテキストを理解するためのこの最適化されたトレーニング スキームにより、BROS は、視覚的な特徴に依存することなく、4 つの KIE ベンチマーク (FUNSD、SROIE*、CORD、および SciTSR) で以前の方法と比較して同等以上のパフォーマンスを示しました。また、この論文では、KIE タスクにおける 2 つの現実世界の課題 ((1) 間違ったテキスト順序によるエラーの最小化、および (2) 少数の下流例からの効率的な学習) を明らかにし、以前の方法に対する BROS の優位性を実証します。*

このモデルは [jinho8345](https://huggingface.co/jinho8345) によって寄稿されました。元のコードは [ここ](https://github.com/clovaai/bros) にあります。

## Usage tips and examples

- [`~transformers.BrosModel.forward`] には、`input_ids` と `bbox` (バウンディング ボックス) が必要です。各境界ボックスは、(x0、y0、x1、y1) 形式 (左上隅、右下隅) である必要があります。境界ボックスの取得は外部 OCR システムに依存します。 「x」座標はドキュメント画像の幅で正規化する必要があり、「y」座標はドキュメント画像の高さで正規化する必要があります。

```python
def expand_and_normalize_bbox(bboxes, doc_width, doc_height):
    # here, bboxes are numpy array

    # Normalize bbox -> 0 ~ 1
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / width
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / height
```

- [`~transformers.BrosForTokenClassification.forward`、`~transformers.BrosSpadeEEForTokenClassification.forward`、`~transformers.BrosSpadeEEForTokenClassification.forward`] では、損失計算に `input_ids` と `bbox` だけでなく `box_first_token_mask` も必要です。これは、各ボックスの先頭以外のトークンを除外するためのマスクです。このマスクは、単語から `input_ids` を作成するときに境界ボックスの開始トークン インデックスを保存することで取得できます。次のコードで`box_first_token_mask`を作成できます。

```python
def make_box_first_token_mask(bboxes, words, tokenizer, max_seq_length=512):

    box_first_token_mask = np.zeros(max_seq_length, dtype=np.bool_)

    # encode(tokenize) each word from words (List[str])
    input_ids_list: List[List[int]] = [tokenizer.encode(e, add_special_tokens=False) for e in words]

    # get the length of each box
    tokens_length_list: List[int] = [len(l) for l in input_ids_list]

    box_end_token_indices = np.array(list(itertools.accumulate(tokens_length_list)))
    box_start_token_indices = box_end_token_indices - np.array(tokens_length_list)

    # filter out the indices that are out of max_seq_length
    box_end_token_indices = box_end_token_indices[box_end_token_indices < max_seq_length - 1]
    if len(box_start_token_indices) > len(box_end_token_indices):
        box_start_token_indices = box_start_token_indices[: len(box_end_token_indices)]

    # set box_start_token_indices to True
    box_first_token_mask[box_start_token_indices] = True

    return box_first_token_mask

```

## Resources

- デモ スクリプトは [こちら](https://github.com/clovaai/bros) にあります。

## BrosConfig

[[autodoc]] BrosConfig

## BrosProcessor

[[autodoc]] BrosProcessor
    - __call__

## BrosModel

[[autodoc]] BrosModel
    - forward


## BrosForTokenClassification

[[autodoc]] BrosForTokenClassification
    - forward

## BrosSpadeEEForTokenClassification

[[autodoc]] BrosSpadeEEForTokenClassification
    - forward

## BrosSpadeELForTokenClassification

[[autodoc]] BrosSpadeELForTokenClassification
    - forward
