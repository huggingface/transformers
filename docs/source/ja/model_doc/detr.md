<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# DETR

## Overview

DETR モデルは、[Transformers を使用したエンドツーエンドのオブジェクト検出](https://huggingface.co/papers/2005.12872) で提案されました。
Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov and Sergey Zagoruyko ルイコ。 DETR
畳み込みバックボーンと、その後にエンドツーエンドでトレーニングできるエンコーダー/デコーダー Transformer で構成されます。
物体の検出。 Faster-R-CNN や Mask-R-CNN などのモデルの複雑さの多くが大幅に簡素化されます。
領域提案、非最大抑制手順、アンカー生成などです。さらに、DETR は次のようにすることもできます。
デコーダ出力の上にマスク ヘッドを追加するだけで、パノプティック セグメンテーションを実行できるように自然に拡張されています。

論文の要約は次のとおりです。

*物体検出を直接集合予測問題として見る新しい方法を紹介します。私たちのアプローチは、
検出パイプラインにより、非最大抑制などの多くの手作業で設計されたコンポーネントの必要性が効果的に排除されます。
タスクに関する事前の知識を明示的にエンコードするプロシージャまたはアンカーの生成。の主な成分は、
DEtection TRansformer または DETR と呼ばれる新しいフレームワークは、セットベースのグローバル損失であり、
二部マッチング、およびトランスフォーマー エンコーダー/デコーダー アーキテクチャ。学習されたオブジェクト クエリの固定された小さなセットが与えられると、
DETR は、オブジェクトとグローバル イメージ コンテキストの関係について推論し、最終セットを直接出力します。
並行して予想も。新しいモデルは概念的にシンプルであり、多くのモデルとは異なり、特殊なライブラリを必要としません。
他の最新の検出器。 DETR は、確立された、および同等の精度と実行時のパフォーマンスを実証します。
困難な COCO 物体検出データセットに基づく、高度に最適化された Faster RCNN ベースライン。さらに、DETR は簡単に実行できます。
統一された方法でパノプティック セグメンテーションを生成するために一般化されました。競合他社を大幅に上回るパフォーマンスを示しています
ベースライン*

このモデルは、[nielsr](https://huggingface.co/nielsr) によって提供されました。元のコードは [こちら](https://github.com/facebookresearch/detr) にあります。

## How DETR works

[`~transformers.DetrForObjectDetection`] がどのように機能するかを説明する TLDR は次のとおりです。

まず、事前にトレーニングされた畳み込みバックボーンを通じて画像が送信されます (論文では、著者らは次のように使用しています)。
ResNet-50/ResNet-101)。バッチ ディメンションも追加すると仮定します。これは、バックボーンへの入力が
画像に 3 つのカラー チャネル (RGB) があると仮定した場合の、形状 `(batch_size, 3, height, width)` のテンソル。 CNNのバックボーン
通常は `(batch_size, 2048, height/32, width/32)` の形状の、新しい低解像度の特徴マップを出力します。これは
次に、DETR の Transformer の隠れ次元 (デフォルトでは `256`) に一致するように投影されます。
`nn.Conv2D` レイヤー。これで、形状 `(batch_size, 256, height/32, width/32)` のテンソルが完成しました。
特徴マップは平坦化および転置され、形状 `(batch_size, seq_len, d_model)` のテンソルを取得します =
`(batch_size, width/32*height/32, 256)`。したがって、NLP モデルとの違いは、シーケンスの長さが実際には
通常よりも長くなりますが、「d_model」は小さくなります (NLP では通常 768 以上です)。

次に、これがエンコーダを介して送信され、同じ形状の `encoder_hidden_​​states` が出力されます (次のように考えることができます)。
これらは画像の特徴として）。次に、いわゆる **オブジェクト クエリ**がデコーダを通じて送信されます。これは形状のテンソルです
`(batch_size, num_queries, d_model)`。通常、`num_queries` は 100 に設定され、ゼロで初期化されます。
これらの入力埋め込みは学習された位置エンコーディングであり、作成者はこれをオブジェクト クエリと呼び、同様に
エンコーダでは、それらは各アテンション層の入力に追加されます。各オブジェクト クエリは特定のオブジェクトを検索します。
画像では。デコーダは、複数のセルフ アテンション レイヤとエンコーダ デコーダ アテンション レイヤを通じてこれらの埋め込みを更新します。
同じ形状の `decoder_hidden_​​states` を出力します: `(batch_size, num_queries, d_model)`。次に頭が２つ
オブジェクト検出のために上部に追加されます。各オブジェクト クエリをオブジェクトの 1 つに分類するための線形レイヤー、または「いいえ」
オブジェクト」、および各クエリの境界ボックスを予測する MLP。

モデルは **2 部マッチング損失**を使用してトレーニングされます。つまり、実際に行うことは、予測されたクラスを比較することです +
グラウンド トゥルース アノテーションに対する N = 100 個の各オブジェクト クエリの境界ボックス (同じ長さ N までパディング)
(したがって、画像にオブジェクトが 4 つしか含まれていない場合、96 個の注釈にはクラスとして「オブジェクトなし」、およびクラスとして「境界ボックスなし」が含まれるだけになります。
境界ボックス)。 [Hungarian matching algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) は、検索に使用されます。
N 個のクエリのそれぞれから N 個の注釈のそれぞれへの最適な 1 対 1 のマッピング。次に、標準クロスエントロピー (
クラス)、および L1 と [generalized IoU loss](https://giou.stanford.edu/) の線形結合 (
境界ボックス) は、モデルのパラメーターを最適化するために使用されます。

DETR は、パノプティック セグメンテーション (セマンティック セグメンテーションとインスタンスを統合する) を実行するように自然に拡張できます。
セグメンテーション）。 [`~transformers.DetrForSegmentation`] はセグメンテーション マスク ヘッドを上に追加します
[`~transformers.DetrForObjectDetection`]。マスク ヘッドは、共同でトレーニングすることも、2 段階のプロセスでトレーニングすることもできます。
ここで、最初に [`~transformers.DetrForObjectDetection`] モデルをトレーニングして、両方の周囲の境界ボックスを検出します。
「もの」（インスタンス）と「もの」（木、道路、空などの背景のもの）をすべて凍結し、すべての重みをフリーズしてのみトレーニングします。
25 エポックのマスクヘッド。実験的には、これら 2 つのアプローチは同様の結果をもたらします。ボックスの予測は
ハンガリー語のマッチングはボックス間の距離を使用して計算されるため、トレーニングを可能にするためにはこれが必要です。

## Usage tips

- DETR は、いわゆる **オブジェクト クエリ** を使用して、画像内のオブジェクトを検出します。クエリの数によって最大値が決まります
  単一の画像内で検出できるオブジェクトの数。デフォルトでは 100 に設定されます (パラメーターを参照)
  [`~transformers.DetrConfig`] の `num_queries`)。ある程度の余裕があるのは良いことです (COCO では、
  著者は 100 を使用しましたが、COCO イメージ内のオブジェクトの最大数は約 70 です)。
- DETR のデコーダーは、クエリの埋め込みを並行して更新します。これは GPT-2 のような言語モデルとは異なります。
  並列ではなく自己回帰デコードを使用します。したがって、因果的注意マスクは使用されません。
- DETR は、投影前に各セルフアテンション層とクロスアテンション層の隠れ状態に位置埋め込みを追加します。
  クエリとキーに。画像の位置埋め込みについては、固定正弦波または学習済みのどちらかを選択できます。
  絶対位置埋め込み。デフォルトでは、パラメータ `position_embedding_type` は
  [`~transformers.DetrConfig`] は `"sine"` に設定されます。
- DETR の作成者は、トレーニング中に、特にデコーダで補助損失を使用すると役立つことに気づきました。
  モデルは各クラスの正しい数のオブジェクトを出力します。パラメータ `auxiliary_loss` を設定すると、
  [`~transformers.DetrConfig`] を`True`に設定し、フィードフォワード ニューラル ネットワークとハンガリー損失を予測します
  は各デコーダ層の後に追加されます (FFN がパラメータを共有する)。
- 複数のノードにわたる分散環境でモデルをトレーニングする場合は、
  _modeling_detr.py_ の _DetrLoss_ クラスの _num_boxes_ 変数。複数のノードでトレーニングする場合、これは次のようにする必要があります
  元の実装で見られるように、すべてのノードにわたるターゲット ボックスの平均数に設定されます [こちら](https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/models/detr.py#L227-L232) 。
- [`~transformers.DetrForObjectDetection`] および [`~transformers.DetrForSegmentation`] は次のように初期化できます。
  [timm ライブラリ](https://github.com/rwightman/pytorch-image-models) で利用可能な畳み込みバックボーン。
  たとえば、MobileNet バックボーンを使用した初期化は、次の `backbone` 属性を設定することで実行できます。
  [`~transformers.DetrConfig`] を `"tf_mobilenetv3_small_075"` に設定し、それを使用してモデルを初期化します。
  構成。
- DETR は、最短辺が一定のピクセル数以上になり、最長辺が一定量以上になるように入力画像のサイズを変更します。
  最大 1333 ピクセル。トレーニング時に、最短辺がランダムに に設定されるようにスケール拡張が使用されます。
  最小 480、最大 800 ピクセル。推論時には、最短辺が 800 に設定されます。

使用できます
  [`~transformers.DetrImageProcessor`] 用の画像 (およびオプションの COCO 形式の注釈) を準備します。
  モデル。このサイズ変更により、バッチ内の画像のサイズが異なる場合があります。 DETR は、画像を最大までパディングすることでこの問題を解決します。
  どのピクセルが実数でどのピクセルがパディングであるかを示すピクセル マスクを作成することによって、バッチ内の最大サイズを決定します。
  あるいは、画像をバッチ処理するためにカスタムの `collat​​e_fn` を定義することもできます。
  [`~transformers.DetrImageProcessor.pad_and_create_pixel_mask`]。
- 画像のサイズによって使用されるメモリの量が決まり、したがって「batch_size」も決まります。
  GPU あたり 2 のバッチ サイズを使用することをお勧めします。詳細については、[この Github スレッド](https://github.com/facebookresearch/detr/issues/150) を参照してください。

DETR モデルをインスタンス化するには 3 つの方法があります (好みに応じて)。

オプション 1: モデル全体の事前トレーニングされた重みを使用して DETR をインスタンス化する

```py
>>> from transformers import DetrForObjectDetection

>>> model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
```

オプション 2: Transformer についてはランダムに初期化された重みを使用して DETR をインスタンス化しますが、バックボーンについては事前にトレーニングされた重みを使用します

```py
>>> from transformers import DetrConfig, DetrForObjectDetection

>>> config = DetrConfig()
>>> model = DetrForObjectDetection(config)
```

オプション 3: バックボーン + トランスフォーマーのランダムに初期化された重みを使用して DETR をインスタンス化します。

```py
>>> config = DetrConfig()
>>> model = DetrForObjectDetection(config)
```

| Task | Object detection | Instance segmentation | Panoptic segmentation |
|------|------------------|-----------------------|-----------------------|
| **Description** |画像内のオブジェクトの周囲の境界ボックスとクラス ラベルを予測する | 画像内のオブジェクト (つまりインスタンス) の周囲のマスクを予測する | 画像内のオブジェクト (インスタンス) と「もの」 (木や道路などの背景) の両方の周囲のマスクを予測します |
| **Model** | [`~transformers.DetrForObjectDetection`] | [`~transformers.DetrForSegmentation`] | [`~transformers.DetrForSegmentation`] |
| **Example dataset** | COCO detection | COCO detection, COCO panoptic | COCO panoptic  |                                                                        |
| **Format of annotations to provide to**  [`~transformers.DetrImageProcessor`] | {'image_id': `int`, 'annotations': `list[Dict]`} each Dict being a COCO object annotation  | {'image_id': `int`, 'annotations': `list[Dict]`}  (in case of COCO detection) or {'file_name': `str`, 'image_id': `int`, 'segments_info': `list[Dict]`} (in case of COCO panoptic) | {'file_name': `str`, 'image_id': `int`, 'segments_info': `list[Dict]`} and masks_path (path to directory containing PNG files of the masks) |
| **Postprocessing** (i.e. converting the output of the model to Pascal VOC format) | [`~transformers.DetrImageProcessor.post_process`] | [`~transformers.DetrImageProcessor.post_process_segmentation`] | [`~transformers.DetrImageProcessor.post_process_segmentation`], [`~transformers.DetrImageProcessor.post_process_panoptic`] |
| **evaluators** | `CocoEvaluator` with `iou_types="bbox"` | `CocoEvaluator` with `iou_types="bbox"` or `"segm"` | `CocoEvaluator` with `iou_tupes="bbox"` or `"segm"`, `PanopticEvaluator` |

つまり、COCO 検出または COCO パノプティック形式でデータを準備してから、次を使用する必要があります。
[`~transformers.DetrImageProcessor`] `pixel_values`、`pixel_mask`、およびオプションを作成します。
「ラベル」。これを使用してモデルをトレーニング (または微調整) できます。評価するには、まず、
[`~transformers.DetrImageProcessor`] の後処理メソッドの 1 つを使用したモデルの出力。これらはできます
`CocoEvaluator` または `PanopticEvaluator` のいずれかに提供され、次のようなメトリクスを計算できます。
平均平均精度 (mAP) とパノラマ品質 (PQ)。後者のオブジェクトは [元のリポジトリ](https://github.com/facebookresearch/detr) に実装されています。評価の詳細については、[サンプル ノートブック](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETR) を参照してください。

## Resources

DETR の使用を開始するのに役立つ公式 Hugging Face およびコミュニティ (🌎 で示されている) リソースのリスト。

<PipelineTag pipeline="object-detection"/>

- カスタム データセットの [`DetrForObjectDetection`] と [`DetrForSegmentation`] の微調整を説明するすべてのサンプル ノートブックは、[こちら](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETR) で見つけることができます。 。
- 参照: [オブジェクト検出タスク ガイド](../tasks/object_detection)

ここに含めるリソースの送信に興味がある場合は、お気軽にプル リクエストを開いてください。審査させていただきます。リソースは、既存のリソースを複製するのではなく、何か新しいものを示すことが理想的です。

## DetrConfig

[[autodoc]] DetrConfig

## DetrImageProcessor

[[autodoc]] DetrImageProcessor
    - preprocess
    - post_process_object_detection
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## DetrImageProcessor

[[autodoc]] DetrImageProcessor
    - preprocess
    - post_process_object_detection
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## DETR specific outputs

[[autodoc]] models.detr.modeling_detr.DetrModelOutput

[[autodoc]] models.detr.modeling_detr.DetrObjectDetectionOutput

[[autodoc]] models.detr.modeling_detr.DetrSegmentationOutput

## DetrModel

[[autodoc]] DetrModel
    - forward

## DetrForObjectDetection

[[autodoc]] DetrForObjectDetection
    - forward

## DetrForSegmentation

[[autodoc]] DetrForSegmentation
    - forward
