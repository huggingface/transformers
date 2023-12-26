<!--
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ このファイルはMarkdown形式ですが、ドキュメンテーションビルダー用の特定の構文を含んでおり、Markdownビューアーでは正しく表示されないことに注意してください。
-->

# Attention mechanism

ほとんどのTransformerモデルは、アテンション行列が正方形であるという意味で完全なアテンションを使用します。
これは、長いテキストを扱う場合に計算のボトルネックとなることがあります。LongformerやReformerは、より効率的でトレーニングを高速化するためにアテンション行列のスパースバージョンを使用しようとするモデルです。

## LSH attention

[Reformer](#reformer)はLSH（局所的に散在ハッシュ）アテンションを使用します。
ソフトマックス(QK^t)では、行列QK^tの中で（ソフトマックス次元で）最も大きな要素のみが有用な寄与を提供します。
したがって、各クエリqについて、クエリqに近いキーkのみを考慮できます。
qとkが近いかどうかを決定するために、ハッシュ関数が使用されます。
アテンションマスクは変更され、現在のトークンをマスク化します（最初の位置を除く）。
なぜなら、それはクエリとキーが等しい（つまり非常に似ている）クエリとキーを提供するからです。
ハッシュは多少ランダムかもしれないため、実際にはいくつかのハッシュ関数が使用され（n_roundsパラメータで決定されます）、それらが平均化されます。

## Local attention

[Longformer](#longformer)はローカルアテンションを使用します。
しばしば、ローカルコンテキスト（例：左右の2つのトークンは何ですか？）は、特定のトークンに対して行動を起こすのに十分です。
また、小さなウィンドウを持つアテンションレイヤーを積み重ねることで、最後のレイヤーはウィンドウ内のトークンだけでなく、ウィンドウ内のトークンを超えて受容野を持つようになり、文全体の表現を構築できます。

一部の事前選択された入力トークンにはグローバルアテンションも与えられます。
これらの少数のトークンに対して、アテンション行列はすべてのトークンにアクセスでき、このプロセスは対称的です。
他のすべてのトークンは、これらの特定のトークンにアクセスできます（ローカルウィンドウ内のトークンに加えて）。
これは、論文の図2dに示されており、以下はサンプルのアテンションマスクです：

<div class="flex justify-center">
    <img scale="50 %" align="center" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/local_attention_mask.png"/>
</div>


## Other tricks

### Axial positional encodings

[Reformer](#reformer)は軸方向の位置エンコーディングを使用しています。伝統的なトランスフォーマーモデルでは、位置エンコーディングEはサイズが \\(l\\) × \\(d\\) の行列で、\\(l\\) はシーケンスの長さ、\\(d\\) は隠れ状態の次元です。非常に長いテキストを扱う場合、この行列は非常に大きく、GPU上で大量のスペースを占有します。これを緩和するために、軸方向の位置エンコーディングは、この大きな行列Eを2つの小さな行列E1とE2に分解します。それぞれの行列はサイズ \\(l_{1} \times d_{1}\\) および \\(l_{2} \times d_{2}\\) を持ち、 \\(l_{1} \times l_{2} = l\\) および \\(d_{1} + d_{2} = d\\) という条件を満たします（長さの積を考えると、これがはるかに小さくなります）。行列E内の時刻 \\(j\\) の埋め込みは、E1内の時刻 \\(j \% l1\\) の埋め込みとE2内の時刻 \\(j // l1\\) の埋め込みを連結することによって得られます。

