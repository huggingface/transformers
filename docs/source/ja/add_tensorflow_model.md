<!--
Copyright 2023 The HuggingFace Team. All rights reserved.

ライセンス：Apache License、バージョン2.0（「ライセンス」）に基づいています。このファイルは、ライセンスに準拠していない限り、使用できません。ライセンスのコピーは以下から入手できます：

http://www.apache.org/licenses/LICENSE-2.0

適用法に従って必要な場合、または書面で同意した場合を除き、ライセンスの下で配布されるソフトウェアは、「AS IS」の基盤で、明示または黙示を問わず、いかなる保証や条件も含みません。ライセンスの詳細については、ライセンス文書をご覧ください。

⚠️ このファイルはMarkdown形式ですが、弊社のドキュメントビルダー（MDXに類似した特定の構文を含む）を含むため、お使いのMarkdownビューアでは正しく表示されない場合があります。

-->


# How to convert a 🤗 Transformers model to TensorFlow?

🤗 Transformersを使用するために複数のフレームワークが利用可能であることは、アプリケーションを設計する際にそれぞれの強みを活かす柔軟性を提供しますが、
互換性をモデルごとに追加する必要があることを意味します。しかし、幸いなことに
既存のモデルにTensorFlow互換性を追加することは、[ゼロから新しいモデルを追加すること](add_new_model)よりも簡単です！
大規模なTensorFlowモデルの詳細を理解したり、主要なオープンソースの貢献を行ったり、
選択したモデルをTensorFlowで有効にするためのガイドです。

このガイドは、コミュニティのメンバーであるあなたに、TensorFlowモデルの重みおよび/または
アーキテクチャを🤗 Transformersで使用するために、Hugging Faceチームからの最小限の監視で貢献できる力を与えます。新しいモデルを書くことは小さな偉業ではありませんが、
このガイドを読むことで、それがローラーコースターのようなものから散歩のようなものになることを願っています🎢🚶。
このプロセスをますます簡単にするために、私たちの共通の経験を活用することは非常に重要ですので、
このガイドの改善を提案することを強くお勧めします！

さらに詳しく調べる前に、以下のリソースをチェックすることをお勧めします。🤗 Transformersが初めての場合：

- [🤗 Transformersの一般的な概要](add_new_model#general-overview-of-transformers)
- [Hugging FaceのTensorFlow哲学](https://huggingface.co/blog/tensorflow-philosophy)

このガイドの残りの部分では、新しいTensorFlowモデルアーキテクチャを追加するために必要なもの、
PyTorchをTensorFlowモデルの重みに変換する手順、およびMLフレームワーク間の不一致を効率的にデバッグする方法について学びます。それでは始めましょう！

<Tip>

使用したいモデルに対応するTensorFlowアーキテクチャがすでに存在するかどうかわからないですか？

&nbsp;

選択したモデルの`config.json`の`model_type`フィールドをチェックしてみてください
（[例](https://huggingface.co/bert-base-uncased/blob/main/config.json#L14)）。
🤗 Transformersの該当するモデルフォルダに、名前が"modeling_tf"で始まるファイルがある場合、それは対応するTensorFlow
アーキテクチャを持っていることを意味します（[例](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert)）。

</Tip>

## Step-by-step guide to add TensorFlow model architecture code

大規模なモデルアーキテクチャを設計する方法はさまざまであり、その設計を実装する方法もさまざまです。
しかし、[🤗 Transformersの一般的な概要](add_new_model#general-overview-of-transformers)から
思い出していただけるかもしれませんが、私たちは意見のあるグループです - 🤗 Transformersの使いやすさは一貫性のある設計の選択肢に依存しています。経験から、TensorFlowモデルを追加する際に重要なことをいくつかお伝えできます：

- 車輪を再発明しないでください！ほとんどの場合、確認すべき少なくとも2つの参照実装があります。それは、
あなたが実装しているモデルのPyTorchバージョンと、同じ種類の問題に対する他のTensorFlowモデルです。
- 優れたモデル実装は時間の試練を乗り越えます。これは、コードがきれいだからではなく、コードが明確で、デバッグしやすく、
構築しやすいからです。TensorFlow実装でPyTorch実装と一致するパターンを複製し、PyTorch実装との不一致を最小限に抑えることで、
あなたの貢献が長期間にわたって有用であることを保証します。
- 行き詰まったら助けを求めてください！ 🤗 Transformersチームはここにいますし、おそらくあなたが直面している同じ問題に対する解決策を見つけています。

TensorFlowモデルアーキテクチャを追加するために必要なステップの概要は次のとおりです：
1. 変換したいモデルを選択
2. transformersの開発環境を準備
3. （オプション）理論的な側面と既存の実装を理解
4. モデルアーキテクチャを実装
5. モデルのテストを実装
6. プルリクエストを提出
7. （オプション）デモを構築して世界と共有

### 1.-3. Prepare your model contribution

**1. 変換したいモデルを選択する**

まず、基本から始めましょう。最初に知っておく必要があることは、変換したいアーキテクチャです。
特定のアーキテクチャを決めていない場合、🤗 Transformers チームに提案を求めることは、影響を最大限にする素晴らしい方法です。
チームは、TensorFlow サイドで不足している最も注目されるアーキテクチャに向けてガイドします。
TensorFlow で使用したい特定のモデルに、🤗 Transformers に既に TensorFlow アーキテクチャの実装が存在しているが、重みが不足している場合、
このページの[重みの追加セクション](#adding-tensorflow-weights-to-hub)に直接移動してください。

簡単にするために、このガイドの残りの部分では、TensorFlow バージョンの *BrandNewBert* を貢献することを決定したと仮定しています
（これは、[新しいモデルの追加ガイド](add_new_model)での例と同じです）。

<Tip>

TensorFlow モデルのアーキテクチャに取り組む前に、それを行うための進行中の取り組みがないかを再確認してください。
GitHub ページの[プルリクエスト](https://github.com/huggingface/transformers/pulls?q=is%3Apr)で `BrandNewBert` を検索して、
TensorFlow 関連のプルリクエストがないことを確認できます。

</Tip>


**2. transformers 開発環境の準備**

モデルアーキテクチャを選択したら、意向を示すためにドラフト PR を開くための環境を設定してください。
以下の手順に従って、環境を設定し、ドラフト PR を開いてください。

1. リポジトリのページで 'Fork' ボタンをクリックして、[リポジトリ](https://github.com/huggingface/transformers)をフォークします。
   これにより、コードのコピーが GitHub ユーザーアカウントの下に作成されます。

2. ローカルディスクにある 'transformers' フォークをクローンし、ベースリポジトリをリモートとして追加します:

```bash
git clone https://github.com/[your Github handle]/transformers.git
cd transformers
git remote add upstream https://github.com/huggingface/transformers.git
```

3. 開発環境を設定します。たとえば、以下のコマンドを実行してください：

```bash
git clone https://github.com/[your Github handle]/transformers.git
cd transformers
git remote add upstream https://github.com/huggingface/transformers.git
```

依存関係が増えているため、OSに応じて、Transformersのオプションの依存関係の数が増えるかもしれません。その場合は、TensorFlowをインストールしてから次のコマンドを実行してください。

```bash
pip install -e ".[quality]"
```

**注意:** CUDAをインストールする必要はありません。新しいモデルをCPUで動作させることが十分です。

4. メインブランチからわかりやすい名前のブランチを作成してください。

```bash
git checkout -b add_tf_brand_new_bert
```
5. 現在のmainブランチにフェッチしてリベースする

```bash
git fetch upstream
git rebase upstream/main
```

6. `transformers/src/models/brandnewbert/`に`modeling_tf_brandnewbert.py`という名前の空の`.py`ファイルを追加します。これはあなたのTensorFlowモデルファイルです。

7. 以下を使用して変更内容をアカウントにプッシュします：

```bash
git add .
git commit -m "initial commit"
git push -u origin add_tf_brand_new_bert
```

8. GitHub上でフォークしたウェブページに移動し、「プルリクエスト」をクリックします。将来の変更に備えて、Hugging Face チームのメンバーのGitHubハンドルをレビュアーとして追加してください。

9. GitHubのプルリクエストウェブページの右側にある「ドラフトに変換」をクリックして、プルリクエストをドラフトに変更します。

これで、🤗 Transformers内に*BrandNewBert*をTensorFlowに移植するための開発環境が設定されました。

**3. (任意) 理論的な側面と既存の実装を理解する**

*BrandNewBert*の論文が存在する場合、その記述的な作業を読む時間を取るべきです。論文には理解が難しい大きなセクションがあるかもしれません。その場合でも問題ありません - 心配しないでください！目標は論文の理論的な理解を深めることではなく、🤗 Transformersを使用してTensorFlowでモデルを効果的に再実装するために必要な情報を抽出することです。とは言え、理論的な側面にあまり時間をかける必要はありません。代わりに、既存のモデルのドキュメンテーションページ（たとえば、[BERTのモデルドキュメント](model_doc/bert)など）に焦点を当てるべきです。

実装するモデルの基本を把握した後、既存の実装を理解することは重要です。これは、動作する実装がモデルに対する期待と一致することを確認する絶好の機会であり、TensorFlow側での技術的な課題を予測することもできます。

情報の多さに圧倒されていると感じるのは完全に自然です。この段階ではモデルのすべての側面を理解する必要はありません。ただし、[フォーラム](https://discuss.huggingface.co/)で急な質問を解決することを強くお勧めします。


### 4. Model implementation

さあ、いよいよコーディングを始めましょう。お勧めする出発点は、PyTorchファイルそのものです。
`src/transformers/models/brand_new_bert/`内の`modeling_brand_new_bert.py`の内容を
`modeling_tf_brand_new_bert.py`にコピーします。このセクションの目標は、
🤗 Transformersのインポート構造を更新し、`TFBrandNewBert`と
`TFBrandNewBert.from_pretrained(model_repo, from_pt=True)`を正常に読み込む動作するTensorFlow *BrandNewBert*モデルを
インポートできるようにすることです。

残念ながら、PyTorchモデルをTensorFlowに変換する明確な方法はありません。ただし、プロセスをできるだけスムーズにするためのヒントを以下に示します：

- すべてのクラスの名前の前に `TF` を付けます（例： `BrandNewBert` は `TFBrandNewBert` になります）。
- ほとんどのPyTorchの操作には、直接TensorFlowの代替があります。たとえば、`torch.nn.Linear` は `tf.keras.layers.Dense` に対応し、`torch.nn.Dropout` は `tf.keras.layers.Dropout` に対応します。特定の操作について不明確な場合は、[TensorFlowのドキュメント](https://www.tensorflow.org/api_docs/python/tf)または[PyTorchのドキュメント](https://pytorch.org/docs/stable/)を参照できます。
- 🤗 Transformersのコードベースにパターンが見つかります。特定の操作に直接的な代替がない場合、誰かがすでに同じ問題に対処している可能性が高いです。
- デフォルトでは、PyTorchと同じ変数名と構造を維持します。これにより、デバッグや問題の追跡、修正の追加が容易になります。
- 一部のレイヤーには、各フレームワークで異なるデフォルト値があります。注目すべき例は、バッチ正規化レイヤーの epsilon です（PyTorchでは`1e-5`、[TensorFlowでは](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) `1e-3` です）。ドキュメントを再確認してください！
- PyTorchの `nn.Parameter` 変数は通常、TF Layerの `build()` 内で初期化する必要があります。次の例を参照してください：[PyTorch](https://github.com/huggingface/transformers/blob/655f72a6896c0533b1bdee519ed65a059c2425ac/src/transformers/models/vit_mae/modeling_vit_mae.py#L212) / [TensorFlow](https://github.com/huggingface/transformers/blob/655f72a6896c0533b1bdee519ed65a059c2425ac/src/transformers/models/vit_mae/modeling_tf_vit_mae.py#L220)
- PyTorchモデルに関数の上部に `#copied from ...` がある場合、TensorFlowモデルも同じアーキテクチャからその関数を借りることができる可能性が高いです。TensorFlowアーキテクチャがある場合です。
- TensorFlow関数内で `name`属性を正しく設定することは、`from_pt=True`のウェイトのクロスロードロードを行うために重要です。通常、`name`はPyTorchコード内の対応する変数の名前です。`name`が正しく設定されていない場合、モデルウェイトのロード時にエラーメッセージで表示されます。
- ベースモデルクラス `BrandNewBertModel` のロジックは実際には `TFBrandNewBertMainLayer` にあります。これはKerasレイヤーのサブクラスです（[例](https://github.com/huggingface/transformers/blob/4fd32a1f499e45f009c2c0dea4d81c321cba7e02/src/transformers/models/bert/modeling_tf_bert.py#L719)）。`TFBrandNewBertModel` は、単にこのレイヤーのラッパーです。
- モデルを読み込むためには、Kerasモデルをビルドする必要があります。そのため、`TFBrandNewBertPreTrainedModel` はモデルへの入力の例、`dummy_inputs` を持つ必要があります（[例](https://github.com/huggingface/transformers/blob/4fd32a1f499e45f009c2c0dea4d81c321cba7e02/src/transformers/models/bert/modeling_tf_bert.py#L916)）。
- 表示が止まった場合は、助けを求めてください。私たちはあなたのお手伝いにここにいます！ 🤗

モデルファイル自体だけでなく、モデルクラスと関連するドキュメンテーションページへのポインターも追加する必要があります。他のPRのパターンに従ってこの部分を完了できます
（[例](https://github.com/huggingface/transformers/pull/18020/files)）。
以下は手動での変更が必要な一覧です：
- *BrandNewBert*のすべてのパブリッククラスを `src/transformers/__init__.py` に含める
- *BrandNewBert*クラスを `src/transformers/models/auto/modeling_tf_auto.py` の対応するAutoクラスに追加
- ドキュメンテーションテストファイルのリストにモデリングファイルを追加する `utils/documentation_tests.txt`
- `src/transformers/utils/dummy_tf_objects.py` に関連する *BrandNewBert* に関連する遅延ロードクラスを追加
- `src/transformers/models/brand_new_bert/__init__.py` でパブリッククラスのインポート構造を更新
- `docs/source/en/model_doc/brand_new_bert.md` に *BrandNewBert* のパブリックメソッドのドキュメンテーションポインターを追加
- `docs/source/en/model_doc/brand_new_bert.md` の *BrandNewBert* の貢献者リストに自分自身を追加
- 最後に、`docs/source/en/index.md` の *BrandNewBert* のTensorFlow列に緑色のチェックマーク ✅ を追加

モデルアーキテクチャが準備できていることを確認するために、以下のチェックリストを実行してください：
1. 訓練時に異なる動作をするすべてのレイヤー（例：Dropout）は、`training`引数を使用して呼び出され、それが最上位クラスから伝播されます。
2. 可能な限り `#copied from ...` を使用しました
3. `TFBrandNewBertMainLayer` およびそれを使用するすべてのクラスの `call` 関数が `@unpack_inputs` でデコレートされています
4. `TFBrandNewBertMainLayer` は `@keras_serializable` でデコレートされています
5. PyTorchウェイトからTensorFlowウェイトを使用してTensorFlowモデルをロードできます `TFBrandNewBert.from_pretrained(model_repo, from_pt=True)`
6. 予期される入力形式を使用してTensorFlowモデルを呼び出すことができます


### 5. Add model tests

やったね、TensorFlowモデルを実装しました！
今度は、モデルが期待通りに動作することを確認するためのテストを追加する時間です。
前のセクションと同様に、`tests/models/brand_new_bert/`ディレクトリ内の`test_modeling_brand_new_bert.py`ファイルを`test_modeling_tf_brand_new_bert.py`にコピーし、必要なTensorFlowの置換を行うことをお勧めします。
今の段階では、すべての`.from_pretrained()`呼び出しで、既存のPyTorchの重みをロードするために`from_pt=True`フラグを使用する必要があります。

作業が完了したら、テストを実行する準備が整いました！ 😬

```bash
NVIDIA_TF32_OVERRIDE=0 RUN_SLOW=1 RUN_PT_TF_CROSS_TESTS=1 \
py.test -vv tests/models/brand_new_bert/test_modeling_tf_brand_new_bert.py
```

最も可能性の高い結果は、多くのエラーが表示されることです。心配しないでください、これは予想される動作です！
MLモデルのデバッグは非常に難しいとされており、成功の鍵は忍耐力（と`breakpoint()`）です。私たちの経験では、
最も難しい問題はMLフレームワーク間の微妙な不一致から発生し、これについてはこのガイドの最後にいくつかのポインタを示します。
他の場合では、一般的なテストが直接モデルに適用できない場合もあり、その場合はモデルのテストクラスレベルでオーバーライドを提案します。
問題の種類に関係なく、詰まった場合は、ドラフトのプルリクエストで助けを求めることをためらわないでください。

すべてのテストがパスしたら、おめでとうございます。あなたのモデルはほぼ🤗 Transformersライブラリに追加する準備が整いました！🎉

**6. プルリクエストを提出する**

実装とテストが完了したら、プルリクエストを提出する準備が整いました。コードをプッシュする前に、
コードフォーマットユーティリティである `make fixup` 🪄 を実行してください。
これにより、自動的なチェックに失敗する可能性のあるフォーマットの問題が自動的に修正されます。

これで、ドラフトプルリクエストを実際のプルリクエストに変換する準備が整いました。
これを行うには、「レビュー待ち」ボタンをクリックし、Joao（`@gante`）とMatt（`@Rocketknight1`）をレビュワーとして追加します。
モデルプルリクエストには少なくとも3人のレビュワーが必要ですが、モデルに適切な追加のレビュワーを見つけるのは彼らの責任です。

すべてのレビュワーがプルリクエストの状態に満足したら、最後のアクションポイントは、`.from_pretrained()` 呼び出しで `from_pt=True` フラグを削除することです。
TensorFlowのウェイトが存在しないため、それらを追加する必要があります！これを行う方法については、以下のセクションを確認してください。

最後に、TensorFlowのウェイトがマージされ、少なくとも3人のレビューアが承認し、すべてのCIチェックが
成功した場合、テストをローカルで最後にもう一度確認してください。

```bash
NVIDIA_TF32_OVERRIDE=0 RUN_SLOW=1 RUN_PT_TF_CROSS_TESTS=1 \
py.test -vv tests/models/brand_new_bert/test_modeling_tf_brand_new_bert.py
```

そして、あなたのPRをマージします！マイルストーン達成おめでとうございます 🎉

**7. (Optional) デモを作成して世界と共有**

オープンソースの最も難しい部分の1つは、発見です。あなたの素晴らしいTensorFlowの貢献が存在することを他のユーザーがどのように知ることができるでしょうか？適切なコミュニケーションです！ 📣

コミュニティとモデルを共有する主要な方法は2つあります。
- デモを作成します。これにはGradioデモ、ノートブック、およびモデルを紹介するための他の楽しい方法が含まれます。[コミュニティ駆動のデモ](https://huggingface.co/docs/transformers/community)にノートブックを追加することを強くお勧めします。
- TwitterやLinkedInなどのソーシャルメディアでストーリーを共有します。あなたの仕事に誇りを持ち、コミュニティとあなたの成果を共有するべきです - あなたのモデルは今や世界中の何千人ものエンジニアや研究者によって使用される可能性があります 🌍！私たちはあなたの投稿をリツイートして共同体と共有するお手伝いを喜んでします。

## Adding TensorFlow weights to 🤗 Hub

TensorFlowモデルのアーキテクチャが🤗 Transformersで利用可能な場合、PyTorchの重みをTensorFlowの重みに変換することは簡単です！

以下がその方法です：
1. ターミナルでHugging Faceアカウントにログインしていることを確認してください。コマンド`huggingface-cli login`を使用してログインできます（アクセストークンは[こちら](https://huggingface.co/settings/tokens)で見つけることができます）。
2. `transformers-cli pt-to-tf --model-name foo/bar`というコマンドを実行します。ここで、`foo/bar`は変換したいPyTorchの重みを含むモデルリポジトリの名前です。
3. 上記のコマンドで作成された🤗 Hub PRに`@joaogante`と`@Rocketknight1`をタグ付けします。

それだけです！ 🎉

## Debugging mismatches across ML frameworks 🐛

新しいアーキテクチャを追加したり、既存のアーキテクチャのTensorFlowの重みを作成したりする際、PyTorchとTensorFlow間の不一致についてのエラーに遭遇することがあります。
場合によっては、PyTorchとTensorFlowのモデルアーキテクチャがほぼ同一であるにもかかわらず、不一致を指摘するエラーが表示されることがあります。
どうしてでしょうか？ 🤔

まず最初に、なぜこれらの不一致を理解することが重要かについて話しましょう。多くのコミュニティメンバーは🤗 Transformersモデルをそのまま使用し、モデルが期待どおりに動作すると信頼しています。
2つのフレームワーク間で大きな不一致があると、少なくとも1つのフレームワークのリファレンス実装に従ってモデルが動作しないことを意味します。
これにより、モデルは実行されますが性能が低下する可能性があり、静かな失敗が発生する可能性があります。これは、全く実行されないモデルよりも悪いと言えるかもしれません！そのため、モデルのすべての段階でのフレームワークの不一致が`1e-5`未満であることを目指しています。

数値計算の問題と同様に、詳細については細かいところにあります。そして、詳細指向の技術である以上、秘密の要素は忍耐です。
この種の問題に遭遇した場合のお勧めのワークフローは次のとおりです：
1. 不一致の原因を特定します。変換中のモデルにはおそらく特定の点までほぼ同一の内部変数があります。
   両方のフレームワークのアーキテクチャに`breakpoint()`ステートメントを配置し、トップダウンの方法で数値変数の値を比較し、問題の原因を見つけます。
2. 問題の原因を特定したら、🤗 Transformersチームと連絡を取りましょう。同様の問題に遭遇したことがあるかもしれず、迅速に解決策を提供できるかもしれません。最終手段として、StackOverflowやGitHubの問題など、人気のあるページをスキャンします。
3. 解決策が見当たらない場合、問題を掘り下げる必要があることを意味します。良いニュースは、問題の原因を特定したことです。したがって、問題のある命令に焦点を当て、モデルの残りを抽象化できます！悪いニュースは、その命令のソース実装に進む必要があることです。一部の場合では、リファレンス実装に問題があるかもしれません - 上流リポジトリで問題を開くのを控えないでください。

🤗 Transformersチームとの話し合いで、不一致を修正することが困難であることが判明することがあります。
出力レイヤーのモデルで不一致が非常に小さい場合（ただし、隠れた状態では大きい可能性がある）、モデルを配布するためにそれを無視することにするかもしれません。
上記で言及した`pt-to-tf` CLIには、重み変換時にエラーメッセージを無視するための`--max-error`フラグがあります。






