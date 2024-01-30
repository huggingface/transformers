<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Training on TPU with TensorFlow

<Tip>

詳細な説明が不要で、単にTPUのコードサンプルを入手してトレーニングを開始したい場合は、[私たちのTPUの例のノートブックをチェックしてください！](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)

</Tip>

### What is a TPU?

TPUは**Tensor Processing Unit（テンソル処理ユニット）**の略です。これらはGoogleが設計したハードウェアで、ニューラルネットワーク内のテンソル計算を大幅に高速化するために使用されます。これはGPUのようなものです。ネットワークのトレーニングと推論の両方に使用できます。一般的にはGoogleのクラウドサービスを介してアクセスされますが、Google ColabとKaggle Kernelsを通じても無料で小規模のTPUに直接アクセスできます。

[🤗 TransformersのすべてのTensorFlowモデルはKerasモデルです](https://huggingface.co/blog/tensorflow-philosophy)ので、この文書のほとんどの方法は一般的にKerasモデル用のTPUトレーニングに適用できます！ただし、TransformersとDatasetsのHuggingFaceエコシステム（hug-o-system？）に固有のポイントもいくつかあり、それについては適用するときにそれを示します。

### What kinds of TPU are available?

新しいユーザーは、さまざまなTPUとそのアクセス方法に関する幅広い情報によく混乱します。理解するための最初の重要な違いは、**TPUノード**と**TPU VM**の違いです。

**TPUノード**を使用すると、事実上リモートのTPUに間接的にアクセスします。別個のVMが必要で、ネットワークとデータパイプラインを初期化し、それらをリモートノードに転送します。Google ColabでTPUを使用すると、**TPUノード**スタイルでアクセスしています。

TPUノードを使用すると、それに慣れていない人々にはかなり予期しない動作が発生することがあります！特に、TPUはPythonコードを実行しているマシンと物理的に異なるシステムに配置されているため、データはローカルマシンにローカルで格納されているデータパイプラインが完全に失敗します。代わりに、データはGoogle Cloud Storageに格納する必要があります。ここでデータパイプラインはリモートのTPUノードで実行されている場合でも、データにアクセスできます。

<Tip>

すべてのデータを`np.ndarray`または`tf.Tensor`としてメモリに収めることができる場合、ColabまたはTPUノードを使用している場合でも、データをGoogle Cloud Storageにアップロードせずに`fit()`でトレーニングできます。

</Tip>

<Tip>

**🤗 Hugging Face固有のヒント🤗:** TFコードの例でよく見るであろう`Dataset.to_tf_dataset()`とその高レベルのラッパーである`model.prepare_tf_dataset()`は、TPUノードで失敗します。これは、`tf.data.Dataset`を作成しているにもかかわらず、それが「純粋な」`tf.data`パイプラインではなく、`tf.numpy_function`または`Dataset.from_generator()`を使用して基盤となるHuggingFace `Dataset`からデータをストリームで読み込むことからです。このHuggingFace `Dataset`はローカルディスク上のデータをバックアップしており、リモートTPUノードが読み取ることができないためです。

</Tip>

TPUにアクセスする第二の方法は、**TPU VM**を介してです。TPU VMを使用する場合、TPUが接続されているマシンに直接接続します。これはGPU VMでトレーニングを行うのと同様です。TPU VMは一般的にデータパイプラインに関しては特に作業がしやすく、上記のすべての警告はTPU VMには適用されません！

これは主観的な文書ですので、こちらの意見です：**可能な限りTPUノードの使用を避けてください。** TPU VMよりも混乱しやすく、デバッグが難しいです。将来的にはサポートされなくなる可能性もあります - Googleの最新のTPUであるTPUv4は、TPU VMとしてのみアクセスできるため、TPUノードは将来的には「レガシー」のアクセス方法になる可能性が高いです。ただし、無料でTPUにアクセスできるのはColabとKaggle Kernelsの場合があります。その場合、どうしても使用しなければならない場合の取り扱い方法を説明しようとします！詳細は[TPUの例のノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)で詳細な説明を確認してください。

### What sizes of TPU are available?

単一のTPU（v2-8/v3-8/v4-8）は8つのレプリカを実行します。TPUは数百から数千のレプリカを同時に実行できる**ポッド**に存在します。単一のTPUよりも多くのTPUを使用するが、ポッド全体ではない場合（たとえばv3-32）、TPUフリートは**ポッドスライス**として参照されます。

Colabを介して無料のTPUにアクセスする場合、通常は単一のv2-8 TPUが提供されます。


### I keep hearing about this XLA thing. What’s XLA, and how does it relate to TPUs?

XLAは、TensorFlowとJAXの両方で使用される最適化コンパイラです。JAXでは唯一のコンパイラであり、TensorFlowではオプションですが（しかしTPUでは必須です！）、Kerasモデルをトレーニングする際に`model.compile()`に引数`jit_compile=True`を渡すことで最も簡単に有効にできます。エラーが発生せず、パフォーマンスが良好であれば、それはTPUに移行する準備が整った良い兆候です！

TPU上でのデバッグは一般的にCPU/GPUよりも少し難しいため、TPUで試す前にまずCPU/GPUでXLAを使用してコードを実行することをお勧めします。もちろん、長時間トレーニングする必要はありません。モデルとデータパイプラインが期待通りに動作するかを確認するための数ステップだけです。

<Tip>

XLAコンパイルされたコードは通常高速です。したがって、TPUで実行する予定がない場合でも、`jit_compile=True`を追加することでパフォーマンスを向上させることができます。ただし、以下のXLA互換性に関する注意事項に注意してください！

</Tip>

<Tip warning={true}>

**苦い経験から生まれたヒント:** `jit_compile=True`を使用することは、CPU/GPUコードがXLA互換であることを確認し、速度を向上させる良い方法ですが、実際にTPUでコードを実行する際には多くの問題を引き起こす可能性があります。 XLAコンパイルはTPU上で暗黙的に行われるため、実際にコードをTPUで実行する前にその行を削除することを忘れないでください！

</Tip>

### How do I make my model XLA compatible?

多くの場合、コードはすでにXLA互換かもしれません！ただし、XLAでは動作する通常のTensorFlowでも動作しないいくつかの要素があります。以下に、3つの主要なルールにまとめています：

<Tip>

**🤗 HuggingFace固有のヒント🤗:** TensorFlowモデルと損失関数をXLA互換に書き直すために多くの努力を払っています。通常、モデルと損失関数はデフォルトでルール＃1と＃2に従っているため、`transformers`モデルを使用している場合はこれらをスキップできます。ただし、独自のモデルと損失関数を記述する場合は、これらのルールを忘れないでください！

</Tip>

#### XLA Rule #1: Your code cannot have “data-dependent conditionals”

これは、任意の`if`ステートメントが`tf.Tensor`内の値に依存していない必要があることを意味します。例えば、次のコードブロックはXLAでコンパイルできません！

```python
if tf.reduce_sum(tensor) > 10:
    tensor = tensor / 2.0
```

これは最初は非常に制限的に思えるかもしれませんが、ほとんどのニューラルネットコードはこれを行う必要はありません。通常、この制約を回避するために`tf.cond`を使用するか（ドキュメントはこちらを参照）、条件を削除して代わりに指示変数を使用したりすることができます。次のように：

```python
sum_over_10 = tf.cast(tf.reduce_sum(tensor) > 10, tf.float32)
tensor = tensor / (1.0 + sum_over_10)
```

このコードは、上記のコードとまったく同じ効果を持っていますが、条件を回避することで、XLAで問題なくコンパイルできることを確認します！

#### XLA Rule #2: Your code cannot have “data-dependent shapes”

これは、コード内のすべての `tf.Tensor` オブジェクトの形状が、その値に依存しないことを意味します。たとえば、`tf.unique` 関数はXLAでコンパイルできないので、このルールに違反します。なぜなら、これは入力 `Tensor` の一意の値の各インスタンスを含む `tensor` を返すためです。この出力の形状は、入力 `Tensor` の重複具合によって異なるため、XLAはそれを処理しないことになります！

一般的に、ほとんどのニューラルネットワークコードはデフォルトでルール＃2に従います。ただし、いくつかの一般的なケースでは問題が発生することがあります。非常に一般的なケースの1つは、**ラベルマスキング**を使用する場合です。ラベルを無視して損失を計算する場所を示すために、ラベルを負の値に設定する方法です。NumPyまたはPyTorchのラベルマスキングをサポートする損失関数を見ると、次のような[ブールインデックス](https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing)を使用したコードがよく見られます：


```python
label_mask = labels >= 0
masked_outputs = outputs[label_mask]
masked_labels = labels[label_mask]
loss = compute_loss(masked_outputs, masked_labels)
mean_loss = torch.mean(loss)
```

このコードはNumPyやPyTorchでは完全に機能しますが、XLAでは動作しません！なぜなら、`masked_outputs`と`masked_labels`の形状はマスクされた位置の数に依存するため、これは**データ依存の形状**になります。ただし、ルール＃1と同様に、このコードを書き直して、データ依存の形状なしでまったく同じ出力を生成できることがあります。


```python
label_mask = tf.cast(labels >= 0, tf.float32)
loss = compute_loss(outputs, labels)
loss = loss * label_mask  # Set negative label positions to 0
mean_loss = tf.reduce_sum(loss) / tf.reduce_sum(label_mask)
```


ここでは、データ依存の形状を避けるために、各位置で損失を計算してから、平均を計算する際に分子と分母の両方でマスクされた位置をゼロ化する方法を紹介します。これにより、最初のアプローチとまったく同じ結果が得られますが、XLA互換性を維持します。注意点として、ルール＃1と同じトリックを使用します - `tf.bool`を`tf.float32`に変換して指標変数として使用します。これは非常に便利なトリックですので、自分のコードをXLAに変換する必要がある場合には覚えておいてください！

#### XLA Rule #3: XLA will need to recompile your model for every different input shape it sees

これは重要なルールです。これはつまり、入力形状が非常に変動的な場合、XLA はモデルを何度も再コンパイルする必要があるため、大きなパフォーマンスの問題が発生する可能性があるということです。これは NLP モデルで一般的に発生し、トークナイズ後の入力テキストの長さが異なる場合があります。他のモダリティでは、静的な形状が一般的であり、このルールはほとんど問題になりません。

ルール＃3を回避する方法は何でしょうか？鍵は「パディング」です - すべての入力を同じ長さにパディングし、次に「attention_mask」を使用することで、可変形状と同じ結果を得ることができますが、XLA の問題は発生しません。ただし、過度のパディングも深刻な遅延を引き起こす可能性があります - データセット全体で最大の長さにすべてのサンプルをパディングすると、多くの計算とメモリを無駄にする可能性があります！

この問題には完璧な解決策はありませんが、いくつかのトリックを試すことができます。非常に便利なトリックの1つは、**バッチのサンプルを32または64トークンの倍数までパディングする**ことです。これにより、トークン数がわずかに増加するだけで、すべての入力形状が32または64の倍数である必要があるため、一意の入力形状の数が大幅に減少します。一意の入力形状が少ないと、XLA の再コンパイルが少なくなります！

<Tip>

**🤗 HuggingFace に関する具体的なヒント🤗:** 弊社のトークナイザーとデータコレクターには、ここで役立つメソッドがあります。トークナイザーを呼び出す際に `padding="max_length"` または `padding="longest"` を使用して、パディングされたデータを出力するように設定できます。トークナイザーとデータコレクターには、一意の入力形状の数を減らすのに役立つ `pad_to_multiple_of` 引数もあります！

</Tip>

### How do I actually train my model on TPU?

一度トレーニングが XLA 互換性があることを確認し、（TPU Node/Colab を使用する場合は）データセットが適切に準備されている場合、TPU 上で実行することは驚くほど簡単です！コードを変更する必要があるのは、いくつかの行を追加して TPU を初期化し、モデルとデータセットが `TPUStrategy` スコープ内で作成されるようにすることだけです。これを実際に見るには、[TPU のサンプルノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)をご覧ください！

### Summary

ここでは多くの情報が提供されましたので、TPU でモデルをトレーニングする際に以下のチェックリストを使用できます：

- コードが XLA の三つのルールに従っていることを確認します。
- CPU/GPU で `jit_compile=True` を使用してモデルをコンパイルし、XLA でトレーニングできることを確認します。
- データセットをメモリに読み込むか、TPU 互換のデータセット読み込みアプローチを使用します（[ノートブックを参照](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)）。
- コードを Colab（アクセラレータを「TPU」に設定）または Google Cloud の TPU VM に移行します。
- TPU 初期化コードを追加します（[ノートブックを参照](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)）。
- `TPUStrategy` を作成し、データセットの読み込みとモデルの作成が `strategy.scope()` 内で行われることを確認します（[ノートブックを参照](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)）。
- TPU に移行する際に `jit_compile=True` を外すのを忘れないでください！
- 🙏🙏🙏🥺🥺🥺
- `model.fit()` を呼び出します。
- おめでとうございます！


