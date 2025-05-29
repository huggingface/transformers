<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Methods and tools for efficient training on a single GPU

このガイドでは、メモリの利用効率を最適化し、トレーニングを高速化することで、モデルのトレーニング効率を向上させるために使用できる実用的なテクニックを紹介します。トレーニング中にGPUがどのように利用されるかを理解したい場合は、最初に「[モデルトレーニングの解剖学](model_memory_anatomy)」のコンセプトガイドを参照してください。このガイドは実用的なテクニックに焦点を当てています。

<Tip>

複数のGPUを搭載したマシンにアクセスできる場合、これらのアプローチは依然として有効です。さらに、[マルチGPUセクション](perf_train_gpu_many)で説明されている追加の方法を活用できます。

</Tip>

大規模なモデルをトレーニングする際、同時に考慮すべき2つの側面があります：

* データのスループット/トレーニング時間
* モデルのパフォーマンス

スループット（サンプル/秒）を最大化することは、トレーニングコストを低減させます。これは一般的に、GPUをできるだけ効果的に活用し、GPUメモリを限界まで埋めることによって達成されます。希望するバッチサイズがGPUメモリの制限を超える場合、勾配蓄積などのメモリ最適化テクニックが役立ちます。

しかし、好みのバッチサイズがメモリに収まる場合、メモリを最適化するテクニックを適用する理由はありません。大きなバッチサイズを使用できるからといって、それを必ずしも使用すべきではありません。ハイパーパラメータの調整の一環として、どのバッチサイズが最良の結果を生み出すかを決定し、リソースを適切に最適化する必要があります。

このガイドでカバーされている方法とツールは、トレーニングプロセスに与える影響に基づいて分類できます：


| Method/tool                                                | Improves training speed | Optimizes memory utilization |
|:-----------------------------------------------------------|:------------------------|:-----------------------------|
| [Batch size choice](#batch-size-choice)                    | Yes                     | Yes                          |
| [Gradient accumulation](#gradient-accumulation)            | No                      | Yes                          |
| [Gradient checkpointing](#gradient-checkpointing)          | No                      | Yes                          |
| [Mixed precision training](#mixed-precision-training)      | Yes                     | (No)                         |
| [Optimizer choice](#optimizer-choice)                      | Yes                     | Yes                          |
| [Data preloading](#data-preloading)                        | Yes                     | No                           |
| [DeepSpeed Zero](#deepspeed-zero)                          | No                      | Yes                          |
| [torch.compile](#using-torchcompile)                       | Yes                     | No                           |

<Tip>

**注意**: 小さなモデルと大きなバッチサイズを使用する場合、メモリの節約が行われますが、大きなモデルと小さなバッチサイズを使用する場合、メモリの使用量が増加します。

</Tip>

これらのテクニックは、[`Trainer`]でモデルをトレーニングしている場合や、純粋なPyTorchループを記述している場合の両方で利用できます。詳細な最適化の設定については、🤗 Accelerateを使用して[これらの最適化を設定できます](#using--accelerate)。

これらの方法が十分な利益をもたらさない場合、以下のオプションを検討できます：
* [効率的なソフトウェアプリビルドを備えたカスタムDockerコンテナの作成](#efficient-software-prebuilds)
* [Mixture of Experts（MoE）を使用するモデルを検討](#mixture-of-experts)
* [モデルをBetterTransformerに変換して、PyTorchネイティブのアテンションを活用](#using-pytorch-native-attention)

最後に、これらの方法がまだ十分でない場合、A100などのサーバーグレードGPUに切り替えても、さらなる改善が必要かもしれません。これらのアプローチは、マルチGPUセットアップでも有効であり、[マルチGPUセクション](perf_train_gpu_many)で説明されている追加の並列化技術を活用できます。

## Batch size choice

最適なパフォーマンスを実現するために、適切なバッチサイズを特定することから始めましょう。2^Nのサイズのバッチサイズと入力/出力ニューロン数を使用することが推奨されています。通常、これは8の倍数ですが、使用するハードウェアとモデルのデータ型に依存することがあります。

参考までに、NVIDIAの[入力/出力ニューロン数の推奨事項](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#input-features)と[バッチサイズ](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#batch-size)を確認してください（これらはGEMM（一般的な行列乗算）に関与します）。

[Tensor Core要件](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc)では、データ型とハードウェアに基づいて乗数が定義されています。たとえば、fp16データ型の場合、64の倍数を使用することが推奨されます（A100 GPUの場合を除く）。

小さなパラメータの場合、[次元量子化効果](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#dim-quantization)も考慮してください。これはタイリングが行われ、適切な乗数が大幅な高速化をもたらす場合があります。

## Gradient Accumulation

**勾配蓄積**メソッドは、GPUのメモリ容量の制約によって課せられる制限を超えた効果的なバッチサイズを実現するために、勾配を小さな増分で計算することを目的としています。このアプローチでは、モデルを順方向および逆方向に小さなバッチで反復的に計算し、その過程で勾配を蓄積します。十分な数の勾配が蓄積されたら、モデルの最適化ステップを実行します。勾配蓄積を使用することで、GPUのメモリ容量による制約を超えて**効果的なバッチサイズ**を増やすことができますが、勾配蓄積によって導入される追加の順方向および逆方向の計算はトレーニングプロセスを遅くする可能性があることに注意が必要です。

`TrainingArguments`に`gradient_accumulation_steps`引数を追加することで、勾配蓄積を有効にすることができます：

```py
training_args = TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=4, **default_args)
```

上記の例では、効果的なバッチサイズは4になります。

また、トレーニングループを完全に制御するために🤗 Accelerateを使用することもできます。🤗 Accelerateの例は、[このガイドの後半にある](#using--accelerate)で見つけることができます。

できるだけGPUの使用率を最大限にすることが推奨されていますが、高い勾配蓄積ステップ数はトレーニングの遅延をより顕著にすることがあります。以下の例を考えてみましょう。`per_device_train_batch_size=4`の場合、勾配蓄積を使用しないとGPUの制限に達します。バッチサイズ64でトレーニングしたい場合、`per_device_train_batch_size`を1に設定し、`gradient_accumulation_steps`を64に設定しないでください。代わりに、`per_device_train_batch_size=4`を保持し、`gradient_accumulation_steps=16`を設定します。これにより、同じ効果的なバッチサイズが得られ、利用可能なGPUリソースが効果的に活用されます。

詳細な情報については、[RTX-3090用のバッチサイズと勾配蓄積のベンチマーク](https://github.com/huggingface/transformers/issues/14608#issuecomment-1004392537)および[A100用のバッチサイズと勾配蓄積のベンチマーク](https://github.com/huggingface/transformers/issues/15026#issuecomment-1005033957)を参照してください。

## Gradient Checkpointing

一部の大きなモデルは、バッチサイズを1に設定し、勾配蓄積を使用している場合でもメモリの問題に直面することがあります。これは、メモリストレージが必要な他のコンポーネントも存在するためです。

前向きパスからのすべてのアクティベーションを保存して、逆向きパスで勾配を計算すると、かなりのメモリオーバーヘッドが発生します。逆向きパスで必要なときにアクティベーションを破棄して再計算する代替アプローチは、計算オーバーヘッドが大幅に増加し、トレーニングプロセスが遅くなります。

**勾配チェックポイント**は、これらの2つのアプローチの折衷案を提供し、計算グラフ全体で戦略的に選択されたアクティベーションのみを保存するため、勾配を再計算する必要があるアクティベーションの一部だけを節約します。勾配チェックポイントの詳細については、[この素晴らしい記事](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9)を参照してください。

[`Trainer`]で勾配チェックポイントを有効にするには、[`TrainingArguments`]に対応するフラグを渡します：


```py
training_args = TrainingArguments(
    per_device_train_batch_size=1, gradient_accumulation_steps=4, gradient_checkpointing=True, **default_args
)
```

代替手段として、🤗 Accelerateを使用することもできます - 🤗 Accelerateの例は[このガイドのさらに後ろにあります](#using--accelerate)。

<Tip>

勾配チェックポイントを使用することでメモリ効率が向上する場合がありますが、トレーニング速度は約20%遅くなることに注意してください。

</Tip>

## Mixed precision training

**混合精度トレーニング**は、モデルのトレーニングの計算効率を最適化する技術で、特定の変数に対して低精度の数値フォーマットを利用します。従来、ほとんどのモデルは変数を表現し処理するために32ビット浮動小数点精度（fp32またはfloat32）を使用しています。しかし、すべての変数が正確な結果を得るためにこの高精度のレベルを必要としない場合があります。一部の変数の精度を16ビット浮動小数点（fp16またはfloat16）などのより低い数値フォーマットに変更することで、計算を高速化できます。このアプローチでは、一部の計算は半精度で行われ、一部はまだ完全な精度で行われるため、このアプローチは混合精度トレーニングと呼ばれています。

最も一般的に混合精度トレーニングは、fp16（float16）データ型を使用して実現されますが、一部のGPUアーキテクチャ（アンペアアーキテクチャなど）ではbf16およびtf32（CUDA内部データ型）データ型も提供されています。これらのデータ型の違いについて詳しく知りたい場合は、[NVIDIAのブログ](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)を確認してください。

### fp16

混合精度トレーニングの主な利点は、半精度（fp16）でアクティベーションを保存することから得られます。
勾配も半精度で計算されますが、最適化ステップでは再び完全精度に変換されるため、ここではメモリは保存されません。
混合精度トレーニングは計算速度を向上させる一方、特に小さなバッチサイズの場合、より多くのGPUメモリを使用することがあります。
これは、モデルがGPU上に16ビットおよび32ビット精度の両方で存在するためです（GPU上の元のモデルの1.5倍）。

混合精度トレーニングを有効にするには、`fp16`フラグを`True`に設定します：

```py
training_args = TrainingArguments(per_device_train_batch_size=4, fp16=True, **default_args)
```

🤗 Accelerateを使用する場合、🤗 Accelerateの例は[このガイドのさらに後ろにあります](#using--accelerate)。

### BF16

Ampereまたはそれ以降のハードウェアにアクセスできる場合、混合精度トレーニングと評価にbf16を使用できます。bf16はfp16よりも精度が劣りますが、はるかに大きな動的範囲を持っています。fp16では、持つことができる最大の数は `65535` であり、それを超える数値はオーバーフローを引き起こします。一方、bf16の数値は `3.39e+38` のように大きく、これはfp32とほぼ同じです - どちらも数値範囲に8ビットを使用しているためです。

BF16を有効にするには、🤗 Trainerで以下のように設定します：


```python
training_args = TrainingArguments(bf16=True, **default_args)
```


### TF32

アンペアハードウェアは、tf32という特別なデータ型を使用します。これは、fp32と同じ数値範囲（8ビット）を持っていますが、23ビットの精度ではなく、10ビットの精度（fp16と同じ）を持ち、合計で19ビットしか使用しません。これは通常のfp32トレーニングおよび推論コードを使用し、tf32サポートを有効にすることで、最大3倍のスループットの向上が得られる点で「魔法のよう」です。行う必要があるのは、次のコードを追加するだけです：

```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```


使用されているGPUがアンペアシリーズであると仮定し、CUDAは可能な限りtf32を使用するように自動的に切り替えます。

[NVIDIAの研究によれば](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)、ほとんどの機械学習トレーニングワークロードはtf32トレーニングとfp32トレーニングで同じ難解度と収束を示します。すでにfp16またはbf16混合精度を使用している場合、スループットの向上に役立つこともあります。

🤗 Trainerでこのモードを有効にすることができます：


```python
TrainingArguments(tf32=True, **default_args)
```

<Tip>

tf32は`tensor.to(dtype=torch.tf32)`を介して直接アクセスできません。これは内部のCUDAデータ型です。tf32データ型を使用するには、`torch>=1.7`が必要です。

</Tip>

tf32と他の精度に関する詳細な情報については、以下のベンチマークを参照してください：
[RTX-3090](https://github.com/huggingface/transformers/issues/14608#issuecomment-1004390803)および
[A100](https://github.com/huggingface/transformers/issues/15026#issuecomment-1004543189)。

## Flash Attention 2

transformersでFlash Attention 2統合を使用することで、トレーニングのスループットを向上させることができます。Flash Attention 2モジュールを含むモデルの読み込み方法については、[single GPU section](./perf_infer_gpu_one#Flash-Attention-2)の適切なセクションを確認して詳細を学びましょう。

## オプティマイザの選択

Transformerモデルをトレーニングするために最も一般的に使用されるオプティマイザはAdamまたはAdamW（重み減衰を伴うAdam）です。Adamは前回の勾配の移動平均を保存することで収束を達成しますが、モデルパラメータの数のオーダーの追加メモリフットプリントを追加します。これを解消するために、代替オプティマイザを使用できます。たとえば、[NVIDIA/apex](https://github.com/NVIDIA/apex)がインストールされている場合、`adamw_apex_fused`はすべてのサポートされているAdamWオプティマイザの中で最も高速なトレーニング体験を提供します。

[`Trainer`]は、直接使用できるさまざまなオプティマイザを統合しており、`adamw_hf`、`adamw_torch`、`adamw_torch_fused`、`adamw_apex_fused`、`adamw_anyprecision`、`adafactor`、または`adamw_bnb_8bit`が含まれています。サードパーティの実装を介してさらに多くのオプティマイザを追加できます。

AdamWオプティマイザの代替手段について詳しく見てみましょう：
1. [`Trainer`]で使用可能な`adafactor`
2. Trainerで使用可能な`adamw_bnb_8bit`は、デモンストレーション用に以下でサードパーティの統合が提供されています。

比較のため、3Bパラメータモデル（例：「google-t5/t5-3b」）の場合：
* 標準のAdamWオプティマイザは、各パラメータに8バイトを使用するため、24GBのGPUメモリが必要です（8 * 3 => 24GB）。
* Adafactorオプティマイザは12GB以上必要です。各パラメータにわずか4バイト以上を使用するため、4 * 3と少し余分になります。
* 8ビットのBNB量子化オプティマイザは、すべてのオプティマイザの状態が量子化されている場合、わずか6GBしか使用しません。

### Adafactor

Adafactorは、重み行列の各要素のために前回の平均を保存しません。代わりに、（行ごとと列ごとの平均の合計など）集


```py
training_args = TrainingArguments(per_device_train_batch_size=4, optim="adafactor", **default_args)
```


他のアプローチ（勾配蓄積、勾配チェックポイント、混合精度トレーニング）と組み合わせることで、スループットを維持しながら最大3倍の向上が見られることがあります！ただし、前述のように、Adafactorの収束性はAdamよりも悪いことがあります。

### 8ビット Adam

Adafactorのようにオプティマイザの状態を集約する代わりに、8ビットのAdamは完全な状態を保持し、それを量子化します。量子化とは、状態を低い精度で保存し、最適化のためだけに非量子化することを意味します。これは混合精度トレーニングの背後にあるアイデアと似ています。

`adamw_bnb_8bit`を使用するには、単に[`TrainingArguments`]で`optim="adamw_bnb_8bit"`を設定するだけです：


```py
training_args = TrainingArguments(per_device_train_batch_size=4, optim="adamw_bnb_8bit", **default_args)
```

ただし、デモンストレーション目的で8ビットオプティマイザをサードパーティの実装を使用することもできます。これを統合する方法を確認するためです。

まず、8ビットAdamオプティマイザを実装した`bitsandbytes`ライブラリをインストールするために、GitHub [リポジトリ](https://github.com/TimDettmers/bitsandbytes)内のインストールガイドに従ってください。

次に、オプティマイザを初期化する必要があります。これには2つのステップが含まれます：
* まず、モデルのパラメータを2つのグループに分けます - 重み減衰を適用するべきグループと、適用すべきでないグループです。通常、バイアスとレイヤー正規化パラメータは重み減衰されません。
* 次に、以前に使用したAdamWオプティマイザと同じパラメータを使用するために、いくつかの引数の調整を行います。


```py
import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names

training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)

decay_parameters = get_parameter_names(model, [nn.LayerNorm], ["bias", "layernorm", "rmsnorm"])
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if n in decay_parameters],
        "weight_decay": training_args.weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
        "weight_decay": 0.0,
    },
]

optimizer_kwargs = {
    "betas": (training_args.adam_beta1, training_args.adam_beta2),
    "eps": training_args.adam_epsilon,
}
optimizer_kwargs["lr"] = training_args.learning_rate
adam_bnb_optim = bnb.optim.Adam8bit(
    optimizer_grouped_parameters,
    betas=(training_args.adam_beta1, training_args.adam_beta2),
    eps=training_args.adam_epsilon,
    lr=training_args.learning_rate,
)
```

最後に、カスタムオプティマイザを`Trainer`に引数として渡します：


```py
trainer = Trainer(model=model, args=training_args, train_dataset=ds, optimizers=(adam_bnb_optim, None))
```

他のアプローチ（勾配蓄積、勾配チェックポイント、混合精度トレーニング）と組み合わせることで、Adafactorの使用と同等以上の3倍のメモリ改善およびわずかに高いスループットを期待できます。

### multi_tensor

pytorch-nightlyは、多くの小さな特徴テンソルがある状況のオプティマイザを大幅に高速化するはずの`torch.optim._multi_tensor`を導入しました。これは最終的にはデフォルトになるはずですが、それを早く試してみたい場合は、このGitHub [issue](https://github.com/huggingface/transformers/issues/9965)をご覧ください。

## データの事前読み込み

優れたトレーニング速度に到達するための重要な要件の1つは、GPUが処理できる最大速度でデータを供給できる能力です。デフォルトではすべてがメインプロセスで行われ、データをディスクから十分速く読み取ることができない場合、GPUのアンダーユーティリゼーションを引き起こすボトルネックが発生する可能性があります。ボトルネックを減らすために、以下の引数を設定します：

- `DataLoader(pin_memory=True, ...)` - データをCPUのピンメモリに事前読み込みし、通常、CPUからGPUメモリへの転送がはるかに高速化されます。
- `DataLoader(num_workers=4, ...)` - データをより速く事前読み込みするために複数のワーカーを生成します。トレーニング中にGPUの利用状況の統計情報を確認し、100％から遠い場合、ワーカーの数を増やす実験を行ってください。もちろん、問題は他の場所にあるかもしれませんので、多くのワーカーが必ずしも性能向上につながるわけではありません。

[`Trainer`]を使用する場合、対応する[`TrainingArguments`]は`dataloader_pin_memory`（デフォルトでは`True`）および`dataloader_num_workers`（デフォルトは`0`）です。

## DeepSpeed ZeRO

DeepSpeedは、🤗 Transformersと🤗 Accelerateと統合されたオープンソースのディープラーニング最適化ライブラリです。
大規模なディープラーニングトレーニングの効率とスケーラビリティを向上させるために設計されたさまざまな機能と最適化を提供します。

モデルが単一のGPUに収まり、小さなバッチサイズを収めるスペースがある場合、DeepSpeedを使用する必要はありません。それはむしろ遅くなります。ただし、モデルが単一のGPUに収まらない場合、または小さなバッチを収めることができない場合、DeepSpeed ZeRO + CPU OffloadまたはNVMe Offloadを利用できます。この場合、[ライブラリを別途インストール](main_classes/deepspeed#installation)し、設定ファイルを作成し、DeepSpeedを起動するためのガイドをフォローする必要があります：

* [`Trainer`]とのDeepSpeed統合の詳細ガイドについては、[該当するドキュメンテーション](main_classes/deepspeed)を確認してください。特に、[単一GPU用のデプロイメント](main_classes/deepspeed#deployment-with-one-gpu)に関するセクションです。DeepSpeedをノートブックで使用するにはいくつかの調整が必要ですので、[該当するガイド](main_classes/deepspeed#deployment-in-notebooks)もご覧ください。
* 🤗 Accelerateを使用する場合は、[🤗 Accelerate DeepSpeedガイド](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed)を参照してください。

## torch.compileの使用

PyTorch 2.0は新しいコンパイル関数を導入しました。これは既存のPyTorchコードを変更する必要はありませんが、1行のコードを追加することでコードを最適化できます：`model = torch.compile(model)`。

[`Trainer`]を使用する場合、[`TrainingArguments`]内の`torch_compile`オプションを渡すだけです：


```python
training_args = TrainingArguments(torch_compile=True, **default_args)
```

`torch.compile`は、既存のPyTorchプログラムからグラフを自動的に作成するためにPythonのフレーム評価APIを使用します。グラフをキャプチャした後、異なるバックエンドを展開して最適化されたエンジンに変換できます。
詳細およびベンチマークについては、[PyTorchドキュメント](https://pytorch.org/get-started/pytorch-2.0/)を参照してください。

`torch.compile`には、オプションの依存関係を持つ成長中のバックエンドのリストがあり、`torchdynamo.list_backends()`を呼び出して確認できます。最も一般的に使用される一部のバックエンドは次のとおりです。

**デバッグ用バックエンド**：
* `dynamo.optimize("eager")` - 抽出されたGraphModuleを実行するためにPyTorchを使用します。これはTorchDynamoの問題をデバッグする際に非常に役立ちます。
* `dynamo.optimize("aot_eager")` - コンパイラーを使用しないAotAutogradを使用してAotAutogradの抽出されたフォワードおよびバックワードグラフに対して単にPyTorch eagerを使用します。これはデバッグに役立ち、高速化は期待できません。

**トレーニングおよび推論バックエンド**：
* `dynamo.optimize("inductor")` - TorchInductorバックエンドを使用し、AotAutogradおよびcudagraphsを活用してコード生成されたTritonカーネルを使用します [詳細はこちら](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
* `dynamo.optimize("nvfuser")` -  nvFuser with TorchScriptを使用します。 [詳細はこちら](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
* `dynamo.optimize("aot_nvfuser")` -  nvFuser with AotAutogradを使用します。 [詳細はこちら](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
* `dynamo.optimize("aot_cudagraphs")` - AotAutogradを使用してcudagraphsを使用します。 [詳細はこちら](https://github.com/pytorch/torchdynamo/pull/757)

**推論専用バックエンド**：
* `dynamo.optimize("ofi")` - Torchscriptの`optimize_for_inference`を使用します。 [詳細はこちら](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html)
* `dynamo.optimize("fx2trt")` - Nvidia TensorRTを使用した推論の最適化にNvidia TensorRTを使用します。 [詳細はこちら](https://pytorch.org/TensorRT/tutorials/getting_started_with_fx_path.html)
* `dynamo.optimize("onnxrt")` - CPU/GPUでの推論にONNX Runtimeを使用します。 [詳細はこちら](https://onnxruntime.ai/)
* `dynamo.optimize("ipex")` - CPUでの推論にIPEXを使用します。 [詳細はこちら](https://github.com/intel/intel-extension-for-pytorch)

🤗 Transformersを使用した`torch.compile`の使用例については、この[ブログ記事](https://www.philschmid.de/getting-started-pytorch-2-0-transformers)をご覧ください。

## Using 🤗 Accelerate

[🤗 Accelerate](https://huggingface.co/docs/accelerate/index)を使用すると、上記の方法を使用しながらトレーニングループを完全に制御でき、基本的には純粋なPyTorchでループを書くことができます。

次に、[`TrainingArguments`]内で方法を組み合わせた場合を想

```py
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
    **default_args,
)
```

🤗 Accelerateを使用した完全なトレーニングループの例は、ほんの数行のコードです：

```py
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader

dataloader = DataLoader(ds, batch_size=training_args.per_device_train_batch_size)

if training_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

accelerator = Accelerator(fp16=training_args.fp16)
model, optimizer, dataloader = accelerator.prepare(model, adam_bnb_optim, dataloader)

model.train()
for step, batch in enumerate(dataloader, start=1):
    loss = model(**batch).loss
    loss = loss / training_args.gradient_accumulation_steps
    accelerator.backward(loss)
    if step % training_args.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

まず、データセットを[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)でラップします。
次に、モデルの[`~PreTrainedModel.gradient_checkpointing_enable`]メソッドを呼び出すことで勾配チェックポイントを有効にできます。
[`Accelerator`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator)を初期化する際に、混合精度トレーニングを使用するかどうかを[`prepare`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.prepare)の呼び出しで指定し、複数のGPUを使用する場合、`prepare`の間にデータローダーもワーカー間で分散されます。同じ[8ビットオプティマイザ](#8-bit-adam)を前の例から使用します。

最後に、主要なトレーニングループを追加できます。`backward`の呼び出しは🤗 Accelerateによって処理されることに注意してください。また、勾配の蓄積がどのように機能するかも確認できます。損失を正規化しているため、蓄積の最後に平均を得て、十分なステップがあると最適化が実行されます。

これらの最適化技術を🤗 Accelerateを使用して実装するのは、わずかなコード行で行うことができ、トレーニングループの柔軟性が向上します。すべての機能の詳細については、[Accelerateのドキュメント](https://huggingface.co/docs/accelerate/index)を参照してください。

## Efficient Software Prebuilds

PyTorchの[pipとcondaビルド](https://pytorch.org/get-started/locally/#start-locally)は、PyTorchを実行するのに十分なcudaツールキットで事前にビルドされていますが、cuda拡張をビルドする必要がある場合には不十分です。

時折、追加の努力が必要な場合があります。たとえば、事前にコンパイルされていない`apex`などのライブラリを使用している場合です。また、システム全体で適切なcudaツールキットをインストールする方法を見つけることが難しい場合もあります。
これらのシナリオに対処するために、PyTorchとNVIDIAはcuda拡張がすでに事前にビルドされているNGC dockerコンテナの新しいバージョンをリリースしました。プログラムをインストールするだけで、そのまま実行できます。

このアプローチは、PyTorchのソースを調整したり、新しいカスタマイズされたビルドを作成したりしたい場合にも役立ちます。
欲しいdockerイメージバージョンを見つけるには、まず[PyTorchのリリースノート](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/)から始め、最新の月次リリースのいずれかを選択します。希望のリリースのリリースノートに移動し、環境のコンポーネントが必要なものと一致していることを確認します（NVIDIA Driverの要件も含む！）、その文書の一番上に行き、対応するNGCページに移動します。なぜかわからない場合は、[すべてのPyTorch NGCイメージのインデックス](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)です。

次に、dockerイメージをダウンロードして展開する手順に従います。

## Mixture of Experts

最近の論文によれば、Transformerモデルに専門家の混合（MoE）を統合することで、トレーニング速度が4〜5倍向上し、推論も高速化されることが報告されています。

より多くのパラメータがより良いパフォーマンスにつながることがわかっているため、この技術はトレーニングコストを増やすことなくパラメータの数を桁違いに増やすことを可能にします。

このアプローチでは、他のFFN層の代わりにMoE層が配置され、各専門家をトークンの位置に応じてバランスよくトレーニングするゲート関数で構成されます。

![MoE Transformer 2x block](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/perf-moe-transformer.png)

（出典: [GLAM](https://ai.googleblog.com/2021/12/more-efficient-in-context-learning-with.html)）

このアプローチの主な欠点は、GPUメモリをほぼ桁違いに多く必要とすることです。メモリ要件がはるかに大きいことがそのまま反映されます。より高いメモリ要件を克服する方法については、さまざまな蒸留およびアプローチが提案されています。

ただし、直接のトレードオフがあります。数人の専門家を使用してベースモデルを2〜3倍小さくすることで、5倍小さなモデルにし、トレーニング速度を適度に向上させ、メモリ要件を適度に増やすことができます。

関連するほとんどの論文および実装はTensorflow/TPUを中心に構築されています。

- [GShard: Conditional Computation and Automatic Shardingを活用した巨大モデルのスケーリング](https://arxiv.org/abs/2006.16668)
- [Switch Transformers: シンプルで効率的なスパース性を備えたトリリオンパラメータモデルへのスケーリング](https://arxiv.org/abs/2101.03961)
- [GLaM: Generalist Language Model (GLaM)](https://ai.googleblog.com/2021/12/more-efficient-in-context-learning-with.html)

PytorchにはDeepSpeedが構築したものもあります: [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://arxiv.org/abs/2201.05596)、[Mixture of Experts](https://www.deepspeed.ai/tutorials/mixture-of-experts/) - ブログ記事: [1](https://www.microsoft.com/en-us/research/blog/deepspeed-powers-8x-larger-moe-model-training-with-high-performance/)、[2](https://www.microsoft.com/en-us/research/publication/scalable-and-efficient-moe-training-for-multitask-multilingual-models/)、大規模なTransformerベースの自然言語生成モデルの具体的な展開については、[ブログ記事](https://www.deepspeed.ai/2021/12/09/deepspeed-moe-nlg.html)、[Megatron-Deepspeedブランチ](https://github.com/microsoft/Megatron-DeepSpeed/tree/moe-training)を参照してください。


## PyTorchネイティブアテンションとFlash Attentionの使用

PyTorch 2.0では、ネイティブの[`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)（SDPA）がリリースされ、[メモリ効率の高いアテンション](https://arxiv.org/abs/2112.05682)や[フラッシュアテンション](https://arxiv.org/abs/2205.14135)などの融合されたGPUカーネルの使用を可能にします。

[`optimum`](https://github.com/huggingface/optimum)パッケージをインストールした後、関連する内部モジュールを置き換えて、PyTorchのネイティブアテンションを使用できます。以下のように設定します：


```python
model = model.to_bettertransformer()
```

変換後、通常通りモデルをトレーニングしてください。

<Tip warning={true}>

PyTorchネイティブの`scaled_dot_product_attention`演算子は、`attention_mask`が提供されていない場合にのみFlash Attentionにディスパッチできます。

デフォルトでは、トレーニングモードでBetterTransformer統合はマスクサポートを削除し、バッチトレーニングにパディングマスクが必要ないトレーニングにしか使用できません。これは、例えばマスク言語モデリングや因果言語モデリングのような、バッチトレーニングにパディングマスクが不要なトレーニングの場合に該当します。BetterTransformerはパディングマスクが必要なタスクに対するモデルの微調整には適していません。

</Tip>

SDPAを使用したアクセラレーションとメモリの節約について詳しく知りたい場合は、この[ブログ記事](https://pytorch.org/blog/out-of-the-box-acceleration/)をチェックしてください。
