<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 🤗 Accelerate を用いた分散学習

モデルが大きくなるにつれて、限られたハードウェアでより大きなモデルを訓練し、訓練速度を大幅に上昇させるための方法として並列処理が浮上してきました。1台のマシンに複数のGPUがあっても、複数のマシンにまたがる複数のGPUがあっても、あらゆるタイプの分散処理セットアップ上でユーザーが簡単に 🤗 Transformers モデルを訓練できるように、 Hugging Face では [🤗 Accelerate](https://huggingface.co/docs/accelerate) ライブラリを作成しました。このチュートリアルでは、PyTorch の訓練ループをカスタマイズして、分散処理環境での訓練を可能にする方法について学びます。

## セットアップ

はじめに 🤗 Accelerate をインストールしましょう:

```bash
pip install accelerate
```

そしたらインポートして [`~accelerate.Accelerator`] オブジェクトを作成しましょう。[`~accelerate.Accelerator`] は分散処理セットアップを自動的に検出し、訓練のために必要な全てのコンポーネントを初期化します。モデルをデバイスに明示的に配置する必要はありません。

```py
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
```

## Accelerate する準備をしましょう

次に、関連する全ての訓練オブジェクトを [`~accelerate.Accelerator.prepare`] メソッドに渡します。これには、訓練と評価それぞれのDataloader、モデル、optimizer が含まれます:

```py
>>> train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
...     train_dataloader, eval_dataloader, model, optimizer
... )
```

## Backward

最後に訓練ループ内の `loss.backward()` を 🤗 Accelerate の [`~accelerate.Accelerator.backward`] メソッドで置き換えます：

```py
>>> for epoch in range(num_epochs):
...     for batch in train_dataloader:
...         outputs = model(**batch)
...         loss = outputs.loss
...         accelerator.backward(loss)

...         optimizer.step()
...         lr_scheduler.step()
...         optimizer.zero_grad()
...         progress_bar.update(1)
```

以下のコードで確認できる通り、訓練ループに4行のコードを追加するだけで分散学習が可能です！

```diff
+ from accelerate import Accelerator
  from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

+ accelerator = Accelerator()

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  optimizer = AdamW(model.parameters(), lr=3e-5)

- device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
- model.to(device)

+ train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
+     train_dataloader, eval_dataloader, model, optimizer
+ )

  num_epochs = 3
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )

  progress_bar = tqdm(range(num_training_steps))

  model.train()
  for epoch in range(num_epochs):
      for batch in train_dataloader:
-         batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)
```

## 訓練する

関連するコードを追加したら、スクリプトまたは Colaboratory などのノートブックで訓練を開始します。

### スクリプトで訓練する

スクリプトから訓練をしている場合は、設定ファイルを作成・保存するために以下のコマンドを実行してください:

```bash
accelerate config
```

そして次のようにして訓練を開始します:

```bash
accelerate launch train.py
```

### ノートブックで訓練する

Colaboratory の TPU の利用をお考えの場合、🤗 Accelerate はノートブック上で実行することもできます。訓練に必要な全てのコードを関数に含め、[`~accelerate.notebook_launcher`] に渡してください:

```py
>>> from accelerate import notebook_launcher

>>> notebook_launcher(training_function)
```

🤗 Accelerate と豊富な機能についてもっと知りたい方は[ドキュメント](https://huggingface.co/docs/accelerate)を参照してください。
