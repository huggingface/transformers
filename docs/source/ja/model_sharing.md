<!--
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ このファイルはMarkdownですが、Hugging Faceのドキュメントビルダー（MDXに類似）向けの特定の構文を含んでいるため、Markdownビューアーで適切にレンダリングされないことがあります。

-->

# Share a Model

最後の2つのチュートリアルでは、PyTorch、Keras、および🤗 Accelerateを使用してモデルをファインチューニングする方法を示しました。次のステップは、モデルをコミュニティと共有することです！Hugging Faceでは、知識とリソースを公開的に共有し、人工知能を誰にでも提供することを信じています。他の人々が時間とリソースを節約できるように、モデルをコミュニティと共有することを検討することをお勧めします。

このチュートリアルでは、訓練済みまたはファインチューニングされたモデルを[Model Hub](https://huggingface.co/models)に共有する2つの方法を学びます：

- プログラムでファイルをHubにプッシュする。
- ウェブインターフェースを使用してファイルをHubにドラッグアンドドロップする。

<iframe width="560" height="315" src="https://www.youtube.com/embed/XvSGPZFEjDY" title="YouTube video player"
frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope;
picture-in-picture" allowfullscreen></iframe>

<Tip>

コミュニティとモデルを共有するには、[huggingface.co](https://huggingface.co/join)でアカウントが必要です。既存の組織に参加したり、新しい組織を作成したりすることもできます。

</Tip>

## Repository Features

Model Hub上の各リポジトリは、通常のGitHubリポジトリのように動作します。リポジトリはバージョニング、コミット履歴、違いの視覚化の機能を提供します。

Model Hubの組み込みバージョニングはgitおよび[git-lfs](https://git-lfs.github.com/)に基づいています。言い換えれば、モデルを1つのリポジトリとして扱うことができ、より大きなアクセス制御とスケーラビリティを実現します。バージョン管理には*リビジョン*があり、コミットハッシュ、タグ、またはブランチ名で特定のモデルバージョンをピン留めする方法です。

その結果、`revision`パラメータを使用して特定のモデルバージョンをロードできます：

```py
>>> model = AutoModel.from_pretrained(
...     "julien-c/EsperBERTo-small", revision="v2.0.1"  # タグ名、またはブランチ名、またはコミットハッシュ
... )
```

ファイルはリポジトリ内で簡単に編集でき、コミット履歴と差分を表示できます：

![vis_diff](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vis_diff.png)

## Set Up

モデルをHubに共有する前に、Hugging Faceの認証情報が必要です。ターミナルへのアクセス権がある場合、🤗 Transformersがインストールされている仮想環境で以下のコマンドを実行します。これにより、アクセストークンがHugging Faceのキャッシュフォルダに保存されます（デフォルトでは `~/.cache/` に保存されます）：

```bash
huggingface-cli login
```

JupyterやColaboratoryのようなノートブックを使用している場合、[`huggingface_hub`](https://huggingface.co/docs/hub/adding-a-library)ライブラリがインストールされていることを確認してください。
このライブラリを使用すると、Hubとプログラム的に対話できます。

```bash
pip install huggingface_hub
```

次に、`notebook_login`を使用してHubにサインインし、[こちらのリンク](https://huggingface.co/settings/token)にアクセスしてログインに使用するトークンを生成します：

```python
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Convert a Model for all frameworks

異なるフレームワークで作業している他のユーザーがあなたのモデルを使用できるようにするために、
PyTorchおよびTensorFlowのチェックポイントでモデルを変換してアップロードすることをお勧めします。
このステップをスキップすると、ユーザーは異なるフレームワークからモデルをロードできますが、
モデルをオンザフライで変換する必要があるため、遅くなります。

別のフレームワーク用にチェックポイントを変換することは簡単です。
PyTorchとTensorFlowがインストールされていることを確認してください（インストール手順については[こちら](installation)を参照）し、
その後、他のフレームワーク向けに特定のタスク用のモデルを見つけます。

<frameworkcontent>
<pt>
TensorFlowからPyTorchにチェックポイントを変換するには、`from_tf=True`を指定します：

```python
>>> pt_model = DistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_tf=True)
>>> pt_model.save_pretrained("path/to/awesome-name-you-picked")
```
</pt>
<tf>

指定して、PyTorchからTensorFlowにチェックポイントを変換するには `from_pt=True` を使用します：

```python
>>> tf_model = TFDistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_pt=True)
```

新しいTensorFlowモデルとその新しいチェックポイントを保存できます：

```python
>>> tf_model.save_pretrained("path/to/awesome-name-you-picked")
```
</tf>
<tf>
<jax>
Flaxでモデルが利用可能な場合、PyTorchからFlaxへのチェックポイントの変換も行うことができます：

```py
>>> flax_model = FlaxDistilBertForSequenceClassification.from_pretrained(
...     "path/to/awesome-name-you-picked", from_pt=True
... )
```
</jax>
</frameworkcontent>

## Push a model during traning

<frameworkcontent>
<pt>
<Youtube id="Z1-XMy-GNLQ"/>

モデルをHubにプッシュすることは、追加のパラメーターまたはコールバックを追加するだけで簡単です。
[ファインチューニングチュートリアル](training)から思い出してください、[`TrainingArguments`]クラスはハイパーパラメーターと追加のトレーニングオプションを指定する場所です。
これらのトレーニングオプションの1つに、モデルを直接Hubにプッシュする機能があります。[`TrainingArguments`]で`push_to_hub=True`を設定します：

```py
>>> training_args = TrainingArguments(output_dir="my-awesome-model", push_to_hub=True)
```

Pass your training arguments as usual to [`Trainer`]:

```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

[`Trainer`]に通常通りトレーニング引数を渡します：

```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

ファインチューニングが完了したら、[`Trainer`]で[`~transformers.Trainer.push_to_hub`]を呼び出して、トレーニング済みモデルをHubにプッシュします。🤗 Transformersは、トレーニングのハイパーパラメータ、トレーニング結果、およびフレームワークのバージョンを自動的にモデルカードに追加します！

```py
>>> trainer.push_to_hub()
```

</pt>
<tf>

[`PushToHubCallback`]を使用してモデルをHubに共有します。[`PushToHubCallback`]関数には、次のものを追加します：

- モデルの出力ディレクトリ。
- トークナイザ。
- `hub_model_id`、つまりHubのユーザー名とモデル名。

```python
>>> from transformers import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="./your_model_save_path", tokenizer=tokenizer, hub_model_id="your-username/my-awesome-model"
... )
```

🤗 Transformersは[`fit`](https://keras.io/api/models/model_training_apis/)にコールバックを追加し、トレーニング済みモデルをHubにプッシュします：

```py
>>> model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3, callbacks=push_to_hub_callback)
```
</tf>
</frameworkcontent>

## `push_to_hub` 関数を使用する

また、モデルを直接Hubにアップロードするために、`push_to_hub` を呼び出すこともできます。

`push_to_hub` でモデル名を指定します：

```py
>>> pt_model.push_to_hub("my-awesome-model")
```

これにより、ユーザー名の下にモデル名 `my-awesome-model` を持つリポジトリが作成されます。
ユーザーは、`from_pretrained` 関数を使用してモデルをロードできます：

```py
>>> from transformers import AutoModel

>>> model = AutoModel.from_pretrained("your_username/my-awesome-model")
```

組織に所属し、モデルを組織名のもとにプッシュしたい場合、`repo_id` にそれを追加してください：

```python
>>> pt_model.push_to_hub("my-awesome-org/my-awesome-model")
```

`push_to_hub`関数は、モデルリポジトリに他のファイルを追加するためにも使用できます。例えば、トークナイザをモデルリポジトリに追加します：

```py
>>> tokenizer.push_to_hub("my-awesome-model")
```

あるいは、ファインチューニングされたPyTorchモデルのTensorFlowバージョンを追加したいかもしれません：

```python
>>> tf_model.push_to_hub("my-awesome-model")
```

Hugging Faceプロフィールに移動すると、新しく作成したモデルリポジトリが表示されるはずです。**Files**タブをクリックすると、リポジトリにアップロードしたすべてのファイルが表示されます。

リポジトリにファイルを作成およびアップロードする方法の詳細については、Hubドキュメンテーション[こちら](https://huggingface.co/docs/hub/how-to-upstream)を参照してください。

## Upload with the web interface

コードを書かずにモデルをアップロードしたいユーザーは、Hubのウェブインターフェースを使用してモデルをアップロードできます。[huggingface.co/new](https://huggingface.co/new)を訪れて新しいリポジトリを作成します：

![new_model_repo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/new_model_repo.png)

ここから、モデルに関するいくつかの情報を追加します：

- リポジトリの**所有者**を選択します。これはあなた自身または所属している組織のいずれかです。
- モデルの名前を選択します。これはリポジトリの名前にもなります。
- モデルが公開か非公開かを選択します。
- モデルのライセンス使用方法を指定します。

その後、**Files**タブをクリックし、**Add file**ボタンをクリックしてリポジトリに新しいファイルをアップロードします。次に、ファイルをドラッグアンドドロップしてアップロードし、コミットメッセージを追加します。

![upload_file](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/upload_file.png)

## Add a model card

ユーザーがモデルの機能、制限、潜在的な偏り、倫理的な考慮事項を理解できるようにするために、モデルリポジトリにモデルカードを追加してください。モデルカードは`README.md`ファイルで定義されます。モデルカードを追加する方法：

* 手動で`README.md`ファイルを作成およびアップロードする。
* モデルリポジトリ内の**Edit model card**ボタンをクリックする。

モデルカードに含めるべき情報の例については、DistilBert [モデルカード](https://huggingface.co/distilbert/distilbert-base-uncased)をご覧ください。`README.md`ファイルで制御できる他のオプション、例えばモデルの炭素フットプリントやウィジェットの例などについての詳細は、[こちらのドキュメンテーション](https://huggingface.co/docs/hub/models-cards)を参照してください。





