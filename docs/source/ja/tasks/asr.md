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

# Automatic speech recognition

[[open-in-colab]]

<Youtube id="TksaY_FDgnk"/>

自動音声認識 (ASR) は音声信号をテキストに変換し、一連の音声入力をテキスト出力にマッピングします。 Siri や Alexa などの仮想アシスタントは ASR モデルを使用してユーザーを日常的に支援しており、ライブキャプションや会議中のメモ取りなど、他にも便利なユーザー向けアプリケーションが数多くあります。

このガイドでは、次の方法を説明します。

1. [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) データセットの [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) を微調整して、音声をテキストに書き起こします。
2. 微調整したモデルを推論に使用します。

> [!TIP]
> このタスクと互換性のあるすべてのアーキテクチャとチェックポイントを確認するには、[タスクページ](https://huggingface.co/tasks/automatic-speech-recognition) を確認することをお勧めします。

始める前に、必要なライブラリがすべてインストールされていることを確認してください。

```bash
pip install transformers datasets evaluate jiwer
```

モデルをアップロードしてコミュニティと共有できるように、Hugging Face アカウントにログインすることをお勧めします。プロンプトが表示されたら、トークンを入力してログインします。

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Load MInDS-14 dataset

まず、🤗 データセット ライブラリから [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) データセットの小さいサブセットをロードします。これにより、完全なデータセットのトレーニングにさらに時間を費やす前に、実験してすべてが機能することを確認する機会が得られます。

```py
>>> from datasets import load_dataset, Audio

>>> minds = load_dataset("PolyAI/minds14", name="en-US", split="train[:100]")
```

[`~Dataset.train_test_split`] メソッドを使用して、データセットの `train` 分割をトレイン セットとテスト セットに分割します。

```py
>>> minds = minds.train_test_split(test_size=0.2)
```

次に、データセットを見てみましょう。

```py
>>> minds
DatasetDict({
    train: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 16
    })
    test: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 4
    })
})
```

データセットには`lang_id`や`english_transcription`などの多くの有用な情報が含まれていますが、このガイドでは「`audio`」と「`transciption`」に焦点を当てます。 [`~datasets.Dataset.remove_columns`] メソッドを使用して他の列を削除します。

```py
>>> minds = minds.remove_columns(["english_transcription", "intent_class", "lang_id"])
```

もう一度例を見てみましょう。

```py
>>> minds["train"][0]
{'audio': {'array': array([-0.00024414,  0.        ,  0.        , ...,  0.00024414,
          0.00024414,  0.00024414], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
  'sampling_rate': 8000},
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
 'transcription': "hi I'm trying to use the banking app on my phone and currently my checking and savings account balance is not refreshing"}
```

次の 2 つのフィールドがあります。

- `audio`: 音声ファイルをロードしてリサンプリングするために呼び出す必要がある音声信号の 1 次元の `array`。
- `transcription`: ターゲットテキスト。

## Preprocess

次のステップでは、Wav2Vec2 プロセッサをロードしてオーディオ信号を処理します。

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
```

MInDS-14 データセットのサンプリング レートは 8000kHz です (この情報は [データセット カード](https://huggingface.co/datasets/PolyAI/minds14) で確認できます)。つまり、データセットを再サンプリングする必要があります。事前トレーニングされた Wav2Vec2 モデルを使用するには、16000kHz に設定します。

```py
>>> minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
>>> minds["train"][0]
{'audio': {'array': array([-2.38064706e-04, -1.58618059e-04, -5.43987835e-06, ...,
          2.78103951e-04,  2.38446111e-04,  1.18740834e-04], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
  'sampling_rate': 16000},
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
 'transcription': "hi I'm trying to use the banking app on my phone and currently my checking and savings account balance is not refreshing"}
```

上の `transcription` でわかるように、テキストには大文字と小文字が混在しています。 Wav2Vec2 トークナイザーは大文字のみでトレーニングされるため、テキストがトークナイザーの語彙と一致することを確認する必要があります。

```py
>>> def uppercase(example):
...     return {"transcription": example["transcription"].upper()}


>>> minds = minds.map(uppercase)
```

次に、次の前処理関数を作成します。

1. `audio`列を呼び出して、オーディオ ファイルをロードしてリサンプリングします。
2. オーディオ ファイルから `input_values` を抽出し、プロセッサを使用して `transcription` 列をトークン化します。

```py
>>> def prepare_dataset(batch):
...     audio = batch["audio"]
...     batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["transcription"])
...     batch["input_length"] = len(batch["input_values"][0])
...     return batch
```

データセット全体に前処理関数を適用するには、🤗 Datasets [`~datasets.Dataset.map`] 関数を使用します。 `num_proc` パラメータを使用してプロセスの数を増やすことで、`map` を高速化できます。 [`~datasets.Dataset.remove_columns`] メソッドを使用して、不要な列を削除します。

```py
>>> encoded_minds = minds.map(prepare_dataset, remove_columns=minds.column_names["train"], num_proc=4)
```

🤗 Transformers には ASR 用のデータ照合器がないため、[`DataCollat​​orWithPadding`] を調整してサンプルのバッチを作成する必要があります。また、テキストとラベルが (データセット全体ではなく) バッチ内の最も長い要素の長さに合わせて動的に埋め込まれ、均一な長さになります。 `padding=True` を設定すると、`tokenizer` 関数でテキストを埋め込むことができますが、動的な埋め込みの方が効率的です。

他のデータ照合器とは異なり、この特定のデータ照合器は、`input_values`と `labels`」に異なるパディング方法を適用する必要があります。

```py
>>> import torch

>>> from dataclasses import dataclass, field
>>> from typing import Any, Dict, List, Optional, Union


>>> @dataclass
... class DataCollatorCTCWithPadding:
...     processor: AutoProcessor
...     padding: Union[bool, str] = "longest"

...     def __call__(self, features: list[dict[str, Union[list[int], torch.Tensor]]]) -> dict[str, torch.Tensor]:
...         # split inputs and labels since they have to be of different lengths and need
...         # different padding methods
...         input_features = [{"input_values": feature["input_values"][0]} for feature in features]
...         label_features = [{"input_ids": feature["labels"]} for feature in features]

...         batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

...         labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

...         # replace padding with -100 to ignore loss correctly
...         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

...         batch["labels"] = labels

...         return batch
```

次に、`DataCollat​​orForCTCWithPadding` をインスタンス化します。

```py
>>> data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")
```

## Evaluate

トレーニング中にメトリクスを含めると、多くの場合、モデルのパフォーマンスを評価するのに役立ちます。 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) ライブラリを使用して、評価メソッドをすばやくロードできます。このタスクでは、[単語エラー率](https://huggingface.co/spaces/evaluate-metric/wer) (WER) メトリクスを読み込みます (🤗 Evaluate [クイック ツアー](https://huggingface.co/docs/evaluate/a_quick_tour) を参照して、メトリクスをロードして計算する方法の詳細を確認してください)。

```py
>>> import evaluate

>>> wer = evaluate.load("wer")
```

次に、予測とラベルを [`~evaluate.EvaluationModule.compute`] に渡して WER を計算する関数を作成します。

```py
>>> import numpy as np


>>> def compute_metrics(pred):
...     pred_logits = pred.predictions
...     pred_ids = np.argmax(pred_logits, axis=-1)

...     pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

...     pred_str = processor.batch_decode(pred_ids)
...     label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

...     wer = wer.compute(predictions=pred_str, references=label_str)

...     return {"wer": wer}
```

これで`compute_metrics`関数の準備が整いました。トレーニングをセットアップするときにこの関数に戻ります。

## Train

> [!TIP]
> [`Trainer`] を使用したモデルの微調整に慣れていない場合は、[ここ](../training#train-with-pytorch-trainer) の基本的なチュートリアルをご覧ください。

これでモデルのトレーニングを開始する準備が整いました。 [`AutoModelForCTC`] で Wav2Vec2 をロードします。 `ctc_loss_reduction` パラメータで適用する削減を指定します。多くの場合、デフォルトの合計ではなく平均を使用する方が適切です。

```py
>>> from transformers import AutoModelForCTC, TrainingArguments, Trainer

>>> model = AutoModelForCTC.from_pretrained(
...     "facebook/wav2vec2-base",
...     ctc_loss_reduction="mean",
...     pad_token_id=processor.tokenizer.pad_token_id,
... )
```

この時点で残っている手順は次の 3 つだけです。

1. [`TrainingArguments`] でトレーニング ハイパーパラメータを定義します。唯一の必須パラメータは、モデルの保存場所を指定する `output_dir` です。 `push_to_hub=True`を設定して、このモデルをハブにプッシュします (モデルをアップロードするには、Hugging Face にサインインする必要があります)。各エポックの終了時に、[`トレーナー`] は WER を評価し、トレーニング チェックポイントを保存します。
2. トレーニング引数を、モデル、データセット、トークナイザー、データ照合器、および `compute_metrics` 関数とともに [`Trainer`] に渡します。
3. [`~Trainer.train`] を呼び出してモデルを微調整します。

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_asr_mind_model",
...     per_device_train_batch_size=8,
...     gradient_accumulation_steps=2,
...     learning_rate=1e-5,
...     warmup_steps=500,
...     max_steps=2000,
...     gradient_checkpointing=True,
...     fp16=True,
...     group_by_length=True,
...     eval_strategy="steps",
...     per_device_eval_batch_size=8,
...     save_steps=1000,
...     eval_steps=1000,
...     logging_steps=25,
...     load_best_model_at_end=True,
...     metric_for_best_model="wer",
...     greater_is_better=False,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=encoded_minds["train"],
...     eval_dataset=encoded_minds["test"],
...     processing_class=processor,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

トレーニングが完了したら、 [`~transformers.Trainer.push_to_hub`] メソッドを使用してモデルをハブに共有し、誰もがモデルを使用できるようにします。

```py
>>> trainer.push_to_hub()
```


> [!TIP]
> 自動音声認識用にモデルを微調整する方法のより詳細な例については、英語 ASR および英語のこのブログ [投稿](https://huggingface.co/blog/fine-tune-wav2vec2-english) を参照してください。多言語 ASR については、この [投稿](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2) を参照してください。

## Inference

モデルを微調整したので、それを推論に使用できるようになりました。

推論を実行したい音声ファイルをロードします。必要に応じて、オーディオ ファイルのサンプリング レートをモデルのサンプリング レートと一致するようにリサンプリングすることを忘れないでください。

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sampling_rate = dataset.features["audio"].sampling_rate
>>> audio_file = dataset[0]["audio"]["path"]
```

推論用に微調整されたモデルを試す最も簡単な方法は、それを [`pipeline`] で使用することです。モデルを使用して自動音声認識用の`pipeline`をインスタンス化し、オーディオ ファイルをそれに渡します。

```py
>>> from transformers import pipeline

>>> transcriber = pipeline("automatic-speech-recognition", model="stevhliu/my_awesome_asr_minds_model")
>>> transcriber(audio_file)
{'text': 'I WOUD LIKE O SET UP JOINT ACOUNT WTH Y PARTNER'}
```

> [!TIP]
> 転写はまあまあですが、もっと良くなる可能性があります。さらに良い結果を得るには、より多くの例でモデルを微調整してみてください。

必要に応じて、「パイプライン」の結果を手動で複製することもできます。


プロセッサをロードしてオーディオ ファイルと文字起こしを前処理し、`input`を PyTorch テンソルとして返します。

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("stevhliu/my_awesome_asr_mind_model")
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
```

Pass your inputs to the model and return the logits:

```py
>>> from transformers import AutoModelForCTC

>>> model = AutoModelForCTC.from_pretrained("stevhliu/my_awesome_asr_mind_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

最も高い確率で予測された `input_ids` を取得し、プロセッサを使用して予測された `input_ids` をデコードしてテキストに戻します。


```py
>>> import torch

>>> predicted_ids = torch.argmax(logits, dim=-1)
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription
['I WOUL LIKE O SET UP JOINT ACOUNT WTH Y PARTNER']
```

