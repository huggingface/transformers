
これでモデルのトレーニングを開始する準備が整いました。 [`AutoModelForQuestionAnswering`] を使用して DitilBERT をロードします。

```py
>>> from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

>>> model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
```

この時点で残っている手順は次の 3 つだけです。

1. [`TrainingArguments`] でトレーニング ハイパーパラメータを定義します。唯一の必須パラメータは、モデルの保存場所を指定する `output_dir` です。 `push_to_hub=True`を設定して、このモデルをハブにプッシュします (モデルをアップロードするには、Hugging Face にサインインする必要があります)。
2. トレーニング引数をモデル、データセット、トークナイザー、データ照合器とともに [`Trainer`] に渡します。
3. [`~Trainer.train`] を呼び出してモデルを微調整します。

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_qa_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_squad["train"],
...     eval_dataset=tokenized_squad["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
... )

>>> trainer.train()
```

トレーニングが完了したら、 [`~transformers.Trainer.push_to_hub`] メソッドを使用してモデルをハブに共有し、誰もがモデルを使用できるようにします。


```py
>>> trainer.push_to_hub()
```

<Tip>

質問応答用のモデルを微調整する方法の詳細な例については、対応するドキュメントを参照してください。
[PyTorch ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb)
または [TensorFlow ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb)。

</Tip>

## Evaluate

質問応答の評価には、大量の後処理が必要です。時間がかかりすぎないように、このガイドでは評価ステップを省略しています。 [`Trainer`] はトレーニング中に評価損失を計算するため、モデルのパフォーマンスについて完全に分からないわけではありません。

もっと時間があり、質問応答用のモデルを評価する方法に興味がある場合は、[質問応答](https://huggingface.co/course/chapter7/7?fw=pt#postprocessing) の章を参照してください。 🤗ハグフェイスコースから！

## Inference

モデルを微調整したので、それを推論に使用できるようになりました。

質問と、モデルに予測させたいコンテキストを考え出します。

```py
>>> question = "How many programming languages does BLOOM support?"
>>> context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
```

推論用に微調整されたモデルを試す最も簡単な方法は、それを [`pipeline`] で使用することです。モデルを使用して質問応答用の`pipeline`をインスタンス化し、それにテキストを渡します。

```py
>>> from transformers import pipeline

>>> question_answerer = pipeline("question-answering", model="my_awesome_qa_model")
>>> question_answerer(question=question, context=context)
{'score': 0.2058267742395401,
 'start': 10,
 'end': 95,
 'answer': '176 billion parameters and can generate text in 46 languages natural languages and 13'}
```

必要に応じて、`pipeline`の結果を手動で複製することもできます。


テキストをトークン化して PyTorch テンソルを返します。

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_qa_model")
>>> inputs = tokenizer(question, context, return_tensors="pt")
```

入力をモデルに渡し、`logits`を返します。


```py
>>> import torch
>>> from transformers import AutoModelForQuestionAnswering

>>> model = AutoModelForQuestionAnswering.from_pretrained("my_awesome_qa_model")
>>> with torch.no_grad():
...     outputs = model(**inputs)
```

モデル出力から開始位置と終了位置の最も高い確率を取得します。

```py
>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()
```

予測されたトークンをデコードして答えを取得します。

```py
>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens)
'176 billion parameters and can generate text in 46 languages natural languages and 13'
```
