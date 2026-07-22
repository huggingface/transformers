<!--Copyright 2020 The HuggingFace Team. All rights reserved.



Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with

the License. You may obtain a copy of the License at



http://www.apache.org/licenses/LICENSE-2.0



Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on

an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

specific language governing permissions and limitations under the License.



⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be

rendered properly in your Markdown viewer.



-->

*このモデルは2019年10月2日にHF papersで公開され、2020年11月16日にHugging Face Transformersへ追加されました。*



<div style="float: right;">

    <div class="flex flex-wrap space-x-1">

        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">

        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">

    </div>

</div>



# DistilBERT



[DistilBERT](https://huggingface.co/papers/1910.01108) は、より小さく、推論が高速で、学習に必要な計算量も少ないモデルを作るために、知識蒸留によって事前学習されています。事前学習では、言語モデリング損失、蒸留損失、コサイン距離損失という3つの損失を組み合わせた目的関数を使います。これにより、DistilBERTは、より大きなTransformer言語モデルに近い性能を示します。



元のDistilBERTチェックポイントはすべて、[DistilBERT](https://huggingface.co/distilbert) organization で確認できます。



> [!TIP]

> 右側のサイドバーにあるDistilBERTモデルをクリックすると、さまざまな言語タスクにDistilBERTを適用する例をさらに確認できます。



以下の例では、[`Pipeline`]、[`AutoModel`]、コマンドラインを使ってテキストを分類する方法を示します。



<hfoptions id="usage">



<hfoption id="Pipeline">



```python

from transformers import pipeline





classifier = pipeline(

    task="text-classification",

    model="distilbert-base-uncased-finetuned-sst-2-english",

    device=0

)



result = classifier("I love using Hugging Face Transformers!")

print(result)

# Output: [{'label': 'POSITIVE', 'score': 0.9998}]

```



</hfoption>



<hfoption id="AutoModel">



```python

import torch



from transformers import AutoModelForSequenceClassification, AutoTokenizer





tokenizer = AutoTokenizer.from_pretrained(

    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",

)

model = AutoModelForSequenceClassification.from_pretrained(

    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",

    device_map="auto",

    attn_implementation="sdpa"

)

inputs = tokenizer("I love using Hugging Face Transformers!", return_tensors="pt").to(model.device)



with torch.no_grad():

    outputs = model(**inputs)



predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()

predicted_label = model.config.id2label[predicted_class_id]

print(f"Predicted label: {predicted_label}")

```



</hfoption>



</hfoptions>



## 注意



- DistilBERTには `token_type_ids` がないため、どのトークンがどのセグメントに属するかを指定する必要はありません。セグメントを分けるには、区切りトークンである `tokenizer.sep_token` または `[SEP]` を使ってください。

- DistilBERTには、入力位置を選択するためのオプション、つまり `position_ids` 入力はありません。ただし、必要であれば追加することも可能なので、このオプションが必要な場合は知らせてください。



## DistilBertConfig



[[autodoc]] DistilBertConfig



## DistilBertTokenizer



[[autodoc]] DistilBertTokenizer



## DistilBertTokenizerFast



[[autodoc]] DistilBertTokenizerFast



## DistilBertModel



[[autodoc]] DistilBertModel

    - forward



## DistilBertForMaskedLM



[[autodoc]] DistilBertForMaskedLM

    - forward



## DistilBertForSequenceClassification



[[autodoc]] DistilBertForSequenceClassification

    - forward



## DistilBertForMultipleChoice



[[autodoc]] DistilBertForMultipleChoice

    - forward



## DistilBertForTokenClassification



[[autodoc]] DistilBertForTokenClassification

    - forward



## DistilBertForQuestionAnswering



[[autodoc]] DistilBertForQuestionAnswering

    - forward