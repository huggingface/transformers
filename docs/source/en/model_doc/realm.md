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
*This model was released on 2020-02-10 and added to Hugging Face Transformers on 2023-06-20 and contributed by [qqaatw](https://huggingface.co/qqaatw).*

> [!WARNING]
> This model is in maintenance mode only, we don’t accept any new PRs changing its code.
>
> If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2. You can do so by running the following command: pip install -U transformers==4.40.2.

# REALM

[REALM: Retrieval-Augmented Language Model Pre-Training](https://huggingface.co/papers/2002.08909) enhances language model pre-training by integrating a latent knowledge retriever. This retriever allows the model to access and utilize documents from a large corpus like Wikipedia during pre-training, fine-tuning, and inference. The model is trained in an unsupervised manner using masked language modeling, with the retrieval step considered during backpropagation across millions of documents. REALM significantly outperforms existing models on Open-domain Question Answering benchmarks, offering improvements of 4-16% in accuracy. It also provides benefits in interpretability and modularity.

<hfoptions id="usage">
<hfoption id="RealmForOpenQA">

```py
import torch
from transformers import RealmForOpenQA, RealmRetriever, AutoTokenizer

retriever = RealmRetriever.from_pretrained("google/realm-orqa-nq-openqa")
tokenizer = AutoTokenizer.from_pretrained("google/realm-orqa-nq-openqa")
model = RealmForOpenQA.from_pretrained("google/realm-orqa-nq-openqa", retriever=retriever, dtype="auto")

question = "How do plants create energy?"
question_ids = tokenizer([question], return_tensors="pt")
answer_ids = tokenizer(
    ["photosynthesis"],
    add_special_tokens=False,
    return_token_type_ids=False,
    return_attention_mask=False,
).input_ids

reader_output, predicted_answer_ids = model(**question_ids, answer_ids=answer_ids, return_dict=False)
print(tokenizer.decode(predicted_answer_ids))
```

</hfoption>
</hfoptions>

## RealmConfig

[[autodoc]] RealmConfig

## RealmTokenizer

[[autodoc]] RealmTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary
    - batch_encode_candidates

## RealmTokenizerFast

[[autodoc]] RealmTokenizerFast
    - batch_encode_candidates

## RealmRetriever

[[autodoc]] RealmRetriever

## RealmEmbedder

[[autodoc]] RealmEmbedder
    - forward

## RealmScorer

[[autodoc]] RealmScorer
    - forward

## RealmKnowledgeAugEncoder

[[autodoc]] RealmKnowledgeAugEncoder
    - forward

## RealmReader

[[autodoc]] RealmReader
    - forward

## RealmForOpenQA

[[autodoc]] RealmForOpenQA
    - block_embedding_to
    - forward

