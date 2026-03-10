<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
    </div>
</div>

# BERT

[BERT](https://huggingface.co/papers/1810.04805) (Bidirectional Encoder Representations 
from Transformers) is a transformer model pretrained by Google on a large corpus of English 
text. Unlike previous models that read text left-to-right or right-to-left, BERT reads in 
both directions at once, giving it a deeper understanding of language context.

BERT is pretrained using two objectives: Masked Language Modeling (MLM), where it learns 
to predict missing words in a sentence, and Next Sentence Prediction (NSP), where it learns 
to understand relationships between sentences. This makes BERT extremely versatile and it 
can be fine-tuned for many NLP tasks like classification, question answering, and named 
entity recognition.

You can find all the original BERT checkpoints under the 
[BERT](https://huggingface.co/collections/google/bert-release-66f0885277ffe4568ef9ee45) 
collection.

> [!TIP]
> This model was contributed by the 
> [Hugging Face team](https://huggingface.co/huggingface).
>
> Click on the BERT models in the right sidebar for more examples of how to use 
> BERT for different NLP tasks like classification and question answering.

The example below demonstrates how to use BERT for masked language modeling with 
[`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">
```py
from transformers import pipeline

pipe = pipeline("fill-mask", model="google-bert/bert-base-uncased")
result = pipe("Paris is the [MASK] of France.")
print(result)
```

</hfoption>
<hfoption id="AutoModel">
```py
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")

inputs = tokenizer("Paris is the [MASK] of France.", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
print(tokenizer.decode(predicted_token_id))
```

</hfoption>
</hfoptions>

## Notes

- BERT uses `[CLS]` token at the start and `[SEP]` token to separate sentences.
- Use `bert-base-uncased` for lowercase text and `bert-base-cased` if your text has 
  important capitalization.
- BERT has a maximum input length of 512 tokens.
```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
print(tokenizer.model_max_length)  # 512
```

## Resources

- [BERT original paper](https://arxiv.org/abs/1810.04805)
- [Google BERT GitHub repository](https://github.com/google-research/bert)
- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
