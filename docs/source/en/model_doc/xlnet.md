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
*This model was released on 2019-06-19 and added to Hugging Face Transformers on 2020-11-16 and contributed by [thomwolf](https://huggingface.co/thomwolf).*

# XLNet

[XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://huggingface.co/papers/1906.08237) extends Transformer-XL by using an autoregressive method to learn bidirectional contexts through the expected likelihood of all input sequence permutations. This approach addresses BERT's limitations by incorporating dependencies between masked positions and reducing pretrain-finetune discrepancies. XLNet outperforms BERT across various tasks, including question answering, natural language inference, sentiment analysis, and document ranking.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-classification", model="xlnet/xlnet-large-cased", dtype="auto")
pipeline("Plants are amazing because they can create energy from the sun.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("xlnet/xlnet-large-cased", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-large-cased")

inputs = tokenizer("Plants are amazing because they can create energy from the sun.", return_tensors="pt")
outputs = model(**inputs)
predicted_class_id = outputs.logits.argmax(dim=-1).item()
label = model.config.id2label[predicted_class_id]
print(f"Predicted label: {label}")
```

</hfoption>
</hfoptions>

## XLNetConfig

[[autodoc]] XLNetConfig

## XLNetTokenizer

[[autodoc]] XLNetTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## XLNetTokenizerFast

[[autodoc]] XLNetTokenizerFast

## XLNet specific outputs

[[autodoc]] models.xlnet.modeling_xlnet.XLNetModelOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput

] models.xlnet.modeling_tf_xlnet.TFXLNetLMHeadModelOutput

] models.xlnet.modeling_tf_xlnet.TFXLNetForMultipleChoiceOutput

] models.xlnet.modeling_tf_xlnet.TFXLNetForQuestionAnsweringSimpleOutput

## XLNetModel

[[autodoc]] XLNetModel
    - forward

## XLNetLMHeadModel

[[autodoc]] XLNetLMHeadModel
    - forward

## XLNetForSequenceClassification

[[autodoc]] XLNetForSequenceClassification
    - forward

## XLNetForMultipleChoice

[[autodoc]] XLNetForMultipleChoice
    - forward

## XLNetForTokenClassification

[[autodoc]] XLNetForTokenClassification
    - forward

## XLNetForQuestionAnsweringSimple

[[autodoc]] XLNetForQuestionAnsweringSimple
    - forward

## XLNetForQuestionAnswering

[[autodoc]] XLNetForQuestionAnswering
    - forward

