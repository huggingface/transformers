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
*This model was released on 2020-04-05 and added to Hugging Face Transformers on 2020-12-15 and contributed by [nielsr](https://huggingface.co/nielsr).*

# TAPAS

[TAPAS](https://huggingface.co/papers/2004.02349) is a table question-answering model that predicts answers directly from table cells rather than generating intermediate logical forms. It extends BERT to encode tables alongside text and is pre-trained on a large corpus of text-table pairs from Wikipedia, then trained end-to-end using weak supervision with denotations. TAPAS can optionally apply aggregation operators to selected cells and achieves state-of-the-art or competitive performance on several semantic parsing benchmarks, including SQA, WIKISQL, and WIKITQ. Its architecture also enables straightforward transfer learning between datasets, improving performance with minimal additional training.

<hfoptions id="usage">
<hfoption id="TapasForQuestionAnswering">

```py
import torch
import pandas as pd
from transformers import AutoTokenizer, TapasForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq", dtype="auto")

data = {
    "Plant Species": ["Rose", "Sunflower", "Oak Tree"],
    "Height (cm)": ["150", "300", "2500"],
    "Flowering Season": ["Spring", "Summer", "Spring"],
    "Water Needs": ["Moderate", "High", "Low"],
}
table = pd.DataFrame.from_dict(data)
queries = ["How tall is a Sunflower?", "Which plants flower in Spring?", "What is the water requirement for Oak Tree?"]

inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
outputs = model(**inputs)

logits = outputs.logits
logits_aggregation = outputs.logits_aggregation

predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
    inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
)

for i, query in enumerate(queries):
    print(f"\nQuestion: {query}")
    coordinates = predicted_answer_coordinates[i]
    if coordinates:
        answers = []
        for coord in coordinates:
            row_idx, col_idx = coord
            if row_idx < len(table) and col_idx < len(table.columns):
                answer = table.iloc[row_idx, col_idx]
                answers.append(answer)
            print(f"Answer: {', '.join(answers)}")
```

</hfoption>
</hfoptions>

## TAPAS specific outputs

[[autodoc]] models.tapas.modeling_tapas.TableQuestionAnsweringOutput

## TapasConfig

[[autodoc]] TapasConfig

## TapasTokenizer

[[autodoc]] TapasTokenizer
    - __call__
    - convert_logits_to_predictions
    - save_vocabulary

## TapasModel

[[autodoc]] TapasModel
    - forward
    
## TapasForMaskedLM

[[autodoc]] TapasForMaskedLM
    - forward

## TapasForSequenceClassification

[[autodoc]] TapasForSequenceClassification
    - forward
    
## TapasForQuestionAnswering

[[autodoc]] TapasForQuestionAnswering
    - forward

