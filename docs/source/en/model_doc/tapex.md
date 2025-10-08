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
*This model was released on 2021-07-16 and added to Hugging Face Transformers on 2023-06-20.*

# TAPEX

[TAPEX](https://huggingface.co/papers/2107.07653) pre-trains a BART model using a synthetic corpus of executable SQL queries and their outputs to address the scarcity of high-quality tabular data. By mimicking a SQL executor, TAPEX achieves state-of-the-art results on four benchmark datasets: SQA, WTQ, WikiSQL, and TabFact, demonstrating significant improvements over previous approaches.

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/tapex-large-finetuned-wtq", dtype="auto")

data = {"Plant Species": ["Oak Tree", "Sunflower", "Algae"], "Energy Production Rate (ATP/photosynthesis)": ["High", "Medium", "Very High"]}
table = pd.DataFrame.from_dict(data)
question = "which plant species produces the most energy?"

encoding = tokenizer(table, question, return_tensors="pt")

outputs = model.generate(**encoding)

predicted_answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(predicted_answer)
```

</hfoption>
</hfoptions>

## TapexTokenizer

[[autodoc]] TapexTokenizer
    - __call__
    - save_vocabulary

