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

# 데이터 콜레이터(Data Collator)[[data-collator]]

데이터 콜레이터는 데이터셋 요소들의 리스트를 입력으로 사용하여 배치를 형성하는 객체입니다. 이러한 요소들은 `train_dataset` 또는 `eval_dataset의` 요소들과 동일한 타입 입니다. 배치를 구성하기 위해, 데이터 콜레이터는 (패딩과 같은) 일부 처리를 적용할 수 있습니다. [`DataCollatorForLanguageModeling`]과 같은 일부 콜레이터는 형성된 배치에 (무작위 마스킹과 같은) 일부 무작위 데이터 증강도 적용합니다. 사용 예시는 [예제 스크립트](../examples)나 [예제 노트북](../notebooks)에서 찾을 수 있습니다.


## 기본 데이터 콜레이터[[transformers.default_data_collator]]

[[autodoc]] data.data_collator.default_data_collator

## DefaultDataCollator[[transformers.DefaultDataCollator]]

[[autodoc]] data.data_collator.DefaultDataCollator

## DataCollatorWithPadding[[transformers.DataCollatorWithPadding]]

[[autodoc]] data.data_collator.DataCollatorWithPadding

## DataCollatorForTokenClassification[[transformers.DataCollatorForTokenClassification]]

[[autodoc]] data.data_collator.DataCollatorForTokenClassification

## DataCollatorForSeq2Seq[[transformers.DataCollatorForSeq2Seq]]

[[autodoc]] data.data_collator.DataCollatorForSeq2Seq

## DataCollatorForLanguageModeling[[transformers.DataCollatorForLanguageModeling]]

[[autodoc]] data.data_collator.DataCollatorForLanguageModeling
    - numpy_mask_tokens
    - tf_mask_tokens
    - torch_mask_tokens

## DataCollatorForWholeWordMask[[transformers.DataCollatorForWholeWordMask]]

[[autodoc]] data.data_collator.DataCollatorForWholeWordMask
    - numpy_mask_tokens
    - tf_mask_tokens
    - torch_mask_tokens

## DataCollatorForPermutationLanguageModeling[[transformers.DataCollatorForPermutationLanguageModeling]]

[[autodoc]] data.data_collator.DataCollatorForPermutationLanguageModeling
    - numpy_mask_tokens
    - tf_mask_tokens
    - torch_mask_tokens

## DataCollatorWithFlatteningtransformers.DataCollatorWithFlattening

[[autodoc]] data.data_collator.DataCollatorWithFlattening

