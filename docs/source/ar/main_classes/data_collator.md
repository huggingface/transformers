# مجمع البيانات

مجمعو البيانات هم أشياء ستشكل دفعة باستخدام قائمة من عناصر مجموعة البيانات كإدخال. هذه العناصر من نفس نوع عناصر "train_dataset" أو "eval_dataset".

ليكون قادرًا على بناء دفعات، قد يطبق مجمعو البيانات بعض المعالجة (مثل الحشو). بعضها (مثل [DataCollatorForLanguageModeling]) يطبق أيضًا بعض التعزيزات العشوائية للبيانات (مثل التعتيم العشوائي) على الدفعة المشكلة.

يمكن العثور على أمثلة للاستخدام في [سكريبتات المثال](../examples) أو [دفاتر الملاحظات](../notebooks).

## مجمع البيانات الافتراضي

[[autodoc]] data.data_collator.default_data_collator

## DefaultDataCollator

[[autodoc]] data.data_collator.DefaultDataCollator

## DataCollatorWithPadding

[[autodoc]] data.data_collator.DataCollatorWithPadding

## DataCollatorForTokenClassification

[[autodoc]] data.data_collator.DataCollatorForTokenClassification

## DataCollatorForSeq2Seq

[[autodoc]] data.data_collator.DataCollatorForSeq2Seq

## DataCollatorForLanguageModeling

[[autodoc]] data.data_collator.DataCollatorForLanguageModeling

- numpy_mask_tokens
- tf_mask_tokens
- torch_mask_tokens

## DataCollatorForWholeWordMask

[[autodoc]] data.data_collator.DataCollatorForWholeWordMask

- numpy_mask_tokens
- tf_mask_tokens
- torch_mask_tokens

## DataCollatorForPermutationLanguageModeling

[[autodoc]] data.data_collator.DataCollatorForPermutationLanguageModeling

- numpy_mask_tokens
- tf_mask_tokens
- torch_mask_tokens