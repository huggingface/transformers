# ERNIE

## نظرة عامة

ERNIE هي سلسلة من النماذج القوية التي اقترحتها Baidu، خاصة في المهام الصينية، بما في ذلك [ERNIE1.0](https://arxiv.org/abs/1904.09223)، [ERNIE2.0](https://ojs.aaai.org/index.php/AAAI/article/view/6428)، [ERNIE3.0](https://arxiv.org/abs/2107.02137)، [ERNIE-Gram](https://arxiv.org/abs/2010.12148)، [ERNIE-health](https://arxiv.org/abs/2110.07244)، وغيرها.

هذه النماذج مساهمة من [nghuyong](https://huggingface.co/nghuyong) ويمكن العثور على الكود الرسمي في [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) (في PaddlePaddle).

### مثال على الاستخدام

خذ `ernie-1.0-base-zh` كمثال:

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")
model = AutoModel.from_pretrained("nghuyong/ernie-1.0-base-zh")
```

### نقاط تفتيش النموذج

| اسم النموذج | اللغة | الوصف |
| :----: | :----: | :----: |
| ernie-1.0-base-zh | الصينية | Layer:12, Heads:12, Hidden:768 |
| ernie-2.0-base-en | الإنجليزية | Layer:12, Heads:12, Hidden:768 |
| ernie-2.0-large-en | الإنجليزية | Layer:24, Heads:16, Hidden:1024 |
| ernie-3.0-base-zh | الصينية | Layer:12, Heads:12, Hidden:768 |
| ernie-3.0-medium-zh | الصينية | Layer:6, Heads:12, Hidden:768 |
| ernie-3.0-mini-zh | الصينية | Layer:6, Heads:12, Hidden:384 |
| ernie-3.0-micro-zh | الصينية | Layer:4, Heads:12, Hidden:384 |
| ernie-3.0-nano-zh | الصينية | Layer:4, Heads:12, Hidden:312 |
| ernie-health-zh | الصينية | Layer:12, Heads:12, Hidden:768 |
| ernie-gram-zh | الصينية | Layer:12, Heads:12, Hidden:768 |

يمكنك العثور على جميع النماذج المدعومة من مركز نماذج Huggingface: [huggingface.co/nghuyong](https://huggingface.co/nghuyong)، وتفاصيل النموذج من المستودع الرسمي لـ Paddle: [PaddleNLP](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers/ERNIE/contents.html) و [ERNIE](https://github.com/PaddlePaddle/ERNIE/blob/repro).

## الموارد

- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهمة تصنيف الرموز](../tasks/token_classification)
- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهمة نمذجة اللغة السببية](../tasks/language_modeling)
- [دليل مهمة نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل المهمة متعددة الخيارات](../tasks/multiple_choice)

## ErnieConfig

[[autodoc]] ErnieConfig

- all

## مخرجات Ernie المحددة

[[autodoc]] models.ernie.modeling_ernie.ErnieForPreTrainingOutput

## ErnieModel

[[autodoc]] ErnieModel

- forward

## ErnieForPreTraining

[[autodoc]] ErnieForPreTraining

- forward

## ErnieForCausalLM

[[autodoc]] ErnieForCausalLM

- forward

## ErnieForMaskedLM

[[autodoc]] ErnieForMaskedLM

- forward

## ErnieForNextSentencePrediction

[[autodoc]] ErnieForNextSentencePrediction

- forward

## ErnieForSequenceClassification

[[autodoc]] ErnieForSequenceClassification

- forward

## ErnieForMultipleChoice

[[autodoc]] ErnieForMultipleChoice

- forward

## ErnieForTokenClassification

[[autodoc]] ErnieForTokenClassification

- forward

## ErnieForQuestionAnswering

[[autodoc]] ErnieForQuestionAnswering

- forward