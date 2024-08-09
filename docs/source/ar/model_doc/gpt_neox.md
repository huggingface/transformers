# GPT-NeoX

## نظرة عامة

نحن نقدم GPT-NeoX-20B، وهو نموذج لغة احتمالي ذو 20 مليار معامل تم تدريبه على Pile، وستكون أوزانه متاحة بحرية وعلانية للجمهور من خلال ترخيص متساهل. وهو، على حد علمنا، أكبر نموذج احتمالي كثيف متاح للجمهور في وقت تقديمه. في هذا العمل، نصف بنية GPT-NeoX-20B والتدريب ونقيم أداءه على مجموعة من مهام فهم اللغة والرياضيات والمعرفة. نجد أن GPT-NeoX-20B هو سبب قوي للغاية ويحقق مكاسب أكبر بكثير في الأداء عند تقييمه بخمسة تسديدات مقارنة بنماذج GPT-3 و FairSeq ذات الحجم المماثل. نقوم بإتاحة المصدر المفتوح لتعليمات البرمجية للتدريب والتقييم، بالإضافة إلى أوزان النموذج، على [https://github.com/EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox).

تمت قيادة تطوير النموذج بواسطة Sid Black وStella Biderman وEric Hallahan، وتم تدريب النموذج بدعم سخي من [CoreWeave](https://www.coreweave.com/).

تم تدريب GPT-NeoX-20B باستخدام fp16، لذلك يوصى بتهيئة النموذج على النحو التالي:

```python
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b").half().cuda()
```

لدى GPT-NeoX-20B أيضًا محدد مواقع مختلف عن المحدد المستخدم في GPT-J-6B وGPT-Neo. يقوم محدد المواقع الجديد بتخصيص رموز إضافية لمحارف المسافة البيضاء، مما يجعل النموذج أكثر ملاءمة لمهام معينة مثل توليد التعليمات البرمجية.

## مثال على الاستخدام

يمكن استخدام طريقة `generate()` لتوليد نص باستخدام نموذج GPT Neo.

```python
>>> from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

>>> model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
>>> tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

>>> prompt = "GPTNeoX20B is a 20B-parameter autoregressive Transformer model developed by EleutherAI."

>>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

>>> gen_tokens = model.generate(
...     input_ids,
...     do_sample=True,
...     temperature=0.9,
...     max_length=100,
... )
>>> gen_text = tokenizer.batch_decode(gen_tokens)[0]```

## استخدام Flash Attention 2

Flash Attention 2 هو إصدار أسرع وأكثر تحسينًا من النموذج.

### التثبيت

أولاً، تحقق مما إذا كان جهازك متوافقًا مع Flash Attention 2. يمكن العثور على أحدث قائمة بالأجهزة المتوافقة في [الوثائق الرسمية](https://github.com/Dao-AILab/flash-attention#installation-and-features). إذا لم يكن جهازك متوافقًا مع Flash Attention 2، فيمكنك الاستفادة من تحسينات نواة الاهتمام من خلال دعم Better Transformer المشمول [أعلاه](https://huggingface.co/docs/transformers/main/en/model_doc/bark#using-better-transformer).

بعد ذلك، قم بتثبيت أحدث إصدار من Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

### الاستخدام

لتحميل نموذج باستخدام Flash Attention 2، يمكننا تمرير الحجة `attn_implementation="flash_attention_2"` إلى [`.from_pretrained`](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained). سنقوم أيضًا بتحميل النموذج في نصف الدقة (على سبيل المثال `torch.float16`)، حيث يؤدي ذلك إلى تقليل استخدام الذاكرة وسرعة الاستدلال بشكل كبير دون أي تدهور تقريبًا في جودة الصوت:

```python
>>> from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(device)
...
```

### تسريع الأداء المتوقع

فيما يلي رسم بياني لتسريع الأداء المتوقع الذي يقارن وقت الاستدلال النقي بين التنفيذ الأصلي في المحولات باستخدام نقطة تفتيش `stockmark/gpt-neox-japanese-1.4b` وإصدار Flash Attention 2 من النموذج باستخدام طول تسلسل يبلغ 2048.

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/gpt-neox-1.8b-speedup.jpg">
</div>


## Using Scaled Dot Product Attention (SDPA)

يتضمن PyTorch عامل تشغيل أصلي لمنتج النقطة (SDPA) كجزء من "torch.nn.functional". هذه الوظيفة
يشمل العديد من التطبيقات التي يمكن تطبيقها اعتمادًا على المدخلات والأجهزة المستخدمة. انظر
[الوثائق الرسمية](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
أو [استدلال وحدة معالجة الرسومات](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)
الصفحة لمزيد من المعلومات.

يتم استخدام SDPA افتراضيًا لـ `torch>=2.1.1` عندما يكون التنفيذ متاحًا، ولكن يمكنك أيضًا تعيين
`attn_implementation="sdpa"` في `from_pretrained()` لطلب استخدام SDPA بشكل صريح.

```python
from transformers import GPTNeoXForCausalLM
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", torch_dtype=torch.float16, attn_implementation="sdpa")
...
```

للحصول على أفضل عمليات التسريع، نوصي بتحميل النموذج بنصف الدقة (على سبيل المثال، `torch.float16` أو `torch.bfloat16`).

على معيار محلي (rtx3080ti-16GB، PyTorch 2.2.1، OS Ubuntu 22.04) باستخدام 'float16' مع
[pythia-410m-deduped](https://huggingface.co/EleutherAI/pythia-410m-deduped)، رأينا
متابعة التسريعات أثناء التدريب والاستدلال.


### Training
| Batch size |    Seq len | Time per batch (Eager - s) |    Time per batch (SDPA - s) | Speedup (%) | Eager peak mem (MB) | SDPA peak mem (MB) |    Mem saving (%) |
|-----------:|-----------:|---------------------------:|-----------------------------:|------------:|--------------------:|-------------------:|------------------:|
|          1 |        128 |                      0.024 |                        0.019 |      28.945 |             1789.95 |            1789.95 |                 0 |
|          1 |        256 |                      0.039 |                        0.031 |       23.18 |             1845.83 |            1844.84 |             0.053 |
|          1 |        512 |                       0.08 |                        0.055 |      45.524 |             2278.38 |            1953.76 |            16.615 |
|          1 |       1024 |                       0.19 |                        0.102 |      86.777 |             4772.36 |            2408.35 |            98.159 |
|          1 |       2048 |                      0.565 |                        0.204 |     177.098 |             13484.1 |            3882.01 |           247.348 |
|          2 |        128 |                      0.037 |                        0.032 |      15.121 |             1843.86 |            1844.78 |             -0.05 |
|          2 |        256 |                      0.067 |                        0.055 |      21.706 |             1999.72 |            1951.67 |             2.462 |
|          2 |        512 |                      0.144 |                        0.096 |      50.046 |             3613.16 |            2406.77 |            50.125 |
|          2 |       1024 |                      0.366 |                        0.193 |      89.666 |             8707.55 |            3878.86 |           124.487 |
|          2 |       2048 |                        OOM |                        0.379 |           / |                 OOM |            6825.13 | SDPA does not OOM |
|          4 |        128 |                       0.06 |                        0.054 |      11.539 |              1947.6 |            1952.06 |            -0.228 |
|          4 |        256 |                      0.119 |                        0.093 |      28.072 |             3008.39 |            2405.99 |            25.038 |
|          4 |        512 |                      0.275 |                        0.187 |      47.145 |             6290.58 |            3877.29 |            62.242 |
|          4 |       1024 |                        OOM |                         0.36 |           / |                 OOM |            6821.98 | SDPA does not OOM |
|          4 |       2048 |                        OOM |                        0.731 |           / |                 OOM |            12705.1 | SDPA does not OOM |

### Inference
|    Batch size |      Seq len |    Per token latency Eager (ms) |    Per token latency SDPA (ms) |    Speedup (%) |    Mem Eager (MB) |   Mem SDPA (MB) |    Mem saved (%) |
|--------------:|-------------:|--------------------------------:|-------------------------------:|---------------:|------------------:|----------------:|-----------------:|
|             1 |          128 |                           6.569 |                          5.858 |          12.14 |           974.831 |         974.826 |                0 |
|             1 |          256 |                           7.009 |                          5.863 |         19.542 |           1029.01 |         1028.08 |             0.09 |
|             1 |          512 |                           7.157 |                          5.965 |         19.983 |           1137.54 |         1137.52 |            0.001 |
|             1 |         1024 |                           7.523 |                          6.506 |         15.637 |            1329.3 |         1329.26 |            0.003 |
|             1 |         2048 |                           9.271 |                          9.205 |          0.713 |           1752.47 |         1734.51 |            1.036 |
|             2 |          128 |                           7.239 |                          5.959 |         21.493 |            1044.8 |         1028.37 |            1.597 |
|             2 |          256 |                           7.228 |                          6.036 |         19.757 |           1167.32 |         1137.73 |            2.601 |
|             2 |          512 |                           7.538 |                          6.693 |         12.628 |           1352.93 |         1329.55 |            1.758 |
|             2 |         1024 |                           8.916 |                          8.632 |          3.291 |           1752.56 |         1734.62 |            1.034 |
|             2 |         2048 |                          12.628 |                         12.606 |          0.181 |           2558.72 |          2545.8 |            0.508 |
|             4 |          128 |                           7.278 |                          6.046 |         20.373 |           1168.41 |         1137.79 |            2.691 |
|             4 |          256 |                           7.614 |                          6.588 |         15.574 |            1353.1 |         1329.79 |            1.753 |
|             4 |          512 |                           8.798 |                          8.144 |          8.028 |           1752.76 |         1734.85 |            1.032 |
|             4 |         1024 |                          11.765 |                         11.303 |           4.09 |           2558.96 |         2546.04 |            0.508 |
|             4 |         2048 |                          19.568 |                         17.735 |          10.33 |            4175.5 |         4165.26 |            0.246 |



## الموارد

- [دليل مهمة نمذجة اللغة السببية](../tasks/language_modeling)

## GPTNeoXConfig

[[autodoc]] GPTNeoXConfig

## GPTNeoXTokenizerFast

[[autodoc]] GPTNeoXTokenizerFast

## GPTNeoXModel

[[autodoc]] GPTNeoXModel

- forward

## GPTNeoXForCausalLM

[[autodoc]] GPTNeoXForCausalLM

- forward

## GPTNeoXForQuestionAnswering

[[autodoc]] GPTNeoXForQuestionAnswering

- forward

## GPTNeoXForSequenceClassification

[[autodoc]] GPTNeoXForSequenceClassification

- forward

## GPTNeoXForTokenClassification

[[autodoc]] GPTNeoXForTokenClassification

- forward