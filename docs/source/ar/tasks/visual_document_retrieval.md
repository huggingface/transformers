<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
# استرجاع المستندات البصرية (Visual document retrieval)

يمكن أن تحتوي المستندات على بيانات متعددة الوسائط إذا تضمّنت رسومًا بيانية وجداول ومواد بصرية بالإضافة إلى النص. يكون استرجاع المعلومات من هذه المستندات أمرًا صعبًا لأن نماذج استرجاع النص وحدها لا يمكنها التعامل مع البيانات البصرية، كما أن نماذج استرجاع الصور تفتقر إلى الدقة الحبيبية وقدرات معالجة المستندات.

يمكن لاستر جاع المستندات البصرية المساعدة في استرجاع المعلومات من جميع أنواع المستندات، بما في ذلك تعزيز الاسترجاع بالتوليد متعدد الوسائط (RAG). تقبل هذه النماذج مستندات (على شكل صور) ونصوصًا وتحسب درجات التشابه بينها.

يوضّح هذا الدليل كيفية فهرسة واسترجاع المستندات باستخدام [ColPali](../model_doc/colpali).  

> [!TIP]
> للحالات على نطاق واسع، قد ترغب في فهرسة واسترجاع المستندات باستخدام قاعدة بيانات متجهات.

تأكد من تثبيت Transformers وDatasets.

```bash
pip install -q datasets transformers
```

سنفهرس مجموعة بيانات لمستندات متعلّقة برصد الأجسام الطائرة المجهولة (UFO). نقوم بتصفية الأمثلة التي تفتقد العمود الذي نهتم به. تحتوي المجموعة على عدة أعمدة؛ نهتم بالعمود `specific_detail_query` الذي يتضمن ملخصًا قصيرًا للمستند، وبالعمود `image` الذي يحتوي على مستنداتنا.

```python
from datasets import load_dataset

dataset = load_dataset("davanstrien/ufo-ColPali")
dataset = dataset["train"]
dataset = dataset.filter(lambda example: example["specific_detail_query"] is not None)
dataset
```
```
Dataset({
    features: ['image', 'raw_queries', 'broad_topical_query', 'broad_topical_explanation', 'specific_detail_query', 'specific_detail_explanation', 'visual_element_query', 'visual_element_explanation', 'parsed_into_json'],
    num_rows: 2172
})
```

لنحمّل النموذج والمُعالج (processor).

```python
import torch
from transformers import ColPaliForRetrieval, ColPaliProcessor

model_name = "vidore/colpali-v1.2-hf"

processor = ColPaliProcessor.from_pretrained(model_name)

model = ColPaliForRetrieval.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
).eval()
```

مرّر استعلام النص إلى المُعالج وأرجِع تمثيلات النص المفهرسة من النموذج. لبحث صورة-إلى-نص، استبدل معامل `text` في [`ColPaliProcessor`] بالمعامل `images` لتمرير الصور.

```python
inputs = processor(text="a document about Mars expedition").to("cuda")
with torch.no_grad():
  text_embeds = model(**inputs, return_tensors="pt").embeddings
```

افهرس الصور دون اتصال، وأثناء الاستدلال أرجِع تمثيلات استعلام النص للحصول على أقرب تمثيلات الصور.

احفظ الصورة وتمثيلات الصورة بكتابتها إلى مجموعة البيانات باستخدام [`~datasets.Dataset.map`] كما هو موضح أدناه. أضِف عمود `embeddings` الذي يحتوي على التمثيلات المفهرسة. تمثيلات ColPali تستهلك مساحة تخزين كبيرة، لذا أزِلها من الـGPU وخزّنها في الـCPU كمصفوفات NumPy.

```python
ds_with_embeddings = dataset.map(lambda example: {'embeddings': model(**processor(images=example["image"]).to("cuda"), return_tensors="pt").embeddings.to(torch.float32).detach().cpu().numpy()})
```

للاستدلال عبر الإنترنت، أنشئ دالة للبحث في تمثيلات الصور على دفعات واسترجاع أعلى k صور ذات صلة. الدالة أدناه تُرجع الفهارس في مجموعة البيانات ودرجاتها لمجموعة بيانات مفهرسة مُعطاة، وتمثيل نصي، وعدد النتائج العليا، وحجم الدفعة.

```python
def find_top_k_indices_batched(dataset, text_embedding, processor, k=10, batch_size=4):
    scores_and_indices = []

    for start_idx in range(0, len(dataset), batch_size):

        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset[start_idx:end_idx]        
        batch_embeddings = [torch.tensor(emb[0], dtype=torch.float32) for emb in batch["embeddings"]]
        scores = processor.score_retrieval(text_embedding.to("cpu").to(torch.float32), batch_embeddings)

        if hasattr(scores, "tolist"):
            scores = scores.tolist()[0]

        for i, score in enumerate(scores):
            scores_and_indices.append((score, start_idx + i))

    sorted_results = sorted(scores_and_indices, key=lambda x: -x[0])

    topk = sorted_results[:k]
    indices = [idx for _, idx in topk]
    scores = [score for score, _ in topk]

    return indices, scores
```

ولّد تمثيلات النص ومرّرها إلى الدالة أعلاه لإرجاع فهارس مجموعة البيانات والدرجات.

```python
with torch.no_grad():
  text_embeds = model(**processor(text="a document about Mars expedition").to("cuda"), return_tensors="pt").embeddings
indices, scores = find_top_k_indices_batched(ds_with_embeddings, text_embeds, processor, k=3, batch_size=4)
print(indices, scores)
```

```
([440, 442, 443],
 [14.370786666870117,
  13.675487518310547,
  12.9899320602417])
```

اعرض الصور للاطلاع على المستندات المتعلقة بالمريخ.

```python
for i in indices:
  display(dataset[i]["image"])
```

<div style="display: flex; align-items: center;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/doc_1.png" 
         alt="Document 1" 
         style="height: 200px; object-fit: contain; margin-right: 10px;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/doc_2.png" 
         alt="Document 2" 
         style="height: 200px; object-fit: contain;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/doc_3.png" 
         alt="Document 3" 
         style="height: 200px; object-fit: contain;">
</div>
