<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Quanto

<Tip>

جرب Quanto + transformers مع هذا [notebook](https://colab.research.google.com/drive/16CXfVmtdQvciSh9BopZUDYcmXCDpvgrT?usp=sharing)!

</Tip>


[🤗 Quanto](https://github.com/huggingface/quanto) هي مكتبة PyTorch للتحويل الكمي متعددة الاستخدامات. طريقة التكميم المستخدمة هي التكميم الخطي. يوفر Quanto العديد من الميزات الفريدة مثل:

- تكميم الأوزان (`float8`,`int8`,`int4`,`int2`)
- تكميم التنشيط (`float8`,`int8`)
- لا يعتمد على طريقة الإدخال (مثل CV، LLM)
- لا يعتمد على الجهاز (مثل CUDA، MPS، CPU)
- التوافق مع `torch.compile`
- من السهل إضافة نواة مخصصة لجهاز محدد
- يدعم التدريب الواعي بالتكميم
<!-- أضف رابطًا إلى المنشور -->

قبل البدء، تأكد من تثبيت المكتبات التالية:

```bash
pip install quanto accelerate transformers
```

الآن يمكنك تحويل نموذج إلى الشكل الكمي عن طريق تمرير [`QuantoConfig`] object في طريقة [`~PreTrainedModel.from_pretrained`]. تعمل هذه الطريقة مع أي نموذج في أي طريقة للإدخال، طالما أنه يحتوي على طبقات `torch.nn.Linear`. 

لا يدعم التكامل مع مكتبة المحولات سوى تكميم الأوزان. بالنسبة لحالات الاستخدام الأكثر تعقيدًا مثل تكميم التنشيط والمعايرة والتدريب الواعي بالتكميم، يجب استخدام مكتبة [quanto](https://github.com/huggingface/quanto) بدلاً من ذلك. 

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig

model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = QuantoConfig(weights="int8")
quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", quantization_config=quantization_config)
```

لاحظ أن التسلسل الهرمي غير مدعوم حتى الآن مع المحولات ولكنها قادمة قريبًا! إذا كنت تريد حفظ النموذج، فيمكنك استخدام مكتبة quanto بدلاً من ذلك.

تستخدم مكتبة Quanto خوارزمية التكميم الخطي للتحويل الكمي. على الرغم من أن هذه تقنية تحويل كمي أساسية، إلا أننا نحصل على نتائج جيدة جدًا! الق نظرة على المعيار المرجعي التالي (llama-2-7b على مقياس الحيرة). يمكنك العثور على المزيد من المعايير المرجعية [هنا](https://github.com/huggingface/quanto/tree/main/bench/generation)

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/quantization/NousResearch-Llama-2-7b-hf_Perplexity.png" alt="llama-2-7b-quanto-perplexity" />
  </div>
</div>

تتمتع المكتبة بمرونة كافية لتكون متوافقة مع معظم خوارزميات تحسين PTQ. وتتمثل الخطة المستقبلية في دمج الخوارزميات الأكثر شعبية بأكثر الطرق سلاسة (AWQ، Smoothquant).