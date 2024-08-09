# RWKV

## نظرة عامة
تم اقتراح نموذج RWKV في [هذا المستودع](https://github.com/BlinkDL/RWKV-LM)
يقترح تعديلاً طفيفًا على اهتمام المحول التقليدي لجعله خطيًا. بهذه الطريقة، يمكن استخدام النموذج كشبكة متكررة: تمرير الإدخالات لختم الوقت 0 وختم الوقت 1 معًا هو نفسه كتمرير الإدخالات في ختم الوقت 0، ثم الإدخالات في ختم الوقت 1 جنبًا إلى جنب مع حالة ختم الوقت 0 (راجع المثال أدناه).
يمكن أن يكون هذا أكثر كفاءة من المحول العادي ويمكنه التعامل مع الجملة بأي طول (حتى إذا استخدم النموذج طول سياق ثابتًا للتدريب).
تمت المساهمة بهذا النموذج من قبل [sgugger](https://huggingface.co/sgugger).
يمكن العثور على الكود الأصلي [هنا](https://github.com/BlinkDL/RWKV-LM).

## مثال الاستخدام

```py
import torch
from transformers import AutoTokenizer, RwkvConfig, RwkvModel

model = RwkvModel.from_pretrained("sgugger/rwkv-430M-pile")
tokenizer = AutoTokenizer.from_pretrained("sgugger/rwkv-430M-pile")

inputs = tokenizer("This is an example.", return_tensors="pt")
# قم بإدخال كل شيء إلى النموذج
outputs = model(inputs["input_ids"])
output_whole = outputs.last_hidden_state

outputs = model(inputs["input_ids"][:, :2])
output_one = outputs.last_hidden_state

# باستخدام الحالة المحسوبة على الإدخالات الأولى، سنحصل على نفس الإخراج
outputs = model(inputs["input_ids"][:, 2:], state=outputs.state)
output_two = outputs.last_hidden_state

torch.allclose(torch.cat([output_one, output_two], dim=1), output_whole, atol=1e-5)
```

إذا كنت تريد التأكد من توقف النموذج عن التوليد عند اكتشاف " \n\n"، نوصي باستخدام معايير التوقف التالية:

```python
from transformers import StoppingCriteria

class RwkvStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [187,187], eos_token_id = 537):
        self.eos_sequence = eos_sequence
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_2_ids = input_ids[:,-2:].tolist()
        return self.eos_sequence in last_2_ids

output = model.generate(inputs["input_ids"], max_new_tokens=64, stopping_criteria = [RwkvStoppingCriteria()])
```

## RwkvConfig

[[autodoc]] RwkvConfig

## RwkvModel

[[autodoc]] RwkvModel

- forward

## RwkvLMHeadModel

[[autodoc]] RwkvForCausalLM

- forward

## اهتمام RWKV والصيغ المتكررة

في محول ذاتي الانحدار تقليدي، يتم كتابة الاهتمام على النحو التالي:

$$O = \hbox{softmax}(QK^{T} / \sqrt{d}) V$$

مع Q و K و V مصفوفات من الشكل 'seq_len x hidden_size' تسمى query و key و value (في الواقع، هي مصفوفات أكبر مع بعد دفعة وبعد رأس اهتمام، ولكننا مهتمون فقط بالأخيرين، حيث يتم أخذ حاصل الضرب للمصفوفة، لذلك من أجل البساطة، نعتبر هذين فقط). ثم يكون للمنتج QK^T شكل 'seq_len x seq_len' ويمكننا أخذ حاصل الضرب للمصفوفة مع V للحصول على الإخراج O بنفس الشكل مثل الآخرين.

يؤدي استبدال softmax بقيمته إلى ما يلي:

$$O_{i} = \frac{\sum_{j=1}^{i} e^{Q_{i} K_{j}^{T} / \sqrt{d}} V_{j}}{\sum_{j=1}^{i} e^{Q_{i} K_{j}^{T} / \sqrt{d}}}$$

لاحظ أن الإدخالات في QK^T المقابلة لـ j > i يتم قناعتها (يتوقف الجمع عند j) لأن الاهتمام غير مسموح له بالنظر إلى الرموز المستقبلية (فقط الماضي).

بالمقارنة، يتم إعطاء اهتمام RWKV بواسطة:

$$O_{i} = \sigma(R_{i}) \frac{\sum_{j=1}^{i} e^{W_{i-j} + K_{j}} V_{j}}{\sum_{j=1}^{i} e^{W_{i-j} + K_{j}}}$$

حيث R هي مصفوفة جديدة تسمى receptance من قبل المؤلف، K و V لا تزال المفتاح والقيمة (σ هنا هي دالة sigmoid). W هو متجه جديد يمثل موضع الرمز ويتم إعطاؤه بواسطة:

$$W_{0} = u \hbox{  and  } W_{k} = (k-1)w \hbox{ for } k \geq 1$$

مع u و w معلمات قابلة للتعلم تسمى في الكود 'time_first' و 'time_decay' على التوالي. يمكن التعبير عن البسط والمقام كليهما بشكل متكرر. من خلال تسميتهما N_i و D_i لدينا:

$$N_{i} = e^{u + K_{i}} V_{i} + \hat{N}_{i} \hbox{  where  } \hat{N}_{i} = e^{K_{i-1}} V_{i-1} + e^{w + K_{i-2}} V_{i-2} \cdots + e^{(i-2)w + K_{1}} V_{1}$$

لذا فإن N_i (تسمى 'numerator_state' في الكود) تفي بما يلي:

$$\hat{N}_{0} = 0 \hbox{  and  } \hat{N}_{j+1} = e^{K_{j}} V_{j} + e^{w} \hat{N}_{j}$$

و:

$$D_{i} = e^{u + K_{i}} + \hat{D}_{i} \hbox{  where  } \hat{D}_{i} = e^{K_{i-1}} + e^{w + K_{i-2}} \cdots + e^{(i-2)w + K_{1}}$$

لذا فإن D_i (تسمى 'denominator_state' في الكود) تفي بما يلي:

$$\hat{D}_{0} = 0 \hbox{  and  } \hat{D}_{j+1} = e^{K_{j}} + e^{w} \hat{D}_{j}$$

الصيغ المتكررة الفعلية المستخدمة أكثر تعقيدًا بقليل، حيث لا نريد، من أجل الاستقرار العددي، حساب الأسس الأسية للأرقام الكبيرة. عادةً لا يتم حساب softmax كما هو، ولكن يتم تقسيم الأس الأساسي لأكبر مصطلح على البسط والمقام:

$$\frac{e^{x_{i}}}{\sum_{j=1}^{n} e^{x_{j}}} = \frac{e^{x_{i} - M}}{\sum_{j=1}^{n} e^{x_{j} - M}}$$

مع M الحد الأقصى لجميع x_j. لذا، بالإضافة إلى حفظ حالة البسط (N_hat) وحالة المقام (D_hat)، نقوم أيضًا بتتبع الحد الأقصى لجميع المصطلحات المواجهة في الأسس الأسية. لذا نستخدم في الواقع:

$$\tilde{N}_{i} = e^{-M_{i}} \hat{N}_{i} \hbox{  and  } \tilde{D}_{i} = e^{-M_{i}} \hat{D}_{i}$$

محددة بواسطة الصيغ المتكررة التالية:

$$\tilde{N}_{0} = 0 \hbox{  and  } \tilde{N}_{j+1} = e^{K_{j} - q} V_{j} + e^{w + M_{j} - q} \tilde{N}_{j} \hbox{  where  } q = \max(K_{j}, w + M_{j})$$

و:

$$\tilde{D}_{0} = 0 \hbox{  and  } \tilde{D}_{j+1} = e^{K_{j} - q} + e^{w + M_{j} - q} \tilde{D}_{j} \hbox{  where  } q = \max(K_{j}, w + M_{j})$$

و M_ {j + 1} = q. مع هذه، يمكننا بعد ذلك حساب:

$$N_{i} = e^{u + K_{i} - q} V_{i} + e^{M_{i}} \tilde{N}_{i} \hbox{  where  } q = \max(u + K_{i}, M_{i})$$

و:

$$D_{i} = e^{u + K_{i} - q} + e^{M_{i}} \tilde{D}_{i} \hbox{  where  } q = \max(u + K_{i}, M_{i})$$

الذي يعطينا أخيرًا:

$$O_{i} = \sigma(R_{i}) \frac{N_{i}}{D_{i}}$$