# كيفية تعديل أي نموذج من نماذج Transformers

توفر مكتبة [🤗 Transformers](https://github.com/huggingface/transformers) مجموعة من النماذج المسبقة التدريب والأدوات لمعالجة اللغات الطبيعية، والرؤية، وما إلى ذلك. على الرغم من أن هذه النماذج تغطي مجموعة واسعة من التطبيقات، فقد تواجه حالات استخدام لا تدعمها المكتبة بشكل افتراضي. يُمكن للتخصيص أن يفتح إمكانيات جديدة، مثل إضافة طبقات جديدة، أو تعديل البنية المعمارية، أو تحسين آليات الانتباه. سيُوضح لك هذا الدليل كيفية تعديل نماذج Transformers الموجودة لتلبية احتياجاتك المحددة. الشيء الرائع هو أنك لست بحاجة إلى الخروج من إطار عمل Transformers لإجراء هذه التغييرات. ي يمكنك تعديل النماذج مباشرةً في Transformers والاستفادة من الميزات مثل [واجهة برمجة التطبيقات Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer)، و [PreTrainedModel](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)، والضبط الدقيق الفعال باستخدام أدوات مثل [PEFT](https://huggingface.co/docs/peft/index).

سنرشدك في هذا الدليل  لكيفية تخصيص نماذج Transformers الموجودة لتلبية متطلباتك، دون فقدان مزايا الإطار. ستتعلم كيفية:

- تعديل بنية نموذج ما من خلال تغيير آلية الانتباه الخاصة به.
- تطبيق تقنيات مثل Low-Rank Adaptation (LoRA) على مكونات نموذج محددة.

نحن نشجعك على المساهمة باختراقاتك الخاصة ومشاركتها هنا مع المجتمع1

## مثال: تعديل آلية الانتباه في نموذج Segment Anything (SAM)

نموذج **Segment Anything (SAM)** هو نموذج رائد في مجال تجزئة الصور. في تنفيذه الافتراضي، يستخدم SAM إسقاطًا مجمعًا للاستعلام والمفتاح والقيمة (`qkv`) في آلية الانتباه الخاصة به. ومع ذلك، قد ترغب في ضبط مكونات محددة فقط من آلية الانتباه، مثل إسقاطات الاستعلام (`q`) والقيمة (`v`)، لتقليل عدد المعلمات القابلة للتدريب والموارد الحسابية المطلوبة.

### الدافع

من خلال تقسيم الإسقاط المجمع `qkv` إلى إسقاطات منفصلة `q` و `k` و `v`، يمكنك تطبيق تقنيات مثل **LoRA** (Low-Rank Adaptation) على إسقاطي `q` و `v` فقط. يسمح لك هذا بما يلي:

- ضبط عدد أقل من المعلمات، مما يقلل من العبء الحسابي.
- تحقيق أداء أفضل من خلال التركيز على مكونات محددة.
- تجربة استراتيجيات تعديل مختلفة في آلية الانتباه.

### التنفيذ

#### **الخطوة 1: إنشاء فئة اهتمام مخصصة**

بعد ذلك، قم بإنشاء فئة فرعية من فئة `SamVisionAttention` الأصلية وعدلها لتضم إسقاطات `q` و `k` و `v` منفصلة.

```python
import torch
import torch.nn as nn
from transformers.models.sam.modeling_sam import SamVisionAttention

class SamVisionAttentionSplit(SamVisionAttention, nn.Module):
    def __init__(self, config, window_size):
        super().__init__(config, window_size)
        del self.qkv
        # إسقاطات منفصلة q و k و v
        self.q = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.k = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.v = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self._register_load_state_dict_pre_hook(self.split_q_k_v_load_hook)

    def split_q_k_v_load_hook(self, state_dict, prefix, *args):
        keys_to_delete = []
        for key in list(state_dict.keys()):
            if "qkv." in key:
                # تقسيم q و k و v من الإسقاط المجمع
                q, k, v = state_dict[key].chunk(3, dim=0)
                # استبدال الإسقاطات الفردية q و k و v
                state_dict[key.replace("qkv.", "q.")] = q
                state_dict[key.replace("qkv.", "k.")] = k
                state_dict[key.replace("qkv.", "v.")] = v
                # وضع علامة على مفتاح qkv القديم للحذف
                keys_to_delete.append(key)

        # حذف مفاتيح qkv القديمة
        for key in keys_to_delete:
            del state_dict[key]

    def forward(self, hidden_states: torch.Tensor, output_attentions=False) -> torch.Tensor:
        batch_size, height, width, _ = hidden_states.shape
        qkv_shapes = (batch_size *  self.num_attention_heads,  height * width, -1)
        query = self.q(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)
        key = self.k(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)
        value = self.v(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)

        attn_weights = (query * self.scale) @ key.transpose(-2, -1)

        if self.use_rel_pos:
            attn_weights = self.add_decomposed_rel_pos(
                attn_weights, query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = (attn_probs @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)
        attn_output = self.proj(attn_output)

        if output_attentions:
            outputs = (attn_output, attn_weights)
        else:
            outputs = (attn_output, None)
        return outputs
```

**الشرح:**

- **الإسقاطات المنفصلة:** يتم إزالة الإسقاط المُجمع `qkv`، وإنشاء إسقاطات خطية منفصلة `q` و `k` و `v`.
- **دالة استدعاء  تحميل الأوزان:** تقوم طريقة `_split_qkv_load_hook` بتقسيم أوزان `qkv` المسبقة التدريب إلى أوزان `q` و `k` و `v` منفصلة عند تحميل النموذج. يضمن هذا التوافق مع أي نموذج مسبق التدريب.
- **التنفيذ الأمامي:** يتم حساب الاستعلامات والمفاتيح والقيم بشكل منفصل، وتستمر آلية الانتباه كالمعتاد.

#### **الخطوة 2: استبدال فئة الانتباه الأصلية**

استبدل فئة `SamVisionAttention` الأصلية بفئتك المخصصة بحيث يستخدم النموذج آلية الانتباه المعدلة.

```python
from transformers import SamModel
from transformers.models.sam import modeling_sam

# استبدال فئة الاهتمام في وحدة نمطية modeling_sam
modeling_sam.SamVisionAttention = SamVisionAttentionSplit

# تحميل نموذج SAM المسبق التدريب
model = SamModel.from_pretrained("facebook/sam-vit-base")
```

**الشرح:**

- **استبدال الفئة:** من خلال تعيين فئتك المخصصة إلى `modeling_sam.SamVisionAttention`، فإن أي حالات من فئة `SamVisionAttention` في النموذج ستستخدم النسخة المعدلة. وبالتالي، عند استدعاء `SamModel`، سيتم استخدام `SamVisionAttentionSplit` المحددة حديثًا.
- **تحميل النموذج:** يتم تحميل النموذج باستخدام `from_pretrained`، ويتم دمج آلية الانتباه المخصصة.

#### **الخطوة 3: تطبيق LoRA على إسقاطات محددة**

مع وجود إسقاطات `q` و `k` و `v` منفصلة، يمكنك الآن تطبيق LoRA على مكونات محددة، مثل إسقاطات `q` و `v`.

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],  # تطبيق LoRA على إسقاطات q و v
    lora_dropout=0.1,
    task_type="mask-generation"
)

# تطبيق LoRA على النموذج
model = get_peft_model(model, config)
```

**الشرح:**

- **تكوين LoRA:** تحدد `LoraConfig` المرتبة `r`، وعامل القياس `lora_alpha`، والوحدات المستهدفة (`"q"` و `"v"`)، ومعدل التخلي، ونوع المهمة.
- **تطبيق LoRA:** تقوم دالة `get_peft_model` بتطبيق LoRA على الوحدات المحددة في النموذج.
- **تقليل المعلمات:** من خلال التركيز على `q` و `v`، فإنك تقلل عدد المعلمات القابلة للتدريب، مما يؤدي إلى تسريع التدريب وتقليل استخدام الذاكرة.

#### **الخطوة 4: التحقق من عدد المعلمات القابلة للتدريب**

من السهل التحقق من عدد المعلمات القابلة للتدريب ومعرفة تأثير تعديلك.

```python
model.print_trainable_parameters()
```

**الناتج المتوقع:**

```
عدد المعلمات القابلة للتدريب: 608,256 || جميع المعلمات: 94,343,728 || نسبة المعلمات القابلة للتدريب: 0.6447
عدد المعلمات القابلة للتدريب: 912,384 || جميع المعلمات: 94,647,856 || نسبة المعلمات القابلة للتدريب: 0.9640 # مع k
```

## المساهمة بابداعاتك الخاصة

يمكن لتعديل النماذج المسبقة التدريب أن يفتح آفاقًا جديدة للبحث والتطبيق. من خلال فهم وتعديل الآليات الداخلية للنماذج مثل SAM، يمكنك تخصيصها لتلبية احتياجاتك المحددة، وتحسين الأداء، وتجربة أفكار جديدة.

إذا قمت بتطوير تعديﻻتك الخاصة لنماذج Transformers وترغب في مشاركتها، ففكر في المساهمة في هذه الوثيقة.

- **إنشاء طلب سحب (Pull Request):** شارك تغييراتك وتحسيناتك في التعليمات البرمجية مباشرة في المستودع.
- **كتابة التوثيق:** قدم تفسيرات وأمثلة واضحة لتعديلاتك.
- **التفاعل مع المجتمع:** ناقش أفكارك واحصل على تعليقات من المطورين والباحثين الآخرين من خلال فتح مشكلة.
