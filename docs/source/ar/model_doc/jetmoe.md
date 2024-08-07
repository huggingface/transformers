# JetMoe

## نظرة عامة

**JetMoe-8B** هو نموذج لغة Mixture-of-Experts (MoE) بسعة 8B طوّره [Yikang Shen](https://scholar.google.com.hk/citations?user=qff5rRYAAAAJ) و [MyShell](https://myshell.ai/). يهدف مشروع JetMoe إلى توفير أداء LLaMA2-level وكفاءة نموذج اللغة بميزانية محدودة.

لتحقيق هذا الهدف، يستخدم JetMoe بنية تنشيط نادرة الإستخدام مستوحاة من [ModuleFormer](https://arxiv.org/abs/2306.04640). يتكون كل كتلة JetMoe من طبقتين MoE: مزيج من رؤوس الانتباه وخبراء مزيج MLP.

بالنسبة لرموز الإدخال، فإنه ينشط مجموعة فرعية من خبرائه لمعالجتها. يسمح مخطط التنشيط النادر هذا لـ JetMoe بتحقيق معدل نقل تدريب أفضل بكثير من النماذج الكثيفة المماثلة في الحجم.

يبلغ معدل نقل التدريب لـ JetMoe-8B حوالي 100 مليار رمز في اليوم على مجموعة من 96 وحدة معالجة رسومات H100 باستخدام استراتيجية تسلسلية ثلاثية الاتجاهات.

تمت المساهمة بهذا النموذج من قبل [Yikang Shen](https://huggingface.co/YikangS).

## JetMoeConfig

[[autodoc]] JetMoeConfig

## JetMoeModel

[[autodoc]] JetMoeModel

- forward

## JetMoeForCausalLM

[[autodoc]] JetMoeForCausalLM

- forward

## JetMoeForSequenceClassification

[[autodoc]] JetMoeForSequenceClassification

- forward