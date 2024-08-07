# LLaMA

## نظرة عامة
اقترح نموذج LLaMA في [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) بواسطة Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. وهي مجموعة من نماذج اللغة الأساسية التي تتراوح من 7B إلى 65B من المعلمات.

الملخص من الورقة هو ما يلي:

*نحن نقدم LLaMA، وهي مجموعة من نماذج اللغة الأساسية التي تتراوح من 7B إلى 65B من المعلمات. نقوم بتدريب نماذجنا على تريليونات من الرموز، ونظهر أنه من الممكن تدريب نماذج متقدمة باستخدام مجموعات البيانات المتاحة للجمهور حصريًا، دون اللجوء إلى مجموعات البيانات المملوكة وغير المتاحة. وعلى وجه الخصوص، يتفوق LLaMA-13B على GPT-3 (175B) في معظم المعايير، وينافس LLaMA-65B أفضل النماذج، Chinchilla-70B و PaLM-540B. نقوم بإطلاق جميع نماذجنا لمجتمع البحث.*

تمت المساهمة بهذا النموذج من قبل [zphang](https://huggingface.co/zphang) بمساهمات من [BlackSamorez](https://huggingface.co/BlackSamorez). يعتمد كود التنفيذ في Hugging Face على GPT-NeoX [here](https://github.com/EleutherAI/gpt-neox). يمكن العثور على الكود الأصلي للمؤلفين [here](https://github.com/facebookresearch/llama).

## نصائح الاستخدام

- يمكن الحصول على أوزان نماذج LLaMA من خلال ملء [هذا النموذج](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform?usp=send_form)

- بعد تنزيل الأوزان، سيتعين تحويلها إلى تنسيق Hugging Face Transformers باستخدام [Script التحويل](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py). يمكن استدعاء النص البرمجي باستخدام الأمر التالي (كمثال):

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
--input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

- بعد التحويل، يمكن تحميل النموذج ومحول الرموز باستخدام ما يلي:

```python
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("/output/path")
model = LlamaForCausalLM.from_pretrained("/output/path")
```

ملاحظة: يتطلب تنفيذ النص البرمجي مساحة كافية من ذاكرة الوصول العشوائي CPU لاستضافة النموذج بالكامل في دقة float16 (حتى إذا كانت الإصدارات الأكبر تأتي في عدة نقاط مرجعية، فإن كل منها يحتوي على جزء من كل وزن للنموذج، لذلك نحتاج إلى تحميلها جميعًا في ذاكرة الوصول العشوائي). بالنسبة للنموذج 65B، نحتاج إلى 130 جيجابايت من ذاكرة الوصول العشوائي.

- محول رموز LLaMA هو نموذج BPE يعتمد على [sentencepiece](https://github.com/google/sentencepiece). إحدى ميزات sentencepiece هي أنه عند فك تشفير تسلسل، إذا كان الرمز الأول هو بداية الكلمة (مثل "Banana")، فإن المحلل اللغوي لا يسبق المسافة البادئة إلى السلسلة.

تمت المساهمة بهذا النموذج من قبل [zphang](https://huggingface.co/zphang) بمساهمات من [BlackSamorez](https://huggingface.co/BlackSamorez). يعتمد كود التنفيذ في Hugging Face على GPT-NeoX [here](https://github.com/EleutherAI/gpt-neox). يمكن العثور على الكود الأصلي للمؤلفين [here](https://github.com/facebookresearch/llama). تم تقديم إصدار Flax من التنفيذ من قبل [afmck](https://huggingface.co/afmck) مع الكود في التنفيذ بناءً على Flax GPT-Neo من Hugging Face.

بناءً على نموذج LLaMA الأصلي، أصدرت Meta AI بعض الأعمال اللاحقة:

- **Llama2**: Llama2 هو إصدار محسن من Llama مع بعض التعديلات المعمارية (Grouped Query Attention)، وهو مُدرب مسبقًا على 2 تريليون رمز. راجع وثائق Llama2 التي يمكن العثور عليها [here](llama2).

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (مشار إليها بـ 🌎) لمساعدتك في البدء باستخدام LLaMA. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فيرجى فتح طلب سحب وسنراجعه! يجب أن يُظهر المورد المثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

<PipelineTag pipeline="text-classification"/>

- A [notebook](https://colab.research.google.com/github/bigscience-workshop/petals/blob/main/examples/prompt-tuning-sst2.ipynb#scrollTo=f04ba4d2) on how to use prompt tuning to adapt the LLaMA model for text classification task. 🌎

<PipelineTag pipeline="question-answering"/>

- [StackLLaMA: A hands-on guide to train LLaMA with RLHF](https://huggingface.co/blog/stackllama#stackllama-a-hands-on-guide-to-train-llama-with-rlhf), a blog post about how to train LLaMA to answer questions on [Stack Exchange](https://stackexchange.com/) with RLHF.

⚗️ Optimization

- A [notebook](https://colab.research.google.com/drive/1SQUXq1AMZPSLD4mk3A3swUIc6Y2dclme?usp=sharing) on how to fine-tune LLaMA model using xturing library on GPU which has limited memory. 🌎

⚡️ Inference

- A [notebook](https://colab.research.google.com/github/DominguesM/alpaca-lora-ptbr-7b/blob/main/notebooks/02%20-%20Evaluate.ipynb) on how to run the LLaMA Model using PeftModel from the 🤗 PEFT library. 🌎

- A [notebook](https://colab.research.google.com/drive/1l2GiSSPbajVyp2Nk3CFT4t3uH6-5TiBe?usp=sharing) on how to load a PEFT adapter LLaMA model with LangChain. 🌎

🚀 Deploy

- A [notebook](https://colab.research.google.com/github/lxe/simple-llama-finetuner/blob/master/Simple_LLaMA_FineTuner.ipynb#scrollTo=3PM_DilAZD8T) on how to fine-tune LLaMA model using LoRA method via the 🤗 PEFT library with intuitive UI. 🌎

- A [notebook](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart-foundation-models/text-generation-open-llama.ipynb) on how to deploy Open-LLaMA model for text generation on Amazon SageMaker. 🌎

## LlamaConfig

[[autodoc]] LlamaConfig

## LlamaTokenizer

[[autodoc]] LlamaTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## LlamaTokenizerFast

[[autodoc]] LlamaTokenizerFast

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- update_post_processor
- save_vocabulary

## LlamaModel

[[autodoc]] LlamaModel

- forward

## LlamaForCausalLM

[[autodoc]] LlamaForCausalLM

- forward

## LlamaForSequenceClassification

[[autodoc]] LlamaForSequenceClassification

- forward

## LlamaForQuestionAnswering

[[autodoc]] LlamaForQuestionAnswering

- forward

## LlamaForTokenClassification

[[autodoc]] LlamaForTokenClassification

- forward

## FlaxLlamaModel

[[autodoc]] FlaxLlamaModel

- __call__

## FlaxLlamaForCausalLM

[[autodoc]] FlaxLlamaForCausalLM

- __call__