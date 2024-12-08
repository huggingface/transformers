# تكامل XLA لنماذج TensorFlow

[[open-in-colab]]

الجبر الخطي المعجل، الذي يُطلق عليه XLA، هو مترجم لتسريع وقت تشغيل نماذج TensorFlow. من [الوثائق الرسمية](https://www.tensorflow.org/xla):

> XLA (Accelerated Linear Algebra) هو مترجم خاص بالمجال للجبر الخطي يمكنه تسريع نماذج TensorFlow دون إجراء أي تغييرات على شفرة المصدر.

إن استخدام XLA في TensorFlow أمر بسيط - فهو يأتي مضمنًا داخل مكتبة `tensorflow`، ويمكن تشغيله باستخدام وسيط `jit_compile` في أي دالة لإنشاء الرسم البياني مثل [`tf.function`](https://www.tensorflow.org/guide/intro_to_graphs). عند استخدام أساليب Keras مثل `fit()` و`predict()`، يمكنك تمكين XLA ببساطة عن طريق تمرير وسيط `jit_compile` إلى `model.compile()`. ومع ذلك، لا تقتصر XLA على هذه الأساليب - يمكن أيضًا استخدامها لتسريع أي دالة `tf.function` عشوائية.

تمت إعادة كتابة العديد من أساليب TensorFlow في 🤗 Transformers لتكون متوافقة مع XLA، بما في ذلك توليد النصوص لنماذج مثل [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2) و[T5](https://huggingface.co/docs/transformers/model_doc/t5) و[OPT](https://huggingface.co/docs/transformers/model_doc/opt)، بالإضافة إلى معالجة الكلام لنماذج مثل [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper).

في حين أن مقدار التسريع الدقيق يعتمد إلى حد كبير على النموذج، فقد لاحظنا تسريعًا يبلغ حوالي 100x لنماذج توليد النصوص TensorFlow داخل 🤗 Transformers. سيوضح هذا المستند كيفية استخدام XLA لهذه النماذج للحصول على أقصى قدر من الأداء. كما سنقدم روابط إلى موارد إضافية إذا كنت مهتمًا بمعرفة المزيد حول المعايير وتفكيرنا في تصميم التكامل مع XLA.

## تشغيل وظائف TF باستخدام XLA

لنأخذ في الاعتبار النموذج التالي في TensorFlow:

```py
import tensorflow as tf

model = tf.keras.Sequential(
    [tf.keras.layers.Dense(10, input_shape=(10,), activation="relu"), tf.keras.layers.Dense(5, activation="softmax")]
)
```

يقبل النموذج أعلاه إدخالات ذات بعد `(10, )`. يمكننا استخدام النموذج لتشغيل عملية تمرير أمامي كما يلي:

```py
# Generate random inputs for the model.
batch_size = 16
input_vector_dim = 10
random_inputs = tf.random.normal((batch_size, input_vector_dim))

# Run a forward pass.
_ = model(random_inputs)
```

لتشغيل عملية التمرير الأمامي باستخدام دالة مجمعة بواسطة XLA، سنحتاج إلى القيام بما يلي:

```py
xla_fn = tf.function(model, jit_compile=True)
_ = xla_fn(random_inputs)
```

تُستخدم دالة `call()` الافتراضية للنموذج لتجميع رسم بياني XLA. ولكن إذا كان هناك أي دالة أخرى للنموذج تريد تجميعها في XLA، فيمكنك أيضًا القيام بذلك:

```py
my_xla_fn = tf.function(model.my_xla_fn, jit_compile=True)
```

## تشغيل نموذج توليد نص TF باستخدام XLA من 🤗 Transformers

لتمكين التوليد المعجل بواسطة XLA داخل 🤗 Transformers، يجب أن يكون لديك إصدار حديث من `transformers` مثبتًا. يمكنك تثبيته عن طريق تشغيل:

```bash
pip install transformers --upgrade
```

ثم يمكنك تشغيل الشفرة التالية:

```py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM
```bash
pip install transformers --upgrade
```

ثم يمكنك تشغيل الشفرة التالية:

```py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

# سيحدث خطأ إذا لم يتم تثبيت الإصدار الأدنى من Transformers.
from transformers.utils import check_min_version

check_min_version("4.21.0")


tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")
input_string = ["TensorFlow is"]

# سطر واحد لإنشاء دالة توليد XLA
xla_generate = tf.function(model.generate, jit_compile=True)

tokenized_input = tokenizer(input_string, return_tensors="tf")
generated_tokens = xla_generate(**tokenized_input, num_beams=2)

decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(f"Generated -- {decoded_text}")
# تم التوليد -- TensorFlow هو إطار تطبيق مفتوح المصدر ومفتوح المصدر ومفتوح المصدر
```

كما تلاحظ، فإن تمكين XLA على `generate()` هو مجرد سطر واحد من الشفرة. تظل بقية الشفرة دون تغيير. ومع ذلك، هناك بعض الأشياء التي يجب مراعاتها في مقتطف الشفرة أعلاه والتي تخص XLA تحديدًا. يجب أن تكون على دراية بها لتحقيق التسريعات التي يمكن أن توفرها XLA. نناقش هذه الأمور في القسم التالي.

## الأشياء التي يجب مراعاتها

عندما تقوم بتنفيذ دالة ممكّنة لـ XLA (مثل `xla_generate()` أعلاه) للمرة الأولى، فسوف تحاول داخليًا استنتاج رسم الحساب، وهو أمر يستغرق وقتًا طويلاً. تُعرف هذه العملية باسم ["التتبع"](https://www.tensorflow.org/guide/intro_to_graphs#when_is_a_function_tracing).

قد تلاحظ أن وقت التوليد ليس سريعًا. لن تحتاج عمليات الاستدعاء المتتالية لـ `xla_generate()` (أو أي دالة أخرى ممكّنة لـ XLA) إلى استنتاج رسم الحساب، بشرط أن تتبع الإدخالات إلى الدالة نفس الشكل الذي تم بناء رسم الحساب به في البداية. في حين أن هذا ليس مشكلة بالنسبة للطرائق ذات أشكال الإدخال الثابتة (مثل الصور)، يجب الانتباه إذا كنت تعمل مع طرائق ذات شكل إدخال متغير (مثل النص).

لضمان عمل `xla_generate()` دائمًا مع أشكال الإدخال نفسها، يمكنك تحديد وسيطات `padding` عند استدعاء tokenizer.

```py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")
input_string = ["TensorFlow is"]

xla_generate = tf.function(model.generate, jit_compile=True)

# هنا، نقوم باستدعاء tokenizer مع خيارات الحشو.
tokenized_input = tokenizer(input_string, pad_to_multiple_of=8, padding=True, return_tensors="tf")

generated_tokens = xla_generate(**tokenized_input, num_beams=2)
decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(f"Generated -- {decoded_text}")
```

بهذه الطريقة، يمكنك التأكد من أن الإدخالات إلى `xla_generate()` ستتلقى دائمًا إدخالات ذات الشكل الذي تم تتبعه به، مما يؤدي إلى تسريع وقت التوليد. يمكنك التحقق من ذلك باستخدام الشفرة التالية:

```py
import time
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")

xla_generate = tf.function(model.generate, jit_compile=True)

for input_string in ["TensorFlow is", "TensorFlow is a", "TFLite is a"]:
    tokenized_input = tokenizer(input_string, pad_to_multiple_of=8, padding=True, return_tensors="tf")
    start = time.time_ns()
    generated_tokens = xla_generate(**tokenized_input, num_beams=2)
    end = time.time_ns()
    print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
```

على وحدة معالجة الرسوميات (GPU) Tesla T4، يمكنك توقع المخرجات كما يلي:

```bash
Execution time -- 30819.6 ms
على وحدة معالجة الرسوميات (GPU) Tesla T4، يمكنك توقع المخرجات كما يلي:

```bash
Execution time -- 30819.6 ms

Execution time -- 79.0 ms

Execution time -- 78.9 ms
```
تستغرق المكالمة الأولى إلى `xla_generate()` وقتًا طويلاً بسبب التتبع، ولكن المكالمات المتتالية أسرع بكثير. ضع في اعتبارك أن أي تغيير في خيارات التوليد في أي نقطة سيؤدي إلى إعادة التتبع، مما يؤدي إلى بطء وقت التوليد.

لم نغطِ جميع خيارات توليد النصوص التي يوفرها 🤗 Transformers في هذه الوثيقة. نشجعك على قراءة الوثائق للحصول على حالات استخدام متقدمة.

## موارد إضافية

نترككم هنا ببعض الموارد الإضافية إذا كنت ترغب في التعمق في XLA في 🤗 Transformers بشكل عام.

* [يوفر دفتر الملاحظات هذا من Colab](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/91_tf_xla_generate.ipynb) عرضًا توضيحيًا تفاعليًا إذا كنت ترغب في العبث بنماذج التوليد المتوافقة مع XLA (مثل [T5](https://huggingface.co/docs/transformers/model_doc/t5)) والترميز فك الترميز (مثل [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)).
* [تقدم هذه التدوينة](https://huggingface.co/blog/tf-xla-generate) نظرة عامة على معايير المقارنة للنماذج المتوافقة مع XLA بالإضافة إلى مقدمة سهلة الاستخدام لـ XLA في TensorFlow.
* [تناقش هذه التدوينة](https://blog.tensorflow.org/2022/11/how-hugging-face-improved-text-generation-performance-with-xla.html) تفكيرنا في التصميم وراء إضافة دعم XLA إلى نماذج TensorFlow في 🤗 Transformers.
* المشاركات الموصى بها لمعرفة المزيد حول XLA ورسومات TensorFlow بشكل عام:
    * [XLA: مترجم محسّن لتعلم الآلة](https://www.tensorflow.org/xla)
    * [مقدمة إلى الرسوم البيانية وtf.function](https://www.tensorflow.org/guide/intro_to_graphs)
    * [أداء أفضل مع tf.function](https://www.tensorflow.org/guide/function)