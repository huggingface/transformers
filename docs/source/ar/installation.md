# التثبيت (Installation)

قم بتثبيت مكتبة 🤗 Transformers المناسبة لمكتبة التعلم العميق التي تستخدمها، وقم بإعداد ذاكرة التخزين المؤقت الخاصة بك، وقم بإعداد 🤗 Transformers للعمل دون اتصال بالإنترنت (اختياري).

تم اختبار 🤗 Transformers على Python 3.10 والإصدارات الأحدث، وPyTorch 2.4 والإصدارات الأحدث. اتبع تعليمات التثبيت أدناه لمكتبة التعلم العميق التي تستخدمها:

* تعليمات تثبيت [PyTorch](https://pytorch.org/get-started/locally/).

## التثبيت باستخدام pip

يجب عليك تثبيت 🤗 Transformers داخل [بيئة افتراضية](https://docs.python.org/3/library/venv.html). إذا لم تكن غير ملم ببيئات Python الافتراضية، فراجع هذا [الدليل](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). البيئة الافتراضية تسهل إدارة المشاريع المختلف، وتجنب مشكلات التوافق بين المكتبات المطلوبة (اعتماديات المشروع).

ابدأ بإنشاء بيئة افتراضية في دليل مشروعك:

```bash
python -m venv .env
```

قم بتفعيل البيئة الافتراضية. على Linux وMacOs:

```bash
source .env/bin/activate
```

قم بتفعيل البيئة الافتراضية على Windows:

```bash
.env/Scripts/activate
```

الآن أنت مستعد لتثبيت 🤗 Transformers باستخدام الأمر التالي:

```bash
pip install transformers
```

للحصول على الدعم الخاص بـ CPU فقط، يمكنك تثبيت 🤗 Transformers ومكتبة التعلم العميق في خطوة واحدة. على سبيل المثال، قم بتثبيت 🤗 Transformers وPyTorch باستخدام:

```bash
pip install 'transformers[torch]'
```

أخيرًا، تحقق مما إذا كان 🤗 Transformers قد تم تثبيته بشكل صحيح عن طريق تشغيل الأمر التالي. سيقوم بتنزيل نموذج مدرب مسبقًا:

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

ثم قم بطباعة التسمية والنتيجة:

```bash
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

## التثبيت من المصدر

قم بتثبيت 🤗 Transformers من المصدر باستخدام الأمر التالي:

```bash
pip install git+https://github.com/huggingface/transformers
```

يقوم هذا الأمر بتثبيت  أحدث إصدار تجريبي `main`  بدلاً من الإصدار المستقر `stable`. يعد إصدار `main` مفيدًا للمواكبة مع أحدث التطورات. على سبيل المثال، إذا تم إصلاح خطأ منذ الإصدار الرسمي الأخير ولكن لم يتم طرح إصدار جديد بعد. ومع ذلك، فإن هذا يعني أن إصدار التجريبي `main` قد لا يكون مستقرًا دائمًا. نسعى جاهدين للحفاظ على تشغيل إصدار `main`، ويتم حل معظم المشكلات عادةً في غضون بضع ساعات أو يوم. إذا واجهتك مشكلة، يرجى فتح [تقرير عن خلل](https://github.com/huggingface/transformers/issues) حتى نتمكن من إصلاحها في أقرب وقت ممكن!

تحقق مما إذا كان 🤗 Transformers قد تم تثبيته بشكل صحيح عن طريق تشغيل الأمر التالي:

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love you'))"
```

تحقق مما إذا كان 🤗 Transformers قد تم تثبيته بشكل صحيح عن طريق تشغيل الأمر التالي:

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love you'))"
```

## التثبيت القابل للتعديل

ستحتاج إلى تثبيت قابل للتعديل إذا كنت ترغب في:

* استخدام إصدار `main` من كود المصدر.
* المساهمة في 🤗 Transformers وتحتاج إلى اختبار التغييرات في الكود.

قم باستنساخ المستودع وقم بتثبيت 🤗 Transformers باستخدام الأوامر التالية:

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

 ستقوم هذه الأوامر بربط المجلد الذي قمت باستنساخ المستودع فيه بمسارات مكتبة Python. بمعنى آخر، سيبحث Python داخل المجلد الذي قمت باستنساخه بالإضافة إلى المسارات المعتادة للمكتبات. على سبيل المثال، إذا تم تثبيت حزم Python الخاصة بك عادةً في `~/anaconda3/envs/main/lib/python3.10/site-packages/`, فسيقوم Python أيضًا بالبحث في المجلد الذي قمت باستنساخه: `~/transformers/`.

<Tip warning={true}>

يجب عليك الاحتفاظ بمجلد `transformers` إذا كنت تريد الاستمرار في استخدام المكتبة.

</Tip>

الآن يمكنك تحديث المستنسخ الخاص بك بسهولة إلى أحدث إصدار من 🤗 Transformers باستخدام الأمر التالي:

```bash
cd ~/transformers/
git pull
```

ستجد بيئة Python الإصدار `main` من 🤗 Transformers في المرة التالية التي تقوم فيها بتشغيله.

## التثبيت باستخدام conda

قم بالتثبيت من قناة conda `conda-forge`:

```bash
conda install conda-forge::transformers
```

## إعداد ذاكرة التخزين المؤقت

تُحمّل النماذج المُسبقة التدريب وتُخزّن مؤقتًا في: `~/.cache/huggingface/hub`. هذا هو المجلد الافتراضي الذي يُحدده متغير البيئة `HF_HUB_CACHE`. على Windows، يكون دليل ذاكرة التخزين المؤقت الافتراضي هو `C:\Users\username\.cache\huggingface\hub`. يمكنك تغيير متغيرات البيئة shell الموضحة أدناه - حسب الأولوية - لتحديد دليل ذاكرة تخزين مؤقت مختلف:

1. متغير البيئة (افتراضي): `HF_HUB_CACHE`.
2. متغير البيئة: `HF_HOME`.
3. متغير البيئة: `XDG_CACHE_HOME` + `/huggingface`.


## الوضع دون اتصال بالإنترنت

قم بتشغيل 🤗 Transformers في بيئة محمية بجدار حماية أو غير متصلة باستخدام الملفات المخزنة مؤقتًا محليًا عن طريق تعيين متغير البيئة `HF_HUB_OFFLINE=1`.

<Tip>

أضف [🤗 Datasets](https://huggingface.co/docs/datasets/) إلى سير عمل التدريب غير المتصل باستخدام متغير البيئة `HF_DATASETS_OFFLINE=1`.

</Tip>

يمكنك أيضًا تجاوز تحميل نموذج من Hub من كل استدعاء [`~PreTrainedModel.from_pretrained`] باستخدام معلمة [`local_files_only`]. عندما يتم تعيينها على `True`، يتم تحميل الملفات المحلية فقط:

```py
from transformers import T5Model

model = T5Model.from_pretrained("./path/to/local/directory", local_files_only=True)
```

### جلب النماذج والمُجزّئات لاستخدامها دون اتصال بالإنترنت

خيار آخر لاستخدام 🤗 Transformers دون اتصال هو تنزيل الملفات مسبقًا، ثم الإشارة إلى مسارها المحلي عند الحاجة إلى استخدامها دون اتصال. هناك ثلاث طرق للقيام بذلك:

* قم بتنزيل ملف عبر واجهة المستخدم على [Model Hub](https://huggingface.co/models) بالنقر فوق أيقونة ↓.

    ![download-icon](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/download-icon.png)

* استخدم سير عمل [`PreTrainedModel.from_pretrained`] و [`PreTrainedModel.save_pretrained`]:

    1. قم بتنزيل ملفاتك مسبقًا باستخدام [`PreTrainedModel.from_pretrained`]:
* استخدم سير عمل [`PreTrainedModel.from_pretrained`] و [`PreTrainedModel.save_pretrained`]:

    1. قم بتنزيل ملفاتك مسبقًا باستخدام [`PreTrainedModel.from_pretrained`]:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")
    ```

    2. احفظ ملفاتك إلى دليل محدد باستخدام [`PreTrainedModel.save_pretrained`]:

    ```py
    >>> tokenizer.save_pretrained("./your/path/bigscience_t0")
    >>> model.save_pretrained("./your/path/bigscience_t0")
    ```

    3. الآن عندما تكون غير متصل بالإنترنت، أعد تحميل ملفاتك باستخدام [`PreTrainedModel.from_pretrained`] من الدليل المحدد:

    ```py
    >>> tokenizer = AutoTokenizer.from_pretrained("./your/path/bigscience_t0")
    >>> model = AutoModel.from_pretrained("./your/path/bigscience_t0")
    ```

* قم بتنزيل الملفات برمجيًا باستخدام مكتبة [huggingface_hub](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub):

    1. قم بتثبيت مكتبة `huggingface_hub` في بيئتك الافتراضية:

    ```bash
    python -m pip install huggingface_hub
    ```

    2. استخدم وظيفة [`hf_hub_download`](https://huggingface.co/docs/hub/adding-a-library#download-files-from-the-hub) لتنزيل ملف إلى مسار محدد. على سبيل المثال، يقوم الأمر التالي بتنزيل ملف `config.json` من نموذج [T0](https://huggingface.co/bigscience/T0_3B) إلى المسار المطلوب:

    ```py
    >>> from huggingface_hub import hf_hub_download

    >>> hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="./your/path/bigscience_t0")
    ```

بمجرد تنزيل ملفك وتخزينه مؤقتًا محليًا، حدد مساره المحلي الخاص به لتحميله واستخدامه:

```py
>>> from transformers import AutoConfig

>>> config = AutoConfig.from_pretrained("./your/path/bigscience_t0/config.json")
```

<Tip>

راجع قسم [كيفية تنزيل الملفات من Hub](https://huggingface.co/docs/hub/how-to-downstream) لمزيد من التفاصيل حول تنزيل الملفات المخزنة على Hub.

</Tip>
