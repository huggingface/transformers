# CPM

## نظرة عامة
اقترح نموذج CPM في [CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://arxiv.org/abs/2012.00413) بواسطة Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin,
Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen,
Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun.

المقتطف من الورقة هو ما يلي:

* أثبتت نماذج اللغة المعالجة مسبقًا (PLMs) فائدتها لمختلف مهام معالجة اللغات الطبيعية (NLP) التنازلية. مؤخرًا، لفت GPT-3،
بـ 175 مليار معلمة و 570 جيجابايت من بيانات التدريب، الكثير من الانتباه بسبب قدرة التعلم القليلة الإشراف (حتى
التعلم بدون إشراف). ومع ذلك، لا يزال تطبيق GPT-3 لحل مهام NLP الصينية يمثل تحديًا، حيث أن فيلق تدريب GPT-3 هو في الأساس باللغة الإنجليزية، والمعلمات غير متاحة للجمهور. في هذا التقرير الفني، نقدم النموذج الصيني المعالج مسبقًا للغة (CPM) مع المعالجة المولدة مسبقًا على بيانات التدريب الصينية واسعة النطاق. على حد علمنا، يعد CPM، الذي يحتوي على 2.6 مليار معلمة و 100 جيجابايت من بيانات التدريب الصينية، أكبر نموذج لغوي صيني معالج مسبقًا، والذي يمكن أن يسهل العديد من مهام NLP الصينية التنازلية، مثل المحادثة، وتوليد المقالات،
اختبار الاختيار من متعدد، وفهم اللغة. تُظهر التجارب المستفيضة أن CPM يحقق أداءً قويًا في العديد من مهام NLP في إعدادات التعلم القليلة الإشراف (حتى بدون إشراف).

تمت المساهمة بهذا النموذج بواسطة [canwenxu](https://huggingface.co/canwenxu). يمكن العثور على التنفيذ الأصلي
هنا: https://github.com/TsinghuaAI/CPM-Generate

<Tip>

تتشابه بنية CPM مع GPT-2، باستثناء طريقة التمييز. راجع [وثائق GPT-2](gpt2) للحصول على معلومات مرجعية حول واجهة برمجة التطبيقات.

</Tip>

## CpmTokenizer

[[autodoc]] CpmTokenizer

## CpmTokenizerFast

[[autodoc]] CpmTokenizerFast