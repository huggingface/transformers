# BERTology

يُشهد في الآونة الأخيرة نمو مجال دراسي يُعنى باستكشاف آلية عمل نماذج المحولات الضخمة مثل BERT (والذي يُطلق عليها البعض اسم "BERTology"). ومن الأمثلة البارزة على هذا المجال ما يلي:

- BERT Rediscovers the Classical NLP Pipeline بواسطة Ian Tenney و Dipanjan Das و Ellie Pavlick:
  https://arxiv.org/abs/1905.05950
- Are Sixteen Heads Really Better than One? بواسطة Paul Michel و Omer Levy و Graham Neubig: https://arxiv.org/abs/1905.10650
- What Does BERT Look At? An Analysis of BERT's Attention بواسطة Kevin Clark و Urvashi Khandelwal و Omer Levy و Christopher D.
  Manning: https://arxiv.org/abs/1906.04341
- CAT-probing: A Metric-based Approach to Interpret How Pre-trained Models for Programming Language Attend Code Structure: https://arxiv.org/abs/2210.04633

لإثراء هذا المجال الناشئ، قمنا بتضمين بعض الميزات الإضافية في نماذج BERT/GPT/GPT-2 للسماح للناس بالوصول إلى التمثيلات الداخلية، والتي تم تكييفها بشكل أساسي من العمل الرائد لـ Paul Michel (https://arxiv.org/abs/1905.10650):

- الوصول إلى جميع الحالات المخفية في BERT/GPT/GPT-2،
- الوصول إلى جميع أوزان الانتباه لكل رأس في BERT/GPT/GPT-2،
- استرجاع قيم ومشتقات  مخرجات الرأس لحساب درجة أهمية الرأس وحذفه كما هو موضح في https://arxiv.org/abs/1905.10650.

ولمساعدتك على فهم واستخدام هذه الميزات بسهولة، أضفنا مثالًا برمجيًا محددًا: [bertology.py](https://github.com/huggingface/transformers/tree/main/examples/research_projects/bertology/run_bertology.py) أثناء استخراج المعلومات  وتقليص من نموذج تم تدريبه مسبقًا على GLUE.