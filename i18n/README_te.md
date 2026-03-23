<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>


<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/transformers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>


<h4 align="center">
    <p>
        <a href="https://github.com/huggingface/transformers/">English</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hant.md">繁體中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_es.md">Español</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ja.md">日本語</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_hd.md">हिन्दी</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ru.md">Русский</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_pt-br.md">Рortuguês</a> |
        <b>తెలుగు</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_it.md">Italiano</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_bn.md">বাংলা</a> |
    </p>
</h4>

<h3 align="center">
    <p>JAX, PyTorch మరియు TensorFlow కోసం అత్యాధునిక యంత్ర అభ్యాసం</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>

🤗 ట్రాన్స్‌ఫార్మర్లు టెక్స్ట్, విజన్ మరియు ఆడియో వంటి విభిన్న పద్ధతులపై టాస్క్‌లను నిర్వహించడానికి వేలాది ముందుగా శిక్షణ పొందిన మోడల్‌లను అందిస్తాయి.

ఈ నమూనాలు వర్తించవచ్చు:

* 📝 టెక్స్ట్, 100కి పైగా భాషల్లో టెక్స్ట్ క్లాసిఫికేషన్, ఇన్ఫర్మేషన్ ఎక్స్‌ట్రాక్షన్, ప్రశ్నలకు సమాధానాలు, సారాంశం, అనువాదం, టెక్స్ట్ జనరేషన్ వంటి పనుల కోసం.
* 🖼️ ఇమేజ్‌లు, ఇమేజ్ వర్గీకరణ, ఆబ్జెక్ట్ డిటెక్షన్ మరియు సెగ్మెంటేషన్ వంటి పనుల కోసం.
* 🗣️ ఆడియో, స్పీచ్ రికగ్నిషన్ మరియు ఆడియో వర్గీకరణ వంటి పనుల కోసం.

ట్రాన్స్‌ఫార్మర్ మోడల్‌లు టేబుల్ క్వశ్చన్ ఆన్సర్ చేయడం, ఆప్టికల్ క్యారెక్టర్ రికగ్నిషన్, స్కాన్ చేసిన డాక్యుమెంట్‌ల నుండి ఇన్ఫర్మేషన్ ఎక్స్‌ట్రాక్షన్, వీడియో క్లాసిఫికేషన్ మరియు విజువల్ క్వశ్చన్ ఆన్సర్ చేయడం వంటి **అనేక పద్ధతులతో కలిపి** పనులను కూడా చేయగలవు.

🤗 ట్రాన్స్‌ఫార్మర్లు అందించిన టెక్స్ట్‌లో ప్రీట్రైన్డ్ మోడల్‌లను త్వరగా డౌన్‌లోడ్ చేయడానికి మరియు ఉపయోగించడానికి, వాటిని మీ స్వంత డేటాసెట్‌లలో ఫైన్-ట్యూన్ చేయడానికి మరియు వాటిని మా [మోడల్ హబ్](https://huggingface.co/models)లో సంఘంతో భాగస్వామ్యం చేయడానికి API లను అందిస్తుంది. అదే సమయంలో, ఆర్కిటెక్చర్‌ని నిర్వచించే ప్రతి పైథాన్ మాడ్యూల్ పూర్తిగా స్వతంత్రంగా ఉంటుంది మరియు త్వరిత పరిశోధన ప్రయోగాలను ప్రారంభించడానికి సవరించవచ్చు.

🤗 ట్రాన్స్‌ఫార్మర్‌లకు మూడు అత్యంత ప్రజాదరణ పొందిన డీప్ లెర్నింగ్ లైబ్రరీలు ఉన్నాయి — [Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/) మరియు [TensorFlow](https://www.tensorflow.org/) — వాటి మధ్య అతుకులు లేని ఏకీకరణతో. మీ మోడల్‌లను ఒకదానితో మరొకదానితో అనుమితి కోసం లోడ్ చేసే ముందు వాటికి శిక్షణ ఇవ్వడం చాలా సులభం.

## ఆన్‌లైన్ డెమోలు

మీరు [మోడల్ హబ్](https://huggingface.co/models) నుండి మా మోడళ్లలో చాలా వరకు వాటి పేజీలలో నేరుగా పరీక్షించవచ్చు. మేము పబ్లిక్ మరియు ప్రైవేట్ మోడల్‌ల కోసం [ప్రైవేట్ మోడల్ హోస్టింగ్, సంస్కరణ & అనుమితి API](https://huggingface.co/pricing)ని కూడా అందిస్తాము.

ఇక్కడ కొన్ని ఉదాహరణలు ఉన్నాయి:

సహజ భాషా ప్రాసెసింగ్‌లో:
- [BERT తో మాస్క్‌డ్ వర్డ్ కంప్లీషన్](https://huggingface.co/google-bert/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [Electra తో పేరు ఎంటిటీ గుర్తింపు](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [GPT-2 తో టెక్స్ట్ జనరేషన్](https://huggingface.co/openai-community/gpt2?text=A+long+time+ago%2C+)
- [RoBERTa తో సహజ భాషా అనుమితి](https://huggingface.co/FacebookAI/roberta-large-mnli?text=The+dog+was+Lost.+Nobody+lost+any+animal)
- [BART తో సారాంశం](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct)
- [DistilBERT తో ప్రశ్న సమాధానం](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species)
- [T5 తో అనువాదం](https://huggingface.co/google-t5/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin)

కంప్యూటర్ దృష్టిలో:
- [VIT తో చిత్ర వర్గీకరణ](https://huggingface.co/google/vit-base-patch16-224)
- [DETR తో ఆబ్జెక్ట్ డిటెక్షన్](https://huggingface.co/facebook/detr-resnet-50)
- [SegFormer తో సెమాంటిక్ సెగ్మెంటేషన్](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- [MaskFormer తో పానోప్టిక్ సెగ్మెంటేషన్](https://huggingface.co/facebook/maskformer-swin-small-coco)
- [DPT తో లోతు అంచనా](https://huggingface.co/docs/transformers/model_doc/dpt)
- [VideoMAE తో వీడియో వర్గీకరణ](https://huggingface.co/docs/transformers/model_doc/videomae)
- [OneFormer తో యూనివర్సల్ సెగ్మెంటేషన్](https://huggingface.co/shi-labs/oneformer_ade20k_dinat_large)

ఆడియోలో:
- [Wav2Vec2 తో ఆటోమేటిక్ స్పీచ్ రికగ్నిషన్](https://huggingface.co/facebook/wav2vec2-base-960h)
- [Wav2Vec2 తో కీవర్డ్ స్పాటింగ్](https://huggingface.co/superb/wav2vec2-base-superb-ks)
- [ఆడియో స్పెక్ట్రోగ్రామ్ ట్రాన్స్‌ఫార్మర్‌తో ఆడియో వర్గీకరణ](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

మల్టీమోడల్ టాస్క్‌లలో:
- [TAPAS తో టేబుల్ ప్రశ్న సమాధానాలు](https://huggingface.co/google/tapas-base-finetuned-wtq)
- [ViLT తో దృశ్యమాన ప్రశ్నకు సమాధానం](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
- [CLIP తో జీరో-షాట్ ఇమేజ్ వర్గీకరణ](https://huggingface.co/openai/clip-vit-large-patch14)
- [LayoutLM తో డాక్యుమెంట్ ప్రశ్నకు సమాధానం](https://huggingface.co/impira/layoutlm-document-qa)
- [X-CLIP తో జీరో-షాట్ వీడియో వర్గీకరణ](https://huggingface.co/docs/transformers/model_doc/xclip)

## ట్రాన్స్‌ఫార్మర్‌లను ఉపయోగించి 100 ప్రాజెక్టులు

ట్రాన్స్‌ఫార్మర్లు ప్రీట్రైన్డ్ మోడల్‌లను ఉపయోగించడానికి టూల్‌కిట్ కంటే ఎక్కువ: ఇది దాని చుట్టూ నిర్మించిన ప్రాజెక్ట్‌ల సంఘం మరియు
హగ్గింగ్ ఫేస్ హబ్. డెవలపర్‌లు, పరిశోధకులు, విద్యార్థులు, ప్రొఫెసర్‌లు, ఇంజనీర్లు మరియు ఎవరినైనా అనుమతించేలా ట్రాన్స్‌ఫార్మర్‌లను మేము కోరుకుంటున్నాము
వారి కలల ప్రాజెక్టులను నిర్మించడానికి.

ట్రాన్స్‌ఫార్మర్‌ల 100,000 నక్షత్రాలను జరుపుకోవడానికి, మేము స్పాట్‌లైట్‌ని ఉంచాలని నిర్ణయించుకున్నాము
సంఘం, మరియు మేము 100 జాబితాలను కలిగి ఉన్న [awesome-transformers](https://github.com/huggingface/transformers/blob/main/awesome-transformers.md) పేజీని సృష్టించాము.
ట్రాన్స్‌ఫార్మర్ల పరిసరాల్లో అద్భుతమైన ప్రాజెక్టులు నిర్మించబడ్డాయి.

జాబితాలో భాగమని మీరు విశ్వసించే ప్రాజెక్ట్‌ను మీరు కలిగి ఉంటే లేదా ఉపయోగిస్తుంటే, దయచేసి దానిని జోడించడానికి PRని తెరవండి!

## మీరు హగ్గింగ్ ఫేస్ టీమ్ నుండి అనుకూల మద్దతు కోసం చూస్తున్నట్లయితే

<a target="_blank" href="https://huggingface.co/support">
    <img alt="HuggingFace Expert Acceleration Program" src="https://cdn-media.huggingface.co/marketing/transformers/new-support-improved.png" style="max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a><br>

## త్వరిత పర్యటన

ఇచ్చిన ఇన్‌పుట్ (టెక్స్ట్, ఇమేజ్, ఆడియో, ...)పై తక్షణమే మోడల్‌ను ఉపయోగించడానికి, మేము `pipeline` API ని అందిస్తాము. పైప్‌లైన్‌లు ఆ మోడల్ శిక్షణ సమయంలో ఉపయోగించిన ప్రీప్రాసెసింగ్‌తో కూడిన ప్రీట్రైన్డ్ మోడల్‌ను సమూహపరుస్తాయి. సానుకూల మరియు ప్రతికూల పాఠాలను వర్గీకరించడానికి పైప్‌లైన్‌ను త్వరగా ఎలా ఉపయోగించాలో ఇక్కడ ఉంది:

```python
>>> from transformers import pipeline

# Allocate a pipeline for sentiment-analysis
>>> classifier = pipeline('sentiment-analysis')
>>> classifier('We are very happy to introduce pipeline to the transformers repository.')
[{'label': 'POSITIVE', 'score': 0.9996980428695679}]
```

రెండవ లైన్ కోడ్ డౌన్‌లోడ్ మరియు పైప్‌లైన్ ఉపయోగించే ప్రీట్రైన్డ్ మోడల్‌ను కాష్ చేస్తుంది, మూడవది ఇచ్చిన టెక్స్ట్‌పై మూల్యాంకనం చేస్తుంది. ఇక్కడ సమాధానం 99.97% విశ్వాసంతో "పాజిటివ్".

చాలా పనులు NLPలో కానీ కంప్యూటర్ విజన్ మరియు స్పీచ్‌లో కూడా ముందుగా శిక్షణ పొందిన `pipeline` సిద్ధంగా ఉన్నాయి. ఉదాహరణకు, మనం చిత్రంలో గుర్తించిన వస్తువులను సులభంగా సంగ్రహించవచ్చు:

``` python
>>> import requests
>>> from PIL import Image
>>> from transformers import pipeline

# Download an image with cute cats
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
>>> image_data = requests.get(url, stream=True).raw
>>> image = Image.open(image_data)

# Allocate a pipeline for object detection
>>> object_detector = pipeline('object-detection')
>>> object_detector(image)
[{'score': 0.9982201457023621,
  'label': 'remote',
  'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}},
 {'score': 0.9960021376609802,
  'label': 'remote',
  'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}},
 {'score': 0.9954745173454285,
  'label': 'couch',
  'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}},
 {'score': 0.9988006353378296,
  'label': 'cat',
  'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}},
 {'score': 0.9986783862113953,
  'label': 'cat',
  'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}]
```

ఇక్కడ మనం ఆబ్జెక్ట్ చుట్టూ ఉన్న బాక్స్ మరియు కాన్ఫిడెన్స్ స్కోర్‌తో చిత్రంలో గుర్తించబడిన వస్తువుల జాబితాను పొందుతాము. ఇక్కడ ఎడమవైపున ఉన్న అసలు చిత్రం, కుడివైపున అంచనాలు ప్రదర్శించబడతాయి:

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png" width="400"></a>
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample_post_processed.png" width="400"></a>
</h3>

మీరు [ఈ ట్యుటోరియల్](https://huggingface.co/docs/transformers/task_summary)లో `pipeline` API ద్వారా సపోర్ట్ చేసే టాస్క్‌ల గురించి మరింత తెలుసుకోవచ్చు.

`pipeline`తో పాటు, మీరు ఇచ్చిన టాస్క్‌లో ఏదైనా ప్రీట్రైన్డ్ మోడల్‌లను డౌన్‌లోడ్ చేయడానికి మరియు ఉపయోగించడానికి, దీనికి మూడు లైన్ల కోడ్ సరిపోతుంది. ఇక్కడ PyTorch వెర్షన్ ఉంది:
```python
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="pt")
>>> outputs = model(**inputs)
```

మరియు TensorFlow కి సమానమైన కోడ్ ఇక్కడ ఉంది:
```python
>>> from transformers import AutoTokenizer, TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="tf")
>>> outputs = model(**inputs)
```

ప్రిట్రైన్డ్ మోడల్ ఆశించే అన్ని ప్రీప్రాసెసింగ్‌లకు టోకెనైజర్ బాధ్యత వహిస్తుంది మరియు నేరుగా ఒకే స్ట్రింగ్ (పై ఉదాహరణలలో వలె) లేదా జాబితాపై కాల్ చేయవచ్చు. ఇది మీరు డౌన్‌స్ట్రీమ్ కోడ్‌లో ఉపయోగించగల నిఘంటువుని అవుట్‌పుట్ చేస్తుంది లేదా ** ఆర్గ్యుమెంట్ అన్‌ప్యాకింగ్ ఆపరేటర్‌ని ఉపయోగించి నేరుగా మీ మోడల్‌కి పంపుతుంది.

మోడల్ కూడా సాధారణ [Pytorch `nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) లేదా [TensorFlow `tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) (మీ బ్యాకెండ్‌ని బట్టి) మీరు మామూలుగా ఉపయోగించవచ్చు. [ఈ ట్యుటోరియల్](https://huggingface.co/docs/transformers/training) అటువంటి మోడల్‌ని క్లాసిక్ PyTorch లేదా TensorFlow ట్రైనింగ్ లూప్‌లో ఎలా ఇంటిగ్రేట్ చేయాలో లేదా మా `Trainer` API ని ఎలా ఉపయోగించాలో వివరిస్తుంది కొత్త డేటాసెట్.

## నేను ట్రాన్స్‌ఫార్మర్‌లను ఎందుకు ఉపయోగించాలి?

1. ఉపయోగించడానికి సులభమైన స్టేట్ ఆఫ్ ది ఆర్ట్ మోడల్‌లు:
    - సహజ భాషా అవగాహన & ఉత్పత్తి, కంప్యూటర్ దృష్టి మరియు ఆడియో పనులపై అధిక పనితీరు.
    - విద్యావేత్తలు మరియు అభ్యాసకుల ప్రవేశానికి తక్కువ అవరోధం.
    - తెలుసుకోవడానికి కేవలం మూడు తరగతులతో కొన్ని వినియోగదారు-ముఖ సంగ్రహణలు.
    - మా అన్ని ప్రీట్రైన్డ్ మోడల్‌లను ఉపయోగించడం కోసం ఏకీకృత API.

2. తక్కువ గణన ఖర్చులు, చిన్న కార్బన్ పాదముద్ర:
    - పరిశోధకులు ఎల్లప్పుడూ మళ్లీ శిక్షణ పొందే బదులు శిక్షణ పొందిన నమూనాలను పంచుకోవచ్చు.
    - అభ్యాసకులు గణన సమయాన్ని మరియు ఉత్పత్తి ఖర్చులను తగ్గించగలరు.
    - అన్ని పద్ధతుల్లో 60,000 కంటే ఎక్కువ ప్రీట్రైన్డ్ మోడల్‌లతో డజన్ల కొద్దీ ఆర్కిటెక్చర్‌లు.

3. మోడల్ జీవితకాలంలో ప్రతి భాగానికి సరైన ఫ్రేమ్‌వర్క్‌ను ఎంచుకోండి:
    - 3 లైన్ల కోడ్‌లో స్టేట్ ఆఫ్ ది ఆర్ట్ మోడల్‌లకు శిక్షణ ఇవ్వండి.
    - TF2.0/PyTorch/JAX ఫ్రేమ్‌వర్క్‌ల మధ్య ఒకే మోడల్‌ను ఇష్టానుసారంగా తరలించండి.
    - శిక్షణ, మూల్యాంకనం మరియు ఉత్పత్తి కోసం సరైన ఫ్రేమ్‌వర్క్‌ను సజావుగా ఎంచుకోండి.

4. మీ అవసరాలకు అనుగుణంగా మోడల్ లేదా ఉదాహరణను సులభంగా అనుకూలీకరించండి:
    - ప్రతి ఆర్కిటెక్చర్ దాని అసలు రచయితలు ప్రచురించిన ఫలితాలను పునరుత్పత్తి చేయడానికి మేము ఉదాహరణలను అందిస్తాము.
    - మోడల్ ఇంటర్నల్‌లు వీలైనంత స్థిరంగా బహిర్గతమవుతాయి.
    - శీఘ్ర ప్రయోగాల కోసం లైబ్రరీ నుండి స్వతంత్రంగా మోడల్ ఫైల్‌లను ఉపయోగించవచ్చు.

## నేను ట్రాన్స్‌ఫార్మర్‌లను ఎందుకు ఉపయోగించకూడదు?

- ఈ లైబ్రరీ న్యూరల్ నెట్‌ల కోసం బిల్డింగ్ బ్లాక్‌ల మాడ్యులర్ టూల్‌బాక్స్ కాదు. మోడల్ ఫైల్‌లలోని కోడ్ ఉద్దేశపూర్వకంగా అదనపు సంగ్రహణలతో రీఫ్యాక్టరింగ్ చేయబడదు, తద్వారా పరిశోధకులు అదనపు సంగ్రహణలు/ఫైళ్లలోకి ప్రవేశించకుండా ప్రతి మోడల్‌పై త్వరగా మళ్లించగలరు.
- శిక్షణ API ఏ మోడల్‌లో పని చేయడానికి ఉద్దేశించబడలేదు కానీ లైబ్రరీ అందించిన మోడల్‌లతో పని చేయడానికి ఆప్టిమైజ్ చేయబడింది. సాధారణ మెషిన్ లెర్నింగ్ లూప్‌ల కోసం, మీరు మరొక లైబ్రరీని ఉపయోగించాలి (బహుశా, [Accelerate](https://huggingface.co/docs/accelerate)).
- మేము వీలైనన్ని ఎక్కువ వినియోగ సందర్భాలను ప్రదర్శించడానికి ప్రయత్నిస్తున్నప్పుడు, మా [ఉదాహరణల ఫోల్డర్](https://github.com/huggingface/transformers/tree/main/examples)లోని స్క్రిప్ట్‌లు కేవలం: ఉదాహరణలు. మీ నిర్దిష్ట సమస్యపై అవి పని చేయవు మరియు వాటిని మీ అవసరాలకు అనుగుణంగా మార్చుకోవడానికి మీరు కొన్ని కోడ్ లైన్‌లను మార్చవలసి ఉంటుంది.

## సంస్థాపన

### పిప్ తో

ఈ రిపోజిటరీ పైథాన్ 3.10+ మరియు PyTorch 2.4+లో పరీక్షించబడింది.

మీరు [వర్చువల్ వాతావరణం](https://docs.python.org/3/library/venv.html)లో 🤗 ట్రాన్స్‌ఫార్మర్‌లను ఇన్‌స్టాల్ చేయాలి. మీకు పైథాన్ వర్చువల్ పరిసరాల గురించి తెలియకుంటే, [యూజర్ గైడ్](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) చూడండి.

ముందుగా, మీరు ఉపయోగించబోతున్న పైథాన్ వెర్షన్‌తో వర్చువల్ వాతావరణాన్ని సృష్టించండి మరియు దానిని సక్రియం చేయండి.

అప్పుడు, మీరు ఫ్లాక్స్, పైటార్చ్ లేదా టెన్సర్‌ఫ్లోలో కనీసం ఒకదానిని ఇన్‌స్టాల్ చేయాలి.
దయచేసి [TensorFlow ఇన్‌స్టాలేషన్ పేజీ](https://www.tensorflow.org/install/), [PyTorch ఇన్‌స్టాలేషన్ పేజీ](https://pytorch.org/get-started/locally/#start-locally) మరియు/ని చూడండి లేదా మీ ప్లాట్‌ఫారమ్ కోసం నిర్దిష్ట ఇన్‌స్టాలేషన్ కమాండ్‌కు సంబంధించి [Flax](https://github.com/google/flax#quick-install) మరియు [Jax](https://github.com/google/jax#installation) ఇన్‌స్టాలేషన్ పేజీలు .

ఆ బ్యాకెండ్‌లలో ఒకటి ఇన్‌స్టాల్ చేయబడినప్పుడు, 🤗 ట్రాన్స్‌ఫార్మర్‌లను ఈ క్రింది విధంగా పిప్‌ని ఉపయోగించి ఇన్‌స్టాల్ చేయవచ్చు:

```bash
pip install transformers
```

మీరు ఉదాహరణలతో ప్లే చేయాలనుకుంటే లేదా కోడ్ యొక్క బ్లీడింగ్ ఎడ్జ్ అవసరం మరియు కొత్త విడుదల కోసం వేచి ఉండలేకపోతే, మీరు తప్పనిసరిగా [మూలం నుండి లైబ్రరీని ఇన్‌స్టాల్ చేయాలి](https://huggingface.co/docs/transformers/installation#installing-from-source).

### కొండా తో

🤗 కింది విధంగా కొండా ఉపయోగించి ట్రాన్స్‌ఫార్మర్‌లను ఇన్‌స్టాల్ చేయవచ్చు:

```shell script
conda install conda-forge::transformers
```

> **_గమనిక:_** `huggingface` ఛానెల్ నుండి `transformers` ఇన్‌స్టాల్ చేయడం పురాతనంగా ఉంది.

Flax, PyTorch లేదా TensorFlow యొక్క ఇన్‌స్టాలేషన్ పేజీలను కొండాతో ఎలా ఇన్‌స్టాల్ చేయాలో చూడటానికి వాటిని అనుసరించండి.

> **_గమనిక:_** Windowsలో, కాషింగ్ నుండి ప్రయోజనం పొందేందుకు మీరు డెవలపర్ మోడ్‌ని సక్రియం చేయమని ప్రాంప్ట్ చేయబడవచ్చు. ఇది మీకు ఎంపిక కాకపోతే, దయచేసి [ఈ సంచిక](https://github.com/huggingface/huggingface_hub/issues/1062)లో మాకు తెలియజేయండి.

## మోడల్ ఆర్కిటెక్చర్లు

**[అన్ని మోడల్ చెక్‌పాయింట్‌లు](https://huggingface.co/models)** 🤗 అందించిన ట్రాన్స్‌ఫార్మర్లు huggingface.co [model hub](https://huggingface.co/models) నుండి సజావుగా ఏకీకృతం చేయబడ్డాయి [users](https://huggingface.co/users) మరియు [organizations](https://huggingface.co/organizations) ద్వారా నేరుగా అప్‌లోడ్ చేయబడతాయి.

ప్రస్తుత తనిఖీ కేంద్రాల సంఖ్య: ![](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)

🤗 ట్రాన్స్‌ఫార్మర్లు ప్రస్తుతం కింది ఆర్కిటెక్చర్‌లను అందజేస్తున్నాయి: వాటిలో ప్రతి ఒక్కటి ఉన్నత స్థాయి సారాంశం కోసం [ఇక్కడ](https://huggingface.co/docs/transformers/model_summary) చూడండి.

ఈ అమలులు అనేక డేటాసెట్‌లలో పరీక్షించబడ్డాయి (ఉదాహరణ స్క్రిప్ట్‌లను చూడండి) మరియు అసలైన అమలుల పనితీరుతో సరిపోలాలి. మీరు [డాక్యుమెంటేషన్](https://github.com/huggingface/transformers/tree/main/examples) యొక్క ఉదాహరణల విభాగంలో పనితీరుపై మరిన్ని వివరాలను కనుగొనవచ్చు.

## ఇంకా నేర్చుకో

| విభాగం | వివరణ |
|-|-|
| [డాక్యుమెంటేషన్](https://huggingface.co/docs/transformers/) | పూర్తి API డాక్యుమెంటేషన్ మరియు ట్యుటోరియల్స్ |
| [టాస్క్ సారాంశం](https://huggingface.co/docs/transformers/task_summary) | 🤗 ట్రాన్స్‌ఫార్మర్‌ల ద్వారా సపోర్ట్ చేయబడిన విధులు |
| [ప్రీప్రాసెసింగ్ ట్యుటోరియల్](https://huggingface.co/docs/transformers/preprocessing) | మోడల్‌ల కోసం డేటాను సిద్ధం చేయడానికి `Tokenizer` క్లాస్‌ని ఉపయోగించడం |
| [ట్రైనింగ్ మరియు ఫైన్-ట్యూనింగ్](https://huggingface.co/docs/transformers/training) | PyTorch/TensorFlow ట్రైనింగ్ లూప్ మరియు `Trainer` APIలో 🤗 ట్రాన్స్‌ఫార్మర్లు అందించిన మోడల్‌లను ఉపయోగించడం |
| [త్వరిత పర్యటన: ఫైన్-ట్యూనింగ్/యూసేజ్ స్క్రిప్ట్‌లు](https://github.com/huggingface/transformers/tree/main/examples) | విస్తృత శ్రేణి టాస్క్‌లపై ఫైన్-ట్యూనింగ్ మోడల్స్ కోసం ఉదాహరణ స్క్రిప్ట్‌లు |
| [మోడల్ భాగస్వామ్యం మరియు అప్‌లోడ్ చేయడం](https://huggingface.co/docs/transformers/model_sharing) | కమ్యూనిటీతో మీ ఫైన్-ట్యూన్డ్ మోడల్‌లను అప్‌లోడ్ చేయండి మరియు భాగస్వామ్యం చేయండి |

## అనులేఖనం

🤗 ట్రాన్స్‌ఫార్మర్స్ లైబ్రరీ కోసం మీరు ఉదహరించగల [పేపర్](https://www.aclweb.org/anthology/2020.emnlp-demos.6/) ఇప్పుడు మా వద్ద ఉంది:
```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```
