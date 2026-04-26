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

<!---
A useful guide for English-Hindi translation of Hugging Face documentation
- Add space around English words and numbers when they appear between Hindi characters. E.g., कुल मिलाकर 100 से अधिक भाषाएँ; ट्रांसफॉर्मर लाइब्रेरी का उपयोग करता है।
- वर्गाकार उद्धरणों का प्रयोग करें, जैसे, "उद्धरण"

Dictionary

Hugging Face: गले लगाओ चेहरा
token: शब्द (और मूल अंग्रेजी को कोष्ठक में चिह्नित करें）
tokenize: टोकननाइज़ करें (और मूल अंग्रेज़ी को चिह्नित करने के लिए कोष्ठक का उपयोग करें)
tokenizer: Tokenizer (मूल अंग्रेजी में कोष्ठक के साथ)
transformer: transformer
pipeline: समनुक्रम
API: API (अनुवाद के बिना)
inference: विचार
Trainer: प्रशिक्षक। कक्षा के नाम के रूप में प्रस्तुत किए जाने पर अनुवादित नहीं किया गया।
pretrained/pretrain: पूर्व प्रशिक्षण
finetune: फ़ाइन ट्यूनिंग
community: समुदाय
example: जब विशिष्ट गोदाम example कैटलॉग करते समय "केस केस" के रूप में अनुवादित
Python data structures (e.g., list, set, dict): मूल अंग्रेजी को चिह्नित करने के लिए सूचियों, सेटों, शब्दकोशों में अनुवाद करें और कोष्ठक का उपयोग करें
NLP/Natural Language Processing: द्वारा NLP अनुवाद के बिना प्रकट होते हैं Natural Language Processing प्रस्तुत किए जाने पर प्राकृतिक भाषा संसाधन में अनुवाद करें
checkpoint: जाँच बिंदु
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
        <b>हिन्दी</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ru.md">Русский</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_pt-br.md">Рortuguês</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_it.md">Italiano</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_bn.md">বাংলা</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/ReADME_id.md">Bahasa Indonesia</a> |
    </p>
</h4>

<h3 align="center">
    <p>Jax, PyTorch और TensorFlow के लिए उन्नत मशीन लर्निंग</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>

🤗 Transformers 100 से अधिक भाषाओं में पाठ वर्गीकरण, सूचना निष्कर्षण, प्रश्न उत्तर, सारांशीकरण, अनुवाद, पाठ निर्माण का समर्थन करने के लिए हजारों पूर्व-प्रशिक्षित मॉडल प्रदान करता है। इसका उद्देश्य सबसे उन्नत एनएलपी तकनीक को सभी के लिए सुलभ बनाना है।

🤗 Transformers त्वरित डाउनलोड और उपयोग के लिए एक एपीआई प्रदान करता है, जिससे आप किसी दिए गए पाठ पर एक पूर्व-प्रशिक्षित मॉडल ले सकते हैं, इसे अपने डेटासेट पर ठीक कर सकते हैं और इसे [मॉडल हब](https://huggingface.co/models) के माध्यम से समुदाय के साथ साझा कर सकते हैं। इसी समय, प्रत्येक परिभाषित पायथन मॉड्यूल पूरी तरह से स्वतंत्र है, जो संशोधन और तेजी से अनुसंधान प्रयोगों के लिए सुविधाजनक है।

🤗 Transformers तीन सबसे लोकप्रिय गहन शिक्षण पुस्तकालयों का समर्थन करता है： [Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/) — और इसके साथ निर्बाध रूप से एकीकृत होता है। आप अपने मॉडल को सीधे एक ढांचे के साथ प्रशिक्षित कर सकते हैं और दूसरे के साथ लोड और अनुमान लगा सकते हैं।

## ऑनलाइन डेमो

आप सबसे सीधे मॉडल पृष्ठ पर परीक्षण कर सकते हैं [model hub](https://huggingface.co/models) मॉडल पर। हम [निजी मॉडल होस्टिंग, मॉडल संस्करण, और अनुमान एपीआई](https://huggingface.co/pricing) भी प्रदान करते हैं।。

यहाँ कुछ उदाहरण हैं：
- [शब्द को भरने के लिए मास्क के रूप में BERT का प्रयोग करें](https://huggingface.co/google-bert/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [इलेक्ट्रा के साथ नामित इकाई पहचान](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [जीपीटी-2 के साथ टेक्स्ट जनरेशन](https://huggingface.co/openai-community/gpt2?text=A+long+time+ago%2C+)
- [रॉबर्टा के साथ प्राकृतिक भाषा निष्कर्ष](https://huggingface.co/FacebookAI/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal)
- [बार्ट के साथ पाठ सारांश](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct)
- [डिस्टिलबर्ट के साथ प्रश्नोत्तर](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species)
- [अनुवाद के लिए T5 का प्रयोग करें](https://huggingface.co/google-t5/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin)

**[Write With Transformer](https://transformer.huggingface.co)**，हगिंग फेस टीम द्वारा बनाया गया, यह एक आधिकारिक पाठ पीढ़ी है demo。

## यदि आप हगिंग फेस टीम से बीस्पोक समर्थन की तलाश कर रहे हैं

<a target="_blank" href="https://huggingface.co/support">
    <img alt="HuggingFace Expert Acceleration Program" src="https://huggingface.co/front/thumbnails/support.png" style="max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a><br>

## जल्दी शुरू करें

हम त्वरित उपयोग के लिए मॉडल प्रदान करते हैं `pipeline` (पाइपलाइन) एपीआई। पाइपलाइन पूर्व-प्रशिक्षित मॉडल और संबंधित पाठ प्रीप्रोसेसिंग को एकत्रित करती है। सकारात्मक और नकारात्मक भावना को निर्धारित करने के लिए पाइपलाइनों का उपयोग करने का एक त्वरित उदाहरण यहां दिया गया है:

```python
>>> from transformers import pipeline

# भावना विश्लेषण पाइपलाइन का उपयोग करना
>>> classifier = pipeline('sentiment-analysis')
>>> classifier('We are very happy to introduce pipeline to the transformers repository.')
[{'label': 'POSITIVE', 'score': 0.9996980428695679}]
```

कोड की दूसरी पंक्ति पाइपलाइन द्वारा उपयोग किए गए पूर्व-प्रशिक्षित मॉडल को डाउनलोड और कैश करती है, जबकि कोड की तीसरी पंक्ति दिए गए पाठ पर मूल्यांकन करती है। यहां उत्तर 99 आत्मविश्वास के स्तर के साथ "सकारात्मक" है।

कई एनएलपी कार्यों में आउट ऑफ़ द बॉक्स पाइपलाइनों का पूर्व-प्रशिक्षण होता है। उदाहरण के लिए, हम किसी दिए गए पाठ से किसी प्रश्न का उत्तर आसानी से निकाल सकते हैं:

``` python
>>> from transformers import pipeline

# प्रश्नोत्तर पाइपलाइन का उपयोग करना
>>> question_answerer = pipeline('question-answering')
>>> question_answerer({
...     'question': 'What is the name of the repository ?',
...     'context': 'Pipeline has been included in the huggingface/transformers repository'
... })
{'score': 0.30970096588134766, 'start': 34, 'end': 58, 'answer': 'huggingface/transformers'}

```

उत्तर देने के अलावा, पूर्व-प्रशिक्षित मॉडल संगत आत्मविश्वास स्कोर भी देता है, जहां उत्तर टोकनयुक्त पाठ में शुरू और समाप्त होता है। आप [इस ट्यूटोरियल](https://huggingface.co/docs/transformers/task_summary) से पाइपलाइन एपीआई द्वारा समर्थित कार्यों के बारे में अधिक जान सकते हैं।

अपने कार्य पर किसी भी पूर्व-प्रशिक्षित मॉडल को डाउनलोड करना और उसका उपयोग करना भी कोड की तीन पंक्तियों की तरह सरल है। यहाँ PyTorch संस्करण के लिए एक उदाहरण दिया गया है:
```python
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="pt")
>>> outputs = model(**inputs)
```
यहाँ समकक्ष है TensorFlow कोड:
```python
>>> from transformers import AutoTokenizer, TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="tf")
>>> outputs = model(**inputs)
```

टोकननाइज़र सभी पूर्व-प्रशिक्षित मॉडलों के लिए प्रीप्रोसेसिंग प्रदान करता है और इसे सीधे एक स्ट्रिंग (जैसे ऊपर दिए गए उदाहरण) या किसी सूची पर बुलाया जा सकता है। यह एक डिक्शनरी (तानाशाही) को आउटपुट करता है जिसे आप डाउनस्ट्रीम कोड में उपयोग कर सकते हैं या `**` अनपैकिंग एक्सप्रेशन के माध्यम से सीधे मॉडल को पास कर सकते हैं।

मॉडल स्वयं एक नियमित [Pytorch `nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) या [TensorFlow `tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) (आपके बैकएंड के आधार पर), जो हो सकता है सामान्य तरीके से उपयोग किया जाता है। [यह ट्यूटोरियल](https://huggingface.co/transformers/training.html) बताता है कि इस तरह के मॉडल को क्लासिक PyTorch या TensorFlow प्रशिक्षण लूप में कैसे एकीकृत किया जाए, या हमारे `ट्रेनर` एपीआई का उपयोग कैसे करें ताकि इसे जल्दी से फ़ाइन ट्यून किया जा सके।एक नया डेटासेट पे।

## ट्रांसफार्मर का उपयोग क्यों करें?

1. उपयोग में आसानी के लिए उन्नत मॉडल:
    - एनएलयू और एनएलजी पर बेहतर प्रदर्शन
    - प्रवेश के लिए कम बाधाओं के साथ शिक्षण और अभ्यास के अनुकूल
    - उपयोगकर्ता-सामना करने वाले सार तत्व, केवल तीन वर्गों को जानने की जरूरत है
    - सभी मॉडलों के लिए एकीकृत एपीआई

1. कम कम्प्यूटेशनल ओवरहेड और कम कार्बन उत्सर्जन:
    - शोधकर्ता हर बार नए सिरे से प्रशिक्षण देने के बजाय प्रशिक्षित मॉडल साझा कर सकते हैं
    - इंजीनियर गणना समय और उत्पादन ओवरहेड को कम कर सकते हैं
    - दर्जनों मॉडल आर्किटेक्चर, 2,000 से अधिक पूर्व-प्रशिक्षित मॉडल, 100 से अधिक भाषाओं का समर्थन

1.मॉडल जीवनचक्र के हर हिस्से को शामिल करता है:
    - कोड की केवल 3 पंक्तियों में उन्नत मॉडलों को प्रशिक्षित करें
    - मॉडल को मनमाने ढंग से विभिन्न डीप लर्निंग फ्रेमवर्क के बीच स्थानांतरित किया जा सकता है, जैसा आप चाहते हैं
    - निर्बाध रूप से प्रशिक्षण, मूल्यांकन और उत्पादन के लिए सबसे उपयुक्त ढांचा चुनें

1. आसानी से अनन्य मॉडल को अनुकूलित करें और अपनी आवश्यकताओं के लिए मामलों का उपयोग करें:
    - हम मूल पेपर परिणामों को पुन: पेश करने के लिए प्रत्येक मॉडल आर्किटेक्चर के लिए कई उपयोग के मामले प्रदान करते हैं
    - मॉडल की आंतरिक संरचना पारदर्शी और सुसंगत रहती है
    - मॉडल फ़ाइल को अलग से इस्तेमाल किया जा सकता है, जो संशोधन और त्वरित प्रयोग के लिए सुविधाजनक है

## मुझे ट्रांसफॉर्मर का उपयोग कब नहीं करना चाहिए?

- यह लाइब्रेरी मॉड्यूलर न्यूरल नेटवर्क टूलबॉक्स नहीं है। मॉडल फ़ाइल में कोड जानबूझकर अल्पविकसित है, बिना अतिरिक्त सार इनकैप्सुलेशन के, ताकि शोधकर्ता अमूर्तता और फ़ाइल जंपिंग में शामिल हुए जल्दी से पुनरावृति कर सकें।
- `ट्रेनर` एपीआई किसी भी मॉडल के साथ संगत नहीं है, यह केवल इस पुस्तकालय के मॉडल के लिए अनुकूलित है। यदि आप सामान्य मशीन लर्निंग के लिए उपयुक्त प्रशिक्षण लूप कार्यान्वयन की तलाश में हैं, तो कहीं और देखें।
- हमारे सर्वोत्तम प्रयासों के बावजूद, [उदाहरण निर्देशिका](https://github.com/huggingface/transformers/tree/main/examples) में स्क्रिप्ट केवल उपयोग के मामले हैं। आपकी विशिष्ट समस्या के लिए, वे जरूरी नहीं कि बॉक्स से बाहर काम करें, और आपको कोड की कुछ पंक्तियों को सूट करने की आवश्यकता हो सकती है।

## स्थापित करना

### पिप का उपयोग करना

इस रिपॉजिटरी का परीक्षण Python 3.10+ और PyTorch 2.4+ के तहत किया गया है।

आप [वर्चुअल एनवायरनमेंट](https://docs.python.org/3/library/venv.html) में 🤗 ट्रांसफॉर्मर इंस्टॉल कर सकते हैं। यदि आप अभी तक पायथन के वर्चुअल एनवायरनमेंट से परिचित नहीं हैं, तो कृपया इसे [उपयोगकर्ता निर्देश](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) पढ़ें।

सबसे पहले, पायथन के उस संस्करण के साथ एक आभासी वातावरण बनाएं जिसका आप उपयोग करने और उसे सक्रिय करने की योजना बना रहे हैं।

फिर, आपको Flax, PyTorch या TensorFlow में से किसी एक को स्थापित करने की आवश्यकता है। अपने प्लेटफ़ॉर्म पर इन फ़्रेमवर्क को स्थापित करने के लिए, [TensorFlow स्थापना पृष्ठ](https://www.tensorflow.org/install/), [PyTorch स्थापना पृष्ठ](https://pytorch.org/get-started/locally)

देखें start-locally या [Flax स्थापना पृष्ठ](https://github.com/google/flax#quick-install).

जब इनमें से कोई एक बैकएंड सफलतापूर्वक स्थापित हो जाता है, तो ट्रांसफॉर्मर निम्नानुसार स्थापित किए जा सकते हैं:

```bash
pip install transformers
```

यदि आप उपयोग के मामलों को आज़माना चाहते हैं या आधिकारिक रिलीज़ से पहले नवीनतम इन-डेवलपमेंट कोड का उपयोग करना चाहते हैं, तो आपको [सोर्स से इंस्टॉल करना होगा](https://huggingface.co/docs/transformers/installation#installing-from-) स्रोत।

### कोंडा का उपयोग करना

ट्रांसफॉर्मर कोंडा के माध्यम से निम्नानुसार स्थापित किया जा सकता है:

```shell script
conda install conda-forge::transformers
```

> **_नोट:_** `huggingface` चैनल से `transformers` इंस्टॉल करना पुराना पड़ चुका है।

कोंडा के माध्यम से Flax, PyTorch, या TensorFlow में से किसी एक को स्थापित करने के लिए, निर्देशों के लिए उनके संबंधित स्थापना पृष्ठ देखें।

## मॉडल आर्किटेक्चर
[उपयोगकर्ता](https://huggingface.co/users) और [organization](https://huggingface.co) द्वारा ट्रांसफॉर्मर समर्थित [**सभी मॉडल चौकियों**](https://huggingface.co/models/users) हगिंगफेस.को/ऑर्गनाइजेशन), सभी को बिना किसी बाधा के हगिंगफेस.को [मॉडल हब](https://huggingface.co) के साथ एकीकृत किया गया है।

चौकियों की वर्तमान संख्या: ![](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)

🤗 ट्रांसफॉर्मर वर्तमान में निम्नलिखित आर्किटेक्चर का समर्थन करते हैं: मॉडल के अवलोकन के लिए [यहां देखें](https://huggingface.co/docs/transformers/model_summary)：

यह जांचने के लिए कि क्या किसी मॉडल में पहले से ही Flax, PyTorch या TensorFlow का कार्यान्वयन है, या यदि उसके पास Tokenizers लाइब्रेरी में संबंधित टोकन है, तो [यह तालिका](https://huggingface.co/docs/transformers/index#supported) देखें। -फ्रेमवर्क)।

इन कार्यान्वयनों का परीक्षण कई डेटासेट पर किया गया है (देखें केस स्क्रिप्ट का उपयोग करें) और वैनिला कार्यान्वयन के लिए तुलनात्मक रूप से प्रदर्शन करना चाहिए। आप उपयोग के मामले के दस्तावेज़ [इस अनुभाग](https://huggingface.co/docs/transformers/examples) में व्यवहार का विवरण पढ़ सकते हैं।


## अधिक समझें

|अध्याय | विवरण |
|-|-|
| [दस्तावेज़ीकरण](https://huggingface.co/transformers/) | पूरा एपीआई दस्तावेज़ीकरण और ट्यूटोरियल |
| [कार्य सारांश](https://huggingface.co/docs/transformers/task_summary) | ट्रांसफॉर्मर समर्थित कार्य |
| [प्रीप्रोसेसिंग ट्यूटोरियल](https://huggingface.co/docs/transformers/preprocessing) | मॉडल के लिए डेटा तैयार करने के लिए `टोकनाइज़र` का उपयोग करना |
| [प्रशिक्षण और फाइन-ट्यूनिंग](https://huggingface.co/docs/transformers/training) | PyTorch/TensorFlow के ट्रेनिंग लूप या `ट्रेनर` API में ट्रांसफॉर्मर द्वारा दिए गए मॉडल का उपयोग करें |
| [क्विक स्टार्ट: ट्वीकिंग एंड यूज़ केस स्क्रिप्ट्स](https://github.com/huggingface/transformers/tree/main/examples) | विभिन्न कार्यों के लिए केस स्क्रिप्ट का उपयोग करें |
| [मॉडल साझा करना और अपलोड करना](https://huggingface.co/docs/transformers/model_sharing) | समुदाय के साथ अपने फाइन टूनड मॉडल अपलोड और साझा करें |
| [माइग्रेशन](https://huggingface.co/docs/transformers/migration) | `पाइटोरच-ट्रांसफॉर्मर्स` या `पाइटोरच-प्रीट्रेनड-बर्ट` से ट्रांसफॉर्मर में माइग्रेट करना |

## उद्धरण

हमने आधिकारिक तौर पर इस लाइब्रेरी का [पेपर](https://www.aclweb.org/anthology/2020.emnlp-demos.6/) प्रकाशित किया है, अगर आप ट्रान्सफ़ॉर्मर्स लाइब्रेरी का उपयोग करते हैं, तो कृपया उद्धृत करें:
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
