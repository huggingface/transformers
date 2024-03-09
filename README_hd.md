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
    <br>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers_logo_name.png" width="400"/>
    <br>
</p>
<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://huggingface.co/docs/transformers/index">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/transformers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <a href="https://github.com/huggingface/transformers/">English</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_zh-hant.md">繁體中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_es.md">Español</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_ja.md">日本語</a> |
        <b>हिन्दी</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_ru.md">Русский</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_pt-br.md">Рortuguês</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_vi.md">Tiếng Việt</a> |
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

इस रिपॉजिटरी का परीक्षण Python 3.8+, Flax 0.4.1+, PyTorch 1.11+ और TensorFlow 2.6+ के तहत किया गया है।

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

🤗 ट्रांसफॉर्मर वर्तमान में निम्नलिखित आर्किटेक्चर का समर्थन करते हैं (मॉडल के अवलोकन के लिए [यहां देखें](https://huggingface.co/docs/transformers/model_summary))：

1. **[ALBERT](https://huggingface.co/docs/transformers/model_doc/albert)** (Google Research and the Toyota Technological Institute at Chicago) साथ थीसिस [ALBERT: A Lite BERT for Self-supervised भाषा प्रतिनिधित्व सीखना](https://arxiv.org/abs/1909.11942), झेंझोंग लैन, मिंगदा चेन, सेबेस्टियन गुडमैन, केविन गिम्पेल, पीयूष शर्मा, राडू सोरिकट
1. **[ALIGN](https://huggingface.co/docs/transformers/model_doc/align)** (Google Research से) Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig. द्वाराअनुसंधान पत्र [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918) के साथ जारी किया गया
1. **[AltCLIP](https://huggingface.co/docs/transformers/model_doc/altclip)** (from BAAI) released with the paper [AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities](https://arxiv.org/abs/2211.06679) by Chen, Zhongzhi and Liu, Guang and Zhang, Bo-Wen and Ye, Fulong and Yang, Qinghong and Wu, Ledell.
1. **[Audio Spectrogram Transformer](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer)** (from MIT) released with the paper [AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) by Yuan Gong, Yu-An Chung, James Glass.
1. **[Autoformer](https://huggingface.co/docs/transformers/model_doc/autoformer)** (from Tsinghua University) released with the paper [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/abs/2106.13008) by Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long.
1. **[Bark](https://huggingface.co/docs/transformers/model_doc/bark)** (from Suno) released in the repository [suno-ai/bark](https://github.com/suno-ai/bark) by Suno AI team.
1. **[BART](https://huggingface.co/docs/transformers/model_doc/bart)** (फेसबुक) साथ थीसिस [बार्ट: प्राकृतिक भाषा निर्माण, अनुवाद के लिए अनुक्रम-से-अनुक्रम पूर्व प्रशिक्षण , और समझ](https://arxiv.org/pdf/1910.13461.pdf) पर निर्भर माइक लुईस, यिनहान लियू, नमन गोयल, मार्जन ग़ज़विनिनेजाद, अब्देलरहमान मोहम्मद, ओमर लेवी, वेस स्टोयानोव और ल्यूक ज़ेटलमॉयर
1. **[BARThez](https://huggingface.co/docs/transformers/model_doc/barthez)** (से École polytechnique) साथ थीसिस [BARThez: a Skilled Pretrained French Sequence-to-Sequence Model](https://arxiv.org/abs/2010.12321) पर निर्भर Moussa Kamal Eddine, Antoine J.-P. Tixier, Michalis Vazirgiannis रिहाई।
1. **[BARTpho](https://huggingface.co/docs/transformers/model_doc/bartpho)** (VinAI Research से) साथ में पेपर [BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese](https://arxiv.org/abs/2109.09701)गुयेन लुओंग ट्रान, डुओंग मिन्ह ले और डाट क्वोक गुयेन द्वारा पोस्ट किया गया।
1. **[BEiT](https://huggingface.co/docs/transformers/model_doc/beit)** (Microsoft से) साथ में कागज [BEiT: BERT इमेज ट्रांसफॉर्मर्स का प्री-ट्रेनिंग](https://arxiv.org/abs/2106.08254) Hangbo Bao, Li Dong, Furu Wei द्वारा।
1. **[BERT](https://huggingface.co/docs/transformers/model_doc/bert)** (गूगल से) साथ वाला पेपर [बीईआरटी: प्री-ट्रेनिंग ऑफ डीप बिडायरेक्शनल ट्रांसफॉर्मर्स फॉर लैंग्वेज अंडरस्टैंडिंग](https://arxiv.org/abs/1810.04805) जैकब डेवलिन, मिंग-वेई चांग, केंटन ली और क्रिस्टीना टौटानोवा द्वारा प्रकाशित किया गया था। .
1. **[BERT For Sequence Generation](https://huggingface.co/docs/transformers/model_doc/bert-generation)** (गूगल से) साथ देने वाला पेपर [सीक्वेंस जेनरेशन टास्क के लिए प्री-ट्रेंड चेकपॉइंट का इस्तेमाल करना](https://arxiv.org/abs/1907.12461) साशा रोठे, शशि नारायण, अलियाक्सि सेवेरिन द्वारा।
1. **[BERTweet](https://huggingface.co/docs/transformers/model_doc/bertweet)** (VinAI Research से) साथ में पेपर [BERTweet: अंग्रेजी ट्वीट्स के लिए एक पूर्व-प्रशिक्षित भाषा मॉडल](https://aclanthology.org/2020.emnlp-demos.2/) डाट क्वोक गुयेन, थान वु और अन्ह तुआन गुयेन द्वारा प्रकाशित।
1. **[BigBird-Pegasus](https://huggingface.co/docs/transformers/model_doc/bigbird_pegasus)** (गूगल रिसर्च से) साथ वाला पेपर [बिग बर्ड: ट्रांसफॉर्मर्स फॉर लॉन्गर सीक्वेंस](https://arxiv.org/abs/2007.14062) मंज़िल ज़हीर, गुरु गुरुगणेश, अविनावा दुबे, जोशुआ आइंस्ली, क्रिस अल्बर्टी, सैंटियागो ओंटानोन, फिलिप फाम, अनिरुद्ध रावुला, किफ़ान वांग, ली यांग, अमर अहमद द्वारा।
1. **[BigBird-RoBERTa](https://huggingface.co/docs/transformers/model_doc/big_bird)** (गूगल रिसर्च से) साथ में पेपर [बिग बर्ड: ट्रांसफॉर्मर्स फॉर लॉन्गर सीक्वेंस](https://arxiv.org/abs/2007.14062) मंज़िल ज़हीर, गुरु गुरुगणेश, अविनावा दुबे, जोशुआ आइंस्ली, क्रिस अल्बर्टी, सैंटियागो ओंटानन, फिलिप फाम द्वारा , अनिरुद्ध रावुला, किफ़ान वांग, ली यांग, अमर अहमद द्वारा पोस्ट किया गया।
1. **[BioGpt](https://huggingface.co/docs/transformers/model_doc/biogpt)** (from Microsoft Research AI4Science) released with the paper [BioGPT: generative pre-trained transformer for biomedical text generation and mining](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbac409/6713511?guestAccessKey=a66d9b5d-4f83-4017-bb52-405815c907b9) by Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon and Tie-Yan Liu.
1. **[BiT](https://huggingface.co/docs/transformers/model_doc/bit)** (from Google AI) released with the paper [Big Transfer (BiT) by Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, Neil Houlsby.
1. **[Blenderbot](https://huggingface.co/docs/transformers/model_doc/blenderbot)** (फेसबुक से) साथ में कागज [एक ओपन-डोमेन चैटबॉट बनाने की विधि](https://arxiv.org/abs/2004.13637) स्टीफन रोलर, एमिली दीनन, नमन गोयल, दा जू, मैरी विलियमसन, यिनहान लियू, जिंग जू, मायल ओट, कर्ट शस्टर, एरिक एम। स्मिथ, वाई-लैन बॉरो, जेसन वेस्टन द्वारा।
1. **[BlenderbotSmall](https://huggingface.co/docs/transformers/model_doc/blenderbot-small)** (फेसबुक से) साथ में पेपर [एक ओपन-डोमेन चैटबॉट बनाने की रेसिपी](https://arxiv.org/abs/2004.13637) स्टीफन रोलर, एमिली दीनन, नमन गोयल, दा जू, मैरी विलियमसन, यिनहान लियू, जिंग जू, मायल ओट, कर्ट शस्टर, एरिक एम स्मिथ, वाई-लैन बॉरो, जेसन वेस्टन द्वारा।
1. **[BLIP](https://huggingface.co/docs/transformers/model_doc/blip)** (from Salesforce) released with the paper [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) by Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi.
1. **[BLIP-2](https://huggingface.co/docs/transformers/model_doc/blip-2)** (Salesforce से) Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi. द्वाराअनुसंधान पत्र [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) के साथ जारी किया गया
1. **[BLOOM](https://huggingface.co/docs/transformers/model_doc/bloom)** (from BigScience workshop) released by the [BigSicence Workshop](https://bigscience.huggingface.co/).
1. **[BORT](https://huggingface.co/docs/transformers/model_doc/bort)** (एलेक्सा से) कागज के साथ [बीईआरटी के लिए ऑप्टिमल सबआर्किटेक्चर एक्सट्रैक्शन](https://arxiv.org/abs/2010.10499) एड्रियन डी विंटर और डैनियल जे पेरी द्वारा।
1. **[BridgeTower](https://huggingface.co/docs/transformers/model_doc/bridgetower)** (हरबिन इंस्टिट्यूट ऑफ़ टेक्नोलॉजी/माइक्रोसॉफ्ट रिसर्च एशिया/इंटेल लैब्स से) कागज के साथ [ब्रिजटॉवर: विजन-लैंग्वेज रिप्रेजेंटेशन लर्निंग में एनकोडर्स के बीच ब्रिज बनाना](<https://arxiv.org/abs/2206.08657>) by Xiao Xu, Chenfei Wu, Shachar Rosenman, Vasudev Lal, Wanxiang Che, Nan Duan.
1. **[BROS](https://huggingface.co/docs/transformers/model_doc/bros)** (NAVER CLOVA से) Teakgyu Hong, Donghyun Kim, Mingi Ji, Wonseok Hwang, Daehyun Nam, Sungrae Park. द्वाराअनुसंधान पत्र [BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents](https://arxiv.org/abs/2108.04539) के साथ जारी किया गया
1. **[ByT5](https://huggingface.co/docs/transformers/model_doc/byt5)** (Google अनुसंधान से) साथ में कागज [ByT5: पूर्व-प्रशिक्षित बाइट-टू-बाइट मॉडल के साथ एक टोकन-मुक्त भविष्य की ओर](https://arxiv.org/abs/2105.13626) Linting Xue, Aditya Barua, Noah Constant, रामी अल-रफू, शरण नारंग, मिहिर काले, एडम रॉबर्ट्स, कॉलिन रैफेल द्वारा पोस्ट किया गया।
1. **[CamemBERT](https://huggingface.co/docs/transformers/model_doc/camembert)** (इनरिया/फेसबुक/सोरबोन से) साथ में कागज [CamemBERT: एक टेस्टी फ्रेंच लैंग्वेज मॉडल](https://arxiv.org/abs/1911.03894) लुई मार्टिन*, बेंजामिन मुलर*, पेड्रो जेवियर ऑर्टिज़ सुआरेज़*, योआन ड्यूपॉन्ट, लॉरेंट रोमरी, एरिक विलेमोन्टे डे ला क्लर्जरी, जैमे सेडाह और बेनोइट सगोट द्वारा।
1. **[CANINE](https://huggingface.co/docs/transformers/model_doc/canine)** (Google रिसर्च से) साथ में दिया गया पेपर [कैनाइन: प्री-ट्रेनिंग ए एफिशिएंट टोकनाइजेशन-फ्री एनकोडर फॉर लैंग्वेज रिप्रेजेंटेशन](https://arxiv.org/abs/2103.06874) जोनाथन एच क्लार्क, डैन गैरेट, यूलिया टर्क, जॉन विएटिंग द्वारा।
1. **[Chinese-CLIP](https://huggingface.co/docs/transformers/model_doc/chinese_clip)** (from OFA-Sys) released with the paper [Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese](https://arxiv.org/abs/2211.01335) by An Yang, Junshu Pan, Junyang Lin, Rui Men, Yichang Zhang, Jingren Zhou, Chang Zhou.
1. **[CLAP](https://huggingface.co/docs/transformers/model_doc/clap)** (LAION-AI से) Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, Shlomo Dubnov. द्वाराअनुसंधान पत्र [Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation](https://arxiv.org/abs/2211.06687) के साथ जारी किया गया
1. **[CLIP](https://huggingface.co/docs/transformers/model_doc/clip)** (OpenAI से) साथ वाला पेपर [लर्निंग ट्रांसफरेबल विजुअल मॉडल फ्रॉम नेचुरल लैंग्वेज सुपरविजन](https://arxiv.org/abs/2103.00020) एलेक रैडफोर्ड, जोंग वूक किम, क्रिस हैलासी, आदित्य रमेश, गेब्रियल गोह, संध्या अग्रवाल, गिरीश शास्त्री, अमांडा एस्केल, पामेला मिश्किन, जैक क्लार्क, ग्रेचेन क्रुएगर, इल्या सुत्स्केवर द्वारा।
1. **[CLIPSeg](https://huggingface.co/docs/transformers/model_doc/clipseg)** (from University of Göttingen) released with the paper [Image Segmentation Using Text and Image Prompts](https://arxiv.org/abs/2112.10003) by Timo Lüddecke and Alexander Ecker.
1. **[CLVP](https://huggingface.co/docs/transformers/model_doc/clvp)** released with the paper [Better speech synthesis through scaling](https://arxiv.org/abs/2305.07243) by James Betker.
1. **[CodeGen](https://huggingface.co/docs/transformers/model_doc/codegen)** (सेल्सफोर्स से) साथ में पेपर [प्रोग्राम सिंथेसिस के लिए एक संवादात्मक प्रतिमान](https://arxiv.org/abs/2203.13474) एरिक निजकैंप, बो पैंग, हिरोआकी हयाशी, लिफू तू, हुआन वांग, यिंगबो झोउ, सिल्वियो सावरेस, कैमिंग जिओंग रिलीज।
1. **[CodeLlama](https://huggingface.co/docs/transformers/model_doc/llama_code)** (MetaAI से) Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, Gabriel Synnaeve. द्वाराअनुसंधान पत्र [Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/) के साथ जारी किया गया
1. **[Conditional DETR](https://huggingface.co/docs/transformers/model_doc/conditional_detr)** (माइक्रोसॉफ्ट रिसर्च एशिया से) कागज के साथ [फास्ट ट्रेनिंग कन्वर्जेंस के लिए सशर्त डीईटीआर](https://arxiv.org/abs/2108.06152) डेपू मेंग, ज़ियाओकांग चेन, ज़ेजिया फैन, गैंग ज़ेंग, होउकियांग ली, युहुई युआन, लेई सन, जिंगडोंग वांग द्वारा।
1. **[ConvBERT](https://huggingface.co/docs/transformers/model_doc/convbert)** (YituTech से) साथ में कागज [ConvBERT: स्पैन-आधारित डायनेमिक कनवल्शन के साथ BERT में सुधार](https://arxiv.org/abs/2008.02496) जिहांग जियांग, वीहाओ यू, डाकान झोउ, युनपेंग चेन, जियाशी फेंग, शुइचेंग यान द्वारा।
1. **[ConvNeXT](https://huggingface.co/docs/transformers/model_doc/convnext)** (Facebook AI से) साथ वाला पेपर [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) ज़ुआंग लियू, हेंज़ी माओ, चाओ-युआन वू, क्रिस्टोफ़ फीचटेनहोफ़र, ट्रेवर डेरेल, सैनिंग ज़ी द्वारा।
1. **[ConvNeXTV2](https://huggingface.co/docs/transformers/model_doc/convnextv2)** (from Facebook AI) released with the paper [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808) by Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, Saining Xie.
1. **[CPM](https://huggingface.co/docs/transformers/model_doc/cpm)** (सिंघुआ यूनिवर्सिटी से) साथ में पेपर [सीपीएम: ए लार्ज-स्केल जेनेरेटिव चाइनीज प्री-ट्रेंड लैंग्वेज मॉडल](https://arxiv.org/abs/2012.00413) झेंग्यान झांग, जू हान, हाओ झोउ, पेई के, युक्सियन गु, डेमिंग ये, युजिया किन, युशेंग सु, हाओझे जी, जियान गुआन, फैंचाओ क्यूई, ज़ियाओझी वांग, यानान झेंग द्वारा , गुओयांग ज़ेंग, हुआनकी काओ, शेंगकी चेन, डाइक्सुआन ली, ज़ेनबो सन, ज़ियुआन लियू, मिनली हुआंग, वेंटाओ हान, जी तांग, जुआनज़ी ली, ज़ियाओयान झू, माओसोंग सन।
1. **[CPM-Ant](https://huggingface.co/docs/transformers/model_doc/cpmant)** (from OpenBMB) released by the [OpenBMB](https://www.openbmb.org/).
1. **[CTRL](https://huggingface.co/docs/transformers/model_doc/ctrl)** (सेल्सफोर्स से) साथ में पेपर [CTRL: ए कंडिशनल ट्रांसफॉर्मर लैंग्वेज मॉडल फॉर कंट्रोलेबल जेनरेशन](https://arxiv.org/abs/1909.05858) नीतीश शिरीष केसकर*, ब्रायन मैककैन*, लव आर. वार्ष्णेय, कैमिंग जिओंग और रिचर्ड द्वारा सोचर द्वारा जारी किया गया।
1. **[CvT](https://huggingface.co/docs/transformers/model_doc/cvt)** (Microsoft से) साथ में दिया गया पेपर [CvT: इंट्रोड्यूसिंग कनवॉल्यूशन टू विजन ट्रांसफॉर्मर्स](https://arxiv.org/एब्स/2103.15808) हैपिंग वू, बिन जिओ, नोएल कोडेला, मेंगचेन लियू, जियांग दाई, लू युआन, लेई झांग द्वारा।
1. **[Data2Vec](https://huggingface.co/docs/transformers/model_doc/data2vec)** (फेसबुक से) साथ में कागज [Data2Vec: भाषण, दृष्टि और भाषा में स्व-पर्यवेक्षित सीखने के लिए एक सामान्य ढांचा](https://arxiv.org/abs/2202.03555) एलेक्सी बाएव्स्की, वेई-निंग सू, कियानटोंग जू, अरुण बाबू, जियाताओ गु, माइकल औली द्वारा पोस्ट किया गया।
1. **[DeBERTa](https://huggingface.co/docs/transformers/model_doc/deberta)** (Microsoft से) साथ में दिया गया पेपर [DeBERta: डिकोडिंग-एन्हांस्ड BERT विद डिसेंटैंगल्ड अटेंशन](https://arxiv.org/abs/2006.03654) पेंगचेंग हे, ज़ियाओडोंग लियू, जियानफेंग गाओ, वीज़ू चेन द्वारा।
1. **[DeBERTa-v2](https://huggingface.co/docs/transformers/model_doc/deberta-v2)** (Microsoft से) साथ में दिया गया पेपर [DeBERTa: डिकोडिंग-एन्हांस्ड BERT विथ डिसेंन्गल्ड अटेंशन](https://arxiv.org/abs/2006.03654) पेंगचेंग हे, ज़ियाओडोंग लियू, जियानफेंग गाओ, वीज़ू चेन द्वारा पोस्ट किया गया।
1. **[Decision Transformer](https://huggingface.co/docs/transformers/model_doc/decision_transformer)** (बर्कले/फेसबुक/गूगल से) पेपर के साथ [डिसीजन ट्रांसफॉर्मर: रीनफोर्समेंट लर्निंग वाया सीक्वेंस मॉडलिंग](https://arxiv.org/abs/2106.01345) लिली चेन, केविन लू, अरविंद राजेश्वरन, किमिन ली, आदित्य ग्रोवर, माइकल लास्किन, पीटर एबील, अरविंद श्रीनिवास, इगोर मोर्डच द्वारा पोस्ट किया गया।
1. **[Deformable DETR](https://huggingface.co/docs/transformers/model_doc/deformable_detr)** (सेंसटाइम रिसर्च से) साथ में पेपर [डिफॉर्मेबल डीईटीआर: डिफॉर्मेबल ट्रांसफॉर्मर्स फॉर एंड-टू-एंड ऑब्जेक्ट डिटेक्शन](https://arxiv.org/abs/2010.04159) Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, जिफेंग दाई द्वारा पोस्ट किया गया।
1. **[DeiT](https://huggingface.co/docs/transformers/model_doc/deit)** (फेसबुक से) साथ में पेपर [ट्रेनिंग डेटा-एफिशिएंट इमेज ट्रांसफॉर्मर और डिस्टिलेशन थ्रू अटेंशन](https://arxiv.org/abs/2012.12877) ह्यूगो टौव्रोन, मैथ्यू कॉर्ड, मैथिज्स डूज़, फ़्रांसिस्को मस्सा, एलेक्ज़ेंडर सबलेरोल्स, हर्वे जेगौ द्वारा।
1. **[DePlot](https://huggingface.co/docs/transformers/model_doc/deplot)** (Google AI से) Fangyu Liu, Julian Martin Eisenschlos, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Wenhu Chen, Nigel Collier, Yasemin Altun. द्वाराअनुसंधान पत्र [DePlot: One-shot visual language reasoning by plot-to-table translation](https://arxiv.org/abs/2212.10505) के साथ जारी किया गया
1. **[Depth Anything](https://huggingface.co/docs/transformers/model_doc/depth_anything)** (University of Hong Kong and TikTok से) Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao. द्वाराअनुसंधान पत्र [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/abs/2401.10891) के साथ जारी किया गया
1. **[DETA](https://huggingface.co/docs/transformers/model_doc/deta)** (from The University of Texas at Austin) released with the paper [NMS Strikes Back](https://arxiv.org/abs/2212.06137) by Jeffrey Ouyang-Zhang, Jang Hyun Cho, Xingyi Zhou, Philipp Krähenbühl.
1. **[DETR](https://huggingface.co/docs/transformers/model_doc/detr)** (फेसबुक से) साथ में कागज [ट्रांसफॉर्मर्स के साथ एंड-टू-एंड ऑब्जेक्ट डिटेक्शन](https://arxiv.org/abs/2005.12872) निकोलस कैरियन, फ़्रांसिस्को मस्सा, गेब्रियल सिनेव, निकोलस उसुनियर, अलेक्जेंडर किरिलोव, सर्गेई ज़ागोरुयको द्वारा।
1. **[DialoGPT](https://huggingface.co/docs/transformers/model_doc/dialogpt)** (माइक्रोसॉफ्ट रिसर्च से) कागज के साथ [DialoGPT: बड़े पैमाने पर जनरेटिव प्री-ट्रेनिंग फॉर कन्वर्सेशनल रिस्पांस जेनरेशन](https://arxiv.org/abs/1911.00536) यिज़े झांग, सिकी सन, मिशेल गैली, येन-चुन चेन, क्रिस ब्रोकेट, जियांग गाओ, जियानफेंग गाओ, जिंगजिंग लियू, बिल डोलन द्वारा।
1. **[DiNAT](https://huggingface.co/docs/transformers/model_doc/dinat)** (from SHI Labs) released with the paper [Dilated Neighborhood Attention Transformer](https://arxiv.org/abs/2209.15001) by Ali Hassani and Humphrey Shi.
1. **[DINOv2](https://huggingface.co/docs/transformers/model_doc/dinov2)** (Meta AI से) Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski. द्वाराअनुसंधान पत्र [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) के साथ जारी किया गया
1. **[DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)** (हगिंगफेस से), साथ में कागज [डिस्टिलबर्ट, बीईआरटी का डिस्टिल्ड वर्जन: छोटा, तेज, सस्ता और हल्का](https://arxiv.org/abs/1910.01108) विक्टर सनह, लिसांड्रे डेब्यू और थॉमस वुल्फ द्वारा पोस्ट किया गया। यही तरीका GPT-2 को [DistilGPT2](https://github.com/huggingface/transformers/tree/main/examples/distillation), RoBERta से [DistilRoBERta](https://github.com) पर कंप्रेस करने के लिए भी लागू किया जाता है। / हगिंगफेस/ट्रांसफॉर्मर्स/ट्री/मेन/उदाहरण/डिस्टिलेशन), बहुभाषी BERT से [DistilmBERT](https://github.com/huggingface/transformers/tree/main/examples/distillation) और डिस्टिलबर्ट का जर्मन संस्करण।
1. **[DiT](https://huggingface.co/docs/transformers/model_doc/dit)** (माइक्रोसॉफ्ट रिसर्च से) साथ में पेपर [DiT: सेल्फ सुपरवाइज्ड प्री-ट्रेनिंग फॉर डॉक्यूमेंट इमेज ट्रांसफॉर्मर](https://arxiv.org/abs/2203.02378) जुनलॉन्ग ली, यिहेंग जू, टेंगचाओ लव, लेई कुई, चा झांग द्वारा फुरु वेई द्वारा पोस्ट किया गया।
1. **[Donut](https://huggingface.co/docs/transformers/model_doc/donut)** (NAVER से) साथ में कागज [OCR-मुक्त डॉक्यूमेंट अंडरस्टैंडिंग ट्रांसफॉर्मर](https://arxiv.org/abs/2111.15664) गीवूक किम, टीकग्यू होंग, मूनबिन यिम, जियोंग्योन नाम, जिनयॉन्ग पार्क, जिनयॉन्ग यिम, वोनसेओक ह्वांग, सांगडू यूं, डोंगयून हान, सेउंग्युन पार्क द्वारा।
1. **[DPR](https://huggingface.co/docs/transformers/model_doc/dpr)** (फेसबुक से) साथ में पेपर [ओपन-डोमेन क्वेश्चन आंसरिंग के लिए डेंस पैसेज रिट्रीवल](https://arxiv.org/abs/2004.04906) व्लादिमीर करपुखिन, बरलास ओज़ुज़, सेवन मिन, पैट्रिक लुईस, लेडेल वू, सर्गेई एडुनोव, डैनकी चेन, और वेन-ताऊ यिह द्वारा।
1. **[DPT](https://huggingface.co/docs/transformers/master/model_doc/dpt)** (इंटेल लैब्स से) साथ में कागज [विज़न ट्रांसफॉर्मर्स फॉर डेंस प्रेडिक्शन](https://arxiv.org/abs/2103.13413) रेने रैनफ्टल, एलेक्सी बोचकोवस्की, व्लादलेन कोल्टन द्वारा।
1. **[EfficientFormer](https://huggingface.co/docs/transformers/model_doc/efficientformer)** (from Snap Research) released with the paper [EfficientFormer: Vision Transformers at MobileNetSpeed](https://arxiv.org/abs/2206.01191) by Yanyu Li, Geng Yuan, Yang Wen, Ju Hu, Georgios Evangelidis, Sergey Tulyakov, Yanzhi Wang, Jian Ren.
1. **[EfficientNet](https://huggingface.co/docs/transformers/model_doc/efficientnet)** (from Google Brain) released with the paper [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) by Mingxing Tan, Quoc V. Le.
1. **[ELECTRA](https://huggingface.co/docs/transformers/model_doc/electra)** (Google रिसर्च/स्टैनफोर्ड यूनिवर्सिटी से) साथ में दिया गया पेपर [इलेक्ट्रा: जेनरेटर के बजाय भेदभाव करने वाले के रूप में टेक्स्ट एन्कोडर्स का पूर्व-प्रशिक्षण](https://arxiv.org/abs/2003.10555) केविन क्लार्क, मिन्ह-थांग लुओंग, क्वोक वी. ले, क्रिस्टोफर डी. मैनिंग द्वारा पोस्ट किया गया।
1. **[EnCodec](https://huggingface.co/docs/transformers/model_doc/encodec)** (Meta AI से) Alexandre Défossez, Jade Copet, Gabriel Synnaeve, Yossi Adi. द्वाराअनुसंधान पत्र [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) के साथ जारी किया गया
1. **[EncoderDecoder](https://huggingface.co/docs/transformers/model_doc/encoder-decoder)** (Google रिसर्च से) साथ में दिया गया पेपर [सीक्वेंस जेनरेशन टास्क के लिए प्री-ट्रेंड चेकपॉइंट का इस्तेमाल करना](https://arxiv.org/abs/1907.12461) साशा रोठे, शशि नारायण, अलियाक्सि सेवेरिन द्वारा।
1. **[ERNIE](https://huggingface.co/docs/transformers/model_doc/ernie)**(Baidu से) साथ देने वाला पेपर [ERNIE: एन्हांस्ड रिप्रेजेंटेशन थ्रू नॉलेज इंटीग्रेशन](https://arxiv.org/abs/1904.09223) यू सन, शुओहुआन वांग, युकुन ली, शिकुन फेंग, ज़ुई चेन, हान झांग, शिन तियान, डैनक्सियांग झू, हाओ तियान, हुआ वू द्वारा पोस्ट किया गया।
1. **[ErnieM](https://huggingface.co/docs/transformers/model_doc/ernie_m)** (Baidu से) Xuan Ouyang, Shuohuan Wang, Chao Pang, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang. द्वाराअनुसंधान पत्र [ERNIE-M: Enhanced Multilingual Representation by Aligning Cross-lingual Semantics with Monolingual Corpora](https://arxiv.org/abs/2012.15674) के साथ जारी किया गया
1. **[ESM](https://huggingface.co/docs/transformers/model_doc/esm)** (मेटा AI से) ट्रांसफॉर्मर प्रोटीन भाषा मॉडल हैं। **ESM-1b** पेपर के साथ जारी किया गया था [अलेक्जेंडर राइव्स, जोशुआ मेयर, टॉम सर्कु, सिद्धार्थ गोयल, ज़ेमिंग लिन द्वारा जैविक संरचना और कार्य असुरक्षित सीखने को 250 मिलियन प्रोटीन अनुक्रमों तक स्केल करने से उभरता है](https://www.pnas.org/content/118/15/e2016239118) जेसन लियू, डेमी गुओ, मायल ओट, सी. लॉरेंस ज़िटनिक, जेरी मा और रॉब फर्गस। **ESM-1v** को पेपर के साथ जारी किया गया था [भाषा मॉडल प्रोटीन फ़ंक्शन पर उत्परिवर्तन के प्रभावों की शून्य-शॉट भविष्यवाणी को सक्षम करते हैं](https://doi.org/10.1101/2021.07.09.450648) जोशुआ मेयर, रोशन राव, रॉबर्ट वेरकुइल, जेसन लियू, टॉम सर्कु और अलेक्जेंडर राइव्स द्वारा। **ESM-2** को पेपर के साथ जारी किया गया था [भाषा मॉडल विकास के पैमाने पर प्रोटीन अनुक्रम सटीक संरचना भविष्यवाणी को सक्षम करते हैं](https://doi.org/10.1101/2022.07.20.500902) ज़ेमिंग लिन, हलील अकिन, रोशन राव, ब्रायन ही, झोंगकाई झू, वेंटिंग लू, ए द्वारा लान डॉस सैंटोस कोस्टा, मरियम फ़ज़ल-ज़रंडी, टॉम सर्कू, साल कैंडिडो, अलेक्जेंडर राइव्स।
1. **[Falcon](https://huggingface.co/docs/transformers/model_doc/falcon)** (from Technology Innovation Institute) by Almazrouei, Ebtesam and Alobeidli, Hamza and Alshamsi, Abdulaziz and Cappelli, Alessandro and Cojocaru, Ruxandra and Debbah, Merouane and Goffinet, Etienne and Heslow, Daniel and Launay, Julien and Malartic, Quentin and Noune, Badreddine and Pannier, Baptiste and Penedo, Guilherme.
1. **[FastSpeech2Conformer](model_doc/fastspeech2_conformer)** (ESPnet and Microsoft Research से) Pengcheng Guo, Florian Boyer, Xuankai Chang, Tomoki Hayashi, Yosuke Higuchi, Hirofumi Inaguma, Naoyuki Kamo, Chenda Li, Daniel Garcia-Romero, Jiatong Shi, Jing Shi, Shinji Watanabe, Kun Wei, Wangyou Zhang, and Yuekai Zhang. द्वाराअनुसंधान पत्र [Fastspeech 2: Fast And High-quality End-to-End Text To Speech](https://arxiv.org/pdf/2006.04558.pdf) के साथ जारी किया गया
1. **[FLAN-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5)** (from Google AI) released in the repository [google-research/t5x](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints) by Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei
1. **[FLAN-UL2](https://huggingface.co/docs/transformers/model_doc/flan-ul2)** (from Google AI) released in the repository [google-research/t5x](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-ul2-checkpoints) by Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei
1. **[FlauBERT](https://huggingface.co/docs/transformers/model_doc/flaubert)** (CNRS से) साथ वाला पेपर [FlauBERT: Unsupervised Language Model Pre-training for फ़्रेंच](https://arxiv.org/abs/1912.05372) Hang Le, Loïc Vial, Jibril Frej, Vincent Segonne, Maximin Coavoux, बेंजामिन लेकोउटेक्स, अलेक्जेंड्रे अल्लाउज़ेन, बेनोइट क्रैबे, लॉरेंट बेसेसियर, डिडिएर श्वाब द्वारा।
1. **[FLAVA](https://huggingface.co/docs/transformers/model_doc/flava)** [FLAVA: A फाउंडेशनल लैंग्वेज एंड विजन अलाइनमेंट मॉडल](https://arxiv.org/abs/2112.04482) साथ वाला पेपर अमनप्रीत सिंह, रोंगहांग हू, वेदानुज गोस्वामी, गुइल्यूम कुएरॉन, वोज्शिएक गालुबा, मार्कस रोहरबैक, और डौवे कीला द्वारा।
1. **[FNet](https://huggingface.co/docs/transformers/model_doc/fnet)** (गूगल रिसर्च से) साथ वाला पेपर [FNet: मिक्सिंग टोकन विद फूरियर ट्रांसफॉर्म्स](https://arxiv.org/abs/2105.03824) जेम्स ली-थॉर्प, जोशुआ आइंस्ली, इल्या एकस्टीन, सैंटियागो ओंटानन द्वारा।
1. **[FocalNet](https://huggingface.co/docs/transformers/model_doc/focalnet)** (Microsoft Research से) Jianwei Yang, Chunyuan Li, Xiyang Dai, Lu Yuan, Jianfeng Gao. द्वाराअनुसंधान पत्र [Focal Modulation Networks](https://arxiv.org/abs/2203.11926) के साथ जारी किया गया
1. **[Funnel Transformer](https://huggingface.co/docs/transformers/model_doc/funnel)** (सीएमयू/गूगल ब्रेन से) साथ में कागज [फ़नल-ट्रांसफॉर्मर: कुशल भाषा प्रसंस्करण के लिए अनुक्रमिक अतिरेक को छानना](https://arxiv.org/abs/2006.03236) जिहांग दाई, गुओकुन लाई, यिमिंग यांग, क्वोक वी. ले द्वारा रिहाई।
1. **[Fuyu](https://huggingface.co/docs/transformers/model_doc/fuyu)** (ADEPT से) रोहन बाविशी, एरिच एलसेन, कर्टिस हॉथोर्न, मैक्सवेल नी, ऑगस्टस ओडेना, अरुशी सोमानी, सागनाक तासिरलार  [blog post](https://www.adept.ai/blog/fuyu-8b)
1. **[Gemma](https://huggingface.co/docs/transformers/main/model_doc/gemma)** (Google से) the Gemma Google team. द्वाराअनुसंधान पत्र [Gemma: Open Models Based on Gemini Technology and Research](https://blog.google/technology/developers/gemma-open-models/) के साथ जारी किया गया
1. **[GIT](https://huggingface.co/docs/transformers/model_doc/git)** (from Microsoft Research) released with the paper [GIT: A Generative Image-to-text Transformer for Vision and Language](https://arxiv.org/abs/2205.14100) by Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, Lijuan Wang.
1. **[GLPN](https://huggingface.co/docs/transformers/model_doc/glpn)** (KAIST से) साथ वाला पेपर [वर्टिकल कटडेप्थ के साथ मोनोकुलर डेप्थ एस्टीमेशन के लिए ग्लोबल-लोकल पाथ नेटवर्क्स](https://arxiv.org/abs/2201.07436) डोयोन किम, वूंगह्युन गा, प्युंगवान आह, डोंगग्यू जू, सेहवान चुन, जुनमो किम द्वारा।
1. **[GPT](https://huggingface.co/docs/transformers/model_doc/openai-gpt)** (OpenAI से) साथ में दिया गया पेपर [जेनरेटिव प्री-ट्रेनिंग द्वारा भाषा की समझ में सुधार](https://openai.com/research/language-unsupervised/) एलेक रैडफोर्ड, कार्तिक नरसिम्हन, टिम सालिमन्स और इल्या सुत्स्केवर द्वारा।
1. **[GPT Neo](https://huggingface.co/docs/transformers/model_doc/gpt_neo)** (EleutherAI से) रिपॉजिटरी के साथ [EleutherAI/gpt-neo](https://github.com/EleutherAI/gpt-neo) रिलीज। सिड ब्लैक, स्टेला बिडरमैन, लियो गाओ, फिल वांग और कॉनर लेही द्वारा पोस्ट किया गया।
1. **[GPT NeoX](https://huggingface.co/docs/transformers/model_doc/gpt_neox)** (EleutherAI से) पेपर के साथ जारी किया गया [GPT-NeoX-20B: एक ओपन-सोर्स ऑटोरेग्रेसिव लैंग्वेज मॉडल](https://arxiv.org/abs/2204.06745) सिड ब्लैक, स्टेला बिडरमैन, एरिक हैलाहन, क्वेंटिन एंथोनी, लियो गाओ, लॉरेंस गोल्डिंग, होरेस हे, कॉनर लेही, काइल मैकडोनेल, जेसन फांग, माइकल पाइलर, यूएसवीएसएन साई प्रशांत द्वारा , शिवांशु पुरोहित, लारिया रेनॉल्ड्स, जोनाथन टो, बेन वांग, सैमुअल वेनबैक
1. **[GPT NeoX Japanese](https://huggingface.co/docs/transformers/model_doc/gpt_neox_japanese)** (अबेजा के जरिए) शिन्या ओटानी, ताकायोशी मकाबे, अनुज अरोड़ा, क्यो हटोरी द्वारा।
1. **[GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2)** (ओपनएआई से) साथ में पेपर [लैंग्वेज मॉडल्स अनसुपरवाइज्ड मल्टीटास्क लर्नर्स हैं](https://openai.com/research/better-language-models/) एलेक रैडफोर्ड, जेफरी वू, रेवन चाइल्ड, डेविड लुआन, डारियो एमोडी द्वारा और इल्या सुत्सकेवर ने पोस्ट किया।
1. **[GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj)** (EleutherAI से) साथ वाला पेपर [kingoflolz/mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax/) बेन वांग और अरन कोमात्सुजाकी द्वारा।
1. **[GPT-Sw3](https://huggingface.co/docs/transformers/model_doc/gpt-sw3)** (from AI-Sweden) released with the paper [Lessons Learned from GPT-SW3: Building the First Large-Scale Generative Language Model for Swedish](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.376.pdf) by Ariel Ekgren, Amaru Cuba Gyllensten, Evangelia Gogoulou, Alice Heiman, Severine Verlinden, Joey Öhman, Fredrik Carlsson, Magnus Sahlgren.
1. **[GPTBigCode](https://huggingface.co/docs/transformers/model_doc/gpt_bigcode)** (BigCode से) Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, Logesh Kumar Umapathi, Carolyn Jane Anderson, Yangtian Zi, Joel Lamy Poirier, Hailey Schoelkopf, Sergey Troshin, Dmitry Abulkhanov, Manuel Romero, Michael Lappert, Francesco De Toni, Bernardo García del Río, Qian Liu, Shamik Bose, Urvashi Bhattacharyya, Terry Yue Zhuo, Ian Yu, Paulo Villegas, Marco Zocca, Sourab Mangrulkar, David Lansky, Huu Nguyen, Danish Contractor, Luis Villa, Jia Li, Dzmitry Bahdanau, Yacine Jernite, Sean Hughes, Daniel Fried, Arjun Guha, Harm de Vries, Leandro von Werra. द्वाराअनुसंधान पत्र [SantaCoder: don't reach for the stars!](https://arxiv.org/abs/2301.03988) के साथ जारी किया गया
1. **[GPTSAN-japanese](https://huggingface.co/docs/transformers/model_doc/gptsan-japanese)** released in the repository [tanreinama/GPTSAN](https://github.com/tanreinama/GPTSAN/blob/main/report/model.md) by Toshiyuki Sakamoto(tanreinama).
1. **[Graphormer](https://huggingface.co/docs/transformers/model_doc/graphormer)** (from Microsoft) released with the paper [Do Transformers Really Perform Bad for Graph Representation?](https://arxiv.org/abs/2106.05234) by Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, Tie-Yan Liu.
1. **[GroupViT](https://huggingface.co/docs/transformers/model_doc/groupvit)** (UCSD, NVIDIA से) साथ में कागज [GroupViT: टेक्स्ट सुपरविजन से सिमेंटिक सेगमेंटेशन इमर्जेस](https://arxiv.org/abs/2202.11094) जियारुई जू, शालिनी डी मेलो, सिफ़ी लियू, वोनमिन बायन, थॉमस ब्रेउएल, जान कौट्ज़, ज़ियाओलोंग वांग द्वारा।
1. **[HerBERT](https://huggingface.co/docs/transformers/model_doc/herbert)** (Allegro.pl, AGH University of Science and Technology से) Piotr Rybak, Robert Mroczkowski, Janusz Tracz, Ireneusz Gawlik. द्वाराअनुसंधान पत्र [KLEJ: Comprehensive Benchmark for Polish Language Understanding](https://www.aclweb.org/anthology/2020.acl-main.111.pdf) के साथ जारी किया गया
1. **[Hubert](https://huggingface.co/docs/transformers/model_doc/hubert)** (फेसबुक से) साथ में पेपर [ह्यूबर्ट: सेल्फ सुपरवाइज्ड स्पीच रिप्रेजेंटेशन लर्निंग बाय मास्क्ड प्रेडिक्शन ऑफ हिडन यूनिट्स](https://arxiv.org/abs/2106.07447) वेई-निंग सू, बेंजामिन बोल्टे, याओ-हंग ह्यूबर्ट त्साई, कुशाल लखोटिया, रुस्लान सालाखुतदीनोव, अब्देलरहमान मोहम्मद द्वारा।
1. **[I-BERT](https://huggingface.co/docs/transformers/model_doc/ibert)** (बर्कले से) साथ में कागज [I-BERT: Integer-only BERT Quantization](https://arxiv.org/abs/2101.01321) सेहून किम, अमीर घोलमी, ज़ेवेई याओ, माइकल डब्ल्यू महोनी, कर्ट केटज़र द्वारा।
1. **[IDEFICS](https://huggingface.co/docs/transformers/model_doc/idefics)** (from HuggingFace) released with the paper [OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents](https://huggingface.co/papers/2306.16527) by Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, Victor Sanh.
1. **[ImageGPT](https://huggingface.co/docs/transformers/model_doc/imagegpt)** (from OpenAI) released with the paper [Generative Pretraining from Pixels](https://openai.com/blog/image-gpt/) by Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, Ilya Sutskever.
1. **[Informer](https://huggingface.co/docs/transformers/model_doc/informer)** (from Beihang University, UC Berkeley, Rutgers University, SEDD Company) released with the paper [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436) by Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang.
1. **[InstructBLIP](https://huggingface.co/docs/transformers/model_doc/instructblip)** (Salesforce से) Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, Steven Hoi. द्वाराअनुसंधान पत्र [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://arxiv.org/abs/2305.06500) के साथ जारी किया गया
1. **[Jukebox](https://huggingface.co/docs/transformers/model_doc/jukebox)** (from OpenAI) released with the paper [Jukebox: A Generative Model for Music](https://arxiv.org/pdf/2005.00341.pdf) by Prafulla Dhariwal, Heewoo Jun, Christine Payne, Jong Wook Kim, Alec Radford, Ilya Sutskever.
1. **[KOSMOS-2](https://huggingface.co/docs/transformers/model_doc/kosmos-2)** (from Microsoft Research Asia) released with the paper [Kosmos-2: Grounding Multimodal Large Language Models to the World](https://arxiv.org/abs/2306.14824) by Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, Furu Wei.
1. **[LayoutLM](https://huggingface.co/docs/transformers/model_doc/layoutlm)** (from Microsoft Research Asia) released with the paper [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318) by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou.
1. **[LayoutLMv2](https://huggingface.co/docs/transformers/model_doc/layoutlmv2)** (from Microsoft Research Asia) released with the paper [LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding](https://arxiv.org/abs/2012.14740) by Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou.
1. **[LayoutLMv3](https://huggingface.co/docs/transformers/model_doc/layoutlmv3)** (माइक्रोसॉफ्ट रिसर्च एशिया से) साथ देने वाला पेपर [लेआउटएलएमवी3: यूनिफाइड टेक्स्ट और इमेज मास्किंग के साथ दस्तावेज़ एआई के लिए पूर्व-प्रशिक्षण](https://arxiv.org/abs/2204.08387) युपन हुआंग, टेंगचाओ लव, लेई कुई, युटोंग लू, फुरु वेई द्वारा पोस्ट किया गया।
1. **[LayoutXLM](https://huggingface.co/docs/transformers/model_doc/layoutxlm)** (from Microsoft Research Asia) released with the paper [LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://arxiv.org/abs/2104.08836) by Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Furu Wei.
1. **[LED](https://huggingface.co/docs/transformers/model_doc/led)** (from AllenAI) released with the paper [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) by Iz Beltagy, Matthew E. Peters, Arman Cohan.
1. **[LeViT](https://huggingface.co/docs/transformers/model_doc/levit)** (मेटा AI से) साथ वाला पेपर [LeViT: A Vision Transformer in ConvNet's Clothing for Faster Inference](https://arxiv.org/abs/2104.01136) बेन ग्राहम, अलाएल्डिन एल-नौबी, ह्यूगो टौवरन, पियरे स्टॉक, आर्मंड जौलिन, हर्वे जेगौ, मैथिज डूज़ द्वारा।
1. **[LiLT](https://huggingface.co/docs/transformers/model_doc/lilt)** (दक्षिण चीन प्रौद्योगिकी विश्वविद्यालय से) साथ में कागज [LiLT: एक सरल लेकिन प्रभावी भाषा-स्वतंत्र लेआउट ट्रांसफार्मर संरचित दस्तावेज़ समझ के लिए](https://arxiv.org/abs/2202.13669) जियापेंग वांग, लियानवेन जिन, काई डिंग द्वारा पोस्ट किया गया।
1. **[LLaMA](https://huggingface.co/docs/transformers/model_doc/llama)** (The FAIR team of Meta AI से) Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample. द्वाराअनुसंधान पत्र [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) के साथ जारी किया गया
1. **[Llama2](https://huggingface.co/docs/transformers/model_doc/llama2)** (The FAIR team of Meta AI से) Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushka rMishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing EllenTan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, Thomas Scialom.. द्वाराअनुसंधान पत्र [Llama2: Open Foundation and Fine-Tuned Chat Models](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/XXX) के साथ जारी किया गया
1. **[LLaVa](https://huggingface.co/docs/transformers/model_doc/llava)** (Microsoft Research & University of Wisconsin-Madison से) Haotian Liu, Chunyuan Li, Yuheng Li and Yong Jae Lee. द्वाराअनुसंधान पत्र [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) के साथ जारी किया गया
1. **[Longformer](https://huggingface.co/docs/transformers/model_doc/longformer)** (from AllenAI) released with the paper [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) by Iz Beltagy, Matthew E. Peters, Arman Cohan.
1. **[LongT5](https://huggingface.co/docs/transformers/model_doc/longt5)** (मैंडी गुओ, जोशुआ आइंस्ली, डेविड यूथस, सैंटियागो ओंटानन, जियानमो नि, यूं-हुआन सुंग, यिनफेई यांग द्वारा पोस्ट किया गया।
1. **[LUKE](https://huggingface.co/docs/transformers/model_doc/luke)** (स्टूडियो औसिया से) साथ में पेपर [LUKE: डीप कॉन्टेक्स्टुअलाइज्ड एंटिटी रिप्रेजेंटेशन विद एंटिटी-अवेयर सेल्फ-अटेंशन](https://arxiv.org/abs/2010.01057) Ikuya Yamada, Akari Asai, Hiroyuki Shindo, Hideaki Takeda, Yuji Matsumoto द्वारा।
1. **[LXMERT](https://huggingface.co/docs/transformers/model_doc/lxmert)** (UNC चैपल हिल से) साथ में पेपर [LXMERT: ओपन-डोमेन क्वेश्चन के लिए ट्रांसफॉर्मर से क्रॉस-मोडलिटी एनकोडर रिप्रेजेंटेशन सीखना Answering](https://arxiv.org/abs/1908.07490) हाओ टैन और मोहित बंसल द्वारा।
1. **[M-CTC-T](https://huggingface.co/docs/transformers/model_doc/mctct)** (from Facebook) released with the paper [Pseudo-Labeling For Massively Multilingual Speech Recognition](https://arxiv.org/abs/2111.00161) by Loren Lugosch, Tatiana Likhomanenko, Gabriel Synnaeve, and Ronan Collobert.
1. **[M2M100](https://huggingface.co/docs/transformers/model_doc/m2m_100)** (फेसबुक से) साथ देने वाला पेपर [बियॉन्ड इंग्लिश-सेंट्रिक मल्टीलिंगुअल मशीन ट्रांसलेशन](https://arxiv.org/एब्स/2010.11125) एंजेला फैन, श्रुति भोसले, होल्गर श्वेन्क, झी मा, अहमद अल-किश्की, सिद्धार्थ गोयल, मनदीप बैनेस, ओनूर सेलेबी, गुइल्लाम वेन्जेक, विश्रव चौधरी, नमन गोयल, टॉम बर्च, विटाली लिपचिंस्की, सर्गेई एडुनोव, एडौर्ड द्वारा ग्रेव, माइकल औली, आर्मंड जौलिन द्वारा पोस्ट किया गया।
1. **[MADLAD-400](https://huggingface.co/docs/transformers/model_doc/madlad-400)** (from Google) released with the paper [MADLAD-400: A Multilingual And Document-Level Large Audited Dataset](https://arxiv.org/abs/2309.04662) by Sneha Kudugunta, Isaac Caswell, Biao Zhang, Xavier Garcia, Christopher A. Choquette-Choo, Katherine Lee, Derrick Xin, Aditya Kusupati, Romi Stella, Ankur Bapna, Orhan Firat.
1. **[Mamba](https://huggingface.co/docs/transformers/main/model_doc/mamba)** (Albert Gu and Tri Dao से) Albert Gu and Tri Dao. द्वाराअनुसंधान पत्र [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) के साथ जारी किया गया
1. **[MarianMT](https://huggingface.co/docs/transformers/model_doc/marian)** Jörg द्वारा [OPUS](http://opus.nlpl.eu/) डेटा से प्रशिक्षित मशीनी अनुवाद मॉडल पोस्ट किया गया टाइडेमैन द्वारा। [मैरियन फ्रेमवर्क](https://marian-nmt.github.io/) माइक्रोसॉफ्ट ट्रांसलेटर टीम द्वारा विकसित।
1. **[MarkupLM](https://huggingface.co/docs/transformers/model_doc/markuplm)** (माइक्रोसॉफ्ट रिसर्च एशिया से) साथ में पेपर [मार्कअपएलएम: विजुअली-रिच डॉक्यूमेंट अंडरस्टैंडिंग के लिए टेक्स्ट और मार्कअप लैंग्वेज का प्री-ट्रेनिंग](https://arxiv.org/abs/2110.08518) जुनलॉन्ग ली, यिहेंग जू, लेई कुई, फुरु द्वारा वी द्वारा पोस्ट किया गया।
1. **[Mask2Former](https://huggingface.co/docs/transformers/model_doc/mask2former)** (FAIR and UIUC से) Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, Rohit Girdhar. द्वाराअनुसंधान पत्र [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527) के साथ जारी किया गया
1. **[MaskFormer](https://huggingface.co/docs/transformers/model_doc/maskformer)** (मेटा और UIUC से) पेपर के साथ जारी किया गया [प्रति-पिक्सेल वर्गीकरण वह सब नहीं है जिसकी आपको सिमेंटिक सेगमेंटेशन की आवश्यकता है](https://arxiv.org/abs/2107.06278) बोवेन चेंग, अलेक्जेंडर जी. श्विंग, अलेक्जेंडर किरिलोव द्वारा >>>>>> रिबेस ठीक करें
1. **[MatCha](https://huggingface.co/docs/transformers/model_doc/matcha)** (Google AI से) Fangyu Liu, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Yasemin Altun, Nigel Collier, Julian Martin Eisenschlos. द्वाराअनुसंधान पत्र [MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering](https://arxiv.org/abs/2212.09662) के साथ जारी किया गया
1. **[mBART](https://huggingface.co/docs/transformers/model_doc/mbart)** (फेसबुक से) साथ में पेपर [न्यूरल मशीन ट्रांसलेशन के लिए मल्टीलिंगुअल डीनोइजिंग प्री-ट्रेनिंग](https://arxiv.org/abs/2001.08210) यिनहान लियू, जियाताओ गु, नमन गोयल, जियान ली, सर्गेई एडुनोव, मार्जन ग़ज़विनिनेजाद, माइक लुईस, ल्यूक ज़ेटलमॉयर द्वारा।
1. **[mBART-50](https://huggingface.co/docs/transformers/model_doc/mbart)** (फेसबुक से) साथ में पेपर [एक्स्टेंसिबल बहुभाषी प्रीट्रेनिंग और फाइनट्यूनिंग के साथ बहुभाषी अनुवाद](https://arxiv.org/abs/2008.00401) युकिंग टैंग, चाउ ट्रान, जियान ली, पेंग-जेन चेन, नमन गोयल, विश्रव चौधरी, जियाताओ गु, एंजेला फैन द्वारा।
1. **[MEGA](https://huggingface.co/docs/transformers/model_doc/mega)** (Facebook से) Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig, Jonathan May, and Luke Zettlemoyer. द्वाराअनुसंधान पत्र [Mega: Moving Average Equipped Gated Attention](https://arxiv.org/abs/2209.10655) के साथ जारी किया गया
1. **[Megatron-BERT](https://huggingface.co/docs/transformers/model_doc/megatron-bert)** (NVIDIA से) कागज के साथ [Megatron-LM: मॉडल का उपयोग करके बहु-अरब पैरामीटर भाषा मॉडल का प्रशिक्षण Parallelism](https://arxiv.org/abs/1909.08053) मोहम्मद शोएबी, मोस्टोफा पटवारी, राउल पुरी, पैट्रिक लेग्रेस्ले, जेरेड कैस्पर और ब्रायन कैटानज़ारो द्वारा।
1. **[Megatron-GPT2](https://huggingface.co/docs/transformers/model_doc/megatron_gpt2)** (NVIDIA से) साथ वाला पेपर [Megatron-LM: ट्रेनिंग मल्टी-बिलियन पैरामीटर लैंग्वेज मॉडल्स यूजिंग मॉडल पैरेललिज़्म](https://arxiv.org/abs/1909.08053) मोहम्मद शोएबी, मोस्टोफा पटवारी, राउल पुरी, पैट्रिक लेग्रेस्ले, जेरेड कैस्पर और ब्रायन कैटानज़ारो द्वारा पोस्ट किया गया।
1. **[MGP-STR](https://huggingface.co/docs/transformers/model_doc/mgp-str)** (Alibaba Research से) Peng Wang, Cheng Da, and Cong Yao. द्वाराअनुसंधान पत्र [Multi-Granularity Prediction for Scene Text Recognition](https://arxiv.org/abs/2209.03592) के साथ जारी किया गया
1. **[Mistral](https://huggingface.co/docs/transformers/model_doc/mistral)** (from Mistral AI) by The Mistral AI team: Albert Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed..
1. **[Mixtral](https://huggingface.co/docs/transformers/model_doc/mixtral)** (from Mistral AI) by The [Mistral AI](https://mistral.ai) team: Albert Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed.
1. **[mLUKE](https://huggingface.co/docs/transformers/model_doc/mluke)** (फ्रॉम Studio Ousia) साथ में पेपर [mLUKE: द पावर ऑफ एंटिटी रिप्रेजेंटेशन इन मल्टीलिंगुअल प्रीट्रेन्ड लैंग्वेज मॉडल्स](https://arxiv.org/abs/2110.08151) रयोकन री, इकुया यामाडा, और योशिमासा त्सुरोका द्वारा।
1. **[MMS](https://huggingface.co/docs/transformers/model_doc/mms)** (Facebook से) Vineel Pratap, Andros Tjandra, Bowen Shi, Paden Tomasello, Arun Babu, Sayani Kundu, Ali Elkahky, Zhaoheng Ni, Apoorv Vyas, Maryam Fazel-Zarandi, Alexei Baevski, Yossi Adi, Xiaohui Zhang, Wei-Ning Hsu, Alexis Conneau, Michael Auli. द्वाराअनुसंधान पत्र [Scaling Speech Technology to 1,000+ Languages](https://arxiv.org/abs/2305.13516) के साथ जारी किया गया
1. **[MobileBERT](https://huggingface.co/docs/transformers/model_doc/mobilebert)** (सीएमयू/गूगल ब्रेन से) साथ में कागज [मोबाइलबर्ट: संसाधन-सीमित उपकरणों के लिए एक कॉम्पैक्ट टास्क-अज्ञेय बीईआरटी](https://arxiv.org/abs/2004.02984) Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, और Denny Zhou द्वारा पोस्ट किया गया।
1. **[MobileNetV1](https://huggingface.co/docs/transformers/model_doc/mobilenet_v1)** (from Google Inc.) released with the paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) by Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam.
1. **[MobileNetV2](https://huggingface.co/docs/transformers/model_doc/mobilenet_v2)** (from Google Inc.) released with the paper [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) by Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen.
1. **[MobileViT](https://huggingface.co/docs/transformers/model_doc/mobilevit)** (Apple से) साथ में कागज [MobileViT: लाइट-वेट, जनरल-पर्पस, और मोबाइल-फ्रेंडली विजन ट्रांसफॉर्मर](https://arxiv.org/abs/2110.02178) सचिन मेहता और मोहम्मद रस्तगरी द्वारा पोस्ट किया गया।
1. **[MobileViTV2](https://huggingface.co/docs/transformers/model_doc/mobilevitv2)** (Apple से) Sachin Mehta and Mohammad Rastegari. द्वाराअनुसंधान पत्र [Separable Self-attention for Mobile Vision Transformers](https://arxiv.org/abs/2206.02680) के साथ जारी किया गया
1. **[MPNet](https://huggingface.co/docs/transformers/model_doc/mpnet)** (from Microsoft Research) released with the paper [MPNet: Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/abs/2004.09297) by Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu.
1. **[MPT](https://huggingface.co/docs/transformers/model_doc/mpt)** (MosaiML से) the MosaicML NLP Team. द्वाराअनुसंधान पत्र [llm-foundry](https://github.com/mosaicml/llm-foundry/) के साथ जारी किया गया
1. **[MRA](https://huggingface.co/docs/transformers/model_doc/mra)** (the University of Wisconsin - Madison से) Zhanpeng Zeng, Sourav Pal, Jeffery Kline, Glenn M Fung, Vikas Singh. द्वाराअनुसंधान पत्र [Multi Resolution Analysis (MRA)](https://arxiv.org/abs/2207.10284) के साथ जारी किया गया
1. **[MT5](https://huggingface.co/docs/transformers/model_doc/mt5)** (Google AI से) साथ वाला पेपर [mT5: एक व्यापक बहुभाषी पूर्व-प्रशिक्षित टेक्स्ट-टू-टेक्स्ट ट्रांसफॉर्मर](https://arxiv.org/abs/2010.11934) लिंटिंग ज़ू, नोआ कॉन्सटेंट, एडम रॉबर्ट्स, मिहिर काले, रामी अल-रफू, आदित्य सिद्धांत, आदित्य बरुआ, कॉलिन रैफेल द्वारा पोस्ट किया गया।
1. **[MusicGen](https://huggingface.co/docs/transformers/model_doc/musicgen)** (from Meta) released with the paper [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) by Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi and Alexandre Défossez.
1. **[MVP](https://huggingface.co/docs/transformers/model_doc/mvp)** (from RUC AI Box) released with the paper [MVP: Multi-task Supervised Pre-training for Natural Language Generation](https://arxiv.org/abs/2206.12131) by Tianyi Tang, Junyi Li, Wayne Xin Zhao and Ji-Rong Wen.
1. **[NarrowBert](https://huggingface.co/docs/transformers/main/model_doc/narrow_bert)** (from University of Washington) released with the paper [NarrowBERT: Accelerating Masked Language Model Pretraining and Inference](https://arxiv.org/abs/2301.04761) by Haoxin Li, Phillip Keung, Daniel Cheng, Jungo Kasai, and Noah A. Smith.
1. **[NAT](https://huggingface.co/docs/transformers/model_doc/nat)** (from SHI Labs) released with the paper [Neighborhood Attention Transformer](https://arxiv.org/abs/2204.07143) by Ali Hassani, Steven Walton, Jiachen Li, Shen Li, and Humphrey Shi.
1. **[Nezha](https://huggingface.co/docs/transformers/model_doc/nezha)** (हुआवेई नूह के आर्क लैब से) साथ में कागज़ [NEZHA: चीनी भाषा समझ के लिए तंत्रिका प्रासंगिक प्रतिनिधित्व](https://arxiv.org/abs/1909.00204) जुन्किउ वेई, ज़ियाओज़े रेन, ज़िआओगुआंग ली, वेनयोंग हुआंग, यी लियाओ, याशेंग वांग, जियाशू लिन, शिन जियांग, जिओ चेन और कुन लियू द्वारा।
1. **[NLLB](https://huggingface.co/docs/transformers/model_doc/nllb)** (फ्रॉम मेटा) साथ में पेपर [नो लैंग्वेज लेफ्ट बिहाइंड: स्केलिंग ह्यूमन-सेंटेड मशीन ट्रांसलेशन](https://arxiv.org/abs/2207.04672) एनएलएलबी टीम द्वारा प्रकाशित।
1. **[NLLB-MOE](https://huggingface.co/docs/transformers/model_doc/nllb-moe)** (Meta से) the NLLB team. द्वाराअनुसंधान पत्र [No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) के साथ जारी किया गया
1. **[Nougat](https://huggingface.co/docs/transformers/model_doc/nougat)** (Meta AI से) Lukas Blecher, Guillem Cucurull, Thomas Scialom, Robert Stojnic. द्वाराअनुसंधान पत्र [Nougat: Neural Optical Understanding for Academic Documents](https://arxiv.org/abs/2308.13418) के साथ जारी किया गया
1. **[Nyströmformer](https://huggingface.co/docs/transformers/model_doc/nystromformer)** (विस्कॉन्सिन विश्वविद्यालय - मैडिसन से) साथ में कागज [Nyströmformer: A Nyström- आधारित एल्गोरिथम आत्म-ध्यान का अनुमान लगाने के लिए](https://arxiv.org/abs/2102.03902) युनयांग ज़िओंग, झानपेंग ज़ेंग, रुद्रसिस चक्रवर्ती, मिंगक्सिंग टैन, ग्लेन फंग, यिन ली, विकास सिंह द्वारा पोस्ट किया गया।
1. **[OneFormer](https://huggingface.co/docs/transformers/model_doc/oneformer)** (SHI Labs से) पेपर [OneFormer: One Transformer to Rule Universal Image Segmentation](https://arxiv.org/abs/2211.06220) जितेश जैन, जिआचेन ली, मांगटिक चिउ, अली हसनी, निकिता ओरलोव, हम्फ्री शि के द्वारा जारी किया गया है।
1. **[OpenLlama](https://huggingface.co/docs/transformers/model_doc/open-llama)** (from [s-JoL](https://huggingface.co/s-JoL)) released on GitHub (now removed).
1. **[OPT](https://huggingface.co/docs/transformers/master/model_doc/opt)** (from Meta AI) released with the paper [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068) by Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen et al.
1. **[OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit)** (Google AI से) साथ में कागज [विज़न ट्रांसफॉर्मर्स के साथ सिंपल ओपन-वोकैबुलरी ऑब्जेक्ट डिटेक्शन](https://arxiv.org/abs/2205.06230) मैथियास मिंडरर, एलेक्सी ग्रिट्सेंको, ऑस्टिन स्टोन, मैक्सिम न्यूमैन, डिर्क वीसेनबोर्न, एलेक्सी डोसोवित्स्की, अरविंद महेंद्रन, अनुराग अर्नब, मुस्तफा देहघानी, ज़ुओरन शेन, जिओ वांग, ज़ियाओहुआ झाई, थॉमस किफ़, और नील हॉल्सबी द्वारा पोस्ट किया गया।
1. **[OWLv2](https://huggingface.co/docs/transformers/model_doc/owlv2)** (Google AI से) Matthias Minderer, Alexey Gritsenko, Neil Houlsby. द्वाराअनुसंधान पत्र [Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683) के साथ जारी किया गया
1. **[PatchTSMixer](https://huggingface.co/docs/transformers/model_doc/patchtsmixer)** ( IBM Research से) Vijay Ekambaram, Arindam Jati, Nam Nguyen, Phanwadee Sinthong, Jayant Kalagnanam. द्वाराअनुसंधान पत्र [TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting](https://arxiv.org/pdf/2306.09364.pdf) के साथ जारी किया गया
1. **[PatchTST](https://huggingface.co/docs/transformers/model_doc/patchtst)** (IBM से) Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam. द्वाराअनुसंधान पत्र [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/pdf/2211.14730.pdf) के साथ जारी किया गया
1. **[Pegasus](https://huggingface.co/docs/transformers/model_doc/pegasus)** (from Google) released with the paper [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777) by Jingqing Zhang, Yao Zhao, Mohammad Saleh and Peter J. Liu.
1. **[PEGASUS-X](https://huggingface.co/docs/transformers/model_doc/pegasus_x)** (Google की ओर से) साथ में दिया गया पेपर [लंबे इनपुट सारांश के लिए ट्रांसफ़ॉर्मरों को बेहतर तरीके से एक्सटेंड करना](https://arxiv.org/abs/2208.04347) जेसन फांग, याओ झाओ, पीटर जे लियू द्वारा।
1. **[Perceiver IO](https://huggingface.co/docs/transformers/model_doc/perceiver)** (दीपमाइंड से) साथ में पेपर [पर्सीवर आईओ: संरचित इनपुट और आउटपुट के लिए एक सामान्य वास्तुकला](https://arxiv.org/abs/2107.14795) एंड्रयू जेगल, सेबेस्टियन बोरग्यूड, जीन-बैप्टिस्ट अलायराक, कार्ल डोर्श, कैटलिन इओनेस्कु, डेविड द्वारा डिंग, स्कंद कोप्पुला, डैनियल ज़ोरान, एंड्रयू ब्रॉक, इवान शेलहैमर, ओलिवियर हेनाफ, मैथ्यू एम। बोट्विनिक, एंड्रयू ज़िसरमैन, ओरिओल विनियल्स, जोआओ कैरेरा द्वारा पोस्ट किया गया।
1. **[Persimmon](https://huggingface.co/docs/transformers/model_doc/persimmon)** (ADEPT से) Erich Elsen, Augustus Odena, Maxwell Nye, Sağnak Taşırlar, Tri Dao, Curtis Hawthorne, Deepak Moparthi, Arushi Somani. द्वाराअनुसंधान पत्र [blog post](https://www.adept.ai/blog/persimmon-8b) के साथ जारी किया गया
1. **[Phi](https://huggingface.co/docs/transformers/model_doc/phi)** (from Microsoft) released with the papers - [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) by Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee and Yuanzhi Li, [Textbooks Are All You Need II: phi-1.5 technical report](https://arxiv.org/abs/2309.05463) by Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar and Yin Tat Lee.
1. **[PhoBERT](https://huggingface.co/docs/transformers/model_doc/phobert)** (VinAI Research से) कागज के साथ [PhoBERT: वियतनामी के लिए पूर्व-प्रशिक्षित भाषा मॉडल](https://www.aclweb.org/anthology/2020.findings-emnlp.92/) डैट क्वोक गुयेन और अन्ह तुआन गुयेन द्वारा पोस्ट किया गया।
1. **[Pix2Struct](https://huggingface.co/docs/transformers/model_doc/pix2struct)** (Google से) Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, Kristina Toutanova. द्वाराअनुसंधान पत्र [Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding](https://arxiv.org/abs/2210.03347) के साथ जारी किया गया
1. **[PLBart](https://huggingface.co/docs/transformers/model_doc/plbart)** (UCLA NLP से) साथ वाला पेपर [प्रोग्राम अंडरस्टैंडिंग एंड जेनरेशन के लिए यूनिफाइड प्री-ट्रेनिंग](https://arxiv.org/abs/2103.06333) वसी उद्दीन अहमद, सैकत चक्रवर्ती, बैशाखी रे, काई-वेई चांग द्वारा।
1. **[PoolFormer](https://huggingface.co/docs/transformers/model_doc/poolformer)** (from Sea AI Labs) released with the paper [MetaFormer is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418) by Yu, Weihao and Luo, Mi and Zhou, Pan and Si, Chenyang and Zhou, Yichen and Wang, Xinchao and Feng, Jiashi and Yan, Shuicheng.
1. **[Pop2Piano](https://huggingface.co/docs/transformers/model_doc/pop2piano)** released with the paper [Pop2Piano : Pop Audio-based Piano Cover Generation](https://arxiv.org/abs/2211.00895) by Jongho Choi, Kyogu Lee.
1. **[ProphetNet](https://huggingface.co/docs/transformers/model_doc/prophetnet)** (माइक्रोसॉफ्ट रिसर्च से) साथ में पेपर [ProphetNet: प्रेडिक्टिंग फ्यूचर एन-ग्राम फॉर सीक्वेंस-टू-सीक्वेंस प्री-ट्रेनिंग](https://arxiv.org/abs/2001.04063) यू यान, वीज़ेन क्यूई, येयुन गोंग, दयाहेंग लियू, नान डुआन, जिउशेंग चेन, रुओफ़ेई झांग और मिंग झोउ द्वारा पोस्ट किया गया।
1. **[PVT](https://huggingface.co/docs/transformers/model_doc/pvt)** (Nanjing University, The University of Hong Kong etc. से) Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao. द्वाराअनुसंधान पत्र [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/pdf/2102.12122.pdf) के साथ जारी किया गया
1. **[QDQBert](https://huggingface.co/docs/transformers/model_doc/qdqbert)** (NVIDIA से) साथ वाला पेपर [डीप लर्निंग इंफ़ेक्शन के लिए इंटीजर क्वांटिज़ेशन: प्रिंसिपल्स एंड एम्पिरिकल इवैल्यूएशन](https://arxiv.org/abs/2004.09602) हाओ वू, पैट्रिक जुड, जिआओजी झांग, मिखाइल इसेव और पॉलियस माइकेविसियस द्वारा।
1. **[Qwen2](https://huggingface.co/docs/transformers/model_doc/qwen2)** (the Qwen team, Alibaba Group से) Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou and Tianhang Zhu. द्वाराअनुसंधान पत्र [Qwen Technical Report](https://arxiv.org/abs/2309.16609) के साथ जारी किया गया
1. **[RAG](https://huggingface.co/docs/transformers/model_doc/rag)** (फेसबुक से) साथ में कागज [रिट्रीवल-ऑगमेंटेड जेनरेशन फॉर नॉलेज-इंटेंसिव एनएलपी टास्क](https://arxiv.org/abs/2005.11401) पैट्रिक लुईस, एथन पेरेज़, अलेक्जेंड्रा पिक्टस, फैबियो पेट्रोनी, व्लादिमीर कारपुखिन, नमन गोयल, हेनरिक कुटलर, माइक लुईस, वेन-ताउ यिह, टिम रॉकटाशेल, सेबस्टियन रिडेल, डौवे कीला द्वारा।
1. **[REALM](https://huggingface.co/docs/transformers/model_doc/realm.html)** (Google अनुसंधान से) केल्विन गु, केंटन ली, ज़ोरा तुंग, पानुपोंग पसुपत और मिंग-वेई चांग द्वारा साथ में दिया गया पेपर [REALM: रिट्रीवल-ऑगमेंटेड लैंग्वेज मॉडल प्री-ट्रेनिंग](https://arxiv.org/abs/2002.08909)।
1. **[Reformer](https://huggingface.co/docs/transformers/model_doc/reformer)** (from Google Research) released with the paper [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.
1. **[RegNet](https://huggingface.co/docs/transformers/model_doc/regnet)** (META रिसर्च से) [डिज़ाइनिंग नेटवर्क डिज़ाइन स्पेस](https://arxiv.org/) पेपर के साथ जारी किया गया एब्स/2003.13678) इलिजा राडोसावोविक, राज प्रतीक कोसाराजू, रॉस गिर्शिक, कैमिंग ही, पिओटर डॉलर द्वारा।
1. **[RemBERT](https://huggingface.co/docs/transformers/model_doc/rembert)** (गूगल रिसर्च से) साथ वाला पेपर [पूर्व-प्रशिक्षित भाषा मॉडल में एम्बेडिंग कपलिंग पर पुनर्विचार](https://arxiv.org/pdf/2010.12821.pdf) ह्युंग वोन चुंग, थिबॉल्ट फ़ेवरी, हेनरी त्साई, एम. जॉनसन, सेबेस्टियन रुडर द्वारा।
1. **[ResNet](https://huggingface.co/docs/transformers/model_doc/resnet)** (माइक्रोसॉफ्ट रिसर्च से) [डीप रेसिडुअल लर्निंग फॉर इमेज रिकग्निशन](https://arxiv.org/abs/1512.03385) कैमिंग हे, जियांग्यु झांग, शाओकिंग रेन, जियान सन द्वारा।
1. **[RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta)** (फेसबुक से), साथ में कागज [मजबूत रूप से अनुकूलित BERT प्रीट्रेनिंग दृष्टिकोण](https://arxiv.org/abs/1907.11692) यिनहान लियू, मायल ओट, नमन गोयल, जिंगफेई डू, मंदार जोशी, डैनकी चेन, ओमर लेवी, माइक लुईस, ल्यूक ज़ेटलमॉयर, वेसेलिन स्टोयानोव द्वारा।
1. **[RoBERTa-PreLayerNorm](https://huggingface.co/docs/transformers/model_doc/roberta-prelayernorm)** (from Facebook) released with the paper [fairseq: A Fast, Extensible Toolkit for Sequence Modeling](https://arxiv.org/abs/1904.01038) by Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, Michael Auli.
1. **[RoCBert](https://huggingface.co/docs/transformers/model_doc/roc_bert)** (from WeChatAI) released with the paper [RoCBert: Robust Chinese Bert with Multimodal Contrastive Pretraining](https://aclanthology.org/2022.acl-long.65.pdf) by HuiSu, WeiweiShi, XiaoyuShen, XiaoZhou, TuoJi, JiaruiFang, JieZhou.
1. **[RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer)** (झुईई टेक्नोलॉजी से), साथ में पेपर [रोफॉर्मर: रोटरी पोजिशन एंबेडिंग के साथ एन्हांस्ड ट्रांसफॉर्मर](https://arxiv.org/pdf/2104.09864v1.pdf) जियानलिन सु और यू लू और शेंगफेंग पैन और बो वेन और युनफेंग लियू द्वारा प्रकाशित।
1. **[RWKV](https://huggingface.co/docs/transformers/model_doc/rwkv)** (Bo Peng से) Bo Peng. द्वाराअनुसंधान पत्र [this repo](https://github.com/BlinkDL/RWKV-LM) के साथ जारी किया गया
1. **[SeamlessM4T](https://huggingface.co/docs/transformers/model_doc/seamless_m4t)** (from Meta AI) released with the paper [SeamlessM4T — Massively Multilingual & Multimodal Machine Translation](https://dl.fbaipublicfiles.com/seamless/seamless_m4t_paper.pdf) by the Seamless Communication team.
1. **[SeamlessM4Tv2](https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2)** (from Meta AI) released with the paper [Seamless: Multilingual Expressive and Streaming Speech Translation](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/) by the Seamless Communication team.
1. **[SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer)** (from NVIDIA) released with the paper [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) by Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo.
1. **[SegGPT](https://huggingface.co/docs/transformers/main/model_doc/seggpt)** (Beijing Academy of Artificial Intelligence (BAAI से) Xinlong Wang, Xiaosong Zhang, Yue Cao, Wen Wang, Chunhua Shen, Tiejun Huang. द्वाराअनुसंधान पत्र [SegGPT: Segmenting Everything In Context](https://arxiv.org/abs/2304.03284) के साथ जारी किया गया
1. **[Segment Anything](https://huggingface.co/docs/transformers/model_doc/sam)** (Meta AI से) Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alex Berg, Wan-Yen Lo, Piotr Dollar, Ross Girshick. द्वाराअनुसंधान पत्र [Segment Anything](https://arxiv.org/pdf/2304.02643v1.pdf) के साथ जारी किया गया
1. **[SEW](https://huggingface.co/docs/transformers/model_doc/sew)** (ASAPP से) साथ देने वाला पेपर [भाषण पहचान के लिए अनसुपरवाइज्ड प्री-ट्रेनिंग में परफॉर्मेंस-एफिशिएंसी ट्रेड-ऑफ्स](https://arxiv.org/abs/2109.06870) फेलिक्स वू, क्वांगयुन किम, जिंग पैन, क्यू हान, किलियन क्यू. वेनबर्गर, योव आर्टज़ी द्वारा।
1. **[SEW-D](https://huggingface.co/docs/transformers/model_doc/sew_d)** (ASAPP से) साथ में पेपर [भाषण पहचान के लिए अनसुपरवाइज्ड प्री-ट्रेनिंग में परफॉर्मेंस-एफिशिएंसी ट्रेड-ऑफ्स](https://arxiv.org/abs/2109.06870) फेलिक्स वू, क्वांगयुन किम, जिंग पैन, क्यू हान, किलियन क्यू. वेनबर्गर, योआव आर्टज़ी द्वारा पोस्ट किया गया।
1. **[SigLIP](https://huggingface.co/docs/transformers/model_doc/siglip)** (Google AI से) Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer. द्वाराअनुसंधान पत्र [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) के साथ जारी किया गया
1. **[SpeechT5](https://huggingface.co/docs/transformers/model_doc/speecht5)** (from Microsoft Research) released with the paper [SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing](https://arxiv.org/abs/2110.07205) by Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, Furu Wei.
1. **[SpeechToTextTransformer](https://huggingface.co/docs/transformers/model_doc/speech_to_text)** (फेसबुक से), साथ में पेपर [फेयरसेक S2T: फास्ट स्पीच-टू-टेक्स्ट मॉडलिंग विद फेयरसेक](https://arxiv.org/abs/2010.05171) चांगहान वांग, यूं तांग, जुताई मा, ऐनी वू, दिमित्रो ओखोनको, जुआन पिनो द्वारा पोस्ट किया गया。
1. **[SpeechToTextTransformer2](https://huggingface.co/docs/transformers/model_doc/speech_to_text_2)** (फेसबुक से) साथ में पेपर [लार्ज-स्केल सेल्फ- एंड सेमी-सुपरवाइज्ड लर्निंग फॉर स्पीच ट्रांसलेशन](https://arxiv.org/abs/2104.06678) चांगहान वांग, ऐनी वू, जुआन पिनो, एलेक्सी बेवस्की, माइकल औली, एलेक्सिस द्वारा Conneau द्वारा पोस्ट किया गया।
1. **[Splinter](https://huggingface.co/docs/transformers/model_doc/splinter)** (तेल अवीव यूनिवर्सिटी से) साथ में पेपर [स्पैन सिलेक्शन को प्री-ट्रेनिंग करके कुछ-शॉट क्वेश्चन आंसरिंग](https://arxiv.org/abs/2101.00438) ओरि राम, युवल कर्स्टन, जोनाथन बेरेंट, अमीर ग्लोबर्सन, ओमर लेवी द्वारा।
1. **[SqueezeBERT](https://huggingface.co/docs/transformers/model_doc/squeezebert)** (बर्कले से) कागज के साथ [SqueezeBERT: कुशल तंत्रिका नेटवर्क के बारे में NLP को कंप्यूटर विज़न क्या सिखा सकता है?](https://arxiv.org/abs/2006.11316) फॉरेस्ट एन. इनडोला, अल्बर्ट ई. शॉ, रवि कृष्णा, और कर्ट डब्ल्यू. केटज़र द्वारा।
1. **[StableLm](https://huggingface.co/docs/transformers/model_doc/stablelm)** (from Stability AI) released with the paper [StableLM 3B 4E1T (Technical Report)](https://stability.wandb.io/stability-llm/stable-lm/reports/StableLM-3B-4E1T--VmlldzoyMjU4?accessToken=u3zujipenkx5g7rtcj9qojjgxpconyjktjkli2po09nffrffdhhchq045vp0wyfo) by  Jonathan Tow, Marco Bellagente, Dakota Mahan, Carlos Riquelme Ruiz, Duy Phung, Maksym Zhuravinskyi, Nathan Cooper, Nikhil Pinnaparaju, Reshinth Adithyan, and James Baicoianu. 
1. **[Starcoder2](https://huggingface.co/docs/transformers/main/model_doc/starcoder2)** (from BigCode team) released with the paper [StarCoder 2 and The Stack v2: The Next Generation](https://arxiv.org/abs/2402.19173) by Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, Tianyang Liu, Max Tian, Denis Kocetkov, Arthur Zucker, Younes Belkada, Zijian Wang, Qian Liu, Dmitry Abulkhanov, Indraneil Paul, Zhuang Li, Wen-Ding Li, Megan Risdal, Jia Li, Jian Zhu, Terry Yue Zhuo, Evgenii Zheltonozhskii, Nii Osae Osae Dade, Wenhao Yu, Lucas Krauß, Naman Jain, Yixuan Su, Xuanli He, Manan Dey, Edoardo Abati, Yekun Chai, Niklas Muennighoff, Xiangru Tang, Muhtasham Oblokulov, Christopher Akiki, Marc Marone, Chenghao Mou, Mayank Mishra, Alex Gu, Binyuan Hui, Tri Dao, Armel Zebaze, Olivier Dehaene, Nicolas Patry, Canwen Xu, Julian McAuley, Han Hu, Torsten Scholak, Sebastien Paquet, Jennifer Robinson, Carolyn Jane Anderson, Nicolas Chapados, Mostofa Patwary, Nima Tajbakhsh, Yacine Jernite, Carlos Muñoz Ferrandis, Lingming Zhang, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries.
1. **[SwiftFormer](https://huggingface.co/docs/transformers/model_doc/swiftformer)** (MBZUAI से) Abdelrahman Shaker, Muhammad Maaz, Hanoona Rasheed, Salman Khan, Ming-Hsuan Yang, Fahad Shahbaz Khan. द्वाराअनुसंधान पत्र [SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications](https://arxiv.org/abs/2303.15446) के साथ जारी किया गया
1. **[Swin Transformer](https://huggingface.co/docs/transformers/model_doc/swin)** (माइक्रोसॉफ्ट से) साथ में कागज [स्वाइन ट्रांसफॉर्मर: शिफ्टेड विंडोज का उपयोग कर पदानुक्रमित विजन ट्रांसफॉर्मर](https://arxiv.org/abs/2103.14030) ज़ी लियू, युटोंग लिन, यू काओ, हान हू, यिक्सुआन वेई, झेंग झांग, स्टीफन लिन, बैनिंग गुओ द्वारा।
1. **[Swin Transformer V2](https://huggingface.co/docs/transformers/model_doc/swinv2)** (Microsoft से) साथ वाला पेपर [Swin Transformer V2: स्केलिंग अप कैपेसिटी एंड रेजोल्यूशन](https://arxiv.org/abs/2111.09883) ज़ी लियू, हान हू, युटोंग लिन, ज़ुलिआंग याओ, ज़ेंडा ज़ी, यिक्सुआन वेई, जिया निंग, यू काओ, झेंग झांग, ली डोंग, फुरु वेई, बैनिंग गुओ द्वारा।
1. **[Swin2SR](https://huggingface.co/docs/transformers/model_doc/swin2sr)** (from University of Würzburg) released with the paper [Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration](https://arxiv.org/abs/2209.11345) by Marcos V. Conde, Ui-Jin Choi, Maxime Burchi, Radu Timofte.
1. **[SwitchTransformers](https://huggingface.co/docs/transformers/model_doc/switch_transformers)** (from Google) released with the paper [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) by William Fedus, Barret Zoph, Noam Shazeer.
1. **[T5](https://huggingface.co/docs/transformers/model_doc/t5)** (来自 Google AI)कॉलिन रैफेल और नोम शज़ीर और एडम रॉबर्ट्स और कैथरीन ली और शरण नारंग और माइकल मटेना द्वारा साथ में पेपर [एक एकीकृत टेक्स्ट-टू-टेक्स्ट ट्रांसफॉर्मर के साथ स्थानांतरण सीखने की सीमा की खोज](https://arxiv.org/abs/1910.10683) और यांकी झोउ और वेई ली और पीटर जे लियू।
1. **[T5v1.1](https://huggingface.co/docs/transformers/model_doc/t5v1.1)** (Google AI से) साथ वाला पेपर [google-research/text-to-text-transfer- ट्रांसफॉर्मर](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511) कॉलिन रैफेल और नोम शज़ीर और एडम रॉबर्ट्स और कैथरीन ली और शरण नारंग द्वारा और माइकल मटेना और यांकी झोउ और वेई ली और पीटर जे लियू।
1. **[Table Transformer](https://huggingface.co/docs/transformers/model_doc/table-transformer)** (माइक्रोसॉफ्ट रिसर्च से) साथ में पेपर [पबटेबल्स-1एम: टूवर्ड्स कॉम्प्रिहेंसिव टेबल एक्सट्रैक्शन फ्रॉम अनस्ट्रक्चर्ड डॉक्यूमेंट्स](https://arxiv.org/abs/2110.00061) ब्रैंडन स्मॉक, रोहित पेसाला, रॉबिन अब्राहम द्वारा पोस्ट किया गया।
1. **[TAPAS](https://huggingface.co/docs/transformers/model_doc/tapas)** (Google AI से) साथ में कागज [TAPAS: पूर्व-प्रशिक्षण के माध्यम से कमजोर पर्यवेक्षण तालिका पार्सिंग](https://arxiv.org/abs/2004.02349) जोनाथन हर्ज़िग, पावेल क्रिज़िस्तोफ़ नोवाक, थॉमस मुलर, फ्रांसेस्को पिकिन्नो और जूलियन मार्टिन ईसेन्च्लोस द्वारा।
1. **[TAPEX](https://huggingface.co/docs/transformers/model_doc/tapex)** (माइक्रोसॉफ्ट रिसर्च से) साथ में पेपर [TAPEX: टेबल प्री-ट्रेनिंग थ्रू लर्निंग अ न्यूरल SQL एक्ज़ीक्यूटर](https://arxiv.org/abs/2107.07653) कियान लियू, बेई चेन, जियाकी गुओ, मोर्टेज़ा ज़ियादी, ज़ेकी लिन, वीज़ू चेन, जियान-गुआंग लू द्वारा पोस्ट किया गया।
1. **[Time Series Transformer](https://huggingface.co/docs/transformers/model_doc/time_series_transformer)** (from HuggingFace).
1. **[TimeSformer](https://huggingface.co/docs/transformers/model_doc/timesformer)** (from Facebook) released with the paper [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095) by Gedas Bertasius, Heng Wang, Lorenzo Torresani.
1. **[Trajectory Transformer](https://huggingface.co/docs/transformers/model_doc/trajectory_transformers)** (from the University of California at Berkeley) released with the paper [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/abs/2106.02039) by Michael Janner, Qiyang Li, Sergey Levine
1. **[Transformer-XL](https://huggingface.co/docs/transformers/model_doc/transfo-xl)** (Google/CMU की ओर से) कागज के साथ [संस्करण-एक्स: एक ब्लॉग मॉडल चौकस चौक मॉडल मॉडल](https://arxivorg/abs/1901.02860) क्वोकोक वी. ले, रुस्लैन सलाखुतदी
1. **[TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr)** (from Microsoft) released with the paper [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282) by Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei.
1. **[TVLT](https://huggingface.co/docs/transformers/model_doc/tvlt)** (from UNC Chapel Hill) released with the paper [TVLT: Textless Vision-Language Transformer](https://arxiv.org/abs/2209.14156) by Zineng Tang, Jaemin Cho, Yixin Nie, Mohit Bansal.
1. **[TVP](https://huggingface.co/docs/transformers/model_doc/tvp)** (from Intel) released with the paper [Text-Visual Prompting for Efficient 2D Temporal Video Grounding](https://arxiv.org/abs/2303.04995) by Yimeng Zhang, Xin Chen, Jinghan Jia, Sijia Liu, Ke Ding.
1. **[UDOP](https://huggingface.co/docs/transformers/main/model_doc/udop)** (Microsoft Research से) Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, Mohit Bansal. द्वाराअनुसंधान पत्र [Unifying Vision, Text, and Layout for Universal Document Processing](https://arxiv.org/abs/2212.02623) के साथ जारी किया गया
1. **[UL2](https://huggingface.co/docs/transformers/model_doc/ul2)** (from Google Research) released with the paper [Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131v1) by Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Xavier Garcia, Dara Bahri, Tal Schuster, Huaixiu Steven Zheng, Neil Houlsby, Donald Metzler
1. **[UMT5](https://huggingface.co/docs/transformers/model_doc/umt5)** (Google Research से) Hyung Won Chung, Xavier Garcia, Adam Roberts, Yi Tay, Orhan Firat, Sharan Narang, Noah Constant. द्वाराअनुसंधान पत्र [UniMax: Fairer and More Effective Language Sampling for Large-Scale Multilingual Pretraining](https://openreview.net/forum?id=kXwdL1cWOAi) के साथ जारी किया गया
1. **[UniSpeech](https://huggingface.co/docs/transformers/model_doc/unispeech)** (माइक्रोसॉफ्ट रिसर्च से) साथ में दिया गया पेपर [UniSpeech: यूनिफाइड स्पीच रिप्रेजेंटेशन लर्निंग विद लेबलेड एंड अनलेबल्ड डेटा](https://arxiv.org/abs/2101.07597) चेंगई वांग, यू वू, याओ कियान, केनिची कुमातानी, शुजी लियू, फुरु वेई, माइकल ज़ेंग, ज़ुएदोंग हुआंग द्वारा।
1. **[UniSpeechSat](https://huggingface.co/docs/transformers/model_doc/unispeech-sat)** (माइक्रोसॉफ्ट रिसर्च से) कागज के साथ [UNISPEECH-SAT: यूनिवर्सल स्पीच रिप्रेजेंटेशन लर्निंग विद स्पीकर अवेयर प्री-ट्रेनिंग](https://arxiv.org/abs/2110.05752) सानयुआन चेन, यू वू, चेंग्यी वांग, झेंगयांग चेन, झूओ चेन, शुजी लियू, जियान वू, याओ कियान, फुरु वेई, जिन्यु ली, जियांगज़ान यू द्वारा पोस्ट किया गया।
1. **[UnivNet](https://huggingface.co/docs/transformers/model_doc/univnet)** (from Kakao Corporation) released with the paper [UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation](https://arxiv.org/abs/2106.07889) by Won Jang, Dan Lim, Jaesam Yoon, Bongwan Kim, and Juntae Kim.
1. **[UPerNet](https://huggingface.co/docs/transformers/model_doc/upernet)** (from Peking University) released with the paper [Unified Perceptual Parsing for Scene Understanding](https://arxiv.org/abs/1807.10221) by Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, Jian Sun.
1. **[VAN](https://huggingface.co/docs/transformers/model_doc/van)** (सिंघुआ यूनिवर्सिटी और ननकाई यूनिवर्सिटी से) साथ में पेपर [विजुअल अटेंशन नेटवर्क](https://arxiv.org/pdf/2202.09741.pdf) मेंग-हाओ गुओ, चेंग-ज़े लू, झेंग-निंग लियू, मिंग-मिंग चेंग, शि-मिन हू द्वारा।
1. **[VideoMAE](https://huggingface.co/docs/transformers/model_doc/videomae)** (मल्टीमीडिया कम्प्यूटिंग ग्रुप, नानजिंग यूनिवर्सिटी से) साथ में पेपर [वीडियोएमएई: मास्क्ड ऑटोएन्कोडर स्व-पर्यवेक्षित वीडियो प्री-ट्रेनिंग के लिए डेटा-कुशल सीखने वाले हैं](https://arxiv.org/abs/2203.12602) ज़ान टोंग, यिबिंग सॉन्ग, जुए द्वारा वांग, लिमिन वांग द्वारा पोस्ट किया गया।
1. **[ViLT](https://huggingface.co/docs/transformers/model_doc/vilt)** (NAVER AI Lab/Kakao Enterprise/Kakao Brain से) साथ में कागज [ViLT: Vision-and-Language Transformer बिना कनवल्शन या रीजन सुपरविजन](https://arxiv.org/abs/2102.03334) वोनजे किम, बोक्यूंग सोन, इल्डू किम द्वारा पोस्ट किया गया।
1. **[VipLlava](https://huggingface.co/docs/transformers/model_doc/vipllava)** (University of Wisconsin–Madison से) Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P. Meyer, Yuning Chai, Dennis Park, Yong Jae Lee. द्वाराअनुसंधान पत्र [Making Large Multimodal Models Understand Arbitrary Visual Prompts](https://arxiv.org/abs/2312.00784) के साथ जारी किया गया
1. **[Vision Transformer (ViT)](https://huggingface.co/docs/transformers/model_doc/vit)** (गूगल एआई से) कागज के साथ [एक इमेज इज़ वर्थ 16x16 वर्ड्स: ट्रांसफॉर्मर्स फॉर इमेज रिकॉग्निशन एट स्केल](https://arxiv.org/abs/2010.11929) एलेक्सी डोसोवित्स्की, लुकास बेयर, अलेक्जेंडर कोलेसनिकोव, डिर्क वीसेनबोर्न, शियाओहुआ झाई, थॉमस अनटरथिनर, मुस्तफा देहघानी, मैथियास मिंडरर, जॉर्ज हेगोल्ड, सिल्वेन गेली, जैकब उस्ज़कोरेइट द्वारा हॉल्सबी द्वारा पोस्ट किया गया।
1. **[VisualBERT](https://huggingface.co/docs/transformers/model_doc/visual_bert)** (UCLA NLP से) साथ वाला पेपर [VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/pdf/1908.03557) लियुनियन हेरोल्ड ली, मार्क यात्स्कर, दा यिन, चो-जुई हसीह, काई-वेई चांग द्वारा।
1. **[ViT Hybrid](https://huggingface.co/docs/transformers/model_doc/vit_hybrid)** (from Google AI) released with the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.
1. **[VitDet](https://huggingface.co/docs/transformers/model_doc/vitdet)** (Meta AI से) Yanghao Li, Hanzi Mao, Ross Girshick, Kaiming He. द्वाराअनुसंधान पत्र [Exploring Plain Vision Transformer Backbones for Object Detection](https://arxiv.org/abs/2203.16527) के साथ जारी किया गया
1. **[ViTMAE](https://huggingface.co/docs/transformers/model_doc/vit_mae)** (मेटा एआई से) साथ में कागज [मास्कड ऑटोएन्कोडर स्केलेबल विजन लर्नर्स हैं](https://arxiv.org/एब्स/2111.06377) कैमिंग हे, ज़िनेली चेन, सेनिंग ज़ी, यांगहो ली, पिओट्र डॉलर, रॉस गिर्शिक द्वारा।
1. **[ViTMatte](https://huggingface.co/docs/transformers/model_doc/vitmatte)** (HUST-VL से) Jingfeng Yao, Xinggang Wang, Shusheng Yang, Baoyuan Wang. द्वाराअनुसंधान पत्र [ViTMatte: Boosting Image Matting with Pretrained Plain Vision Transformers](https://arxiv.org/abs/2305.15272) के साथ जारी किया गया
1. **[ViTMSN](https://huggingface.co/docs/transformers/model_doc/vit_msn)** (मेटा एआई से) साथ में कागज [लेबल-कुशल सीखने के लिए मास्क्ड स्याम देश के नेटवर्क](https://arxiv.org/abs/2204.07141) महमूद असरान, मथिल्डे कैरन, ईशान मिश्रा, पियोट्र बोजानोवस्की, फ्लोरियन बोर्डेस, पास्कल विंसेंट, आर्मंड जौलिन, माइकल रब्बत, निकोलस बल्लास द्वारा।
1. **[VITS](https://huggingface.co/docs/transformers/model_doc/vits)** (Kakao Enterprise से) Jaehyeon Kim, Jungil Kong, Juhee Son. द्वाराअनुसंधान पत्र [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103) के साथ जारी किया गया
1. **[ViViT](https://huggingface.co/docs/transformers/model_doc/vivit)** (from Google Research) released with the paper [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691) by Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, Cordelia Schmid.
1. **[Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2)** (फेसबुक एआई से) साथ में पेपर [wav2vec 2.0: ए फ्रेमवर्क फॉर सेल्फ-सुपरवाइज्ड लर्निंग ऑफ स्पीच रिप्रेजेंटेशन](https://arxiv.org/abs/2006.11477) एलेक्सी बेवस्की, हेनरी झोउ, अब्देलरहमान मोहम्मद, माइकल औली द्वारा।
1. **[Wav2Vec2-BERT](https://huggingface.co/docs/transformers/model_doc/wav2vec2-bert)** (from Meta AI) released with the paper [Seamless: Multilingual Expressive and Streaming Speech Translation](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/) by the Seamless Communication team.
1. **[Wav2Vec2-Conformer](https://huggingface.co/docs/transformers/model_doc/wav2vec2-conformer)** (Facebook AI से) साथ वाला पेपर [FAIRSEQ S2T: FAIRSEQ के साथ फास्ट स्पीच-टू-टेक्स्ट मॉडलिंग](https://arxiv.org/abs/2010.05171) चांगहान वांग, यूं तांग, जुताई मा, ऐनी वू, सरव्या पोपुरी, दिमित्रो ओखोनको, जुआन पिनो द्वारा पोस्ट किया गया।
1. **[Wav2Vec2Phoneme](https://huggingface.co/docs/transformers/model_doc/wav2vec2_phoneme)** (Facebook AI से) साथ वाला पेपर [सरल और प्रभावी जीरो-शॉट क्रॉस-लिंगुअल फोनेम रिकॉग्निशन](https://arxiv.org/abs/2109.11680) कियानटोंग जू, एलेक्सी बाएव्स्की, माइकल औली द्वारा।
1. **[WavLM](https://huggingface.co/docs/transformers/model_doc/wavlm)** (माइक्रोसॉफ्ट रिसर्च से) पेपर के साथ जारी किया गया [WavLM: फुल स्टैक के लिए बड़े पैमाने पर स्व-पर्यवेक्षित पूर्व-प्रशिक्षण स्पीच प्रोसेसिंग](https://arxiv.org/abs/2110.13900) सानयुआन चेन, चेंगयी वांग, झेंगयांग चेन, यू वू, शुजी लियू, ज़ुओ चेन, जिन्यु ली, नाओयुकी कांडा, ताकुया योशियोका, ज़िओंग जिओ, जियान वू, लॉन्ग झोउ, शुओ रेन, यानमिन कियान, याओ कियान, जियान वू, माइकल ज़ेंग, फुरु वेई।
1. **[Whisper](https://huggingface.co/docs/transformers/model_doc/whisper)** (OpenAI से) साथ में कागज [बड़े पैमाने पर कमजोर पर्यवेक्षण के माध्यम से मजबूत भाषण पहचान](https://cdn.openai.com/papers/whisper.pdf) एलेक रैडफोर्ड, जोंग वूक किम, ताओ जू, ग्रेग ब्रॉकमैन, क्रिस्टीन मैकलीवे, इल्या सुत्स्केवर द्वारा।
1. **[X-CLIP](https://huggingface.co/docs/transformers/model_doc/xclip)** (माइक्रोसॉफ्ट रिसर्च से) कागज के साथ [एक्सपैंडिंग लैंग्वेज-इमेज प्रीट्रेन्ड मॉडल फॉर जनरल वीडियो रिकग्निशन](https://arxiv.org/abs/2208.02816) बोलिन नी, होउवेन पेंग, मिंगाओ चेन, सोंगयांग झांग, गाओफेंग मेंग, जियानलोंग फू, शिमिंग जियांग, हैबिन लिंग द्वारा।
1. **[X-MOD](https://huggingface.co/docs/transformers/model_doc/xmod)** (Meta AI से) Jonas Pfeiffer, Naman Goyal, Xi Lin, Xian Li, James Cross, Sebastian Riedel, Mikel Artetxe. द्वाराअनुसंधान पत्र [Lifting the Curse of Multilinguality by Pre-training Modular Transformers](http://dx.doi.org/10.18653/v1/2022.naacl-main.255) के साथ जारी किया गया
1. **[XGLM](https://huggingface.co/docs/transformers/model_doc/xglm)** (From Facebook AI) released with the paper [Few-shot Learning with Multilingual Language Models](https://arxiv.org/abs/2112.10668) by Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O'Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona Diab, Veselin Stoyanov, Xian Li.
1. **[XLM](https://huggingface.co/docs/transformers/model_doc/xlm)** (फेसबुक से) साथ में पेपर [क्रॉस-लिंगुअल लैंग्वेज मॉडल प्रीट्रेनिंग](https://arxiv.org/abs/1901.07291) गिलाउम लैम्पल और एलेक्सिस कोनो द्वारा।
1. **[XLM-ProphetNet](https://huggingface.co/docs/transformers/model_doc/xlm-prophetnet)** (माइक्रोसॉफ्ट रिसर्च से) साथ में कागज [ProphetNet: प्रेडिक्टिंग फ्यूचर एन-ग्राम फॉर सीक्वेंस-टू- सीक्वेंस प्री-ट्रेनिंग](https://arxiv.org/abs/2001.04063) यू यान, वीज़ेन क्यूई, येयुन गोंग, दयाहेंग लियू, नान डुआन, जिउशेंग चेन, रुओफ़ेई झांग और मिंग झोउ द्वारा।
1. **[XLM-RoBERTa](https://huggingface.co/docs/transformers/model_doc/xlm-roberta)** (फेसबुक एआई से), साथ में पेपर [अनसुपरवाइज्ड क्रॉस-लिंगुअल रिप्रेजेंटेशन लर्निंग एट स्केल](https://arxiv.org/abs/1911.02116) एलेक्सिस कोन्यू*, कार्तिकेय खंडेलवाल*, नमन गोयल, विश्रव चौधरी, गिलाउम वेनज़ेक, फ्रांसिस्को गुज़मैन द्वारा , एडौर्ड ग्रेव, मायल ओट, ल्यूक ज़ेटलमॉयर और वेसेलिन स्टोयानोव द्वारा।
1. **[XLM-RoBERTa-XL](https://huggingface.co/docs/transformers/model_doc/xlm-roberta-xl)** (Facebook AI से) साथ में कागज [बहुभाषी नकाबपोश भाषा के लिए बड़े पैमाने पर ट्रांसफॉर्मर मॉडलिंग](https://arxiv.org/abs/2105.00572) नमन गोयल, जिंगफेई डू, मायल ओट, गिरि अनंतरामन, एलेक्सिस कोनो द्वारा पोस्ट किया गया।
1. **[XLM-V](https://huggingface.co/docs/transformers/model_doc/xlm-v)** (from Meta AI) released with the paper [XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models](https://arxiv.org/abs/2301.10472) by Davis Liang, Hila Gonen, Yuning Mao, Rui Hou, Naman Goyal, Marjan Ghazvininejad, Luke Zettlemoyer, Madian Khabsa.
1. **[XLNet](https://huggingface.co/docs/transformers/model_doc/xlnet)** (Google/CMU से) साथ वाला पेपर [XLNet: जनरलाइज्ड ऑटोरेग्रेसिव प्रीट्रेनिंग फॉर लैंग्वेज अंडरस्टैंडिंग](https://arxiv.org/abs/1906.08237) ज़ीलिन यांग*, ज़िहांग दाई*, यिमिंग यांग, जैम कार्बोनेल, रुस्लान सलाखुतदीनोव, क्वोक वी. ले द्वारा।
1. **[XLS-R](https://huggingface.co/docs/transformers/model_doc/xls_r)** (Facebook AI से) साथ वाला पेपर [XLS-R: सेल्फ सुपरवाइज्ड क्रॉस-लिंगुअल स्पीच रिप्रेजेंटेशन लर्निंग एट स्केल](https://arxiv.org/abs/2111.09296) अरुण बाबू, चांगहान वांग, एंड्रोस तजंद्रा, कुशाल लखोटिया, कियानटोंग जू, नमन गोयल, कृतिका सिंह, पैट्रिक वॉन प्लैटन, याथार्थ सराफ, जुआन पिनो, एलेक्सी बेवस्की, एलेक्सिस कोन्यू, माइकल औली द्वारा पोस्ट किया गया।
1. **[XLSR-Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/xlsr_wav2vec2)** (फेसबुक एआई से) साथ में पेपर [अनसुपरवाइज्ड क्रॉस-लिंगुअल रिप्रेजेंटेशन लर्निंग फॉर स्पीच रिकग्निशन](https://arxiv.org/abs/2006.13979) एलेक्सिस कोन्यू, एलेक्सी बेवस्की, रोनन कोलोबर्ट, अब्देलरहमान मोहम्मद, माइकल औली द्वारा।
1. **[YOLOS](https://huggingface.co/docs/transformers/model_doc/yolos)** (हुआझोंग यूनिवर्सिटी ऑफ साइंस एंड टेक्नोलॉजी से) साथ में पेपर [यू ओनली लुक एट वन सीक्वेंस: रीथिंकिंग ट्रांसफॉर्मर इन विज़न थ्रू ऑब्जेक्ट डिटेक्शन](https://arxiv.org/abs/2106.00666) युक्सिन फेंग, बेनचेंग लियाओ, जिंगगैंग वांग, जेमिन फेंग, जियांग क्यूई, रुई वू, जियानवेई नीयू, वेन्यू लियू द्वारा पोस्ट किया गया।
1. **[YOSO](https://huggingface.co/docs/transformers/model_doc/yoso)** (विस्कॉन्सिन विश्वविद्यालय - मैडिसन से) साथ में पेपर [यू ओनली सैंपल (लगभग) ज़ानपेंग ज़ेंग, युनयांग ज़िओंग द्वारा , सत्य एन. रवि, शैलेश आचार्य, ग्लेन फंग, विकास सिंह द्वारा पोस्ट किया गया।
1. एक नए मॉडल में योगदान देना चाहते हैं? नए मॉडल जोड़ने में आपका मार्गदर्शन करने के लिए हमारे पास एक **विस्तृत मार्गदर्शिका और टेम्प्लेट** है। आप उन्हें [`टेम्पलेट्स`](./templates) निर्देशिका में पा सकते हैं। पीआर शुरू करने से पहले [योगदान दिशानिर्देश](./CONTRIBUTING.md) देखना और अनुरक्षकों से संपर्क करना या प्रतिक्रिया प्राप्त करने के लिए एक नया मुद्दा खोलना याद रखें।

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
