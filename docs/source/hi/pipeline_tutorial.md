<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# अनुमान के लिए पाइपलाइन

[`pipeline`] किसी भी भाषा, कंप्यूटर दृष्टि, भाषण और मल्टीमॉडल कार्यों पर अनुमान लगाने के लिए [Hub](https://huggingface.co/models) से किसी भी मॉडल का उपयोग करना आसान बनाता है। भले ही आपके पास किसी विशिष्ट तौर-तरीके का अनुभव न हो या आप मॉडलों के पीछे अंतर्निहित कोड से परिचित न हों, फिर भी आप [`pipeline`] के अनुमान के लिए उनका उपयोग कर सकते हैं! यह ट्यूटोरियल आपको ये सिखाएगा:

* अनुमान के लिए [`pipeline`] का उपयोग करें।
* एक विशिष्ट टोकननाइज़र या मॉडल का उपयोग करें।
* ऑडियो, विज़न और मल्टीमॉडल कार्यों के लिए [`pipeline`] का उपयोग करें।

<Tip>

समर्थित कार्यों और उपलब्ध मापदंडों की पूरी सूची के लिए [`pipeline`] दस्तावेज़ पर एक नज़र डालें।

</Tip>

## पाइपलाइन का उपयोग

जबकि प्रत्येक कार्य में एक संबद्ध [`pipeline`] होता है, सामान्य [`pipeline`] अमूर्त का उपयोग करना आसान होता है जिसमें शामिल होता है
सभी कार्य-विशिष्ट पाइपलाइनें। [`pipeline`] स्वचालित रूप से एक डिफ़ॉल्ट मॉडल और सक्षम प्रीप्रोसेसिंग क्लास लोड करता है
आपके कार्य के लिए अनुमान का. आइए स्वचालित वाक् पहचान (एएसआर) के लिए [`pipeline`] का उपयोग करने का उदाहरण लें, या
वाक्-से-पाठ.


1. एक [`pipeline`] बनाकर प्रारंभ करें और अनुमान कार्य निर्दिष्ट करें:

```py
>>> from transformers import pipeline

>>> transcriber = pipeline(task="automatic-speech-recognition")
```

2. अपना इनपुट [`pipeline`] पर भेजें। वाक् पहचान के मामले में, यह एक ऑडियो इनपुट फ़ाइल है:

```py
>>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': 'I HAVE A DREAM BUT ONE DAY THIS NATION WILL RISE UP LIVE UP THE TRUE MEANING OF ITS TREES'}
```

क्या वह परिणाम नहीं जो आपके मन में था? कुछ [सबसे अधिक डाउनलोड किए गए स्वचालित वाक् पहचान मॉडल](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending) देखें
यह देखने के लिए हब पर जाएं कि क्या आपको बेहतर ट्रांस्क्रिप्शन मिल सकता है।

आइए OpenAI से [व्हिस्पर लार्ज-v2](https://huggingface.co/openai/whisper-large) मॉडल आज़माएं। व्हिस्पर जारी किया गया
Wav2Vec2 की तुलना में 2 साल बाद, और लगभग 10 गुना अधिक डेटा पर प्रशिक्षित किया गया था। इस प्रकार, यह अधिकांश डाउनस्ट्रीम पर Wav2Vec2 को मात देता है
बेंचमार्क. इसमें विराम चिह्न और आवरण की भविष्यवाणी करने का अतिरिक्त लाभ भी है, जिनमें से कोई भी संभव नहीं है
Wav2Vec2.

आइए इसे यहां आज़माकर देखें कि यह कैसा प्रदर्शन करता है:

```py
>>> transcriber = pipeline(model="openai/whisper-large-v2")
>>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

अब यह परिणाम अधिक सटीक दिखता है! Wav2Vec2 बनाम व्हिस्पर पर गहन तुलना के लिए, [ऑडियो ट्रांसफॉर्मर्स कोर्स](https://huggingface.co/learn/audio-course/chapter5/asr_models) देखें।
हम वास्तव में आपको विभिन्न भाषाओं में मॉडल, आपके क्षेत्र में विशेषीकृत मॉडल और बहुत कुछ के लिए हब की जांच करने के लिए प्रोत्साहित करते हैं।
आप हब पर सीधे अपने ब्राउज़र से मॉडल परिणामों की जांच और तुलना कर सकते हैं कि यह फिट बैठता है या नहीं
अन्य मामलों की तुलना में कोने के मामलों को बेहतर ढंग से संभालता है।
और यदि आपको अपने उपयोग के मामले के लिए कोई मॉडल नहीं मिलता है, तो आप हमेशा अपना खुद का [प्रशिक्षण](training) शुरू कर सकते हैं!

यदि आपके पास कई इनपुट हैं, तो आप अपने इनपुट को एक सूची के रूप में पास कर सकते हैं:

```py
transcriber(
    [
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
    ]
)
```

पाइपलाइनें प्रयोग के लिए बहुत अच्छी हैं क्योंकि एक मॉडल से दूसरे मॉडल पर स्विच करना मामूली काम है; हालाँकि, प्रयोग की तुलना में बड़े कार्यभार के लिए उन्हें अनुकूलित करने के कुछ तरीके हैं। संपूर्ण डेटासेट पर पुनरावृत्ति करने या वेबसर्वर में पाइपलाइनों का उपयोग करने के बारे में निम्नलिखित मार्गदर्शिकाएँ देखें:
दस्तावेज़ों में से:
* [डेटासेट पर पाइपलाइनों का उपयोग करना](#using-pipelines-on-a-dataset)
* [वेबसर्वर के लिए पाइपलाइनों का उपयोग करना](./pipeline_webserver)

## प्राचल

[`pipeline`] कई मापदंडों का समर्थन करता है; कुछ कार्य विशिष्ट हैं, और कुछ सभी पाइपलाइनों के लिए सामान्य हैं।
सामान्य तौर पर, आप अपनी इच्छानुसार कहीं भी पैरामीटर निर्दिष्ट कर सकते हैं:

```py
transcriber = pipeline(model="openai/whisper-large-v2", my_parameter=1)

out = transcriber(...)  # This will use `my_parameter=1`.
out = transcriber(..., my_parameter=2)  # This will override and use `my_parameter=2`.
out = transcriber(...)  # This will go back to using `my_parameter=1`.
```

आइए 3 महत्वपूर्ण बातों पर गौर करें:

### उपकरण

यदि आप `device=0` का उपयोग करते हैं, तो पाइपलाइन स्वचालित रूप से मॉडल को निर्दिष्ट डिवाइस पर डाल देती है।
यह इस पर ध्यान दिए बिना काम करेगा कि आप PyTorch या Tensorflow का उपयोग कर रहे हैं या नहीं।

```py
transcriber = pipeline(model="openai/whisper-large-v2", device=0)
```

यदि मॉडल एकल GPU के लिए बहुत बड़ा है और आप PyTorch का उपयोग कर रहे हैं, तो आप `device_map="auto"` को स्वचालित रूप से सेट कर सकते हैं
निर्धारित करें कि मॉडल वज़न को कैसे लोड और संग्रहीत किया जाए। `device_map` तर्क का उपयोग करने के लिए 🤗 [Accelerate](https://huggingface.co/docs/accelerate) की आवश्यकता होती है
पैकेट:

```bash
pip install --upgrade accelerate
```

निम्नलिखित कोड स्वचालित रूप से सभी डिवाइसों में मॉडल भार को लोड और संग्रहीत करता है:

```py
transcriber = pipeline(model="openai/whisper-large-v2", device_map="auto")
```

ध्यान दें कि यदि `device_map='auto'` पारित हो गया है, तो अपनी `pipeline` को चालू करते समय `device=device` तर्क जोड़ने की कोई आवश्यकता नहीं है क्योंकि आपको कुछ अप्रत्याशित व्यवहार का सामना करना पड़ सकता है!

### बैच का आकार

डिफ़ॉल्ट रूप से, पाइपलाइनें [यहां](https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching) विस्तार से बताए गए कारणों के लिए बैच अनुमान नहीं लगाएंगी। इसका कारण यह है कि बैचिंग आवश्यक रूप से तेज़ नहीं है, और वास्तव में कुछ मामलों में काफी धीमी हो सकती है।

लेकिन अगर यह आपके उपयोग के मामले में काम करता है, तो आप इसका उपयोग कर सकते हैं:

```py
transcriber = pipeline(model="openai/whisper-large-v2", device=0, batch_size=2)
audio_filenames = [f"https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/{i}.flac" for i in range(1, 5)]
texts = transcriber(audio_filenames)
```

यह प्रदान की गई 4 ऑडियो फाइलों पर पाइपलाइन चलाता है, लेकिन यह उन्हें 2 के बैच में पास करेगा
आपसे किसी और कोड की आवश्यकता के बिना मॉडल (जो एक जीपीयू पर है, जहां बैचिंग से मदद मिलने की अधिक संभावना है) पर जाएं।
आउटपुट हमेशा उसी से मेल खाना चाहिए जो आपको बैचिंग के बिना प्राप्त हुआ होगा। इसका उद्देश्य केवल पाइपलाइन से अधिक गति प्राप्त करने में आपकी सहायता करना है।

पाइपलाइनें बैचिंग की कुछ जटिलताओं को भी कम कर सकती हैं क्योंकि, कुछ पाइपलाइनों के लिए, एक एकल आइटम (जैसे एक लंबी ऑडियो फ़ाइल) को एक मॉडल द्वारा संसाधित करने के लिए कई भागों में विभाजित करने की आवश्यकता होती है। पाइपलाइन आपके लिए यह [*chunk batching*](./main_classes/pipelines#pipeline-chunk-batching) करती है।

### कार्य विशिष्ट प्राचल

सभी कार्य कार्य विशिष्ट प्राचल प्रदान करते हैं जो आपको अपना काम पूरा करने में मदद करने के लिए अतिरिक्त लचीलेपन और विकल्पों की अनुमति देते हैं।
उदाहरण के लिए, [`transformers.AutomaticSpeechRecognitionPipeline.__call__`] विधि में एक `return_timestamps` प्राचल है जो वीडियो उपशीर्षक के लिए आशाजनक लगता है:


```py
>>> transcriber = pipeline(model="openai/whisper-large-v2", return_timestamps=True)
>>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.', 'chunks': [{'timestamp': (0.0, 11.88), 'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its'}, {'timestamp': (11.88, 12.38), 'text': ' creed.'}]}
```

जैसा कि आप देख सकते हैं, मॉडल ने पाठ का अनुमान लगाया और **when** विभिन्न वाक्यों का उच्चारण किया गया तो आउटपुट भी दिया।

प्रत्येक कार्य के लिए कई प्राचल उपलब्ध हैं, इसलिए यह देखने के लिए कि आप किसके साथ छेड़छाड़ कर सकते हैं, प्रत्येक कार्य का API संदर्भ देखें!
उदाहरण के लिए, [`~transformers.AutomaticSpeechRecognitionPipeline`] में एक `chunk_length_s` प्राचल है जो सहायक है
वास्तव में लंबी ऑडियो फ़ाइलों पर काम करने के लिए (उदाहरण के लिए, संपूर्ण फिल्मों या घंटे-लंबे वीडियो को उपशीर्षक देना) जो आमतौर पर एक मॉडल होता है
अपने आप संभाल नहीं सकता:

```python
>>> transcriber = pipeline(model="openai/whisper-large-v2", chunk_length_s=30, return_timestamps=True)
>>> transcriber("https://huggingface.co/datasets/sanchit-gandhi/librispeech_long/resolve/main/audio.wav")
{'text': " Chapter 16. I might have told you of the beginning of this liaison in a few lines, but I wanted you to see every step by which we came.  I, too, agree to whatever Marguerite wished, Marguerite to be unable to live apart from me. It was the day after the evening...
```

यदि आपको कोई ऐसा पैरामीटर नहीं मिल रहा है जो वास्तव में आपकी मदद करेगा, तो बेझिझक [अनुरोध करें](https://github.com/huggingface/transformers/issues/new?assignees=&labels=feature&template=feature-request.yml)!


## डेटासेट पर पाइपलाइनों का उपयोग करना

पाइपलाइन बड़े डेटासेट पर भी अनुमान चला सकती है। ऐसा करने का सबसे आसान तरीका हम एक पुनरावर्तक का उपयोग करने की सलाह देते हैं:

```py
def data():
    for i in range(1000):
        yield f"My example {i}"


pipe = pipeline(model="openai-community/gpt2", device=0)
generated_characters = 0
for out in pipe(data()):
    generated_characters += len(out[0]["generated_text"])
```

पुनरावर्तक `data()` प्रत्येक परिणाम और पाइपलाइन स्वचालित रूप से उत्पन्न करता है
पहचानता है कि इनपुट पुनरावर्तनीय है और डेटा प्राप्त करना शुरू कर देगा
यह इसे GPU पर प्रोसेस करना जारी रखता है (यह हुड के तहत [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) का उपयोग करता है)।
यह महत्वपूर्ण है क्योंकि आपको संपूर्ण डेटासेट के लिए मेमोरी आवंटित करने की आवश्यकता नहीं है
और आप जितनी जल्दी हो सके GPU को फीड कर सकते हैं।

चूंकि बैचिंग से चीज़ें तेज़ हो सकती हैं, इसलिए यहां `batch_size` प्राचल को ट्यून करने का प्रयास करना उपयोगी हो सकता है।

किसी डेटासेट पर पुनरावृति करने का सबसे सरल तरीका बस एक को 🤗 [Dataset](https://github.com/huggingface/datasets/) से लोड करना है:

```py
# KeyDataset is a util that will just output the item we're interested in.
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset

pipe = pipeline(model="hf-internal-testing/tiny-random-wav2vec2", device=0)
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:10]")

for out in pipe(KeyDataset(dataset, "audio")):
    print(out)
```


## वेबसर्वर के लिए पाइपलाइनों का उपयोग करना

<Tip>
एक अनुमान इंजन बनाना एक जटिल विषय है जो अपने आप में उपयुक्त है
पृष्ठ।
</Tip>

[Link](./pipeline_webserver)

## विज़न पाइपलाइन

दृष्टि कार्यों के लिए [`pipeline`] का उपयोग करना व्यावहारिक रूप से समान है।

अपना कार्य निर्दिष्ट करें और अपनी छवि क्लासिफायरियर को भेजें। छवि एक लिंक, एक स्थानीय पथ या बेस64-एन्कोडेड छवि हो सकती है। उदाहरण के लिए, बिल्ली की कौन सी प्रजाति नीचे दिखाई गई है?

![pipeline-cat-chonk](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg)

```py
>>> from transformers import pipeline

>>> vision_classifier = pipeline(model="google/vit-base-patch16-224")
>>> preds = vision_classifier(
...     images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.4335, 'label': 'lynx, catamount'}, {'score': 0.0348, 'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor'}, {'score': 0.0324, 'label': 'snow leopard, ounce, Panthera uncia'}, {'score': 0.0239, 'label': 'Egyptian cat'}, {'score': 0.0229, 'label': 'tiger cat'}]
```

## पाठ पाइपलाइन

NLP कार्यों के लिए [`pipeline`] का उपयोग करना व्यावहारिक रूप से समान है।

```py
>>> from transformers import pipeline

>>> # This model is a `zero-shot-classification` model.
>>> # It will classify text, except you are free to choose any label you might imagine
>>> classifier = pipeline(model="facebook/bart-large-mnli")
>>> classifier(
...     "I have a problem with my iphone that needs to be resolved asap!!",
...     candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
... )
{'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'], 'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}
```

## बहुविध पाइपलाइन

[`pipeline`] एक से अधिक तौर-तरीकों का समर्थन करती है। उदाहरण के लिए, एक दृश्य प्रश्न उत्तर (VQA) कार्य पाठ और छवि को जोड़ता है। अपनी पसंद के किसी भी छवि लिंक और छवि के बारे में कोई प्रश्न पूछने के लिए स्वतंत्र महसूस करें। छवि एक URL या छवि का स्थानीय पथ हो सकती है।

उदाहरण के लिए, यदि आप इस [invoice image](https://huggingface.co/datasets/hf-internal-testing/transformers-synthetic-assets/resolve/main/images/invoice_docquery_a.png) का उपयोग करते हैं:

```py
>>> from transformers import pipeline

>>> vqa = pipeline(model="impira/layoutlm-document-qa")
>>> output = vqa(
...     image="https://huggingface.co/datasets/hf-internal-testing/transformers-synthetic-assets/resolve/main/images/invoice_docquery_a.png",
...     question="What is the invoice number?",
... )
>>> output[0]["score"] = round(output[0]["score"], 3)
>>> output
[{'score': 0.425, 'answer': 'us-001', 'start': 16, 'end': 16}]
```

<Tip>

ऊपर दिए गए उदाहरण को चलाने के लिए आपको 🤗 ट्रांसफॉर्मर के अलावा [`pytesseract`](https://pypi.org/project/pytesseract/) इंस्टॉल करना होगा:

```bash
sudo apt install -y tesseract-ocr
pip install pytesseract
```

</Tip>

## 🤗 `त्वरण` के साथ बड़े मॉडलों पर `pipeline` का उपयोग करना:

आप 🤗 `accelerate` का उपयोग करके बड़े मॉडलों पर आसानी से `pipeline` चला सकते हैं! पहले सुनिश्चित करें कि आपने `accelerate` को `pip install accelerate` के साथ इंस्टॉल किया है।

सबसे पहले `device_map='auto'` का उपयोग करके अपना मॉडल लोड करें! हम अपने उदाहरण के लिए `facebook/opt-1.3b` का उपयोग करेंगे।

```py
# pip install accelerate
import torch
from transformers import pipeline

pipe = pipeline(model="facebook/opt-1.3b", dtype=torch.bfloat16, device_map="auto")
output = pipe("This is a cool example!", do_sample=True, top_p=0.95)
```

यदि आप `bitsandbytes` इंस्टॉल करते हैं और `load_in_8bit=True` तर्क जोड़ते हैं तो आप 8-बिट लोडेड मॉडल भी पास कर सकते हैं

```py
# pip install accelerate bitsandbytes
import torch
from transformers import pipeline, BitsAndBytesConfig

pipe = pipeline(model="facebook/opt-1.3b", device_map="auto", model_kwargs={"quantization_config": BitsAndBytesConfig(load_in_8bit=True)})
output = pipe("This is a cool example!", do_sample=True, top_p=0.95)
```

ध्यान दें कि आप चेकपॉइंट को किसी भी हगिंग फेस मॉडल से बदल सकते हैं जो BLOOM जैसे बड़े मॉडल लोडिंग का समर्थन करता है!
