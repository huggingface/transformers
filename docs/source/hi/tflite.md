<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# TFLite में निर्यात करें

[TensorFlow Lite](https://www.tensorflow.org/lite/guide) एक हल्का ढांचा है जो मशीन लर्निंग मॉडल को संसाधन-सीमित उपकरणों, जैसे मोबाइल फोन, एम्बेडेड सिस्टम और इंटरनेट ऑफ थिंग्स (IoT) उपकरणों पर तैनात करने के लिए है। TFLite को इन उपकरणों पर सीमित गणनात्मक शक्ति, मेमोरी और ऊर्जा खपत के साथ मॉडल को कुशलता से ऑप्टिमाइज़ और चलाने के लिए डिज़ाइन किया गया है। एक TensorFlow Lite मॉडल को एक विशेष कुशल पोर्टेबल प्रारूप में दर्शाया जाता है जिसे `.tflite` फ़ाइल एक्सटेंशन द्वारा पहचाना जाता है।

🤗 Optimum में `exporters.tflite` मॉड्यूल के माध्यम से 🤗 Transformers मॉडल को TFLite में निर्यात करने की कार्यक्षमता है। समर्थित मॉडल आर्किटेक्चर की सूची के लिए, कृपया [🤗 Optimum दस्तावेज़](https://huggingface.co/docs/optimum/exporters/tflite/overview) देखें।

TFLite में एक मॉडल निर्यात करने के लिए, आवश्यक निर्भरताएँ स्थापित करें:

```bash
pip install optimum[exporters-tf]
```

सभी उपलब्ध तर्कों की जांच करने के लिए, [🤗 Optimum दस्तावेज़](https://huggingface.co/docs/optimum/main/en/exporters/tflite/usage_guides/export_a_model) देखें,
या कमांड लाइन में मदद देखें:

```bash
optimum-cli export tflite --help
```

यदि आप 🤗 Hub से एक मॉडल का चेकपॉइंट निर्यात करना चाहते हैं, उदाहरण के लिए, `google-bert/bert-base-uncased`, निम्नलिखित कमांड चलाएँ:

```bash
optimum-cli export tflite --model google-bert/bert-base-uncased --sequence_length 128 bert_tflite/
```

आपको प्रगति को दर्शाते हुए लॉग दिखाई देंगे और यह दिखाएंगे कि परिणामस्वरूप `model.tflite` कहाँ सहेजा गया है, जैसे:

```bash
Validating TFLite model...
	-[✓] TFLite model output names match reference model (logits)
	- Validating TFLite Model output "logits":
		-[✓] (1, 128, 30522) matches (1, 128, 30522)
		-[x] values not close enough, max diff: 5.817413330078125e-05 (atol: 1e-05)
The TensorFlow Lite export succeeded with the warning: The maximum absolute difference between the output of the reference model and the TFLite exported model is not within the set tolerance 1e-05:
- logits: max diff = 5.817413330078125e-05.
 The exported model was saved at: bert_tflite
```

उपरोक्त उदाहरण 🤗 Hub से एक चेकपॉइंट निर्यात करने को दर्शाता है। जब एक स्थानीय मॉडल निर्यात करते हैं, तो पहले सुनिश्चित करें कि आपने मॉडल के वज़न और टोकनाइज़र फ़ाइलों को एक ही निर्देशिका (`local_path`) में सहेजा है। CLI का उपयोग करते समय, चेकपॉइंट नाम के बजाय `model` तर्क में `local_path` पास करें।
