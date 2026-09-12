<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# விரைவாக தொடங்க (Quickstart)

[[open-in-colab]]

டிரான்ஸ்ஃபார்மர்ஸ், அனைவரும் எளிதாகவும் வேகமாகவும் டிரான்ஸ்ஃபார்மர் மாடல்களைப் பற்றிக் கற்றுக்கொள்ளவும் அல்லது உருவாக்கவும் ஏற்ற வகையில் வடிவமைக்கப்பட்டுள்ளது.

பயனர்களுக்கான முதன்மை அமைப்புகள் (abstractions) மூன்று முக்கிய வகுப்புகளாகவும் (classes), அனுமானம் (inference) அல்லது பயிற்சிக்கான (training) இரண்டு ஏபிஐ-களாகவும் (APIs) சுருக்கப்பட்டுள்ளன. இந்த விரைவுத் தொடக்கம் டிரான்ஸ்ஃபார்மர்ஸின் முக்கிய அம்சங்களை அறிமுகப்படுத்துவதுடன், பின்வருவனவற்றை எவ்வாறு செய்வது என்பதையும் காட்டுகிறது:

- முன்பயிற்சி அளிக்கப்பட்ட மாடலைப் (pretrained model) பதிவேற்றுவது
- [`Pipeline`] மூலம் அனுமானத்தை இயக்குவது
- [`Trainer`] மூலம் மாடலைச் சீரமைப்பது (fine-tune)

## அமைப்பு முறை (Set up)

தொடங்குவதற்கு, ஒரு ஹக்கிங் ஃபேஸ் (Hugging Face) [கணக்கை](https://hf.co/join) உருவாக்கப் பரிந்துரைக்கிறோம். ஒரு கணக்கு உங்களுக்கு பதிப்பு-கட்டுப்படுத்தப்பட்ட (version controlled) மாடல்கள், தரவுத் தொகுப்புகள் (datasets) மற்றும் [Spaces](https://hf.co/spaces) ஆகியவற்றை [ஹக்கிங் ஃபேஸ் ஹப்பில்](https://hf.co/docs/hub/index) பதிவேற்றவும் அணுகவும் உதவுகிறது.

ஒரு பயனர் அணுகல் டோக்கனை [User Access Token](https://hf.co/docs/hub/security-tokens#user-access-tokens) உருவாக்கி உங்கள் கணக்கில் உள்நுழையவும் (log in).

<hfoptions id="authenticate">
<hfoption id="notebook">

உள்நுழையக் கேட்கப்படும் போது [`~huggingface_hub.notebook_login`]-இல் உங்கள் பயனர் அணுகல் டோக்கனை ஒட்டவும் (paste).

```py
from huggingface_hub import notebook_login

notebook_login()
```

</hfoption>
<hfoption id="CLI">

[huggingface_hub[cli]](https://huggingface.co/docs/huggingface_hub/guides/cli#getting-started) தொகுப்பு நிறுவப்பட்டுள்ளதா என்பதை உறுதிசெய்து, கீழே உள்ள கட்டளையை இயக்கவும். உள்நுழையக் கேட்கப்படும் போது உங்கள் பயனர் அணுகல் டோக்கனை ஒட்டவும்.

```bash
hf auth login
```

</hfoption>
</hfoptions>

PyTorch-ஐ நிறுவவும்.

```bash
# கட்டளை முனையத்தில் (CLI) நிறுவினால் '!' குறியீட்டை நீக்கிவிடவும்
!pip install torch
```

பின்னர், தரவுத் தொகுப்புகளை அணுகுதல், கணினிப் பார்வை (vision) மாடல்களைப் பயன்படுத்துதல், பயிற்சியின் முடிவுகளை மதிப்பிடுதல் மற்றும் பெரிய மாடல்களுக்கான பயிற்சியை உகந்ததாக்குதல் (optimizing) போன்ற பணிகளுக்காகத் டிரான்ஸ்ஃபார்மர்ஸின் புதுப்பித்த பதிப்பையும், ஹக்கிங் ஃபேஸ் சூழலின் பிற துணை நூலகங்களையும் நிறுவவும்.

```bash
# கட்டளை முனையத்தில் (CLI) நிறுவினால் '!' குறியீட்டை நீக்கிவிடவும்
!pip install -U transformers datasets evaluate accelerate timm
```

## ஏஜென்ட் திறன்கள் (Agent skills)

ஹக்கிங் ஃபேஸ் திறன்கள் [Hugging Face Skills](https://github.com/huggingface/skills), டிரான்ஸ்ஃபார்மர்ஸ் நூலகத்துடன் எப்படி வேலை செய்வது என்பதை உங்கள் நிரலாக்க ஏஜென்ட்டுகளுக்குக் (coding agent) கற்றுத் தருகின்றன. நீங்களே சொந்தமாகப் பயிற்சி நிரலை (training script) எழுதுவதற்குப் பதிலாக, ஒரு விஷன் மாடலைச் சீரமைக்க (fine-tune) உங்கள் ஏஜென்ட்டிடம் கேட்டு, இந்தத் திறன்களைப் பயன்படுத்தி வழிகாட்டச் செய்யலாம்.

<hfoptions id="install">
<hfoption id="Claude Code">

```sh
/plugin marketplace add huggingface/skills
/plugin install hf-cli@huggingface/skills
```

</hfoption>
<hfoption id="Codex">

```sh
codex plugin marketplace add huggingface/skills
```

பின்னர் Codex-இல் `/plugins` என இயக்கி, `huggingface/skills` மூலக் களஞ்சியத்திலிருந்து (repository) உங்களுக்குத் தேவையான திறனை நிறுவவும்.


</hfoption>
</hfoptions>

Cursor மற்றும் Gemini போன்ற பிற ஆதரிக்கப்படும் தளங்களுக்கு அதன் நிறுவல் வழிகாட்டியைப்  [Installation Guide](https://github.com/huggingface/skills#installation) பார்க்கவும்.

ஒரு திறன் நிறுவப்பட்டதும், அதை உங்கள் நிரலாக்க ஏஜென்ட்டுக்கான அறிவுறுத்தல்களில் சேர்க்கலாம்:

```md
கூடைப்பந்து வீரர்களைக் கண்காணிப்பதற்காக (basketball player tracking) [Lekim89/sportsmot](https://huggingface.co/datasets/Lekim89/sportsmot) தரவுத்தளத்தில் RT-DETRv2 மாடலைச் சீரமைக்க HF Trainer திறனைப் பயன்படுத்தவும்.
```

## முன்பயிற்சி அளிக்கப்பட்ட மாடல்கள் (Pretrained models)

ஒவ்வொரு முன்பயிற்சி அளிக்கப்பட்ட மாடலும் மூன்று அடிப்படை வகுப்புகளிலிருந்து (base classes) பெறப்படுகிறது:

| **வகுப்பு (Class)** | **விளக்கம்** |
|---|---|
| [`PreTrainedConfig`] | கவனத் தலைகளின் எண்ணிக்கை (number of attention heads) அல்லது சொற்களஞ்சிய அளவு (vocabulary size) போன்ற ஒரு மாடலின் பண்புகளைக் குறிப்பிடும் ஒரு கோப்பு. |
| [`PreTrainedModel`] |அமைப்புக் கோப்பில் உள்ள மாடல் பண்புகளால் வரையறுக்கப்பட்ட ஒரு மாடல் (அல்லது வடிவமைப்பு). முன்பயிற்சி அளிக்கப்பட்ட மாடல் மூல மறைமுக நிலைகளை (raw hidden states) மட்டுமே வழங்கும். ஒரு குறிப்பிட்ட பணிக்கு, இந்த மூல மறைமுக நிலைகளைப் பயனுள்ள முடிவாக மாற்ற பொருத்தமான 'Model Head'-ஐப் பயன்படுத்தவும் (எடுத்துக்காட்டாக, [`LlamaModel`]-க்கு எதிராக [`LlamaForCausalLM`]). |
| Preprocessor | மூல உள்ளீடுகளை (உரை, படங்கள், ஆடியோ, பன்முறைத் தரவு) மாடலுக்கான எண் வடிவ உள்ளீடுகளாக (numerical inputs) மாற்றுவதற்கான ஒரு வகுப்பு. எடுத்துக்காட்டாக, [`PreTrainedTokenizer`] உரையை டென்சர்களாகவும் (tensors), [`ImageProcessingMixin`] பிக்சல்களை டென்சர்களாகவும் மாற்றுகின்றன. |

மாடல்களையும் தரவுத் தயாரிப்பு வகுப்புகளையும் (preprocessors) பதிவேற்ற [AutoClass](./model_doc/auto) API-ஐப் பயன்படுத்தப் பரிந்துரைக்கிறோம். ஏனெனில் இது முன்பயிற்சி அளிக்கப்பட்ட எடைகள் (pretrained weights) மற்றும் அமைப்புக் கோப்பின் பெயர் அல்லது பாதையின் அடிப்படையில் ஒவ்வொரு பணிக்கும் மற்றும் இயந்திரக் கற்றல் கட்டமைப்புக்கும் ஏற்ற சரியான வடிவமைப்பைத் தானாகவே கண்டறியும்.

ஹப்பில் உள்ள எடைகளையும் அமைப்புக் கோப்பையும் மாடல் மற்றும் தரவுத் தயாரிப்பு வகுப்பில் பதிவேற்ற [`~PreTrainedModel.from_pretrained`]-ஐப் பயன்படுத்தவும்.

ஒரு மாடலைப் பதிவேற்றும் போது, அது மிகச் சிறந்த முறையில் இயங்குவதை உறுதிசெய்ய பின்வரும் அளபுருக்களைக் (parameters) கட்டமைக்கவும்:

- `device_map="auto"` மாடலின் எடைகளை உங்கள் அதிவேக சாதனத்திற்குத் (GPU போன்றவை) தானாகவே முதலில் ஒதுக்கும்.
- `dtype="auto"` மாடலின் எடைகள் எந்தத் தரவு வகையில் சேமிக்கப்பட்டுள்ளதோ, அதே தரவு வகையில் நேரடியாகத் தொடங்கும். இது எடைகளை இருமுறை பதிவேற்றுவதைத் தவிர்க்க உதவும் (PyTorch இயல்பாகவே எடைகளை `torch.float32`-இல் பதிவேற்றும்).

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
```
உரையை டோக்கன்களாக (tokens) மாற்றி, டோக்கனைசர் (tokenizer) மூலம் PyTorch டென்சர்களாகப் பெறுங்கள். மாடல் அவற்றைச் செயல்படுத்தி அனுமானம் (inference) செய்வதற்கு முன்பாக, இந்த டோக்கன் உள்ளீடுகளை மாடல் இருக்கும் அதே சாதனத்திற்கு (device) மாற்றவும்.

```py
model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to(model.device)
```

இப்போது மாடல் அனுமானம் அல்லது பயிற்சிக்குத் தயாராக உள்ளது.

அனுமானத்திற்கு, உரையை உருவாக்கத் டோக்கனாக்கப்பட்ட உள்ளீடுகளை [`~GenerationMixin.generate`]-க்கு அனுப்பவும். உருவாக்கப்படும் டோக்கன் ஐடிகளை (token ids) மீண்டும் உரையாக மாற்ற [`~PreTrainedTokenizerBase.batch_decode`]-ஐப் பயன்படுத்தவும்.


```py
generated_ids = model.generate(**model_inputs, max_length=30)
tokenizer.batch_decode(generated_ids)[0]
'<s> The secret to baking a good cake is 100% in the preparation. There are so many recipes out there,'
```

> [!TIP]
> மாடலை எப்படிச் சீரமைப்பது (fine-tune) என்பதைக் கற்றுக்கொள்ள [Trainer](#ட்ரெய்னர்-trainer)) பகுதிக்குச் செல்லவும்.

## பைப்லைன் (Pipeline)

முன்பயிற்சி அளிக்கப்பட்ட ஒரு மாடலின் மூலம் அனுமானத்தை இயக்குவதற்கு [`Pipeline`] வகுப்பு மிகவும் வசதியான வழியாகும். இது உரை உருவாக்கம் (text generation), படப் பிரிவாக்கம் (image segmentation), தானியங்கி குரல் அறிதல் (automatic speech recognition), ஆவணக் கேள்வி-பதில் (document question answering) போன்ற பல பயன்பாடுகளுக்குப் பயன்படுகிறது.

> [!TIP]
> பயன்பாடுகளின் முழுமையான பட்டியலுக்கு [Pipeline](./main_classes/pipelines) குறிப்புகளைப் பார்க்கவும்.

ஒரு [`Pipeline`] பொருளை (object) உருவாக்கி, ஒரு குறிப்பிட்ட பணியைத் தேர்ந்தெடுக்கவும். இயல்பாகவே, [`Pipeline`] தேர்ந்தெடுக்கப்பட்ட பணிக்கு உரிய முன்பயிற்சி அளிக்கப்பட்ட மாடலைப் பதிவிறக்கம் செய்து சேமிக்கும். ஒரு குறிப்பிட்ட மாடலைத் தேர்ந்தெடுக்க model அளபுருக்களைக்கு (parameter) அந்த மாடலின் பெயரைக் கொடுக்கவும்.

<hfoptions id="pipeline-tasks">
<hfoption id="text generation">

அனுமானத்திற்கான வேகப்படுத்தியை (accelerator) தானாகவே கண்டறிய [`Accelerator`]-ஐப் பயன்படுத்தவும்.

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf", device=device)
```

கூடுதல் உரையை உருவாக்கத் தொடக்க உரையை [`Pipeline`]-க்கு உள்ளீடாக வழங்கவும்.

```py
pipe("The secret to baking a good cake is ", max_length=50)
[{'generated_text': 'The secret to baking a good cake is 100% in the batter. The secret to a great cake is the icing.\nThis is why we’ve created the best buttercream frosting reci'}]
```

</hfoption>
<hfoption id="image segmentation">

அனுமானத்திற்கான வேகப்படுத்தியை (accelerator) தானாகவே கண்டறிய [`Accelerator`]-ஐப் பயன்படுத்தவும்.

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipeline = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic", device=device)
```
ஒரு படத்தை (படத்தின் URL அல்லது கணினியில் உள்ள பாதை) [`Pipeline`]-க்கு உள்ளீடாக அனுப்பவும்.

<div class="flex justify-center">
   <img src="https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"/>
</div>

```py
segments = pipeline("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
segments[0]["label"]
'bird'
segments[1]["label"]
'bird'
```

</hfoption>
<hfoption id="automatic speech recognition">

அனுமானத்திற்கான வேகப்படுத்தியை (accelerator) தானாகவே கண்டறிய [`Accelerator`]-ஐப் பயன்படுத்தவும்.

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=device)
```

Pass an audio file to [`Pipeline`].

```py
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
{'text': ' He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flour-fatten sauce.'}
```

</hfoption>
</hfoptions>

## ட்ரெய்னர் (Trainer)

[`Trainer`] என்பது PyTorch மாடல்களுக்கான முழுமையான பயிற்சி மற்றும் மதிப்பீட்டுச் சுழற்சியாகும் (training and evaluation loop). கையால் பயிற்சிச் சுழற்சியை எழுதும்போது ஏற்படும் பல மீண்டும் மீண்டும் எழுதப்படும் குறியீடுகளை (boilerplate code) இது தவிர்க்க உதவுகிறது. இதனால் நீங்கள் மிக விரைவாகப் பயிற்சியைத் தொடங்குவதுடன், அமைப்புகளை வடிவமைப்பதிலும் கவனம் செலுத்த முடியும். ஒரு மாடல், தரவுத் தொகுதி (dataset), தரவுத் தயாரிப்பு வகுப்பு (preprocessor) மற்றும் தரவுத் தொகுப்பிலிருந்து தொகுப்புகளைப் (batches) பிரிக்கும் தரவுத் தொகுப்பி (data collator) மட்டுமே உங்களுக்குத் தேவைப்படும்.

பயிற்சிப் ሂደையைத் தேவைக்கேற்ப மாற்ற [`TrainingArguments`] வகுப்பைப் பயன்படுத்தவும். இது பயிற்சி மற்றும் மதிப்பீட்டிற்குப் பல தேர்வுகளை வழங்கி உதவுகிறது. உங்கள் தேவைக்கு ஏற்ப, பேட்ச் அளவு (batch size), கற்றல் விகிதம் (learning rate), கலப்புத் துல்லியம் (mixed precision), torch.compile போன்ற பல உயர் அளபுருக்களைக் (hyperparameters) அம்சங்களையும் சோதித்துப் பார்க்கலாம். அல்லது ஒரு அடிப்படை முடிவைப் (baseline) பெற இயல்பான பயிற்சி அளபுருக்களையும் பயன்படுத்தலாம்.

பயிற்சிக்காக மாடல், டோக்கனைசர் மற்றும் தரவுத் தொகுப்பைப் பதிவேற்றவும்.

```py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
dataset = load_dataset("rotten_tomatoes")
```

உரையைத் டோக்கன்களாக மாற்றி, PyTorch டென்சர்களாக மாற்றும் ஒரு செயல்பாட்டை உருவாக்கவும். [`~datasets.Dataset.map`] முறையைப் பயன்படுத்தி முழு தரவுத் தொகுப்பிலும் இந்தச் செயல்பாட்டைப் பயன்படுத்தவும்.


```py
def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])
dataset = dataset.map(tokenize_dataset, batched=True)
```

தரவுகளின் தொகுப்புகளை உருவாக்க ஒரு தரவுத் தொகுப்பியைப் (data collator) பதிவேற்றி, அதில் டோக்கனைசரை அனுப்பவும்.

```py
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

அடுத்து, பயிற்சி அம்சங்கள் மற்றும் உயர் அளபுருக்களைக் (hyperparameters) [`TrainingArguments`]-ஐ அமைக்கவும்.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="distilbert-rotten-tomatoes",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    push_to_hub=True,
)
```

இறுதியாக, இந்தத் தனித்தனி பகுதிகள் அனைத்தையும் [`Trainer`]-க்கு அனுப்பி, பயிற்சியைத் தொடங்க [`~Trainer.train`]-ஐ அழைக்கவும்.

```py
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()
```

[`~Trainer.push_to_hub`] மூலம் உங்கள் மாடலையும் டோக்கனைசரையும் ஹப்பிற்குப் (Hub) பகிரவும்.

```py
trainer.push_to_hub()
```

வாழ்த்துகள்! டிரான்ஸ்ஃபார்மர்ஸ் மூலம் உங்கள் முதல் மாடலுக்கு வெற்றிகரமாகப் பயிற்சி அளித்துவிட்டீர்கள்!

## அடுத்தப் படிகள் (Next steps)

டிரான்ஸ்ஃபார்மர்ஸ் பற்றியும் அது வழங்கும் வசதிகள் பற்றியும் இப்போது நன்கு புரிந்து கொண்டதால், உங்களுக்கு மிகவும் ஆர்வமுள்ளவற்றை ஆராய்ந்து கற்கத் தொடங்கலாம்.

- **அடிப்படை வகுப்புகள் (Base classes)**: அமைப்புகள் (configuration), மாடல் மற்றும் தயாரிப்பு வகுப்புகளைப் பற்றி மேலும் அறியவும். இது மாடல்களை எவ்வாறு உருவாக்குவது மற்றும் மாற்றி அமைப்பது, பல்வேறு வகையான உள்ளீடுகளைச் (ஆடியோ, படங்கள், பன்முறைத் தரவு) செயல்படுத்துவது மற்றும் உங்கள் மாடலை எவ்வாறு பகிர்வது என்பதைப் புரிந்துகொள்ள உதவும்.

- **அனுமானம் (Inference)**: [Pipeline], அனுமானம் மற்றும் LLM-களுடன் உரையாடுவது, ஏஜென்ட்டுகள் மற்றும் உங்கள் இயந்திரக் கற்றல் கட்டமைப்பு மற்றும் வன்பொருளுக்கு ஏற்ப அனுமானத்தை எவ்வாறு உகந்ததாக்குவது என்பதை மேலும் ஆராயுங்கள்.

- **பயிற்சி (Training)**: [Trainer], பகிர்ந்தளிக்கப்பட்ட பயிற்சி (distributed training) மற்றும் குறிப்பிட்ட வன்பொருளில் பயிற்சியை உகந்ததாக்குதல் பற்றி விரிவாகப் படியுங்கள்.

- **குவாண்டாசேஷன் (Quantization**): குவாண்டாசேஷன் மூலம் நினைவகம் மற்றும் சேமிப்பகத் தேவைகளைக் குறைத்து, குறைந்த பிட்களில் எடைகளைக் குறிப்பதன் மூலம் அனுமானத்தை வேகப்படுத்துங்கள்.

- **வளங்கள் (Resources)**: ஒரு குறிப்பிட்ட பணிக்கு ஒரு மாடலைக் கொண்டு எவ்வாறு பயிற்சி அளிப்பது மற்றும் அனுமானம் செய்வது என்பதற்கான முழுமையான வழிகாட்டிகளைத் தேடுகிறீர்களா? பணி வழிகாட்டிகளைப் (task recipes) பார்க்கவும்!
