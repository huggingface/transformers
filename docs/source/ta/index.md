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

# டிரான்ஸ்ஃபார்மர்ஸ்

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

டிரான்ஸ்ஃபார்மர்ஸ் (Transformers) என்பது உரை (text), படங்கள் (computer vision), ஒலி (audio), காணொளி (video) மற்றும் பலவகைத் தரவுகளைக் (multimodal) கையாளும் அதிநவீன ஏஐ (AI) மாடல்களை உருவாக்குவதற்கான ஒரு முதன்மை அமைப்பாகும்.

இது தரவுகளைக் கொண்டு மாடல்களுக்குப் பயிற்சி அளிக்கவும் (training), புதிய தரவுகளை பகுப்பாய்வு செய்து உடனடி பதில்களைப் பெறவும் (inference) பயன்படுகிறது.

இது மாடல் அமைப்பை ஒரே மையத்தில் வரையறுப்பதால், ஒட்டுமொத்த ஏஐ தொழினுட்ப உலகமும் இந்த ஒரே மாதிரியை ஏற்றுக்கொள்கிறது. `transformers` பல்வேறு கட்டமைப்பு முறைகளுக்கு இடையே ஒரு முக்கிய இணைப்புப் புள்ளியாகச் செயல்படுகிறது. 

ஒரு மாடல் அமைப்பு இதில் ஆதரிக்கப்பட்டால், அது பெரும்பாலான பயிற்சி கட்டமைப்புகள் (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, ...), அனுமான இயங்கிகள் (vLLM, SGLang, TGI, ...), மற்றும் இதன் மாடல் அமைப்பைப் பயன்படுத்தும் தொடர்புடைய பிற நூலகங்களுடன் (llama.cpp, mlx, ...) எளிதில் பொருந்திச் செயல்படும்.

அதிநவீன ஏஐ மாடல்களின் அமைப்பை எளிமையாகவும், விருப்பத்திற்கேற்ப மாற்றக்கூடியதாகவும், அதிவேகத்திறன் கொண்டதாகவும் வடிவமைப்பதன் மூலம், புதிய மாடல்களுக்கு ஆதரவளிக்கவும், அவற்றின் பயன்பாட்டை அனைவருக்கும் சமமாக விரிவுபடுத்தவும் நாங்கள் உறுதியளிக்கிறோம்.

ஹக்கிங் ஃபேஸ் தளத்தில் [Hugging Face Hub](https://huggingface.com/models) நீங்கள் பயன்படுத்துவதற்காக 10 லட்சத்திற்கும் அதிகமான (1M+) டிரான்ஸ்ஃபார்மர்ஸ் மாடல் செக்பாயிண்ட்கள் [model checkpoints](https://huggingface.co/models?library=transformers&sort=trending) உள்ளன.

உங்களுக்கு தேவையான மாடலைக் கண்டறிந்து, டிரான்ஸ்ஃபார்மர்ஸ் உதவியுடன் உடனடியாக உங்கள் பணிகளைத் தொடங்க இன்றே ஹக்கிங் ஃபேஸ் தளத்தை [Hub](https://huggingface.com/) ஆராயுங்கள்.

டிரான்ஸ்ஃபார்மர்ஸ் அமைப்பில் உள்ள சமீபத்திய உரை (text), பார்வை (vision), ஒலி மற்றும் பன்முறைச் செய்தித் தொடர்பு (multimodal) மாடல் வடிவமைப்புகளைப் பற்றி அறிய, எங்கள் மாடல்கள் காலவரிசைப்பக்கத்தை [Models Timeline](./models_timeline) ஆராயுங்கள்.



## அம்சங்கள்

டிரான்ஸ்ஃபார்மர்ஸ், அதிநவீன முன்பயிற்சி அளிக்கப்பட்ட மாடல்களைக் (pretrained models) கொண்டு அனுமானம் (inference) அல்லது பயிற்சி (training) செய்வதற்கான அனைத்து வசதிகளையும் வழங்குகிறது. இதன் சில முக்கிய அம்சங்கள் கீழே கொடுக்கப்பட்டுள்ளன:

- [Pipeline](./pipeline_tutorial): உரை உருவாக்கம் (text generation), படப் பிரிவாக்கம் (image segmentation), தானியங்கி குரல் அறிதல் (automatic speech recognition), ஆவணக் கேள்வி-பதில் (document question answering) போன்ற பல பயன்பாடுகளுக்குப் பயன்படும் எளிமையான, அதிவேக அனுமானக் கட்டமைப்பு (inference class).

- [Trainer](./trainer): PyTorch மாடல்களின் பயிற்சி மற்றும் பகிர்ந்தளிக்கப்பட்ட பயிற்சிக்குக் (distributed training) கலப்புத் துல்லியம் (mixed precision), torch.compile, மற்றும் FlashAttention போன்ற பல வசதிகளை வழங்கும் ஒரு முழுமையான பயிற்சி அமைப்பு.

- [generate](./llm_tutorial): பெரிய மொழியியல் மாடல்கள் (LLMs) மற்றும் காட்சி மொழியியல் மாடல்கள் (VLMs) மூலம் மிக வேகமான உரை உருவாக்கம். இதில் நேரடித் தரவுப் பகிர்வு (streaming) மற்றும் பல்வேறு குறியீடிறக்க முறைகளுக்கான (decoding strategies) ஆதரவும் சேர்க்கப்பட்டுள்ளது.

## வடிவமைப்பு

> [!TIP]
> டிரான்ஸ்ஃபார்மர்ஸ் அமைப்பின் வடிவமைப்புத் தத்துவங்களைப் பற்றி மேலும் அறிய எங்கள் தத்துவக் கோட்பாடுகள் [Philosophy](./philosophy) பக்கத்தைப் படியுங்கள்.

டிரான்ஸ்ஃபார்மர்ஸ், மென்பொருள் உருவாக்குநர்கள், இயந்திரக் கற்றல் பொறியாளர்கள் மற்றும் ஆராய்ச்சியாளர்களுக்காக வடிவமைக்கப்பட்டுள்ளது. இதன் முக்கிய வடிவமைப்புத் தத்துவங்கள்:

1. வேகமானது மற்றும் எளிதானது: ஒவ்வொரு மாடலும் மூன்று முக்கிய அமைப்புகளில் இருந்து மட்டுமே உருவாக்கப்படுகிறது (configuration, model, மற்றும் preprocessor). மேலும், [`Pipeline`] அல்லது [`Trainer`] மூலம் இவற்றை அனுமானத்திற்கு (inference) அல்லது பயிற்சிக்கு (training) மிக விரைவாகப் பயன்படுத்த முடியும்.

2. முன்பயிற்சி அளிக்கப்பட்ட மாடல்கள் (Pretrained models): புதிதாக ஒரு மாடலுக்கு பயிற்சி அளிப்பதற்குப் பதிலாக, ஏற்கனவே பயிற்சி அளிக்கப்பட்ட மாடலைப் பயன்படுத்துவதன் மூலம் கார்பன் வெளியீடு, கணக்கீட்டுச் செலவு மற்றும் நேரம் ஆகியவை பெருமளவில் குறைகின்றன. ஒவ்வொரு முன்பயிற்சி அளிக்கப்பட்ட மாடலும், அதன் மூல அமைப்பிற்கு மிக நெருக்கமாகவும் அதிநவீன செயல்திறன் கொண்டதாகவும் மறுஉருவாக்கம் செய்யப்பட்டுள்ளது.


<div class="flex justify-center">
  <a target="_blank" href="https://huggingface.co/support">
      <img alt="HuggingFace Expert Acceleration Program" src="https://hf.co/datasets/huggingface/documentation-images/resolve/81d7d9201fd4ceb537fc4cebc22c29c37a2ed216/transformers/transformers-index.png" style="width: 100%; max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
  </a>
</div>

## கற்றுக்கொள்ள

நீங்கள் டிரான்ஸ்ஃபார்மர்ஸ் தொழில்நுட்பத்திற்குப் புதியவராக இருந்தாலோ அல்லது டிரான்ஸ்ஃபார்மர் மாடல்களைப் பற்றி மேலும் அறிய விரும்பினாலோ, LLM பாடத்திட்டத்திலிருந்து [LLM course](https://huggingface.co/learn/llm-course/chapter1/1?fw=pt) தொடங்குமாறு பரிந்துரைக்கிறோம். இந்த முழுமையான பாடத்திட்டம், டிரான்ஸ்ஃபார்மர் மாடல்கள் எவ்வாறு செயல்படுகின்றன என்ற அடிப்படை முதல் பல்வேறு பணிகளுக்கான நடைமுறைப் பயன்பாடுகள் வரை அனைத்தையும் உள்ளடக்கியது. உயர்தரத் தரவுத் தொகுப்புகளைப் (datasets) பராமரிப்பது முதல் பெரிய மொழியியல் மாடல்களைச் சீரமைப்பது (fine-tuning) மற்றும் பகுத்தறியும் திறன்களை (reasoning capabilities) நடைமுறைப்படுத்துவது வரையிலான முழுமையான வேலைப்பாய்வை (workflow) நீங்கள் இதில் கற்றுக்கொள்வீர்கள். இதில் கோட்பாட்டு ரீதியிலான விளக்கங்கள் மற்றும் செயல்முறைப் பயிற்சிகள் (hands-on exercises) இரண்டும் உள்ளதால், நீங்கள் படிக்கும் போதே டிரான்ஸ்ஃபார்மர் மாடல்களைப் பற்றிய வலுவான அடிப்படை அறிவை வளர்த்துக் கொள்ள முடியும்.
