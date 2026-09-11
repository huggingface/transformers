<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# நிறுவல் (Installation)
டிரான்ஸ்ஃபார்மர்ஸ் (Transformers) PyTorch தொழில்நுட்பத்துடன் இணைந்து செயல்படுகிறது. இது Python 3.10+ மற்றும் PyTorch 2.5+ ஆகிய பதிப்புகளில் பரிசோதிக்கப்பட்டு உறுதிசெய்யப்பட்டுள்ளது.

## மெய்நிகர் சூழல் (Virtual Environment)

[uv](https://docs.astral.sh/uv/) என்பது ரஸ்ட் (Rust) மொழியில் எழுதப்பட்ட மிக வேகமான Python தொகுப்பு மற்றும் திட்ட மேலாளர் (package and project manager) ஆகும். இது வெவ்வேறு திட்டங்களை தனித்தனியாக நிர்வகிக்கவும், தேவைகளுக்கு இடையேயான இணக்கத்தன்மை சிக்கல்களைத் தவிர்க்கவும் இயல்பாகவே ஒரு மெய்நிகர் சூழலை [virtual environment](https://docs.astral.sh/uv/pip/environments/)  கோருகிறது.

இதனை [pip](https://pip.pypa.io/en/stable/)-க்கு  நேரடிப் மாற்றாகப் பயன்படுத்தலாம். ஒருவேளை நீங்கள் pip-ஐப் பயன்படுத்த விரும்பினால், கீழே உள்ள கட்டளைகளில் இருந்து `uv` என்பதை நீக்கிவிடலாம்.

> [!TIP]
> uv-ஐ நிறுவ, அதன் நிறுவல் ஆவணங்களைப் [installation](https://docs.astral.sh/uv/guides/install-python/) பாருங்கள்.

டிரான்ஸ்ஃபார்மர்ஸை நிறுவுவதற்கு ஒரு மெய்நிகர் சூழலை (virtual environment) உருவாக்கவும்.

```bash
uv venv .env
source .env/bin/activate
```

## பைதான் (Python)
கீழே உள்ள கட்டளையைப் பயன்படுத்தி டிரான்ஸ்ஃபார்மர்ஸை நிறுவவும்.

[uv](https://docs.astral.sh/uv/) என்பது ரஸ்ட் (Rust) மொழியில் அடிப்படையாகக் கொண்டு செயல்படும் மிக வேகமான Python தொகுப்பு மற்றும் திட்ட மேலாளர் ஆகும்.


<hfoptions id="installation">
<hfoption id="CUDA">

NVIDIA GPU (CUDA) உடன் டிரான்ஸ்ஃபார்மர்ஸை PyTorch வழியே நிறுவ, PyTorch தளத்திற்கு ஏற்ற CUDA இயக்கிகளை [PyTorch](https://pytorch.org/get-started/locally) நிறுவவும்.

உங்கள் கணினியில் NVIDIA GPU சரியாகக் கண்டறியப்படுகிறதா என்பதைச் சரிபார்க்க கீழே உள்ள கட்டளையை இயக்கவும்.


```bash
nvidia-smi
```

```bash
uv pip install "transformers[torch]"
```

</hfoption>
<hfoption id="NVIDIA Spark (ARM64)">

ARM64 அமைப்பில் இயங்கும் NVIDIA Spark சாதனங்களில் (RTX Spark லேப்டாப் போன்றவை) PyTorch உடன் டிரான்ஸ்ஃபார்மர்ஸை நிறுவ, PyTorch-ஐ NVIDIA PyPI கோப்பகத்திலிருந்து (index) நிறுவவும். இந்தச் சாதனங்களுக்கு NVIDIA வழங்கும் ARM64 PyTorch பதிப்புகள் தேவைப்படும். இவை இயல்பான PyPI கோப்பகத்திலோ அல்லது வழக்கமான PyTorch விளிம்பு கோப்புகளிலோ (wheel index) கிடைக்காது.

உங்கள் கணினியில் NVIDIA GPU சரியாகக் கண்டறியப்படுகிறதா என்பதைச் சரிபார்க்க கீழே உள்ள கட்டளையை இயக்கவும்.

```bash
nvidia-smi
```

NVIDIA PyPI கோப்பகத்திலிருந்து PyTorch-ஐ நிறுவிய பின், டிரான்ஸ்ஃபார்மர்ஸை நிறுவவும்.

```bash
uv pip install torch --index-url https://pypi.nvidia.com
uv pip install transformers
```

</hfoption>
<hfoption id="CPU">

CPU-மட்டும் பயன்படுத்தும் (CPU-only) டிரான்ஸ்ஃபார்மர்ஸ் பதிப்பை நிறுவ, கீழே உள்ள கட்டளையை இயக்கவும்.

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers
```

</hfoption>
<hfoption id="Intel GPU (XPU)">

Intel GPU (XPU)-க்கான PyTorch உடன் டிரான்ஸ்ஃபார்மர்ஸை நிறுவ, ஏற்ற [Intel GPU (XPU) PyTorch இயக்கிகளை](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-13.html) நிறுவி, Intel GPU (XPU) PyTorch கோப்பகத்தின் (index) முகவரியைச் சேர்க்கவும்.

இயக்கிகளை நிறுவிய பின், [உங்கள் கணினியில் Intel GPU கண்டறியப்படுகிறதா என்பதைச் சரிபார்க்க](https://dgpu-docs.intel.com/driver/verification.html). கீழே உள்ள கட்டளையை இயக்கவும். 

```bash
xpu-smi
```

```bash
uv pip install "transformers[torch]" --extra-index-url https://download.pytorch.org/whl/xpu
```

</hfoption>
</hfoptions>

கீழே உள்ள கட்டளையைப் பயன்படுத்தி நிறுவல் வெற்றிகரமாக முடிந்ததுதானா என்பதைச் சோதிக்கவும். இது கொடுக்கப்பட்ட உரைக்கான லேபிள் (label) மற்றும் மதிப்பீட்டைப் (score) பதிலாகத் தர வேண்டும்.

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))"
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

### மூலம் வழியாக நிறுவுதல் (Source Install)

மூலக் குறியீடு (source) மூலம் நிறுவுவது, மென்பொருள் நூலகத்தின் நிலையான *(stable)* பதிப்பிற்குப் பதிலாக அதன் சமீபத்திய *(latest)* பதிப்பை நிறுவும். இது டிரான்ஸ்ஃபார்மர்ஸின் மிகச் சமீபத்திய மாற்றங்களை நீங்கள் வைத்திருப்பதை உறுதிசெய்வதுடன், புதிய அம்சங்களைப் பரிசோதிக்கவும், நிலையான பதிப்பில் இன்னும் அதிகாரப்பூர்வமாக வெளியிடப்படாத பிழைகளைச் (bugs) சரிசெய்யவும் பயனுள்ளதாக இருக்கும்.

இதன் ஒரே பின்னடைவு என்னவென்றால், இந்தச் சமீபத்திய பதிப்பு எப்போதும் நிலையானதாக இல்லாமல் போகலாம். இதில் ஏதேனும் சிக்கல்களைச் சந்தித்தால், நாங்கள் அதை விரைவில் சரிசெய்ய ஒரு GitHub பிழை அறிக்கையை [GitHub Issue](https://github.com/huggingface/transformers/issues) தொடங்கவும்.

மூலக் குறியீடு மூலம் நிறுவ கீழே உள்ள கட்டளையைப் பயன்படுத்தவும்.

```bash
uv pip install git+https://github.com/huggingface/transformers
```

கீழே உள்ள கட்டளையைக் கொண்டு நிறுவல் வெற்றிகரமாக அமைந்துள்ளதா என்பதைச் சரிபார்க்கவும். இது கொடுக்கப்பட்ட உரைக்கான பிரிவு (label) மற்றும் துல்லிய மதிப்பீட்டைப் (score) பதிலாகத் தர வேண்டும்.

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))"
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

### மாற்றியமைக்கக்கூடிய நிறுவல் (Editable Install)

நீங்கள் டிரான்ஸ்ஃபார்மர்ஸ் அமைப்பில் உள் கணினியிலேயே (locally) புதிய மாற்றங்களை உருவாக்கிக் கொண்டிருக்கிறீர்கள் என்றால், மாற்றியமைக்கக்கூடிய நிறுவல் [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) பயனுள்ளதாக இருக்கும். இது கோப்புகளை நகலெடுப்பதற்குப் பதிலாக, உங்கள் கணினியில் உள்ள டிரான்ஸ்ஃபார்மர்ஸ் பிரதியை அதன் மூலக் களஞ்சியத்துடன் [repository](https://github.com/huggingface/transformers) நேரடியாக இணைக்கிறது. இந்த கோப்புகள் Python-ன் 'import path'-இல் சேர்க்கப்படும்.

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
uv pip install -e .
```

> [!WARNING]
> டிரான்ஸ்ஃபார்மர்ஸைத் தொடர்ந்து பயன்படுத்த, உங்கள் கணினியில் உள்ள டிரான்ஸ்ஃபார்மர்ஸ் கோப்புறையை (folder) நீக்காமல் வைத்திருக்க வேண்டும்.

முதன்மைக் களஞ்சியத்தில் (main repository) உருவாக்கப்பட்டுள்ள சமீபத்திய மாற்றங்களுடன் உங்கள் கணினியில் உள்ள டிரான்ஸ்ஃபார்மர்ஸ் பதிப்பைப் புதுப்பிக்க கீழே உள்ள கட்டளையைப் பயன்படுத்தவும்.

```bash
cd ~/transformers/
git pull
```

## கோண்டா (conda)
[conda](https://docs.conda.io/projects/conda/en/stable/#) என்பது பல்வேறு நிரலாக்க மொழிகளுக்கும் பொதுவான ஒரு தொகுப்பு மேலாளர் (language-agnostic package manager) ஆகும். புதிதாக உருவாக்கப்பட்ட உங்கள் மெய்நிகர் சூழலில் [conda-forge](https://anaconda.org/conda-forge/transformers) தளம் வழியாக டிரான்ஸ்ஃபார்மர்ஸை நிறுவலாம்.


```bash
conda install conda-forge::transformers
```

## அமைப்பு முறை (Set up)
நிறுவலுக்குப் பிறகு, டிரான்ஸ்ஃபார்மர்ஸ் தற்காலிக சேமிப்பகத்தின் (cache) இருப்பிடத்தை நீங்கள் மாற்றியமைக்கலாம் அல்லது இணைய இணைப்பு இல்லாத நிலையிலும் (offline) பயன்படுத்தும் வகையில் இந்த நூலகத்தை அமைக்கலாம்.

### தற்காலிக சேமிப்பக முகவரி (Cache directory)

[`~PreTrainedModel.from_pretrained`] செயல்பாட்டின் மூலம் ஒரு முன்பயிற்சி அளிக்கப்பட்ட மாடலை நீங்கள் பதிவேற்றும்போது, அந்த மாடல் ஹப்பில் (Hub) இருந்து பதிவிறக்கம் செய்யப்பட்டு உங்கள் கணினியில் சேமிக்கப்படும் (cached).

ஒவ்வொரு முறையும் நீங்கள் ஒரு மாடலைத் திறக்கும்போது, சேமிப்பகத்தில் உள்ள மாடல் புதுப்பித்த நிலையில் உள்ளதா என்று அது சரிபார்க்கும். அதுவே புதுப்பித்த நிலையில் இருந்தால், கணினியில் உள்ள மாடலே பயன்பாட்டிற்கு வரும். அதில் மாற்றம் ஏதேனும் இருந்தால், புதிய மாடல் பதிவிறக்கம் செய்யப்பட்டுப் புதுப்பிக்கப்படும்.

முனையத்தின் சூழல் மாறியான (shell environment variable) `HF_HUB_CACHE` மூலம் அமைக்கப்படும் இயல்பான முகவரி `~/.cache/huggingface/hub` ஆகும். விண்டோஸ் (Windows) கணினிகளில் இதன் இயல்பான முகவரி `C:\Users\username\.cache\huggingface\hub` ஆகும்.

கீழே கொடுக்கப்பட்டுள்ள சூழல் மாறிகளின் (முன்னுரிமை வரிசைப்படி) பாதையை மாற்றுவதன் மூலம், மாடலை வேறு முகவரியில் சேமிக்க முடியும்.

1. [HF_HUB_CACHE](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#hfhubcache)  (இயல்பானது)
2. [HF_HOME](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#hfhome)
3. [XDG_CACHE_HOME](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#xdgcachehome) + `/huggingface` (HF_HOME அமைக்கப்படாத போது மட்டும்)

### இணையமில்லா பயன்முறை (Offline mode)

இணைய வசதியற்ற (offline) அல்லது ஃபயர்வால் (firewall) பாதுகாக்கப்பட்ட சூழலில் டிரான்ஸ்ஃபார்மர்ஸைப் பயன்படுத்த, தேவையான கோப்புகளை முன்னதாகவே பதிவிறக்கம் செய்து சேமிப்பகத்தில் (cache) வைத்திருக்க வேண்டும். [`~huggingface_hub.snapshot_download`] முறையைப் பயன்படுத்தி ஹப்பிலிருந்து (Hub) ஒரு மாடலின் மூலக் களஞ்சியத்தை (repository) நீங்கள் பதிவிறக்கம் செய்யலாம்.

> [!TIP]
> ஹப்பிலிருந்து கோப்புகளைப் பதிவிறக்குவது பற்றிய கூடுதல் விருப்பங்களை அறிய [Download files from the Hub](https://hf.co/docs/huggingface_hub/guides/download) வழிகாட்டியைப் பார்க்கவும். குறிப்பிட்ட பதிப்புகளில் (revisions) இருந்தும், CLI வழியாகவும் நீங்கள் கோப்புகளைப் பதிவிறக்க முடியும். மேலும், ஒரு களஞ்சியத்திலிருந்து எந்தெந்த கோப்புகள் வேண்டும் என்பதை வடிகட்டியும் (filter) பதிவிறக்கம் செய்ய முடியும்.

```py
from huggingface_hub import snapshot_download

snapshot_download(repo_id="meta-llama/Llama-2-7b-hf", repo_type="model")
```

மாடலைப் பதிவேற்றும்போது ஹப்பிற்கு (Hub) செல்லும் HTTP அழைப்புகளைத் (HTTP calls) தவிர்க்க `HF_HUB_OFFLINE=1` என்ற சூழல் மாறியை அமைக்கவும்.

```bash
HF_HUB_OFFLINE=1 \
python examples/pytorch/language-modeling/run_clm.py --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name wikitext ...
```

ஏற்கனவே சேமிக்கப்பட்ட (cached) கோப்புகளை மட்டும் பயன்படுத்துவதற்கான மற்றொரு வழி, [`~PreTrainedModel.from_pretrained`]-இல் `local_files_only=True` என அமைப்பது ஆகும்.

```py
from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("./path/to/local/directory", local_files_only=True)
```
