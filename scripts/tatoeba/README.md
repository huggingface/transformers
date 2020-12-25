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

Setup transformers following instructions in README.md, (I would fork first).
```bash
git clone git@github.com:huggingface/transformers.git
cd transformers
pip install -e .
pip install pandas GitPython wget
```

Get required metadata
```
curl https://cdn-datasets.huggingface.co/language_codes/language-codes-3b2.csv  > language-codes-3b2.csv
curl https://cdn-datasets.huggingface.co/language_codes/iso-639-3.csv > iso-639-3.csv
```

Install Tatoeba-Challenge repo inside transformers
```bash
git clone git@github.com:Helsinki-NLP/Tatoeba-Challenge.git
```

To convert a few models, call the conversion script from command line:
```bash
python src/transformers/models/marian/convert_marian_tatoeba_to_pytorch.py --models heb-eng eng-heb --save_dir converted
```

To convert lots of models you can pass your list of Tatoeba model names to `resolver.convert_models` in a python client or script.

```python
from transformers.convert_marian_tatoeba_to_pytorch import TatoebaConverter
resolver = TatoebaConverter(save_dir='converted')
resolver.convert_models(['heb-eng', 'eng-heb'])
```


### Upload converted models
Since version v3.5.0, the model sharing workflow is switched to git-based system . Refer to [model sharing doc](https://huggingface.co/transformers/master/model_sharing.html#model-sharing-and-uploading) for more details.

To upload all converted models, 

1. Install [git-lfs](https://git-lfs.github.com/).

2. Login to `transformers-cli`

```bash
transformers-cli login
```

3. Run the `upload_models` script

```bash
./scripts/tatoeba/upload_models.sh
```


### Modifications
- To change naming logic, change the code near `os.rename`. The model card creation code may also need to change.
- To change model card content, you must modify `TatoebaCodeResolver.write_model_card`
