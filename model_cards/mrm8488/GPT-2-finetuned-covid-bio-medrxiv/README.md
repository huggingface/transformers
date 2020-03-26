---
language: english
thumbnail:
---

# GPT-2 + bio/medrxiv files from CORD19: 🦠 ✍ ⚕

**GPT-2** fine-tuned on **biorxiv_medrxiv** files from [CORD-19](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) dataset.


## Datasets details:

| Dataset                | # Files |
| ---------------------- | ----- |
| biorxiv_medrxiv        | 885  |


## Model training:

The model was trained on a Tesla P100 GPU and 25GB of RAM with the following command:

```bash

export TRAIN_FILE=/path/to/dataset/train.txt

python run_language_modeling.py \
    --model_type gpt2 \
    --model_name_or_path gpt2 \
    --do_train \
    --train_data_file $TRAIN_FILE \
    --num_train_epochs 4 \
    --output_dir model_output \
    --overwrite_output_dir \
    --save_steps 2000 \
    --per_gpu_train_batch_size 3
```

## Model in action / Example of usage: ✒

You can get the following script [here](https://github.com/huggingface/transformers/blob/master/examples/run_generation.py)

```bash
python run_generation.py \
    --model_type gpt2 \
    --model_name_or_path mrm8488/GPT-2-finetuned-CORD19 \
    --length 200
```
```txt
👵👴🦠
# Input: Old people with COVID-19 tends to suffer 
# Output: === GENERATED SEQUENCE 1 ===
Old people with COVID-19 tends to suffer more symptom onset time and death. It is well known that many people with COVID-19 have high homozygous ZIKV infection in the face of severe symptoms in both severe and severe cases.
The origin of Wuhan Fever was investigated by Prof. Shen Jiang at the outbreak of Wuhan Fever [34]. As Huanan Province is the epicenter of this outbreak, Huanan, the epicenter of epidemic Wuhan Fever, is the most potential location for the direct transmission of infection (source: Zhongzhen et al., 2020). A negative risk ratio indicates more frequent underlying signs in the people in Huanan Province with COVID-19 patients. Further analysis of reported Huanan Fever onset data in the past two years indicated that the intensity of exposure is the key risk factor for developing MERS-CoV infection in this region, especially among children and elderly. To be continued to develop infected patients would be a very important area for
```

![Model in action](https://media.giphy.com/media/TgUdO72Iwk9h7hhm7G/giphy.gif)



> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
