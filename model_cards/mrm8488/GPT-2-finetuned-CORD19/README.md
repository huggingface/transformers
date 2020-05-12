---
language: english
thumbnail:
---

# GPT-2 + CORD19 dataset : 🦠 ✍ ⚕

**GPT-2** fine-tuned on **biorxiv_medrxiv**, **comm_use_subset** and **custom_license** files from [CORD-19](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) dataset.


## Datasets details

| Dataset                | # Files |
| ---------------------- | ----- |
| biorxiv_medrxiv        | 885  |
| comm_use_subset         | 9K   |
| custom_license         | 20.6K   |

## Model training

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
    --save_steps 10000 \
    --per_gpu_train_batch_size 3
```

<img alt="training loss" src="https://svgshare.com/i/JTf.svg' title='GTP-2-finetuned-CORDS19-loss" width="600" height="300" />

## Model in action / Example of usage ✒

You can get the following script [here](https://github.com/huggingface/transformers/blob/master/examples/text-generation/run_generation.py)

```bash
python run_generation.py \
    --model_type gpt2 \
    --model_name_or_path mrm8488/GPT-2-finetuned-CORD19 \
    --length 200
```
```txt
# Input: the effects of COVID-19 on the lungs
# Output: === GENERATED SEQUENCE 1 ===
the effects of COVID-19 on the lungs are currently debated (86). The role of this virus in the pathogenesis of pneumonia and lung cancer is still debated. MERS-CoV is also known to cause acute respiratory distress syndrome (87) and is associated with increased expression of pulmonary fibrosis markers (88). Thus, early airway inflammation may play an important role in the pathogenesis of coronavirus pneumonia and may contribute to the severe disease and/or mortality observed in coronavirus patients.
Pneumonia is an acute, often fatal disease characterized by severe edema, leakage of oxygen and bronchiolar inflammation. Viruses include coronaviruses, and the role of oxygen depletion is complicated by lung injury and fibrosis in the lung, in addition to susceptibility to other lung diseases. The progression of the disease may be variable, depending on the lung injury, pathologic role, prognosis, and the immune status of the patient. Inflammatory responses to respiratory viruses cause various pathologies of the respiratory
```


> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
