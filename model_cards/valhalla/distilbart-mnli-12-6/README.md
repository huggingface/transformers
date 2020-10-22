---
datasets:
- mnli
tags:
- distilbart
- distilbart-mnli
pipeline_tag: zero-shot-classification
---

# DistilBart-MNLI

distilbart-mnli is the distilled version of bart-large-mnli created using the **No Teacher Distillation** technique proposed for BART summarisation by Huggingface, [here](https://github.com/huggingface/transformers/tree/master/examples/seq2seq#distilbart).

We just copy alternating layers from `bart-large-mnli` and finetune more on the same data. 


|                                                                                      | matched acc | mismatched acc |
| ------------------------------------------------------------------------------------ | ----------- | -------------- |
| [bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) (baseline, 12-12) | 89.9        | 90.01          |
| [distilbart-mnli-12-1](https://huggingface.co/valhalla/distilbart-mnli-12-1)         | 87.08       | 87.5           |
| [distilbart-mnli-12-3](https://huggingface.co/valhalla/distilbart-mnli-12-3)         | 88.1        | 88.19          |
| [distilbart-mnli-12-6](https://huggingface.co/valhalla/distilbart-mnli-12-6)         | 89.19       | 89.01          |
| [distilbart-mnli-12-9](https://huggingface.co/valhalla/distilbart-mnli-12-9)         | 89.56       | 89.52          |


This is a very simple and effective technique, as we can see the performance drop is very little.

Detailed performace trade-offs will be posted in this [sheet](https://docs.google.com/spreadsheets/d/1dQeUvAKpScLuhDV1afaPJRRAE55s2LpIzDVA5xfqxvk/edit?usp=sharing).


## Fine-tuning
If you want to train these models yourself, clone the [distillbart-mnli repo](https://github.com/patil-suraj/distillbart-mnli) and follow the steps below

Clone and install transformers from source
```bash
git clone https://github.com/huggingface/transformers.git
pip install -qqq -U ./transformers
```

Download MNLI data
```bash
python transformers/utils/download_glue_data.py --data_dir glue_data --tasks MNLI
```

Create student model
```bash
python create_student.py \
  --teacher_model_name_or_path facebook/bart-large-mnli \
  --student_encoder_layers 12 \
  --student_decoder_layers 6 \
  --save_path student-bart-mnli-12-6 \
```

Start fine-tuning
```bash
python run_glue.py args.json
```

You can find the logs of these trained models in this [wandb project](https://wandb.ai/psuraj/distilbart-mnli).