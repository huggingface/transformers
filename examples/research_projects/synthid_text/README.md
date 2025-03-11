# SynthID Text

This project showcases the use of SynthIDText for watermarking LLMs. The code shown in this repo also
demostrates the training of the detector for detecting such watermarked text. This detector can be uploaded onto
a private HF hub repo (private for security reasons) and can be initialized again through pretrained model loading also shown in this script.

See our blog post: https://huggingface.co/blog/synthid-text


## Python version

User would need python 3.9 to run this example.

## Installation and running

Once you install transformers you would need to install requirements for this project through requirements.txt provided in this folder.

```
pip install -r requirements.txt
```

## To run the detector training

```
python detector_training.py --model_name=google/gemma-7b-it
```

Check the script for more parameters are are tunable and check out paper at link
https://www.nature.com/articles/s41586-024-08025-4 for more information on these parameters.

## Caveat

Make sure to run the training of the detector and the detection on the same hardware
CPU, GPU or TPU to get consistent results (we use detecterministic randomness which is hardware dependent).
