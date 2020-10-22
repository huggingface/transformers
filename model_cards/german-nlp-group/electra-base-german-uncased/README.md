---
language: de
license: mit
thumbnail: "https://raw.githubusercontent.com/German-NLP-Group/german-transformer-training/master/model_cards/german-electra-logo.png"
tags:
- electra
- commoncrawl
- uncased
- umlaute
- umlauts
- german
- deutsch
---

# German Electra Uncased
<img width="300px" src="https://raw.githubusercontent.com/German-NLP-Group/german-transformer-training/master/model_cards/german-electra-logo.png">
[¹]

# Model Info
This Model is suitable for Training on many downstream tasks in German (Q&A, Sentiment Analysis, etc.).

It can be used as a drop-in Replacement for **BERT** in most down-stream tasks (**ELECTRA** is even implemented as an extended **BERT** Class).

At the time of release (August 2020) this Model is the best performing publicly available German NLP Model on various German Evaluation Metrics (CONLL03-DE, GermEval18 Coarse, GermEval18 Fine). For GermEval18 Coarse results see below. More will be published soon.

# Installation
This model has the special feature that it is **uncased** but does **not strip accents**.
This possibility was added by us with [PR #6280](https://github.com/huggingface/transformers/pull/6280).
To use it you have to use Transformers version 3.1.0 or newer.

```bash
pip install transformers -U
```

# Uncase and Umlauts ('Ö', 'Ä', 'Ü')
This model is uncased. This helps especially for domains where colloquial terms with uncorrect capitalization is often used.

The special characters 'ö', 'ü', 'ä' are included through the `strip_accent=False` option, as this leads to an improved precision.

# Creators
This model was trained and open sourced in conjunction with the [**German NLP Group**](https://github.com/German-NLP-Group) in equal parts by:
- [**Philip May**](https://eniak.de) - [T-Systems on site services GmbH](https://www.t-systems-onsite.de/)
- [**Philipp Reißel**](https://www.reissel.eu) - [ambeRoad](https://amberoad.de/)

# Evaluation: GermEval18 Coarse

| Model Name                                              | F1 macro<br/>Mean | F1 macro<br/>Median | F1 macro<br/>Std |
|---|---|---|---|
| dbmdz-bert-base-german-europeana-cased                  | 0.727     | 0.729     | 0.00674     |
| dbmdz-bert-base-german-europeana-uncased                | 0.736     | 0.737     | 0.00476     |
| dbmdz/electra-base-german-europeana-cased-discriminator | 0.745     | 0.745     | 0.00498     |
| distilbert-base-german-cased                            | 0.752     | 0.752     | 0.00341     |
| bert-base-german-cased                                  | 0.762     | 0.761     | 0.00597     |
| dbmdz/bert-base-german-cased                            | 0.765     | 0.765     | 0.00523     |
| dbmdz/bert-base-german-uncased                          | 0.770     | 0.770     | 0.00572     |
| **ELECTRA-base-german-uncased (this model)**            | **0.778** | **0.778** | **0.00392** |

- (1): Hyperparameters taken from the [FARM project](https://farm.deepset.ai/) "[germEval18Coarse_config.json](https://github.com/deepset-ai/FARM/blob/master/experiments/german-bert2.0-eval/germEval18Coarse_config.json)"

![GermEval18 Coarse Model Evaluation](https://raw.githubusercontent.com/German-NLP-Group/german-transformer-training/master/model_cards/model_eval.png)

# Checkpoint evaluation
Since it it not guaranteed that the last checkpoint is the best, we evaluated the checkpoints on GermEval18. We found that the last checkpoint is indeed the best. The training was stable and did not overfit the text corpus. Below is a boxplot chart showing the different checkpoints.

![Checkpoint Evaluation on GermEval18](https://raw.githubusercontent.com/German-NLP-Group/german-transformer-training/master/model_cards/checkpoint_eval.png)

# Pre-training details

## Data
- Cleaned Common Crawl Corpus 2019-09 German: [CC_net](https://github.com/facebookresearch/cc_net) (Only head coprus and filtered for language_score > 0.98) - 62 GB
- German Wikipedia Article Pages Dump (20200701) - 5.5 GB
- German Wikipedia Talk Pages Dump (20200620) - 1.1 GB
- Subtitles - 823 MB
- News 2018 - 4.1 GB

The sentences were split with [SojaMo](https://github.com/tsproisl/SoMaJo). We took the German Wikipedia Article Pages Dump 3x to oversample. This approach was also used in a similar way in GPT-3 (Table 2.2).

More Details can be found here [Preperaing Datasets for German Electra Github](https://github.com/German-NLP-Group/german-transformer-training)

## Electra Branch no_strip_accents
Because we do not want to stip accents in our training data we made a change to Electra and used this repo [Electra no_strip_accents](https://github.com/PhilipMay/electra/tree/no_strip_accents) (branch `no_strip_accents`). Then created the tf dataset with:

```bash
python build_pretraining_dataset.py --corpus-dir <corpus_dir> --vocab-file <dir>/vocab.txt --output-dir ./tf_data --max-seq-length 512 --num-processes 8 --do-lower-case --no-strip-accents
```

## The training
The training itself can be performed with the Original Electra Repo (No special case for this needed).
We run it with the following Config:

<details>
<summary>The exact Training Config</summary>
<br/>debug False
<br/>disallow_correct False
<br/>disc_weight 50.0
<br/>do_eval False
<br/>do_lower_case True
<br/>do_train True
<br/>electra_objective True
<br/>embedding_size 768
<br/>eval_batch_size 128
<br/>gcp_project None
<br/>gen_weight 1.0
<br/>generator_hidden_size 0.33333
<br/>generator_layers 1.0
<br/>iterations_per_loop 200
<br/>keep_checkpoint_max 0
<br/>learning_rate 0.0002
<br/>lr_decay_power 1.0
<br/>mask_prob 0.15
<br/>max_predictions_per_seq 79
<br/>max_seq_length 512
<br/>model_dir gs://XXX
<br/>model_hparam_overrides {}
<br/>model_name 02_Electra_Checkpoints_32k_766k_Combined
<br/>model_size base
<br/>num_eval_steps 100
<br/>num_tpu_cores 8
<br/>num_train_steps 766000
<br/>num_warmup_steps 10000
<br/>pretrain_tfrecords gs://XXX
<br/>results_pkl gs://XXX
<br/>results_txt gs://XXX
<br/>save_checkpoints_steps 5000
<br/>temperature 1.0
<br/>tpu_job_name None
<br/>tpu_name electrav5
<br/>tpu_zone None
<br/>train_batch_size 256
<br/>uniform_generator False
<br/>untied_generator True
<br/>untied_generator_embeddings False
<br/>use_tpu True
<br/>vocab_file gs://XXX
<br/>vocab_size 32767
<br/>weight_decay_rate 0.01
 </details>

![Training Loss](https://raw.githubusercontent.com/German-NLP-Group/german-transformer-training/master/model_cards/loss.png)

Please Note: *Due to the GAN like strucutre of Electra the loss is not that meaningful*

It took about 7 Days on a preemtible TPU V3-8. In total, the Model went through approximately 10 Epochs. For an automatically recreation of a cancelled TPUs we used [tpunicorn](https://github.com/shawwn/tpunicorn). The total cost of training summed up to about 450 $ for one run. The Data-pre processing and Vocab Creation needed approximately 500-1000 CPU hours. Servers were fully provided by [T-Systems on site services GmbH](https://www.t-systems-onsite.de/), [ambeRoad](https://amberoad.de/).
Special thanks to [Stefan Schweter](https://github.com/stefan-it) for your feedback and providing parts of the text corpus.

[¹]: Source for the picture [Pinterest](https://www.pinterest.cl/pin/371828512984142193/)

# Negative Results
We tried the following approaches which we found had no positive influence:

-  **Increased Vocab Size**: Leads to more parameters and thus reduced examples/sec while no visible Performance gains were measured
-  **Decreased Batch-Size**: The original Electra was trained with a Batch Size per TPU Core of 16 whereas this Model was trained with 32 BS / TPU Core. We found out that 32 BS leads to better results when you compare metrics over computation time
