---
language: german
thumbnail: kfold.png
---

# German BERT for literary texts

This German BERT is based on `bert-base-german-dbmdz-cased`, and has been adapted to the domain of literary texts by fine-tuning the language modeling task on the [Corpus of German-Language Fiction](https://figshare.com/articles/Corpus_of_German-Language_Fiction_txt_/4524680/1). Afterwards the model was fine-tuned for named entity recognition on the [DROC](https://gitlab2.informatik.uni-wuerzburg.de/kallimachos/DROC-Release) corpus, so you can use it to recognize protagonists in German novels.


# Stats

## Language modeling

The [Corpus of German-Language Fiction](https://figshare.com/articles/Corpus_of_German-Language_Fiction_txt_/4524680/1) consists of 3,194 documents with 203,516,988 tokens or 1,520,855 types. The publication year of the texts ranges from the 18th to the 20th century:

![years](prosa-jahre.png)


### Results

After one epoch:

| Model            | Perplexity |
| ---------------- | ---------- |
| Vanilla BERT     | 6.82       |
| Fine-tuned BERT  | 4.98       |


## Named entity recognition

The provided model was also fine-tuned for two epochs on 10,799 sentences for training, validated on 547 and tested on 1,845 with three labels: `B-PER`, `I-PER` and `O`.


## Results

| Dataset | Precision | Recall | F1   |
| ------- | --------- | ------ | ---- |
| Dev     | 96.4      | 87.3   | 91.6 |
| Test    | 92.8      | 94.9   | 93.8 |

The model has also been evaluated using 10-fold cross validation and compared with a classic Conditional Random Field baseline described in [Jannidis et al.](https://opus.bibliothek.uni-wuerzburg.de/opus4-wuerzburg/frontdoor/deliver/index/docId/14333/file/Jannidis_Figurenerkennung_Roman.pdf) (2015):

![kfold](kfold.png)


# References

Markus Krug, Lukas Weimer, Isabella Reger, Luisa Macharowsky, Stephan Feldhaus, Frank Puppe, Fotis Jannidis, [Description of a Corpus of Character References in German Novels](http://webdoc.sub.gwdg.de/pub/mon/dariah-de/dwp-2018-27.pdf), 2018.

Fotis Jannidis, Isabella Reger, Lukas Weimer, Markus Krug, Martin Toepfer, Frank Puppe, [Automatische Erkennung von Figuren in deutschsprachigen Romanen](https://opus.bibliothek.uni-wuerzburg.de/opus4-wuerzburg/frontdoor/deliver/index/docId/14333/file/Jannidis_Figurenerkennung_Roman.pdf), 2015.
