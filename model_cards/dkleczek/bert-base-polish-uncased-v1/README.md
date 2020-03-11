---
language: polish
thumbnail: https://raw.githubusercontent.com/kldarek/polbert/master/img/polbert.png
---

# Polbert - Polish BERT
Polish version of BERT language model is here! While this is still work in progress, I'm happy to share the first model, similar to BERT-Base and trained on a large Polish corpus. If you'd like to contribute to this project, please reach out to me!

![PolBERT image](https://raw.githubusercontent.com/kldarek/polbert/master/img/polbert.png)

## Pre-training corpora

Below is the list of corpora used along with the output of `wc` command (counting lines, words and characters). These corpora were divided into sentences with srxsegmenter (see references), concatenated and tokenized with HuggingFace BERT Tokenizer. 

| Tables        | Lines           | Words  | Characters  |
| ------------- |--------------:| -----:| -----:|
| [Polish subset of Open Subtitles](http://opus.nlpl.eu/OpenSubtitles-v2018.php)      | 236635408| 1431199601 | 7628097730 |
| [Polish subset of ParaCrawl](http://opus.nlpl.eu/ParaCrawl.php)     | 8470950      |   176670885 | 1163505275 |
| [Polish Parliamentary Corpus](http://clip.ipipan.waw.pl/PPC) | 9799859      |    121154785 | 938896963 |
| [Polish Wikipedia - Feb 2020](https://dumps.wikimedia.org/plwiki/latest/plwiki-latest-pages-articles.xml.bz2) | 8014206      |    132067986 | 1015849191 |
| Total | 262920423      |    1861093257 | 10746349159 |

## Pre-training details
* Polbert was trained with code provided in Google BERT's github repository (https://github.com/google-research/bert)
* Currently released model follows bert-base-uncased model architecture (12-layer, 768-hidden, 12-heads, 110M parameters)
* Training set-up: in total 1 million training steps: 
    * 100.000 steps - 128 sequence length, batch size 512, learning rate 1e-4 (10.000 steps warmup)
    * 800.000 steps - 128 sequence length, batch size 512, learning rate 5e-5
    * 100.000 steps - 512 sequence length, batch size 256, learning rate 2e-5
* The model was trained on a single Google Cloud TPU v3-8 

## Usage
Polbert is released via [HuggingFace Transformers library](https://huggingface.co/transformers/).

For an example use as language model, see [this notebook](https://github.com/kldarek/polbert/blob/master/LM_testing.ipynb) file. 

```python
import numpy as np
import torch
import transformers as ppb

tokenizer = ppb.BertTokenizer.from_pretrained('dkleczek/bert-base-polish-uncased-v1')
bert_model = ppb.BertForMaskedLM.from_pretrained('dkleczek/bert-base-polish-uncased-v1') 
string1 = 'Adam mickiewicz wielkim polskim [MASK] był .'
indices = tokenizer.encode(string1, add_special_tokens=True)
masked_token = np.argwhere(np.array(indices) == 3).flatten()[0] # 3 is the vocab id for [MASK] token
input_ids = torch.tensor([indices])
with torch.no_grad():
    last_hidden_states = bert_model(input_ids)[0]
more_words = np.argsort(np.asarray(last_hidden_states[0,masked_token,:]))[-4:]
print(more_words)

# Output: 
# poeta
# bohaterem
# człowiekiem
# pisarzem
```

See the next section for an example usage of Polbert in downstream tasks. 

## Evaluation
I'd love to get some help from the Polish NLP community here! If you feel like evaluating Polbert on some benchmark tasks, it would be great if you can share the results. 

So far, I've compared the performance of Polbert vs Multilingual BERT on PolEmo 2.0 sentiment classification, here are the results. These results are are produced with a linear classification layer on top of pooled output, trained for 10 epochs with learning rate 3e-5. The checkpoint with the lowest loss on validation set is evaluated on the test set. 

| PolEmo 2.0 Sentiment Classifcation | Test Accuracy | 
| ------------- |--------------:|
| Multilingual BERT | 0.78 |
| Polbert | 0.85 |

## Bias
The data used to train the model is biased. It may reflect stereotypes related to gender, ethnicity etc. Please be careful when using the model for downstream task to consider these biases and mitigate them.  

## Acknowledgements
I'd like to express my gratitude to Google [TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc) for providing the free TPU credits - thank you! Also appreciate the help from Timo Möller from [deepset](https://deepset.ai) for sharing tips and scripts based on their experience training German BERT model. Finally, thanks to Rachel Thomas, Jeremy Howard and Sylvain Gugger from [fastai](https://www.fast.ai) for their NLP and Deep Learning courses!

## Author
Darek Kłeczek - contact me on Twitter [@dk21](https://twitter.com/dk21)

## References
* https://github.com/google-research/bert
* https://github.com/narusemotoki/srx_segmenter
* SRX rules file for sentence splitting in Polish, written by Marcin Miłkowski: https://raw.githubusercontent.com/languagetool-org/languagetool/master/languagetool-core/src/main/resources/org/languagetool/resource/segment.srx
* PolEmo 2.0 Sentiment Analysis Dataset for CoNLL: https://clarin-pl.eu/dspace/handle/11321/710

