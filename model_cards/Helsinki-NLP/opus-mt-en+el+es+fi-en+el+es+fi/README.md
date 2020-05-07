# opus-2020-03-02.zip

* dataset: opus
* model: transformer
* pre-processing: normalization + SentencePiece
* a sentence initial language token is required in the form of `>>id<<` (id = valid target language ID)
* download: [opus-2020-03-02.zip](https://object.pouta.csc.fi/OPUS-MT-models/en+el+es+fi-en+el+es+fi/opus-2020-03-02.zip)
* test set translations: [opus-2020-03-02.test.txt](https://object.pouta.csc.fi/OPUS-MT-models/en+el+es+fi-en+el+es+fi/opus-2020-03-02.test.txt)
* test set scores: [opus-2020-03-02.eval.txt](https://object.pouta.csc.fi/OPUS-MT-models/en+el+es+fi-en+el+es+fi/opus-2020-03-02.eval.txt)

## Benchmarks

| testset               | BLEU  | chr-F |
|-----------------------|-------|-------|
| newsdev2015-enfi.en.fi 	| 16.0 	| 0.498 |
| newssyscomb2009.en.es 	| 29.9 	| 0.570 |
| newssyscomb2009.es.en 	| 29.7 	| 0.569 |
| news-test2008.en.es 	| 27.3 	| 0.549 |
| news-test2008.es.en 	| 27.3 	| 0.548 |
| newstest2009.en.es 	| 28.4 	| 0.564 |
| newstest2009.es.en 	| 28.4 	| 0.564 |
| newstest2010.en.es 	| 34.0 	| 0.599 |
| newstest2010.es.en 	| 34.0 	| 0.599 |
| newstest2011.en.es 	| 35.1 	| 0.600 |
| newstest2012.en.es 	| 35.4 	| 0.602 |
| newstest2013.en.es 	| 31.9 	| 0.576 |
| newstest2015-enfi.en.fi 	| 17.8 	| 0.509 |
| newstest2016-enfi.en.fi 	| 19.0 	| 0.521 |
| newstest2017-enfi.en.fi 	| 21.2 	| 0.539 |
| newstest2018-enfi.en.fi 	| 13.9 	| 0.478 |
| newstest2019-enfi.en.fi 	| 18.8 	| 0.503 |
| newstestB2016-enfi.en.fi 	| 14.9 	| 0.491 |
| newstestB2017-enfi.en.fi 	| 16.9 	| 0.503 |
| simplification.en.en 	| 63.0 	| 0.798 |
| Tatoeba.en.fi 	| 56.7 	| 0.719 |

