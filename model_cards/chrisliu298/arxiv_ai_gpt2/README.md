---
language: "en"
tags:
- gpt2
- arxiv
- transformers
datasets:
- https://github.com/staeiou/arxiv_archive/tree/v1.0.1
---

# ArXiv AI GPT-2

## Model description

This GPT-2 (774M) model is capable of generating abstracts given paper titles. It was trained using all research paper titles and abstracts under artificial intelligence (AI), machine learning (LG), computation and language (CL), and computer vision and pattern recognition (CV) on arXiv.

## Intended uses & limitations

#### How to use

To generate paper abstracts, use the provided `generate.py` [here](https://gist.github.com/chrisliu298/ccb8144888eace069da64ad3e6472d64). This is very similar to the HuggingFace's `run_generation.py` [here](https://github.com/huggingface/transformers/tree/master/examples/text-generation). You can simply replace the text with with your own model path (line 89) and change the input string to your paper title (line 127). If you want to use your own script, make sure to prepend `<|startoftext|> ` at the front and append ` <|sep|>` at the end of the paper title.

## Training data
I selected a subset of the [arXiv Archive](https://github.com/staeiou/arxiv_archive) dataset (Geiger, 2019) as the training and evaluation data to fine-tune GPT-2. The original arXiv Archive dataset contains a full archive of metadata about papers on arxiv.org, from the start of the site in 1993 to the end of 2019. Our subset includes all the paper titles (query) and abstracts (context) under the Artificial Intelligence (cs.AI), Machine Learning (cs.LG), Computation and Language (cs.CL), and Computer Vision and Pattern Recognition (cs.CV) categories. I provide the information  of the sub-dataset and the distribution of the training and evaluation dataset as follows.


|   Splits   |   Count    | Percentage (%) | BPE Token Count |
| :--------: | :--------: | :------------: | :-------------: |
|   Train    |   90,000   |     90.11      |   20,834,012    |
| Validation |    4,940   |      4.95      |    1,195,056    |
|    Test    |    4,940   |      4.95      |    1,218,754    |
| **Total**  | **99,880** |    **100**     | **23,247,822**  |

The original dataset is in the format of a tab-separated value, so we wrote a simple preprocessing script to convert it into a text file format, which is the input file type (a document) of the GPT-2 model. An example of a paper’s title and its abstract is shown below.

```text
<|startoftext|> Some paper title <|sep|> Some paper abstract <|endoftext|>
```

Because there are a lot of cross-domain papers in the dataset, I deduplicate the dataset using the arXiv ID, which is unique for every paper. I sort the paper by submission date, by doing so, one can examine GPT-2’s ability to use learned terminologies when it is prompted with paper titles from the “future.”


## Training procedure

I used block size = 512, batch size = 1, gradidnet accumulation = 1, learning rate = 1e-5, epochs = 5, and everything else follows the default model configuration.

## Eval results

The resulting GPT-2 large model's perplexity score on the test set is **14.9413**.

## Reference

```bibtex
@dataset{r_stuart_geiger_2019_2533436,
    author= {R. Stuart Geiger},
    title={{ArXiV Archive: A tidy and complete archive of metadata for papers on arxiv.org, 1993-2019}},
    month=jan,
    year= 2019,
    publisher={Zenodo},
    version= {v1.0.1},
    doi={10.5281/zenodo.2533436},
    url={https://doi.org/10.5281/zenodo.2533436}
}
```

