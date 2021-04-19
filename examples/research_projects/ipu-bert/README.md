# IPU BERT

Implementation of BERT for pretraining model in PyTorch for the [Graphcore](https://www.graphcore.ai/) IPU, leveraging HuggingFace Transformers library.

## Environment Setup

First, install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh` scripts for poplar and popART.

Then, create a virtual environment, install the required packages and build the custom ops.

```console
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt
make
```

## Run the application

Setup your environment as explained above and run the example with the configuration of your choice.

```console
python bert.py --config --config demo_tiny_128
```

## Configurations

To see the available configurations see the `configs.yml` file.
To see the available options available to use in the command line interface use `--help` argument.

```console
python bert.py --help
```

## Caching executables

When running the application, it is possible to save/load executables to/from a cache store. This allows for reusing a saved executable instead of re-compiling the model when re-running identical model configurations. To enable saving/loading from the cache store, use `--executable-cache-dir <relative/path/to/cache/store>` when running the application.


## Run the tests

Setup your environment and generate the sample dataset as explained above and run `pytest` from the root folder.


## Generate sample_text dataset (Optional)

The sample text provided, enable training on a very small dataset for small scale testing.
For convenience it is already provided in the `/data` folder in txt and tfrecord format.
In order to re-generate the sample dataset, run the following script.

```console
python third_party/create_pretraining_data.py --input-file data/sample_text.txt --output-file data/sample_text.tfrecord --sequence-length 128 --mask-tokens 20 --duplication-factor 4 --do-lower-case --model bert-base-uncased
```

## Generate pretraining dataset (Optional)

The dataset used for pretraining is WIKI-103. It can be generated from a RAW dump of wikipedia following a four steps process.

### Download

Use the `wikipedia_download.sh` script to download the latest wikipedia dump, about 20GB in size.

```console
./data/wikipedia_download.sh <chosen-path-for-dump-file>
```

### Extraction

Install extractor package with `pip install wikiextractor==0.1`.
Use the `wikipedia_extract.sh` script to extract the data dump.

```console
./data/wikipedia_extract.sh <chosen-path-for-dump-file>/wikidump.xml <chosen-folder-for-extracted-files>
```

The result should be a folder containing directories named `AA`, `AB`...

### Pre-processing

Install nltk package with `pip install nltk`.
Use the `wikipedia_preprocess.py` script to preprocess the extracted files.

```console
./data/wikipedia_preprocess.py --input-file-path <chosen-folder-for-extracted-files> --output-file-path <chosen-folder-for-preprocessed-files>
```

### Tokenization

The script `create_pretraining_data.py` can accept a glob of input files to tokenise. However, attempting to process them all at once may result in the process being killed by the OS for consuming too much memory. It is therefore preferable to convert the files in groups. This will be handled by the `./data/wikipedia_tokenize.py` script. At the same time, it is worth bearing in mind that `create_pretraining_data.py` shuffles the training instances across the loaded group of files, so a larger group would result in better shuffling of the samples seen by BERT during pre-training.

sequence length 128
```console
./data/wikipedia_tokenize.py <chosen-folder-for-preprocessed-files> <chosen-folder-for-dataset-files> --sequence-length 128 --mask-tokens 20
```

sequence length 384
```console
./data/wikipedia_tokenize.py <chosen-folder-for-preprocessed-files> <chosen-folder-for-dataset-files> --sequence-length 384 --mask-tokens 56
```

### Indexing

In order to use the multithreaded dataloader, tfrecord index files need to be generated.
First install the `tfrecord` Python package into your Python environment:

```console
pip install tfrecord
```

Then go to the directory containing the preprocessed wikipedia files and run:

```console
for f in *.tfrecord; do python3 -m tfrecord.tools.tfrecord2idx $f `basename $f .tfrecord`.index; done
```

## Licensing

The code presented here is provided under Apache 2.0 Lience, see LICENSE file

This directory includes derived work from the following:

BERT, https://github.com/google-research/bert

Copyright 2018 The Google AI Language Team Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
