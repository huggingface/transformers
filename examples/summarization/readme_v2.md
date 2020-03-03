Following the example script, dump your summaries into a file with one per line.
 
### CNN/Daily Mail Data
To be able to reproduce the authors' results on the CNN/Daily Mail dataset you first need to download both CNN and Daily Mail datasets [from Kyunghyun Cho's website](https://cs.nyu.edu/~kcho/DMQA/) (the links next to "Stories") in the same folder. Then uncompress the archives by running:

```bash
tar -xvf cnn_stories.tgz && tar -xvf dailymail_stories.tgz
```
this should make a directory called cnn_dm/ with files like test.source.



# Rouge Scores
### Stanford CoreNLP Setup
```
ptb_tokenize () {
    cat $1 | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $2
}

sudo apt install openjdk-8-jre-headless
sudo apt-get install ant
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip stanford-corenlp-full-2018-10-05.zip
cd stanford-corenlp-full-2018-10-05
export CLASSPATH=stanford-corenlp-3.9.2.jar:stanford-corenlp-3.9.2-models.jar
```
### Rouge Setup


Install `files2rouge` following the instructions at [here](https://github.com/pltrdy/files2rouge).
I also needed to run `sudo apt-get install libxml-parser-perl`

```
from files2rouge import files2rouge
from files2rouge import settings
files2rouge.run('/home/shleifer/transformers_fork/fs_dec28.tokenized.hypo',
                '/home/shleifer/transformers_fork/test.tokenized.target',
               saveto='fs_rouge_output.txt')
```
